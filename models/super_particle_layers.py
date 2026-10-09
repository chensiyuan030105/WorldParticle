import torch
import torch.nn as nn
from .layers.attention import MultiHeadAttention, FeedForwardGeLU, FeedForwardSwiGLU
from .encodings.rms_norm import RMSNorm
from .continuous_conv import ContinuousConv, ParticleRadiusResearch, FixedRadiusSearch
from .invariant import RadialMessagePassing
from open3d.ml.torch.ops import reduce_subarrays_sum
import numpy as np
import math
from TrajDataset import uses_obstacle_features


class CubicSplineKernel:
    """
    Cubic Spline Kernel implementation based on SPlisHSPlasH
    W(r,h) = (8/(pi*h^3)) * {
        6*q^3 - 6*q^2 + 1     if 0 <= q <= 0.5
        2*(1-q)^3             if 0.5 < q <= 1
        0                     if q > 1
    }
    where q = ||r|| / h
    """
    
    def __init__(self, particle_radius=0.025):
        h = 2.0 * particle_radius  # Kernel support radius
        self.h = h
        self.h2 = h * h
        self.h3 = h * h * h
        
        # Normalization constants
        self.k = 8.0 / (math.pi * self.h3)
        self.l = 48.0 / (math.pi * self.h3)


class SPHVelocityInterpolator:
    """
    SPH Velocity Interpolator using the formulation from SPlisHSPlasH
    
    The velocity at query point x_i is interpolated as:
    v_i = sum_j (m_j / rho_j) * v_j * W(x_i - x_j, h)
    
    For simplicity, we assume uniform mass and density (m_j/rho_j = constant)
    """
    
    def __init__(self, kernel_radius):
        """
        Initialize the interpolator
        
        Args:
            particle_positions: numpy array of shape (N, 3) - particle positions
            particle_velocities: numpy array of shape (N, 3) - particle velocities  
            kernel_radius: float - SPH kernel support radius
        """
        # TODO: check if this is correct
        self.kernel_radius = kernel_radius
        self.kernel = CubicSplineKernel(kernel_radius)
        self.fixed_radius_search = FixedRadiusSearch(
            metric='L2',
            ignore_query_point=False,
            return_distances=True)
        
    def interpolate_velocity(self, query_points, input_positions, input_velocities, input_pos_hash_table=None):
        """
        Args:
            query_points: (Q, 3) torch.float
            input_positions: (N, 3) torch.float
            input_velocities: (N, 3) torch.float
        Returns:
            (Q, 3) torch.float interpolated velocities
        """
        # Neighbor search (assumes torch tensors returned)
        if input_pos_hash_table is not None:
            nns = self.fixed_radius_search(
                input_positions,
                queries=query_points,
                radius=self.kernel_radius,
                hash_table=input_pos_hash_table)
        else:
            nns, input_pos_hash_table = self.fixed_radius_search(
                input_positions,
                queries=query_points,
                radius=self.kernel_radius,
                return_hash_table=True
            )

        # Flattened neighbor info
        d = nns.neighbors_distance.sqrt()        # (M,) distances (because input were squared)
        nbr_idx = nns.neighbors_index            # (M,) long
        row_splits = nns.neighbors_row_splits    # (Q+1,) long

        device = input_velocities.device
        dtype = input_velocities.dtype

        # Kernel constants
        h = torch.as_tensor(self.kernel.h, device=device, dtype=dtype)
        k = torch.as_tensor(self.kernel.k, device=device, dtype=dtype)

        # Vectorized cubic spline weights W(r,h)
        q = d.to(dtype) / h  # (M,)
        w = torch.zeros_like(q)

        m1 = q <= 0.5
        m2 = (q > 0.5) & (q <= 1.0)

        q2 = q * q
        q3 = q2 * q
        w = w.masked_scatter(m1, k * (6.0 * q3[m1] - 6.0 * q2[m1] + 1.0))
        w = w.masked_scatter(m2, k * 2.0 * (1.0 - q[m2])**3)
        # values where q > 1 remain 0

        # Neighbor velocity contributions (M, 3)
        contrib = input_velocities[nbr_idx] * w.unsqueeze(-1)

        # Build query index per neighbor via row_splits
        lengths = (row_splits[1:] - row_splits[:-1]).to(torch.long)  # (Q,)
        Q = lengths.numel()
        q_idx_per_nbr = torch.repeat_interleave(
            torch.arange(Q, device=device, dtype=torch.long),
            lengths
        )  # (M,)

        # Accumulate per-query sums: out[q] += contrib[m] where q = q_idx_per_nbr[m]
        out = torch.zeros((Q, 3), device=device, dtype=dtype)
        out.scatter_add_(0, q_idx_per_nbr.view(-1, 1).expand(-1, 3), contrib)

        # Optional: Shepard-style normalization for smoother interpolation
        # sum_w[q] = sum of weights for query q
        sum_w = torch.zeros(Q, device=device, dtype=dtype)
        sum_w.scatter_add_(0, q_idx_per_nbr, w)
        mask = sum_w > 0
        out[mask] = out[mask] / sum_w[mask].unsqueeze(-1)

        return out, input_pos_hash_table
    
    
class LocalFeatureExtractor(torch.nn.Module):
    def __init__(
            self,
            # ============= Continuous Convolution Parameters =============
            kernel_size=[4, 4, 4],
            cconv_embedding_dim=32,
            data=None,
            use_topology_conv=False,
            topology_search_radius=None,
            radius_scale=1.5,
            coordinate_mapping='ball_to_cube_volume_preserving',
            interpolation='linear',
            use_window=True,
            particle_radius=0.025,
            other_feats_channels=0,
            obstacle_feats_channels=3,
            topology_blend_alpha=None,
            normalize_other_feats=True,
    ):
        super().__init__()

        # ============= Continuous Convolution Configuration =============
        self.data = data
        self.kernel_size = kernel_size
        self.cconv_embedding_dim = cconv_embedding_dim
        self.radius_scale = radius_scale
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        self.particle_radius = particle_radius
        self.other_feats_channels = other_feats_channels
        self.obstacle_feats_channels = obstacle_feats_channels
        self.is_local_molecular = self.data == "local_fluid"
        self.topology_blend_alpha = topology_blend_alpha
        self.filter_extent = np.float32(self.radius_scale * 6 * self.particle_radius)
        self.use_topology_conv = bool(use_topology_conv)
        self.topology_search_radius = None if topology_search_radius is None else float(topology_search_radius)
        self.topology_alpha_param = None
        self.topology_alpha_fixed = None
        self.topology_fusion_mode = "disabled"
        if self.use_topology_conv:
            if topology_blend_alpha is None:
                self.topology_fusion_mode = "concat"
            else:
                alpha_fixed = float(topology_blend_alpha)
                if alpha_fixed < 0.0 or alpha_fixed > 1.0:
                    raise ValueError(
                        f"topology_blend_alpha must be in [0, 1], got {alpha_fixed}"
                    )
                if alpha_fixed not in (0.0, 1.0):
                    raise ValueError(
                        "topology_blend_alpha now only supports null/0/1: "
                        "null=concat(auto), 0=topology-only, 1=molecular-only."
                    )
                self.topology_alpha_fixed = alpha_fixed
                if alpha_fixed == 0.0:
                    self.topology_fusion_mode = "topology_only"
                else:
                    self.topology_fusion_mode = "molecular_only"
        if self.topology_search_radius is None:
            self.topology_filter_extent = self.filter_extent
        else:
            self.topology_filter_extent = np.float32(self.radius_scale * 6 * self.topology_search_radius)
        
        # ============= Initialize Neighbor Search =============
        # Define window function for continuous convolution
        def window_poly6(r_sqr):
            return torch.clamp((1 - r_sqr)**3, 0, 1)

        window_fn = window_poly6 if self.use_window else None

        # Initialize particle radius search for neighbor finding
        self.particle_radius_search = ParticleRadiusResearch(
            radius_search_metric='L2',
            radius_search_ignore_query_points=True,
            window_function=window_fn
        )

        # ============= Initialize Continuous Convolution Layers =============
        # Helper function to create continuous convolution layers
        def Conv(activation=None, **kwargs):
            conv = ContinuousConv(
                kernel_size=self.kernel_size,
                activation=activation,
                align_corners=True,
                interpolation=self.interpolation,
                coordinate_mapping=self.coordinate_mapping,
                normalize=False,
                **kwargs
            )
            return conv

        # Molecular-molecular convolution: particle-particle interactions
        molecular_conv_out_channels = cconv_embedding_dim
        topology_conv_out_channels = cconv_embedding_dim
        if self.topology_fusion_mode == "concat":
            if cconv_embedding_dim % 2 != 0:
                raise ValueError(
                    f"cconv_embedding_dim must be even for topology concat mode, got {cconv_embedding_dim}"
                )
            molecular_conv_out_channels = cconv_embedding_dim // 2
            topology_conv_out_channels = cconv_embedding_dim // 2

        # P011 D1(a).  Was `Conv(...)`, i.e. open3d's ContinuousConv, which
        # indexes a fixed-orientation 4x4x4 kernel by the neighbour's relative
        # offset *vector* -- rotate the molecule and a neighbour lands in a
        # different cell.  `RadialMessagePassing` keeps the same neighbour
        # lists, the same window function and the same forward signature, and
        # reads only ||p_i - p_j||.  See models/invariant.py.
        #
        # The RBF spans [0, search radius], not [0, filter_extent]:
        # ParticleRadiusResearch searches at `0.5 * extents`
        # (continuous_conv.py:313), so no pair is ever farther apart than half
        # the filter extent and centres beyond that would never be reached.
        # (The plan's section 4 said `[0, filter_extent]`; corrected in v0.2.)
        self.conv0_molecular = RadialMessagePassing(
            in_channels=4 + other_feats_channels,
            filters=molecular_conv_out_channels,
            cutoff=float(self.filter_extent) * 0.5,
        )
        if self.use_topology_conv:
            self.conv0_topology = RadialMessagePassing(
                in_channels=4 + other_feats_channels,
                filters=topology_conv_out_channels,
                cutoff=float(self.topology_filter_extent) * 0.5,
            )

        # Molecular-obstacle convolution: particle-boundary interactions.
        # Deliberately still a ContinuousConv, i.e. still not rotation
        # invariant.  `uses_obstacle_features("molecular")` is False, so for
        # every Kineidos arm this branch is not built at all, and converting a
        # code path this plan cannot run or test would be a change nobody has
        # measured.  Anything that does turn obstacle data on in `wp-v3` has to
        # redo the symmetry probe in section 4 first.  Same for the unused
        # `SuperParticleLocalFeatureExtractor` below.
        if uses_obstacle_features(self.data):
            self.conv0_obstacle = Conv(
                in_channels=obstacle_feats_channels,
                filters=cconv_embedding_dim,
                activation=None
            )

        # Dense layer for direct feature transformation
        dense0_in_features = 7 + other_feats_channels if self.is_local_molecular else 4 + other_feats_channels
        dense0_out_features = 3 * cconv_embedding_dim if self.is_local_molecular else cconv_embedding_dim
        self.dense0_molecular = nn.Linear(
            in_features=dense0_in_features,
            out_features=dense0_out_features
        )
        nn.init.xavier_uniform_(self.dense0_molecular.weight)
        nn.init.zeros_(self.dense0_molecular.bias)

        # P011 D2 / STAR-MD A12.  The particle feature vector is a concat of
        # branches on wildly different scales, and both the conv and dense0
        # consume it: P009 10.7 measured `ones` = 1.0 against velocities of
        # 1.8-2.7e-4, i.e. 5000:1, and the per-atom part of dense0's output was
        # 1.57e-07 of its magnitude -- the two constant channels decided the
        # layer by themselves.  STAR-MD hit the same bug at 220:1 and fixed it
        # by normalising each branch *before* the concat; a norm after it
        # "cannot undo the imbalance".
        #
        # Only `other_feats` gets a LayerNorm.  Not the other two, for reasons
        # rather than by omission:
        #   * `ones` is upstream's deliberate bias channel for the convolution
        #     (the comment at the concat says so).  It is one channel with zero
        #     variance, so a LayerNorm over it returns exactly beta and the
        #     bias is destroyed.  It is also the reference scale -- rms 1.0 is
        #     what the other branches are being normalised *to*.
        #   * the velocity slot is fed zeros under this plan's contract (D2),
        #     so there is nothing to normalise; it is kept rather than removed
        #     so that in_channels, and therefore the state_dict, is unchanged.
        self.normalize_other_feats = bool(normalize_other_feats)
        if self.normalize_other_feats and other_feats_channels > 0:
            self.other_feats_norm = nn.LayerNorm(other_feats_channels)
        else:
            self.other_feats_norm = None
        
        self.register_buffer(
            "filter_extent_tensor",
            torch.tensor(self.filter_extent, dtype=torch.float32)
        )
        self.register_buffer(
            "topology_filter_extent_tensor",
            torch.tensor(self.topology_filter_extent, dtype=torch.float32)
        )
    
    def _prepare_other_feats(self, other_feats):
        """Shape-fix and per-branch normalise the `other_feats` branch.

        P011 D2.  See the constructor for why this branch and not the other
        two.  The LayerNorm runs over channels, per particle, so it equalises
        the branch against `ones` without erasing which atom is which.
        """
        if other_feats.dim() == 1:
            other_feats = other_feats.unsqueeze(-1)
        if other_feats.shape[-1] != self.other_feats_channels:
            raise ValueError(
                f"other_feats has {other_feats.shape[-1]} channels but this "
                f"LocalFeatureExtractor was built for "
                f"{self.other_feats_channels}. The channel count is a "
                f"constructor argument, so a mismatch means the bridge and the "
                f"model disagree about the feature contract (plan section 5.1)."
            )
        if self.other_feats_norm is not None:
            other_feats = self.other_feats_norm(other_feats)
        return other_feats

    def forward(self, inp_pos, query_pos, inp_vel, other_feats, 
                box=None, 
                box_feats=None, 
                query_vel=None, 
                fixed_radius_search_hash_table=None,
                batch_info=None,
                topology_neighbors_index=None,
                topology_neighbors_row_splits=None):
        filter_extent = self.filter_extent_tensor
        topology_filter_extent = self.topology_filter_extent_tensor

        if self.is_local_molecular:
            molecular_feats = [torch.ones_like(inp_pos[:, 0:1]), inp_vel, inp_pos]
            if other_feats is not None:
                other_feats = self._prepare_other_feats(other_feats)
                molecular_feats.append(other_feats)
            molecular_feats = torch.cat(molecular_feats, axis=-1)
            dense0_molecular_out = self.dense0_molecular(molecular_feats)
            particle_features = dense0_molecular_out
            return particle_features, None, None

        # ============= Step 1: Neighbor Search =============
        # Find molecular-molecular neighbors (particle-particle interactions)
        if batch_info is not None:
            batch_size = batch_info['batch_size']
            batch_splits = batch_info['batch_splits'].to(
                device=query_pos.device, dtype=torch.int64
            ).contiguous()
            if batch_splits.numel() != batch_size + 1:
                raise ValueError(
                    f"batch_splits must contain {batch_size + 1} entries, "
                    f"got {batch_splits.numel()}"
                )

            molecular_molecular_nns = self.particle_radius_search(
                inp_positions=inp_pos,
                out_positions=query_pos,
                extents=filter_extent,
                fixed_radius_search_hash_table=fixed_radius_search_hash_table,
                inp_row_splits=batch_splits,
                out_row_splits=batch_splits,
            )

            if uses_obstacle_features(self.data):
                box_splits = batch_info['box_splits'].to(
                    device=box.device, dtype=torch.int64
                ).contiguous()
                if box_splits.numel() != batch_size + 1:
                    raise ValueError(
                        f"box_splits must contain {batch_size + 1} entries, "
                        f"got {box_splits.numel()}"
                    )

                molecular_obstacle_nns = self.particle_radius_search(
                    inp_positions=box,
                    out_positions=query_pos,
                    extents=filter_extent,
                    fixed_radius_search_hash_table=fixed_radius_search_hash_table,
                    inp_row_splits=box_splits,
                    out_row_splits=batch_splits,
                )
                
        else:
            molecular_molecular_nns = self.particle_radius_search(
                inp_positions=inp_pos,
                out_positions=query_pos,
                extents=filter_extent,
                fixed_radius_search_hash_table=fixed_radius_search_hash_table)
            if uses_obstacle_features(self.data):
                # Find molecular-obstacle neighbors (particle-boundary interactions)
                molecular_obstacle_nns = self.particle_radius_search(
                    inp_positions=box,
                    out_positions=query_pos,
                    extents=filter_extent,
                    fixed_radius_search_hash_table=fixed_radius_search_hash_table)

        # ============= Step 2: Feature Extraction via Continuous Convolutions =============
        # Prepare molecular particle features: [1, velocities, other_features]
        molecular_feats = [torch.ones_like(inp_pos[:, 0:1]), inp_vel]  # Add constant feature for bias
        if other_feats is not None:
            other_feats = self._prepare_other_feats(other_feats)
            molecular_feats.append(other_feats)
        molecular_feats = torch.cat(molecular_feats, axis=-1)
        # Reported by kineidos/probe_wp_content.py (plan section 4, item 4):
        # the per-branch rms values that D2's imbalance is about.
        self._branch_rms = {
            "ones": 1.0,
            "vel": float(inp_vel.detach().float().pow(2).mean().sqrt()),
            "other_feats": (
                float(other_feats.detach().float().pow(2).mean().sqrt())
                if other_feats is not None else float("nan")
            ),
        }

        # Extract molecular branch first when required by fusion mode.
        conv0_molecular_out = None
        if self.topology_fusion_mode in {"disabled", "concat", "molecular_only"}:
            conv0_molecular_out = self.conv0_molecular(
                molecular_feats, inp_pos, query_pos, filter_extent,
                inp_importance=molecular_molecular_nns['inp_importance'],
                neighbors_index=molecular_molecular_nns['neighbors_index'],
                neighbors_row_splits=molecular_molecular_nns['neighbors_row_splits'],
                neighbors_importance=molecular_molecular_nns['neighbors_importance'])

        topology_neighbors_available = (
            self.use_topology_conv
            and topology_neighbors_index is not None
            and topology_neighbors_row_splits is not None
        )

        conv0_topology_out = None
        # Topology neighborhood branch (enabled when topology neighbors are provided).
        if topology_neighbors_available and self.topology_fusion_mode in {"concat", "topology_only"}:
            topology_neighbors_index = topology_neighbors_index.to(
                device=inp_pos.device, dtype=torch.int32
            ).contiguous()
            topology_neighbors_row_splits = topology_neighbors_row_splits.to(
                device=inp_pos.device, dtype=torch.int64
            ).contiguous()
            expected_num_row_splits = query_pos.shape[0] + 1
            if topology_neighbors_row_splits.numel() != expected_num_row_splits:
                raise ValueError(
                    f"topology_neighbors_row_splits size mismatch: got "
                    f"{topology_neighbors_row_splits.numel()}, expected {expected_num_row_splits}"
                )

            topology_nns = self.particle_radius_search(
                inp_positions=inp_pos,
                out_positions=query_pos,
                extents=topology_filter_extent,
                user_neighbors_index=topology_neighbors_index,
                user_neighbors_row_splits=topology_neighbors_row_splits,
                user_neighbors_importance=torch.ones_like(
                    topology_neighbors_index, dtype=torch.float32, device=inp_pos.device
                )
            )
            conv0_topology_out = self.conv0_topology(
                molecular_feats, inp_pos, query_pos, topology_filter_extent,
                inp_importance=topology_nns['inp_importance'],
                neighbors_index=topology_nns['neighbors_index'],
                neighbors_row_splits=topology_nns['neighbors_row_splits'],
                neighbors_importance=topology_nns['neighbors_importance'])

        # Topology + molecular fusion policy:
        # - disabled: molecular-only
        # - concat: [molecular_half, topology_half] -> cconv_embedding_dim
        # - topology_only: use topology branch only (alpha=0)
        # - molecular_only: use molecular branch only (alpha=1)
        if self.topology_fusion_mode == "concat":
            if conv0_molecular_out is None:
                raise ValueError("concat mode expects molecular branch output, got None.")
            if conv0_topology_out is None:
                conv0_topology_out = torch.zeros_like(conv0_molecular_out)
            conv0_molecular_out = torch.cat([conv0_molecular_out, conv0_topology_out], dim=-1)
        elif self.topology_fusion_mode == "topology_only":
            if conv0_topology_out is None:
                raise ValueError(
                    "topology_blend_alpha=0 requires topology neighbors, "
                    "but topology neighbors are missing in the current batch."
                )
            conv0_molecular_out = conv0_topology_out
        elif self.topology_fusion_mode == "molecular_only":
            if conv0_molecular_out is None:
                raise ValueError("topology_blend_alpha=1 expects molecular branch output, got None.")
        elif self.topology_fusion_mode == "disabled":
            if conv0_molecular_out is None:
                raise ValueError("topology disabled mode expects molecular branch output, got None.")
        else:
            raise ValueError(f"Unsupported topology_fusion_mode={self.topology_fusion_mode}")

        # Dense layer: Direct transformation of particle features
        if query_vel is not None:
            query_molecular_feats = [torch.ones_like(query_pos[:, 0:1]), query_vel]
            dense0_molecular_out = self.dense0_molecular(torch.cat(query_molecular_feats, axis=-1))
        else:
            dense0_molecular_out = self.dense0_molecular(molecular_feats)

        if uses_obstacle_features(self.data):
            if box_feats is None:
                raise ValueError(
                    "box_feats is required for obstacle-feature data."
                )
            if box_feats.dim() == 1:
                box_feats = box_feats.unsqueeze(-1)
            if box_feats.shape[-1] != self.obstacle_feats_channels:
                raise ValueError(
                    f"box_feats channel mismatch: got {box_feats.shape[-1]}, "
                    f"expected {self.obstacle_feats_channels}. "
                    "Set `obstacle_feats_channels` to match your data."
                )
        # Molecular-obstacle convolution: Extract features from boundary interactions
            conv0_obstacle_out = self.conv0_obstacle(
                box_feats, box, query_pos, filter_extent,
                inp_importance=molecular_obstacle_nns['inp_importance'],
                neighbors_index=molecular_obstacle_nns['neighbors_index'],
                neighbors_row_splits=molecular_obstacle_nns['neighbors_row_splits'],
                neighbors_importance=molecular_obstacle_nns['neighbors_importance'])

            # Combine all extracted features
            particle_features = torch.cat([
                conv0_obstacle_out,  # Boundary interaction features
                conv0_molecular_out,     # Particle-particle features
                dense0_molecular_out,    # Direct particle features
            ], axis=-1)  # Shape: [N, combined_feat_dim]

        else:
            particle_features = torch.cat([
                conv0_molecular_out,     # Particle-particle features
                dense0_molecular_out,    # Direct particle features
            ], axis=-1)
        
        num_molecular_neighbors = reduce_subarrays_sum(
            torch.ones_like(molecular_molecular_nns['neighbors_index'], dtype=torch.float32),
            molecular_molecular_nns['neighbors_row_splits']
        )
        return particle_features, num_molecular_neighbors, molecular_molecular_nns
        

class SuperParticleLocalFeatureExtractor(torch.nn.Module):
    def __init__(
            self,
            super_particle_compression_ratio=10.0,
            # ============= Continuous Convolution Parameters =============
            kernel_size=[4, 4, 4],
            cconv_embedding_dim=32,
            radius_scale=1.5,
            coordinate_mapping='ball_to_cube_volume_preserving',
            interpolation='linear',
            use_window=True,
            particle_radius=0.025,
            other_feats_channels=0,
    ):
        super().__init__()

        # ============= Continuous Convolution Configuration =============
        self.kernel_size = kernel_size
        self.cconv_embedding_dim = cconv_embedding_dim
        self.radius_scale = radius_scale
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        self.particle_radius = particle_radius
        self.other_feats_channels = other_feats_channels
        self.filter_extent = np.float32(self.radius_scale * 6 * self.particle_radius * math.pow(super_particle_compression_ratio, 1.0/3.0))
        
        # ============= Initialize Neighbor Search =============
        # Define window function for continuous convolution
        def window_poly6(r_sqr):
            return torch.clamp((1 - r_sqr)**3, 0, 1)

        window_fn = window_poly6 if self.use_window else None

        # Initialize particle radius search for neighbor finding
        self.particle_radius_search = ParticleRadiusResearch(
            radius_search_metric='L2',
            radius_search_ignore_query_points=True,
            window_function=window_fn
        )

        # ============= Initialize Continuous Convolution Layers =============
        # Helper function to create continuous convolution layers
        def Conv(activation=None, **kwargs):
            conv = ContinuousConv(
                kernel_size=self.kernel_size,
                activation=activation,
                align_corners=True,
                interpolation=self.interpolation,
                coordinate_mapping=self.coordinate_mapping,
                normalize=False,
                **kwargs
            )
            return conv

        # Molecular-molecular convolution: particle-particle interactions
        self.conv0_molecular = Conv(
            in_channels=4 + other_feats_channels,
            filters=cconv_embedding_dim,
            activation=None
        )

        # Molecular-obstacle convolution: particle-boundary interactions
        self.conv0_obstacle = Conv(
            in_channels=3,
            filters=cconv_embedding_dim,
            activation=None
        )

        # Dense layer for direct feature transformation
        self.dense0_molecular = nn.Linear(
            in_features=4 + other_feats_channels,
            out_features=cconv_embedding_dim
        )
        nn.init.xavier_uniform_(self.dense0_molecular.weight)
        nn.init.zeros_(self.dense0_molecular.bias)
    
    def forward(self, inp_pos, query_pos, inp_vel, other_feats, box, box_feats, query_vel=None,
                inp_pos_hash_table=None, box_hash_table=None):
        filter_extent = torch.tensor(self.filter_extent, device=inp_pos.device, dtype=inp_pos.dtype)
        # ============= Step 1: Neighbor Search =============
        # Find molecular-molecular neighbors (particle-particle interactions)
        if inp_pos_hash_table is not None:
            molecular_molecular_nns = self.particle_radius_search(
                inp_positions=inp_pos,
                out_positions=query_pos,
                extents=filter_extent,
                fixed_radius_search_hash_table=inp_pos_hash_table)
        else:
            molecular_molecular_nns, inp_pos_hash_table = self.particle_radius_search(
                inp_positions=inp_pos,
                out_positions=query_pos,
                extents=filter_extent,
                return_hash_table=True)

        # Find molecular-obstacle neighbors (particle-boundary interactions)
        if box_hash_table is not None:
            molecular_obstacle_nns = self.particle_radius_search(
                inp_positions=box,
                out_positions=query_pos,
                extents=filter_extent,
                fixed_radius_search_hash_table=box_hash_table)
        else:
            molecular_obstacle_nns, box_hash_table = self.particle_radius_search(
                inp_positions=box,
                out_positions=query_pos,
                extents=filter_extent,
                return_hash_table=True)

        # ============= Step 2: Feature Extraction via Continuous Convolutions =============
        # Prepare molecular particle features: [1, velocities, other_features]
        molecular_feats = [torch.ones_like(inp_pos[:, 0:1]), inp_vel]  # Add constant feature for bias
        if other_feats is not None:
            molecular_feats.append(other_feats)
        molecular_feats = torch.cat(molecular_feats, axis=-1)

        # Extract features using continuous convolutions
        # Molecular-molecular convolution: Extract features from neighboring particles
        conv0_molecular_out = self.conv0_molecular(
            molecular_feats, inp_pos, query_pos, filter_extent,
            inp_importance=molecular_molecular_nns['inp_importance'],
            neighbors_index=molecular_molecular_nns['neighbors_index'],
            neighbors_row_splits=molecular_molecular_nns['neighbors_row_splits'],
            neighbors_importance=molecular_molecular_nns['neighbors_importance'])

        # Dense layer: Direct transformation of particle features
        if query_vel is not None:
            query_molecular_feats = [torch.ones_like(query_pos[:, 0:1]), query_vel]
            dense0_molecular_out = self.dense0_molecular(torch.cat(query_molecular_feats, axis=-1))
        else:
            dense0_molecular_out = self.dense0_molecular(molecular_feats)

        # Molecular-obstacle convolution: Extract features from boundary interactions
        conv0_obstacle_out = self.conv0_obstacle(
            box_feats, box, query_pos, filter_extent,
            inp_importance=molecular_obstacle_nns['inp_importance'],
            neighbors_index=molecular_obstacle_nns['neighbors_index'],
            neighbors_row_splits=molecular_obstacle_nns['neighbors_row_splits'],
            neighbors_importance=molecular_obstacle_nns['neighbors_importance'])

        # Combine all extracted features
        particle_features = torch.cat([
            conv0_obstacle_out,  # Boundary interaction features
            conv0_molecular_out,     # Particle-particle features
            dense0_molecular_out,    # Direct particle features
        ], axis=-1)  # Shape: [N, combined_feat_dim]
        
        return particle_features, inp_pos_hash_table, box_hash_table
        
        
class SuperParticleDecoderLayer(nn.Module):
    """
    A single decoder layer that performs:
    1. Cross-attention from query particles to super-particles
    2. Self-attention among query particles
    3. Feed-forward network

    This follows a standard transformer decoder architecture with pre-normalization.
    """

    def __init__(
        self,
        query_particle_feat_dim: int,
        super_particle_feat_dim: int,
        num_heads: int = 8,
        ffn_hidden_dim: int = 256,
        dropout: float = 0.1,
        bias: bool = True,
        activation: str = 'swiglu',
        norm_type: str = 'rms_norm',
        qk_norm: bool = True,
        enable_self_attn: bool = True,
        enable_ffn: bool = True,
    ):
        """
        Args:
            query_particle_feat_dim: Feature dimension of query particles
            super_particle_feat_dim: Feature dimension of super-particles (keys/values)
            num_heads: Number of attention heads
            ffn_hidden_dim: Hidden dimension for feed-forward network
            dropout: Dropout rate
            bias: Whether to use bias in linear layers
            activation: FFN activation function ('swiglu' or 'gelu')
            norm_type: Normalization type ('layer_norm' or 'rms_norm')
            qk_norm: Whether to apply QK normalization in attention
            enable_self_attn: Whether to keep self-attention in this layer
            enable_ffn: Whether to keep FFN in this layer
        """
        super().__init__()

        self.query_particle_feat_dim = query_particle_feat_dim
        self.super_particle_feat_dim = super_particle_feat_dim
        self.num_heads = num_heads
        self.head_dim = query_particle_feat_dim // num_heads
        self.enable_self_attn = bool(enable_self_attn)
        self.enable_ffn = bool(enable_ffn)

        assert query_particle_feat_dim % num_heads == 0, \
            f"query_particle_feat_dim {query_particle_feat_dim} must be divisible by num_heads {num_heads}"

        # Choose normalization module
        if norm_type == 'layer_norm':
            norm_module = nn.LayerNorm
        elif norm_type == 'rms_norm':
            norm_module = RMSNorm
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ============= Cross-Attention Components =============
        # Pre-norm for cross-attention
        self.query_norm = norm_module(query_particle_feat_dim, eps=1e-6)
        self.cross_kv_norm = norm_module(super_particle_feat_dim, eps=1e-6)

        # Multi-head cross-attention: query particles attend to super-particles
        self.multihead_cross_attn = MultiHeadAttention(
            query_dim=query_particle_feat_dim,
            num_heads=num_heads,
            kv_dim=super_particle_feat_dim,
            bias=bias,
            qk_norm=qk_norm,
            norm_type=norm_type
        )

        # ============= Self-Attention Components =============
        if self.enable_self_attn:
            # Pre-norm for self-attention
            self.self_attn_norm = norm_module(query_particle_feat_dim, eps=1e-6)

            # Multi-head self-attention among query particles
            self.self_attn = MultiHeadAttention(
                query_dim=query_particle_feat_dim,
                num_heads=num_heads,
                kv_dim=None,  # Self-attention
                bias=bias,
                qk_norm=qk_norm,
                norm_type=norm_type
            )
        else:
            self.self_attn_norm = None
            self.self_attn = None

        # ============= Feed-Forward Network =============
        if self.enable_ffn:
            # Pre-norm for FFN
            self.ffn_norm = norm_module(query_particle_feat_dim, eps=1e-6)

            # Choose FFN activation type
            if activation == 'swiglu':
                self.ffn = FeedForwardSwiGLU(
                    dim=query_particle_feat_dim,
                    hidden_dim=ffn_hidden_dim,
                    dropout=dropout,
                    bias=bias
                )
            elif activation == 'gelu':
                self.ffn = FeedForwardGeLU(
                    dim=query_particle_feat_dim,
                    hidden_dim=ffn_hidden_dim,
                    dropout=dropout,
                    bias=bias
                )
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        else:
            self.ffn_norm = None
            self.ffn = None

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        cross_attn_bias: torch.Tensor = None,
        self_attn_bias: torch.Tensor = None,
    ):
        """
        Forward pass of the decoder layer.

        Args:
            query: Query particle features [B, N_query, query_dim]
            kv: Super-particle features (keys/values) [B, N_super, super_dim]
            cross_attn_bias: [B, H, N_query, N_super] additive logit bias from
                the query-atom-to-super-particle distances (P011 D1(c)).
            self_attn_bias: [B, H, N_query, N_query] additive logit bias from
                the atom-atom distances.

        Returns:
            Updated query particle features [B, N_query, query_dim]

        P011 D1(c).  Upstream passed four RoPE tensors here, built from the
        query atoms' and the super-particles' absolute positions
        (`particle_rope` / `super_particle_rope`), and used them in *both* the
        cross-attention and the self-attention.  3D RoPE reads the lab-frame x,
        y and z components, so those were two of the three places where
        `h(Rx) != h(x)`.  Both are now additive per-head biases over pair
        distances.  The self-attention gets one too: dropping its RoPE without
        a replacement would leave eight layers of attention among 470 atoms
        with no geometry at all, which is a larger change in what the model
        *can* represent than this plan intends to make.
        """
        # ============= Cross-Attention Block =============
        # Query particles attend to super-particles
        residual = query
        q = self.query_norm(query)
        kv_normed = self.cross_kv_norm(kv)

        cross_attn_output = self.multihead_cross_attn(
            q=q,
            k=kv_normed,
            v=kv_normed,
            src_key_padding_mask=None,
            attn_bias=cross_attn_bias,
        )

        query = residual + self.dropout(cross_attn_output)

        # ============= Self-Attention Block =============
        # Query particles attend to each other
        if self.enable_self_attn:
            residual = query
            q = self.self_attn_norm(query)

            self_attn_output = self.self_attn(
                q=q,
                k=q,
                v=q,
                src_key_padding_mask=None,
                attn_bias=self_attn_bias,
            )
            query = residual + self.dropout(self_attn_output)

        # ============= Feed-Forward Block =============
        if self.enable_ffn:
            residual = query
            ffn_output = self.ffn(self.ffn_norm(query))
            query = residual + self.dropout(ffn_output)

        return query

class SuperParticleDecoder(nn.Module):
    """
    Super-Particle Decoder that stacks self-attention, cross-attention, and FFN layers.

    This decoder processes super-particles (from slot attention) by:
    1. Self-attention among super-particles (with optional RoPE)
    2. Cross-attention from super-particles to all particles (with RoPE)
    3. Feed-forward network

    All with residual connections and layer normalization.

    Architecture inspired by standard transformer decoder blocks.

    """

    def __init__(
        self,
        num_layers: int,
        query_particle_feat_dim: int,
        super_particle_feat_dim: int,
        num_heads: int = 8,
        ffn_hidden_dim: int = 256,
        dropout: float = 0.1,
        bias: bool = True,
        activation: str = 'swiglu',
        norm_type: str = 'rms_norm',
        qk_norm: bool = True,
        enable_self_attn: bool = True,
        enable_ffn: bool = True,
    ):
        """
        Args:
            dim: Model dimension (must match input feature dim)
            num_heads: Number of attention heads
            ffn_hidden_dim: Hidden dimension for FFN
            dropout: Dropout rate
            bias: Whether to use bias in linear layers
            activation: FFN activation ('swiglu' or 'gelu')
            norm_type: Normalization type ('layer_norm' or 'rms_norm')
            qk_norm: Whether to apply QK normalization in attention
            enable_self_attn: Whether to keep self-attention in each decoder layer
            enable_ffn: Whether to keep FFN in each decoder layer
            use_rope_self_attn: Whether to use RoPE in self-attention
            use_rope_cross_attn: Whether to use RoPE in cross-attention
            rope_dim: Dimension for RoPE encoding (auto-computed if None)
        """
        super().__init__()

        self.query_particle_feat_dim = query_particle_feat_dim
        self.super_particle_feat_dim = super_particle_feat_dim
        self.num_heads = num_heads
        self.head_dim = query_particle_feat_dim // num_heads
        assert query_particle_feat_dim % num_heads == 0, f"query_particle_feat_dim {query_particle_feat_dim} must be divisible by num_heads {num_heads}"

        # layers
        self.layers = nn.ModuleList([
            SuperParticleDecoderLayer(
                query_particle_feat_dim=query_particle_feat_dim,
                super_particle_feat_dim=super_particle_feat_dim,
                num_heads=num_heads,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                bias=bias,
                activation=activation,
                norm_type=norm_type,
                qk_norm=qk_norm,
                enable_self_attn=enable_self_attn,
                enable_ffn=enable_ffn,
            ) for _ in range(num_layers)
        ])
                

    def forward(
        self,
        super_particles_feat: torch.Tensor,
        particles_feat: torch.Tensor,
        cross_attn_bias: torch.Tensor = None,
        self_attn_bias: torch.Tensor = None,
    ):
        # Both biases are computed once by the caller and shared across layers:
        # positions do not change inside the decoder, so recomputing them per
        # layer would cost eight times as much for the same numbers.
        for idx, layer in enumerate(self.layers):
            particles_feat = layer(
                query=particles_feat,
                kv=super_particles_feat,
                cross_attn_bias=cross_attn_bias,
                self_attn_bias=self_attn_bias,
            )
        return particles_feat
