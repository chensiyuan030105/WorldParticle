import torch
import torch.nn as nn
import numpy as np
import time
from .cross_attn_feat import CrossAttentionWeightedPoswithFeature
from .invariant import DistanceBias
from .super_particle_layers import LocalFeatureExtractor, SuperParticleDecoder
from TrajDataset import uses_obstacle_features, uses_topology_neighbors

class ParticleNetworkCrossAttnLocalFeature(torch.nn.Module):
    """
    Particle Network with Slot Attention for adaptive super-particle generation.

    Architecture:
    1. Initial feature extraction via continuous convolutions
    2. Slot attention to generate super-particles (adaptive sampling)
    3. Super-particle position prediction
    4. Self-attention among super-particles
    5. Cross-attention from all particles to super-particles with RoPE
    6. Position correction prediction
    """

    def __init__(
            self,
            # ============= Continuous Convolution Parameters =============
            kernel_size=[4, 4, 4],
            cconv_embedding_dim=32,
            radius_scale=1.5,
            data=None,
            use_topology_conv=None,
            topology_search_radius=None,
            topology_blend_alpha=None,
            coordinate_mapping='ball_to_cube_volume_preserving',
            interpolation='linear',
            use_window=True,
            particle_radius=0.025,

            # ============= Position Encoding Parameters =============
            particle_position_rope_dim=96,  # Must be divisible by 6 for 3D positions

            # ============= Slot Attention Parameters =============
            slot_compression_ratio=10,  # N_particles / N_slots ratio
            slot_attn_iters=3,
            slot_attn_heads=6,
            
            # ============= Decoder Parameters =============
            num_decoder_layers=3,
            decoder_attn_heads=4,
            decoder_attn_ffn_hidden_dim=384,
            decoder_attn_dropout=0.1,

            # ============= Output Network Parameters =============
            output_hidden_dim=128,
            output_layers=2,

            # ============= Physics Parameters =============
            timestep=1 / 50,
            gravity=(0, -9.81, 0),
            other_feats_channels=0,
            obstacle_feats_channels=3,

            # ============= Invariance Parameters (P011 D1) =============
            # Range of the attention distance bias, in the same length unit as
            # the positions (nm for Kineidos).  Unlike the convolution, whose
            # pairs are bounded by the search radius, attention is all-to-all:
            # two atoms can be the whole diameter of the molecule apart.  2.0
            # nm covers GAGU's 470 heavy atoms (extent ~2.5 nm) with 16 centres
            # 0.13 nm apart; beyond it the monotone tail channel of the RBF
            # keeps "far" ordered rather than saturated (models/invariant.py).
            # It is a length scale like particle_radius, so it is configured,
            # not inferred.
            distance_bias_cutoff=2.0,
            distance_bias_num_basis=16,
            normalize_other_feats=True,
            normalize_feature_branches=True,

            # ============= Debug Parameters =============
            verbose=False,
            velhead=False,
    ):
        super().__init__()

        # ============= Store Basic Configuration =============
        self.velhead = velhead
        self.data = data
        self.topology_search_radius = topology_search_radius
        self.verbose = verbose
        self.timestep = timestep
        self.gravity_values = list(gravity)
        self.slot_compression_ratio = slot_compression_ratio
        self.decoder_enable_self_attn = self.data not in {"cloth_no_self_attention", "cloth_no_all"}
        self.decoder_enable_ffn = self.data not in {"cloth_no_ffn", "cloth_no_all"}

        if use_topology_conv is None:
            self.use_topology_conv = uses_topology_neighbors(self.data)
        else:
            self.use_topology_conv = bool(use_topology_conv)

        # ============ RoPE is gone (P011 D1) =============
        # `particle_position_rope_dim` is still accepted so that every existing
        # caller and config keeps working, but nothing reads it: the two 3D
        # RoPEs it sized have been replaced by distance biases, which have no
        # per-head dimension to match.  The two asserts that used to live here
        # constrained it to be divisible by 6 and to fit in a head -- both
        # statements about RoPE, both now vacuous.
        self.particle_position_rope_dim = particle_position_rope_dim
        self.distance_bias_cutoff = float(distance_bias_cutoff)
        self.distance_bias_num_basis = int(distance_bias_num_basis)

        # ============= Continuous Convolution Configuration =============
        self.local_feature_extractor = LocalFeatureExtractor(
            kernel_size=kernel_size,
            cconv_embedding_dim=cconv_embedding_dim,
            data=data,
            use_topology_conv=self.use_topology_conv,
            topology_search_radius=topology_search_radius,
            topology_blend_alpha=topology_blend_alpha,
            radius_scale=radius_scale,
            coordinate_mapping=coordinate_mapping,
            interpolation=interpolation,
            use_window=use_window,
            particle_radius=particle_radius,
            other_feats_channels=other_feats_channels,
            obstacle_feats_channels=obstacle_feats_channels,
            normalize_other_feats=normalize_other_feats,
            normalize_feature_branches=normalize_feature_branches,
        )
        
        # ============= Calculate Feature Dimensions =============
        # Combined dimension: obstacle conv + molecular conv + dense layer
        if self.data == "local_fluid":
            combined_feat_dim = cconv_embedding_dim * 3
        elif uses_obstacle_features(self.data):
            combined_feat_dim = cconv_embedding_dim * 3
        else:
            combined_feat_dim = cconv_embedding_dim * 2 
        
        # ============= Initialize Slot Attention =============
        self.super_particle_cross_attn_feat = CrossAttentionWeightedPoswithFeature(
            dim=combined_feat_dim,
            filter_extent=self.local_feature_extractor.filter_extent,
            iters=slot_attn_iters,
            n_heads=slot_attn_heads,
            eps=1e-8,
            distance_bias_cutoff=self.distance_bias_cutoff,
            distance_bias_num_basis=self.distance_bias_num_basis,
        )

        # ============= Distance biases for the decoder (P011 D1(c)) =========
        # These two replace `particle_rope` and `super_particle_rope`.  Two
        # modules rather than one because the pairs are different kinds --
        # atom-to-super-particle for the cross-attention, atom-to-atom for the
        # self-attention -- and sharing weights would force one radial profile
        # on both.  Randomly initialised, not zero: with a zero bias 470 query
        # atoms attend to 7 super-particles with nothing distinguishing the
        # pairs, so every atom receives the same mixture and `h` comes out a
        # cross-atom constant.  Measured -- see models/invariant.py's
        # DistanceBias.__init__.
        self.decoder_cross_bias = DistanceBias(
            n_heads=decoder_attn_heads,
            cutoff=self.distance_bias_cutoff,
            num_basis=self.distance_bias_num_basis,
        )
        self.decoder_self_bias = DistanceBias(
            n_heads=decoder_attn_heads,
            cutoff=self.distance_bias_cutoff,
            num_basis=self.distance_bias_num_basis,
        )
        
        # ============= Initialize Super-Particle Decoder =============
        self.super_particle_decoder = SuperParticleDecoder(
            num_layers=num_decoder_layers,
            query_particle_feat_dim=combined_feat_dim,
            super_particle_feat_dim=combined_feat_dim,
            num_heads=decoder_attn_heads,
            ffn_hidden_dim=decoder_attn_ffn_hidden_dim,
            dropout=decoder_attn_dropout,
            bias=True,
            activation='swiglu',
            norm_type='rms_norm',
            qk_norm=True,
            enable_self_attn=self.decoder_enable_self_attn,
            enable_ffn=self.decoder_enable_ffn,
        )

        # ============= Initialize Output Network =============
        self.output_network = self._build_output_network(
            input_dim=combined_feat_dim,
            hidden_dim=output_hidden_dim,
            num_layers=output_layers
        )

        # Output scaling factor for numerical stability
        self.output_scale = 1.0 / 128

    def apply_radius_override(self, radius_scale=None, particle_radius=None):
        """
        Re-apply radius-dependent values after checkpoint loading.
        This is needed because state_dict restores buffers such as
        `local_feature_extractor.filter_extent_tensor`.
        """
        lfe = self.local_feature_extractor
        if radius_scale is not None:
            lfe.radius_scale = float(radius_scale)
        if particle_radius is not None:
            lfe.particle_radius = float(particle_radius)

        new_filter_extent = np.float32(lfe.radius_scale * 6 * lfe.particle_radius)
        lfe.filter_extent = new_filter_extent

        if hasattr(lfe, "filter_extent_tensor"):
            lfe.filter_extent_tensor.data.fill_(float(new_filter_extent))

        if hasattr(lfe, "topology_filter_extent_tensor"):
            if getattr(lfe, "topology_search_radius", None) is None:
                lfe.topology_filter_extent = new_filter_extent
            else:
                lfe.topology_filter_extent = np.float32(
                    lfe.radius_scale * 6 * float(lfe.topology_search_radius)
                )
            lfe.topology_filter_extent_tensor.data.fill_(float(lfe.topology_filter_extent))

        if hasattr(self, "super_particle_cross_attn_feat"):
            self.super_particle_cross_attn_feat.filter_extent = float(new_filter_extent)


    def _build_output_network(self, input_dim, hidden_dim, num_layers):
        """Build the output MLP network for position correction."""
        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            if i < num_layers - 1:
                # Hidden layers with activation and normalization
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim)
                ])
                current_dim = hidden_dim
            else:
                # Final layer outputs 3D position corrections
                if self.velhead:
                    layers.append(nn.Linear(current_dim, 6))
                else:
                    layers.append(nn.Linear(current_dim, 3))

        return nn.Sequential(*layers)

    def integrate_pos_vel(self, pos1, vel1, external_force=None):
        """
        Apply gravity and integrate position and velocity using Verlet integration.

        Uses semi-implicit Euler method for stable integration:
        1. Update velocity with gravity acceleration
        2. Update position using average of old and new velocity

        Args:
            pos1: Current positions [N, 3]
            vel1: Current velocities [N, 3]

        Returns:
            pos2: Predicted positions after timestep [N, 3]
            vel2: Predicted velocities after timestep [N, 3]
        """
        dt = self.timestep

        if external_force is not None:
            gravity = external_force.to(dtype=vel1.dtype, device=vel1.device)
        else:
            # Create gravity tensor fresh each time to avoid inplace operation issues
            gravity = torch.tensor(self.gravity_values, dtype=vel1.dtype, device=vel1.device)
            # Broadcast gravity to match particle count
            if vel1.dim() == 2:
                gravity = gravity.unsqueeze(0).expand(vel1.size(0), -1)

        # Semi-implicit Euler integration
        vel2 = vel1 + (dt * gravity)  # Update velocity with gravity
        pos2 = pos1 + (dt * (vel2 + vel1) / 2)  # Update position with average velocity
        return pos2, vel2

    def compute_new_pos_vel(self, pos1, vel1, pos2, vel2, pos_correction):
        """
        Apply position correction and compute corrected velocities.

        The correction is applied to the predicted position, and velocity
        is recomputed based on the total displacement from the original position.

        Args:
            pos1: Original positions at start of timestep [N, 3]
            vel1: Original velocities at start of timestep [N, 3]
            pos2: Predicted positions after integration [N, 3]
            vel2: Predicted velocities after integration [N, 3]
            pos_correction: Learned position corrections [N, 3]

        Returns:
            pos: Corrected final positions [N, 3]
            vel: Corrected final velocities [N, 3]
        """
        dt = self.timestep
        pos = pos2 + pos_correction  # Apply correction to predicted position
        vel = (pos - pos1) / dt  # Recompute velocity from total displacement
        return pos, vel

    def compute_correction(self,
                           pos,
                           vel,
                           other_feats,
                           box=None,
                           box_feats=None,
                           k_partite: int = -1,
                           fixed_radius_search_hash_table=None,
                           batch_info=None,
                           topology_neighbors_index=None,
                           topology_neighbors_row_splits=None,
                           return_tokens: bool = False,
                           skip_output_network: bool = False):
        """
        Compute position correction using slot attention and super-particle decoder.

        This is the main processing pipeline that:
        1. Finds particle neighbors
        2. Extracts local features via continuous convolutions
        3. Creates super-particles via slot attention
        4. Processes super-particles through the decoder
        5. Outputs position corrections

        Args:
            pos: Particle positions [N, 3]
            vel: Particle velocities [N, 3]
            other_feats: Additional particle features [N, C] or None
            box: Obstacle/boundary positions [M, 3]
            box_feats: Obstacle/boundary features [M, Cb]
            fixed_radius_search_hash_table: Pre-computed spatial hash table (optional)
            batch_info: Optional batching information (currently unused)
            return_tokens: Also return the per-particle tokens `h` -- the
                super-particle decoder's output, taken before output_network.
                Kineidos conditions Protenix's diffusion head on these.
            skip_output_network: Only meaningful with return_tokens.  Stop after
                the decoder instead of running WorldParticle's own
                position/velocity head.

        Returns:
            With return_tokens=False (the default, unchanged from upstream):
                a tuple, `(pos_correction, vel_correction, super_particle_positions)`
                when self.velhead else `(pos_correction, super_particle_positions)`.

            With return_tokens=True: a dict, because the tuple's arity already
                varies with self.velhead and adding a token entry to it would
                make callers guess at positions.  Always has `tokens` and
                `super_particle_positions`; has `pos_correction` (and
                `vel_correction` when self.velhead) unless skip_output_network.

        `skip_output_network` exists because the two callers want different
        things.  With `wp.aux_next_frame_loss` on, output_network is reattached
        as an auxiliary supervision head, so we need `h` *and* its next-frame
        prediction -- the head still runs.  With the switch off nothing consumes
        the prediction, and running the head would only build a graph that is
        then discarded.  Hence a flag rather than an unconditional early return.
        """
        t2 = time.time()
        
        particle_features, num_molecular_neighbors, _ = self.local_feature_extractor(
            inp_pos=pos, query_pos=pos, inp_vel=vel, other_feats=other_feats,
            box=box, box_feats=box_feats,
            fixed_radius_search_hash_table=fixed_radius_search_hash_table,
            batch_info=batch_info,
            topology_neighbors_index=topology_neighbors_index,
            topology_neighbors_row_splits=topology_neighbors_row_splits)

        if num_molecular_neighbors is None:
            self.num_molecular_neighbors = torch.zeros(
                particle_features.shape[0],
                dtype=particle_features.dtype,
                device=particle_features.device,
            )
        else:
            self.num_molecular_neighbors = num_molecular_neighbors # for loss calculation

        t3 = time.time()
        if self.verbose:
            print(f"Feature extraction time: {t3 - t2:.4f} seconds")
            import pdb; pdb.set_trace()

        # ============= Step 3: Slot Attention for Super-Particle Generation =============
        # Convert to batch format for slot attention
        if batch_info is None:
            particle_features_batch = particle_features.unsqueeze(0)  # [1, N, D]
            pos_batch = pos.unsqueeze(0)  # [1, N, 3]
        else:
            Batch = batch_info['batch_size']
            particle_features_batch = particle_features.reshape(Batch, -1, particle_features.shape[-1])  # [B, N/B, D]
            pos_batch = pos.reshape(Batch, -1, 3)  # [B, N/B, 3]

        # Generate super-particles through iterative slot attention
        super_particle_features_batch, super_particle_positions_batch = self.super_particle_cross_attn_feat(
            inputs_feat_batch=particle_features_batch,
            input_pos_batch=pos_batch,
            k_partite=k_partite,
        )  # Output: [1, num_slots, D]
        
        t4 = time.time()
        if self.verbose:
            print(f"Slot attention time: {t4 - t3:.4f} seconds")
            import pdb; pdb.set_trace()

        # ============= Step 4: Compute the decoder's distance biases =========
        # P011 D1(c).  Was two 3D RoPEs over absolute positions.  Computed once
        # here and shared across all decoder layers: positions do not change
        # inside the decoder.  The super-particle positions come from ToMe's
        # weighted mean of the merged particles' positions, which is
        # equivariant, so a distance to them is invariant.
        decoder_cross_bias = self.decoder_cross_bias(
            pos_batch, super_particle_positions_batch
        )
        decoder_self_bias = (
            self.decoder_self_bias(pos_batch, pos_batch)
            if self.super_particle_decoder.layers[0].enable_self_attn else None
        )

        t5 = time.time()
        if self.verbose:
            print(f"Distance bias computation time: {t5 - t4:.4f} seconds")
            import pdb; pdb.set_trace()

        # ============= Step 6: Super-Particle Decoder Processing =============
        # Process through decoder with self-attention and cross-attention layers
        output_features = self.super_particle_decoder(
            particles_feat=particle_features_batch, # query
            super_particles_feat=super_particle_features_batch, # kv
            cross_attn_bias=decoder_cross_bias,
            self_attn_bias=decoder_self_bias,
        )

        t6 = time.time()
        if self.verbose:
            print(f"Decoder processing time: {t6 - t5:.4f} seconds")
            import pdb; pdb.set_trace()

        # Kineidos: `output_features` is the per-particle tokens h.  Squeeze the
        # batch dim on the unbatched path so h follows the same convention as
        # `out` below, which upstream squeezes under the same condition.
        if return_tokens:
            tokens = output_features.squeeze(0) if batch_info is None else output_features
            if skip_output_network:
                return {"tokens": tokens,
                        "super_particle_positions": super_particle_positions_batch}

        # ============= Step 7: Generate Position Corrections =============
        # Pass decoder output through final network to get position corrections
        out = self.output_network(output_features)

        if batch_info is None:
            out = out.squeeze(0)

            if self.velhead:
                pos_correction = out[:, :3]
                vel_correction = out[:, 3:]
                pos_correction = pos_correction  * self.output_scale
                vel_correction = vel_correction * (self.output_scale )
            else:
                pos_correction = out
                pos_correction = pos_correction  * self.output_scale
        else:
            if self.velhead:
                pos_correction = out[:, :, :3].reshape(-1, 3)
                vel_correction = out[:, :, 3:].reshape(-1, 3)
                pos_correction = pos_correction  * self.output_scale
                vel_correction = vel_correction * (self.output_scale )
            else:
                pos_correction = out.reshape(-1, 3)
                pos_correction = pos_correction  * self.output_scale
        
        if return_tokens:
            result = {"tokens": tokens,
                      "super_particle_positions": super_particle_positions_batch,
                      "pos_correction": pos_correction}
            if self.velhead:
                result["vel_correction"] = vel_correction
            return result

        if self.velhead:
            return pos_correction, vel_correction, super_particle_positions_batch
        else:
            return pos_correction, super_particle_positions_batch

    def forward(self, inputs, fixed_radius_search_hash_table=None, batch_info=None,
                return_super_particle_positions=False,
                k_partite: int = -1,
                topology_neighbors_index=None,
                topology_neighbors_row_splits=None,
                external_force=None):
        """
        Compute one simulation timestep.

        This is the main entry point that orchestrates the full simulation step:
        1. Integrate positions/velocities with physics (gravity)
        2. Compute learned corrections via neural network
        3. Apply corrections to get final positions/velocities

        Args:
            inputs: Tuple of (pos, vel, feats, box, box_feats) where:
                - pos: Particle positions [N, 3]
                - vel: Particle velocities [N, 3]
                - feats: Additional particle features [N, C] or None
                - box: Obstacle/boundary positions [M, 3]
                - box_feats: Obstacle/boundary features [M, Cb]
            fixed_radius_search_hash_table: Pre-computed spatial hash (optional)
            batch_info: Batching information (currently unused)
            return_super_particle_positions: Whether to return super-particle info
            compute_jsd_loss: Whether to compute JSD loss (saves computation if False)

        Returns:
            Tuple of (pos_corrected, vel_corrected):
                - pos_corrected: Corrected positions after timestep [N, 3]
                - vel_corrected: Corrected velocities after timestep [N, 3]
        """
        # Unpack input tensors
        if len(inputs) == 5:
            pos, vel, feats, box, box_feats = inputs
        elif len(inputs) == 7:
            pos, vel, feats, box, box_feats, topology_neighbors_index_inp, topology_neighbors_row_splits_inp = inputs
            if topology_neighbors_index is None:
                topology_neighbors_index = topology_neighbors_index_inp
            if topology_neighbors_row_splits is None:
                topology_neighbors_row_splits = topology_neighbors_row_splits_inp
        else:
            raise ValueError(f"Unexpected input tuple length: {len(inputs)}. Expected 5 or 7.")

        # Step 1: Physics-based integration (gravity + velocity)
        if self.velhead:
            pos2, vel2 = self.integrate_pos_vel(pos, vel, external_force=external_force)
        else:
            pos2, vel2 = pos, vel
        
        # Step 2: Compute learned corrections using neural network
        if self.velhead:
            pos_correction, vel_correction, super_particle_positions_batch = self.compute_correction(
            pos2, vel2, feats, 
            box, box_feats,
            k_partite=k_partite,
            fixed_radius_search_hash_table=fixed_radius_search_hash_table,
            batch_info=batch_info,
            topology_neighbors_index=topology_neighbors_index,
            topology_neighbors_row_splits=topology_neighbors_row_splits)

            pos2_corrected = pos2 + pos_correction
            vel2_corrected = vel2 + vel_correction
        else: 
            pos_correction, super_particle_positions_batch = self.compute_correction(
                pos2, vel2, feats, 
                box, box_feats,
                k_partite=k_partite,
                fixed_radius_search_hash_table=fixed_radius_search_hash_table,
                batch_info=batch_info,
                topology_neighbors_index=topology_neighbors_index,
                topology_neighbors_row_splits=topology_neighbors_row_splits)
            pos2_corrected, vel2_corrected = self.compute_new_pos_vel(
                pos, vel, pos2, vel2, pos_correction)

        if return_super_particle_positions:
            return pos2_corrected, vel2_corrected, super_particle_positions_batch
        
        return pos2_corrected, vel2_corrected
