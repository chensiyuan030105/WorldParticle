"""SE(3)-invariant replacements for WorldParticle's three lab-frame pieces.

P011 D1.  `wp-v2` feeds the *direction* of a neighbour offset into three
places, and each one makes `h(Rx) != h(x)`:

  1. `LocalFeatureExtractor.conv0_molecular` -- open3d `ContinuousConv` looks
     the kernel up in a fixed-orientation 4x4x4 cube indexed by the relative
     offset *vector* (`coordinate_mapping='ball_to_cube_volume_preserving'`),
     so rotating the molecule changes which cell a neighbour lands in.
  2. the slot attention's `input_rope` and
  3. the decoder's `particle_rope` / `super_particle_rope` -- 3D RoPE rotates
     q and k by angles read off the lab-frame x, y and z components.

P004 section 2.9.1 measured the result: `h` is translation invariant to 3.7e-07
but rotation leaves a residual of 0.999 even after fitting the best linear map.
P010 section 15.2 then showed the fusion pathway can carry per-atom
conformational information (training objective +3.08% -> -13.34%) provided what
goes in is an SE(3) invariant.  So the fix belongs in the architecture, not in
a pre-alignment step: `canonicalize_window` hides the asymmetry from one
caller but the content still carries the window's pose.

What replaces them here keeps the same three computations but reads only
`||p_i - p_j||`, a scalar invariant:

  * `RadialMessagePassing` -- `out_i = sum_j imp_ij * filter(d_ij) * (W f_j)`,
    a SchNet-style continuous-filter convolution.  Same forward signature as
    `ContinuousConv`, same neighbour lists from the same `FixedRadiusSearch`.
  * `DistanceBias` -- a per-head additive attention bias `MLP(RBF(d_ij))`,
    which is what a relative position encoding does for a sequence and is the
    standard way to give attention geometry without giving it a frame.

Precision is load-bearing and is handled explicitly.  P010 section 16 records a
matmul that was left outside `autocast`, came back bf16, rounded coordinates to
about three significant digits and turned a 25 rad/nm channel into a 0.41
error -- in an arm whose whole point was that the frames agreed.  The same
shape of bug would silently cap invariance here, because every digit lost in
`d_ij` is a digit lost in `h(Rx) - h(x)`.  So the geometry *and* the small nets
that consume it run in float32 with autocast disabled; the content path
(`W f_j`, the attention itself) is left in whatever precision the caller chose.
The nets are tiny -- a few hundred MFLOPs against the model's tens of GFLOPs --
so this costs almost nothing and buys an invariance residual set by float32
rather than by bf16.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


def _no_autocast(t: torch.Tensor):
    """Autocast off for `t`'s device, so float32 geometry stays float32.

    `torch.amp.autocast(..., enabled=False)` needs a device type, and a cuda
    context is not a no-op on a CPU-only run, so the device comes from the
    tensor.  Returns `nullcontext()` for devices autocast does not cover.
    """
    dev = t.device.type
    if dev in ("cuda", "cpu"):
        return torch.amp.autocast(dev, enabled=False)
    return nullcontext()


def pair_distances(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12
                   ) -> torch.Tensor:
    """`[..., M, N]` distances between `[..., M, 3]` and `[..., N, 3]`.

    `clamp_min` before the square root, not after: `sqrt` has an infinite
    derivative at 0, so a coincident pair (or a point against itself) would put
    a NaN in the backward pass while the forward pass looked fine.
    """
    a32, b32 = a.float(), b.float()
    sq = (a32.unsqueeze(-2) - b32.unsqueeze(-3)).pow(2).sum(-1)
    return sq.clamp_min(eps).sqrt()


class RadialBasis(nn.Module):
    """A distance expanded into `num_basis` Gaussians plus a monotone tail.

    The Gaussians are evenly spaced over `[0, cutoff]` with width equal to the
    spacing, which is the usual SchNet choice: narrower leaves gaps between
    centres, wider washes the resolution out.

    The extra `tanh(d / cutoff)` channel exists because the Gaussians are
    *local*.  Beyond about `cutoff + 2 * width` every one of them underflows and
    all such pairs look identical -- fine for `RadialMessagePassing`, whose
    neighbours are bounded by the search radius by construction, but wrong for
    `DistanceBias`, where two atoms can be the full diameter of the molecule
    apart.  One monotone channel keeps "far" ordered instead of saturated, and
    costs one input to a small MLP.
    """

    def __init__(self, num_basis: int = 16, cutoff: float = 1.0) -> None:
        super().__init__()
        if num_basis < 2:
            raise ValueError(f"num_basis must be >= 2, got {num_basis}")
        if not cutoff > 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        self.num_basis = int(num_basis)
        self.cutoff = float(cutoff)
        centers = torch.linspace(0.0, self.cutoff, self.num_basis)
        self.register_buffer("centers", centers)
        width = self.cutoff / (self.num_basis - 1)
        self.register_buffer("gamma", torch.tensor(1.0 / (2.0 * width * width)))

    @property
    def out_channels(self) -> int:
        return self.num_basis + 1

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """`[...]` -> `[..., num_basis + 1]`, float32."""
        d32 = d.float()
        gauss = torch.exp(-self.gamma * (d32.unsqueeze(-1) - self.centers) ** 2)
        tail = torch.tanh(d32 / self.cutoff).unsqueeze(-1)
        return torch.cat([gauss, tail], dim=-1)


class RadialMessagePassing(nn.Module):
    """Invariant stand-in for `ContinuousConv` on the particle branches.

        out_i = sum_{j in N(i)} imp_ij * filter(d_ij) (*) (W f_j) + b

    `forward`'s signature matches `ContinuousConv.forward` exactly, including
    the unused `extents` and `inp_importance`, so the call sites in
    `LocalFeatureExtractor` are unchanged and a reader can diff the two
    implementations rather than the plumbing.

    `neighbors_importance` is kept and is not decoration.  It is
    `window_poly6(d^2 / radius^2)`, a function of the distance alone, so it is
    invariant; and it goes smoothly to zero at the cutoff, which is what keeps
    invariance *robust*.  Without it an atom sitting exactly at the search
    radius could enter or leave `N(i)` on the 1e-7 coordinate change that
    rotating a float32 window costs, and its full-weight contribution would
    appear or vanish.  With it, such an atom carries weight ~0 either way.

    The centre atom is absent on purpose: the search runs with
    `ignore_query_point=True`, and `LocalFeatureExtractor` already supplies the
    self term through `dense0_molecular`.  That split is upstream's and is
    preserved.
    """

    def __init__(self, in_channels: int, filters: int, cutoff: float,
                 num_basis: int = 16, filter_hidden: int = 64) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.filters = int(filters)
        self.rbf = RadialBasis(num_basis=num_basis, cutoff=cutoff)
        # No bias on `value`: a bias here would be summed once per neighbour,
        # turning the layer's offset into a function of the neighbour count.
        # The per-channel offset belongs on the output, where `self.bias` is.
        self.value = nn.Linear(self.in_channels, self.filters, bias=False)
        self.filter_net = nn.Sequential(
            nn.Linear(self.rbf.out_channels, filter_hidden),
            nn.SiLU(),
            nn.Linear(filter_hidden, self.filters),
        )
        self.bias = nn.Parameter(torch.zeros(self.filters))
        nn.init.xavier_uniform_(self.value.weight)
        for m in self.filter_net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Read by kineidos/probe_h_symmetry.py and by the plan's neighbour-graph
        # sanity check; ContinuousConv sets the same attribute.
        self._avg_neighbors = 0.0

    def forward(self, inp_features, inp_positions, out_positions, extents,
                inp_importance, neighbors_index, neighbors_row_splits,
                neighbors_importance):
        n_out = out_positions.shape[0]
        counts = (neighbors_row_splits[1:] - neighbors_row_splits[:-1]).to(
            torch.long)
        n_pairs = int(neighbors_index.shape[0])
        self._avg_neighbors = n_pairs / max(n_out, 1)

        out = inp_features.new_zeros((n_out, self.filters), dtype=torch.float32)
        if n_pairs == 0:
            return out + self.bias.float()

        qi = torch.repeat_interleave(
            torch.arange(n_out, device=out_positions.device), counts)
        nj = neighbors_index.to(torch.long)

        with _no_autocast(out_positions):
            d = (out_positions.float()[qi] - inp_positions.float()[nj])
            d = d.pow(2).sum(-1).clamp_min(1e-12).sqrt()        # [P]
            w = self.filter_net(self.rbf(d))                    # [P, filters]
            if neighbors_importance is not None and neighbors_importance.numel():
                w = w * neighbors_importance.float().unsqueeze(-1)

        # The content path keeps the caller's precision; the product promotes to
        # float32 because `w` is float32.
        msg = w * self.value(inp_features)[nj].float()
        out = out.index_add(0, qi, msg)
        return out + self.bias.float()


class DistanceBias(nn.Module):
    """Per-head additive attention bias from pair distances.

    Replaces 3D RoPE.  RoPE rotates q and k by angles taken from the lab-frame
    coordinates, which is why `wp-v2` is not rotation invariant; an additive
    `bias_ij = MLP(RBF(||p_i - p_j||))` gives attention the same kind of
    geometric prior out of a scalar invariant instead.

    Returns `[B, n_heads, M, N]`, the shape `F.scaled_dot_product_attention`
    takes as a float `attn_mask`.
    """

    def __init__(self, n_heads: int, cutoff: float, num_basis: int = 16,
                 hidden: int = 32) -> None:
        super().__init__()
        self.n_heads = int(n_heads)
        self.rbf = RadialBasis(num_basis=num_basis, cutoff=cutoff)
        self.net = nn.Sequential(
            nn.Linear(self.rbf.out_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.n_heads),
        )
        # Zero on the output layer, so at initialisation the bias is exactly 0
        # and attention is the plain content attention.  The alternative -- a
        # random bias over 470x470 pairs -- would hand the untrained model a
        # strong arbitrary geometric prior, and the checkup in plan section 5.3
        # reads activations *at random initialisation*, where that prior would
        # be indistinguishable from signal.
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, q_pos: torch.Tensor, kv_pos: torch.Tensor
                ) -> torch.Tensor:
        """`[B, M, 3]`, `[B, N, 3]` -> `[B, n_heads, M, N]` float32."""
        with _no_autocast(q_pos):
            d = pair_distances(q_pos, kv_pos)            # [B, M, N]
            bias = self.net(self.rbf(d))                 # [B, M, N, n_heads]
        return bias.permute(0, 3, 1, 2).contiguous()
