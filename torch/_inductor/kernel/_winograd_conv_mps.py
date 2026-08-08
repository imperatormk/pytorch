# mypy: allow-untyped-defs
"""Split winograd F(4x4, 3x3) forward convolution for MPS.

Three stages: an input-transform kernel writes U[36, T, C] (T = N * ceil(OH/4)
* ceil(OW/4) tiles), one batched matmul U @ V -> M[36, T, K] against the
transformed weights, and an output-transform kernel scatters M into the
output. The transform kernels are generated code: 36 named patch values per
program with the B^T / A^T coefficients folded in as literals (zeros
skipped), so there is no runtime tap indexing. The lane axis is the channel
axis, which is unit-stride for channels-last tensors; strides are taken from
the real tensors so any layout is handled.

F(4x4) is numerically weaker than direct convolution: ~1e-5 relative error
against float64 (direct conv is ~5e-7), from the larger transform
coefficients. That is two orders below the fp32 long-reduction error this
backend already accepts in weight gradients.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _wino_u_kernel(X, U, C, H, W, PH, PW, TW, THW, T, sxn, sxc, sxh, sxw,
                   BLOCK_C: tl.constexpr):
    t = tl.program_id(0)
    cb = tl.program_id(1)
    n = t // THW
    th = (t % THW) // TW
    tw = t % TW
    c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = c < C
    h0 = th * 4 - PH
    w0 = tw * 4 - PW
    base = X + n * sxn + c * sxc
    m_0_0 = cmask & (h0 + 0 >= 0) & (h0 + 0 < H) & (w0 + 0 >= 0) & (w0 + 0 < W)
    d_0_0 = tl.load(base + (h0 + 0) * sxh + (w0 + 0) * sxw, mask=m_0_0, other=0.0)
    m_0_1 = cmask & (h0 + 0 >= 0) & (h0 + 0 < H) & (w0 + 1 >= 0) & (w0 + 1 < W)
    d_0_1 = tl.load(base + (h0 + 0) * sxh + (w0 + 1) * sxw, mask=m_0_1, other=0.0)
    m_0_2 = cmask & (h0 + 0 >= 0) & (h0 + 0 < H) & (w0 + 2 >= 0) & (w0 + 2 < W)
    d_0_2 = tl.load(base + (h0 + 0) * sxh + (w0 + 2) * sxw, mask=m_0_2, other=0.0)
    m_0_3 = cmask & (h0 + 0 >= 0) & (h0 + 0 < H) & (w0 + 3 >= 0) & (w0 + 3 < W)
    d_0_3 = tl.load(base + (h0 + 0) * sxh + (w0 + 3) * sxw, mask=m_0_3, other=0.0)
    m_0_4 = cmask & (h0 + 0 >= 0) & (h0 + 0 < H) & (w0 + 4 >= 0) & (w0 + 4 < W)
    d_0_4 = tl.load(base + (h0 + 0) * sxh + (w0 + 4) * sxw, mask=m_0_4, other=0.0)
    m_0_5 = cmask & (h0 + 0 >= 0) & (h0 + 0 < H) & (w0 + 5 >= 0) & (w0 + 5 < W)
    d_0_5 = tl.load(base + (h0 + 0) * sxh + (w0 + 5) * sxw, mask=m_0_5, other=0.0)
    m_1_0 = cmask & (h0 + 1 >= 0) & (h0 + 1 < H) & (w0 + 0 >= 0) & (w0 + 0 < W)
    d_1_0 = tl.load(base + (h0 + 1) * sxh + (w0 + 0) * sxw, mask=m_1_0, other=0.0)
    m_1_1 = cmask & (h0 + 1 >= 0) & (h0 + 1 < H) & (w0 + 1 >= 0) & (w0 + 1 < W)
    d_1_1 = tl.load(base + (h0 + 1) * sxh + (w0 + 1) * sxw, mask=m_1_1, other=0.0)
    m_1_2 = cmask & (h0 + 1 >= 0) & (h0 + 1 < H) & (w0 + 2 >= 0) & (w0 + 2 < W)
    d_1_2 = tl.load(base + (h0 + 1) * sxh + (w0 + 2) * sxw, mask=m_1_2, other=0.0)
    m_1_3 = cmask & (h0 + 1 >= 0) & (h0 + 1 < H) & (w0 + 3 >= 0) & (w0 + 3 < W)
    d_1_3 = tl.load(base + (h0 + 1) * sxh + (w0 + 3) * sxw, mask=m_1_3, other=0.0)
    m_1_4 = cmask & (h0 + 1 >= 0) & (h0 + 1 < H) & (w0 + 4 >= 0) & (w0 + 4 < W)
    d_1_4 = tl.load(base + (h0 + 1) * sxh + (w0 + 4) * sxw, mask=m_1_4, other=0.0)
    m_1_5 = cmask & (h0 + 1 >= 0) & (h0 + 1 < H) & (w0 + 5 >= 0) & (w0 + 5 < W)
    d_1_5 = tl.load(base + (h0 + 1) * sxh + (w0 + 5) * sxw, mask=m_1_5, other=0.0)
    m_2_0 = cmask & (h0 + 2 >= 0) & (h0 + 2 < H) & (w0 + 0 >= 0) & (w0 + 0 < W)
    d_2_0 = tl.load(base + (h0 + 2) * sxh + (w0 + 0) * sxw, mask=m_2_0, other=0.0)
    m_2_1 = cmask & (h0 + 2 >= 0) & (h0 + 2 < H) & (w0 + 1 >= 0) & (w0 + 1 < W)
    d_2_1 = tl.load(base + (h0 + 2) * sxh + (w0 + 1) * sxw, mask=m_2_1, other=0.0)
    m_2_2 = cmask & (h0 + 2 >= 0) & (h0 + 2 < H) & (w0 + 2 >= 0) & (w0 + 2 < W)
    d_2_2 = tl.load(base + (h0 + 2) * sxh + (w0 + 2) * sxw, mask=m_2_2, other=0.0)
    m_2_3 = cmask & (h0 + 2 >= 0) & (h0 + 2 < H) & (w0 + 3 >= 0) & (w0 + 3 < W)
    d_2_3 = tl.load(base + (h0 + 2) * sxh + (w0 + 3) * sxw, mask=m_2_3, other=0.0)
    m_2_4 = cmask & (h0 + 2 >= 0) & (h0 + 2 < H) & (w0 + 4 >= 0) & (w0 + 4 < W)
    d_2_4 = tl.load(base + (h0 + 2) * sxh + (w0 + 4) * sxw, mask=m_2_4, other=0.0)
    m_2_5 = cmask & (h0 + 2 >= 0) & (h0 + 2 < H) & (w0 + 5 >= 0) & (w0 + 5 < W)
    d_2_5 = tl.load(base + (h0 + 2) * sxh + (w0 + 5) * sxw, mask=m_2_5, other=0.0)
    m_3_0 = cmask & (h0 + 3 >= 0) & (h0 + 3 < H) & (w0 + 0 >= 0) & (w0 + 0 < W)
    d_3_0 = tl.load(base + (h0 + 3) * sxh + (w0 + 0) * sxw, mask=m_3_0, other=0.0)
    m_3_1 = cmask & (h0 + 3 >= 0) & (h0 + 3 < H) & (w0 + 1 >= 0) & (w0 + 1 < W)
    d_3_1 = tl.load(base + (h0 + 3) * sxh + (w0 + 1) * sxw, mask=m_3_1, other=0.0)
    m_3_2 = cmask & (h0 + 3 >= 0) & (h0 + 3 < H) & (w0 + 2 >= 0) & (w0 + 2 < W)
    d_3_2 = tl.load(base + (h0 + 3) * sxh + (w0 + 2) * sxw, mask=m_3_2, other=0.0)
    m_3_3 = cmask & (h0 + 3 >= 0) & (h0 + 3 < H) & (w0 + 3 >= 0) & (w0 + 3 < W)
    d_3_3 = tl.load(base + (h0 + 3) * sxh + (w0 + 3) * sxw, mask=m_3_3, other=0.0)
    m_3_4 = cmask & (h0 + 3 >= 0) & (h0 + 3 < H) & (w0 + 4 >= 0) & (w0 + 4 < W)
    d_3_4 = tl.load(base + (h0 + 3) * sxh + (w0 + 4) * sxw, mask=m_3_4, other=0.0)
    m_3_5 = cmask & (h0 + 3 >= 0) & (h0 + 3 < H) & (w0 + 5 >= 0) & (w0 + 5 < W)
    d_3_5 = tl.load(base + (h0 + 3) * sxh + (w0 + 5) * sxw, mask=m_3_5, other=0.0)
    m_4_0 = cmask & (h0 + 4 >= 0) & (h0 + 4 < H) & (w0 + 0 >= 0) & (w0 + 0 < W)
    d_4_0 = tl.load(base + (h0 + 4) * sxh + (w0 + 0) * sxw, mask=m_4_0, other=0.0)
    m_4_1 = cmask & (h0 + 4 >= 0) & (h0 + 4 < H) & (w0 + 1 >= 0) & (w0 + 1 < W)
    d_4_1 = tl.load(base + (h0 + 4) * sxh + (w0 + 1) * sxw, mask=m_4_1, other=0.0)
    m_4_2 = cmask & (h0 + 4 >= 0) & (h0 + 4 < H) & (w0 + 2 >= 0) & (w0 + 2 < W)
    d_4_2 = tl.load(base + (h0 + 4) * sxh + (w0 + 2) * sxw, mask=m_4_2, other=0.0)
    m_4_3 = cmask & (h0 + 4 >= 0) & (h0 + 4 < H) & (w0 + 3 >= 0) & (w0 + 3 < W)
    d_4_3 = tl.load(base + (h0 + 4) * sxh + (w0 + 3) * sxw, mask=m_4_3, other=0.0)
    m_4_4 = cmask & (h0 + 4 >= 0) & (h0 + 4 < H) & (w0 + 4 >= 0) & (w0 + 4 < W)
    d_4_4 = tl.load(base + (h0 + 4) * sxh + (w0 + 4) * sxw, mask=m_4_4, other=0.0)
    m_4_5 = cmask & (h0 + 4 >= 0) & (h0 + 4 < H) & (w0 + 5 >= 0) & (w0 + 5 < W)
    d_4_5 = tl.load(base + (h0 + 4) * sxh + (w0 + 5) * sxw, mask=m_4_5, other=0.0)
    m_5_0 = cmask & (h0 + 5 >= 0) & (h0 + 5 < H) & (w0 + 0 >= 0) & (w0 + 0 < W)
    d_5_0 = tl.load(base + (h0 + 5) * sxh + (w0 + 0) * sxw, mask=m_5_0, other=0.0)
    m_5_1 = cmask & (h0 + 5 >= 0) & (h0 + 5 < H) & (w0 + 1 >= 0) & (w0 + 1 < W)
    d_5_1 = tl.load(base + (h0 + 5) * sxh + (w0 + 1) * sxw, mask=m_5_1, other=0.0)
    m_5_2 = cmask & (h0 + 5 >= 0) & (h0 + 5 < H) & (w0 + 2 >= 0) & (w0 + 2 < W)
    d_5_2 = tl.load(base + (h0 + 5) * sxh + (w0 + 2) * sxw, mask=m_5_2, other=0.0)
    m_5_3 = cmask & (h0 + 5 >= 0) & (h0 + 5 < H) & (w0 + 3 >= 0) & (w0 + 3 < W)
    d_5_3 = tl.load(base + (h0 + 5) * sxh + (w0 + 3) * sxw, mask=m_5_3, other=0.0)
    m_5_4 = cmask & (h0 + 5 >= 0) & (h0 + 5 < H) & (w0 + 4 >= 0) & (w0 + 4 < W)
    d_5_4 = tl.load(base + (h0 + 5) * sxh + (w0 + 4) * sxw, mask=m_5_4, other=0.0)
    m_5_5 = cmask & (h0 + 5 >= 0) & (h0 + 5 < H) & (w0 + 5 >= 0) & (w0 + 5 < W)
    d_5_5 = tl.load(base + (h0 + 5) * sxh + (w0 + 5) * sxw, mask=m_5_5, other=0.0)
    t_0_0 = (4.0 * d_0_0) + (-5.0 * d_2_0) + d_4_0
    t_0_1 = (4.0 * d_0_1) + (-5.0 * d_2_1) + d_4_1
    t_0_2 = (4.0 * d_0_2) + (-5.0 * d_2_2) + d_4_2
    t_0_3 = (4.0 * d_0_3) + (-5.0 * d_2_3) + d_4_3
    t_0_4 = (4.0 * d_0_4) + (-5.0 * d_2_4) + d_4_4
    t_0_5 = (4.0 * d_0_5) + (-5.0 * d_2_5) + d_4_5
    t_1_0 = (-4.0 * d_1_0) + (-4.0 * d_2_0) + d_3_0 + d_4_0
    t_1_1 = (-4.0 * d_1_1) + (-4.0 * d_2_1) + d_3_1 + d_4_1
    t_1_2 = (-4.0 * d_1_2) + (-4.0 * d_2_2) + d_3_2 + d_4_2
    t_1_3 = (-4.0 * d_1_3) + (-4.0 * d_2_3) + d_3_3 + d_4_3
    t_1_4 = (-4.0 * d_1_4) + (-4.0 * d_2_4) + d_3_4 + d_4_4
    t_1_5 = (-4.0 * d_1_5) + (-4.0 * d_2_5) + d_3_5 + d_4_5
    t_2_0 = (4.0 * d_1_0) + (-4.0 * d_2_0) + (-d_3_0) + d_4_0
    t_2_1 = (4.0 * d_1_1) + (-4.0 * d_2_1) + (-d_3_1) + d_4_1
    t_2_2 = (4.0 * d_1_2) + (-4.0 * d_2_2) + (-d_3_2) + d_4_2
    t_2_3 = (4.0 * d_1_3) + (-4.0 * d_2_3) + (-d_3_3) + d_4_3
    t_2_4 = (4.0 * d_1_4) + (-4.0 * d_2_4) + (-d_3_4) + d_4_4
    t_2_5 = (4.0 * d_1_5) + (-4.0 * d_2_5) + (-d_3_5) + d_4_5
    t_3_0 = (-2.0 * d_1_0) + (-d_2_0) + (2.0 * d_3_0) + d_4_0
    t_3_1 = (-2.0 * d_1_1) + (-d_2_1) + (2.0 * d_3_1) + d_4_1
    t_3_2 = (-2.0 * d_1_2) + (-d_2_2) + (2.0 * d_3_2) + d_4_2
    t_3_3 = (-2.0 * d_1_3) + (-d_2_3) + (2.0 * d_3_3) + d_4_3
    t_3_4 = (-2.0 * d_1_4) + (-d_2_4) + (2.0 * d_3_4) + d_4_4
    t_3_5 = (-2.0 * d_1_5) + (-d_2_5) + (2.0 * d_3_5) + d_4_5
    t_4_0 = (2.0 * d_1_0) + (-d_2_0) + (-2.0 * d_3_0) + d_4_0
    t_4_1 = (2.0 * d_1_1) + (-d_2_1) + (-2.0 * d_3_1) + d_4_1
    t_4_2 = (2.0 * d_1_2) + (-d_2_2) + (-2.0 * d_3_2) + d_4_2
    t_4_3 = (2.0 * d_1_3) + (-d_2_3) + (-2.0 * d_3_3) + d_4_3
    t_4_4 = (2.0 * d_1_4) + (-d_2_4) + (-2.0 * d_3_4) + d_4_4
    t_4_5 = (2.0 * d_1_5) + (-d_2_5) + (-2.0 * d_3_5) + d_4_5
    t_5_0 = (4.0 * d_1_0) + (-5.0 * d_3_0) + d_5_0
    t_5_1 = (4.0 * d_1_1) + (-5.0 * d_3_1) + d_5_1
    t_5_2 = (4.0 * d_1_2) + (-5.0 * d_3_2) + d_5_2
    t_5_3 = (4.0 * d_1_3) + (-5.0 * d_3_3) + d_5_3
    t_5_4 = (4.0 * d_1_4) + (-5.0 * d_3_4) + d_5_4
    t_5_5 = (4.0 * d_1_5) + (-5.0 * d_3_5) + d_5_5
    u_0_0 = (4.0 * t_0_0) + (-5.0 * t_0_2) + t_0_4
    tl.store(U + 0 * T * C + t * C + c, u_0_0, mask=cmask)
    u_0_1 = (-4.0 * t_0_1) + (-4.0 * t_0_2) + t_0_3 + t_0_4
    tl.store(U + 1 * T * C + t * C + c, u_0_1, mask=cmask)
    u_0_2 = (4.0 * t_0_1) + (-4.0 * t_0_2) + (-t_0_3) + t_0_4
    tl.store(U + 2 * T * C + t * C + c, u_0_2, mask=cmask)
    u_0_3 = (-2.0 * t_0_1) + (-t_0_2) + (2.0 * t_0_3) + t_0_4
    tl.store(U + 3 * T * C + t * C + c, u_0_3, mask=cmask)
    u_0_4 = (2.0 * t_0_1) + (-t_0_2) + (-2.0 * t_0_3) + t_0_4
    tl.store(U + 4 * T * C + t * C + c, u_0_4, mask=cmask)
    u_0_5 = (4.0 * t_0_1) + (-5.0 * t_0_3) + t_0_5
    tl.store(U + 5 * T * C + t * C + c, u_0_5, mask=cmask)
    u_1_0 = (4.0 * t_1_0) + (-5.0 * t_1_2) + t_1_4
    tl.store(U + 6 * T * C + t * C + c, u_1_0, mask=cmask)
    u_1_1 = (-4.0 * t_1_1) + (-4.0 * t_1_2) + t_1_3 + t_1_4
    tl.store(U + 7 * T * C + t * C + c, u_1_1, mask=cmask)
    u_1_2 = (4.0 * t_1_1) + (-4.0 * t_1_2) + (-t_1_3) + t_1_4
    tl.store(U + 8 * T * C + t * C + c, u_1_2, mask=cmask)
    u_1_3 = (-2.0 * t_1_1) + (-t_1_2) + (2.0 * t_1_3) + t_1_4
    tl.store(U + 9 * T * C + t * C + c, u_1_3, mask=cmask)
    u_1_4 = (2.0 * t_1_1) + (-t_1_2) + (-2.0 * t_1_3) + t_1_4
    tl.store(U + 10 * T * C + t * C + c, u_1_4, mask=cmask)
    u_1_5 = (4.0 * t_1_1) + (-5.0 * t_1_3) + t_1_5
    tl.store(U + 11 * T * C + t * C + c, u_1_5, mask=cmask)
    u_2_0 = (4.0 * t_2_0) + (-5.0 * t_2_2) + t_2_4
    tl.store(U + 12 * T * C + t * C + c, u_2_0, mask=cmask)
    u_2_1 = (-4.0 * t_2_1) + (-4.0 * t_2_2) + t_2_3 + t_2_4
    tl.store(U + 13 * T * C + t * C + c, u_2_1, mask=cmask)
    u_2_2 = (4.0 * t_2_1) + (-4.0 * t_2_2) + (-t_2_3) + t_2_4
    tl.store(U + 14 * T * C + t * C + c, u_2_2, mask=cmask)
    u_2_3 = (-2.0 * t_2_1) + (-t_2_2) + (2.0 * t_2_3) + t_2_4
    tl.store(U + 15 * T * C + t * C + c, u_2_3, mask=cmask)
    u_2_4 = (2.0 * t_2_1) + (-t_2_2) + (-2.0 * t_2_3) + t_2_4
    tl.store(U + 16 * T * C + t * C + c, u_2_4, mask=cmask)
    u_2_5 = (4.0 * t_2_1) + (-5.0 * t_2_3) + t_2_5
    tl.store(U + 17 * T * C + t * C + c, u_2_5, mask=cmask)
    u_3_0 = (4.0 * t_3_0) + (-5.0 * t_3_2) + t_3_4
    tl.store(U + 18 * T * C + t * C + c, u_3_0, mask=cmask)
    u_3_1 = (-4.0 * t_3_1) + (-4.0 * t_3_2) + t_3_3 + t_3_4
    tl.store(U + 19 * T * C + t * C + c, u_3_1, mask=cmask)
    u_3_2 = (4.0 * t_3_1) + (-4.0 * t_3_2) + (-t_3_3) + t_3_4
    tl.store(U + 20 * T * C + t * C + c, u_3_2, mask=cmask)
    u_3_3 = (-2.0 * t_3_1) + (-t_3_2) + (2.0 * t_3_3) + t_3_4
    tl.store(U + 21 * T * C + t * C + c, u_3_3, mask=cmask)
    u_3_4 = (2.0 * t_3_1) + (-t_3_2) + (-2.0 * t_3_3) + t_3_4
    tl.store(U + 22 * T * C + t * C + c, u_3_4, mask=cmask)
    u_3_5 = (4.0 * t_3_1) + (-5.0 * t_3_3) + t_3_5
    tl.store(U + 23 * T * C + t * C + c, u_3_5, mask=cmask)
    u_4_0 = (4.0 * t_4_0) + (-5.0 * t_4_2) + t_4_4
    tl.store(U + 24 * T * C + t * C + c, u_4_0, mask=cmask)
    u_4_1 = (-4.0 * t_4_1) + (-4.0 * t_4_2) + t_4_3 + t_4_4
    tl.store(U + 25 * T * C + t * C + c, u_4_1, mask=cmask)
    u_4_2 = (4.0 * t_4_1) + (-4.0 * t_4_2) + (-t_4_3) + t_4_4
    tl.store(U + 26 * T * C + t * C + c, u_4_2, mask=cmask)
    u_4_3 = (-2.0 * t_4_1) + (-t_4_2) + (2.0 * t_4_3) + t_4_4
    tl.store(U + 27 * T * C + t * C + c, u_4_3, mask=cmask)
    u_4_4 = (2.0 * t_4_1) + (-t_4_2) + (-2.0 * t_4_3) + t_4_4
    tl.store(U + 28 * T * C + t * C + c, u_4_4, mask=cmask)
    u_4_5 = (4.0 * t_4_1) + (-5.0 * t_4_3) + t_4_5
    tl.store(U + 29 * T * C + t * C + c, u_4_5, mask=cmask)
    u_5_0 = (4.0 * t_5_0) + (-5.0 * t_5_2) + t_5_4
    tl.store(U + 30 * T * C + t * C + c, u_5_0, mask=cmask)
    u_5_1 = (-4.0 * t_5_1) + (-4.0 * t_5_2) + t_5_3 + t_5_4
    tl.store(U + 31 * T * C + t * C + c, u_5_1, mask=cmask)
    u_5_2 = (4.0 * t_5_1) + (-4.0 * t_5_2) + (-t_5_3) + t_5_4
    tl.store(U + 32 * T * C + t * C + c, u_5_2, mask=cmask)
    u_5_3 = (-2.0 * t_5_1) + (-t_5_2) + (2.0 * t_5_3) + t_5_4
    tl.store(U + 33 * T * C + t * C + c, u_5_3, mask=cmask)
    u_5_4 = (2.0 * t_5_1) + (-t_5_2) + (-2.0 * t_5_3) + t_5_4
    tl.store(U + 34 * T * C + t * C + c, u_5_4, mask=cmask)
    u_5_5 = (4.0 * t_5_1) + (-5.0 * t_5_3) + t_5_5
    tl.store(U + 35 * T * C + t * C + c, u_5_5, mask=cmask)

@triton.jit
def _wino_y_kernel(M, Y, K, OH, OW, TW, THW, T, syn, syk, syh, syw,
                   BLOCK_K: tl.constexpr):
    t = tl.program_id(0)
    kb = tl.program_id(1)
    n = t // THW
    th = (t % THW) // TW
    tw = t % TW
    k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    kmask = k < K
    m_0_0 = tl.load(M + 0 * T * K + t * K + k, mask=kmask, other=0.0)
    m_0_1 = tl.load(M + 1 * T * K + t * K + k, mask=kmask, other=0.0)
    m_0_2 = tl.load(M + 2 * T * K + t * K + k, mask=kmask, other=0.0)
    m_0_3 = tl.load(M + 3 * T * K + t * K + k, mask=kmask, other=0.0)
    m_0_4 = tl.load(M + 4 * T * K + t * K + k, mask=kmask, other=0.0)
    m_0_5 = tl.load(M + 5 * T * K + t * K + k, mask=kmask, other=0.0)
    m_1_0 = tl.load(M + 6 * T * K + t * K + k, mask=kmask, other=0.0)
    m_1_1 = tl.load(M + 7 * T * K + t * K + k, mask=kmask, other=0.0)
    m_1_2 = tl.load(M + 8 * T * K + t * K + k, mask=kmask, other=0.0)
    m_1_3 = tl.load(M + 9 * T * K + t * K + k, mask=kmask, other=0.0)
    m_1_4 = tl.load(M + 10 * T * K + t * K + k, mask=kmask, other=0.0)
    m_1_5 = tl.load(M + 11 * T * K + t * K + k, mask=kmask, other=0.0)
    m_2_0 = tl.load(M + 12 * T * K + t * K + k, mask=kmask, other=0.0)
    m_2_1 = tl.load(M + 13 * T * K + t * K + k, mask=kmask, other=0.0)
    m_2_2 = tl.load(M + 14 * T * K + t * K + k, mask=kmask, other=0.0)
    m_2_3 = tl.load(M + 15 * T * K + t * K + k, mask=kmask, other=0.0)
    m_2_4 = tl.load(M + 16 * T * K + t * K + k, mask=kmask, other=0.0)
    m_2_5 = tl.load(M + 17 * T * K + t * K + k, mask=kmask, other=0.0)
    m_3_0 = tl.load(M + 18 * T * K + t * K + k, mask=kmask, other=0.0)
    m_3_1 = tl.load(M + 19 * T * K + t * K + k, mask=kmask, other=0.0)
    m_3_2 = tl.load(M + 20 * T * K + t * K + k, mask=kmask, other=0.0)
    m_3_3 = tl.load(M + 21 * T * K + t * K + k, mask=kmask, other=0.0)
    m_3_4 = tl.load(M + 22 * T * K + t * K + k, mask=kmask, other=0.0)
    m_3_5 = tl.load(M + 23 * T * K + t * K + k, mask=kmask, other=0.0)
    m_4_0 = tl.load(M + 24 * T * K + t * K + k, mask=kmask, other=0.0)
    m_4_1 = tl.load(M + 25 * T * K + t * K + k, mask=kmask, other=0.0)
    m_4_2 = tl.load(M + 26 * T * K + t * K + k, mask=kmask, other=0.0)
    m_4_3 = tl.load(M + 27 * T * K + t * K + k, mask=kmask, other=0.0)
    m_4_4 = tl.load(M + 28 * T * K + t * K + k, mask=kmask, other=0.0)
    m_4_5 = tl.load(M + 29 * T * K + t * K + k, mask=kmask, other=0.0)
    m_5_0 = tl.load(M + 30 * T * K + t * K + k, mask=kmask, other=0.0)
    m_5_1 = tl.load(M + 31 * T * K + t * K + k, mask=kmask, other=0.0)
    m_5_2 = tl.load(M + 32 * T * K + t * K + k, mask=kmask, other=0.0)
    m_5_3 = tl.load(M + 33 * T * K + t * K + k, mask=kmask, other=0.0)
    m_5_4 = tl.load(M + 34 * T * K + t * K + k, mask=kmask, other=0.0)
    m_5_5 = tl.load(M + 35 * T * K + t * K + k, mask=kmask, other=0.0)
    q_0_0 = m_0_0 + m_1_0 + m_2_0 + m_3_0 + m_4_0
    q_0_1 = m_0_1 + m_1_1 + m_2_1 + m_3_1 + m_4_1
    q_0_2 = m_0_2 + m_1_2 + m_2_2 + m_3_2 + m_4_2
    q_0_3 = m_0_3 + m_1_3 + m_2_3 + m_3_3 + m_4_3
    q_0_4 = m_0_4 + m_1_4 + m_2_4 + m_3_4 + m_4_4
    q_0_5 = m_0_5 + m_1_5 + m_2_5 + m_3_5 + m_4_5
    q_1_0 = m_1_0 + (-m_2_0) + (2.0 * m_3_0) + (-2.0 * m_4_0)
    q_1_1 = m_1_1 + (-m_2_1) + (2.0 * m_3_1) + (-2.0 * m_4_1)
    q_1_2 = m_1_2 + (-m_2_2) + (2.0 * m_3_2) + (-2.0 * m_4_2)
    q_1_3 = m_1_3 + (-m_2_3) + (2.0 * m_3_3) + (-2.0 * m_4_3)
    q_1_4 = m_1_4 + (-m_2_4) + (2.0 * m_3_4) + (-2.0 * m_4_4)
    q_1_5 = m_1_5 + (-m_2_5) + (2.0 * m_3_5) + (-2.0 * m_4_5)
    q_2_0 = m_1_0 + m_2_0 + (4.0 * m_3_0) + (4.0 * m_4_0)
    q_2_1 = m_1_1 + m_2_1 + (4.0 * m_3_1) + (4.0 * m_4_1)
    q_2_2 = m_1_2 + m_2_2 + (4.0 * m_3_2) + (4.0 * m_4_2)
    q_2_3 = m_1_3 + m_2_3 + (4.0 * m_3_3) + (4.0 * m_4_3)
    q_2_4 = m_1_4 + m_2_4 + (4.0 * m_3_4) + (4.0 * m_4_4)
    q_2_5 = m_1_5 + m_2_5 + (4.0 * m_3_5) + (4.0 * m_4_5)
    q_3_0 = m_1_0 + (-m_2_0) + (8.0 * m_3_0) + (-8.0 * m_4_0) + m_5_0
    q_3_1 = m_1_1 + (-m_2_1) + (8.0 * m_3_1) + (-8.0 * m_4_1) + m_5_1
    q_3_2 = m_1_2 + (-m_2_2) + (8.0 * m_3_2) + (-8.0 * m_4_2) + m_5_2
    q_3_3 = m_1_3 + (-m_2_3) + (8.0 * m_3_3) + (-8.0 * m_4_3) + m_5_3
    q_3_4 = m_1_4 + (-m_2_4) + (8.0 * m_3_4) + (-8.0 * m_4_4) + m_5_4
    q_3_5 = m_1_5 + (-m_2_5) + (8.0 * m_3_5) + (-8.0 * m_4_5) + m_5_5
    base = Y + n * syn + k * syk
    y_0_0 = q_0_0 + q_0_1 + q_0_2 + q_0_3 + q_0_4
    om_0_0 = kmask & (th * 4 + 0 < OH) & (tw * 4 + 0 < OW)
    tl.store(base + (th * 4 + 0) * syh + (tw * 4 + 0) * syw, y_0_0, mask=om_0_0)
    y_0_1 = q_0_1 + (-q_0_2) + (2.0 * q_0_3) + (-2.0 * q_0_4)
    om_0_1 = kmask & (th * 4 + 0 < OH) & (tw * 4 + 1 < OW)
    tl.store(base + (th * 4 + 0) * syh + (tw * 4 + 1) * syw, y_0_1, mask=om_0_1)
    y_0_2 = q_0_1 + q_0_2 + (4.0 * q_0_3) + (4.0 * q_0_4)
    om_0_2 = kmask & (th * 4 + 0 < OH) & (tw * 4 + 2 < OW)
    tl.store(base + (th * 4 + 0) * syh + (tw * 4 + 2) * syw, y_0_2, mask=om_0_2)
    y_0_3 = q_0_1 + (-q_0_2) + (8.0 * q_0_3) + (-8.0 * q_0_4) + q_0_5
    om_0_3 = kmask & (th * 4 + 0 < OH) & (tw * 4 + 3 < OW)
    tl.store(base + (th * 4 + 0) * syh + (tw * 4 + 3) * syw, y_0_3, mask=om_0_3)
    y_1_0 = q_1_0 + q_1_1 + q_1_2 + q_1_3 + q_1_4
    om_1_0 = kmask & (th * 4 + 1 < OH) & (tw * 4 + 0 < OW)
    tl.store(base + (th * 4 + 1) * syh + (tw * 4 + 0) * syw, y_1_0, mask=om_1_0)
    y_1_1 = q_1_1 + (-q_1_2) + (2.0 * q_1_3) + (-2.0 * q_1_4)
    om_1_1 = kmask & (th * 4 + 1 < OH) & (tw * 4 + 1 < OW)
    tl.store(base + (th * 4 + 1) * syh + (tw * 4 + 1) * syw, y_1_1, mask=om_1_1)
    y_1_2 = q_1_1 + q_1_2 + (4.0 * q_1_3) + (4.0 * q_1_4)
    om_1_2 = kmask & (th * 4 + 1 < OH) & (tw * 4 + 2 < OW)
    tl.store(base + (th * 4 + 1) * syh + (tw * 4 + 2) * syw, y_1_2, mask=om_1_2)
    y_1_3 = q_1_1 + (-q_1_2) + (8.0 * q_1_3) + (-8.0 * q_1_4) + q_1_5
    om_1_3 = kmask & (th * 4 + 1 < OH) & (tw * 4 + 3 < OW)
    tl.store(base + (th * 4 + 1) * syh + (tw * 4 + 3) * syw, y_1_3, mask=om_1_3)
    y_2_0 = q_2_0 + q_2_1 + q_2_2 + q_2_3 + q_2_4
    om_2_0 = kmask & (th * 4 + 2 < OH) & (tw * 4 + 0 < OW)
    tl.store(base + (th * 4 + 2) * syh + (tw * 4 + 0) * syw, y_2_0, mask=om_2_0)
    y_2_1 = q_2_1 + (-q_2_2) + (2.0 * q_2_3) + (-2.0 * q_2_4)
    om_2_1 = kmask & (th * 4 + 2 < OH) & (tw * 4 + 1 < OW)
    tl.store(base + (th * 4 + 2) * syh + (tw * 4 + 1) * syw, y_2_1, mask=om_2_1)
    y_2_2 = q_2_1 + q_2_2 + (4.0 * q_2_3) + (4.0 * q_2_4)
    om_2_2 = kmask & (th * 4 + 2 < OH) & (tw * 4 + 2 < OW)
    tl.store(base + (th * 4 + 2) * syh + (tw * 4 + 2) * syw, y_2_2, mask=om_2_2)
    y_2_3 = q_2_1 + (-q_2_2) + (8.0 * q_2_3) + (-8.0 * q_2_4) + q_2_5
    om_2_3 = kmask & (th * 4 + 2 < OH) & (tw * 4 + 3 < OW)
    tl.store(base + (th * 4 + 2) * syh + (tw * 4 + 3) * syw, y_2_3, mask=om_2_3)
    y_3_0 = q_3_0 + q_3_1 + q_3_2 + q_3_3 + q_3_4
    om_3_0 = kmask & (th * 4 + 3 < OH) & (tw * 4 + 0 < OW)
    tl.store(base + (th * 4 + 3) * syh + (tw * 4 + 0) * syw, y_3_0, mask=om_3_0)
    y_3_1 = q_3_1 + (-q_3_2) + (2.0 * q_3_3) + (-2.0 * q_3_4)
    om_3_1 = kmask & (th * 4 + 3 < OH) & (tw * 4 + 1 < OW)
    tl.store(base + (th * 4 + 3) * syh + (tw * 4 + 1) * syw, y_3_1, mask=om_3_1)
    y_3_2 = q_3_1 + q_3_2 + (4.0 * q_3_3) + (4.0 * q_3_4)
    om_3_2 = kmask & (th * 4 + 3 < OH) & (tw * 4 + 2 < OW)
    tl.store(base + (th * 4 + 3) * syh + (tw * 4 + 2) * syw, y_3_2, mask=om_3_2)
    y_3_3 = q_3_1 + (-q_3_2) + (8.0 * q_3_3) + (-8.0 * q_3_4) + q_3_5
    om_3_3 = kmask & (th * 4 + 3 < OH) & (tw * 4 + 3 < OW)
    tl.store(base + (th * 4 + 3) * syh + (tw * 4 + 3) * syw, y_3_3, mask=om_3_3)


@triton.jit
def _wino_v_kernel(W, V, C, K, swk, swc, swh, sww,
                   BLOCK_K: tl.constexpr):
    c = tl.program_id(0)
    kb = tl.program_id(1)
    k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    kmask = k < K
    base = W + k * swk + c * swc
    w_0_0 = tl.load(base + 0 * swh + 0 * sww, mask=kmask, other=0.0)
    w_0_1 = tl.load(base + 0 * swh + 1 * sww, mask=kmask, other=0.0)
    w_0_2 = tl.load(base + 0 * swh + 2 * sww, mask=kmask, other=0.0)
    w_1_0 = tl.load(base + 1 * swh + 0 * sww, mask=kmask, other=0.0)
    w_1_1 = tl.load(base + 1 * swh + 1 * sww, mask=kmask, other=0.0)
    w_1_2 = tl.load(base + 1 * swh + 2 * sww, mask=kmask, other=0.0)
    w_2_0 = tl.load(base + 2 * swh + 0 * sww, mask=kmask, other=0.0)
    w_2_1 = tl.load(base + 2 * swh + 1 * sww, mask=kmask, other=0.0)
    w_2_2 = tl.load(base + 2 * swh + 2 * sww, mask=kmask, other=0.0)
    t_0_0 = (0.25 * w_0_0)
    t_0_1 = (0.25 * w_0_1)
    t_0_2 = (0.25 * w_0_2)
    t_1_0 = (-0.16666666666666666 * w_0_0) + (-0.16666666666666666 * w_1_0) + (-0.16666666666666666 * w_2_0)
    t_1_1 = (-0.16666666666666666 * w_0_1) + (-0.16666666666666666 * w_1_1) + (-0.16666666666666666 * w_2_1)
    t_1_2 = (-0.16666666666666666 * w_0_2) + (-0.16666666666666666 * w_1_2) + (-0.16666666666666666 * w_2_2)
    t_2_0 = (-0.16666666666666666 * w_0_0) + (0.16666666666666666 * w_1_0) + (-0.16666666666666666 * w_2_0)
    t_2_1 = (-0.16666666666666666 * w_0_1) + (0.16666666666666666 * w_1_1) + (-0.16666666666666666 * w_2_1)
    t_2_2 = (-0.16666666666666666 * w_0_2) + (0.16666666666666666 * w_1_2) + (-0.16666666666666666 * w_2_2)
    t_3_0 = (0.041666666666666664 * w_0_0) + (0.08333333333333333 * w_1_0) + (0.16666666666666666 * w_2_0)
    t_3_1 = (0.041666666666666664 * w_0_1) + (0.08333333333333333 * w_1_1) + (0.16666666666666666 * w_2_1)
    t_3_2 = (0.041666666666666664 * w_0_2) + (0.08333333333333333 * w_1_2) + (0.16666666666666666 * w_2_2)
    t_4_0 = (0.041666666666666664 * w_0_0) + (-0.08333333333333333 * w_1_0) + (0.16666666666666666 * w_2_0)
    t_4_1 = (0.041666666666666664 * w_0_1) + (-0.08333333333333333 * w_1_1) + (0.16666666666666666 * w_2_1)
    t_4_2 = (0.041666666666666664 * w_0_2) + (-0.08333333333333333 * w_1_2) + (0.16666666666666666 * w_2_2)
    t_5_0 = w_2_0
    t_5_1 = w_2_1
    t_5_2 = w_2_2
    v_0_0 = (0.25 * t_0_0)
    tl.store(V + 0 * C * K + c * K + k, v_0_0, mask=kmask)
    v_0_1 = (-0.16666666666666666 * t_0_0) + (-0.16666666666666666 * t_0_1) + (-0.16666666666666666 * t_0_2)
    tl.store(V + 1 * C * K + c * K + k, v_0_1, mask=kmask)
    v_0_2 = (-0.16666666666666666 * t_0_0) + (0.16666666666666666 * t_0_1) + (-0.16666666666666666 * t_0_2)
    tl.store(V + 2 * C * K + c * K + k, v_0_2, mask=kmask)
    v_0_3 = (0.041666666666666664 * t_0_0) + (0.08333333333333333 * t_0_1) + (0.16666666666666666 * t_0_2)
    tl.store(V + 3 * C * K + c * K + k, v_0_3, mask=kmask)
    v_0_4 = (0.041666666666666664 * t_0_0) + (-0.08333333333333333 * t_0_1) + (0.16666666666666666 * t_0_2)
    tl.store(V + 4 * C * K + c * K + k, v_0_4, mask=kmask)
    v_0_5 = t_0_2
    tl.store(V + 5 * C * K + c * K + k, v_0_5, mask=kmask)
    v_1_0 = (0.25 * t_1_0)
    tl.store(V + 6 * C * K + c * K + k, v_1_0, mask=kmask)
    v_1_1 = (-0.16666666666666666 * t_1_0) + (-0.16666666666666666 * t_1_1) + (-0.16666666666666666 * t_1_2)
    tl.store(V + 7 * C * K + c * K + k, v_1_1, mask=kmask)
    v_1_2 = (-0.16666666666666666 * t_1_0) + (0.16666666666666666 * t_1_1) + (-0.16666666666666666 * t_1_2)
    tl.store(V + 8 * C * K + c * K + k, v_1_2, mask=kmask)
    v_1_3 = (0.041666666666666664 * t_1_0) + (0.08333333333333333 * t_1_1) + (0.16666666666666666 * t_1_2)
    tl.store(V + 9 * C * K + c * K + k, v_1_3, mask=kmask)
    v_1_4 = (0.041666666666666664 * t_1_0) + (-0.08333333333333333 * t_1_1) + (0.16666666666666666 * t_1_2)
    tl.store(V + 10 * C * K + c * K + k, v_1_4, mask=kmask)
    v_1_5 = t_1_2
    tl.store(V + 11 * C * K + c * K + k, v_1_5, mask=kmask)
    v_2_0 = (0.25 * t_2_0)
    tl.store(V + 12 * C * K + c * K + k, v_2_0, mask=kmask)
    v_2_1 = (-0.16666666666666666 * t_2_0) + (-0.16666666666666666 * t_2_1) + (-0.16666666666666666 * t_2_2)
    tl.store(V + 13 * C * K + c * K + k, v_2_1, mask=kmask)
    v_2_2 = (-0.16666666666666666 * t_2_0) + (0.16666666666666666 * t_2_1) + (-0.16666666666666666 * t_2_2)
    tl.store(V + 14 * C * K + c * K + k, v_2_2, mask=kmask)
    v_2_3 = (0.041666666666666664 * t_2_0) + (0.08333333333333333 * t_2_1) + (0.16666666666666666 * t_2_2)
    tl.store(V + 15 * C * K + c * K + k, v_2_3, mask=kmask)
    v_2_4 = (0.041666666666666664 * t_2_0) + (-0.08333333333333333 * t_2_1) + (0.16666666666666666 * t_2_2)
    tl.store(V + 16 * C * K + c * K + k, v_2_4, mask=kmask)
    v_2_5 = t_2_2
    tl.store(V + 17 * C * K + c * K + k, v_2_5, mask=kmask)
    v_3_0 = (0.25 * t_3_0)
    tl.store(V + 18 * C * K + c * K + k, v_3_0, mask=kmask)
    v_3_1 = (-0.16666666666666666 * t_3_0) + (-0.16666666666666666 * t_3_1) + (-0.16666666666666666 * t_3_2)
    tl.store(V + 19 * C * K + c * K + k, v_3_1, mask=kmask)
    v_3_2 = (-0.16666666666666666 * t_3_0) + (0.16666666666666666 * t_3_1) + (-0.16666666666666666 * t_3_2)
    tl.store(V + 20 * C * K + c * K + k, v_3_2, mask=kmask)
    v_3_3 = (0.041666666666666664 * t_3_0) + (0.08333333333333333 * t_3_1) + (0.16666666666666666 * t_3_2)
    tl.store(V + 21 * C * K + c * K + k, v_3_3, mask=kmask)
    v_3_4 = (0.041666666666666664 * t_3_0) + (-0.08333333333333333 * t_3_1) + (0.16666666666666666 * t_3_2)
    tl.store(V + 22 * C * K + c * K + k, v_3_4, mask=kmask)
    v_3_5 = t_3_2
    tl.store(V + 23 * C * K + c * K + k, v_3_5, mask=kmask)
    v_4_0 = (0.25 * t_4_0)
    tl.store(V + 24 * C * K + c * K + k, v_4_0, mask=kmask)
    v_4_1 = (-0.16666666666666666 * t_4_0) + (-0.16666666666666666 * t_4_1) + (-0.16666666666666666 * t_4_2)
    tl.store(V + 25 * C * K + c * K + k, v_4_1, mask=kmask)
    v_4_2 = (-0.16666666666666666 * t_4_0) + (0.16666666666666666 * t_4_1) + (-0.16666666666666666 * t_4_2)
    tl.store(V + 26 * C * K + c * K + k, v_4_2, mask=kmask)
    v_4_3 = (0.041666666666666664 * t_4_0) + (0.08333333333333333 * t_4_1) + (0.16666666666666666 * t_4_2)
    tl.store(V + 27 * C * K + c * K + k, v_4_3, mask=kmask)
    v_4_4 = (0.041666666666666664 * t_4_0) + (-0.08333333333333333 * t_4_1) + (0.16666666666666666 * t_4_2)
    tl.store(V + 28 * C * K + c * K + k, v_4_4, mask=kmask)
    v_4_5 = t_4_2
    tl.store(V + 29 * C * K + c * K + k, v_4_5, mask=kmask)
    v_5_0 = (0.25 * t_5_0)
    tl.store(V + 30 * C * K + c * K + k, v_5_0, mask=kmask)
    v_5_1 = (-0.16666666666666666 * t_5_0) + (-0.16666666666666666 * t_5_1) + (-0.16666666666666666 * t_5_2)
    tl.store(V + 31 * C * K + c * K + k, v_5_1, mask=kmask)
    v_5_2 = (-0.16666666666666666 * t_5_0) + (0.16666666666666666 * t_5_1) + (-0.16666666666666666 * t_5_2)
    tl.store(V + 32 * C * K + c * K + k, v_5_2, mask=kmask)
    v_5_3 = (0.041666666666666664 * t_5_0) + (0.08333333333333333 * t_5_1) + (0.16666666666666666 * t_5_2)
    tl.store(V + 33 * C * K + c * K + k, v_5_3, mask=kmask)
    v_5_4 = (0.041666666666666664 * t_5_0) + (-0.08333333333333333 * t_5_1) + (0.16666666666666666 * t_5_2)
    tl.store(V + 34 * C * K + c * K + k, v_5_4, mask=kmask)
    v_5_5 = t_5_2
    tl.store(V + 35 * C * K + c * K + k, v_5_5, mask=kmask)


def winograd_conv2d_fwd(x, w, *, padding, out):
    N, C, H, W = x.shape
    K = w.shape[0]
    ph, pw = padding
    OH, OW = out.shape[2], out.shape[3]
    TH, TW = (OH + 3) // 4, (OW + 3) // 4
    T = N * TH * TW

    v = torch.empty(36, C, K, device=x.device, dtype=x.dtype)
    u = torch.empty(36, T, C, device=x.device, dtype=x.dtype)
    m = torch.empty(36, T, K, device=x.device, dtype=x.dtype)

    sw = w.stride()
    sx = x.stride()
    sy = out.stride()
    _wino_v_kernel[(C, (K + 31) // 32)](
        w, v, C, K, sw[0], sw[1], sw[2], sw[3], BLOCK_K=32,
    )
    _wino_u_kernel[(T, (C + 31) // 32)](
        x, u, C, H, W, ph, pw, TW, TH * TW, T,
        sx[0], sx[1], sx[2], sx[3], BLOCK_C=32,
    )
    torch.bmm(u, v, out=m)
    _wino_y_kernel[(T, (K + 31) // 32)](
        m, out, K, OH, OW, TW, TH * TW, T,
        sy[0], sy[1], sy[2], sy[3], BLOCK_K=32,
    )
    return out
