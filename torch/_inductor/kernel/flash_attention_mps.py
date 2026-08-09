# mypy: allow-untyped-defs
"""Triton flash-attention lowering for the MPS SDPA op.

`aten._scaled_dot_product_attention_math_for_mps` returns (out, attn_weights).
Only the first is read in practice, and the second is the full S x S score
matrix, which a tiled attention never forms. The lowering therefore emits the
tiled kernel for the output and leaves the weights to the fallback, so a graph
that does read them keeps working.
"""

import sympy

import torch

from .. import config
from ..ir import FixedLayout, FlexibleLayout
from ..lowering import register_lowering
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    realize_inputs,
    SymbolicGridFn,
    TritonTemplate,
)
from ..utils import _use_autotune_backend
from .mm_common import load_kernel_template


aten = torch.ops.aten


def _aten_sdpa_mps(query, key, value, attn_mask, *, scale, is_causal, has_mask):
    # The aten op returns (out, attn_weights); the template produces only the
    # output, so the extern choice has to match that layout. When there is no
    # mask the lowering passes `query` in that slot as a placeholder, because
    # every choice must be bound to the same input-node list.
    return aten._scaled_dot_product_attention_math_for_mps.default(
        query,
        key,
        value,
        attn_mask if has_mask else None,
        0.0,
        is_causal,
        None,
        scale=scale,
    )[0]


def _aten_sdpa_mps_bwd_dk(
    query, key, value, grad_out, output, logsumexp, delta, grad_query,
    grad_value, *, scale, is_causal
):
    # Competes with the Triton template, so it has to present the same shape:
    # one returned tensor (dK) plus dQ/dV written into the mutated buffers. It
    # recomputes delta internally, which is why the precomputed one is unused.
    del delta
    dq, dk, dv = aten._scaled_dot_product_attention_flash_mps_backward.default(
        grad_out, query, key, value, output, logsumexp, is_causal, scale=scale
    )
    grad_query.copy_(dq)
    grad_value.copy_(dv)
    return dk


aten_sdpa_mps_bwd = ExternKernelChoice(
    _aten_sdpa_mps_bwd_dk,
    None,
    name="sdpa_flash_mps_backward",
    has_out_variant=False,
)


aten_sdpa_mps = ExternKernelChoice(
    _aten_sdpa_mps,
    None,
    name="sdpa_math_for_mps",
    has_out_variant=False,
)


@SymbolicGridFn
def flash_grid(batch, heads, q_len, head_dim, meta, *, cdiv):
    return (cdiv(q_len, meta["BLOCK_M"]), batch * heads, 1)


flash_template = TritonTemplate(
    name="flash_attention_mps",
    grid=flash_grid,
    source=load_kernel_template("triton_flash_attention_mps"),
)


@SymbolicGridFn
def flash_bwd_grid(batch, heads, q_len, kv_len, head_dim, meta, *, cdiv):
    # Two phases share one grid: query blocks first (dQ), then key blocks
    # (dK/dV), so each gradient is written by exactly one program and none of
    # them needs an atomic accumulate.
    return (
        cdiv(q_len, meta["BLOCK_M"]) + cdiv(kv_len, meta["BLOCK_N"]),
        batch * heads,
        1,
    )


flash_lse_template = TritonTemplate(
    name="flash_attention_mps_lse",
    grid=flash_grid,
    source=load_kernel_template("triton_flash_attention_mps_lse"),
)


flash_bwd_template = TritonTemplate(
    name="flash_attention_mps_bwd",
    grid=flash_bwd_grid,
    source=load_kernel_template("triton_flash_attention_mps_bwd"),
)


# Apple's hard threadgroup-memory cap. A dot whose two operand tiles do not
# both fit is denied in-place aliasing and falls back to a per-fragment device
# path, which measures 5-16% of the MMA ceiling against 36-43% for a tile that
# fits: a 7.7x cliff, not a gradient. Both of attention's dots have to fit,
# and the QK dot is the binding one at large head_dim.
_TG_BUDGET_BYTES = 32768


def _fits_threadgroup_budget(block_m, block_n, head_dim, itemsize=4):
    qk = (block_m * head_dim + head_dim * block_n) * itemsize
    pv = (block_m * block_n + block_n * head_dim) * itemsize
    return max(qk, pv) <= _TG_BUDGET_BYTES


def _configs(q_len, head_dim):
    out = []
    for block_m in (16, 32, 64):
        for block_n in (16, 32, 64):
            for num_warps in (4, 8):
                if block_m > q_len or block_n > q_len:
                    continue
                if not _fits_threadgroup_budget(block_m, block_n, head_dim):
                    continue
                out.append((block_m, block_n, num_warps))
    return out or [(16, 16, 4)]


def _mps_flash_supported(
    query, key, value, attn_mask, dropout_p, is_causal, dropout_mask, scale, enable_gqa
):
    if dropout_p != 0.0 or dropout_mask is not None:
        return False
    if attn_mask is not None:
        if attn_mask.get_dtype() != query.get_dtype():
            return False
        if len(attn_mask.get_size()) != 4:
            return False
    if enable_gqa:
        return False
    if any(x.get_device().type != "mps" for x in (query, key, value)):
        return False
    if len(query.get_size()) != 4:
        return False
    for x in (query, key, value):
        if x.get_dtype() not in (torch.float32, torch.float16, torch.bfloat16):
            return False
    if query.get_dtype() != key.get_dtype() or query.get_dtype() != value.get_dtype():
        return False
    q_sz = query.get_size()
    k_sz = key.get_size()
    v_sz = value.get_size()
    if q_sz[1] != k_sz[1] or q_sz[1] != v_sz[1]:
        return False
    if k_sz[2] != v_sz[2]:
        return False
    head_dim = q_sz[3]
    if head_dim != v_sz[3] or head_dim != k_sz[3]:
        return False

    dims = []
    for d in (*q_sz, *k_sz, *v_sz):
        if not isinstance(d, (int, sympy.Integer)):
            return False
        dims.append(int(d))
    head_dim = dims[3]
    # tl.arange(0, HEAD_DIM) requires a power of two, so 40/72/96/120 are
    # rejected and fall back to aten. Padding the lane range and masking the
    # tail would admit them, and was measured as not worth doing: the aten
    # choice below beats this template by 4-22% at the power-of-two dims it
    # already accepts, so admitting more dims only adds losing choices.
    if head_dim & (head_dim - 1) or head_dim > 128:
        return False
    if dims[2] % 8 or dims[6] % 8:
        return False
    return True


def _fallback(*args, **kwargs):
    from ..lowering import fallback_handler

    return fallback_handler(aten._scaled_dot_product_attention_math_for_mps.default)(
        *args, **kwargs
    )


@register_lowering(aten._scaled_dot_product_attention_math_for_mps)
def scaled_dot_product_attention_math_for_mps(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    dropout_mask=None,
    scale=None,
    enable_gqa=False,
):
    if not _mps_flash_supported(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        dropout_mask,
        scale,
        enable_gqa,
    ):
        return _fallback(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            dropout_mask,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    from ..lowering import expand, get_constant_value

    # An additive mask that folds to zero contributes nothing to the scores and
    # only costs a tile load per key block.
    if attn_mask is not None:
        const = get_constant_value(attn_mask)
        if const is not None and const.value == 0:
            attn_mask = None

    has_mask = attn_mask is not None
    B, H, q_len, head_dim = query.get_size()
    kv = key.get_size()[2]
    if has_mask:
        attn_mask = expand(attn_mask, [B, H, q_len, kv])
        query, key, value, attn_mask = realize_inputs(query, key, value, attn_mask)
    else:
        query, key, value = realize_inputs(query, key, value)
        attn_mask = query
    kv_len = key.get_size()[2]
    sm_scale = (1.0 / (head_dim**0.5)) if scale is None else scale

    layout = FixedLayout(
        query.get_device(),
        query.get_dtype(),
        [B, H, q_len, head_dim],
        FlexibleLayout.contiguous_strides([B, H, q_len, head_dim]),
    )

    # The aten kernel competes rather than only serving as a fallback for
    # shapes the template rejects. Without it the autotuner sees Triton
    # templates only, so an ATEN arm silently still runs Triton and the eager
    # kernel is never timed against ours -- and the tiled template is not
    # uniformly better: at S=1024 head_dim=80 it measures 0.797x of the
    # decomposition, so which one wins is a per-shape question that the
    # autotuner is already built to answer.
    choices = []
    if _use_autotune_backend("ATEN") or config.max_autotune or config.max_autotune_gemm:
        choices.append(
            aten_sdpa_mps.bind(
                (query, key, value, attn_mask),
                layout,
                scale=sm_scale,
                is_causal=is_causal,
                has_mask=has_mask,
            )
        )
    for block_m, block_n, num_warps in _configs(q_len, head_dim):
        flash_template.maybe_append_choice(
            choices,
            input_nodes=(query, key, value, attn_mask),
            layout=layout,
            num_stages=2,
            num_warps=num_warps,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HEAD_DIM=head_dim,
            SM_SCALE=sm_scale,
            IS_CAUSAL=is_causal,
            HAS_MASK=has_mask,
        )
    if not choices:
        return _fallback(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            dropout_mask,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    out, _ = autotune_select_algorithm(
        "flash_attention_mps", choices, [query, key, value, attn_mask], layout
    )
    return out, None


def _bwd_configs(kv_len, head_dim):
    """The autotuner consistently picks the smallest tile offered, and this
    backend favours few warps (more threadgroups, fewer barriers), so the sweep
    goes down to 1 warp rather than starting at 2."""
    out = []
    for block_m in (16, 32, 64):
        for block_n in (16, 32, 64):
            for num_warps in (1, 2, 4, 8):
                if block_n > kv_len:
                    continue
                if not _fits_threadgroup_budget(block_m, block_n, head_dim):
                    continue
                out.append((block_m, block_n, num_warps))
    return out


@register_lowering(aten._scaled_dot_product_attention_flash_mps_backward)
def scaled_dot_product_attention_flash_mps_backward(
    grad_out,
    query,
    key,
    value,
    output,
    logsumexp,
    is_causal=False,
    scale=None,
):
    """Triton backward for the MPS flash op.

    The op exists so that training has an attention node at all: sdpa on MPS
    decomposes before inductor sees it unless a derivative is registered, which
    is why there was previously nothing here to lower.
    """
    from ..lowering import empty_strided, mul, sum_ as reduce_sum, to_dtype

    B, H, q_len, head_dim = query.get_size()
    kv_len = key.get_size()[2]

    if not isinstance(head_dim, (int, sympy.Integer)) or int(head_dim) > 128:
        return _bwd_fallback(
            grad_out, query, key, value, output, logsumexp, is_causal, scale
        )
    head_dim = int(head_dim)
    # tl.arange needs a power of two, so the tail lanes are padded and masked.
    head_dim_pad = 1 << (head_dim - 1).bit_length()

    sm_scale = (1.0 / (head_dim**0.5)) if scale is None else scale

    for t in (grad_out, query, key, value, output, logsumexp):
        t.realize()

    # delta = rowsum(dO * O), the term that makes ds = P * (dP - delta) correct
    # for a softmax whose normaliser depends on every score in the row. Left to
    # the pointwise scheduler rather than folded into the template: it is a
    # single cheap reduction and fuses with its neighbours.
    delta = reduce_sum(
        mul(to_dtype(grad_out, torch.float32), to_dtype(output, torch.float32)),
        axis=-1,
    )
    delta.realize()

    layout_k = FixedLayout(
        key.get_device(),
        key.get_dtype(),
        [B, H, kv_len, head_dim],
        FlexibleLayout.contiguous_strides([B, H, kv_len, head_dim]),
    )

    grad_query = empty_strided(
        query.get_size(),
        FlexibleLayout.contiguous_strides(query.get_size()),
        dtype=query.get_dtype(),
        device=query.get_device(),
    )
    grad_value = empty_strided(
        value.get_size(),
        FlexibleLayout.contiguous_strides(value.get_size()),
        dtype=value.get_dtype(),
        device=value.get_device(),
    )
    # realize_inputs unwraps to StorageBox, which is not a valid lowering
    # return; keep the TensorBox handles and realize copies for the template.
    grad_query.realize()
    grad_value.realize()

    choices = []
    for block_m, block_n, num_warps in _bwd_configs(kv_len, head_dim_pad):
        flash_bwd_template.maybe_append_choice(
            choices,
            input_nodes=(
                query,
                key,
                value,
                grad_out,
                output,
                logsumexp,
                delta,
                grad_query,
                grad_value,
            ),
            layout=layout_k,
            mutated_inputs=[grad_query, grad_value],
            call_sizes=[B, H, q_len, kv_len, head_dim_pad],
            num_stages=2,
            num_warps=num_warps,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HEAD_DIM=head_dim_pad,
            HEAD_DIM_REAL=head_dim,
            SM_SCALE=sm_scale,
            IS_CAUSAL=is_causal,
        )

    # The ATen kernel competes rather than only serving as a fallback, so which
    # implementation runs is a per-shape autotune decision instead of one this
    # code makes.
    if _use_autotune_backend("ATEN") or config.max_autotune or config.max_autotune_gemm:
        choices.append(
            aten_sdpa_mps_bwd.bind(
                (query, key, value, grad_out, output, logsumexp, delta,
                 grad_query, grad_value),
                layout_k,
                scale=sm_scale,
                is_causal=is_causal,
            )
        )

    if not choices:
        return _bwd_fallback(
            grad_out, query, key, value, output, logsumexp, is_causal, scale
        )

    grad_key, _ = autotune_select_algorithm(
        "flash_attention_mps_bwd",
        choices,
        [query, key, value, grad_out, output, logsumexp, delta, grad_query,
         grad_value],
        layout_k,
    )
    return grad_query, grad_key, grad_value


def _bwd_fallback(grad_out, query, key, value, output, logsumexp, is_causal, scale):
    from ..lowering import fallback_handler

    return fallback_handler(
        aten._scaled_dot_product_attention_flash_mps_backward.default
    )(grad_out, query, key, value, output, logsumexp, is_causal, scale=scale)


@register_lowering(aten._scaled_dot_product_attention_flash_mps)
def scaled_dot_product_attention_flash_mps(query, key, value, is_causal=False, scale=None):
    """Triton forward for the MPS flash op, emitting the log-sum-exp.

    Without this the op falls back to its ATen implementation, whose tiling is a
    loop of matmuls: that measured 7.98 ms against 4.72 ms for the decomposition
    at DiT's shape, and was the entire reason the fused path lost overall even
    though its backward already won.
    """
    from ..lowering import empty_strided

    B, H, q_len, head_dim = query.get_size()
    kv_len = key.get_size()[2]

    # Unlike the math_for_mps lowering this one pads the lane range to the next
    # power of two and masks the tail, so head_dim 40 (SD1.5 UNet) is served
    # rather than falling back to the op's ATen implementation.
    if (
        not isinstance(head_dim, (int, sympy.Integer))
        or int(head_dim) > 128
        or any(x.get_device().type != "mps" for x in (query, key, value))
        or query.get_dtype() != torch.float32
    ):
        return _flash_fallback(query, key, value, is_causal, scale)
    head_dim = int(head_dim)
    head_dim_pad = 1 << (head_dim - 1).bit_length()

    for t in (query, key, value):
        t.realize()

    sm_scale = (1.0 / (head_dim**0.5)) if scale is None else scale

    layout = FixedLayout(
        query.get_device(),
        query.get_dtype(),
        [B, H, q_len, head_dim],
        FlexibleLayout.contiguous_strides([B, H, q_len, head_dim]),
    )
    lse = empty_strided(
        [B, H, q_len],
        FlexibleLayout.contiguous_strides([B, H, q_len]),
        dtype=torch.float32,
        device=query.get_device(),
    )
    lse.realize()

    choices = []
    for block_m, block_n, num_warps in _configs(q_len, head_dim_pad):
        flash_lse_template.maybe_append_choice(
            choices,
            input_nodes=(query, key, value, query, lse),
            layout=layout,
            mutated_inputs=[lse],
            num_stages=2,
            num_warps=num_warps,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HEAD_DIM=head_dim_pad,
            HEAD_DIM_REAL=head_dim,
            SM_SCALE=sm_scale,
            IS_CAUSAL=is_causal,
            HAS_MASK=False,
        )

    if not choices:
        return _flash_fallback(query, key, value, is_causal, scale)

    out, _ = autotune_select_algorithm(
        "flash_attention_mps_lse", choices, [query, key, value, query, lse], layout
    )
    return out, lse


def _flash_fallback(query, key, value, is_causal, scale):
    from ..lowering import fallback_handler

    return fallback_handler(aten._scaled_dot_product_attention_flash_mps.default)(
        query, key, value, is_causal, scale=scale
    )
