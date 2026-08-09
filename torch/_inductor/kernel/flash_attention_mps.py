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
