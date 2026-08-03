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

from ..ir import FixedLayout, FlexibleLayout
from ..lowering import register_lowering
from ..select_algorithm import (
    autotune_select_algorithm,
    realize_inputs,
    SymbolicGridFn,
    TritonTemplate,
)
from .mm_common import load_kernel_template


aten = torch.ops.aten


@SymbolicGridFn
def flash_grid(batch, heads, q_len, head_dim, meta, *, cdiv):
    return (cdiv(q_len, meta["BLOCK_M"]), batch * heads, 1)


flash_template = TritonTemplate(
    name="flash_attention_mps",
    grid=flash_grid,
    source=load_kernel_template("triton_flash_attention_mps"),
)


def _configs(q_len, head_dim):
    out = []
    for block_m in (32, 64):
        for block_n in (32, 64):
            for num_warps in (4, 8):
                if block_m > q_len or block_n > q_len:
                    continue
                out.append((block_m, block_n, num_warps))
    return out or [(32, 32, 4)]


def _mps_flash_supported(query, key, value, attn_mask, dropout_p, is_causal,
                         dropout_mask, scale, enable_gqa):
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
    from ..virtualized import V

    dims = []
    for d in (*q_sz, *k_sz, *v_sz):
        if not isinstance(d, (int, sympy.Integer)):
            return False
        dims.append(int(d))
    head_dim = dims[3]
    if head_dim % 8 or head_dim > 128:
        return False
    if dims[2] % 8 or dims[6] % 8:
        return False
    return True


def _fallback(*args, **kwargs):
    from ..lowering import fallback_handler

    return fallback_handler(
        aten._scaled_dot_product_attention_math_for_mps.default
    )(*args, **kwargs)


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
        query, key, value, attn_mask, dropout_p, is_causal, dropout_mask,
        scale, enable_gqa,
    ):
        return _fallback(
            query, key, value, attn_mask, dropout_p, is_causal, dropout_mask,
            scale=scale, enable_gqa=enable_gqa,
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
        query, key, value, attn_mask = realize_inputs(
            query, key, value, attn_mask
        )
    else:
        query, key, value = realize_inputs(query, key, value)
        attn_mask = query
    kv_len = key.get_size()[2]
    sm_scale = (1.0 / (head_dim ** 0.5)) if scale is None else scale

    layout = FixedLayout(
        query.get_device(),
        query.get_dtype(),
        [B, H, q_len, head_dim],
        FlexibleLayout.contiguous_strides([B, H, q_len, head_dim]),
    )

    choices = []
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
            query, key, value, attn_mask, dropout_p, is_causal, dropout_mask,
            scale=scale, enable_gqa=enable_gqa,
        )

    out, _ = autotune_select_algorithm(
        "flash_attention_mps", choices, [query, key, value, attn_mask], layout
    )
    return out, None
