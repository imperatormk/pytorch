from __future__ import annotations

from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch._inductor.heuristics.registry import register_template_heuristic

from ... import config
from ...ir import get_free_symbols
from ...kernel.mm import (
    addmm_contiguous_subgraph_template,
    mm_contiguous_subgraph_template,
)
from ...kernel_inputs import KernelInputs, MMKernelInputs
from ...utils import use_contiguous
from ...virtualized import V
from .base import TemplateConfigHeuristics
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics


if TYPE_CHECKING:
    from collections.abc import Generator


@register_template_heuristic(mm_contiguous_subgraph_template.uid, None, op_name="mm")
@register_template_heuristic(
    addmm_contiguous_subgraph_template.uid, None, op_name="addmm"
)
class EmptyContiguousMMConfigHeuristics(TemplateConfigHeuristics):
    """empty heuristics to skip contiguous mm on not cuda"""


@register_template_heuristic(
    mm_contiguous_subgraph_template.uid,
    "cuda",
    register=torch.version.hip is not None,
    op_name="mm",
)
@register_template_heuristic(
    addmm_contiguous_subgraph_template.uid,
    "cuda",
    register=torch.version.hip is not None,
    op_name="addmm",
)
class ContiguousMMHeuristics(GemmMaxAutotuneTemplateConfigHeuristics):
    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Get all the valid k_splits for the given m, n, k.
        """
        if not isinstance(kernel_inputs, MMKernelInputs):
            raise AssertionError(f"{self.__class__.__name__} requires MMKernelInputs")
        # Check for unbacked symbols - if found, yield nothing
        unbacked_symbols = any(
            len(get_free_symbols(itr, unbacked_only=True)) > 0
            for itr in (
                *kernel_inputs.shapes_symbolic(),
                *kernel_inputs.strides_symbolic(),
            )
        )
        if unbacked_symbols:
            return
        mat2 = kernel_inputs.mat1mat2()[1]
        if mat2.get_layout().is_contiguous():
            # no need for contiguous decomposition
            return
        m, n, k = kernel_inputs.mnk_symbolic()
        if not use_contiguous(m, n, k):
            return
        yield {}


@register_template_heuristic(mm_contiguous_subgraph_template.uid, "mps", op_name="mm")
@register_template_heuristic(
    addmm_contiguous_subgraph_template.uid, "mps", op_name="addmm"
)
class ContiguousMMHeuristicsMPS(ContiguousMMHeuristics):
    """
    Offer the contiguous-mat2 transform for a K-major mat2 on MPS.

    Unlike use_contiguous, this does not try to predict whether copying beats
    gathering: measured on an M4 it is worth 1.22x at k=30522/n=768 and 0.90x
    at k=8192/n=768, not monotone in k, n or bytes. The autotuner decides.
    """

    def _is_k_major(self, mat2: Any) -> bool:
        sizevars = V.graph.sizevars
        strides = mat2.get_layout().stride
        return (
            len(strides) == 2
            and sizevars.statically_known_true(sympy.Eq(strides[0], 1))
            and not sizevars.statically_known_true(sympy.Eq(strides[1], 1))
        )

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        if not isinstance(kernel_inputs, MMKernelInputs):
            raise AssertionError(f"{self.__class__.__name__} requires MMKernelInputs")
        if not config.mps_densify_mat2:
            return
        # The subgraph is lowered through a nested GraphLowering.
        if V.graph.aot_mode or V.graph.cpp_wrapper:
            return
        unbacked_symbols = any(
            len(get_free_symbols(itr, unbacked_only=True)) > 0
            for itr in (
                *kernel_inputs.shapes_symbolic(),
                *kernel_inputs.strides_symbolic(),
            )
        )
        if unbacked_symbols:
            return
        mat2 = kernel_inputs.mat1mat2()[1]
        if mat2.get_layout().is_contiguous() or not self._is_k_major(mat2):
            return
        yield {}
