#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Pool.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/Embedding.h>

#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/embedding_dense_backward_native.h>
#include <ATen/ops/embedding_renorm_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Embedding_metallib.h>
#endif

Tensor embedding_dense_backward_mps(const Tensor& grad_,
                                    const Tensor& indices,
                                    int64_t num_weights,
                                    int64_t padding_idx,
                                    bool scale_grad_by_freq) {
  using namespace at::native::mps;

  auto indices_arg = TensorArg(indices, "indices", 2);
  checkScalarTypes("embedding_backward", indices_arg, {kLong, kInt});
  auto grad_arg = TensorArg(grad_, "grad", 1);
  checkScalarTypes("embedding_backward", grad_arg, {kFloat, kHalf, kBFloat16});

  auto D = grad_.size(-1);
  const bool low_prec = grad_.scalar_type() == kHalf || grad_.scalar_type() == kBFloat16;

  auto num_indices = indices.numel();
  if (num_indices == 0 || D == 0 || num_weights == 0) {
    return at::zeros({num_weights, D}, grad_.options());
  }

  // Accumulate into float for stable precision
  auto grad_weight_acc = at::zeros({num_weights, D}, grad_.options().dtype(low_prec ? kFloat : grad_.scalar_type()));

  auto outer_ndim = indices.dim();
  TORCH_CHECK(grad_.dim() == outer_ndim + 1,
              "embedding_dense_backward_mps: grad dim (",
              grad_.dim(),
              ") must equal indices dim + 1 (",
              outer_ndim + 1,
              ")");
  TORCH_CHECK(outer_ndim < static_cast<int64_t>(c10::metal::max_ndim),
              "embedding_dense_backward_mps: indices ndim ",
              outer_ndim,
              " exceeds metal max_ndim ",
              c10::metal::max_ndim);
  TORCH_CHECK(num_indices <= std::numeric_limits<int32_t>::max() &&
                  num_indices * D <= std::numeric_limits<int32_t>::max() &&
                  num_weights * D <= std::numeric_limits<int32_t>::max(),
              "embedding_dense_backward_mps: tensor is larger than INT32_MAX");

  EmbeddingDenseBackwardParams<uint32_t> params{};
  params.outer_ndim = static_cast<uint32_t>(outer_ndim);
  for (auto d = 0; d < outer_ndim; ++d) {
    params.outer_sizes[d] = safe_downcast<uint32_t, int64_t>(indices.size(d));
    params.indices_strides[d] = safe_downcast<uint32_t, int64_t>(indices.stride(d));
    params.grad_outer_strides[d] = safe_downcast<uint32_t, int64_t>(grad_.stride(d));
  }
  params.grad_feature_stride = safe_downcast<uint32_t, int64_t>(grad_.stride(-1));
  params.feature_size = static_cast<uint32_t>(D);
  params.padding_idx = padding_idx;
  params.scale_grad_by_freq = scale_grad_by_freq;

  const auto use_32bit_offsets = at::native::canUse32BitIndexMath(grad_) && at::native::canUse32BitIndexMath(indices);
  const auto offset_suffix = use_32bit_offsets ? "32" : "64";

  Tensor counts;
  if (scale_grad_by_freq) {
    counts = at::zeros({num_weights}, indices.options().dtype(kUInt32));
  }

  auto stream = at::mps::getCurrentMPSStream();
  const auto idx_type_str = scalarToMetalTypeString(indices);
  const auto grad_type_str = scalarToMetalTypeString(grad_);

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = stream->commandEncoder();

      if (scale_grad_by_freq) {
        auto count_pso = lib.getPipelineStateForFunc(
            fmt::format("embedding_dense_backward_count_{}_{}", idx_type_str, offset_suffix));
        [computeEncoder setComputePipelineState:count_pso];
        mtl_setArgs(computeEncoder, indices, counts, params);
        mtl_dispatch1DJob(computeEncoder, count_pso, num_indices);
      }

      auto bwd_pso = lib.getPipelineStateForFunc(
          fmt::format("embedding_dense_backward_{}_{}_{}", grad_type_str, idx_type_str, offset_suffix));
      [computeEncoder setComputePipelineState:bwd_pso];
      const auto counts_buf = scale_grad_by_freq ? counts : grad_weight_acc;
      mtl_setArgs(computeEncoder, grad_, indices, counts_buf, grad_weight_acc, params);
      mtl_dispatch1DJob(computeEncoder, bwd_pso, num_indices * D);
    }
  });

  return low_prec ? grad_weight_acc.to(grad_.scalar_type()) : grad_weight_acc;
}

Tensor& embedding_renorm_mps_(Tensor& self, const Tensor& indices, double max_norm, double norm_type) {
  using namespace at::native::mps;

  auto self_arg = TensorArg(self, "self", 1);
  auto indices_arg = TensorArg(indices, "indices", 2);
  checkDim("embedding_renorm_", self_arg, 2);
  checkScalarTypes("embedding_renorm_", indices_arg, {kLong, kInt});
  checkScalarTypes("embedding_renorm_", self_arg, {kFloat, kHalf, kBFloat16});

  auto num_indices = indices.numel();
  if (num_indices == 0 || self.size(1) == 0) {
    return self;
  }
  TORCH_CHECK(num_indices <= std::numeric_limits<int32_t>::max(),
              "embedding_renorm_mps_: indices is larger than INT32_MAX");

  auto indices_contig = indices.contiguous();
  // The kernel walks a row with a plain stride, and scaling in place needs the
  // rows it writes to be the caller's.
  auto self_contig = self.is_contiguous() ? self : self.contiguous();

  EmbeddingRenormParams params{};
  params.num_indices = static_cast<uint32_t>(num_indices);
  params.feature_size = safe_downcast<uint32_t, int64_t>(self_contig.size(1));
  params.weight_row_stride = safe_downcast<uint32_t, int64_t>(self_contig.stride(0));
  params.max_norm = static_cast<float>(max_norm);
  params.norm_type = static_cast<float>(norm_type);

  auto stream = at::mps::getCurrentMPSStream();
  const auto idx_type_str = scalarToMetalTypeString(indices_contig);
  const auto weight_type_str = scalarToMetalTypeString(self_contig);

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = stream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(fmt::format("embedding_renorm_{}_{}", weight_type_str, idx_type_str));
      [computeEncoder setComputePipelineState:pso];
      mtl_setArgs(computeEncoder, self_contig, indices_contig, params);
      mtl_dispatch1DJob(computeEncoder, pso, num_indices);
    }
  });

  if (!self.is_contiguous()) {
    self.copy_(self_contig);
  }
  return self;
}

} // namespace at::native
