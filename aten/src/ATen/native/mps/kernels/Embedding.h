#pragma once
#include <c10/metal/common.h>

struct EmbeddingRenormParams {
  uint32_t num_indices;
  uint32_t feature_size;
  uint32_t weight_row_stride;
  float max_norm;
  float norm_type;
};

template <typename idx_type_t = uint32_t>
struct EmbeddingDenseBackwardParams {
  ::c10::metal::array<idx_type_t, ::c10::metal::max_ndim> outer_sizes;
  ::c10::metal::array<idx_type_t, ::c10::metal::max_ndim> grad_outer_strides;
  ::c10::metal::array<idx_type_t, ::c10::metal::max_ndim> indices_strides;
  idx_type_t grad_feature_stride;
  idx_type_t outer_ndim;
  idx_type_t feature_size;
  int64_t padding_idx;
  bool scale_grad_by_freq;
};
