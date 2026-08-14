#include <ATen/native/mps/kernels/Embedding.h>
#include <c10/metal/atomic.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

template <typename O>
static inline void resolve_outer_offsets(
    uint tid,
    constant EmbeddingDenseBackwardParams<uint32_t>& params,
    thread O& grad_offset,
    thread O& indices_offset) {
  grad_offset = 0;
  indices_offset = 0;
  uint remaining = tid;
  uint nd = params.outer_ndim;
  for (uint d = 0; d < nd; ++d) {
    uint dim_idx = nd - 1 - d;
    uint dim_size = params.outer_sizes[dim_idx];
    uint coord = remaining % dim_size;
    remaining /= dim_size;
    grad_offset += static_cast<O>(coord) *
        static_cast<O>(params.grad_outer_strides[dim_idx]);
    indices_offset +=
        static_cast<O>(coord) * static_cast<O>(params.indices_strides[dim_idx]);
  }
}

template <typename I, typename O>
kernel void embedding_dense_backward_count(
    constant I* indices [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],
    constant EmbeddingDenseBackwardParams<uint32_t>& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  O grad_offset_unused = 0;
  O indices_offset = 0;
  resolve_outer_offsets<O>(tid, params, grad_offset_unused, indices_offset);
  long weight_idx = static_cast<long>(indices[indices_offset]);
  if (weight_idx == params.padding_idx) {
    return;
  }
  atomic_fetch_add_explicit(counts + weight_idx, 1u, memory_order_relaxed);
}

template <typename T, typename I, typename O>
kernel void embedding_dense_backward(
    constant T* grad [[buffer(0)]],
    constant I* indices [[buffer(1)]],
    constant uint* counts [[buffer(2)]],
    device AtomicType_t<float>* grad_weight [[buffer(3)]],
    constant EmbeddingDenseBackwardParams<uint32_t>& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  uint feature_size = params.feature_size;
  uint index_pos = tid / feature_size;
  uint feature_idx = tid % feature_size;

  O grad_outer_offset = 0;
  O indices_offset = 0;
  resolve_outer_offsets<O>(
      index_pos, params, grad_outer_offset, indices_offset);

  long weight_idx = static_cast<long>(indices[indices_offset]);
  if (weight_idx == params.padding_idx) {
    return;
  }

  O grad_offset = grad_outer_offset +
      static_cast<O>(feature_idx) * static_cast<O>(params.grad_feature_stride);
  float grad_val = static_cast<float>(grad[grad_offset]);
  if (params.scale_grad_by_freq) {
    uint c = counts[weight_idx];
    if (c == 0) {
      return;
    }
    grad_val /= static_cast<float>(c);
  }

  AtomicType<float>::atomic_add(
      grad_weight,
      static_cast<long>(weight_idx) * feature_size + feature_idx,
      grad_val);
}

// One thread per index. Duplicated indices would each scale the same row, so
// only the first occurrence of a value acts -- the CPU kernel gets the same
// effect by sorting and skipping equal neighbours.
template <typename T, typename I>
kernel void embedding_renorm(
    device T* weight [[buffer(0)]],
    constant I* indices [[buffer(1)]],
    constant EmbeddingRenormParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  long row = static_cast<long>(indices[tid]);
  for (uint i = 0; i < tid; ++i) {
    if (static_cast<long>(indices[i]) == row) {
      return;
    }
  }
  device T* r = weight + row * static_cast<long>(params.weight_row_stride);
  float acc = 0.0;
  for (uint j = 0; j < params.feature_size; ++j) {
    acc += ::metal::pow(::metal::fabs(static_cast<float>(r[j])), params.norm_type);
  }
  float norm = ::metal::pow(acc, 1.0f / params.norm_type);
  if (norm <= params.max_norm) {
    return;
  }
  float scale = params.max_norm / (norm + 1e-7f);
  for (uint j = 0; j < params.feature_size; ++j) {
    r[j] = static_cast<T>(static_cast<float>(r[j]) * scale);
  }
}

#define REGISTER_EMBEDDING_RENORM(T, I)                       \
  template [[host_name("embedding_renorm_" #T "_" #I)]]       \
  kernel void embedding_renorm<T, I>(                         \
      device T * weight [[buffer(0)]],                        \
      constant I * indices [[buffer(1)]],                     \
      constant EmbeddingRenormParams & params [[buffer(2)]],  \
      uint tid [[thread_position_in_grid]])

REGISTER_EMBEDDING_RENORM(float, int);
REGISTER_EMBEDDING_RENORM(float, long);
REGISTER_EMBEDDING_RENORM(half, int);
REGISTER_EMBEDDING_RENORM(half, long);
REGISTER_EMBEDDING_RENORM(bfloat, int);
REGISTER_EMBEDDING_RENORM(bfloat, long);

#define REGISTER_EMBEDDING_DENSE_BACKWARD_COUNT(I, O, OSUFFIX)                \
  template [[host_name("embedding_dense_backward_count_" #I "_" #OSUFFIX)]]   \
  kernel void embedding_dense_backward_count<I, O>(                           \
      constant I * indices [[buffer(0)]],                                     \
      device atomic_uint * counts [[buffer(1)]],                              \
      constant EmbeddingDenseBackwardParams<uint32_t> & params [[buffer(2)]], \
      uint tid [[thread_position_in_grid]])

#define REGISTER_EMBEDDING_DENSE_BACKWARD(T, I, O, OSUFFIX)                   \
  template [[host_name("embedding_dense_backward_" #T "_" #I "_" #OSUFFIX)]]  \
  kernel void embedding_dense_backward<T, I, O>(                              \
      constant T * grad [[buffer(0)]],                                        \
      constant I * indices [[buffer(1)]],                                     \
      constant uint * counts [[buffer(2)]],                                   \
      device AtomicType_t<float> * grad_weight [[buffer(3)]],                 \
      constant EmbeddingDenseBackwardParams<uint32_t> & params [[buffer(4)]], \
      uint tid [[thread_position_in_grid]])

REGISTER_EMBEDDING_DENSE_BACKWARD_COUNT(int, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD_COUNT(int, ulong, 64);
REGISTER_EMBEDDING_DENSE_BACKWARD_COUNT(long, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD_COUNT(long, ulong, 64);

REGISTER_EMBEDDING_DENSE_BACKWARD(float, int, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD(float, int, ulong, 64);
REGISTER_EMBEDDING_DENSE_BACKWARD(float, long, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD(float, long, ulong, 64);
REGISTER_EMBEDDING_DENSE_BACKWARD(half, int, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD(half, int, ulong, 64);
REGISTER_EMBEDDING_DENSE_BACKWARD(half, long, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD(half, long, ulong, 64);
REGISTER_EMBEDDING_DENSE_BACKWARD(bfloat, int, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD(bfloat, int, ulong, 64);
REGISTER_EMBEDDING_DENSE_BACKWARD(bfloat, long, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD(bfloat, long, ulong, 64);
