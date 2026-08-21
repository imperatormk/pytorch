#include <ATen/native/mps/kernels/Pooling.h>
#include <c10/metal/atomic.h>
#include <c10/metal/error.h>
#include <c10/metal/utils.h>
#include <metal_array>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

template <typename T>
struct IterBounds {
  T start;
  T end;
};

template <int32_t dim>
IterBounds<int32_t> get_input_iter_bounds(
    constant int32_t* input_sizes,
    thread int32_t (&pooling_dim_indices)[3],
    constant int32_t* kernel_size,
    constant int32_t* stride,
    constant int32_t* padding,
    constant int32_t* dilation) {
  auto d = dilation[dim];
  auto start = stride[dim] * pooling_dim_indices[dim] - padding[dim];
  auto end = min(start + kernel_size[dim] * d, input_sizes[dim]);
  auto start_correction = d * ((-start - 1 + d) / d);
  start += start < 0 ? start_correction : 0;
  return IterBounds<int32_t>{start, end};
}

// Iterates through all the input elements that this kernel needs to
// apply max to. Specialized for 3 pooling dimensions.
// TODO: Support any number of pooling dims
template <typename T>
void max_pool_3d_input_iter(
    constant T* input,
    device T* output,
    device int64_t* indices,
    constant int32_t* input_sizes,
    constant int32_t* input_strides,
    thread int32_t (&pooling_dim_indices)[3],
    constant int32_t* kernel_size,
    constant int32_t* stride,
    constant int32_t* padding,
    constant int32_t* dilation,
    bool return_indices) {
  auto bounds0 = get_input_iter_bounds<0>(
      input_sizes, pooling_dim_indices, kernel_size, stride, padding, dilation);
  auto bounds1 = get_input_iter_bounds<1>(
      input_sizes, pooling_dim_indices, kernel_size, stride, padding, dilation);
  auto bounds2 = get_input_iter_bounds<2>(
      input_sizes, pooling_dim_indices, kernel_size, stride, padding, dilation);

  auto d0 = dilation[0];
  auto d1 = dilation[1];
  auto d2 = dilation[2];

  T max_value = input
      [input_strides[0] * bounds0.start + input_strides[1] * bounds1.start +
       input_strides[2] * bounds2.start];
  auto size12 = input_sizes[1] * input_sizes[2];
  auto max_index =
      bounds0.start * size12 + bounds1.start * input_sizes[2] + bounds2.start;

  for (auto i0 = bounds0.start; i0 < bounds0.end; i0 += d0) {
    auto offset0 = input_strides[0] * i0;

    for (auto i1 = bounds1.start; i1 < bounds1.end; i1 += d1) {
      auto offset1 = input_strides[1] * i1;

      for (auto i2 = bounds2.start; i2 < bounds2.end; i2 += d2) {
        auto offset2 = input_strides[2] * i2;
        auto input_value = input[offset0 + offset1 + offset2];
        bool is_greater = input_value > max_value;

        max_value = is_greater ? input_value : max_value;

        if (return_indices) {
          auto input_index = i0 * size12 + i1 * input_sizes[2] + i2;
          max_index = is_greater ? input_index : max_index;
        }
      }
    }
  }
  *output = max_value;
  if (return_indices) {
    *indices = max_index;
  }
}

template <typename T, bool return_indices>
void max_pool_2d_input_iter(
    constant T* input,
    device T* output,
    device int64_t* indices,
    constant int32_t* input_sizes,
    constant int32_t* input_strides,
    thread int32_t (&pooling_dim_indices)[3],
    constant int32_t* kernel_size,
    constant int32_t* stride,
    constant int32_t* padding,
    constant int32_t* dilation) {
  auto bounds0 = get_input_iter_bounds<0>(
      input_sizes, pooling_dim_indices, kernel_size, stride, padding, dilation);
  auto bounds1 = get_input_iter_bounds<1>(
      input_sizes, pooling_dim_indices, kernel_size, stride, padding, dilation);

  auto d0 = dilation[0];
  auto d1 = dilation[1];

  T max_value = input
      [input_strides[0] * bounds0.start + input_strides[1] * bounds1.start];
  auto max_index = bounds0.start * input_sizes[1] + bounds1.start;

  for (auto i0 = bounds0.start; i0 < bounds0.end; i0 += d0) {
    auto offset0 = input_strides[0] * i0;

    for (auto i1 = bounds1.start; i1 < bounds1.end; i1 += d1) {
      auto offset1 = input_strides[1] * i1;

      auto input_value = input[offset0 + offset1];
      bool is_greater = input_value > max_value;

      max_value = is_greater ? input_value : max_value;

      if (return_indices) {
        auto input_index = i0 * input_sizes[1] + i1;
        max_index = is_greater ? input_index : max_index;
      }
    }
  }
  *output = max_value;
  if (return_indices) {
    *indices = max_index;
  }
}

struct PoolOffsets {
  int32_t output;
  int32_t indices;
  int32_t input_leading;

  PoolOffsets() : output(0), indices(0), input_leading(0) {}
};

// Finds the offset of the output element that a forward pass thread will
// calculate, `output[N, C, d, h, w]`. Also, find the offset of the input for
// the leading dim indices, `input[N, C]`. Optionally, keep track of the output
// pooling dimension indices, `[d, h , w]`.
// NOTE: This is templated per number of dimensions so that the compiler can
// unroll the loop, giving better performance.
template <int32_t dims>
PoolOffsets find_pool_offsets_dim_specific(
    constant int32_t* output_sizes,
    constant int32_t* output_strides,
    constant int32_t* indices_strides,
    constant int32_t* input_strides,
    int32_t pooling_dim_indices[3],
    int32_t leading_dims,
    bool return_indices,
    uint tid) {
  auto output_idx = static_cast<int32_t>(tid);
  PoolOffsets offsets;

  for (auto dim = dims - 1; dim >= 0; dim--) {
    auto dim_idx = output_idx % (output_sizes[dim]);
    offsets.output += output_strides[dim] * dim_idx;
    if (return_indices) {
      offsets.indices += indices_strides[dim] * dim_idx;
    }

    if (dim < leading_dims) {
      offsets.input_leading += input_strides[dim] * dim_idx;
    } else {
      // Keep track of pooling dimension indices of the output element, so we
      // can use them in the input iteration later on.
      if (pooling_dim_indices != nullptr) {
        pooling_dim_indices[dim - leading_dims] = dim_idx;
      }
    }
    output_idx = output_idx / output_sizes[dim];
  }

  return offsets;
}

PoolOffsets find_pool_offsets(
    constant int32_t* output_sizes,
    constant int32_t* output_strides,
    constant int32_t* indices_strides,
    constant int32_t* input_strides,
    int32_t pooling_dim_indices[3],
    int32_t dims,
    int32_t leading_dims,
    bool return_indices,
    uint tid) {
  switch (dims) {
    case 5:
      return find_pool_offsets_dim_specific<5>(
          output_sizes,
          output_strides,
          indices_strides,
          input_strides,
          pooling_dim_indices,
          leading_dims,
          return_indices,
          tid);
    case 4:
      return find_pool_offsets_dim_specific<4>(
          output_sizes,
          output_strides,
          indices_strides,
          input_strides,
          pooling_dim_indices,
          leading_dims,
          return_indices,
          tid);
    case 3:
      return find_pool_offsets_dim_specific<3>(
          output_sizes,
          output_strides,
          indices_strides,
          input_strides,
          pooling_dim_indices,
          leading_dims,
          return_indices,
          tid);
  }
  return PoolOffsets();
}

// Kernel computes one element of the output per kernel call.
template <typename T>
kernel void max_pool(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    device int64_t* indices [[buffer(2)]],
    constant PoolingParams<5>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  bool return_indices = params.return_indices;
  auto pooling_dims = params.pooling_dims;
  auto dims = params.dims;
  auto input_sizes = params.input_sizes.data();
  auto input_strides = params.input_strides.data();
  auto output_sizes = params.output_sizes.data();
  auto output_strides = params.output_strides.data();
  auto indices_strides = params.indices_strides.data();
  auto kernel_size = params.kernel_size.data();
  auto stride = params.stride.data();
  auto padding = params.padding.data();
  auto dilation = params.dilation.data();

  auto leading_dims = dims - pooling_dims;

  // This buffer keeps track of the pooling dimension indices of this thread's
  // element of the output. We need to fill it with the proper values below.
  int32_t pooling_dim_indices[3];

  PoolOffsets offsets = find_pool_offsets(
      output_sizes,
      output_strides,
      return_indices ? indices_strides : nullptr,
      input_strides,
      pooling_dim_indices,
      dims,
      leading_dims,
      return_indices,
      tid);

  output += offsets.output;
  indices += offsets.indices;
  input += offsets.input_leading;

  switch (pooling_dims) {
    case 2:
      if (return_indices) {
        return max_pool_2d_input_iter<T, /*return_indices=*/true>(
            input,
            output,
            indices,
            input_sizes + leading_dims,
            input_strides + leading_dims,
            pooling_dim_indices,
            kernel_size,
            stride,
            padding,
            dilation);
      } else {
        return max_pool_2d_input_iter<T, /*return_indices=*/false>(
            input,
            output,
            indices,
            input_sizes + leading_dims,
            input_strides + leading_dims,
            pooling_dim_indices,
            kernel_size,
            stride,
            padding,
            dilation);
      }
    case 3:
      return max_pool_3d_input_iter<T>(
          input,
          output,
          indices,
          input_sizes + leading_dims,
          input_strides + leading_dims,
          pooling_dim_indices,
          kernel_size,
          stride,
          padding,
          dilation,
          return_indices);
  }
}

// Finds the element in the grad input which corresponds to the index into the
// pool, and then adds the grad output element to it.
template <typename T>
void max_pool_backward_impl(
    device AtomicType_t<T>* grad_input,
    T grad_output_element,
    int32_t input_index,
    constant int32_t* grad_input_sizes,
    constant int32_t* grad_input_strides,
    int32_t grad_input_leading_offset,
    int32_t pooling_dims) {
  int32_t size_prod = 1;
  int32_t pool_offset = 0;

  for (auto dim = pooling_dims - 1; dim >= 0; dim--) {
    auto next_size_prod = grad_input_sizes[dim] * size_prod;
    pool_offset +=
        grad_input_strides[dim] * ((input_index % next_size_prod) / size_prod);
    size_prod *= grad_input_sizes[dim];
  }

  AtomicType<T>::atomic_add(
      grad_input, grad_input_leading_offset + pool_offset, grad_output_element);
}

// Kernel computes one element of the grad input per kernel call.
template <typename T>
kernel void max_pool_backward(
    device AtomicType_t<T>* grad_input [[buffer(0)]],
    constant T* grad_output [[buffer(1)]],
    constant int64_t* indices [[buffer(2)]],
    constant PoolingBackwardParams<5>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  auto pooling_dims = params.pooling_dims;
  auto dims = params.dims;
  auto grad_input_sizes = params.grad_input_sizes.data();
  auto grad_input_strides = params.grad_input_strides.data();
  auto grad_output_sizes = params.grad_output_sizes.data();
  auto grad_output_strides = params.grad_output_strides.data();
  auto indices_strides = params.indices_strides.data();

  auto leading_dims = dims - pooling_dims;

  PoolOffsets offsets = find_pool_offsets(
      grad_output_sizes,
      grad_output_strides,
      indices_strides,
      grad_input_strides,
      nullptr,
      dims,
      leading_dims,
      /*return_indices=*/true,
      tid);

  max_pool_backward_impl<T>(
      grad_input,
      grad_output[offsets.output],
      indices[offsets.indices],
      grad_input_sizes + leading_dims,
      grad_input_strides + leading_dims,
      offsets.input_leading,
      pooling_dims);
}

template <typename T>
void max_unpool_impl(
    device T* output,
    T input_element,
    int32_t input_index,
    constant int32_t* output_sizes,
    constant int32_t* output_strides,
    int32_t pooling_dims,
    device c10::metal::ErrorMessages* error_buffer) {
  int32_t size_prod = 1;
  int32_t pool_offset = 0;

  for (auto dim = pooling_dims - 1; dim >= 0; dim--) {
    auto next_size_prod = output_sizes[dim] * size_prod;
    pool_offset +=
        output_strides[dim] * ((input_index % next_size_prod) / size_prod);
    size_prod *= output_sizes[dim];
  }

  // Check that the index is within the valid output range
  if (input_index < 0 || input_index >= size_prod) {
    TORCH_REPORT_ERROR(
        error_buffer,
        "Found an invalid max index: ",
        input_index,
        " (size_prod is ",
        size_prod,
        ")");
    return;
  }

  output[pool_offset] = input_element;
}

// Kernel computes one element of the grad input per kernel call.
template <typename T>
kernel void max_unpool(
    device T* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant int64_t* indices [[buffer(2)]],
    constant MaxUnpoolingParams<5>& params [[buffer(3)]],
    device c10::metal::ErrorMessages* error_buffer [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  auto pooling_dims = params.pooling_dims;
  auto dims = params.dims;
  auto input_sizes = params.input_sizes.data();
  auto input_strides = params.input_strides.data();
  auto output_sizes = params.output_sizes.data();
  auto output_strides = params.output_strides.data();
  auto indices_strides = params.indices_strides.data();

  auto leading_dims = dims - pooling_dims;

  // NOTE: Since we're doing unpooling, the variable names "input" and "output"
  // are reversed compared to the pooling operations. So in `find_pool_offsets`,
  // we need to map "input" -> "output" and "output" -> "input".
  PoolOffsets offsets = find_pool_offsets(
      /*output_sizes=*/input_sizes,
      /*output_strides=*/input_strides,
      indices_strides,
      /*input_strides=*/output_strides,
      /*pooling_dim_indices=*/nullptr,
      dims,
      leading_dims,
      /*return_indices=*/true,
      tid);

  max_unpool_impl<T>(
      output + offsets.input_leading,
      input[offsets.output],
      indices[offsets.indices],
      output_sizes + leading_dims,
      output_strides + leading_dims,
      pooling_dims,
      error_buffer);
}

template <typename T>
struct AvgPoolIterBounds {
  T start;
  T end;
  T count;
};

template <int32_t dim>
AvgPoolIterBounds<int32_t> get_avg_pool_input_iter_bounds(
    constant int32_t* input_sizes,
    thread int32_t (&pooling_dim_indices)[3],
    constant int32_t* kernel_size,
    constant int32_t* stride,
    constant int32_t* padding,
    bool count_include_pad) {
  auto start = stride[dim] * pooling_dim_indices[dim] - padding[dim];
  auto end = start + kernel_size[dim];
  auto end_corrected = min(start + kernel_size[dim], input_sizes[dim]);
  auto start_corrected = (start < 0) ? 0 : start;
  auto count = count_include_pad
      ? (min(end, input_sizes[dim] + padding[dim]) - start)
      : (end_corrected - start_corrected);
  return {start_corrected, end_corrected, count};
}

// Iterates through all the input elements that this kernel needs to
// apply max to. Specialized for 3 pooling dimensions.
template <typename T>
void avg_pool_3d_input_iter(
    constant T* input,
    device T* output,
    constant int32_t* input_sizes,
    constant int32_t* input_strides,
    thread int32_t (&pooling_dim_indices)[3],
    constant int32_t* kernel_size,
    constant int32_t* stride,
    constant int32_t* padding,
    bool count_include_pad,
    bool has_divisor_override,
    int32_t divisor_override) {
  auto bounds0 = get_avg_pool_input_iter_bounds<0>(
      input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad);
  auto bounds1 = get_avg_pool_input_iter_bounds<1>(
      input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad);
  auto bounds2 = get_avg_pool_input_iter_bounds<2>(
      input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad);

  opmath_t<T> value_sum = 0;
  opmath_t<T> divisor = has_divisor_override
      ? divisor_override
      : (bounds0.count) * (bounds1.count) * (bounds2.count);

  for (auto i0 = bounds0.start; i0 < bounds0.end; i0++) {
    auto offset0 = input_strides[0] * i0;

    for (auto i1 = bounds1.start; i1 < bounds1.end; i1++) {
      auto offset1 = input_strides[1] * i1;

      for (auto i2 = bounds2.start; i2 < bounds2.end; i2++) {
        auto offset2 = input_strides[2] * i2;
        auto input_value = input[offset0 + offset1 + offset2];
        value_sum += static_cast<opmath_t<T>>(input_value);
      }
    }
  }
  *output = static_cast<T>(value_sum / divisor);
}

// Iterates through all the input elements that this kernel needs to
// apply max to. Specialized for 2 pooling dimensions.
template <typename T>
void avg_pool_2d_input_iter(
    constant T* input,
    device T* output,
    constant int32_t* input_sizes,
    constant int32_t* input_strides,
    thread int32_t (&pooling_dim_indices)[3],
    constant int32_t* kernel_size,
    constant int32_t* stride,
    constant int32_t* padding,
    bool count_include_pad,
    bool has_divisor_override,
    int32_t divisor_override) {
  auto bounds0 = get_avg_pool_input_iter_bounds<0>(
      input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad);
  auto bounds1 = get_avg_pool_input_iter_bounds<1>(
      input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad);

  opmath_t<T> value_sum = 0;
  opmath_t<T> divisor = has_divisor_override
      ? divisor_override
      : (bounds0.count) * (bounds1.count);

  for (auto i0 = bounds0.start; i0 < bounds0.end; i0++) {
    auto offset0 = input_strides[0] * i0;

    for (auto i1 = bounds1.start; i1 < bounds1.end; i1++) {
      auto offset1 = input_strides[1] * i1;
      auto input_value = input[offset0 + offset1];
      value_sum += static_cast<opmath_t<T>>(input_value);
    }
  }
  *output = static_cast<T>(value_sum / divisor);
}

template <typename T>
void avg_pool_backward_3d_input_iter(
    device AtomicType_t<T>* grad_input,
    constant T* grad_output,
    constant int32_t* grad_input_sizes,
    constant int32_t* grad_input_strides,
    int32_t grad_input_leading_offset,
    thread int32_t (&pooling_dim_indices)[3],
    constant int32_t* kernel_size,
    constant int32_t* stride,
    constant int32_t* padding,
    bool count_include_pad,
    bool has_divisor_override,
    int32_t divisor_override) {
  auto bounds0 = get_avg_pool_input_iter_bounds<0>(
      grad_input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad);
  auto bounds1 = get_avg_pool_input_iter_bounds<1>(
      grad_input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad);
  auto bounds2 = get_avg_pool_input_iter_bounds<2>(
      grad_input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad);

  auto divisor = has_divisor_override
      ? divisor_override
      : (bounds0.count) * (bounds1.count) * (bounds2.count);
  auto grad_val = *grad_output / static_cast<T>(divisor);

  for (auto i0 = bounds0.start; i0 < bounds0.end; i0++) {
    auto offset0 = grad_input_strides[0] * i0;

    for (auto i1 = bounds1.start; i1 < bounds1.end; i1++) {
      auto offset1 = grad_input_strides[1] * i1;

      for (auto i2 = bounds2.start; i2 < bounds2.end; i2++) {
        auto offset2 = grad_input_strides[2] * i2;
        auto pool_offset = offset0 + offset1 + offset2;

        AtomicType<T>::atomic_add(
            grad_input, grad_input_leading_offset + pool_offset, grad_val);
      }
    }
  }
}

// Kernel computes one element of the output per kernel call.
template <typename T>
kernel void avg_pool(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant AvgPoolingParams<5>& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  auto pooling_dims = params.pooling_dims;
  auto dims = params.dims;
  auto input_sizes = params.input_sizes.data();
  auto input_strides = params.input_strides.data();
  auto output_sizes = params.output_sizes.data();
  auto output_strides = params.output_strides.data();
  auto kernel_size = params.kernel_size.data();
  auto stride = params.stride.data();
  auto padding = params.padding.data();
  auto leading_dims = dims - pooling_dims;

  // This buffer keeps track of the pooling dimension indices of this thread's
  // element of the output. We need to fill it with the proper values below.
  int32_t pooling_dim_indices[3];

  PoolOffsets offsets = find_pool_offsets(
      output_sizes,
      output_strides,
      /*indices_strides=*/nullptr,
      input_strides,
      pooling_dim_indices,
      dims,
      leading_dims,
      /*return_indices=*/false,
      tid);

  output += offsets.output;
  input += offsets.input_leading;
  input_sizes += leading_dims;
  input_strides += leading_dims;

  if (pooling_dims == 3) {
    avg_pool_3d_input_iter<T>(
        input,
        output,
        input_sizes,
        input_strides,
        pooling_dim_indices,
        kernel_size,
        stride,
        padding,
        params.count_include_pad,
        params.has_divisor_override,
        params.divisor_override);
  } else if (pooling_dims == 2) {
    avg_pool_2d_input_iter<T>(
        input,
        output,
        input_sizes,
        input_strides,
        pooling_dim_indices,
        kernel_size,
        stride,
        padding,
        params.count_include_pad,
        params.has_divisor_override,
        params.divisor_override);
  }
}

// Adaptive pooling bin bounds. Unlike a strided pool, each output bin covers
// [floor(i*I/O), ceil((i+1)*I/O)), so bins vary in size whenever O does not
// divide I -- which is exactly the case a single stride cannot express.
inline int32_t adaptive_start(int32_t out_idx, int32_t in_size, int32_t out_size) {
  return (out_idx * in_size) / out_size;
}

inline int32_t adaptive_end(int32_t out_idx, int32_t in_size, int32_t out_size) {
  return ((out_idx + 1) * in_size + out_size - 1) / out_size;
}

// One thread per output element.
template <typename T>
kernel void adaptive_avg_pool(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant AdaptiveAvgPoolingParams<5>& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  auto pooling_dims = params.pooling_dims;
  auto dims = params.dims;
  auto input_sizes = params.input_sizes.data();
  auto input_strides = params.input_strides.data();
  auto output_sizes = params.output_sizes.data();
  auto output_strides = params.output_strides.data();
  auto leading_dims = dims - pooling_dims;

  int32_t pooling_dim_indices[3];

  PoolOffsets offsets = find_pool_offsets(
      output_sizes,
      output_strides,
      /*indices_strides=*/nullptr,
      input_strides,
      pooling_dim_indices,
      dims,
      leading_dims,
      /*return_indices=*/false,
      tid);

  output += offsets.output;
  input += offsets.input_leading;
  input_sizes += leading_dims;
  input_strides += leading_dims;

  int32_t start[3];
  int32_t end[3];
  int32_t count = 1;

  for (auto dim = 0; dim < pooling_dims; dim++) {
    start[dim] =
        adaptive_start(pooling_dim_indices[dim], input_sizes[dim], output_sizes[leading_dims + dim]);
    end[dim] =
        adaptive_end(pooling_dim_indices[dim], input_sizes[dim], output_sizes[leading_dims + dim]);
    count *= (end[dim] - start[dim]);
  }

  auto sum = static_cast<float>(0);

  if (pooling_dims == 1) {
    for (auto i0 = start[0]; i0 < end[0]; i0++) {
      sum += static_cast<float>(input[input_strides[0] * i0]);
    }
  } else if (pooling_dims == 2) {
    for (auto i0 = start[0]; i0 < end[0]; i0++) {
      auto offset0 = input_strides[0] * i0;
      for (auto i1 = start[1]; i1 < end[1]; i1++) {
        sum += static_cast<float>(input[offset0 + input_strides[1] * i1]);
      }
    }
  } else {
    for (auto i0 = start[0]; i0 < end[0]; i0++) {
      auto offset0 = input_strides[0] * i0;
      for (auto i1 = start[1]; i1 < end[1]; i1++) {
        auto offset1 = offset0 + input_strides[1] * i1;
        for (auto i2 = start[2]; i2 < end[2]; i2++) {
          sum += static_cast<float>(input[offset1 + input_strides[2] * i2]);
        }
      }
    }
  }

  *output = static_cast<T>(sum / static_cast<float>(count));
}

// One thread per output element. Indices are flattened over the pooling dims of
// the input, matching the CPU kernel, and ties keep the first element scanned.
template <typename T>
kernel void adaptive_max_pool(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    device int64_t* indices [[buffer(2)]],
    constant AdaptiveMaxPoolingParams<5>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  auto pooling_dims = params.pooling_dims;
  auto dims = params.dims;
  auto input_sizes = params.input_sizes.data();
  auto input_strides = params.input_strides.data();
  auto output_sizes = params.output_sizes.data();
  auto output_strides = params.output_strides.data();
  auto indices_strides = params.indices_strides.data();
  auto leading_dims = dims - pooling_dims;

  int32_t pooling_dim_indices[3];

  PoolOffsets offsets = find_pool_offsets(
      output_sizes,
      output_strides,
      indices_strides,
      input_strides,
      pooling_dim_indices,
      dims,
      leading_dims,
      /*return_indices=*/true,
      tid);

  output += offsets.output;
  indices += offsets.indices;
  input += offsets.input_leading;
  input_sizes += leading_dims;
  input_strides += leading_dims;

  int32_t start[3];
  int32_t end[3];

  for (auto dim = 0; dim < pooling_dims; dim++) {
    start[dim] = adaptive_start(
        pooling_dim_indices[dim], input_sizes[dim], output_sizes[leading_dims + dim]);
    end[dim] = adaptive_end(
        pooling_dim_indices[dim], input_sizes[dim], output_sizes[leading_dims + dim]);
  }

  // Adaptive bins are never empty, so the first element is a valid seed.
  T max_value;
  int32_t max_index;

  if (pooling_dims == 1) {
    max_value = input[input_strides[0] * start[0]];
    max_index = start[0];

    for (auto i0 = start[0]; i0 < end[0]; i0++) {
      auto input_value = input[input_strides[0] * i0];
      bool is_greater = input_value > max_value;

      max_value = is_greater ? input_value : max_value;
      max_index = is_greater ? i0 : max_index;
    }
  } else if (pooling_dims == 2) {
    auto size1 = input_sizes[1];
    max_value = input[input_strides[0] * start[0] + input_strides[1] * start[1]];
    max_index = start[0] * size1 + start[1];

    for (auto i0 = start[0]; i0 < end[0]; i0++) {
      auto offset0 = input_strides[0] * i0;

      for (auto i1 = start[1]; i1 < end[1]; i1++) {
        auto input_value = input[offset0 + input_strides[1] * i1];
        bool is_greater = input_value > max_value;

        max_value = is_greater ? input_value : max_value;
        max_index = is_greater ? (i0 * size1 + i1) : max_index;
      }
    }
  } else {
    auto size2 = input_sizes[2];
    auto size12 = input_sizes[1] * size2;
    max_value = input
        [input_strides[0] * start[0] + input_strides[1] * start[1] +
         input_strides[2] * start[2]];
    max_index = start[0] * size12 + start[1] * size2 + start[2];

    for (auto i0 = start[0]; i0 < end[0]; i0++) {
      auto offset0 = input_strides[0] * i0;

      for (auto i1 = start[1]; i1 < end[1]; i1++) {
        auto offset1 = offset0 + input_strides[1] * i1;

        for (auto i2 = start[2]; i2 < end[2]; i2++) {
          auto input_value = input[offset1 + input_strides[2] * i2];
          bool is_greater = input_value > max_value;

          max_value = is_greater ? input_value : max_value;
          max_index =
              is_greater ? (i0 * size12 + i1 * size2 + i2) : max_index;
        }
      }
    }
  }

  *output = max_value;
  *indices = max_index;
}

template <typename T>
kernel void avg_pool_backward(
    device AtomicType_t<T>* grad_input [[buffer(0)]],
    constant T* grad_output [[buffer(1)]],
    constant AvgPoolingParams<5>& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  auto pooling_dims = params.pooling_dims;
  auto dims = params.dims;
  auto grad_input_sizes = params.input_sizes.data();
  auto grad_input_strides = params.input_strides.data();
  auto grad_output_sizes = params.output_sizes.data();
  auto grad_output_strides = params.output_strides.data();
  auto kernel_size = params.kernel_size.data();
  auto stride = params.stride.data();
  auto padding = params.padding.data();
  auto leading_dims = dims - pooling_dims;

  // This buffer keeps track of the pooling dimension indices of this thread's
  // element of the output. We need to fill it with the proper values below.
  int32_t pooling_dim_indices[3];

  PoolOffsets offsets = find_pool_offsets(
      grad_output_sizes,
      grad_output_strides,
      /*indices_strides=*/nullptr,
      grad_input_strides,
      pooling_dim_indices,
      dims,
      leading_dims,
      /*return_indices=*/false,
      tid);

  grad_output += offsets.output;
  grad_input_sizes += leading_dims;
  grad_input_strides += leading_dims;

  avg_pool_backward_3d_input_iter<T>(
      grad_input,
      grad_output,
      grad_input_sizes,
      grad_input_strides,
      offsets.input_leading,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      params.count_include_pad,
      params.has_divisor_override,
      params.divisor_override);
}

// One thread per grad_output element; each scatters its share into the bin it
// came from. Bins overlap when O does not divide I, hence the atomic add.
//
// `A` is the accumulator's type and is not always `T`. The half atomic is a
// CAS loop that rounds to half on every iteration, so an input cell inside
// several overlapping bins gets an answer that depends on the order the adds
// land in -- not even reproducible run to run. The host widens the
// accumulator to float for the narrow types and narrows once at the end.
template <typename T, typename A>
kernel void adaptive_avg_pool_backward(
    device AtomicType_t<A>* grad_input [[buffer(0)]],
    constant T* grad_output [[buffer(1)]],
    constant AdaptiveAvgPoolingParams<5>& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  auto pooling_dims = params.pooling_dims;
  auto dims = params.dims;
  auto grad_input_sizes = params.input_sizes.data();
  auto grad_input_strides = params.input_strides.data();
  auto grad_output_sizes = params.output_sizes.data();
  auto grad_output_strides = params.output_strides.data();
  auto leading_dims = dims - pooling_dims;

  int32_t pooling_dim_indices[3];

  PoolOffsets offsets = find_pool_offsets(
      grad_output_sizes,
      grad_output_strides,
      /*indices_strides=*/nullptr,
      grad_input_strides,
      pooling_dim_indices,
      dims,
      leading_dims,
      /*return_indices=*/false,
      tid);

  grad_output += offsets.output;
  grad_input_sizes += leading_dims;
  grad_input_strides += leading_dims;

  int32_t start[3];
  int32_t end[3];

  // Divided ONCE PER DIMENSION, in the element's own type, because that is
  // what the CPU kernel does (`grad_output / kh / kw`, AdaptiveAvgPoolKernel
  // .cpp:290). Dividing by the product instead rounds once where CPU rounds
  // twice, and at half precision the two disagree by a ULP whenever the
  // window area is not a power of two -- a 3x3 window is enough.
  A grad_val = static_cast<A>(*grad_output);

  for (auto dim = 0; dim < pooling_dims; dim++) {
    start[dim] = adaptive_start(
        pooling_dim_indices[dim],
        grad_input_sizes[dim],
        grad_output_sizes[leading_dims + dim]);
    end[dim] = adaptive_end(
        pooling_dim_indices[dim],
        grad_input_sizes[dim],
        grad_output_sizes[leading_dims + dim]);
    grad_val /= static_cast<A>(end[dim] - start[dim]);
  }

  if (pooling_dims == 1) {
    for (auto i0 = start[0]; i0 < end[0]; i0++) {
      AtomicType<A>::atomic_add(
          grad_input, offsets.input_leading + grad_input_strides[0] * i0, grad_val);
    }
  } else if (pooling_dims == 2) {
    for (auto i0 = start[0]; i0 < end[0]; i0++) {
      auto offset0 = grad_input_strides[0] * i0;
      for (auto i1 = start[1]; i1 < end[1]; i1++) {
        AtomicType<A>::atomic_add(
            grad_input,
            offsets.input_leading + offset0 + grad_input_strides[1] * i1,
            grad_val);
      }
    }
  } else {
    for (auto i0 = start[0]; i0 < end[0]; i0++) {
      auto offset0 = grad_input_strides[0] * i0;
      for (auto i1 = start[1]; i1 < end[1]; i1++) {
        auto offset1 = offset0 + grad_input_strides[1] * i1;
        for (auto i2 = start[2]; i2 < end[2]; i2++) {
          AtomicType<A>::atomic_add(
              grad_input,
              offsets.input_leading + offset1 + grad_input_strides[2] * i2,
              grad_val);
        }
      }
    }
  }
}

// Start of the `i`th pooling interval along one dimension, matching
// at::native::generate_intervals. Every output element recomputes the one
// interval it needs rather than materializing the whole sequence.
template <typename T>
int32_t fractional_pool_start(
    T sample,
    int32_t input_size,
    int32_t output_size,
    int32_t pool_size,
    int32_t i) {
  if (i == output_size - 1) {
    return input_size - pool_size;
  }
  T alpha = static_cast<T>(input_size - pool_size) /
      static_cast<T>(output_size - 1);
  return static_cast<int32_t>((static_cast<T>(i) + sample) * alpha) -
      static_cast<int32_t>(sample * alpha);
}

template <typename T>
kernel void fractional_max_pool2d(
    constant T* input [[buffer(0)]],
    constant T* random_samples [[buffer(1)]],
    device T* output [[buffer(2)]],
    device int64_t* indices [[buffer(3)]],
    constant FractionalMaxPoolingParams<5>& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  auto dims = params.dims;
  auto pooling_dims = params.pooling_dims;
  auto input_sizes = params.input_sizes.data();
  auto input_strides = params.input_strides.data();
  auto output_sizes = params.output_sizes.data();
  auto output_strides = params.output_strides.data();
  auto indices_strides = params.indices_strides.data();
  auto pool_size = params.pool_size.data();

  auto leading_dims = dims - pooling_dims;

  int32_t pooling_dim_indices[3];
  PoolOffsets offsets = find_pool_offsets(
      output_sizes,
      output_strides,
      indices_strides,
      input_strides,
      pooling_dim_indices,
      dims,
      leading_dims,
      /*return_indices=*/true,
      tid);

  // Two samples per plane, one per pooling dimension. A plane is one flattened
  // leading-dim coordinate: for (N, C, H, W) that is n * C + c, and for an
  // unbatched (C, H, W) just c. tid enumerates the output in row-major order,
  // so dividing out the pooled extents leaves exactly that flattened index.
  int32_t pooled_numel = 1;
  for (auto dim = leading_dims; dim < dims; dim++) {
    pooled_numel *= output_sizes[dim];
  }
  int32_t plane = static_cast<int32_t>(tid) / pooled_numel;

  auto in_sizes = input_sizes + leading_dims;
  auto in_strides = input_strides + leading_dims;

  int32_t start_h = fractional_pool_start<T>(
      random_samples[2 * plane + 1],
      in_sizes[0],
      output_sizes[leading_dims],
      pool_size[0],
      pooling_dim_indices[0]);
  int32_t start_w = fractional_pool_start<T>(
      random_samples[2 * plane],
      in_sizes[1],
      output_sizes[leading_dims + 1],
      pool_size[1],
      pooling_dim_indices[1]);

  input += offsets.input_leading;

  T max_val = -::metal::numeric_limits<T>::infinity();
  int64_t max_index = static_cast<int64_t>(start_h) * in_sizes[1] + start_w;

  for (auto h = start_h; h < start_h + pool_size[0]; h++) {
    for (auto w = start_w; w < start_w + pool_size[1]; w++) {
      T val = input[h * in_strides[0] + w * in_strides[1]];
      if (val > max_val || ::metal::isnan(static_cast<float>(val))) {
        max_val = val;
        max_index = static_cast<int64_t>(h) * in_sizes[1] + w;
      }
    }
  }

  output[offsets.output] = max_val;
  indices[offsets.indices] = max_index;
}

#define REGISTER_FRACTIONAL_POOL_OP(DTYPE)                     \
  template [[host_name("fractional_max_pool2d_" #DTYPE)]]      \
  kernel void fractional_max_pool2d<DTYPE>(                    \
      constant DTYPE * input [[buffer(0)]],                    \
      constant DTYPE * random_samples [[buffer(1)]],           \
      device DTYPE * output [[buffer(2)]],                     \
      device int64_t* indices [[buffer(3)]],                   \
      constant FractionalMaxPoolingParams<5>& params           \
      [[buffer(4)]],                                           \
      uint tid [[thread_position_in_grid]]);

REGISTER_FRACTIONAL_POOL_OP(float);
REGISTER_FRACTIONAL_POOL_OP(half);
REGISTER_FRACTIONAL_POOL_OP(bfloat);

#define REGISTER_POOL_OP(DTYPE)                                               \
  template [[host_name("max_pool_" #DTYPE)]] kernel void max_pool<DTYPE>(     \
      constant DTYPE * input [[buffer(0)]],                                   \
      device DTYPE * output [[buffer(1)]],                                    \
      device int64_t* indices [[buffer(2)]],                                  \
      constant PoolingParams<5>& params [[buffer(3)]],                        \
      uint tid [[thread_position_in_grid]]);                                  \
                                                                              \
  template [[host_name("max_unpool_" #DTYPE)]] kernel void max_unpool<DTYPE>( \
      device DTYPE * output [[buffer(0)]],                                    \
      constant DTYPE * input [[buffer(1)]],                                   \
      constant int64_t* indices [[buffer(2)]],                                \
      constant MaxUnpoolingParams<5>& params [[buffer(3)]],                   \
      device ::c10::metal::ErrorMessages* error_buffer [[buffer(4)]],         \
      uint tid [[thread_position_in_grid]]);                                  \
                                                                              \
  template [[host_name("avg_pool_" #DTYPE)]] kernel void avg_pool<DTYPE>(     \
      constant DTYPE * input [[buffer(0)]],                                   \
      device DTYPE * output [[buffer(1)]],                                    \
      constant AvgPoolingParams<5> & params [[buffer(2)]],                    \
      uint tid [[thread_position_in_grid]]);                                  \
                                                                              \
  template [[host_name("adaptive_avg_pool_" #DTYPE)]]                         \
  kernel void adaptive_avg_pool<DTYPE>(                                       \
      constant DTYPE * input [[buffer(0)]],                                   \
      device DTYPE * output [[buffer(1)]],                                    \
      constant AdaptiveAvgPoolingParams<5> & params [[buffer(2)]],            \
      uint tid [[thread_position_in_grid]]);                                  \
                                                                              \
  template [[host_name("adaptive_max_pool_" #DTYPE)]]                         \
  kernel void adaptive_max_pool<DTYPE>(                                       \
      constant DTYPE * input [[buffer(0)]],                                   \
      device DTYPE * output [[buffer(1)]],                                    \
      device int64_t* indices [[buffer(2)]],                                  \
      constant AdaptiveMaxPoolingParams<5> & params [[buffer(3)]],            \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_POOL_BACKWARD_OP(DTYPE)                       \
  template [[host_name("max_pool_backward_" #DTYPE)]]          \
  kernel void max_pool_backward<DTYPE>(                        \
      device AtomicType_t<DTYPE> * grad_input [[buffer(0)]],   \
      constant DTYPE * grad_output_ [[buffer(1)]],             \
      constant int64_t* grad_indices_ [[buffer(2)]],           \
      constant PoolingBackwardParams<5>& params [[buffer(3)]], \
      uint tid [[thread_position_in_grid]]);                   \
                                                               \
  template [[host_name("avg_pool_backward_" #DTYPE)]]          \
  kernel void avg_pool_backward<DTYPE>(                        \
      device AtomicType_t<DTYPE> * grad_input [[buffer(0)]],   \
      constant DTYPE * grad_output [[buffer(1)]],              \
      constant AvgPoolingParams<5> & params [[buffer(2)]],     \
      uint tid [[thread_position_in_grid]]);                   \
                                                               \
  template [[host_name("adaptive_avg_pool_backward_" #DTYPE)]] \
  kernel void adaptive_avg_pool_backward<DTYPE, DTYPE>(        \
      device AtomicType_t<DTYPE> * grad_input [[buffer(0)]],   \
      constant DTYPE * grad_output [[buffer(1)]],              \
      constant AdaptiveAvgPoolingParams<5> & params            \
      [[buffer(2)]],                                           \
      uint tid [[thread_position_in_grid]]);

REGISTER_POOL_OP(float);
REGISTER_POOL_OP(half);
REGISTER_POOL_OP(bfloat);
REGISTER_POOL_OP(int);
REGISTER_POOL_OP(long);
REGISTER_POOL_OP(short);
REGISTER_POOL_OP(char);
REGISTER_POOL_OP(uchar);
REGISTER_POOL_OP(bool);

REGISTER_POOL_BACKWARD_OP(float);
REGISTER_POOL_BACKWARD_OP(half);
REGISTER_POOL_BACKWARD_OP(bfloat);

// The float-accumulating adaptive backward, for the narrow types. Named by the
// GRAD_OUTPUT type because that is what the host has to select on; the `f32`
// suffix says where the sum is kept.
#define REGISTER_ADAPTIVE_AVG_POOL_BACKWARD_F32_ACC(DTYPE)             \
  template [[host_name("adaptive_avg_pool_backward_" #DTYPE "_f32")]]  \
  kernel void adaptive_avg_pool_backward<DTYPE, float>(                \
      device AtomicType_t<float> * grad_input [[buffer(0)]],           \
      constant DTYPE * grad_output [[buffer(1)]],                      \
      constant AdaptiveAvgPoolingParams<5> & params [[buffer(2)]],     \
      uint tid [[thread_position_in_grid]]);

REGISTER_ADAPTIVE_AVG_POOL_BACKWARD_F32_ACC(half);
REGISTER_ADAPTIVE_AVG_POOL_BACKWARD_F32_ACC(bfloat);
