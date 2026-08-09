#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/mps/OperationUtils.h>
// Supplies the free operators (a - b, a * b, a / b) used on score tiles.
#include <ATen/TensorOperators.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_scaled_dot_product_attention_flash_mps_backward_native.h>
#include <ATen/ops/_scaled_dot_product_attention_flash_mps_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/maximum.h>
#include <ATen/ops/zeros_like.h>
#endif

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Attention_metallib.h>
#endif

// Tiled attention that returns the log-sum-exp alongside the output.
//
// The existing MPS sdpa entry points return (output, attention weights) and
// have no derivative, so autograd differentiates the decomposition instead and
// materialises an S x S score matrix per head. Saving the log-sum-exp is what
// lets backward reconstruct the softmax from O(S*D) state instead.
//
// Tiles are processed with the standard online-softmax rescaling so no
// S x S tensor is ever live: at UNet's shape a full one is 537 MB.

namespace {

// The forward keeps its tile loop in ATen: it is only ever two matmuls plus a
// softmax per tile, and the whole sequence fits one pass. The backward is the
// one that matters and runs as two Metal kernels (see
// kernels/FlashAttentionBackward.h), because expressing its tiling as ATen
// calls issued 192 launches against the decomposition's 87 and measured
// 0.507x -- 2.21x the launches predicting 1.97x the time.
constexpr int64_t kQueryTile = 8192;
constexpr int64_t kKeyTile = 8192;

// Must match INSTANTIATE_FLASH_BWD in kernels/FlashAttentionBackward.h.
constexpr int64_t kBlockQ = 16;
constexpr int64_t kBlockK = 16;
constexpr int64_t kThreads = 128;
constexpr int64_t kMaxHeadDim = 64;
constexpr int64_t kFusedBlockK = 32;

// Must match FlashAttnBwdParams in kernels/FlashAttentionBackward.h.
struct FlashAttnBwdParams {
  int B;
  int H;
  int D;
  int qL;
  int kL;
  float scale;
  int64_t Q_strides[3];
  int64_t K_strides[3];
  int64_t V_strides[3];
  int64_t O_strides[3];
  int64_t L_strides[3];
};

inline double sdpa_scale(const Tensor& query, std::optional<double> scale) {
  return scale.has_value() ? *scale
                           : (1.0 / std::sqrt(static_cast<double>(query.size(-1))));
}

} // namespace

std::tuple<Tensor, Tensor> _scaled_dot_product_attention_flash_mps(const Tensor& query,
                                                                  const Tensor& key,
                                                                  const Tensor& value,
                                                                  bool is_causal,
                                                                  std::optional<double> scale) {
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
              "_scaled_dot_product_attention_flash_mps expects 4D (B, H, S, D) tensors");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() && query.scalar_type() == value.scalar_type(),
              "_scaled_dot_product_attention_flash_mps requires matching dtypes");
  TORCH_CHECK(c10::isFloatingType(query.scalar_type()),
              "_scaled_dot_product_attention_flash_mps requires a floating point dtype");

  const auto q = query.contiguous();
  const auto k = key.contiguous();
  const auto v = value.contiguous();

  const int64_t B = q.size(0), H = q.size(1), S = q.size(2), D = q.size(3);
  const int64_t KV = k.size(2);
  const double s = sdpa_scale(q, scale);

  auto out = at::empty_like(q);
  auto lse = at::empty({B, H, S}, q.options().dtype(at::kFloat));

  for (int64_t m0 = 0; m0 < S; m0 += kQueryTile) {
    const int64_t m1 = std::min(m0 + kQueryTile, S);
    auto q_tile = q.slice(2, m0, m1);

    // Under causal masking a query only attends to keys at or before it, so a
    // query tile never needs key tiles past its own last row.
    const int64_t kv_end = is_causal ? std::min(m1, KV) : KV;

    Tensor acc, row_max, row_sum;
    bool first = true;

    for (int64_t n0 = 0; n0 < kv_end; n0 += kKeyTile) {
      const int64_t n1 = std::min(n0 + kKeyTile, kv_end);
      auto k_tile = k.slice(2, n0, n1);
      auto v_tile = v.slice(2, n0, n1);

      auto scores = at::matmul(q_tile, k_tile.transpose(-2, -1)).mul_(s).to(at::kFloat);
      if (is_causal) {
        auto rows = at::arange(m0, m1, scores.options().dtype(at::kLong)).unsqueeze(1);
        auto cols = at::arange(n0, n1, scores.options().dtype(at::kLong)).unsqueeze(0);
        scores.masked_fill_(rows.lt(cols), -std::numeric_limits<float>::infinity());
      }

      auto tile_max = std::get<0>(scores.max(-1, /*keepdim=*/true));
      if (first) {
        row_max = tile_max;
        auto p = (scores - row_max).exp_();
        row_sum = p.sum(-1, /*keepdim=*/true);
        acc = at::matmul(p.to(v_tile.scalar_type()), v_tile).to(at::kFloat);
        first = false;
      } else {
        auto new_max = at::maximum(row_max, tile_max);
        auto rescale = (row_max - new_max).exp_();
        auto p = (scores - new_max).exp_();
        row_sum = row_sum * rescale + p.sum(-1, /*keepdim=*/true);
        acc = acc * rescale + at::matmul(p.to(v_tile.scalar_type()), v_tile).to(at::kFloat);
        row_max = new_max;
      }
    }

    if (first) {
      // Fully masked query tile: no key is visible, so the row is defined to be
      // zero and its log-sum-exp is -inf.
      out.slice(2, m0, m1).zero_();
      lse.slice(2, m0, m1).fill_(-std::numeric_limits<float>::infinity());
      continue;
    }

    auto denom = row_sum.clamp_min(std::numeric_limits<float>::min());
    out.slice(2, m0, m1).copy_((acc / denom).to(out.scalar_type()));
    lse.slice(2, m0, m1).copy_((row_max + row_sum.log()).squeeze(-1));
  }

  return std::make_tuple(out, lse);
}

std::tuple<Tensor, Tensor, Tensor> _scaled_dot_product_attention_flash_mps_backward(
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& output,
    const Tensor& logsumexp,
    bool is_causal,
    std::optional<double> scale) {
  using namespace mps;
  const auto go = grad_out.contiguous();
  const auto q = query.contiguous();
  const auto k = key.contiguous();
  const auto v = value.contiguous();
  const auto o = output.contiguous();
  const auto lse = logsumexp.contiguous();

  const int64_t B = q.size(0), H = q.size(1), S = q.size(2), D = q.size(3);
  const int64_t KV = k.size(2);
  TORCH_CHECK(D <= kMaxHeadDim,
              "_scaled_dot_product_attention_flash_mps_backward: head_dim ",
              D,
              " exceeds ",
              kMaxHeadDim);

  // dq is accumulated atomically by the key-parallel kernel, so it starts at
  // zero; dk/dv are owned outright by one threadgroup each and are written once.
  auto dq = at::zeros_like(q);
  auto dk = at::empty_like(k);
  auto dv = at::empty_like(v);
  auto delta = at::empty_like(lse);

  FlashAttnBwdParams params{};
  params.B = static_cast<int>(B);
  params.H = static_cast<int>(H);
  params.D = static_cast<int>(D);
  params.qL = static_cast<int>(S);
  params.kL = static_cast<int>(KV);
  params.scale = static_cast<float>(sdpa_scale(q, scale));
  auto fill = [](int64_t* dst, const Tensor& t) {
    dst[0] = t.stride(0);
    dst[1] = t.stride(1);
    dst[2] = t.stride(2);
  };
  fill(params.Q_strides, q);
  fill(params.K_strides, k);
  fill(params.V_strides, v);
  fill(params.O_strides, o);
  fill(params.L_strides, lse);

  const std::string dtype = q.scalar_type() == at::kHalf ? "half" : "float";
  const std::string causal = is_causal ? "_causal" : "";
  // head_dim 40 gets a variant with a 32-wide query tile: at BD=40 the staging
  // buffers leave enough of the 32768-byte threadgroup budget for it, which
  // halves the inner pass count.
  const bool d40 = (D <= 40);
  const std::string suffix = d40 ? "_d40" : "";
  const int64_t blockQ = d40 ? 32 : kBlockQ;
  auto preprocessPSO =
      lib.getPipelineStateForFunc("flash_attn_bwd_preprocess" + suffix + "_" + dtype);
  // Key-parallel is the faster of the two backward layouts here: a
  // persistent-Q variant (flash_attn_bwd_qpar, kept in the metallib) measured
  // 541 ms against this one's 166 ms at S=4096, because holding Q resident
  // forces dK/dV to re-read Q/dO from device memory for every key block.
  // Key-parallel two-phase is the fastest correct backward layout measured:
  // a persistent-Q variant ran 541 ms and a fused single-pass variant with
  // transposed-score MMA ran 328 ms and failed 10 of 12 gradient checks,
  // against this one's 167 ms at S=4096 with 12/12 passing.
  auto bwdPSO = lib.getPipelineStateForFunc(
      "flash_attn_bwd_dkdv" + suffix + causal + "_" + dtype);

  const int64_t nQ = (S + blockQ - 1) / blockQ;
  const int64_t nK = (KV + kBlockK - 1) / kBlockK;
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^{
    @autoreleasepool {
      auto enc = stream->commandEncoder();

      [enc setComputePipelineState:preprocessPSO];
      mtl_setArgs(enc, o, go, delta, params);
      [enc dispatchThreadgroups:MTLSizeMake(nQ, 1, B * H)
          threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];

      [enc setComputePipelineState:bwdPSO];
      mtl_setArgs(enc, q, k, v, go, lse, delta, dk, dv, dq, params);
      [enc dispatchThreadgroups:MTLSizeMake(nK, 1, B * H)
          threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
    }
  });

  return std::make_tuple(dq, dk, dv);
}

} // namespace at::native
