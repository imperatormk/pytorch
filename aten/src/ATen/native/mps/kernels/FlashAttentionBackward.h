// Flash-attention backward for MPS.
//
// Recomputes the softmax from the log-sum-exp saved by the forward instead of
// reading back an S x S score matrix: at S=4096 with 8 heads that matrix is
// 537 MB, which the decomposition autograd otherwise writes and re-reads twice.
//
// Parallel over KEY blocks: dK and dV accumulate across the whole query loop and
// are stored once, dQ is accumulated with atomic_float adds because a query row
// is touched by every key block. Metal has native float atomics, so no 64-bit
// atomic is involved.
//
// Every matrix product goes through the simdgroup MMA units via the MMATile /
// tile_matmad machinery in PrefillAttention.h. Two earlier versions of this
// kernel are the reason that is spelled out: a fully scalar one measured 0.097x
// of the decomposition, and a threadgroup-staged one with scalar inner products
// measured 0.331x at 4.4% of the MMA peak against the decomposition's 13.7%.
//
// Included from Attention.metal AFTER PrefillAttention.h - no top-level
// includes or `using` here so the includer controls them.
#pragma once

#include <c10/metal/common.h>

struct FlashAttnBwdParams {
  int B;
  int H;
  int D;
  int qL;
  int kL;
  float scale;
  // Strides (B, H, L); the last dim is contiguous.
  int64_t Q_strides[3];
  int64_t K_strides[3];
  int64_t V_strides[3];
  int64_t O_strides[3];
  int64_t L_strides[3]; // logsumexp / delta, (B, H, L)
};

// delta = rowsum(dO * O), the term that makes ds = P * (dP - delta) correct for
// a softmax whose normaliser depends on every score in its row.
template <typename T, int BQ, int NTHREADS>
[[kernel]] void flash_attn_bwd_preprocess(
    const device T* O [[buffer(0)]],
    const device T* DO [[buffer(1)]],
    device float* Delta [[buffer(2)]],
    const constant FlashAttnBwdParams* params [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  const int D = params->D;
  const int qL = params->qL;
  const int H = params->H;
  const int b = tid.z / H;
  const int h = tid.z % H;
  const int q0 = tid.x * BQ;

  const device T* o_base =
      O + b * params->O_strides[0] + h * params->O_strides[1];
  const device T* do_base =
      DO + b * params->O_strides[0] + h * params->O_strides[1];
  device float* d_base =
      Delta + b * params->L_strides[0] + h * params->L_strides[1];

  constexpr int kLanes = 32;
  constexpr int kSubgroups = NTHREADS / kLanes;
  const int lane = lid.x % kLanes;
  const int sub = lid.x / kLanes;

  for (int i = sub; i < BQ; i += kSubgroups) {
    const int q = q0 + i;
    if (q >= qL) {
      break;
    }
    const device T* o_row = o_base + q * params->O_strides[2];
    const device T* do_row = do_base + q * params->O_strides[2];
    float acc = 0.0f;
    for (int d = lane; d < D; d += kLanes) {
      acc += static_cast<float>(o_row[d]) * static_cast<float>(do_row[d]);
    }
    acc = metal::simd_sum(acc);
    if (lane == 0) {
      d_base[q] = acc;
    }
  }
}

// One threadgroup per key block, WM*WN simdgroups inside it, all four matrix
// products on the MMA units:
//   S  = Q  Kt     (recompute the scores)
//   dP = dO Vt
//   dV += Pt  dO
//   dK += dSt Q
//   dQ += dS  K     (atomic, other key blocks own the same query rows)
template <typename T, int BQ, int BK, int BD, int WM, int WN, bool DO_CAUSAL>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
flash_attn_bwd_dkdv(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    const device T* DO [[buffer(3)]],
    const device float* Lse [[buffer(4)]],
    const device float* Delta [[buffer(5)]],
    device T* DK [[buffer(6)]],
    device T* DV [[buffer(7)]],
    device atomic_float* DQ [[buffer(8)]],
    const constant FlashAttnBwdParams* params [[buffer(9)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  (void)lid;
  using AccumType = float;

  const int D = params->D;
  const int qL = params->qL;
  const int kL = params->kL;
  const int H = params->H;
  const float scale = params->scale;
  const int b = tid.z / H;
  const int h = tid.z % H;
  const int k0 = tid.x * BK;

  const device T* q_base =
      Q + b * params->Q_strides[0] + h * params->Q_strides[1];
  const device T* k_base =
      K + b * params->K_strides[0] + h * params->K_strides[1];
  const device T* v_base =
      V + b * params->V_strides[0] + h * params->V_strides[1];
  const device T* do_base =
      DO + b * params->O_strides[0] + h * params->O_strides[1];
  const device float* lse_base =
      Lse + b * params->L_strides[0] + h * params->L_strides[1];
  const device float* delta_base =
      Delta + b * params->L_strides[0] + h * params->L_strides[1];
  device T* dk_base = DK + b * params->K_strides[0] + h * params->K_strides[1];
  device T* dv_base = DV + b * params->V_strides[0] + h * params->V_strides[1];
  device atomic_float* dq_base =
      DQ + b * params->Q_strides[0] + h * params->Q_strides[1];

  constexpr short kFrag = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFrag, kFrag>;
  constexpr int kNWarps = WM * WN;
  constexpr int kNThreads = kNWarps * 32;

  constexpr int TK = BK / kFrag; // key fragments per tile
  constexpr int TD = BD / kFrag; // head-dim fragments per tile
  constexpr int TQ = BQ / kFrag; // query fragments per tile

  // Padded to keep 8-wide fragment loads off the same threadgroup banks.
  constexpr int LD = BD + kFrag;
  constexpr int LK = BK + kFrag;

  threadgroup AccumType Ks[BK * LD];
  threadgroup AccumType Vs[BK * LD];
  threadgroup AccumType Qs[BQ * LD];
  threadgroup AccumType dOs[BQ * LD];
  threadgroup AccumType Ps[BQ * LK];  // P transposed view is read as [k][q]
  threadgroup AccumType dSs[BQ * LK];
  // One dQ staging tile. Metal has no threadgroup float atomic, so the key
  // warps add into it in turn, separated by barriers -- BK/kFrag short rounds
  // per query block, which is far cheaper than the device traffic it avoids.
  threadgroup AccumType dQs[BQ * LD];
  threadgroup float Ls[BQ];
  threadgroup float Ds[BQ];

  const int tidx = int(simd_group_id) * 32 + int(simd_lane_id);

  for (int i = tidx; i < BK * LD; i += kNThreads) {
    Ks[i] = 0;
    Vs[i] = 0;
  }
  for (int i = tidx; i < BQ * LK; i += kNThreads) {
    Ps[i] = 0;
    dSs[i] = 0;
  }
  for (int i = tidx; i < BK * BD; i += kNThreads) {
    const int r = i / BD;
    const int c = i % BD;
    const int kk = k0 + r;
    if (kk < kL && c < D) {
      Ks[r * LD + c] = static_cast<AccumType>(k_base[kk * params->K_strides[2] + c]);
      Vs[r * LD + c] = static_cast<AccumType>(v_base[kk * params->V_strides[2] + c]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // dK/dV accumulate over the whole query loop. Each warp owns one key
  // fragment row-band and the full head dim.
  MMATile<AccumType, 1, TD, MMAFrag_acc_t> dKtile;
  MMATile<AccumType, 1, TD, MMAFrag_acc_t> dVtile;
  dKtile.clear();
  dVtile.clear();

  const short2 sc = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = sc.y;
  const short sn = sc.x;

  const int q_start = DO_CAUSAL ? (k0 / BQ) * BQ : 0;

  for (int q0 = q_start; q0 < qL; q0 += BQ) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tidx; i < BQ * LD; i += kNThreads) {
      Qs[i] = 0;
      dOs[i] = 0;
    }
    for (int i = tidx; i < BQ * BD; i += kNThreads) {
      const int r = i / BD;
      const int c = i % BD;
      const int q = q0 + r;
      if (q < qL && c < D) {
        Qs[r * LD + c] =
            static_cast<AccumType>(q_base[q * params->Q_strides[2] + c]);
        dOs[r * LD + c] =
            static_cast<AccumType>(do_base[q * params->O_strides[2] + c]);
      }
    }
    for (int i = tidx; i < BQ; i += kNThreads) {
      const int q = q0 + i;
      Ls[i] = (q < qL) ? lse_base[q] : INFINITY;
      Ds[i] = (q < qL) ? delta_base[q] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S = Q Kt and dP = dO Vt, both on the MMA units. Warp w owns query
    // fragment band w, so each warp computes BQ/kNWarps rows of S.
    if (simd_group_id < BQ / kFrag) {
      MMATile<AccumType, 1, 1, MMAFrag_acc_t> Qfrag;
      MMATile<AccumType, 1, 1, MMAFrag_acc_t> dOfrag;
      MMATile<AccumType, 1, TK, MMAFrag_acc_t> Kfrag;
      MMATile<AccumType, 1, TK, MMAFrag_acc_t> Vfrag;
      MMATile<AccumType, 1, TK, MMAFrag_acc_t> Stile;
      MMATile<AccumType, 1, TK, MMAFrag_acc_t> dPtile;
      Stile.clear();
      dPtile.clear();

      const short qband = short(simd_group_id) * kFrag;

      for (short dd = 0; dd < TD; ++dd) {
        simdgroup_barrier(mem_flags::mem_none);
        Qfrag.template load<AccumType, 1, 1, LD, 1>(
            &Qs[(qband + sm) * LD + sn + dd * kFrag]);
        dOfrag.template load<AccumType, 1, 1, LD, 1>(
            &dOs[(qband + sm) * LD + sn + dd * kFrag]);
        // K and V are [key][dim]; loading with row stride LD and column stride
        // 1 from &Ks[sm*LD + dd*kFrag] gives the transposed operand the MMA
        // wants for Q Kt.
        Kfrag.template load<AccumType, 1, 1, 1, LD>(
            &Ks[(sn)*LD + sm + dd * kFrag]);
        Vfrag.template load<AccumType, 1, 1, 1, LD>(
            &Vs[(sn)*LD + sm + dd * kFrag]);
        simdgroup_barrier(mem_flags::mem_none);
        tile_matmad(Stile, Qfrag, Kfrag, Stile);
        tile_matmad(dPtile, dOfrag, Vfrag, dPtile);
      }

      // P and dS elementwise from the saved lse/delta, then published so the
      // dV/dK/dQ products can read them as plain tiles.
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < TK; ++j) {
        PREFILL_PRAGMA_UNROLL
        for (short e = 0; e < MMAFrag_acc_t::kElemsPerFrag; ++e) {
          const short er = e / MMAFrag_acc_t::kElemCols;
          const short ec = e % MMAFrag_acc_t::kElemCols;
          const int qi = qband + sm + er;
          const int ki = j * kFrag + sn + ec;
          const int q = q0 + qi;
          const int kk = k0 + ki;
          float p = 0.0f;
          float ds = 0.0f;
          if (q < qL && kk < kL && !(DO_CAUSAL && q < kk)) {
            const float lse = Ls[qi];
            if (!metal::isinf(lse)) {
              p = metal::exp(Stile.frag_at(0, j)[e] * scale - lse);
              ds = p * (dPtile.frag_at(0, j)[e] - Ds[qi]) * scale;
            }
          }
          Ps[qi * LK + ki] = p;
          dSs[qi * LK + ki] = ds;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dV += Pt dO and dK += dSt Q. Warp w owns key fragment band w.
    if (simd_group_id < BK / kFrag) {
      const short kband = short(simd_group_id) * kFrag;
      MMATile<AccumType, 1, 1, MMAFrag_acc_t> Pfrag;
      MMATile<AccumType, 1, 1, MMAFrag_acc_t> dSfrag;
      MMATile<AccumType, 1, TD, MMAFrag_acc_t> dOtile;
      MMATile<AccumType, 1, TD, MMAFrag_acc_t> Qtile;

      for (short qq = 0; qq < TQ; ++qq) {
        simdgroup_barrier(mem_flags::mem_none);
        // Transposed load: [q][k] read with row stride 1, col stride LK.
        Pfrag.template load<AccumType, 1, 1, 1, LK>(
            &Ps[(qq * kFrag + sn) * LK + kband + sm]);
        dSfrag.template load<AccumType, 1, 1, 1, LK>(
            &dSs[(qq * kFrag + sn) * LK + kband + sm]);
        dOtile.template load<AccumType, 1, 1, LD, 1>(
            &dOs[(qq * kFrag + sm) * LD + sn]);
        Qtile.template load<AccumType, 1, 1, LD, 1>(
            &Qs[(qq * kFrag + sm) * LD + sn]);
        simdgroup_barrier(mem_flags::mem_none);
        tile_matmad(dVtile, Pfrag, dOtile, dVtile);
        tile_matmad(dKtile, dSfrag, Qtile, dKtile);
      }
    }

    // dQ += dS K on the MMA units, staged through threadgroup memory so the
    // atomic traffic is one add per (query, dim) element rather than one per
    // multiply-accumulate. Warp w owns query fragment band w.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tidx; i < BQ * LD; i += kNThreads) {
      dQs[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id < BQ / kFrag) {
      const short qband = short(simd_group_id) * kFrag;
      MMATile<AccumType, 1, 1, MMAFrag_acc_t> dSfrag2;
      MMATile<AccumType, 1, TD, MMAFrag_acc_t> Ktile2;
      MMATile<AccumType, 1, TD, MMAFrag_acc_t> dQtile;
      dQtile.clear();

      for (short kk2 = 0; kk2 < TK; ++kk2) {
        simdgroup_barrier(mem_flags::mem_none);
        dSfrag2.template load<AccumType, 1, 1, LK, 1>(
            &dSs[(qband + sm) * LK + kk2 * kFrag + sn]);
        Ktile2.template load<AccumType, 1, 1, LD, 1>(
            &Ks[(kk2 * kFrag + sm) * LD + sn]);
        simdgroup_barrier(mem_flags::mem_none);
        tile_matmad(dQtile, dSfrag2, Ktile2, dQtile);
      }

      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < TD; ++j) {
        PREFILL_PRAGMA_UNROLL
        for (short e = 0; e < MMAFrag_acc_t::kElemsPerFrag; ++e) {
          const short er = e / MMAFrag_acc_t::kElemCols;
          const short ec = e % MMAFrag_acc_t::kElemCols;
          dQs[(qband + sm + er) * LD + j * kFrag + sn + ec] =
              dQtile.frag_at(0, j)[e];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = tidx; i < BQ * BD; i += kNThreads) {
      const int qi = i / BD;
      const int d = i % BD;
      const int q = q0 + qi;
      if (q >= qL || d >= D) {
        continue;
      }
      const float dq = dQs[qi * LD + d];
      if (dq != 0.0f) {
        atomic_fetch_add_explicit(
            &dq_base[q * params->Q_strides[2] + d], dq, memory_order_relaxed);
      }
    }
  }

  // Store dK/dV once, from the warp that owns each key band.
  if (simd_group_id < BK / kFrag) {
    const short kband = short(simd_group_id) * kFrag;
    PREFILL_PRAGMA_UNROLL
    for (short j = 0; j < TD; ++j) {
      PREFILL_PRAGMA_UNROLL
      for (short e = 0; e < MMAFrag_acc_t::kElemsPerFrag; ++e) {
        const short er = e / MMAFrag_acc_t::kElemCols;
        const short ec = e % MMAFrag_acc_t::kElemCols;
        const int ki = kband + sm + er;
        const int d = j * kFrag + sn + ec;
        const int kk = k0 + ki;
        if (kk < kL && d < D) {
          dk_base[kk * params->K_strides[2] + d] =
              static_cast<T>(dKtile.frag_at(0, j)[e]);
          dv_base[kk * params->V_strides[2] + d] =
              static_cast<T>(dVtile.frag_at(0, j)[e]);
        }
      }
    }
  }
}



// Fused single-pass backward. One threadgroup per key block; Q/dO fragments are
// pulled into registers once per query block and every product is consumed
// while they are still live, so nothing but K/V and a dQ staging tile needs
// threadgroup storage. Freeing those buffers is what lets BQ reach 64 and cuts
// the inner pass count 4x against the two-phase variant above.
//
// Scores are computed TRANSPOSED (St = K Qt) on purpose: dV += Pt dO and
// dK += dSt Q then consume the fragments in the orientation the MMA already
// produced, and dQ -- the one product wanting the other orientation -- is the
// only one routed through threadgroup scratch.
template <typename T, int BQ, int BK, int BD, int WM, int WN, bool DO_CAUSAL>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
flash_attn_bwd_fused(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    const device T* DO [[buffer(3)]],
    const device float* Lse [[buffer(4)]],
    const device float* Delta [[buffer(5)]],
    device T* DK [[buffer(6)]],
    device T* DV [[buffer(7)]],
    device atomic_float* DQ [[buffer(8)]],
    const constant FlashAttnBwdParams* params [[buffer(9)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  (void)lid;
  using AccumType = float;

  const int D = params->D;
  const int qL = params->qL;
  const int kL = params->kL;
  const int H = params->H;
  const float scale = params->scale;
  const int b = tid.z / H;
  const int h = tid.z % H;
  const int k0 = tid.x * BK;

  const device T* q_base =
      Q + b * params->Q_strides[0] + h * params->Q_strides[1];
  const device T* k_base =
      K + b * params->K_strides[0] + h * params->K_strides[1];
  const device T* v_base =
      V + b * params->V_strides[0] + h * params->V_strides[1];
  const device T* do_base =
      DO + b * params->O_strides[0] + h * params->O_strides[1];
  const device float* lse_base =
      Lse + b * params->L_strides[0] + h * params->L_strides[1];
  const device float* delta_base =
      Delta + b * params->L_strides[0] + h * params->L_strides[1];
  device T* dk_base = DK + b * params->K_strides[0] + h * params->K_strides[1];
  device T* dv_base = DV + b * params->V_strides[0] + h * params->V_strides[1];
  device atomic_float* dq_base =
      DQ + b * params->Q_strides[0] + h * params->Q_strides[1];

  constexpr short kFrag = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFrag, kFrag>;
  constexpr int kNWarps = WM * WN;
  constexpr int kNThreads = kNWarps * 32;
  constexpr int TQ = BQ / kFrag;
  constexpr int TD = BD / kFrag;
  constexpr int LD = BD + kFrag;

  threadgroup AccumType Ks[BK * LD];
  threadgroup AccumType Vs[BK * LD];
  // One dQ staging tile. Metal has no threadgroup float atomic, so the key
  // warps add into it in turn, separated by barriers -- BK/kFrag short rounds
  // per query block, which is far cheaper than the device traffic it avoids.
  threadgroup AccumType dQs[BQ * LD];
  threadgroup float Ls[BQ];
  threadgroup float Ds[BQ];

  const int tidx = int(simd_group_id) * 32 + int(simd_lane_id);
  const short2 sc = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = sc.y;
  const short sn = sc.x;

  for (int i = tidx; i < BK * LD; i += kNThreads) {
    Ks[i] = 0;
    Vs[i] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = tidx; i < BK * BD; i += kNThreads) {
    const int r = i / BD;
    const int c = i % BD;
    const int kk = k0 + r;
    if (kk < kL && c < D) {
      Ks[r * LD + c] =
          static_cast<AccumType>(k_base[kk * params->K_strides[2] + c]);
      Vs[r * LD + c] =
          static_cast<AccumType>(v_base[kk * params->V_strides[2] + c]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // dK/dV stay in registers for the whole query loop: this warp owns key
  // fragment band `kband` and the full head dim.
  const bool warp_has_k = (simd_group_id < BK / kFrag);
  const short kband = short(simd_group_id) * kFrag;
  MMATile<AccumType, 1, TD, MMAFrag_acc_t> dKreg;
  MMATile<AccumType, 1, TD, MMAFrag_acc_t> dVreg;
  dKreg.clear();
  dVreg.clear();

  const int q_start = DO_CAUSAL ? (k0 / BQ) * BQ : 0;

  for (int q0 = q_start; q0 < qL; q0 += BQ) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tidx; i < BQ * LD; i += kNThreads) {
      dQs[i] = 0;
    }
    for (int i = tidx; i < BQ; i += kNThreads) {
      const int q = q0 + i;
      Ls[i] = (q < qL) ? lse_base[q] : INFINITY;
      Ds[i] = (q < qL) ? delta_base[q] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp_has_k) {
      // Walk the query block one fragment at a time. Q and dO for the current
      // fragment are pulled straight from device into registers and every
      // product that needs them is issued before they are dropped.
      for (short qq = 0; qq < TQ; ++qq) {
        MMATile<AccumType, 1, TD, MMAFrag_acc_t> Qreg;
        MMATile<AccumType, 1, TD, MMAFrag_acc_t> dOreg;
        PREFILL_PRAGMA_UNROLL
        for (short j = 0; j < TD; ++j) {
          PREFILL_PRAGMA_UNROLL
          for (short e = 0; e < MMAFrag_acc_t::kElemsPerFrag; ++e) {
            const short er = e / MMAFrag_acc_t::kElemCols;
            const short ec = e % MMAFrag_acc_t::kElemCols;
            const int qi = qq * kFrag + sm + er;
            const int d = j * kFrag + sn + ec;
            const int q = q0 + qi;
            AccumType qv = 0;
            AccumType dov = 0;
            if (q < qL && d < D) {
              qv = static_cast<AccumType>(q_base[q * params->Q_strides[2] + d]);
              dov = static_cast<AccumType>(do_base[q * params->O_strides[2] + d]);
            }
            Qreg.frag_at(0, j)[e] = qv;
            dOreg.frag_at(0, j)[e] = dov;
          }
        }

        // St = K Qt and dPt = V dOt, so the fragments come out [k][q].
        MMATile<AccumType, 1, 1, MMAFrag_acc_t> Stile;
        MMATile<AccumType, 1, 1, MMAFrag_acc_t> dPtile;
        MMATile<AccumType, 1, 1, MMAFrag_acc_t> Kf;
        MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vf;
        MMATile<AccumType, 1, 1, MMAFrag_acc_t> Qf;
        MMATile<AccumType, 1, 1, MMAFrag_acc_t> dOf;
        Stile.clear();
        dPtile.clear();

        PREFILL_PRAGMA_UNROLL
        for (short dd = 0; dd < TD; ++dd) {
          simdgroup_barrier(mem_flags::mem_none);
          Kf.template load<AccumType, 1, 1, LD, 1>(
              &Ks[(kband + sm) * LD + dd * kFrag + sn]);
          Vf.template load<AccumType, 1, 1, LD, 1>(
              &Vs[(kband + sm) * LD + dd * kFrag + sn]);
          PREFILL_PRAGMA_UNROLL
          for (short e = 0; e < MMAFrag_acc_t::kElemsPerFrag; ++e) {
            Qf.frag_at(0, 0)[e] = Qreg.frag_at(0, dd)[e];
            dOf.frag_at(0, 0)[e] = dOreg.frag_at(0, dd)[e];
          }
          simdgroup_barrier(mem_flags::mem_none);
          // Qf/dOf enter as the transposed operand, giving [k][q] output.
          tile_matmad(Stile, Kf, Qf, Stile);
          tile_matmad(dPtile, Vf, dOf, dPtile);
        }

        // P and dS, still [k][q] and still in registers.
        MMATile<AccumType, 1, 1, MMAFrag_acc_t> Pt;
        MMATile<AccumType, 1, 1, MMAFrag_acc_t> dSt;
        PREFILL_PRAGMA_UNROLL
        for (short e = 0; e < MMAFrag_acc_t::kElemsPerFrag; ++e) {
          const short er = e / MMAFrag_acc_t::kElemCols;
          const short ec = e % MMAFrag_acc_t::kElemCols;
          const int ki = kband + sm + er;
          const int qi = qq * kFrag + sn + ec;
          const int kk = k0 + ki;
          const int q = q0 + qi;
          float p = 0.0f;
          float ds = 0.0f;
          if (q < qL && kk < kL && !(DO_CAUSAL && q < kk)) {
            const float lse = Ls[qi];
            if (!metal::isinf(lse)) {
              p = metal::exp(Stile.frag_at(0, 0)[e] * scale - lse);
              ds = p * (dPtile.frag_at(0, 0)[e] - Ds[qi]) * scale;
            }
          }
          Pt.frag_at(0, 0)[e] = p;
          dSt.frag_at(0, 0)[e] = ds;
        }

        // dV += Pt dO and dK += dSt Q, consuming dOreg/Qreg while live.
        PREFILL_PRAGMA_UNROLL
        for (short j = 0; j < TD; ++j) {
          MMATile<AccumType, 1, 1, MMAFrag_acc_t> src;
          MMATile<AccumType, 1, 1, MMAFrag_acc_t> acc;
          PREFILL_PRAGMA_UNROLL
          for (short e = 0; e < MMAFrag_acc_t::kElemsPerFrag; ++e) {
            src.frag_at(0, 0)[e] = dOreg.frag_at(0, j)[e];
            acc.frag_at(0, 0)[e] = dVreg.frag_at(0, j)[e];
          }
          tile_matmad(acc, Pt, src, acc);
          PREFILL_PRAGMA_UNROLL
          for (short e = 0; e < MMAFrag_acc_t::kElemsPerFrag; ++e) {
            dVreg.frag_at(0, j)[e] = acc.frag_at(0, 0)[e];
            src.frag_at(0, 0)[e] = Qreg.frag_at(0, j)[e];
            acc.frag_at(0, 0)[e] = dKreg.frag_at(0, j)[e];
          }
          tile_matmad(acc, dSt, src, acc);
          PREFILL_PRAGMA_UNROLL
          for (short e = 0; e < MMAFrag_acc_t::kElemsPerFrag; ++e) {
            dKreg.frag_at(0, j)[e] = acc.frag_at(0, 0)[e];
          }
        }

        // dQ += dS K needs the untransposed orientation, so it is the one
        // product that goes through threadgroup scratch. Key warps take turns
        // so the accumulation needs no atomic.
        for (short turn = 0; turn < BK / kFrag; ++turn) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (turn == short(simd_group_id)) {
        PREFILL_PRAGMA_UNROLL
        for (short e = 0; e < MMAFrag_acc_t::kElemsPerFrag; ++e) {
          const short er = e / MMAFrag_acc_t::kElemCols;
          const short ec = e % MMAFrag_acc_t::kElemCols;
          const int ki = kband + sm + er;
          const int qi = qq * kFrag + sn + ec;
          const float ds = dSt.frag_at(0, 0)[e];
          if (ds == 0.0f) {
            continue;
          }
          for (int d = 0; d < D; ++d) {
            dQs[qi * LD + d] += ds * Ks[ki * LD + d];
          }
        }
        }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tidx; i < BQ * BD; i += kNThreads) {
      const int qi = i / BD;
      const int d = i % BD;
      const int q = q0 + qi;
      if (q >= qL || d >= D) {
        continue;
      }
      const AccumType val = dQs[qi * LD + d];
      if (val != 0) {
        atomic_fetch_add_explicit(
            &dq_base[q * params->Q_strides[2] + d], val, memory_order_relaxed);
      }
    }
  }

  if (warp_has_k) {
    PREFILL_PRAGMA_UNROLL
    for (short j = 0; j < TD; ++j) {
      PREFILL_PRAGMA_UNROLL
      for (short e = 0; e < MMAFrag_acc_t::kElemsPerFrag; ++e) {
        const short er = e / MMAFrag_acc_t::kElemCols;
        const short ec = e % MMAFrag_acc_t::kElemCols;
        const int ki = kband + sm + er;
        const int d = j * kFrag + sn + ec;
        const int kk = k0 + ki;
        if (kk < kL && d < D) {
          dk_base[kk * params->K_strides[2] + d] =
              static_cast<T>(dKreg.frag_at(0, j)[e]);
          dv_base[kk * params->V_strides[2] + d] =
              static_cast<T>(dVreg.frag_at(0, j)[e]);
        }
      }
    }
  }
}

#define INSTANTIATE_FLASH_BWD_NAMED(SUF, DTYPE, BQ, BK, BD, WM, WN, NT)                   \
  template [[host_name("flash_attn_bwd_preprocess" SUF "_" #DTYPE)]] [[kernel]] void  \
  flash_attn_bwd_preprocess<DTYPE, BQ, NT>(                                    \
      const device DTYPE*,                                                     \
      const device DTYPE*,                                                     \
      device float*,                                                           \
      const constant FlashAttnBwdParams*,                                      \
      uint3,                                                                   \
      uint3);                                                                  \
  template [[host_name("flash_attn_bwd_dkdv" SUF "_" #DTYPE)]] [[kernel]] void        \
  flash_attn_bwd_dkdv<DTYPE, BQ, BK, BD, WM, WN, false>(                       \
      const device DTYPE*,                                                     \
      const device DTYPE*,                                                     \
      const device DTYPE*,                                                     \
      const device DTYPE*,                                                     \
      const device float*,                                                     \
      const device float*,                                                     \
      device DTYPE*,                                                           \
      device DTYPE*,                                                           \
      device atomic_float*,                                                    \
      const constant FlashAttnBwdParams*,                                      \
      uint,                                                                    \
      uint,                                                                    \
      uint3,                                                                   \
      uint3);                                                                  \
  template [[host_name("flash_attn_bwd_dkdv" SUF "_causal_" #DTYPE)]] [[kernel]] void \
  flash_attn_bwd_dkdv<DTYPE, BQ, BK, BD, WM, WN, true>(                        \
      const device DTYPE*,                                                     \
      const device DTYPE*,                                                     \
      const device DTYPE*,                                                     \
      const device DTYPE*,                                                     \
      const device float*,                                                     \
      const device float*,                                                     \
      device DTYPE*,                                                           \
      device DTYPE*,                                                           \
      device atomic_float*,                                                    \
      const constant FlashAttnBwdParams*,                                      \
      uint,                                                                    \
      uint,                                                                    \
      uint3,                                                                   \
      uint3);

// BD=64 covers every head dim this kernel accepts (larger ones fall back).
// BQ=BK=16 gives 2 fragment bands, one per pair of warps, and threadgroup use
// (BK*LD*2 + BQ*LD*2)*4 + BQ*LK*2*4 + BQ*2*4 = 21632 bytes against Apple's
// 32768 cap (26240 with the dQ staging buffer). BQ=BK=64 with BD=64 needs
// 111104 and fails PSO creation.
INSTANTIATE_FLASH_BWD_NAMED("", float, 16, 16, 64, 2, 2, 128)
INSTANTIATE_FLASH_BWD_NAMED("", half, 16, 16, 64, 2, 2, 128)
// head_dim 40 (SD1.5 UNet) leaves enough budget for a 32-wide query tile, which
// halves the inner pass count against the BD=64 variant.
INSTANTIATE_FLASH_BWD_NAMED("_d40", float, 32, 16, 40, 2, 2, 128)
INSTANTIATE_FLASH_BWD_NAMED("_d40", half, 32, 16, 40, 2, 2, 128)
// head_dim 128 doubles LD, so the 16-wide tiles above would need 43136 bytes.
// Halving both tiles brings it to 21056. BQ=BK=8 is one fragment per axis, so
// the warp grid has to be 1x1: 2x2 leaves three of four warps with no band and
// the dK/dV accumulators come out wrong.
INSTANTIATE_FLASH_BWD_NAMED("_d128", float, 8, 8, 128, 1, 1, 32)
INSTANTIATE_FLASH_BWD_NAMED("_d128", half, 8, 8, 128, 1, 1, 32)

#define INSTANTIATE_FLASH_BWD_FUSED(SUF, DTYPE, BQ, BK, BD, WM, WN)              \
  template [[host_name("flash_attn_bwd_fused" SUF "_" #DTYPE)]] [[kernel]] void  \
  flash_attn_bwd_fused<DTYPE, BQ, BK, BD, WM, WN, false>(                        \
      const device DTYPE*,                                                       \
      const device DTYPE*,                                                       \
      const device DTYPE*,                                                       \
      const device DTYPE*,                                                       \
      const device float*,                                                       \
      const device float*,                                                       \
      device DTYPE*,                                                             \
      device DTYPE*,                                                             \
      device atomic_float*,                                                      \
      const constant FlashAttnBwdParams*,                                        \
      uint,                                                                      \
      uint,                                                                      \
      uint3,                                                                     \
      uint3);                                                                    \
  template [[host_name("flash_attn_bwd_fused" SUF "_causal_" #DTYPE)]] [[kernel]] void \
  flash_attn_bwd_fused<DTYPE, BQ, BK, BD, WM, WN, true>(                         \
      const device DTYPE*,                                                       \
      const device DTYPE*,                                                       \
      const device DTYPE*,                                                       \
      const device DTYPE*,                                                       \
      const device float*,                                                       \
      const device float*,                                                       \
      device DTYPE*,                                                             \
      device DTYPE*,                                                             \
      device atomic_float*,                                                      \
      const constant FlashAttnBwdParams*,                                        \
      uint,                                                                      \
      uint,                                                                      \
      uint3,                                                                     \
      uint3);

// BQ=64/BK=32 at BD=40 uses 25088 of the 32768-byte threadgroup budget and
// quarters the inner pass count against the two-phase kernel above.
INSTANTIATE_FLASH_BWD_FUSED("_d40", float, 64, 32, 40, 2, 2)
INSTANTIATE_FLASH_BWD_FUSED("_d40", half, 64, 32, 40, 2, 2)
INSTANTIATE_FLASH_BWD_FUSED("", float, 32, 32, 64, 2, 2)
INSTANTIATE_FLASH_BWD_FUSED("", half, 32, 32, 64, 2, 2)
