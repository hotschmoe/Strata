// include/strata/kernels/qsa_prompt_attn_union.hpp - night-lab STRATA_NIGHT_PATTN_XMX: the prompt path's QSA attention
// with QUERY TILES on XMX.
//
// The FP32 prompt attention (qsa_decode_attn_batch / the S93 value pipeline) gathers every query's own ~2,051 selected
// cells and runs 12 heads per KV head on the vector engines (~2.4 TFLOP/s); the XMX v2 port keeps one query per
// work-group (12 real rows of 16) and is gather-bound. Here 8 neighbouring queries share one pass over the UNION of their
// selections:
//   pass 1 (one work-group per 8-query tile): the sorted union of the 8 selections, each cell tagged with the 8-bit
//          set of tile queries that selected it (cell | mask << 24);
//   pass 2 (one work-group per (tile, KV head), 12 sub-groups x 8 rows = the tile's 8 queries x 12 heads): 32-cell
//          chunks of the union staged once in local memory as FP16 K^T / V (INT8 code x its 64-value scale), S = Q K^T
//          and O += P V on joint_matrix (FP16 x FP16 -> FP32, 8x16x16), an online softmax per row in which a cell the
//          row's query did not select is -inf - so every query attends to exactly its own selection.
// Not bitwise: Q, K, V and P enter the products as FP16 (one rounding each) and the summation order is the tile's.
// INT8 K/V pools only (the serving format); anything else returns false and the caller takes its fallback.
#pragma once

#include "strata/kernels/qsa_decode_attn.hpp"

#include <cstdint>

namespace strata::kernels {

/// Same arguments and output as qsa_prompt_attn_batch. False (nothing launched) when not INT8 K/V, not 24/2/256.
bool qsa_prompt_attn_union(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, void* stream);

/// STRATA_NIGHT_PATTN_XMX=1 (latched; one stderr line when on).
bool qsa_night_pattn_xmx();

}  // namespace strata::kernels
