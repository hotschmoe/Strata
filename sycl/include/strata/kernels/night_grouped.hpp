// Night lab 2026-10-16 (STRATA_NIGHT_PREFILL_GROUPED): grouped prefill expert GEMM straight from the native blocks.
// One launch covers every entry (an expert's run of rows) of a layer on a card: work-group (entry, N-tile), each
// sub-group 16 output columns; every lane decodes 32 values of its weight row per K step into its sub-group's slice of
// local memory, the sub-group multiplies them with joint_matrix (8x16x16 FP16 -> FP32) against the entry's FP16 rows.
// No FP16 copy of the expert in global memory, no per-expert host launch.
#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// One entry: `rows` (1 .. 8 * mt) consecutive rows starting at row0 of X / Y, multiplied with the expert at `w`.
struct NightGroupedEntry {
    const uint8_t* w;   ///< gate/up: the blob (gate rows at +0, up rows at +up_off); down: the down matrix itself
    int64_t row0;
    int32_t rows;
    int32_t pad;
};

/// Y[row0 + m][n] = sum_k X[row0 + m][k] * W[n][k], X FP16 [*, K], Y FP32 [*, N] (row stride N).
/// gu: W's row n is (n & 1 ? up : gate) row n >> 1 (iq_dequant_gu_f16's interleave), N = 2 * n_ff; else W = w's rows.
/// mt = 8-row tiles per entry (2, 4 or 8).  N a multiple of 64, K of 32; ty one of 16 17 18 20 21 22 42.
/// `ent` is device memory (n_ent entries).  Returns false (nothing launched) for an unsupported type / shape.
bool night_grouped_f16(int ty, bool gu, const NightGroupedEntry* ent, int n_ent, int mt, const uint16_t* X, float* Y,
                       int N, int K, size_t up_off, void* stream);
/// STRATA_NIGHT_PREFILL_DQ32: iq_dequant_expert_f16's values and layout from one work-item per 32-value sub-block;
/// false (nothing launched) for a type pair it does not cover.  night_dq32_on() = the flag (night_dq32_override(0/1)
/// forces it for the parity probe; -1 back to the environment).
bool night_dequant_expert_f16(int gu_type, int d_type, const void* gate, const void* up, const void* down, int64_t n_ff,
                              int64_t n_embd, uint16_t* dst_gu, uint16_t* dst_d, void* stream);
bool night_dq32_on();
void night_dq32_override(int on);
/// The entry row cap for mt.
constexpr int night_grouped_rows(int mt) { return 8 * mt; }

}  // namespace strata::kernels
