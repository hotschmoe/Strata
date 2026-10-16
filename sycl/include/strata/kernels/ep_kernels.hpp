// include/strata/kernels/ep_kernels.hpp - the SYCL port's expert parallelism.  Card 0 runs every layer and pushes each
// layer's routed ids and q8_1 rows into card 1's inbox; card 1 computes the rows of the experts card 0 does not hold
// and writes the rows themselves (not their weighted sum) into card 0's parts_, so card 0's combine is the single-card
// one.  A flag holds the epoch its card's window graph advances in device memory (a host value would freeze at
// capture), and every spin is bounded.
#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// *ctr += 1: the first node of each card's window graph, so the flags this window raises carry a new epoch.
void ep_bump(uint32_t* ctr, void* stream);
/// Copy the routed ids and q8_1 rows into the peer's inbox and raise its flag to *ctr.  peer_ids, peer_xq and
/// peer_flag are the peer's memory as this card's kernels address it.
void ep_push(const int32_t* ids, int n_ids, const uint8_t* xq, size_t xq_bytes, int32_t* peer_ids, uint8_t* peer_xq,
             uint32_t* peer_flag, const uint32_t* ctr, void* stream);
/// Spin until *flag (this card's memory, written by the other card) reaches *ctr (this card's).  On the bound, *err
/// (mapped host memory, may be null) gets `code` and the wait gives up.
void ep_wait(const uint32_t* flag, const uint32_t* ctr, uint32_t spin_max, uint32_t* err, uint32_t code, void* stream);
/// The peer: copy the rows of the entries card 0 does not own (`res0`) into card 0's parts_ and raise *peer_flag to
/// *ctr, in one kernel (each work-group fences at system scope after its row; the last to finish raises the flag;
/// `done` is zero between calls).  With the flag in a separate later kernel, card 0 sometimes read a row before all of
/// it had arrived.
void ep_send_rows(float* peer_rows, const float* rows, const int32_t* ids, const int32_t* res0, int n, int64_t row,
                  uint32_t* done, uint32_t* peer_flag, const uint32_t* ctr, void* stream);

/// Night lab STRATA_NIGHT_EP_PSUM (default off): ep_push plus the n_ids routing weights into peer_w (the same slot),
/// so the peer can weight its own rows itself.
/// STRATA_NIGHT_EP_PLAN1 (default off): also plan_words int32 of `plan` (card 1's plan, made on card 0) into
/// peer_plan.  w / peer_w (and plan / peer_plan) may be null: not copied.
void ep_push_w(const int32_t* ids, const float* w, int n_ids, const uint8_t* xq, size_t xq_bytes, int32_t* peer_ids,
               float* peer_w, uint8_t* peer_xq, uint32_t* peer_flag, const uint32_t* ctr, void* stream,
               const int32_t* plan = nullptr, int plan_words = 0, int32_t* peer_plan = nullptr);
/// Night lab STRATA_NIGHT_EP_PSUM: the peer sends ONE row per token instead of one per entry: peer_sum[t] = the
/// weighted sum (in entry order, first product rounded then one FMA per entry) of the rows of token t's entries card 0
/// does not own (`res0`, the same partition as ep_send_rows); then *peer_flag = *ctr as ep_send_rows (`done` counts
/// the work-groups, zero between calls).  A token with no peer entry gets a zero row.
void ep_send_psum(float* peer_sum, const float* rows, const int32_t* ids, const float* w, const int32_t* res0,
                  int n_tok, int k, int64_t row, uint32_t* done, uint32_t* peer_flag, const uint32_t* ctr, void* stream);
/// Night lab STRATA_NIGHT_EP_PSUM: card 0's combine: out[t] = sum over the entries card 0 owns (res0[e] >= 0; first
/// product rounded, then one FMA per entry, in entry order) of w * parts, + psum[t] (plain loads, as the combine reads
/// the peer's rows in parts_; -DSTRATA_NIGHT_PSUM_UNCACHED=1 reads past the caches), + the shared row (gated: shared * sigmoid(sg[t]) rounded on its own, as
/// native_moe_combine_multi_hits_gated; sg == nullptr: plain).  FP32 order differs from native_moe_combine_multi
/// (the peer's products are summed first), so results agree to FP32 rounding, not bitwise.
void ep_combine_psum(const float* parts, const float* w, const int32_t* ids, const int32_t* res0, const float* psum,
                     const float* shared, const float* sg, float* out, int64_t row, int k, int n_tok, void* stream);

/// Copy `bytes` read past this card's caches, for data the other card wrote into this card's memory: an inbound PCIe
/// write does not invalidate this card's caches, so a plain load can return the previous window's data.  `bytes` is a
/// multiple of 4; both pointers are 4-byte aligned.
void ep_copy_uncached(void* dst, const void* src, size_t bytes, void* stream);

/// Opt-in prompt-only reduction: sum the peer-owned routed contributions into one row per token.
/// slot maps original router order to grouped rows; peer rows start at local_rows. Changes FP32 addition order.
/// Night lab (STRATA_NIGHT_PREFILL_HALVES): with `flag` set, every work-group fences its stores at system scope and the
/// last one to count itself in `done` (this card's memory, zero between calls) resets it and stores `value` into *flag
/// (the other card's memory): the other card's doorbell for these sums.
void ep_prefill_reduce(const float* rows, const int32_t* slot, const float* weights, float* sum,
                       int64_t tokens, int64_t local_rows, int64_t width, int k, void* stream, bool push = false,
                       uint32_t* done = nullptr, uint32_t* flag = nullptr, uint32_t value = 0);
/// Add the primary contributions and shared expert. `out` initially holds the peer sum, then is replaced in place.
void ep_prefill_finish(const float* rows, const int32_t* slot, const float* weights,
                       const float* shared, const float* gate, float* out,
                       int64_t tokens, int64_t local_rows, int64_t width, int k, void* stream);

/// Night lab (STRATA_NIGHT_PREFILL_F16SUM=1): ep_prefill_reduce with the per-token sum sent as FP16 times a per-token
/// power-of-two scale (half the PCIe bytes).  The FP32 sums are those of ep_prefill_reduce (same FMA order); each row is
/// scaled by 2^-e so its largest magnitude lands in [2^14, 2^15), rounded to FP16 (relative error <= 2^-11 for every
/// value above max * 2^-28), and 2^e is written to scale[t].  sum_h holds tokens * width halves, `width` / 4 a multiple
/// of 160 work-items x 4 vectors (width 2560).
void ep_prefill_reduce_h(const float* rows, const int32_t* slot, const float* weights, uint16_t* sum_h, float* scale,
                         int64_t tokens, int64_t local_rows, int64_t width, int k, void* stream, bool push,
                         uint32_t* done = nullptr, uint32_t* flag = nullptr, uint32_t value = 0);
/// ep_prefill_finish reading the peer sum straight from ep_prefill_reduce_h's FP16 rows (past this card's caches, as
/// ep_copy_uncached): out = (primary sum + peer) + shared * sigmoid(gate), the order of ep_prefill_finish.
void ep_prefill_finish_h(const float* rows, const int32_t* slot, const float* weights, const float* shared,
                         const float* gate, const uint16_t* sum_h, const float* scale, float* out, int64_t tokens,
                         int64_t local_rows, int64_t width, int k, void* stream);

/// Night lab (STRATA_NIGHT_PREFILL_EARLY): copy n4 float4s (16-byte loads and stores, the grid of copy_f32_wide) into
/// the other card's memory, then raise *peer_flag to `value` once EVERY work-group has fenced its stores at system scope
/// (the last group to count itself in `done` - this card's memory, zero between calls - resets it and stores the flag),
/// as ep_send_rows: a flag raised by a separate later kernel was seen before all rows had arrived.
void ep_push_signal(void* peer_dst, const void* src, int64_t n4, uint32_t* done, uint32_t* peer_flag, uint32_t value,
                    void* stream);
/// Night lab: spin until *flag (this card's memory) - value >= 0 as int32 (epochs only grow), then fence (acquire); at
/// spin_max loads *err (mapped host memory) = code and the wait gives up.
void ep_wait_value(const uint32_t* flag, uint32_t value, uint32_t spin_max, uint32_t* err, uint32_t code, void* stream);

/// STRATA_EP_VERIFY=1: add to *mismatch (host) the peer-owned rows where peer_rows (read past the caches, as
/// ep_copy_uncached) differs from this card's parts.
void ep_compare(const float* parts, const float* peer_rows, const int32_t* ids, const int32_t* res0, int n, int64_t row,
                uint32_t* mismatch, void* stream);

/// resident_plan for one card of the pair.  An entry belongs to card 0 when its expert is resident there (`res0`),
/// else to card 1 (`res1`); an expert on neither is a plan error.  Role 1 is card 0, role 2 card 1: each plans only
/// its own entries, and counts[1] is their number.  res0 / res1 are this card's copies of both tables for the layer.
/// Night lab STRATA_NIGHT_EP_PLAN1: both roles' plans on card 0 in ONE launch (work-group 0: role 1 into plan0 with
/// card 0's arena; work-group 1: role 2 into plan1 with card 1's arena base / slot offsets as card 1 addresses them,
/// slot_off1 a copy in card 0's memory or null), word for word what resident_plan_ep makes on each card.
void resident_plan_ep_pair(const int32_t* ids, int n_entries, int k, const int32_t* res0, const int32_t* res1,
                           int n_expert, const uint8_t* cache_base0, const unsigned long long* slot_off0, long long blob0,
                           int32_t* plan0, const uint8_t* cache_base1, const unsigned long long* slot_off1,
                           long long blob1, int32_t* plan1, long long capx, void* stream, uint32_t* plan_err);
void resident_plan_ep(const int32_t* ids, int n_entries, int k, const int32_t* res0, const int32_t* res1, int role,
                      int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off, long long blob,
                      int32_t* plan, long long capx, void* stream, uint32_t* plan_err);

}  // namespace strata::kernels
