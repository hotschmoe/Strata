#pragma once
// STRATA_BATCH_MTP_DEPTH (experimental, opt-in): how many MTP proposals a --batch-mtp slot verifies per window.
// 1 (the default) is exactly the two-row-per-slot packer of 0.1.44; 2 or 3 give a slot 3 or 4 contiguous rows (its
// current token and up to that many chained drafts).  Pure host logic so a CPU test can pin it down: the clamp,
// the window packing (rotation and the 8-row verifier cap), and the chained prefix acceptance.  Nothing here touches
// the verifier, the commit, or the drafter: a slot's group is still a contiguous run of rows at consecutive
// positions (batch_rows.hpp), committed by Verifier::commit_slot_prefixes with the kept prefix length.

#include <cstdint>
#include <cstdlib>

namespace strata::core {

/// The depth `env` asks for (null/empty/unparsable = 1), clamped to [1, min(spec - 1, max_t - 1)]: a slot's group of
/// 1 + depth rows must fit the drafter's window (`spec` rows, so spec - 1 drafts) and the verifier's window (`max_t`).
inline int batch_depth_resolve(const char* env, int spec, int max_t) {
    int d = 1;
    if (env != nullptr && env[0] != '\0') d = std::atoi(env);
    int cap = spec - 1 < max_t - 1 ? spec - 1 : max_t - 1;
    if (cap < 1) cap = 1;
    if (d < 1) d = 1;
    if (d > cap) d = cap;
    return d;
}

// Never send rows that batch_keep cannot accept to the verifier. In particular,
// a deeper window near the context end must not write K/V beyond its allocation.
inline int batch_drafts_for_context(int drafts, int64_t p, int64_t max_context) {
    const int64_t room = max_context - p - 2;
    if (drafts <= 0 || room <= 0) return 0;
    return room < drafts ? (int) room : drafts;
}

/// Packs one window.  `active[b]`: slot b has a live request; `ndraft[b]`: the drafts it holds ready (0..depth; 0 =
/// its group is the one token row).  Slots are visited from `next_slot` in rotation order and the first one that does
/// not fit `max_t` rows ends the window (the same rule as the fixed two-row packer: no slot is skipped for a smaller
/// one behind it, so the rotation stays fair).  Out: `rows[t]` the slot of row t, and per packed slot a: `first[a]`
/// its first row, `glen[a]` its rows (1 + the drafts it offers), `slot[a]` its index.  Returns the packed slots A;
/// `*S_out` the rows.  Without `mtp` every group is one row.
inline int batch_pack(const bool* active, const int* ndraft, int n_slots, size_t next_slot, bool mtp, int max_t,
                      int* rows, int* first, int* glen, int* slot, int* S_out) {
    int S = 0, A = 0;
    for (int offset = 0; offset < n_slots; ++offset) {
        const int b = (int) ((next_slot + (size_t) offset) % (size_t) n_slots);
        if (!active[b]) continue;
        int g = 1;
        if (mtp) {
            int nd = ndraft[b];
            if (nd < 0) nd = 0;
            g = 1 + nd;
        }
        if (S + g > max_t) break;
        first[A] = S;
        glen[A] = g;
        slot[A] = b;
        ++A;
        for (int j = 0; j < g; ++j) rows[S + j] = b;
        S += g;
    }
    *S_out = S;
    return A;
}

/// The rows a slot keeps of its group: `outb[j]` is the target's pick after row j, `tok[j]` row j's token (tok[0] the
/// current token, tok[1..glen) the drafts).  Row j (j >= 1) is kept when every row before it was, the pick after
/// row j - 1 is its draft and not an end token, the request is not cancelled, and emitting j + 1 tokens stays within
/// `max_new` and the context (`p` is row 0's position; a kept row j puts the slot at p + j + 1 and the next window
/// needs one more cell).  Returns keep in [1, glen]; glen == 2 is exactly the 0.1.44 rule.
template <class IsEos>
inline int batch_keep(const int32_t* outb, const int32_t* tok, int glen, IsEos is_eos, bool stop, int64_t produced,
                      int64_t max_new, int64_t p, int64_t max_context) {
    int keep = 1;
    for (int j = 1; j < glen; ++j) {
        if (outb[j - 1] != tok[j] || is_eos(outb[j - 1]) || stop) break;
        if (produced + (int64_t) j + 1 > max_new) break;
        if (p + (int64_t) j + 2 > max_context) break;
        keep = j + 1;
    }
    return keep;
}

}  // namespace strata::core
