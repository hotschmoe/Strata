// STRATA_BATCH_MTP_DEPTH's host logic (include/strata/core/batch_depth.hpp): the clamp, the window packing under the
// 8-row verifier cap with variable group sizes and the slot rotation, and the chained prefix acceptance (mismatch,
// end token, cancel, max_new, context).  Header-only, CPU.  Depth 1 must reproduce the 0.1.44 two-row rule exactly.
#include "strata/core/batch_depth.hpp"
#include "strata/core/batch_rows.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
constexpr int kMaxT = 8;
const auto eos_7 = [](int32_t t) { return t == 7; };

// the 0.1.44 rule, verbatim from batch_step, for the depth-1 equivalence check
int keep_0144(const int32_t* outb, const int32_t* tok, bool eos0, bool stop, long long produced, long long max_new,
              long long p, long long max_context) {
    return outb[0] == tok[1] && !eos0 && !stop && produced + 2 <= max_new && p + 3 <= max_context ? 2 : 1;
}
}  // namespace

int main() {
    using namespace strata::core;
    // ---- the clamp
    check(batch_depth_resolve(nullptr, 4, kMaxT) == 1, "unset -> 1");
    check(batch_depth_resolve("", 4, kMaxT) == 1, "empty -> 1");
    check(batch_depth_resolve("x", 4, kMaxT) == 1, "unparsable -> 1");
    check(batch_depth_resolve("0", 4, kMaxT) == 1, "0 -> 1");
    check(batch_depth_resolve("-3", 4, kMaxT) == 1, "negative -> 1");
    check(batch_depth_resolve("2", 4, kMaxT) == 2, "2 with spec 4");
    check(batch_depth_resolve("3", 4, kMaxT) == 3, "3 with spec 4");
    check(batch_depth_resolve("9", 4, kMaxT) == 3, "9 clamps to spec - 1 = 3");
    check(batch_depth_resolve("3", 2, kMaxT) == 1, "spec 2 caps at 1");
    check(batch_depth_resolve("3", 3, kMaxT) == 2, "spec 3 caps at 2");
    check(batch_depth_resolve("7", 16, kMaxT) == 7, "verifier cap max_t - 1");
    check(batch_depth_resolve("9", 16, kMaxT) == 7, "9 clamps to max_t - 1");

    // Context-end windows: all proposed rows must fit before verifier writes,
    // including drafts later rejected by token equality or length acceptance.
    for (int64_t p = 0; p < 64; ++p) {
        for (int d = 0; d <= 3; ++d) {
            const int n = batch_drafts_for_context(d, p, 64);
            check(n >= 0 && n <= d, "bounded draft count");
            check(n == 0 || p + n + 2 <= 64, "proposed rows respect context acceptance bound");
            check(n == d || p + n + 3 > 64, "maximal safe proposal prefix");
        }
    }
    check(batch_drafts_for_context(3, 60, 64) == 2, "near end: two drafts");
    check(batch_drafts_for_context(3, 61, 64) == 1, "near end: one draft");
    check(batch_drafts_for_context(3, 62, 64) == 0, "near end: current row only");

    // ---- packing
    int rows[kMaxT], first[kMaxT], glen[kMaxT], slot[kMaxT], S = 0;
    {
        // plain --batch: one row each, every active slot
        const bool act[4] = {true, false, true, true};
        const int nd[4] = {3, 3, 3, 3};
        const int A = batch_pack(act, nd, 4, 0, false, kMaxT, rows, first, glen, slot, &S);
        check(A == 3 && S == 3 && glen[0] == 1 && glen[1] == 1 && glen[2] == 1, "plain: one row per slot");
        check(slot[0] == 0 && slot[1] == 2 && slot[2] == 3, "plain: rotation order from 0");
    }
    {
        // depth 1 (ndraft 1 everywhere): the two-row packer, four slots fill eight rows
        const bool act[4] = {true, true, true, true};
        const int nd[4] = {1, 1, 1, 1};
        const int A = batch_pack(act, nd, 4, 0, true, kMaxT, rows, first, glen, slot, &S);
        check(A == 4 && S == 8, "depth 1: four slots, eight rows");
        for (int a = 0; a < A; ++a) check(glen[a] == 2 && first[a] == 2 * a, "depth 1: two rows per slot");
        const int rows_want[8] = {0, 0, 1, 1, 2, 2, 3, 3};
        for (int t = 0; t < 8; ++t) check(rows[t] == rows_want[t], "depth 1: rows layout");
    }
    {
        // depth 1 with five active slots: the fifth does not fit; rotation then starts after the last packed
        const bool act[5] = {true, true, true, true, true};
        const int nd[5] = {1, 1, 1, 1, 1};
        int A = batch_pack(act, nd, 5, 0, true, kMaxT, rows, first, glen, slot, &S);
        check(A == 4 && S == 8 && slot[3] == 3, "depth 1: five slots -> four packed");
        A = batch_pack(act, nd, 5, (size_t) slot[3] + 1, true, kMaxT, rows, first, glen, slot, &S);
        check(A == 4 && slot[0] == 4 && slot[1] == 0 && slot[2] == 1 && slot[3] == 2, "depth 1: rotation resumes at 4");
    }
    {
        // depth 3, two slots: 4 + 4 = 8 rows exactly
        const bool act[2] = {true, true};
        const int nd[2] = {3, 3};
        const int A = batch_pack(act, nd, 2, 0, true, kMaxT, rows, first, glen, slot, &S);
        check(A == 2 && S == 8 && glen[0] == 4 && glen[1] == 4 && first[1] == 4, "depth 3: two slots fill the window");
        int64_t pos[8];
        for (int t = 0; t < 8; ++t) pos[t] = (rows[t] == 0 ? 100 : 5000) + (t % 4);
        check(batch_rows_error(rows, S, pos, 2, kMaxT, 0, false, kMaxT) == nullptr, "depth 3: a valid verifier layout");
    }
    {
        // depth 3, three slots: the third does not fit (4 + 4 + 4 > 8) and is NOT skipped for; it leads next time
        const bool act[3] = {true, true, true};
        const int nd[3] = {3, 3, 3};
        int A = batch_pack(act, nd, 3, 0, true, kMaxT, rows, first, glen, slot, &S);
        check(A == 2 && S == 8 && slot[0] == 0 && slot[1] == 1, "depth 3: three slots -> two packed");
        A = batch_pack(act, nd, 3, (size_t) slot[1] + 1, true, kMaxT, rows, first, glen, slot, &S);
        check(A == 2 && slot[0] == 2 && slot[1] == 0, "depth 3: the third slot leads the next window");
    }
    {
        // variable groups: a slot whose drafter returned fewer drafts (or none) takes a shorter group
        const bool act[3] = {true, true, true};
        const int nd[3] = {3, 0, 2};
        const int A = batch_pack(act, nd, 3, 0, true, kMaxT, rows, first, glen, slot, &S);
        check(A == 3 && S == 8 && glen[0] == 4 && glen[1] == 1 && glen[2] == 3, "variable groups: 4 + 1 + 3");
        check(first[0] == 0 && first[1] == 4 && first[2] == 5, "variable groups: first rows");
        const int rows_want[8] = {0, 0, 0, 0, 1, 2, 2, 2};
        for (int t = 0; t < 8; ++t) check(rows[t] == rows_want[t], "variable groups: rows layout");
    }
    {
        // depth 2, four slots: 3 + 3 = 6, the third (3 rows) does not fit in the 2 left
        const bool act[4] = {true, true, true, true};
        const int nd[4] = {2, 2, 2, 2};
        const int A = batch_pack(act, nd, 4, 1, true, kMaxT, rows, first, glen, slot, &S);
        check(A == 2 && S == 6 && slot[0] == 1 && slot[1] == 2, "depth 2: two of four slots from slot 1");
    }
    {
        // nothing active
        const bool act[2] = {false, false};
        const int nd[2] = {1, 1};
        const int A = batch_pack(act, nd, 2, 0, true, kMaxT, rows, first, glen, slot, &S);
        check(A == 0 && S == 0, "no active slot: empty window");
    }

    // ---- acceptance: depth 1 (glen 2) equals the 0.1.44 rule over a grid of conditions
    {
        const int32_t toks[2][2] = {{10, 11}, {10, 12}};
        for (int match = 0; match < 2; ++match)
            for (int eos = 0; eos < 2; ++eos)
                for (int stop = 0; stop < 2; ++stop)
                    for (long long produced = 1; produced <= 3; ++produced)
                        for (long long p = 1; p <= 4; ++p) {
                            const int32_t outb[2] = {eos ? 7 : 11, 99};
                            const int32_t* tok = toks[match ? 0 : 1];
                            const int want = keep_0144(outb, tok, eos != 0, stop != 0, produced, 3, p, 5);
                            const int got = batch_keep(outb, tok, 2, eos_7, stop != 0, produced, 3, p, 5);
                            if (eos && match) check(want == 1 && got == 1, "depth 1: eos pick is not a kept draft");
                            check(got == want, "depth 1: equals the 0.1.44 rule");
                        }
    }
    // ---- acceptance: chains
    {
        const int32_t tok[4] = {10, 11, 12, 13};
        const int32_t all[4] = {11, 12, 13, 14};
        check(batch_keep(all, tok, 4, eos_7, false, 1, 100, 10, 1000) == 4, "chain: all three drafts kept");
        check(batch_keep(all, tok, 3, eos_7, false, 1, 100, 10, 1000) == 3, "chain: a two-draft group");
        check(batch_keep(all, tok, 1, eos_7, false, 1, 100, 10, 1000) == 1, "chain: a one-row group keeps one");
        const int32_t miss1[4] = {11, 99, 13, 14};
        check(batch_keep(miss1, tok, 4, eos_7, false, 1, 100, 10, 1000) == 2, "chain: mismatch at draft 2 keeps 2");
        const int32_t miss0[4] = {99, 12, 13, 14};
        check(batch_keep(miss0, tok, 4, eos_7, false, 1, 100, 10, 1000) == 1, "chain: mismatch at draft 1 keeps 1");
        const int32_t late[4] = {11, 12, 99, 14};
        check(batch_keep(late, tok, 4, eos_7, false, 1, 100, 10, 1000) == 3, "chain: mismatch at draft 3 keeps 3");
        // an end token picked after row 1: row 1 kept (its draft matched), row 2 not (the eos is emitted, not fed)
        const int32_t tok_e[4] = {10, 11, 7, 13};
        const int32_t out_e[4] = {11, 7, 13, 14};
        check(batch_keep(out_e, tok_e, 4, eos_7, false, 1, 100, 10, 1000) == 2, "chain: eos after row 1 keeps 2");
        // cancel: nothing beyond row 0
        check(batch_keep(all, tok, 4, eos_7, true, 1, 100, 10, 1000) == 1, "chain: cancel keeps 1");
        // max_new: produced 1 of 3 -> at most 2 more tokens -> keep 2
        check(batch_keep(all, tok, 4, eos_7, false, 1, 3, 10, 1000) == 2, "chain: max_new bounds the kept rows");
        check(batch_keep(all, tok, 4, eos_7, false, 1, 2, 10, 1000) == 1, "chain: max_new reached keeps 1");
        check(batch_keep(all, tok, 4, eos_7, false, 1, 4, 10, 1000) == 3, "chain: max_new 4 keeps 3");
        // context: p + keep + 1 <= max_context
        check(batch_keep(all, tok, 4, eos_7, false, 1, 100, 10, 15) == 4, "chain: context 15 at p 10 keeps 4");
        check(batch_keep(all, tok, 4, eos_7, false, 1, 100, 10, 14) == 3, "chain: context 14 at p 10 keeps 3");
        check(batch_keep(all, tok, 4, eos_7, false, 1, 100, 10, 13) == 2, "chain: context 13 at p 10 keeps 2");
        check(batch_keep(all, tok, 4, eos_7, false, 1, 100, 10, 12) == 1, "chain: context 12 at p 10 keeps 1");
        // the kept prefix never exceeds the group
        check(batch_keep(all, tok, 2, eos_7, false, 1, 100, 10, 1000) == 2, "chain: keep <= glen");
    }

    if (failures == 0) std::printf("batch_depth_test: ok\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
