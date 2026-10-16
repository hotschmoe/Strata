// src/kernels/ep_night_bench.cpp (night lab, SYCL port only) - the EP decode MoE pieces on ONE card, synthetic.
//
//   ep_night_bench grouped [reps]   routed grouped expert calls (native_expert_grouped) at the EP verify shapes:
//                                   48 layers per graph, each with its own experts (weights stream from DRAM),
//                                   gate/up+SwiGLU phase, down phase and the whole call; us per call and GB/s.
//   ep_night_bench send             STRATA_NIGHT_EP_PSUM: ep_send_rows + combine (old) against ep_send_psum +
//                                   combine_ep (new); max abs / rel error and timings (the peer buffer local or host).
//
// Random bytes are valid codes for every expert format here; only the fp16 block scales are set (small, finite).
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/ep_kernels.hpp"
#include <type_traits>
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {

const char* name_of(int t) {
    switch (t) {
        case 16: return "IQ2_XXS"; case 17: return "IQ2_XS"; case 18: return "IQ3_XXS"; case 20: return "IQ4_NL";
        case 21: return "IQ3_S"; case 22: return "IQ2_S"; case 42: return "Q2_0"; default: return "?";
    }
}
int block_values(int t) { return t == 20 ? 32 : t == 42 ? 64 : 256; }

std::vector<uint8_t> random_rows(int t, int64_t rows, int64_t n, std::mt19937& rng) {
    const size_t rb = k::iq_row_bytes(t, n), bs = k::iq_row_bytes(t, block_values(t));
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255), ex(2, 8), man(0, 1023), sgn(0, 1);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        const uint16_t h = (uint16_t) ((sgn(rng) ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));
        std::memcpy(&w[o], &h, 2);
    }
    return w;
}

sycl::queue& Q() { static sycl::queue* q = dpct::get_current_device().create_queue(true); return *q; }

template <class F>
double time_graph_us(sycl::queue& q, F&& record, int calls, int reps) {
    namespace ex = sycl::ext::oneapi::experimental;
    ex::command_graph<ex::graph_state::modifiable> g(q.get_context(), q.get_device());
    g.begin_recording(q);
    record();
    g.end_recording(q);
    auto ge = g.finalize();
    for (int i = 0; i < 5; ++i) q.ext_oneapi_graph(ge);
    q.wait();
    std::vector<double> t;
    for (int trial = 0; trial < 5; ++trial) {
        const auto a = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) q.ext_oneapi_graph(ge);
        q.wait();
        const auto b = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double, std::micro>(b - a).count() / ((double) reps * calls));
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// ------------------------------------------------------------------------------------------------ grouped
struct Scenario {
    const char* what;
    int T;                  // window rows
    std::vector<int> sizes; // entries per group (group = one expert)
};

struct Grouped {
    k::NativeExpertLayout L;
    int T, K = 10, cap, ng, NL;
    size_t slot;
    uint8_t* arena = nullptr;
    std::vector<unsigned long long*> lptr;
    uint8_t* xq = nullptr; uint8_t* scr = nullptr; float* out = nullptr;
    int32_t *start = nullptr, *n = nullptr, *dst = nullptr, *tok = nullptr;
    Grouped(int gu, int dt, const Scenario& sc, int nl, std::mt19937& rng, sycl::queue& q) {
        L = k::native_expert_layout(gu, dt, 2560, 640);
        T = sc.T; cap = T * K; ng = (int) sc.sizes.size(); NL = nl;
        slot = (L.bytes + 255) & ~(size_t) 255;
        const int nb = 8;   // distinct random blobs, replicated over the arena
        std::vector<std::vector<uint8_t>> hb;
        for (int b = 0; b < nb; ++b) {
            auto blob = random_rows(gu, 2 * 640, 2560, rng);
            const auto down = random_rows(dt, 2560, 640, rng);
            blob.insert(blob.end(), down.begin(), down.end());
            hb.push_back(std::move(blob));
        }
        arena = sycl::malloc_device<uint8_t>(slot * (size_t) (nl * ng) + 256, q);
        for (int l = 0; l < nl; ++l) {
            std::vector<unsigned long long> p;
            for (int j = 0; j < ng; ++j) {
                uint8_t* d = arena + slot * (size_t) (l * ng + j);
                q.memcpy(d, hb[(size_t) ((l * ng + j) % nb)].data(), L.bytes);
                p.push_back((unsigned long long) d);
            }
            auto* dp = sycl::malloc_device<unsigned long long>(std::max(ng, 1), q);
            if (ng) q.memcpy(dp, p.data(), p.size() * 8);
            q.wait();
            lptr.push_back(dp);
        }
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) T * 2560);
        for (auto& v : x) v = nd(rng);
        float* dx = sycl::malloc_device<float>(x.size(), q);
        q.memcpy(dx, x.data(), x.size() * 4).wait();
        xq = sycl::malloc_device<uint8_t>((size_t) T * 80 * 36, q);
        k::quantize_q8_1_rows(dx, T, 2560, xq, &q);
        q.wait();
        sycl::free(dx, q);
        scr = sycl::malloc_device<uint8_t>(k::native_expert_scratch_bytes(cap, 640), q);
        out = sycl::malloc_device<float>((size_t) cap * 2560, q);
        start = sycl::malloc_device<int32_t>(cap + 1, q);
        n = sycl::malloc_device<int32_t>(1, q);
        dst = sycl::malloc_device<int32_t>(cap, q);
        tok = sycl::malloc_device<int32_t>(cap, q);
        // the plan: groups in order, each group's tokens distinct, destinations scattered
        std::vector<int32_t> st(1, 0), tk, ds(cap);
        std::iota(ds.begin(), ds.end(), 0);
        std::shuffle(ds.begin(), ds.end(), rng);
        for (int m : sc.sizes) {
            std::vector<int> tt(T);
            std::iota(tt.begin(), tt.end(), 0);
            std::shuffle(tt.begin(), tt.end(), rng);
            for (int i = 0; i < m; ++i) tk.push_back(tt[(size_t) (i % T)]);
            st.push_back(st.back() + m);
        }
        tk.resize(cap, 0);
        q.memcpy(start, st.data(), st.size() * 4);
        q.memcpy(n, &ng, 4);
        q.memcpy(dst, ds.data(), ds.size() * 4);
        q.memcpy(tok, tk.data(), tk.size() * 4);
        q.wait();
    }
    ~Grouped() {
        sycl::queue& q = Q();
        for (auto* p : lptr) sycl::free(p, q);
        for (void* p : {(void*) arena, (void*) xq, (void*) scr, (void*) out, (void*) start, (void*) n, (void*) dst,
                        (void*) tok})
            sycl::free(p, q);
    }
    // the whole call once from a NaN-filled output; the output bits
    std::vector<uint32_t> result(sycl::queue& q, int rows, int iq = 0, int layer = 0) {
        q.memset(out, 0xFF, (size_t) cap * 2560 * 4).wait();
        q.memset(scr, 0x3C, k::native_expert_scratch_bytes(cap, 640)).wait();
        k::native_expert_set_night_rows(rows);
        k::native_expert_set_night_iq(iq);
        k::native_expert_grouped(L, lptr[(size_t) layer], start, n, dst, tok, cap, cap, xq, scr, out, &q, 0);
        k::native_expert_set_night_rows(0);
        k::native_expert_set_night_iq(0);
        q.wait();
        std::vector<uint32_t> o((size_t) cap * 2560);
        q.memcpy(o.data(), out, o.size() * 4).wait();
        return o;
    }
    double run(sycl::queue& q, int phase, int64_t gy, int reps, int rows = 0, int iq = 0) {
        k::native_expert_set_night_rows(rows);
        k::native_expert_set_night_iq(iq);
        k::native_expert_set_mode(-1, phase);
        const double us = time_graph_us(q, [&] {
            for (int l = 0; l < NL; ++l)
                k::native_expert_grouped(L, lptr[(size_t) l], start, n, dst, tok, cap, cap, xq, scr, out, &q, gy);
        }, NL, reps);
        k::native_expert_set_mode(-1, 0);
        k::native_expert_set_night_rows(0);
        k::native_expert_set_night_iq(0);
        return us;
    }
};

std::vector<int> sizes_of(std::initializer_list<std::pair<int, int>> v) {   // {count, size} pairs
    std::vector<int> s;
    for (auto [c, m] : v) for (int i = 0; i < c; ++i) s.push_back(m);
    return s;
}

int g_bad = 0;
int grouped_bench(int reps) {
    sycl::queue& q = Q();
    if (const char* t = std::getenv("EPB_NLTAB")) k::native_expert_set_night_iq4nl_tab(std::atoi(t));
    if (const char* t = std::getenv("EPB_ALU")) k::native_expert_set_night_iq_alu(std::atoi(t));
    std::mt19937 rng(1016);
    const Scenario scs[] = {
        {"E1 30g x1 (T8)", 8, sizes_of({{30, 1}})},
        {"E2 15g x2 (T8)", 8, sizes_of({{15, 2}})},
        {"E3 10g x3 (T8)", 8, sizes_of({{10, 3}})},
        {"C1 card0 10g/20e", 4, sizes_of({{2, 3}, {6, 2}, {2, 1}})},
        {"C1 card1 18g/20e", 4, sizes_of({{2, 2}, {16, 1}})},
        {"C2 card0 14g/30e", 6, sizes_of({{1, 4}, {3, 3}, {7, 2}, {3, 1}})},
        {"C2 card1 26g/30e", 6, sizes_of({{4, 2}, {22, 1}})},
        {"C4 card0 19g/40e", 8, sizes_of({{2, 4}, {4, 3}, {7, 2}, {6, 1}})},
        {"C4 card1 32g/40e", 8, sizes_of({{8, 2}, {24, 1}})},
    };
    const int pairs[][2] = {{16, 42}, {17, 20}, {21, 42}, {22, 42}, {18, 20}};
    // EPB_PAIRS="16:42,21:42" / EPB_SCEN="0123" restrict the table (default: all)
    const char* fp = std::getenv("EPB_PAIRS");
    const char* fs = std::getenv("EPB_SCEN");
    std::printf("grouped: us per call (48 layers per graph, median of 5 x %d replays); base = current kernels, R2 =\n"
                "STRATA_NIGHT_EP_ROWS=2, IQ = STRATA_NIGHT_IQ=1 (1 row), IQR2 = both; gu = gate/up + swiglu phase, dn = all - gu;\n"
                "GB/s = groups x blob bytes / all; every variant compared bitwise with base\n", reps);
    std::printf("%-8s/%-7s %-17s | %7s %7s %7s | %7s %7s | %7s %7s %7s | %7s %7s %7s | %6s %6s | %s\n", "gate/up", "down",
                "scenario", "gu", "dn", "all", "R2:gu", "R2:all", "IQ:gu", "IQ:dn", "IQ:all", "IQR2:gu", "IQR2:dn", "IQR2:all",
                "GB/s", "IQGB/s", "bitwise");
    for (const auto& pr : pairs)
        for (const auto& sc : scs) {
            const int si = (int) (&sc - scs);
            if (fp && !std::strstr(fp, (std::to_string(pr[0]) + ":" + std::to_string(pr[1])).c_str())) continue;
            if (fs && !std::strchr(fs, '0' + si)) continue;
            Grouped G(pr[0], pr[1], sc, 48, rng, q);
            const double gu = G.run(q, 1, 0, reps), all = G.run(q, 0, 0, reps);
            const double g2 = G.run(q, 1, 0, reps, 2), a2 = G.run(q, 0, 0, reps, 2);
            const double gi = G.run(q, 1, 0, reps, 0, 1), ai = G.run(q, 0, 0, reps, 0, 1);
            const double gj = G.run(q, 1, 0, reps, 2, 1), aj = G.run(q, 0, 0, reps, 2, 1);
            size_t diff = 0, written = 0;
            for (int layer : {0, 17, 47}) {   // three layers' blobs
                const auto a = G.result(q, 0, 0, layer), b = G.result(q, 2, 0, layer), c = G.result(q, 0, 1, layer),
                           d = G.result(q, 2, 1, layer);
                for (size_t i = 0; i < a.size(); ++i) {
                    diff += (a[i] != b[i]) + (a[i] != c[i]) + (a[i] != d[i]);
                    written += a[i] != 0xFFFFFFFFu;
                }
            }
            g_bad += diff != 0 || written == 0;
            const double mb = (double) G.ng * (double) G.L.bytes / 1e6;
            std::printf("%-8s/%-7s %-17s | %7.2f %7.2f %7.2f | %7.2f %7.2f | %7.2f %7.2f %7.2f | %7.2f %7.2f %7.2f | %6.1f %6.1f | %s (%zu, %zu)\n",
                        name_of(pr[0]), name_of(pr[1]), sc.what, gu, all - gu, all, g2, a2, gi, ai - gi, ai, gj, aj - gj, aj,
                        mb * 1e3 / all, mb * 1e3 / std::min(ai, aj), diff == 0 && written ? "yes" : "FAIL", written, diff);
            std::fflush(stdout);
        }
    std::printf("grouped: %d failures\n", g_bad);
    return g_bad ? 1 : 0;
}

// ------------------------------------------------------------------------------------------------ send (PSUM)
struct Err { double max_abs = 0, max_rel = 0, ref_abs = 0; };
Err compare(const std::vector<float>& a, const std::vector<float>& b) {
    Err e;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs((double) a[i] - (double) b[i]);
        e.max_abs = std::max(e.max_abs, d);
        e.ref_abs = std::max(e.ref_abs, std::fabs((double) a[i]));
        if (std::fabs(a[i]) > 1e-3) e.max_rel = std::max(e.max_rel, d / std::fabs((double) a[i]));
        if (!std::isfinite(b[i]) && std::isfinite(a[i])) { e.max_abs = INFINITY; }
    }
    return e;
}

int send_bench(int reps) {
    sycl::queue& q = Q();
    std::mt19937 rng(77);
    const int K = 10, NE = 512, NRES0 = 145;
    const int64_t N = 2560;
    int fails = 0;
    for (const int T : {6, 4, 1}) {
        const int cap = T * K;
        std::vector<int> perm(NE);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        std::vector<int32_t> res0(NE, -1);
        for (int i = 0; i < NRES0; ++i) res0[(size_t) perm[(size_t) i]] = i;
        std::vector<int32_t> ids(cap);
        std::vector<float> w(cap), sg(T);
        std::uniform_real_distribution<float> uw(0.02f, 1.0f), u01(0.f, 1.f);
        std::normal_distribution<float> nd(0.f, 1.f);
        int own = 0;
        for (int t = 0; t < T; ++t) {
            std::vector<int> used;
            float ws = 0;
            for (int j = 0; j < K; ++j) {
                int e;
                do {   // about half the entries on card 0 (its hot experts)
                    e = u01(rng) < 0.5f ? perm[(size_t) (rng() % NRES0)] : perm[(size_t) (NRES0 + rng() % (NE - NRES0))];
                } while (std::find(used.begin(), used.end(), e) != used.end());
                used.push_back(e);
                ids[(size_t) (t * K + j)] = e;
                own += res0[(size_t) e] >= 0;
                w[(size_t) (t * K + j)] = uw(rng);
                ws += w[(size_t) (t * K + j)];
            }
            for (int j = 0; j < K; ++j) w[(size_t) (t * K + j)] /= ws;
            sg[(size_t) t] = nd(rng);
        }
        std::vector<float> rows1((size_t) cap * N), ownr((size_t) cap * N), shared((size_t) T * N);
        for (auto& v : rows1) v = 0.05f * nd(rng);
        for (auto& v : ownr) v = 0.05f * nd(rng);
        for (auto& v : shared) v = 0.05f * nd(rng);
        std::vector<float> parts_init((size_t) cap * N, std::nanf(""));
        for (int e = 0; e < cap; ++e)
            if (res0[(size_t) ids[(size_t) e]] >= 0)
                std::memcpy(&parts_init[(size_t) e * N], &ownr[(size_t) e * N], N * 4);
        auto dev = [&](const void* h, size_t bytes) {
            void* d = sycl::malloc_device(bytes + 256, q);
            q.memcpy(d, h, bytes).wait();
            return d;
        };
        auto* d_ids = (int32_t*) dev(ids.data(), ids.size() * 4);
        auto* d_w = (float*) dev(w.data(), w.size() * 4);
        auto* d_sg = (float*) dev(sg.data(), sg.size() * 4);
        auto* d_res0 = (int32_t*) dev(res0.data(), res0.size() * 4);
        auto* d_rows1 = (float*) dev(rows1.data(), rows1.size() * 4);
        auto* d_shared = (float*) dev(shared.data(), shared.size() * 4);
        auto* d_pold = (float*) dev(parts_init.data(), parts_init.size() * 4);
        auto* d_pnew = (float*) dev(parts_init.data(), parts_init.size() * 4);
        auto* d_psum = sycl::malloc_device<float>((size_t) T * N, q);
        auto* d_oold = sycl::malloc_device<float>((size_t) T * N, q);
        auto* d_onew = sycl::malloc_device<float>((size_t) T * N, q);
        auto* d_ctl = sycl::malloc_device<uint32_t>(64, q);   // done, flag, ctr
        q.memset(d_ctl, 0, 256).wait();
        uint32_t *done = d_ctl, *flag = d_ctl + 16, *ctr = d_ctl + 32;
        // host reference (double): sum_j w * row + shared (plain) / + shared * sigmoid(sg) (gated)
        std::vector<float> ref((size_t) T * N), refg((size_t) T * N);
        for (int t = 0; t < T; ++t)
            for (int64_t c = 0; c < N; ++c) {
                double s = 0;
                for (int j = 0; j < K; ++j) {
                    const size_t e = (size_t) (t * K + j);
                    const float v = res0[(size_t) ids[e]] >= 0 ? ownr[e * N + c] : rows1[e * N + c];
                    s += (double) w[e] * v;
                }
                ref[(size_t) t * N + c] = (float) (s + shared[(size_t) t * N + c]);
                refg[(size_t) t * N + c] = (float) (s + shared[(size_t) t * N + c] / (1.0 + std::exp(-(double) sg[(size_t) t])));
            }
        for (const bool gated : {false, true}) {
            q.memcpy(d_pold, parts_init.data(), parts_init.size() * 4).wait();
            k::ep_send_rows(d_pold, d_rows1, d_ids, d_res0, cap, N, done, flag, ctr, &q);
            if (gated) k::native_moe_combine_multi_hits_gated(d_pold, d_w, d_shared, d_sg, d_oold, N, K, T, &q);
            else if (T > 1) k::native_moe_combine_multi(d_pold, d_w, d_shared, d_oold, N, K, T, &q);
            else k::native_moe_combine(d_pold, d_w, d_shared, d_oold, N, K, &q);
            k::ep_send_psum(d_psum, d_rows1, d_ids, d_w, d_res0, T, K, N, done, flag, ctr, &q);
            k::ep_combine_psum(d_pnew, d_w, d_ids, d_res0, d_psum, d_shared, gated ? d_sg : nullptr, d_onew, N, K, T, &q);
            q.wait();
            std::vector<float> o_old((size_t) T * N), o_new((size_t) T * N);
            q.memcpy(o_old.data(), d_oold, o_old.size() * 4);
            q.memcpy(o_new.data(), d_onew, o_new.size() * 4).wait();
            uint32_t ctl[64];
            q.memcpy(ctl, d_ctl, 256).wait();
            const Err on = compare(o_old, o_new), ro = compare(gated ? refg : ref, o_old), rn = compare(gated ? refg : ref, o_new);
            const bool ok = on.max_abs <= 4e-6 * std::max(1.0, on.ref_abs) * 8 && ctl[0] == 0;
            fails += !ok;
            std::printf("psum parity T=%d %s: own entries %d/%d; old vs new max|d| %.3g (max rel %.3g, |out| max %.3g); "
                        "vs fp64 ref: old %.3g new %.3g; done=%u %s\n",
                        T, gated ? "gated" : "plain", own, cap, on.max_abs, on.max_rel, on.ref_abs, ro.max_abs, rn.max_abs,
                        ctl[0], ok ? "OK" : "FAIL");
        }
        // timings: 48 calls per graph; the peer destination in device memory (local) or pinned host memory (PCIe)
        if (T > 1) {
            float* h_parts = sycl::malloc_host<float>((size_t) cap * N, q);
            float* h_psum = sycl::malloc_host<float>((size_t) T * N, q);
            const double s_old_d = time_graph_us(q, [&] { for (int i = 0; i < 48; ++i) k::ep_send_rows(d_pold, d_rows1, d_ids, d_res0, cap, N, done, flag, ctr, &q); }, 48, reps);
            const double s_new_d = time_graph_us(q, [&] { for (int i = 0; i < 48; ++i) k::ep_send_psum(d_psum, d_rows1, d_ids, d_w, d_res0, T, K, N, done, flag, ctr, &q); }, 48, reps);
            const double s_old_h = time_graph_us(q, [&] { for (int i = 0; i < 48; ++i) k::ep_send_rows(h_parts, d_rows1, d_ids, d_res0, cap, N, done, flag, ctr, &q); }, 48, reps);
            const double s_new_h = time_graph_us(q, [&] { for (int i = 0; i < 48; ++i) k::ep_send_psum(h_psum, d_rows1, d_ids, d_w, d_res0, T, K, N, done, flag, ctr, &q); }, 48, reps);
            const double c_old = time_graph_us(q, [&] { for (int i = 0; i < 48; ++i) k::native_moe_combine_multi(d_pold, d_w, d_shared, d_oold, N, K, T, &q); }, 48, reps);
            const double c_new = time_graph_us(q, [&] { for (int i = 0; i < 48; ++i) k::ep_combine_psum(d_pnew, d_w, d_ids, d_res0, d_psum, d_shared, nullptr, d_onew, N, K, T, &q); }, 48, reps);
            const int peer = cap - own;
            std::printf("psum timing T=%d (peer entries %d): send old/new  device %.2f / %.2f us  host(PCIe) %.2f / %.2f us "
                        "(%.0f / %.0f KB); combine old/new %.2f / %.2f us\n",
                        T, peer, s_old_d, s_new_d, s_old_h, s_new_h, peer * N * 4 / 1024.0, T * N * 4 / 1024.0, c_old, c_new);
            sycl::free(h_parts, q);
            sycl::free(h_psum, q);
        }
        for (void* p : {(void*) d_ids, (void*) d_w, (void*) d_sg, (void*) d_res0, (void*) d_rows1, (void*) d_shared,
                        (void*) d_pold, (void*) d_pnew, (void*) d_psum, (void*) d_oold, (void*) d_onew, (void*) d_ctl})
            sycl::free(p, q);
    }
    std::printf("send: %d failures\n", fails);
    return fails ? 1 : 0;
}

// ------------------------------------------------------------------------------------------------ plan (PLAN1)
// resident_plan_ep_pair (both roles in one launch on card 0) against resident_plan_ep role 1 and role 2 (as each card
// makes it today), word for word over the words native_expert_grouped reads; and the launch times.
int plan_bench(int reps) {
    sycl::queue& q = Q();
    std::mt19937 rng(4242);
    const int K = 10, NE = 512, MT = 8;
    const long long capx = (long long) MT * K;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    const size_t words = (size_t) (ptr_off + 4 * capx + capx + 1 + 1 + 16);
    const int used = (int) (ptr_off + 2 * capx);
    auto* d_ids = sycl::malloc_device<int32_t>(capx, q);
    auto* d_r0 = sycl::malloc_device<int32_t>(NE, q);
    auto* d_r1 = sycl::malloc_device<int32_t>(NE, q);
    auto* d_off0 = sycl::malloc_device<unsigned long long>(NE, q);
    auto* d_off1 = sycl::malloc_device<unsigned long long>(NE, q);
    int32_t* pl[4];
    for (auto& p : pl) { p = sycl::malloc_device<int32_t>(words, q); }
    auto* d_err = sycl::malloc_device<uint32_t>(4, q);
    const uint8_t* base0 = reinterpret_cast<const uint8_t*>(uintptr_t(0x100000000ull));   // never dereferenced
    const uint8_t* base1 = reinterpret_cast<const uint8_t*>(uintptr_t(0x200000000ull));
    int fails = 0, cases = 0;
    for (int it = 0; it < 200; ++it) {
        const int T = 1 + (int) (rng() % MT), n = T * K;
        std::vector<int> perm(NE);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        std::vector<int32_t> r0(NE, -1), r1(NE, -1);
        const int n0 = 100 + (int) (rng() % 100);
        for (int i = 0; i < NE; ++i) { if (i < n0) r0[(size_t) perm[(size_t) i]] = i; else r1[(size_t) perm[(size_t) i]] = i - n0; }
        for (int i = 0; i < 20; ++i) r1[(size_t) perm[(size_t) i]] = 400 + i;   // some experts on both cards
        std::vector<int32_t> ids((size_t) n);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < K; ++j) {
                int e;
                bool dup;
                do { e = (it % 3 == 0) ? perm[(size_t) (rng() % 40)] : (int) (rng() % NE);   // hot sets: repeats across tokens
                     dup = false; for (int jj = 0; jj < j; ++jj) dup |= ids[(size_t) (t * K + jj)] == e; } while (dup);
                ids[(size_t) (t * K + j)] = e;
            }
        std::vector<unsigned long long> off0(NE), off1(NE);
        for (int i = 0; i < NE; ++i) { off0[(size_t) i] = (unsigned long long) i * 1868800ull + 4096; off1[(size_t) i] = (unsigned long long) i * 1305600ull; }
        q.memcpy(d_ids, ids.data(), (size_t) n * 4);
        q.memcpy(d_r0, r0.data(), NE * 4);
        q.memcpy(d_r1, r1.data(), NE * 4);
        q.memcpy(d_off0, off0.data(), NE * 8);
        q.memcpy(d_off1, off1.data(), NE * 8);
        for (auto* p : pl) q.memset(p, 0x5A, words * 4);
        q.memset(d_err, 0, 16).wait();
        const bool use_off = it % 2 == 0;
        k::resident_plan_ep(d_ids, n, K, d_r0, d_r1, 1, NE, base0, use_off ? d_off0 : nullptr, 1868800, pl[0], capx, &q, d_err);
        k::resident_plan_ep(d_ids, n, K, d_r0, d_r1, 2, NE, base1, use_off ? d_off1 : nullptr, 1305600, pl[1], capx, &q, d_err + 1);
        k::resident_plan_ep_pair(d_ids, n, K, d_r0, d_r1, NE, base0, use_off ? d_off0 : nullptr, 1868800, pl[2], base1,
                                 use_off ? d_off1 : nullptr, 1305600, pl[3], capx, &q, d_err + 2);
        q.wait();
        std::vector<int32_t> h[4];
        for (int i = 0; i < 4; ++i) { h[i].resize(words); q.memcpy(h[i].data(), pl[i], words * 4).wait(); }
        // the words the grouped call reads: counts, start, dst, tok over the used groups / entries, ptr of the used groups
        auto cmp = [&](const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
            const int ng = a[0], ne = a[1];
            bool ok = a[0] == b[0] && a[1] == b[1];
            for (int g = 0; g <= ng && ok; ++g) ok = a[(size_t) (4 + g)] == b[(size_t) (4 + g)];
            for (int e = 0; e < ne && ok; ++e)
                ok = a[(size_t) (4 + capx + 1 + e)] == b[(size_t) (4 + capx + 1 + e)] &&
                     a[(size_t) (4 + 2 * capx + 1 + e)] == b[(size_t) (4 + 2 * capx + 1 + e)];
            for (int g = 0; g < ng && ok; ++g)
                ok = a[(size_t) (ptr_off + 2 * g)] == b[(size_t) (ptr_off + 2 * g)] && a[(size_t) (ptr_off + 2 * g + 1)] == b[(size_t) (ptr_off + 2 * g + 1)];
            bool full = true;   // and every word up to `used`
            for (int i = 0; i < used; ++i) full &= a[(size_t) i] == b[(size_t) i];
            return std::make_pair(ok, full);
        };
        const auto c0 = cmp(h[0], h[2]), c1 = cmp(h[1], h[3]);
        ++cases;
        if (!c0.first || !c1.first) {
            ++fails;
            if (fails < 5) std::printf("plan pair case %d (T=%d): role1 %s role2 %s\n", it, T, c0.first ? "ok" : "DIFF", c1.first ? "ok" : "DIFF");
        }
        if (it < 3) std::printf("plan pair case %d: T=%d groups r1=%d/%d r2=%d/%d entries %d+%d; all %d words equal: %s/%s\n", it, T,
                                h[0][0], h[2][0], h[1][0], h[3][0], h[0][1], h[1][1], used, c0.second ? "yes" : "no", c1.second ? "yes" : "no");
    }
    // launch cost: 48 calls in a graph (T = 6)
    const double t1 = time_graph_us(q, [&] { for (int i = 0; i < 48; ++i) k::resident_plan_ep(d_ids, 60, K, d_r0, d_r1, 1, NE, base0, d_off0, 1868800, pl[0], capx, &q, d_err); }, 48, reps);
    const double t2 = time_graph_us(q, [&] { for (int i = 0; i < 48; ++i) k::resident_plan_ep_pair(d_ids, 60, K, d_r0, d_r1, NE, base0, d_off0, 1868800, pl[2], base1, d_off1, 1305600, pl[3], capx, &q, d_err); }, 48, reps);
    std::printf("plan: %d cases, %d failures; launch (in-order graph, back to back) role-1 only %.2f us, pair %.2f us\n", cases, fails, t1, t2);
    for (void* p : {(void*) d_ids, (void*) d_r0, (void*) d_r1, (void*) d_off0, (void*) d_off1, (void*) d_err}) sycl::free(p, q);
    for (auto* p : pl) sycl::free(p, q);
    return fails ? 1 : 0;
}

// ------------------------------------------------------------------------------------------------ router (TOPK10)
int router_bench(int reps) {
    sycl::queue& q = Q();
    std::mt19937 rng(5150);
    const int NE = 512, MAXT = 8, TRIALS = 400;
    float* d_l = sycl::malloc_device<float>((size_t) MAXT * NE * 48, q);
    int32_t* d_i[2];
    float* d_w[2];
    for (int v = 0; v < 2; ++v) { d_i[v] = sycl::malloc_device<int32_t>(MAXT * 10 * 48, q); d_w[v] = sycl::malloc_device<float>(MAXT * 10 * 48, q); }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> pick(0, NE - 1);
    int bad = 0;
    long checked = 0;
    for (int trial = 0; trial < TRIALS; ++trial) {
        const int T = 1 + trial % MAXT;
        std::vector<float> l((size_t) T * NE);
        const int kind = trial % 5;
        for (auto& v : l) v = (kind == 0 ? 0.5f : kind == 1 ? 3.0f : 1.5f) * nd(rng);
        if (kind == 2) for (auto& v : l) v = std::round(v * 4.0f) / 4.0f;   // many exact ties
        if (kind == 3) for (int t = 0; t < T; ++t) for (int k = 0; k < 40; ++k) l[(size_t) t * NE + pick(rng)] = 2.5f;   // tied leaders
        if (kind == 4) for (int t = 0; t < T; ++t) for (int k = 0; k < NE; ++k) l[(size_t) t * NE + k] = (k % 7 == 0) ? 1.0f : -1.0f;
        q.memcpy(d_l, l.data(), l.size() * 4).wait();
        std::vector<int32_t> ids[2];
        std::vector<float> w[2];
        for (int v = 0; v < 2; ++v) {
            q.memset(d_i[v], 0xFF, MAXT * 40).wait();
            q.memset(d_w[v], 0xFF, MAXT * 40).wait();
            k::native_router_set_night_topk(v);
            if (T == 1 && trial % 2) k::native_router_top10(d_l, d_i[v], d_w[v], &q);
            else k::native_router_top10_multi(d_l, d_i[v], d_w[v], T, &q);
            q.wait();
            ids[v].resize((size_t) T * 10); w[v].resize((size_t) T * 10);
            q.memcpy(ids[v].data(), d_i[v], (size_t) T * 40).wait();
            q.memcpy(w[v].data(), d_w[v], (size_t) T * 40).wait();
        }
        k::native_router_set_night_topk(-1);
        checked += T;
        if (std::memcmp(ids[0].data(), ids[1].data(), (size_t) T * 40) || std::memcmp(w[0].data(), w[1].data(), (size_t) T * 40)) {
            if (bad < 5) std::printf("router trial %d (T=%d kind %d): DIFF ids0 %d ids1 %d\n", trial, T, kind, ids[0][0], ids[1][0]);
            ++bad;
        }
    }
    std::printf("router top-10: %ld token rows (ties, flat, wide, tied leaders), night vs default ids+weights bitwise: %s\n",
                checked, bad ? "FAIL" : "equal");
    for (int T : {1, 4, 6, 8}) {
        double t[2];
        for (int v = 0; v < 2; ++v) {
            k::native_router_set_night_topk(v);
            t[v] = time_graph_us(q, [&] { for (int i = 0; i < 48; ++i) k::native_router_top10_multi(d_l + (size_t) i * MAXT * NE, d_i[v] + i * MAXT * 10, d_w[v] + i * MAXT * 10, T, &q); }, 48, reps);
        }
        k::native_router_set_night_topk(-1);
        std::printf("router top-10 T=%d: default %.2f us, night %.2f us per call (48 in a graph)\n", T, t[0], t[1]);
    }
    sycl::free(d_l, q);
    for (int v = 0; v < 2; ++v) { sycl::free(d_i[v], q); sycl::free(d_w[v], q); }
    return bad ? 1 : 0;
}

// ------------------------------------------------------------------------------------------------ gdnab (GDN_AB)
int gdnab_bench(int reps) {
    sycl::queue& q = Q();
    std::mt19937 rng(77);
    const int n = 2560, hv = 48, MAXT = 8, NL = 36;
    std::normal_distribution<float> nd(0.f, 1.f);
    auto bf16 = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t) ((u + 0x7FFF + ((u >> 16) & 1)) >> 16); };
    std::vector<uint16_t> wa((size_t) NL * hv * n), wb((size_t) NL * hv * n);
    for (auto& v : wa) v = bf16(0.05f * nd(rng));
    for (auto& v : wb) v = bf16(0.05f * nd(rng));
    std::vector<float> x((size_t) MAXT * n), dt(hv), sa(hv);
    for (auto& v : x) v = nd(rng);
    for (auto& v : dt) v = nd(rng);
    for (auto& v : sa) v = -std::fabs(nd(rng));
    uint16_t* d_wa = sycl::malloc_device<uint16_t>(wa.size(), q);
    uint16_t* d_wb = sycl::malloc_device<uint16_t>(wb.size(), q);
    float* d_x = sycl::malloc_device<float>(x.size(), q);
    float* d_dt = sycl::malloc_device<float>(hv, q);
    float* d_sa = sycl::malloc_device<float>(hv, q);
    float* d_g[2]; float* d_b[2];
    for (int v = 0; v < 2; ++v) { d_g[v] = sycl::malloc_device<float>(MAXT * hv, q); d_b[v] = sycl::malloc_device<float>(MAXT * hv, q); }
    q.memcpy(d_wa, wa.data(), wa.size() * 2); q.memcpy(d_wb, wb.data(), wb.size() * 2);
    q.memcpy(d_x, x.data(), x.size() * 4); q.memcpy(d_dt, dt.data(), hv * 4); q.memcpy(d_sa, sa.data(), hv * 4); q.wait();
    int bad = 0;
    for (int T = 1; T <= MAXT; ++T)
        for (int layer : {0, 13, 35}) {
            std::vector<float> g[2], b[2];
            for (int v = 0; v < 2; ++v) {
                q.memset(d_g[v], 0xFF, MAXT * hv * 4).wait(); q.memset(d_b[v], 0xFF, MAXT * hv * 4).wait();
                k::gdn_ab_set_night(v);
                k::gdn_ab_multi(d_x, d_wa + (size_t) layer * hv * n, d_wb + (size_t) layer * hv * n, d_dt, d_sa, d_g[v], d_b[v], n, hv, T, &q);
                q.wait();
                g[v].resize((size_t) T * hv); b[v].resize((size_t) T * hv);
                q.memcpy(g[v].data(), d_g[v], (size_t) T * hv * 4).wait(); q.memcpy(b[v].data(), d_b[v], (size_t) T * hv * 4).wait();
            }
            if (std::memcmp(g[0].data(), g[1].data(), (size_t) T * hv * 4) || std::memcmp(b[0].data(), b[1].data(), (size_t) T * hv * 4)) {
                ++bad; std::printf("gdn_ab T=%d layer %d: DIFF\n", T, layer);
            }
        }
    k::gdn_ab_set_night(-1);
    std::printf("gdn_ab_multi: night vs default gate+beta bitwise for T=1..8 x 3 weight sets: %s\n", bad ? "FAIL" : "equal");
    for (int T : {1, 4, 6, 8}) {
        double t[2];
        for (int v = 0; v < 2; ++v) {
            k::gdn_ab_set_night(v);
            t[v] = time_graph_us(q, [&] { for (int l = 0; l < NL; ++l) k::gdn_ab_multi(d_x, d_wa + (size_t) l * hv * n, d_wb + (size_t) l * hv * n, d_dt, d_sa, d_g[v], d_b[v], n, hv, T, &q); }, NL, reps);
        }
        k::gdn_ab_set_night(-1);
        std::printf("gdn_ab_multi T=%d: default %.2f us, night %.2f us per call (36 layers' weights in a graph)\n", T, t[0], t[1]);
    }
    for (void* p : {(void*) d_wa, (void*) d_wb, (void*) d_x, (void*) d_dt, (void*) d_sa, (void*) d_g[0], (void*) d_g[1], (void*) d_b[0], (void*) d_b[1]}) sycl::free(p, q);
    return bad ? 1 : 0;
}

// ------------------------------------------------------------------------------------------------ gdnstep (GDN_GRF)
int gdnstep_bench(int reps) {
    sycl::queue& q = Q();
    std::mt19937 rng(99);
    const int hk = 16, hv = 48, S = 128, C = 2 * S * hk + S * hv, VD = S * hv, MAXT = 8, NL = 36, NS = 2;
    const size_t st_floats = (size_t) hv * S * S;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> u01(0.05f, 0.95f);
    float* d_st = sycl::malloc_device<float>(st_floats * NL * NS, q);
    {
        std::vector<float> st(st_floats);
        for (auto& v : st) v = 0.05f * nd(rng);
        for (int i = 0; i < NL * NS; ++i) q.memcpy(d_st + st_floats * i, st.data(), st_floats * 4);
        q.wait();
    }
    std::vector<float> h((size_t) MAXT * C), g(MAXT * hv), b(MAXT * hv), z((size_t) MAXT * VD), gam(S);
    for (auto& v : h) v = 0.1f * nd(rng);
    for (auto& v : g) v = -std::fabs(0.3f * nd(rng));
    for (auto& v : b) v = u01(rng);
    for (auto& v : z) v = nd(rng);
    for (auto& v : gam) v = 1.0f + 0.1f * nd(rng);
    float* d_h = sycl::malloc_device<float>(h.size(), q); float* d_g = sycl::malloc_device<float>(g.size(), q);
    float* d_b = sycl::malloc_device<float>(b.size(), q); float* d_z = sycl::malloc_device<float>(z.size(), q);
    float* d_gam = sycl::malloc_device<float>(S, q);
    float* d_y[2] = {sycl::malloc_device<float>((size_t) MAXT * VD, q), sycl::malloc_device<float>((size_t) MAXT * VD, q)};
    uint8_t* d_xq[2] = {sycl::malloc_device<uint8_t>((size_t) MAXT * VD / 32 * 36, q), sycl::malloc_device<uint8_t>((size_t) MAXT * VD / 32 * 36, q)};
    q.memcpy(d_h, h.data(), h.size() * 4); q.memcpy(d_g, g.data(), g.size() * 4); q.memcpy(d_b, b.data(), b.size() * 4);
    q.memcpy(d_z, z.data(), z.size() * 4); q.memcpy(d_gam, gam.data(), S * 4); q.wait();
    const float eps = 1e-6f;
    int bad = 0;
    auto run_one = [&](int v, int T, int layer, bool slots, bool xq) {
        k::gdn_grf_set_night(v);
        float* st = d_st + st_floats * (size_t) (layer * NS);
        if (slots) {
            float* sts[2] = {st, st + st_floats};
            const int r0[2] = {0, T / 2}, nr[2] = {T / 2, T - T / 2};
            if (!k::night_gdn_step_norm_slots(sts, r0, nr, 2, d_h, C, d_g, d_b, d_z, d_gam, eps, d_y[v], hk, hv, &q)) std::printf("slots launch refused\n");
        } else {
            k::gdn_step_norm_multi(st, d_h, C, d_g, d_b, d_z, d_gam, eps, d_y[v], hk, hv, T, nullptr, &q, 0, xq ? d_xq[v] : nullptr);
        }
    };
    for (int T : {2, 3, 4, 6, 8})
        for (int mode = 0; mode < 3; ++mode) {
            const bool slots = mode == 2, xq = mode == 1;
            if (slots && T < 2) continue;
            for (int v = 0; v < 2; ++v) { q.memset(d_y[v], 0xFF, (size_t) MAXT * VD * 4).wait(); q.memset(d_xq[v], 0x5A, (size_t) MAXT * VD / 32 * 36).wait(); run_one(v, T, 5, slots, xq); q.wait(); }
            std::vector<float> y0((size_t) T * VD), y1((size_t) T * VD);
            q.memcpy(y0.data(), d_y[0], y0.size() * 4).wait(); q.memcpy(y1.data(), d_y[1], y1.size() * 4).wait();
            bool same = !std::memcmp(y0.data(), y1.data(), y0.size() * 4);
            if (xq) {
                std::vector<uint8_t> a((size_t) T * VD / 32 * 36), c(a.size());
                q.memcpy(a.data(), d_xq[0], a.size()).wait(); q.memcpy(c.data(), d_xq[1], c.size()).wait();
                same = same && !std::memcmp(a.data(), c.data(), a.size());
            }
            if (!same) { ++bad; std::printf("gdn step T=%d mode %d: DIFF\n", T, mode); }
        }
    k::gdn_grf_set_night(-1);
    std::printf("gdn verify recurrence: 256-GRF vs default y (+q8_1 image) bitwise, T=2..8, per-slot / q8 / 2-slot: %s\n", bad ? "FAIL" : "equal");
    for (int T : {3, 4, 6, 8})
        for (int mode : {0, 2}) {
            double t[2];
            for (int v = 0; v < 2; ++v) {
                t[v] = time_graph_us(q, [&] { for (int l = 0; l < NL; ++l) run_one(v, T, l, mode == 2, false); }, NL, reps);
            }
            k::gdn_grf_set_night(-1);
            std::printf("gdn verify recurrence %s T=%d: default %.2f us, 256 GRF %.2f us per layer (%d layers' states in a graph)\n",
                        mode == 2 ? "2 slots, one launch" : "1 slot", T, t[0], t[1], NL);
        }
    for (void* p : {(void*) d_st, (void*) d_h, (void*) d_g, (void*) d_b, (void*) d_z, (void*) d_gam, (void*) d_y[0], (void*) d_y[1], (void*) d_xq[0], (void*) d_xq[1]}) sycl::free(p, q);
    return bad ? 1 : 0;
}

// ------------------------------------------------------------------------------------------------ t12 (VERIFY_T12)
// Each entry point that chunks above 8 rows (STRATA_NIGHT_VERIFY_T12=1): the 12-row call against the same entry called
// on rows [0, 8) and [8, 12) with offset pointers (must be memcmp-equal: the offsets are right), and for MMVQ against
// two 6-row calls (the column arithmetic of a 6- vs 8-/4-column kernel; reported).
size_t blk_bytes(int t) { return k::iq_row_bytes(t, (t == 20 || t == 8) ? 32 : t == 42 ? 64 : 256); }
std::vector<uint8_t> rand_w(int t, int64_t rows, int64_t n, std::mt19937& rng) {
    const size_t rb = k::iq_row_bytes(t, n), bs = blk_bytes(t);
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255), ex(1, 5), man(0, 1023), sgn(0, 1);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        auto h = [&] { return (uint16_t) ((sgn(rng) ? 0x8000 : 0) | (ex(rng) << 10) | man(rng)); };
        if (t == 14) { const uint16_t d = h(); std::memcpy(&w[o + 208], &d, 2); }
        else {
            const uint16_t d = h(); std::memcpy(&w[o], &d, 2);
            if (t == 12 || t == 13) { const uint16_t m = h(); std::memcpy(&w[o + 2], &m, 2); }
        }
    }
    return w;
}
template <class T> T* up(sycl::queue& q, const std::vector<T>& h) { T* d = sycl::malloc_device<T>(h.size() + 64, q); q.memcpy(d, h.data(), h.size() * sizeof(T)).wait(); return d; }
template <class T> std::vector<T> down(sycl::queue& q, const T* d, size_t n) { std::vector<T> h(n); q.memcpy(h.data(), d, n * sizeof(T)).wait(); return h; }
struct Cmp { bool same = true; double max_rel = 0; };
template <class T> Cmp cmpv(const std::vector<T>& a, const std::vector<T>& b) {
    Cmp c; c.same = !std::memcmp(a.data(), b.data(), a.size() * sizeof(T));
    if constexpr (std::is_floating_point_v<T>)
        for (size_t i = 0; i < a.size(); ++i) {
            const double d = std::fabs((double) a[i] - (double) b[i]), m = std::fabs((double) a[i]);
            if (m > 1e-6) c.max_rel = std::max(c.max_rel, d / m);
        }
    return c;
}
int t12_bench() {
    sycl::queue& q = Q();
    std::mt19937 rng(1212);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int T = 12;
    int bad = 0;
    auto report = [&](const char* what, bool same, double rel6 = -1) {
        bad += !same;
        if (rel6 >= 0) std::printf("t12 %-34s 12 rows vs [0,8)+[8,12): %s; vs 6+6 rows max rel %.3g\n", what, same ? "bitwise" : "FAIL", rel6);
        else std::printf("t12 %-34s 12 rows vs [0,8)+[8,12): %s\n", what, same ? "bitwise" : "FAIL");
    };
    // quantize
    const int NI = 2560;
    std::vector<float> x((size_t) T * NI);
    for (auto& v : x) v = nd(rng);
    float* d_x = up(q, x);
    const size_t xr = (size_t) (NI / 32) * 36;
    uint8_t* xq12 = sycl::malloc_device<uint8_t>(xr * T, q);
    uint8_t* xq84 = sycl::malloc_device<uint8_t>(xr * T, q);
    k::native_quantize_q8_1(d_x, xq12, NI, T, &q);
    k::native_quantize_q8_1(d_x, xq84, NI, 8, &q);
    k::native_quantize_q8_1(d_x + 8 * NI, xq84 + 8 * xr, NI, 4, &q);
    q.wait();
    report("native_quantize_q8_1", cmpv(down(q, xq12, xr * T), down(q, xq84, xr * T)).same);
    // MMVQ, every format the window uses
    for (int t : {12, 13, 14, 23, 21, 20, 8, 42}) {
        const int NO = 1024;
        auto w = rand_w(t, NO, NI, rng);
        uint8_t* d_w = up(q, w);
        float* y12 = sycl::malloc_device<float>((size_t) T * NO, q);
        float* y84 = sycl::malloc_device<float>((size_t) T * NO, q);
        float* y66 = sycl::malloc_device<float>((size_t) T * NO, q);
        k::native_mmvq(t, d_w, xq12, y12, NI, NO, T, &q);
        k::native_mmvq(t, d_w, xq12, y84, NI, NO, 8, &q);
        k::native_mmvq(t, d_w, xq12 + 8 * xr, y84 + 8 * NO, NI, NO, 4, &q);
        k::native_mmvq(t, d_w, xq12, y66, NI, NO, 6, &q);
        k::native_mmvq(t, d_w, xq12 + 6 * xr, y66 + 6 * NO, NI, NO, 6, &q);
        q.wait();
        const auto a = down(q, y12, (size_t) T * NO), b = down(q, y84, (size_t) T * NO), c = down(q, y66, (size_t) T * NO);
        char nm[48]; std::snprintf(nm, sizeof nm, "native_mmvq type %d 2560->1024", t);
        report(nm, cmpv(a, b).same, cmpv(a, c).max_rel);
        for (void* p : {(void*) d_w, (void*) y12, (void*) y84, (void*) y66}) sycl::free(p, q);
    }
    // BF16 multi GEMV (the router 2560 -> 512)
    {
        std::vector<uint16_t> w((size_t) 512 * NI);
        for (auto& v : w) { float f = 0.05f * nd(rng); uint32_t u; std::memcpy(&u, &f, 4); v = (uint16_t) (u >> 16); }
        uint16_t* d_w = up(q, w);
        float* y12 = sycl::malloc_device<float>((size_t) T * 512, q);
        float* y84 = sycl::malloc_device<float>((size_t) T * 512, q);
        k::bf16_gemv_fp32_mmvf_multi(d_x, NI, d_w, y12, 512, NI, 512, T, &q);
        k::bf16_gemv_fp32_mmvf_multi(d_x, NI, d_w, y84, 512, NI, 512, 8, &q);
        k::bf16_gemv_fp32_mmvf_multi(d_x + 8 * NI, NI, d_w, y84 + 8 * 512, 512, NI, 512, 4, &q);
        q.wait();
        report("bf16_gemv_fp32_mmvf_multi 2560->512", cmpv(down(q, y12, (size_t) T * 512), down(q, y84, (size_t) T * 512)).same);
        for (void* p : {(void*) d_w, (void*) y12, (void*) y84}) sycl::free(p, q);
    }
    // GDN ab + conv/l2
    {
        const int hv = 48, C = 10240;
        std::vector<uint16_t> wa((size_t) hv * NI), wb((size_t) hv * NI);
        for (auto* v : {&wa, &wb}) for (auto& e : *v) { float f = 0.05f * nd(rng); uint32_t u; std::memcpy(&u, &f, 4); e = (uint16_t) (u >> 16); }
        std::vector<float> dt(hv), sa(hv);
        for (auto& e : dt) e = nd(rng);
        for (auto& e : sa) e = -std::fabs(nd(rng));
        uint16_t *d_wa = up(q, wa), *d_wb = up(q, wb);
        float *d_dt = up(q, dt), *d_sa = up(q, sa);
        float* g12 = sycl::malloc_device<float>(T * hv, q); float* b12 = sycl::malloc_device<float>(T * hv, q);
        float* g84 = sycl::malloc_device<float>(T * hv, q); float* b84 = sycl::malloc_device<float>(T * hv, q);
        k::gdn_ab_multi(d_x, d_wa, d_wb, d_dt, d_sa, g12, b12, NI, hv, T, &q);
        k::gdn_ab_multi(d_x, d_wa, d_wb, d_dt, d_sa, g84, b84, NI, hv, 8, &q);
        k::gdn_ab_multi(d_x + 8 * NI, d_wa, d_wb, d_dt, d_sa, g84 + 8 * hv, b84 + 8 * hv, NI, hv, 4, &q);
        q.wait();
        report("gdn_ab_multi", cmpv(down(q, g12, T * hv), down(q, g84, T * hv)).same && cmpv(down(q, b12, T * hv), down(q, b84, T * hv)).same);
        std::vector<float> hist((size_t) C * 3), qkv((size_t) T * C), cw((size_t) C * 4);
        for (auto* v : {&hist, &qkv, &cw}) for (auto& e : *v) e = 0.3f * nd(rng);
        float *d_hist = up(q, hist), *d_qkv = up(q, qkv), *d_cw = up(q, cw);
        float* h12 = sycl::malloc_device<float>((size_t) T * C, q); float* h84 = sycl::malloc_device<float>((size_t) T * C, q);
        k::gdn_conv_l2_multi(d_hist, d_qkv, d_cw, h12, C, 32, 1e-6f, T, &q);
        k::gdn_conv_l2_multi(d_hist, d_qkv, d_cw, h84, C, 32, 1e-6f, 8, &q, 0, false);
        k::gdn_conv_l2_multi(d_hist, d_qkv, d_cw, h84, C, 32, 1e-6f, 4, &q, 8, false);
        q.wait();
        report("gdn_conv_l2_multi", cmpv(down(q, h12, (size_t) T * C), down(q, h84, (size_t) T * C)).same);
        for (void* p : {(void*) d_wa, (void*) d_wb, (void*) d_dt, (void*) d_sa, (void*) g12, (void*) b12, (void*) g84, (void*) b84,
                        (void*) d_hist, (void*) d_qkv, (void*) d_cw, (void*) h12, (void*) h84}) sycl::free(p, q);
    }
    // shared expert (IQ3_S gate/up 2560 -> 640, IQ4_NL down 640 -> 2560, BF16 gate row)
    {
        const int FF = 640;
        auto wg = rand_w(21, FF, NI, rng), wu = rand_w(21, FF, NI, rng), wd = rand_w(20, NI, FF, rng);
        std::vector<uint16_t> gi(NI);
        for (auto& e : gi) { float f = 0.05f * nd(rng); uint32_t u; std::memcpy(&u, &f, 4); e = (uint16_t) (u >> 16); }
        uint8_t *d_wg = up(q, wg), *d_wu = up(q, wu), *d_wd = up(q, wd);
        uint16_t* d_gi = up(q, gi);
        std::vector<uint16_t> xb((size_t) T * NI);
        for (size_t i = 0; i < xb.size(); ++i) { uint32_t u; std::memcpy(&u, &x[i], 4); xb[i] = (uint16_t) (u >> 16); }
        uint16_t* d_xb = up(q, xb);
        uint8_t* scr = sycl::malloc_device<uint8_t>(xr * 16 + (size_t) 16 * (FF / 32) * 36, q);
        float* gate = sycl::malloc_device<float>((size_t) T * FF, q); float* upb = sycl::malloc_device<float>((size_t) T * FF, q);
        float* gg[2] = {sycl::malloc_device<float>(T, q), sycl::malloc_device<float>(T, q)};
        float* out[2] = {sycl::malloc_device<float>((size_t) T * NI, q), sycl::malloc_device<float>((size_t) T * NI, q)};
        k::NativeSharedWeights nw;
        nw.gate_type = 21; nw.gate_data = d_wg; nw.up_type = 21; nw.up_data = d_wu; nw.down_type = 20; nw.down_data = d_wd;
        nw.q8_1 = scr;
        k::shared_expert_multi(T, d_x, d_xb, nw, d_gi, gate, upb, gg[0], out[0], NI, FF, &q, nullptr, 0);
        k::shared_expert_multi(8, d_x, d_xb, nw, d_gi, gate, upb, gg[1], out[1], NI, FF, &q, nullptr, 0);
        k::shared_expert_multi(4, d_x + 8 * NI, d_xb + 8 * NI, nw, d_gi, gate + 8 * FF, upb + 8 * FF, gg[1] + 8, out[1] + 8 * NI, NI, FF, &q, nullptr, 0);
        q.wait();
        report("shared_expert_multi (IQ3_S/IQ4_NL)", cmpv(down(q, out[0], (size_t) T * NI), down(q, out[1], (size_t) T * NI)).same &&
                                                     cmpv(down(q, gg[0], T), down(q, gg[1], T)).same);
        for (void* p : {(void*) d_wg, (void*) d_wu, (void*) d_wd, (void*) d_gi, (void*) d_xb, (void*) scr, (void*) gate, (void*) upb,
                        (void*) gg[0], (void*) gg[1], (void*) out[0], (void*) out[1]}) sycl::free(p, q);
    }
    // the HC read (BF16 weights: the per-layer read with inject, no pending write)
    {
        const int D = 10240, LR = 320, HCn = 4;
        auto bfv = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto& e : v) { float f = s * nd(rng); uint32_t u; std::memcpy(&u, &f, 4); e = (uint16_t) (u >> 16); } return v; };
        std::vector<float> R((size_t) T * D), wn(D);
        for (auto& e : R) e = nd(rng);
        for (auto& e : wn) e = 1.0f + 0.1f * nd(rng);
        float *d_R = up(q, R), *d_wn = up(q, wn);
        uint16_t *d_wd = up(q, bfv((size_t) LR * D, 0.02f)), *d_wu = up(q, bfv((size_t) D * LR, 0.02f)), *d_wi = up(q, bfv((size_t) HCn * D, 0.02f));
        float* lo[2]; float* rs[2]; float* io[2]; float* mx[2]; float* xn[2];
        for (int v = 0; v < 2; ++v) {
            lo[v] = sycl::malloc_device<float>((size_t) T * LR, q); rs[v] = sycl::malloc_device<float>(T * HCn, q);
            io[v] = sycl::malloc_device<float>(T * HCn, q); mx[v] = sycl::malloc_device<float>((size_t) T * 2560, q);
            xn[v] = sycl::malloc_device<float>((size_t) T * D, q);
        }
        auto args = [&](int v, int t) {
            k::FusedGrArgs a;
            a.R = d_R + (size_t) t * D; a.R_out = const_cast<float*>(a.R); a.apply = false;
            a.w_norm = d_wn; a.w_down = d_wd; a.w_up = d_wu; a.w_inject = d_wi; a.eps = 1e-6f;
            a.lo = lo[v] + t * LR; a.rs = rs[v] + t * HCn; a.inject_out = io[v] + t * HCn; a.mixed = mx[v] + (size_t) t * 2560;
            return a;
        };
        k::FusedGrArgs f0[16], f1[16];
        for (int t = 0; t < T; ++t) { f0[t] = args(0, t); f1[t] = args(1, t); }
        k::fused_gr_read_multi(f0, T, xn[0], &q);
        k::fused_gr_read_multi(f1, 8, xn[1], &q);
        k::fused_gr_read_multi(f1 + 8, 4, xn[1] + (size_t) 8 * D, &q);
        q.wait();
        report("fused_gr_read_multi (HC read)", cmpv(down(q, mx[0], (size_t) T * 2560), down(q, mx[1], (size_t) T * 2560)).same &&
                                               cmpv(down(q, io[0], T * HCn), down(q, io[1], T * HCn)).same);
        for (void* p : {(void*) d_R, (void*) d_wn, (void*) d_wd, (void*) d_wu, (void*) d_wi}) sycl::free(p, q);
        for (int v = 0; v < 2; ++v) for (void* p : {(void*) lo[v], (void*) rs[v], (void*) io[v], (void*) mx[v], (void*) xn[v]}) sycl::free(p, q);
    }
    for (void* p : {(void*) d_x, (void*) xq12, (void*) xq84}) sycl::free(p, q);
    std::printf("t12: %d failures (verify_max_t() = %d)\n", bad, k::verify_max_t());
    return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "grouped";
    if (mode == "grouped") return grouped_bench(argc > 2 ? std::atoi(argv[2]) : 20);
    if (mode == "send") return send_bench(argc > 2 ? std::atoi(argv[2]) : 20);
    if (mode == "plan") return plan_bench(argc > 2 ? std::atoi(argv[2]) : 20);
    if (mode == "router") return router_bench(argc > 2 ? std::atoi(argv[2]) : 20);
    if (mode == "gdnab") return gdnab_bench(argc > 2 ? std::atoi(argv[2]) : 20);
    if (mode == "t12") return t12_bench();
    if (mode == "gdnstep") return gdnstep_bench(argc > 2 ? std::atoi(argv[2]) : 20);
    std::fprintf(stderr, "usage: ep_night_bench grouped|send\n");
    return 2;
}
