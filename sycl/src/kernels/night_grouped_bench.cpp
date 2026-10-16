// night_grouped_bench (night lab 2026-10-16): night_grouped_f16 (grouped decode + joint_matrix expert GEMM, one launch)
// against the engine's path per expert (iq_dequant_gu_f16 / iq_dequant_f16 + oneMKL FP16 GEMM), on ONE card.
//   night_grouped_bench gu|down ty n_exp M [M ...]        (fixed M rows per expert)
//   night_grouped_bench mix gt dt T n_exp median sigma max_rows   (lognormal mix, experts <= max_rows only, GU + down)
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/night_grouped.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
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

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point a) { return std::chrono::duration<double, std::milli>(Clock::now() - a).count(); }
static uint16_t f2h(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
namespace sk = strata::kernels;

struct Ctx {
    sycl::queue* q;   // in order, profiling
    uint32_t* sink;
    uint32_t busy_iters = 1u << 20;
    sycl::event busy() {
        uint32_t* s = sink; const uint32_t n = busy_iters;
        return q->single_task<class ng_busy>([=] { uint32_t x = 1; for (uint32_t i = 0; i < n; ++i) x = x * 1664525u + 1013904223u; *s = x; });
    }
    sycl::event marker() { return q->single_task<class ng_marker>([] {}); }
    void calibrate() {
        for (int t = 0; t < 12; ++t) {
            q->wait(); auto a = Clock::now(); busy(); q->wait(); const double ms = ms_since(a);
            if (ms > 20) break;
            busy_iters = (uint32_t) std::min<double>(4.0e9, busy_iters * std::max(2.0, 30.0 / std::max(ms, 0.01)));
        }
    }
    // best of reps: host enqueue ms (queued behind the busy kernel) and GPU ms (profiling: from the busy kernel's
    // end to a marker kernel's start)
    template <class F> void time(F f, int reps, double& host, double& gpu) {
        f(); q->wait();
        host = gpu = 1e30;
        for (int r = 0; r < reps; ++r) {
            q->wait();
            sycl::event e0 = busy();
            auto b = Clock::now();
            f();
            const double h = ms_since(b);
            sycl::event e1 = marker();
            q->wait();
            const double g = (double) (e1.get_profiling_info<sycl::info::event_profiling::command_start>() -
                                       e0.get_profiling_info<sycl::info::event_profiling::command_end>()) * 1e-6;
            host = std::min(host, h); gpu = std::min(gpu, g);
        }
    }
};

static std::vector<uint8_t> rand_mat(std::mt19937& rng, int ty, int64_t rows, int64_t K) {
    const size_t rb = sk::iq_row_bytes(ty, K);
    const int blk = ty == 20 ? 32 : ty == 42 ? 64 : 256;
    const size_t bb = sk::iq_row_bytes(ty, blk);
    std::vector<uint8_t> b(rb * rows);
    for (auto& x : b) x = (uint8_t) rng();
    for (size_t o = 0; o + 1 < b.size(); o += bb) { b[o] = 0x00; b[o + 1] = 0x2C; }   // d = 0.0625
    return b;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) { std::printf("usage: see source\n"); return 2; }
    const std::string mode = argv[1];
    sycl::queue* q = new sycl::queue(dpct::get_in_order_queue().get_context(), dpct::get_in_order_queue().get_device(),
                                     {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    Ctx cx{q, sycl::malloc_device<uint32_t>(1, *q)};
    cx.calibrate();
    std::mt19937 rng(11);
    const int64_t NE = 2560, FF = 640;
    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(q, 1 << 20, err)) { std::printf("gemm init: %s\n", err.c_str()); return 2; }
    std::normal_distribution<float> nd(0.f, 1.f);

    // one projection over a list of experts with their row counts; returns nothing, prints a line
    auto run = [&](bool gu, int ty, const std::vector<int32_t>& cnt, const char* tag) {
        const int n = (int) cnt.size();
        const int64_t N = gu ? 2 * FF : NE, K = gu ? NE : FF;
        const size_t mat = sk::iq_row_bytes(ty, K) * (gu ? FF : NE);
        const size_t up_off = gu ? mat : 0, blob = gu ? 2 * mat : mat;
        std::vector<int64_t> off(n + 1, 0);
        for (int i = 0; i < n; ++i) off[i + 1] = off[i] + cnt[i];
        const int64_t rows = off[n];
        uint8_t* W = sycl::malloc_device<uint8_t>(blob * n, *q);
        for (int i = 0; i < n; ++i) {
            auto g = rand_mat(rng, ty, gu ? FF : NE, K);
            q->memcpy(W + (size_t) i * blob, g.data(), mat);
            if (gu) { auto u = rand_mat(rng, ty, FF, K); q->memcpy(W + (size_t) i * blob + up_off, u.data(), mat); }
            q->wait();
        }
        std::vector<uint16_t> hx((size_t) (rows + 64) * K);
        for (auto& x : hx) x = f2h(nd(rng) * 0.25f);
        uint16_t* X = sycl::malloc_device<uint16_t>(hx.size(), *q);
        q->memcpy(X, hx.data(), hx.size() * 2);
        uint16_t* dq = sycl::malloc_device<uint16_t>((size_t) N * K, *q);
        float* yr = sycl::malloc_device<float>((size_t) rows * N, *q);
        float* yn = sycl::malloc_device<float>((size_t) rows * N, *q);
        q->wait();
        auto ref = [&]() {
            for (int i = 0; i < n; ++i) {
                const uint8_t* b = W + (size_t) i * blob;
                if (gu) sk::iq_dequant_gu_f16(ty, b, b + up_off, FF, NE, dq, q);
                else sk::iq_dequant_f16(ty, b, NE * FF, dq, q);
                gemm.f16(X + off[i] * K, dq, yr + off[i] * N, cnt[i], N, K);
            }
        };
        // entries: each expert cut into chunks of <= 8 * mt rows; mt from the expert's own count
        static const int mt_max = [] { const char* e = std::getenv("NIGHTG_MTMAX"); return e ? std::atoi(e) : 8; }();
        auto pick_mt = [](int c) { return c <= 16 ? 2 : c <= 32 ? 4 : (c <= 64 || mt_max < 16) ? 8 : 16; };
        std::vector<sk::NightGroupedEntry> ents[4];
        for (int i = 0; i < n; ++i) {
            const int mt = pick_mt(cnt[i]);
            const int cap = sk::night_grouped_rows(mt);
            for (int r = 0; r < cnt[i]; r += cap)
                ents[mt == 2 ? 0 : mt == 4 ? 1 : mt == 8 ? 2 : 3].push_back({W + (size_t) i * blob, off[i] + r, std::min(cap, cnt[i] - r), 0});
        }
        sk::NightGroupedEntry* ed[4];
        for (int t = 0; t < 4; ++t) {
            ed[t] = sycl::malloc_device<sk::NightGroupedEntry>(std::max<size_t>(1, ents[t].size()), *q);
            if (!ents[t].empty()) q->memcpy(ed[t], ents[t].data(), ents[t].size() * sizeof(sk::NightGroupedEntry));
        }
        q->wait();
        bool ok = true;
        auto mine = [&]() {
            const int mts[4] = {2, 4, 8, 16};
            for (int t = 0; t < 4; ++t)
                if (!ents[t].empty())
                    ok &= sk::night_grouped_f16(ty, gu, ed[t], (int) ents[t].size(), mts[t], X, yn, (int) N, (int) K, up_off, q);
        };
        double hr, gr, hm, gm;
        cx.time(ref, 3, hr, gr);
        cx.time(mine, 3, hm, gm);
        if (!ok) { std::printf("%s: night_grouped_f16 refused type %d\n", tag, ty); return; }
        std::vector<float> a((size_t) rows * N), b((size_t) rows * N);
        q->memcpy(a.data(), yr, a.size() * 4); q->memcpy(b.data(), yn, b.size() * 4).wait();
        double mx = 0, em = 0, s2 = 0, d2 = 0; size_t nf = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (!std::isfinite(b[i])) { ++nf; continue; }
            const double d = std::fabs((double) a[i] - b[i]);
            em = std::max(em, d); mx = std::max(mx, (double) std::fabs(a[i])); s2 += (double) a[i] * a[i]; d2 += d * d;
        }
        const double flop = 2.0 * rows * N * K;
        std::printf("%-6s %-4s ty %2d experts %3d rows %6lld | ref host %7.3f gpu %7.3f ms | grouped host %6.3f gpu %7.3f ms "
                    "(%5.2fx, %5.1f TFLOPS) | max abs %.2e / max %.2e, rms rel %.2e%s\n",
                    tag, gu ? "GU" : "down", ty, n, (long long) rows, hr, gr, hm, gm, gr / gm, flop / gm * 1e-9, em, mx,
                    std::sqrt(d2 / std::max(s2, 1e-30)), nf ? " NONFINITE" : "");
        sycl::free(W, *q); sycl::free(X, *q); sycl::free(dq, *q); sycl::free(yr, *q); sycl::free(yn, *q);
        for (auto* p : ed) sycl::free(p, *q);
    };

    // pipe: the engine's card-1 layer as STRATA_NIGHT_PREFILL_GROUPED runs it (prefill.cpp): rows small-first, ONE
    // FP16 gather into the FP32 rows region (alias), large experts per expert from the highest offset (dequant + oneMKL,
    // in place), then batches of small experts (<= T rows) from the last batch down: grouped gate/up -> GUf, one
    // swiglu -> Hh, grouped down -> rows in place.  Compared with every expert on the reference path (no alias).
    //   night_grouped_bench pipe gt dt T n_exp median sigma G
    if (mode == "pipe") {
        const int gt = std::atoi(argv[2]), dt = std::atoi(argv[3]);
        const int64_t T = std::atoll(argv[4]);
        const int n = std::atoi(argv[5]);
        const double med = std::atof(argv[6]), sig = std::atof(argv[7]);
        const int G = std::atoi(argv[8]);
        std::lognormal_distribution<double> ld(std::log(med), sig);
        std::vector<int32_t> cnt(n);
        for (auto& c : cnt) c = (int) std::clamp<double>(std::round(ld(rng)), 1.0, (double) T);
        std::vector<int> order;   // small first (id order), then large
        for (int e = 0; e < n; ++e) if (cnt[e] <= G) order.push_back(e);
        const size_t ng_n = order.size();
        for (int e = 0; e < n; ++e) if (cnt[e] > G) order.push_back(e);
        std::vector<int64_t> off(n);
        int64_t R = 0;
        for (int e : order) { off[e] = R; R += cnt[e]; }
        const size_t gmat = sk::iq_row_bytes(gt, NE) * FF, dmat = sk::iq_row_bytes(dt, FF) * NE;
        const size_t up_off = gmat, down_off = 2 * gmat, blob = (down_off + dmat + 255) & ~(size_t) 255;
        uint8_t* W = sycl::malloc_device<uint8_t>(blob * n, *q);
        for (int e = 0; e < n; ++e) {
            auto g = rand_mat(rng, gt, FF, NE), u = rand_mat(rng, gt, FF, NE), d = rand_mat(rng, dt, NE, FF);
            q->memcpy(W + e * blob, g.data(), gmat); q->memcpy(W + e * blob + up_off, u.data(), gmat);
            q->memcpy(W + e * blob + down_off, d.data(), dmat); q->wait();
        }
        std::vector<uint16_t> hx((size_t) R * NE);
        for (auto& x : hx) x = f2h(nd(rng) * 0.25f);
        uint16_t* Xg = sycl::malloc_device<uint16_t>(hx.size(), *q);          // the gathered rows (stand-in for the gather)
        q->memcpy(Xg, hx.data(), hx.size() * 2);
        float* rows = sycl::malloc_device<float>((size_t) R * NE, *q);        // alias region (FP32 rows; FP16 inputs first)
        float* ref = sycl::malloc_device<float>((size_t) R * NE, *q);
        uint16_t* dqg = sycl::malloc_device<uint16_t>((size_t) 1280 * NE, *q);
        uint16_t* dqd = sycl::malloc_device<uint16_t>((size_t) NE * 640, *q);
        float* GUf = sycl::malloc_device<float>((size_t) T * 1280, *q);
        uint16_t* Hh = sycl::malloc_device<uint16_t>((size_t) T * 640, *q);
        auto* ent = sycl::malloc_device<sk::NightGroupedEntry>(2 * 512 * 16, *q);
        q->wait();
        auto one = [&](int e, const uint16_t* x, float* y) {   // the engine's per-expert path
            const uint8_t* bd = W + e * blob;
            sk::iq_dequant_gu_f16(gt, bd, bd + up_off, FF, NE, dqg, q);
            sk::iq_dequant_f16(dt, bd + down_off, NE * FF, dqd, q);
            gemm.f16(x, dqg, GUf, cnt[e], 1280, NE);
            strata::prefill::swiglu_interleaved(GUf, Hh, cnt[e], q);
            gemm.f16(Hh, dqd, y, cnt[e], NE, 640);
        };
        // reference: every expert, inputs from Xg, outputs to ref
        for (int e = 0; e < n; ++e) one(e, Xg + off[e] * NE, ref + off[e] * NE);
        q->wait();
        std::vector<sk::NightGroupedEntry> host;
        auto engine = [&]() {
            uint16_t* alias_x = reinterpret_cast<uint16_t*>(rows);
            q->memcpy(alias_x, Xg, (size_t) R * NE * 2);   // the one gather
            for (size_t j = order.size(); j-- > ng_n;) { const int e = order[j]; one(e, alias_x + off[e] * NE, rows + off[e] * NE); }
            struct Batch { int64_t a, rows; size_t gu[4][2]; };
            std::vector<Batch> batches;
            host.clear();
            std::vector<sk::NightGroupedEntry> dn;
            size_t j = 0;
            while (j < ng_n) {
                Batch b{};
                b.a = off[order[j]];
                size_t j1 = j; int64_t br = 0;
                while (j1 < ng_n && br + cnt[order[j1]] <= T) br += cnt[order[j1++]];
                b.rows = br;
                for (int t = 0; t < 4; ++t) {
                    const int cap = sk::night_grouped_rows(2 << t);
                    b.gu[t][0] = host.size();
                    for (size_t i = j; i < j1; ++i) {
                        const int e = order[i]; const int c = cnt[e];
                        if ((c <= 16 ? 0 : c <= 32 ? 1 : c <= 64 ? 2 : 3) != t) continue;
                        const uint8_t* bd = W + e * blob;
                        for (int r0 = 0; r0 < c; r0 += cap) {
                            host.push_back({bd, off[e] - b.a + r0, std::min(cap, c - r0), 0});
                            dn.push_back({bd + down_off, off[e] - b.a + r0, std::min(cap, c - r0), 0});
                        }
                    }
                    b.gu[t][1] = host.size();
                }
                batches.push_back(b);
                j = j1;
            }
            const size_t ne_gu = host.size();
            host.insert(host.end(), dn.begin(), dn.end());
            q->memcpy(ent, host.data(), host.size() * sizeof(sk::NightGroupedEntry));
            for (size_t bi = batches.size(); bi-- > 0;) {
                const Batch& b = batches[bi];
                for (int t = 0; t < 4; ++t) {
                    const int c = (int) (b.gu[t][1] - b.gu[t][0]);
                    if (c > 0) sk::night_grouped_f16(gt, true, ent + b.gu[t][0], c, 2 << t, alias_x + b.a * NE, GUf, 1280, (int) NE, up_off, q);
                }
                strata::prefill::swiglu_interleaved(GUf, Hh, b.rows, q);
                for (int t = 0; t < 4; ++t) {
                    const int c = (int) (b.gu[t][1] - b.gu[t][0]);
                    if (c > 0) sk::night_grouped_f16(dt, false, ent + ne_gu + b.gu[t][0], c, 2 << t, Hh, rows + b.a * NE, (int) NE, 640, 0, q);
                }
            }
            return batches.size();
        };
        double hb, gb;
        size_t nb = 0;
        cx.time([&] { nb = engine(); }, 3, hb, gb);
        // reference timing: the same layer all per expert (with the gather copy)
        double hr, gr;
        cx.time([&] {
            uint16_t* alias_x = reinterpret_cast<uint16_t*>(rows + (size_t) 0);
            (void) alias_x;
            for (size_t jj = order.size(); jj-- > 0;) { const int e = order[jj]; one(e, Xg + off[e] * NE, ref + off[e] * NE); }
        }, 3, hr, gr);
        cx.time([&] { nb = engine(); }, 1, hb, gb);   // leave the engine's result in rows
        std::vector<float> a((size_t) R * NE), b((size_t) R * NE);
        q->memcpy(a.data(), ref, a.size() * 4); q->memcpy(b.data(), rows, b.size() * 4).wait();
        double mxs = 0, ems = 0, s2 = 0, d2 = 0; size_t diff_large = 0, nf = 0;
        for (int e = 0; e < n; ++e)
            for (int64_t i = off[e] * NE; i < (off[e] + cnt[e]) * NE; ++i) {
                if (!std::isfinite(b[i])) { ++nf; continue; }
                const double d = std::fabs((double) a[i] - b[i]);
                if (cnt[e] > G) diff_large += a[i] != b[i];
                else { ems = std::max(ems, d); mxs = std::max(mxs, (double) std::fabs(a[i])); s2 += (double) a[i] * a[i]; d2 += d * d; }
            }
        std::printf("pipe gt %d dt %d T %lld: %d experts (%zu <= %d rows), %lld rows, %zu batches | per-expert path host %.2f gpu %.2f ms | "
                    "engine path host %.2f gpu %.2f ms | large experts: %zu values differ (want 0); small: max abs %.2e / max %.2e, "
                    "rms rel %.2e; non-finite %zu\n", gt, dt, (long long) T, n, ng_n, G, (long long) R, nb, hr, gr, hb, gb,
                    diff_large, ems, mxs, std::sqrt(d2 / std::max(s2, 1e-30)), nf);
        return 0;
    }
    if (mode == "gu" || mode == "down") {
        const int ty = std::atoi(argv[2]);
        const int n = std::atoi(argv[3]);
        for (int i = 4; i < argc; ++i) {
            const int M = std::atoi(argv[i]);
            std::vector<int32_t> cnt(n, M);
            run(mode == "gu", ty, cnt, ("M=" + std::to_string(M)).c_str());
        }
    } else if (mode == "mix") {
        const int gt = std::atoi(argv[2]), dt = std::atoi(argv[3]);
        const int64_t T = std::atoll(argv[4]);
        const int n = std::atoi(argv[5]);
        const double med = std::atof(argv[6]), sig = std::atof(argv[7]);
        const int maxr = std::atoi(argv[8]);
        std::lognormal_distribution<double> ld(std::log(med), sig);
        std::vector<int32_t> cnt;
        int64_t all = 0;
        for (int i = 0; i < n; ++i) {
            const int c = (int) std::clamp<double>(std::round(ld(rng)), 1.0, (double) T);
            all += c;
            if (c <= maxr) cnt.push_back(c);
        }
        std::printf("mix: %d of %d experts have <= %d rows (%lld of %lld rows)\n", (int) cnt.size(), n, maxr,
                    (long long) std::accumulate(cnt.begin(), cnt.end(), 0LL), (long long) all);
        run(true, gt, cnt, "mix");
        run(false, dt, cnt, "mix");
    }
    return 0;
}
