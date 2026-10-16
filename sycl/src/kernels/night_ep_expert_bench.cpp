// night_ep_expert_bench (night lab 2026-10-16): the EP prefill peer's expert loop on ONE card.
//   ORIG : per expert gather_rows16 + iq_dequant_expert_f16 + oneMKL GU + swiglu_interleaved + oneMKL down (5 launches)
//   GRP  : experts with <= small_max rows first (contiguous rows), batched: one gather per batch, iq_xmx_grouped
//          GU over up to 16 experts, one swiglu, iq_xmx_grouped down; the large experts as ORIG.
// Reports host enqueue time, end-to-end wall time and GPU-only time (queue held by a gate kernel while enqueuing),
// and the error of GRP against ORIG (rows of large experts must be bitwise equal).
//   night_ep_expert_bench T n_exp median sigma gu_type d_type [small_max=64] [reps=3] [seed=1]
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/iq_dequant_expert.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/xmx_moe.hpp"
#include <oneapi/mkl.hpp>
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <pthread.h>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point a) { return std::chrono::duration<double, std::milli>(Clock::now() - a).count(); }
static uint16_t f2h(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }

int main(int argc, char** argv) {
    const int64_t T = argc > 1 ? std::atoll(argv[1]) : 12288;
    const int n_exp = argc > 2 ? std::atoi(argv[2]) : 245;
    const double median = argc > 3 ? std::atof(argv[3]) : 60;
    const double sigma = argc > 4 ? std::atof(argv[4]) : 1.88;
    const int gt = argc > 5 ? std::atoi(argv[5]) : 21;
    const int dt = argc > 6 ? std::atoi(argv[6]) : 42;
    const int small_max = argc > 7 ? std::atoi(argv[7]) : 64;
    const int reps = argc > 8 ? std::atoi(argv[8]) : 3;
    const int seed = argc > 9 ? std::atoi(argv[9]) : 1;
    const int64_t N = 2560, FF = 640;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    sycl::queue* q = &dpct::get_in_order_queue();
    std::printf("device: %s\n", q->get_device().get_info<sycl::info::device::name>().c_str());
    std::mt19937 rng(seed);
    // row counts per expert
    std::lognormal_distribution<double> ld(std::log(median), sigma);
    std::vector<int32_t> cnt(n_exp);
    int64_t rows = 0;
    for (auto& c : cnt) { c = (int32_t) std::clamp<double>(std::round(ld(rng)), 1.0, (double) T); rows += c; }
    int n_small = 0; int64_t rows_small = 0;
    for (int32_t c : cnt) if (c <= small_max) { ++n_small; rows_small += c; }
    std::printf("T %lld experts %d rows %lld (mean %.1f) small(<=%d) %d experts %lld rows; gu %d down %d\n",
                (long long) T, n_exp, (long long) rows, (double) rows / n_exp, small_max, n_small,
                (long long) rows_small, gt, dt);
    // expert blobs: gate | up | down, random bytes with d = 0.0625 at each block start
    const size_t gu_mat = strata::kernels::iq_row_bytes(gt, N) * FF;
    const size_t d_mat = strata::kernels::iq_row_bytes(dt, FF) * N;
    const size_t up_off = gu_mat, down_off = 2 * gu_mat;   // native_expert_layout
    const size_t blob_bytes = (down_off + d_mat + 255) & ~(size_t) 255;
    auto blk_of = [](int ty) { return ty == 20 ? 32 : ty == 42 ? 64 : 256; };
    const size_t gblk = strata::kernels::iq_row_bytes(gt, blk_of(gt)), dblk = strata::kernels::iq_row_bytes(dt, blk_of(dt));
    if (gu_mat == 0 || d_mat == 0) { std::printf("unsupported types\n"); return 2; }
    uint8_t* blobs = sycl::malloc_device<uint8_t>(blob_bytes * n_exp, *q);
    {
        std::vector<uint8_t> h(blob_bytes);
        for (int e = 0; e < n_exp; ++e) {
            for (auto& x : h) x = (uint8_t) rng();
            for (size_t o = 0; o + 1 < gu_mat; o += gblk) { h[o] = 0x00; h[o + 1] = 0x2C; h[up_off + o] = 0x00; h[up_off + o + 1] = 0x2C; }
            for (size_t o = 0; o + 1 < d_mat; o += dblk) { h[down_off + o] = 0x00; h[down_off + o + 1] = 0x2C; }
            q->memcpy(blobs + (size_t) e * blob_bytes, h.data(), blob_bytes).wait();
        }
    }
    // activations (T x N fp16), the source token per row, outputs
    std::vector<uint16_t> hx((size_t) T * N);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& x : hx) x = f2h(nd(rng) * 0.25f);
    uint16_t* X = sycl::malloc_device<uint16_t>(hx.size(), *q);
    q->memcpy(X, hx.data(), hx.size() * 2);
    std::vector<int32_t> hsrc((size_t) rows);
    for (auto& s : hsrc) s = (int32_t) (rng() % T);
    int32_t* src = sycl::malloc_device<int32_t>(rows, *q);
    q->memcpy(src, hsrc.data(), rows * 4);
    uint16_t* Xs = sycl::malloc_device<uint16_t>((size_t) T * N, *q);
    float* GUf = sycl::malloc_device<float>((size_t) T * 1280, *q);
    uint16_t* Hh = sycl::malloc_device<uint16_t>((size_t) T * 640, *q);
    uint16_t* dq_gu = sycl::malloc_device<uint16_t>((size_t) 1280 * N, *q);
    uint16_t* dq_d = sycl::malloc_device<uint16_t>((size_t) N * 640, *q);
    float* out_a = sycl::malloc_device<float>((size_t) rows * N, *q);
    float* out_b = sycl::malloc_device<float>((size_t) rows * N, *q);
    q->wait();
    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(q, 1 << 20, err)) { std::printf("gemm init: %s\n", err.c_str()); return 2; }
    // busy: a kernel that keeps the queue occupied for ~busy_ms so a whole loop can be queued before the GPU reaches
    // it (bounded: no host flag).  GPU time of a loop = wall(busy + loop) - wall(busy).
    uint32_t* sink = sycl::malloc_device<uint32_t>(1, *q);
    auto busy = [&](uint32_t iters) {
        q->single_task<class night_busy_kernel>([=] {
            uint32_t x = 1;
            for (uint32_t i = 0; i < iters; ++i) x = x * 1664525u + 1013904223u;
            *sink = x;
        });
    };
    uint32_t busy_iters = 1u << 20;
    double busy_ms = 0;
    for (int tries = 0; tries < 12; ++tries) {
        q->wait();
        auto a = Clock::now();
        busy(busy_iters);
        q->wait();
        busy_ms = ms_since(a);
        if (busy_ms > 120) break;
        busy_iters = (uint32_t) std::min<double>(4.0e9, busy_iters * std::max(2.0, 150.0 / std::max(busy_ms, 0.01)));
    }
    std::printf("busy kernel: %u iterations = %.1f ms\n", busy_iters, busy_ms);
    auto gate = [&]() { busy(busy_iters); };
    // pageable vs pinned host -> device copies queued behind a busy kernel: does the call block until it executes?
    {
        const size_t nb = 490u << 10;
        std::vector<uint8_t> pg(nb, 1);
        uint8_t* pin = sycl::malloc_host<uint8_t>(nb, *q);
        uint8_t* dd = sycl::malloc_device<uint8_t>(nb, *q);
        std::memset(pin, 1, nb);
        q->memcpy(dd, pg.data(), nb); q->memcpy(dd, pin, nb); q->wait();
        gate();
        auto a = Clock::now();
        q->memcpy(dd, pg.data(), nb);
        const double t_pg = ms_since(a);
        q->wait();
        gate();
        a = Clock::now();
        q->memcpy(dd, pin, nb);
        const double t_pin = ms_since(a);
        q->wait();
        std::printf("host->device 490 KB memcpy queued behind a %.0f ms kernel: pageable call returns after %.2f ms, "
                    "pinned after %.3f ms\n", busy_ms, t_pg, t_pin);
        std::vector<uint8_t> back(nb);
        gate();
        a = Clock::now();
        q->memcpy(back.data(), dd, nb);
        const double t_d2h = ms_since(a);
        q->wait();
        std::printf("device->pageable host 490 KB memcpy behind the busy kernel: call returns after %.2f ms\n", t_d2h);
        sycl::free(pin, *q); sycl::free(dd, *q);
    }
    // ORIG: experts in id order, rows contiguous by expert
    std::vector<int64_t> off_a(n_exp + 1, 0);
    for (int e = 0; e < n_exp; ++e) off_a[e + 1] = off_a[e] + cnt[e];
    auto orig = [&](float* out) {
        for (int e = 0; e < n_exp; ++e) {
            const int64_t o = off_a[e], ne = cnt[e];
            const uint8_t* bd = blobs + (size_t) e * blob_bytes;
            strata::prefill::gather_rows16(X, src + o, Xs, ne, N, q);
            strata::kernels::iq_dequant_expert_f16(gt, dt, bd, bd + up_off, bd + down_off, FF, N, dq_gu, dq_d, q);
            gemm.f16(Xs, dq_gu, GUf, ne, 1280, N);
            strata::prefill::swiglu_interleaved(GUf, Hh, ne, q);
            gemm.f16(Hh, dq_d, out + o * N, ne, N, 640);
        }
    };
    // GRP: the same rows in a different place: small experts first.  src_b holds the rows in that order; the
    // comparison maps them back.
    std::vector<int> order;
    for (int e = 0; e < n_exp; ++e) if (cnt[e] <= small_max) order.push_back(e);
    const int n_sm = (int) order.size();
    for (int e = 0; e < n_exp; ++e) if (cnt[e] > small_max) order.push_back(e);
    std::vector<int64_t> off_b(n_exp, 0);
    std::vector<int32_t> hsrc_b((size_t) rows);
    {
        int64_t r = 0;
        for (int e : order) { off_b[e] = r; std::memcpy(&hsrc_b[r], &hsrc[off_a[e]], cnt[e] * 4); r += cnt[e]; }
    }
    int32_t* src_b = sycl::malloc_device<int32_t>(rows, *q);
    q->memcpy(src_b, hsrc_b.data(), rows * 4).wait();
    int launches_b = 0;
    auto grp = [&](float* out) {
        launches_b = 0;
        int j = 0;
        while (j < n_sm) {   // batches of <= T rows
            int j1 = j; int64_t br = 0;
            while (j1 < n_sm && br + cnt[order[j1]] <= T) br += cnt[order[j1++]];
            const int64_t o0 = off_b[order[j]];
            strata::prefill::gather_rows16(X, src_b + o0, Xs, br, N, q); ++launches_b;
            for (int g0 = j; g0 < j1; g0 += strata::kernels::kXmxGroupMax) {
                const int g1 = std::min(j1, g0 + strata::kernels::kXmxGroupMax), ng = g1 - g0;
                const uint8_t* ga[strata::kernels::kXmxGroupMax]; const uint8_t* gb[strata::kernels::kXmxGroupMax];
                const uint8_t* gd[strata::kernels::kXmxGroupMax]; int32_t gc[strata::kernels::kXmxGroupMax];
                int64_t gr = 0;
                for (int i = 0; i < ng; ++i) {
                    const uint8_t* bd = blobs + (size_t) order[g0 + i] * blob_bytes;
                    ga[i] = bd; gb[i] = bd + up_off; gd[i] = bd + down_off; gc[i] = cnt[order[g0 + i]]; gr += gc[i];
                }
                const int64_t lo = off_b[order[g0]] - o0;   // the group's first row inside the batch
                strata::kernels::iq_xmx_grouped(gt, ga, gb, gc, ng, Xs + lo * N, GUf + lo * 1280, 1280, (int) N, q);
                strata::prefill::swiglu_interleaved(GUf + lo * 1280, Hh + lo * 640, gr, q);
                strata::kernels::iq_xmx_grouped(dt, gd, nullptr, gc, ng, Hh + lo * 640, out + off_b[order[g0]] * N,
                                                (int) N, 640, q);
                launches_b += 3;
            }
            j = j1;
        }
        for (int jj = n_sm; jj < n_exp; ++jj) {
            const int e = order[jj];
            const int64_t o = off_b[e], ne = cnt[e];
            const uint8_t* bd = blobs + (size_t) e * blob_bytes;
            strata::prefill::gather_rows16(X, src_b + o, Xs, ne, N, q);
            strata::kernels::iq_dequant_expert_f16(gt, dt, bd, bd + up_off, bd + down_off, FF, N, dq_gu, dq_d, q);
            gemm.f16(Xs, dq_gu, GUf, ne, 1280, N);
            strata::prefill::swiglu_interleaved(GUf, Hh, ne, q);
            gemm.f16(Hh, dq_d, out + o * N, ne, N, 640);
            launches_b += 5;
        }
    };
    // WAVE: waves of <= 16 experts and <= T rows in ORIG's order: one gather, one grouped dequant into 16 slots, GU
    // per expert (direct oneMKL or one gemm_batch), one swiglu, down likewise.
    constexpr int W = 16;
    std::vector<uint16_t*> wgu(W), wd(W);
    uint16_t* wgu0 = sycl::malloc_device<uint16_t>((size_t) W * 1280 * N, *q);
    uint16_t* wd0 = sycl::malloc_device<uint16_t>((size_t) W * N * 640, *q);
    for (int i = 0; i < W; ++i) { wgu[i] = wgu0 + (size_t) i * 1280 * N; wd[i] = wd0 + (size_t) i * N * 640; }
    float* out_c = sycl::malloc_device<float>((size_t) rows * N, *q);
    float* out_d = sycl::malloc_device<float>((size_t) rows * N, *q);
    struct BatchArgs {
        oneapi::mkl::transpose ta[W], tb[W];
        int64_t m[W], n[W], k[W], lda[W], ldb[W], ldc[W], gs[W];
        float alpha[W], beta[W];
        const sycl::half* a[W]; const sycl::half* b[W]; float* c[W];
    };
    BatchArgs* ba = sycl::malloc_shared<BatchArgs>(4, *q);   // ring of 2 waves x (GU, down)
    int wave_launches = 0;
    auto mkl_f16 = [&](const uint16_t* Xp, const uint16_t* Wp, float* Yp, int64_t M, int64_t Nn, int64_t Kk) {
        oneapi::mkl::blas::column_major::gemm(*q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                                              Nn, M, Kk, 1.0f, (const sycl::half*) Wp, Kk, (const sycl::half*) Xp, Kk,
                                              0.0f, Yp, Nn, oneapi::mkl::blas::compute_mode::unset);
    };
    auto wave = [&](float* out, bool batch, bool direct) {
        wave_launches = 0;
        int j = 0, wv = 0;
        while (j < n_exp) {
            int j1 = j; int64_t br = 0;
            while (j1 < n_exp && j1 - j < W && br + cnt[j1] <= T) br += cnt[j1++];
            const int nw = j1 - j;
            const int64_t o0 = off_a[j];
            strata::prefill::gather_rows16(X, src + o0, Xs, br, N, q);
            const uint8_t* bl[W];
            for (int i = 0; i < nw; ++i) bl[i] = blobs + (size_t) (j + i) * blob_bytes;
            strata::kernels::iq_dequant_experts_f16(gt, dt, nw, bl, up_off, down_off, FF, N, wgu0, wd0, q);
            wave_launches += 2;
            if (batch) {
                BatchArgs& A = ba[(wv & 1) * 2];
                for (int i = 0; i < nw; ++i) {
                    const int64_t lo = off_a[j + i] - o0;
                    A.ta[i] = oneapi::mkl::transpose::trans; A.tb[i] = oneapi::mkl::transpose::nontrans;
                    A.m[i] = 1280; A.n[i] = cnt[j + i]; A.k[i] = N; A.lda[i] = N; A.ldb[i] = N; A.ldc[i] = 1280; A.gs[i] = 1;
                    A.alpha[i] = 1.0f; A.beta[i] = 0.0f;
                    A.a[i] = (const sycl::half*) wgu[i]; A.b[i] = (const sycl::half*) (Xs + lo * N); A.c[i] = GUf + lo * 1280;
                }
                oneapi::mkl::blas::column_major::gemm_batch(*q, A.ta, A.tb, A.m, A.n, A.k, A.alpha, A.a, A.lda, A.b, A.ldb,
                                                            A.beta, A.c, A.ldc, (int64_t) nw, A.gs,
                                                            oneapi::mkl::blas::compute_mode::unset);
            } else {
                for (int i = 0; i < nw; ++i) {
                    const int64_t lo = off_a[j + i] - o0;
                    if (direct) mkl_f16(Xs + lo * N, wgu[i], GUf + lo * 1280, cnt[j + i], 1280, N);
                    else gemm.f16(Xs + lo * N, wgu[i], GUf + lo * 1280, cnt[j + i], 1280, N);
                }
            }
            strata::prefill::swiglu_interleaved(GUf, Hh, br, q);
            if (batch) {
                BatchArgs& A = ba[(wv & 1) * 2 + 1];
                for (int i = 0; i < nw; ++i) {
                    const int64_t lo = off_a[j + i] - o0;
                    A.ta[i] = oneapi::mkl::transpose::trans; A.tb[i] = oneapi::mkl::transpose::nontrans;
                    A.m[i] = N; A.n[i] = cnt[j + i]; A.k[i] = 640; A.lda[i] = 640; A.ldb[i] = 640; A.ldc[i] = N; A.gs[i] = 1;
                    A.alpha[i] = 1.0f; A.beta[i] = 0.0f;
                    A.a[i] = (const sycl::half*) wd[i]; A.b[i] = (const sycl::half*) (Hh + lo * 640); A.c[i] = out + off_a[j + i] * N;
                }
                oneapi::mkl::blas::column_major::gemm_batch(*q, A.ta, A.tb, A.m, A.n, A.k, A.alpha, A.a, A.lda, A.b, A.ldb,
                                                            A.beta, A.c, A.ldc, (int64_t) nw, A.gs,
                                                            oneapi::mkl::blas::compute_mode::unset);
                wave_launches += 3;
                q->wait();   // the shared argument ring: keep it simple in the probe (2 slots would race)
            } else {
                for (int i = 0; i < nw; ++i) {
                    const int64_t lo = off_a[j + i] - o0;
                    if (direct) mkl_f16(Hh + lo * 640, wd[i], out + off_a[j + i] * N, cnt[j + i], N, 640);
                    else gemm.f16(Hh + lo * 640, wd[i], out + off_a[j + i] * N, cnt[j + i], N, 640);
                }
                wave_launches += 1 + 2 * nw;
            }
            j = j1; ++wv;
        }
    };
    struct R { double host, wall, gpu; };
    auto measure2 = [&](auto f, float* out) {
        R best{1e30, 1e30, 1e30};
        f(out); q->wait();
        for (int r = 0; r < reps; ++r) {
            q->wait();
            auto a = Clock::now();
            f(out);
            const double h = ms_since(a);
            q->wait();
            const double w = ms_since(a);
            auto b = Clock::now();
            gate();
            f(out);
            const double hb = ms_since(b);
            q->wait();
            const double g = ms_since(b) - busy_ms;
            if (hb > busy_ms * 0.8) std::printf("  (warning: host enqueue %.1f ms under the busy kernel)\n", hb);
            best.host = std::min(best.host, h); best.wall = std::min(best.wall, w); best.gpu = std::min(best.gpu, g);
        }
        return best;
    };
    const R ra = measure2(orig, out_a);
    const R rb = measure2(grp, out_b);
    std::printf("ORIG: launches %d  host %.2f ms  wall %.2f ms  gpu %.2f ms\n", 5 * n_exp, ra.host, ra.wall, ra.gpu);
    std::printf("GRP : launches %d  host %.2f ms  wall %.2f ms  gpu %.2f ms\n", launches_b, rb.host, rb.wall, rb.gpu);
    const R rc = measure2([&](float* o) { wave(o, false, false); }, out_c);
    std::printf("WAVE-dpct : launches %d  host %.2f ms  wall %.2f ms  gpu %.2f ms\n", wave_launches, rc.host, rc.wall, rc.gpu);
    const R rd = measure2([&](float* o) { wave(o, false, true); }, out_d);
    std::printf("WAVE-mkl  : launches %d  host %.2f ms  wall %.2f ms  gpu %.2f ms\n", wave_launches, rd.host, rd.wall, rd.gpu);
    {
        std::vector<float> a((size_t) rows * N), c((size_t) rows * N), d((size_t) rows * N);
        q->memcpy(a.data(), out_a, a.size() * 4); q->memcpy(c.data(), out_c, c.size() * 4);
        q->memcpy(d.data(), out_d, d.size() * 4).wait();
        std::printf("WAVE-dpct vs ORIG: %s; WAVE-mkl vs ORIG: %s\n",
                    std::memcmp(a.data(), c.data(), a.size() * 4) ? "DIFFER" : "bitwise equal",
                    std::memcmp(a.data(), d.data(), a.size() * 4) ? "DIFFER" : "bitwise equal");
    }
    if (std::getenv("NIGHT_BENCH_BATCH")) {
        float* out_e = out_c;
        // the batched one (waits per wave: host time is not representative, GPU time is)
        auto t0 = Clock::now();
        wave(out_e, true, false); q->wait();
        const double tb = ms_since(t0);
        std::vector<float> a((size_t) rows * N), e((size_t) rows * N);
        q->memcpy(a.data(), out_a, a.size() * 4); q->memcpy(e.data(), out_e, e.size() * 4).wait();
        double mx = 0, em = 0; size_t nd2 = 0;
        for (size_t i = 0; i < a.size(); ++i) { mx = std::max(mx, (double) std::fabs(a[i])); const double df = std::fabs((double) a[i] - e[i]); em = std::max(em, df); nd2 += df != 0; }
        std::printf("WAVE-batch: launches %d  wall(with per-wave waits) %.2f ms; vs ORIG max abs %.3e (max |ref| %.3e), %zu values differ\n",
                    wave_launches, tb, em, mx, nd2);
    }
    // compare
    std::vector<float> a((size_t) rows * N), b((size_t) rows * N);
    q->memcpy(a.data(), out_a, a.size() * 4);
    q->memcpy(b.data(), out_b, b.size() * 4).wait();
    double emax_s = 0, mx_s = 0, emax_l = 0;
    size_t neq_l = 0, nonfin = 0;
    for (int e = 0; e < n_exp; ++e) {
        const bool sm = cnt[e] <= small_max;
        for (int64_t r = 0; r < cnt[e]; ++r) {
            const float* pa = &a[(size_t) (off_a[e] + r) * N];
            const float* pb = &b[(size_t) (off_b[e] + r) * N];
            for (int64_t i = 0; i < N; ++i) {
                if (!std::isfinite(pa[i]) || !std::isfinite(pb[i])) { ++nonfin; continue; }
                const double d = std::fabs((double) pa[i] - pb[i]);
                if (sm) { emax_s = std::max(emax_s, d); mx_s = std::max(mx_s, (double) std::fabs(pa[i])); }
                else { emax_l = std::max(emax_l, d); neq_l += std::memcmp(&pa[i], &pb[i], 4) != 0; }
            }
        }
    }
    std::printf("error: small experts max abs %.3e (max |ref| %.3e, rel %.2e); large experts max abs %.3e, %zu words differ; "
                "non-finite %zu\n", emax_s, mx_s, mx_s > 0 ? emax_s / mx_s : 0.0, emax_l, neq_l, nonfin);
    // PAR: the ORIG loop split over P host threads, each with its own in-order queue, Gemm and buffers on the SAME
    // card (experts [p*n/P, (p+1)*n/P) to thread p); outputs land at the same rows, so the result must be bitwise ORIG's.
    for (int P : {2, 3}) {
        std::vector<sycl::queue*> qs(P);
        std::vector<strata::prefill::Gemm*> gs(P);
        std::vector<uint16_t*> xs(P), hs(P), dg(P), dd(P);
        std::vector<float*> gus(P);
        for (int p = 0; p < P; ++p) {
            qs[p] = p == 0 ? q : new sycl::queue(q->get_context(), q->get_device(), sycl::property::queue::in_order());
            gs[p] = new strata::prefill::Gemm();
            if (!gs[p]->init(qs[p], 1 << 20, err)) { std::printf("gemm init: %s\n", err.c_str()); return 2; }
            xs[p] = sycl::malloc_device<uint16_t>((size_t) T * N, *q);
            hs[p] = sycl::malloc_device<uint16_t>((size_t) T * 640, *q);
            gus[p] = sycl::malloc_device<float>((size_t) T * 1280, *q);
            dg[p] = sycl::malloc_device<uint16_t>((size_t) 1280 * N, *q);
            dd[p] = sycl::malloc_device<uint16_t>((size_t) N * 640, *q);
        }
        float* out_p = out_c;
        auto part = [&](int p) {
            sycl::queue* qq = qs[p];
            for (int e = p * n_exp / P; e < (p + 1) * n_exp / P; ++e) {
                const int64_t o = off_a[e], ne = cnt[e];
                const uint8_t* bd = blobs + (size_t) e * blob_bytes;
                strata::prefill::gather_rows16(X, src + o, xs[p], ne, N, qq);
                strata::kernels::iq_dequant_expert_f16(gt, dt, bd, bd + up_off, bd + down_off, FF, N, dg[p], dd[p], qq);
                gs[p]->f16(xs[p], dg[p], gus[p], ne, 1280, N);
                strata::prefill::swiglu_interleaved(gus[p], hs[p], ne, qq);
                gs[p]->f16(hs[p], dd[p], out_p + o * N, ne, N, 640);
            }
        };
        // persistent workers (as the engine would keep them), released by a generation counter
        std::atomic<int> gen{0}, done{0};
        std::atomic<bool> quit{false};
        std::vector<std::thread> th;
        for (int p = 1; p < P; ++p)
            th.emplace_back([&, p] {
                cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(16 + p, &cs); pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
                int seen = 0;
                while (!quit.load()) {
                    const int g = gen.load();
                    if (g == seen) { std::this_thread::yield(); continue; }
                    seen = g;
                    part(p);
                    done.fetch_add(1);
                }
            });
        auto run = [&]() {
            done.store(0);
            gen.fetch_add(1);
            part(0);
            while (done.load() < P - 1) std::this_thread::yield();
        };
        run(); for (auto* qq : qs) qq->wait();
        double hbest = 1e30, wbest = 1e30;
        for (int r = 0; r < reps; ++r) {
            for (auto* qq : qs) qq->wait();
            auto a = Clock::now();
            run();
            const double h = ms_since(a);
            for (auto* qq : qs) qq->wait();
            const double w = ms_since(a);
            hbest = std::min(hbest, h); wbest = std::min(wbest, w);
        }
        quit.store(true);
        for (auto& t : th) t.join();
        std::vector<float> a((size_t) rows * N), c((size_t) rows * N);
        q->memcpy(a.data(), out_a, a.size() * 4); q->memcpy(c.data(), out_p, c.size() * 4).wait();
        std::printf("PAR%d (threads, queues): host %.2f ms  wall %.2f ms; vs ORIG %s\n", P, hbest, wbest,
                    std::memcmp(a.data(), c.data(), a.size() * 4) ? "DIFFER" : "bitwise equal");
    }
    // per-launch host cost of each kernel type (M = 32)
    {
        const int L = 400;
        auto per = [&](const char* name, auto f) {
            q->wait();
            gate();
            auto a0 = Clock::now();
            for (int i = 0; i < L; ++i) f();
            const double h = ms_since(a0) * 1000.0 / L;
            q->wait();
            auto b0 = Clock::now();
            for (int i = 0; i < L; ++i) f();
            q->wait();
            const double w = ms_since(b0) * 1000.0 / L;
            std::printf("  %-26s host %6.2f us/launch   back-to-back %7.2f us/launch\n", name, h, w);
        };
        const uint8_t* bd = blobs;
        per("gather_rows16 M=32", [&] { strata::prefill::gather_rows16(X, src, Xs, 32, N, q); });
        per("iq_dequant_expert_f16", [&] { strata::kernels::iq_dequant_expert_f16(gt, dt, bd, bd + up_off, bd + down_off, FF, N, dq_gu, dq_d, q); });
        per("oneMKL f16 GU M=32", [&] { gemm.f16(Xs, dq_gu, GUf, 32, 1280, N); });
        per("swiglu_interleaved M=32", [&] { strata::prefill::swiglu_interleaved(GUf, Hh, 32, q); });
        per("oneMKL f16 down M=32", [&] { gemm.f16(Hh, dq_d, out_a, 32, N, 640); });
        const uint8_t* ga[16]; const uint8_t* gb[16]; const uint8_t* gd[16]; int32_t gc[16];
        for (int i = 0; i < 16; ++i) { const uint8_t* p = blobs + (size_t) (i % n_exp) * blob_bytes; ga[i] = p; gb[i] = p + up_off; gd[i] = p + down_off; gc[i] = 32; }
        per("direct oneMKL GU M=32", [&] { mkl_f16(Xs, dq_gu, GUf, 32, 1280, N); });
        per("direct oneMKL down M=32", [&] { mkl_f16(Hh, dq_d, out_a, 32, N, 640); });
        {
            const uint8_t* bl[16];
            for (int i = 0; i < 16; ++i) bl[i] = blobs + (size_t) (i % n_exp) * blob_bytes;
            per("iq_dequant_experts_f16 x16", [&] { strata::kernels::iq_dequant_experts_f16(gt, dt, 16, bl, up_off, down_off, FF, N, wgu0, wd0, q); });
            const uint16_t* wg16[16]; const uint16_t* wd16[16]; int32_t c16[16];
            for (int i = 0; i < 16; ++i) { wg16[i] = wgu[i]; wd16[i] = wd[i]; c16[i] = 32; }
            per("xmx::grouped_f16 GU 16x32", [&] { strata::prefill::xmx::grouped_f16(Xs, wg16, c16, 16, GUf, 1280, (int) N, q); });
            per("xmx::grouped_f16 down 16x32", [&] { strata::prefill::xmx::grouped_f16(Hh, wd16, c16, 16, out_a, (int) N, 640, q); });
            BatchArgs& A = ba[0];
            for (int i = 0; i < 16; ++i) {
                A.ta[i] = oneapi::mkl::transpose::trans; A.tb[i] = oneapi::mkl::transpose::nontrans;
                A.m[i] = 1280; A.n[i] = 32; A.k[i] = N; A.lda[i] = N; A.ldb[i] = N; A.ldc[i] = 1280; A.gs[i] = 1;
                A.alpha[i] = 1.0f; A.beta[i] = 0.0f;
                A.a[i] = (const sycl::half*) wgu[i]; A.b[i] = (const sycl::half*) (Xs + i * 32 * N); A.c[i] = GUf + i * 32 * 1280;
            }
            per("gemm_batch GU 16 groups x32", [&] {
                oneapi::mkl::blas::column_major::gemm_batch(*q, A.ta, A.tb, A.m, A.n, A.k, A.alpha, A.a, A.lda, A.b, A.ldb,
                                                            A.beta, A.c, A.ldc, (int64_t) 16, A.gs,
                                                            oneapi::mkl::blas::compute_mode::unset); });
        }
        per("iq_xmx_grouped GU 16x32", [&] { strata::kernels::iq_xmx_grouped(gt, ga, gb, gc, 16, Xs, GUf, 1280, (int) N, q); });
        per("iq_xmx_grouped down 16x32", [&] { strata::kernels::iq_xmx_grouped(dt, gd, nullptr, gc, 16, Hh, out_a, (int) N, 640, q); });
    }
    return 0;
}
