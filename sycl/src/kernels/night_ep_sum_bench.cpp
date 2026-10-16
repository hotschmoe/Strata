// night_ep_sum_bench (night lab 2026-10-16): STRATA_NIGHT_PREFILL_F16SUM's kernels against the FP32 reduce/finish on
// ONE card (the "remote" sum region is local here, so the PCIe time is not measured: the bytes halve), and the
// STRATA_NIGHT_PREFILL_EARLY doorbell kernels (signal on one queue, wait on another).
//   night_ep_sum_bench [T=12288] [peer_frac=0.7] [outlier_scale=100]
#include "strata/kernels/ep_kernels.hpp"
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
#include <vector>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point a) { return std::chrono::duration<double, std::milli>(Clock::now() - a).count(); }

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const int64_t T = argc > 1 ? std::atoll(argv[1]) : 12288;
    const double peer_frac = argc > 2 ? std::atof(argv[2]) : 0.7;
    const float outlier = argc > 3 ? (float) std::atof(argv[3]) : 100.0f;
    const int64_t N = 2560;
    const int K = 10;
    sycl::queue* q = &dpct::get_in_order_queue();
    std::mt19937 rng(7);
    // grouped rows: local rows first, then peer rows; every (t, k) gets a distinct row
    const int64_t R = T * K;
    std::vector<int32_t> slot((size_t) R);
    std::vector<char> is_peer((size_t) R);
    int64_t n_peer = 0;
    std::uniform_real_distribution<double> u01(0, 1);
    for (int64_t i = 0; i < R; ++i) { is_peer[i] = u01(rng) < peer_frac; n_peer += is_peer[i]; }
    const int64_t local_rows = R - n_peer;
    {
        std::vector<int64_t> perm_l, perm_p;
        for (int64_t i = 0; i < local_rows; ++i) perm_l.push_back(i);
        for (int64_t i = 0; i < n_peer; ++i) perm_p.push_back(local_rows + i);
        std::shuffle(perm_l.begin(), perm_l.end(), rng);
        std::shuffle(perm_p.begin(), perm_p.end(), rng);
        int64_t a = 0, b = 0;
        for (int64_t i = 0; i < R; ++i) slot[i] = (int32_t) (is_peer[i] ? perm_p[b++] : perm_l[a++]);
    }
    std::vector<float> w((size_t) R), gate((size_t) T);
    for (int64_t t = 0; t < T; ++t) {
        double s = 0; for (int k = 0; k < K; ++k) { w[t * K + k] = (float) u01(rng); s += w[t * K + k]; }
        for (int k = 0; k < K; ++k) w[t * K + k] = (float) (w[t * K + k] / s);
        gate[t] = (float) (u01(rng) * 4 - 2);
    }
    // expert rows: N(0, 0.05) with 0.1% outliers x outlier_scale (massive-activation style), and some tiny rows
    std::normal_distribution<float> nd(0.f, 0.05f);
    std::vector<float> rows((size_t) R * N), shared((size_t) T * N);
    for (size_t i = 0; i < rows.size(); ++i) { rows[i] = nd(rng); if (u01(rng) < 1e-3) rows[i] *= outlier; }
    for (int64_t r = 0; r < R; r += 997) for (int64_t i = 0; i < N; ++i) rows[(size_t) r * N + i] *= 1e-6f;
    for (auto& x : shared) x = nd(rng);
    auto dev = [&](auto& v) { using E = typename std::decay_t<decltype(v)>::value_type; E* p = sycl::malloc_device<E>(v.size(), *q); q->memcpy(p, v.data(), v.size() * sizeof(E)); return p; };
    float* d_rows = dev(rows); float* d_w = dev(w); float* d_gate = dev(gate); float* d_shared = dev(shared);
    int32_t* d_slot = dev(slot);
    float* sum32 = sycl::malloc_device<float>((size_t) T * N, *q);
    float* bo_a = sycl::malloc_device<float>((size_t) T * N, *q);
    float* bo_b = sycl::malloc_device<float>((size_t) T * N, *q);
    uint8_t* sumh = sycl::malloc_device<uint8_t>((size_t) T * N * 4, *q);
    q->wait();
    std::printf("T %lld rows %lld (peer %lld)\n", (long long) T, (long long) R, (long long) n_peer);
    auto ref = [&](bool push) {
        strata::kernels::ep_prefill_reduce(d_rows + (size_t) local_rows * N, d_slot, d_w, sum32, T, local_rows, N, K, q, push);
        strata::kernels::ep_copy_uncached(bo_a, sum32, (size_t) T * N * 4, q);
        strata::kernels::ep_prefill_finish(d_rows, d_slot, d_w, d_shared, d_gate, bo_a, T, local_rows, N, K, q);
    };
    auto mine = [&](bool push) {
        strata::kernels::ep_prefill_reduce_h(d_rows + (size_t) local_rows * N, d_slot, d_w, (uint16_t*) sumh,
                                             (float*) (sumh + (size_t) T * N * 2), T, local_rows, N, K, q, push);
        strata::kernels::ep_prefill_finish_h(d_rows, d_slot, d_w, d_shared, d_gate, (const uint16_t*) sumh,
                                             (const float*) (sumh + (size_t) T * N * 2), bo_b, T, local_rows, N, K, q);
    };
    auto tm = [&](auto f) { f(); q->wait(); double best = 1e30; for (int r = 0; r < 5; ++r) { auto a = Clock::now(); f(); q->wait(); best = std::min(best, ms_since(a)); } return best; };
    const double t_red32 = tm([&] { strata::kernels::ep_prefill_reduce(d_rows + (size_t) local_rows * N, d_slot, d_w, sum32, T, local_rows, N, K, q, true); });
    const double t_redh = tm([&] { strata::kernels::ep_prefill_reduce_h(d_rows + (size_t) local_rows * N, d_slot, d_w, (uint16_t*) sumh, (float*) (sumh + (size_t) T * N * 2), T, local_rows, N, K, q, true); });
    const double t_fin32 = tm([&] { strata::kernels::ep_copy_uncached(bo_a, sum32, (size_t) T * N * 4, q); strata::kernels::ep_prefill_finish(d_rows, d_slot, d_w, d_shared, d_gate, bo_a, T, local_rows, N, K, q); });
    const double t_finh = tm([&] { strata::kernels::ep_prefill_finish_h(d_rows, d_slot, d_w, d_shared, d_gate, (const uint16_t*) sumh, (const float*) (sumh + (size_t) T * N * 2), bo_b, T, local_rows, N, K, q); });
    std::printf("push-mode reduce (64 groups, local target): FP32 %.3f ms (%.1f MB out) | FP16 %.3f ms (%.1f MB out)\n",
                t_red32, T * N * 4 / 1e6, t_redh, (T * N * 2 + T * 4) / 1e6);
    std::printf("card-0 side: copy_uncached + finish %.3f ms | finish_h %.3f ms\n", t_fin32, t_finh);
    ref(true); mine(true); q->wait();
    std::vector<float> a((size_t) T * N), b((size_t) T * N), s32((size_t) T * N);
    q->memcpy(a.data(), bo_a, a.size() * 4); q->memcpy(b.data(), bo_b, b.size() * 4); q->memcpy(s32.data(), sum32, s32.size() * 4).wait();
    // error of the peer sum itself and of the layer output
    std::vector<uint16_t> hh((size_t) T * N); std::vector<float> sc((size_t) T);
    q->memcpy(hh.data(), sumh, hh.size() * 2); q->memcpy(sc.data(), sumh + (size_t) T * N * 2, T * 4).wait();
    double max_rel_sum = 0, mean_rel_sum = 0, max_abs_out = 0, max_out = 0, rms_d = 0, rms_o = 0; size_t nz = 0, nonfin = 0;
    for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < N; ++i) {
            const size_t j = (size_t) t * N + i;
            const float hv = (float) sycl::bit_cast<sycl::half>(hh[j]) * sc[t];
            if (!std::isfinite(hv) || !std::isfinite(b[j])) { ++nonfin; continue; }
            if (s32[j] != 0.0f) { const double r = std::fabs((hv - s32[j]) / s32[j]); max_rel_sum = std::max(max_rel_sum, r); mean_rel_sum += r; ++nz; }
            const double d = std::fabs((double) a[j] - b[j]);
            max_abs_out = std::max(max_abs_out, d); max_out = std::max(max_out, (double) std::fabs(a[j]));
            rms_d += d * d; rms_o += (double) a[j] * a[j];
        }
    std::printf("peer sum FP16xscale vs FP32: max rel %.3e, mean rel %.3e (2^-11 = %.3e); layer output: max abs %.3e "
                "(max |out| %.3e), rms err / rms out %.3e; non-finite %zu\n",
                max_rel_sum, mean_rel_sum / std::max<size_t>(nz, 1), std::ldexp(1.0, -11), max_abs_out, max_out,
                std::sqrt(rms_d / std::max(rms_o, 1e-30)), nonfin);
    // doorbell on ONE card: a cross-queue spin deadlocks on one compute engine, so this checks the semantics only:
    // ep_push_signal copies bit-exactly and raises the flag to the value (counter back to 0); ep_wait_value returns at
    // once on a flag already at/after the value and gives up with err at the bound otherwise.
    {
        uint32_t* flag = sycl::malloc_device<uint32_t>(1, *q);
        uint32_t* done = sycl::malloc_device<uint32_t>(1, *q);
        uint32_t* err = sycl::malloc_host<uint32_t>(1, *q);
        uint8_t* dst = sycl::malloc_device<uint8_t>((size_t) T * N * 2, *q);
        q->memset(flag, 0, 4); q->memset(done, 0, 4); q->memset(dst, 0, (size_t) T * N * 2); q->wait(); *err = 0;
        const double tp = tm([&] { strata::kernels::ep_push_signal(dst, d_rows, T * N / 8, done, flag, 3u, q); });
        std::vector<uint8_t> hd((size_t) T * N * 2);
        uint32_t hf = 0, hdn = 0;
        q->memcpy(hd.data(), dst, hd.size()); q->memcpy(&hf, flag, 4); q->memcpy(&hdn, done, 4).wait();
        const bool same = std::memcmp(hd.data(), rows.data(), hd.size()) == 0;
        std::printf("ep_push_signal (local target): %.3f ms for %.1f MB, copy %s, flag %u (want 3), counter %u (want 0)\n",
                    tp, hd.size() / 1e6, same ? "bit-exact" : "WRONG", hf, hdn);
        auto a0 = Clock::now();
        strata::kernels::ep_wait_value(flag, 2u, 50000000u, err, 1u, q);
        strata::kernels::ep_wait_value(flag, 3u, 50000000u, err, 2u, q);
        q->wait();
        std::printf("ep_wait_value on flag 3 for 2 and 3: %.3f ms, err %u (want 0)\n", ms_since(a0), *err);
        a0 = Clock::now();
        strata::kernels::ep_wait_value(flag, 4u, 200000u, err, 3u, q);   // never signalled: must give up with err = 3
        q->wait();
        std::printf("ep_wait_value bound: err %u (want 3) after %.1f ms (200000 spins)\n", *err, ms_since(a0));
    }
    return 0;
}
