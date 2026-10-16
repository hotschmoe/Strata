// src/kernels/gdn_chunk_bench.cpp - night-lab STRATA_NIGHT_GDN_CHUNK: the prompt GDN recurrence, token scan
// (gdn_recurrence_variant 0, the production kernels on the B70) against the chunkwise form (gdn_recurrence_chunked):
// final state, raw outputs y and the normalized FP16 outputs, max abs / rel error and time.
// Inputs as gdn_parity's prefill fixture: S=128, HK=16, HV=48 (value head h reads key head h % 16), nonzero initial
// state, q/k ~ N(0, 0.08^2) per element (unit-norm-ish heads), v ~ N(0, 0.7^2), z ~ N(0,1), gamma ~ 1 + 0.2 N,
// beta in [0.1, 0.9], log decay -(0.02 | 2.0) * (1 + |N|) (shallow / deep).
//   gdn_chunk_bench [--T 12288] [--seed 1] [--reps 3] [--variant 0]
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/prefill/kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    int T = 12288, seed = 1, reps = 3, variant = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--T") && i + 1 < argc) T = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--variant") && i + 1 < argc) variant = std::atoi(argv[++i]);
    }
    constexpr int S = 128, HK = 16, HV = 48, C = 10240;
    sycl::queue q(sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order{}});
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> st(size_t(S) * HV * S), h(size_t(T) * C), gate(size_t(T) * HV), beta(gate.size()),
        z(size_t(T) * HV * S), gamma(S);
    for (auto& x : st) x = gauss(rng) * 0.03f;
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < C; ++j) h[size_t(t) * C + j] = gauss(rng) * (j < 2 * HK * S ? 0.08f : 0.7f);
    for (auto& x : gate) x = -(seed % 2 ? 0.02f : 2.0f) * (1.0f + std::fabs(gauss(rng)));
    for (auto& x : beta) x = 0.1f + float(rng() % 800) / 1000.0f;
    for (auto& x : z) x = gauss(rng);
    for (auto& x : gamma) x = 1.0f + 0.2f * gauss(rng);
    auto copy = [&](const std::vector<float>& src) {
        float* p = sycl::malloc_device<float>(src.size(), q);
        q.memcpy(p, src.data(), src.size() * 4).wait();
        return p;
    };
    float *dst = copy(st), *dh = copy(h), *dg = copy(gate), *db = copy(beta), *dz = copy(z), *dgm = copy(gamma);
    float* dy = sycl::malloc_device<float>(z.size(), q);
    uint16_t* dy16 = sycl::malloc_device<uint16_t>(z.size(), q);
    std::vector<float> st_a(st.size()), st_b(st.size()), y_a(z.size()), y_b(z.size());
    std::vector<uint16_t> h_a(z.size()), h_b(z.size());
    auto run = [&](bool chunked, std::vector<float>& ost, std::vector<float>& oy, std::vector<uint16_t>& o16) {
        double best = 1e30;
        for (int r = 0; r <= reps; ++r) {
            q.memcpy(dst, st.data(), st.size() * 4).wait();
            const auto t0 = std::chrono::steady_clock::now();
            if (chunked) {
                if (!strata::prefill::gdn_recurrence_chunked(dst, dh, dg, db, dz, dgm, 1e-6f, dy, dy16, T, &q)) {
                    std::fprintf(stderr, "chunked refused\n");
                    std::exit(3);
                }
            } else {
                strata::prefill::gdn_recurrence_variant(variant, dst, dh, dg, db, dz, dgm, 1e-6f, dy, dy16, T, &q);
            }
            q.wait_and_throw();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (r > 0) best = std::min(best, ms);   // run 0 warms up
        }
        q.memcpy(ost.data(), dst, ost.size() * 4).wait();
        q.memcpy(oy.data(), dy, oy.size() * 4).wait();
        q.memcpy(o16.data(), dy16, o16.size() * 2).wait();
        return best;
    };
    const double ta = run(false, st_a, y_a, h_a);
    const double tb = run(true, st_b, y_b, h_b);
    auto cmp = [](const char* what, const std::vector<float>& a, const std::vector<float>& b) {
        double mabs = 0, mref = 0, sabs = 0, srel = 0;
        size_t bad = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (!std::isfinite(b[i])) { ++bad; continue; }
            const double d = std::fabs((double) a[i] - b[i]);
            mabs = std::max(mabs, d);
            mref = std::max(mref, std::fabs((double) a[i]));
            sabs += d;
            srel += std::fabs((double) a[i]);
        }
        std::printf("  %-12s max abs %.3e  (max |ref| %.3e -> %.3e of it)  rel L1 %.3e  non-finite %zu\n", what, mabs,
                    mref, mref > 0 ? mabs / mref : 0.0, srel > 0 ? sabs / srel : 0.0, bad);
    };
    std::vector<float> fa(h_a.size()), fb(h_b.size());
    for (size_t i = 0; i < h_a.size(); ++i) {
        fa[i] = (float) sycl::bit_cast<sycl::half>(h_a[i]);
        fb[i] = (float) sycl::bit_cast<sycl::half>(h_b[i]);
    }
    std::printf("T=%d seed=%d (%s decay): scan variant %d %.3f ms, chunked %.3f ms -> %.2fx\n", T, seed,
                seed % 2 ? "shallow" : "deep", variant, ta, tb, ta / tb);
    cmp("state", st_a, st_b);
    cmp("y (raw o)", y_a, y_b);
    cmp("y16 (normed)", fa, fb);
    std::printf("RESULT T=%d seed=%d scan_ms=%.3f chunk_ms=%.3f\n", T, seed, ta, tb);
    return 0;
}
