// src/kernels/qsa_pattn_bench.cpp - night-lab STRATA_NIGHT_PATTN_XMX: the prompt path's QSA attention for one prefill
// chunk of one layer, current kernels (the prefill fallback loop: the S93 value pipeline for full-width 32-query
// batches, else qsa_decode_attn_batch) against the union-tile XMX kernel (qsa_prompt_attn_union), plus the selection
// union sizes per query tile.
//
// Realistic shapes: 24 q heads, 2 KV heads, head_dim 256, INT8 K/V (fp16 64-value scales), page size 4, identity page
// table (resident pools); indexer 4 x 128, pooled keys random, indexer queries an AR(1) sequence over positions
// (--rho: neighbouring positions' selections overlap as in a real prompt; calibrate against INTEL.md's 80K numbers),
// selections by the real block-scores + top-k kernels at the serving capacity (max_blocks 50002), 256 queries per call
// as the prefill does.  Or --sel FILE: a STRATA_DUMP_SEL dump (int32 T, cap, T*cap ids, T widths) replaces them.
//
//   qsa_pattn_bench --p0 12288 --T 12288 [--rho 0.97] [--reps 3] [--sel dump.bin]
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prefill_attn.hpp"
#include "strata/kernels/qsa_prompt_attn_union.hpp"
#include "strata/kernels/qsa_prompt_attn_xmx.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {
inline uint32_t hsh(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
template <typename T>
T* dalloc(sycl::queue& q, size_t n) {
    T* p = sycl::malloc_device<T>(n + 64, q);
    if (!p) { std::fprintf(stderr, "malloc_device failed (%zu)\n", n); std::exit(2); }
    q.memset(p, 0, (n + 64) * sizeof(T)).wait();
    return p;
}
}  // namespace

int main(int argc, char** argv) {
    int64_t p0 = 12288, T = 12288;
    double rho = 0.97;
    int reps = 3;
    std::string sel_path;
    bool no_old = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--p0") && i + 1 < argc) p0 = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "--T") && i + 1 < argc) T = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "--rho") && i + 1 < argc) rho = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--sel") && i + 1 < argc) sel_path = argv[++i];
        else if (!std::strcmp(argv[i], "--no-old")) no_old = true;
    }
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order()}};
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NH = s.n_head, NKV = s.n_head_kv, ZV = NH * HD, ID = s.idx_dim, IQ = s.idx_n_head;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    const int64_t max_blocks = 200000 / s.idx_block + 2;
    const int64_t cells = ((p0 + T + 63) / 64) * 64, pages = cells / s.page_size;
    std::printf("device %s | chunk p0=%lld T=%lld (context %lld..%lld) cap %lld rho %.3f\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), (long long) p0, (long long) T,
                (long long) p0 + 1, (long long) (p0 + T), (long long) cap, rho);

    // ---- KV pools (INT8, identity pages), random codes and scales
    const size_t rows = (size_t) pages * NKV * s.page_size;
    int8_t* k_q = dalloc<int8_t>(q, rows * HD);
    int8_t* v_q = dalloc<int8_t>(q, rows * HD);
    uint16_t* k_s = dalloc<uint16_t>(q, rows * (HD / 64));
    uint16_t* v_s = dalloc<uint16_t>(q, rows * (HD / 64));
    int32_t* pt = dalloc<int32_t>(q, (size_t) pages);
    q.parallel_for(sycl::range<1>(rows * HD), [=](sycl::id<1> i) {
        k_q[i] = (int8_t) ((int) (hsh((uint32_t) i[0] * 2u + 1u) % 255u) - 127);
        v_q[i] = (int8_t) ((int) (hsh((uint32_t) i[0] * 2u + 2u) % 255u) - 127);
    });
    q.parallel_for(sycl::range<1>(rows * (HD / 64)), [=](sycl::id<1> i) {
        k_s[i] = sycl::bit_cast<uint16_t>(sycl::half(0.002f + (float) (hsh((uint32_t) i[0] ^ 0xabcdu) % 1000u) * 2e-5f));
        v_s[i] = sycl::bit_cast<uint16_t>(sycl::half(0.002f + (float) (hsh((uint32_t) i[0] ^ 0x1234u) % 1000u) * 2e-5f));
    });
    q.parallel_for(sycl::range<1>((size_t) pages), [=](sycl::id<1> i) { pt[i] = (int32_t) i[0]; });
    k::QsaAttnPools pools;
    pools.k_q = k_q; pools.v_q = v_q; pools.k_scale = k_s; pools.v_scale = v_s; pools.page_table = pt;

    // ---- steps and selections
    std::vector<int32_t> h_steps((size_t) T * k::kStepCount);
    for (int64_t t = 0; t < T; ++t) k::qsa_step_fill(h_steps.data() + t * k::kStepCount, p0 + t, s);
    int32_t* steps = dalloc<int32_t>(q, h_steps.size());
    q.memcpy(steps, h_steps.data(), h_steps.size() * 4).wait();
    int32_t* sel = dalloc<int32_t>(q, (size_t) T * cap);
    std::vector<int32_t> h_sel((size_t) T * cap);
    if (!sel_path.empty()) {
        FILE* f = std::fopen(sel_path.c_str(), "rb");
        int32_t hdr[2] = {0, 0};
        if (!f || std::fread(hdr, 4, 2, f) != 2 || hdr[0] != T || hdr[1] != cap) { std::fprintf(stderr, "bad --sel file (want T=%lld cap=%lld)\n", (long long) T, (long long) cap); return 2; }
        if (std::fread(h_sel.data(), 4, h_sel.size(), f) != h_sel.size()) { std::fprintf(stderr, "short --sel file\n"); return 2; }
        std::fclose(f);
        q.memcpy(sel, h_sel.data(), h_sel.size() * 4).wait();
    } else {
        const int64_t nblk = cells / s.idx_block + 2;
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> h_pooled((size_t) nblk * ID), h_dead((size_t) ID), h_qi((size_t) T * IQ * ID);
        for (auto& v : h_pooled) v = nd(rng);
        for (auto& v : h_dead) v = nd(rng);
        std::vector<float> cur((size_t) IQ * ID);
        for (auto& v : cur) v = nd(rng);
        const float a = (float) rho, b = (float) std::sqrt(std::max(0.0, 1.0 - rho * rho));
        for (int64_t t = 0; t < T; ++t) {
            for (auto& v : cur) v = a * v + b * nd(rng);
            std::copy(cur.begin(), cur.end(), h_qi.begin() + t * IQ * ID);
        }
        float* pooled = dalloc<float>(q, h_pooled.size());
        float* dead = dalloc<float>(q, h_dead.size());
        float* qidx = dalloc<float>(q, h_qi.size());
        float* scores = dalloc<float>(q, (size_t) 256 * max_blocks);
        q.memcpy(pooled, h_pooled.data(), h_pooled.size() * 4);
        q.memcpy(dead, h_dead.data(), h_dead.size() * 4);
        q.memcpy(qidx, h_qi.data(), h_qi.size() * 4).wait();
        for (int64_t t0 = 0; t0 < T; t0 += 256) {
            const int64_t nb = std::min<int64_t>(256, T - t0);
            const int64_t active = (int64_t) h_steps[(size_t) ((t0 + nb - 1) * k::kStepCount + k::kStepNBid)] + 1;
            k::qsa_block_scores(pooled, dead, qidx + t0 * IQ * ID, steps + t0 * k::kStepCount, nb, max_blocks, s, scores, &q, active);
            k::qsa_block_topk(scores, steps + t0 * k::kStepCount, nb, max_blocks, cap, s, sel + t0 * cap, &q, active);
        }
        q.wait();
        q.memcpy(h_sel.data(), sel, h_sel.size() * 4).wait();
        sycl::free(pooled, q); sycl::free(dead, q); sycl::free(qidx, q); sycl::free(scores, q);
    }
    // union statistics per query tile
    for (int tq : {4, 8, 16}) {
        double su = 0, sw = 0;
        std::vector<int32_t> u;
        for (int64_t t0 = 0; t0 + tq <= T; t0 += tq) {
            u.clear();
            for (int64_t t = t0; t < t0 + tq; ++t) {
                const int64_t w = h_steps[(size_t) (t * k::kStepCount + k::kStepWidth)];
                sw += (double) w;
                u.insert(u.end(), h_sel.begin() + t * cap, h_sel.begin() + t * cap + w);
            }
            std::sort(u.begin(), u.end());
            su += (double) (std::unique(u.begin(), u.end()) - u.begin());
        }
        std::printf("union of %2d-query tiles: %.2fx one query's cells (%.1f%% of the per-query reads)\n", tq,
                    su / (sw / tq), 100.0 * su / sw);
    }

    // ---- q (T x 24 x 256), random ~N(0,1)-ish
    float* qv = dalloc<float>(q, (size_t) T * ZV);
    q.parallel_for(sycl::range<1>((size_t) T * ZV), [=](sycl::id<1> i) {
        const uint32_t h1 = hsh((uint32_t) i[0] * 3u + 7u), h2 = hsh((uint32_t) i[0] * 5u + 11u);
        qv[i] = ((float) (h1 & 0xffffu) / 65536.0f + (float) (h2 & 0xffffu) / 65536.0f - 1.0f) * 2.4f;
    }).wait();
    float* out_a = dalloc<float>(q, (size_t) T * ZV);
    float* out_b = dalloc<float>(q, (size_t) T * ZV);
    const int64_t attn_batch = 32;
    float* scratch = dalloc<float>(q, (size_t) attn_batch * k::qsa_decode_attn_scratch_floats(cap, s));

    auto run_old = [&] {
        for (int64_t t0 = 0; t0 < T; t0 += attn_batch) {
            const int64_t nb = std::min(attn_batch, T - t0);
            const bool value_domain = nb == 32 && cap == 2051 &&
                                      h_steps[(size_t) (t0 * k::kStepCount + k::kStepWidth)] == cap;
            if (value_domain)
                k::qsa_prefill_attn_value_batch(qv + t0 * ZV, pools, sel + t0 * cap, steps + t0 * k::kStepCount, cap, s,
                                                scratch, out_a + t0 * ZV, nb, &q);
            else
                k::qsa_decode_attn_batch(qv + t0 * ZV, pools, sel + t0 * cap, steps + t0 * k::kStepCount, cap, s, scratch,
                                         out_a + t0 * ZV, nb, &q);
        }
    };
    auto run_new = [&] {
        if (!k::qsa_prompt_attn_union(qv, pools, sel, steps, cap, s, out_b, T, &q)) { std::fprintf(stderr, "union kernel refused\n"); std::exit(3); }
    };
    auto timeit = [&](auto&& f) {
        f(); q.wait();
        double best = 1e30;
        for (int r = 0; r < reps; ++r) { const double t0 = now_ms(); f(); q.wait(); best = std::min(best, now_ms() - t0); }
        return best;
    };
    double t_old = 0;
    if (!no_old) t_old = timeit(run_old);
    const double t_new = timeit(run_new);
    double t_x = -1;
    if (std::getenv("STRATA_PROMPT_ATTN_XMX")) {
        float* out_c = out_b;   // overwritten after the comparison below
        (void) out_c;
    }
    std::printf("RESULT p0=%lld T=%lld old_ms=%.2f union_xmx_ms=%.2f speedup=%.2fx (us/query old %.2f new %.2f)\n",
                (long long) p0, (long long) T, t_old, t_new, t_old > 0 ? t_old / t_new : 0.0, 1e3 * t_old / T,
                1e3 * t_new / T);
    if (!no_old) {
        std::vector<float> a((size_t) T * ZV), b((size_t) T * ZV);
        q.memcpy(a.data(), out_a, a.size() * 4);
        q.memcpy(b.data(), out_b, b.size() * 4).wait();
        double mabs = 0, mref = 0, sabs = 0, mrel_row = 0;
        size_t nan = 0, worst = 0;
        for (int64_t r = 0; r < T * NH; ++r) {
            double rmax = 0, rdiff = 0;
            for (int64_t d = 0; d < HD; ++d) {
                const size_t i = (size_t) r * HD + d;
                if (!std::isfinite(b[i])) { ++nan; continue; }
                const double df = std::fabs((double) a[i] - (double) b[i]);
                rmax = std::max(rmax, std::fabs((double) a[i]));
                rdiff = std::max(rdiff, df);
                sabs += df;
                if (df > mabs) { mabs = df; worst = i; }
            }
            mref = std::max(mref, rmax);
            if (rmax > 0) mrel_row = std::max(mrel_row, rdiff / rmax);
        }
        std::printf("ACCURACY union vs current: max abs %.3e (at %zu: %.6f vs %.6f), mean abs %.3e, max |ref| %.3e, "
                    "max per-row rel (max diff / max |ref| of the row) %.3e, non-finite %zu\n",
                    mabs, worst, a[worst], b[worst], sabs / (double) a.size(), mref, mrel_row, nan);
    }
    if (std::getenv("STRATA_PROMPT_ATTN_XMX")) {
        auto run_x = [&] { k::qsa_prompt_attn_xmx(qv, pools, sel, steps, cap, s, out_b, T, &q); };
        t_x = timeit(run_x);
        std::printf("RESULT p0=%lld T=%lld xmx_v2_ms=%.2f\n", (long long) p0, (long long) T, t_x);
    }
    return 0;
}
