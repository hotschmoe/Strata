// sycl/src/kernels/night_bf16_bench.cpp - STRATA_NIGHT_BF16 A/B: bf16_gemv_fp32_mmvf_multi (T > 1) / bf16_gemv_fp32_mmvf
// (T = 1) original vs the night KREG kernel at the decode shapes, NMAT distinct weight copies per graph (> L2), graph
// replay; us per call, GB/s of weights, bitwise comparison.
//     night_bf16_bench [T=1,2,3,4,6,8]
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/bf16_gemv.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace strata::kernels {
void night_bf16_set(int on);
bool night_bf16_gemv_pair(const float* x, int64_t ldx, const uint16_t* w1, float* y1, int64_t ldy1, int64_t n_out1,
                          const uint16_t* w2, float* y2, int64_t ldy2, int64_t n_out2, int64_t n_in, int n_tok, void* stream);
}
namespace k = strata::kernels;
namespace sx = sycl::ext::oneapi::experimental;

namespace {
uint64_t rs = 0x9E3779B97F4A7C15ull;
inline uint64_t xr() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
inline float urand() { return (float) ((double) (xr() >> 11) / 9007199254740992.0 * 2.0 - 1.0); }
std::vector<int> parse_list(const char* s) {
    std::vector<int> v; std::string t(s); size_t p = 0;
    while (p < t.size()) { size_t e = t.find(',', p); if (e == std::string::npos) e = t.size(); v.push_back(std::atoi(t.substr(p, e - p).c_str())); p = e + 1; }
    return v;
}
double time_graph(sycl::queue& q, int n, const std::function<void(int)>& f, int reps) {
    sx::command_graph g(q.get_context(), q.get_device());
    g.begin_recording(q);
    for (int i = 0; i < n; ++i) f(i);
    g.end_recording(q);
    auto ge = g.finalize();
    for (int i = 0; i < 2; ++i) q.ext_oneapi_graph(ge);
    q.wait();
    std::vector<double> us;
    for (int i = 0; i < reps; ++i) {
        sycl::event e0 = q.ext_oneapi_submit_barrier();
        q.ext_oneapi_graph(ge);
        sycl::event e1 = q.ext_oneapi_submit_barrier();
        e1.wait();
        us.push_back(double(e1.get_profiling_info<sycl::info::event_profiling::command_end>() -
                            e0.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1e3 / n);
    }
    std::sort(us.begin(), us.end());
    return us[us.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
    const std::vector<int> tl = parse_list(argc > 1 ? argv[1] : "1,2,3,4,6,8");
    sycl::queue q(sycl::gpu_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    struct Shape { int n_in, n_out; const char* name; };
    const Shape shapes[] = {{2560, 512, "router/idx_q"}, {2560, 128, "idx_k"}, {2560, 48, "ssm_ab"}, {2560, 2560, "2560sq"}, {640, 2560, "640x2560"}};
    const size_t POOL = (size_t) 512 << 20;
    uint16_t* wp = (uint16_t*) sycl::malloc_device(POOL, q);
    {
        std::vector<uint16_t> h(POOL / 2);
        for (auto& v : h) { const float f = urand() * 0.05f; uint32_t b; std::memcpy(&b, &f, 4); v = (uint16_t) (b >> 16); }
        q.memcpy(wp, h.data(), POOL).wait();
    }
    float* x = sycl::malloc_device<float>((size_t) 8 * 2560, q);
    {
        std::vector<float> h((size_t) 8 * 2560);
        for (auto& v : h) v = urand();
        q.memcpy(x, h.data(), h.size() * 4).wait();
    }
    float *y = sycl::malloc_device<float>((size_t) 8 * 2560, q), *y2 = sycl::malloc_device<float>((size_t) 8 * 2560, q);
    std::printf("night_bf16_bench: original bf16 mmvf vs STRATA_NIGHT_BF16 KREG; us/call, GB/s of weights, bits\n");
    for (const Shape& sh : shapes) {
        const size_t wb = (size_t) sh.n_in * sh.n_out * 2, slot = (wb + 255) & ~(size_t) 255;
        const int ndist = (int) std::min<size_t>(128, POOL / slot), nmat = std::max(ndist, 32);
        for (const int T : tl) {
            auto call = [&](const uint16_t* w, float* o) {
                if (T == 1) k::bf16_gemv_fp32_mmvf(x, w, o, sh.n_in, sh.n_out, &q);
                else k::bf16_gemv_fp32_mmvf_multi(x, sh.n_in, w, o, sh.n_out, sh.n_in, sh.n_out, T, &q);
            };
            std::vector<float> r[2]; double us[2];
            for (int nt = 0; nt < 2; ++nt) {
                k::night_bf16_set(nt);
                r[nt].resize((size_t) T * sh.n_out);
                float* o = nt ? y2 : y;
                q.memset(o, 0, r[nt].size() * 4).wait();
                call(wp, o); q.wait();
                q.memcpy(r[nt].data(), o, r[nt].size() * 4).wait();
                us[nt] = time_graph(q, nmat, [&](int m) { call(wp + (size_t) (m % ndist) * (slot / 2), o); }, 11);
            }
            k::night_bf16_set(-1);
            size_t neq = 0; double mabs = 0;
            for (size_t i = 0; i < r[0].size(); ++i) {
                if (std::memcmp(&r[0][i], &r[1][i], 4) != 0) ++neq;
                mabs = std::max(mabs, (double) std::fabs(r[0][i] - r[1][i]));
            }
            std::printf("%-12s %5d x %5d T %d  orig %7.2f us %6.1f GB/s  night %7.2f us %6.1f GB/s (%4.1f%%)  x%.2f  bits differ %zu/%zu (maxabs %.1e)\n",
                        sh.name, sh.n_in, sh.n_out, T, us[0], wb / us[0] / 1e3, us[1], wb / us[1] / 1e3, wb / us[1] / 1e3 / 6.08,
                        us[0] / us[1], neq, r[0].size(), mabs);
            std::fflush(stdout);
        }
    }
    // the QSA indexer k_proj (2560 -> 128) + q_proj (2560 -> 512) on the same rows: two calls vs one pair launch
    {
        const int N = 2560, NK = 128, NQ = 512;
        const size_t kb = (size_t) N * NK * 2, qb = (size_t) N * NQ * 2, slot = ((kb + qb) + 255) & ~(size_t) 255;
        const int ndist = (int) std::min<size_t>(128, POOL / slot);
        float *yk = sycl::malloc_device<float>((size_t) 8 * NK, q), *yq = sycl::malloc_device<float>((size_t) 8 * NQ, q);
        for (const int T : tl) {
            if (T < 2) continue;
            auto two = [&](int m) {
                const uint16_t* wk = wp + (size_t) (m % ndist) * (slot / 2); const uint16_t* wq = wk + (size_t) N * NK;
                k::night_bf16_set(0);
                k::bf16_gemv_fp32_mmvf_multi(x, N, wk, yk, NK, N, NK, T, &q);
                k::bf16_gemv_fp32_mmvf_multi(x, N, wq, yq, NQ, N, NQ, T, &q);
            };
            auto two_n = [&](int m) {
                const uint16_t* wk = wp + (size_t) (m % ndist) * (slot / 2); const uint16_t* wq = wk + (size_t) N * NK;
                k::night_bf16_set(1);
                k::bf16_gemv_fp32_mmvf_multi(x, N, wk, yk, NK, N, NK, T, &q);
                k::bf16_gemv_fp32_mmvf_multi(x, N, wq, yq, NQ, N, NQ, T, &q);
            };
            auto pair = [&](int m) {
                const uint16_t* wk = wp + (size_t) (m % ndist) * (slot / 2); const uint16_t* wq = wk + (size_t) N * NK;
                k::night_bf16_set(1);
                k::night_bf16_gemv_pair(x, N, wk, yk, NK, NK, wq, yq, NQ, NQ, N, T, &q);
            };
            std::vector<float> rk[2], rq[2];
            for (int v = 0; v < 2; ++v) {
                if (v) pair(0); else two(0);
                q.wait();
                rk[v].resize((size_t) T * NK); rq[v].resize((size_t) T * NQ);
                q.memcpy(rk[v].data(), yk, rk[v].size() * 4).wait(); q.memcpy(rq[v].data(), yq, rq[v].size() * 4).wait();
            }
            size_t neq = 0;
            for (size_t i = 0; i < rk[0].size(); ++i) neq += std::memcmp(&rk[0][i], &rk[1][i], 4) != 0;
            for (size_t i = 0; i < rq[0].size(); ++i) neq += std::memcmp(&rq[0][i], &rq[1][i], 4) != 0;
            const double t2 = time_graph(q, 64, two, 11), t2n = time_graph(q, 64, two_n, 11), tp = time_graph(q, 64, pair, 11);
            k::night_bf16_set(-1);
            std::printf("idx k+q pair  2560 x 128+512 T %d  orig 2 calls %7.2f us  night 2 calls %7.2f us  night pair %7.2f us  x%.2f  bits differ %zu\n",
                        T, t2, t2n, tp, t2 / tp, neq);
            std::fflush(stdout);
        }
    }
    return 0;
}
