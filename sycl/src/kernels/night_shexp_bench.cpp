// sycl/src/kernels/night_shexp_bench.cpp - STRATA_NIGHT_SHEXP A/B: shared_expert_multi (native bf16 gate path, x_q8_1
// ready as in the verifier) with the original launches vs the fused night path, for the (gate, up, down) type triples
// of the Flash-Next pack, NL distinct layers per graph (> L2), in-order queue graph replay; us per layer call and the
// max abs / bitwise differences of `out`.
//     night_shexp_bench [ncols=1,4,6,8] [mmvq_level=2]
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/shared_expert.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "strata/kernels/bf16_gemv.hpp"
namespace strata::kernels {
void native_mmvq_set_night(int on); void night_shexp_set(int on);
bool night_shexp_gu_swiglu_q8(int gate_type, const void* wg, int up_type, const void* wu, const void* x_q8_1, float* h,
                              void* q8, int n_embd, int n_ff, int ncols, void* stream, const float* xf, const uint16_t* wrow,
                              float* graw);
bool night_shexp_down_scale(int down_type, const void* wd, const void* q8, const float* graw, float* out, int n_ff, int n_embd,
                            int ncols, void* stream);
}
namespace k = strata::kernels;
namespace sx = sycl::ext::oneapi::experimental;

namespace {
uint64_t rs = 0x9E3779B97F4A7C15ull;
inline uint64_t xr() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
size_t bsz(int t) { return t == 12 ? 144 : t == 13 ? 176 : t == 14 ? 210 : t == 23 ? 136 : t == 21 ? 110 : t == 20 ? 18 : t == 42 ? 18 : 34; }
void fill(int t, uint8_t* h, size_t n) {
    uint64_t* p = (uint64_t*) h;
    for (size_t i = 0; i < n / 8; ++i) p[i] = xr();
    const size_t bs = bsz(t);
    for (size_t o = 0; o + bs <= n; o += bs) {
        const uint16_t d = (uint16_t) (0x1400 | (xr() & 0x3ff)), m = (uint16_t) (0x1400 | (xr() & 0x3ff));
        std::memcpy(h + o + (t == 14 ? 208 : 0), &d, 2);
        if (t == 12 || t == 13) std::memcpy(h + o + 2, &m, 2);
    }
}
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
    const std::vector<int> ncl = parse_list(argc > 1 ? argv[1] : "1,4,6,8");
    const int level = argc > 2 ? std::atoi(argv[2]) : 2;
    sycl::queue q(sycl::gpu_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    const int N = 2560, FF = 640, NL = 48;
    struct Combo { int g, u, d, layers; };
    const Combo combos[] = {{23, 21, 20, 8}, {21, 21, 20, 7}, {12, 21, 20, 8}, {23, 23, 20, 6}, {21, 23, 20, 3}, {23, 12, 20, 2},
                            {23, 14, 20, 2}, {12, 12, 20, 2}, {21, 21, 42, 2}, {23, 14, 42, 1}};
    k::shared_expert_set_native_bf16(true);
    k::native_mmvq_set_night(level);
    std::printf("night_shexp_bench: %d layers per graph, mmvq level %d; us per shared-expert call: orig (6 launches) vs night\n", NL, level);
    float* x = sycl::malloc_device<float>((size_t) 8 * N, q);
    std::vector<float> hx((size_t) 8 * N);
    for (auto& v : hx) v = (float) ((double) (xr() >> 11) / 9007199254740992.0 * 2.0 - 1.0);
    q.memcpy(x, hx.data(), hx.size() * 4).wait();
    void* xq = sycl::malloc_device(k::native_q8_1_bytes(N, 8) + 256, q);
    float *gate = sycl::malloc_device<float>((size_t) 8 * FF, q), *up = sycl::malloc_device<float>((size_t) 8 * FF, q);
    void* q8 = sycl::malloc_device(k::native_q8_1_bytes(FF, 8) + 256, q);
    float *g = sycl::malloc_device<float>(64, q), *out = sycl::malloc_device<float>((size_t) 8 * N, q), *out2 = sycl::malloc_device<float>((size_t) 8 * N, q);
    uint16_t* gi = sycl::malloc_device<uint16_t>((size_t) NL * N, q);
    {
        std::vector<uint16_t> h((size_t) NL * N);
        for (auto& v : h) { const float f = (float) ((double) (xr() >> 11) / 9007199254740992.0 - 0.5) * 0.05f; uint32_t b; std::memcpy(&b, &f, 4); v = (uint16_t) (b >> 16); }
        q.memcpy(gi, h.data(), h.size() * 2).wait();
    }
    for (const Combo& c : combos) {
        const size_t gb = k::native_mmvq_weight_bytes(c.g, N, FF), ub = k::native_mmvq_weight_bytes(c.u, N, FF), db = k::native_mmvq_weight_bytes(c.d, FF, N);
        const size_t gs = (gb + 255) & ~(size_t) 255, us_ = (ub + 255) & ~(size_t) 255, ds = (db + 255) & ~(size_t) 255;
        uint8_t *wg = (uint8_t*) sycl::malloc_device(gs * NL, q), *wu = (uint8_t*) sycl::malloc_device(us_ * NL, q), *wd = (uint8_t*) sycl::malloc_device(ds * NL, q);
        std::vector<uint8_t> h(std::max(gs, std::max(us_, ds)) * NL);
        fill(c.g, h.data(), gs * NL); q.memcpy(wg, h.data(), gs * NL).wait();
        fill(c.u, h.data(), us_ * NL); q.memcpy(wu, h.data(), us_ * NL).wait();
        fill(c.d, h.data(), ds * NL); q.memcpy(wd, h.data(), ds * NL).wait();
        for (const int nc : ncl) {
            k::native_quantize_q8_1(x, xq, N, nc, &q);
            q.wait();
            auto call = [&](int l, float* o) {
                k::NativeSharedWeights nw{};
                nw.gate_type = c.g; nw.gate_data = wg + (size_t) l * gs; nw.up_type = c.u; nw.up_data = wu + (size_t) l * us_;
                nw.down_type = c.d; nw.down_data = wd + (size_t) l * ds; nw.q8_1 = q8;
                k::shared_expert_multi(nc, x, nullptr, nw, gi + (size_t) l * N, gate, up, g, o, N, FF, &q, xq, 0);
            };
            double t[2]; std::vector<float> r[2];
            for (int nt = 0; nt < 2; ++nt) {
                k::night_shexp_set(nt);
                r[nt].resize((size_t) nc * N);
                float* o = nt ? out2 : out;
                q.memset(o, 0, (size_t) nc * N * 4).wait();
                call(0, o); q.wait();
                q.memcpy(r[nt].data(), o, (size_t) nc * N * 4).wait();
                t[nt] = time_graph(q, NL, [&](int l) { call(l, o); }, 9);
            }
            k::night_shexp_set(-1);
            if (std::getenv("SHEXP_PROFILE")) {   // per-launch breakdown, NL layers per graph
                auto T = [&](const char* nm, const std::function<void(int)>& f) { std::printf("    %-22s %7.2f us\n", nm, time_graph(q, NL, f, 9)); };
                T("orig gate mmvq", [&](int l) { k::native_mmvq(c.g, wg + (size_t) l * gs, xq, gate, N, FF, nc, &q); });
                T("orig up mmvq", [&](int l) { k::native_mmvq(c.u, wu + (size_t) l * us_, xq, up, N, FF, nc, &q); });
                T("orig swiglu+q8", [&](int) { k::native_swiglu_quantize_q8_1(gate, up, q8, FF, nc, &q); });
                T("orig down mmvq", [&](int l) { k::native_mmvq(c.d, wd + (size_t) l * ds, q8, out, FF, N, nc, &q); });
                T("bf16 gate row", [&](int l) { if (nc > 1) k::bf16_gemv_fp32_mmvf_multi(x, N, gi + (size_t) l * N, g, 1, N, 1, nc, &q);
                                                 else k::bf16_gemv_fp32_mmvf(x, gi + (size_t) l * N, g, N, 1, &q); });
                T("night gate+up+sw+q8", [&](int l) { k::night_shexp_gu_swiglu_q8(c.g, wg + (size_t) l * gs, c.u, wu + (size_t) l * us_, xq, gate, q8, N, FF, nc, &q,
                                                                                 x, gi + (size_t) l * N, g); });
                T("night down+scale", [&](int l) { k::night_shexp_down_scale(c.d, wd + (size_t) l * ds, q8, g, out, FF, N, nc, &q); });
            }
            double mabs = 0, rmax = 0; size_t neq = 0, bad = 0;
            for (size_t i = 0; i < r[0].size(); ++i) {
                rmax = std::max(rmax, (double) std::fabs(r[0][i]));
                const double e = std::fabs((double) r[1][i] - r[0][i]);
                if (!(e <= 1e30)) ++bad;
                mabs = std::max(mabs, e);
                if (std::memcmp(&r[0][i], &r[1][i], 4) != 0) ++neq;
            }
            std::printf("gate %2d up %2d down %2d (x%d layers) nc %d  orig %7.2f us  night %7.2f us  x%.2f  maxabs %.2e of %.2e  "
                        "bits differ %zu/%zu%s\n", c.g, c.u, c.d, c.layers, nc, t[0], t[1], t[0] / t[1], mabs, rmax, neq, r[0].size(),
                        bad ? " NONFINITE" : "");
            std::fflush(stdout);
        }
        sycl::free(wg, q); sycl::free(wu, q); sycl::free(wd, q);
    }
    return 0;
}
