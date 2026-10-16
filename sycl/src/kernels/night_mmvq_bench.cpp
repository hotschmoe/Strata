// sycl/src/kernels/night_mmvq_bench.cpp - STRATA_NIGHT_MMVQ A/B through the public native_mmvq / native_mmvq_head API:
// the original dispatch (night off) against the night KREG kernels (night on) at the decode shapes and window widths,
// random valid weights, native_quantize_q8_1 activations, NMAT distinct weight copies per graph (>= ~600 MB streamed
// per replay, so L2 never holds a matrix), in-order queue graph replay. Prints us/call, GB/s of weights, the speedup
// and the max abs / max rel error of night against the original on matrix 0.
//     night_mmvq_bench [types=14,13,12] [ncols=1,2,3,4,6] [shape=all|name]
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_head_mmvq.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace strata::kernels { void native_mmvq_set_night(int on); }
namespace k = strata::kernels;
namespace sx = sycl::ext::oneapi::experimental;

namespace {
uint64_t rs = 0x9E3779B97F4A7C15ull;
inline uint64_t xr() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
int block_bytes(int t) { return t == 12 ? 144 : t == 13 ? 176 : t == 14 ? 210 : t == 23 ? 136 : t == 20 ? 18 : t == 21 ? 110 : 34; }
const char* fname(int t) { return t == 12 ? "Q4_K" : t == 13 ? "Q5_K" : t == 14 ? "Q6_K" : t == 23 ? "IQ4_XS" : t == 20 ? "IQ4_NL" : t == 21 ? "IQ3_S" : "Q8_0"; }
void fill_pool(int t, std::vector<uint8_t>& h) {   // random bytes, finite small fp16 scales (~2^-10) at the format's offsets
    uint64_t* p = (uint64_t*) h.data();
    for (size_t i = 0; i < h.size() / 8; ++i) p[i] = xr();
    const size_t bs = block_bytes(t), nb = h.size() / bs;
    for (size_t b = 0; b < nb; ++b) {
        uint8_t* o = &h[b * bs];
        const uint16_t d = (uint16_t) (0x1400 | (xr() & 0x3ff)), m = (uint16_t) (0x1400 | (xr() & 0x3ff));
        std::memcpy(o + (t == 14 ? 208 : 0), &d, 2);
        if (t == 12 || t == 13) std::memcpy(o + 2, &m, 2);
    }
}
std::vector<int> parse_list(const char* s) {
    std::vector<int> v; std::string t(s); size_t p = 0;
    while (p < t.size()) { size_t e = t.find(',', p); if (e == std::string::npos) e = t.size(); v.push_back(std::atoi(t.substr(p, e - p).c_str())); p = e + 1; }
    return v;
}
double time_graph(sycl::queue& q, int nmat, const std::function<void(int)>& launch, int reps) {
    sx::command_graph g(q.get_context(), q.get_device());
    g.begin_recording(q);
    for (int m = 0; m < nmat; ++m) launch(m);
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
                            e0.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1e3 / nmat);
    }
    std::sort(us.begin(), us.end());
    return us[us.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
    const std::vector<int> types = parse_list(argc > 1 ? argv[1] : "14,13,12");
    const std::vector<int> ncl = parse_list(argc > 2 ? argv[2] : "1,2,3,4,6");
    const std::string ssel = argc > 3 ? argv[3] : "all";
    const size_t POOL = (size_t) 640 << 20;
    sycl::queue q(sycl::gpu_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    struct Shape { int n_in, n_out; const char* name; };
    const Shape shapes[] = {{2560, 10240, "qkv"}, {2560, 6144, "gate"}, {6144, 2560, "out"}, {2560, 12288, "attn_q"},
                            {2560, 512, "kv"}, {2560, 640, "sh_gu"}, {2560, 248320, "head"}, {640, 2560, "sh_down"}};
    uint8_t* dpool = (uint8_t*) sycl::malloc_device(POOL + 4096, q);
    std::vector<uint8_t> hpool(POOL);
    const int maxc = 16, maxout = 248320;
    float* y = sycl::malloc_device<float>((size_t) maxc * maxout, q);
    float* y2 = sycl::malloc_device<float>((size_t) maxc * maxout, q);
    std::vector<float> hx((size_t) maxc * 6144);
    for (auto& v : hx) {
        const double u1 = (xr() >> 11) * (1.0 / 9007199254740992.0) + 1e-12, u2 = (xr() >> 11) * (1.0 / 9007199254740992.0);
        v = (float) (std::sqrt(-2 * std::log(u1)) * std::cos(6.283185307179586 * u2));
    }
    float* dx = sycl::malloc_device<float>(hx.size(), q);
    q.memcpy(dx, hx.data(), hx.size() * 4).wait();
    void* xq = sycl::malloc_device((size_t) maxc * (6144 / 32) * 36 + 256, q);
    std::printf("night_mmvq_bench: original dispatch vs STRATA_NIGHT_MMVQ (KREG); us/call median, GB/s of weights\n");
    for (const int t : types) {
        fill_pool(t, hpool);
        q.memcpy(dpool, hpool.data(), POOL).wait();
        for (const Shape& sh : shapes) {
            if (ssel != "all" && ssel != sh.name) continue;
            const bool head = std::string(sh.name) == "head";
            if (head && t != 13) continue;
            if ((t == 20) != (std::string(sh.name) == "sh_down")) continue;   // IQ4_NL: the shared-expert down only
            const size_t wb = k::native_mmvq_weight_bytes(t, sh.n_in, sh.n_out);
            const size_t slot = (wb + 255) & ~(size_t) 255;
            const int ndist = (int) std::min<size_t>(128, POOL / slot), nmat = std::max(ndist, 16);
            for (const int nc : ncl) {
                for (int c0 = 0; c0 < nc; c0 += 8)   // the quantizer takes <= 8 columns per call
                    k::native_quantize_q8_1(dx + (size_t) c0 * sh.n_in, (uint8_t*) xq + (size_t) c0 * (sh.n_in / 32) * 36, sh.n_in,
                                            std::min(8, nc - c0), &q);
                q.wait();
                // level 0 = original, 1 = STRATA_NIGHT_MMVQ, 2 = + STRATA_NIGHT_MMVQ2. Above 8 columns levels 0/1 are two
                // calls of 8 + (nc - 8) columns (what a caller has to do without level 2).
                auto call = [&](const uint8_t* w, float* out, int level) {
                    auto one = [&](const uint8_t* xx, float* oo, int c) {
                        if (head) k::native_mmvq_head(t, w, xx, oo, sh.n_in, sh.n_out, c, &q);
                        else k::native_mmvq(t, w, xx, oo, sh.n_in, sh.n_out, c, &q);
                    };
                    if (nc <= 8 || level == 2) one((const uint8_t*) xq, out, nc);
                    else { one((const uint8_t*) xq, out, 8); one((const uint8_t*) xq + (size_t) 8 * (sh.n_in / 32) * 36, out + (size_t) 8 * sh.n_out, nc - 8); }
                };
                const size_t ny = (size_t) nc * sh.n_out;
                std::vector<float> res[3];
                double us[3];
                for (int lv = 0; lv < 3; ++lv) {
                    k::native_mmvq_set_night(lv);
                    res[lv].resize(ny);
                    q.memset(y2, 0, ny * 4).wait();
                    call(dpool, y2, lv); q.wait();
                    q.memcpy(res[lv].data(), y2, ny * 4).wait();
                    us[lv] = time_graph(q, nmat, [&](int m) { call(dpool + (size_t) (m % ndist) * slot, y, lv); }, wb > (100u << 20) ? 7 : 11);
                }
                k::native_mmvq_set_night(-1);
                const std::vector<float>& ref = res[0];
                double rmax = 0; size_t bad = 0;
                for (size_t i = 0; i < ny; ++i) rmax = std::max(rmax, (double) std::fabs(ref[i]));
                double mabs[3] = {0, 0, 0}, mrel[3] = {0, 0, 0};
                for (int lv = 1; lv < 3; ++lv)
                    for (size_t i = 0; i < ny; ++i) {
                        const double e = std::fabs((double) res[lv][i] - ref[i]);
                        if (!(e <= 1e30)) ++bad;
                        mabs[lv] = std::max(mabs[lv], e);
                        if (std::fabs(ref[i]) > 1e-3 * rmax) mrel[lv] = std::max(mrel[lv], e / std::fabs(ref[i]));
                    }
                std::printf("%-6s %-6s %5d x %6d nc %2d  orig %8.2f  lv1 %8.2f  lv2 %8.2f us (%4.1f%% of 608)  lv2/lv1 x%.2f  orig/lv2 x%.2f  "
                            "err lv1 %.1e/%.1e lv2 %.1e/%.1e (abs/rel, max|y| %.1e)%s\n",
                            fname(t), sh.name, sh.n_in, sh.n_out, nc, us[0], us[1], us[2], wb / us[2] / 1e3 / 6.08,
                            us[1] / us[2], us[0] / us[2], mabs[1], mrel[1], mabs[2], mrel[2], rmax, bad ? " NONFINITE" : "");
                std::fflush(stdout);
            }
        }
    }
    sycl::free(dpool, q); sycl::free(y, q); sycl::free(y2, q); sycl::free(dx, q); sycl::free(xq, q);
    return 0;
}
