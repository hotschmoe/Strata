// SYCL (night-lab STRATA_NIGHT_GDN_CHUNK=1): the prompt path's gated delta-net recurrence in chunkwise form.
//
// gdn_recurrence walks the T tokens one after another (S_t = g_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T,
// o_t = S_t^T q_t / sqrt(128)): ~1.45 us per token on the B70, 17.9-22.7 ms per 12288-token chunk and GDN layer.
// The standard chunkwise (WY) form replaces 64 sequential token steps by one step on 64-token blocks. Within a chunk,
// with G = cumsum(log decay) (all exponents below are <= 0, so nothing overflows):
//   A_ij  = b_i exp(G_i - G_j) k_i.k_j  (j < i),   (I + A) W = diag(b e^G) K,   (I + A) U = diag(b) V
//   QK_ij = exp(G_i - G_j) q_i.k_j      (j <= i)
// and across chunks (the only sequential part), per 32-column block of the 128x128 state:
//   Vn = U - W S,   O = (q e^G) S + QK Vn = (q e^G - QK W) S + QK U,   S' = e^{G_L} S + (k e^{G_L - G})^T Vn.
// K1 (every chunk and head in parallel) builds W, U, Qt = q e^G - QK W, Ou = QK U and G; K2 (head x column block,
// chunks in order) carries the state in registers; K3 is gdn_out_norm_kernel's norm.
// STATUS (night lab): correct (FP32 rounding level vs the scan) but 4.3x SLOWER on the B70 in this FP32 form - K2
// re-reads W/Qt/k rows from L1/L2 in every sub-group and K1 runs one 101 KB work-group per Xe core. Next: the K1
// products (k k^T, q k^T, QK W, QK U) on XMX and K2's rows staged once per work-group. Keep the flag off. FP32 throughout; the summation order differs from the scan,
// so the result agrees at FP32 rounding level, not bitwise.
#include "strata/prefill/kernels.hpp"
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::prefill {
class gdn_chunk_k1_night;
class gdn_chunk_k2_night;
class gdn_chunk_norm_night;
namespace {
constexpr int S = 128, HK = 16, HV = 48, C = 10240;
constexpr int CL = 64;          // tokens per chunk
constexpr int SEG = 8;          // chunks per K1/K2 launch pair (scratch)
constexpr int KP = S + 4;       // local row stride (floats)
constexpr int CB = 32;          // state columns per K2 work-group

inline float sigm_(float x) { return 1.0f / (1.0f + sycl::native::exp(-x)); }
inline uint16_t hf_(float f) {
    return sycl::bit_cast<unsigned short, sycl::half>(sycl::vec<float, 1>(f).convert<sycl::half, sycl::rounding_mode::rte>()[0]);
}

// K1: one work-group (256) per (segment chunk ci, value head)
void k1(const float* __restrict__ h, const float* __restrict__ gate, const float* __restrict__ beta, int64_t T,
        int64_t c0, float* __restrict__ W, float* __restrict__ U, float* __restrict__ Qt, float* __restrict__ Ou,
        float* __restrict__ Gs, float* smem) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
    const int head = (int) it.get_group(0), ci = (int) it.get_group(1), tid = (int) it.get_local_id(1);
    const int qh = head % HK;
    const int64_t t0 = (c0 + ci) * CL;
    const int L = (int) (T - t0 < CL ? T - t0 : CL);
    float* ks = smem;                       // [CL][KP]
    float* qs = ks + CL * KP;               // [CL][KP]
    float* sA = qs + CL * KP;               // [CL][CL + 1]
    float* sQ = sA + CL * (CL + 1);         // [CL][CL + 1]  q_i.k_j exp(G_i - G_j), j <= i
    float* sG = sQ + CL * (CL + 1);         // [CL]
    float* sB = sG + CL;                    // [CL]
    for (int e = tid; e < CL * S; e += 256) {
        const int i = e / S, d = e % S;
        const bool ok = i < L;
        ks[i * KP + d] = ok ? h[(t0 + i) * C + HK * S + qh * S + d] : 0.0f;
        qs[i * KP + d] = ok ? h[(t0 + i) * C + qh * S + d] : 0.0f;
    }
    if (tid < CL) sB[tid] = tid < L ? beta[(t0 + tid) * HV + head] : 0.0f;
    if (tid == 0) {
        float g = 0.0f;
        for (int i = 0; i < CL; ++i) { if (i < L) g += gate[(t0 + i) * HV + head]; sG[i] = g; }
    }
    it.barrier(sycl::access::fence_space::local_space);
    // Gram (k.k) and q.k on 4x4 blocks: block (bi, bj) = (tid / 16, tid % 16), lower blocks only
    {
        const int bi = tid / 16, bj = tid % 16;
        if (bi >= bj) {
            float kk[4][4], qk[4][4];
#pragma unroll
            for (int a = 0; a < 4; ++a)
#pragma unroll
                for (int b = 0; b < 4; ++b) { kk[a][b] = 0.0f; qk[a][b] = 0.0f; }
            for (int d = 0; d < S; ++d) {
                float ki[4], qi[4], kj[4];
#pragma unroll
                for (int a = 0; a < 4; ++a) { ki[a] = ks[(4 * bi + a) * KP + d]; qi[a] = qs[(4 * bi + a) * KP + d]; kj[a] = ks[(4 * bj + a) * KP + d]; }
#pragma unroll
                for (int a = 0; a < 4; ++a)
#pragma unroll
                    for (int b = 0; b < 4; ++b) { kk[a][b] = sycl::fma(ki[a], kj[b], kk[a][b]); qk[a][b] = sycl::fma(qi[a], kj[b], qk[a][b]); }
            }
#pragma unroll
            for (int a = 0; a < 4; ++a)
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const int i = 4 * bi + a, j = 4 * bj + b;
                    const float dec = sycl::exp(sG[i] - sG[j]);
                    sA[i * (CL + 1) + j] = j < i ? sB[i] * dec * kk[a][b] : 0.0f;
                    sQ[i * (CL + 1) + j] = (j <= i && i < L) ? dec * qk[a][b] : 0.0f;
                }
        }
    }
    it.barrier(sycl::access::fence_space::local_space);
    // forward substitution, one column per thread in registers: threads 0..127 W's columns, 128..255 U's
    {
        const int col = tid % S;
        const bool isU = tid >= S;
        float x[CL];
#pragma unroll
        for (int i = 0; i < CL; ++i) {
            float v = 0.0f;
            if (i < L) v = isU ? sB[i] * h[(t0 + i) * C + 2 * HK * S + head * S + col] : sB[i] * sycl::exp(sG[i]) * ks[i * KP + col];
            x[i] = v;
        }
#pragma unroll
        for (int i = 1; i < CL; ++i) {
            float acc = x[i];
#pragma unroll
            for (int j = 0; j < i; ++j) acc = sycl::fma(-sA[i * (CL + 1) + j], x[j], acc);
            x[i] = acc;
        }
        float* out = (isU ? U : W) + ((size_t) ci * HV + head) * CL * S + col;
#pragma unroll
        for (int i = 0; i < CL; ++i) out[(size_t) i * S] = x[i];
        // the state-independent parts of the output: O = (q e^G - QK W) S + QK U
        float* out2 = (isU ? Ou : Qt) + ((size_t) ci * HV + head) * CL * S + col;
#pragma unroll
        for (int i = 0; i < CL; ++i) {
            float acc = 0.0f;
#pragma unroll
            for (int jj = 0; jj <= i; ++jj) acc = sycl::fma(sQ[i * (CL + 1) + jj], x[jj], acc);
            out2[(size_t) i * S] = isU ? acc : (i < L ? qs[i * KP + col] * sycl::exp(sG[i]) - acc : 0.0f);
        }
    }
    if (tid < CL) Gs[((size_t) ci * HV + head) * CL + tid] = sG[tid];
}

// K2: one work-group (256 = 16 sub-groups of 16) per (value head, 32-column block); the segment's chunks in order.
// Lane l of sub-group s: column cb * 32 + 2 s + l / 8, state rows (l % 8) * 16 .. + 16 in registers.
void k2(float* __restrict__ state, const float* __restrict__ h, int64_t T, int64_t c0, int nc,
        const float* __restrict__ W, const float* __restrict__ U, const float* __restrict__ Qt,
        const float* __restrict__ Ou, const float* __restrict__ Gs, float* __restrict__ y) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
    auto sg = it.get_sub_group();
    const int head = (int) it.get_group(0), cb = (int) it.get_group(1);
    const int lane = (int) sg.get_local_id()[0], sgid = (int) sg.get_group_id()[0];
    const int rgp = lane % 8, j = cb * CB + 2 * sgid + lane / 8, qh = head % HK;
    const int r0 = rgp * 16;
    float sv[16];
#pragma unroll
    for (int q = 0; q < 16; ++q) sv[q] = state[((size_t) (r0 + q) * HV + head) * S + j];
    const float rs = sycl::rsqrt((float) S);
    for (int ci = 0; ci < nc; ++ci) {
        const int64_t t0 = (c0 + ci) * CL;
        const int L = (int) (T - t0 < CL ? T - t0 : CL);
        const size_t base = ((size_t) ci * HV + head) * CL * S;
        const float* Gc = Gs + ((size_t) ci * HV + head) * CL;
        float vn[CL];
        // Vn = U - W S and O = Qt S + Ou (rows r; the 8 lanes of a column hold 16 state rows each)
#pragma unroll
        for (int r = 0; r < CL; ++r) {
            float pw = 0.0f, pq = 0.0f;
            if (r < L) {
                const sycl::float4* w4 = reinterpret_cast<const sycl::float4*>(W + base + (size_t) r * S + r0);
                const sycl::float4* q4 = reinterpret_cast<const sycl::float4*>(Qt + base + (size_t) r * S + r0);
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const sycl::float4 w = w4[u], qq = q4[u];
                    pw = sycl::fma(w.x(), sv[4 * u], pw); pw = sycl::fma(w.y(), sv[4 * u + 1], pw);
                    pw = sycl::fma(w.z(), sv[4 * u + 2], pw); pw = sycl::fma(w.w(), sv[4 * u + 3], pw);
                    pq = sycl::fma(qq.x(), sv[4 * u], pq); pq = sycl::fma(qq.y(), sv[4 * u + 1], pq);
                    pq = sycl::fma(qq.z(), sv[4 * u + 2], pq); pq = sycl::fma(qq.w(), sv[4 * u + 3], pq);
                }
            }
            pw += sycl::permute_group_by_xor(sg, pw, 1);
            pq += sycl::permute_group_by_xor(sg, pq, 1);
            pw += sycl::permute_group_by_xor(sg, pw, 2);
            pq += sycl::permute_group_by_xor(sg, pq, 2);
            pw += sycl::permute_group_by_xor(sg, pw, 4);
            pq += sycl::permute_group_by_xor(sg, pq, 4);
            vn[r] = r < L ? U[base + (size_t) r * S + j] - pw : 0.0f;
            if (rgp == 0 && r < L) y[(t0 + r) * HV * S + (size_t) head * S + j] = (pq + Ou[base + (size_t) r * S + j]) * rs;
        }
        // S' = e^{G_L} S + (k e^{G_L - G})^T Vn
        const float gl = Gc[L - 1];
        const float egl = sycl::exp(gl);
#pragma unroll
        for (int q = 0; q < 16; ++q) sv[q] *= egl;
#pragma unroll
        for (int r = 0; r < CL; ++r) {
            if (r >= L) continue;
            const float c = sycl::exp(gl - Gc[r]) * vn[r];
            const sycl::float4* k4 = reinterpret_cast<const sycl::float4*>(h + (t0 + r) * C + HK * S + qh * S + r0);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const sycl::float4 kk = k4[u];
                sv[4 * u] = sycl::fma(kk.x(), c, sv[4 * u]); sv[4 * u + 1] = sycl::fma(kk.y(), c, sv[4 * u + 1]);
                sv[4 * u + 2] = sycl::fma(kk.z(), c, sv[4 * u + 2]); sv[4 * u + 3] = sycl::fma(kk.w(), c, sv[4 * u + 3]);
            }
        }
    }
#pragma unroll
    for (int q = 0; q < 16; ++q) state[((size_t) (r0 + q) * HV + head) * S + j] = sv[q];
}

struct Scratch { sycl::device dev; float* buf = nullptr; };
std::vector<Scratch>& scratches() { static std::vector<Scratch> v; return v; }
}  // namespace

bool gdn_night_chunk() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_NIGHT_GDN_CHUNK");
        const bool e = v != nullptr && v[0] == '1' && v[1] == '\0';
        if (e) std::fprintf(stderr, "strata night: STRATA_NIGHT_GDN_CHUNK=1 (prompt GDN recurrence in 64-token chunkwise form, FP32)\n");
        return e;
    }();
    return on;
}

bool gdn_recurrence_chunked(float* state, const float* h, const float* gate, const float* beta, const float* z,
                            const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream, int64_t ld16) {
    if (T < CL) return false;
    if (ld16 <= 0) ld16 = (int64_t) HV * S;
    sycl::queue* q = strata::q_of(stream);
    const size_t wsz = (size_t) SEG * HV * CL * S, gsz = (size_t) SEG * HV * CL;
    Scratch* sc = nullptr;
    for (auto& x : scratches())
        if (x.dev == q->get_device()) sc = &x;
    if (sc == nullptr) {
        scratches().push_back(Scratch{q->get_device()});
        sc = &scratches().back();
        sc->buf = sycl::malloc_device<float>(4 * wsz + gsz, *q);
        if (!sc->buf) { std::fprintf(stderr, "gdn_recurrence_chunked: no scratch\n"); return false; }
    }
    float* W = sc->buf;
    float* U = W + wsz;
    float* Qt = U + wsz;
    float* Ou = Qt + wsz;
    float* Gs = Ou + wsz;
    const int64_t nchunks = (T + CL - 1) / CL;
    constexpr size_t k1_floats = (size_t) 2 * CL * KP + 2 * CL * (CL + 1) + 2 * CL;
    for (int64_t c0 = 0; c0 < nchunks; c0 += SEG) {
        const int nc = (int) (nchunks - c0 < SEG ? nchunks - c0 : SEG);
        q->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> sm(sycl::range<1>(k1_floats), cgh);
            cgh.parallel_for<gdn_chunk_k1_night>(
                sycl::nd_range<2>(sycl::range<2>(HV, (size_t) nc * 256), sycl::range<2>(1, 256)),
                [=](sycl::nd_item<2>) [[sycl::reqd_sub_group_size(16)]] {
                    k1(h, gate, beta, T, c0, W, U, Qt, Ou, Gs, sm.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
        q->parallel_for<gdn_chunk_k2_night>(
            sycl::nd_range<2>(sycl::range<2>(HV, (size_t) (S / CB) * 256), sycl::range<2>(1, 256)),
            sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},
            [=](sycl::nd_item<2>) [[sycl::reqd_sub_group_size(16)]] { k2(state, h, T, c0, nc, W, U, Qt, Ou, Gs, y); });
    }
    // the output norm: gdn_out_norm_kernel's arithmetic (32-lane warp sums of 4 partials, the same order)
    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> wsum(sycl::range<1>(4), cgh);
        cgh.parallel_for<gdn_chunk_norm_night>(
            sycl::nd_range<2>(sycl::range<2>((size_t) T, (size_t) HV * S), sycl::range<2>(1, S)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
                auto sg = it.get_sub_group();
                const int64_t t = it.get_group(0);
                const int head = (int) it.get_group(1), col = (int) it.get_local_id(1);
                const size_t at = (size_t) t * HV * S + (size_t) head * S + col;
                const float oc = y[at];
                float sp = oc * oc;
                for (int o = 16; o > 0; o >>= 1) sp += sycl::permute_group_by_xor(sg, sp, o);
                if ((col & 31) == 0) wsum[col >> 5] = sp;
                it.barrier(sycl::access::fence_space::local_space);
                const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
                const float v = oc * sycl::rsqrt(ss / (float) S + eps) * gamma[col] * sigm_(z[t * HV * S + head * S + col]);
                y16[(size_t) t * ld16 + (size_t) head * S + col] = hf_(v);
            });
    });
    return true;
}

}  // namespace strata::prefill
