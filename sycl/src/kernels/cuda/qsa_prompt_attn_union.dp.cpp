// SYCL (night-lab STRATA_NIGHT_PATTN_XMX): prompt QSA attention over 8-query tiles on XMX. See
// include/strata/kernels/qsa_prompt_attn_union.hpp.
#include "strata/kernels/qsa_prompt_attn_union.hpp"
#include "strata/core/geometry.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::kernels {
class pattn_union_build_night;    // kernel names: namespace scope
class pattn_union_attn_night;
namespace {
namespace jm = sycl::ext::oneapi::experimental::matrix;
namespace jmi = sycl::ext::intel::experimental::matrix;
constexpr int HD = 256, G = 12, TQ = 8, ROWS = TQ * G;        // 96 rows: tile query j, head h -> row j * 12 + h
constexpr int SG = 16, NSG = ROWS / 8, THREADS = SG * NSG;   // 12 sub-groups, 8 rows each
constexpr int CH = 32;                                       // union cells per chunk
constexpr int QS = HD, SS = CH + 1, PS = CH + 8;             // local strides (halves / floats / halves)
constexpr int RANGE = 16384;                                 // union build: cells per local bitmap pass
constexpr int BUILD_THREADS = 256;
constexpr int64_t TILE_BATCH = 256;                          // tiles per (build, attention) launch pair
static_assert(HD == strata::core::kCompiledGeometry.head_dim &&
              G == strata::core::kCompiledGeometry.n_head / strata::core::kCompiledGeometry.n_head_kv,
              "qsa_prompt_attn_union is compiled for core/geometry.hpp");

struct Lay {
    static constexpr size_t q = (size_t) ROWS * QS * 2;                // Q, FP16 (A tiles loaded per chunk)
    static constexpr size_t s = (size_t) NSG * 8 * SS * 4;             // per sub-group S, then P in place
    static constexpr size_t al = (size_t) NSG * 8 * 4 * 2;
    static constexpr size_t rows = (size_t) 2 * CH * 8;
    static constexpr size_t mask = (size_t) 2 * CH * 4;
    static constexpr size_t bytes = q + s + al + rows + mask + 16 * 6;
};
static_assert(8 * PS * 2 <= 8 * SS * 4, "P fits in place of S");

template <typename T>
inline auto lptr(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}

// pass 1: one work-group per tile; the union of the tile queries' selections, ascending, as cell | mask << 24
void union_build(const int32_t* __restrict__ ids, const int32_t* __restrict__ steps, int64_t cap, int64_t q0,
                 int64_t n_q, int32_t* __restrict__ ulist, int32_t* __restrict__ ucount, uint32_t* smask) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    auto grp = it.get_group();
    const int tile = (int) it.get_group(0), t = (int) it.get_local_id(0);
    const int64_t t0 = q0 + (int64_t) tile * TQ;
    const int nq = (int) (n_q - t0 < TQ ? n_q - t0 : TQ);
    int64_t nkv = 0;
    for (int j = 0; j < nq; ++j) { const int64_t v = steps[(t0 + j) * kStepCount + kStepNKv]; nkv = v > nkv ? v : nkv; }
    int32_t* out = ulist + (size_t) tile * TQ * cap;
    int total = 0;
    for (int64_t r0 = 0; r0 < nkv; r0 += RANGE) {
        for (int w = t; w < RANGE / 4; w += BUILD_THREADS) smask[w] = 0u;
        it.barrier(sycl::access::fence_space::local_space);
        for (int64_t k = t; k < (int64_t) TQ * cap; k += BUILD_THREADS) {
            const int j = (int) (k / cap);
            const int64_t i = k - (int64_t) j * cap;
            if (j >= nq || i >= steps[(t0 + j) * kStepCount + kStepWidth]) continue;
            const int64_t id = ids[(t0 + j) * cap + i] - r0;
            if (id < 0 || id >= RANGE) continue;
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                             sycl::access::address_space::local_space>
                a(smask[id >> 2]);
            a.fetch_or((1u << j) << (8 * (id & 3)));
        }
        it.barrier(sycl::access::fence_space::local_space);
        constexpr int WPT = RANGE / 4 / BUILD_THREADS;   // words per thread (16 = 64 cells), contiguous
        int cnt = 0;
        for (int w = 0; w < WPT; ++w) {
            const uint32_t m = smask[t * WPT + w];
            cnt += (m & 0xFFu) != 0;
            cnt += (m & 0xFF00u) != 0;
            cnt += (m & 0xFF0000u) != 0;
            cnt += (m & 0xFF000000u) != 0;
        }
        const int base = sycl::exclusive_scan_over_group(grp, cnt, sycl::plus<int>());
        const int sum = sycl::reduce_over_group(grp, cnt, sycl::plus<int>());
        int o = total + base;
        for (int w = 0; w < WPT; ++w) {
            const uint32_t m = smask[t * WPT + w];
            if (m == 0u) continue;
            for (int b = 0; b < 4; ++b) {
                const uint32_t mb = (m >> (8 * b)) & 0xFFu;
                if (mb) out[o++] = (int32_t) ((uint32_t) (r0 + (t * WPT + w) * 4 + b) | (mb << 24));
            }
        }
        total += sum;
        it.barrier(sycl::access::fence_space::local_space);
    }
    if (t == 0) ucount[tile] = total;
}

using TA = jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 8, 16, jm::layout::row_major>;
using TB = jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::ext_intel_packed>;
using TC = jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16>;

// pass 2: one work-group per (tile, KV head)
template <typename T>
inline auto gptr(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p);
}
void union_attn(const float* __restrict__ q, QsaAttnPools p, const int32_t* __restrict__ ulist,
                const int32_t* __restrict__ ucount, int64_t cap, int64_t q0, int64_t n_q, int n_kv_heads,
                int page_size, float scale_log2, float* __restrict__ attn, uint8_t* smem, sycl::half* gstage) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg = it.get_sub_group();
    uint8_t* at = smem;
    auto take = [&](size_t b) { uint8_t* r = at; at += (b + 15) & ~(size_t) 15; return r; };
    sycl::half* qs = (sycl::half*) take(Lay::q);
    float* s_all = (float*) take(Lay::s);
    float* al_all = (float*) take(Lay::al);
    long long* srow2 = (long long*) take(Lay::rows);   // [2][CH]
    uint32_t* smk2 = (uint32_t*) take(Lay::mask);      // [2][CH]
    // the chunk's packed K^T and V in this work-group's global staging slot (L1/L2-resident): a B tile is one 2D block
    // load
    sycl::half* kp = gstage + ((size_t) it.get_group(2) * n_kv_heads + it.get_group(1)) * (2 * CH * HD);
    sycl::half* vp = kp + CH * HD;

    const int tile = (int) it.get_group(2), kvh = (int) it.get_group(1);
    const int t = (int) it.get_local_id(2), lane = (int) sg.get_local_id()[0], sgid = (int) sg.get_group_id()[0];
    const int64_t t0 = q0 + (int64_t) tile * TQ;
    const int n_head = n_kv_heads * G;
    const int32_t* list = ulist + (size_t) tile * TQ * cap;
    const int U = ucount[tile];
    float* sS = s_all + (size_t) sgid * 8 * SS;
    sycl::half* sP = (sycl::half*) sS;          // P overwrites S in place (after a sub-group barrier)
    float* sA = al_all + (size_t) sgid * 16;   // [8] alpha, [8] l

    auto entry = [&](int c, long long& r, uint32_t& mk) {
        r = -1; mk = 0u;
        if (c < U) {
            const uint32_t e = (uint32_t) list[c];
            const int cell = (int) (e & 0xFFFFFFu);
            const long long page = (long long) p.page_table[cell / page_size];
            if (page >= 0) { r = (page * n_kv_heads + kvh) * page_size + (cell % page_size); mk = e >> 24; }
        }
    };
    // Q of the tile's 8 queries x 12 heads, FP16
    for (int i = t; i < ROWS * (HD / 4); i += THREADS) {
        const int r = i / (HD / 4), d4 = (i % (HD / 4)) * 4;
        const int j = r / G, h = r % G;
        sycl::float4 v(0.0f);
        if (t0 + j < n_q) v = *reinterpret_cast<const sycl::float4*>(q + (size_t) (t0 + j) * n_head * HD + (size_t) (kvh * G + h) * HD + d4);
        sycl::half* dst = qs + (size_t) r * QS + d4;
        dst[0] = sycl::half(v.x()); dst[1] = sycl::half(v.y()); dst[2] = sycl::half(v.z()); dst[3] = sycl::half(v.w());
    }
    TC O[HD / 16];
#pragma unroll
    for (int d = 0; d < HD / 16; ++d) jm::joint_matrix_fill(sg, O[d], 0.0f);
    float m_run = -INFINITY, l_run = 0.0f;   // row lane >> 1 (two lanes per row)
    const int myrow = lane >> 1, half_ = lane & 1;
    const int qrow_j = (sgid * 8 + myrow) / G;   // the tile query of this lane's row
    if (t < CH) { long long r; uint32_t mk; entry(t, r, mk); srow2[t] = r; smk2[t] = mk; }
    it.barrier(sycl::access::fence_space::local_space);
    const sycl::half* qa = qs + (size_t) sgid * 8 * QS;
    // the first QR k-steps of this sub-group's Q A tiles in registers (the rest are loaded per chunk)
    constexpr int QR = 8;
    TA Qr[QR];
#pragma unroll
    for (int kk = 0; kk < QR; ++kk) jm::joint_matrix_load(sg, Qr[kk], lptr(qa + kk * 16), (size_t) QS);

    for (int c0 = 0, buf = 0; c0 < U; c0 += CH, buf ^= 1) {
        const long long* srow = srow2 + buf * CH;
        const uint32_t* smk = smk2 + buf * CH;
        // K^T packed (dim pairs interleaved per cell): word (k/2) * CH + n holds dims k, k+1 of cell n; 16 dims per item
        for (int i = t; i < CH * (HD / 16); i += THREADS) {
            const int n = i % CH, k0 = (i / CH) * 16;
            const long long r = srow[n];
            uint32_t* dst = reinterpret_cast<uint32_t*>(kp) + (size_t) (k0 / 2) * CH + n;
            if (r >= 0) {
                const sycl::vec<int8_t, 16> c = *reinterpret_cast<const sycl::vec<int8_t, 16>*>(p.k_q + r * HD + k0);
                const float sc = (float) sycl::bit_cast<sycl::half>(p.k_scale[r * (HD / KV_Q8_GROUP) + k0 / KV_Q8_GROUP]);
#pragma unroll
                for (int u = 0; u < 8; ++u)
                    dst[(size_t) u * CH] = sycl::bit_cast<uint32_t>(sycl::half2((float) c[2 * u] * sc, (float) c[2 * u + 1] * sc));
            } else {
#pragma unroll
                for (int u = 0; u < 8; ++u) dst[(size_t) u * CH] = 0u;
            }
        }
        // V packed (cell pairs interleaved per dim): word (c/2) * HD + d holds cells c, c+1 of dim d; 16 dims per item
        for (int i = t; i < (CH / 2) * (HD / 16); i += THREADS) {
            const int cp = i / (HD / 16), d0 = (i % (HD / 16)) * 16;
            float v0[16], v1[16];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const long long r = srow[2 * cp + e];
                float* v = e ? v1 : v0;
                if (r >= 0) {
                    const sycl::vec<int8_t, 16> c = *reinterpret_cast<const sycl::vec<int8_t, 16>*>(p.v_q + r * HD + d0);
                    const float sc = (float) sycl::bit_cast<sycl::half>(p.v_scale[r * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP]);
#pragma unroll
                    for (int u = 0; u < 16; ++u) v[u] = (float) c[u] * sc;
                } else {
#pragma unroll
                    for (int u = 0; u < 16; ++u) v[u] = 0.0f;
                }
            }
            sycl::vec<uint32_t, 8> o0, o1;
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                o0[u] = sycl::bit_cast<uint32_t>(sycl::half2(v0[u], v1[u]));
                o1[u] = sycl::bit_cast<uint32_t>(sycl::half2(v0[u + 8], v1[u + 8]));
            }
            uint32_t* dst = reinterpret_cast<uint32_t*>(vp) + (size_t) cp * HD + d0;
            *reinterpret_cast<sycl::vec<uint32_t, 8>*>(dst) = o0;
            *reinterpret_cast<sycl::vec<uint32_t, 8>*>(dst + 8) = o1;
        }
        it.barrier(sycl::access::fence_space::global_and_local);
        // the next chunk's rows: loads issued now, stored after this chunk's products (their latency overlaps them)
        long long nr = -1;
        uint32_t nmk = 0u;
        if (t < CH) entry(c0 + CH + t, nr, nmk);
        // S = Q K^T for this sub-group's 8 rows x 32 cells; the next k-step's B tiles in flight during this one's
        {
            TC S0, S1;
            jm::joint_matrix_fill(sg, S0, 0.0f);
            jm::joint_matrix_fill(sg, S1, 0.0f);
            TB Bc0, Bc1, Bn0, Bn1;
            jm::joint_matrix_load(sg, Bc0, gptr(kp), (size_t) (CH * 2));
            jm::joint_matrix_load(sg, Bc1, gptr(kp + 16 * 2), (size_t) (CH * 2));
#pragma unroll
            for (int kk = 0; kk < HD / 16; kk += 2) {
                jm::joint_matrix_load(sg, Bn0, gptr(kp + (size_t) ((kk + 1) * 8) * (CH * 2)), (size_t) (CH * 2));
                jm::joint_matrix_load(sg, Bn1, gptr(kp + (size_t) ((kk + 1) * 8) * (CH * 2) + 16 * 2), (size_t) (CH * 2));
                if (kk < QR) {
                    jm::joint_matrix_mad(sg, S0, Qr[kk < QR ? kk : 0], Bc0, S0);
                    jm::joint_matrix_mad(sg, S1, Qr[kk < QR ? kk : 0], Bc1, S1);
                } else {
                    TA A;
                    jm::joint_matrix_load(sg, A, lptr(qa + kk * 16), (size_t) QS);
                    jm::joint_matrix_mad(sg, S0, A, Bc0, S0);
                    jm::joint_matrix_mad(sg, S1, A, Bc1, S1);
                }
                if (kk + 2 < HD / 16) {
                    jm::joint_matrix_load(sg, Bc0, gptr(kp + (size_t) ((kk + 2) * 8) * (CH * 2)), (size_t) (CH * 2));
                    jm::joint_matrix_load(sg, Bc1, gptr(kp + (size_t) ((kk + 2) * 8) * (CH * 2) + 16 * 2), (size_t) (CH * 2));
                }
                if (kk + 1 < QR) {
                    jm::joint_matrix_mad(sg, S0, Qr[kk + 1 < QR ? kk + 1 : 0], Bn0, S0);
                    jm::joint_matrix_mad(sg, S1, Qr[kk + 1 < QR ? kk + 1 : 0], Bn1, S1);
                } else {
                    TA A2;
                    jm::joint_matrix_load(sg, A2, lptr(qa + (kk + 1) * 16), (size_t) QS);
                    jm::joint_matrix_mad(sg, S0, A2, Bn0, S0);
                    jm::joint_matrix_mad(sg, S1, A2, Bn1, S1);
                }
            }
            jm::joint_matrix_store(sg, S0, lptr(sS), (size_t) SS, jm::layout::row_major);
            jm::joint_matrix_store(sg, S1, lptr(sS + 16), (size_t) SS, jm::layout::row_major);
        }
        sycl::group_barrier(sg);
        // online softmax: lanes 2i, 2i+1 own row i, 16 cells each; a cell the row's query did not select is -inf
        float x[16];
        float mx = -INFINITY;
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            const int c = half_ * 16 + u;
            const bool ok = (smk[c] >> qrow_j) & 1u;
            x[u] = ok ? sS[myrow * SS + c] * scale_log2 : -INFINITY;
            mx = sycl::fmax(mx, x[u]);
        }
        mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, 1));
        const float m_new = sycl::fmax(m_run, mx);
        float alpha = 1.0f, sum = 0.0f;
        sycl::half ph[16];
        if (m_new != -INFINITY) {
            alpha = m_run == -INFINITY ? 0.0f : sycl::exp2(m_run - m_new);
#pragma unroll
            for (int u = 0; u < 16; ++u) {
                const float e = x[u] == -INFINITY ? 0.0f : sycl::exp2(x[u] - m_new);
                ph[u] = sycl::half(e);
                sum += e;
            }
        } else {
#pragma unroll
            for (int u = 0; u < 16; ++u) ph[u] = sycl::half(0.0f);
        }
        sum += sycl::permute_group_by_xor(sg, sum, 1);
        l_run = l_run * alpha + sum;
        m_run = m_new;
        const bool rescale = sycl::any_of_group(sg, alpha != 1.0f);
        sycl::group_barrier(sg);   // every lane has read its S before P overwrites it
#pragma unroll
        for (int u = 0; u < 16; ++u) sP[myrow * PS + half_ * 16 + u] = ph[u];
        if (half_ == 0) sA[myrow] = alpha;
        sycl::group_barrier(sg);
        if (rescale) {
#pragma unroll
            for (int d = 0; d < HD / 16; ++d)
                jmi::joint_matrix_apply(sg, O[d], [=](float& v, size_t row, size_t) { v *= sA[row]; });
        }
        // O += P V; the next d-tile's V tiles in flight during this one's
        {
            TA P0, P1;
            jm::joint_matrix_load(sg, P0, lptr(sP), (size_t) PS);
            jm::joint_matrix_load(sg, P1, lptr(sP + 16), (size_t) PS);
            TB Vc0, Vc1, Vn0, Vn1;
            jm::joint_matrix_load(sg, Vc0, gptr(vp), (size_t) (HD * 2));
            jm::joint_matrix_load(sg, Vc1, gptr(vp + (size_t) 8 * (HD * 2)), (size_t) (HD * 2));
#pragma unroll
            for (int d = 0; d < HD / 16; d += 2) {
                jm::joint_matrix_load(sg, Vn0, gptr(vp + (size_t) (d + 1) * 16 * 2), (size_t) (HD * 2));
                jm::joint_matrix_load(sg, Vn1, gptr(vp + (size_t) 8 * (HD * 2) + (d + 1) * 16 * 2), (size_t) (HD * 2));
                jm::joint_matrix_mad(sg, O[d], P0, Vc0, O[d]);
                jm::joint_matrix_mad(sg, O[d], P1, Vc1, O[d]);
                if (d + 2 < HD / 16) {
                    jm::joint_matrix_load(sg, Vc0, gptr(vp + (size_t) (d + 2) * 16 * 2), (size_t) (HD * 2));
                    jm::joint_matrix_load(sg, Vc1, gptr(vp + (size_t) 8 * (HD * 2) + (d + 2) * 16 * 2), (size_t) (HD * 2));
                }
                jm::joint_matrix_mad(sg, O[d + 1], P0, Vn0, O[d + 1]);
                jm::joint_matrix_mad(sg, O[d + 1], P1, Vn1, O[d + 1]);
            }
        }
        if (t < CH) { srow2[(buf ^ 1) * CH + t] = nr; smk2[(buf ^ 1) * CH + t] = nmk; }
        it.barrier(sycl::access::fence_space::global_and_local);   // K, V of this chunk consumed; the next rows visible
    }
    // out = O / l
    if (half_ == 0) sA[8 + myrow] = l_run;
    sycl::group_barrier(sg);
    const int rbase = sgid * 8;
#pragma unroll
    for (int d = 0; d < HD / 16; ++d)
        jmi::joint_matrix_apply(sg, O[d], [=](float& v, size_t row, size_t col) {
            const int rr = rbase + (int) row, j = rr / G, h = rr % G;
            if (t0 + j < n_q) {
                const float l = sA[8 + row];
                attn[(size_t) (t0 + j) * n_head * HD + (size_t) (kvh * G + h) * HD + d * 16 + col] = l > 0.0f ? v / l : 0.0f;
            }
        });
}

struct Scratch {
    sycl::device dev;
    int32_t* ulist = nullptr;
    int32_t* ucount = nullptr;
    sycl::half* gstage = nullptr;
    int64_t cap = 0;
};
std::vector<Scratch>& scratches() { static std::vector<Scratch> v; return v; }
}  // namespace

bool qsa_night_pattn_xmx() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_NIGHT_PATTN_XMX");
        const bool e = v != nullptr && v[0] == '1' && v[1] == '\0';
        if (e) std::fprintf(stderr, "strata night: STRATA_NIGHT_PATTN_XMX=1 (prompt QSA attention, 8-query union tiles on XMX, FP16 products)\n");
        return e;
    }();
    return on;
}

bool qsa_prompt_attn_union(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return true;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || cap > (1 << 20) || !ids || !steps ||
        !pools.page_table || !pools.k_q || !pools.v_q || !pools.k_scale || !pools.v_scale || pools.k_q4 || pools.v_q4)
        return false;
    sycl::queue* st = strata::q_of(stream);
    if (st->get_device().get_info<sycl::info::device::local_mem_size>() < Lay::bytes) return false;
    Scratch* sc = nullptr;
    for (auto& x : scratches())
        if (x.dev == st->get_device()) sc = &x;
    if (sc == nullptr || sc->cap < cap) {
        if (sc == nullptr) { scratches().push_back(Scratch{st->get_device()}); sc = &scratches().back(); }
        st->wait();
        if (sc->ulist) sycl::free(sc->ulist, *st);
        if (sc->ucount) sycl::free(sc->ucount, *st);
        sc->ulist = sycl::malloc_device<int32_t>((size_t) TILE_BATCH * TQ * cap, *st);
        sc->ucount = sycl::malloc_device<int32_t>((size_t) TILE_BATCH, *st);
        if (!sc->gstage) sc->gstage = sycl::malloc_device<sycl::half>((size_t) TILE_BATCH * 2 * 2 * CH * HD, *st);
        sc->cap = cap;
        if (!sc->ulist || !sc->ucount || !sc->gstage) { std::fprintf(stderr, "qsa_prompt_attn_union: no scratch\n"); return false; }
    }
    int32_t* ulist = sc->ulist;
    int32_t* ucount = sc->ucount;
    sycl::half* gstage = sc->gstage;
    const float scale_log2 = 1.4426950408889634f / std::sqrt((float) HD);
    const int nkv = (int) s.n_head_kv, ps = (int) s.page_size;
    const int64_t n_tiles = (n_q + TQ - 1) / TQ;
    for (int64_t b0 = 0; b0 < n_tiles; b0 += TILE_BATCH) {
        const int64_t nt = sycl::min<int64_t>(TILE_BATCH, n_tiles - b0);
        const int64_t q0 = b0 * TQ;
        st->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint32_t, 1> sm(sycl::range<1>(RANGE / 4), cgh);
            cgh.parallel_for<pattn_union_build_night>(
                sycl::nd_range<1>(sycl::range<1>((size_t) nt * BUILD_THREADS), sycl::range<1>(BUILD_THREADS)),
                [=](sycl::nd_item<1>) {
                    union_build(ids, steps, cap, q0, n_q, ulist, ucount, sm.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
        st->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(Lay::bytes), cgh);
            cgh.parallel_for<pattn_union_attn_night>(
                sycl::nd_range<3>(sycl::range<3>(1, (size_t) nkv, (size_t) nt * THREADS), sycl::range<3>(1, 1, THREADS)),
                sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},
                [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(SG)]] {
                    union_attn(q, pools, ulist, ucount, cap, q0, n_q, nkv, ps, scale_log2, attn,
                               slm.get_multi_ptr<sycl::access::decorated::no>().get(), gstage);
                });
        });
    }
    return true;
}

}  // namespace strata::kernels
