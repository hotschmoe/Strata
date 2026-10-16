// Night lab 2026-10-16: grouped prefill expert GEMM straight from the native blocks (night_grouped.hpp).
// Included at the end of iq_kernels.dp.cpp (inside namespace strata::kernels) for the dq_* decoders.
// (night_grouped.hpp is included at the top of iq_kernels.dp.cpp)

namespace nightg {
namespace jm = sycl::ext::oneapi::experimental::matrix;
namespace jmi = sycl::ext::intel::experimental::matrix;
using half = sycl::half;

// the 8-value run decoders (values [8 tid, 8 tid + 8) of superblock ibs, written to y + 8 tid)
template <int TY>
inline void dq_run(const void* base, int64_t ibs, half* y, int tid) {
    if constexpr (TY == 16) dq_iq2_xxs<half>(base, ibs, y, tid);
    else if constexpr (TY == 17) dq_iq2_xs<half>(base, ibs, y, tid);
    else if constexpr (TY == 18) dq_iq3_xxs<half>(base, ibs, y, tid);
    else if constexpr (TY == 20) dq_iq4_nl<half>(base, ibs, y, tid);
    else if constexpr (TY == 21) dq_iq3_s<half>(base, ibs, y, tid);
    else if constexpr (TY == 22) dq_iq2_s<half>(base, ibs, y, tid);
    else if constexpr (TY == 42) dq_q2_0<half>(base, ibs, y, tid);
}

// Bit-exact copies of dq_iq2_xxs / dq_iq3_xxs / dq_iq3_s with the sign mask as a shift (kmask_iq2xs[j] = 1 << j) and
// the IQ2 sign byte computed (ksigns_iq2xs[i] = i | parity(i) << 7) instead of two device-global table loads per value.
inline uint32_t ksign7(uint32_t s7) { return s7 | ((sycl::popcount(s7) & 1u) << 7); }
template <int TY>
inline void dq_fast(const void* vx, int64_t ibs, half* yy, int tid) {
    const int il = tid % 4, ib = tid / 4;
    half* y = yy + 32 * ib + 8 * il;
    float v[8];
    if constexpr (TY == 21) {
        const block_iq3_s* x = (const block_iq3_s*) vx + ibs;
        const uint8_t* qs = x->qs + 8 * ib;
        const uint32_t qh = x->qh[ib];
        const uint8_t* grid1 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 256)));
        const uint8_t* grid2 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 256)));
        const float d = (float) x->d * (1 + 2 * ((x->scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
        const uint32_t signs = x->signs[4 * ib + il];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[j + 0] = d * grid1[j] * ((signs >> (j + 0)) & 1 ? -1.f : 1.f);
            v[j + 4] = d * grid2[j] * ((signs >> (j + 4)) & 1 ? -1.f : 1.f);
        }
    } else if constexpr (TY == 16) {
        const block_iq2_xxs* x = (const block_iq2_xxs*) vx + ibs;
        const uint16_t* q2 = x->qs + 4 * ib;
        const uint8_t* aux8 = (const uint8_t*) q2;
        const uint8_t* grid = (const uint8_t*) (iq2xxs_grid + aux8[il]);
        const uint32_t aux32 = q2[2] | (q2[3] << 16);
        const float d = (float) x->d * (0.5f + (aux32 >> 28)) * 0.25f;
        const uint32_t signs = ksign7((aux32 >> 7 * il) & 127);
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = d * grid[j] * ((signs >> j) & 1 ? -1.f : 1.f);
    } else if constexpr (TY == 18) {
        const block_iq3_xxs* x = (const block_iq3_xxs*) vx + ibs;
        const uint8_t* q3 = x->qs + 8 * ib;
        const uint16_t* gas = (const uint16_t*) (x->qs + QK_K / 4) + 2 * ib;
        const uint8_t* grid1 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 0]);
        const uint8_t* grid2 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 1]);
        const uint32_t aux32 = gas[0] | (gas[1] << 16);
        const float d = (float) x->d * (0.5f + (aux32 >> 28)) * 0.5f;
        const uint32_t signs = ksign7((aux32 >> 7 * il) & 127);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[j + 0] = d * grid1[j] * ((signs >> (j + 0)) & 1 ? -1.f : 1.f);
            v[j + 4] = d * grid2[j] * ((signs >> (j + 4)) & 1 ? -1.f : 1.f);
        }
    }
    store_run<half, 8>(y, v);
}
template <int TY>
inline void dq_best(const void* base, int64_t ibs, half* y, int tid) {
    if constexpr (TY == 16 || TY == 18 || TY == 21) dq_fast<TY>(base, ibs, y, tid);
    else dq_run<TY>(base, ibs, y, tid);
}

// v5: one 32-value sub-block of a row = load_raw (the block fields it needs, 2-byte loads: the blocks are only 2-byte
// aligned) + decode (registers -> 32 values), split so the kernel can issue the NEXT step's loads before decoding this
// one.  Values bit for bit those of the dq_* run decoders (kmask_iq2xs[j] = 1 << j, ksigns_iq2xs[i] = i | parity << 7).
inline uint32_t ld16(const uint8_t* p) { return *reinterpret_cast<const uint16_t*>(p); }
struct Raw { uint32_t r[9]; };
inline float h2f(uint32_t bits) { return (float) sycl::bit_cast<half>((uint16_t) bits); }
template <int TY>
inline Raw load_raw(const uint8_t* vx, int64_t ibs, int ib) {
    Raw w{};
    if constexpr (TY == 21) {   // IQ3_S 110 B: d | qs[64] | qh[8] | signs[32] | scales[4]
        const uint8_t* x = vx + ibs * 110;
        w.r[0] = ld16(x);
#pragma unroll
        for (int i = 0; i < 4; ++i) w.r[1 + i] = ld16(x + 2 + 8 * ib + 2 * i);
        w.r[5] = ld16(x + 74 + 4 * ib); w.r[6] = ld16(x + 76 + 4 * ib);
        w.r[7] = x[66 + ib]; w.r[8] = x[106 + ib / 2];
    } else if constexpr (TY == 16) {   // IQ2_XXS 66 B: d | qs[32] u16
        const uint8_t* x = vx + ibs * 66;
        w.r[0] = ld16(x);
#pragma unroll
        for (int i = 0; i < 4; ++i) w.r[1 + i] = ld16(x + 2 + 8 * ib + 2 * i);
    } else if constexpr (TY == 18) {   // IQ3_XXS 98 B: d | qs[64] | gas[32]
        const uint8_t* x = vx + ibs * 98;
        w.r[0] = ld16(x);
#pragma unroll
        for (int i = 0; i < 4; ++i) w.r[1 + i] = ld16(x + 2 + 8 * ib + 2 * i);
        w.r[5] = ld16(x + 66 + 4 * ib); w.r[6] = ld16(x + 68 + 4 * ib);
    } else if constexpr (TY == 17) {   // IQ2_XS 74 B: d | qs[32] u16 | scales[8]
        const uint8_t* x = vx + ibs * 74;
        w.r[0] = ld16(x);
#pragma unroll
        for (int i = 0; i < 4; ++i) w.r[1 + i] = ld16(x + 2 + 8 * ib + 2 * i);
        w.r[5] = x[66 + ib];
    } else if constexpr (TY == 22) {   // IQ2_S 82 B: d | qs[32] | signs[32] | qh[8] | scales[8]
        const uint8_t* x = vx + ibs * 82;
        w.r[0] = ld16(x);
        w.r[1] = ld16(x + 2 + 4 * ib); w.r[2] = ld16(x + 4 + 4 * ib);
        w.r[3] = ld16(x + 34 + 4 * ib); w.r[4] = ld16(x + 36 + 4 * ib);
        w.r[5] = x[66 + ib]; w.r[6] = x[74 + ib];
    } else if constexpr (TY == 42) {   // Q2_0: 18 B per 64 (d | qs[16]); sub-block = half ib % 2 of block ib / 2
        const uint8_t* x = vx + (ibs * 4 + ib / 2) * 18;
        w.r[0] = ld16(x);
#pragma unroll
        for (int i = 0; i < 4; ++i) w.r[1 + i] = ld16(x + 2 + 8 * (ib % 2) + 2 * i);
    } else if constexpr (TY == 20) {   // IQ4_NL: 18 B per 32 (d | qs[16])
        const uint8_t* x = vx + (ibs * 8 + ib) * 18;
        w.r[0] = ld16(x);
#pragma unroll
        for (int i = 0; i < 8; ++i) w.r[1 + i] = ld16(x + 2 + 2 * i);
    }
    return w;
}
// grid lookups: the device-global tables, or (TAB) their copy in local memory as 32-bit words
template <int TY, bool TAB>
inline uint32_t g32(const uint32_t* tab, uint32_t i) {
    if constexpr (TAB) return tab[i];
    else if constexpr (TY == 21) return iq3s_grid[i];
    else return iq3xxs_grid[i];
}
template <int TY, bool TAB>
inline uint64_t g64(const uint32_t* tab, uint32_t i) {
    if constexpr (TAB) return (uint64_t) tab[2 * i] | ((uint64_t) tab[2 * i + 1] << 32);
    else if constexpr (TY == 16) return iq2xxs_grid[i];
    else if constexpr (TY == 17) return iq2xs_grid[i];
    else return iq2s_grid[i];
}
template <int TY, bool TAB = false>
inline void decode(const Raw& w, int ib, float* v, const uint32_t* tab = nullptr) {
    if constexpr (TY == 21) {
        const float d = h2f(w.r[0]) * (1 + 2 * ((w.r[8] >> 4 * (ib % 2)) & 0xf));
        const uint32_t qh = w.r[7];
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t g1 = g32<21, TAB>(tab, (w.r[1 + il] & 0xff) | ((qh << (8 - 2 * il)) & 256));
            const uint32_t g2 = g32<21, TAB>(tab, (w.r[1 + il] >> 8) | ((qh << (7 - 2 * il)) & 256));
            const uint32_t signs = (w.r[5 + il / 2] >> (8 * (il % 2))) & 0xff;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v[8 * il + j] = d * (float) ((g1 >> (8 * j)) & 0xff) * ((signs >> j) & 1 ? -1.f : 1.f);
                v[8 * il + 4 + j] = d * (float) ((g2 >> (8 * j)) & 0xff) * ((signs >> (j + 4)) & 1 ? -1.f : 1.f);
            }
        }
    } else if constexpr (TY == 16) {
        const uint32_t aux32 = w.r[3] | (w.r[4] << 16);
        const float d = h2f(w.r[0]) * (0.5f + (aux32 >> 28)) * 0.25f;
        const uint32_t idx[4] = {w.r[1] & 0xff, w.r[1] >> 8, w.r[2] & 0xff, w.r[2] >> 8};
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint64_t g = g64<16, TAB>(tab, idx[il]);
            const uint32_t signs = ksign7((aux32 >> 7 * il) & 127);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                v[8 * il + j] = d * (float) ((g >> (8 * j)) & 0xff) * ((signs >> j) & 1 ? -1.f : 1.f);
        }
    } else if constexpr (TY == 18) {
        const uint32_t aux32 = w.r[5] | (w.r[6] << 16);
        const float d = h2f(w.r[0]) * (0.5f + (aux32 >> 28)) * 0.5f;
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t g1 = g32<18, TAB>(tab, w.r[1 + il] & 0xff), g2 = g32<18, TAB>(tab, w.r[1 + il] >> 8);
            const uint32_t signs = ksign7((aux32 >> 7 * il) & 127);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v[8 * il + j] = d * (float) ((g1 >> (8 * j)) & 0xff) * ((signs >> j) & 1 ? -1.f : 1.f);
                v[8 * il + 4 + j] = d * (float) ((g2 >> (8 * j)) & 0xff) * ((signs >> (j + 4)) & 1 ? -1.f : 1.f);
            }
        }
    } else if constexpr (TY == 17) {
        const float db = h2f(w.r[0]);
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t q = w.r[1 + il];
            const uint64_t g = g64<17, TAB>(tab, q & 511);
            const float d = db * (0.5f + ((w.r[5] >> 4 * (il / 2)) & 0xf)) * 0.25f;
            const uint32_t signs = ksign7(q >> 9);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                v[8 * il + j] = d * (float) ((g >> (8 * j)) & 0xff) * ((signs >> j) & 1 ? -1.f : 1.f);
        }
    } else if constexpr (TY == 22) {
        const float db = h2f(w.r[0]);
        const uint32_t qh = w.r[5];
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t qb = (w.r[1 + il / 2] >> (8 * (il % 2))) & 0xff;
            const uint64_t g = g64<22, TAB>(tab, qb | ((qh << (8 - 2 * il)) & 0x300));
            const float d = db * (0.5f + ((w.r[6] >> 4 * (il / 2)) & 0xf)) * 0.25f;
            const uint32_t signs = (w.r[3 + il / 2] >> (8 * (il % 2))) & 0xff;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                v[8 * il + j] = d * (float) ((g >> (8 * j)) & 0xff) * ((signs >> j) & 1 ? -1.f : 1.f);
        }
    } else if constexpr (TY == 42) {
        const float d = h2f(w.r[0]);
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j) v[8 * i + j] = d * (float) ((int) ((w.r[1 + i] >> (2 * j)) & 3) - 1);
    } else if constexpr (TY == 20) {
        const float d = h2f(w.r[0]);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t q = w.r[1 + i];
            v[2 * i] = d * kvalues_iq4nl[q & 0xf];
            v[2 * i + 1] = d * kvalues_iq4nl[(q >> 8) & 0xf];
            v[16 + 2 * i] = d * kvalues_iq4nl[(q >> 4) & 0xf];
            v[16 + 2 * i + 1] = d * kvalues_iq4nl[(q >> 12) & 0xf];
        }
    }
}
template <int TY> constexpr bool has_dq32() {
    return TY == 16 || TY == 17 || TY == 18 || TY == 20 || TY == 21 || TY == 22 || TY == 42;
}

constexpr int kSg = 16;

// v4 (v1's shape): a sub-group owns 16 output columns (weight rows), each lane one row; per 32-value K step the lane
// decodes its row's sub-block (dq32: shared fields once) into its slot of the sub-group's local slice, then 2 x MT
// DPAS against the entry's rows.
template <int TY, int MT, int SGN, bool GU, int MODE>
class k_night_grouped;

// MODE (bench only, NIGHTG_MODE): 0 full, 1 decode only (no DPAS), 2 DPAS only (no decode)
template <int TY, int MT, int SGN, bool GU, int MODE = 0>
void launch(sycl::queue& q, const NightGroupedEntry* ent, int n_ent, const half* X, float* Y, int N, int K,
            size_t up_off) {
    constexpr int WG = SGN * kSg, LDB = 32 + 8;
    const int nb = N / (SGN * kSg);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> slm(SGN * kSg * LDB, h);
        auto props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::intel::experimental::grf_size<(MT > 4 ? 256 : 128)>};
        h.parallel_for<k_night_grouped<TY, MT, SGN, GU, MODE>>(
            sycl::nd_range<2>({(size_t) n_ent, (size_t) nb * WG}, {1, (size_t) WG}), props,
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSg)]] {
                auto sg = it.get_sub_group();
                const int lane = (int) sg.get_local_linear_id();
                const int sgid = (int) it.get_local_id(1) / kSg;
                const NightGroupedEntry en = ent[it.get_group(0)];
                const int rows = en.rows;
                const int n0 = (int) it.get_group(1) * (SGN * kSg) + sgid * kSg;
                const int vr = n0 + lane;
                const uint8_t* base = GU ? ((vr & 1) ? en.w + up_off : en.w) : en.w;
                const int64_t row = GU ? (vr >> 1) : vr;
                auto sp = slm.template get_multi_ptr<sycl::access::decorated::no>() + sgid * kSg * LDB;
                auto myp = sp + lane * LDB;
                jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, kSg> acc[MT];
#pragma unroll
                for (int i = 0; i < MT; ++i) jm::joint_matrix_fill(sg, acc[i], 0.0f);
                auto xp = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                   sycl::access::decorated::no>(const_cast<half*>(X + en.row0 * K));
                Raw cur{};
                if constexpr (has_dq32<TY>() && MODE != 2) cur = load_raw<TY>(base, (row * K) >> 8, (int) ((row * K) & 255) >> 5);
                for (int k0 = 0; k0 < K; k0 += 32) {
                    const int64_t flat = row * K + k0;
                    const int64_t ibs = flat >> 8;
                    const int o = (int) (flat & 255);
                    if constexpr (MODE == 2) {
                    } else if constexpr (has_dq32<TY>()) {
                        // the next step's block fields first (no dependency on this step), then this step's values
                        Raw nxt{};
                        if (k0 + 32 < K) nxt = load_raw<TY>(base, (flat + 32) >> 8, (int) ((flat + 32) & 255) >> 5);
                        float v[32];
                        decode<TY>(cur, o >> 5, v);
                        cur = nxt;
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            sycl::vec<half, 8> hv;
#pragma unroll
                            for (int j = 0; j < 8; ++j) hv[j] = half(v[8 * c + j]);
                            hv.store(c, myp);
                        }
                    } else {
                        half* my = myp.get();
#pragma unroll
                        for (int j = 0; j < 4; ++j) dq_run<TY>(base, ibs, my - o, (o >> 3) + j);
                    }
                    sycl::group_barrier(sg);
                    if constexpr (MODE != 1)
#pragma unroll
                    for (int kk = 0; kk < 32; kk += 16) {
                        jm::joint_matrix<sycl::sub_group, half, jm::use::b, 16, kSg, jm::layout::col_major> b;
                        jm::joint_matrix_load(sg, b, sp + kk, LDB);
#pragma unroll
                        for (int i = 0; i < MT; ++i) {
                            if (i * 8 < rows) {
                                jm::joint_matrix<sycl::sub_group, half, jm::use::a, 8, 16, jm::layout::row_major> a;
                                jmi::joint_matrix_load_checked(sg, a, xp, (size_t) K, (size_t) rows, (size_t) K,
                                                               (size_t) (i * 8), (size_t) (k0 + kk));
                                jm::joint_matrix_mad(sg, acc[i], a, b, acc[i]);
                            }
                        }
                    }
                    sycl::group_barrier(sg);
                }
                auto yp = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                   sycl::access::decorated::no>(Y + en.row0 * N);
#pragma unroll
                for (int i = 0; i < MT; ++i)
                    if (i * 8 < rows)
                        jmi::joint_matrix_store_checked(sg, acc[i], yp, (size_t) N, jm::layout::row_major,
                                                        (size_t) rows, (size_t) N, (size_t) (i * 8), (size_t) n0);
            });
    });
}

template <int TY, int SGN, bool GU>
bool by_mt_sgn(sycl::queue& q, int mt, const NightGroupedEntry* ent, int n_ent, const half* X, float* Y, int N, int K,
               size_t up_off) {
    if (N % (SGN * kSg) != 0 || K % 32 != 0) return false;
    static const int mode = [] { const char* e = std::getenv("NIGHTG_MODE"); return e ? std::atoi(e) : 0; }();
    if (mode == 1 && mt == 2) { launch<TY, 2, SGN, GU, 1>(q, ent, n_ent, X, Y, N, K, up_off); return true; }
    if (mode == 2 && mt == 2) { launch<TY, 2, SGN, GU, 2>(q, ent, n_ent, X, Y, N, K, up_off); return true; }
    switch (mt) {
        case 2: launch<TY, 2, SGN, GU>(q, ent, n_ent, X, Y, N, K, up_off); return true;
        case 4: launch<TY, 4, SGN, GU>(q, ent, n_ent, X, Y, N, K, up_off); return true;
        case 8: launch<TY, 8, SGN, GU>(q, ent, n_ent, X, Y, N, K, up_off); return true;
        case 16: launch<TY, 16, SGN, GU>(q, ent, n_ent, X, Y, N, K, up_off); return true;
        default: return false;
    }
}
// sub-groups per work-group: 4 (measured default; 2 / 8 within 4%); NIGHTG_SGN=2 / 8 for the bench's tile sweep
template <int TY, bool GU>
bool by_mt(sycl::queue& q, int mt, const NightGroupedEntry* ent, int n_ent, const half* X, float* Y, int N, int K,
           size_t up_off) {
    static const int sgn = [] { const char* e = std::getenv("NIGHTG_SGN"); return e ? std::atoi(e) : 4; }();
    (void) sgn;   // (2 / 8 measured within 4% of 4: only 4 is built)
    return by_mt_sgn<TY, 4, GU>(q, mt, ent, n_ent, X, Y, N, K, up_off);
}
}  // namespace nightg

bool night_grouped_f16(int ty, bool gu, const NightGroupedEntry* ent, int n_ent, int mt, const uint16_t* X, float* Y,
                       int N, int K, size_t up_off, void* stream) {
    if (n_ent <= 0) return true;
    if (K % 32 != 0) return false;
    sycl::queue& q = *strata::q_of(stream);
    const auto* Xh = reinterpret_cast<const sycl::half*>(X);
    if (gu) {
        switch (ty) {
            case 16: return nightg::by_mt<16, true>(q, mt, ent, n_ent, Xh, Y, N, K, up_off);
            case 17: return nightg::by_mt<17, true>(q, mt, ent, n_ent, Xh, Y, N, K, up_off);
            case 18: return nightg::by_mt<18, true>(q, mt, ent, n_ent, Xh, Y, N, K, up_off);
            case 21: return nightg::by_mt<21, true>(q, mt, ent, n_ent, Xh, Y, N, K, up_off);
            case 22: return nightg::by_mt<22, true>(q, mt, ent, n_ent, Xh, Y, N, K, up_off);
            default: return false;
        }
    }
    switch (ty) {
        case 20: return nightg::by_mt<20, false>(q, mt, ent, n_ent, Xh, Y, N, K, up_off);
        case 42: return nightg::by_mt<42, false>(q, mt, ent, n_ent, Xh, Y, N, K, up_off);
        default: return false;
    }
}
