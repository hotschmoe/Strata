// src/kernels/cuda/ep_kernels.dp.cpp (SYCL port only) - see include/strata/kernels/ep_kernels.hpp.
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include "strata/sycl_queue.hpp"
#include "strata/sycl_doorbell.hpp"
#include "strata/kernels/ep_kernels.hpp"

namespace strata::kernels {
namespace {
// the doorbell's uncached L1+L3 read (sycl_doorbell.hpp), for whole buffers
inline uint32_t load_uncached(const uint32_t* p) {
    sycl::ext::oneapi::experimental::annotated_ptr<uint32_t, doorbell_uncached_read> u(const_cast<uint32_t*>(p));
    return u[0];
}
}  // namespace

void ep_copy_uncached(void* dst, const void* src, size_t bytes, void* stream) {
    if (bytes == 0) return;
    constexpr size_t WG = 256;
    const size_t words = bytes / 4;
    const size_t groups = std::min<size_t>((words + WG - 1) / WG, 64);
    uint32_t* d = (uint32_t*) dst;
    const uint32_t* s = (const uint32_t*) src;
    q_of(stream)->parallel_for<class ep_copy_uncached_kernel>(sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) {
        for (size_t i = it.get_global_id(0); i < words; i += groups * WG) d[i] = load_uncached(s + i);
    });
}

// the last work-group of a reduce raises the other card's doorbell (night lab HALVES; flag == nullptr: nothing)
inline void reduce_signal(sycl::nd_item<1> it, size_t groups, uint32_t* done, uint32_t* flag, uint32_t value) {
    using sys_rmw = sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::system>;
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
    sycl::group_barrier(it.get_group(), sycl::memory_scope::system);
    if (it.get_local_id(0) == 0 && sys_rmw(*done).fetch_add(1u) == (uint32_t) groups - 1) {
        *done = 0;
        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
        sys_store(flag, value);
    }
}
template<bool Push> class ep_prefill_reduce_kernel;
template<bool Push>
static void reduce_impl(const float* rows, const int32_t* slot, const float* weights, float* sum,
                        int64_t tokens, int64_t local_rows, int64_t width, int k, void* stream,
                        uint32_t* done, uint32_t* flag, uint32_t value) {
    constexpr size_t WG = 256;
    const size_t vectors = (size_t) tokens * (size_t) width / 4;
    const size_t groups = Push ? std::min<size_t>((vectors + WG - 1) / WG, 64) : (vectors + WG - 1) / WG;
    q_of(stream)->parallel_for<ep_prefill_reduce_kernel<Push>>(
        sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) {
            for (size_t i = it.get_global_id(0); i < vectors; i += groups * WG) {
                const int64_t t = i / (width / 4), d = i % (width / 4) * 4;
                sycl::float4 acc(0.0f);
                for (int j = 0; j < k; ++j) {
                    const int64_t row = slot[t * k + j] - local_rows;
                    if (row >= 0) {
                        const auto value = *reinterpret_cast<const sycl::float4*>(rows + row * width + d);
                        acc = sycl::fma(sycl::float4(weights[t * k + j]), value, acc);
                    }
                }
                reinterpret_cast<sycl::float4*>(sum)[i] = acc;
            }
            if constexpr (Push) sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
            if (flag != nullptr) reduce_signal(it, groups, done, flag, value);
        });
}
void ep_prefill_reduce(const float* rows, const int32_t* slot, const float* weights, float* sum,
                       int64_t tokens, int64_t local_rows, int64_t width, int k, void* stream, bool push,
                       uint32_t* done, uint32_t* flag, uint32_t value) {
    if (push) reduce_impl<true>(rows, slot, weights, sum, tokens, local_rows, width, k, stream, done, flag, value);
    else reduce_impl<false>(rows, slot, weights, sum, tokens, local_rows, width, k, stream, done, flag, value);
}

void ep_prefill_finish(const float* rows, const int32_t* slot, const float* weights,
                       const float* shared, const float* gate, float* out,
                       int64_t tokens, int64_t local_rows, int64_t width, int k, void* stream) {
    constexpr size_t WG = 256;
    const size_t vectors = (size_t) tokens * (size_t) width / 4;
    q_of(stream)->parallel_for<class ep_prefill_finish_kernel>(
        sycl::nd_range<1>(((vectors + WG - 1) / WG) * WG, WG), [=](sycl::nd_item<1> it) {
            const size_t i = it.get_global_id(0);
            if (i >= vectors) return;
            const int64_t t = i / (width / 4), d = i % (width / 4) * 4;
            sycl::float4 acc(0.0f);
            for (int j = 0; j < k; ++j) {
                const int64_t row = slot[t * k + j];
                if (row < local_rows) {
                    const auto value = *reinterpret_cast<const sycl::float4*>(rows + row * width + d);
                    acc = sycl::fma(sycl::float4(weights[t * k + j]), value, acc);
                }
            }
            const float g = 1.0f / (1.0f + sycl::native::exp(-gate[t]));
            const auto sh = reinterpret_cast<const sycl::float4*>(shared)[i];
            auto* dst = reinterpret_cast<sycl::float4*>(out);
            dst[i] = acc + dst[i] + sh * g;
        });
}

// Night lab: FP16 + per-token power-of-two scale return of the peer sums (see ep_kernels.hpp).
template<bool Push> class ep_prefill_reduce_h_kernel;
template<bool Push>
static void reduce_h_impl(const float* rows, const int32_t* slot, const float* weights, uint16_t* sum_h, float* scale,
                          int64_t tokens, int64_t local_rows, int64_t width, int k, void* stream,
                          uint32_t* done, uint32_t* flag, uint32_t value) {
    constexpr int WG = 160, V = 4;                       // 160 x 4 float4 = 2560 values per token row
    const int64_t vec_row = width / 4;
    // STRATA_NIGHT_PREFILL_F16SUM_GROUPS: the work-groups of the push (default 64, as ep_prefill_reduce's push)
    static const int64_t push_groups = [] {
        const char* v = std::getenv("STRATA_NIGHT_PREFILL_F16SUM_GROUPS");
        return v ? std::max<int64_t>(1, std::atoll(v)) : (int64_t) 64;
    }();
    const size_t groups = Push ? (size_t) std::min<int64_t>(tokens, push_groups) : (size_t) std::min<int64_t>(tokens, 1024);
    q_of(stream)->parallel_for<ep_prefill_reduce_h_kernel<Push>>(
        sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) {
            const int lid = (int) it.get_local_id(0);
            for (int64_t t = it.get_group(0); t < tokens; t += (int64_t) groups) {
                sycl::float4 acc[V];
                float mx = 0.0f;
                for (int v = 0; v < V; ++v) {
                    const int64_t d = ((int64_t) lid + (int64_t) v * WG) * 4;
                    sycl::float4 a(0.0f);
                    if (d < width) {
                        for (int j = 0; j < k; ++j) {
                            const int64_t row = slot[t * k + j] - local_rows;
                            if (row >= 0) {
                                const auto value = *reinterpret_cast<const sycl::float4*>(rows + row * width + d);
                                a = sycl::fma(sycl::float4(weights[t * k + j]), value, a);
                            }
                        }
                    }
                    acc[v] = a;
                    mx = sycl::fmax(mx, sycl::fmax(sycl::fmax(sycl::fabs(a.x()), sycl::fabs(a.y())),
                                                   sycl::fmax(sycl::fabs(a.z()), sycl::fabs(a.w()))));
                }
                mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
                int e = 0;
                if (mx > 0.0f && sycl::isfinite(mx)) e = sycl::ilogb(mx) - 14;   // max * 2^-e in [2^14, 2^15)
                const float inv = sycl::ldexp(1.0f, -e);
                for (int v = 0; v < V; ++v) {
                    const int64_t d = ((int64_t) lid + (int64_t) v * WG) * 4;
                    if (d >= width) continue;
                    const sycl::float4 a = acc[v] * inv;
                    sycl::vec<sycl::half, 4> h(sycl::half(a.x()), sycl::half(a.y()), sycl::half(a.z()), sycl::half(a.w()));
                    *reinterpret_cast<sycl::vec<sycl::half, 4>*>(sum_h + t * width + d) = h;
                }
                if (lid == 0) scale[t] = sycl::ldexp(1.0f, e);
                (void) vec_row;
            }
            if constexpr (Push) sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
            if (flag != nullptr) reduce_signal(it, groups, done, flag, value);
        });
}
void ep_prefill_reduce_h(const float* rows, const int32_t* slot, const float* weights, uint16_t* sum_h, float* scale,
                         int64_t tokens, int64_t local_rows, int64_t width, int k, void* stream, bool push,
                         uint32_t* done, uint32_t* flag, uint32_t value) {
    if (tokens <= 0) return;
    if (push) reduce_h_impl<true>(rows, slot, weights, sum_h, scale, tokens, local_rows, width, k, stream, done, flag, value);
    else reduce_h_impl<false>(rows, slot, weights, sum_h, scale, tokens, local_rows, width, k, stream, done, flag, value);
}

void ep_prefill_finish_h(const float* rows, const int32_t* slot, const float* weights, const float* shared,
                         const float* gate, const uint16_t* sum_h, const float* scale, float* out, int64_t tokens,
                         int64_t local_rows, int64_t width, int k, void* stream) {
    constexpr size_t WG = 256;
    const size_t vectors = (size_t) tokens * (size_t) width / 4;
    q_of(stream)->parallel_for<class ep_prefill_finish_h_kernel>(
        sycl::nd_range<1>(((vectors + WG - 1) / WG) * WG, WG), [=](sycl::nd_item<1> it) {
            const size_t i = it.get_global_id(0);
            if (i >= vectors) return;
            const int64_t t = i / (width / 4), d = i % (width / 4) * 4;
            sycl::float4 acc(0.0f);
            for (int j = 0; j < k; ++j) {
                const int64_t row = slot[t * k + j];
                if (row < local_rows) {
                    const auto value = *reinterpret_cast<const sycl::float4*>(rows + row * width + d);
                    acc = sycl::fma(sycl::float4(weights[t * k + j]), value, acc);
                }
            }
            // 4 halves = 2 words, read past the caches (the other card wrote them over PCIe)
            const uint32_t* w = reinterpret_cast<const uint32_t*>(sum_h + t * width + d);
            const uint32_t w0 = load_uncached(w), w1 = load_uncached(w + 1);
            const uint32_t sw = load_uncached(reinterpret_cast<const uint32_t*>(scale + t));
            const float sc = sycl::bit_cast<float>(sw);
            const auto h0 = sycl::bit_cast<sycl::vec<sycl::half, 2>>(w0), h1 = sycl::bit_cast<sycl::vec<sycl::half, 2>>(w1);
            const sycl::float4 peer(sc * (float) h0[0], sc * (float) h0[1], sc * (float) h1[0], sc * (float) h1[1]);
            const float g = 1.0f / (1.0f + sycl::native::exp(-gate[t]));
            const auto sh = reinterpret_cast<const sycl::float4*>(shared)[i];
            auto* dst = reinterpret_cast<sycl::float4*>(out);
            dst[i] = acc + peer + sh * g;
        });
}

void ep_push_signal(void* peer_dst, const void* src, int64_t n4, uint32_t* done, uint32_t* peer_flag, uint32_t value,
                    void* stream) {
    if (n4 <= 0) return;
    constexpr size_t WG = 256;
    const size_t groups = (size_t) std::min<int64_t>((n4 + (int64_t) WG - 1) / (int64_t) WG, 4096);
    using sys_rmw = sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::system>;
    sycl::float4* d = (sycl::float4*) peer_dst;
    const sycl::float4* s = (const sycl::float4*) src;
    q_of(stream)->parallel_for<class night_ep_push_signal_kernel>(sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) {
        for (int64_t i = (int64_t) it.get_global_id(0); i < n4; i += (int64_t) (groups * WG)) d[i] = s[i];
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        sycl::group_barrier(it.get_group(), sycl::memory_scope::system);
        if (it.get_local_id(0) == 0 && sys_rmw(*done).fetch_add(1u) == (uint32_t) groups - 1) {
            *done = 0;   // every group has counted itself: the next call starts from zero
            sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
            sys_store(peer_flag, value);
        }
    });
}

void ep_wait_value(const uint32_t* flag, uint32_t value, uint32_t spin_max, uint32_t* err, uint32_t code, void* stream) {
    q_of(stream)->single_task<class night_ep_wait_value_kernel>([=] {
        uint32_t spin = 0;
        while ((int32_t) (sys_load(flag) - value) < 0) {
            if (++spin >= spin_max) {
                if (err != nullptr) sys_store(err, code);
                break;
            }
        }
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    });
}

void ep_bump(uint32_t* ctr, void* stream) {
    q_of(stream)->single_task<class ep_bump_kernel>([=] { *ctr += 1; });
}

void ep_push(const int32_t* ids, int n_ids, const uint8_t* xq, size_t xq_bytes, int32_t* peer_ids, uint8_t* peer_xq,
             uint32_t* peer_flag, const uint32_t* ctr, void* stream) {
    constexpr size_t WG = 256;
    // q8_1 rows are 36-byte blocks: copied as 4-byte words (both buffers are 4-byte aligned, xq_bytes a multiple of 4)
    const size_t words = xq_bytes / 4;
    q_of(stream)->parallel_for<class ep_push_kernel>(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
        const size_t lid = it.get_local_id(0);
        for (size_t i = lid; i < (size_t) n_ids; i += WG) peer_ids[i] = ids[i];
        const uint32_t* s = (const uint32_t*) xq;
        uint32_t* d = (uint32_t*) peer_xq;
        for (size_t i = lid; i < words; i += WG) d[i] = s[i];
        // Keep cross-device ordering on the barrier itself: a default-scope
        // barrier can absorb the adjacent fence with only work-group scope.
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        sycl::group_barrier(it.get_group(), sycl::memory_scope::system);
        if (lid == 0) sys_store(peer_flag, *ctr);
    });
}

void ep_wait(const uint32_t* flag, const uint32_t* ctr, uint32_t spin_max, uint32_t* err, uint32_t code, void* stream) {
    q_of(stream)->single_task<class ep_wait_kernel>([=] {
        const uint32_t want = *ctr;
        uint32_t spin = 0;
        // the epoch counters only grow and wrap: compare the difference, not the values
        while ((int32_t) (sys_load(flag) - want) < 0) {
            if (++spin >= spin_max) {
                if (err != nullptr) sys_store(err, code);
                break;
            }
        }
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    });
}

void ep_send_rows(float* peer_rows, const float* rows, const int32_t* ids, const int32_t* res0, int n, int64_t row,
                  uint32_t* done, uint32_t* peer_flag, const uint32_t* ctr, void* stream) {
    if (n <= 0) return;
    constexpr size_t WG = 256;
    using sys_rmw = sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::system>;
    q_of(stream)->parallel_for<class ep_send_rows_kernel>(sycl::nd_range<1>((size_t) n * WG, WG), [=](sycl::nd_item<1> it) {
        const int r = (int) it.get_group(0);
        const int32_t e = ids[r];
        if (!(e >= 0 && res0[e] >= 0)) {   // the peer's entry: its row goes to card 0
            const sycl::float4* s = (const sycl::float4*) (rows + (size_t) r * row);
            sycl::float4* d = (sycl::float4*) (peer_rows + (size_t) r * row);
            for (int64_t i = it.get_local_id(0); i < row / 4; i += WG) d[i] = s[i];
        }
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        sycl::group_barrier(it.get_group());
        if (it.get_local_id(0) == 0 && sys_rmw(*done).fetch_add(1u) == (uint32_t) n - 1) {
            *done = 0;   // every group has counted itself: the next call starts from zero
            sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
            sys_store(peer_flag, *ctr);
        }
    });
}


// ---- night lab STRATA_NIGHT_EP_PSUM ----
#ifndef STRATA_NIGHT_PSUM_UNCACHED
#define STRATA_NIGHT_PSUM_UNCACHED 0
#endif
// ----------------------------------------------------------------------------------
void ep_push_w(const int32_t* ids, const float* w, int n_ids, const uint8_t* xq, size_t xq_bytes, int32_t* peer_ids,
               float* peer_w, uint8_t* peer_xq, uint32_t* peer_flag, const uint32_t* ctr, void* stream,
               const int32_t* plan, int plan_words, int32_t* peer_plan) {
    constexpr size_t WG = 256;
    const size_t words = xq_bytes / 4;
    q_of(stream)->parallel_for<class ep_push_w_kernel>(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
        const size_t lid = it.get_local_id(0);
        for (size_t i = lid; i < (size_t) n_ids; i += WG) {
            peer_ids[i] = ids[i];
            if (w != nullptr) peer_w[i] = w[i];
        }
        if (plan != nullptr)
            for (size_t i = lid; i < (size_t) plan_words; i += WG) peer_plan[i] = plan[i];
        const uint32_t* s = (const uint32_t*) xq;
        uint32_t* d = (uint32_t*) peer_xq;
        for (size_t i = lid; i < words; i += WG) d[i] = s[i];
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        sycl::group_barrier(it.get_group(), sycl::memory_scope::system);
        if (lid == 0) sys_store(peer_flag, *ctr);
    });
}

void ep_send_psum(float* peer_sum, const float* rows, const int32_t* ids, const float* w, const int32_t* res0,
                  int n_tok, int k, int64_t row, uint32_t* done, uint32_t* peer_flag, const uint32_t* ctr, void* stream) {
    if (n_tok <= 0) return;
    constexpr size_t WG = 64;                       // 64 float4 = 256 columns per work-group
    const int64_t n4 = row / 4;
    const int chunks = (int) ((n4 + WG - 1) / WG);
    const uint32_t total = (uint32_t) (n_tok * chunks);
    using sys_rmw = sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::system>;
    q_of(stream)->parallel_for<class ep_send_psum_kernel>(
        sycl::nd_range<1>((size_t) total * WG, WG), [=](sycl::nd_item<1> it) {
            const int g = (int) it.get_group(0);
            const int t = g / chunks;
            const int64_t c4 = (int64_t) (g % chunks) * WG + (int64_t) it.get_local_id(0);
            if (c4 < n4) {
                sycl::float4 acc(0.0f);
                for (int j = 0; j < k; ++j) {
                    const int32_t e = ids[t * k + j];
                    if (e >= 0 && res0[e] >= 0) continue;   // card 0's entry
                    const sycl::float4 v = reinterpret_cast<const sycl::float4*>(rows + (size_t) (t * k + j) * row)[c4];
                    acc = sycl::fma(v, sycl::float4(w[t * k + j]), acc);   // fma(v, w, 0) = the rounded product
                }
                reinterpret_cast<sycl::float4*>(peer_sum + (size_t) t * row)[c4] = acc;
            }
            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
            sycl::group_barrier(it.get_group());
            if (it.get_local_id(0) == 0 && sys_rmw(*done).fetch_add(1u) == total - 1) {
                *done = 0;
                sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
                sys_store(peer_flag, *ctr);
            }
        });
}

template <int KK, bool GATED> class ep_combine_psum_kernel;
template <int KK, bool GATED>
static void combine_psum_impl(const float* parts, const float* w, const int32_t* ids, const int32_t* res0,
                              const float* psum, const float* shared, const float* sg, float* out, int64_t row,
                              int n_tok, void* stream) {
    constexpr size_t WG = 128;
    const int64_t n4 = row / 4;
    const size_t gx = (size_t) ((n4 + WG - 1) / WG);
    q_of(stream)->parallel_for<ep_combine_psum_kernel<KK, GATED>>(
        sycl::nd_range<2>(sycl::range<2>((size_t) n_tok, gx * WG), sycl::range<2>(1, WG)), [=](sycl::nd_item<2> it) {
            const int t = (int) it.get_global_id(0);
            const int64_t c4 = (int64_t) it.get_global_id(1);
            if (c4 >= n4) return;
            // every load first (they are independent): the peer's sum (past the caches), the shared row, ids, weights,
            // all KK part rows (parts_ holds KK rows per token; the peer's are stale and only selected away)
#if STRATA_NIGHT_PSUM_UNCACHED
            const uint64_t* pw = reinterpret_cast<const uint64_t*>(psum + (size_t) t * row) + 2 * c4;
            const auto lo = sycl::bit_cast<sycl::float2>(sys_load_mapped(pw));
            const auto hi = sycl::bit_cast<sycl::float2>(sys_load_mapped(pw + 1));
#else   // plain loads, as native_moe_combine_multi reads the peer's rows in parts_ (same region kind, same flag protocol)
            const sycl::float4 pv = reinterpret_cast<const sycl::float4*>(psum + (size_t) t * row)[c4];
            const sycl::float2 lo(pv.x(), pv.y()), hi(pv.z(), pv.w());
#endif
            const sycl::float4 sh = reinterpret_cast<const sycl::float4*>(shared + (size_t) t * row)[c4];
            int32_t e[KK];
            float wv[KK];
            sycl::float4 p[KK];
#pragma unroll
            for (int j = 0; j < KK; ++j) {
                e[j] = ids[t * KK + j];
                wv[j] = w[t * KK + j];
                p[j] = reinterpret_cast<const sycl::float4*>(parts + (size_t) (t * KK + j) * row)[c4];
            }
            bool own[KK];
#pragma unroll
            for (int j = 0; j < KK; ++j) own[j] = e[j] >= 0 && res0[e[j]] >= 0;
            sycl::float4 sum(0.0f);
            bool first = true;
#pragma unroll
            for (int j = 0; j < KK; ++j) {
                if (!own[j]) continue;   // the peer's entry: it is in psum
                if (first) { sum = p[j] * sycl::float4(wv[j]); first = false; }
                else sum = sycl::fma(p[j], sycl::float4(wv[j]), sum);
            }
            sum += sycl::float4(lo.x(), lo.y(), hi.x(), hi.y());
            if constexpr (GATED) {
#pragma clang fp contract(off)
                const float gg = 1.0f / (1.0f + sycl::native::exp(-sg[t]));
                const sycl::float4 shg = sh * sycl::float4(gg);
                sum += shg;
            } else {
                sum += sh;
            }
            reinterpret_cast<sycl::float4*>(out + (size_t) t * row)[c4] = sum;
        });
}
void ep_combine_psum(const float* parts, const float* w, const int32_t* ids, const int32_t* res0, const float* psum,
                     const float* shared, const float* sg, float* out, int64_t row, int k, int n_tok, void* stream) {
    if (n_tok <= 0) return;
    if (k != 10 || (row & 3) != 0) throw std::invalid_argument("ep_combine_psum: k must be 10 and row a multiple of 4");
    if (sg != nullptr) combine_psum_impl<10, true>(parts, w, ids, res0, psum, shared, sg, out, row, n_tok, stream);
    else combine_psum_impl<10, false>(parts, w, ids, res0, psum, shared, sg, out, row, n_tok, stream);
}

void ep_compare(const float* parts, const float* peer_rows, const int32_t* ids, const int32_t* res0, int n, int64_t row,
                uint32_t* mismatch, void* stream) {
    if (n <= 0) return;
    constexpr size_t WG = 256;
    q_of(stream)->parallel_for<class ep_compare_kernel>(sycl::nd_range<1>((size_t) n * WG, WG), [=](sycl::nd_item<1> it) {
        const int r = (int) it.get_group(0);
        const int32_t e = ids[r];
        if (e >= 0 && res0[e] >= 0) return;
        const uint32_t* a = (const uint32_t*) (parts + (size_t) r * row);
        const uint32_t* b = (const uint32_t*) (peer_rows + (size_t) r * row);
        int bad = 0;
        for (int64_t i = it.get_local_id(0); i < row; i += WG) bad |= a[i] != load_uncached(b + i);
        bad = sycl::reduce_over_group(it.get_group(), bad, sycl::bit_or<int>());
        if (it.get_local_id(0) == 0 && bad)
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system>(*mismatch).fetch_add(1u);
    });
}

}  // namespace strata::kernels
