// src/kernels/qsa_verify_batch_parity.cpp - night-lab (STRATA_NIGHT_ATTN_BATCH): a verify window's QSA select +
// attention + K/V / indexer appends, row by row (the batch-window path of verify.cpp) against the per-slot calls and
// the one-launch-per-window form, bitwise, and their graph-replay time per layer.
//
// Realistic shapes: 24 q heads, 2 KV heads, head_dim 256, INT8 K/V (64-value fp16 scales), page size 4 through a
// random page permutation, indexer 4 x 128 with random pooled keys (random top-k selections), the serving capacity
// (200000 cells: max_blocks 50002, the wide top-k kernel). 12 attention layers with their own pools (no L2 reuse
// across layers). Shapes: C2 = 2 slots x 3 rows, C1 = 1 slot x 4 rows, C2d1 = 2 slots x 2 rows; contexts from argv.
// Also the MTP drafter call (1 row, 32K window = 512 chunks).
//
//   qsa_verify_batch_parity [--ctx 1024,16384,65536] [--ref FILE] [--reps N]
// --ref: the first run (no night env) writes one FNV-1a hash per output; later runs (STRATA_NIGHT_ATTN_VPIPE=1,
// STRATA_NIGHT_ATTN_MERGE_ACTIVE=1) compare against it.  In-process, the batched paths are compared byte for byte
// against the row-by-row path.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/rope_scaling.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;
namespace sx = sycl::ext::oneapi::experimental;

namespace {
int g_fail = 0;
constexpr int L = 12;            // attention layers
constexpr int64_t kMaxCells = 200000;

inline uint32_t hsh(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
void fill_i8(sycl::queue& q, int8_t* p, size_t n, uint32_t seed) {
    const uint32_t s0 = hsh(seed * 0x9e3779b9u + 1);
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        const uint32_t h = hsh((uint32_t) i[0] ^ s0) ^ hsh((uint32_t) (i[0] >> 32) + s0);
        p[i] = (int8_t) ((int) (h % 255u) - 127);
    });
}
void fill_scale(sycl::queue& q, uint16_t* p, size_t n, uint32_t seed) {
    const uint32_t s0 = hsh(seed * 0x9e3779b9u + 7);
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        const uint32_t h = hsh((uint32_t) i[0] ^ s0);
        const float v = 0.002f + (float) (h % 1000u) * 2e-5f;
        p[i] = sycl::bit_cast<uint16_t>(sycl::half(v));
    });
}
void fill_f32(sycl::queue& q, float* p, size_t n, uint32_t seed, float amp) {
    const uint32_t s0 = hsh(seed * 0x9e3779b9u + 13);
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        const uint32_t h = hsh((uint32_t) i[0] ^ s0);
        p[i] = ((float) (h & 0xffffffu) / 16777216.0f * 2.0f - 1.0f) * amp;
    });
}
template <typename T>
T* dalloc(sycl::queue& q, size_t n) {
    T* p = sycl::malloc_device<T>(n + 64, q);
    if (!p) { std::fprintf(stderr, "malloc_device failed (%zu)\n", n); std::exit(2); }
    q.memset(p, 0, (n + 64) * sizeof(T)).wait();
    return p;
}
template <typename T>
std::vector<T> fetch(sycl::queue& q, const T* d, size_t n) {
    std::vector<T> h(n);
    q.memcpy(h.data(), d, n * sizeof(T)).wait();
    return h;
}
uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct SlotLayer {   // one slot's QSA state of ONE layer
    int8_t *k_q = nullptr, *v_q = nullptr;
    uint16_t *k_s = nullptr, *v_s = nullptr;
    int32_t* pt = nullptr;
    float *pooled = nullptr, *dead = nullptr, *tail = nullptr;
    int32_t* bpos = nullptr;
    int64_t cells = 0, pages = 0;
    k::QsaAttnPools pools() const {
        k::QsaAttnPools p;
        p.k_q = k_q; p.v_q = v_q; p.k_scale = k_s; p.v_scale = v_s; p.page_table = pt;
        return p;
    }
    k::QsaIndexerBuffers ib() const { return k::QsaIndexerBuffers{tail, dead, pooled, bpos}; }
};

struct Shape {
    const char* name;
    int slots, per;   // rows = slots * per, contiguous per slot
};

struct Ctx {
    sycl::queue& q;
    k::QsaShapes s;
    int64_t ctx;
    int64_t cap, max_blocks;
    int rows_max = 8;
    std::vector<SlotLayer> sl;   // [L][2]
    // window buffers (shared by the layers, as the verifier's are)
    float *qcur, *qidx, *scores, *scratch, *attn, *idx_raw, *kcur, *vcur, *gamma;
    int32_t *sel, *steps;
    int64_t scratch_floats;
    SlotLayer& at(int l, int slot) { return sl[(size_t) l * 2 + slot]; }
};

void make_ctx(Ctx& c) {
    sycl::queue& q = c.q;
    const k::QsaShapes& s = c.s;
    const int64_t HD = s.head_dim, NKV = s.n_head_kv, G = HD / k::KV_Q8_GROUP;
    c.cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    c.max_blocks = kMaxCells / s.idx_block + 2;
    c.scratch_floats = (int64_t) k::qsa_decode_attn_scratch_floats(c.cap, s);
    c.sl.resize((size_t) L * 2);
    std::mt19937 rng(1234u + (uint32_t) c.ctx);
    for (int l = 0; l < L; ++l)
        for (int j = 0; j < 2; ++j) {
            SlotLayer& x = c.at(l, j);
            x.cells = ((c.ctx + 64 + 3) / 4) * 4;
            x.pages = x.cells / s.page_size;
            const size_t rows = (size_t) x.pages * NKV * s.page_size;
            x.k_q = dalloc<int8_t>(q, rows * HD);
            x.v_q = dalloc<int8_t>(q, rows * HD);
            x.k_s = dalloc<uint16_t>(q, rows * G);
            x.v_s = dalloc<uint16_t>(q, rows * G);
            const uint32_t seed = (uint32_t) (l * 2 + j + 1) * 101u + (uint32_t) c.ctx;
            fill_i8(q, x.k_q, rows * HD, seed);
            fill_i8(q, x.v_q, rows * HD, seed + 1);
            fill_scale(q, x.k_s, rows * G, seed + 2);
            fill_scale(q, x.v_s, rows * G, seed + 3);
            std::vector<int32_t> perm((size_t) x.pages);
            std::iota(perm.begin(), perm.end(), 0);
            std::shuffle(perm.begin(), perm.end(), rng);
            x.pt = dalloc<int32_t>(q, (size_t) x.pages);
            q.memcpy(x.pt, perm.data(), perm.size() * 4).wait();
            const size_t prow = (size_t) (x.cells / s.idx_block + 2);
            x.pooled = dalloc<float>(q, prow * s.idx_dim);
            x.dead = dalloc<float>(q, (size_t) s.idx_dim);
            x.tail = dalloc<float>(q, (size_t) (s.idx_block - 1) * s.idx_dim);
            x.bpos = dalloc<int32_t>(q, 1);
            fill_f32(q, x.pooled, prow * s.idx_dim, seed + 4, 1.0f);
            fill_f32(q, x.dead, (size_t) s.idx_dim, seed + 5, 1.0f);
            fill_f32(q, x.tail, (size_t) (s.idx_block - 1) * s.idx_dim, seed + 6, 1.0f);
        }
    const int R = c.rows_max;
    c.qcur = dalloc<float>(q, (size_t) R * s.n_head * HD);
    c.qidx = dalloc<float>(q, (size_t) R * s.idx_n_head * s.idx_dim);
    c.scores = dalloc<float>(q, (size_t) R * c.max_blocks);
    c.scratch = dalloc<float>(q, (size_t) R * c.scratch_floats);
    c.attn = dalloc<float>(q, (size_t) L * R * s.n_head * HD);
    c.sel = dalloc<int32_t>(q, (size_t) L * R * c.cap);
    c.steps = dalloc<int32_t>(q, (size_t) R * k::kStepCount);
    c.idx_raw = dalloc<float>(q, (size_t) R * s.idx_dim);
    c.kcur = dalloc<float>(q, (size_t) R * NKV * HD);
    c.vcur = dalloc<float>(q, (size_t) R * NKV * HD);
    c.gamma = dalloc<float>(q, (size_t) s.idx_dim);
    fill_f32(q, c.qcur, (size_t) R * s.n_head * HD, 77, 1.0f);
    fill_f32(q, c.qidx, (size_t) R * s.idx_n_head * s.idx_dim, 78, 1.0f);
    fill_f32(q, c.idx_raw, (size_t) R * s.idx_dim, 79, 1.0f);
    fill_f32(q, c.kcur, (size_t) R * NKV * HD, 80, 2.0f);
    fill_f32(q, c.vcur, (size_t) R * NKV * HD, 81, 2.0f);
    std::vector<float> gam((size_t) s.idx_dim);
    for (size_t i = 0; i < gam.size(); ++i) gam[i] = 0.75f + 0.5f * (float) ((i * 37) % 64) / 64.0f;
    q.memcpy(c.gamma, gam.data(), gam.size() * 4).wait();
}
void free_ctx(Ctx& c) {
    sycl::queue& q = c.q;
    for (auto& x : c.sl) {
        sycl::free(x.k_q, q); sycl::free(x.v_q, q); sycl::free(x.k_s, q); sycl::free(x.v_s, q); sycl::free(x.pt, q);
        sycl::free(x.pooled, q); sycl::free(x.dead, q); sycl::free(x.tail, q); sycl::free(x.bpos, q);
    }
    for (void* p : {(void*) c.qcur, (void*) c.qidx, (void*) c.scores, (void*) c.scratch, (void*) c.attn, (void*) c.sel,
                    (void*) c.steps, (void*) c.idx_raw, (void*) c.kcur, (void*) c.vcur, (void*) c.gamma})
        sycl::free(p, q);
    c.sl.clear();
}

// positions of a shape's rows: slot j's window starts at ctx - 8 j (slot B a little shorter)
void set_steps(Ctx& c, const Shape& sh) {
    std::vector<int32_t> h((size_t) sh.slots * sh.per * k::kStepCount);
    for (int j = 0; j < sh.slots; ++j)
        for (int r = 0; r < sh.per; ++r) {
            const int64_t pos = c.ctx - 8 * j + r;
            k::qsa_step_fill(h.data() + (size_t) (j * sh.per + r) * k::kStepCount, pos, c.s);
        }
    c.q.memcpy(c.steps, h.data(), h.size() * 4).wait();
}

enum Path { kRow = 0, kSlot = 1, kOne = 2 };
const char* path_name[3] = {"row", "slot", "one"};

// one layer's select + attention for the window, as verify.cpp's batch branch issues it
void attn_layer(Ctx& c, const Shape& sh, int l, Path p, int mask = 7) {   // mask: 1 scores, 2 top-k, 4 attention
    const k::QsaShapes& s = c.s;
    void* st = &c.q;
    const int64_t NH = s.n_head, HD = s.head_dim, IQ = s.idx_n_head, ID = s.idx_dim;
    const int n = sh.slots * sh.per;
    float* attn_l = c.attn + (size_t) l * c.rows_max * NH * HD;
    int32_t* sel_l = c.sel + (size_t) l * c.rows_max * c.cap;
    if (p == kRow) {
        for (int t = 0; t < n; ++t) {
            const SlotLayer& x = c.at(l, t / sh.per);
            if (mask & 1)
            k::qsa_block_scores(x.pooled, x.dead, c.qidx + t * IQ * ID, c.steps + t * k::kStepCount, 1, c.max_blocks, s,
                                c.scores + (size_t) t * c.max_blocks, st);
            if (mask & 2)
            k::qsa_block_topk(c.scores + (size_t) t * c.max_blocks, c.steps + t * k::kStepCount, 1, c.max_blocks, c.cap,
                              s, sel_l + (size_t) t * c.cap, st);
            if (mask & 4)
            k::qsa_decode_attn_batch(c.qcur + t * NH * HD, x.pools(), sel_l + (size_t) t * c.cap,
                                     c.steps + t * k::kStepCount, c.cap, s, c.scratch + (size_t) t * c.scratch_floats,
                                     attn_l + t * NH * HD, 1, st);
        }
        return;
    }
    for (int j = 0; j < sh.slots; ++j) {
        const int f = j * sh.per;
        const SlotLayer& x = c.at(l, j);
        if (mask & 1)
        k::qsa_block_scores(x.pooled, x.dead, c.qidx + f * IQ * ID, c.steps + f * k::kStepCount, sh.per, c.max_blocks,
                            s, c.scores + (size_t) f * c.max_blocks, st);
        if (p == kSlot && (mask & 2))
            k::qsa_block_topk(c.scores + (size_t) f * c.max_blocks, c.steps + f * k::kStepCount, sh.per, c.max_blocks,
                              c.cap, s, sel_l + (size_t) f * c.cap, st);
    }
    if (p == kOne) {
        if (mask & 2) k::qsa_block_topk(c.scores, c.steps, n, c.max_blocks, c.cap, s, sel_l, st);
        if (!(mask & 4)) return;
        k::QsaAttnPools pp[k::kQsaAttnMaxPools];
        int rows[k::kQsaAttnMaxPools];
        for (int j = 0; j < sh.slots; ++j) { pp[j] = c.at(l, j).pools(); rows[j] = sh.per; }
        k::qsa_decode_attn_multi(c.qcur, pp, rows, sh.slots, sel_l, c.steps, c.cap, s, c.scratch, attn_l, st);
        return;
    }
    if (!(mask & 4)) return;
    for (int j = 0; j < sh.slots; ++j) {
        const int f = j * sh.per;
        k::qsa_decode_attn_batch(c.qcur + f * NH * HD, c.at(l, j).pools(), sel_l + (size_t) f * c.cap,
                                 c.steps + f * k::kStepCount, c.cap, s, c.scratch + (size_t) f * c.scratch_floats,
                                 attn_l + f * NH * HD, sh.per, st);
    }
}

// one layer's K/V append and indexer append: row by row, or per slot
void prep_layer(Ctx& c, const Shape& sh, int l, bool per_slot) {
    const k::QsaShapes& s = c.s;
    void* st = &c.q;
    const int64_t NKV = s.n_head_kv, HD = s.head_dim, ID = s.idx_dim;
    const float eps = 1e-6f;
    if (!per_slot) {
        for (int t = 0; t < sh.slots * sh.per; ++t) {
            const SlotLayer& x = c.at(l, t / sh.per);
            k::kv_append_q8_step(x.k_q, x.v_q, x.k_s, x.v_s, x.pt, c.steps + t * k::kStepCount, c.kcur + t * NKV * HD,
                                 c.vcur + t * NKV * HD, s, st, nullptr);
        }
        for (int t = 0; t < sh.slots * sh.per; ++t) {
            const SlotLayer& x = c.at(l, t / sh.per);
            k::native_qsa_indexer_append(c.idx_raw + t * ID, c.steps + t * k::kStepCount + k::kStepPos, 0, c.gamma, eps,
                                         x.ib(), s, x.cells, k::rope_scaling(), st);
        }
        return;
    }
    for (int j = 0; j < sh.slots; ++j) {
        const int f = j * sh.per;
        const SlotLayer& x = c.at(l, j);
        k::kv_append_q8_steps(x.k_q, x.v_q, x.k_s, x.v_s, x.pt, c.steps + f * k::kStepCount, k::kStepCount,
                              c.kcur + f * NKV * HD, c.vcur + f * NKV * HD, (int) (NKV * HD), sh.per, s, st, nullptr);
    }
    for (int j = 0; j < sh.slots; ++j) {
        const int f = j * sh.per;
        const SlotLayer& x = c.at(l, j);
        k::native_qsa_indexer_append_steps(c.idx_raw + f * ID, c.steps + f * k::kStepCount + k::kStepPos, k::kStepCount,
                                           sh.per, 0, c.gamma, eps, x.ib(), s, x.cells, k::rope_scaling(), st);
    }
}

template <class F>
double time_graph(sycl::queue& q, int reps, F&& body) {
    sx::command_graph g{q.get_context(), q.get_device()};
    g.begin_recording(q);
    body();
    g.end_recording();
    auto ex = g.finalize();
    for (int i = 0; i < 3; ++i) q.ext_oneapi_graph(ex);
    q.wait();
    std::vector<double> v;
    for (int b = 0; b < 7; ++b) {
        const double t0 = now_us();
        for (int r = 0; r < reps; ++r) q.ext_oneapi_graph(ex);
        q.wait();
        v.push_back((now_us() - t0) / reps);
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

std::map<std::string, uint64_t> g_ref, g_out;
bool g_have_ref = false;
void record(const std::string& key, uint64_t h) {
    g_out[key] = h;
    if (g_have_ref) {
        auto it = g_ref.find(key);
        if (it == g_ref.end()) std::printf("  ref %-34s missing\n", key.c_str());
        else if (it->second != h) { std::printf("  ref %-34s DIFF (%016llx vs %016llx)\n", key.c_str(), (unsigned long long) h, (unsigned long long) it->second); ++g_fail; }
    }
}

void compare_f32(const char* what, const std::vector<float>& a, const std::vector<float>& b) {
    double mabs = 0, mrel = 0;
    size_t nd = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], 4) != 0) {
            ++nd;
            const double d = std::fabs((double) a[i] - (double) b[i]);
            mabs = std::max(mabs, d);
            mrel = std::max(mrel, d / std::max(1e-30, std::fabs((double) a[i])));
        }
    }
    if (nd) { std::printf("  %-40s %zu of %zu floats differ, max abs %.3g max rel %.3g\n", what, nd, a.size(), mabs, mrel); ++g_fail; }
    else std::printf("  %-40s bitwise (%zu floats)\n", what, a.size());
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<int64_t> ctxs = {1024, 16384, 65536};
    std::string ref_path;
    int reps = 20;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--ctx") && i + 1 < argc) {
            ctxs.clear();
            std::string a = argv[++i];
            size_t p = 0;
            while (p < a.size()) { size_t e = a.find(',', p); if (e == std::string::npos) e = a.size(); ctxs.push_back(std::atoll(a.substr(p, e - p).c_str())); p = e + 1; }
        } else if (!std::strcmp(argv[i], "--ref") && i + 1 < argc) ref_path = argv[++i];
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
    }
    if (!ref_path.empty()) {
        if (FILE* f = std::fopen(ref_path.c_str(), "r")) {
            char key[128]; unsigned long long h;
            while (std::fscanf(f, "%127s %llx", key, &h) == 2) g_ref[key] = h;
            std::fclose(f);
            g_have_ref = !g_ref.empty();
        }
    }
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order()}};
    std::printf("device: %s  ref: %s  env VPIPE=%d MERGE_ACTIVE=%d\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(),
                g_have_ref ? "compare" : (ref_path.empty() ? "none" : "write"), k::qsa_night_vpipe() ? 1 : 0,
                k::qsa_night_merge_active() ? 1 : 0);
    const Shape shapes[] = {{"C2_2x3", 2, 3}, {"C1_1x4", 1, 4}, {"C2d1_2x2", 2, 2}};
    for (int64_t ctx : ctxs) {
        Ctx c{q, k::qsa_real_shapes(), ctx, 0, 0};
        make_ctx(c);
        q.wait();
        std::printf("\n=== context %lld (cap %lld, max_blocks %lld, 12 layers) ===\n", (long long) ctx, (long long) c.cap,
                    (long long) c.max_blocks);
        for (const Shape& sh : shapes) {
            const int n = sh.slots * sh.per;
            set_steps(c, sh);
            const size_t attn_n = (size_t) L * c.rows_max * c.s.n_head * c.s.head_dim, sel_n = (size_t) L * c.rows_max * c.cap;
            std::vector<float> a_ref;
            std::vector<int32_t> s_ref;
            double t_us[3] = {0, 0, 0};
            for (int p = 0; p < 3; ++p) {
                if (p == kSlot && sh.slots == 1) {}   // C1: slot == one launch per kernel already (still run it)
                q.memset(c.attn, 0, attn_n * 4).wait();
                q.memset(c.sel, 0, sel_n * 4).wait();
                for (int l = 0; l < L; ++l) attn_layer(c, sh, l, (Path) p);
                q.wait();
                auto a = fetch(q, c.attn, attn_n);
                auto sv = fetch(q, c.sel, sel_n);
                char key[96];
                std::snprintf(key, sizeof key, "attn_%lld_%s", (long long) ctx, sh.name);
                record(std::string(key) + "_" + path_name[p], fnv(a.data(), a.size() * 4, fnv(sv.data(), sv.size() * 4)));
                if (p == kRow) { a_ref = a; s_ref = sv; }
                else {
                    char w[96];
                    std::snprintf(w, sizeof w, "%s %s vs row: attn", sh.name, path_name[p]);
                    compare_f32(w, a_ref, a);
                    if (std::memcmp(s_ref.data(), sv.data(), sv.size() * 4) != 0) { std::printf("  %s %s vs row: selection DIFFERS\n", sh.name, path_name[p]); ++g_fail; }
                }
                t_us[p] = time_graph(q, reps, [&] { for (int l = 0; l < L; ++l) attn_layer(c, sh, l, (Path) p); }) / L;
            }
            {   // component split (graph replay, 12 layers): scores / top-k / chunk+merge per path
                double comp[3][3];
                for (int p = 0; p < 3; ++p)
                    for (int m = 0; m < 3; ++m)
                        comp[p][m] = time_graph(q, reps, [&] { for (int l = 0; l < L; ++l) attn_layer(c, sh, l, (Path) p, 1 << m); }) / L;
                std::printf("  %-9s split us/layer (scores/topk/attn): row %.1f/%.1f/%.1f  slot %.1f/%.1f/%.1f  one %.1f/%.1f/%.1f\n",
                            sh.name, comp[0][0], comp[0][1], comp[0][2], comp[1][0], comp[1][1], comp[1][2], comp[2][0],
                            comp[2][1], comp[2][2]);
            }
            // the K/V + indexer appends: parity from one saved state, then time
            std::vector<std::vector<float>> tail0, pooled0;
            for (int j = 0; j < sh.slots; ++j) {
                const SlotLayer& x = c.at(0, j);
                tail0.push_back(fetch(q, x.tail, (size_t) (c.s.idx_block - 1) * c.s.idx_dim));
                pooled0.push_back(fetch(q, x.pooled, (size_t) (x.cells / c.s.idx_block + 2) * c.s.idx_dim));
            }
            auto snap = [&](int j) {
                const SlotLayer& x = c.at(0, j);
                const size_t rows = (size_t) x.pages * c.s.n_head_kv * c.s.page_size;
                std::vector<uint8_t> out;
                auto push = [&](const void* d, size_t bytes) { std::vector<uint8_t> h(bytes); q.memcpy(h.data(), d, bytes).wait(); out.insert(out.end(), h.begin(), h.end()); };
                push(x.k_q, rows * c.s.head_dim); push(x.v_q, rows * c.s.head_dim);
                push(x.k_s, rows * 8); push(x.v_s, rows * 8);
                push(x.tail, (size_t) (c.s.idx_block - 1) * c.s.idx_dim * 4);
                push(x.pooled, (size_t) (x.cells / c.s.idx_block + 2) * c.s.idx_dim * 4);
                push(x.dead, (size_t) c.s.idx_dim * 4); push(x.bpos, 4);
                return out;
            };
            auto restore = [&]() {
                for (int j = 0; j < sh.slots; ++j) {
                    const SlotLayer& x = c.at(0, j);
                    q.memcpy(x.tail, tail0[(size_t) j].data(), tail0[(size_t) j].size() * 4).wait();
                    q.memcpy(x.pooled, pooled0[(size_t) j].data(), pooled0[(size_t) j].size() * 4).wait();
                    q.memset(x.bpos, 0, 4).wait();
                }
            };
            double tp[2] = {0, 0};
            std::vector<std::vector<uint8_t>> st_row;
            for (int ps = 0; ps < 2; ++ps) {
                restore();
                prep_layer(c, sh, 0, ps == 1);
                q.wait();
                for (int j = 0; j < sh.slots; ++j) {
                    auto sn = snap(j);
                    if (ps == 0) st_row.push_back(sn);
                    else if (sn != st_row[(size_t) j]) { std::printf("  %s prep slot %d: per-slot state DIFFERS from per-row\n", sh.name, j); ++g_fail; }
                    char key[96];
                    std::snprintf(key, sizeof key, "prep_%lld_%s_s%d_%s", (long long) ctx, sh.name, j, ps ? "slot" : "row");
                    record(key, fnv(sn.data(), sn.size()));
                }
                tp[ps] = time_graph(q, reps, [&] { for (int l = 0; l < L; ++l) prep_layer(c, sh, l, ps == 1); }) / L;
            }
            restore();
            std::printf("  %-9s rows %d | attn us/layer: row %7.1f  slot %7.1f  one %7.1f | prep us/layer: row %5.1f slot %5.1f | launches/layer row %d slot %d one %d\n",
                        sh.name, n, t_us[0], t_us[1], t_us[2], tp[0], tp[1], 4 * n, 4 * sh.slots, sh.slots + 3);
            std::printf("  RESULT ctx=%lld shape=%s rows=%d attn_row=%.2f attn_slot=%.2f attn_one=%.2f prep_row=%.2f prep_slot=%.2f\n",
                        (long long) ctx, sh.name, n, t_us[0], t_us[1], t_us[2], tp[0], tp[1]);
        }
        // the block scores alone for 5..8 queries of one slot (the verify shapes above reach 4 per slot)
        for (int nq = 5; nq <= 8; ++nq) {
            const Shape sh{"S1", 1, nq};
            set_steps(c, sh);
            q.memset(c.scores, 0, (size_t) c.rows_max * c.max_blocks * 4).wait();
            const SlotLayer& x = c.at(0, 0);
            k::qsa_block_scores(x.pooled, x.dead, c.qidx, c.steps, nq, c.max_blocks, c.s, c.scores, &q);
            q.wait();
            auto sc = fetch(q, c.scores, (size_t) nq * c.max_blocks);
            char key[64];
            std::snprintf(key, sizeof key, "scores%d_%lld", nq, (long long) ctx);
            record(key, fnv(sc.data(), sc.size() * 4));
        }
        // the top-k alone on tie-heavy scores (8 distinct values): 6 queries, the C2 rows' steps
        {
            const Shape sh{"C2_2x3", 2, 3};
            set_steps(c, sh);
            const uint32_t s0 = 0x51ed27u + (uint32_t) ctx;
            float* sc = c.scores;
            q.parallel_for(sycl::range<1>((size_t) 6 * c.max_blocks), [=](sycl::id<1> i) {
                sc[i] = (float) (hsh((uint32_t) i[0] ^ s0) % 8u) * 0.25f;
            }).wait();
            k::qsa_block_topk(c.scores, c.steps, 6, c.max_blocks, c.cap, c.s, c.sel, &q);
            q.wait();
            auto ids = fetch(q, c.sel, (size_t) 6 * c.cap);
            char key[64];
            std::snprintf(key, sizeof key, "topk_ties_%lld", (long long) ctx);
            record(key, fnv(ids.data(), ids.size() * 4));
            const double t1 = time_graph(q, reps, [&] { for (int l = 0; l < L; ++l) k::qsa_block_topk(c.scores, c.steps, 1, c.max_blocks, c.cap, c.s, c.sel, &q); }) / L;
            const double t6 = time_graph(q, reps, [&] { for (int l = 0; l < L; ++l) k::qsa_block_topk(c.scores, c.steps, 6, c.max_blocks, c.cap, c.s, c.sel, &q); }) / L;
            std::printf("  top-k (ties) us/launch: 1 query %.1f, 6 queries %.1f\n", t1, t6);
            std::printf("  RESULT ctx=%lld topk1=%.2f topk6=%.2f\n", (long long) ctx, t1, t6);
        }
        // the MTP drafter's call: 1 row, 32K window = 512 chunks, ids = the window's cells
        {
            const int64_t capm = 32768;
            const int64_t sfl = (int64_t) k::qsa_decode_attn_scratch_floats(capm, c.s);
            float* scr = dalloc<float>(q, (size_t) sfl);
            int32_t* idm = dalloc<int32_t>(q, (size_t) capm * 4);
            int32_t* stm = dalloc<int32_t>(q, (size_t) k::kStepCount * 4);
            float* outm = dalloc<float>(q, (size_t) 4 * c.s.n_head * c.s.head_dim);
            const int64_t nkv = ctx + 1, W = std::min<int64_t>(nkv, capm);
            std::vector<int32_t> hid((size_t) capm * 4, 0), hst((size_t) k::kStepCount * 4);
            for (int u = 0; u < 4; ++u) {
                for (int64_t i = 0; i < W; ++i) hid[(size_t) (u * capm + i)] = (int32_t) (nkv - W + i);
                int32_t* sp = hst.data() + u * k::kStepCount;
                sp[k::kStepPos] = (int32_t) (nkv - 1); sp[k::kStepNKv] = (int32_t) nkv;
                sp[k::kStepNBid] = (int32_t) (nkv / c.s.idx_block); sp[k::kStepWidth] = (int32_t) W;
            }
            q.memcpy(idm, hid.data(), hid.size() * 4).wait();
            q.memcpy(stm, hst.data(), hst.size() * 4).wait();
            auto body = [&] {
                for (int u = 0; u < 4; ++u)   // four replays' worth (C2: 2 depths x 2 owners), each on its own layer pools
                    k::qsa_decode_attn_batch(c.qcur, c.at(u, 0).pools(), idm + (size_t) u * capm, stm + u * k::kStepCount,
                                             capm, c.s, scr, outm + (size_t) u * c.s.n_head * c.s.head_dim, 1, &q);
            };
            body();
            q.wait();
            auto om = fetch(q, outm, (size_t) 4 * c.s.n_head * c.s.head_dim);
            char key[64];
            std::snprintf(key, sizeof key, "mtp_%lld", (long long) ctx);
            record(key, fnv(om.data(), om.size() * 4));
            const double tm = time_graph(q, reps, body) / 4;
            std::printf("  MTP 1 row, 512 chunks, window %lld: %.1f us/call\n", (long long) W, tm);
            std::printf("  RESULT ctx=%lld mtp_call=%.2f\n", (long long) ctx, tm);
            sycl::free(scr, q); sycl::free(idm, q); sycl::free(stm, q); sycl::free(outm, q);
        }
        free_ctx(c);
    }
    if (!ref_path.empty() && !g_have_ref) {
        if (FILE* f = std::fopen(ref_path.c_str(), "w")) {
            for (auto& kv : g_out) std::fprintf(f, "%s %016llx\n", kv.first.c_str(), (unsigned long long) kv.second);
            std::fclose(f);
            std::printf("\nwrote %zu reference hashes to %s\n", g_out.size(), ref_path.c_str());
        }
    }
    std::printf("\n%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
