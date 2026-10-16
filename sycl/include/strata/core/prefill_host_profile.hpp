#pragma once
// Separate opt-in CPU-only prefill markers. No GPU API or per-layer output.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
namespace strata::core::prefill_host {
inline bool enabled() {
    static const bool on = [] { const char* p = std::getenv("STRATA_PREFILL_HOST_BOUNDARY"); return p && p[0] == '1'; }();
    return on;
}
struct Record {
    int64_t ns; uint64_t request, chunk; const char* tag;
    int64_t pos, tokens, layer, local_rows, peer_rows, peer_experts;
};
struct State {
    std::vector<Record> records;
    uint64_t next = 0, request = 0, chunk = 0;
    int64_t prompt = 0, pos = -1, tokens = 0, layer = -1, local_rows = 0, peer_rows = 0, peer_experts = 0;
    size_t dropped = 0;
    bool owner = false, active = false, failed = false, closed = false, complete = false, async = false;
};
inline State& state() { thread_local State s; return s; }
inline void mark(const char* tag) noexcept {
    if (!enabled()) return;
    State& s = state();
    if (!s.active) return;
    if (s.failed || s.records.size() >= 8192) { ++s.dropped; return; }
    try {
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        s.records.push_back({ns, s.request, s.chunk, tag, s.pos, s.tokens, s.layer,
                             s.local_rows, s.peer_rows, s.peer_experts});
    } catch (...) { s.failed = true; ++s.dropped; }
}
inline void chunk(int64_t pos, int64_t tokens) noexcept {
    if (!enabled() || !state().active) return;
    State& s = state(); ++s.chunk; s.pos = pos; s.tokens = tokens;
    s.layer = -1; s.local_rows = s.peer_rows = s.peer_experts = 0;
    mark("chunk_begin");
}
inline void layer(int64_t id, int64_t local, int64_t peer, int64_t experts) noexcept {
    if (!enabled() || !state().active) return;
    State& s = state(); s.layer = id; s.local_rows = local; s.peer_rows = peer; s.peer_experts = experts;
}
class Scope {
    bool owns_ = false;
public:
    Scope(int64_t prompt, bool eligible) noexcept {
        if (!enabled() || !eligible) return;
        State& s = state();
        if (s.owner) { ++s.dropped; return; }
        if (s.next != 0) return;  // only the first eligible fresh admission is recorded
        owns_ = true; s.owner = true; s.request = ++s.next; s.chunk = 0; s.prompt = prompt;
        s.pos = -1; s.tokens = 0; s.layer = -1; s.local_rows = s.peer_rows = s.peer_experts = 0;
        s.records.clear(); s.dropped = 0; s.failed = false; s.closed = false; s.complete = false;
        const char* a = std::getenv("STRATA_EP_PREFILL_ASYNC"); s.async = a && a[0] == '1';
        s.active = !s.async;
        try { if (s.records.capacity() == 0) s.records.reserve(8192); }
        catch (...) { s.failed = true; ++s.dropped; }
        mark("request_begin");
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    void finish(bool complete) noexcept {
        if (!owns_) return;
        State& s = state(); s.layer = -1; s.local_rows = s.peer_rows = s.peer_experts = 0;
        mark("prefill_end"); s.active = false; s.closed = true; s.complete = complete;
    }
    ~Scope() noexcept {
        if (!owns_) return;
        State& s = state();
        const char* status = s.async ? "skipped_async" : !s.closed ? "aborted" : s.complete ? "complete" : "incomplete";
        std::fprintf(stderr, "strata prefill host: begin req=%llu prompt=%lld records=%zu dropped=%zu failed=%d status=%s clock=steady_ns\n",
                     (unsigned long long)s.request, (long long)s.prompt, s.records.size(), s.dropped, (int)s.failed, status);
        for (const auto& r : s.records)
            std::fprintf(stderr, "strata prefill host: req=%llu chunk=%llu t=%lld tag=%s pos=%lld T=%lld layer=%lld local=%lld peer=%lld experts=%lld\n",
                         (unsigned long long)r.request, (unsigned long long)r.chunk, (long long)r.ns, r.tag,
                         (long long)r.pos, (long long)r.tokens, (long long)r.layer, (long long)r.local_rows,
                         (long long)r.peer_rows, (long long)r.peer_experts);
        std::fprintf(stderr, "strata prefill host: end req=%llu records=%zu dropped=%zu\n",
                     (unsigned long long)s.request, s.records.size(), s.dropped);
        std::fflush(stderr);
        s.records.clear(); s.owner = s.active = false;
    }
};
} // namespace strata::core::prefill_host
