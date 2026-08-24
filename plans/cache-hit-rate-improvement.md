# Plan: Raise Provider Prefix Cache Hit Rate from ~85% to ~95%+

**Goal**: Minimize provider-side KV-cache misses by ensuring the request prefix stays byte-identical across turns, and by caching converted request bytes to avoid re-serialization.

**Current state**: ~85% cache hit rate. Main cause: compaction replaces the ENTIRE content window with a summary generated from a different system prefix, causing a 100% cold miss of 512K+ tokens in one shot. Between compaction events, the prefix is stable but re-serialized from scratch every turn.

**Target state**: ~95%+ hit rate by making compaction cache-friendly (summary with same prefix, mechanical fold) and caching request bytes between turns.

**Reference**: opencode achieves ~99% via two-layer compaction (summary out-of-band + 0-LLM-token fold), 4-slot system, checkpoint prefix reuse, and runtime stability detectors. DeepSeek handles prefix caching server-side (no `cache_control` markers needed).

---

## Root Cause Analysis

| Cause | Impact | Avoidable? |
|-------|--------|------------|
| **Compaction: summary with different prefix** — generates summary using different system prompt → new prefix → full cold miss of ENTIRE context window (512K+ tokens) | **~15% miss per fold, but 100% of tokens lost per miss** | Yes — T8: two-layer compaction |
| **No request byte cache** — every turn re-serializes Messages+Tools from scratch | ~1-2% micro-differences | Yes — T1: checkpoint reuse |
| **Single-block system prompt** — any change invalidates entire prefix | Low impact today, fragile long-term | Yes — T7: slot splitting |
| **Extension interceptors** — may rewrite each turn | Unknown drift | Yes — T4: stability guard |

**Key insight from opencode** (`docs/compaction.md`):
- Summary generated with **same prefix** as normal turns → cache hit
- Summary stored in DB **outside content window** → never pollutes prefix
- Compact = `m* = [s1,s2,...recent m]` — **0 LLM tokens**, pure mechanical fold
- opencode loses at most 64K tokens (recent tail reorder), DeepSeek-Reasonix loses 512K+ (full window)

---

## Task 8: Two-Layer Compaction (Priority: HIGHEST)

**What**: Separate summary generation from compact fold. Summaries are generated out-of-band with the same prefix as normal turns (cache hit), stored outside the content window, and compacted mechanically (0 LLM tokens).

**Files**:
- `internal/agent/compact_projection.go` — rewrite to two-layer architecture
- `internal/agent/compact_commit.go` — summary storage separate from content window
- `internal/agent/compact.go` — cadence logic (summary at ~65K tokens, compact at usable(model))
- `internal/agent/session.go` — summary store (DB-like, outside content)
- `internal/agent/preflight.go` — content window = real messages only, summaries outside

**Design** (following `docs/compaction.md`):

### Layer-1: Summary (out-of-band)
```
1. Normal turn finishes (all tool/reasoning inference done)
2. Checkpoint M (save exact visible messages)
3. Request summary via user-message shape (ephemeral stream)
   — SAME prefix as normal turns → cache HIT
4. Store s in summary store (outside content window)
5. Restore prior M (content window unchanged)
6. Continue work
```

### Layer-2: Compact (mechanical fold, 0 LLM tokens)
```
When open content ≥ usable(model) — NOT 80%:
  compact() — ZERO LLM tokens
  m* = [s1,s2,...(≤32K tokens), recent m,m,m]
  — Pure system fold, no LLM call → cache HIT on prefix
```

### Key parameters
| Parameter | Value | Source |
|-----------|-------|--------|
| Summary cadence | ~65K tokens (256K chars) | opencode `SUMMARY_INTERVAL_TOKENS` |
| Compact trigger | `usable(model)` = context − headroom | NOT 80% × context_window |
| Summary cap in m* | 32K tokens | opencode `MAX_SUMMARY_BODY_TOKENS` |
| Recent tail floor | ~32K tokens | opencode `RECENT_MIN_TOKENS` |
| Compact cost | **0 LLM tokens** | Mechanical fold only |

### Why this works for cache
1. Summary generation uses **same prefix** as normal turns → DeepSeek cache hits on summary request
2. Summary stored outside content window → never pollutes prefix between turns
3. Compact is mechanical: `[s1,s2,recent m]` → prefix after compact = [system + s1 + s2 + recent tail]
4. First post-compact request: cold miss on message prefix (new), but **only ~64K tokens lost** (not 512K+)
5. Subsequent turns after compact: prefix stable → cache hits

**Expected impact**: **BIGGEST improvement** — reduces cold miss from 512K+ tokens to ~64K tokens per fold. This alone should push hit rate from ~85% to ~93%+.

---

## Task 1: Request Byte Cache (Checkpoint Prefix Reuse)

**What**: After each successful provider response, store the frozen `samplingRequest` (messages, tools). On the next turn, detect the longest common prefix with the stored request and reuse those bytes directly — only converting/appending the new suffix.

**Files**:
- `internal/agent/request_cache.go` (new) — `RequestByteCache` type
- `internal/agent/run_loop.go` — integrate cache into `prepareSamplingRequest`
- `internal/agent/sampling_request.go` — `buildSamplingRequest` returns prefix reuse info
- `internal/agent/session.go` — persist cache alongside compaction state

**Design**:
```go
type RequestByteCache struct {
    Messages       []provider.Message  // frozen messages from last successful request
    Tools          []provider.ToolSchema
    SystemHash     string              // SHA-256 of system prompt
    PromptCacheKey string              // lineage guard
}
```

- After successful `Stream`, store `frozen.req.Messages` + `frozen.req.Tools` in cache
- On next `buildSamplingRequest`, compare new messages against cached prefix
- Reuse identical prefix bytes (zero-copy slice), only convert new suffix
- Invalidate on: compaction, rewind, model change, session switch

**Why**: Eliminates re-serialization differences between turns. The `normalizeModelRequestMessages` path (zeroing CreatedAt, stripping execution policy) runs once and is cached.

**Expected impact**: Eliminates ~1-2% of non-compaction misses from re-serialization edge cases.

---

## Task 2: Compaction Cache Alignment

**What**: After compaction installs a new projection, seed the request byte cache with the new projection messages so subsequent turns immediately hit.

**Files**:
- `internal/agent/compact_commit.go` — after projection install, seed `RequestByteCache`
- `internal/agent/compact_projection.go` — `CompressContext` also seeds cache

**Design**:
- After `commitSummaryProjection` or `commitPruneProjection`, snapshot new projection as cached prefix
- Subsequent turns after compaction reuse projection bytes → stable prefix → hits

**Expected impact**: Post-compaction turns hit immediately instead of re-serializing.

---

## Task 3: Compaction Diagnostics Visibility

**What**: Wire compaction events into `PrefixChangeReasonCounts` so cache misses are attributable.

**Files**:
- `internal/agent/compact_commit.go` — call `NoteContentRewrite("compact_auto")` after projection install
- `internal/agent/prune.go` — call `NoteContentRewrite("prune")` after tool-result pruning

**Design**:
- After any provider-visible projection install, queue a content rewrite reason
- `CompareShape` reports `"compact_auto"` or `"prune"` in `PrefixChangeReasonCounts`
- No behavior change — purely diagnostic

**Expected impact**: 0% cache improvement, but enables measuring whether T1/T2/T8 work.

---

## Task 4: Extension Interceptor Stability Guard

**What**: Detect when `interceptContextPrepare` or `interceptProviderRequest` rewrites the request in a way that breaks prefix stability.

**Files**:
- `internal/agent/extensions.go` — add fingerprint comparison
- `internal/agent/run_loop.go` — log warning on prefix-breaking interceptor

**Design**:
- Before interceptors: compute `SHA-256(normalizedMessages)`
- After interceptors: recompute and compare
- If hash changed and no compaction happened → log `"extension broke prefix stability"`
- Guard: only runs in debug/metrics mode, zero-cost in production

**Expected impact**: 0% cache improvement, but identifies hidden miss sources.

---

## Task 5: Tool Schema Stability Guard

**What**: Detect mid-session tool schema changes that break the tools prefix.

**Files**:
- `internal/agent/run_loop.go` — add `checkToolStability`
- `internal/tool/tool.go` — expose schema fingerprint

**Design**:
- After `a.svc.tools.Schemas()`, compute hash of sorted (name+description+schema)
- Compare against previous turn's hash
- If changed → log `"tools broke prefix stability: [added=X, removed=Y, changed=Z]"`

**Expected impact**: 0% cache improvement, but detects MCP reconnect/tool-add misses.

---

## Task 6: End-to-End Cache Hit Rate Measurement

**What**: Add per-turn cache hit rate to metrics output.

**Files**:
- `internal/event/event.go` — ensure `CacheDiagnostics` includes hit/miss tokens
- `internal/cli/run_metrics.go` — display cache hit ratio in status line
- `internal/agent/cachehit_e2e_test.go` — extend with post-compaction assertion

**Design**:
- Compute `ratio = hit / (hit + miss)` per turn
- Display in status line: `cache: 95.2% (turn 12)`
- Add test: 10-turn session with 1 compaction → assert ratio ≥ 90%

**Expected impact**: 0% cache improvement, but validates all other tasks.

---

## Task 7: System Prompt Slot Splitting (Future-Proofing)

**What**: Split the monolithic system prompt into cache-granular slots.

**Files**:
- `internal/boot/boot.go` — split `ResolveSystemPromptForRoot` output into slots
- `internal/agent/agent.go` — `systemPrompt()` returns slots
- `internal/provider/provider.go` — `Request.System` stays as single string, slots internal

**Design** (following `docs/system-prompt-order.md`):
```
[0] UNIVERSAL_ENV    — ~500B, immutable: role + handoff + epistemic stance
[1] tool schemas     — stable per app version
[2] stable body      — reasoning → kernel → workspace → env → policies
[3] mutable tail     — session banner, extensions, dynamic content — ALWAYS LAST
```

**Expected impact**: 0% immediate improvement (system already stable), but prevents future degradation.

---

## Implementation Order

| Priority | Task | Time | Impact |
|----------|------|------|--------|
| **1** | **T8** (two-layer compaction) | **4-6h** | **~512K→64K cold miss reduction** |
| 2 | T3 (diagnostics) | 30 min | measurement foundation |
| 3 | T6 (measurement) | 30 min | proves improvement |
| 4 | T1 (request byte cache) | 2-3h | eliminates re-serialization misses |
| 5 | T2 (compaction alignment) | 1h | post-compaction cache hits |
| 6 | T5 (tool stability guard) | 30 min | detection |
| 7 | T4 (extension stability guard) | 30 min | detection |
| 8 | T7 (system prompt slots) | 2-3h | future-proofing (deferred) |

**Total**: ~11-14 hours (or ~8-10h without T7)

---

## Smoke Tests

- Existing `cachehit_e2e_test.go` must pass (prefix byte-identical assertions)
- New test: 10-turn session with 2-layer compact at turn 6 → assert cache ratio ≥ 93%
- New test: request byte cache hit on turn 3 after cache seeded at turn 2
- New test: summary generation uses same prefix as normal turns (compare SHA-256)
- New test: compact is 0 LLM tokens (no provider.Stream call during fold)
- Existing `TestCacheHitPrefixStable` must still pass after T7

## Risk

- **Low**: Tasks 3-6 are diagnostic only, no behavior change
- **Medium**: Task 1 (request byte cache) adds memory pressure — mitigated by evicting after compaction
- **High**: Task 8 (two-layer compaction) is major architectural change — requires careful testing of summary quality, compact correctness, and cache behavior
- **Medium**: Task 7 (slot splitting) changes system prompt assembly — must preserve byte-identical output

## Out of Scope

- Explicit `cache_control` markers (DeepSeek handles this server-side — see `anthropic.go:424-425`)
- Compaction threshold tuning (separate concern, user preference)
- Checkpoint persistence to disk (in-memory only for now)
