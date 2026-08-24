# Master Plan: Cache Hit Rate Improvement

**plan_id**: `cache-hit-rate-v1`
**revision**: 2
**created_by**: plan_mode
**state**: DRAFT

## Description

Raise provider prefix cache hit rate from ~85% to ~95%+ by implementing two-layer compaction (summary out-of-band + 0-LLM-token fold) and caching converted request bytes between turns.

## Premises (Claim Ledger)

| ID | Text | Status | Provenance |
|----|------|--------|------------|
| C1 | DeepSeek handles prefix caching server-side; no explicit `cache_control` markers needed | Exact | `anthropic.go:424-425` — "DeepSeek ignores cache_control and manages prefix caching automatically" |
| C2 | Current compaction generates summary with DIFFERENT system prefix → full cold miss of ENTIRE context window (512K+ tokens) | Inferred | `compact_projection.go` — summary request has different prefix than normal turns |
| C3 | Between compaction events, the prefix is byte-identical (zero-copy fast path, CreatedAt zeroed, tools name-sorted) | Exact | `projection.go:27-29` zero-copy fast path; `sampling_request.go:33-38` CreatedAt zeroed |
| C4 | `interceptContextPrepare`/`interceptProviderRequest` may rewrite request each turn | Exact | `sampling_request.go:136-150` — extensions mutate request copy |
| C5 | Tool schemas are name-sorted at registration, stable between turns unless MCP reconnects | Exact | `tool.go:377,632` canonicalized at registration |
| C6 | Current e2e tests assert prefix byte-identical replay with 90% threshold | Exact | `cachehit_e2e_test.go:358` |
| C7 | opencode's two-layer compaction: summary with same prefix (cache hit) + compact at usable(model) (0 LLM tokens) | Exact | `docs/compaction.md` — Layer-1 summary out-of-band, Layer-2 compact mechanical fold |
| C8 | opencode compact loses max 64K tokens (recent tail reorder), DeepSeek-Reasonix loses 512K+ (full window) | Inferred | C7 + C2 comparison |

## Open Questions

- None — all architectural questions resolved during GATE_1_GROUND + user clarification.

## Goals

### T8: Two-Layer Compaction (HIGHEST PRIORITY)
- **what**: Separate summary generation from compact fold. Summaries out-of-band with same prefix (cache hit), stored outside content window, compacted mechanically (0 LLM tokens)
- **files**: `internal/agent/compact_projection.go`, `internal/agent/compact_commit.go`, `internal/agent/compact.go`, `internal/agent/session.go`, `internal/agent/preflight.go`
- **depends_on_claims**: [C2, C7, C8]
- **oracle**: `go test ./internal/agent/ -run TestTwoLayerCompaction -v`
- **status**: `[ ]`

### T1: Request Byte Cache (Checkpoint Prefix Reuse)
- **what**: Cache frozen `samplingRequest` between turns; reuse longest common prefix bytes
- **files**: `internal/agent/request_cache.go` (new), `internal/agent/run_loop.go`, `internal/agent/sampling_request.go`, `internal/agent/session.go`
- **depends_on_claims**: [C3]
- **oracle**: `go test ./internal/agent/ -run TestRequestByteCache -v` + existing `TestCacheHitPrefixStable`
- **status**: `[ ]`

### T2: Compaction Cache Alignment
- **what**: Seed request byte cache after compaction install; post-compaction requests reuse projection bytes
- **files**: `internal/agent/compact_commit.go`, `internal/agent/compact_projection.go`
- **depends_on_claims**: [C1, C2]
- **oracle**: `go test ./internal/agent/ -run TestCompactionCacheAlignment -v`
- **status**: `[ ]`

### T3: Compaction Diagnostics Visibility
- **what**: Wire compaction/prune events into `PrefixChangeReasonCounts`
- **files**: `internal/agent/compact_commit.go`, `internal/agent/prune.go`
- **depends_on_claims**: [C2]
- **oracle**: `go test ./internal/agent/ -run TestCompareShapeCompactionReason -v`
- **status**: `[ ]`

### T4: Extension Interceptor Stability Guard
- **what**: Detect when interceptors break prefix stability
- **files**: `internal/agent/extensions.go`, `internal/agent/run_loop.go`
- **depends_on_claims**: [C4]
- **oracle**: `go test ./internal/agent/ -run TestExtensionStabilityGuard -v`
- **status**: `[ ]`

### T5: Tool Schema Stability Guard
- **what**: Detect mid-session tool schema changes
- **files**: `internal/agent/run_loop.go`, `internal/tool/tool.go`
- **depends_on_claims**: [C5]
- **oracle**: `go test ./internal/agent/ -run TestToolSchemaStability -v`
- **status**: `[ ]`

### T6: End-to-End Cache Hit Rate Measurement
- **what**: Per-turn cache hit ratio in status line + e2e assertion
- **files**: `internal/cli/run_metrics.go`, `internal/agent/cachehit_e2e_test.go`
- **depends_on_claims**: [C1, C6]
- **oracle**: `go test ./internal/agent/ -run TestCacheHitRateE2E -v`
- **status**: `[ ]`

### T7: System Prompt Slot Splitting (Future-Proofing)
- **what**: Split monolithic system prompt into cache-granular slots
- **files**: `internal/boot/boot.go`, `internal/agent/agent.go`, `internal/provider/provider.go`
- **depends_on_claims**: [C1]
- **oracle**: `go test ./internal/agent/ -run TestSystemPromptSlots -v`
- **status**: `[ ]`

## Execution Order

```
T8 (two-layer compaction) → T3 (diagnostics) → T6 (measurement) → T1 (request cache) → T2 (compaction alignment) → T5 (tool guard) → T4 (extension guard) → T7 (slot splitting, deferred)
```

## Smoke Contract

```yaml
smoke_na: false
baseline:
  - label: existing tests pass
    cmd: go test ./internal/agent/ -count=1
    expected_exit: 0
  - label: cache e2e tests pass
    cmd: go test ./internal/agent/ -run TestCacheHit -v -count=1
    expected_exit: 0
post_checks:
  - label: new cache tests pass
    cmd: go test ./internal/agent/ -run "TestTwoLayerCompaction|TestRequestByteCache|TestCompactionCacheAlignment|TestCacheHitRateE2E" -v -count=1
    expected_exit: 0
  - label: summary uses same prefix as normal turns
    cmd: go test ./internal/agent/ -run TestSummaryPrefixStable -v -count=1
    expected_exit: 0
  - label: compact is 0 LLM tokens
    cmd: go test ./internal/agent/ -run TestCompactZeroLLMTokens -v -count=1
    expected_exit: 0
blast_radius: agent compaction, session storage, request assembly, event diagnostics
```
