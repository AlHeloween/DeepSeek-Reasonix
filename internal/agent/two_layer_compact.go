package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	"reasonix/internal/event"
	"reasonix/internal/provider"
)

// Two-layer compaction constants. Layer-1 generates summaries out-of-band
// (same prefix as normal turns → cache hit). Layer-2 mechanically folds
// stored summaries + recent tail (0 LLM tokens → cache hit on prefix).
const (
	summaryCadenceTokens = 65_536 // ~256K chars; generate out-of-band summary every ~65K tokens of new content
	maxStoredSummaries   = 8      // cap on out-of-band summaries in projection
	maxSummaryBodyTokens = 32_768 // cap on total summary body text in projection (per opencode)
	recentTailMinTokens  = 32_768 // minimum recent tail tokens after compact
)

// StoredSummary is one out-of-band summary generated with the same prefix
// as normal turns (cache hit). Stored outside the content window.
type StoredSummary struct {
	Text       string `json:"text"`
	Tokens     int    `json:"tokens"`      // estimated token count
	CoveredMsg int    `json:"covered_msg"` // canonical message index this summary covers up to
	Hash       string `json:"hash"`        // SHA-256 of text for dedup
}

// StoredSummaries is the out-of-band summary store, persisted in CompactionState.
type StoredSummaries struct {
	Summaries   []StoredSummary `json:"summaries"`
	TotalTokens int             `json:"total_tokens"` // sum of all summary body tokens
	Generation  uint64          `json:"generation"`   // bumped on each store
}

// twoLayerSummaryCadence returns the token threshold for triggering
// out-of-band summary generation. This is separate from the compaction trigger.
func (a *Agent) twoLayerSummaryCadence() int {
	window := a.effectiveContextWindow()
	if window <= 0 {
		return summaryCadenceTokens
	}
	// Cadence is ~12.5% of window, capped at summaryCadenceTokens
	return min(summaryCadenceTokens, max(1024, window/8))
}

// maybeGenerateOutOfBandSummary checks if enough new content has accumulated
// since the last out-of-band summary, and if so, generates one with the same
// prefix as normal turns (cache hit).
func (a *Agent) maybeGenerateOutOfBandSummary(ctx context.Context) error {
	a.sess.compactionMu.Lock()
	st := a.sess.compactionState
	a.sess.compactionMu.Unlock()

	canonical, _ := a.sess.conversation.snapshotMessagesVersion()
	cacheKey := a.currentPromptCacheKey()

	// Get the visible view (projection + tail or full canonical)
	visible := canonical
	if projectionValid(st, canonical, cacheKey) {
		if projected := modelVisibleFromProjection(st.Projection, canonical); len(projected) > 0 {
			visible = projected
		}
	}

	// Cadence gate: single pass over the canonical range the new content occupies.
	lastSummaryCovered := 0
	if st.StoredSummaries != nil && len(st.StoredSummaries.Summaries) > 0 {
		last := st.StoredSummaries.Summaries[len(st.StoredSummaries.Summaries)-1]
		lastSummaryCovered = last.CoveredMsg
	}
	if lastSummaryCovered >= len(canonical) {
		return nil
	}
	newTokens := estimateMessagesTokens(modelInputMessages(canonical[lastSummaryCovered:]))
	if newTokens < a.twoLayerSummaryCadence() {
		return nil // not enough new content
	}

	head := 0
	if len(visible) > 0 && visible[0].Role == provider.RoleSystem {
		head = 1
	}

	// Layer-1 rides the LIVE window: the summarizer request is the ordinary
	// visible prefix byte-for-byte (system + every message, recent tail included)
	// plus the trailing instruction — alignment holds by construction, matching
	// the append-instruction/rollback shape used elsewhere without mutating the
	// shared transcript. summarize() emits usage telemetry itself.
	instructions := ""
	var summary string
	var err error
	for attempt := 0; attempt < 2; attempt++ {
		summary, _, err = a.summarize(ctx, visible[head:], instructions)
		if err != nil {
			return fmt.Errorf("out-of-band summary: %w", err)
		}
		if summary == "" {
			return nil
		}
		missing := missingSummarySections(summary)
		if len(missing) == 0 {
			break
		}
		if attempt == 0 {
			// One corrective retry: name the gaps and demand all headings.
			instructions = "Your previous briefing omitted required sections: " +
				strings.Join(missing, ", ") +
				". Re-output ALL seven headings exactly; write \"(none)\" under a section that genuinely has no content."
			continue
		}
		// Persist failure: store nothing, keep cadence counter armed so the
		// next Prepare retries instead of silently accepting a broken digest.
		return fmt.Errorf("out-of-band summary incomplete after retry; missing: %s", strings.Join(missing, ", "))
	}

	stored := StoredSummary{
		Text: summary + fmt.Sprintf("\n\n<summary-archive from=\"%d\" to=\"%d\"/>", lastSummaryCovered, len(canonical)),
		Tokens:     estimateTextTokens(summary),
		CoveredMsg: len(canonical), // captured against the full live window
		Hash:       hex.EncodeToString(sha256Sum8(summary)),
	}

	a.sess.compactionMu.Lock()
	defer a.sess.compactionMu.Unlock()

	// Re-check state hasn't changed
	current, currentVersion := a.sess.conversation.snapshotMessagesVersion()
	_ = currentVersion // version checked implicitly by length
	if len(current) != len(canonical) {
		return nil // stale, skip
	}

	st = a.sess.compactionState
	if st.StoredSummaries == nil {
		st.StoredSummaries = &StoredSummaries{}
	}

	// Append and cap
	st.StoredSummaries.Summaries = append(st.StoredSummaries.Summaries, stored)
	st.StoredSummaries.TotalTokens += stored.Tokens
	st.StoredSummaries.Generation++

	// Enforce cap: drop oldest summaries if over limit
	for len(st.StoredSummaries.Summaries) > maxStoredSummaries {
		dropped := st.StoredSummaries.Summaries[0]
		st.StoredSummaries.Summaries = st.StoredSummaries.Summaries[1:]
		st.StoredSummaries.TotalTokens -= dropped.Tokens
	}

	// Enforce body token cap
	for st.StoredSummaries.TotalTokens > maxSummaryBodyTokens && len(st.StoredSummaries.Summaries) > 1 {
		dropped := st.StoredSummaries.Summaries[0]
		st.StoredSummaries.Summaries = st.StoredSummaries.Summaries[1:]
		st.StoredSummaries.TotalTokens -= dropped.Tokens
	}

	a.sess.compactionState = st
	if err := a.persistCompactionStateLocked(); err != nil {
		return fmt.Errorf("persist stored summaries: %w", err)
	}

	return nil
}

// twoLayerCompact is the self-locking entry for direct callers (tests, tools).
// Production Prepare paths must use twoLayerCompactLocked: ContextManager
// already holds compactionRunMu for the whole maintenance transaction, and
// sync.Mutex is not reentrant — re-locking here deadlocks the run loop.
func (a *Agent) twoLayerCompact(ctx context.Context, trigger string) (CompactionOutcome, error) {
	a.sess.compactionRunMu.Lock()
	defer a.sess.compactionRunMu.Unlock()
	return a.twoLayerCompactLocked(ctx, trigger)
}

// twoLayerCompactLocked performs a mechanical fold of stored summaries + recent
// tail. Caller holds compactionRunMu. This is 0 LLM tokens — pure system
// operation. The projection becomes: [system + s1 + s2 + ... + recent tail]
func (a *Agent) twoLayerCompactLocked(ctx context.Context, trigger string) (CompactionOutcome, error) {
	canonical, transcriptVersion := a.sess.conversation.snapshotMessagesVersion()
	a.sess.compactionMu.Lock()
	stateSnapshot := a.sess.compactionState
	startProjectionVersion := a.sess.compactionState.Projection.ProjectionVersion
	startGeneration := a.sess.compactionState.Generation
	a.sess.compactionMu.Unlock()

	// Check if we have stored summaries
	if stateSnapshot.StoredSummaries == nil || len(stateSnapshot.StoredSummaries.Summaries) == 0 {
		return CompactionNoop, nil
	}

	// Determine what to keep: recent tail
	head := 0
	if len(canonical) > 0 && canonical[0].Role == provider.RoleSystem {
		head = 1
	}

	// Find the fold boundary: keep recent tail that fits in recentTailBudget
	recentBudget := a.recentTailBudget()
	foldStart := len(canonical)
	acc := 0
	for i := len(canonical) - 1; i > head; i-- {
		c := msgChars(canonical[i])
		if len(canonical)-i > minRecentKeep && acc+c > recentBudget {
			break
		}
		acc += c
		foldStart = i
	}
	// Align off tool results
	for foldStart > head && foldStart < len(canonical) && canonical[foldStart].Role == provider.RoleTool {
		foldStart--
	}

	if foldStart <= head {
		return CompactionNoop, nil // nothing to fold
	}

	// Build projection: [system] + [stored summaries] + [recent tail]
	projMsgs := make([]provider.Message, 0, head+len(stateSnapshot.StoredSummaries.Summaries)+len(canonical)-foldStart)

	// System message
	if head > 0 {
		projMsgs = append(projMsgs, canonical[0])
	}

	// Stored summaries as compaction-summary messages
	for _, s := range stateSnapshot.StoredSummaries.Summaries {
		projMsgs = append(projMsgs, formatSummaryMessage(s.Text))
	}
	// Post-fold recovery briefing (R5): one deterministic notice telling the
	// model what just happened and which tools restore lost context. Lives in
	// the frozen body, so its bytes ride the fold's own cold miss exactly once.
	if len(stateSnapshot.StoredSummaries.Summaries) > 0 {
		projMsgs = append(projMsgs, formatCompactionRecoveryNotice())
	}

	// Recent tail
	projMsgs = append(projMsgs, canonical[foldStart:]...)

	// Project for provider visibility
	projMsgs = provider.ProjectionMessages(projMsgs)

	// Splice with canonical tail for live updates. canonical[len:] is empty by
	// construction (the fold covers through the end); kept as an explicit splice
	// point so a future covered<len variant cannot forget tail replay.
	spliced := append(append([]provider.Message(nil), projMsgs...), canonical[len(canonical):]...)

	projTokens := a.estimatedVisibleRequestTokens(spliced)
	sourceTokens := a.estimatedVisibleRequestTokens(canonical)

	// Accept check: must reduce tokens
	if projTokens >= sourceTokens {
		return CompactionNoop, nil
	}

	// Event parity with single-layer compaction: UI cards and metrics count the
	// Started/Done pair; emitting only Done left mechanical folds invisible to
	// compactionsPerTurn and rendered no CompactionCard.
	a.svc.sink.Emit(event.Event{Kind: event.CompactionStarted, Compaction: event.Compaction{Trigger: trigger}})

	// Commit the mechanical fold
	activeTurn := a.activeTurnCreatedAt.Load()
	viewInputHash := providerVisibleFingerprint(modelInputMessages(canonical))
	viewOutputHash := providerVisibleFingerprint(modelInputMessages(spliced))

	_, err := a.commitSummaryProjection(summaryProjectionCommit{
		canonical: canonical, fold: canonical[head:foldStart], projected: projMsgs,
		result: foldSummary{Mode: CompactionModeSummarized, FoldTokens: sourceTokens - projTokens},
		transcriptVersion: transcriptVersion, projectionVersion: startProjectionVersion,
		generation: startGeneration, activeTurn: activeTurn, trigger: trigger,
		summary: mergeStoredSummaries(stateSnapshot.StoredSummaries.Summaries),
		inputHash: viewInputHash, outputHash: viewOutputHash,
		sourceTokens: sourceTokens, projectionTokens: projTokens, covered: len(canonical),
	})
	if err != nil {
		return CompactionNoop, err
	}

	// Clear stored summaries after successful compact
	a.sess.compactionMu.Lock()
	st := a.sess.compactionState
	st.StoredSummaries = &StoredSummaries{} // reset
	a.sess.compactionState = st
	_ = a.persistCompactionStateLocked()
	a.sess.compactionMu.Unlock()

	// Queue cache diagnostics: mechanical fold changes the provider-visible prefix.
	a.sess.conversation.NoteContentRewrite("compact_auto")
	// Invalidate request byte cache: compaction changes the prefix.
	a.sess.requestCache.invalidate()

	a.svc.sink.Emit(event.Event{Kind: event.CompactionDone, Compaction: event.Compaction{
		Trigger: trigger, Messages: foldStart - head,
		Summary: mergeStoredSummaries(stateSnapshot.StoredSummaries.Summaries),
	}})

	return CompactionInstalled, nil
}

// summarySections lists the headings every stored digest must carry. A section
// counts as filled when any non-blank line follows the heading; "(none)" is an
// acceptable explicit empty. Broken handles poison later folds, so Layer-1
// validates before storing and retries once with corrective feedback.
var summarySections = []string{
	"## Standing facts & constraints",
	"## Goal",
	"## Decisions & rationale",
	"## Files & code",
	"## Commands & outcomes",
	"## Errors & fixes",
	"## Pending & next step",
}

// missingSummarySections returns required headings that are absent or empty.
func missingSummarySections(text string) []string {
	var missing []string
	for i, heading := range summarySections {
		idx := strings.Index(text, heading)
		if idx < 0 {
			missing = append(missing, heading)
			continue
		}
		bodyStart := idx + len(heading)
		bodyEnd := len(text)
		if i+1 < len(summarySections) {
			if next := strings.Index(text[bodyStart:], "\n## "); next >= 0 {
				bodyEnd = bodyStart + next
			}
		}
		body := strings.TrimSpace(text[bodyStart:bodyEnd])
		if body == "" {
			missing = append(missing, heading)
		}
	}
	return missing
}

// sha256Sum8 returns the first 8 bytes of SHA-256 as a slice for hex encoding.
func sha256Sum8(s string) []byte {
	sum := sha256.Sum256([]byte(s))
	return sum[:8]
}

// compactionRecoveryNoticeText is byte-stable by design: no clocks, no counts.
// It rides the fold's own cold miss once and then stays a fixed part of every
// later request's cached prefix.
const compactionRecoveryNoticeText = "<compaction-recovery>\n" +
	"The conversation above was just compacted: older turns exist only inside the <compaction-summary> digests.\n" +
	"The full transcript archive is intact and addressable — restoring context is cheap:\n" +
	"- use_capability(action=\"call\", capability_id=\"session:read\", arguments={\"from\":N,\"to\":M}) — replay exact earlier messages; each digest ends with its <summary-archive from to/> range.\n" +
	"- use_capability(action=\"call\", capability_id=\"session:tool_result\"). — full originals of locally truncated tool outputs.\n" +
	"Rules: treat digest facts as pointers, not ground truth; verify against the archive before any consequential action that depends on them. When a summarized thread becomes relevant again, reopen it proactively instead of guessing from the digest.\n" +
	"</compaction-recovery>"

// formatCompactionRecoveryNotice builds the single post-fold user-turn notice.
func formatCompactionRecoveryNotice() provider.Message {
	return provider.Message{Role: provider.RoleUser, Content: compactionRecoveryNoticeText}
}

// mergeStoredSummaries concatenates stored summaries into a single text block.
func mergeStoredSummaries(summaries []StoredSummary) string {
	if len(summaries) == 0 {
		return ""
	}
	var b strings.Builder
	for i, s := range summaries {
		if i > 0 {
			b.WriteString("\n\n---\n\n")
		}
		b.WriteString(s.Text)
	}
	return b.String()
}
