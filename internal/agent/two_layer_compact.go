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

// twoLayerCompactTrigger returns the token threshold for mechanical compact.
// This is higher than the current 80% trigger — compact only when the model
// window is nearly full.
func (a *Agent) twoLayerCompactTrigger() int {
	window := a.effectiveContextWindow()
	if window <= 0 {
		return a.compactTrigger() // fallback to single-layer
	}
	// Compact at usable(model) = context − headroom
	// headroom = output budget + protocol reserve
	headroom := a.maxOutputTokens + protocolReserveTokens
	if headroom <= 0 {
		headroom = 32_768 // default 32K headroom
	}
	return max(1, window-headroom)
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

	// Calculate tokens since last summary
	tokensSinceLastSummary := 0
	lastSummaryCovered := 0
	if st.StoredSummaries != nil && len(st.StoredSummaries.Summaries) > 0 {
		last := st.StoredSummaries.Summaries[len(st.StoredSummaries.Summaries)-1]
		lastSummaryCovered = last.CoveredMsg
	}

	// Count tokens in new messages since last summary
	for i := lastSummaryCovered; i < len(canonical); i++ {
		if i < len(visible) {
			tokensSinceLastSummary += estimateMessagesTokens(visible[i : i+1])
		}
	}

	// Check cadence
	if tokensSinceLastSummary < a.twoLayerSummaryCadence() {
		return nil // not enough new content
	}

	// Determine fold region: from last summary cover point to ~80% of new content
	// (keep recent tail for context)
	foldStart := lastSummaryCovered
	foldEnd := len(canonical)
	if foldEnd <= foldStart {
		return nil
	}

	// Keep recent tail (16% of window)
	recentBudget := a.recentTailBudget()
	recentEnd := foldEnd
	for recentEnd > foldStart && estimateMessagesTokens(modelInputMessages(canonical[recentEnd-1:foldEnd])) < recentBudget {
		recentEnd--
	}
	if recentEnd <= foldStart+1 {
		return nil // too little to summarize
	}

	fold := canonical[foldStart:recentEnd]
	if len(fold) == 0 {
		return nil
	}

	// Generate summary with SAME prefix as normal turns (cache hit)
	// The summaryRequest builder already prepends system message
	summary, usage, err := a.summarize(ctx, fold, "")
	if err != nil {
		return fmt.Errorf("out-of-band summary: %w", err)
	}
	if summary == "" {
		return nil
	}

	// Emit usage
	if usage != nil && (usage.TotalTokens > 0 || usage.RequestCount > 0) {
		a.svc.sink.Emit(event.Event{Kind: event.Usage, ModelRef: a.modelRef, Usage: usage, Pricing: a.svc.pricing, UsageSource: event.UsageSourceCompaction})
	}

	// Store summary
	hash := sha256.Sum256([]byte(summary))
	stored := StoredSummary{
		Text:       summary,
		Tokens:     estimateTextTokens(summary),
		CoveredMsg: recentEnd,
		Hash:       hex.EncodeToString(hash[:8]),
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

// twoLayerCompact performs a mechanical fold of stored summaries + recent tail.
// This is 0 LLM tokens — pure system operation. The projection becomes:
// [system + s1 + s2 + ... + recent tail]
func (a *Agent) twoLayerCompact(ctx context.Context, trigger string) (CompactionOutcome, error) {
	a.sess.compactionRunMu.Lock()
	defer a.sess.compactionRunMu.Unlock()

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

	// Recent tail
	projMsgs = append(projMsgs, canonical[foldStart:]...)

	// Project for provider visibility
	projMsgs = provider.ProjectionMessages(projMsgs)

	// Splice with canonical tail for live updates
	spliced := append(append([]provider.Message(nil), projMsgs...), canonical[len(canonical):]...)

	projTokens := a.estimatedVisibleRequestTokens(spliced)
	sourceTokens := a.estimatedVisibleRequestTokens(canonical)

	// Accept check: must reduce tokens
	if projTokens >= sourceTokens {
		return CompactionNoop, nil
	}

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

// modelVisibleWithStoredSummaries extends modelVisibleFromProjection to include
// stored summaries in the visible view. This is used by the two-layer path.
func modelVisibleWithStoredSummaries(st CompactionState, canonical []provider.Message) []provider.Message {
	base := modelVisibleFromProjection(st.Projection, canonical)
	if st.StoredSummaries == nil || len(st.StoredSummaries.Summaries) == 0 {
		return base
	}
	// Stored summaries are already in the projection messages via formatSummaryMessage
	// during twoLayerCompact. For out-of-band summaries not yet compacted,
	// they live outside the content window and don't affect the visible view.
	return base
}
