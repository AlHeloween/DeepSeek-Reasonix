package agent

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"reasonix/internal/event"
	"reasonix/internal/provider"
	"reasonix/internal/tool"
)

// echoSummarizer is a trivial summarizer that returns a fixed summary.
type echoSummarizer struct{}

func (echoSummarizer) Name() string        { return "echo_summarizer" }
func (echoSummarizer) Description() string { return "echo back a summary" }
func (echoSummarizer) Schema() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{}}`)
}
func (echoSummarizer) ReadOnly() bool { return true }
func (echoSummarizer) Execute(_ context.Context, args json.RawMessage) (string, error) {
	return "summary of conversation", nil
}

func TestStoredSummariesField(t *testing.T) {
	// Verify that CompactionState can store and retrieve StoredSummaries
	st := CompactionState{
		SchemaVersion:     compactionStateSchemaCurrent,
		TranscriptVersion: 1,
		StoredSummaries: &StoredSummaries{
			Summaries: []StoredSummary{
				{Text: "first summary", Tokens: 100, CoveredMsg: 5, Hash: "abc123"},
				{Text: "second summary", Tokens: 150, CoveredMsg: 10, Hash: "def456"},
			},
			TotalTokens: 250,
			Generation:  2,
		},
	}

	// Marshal and unmarshal
	b, err := json.Marshal(st)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var loaded CompactionState
	if err := json.Unmarshal(b, &loaded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if loaded.StoredSummaries == nil {
		t.Fatal("StoredSummaries is nil after unmarshal")
	}
	if len(loaded.StoredSummaries.Summaries) != 2 {
		t.Fatalf("got %d summaries, want 2", len(loaded.StoredSummaries.Summaries))
	}
	if loaded.StoredSummaries.Summaries[0].Text != "first summary" {
		t.Fatalf("first summary text = %q, want %q", loaded.StoredSummaries.Summaries[0].Text, "first summary")
	}
	if loaded.StoredSummaries.TotalTokens != 250 {
		t.Fatalf("total tokens = %d, want 250", loaded.StoredSummaries.TotalTokens)
	}
}

func TestStoredSummariesEmptyOmit(t *testing.T) {
	// Empty StoredSummaries should be omitted from JSON (omitempty)
	st := CompactionState{
		SchemaVersion:     compactionStateSchemaCurrent,
		TranscriptVersion: 1,
	}

	b, err := json.Marshal(st)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	if strings.Contains(string(b), "stored_summaries") {
		t.Fatal("empty StoredSummaries should be omitted from JSON")
	}
}

func TestMergeStoredSummaries(t *testing.T) {
	summaries := []StoredSummary{
		{Text: "summary one"},
		{Text: "summary two"},
		{Text: "summary three"},
	}

	merged := mergeStoredSummaries(summaries)
	if !strings.Contains(merged, "summary one") {
		t.Fatal("merged missing first summary")
	}
	if !strings.Contains(merged, "summary two") {
		t.Fatal("merged missing second summary")
	}
	if !strings.Contains(merged, "summary three") {
		t.Fatal("merged missing third summary")
	}
	if !strings.Contains(merged, "---") {
		t.Fatal("merged missing separator")
	}
}

func TestMergeStoredSummariesEmpty(t *testing.T) {
	merged := mergeStoredSummaries(nil)
	if merged != "" {
		t.Fatalf("empty merge = %q, want empty", merged)
	}
}

func TestTwoLayerSummaryCadence(t *testing.T) {
	a := &Agent{
		agentConfig: agentConfig{contextWindow: 512_000},
	}
	cadence := a.twoLayerSummaryCadence()
	// Should be min(65536, 512000/8) = min(65536, 64000) = 64000
	if cadence != 64000 {
		t.Fatalf("twoLayerSummaryCadence = %d, want 64000", cadence)
	}
}

func TestTwoLayerCompactNoSummaries(t *testing.T) {
	// twoLayerCompact should return CompactionNoop when no stored summaries
	a := &Agent{
		agentConfig: agentConfig{contextWindow: 512_000},
		svc:         agentServices{tools: tool.NewRegistry()},
	}
	a.sess.conversation = NewSession("system prompt")
	a.sess.compactionState = CompactionState{}

	outcome, err := a.twoLayerCompact(context.Background(), CompactionTriggerPressure)
	if err != nil {
		t.Fatalf("twoLayerCompact: %v", err)
	}
	if outcome != CompactionNoop {
		t.Fatalf("outcome = %d, want CompactionNoop", outcome)
	}
}

func TestFormatSummaryMessage(t *testing.T) {
	msg := formatSummaryMessage("test summary")
	if msg.Role != provider.RoleUser {
		t.Fatalf("role = %v, want RoleUser", msg.Role)
	}
	if !strings.Contains(msg.Content, summaryTagOpen) {
		t.Fatal("missing summary tag open")
	}
	if !strings.Contains(msg.Content, summaryTagClose) {
		t.Fatal("missing summary tag close")
	}
	if !strings.Contains(msg.Content, "test summary") {
		t.Fatal("missing summary text")
	}
}

// TestTwoLayerCompactUnderPrepareNoDeadlock is the regression guard for the
// production deadlock: ContextManager.Prepare holds compactionRunMu for the
// whole transaction, so the mechanical fold must run on the Locked variant.
// With stored summaries present above the trigger this used to re-lock the
// non-reentrant mutex and hang the run loop forever.
func TestTwoLayerCompactUnderPrepareNoDeadlock(t *testing.T) {
	a := &Agent{
		agentConfig: agentConfig{contextWindow: 40_000},
		svc:         agentServices{tools: tool.NewRegistry()},
	}
	a.sess.conversation = NewSession("system prompt")
	// Enough history that the 16% tail budget cannot swallow everything:
	// ten ~2K-token turns leave a mid-history fold boundary.
	for i := range 10 {
		a.sess.conversation.Add(provider.Message{Role: provider.RoleUser, Content: strings.Repeat("turn body ", 200) + string(rune('a'+i))})
		a.sess.conversation.Add(provider.Message{Role: provider.RoleAssistant, Content: strings.Repeat("answer body ", 200)})
	}
	a.sess.compactionState = CompactionState{
		SchemaVersion: compactionStateSchemaCurrent,
		StoredSummaries: &StoredSummaries{
			Summaries:   []StoredSummary{{Text: "prior summary", Tokens: 10, CoveredMsg: 2}},
			TotalTokens: 10,
			Generation:  1,
		},
	}

	var started int
	a.svc.sink = event.FuncSink(func(e event.Event) {
		if e.Kind == event.CompactionStarted {
			started++
		}
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	done := make(chan struct{})
	go func() {
		defer close(done)
		// Force pressure Prepare: crosses the fold check with summaries present,
		// exercising exactly the path that deadlocked before the Locked split.
		_, _ = a.contextManager().Prepare(ctx, ContextPreparePolicy{
			Trigger: CompactionTriggerPressure, Force: true,
		})
	}()

	select {
	case <-done:
	case <-time.After(6 * time.Second):
		t.Fatal("Prepare hung: two-layer compact deadlocked on compactionRunMu")
	}

	if v := a.currentProjectionVersion(); v == 0 {
		t.Fatal("mechanical fold did not install a projection (version=0)")
	}
	if started == 0 {
		t.Fatal("mechanical fold must emit CompactionStarted (event parity with single-layer)")
	}
	// R5: the frozen body must carry the recovery briefing after the digests.
	a.sess.compactionMu.Lock()
	projMsgs := a.sess.compactionState.Projection.Messages
	a.sess.compactionMu.Unlock()
	foundNotice := false
	for _, m := range projMsgs {
		if strings.Contains(m.Content, "<compaction-recovery>") {
			foundNotice = true
		}
	}
	if !foundNotice {
		t.Fatal("projection lacks <compaction-recovery> briefing")
	}
	stored := a.sess.compactionState.StoredSummaries
	if stored == nil || len(stored.Summaries) != 0 {
		t.Fatalf("stored summaries must reset after compact, got %+v", stored)
	}
}

// validBriefingFixture satisfies missingSummarySections so Layer-1 accepts it.
const validBriefingFixture = "## Standing facts & constraints\n- go1.26 required\n\n## Goal\n- keep cache aligned\n\n## Decisions & rationale\n- live-prefix probes\n\n## Files & code\n- internal/agent/two_layer_compact.go\n\n## Commands & outcomes\n- go build ./... ok\n\n## Errors & fixes\n- deadlock fixed\n\n## Pending & next step\n- land R3 handles"

func TestMissingSummarySections(t *testing.T) {
	full := validBriefingFixture
	if got := missingSummarySections(full); len(got) != 0 {
		t.Fatalf("valid briefing flagged missing %v", got)
	}
	partial := "## Goal\n- x\n\n## Pending & next step\n- y"
	got := missingSummarySections(partial)
	if len(got) != 5 {
		t.Fatalf("missing = %v (%d), want 5", got, len(got))
	}
	emptyBody := strings.Replace(full, "- keep cache aligned\n", "", 1)
	got = missingSummarySections(emptyBody)
	found := false
	for _, m := range got {
		if m == "## Goal" {
			found = true
		}
	}
	if !found {
		t.Fatal("empty section body must be reported missing")
	}
}

// TestOutOfBandSummaryRequestPrefixAligned guards the F2 fix: the Layer-1
// summarizer request must share its leading prefix with ordinary sampling
// requests (system + earliest history). Building the fold from
// lastSummaryCovered instead diverged at the second message position and
// cold-missed the entire summary call.
func TestOutOfBandSummaryRequestPrefixAligned(t *testing.T) {
	var mu sync.Mutex
	var summaryReq []json.RawMessage
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if isSummarizeRequest(body) {
			mu.Lock()
			summaryReq = decodeMessages(body)
			mu.Unlock()
			writeSSE(w, t,
				streamChunk(deltaText(validBriefingFixture)),
				finishChunk("stop"),
				usageChunk(100, 20, 0, 100))
			return
		}
		writeSSE(w, t,
			streamChunk(deltaText("ok")),
			finishChunk("stop"),
			usageChunk(10, 5, 0, 10))
	}))
	defer srv.Close()

	a, _ := newAgent(t, srv.URL, tool.NewRegistry(), 40_000, 4)

	// History big enough to cross the cadence (max(1024, 40000/8) = 5000 tokens).
	const turnBody = "history body for alignment probe "
	for i := range 6 {
		a.sess.conversation.Add(provider.Message{Role: provider.RoleUser, Content: strings.Repeat(turnBody, 150) + string(rune('a'+i))})
		a.sess.conversation.Add(provider.Message{Role: provider.RoleAssistant, Content: strings.Repeat(turnBody, 150)})
	}
	// A prior summary covering mid-history: with the pre-fix code the request
	// started at canonical[3] and lost prefix alignment; it must not.
	a.sess.compactionState = CompactionState{
		SchemaVersion: compactionStateSchemaCurrent,
		StoredSummaries: &StoredSummaries{
			Summaries:   []StoredSummary{{Text: "prior", Tokens: 1, CoveredMsg: 3}},
			TotalTokens: 1,
			Generation:  1,
		},
	}

	if err := a.maybeGenerateOutOfBandSummary(context.Background()); err != nil {
		t.Fatalf("maybeGenerateOutOfBandSummary: %v", err)
	}

	// R3: the stored digest must carry its exact canonical range handle.
	a.sess.compactionMu.Lock()
	storedList := a.sess.compactionState.StoredSummaries.Summaries
	last := storedList[len(storedList)-1]
	a.sess.compactionMu.Unlock()
	if !strings.Contains(last.Text, `<summary-archive from="3" to="13"/>`) {
		t.Fatalf("archive handle missing or wrong range: %q", last.Text[max(0, len(last.Text)-80):])
	}

	mu.Lock()
	defer mu.Unlock()
	if len(summaryReq) < 4 {
		t.Fatalf("summary request captured %d messages, want >=4", len(summaryReq))
	}
	var first struct {
		Role    provider.Role `json:"role"`
		Content string        `json:"content"`
	}
	if err := json.Unmarshal(summaryReq[0], &first); err != nil {
		t.Fatalf("decode first message: %v", err)
	}
	if first.Role != provider.RoleSystem {
		t.Fatalf("first message role = %q, want system", first.Role)
	}
	var second struct {
		Role    provider.Role `json:"role"`
		Content string        `json:"content"`
	}
	if err := json.Unmarshal(summaryReq[1], &second); err != nil {
		t.Fatalf("decode second message: %v", err)
	}
	if second.Content == "" || !strings.Contains(second.Content, strings.TrimSpace(strings.Repeat(turnBody, 150))) {
		t.Fatal("second message must be the EARLIEST history turn (prefix-aligned), got a later slice")
	}
	if second.Role != provider.RoleUser {
		t.Fatalf("second message role = %q, want user", second.Role)
	}
}
