package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"unicode/utf8"

	"reasonix/internal/provider"
	"reasonix/internal/tool"
)

// session:read gives the model addressable access to the canonical transcript
// archive. Compaction digests end with <summary-archive from=… to=…/> handles;
// this capability replays those exact index ranges so summarized facts stay
// verifiable and lost threads can be reopened on demand. Paging is by message
// index with a byte budget per page, mirroring session:tool_result.
const (
	sessionReadCapabilityID = "session:read"
	sessionReadPageDefault  = 16 * 1024
	sessionReadPageMax      = 32 * 1024
	// Per-message render cap keeps one huge historical turn from starving the
	// rest of the page; full bodies remain reachable via their own tools.
	sessionReadMessageCap = 4 * 1024
)

type sessionReadBinder interface {
	bindSessionReadSession(func() *Session)
}

type sessionReadTarget struct {
	session func() *Session
}

func (*sessionReadTarget) Name() string { return "session_read" }

func (*sessionReadTarget) Description() string {
	return "Replay exact earlier conversation messages by canonical index range. Digests reference their coverage via <summary-archive from to/>."
}

func (*sessionReadTarget) ReadOnly() bool { return true }

func (*sessionReadTarget) Schema() json.RawMessage {
	return json.RawMessage(`{
		"type":"object",
		"properties":{
			"from":{"type":"integer","minimum":0,"description":"inclusive start index into the canonical transcript"},
			"to":{"type":"integer","minimum":0,"description":"exclusive end index"},
			"max_bytes":{"type":"integer","minimum":1,"maximum":32768}
		},
		"required":["from","to"]
	}`)
}

type sessionReadParams struct {
	From    int `json:"from"`
	To      int `json:"to"`
	MaxByte int `json:"max_bytes"`
}

// Execute renders messages [from, to) of the canonical transcript, provider
// view only (LocalOnly stripped, RawContent excluded), under a JSON paging
// header. The response is deterministic for a given transcript prefix — no
// timestamps — so replaying a range never perturbs prompt-cache prefixes.
func (t *sessionReadTarget) Execute(_ context.Context, args json.RawMessage) (string, error) {
	var p sessionReadParams
	if err := json.Unmarshal(args, &p); err != nil {
		return "", fmt.Errorf("session read: invalid args: %w", err)
	}
	if p.From < 0 || p.To < 0 || p.To < p.From {
		return "", fmt.Errorf("session read: invalid range from=%d to=%d", p.From, p.To)
	}
	if p.MaxByte == 0 {
		p.MaxByte = sessionReadPageDefault
	}
	if p.MaxByte < 1 || p.MaxByte > sessionReadPageMax {
		return "", fmt.Errorf("session read: max_bytes must be between 1 and %d", sessionReadPageMax)
	}
	if t == nil || t.session == nil {
		return "", fmt.Errorf("session read: current session is unavailable")
	}
	session := t.session()
	if session == nil {
		return "", fmt.Errorf("session read: current session is unavailable")
	}
	msgs := provider.ModelMessages(session.Snapshot())
	to := min(p.To, len(msgs))
	if p.From > len(msgs) {
		return "", fmt.Errorf("session read: from=%d exceeds transcript length %d", p.From, len(msgs))
	}

	var b strings.Builder
	next := to
	used := 0
	emitted := 0
	for i := p.From; i < to; i++ {
		rendered := renderArchivedMessage(i, msgs[i])
		if used+len(rendered) > p.MaxByte && emitted > 0 {
			next = i
			break
		}
		if used+len(rendered) > p.MaxByte {
			// First message alone exceeds the budget: clip it and stop.
			clipped := snapToRuneBoundary(rendered, 0, max(0, p.MaxByte-used))
			b.WriteString(clipped + "…[message clipped]")
			next = i + 1
			used += len(clipped)
			emitted++
			break
		}
		b.WriteString(rendered)
		used += len(rendered)
		emitted++
	}
	header, _ := json.Marshal(struct {
		From     int  `json:"from"`
		To       int  `json:"to"`
		NextFrom int  `json:"next_from,omitempty"`
		Total    int  `json:"transcript_length"`
		Bytes    int  `json:"bytes"`
		Complete bool `json:"complete"`
	}{From: p.From, To: to, NextFrom: next, Total: len(msgs), Bytes: used, Complete: next >= to})
	return string(header) + "\n" + strings.TrimRight(b.String(), "\n"), nil
}

// renderArchivedMessage formats one archived message with its canonical index,
// bounded body, and a pointer when the body was clipped locally.
func renderArchivedMessage(idx int, m provider.Message) string {
	body := m.Content
	clipped := ""
	if len(body) > sessionReadMessageCap {
		cut := snapToRuneBoundary(body, 0, sessionReadMessageCap)
		body = cut
		clipped = fmt.Sprintf("\n…[clipped at %d bytes]", sessionReadMessageCap)
	}
	if !utf8.ValidString(body) {
		body = strings.ToValidUTF8(body, "\uFFFD")
	}
	role := string(m.Role)
	name := ""
	if m.Name != "" && role == "tool" {
		name = " name=" + m.Name
	}
	return fmt.Sprintf("[%d] %s%s: %s%s\n", idx, role, name, body, clipped)
}

// --- use_capability wiring -------------------------------------------------

// bindSessionReadSessionCapability wires the archive reader to this agent's
// own session, mirroring bindToolResultSessionCapability.
func (a *Agent) bindSessionReadSessionCapability() {
	if a == nil || a.svc.tools == nil {
		return
	}
	proxy, ok := a.svc.tools.Get("use_capability")
	if !ok {
		return
	}
	binder, ok := proxy.(sessionReadBinder)
	if !ok {
		return
	}
	binder.bindSessionReadSession(func() *Session { return a.Session() })
}

func (t *UseCapabilityTool) bindSessionReadSession(session func() *Session) {
	if t == nil {
		return
	}
	t.sessionReadMu.Lock()
	t.sessionReadSession = session
	t.sessionReadMu.Unlock()
}

func (t *UseCapabilityTool) currentSessionReadTarget() tool.Tool {
	if t == nil {
		return nil
	}
	t.sessionReadMu.RLock()
	session := t.sessionReadSession
	t.sessionReadMu.RUnlock()
	if session == nil {
		return nil
	}
	return &sessionReadTarget{session: session}
}

func (t *UseCapabilityTool) resolveSessionRead(args json.RawMessage, base tool.ResolvedCall) (tool.ResolvedCall, error) {
	target := t.currentSessionReadTarget()
	if target == nil {
		return tool.ResolvedCall{}, fmt.Errorf("capability %q is unavailable without a current agent session", sessionReadCapabilityID)
	}
	base.TargetName = target.Name()
	base.Target = target
	base.Args = args
	base.ReadOnly = true
	return base, nil
}

func (t *UseCapabilityTool) inspectSessionRead() (string, error) {
	if t.currentSessionReadTarget() == nil {
		return "", fmt.Errorf("capability %q is unavailable without a current agent session", sessionReadCapabilityID)
	}
	payload := map[string]any{
		"id": sessionReadCapabilityID, "kind": "session", "name": "read",
		"description": "Replay exact earlier conversation messages by canonical index range; digests carry <summary-archive from to/> handles.",
		"status":      "ready", "read_only": true,
		"arguments": map[string]any{
			"from": "required inclusive start", "to": "required exclusive end",
			"max_bytes_default": sessionReadPageDefault, "max_bytes_max": sessionReadPageMax,
		},
	}
	b, err := json.MarshalIndent(payload, "", "  ")
	return string(b), err
}

// ensure interfaces hold
var (
	_ sessionReadBinder = (*UseCapabilityTool)(nil)
	_ tool.Tool         = (*sessionReadTarget)(nil)
)
