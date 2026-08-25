package agent

import (
	"regexp"
	"time"
)

// utcTurnStampRe matches an existing trailing UTC stamp so a turn composed
// twice (retry, recovery replay) never accumulates duplicate stamps.
var utcTurnStampRe = regexp.MustCompile(`(?:^|\n)UTC: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s*$`)

// appendUTCTurnStamp pins wall-clock truth onto a persisted user turn.
//
// The model has no clock: without a stamp it cannot scope web searches
// ("releases this year"), judge whether found docs are current, or reason
// about elapsed session time. The stamp rides the message text itself —
// written ONCE when the turn enters the transcript and immutable afterwards,
// so every later request replays byte-identical history and the provider's
// prompt-cache prefix stays intact (opencode-style cache-stable dating).
//
// The stamp goes at the END of the content deliberately: the leading bytes are
// owned by transient language-preference blocks (hasLeadingInjectedBlock keys
// off the start), and compress anchors match excerpts anywhere in the text,
// so a trailing suffix collides with neither.
func appendUTCTurnStamp(content string) string {
	if utcTurnStampRe.MatchString(content) {
		return content
	}
	return content + "\n\nUTC: " + time.Now().UTC().Format("2006-01-02T15:04:05.000Z")
}
