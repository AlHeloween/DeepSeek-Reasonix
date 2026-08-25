package agent

import (
	"regexp"
	"strings"
	"testing"
)

var stampShape = regexp.MustCompile(`\n\nUTC: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$`)

func TestAppendUTCTurnStampAddsTrailingStamp(t *testing.T) {
	got := appendUTCTurnStamp("fix the parser")
	if !strings.HasPrefix(got, "fix the parser\n\nUTC: ") {
		t.Fatalf("stamp must trail content verbatim, got %q", got)
	}
	if !stampShape.MatchString(got) {
		t.Fatalf("stamp shape mismatch: %q", got)
	}
}

func TestAppendUTCTurnStampIdempotent(t *testing.T) {
	once := appendUTCTurnStamp("fix the parser")
	twice := appendUTCTurnStamp(once)
	if once != twice {
		t.Fatalf("double-stamp: %q vs %q", once, twice)
	}
}

func TestAppendUTCTurnStampPreservesContentBytes(t *testing.T) {
	content := "line one\n<execution-policy>keep</execution-policy>\n你好 world"
	got := appendUTCTurnStamp(content)
	if !strings.HasPrefix(got, content+"\n\nUTC: ") {
		t.Fatalf("content bytes changed:\nwant prefix %q\ngot          %q", content, got)
	}
}

func TestAppendUTCTurnStampDoesNotTouchLeadingBlocks(t *testing.T) {
	// Language blocks prepend at the START; the stamp must not interfere
	// with leading-block detection or reorder anything.
	content := "<response-language>ru</response-language>\n\nзадача"
	got := appendUTCTurnStamp(content)
	if !strings.HasPrefix(got, "<response-language>") {
		t.Fatalf("leading block displaced: %q", got)
	}
	if !stampShape.MatchString(got) {
		t.Fatalf("missing trailing stamp: %q", got)
	}
}
