package agent

import (
	"reasonix/internal/provider"
)

// requestByteCache holds the frozen provider request from the last successful
// turn. On the next turn, the longest common prefix with the new request is
// reused directly — only the new suffix is converted. This eliminates
// re-serialization differences between turns that could break prefix stability.
type requestByteCache struct {
	messages       []provider.Message  // frozen messages from last successful request
	tools          []provider.ToolSchema // frozen tool schemas
	systemHash     string              // SHA-256 of system prompt at cache time
	promptCacheKey string              // lineage guard (workspace|session|model)
	valid          bool                // cache is populated
}

// cacheRequest stores the frozen request for prefix reuse on the next turn.
func (c *requestByteCache) cacheRequest(req provider.Request, systemHash, promptCacheKey string) {
	// Store a copy of the messages slice (not deep-copying individual messages
	// since they are immutable between turns).
	msgs := make([]provider.Message, len(req.Messages))
	copy(msgs, req.Messages)
	tools := make([]provider.ToolSchema, len(req.Tools))
	copy(tools, req.Tools)
	c.messages = msgs
	c.tools = tools
	c.systemHash = systemHash
	c.promptCacheKey = promptCacheKey
	c.valid = true
}

// reusePrefix returns the cached prefix messages if they are still valid
// (same system prompt, same prompt cache key, same tools). The caller can
// append new messages after the common prefix without re-serializing the old ones.
func (c *requestByteCache) reusePrefix(systemHash, promptCacheKey string, newTools []provider.ToolSchema) []provider.Message {
	if !c.valid {
		return nil
	}
	if c.systemHash != systemHash || c.promptCacheKey != promptCacheKey {
		c.valid = false
		return nil
	}
	if len(c.tools) != len(newTools) {
		return nil
	}
	for i := range c.tools {
		if c.tools[i].Name != newTools[i].Name {
			return nil
		}
	}
	return c.messages
}

// invalidate clears the cache. Called after compaction, rewind, model change, etc.
func (c *requestByteCache) invalidate() {
	c.valid = false
	c.messages = nil
	c.tools = nil
	c.systemHash = ""
	c.promptCacheKey = ""
}
