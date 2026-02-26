---
name: cems-remember
description: Store memories with project context, categories, and tags into CEMS long-term memory
---

# Store a Memory in CEMS

Use the CEMS `memory_add` MCP tool to store information for future recall across sessions.

## Prerequisites

CEMS MCP server must be configured. See [github.com/Chocksy/cems](https://github.com/Chocksy/cems) for setup.

## Usage

When the user says "remember this" or you discover something worth persisting:

1. Determine the content to store
2. Choose the appropriate scope:
   - `personal` (default) — only you see it
   - `shared` — visible to the whole team
3. Pick a category: `preferences`, `conventions`, `architecture`, `decisions`, `workflow`, `errors`, `learnings`, or `general`
4. **Detect the current project** from git remote for scoped recall:
   - Run `git remote get-url origin` to extract `org/repo` format
   - SSH: `git@github.com:org/repo.git` → `org/repo`
   - HTTPS: `https://github.com/org/repo.git` → `org/repo`
   - Pass as `source_ref: "project:org/repo"`

## MCP Tool Call

```json
{
  "tool": "memory_add",
  "arguments": {
    "content": "The description of what to remember",
    "scope": "personal",
    "category": "decisions",
    "tags": ["auth", "security"],
    "source_ref": "project:org/repo"
  }
}
```

## What to Store

- User preferences and style choices
- Project conventions and naming patterns
- Architecture and infrastructure decisions
- Debugging insights and solutions to recurring problems
- Workflow patterns and deployment processes

## What NOT to Store

- Session-specific context (current task, temporary state)
- Information already in the codebase
- Build output or error messages being debugged right now
- Speculative conclusions from a single observation
