---
name: cems-share
description: Add a memory to shared team memory. Use when the user asks to share knowledge with the team.
---

# Share with Team

Use the `memory_add` MCP tool with `scope: "shared"` to store team knowledge.

## Project Context

Always include the project reference:
1. Run `git remote get-url origin` to extract `org/repo` format
2. Pass as `source_ref: "project:org/repo"`

## MCP Tool Call

```json
{
  "tool": "memory_add",
  "arguments": {
    "content": "information to share with team",
    "scope": "shared",
    "category": "conventions",
    "tags": [],
    "source_ref": "project:org/repo"
  }
}
```

## When to Use Shared Memory

- Team conventions and coding standards
- Architecture decisions (ADRs)
- Deployment and release processes
- Codebase patterns and onboarding info

## Categories

| Category | Use For |
|----------|---------|
| `conventions` | Coding standards, naming patterns |
| `architecture` | System design, patterns, decisions |
| `workflow` | Deployment, review, release processes |
| `general` | Other shared knowledge |
