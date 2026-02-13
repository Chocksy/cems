# CEMS Memory Guide

You have access to a persistent memory system (CEMS). Use it proactively — don't wait for the user to ask you to remember things.

## When to Store Memories

**Proactively use `/remember` when you discover:**

- Project conventions: "This project uses Tailwind CSS with a utility-first approach"
- Architecture decisions: "Authentication uses JWT with refresh tokens stored in httpOnly cookies"
- User preferences: "User prefers functional components over class components"
- Tech stack details: "Production database is PostgreSQL 16 with pgvector"
- Workflow patterns: "Deployments go through Coolify on the Hetzner server"
- Debugging insights: "The test suite needs Docker running for integration tests"

**Use `/share` for team knowledge:**

- Team conventions: "API responses always include a `success` boolean"
- Shared infrastructure: "Staging environment is at staging.example.com"
- Common patterns: "All models use soft-delete via `deleted_at` column"

## When NOT to Store Memories

- Session-specific context (what file you just read, current task details)
- Temporary state (build output, error messages being debugged right now)
- Information that's already in the codebase (don't duplicate README content)
- Speculative conclusions from a single observation

The observer daemon automatically captures high-level session observations — you don't need to duplicate that.

## Categories

Use these categories for better organization:

| Category | What goes here |
|----------|---------------|
| `preferences` | User likes/dislikes, style preferences |
| `conventions` | Naming patterns, code style, project rules |
| `architecture` | Tech stack, system design, infrastructure |
| `decisions` | Why something was chosen over alternatives |
| `workflow` | How tasks are done, deployment process, CI/CD |
| `general` | Everything else |

## Examples

```
/remember --category architecture The API uses Starlette with uvicorn, not FastAPI
/remember --category conventions All database columns use snake_case
/remember --category preferences User prefers concise responses without excessive comments
/share --category conventions API errors return {"success": false, "error": "message"}
```

## How It Works Behind the Scenes

1. **SessionStart** hook injects your profile (preferences, recent context) at session start
2. **UserPromptSubmit** hook searches memories on every prompt and injects relevant ones
3. **Stop** hook extracts learnings from your session when it ends
4. **Observer** daemon watches transcripts and extracts high-level observations
5. **Maintenance** jobs deduplicate, compress, and prune memories automatically

Your job is to catch the important stuff that the passive systems might miss — especially decisions, preferences, and conventions that emerge during conversations.
