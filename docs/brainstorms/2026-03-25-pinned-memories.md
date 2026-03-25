---
date: 2026-03-25
topic: pinned-memories
---

# Pinned Memories — Tag-Based Protection from Distillation and Consolidation

## What We're Building

A `pinned` tag that marks individual memories as untouchable — no distillation (condensation), no consolidation (merging). Pinned memories stay exactly as stored. LLMs auto-detect when users say "always remember", "don't forget", "this is important" and add the tag. Users can also explicitly pin via `/pin <id>`.

## Why This Approach

- **Tag-based**: Zero migration. Tags are already `TEXT[]` on `memory_documents`. Distillation and consolidation already check tags (`consolidated:N`). Just add `pinned` to the skip list.
- **LLM-side detection**: The LLM already understands emphasis. Adding a prompt instruction to the UserPromptSubmit hook is simpler and more natural than server-side keyword detection.
- **Concrete problem**: PR template preferences in hubstaff-server keep disappearing. User has to repeat themselves. A pinned memory would survive all maintenance jobs permanently.

## Key Decisions

- **Mechanism**: `pinned` tag on memory_documents.tags array
- **Protection scope**: Both distillation AND consolidation skip pinned memories
- **Auto-pin**: LLM detects emphasis phrases ("always remember", "don't forget", "this is important", "permanent") via UserPromptSubmit hook prompt instructions
- **Explicit pin**: `/pin <id>` command and `memory_update` with tags addition
- **Dashboard**: Show pinned badge/indicator on memories in the debug UI
- **No server-side detection**: Keep it simple — LLM handles intent detection

## Scope

### In scope
- Distillation job: skip docs with `pinned` tag
- Consolidation job: skip docs with `pinned` tag
- Summarization job: skip docs with `pinned` tag (in _prune_chronically_noisy)
- /pin command (new skill) — adds `pinned` tag to a memory by ID
- /unpin command — removes `pinned` tag
- UserPromptSubmit hook: add instruction for LLM to auto-pin emphatic memories
- /store skill: add instruction to pin when user says "always"/"permanent"
- Dashboard: pinned badge on memory cards
- memory_update API: support adding/removing tags (may already work)

### Out of scope
- Server-side auto-detection of important content
- Bulk pin/unpin operations
- Pin expiry or TTL

## Open Questions

1. Should the `/pin` command also prevent noise-based pruning (_prune_chronically_noisy)?
   → Yes, pinned = fully untouchable.
2. Should pinned memories get a search boost?
   → Maybe later. Not in MVP.

## Next Steps
→ Implement via `/workflows:plan` or directly (small feature)
