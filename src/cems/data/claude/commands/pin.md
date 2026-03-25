---
name: pin
description: Pin or unpin a CEMS memory to protect it from distillation and consolidation
---

# Pin/Unpin Memory

Pin a memory to protect it permanently from distillation (condensation), consolidation (merging), and noise pruning. Pinned memories stay exactly as stored.

## Usage

### Pin by ID
If given a memory ID (or short ID):
```
mcp__cems__memory_pin with memory_id=<full_id>, pin=true
```

### Unpin by ID
```
mcp__cems__memory_pin with memory_id=<full_id>, pin=false
```

### Pin by search
If given a description instead of an ID:
1. Search: `mcp__cems__memory_search` with the description as query
2. Show matching results to the user
3. Ask which one(s) to pin
4. Pin each selected one

## Auto-pinning

When storing memories, automatically pin if the user says:
- "always remember", "don't forget", "this is important"
- "permanent", "never change this", "keep this forever"

After storing via `mcp__cems__memory_add`, immediately call `mcp__cems__memory_pin` with the returned memory ID.
