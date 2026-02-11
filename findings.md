# Findings: Memory Quality Investigation

## Production Data Snapshot (2026-02-10)

### Scale (Corrected — much bigger than initially thought)
- **Total memories: 2,453** (not 200 — that was just the search candidate limit)
- **477 unique categories** for 2,453 memories (extreme fragmentation)
- **241 singleton categories** (50.5% of all categories used for exactly 1 memory)

### source_ref Coverage
| Offset Range | Count | With source_ref | Without |
|---|---|---|---|
| 0-300 (newest) | 300 | ~175 (58%) | ~125 (42%) |
| 300-2453 (older) | 2,153 | 0 (0%) | 2,153 (100%) |
| **Total** | **2,453** | **~175 (7.1%)** | **~2,278 (92.9%)** |

### Category Chaos — 36 Duplicate Groups
| Category | Variants | Combined Count |
|---|---|---|
| seo | `SEO(6)`, `seo(138)` | 144 |
| task management | 3 variants | 44 |
| project management | 3 variants | 39 |
| docker | `Docker(5)`, `docker(23)` | 28 |
| error handling | 3 variants | 19 |
| patterns | 1 (legacy dump) | 216 |

### Content Quality Issues
- **~11+ memories** contain ephemeral `/private/tmp/claude-501/...` paths (useless)
- **~52 memories** under 80 chars with no actionable content
- **Near-duplicates**: 6x "Background command failed...", 3x "Check production logs..."
- Content hash dedup fails because `(session: XXXX)` suffix differs
- **68.5% of recent 200 memories have shown_count=0** — never surfaced to user

### "patterns" Category — 216 Legacy Entries
- Migrated from old Mem0-based `memories` table
- No `[TYPE]` prefix, no session reference, no source_ref
- Some contain useful info but nearly impossible to find via search
- Example: `"Device secrets have a length of 72 characters and 288 bits of entropy"`

### Ingestion Pipeline Issues (from codex-investigator)
1. **Confidence threshold too low** — 0.3 in `learning_extraction.py:234`
2. **No content length minimum** — 30-char "learnings" get stored
3. **Category is freeform LLM text** — no controlled vocabulary, no normalization
4. **Dedup is content-hash only** — no semantic dedup
5. **No noise filtering** — tmp paths, exit codes, "background command" all stored

## Codex-Investigator Verdict

> **Both bad data AND bad search context, but bad data is the primary issue.**
> The search infrastructure is solid (98% Recall@5 on LongMemEval).
> The core problem is garbage in, garbage out.

## Phase 2: Hybrid Cleanup — Local DB Analysis (2026-02-10)

### Local DB Restored from Production
- pg_dump: 32MB, restored to local `cems-postgres` Docker container
- Exact match: 2,499 memory_documents, 2,010 legacy memories, 453 relations

### Cleanup Buckets
| Bucket | Count | Action |
|---|---|---|
| Gate rules (category = 'gate-rules') | 2 | **KEEP** — critical |
| Has source_ref (project-tagged) | 212 | **KEEP** — these are good |
| Ever shown, no ref | 399 | **KEEP** — user saw these, some value |
| Noise (tmp paths, bg commands, <50 chars) | 127 | **SOFT-DELETE** — garbage |
| Legacy 'patterns' category | 216 | **SOFT-DELETE** — migrated from old Mem0, mostly useless context fragments |
| Never shown, no ref, non-patterns | 1,681 | **SOFT-DELETE** — never surfaced, no project, low value |
| **Total keep** | **~613** | Gate rules + source_ref + ever-shown |
| **Total soft-delete** | **~2,024** | Noise + patterns + never-shown/no-ref |

### Category Normalization (14 case-duplicate groups)
| Canonical | Variants to merge |
|---|---|
| ai | AI(2) → ai(16) |
| coolify | Coolify(1) → coolify(4) |
| css | CSS(4), CSS styling(4), CSS Styling(1) → css(19) |
| docker | Docker(5), Docker Management(1) → docker(23), docker management(1) |
| google-app-verification | Google app verification(1) → Google App Verification(2) |
| oauth-configuration | OAuth configuration(1) → OAuth Configuration(4) |
| python-imports | python imports(1) → Python imports(1) |
| rails | Rails(1) → rails(9) |
| seo | SEO(6) → seo(138) |
| supabase-rls | supabase rls(3) → Supabase RLS(1) |
| ui-design | ui design(3) → UI Design(3) |
| ui-ux | ui/ux(4) → UI/UX(1) |
| project-management | project management(25), project-management(11) → merge |
| task-management | task-management(30), tool usage(11) → keep separate |

### source_ref Normalization
| Current | Canonical |
|---|---|
| project:pxls | project:EpicCoders/pxls |
| project:pos | project:Chocksy/pos |
| project:cems-analysis | project:Chocksy/cems |

### Decisions Made
1. **Always-on context search** — DONE (Phase 1 complete, committed 5af1f05)
2. **Data cleanup** — DONE locally (Option C Hybrid): 2,499 → 584 active memories, 500 → 33 categories
3. **Ingestion quality gates** — DONE (Phase 3 complete)

## Phase 3: Ingestion Quality Gates (2026-02-10)

### Changes Implemented
| Gate | File | Before | After |
|---|---|---|---|
| Confidence threshold | `learning_extraction.py` | 0.3 (session), 0.5 (tool) | 0.6 (both) |
| Min content length | `learning_extraction.py` | none | 80 chars |
| Noise filtering | `learning_extraction.py` | none | `/private/tmp/claude`, `background command`, `exit code` |
| Category vocabulary | `learning_extraction.py` | freeform LLM text (500 categories) | 30 canonical + alias mapping |
| Relevance threshold | `config.py` | 0.005 (passes everything) | 0.3 (meaningful filter) |
| Semantic dedup | `document_store.py` | content-hash only | content-hash + cosine > 0.92 |

### Expected Impact
- New memories will be **higher quality** (confidence 0.6+, 80+ chars, no noise)
- Categories will stay within 30 canonical values (no more `CSS Styling` vs `css` vs `CSS/UI`)
- Search results will be **more relevant** (threshold 0.3 instead of 0.005)
- Near-duplicate memories (`(session: XXXX)` suffix variants) will be caught by semantic dedup
- Combined with Phase 2 cleanup (584 quality memories), search signal-to-noise ratio should improve significantly
