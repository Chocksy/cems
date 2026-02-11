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

## Observer Real-World Test Results (2026-02-11)

### Test 1: Session c10f0702 (earlier today — confirmatory prompts + hook work)
- **127 messages**, 23K chars (~5.7K tokens)
- **5 observations extracted** — all HIGH priority:

| # | Observation | Quality |
|---|---|---|
| 1 | "User wants to implement context-aware memory search for confirmatory prompts (e.g., 'yes' to 'push and deploy?') to surface relevant memories" | Excellent — captures project goal, includes concrete example |
| 2 | "User confirmed that hooks are copied to their system and requested a commit and push of changes, followed by log verification" | Good — captures workflow step |
| 3 | "User expressed strong dissatisfaction with the limited scope of confirmatory prompt detection ('yes', 'go ahead', 'ship it') and wants it applied 'always all the time'" | Excellent — captures user sentiment AND preference, quotes their words |
| 4 | "User wants to use persistent markdown files (planning-with-files skill) as working memory" | Good — captures tool preference |
| 5 | "User decided to export production CEMS data as a PostgreSQL dump for local cleanup, rather than JSON" | Excellent — captures decision + alternative rejected |

**Verdict**: 4/5 excellent or good. Captures decisions, preferences, sentiment. No implementation noise.

### Test 2: Session a968ab3a (current session — observer build)
- **187 messages**, 37K chars (~9.3K tokens)
- **4 observations extracted** (mix of priorities):

| # | Priority | Observation | Quality |
|---|---|---|---|
| 1 | HIGH | "User wants to test the new Observer implementation on a real session to evaluate its intelligence and effectiveness" | Good — captures intent |
| 2 | MEDIUM | "User asked if the current plan is complete and to redeploy local Docker to test the Observer on a recent session" | Mediocre — too specific to one moment |
| 3 | LOW | "Assistant will use an earlier session (c10f0702) for testing" | Weak — assistant action, not user context |
| 4 | LOW | "Assistant will write a script to simulate the Observer daemon's behavior" | Weak — transient detail |

**Verdict**: 1/4 good. This session was after context compaction, so the transcript was mostly the tail-end testing phase. The observer correctly identified USER intent (#1) but also captured assistant implementation details that it shouldn't have (#3, #4).

### Assessment
- **Strong when sessions have rich user-assistant dialogue about goals, decisions, preferences**
- **Weaker when transcript is dominated by assistant implementation (code, tests, debugging)**
- The Mastra-inspired prompt successfully avoids CLI commands, file paths, and error messages
- Priority classification works: assertions → HIGH, questions → MEDIUM, informational → LOW
- Temporal anchoring not tested yet (no date references in these sessions)

## API Endpoint Tests (2026-02-11)

### Bug Found: Markdown-fenced JSON parsing
- **Symptom**: `/api/session/observe` returned `observations_stored: 0` for valid transcripts
- **Root cause**: LLM (Gemini 2.5 Flash) returned JSON wrapped in ` ```json ``` ` with truncated closing backticks
- **Fix**: Added fallback regex in `extract_json_from_response()` to handle opening backticks without closing ones
- **File**: `src/cems/lib/json_parsing.py` — line 48-51

### Test 3: API endpoint with assistant-heavy transcript (session a968ab3a, 7KB)
- 5 observations stored, all about Assistant actions (no user messages in that chunk)
- Quality: captures implementation changes but attributes them to "Assistant" not "User"
- **Issue**: When transcript has no user messages, observations are about assistant work

### Test 4: API endpoint with rich user dialogue (session c10f0702, 18KB)
- 3 observations stored after JSON parsing fix
- Quality: 2/3 good — captures user concern about restrictive threshold, testing intent
- **Issue**: Obs #2 includes specific file references (`@test_integration.py`) despite prompt saying not to

### Overall API Test Verdict
- Endpoint works end-to-end: accepts transcript → LLM extraction → stores observations → returns IDs
- JSON parsing fix was needed for production reliability
- Observation quality depends heavily on user message density in the transcript
- The observer daemon's transcript extraction (in `daemon.py`) should ensure user messages are included

## LongMemEval End-to-End Eval Research (2026-02-11)

### Key Discovery: CEMS Eval Measures Something Different From the Leaderboard

| | CEMS Current | Leaderboard (Mastra, Supermemory, etc.) |
|---|---|---|
| **Dataset** | Oracle (1-6 evidence sessions only) | S variant (40 sessions, 30+ distractors) |
| **Metric** | Retrieval Recall@5 | Answer accuracy (LLM-as-judge) |
| **LLM involved?** | No (pure vector search) | Yes (LLM reads context and answers) |
| **Judge** | Session ID match | GPT-4o says "yes/no" to answer quality |
| **Our score** | 98% Recall@5 | NOT COMPARABLE to leaderboard |

Our 98% is on Oracle (finding the needle among 1-6 sessions). The leaderboard tests finding + reading among 40 sessions. These are fundamentally different difficulties.

### How Mastra's Benchmark Works (from source code analysis)

**Phase 1: Prepare** — Per question, creates isolated memory store, feeds sessions through observer/reflector (Gemini 2.5 Flash). Observer runs DURING prepare as a processor on agent.generate(). Shortcut configs do single finalize() pass instead.

**Phase 2: Run** — Answering agent (e.g., GPT-4o) receives question + memory context (observations or semantic recall). System prompt: "You are a helpful assistant with access to extensive conversation history." Answer generated at temperature=0.

**Phase 3: Judge** — Separate GPT-4o judge receives question + correct answer + model answer. Evaluates with type-specific prompts:
- Standard types: "does the response contain the correct answer?"
- Temporal: allows off-by-one errors
- Knowledge-update: accepts old+new if updated answer present
- Preference: lenient, just needs correct use of personal info
- Abstention: "did model correctly identify as unanswerable?"

**Key insight: Per-question isolation.** Each question gets its own independent memory store. No cross-contamination between questions. 500 independent memory instances.

### Leaderboard Scores (LongMemEval_S, GPT-4o reader)

| System | Score | Notes |
|--------|-------|-------|
| Mastra OM (gpt-5-mini) | 94.87% | Best model, not just best memory |
| Mastra OM (gpt-4o) | 84.23% | Fair comparison to others |
| Emergence Internal | 86.0% | Not reproducible |
| Supermemory (gpt-4o) | 81.6% | Open source |
| Emergence Simple | 82.4% | Surprisingly close to fancy systems |
| Mastra RAG (topK=20) | 80.0% | Pure RAG, no observer |
| Zep (gpt-4o) | 71.2% | Knowledge graph approach |
| GPT-4o full context | 60.6% | Baseline (everything in context window) |

### CEMS Adaptation Design

**Option A: Retrieval-only upgrade (easy)**
- Switch from Oracle → S dataset (40 sessions)
- Keep retrieval-only metric (Recall@5)
- Would reveal true retrieval difficulty (currently Oracle makes it too easy)
- Estimated: ~1 day work

**Option B: Full end-to-end eval (Mastra-comparable)**
- Switch to S dataset
- Add LLM answer generation step (GPT-4o reads retrieved context + answers)
- Add LLM-as-judge step (GPT-4o evaluates answer correctness)
- Use same type-specific judge prompts as paper/Mastra
- Would produce a score directly comparable to leaderboard
- Estimated: ~2-3 days work

**Option C: Observer impact measurement (Option B + observer)**
- Run Option B as baseline
- Also run sessions through observer (create observations)
- At query time, include observations alongside vector search results
- Compare: baseline retrieval-only vs retrieval + observations
- Measures the VALUE of the observer system we just built
- Estimated: ~3-4 days (includes Option B)

### Would It Skip CEMS Tools?

**No — it would exercise MORE of the pipeline than current eval:**

| Pipeline Component | Current Eval | Option B | Option C |
|---|---|---|---|
| Ingestion (add_async) | ✅ | ✅ | ✅ |
| Chunking (800 tokens) | ✅ | ✅ | ✅ |
| Embedding (1536-dim) | ✅ | ✅ | ✅ |
| Dedup (hash + semantic) | ✅ | ✅ | ✅ |
| Vector search | ✅ | ✅ | ✅ |
| Hybrid search (BM25) | partial | ✅ | ✅ |
| Query synthesis (LLM) | ❌ | ✅ | ✅ |
| Score adjustments | ✅ | ✅ | ✅ |
| Observer extraction | ❌ | ❌ | ✅ |
| Observation retrieval | ❌ | ❌ | ✅ |
| LLM answer generation | ❌ | ✅ | ✅ |
| LLM-as-judge | ❌ | ✅ | ✅ |

**Option C is the most comprehensive test** — it exercises every component in the CEMS pipeline including the observer we just built.

### Per-Question Isolation Question

Mastra uses per-question isolation (fresh memory store per question). Two approaches for CEMS:

1. **Shared store** (simpler): All sessions go in one store, tagged with source_ref. At query time, no filtering — tests real cross-question contamination. Simpler but less pure.
2. **Per-question store** (purer): Clean eval DB before each question, ingest only that question's sessions. More work but directly comparable to Mastra. Could use admin cleanup API we already have.

**Recommendation**: Start with shared store (it's what CEMS actually does in production). If scores are anomalously low/high due to cross-contamination, switch to per-question isolation.

### Cost Estimate for Full Eval (500 questions)

| Component | API Calls | Est. Cost |
|---|---|---|
| Session ingestion (S: ~20K unique sessions) | 20K embeddings | ~$2 (text-embedding-3-small) |
| Observer extraction (20K sessions) | 20K Gemini Flash calls | ~$5-10 |
| Search (500 questions) | 500 hybrid searches | ~$0.50 |
| Query synthesis (500 questions) | 500 Gemini Flash calls | ~$1 |
| LLM answer generation (500 questions) | 500 GPT-4o calls | ~$15-25 |
| LLM-as-judge (500 questions) | 500 GPT-4o calls | ~$10-15 |
| **Total Option B** | | **~$30-45** |
| **Total Option C** | | **~$35-55** |
