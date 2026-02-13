# Task: CEMS Polish — CLAUDE.md Memory Instructions, Banner/Logo, Deploy Guides

## Goal
1. Add CLAUDE.md instructions so Claude Code proactively stores memories
2. Generate a creative banner/logo with mascot for the README
3. Create deployment guides (AWS, local, self-hosted)
4. Update README with banner and deploy sections

## Context
- README already rewritten with correct architecture (no Mem0/Kuzu/Qdrant)
- `cems setup` and `cems uninstall` work
- Observer handles passive memory extraction from session transcripts
- Skills (/remember, /recall, etc.) handle manual memory operations
- Gap: Claude Code doesn't proactively add memories during sessions

## Image Generation Models (OpenRouter, Feb 2026)

| Model ID | Name | Best For | Price |
|----------|------|----------|-------|
| `google/gemini-3-pro-image-preview` | Nano Banana Pro | Best text rendering, complex compositions | $0.12/1K img tokens |
| `openai/gpt-5-image` | GPT-5 Image | Superior instruction following, editing | $40/M img tokens |

**Decision**: Use Nano Banana Pro (`google/gemini-3-pro-image-preview`) — best text rendering for logo text, cheaper.

## Mascot/Brand Ideas

CEMS = Continuous Evolving Memory System

Creative angles:
- **CEMS the Cephalopod** — A cuttlefish (not octopus — GitHub has that). Cuttlefish have the best memory of all invertebrates, can camouflage (adapt/evolve), and have 3 hearts (personal + shared + system memory). Tentacles reaching out to grab data from different sources.
- **CEMS the Elephant** — Elephants never forget. Classic but effective. Trunk could wrap around code symbols.
- **Brain with circuits** — Neural network meets code. More abstract, less fun.
- **Memory palace / library** — Architectural metaphor. Filing cabinet with tentacles.

**Winner: Cuttlefish** — Unique (not octopus), scientifically accurate (great memory), visually distinctive, fits "evolving" theme (camouflage = adaptation).

---

## Phase 1: CLAUDE.md Memory Instructions — `status: pending`

Add a section to the CEMS skills or CLAUDE.md that instructs Claude Code to:
- Proactively use /remember when it discovers project conventions
- Store architectural decisions, preferences, and workflow patterns
- NOT store session-specific details (that's the observer's job)
- Use appropriate categories (preferences, conventions, architecture, workflow)

Options:
a) Add to `~/.claude/CLAUDE.md` (global, user's file — we shouldn't touch)
b) Add a `.claude/CLAUDE.md` per-project file (project-level instructions)
c) Bundle instructions in a CEMS skill file that gets injected via SessionStart hook
d) Add to the SessionStart profile context that already gets injected

**Best approach**: Option (c) or (d) — add memory-proactive instructions to the profile context that SessionStart hook injects. This way it's automatic and doesn't require modifying user's CLAUDE.md.

### Files to modify
- `src/cems/data/claude/skills/cems/memory-guide.md` — NEW skill with guidelines
- OR modify the server's profile endpoint response to include instructions

## Phase 2: Banner & Logo Generation — `status: pending`

### Step 1: Generate banner with cuttlefish mascot
- Model: `google/gemini-3-pro-image-preview` (Nano Banana Pro)
- Prompt: Dark background, cuttlefish with data/memory tentacles, "CEMS" text, developer aesthetic
- Size: 1280x320 (4:1 aspect ratio for GitHub README)
- Generate dark and light variants

### Step 2: Create assets directory
- `assets/banner-dark.png`
- `assets/banner-light.png`

### Step 3: Update README header
- Use `<picture>` element for dark/light mode switching

## Phase 3: Deploy Guides — `status: pending`

### Local Development
- Simple `docker compose up -d` guide (already in README)
- Add: how to run without Docker (direct Python + PostgreSQL)

### Self-Hosted (VPS)
- Hetzner/DigitalOcean VPS + Docker Compose
- Add Coolify deployment guide (we already use it)
- Nginx reverse proxy + SSL via Certbot or Coolify

### AWS
- ECS/Fargate option (most common)
- Or simple EC2 + docker-compose (cheapest)
- Include a basic CloudFormation template or step-by-step

### What to add to README
- Collapsible `<details>` sections for each deploy target
- Keep the Server Deployment section clean, link to detailed guides

## Phase 4: Final README Update — `status: pending`

- Add banner with `<picture>` dark/light
- Add deploy guide sections
- Verify all links work

---

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| (none yet) | | |
