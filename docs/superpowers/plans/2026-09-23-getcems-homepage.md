# getcems.com Homepage Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the ocean-themed homepage with one plain, dark, neutral page that sells CEMS as a portable team memory that can run fully on the customer's servers.

**Architecture:** Astro 5 + Tailwind 4 static site in `/Volumes/External/Development/getcems.com`. One page (`src/pages/index.astro`) composed of section components. No new dependencies.

**Tech Stack:** Astro 5, Tailwind 4 (`@tailwindcss/vite`), Geist fonts already installed.

**Spec:** `docs/superpowers/specs/2026-09-22-private-mode-and-enterprise-positioning-design.md`, Part 3 (in the cems repo).

## Global Constraints

- One homepage. No `/enterprise` page.
- Dark neutral palette, one accent colour. Remove the ocean theme: `OceanFloorBg.astro`, `src/assets/ocean-*`, `public/ocean-rays-bg.png`, the ray/bubble/plankton generator in `Layout.astro`, and every `ocean` reference.
- No new npm dependencies. No illustrations beyond one data-flow diagram (inline SVG or HTML/CSS) and the three-panel switching graphic (HTML/CSS).
- Copy rules: no em dashes anywhere (use commas, colons, parentheses). No "seamless", "robust", "leverage", "game-changer", "supercharge". No "This isn't X. This is Y." pattern. Short sentences.
- Every claim in sections 3, 5 and 7 must match `docs/DEPLOYMENT.md#private-mode` and `docs/CLIENT.md` in the cems repo.
- Enterprise contact address: `hello@getcems.com` held in one constant so it can change in one place.
- `npm run build` passes. Lighthouse accessibility 90 or above.

---

### Task 1: Homepage rewrite

**Files:**
- Modify: `src/pages/index.astro`, `src/layouts/Layout.astro`, `src/styles/global.css`, `src/components/Nav.astro`, `src/components/Hero.astro`, `src/components/HowItWorks.astro`, `src/components/FAQ.astro`, `src/components/Footer.astro`, `src/components/InstallSnippet.astro`, `public/_redirects`
- Create: `src/components/Switching.astro`, `src/components/DataFlow.astro`, `src/components/WorksWith.astro`, `src/components/Proof.astro`, `src/components/Install.astro`, `src/components/Enterprise.astro`
- Delete: `src/components/OceanFloorBg.astro`, `src/components/Integrations.astro`, `src/components/SocialProof.astro`, `src/components/Features.astro`, `src/components/CodeExample.astro`, `src/components/FinalCTA.astro`, `src/components/LivelyMemories.astro` (only if nothing else imports them after the rewrite), `src/assets/ocean-*`, `public/ocean-rays-bg.png`

**Section order in `index.astro`:** Nav, Hero, Switching, DataFlow, HowItWorks, WorksWith, Proof, Install (`id="install"`), Enterprise (`id="enterprise"`), FAQ (`id="faq"`), Footer.

**Copy (use verbatim; adjust only line breaks):**

1. **Hero**
   - H1: "Your engineering team's memory. Portable across every AI agent. Running on your servers."
   - Sub: "CEMS remembers your decisions, conventions and fixes, and brings them back in every session. It works with Claude Code, Cursor, Codex, Goose and any MCP client."
   - Buttons: "Install" (anchor `#install`) and "Run it on your servers" (anchor `#enterprise`).

2. **Switching** (heading "Switch agents. Keep the memory.")
   - Three panels: "Claude Code today", "Codex tomorrow", "A local model next year". Each panel shows the same two example memories: "We deploy with Coolify on Hetzner" and "Use Postgres advisory locks for the job queue".
   - Line under: "Your memory lives in your database, not in a vendor's chat history. Change tools whenever you want."

3. **DataFlow** (heading "What leaves your network")
   - Table, columns: "Default (OpenRouter)", "Private cloud (Bedrock, Azure, Vertex via a gateway)", "Fully local (Ollama)".
   - Rows:
     - "Memories": Your Postgres / Your Postgres / Your Postgres
     - "Model calls (extraction, summaries)": OpenRouter, then the model vendor / Your cloud account / Your server
     - "Embeddings": OpenRouter / Your cloud account / Your server
     - "Your coding agent": Its vendor / Its vendor, or your cloud if the agent supports it / Its vendor, or a local model if the agent supports it
   - Note under table: "CEMS does not change what your coding agent sends to its vendor. If that matters, pick an agent that runs on open-weight models (see Works with)."
   - One simple diagram: agent → CEMS server → Postgres, with the model box labelled "OpenRouter, your cloud, or local Ollama".

4. **HowItWorks** (heading "How it works"). Three steps:
   - "Install the hooks. One command adds CEMS to Claude Code, Cursor, Codex or Goose."
   - "Work as usual. CEMS picks out decisions, preferences and fixes from your sessions."
   - "Get it back. Relevant memories arrive at the start of each session and when you ask."

5. **WorksWith** (heading "Works with"). Table, columns "Agent", "Memory integration", "Private model path":
   - Claude Code / Tested (hooks + MCP) / Enterprise cloud (Bedrock, Azure, Vertex)
   - Cursor / Tested (MCP) / Enterprise cloud keys (Bedrock, Azure)
   - Codex CLI / Tested (MCP) / Open-weight (`--oss` with Ollama), enterprise cloud
   - Goose / Tested (MCP) / Open-weight, enterprise cloud
   - OpenCode / MCP / Open-weight, enterprise cloud
   - Aider, Cline, Continue, Roo Code, Kilo Code / MCP / Open-weight, enterprise cloud

6. **Proof**: "Runs at Hubstaff across 50 engineers since July 2026." Beside it two stats: "98% Recall@5" and "<50ms injection latency".

7. **Install** (heading "Install"). Three tabs (plain HTML radio or `<details>`-free tab buttons with minimal inline JS, keyboard accessible):
   - "Solo developer": `curl -fsSL https://getcems.com/install.sh | bash`
   - "Team server": `git clone https://github.com/chocksy/cems && cd cems/deploy && cp .env.example .env && docker compose up -d`
   - "Private mode": `curl -fsSL https://getcems.com/install-server.sh | bash -s -- --private --yes` with the note "Runs the LLM and embeddings on the same box with Ollama. Cloud-init files for AWS, Hetzner and DigitalOcean are in `deploy/cloud-init/`." Link "Private mode guide" to `https://github.com/chocksy/cems/blob/main/docs/DEPLOYMENT.md#private-mode`.
   - Each tab has a copy button (reuse the existing copy logic from `InstallSnippet.astro`, restyled).

8. **Enterprise** (heading "We'll set it up for you")
   - "We install it in your cloud. You pay for your servers. One-time setup fee."
   - Button "Talk to us" → `mailto:hello@getcems.com?subject=CEMS%20private%20install`.

9. **FAQ** (rewrite the `faqs` array; JSON-LD regenerates from it):
   - "Where is my data stored?" → "Memories never leave your Postgres. Model calls go to OpenRouter by default, or to your own models in private mode."
   - "Does my coding agent still send code to its vendor?" → "Yes. CEMS only controls its own traffic. Claude Code, Cursor and Codex still talk to their vendors. Agents like Goose, OpenCode and Aider can run on open-weight models if you need everything in-house."
   - "Can I switch agents and keep the memory?" → "Yes. The memory lives on your CEMS server. Point a new agent at it and the same memories show up."
   - "Does CEMS work with my current tools?" → "Claude Code, Cursor, Codex and Goose are tested. Any MCP client can connect."
   - "How is this different from CLAUDE.md?" → keep the existing answer, fix em dashes.
   - "Will it slow down my AI assistant?" → keep the existing answer.
   - "What happens if I uninstall?" → keep the existing answer, fix em dashes.

**Meta:**
- Title: "CEMS: portable memory for AI coding agents, on your servers"
- Description: "CEMS gives your team one memory across Claude Code, Cursor, Codex and Goose. Self-hosted, with an optional fully local mode on your own models."
- Update `softwareSchema.description` to the same description and `softwareVersion` to the current cems version (read `pyproject.toml` in the cems repo).
- `WebSite` schema name: "CEMS".

**Nav:** links "How it works", "Works with", "Install", "Enterprise", "FAQ", "GitHub"; CTA "Install" → `#install`. Replace the light ocean nav colours with the neutral palette.

**Redirect:** add to `public/_redirects`:
```
/install-server.sh https://raw.githubusercontent.com/chocksy/cems/main/install-server.sh 302
```

**Palette:** background near-black neutral (e.g. `#0a0a0a`), surfaces `#141414`, borders `rgba(255,255,255,0.08)`, text `#ededed` / `#a1a1a1`, one accent (emerald `#34d399`). Geist Sans for text, Geist Mono for code. Remove the Google Fonts Bricolage link.

- [ ] **Step 1:** Delete ocean assets and components; rewrite `Layout.astro` and `global.css` to the neutral palette.
- [ ] **Step 2:** Build the section components with the copy above.
- [ ] **Step 3:** `npm run build` passes; `grep -rni "ocean" src public` returns nothing; `grep -rn "—" src` returns nothing.
- [ ] **Step 4:** `npm run preview` and run Lighthouse accessibility (`npx lighthouse http://localhost:4321 --only-categories=accessibility --chrome-flags="--headless" --quiet --output=json | jq .categories.accessibility.score`), score 0.9 or above. Take a full-page screenshot for the reviewer.
- [ ] **Step 5:** Commit.
