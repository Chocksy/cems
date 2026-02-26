# getcems.com Website Research

> Research date: 2026-02-20

## Domain

- **Primary**: `getcems.com` (purchased, on Cloudflare)
- **Also owned**: `evolving-memory.com` (could redirect to getcems.com)

## How steipete builds his sites (14 repos analyzed)

Peter picks the simplest tool for the job. Three tiers:

| Tier | Sites | Stack | Hosting |
|------|-------|-------|---------|
| **Full blog** | steipete.me | Astro 5 + Tailwind 4 + React + TS + pagefind | Vercel |
| **Product landing** | openclaw.ai | Astro 5 + custom CSS (no Tailwind) | Vercel |
| **Next.js app** | demark.md | Next.js 16 + Tailwind + shadcn/ui | unknown |
| **CLI splash pages** | gifgrep, trimmy, codexbar, repobar, songsee | Single HTML + CSS in `docs/` | GitHub Pages |
| **Legacy/simple** | clawdbot.com, gogcli, SOUL.md | Jekyll | GitHub Pages |

Key deps on steipete.me: `astro ^5.16`, `tailwindcss ^4.1`, `@tailwindcss/vite`, `@astrojs/react`, `@pagefind/default-ui`, `@astrojs/mdx`, `@astrojs/sitemap`, `@astrojs/rss`, `satori` (OG images), `sharp`

## openclaw.ai deep dive

Repo: https://github.com/openclaw/openclaw.ai (public, 171 stars)

### Stack
- **Astro 5** with `output: 'static'` (fully pre-rendered)
- **Zero UI frameworks** — no React, no Tailwind, no component library
- **Pure custom CSS** with CSS custom properties, `clamp()` responsive type, glassmorphism (`backdrop-filter: blur()`)
- **Minimal deps**: `astro`, `@lucide/astro` (icons), `simple-icons` (brand logos), `@vercel/analytics`

### Structure
```
src/
  components/    # Just Icon.astro
  content/       # Blog posts (content collections)
  data/          # JSON data files (testimonials, etc.)
  i18n/          # Translations
  layouts/       # Single Layout.astro
  lib/           # Utilities
  pages/
    index.astro        # Landing page (one big file)
    integrations.astro
    showcase.astro
    shoutouts.astro
    blog/
    trust/
    rss.xml.js
```

### Landing page sections (index.astro)
1. Background effects (CSS-only stars/nebula)
2. Hero — mascot SVG with animations, gradient title, tagline
3. Latest blog post banner
4. Testimonials carousel (two rows, opposite scroll directions)
5. Quick-start code block (tabbed installer: npm/curl/git/macOS)
6. Features grid (3x2 cards linking to docs)
7. Integrations pill row (brand icons)
8. Press section (editorial quotes)
9. CTA grid (Discord, Docs, GitHub, etc.)
10. Newsletter signup (Buttondown embed)
11. Sponsors
12. Footer

### CSS approach
- CSS custom properties for theming (`--coral-bright`, `--bg-deep`)
- `clamp()` for responsive typography
- `backdrop-filter: blur()` on cards (glassmorphism)
- `@media (prefers-reduced-motion: reduce)` kills animations
- Mobile breakpoints at 480px and 640px

## Hosting: Cloudflare Pages (free)

Docs: https://docs.astro.build/en/guides/deploy/cloudflare/

### Why Cloudflare Pages
- **Free tier**: unlimited bandwidth, global CDN, automatic SSL
- **Git integration**: push to `main` → auto-deploy in ~2 min
- **Custom domain**: getcems.com already on Cloudflare → DNS setup is automatic
- **Preview deployments**: every PR gets a preview URL
- **No adapter needed** for static output — just build and serve `dist/`

### Deploy steps
1. Create Astro project, push to GitHub
2. Cloudflare dashboard → Workers & Pages → Create → Connect to Git
3. Select repo, set framework preset to Astro, build command `npm run build`
4. Add custom domain `getcems.com`
5. Every push to `main` auto-deploys

## getlate.dev analysis (SEO reference)

**Stack**: Next.js App Router + Tailwind + Vercel
**Why it's SEO-performant**:

### Page sections
1. Nav (logo, links, language selector, 2 CTAs)
2. Hero — H1 + value prop + code snippet + OAuth CTA + "no credit card" reassurance
3. Platform logos grid (13 icons, each links to dedicated docs page)
4. Social proof banner — company logos + animated live counter ("301K posts this week")
5. Testimonials carousel (6 quotes, infinite scroll)
6. How it works (3-step numbered guide)
7. Feature cards (5 API endpoint cards with HTTP method labels)
8. FAQ accordion (5 items)
9. Final CTA with GDPR trust badge
10. Footer (5-column: Product, Integrations, Company, Comparisons, Tools)

### SEO tactics
- **5 JSON-LD schemas on homepage**: Organization, WebSite, SoftwareApplication (w/ AggregateRating), HowTo, FAQPage
- **Programmatic SEO**: ~600+ pages across 6 languages — platform pages, API pages, comparison pages, tool pages, error pages
- **Single H1**, clean heading hierarchy, semantic HTML
- **Self-hosted font** (single WOFF2, no Google Fonts)
- **Server-side analytics proxy** (`sst.getlate.dev`) — avoids ad-blockers
- **17 competitor comparison pages** (vs Hootsuite, Buffer, etc.)
- **Content velocity**: ~180 blog posts, ~25 changelog entries

### Design patterns worth stealing
- Code snippet in hero (developer trust signal)
- Live animated counter (social proof)
- HTTP method labels on feature cards (`POST /posts`, `GET /analytics`)
- GDPR trust badge near CTA
- Testimonials with real names + company affiliations

---

## Component framework research

### Ranked options for reusable Astro + Tailwind landing page components

| # | Framework | Stars | Components | Approach | License | Price |
|---|-----------|-------|-----------|----------|---------|-------|
| 1 | **AstroWind** | 5,500 | 22 widgets | Fork template, use data-driven widgets | MIT | Free |
| 2 | **Starwind Pro** | 515 | 190+ blocks, 43 UI primitives | CLI install per block | MIT/Paid | $89-179 |
| 3 | **Fulldev UI** | 533 | 100+ components + blocks | shadcn-style CLI | MIT | Free |
| 4 | **Astroship** | 1,900 | 9 sections | Fork template | GPL-3.0 | Free |

### AstroWind (recommended base)
22 data-driven widget components — import and configure with props:
- Hero (3 variants), Features (3 variants), Steps (2 variants)
- Pricing, Testimonials, CallToAction, Contact, FAQs, Stats, Brands
- Content, Note, Announcement, Header, Footer, BlogHighlightedPosts
- All Astro 5 + Tailwind v4 + dark mode + RTL
- PageSpeed 100 across the board
- v2.0 planned: extract widgets into standalone `@astrowind/` packages

### Starwind Pro (if you need variety)
190+ blocks across 24 categories: 14 hero variants, 25 feature variants, 14 pricing variants, 10 CTA, 8 testimonial, 9 footer, 6 FAQ, etc.
CLI: `npx starwind@latest add @starwind-pro/hero-1`
Worth evaluating the 45 free blocks before buying.

---

## Decisions (settled)

- [x] **Separate repo** (`getcems.com`) — clean deploy pipeline
- [x] **Tailwind 4** — faster iteration, matches steipete.me approach
- [x] **Cloudflare Pages** — domain already on CF, free tier, auto-deploy
- [x] **No blog at launch** — ship landing page first, add later
- [x] **Cloudflare Web Analytics** — free, privacy-friendly, zero JS bundle
- [x] **AstroWind widgets as base** — fork, strip demo, keep widget architecture
- [ ] Evaluate Starwind Pro free blocks for extra variety

## Sections to build (inspired by getlate.dev + openclaw.ai)

1. **Nav** — CEMS logo, links (Docs, GitHub, Pricing?), CTA button
2. **Hero** — H1 tagline, value prop paragraph, install code snippet, CTA
3. **Integrations row** — Claude Code, Cursor, Codex, Goose logos
4. **Social proof** — "X memories stored" counter or testimonial quotes
5. **How it works** — 3-step: Install → Configure → Remember (like getlate.dev)
6. **Features grid** — 4-6 cards: persistent memory, observer, hooks, search, MCP
7. **Code example** — syntax-highlighted API call or hook output
8. **FAQ accordion** — 4-5 common questions
9. **Final CTA** — "Start remembering" + install command
10. **Footer** — GitHub, docs, social links

## SEO plan (from getlate.dev playbook)

- JSON-LD schemas: Organization, SoftwareApplication, FAQPage, HowTo
- Single H1, semantic heading hierarchy
- Self-hosted font (single WOFF2)
- Sitemap via `@astrojs/sitemap`
- Future: comparison pages (vs Mem0, vs Zep, etc.), integration-specific pages
