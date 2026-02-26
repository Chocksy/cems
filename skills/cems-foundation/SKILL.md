---
name: cems-foundation
description: View foundation guidelines (rules, principles, constraints) stored in CEMS memory
---

# View Foundation Guidelines

Search CEMS memory for foundation guidelines — stable rules and principles that apply across all sessions.

## Prerequisites

CEMS MCP server must be configured. See [github.com/Chocksy/cems](https://github.com/Chocksy/cems) for setup.

## Usage

When this skill is invoked, or the user asks about project rules/guidelines:

1. **Detect the current project** from the working directory:
   - Run `git remote get-url origin` to extract `org/repo` format
   - SSH: `git@github.com:org/repo.git` → `org/repo`
   - HTTPS: `https://github.com/org/repo.git` → `org/repo`

2. **Search for foundation guidelines** via MCP:
   ```json
   {
     "tool": "memory_search",
     "arguments": {
       "query": "foundation guidelines rules principles constitution",
       "max_results": 20,
       "scope": "both",
       "project": "org/repo"
     }
   }
   ```

3. **Filter results** to only include memories that:
   - Have tags containing "foundation" or "constitution"
   - OR have category "guidelines"

4. **Present results** clearly:
   ```
   ## Foundation Guidelines

   Found N foundation guidelines:

   1. <guideline content>
   2. <guideline content>
   ...

   These are your foundational principles. They apply to all sessions.
   ```

## Context

Foundation guidelines are stable rules that apply across all sessions. They are typically:
- Coding standards and conventions
- Project-specific constraints
- Team agreements and principles
- Quality gates and review criteria

They differ from regular memories in that they are meant to be always-on constraints, not just recalled information.
