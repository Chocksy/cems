# Supermemory Claude Code Plugin - Article Analysis

**Source**: https://x.com/DhravyaShah/status/2017039283367137690
**Author**: Dhravya Shah (@DhravyaShah)
**Date**: Jan 30, 2026

---

## Full Article Text

We added supermemory to Claude Code. It's INSANELY powerful now...

Today, we are launching the Supermemory plugin for Claude Code!

TLDR: You can use supermemory in claude code now. - https://github.com/supermemoryai/claude-supermemory

Claude code has genuinely changed how I work. But there's this one thing that drives me crazy... Every day, I have to explain the same exact things to claude code. I have to keep repeating my coding style, preferences, etc.

"The user service connects to Postgres, not MySQL."
"Don't refactor that function—I know it looks ugly but there's a reason it's like that."

Claude writes great code. Then I close the session, and it forgets everything.

Next day? Groundhog Day. Again. We have built all of these workarounds - Massive CLAUDE.md files, copy pasting the context at the start of every prompt, maintaining "memory" documents that feel like the agent is never looking at them..

After our success with the clawd bot plugin and the opencode plugins, we knew that we're in the right spot to do something about it.

So we built something.

At Supermemory, we've been working on memory infrastructure for AI agents for a while now. We power memory for tens of thousands of AI applications. And we kept hearing the same thing from developers: "I wish my coding agent actually remembered stuff."

Today we're launching a Supermemory plugin for Claude Code.

**The idea is simple: Claude Code should know you. Not just for this session—forever. It should know your codebase, your preferences, your team's decisions, and the context from every tool you use.**

## Key Features

### 1. It remembers where you left off

We utilize **user profiles** in supermemory to create a profile of you, which contains both **episodic content** about you, as well as the **"static" information**. Claude knows that this week, your entire goal is to drive the costs down and migrate to another postgres provider.

### 2. It learns your style

Instead of writing slop code just like everyone else, it will learn as you use it - like "Use less useEffects!!!".

Claude code will now remember exactly how you fixed an error last time, and this knowledge compounds into an agent that feels truly insanely customized for your use case... slowly.

### 3. It knows YOU

It knows that you're a founder, or college student, or a system engineer, and will suggest tools and practices accordingly. Claude code learns your requirements, your style, your taste. Because taste is the #1 thing that differentiates good engineering vs bad.

**Example:**
> Developer: "I need to add rate limiting to this endpoint"
> Agent: "Based on the rate limiting you implemented in the payments-api last month (using sliding window with Redis), and your preference for the express-rate-limit middleware, here's an approach that matches your existing patterns..."

---

## Technical Architecture: Hybrid Memory

Most "memory" solutions for AI are just RAG—retrieve some similar documents and stuff them in the context. That works for knowledge bases. **It doesn't work for memory.**

Memory isn't just "find similar stuff." It's understanding that when you say "the auth bug," you mean the specific issue you've been debugging for three days. It's knowing that your preferences have evolved—you used to like classes, now you prefer functions. It's tracking that a decision was made, then revisited, then changed.

**We built a system that actually:**
- Extracts facts
- Tracks how they change over time
- Builds a profile of you that's always current
- Retrieves the right context at the right moment

Not just similar context—**relevant context**.

---

## Benchmark Results

> The benchmark we use for this (LongMemEval) puts us at **81.6%**. For comparison, most RAG systems score in the 40-60% range on memory-specific tasks.

---

## How is this different from the MCP?

The supermemory MCP is great for things like this, but comes with one big limitation: **We cannot control when claude code chooses to run the tools.** This means that we have no control / data point to learn things from, and a memory system is only good if there's things to recall later.

**This plugin adds:**

1. **Context Injection**: On session start, a User Profile is automatically injected into Claude's context
2. **Automatic Capture**: Conversation turns are captured and stored for future context

Both of these things were not possible with the MCP before.

---

## Links

- GitHub: https://github.com/supermemoryai/claude-supermemory
- Discord: https://supermemory.link/discord

---

## Key Takeaways for CEMS

1. **User Profile Injection** - They inject a "User Profile" at session start containing:
   - Episodic content (recent activities, current goals)
   - Static information (preferences, style, role)

2. **Hybrid Memory** - Not just RAG, but:
   - Fact extraction
   - Temporal tracking (preferences evolve)
   - Profile building (always current)
   - Relevance over similarity

3. **LongMemEval Score: 81.6%** - This is likely Recall@5 on the same benchmark we use

4. **Automatic capture** - They capture conversation turns automatically, not just when tools are called
