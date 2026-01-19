# CEMS User Onboarding Guide

Welcome to CEMS (Continuous Evolving Memory System)! This guide will help you connect your IDE to your company's shared memory system.

## What is CEMS?

CEMS is a memory system that helps AI assistants (Claude, Cursor, etc.) remember:
- **Personal memories**: Your preferences, patterns, decisions
- **Shared memories**: Team conventions, RSpec patterns, architecture decisions

Think of it as a shared brain for your team's AI tools.

## Quick Setup (5 minutes)

### Step 1: Get Your Credentials

Ask your DevOps team or team lead for:
- CEMS server URL (e.g., `https://cems.yourcompany.com`)
- CEMS API key (for authentication)
- Your username and team ID

### Step 2: Configure Claude Code

Add CEMS to your Claude Code MCP configuration:

```bash
# Open Claude Code config
code ~/.claude.json
```

Find the `mcpServers` section (or create one) and add:

```json
{
  "mcpServers": {
    "cems": {
      "type": "http",
      "url": "https://cems.yourcompany.com/mcp",
      "headers": {
        "X-API-Key": "your-api-key-from-devops",
        "X-User-ID": "your.username",
        "X-Team-ID": "your-team"
      }
    }
  }
}
```

### Step 3: Restart Claude Code

```bash
# Restart to load the new configuration
claude --restart
```

### Step 4: Verify Connection

In Claude Code, try:

```
/recall team conventions
```

You should see shared memories from your team.

## Alternative: Cursor IDE Setup

For Cursor users:

```bash
# Open Cursor settings
code ~/.cursor/mcp.json
```

Add:

```json
{
  "mcpServers": {
    "cems": {
      "type": "http",
      "url": "https://cems.yourcompany.com/mcp",
      "headers": {
        "X-API-Key": "your-api-key-from-devops",
        "X-User-ID": "your.username",
        "X-Team-ID": "your-team"
      }
    }
  }
}
```

## Using CEMS

### Basic Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/remember` | Add personal memory | `/remember I prefer descriptive variable names` |
| `/share` | Add team memory | `/share Our API uses snake_case for JSON keys` |
| `/recall` | Search memories | `/recall How do we write RSpec tests?` |
| `/forget` | Remove a memory | `/forget abc123` |

### Example Workflow

#### Starting a new task

```
/recall authentication patterns
```

Claude will retrieve relevant memories about how your team handles auth.

#### Learning something new

```
/share We use JWT tokens with 24h expiry for API authentication
```

This becomes available to all team members.

#### Personal preferences

```
/remember I prefer using Rubocop's compact style for modules
```

Only you will see this memory.

### Searching Effectively

```
# Search both personal and shared
/recall database migrations

# Search only shared memories
/recall --scope shared RSpec conventions

# Search by category
/recall --category architecture microservices
```

## Understanding Memory Types

### Personal Memories

- **Scope**: Only you can see
- **Examples**:
  - "I prefer Python over JavaScript for scripting"
  - "My PR template uses bullet points"
  - "I like to write tests first"

### Shared Memories

- **Scope**: Everyone on your team
- **Examples**:
  - "Our RSpec tests use FactoryBot for fixtures"
  - "API responses follow JSON:API spec"
  - "Database migrations must be reversible"

### Pinned Memories

- **Scope**: Shared, but never auto-deleted
- **Source**: Usually from repository indexing
- **Examples**:
  - Architecture Decision Records (ADRs)
  - Coding guidelines from README
  - RSpec configuration and helpers

## What Gets Indexed Automatically?

Your DevOps team has likely indexed your repositories. This means CEMS already knows about:

| Content | What's Extracted |
|---------|------------------|
| `README.md` | Project setup, conventions |
| `CONTRIBUTING.md` | Contribution guidelines |
| `spec/spec_helper.rb` | RSpec configuration |
| `spec/support/*.rb` | Custom matchers, helpers |
| `.rubocop.yml` | Code style rules |
| `docs/adr/*.md` | Architecture decisions |
| `.github/workflows/*.yml` | CI/CD patterns |

## Best Practices

### 1. Be Specific

```
# Good
/share RSpec shared examples for API authentication: use `include_context 'authenticated user'`

# Too vague
/share We have RSpec helpers
```

### 2. Use Categories

```
/share --category testing We mock external APIs using WebMock
/share --category architecture Services communicate via RabbitMQ
```

### 3. Check Before Adding

```
# First check if it already exists
/recall external API mocking

# Then add if needed
/share We use WebMock to stub external HTTP calls in tests
```

### 4. Pin Important Memories

If you create a memory that should never decay:

```
# Ask your team lead to pin it
cems pin abc123 --reason "Core testing convention" --category convention
```

## Troubleshooting

### "Memory not found"

- Check your API key is correct
- Verify server URL
- Try `/context` to see system status

### "Cannot add shared memory"

- You need team membership
- Contact your DevOps team

### Slow responses

- Large memories take longer
- Try narrowing your search with `--scope` or `--category`

### Connection errors

- Check network/VPN
- Verify server URL
- Check if server is healthy: `curl https://cems.yourcompany.com/health`

## Getting Help

- **Team Lead**: For access issues
- **DevOps**: For server/connection problems
- **#cems-support**: Slack channel for questions

## FAQ

**Q: Can others see my personal memories?**
A: No. Personal memories are completely private.

**Q: What happens to old memories?**
A: Regular memories are archived after 90 days of no access. Pinned memories never expire.

**Q: Can I delete shared memories?**
A: Only team admins can delete shared memories.

**Q: How do I know what's already in memory?**
A: Use `/recall` with broad queries, or ask Claude "What do you know about X?"

**Q: Does this work offline?**
A: No, CEMS requires connection to the server.
