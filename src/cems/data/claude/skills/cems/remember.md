# /remember

Add a memory to your personal memory store.

## Usage

```
/remember <fact or information to remember>
```

## Examples

```
/remember I prefer Python for backend development
/remember The database schema uses snake_case for column names
/remember User authentication uses JWT tokens with 24h expiry
```

## How It Works

This skill uses the CEMS `memory_add` MCP tool to store information in your personal memory namespace. The memory system automatically:

1. Extracts atomic facts from your input
2. Checks for existing similar memories
3. Decides whether to ADD (new), UPDATE (modify existing), or skip (duplicate)
4. Stores with timestamp and access tracking

## Options

You can add categories and tags for better organization:

```
/remember --category preferences I like dark mode in all editors
/remember --category decisions We chose PostgreSQL for the main database
/remember --tags auth,security Session tokens expire after 1 hour of inactivity
```

## MCP Tool Used

`memory_add` with:
- `content`: Your input
- `scope`: "personal"
- `category`: Default "general" or specified
- `tags`: As specified
- `source_ref`: Project reference (see below)

## Project Context

When running in a git repository, **always include the project reference** for project-scoped recall:

1. Get git remote: `git remote get-url origin`
2. Extract org/repo from URL:
   - SSH: `git@github.com:org/repo.git` → `org/repo`
   - HTTPS: `https://github.com/org/repo.git` → `org/repo`
3. Pass as source_ref: `source_ref: "project:org/repo"`

This enables project-scoped memory recall - memories created in a project will be boosted when searching from that same project, and deprioritized when searching from other projects.

**Example:**
```
/remember --category decisions We use PostgreSQL for the main database
```

If running in the `myorg/webapp` repo, Claude should automatically add:
```json
{
  "content": "We use PostgreSQL for the main database",
  "scope": "personal",
  "category": "decisions",
  "source_ref": "project:myorg/webapp"
}
```

## Related Skills

- `/recall` - Search your memories
- `/forget` - Remove a memory
- `/share` - Add to shared team memory
