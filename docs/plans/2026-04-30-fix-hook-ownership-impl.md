# Hook Ownership Rewrite — Implementation Plan (v0.13.3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CEMS authoritatively own its hook entries in `~/.claude/settings.json` so future template changes (starting with dropping the `run_with_uv.sh` wrapper) actually reach existing users on `cems update`.

**Architecture:** Add a `_strip_cems_hook_entries` function that removes any hook whose command references a `cems_*.py` script (hook-level granularity, preserving non-CEMS hooks in mixed entries). Wire it into `_merge_settings` so each setup/update strips CEMS-owned hooks then re-appends fresh template entries. Delete the now-redundant `_migrate_removed_hooks`. Update the template to drop the wrapper invocation in favor of the existing `#!/usr/bin/env -S uv run --script` shebang.

**Tech Stack:** Python 3.11+, pytest, uv. No new dependencies.

**Spec:** [docs/plans/2026-04-30-fix-hook-ownership-plan.md](2026-04-30-fix-hook-ownership-plan.md)

---

## File Structure

| File | Purpose | Action |
|------|---------|--------|
| `src/cems/commands/setup.py` | Hook installer & settings merger | Modify (`_merge_settings`, add `_strip_cems_hook_entries`, delete `_migrate_removed_hooks`) |
| `src/cems/data/claude/settings.json` | Hook command template merged into user's settings | Modify (drop `run_with_uv.sh` from all 5 commands) |
| `tests/test_setup.py` | Existing setup tests | Modify (add `TestStripCemsHookEntries` and `TestMergeSettings` classes) |
| `pyproject.toml` | Version | Modify (bump `0.13.2` → `0.13.3`) |
| `uv.lock` | Locked deps | Modify (auto-updated by uv) |

---

## Task 1: Add `_strip_cems_hook_entries` with the basic stale-wrapper case

**Files:**
- Modify: `src/cems/commands/setup.py` (add new function near the existing `_migrate_*` helpers, around line 612)
- Modify: `tests/test_setup.py` (add `TestStripCemsHookEntries` class)

- [ ] **Step 1: Add the failing test**

Append to `tests/test_setup.py`:

```python
class TestStripCemsHookEntries:
    """Tests for _strip_cems_hook_entries — removes CEMS-owned hook entries
    from a settings hooks dict so the caller can re-append fresh template
    entries.
    """

    def test_strips_stale_run_with_uv_wrapper(self):
        """A SessionStart hook using the old run_with_uv.sh wrapper must
        be removed so the next merge can replace it with the new form.
        """
        from cems.commands.setup import _strip_cems_hook_entries

        hooks = {
            "SessionStart": [
                {
                    "matcher": "",
                    "hooks": [
                        {
                            "type": "command",
                            "command": (
                                "$HOME/.claude/hooks/run_with_uv.sh "
                                "$HOME/.claude/hooks/cems_session_start.py"
                            ),
                        }
                    ],
                }
            ]
        }

        _strip_cems_hook_entries(hooks)

        # The CEMS hook is gone. The matcher had no surviving hooks, so
        # the entry is dropped. The event had no surviving entries, so
        # the event key is dropped.
        assert "SessionStart" not in hooks
```

- [ ] **Step 2: Run test, verify it fails**

Run: `uv run pytest tests/test_setup.py::TestStripCemsHookEntries::test_strips_stale_run_with_uv_wrapper -v`

Expected: FAIL with `ImportError: cannot import name '_strip_cems_hook_entries' from 'cems.commands.setup'` (function doesn't exist yet).

- [ ] **Step 3: Implement minimal `_strip_cems_hook_entries`**

In `src/cems/commands/setup.py`, add this function immediately after `_migrate_old_hook_names` (so around line 612, before `_migrate_removed_hooks`):

```python
def _strip_cems_hook_entries(hooks: dict) -> None:
    """Remove CEMS-owned hook entries from a hooks dict.

    A hook command is "owned by CEMS" if any whitespace-separated token
    ends in a ``cems_*.py`` filename. This catches every form CEMS has
    ever shipped:

    - ``$HOME/.claude/hooks/run_with_uv.sh $HOME/.claude/hooks/cems_X.py``
    - ``$HOME/.claude/hooks/cems_X.py``
    - ``uv run --script $HOME/.claude/hooks/cems_X.py``

    Stripping is hook-level: a single entry with both a CEMS hook and a
    user's custom hook keeps the custom one. Entries whose ``hooks`` list
    becomes empty are dropped. Events whose entries array becomes empty
    are deleted from the dict.

    Mutates ``hooks`` in place.
    """
    def _is_cems_command(cmd: str) -> bool:
        for token in cmd.split():
            name = token.rsplit("/", 1)[-1]
            if name.startswith("cems_") and name.endswith(".py"):
                return True
        return False

    for event_name in list(hooks.keys()):
        new_entries = []
        for entry in hooks[event_name]:
            kept_hooks = [
                h for h in entry.get("hooks", [])
                if not _is_cems_command(h.get("command", ""))
            ]
            if kept_hooks:
                entry["hooks"] = kept_hooks
                new_entries.append(entry)
        if new_entries:
            hooks[event_name] = new_entries
        else:
            del hooks[event_name]
```

- [ ] **Step 4: Run test, verify it passes**

Run: `uv run pytest tests/test_setup.py::TestStripCemsHookEntries -v`

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cems/commands/setup.py tests/test_setup.py
git commit -m "feat(setup): add _strip_cems_hook_entries for owned-hook removal"
```

---

## Task 2: Lock in mixed-entry preservation (non-CEMS hook stays)

**Files:**
- Modify: `tests/test_setup.py` (extend `TestStripCemsHookEntries`)

- [ ] **Step 1: Add the failing test**

Append inside `class TestStripCemsHookEntries`:

```python
    def test_preserves_non_cems_hook_in_mixed_entry(self):
        """A matcher that has both a CEMS hook and a user's custom hook
        must keep the custom hook; only the CEMS hook is stripped.
        """
        from cems.commands.setup import _strip_cems_hook_entries

        hooks = {
            "SessionStart": [
                {
                    "matcher": "",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "$HOME/.claude/hooks/cems_session_start.py",
                        },
                        {
                            "type": "command",
                            "command": "$HOME/bin/my_audit_log.sh",
                        },
                    ],
                }
            ]
        }

        _strip_cems_hook_entries(hooks)

        # Event survives because the user's custom hook survives.
        assert list(hooks.keys()) == ["SessionStart"]
        assert len(hooks["SessionStart"]) == 1
        kept = hooks["SessionStart"][0]["hooks"]
        assert len(kept) == 1
        assert kept[0]["command"] == "$HOME/bin/my_audit_log.sh"
```

- [ ] **Step 2: Run test, verify it passes**

Run: `uv run pytest tests/test_setup.py::TestStripCemsHookEntries::test_preserves_non_cems_hook_in_mixed_entry -v`

Expected: 1 passed (the hook-level filter from Task 1 already handles this).

If the test FAILS, your Task 1 implementation strips at entry level — fix it to filter at the hook level, then re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_setup.py
git commit -m "test(setup): lock in non-CEMS hook preservation in mixed entries"
```

---

## Task 3: Lock in empty-entry pruning and empty-event removal

**Files:**
- Modify: `tests/test_setup.py` (extend `TestStripCemsHookEntries`)

- [ ] **Step 1: Add the failing test**

Append inside `class TestStripCemsHookEntries`:

```python
    def test_prunes_empty_entries_and_empty_event_keys(self):
        """An entry whose only hook was CEMS gets dropped (no orphan
        ``{"matcher": "", "hooks": []}``). An event whose only entry was
        pure-CEMS gets removed from the dict entirely.
        """
        from cems.commands.setup import _strip_cems_hook_entries

        hooks = {
            "SessionStart": [
                {
                    "matcher": "",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "$HOME/.claude/hooks/cems_session_start.py",
                        }
                    ],
                }
            ],
            "Stop": [
                {
                    "matcher": "",
                    "hooks": [
                        {"type": "command", "command": "$HOME/bin/my_other_hook.sh"}
                    ],
                }
            ],
        }

        _strip_cems_hook_entries(hooks)

        # SessionStart event removed entirely (only had a CEMS hook).
        assert "SessionStart" not in hooks
        # Stop preserved — non-CEMS hook stays.
        assert "Stop" in hooks
        assert len(hooks["Stop"]) == 1
```

- [ ] **Step 2: Run test, verify it passes**

Run: `uv run pytest tests/test_setup.py::TestStripCemsHookEntries::test_prunes_empty_entries_and_empty_event_keys -v`

Expected: 1 passed (Task 1's implementation already does both prunings).

- [ ] **Step 3: Commit**

```bash
git add tests/test_setup.py
git commit -m "test(setup): lock in empty-entry pruning and event removal"
```

---

## Task 4: Lock in deprecated `cems_post_tool_use.py` capture

**Files:**
- Modify: `tests/test_setup.py` (extend `TestStripCemsHookEntries`)

- [ ] **Step 1: Add the failing test**

Append inside `class TestStripCemsHookEntries`:

```python
    def test_strips_deprecated_post_tool_use_hook(self):
        """The disabled tool-learning hook (cems_post_tool_use.py) must be
        captured by the detector — it follows the cems_*.py pattern, so
        no special-case handler is needed.
        """
        from cems.commands.setup import _strip_cems_hook_entries

        hooks = {
            "PostToolUse": [
                {
                    "matcher": "",
                    "hooks": [
                        {
                            "type": "command",
                            "command": (
                                "$HOME/.claude/hooks/run_with_uv.sh "
                                "$HOME/.claude/hooks/cems_post_tool_use.py"
                            ),
                        }
                    ],
                }
            ]
        }

        _strip_cems_hook_entries(hooks)

        assert "PostToolUse" not in hooks
```

- [ ] **Step 2: Run test, verify it passes**

Run: `uv run pytest tests/test_setup.py::TestStripCemsHookEntries::test_strips_deprecated_post_tool_use_hook -v`

Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_setup.py
git commit -m "test(setup): lock in deprecated cems_post_tool_use.py capture"
```

---

## Task 5: Wire strip into `_merge_settings`, delete `_migrate_removed_hooks`

**Files:**
- Modify: `src/cems/commands/setup.py` (replace the dedup loop in `_merge_settings`, delete `_migrate_removed_hooks`, update the now-stale comment)
- Modify: `tests/test_setup.py` (new `TestMergeSettings` class)

- [ ] **Step 1: Add the failing integration test**

Append to `tests/test_setup.py` (new class):

```python
class TestMergeSettings:
    """Integration tests for _merge_settings — full strip-and-replace flow."""

    def _template(self, tmp_path: Path) -> Path:
        """Write a minimal CEMS template settings.json (W2 form)."""
        template = {
            "hooks": {
                "SessionStart": [
                    {
                        "matcher": "",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "$HOME/.claude/hooks/cems_session_start.py",
                            }
                        ],
                    }
                ]
            }
        }
        path = tmp_path / "template.json"
        path.write_text(json.dumps(template))
        return path

    def test_rewrites_stale_run_with_uv_reference_to_template_form(self, tmp_path):
        """Given a settings.json with the old run_with_uv.sh wrapper command,
        _merge_settings must rewrite it to the new bare-path form from the
        template.
        """
        from cems.commands.setup import _merge_settings

        claude_dir = tmp_path
        settings_file = claude_dir / "settings.json"
        settings_file.write_text(json.dumps({
            "hooks": {
                "SessionStart": [
                    {
                        "matcher": "",
                        "hooks": [
                            {
                                "type": "command",
                                "command": (
                                    "$HOME/.claude/hooks/run_with_uv.sh "
                                    "$HOME/.claude/hooks/cems_session_start.py"
                                ),
                            }
                        ],
                    }
                ]
            }
        }))

        _merge_settings(claude_dir, self._template(tmp_path))

        result = json.loads(settings_file.read_text())
        commands = [
            h["command"]
            for entry in result["hooks"]["SessionStart"]
            for h in entry["hooks"]
        ]
        assert commands == ["$HOME/.claude/hooks/cems_session_start.py"]
        # Defensive: no surviving wrapper reference anywhere under hooks.
        assert "run_with_uv.sh" not in json.dumps(result["hooks"])
```

- [ ] **Step 2: Run test, verify it fails**

Run: `uv run pytest tests/test_setup.py::TestMergeSettings::test_rewrites_stale_run_with_uv_reference_to_template_form -v`

Expected: FAIL — current `_merge_settings` is additive; it sees `cems_session_start.py` in both existing and template, dedups, and keeps the stale command. Failure assertion will be on the `commands == [...]` equality.

- [ ] **Step 3: Replace the dedup loop in `_merge_settings`**

Open `src/cems/commands/setup.py`. Locate the current `_merge_settings` body starting at the line `existing_hooks = existing.setdefault("hooks", {})` (around line 670). Replace everything from that line down to the end of the function (just before `settings_file.write_text(...)`) with:

```python
    existing_hooks = existing.setdefault("hooks", {})

    # Step 1: rename pre-prefix legacy hook commands so the strip detector catches them.
    _migrate_old_hook_names(existing_hooks)

    # Step 2: remove every CEMS-owned hook (any form, any wrapper).
    _strip_cems_hook_entries(existing_hooks)

    # Step 3: append fresh template entries. Strip has already removed any CEMS
    # hooks, so appending never duplicates. Non-CEMS entries in the same event
    # are preserved.
    for event_name, cems_entries in cems_hooks.items():
        existing_hooks.setdefault(event_name, []).extend(cems_entries)
```

Then **delete** the entire `_migrate_removed_hooks` function (now redundant — the strip detector matches `cems_post_tool_use.py` and the template doesn't include it). Find it at the line `def _migrate_removed_hooks(hooks: dict) -> bool:` and delete the whole function body through its `return changed` line.

Also delete the call site inside `_merge_settings` if it's still there:

```python
    # Remove deprecated hooks (tool learning superseded by observer daemon)
    if _migrate_removed_hooks(existing_hooks):
        console.print("  Removed tool learning hook (superseded by observer daemon)")
```

The replacement block above already omits this; if your edit landed cleanly, this paragraph is a no-op verification step.

- [ ] **Step 4: Run test, verify it passes**

Run: `uv run pytest tests/test_setup.py::TestMergeSettings::test_rewrites_stale_run_with_uv_reference_to_template_form -v`

Expected: 1 passed.

- [ ] **Step 5: Run the full setup test class to confirm no regression**

Run: `uv run pytest tests/test_setup.py -v`

Expected: all pass (existing MCP-registration tests untouched).

- [ ] **Step 6: Commit**

```bash
git add src/cems/commands/setup.py tests/test_setup.py
git commit -m "refactor(setup): _merge_settings owns CEMS hooks via strip-and-replace

Replaces the additive dedup loop with strip-then-append. Deletes
_migrate_removed_hooks (deprecated cems_post_tool_use.py is now
captured by the cems_*.py detector and naturally not re-added since
the template omits it). Future hook command changes reach existing
users on the next 'cems update'."
```

---

## Task 6: Lock in idempotency

**Files:**
- Modify: `tests/test_setup.py` (extend `TestMergeSettings`)

- [ ] **Step 1: Add the test**

Append inside `class TestMergeSettings`:

```python
    def test_merge_settings_is_idempotent(self, tmp_path):
        """Running _merge_settings twice on the same starting file must
        produce identical output — the second invocation strips its own
        prior writes and re-appends, with no growth or duplication.
        """
        from cems.commands.setup import _merge_settings

        claude_dir = tmp_path
        settings_file = claude_dir / "settings.json"
        settings_file.write_text(json.dumps({"hooks": {}}))

        template_path = self._template(tmp_path)

        _merge_settings(claude_dir, template_path)
        first = settings_file.read_text()

        _merge_settings(claude_dir, template_path)
        second = settings_file.read_text()

        assert json.loads(first) == json.loads(second)
```

- [ ] **Step 2: Run test, verify it passes**

Run: `uv run pytest tests/test_setup.py::TestMergeSettings::test_merge_settings_is_idempotent -v`

Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_setup.py
git commit -m "test(setup): lock in _merge_settings idempotency"
```

---

## Task 7: Legacy un-prefixed names are migrated and stripped

**Files:**
- Modify: `tests/test_setup.py` (extend `TestMergeSettings`)

This locks in the interaction between `_migrate_old_hook_names` (removes the pre-prefix legacy entries like `stop.py`) and the new strip-and-replace flow (which then adds the `cems_stop.py` entry from the template). End state: no orphaned legacy reference, only the prefixed form.

- [ ] **Step 1: Add the test**

Append inside `class TestMergeSettings`:

```python
    def test_migrates_legacy_unprefixed_names(self, tmp_path):
        """A user whose settings still reference the pre-rename
        ~/.claude/hooks/stop.py (no cems_ prefix) must end up with
        cems_stop.py after merge — _migrate_old_hook_names removes the
        legacy entry and the template adds the prefixed one.
        """
        from cems.commands.setup import _merge_settings

        claude_dir = tmp_path
        settings_file = claude_dir / "settings.json"
        settings_file.write_text(json.dumps({
            "hooks": {
                "Stop": [
                    {
                        "matcher": "",
                        "hooks": [
                            {
                                "type": "command",
                                "command": (
                                    "$HOME/.claude/hooks/run_with_uv.sh "
                                    "$HOME/.claude/hooks/stop.py"
                                ),
                            }
                        ],
                    }
                ]
            }
        }))

        # Use a template that includes cems_stop.py so we can verify the
        # rename outcome.
        template_path = tmp_path / "template.json"
        template_path.write_text(json.dumps({
            "hooks": {
                "Stop": [
                    {
                        "matcher": "",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "$HOME/.claude/hooks/cems_stop.py",
                            }
                        ],
                    }
                ]
            }
        }))

        _merge_settings(claude_dir, template_path)

        result = json.loads(settings_file.read_text())
        commands = [
            h["command"]
            for entry in result["hooks"]["Stop"]
            for h in entry["hooks"]
        ]
        assert commands == ["$HOME/.claude/hooks/cems_stop.py"]
        # Defensive: no legacy reference survives.
        assert "/stop.py" not in json.dumps(result["hooks"])
```

- [ ] **Step 2: Run test, verify it passes**

Run: `uv run pytest tests/test_setup.py::TestMergeSettings::test_migrates_legacy_unprefixed_names -v`

Expected: 1 passed. (`_migrate_old_hook_names` removes the legacy entry; template adds `cems_stop.py`.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_setup.py
git commit -m "test(setup): lock in legacy un-prefixed hook name migration"
```

---

## Task 8: Update the Claude template — drop `run_with_uv.sh`

**Files:**
- Modify: `src/cems/data/claude/settings.json` (5 commands: drop the wrapper prefix)
- Modify: `tests/test_setup.py` (new `TestClaudeTemplate` class)

- [ ] **Step 1: Add the failing test**

Append to `tests/test_setup.py` (new class):

```python
class TestClaudeTemplate:
    """Sanity checks on the shipped src/cems/data/claude/settings.json."""

    def test_no_run_with_uv_wrapper_in_template(self):
        """The CEMS-shipped template must not reference run_with_uv.sh —
        we don't ship that script and direct invocation works (verified
        via probe on 2026-04-30).
        """
        from importlib.resources import files

        template_text = (
            files("cems.data.claude").joinpath("settings.json").read_text()
        )
        assert "run_with_uv.sh" not in template_text, (
            "Template still references run_with_uv.sh — drop it and rely "
            "on the cems_*.py shebang."
        )
```

- [ ] **Step 2: Run test, verify it fails**

Run: `uv run pytest tests/test_setup.py::TestClaudeTemplate::test_no_run_with_uv_wrapper_in_template -v`

Expected: FAIL — the template currently contains 5 occurrences of `run_with_uv.sh`.

- [ ] **Step 3: Update the template**

Open `src/cems/data/claude/settings.json`. Replace its **entire** contents with:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "$HOME/.claude/hooks/cems_session_start.py"
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "$HOME/.claude/hooks/cems_user_prompts_submit.py"
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "$HOME/.claude/hooks/cems_pre_tool_use.py"
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "$HOME/.claude/hooks/cems_stop.py"
          }
        ]
      }
    ],
    "PreCompact": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "$HOME/.claude/hooks/cems_pre_compact.py"
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 4: Run test, verify it passes**

Run: `uv run pytest tests/test_setup.py::TestClaudeTemplate -v`

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cems/data/claude/settings.json tests/test_setup.py
git commit -m "fix(setup): drop run_with_uv.sh wrapper from Claude hooks template

The wrapper script was never shipped by CEMS but the template
referenced it, so fresh installs got SessionStart hook errors
('No such file or directory') until the user manually created
the wrapper. Probe testing on 2026-04-30 confirmed Claude Code's
hook executor passes through the parent shell's PATH, so 'uv' is
directly resolvable and the cems_*.py shebang works on its own.

Existing users with stale wrapper references get rewritten on
next 'cems update' via the strip-and-replace _merge_settings."
```

---

## Task 9: Smoke test — every script the template references must exist

**Files:**
- Modify: `tests/test_setup.py` (extend `TestClaudeTemplate`)

- [ ] **Step 1: Add the test**

Append inside `class TestClaudeTemplate`:

```python
    def test_every_template_command_points_to_a_shipped_script(self):
        """For every hook command in the template, the referenced .py file
        must exist in src/cems/data/claude/hooks/. Catches the original
        bug class: template entries pointing at files we never ship.
        """
        from importlib.resources import files

        template = json.loads(
            files("cems.data.claude").joinpath("settings.json").read_text()
        )

        hooks_root = files("cems.data.claude.hooks")

        for event_name, entries in template["hooks"].items():
            for entry in entries:
                for hook in entry["hooks"]:
                    cmd = hook["command"]
                    # Extract the .py token (last whitespace-separated
                    # token ending in .py).
                    py_tokens = [
                        t for t in cmd.split() if t.endswith(".py")
                    ]
                    assert py_tokens, (
                        f"{event_name} command has no .py script: {cmd}"
                    )
                    script_name = py_tokens[-1].rsplit("/", 1)[-1]
                    shipped = hooks_root.joinpath(script_name)
                    assert shipped.is_file(), (
                        f"{event_name} references {script_name} but "
                        f"src/cems/data/claude/hooks/{script_name} does "
                        f"not exist — this would crash the user's hook."
                    )
```

- [ ] **Step 2: Run test, verify it passes**

Run: `uv run pytest tests/test_setup.py::TestClaudeTemplate::test_every_template_command_points_to_a_shipped_script -v`

Expected: 1 passed (Task 8's template only references the 5 `cems_*.py` files that already exist in `src/cems/data/claude/hooks/`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_setup.py
git commit -m "test(setup): smoke-test that template scripts ship with CEMS"
```

---

## Task 10: Bump version, run full suite, tag, push

**Files:**
- Modify: `pyproject.toml` (version `0.13.2` → `0.13.3`)
- Modify: `uv.lock` (auto-updated)

- [ ] **Step 1: Bump `pyproject.toml`**

In `pyproject.toml`, change line 3 from:

```
version = "0.13.2"
```

to:

```
version = "0.13.3"
```

- [ ] **Step 2: Sync lockfile**

Run: `uv sync --extra dev`

Expected: silent success; `uv.lock` shows the new version.

Verify with: `grep -A 1 'name = "cems"' uv.lock | head -3`

Expected: shows `version = "0.13.3"`.

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest --ignore=tests/test_server.py -q`

(The `--ignore=tests/test_server.py` matches the v0.13.2 release flow — `test_server.py` has unrelated in-progress modifications.)

Expected: all pass, no regressions. New tests from Tasks 1–9 are part of the count.

- [ ] **Step 4: Commit the bump**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: bump to 0.13.3 — hook ownership rewrite"
```

- [ ] **Step 5: Tag and push**

```bash
git tag v0.13.3
git push origin main v0.13.3
```

Expected: GitHub Actions `docker-publish.yml` workflow starts; `chocksy/cems-server:0.13.3` will be available within ~5 minutes. Python users on `uv tool install "cems @ git+..."` pick up the fix immediately on next `cems update`.

- [ ] **Step 6: Verify locally**

Reinstall the tool and run setup against a tmp `HOME` to confirm the new template lands cleanly:

```bash
uv tool install "cems @ /Volumes/External/Development/cems" --force
TMP=$(mktemp -d)
HOME="$TMP" cems setup --claude --api-url http://localhost:65535 --api-key test
grep -c 'run_with_uv.sh' "$TMP/.claude/settings.json" || echo "0 occurrences ✓"
```

Expected: `0 occurrences ✓`. The five hook commands point at `$HOME/.claude/hooks/cems_*.py`.

---

## Summary of expected outcomes

| Concern | Resolved by |
|---------|-------------|
| Alex's missing-script error | Task 8 (template no longer references `run_with_uv.sh`) |
| Existing users healing automatically | Task 5 (strip-and-replace `_merge_settings`) |
| Future hook command changes reaching users | Task 5 (same — no longer additive) |
| Deprecated `cems_post_tool_use.py` cleanup | Task 5 (handled by detector + missing template entry) |
| Regression protection | Tasks 1–4, 6, 7, 9 (seven unit tests covering strip semantics, idempotency, legacy migration, template-script existence) |

After v0.13.3 ships, Alex runs `cems update` once. His settings.json gets rewritten. His next session fires SessionStart with the new bare-path command, the Python hook runs via its `uv run --script` shebang, and the missing-script error is gone.
