"""Rule commands for constitution/playbook memories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from rich.table import Table

from cems.cli_utils import console, get_client, handle_error
from cems.client import CEMSClientError

KIND_CHOICES = click.Choice(["constitution", "playbook"], case_sensitive=False)
SCOPE_CHOICES = click.Choice(["personal", "shared"], case_sensitive=False)
CATEGORY_CHOICES = click.Choice(["guidelines", "project"], case_sensitive=False)

DEFAULT_FOUNDATION_FILE = Path("docs/constitution/foundation_memory_seed.json")
DEFAULT_PLAYBOOK_FILE = Path("docs/constitution/playbook_memory_seed.json")


def _kind_defaults(kind: str) -> dict[str, Any]:
    if kind == "constitution":
        return {
            "category": "guidelines",
            "base_tags": ["foundation", "constitution"],
            "source_ref": "foundation:constitution:v2",
            "pin_reason": "foundational constitution memory",
        }

    return {
        "category": "guidelines",
        "base_tags": ["playbook", "operational"],
        "source_ref": "playbook:operational:v1",
        "pin_reason": "operational playbook memory",
    }


def _normalize_principle_code(code: str) -> str:
    digits = "".join(ch for ch in code if ch.isdigit())
    if digits:
        return f"{int(digits):02d}"
    return code.strip()


def _normalize_rule_code(code: str) -> str:
    return "".join(ch for ch in code.strip().upper() if ch.isalnum() or ch in "-_")


def _parse_csv_tags(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [tag.strip() for tag in raw.split(",") if tag.strip()]


def _dedupe_tags(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        normalized = tag.strip().lower().replace(" ", "-")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _build_rule_content(kind: str, code: str, title: str, statement: str) -> str:
    statement = statement.strip()
    title = title.strip()

    if kind == "constitution":
        prefix = ""
        if code:
            prefix = f"Foundation {code}"
            if title:
                prefix += f" {title}"
            prefix += ": "
        elif title:
            prefix = f"{title}: "
        return f"{prefix}{statement}".strip()

    prefix = "Playbook"
    if code:
        prefix += f" {code}"
    if title:
        prefix += f" {title}"
    prefix += ": "
    return f"{prefix}{statement}".strip()


def _default_bundle_file(kind: str) -> Path:
    if kind == "constitution":
        return DEFAULT_FOUNDATION_FILE
    return DEFAULT_PLAYBOOK_FILE


def _load_bundle_file(bundle_file: Path) -> dict[str, Any]:
    if not bundle_file.exists():
        raise CEMSClientError(f"Bundle file not found: {bundle_file}")

    try:
        data = json.loads(bundle_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CEMSClientError(f"Invalid JSON in bundle file: {e}") from e

    memories = data.get("memories")
    if not isinstance(memories, list) or not memories:
        raise CEMSClientError("Bundle file must contain a non-empty 'memories' array")

    for idx, memory in enumerate(memories, start=1):
        if not isinstance(memory, dict):
            raise CEMSClientError(f"Memory #{idx} must be an object")
        content = memory.get("content")
        if not isinstance(content, str) or not content.strip():
            raise CEMSClientError(f"Memory #{idx} is missing non-empty string 'content'")

    return data


def _extract_event(result: dict[str, Any]) -> tuple[str, str]:
    rows = (result.get("result") or {}).get("results") or []
    row = rows[0] if rows else {}
    return str(row.get("event", "UNKNOWN")), str(row.get("id", "-"))


@click.group("rule")
def rule() -> None:
    """Create rules and load reusable rule bundles."""


@rule.command("add")
@click.option("--kind", type=KIND_CHOICES, help="Rule kind (constitution or playbook)")
@click.option("--scope", type=SCOPE_CHOICES, help="Memory scope (default from prompt)")
@click.pass_context
def rule_add(ctx: click.Context, kind: str | None, scope: str | None) -> None:
    """Interactive rule wizard."""
    try:
        client = get_client(ctx)

        selected_kind = (kind or "").strip().lower()
        if not selected_kind:
            selected_kind = click.prompt(
                "Rule kind",
                type=KIND_CHOICES,
                default="constitution",
                show_choices=True,
            ).lower()

        defaults = _kind_defaults(selected_kind)

        rule_code = ""
        title = ""
        if selected_kind == "constitution":
            raw_code = click.prompt(
                "Principle number (optional, e.g. 13)",
                default="",
                show_default=False,
            )
            if raw_code:
                rule_code = _normalize_principle_code(raw_code)
            title = click.prompt(
                "Short title (optional)",
                default="",
                show_default=False,
            )
        else:
            raw_code = click.prompt(
                "Playbook code (optional, e.g. U1 or A1)",
                default="",
                show_default=False,
            )
            if raw_code:
                rule_code = _normalize_rule_code(raw_code)
            title = click.prompt(
                "Short title (optional)",
                default="",
                show_default=False,
            )

        statement = click.prompt("Rule statement")
        category = click.prompt(
            "Category",
            type=CATEGORY_CHOICES,
            default=str(defaults["category"]),
            show_choices=True,
        ).lower()
        selected_scope = (scope or "").strip().lower()
        if not selected_scope:
            selected_scope = click.prompt(
                "Scope",
                type=SCOPE_CHOICES,
                default="personal",
                show_choices=True,
            ).lower()

        source_ref = click.prompt(
            "Source reference",
            default=str(defaults["source_ref"]),
            show_default=True,
        ).strip()

        extra_tags_raw = click.prompt(
            "Extra tags (comma-separated, optional)",
            default="",
            show_default=False,
        )

        tags = list(defaults["base_tags"])
        if selected_kind == "constitution" and rule_code.isdigit():
            tags.append(f"principle:{rule_code}")
            tags.append(f"rule-id:p{rule_code}")
        elif selected_kind == "playbook" and rule_code:
            tags.append(f"rule-id:{rule_code.lower()}")
        tags.extend(_parse_csv_tags(extra_tags_raw))
        tags = _dedupe_tags(tags)

        content = _build_rule_content(
            kind=selected_kind,
            code=rule_code,
            title=title,
            statement=statement,
        )

        pinned = click.confirm("Pin memory?", default=True)
        pin_reason = None
        if pinned:
            pin_reason = click.prompt(
                "Pin reason",
                default=str(defaults["pin_reason"]),
                show_default=True,
            ).strip()

        table = Table(title="Rule Preview")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Kind", selected_kind)
        table.add_row("Scope", selected_scope)
        table.add_row("Category", category)
        table.add_row("Source Ref", source_ref)
        table.add_row("Tags", ", ".join(tags) if tags else "-")
        table.add_row("Pinned", "yes" if pinned else "no")
        table.add_row("Content", content)
        console.print(table)

        if not click.confirm("Add this rule?", default=True):
            console.print("[yellow]Cancelled[/yellow]")
            return

        result = client.add(
            content=content,
            category=category,
            scope=selected_scope,  # type: ignore[arg-type]
            tags=tags,
            source_ref=source_ref or None,
            pinned=pinned,
            pin_reason=pin_reason or None,
        )
        event, memory_id = _extract_event(result)
        console.print(f"[green]Rule stored[/green] (event={event}, id={memory_id})")

    except CEMSClientError as e:
        handle_error(e)


@rule.command("load")
@click.option(
    "--file",
    "bundle_file",
    type=click.Path(path_type=Path, exists=False, dir_okay=False),
    help="Path to rule bundle JSON file",
)
@click.option(
    "--kind",
    type=KIND_CHOICES,
    help="Load default bundle for kind when --file is omitted",
)
@click.option("--scope", type=SCOPE_CHOICES, help="Override scope for all memories")
@click.option("--dry-run", is_flag=True, help="Preview entries without writing memories")
@click.pass_context
def rule_load(
    ctx: click.Context,
    bundle_file: Path | None,
    kind: str | None,
    scope: str | None,
    dry_run: bool,
) -> None:
    """Load a reusable rule bundle."""
    try:
        selected_kind = (kind or "").strip().lower()
        selected_file = bundle_file

        if selected_file is None:
            if not selected_kind:
                selected_kind = click.prompt(
                    "Rule bundle kind",
                    type=KIND_CHOICES,
                    default="constitution",
                    show_choices=True,
                ).lower()
            selected_file = _default_bundle_file(selected_kind)

        data = _load_bundle_file(selected_file)
        memories: list[dict[str, Any]] = data["memories"]

        default_scope = (scope or str(data.get("scope", "personal"))).lower()
        if default_scope not in {"personal", "shared"}:
            raise CEMSClientError(f"Invalid scope in bundle file: {default_scope}")

        console.print(f"Bundle name: {data.get('name', 'unnamed')}")
        console.print(f"Source: {data.get('source', 'unknown')}")
        console.print(f"Entries: {len(memories)}")
        console.print(f"Scope: {default_scope}")
        console.print(f"Bundle file: {selected_file}")
        console.print(f"Mode: {'dry-run' if dry_run else 'apply'}")

        if dry_run:
            for i, memory in enumerate(memories, start=1):
                content = str(memory["content"]).strip()
                tags = memory.get("tags") or []
                category = memory.get("category", "guidelines")
                console.print(f"[{i:02d}] {category} tags={tags} content={content[:96]}")
            return

        client = get_client(ctx)
        counts = {"ADD": 0, "DUPLICATE": 0, "ERROR": 0, "OTHER": 0}

        for i, memory in enumerate(memories, start=1):
            tags = memory.get("tags") if isinstance(memory.get("tags"), list) else []
            payload_scope = str(memory.get("scope", default_scope)).lower()
            if scope:
                payload_scope = default_scope
            if payload_scope not in {"personal", "shared"}:
                payload_scope = default_scope

            result = client.add(
                content=str(memory["content"]),
                category=str(memory.get("category", "guidelines")),
                scope=payload_scope,  # type: ignore[arg-type]
                tags=tags,
                source_ref=memory.get("source_ref"),
                pinned=bool(memory.get("pinned", True)),
                pin_reason=memory.get("pin_reason") or f"{data.get('name', 'bundle')} memory",
            )
            event, memory_id = _extract_event(result)
            if event in counts:
                counts[event] += 1
            else:
                counts["OTHER"] += 1
            console.print(f"[{i:02d}] {event:<9} id={memory_id}")

        total = sum(counts.values())
        console.print("\nSummary")
        console.print(f"- Processed: {total}")
        console.print(f"- Added: {counts['ADD']}")
        console.print(f"- Duplicates: {counts['DUPLICATE']}")
        console.print(f"- Errors: {counts['ERROR']}")
        console.print(f"- Other: {counts['OTHER']}")

    except CEMSClientError as e:
        handle_error(e)
