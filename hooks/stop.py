#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

"""
Stop Hook - Session End Handler

This hook runs when Claude stops and:
1. Logs session data
2. Sends transcript to CEMS for intelligent learning extraction
3. Announces completion via TTS
"""

import argparse
import json
import os
import re
import sys
import random
import subprocess
from pathlib import Path

# Import utilities
sys.path.insert(0, str(Path(__file__).parent))
from utils.constants import ensure_session_log_dir
from utils.credentials import get_cems_key, get_cems_url
from utils.hook_logger import log_hook_event

# CEMS configuration — env vars first, then ~/.cems/credentials fallback
CEMS_API_URL = get_cems_url()
CEMS_API_KEY = get_cems_key()


def get_project_id(cwd: str) -> str | None:
    """Extract project ID from git remote (e.g., 'org/repo')."""
    if not cwd:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            if url.startswith("git@"):
                match = re.search(r":(.+?)(?:\.git)?$", url)
            else:
                match = re.search(r"[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
            if match:
                return match.group(1).removesuffix('.git')
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_completion_messages():
    """Return list of friendly completion messages."""
    return [
        "Work complete!",
        "All done!",
        "Task finished!",
        "Job complete!",
        "Ready for next task!",
    ]


def get_tts_scripts():
    """
    Return ordered list of TTS scripts to try based on available API keys.
    Priority order: ElevenLabs > OpenAI > pyttsx3
    """
    script_dir = Path(__file__).parent
    tts_dir = script_dir / "utils" / "tts"
    scripts = []

    if os.getenv("ELEVENLABS_API_KEY"):
        elevenlabs_script = tts_dir / "elevenlabs_tts.py"
        if elevenlabs_script.exists():
            scripts.append(str(elevenlabs_script))

    if os.getenv("OPENAI_API_KEY"):
        openai_script = tts_dir / "openai_tts.py"
        if openai_script.exists():
            scripts.append(str(openai_script))

    pyttsx3_script = tts_dir / "pyttsx3_tts.py"
    if pyttsx3_script.exists():
        scripts.append(str(pyttsx3_script))

    return scripts


def get_llm_completion_message():
    """
    Generate completion message using available LLM services.
    Priority order: OpenAI > Anthropic > fallback to random message
    """
    script_dir = Path(__file__).parent
    llm_dir = script_dir / "utils" / "llm"

    if os.getenv("OPENAI_API_KEY"):
        oai_script = llm_dir / "oai.py"
        if oai_script.exists():
            try:
                result = subprocess.run(
                    ["uv", "run", str(oai_script), "--completion"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass

    if os.getenv("ANTHROPIC_API_KEY"):
        anth_script = llm_dir / "anth.py"
        if anth_script.exists():
            try:
                result = subprocess.run(
                    ["uv", "run", str(anth_script), "--completion"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass

    return random.choice(get_completion_messages())


def read_transcript(transcript_path: str) -> list[dict] | None:
    """Read transcript from .jsonl file and return as list of messages."""
    if not transcript_path or not os.path.exists(transcript_path):
        return None

    try:
        messages = []
        with open(transcript_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        messages.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return messages if messages else None
    except Exception:
        return None


def analyze_session(
    transcript: list[dict],
    session_id: str,
    working_dir: str | None = None,
    project: str | None = None,
) -> bool:
    """Send transcript to CEMS API for analysis.

    Returns True if the API call succeeded.
    """
    if not CEMS_API_KEY:
        return False

    try:
        import urllib.request
        import urllib.error

        payload = {
            "transcript": transcript,
            "session_id": session_id,
            "working_dir": working_dir,
        }
        if project:
            payload["source_ref"] = f"project:{project}"

        data = json.dumps(payload).encode('utf-8')

        req = urllib.request.Request(
            f"{CEMS_API_URL}/api/session/analyze",
            data=data,
            headers={
                "Authorization": f"Bearer {CEMS_API_KEY}",
                "Content-Type": "application/json"
            },
            method='POST'
        )

        # Use a short timeout since we're fire-and-forget
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status == 200

    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return False
    except Exception:
        return False


def _extract_transcript_text(transcript: list[dict]) -> str:
    """Extract readable text from transcript using shared extraction logic.

    Extracts user messages, assistant text, and tool action summaries.
    Uses the same logic as the observer daemon for consistency.
    """
    # Inline extraction to avoid import dependency on cems package in hooks
    SUMMARIZABLE_TOOLS = {
        "Read", "Edit", "Write", "Bash", "Grep", "Glob",
        "MultiEdit", "NotebookEdit", "WebFetch", "WebSearch",
    }

    def _summarize_tool(name, tool_input):
        if name not in SUMMARIZABLE_TOOLS or not isinstance(tool_input, dict):
            return None
        if name == "Read":
            p = tool_input.get("file_path", "")
            return f"Read: {p}" if p else None
        if name in ("Edit", "MultiEdit"):
            p = tool_input.get("file_path", "")
            return f"Edit: {p}" if p else None
        if name == "Write":
            p = tool_input.get("file_path", "")
            return f"Write: {p}" if p else None
        if name == "Bash":
            cmd = tool_input.get("command", "").strip().split("\n")[0][:120]
            return f"Bash: {cmd}" if cmd else None
        if name == "Grep":
            pat = tool_input.get("pattern", "")
            path = tool_input.get("path", ".")
            return f"Grep: '{pat}' in {path}" if pat else None
        if name == "Glob":
            pat = tool_input.get("pattern", "")
            return f"Glob: {pat}" if pat else None
        if name == "NotebookEdit":
            p = tool_input.get("notebook_path", "")
            return f"NotebookEdit: {p}" if p else None
        if name == "WebFetch":
            url = tool_input.get("url", "")
            if url:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    return f"WebFetch: {domain}" if domain else f"WebFetch: {url[:80]}"
                except Exception:
                    return f"WebFetch: {url[:80]}"
            return None
        if name == "WebSearch":
            q = tool_input.get("query", "")
            return f"WebSearch: '{q}'" if q else None
        return None

    def _compact_lines(raw_lines):
        """Compact consecutive [TOOL] lines into [ACTIVITY] summaries."""
        result = []
        tool_buf = []

        def _flush(buf):
            if not buf:
                return []
            reads, edits, writes, bash_cmds, searches, web = [], [], [], [], [], []
            for tl in buf:
                c = tl[7:]  # Strip "[TOOL] "
                if c.startswith("Read: "): reads.append(c[6:])
                elif c.startswith("Edit: "): edits.append(c[6:])
                elif c.startswith("Write: "): writes.append(c[7:])
                elif c.startswith("Bash: "): bash_cmds.append(c[6:])
                elif c.startswith(("Grep: ", "Glob: ")): searches.append(c)
                elif c.startswith(("WebFetch: ", "WebSearch: ")): web.append(c)
                elif c.startswith("NotebookEdit: "): edits.append(c[14:])

            acts = []
            for verb, paths in [("Read", reads), ("Edited", edits), ("Created", writes)]:
                if not paths:
                    continue
                if len(paths) == 1:
                    acts.append(f"{verb} {paths[0]}")
                else:
                    try:
                        dirs = [os.path.dirname(p) for p in paths]
                        common = os.path.commonpath(dirs) if all(dirs) else ""
                    except ValueError:
                        common = ""
                    names = [os.path.basename(p) for p in paths]
                    if common and common not in (".", "/"):
                        if len(names) <= 3:
                            acts.append(f"{verb} {len(paths)} files in {common}/ ({', '.join(names)})")
                        else:
                            acts.append(f"{verb} {len(paths)} files in {common}/")
                    elif len(names) <= 3:
                        acts.append(f"{verb} {len(paths)} files ({', '.join(names)})")
                    else:
                        acts.append(f"{verb} {len(paths)} files across multiple directories")

            if bash_cmds:
                if len(bash_cmds) == 1:
                    acts.append(f"Ran: {bash_cmds[0]}")
                else:
                    fw = [c.split()[0] if c.split() else "?" for c in bash_cmds]
                    if len(set(fw)) == 1:
                        acts.append(f"Ran {len(bash_cmds)} {fw[0]} commands")
                    elif len(bash_cmds) <= 3:
                        acts.append(f"Ran commands: {'; '.join(c[:60] for c in bash_cmds)}")
                    else:
                        acts.append(f"Ran {len(bash_cmds)} shell commands")

            if searches:
                if len(searches) == 1:
                    acts.append(f"Searched: {searches[0]}")
                else:
                    acts.append(f"Searched codebase ({len(searches)} queries)")

            for w in web:
                if w.startswith("WebFetch: "): acts.append(f"Visited {w[10:]}")
                elif w.startswith("WebSearch: "): acts.append(f"Web search: {w[11:]}")

            return [f"[ACTIVITY] {a}" for a in acts]

        for line in raw_lines:
            if line.startswith("[TOOL] "):
                tool_buf.append(line)
            else:
                if tool_buf:
                    result.extend(_flush(tool_buf))
                    tool_buf = []
                result.append(line)
        if tool_buf:
            result.extend(_flush(tool_buf))
        return result

    lines = []
    for msg in transcript:
        msg_type = msg.get("type", "")
        if msg_type not in ("user", "assistant"):
            continue
        if msg.get("isMeta"):
            continue

        message = msg.get("message", {})
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")

        if msg_type == "user":
            if isinstance(content, str):
                text = content.strip()[:2000]
                if text and len(text) > 5 and not text.startswith("<") and not text.startswith("/"):
                    lines.append(f"[USER]: {text}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "").strip()[:2000]
                        if text and len(text) > 10 and not text.startswith("Base directory for this skill:"):
                            lines.append(f"[USER]: {text}")

        elif msg_type == "assistant":
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    bt = block.get("type", "")
                    if bt == "text":
                        text = block.get("text", "").strip()[:1000]
                        if text and len(text) > 20:
                            lines.append(f"[ASSISTANT]: {text}")
                    elif bt == "tool_use":
                        summary = _summarize_tool(block.get("name", ""), block.get("input", {}))
                        if summary:
                            lines.append(f"[TOOL] {summary}")

    lines = _compact_lines(lines)
    return "\n\n".join(lines)


def observe_session(
    transcript: list[dict],
    session_id: str,
    working_dir: str | None = None,
    project: str | None = None,
) -> bool:
    """Send transcript to CEMS API for observation extraction.

    Compresses transcript to user/assistant text + tool summaries and sends
    to /api/session/observe for high-level observation extraction.

    Returns True if the API call succeeded.
    """
    if not CEMS_API_KEY:
        return False

    try:
        import urllib.request
        import urllib.error

        content = _extract_transcript_text(transcript)
        if len(content) < 200:
            return False  # Too short for meaningful observations

        # Cap at ~25k tokens
        content = content[:100_000]

        project_context = f"{project} ({working_dir})" if project else (working_dir or "unknown")
        payload = {
            "content": content,
            "session_id": session_id,
            "project_context": project_context,
        }
        if project:
            payload["source_ref"] = f"project:{project}"

        data = json.dumps(payload).encode('utf-8')

        req = urllib.request.Request(
            f"{CEMS_API_URL}/api/session/observe",
            data=data,
            headers={
                "Authorization": f"Bearer {CEMS_API_KEY}",
                "Content-Type": "application/json"
            },
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status == 200

    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return False
    except Exception:
        return False


def announce_completion():
    """Announce completion, trying each TTS provider until one succeeds."""
    try:
        tts_scripts = get_tts_scripts()
        if not tts_scripts:
            return

        completion_message = get_llm_completion_message()

        for tts_script in tts_scripts:
            try:
                result = subprocess.run(
                    ["uv", "run", tts_script, completion_message],
                    capture_output=True,
                    timeout=15,
                )
                if result.returncode == 0:
                    return  # Success, stop trying
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                continue  # Try next provider

    except (FileNotFoundError, Exception):
        pass


def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--chat", action="store_true", help="Copy transcript to chat.json"
        )
        args = parser.parse_args()

        # Read JSON input from stdin
        input_data = json.load(sys.stdin)

        # Extract required fields
        session_id = input_data.get("session_id", "")
        transcript_path = input_data.get("transcript_path", "")
        cwd = input_data.get("cwd", os.getcwd())

        log_hook_event("Stop", session_id, {"cwd": cwd}, input_data=input_data)

        # Ensure session log directory exists
        log_dir = ensure_session_log_dir(session_id)
        log_path = log_dir / "stop.json"

        # Read existing log data or initialize empty list
        if log_path.exists():
            with open(log_path, "r") as f:
                try:
                    log_data = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    log_data = []
        else:
            log_data = []

        # Append new data
        log_data.append(input_data)

        # Write back to file with formatting
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        # Handle --chat switch
        if args.chat and transcript_path:
            transcript = read_transcript(transcript_path)
            if transcript:
                chat_file = log_dir / "chat.json"
                with open(chat_file, "w") as f:
                    json.dump(transcript, f, indent=2)

        # Send transcript to CEMS for intelligent analysis + observation
        if transcript_path:
            transcript = read_transcript(transcript_path)
            if transcript and len(transcript) > 2:  # Skip very short sessions
                project = get_project_id(cwd)
                analyze_session(
                    transcript=transcript,
                    session_id=session_id,
                    working_dir=cwd,
                    project=project,
                )
                observe_session(
                    transcript=transcript,
                    session_id=session_id,
                    working_dir=cwd,
                    project=project,
                )

        # Announce completion via TTS
        announce_completion()

        sys.exit(0)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
