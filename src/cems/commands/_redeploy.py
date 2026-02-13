"""Internal helper: re-deploy hooks/skills without prompting for credentials.

Called by `cems update` in a subprocess so it uses the latest installed code.
Not a user-facing command.
"""

import sys

from cems.commands.setup import _get_data_path, _install_claude_hooks, _install_cursor_hooks
from cems.cli_utils import console


def main() -> None:
    args = sys.argv[1:]
    data_path = _get_data_path()

    if "--claude" in args:
        console.print("[bold blue]Claude Code[/bold blue]")
        _install_claude_hooks(data_path)

    if "--cursor" in args:
        console.print("[bold blue]Cursor[/bold blue]")
        _install_cursor_hooks(data_path)


if __name__ == "__main__":
    main()
