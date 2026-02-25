"""Debug dashboard command — local web UI for hook transparency."""

import webbrowser

import click


@click.command()
@click.option("--port", default=8767, help="Port for the debug server")
@click.option("--no-open", is_flag=True, help="Don't auto-open browser")
def debug(port: int, no_open: bool) -> None:
    """Launch the debug dashboard — see what CEMS injects into Claude.

    Starts a local web server that reads hook event logs and shows
    a timeline of sessions, memory retrievals, and injected context.

    No auth needed — reads local log files only.
    """
    from cems.debug.server import serve

    if not no_open:
        # Open browser before blocking on serve_forever
        import threading
        threading.Timer(0.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    serve(port=port)
