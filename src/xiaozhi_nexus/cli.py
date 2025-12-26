from __future__ import annotations

import typer


app = typer.Typer(add_completion=False)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    reload: bool = typer.Option(False, help="Auto-reload on code changes"),
) -> None:
    import uvicorn

    uvicorn.run(
        "xiaozhi_nexus.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

