from pathlib import Path

import typer

app = typer.Typer(help="Run retrieval benchmarks.")


@app.command()
def main(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    typer.echo(f"Benchmark runner scaffold ready: {config}")


if __name__ == "__main__":
    app()
