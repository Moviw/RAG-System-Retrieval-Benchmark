import asyncio
from pathlib import Path

import typer

from app.evaluation.runner import run_benchmark

app = typer.Typer(help="Run retrieval benchmarks.")


@app.command()
def main(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    payload = asyncio.run(run_benchmark(config))
    typer.echo(f"Benchmark complete: run_id={payload['run_id']}")
    typer.echo(f"Results directory: benchmarks/results/{payload['run_id']}")


if __name__ == "__main__":
    app()
