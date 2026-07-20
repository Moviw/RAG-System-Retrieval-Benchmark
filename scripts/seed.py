from pathlib import Path

import typer

app = typer.Typer(help="Seed benchmark datasets.")


@app.command()
def main(config: Path = typer.Option(..., "--config", exists=True)) -> None:
    typer.echo(f"Seed scaffold ready: {config}")


if __name__ == "__main__":
    app()
