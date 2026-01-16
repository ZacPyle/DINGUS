# src/dingus/cli.py
import typer
from rich import print
from .config import load_case_yaml
from .runner import run_case

app = typer.Typer(help="DINGUS — DG spectral element CFD (Python)")


@app.command()
def run(case: str):
    """Run a case YAML file (path provided by `case`)."""
    cfg = load_case_yaml(case)
    print(f"[bold green]Running case:[/bold green] {cfg.name}")
    run_case(cfg)

if __name__ == "__main__":
    app()
