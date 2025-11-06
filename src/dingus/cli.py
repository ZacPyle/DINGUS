# src/dingus/cli.py
import typer, yaml
from rich import print
from .config import CaseCfg
from .runner import run_case

app = typer.Typer(help="DINGUS — DG spectral element CFD (Python)")

@app.command()
def run(case: str):
    with open(case, "r") as f:
        data = yaml.safe_load(f)
    cfg = CaseCfg(**data)
    print(f"[bold green]Running case:[/bold green] {cfg.meta.get('name', '(unnamed)')}")
    run_case(cfg)

if __name__ == "__main__":
    app()
