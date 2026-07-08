# src/dingus/cli.py
from pathlib import Path

import typer
from rich import print

from dingus.config import load_case_yaml
from .runner import run_case

app = typer.Typer(help="DINGUS — DG spectral element CFD (Python)")


@app.callback()
def main():
    """DINGUS command-line interface. Use a subcommand (e.g. `dingus run <case.yaml>`)."""
    # A no-op callback forces Typer to require an explicit subcommand name, so `run` stays a
    # subcommand (`dingus run <case>`) instead of being promoted to the default command.
    pass


@app.command()
def run(case: str):
    """Run a case from its control YAML file (path provided by 'case')."""
    # Resolve the control file path; mesh_file / IC_file / output_dir are relative to its directory.
    case_path = Path(case)

    # Construct the case configuration object from the control file
    cfg       = load_case_yaml(case_path)

    # Feed the case configuration into the runner.py (aka the driver script)
    print(f"[bold green]Running case:[/bold green] {cfg.name}")
    run_case(cfg, case_path.parent)


if __name__ == "__main__":
    app()
