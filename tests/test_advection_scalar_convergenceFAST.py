# tests/test_advection_scalar_convergence.py
#
# Accuracy test driven by the tests/test_case_advection_scalar/ case directory. It loads that
# case's control.yaml, mesh, and initial_condition.py, and sweeps ONLY the polynomial degree.
#
# Physics of the check: for a PERIODIC domain with advection_velocity * final_time equal to a
# whole number of domain lengths, the exact solution at final_time equals the initial condition.
# The L2 error between the final and initial fields is therefore a pure accuracy measure, and for
# a high-order DG method on a smooth field it must fall SPECTRALLY (exponentially) with poly_deg.
#
# Produces a table in the terminal (use `pytest -s`) and a semilog convergence plot in
# tests/test_case_advection_scalar/outputs/figures/.
import numpy as np
import pytest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless backend: save figures without a display (CI-safe)
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

from dingus.config import load_case_yaml
from dingus.initialConditions.initialize_solution import initialize
from dingus.mesh import mesh_class
from dingus.timeIntegrators.timeIntegration import compute_dt, time_step

# This test drives a specific case directory (the standard pytest-per-case pattern).
CASE_DIR  = Path(__file__).resolve().parent / "test_case_advection_scalar"
CTRL_FILE = CASE_DIR / "control.yaml"
FIG_DIR   = CASE_DIR / "outputs" / "figures"


def _roundtrip_l2_error(poly_deg: int) -> float:
    '''
    Load the case, override only poly_deg, advect one period, and return the relative L2 error
    between the final and initial fields. Paths (mesh, IC) are resolved relative to the control
    file's directory, exactly as the runner does.
    '''
    cfg = load_case_yaml(CTRL_FILE)
    cfg.mesh.poly_deg = poly_deg                       # the only swept parameter

    mesh = mesh_class.Mesh()
    mesh.read_mesh(CASE_DIR / cfg.mesh.mesh_file)
    mesh.construct_mesh(cfg)
    initialize(cfg, CASE_DIR / str(cfg.initialization.IC_file), mesh)

    q0 = [e.solution.copy() for e in mesh.elements]    # exact solution at final_time == initial

    T = cfg.time_stepping.final_time
    t = 0.0
    while t < T:
        dt = min(compute_dt(mesh, cfg), T - t)
        time_step(mesh, cfg, t, dt)
        t += dt

    num = den = 0.0
    for e, qi in zip(mesh.elements, q0):
        w    = e.jacobian_det * mesh.quad_weights[:, None] * mesh.quad_weights[None, :]
        num += np.sum(w * (e.solution[..., 0] - qi[..., 0])**2)
        den += np.sum(w * qi[..., 0]**2)
    return float(np.sqrt(num / den))


def _print_table(degrees: list, errors: list, slope: float) -> None:
    '''Pretty-print the convergence data as a rich table (visible with `pytest -s`).'''
    table = Table(title="Scalar advection: periodic round-trip convergence")
    table.add_column("poly_deg", justify="right")
    table.add_column("L2 error", justify="right")
    table.add_column("reduction", justify="right")
    for i, (P, err) in enumerate(zip(degrees, errors)):
        ratio = "-" if i == 0 else f"{errors[i - 1] / err:6.2f}x"
        table.add_row(str(P), f"{err:.4e}", ratio)
    Console().print(table)
    Console().print(f"  log10(error) slope: [bold]{slope:.3f}[/bold] per degree "
                    f"(more negative = faster spectral decay)")


def _plot_convergence(degrees: list, errors: list, slope: float) -> Path:
    '''Save a semilog (log error vs degree) convergence plot; a straight line == spectral decay.'''
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(degrees, errors, 'o-', color="tab:blue", label="DGSEM round-trip error")
    # Least-squares exponential fit line, for visual reference of the spectral rate.
    fit = 10.0 ** np.polyval(np.polyfit(degrees, np.log10(errors), 1), degrees)
    ax.semilogy(degrees, fit, '--', color="tab:gray", label=f"fit: slope {slope:.2f}/degree")
    ax.set_xlabel("Polynomial degree  P")
    ax.set_ylabel("Relative $L_2$ error")
    ax.set_title("Scalar advection: spectral (p-)convergence")
    ax.set_xticks(degrees)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    out_path = FIG_DIR / "scalar_advection_convergence.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_scalar_advection_spectral_convergence():
    '''Periodic round-trip L2 error must fall spectrally (exponentially) with poly_deg.'''

    # Guard: the round-trip is only exact if the case is fully periodic and the bump translates a
    # whole number of domain lengths. Fail loudly if control.yaml isn't set up for it.
    cfg  = load_case_yaml(CTRL_FILE)
    disp = np.asarray(cfg.physics.advection_velocity, dtype=float) * cfg.time_stepping.final_time
    assert all(bc.type == 'periodic' for bc in cfg.boundary_conditions.values()), \
        "round-trip convergence test requires all-periodic boundaries in control.yaml"
    assert np.allclose(disp, np.round(disp)), \
        f"round-trip requires advection_velocity * final_time to be whole domain lengths, got {disp}"

    degrees = [2, 3, 4, 5]
    errors  = [_roundtrip_l2_error(P) for P in degrees]
    slope   = float(np.polyfit(degrees, np.log10(errors), 1)[0])

    # Report: table to the terminal + a convergence plot saved under outputs/figures/.
    _print_table(degrees, errors, slope)
    fig_path = _plot_convergence(degrees, errors, slope)
    Console().print(f"  convergence plot saved to: [green]{fig_path}[/green]")

    # 1. finite and strictly decreasing with polynomial degree
    assert all(np.isfinite(errors)), f"non-finite error in {errors}"
    assert all(errors[i + 1] < errors[i] for i in range(len(errors) - 1)), \
        f"error not decreasing with poly_deg: {errors}"

    # 2. spectral (exponential) convergence: log10(error) drops at least ~0.6 per degree
    assert slope < -0.6, f"convergence slope {slope:.3f}/degree is not spectral"

    # 3. sanity: the finest resolution is already quite accurate
    assert errors[-1] < 1e-2, f"finest-degree error {errors[-1]:.2e} unexpectedly large"
