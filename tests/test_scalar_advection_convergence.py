# tests/test_scalar_advection_convergence.py
'''
Accuracy test driven by the tests/test_case_scalar_advection/ case directory. It loads that
case's control.yaml, mesh, and initial_condition.py, and sweeps ONLY the polynomial degree.

Physics of the check: for a PERIODIC domain with advection_velocity * final_time equal to a
whole number of domain lengths, the exact solution at final_time equals the initial condition.
The L2 error between the final and initial fields is therefore a pure accuracy measure, and for
a high-order DG method on a smooth field it must fall SPECTRALLY (exponentially) with poly_deg.

Produces a table in the terminal (use `pytest -s`) and a semilog convergence plot in
tests/test_case_scalar_advection/outputs/figures/.
'''

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
CASE_DIR  = Path(__file__).resolve().parent / "test_case_scalar_advection"
CTRL_FILE = CASE_DIR / "control.yaml"
FIG_DIR   = CASE_DIR / "outputs" / "figures"

# The test-required run length, owned here so control.yaml is free for interactive use. With
# advection_velocity = [1, 1] on the unit-square domain, final_time = 1.0 translates the bump exactly
# one domain length in each direction, so the exact solution returns to the initial field.
ROUNDTRIP_FINAL_TIME = 1.0


def _roundtrip_l2_error(poly_deg: int, cfl: float | None = None) -> float:
    '''
    Load the case, override only poly_deg, advect one period, and return the relative L2 error
    between the final and initial fields. Paths (mesh, IC) are resolved relative to the control
    file's directory, exactly as the runner does.

    `cfl` optionally tightens the time step. The deep sweep needs it: this measures the SPATIAL error,
    but RK4's O(dt^4) time error does not care how high poly_deg goes. Past ~degree 8 the spatial error
    drops below the time error and the curve would flatten onto a TIME-integration floor -- which looks
    exactly like a broken spatial scheme. Shrinking dt pushes that floor down out of the way.
    '''
    cfg = load_case_yaml(CTRL_FILE)
    cfg.mesh.poly_deg = poly_deg                       # the swept parameter

    # The test OWNS its final_time rather than inheriting it from control.yaml, so the case file stays
    # free for interactive runs (edit its final_time / IO to watch an animation without touching this
    # test). Here it is a CORRECTNESS constraint, not a preference: the round-trip is only exact when
    # advection_velocity * final_time is a whole number of domain lengths (the guard below enforces it).
    cfg.time_stepping.final_time = ROUNDTRIP_FINAL_TIME
    if cfl is not None:
        cfg.time_stepping.cfl = cfl

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
def _run_sweep(degrees: list, cfl: float | None = None):
    '''Run the poly_deg sweep, report it, and assert spectral decay. Shared by both depths below.'''

    # Guard: the round-trip is only exact if the case is fully periodic and the bump translates a
    # whole number of domain lengths. Fail loudly if control.yaml isn't set up for it.
    cfg  = load_case_yaml(CTRL_FILE)
    disp = np.asarray(cfg.physics.advection_velocity, dtype=float) * cfg.time_stepping.final_time
    assert all(bc.type == 'periodic' for bc in cfg.boundary_conditions.values()), \
        "round-trip convergence test requires all-periodic boundaries in control.yaml"
    assert np.allclose(disp, np.round(disp)), \
        f"round-trip requires advection_velocity * final_time to be whole domain lengths, got {disp}"

    errors = [_roundtrip_l2_error(P, cfl) for P in degrees]
    slope  = float(np.polyfit(degrees, np.log10(errors), 1)[0])

    # Report: table to the terminal + a convergence plot saved under outputs/figures/.
    _print_table(degrees, errors, slope)
    fig_path = _plot_convergence(degrees, errors, slope)
    Console().print(f"  convergence plot saved to: [green]{fig_path}[/green]")

    # 1. finite, and decreasing with poly_deg UNTIL it reaches the round-off floor. The deep sweep
    #    drives the error down to ~1e-12 (near double-precision epsilon times the accumulated operation
    #    count over the whole march), and once there it JITTERS rather than decreasing -- a higher degree
    #    does more work per step, so it carries slightly more round-off. Requiring STRICT monotonicity
    #    all the way down would flag that expected floor jitter as a failure (it did: the degree-12 point
    #    came in at 2.6e-12 vs degree-10's 2.3e-12). So require strict decrease only while the error is
    #    still ABOVE the floor; below it, the field has fully converged and small wiggles are physical.
    ROUNDOFF_FLOOR = 1e-11
    assert all(np.isfinite(errors)), f"non-finite error in {errors}"
    assert all(errors[i + 1] < errors[i]
               for i in range(len(errors) - 1) if errors[i] > ROUNDOFF_FLOOR), \
        f"error not decreasing with poly_deg (above the {ROUNDOFF_FLOOR:.0e} round-off floor): {errors}"

    # 2. spectral (exponential) convergence: log10(error) drops at least ~0.6 per degree
    assert slope < -0.6, f"convergence slope {slope:.3f}/degree is not spectral"

    # 3. sanity: the finest resolution is already quite accurate
    assert errors[-1] < 1e-2, f"finest-degree error {errors[-1]:.2e} unexpectedly large"


@pytest.mark.numerics
@pytest.mark.slow              # advects one full period at each degree -> time-marching, ~15 s
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_scalar_advection_spectral_convergence():
    '''
    Periodic round-trip L2 error must fall spectrally (exponentially) with poly_deg.

    SHALLOW sweep: enough to establish the exponential trend and catch a broken flux. Still `slow` --
    each degree advects the bump a full period through many RK4 steps. The deep sweep below carries the
    same measurement to machine precision.
    '''
    _run_sweep([2, 3, 4, 5])


@pytest.mark.numerics
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_scalar_advection_spectral_convergence_deep():
    '''
    The same measurement, swept to degree 12 -- far enough to watch the error fall through many orders
    of magnitude and flatten onto machine precision.

    The tightened CFL is essential, not cosmetic: this test measures the SPATIAL error, but RK4's
    O(dt^4) TIME error is indifferent to poly_deg. Past ~degree 8 the spatial error drops below the time
    error at the case's default CFL, and the curve flattens onto a time-integration floor -- which is
    indistinguishable, from the outside, from a spatial scheme that stopped converging. Shrinking dt
    pushes that floor down so the spatial convergence stays the thing being measured.
    '''
    _run_sweep([2, 3, 4, 6, 8, 10, 12], cfl=0.05)
