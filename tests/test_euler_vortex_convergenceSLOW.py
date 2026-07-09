# tests/test_euler_vortex_convergenceFAST.py
'''
Accuracy test driven by the tests/test_case_euler_vortex/ case directory. It loads that case's
control.yaml, mesh, and initial_condition_vortex.py, and sweeps ONLY the polynomial degree.

Physics of the check: the isentropic Euler vortex is an EXACT solution that simply convects at
the freestream velocity (U_INF, V_INF) without changing shape. The exact solution at final_time
is therefore the initial vortex translated by (U_INF*T, V_INF*T) (with periodic wrap). The L2
error between the computed and exact fields -- summed over ALL conserved variables -- is a pure
accuracy measure, and for a high-order DG method on this smooth field it falls SPECTRALLY
(exponentially) with poly_deg.

This is the Euler analogue of test_advection_scalar_convergenceFAST.py. The scalar test's
"exact == initial after a whole-domain round trip" is just the special case where the
translation happens to be a whole number of domain lengths; here we compare against the
analytically translated solution so the test stays fast (short final_time) on the larger
[0,10]^2 domain.

NOTE on the error floor: the analytic vortex is only exactly periodic once its swirl has decayed
to the freestream at the boundaries. On [0,10]^2 with R=1, beta=5 the boundary perturbation is
~1e-5, which caps how far the error can fall. To push convergence further (the full "hockey
stick"), enlarge the domain (and add elements) so the vortex decays more before the boundary.

Produces a table in the terminal (use `pytest -s`) and a semilog convergence plot in
tests/test_case_euler_vortex/outputs/figures/.
'''
import importlib.util
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
CASE_DIR  = Path(__file__).resolve().parent / "test_case_euler_vortex"
CTRL_FILE = CASE_DIR / "control.yaml"
FIG_DIR   = CASE_DIR / "outputs" / "figures"


def _load_ic_module(cfg):
    '''Import the case's initial_condition module so we can reuse its function AND its freestream /
    center constants to build the analytic exact solution consistently with what was initialized.'''
    ic_path = CASE_DIR / str(cfg.initialization.IC_file)
    spec    = importlib.util.spec_from_file_location("vortex_ic", ic_path)
    module  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _exact_solution(mesh, cfg, ic_mod, t: float) -> list:
    '''
    Exact Euler-vortex solution at time t: the initial vortex translated by (U_INF*t, V_INF*t) with
    periodic wrap. Evaluated per element at that element's quadrature nodes, so it lines up with the
    computed e.solution. Domain bounds are read from the mesh geometry (element corner nodes), so the
    periodic wrap is correct regardless of the exact extent (e.g. [0,10] vs [0,10.01]) and does not
    assume the vortex sits at the domain center.
    '''
    ndim    = cfg.mesh.ndim
    corners = np.concatenate([np.asarray(e.node_coords).reshape(-1, ndim) for e in mesh.elements])
    xmin, ymin = corners[:, 0].min(), corners[:, 1].min()
    Lx,   Ly   = corners[:, 0].max() - xmin, corners[:, 1].max() - ymin
    shift_x, shift_y = ic_mod.U_INF * t, ic_mod.V_INF * t

    exact = []
    for e in mesh.elements:
        coords = e.quad_node_coords              # (nx, ny, ndim)
        # Trace each node back to where its fluid came from, wrapped into the periodic domain.
        xs = xmin + np.mod(coords[..., 0] - shift_x - xmin, Lx)
        ys = ymin + np.mod(coords[..., 1] - shift_y - ymin, Ly)
        exact.append(ic_mod.initial_condition(cfg, xs, ys))
    return exact


def _l2_error(poly_deg: int) -> float:
    '''
    Load the case, override only poly_deg, convect the vortex to final_time, and return the relative
    L2 error (over all conserved variables) between the computed and the analytic exact solution.
    '''
    cfg = load_case_yaml(CTRL_FILE)
    cfg.mesh.poly_deg = poly_deg                       # the only swept parameter

    mesh = mesh_class.Mesh()
    mesh.read_mesh(CASE_DIR / cfg.mesh.mesh_file)
    mesh.construct_mesh(cfg)
    initialize(cfg, CASE_DIR / str(cfg.initialization.IC_file), mesh)

    ic_mod = _load_ic_module(cfg)
    T      = cfg.time_stepping.final_time
    exact  = _exact_solution(mesh, cfg, ic_mod, T)     # exact field at final_time, per element

    t = 0.0
    while t < T:
        dt = min(compute_dt(mesh, cfg), T - t)
        time_step(mesh, cfg, t, dt)
        t += dt

    # Relative L2 error, quadrature-weighted, summed over every conserved variable.
    num = den = 0.0
    for e, qe in zip(mesh.elements, exact):
        w = e.jacobian_det * mesh.quad_weights[:, None] * mesh.quad_weights[None, :]  # (nx, ny)
        num += np.sum(w[..., None] * (e.solution - qe)**2)
        den += np.sum(w[..., None] * qe**2)
    return float(np.sqrt(num / den))


def _print_table(degrees: list, errors: list, slope: float) -> None:
    '''Pretty-print the convergence data as a rich table (visible with `pytest -s`).'''
    table = Table(title="Isentropic Euler vortex: convection accuracy")
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
    ax.semilogy(degrees, errors, 'o-', color="tab:red", label="DGSEM vortex error")
    # Least-squares exponential fit line, for visual reference of the spectral rate.
    fit = 10.0 ** np.polyval(np.polyfit(degrees, np.log10(errors), 1), degrees)
    ax.semilogy(degrees, fit, '--', color="tab:gray", label=f"fit: slope {slope:.2f}/degree")
    ax.set_xlabel("Polynomial degree  P")
    ax.set_ylabel("Relative $L_2$ error")
    ax.set_title("Isentropic Euler vortex: spectral (p-)convergence")
    ax.set_xticks(degrees)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    out_path = FIG_DIR / "euler_vortex_convergence.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_euler_vortex_spectral_convergence():
    '''Vortex convection L2 error must fall (near-)spectrally with poly_deg on the smooth field.'''

    # Guard: the analytic exact solution assumes a fully periodic domain.
    cfg = load_case_yaml(CTRL_FILE)
    assert all(bc.type == 'periodic' for bc in cfg.boundary_conditions.values()), \
        "vortex convergence test requires all-periodic boundaries in control.yaml"
    assert cfg.time_stepping.final_time > 0.0, "final_time must be positive"

    degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18]
    errors  = [_l2_error(P) for P in degrees]
    slope   = float(np.polyfit(degrees, np.log10(errors), 1)[0])

    # Report: table to the terminal + a convergence plot saved under outputs/figures/.
    _print_table(degrees, errors, slope)
    fig_path = _plot_convergence(degrees, errors, slope)
    Console().print(f"  convergence plot saved to: [green]{fig_path}[/green]")

    # 1. finite
    assert all(np.isfinite(errors)), f"non-finite error in {errors}"

    # 2. net accuracy gain across the sweep. We deliberately do NOT require STRICT monotonicity: a
    #    residual even/odd node-sampling wiggle can survive even with the vortex at an element center
    #    (LG nodes place a point on the vortex peak for even poly_deg only). A broken flux or wave
    #    speed would fail the overall-reduction and slope checks below, so real regressions are still
    #    caught; benign parity wiggles are not.
    assert errors[-1] < 0.2 * errors[0], \
        f"insufficient overall error reduction across the poly_deg sweep: {errors}"

    # 3. spectral-ish average decay rate (gentler than the scalar test's -0.6: the ~1e-5 boundary
    #    floor and the short final_time soften the achievable rate on this domain).
    assert slope < -0.3, f"convergence slope {slope:.3f}/degree is not decaying fast enough"

    # 4. sanity: the finest resolution is already quite accurate
    assert errors[-1] < 1e-2, f"finest-degree error {errors[-1]:.2e} unexpectedly large"
