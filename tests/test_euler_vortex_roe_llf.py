# tests/test_euler_vortex_roe_llf.py
'''
Riemann-solver comparison on the isentropic Euler vortex: LLF (Rusanov) vs Roe.

Both solvers share the identical central flux and differ ONLY in their interface dissipation
(scalar |lambda_max| for LLF, the full matrix |A_Roe| for Roe). On the smooth, convecting vortex
this test confirms two things:

  1. CORRECTNESS INSIDE THE FULL SOLVER: with riemann_solver='roe', the DG scheme still converges
     (near-)spectrally with poly_deg -- i.e. the Roe dissipation is consistent and doesn't wreck the
     high-order accuracy. (The single-face algebra is checked separately in scratchpad/roe_check.py;
     this is the end-to-end analogue.)

  2. ROE IS LESS DISSIPATIVE THAN LLF where that is observable. Roe dissipates each characteristic
     wave by its own |eigenvalue| instead of the global worst case |lambda_max|, so on a
     resolution-limited field it should carry LESS error than LLF.

IMPORTANT nuance (see test_euler_vortex_convergenceFAST.py's docstring): on this [0,10]^2 domain the
analytic vortex is only ~1e-5 periodic at the boundary, which puts a ~1e-5 FLOOR on the achievable
L2 error. Once poly_deg is high enough to reach that floor (here P>=4), the error is set by the
boundary mismatch, NOT by interface dissipation, so the LLF/Roe ordering there is meaningless (and
can even invert on parity wiggles). We therefore only assert the "Roe < LLF" dissipation ordering in
the RESOLVED regime, where the error sits comfortably above the floor.

Run `pytest -s tests/test_euler_vortex_roe_llf.py` to see the comparison table; a two-curve semilog
convergence plot is saved under tests/test_case_euler_vortex/outputs/figures/.
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

CASE_DIR  = Path(__file__).resolve().parent / "test_case_euler_vortex"
CTRL_FILE = CASE_DIR / "control.yaml"
FIG_DIR   = CASE_DIR / "outputs" / "figures"

# Polynomial degrees to sweep. P=2,3 sit above the boundary floor (resolved regime); P=4,5 reach it.
DEGREES = [2, 3, 4]

# Error level above which the L2 error is dominated by discretization/dissipation rather than the
# ~1e-5 analytic-periodicity boundary floor. Only in this regime is the LLF-vs-Roe ordering physical.
RESOLVED_FLOOR = 1.0e-4


def _load_ic_module(cfg):
    '''Import the case's initial_condition module to reuse its function AND freestream constants.'''
    ic_path = CASE_DIR / str(cfg.initialization.IC_file)
    spec    = importlib.util.spec_from_file_location("vortex_ic", ic_path)
    module  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _exact_solution(mesh, cfg, ic_mod, t: float) -> list:
    '''Exact vortex at time t: the initial field translated by (U_INF*t, V_INF*t) with periodic wrap,
    sampled at each element's quadrature nodes so it aligns with e.solution.'''
    ndim    = cfg.mesh.ndim
    corners = np.concatenate([np.asarray(e.node_coords).reshape(-1, ndim) for e in mesh.elements])
    xmin, ymin = corners[:, 0].min(), corners[:, 1].min()
    Lx,   Ly   = corners[:, 0].max() - xmin, corners[:, 1].max() - ymin
    shift_x, shift_y = ic_mod.U_INF * t, ic_mod.V_INF * t

    exact = []
    for e in mesh.elements:
        coords = e.quad_node_coords
        xs = xmin + np.mod(coords[..., 0] - shift_x - xmin, Lx)
        ys = ymin + np.mod(coords[..., 1] - shift_y - ymin, Ly)
        exact.append(ic_mod.initial_condition(cfg, xs, ys))
    return exact


def _l2_error(poly_deg: int, riemann_solver: str) -> float:
    '''
    Convect the vortex to final_time with the given poly_deg and Riemann solver, and return the
    relative L2 error (summed over all conserved variables) vs the analytic translated solution.
    Only poly_deg and riemann_solver are overridden; everything else comes from control.yaml.
    '''
    cfg = load_case_yaml(CTRL_FILE)
    cfg.mesh.poly_deg          = poly_deg          # swept parameter
    cfg.physics.riemann_solver = riemann_solver    # the quantity under comparison

    mesh = mesh_class.Mesh()
    mesh.read_mesh(CASE_DIR / cfg.mesh.mesh_file)
    mesh.construct_mesh(cfg)
    initialize(cfg, CASE_DIR / str(cfg.initialization.IC_file), mesh)

    ic_mod = _load_ic_module(cfg)
    T      = cfg.time_stepping.final_time
    exact  = _exact_solution(mesh, cfg, ic_mod, T)

    t = 0.0
    while t < T:
        dt = min(compute_dt(mesh, cfg), T - t)
        time_step(mesh, cfg, t, dt)
        t += dt

    num = den = 0.0
    for e, qe in zip(mesh.elements, exact):
        w = e.jacobian_det * mesh.quad_weights[:, None] * mesh.quad_weights[None, :]
        num += np.sum(w[..., None] * (e.solution - qe)**2)
        den += np.sum(w[..., None] * qe**2)
    return float(np.sqrt(num / den))


def _print_table(degrees, llf, roe) -> None:
    '''Pretty-print the LLF vs Roe comparison (visible with `pytest -s`).'''
    table = Table(title="Isentropic Euler vortex: LLF vs Roe Riemann solver")
    table.add_column("poly_deg", justify="right")
    table.add_column("LLF L2 error", justify="right")
    table.add_column("Roe L2 error", justify="right")
    table.add_column("Roe/LLF", justify="right")
    table.add_column("regime", justify="left")
    for P, el, er in zip(degrees, llf, roe):
        regime = "resolved" if el > RESOLVED_FLOOR else "floor-limited"
        table.add_row(str(P), f"{el:.4e}", f"{er:.4e}", f"{er/el:6.3f}", regime)
    Console().print(table)


def _plot_comparison(degrees, llf, roe) -> Path:
    '''Save a two-curve semilog convergence plot (LLF vs Roe).'''
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(degrees, llf, 'o-', color="tab:blue", label="LLF (Rusanov)")
    ax.semilogy(degrees, roe, 's-', color="tab:red",  label="Roe")
    ax.axhline(RESOLVED_FLOOR, ls=":", color="tab:gray", alpha=0.7,
               label=f"~boundary floor ({RESOLVED_FLOOR:.0e})")
    ax.set_xlabel("Polynomial degree  P")
    ax.set_ylabel("Relative $L_2$ error")
    ax.set_title("Euler vortex: LLF vs Roe interface dissipation")
    ax.set_xticks(degrees)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    out_path = FIG_DIR / "euler_vortex_roe_vs_llf.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_euler_vortex_roe_vs_llf():
    '''Roe must (a) converge spectrally like LLF and (b) be less dissipative in the resolved regime.'''

    # Guard: the analytic exact solution assumes a fully periodic domain.
    cfg = load_case_yaml(CTRL_FILE)
    assert all(bc.type == 'periodic' for bc in cfg.boundary_conditions.values()), \
        "vortex comparison test requires all-periodic boundaries in control.yaml"
    assert cfg.time_stepping.final_time > 0.0, "final_time must be positive"

    llf = [_l2_error(P, "LLF") for P in DEGREES]
    roe = [_l2_error(P, "roe") for P in DEGREES]

    _print_table(DEGREES, llf, roe)
    fig_path = _plot_comparison(DEGREES, llf, roe)
    Console().print(f"  comparison plot saved to: [green]{fig_path}[/green]")

    # 1. Both solvers produce finite errors.
    assert all(np.isfinite(llf)), f"non-finite LLF error in {llf}"
    assert all(np.isfinite(roe)), f"non-finite Roe error in {roe}"

    # 2. Roe converges (near-)spectrally, just like LLF: net reduction + decaying slope. These mirror
    #    the thresholds in test_euler_vortex_convergenceFAST.py, applied to BOTH solvers.
    for name, errs in (("LLF", llf), ("Roe", roe)):
        assert errs[-1] < 0.2 * errs[0], f"{name}: insufficient overall reduction across sweep: {errs}"
        slope = float(np.polyfit(DEGREES, np.log10(errs), 1)[0])
        assert slope < -0.3, f"{name}: convergence slope {slope:.3f}/degree not decaying fast enough"

    # 3. Roe is LESS dissipative than LLF in the RESOLVED regime (error above the ~1e-5 boundary
    #    floor). We deliberately restrict the comparison to degrees where dissipation -- not the
    #    boundary floor -- sets the error; in the floor-limited regime the ordering is not physical.
    resolved = [(P, el, er) for P, el, er in zip(DEGREES, llf, roe) if el > RESOLVED_FLOOR]
    assert resolved, (
        f"no degree left the boundary-floor regime (all LLF errors <= {RESOLVED_FLOOR:.0e}); "
        f"cannot assess dissipation ordering. LLF={llf}"
    )
    for P, el, er in resolved:
        assert er < el, (
            f"Roe should be less dissipative than LLF at the resolved degree P={P}, "
            f"but Roe error {er:.4e} >= LLF error {el:.4e}"
        )
