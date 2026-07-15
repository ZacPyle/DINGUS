# tests/test_mms_viscous.py
'''
Rigorous order-of-accuracy test for the Navier-Stokes VISCOUS discretization, via the Method of
Manufactured Solutions (MMS). Driven by tests/test_case_mms_viscous/ (see
inputs/mms_manufactured.py for the manufactured solution and its sympy-derived source term).

WHAT THIS ADDS OVER COUETTE/POISEUILLE. Those validate the wall BCs against exact solutions whose
velocity/temperature are low-degree POLYNOMIALS -- which a high-order scheme represents almost exactly,
so they confirm correctness but cannot measure the high-order convergence RATE of the viscous operator.
This manufactured solution is built from sin/cos of 2*pi*x: it is NOT representable at any finite
polynomial degree, so the residual genuinely measures the truncation error of the full spatial operator
(volume flux, BR1 gradient pass, interior-interface coupling, and the source term) and must fall
SPECTRALLY with poly_deg.

THE TEST IS RESIDUAL-BASED. The manufactured solution is steady, and its source is S = div(F(q_exact)),
so q_exact is an exact steady state of the forced system: continuously, dq/dt = S - div(F(q_exact)) = 0.
Initialize q_exact and evaluate the discrete residual dq/dt; what remains is purely the discretization
error  (div_continuous - div_discrete)(F(q_exact)) -> 0 spectrally. This needs NO time-marching, so it
is fast and it isolates the SPATIAL operator from time integration.

SELF-VALIDATING. The source term is the divergence of the full inviscid-minus-viscous flux, derived
symbolically in the fixture. If that derivation were wrong, S would not equal div(F) even continuously,
and the residual would FLOOR at O(viscous magnitude) instead of converging. So spectral decay to near
round-off is simultaneous proof that the scheme AND the manufactured source are correct -- no separate
check of the algebra is needed.

Produces a table in the terminal (use `pytest -s`) and a semilog convergence plot in
tests/test_case_mms_viscous/outputs/figures/.
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
from dingus.spatialOperators.residual import compute_residual

CASE_DIR  = Path(__file__).resolve().parent / "test_case_mms_viscous"
CTRL_FILE = CASE_DIR / "control.yaml"
FIG_DIR   = CASE_DIR / "outputs" / "figures"


def _residual_l2(poly_deg: int) -> float:
    '''
    Build the case at the given poly_deg, initialize the manufactured solution q_exact, evaluate the
    forced residual dq/dt (which adds the manufactured source), and return its quadrature-weighted L2
    norm over the whole domain. For the exact steady solution this residual is purely truncation error.
    '''
    cfg = load_case_yaml(CTRL_FILE)
    cfg.mesh.poly_deg = poly_deg

    # This test OWNS its mesh: the 5x5 Square mesh gives a clean spectral curve to round-off, whereas a
    # coarser mesh converges ~2x slower per degree (each element must resolve more of the 2*pi wave). The
    # test is residual-based (no time-marching), so a coarser mesh buys NO speed -- only a weaker curve.
    # control.yaml's mesh is therefore free for interactive runs.
    cfg.mesh.mesh_file     = "./inputs/Square.mesh"
    cfg.source.source_file = str(CASE_DIR / str(cfg.source.source_file))

    mesh = mesh_class.Mesh()
    mesh.read_mesh(CASE_DIR / cfg.mesh.mesh_file)
    mesh.construct_mesh(cfg)

    # q_exact as the initial (and only) state; compute_residual adds S = div(F(q_exact)) at the end.
    initialize(cfg, CASE_DIR / str(cfg.initialization.IC_file), mesh)
    compute_residual(mesh, cfg, t=0.0)

    total = 0.0
    for e in mesh.elements:
        w      = e.jacobian_det * mesh.quad_weights[:, None] * mesh.quad_weights[None, :]
        total += np.sum(w[..., None] * e.residual**2)
    return float(np.sqrt(total))


def _print_table(degrees, norms, slope) -> None:
    '''MMS residual-convergence table (visible with `pytest -s`).'''
    table = Table(title="Viscous MMS: residual of the manufactured solution (spatial order of accuracy)")
    table.add_column("poly_deg",     justify="right")
    table.add_column("||dq/dt||_2",  justify="right")
    table.add_column("reduction",    justify="right")
    for i, (P, r) in enumerate(zip(degrees, norms)):
        ratio = "-" if i == 0 else f"{norms[i - 1] / r:7.2f}x"
        table.add_row(str(P), f"{r:.4e}", ratio)
    Console().print(table)
    Console().print(f"  log10(residual) slope: [bold]{slope:.3f}[/bold] per degree "
                    f"(more negative = faster spectral decay)")
    Console().print("  This is div(F(q_exact)) truncation error for a fully SMOOTH manufactured solution, "
                    "so it must fall\n  to near round-off. A wrong viscous source would instead FLOOR it "
                    "at ~O(1/Re) -- the self-check.")


def _plot_convergence(degrees, norms, slope) -> Path:
    '''Semilog residual-vs-degree plot; a straight line == spectral decay.'''
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(degrees, norms, 'o-', color="tab:purple", label="viscous-MMS residual")
    fit = 10.0 ** np.polyval(np.polyfit(degrees, np.log10(norms), 1), degrees)
    ax.semilogy(degrees, fit, '--', color="tab:gray", label=f"fit: slope {slope:.2f}/degree")
    ax.set_xlabel("Polynomial degree  P")
    ax.set_ylabel(r"$\|dq/dt\|_2$  (manufactured residual)")
    ax.set_title("Navier-Stokes viscous MMS: spectral (p-)convergence")
    ax.set_xticks(degrees)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    out_path = FIG_DIR / "mms_viscous_convergence.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


@pytest.mark.numerics
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_viscous_mms_residual_converges_spectrally():
    '''The manufactured-solution residual must fall spectrally with poly_deg (down to the round-off
    floor), proving the viscous spatial operator is high-order AND the sympy source is correct.'''
    # Swept far enough to show the spectral line unambiguously: the residual falls a near-constant ~7x
    # PER DEGREE, from ~1 at P=2 to ~2.5e-7 at P=10 (about 6.6 orders on a dead-straight semilog line).
    degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    norms   = [_residual_l2(P) for P in degrees]
    slope   = float(np.polyfit(degrees, np.log10(norms), 1)[0])

    # Report: table to the terminal + a convergence plot under outputs/figures/.
    _print_table(degrees, norms, slope)
    fig_path = _plot_convergence(degrees, norms, slope)
    Console().print(f"  convergence plot saved to: [green]{fig_path}[/green]")

    # 1. finite and strictly decreasing with poly_deg (still well above round-off at P=10, so no floor
    #    jitter to tolerate here -- a genuine spectral run just keeps dropping).
    assert all(np.isfinite(norms)), f"non-finite residual in {norms}"
    assert all(norms[i + 1] < norms[i] for i in range(len(norms) - 1)), \
        f"residual not decreasing with poly_deg: {norms}"

    # 2. spectral decay. THE assertion: a wrong viscous source (or a low-order/broken viscous operator)
    #    does not converge -- it plateaus at slope ~0. The measured spectral rate here is ~-0.83
    #    decades/degree (~7x per degree); -0.6 cleanly separates that from a floor without being flaky.
    assert slope < -0.6, (
        f"residual slope {slope:.3f}/degree is not spectral. A viscous-flux or manufactured-source "
        f"error floors the residual instead of converging."
    )

    # 3. large overall reduction -- a smooth manufactured field must drop many orders, not merely shrink.
    assert norms[-1] < norms[0] / 1e5, (
        f"residual fell only from {norms[0]:.2e} to {norms[-1]:.2e} over P={degrees[0]}..{degrees[-1]}; "
        f"far too little for a smooth manufactured solution -- suspect the source term or the operator."
    )
    assert norms[-1] < 1e-5, f"finest-degree residual {norms[-1]:.2e} is too large for a smooth MMS field"
