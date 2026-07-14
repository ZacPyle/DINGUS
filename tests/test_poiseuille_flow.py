# tests/test_poiseuille_flow.py
'''
Validation of the STATIONARY no-slip wall + the SOURCE-TERM machinery against forced compressible
POISEUILLE FLOW, an exact steady solution (see
test_case_poiseuille_flow/inputs/initial_condition_poiseuille.py for the derivation).

Poiseuille is the companion to Couette, and the two are complementary:

    Couette     : driven by a MOVING wall.       Tests that no-slip TRANSFERS wall motion into the fluid.
    Poiseuille  : driven by a BODY FORCE, with   Tests that no-slip HOLDS THE FLUID STILL against a force
                  both walls stationary.         that is actively trying to push it -- and tests the
                                                 source term.

The velocity profile is the tell. With a body force G pushing on the fluid and no walls to stop it, the
flow would accelerate without bound (uniform u, growing forever). It reaches a steady parabola ONLY
because the stationary no-slip walls hold u = 0 and the resulting shear balances the force:

    u(y) = (G Re / 2) y (H - y),      u_max = G Re H^2 / 8   at the centreline

The temperature is the tell for the SOURCE's energy component. The body force does work at a rate G*u,
and in the steady energy balance that work exactly cancels the u u'' term, leaving pure viscous
dissipation and a QUARTIC T(y). Omit the G*u energy source and this profile comes out wrong.

Produces tables in the terminal (use `pytest -s`) and figures in
tests/test_case_poiseuille_flow/outputs/figures/: a spectral-convergence plot of the steady residual,
and the relaxed velocity/temperature profiles against the exact solution.
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
from dingus.physics.constitutiveRelations import compute_temperature
from dingus.spatialOperators.residual import compute_residual
from dingus.timeIntegrators.timeIntegration import compute_dt, time_step

CASE_DIR  = Path(__file__).resolve().parent / "test_case_poiseuille_flow"
CTRL_FILE = CASE_DIR / "control.yaml"
FIG_DIR   = CASE_DIR / "outputs" / "figures"

# The relaxation-from-rest test runs to this time; owned here so control.yaml is free for interactive
# runs. It is ~6 viscous relaxation times tau = H^2/(nu pi^2) = Re/pi^2 ~ 1. (The steady-residual /
# negative-control / wall-stability tests do not use it.)
RELAX_FINAL_TIME = 6.0

pytestmark = pytest.mark.skipif(
    not (CASE_DIR / "inputs" / "NarrowChannel.mesh").is_file(),
    reason="test_case_poiseuille_flow/inputs/NarrowChannel.mesh not found -- generate it with HOHQMesh "
           "(channel [0, 0.5] x [0, 1]; 2 elements in x, 4 in y; boundaries named Left/Right/Bottom/Top)."
)


def _load_ic_module(cfg):
    ic_path = CASE_DIR / str(cfg.initialization.IC_file)
    spec    = importlib.util.spec_from_file_location("poiseuille_ic", ic_path)
    module  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_mesh(poly_deg: int | None = None, with_source: bool = True):
    '''Load the Poiseuille case; `with_source=False` is the negative control (removes the driving force).'''
    cfg = load_case_yaml(CTRL_FILE)
    cfg.time_stepping.final_time = RELAX_FINAL_TIME     # test owns its timing; control.yaml is free
    if poly_deg is not None:
        cfg.mesh.poly_deg = poly_deg
    if not with_source:
        cfg.source.source_method = 'none'

    # The source file path in the YAML is relative to the case directory.
    if cfg.source.source_method == 'analytical':
        cfg.source.source_file = str(CASE_DIR / str(cfg.source.source_file))

    mesh = mesh_class.Mesh()
    mesh.read_mesh(CASE_DIR / cfg.mesh.mesh_file)
    mesh.construct_mesh(cfg)
    return mesh, cfg


def _residual_norm(mesh, cfg) -> float:
    compute_residual(mesh, cfg, t=0.0)
    total = 0.0
    for e in mesh.elements:
        w      = e.jacobian_det * mesh.quad_weights[:, None] * mesh.quad_weights[None, :]
        total += np.sum(w[..., None] * e.residual**2)
    return float(np.sqrt(total))


def _sample_profiles(mesh, cfg):
    ys, us, Ts = [], [], []
    for e in mesh.elements:
        q   = e.solution
        rho = q[..., 0]
        ys.append(e.quad_node_coords[..., 1].ravel())
        us.append((q[..., 1] / rho).ravel())
        Ts.append(compute_temperature(q, cfg).ravel())
    return np.concatenate(ys), np.concatenate(us), np.concatenate(Ts)


def _print_decay_table(case, checkpoints, measured, theory, ratio, tau) -> None:
    '''
    Wall-condition violation vs. its analytic decay (visible with `pytest -s`).

    The ratio column is the point of the whole test. An UNSTABLE wall also passes through small errors
    on its way up, so "the error is small" proves nothing -- what separates a stable wall from an
    unstable one is whether the violation dies at the rate the viscosity dictates. Ratio ~ 1 = yes.
    '''
    table = Table(title=f"{case}: wall-condition violation vs. analytic decay  (tau = {tau:.3f})")
    table.add_column("t",                justify="right")
    table.add_column("measured",         justify="right")
    table.add_column("analytic",         justify="right")
    table.add_column("measured/analytic", justify="right")
    for t, m, th, r in zip(checkpoints, measured, theory, ratio):
        colour = "green" if 0.8 < r < 1.25 else ("yellow" if 0.5 < r < 2.0 else "red")
        table.add_row(f"{t:.1f}", f"{100*m:.4f}%", f"{100*th:.4f}%", f"[{colour}]{r:.3f}[/{colour}]")
    Console().print(table)
    Console().print("  (ratio ~ 1 means the boundary-condition violation is decaying at exactly the rate "
                    "the viscosity\n  dictates -- which is what distinguishes a STABLE wall from an "
                    "unstable one that merely happens\n  to be passing through a small error.)")


def _print_residual_table(degrees, norms, slope) -> None:
    '''Steady-residual p-convergence table (visible with `pytest -s`).'''
    table = Table(title="Poiseuille flow: steady residual of the EXACT solution (wall BC + source term)")
    table.add_column("poly_deg",    justify="right")
    table.add_column("||dq/dt||_2", justify="right")
    table.add_column("reduction",   justify="right")
    for i, (P, r) in enumerate(zip(degrees, norms)):
        ratio = "-" if i == 0 else f"{norms[i - 1] / r:6.2f}x"
        table.add_row(str(P), f"{r:.4e}", ratio)
    Console().print(table)
    Console().print(f"  log10(residual) slope: [bold]{slope:.3f}[/bold] per degree "
                    f"(more negative = faster spectral decay)")
    Console().print("  the residual is NOT machine-zero: rho = gamma M^2 P0 / T(y) is RATIONAL, so it "
                    "cannot be represented\n  exactly in the polynomial basis. The EXPONENTIAL decay is "
                    "the statement of high-order consistency --\n  here of the wall BC and the source "
                    "term together.")


def _plot_residual_convergence(degrees, norms, slope) -> Path:
    '''Semilog residual-vs-degree plot; a straight line == spectral decay.'''
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(degrees, norms, 'o-', color="tab:green", label="steady residual of exact solution")
    fit = 10.0 ** np.polyval(np.polyfit(degrees, np.log10(norms), 1), degrees)
    ax.semilogy(degrees, fit, '--', color="tab:gray", label=f"fit: slope {slope:.2f}/degree")
    ax.set_xlabel("Polynomial degree  P")
    ax.set_ylabel(r"$\|dq/dt\|_2$")
    ax.set_title("Poiseuille: spectral consistency of the walls + body force")
    ax.set_xticks(degrees)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    out_path = FIG_DIR / "poiseuille_source_convergence.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _print_profile_table(cfg, ic, y, u, T) -> None:
    '''Relaxed-from-rest profile accuracy, sampled across the channel.'''
    order = np.argsort(y)
    y_s, u_s, T_s = y[order], u[order], T[order]
    u_max = ic.centreline_velocity(cfg)

    table = Table(title="Poiseuille flow: profile after relaxing FROM REST (vs. exact steady solution)")
    table.add_column("y",            justify="right")
    table.add_column("u (measured)", justify="right")
    table.add_column("u (exact)",    justify="right")
    table.add_column("% err (u)",    justify="right")
    table.add_column("T (measured)", justify="right")
    table.add_column("T (exact)",    justify="right")

    for idx in np.linspace(0, len(y_s) - 1, 10).astype(int):
        yy  = y_s[idx]
        ue  = float(ic.exact_velocity(cfg, yy))
        Te  = float(ic.exact_temperature(cfg, yy))
        pct = 100.0 * (u_s[idx] - ue) / u_max            # normalized by u_max (u -> 0 at both walls)
        col = "green" if abs(pct) < 1.0 else ("yellow" if abs(pct) < 3.0 else "red")
        table.add_row(f"{yy:.4f}", f"{u_s[idx]:+.5f}", f"{ue:+.5f}",
                      f"[{col}]{pct:+.3f}%[/{col}]", f"{T_s[idx]:.6f}", f"{Te:.6f}")

    Console().print(table)
    Console().print(f"  (% err in u is normalized by the centreline speed u_max = {u_max:.4f}, since "
                    f"u -> 0 at BOTH walls)")
    Console().print(f"  centreline temperature rise from viscous heating: "
                    f"[bold]{ic.viscous_heating_amplitude(cfg):.5f}[/bold] "
                    f"(this is the check on the source's G*u energy component)")


def _plot_profiles(cfg, ic, y, u, T) -> Path:
    '''
    Measured u(y) and T(y) against the exact solution.

    The velocity panel is the tell for the STATIONARY no-slip walls: without them there is nothing to
    oppose the body force and the fluid would accelerate as a uniform slug forever. A bounded parabola
    anchored at u = 0 on BOTH walls can only come from no-slip. The temperature panel is the quartic
    viscous-heating profile -- the check on the source term's G*u energy component.
    '''
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    order = np.argsort(y)
    y_s, u_s, T_s = y[order], u[order], T[order]
    y_fine = np.linspace(0.0, ic.H, 200)

    fig, (ax_u, ax_T) = plt.subplots(1, 2, figsize=(10, 4.5))

    # --- velocity ----------------------------------------------------------------------------------
    ax_u.plot(ic.exact_velocity(cfg, y_fine), y_fine, '-', color="tab:gray", lw=2, label="exact")
    ax_u.plot(u_s, y_s, 'o', color="tab:green", ms=4, label="DGSEM (relaxed from rest)")
    ax_u.axvline(0.0, color="tab:gray", ls=":", lw=1)
    ax_u.set_xlabel("u")
    ax_u.set_ylabel("y")
    ax_u.set_title("Velocity: STATIONARY no-slip walls vs. the body force")
    ax_u.grid(True, ls=":", alpha=0.6)
    ax_u.legend()

    # --- temperature -------------------------------------------------------------------------------
    ax_T.plot(ic.exact_temperature(cfg, y_fine), y_fine, '-', color="tab:gray", lw=2, label="exact")
    ax_T.axvline(ic.T_WALL, color="tab:orange", ls=":", lw=1.5, label="wall temperature")
    ax_T.plot(T_s, y_s, 'o', color="tab:red", ms=4, label="DGSEM (relaxed from rest)")
    ax_T.set_xlabel("T")
    ax_T.set_ylabel("y")
    ax_T.set_title("Temperature: quartic viscous heating")
    ax_T.grid(True, ls=":", alpha=0.6)
    ax_T.legend()

    fig.suptitle("Forced compressible Poiseuille flow: relaxed from rest vs. exact steady solution")
    fig.tight_layout()
    out_path = FIG_DIR / "poiseuille_profiles.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


@pytest.mark.numerics
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_exact_poiseuille_is_steady_and_converges_spectrally():
    '''
    The exact forced-Poiseuille solution must have (near) zero residual, falling spectrally with P.

    As in Couette the residual is not machine-zero, because rho = gamma M^2 P0 / T(y) is rational while
    the basis is polynomial. The exponential decay is the statement of high-order consistency -- here of
    the wall BC AND the source term together.
    '''
    degrees, norms = [3, 4, 5, 6], []
    for P in degrees:
        mesh, cfg = _build_mesh(poly_deg=P)
        initialize(cfg, CASE_DIR / str(cfg.initialization.IC_file), mesh)
        norms.append(_residual_norm(mesh, cfg))

    norms = np.array(norms)
    slope = float(np.polyfit(degrees, np.log10(norms), 1)[0])

    # Report: table to the terminal + a convergence plot saved under outputs/figures/.
    _print_residual_table(degrees, norms, slope)
    fig_path = _plot_residual_convergence(degrees, norms, slope)
    Console().print(f"  convergence plot saved to: [green]{fig_path}[/green]")

    assert np.all(np.diff(norms) < 0), (
        f"steady residual must fall monotonically under p-refinement, got {norms} for P={degrees}"
    )
    assert norms[-1] < norms[0] / 50.0, (
        f"steady residual decayed only from {norms[0]:.2e} to {norms[-1]:.2e}; too slow for an analytic "
        f"solution -- suspect an inconsistent wall BC or a mis-scaled source term."
    )


@pytest.mark.numerics
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_without_the_body_force_the_exact_solution_is_not_steady():
    '''
    NEGATIVE CONTROL for the source term. The parabolic profile is held up ENTIRELY by the body force
    balancing the viscous shear. Remove the force and the same state is no longer an equilibrium at all
    (nothing opposes the shear, so the flow must immediately decelerate), and the residual must jump.

    Without this, "the residual is small" would not prove the source term was contributing anything.
    '''
    mesh_on, cfg_on = _build_mesh(poly_deg=5, with_source=True)
    initialize(cfg_on, CASE_DIR / str(cfg_on.initialization.IC_file), mesh_on)
    r_forced = _residual_norm(mesh_on, cfg_on)

    mesh_off, cfg_off = _build_mesh(poly_deg=5, with_source=False)
    initialize(cfg_off, CASE_DIR / str(cfg_off.initialization.IC_file), mesh_off)
    r_unforced = _residual_norm(mesh_off, cfg_off)

    assert r_unforced > 100.0 * r_forced, (
        f"removing the driving body force left the residual at {r_unforced:.2e} vs {r_forced:.2e} with it. "
        f"The source term is not actually contributing to the residual."
    )


@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_wall_bc_is_stable_and_relaxes_at_the_analytic_rate():
    '''
    REGRESSION TEST for the wall instability, on STATIONARY walls -- the companion to the Couette one.

    THE BUG: a Dirichlet wall imposed WEAKLY through the BR1 central trace, with no penalty term, has
    nothing damping a boundary-condition violation, so one GROWS. It hit STATIONARY walls too, not just
    moving ones -- which means this case was, for a while, PASSING FOR THE WRONG REASON. Its relaxation
    test only looked at the profile at the finish line (t=6), and the unstable mode had not yet grown
    enough to break a 3% tolerance. It got the right answer while being wrong.

    That is the trap this test exists to close: "the error is small at the end" is NOT evidence of a
    stable scheme, because an unstable scheme passes through small errors on its way UP. What separates
    them is the RATE -- so assert against the analytic decay and require monotonicity.

    Poiseuille startup from rest. The deviation d = u - u_steady obeys the homogeneous heat equation
    with d(y,0) = -u_steady(y) = -(G Re/2) y (H - y). The Fourier sine coefficients of a parabola are
    8 H^2 / (n^3 pi^3) (odd n only), so

        d(y,t) = -(32/pi^3) u_max sum_{n odd} (1/n^3) sin(n pi y/H) exp(-n^2 t/tau),   tau = H^2/(nu pi^2)

    Once the higher modes die (they decay as 1/n^3 AND exp(-n^2 t/tau), so very fast), the error must
    follow  (32/pi^3) u_max exp(-t/tau) = 1.032 u_max exp(-t/tau).

    Note the amplitude differs from Couette's 2/pi = 0.637 -- a parabola is not a straight line -- while
    tau is the same. Getting that constant right is what makes this a real check on the RATE rather than
    a fudge factor.
    '''
    mesh, cfg = _build_mesh(poly_deg=2)
    ic        = _load_ic_module(cfg)

    nu    = 1.0 / cfg.physics.Re
    tau   = ic.H**2 / (nu * np.pi**2)          # relaxation time of the slowest (n=1) mode
    amp   = 32.0 / np.pi**3                    # |b_1| / u_max  for a parabolic initial deviation
    u_max = ic.centreline_velocity(cfg)

    # Fluid AT REST: the body force must accelerate it while the STATIONARY no-slip walls hold it back.
    gamma   = cfg.physics.gamma
    gammaM2 = gamma * cfg.physics.mach_ref**2
    p0      = ic.T_WALL / gammaM2
    for e in mesh.elements:
        q = np.zeros(e.quad_node_coords.shape[:-1] + (cfg.physics.num_eq,))
        q[..., 0] = 1.0
        q[..., 3] = p0 / (gamma - 1.0)
        e.solution = q

    checkpoints = [2.0, 4.0, 6.0]
    t, measured = 0.0, []
    for target in checkpoints:
        while t < target - 1e-12:
            dt = min(compute_dt(mesh, cfg), target - t)
            time_step(mesh, cfg, t, dt)
            t += dt
        y, u, _ = _sample_profiles(mesh, cfg)
        measured.append(np.max(np.abs(u - ic.exact_velocity(cfg, y))) / u_max)

    measured = np.array(measured)
    theory   = amp * np.exp(-np.array(checkpoints) / tau)
    ratio    = measured / theory

    _print_decay_table("Poiseuille", checkpoints, measured, theory, ratio, tau)

    # 1. STABILITY: the violation must decay MONOTONICALLY. The unstable scheme turned around and grew.
    assert np.all(np.diff(measured) < 0), (
        f"the wall-condition violation is GROWING: {100*measured} % at t={checkpoints}. The weakly-imposed "
        f"Dirichlet wall is unstable -- check the interior penalty in wall_viscous_normal_flux."
    )

    # 2. RATE, TWO-SIDED. Bounding the ratio only from ABOVE would be a half-test: it would catch a wall
    #    that transfers momentum too slowly, but an over-estimated analytic amplitude (or a wall that
    #    over-drives the fluid) would sail through as a comfortably small ratio. Pin it from both sides,
    #    so the assertion is really "the violation decays at the rate the physics dictates".
    assert np.all((ratio > 0.5) & (ratio < 2.0)), (
        f"the violation is not decaying at the analytic rate: measured/analytic = {ratio} at "
        f"t={checkpoints} (want ~1). The stationary walls are not absorbing momentum at the rate the "
        f"viscosity dictates."
    )
    assert measured[-1] < 0.02, (
        f"the wall-condition violation is still {100*measured[-1]:.2f}% of u_max after {checkpoints[-1]} "
        f"(~{checkpoints[-1]/tau:.1f} relaxation times); it should be well under 1%."
    )


@pytest.mark.physics
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_poiseuille_relaxes_from_rest_to_the_exact_parabola():
    '''
    End-to-end: start at rest, let the body force accelerate the fluid, and let the two STATIONARY
    no-slip walls arrest it into the steady parabola.

    The velocity profile is the discriminating measurement. If the walls did not enforce no-slip, there
    would be nothing to oppose the force: the fluid would accelerate as a uniform slug (u ~ G t / rho,
    reaching ~ 5 by the end of this run) instead of settling at u_max = G Re H^2 / 8 = 1 with u = 0 at the
    walls. A bounded, parabolic, wall-anchored profile can only come from no-slip.
    '''
    mesh, cfg = _build_mesh(poly_deg=2)     # low degree keeps the explicit viscous run affordable
    ic        = _load_ic_module(cfg)

    # --- Initial condition: FLUID AT REST at the wall temperature ---------------------------------
    gamma   = cfg.physics.gamma
    gammaM2 = gamma * cfg.physics.mach_ref**2
    p0      = ic.T_WALL / gammaM2                    # so rho = 1
    q_rest  = np.array([1.0, 0.0, 0.0, p0 / (gamma - 1.0)])
    for e in mesh.elements:
        e.solution = np.ones(e.quad_node_coords.shape[:-1] + (cfg.physics.num_eq,)) * q_rest

    # --- March to (near) steady state --------------------------------------------------------------
    t, T_final = 0.0, cfg.time_stepping.final_time
    while t < T_final:
        dt = min(compute_dt(mesh, cfg), T_final - t)
        time_step(mesh, cfg, t, dt)
        t += dt

    y, u, T = _sample_profiles(mesh, cfg)
    u_max   = ic.centreline_velocity(cfg)

    # Report: profile table to the terminal + a u(y)/T(y) plot saved under outputs/figures/.
    _print_profile_table(cfg, ic, y, u, T)
    fig_path = _plot_profiles(cfg, ic, y, u, T)
    Console().print(f"  profile plot saved to: [green]{fig_path}[/green]")

    # 1. The force actually drove a flow...
    assert np.max(u) > 0.5 * u_max, (
        f"the body force did not drive the flow (max u = {np.max(u):.3e}, expected ~{u_max:.3f})"
    )

    # 2. ...and the no-slip walls ARRESTED it into a bounded parabola rather than a runaway slug.
    assert np.max(u) < 1.5 * u_max, (
        f"max u = {np.max(u):.3f} far exceeds the steady centreline {u_max:.3f} -- the flow is still "
        f"accelerating, so the no-slip walls are not opposing the body force."
    )

    u_err = np.max(np.abs(u - ic.exact_velocity(cfg, y))) / u_max
    assert u_err < 0.03, f"velocity profile is off by {u_err*100:.2f}% of the centreline speed"

    # 3. The quartic viscous-heating temperature profile -- the check on the source's ENERGY component.
    bump = ic.viscous_heating_amplitude(cfg)
    assert bump > 0.01, f"viscous heating bump {bump:.4f} is too small for this test to be meaningful"

    T_err = np.max(np.abs(T - ic.exact_temperature(cfg, y)))
    assert T_err < 0.10 * bump, (
        f"temperature is off by {T_err:.4e} against a viscous-heating amplitude of {bump:.4e}. Suspect the "
        f"G*u energy component of the body force."
    )
