# tests/test_couette_flow.py
'''
Validation of the NO-SLIP and ISOTHERMAL wall boundary conditions against compressible COUETTE FLOW,
an exact steady solution of the Navier-Stokes equations (see
test_case_couette_flow/inputs/initial_condition_couette.py for the derivation).

There are TWO tests here, and the split is deliberate -- neither one alone is sufficient:

  1. STEADY-RESIDUAL (fast).  Initialize the EXACT solution and demand dq/dt = 0. This asks: "given the
     right answer, does the discretization AGREE that it is the right answer?" It is cheap (no time
     stepping) and it converges SPECTRALLY under p-refinement, which is a strong statement.

     But it has a blind spot worth being explicit about: the exact Couette solution ALREADY satisfies
     the wall conditions, so at the wall the interior trace and the wall state coincide, the wall's flux
     correction vanishes, and a SLIP wall would produce exactly the same answer as a NO-SLIP one. (At the
     bottom wall u = 0 anyway; at the top wall the fluid is already moving at the wall speed.) A
     consistency test cannot see a condition that is not being violated.

     So it is backed by a NEGATIVE CONTROL: swap the walls to ADIABATIC and the same exact solution must
     now produce a LARGE residual, because a real Couette flow conducts its viscous heat out through the
     walls and an adiabatic wall forbids exactly that. If the negative control failed, the residual test
     would be vacuous.

  2. RELAXATION (slow).  Start the fluid AT REST and let the moving wall drag it into motion, then check
     it lands on the exact profile. THIS is the test that shows no-slip actually ENFORCES something: from
     rest, a slip wall transmits no tangential momentum at all and the fluid simply never moves. It also
     confirms that the isothermal walls carry the viscous heat out to a genuine steady balance.

Together: (1) says the BC is CONSISTENT and high-order; (2) says the BC is DOING ITS JOB.

Produces tables in the terminal (use `pytest -s`) and figures in
tests/test_case_couette_flow/outputs/figures/: a spectral-convergence plot of the steady residual, and
the relaxed velocity/temperature profiles against the exact solution.
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

CASE_DIR  = Path(__file__).resolve().parent / "test_case_couette_flow"
CTRL_FILE = CASE_DIR / "control.yaml"
FIG_DIR   = CASE_DIR / "outputs" / "figures"

# The relaxation-from-rest test runs to this time; owned here so control.yaml is free for interactive
# runs. It is ~6 viscous relaxation times tau = H^2/(nu pi^2) = Re/pi^2 ~ 1, which brings the flow to
# within ~0.3% of the exact steady profile. (The steady-residual / negative-control / wall-stability
# tests do not use it -- they either do no time-marching or set their own checkpoints.)
RELAX_FINAL_TIME = 6.0

# The channel mesh is generated with HOHQMesh and dropped into the case directory. Skip cleanly (rather
# than erroring) if it has not been generated yet.
pytestmark = pytest.mark.skipif(
    not (CASE_DIR / "inputs" / "NarrowChannel.mesh").is_file(),
    reason="test_case_couette_flow/inputs/NarrowChannel.mesh not found -- generate it with HOHQMesh "
           "(channel [0, 0.5] x [0, 1]; 2 elements in x, 4 in y; boundaries named Left/Right/Bottom/Top)."
)


def _load_ic_module(cfg):
    '''Import the case's IC module so the test reuses its exact solution -- one source of truth.'''
    ic_path = CASE_DIR / str(cfg.initialization.IC_file)
    spec    = importlib.util.spec_from_file_location("couette_ic", ic_path)
    module  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_mesh(poly_deg: int | None = None, wall_type: str | None = None):
    '''Load the Couette case, optionally overriding the polynomial degree and/or the wall BC type.'''
    cfg = load_case_yaml(CTRL_FILE)
    cfg.time_stepping.final_time = RELAX_FINAL_TIME     # test owns its timing; control.yaml is free
    if poly_deg is not None:
        cfg.mesh.poly_deg = poly_deg
    if wall_type is not None:
        # Negative control: retype the walls, keeping their temperature / motion.
        for name in ('Bottom', 'Top'):
            cfg.boundary_conditions[name].type = wall_type

    mesh = mesh_class.Mesh()
    mesh.read_mesh(CASE_DIR / cfg.mesh.mesh_file)
    mesh.construct_mesh(cfg)
    return mesh, cfg


def _residual_norm(mesh, cfg) -> float:
    '''L2 norm of the steady residual dq/dt over the whole domain (quadrature-weighted).'''
    compute_residual(mesh, cfg, t=0.0)
    total = 0.0
    for e in mesh.elements:
        w      = e.jacobian_det * mesh.quad_weights[:, None] * mesh.quad_weights[None, :]
        total += np.sum(w[..., None] * e.residual**2)
    return float(np.sqrt(total))


def _sample_profiles(mesh, cfg):
    '''Flatten every quadrature node into (y, u, T) arrays for comparison against the exact profile.'''
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
    table.add_column("t",                 justify="right")
    table.add_column("measured",          justify="right")
    table.add_column("analytic",          justify="right")
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
    table = Table(title="Couette flow: steady residual of the EXACT solution (wall-BC consistency)")
    table.add_column("poly_deg",         justify="right")
    table.add_column("||dq/dt||_2",      justify="right")
    table.add_column("reduction",        justify="right")
    for i, (P, r) in enumerate(zip(degrees, norms)):
        ratio = "-" if i == 0 else f"{norms[i - 1] / r:6.2f}x"
        table.add_row(str(P), f"{r:.4e}", ratio)
    Console().print(table)
    Console().print(f"  log10(residual) slope: [bold]{slope:.3f}[/bold] per degree "
                    f"(more negative = faster spectral decay)")
    Console().print("  the residual is NOT machine-zero: rho = gamma M^2 P0 / T(y) is RATIONAL, so it "
                    "cannot be represented\n  exactly in the polynomial basis. What matters is that it "
                    "decays EXPONENTIALLY -- an inconsistent\n  wall BC would floor it at a fixed level "
                    "no matter how high the degree went.")


def _plot_residual_convergence(degrees, norms, slope) -> Path:
    '''Semilog residual-vs-degree plot; a straight line == spectral decay == a high-order-consistent wall.'''
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(degrees, norms, 'o-', color="tab:blue", label="steady residual of exact solution")
    fit = 10.0 ** np.polyval(np.polyfit(degrees, np.log10(norms), 1), degrees)
    ax.semilogy(degrees, fit, '--', color="tab:gray", label=f"fit: slope {slope:.2f}/degree")
    ax.set_xlabel("Polynomial degree  P")
    ax.set_ylabel(r"$\|dq/dt\|_2$")
    ax.set_title("Couette: spectral consistency of the no-slip / isothermal walls")
    ax.set_xticks(degrees)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    out_path = FIG_DIR / "couette_wall_bc_convergence.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _print_profile_table(cfg, ic, y, u, T) -> None:
    '''Relaxed-from-rest profile accuracy, sampled across the channel.'''
    order = np.argsort(y)
    y_s, u_s, T_s = y[order], u[order], T[order]

    table = Table(title="Couette flow: profile after relaxing FROM REST (vs. exact steady solution)")
    table.add_column("y",            justify="right")
    table.add_column("u (measured)", justify="right")
    table.add_column("u (exact)",    justify="right")
    table.add_column("% err (u)",    justify="right")
    table.add_column("T (measured)", justify="right")
    table.add_column("T (exact)",    justify="right")

    # Sample ~10 stations across the channel rather than dumping every quadrature node.
    for idx in np.linspace(0, len(y_s) - 1, 10).astype(int):
        yy   = y_s[idx]
        ue   = float(ic.exact_velocity(yy))
        Te   = float(ic.exact_temperature(cfg, yy))
        pct  = 100.0 * (u_s[idx] - ue) / ic.U_WALL          # normalized by wall speed (u -> 0 at a wall)
        col  = "green" if abs(pct) < 1.0 else ("yellow" if abs(pct) < 2.0 else "red")
        table.add_row(f"{yy:.4f}", f"{u_s[idx]:+.5f}", f"{ue:+.5f}",
                      f"[{col}]{pct:+.3f}%[/{col}]", f"{T_s[idx]:.6f}", f"{Te:.6f}")

    Console().print(table)
    Console().print(f"  (% err in u is normalized by the wall speed U = {ic.U_WALL}, since u -> 0 at the "
                    f"bottom wall)")
    Console().print(f"  viscous-heating bump amplitude: [bold]{ic.viscous_heating_amplitude(cfg):.5f}[/bold] "
                    f"(the eta(1-eta) rise in T, on top of the linear wall-to-wall conduction)")


def _plot_profiles(cfg, ic, y, u, T) -> Path:
    '''
    The money picture for a wall BC: measured u(y) and T(y) against the exact solution.

    The velocity panel shows no-slip doing its job -- u pinned to 0 at the stationary bottom wall and to
    U at the driven top wall, linear in between. The temperature panel shows the isothermal walls plus
    the viscous-heating bump; the dotted line is the pure linear conduction profile, so the gap between
    it and the exact curve IS the viscous heating.
    '''
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    order = np.argsort(y)
    y_s, u_s, T_s = y[order], u[order], T[order]
    y_fine = np.linspace(0.0, ic.H, 200)

    fig, (ax_u, ax_T) = plt.subplots(1, 2, figsize=(10, 4.5))

    # --- velocity ----------------------------------------------------------------------------------
    ax_u.plot(ic.exact_velocity(y_fine), y_fine, '-', color="tab:gray", lw=2, label="exact")
    ax_u.plot(u_s, y_s, 'o', color="tab:blue", ms=4, label="DGSEM (relaxed from rest)")
    ax_u.set_xlabel("u")
    ax_u.set_ylabel("y")
    ax_u.set_title("Velocity: the NO-SLIP walls")
    ax_u.grid(True, ls=":", alpha=0.6)
    ax_u.legend()

    # --- temperature -------------------------------------------------------------------------------
    T_linear = ic.T_BOTTOM + (ic.T_TOP - ic.T_BOTTOM) * (y_fine / ic.H)
    ax_T.plot(ic.exact_temperature(cfg, y_fine), y_fine, '-', color="tab:gray", lw=2, label="exact")
    ax_T.plot(T_linear, y_fine, ':', color="tab:orange", lw=1.5, label="pure conduction (no heating)")
    ax_T.plot(T_s, y_s, 'o', color="tab:red", ms=4, label="DGSEM (relaxed from rest)")
    ax_T.set_xlabel("T")
    ax_T.set_ylabel("y")
    ax_T.set_title("Temperature: ISOTHERMAL walls + viscous heating")
    ax_T.grid(True, ls=":", alpha=0.6)
    ax_T.legend()

    fig.suptitle("Compressible Couette flow: relaxed from rest vs. exact steady solution")
    fig.tight_layout()
    out_path = FIG_DIR / "couette_profiles.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# -------------------------------------------------------------------------------------------------
# 1. STEADY RESIDUAL  (fast: no time stepping)
# -------------------------------------------------------------------------------------------------

@pytest.mark.numerics
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_exact_couette_is_steady_and_converges_spectrally():
    '''
    Feed the solver the exact steady solution: the residual must be small and must fall SPECTRALLY as
    the polynomial degree rises.

    It is not machine-zero, and that is expected: u and T are polynomials (degree 1 and 2), but the
    density rho = gamma M^2 P0 / T(y) is a RATIONAL function, so it cannot be represented exactly in the
    polynomial basis. The residual therefore measures the interpolation error of rho -- which, rho being
    analytic and pole-free on the channel, decays exponentially with P. That exponential decay IS the
    statement that the wall BCs are high-order consistent; an O(1)-inconsistent wall would floor the
    residual at some fixed level no matter how much the degree grew.
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

    # Spectral (exponential) decay: every refinement must cut the residual by a healthy factor. A wall
    # BC that was merely INCONSISTENT would stall here instead of continuing to fall.
    assert np.all(np.diff(norms) < 0), (
        f"steady residual must fall monotonically under p-refinement, got {norms} for P={degrees}"
    )
    assert norms[-1] < norms[0] / 50.0, (
        f"steady residual decayed only from {norms[0]:.2e} to {norms[-1]:.2e} over P={degrees[0]}..{degrees[-1]}; "
        f"that is far too slow for an analytic solution and points at an inconsistent wall BC."
    )
    assert norms[-1] < 1e-4, f"steady residual {norms[-1]:.2e} at P={degrees[-1]} is too large for the exact solution"


@pytest.mark.numerics
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_adiabatic_walls_break_the_exact_solution():
    '''
    NEGATIVE CONTROL, without which the test above would be vacuous.

    Couette flow reaches a steady temperature only because the viscous heat generated in the fluid is
    CONDUCTED OUT through the two walls -- the exact solution has a nonzero dT/dy at each wall. An
    adiabatic wall forbids precisely that heat flux. So if we keep the same exact solution but declare
    the walls adiabatic, it can no longer be steady, and the residual must jump by orders of magnitude.

    This proves the residual test above is actually SENSITIVE to what the wall does with the heat flux.
    '''
    mesh_iso, cfg_iso = _build_mesh(poly_deg=5)
    initialize(cfg_iso, CASE_DIR / str(cfg_iso.initialization.IC_file), mesh_iso)
    r_isothermal = _residual_norm(mesh_iso, cfg_iso)

    mesh_ad, cfg_ad = _build_mesh(poly_deg=5, wall_type='adiabatic_no_slip_wall')
    initialize(cfg_ad, CASE_DIR / str(cfg_ad.initialization.IC_file), mesh_ad)
    r_adiabatic = _residual_norm(mesh_ad, cfg_ad)

    assert r_adiabatic > 100.0 * r_isothermal, (
        f"declaring the walls ADIABATIC left the residual at {r_adiabatic:.2e} vs {r_isothermal:.2e} for the "
        f"correct isothermal walls. The steady-residual test is not sensitive to the wall heat flux, so it "
        f"cannot be validating the thermal BC."
    )


# -------------------------------------------------------------------------------------------------
# 2. RELAXATION FROM REST  (slow: the real end-to-end validation)
# -------------------------------------------------------------------------------------------------

@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_wall_bc_is_stable_and_relaxes_at_the_analytic_rate():
    '''
    REGRESSION TEST for the wall instability -- and the test whose absence let it through.

    THE BUG: the wall is a Dirichlet condition imposed WEAKLY. Without a penalty term there is nothing
    damping a boundary-condition violation, so one GROWS. A Couette run started from rest decayed toward
    the exact solution until t~4, then diverged exponentially (rate ~0.4) and killed itself on a negative
    pressure by t~12. It hit stationary walls as well as moving ones.

    WHY EVERY OTHER TEST MISSED IT: every wall term in the scheme -- the BR1 gradient lift (q_w - q^-),
    the flux difference F_v(q_w) - F_v(q^-), and the penalty itself -- is proportional to the BC
    VIOLATION. On the exact solution, q_w == q^-, so all of them vanish IDENTICALLY: the wall contributes
    nothing to the residual at all. The steady-residual test, its spectral-convergence sweep, and even
    "initialize the exact solution and march" are therefore structurally incapable of seeing ANY error in
    the wall treatment. They only ever confirmed that the exact solution is an equilibrium -- never that
    it is a STABLE one.

    THE FIX IS THE TEST: perturb the wall and watch. Start the fluid at rest against the moving wall (a
    100%-of-U boundary-condition violation) and require the violation to decay at the rate the physics
    says it must. Couette startup has the classical solution

        u(y,t) = U y/H  -  (2U/pi) sum_n (1/n) sin(n pi y/H) exp(-n^2 t / tau),   tau = H^2/(nu pi^2)

    so once the higher modes have died the error must follow 0.637 U exp(-t/tau). Asserting against that
    ANALYTIC DECAY RATE -- not merely "the error is small" -- is what makes this test able to distinguish
    a stable wall from an unstable one: the unstable scheme also passed through small errors on its way up.
    '''
    mesh, cfg = _build_mesh(poly_deg=2)
    ic        = _load_ic_module(cfg)

    nu  = 1.0 / cfg.physics.Re
    tau = ic.H**2 / (nu * np.pi**2)          # viscous relaxation time of the slowest (n=1) mode
    amp = 2.0 / np.pi                        # |b_1|/U for the linear initial deviation

    # Fluid AT REST against a wall moving at U: the boundary condition is violated by 100% of U.
    gamma   = cfg.physics.gamma
    gammaM2 = gamma * cfg.physics.mach_ref**2
    p0      = ic.T_BOTTOM / gammaM2
    for e in mesh.elements:
        q = np.zeros(e.quad_node_coords.shape[:-1] + (cfg.physics.num_eq,))
        q[..., 0] = 1.0
        q[..., 3] = p0 / (gamma - 1.0)
        e.solution = q

    # March, sampling the error against the analytic decay at each checkpoint.
    checkpoints = [2.0, 4.0, 6.0]
    t, measured = 0.0, []
    for target in checkpoints:
        while t < target - 1e-12:
            dt = min(compute_dt(mesh, cfg), target - t)
            time_step(mesh, cfg, t, dt)
            t += dt
        y, u, _ = _sample_profiles(mesh, cfg)
        measured.append(np.max(np.abs(u - ic.exact_velocity(y))) / ic.U_WALL)

    measured = np.array(measured)
    theory   = amp * np.exp(-np.array(checkpoints) / tau)
    ratio    = measured / theory

    _print_decay_table("Couette", checkpoints, measured, theory, ratio, tau)

    # 1. STABILITY: the violation must decay, monotonically. The unstable scheme decayed to ~1.7% by
    #    t=4 and was back up to 4.2% by t=6 -- so "is it small?" is NOT enough; it must never turn around.
    assert np.all(np.diff(measured) < 0), (
        f"the wall-condition violation is GROWING: {100*measured} % at t={checkpoints}. The weakly-imposed "
        f"Dirichlet wall is unstable -- check the interior penalty in wall_viscous_normal_flux."
    )

    # 2. RATE, TWO-SIDED. Bounding the ratio only from ABOVE would be a half-test: it would catch a wall
    #    that transfers momentum too slowly, but an over-estimated analytic amplitude (or a wall that
    #    over-drives the fluid) would sail through as a comfortably small ratio. Pin it from both sides.
    assert np.all((ratio > 0.5) & (ratio < 2.0)), (
        f"the violation is not decaying at the analytic rate: measured/analytic = {ratio} at "
        f"t={checkpoints} (want ~1). The wall is not transferring momentum at the rate the viscosity "
        f"dictates."
    )
    assert measured[-1] < 0.01, (
        f"the wall-condition violation is still {100*measured[-1]:.2f}% of U after {checkpoints[-1]} "
        f"(~{checkpoints[-1]/tau:.1f} relaxation times); it should be well under 1%."
    )


@pytest.mark.physics
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_couette_relaxes_from_rest_to_the_exact_profile():
    '''
    THE test for no-slip. Start the fluid completely at rest and let the moving top wall drag it into
    motion. After ~6 viscous relaxation times it must land on the exact Couette profile.

    A slip wall transmits NO tangential momentum, so under a slip wall this run would simply stay at
    rest forever -- the linear velocity profile can only appear if no-slip is genuinely transferring the
    wall's motion into the fluid. Likewise, the temperature can only settle if the isothermal walls are
    conducting the viscous heat back out.

    Note u and T are compared, but NOT rho: the base pressure P0 is a free constant of the steady
    solution, and in this closed domain it is pinned by the conserved total mass rather than by our
    (arbitrary) choice in the IC file. u and T are independent of P0; rho is not.
    '''
    mesh, cfg = _build_mesh(poly_deg=2)     # low degree keeps the ~1e4 explicit viscous steps affordable
    ic        = _load_ic_module(cfg)

    # --- Initial condition: FLUID AT REST, uniform density and wall temperature -------------------
    gamma   = cfg.physics.gamma
    gammaM2 = gamma * cfg.physics.mach_ref**2
    T0      = ic.T_BOTTOM
    p0      = T0 / gammaM2                          # so rho = 1
    rho0    = 1.0
    q_rest  = np.array([rho0, 0.0, 0.0, p0 / (gamma - 1.0)])
    for e in mesh.elements:
        e.solution = np.ones(e.quad_node_coords.shape[:-1] + (cfg.physics.num_eq,)) * q_rest

    # --- March to (near) steady state --------------------------------------------------------------
    t, T_final = 0.0, cfg.time_stepping.final_time
    while t < T_final:
        dt = min(compute_dt(mesh, cfg), T_final - t)
        time_step(mesh, cfg, t, dt)
        t += dt

    y, u, T = _sample_profiles(mesh, cfg)

    u_exact = ic.exact_velocity(y)
    T_exact = ic.exact_temperature(cfg, y)

    # Report: profile table to the terminal + a u(y)/T(y) plot saved under outputs/figures/.
    _print_profile_table(cfg, ic, y, u, T)
    fig_path = _plot_profiles(cfg, ic, y, u, T)
    Console().print(f"  profile plot saved to: [green]{fig_path}[/green]")

    # 1. The fluid actually MOVED. (A slip wall would leave it at rest; this is the headline assertion.)
    assert np.max(np.abs(u)) > 0.5 * ic.U_WALL, (
        f"the fluid barely moved (max |u| = {np.max(np.abs(u)):.3e}, wall speed {ic.U_WALL}). The no-slip "
        f"wall is not transferring its motion into the fluid."
    )

    # 2. It moved into the EXACT linear profile. Tolerance covers the ~0.3% residual transient at 6 tau
    #    plus the P=2 spatial error.
    u_err = np.max(np.abs(u - u_exact)) / ic.U_WALL
    assert u_err < 0.02, f"velocity profile is off by {u_err*100:.2f}% of the wall speed (max |u - u_exact|)"

    # 3. The temperature settled onto the conduction + viscous-heating balance. First confirm the
    #    viscous-heating bump is actually big enough for this to be measuring the energy balance at all.
    bump = ic.viscous_heating_amplitude(cfg)
    assert bump > 0.01, f"viscous heating bump {bump:.4f} is too small for this test to be meaningful"

    T_err = np.max(np.abs(T - T_exact))
    assert T_err < 0.02 * bump + 0.01, (
        f"temperature profile is off by {T_err:.4e} (viscous-heating amplitude is {bump:.4e}). The "
        f"isothermal walls are not carrying the viscous heat out to the right steady balance."
    )
