# tests/test_taylor_green_vortex.py
'''
End-to-end TIME-MARCHING verification of the compressible Navier-Stokes solver (BR1 viscous terms),
driven by the tests/test_case_taylor_green_vortex/ case directory (control.yaml, inputs/Square.mesh,
and inputs/initial_condition_tgv.py).

THE TAYLOR-GREEN VORTEX DECAYS AT THE ANALYTIC RATE. The 2D Taylor-Green vortex on a periodic box is
an exact solution of the INCOMPRESSIBLE Navier-Stokes equations whose velocity decays as
exp(-2 K^2 t / Re), and therefore whose kinetic energy decays as

        KE(t) = KE(0) * exp(-4 K^2 t / Re)          (K = the box wavenumber)

Run at the case's low reference Mach, the compressible solver must reproduce this closely; the small
residual is the O(M^2) compressibility correction, which is real physics rather than numerical error.

This is the headline "is my whole NSE solver right" check. Its power is that the decay rate is a single
QUANTITATIVE number that the entire viscous pipeline has to conspire to get right: a wrong 1/Re
prefactor, a botched (2/3) div(u) term in the stress tensor, a wrong kappa in the heat flux, or a BR1
gradient that is off by a factor all still produce a plausible-looking decaying vortex -- but they all
shift this rate.

Unlike the isolated unit tests (gradient pass, viscous flux algebra), this drives the FULL pipeline in a
time loop: gradient pass -> combined inviscid+viscous volume flux -> BR1 interface coupling ->
viscous-limited timestep -> RK4.

(Free-stream preservation -- "a uniform state must not drift" -- used to live here too. It is a
different kind of statement and has nothing to do with the vortex, so it now has its own file:
tests/SingleTests/test_freestream_preservation.py.)

Produces a table in the terminal (use `pytest -s`) comparing the measured and analytic kinetic energy
at each IO time, and a decay plot in tests/test_case_taylor_green_vortex/outputs/figures/.

Runs a few hundred RK4 steps, so this is a SLOW test.
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

# This test drives a dedicated case directory (the standard pytest-per-case pattern).
CASE_DIR  = Path(__file__).resolve().parent / "test_case_taylor_green_vortex"
CTRL_FILE = CASE_DIR / "control.yaml"
FIG_DIR   = CASE_DIR / "outputs" / "figures"

# The test owns its timing so control.yaml is free for interactive runs. The 3% decay-rate tolerance is
# calibrated to this final_time; output_interval only sets how many rows the decay TABLE reports, so it
# is owned here too to keep that table stable regardless of the case file.
TGV_FINAL_TIME      = 0.2
TGV_OUTPUT_INTERVAL = 0.05


def _load_ic_module(cfg):
    '''Import the case's initial-condition module to reuse its wavenumber K (single source of truth
    for the analytic decay rate), mirroring how the vortex tests pull U_INF/V_INF.'''
    ic_path = CASE_DIR / str(cfg.initialization.IC_file)
    spec    = importlib.util.spec_from_file_location("tgv_ic", ic_path)
    module  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_mesh(poly_deg: int | None = None):
    '''Load the Taylor-Green case (physics/BCs come from control.yaml), optionally override poly_deg,
    and construct the mesh.'''
    cfg = load_case_yaml(CTRL_FILE)
    if poly_deg is not None:
        cfg.mesh.poly_deg = poly_deg
    cfg.time_stepping.final_time = TGV_FINAL_TIME       # test owns its timing; control.yaml is free
    cfg.io.output_interval       = TGV_OUTPUT_INTERVAL
    mesh = mesh_class.Mesh()
    mesh.read_mesh(CASE_DIR / cfg.mesh.mesh_file)
    mesh.construct_mesh(cfg)
    return mesh, cfg


def _kinetic_energy(mesh) -> float:
    '''Domain-integrated kinetic energy  integral 1/2 (rho u^2 + rho v^2)  (quadrature-weighted).'''
    ke = 0.0
    for e in mesh.elements:
        w   = e.jacobian_det * mesh.quad_weights[:, None] * mesh.quad_weights[None, :]
        rho = e.solution[..., 0]
        mx, my = e.solution[..., 1], e.solution[..., 2]
        ke += np.sum(w * 0.5 * (mx * mx + my * my) / rho)
    return float(ke)


def _print_table(io_times, ke_measured, ke_analytic, rate_measured, rate_analytic) -> None:
    '''
    Pretty-print measured-vs-analytic kinetic energy at each IO time (visible with `pytest -s`).

    The per-time percentage error is the honest headline number: it folds in BOTH the O(M^2)
    compressibility offset (the vortex is an INCOMPRESSIBLE exact solution) and any discretization
    error, and it grows slowly with t as the two solutions drift apart.
    '''
    table = Table(title="Taylor-Green vortex: kinetic-energy decay vs. analytic")
    table.add_column("t",              justify="right")
    table.add_column("KE (measured)",  justify="right")
    table.add_column("KE (analytic)",  justify="right")
    table.add_column("abs error",      justify="right")
    table.add_column("% error",        justify="right")

    for t, ke_m, ke_a in zip(io_times, ke_measured, ke_analytic):
        err     = ke_m - ke_a
        pct     = 100.0 * err / ke_a
        colour  = "green" if abs(pct) < 1.0 else ("yellow" if abs(pct) < 3.0 else "red")
        table.add_row(f"{t:.3f}", f"{ke_m:.6e}", f"{ke_a:.6e}",
                      f"{err:+.3e}", f"[{colour}]{pct:+.3f}%[/{colour}]")

    Console().print(table)

    rate_pct = 100.0 * abs(rate_measured - rate_analytic) / rate_analytic
    Console().print(f"  KE decay rate: measured [bold]{rate_measured:.5f}[/bold]  vs  "
                    f"analytic 4K^2/Re = [bold]{rate_analytic:.5f}[/bold]   "
                    f"([bold]{rate_pct:.3f}%[/bold] error)")
    Console().print("  (the residual is the O(M^2) compressibility correction: the Taylor-Green vortex "
                    "is an exact solution of the INCOMPRESSIBLE equations)")


def _plot_decay(times, kes, io_times, ke_measured, ke_analytic, ke0, rate_analytic) -> Path:
    '''
    Save the kinetic-energy decay: simulation vs. the analytic exponential.

    Plotted on a LOG y-axis, where the analytic KE(t) = KE(0) exp(-4K^2 t/Re) is a straight line -- so
    the eye reads the DECAY RATE directly as a slope, and any drift in the rate (as opposed to a mere
    offset) shows up as a bend. The lower panel shows the percentage error at the IO times.
    '''
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax, ax_err) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1]})

    # --- top: the decay itself ---------------------------------------------------------------------
    t_fine = np.linspace(0.0, times[-1], 300)
    ax.semilogy(t_fine, ke0 * np.exp(-rate_analytic * t_fine), '--', color="tab:gray",
                label=r"analytic  $KE_0\,e^{-4K^2t/Re}$")
    ax.semilogy(times, kes, '-', color="tab:blue", lw=1.6, label="DGSEM (every step)")
    ax.semilogy(io_times, ke_measured, 'o', color="tab:red", ms=5, label="IO samples")
    ax.set_ylabel("Kinetic energy")
    ax.set_title("Taylor-Green vortex: viscous kinetic-energy decay")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()

    # --- bottom: percentage error at the IO times --------------------------------------------------
    pct = 100.0 * (np.asarray(ke_measured) - np.asarray(ke_analytic)) / np.asarray(ke_analytic)
    ax_err.plot(io_times, pct, 'o-', color="tab:red", ms=4)
    ax_err.axhline(0.0, color="tab:gray", ls="--", lw=1)
    ax_err.set_xlabel("Time  t")
    ax_err.set_ylabel("% error")
    ax_err.grid(True, ls=":", alpha=0.6)

    fig.tight_layout()
    out_path = FIG_DIR / "taylor_green_decay.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


@pytest.mark.physics
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_nse_taylor_green_decay_rate():
    '''Taylor-Green kinetic energy must decay monotonically at the analytic rate 4 K^2 / Re.'''
    mesh, cfg = _build_mesh()                              # poly_deg from control.yaml
    initialize(cfg, CASE_DIR / str(cfg.initialization.IC_file), mesh)

    K  = _load_ic_module(cfg).K                            # wavenumber, straight from the IC module
    Re = cfg.physics.Re
    T  = cfg.time_stepping.final_time

    rate_analytic = 4.0 * K * K / Re                       # incompressible KE decay rate
    ke0           = _kinetic_energy(mesh)

    # The IO times we report in the table, taken from the case's own output_interval (always including
    # the final time, so the table ends where the run does).
    interval = cfg.io.output_interval
    targets  = [float(x) for x in np.arange(interval, T - 1e-12, interval)] + [float(T)]

    # March. Every step feeds the decay-rate fit and the plotted curve; the run is additionally clamped
    # to land EXACTLY on each IO time, so the table samples the true solution rather than an interpolant.
    times, kes                = [0.0], [ke0]
    io_times, ke_measured     = [0.0], [ke0]
    t = 0.0
    for target in targets:
        while t < target - 1e-12:
            dt = min(compute_dt(mesh, cfg), target - t)
            time_step(mesh, cfg, t, dt)
            t += dt
            times.append(t)
            kes.append(_kinetic_energy(mesh))
        io_times.append(t)
        ke_measured.append(kes[-1])

    times, kes = np.array(times), np.array(kes)
    ke_analytic = [ke0 * np.exp(-rate_analytic * ti) for ti in io_times]

    rate_measured = -np.polyfit(times, np.log(kes), 1)[0]   # KE ~ exp(-rate t)

    # Report: table to the terminal + a decay plot saved under outputs/figures/.
    _print_table(io_times, ke_measured, ke_analytic, rate_measured, rate_analytic)
    fig_path = _plot_decay(times, kes, io_times, ke_measured, ke_analytic, ke0, rate_analytic)
    Console().print(f"  decay plot saved to: [green]{fig_path}[/green]")

    # 1. monotonic decay (viscosity only removes energy)
    assert np.all(np.diff(kes) < 0), "kinetic energy must decay monotonically under viscous dissipation"

    # 2. decay rate matches the analytic incompressible rate to within the O(M^2) compressibility
    #    correction (empirically ~0.3% at M = 0.1; 3% is a comfortable, non-flaky bound).
    rel_err = abs(rate_measured - rate_analytic) / rate_analytic
    assert rel_err < 0.03, (
        f"Taylor-Green decay rate {rate_measured:.4f} differs from analytic {rate_analytic:.4f} "
        f"by {rel_err*100:.2f}% (> 3%)"
    )
