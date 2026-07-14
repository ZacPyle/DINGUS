# src/dingus/runner.py

from pathlib import Path
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from dingus.config import CaseCfg
from dingus.mesh import mesh_class
from dingus.initialConditions.initialize_solution import initialize
from dingus.io.outputFileWriter import write_state_vars_to_file, write_pvd_collection
from dingus.timeIntegrators.timeIntegration import compute_dt, time_step

def run_case(cfg: CaseCfg, case_dir: Path = Path('.')) -> None:
    '''
    Top-level driver of DINGUS simuilations. The basic steps taken by this runner are:
        1. Build the mesh
        2. Initialize the solution on the mesh
        3. Compute volume flux / divergence
        4. Compute surface flux (this is where/how boundary conditions are applied)
        5. Advance one time step
        6. Execute IO operations if necessary
        7. Check if simulation has fullfilled runtime requirements; if not back to step 1

    User inputs are relayed to simulations via the "cfg" Case Configuration object. The 
    cfg is constructed from a user-written control.yaml file.

    An example of calling this runner on a case is:


    Inputs:
    - cfg      : validated case configuration.
    - case_dir : directory of the control file; mesh_file / IC_file / source_file are resolved
                 relative to it.
    '''

    case_dir = Path(case_dir)

    # Resolve the source-term file relative to the control file's directory, exactly like mesh_file and
    # IC_file below. Unlike those two (loaded here at the call site), the source term is loaded deep in
    # the residual (add_source_terms, every RK stage), which never sees case_dir -- so we make the path
    # absolute up front. Path join is a no-op if the user already gave an absolute path.
    if cfg.source.source_method != 'none' and cfg.source.source_file is not None:
        cfg.source.source_file = str(case_dir / cfg.source.source_file)

    ##### 1. Build the mesh #####
    mesh = mesh_class.Mesh()
    mesh.read_mesh(case_dir / cfg.mesh.mesh_file)
    mesh.construct_mesh(cfg)

    ##### 2. Initialize the solution #####
    initialize(cfg, case_dir / str(cfg.initialization.IC_file), mesh)

    ##### 2a. Write out the initial state (step 0) #####
    write_state_vars_to_file(input_mesh   = mesh, 
                             input_config = cfg , 
                             time         = 0.0 , 
                             step         = 0   , 
                             CASE_DIR     = case_dir)
    
    ##### 2b. Initialize time marching loop #####
    t            = cfg.time_stepping.start_time
    step         = 0
    final_time   = cfg.time_stepping.final_time
    next_io_time = t + cfg.io.output_interval

    # ParaView .pvd collection: (time, step) for every frame written, so the timeline shows real
    # simulation times. output_root is the dir that contains vtk/ (where the .pvd lives).
    output_root  = case_dir / cfg.io.output_dir
    pvd_records  = [(t, 0)]
    write_pvd_collection(cfg.name, pvd_records, output_root)


    ##### 3 - 7: Time Loop #####
    # Use rich.Progress to create a live-updating status line. 
    # total=final_time + completed=t make the bar
    # track SIMULATION TIME (0 -> final_time), not step count. 
    # The columns are, left to right:
    #   description (step number) | bar | custom info string | elapsed wall time | ETA
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.fields[info]}"),
        TextColumn("|"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("step       0", total=final_time, info="")

        ##### BEGINNING OF TIME INTEGRATION HERE #####
        while t < final_time:
            dt = compute_dt(mesh, cfg)

            # Clamp so the step lands exactly on the next output time AND on final_time -- dt is
            # never allowed to step past either. Both are only ever SMALLER than the CFL dt, so
            # stability is preserved; the clamp just trims the last sub-step before each target.
            dt = min(dt, next_io_time - t, final_time - t)
            
            ##### Steps 3, 4, and 5 #####
            # Volume flux, surface flux, and time integration happens under the hood here. 
            # time_step() points to a time integrator, that time_integrator calls compute_residual(),
            # and compute_residual() computes the volumetric flux divergence and the surface flux correction
            # to output the residual.
            time_step(mesh, cfg, t, dt)

            # Update time and time step
            t    += dt
            step += 1

            ##### 6. IO operations #####
            # Write output whenever we cross an output-interval boundary.
            doing_io = t >= next_io_time - 1e-12
            if doing_io:
                write_state_vars_to_file(input_mesh=mesh, input_config=cfg, time=t, step=step, CASE_DIR=case_dir)
                next_io_time += cfg.io.output_interval
                pvd_records.append((t, step))
                write_pvd_collection(cfg.name, pvd_records, output_root)   # refresh so a crash still leaves a valid .pvd
                # Log the IO event as a line ABOVE the live bar (rich pins the bar to the bottom).
                progress.console.print(f"[cyan]  IO -> wrote output: step {step}, t = {t:.5f}[/cyan]")

            # Refresh the live status line: advance to current sim time, update the fields.
            progress.update(
                task,
                completed=t,
                description=f"step {step:>7d}",
                info=f"dt={dt:.3e}  t={t:.5f}" + ("  [cyan]<IO>[/cyan]" if doing_io else "        "),
            )

    # ---- 5. Final write (guarantees the last state is on disk even if final_time isn't on an interval) ----
    write_state_vars_to_file(input_mesh=mesh, input_config=cfg, time=t, step=step, CASE_DIR=case_dir)
    if pvd_records[-1] != (t, step):            # avoid a duplicate frame if the last step was already an output
        pvd_records.append((t, step))
    write_pvd_collection(cfg.name, pvd_records, output_root)
    progress.console.print(f"[bold green]Done.[/bold green] {step} steps to t = {t:.5f}")


# In cli.py, thread the control file's directory into run_case:
#
#     from pathlib import Path
#     @app.command()
#     def run(case: str):
#         case_path = Path(case)
#         cfg = load_case_yaml(case_path)
#         print(f"[bold green]Running case:[/bold green] {cfg.name}")
#         run_case(cfg, case_path.parent)