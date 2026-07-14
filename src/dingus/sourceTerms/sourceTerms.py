# src/dingus/sourceTerms/sourceTerms.py

import importlib.util
import numpy as np
from dingus.config import CaseCfg
from pathlib import Path

'''
Source terms turn the homogeneous conservation law into a FORCED one:

        q_t + div(F_vec) = S(q, x, t)

so the semi-discrete update picks up one extra, purely LOCAL term (no derivatives, no neighbours):

        q_t = -(1/J)[ d(F~)/dxi + d(G~)/deta + ... ]  +  S

Because S is evaluated pointwise at the quadrature nodes, adding it to the residual is the last step
of compute_residual, AFTER the -1/J scaling (the flux divergence is what carries the metric Jacobian;
the source does not).

Two things need this:

  1. FORCED POISEUILLE. A channel that is periodic in x has no mean pressure drop to push the fluid,
     so we drive it with a constant body force instead: S = [0, G, 0, G*u], G = -dp/dx. (The energy
     component G*u is the work done by that force -- forget it and the flow silently heats wrong.)
     Couette needs no source at all: it is driven by the moving WALL.

  2. MMS (Method of Manufactured Solutions). Choose any smooth q_exact you like -- it need not solve
     anything -- and push it through the PDE. The leftover
            S := (q_exact)_t + div(F_vec(q_exact))
     is, by construction, exactly the forcing that MAKES q_exact a solution of the forced system. Run
     the code with that S, compare against q_exact, refine, and you get a rigorous order-of-accuracy
     measurement for terms (like the viscous ones) that have no convenient analytic test case.

The user supplies S the same way they supply an initial condition: a Python file with a function of a
fixed signature, named in the case YAML. The signature carries q as well as (x, t), since a body force
generally depends on the state (rho, u) while an MMS source usually does not.

        1D:  source_term(case_config, q, x, t)
        2D:  source_term(case_config, q, x, y, t)
        3D:  source_term(case_config, q, x, y, z, t)

and must return an array shaped like q -- (P+1,)*ndim + (num_eq,) -- one source value per equation
per quadrature node.
'''

# The user's source module is imported ONCE and cached here, keyed by file path. compute_residual runs
# this on every RK stage of every step, so re-importing the module each call would dominate the runtime.
_SOURCE_CACHE: dict[str, object] = {}

def _load_source_function(source_file: Path):
    '''
    Dynamically load the user-defined module holding `source_term()`, and return that function.
    Mirrors initialize_solution._load_ic_module, but caches: this is on the hot path.
    '''

    source_file = Path(source_file)
    key         = str(source_file.resolve()) if source_file.is_file() else str(source_file)

    if key in _SOURCE_CACHE:
        return _SOURCE_CACHE[key]

    if not source_file.is_file():
        raise FileNotFoundError(f"Source term file not found! Specified path is: '{source_file}'")

    # Create a module spec from the specified source term file
    spec = importlib.util.spec_from_file_location("user_source", source_file)

    # Ensure the module spec was successfully created
    if spec is None:
        raise ImportError(f"Could not create a module spec from the source term file: '{source_file}'")
    if spec.loader is None:
        raise ImportError(f"Module spec for '{source_file}' has no loader.")

    # Load the module from the spec and make its contents available in this script.
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Check that the function 'source_term()' is defined in the user-defined source file
    if not hasattr(module, "source_term"):
        raise AttributeError(f"No function 'source_term()' found in: '{source_file}'")

    _SOURCE_CACHE[key] = module.source_term
    return module.source_term

def _call_source_function(src_func, q: np.ndarray, coords: np.ndarray, t: float,
                          case_cfg: CaseCfg) -> np.ndarray:
    '''
    Call the user's source_term() with the coordinate arguments its dimensionality expects. Unlike the
    initial condition (which may legitimately be lower-dimensional than the mesh), a source term must
    match the mesh dimension: it is the residual of THIS PDE on THIS mesh, so a mismatch is a bug.
    '''

    ndim = case_cfg.mesh.ndim

    match ndim:
        case 1:
            return src_func(case_cfg, q, coords[..., 0], t)
        case 2:
            return src_func(case_cfg, q, coords[..., 0], coords[..., 1], t)
        case 3:
            return src_func(case_cfg, q, coords[..., 0], coords[..., 1], coords[..., 2], t)
        case _:
            raise ValueError(f"Unsupported dimensionality for a source term: {ndim}")

def add_source_terms(mesh, case_cfg: CaseCfg, t: float = 0.0) -> None:
    '''
    Add the source term S(q, x, t) to every element's residual, IN PLACE.

    Called at the very end of compute_residual, after the residual has been scaled by -1/J, because S
    enters the semi-discrete equation as a bare additive term:  q_t = -(1/J)[div F~] + S.

    A no-op (and free) when the case declares no source term, which is the default.

    Inputs:
    - mesh     : constructed Mesh, with e.residual already holding -(1/J) div(F~) and e.solution current.
    - case_cfg : validated case configuration.
    - t        : current time (an MMS source is generally time-dependent).
    '''

    if case_cfg.source.source_method == 'none':
        return

    src_func = _load_source_function(Path(case_cfg.source.source_file))

    for e in mesh.elements:
        S = _call_source_function(src_func, e.solution, e.quad_node_coords, t, case_cfg)

        if S.shape != e.residual.shape:
            raise ValueError(
                f"source_term() returned shape {S.shape}, but the residual for this element has shape "
                f"{e.residual.shape}. The source must supply one value per equation per quadrature node."
            )

        e.residual += S

    return
