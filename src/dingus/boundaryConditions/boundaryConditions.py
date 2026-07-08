# src/dingus/boundaryConditions/boundaryConditions.py

import numpy as np
from dingus.config import CaseCfg
from dingus.mesh import mortar_class

'''
Boundary conditions in a DG scheme are imposed WEAKLY, through the numerical (interface) flux.
At a boundary face the scheme still evaluates a Riemann problem between two states:

    q_minus : the interior trace (from the element that owns the boundary face)
    q_plus  : an EXTERIOR "ghost" state that encodes the boundary condition

There is no physical element on the outside, so `q_plus` is *manufactured* from the BC type.
Feeding (q_minus, q_plus) into the same `compute_numerical_flux` used for interior interfaces
then enforces the condition automatically (e.g. upwinding decides whether the interior or the
prescribed exterior state is used, based on the flow direction).

So given a boundary mortar and its interior trace, this module will return q_plus.
The surface term calls it ONLY for boundary mortars; interior mortars get q_plus from the
neighbor element directly.
'''

def exterior_state(mort     : mortar_class.SpectralMortar, 
                   q_minus  : np.ndarray                 ,
                   case_cfg : CaseCfg                    ,
                   t        : float=0.0) -> np.ndarray:
    '''
    Returns the exterior ("ghost") state q_plus at a boundary mortar, to be paired with the
    interior trace q_minus in the numerical flux.

    Inputs:
    - mort     : the boundary SpectralMortar
    - q_minus  : (P+1, num_eq) interior face trace at the P+1 face nodes.
    - case_cfg : validated case configuration.
    - t        : current time (Only required for unsteady boundary conditions).

    Outputs:
    - q_plus   : (P+1, num_eq) exterior state.
    '''

    # Extract the boundary condition configuration object from the mortar
    bc = mort.boundary_condition

    match bc.type:
        case 'inflow':
            # A prescribed incoming state. 
            q_plus = np.ones_like(q_minus) * bc.state  # works for any number of equations and nodes

        case 'outflow':
            # The exterior state equals the interior state, so the numerical flux reduces to 
            # the interior physical flux
            q_plus = q_minus

        case 'periodic':
            # Periodic faces are NOT handled here — they are paired with their partner mortar in
            # apply_boundary_conditions and treated like an interior interface, pulling q_plus from
            # the partner element's trace. So exterior_state should never be called on them.
            raise NotImplementedError(
                "Periodic BCs are handled via mortar pairing in the surface term, not exterior_state."
            )

        case _:
             raise NotImplementedError(
                f"Boundary condition type '{bc.type}' not yet implemented."
            )

    return q_plus