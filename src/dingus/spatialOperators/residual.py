# src/dingus/spatialOperators/residual.py

import numpy as np
from dingus.boundaryConditions.boundaryConditions import (exterior_state, gradient_exterior_state,
                                                          is_wall, is_prescribed, is_characteristic,
                                                          wall_viscous_normal_flux,
                                                          prescribed_viscous_normal_flux,
                                                          characteristic_inviscid_normal_flux)
from dingus.config import CaseCfg
from dingus.mesh import mesh_class
from dingus.physics import fluxes
from dingus.sourceTerms.sourceTerms import add_source_terms

'''
This module computes the DG spatial residual, R = dq/dt, for every element. In other words, it computes
the right-hand side of the semi-discrete system dq/dt = R(q). Recall that we are solving the MAPPED conservation
law on the reference square [-1,1]^2, so the semi-discrete equation is written as:

        J q_t + d/dxi(F~) + d/deta(G~) + d/dzeta(H~) = 0

where F~, G~, H~ are the CONTRAVARIANT fluxes — the physical flux vector projected onto the contravariant basis
vectors using the metric terms (Kopriva Ch. 6):

        F~ = (J a^1) . F_vec        G~ = (J a^2) . F_vec        H~ = (J a^3) . F_vec

Here, (J a^1), (J a^2), (J a^3) are the (Jacobian-scaled) contravariant basis vectors, already available
per element as jacobian_det * contravar_xi, jacobian_det * contravar_eta, and jacobian_det * contravar_zeta.

Thus, the semi-discrete update is:

        q_t = -(1/J) [ d/dxi(F~) + d/deta(G~) + d/dzeta(H~)  +  (surface coupling term) ]

VOLUME term:  divergence of the contravariant flux, computed with the derivative matrix D.
SURFACE term: at every face, REPLACE the interior boundary flux with the shared NUMERICAL
              flux. Strong form adds the lifted difference  (F~* - F~^-)  at each face.

Dimensionality is handled the same way as mapping.py: a thin `compute_residual` wrapper
dispatches to a per-dimension worker.

Face convention (e.connected_mortars[0..3] = faces 1..4):
    face 1 = bottom (eta=-1)   free axis = xi  (i)   prolong along eta with I_min
    face 2 = right  (xi =+1)   free axis = eta (j)   prolong along xi  with I_max
    face 3 = top    (eta=+1)   free axis = xi  (i)   prolong along eta with I_max
    face 4 = left   (xi =-1)   free axis = eta (j)   prolong along xi  with I_min

The face trace ordering (by i for faces 1/3, by j for faces 2/4) matches how mort.normal_vector
was built in compute_normals_2d, so face arrays line up node-for-node with the normals.
'''

# Per-face facedata: (prolong axis, which face-interp row, free-axis label).
#   'axis'  : which solution axis is the face-NORMAL axis to contract away (0=xi, 1=eta)
#   'side'  : 'min' -> xi/eta = -1 face,  'max' -> +1 face
#   'free'  : 'i' (xi) or 'j' (eta) -- the axis that survives along the face
_FACE_2D = {
    1: {'axis': 1, 'side': 'min', 'free': 'i'},  # bottom
    2: {'axis': 0, 'side': 'max', 'free': 'j'},  # right
    3: {'axis': 1, 'side': 'max', 'free': 'i'},  # top
    4: {'axis': 0, 'side': 'min', 'free': 'j'}   # left
}

#######################################################################################
##### HELPER FUNCTIONS ################################################################
#######################################################################################

def _prolong_to_face_2d(sol: np.ndarray, face_id: int, I_min: np.ndarray, I_max: np.ndarray) -> np.ndarray:
    '''
    Interpolate a (P+1, P+1, num_eq) solution to one face -> (P+1, num_eq) trace along the free axis.
    '''

    # Grab the facedata for the input face ID
    facedata = _FACE_2D[face_id]

    # Grab the face interpolation matrix
    I = I_min if facedata['side'] == 'min' else I_max

    #
    match facedata['axis']:
        case 0:
            # prolonging to an x-normal face
            return np.einsum('mi,ije->je', I, sol)  # (P+1, 2)

        case 1:
            # prolonging to a y-normal face
            return np.einsum('mj,ije->ie', I, sol)  # (P+1, 2)
        case _:
            raise ValueError(f"Error in prolonging solution to face along axis {facedata['axis']}.")

def _prolong_metric_to_face_2d(Ja: np.ndarray, face_id: int, I_min: np.ndarray, I_max: np.ndarray) -> np.ndarray:
    '''
    Interpolate a (P+1, P+1, 2) metric field (J*a^i) to a face -> (P+1, 2). Same contraction as
    _prolong_to_face but the trailing axis is the 2 spatial components, not num_eq.
    '''

    # Grab the facedata for the input face ID
    facedata = _FACE_2D[face_id]

    # Grab the face interpolation matrix
    I = I_min if facedata['side'] == 'min' else I_max

    # 
    match facedata['axis']:
        case 0:
            # prolonging to an x-normal face
            return np.einsum('mi,ijd->jd', I, Ja)  # (P+1, 2)

        case 1:
            # prolonging to a y-normal face
            return np.einsum('mj,ijd->id', I, Ja)  # (P+1, 2)
        
        case _:
            raise ValueError(f"Error in prolonging metrics to face along axis {facedata['axis']}.")

def _prolong_grad_to_face_2d(grad: np.ndarray, face_id: int,
                             I_min: np.ndarray, I_max: np.ndarray) -> np.ndarray:
    '''
    Interpolate a (P+1, P+1, num_eq, ndim) gradient field to one face -> (P+1, num_eq, ndim).
    Identical to _prolong_to_face_2d, but the trailing axis is now (num_eq, ndim) instead of just
    num_eq, so the einsum gains a 'd' index.
    '''
    facedata = _FACE_2D[face_id]
    I = I_min if facedata['side'] == 'min' else I_max
    match facedata['axis']:
        case 0:   # x-normal face
            return np.einsum('mi,ijed->jed', I, grad)   # (P+1, num_eq, ndim)
        case 1:   # y-normal face
            return np.einsum('mj,ijed->ied', I, grad)   # (P+1, num_eq, ndim)
        case _:
            raise ValueError(f"Error prolonging gradient to face along axis {facedata['axis']}.")
                
def _lift_face_to_volume_2d(corr: np.ndarray, face_id: int,
                            I_min: np.ndarray, I_max: np.ndarray, w: np.ndarray) -> np.ndarray:
    '''
    Lift a face correction (P+1, num_eq) back into a (P+1, P+1, num_eq) volume array using the
    boundary basis over the quadrature weights:  contribution at node (i,j) = corr[free] * L(normal)/w.

    For a normal axis = xi=+1 face, L_i(+1)/w_i comes from I_max[0]/w applied to the i axis, etc.
    '''
    facedata  = _FACE_2D[face_id]
    Irow      = (I_min if facedata['side'] == 'min' else I_max)[0]     # (P+1,) = L_k(-1) or L_k(+1)
    basis     = Irow / w                                               # (P+1,) boundary basis / weight

    if facedata['axis'] == 1:      # eta face: corr indexed by i (free), basis spreads over j (normal)
        return np.einsum('ie,j->ije', corr, basis)            # (P+1, P+1, num_eq)
    else:                          # xi face: corr indexed by j (free), basis spreads over i (normal)
        return np.einsum('je,i->ije', corr, basis)
    
def _lift_grad_face_to_volume_2d(corr   : np.ndarray, 
                                 face_id: int,
                                 I_min  : np.ndarray, 
                                 I_max  : np.ndarray, 
                                 w      : np.ndarray) -> np.ndarray:
    '''
    Lift a face correction (P+1, num_eq, ndim) back into a (P+1, P+1, num_eq, ndim) volume array.
    Identical to _lift_face_to_volume_2d, but the correction carries the extra trailing 
    spatial-direction (ndim) axis, so the einsum gains a 'd' index.
    '''
    facedata = _FACE_2D[face_id]
    Irow     = (I_min if facedata['side'] == 'min' else I_max)[0]     # (P+1,) = L_k(-1) or L_k(+1)
    basis    = Irow / w                                               # (P+1,) boundary basis / weight

    if facedata['axis'] == 1:      # eta face: corr indexed by i (free), basis spreads over j (normal)
        return np.einsum('ied,j->ijed', corr, basis)         # (P+1, P+1, num_eq, ndim)
    else:                          # xi face: corr indexed by j (free), basis spreads over i (normal)
        return np.einsum('jed,i->ijed', corr, basis)

def _compute_divergence_1d(mesh : mesh_class.Mesh, case_cfg : CaseCfg) -> None:
    """
    Computes the 1D strong-form DGSEM divergence (VOLUME TERM ONLY).
    """

    # Extract the derivative matrix from the mesh object
    D = mesh.deriv_mat  # shape (P+1, P+1)

    # Loop through all elements
    for e in mesh.elements:
        # Extract the solution and contravariant basis vectors
        q   = e.solution                       # shape (P+1, num_eq)
        J   = e.jacobian_det                   # shape (P+1)
        Ja1 = J[..., None] * e.contravar_xi    # shape (P+1, 2)

        # Compute the physical volume fluxes for the current solution
        F_vec = fluxes.compute_volume_flux(q, case_cfg, e.grad_q)  # shape (P+1, num_eq, 2)

        # Project the physical fluxes onto the contravariant basis vectors to get the 
        # contravariant flux terms.
        F_tilde = np.einsum('ied,id->ie', F_vec, Ja1)

        # Compute the derivatives of the contravariant fluxes using the derivative matrix
        dF_tilde_dxi  = np.einsum('il,le->ie', D, F_tilde)

        # Combine the derivatives to form the divergence term. This will be saved as the residual
        # in the element object, then overwritten later after the surface coupling term is added.
        e.residual = dF_tilde_dxi  # shape (P+1, num_eq)
    return

def _compute_divergence_2d(mesh : mesh_class.Mesh, case_cfg : CaseCfg) -> None:
    """
    Computes the 2D strong-form DGSEM divergence (VOLUME TERM ONLY).
    """

    # Extract the derivative matrix from the mesh object
    D = mesh.deriv_mat  # shape (P+1, P+1)

    # Loop through all elements
    for e in mesh.elements:
        # Extract the solution and contravariant basis vectors
        q   = e.solution                       # shape (P+1, P+1, num_eq)
        J   = e.jacobian_det                   # shape (P+1, P+1)
        Ja1 = J[..., None] * e.contravar_xi    # shape (P+1, P+1, 2)
        Ja2 = J[..., None] * e.contravar_eta   # shape (P+1, P+1, 2)

        # Compute the physical volume fluxes for the current solution
        F_vec = fluxes.compute_volume_flux(q, case_cfg, e.grad_q)  # shape (P+1, P+1, num_eq, 2)

        # Project the physical fluxes onto the contravariant basis vectors to get the 
        # contravariant flux terms.
        F_tilde = np.einsum('ijed,ijd->ije', F_vec, Ja1)
        G_tilde = np.einsum('ijed,ijd->ije', F_vec, Ja2)

        # Compute the derivatives of the contravariant fluxes using the derivative matrix
        dF_tilde_dxi  = np.einsum('il,lje->ije', D, F_tilde)
        dG_tilde_deta = np.einsum('jl,ile->ije', D, G_tilde)

        # Combine the derivatives to form the divergence term. This will be saved as the residual
        # in the element object, then overwritten later after the surface coupling term is added.
        e.residual = dF_tilde_dxi + dG_tilde_deta  # shape (P+1, P+1, num_eq)
    return

def _compute_divergence_3d(mesh : mesh_class.Mesh, case_cfg : CaseCfg) -> None:
    """
    Computes the 3D strong-form DGSEM divergence (VOLUME TERM ONLY).
    """

    # Extract the derivative matrix from the mesh object
    D = mesh.deriv_mat  # shape (P+1, P+1)

    # Loop through all elements
    for e in mesh.elements:
        # Extract the solution and contravariant basis vectors
        q   = e.solution                       # shape (P+1, P+1, P+1, num_eq)
        J   = e.jacobian_det                   # shape (P+1, P+1, P+1)
        Ja1 = J[..., None] * e.contravar_xi    # shape (P+1, P+1, P+1, 3)
        Ja2 = J[..., None] * e.contravar_eta   # shape (P+1, P+1, P+1, 3)
        Ja3 = J[..., None] * e.contravar_zeta  # shape (P+1, P+1, P+1, 3)

        # Compute the physical volume fluxes for the current solution
        F_vec = fluxes.compute_volume_flux(q, case_cfg, e.grad_q)  # shape (P+1, P+1, P+1, num_eq, 3)

        # Project the physical fluxes onto the contravariant basis vectors to get the 
        # contravariant flux terms.
        F_tilde = np.einsum('ijked,ijkd->ijke', F_vec, Ja1)
        G_tilde = np.einsum('ijked,ijkd->ijke', F_vec, Ja2)
        H_tilde = np.einsum('ijked,ijkd->ijke', F_vec, Ja3)

        # Compute the derivatives of the contravariant fluxes using the derivative matrix
        dF_tilde_dxi   = np.einsum('il,ljke->ijke', D, F_tilde)
        dG_tilde_deta  = np.einsum('jl,ilke->ijke', D, G_tilde)
        dH_tilde_dzeta = np.einsum('kl,ijle->ijke', D, H_tilde)

        # Combine the derivatives to form the divergence term. This will be saved as the residual
        # in the element object, then overwritten later after the surface coupling term is added.
        e.residual = dF_tilde_dxi + dG_tilde_deta + dH_tilde_dzeta # shape (P+1, P+1, P+1, num_eq)
    return

def _compute_surface_1d(mesh, case_cfg: CaseCfg, t: float = 0.0) -> None:
    raise NotImplementedError(f"Boundary flux computations are not implemented in 1D yet!")
    return

def _compute_surface_2d(mesh, case_cfg: CaseCfg, t: float = 0.0) -> None:
    '''
    Add the 2D surface (interface) correction to each element's residual, which already holds the
    volume divergence. Loops elements x faces; each interior interface flux is computed twice
    (once from each side) but the numerical flux is conservative, so the two agree up to the
    normal sign.
    '''
    I_min, I_max, w = mesh.face_interp_min, mesh.face_interp_max, mesh.quad_weights
    is_nse = case_cfg.physics.model == 'navier-stokes'      # only NSE needs face gradients

    for e in mesh.elements:
        for face_idx0, mort in enumerate(e.connected_mortars):
            face_id = face_idx0 + 1                            # 1-based face id

            # Interior trace: prolong THIS element's solution to the face
            q_minus    = _prolong_to_face_2d(e.solution, face_id, I_min, I_max)   # (P+1, num_eq)
            grad_minus = _prolong_grad_to_face_2d(e.grad_q, face_id, I_min, I_max) if is_nse else None

            left, right = mort.connected_elements

            # Outward normal + surface scale for THIS element, in THIS element's face-node order.
            # Computed from e's OWN face metric (J a^i) so they line up node-for-node with q_minus /
            # grad_minus (also from e). This is the general-mesh-safe choice: the stored
            # mort.normal_vector is in the OWNER's (=left) node order, which is mis-ordered for the
            # non-owner element on a FLIPPED (orientation-reversed) mortar. For axis-aligned meshes the
            # two orders coincide, so this reproduces the old result exactly.
            Ja      = e.jacobian_det[..., None] * (e.contravar_xi if _FACE_2D[face_id]['axis'] == 0
                                                   else e.contravar_eta)                        # (P+1,P+1,2)
            Ja_face = _prolong_metric_to_face_2d(Ja, face_id, I_min, I_max)                     # (P+1, 2)
            S       = np.linalg.norm(Ja_face, axis=-1)                                          # (P+1,)
            normal  = e.face_sign_map[face_id - 1] * Ja_face / S[:, None]                       # (P+1, 2) outward

            # Get or compute the exterior trace: q_plus. `visc_override` stays None everywhere except
            # at a WALL, where the BR1 central average of the viscous flux is not what we want: the
            # wall's q_plus is a reflection built for the Riemann solver, not a physical neighbor, so
            # averaging its viscous flux in would be meaningless. Walls hand the numerical flux an
            # explicit viscous flux instead (built from the wall state + the interior gradient).
            visc_override     = None
            inviscid_override = None      # set only by a characteristic far-field BC

            if None in mort.connected_elements:
                bc = mort.boundary_condition
                if bc.type == 'periodic':
                    # pull q_plus from the partner element across the domain (like an interior neighbor)
                    partner, pface = mort.periodic_partner_element, mort.periodic_partner_face
                    q_plus         = _prolong_to_face_2d(partner.solution, pface, I_min, I_max)
                    grad_plus      = _prolong_grad_to_face_2d(partner.grad_q, pface, I_min, I_max) if is_nse else None
                else:
                    # boundary mortar. Both prescribed and characteristic BCs read the face-node
                    # coordinates (to evaluate a state function / far-field reference), so prolong them
                    # the same way as the solution.
                    face_coords = (_prolong_metric_to_face_2d(e.quad_node_coords, face_id, I_min, I_max)
                                   if (is_prescribed(bc) or is_characteristic(bc)) else None)

                    if is_characteristic(bc):
                        # NON-REFLECTING far-field: the inviscid flux is F(q_bc).n for the characteristic
                        # ghost (outgoing waves from the interior, incoming from the far-field reference),
                        # imposed directly. The BC is viscously TRANSPARENT: extrapolate the state and the
                        # gradient (q_plus = q_minus, grad_plus = grad_minus) so the viscous flux is just
                        # the interior one -- NO Dirichlet target, NO penalty.
                        inviscid_override = characteristic_inviscid_normal_flux(bc, q_minus, normal,
                                                                               face_coords, case_cfg, t)
                        q_plus    = q_minus
                        grad_plus = grad_minus if is_nse else None
                    else:
                        # For a wall this is the impermeability reflection; for a prescribed BC it is the
                        # imposed state itself.
                        q_plus    = exterior_state(mort, q_minus, case_cfg, t, normal, face_coords)
                        grad_plus = grad_minus if is_nse else None

                        if is_nse and is_wall(bc):
                            # The real wall condition: viscous flux from the DIRICHLET wall state and the
                            # INTERIOR gradient, with the wall-normal heat flux zeroed out if adiabatic,
                            # PLUS the interior penalty that makes the weak Dirichlet imposition stable
                            # (e.h_min sets the penalty's length scale).
                            visc_override = wall_viscous_normal_flux(bc, q_minus, grad_minus, normal,
                                                                     case_cfg, e.h_min)
                        elif is_nse and is_prescribed(bc):
                            # Same story for a prescribed Dirichlet BC: the viscous flux is built from the
                            # imposed state + interior gradient, plus the stabilizing interior penalty.
                            visc_override = prescribed_viscous_normal_flux(bc, q_minus, grad_minus, normal,
                                                                           case_cfg, e.h_min, face_coords, t)
            else:
                # interior mortar, neighbor's trace at the SHARED face
                neighbor      = right if (e is left) else left
                nbr_face_id   = neighbor.connected_mortars.index(mort) + 1
                q_plus        = _prolong_to_face_2d(neighbor.solution, nbr_face_id, I_min, I_max)
                grad_plus     = _prolong_grad_to_face_2d(neighbor.grad_q, nbr_face_id, I_min, I_max) if is_nse else None

                # FACE-NODE FLIP: on an orientation-reversed mortar the two elements traverse the shared
                # face in OPPOSITE node order, so the neighbor's trace is reversed relative to q_minus.
                # Flip it (and grad_plus) back so it lines up node-for-node. Axis-aligned meshes have no
                # reversed mortars, so this is a no-op there; a general/curved mesh (e.g. the cylinder)
                # does. The normal is already in e's order (computed from e's metric above), so it needs
                # no flip.
                if mort.orientation_reversed:
                    q_plus = q_plus[::-1]
                    if grad_plus is not None:
                        grad_plus = grad_plus[::-1]

            # Compute the strong-form correction with the numerical and interior normal flux (physical)
            f_star     = fluxes.compute_numerical_flux(q_minus, q_plus, normal, case_cfg,
                                                       grad_minus, grad_plus, visc_override,
                                                       inviscid_override)   # (P+1, num_eq)
            f_interior = np.einsum('med,md->me',
                                   fluxes.compute_volume_flux(q_minus, case_cfg, grad_minus), normal)   # (P+1, num_eq)

            # S (surface scale = |J a^i| at the face) was computed with the normal above.
            corr = S[:, None] * (f_star - f_interior)          # (P+1, num_eq) contravariant correction

            # Lift the correction into the element's residual (strong form: ADD) ---
            e.residual += _lift_face_to_volume_2d(corr, face_id, I_min, I_max, w)

def _compute_surface_3d(mesh, case_cfg: CaseCfg, t: float = 0.0) -> None:
    raise NotImplementedError(f"Boundary flux computations are not implemented in 3D yet!")
    return

def _compute_gradient_volume_1d(mesh, case_cfg: CaseCfg) -> None:
    raise NotImplementedError("Gradient (BR1) volume term is not implemented in 1D yet!")

def _compute_gradient_volume_2d(mesh, case_cfg: CaseCfg) -> None:
    '''
    VOLUME term of g = grad(q): the strong-form contravariant gradient of the solution.
    Writes J*g (un-scaled) into e.grad_q; the surface term adds to it and the final 1/J scaling
    happens in the wrapper. Mirrors _compute_divergence_2d, but with q as the pseudo-flux and an
    extra trailing ndim axis on the output.
    '''
    D = mesh.deriv_mat  # (P+1, P+1)

    for e in mesh.elements:
        q   = e.solution                       # (P+1, P+1, num_eq)
        J   = e.jacobian_det                   # (P+1, P+1)
        Ja1 = J[..., None] * e.contravar_xi    # (P+1, P+1, 2)   (J a^1)
        Ja2 = J[..., None] * e.contravar_eta   # (P+1, P+1, 2)   (J a^2)

        # Pseudo-fluxes: q_e * (J a^i)_d  ->  (P+1, P+1, num_eq, ndim)
        Phi_xi  = np.einsum('ije,ijd->ijed', q, Ja1)
        Phi_eta = np.einsum('ije,ijd->ijed', q, Ja2)

        # Collocation derivatives along each reference direction (D contracts the free axis)
        dPhi_xi_dxi  = np.einsum('il,ljed->ijed', D, Phi_xi)
        dPhi_eta_deta = np.einsum('jl,iled->ijed', D, Phi_eta)

        # J * grad(q), before the surface correction and the 1/J scaling
        e.grad_q = dPhi_xi_dxi + dPhi_eta_deta   # (P+1, P+1, num_eq, ndim)

def _compute_gradient_volume_3d(mesh, case_cfg: CaseCfg) -> None:
    raise NotImplementedError("Gradient (BR1) volume term is not implemented in 3D yet!")

def _compute_gradient_surface_1d(mesh, case_cfg: CaseCfg, t: float = 0.0) -> None:
    raise NotImplementedError("Gradient (BR1) surface term is not implemented in 1D yet!")

def _compute_gradient_surface_2d(mesh, case_cfg: CaseCfg, t: float = 0.0) -> None:
    '''
    SURFACE term of g = grad(q): replace the interior boundary trace with the CENTRAL numerical
    trace q* = 1/2 (q- + q+) and lift (q* - q-) * (metric normal) into e.grad_q. This is
    _compute_surface_2d with the numerical-flux jump (f* - f_int) swapped for the central-trace
    jump (q* - q-), no Riemann solver, and a per-direction (ndim) correction.
    '''
    I_min, I_max, w = mesh.face_interp_min, mesh.face_interp_max, mesh.quad_weights

    for e in mesh.elements:
        for face_idx0, mort in enumerate(e.connected_mortars):
            face_id = face_idx0 + 1

            # Interior trace of THIS element at the face
            q_minus = _prolong_to_face_2d(e.solution, face_id, I_min, I_max)   # (P+1, num_eq)

            left, right = mort.connected_elements

            # Outward metric normal (J a^i) and unit normal for THIS element, in e's face-node order
            # (from e's own metric -- general-mesh-safe, see _compute_surface_2d for the reasoning).
            Ja            = e.jacobian_det[..., None] * (e.contravar_xi if _FACE_2D[face_id]['axis'] == 0
                                                        else e.contravar_eta)                   # (P+1,P+1,2)
            Ja_face       = _prolong_metric_to_face_2d(Ja, face_id, I_min, I_max)               # (P+1, 2)
            metric_normal = e.face_sign_map[face_id - 1] * Ja_face                              # (P+1, 2) = S*n
            normal        = metric_normal / np.linalg.norm(Ja_face, axis=-1)[:, None]           # (P+1, 2) unit

            # Exterior trace q_plus -- IDENTICAL selection logic to _compute_surface_2d
            if None in mort.connected_elements:
                if mort.boundary_condition.type == 'periodic':
                    q_plus = _prolong_to_face_2d(mort.periodic_partner_element.solution,
                                                 mort.periodic_partner_face, I_min, I_max)
                else:
                    # BR1 boundary trace. At a WALL this is the mirror of the interior about the
                    # Dirichlet wall state (q+ = 2 q_w - q-), chosen precisely so the central trace
                    # below lands on q_w EXACTLY -- that is what makes no-slip / isothermal a hard
                    # condition on grad(q) rather than a half-strength one. Every other BC just
                    # reuses its inviscid ghost. A prescribed BC additionally needs the face-node
                    # coordinates so it can evaluate its state function there.
                    face_coords = (_prolong_metric_to_face_2d(e.quad_node_coords, face_id, I_min, I_max)
                                   if is_prescribed(mort.boundary_condition) else None)
                    q_plus = gradient_exterior_state(mort, q_minus, case_cfg, t, normal, face_coords)
            else:
                neighbor    = right if (e is left) else left
                nbr_face_id = neighbor.connected_mortars.index(mort) + 1
                q_plus      = _prolong_to_face_2d(neighbor.solution, nbr_face_id, I_min, I_max)
                # Face-node flip on an orientation-reversed mortar (see _compute_surface_2d).
                if mort.orientation_reversed:
                    q_plus = q_plus[::-1]

            # Central numerical trace and its jump from the interior value
            q_star = 0.5 * (q_minus + q_plus)             # (P+1, num_eq)
            dq     = q_star - q_minus                     # = 1/2 (q_plus - q_minus)

            # metric_normal (= S * n = the outward (J a^i) vector) was computed with the normal above.
            # Per-direction correction: (q* - q-)_e * (metric normal)_d  ->  (P+1, num_eq, ndim)
            corr = dq[..., None] * metric_normal[:, None, :]

            # Lift into the volume (strong form: ADD), same boundary basis / weight as the residual
            e.grad_q += _lift_grad_face_to_volume_2d(corr, face_id, I_min, I_max, w)

def _compute_gradient_surface_3d(mesh, case_cfg: CaseCfg, t: float = 0.0) -> None:
    raise NotImplementedError("Gradient (BR1) surface term is not implemented in 3D yet!")

#######################################################################################
##### PRIMARY FUNCTIONS ###############################################################
#######################################################################################

def _compute_gradient_volume(mesh, case_cfg: CaseCfg) -> None:
    """
    Dimension-dispatch wrapper for the BR1 gradient VOLUME term. Writes J * grad(q) into e.grad_q
    for every element (the surface correction and 1/J scaling are applied afterward by
    _compute_gradient). Mirrors _compute_divergence.
    """

    # Create a dispatch dictionary to call the appropriate gradient-volume function by dimensionality.
    GRADIENT_VOLUME_DISPATCH = {
        1: _compute_gradient_volume_1d,
        2: _compute_gradient_volume_2d,
        3: _compute_gradient_volume_3d,
    }

    # Call the appropriate gradient-volume function based on the mesh dimensionality
    if mesh.dim not in GRADIENT_VOLUME_DISPATCH:
        raise NotImplementedError(f"Gradient volume term not implemented for dimensionality: {mesh.dim}")
    GRADIENT_VOLUME_DISPATCH[mesh.dim](mesh, case_cfg)

def _compute_gradient_surface(mesh, case_cfg: CaseCfg, t: float = 0.0) -> None:
    """
    Dimension-dispatch wrapper for the BR1 gradient SURFACE term. Adds the central-trace interface
    correction to e.grad_q for every element. Mirrors _compute_surface.
    """

    # Create a dispatch dictionary to call the appropriate gradient-surface function by dimensionality.
    GRADIENT_SURFACE_DISPATCH = {
        1: _compute_gradient_surface_1d,
        2: _compute_gradient_surface_2d,
        3: _compute_gradient_surface_3d,
    }

    # Call the appropriate gradient-surface function based on the mesh dimensionality
    if mesh.dim not in GRADIENT_SURFACE_DISPATCH:
        raise NotImplementedError(f"Gradient surface term not implemented for dimensionality: {mesh.dim}")
    GRADIENT_SURFACE_DISPATCH[mesh.dim](mesh, case_cfg, t)

def _compute_gradient(mesh, case_cfg: CaseCfg, t: float = 0.0) -> None:
    '''
    BR1 gradient wrapper. Writes the physical gradient g = grad(q) into e.grad_q for every element,
    shape (P+1, P+1, num_eq, ndim). Structure mirrors compute_residual:
        1. volume term  -> e.grad_q = J * (contravariant gradient of q)
        2. surface term -> add the central-trace interface correction
        3. scale by 1/J -> e.grad_q = grad(q)
    Both terms are dimension-dispatched like the residual.
    '''
    _compute_gradient_volume (mesh, case_cfg   )
    _compute_gradient_surface(mesh, case_cfg, t)
    for e in mesh.elements:
        e.grad_q *= e.jacobian_det_inv[..., None, None]   # two extra axes: num_eq and ndim


def _compute_divergence(mesh : mesh_class.Mesh, case_cfg : CaseCfg) -> None:
    """
    Dimension-dispatch wrapper. Writes `e.residual` (shape (P+1,)*dim + (num_eq,)) for every
    element in `mesh`.
    Note: This function computes the divergence of the contravariant VOLUME fluxes ONLY. The
    surface coupling terms are used to update e.residual after this function is called.

    Inputs:
    - mesh : constructed Mesh (elements, metrics, deriv_mat, face_interp_min/max all populated).
    - case_cfg  : validated case configuration.
    """

    # Create a dispatch dictionary to call appropriate divergence function based on dimensionality.
    DIVERGENCE_DISPATCH = {
        1: _compute_divergence_1d,
        2: _compute_divergence_2d,
        3: _compute_divergence_3d,
    }

    # Call the appropriate divergence function based on the mesh dimensionality
    if mesh.dim not in DIVERGENCE_DISPATCH:
        raise NotImplementedError(f"Divergence not implemented for dimensionality: {mesh.dim}")
    DIVERGENCE_DISPATCH[mesh.dim](mesh, case_cfg)

def _compute_surface(mesh, case_cfg: CaseCfg, t: float = 0.0) -> None:
    """
    Dimension-dispatch wrapper. Corrects `e.residual` with the surface terms for every
    element in `mesh`. 

    Inputs:
    - mesh : constructed Mesh (elements, metrics, deriv_mat, face_interp_min/max all populated).
    - case_cfg  : validated case configuration.
    """

    # Create a dispatch dictionary to call appropriate divergence function based on dimensionality.
    SURFACE_DISPATCH = {
        1: _compute_surface_1d,
        2: _compute_surface_2d,
        3: _compute_surface_3d,
    }

    # Call the appropriate surface computation function based on the mesh dimensionality
    if mesh.dim not in SURFACE_DISPATCH:
        raise NotImplementedError(f"Surface correction not implemented for dimensionality: {mesh.dim}")
    SURFACE_DISPATCH[mesh.dim](mesh, case_cfg, t)

def compute_residual(mesh : mesh_class.Mesh, case_cfg : CaseCfg, t : float=0.0) -> None:
    """
    Dimension-dispatch wrapper. Writes `e.residual` (shape (P+1,)*dim + (num_eq,)) for every
    element in `mesh`. First, the divergence of the contravariant VOLUME fluxes is computed 
    and stored in e.residual. Then, the surface coupling terms are added to e.residual. 
    Finally, the residual is scaled by -1/J to satisfy the semi-discrete update equation:

        q_t = -(1/J) [ d(F~)/dxi + d(G~)/deta + d(H~)/dzeta ]  +  (surface coupling term)

    Inputs:
    - mesh : constructed Mesh
    - case_cfg  : validated case configuration.
    - t         : time (used for unsteady boundary conditions, if applicable)
    """

    # Only compute the solution gradient if required (aka if using navier-stokes model)
    if case_cfg.physics.model == 'navier-stokes':
        _compute_gradient(mesh, case_cfg, t)

    # Compute the divergence of the contravariant VOLUME fluxes and store in e.residual
    _compute_divergence(mesh, case_cfg)

    # Add the surface coupling terms to e.residual
    _compute_surface(mesh, case_cfg, t)

    # Scale the residual by -1/J to satisfy the semi-discrete update equation
    for e in mesh.elements:
        e.residual *= -e.jacobian_det_inv[..., None]

    # Finally, add any source term S(q, x, t):  q_t = -(1/J)[div F~] + S. This comes AFTER the -1/J
    # scaling because S enters the PDE as a bare additive term -- it is the flux divergence, not the
    # source, that carries the metric Jacobian. A no-op unless the case declares a source term.
    add_source_terms(mesh, case_cfg, t)
    