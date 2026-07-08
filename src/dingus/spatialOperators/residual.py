# src/dingus/spatialOperators/residual.py

import numpy as np
from dingus.boundaryConditions.boundaryConditions import exterior_state
from dingus.config import CaseCfg
from dingus.mesh import mesh_class
from dingus.physics import fluxes

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
        F_vec = fluxes.compute_volume_flux(q, case_cfg)  # shape (P+1, num_eq, 2)

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
        F_vec = fluxes.compute_volume_flux(q, case_cfg)  # shape (P+1, P+1, num_eq, 2)

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
        F_vec = fluxes.compute_volume_flux(q, case_cfg)  # shape (P+1, P+1, P+1, num_eq, 3)

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

    for e in mesh.elements:
        for face_idx0, mort in enumerate(e.connected_mortars):
            face_id = face_idx0 + 1                            # 1-based face id

            # Interior trace: prolong THIS element's solution to the face
            q_minus = _prolong_to_face_2d(e.solution, face_id, I_min, I_max)   # (P+1, num_eq)

            # Grab the outward normal for THIS element (stored normal is outward-from-left owner)
            left, right = mort.connected_elements
            owner  = left if left is not None else right       # matches _get_mortar_owner
            normal = mort.normal_vector if (e is owner) else -mort.normal_vector   # (P+1, 2)

            # Get or compute the exterior trace: q_plus
            if None in mort.connected_elements:
                # boundary mortar, compute the ghost state from the BC dispatch
                q_plus = exterior_state(mort, q_minus, case_cfg, t)
            else:
                # interior mortar, neighbor's trace at the SHARED face
                neighbor      = right if (e is left) else left
                nbr_face_id   = neighbor.connected_mortars.index(mort) + 1
                q_plus        = _prolong_to_face_2d(neighbor.solution, nbr_face_id, I_min, I_max)
                # NOTE: for this axis-aligned Square, the two elements order the shared face nodes
                # identically, so no flip is needed. A general/curved mesh may require reversing
                # q_plus (and the metric) to match node ordering -- assert/flag when you get there.

            # Compute the strong-form correction with the numerical and interior normal flux (physical)
            f_star     = fluxes.compute_numerical_flux(q_minus, q_plus, normal, case_cfg)   # (P+1, num_eq)
            f_interior = np.einsum('med,md->me',
                                   fluxes.compute_volume_flux(q_minus, case_cfg), normal)   # (P+1, num_eq)

            # surface scale S = |J a^i| at the face: prolong the metric vector and take its norm.
            Ja  = e.jacobian_det[..., None] * (e.contravar_xi if _FACE_2D[face_id]['axis'] == 0
                                               else e.contravar_eta)                              # (P+1,P+1,2)
            S   = np.linalg.norm(_prolong_metric_to_face_2d(Ja, face_id, I_min, I_max), axis=-1)  # (P+1,)

            corr = S[:, None] * (f_star - f_interior)          # (P+1, num_eq) contravariant correction

            # Lift the correction into the element's residual (strong form: ADD) ---
            e.residual += _lift_face_to_volume_2d(corr, face_id, I_min, I_max, w)

def _compute_surface_3d(mesh, case_cfg: CaseCfg, t: float = 0.0) -> None:
    raise NotImplementedError(f"Boundary flux computations are not implemented in 3D yet!")
    return

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

    # Compute the divergence of the contravariant VOLUME fluxes and store in e.residual
    _compute_divergence(mesh, case_cfg)

    # Add the surface coupling terms to e.residual
    _compute_surface(mesh, case_cfg, t)
    
    # Scale the residual by -1/J to satisfy the semi-discrete update equation
    for e in mesh.elements:
        e.residual *= -e.jacobian_det_inv[..., None] 
    