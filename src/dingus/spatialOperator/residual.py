# src/dingus/spatialOperator/residual.py

import numpy as np
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

        q_t = -(1/J) [ d/dxi(F~) + d/deta(G~) + d/dzeta(H~) ]  +  (surface coupling term, added later)

Dimensionality is handled the same way as mapping.py: a thin `compute_residual` wrapper
dispatches to a per-dimension worker.
'''

def _compute_divergence_1d(case_mesh : mesh_class.Mesh, case_cfg : CaseCfg) -> None:
    """
    Computes the 1D strong-form DGSEM divergence (VOLUME TERM ONLY).
    """

    # Extract the derivative matrix from the mesh object
    D = case_mesh.deriv_mat  # shape (P+1, P+1)

    # Loop through all elements
    for e in case_mesh.elements:
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

def _compute_divergence_2d(case_mesh : mesh_class.Mesh, case_cfg : CaseCfg) -> None:
    """
    Computes the 2D strong-form DGSEM divergence (VOLUME TERM ONLY).
    """

    # Extract the derivative matrix from the mesh object
    D = case_mesh.deriv_mat  # shape (P+1, P+1)

    # Loop through all elements
    for e in case_mesh.elements:
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

def _compute_divergence_3d(case_mesh : mesh_class.Mesh, case_cfg : CaseCfg) -> None:
    """
    Computes the 3D strong-form DGSEM divergence (VOLUME TERM ONLY).
    """

    # Extract the derivative matrix from the mesh object
    D = case_mesh.deriv_mat  # shape (P+1, P+1)

    # Loop through all elements
    for e in case_mesh.elements:
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

def _compute_divergence(case_mesh : mesh_class.Mesh, case_cfg : CaseCfg) -> None:
    """
    Dimension-dispatch wrapper. Writes `e.residual` (shape (P+1,)*dim + (num_eq,)) for every
    element in `case_mesh`. Mirrors the dispatch pattern in mesh_class.compute_element_metrics.
    Note: This function computes the divergence of the contravariant VOLUME fluxes ONLY. The
    surface coupling terms are used to update e.residual after this function is called.

    Inputs:
    - case_mesh : constructed Mesh (elements, metrics, deriv_mat, face_interp_min/max all populated).
    - case_cfg  : validated case configuration.
    """

    # Create a dispatch dictionary to call appropriate divergence function based on dimensionality.
    DIVERGENCE_DISPATCH = {
        1: _compute_divergence_1d,
        2: _compute_divergence_2d,
        3: _compute_divergence_3d,
    }

    # Call the appropriate divergence function based on the mesh dimensionality
    if case_mesh.dim not in DIVERGENCE_DISPATCH:
        raise NotImplementedError(f"Divergence not implemented for dimensionality: {case_mesh.dim}")
    DIVERGENCE_DISPATCH[case_mesh.dim](case_mesh, case_cfg)

def compute_residual(case_mesh : mesh_class.Mesh, case_cfg : CaseCfg) -> None:
    """
    Dimension-dispatch wrapper. Writes `e.residual` (shape (P+1,)*dim + (num_eq,)) for every
    element in `case_mesh`. First, the divergence of the contravariant VOLUME fluxes is computed 
    and stored in e.residual. Then, the surface coupling terms are added to e.residual. 
    Finally, the residual is scaled by -1/J to satisfy the semi-discrete update equation:

        q_t = -(1/J) [ d(F~)/dxi + d(G~)/deta + d(H~)/dzeta ]  +  (surface coupling term)

    Inputs:
    - case_mesh : constructed Mesh
    - case_cfg  : validated case configuration.
    """

    # Compute the divergence of the contravariant VOLUME fluxes and store in e.residual
    _compute_divergence(case_mesh, case_cfg)

    # Add the surface coupling terms to e.residual
    
    # Scale the residual by -1/J to satisfy the semi-discrete update equation
    for e in case_mesh.elements:
        e.residual *= -e.jacobian_det_inv[..., None] 
    