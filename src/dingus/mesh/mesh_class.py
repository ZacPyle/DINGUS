# src/dingus/mesh/mesh_class.py
import dingus.coreNumerics.interpolation as interpolation
import dingus.coreNumerics.mapping as mapping
import dingus.coreNumerics.quadrature as quadrature
import dingus.mesh.hohqmesh_handler as hohqmesh
import dingus.mesh.gmsh_handler as gmsh
import numpy as np
from dingus.config import CaseCfg
from dingus.mesh import element_class
from dingus.mesh import mortar_class
from pathlib import Path
from pprint import pprint
from scipy.spatial import Delaunay
from typing import Any, Dict, List, Optional, Union


class Mesh:
    """
    The Mesh class represents the computational grid (or 'mesh') used in simulations. It manages the
    elements, nodes, and connectivity (mortar) information of the mesh; the idea is that all relevant
    information is contained in a single, easy-to-access object. The elements and mortars are collections
    of Element and Mortar classed objects.
    """

    def __init__(self) -> None:
        # Instantiate empty attributes
        self.boundary_mortars         : List['mortar_class.SpectralMortar'  ] = []
        self.deriv_mat                : np.ndarray = np.array([])
        self.dim                      : int = 2
        self.el_poly_order            : int = 0
        self.elements                 : List['element_class.SpectralElement'] = []
        self.face_interp_min          : np.ndarray = np.array([])
        self.face_interp_max          : np.ndarray = np.array([])
        self.mesh_format              : str = ""
        self.mortars                  : List['mortar_class.SpectralMortar'  ] = []
        self.num_elements             : int = 0
        self.num_mortars              : int = 0
        self.num_nodes                : int = 0
        self.quad_nodes               : np.ndarray = np.array([])
        self.quad_weights             : np.ndarray = np.array([])
        self.quad_type                : str = ""
        self.raw_data                 : Dict[str, Any] = {}

    def _link_periodic_mortars(self) -> None:
        '''
        Pairs each periodic boundary mortar with its partner on the opposite boundary, so the DG
        surface term can pull q_plus from the element ACROSS the domain (treating a periodic face
        like an interior interface). Matching is by TRANSVERSE coordinate: a Left mortar (x=0) and
        its Right partner (x=1) differ along the periodic axis but share the same transverse position.

        Requires each boundary mortar to already have `.boundary_condition` assigned. Stores on each
        periodic mortar:
            mort.periodic_partner_element : owner element of the partner mortar (across the domain)
            mort.periodic_partner_face    : that element's 1-based local face id for the shared face
        '''

        # Only periodic mortars need pairing.
        periodic = [m for m in self.boundary_mortars if m.boundary_condition.type == 'periodic']

        for m in periodic:
            # Candidate partners: periodic mortars on the boundary m names as its partner.
            partner_name = m.boundary_condition.partner
            candidates   = [p for p in periodic if p.boundary_condition_name == partner_name]
            if not candidates:
                raise ValueError(
                    f"Periodic boundary '{m.boundary_condition_name}' names partner '{partner_name}', "
                    f"but no periodic mortars with that name were found."
                )

            # Face centroids (mean of the mortar's endpoint node coords) in physical space.
            m_c    = m.node_coords[:, :self.dim].mean(axis=0)                                   # (dim,)
            cand_c = np.array([p.node_coords[:, :self.dim].mean(axis=0) for p in candidates])   # (n, dim)

            # PERIODIC axis = the direction the two boundaries are separated along (largest offset).
            # Everything else is TRANSVERSE and should match between a mortar and its partner.
            periodic_axis = int(np.argmax(np.abs(cand_c.mean(axis=0) - m_c)))

            # Closest candidate in the transverse coords only (zero out the periodic axis).
            d = cand_c - m_c
            d[:, periodic_axis] = 0.0
            partner = candidates[int(np.argmin(np.linalg.norm(d, axis=1)))]

            # Store the partner's OWNER element + that element's local face id, so the surface term
            # can prolong the partner element's trace as q_plus.
            owner = partner.connected_elements[0] or partner.connected_elements[1]
            m.periodic_partner_element = owner
            m.periodic_partner_face    = owner.connected_mortars.index(partner) + 1

    def read_mesh(self, fileName: Union[str, Path]) -> None:
        """
        Reads in a mesh file. To do this, we identify:
        - the file format (e.g., HOHQMesh-ISM-V2, ABAQUS, Gmsh, etc.)
        - the dimensionality of the mesh (2D or 3D)
        The basic information attributes (num_elements, num_mortars, ....) are populated using
        the data read from the mesh file. Then, the raw_data attribute is populated for use with
        element and mortar constructors later.

        Currently supported file types and formats are:
        - HOHQMesh ISM-V2 (2D)

        File types and formats that are not yet supported but we plan to implement in the future:
        - HOHQMesh ISM-V2 (3D)
        - HOHQMesh ISM (2D and 3D)
        - HOHQMesh ISM-MM (2D and 3D)
        - ABAQUS (2D and 3D)
        - Gmsh (2D and 3D)

        Args:
            fileName (Union[str, Path]): Path to the mesh file, including the file name.
                Example: 'mesh/Square.mesh'
        """
        # Ensure fileName is a Path object
        fileName = Path(fileName)

        # Grab the mesh file extension to identify the mesh file type
        extension = fileName.suffix.lstrip(".").lower()

        # Call appropriate mesh file reader based on the file extension
        match extension:
            case "mesh":
                self.raw_data    = hohqmesh.read_hohqmesh(fileName)
                self.mesh_format = 'ism-v2'

            case "msh":
                raise NotImplementedError("Gmsh mesh file reading still in the works!")
                # self.raw_data    = gmsh.read_gmsh(fileName)
                # self.mesh_format = 'gmsh'

            case "ini":
                raise NotImplementedError("ABAQUS mesh file reading still in the works!")
                # self.mesh_format = 'abaqus'

            case _:
                raise NotImplementedError(f"Mesh file type not supported: {extension}")

        # Fill in the Mesh object attributes using the mesh data dictionary
        self.num_nodes      = self.raw_data["num_nodes"]
        self.num_elements   = self.raw_data["num_elements"]
        self.num_mortars    = self.raw_data["num_mortars"]
        self.bc_poly_order  = self.raw_data["poly_order"]
        self.el_poly_order  = self.raw_data["poly_order"]

        # ADD CONTROL FILE READING TO GRAB DIMENSIONALITY AND QUADRATURE TYPE IN THE FUTURE, 
        # FOE NOW WE JUST HARDCODE FOR TESTING
        #self.quad_type = "LG"

    def construct_elements(self) -> None:
        """
        Constructs Element objects for each element in the mesh. This uses the attribute self.raw_data to
        determine the element information. This MUST be called after self.read_mesh().
        """

        # Make sure we have read in a mesh file first
        if not self.raw_data:
            raise RuntimeError("Must call the read_mesh() method before the construct_elements() method!")

        # Extract basic data from the mesh
        node_ids_per_element = self.raw_data["elementNodeIDs"]
        node_coordinates     = self.raw_data["nodes"]       

        # Create collection of element objects for each element described in the list of node IDs using
        # list comprehension
        for el_id in range(self.num_elements):
            # Create the element object
            self.elements.append(element_class.SpectralElement(
                element_id  = el_id+1,  # Adding 1 to convert from 0-based to 1-based indexing
                node_ids    = node_ids_per_element[el_id, :],
                node_coords = node_coordinates[node_ids_per_element[el_id, :] - 1, :],  # node IDs are 1-based; must convert to 0-based for indexing
                poly_order  = self.el_poly_order,
                el_type     = "line" if self.dim == 1 else "quad" if self.dim == 2 else "hex" if self.dim == 3 else "unknown"
            ))

            # Save the boundary condition names to the element object
            self.elements[el_id].boundary_condition_names = self.raw_data["elementBCNames"][el_id,:]
            self.elements[el_id].mortar_curvature         = self.raw_data["mortarCurvatureType"][el_id,:]

            # Create the face sign map for this element. This is an array containing the sign of the outward-facing normal
            # vector at each face of the element. 
            match self.dim:
                case 1:
                    self.elements[el_id].face_sign_map = np.array([-1, 1]) # Left face, right face

                case 2:
                    self.elements[el_id].face_sign_map = np.array([-1, 1, 1, -1]) # Bottom face, right face, top face, left face

                case 3:
                    self.elements[el_id].face_sign_map = np.array([-1, 1, -1, 1, 1, -1]) 

    def construct_mortars(self) -> None:
        """
        Constructs Mortar objects for each mortar in the mesh. This uses the attribute self.raw_data to
        determine the element information. This MUST be called after self.read_mesh().
        """

        # Extract basic data from the mesh
        mortar_data      = self.raw_data["mortars"]
        node_coordinates = self.raw_data["nodes"] 

        for mortar_id in range(self.num_mortars):
            self.mortars.append(mortar_class.SpectralMortar(
                mortar_id   = mortar_id+1,  # Adding 1 to convert from 0-based to 1-based indexing
                node_ids    = mortar_data[mortar_id, :2],
                node_coords = node_coordinates[mortar_data[mortar_id, :2] -1, :],  # node IDs are 1-based; must convert to 0-based for indexing
                poly_order  = self.el_poly_order,
                mortar_type = "edge" if self.dim == 2 else "face" if self.dim == 3 else "unknown"
            ))

    def link_elements_and_mortars(self) -> None:
        """
        Links Element and Mortar objects together (i.e., what mortars are associated with each element). This 
        MUST be called after self.construct_elements() and self.construct_mortars().
        """

        # Loop through the mortars and link them to appropriate elements. Although the list of mortars is larger than 
        # the list of elements, this 'shoud' be faster because if we loop through the elements we must search through all 
        # mortars and find the appropriate nodes that bound the mortar. Unfortunately, that may need to be implemented
        # in the future for other mesh file formats (e.g., Gmsh) that do not provide explicit connectivity information.

        for mort in self.mortars:
            # Assign connectivity information to variables for readability
            el_id_left    = int(self.raw_data["mortars"][mort.id_global-1, 2])
            el_face_left  = int(self.raw_data["mortars"][mort.id_global-1, 4])
            el_id_right   = int(self.raw_data["mortars"][mort.id_global-1, 3])
            el_face_right = int(self.raw_data["mortars"][mort.id_global-1, 5])

            # Link the elements to the mortar. Remember to convert from 1-based to 0-based indexing and to handle
            # boundary mortars (where one of the element IDs is zero)
            # TODO: el_face_right can be negative for non-orthogonal meshes, indicating that the
            # node ordering along the shared side runs in the opposite direction relative to the
            # left element. For now we use abs() to get the correct face index, but when implementing
            # curved boundaries the sign must be used to reverse the spline knot ordering for this
            # side to ensure consistent interpolation between elements. See ISM-V2 docs for details.
            mort.connected_elements = [self.elements[el_id_left-1]  if el_id_left  != 0 else None,
                                       self.elements[el_id_right-1] if el_id_right != 0 else None]
            
            # Link the mortar to the elements
            if el_id_left != 0:
                self.elements[el_id_left-1 ].connected_mortars[abs(el_face_left)-1 ] = mort
            if el_id_right != 0:
                self.elements[el_id_right-1].connected_mortars[abs(el_face_right)-1] = mort

            # If this is a boundary mortar, grab the boundary condition name from the element. Remember, if the
            # RIGHT element id = 0, that means the LEFT element is physical and contains the BC information; if
            # the LEFT element id = 0, that means the RIGHT element is physical contains the BC information.
            if el_id_left == 0:
                mort.boundary_condition_name = self.elements[el_id_right-1].boundary_condition_names[abs(el_face_right)-1]
            if el_id_right == 0:
                mort.boundary_condition_name = self.elements[el_id_left-1 ].boundary_condition_names[abs(el_face_left)-1 ]

            # Create the face sign map for this mortar. This is an array containing the sign of the outward-facing normal
            # vector at each face of the element. 
            mort.compute_mortar_face_sign_map(self.dim, el_face_left, el_face_right)          

    def construct_mesh(self, case_config : CaseCfg) -> None:
        """
        Constructs the mesh by:
        - creating each element object
        - creating each mortar object
        - linking mortars to proper elements
        - computing the quadrature nodes and weights to span the [-1, 1]^d reference domain (assuming uniform polynomial order)
        - computing the element metrics for every element
        - building a delaunnay triangulation for IO interpolation / in-run plotting
        - computing the derivative matrix and face interpolation matrices (used for all elements)
        - assigning a BCCfg object to each boundary mortar
        """

        # TODO: Add the CaseCfg as an argument to this, then populate the mesh.quad_nodes type. This will allow
        # all mesh functions to be handled here, instead of building the elements and mortars, then separately
        # Saving the quadrature type, computing the quadrature nodes, then element metrics, then building
        # the triangulation.

        self.quad_type     = str(case_config.mesh.quad_type)
        self.el_poly_order = case_config.mesh.poly_deg

        # Construct elements and mortars, then link them
        self.construct_elements()
        self.construct_mortars()
        self.link_elements_and_mortars()

        # Construct quadrature nodes within each element and compute element metrics
        quadrature.Compute_Quadrature_Nodes_And_Weights(self)
        self.compute_element_metrics()

        # Compute Delaunay triangulation for uniform plotting if desired
        self.build_delaunay_tri()

        # Compute the polynomial derivative matrix. Recall the derivative matrix is the same
        # for all elements, so we compute it once and save it to the mesh object rather than
        # to individual element objects.
        self.deriv_mat = interpolation.Polynomial_Derivative_Matrix(self.quad_nodes)

        # Compute the face interpolation matrices for each element if using LG quadrature.
        # Again, these are the same for all elements so they are computes once and stored
        # in the mesh object rather than in each element object.
        self.face_interp_min = interpolation.Polynomial_Interpolation_Matrix(self.quad_nodes, np.array([-1]))
        self.face_interp_max = interpolation.Polynomial_Interpolation_Matrix(self.quad_nodes, np.array([ 1]))
        
        # Loop through the mortars and apply boundary conditions to the boundary mortars.
        self.apply_boundary_conditions(case_config)

        # self.apply_mortar_curvature()

    def compute_element_metrics(self) -> None:
        """
        Computes the metric terms (e.g., the co/contra-variant vectors, Jacobian determinant, ...) within each element
        of a mesh. These are needed for mapping the Navier-Stokes equations from the physical domain to the 
        computational / reference domain. For details on the equation mapping and the derivation of the metric equations
        used in this function, see Chapter 6 of Kopriva.
        """

        # TODO: Compute element metrics:
        # self.compute_isoparametric_mapping()
        # - mapping derivatives    DONE
        # - covariant vectors      DONE
        # - contravariant vectors  DONE
        # - Jacobian determinant   DONE
        # - Inverse Jacobian determinant? DONE
        # - outward unit normal at element boundaries DONE

        """
        Currently refactoring so this is a master wrapper function that calls all appropriate metric funcitons, depending on
        dimensionality.
        """

        # Create a dispatch dictionary to call appropriate mapping funciton based on dimensionality.
        MAPPING_DISPATCH = {
            1: mapping._isop_map_1d,
            2: mapping._isop_map_2d,
            # 3: mapping._isop_map_3d,
        }
        if self.dim not in MAPPING_DISPATCH:
            raise NotImplementedError(f"Isoparametric mapping not implemented for dimensionality: {self.dim}")

        # Create a dispatch dictionary to call appropriate mapping function based on dimensionality.
        DERIVATIVES_DISPATCH = {
            1: mapping.compute_mapping_derivatives_1d,
            2: mapping.compute_mapping_derivatives_2d,
            3: mapping.compute_mapping_derivatives_3d,
        }
        if self.dim not in DERIVATIVES_DISPATCH:
            raise NotImplementedError(f"Spatial derivative mapping not implemented for dimensionality: {self.dim}")

        # Create a dispatch dictionary to call appropriate mapping function based on dimensionality.
        JACOBIAN_DISPATCH = {
            1: mapping.compute_jacobian_metrics_1d,
            2: mapping.compute_jacobian_metrics_2d,
            3: mapping.compute_jacobian_metrics_3d,
        }
        if self.dim not in JACOBIAN_DISPATCH:
            raise NotImplementedError(f'Jacobian metric computations not implemented for dimensionality: {self.dim}')

        # Create a dispatch dictionary to call appropriate mapping function based on dimensionality.
        VECTOR_DISPATCH = {
            1: mapping.compute_covariant_and_contravariant_vectors_1d,
            2: mapping.compute_covariant_and_contravariant_vectors_2d,
            3: mapping.compute_covariant_and_contravariant_vectors_3d,
        }
        if self.dim not in VECTOR_DISPATCH:
            raise NotImplementedError(f'Covariant and contravariant vector computations not implemented for dimensionality: {self.dim}')

        # Create a dispath dictionary to call appropriate normal computation function based on dimensionality.
        NORMALS_DISPATCH = {
            1: mapping.compute_normals_1d,
            2: mapping.compute_normals_2d,
            3: mapping.compute_normals_3d,
        }
        if self.dim not in NORMALS_DISPATCH:
            raise NotImplementedError(f'Normal vector computations not implemented for dimensionality: {self.dim}')

        # Ensure elements have been constructed
        if not self.elements:
            raise RuntimeError("Must call construct_mesh() before compute_element_metrics()!")
        
        # Call appropriate metric functions
        for e in self.elements:
            MAPPING_DISPATCH    [self.dim](self, e)

            # if not self.quad_nodes.size:
            #     raise RuntimeError("Must call Compute_Quadrature_Nodes_And_Weights() before compute_mapping_derivatives()!") 
            DERIVATIVES_DISPATCH[self.dim](self, e)

            # if not hasattr(e, 'dXdXi'):
            #     raise RuntimeError("Must call compute_mapping_derivatives() before compute_jacobian_metrics()!")
            JACOBIAN_DISPATCH   [self.dim](e)

            # if not hasattr(e, 'jacobian'):
            #     raise RuntimeError("Must call compute_jacobian_metrics() before compute_covariant_and_contravariant_vectors()!")
            VECTOR_DISPATCH     [self.dim](e)

            # Compute the minimum length scale based on contravariant vectors for time stepping
            match self.dim:
                case 1:
                    contravars = [e.contravar_xi]
                case 2:
                    contravars = [e.contravar_xi, e.contravar_eta]
                case 3:
                    contravars = [e.contravar_xi, e.contravar_eta, e.contravar_zeta]
                case _:
                    raise ValueError(f"Unsupported dimensionality detected during metric computation: {self.dim}.")
            e.h_min = 1.0 / np.max(sum(np.linalg.norm(cv, axis=-1) for cv in contravars))
            
        # Compute normals outside of previous loop because it requires looping through elements AND mortars.
        NORMALS_DISPATCH[self.dim](self)
    
    def apply_mortar_curvature(self) -> None:
        """
        Applies mortar curvature to the mortar objects based on the information in self.raw_data.
        This MUST be called after self.construct_elements() and self.construct_mortars().
        """

        raise NotImplementedError("Applying mortar curvature not implemented yet!")
    
    def apply_boundary_conditions(self, case_cfg : CaseCfg) -> None:
        """
        Applies boundary conditions to the mortars based on the information in self.raw_data.
        This MUST be called after self.construct_elements() and self.construct_mortars().
        """

        # Extract the BC dictionary from the case configuration object
        BCDict           = case_cfg.boundary_conditions
        for mortar in self.mortars:
            if None not in mortar.connected_elements:   # Interior mortars (No BC)
                continue
            if not mortar.boundary_condition_name in BCDict:
                raise ValueError(f"Boundary condition: '{mortar.boundary_condition_name}' found in {case_cfg.mesh.mesh_file} but not in the control file!")
            mortar.boundary_condition = BCDict[mortar.boundary_condition_name]
            self.boundary_mortars.append(mortar)

        # Link all periodic boundary conditions. If there are none, then this function call does nothing
        self._link_periodic_mortars()


    
    def build_delaunay_tri(self) -> None:
        """
        Creates a Delaunay triangulation based on the quadrature node coordinates. This is saved to 
        the mesh object for future interpolation (e.g. when / if writing output data on a uniform grid).
        This must be called AFTER compute_element_metrics() so the quad_node_coords field is actually
        populated.
        """

        # Make sure the quadrature nodes are already populated
        if not self.elements:
            raise RuntimeError("Must call 'construct_mesh()' BEFORE 'build_delaunay_tri()'!")
        if self.elements[0].quad_node_coords is None or np.asarray(self.elements[0].quad_node_coords).size == 0:
            raise RuntimeError("Must call 'compute_element_metrics() BEFORE 'buid_delaunay_tri()'!")
        
        # Collect all quadrature node coordinates from the elements and construct an array of 3 column vectors; 
        # one vector for x, y, and z coordinates (as appropriate).
        all_coords = np.vstack([
            e.quad_node_coords[:,:,:self.dim].reshape(-1,2)
            for e in self.elements
        ])

        # If Legendre-Gauss-Lobatto (LGL) quadrature is used instead of Legendre-Gauss (LG) then duplicate
        # quadrature nodes exist on element boundaries. De-dupe them to prevent doubled nodes (and doubled
        # solution values) at shared interfaces, which would corrupt post-processing derivatives.
        #
        # A shared interface node is computed independently by each adjacent element's isoparametric map,
        # so the two copies can differ at the ~1e-16 rounding level. An exact np.unique() therefore misses
        # them; quantize to a length-scale-relative tolerance first so physically-coincident nodes collapse.
        if self.quad_type == "LGL":
            scale         = np.linalg.norm(all_coords.max(axis=0) - all_coords.min(axis=0))
            tol           = 1e-8 * scale
            _, unique_idx = np.unique(np.round(all_coords / tol), axis=0, return_index=True)
            unique_idx    = np.sort(unique_idx)   # preserve original element-stacked node ordering
            self.delaunay_unique_idx = unique_idx
            unique_coords = all_coords[unique_idx]
        else:
            unique_coords  = all_coords

        # Create and save the Delaunay triangulation
        self.delaunay_coords     = unique_coords
        self.delaunay_tri        = Delaunay(unique_coords)
        