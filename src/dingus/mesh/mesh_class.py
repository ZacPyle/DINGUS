# src/dingus/mesh/mesh_class.py
import dingus.coreNumerics.mapping as mapping
import dingus.mesh.hohqmesh_handler as hohqmesh
import dingus.mesh.gmsh_handler as gmsh
from dingus.mesh import element_class
from dingus.mesh import mortar_class
import numpy as np
from pathlib import Path
from pprint import pprint
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
        self.dim           : int = 2
        self.el_poly_order : int = 0
        self.elements      : List['element_class.SpectralElement'] = []
        self.mesh_format   : str = ""
        self.mortars       : List['mortar_class.SpectralMortar'  ] = []
        self.num_elements  : int = 0
        self.num_mortars   : int = 0
        self.num_nodes     : int = 0
        self.quad_nodes    : np.ndarray = np.array([])
        self.quad_weights  : np.ndarray = np.array([])
        self.quad_type     : str = ""
        self.raw_data      : Dict[str, Any] = {}

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
            mort.connected_elements = [self.elements[el_id_left-1]  if el_id_left  != 0 else None,
                                       self.elements[el_id_right-1] if el_id_right != 0 else None]
            
            # Link the mortar to the elements
            if el_id_left != 0:
                self.elements[el_id_left-1 ].connected_mortars[el_face_left-1 ] = mort
            if el_id_right != 0:
                self.elements[el_id_right-1].connected_mortars[el_face_right-1] = mort

            # If this is a boundary mortar, grab the boundary condition name from the element. Remember, if the
            # RIGHT element id = 0, that means the LEFT element is physical and contains the BC information; if
            # the LEFT element id = 0, that means the RIGHT element is physical contains the BC information.
            if el_id_left == 0:
                mort.bc_name = self.elements[el_id_right-1].boundary_condition_names[el_face_right-1]
            if el_id_right == 0:
                mort.bc_name = self.elements[el_id_left-1 ].boundary_condition_names[el_face_left-1 ]

    def construct_mesh(self) -> None:
        """
        Constructs the mesh by calling the appropriate methods in the correct order. This is a wrapper function
        that calls the following methods in order:
            - self.construct_elements()
            - self.construct_mortars()
            - self.link_elements_and_mortars()
            - self.compute_isoparametric_mapping()
            - self.apply_mortar_curvature()
            - self.apply_boundary_conditions()
        """

        self.construct_elements()
        self.construct_mortars()
        self.link_elements_and_mortars()
        # self.compute_isoparametric_mapping()
        # self.apply_mortar_curvature()
        # self.apply_boundary_conditions()

    def compute_isoparametric_mapping(self) -> None:
        """
        Maps coordinates from the reference element [-1, 1]^D (where D is the dimensionality of the mesh) to 
        the physical domain in each element. This is done following the standard isoparametric mapping procedure 
        described in         Chapter 6 of David Kopriva's "Implementing Spectral Methods for Partial Differential
        Equations".
        """

        # Create a dispatch dictionary to call appropriate mapping funciton based on dimensionality.
        MAPPING_DISPATCH = {
            1: mapping.isop_map_1d,
            2: mapping.isop_map_2d,
            3: mapping.isop_map_3d,
        }

        # Throw error if dimensionality is not supported
        if self.dim not in MAPPING_DISPATCH:
            raise NotImplementedError(f"Isoparametric mapping not implemented for dimensionality: {self.dim}")
        
        # Call map dispatcher
        MAPPING_DISPATCH[self.dim](self)
    
    def apply_mortar_curvature(self) -> None:
        """
        Applies mortar curvature to the mortar objects based on the information in self.raw_data.
        This MUST be called after self.construct_elements() and self.construct_mortars().
        """

        raise NotImplementedError("Applying mortar curvature not implemented yet!")
    
    def apply_boundary_conditions(self) -> None:
        """
        Applies boundary conditions to the mortars based on the information in self.raw_data.
        This MUST be called after self.construct_elements() and self.construct_mortars().
        """

        raise NotImplementedError("Applying boundary conditions not implemented yet!")