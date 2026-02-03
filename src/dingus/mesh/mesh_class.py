# src/dingus/mesh/mesh_class.py

from pathlib import Path
from typing import Dict, Any, Union
from pprint import pprint

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
        self.num_elements  : int = 0
        self.num_nodes     : int = 0
        self.num_mortars   : int = 0
        self.el_poly_order : int = 0
        self.bc_poly_order : int = 0
        self.mesh_format   : str = ""
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
        import dingus.mesh.hohqmesh_handler as hohqmesh
        import dingus.mesh.gmsh_handler as gmsh

        # Instantiate empty attributes
        self.dim          : int = 2
        self.num_elements : int = 0
        self.num_nodes    : int = 0
        self.num_mortars  : int = 0
        self.bc_poly_order: int = 0

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
        self.bc_poly_order  = self.raw_data["bc_poly_order"]

    def construct_elements(self) -> None:
        """
        Constructs Element objects for each element in the mesh. This uses the attribute self.raw_data to
        determine the element information. This MUST be called after self.read_mesh().
        """

        from dingus.mesh import element_class

        # Make sure we have read in a mesh file first
        if not self.raw_data:
            raise RuntimeError("Must call the read_mesh() method before the construct_elements() method!")

        # Extract basic data from the mesh
        node_ids_per_element = self.raw_data["elementNodeIDs"]
        node_coordinates     = self.raw_data["nodes"]       

        # Create collection of element objects for each element described in the list of node IDs using
        # list comprehension
        self.elements = [
            element_class.SpectralElement(
                element_id  = el_id+1,  # Adding 1 to convert from 0-based to 1-based indexing
                node_ids    = node_ids_per_element[el_id, :],
                node_coords = node_coordinates[node_ids_per_element[el_id, :] - 1, :],  # node IDs are 1-based; must convert to 0-based for indexing
                poly_order  = self.el_poly_order,
                el_type     = "line" if self.dim == 1 else "quad" if self.dim == 2 else "hex" if self.dim == 3 else "unknown"
            )
            for el_id in range(self.num_elements)
        ]

    def construct_mortars(self) -> None:
        """
        Constructs Mortar objects for each mortar in the mesh. This uses the attribute self.raw_data to
        determine the element information. This MUST be called after self.read_mesh().
        """

        from dingus.mesh import mortar_class

        # Extract basic data from the mesh
        mortar_data      = self.raw_data["mortars"]
        node_coordinates = self.raw_data["nodes"]  

        self.mortars = [
            # Determine if 2D or 3D mortar based on the number of nodes per mortar
            mortar_class.SpectralMortar(
                mortar_id   = mortar_id+1,  # Adding 1 to convert from 0-based to 1-based indexing
                node_ids    = mortar_data[mortar_id, :2],
                node_coords = node_coordinates[mortar_data[mortar_id, :2] -1, :],  # node IDs are 1-based; must convert to 0-based for indexing
                poly_order  = self.el_poly_order,
                mortar_type = "edge" if self.dim == 2 else "face" if self.dim == 3 else "unknown"
            )
            for mortar_id in range(self.num_mortars)
        ]

    def link_elements_and_mortars(self) -> None:
        """
        Links Element and Mortar objects together (i.e., what mortars are associated with each element). This 
        MUST be called after self.construct_elements() and self.construct_mortars().
        """

        raise NotImplementedError("Linking elements and mortars not implemented yet!")
    
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