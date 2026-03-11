# src/dingus/mesh/hohqmesh_handler.py
from pathlib import Path
from typing import Dict, Any, Tuple, Union
from itertools import islice
import numpy as np

def read_hohqmesh(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Reads a HOHQMesh mesh file (see https://github.com/trixi-framework/HOHQMesh) and returns a
    general mesh dictionary. Because HOHQMesh can output multiple formats (i.e., ISM, ISM-v2,
    ABAQUS, ...), this function will call a specific reader based on the format type.

    Args:
        path (str): Path to the mesh file, including the file name. Example: 'mesh/Square.mesh'

    Returns:
        meshData[str, Any]: Dictionary containing the mesh data. Keys are strings, values can be
                            any variable type.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mesh file not found: {p}")

    # Read in the mesh file header to check the format type. Valid
    # formats are ISM, ISM-V2, ISM-MM, and ABAQUS.
    with open(p, "r") as meshFile:
        header_line = meshFile.readline()

    # Check the format type and call appropriate reader
    header = header_line.strip().lower()
    match header:
        case "ism-v2":
            return read_ism_v2_2D(p)

        case "*heading":
            # Call ABAQUS reader
            raise NotImplementedError("ABAQUS format not implemented for HOHQMesh mesh files!")

        case _:
            # This covers ism, ism-mm, and unknown formats
            raise NotImplementedError("ISM and ISM-MM format not implemented for HOHQMesh mesh files!")

    raise ValueError(f"Unknown format for HOHQMesh mesh file! Header: {header}")


def read_ism_v2_2D(p: Path) -> dict[str, Any]:
    """
    Reads a HOHQMesh mesh file writtin in the ISM-V2 format.

    Args:
        p (Path): Path to the mesh file, including the file name. Example: 'mesh/Square.mesh'

    Returns:
        meshData[str, Any]: Dictionary containing the mesh data. Keys are strings, values can be
                            any variable type.
    """

    # Initialize the output Dictionary
    meshData: Dict[str, Any] = {}

    # Open and parse the mesh file
    with open(p, "r") as meshFile:
        # Read the header and save for verification
        meshData["format"] = meshFile.readline().strip()

        # Read in the metadata: NumNodes, numMortars, NumElements, BC_PolyOrder
        # Note: numMortars includes inter-element boundaries while BC_PolyOrder
        # refers only to elements with physical boundary conditions.
        header = next(meshFile).split()
        numNodes, numMortars, numElements, BCPolyOrder = map(int, header[:4])
        meshData["num_nodes"]     = numNodes
        meshData["num_mortars"]   = numMortars
        meshData["num_elements"]  = numElements
        meshData["poly_order"]    = BCPolyOrder

        # Spline info in case of curved boundaries
        numSplineNodes = BCPolyOrder + 1

        # --- Nodes ---
        # Read in the node coordinates using np.loadtxt. This is more robust than
        # np.fromfile, however for large meshes it is slower.
        meshData["nodes"] = np.loadtxt(
            meshFile, dtype=np.float64, usecols=(0, 1, 2), max_rows=numNodes
        )

        # Read in the node coordinates using np.fromfile. This is the fastest option,
        # # but is notably less robust than np.loadtxt.
        # meshData["nodes"] = np.fromfile(
        #     meshFile,
        #     dtype=np.float64,
        #     sep=" ",
        #     count=numNodes * 3
        # ).reshape(numNodes, 3)

        # --- Mortars ---
        meshData["mortars"] = np.fromfile(
            meshFile, dtype=np.int32, sep=" ", count=numMortars * 6
        ).reshape(numMortars, 6)

        # --- Elements ---
        # Pre-allocate arrays for the straight-sided element data
        elementNodeIDs      = np.zeros((numElements, 4), dtype=np.int32)
        mortarCurvatureType = np.zeros((numElements, 4), dtype=np.int32)
        elementBCNames      = np.empty((numElements, 4), dtype=object)

        # Preallocate a dictionary for the spline data, keyed as
        # (element_idx, side_idx) -> np.ndarray of shape (numSplineNodes, 3)
        splineData: Dict[tuple, np.ndarray] = {}

        for el_idx in range(numElements):
            # First line of each element block contains the node IDs for each corner node
            elementNodeIDs[el_idx, :] = list(map(int, next(meshFile).split()))

            # Second line of each element block contains the curvature flags for each mortar
            bFlags = list(map(int, next(meshFile).split()))
            mortarCurvatureType[el_idx, :] = bFlags

            # Read in spline data for each curved mortar IF APPLICABLE
            for side_idx, flag in enumerate(bFlags):
                if flag == 1:
                    # Read in the spline knot points for the mortar. Each spline is composed
                    # of BCPolyOrder + 1 knot points.
                    knots = np.loadtxt(
                        islice(meshFile, numSplineNodes),
                        dtype=np.float64,
                        usecols=(0, 1, 2)
                    )
                    splineData[(el_idx, side_idx)] = knots

            # Final line in each element block is the BC name for each side. If no BC is applied, the name is '---'
            bc_names = next(meshFile).split()
            elementBCNames[el_idx, :] = [None if name == "---" else name for name in bc_names]

        meshData["elementNodeIDs"]      = elementNodeIDs
        meshData["mortarCurvatureType"] = mortarCurvatureType
        meshData["elementBCNames"]      = elementBCNames
        meshData["splineData"]          = splineData

        return meshData