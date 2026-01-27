# src/dingus/mesh/hohqmesh_handler.py
from pathlib import Path
from typing import Dict, Any
from itertools import islice
import numpy as np

def read_hohqmesh(path: str) -> Dict[str, Any]:
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
        raise FileNotFoundError(f'Mesh file not found: {p}')

    # Read in the mesh file header to check the format type. Valid
    # formats are ISM, ISM-V2, ISM-MM, and ABAQUS.
    with open(p, 'r') as meshFile:
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
            raise NotImplementedError(f"ISM and ISM-MM format not implemented for HOHQMesh mesh files!")
        
    raise ValueError(f"Unknown format for HOHQMesh mesh file! Header: {header}")
    
def read_ism_v2_2D(p: Path) -> Dict[str, Any]:
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
    with open(p, 'r') as meshFile:
        # Read the header and save for verification
        meshData["format"] = meshFile.readline().strip()

        # Read in the metadata: NumNodes, numMortars, NumElements, BC_PolyOrder
        # Note: numMortars includes inter-element boundaries while BC_PolyOrder 
        # refers only to elements with physical boundary conditions.
        header = next(meshFile).split()
        numNodes, numMortars, numElements, BCPolyOrder = map(int, header[:4])
        meshData["num_nodes"]       = numNodes
        meshData["num_mortars"]  = numMortars
        meshData["num_elements"]    = numElements
        meshData["bc_poly_order"]   = BCPolyOrder

        # Read in the node coordinates using np.loadtxt. This is more robust than
        # np.fromfile, however for large meshes it is slower.
        meshData["nodes"] = np.loadtxt(
            meshFile,
            dtype=np.float64,
            usecols=(0, 1, 2),
            max_rows=numNodes
        )

        # Read in the node coordinates using np.fromfile. This is the fastest option,
        # # but is notably less robust than np.loadtxt. 
        # meshData["nodes"] = np.fromfile(
        #     meshFile,
        #     dtype=np.float64,
        #     sep=" ",
        #     count=numNodes * 3
        # ).reshape(numNodes, 3)

        # Read in the connectivity / mortar data
        meshData["mortars"] = np.fromfile(
            meshFile,
            dtype=np.int32,
            sep=" ",
            count=numMortars * 6
        ).reshape(numMortars, 6)

        # Read in the boundary condition and element data
        rawElementData = list(islice(meshFile, numElements*3))

        # eliminate trailing newline characters and spaces
        rawElementData = [line.strip() for line in rawElementData]

        # Grab the nodeIDs for each element, mortar curvature type, and BC names for 
        # each element. Because the data is stored in the same line per element 
        # block, we can use slicing to quickly extract the data.
        meshData["elementNodeIDs"]      = np.fromstring(" ".join(rawElementData[0::3]), sep=" ", dtype=np.int32).reshape(numElements,4)
        meshData["mortarCurvatureType"] = np.fromstring(" ".join(rawElementData[1::3]), sep=" ", dtype=np.int32).reshape(numElements,4)

        # Because the BC data is string-based, we need to parse it differently
        tempBCData                  = rawElementData[2::3]
        tempBCData_split            = " ".join(tempBCData).split()
        meshData["elementBCNames"]  = np.array(tempBCData_split, dtype=object).reshape(numElements, 4)

        # Optional: convert '---' to None
        meshData["elementBCNames"][meshData["elementBCNames"] == "---"] = None

        return meshData