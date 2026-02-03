# src/dingus/mesh/gmsh_handler.py
from pathlib import Path
from typing import Dict, Any, Optional, Union
import subprocess
import numpy as np
import meshio

# List of cell types for 1D, 2D, and 3D meshes in order of preference
CELL_TYPES = ["line", "line3", "triangle", "quad", "quad9", "tetra", "hexahedron"]


def read_gmsh(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a gmsh file (.msh) and return a general mesh dictionary."""

    p = Path(path)  # Path to the mesh file
    mesh = meshio.read(p)  # Mesh object from Python package 'meshio'

    pts = mesh.points  # (N_pts, 3) array of point coordinates

    # dictionary of cell blocks by type (e.g., 'vertex', 'line', 'triangle', etc.)
    # cell_blocks = {c.type: c.data for c in mesh.cells}

    cell_blocks = {}
    for c in mesh.cells:
        if c.type not in cell_blocks.keys():
            cell_blocks[c.type] = c.data
        else:
            cell_blocks[c.type] = np.concatenate((cell_blocks[c.type], c.data), axis=0)

    # physical tags: meshio exposes gmsh physical groups in cell_data_dict
    phys_tags = (
        mesh.cell_data_dict.get("gmsh:physical", {}) if hasattr(mesh, "cell_data_dict") else {}
    )

    # Build a canonical elements list that aggregates all preferred cell blocks.
    # Each entry is a dict: { 'type': str, 'nodes': ndarray, 'cell_block': str, 'local_index': int, 'physical_tag': Optional[int] }
    elements_list = []
    elements_by_block = {}
    primary_cell_type = None

    for t in CELL_TYPES:
        # if this cell blocks is one of the types defined in CELL_TYPES...
        if t in cell_blocks:
            block = np.asarray(cell_blocks[t], dtype=int)
            elements_by_block[t] = block

            # mark first found as primary
            primary_cell_type = t

            # physical tags for this block (may be None)
            phys_block = None
            if isinstance(phys_tags, dict):
                phys_block = phys_tags.get(t)

            for local_idx, conn in enumerate(block):
                tag = None
                if phys_block is not None and len(phys_block) > local_idx:
                    try:
                        tag = int(phys_block[local_idx])
                    except Exception:
                        tag = phys_block[local_idx]

                elements_list.append(
                    {
                        "type": t,
                        "nodes": np.asarray(conn, dtype=int),
                        "cell_block": t,
                        "local_index": int(local_idx),
                        "physical_tag": tag,
                    }
                )

    return {
        "points": pts,
        "cells": {k: np.asarray(v, dtype=int) for k, v in cell_blocks.items()},
        "primary_cell_type": primary_cell_type,
        "elements": elements_list,
        "elements_by_block": elements_by_block,
        "cell_data": mesh.cell_data,
        "physical_tags": phys_tags,
        "meshio_mesh": mesh,
    }
