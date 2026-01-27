# src/dingus/mesh/general_mesh_handler.py
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

def plot_gmesh(mesh: Dict[str, Any], ax: Optional[plt.Axes] = None, show: bool = True, edge_color="k", node_color="r", lw=1.0, ms=3.0):
    """
    Visualize a mesh from the standardized dict returned by read_mesh().

    Supports:
      - 1D: line cells
      - 2D: triangle cells (filled outline) and/or quad cells (outline)
    """
    pts          = np.asarray(mesh["points"], dtype=float)
    cells        = mesh.get("cells", {})
    primary_type = mesh.get("primary_cell_type", None)

    if ax is None:
        fig, ax = plt.subplots()

    # ---- 1D: draw line segments in x-y (z ignored) ----
    if primary_type == "line":
        conn = np.asarray(cells["line"], dtype=int)
        xy   = pts[:, :2]  # use x,y

        # draw each segment
        for (i, j) in conn:
            ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], "-",color=edge_color, linewidth=lw)

        # draw nodes
        ax.plot(xy[:, 0], xy[:, 1], "o", color=node_color, markersize=ms)
        ax.set_aspect("equal", adjustable="box")

        # for a pure 1D mesh, y is constant -> make it readable
        yspan = xy[:, 1].max() - xy[:, 1].min()
        if yspan < 1e-12:
            ax.set_ylim(xy[0, 1] - 0.1, xy[0, 1] + 0.1)

        # titles and labels
        ax.set_title(" ")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if show:
            plt.show()
        return ax

    # ---- 2D triangles: use matplotlib Triangulation ----
    if primary_type == "triangle":
        import matplotlib.tri as mtri

        # Create triangulation from the points and triangle connectivity
        tri = np.asarray(cells["triangle"], dtype=int)
        x = pts[:, 0]
        y = pts[:, 1]
        triang = mtri.Triangulation(x, y, tri)

        # Plot the triangulation
        ax.triplot(triang, color=edge_color, linewidth=lw)
        ax.plot(x, y, "o", color=node_color, markersize=ms)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(" ")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if show:
            plt.show()
        return ax

    # ---- 2D quads: draw polygon outlines ----
    if primary_type == "quad":
        quad = np.asarray(cells["quad"], dtype=int)
        xy = pts[:, :2]

        # Plot the outline of the polynomial
        for q in quad:
            poly = np.vstack([xy[q], xy[q[0]]])  # close loop
            ax.plot(poly[:, 0], poly[:, 1], "-", linewidth=lw, color=edge_color)

        ax.plot(xy[:, 0], xy[:, 1], "o", color=node_color, markersize=ms)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(" ")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if show:
            plt.show()
        return ax

    raise NotImplementedError(f"No supported cell types found. Available: {list(cells.keys())}")