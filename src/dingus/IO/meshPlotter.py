# src/dingus/plotting/meshPlotter.py
from __future__ import annotations

from matplotlib.axes import Axes
from dingus.mesh import mesh_class
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

def plot_mesh(myMesh: 'mesh_class.Mesh', ax: Optional[plt.Axes] = None, show: bool = True,
                 edge_color="k", node_color="b", quad_node_color="r", lw=1.0, ms=4.0, quad_ms=2.5):
    '''
    Visualize the mesh structure (elements and quadrature nodes). This is primarily for debugging purposes to ensure
    mesh structures are properly generated.

    Inputs: 
    - myMesh: Mesh object containing the element structure and corresponding quadrature nodes.
    '''

    # Create a dispatch dictionary to call the appropriate plotting function based on mesh dimensionality
    PLOTTING_DISPATCH = {
        1: plot_mesh_1D,
        2: plot_mesh_2D,
        3: plot_mesh_3D
    }

    # Call the appropriate plotting function based on mesh dimensionality
    if myMesh.dim in PLOTTING_DISPATCH:
        return PLOTTING_DISPATCH[myMesh.dim](myMesh, ax=ax, show=show, edge_color=edge_color, node_color=node_color,
                                              quad_node_color=quad_node_color, lw=lw, ms=ms, quad_ms=quad_ms)
    else:
        raise NotImplementedError(f"Mesh dimensionality {myMesh.dim} not supported.")

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

def plot_mesh_1D(myMesh: 'mesh_class.Mesh', ax: Optional[plt.Axes] = None, show: bool = True,
                 edge_color="k", node_color="b", quad_node_color="r", lw=1.0, ms=4.0, quad_ms=2.5):
    '''
    Visualize the 1D mesh structure (elements and quadrature nodes). This is primarily for debugging purposes to ensure
    mesh structures are properly generated.

    Inputs: 
    - myMesh: Mesh object containing the element structure and corresponding quadrature nodes.
    '''

    # TODO: Implement 1D mesh plotting (line segments for elements, points for nodes and quadrature nodes)

    raise NotImplementedError("1D mesh plotting not yet implemented.")

def plot_mesh_2D(myMesh: 'mesh_class.Mesh', ax: Optional[plt.Axes] = None, show: bool = True,
                 edge_color="k", node_color="b", quad_node_color="r", lw=1.0, ms=4.0, quad_ms=2.5):
    '''
    Visualize the 2D mesh structure (elements and quadrature nodes). This is primarily for debugging purposes to ensure
    mesh structures are properly generated.

    Plots:
    - Element boundaries (quad outlines) using corner node coordinates
    - Quadrature nodes inside each element, if isoparametric mapping has been computed
    (i.e., e.quad_node_coords is non-empty)

    Inputs: 
    - myMesh          : Mesh object containing the element structure and corresponding quadrature nodes.
    - ax              : Optional existing Axes to plot into. A new figure is created if None.
    - show            : If True, call plt.show() at the end.
    - edge_color      : Color for element boundary edges.
    - node_color      : Color for element corner nodes.
    - quad_node_color : Color for quadrature nodes inside each element.
    - lw              : Line width for element edges.
    - ms              : Marker size for element corner nodes.
    - quad_ms         : Marker size for quadrature nodes.

    Outputs:
    - ax: The matplotlib Axes object.
    '''

    # Create a figure if Axes is not provides
    if ax is None:
        fig, ax = plt.subplots()

    # Initialize a switch that signals if quadrature nodes are present in the mesh or not
    has_quad_nodes = False

    # Loop through each element, plotting the element boundaries and quadrature nodes (if present)
    for e in myMesh.elements:
        # --- Draw element boundary ---
        corners = e.node_coords[:,:2]  # shape (4, 2) for 2D quads

        # Create a closed polygon by appending the first corner to the end of the coordinate list
        el_poly = np.vstack([corners, corners[0]])

        # Plot the element
        ax.plot(el_poly[:, 0], el_poly[:, 1], "-", color=edge_color, linewidth=lw)

        # --- Draw the corner nodes --
        ax.plot(corners[:, 0], corners[:, 1], "o", color=node_color, markersize=ms)

        # Draw the quadrature nodes if the isoparametric mapping has been computed
        if e.quad_node_coords is not None and np.asarray(e.quad_node_coords).size > 0:
            has_quad_nodes = True    # Update switch
            quad_coords    = np.asarray(e.quad_node_coords) # Shape (P, P, 2) OR (P*P, 2)

            # Plot quadrature nodes
            ax.plot(quad_coords[:, :, 0].ravel(), quad_coords[:, :, 1].ravel(), ".", 
                    color=quad_node_color, markersize=quad_ms)
            
            # # Draw dashed lines along rows (constant xi)
            # for i in range(quad_coords.shape[0]):
            #     ax.plot(quad_coords[i, :, 0], quad_coords[i, :, 1], "--", color=quad_node_color, linewidth=0.5)

            # # Draw dashed lines along columns (constant eta)
            # for j in range(quad_coords.shape[1]):
            #     ax.plot(quad_coords[:, j, 0], quad_coords[:, j, 1], "--", color=quad_node_color, linewidth=0.5)
    
    # Final adjustments to the plot
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    title = "2D Mesh"
    if has_quad_nodes:
        title += " with Quadrature Nodes"
    ax.set_title(title)

    if show:
        plt.show()

    return ax

def plot_mesh_3D(myMesh: 'mesh_class.Mesh', ax: Optional[plt.Axes] = None, show: bool = True,
                 edge_color="k", node_color="b", quad_node_color="r", lw=1.0, ms=4.0, quad_ms=2.5):
    '''
    Visualize the 3D mesh structure (elements and quadrature nodes). This is primarily for debugging purposes to ensure
    mesh structures are properly generated.

    Inputs: 
    - myMesh: Mesh object containing the element structure and corresponding quadrature nodes.
    '''

    raise NotImplementedError("3D mesh plotting not yet implemented.")