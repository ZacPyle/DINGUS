# src/dingus/IO/solutionPlotter.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from dingus.config import CaseCfg
from dingus.mesh import mesh_class
from typing import Dict, Any, Optional

def plot_sol_1D(input_mesh: 'mesh_class.Mesh', ax: Optional[plt.Axes] = None, eq_idx: int = 1, cmap: str = 'jet',
                visible: bool = False, plot_elements: bool = False, plot_quadrature_nodes: bool = False, 
                edge_color="k", node_color="k", lw=1.0, ms=4.0, plot_title: str = ""):
    '''
    Plots 1D line plots of the solution at all quadrature nodes. This is primarily used for debugging OR
    for monitoring a solution as it computes. As plotting is somewhat expensive, it is much faster to write
    the data out to files and post-process / visualize them AFTER the solution has been computed.

    Inputs: 
    - input_mesh          : Mesh object containing the element structure and corresponding quadrature nodes.
    - eq_idx                : Index of the equation/variable to plot (1-based). For example,
                              1 for density, 2 for x-momentum, etc
    - ax              : Optional existing Axes to plot into. A new figure is created if None.
    - visible         : If True, make the plot visible.
    - plot_elements   : If True, plot the element boundaries.
    - plot_nodes      : If True, plot the nodes.
    - edge_color      : Color for element boundary edges.
    - node_color      : Color for element corner nodes.
    - lw              : Line width for plot edges.

    Outputs:
    - ax: The matplotlib Axes object.
    '''
    raise NotImplementedError("solutionPlotter.plot_sol_1D() is not yet implemented!")
    return

def plot_sol_2D(input_mesh: 'mesh_class.Mesh', ax: Optional[plt.Axes] = None, eq_idx: int = 1, cmap: str = 'jet',
                visible: bool = False, plot_elements: bool = False, plot_quadrature_nodes: bool = False, 
                edge_color="k", node_color="k", lw=1.0, ms=4.0, plot_title: str = ""):
    '''
    Plots 2D contour plots of the solution at all quadrature nodes. This is primarily used for debugging OR
    for monitoring a solution as it computes. As plotting is somewhat expensive, it is much faster to write
    the data out to files and post-process / visualize them AFTER the solution has been computed.

    Inputs: 
    - input_mesh          : Mesh object containing the element structure and corresponding quadrature nodes.
    - eq_idx                : Index of the equation/variable to plot (1-based). For example,
                              1 for density, 2 for x-momentum, etc
    - ax              : Optional existing Axes to plot into. A new figure is created if None.
    - visible         : If True, make the plot visible.
    - plot_elements   : If True, plot the element boundaries.
    - plot_nodes      : If True, plot the nodes.
    - edge_color      : Color for element boundary edges.
    - node_color      : Color for element corner nodes.
    - lw              : Line width for plot edges.

    Outputs:
    - ax: The matplotlib Axes object.
    '''

    # TODO: CONVERT TO A GENERIC STAT VARIABLE PLOTTER
    # TODO: CREATE PLOT2D AND PLOT3D VERSIONS
    # TODO: WRITE GENERIC WRAPPER WITH DISPATCHES AS DONE IN MESH ROUTINES
    # TODO: ADD PRESSURE() AND TEMPERATURE() METHODS TO A PHYSICS(?) CLASS

    # Create a figure if Axes is not provides
    if ax is None:
        fig, ax = plt.subplots()

    # Assemble solution values at all quadrature nodes, in the same order
    # as the coordinates were assembled in build_delaunay_triangulation()
    #
    # element.solution shape is assumed to be (num_eq, P+1, P+1), so we extract
    # equation eq_idx and flatten the spatial dimensions to (P+1)^2 values.
    all_coords = np.vstack([
        e.quad_node_coords[:, :, :2].reshape(-1, 2)
        for e in input_mesh.elements
    ])
    all_values = np.concatenate([
        e.solution[:, :, eq_idx-1].ravel()
        for e in input_mesh.elements
    ])

    # Apply the same deduplication that build_delaunay_triangulation() used,
    # so that all_values[unique_idx] corresponds exactly to delaunay_points.
    if input_mesh.quad_type == "LGL":
        unique_idx  = input_mesh.delaunay_unique_idx
        plot_values = all_values[unique_idx]
    else:
        plot_values   = all_values

    # Filled contour plot over the full triangulation
    pts  = input_mesh.delaunay_coords
    tris = input_mesh.delaunay_tri.simplices
    cf   = ax.tripcolor(pts[:, 0], pts[:, 1], tris, plot_values,
                      shading='gouraud', cmap=cmap)
    plt.colorbar(cf, ax=ax)

    # Loop through each element, plotting the solution and element boundaries / quadrature nodes (if desired)
    for e in input_mesh.elements:
        # Plot the solution at the quadrature nodes as a contour plot
        

        # Plot the element boundary
        if plot_elements:
            corners = e.node_coords[:,:2]  # shape (4, 2) for 2D quads

            # Create a closed polygon by appending the first corner to the end of the coordinate list
            el_poly = np.vstack([corners, corners[0]])

            # Plot the element
            ax.plot(el_poly[:, 0], el_poly[:, 1], "-", color=edge_color, linewidth=lw)

            # --- Draw the corner nodes --
            # ax.plot(corners[:, 0], corners[:, 1], "o", color=node_color, markersize=ms)

        # Plot the quadrature nodes
        if plot_quadrature_nodes:
            quad_coords    = np.asarray(e.quad_node_coords) # Shape (P, P, 2) OR (P*P, 2)

            # Plot quadrature nodes
            ax.plot(quad_coords[:, :, 0].ravel(), quad_coords[:, :, 1].ravel(), ".", 
                    color=node_color, markersize=ms)
    
    # Final adjustments to the plot
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(plot_title)

    if visible:
        plt.show()

    return ax

def plot_sol_3D(input_mesh: 'mesh_class.Mesh', ax: Optional[plt.Axes] = None, eq_idx: int = 1, cmap: str = 'jet',
                visible: bool = False, plot_elements: bool = False, plot_quadrature_nodes: bool = False, 
                edge_color="k", node_color="k", lw=1.0, ms=4.0, plot_title: str = ""):
    '''
    Plots 3D isocontour plots of the solution at all quadrature nodes. This is primarily used for debugging OR
    for monitoring a solution as it computes. As plotting is somewhat expensive, it is much faster to write
    the data out to files and post-process / visualize them AFTER the solution has been computed.

    Inputs: 
    - input_mesh          : Mesh object containing the element structure and corresponding quadrature nodes.
    - eq_idx                : Index of the equation/variable to plot (1-based). For example,
                              1 for density, 2 for x-momentum, etc
    - ax              : Optional existing Axes to plot into. A new figure is created if None.
    - visible         : If True, make the plot visible.
    - plot_elements   : If True, plot the element boundaries.
    - plot_nodes      : If True, plot the nodes.
    - edge_color      : Color for element boundary edges.
    - node_color      : Color for element corner nodes.
    - lw              : Line width for plot edges.

    Outputs:
    - ax: The matplotlib Axes object.
    '''
    raise NotImplementedError("solutionPlotter.plot_sol_3D() is not yet implemented!")
    return

def plot_state_vars(input_mesh: 'mesh_class.Mesh', input_config: 'CaseCfg', cmap: str = 'jet',
                visible: bool = False, plot_elements: bool = False, plot_quadrature_nodes: bool = False, 
                edge_color="k", node_color="k", lw=1.0, ms=4.0):
    '''
    This function plots the state variables (density, momentum, energy) across the domain. It can be used to visualize the solution 
    at a point in time. This should primarily be used for debugging or monitoring the solution as it computes, as plotting can be
    expensive.
    '''

    # Create a dispatch dictionary to call appropriate plotting function based on mesh dimension.
    PLOTTER_DISPATCH = {
        1: plot_sol_1D,
        2: plot_sol_2D,
        3: plot_sol_3D
    }

    STATE_VAR_NAMES = {
        1: ["Density", "X-Momentum", "Energy"],
        2: ["Density", "X-Momentum", "Y-Momentum", "Energy"],
        3: ["Density", "X-Momentum", "Y-Momentum", "Z-Momentum", "Energy"]
    }

    # Throw error if not 1, 2, or 3D mesh.
    if input_mesh.dim not in PLOTTER_DISPATCH:
        raise ValueError(f"Unsupported mesh dimension {input_mesh.dim} for plotting. Supported dimensions: {list(PLOTTER_DISPATCH.keys())}")
    
    # Make sure you switch to interactive mode, otherwise the plots will block code execution until they are closed
    plt.ion()

    # Call the solution plotter for each state variable
    for eq in range(0, input_config.physics.num_eq):
        PLOTTER_DISPATCH[input_mesh.dim](input_mesh, ax = None, eq_idx = eq+1, cmap = 'jet',
                visible= True, plot_elements = False, plot_quadrature_nodes = False, 
                edge_color="k", node_color="k", lw=1.0, ms=4.0, plot_title = STATE_VAR_NAMES[input_mesh.dim][eq])
        
        plt.draw()
        plt.pause(0.01)  # Briefly yields control to the GUI event loop to render the plot

    plt.ioff()
    return
    