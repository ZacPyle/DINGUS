# src/dingus/IO/outputFileWriter.py
import dingus.physics.constitutiveRelations as constRelations
import h5py
import meshio
import numpy as np
import os
from dingus.config import CaseCfg
from dingus.mesh import mesh_class
from pathlib import Path
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

def _build_quad_connectivity(n_elements: int, p_plus_1: int) -> np.ndarray:
    '''
    Builds a (n_quad_cells, 4) connectivity array of global node indices.
 
    Each spectral element contributes a (P+1, P+1) grid of nodes,
    flattened in row-major order (matching `arr.reshape(-1, dim)` on a
    (P+1, P+1, dim) array, i.e., node k <-> (i, j) = (k // P1, k % P1)).
    Within an element, that grid is decomposed into (P)x(P) bilinear quad
    cells connecting four neighboring nodes:
 
        (i,   j)  ->  (i+1, j)  ->  (i+1, j+1)  ->  (i, j+1)
 
    Global indices are offset by `el_idx * P+1**2` so connectivity refers
    correctly into the flat (N_total,) coordinate/solution arrays built by
    `write_state_vars_to_file`, which stack elements one after another in
    element order.
 
    Inputs:
    - n_elements : number of spectral elements in the mesh
    - p_plus_1   : number of nodes per direction within each element (P+1)
 
    Outputs:
    - connectivity : (n_elements * (p_plus_1-1)**2, 4) int array of global
                      node indices, ordered counter-clockwise within each cell
    '''
 
    P1 = p_plus_1
    P  = P1 - 1  # number of cells per direction within an element
 
    # Local (single-element) connectivity for a (P1, P1) grid of nodes
    i_idx, j_idx = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    i_idx = i_idx.ravel()
    j_idx = j_idx.ravel()
 
    n00 = i_idx * P1 + j_idx
    n10 = (i_idx + 1) * P1 + j_idx
    n11 = (i_idx + 1) * P1 + (j_idx + 1)
    n01 = i_idx * P1 + (j_idx + 1)
 
    local_conn = np.stack([n00, n10, n11, n01], axis=1)  # (P*P, 4)
 
    # Tile across elements with a per-element global offset
    el_offsets = (np.arange(n_elements) * P1 * P1)[:, None, None]   # (n_el, 1, 1)
    connectivity = (local_conn[None, :, :] + el_offsets).reshape(-1, 4)
 
    return connectivity.astype(np.int64)

def _build_structured_quad_connectivity(n_rows: int, n_cols: int) -> np.ndarray:
    '''
    Builds a (n_cells, 4) connectivity array for a regular (n_rows x n_cols) grid
    of nodes flattened row-major (flat index k = row * n_cols + col), matching a
    `np.meshgrid(..., indexing="xy").ravel()` layout.

    The grid is decomposed into (n_rows-1)*(n_cols-1) bilinear quad cells, each
    wound counter-clockwise:

        (r,   c)  ->  (r,   c+1)  ->  (r+1, c+1)  ->  (r+1, c)

    Inputs:
    - n_rows : number of grid nodes in the (row / y) direction
    - n_cols : number of grid nodes in the (column / x) direction

    Outputs:
    - connectivity : (n_cells, 4) int64 array of node indices
    '''

    r, c = np.meshgrid(np.arange(n_rows - 1), np.arange(n_cols - 1), indexing="ij")
    r = r.ravel()
    c = c.ravel()

    n00 =  r      * n_cols +  c
    n10 =  r      * n_cols + (c + 1)
    n11 = (r + 1) * n_cols + (c + 1)
    n01 = (r + 1) * n_cols +  c

    return np.stack([n00, n10, n11, n01], axis=1).astype(np.int64)

def _interpolate_to_uniform_grid(input_mesh: 'mesh_class.Mesh',
                                 node_values: np.ndarray,
                                 res: int) -> tuple:
    '''
    Interpolates scattered per-node state data onto a (res x res) uniform grid
    spanning the domain bounding box, reusing the mesh's precomputed Delaunay
    triangulation (it is NOT rebuilt here).

    Linear interpolation is used inside the triangulation's convex hull; any grid
    points that fall marginally outside the hull (which would otherwise be NaN)
    are filled by nearest-neighbour lookup so the exported field is gap-free.

    The de-duplicated triangulation node set already excludes the doubled element
    -interface nodes present for LGL quadrature (see Mesh.build_delaunay_tri), so
    the interpolated grid carries a single, well-defined value at every point.

    Inputs:
    - input_mesh  : Mesh with `delaunay_tri` / `delaunay_coords` populated
    - node_values : (n_nodes, n_vars) values ordered like the element-stacked
                     quadrature nodes (i.e. `all_coords` order)
    - res         : number of uniform grid points per direction

    Outputs:
    - grid_coords : (res*res, 2) uniform grid node coordinates
    - grid_values : (res*res, n_vars) interpolated state values
    - connectivity: (n_cells, 4) structured quad connectivity for the grid
    '''

    if input_mesh.dim != 2:
        raise NotImplementedError("Uniform-grid output is only implemented for 2D meshes.")

    tri = input_mesh.delaunay_tri
    pts = input_mesh.delaunay_coords

    # Align values to the (de-duplicated) triangulation node set. LGL drops the
    # doubled interface nodes; LG uses every node as-is.
    if input_mesh.quad_type == "LGL":
        vals = node_values[input_mesh.delaunay_unique_idx]
    else:
        vals = node_values

    # Uniform grid over the domain bounding box.
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    xs = np.linspace(xmin, xmax, res)
    ys = np.linspace(ymin, ymax, res)
    gx, gy      = np.meshgrid(xs, ys, indexing="xy")   # (res, res); row -> y, col -> x
    grid_coords = np.column_stack([gx.ravel(), gy.ravel()])

    # Linear interpolation inside the hull; nearest-neighbour fallback for any
    # grid points that land just outside it.
    grid_values = LinearNDInterpolator(tri, vals)(grid_coords)
    nan_rows    = np.isnan(grid_values).any(axis=1)
    if nan_rows.any():
        grid_values[nan_rows] = NearestNDInterpolator(pts, vals)(grid_coords[nan_rows])

    connectivity = _build_structured_quad_connectivity(res, res)
    return grid_coords, grid_values, connectivity

def _write_xmf(xmf_path: Path,
               h5_filename: str, 
               var_names: list, 
               n_nodes: int, 
               n_quad_cells: int,
               dim: int, 
               time: float):
    '''
    Writes the XMF (XDMF) sidecar file that points ParaView to the HDF5 data.
 
    Preserves the DG element structure by using an explicit `Quadrilateral`
    topology (one cell per (P, P) sub-quad inside each spectral element)
    rather than a disconnected `Polyvertex` point cloud. This keeps the
    non-uniform LG/LGL node spacing visible and lets ParaView treat the mesh
    as a proper unstructured grid (contours, slices, streamlines, etc. all
    work correctly across element boundaries).
 
    Connectivity is stored as its own HDF5 dataset (`/Connectivity`, shape
    (n_quad_cells, 4)) rather than inlined as XML text, since for realistic
    node counts an inline integer list would bloat the XMF file considerably.
    '''
 
    coord_labels  = ["X", "Y", "Z"][:dim]
    geometry_type = {1: "X", 2: "XY", 3: "XYZ"}[dim]
 
    coord_lines = "\n".join([
        f'        <DataItem Dimensions="{n_nodes}" NumberType="Float" Precision="8" Format="HDF">\n'
        f'          {h5_filename}:/Coordinates/{ax}\n'
        f'        </DataItem>'
        for ax in coord_labels
    ])
 
    attr_lines = "\n".join([
        f'      <Attribute Name="{name}" AttributeType="Scalar" Center="Node">\n'
        f'        <DataItem Dimensions="{n_nodes}" NumberType="Float" Precision="8" Format="HDF">\n'
        f'          {h5_filename}:/Solution/{name}\n'
        f'        </DataItem>\n'
        f'      </Attribute>'
        for name in var_names
    ])
 
    xmf_content = f"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
  <Domain>
    <Grid Name="mesh" GridType="Uniform">
      <Time Value="{time}" />
      <Topology TopologyType="Quadrilateral" NumberOfElements="{n_quad_cells}">
        <DataItem Dimensions="{n_quad_cells} 4" NumberType="Int" Format="HDF">
          {h5_filename}:/Connectivity
        </DataItem>
      </Topology>
      <Geometry GeometryType="{geometry_type}">
{coord_lines}
      </Geometry>
{attr_lines}
    </Grid>
  </Domain>
</Xdmf>
"""
    xmf_path.write_text(xmf_content)

def _write_hdf5(case_name: str, 
                output_dir: Path,
                time: float, 
                step: int, 
                all_coords: np.ndarray, 
                all_solutions: np.ndarray, 
                output_var_names: list,
                connectivity: np.ndarray) -> None:
    '''
    Writes coordinate, solution, and connectivity data to HDF5 + companion XMF.
 
    HDF5 layout
    -----------
    /Coordinates/X   (n_nodes,)   — flat 1-D x-coordinate array
    /Coordinates/Y   (n_nodes,)   — flat 1-D y-coordinate array  (if applicable)
    /Coordinates/Z   (n_nodes,)   — flat 1-D z-coordinate array  (If applicable)
    /Solution/<name> (n_nodes,)   — one dataset per state variable
    /Connectivity    (n_quad_cells, 4) — global node indices per quad cell,
                                          preserving per-element DG structure

    Storing each coordinate axis separately (rather than a single (N, dim)
    array) keeps every Geometry DataItem in the XMF file 1-D, which is the
    layout ParaView expects when GeometryType is "XY" or "XYZ".
    '''

    # File information for hdf5 file(s)
    h5_filename  = f"{case_name}_step{step:06d}.h5"
    xmf_filename = f"{case_name}_step{step:06d}.xmf"
    h5_path      = output_dir / h5_filename
    xmf_path     = output_dir / xmf_filename

    # Extract number of nodes and dimensionality from coordinate array
    n_nodes      = all_coords.shape[0]
    dim          = all_coords.shape[1]
    n_quad_cells = connectivity.shape[0]

    # Coordinate lables, but only use 0:dim based on mesh dimensionality (e.g., for 2D, only use X and Y)
    coord_labels = ['X', 'Y', 'Z'][:dim]

    # # Create a dictionary mapping the output variable names to corresponding data arrays
    # output_data_dict = {var_name: all_solutions[:, idx] for idx, var_name in enumerate(output_var_names)}

    # Write the hdf5 file(s)
    with h5py.File(h5_path, 'w') as hf:
        # Write metadata
        hf.attrs['time'] = time
        hf.attrs['step'] = step

        # Write node coordinates; split into per-axis 1D arrays
        coord_grp = hf.create_group('Coordinates')
        for i, ax in enumerate(coord_labels):
            coord_grp.create_dataset(ax, data=all_coords[:, i], compression='gzip')

        # Write each solution variable
        sol_grp = hf.create_group('Solution')
        for idx, name in enumerate(output_var_names):
            #breakpoint()
            sol_grp.create_dataset(name, data=all_solutions[:, idx], compression='gzip')

        # Connectivity (per-element quad cells, preserves DG structure)
        hf.create_dataset("Connectivity", data=connectivity, compression="gzip")

    # Call helper function to write the accompanying XMF file for ParaView visualizations.
    _write_xmf(
        xmf_path, h5_filename, output_var_names, n_nodes, n_quad_cells, dim, time)

def _write_vtk(case_name: str,
               output_dir: Path,
               time: float,
               step: int,
               all_coords: np.ndarray,
               all_solutions: np.ndarray,
               output_var_names: list,
               connectivity: np.ndarray) -> None:
    '''
    Writes a native VTK unstructured-grid file (`.vtu`) via meshio.

    Unlike the HDF5 + XDMF path, a `.vtu` file is fully self-contained (geometry,
    connectivity, and field data in one file) and needs no companion sidecar,
    which avoids the XDMF `filename:/dataset` path-parsing pitfalls (notably the
    Windows drive-letter colon) that make hand-written XDMF fragile in ParaView.

    The DG element structure is preserved by emitting one `quad` cell per (P, P)
    sub-quad inside each spectral element, using the same per-element
    connectivity built by `_build_quad_connectivity`. Non-uniform LG/LGL node
    spacing is therefore visible and ParaView treats the result as a proper
    unstructured grid.

    ParaView auto-groups files sharing the `<case>_step<NNNNNN>.vtu` naming into
    a time series, so per-step files are sufficient until a `.pvd` collection is
    added alongside the solver's time loop.

    Inputs:
    - all_coords       : (n_nodes, dim) node coordinates (dim in {1, 2, 3})
    - all_solutions    : (n_nodes, n_vars) state-variable values, column order
                          matching `output_var_names`
    - output_var_names : names of the columns in `all_solutions`
    - connectivity     : (n_quad_cells, 4) global node indices per quad cell
    '''

    vtu_filename = f"{case_name}_step{step:06d}.vtu"
    vtu_path     = output_dir / vtu_filename

    n_nodes = all_coords.shape[0]
    dim     = all_coords.shape[1]

    # meshio always stores points as 3-D; pad the unused axes with zeros so a 2-D
    # (or 1-D) mesh lands on the z = 0 (and y = 0) plane in ParaView.
    points = np.zeros((n_nodes, 3), dtype=np.float64)
    points[:, :dim] = all_coords

    # One VTK "quad" cell block; connectivity already references the flat,
    # element-stacked node ordering used to build `points`.
    cells = [("quad", connectivity.astype(np.int64))]

    # Attach each state variable as node-centered point data.
    point_data = {
        name: all_solutions[:, idx]
        for idx, name in enumerate(output_var_names)
    }

    mesh = meshio.Mesh(
        points=points,
        cells=cells,
        point_data=point_data,
        field_data={"time": np.array([time])},
    )
    mesh.write(vtu_path)

def write_state_vars_to_file(input_mesh: 'mesh_class.Mesh', 
                             input_config: 'CaseCfg', 
                             time: float = 0.0, 
                             step: int = 0,
                             CASE_DIR: Path = Path('./')) -> None:
    '''
    Writes the solution state variables at all quadrature nodes to output files. This is intended to be called
    at specified intervals to save the solution data for post-processing and visualization.

    Inputs:
    - input_mesh          : Mesh object containing the element structure and corresponding quadrature nodes.
    - input_config        : Case configuration object containing simulation parameters and output settings.

    Outputs:
    - None (writes files to disk)
    '''

    # Extract relevant IO parameters from the case configuration
    plot_uniform_grid = input_config.io.plot_uniform_grid
    case_name         = input_config.name
    output_format     = input_config.io.output_format

    # Create output directory if it doesn't exist
    output_dir        = CASE_DIR / input_config.io.output_dir / output_format
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create IO writer dispatch based on output format
    WRITER_DISPATCH = {
        'hdf5': _write_hdf5,
        'vtk':  _write_vtk,
    }

    # State variable names by dimensionality
    STATE_VAR_NAMES = {
        1: ["Rho", "U",           "P", "T"],
        2: ["Rho", "U", "V",      "P", "T"],
        3: ["Rho", "U", "V", "W", "P", "T"]
    }
    var_names = STATE_VAR_NAMES[input_mesh.dim]

    # Extract the coordinate data from all quadrature nodes across all elements and reshape into a 2D array of shape (num_nodes, dim)
    all_coords = np.vstack([
        e.quad_node_coords[..., :input_mesh.dim].reshape(-1, input_mesh.dim)
        for e in input_mesh.elements
    ])

    # Extract the solution data (conservative variables) from all elements at the quadrature nodes and reshape into 2D array of shape (num_nodes, num_eq)
    all_sol = np.column_stack([
        np.concatenate([e.solution[..., eq_idx].ravel() for e in input_mesh.elements])
        for eq_idx in range(input_config.physics.num_eq)
    ])

    # Compute the state variables rather than the conservative variables (recall, the elements contain conservative variables)
    rho         = all_sol[:, 0:1] # must use 0:1 to maintain 2D shape
    pressure    = constRelations.compute_pressure(all_sol, input_config)
    temperature = constRelations.compute_temperature(all_sol, input_config)
    velocity    = all_sol[:, 1:-1] / rho

    # Combine state variables into a single output array
    state_vars = np.hstack([
        rho                       ,   # density 
        velocity                  ,   # velocity components
        pressure   [:, np.newaxis],   # pressure (must use np.newaxis to create 2D shape)
        temperature[:, np.newaxis]    # temperature (must use np.newaxis to create 2D shape)
    ])

    # Build per-element Quadrilateral connectivity (preserves DG element
    # structure and non-uniform quadrature node spacing in ParaView).
    # All elements share the same polynomial order (Mesh.el_poly_order is
    # global), so every element contributes the same number of nodes/cells.
    p_plus_1     = input_mesh.el_poly_order + 1
    connectivity = _build_quad_connectivity(len(input_mesh.elements), p_plus_1)

    # Select the output geometry: either the raw quadrature nodes (preserves the
    # exact DG nodal solution) or a uniform grid interpolated from them (regular,
    # post-processing-friendly). Both paths feed the same format writers.
    if plot_uniform_grid:
        out_coords, out_vars, out_connectivity = _interpolate_to_uniform_grid(
            input_mesh, state_vars, input_config.io.uniform_grid_res)
    else:
        out_coords, out_vars, out_connectivity = all_coords, state_vars, connectivity

    # Write the solution data to output file in specified format.
    WRITER_DISPATCH[output_format](
        case_name, output_dir, time, step, out_coords, out_vars, var_names, out_connectivity)