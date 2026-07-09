# src/dingus/IO/outputFileWriter.py
import dingus.physics.constitutiveRelations as constRelations
import h5py
import meshio
import numpy as np
import os
from dingus.config import CaseCfg
from dingus.coreNumerics import interpolation
from dingus.mesh import mesh_class
from pathlib import Path

def _build_quad_connectivity(n_elements: int, p_plus_1: int) -> np.ndarray:
    '''
    Builds a (n_quad_cells, 4) connectivity array of global node indices.
 
    Each spectral element contributes a (P+1, P+1) grid of nodes,
    flattened in row-major order (matching 'arr.reshape(-1, dim)' on a
    (P+1, P+1, dim) array, i.e., node k <-> (i, j) = (k // P1, k % P1)).
    Within an element, that grid is decomposed into (P)x(P) bilinear quad
    cells connecting four neighboring nodes:
 
        (i,   j)  ->  (i+1, j)  ->  (i+1, j+1)  ->  (i, j+1)
 
    Global indices are offset by 'el_idx * P+1**2' so connectivity refers
    correctly into the flat (N_total,) coordinate/solution arrays built by
    'write_state_vars_to_file', which stack elements one after another in
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

def _interpolate_to_uniform_grid(input_mesh: 'mesh_class.Mesh',
                                 node_values: np.ndarray,
                                 res: int) -> tuple:
    '''
    Resamples the solution onto a finer, uniformly-spaced grid for smooth visualization by
    evaluating each element's ACTUAL degree-P DG polynomial at uniform sub-points -- NOT by linear
    interpolation of the scattered nodal values.

    Why high-order (vs the previous LinearNDInterpolator approach): the DG solution inside an element
    is a smooth degree-P Lagrange polynomial. Linear interpolation over a triangulation of the nodes
    reconstructs only a faceted, C0 surface, so refining the grid just samples that same faceted
    surface more densely and the contours never get smoother past the node density. Here we instead
    apply the element's Lagrange interpolation matrix (from the reference quadrature nodes to a
    uniform reference sub-grid) as a tensor product, recovering the true curved field. Now increasing
    `res` genuinely smooths the output down to the real solution.

    Per-element subdivision (the standard high-order viz approach) is used: every element gets its own
    (m x m) uniform sub-grid in reference space [-1,1]^2. This keeps the DG element structure and
    faithfully shows any inter-element jumps (adjacent elements evaluate their own polynomial at the
    shared face). Both the state values AND the physical coordinates are interpolated with the same
    matrix, so each output point's location and value come from the same reference point (exact for
    straight-sided elements, isoparametric for curved ones).

    Inputs:
    - input_mesh  : constructed Mesh (provides quad_nodes, el_poly_order, and per-element geometry)
    - node_values : (n_nodes, n_vars) values ordered like the element-stacked quadrature nodes
                     (i.e. 'all_coords' order): element-by-element, row-major (P+1, P+1) within each.
    - res         : approximate number of sample points per direction ACROSS THE WHOLE DOMAIN. It is
                     split evenly among the elements (m ~ res / sqrt(n_elements) points per element per
                     direction), and clamped to at least P+1 so we never coarsen below the solution's
                     own node density.

    Outputs:
    - coords_out  : (n_elements*m*m, 2) sub-grid node coordinates
    - values_out  : (n_elements*m*m, n_vars) high-order-interpolated state values
    - connectivity: (n_cells, 4) per-element structured quad connectivity for the sub-grids
    '''

    if input_mesh.dim != 2:
        raise NotImplementedError("Uniform-grid output is only implemented for 2D meshes.")

    dim   = input_mesh.dim
    P1    = input_mesh.el_poly_order + 1                 # nodes per direction per element (P+1)
    n_el  = len(input_mesh.elements)
    nvars = node_values.shape[1]

    # Interpret `res` as points per direction across the whole domain; split evenly among elements.
    # Clamp to at least P1 so we never sample the polynomial more coarsely than its own nodes.
    n_el_per_dir = max(1, int(round(np.sqrt(n_el))))
    m            = max(P1, int(round(res / n_el_per_dir)))

    # One interpolation matrix, reused for every element: reference quad nodes -> m uniform points in
    # [-1, 1]. f_uniform = T @ f_nodes along each direction (tensor product for 2D).
    xi_uniform = np.linspace(-1.0, 1.0, m)
    T = interpolation.Polynomial_Interpolation_Matrix(input_mesh.quad_nodes, xi_uniform)  # (m, P1)

    coords_out = np.empty((n_el * m * m, dim),   dtype=float)
    values_out = np.empty((n_el * m * m, nvars), dtype=float)
    block      = P1 * P1

    for e_idx, e in enumerate(input_mesh.elements):
        # Reshape this element's flat node data back to its (P1, P1, .) tensor layout.
        vals = node_values[e_idx * block:(e_idx + 1) * block].reshape(P1, P1, nvars)
        xy   = e.quad_node_coords[..., :dim]                          # (P1, P1, dim)

        # Tensor-product high-order interpolation onto the (m, m) uniform reference grid.
        vals_u = np.einsum('ai,bj,ijv->abv', T, T, vals)             # (m, m, nvars)
        xy_u   = np.einsum('ai,bj,ijd->abd', T, T, xy)               # (m, m, dim)

        sl = slice(e_idx * m * m, (e_idx + 1) * m * m)
        values_out[sl] = vals_u.reshape(m * m, nvars)
        coords_out[sl] = xy_u.reshape(m * m, dim)

    # Per-element structured quad connectivity for the (m x m) sub-grids -- same builder as the raw
    # node path, with m nodes per direction instead of P1.
    connectivity = _build_quad_connectivity(n_el, m)
    return coords_out, values_out, connectivity

def _write_xmf(xmf_path: Path,
               h5_filename: str, 
               var_names: list, 
               n_nodes: int, 
               n_quad_cells: int,
               dim: int, 
               time: float):
    '''
    Writes the XMF (XDMF) sidecar file that points ParaView to the HDF5 data.
 
    Preserves the DG element structure by using an explicit 'Quadrilateral'
    topology (one cell per (P, P) sub-quad inside each spectral element)
    rather than a disconnected 'Polyvertex' point cloud. This keeps the
    non-uniform LG/LGL node spacing visible and lets ParaView treat the mesh
    as a proper unstructured grid (contours, slices, streamlines, etc. all
    work correctly across element boundaries).
 
    Connectivity is stored as its own HDF5 dataset ('/Connectivity', shape
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
    Writes a native VTK unstructured-grid file ('.vtu') via meshio.

    Unlike the HDF5 + XDMF path, a '.vtu' file is fully self-contained (geometry,
    connectivity, and field data in one file) and needs no companion sidecar,
    which avoids the XDMF 'filename:/dataset' path-parsing pitfalls (notably the
    Windows drive-letter colon) that make hand-written XDMF fragile in ParaView.

    The DG element structure is preserved by emitting one 'quad' cell per (P, P)
    sub-quad inside each spectral element, using the same per-element
    connectivity built by '_build_quad_connectivity'. Non-uniform LG/LGL node
    spacing is therefore visible and ParaView treats the result as a proper
    unstructured grid.

    ParaView auto-groups files sharing the '<case>_step<NNNNNN>.vtu' naming into
    a time series, so per-step files are sufficient until a '.pvd' collection is
    added alongside the solver's time loop.

    Inputs:
    - all_coords       : (n_nodes, dim) node coordinates (dim in {1, 2, 3})
    - all_solutions    : (n_nodes, n_vars) state-variable values, column order
                          matching 'output_var_names'
    - output_var_names : names of the columns in 'all_solutions'
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
    # element-stacked node ordering used to build 'points'.
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
    if input_config.physics.model == "scalar_advection":
        var_names = ["Phi"]
    else:
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

    # Convert the stored conservative variables into the primitive state variables that get written.
    # Scalar advection has no compressible state (its slot-0 value is the transported scalar Phi, which
    # legitimately goes negative), so deriving density/pressure/temperature for it is both meaningless
    # and would (correctly) trip the positivity guard in compute_pressure. Write the scalar directly.
    if input_config.physics.model == "scalar_advection":
        state_vars = all_sol                          # (num_nodes, 1), matches var_names = ["Phi"]
    else:
        rho         = all_sol[:, 0:1] # must use 0:1 to maintain 2D shape
        pressure    = constRelations.compute_pressure   (all_sol, input_config)
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


def write_pvd_collection(case_name: str, records: list, output_root: Path) -> None:
    '''
    Writes a ParaView '.pvd' collection file that maps each per-step '.vtu' to its true
    SIMULATION TIME. This lets ParaView show real times (0.0, 0.02, ...) on the timeline while
    the '.vtu' files keep clean, sortable '_step<NNNNNN>' names (decimal times in filenames break
    ParaView's automatic time-series parsing, so the time lives in the collection instead).

    Open the resulting '<case_name>.pvd' in ParaView instead of the folder; it also fixes the
    ordering of interval-based output and ignores stale '.vtu' files from previous runs (only the
    files listed here are loaded).

    Inputs:
    - case_name   : the case name (matches the '<case_name>_step<NNNNNN>.vtu' file prefix).
    - records     : list of (time, step) tuples for every frame written so far.
    - output_root : directory containing the 'vtk/' folder; the '.pvd' is written here, and the
                    'file="..."' paths inside it are relative to this location.

    Outputs:
    - None (writes '<output_root>/<case_name>.pvd').
    '''

    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
             '  <Collection>']

    # One <DataSet> per frame: timestep is the real sim time; file is relative to the .pvd.
    for time, step in records:
        lines.append(
            f'    <DataSet timestep="{time:.10g}" file="vtk/{case_name}_step{step:06d}.vtu"/>'
        )

    lines += ['  </Collection>', '</VTKFile>']

    (Path(output_root) / f"{case_name}.pvd").write_text("\n".join(lines) + "\n")