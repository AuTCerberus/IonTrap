import os, json, pathlib
import numpy as np
import meshio
import gmsh
import sys
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from typing import Callable, Dict, Optional, Union
from skfem import MeshTet, Basis, asm, condense, ElementTetP1, solve as fem_solve
from skfem.io.meshio import from_meshio as skfem_from_meshio
from skfem.models.poisson import laplace

# import numba for JIT compilation - makes things faaaast
try:
    from numba import jit, prange, float64, int32, int64
    from numba.typed import List
    NUMBA_AVAILABLE = True
    print("[Performance] Numba JIT compilation enabled")
except ImportError:
    NUMBA_AVAILABLE = False
    print("[Performance] Numba not available. Install with: pip install numba")
    # Dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range
    List = None

#directories
ROOT = pathlib.Path(__file__).resolve().parent
ELECTRODE_DIR = pathlib.Path(os.environ.get("CAD_NUMERIC_ELECTRODES", str(ROOT / "electrodes"))).resolve()
OUTDIR = pathlib.Path(os.environ.get("CAD_NUMERIC_OUT", str(ROOT / "numeric_out"))).resolve()
ELECTRODE_DIR.mkdir(parents=True, exist_ok=True)
OUTDIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CFG = {
    "vacuum_box_half_extent_m": 0.02,   # 2 cm default half-extent of vacuum box
    "lc_surface_m": 5e-5,               #surface mesh target size 
    "lc_volume_m":  2e-4,               #volume mesh target size
    "outer_box_tag": 9999               #tag for outer box surfaces
}

def _load_mesh_config():
    """
    Load mesh overrides from JSON.

    Resolution order:
    1) CAD_NUMERIC_MESH_CONFIG env var (path to json)
    2) ELECTRODE_DIR/mesh_config.json
    3) OUTDIR/mesh_config.json
    """
    cfg = {}
    sources = []
    candidates = []
    env_path = os.environ.get("CAD_NUMERIC_MESH_CONFIG")
    if env_path:
        try:
            candidates.append(pathlib.Path(env_path).expanduser().resolve())
        except Exception:
            candidates.append(pathlib.Path(env_path))
    candidates.append(ELECTRODE_DIR / "mesh_config.json")
    candidates.append(OUTDIR / "mesh_config.json")

    for path in candidates:
        if not path or not path.exists():
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                cfg.update(data)
                sources.append(str(path))
            else:
                print(f"[mesh] Warning: {path} did not contain a JSON object; ignoring.")
        except Exception as exc:
            print(f"[mesh] Warning: Failed to read mesh config {path}: {exc}")

    return cfg, sources

def _infer_unit_scale(points, meta_scale = None):
    """
    Infer unit scale conversion factor (to meters) for mesh coordinates.
    
    Res order:
    1. Environment variable CAD_NUMERIC_UNIT_SCALE
    2. Metadata from previous solve (meta_scale)
    3. Heuristic inference from point magnitudes
    
    Returns:
        tuple: (scale_factor, reason_string)
            scale_factor: float, multiply coordinates by this to get meters
            reason_string: str, explanation of how scale was determined
    """
    # 1. env variable override
    env_scale = os.environ.get("CAD_NUMERIC_UNIT_SCALE")
    if env_scale:
        try:
            val = float(env_scale)
            if val > 0:
                return val, f"env CAD_NUMERIC_UNIT_SCALE={val}"
        except (ValueError, TypeError):
            pass
    
    # 2. metadata from previous solve
    if meta_scale is not None:
        try:
            val = float(meta_scale)
            if val > 0:
                return val, f"facet_names.json (unit_scale_to_m={val})"
        except (ValueError, TypeError):
            pass
    
    # 3. from mesh points magnitueds
    pts = np.asarray(points)
    if pts.size == 0:
        return 1.0, "default (empty mesh)"
    
    max_abs = float(np.max(np.abs(pts)))
    
    # Guard against degenerate cases
    if not np.isfinite(max_abs) or max_abs == 0:
        return 1.0, "default (degenerate extent: all zeros or NaN)"
    
    # Heuristic thresholds
    # Most ion traps are ca. 1 mm to 10 cm, so coords should be ca 0.001 to 0.1 m
    if max_abs > 1e6:
        # Very large numbers - likely nm
        return 1e-9, f"inferred nanometers (max coord ~ {max_abs:.3g})"
    elif max_abs > 1e4:
        # Large numbers - likely um
        return 1e-6, f"inferred micrometers (max coord ~ {max_abs:.3g})"
    elif max_abs > 100:
        # Moderately large - likely mm (try to export CAD at this scale please)
        return 1e-3, f"inferred millimeters (max coord ~ {max_abs:.3g})"
    elif max_abs > 10:
        # assume cm since that is within typical trap size
        return 1e-2, f"inferred centimeters (max coord ~ {max_abs:.3g})"
    elif max_abs > 1:
        # Could be mm interpreted as m, or already in meters but large trap
        #ambiguous - log and default to mm assumption (please export CAD at mm scale)
        print(f"[WARNING] Ambiguous CAD units (max coord ~ {max_abs:.3g})")
        print("[WARNING] Assuming millimeters. Set CAD_NUMERIC_UNIT_SCALE environment variable if incorrect.")
        print("[WARNING] Example: export CAD_NUMERIC_UNIT_SCALE=0.001  # if CAD is in mm")
        return 1e-3, f"ambiguous (max coord ~ {max_abs:.3g}, assuming millimeters)"
    else:
        # Coordinates in [0, 1] -> assume already in meters
        return 1.0, f"assume meters (max coord ~ {max_abs:.3g})"


def mesh_from_cad(config: Optional[dict] = None, simplify_mesh=False):
    file_cfg, file_sources = _load_mesh_config()
    cfg = DEFAULT_CFG | file_cfg | (config or {}) # Merge with defaults
    if file_sources:
        print(f"[mesh] Loaded mesh overrides from: {', '.join(file_sources)}")

    #type casting
    default_half_m = float(cfg["vacuum_box_half_extent_m"])
    lc_s_m = float(cfg["lc_surface_m"])
    lc_v_m = float(cfg["lc_volume_m"])
    if simplify_mesh: #for faster render
        lc_s_m = max(lc_s_m, 2e-4)  # At least 0.2 mm
        lc_v_m = max(lc_v_m, 5e-4)  # At least 0.5 mm
    outer_tag = int(cfg["outer_box_tag"])
    meta_scale = None
    meta_path = OUTDIR / "facet_names.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta_scale = json.load(f).get("unit_scale_to_m")
        except Exception:
            meta_scale = None

    #1. Init Gmsh
    # Avoid signal handling errors when running from a GUI worker thread.
    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # 1=Delaunay, 4=Frontal Todo: thourogh testing cases;
    # Optimize mesh quality - increase iterations for better symmetry
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.3)  #only optimize bad elements
    # Additional smoothing passes for symmetry; at cost of runtime, also only useful if symmetric electrodes
    gmsh.option.setNumber("Mesh.Smoothing", 2)  # 2 smoothing iterations
    # Finer control near curved surfaces
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumCirclePoints", 24)  # More points for smoother circles (20-32; huge time cost)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)  # Better transition
    # better element quality
    gmsh.option.setNumber("Mesh.QualityType", 2)  # 2 = gamma (radius ratio)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)  # Linear elements (can try 2 for quadratic)
    # Improve algorithm consistency
    gmsh.option.setNumber("Mesh.RandomFactor", 1e-9)  # Reduce randomness for repeatability
    gmsh.option.setNumber("General.Terminal", 1) #Set terminal output , 0 for no output
    gmsh.model.add("PaulTrap") #Create new model

    #2. Find files
    cad_files = sorted([p for p in ELECTRODE_DIR.iterdir()
                        if p.suffix.lower() in [".step", ".stp", ".iges", ".igs", ".brep"]])
    if not cad_files:
        print(f"No CAD files found in {ELECTRODE_DIR}. Looking for STLs instead.") #for the love of everything please export CAD with solid geometry, not just surface meshes and dont use STL
        cad_files = sorted([p for p in ELECTRODE_DIR.iterdir()
                            if p.suffix.lower() == ".stl"])
        
    if not cad_files:
        raise FileNotFoundError(f"No CAD or STL files found in {ELECTRODE_DIR}.")

    #3. Import eelectrodes
    electrode_volumes = {}
    for cad in cad_files:
        print(f"[mesh] Importing {cad.name}...")
        try:
            out = gmsh.model.occ.importShapes(str(cad)) #openCASCADE import 

            if not out:
                gmsh.merge(str(cad)) #try general meshing instead; !fallback, does not care about entity information

            gmsh.model.occ.synchronize() #synchronize CAD kernel with Gmsh model
        
            #3D Volume identifiers
            curr_vols = []
            if out:
                curr_vols = [tag for dim, tag in out if dim == 3] #check out for 3D volumes - gmsh only returns (dim, tag) pairs
            else: #fallback - if importShapes empty
                all_vols = [tag for dim, tag in gmsh.model.getEntities(3)] #query all 3D volumes
                assigned_vols = [v for vlist in electrode_volumes.values() for v in vlist] #flatten previously assigned volumes
                curr_vols = [v for v in all_vols if v not in assigned_vols] #exclude previously assigned volumes

            if not curr_vols:
                print(f"[mesh] Warning: {cad.name} did not result in a valid 3D volume. Is it a surface mesh?")
            
            electrode_volumes[cad.stem] = curr_vols
        
        except Exception as e:
            raise ImportError(f"[mesh]Failed to import {cad.name}: {e}")
        
    #although there are fallbacks; most errors I have encountered are from either not solid geometry or from wrong scaling
    # FreeCAD is useful for setting correct scale at export
        
    #4. Create vacuum bounding box
    all_elec_tags = [v for vlist in electrode_volumes.values() for v in vlist]

    if all_elec_tags:
        bboxes = [gmsh.model.getBoundingBox(3, t) for t in all_elec_tags]
        bboxes = np.array(bboxes)
        max_coord = np.max(np.abs(bboxes))

        print(f"[mesh] Electrode bounding box max coordinate: {max_coord:.2f} (CAD units)")
        scale_points = bboxes
    else:
        scale_points = np.asarray([])

    unit_scale, scale_reason = _infer_unit_scale(scale_points, meta_scale) #unit call
    if unit_scale <= 0:
        unit_scale = 1.0
        scale_reason = "invalid scale, default to meters"

    # Convert meter-based config to CAD units
    default_half = default_half_m / unit_scale
    lc_s = lc_s_m / unit_scale
    lc_v = lc_v_m / unit_scale

    if all_elec_tags:
        # Set box size based on CAD units 
        half = max(default_half, max_coord * 1.5)
    else:
        half = default_half

    print(f"[mesh] CAD unit scale: {unit_scale:g} m per unit ({scale_reason})")
    print(f"[mesh] Creating vacuum box with half-extent: {half:.4f} CAD units ({half * unit_scale:.4g} m)")
    box_tag = gmsh.model.occ.addBox(-half, -half, -half, 2*half, 2*half, 2*half) #centered at origin
    gmsh.model.occ.synchronize()

    #5. Boolean difference to create vacuum volume (Vaccum = Box - Electrodes)
    box_dimtag = [(3, box_tag)] #3D volume tag for box
    tool_dimtags = [(3, t) for t in all_elec_tags] #3D volume tags for electrodes

    if tool_dimtags:
        print(f"[mesh] Cutting {len(tool_dimtags)} electrode volumes from vacuum...")
        vacuum_dimtags, _ = gmsh.model.occ.cut(box_dimtag, tool_dimtags, removeObject=True, removeTool=False) #Remove box, keep electrodes
        gmsh.model.occ.synchronize()
    else:
        vacuum_dimtags = box_dimtag

    #6. id surfaces physical groups 
    #Get all surfaces of the final vacuum volume
    vacuum_surfaces = []
    for dim, tag in vacuum_dimtags:
        # Get boundary surfaces of this volume
        bnd = gmsh.model.getBoundary([(dim, tag)], combined=False, oriented=False)
        vacuum_surfaces.extend([t for d, t in bnd if d == 2])
    vacuum_surfaces = set(vacuum_surfaces)

    # find outer box faces
    outer_box_faces = []
    tol = 1e-5
    for surf in vacuum_surfaces:
        b = gmsh.model.getBoundingBox(2, surf)
        # Check if surface touches the outer limits of the box
        touches_limit = (
            abs(b[0] - (-half)) < tol or abs(b[3] - half) < tol or # X
            abs(b[1] - (-half)) < tol or abs(b[4] - half) < tol or # Y
            abs(b[2] - (-half)) < tol or abs(b[5] - half) < tol    # Z
        )
        if touches_limit:
            outer_box_faces.append(surf)

    # Create Outer Physical Group - Box walls
    if outer_box_faces:
        pg_out = gmsh.model.addPhysicalGroup(2, outer_box_faces, tag=outer_tag)
        gmsh.model.setPhysicalName(2, pg_out, "outer_box")
    
    # Create Electrode Physical Groups
    created_electrode_tags = {}
    # Remove outer faces from consideration for electrodes
    internal_surfaces = vacuum_surfaces - set(outer_box_faces)
    
    for name, vol_tags in electrode_volumes.items():
        # Get all surfaces belonging to this electrodes original volumes
        elec_surf_candidates = []
        for v in vol_tags:
            bnd = gmsh.model.getBoundary([(3, v)], combined=False, oriented=False)
            elec_surf_candidates.extend([t for d, t in bnd if d == 2])
        
        # and the vacuums internal surfaces
        interface_surfaces = list(set(elec_surf_candidates) & internal_surfaces)
        
        if interface_surfaces:
            print(f"[mesh] Grouping {len(interface_surfaces)} vacuum-interface surfaces for electrode: {name}")
            pg_tag = gmsh.model.addPhysicalGroup(2, interface_surfaces)
            gmsh.model.setPhysicalName(2, pg_tag, name)
            created_electrode_tags[name] = pg_tag
            
            # Get edges of these surfaces for mesh sizing
            edges = []
            for s in interface_surfaces:
                edge_bnd = gmsh.model.getBoundary([(2, s)], combined=False, oriented=False)
                edges.extend([t for d, t in edge_bnd if d == 1])

            if edges:
                pts = []
                for e in set(edges):
                    pt_bnd = gmsh.model.getBoundary([(1, e)], combined=False, oriented=False)
                    pts.extend([t for d, t in pt_bnd if d == 0])

                if pts:
                    gmsh.model.mesh.setSize([(0, p) for p in set(pts)], lc_s * 0.5) # 2x finer near electrodes
        
        else:
            # No vacuum boundary found - this might be an internal electrode
            # Get all surfaces of the electrode volume that exist in the final geometry
            all_surfaces = gmsh.model.getEntities(2)
            all_surface_tags = [t for d, t in all_surfaces]
            
            # Find which original electrode surfaces still exist
            existing_elec_surfaces = [s for s in elec_surf_candidates if s in all_surface_tags]
            
            if existing_elec_surfaces:
                print(f"[mesh] Grouping {len(existing_elec_surfaces)} internal surfaces for electrode: {name}")
                pg_tag = gmsh.model.addPhysicalGroup(2, existing_elec_surfaces)
                gmsh.model.setPhysicalName(2, pg_tag, name)
                created_electrode_tags[name] = pg_tag
            else:
                print(f"[mesh] Warning: No surfaces found for electrode '{name}'. ")
        
    #7. define volume physical group for vacuum
    vac_vol_tags = [t for d, t in vacuum_dimtags]
    if vac_vol_tags:
        pg_vol = gmsh.model.addPhysicalGroup(3, vac_vol_tags)
        gmsh.model.setPhysicalName(3, pg_vol, "vacuum")

    #8. Mesh gen - only vacuum field region
    gmsh.model.occ.remove(tool_dimtags, recursive=True) #remove electrode volumes so only mesh of vacuum remains
    gmsh.model.occ.synchronize()

    target_segments = float(cfg.get("target_segments", 20.0)) #subdivisions per edge of box
    lc_s_eff = (2*half) / target_segments
    
    lc_min = min(lc_v, lc_s_eff)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_s_eff)
    

    # Box field - fine mesh in the trapping region
    box_field = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(box_field, "VIn", lc_s * 0.5)   # Fine inside
    gmsh.model.mesh.field.setNumber(box_field, "VOut", lc_v)        # Coarse outside
    gmsh.model.mesh.field.setNumber(box_field, "XMin", -0.005)      # 5mm box
    gmsh.model.mesh.field.setNumber(box_field, "XMax", 0.005)
    gmsh.model.mesh.field.setNumber(box_field, "YMin", -0.005)
    gmsh.model.mesh.field.setNumber(box_field, "YMax", 0.005)
    gmsh.model.mesh.field.setNumber(box_field, "ZMin", -0.001)      # From electrode surface
    gmsh.model.mesh.field.setNumber(box_field, "ZMax", 0.003)       # To 3mm above
    gmsh.model.mesh.generate(3)

    # Distance field - refine near electrode surfaces
    dist_field = gmsh.model.mesh.field.add("Distance")
    # Get all electrode surface tags
    elec_surfaces = []
    for name, surfs in created_electrode_tags.items():
        elec_surfaces.extend(surfs if isinstance(surfs, list) else [surfs])
    gmsh.model.mesh.field.setNumbers(dist_field, "FacesList", elec_surfaces)

    # Threshold field - interpolate mesh size based on distance
    thresh_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thresh_field, "IField", dist_field)
    gmsh.model.mesh.field.setNumber(thresh_field, "LcMin", lc_s * 0.5)  # Fine near electrodes
    gmsh.model.mesh.field.setNumber(thresh_field, "LcMax", lc_v)         # Coarse far away
    gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", 0.0001)     # Start refining at 0.1mm
    gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", 0.005)      # Full coarsening at 5mm

    # Minimum field - use smallest of all fields
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [box_field, thresh_field])

    # Apply the field
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    #9. write
    msh_path = OUTDIR / "mesh.msh"
    gmsh.write(str(msh_path))
    gmsh.finalize()

    #10. Metadata: electrode tags, outer box tag, box size export to json
    meta = {
        "electrodes": {k: v for k, v in created_electrode_tags.items()},
        "outer_box_tag": outer_tag,
        "vacuum_box_half_extent_m": half * unit_scale,
        "unit_scale_to_m": unit_scale,
        "unit_scale_reason": scale_reason
    }
    with open(OUTDIR / "facet_names.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[mesh] Success. Output at {msh_path}")
    return msh_path

#--------------------------------------------#
#Solving field phi_i with scikitfem and precompute per element delta phi_i

def _prepare_mesh(m: meshio.Mesh): #convert meshio mesh to skfem 
    if "tetra" not in m.cells_dict or "triangle" not in m.cells_dict: #test membership 
        kinds = ", ".join(m.cells_dict.keys())
        raise ValueError(
            "Mesh must contain tetra (3D) and triangle (surface) cells."
            f"Found: {kinds or 'none'}.\n"
            "Check that meshing created a 3D volume (vacuum) and that boundary triangles exist.")
    return skfem_from_meshio(m)

def solve_basis_fields(config: Optional[dict] = None):

    cfg = DEFAULT_CFG | (config or {})
    msh_path = OUTDIR / "mesh.msh"
    if not msh_path.exists():
        raise FileNotFoundError("mesh.msh not found. Run mesh_from_cad() first.")

    with open(OUTDIR / "facet_names.json", "r") as f:
        meta = json.load(f)

    mesh = meshio.read(str(msh_path))

    # Detect/record the unit scale so evaluation can convert to meters later
    meta_scale = meta.get("unit_scale_to_m")
    unit_scale, scale_reason = _infer_unit_scale(mesh.points, meta_scale) #get scale
    if unit_scale != 1.0: #report if not unity
        print(f"[solve] Using unit scale {unit_scale:g} ({scale_reason})")
    if meta.get("unit_scale_to_m") != unit_scale or meta.get("unit_scale_reason") != scale_reason: #update if changed
        try:
            meta["unit_scale_to_m"] = unit_scale
            meta["unit_scale_reason"] = scale_reason
            with open(OUTDIR / "facet_names.json", "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as ex:
            print(f"[solve] Warning: failed to update unit scale in facet_names.json: {ex}")
    
    # Debug: Print found physical groups
    print("[solve] Mesh physical groups found (cell_data):", mesh.cell_data_dict.get("gmsh:physical", {}).keys())

    mtet = _prepare_mesh(mesh)

    # Build map of physical tag - facet indices
    boundary_facets = {}
    if mtet.boundaries:
        for name, idxs in mtet.boundaries.items():
            boundary_facets[name] = idxs

    #id electrodes
    electrode_facets = {}
    meta_electrodes = meta.get("electrodes", {})

    for name, tag in meta_electrodes.items():
        #skfem usually loads boundaries by Name if gmsh provided them
        if name in boundary_facets:
            electrode_facets[name] = boundary_facets[name]
        #Sometimes it loads by tag string
        elif str(tag) in boundary_facets:
            electrode_facets[name] = boundary_facets[str(tag)]
        else:
            print(f"[solve] Warning: Could not find facets for electrode '{name}' (tag {tag}) in mesh.")

    if not electrode_facets:
         raise RuntimeError("No electrode boundaries found in mesh. Meshing step likely failed to tag surfaces correctly.")
    outer_tag = meta["outer_box_tag"]
    outer_dofs_indices = None

    # Find outer box
    if "outer_box" in boundary_facets:
        outer_dofs_indices = boundary_facets["outer_box"]
    elif str(outer_tag) in boundary_facets:
        outer_dofs_indices = boundary_facets[str(outer_tag)]
    
    if outer_dofs_indices is None:
        # Fallback: find any boundary that isnt an electrode
        all_elec_indices = np.concatenate(list(electrode_facets.values()))
        # risky
        print("[solve] Warning: 'outer_box' group missing. Assuming Dirichlet 0 at infinity (not ideal).")
        # In this specific FEM formulation, nodes not constrained are Neumann (E_perp=0), 
        # but we need Ground at infinity. 
        # condense will handle unlisted DOFs as free.
        outer_dofs = np.array([], dtype=int)
    else:
        basis_temp = Basis(mtet, ElementTetP1())
        outer_dofs = basis_temp.get_dofs(facets=outer_dofs_indices).all()

    V = Basis(mtet, ElementTetP1())
    A = asm(laplace, V) #finite element assembly
    rhs = np.zeros(V.N) #right-hand side zero for Laplace

    # skfem linear tet: gradients are constant per element.
    # construct a matrix G mapping nodal values (4 per tet) to gradient (3 components).
    # Shape of coords P: (3, n_nodes)
    # Shape of connectivity T: (4, n_elems)
    P = mtet.p
    T = mtet.t 

    # Vectorized Jacobian calc
    # nodes a,b,c,d for all elements
    # P[:, T[0, :]] is (3, n_elems) - node A coords
    nA = P[:, T[0, :]]
    nB = P[:, T[1, :]]
    nC = P[:, T[2, :]]
    nD = P[:, T[3, :]]
    
    # Edges (3, n_elems)
    v1 = nB - nA
    v2 = nC - nA
    v3 = nD - nA
    #jacobain
    J = np.array([v1, v2, v3]).transpose(1, 0, 2)

    #invert jacobian -inv(J) = 1/det(J) * adj(J)
    # Note: J has shape (3, 3, n_elems), so need to slice the last dimension
    detJ = (J[0,0,:]*J[1,1,:]*J[2,2,:] + J[0,1,:]*J[1,2,:]*J[2,0,:] + J[0,2,:]*J[1,0,:]*J[2,1,:]
          - J[0,2,:]*J[1,1,:]*J[2,0,:] - J[0,1,:]*J[1,0,:]*J[2,2,:] - J[0,0,:]*J[1,2,:]*J[2,1,:])
    
    # no div by zero
    detJ[detJ == 0] = 1e-12
    invDetJ = 1.0 / detJ

    #Compute adjoint - manually is faster
    # J_inv shape: (3, 3, n_elems)
    J_inv = np.zeros_like(J)
    J_inv[0,0,:] = (J[1,1,:]*J[2,2,:] - J[1,2,:]*J[2,1,:]) * invDetJ
    J_inv[0,1,:] = (J[0,2,:]*J[2,1,:] - J[0,1,:]*J[2,2,:]) * invDetJ
    J_inv[0,2,:] = (J[0,1,:]*J[1,2,:] - J[0,2,:]*J[1,1,:]) * invDetJ
    J_inv[1,0,:] = (J[1,2,:]*J[2,0,:] - J[1,0,:]*J[2,2,:]) * invDetJ
    J_inv[1,1,:] = (J[0,0,:]*J[2,2,:] - J[0,2,:]*J[2,0,:]) * invDetJ
    J_inv[1,2,:] = (J[0,2,:]*J[1,0,:] - J[0,0,:]*J[1,2,:]) * invDetJ
    J_inv[2,0,:] = (J[1,0,:]*J[2,1,:] - J[1,1,:]*J[2,0,:]) * invDetJ
    J_inv[2,1,:] = (J[0,1,:]*J[2,0,:] - J[0,0,:]*J[2,1,:]) * invDetJ
    J_inv[2,2,:] = (J[0,0,:]*J[1,1,:] - J[0,1,:]*J[1,0,:]) * invDetJ

    J_inv_T = J_inv.transpose(1, 0, 2)

    # Shapes: (3,3,Ne) @ (3,) -> (3,Ne)
    # result (Ne, 4, 3) : for each element, for each of 4 nodes, 3 grad components
    n_elems = T.shape[1]
    GRADS_PER_ELEM = np.zeros((n_elems, 4, 3))
    
    ref_grads = np.array([[-1.,-1.,-1.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])

    for i in range(4):
        # g_ref is (3,)
        g_ref = ref_grads[i]
        # (3,3,Ne) * (3,1) -> (3,Ne)
        # manually do matmul(np @) broadcast (not enough elem for @)
        res = np.einsum('ijk,j->ik', J_inv_T, g_ref)
        GRADS_PER_ELEM[:, i, :] = res.T #per element gradient for node i
        
    centroids = P[:, T].mean(axis=1).T
    np.save(OUTDIR / "centroids.npy", centroids)
    np.savez(OUTDIR / "mesh_topology.npz", nodes=P.T, tets=T.T)

    # Solve per electrode
    for name, facets in electrode_facets.items():
        # Get DOFs for this electrode
        e_dofs = V.get_dofs(facets=facets).all()
        
        # Constraints: This electrode = 1.0, Outer Box = 0.0, Other electrodes = 0.0
        # Collect all other Dirichlet DOFs
        other_elec_dofs = []
        for other_name, other_facets in electrode_facets.items():
            if other_name != name:
                other_elec_dofs.append(V.get_dofs(facets=other_facets).all())
        
        if len(other_elec_dofs) > 0:
            all_zero_dofs = np.concatenate([outer_dofs] + other_elec_dofs)
        else:
            all_zero_dofs = outer_dofs
            
        all_zero_dofs = np.unique(all_zero_dofs)
        
        # Full set of Dirichlet DOFs
        D_idx = np.unique(np.concatenate([e_dofs, all_zero_dofs]))
        
        # Set values
        x = np.zeros(V.N)
        x[e_dofs] = 1.0
        x[all_zero_dofs] = 0.0
        
        # Condense and solve
        A_c, b_c, x_c, I = condense(A, rhs, x=x, D=D_idx, expand=True)
        x_sol = spsolve(A_c, b_c)
        x_c[I] = x_sol
        
        # Compute gradient field
        # uh is nodal values (N_nodes,)
        # grads = uh[T] (Ne, 4) ... GRADS (Ne, 4, 3)
        # dot product over the 4 nodes
        uh = x_c
        grads = np.einsum('ni,nij->nj', uh[T.T], GRADS_PER_ELEM, optimize=True)
        
        np.savez(OUTDIR / f"basis__{name}.npz", phi=uh, grads=grads)
        print(f"[solve] Solved basis for {name}")

    print("[solve] All bases solved.")

'''--------------------------------------------'''
#Runtime eval
class NumericFieldSKFEM:
    """Loads mesh + precomputed per-electrode {phi, grads_per_element}.
    Evaluates (phi, E) at arbitrary points in the vacuum region.
    E = -sum V_i(t) * delta phi_i  (constant inside each tetra)"""

    def __init__(self, basis_dir: Union[str, pathlib.Path] = OUTDIR):
        basis_dir = pathlib.Path(basis_dir)
        topo = np.load(basis_dir / "mesh_topology.npz")
        nodes = topo["nodes"]              # (n_nodes, 3) 
        self.tets  = topo["tets"]               # (n_elem, 4)
        centroids = np.load(basis_dir / "centroids.npy")  # (n_elem, 3)

        # Determine coordinate scale (CAD units to meters)
        meta = {}
        meta_scale = None
        meta_path = basis_dir / "facet_names.json"
        if meta_path.exists():
            try:
                meta = json.load(open(meta_path, "r"))
                meta_scale = meta.get("unit_scale_to_m")
            except Exception:
                meta = {}
        unit_scale, scale_reason = _infer_unit_scale(nodes, meta_scale)
        if unit_scale != 1.0:
            nodes = nodes * unit_scale
            centroids = centroids * unit_scale
            print(f"[numeric] Applied unit scale {unit_scale:g} ({scale_reason})")
        else:
            print(f"[numeric] Using unit scale {unit_scale:g} ({scale_reason})")

        self.unit_scale_to_m = unit_scale
        self.unit_scale_reason = scale_reason
        self.nodes = nodes
        self.centroids = centroids
        self.tree = cKDTree(self.centroids) #tree ti scan for nearest tet 

        # load bases
        self.bases = {}         # name -> dict(phi, grads)
        for f in basis_dir.glob("basis__*.npz"): #scan fior per electrode basis files
            name = f.stem.split("__", 1)[1] #extract electrode name
            dat = np.load(f)
            grads = dat["grads"]
            if grads.ndim == 3 and grads.shape[0] == grads.shape[1]:
                # Legacy files produced (n_elem, n_elem, 3); take diagonal slice
                idx = np.arange(grads.shape[0])
                grads = grads[idx, idx]
            if unit_scale != 1.0:
                grads = grads / unit_scale  # convert from CAD-unit gradients to V/m
            self.bases[name] = {"phi": dat["phi"], "grads": grads}
        if not self.bases:
            raise RuntimeError("No bases found. Run solve_basis_fields() first.")
        self.electrodes = sorted(self.bases.keys())

    # barycentric membership test + weights
    def _barycentric(self, p, tet_nodes):
        a, b, c, d = self.nodes[tet_nodes]  # 4x3 tetra 
        v0 = b - a; v1 = c - a; v2 = d - a; vp = p - a
        M = np.column_stack((v0, v1, v2)) #tetra edge matrix
        try:
            w = np.linalg.solve(M, vp)  # w0,w1,w2 corresponding to b,c,d
        except np.linalg.LinAlgError:
            return False, None
        l1, l2, l3 = w
        l0 = 1.0 - l1 - l2 - l3
        inside = (l0 >= -1e-9) and (l1 >= -1e-9) and (l2 >= -1e-9) and (l3 >= -1e-9)
        return inside, np.array([l0, l1, l2, l3])

    def _find_element(self, p, k_neighbors=24):
        # Try a handful of nearest centroids, check barycentrics
        dists, idxs = self.tree.query(p, k=min(k_neighbors, len(self.tets)))
        if np.isscalar(idxs):
            idxs = [int(idxs)]
        for ei in np.atleast_1d(idxs):
            nodes = self.tets[ei]
            inside, _ = self._barycentric(p, nodes)
            if inside:
                return int(ei)
        return None
    
    def evaluate(self, points_xyz: np.ndarray, voltages: dict[str, float]):

        #ensure 2D array and init phi and E
        pts = np.atleast_2d(points_xyz).astype(np.float64) 
        N = pts.shape[0]
        phi = np.zeros(N, dtype=np.float64)
        E = np.zeros((N, 3), dtype=np.float64)
        
        # Precompute combined fields
        names = [k for k in self.electrodes if abs(voltages.get(k, 0.0)) > 0.0]
        if not names:
            return phi, E
        
        G_elem = np.zeros((self.tets.shape[0], 3), dtype=np.float64) #gradients per element = grad(phi)
        PHI_nodes = np.zeros(self.nodes.shape[0], dtype=np.float64) #potential per node
        
        for name in names:
            V_i = float(voltages.get(name, 0.0))
            if V_i == 0.0:
                continue
            G_elem += V_i * self.bases[name]["grads"]
            PHI_nodes += V_i * self.bases[name]["phi"]
        
        #Batch find elements
        element_ids = self._find_elements_batch(pts)
        
        # Evaluate fields for all valid points
        valid_mask = element_ids >= 0
        valid_elements = element_ids[valid_mask]
        
        # Electric field (constant per element)
        E[valid_mask] = -G_elem[valid_elements] # E = -grad(phi)
        
        # Potential (requires barycentric interpolation)
        for i in np.where(valid_mask)[0]:
            ei = element_ids[i]
            nodes = self.tets[ei]
            _, w = self._barycentric(pts[i], nodes)
            phi[i] = np.dot(PHI_nodes[nodes], w)
        
        # Mark invalid points
        phi[~valid_mask] = np.nan #bitwise not
        E[~valid_mask] = np.nan
        
        return phi, E


    # Convenience for time programs
    @staticmethod #  does not receive an implicit first argument
    def voltages_at_time(t, program: Dict[str, Union[float, Callable[[float], float]]]):
        out = {}
        for k, f in program.items():
            out[k] = float(f(t)) if callable(f) else float(f)
        return out
    
    def _find_elements_batch(self, points, k_neighbors=24):
        #Find containing elements for multiple points at once
        points = np.asarray(points, dtype=np.float64)
        points = np.atleast_2d(points)
        if points.shape[1] != 3 and points.shape[0] == 3:
            points = points.T
        if points.shape[1] != 3:
            raise ValueError(f"points must be (N, 3), got {points.shape}")
        N = points.shape[0]
        element_ids = np.full(N, -1, dtype=int)
        if N == 0:
            return element_ids
        
        # Query k nearest centroids for all points at once
        dists, neighbor_idxs = self.tree.query(
            points, 
            k=min(k_neighbors, len(self.tets)),
            workers=-1  # Use all CPU cores -speedbuff
        )

        neighbor_idxs = np.asarray(neighbor_idxs)
        if neighbor_idxs.ndim == 0:
            neighbor_idxs = neighbor_idxs.reshape(1, 1)
        elif neighbor_idxs.ndim == 1:
            neighbor_idxs = neighbor_idxs.reshape(N, -1)
        k = neighbor_idxs.shape[1]

        # Vectorized barycentric check for all (point, neighbor) pairs.
        tet_nodes = self.tets[neighbor_idxs]  # (N, k, 4)
        a = self.nodes[tet_nodes[..., 0]]
        b = self.nodes[tet_nodes[..., 1]]
        c = self.nodes[tet_nodes[..., 2]]
        d = self.nodes[tet_nodes[..., 3]]

        v0 = b - a
        v1 = c - a
        v2 = d - a
        vp = points[:, None, :] - a

        M = np.stack((v0, v1, v2), axis=-1)  # (N, k, 3, 3)
        M_flat = M.reshape(-1, 3, 3)
        vp_flat = vp.reshape(-1, 3)
        det = np.linalg.det(M_flat)
        valid = np.abs(det) > 1e-15

        w_flat = np.zeros_like(vp_flat)
        if np.any(valid):
            w_flat[valid] = np.linalg.solve(
                M_flat[valid], 
                vp_flat[valid, :, np.newaxis]
            ).squeeze(-1)
        w = w_flat.reshape(N, k, 3)
        valid = valid.reshape(N, k)

        l0 = 1.0 - w.sum(axis=-1)
        inside = valid & (l0 >= -1e-9) & (w >= -1e-9).all(axis=-1)

        has_inside = inside.any(axis=1)
        first = inside.argmax(axis=1)
        element_ids = np.where(
            has_inside,
            neighbor_idxs[np.arange(N), first],
            -1,
        ).astype(int)
    
        return element_ids


class OptimizedNumericFieldSKFEM(NumericFieldSKFEM):
    """Optimized version with spatial grid for O(1) point location"""
    
    def __init__(self, basis_dir: Union[str, pathlib.Path] = OUTDIR, grid_resolution: int = 50):
        super().__init__(basis_dir)
        self._build_spatial_grid(grid_resolution)
    
    def _build_spatial_grid(self, resolution: int):
        #spatial lookup grid for O(1) point location
        print(f"[Optimized] Building {resolution}^3 spatial grid...")
        
        # Get bounding box of mesh
        self.grid_min = self.nodes.min(axis=0) - 1e-6
        self.grid_max = self.nodes.max(axis=0) + 1e-6
        self.grid_size = self.grid_max - self.grid_min
        
        # Create grid
        self.grid_res = resolution
        self.grid_cell_size = self.grid_size / resolution
        
        # Initialize empty grid
        self.spatial_grid = [[] for _ in range(resolution**3)]
        
        # Assign each tetrahedron to grid cells it touches
        if NUMBA_AVAILABLE:
            # Numba-accelerated grid building
            try:
                self._build_grid_numba()
            except Exception as exc:
                print(f"[Optimized] Numba grid build failed; falling back to NumPy: {exc}")
                self._build_grid_numpy()
        else:
            # Fallback to numpy
            self._build_grid_numpy()
        
        # Convert to arrays for faster access
        self.grid_indices = []
        self.grid_data = []
        for cell_idx, tets in enumerate(self.spatial_grid):
            if tets:
                self.grid_indices.append(cell_idx)
                self.grid_data.append(np.array(tets, dtype=np.int32))
        
        print(f"[Optimized] Spatial grid built with {len(self.grid_data)} non-empty cells")
    
    def _build_grid_numba(self):
        @jit(nopython=True)
        def build_grid_numba(tets, nodes, grid_min, cell_size, res):
            grid = List()
            for _ in range(res * res * res):
                grid.append(List.empty_list(int64))
            
            for tet_idx in range(len(tets)):
                tet = tets[tet_idx]
                # Get bounding box of tetrahedron
                tet_verts = nodes[tet]
                min_coords = np.array([
                    tet_verts[:, 0].min(), 
                    tet_verts[:, 1].min(), 
                    tet_verts[:, 2].min()
                ])
                max_coords = np.array([
                    tet_verts[:, 0].max(), 
                    tet_verts[:, 1].max(), 
                    tet_verts[:, 2].max()
                ])
                
                # Convert to grid coordinates
                min_cell = np.floor((min_coords - grid_min) / cell_size).astype(np.int32)
                max_cell = np.floor((max_coords - grid_min) / cell_size).astype(np.int32)
                
                # Clamp to grid bounds
                min_cell = np.maximum(min_cell, 0)
                max_cell = np.minimum(max_cell, res - 1)
                
                # Add tet to all cells in its bounding box
                for i in range(min_cell[0], max_cell[0] + 1):
                    for j in range(min_cell[1], max_cell[1] + 1):
                        for k in range(min_cell[2], max_cell[2] + 1):
                            cell_idx = i * res * res + j * res + k
                            grid[cell_idx].append(tet_idx)
            
            return grid
        
        self.spatial_grid = build_grid_numba(
            self.tets, self.nodes, self.grid_min, 
            self.grid_cell_size, self.grid_res
        )
    
    def _build_grid_numpy(self):
        #NumPy-based grid building if shit hits the fan
        for tet_idx, tet in enumerate(self.tets):
            tet_verts = self.nodes[tet]
            min_coords = tet_verts.min(axis=0)
            max_coords = tet_verts.max(axis=0)
            
            min_cell = np.floor((min_coords - self.grid_min) / self.grid_cell_size).astype(int)
            max_cell = np.floor((max_coords - self.grid_min) / self.grid_cell_size).astype(int)
            
            # Clamp
            min_cell = np.maximum(min_cell, 0)
            max_cell = np.minimum(max_cell, self.grid_res - 1)
            
            for i in range(min_cell[0], max_cell[0] + 1):
                for j in range(min_cell[1], max_cell[1] + 1):
                    for k in range(min_cell[2], max_cell[2] + 1):
                        cell_idx = i * self.grid_res * self.grid_res + j * self.grid_res + k
                        self.spatial_grid[cell_idx].append(tet_idx)
    
    def _find_element_fast(self, point):
        #O(1) point location using spatial grid"
        # Find grid cell
        cell_coords = np.floor((point - self.grid_min) / self.grid_cell_size).astype(int)
        
        if np.any(cell_coords < 0) or np.any(cell_coords >= self.grid_res):
            return -1
        
        cell_idx = (cell_coords[0] * self.grid_res * self.grid_res + 
                   cell_coords[1] * self.grid_res + cell_coords[2])
        
        # Check if cell has tets
        if cell_idx >= len(self.spatial_grid) or not self.spatial_grid[cell_idx]:
            return -1
        
        # Check tets in this cell
        for tet_idx in self.spatial_grid[cell_idx]:
            nodes = self.tets[tet_idx]
            inside, _ = self._barycentric(point, nodes)
            if inside:
                return tet_idx
        
        return -1
    
    def _find_elements_batch_fast(self, points):
        #Vectorized O(1) point location using spatial grid
        N = points.shape[0]
        element_ids = np.full(N, -1, dtype=int)
        
        # Find grid cells for all points
        cell_coords = np.floor((points - self.grid_min) / self.grid_cell_size).astype(int)
        
        # Vectorized barycentric test for each cell
        for cell_idx in range(len(self.spatial_grid)):
            if not self.spatial_grid[cell_idx]:
                continue
            
            # Find points in this cell
            cell_x = cell_idx // (self.grid_res * self.grid_res)
            cell_y = (cell_idx // self.grid_res) % self.grid_res
            cell_z = cell_idx % self.grid_res
            
            in_cell_mask = (
                (cell_coords[:, 0] == cell_x) &
                (cell_coords[:, 1] == cell_y) &
                (cell_coords[:, 2] == cell_z)
            )
            
            point_indices = np.where(in_cell_mask)[0]
            if len(point_indices) == 0:
                continue
            
            # Check each tet in cell for each point
            for tet_idx in self.spatial_grid[cell_idx]:
                if len(point_indices) == 0:
                    break
                
                nodes = self.tets[tet_idx]
                tet_nodes_coords = self.nodes[nodes]
                
                # Vectorized barycentric test for all points at once
                inside_mask = self._barycentric_batch(points[point_indices], tet_nodes_coords)
                
                # Assign found elements
                found_indices = point_indices[inside_mask]
                element_ids[found_indices] = tet_idx
                
                # Remove found points from search
                point_indices = point_indices[~inside_mask]
        
        return element_ids
    
    def _barycentric_batch(self, points, tet_nodes):
        """Vectorized barycentric test for multiple points
        # tet_nodes: (4, 3) array of tetrahedron vertices
        # points: (N, 3) array of points to test"""
        
        # Use vectorized linear algebra
        M = tet_nodes[1:] - tet_nodes[0]  # (3, 3)
        M_inv = np.linalg.inv(M.T)  # (3, 3)
        
        # Compute barycentric coordinates for all points
        vp = points - tet_nodes[0]  # (N, 3)
        w = vp @ M_inv.T  # (N, 3)
        
        l0 = 1.0 - w.sum(axis=1)
        l1, l2, l3 = w[:, 0], w[:, 1], w[:, 2]
        
        # Check if all barycentric coordinates are >= -epsilon
        inside = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        
        return inside
    
    def evaluate_fast(self, points_xyz: np.ndarray, voltages: dict[str, float]):
        #Optimized evaluation using spatial grid
        pts = np.atleast_2d(points_xyz).astype(np.float64)
        N = pts.shape[0]
        
        # Skip if no valid voltages
        names = [k for k in self.electrodes if abs(voltages.get(k, 0.0)) > 0.0]
        if not names:
            return np.zeros(N), np.zeros((N, 3))
        
        # Precompute combined fields
        G_elem = np.zeros((self.tets.shape[0], 3), dtype=np.float64)
        PHI_nodes = np.zeros(self.nodes.shape[0], dtype=np.float64)
        
        for name in names:
            V_i = float(voltages.get(name, 0.0))
            G_elem += V_i * self.bases[name]["grads"]
            PHI_nodes += V_i * self.bases[name]["phi"]
        
        # Find elements using spatial grid
        element_ids = self._find_elements_batch_fast(pts)
        
        # Prepare output
        phi = np.full(N, np.nan, dtype=np.float64)
        E = np.full((N, 3), np.nan, dtype=np.float64)
        
        valid_mask = element_ids >= 0
        valid_elements = element_ids[valid_mask]
        
        # Electric field (constant per element)
        if np.any(valid_mask):
            E[valid_mask] = -G_elem[valid_elements]
        
        # Potential via barycentric interpolation
        if np.any(valid_mask):
            valid_pts = pts[valid_mask]
            valid_tets = self.tets[valid_elements]
            
            # Precompute all tetrahedron transformations
            tet_nodes_coords = self.nodes[valid_tets]  # (M, 4, 3)
            
            # Vectorized barycentric calculation
            M = tet_nodes_coords[:, 1:] - tet_nodes_coords[:, 0:1]  # (M, 3, 3)
            
            try:
                # Vectorized inversion
                M_inv = np.linalg.inv(M)
                
                # Compute barycentric coordinates for all points
                vp = valid_pts - tet_nodes_coords[:, 0, :]  # (M, 3)
                w = np.einsum('mij,mj->mi', M_inv, vp)  # (M, 3)
                
                l0 = 1.0 - w.sum(axis=1)
                l123 = w
                
                # Get nodal values for each tetrahedron
                phi_vals = PHI_nodes[valid_tets]  # (M, 4)
                
                # Interpolate
                phi_interp = (
                    l0[:, np.newaxis] * phi_vals[:, 0:1] +
                    l123[:, 0:1] * phi_vals[:, 1:2] +
                    l123[:, 1:2] * phi_vals[:, 2:3] +
                    l123[:, 2:3] * phi_vals[:, 3:4]
                )
                
                phi[valid_mask] = phi_interp[:, 0]
                
            except np.linalg.LinAlgError:
                # Fallback for singular matrices
                for i, (pt, tet) in enumerate(zip(valid_pts, valid_tets)):
                    nodes = self.tets[tet]
                    _, w = self._barycentric(pt, nodes)
                    phi[valid_mask][i] = np.dot(PHI_nodes[nodes], w)
        
        return phi, E


# Numba-optimized helper functions
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def compute_barycentric_coords_numba(points, tet_nodes):
        #Numba-accelerated barycentric coordinate computation

        N = points.shape[0]
        coords = np.zeros((N, 4))
        inside = np.zeros(N, dtype=np.bool_)
        
        for i in prange(N):
            a = tet_nodes[0]
            b = tet_nodes[1]
            c = tet_nodes[2]
            d = tet_nodes[3]
            
            v0 = b - a
            v1 = c - a
            v2 = d - a
            vp = points[i] - a
            
            # Solve linear system
            det = (v0[0]*(v1[1]*v2[2] - v1[2]*v2[1]) -
                   v0[1]*(v1[0]*v2[2] - v1[2]*v2[0]) +
                   v0[2]*(v1[0]*v2[1] - v1[1]*v2[0]))
            
            if abs(det) < 1e-15:
                inside[i] = False
                continue
            
            inv_det = 1.0 / det
            
            # Solve using Cramers rule
            w1 = (vp[0]*(v1[1]*v2[2] - v1[2]*v2[1]) -
                  vp[1]*(v1[0]*v2[2] - v1[2]*v2[0]) +
                  vp[2]*(v1[0]*v2[1] - v1[1]*v2[0])) * inv_det
            
            w2 = (v0[0]*(vp[1]*v2[2] - vp[2]*v2[1]) -
                  v0[1]*(vp[0]*v2[2] - vp[2]*v2[0]) +
                  v0[2]*(vp[0]*v2[1] - vp[1]*v2[0])) * inv_det
            
            w3 = (v0[0]*(v1[1]*vp[2] - v1[2]*vp[1]) -
                  v0[1]*(v1[0]*vp[2] - v1[2]*vp[0]) +
                  v0[2]*(v1[0]*vp[1] - v1[1]*vp[0])) * inv_det
            
            w0 = 1.0 - w1 - w2 - w3
            
            coords[i] = np.array([w0, w1, w2, w3])
            inside[i] = (w0 >= -1e-9) and (w1 >= -1e-9) and (w2 >= -1e-9) and (w3 >= -1e-9)
        
        return inside, coords

    
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "mesh":
        mesh_from_cad()
    elif cmd == "solve":
        solve_basis_fields()
    else:
        print("Usage: python cad_numeric.py [mesh|solve]")
