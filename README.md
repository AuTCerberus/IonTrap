# Paul Trap Simulator

Simulate charged particles (microspheres / ions / electrons) in RF Paul traps using:

- **Analytic fields** (idealized quadrupole / linear trap)
- **Numeric fields from CAD electrodes** (Gmsh tetra mesh + scikit-fem basis potentials)

The main entrypoint is a **Tkinter GUI**: `trap_gui.py`.

## Features

- GUI workflow to generate numeric fields from CAD (`Mesh` -> `Solve`)
- Single-particle and particle-cloud trajectory simulation
- Optional performance acceleration via **Numba** (auto-detected)
- Optional enhanced 3D visualization via **PyVista** / **pyvistaqt** (auto-detected)
- Parameter sweep helpers
- Optional particle beam model (phase-locked to RF)

## Install

### 1) Create a virtual environment

Windows (PowerShell):

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
```

### 2) Install dependencies

```powershell
.\.venv\Scripts\pip install -r requirements.txt
```

Notes:
- `gmsh` is used for CAD -> tetra meshing.
- `PyQt5` is only needed for the optional PyVista/Qt 3D viewer (the GUI itself uses Tkinter).

## Quick start (GUI)

Run:

```powershell
.\.venv\Scripts\python trap_gui.py
```

Recommended first run:
1. In the GUI, set **Electrodes dir** to a folder that contains CAD files directly (not nested). This repo ships examples:
   - `electrodes\three`
   - `electrodes\surface`
2. Set **Output dir** to a matching folder, e.g.:
   - `numeric_out\three`
   - `numeric_out\surface`
3. Click **Mesh** (generates `mesh.msh`).
4. Click **Solve** (generates per-electrode basis files like `basis__*.npz`).
5. Enable **Use numeric field** in the simulation settings (if you want to simulate the CAD-based trap), then run a trajectory / cloud simulation.

The GUI also supports exporting/importing settings JSON via the **Export Settings** / **Import Settings** buttons.

## Numeric pipeline (CLI)

The numeric field pipeline lives in `cad_numeric.py`:

```powershell
# From the repo root
.\.venv\Scripts\python cad_numeric.py mesh
.\.venv\Scripts\python cad_numeric.py solve
```

Directories are controlled by environment variables (same ones used by the GUI):

- `CAD_NUMERIC_ELECTRODES` (default: `.\electrodes`)
- `CAD_NUMERIC_OUT` (default: `.\numeric_out`)
- `CAD_NUMERIC_MESH_CONFIG` (optional path to `mesh_config.json`)
- `CAD_NUMERIC_UNIT_SCALE` (optional coordinate scale factor to meters)

Example:

```powershell
$env:CAD_NUMERIC_ELECTRODES = "electrodes\\three"
$env:CAD_NUMERIC_OUT = "numeric_out\\three"
.\.venv\Scripts\python cad_numeric.py mesh
.\.venv\Scripts\python cad_numeric.py solve
```

### Mesh configuration

You can override mesh parameters by placing a `mesh_config.json` in either:
- the selected electrodes folder (e.g. `electrodes\three\mesh_config.json`), or
- the output folder (e.g. `numeric_out\three\mesh_config.json`), or
- pointing `CAD_NUMERIC_MESH_CONFIG` at a JSON file.

### CAD electrode files (formats + naming)

The numeric pipeline treats **each CAD file as one electrode**:

- **Preferred format:** `.step` / `.stp` (STEP *solid* geometry). This path uses Gmsh/OpenCASCADE and is the most robust.
- Also supported: `.brep`, `.iges` / `.igs`.  
  `.stl` is only used as a fallback when no solid CAD files are found (STL is a surface mesh and is easier to break / mis-scale).
- Native CAD project files (e.g. FreeCAD `.FCStd`) are **not** read directly - export solids to STEP.
- **Electrode name = filename stem.** Example: `washer_rf.step` becomes electrode name `washer_rf` and produces `basis__washer_rf.npz`.

The GUI can auto-build an `electrode_program` from electrode names using simple keyword matching (case-insensitive):

- RF electrodes: names containing `rf`, `ring`, or `washer` -> `V_rf * cos(2*pi*f*t)`
- Rod electrodes: names containing `rod` (or unassigned `rf`) -> paired as `+RF` and `-RF` if two are present
- DC/ground electrodes: names containing `endcap`, `ground`, `gnd`, or patterns like `*cap`, `cap_`
  - include `+` / `plus` / `pos` for `+V_dc`, and `-` / `minus` / `neg` for `-V_dc`

If your names don't match these patterns, you can still use any names you want - just provide an explicit `electrode_program` mapping using the exact electrode names.

## Python API (non-GUI)

### Run a simulation

```python
import numpy as np
from fields import TrapSimulator

sim = TrapSimulator(
    V_rf=200.0,
    V_dc=0.0,
    Omega=2 * np.pi * 1e6,
    r0=1e-3,
    use_numeric=True,
    numeric_field_dir="numeric_out/three",
)

t, pos, vel = sim.sim_single_particle(
    initial_position=[0.0, 0.0, 0.0],
    initial_velocity=[0.0, 0.0, 0.0],
    t_span=(0.0, 1e-3),
    n_charges=100,
)
```

Key modules:
- `fields.py`: `TrapSimulator` (simulation + integration)
- `fieldsolver.py`: analytic and numeric field evaluators (with caching)
- `cad_numeric.py`: `mesh_from_cad()`, `solve_basis_fields()`, `NumericFieldSKFEM` loaders
- `beam_field.py` / `beam_gui.py`: particle-beam model + GUI panel

## Outputs and folders

- `electrodes/`: electrode CAD files (`.step`, `.stp`, `.iges`, `.igs`, `.brep`, ...)
- `numeric_out/`: numeric field outputs (mesh + basis files)
- `settings/`: saved GUI settings JSON examples

## Licenses (third-party dependencies)

See `THIRD_PARTY_LICENSES.md` for a dependency/license summary from the current `requirements.txt`.

## Project license

This project is licensed under the **GNU General Public License v3.0** (GPL-3.0-only). See `LICENSE`.
