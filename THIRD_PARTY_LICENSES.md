
## Project license

This project is licensed under **GNU GPL v3.0** (GPL-3.0-only). See `LICENSE`.

The versions below were read from the local `.venv` on **2026-02-11**.

## Direct dependencies (`requirements.txt`)

| Package | Version | License (package metadata) | Notes |
|---|---:|---|---|
| `numpy` | 2.2.6 | OSI Approved BSD License | Wheels bundle additional libraries (e.g., OpenBLAS) with their own licenses. |
| `scipy` | 1.15.3 | OSI Approved BSD License | Wheels may bundle additional libraries; review SciPy’s bundled license notices if redistributing. |
| `matplotlib` | 3.10.8 | Python Software Foundation License | See Matplotlib’s license file for full terms. |
| `scikit-fem` | 12.0.1 | OSI Approved BSD License | License field is provided via classifiers. |
| `meshio` | 5.3.5 | MIT |  |
| `gmsh` | 4.15.0 | GPLv2+ | **Copyleft**; affects distribution if you import/link the library in-process. |
| `numba` | 0.63.1 | BSD | Optional acceleration (auto-detected). |
| `joblib` | 1.5.3 | BSD-3-Clause |  |
| `pyvista` | 0.46.5 | MIT | Optional enhanced 3D visualization. |
| `pyvistaqt` | 0.11.3 | MIT | Optional enhanced 3D visualization. |
| `PyQt5` | 5.15.11 | GPL v3 | **Copyleft / commercial dual-license**; required by `pyvistaqt` when using PyQt5 as the Qt binding. |

## Notable transitive dependencies

These are pulled in by the direct dependencies above and are often relevant for compliance:

| Package | Version | License (package metadata) | Required by |
|---|---:|---|---|
| `vtk` | 9.5.2 | BSD | `pyvista` |
| `QtPy` | 2.4.3 | MIT | `pyvistaqt` |
| `PyQt5-Qt5` | 5.15.2 | LGPL v3 | `PyQt5` |
| `PyQt5_sip` | 12.18.0 | BSD-2-Clause | `PyQt5` |
