import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from scipy.constants import e
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import fields
import json
import meshio
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import threading
import os, sys, pathlib, subprocess
import traceback
import time
from queue import Queue, Empty
from datetime import datetime
from beam_gui import BeamControlPanel

# Parameter sweep imports
try:
    from parameter_sweep import ParameterSweep, SweepConfig, SweepParameter
    from sweep_gui import SweepConfigPanel
    import sweep_visualization
    SWEEP_AVAILABLE = True
except ImportError as e:
    SWEEP_AVAILABLE = False
    print(f"[Warning] Parameter sweep not available: {e}")

# PyVista imports (optional - for enhanced 3D visualization)
try:
    import pyvista as pv
    from pyvistaqt import BackgroundPlotter
    PYVISTA_AVAILABLE = True
except ImportError as e:
    PYVISTA_AVAILABLE = False
    print(f"[Warning] PyVista not available: {e}")
    print("  Install with: pip install pyvista pyvistaqt PyQt5")
except RuntimeError as e:
    # This happens when pyvistaqt is imported but no Qt binding is found
    PYVISTA_AVAILABLE = False
    print(f"[Warning] PyVista Qt binding not available: {e}")
    print("  Install with: pip install PyQt5")

def _get_project_root():
    if getattr(sys, "frozen", False):
        return pathlib.Path(sys.executable).resolve().parent
    return pathlib.Path(__file__).resolve().parent

PROJECT_ROOT = _get_project_root()
DEFAULT_ELECTRODES = PROJECT_ROOT / "electrodes"
DEFAULT_NUMERIC_OUT = PROJECT_ROOT / "numeric_out"
DEFAULT_SETTINGS = PROJECT_ROOT / "settings"
PIPELINE_SCRIPT = PROJECT_ROOT / "cad_numeric.py"
SUPPORTED_CAD_EXTS = {".step", ".stp", ".stl", ".iges", ".igs", ".brep", ".obj"}
MAX_TRAJ_POINTS = 1000
MAX_TIME_SERIES_POINTS = 5000
MAX_CLOUD_POINTS = 500
MAX_CAD_EDGES = 50000
MAX_CAD_EDGES_WIREFRAME = 1200

def _adaptive_downsample_indices(count, max_points):
    """Adaptive downsampling for smoother GUI interaction."""
    if max_points <= 0 or count <= max_points:
        return None

    if count > max_points * 10:
        step = count // max(max_points // 2, 1)
    else:
        step = count // max_points

    idx = np.arange(0, count, max(step, 1))
    if len(idx) == 0 or idx[-1] != count - 1:
        idx = np.append(idx, count - 1)
    return idx

# === helpers ===
def resolve_project_path(path_value):
    path = pathlib.Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()

def ensure_project_structure(
    electrodes_dir=DEFAULT_ELECTRODES,
    numeric_out=DEFAULT_NUMERIC_OUT,
    settings_dir=DEFAULT_SETTINGS,
):
    electrodes_dir.mkdir(parents=True, exist_ok=True)
    numeric_out.mkdir(parents=True, exist_ok=True)
    settings_dir.mkdir(parents=True, exist_ok=True)

def numeric_ready(numeric_out=DEFAULT_NUMERIC_OUT):
    """Return (is_ready, reason). Ready if mesh exists AND at least one basis file exists."""
    mesh_ok = (numeric_out / "mesh.msh").exists()
    basis_files = list(numeric_out.glob("basis__*.npz"))
    if mesh_ok and basis_files:
        return True, f"Found mesh.msh and {len(basis_files)} basis files."
    if not mesh_ok and not basis_files:
        return False, "No mesh or basis files. Run Mesh, then Solve."
    if mesh_ok and not basis_files:
        return False, "Mesh found but no basis files. Run Solve."
    return False, "Basis files exist but mesh.msh missing (unexpected)."

def electrodes_present(electrodes_dir=DEFAULT_ELECTRODES):
    cad_files = []
    ignored_files = []
    try:
        entries = list(electrodes_dir.iterdir())
    except FileNotFoundError:
        return False, cad_files, ignored_files

    for entry in entries:
        if entry.is_dir():
            continue
        if entry.suffix.lower() in SUPPORTED_CAD_EXTS:
            cad_files.append(entry)
        else:
            ignored_files.append(entry)
    return (len(cad_files) > 0), cad_files, ignored_files

class _LogCapture:
    def __init__(self, log_func):
        self._log_func = log_func or (lambda *_: None)
        self._buffer = ""

    def write(self, message):
        if not message:
            return 0
        if isinstance(message, bytes):
            message = message.decode(errors="replace")
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self._log_func(line)
        return len(message)

    def flush(self):
        if self._buffer:
            self._log_func(self._buffer)
            self._buffer = ""

def _run_pipeline_in_process(step, electrodes_dir, numeric_out, log_func=print):
    log_func(f"[Info] Running {step} in-process (frozen build)")
    os.environ["CAD_NUMERIC_ELECTRODES"] = str(electrodes_dir)
    os.environ["CAD_NUMERIC_OUT"] = str(numeric_out)
    try:
        import cad_numeric
    except Exception as exc:
        return False, f"Failed to import cad_numeric: {exc}"

    cad_numeric.ELECTRODE_DIR = pathlib.Path(electrodes_dir).resolve()
    cad_numeric.OUTDIR = pathlib.Path(numeric_out).resolve()
    cad_numeric.ELECTRODE_DIR.mkdir(parents=True, exist_ok=True)
    cad_numeric.OUTDIR.mkdir(parents=True, exist_ok=True)

    capture = _LogCapture(log_func)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = capture
    sys.stderr = capture
    try:
        if step == "mesh":
            cad_numeric.mesh_from_cad()
        elif step == "solve":
            cad_numeric.solve_basis_fields()
        else:
            return False, f"Unknown pipeline step: {step}"
    except Exception:
        return False, traceback.format_exc().strip()
    finally:
        try:
            capture.flush()
        except Exception:
            pass
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return True, ""

def run_pipeline_step(step, electrodes_dir, numeric_out, log_func=print):
    """
    step: 'mesh' or 'solve'
    Calls: python cad_numeric.py <step>
    """
    electrodes_dir = resolve_project_path(electrodes_dir)
    numeric_out = resolve_project_path(numeric_out)

    if getattr(sys, "frozen", False):
        return _run_pipeline_in_process(step, electrodes_dir, numeric_out, log_func=log_func)

    if not PIPELINE_SCRIPT.exists():
        return False, f"Missing {PIPELINE_SCRIPT.name} at {PIPELINE_SCRIPT}"

    env = os.environ.copy()
    env["CAD_NUMERIC_ELECTRODES"] = str(electrodes_dir)
    env["CAD_NUMERIC_OUT"] = str(numeric_out)

    cwd = PROJECT_ROOT

    cmd = [sys.executable, str(PIPELINE_SCRIPT), step]
    log_func(f"> {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
        if proc.stdout:
            for line in proc.stdout.splitlines():
                if line:
                    log_func(line)
        if proc.stderr:
            for line in proc.stderr.splitlines():
                if line:
                    log_func(line)
        if proc.returncode != 0:
            err = proc.stderr.strip() or f"{step} failed with exit code {proc.returncode}"
            return False, err
        return True, ""
    except Exception as e:
        return False, f"Failed to run {step}: {e}"

# === end helpers ===

def _build_electrode_program_from_numeric(numeric_dir, V_rf, V_dc, freq_hz):
    #Build electrode voltage program from numeric basis files
    try:
        ndir = pathlib.Path(numeric_dir)
        basis_files = sorted(ndir.glob("basis__*.npz"))
        names = [bf.stem.split("__", 1)[1] for bf in basis_files]
        if not names:
            return {}
        lowmap = {n.lower(): n for n in names}
        prog = {}
        two_pi_f = 2*np.pi*freq_hz

        # RF electrode: oscillating voltage
        rf = lambda t, amp=V_rf, w=two_pi_f: amp*np.cos(w*t)

        # Ring / washer -> RF
        for key in list(lowmap.keys()):
            if any(k in key for k in ("ring", "washer", "rf")):
                prog[lowmap[key]] = rf

        # Endcaps / ground electrodes -> +/- DC (for linear traps) or 0V (for surface traps)
        for key in list(lowmap.keys()):
            if ("endcap" in key or key.endswith("cap") or "cap_" in key or
                    "ground" in key or "gnd" in key):
                # For surface traps, all ground electrodes should be at 0V
                # Only apply +/- DC for linear/3D traps with axial confinement
                if abs(V_dc) < 1e-6:  # V_dc ~ 0, treat as surface trap
                    prog[lowmap[key]] = 0.0  # Explicitly 0.0, not -0.0
                elif any(k in key for k in ("+", "plus", "pos")):
                    prog[lowmap[key]] = float(V_dc)
                elif any(k in key for k in ("-", "minus", "neg")):
                    prog[lowmap[key]] = -float(V_dc)
                else:
                    if not any(isinstance(v, (int, float)) and v < 0 for v in prog.values()):
                        prog[lowmap[key]] = float(V_dc)
                    else:
                        prog[lowmap[key]] = -float(V_dc)

        # Rods / rf -> pair as +/- RF if not already assigned
        rf_candidates = [lowmap[k] for k in lowmap.keys() if ("rod" in k or ("rf" in k and "endcap" not in k and "ring" not in k and "washer" not in k))]
        rf_candidates = [n for n in rf_candidates if n not in prog]
        if len(rf_candidates) >= 2:
            prog[rf_candidates[0]] = rf
            prog[rf_candidates[1]] = -rf

        # Single electrode total -> Vrf*cos + Vdc
        if len(names) == 1 and not prog:
            prog[names[0]] = (lambda t, amp=V_rf, off=V_dc, w=two_pi_f: off + amp*np.cos(w*t))

        return prog
    except Exception:
        return {}


class PyVista3DViewer:
    """Interactive 3D viewer using PyVista for CAD mesh and trajectory visualization."""

    def __init__(self, title="3D Trap Viewer"):
        if not PYVISTA_AVAILABLE:
            raise RuntimeError("PyVista is not installed. Run: pip install pyvista pyvistaqt")

        self.plotter = None
        self.title = title
        self._mesh_actor = None
        self._trajectory_actor = None
        self._marker_actor = None
        self._start_marker_actor = None
        self._end_marker_actor = None
        self._mesh_data = None
        self._trajectory_points = None
        self._unit_scale = 1e6  # Default to micrometers
        self._unit_label = "μm"
        self._is_open = False

        # Configurable marker sizes (in display units, e.g., micrometers)
        self.start_marker_size = 4.0
        self.end_marker_size = 4.0
        self.position_marker_size = 3.0
        self.trajectory_line_width = 2.0

    def open(self):
        """Open the PyVista plotter window."""
        if self._is_open and self.plotter is not None:
            return

        self.plotter = BackgroundPlotter(
            title=self.title,
            window_size=(800, 600),
            toolbar=True,
            menu_bar=True,
        )
        self.plotter.add_axes()
        self.plotter.enable_trackball_style()
        self._is_open = True

        # Handle window close
        self.plotter.app_window.signal_close.connect(self._on_close)

    def _on_close(self):
        """Handle window close event."""
        self._is_open = False
        self._mesh_actor = None
        self._trajectory_actor = None
        self._marker_actor = None
        self.plotter = None

    def is_open(self):
        """Check if the viewer window is open."""
        return self._is_open and self.plotter is not None

    def close(self):
        """Close the viewer window."""
        if self.plotter is not None:
            try:
                self.plotter.close()
            except:
                pass
        self._is_open = False
        self.plotter = None

    def set_unit_scale(self, scale, label):
        """Set the unit scale for display (e.g., 1e6 for micrometers)."""
        self._unit_scale = scale
        self._unit_label = label

    def load_mesh(self, mesh_path, unit_scale_to_m=1.0, electrode_tags=None):
        """Load a mesh file and display it.

        Args:
            mesh_path: Path to the mesh file (.msh)
            unit_scale_to_m: Scale factor from mesh units to meters
            electrode_tags: Set of physical group tags to filter (electrodes only)
        """
        if not self.is_open():
            return False

        try:
            mesh = meshio.read(str(mesh_path))

            # Get triangle cells
            if "triangle" not in mesh.cells_dict:
                print(f"[PyVista] No triangle cells found in mesh")
                return False

            tris = mesh.cells_dict["triangle"]

            # Filter by electrode tags if available
            if electrode_tags and "gmsh:physical" in mesh.cell_data_dict:
                phys_tags = None
                if "triangle" in mesh.cell_data_dict.get("gmsh:physical", {}):
                    phys_tags = mesh.cell_data_dict["gmsh:physical"]["triangle"]
                else:
                    for i, cell_block in enumerate(mesh.cells):
                        if cell_block.type == "triangle":
                            if i < len(mesh.cell_data.get("gmsh:physical", [])):
                                phys_tags = mesh.cell_data["gmsh:physical"][i]
                            break

                if phys_tags is not None:
                    mask = np.isin(phys_tags, list(electrode_tags))
                    tris = tris[mask]
                    print(f"[PyVista] Filtered to {len(tris)} electrode triangles")

            if len(tris) == 0:
                print("[PyVista] No triangles to display after filtering")
                return False

            # Scale points: mesh units -> meters -> display units
            points = mesh.points * unit_scale_to_m * self._unit_scale

            # Create PyVista mesh (need to prepend face count for each face)
            n_tris = len(tris)
            faces = np.column_stack([np.full(n_tris, 3), tris]).ravel()

            pv_mesh = pv.PolyData(points, faces=faces)

            # Remove old mesh actor
            if self._mesh_actor is not None:
                self.plotter.remove_actor(self._mesh_actor)

            # Add mesh with nice appearance
            self._mesh_actor = self.plotter.add_mesh(
                pv_mesh,
                color='lightblue',
                opacity=0.7,
                show_edges=True,
                edge_color='darkblue',
                line_width=0.5,
                name='cad_mesh'
            )

            self._mesh_data = pv_mesh
            self.plotter.reset_camera()
            print(f"[PyVista] Loaded mesh with {n_tris} faces")
            return True

        except Exception as e:
            print(f"[PyVista] Error loading mesh: {e}")
            return False

    def update_trajectory(self, positions, unit_scale_to_m=1.0):
        """Update the trajectory line display.

        Args:
            positions: Nx3 array of positions in meters
            unit_scale_to_m: Scale factor (usually 1.0 if positions are already in meters)
        """
        if not self.is_open() or positions is None or len(positions) < 2:
            return

        # Scale to display units
        scaled_pos = positions * unit_scale_to_m * self._unit_scale
        self._trajectory_points = scaled_pos

        # Remove old actors
        for actor in [self._trajectory_actor, self._start_marker_actor, self._end_marker_actor]:
            if actor is not None:
                try:
                    self.plotter.remove_actor(actor)
                except:
                    pass

        # Create line from points
        line = pv.lines_from_points(scaled_pos)

        self._trajectory_actor = self.plotter.add_mesh(
            line,
            color='red',
            line_width=self.trajectory_line_width,
            name='trajectory'
        )

        # Add start/end markers with configurable sizes
        start_point = pv.Sphere(radius=self.start_marker_size, center=scaled_pos[0])
        end_point = pv.Sphere(radius=self.end_marker_size, center=scaled_pos[-1])

        self._start_marker_actor = self.plotter.add_mesh(start_point, color='green', name='start_marker')
        self._end_marker_actor = self.plotter.add_mesh(end_point, color='blue', name='end_marker')

    def update_marker(self, position, unit_scale_to_m=1.0):
        """Update the position marker (for highlighting clicked time point).

        Args:
            position: 3-element array [x, y, z] in meters
            unit_scale_to_m: Scale factor (usually 1.0 if position is already in meters)
        """
        if not self.is_open() or position is None:
            return

        # Scale to display units
        scaled_pos = np.array(position) * unit_scale_to_m * self._unit_scale

        # Remove old marker
        if self._marker_actor is not None:
            try:
                self.plotter.remove_actor(self._marker_actor)
            except:
                pass

        # Create marker sphere with configurable size
        marker = pv.Sphere(radius=self.position_marker_size, center=scaled_pos)
        self._marker_actor = self.plotter.add_mesh(
            marker,
            color='orange',
            name='position_marker'
        )

    def set_marker_sizes(self, start=None, end=None, position=None, line_width=None):
        """Update marker sizes. Pass None to keep current value."""
        if start is not None:
            self.start_marker_size = start
        if end is not None:
            self.end_marker_size = end
        if position is not None:
            self.position_marker_size = position
        if line_width is not None:
            self.trajectory_line_width = line_width

    def clear_trajectory(self):
        """Clear the trajectory display."""
        if not self.is_open():
            return

        for name in ['trajectory', 'start_marker', 'end_marker', 'position_marker']:
            try:
                self.plotter.remove_actor(name)
            except:
                pass

        self._trajectory_actor = None
        self._marker_actor = None
        self._trajectory_points = None


class PaulTrapGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Paul Trap Simulator")
        self.electrode_dir_path = DEFAULT_ELECTRODES
        self.numeric_field_dir = DEFAULT_NUMERIC_OUT
        self.use_numeric_mode = False

        # PyVista 3D viewer (lazy initialized)
        self._pyvista_viewer = None

        # cache for numeric mesh overlay (per numeric_dir)
        self._numeric_overlay_cache = {}
        self._cad_mesh_data = None
        self._electrode_bounds_cache = {}
        self.params = {
            'gravity': True,
            'damping_gamma': 0.0,
            'escape_radius': 1e-2
        }
        self.sim = None
        self.running = False
        self.simulation_thread = None
        self.plot_queue = Queue()
        self._stop_requested = False
        self._last_auto_grid_value = None
        
        self.build_gui()
        self.root.after(100, self.check_plot_queue)
        
    def build_gui(self):
        # Make stretchable
        self.root.columnconfigure(0, weight=1) 
        self.root.rowconfigure(0, weight=1)
        
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.grid(row=0, column=0, sticky="nsew")
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(0, weight=1)

        # LEFT: Tabbed controls
        left_container = ttk.Frame(main_container, width=300)
        left_container.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        left_container.rowconfigure(0, weight=1)
        left_container.columnconfigure(0, weight=1)
        
        self.notebook = ttk.Notebook(left_container)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        
        # Create tabs
        self.tab_field = ttk.Frame(self.notebook)
        self.tab_numeric = ttk.Frame(self.notebook)
        self.tab_particle = ttk.Frame(self.notebook)
        self.tab_beam = ttk.Frame(self.notebook)
        self.tab_simulation = ttk.Frame(self.notebook)
        self.tab_view = ttk.Frame(self.notebook)
        self.tab_sweep = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_field, text="Field")
        self.notebook.add(self.tab_numeric, text="CAD")
        self.notebook.add(self.tab_particle, text="Particle")
        self.notebook.add(self.tab_beam, text="Beam")
        self.notebook.add(self.tab_simulation, text="Simulation")
        self.notebook.add(self.tab_view, text="View")
        if SWEEP_AVAILABLE:
            self.notebook.add(self.tab_sweep, text="Sweep")

        # Build each tab
        self.build_field_tab()
        self.build_numeric_tab()
        self.build_particle_tab()
        self.build_beam_tab()
        self.build_simulation_tab()
        self.build_view_tab()
        if SWEEP_AVAILABLE:
            self.build_sweep_tab()

        # RIGHT: Plot area
        right = ttk.Frame(main_container)
        right.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        main_container.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        
        self.fig = Figure(figsize=(8, 5))
        self.ax3d = self.fig.add_subplot(2, 2, 1, projection='3d')
        self.ax_x = self.fig.add_subplot(2, 2, 2)
        self.ax_y = self.fig.add_subplot(2, 2, 3)
        self.ax_z = self.fig.add_subplot(2, 2, 4)
        self.fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Connect interactive events for trajectory selection
        self.canvas.mpl_connect('button_press_event', self._on_plot_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_plot_hover)

        # Plot state caches
        self._last_plot_kind = None
        self._last_plot_payload = None
        self._field_vector_cache = {"key": None, "vectors": None}
        self._trajectory_data = None
        self._selected_point_markers = {}
        self._selected_fft_markers = {}
        self._hover_annotations = {}
        self._fft_data = None
        
        # Save panel references for hide/show
        self.left_panel = left_container
        self.right_panel = right
        self._controls_hidden = False
        self._fullscreen = False
        
        # Keyboard shortcuts
        self.root.bind('<F11>', lambda e: self.toggle_fullscreen())
        self.root.bind('<Escape>', lambda e: self.exit_fullscreen())
        self.root.bind('<Control-r>', lambda e: self.on_run())
        self.root.bind('<Control-s>', lambda e: self.on_stop())
        self.root.bind('<Control-p>', lambda e: self.pop_out_plot())
        self.root.bind('<F5>', lambda e: self.canvas.draw())

    def on_auto_duration_click(self):
        """Auto-set duration to 100 RF cycles when button is clicked"""
        try:
            freq = self.freq_var.get()
            if freq <= 0:
                return

            # Simulate 100 RF cycles
            target_cycles = 100
            duration = target_cycles / freq

            self.duration_var.set(float(f"{duration:.2e}"))

        except Exception:
            pass

    def on_auto_dt_click(self):
        """Auto-set time step to resolve RF oscillations (20 samples per cycle)"""
        try:
            freq = self.freq_var.get()
            if freq <= 0:
                return

            # For ion traps at MHz frequencies, need more samples to avoid energy drift
            # 50 samples per cycle gives better energy conservation
            samples_per_cycle = 50
            dt = 1.0 / (freq * samples_per_cycle)

            # Round to a nice value
            self.dt_var.set(float(f"{dt:.2e}"))

        except Exception:
            pass

    def build_field_tab(self):
        """Field settings tab"""
        tab = self.tab_field
        row = 0
        
        # Field source mode selector (prominent)
        mode_frame = ttk.LabelFrame(tab, text="Field Source", padding=10)
        mode_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=8)
        row += 1
        
        self.mode_var = tk.StringVar(value="Analytic")
        ttk.Radiobutton(
            mode_frame, 
            text="Analytic", 
            variable=self.mode_var,
            value="Analytic",
            command=self.on_field_mode_change
        ).pack(anchor='w', pady=2)
        
        ttk.Radiobutton(
            mode_frame, 
            text="CAD Numeric", 
            variable=self.mode_var,
            value="CAD numeric",
            command=self.on_field_mode_change
        ).pack(anchor='w', pady=2)
        
        ttk.Separator(tab, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky="we", pady=8)
        row += 1
        
        # Voltage settings
        voltage_frame = ttk.LabelFrame(tab, text="Voltages", padding=8)
        voltage_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row += 1
        
        self.Vdc_var = tk.DoubleVar(value=0.0)
        self.Vrf_var = tk.DoubleVar(value=1000.0)
        self.freq_var = tk.DoubleVar(value=100)
        
        ttk.Label(voltage_frame, text="DC voltage U (V)").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(voltage_frame, textvariable=self.Vdc_var, width=15).grid(row=0, column=1, pady=2)
        
        ttk.Label(voltage_frame, text="RF amplitude V (V)").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(voltage_frame, textvariable=self.Vrf_var, width=15).grid(row=1, column=1, pady=2)
        
        ttk.Label(voltage_frame, text="RF frequency (Hz)").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(voltage_frame, textvariable=self.freq_var, width=15).grid(row=2, column=1, pady=2)

        # Trap depth calculator button
        ttk.Button(voltage_frame, text="Check Trap Depth", command=self.check_trap_depth).grid(
            row=3, column=0, columnspan=2, sticky="we", pady=8
        )

        # Geometry settings
        geometry_frame = ttk.LabelFrame(tab, text="Trap Geometry", padding=8)
        geometry_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row += 1
        
        self.r0_var = tk.DoubleVar(value=1.0e-3)
        ttk.Label(geometry_frame, text="Characteristic size (m)").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(geometry_frame, textvariable=self.r0_var, width=15).grid(row=0, column=1, pady=2)
        
        ttk.Label(geometry_frame, text="Trap type").grid(row=1, column=0, sticky="w", pady=2)
        self.trap_type = tk.StringVar(value="Planar (washer)")
        trap_box = ttk.Combobox(
            geometry_frame,
            textvariable=self.trap_type,
            values=["Planar (washer)", "3D (hyperbolic)", "Linear", "Custom (a,b,c)"],
            state="readonly",
            width=18
        )
        trap_box.grid(row=1, column=1, sticky="w", pady=2)
        trap_box.bind('<<ComboboxSelected>>', self.on_trap_type_change)
        
        # Custom coefficients (shown only for custom trap)
        self.custom_abc_frame = ttk.Frame(geometry_frame)
        self.custom_abc_frame.grid(row=2, column=0, columnspan=2, sticky="we", pady=4)
        
        self.a_var = tk.DoubleVar(value=1.0)
        self.b_var = tk.DoubleVar(value=1.0)
        self.c_var = tk.DoubleVar(value=-2.0)
        
        ttk.Label(self.custom_abc_frame, text="a, b, c coefficients:").grid(row=0, column=0, sticky="w", pady=2)
        abc_entry_frame = ttk.Frame(self.custom_abc_frame)
        abc_entry_frame.grid(row=0, column=1, sticky="w", pady=2)
        ttk.Entry(abc_entry_frame, textvariable=self.a_var, width=5).pack(side='left', padx=1)
        ttk.Entry(abc_entry_frame, textvariable=self.b_var, width=5).pack(side='left', padx=1)
        ttk.Entry(abc_entry_frame, textvariable=self.c_var, width=5).pack(side='left', padx=1)
        ttk.Label(self.custom_abc_frame, text="(Laplace: a+b+c=0)", font=('', 8)).grid(row=1, column=0, columnspan=2, sticky="w")
        
        self.on_trap_type_change()  # Initialize visibility
        
        # Spacer
        ttk.Label(tab, text="").grid(row=row, column=0)

    def build_particle_tab(self):
        """Particle properties tab"""
        tab = self.tab_particle
        row = 0

        # Physical properties
        physical_frame = ttk.LabelFrame(tab, text="Physical Properties", padding=8)
        physical_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=8)
        row += 1

        self.radius_var = tk.DoubleVar(value=5e-6)
        self.density_var = tk.DoubleVar(value=2200)
        self.disable_stokes_var = tk.BooleanVar(value=True)

        ttk.Label(physical_frame, text="Radius (m)").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(physical_frame, textvariable=self.radius_var, width=15).grid(row=0, column=1, pady=2)

        ttk.Label(physical_frame, text="Density (kg/m^3)").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(physical_frame, textvariable=self.density_var, width=15).grid(row=1, column=1, pady=2)

        ttk.Checkbutton(
            physical_frame,
            text="Disable Stokes drag/Pressure",
            variable=self.disable_stokes_var,
            command=self.on_stokes_drag_toggle
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=4)

        # Pressure settings (nested frame for clarity)
        pressure_frame = ttk.Frame(physical_frame)
        pressure_frame.grid(row=3, column=0, columnspan=2, sticky="we", pady=4)

        ttk.Label(pressure_frame, text="Buffer gas pressure (Torr):").pack(side='left', padx=(20, 5))
        self.pressure_torr_var = tk.DoubleVar(value=1e-6)
        self.pressure_entry = ttk.Entry(pressure_frame, textvariable=self.pressure_torr_var, width=15, state="disabled")
        self.pressure_entry.pack(side='left', padx=5)
        ttk.Label(pressure_frame, text="(disabled when drag off)", font=('', 8), foreground='gray').pack(side='left', padx=5)

        self.damping_var = tk.DoubleVar(value=0.0)
        ttk.Label(physical_frame, text="Additional damping gamma (1/s)").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Entry(physical_frame, textvariable=self.damping_var, width=15).grid(row=4, column=1, pady=2)

        self.gravity_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(physical_frame, text="Include gravity", variable=self.gravity_var).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=2
        )
        
        
        # Charge settings
        charge_frame = ttk.LabelFrame(tab, text="Charge", padding=8)
        charge_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row += 1
        
        self.n_charges_var = tk.IntVar(value=100)
        ttk.Label(charge_frame, text="Elementary charges").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(charge_frame, textvariable=self.n_charges_var, width=15).grid(row=0, column=1, pady=2)
        
        # q/m ratio
        self.qm_ratio_var = tk.DoubleVar(value=0.0)
        self.use_qm_override_var = tk.BooleanVar(value=False)
        self.qm_ratio_info_var = tk.StringVar(value="computed: n/a")
        self._updating_qm_ratio = False
        self.use_mc_override_var = tk.BooleanVar(value=False)
        self.mass_var = tk.DoubleVar(value=0.0)
        self.charge_var = tk.DoubleVar(value=0.0)
        
        ttk.Label(charge_frame, text="q/m ratio (C/kg)").grid(row=1, column=0, sticky="w", pady=2)
        self.qm_entry = ttk.Entry(charge_frame, textvariable=self.qm_ratio_var, width=15, state="readonly")
        self.qm_entry.grid(row=1, column=1, pady=2)
        
        ttk.Checkbutton(
            charge_frame,
            text="Override q/m ratio",
            variable=self.use_qm_override_var,
            command=self.on_qm_override_toggle
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=4)
        
        ttk.Label(charge_frame, textvariable=self.qm_ratio_info_var, font=('', 8)).grid(
            row=3, column=0, columnspan=2, sticky="w"
        )

        ttk.Checkbutton(
            charge_frame,
            text="Override mass/charge (ignore radius/density)",
            variable=self.use_mc_override_var,
            command=self.on_mc_override_toggle
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=4)
        
        ttk.Label(charge_frame, text="Mass (kg)").grid(row=5, column=0, sticky="w", pady=2)
        self.mass_entry = ttk.Entry(charge_frame, textvariable=self.mass_var, width=15, state="readonly")
        self.mass_entry.grid(row=5, column=1, pady=2)
        
        ttk.Label(charge_frame, text="Charge (C)").grid(row=6, column=0, sticky="w", pady=2)
        self.charge_entry = ttk.Entry(charge_frame, textvariable=self.charge_var, width=15, state="readonly")
        self.charge_entry.grid(row=6, column=1, pady=2)
        
        # Bind updates
        for var in (self.radius_var, self.density_var, self.n_charges_var, self.mass_var, self.charge_var):
            var.trace_add("write", self.update_qm_ratio_display)

        self.update_qm_ratio_display()
        self.on_qm_override_toggle()
        self.on_mc_override_toggle()
        self.on_stokes_drag_toggle()

    def build_beam_tab(self):
        """Particle beam tab"""
        tab = self.tab_beam

        # Create a scrollable frame for the beam control panel
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack the scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Create the beam control panel
        self.beam_panel = BeamControlPanel(scrollable_frame, on_change=self.on_beam_change)
        self.beam_panel.pack(fill='both', expand=True, padx=10, pady=10)

    def on_beam_change(self, params):
        """Callback when beam parameters change"""
        # Update the beam with current RF frequency if available
        if hasattr(self, 'freq_var'):
            try:
                rf_freq = self.freq_var.get()
                self.beam_panel.set_rf_parameters(frequency=rf_freq)
            except:
                pass

    def build_simulation_tab(self):
        #Simulation control tab
        tab = self.tab_simulation
        row = 0
        
        # Particle count
        particle_frame = ttk.LabelFrame(tab, text="Particles", padding=8)
        particle_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=8)
        row += 1
        
        self.num_var = tk.IntVar(value=1)
        ttk.Label(particle_frame, text="Number of particles").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(particle_frame, textvariable=self.num_var, width=15).grid(row=0, column=1, pady=2)
        
        # Time settings
        time_frame = ttk.LabelFrame(tab, text="Time Settings", padding=8)
        time_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row += 1
        
        self.duration_var = tk.DoubleVar(value=0.1)
        self.dt_var = tk.DoubleVar(value=1e-9)
        
        ttk.Label(time_frame, text="Duration (s)").grid(row=0, column=0, sticky="w", pady=2)
        self.dur_entry = ttk.Entry(time_frame, textvariable=self.duration_var, width=15)
        self.dur_entry.grid(row=0, column=1, pady=2)

        # Button to auto-set duration based on RF frequency (100 cycles)
        ttk.Button(
            time_frame,
            text="Auto (100 cycles)",
            command=self.on_auto_duration_click,
            width=14
        ).grid(row=0, column=2, padx=4, sticky="w")

        ttk.Label(time_frame, text="Time step dt (s)").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(time_frame, textvariable=self.dt_var, width=15).grid(row=1, column=1, pady=2)

        # Button to auto-set time step based on RF frequency (20 samples/cycle)
        ttk.Button(
            time_frame,
            text="Auto dt",
            command=self.on_auto_dt_click,
            width=14
        ).grid(row=1, column=2, padx=4, sticky="w")

        self.fast_single_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            time_frame,
            text="Fast single-particle (fixed-step RK4)",
            variable=self.fast_single_var
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=2)
        
        # Performance settings
        perf_frame = ttk.LabelFrame(tab, text="Performance Settings", padding=8)
        perf_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row += 1
        
        ttk.Label(perf_frame, text="Integrator:").grid(row=0, column=0, sticky="w", pady=2)
        self.integration_method = tk.StringVar(value="Auto")
        ttk.Combobox(
            perf_frame,
            textvariable=self.integration_method,
            values=["Auto", "Adaptive (RK45)", "Fixed-step (RK4)", "Velocity Verlet"],
            state="readonly",
            width=18
        ).grid(row=0, column=1, sticky="w", pady=2)
        
        self.use_secular_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(perf_frame, text="Secular approx (fast)", 
                        variable=self.use_secular_var).grid(row=1, column=0, columnspan=2, sticky="w")

        self.use_numeric_grid_var = tk.BooleanVar(value=True)  # Default ON now
        ttk.Checkbutton(
            perf_frame,
            text="Use grid interpolation (fast)",
            variable=self.use_numeric_grid_var
        ).grid(row=2, column=0, columnspan=2, sticky="w")

        # Grid resolution
        ttk.Label(perf_frame, text="Grid resolution:").grid(row=3, column=0, sticky="e", pady=2)
        self.grid_resolution_var = tk.IntVar(value=100)
        self.grid_resolution_spin = ttk.Spinbox(
            perf_frame,
            from_=50,
            to=200,
            increment=10,
            textvariable=self.grid_resolution_var,
            width=8,
        )
        self.grid_resolution_spin.grid(row=3, column=1, sticky="w", pady=2)
        self._last_manual_grid_resolution = self.grid_resolution_var.get()

        self.auto_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            perf_frame,
            text="Auto grid resolution",
            variable=self.auto_grid_var,
            command=self.on_auto_grid_toggle,
        ).grid(row=4, column=0, columnspan=2, sticky="w")

        self.auto_grid_display_var = tk.StringVar(value="Auto grid: pending")
        ttk.Label(perf_frame, textvariable=self.auto_grid_display_var, font=('', 8)).grid(
            row=5, column=0, columnspan=2, sticky="w"
        )

        # Adaptive grid refinement
        self.use_adaptive_grid_var = tk.BooleanVar(value=True)  # Default ON
        ttk.Checkbutton(
            perf_frame,
            text="Adaptive refinement (better accuracy)",
            variable=self.use_adaptive_grid_var
        ).grid(row=6, column=0, columnspan=2, sticky="w")

        ttk.Label(perf_frame, text="Grid smoothing (cells):").grid(row=7, column=0, sticky="e", pady=2)
        self.grid_smoothing_var = tk.DoubleVar(value=0.6)
        ttk.Spinbox(
            perf_frame,
            from_=0.0,
            to=2.0,
            increment=0.1,
            textvariable=self.grid_smoothing_var,
            width=8,
        ).grid(row=7, column=1, sticky="w", pady=2)

        self.use_parallel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            perf_frame,
            text="Use parallel processing",
            variable=self.use_parallel_var
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=2)
        
        ttk.Label(perf_frame, text="Field cache (entries):").grid(row=9, column=0, sticky="w", pady=2)
        self.cache_size_var = tk.IntVar(value=50000)
        ttk.Entry(perf_frame, textvariable=self.cache_size_var, width=10).grid(row=9, column=1, sticky="w", pady=2)

        # Control buttons
        control_frame = ttk.Frame(tab)
        control_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=8)
        row += 1
        
        self.run_button = ttk.Button(control_frame, text="Run Simulation", command=self.on_run)
        self.run_button.pack(side='left', padx=4, fill='x', expand=True)

        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.on_stop, state='disabled')
        self.stop_button.pack(side='left', padx=4, fill='x', expand=True)

        self.export_button = ttk.Button(control_frame, text="Export Data", command=self.export_data)
        self.export_button.pack(side='left', padx=4, fill='x', expand=True)

        # Progress bar
        progress_frame = ttk.LabelFrame(tab, text="Progress", padding=8)
        progress_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row += 1
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill='x', pady=4)
        
        self.progress_label = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.progress_label, font=('', 8)).pack()

        # Settings import/export
        settings_frame = ttk.LabelFrame(tab, text="Settings", padding=8)
        settings_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row += 1
        ttk.Button(settings_frame, text="Export Settings", command=self.export_settings).pack(
            side="left", padx=4, fill="x", expand=True
        )
        ttk.Button(settings_frame, text="Import Settings", command=self.import_settings).pack(
            side="left", padx=4, fill="x", expand=True
        )
        
        # Status log
        status_frame = ttk.LabelFrame(tab, text="Status Log", padding=8)
        status_frame.grid(row=row, column=0, columnspan=2, sticky="nsew", padx=8, pady=4)
        row += 1
        tab.rowconfigure(row-1, weight=1)
        
        # Scrollable text widget
        status_scroll = ttk.Scrollbar(status_frame, orient='vertical')
        self.status_text = tk.Text(status_frame, height=8, width=35, state='disabled', 
                                   yscrollcommand=status_scroll.set, wrap='word')
        status_scroll.config(command=self.status_text.yview)
        self.status_text.pack(side='left', fill='both', expand=True)
        status_scroll.pack(side='right', fill='y')
        
        # Configure tags for colored output
        self.status_text.tag_config('info', foreground='black')
        self.status_text.tag_config('warning', foreground='orange')
        self.status_text.tag_config('error', foreground='red')
        self.status_text.tag_config('success', foreground='green')

        self.on_auto_grid_toggle()
        
        self.update_status("Ready to simulate", 'info')

    def build_view_tab(self):
        """View and visualization settings tab"""
        tab = self.tab_view
        row = 0
        
        # Display options
        display_frame = ttk.LabelFrame(tab, text="Display Options", padding=8)
        display_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=8)
        row += 1
        
        ttk.Button(
            display_frame, 
            text="Toggle Fullscreen (F11)", 
            command=self.toggle_fullscreen
        ).pack(fill='x', pady=2)
        
        """self.hide_controls_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            display_frame, 
            text="Hide controls (maximize plot)", 
            variable=self.hide_controls_var, 
            command=self.toggle_controls
        ).pack(anchor='w', pady=2)"""
        
        ttk.Button(
            display_frame,
            text="Pop-out Plot (Ctrl+P)",
            command=self.pop_out_plot
        ).pack(fill='x', pady=2)
        
        ttk.Separator(tab, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky="we", pady=8)
        row += 1
        
        # 3D view controls
        view3d_frame = ttk.LabelFrame(tab, text="3D View Angles", padding=8)
        view3d_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row += 1
        
        self.azim_var = tk.DoubleVar(value=-60.0)
        self.elev_var = tk.DoubleVar(value=30.0)
        
        ttk.Label(view3d_frame, text="Azimuth").grid(row=0, column=0, sticky="w", pady=2)
        az_scale = ttk.Scale(view3d_frame, from_=-180, to=180, variable=self.azim_var,
                            command=lambda *_: self.update_view())
        az_scale.grid(row=0, column=1, sticky="we", pady=2)
        
        ttk.Label(view3d_frame, text="Elevation").grid(row=1, column=0, sticky="w", pady=2)
        el_scale = ttk.Scale(view3d_frame, from_=-10, to=90, variable=self.elev_var,
                            command=lambda *_: self.update_view())
        el_scale.grid(row=1, column=1, sticky="we", pady=2)
        
        # Preset view buttons
        preset_frame = ttk.Frame(view3d_frame)
        preset_frame.grid(row=2, column=0, columnspan=2, pady=4)
        ttk.Button(preset_frame, text="XY", width=6, command=lambda: self.set_view('xy')).pack(side='left', padx=2)
        ttk.Button(preset_frame, text="XZ", width=6, command=lambda: self.set_view('xz')).pack(side='left', padx=2)
        ttk.Button(preset_frame, text="YZ", width=6, command=lambda: self.set_view('yz')).pack(side='left', padx=2)
        ttk.Button(preset_frame, text="Iso", width=6, command=lambda: self.set_view('iso')).pack(side='left', padx=2)
        
        #additional visualization
        viz_frame = ttk.LabelFrame(tab, text="Visualization", padding=8)
        viz_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row += 1
        
        self.show_field_vectors_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            viz_frame,
            text="Show field gradient (arrows)",
            variable=self.show_field_vectors_var,
            command=self.on_field_vectors_toggle
        ).pack(anchor='w', pady=2)

        self.show_cad_mesh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            viz_frame,
            text="Show CAD/Mesh",
            variable=self.show_cad_mesh_var,
            command=self.on_cad_mesh_toggle
        ).pack(anchor='w', pady=2)

        self.show_fft_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            viz_frame,
            text="Show FFT",
            variable=self.show_fft_var,
            command=self.on_fft_toggle
        ).pack(anchor='w', pady=2)

        # PyVista 3D viewer button
        ttk.Separator(viz_frame, orient='horizontal').pack(fill='x', pady=8)
        self._pyvista_btn = ttk.Button(
            viz_frame,
            text="Open 3D Viewer (PyVista)",
            command=self.open_pyvista_viewer
        )
        self._pyvista_btn.pack(fill='x', pady=2)
        if not PYVISTA_AVAILABLE:
            self._pyvista_btn.configure(state='disabled')
            ttk.Label(viz_frame, text="(Install pyvista pyvistaqt)", font=('', 7)).pack(anchor='w')

        # PyVista marker size controls
        pv_marker_frame = ttk.Frame(viz_frame)
        pv_marker_frame.pack(fill='x', pady=4)

        ttk.Label(pv_marker_frame, text="Marker size (μm):", font=('', 8)).pack(anchor='w')

        marker_controls = ttk.Frame(pv_marker_frame)
        marker_controls.pack(fill='x')

        self._pv_marker_size_var = tk.DoubleVar(value=5.0)
        ttk.Label(marker_controls, text="Start/End:", font=('', 7)).grid(row=0, column=0, sticky='w')
        pv_marker_scale = ttk.Scale(
            marker_controls, from_=0.5, to=50.0,
            variable=self._pv_marker_size_var,
            command=lambda *_: self._on_pv_marker_size_change()
        )
        pv_marker_scale.grid(row=0, column=1, sticky='we', padx=4)

        self._pv_pos_marker_size_var = tk.DoubleVar(value=8.0)
        ttk.Label(marker_controls, text="Position:", font=('', 7)).grid(row=1, column=0, sticky='w')
        pv_pos_scale = ttk.Scale(
            marker_controls, from_=0.5, to=50.0,
            variable=self._pv_pos_marker_size_var,
            command=lambda *_: self._on_pv_marker_size_change()
        )
        pv_pos_scale.grid(row=1, column=1, sticky='we', padx=4)

        marker_controls.columnconfigure(1, weight=1)

        ttk.Label(viz_frame, text="Z-axis scale (10^x)").pack(anchor='w', pady=(8, 2))
        self.axis_scale_var = tk.DoubleVar(value=0.0)
        ttk.Scale(
            viz_frame,
            from_=-4.0,
            to=4.0,
            orient="horizontal",
            variable=self.axis_scale_var,
            command=lambda *_: self.on_axis_scale_change()
        ).pack(fill='x', pady=2)
        
        self.axis_scale_display = tk.StringVar(value="1.0x")
        ttk.Label(viz_frame, textvariable=self.axis_scale_display, font=('', 8)).pack(anchor='w')

        # Keyboard shortcuts info
        ttk.Separator(tab, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky="we", pady=8)
        row += 1
        
        shortcuts_frame = ttk.LabelFrame(tab, text="Keyboard Shortcuts", padding=8)
        shortcuts_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row += 1
        
        shortcuts_text = """F11: Toggle fullscreen
Esc: Exit fullscreen
Ctrl+R: Run simulation
Ctrl+S: Stop simulation
Ctrl+P: Pop-out plot
F5: Refresh plot"""
        
        ttk.Label(shortcuts_frame, text=shortcuts_text, font=('Courier', 8), justify='left').pack(anchor='w')

    def build_numeric_tab(self):
        """CAD/Numeric field settings tab"""
        tab = self.tab_numeric
        
        # Instructions
        info_frame = ttk.LabelFrame(tab, text="Information", padding=8)
        info_frame.grid(row=0, column=0, columnspan=2, sticky="we", padx=8, pady=8)
        
        info_text = """To use CAD-based numeric fields:
1. Place CAD files in electrodes folder
2. Click 'Mesh' to generate mesh
3. Click 'Solve' to compute basis fields
4. Switch to 'CAD numeric' in Field tab"""
        
        ttk.Label(info_frame, text=info_text, justify='left', wraplength=250).pack(anchor='w')
        
        # Embed the NumericFieldPanel
        self.numeric_panel = NumericFieldPanel(tab, on_paths_changed=self.on_numeric_paths_changed)
        self.numeric_panel.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=8, pady=8)
        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)

    # === Callback methods ===
    
    def on_field_mode_change(self):
        """Handle field mode switch between analytic and numeric"""
        mode = self.mode_var.get()
        self.use_numeric_mode = (mode == "CAD numeric")
        
        if self.use_numeric_mode:
            # Check if numeric fields are ready
            ready, reason = numeric_ready(self.numeric_field_dir)
            if not ready:
                self.update_status(f"Numeric fields not ready: {reason}", 'warning')
                # Automatically switch to CAD tab
                self.notebook.select(self.tab_numeric)
            else:
                self.update_status("Using CAD numeric fields", 'success')
        else:
            self.update_status("Using analytic fields", 'info')
    
    def on_trap_type_change(self, event=None):
        """Show/hide custom abc coefficients based on trap type"""
        if self.trap_type.get() == "Custom (a,b,c)":
            self.custom_abc_frame.grid()
        else:
            self.custom_abc_frame.grid_remove()
    
    def on_stokes_drag_toggle(self):
        """Enable/disable pressure field when Stokes drag is toggled"""
        if self.disable_stokes_var.get():
            # Drag is DISABLED - disable pressure field (vacuum mode)
            self.pressure_entry.configure(state="disabled")
        else:
            # Drag is ENABLED - enable pressure field
            self.pressure_entry.configure(state="normal")

    def on_qm_override_toggle(self):
        """Enable/disable q/m ratio manual entry"""
        if self.use_qm_override_var.get():
            self.qm_entry.configure(state="normal")
            self.qm_ratio_info_var.set("manual override enabled")
        else:
            self.qm_entry.configure(state="readonly")
            self.update_qm_ratio_display()

    def on_mc_override_toggle(self):
        """Enable/disable direct mass/charge override."""
        if self.use_mc_override_var.get():
            self.mass_entry.configure(state="normal")
            self.charge_entry.configure(state="normal")
        else:
            self.mass_entry.configure(state="readonly")
            self.charge_entry.configure(state="readonly")
        self.update_qm_ratio_display()
    
    def update_qm_ratio_display(self, *args):
        """Update the displayed q/m ratio"""
        if self._updating_qm_ratio or self.use_qm_override_var.get():
            return
        
        try:
            self._updating_qm_ratio = True
            if self.use_mc_override_var.get():
                mass = self.mass_var.get()
                charge = self.charge_var.get()
                if mass > 0 and charge != 0:
                    qm_ratio = charge / mass
                    self.qm_ratio_var.set(qm_ratio)
                    self.qm_ratio_info_var.set(f"computed: {qm_ratio:.3e} C/kg (mass/charge)")
                else:
                    self.qm_ratio_info_var.set("computed: set mass and charge")
            else:
                radius = self.radius_var.get()
                density = self.density_var.get()
                n_charges = self.n_charges_var.get()
                
                if radius > 0 and density > 0:
                    volume = (4/3) * np.pi * radius**3
                    mass = volume * density
                    charge = n_charges * e
                    qm_ratio = charge / mass if mass > 0 else 0
                    
                    self.qm_ratio_var.set(qm_ratio)
                    self.qm_ratio_info_var.set(f"computed: {qm_ratio:.3e} C/kg")
                else:
                    self.qm_ratio_info_var.set("computed: invalid parameters")
        except Exception:
            self.qm_ratio_info_var.set("computed: error")
        finally:
            self._updating_qm_ratio = False

    def build_sweep_tab(self):
        """Parameter sweep tab"""
        tab = self.tab_sweep

        # Create scrollable frame for sweep panel
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Initialize sweep state
        self.sweep_runner = None
        self.sweep_thread = None
        self.sweep_results = None

        # Create sweep config panel
        self.sweep_panel = SweepConfigPanel(
            scrollable_frame,
            on_run=self._on_sweep_run,
            on_stop=self._on_sweep_stop,
            get_base_params=self._get_base_sweep_params
        )
        self.sweep_panel.pack(fill='both', expand=True, padx=10, pady=10)

    def _get_base_sweep_params(self) -> dict:
        """Get base parameters for sweep from current GUI settings."""
        # Gather parameters similar to _collect_settings but for sweep
        params = {}

        # Basic trap parameters
        params['V_rf'] = self.Vrf_var.get()
        params['V_dc'] = self.Vdc_var.get()
        params['Omega'] = 2 * np.pi * self.freq_var.get()
        params['r0'] = self.r0_var.get()

        # Trap geometry
        if self.trap_type.get() == "Custom (a,b,c)":
            params['trap_type'] = None
            params['custom_abc'] = (self.a_var.get(), self.b_var.get(), self.c_var.get())
        else:
            params['trap_type'] = self.trap_type.get()
            params['custom_abc'] = None

        # Particle properties
        params['particle_radius'] = self.radius_var.get()
        params['particle_density'] = self.density_var.get()

        # Charge (handle different modes)
        if self.use_mc_override_var.get():
            params['particle_mass'] = self.mass_var.get()
            params['particle_charge'] = self.charge_var.get()
        else:
            # Use n_charges
            pass  # ParameterSweep will handle this

        # Damping
        if not self.disable_stokes_var.get():
            # Stokes drag enabled - simulator will calculate from particle properties
            pass
        else:
            params['damping_gamma'] = self.damping_gamma_var.get()

        params['enable_gravity'] = self.gravity_var.get()

        # Numeric field settings
        if self.use_numeric_mode:
            numeric_dir = pathlib.Path(getattr(self, "active_numeric_field_dir", self.numeric_field_dir))
            params['numeric_field_dir'] = str(numeric_dir)

            # Get electrode program
            V_rf = self.Vrf_var.get()
            V_dc = self.Vdc_var.get()
            freq_hz = self.freq_var.get()
            params['electrode_program'] = self._resolve_electrode_program(V_rf, V_dc, freq_hz, numeric_dir)

        return params

    def _on_sweep_run(self, config_dict: dict):
        """Handle sweep run button click."""
        if self.sweep_thread and self.sweep_thread.is_alive():
            messagebox.showwarning("Sweep Running", "A sweep is already in progress!")
            return

        # Convert config_dict to SweepConfig
        try:
            param1 = SweepParameter(**config_dict['param1'])
            param2 = SweepParameter(**config_dict['param2']) if config_dict.get('param2') else None

            sweep_config = SweepConfig(
                param1=param1,
                param2=param2,
                duration=config_dict['duration'],
                escape_radius=config_dict['escape_radius'],
                use_secular=config_dict['use_secular'],
                randomize_initial=config_dict['randomize_initial'],
                initial_position=config_dict['initial_position'],
                output_dir=config_dict['output_dir'],
            )

            # Get base params
            base_params = self._get_base_sweep_params()

            # Create sweep runner
            self.sweep_runner = ParameterSweep(
                base_params=base_params,
                numeric_dir=str(self.numeric_field_dir) if self.use_numeric_mode else None,
                use_numeric=self.use_numeric_mode,
                use_grid=self.use_numeric_grid_var.get() if self.use_numeric_mode else False,
                progress_callback=self._on_sweep_progress
            )

            # Start sweep in background thread
            self.sweep_panel.set_running(True)
            self.sweep_thread = threading.Thread(
                target=self._run_sweep_thread,
                args=(sweep_config,),
                daemon=True
            )
            self.sweep_thread.start()

            self.update_status("Sweep started...", 'info')

        except Exception as e:
            messagebox.showerror("Sweep Error", f"Failed to start sweep:\n{e}\n\n{traceback.format_exc()}")
            self.sweep_panel.set_running(False)

    def _on_sweep_stop(self):
        """Handle sweep stop button click."""
        if self.sweep_runner:
            self.sweep_runner.stop()
            self.update_status("Sweep stop requested...", 'warning')

    def _on_sweep_progress(self, current: int, total: int, message: str):
        """Handle sweep progress updates."""
        # This is called from the sweep thread, so we need to use root.after
        # to update GUI from the main thread
        self.root.after(0, self._update_sweep_progress_gui, current, total, message)

    def _update_sweep_progress_gui(self, current: int, total: int, message: str):
        """Update sweep progress in GUI (called from main thread)."""
        if hasattr(self, 'sweep_panel'):
            self.sweep_panel.update_progress(current, total, message)

    def _run_sweep_thread(self, config: SweepConfig):
        """Run sweep in background thread."""
        try:
            results = self.sweep_runner.run(config)
            self.sweep_results = results

            # Notify completion on main thread
            self.root.after(0, self._on_sweep_complete, results)

        except Exception as e:
            error_msg = f"Sweep failed: {e}\n\n{traceback.format_exc()}"
            self.root.after(0, self._on_sweep_error, error_msg)

    def _on_sweep_complete(self, results: dict):
        """Handle sweep completion (called from main thread)."""
        self.sweep_panel.set_running(False)

        sweep_type = results.get('type', '1d')
        summary = results.get('summary', {})
        stopped_early = results.get('stopped_early', False)

        if stopped_early:
            completed_runs = results.get('completed_runs', 0)
            msg = f"Sweep stopped!\n\n"
            msg += f"️ Partial results saved\n"
            msg += f"Type: {sweep_type}\n"
            msg += f"Completed: {completed_runs} runs\n"
            msg += f"Stable: {summary.get('stable_count', 'N/A')} ({summary.get('stable_fraction', 0)*100:.1f}%)\n"
            self.update_status(f"Sweep stopped - partial results saved ({completed_runs} runs)", 'warning')
            title = "Sweep Stopped"
        else:
            msg = f"Sweep complete!\n\n"
            msg += f"Type: {sweep_type}\n"
            msg += f"Total runs: {summary.get('total_runs', 'N/A')}\n"
            msg += f"Stable: {summary.get('stable_count', 'N/A')} ({summary.get('stable_fraction', 0)*100:.1f}%)\n"
            self.update_status(f"Sweep complete: {summary.get('stable_fraction', 0)*100:.1f}% stable", 'success')
            title = "Sweep Complete"

        # Ask if user wants to visualize
        if messagebox.askyesno(title, msg + "\nVisualize results now?"):
            self._visualize_sweep_results(results)

    def _on_sweep_error(self, error_msg: str):
        """Handle sweep error (called from main thread)."""
        self.sweep_panel.set_running(False)
        self.update_status("Sweep failed!", 'error')
        messagebox.showerror("Sweep Error", error_msg)

    def _visualize_sweep_results(self, results: dict):
        """Visualize sweep results in a new window."""
        try:
            # Create new window
            viz_window = tk.Toplevel(self.root)
            viz_window.title("Sweep Results")
            viz_window.geometry("900x700")

            # Create matplotlib figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            from matplotlib.figure import Figure

            fig = Figure(figsize=(10, 8))

            if results['type'] == '1d':
                # 1D sweep: create multiple subplots
                ax1 = fig.add_subplot(2, 1, 1)
                ax2 = fig.add_subplot(2, 1, 2)

                sweep_visualization.plot_1d_sweep(results, metric='stability', ax=ax1)
                sweep_visualization.plot_1d_sweep(results, metric='max_amplitude_r', ax=ax2)
            else:
                # 2D sweep: stability diagram
                ax = fig.add_subplot(1, 1, 1)
                sweep_visualization.plot_stability_diagram(results, ax=ax)

            fig.tight_layout()

            # Embed in Tkinter
            canvas = FigureCanvasTkAgg(fig, master=viz_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)

            toolbar = NavigationToolbar2Tk(canvas, viz_window)
            toolbar.update()

            # Add summary text
            summary_frame = ttk.Frame(viz_window)
            summary_frame.pack(fill='x', padx=10, pady=10)

            summary = results.get('summary', {})
            stopped_early = results.get('stopped_early', False)

            if stopped_early:
                completed_runs = results.get('completed_runs', 0)
                summary_text = f"️ PARTIAL RESULTS | Completed: {completed_runs} runs | "
                summary_text += f"Stable: {summary.get('stable_count', 'N/A')} ({summary.get('stable_fraction', 0)*100:.1f}%)"
                ttk.Label(summary_frame, text=summary_text, font=('', 10, 'bold'), foreground='orange').pack()
            else:
                summary_text = f"Total: {summary.get('total_runs', 'N/A')} | "
                summary_text += f"Stable: {summary.get('stable_count', 'N/A')} ({summary.get('stable_fraction', 0)*100:.1f}%)"
                ttk.Label(summary_frame, text=summary_text, font=('', 10, 'bold')).pack()

        except Exception as e:
            messagebox.showerror("Visualization Error", f"Failed to visualize results:\n{e}\n\n{traceback.format_exc()}")

    def update_status(self, message, level='info'):
        """Add a timestamped message to the status log"""
        self.status_text.configure(state='normal')
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert('end', f"[{timestamp}] {message}\n", level)
        self.status_text.see('end')
        self.status_text.configure(state='disabled')

    def _normalize_setting_value(self, value):
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    def _collect_settings(self):
        settings = {
            "mode": self.mode_var.get(),
            "Vdc": self.Vdc_var.get(),
            "Vrf": self.Vrf_var.get(),
            "freq_hz": self.freq_var.get(),
            "r0": self.r0_var.get(),
            "trap_type": self.trap_type.get(),
            "a": self.a_var.get(),
            "b": self.b_var.get(),
            "c": self.c_var.get(),
            "particle_radius": self.radius_var.get(),
            "particle_density": self.density_var.get(),
            "disable_stokes": self.disable_stokes_var.get(),
            "pressure_torr": self.pressure_torr_var.get(),
            "damping_gamma": self.damping_var.get(),
            "gravity": self.gravity_var.get(),
            "n_charges": self.n_charges_var.get(),
            "use_qm_override": self.use_qm_override_var.get(),
            "qm_ratio": self.qm_ratio_var.get(),
            "use_mc_override": self.use_mc_override_var.get(),
            "mass": self.mass_var.get(),
            "charge": self.charge_var.get(),
            "num_particles": self.num_var.get(),
            "duration": self.duration_var.get(),
            "dt": self.dt_var.get(),
            "fast_single": self.fast_single_var.get(),
            "integration_method": self.integration_method.get(),
            "use_parallel": self.use_parallel_var.get(),
            "cache_size": self.cache_size_var.get(),
            "use_secular": self.use_secular_var.get(),
            "use_numeric_grid": self.use_numeric_grid_var.get(),
            "grid_resolution": self.grid_resolution_var.get(),
            "auto_grid": self.auto_grid_var.get(),
            "grid_resolution_manual": getattr(self, "_last_manual_grid_resolution", self.grid_resolution_var.get()),
            "use_adaptive_grid": self.use_adaptive_grid_var.get(),
            "grid_smoothing": self.grid_smoothing_var.get(),
            "azim": self.azim_var.get(),
            "elev": self.elev_var.get(),
            "show_field_vectors": self.show_field_vectors_var.get(),
            "show_cad_mesh": self.show_cad_mesh_var.get(),
            "show_fft": self.show_fft_var.get(),
            "axis_scale": self.axis_scale_var.get(),
        }

        # Add beam settings
        if hasattr(self, "beam_panel"):
            try:
                beam_params = self.beam_panel.get_parameters()
                settings["beam_enabled"] = beam_params.enabled
                settings["beam_type"] = beam_params.beam_type.value
                settings["beam_current_nA"] = beam_params.current * 1e9  # A to nA
                settings["beam_energy_eV"] = beam_params.energy
                settings["beam_radius_mm"] = beam_params.beam_radius * 1e3  # m to mm
                settings["beam_propagation_axis"] = beam_params.propagation_axis
                settings["beam_center_x_mm"] = beam_params.beam_center[0] * 1e3  # m to mm
                settings["beam_center_y_mm"] = beam_params.beam_center[1] * 1e3  # m to mm
                settings["beam_interaction_strength"] = beam_params.interaction_strength
                settings["beam_phase_mode"] = beam_params.phase_mode.value
                settings["beam_phase_window_pct"] = beam_params.phase_window * 100  # fraction to %
                settings["beam_custom_phase_start_deg"] = np.degrees(beam_params.custom_phase_start)
                settings["beam_custom_phase_end_deg"] = np.degrees(beam_params.custom_phase_end)
                settings["beam_delay_start_ms"] = beam_params.delay_start * 1e3  # s to ms
                settings["beam_pulse_duration_ms"] = beam_params.pulse_duration * 1e3 if not np.isinf(beam_params.pulse_duration) else "inf"
            except Exception as e:
                print(f"[Warning] Failed to collect beam settings: {e}")

        settings = {k: self._normalize_setting_value(v) for k, v in settings.items()}
        paths = {}
        if hasattr(self, "numeric_panel"):
            paths = {
                "electrodes_dir": self.numeric_panel.electrodes_dir_var.get(),
                "numeric_out": self.numeric_panel.numeric_out_var.get(),
            }
        return {"version": 1, "settings": settings, "paths": paths}

    def export_settings(self):
        """Export GUI settings to JSON."""
        data = self._collect_settings()
        try:
            DEFAULT_SETTINGS.mkdir(parents=True, exist_ok=True)
            initial_dir = DEFAULT_SETTINGS
        except Exception:
            initial_dir = PROJECT_ROOT
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(initial_dir),
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=True)
            self.update_status(f"Settings exported to {path}", "success")
        except Exception as exc:
            messagebox.showerror("Export Failed", f"Failed to export settings:\n{exc}")

    def import_settings(self):
        """Import GUI settings from JSON."""
        try:
            DEFAULT_SETTINGS.mkdir(parents=True, exist_ok=True)
            initial_dir = DEFAULT_SETTINGS
        except Exception:
            initial_dir = PROJECT_ROOT
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(initial_dir),
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            messagebox.showerror("Import Failed", f"Failed to load settings:\n{exc}")
            return

        settings = data.get("settings", data)
        paths = data.get("paths", {})

        def set_var(var, key, cast=None):
            if key not in settings:
                return
            value = settings.get(key)
            if value is None:
                return
            if cast:
                try:
                    value = cast(value)
                except Exception:
                    return
            try:
                var.set(value)
            except Exception:
                return

        set_var(self.mode_var, "mode", str)
        set_var(self.Vdc_var, "Vdc", float)
        set_var(self.Vrf_var, "Vrf", float)
        set_var(self.freq_var, "freq_hz", float)
        set_var(self.r0_var, "r0", float)
        set_var(self.trap_type, "trap_type", str)
        set_var(self.a_var, "a", float)
        set_var(self.b_var, "b", float)
        set_var(self.c_var, "c", float)
        set_var(self.radius_var, "particle_radius", float)
        set_var(self.density_var, "particle_density", float)
        set_var(self.disable_stokes_var, "disable_stokes", bool)
        set_var(self.pressure_torr_var, "pressure_torr", float)
        set_var(self.damping_var, "damping_gamma", float)
        set_var(self.gravity_var, "gravity", bool)
        set_var(self.n_charges_var, "n_charges", int)
        set_var(self.use_qm_override_var, "use_qm_override", bool)
        set_var(self.qm_ratio_var, "qm_ratio", float)
        set_var(self.use_mc_override_var, "use_mc_override", bool)
        set_var(self.mass_var, "mass", float)
        set_var(self.charge_var, "charge", float)
        set_var(self.num_var, "num_particles", int)
        set_var(self.duration_var, "duration", float)
        set_var(self.dt_var, "dt", float)
        set_var(self.fast_single_var, "fast_single", bool)
        set_var(self.integration_method, "integration_method", str)
        set_var(self.use_parallel_var, "use_parallel", bool)
        set_var(self.cache_size_var, "cache_size", int)
        set_var(self.use_secular_var, "use_secular", bool)
        set_var(self.use_numeric_grid_var, "use_numeric_grid", bool)
        set_var(self.use_adaptive_grid_var, "use_adaptive_grid", bool)
        set_var(self.grid_smoothing_var, "grid_smoothing", float)
        set_var(self.azim_var, "azim", float)
        set_var(self.elev_var, "elev", float)
        set_var(self.show_field_vectors_var, "show_field_vectors", bool)
        set_var(self.show_cad_mesh_var, "show_cad_mesh", bool)
        set_var(self.show_fft_var, "show_fft", bool)
        set_var(self.axis_scale_var, "axis_scale", float)

        if "grid_resolution_manual" in settings:
            try:
                self._last_manual_grid_resolution = int(settings["grid_resolution_manual"])
            except Exception:
                pass

        if "auto_grid" in settings:
            try:
                self.auto_grid_var.set(bool(settings["auto_grid"]))
            except Exception:
                pass

        grid_res = settings.get("grid_resolution")
        if self.auto_grid_var.get():
            self.grid_resolution_var.set(0)
        elif grid_res is not None:
            try:
                grid_res = int(grid_res)
                if grid_res <= 0:
                    grid_res = getattr(self, "_last_manual_grid_resolution", 100) or 100
                self.grid_resolution_var.set(grid_res)
            except Exception:
                pass
        self.on_auto_grid_toggle()

        if paths and hasattr(self, "numeric_panel"):
            if "electrodes_dir" in paths:
                self.numeric_panel.electrodes_dir_var.set(paths["electrodes_dir"])
            if "numeric_out" in paths:
                self.numeric_panel.numeric_out_var.set(paths["numeric_out"])
            self.numeric_panel.notify_paths_changed()
            self.numeric_panel.update_status()

        # Load beam settings
        if hasattr(self, "beam_panel"):
            try:
                from beam_field import BeamParameters, BeamType, PhaseMode

                # Build beam parameters from settings
                beam_kwargs = {}
                if "beam_enabled" in settings:
                    beam_kwargs["enabled"] = bool(settings.get("beam_enabled", False))
                if "beam_type" in settings:
                    beam_type_str = settings.get("beam_type", "electron")
                    beam_kwargs["beam_type"] = BeamType.ELECTRON if beam_type_str == "electron" else BeamType.ION
                if "beam_current_nA" in settings:
                    beam_kwargs["current"] = float(settings["beam_current_nA"]) * 1e-9
                if "beam_energy_eV" in settings:
                    beam_kwargs["energy"] = float(settings["beam_energy_eV"])
                if "beam_radius_mm" in settings:
                    beam_kwargs["beam_radius"] = float(settings["beam_radius_mm"]) * 1e-3
                if "beam_propagation_axis" in settings:
                    beam_kwargs["propagation_axis"] = str(settings["beam_propagation_axis"])
                if "beam_center_x_mm" in settings and "beam_center_y_mm" in settings:
                    beam_kwargs["beam_center"] = (
                        float(settings["beam_center_x_mm"]) * 1e-3,
                        float(settings["beam_center_y_mm"]) * 1e-3
                    )
                if "beam_interaction_strength" in settings:
                    beam_kwargs["interaction_strength"] = float(settings["beam_interaction_strength"])
                if "beam_phase_mode" in settings:
                    phase_mode_map = {
                        "zero_crossing_both": PhaseMode.ZERO_CROSSING_BOTH,
                        "zero_crossing_positive": PhaseMode.ZERO_CROSSING_POSITIVE,
                        "zero_crossing_negative": PhaseMode.ZERO_CROSSING_NEGATIVE,
                        "continuous": PhaseMode.CONTINUOUS,
                        "custom": PhaseMode.CUSTOM,
                    }
                    phase_mode_str = settings.get("beam_phase_mode", "zero_crossing_both")
                    beam_kwargs["phase_mode"] = phase_mode_map.get(phase_mode_str, PhaseMode.ZERO_CROSSING_BOTH)
                if "beam_phase_window_pct" in settings:
                    beam_kwargs["phase_window"] = float(settings["beam_phase_window_pct"]) / 100.0
                if "beam_custom_phase_start_deg" in settings:
                    beam_kwargs["custom_phase_start"] = np.radians(float(settings["beam_custom_phase_start_deg"]))
                if "beam_custom_phase_end_deg" in settings:
                    beam_kwargs["custom_phase_end"] = np.radians(float(settings["beam_custom_phase_end_deg"]))
                if "beam_delay_start_ms" in settings:
                    beam_kwargs["delay_start"] = float(settings["beam_delay_start_ms"]) * 1e-3
                if "beam_pulse_duration_ms" in settings:
                    duration_val = settings["beam_pulse_duration_ms"]
                    if duration_val == "inf" or duration_val == "infinity":
                        beam_kwargs["pulse_duration"] = np.inf
                    else:
                        beam_kwargs["pulse_duration"] = float(duration_val) * 1e-3

                # Create beam parameters and set them
                if beam_kwargs:
                    # Get current params and update with loaded values
                    current_params = self.beam_panel.get_parameters()
                    for key, value in beam_kwargs.items():
                        setattr(current_params, key, value)
                    self.beam_panel.set_parameters(current_params)
            except Exception as e:
                print(f"[Warning] Failed to load beam settings: {e}")
                # Continue with import even if beam settings fail

        self.on_trap_type_change()
        self.on_qm_override_toggle()
        self.on_mc_override_toggle()
        self.on_stokes_drag_toggle()
        self.on_field_mode_change()
        self.on_axis_scale_change()
        self.update_view()
        self.update_status(f"Settings imported from {path}", "success")
    
    def on_numeric_paths_changed(self, paths):
        """Handle numeric field path changes"""
        electrodes_dir, numeric_dir = paths
        self.electrode_dir_path = pathlib.Path(electrodes_dir)
        self.numeric_field_dir = pathlib.Path(numeric_dir)
        self.update_status(f"Paths updated: {numeric_dir}", 'info')
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self._fullscreen = not self._fullscreen
        self.root.attributes('-fullscreen', self._fullscreen)
        if self._fullscreen:
            self.update_status("Fullscreen enabled (Esc to exit)", 'info')
    
    def exit_fullscreen(self):
        """Exit fullscreen mode"""
        if self._fullscreen:
            self._fullscreen = False
            self.root.attributes('-fullscreen', False)
            self.update_status("Fullscreen disabled", 'info')
    
    def toggle_controls(self):
        """Hide/show control panel to maximize plot area"""
        if self.hide_controls_var.get():
            self.left_panel.grid_remove()
            self._controls_hidden = True
            self.update_status("Controls hidden", 'info')
        else:
            self.left_panel.grid()
            self._controls_hidden = False
            self.update_status("Controls shown", 'info')
    
    def update_view(self):
        """Update 3D plot viewing angle"""
        try:
            self.ax3d.view_init(elev=self.elev_var.get(), azim=self.azim_var.get())
            self.canvas.draw_idle()
        except (AttributeError, tk.TclError):
            return
    
    def set_view(self, preset):
        """Set predefined viewing angles"""
        views = {
            'xy': (90, -90),
            'xz': (0, -90),
            'yz': (0, 0),
            'iso': (30, -60)
        }
        if preset in views:
            elev, azim = views[preset]
            self.elev_var.set(elev)
            self.azim_var.set(azim)
            self.update_view()
    
    def on_field_vectors_toggle(self):
        """Toggle field vector display"""
        # Replot with/without vectors
        if self._last_plot_kind and self._last_plot_payload:
            if self._last_plot_kind == 'single':
                self.update_single_particle_plots(*self._last_plot_payload)
            elif self._last_plot_kind == 'cloud':
                self.update_cloud_plots(*self._last_plot_payload)
    
    def on_axis_scale_change(self):
        """Update axis scale display and apply to plot"""
        scale = 10 ** self.axis_scale_var.get()
        self.axis_scale_display.set(f"{scale:.2g}x")
        
        # Apply scale to current plot
        try:
            self._apply_axis_scale(self.ax3d)
            self.canvas.draw_idle()
        except (AttributeError, TypeError, ValueError, tk.TclError):
            return

    def on_auto_grid_toggle(self):
        """Toggle auto grid resolution."""
        if not hasattr(self, "grid_resolution_spin"):
            return
        if self.auto_grid_var.get():
            current = self.grid_resolution_var.get()
            if current > 0:
                self._last_manual_grid_resolution = current
            self.grid_resolution_var.set(0)
            self.grid_resolution_spin.configure(state="disabled")
            self._update_auto_grid_display()
            try:
                self.update_status("Grid resolution set to auto", "info")
            except Exception:
                pass
        else:
            if self.grid_resolution_var.get() <= 0:
                restore = getattr(self, "_last_manual_grid_resolution", 100) or 100
                self.grid_resolution_var.set(restore)
            self.grid_resolution_spin.configure(state="normal")
            self._update_auto_grid_display()

    def _update_auto_grid_display(self):
        value = self._last_auto_grid_value
        if value is None:
            label = "Auto grid: pending"
        else:
            suffix = "" if self.auto_grid_var.get() else " (manual active)"
            label = f"Auto grid: {value}{suffix}"
        self.auto_grid_display_var.set(label)

    def _set_auto_grid_value(self, value):
        try:
            value = int(value)
        except (TypeError, ValueError):
            return
        if value <= 0:
            return
        self._last_auto_grid_value = value
        self._update_auto_grid_display()
    
    def _apply_axis_scale(self, ax):
        """Apply z-axis scaling to 3D plot"""
        scale = 10 ** self.axis_scale_var.get()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        
        # Scale z-axis
        z_center = (zlim[0] + zlim[1]) / 2
        z_range = zlim[1] - zlim[0]
        new_zlim = (z_center - z_range * scale / 2, z_center + z_range * scale / 2)
        ax.set_zlim(new_zlim)
    
    def pop_out_plot(self):
        """Create a separate window with the current plot"""
        if not hasattr(self, '_last_plot_payload') or self._last_plot_payload is None:
            self.update_status("No plot to pop out", 'warning')
            return
        
        popup = tk.Toplevel(self.root)
        popup.title("Paul Trap - Plot View")
        popup.geometry("1000x700")
        
        # Create new figure
        fig = Figure(figsize=(10, 7))
        ax3d = fig.add_subplot(2, 2, 1, projection='3d')
        ax_x = fig.add_subplot(2, 2, 2)
        ax_y = fig.add_subplot(2, 2, 3)
        ax_z = fig.add_subplot(2, 2, 4)
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=popup)
        toolbar = NavigationToolbar2Tk(canvas, popup)
        toolbar.update()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Copy current plot data to new figure
        if self._last_plot_kind == 'single':
            t, pos, vel = self._last_plot_payload
            self._plot_single_particle_to_axes(ax3d, ax_x, ax_y, ax_z, t, pos, vel)
        elif self._last_plot_kind == 'cloud':
            t, pos = self._last_plot_payload
            self._plot_cloud_to_axes(ax3d, ax_x, ax_y, ax_z, t, pos)
        
        fig.tight_layout()
        canvas.draw()
        
        # Add save button
        button_frame = ttk.Frame(popup)
        button_frame.pack(side='bottom', fill='x', padx=8, pady=8)
        
        def save_figure():
            filepath = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("All files", "*.*")
                ]
            )
            if filepath:
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                self.update_status(f"Saved plot to {filepath}", 'success')
        
        ttk.Button(button_frame, text="Save Figure", command=save_figure).pack(side='left', padx=4)
        ttk.Button(button_frame, text="Close", command=popup.destroy).pack(side='right', padx=4)
        
        self.update_status("Plot popped out to new window", 'info')
    
    # === Simulation methods ===
    
    def on_run(self):
        """Start simulation"""
        if self.running:
            self.update_status("Simulation already running", 'warning')
            return
        
        self.running = True
        self._stop_requested = False
        self.run_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.progress_var.set(0)
        self.progress_label.set("Starting simulation...")
        
        self.update_status("Starting simulation...", 'info')
        
        # Start simulation in separate thread
        self.simulation_thread = threading.Thread(target=self.run_simulation, daemon=True)
        self.simulation_thread.start()
    
    def on_stop(self):
        """Stop running simulation"""
        self._stop_requested = True
        if self.sim:
            self.sim.stop_flag = True
        self.running = False
        self.update_status("Stop requested", 'warning')
    
    def on_sim_end(self):
        """Called when simulation completes"""
        self.run_button.configure(state='normal')
        self.stop_button.configure(state='disabled')
        self.progress_label.set("Complete")
        self.update_status("Simulation complete", 'success')

    def export_data(self):
        """Export simulation data to text file"""
        if not hasattr(self, '_last_plot_payload') or self._last_plot_payload is None:
            messagebox.showwarning("No Data", "No simulation data available to export. Run a simulation first.")
            return

        # Ask user for save location
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ],
            title="Export Simulation Data"
        )

        if not filename:
            return

        try:
            kind = getattr(self, '_last_plot_kind', 'unknown')

            if kind == 'single':
                # Single particle data: t, positions, velocities
                t, positions, velocities = self._last_plot_payload

                # Determine file format from extension
                is_csv = filename.lower().endswith('.csv')
                delimiter = ',' if is_csv else '\t'

                # Create header
                header_lines = [
                    "# Paul Trap Simulation - Single Particle Data",
                    f"# Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"# V_rf = {self.Vrf_var.get():.2f} V",
                    f"# V_dc = {self.Vdc_var.get():.2f} V",
                    f"# Frequency = {self.freq_var.get():.2e} Hz",
                    f"# Duration = {self.duration_var.get():.3e} s",
                    f"# Data points = {len(t)}",
                    "#",
                    "# Columns:",
                    "# time(s), x(m), y(m), z(m), vx(m/s), vy(m/s), vz(m/s)"
                ]

                with open(filename, 'w') as f:
                    # Write header
                    for line in header_lines:
                        f.write(line + '\n')

                    # Write data
                    for i in range(len(t)):
                        row = [
                            t[i],
                            positions[i, 0], positions[i, 1], positions[i, 2],
                            velocities[i, 0], velocities[i, 1], velocities[i, 2]
                        ]
                        f.write(delimiter.join(f"{val:.12e}" for val in row) + '\n')

                messagebox.showinfo("Export Successful", f"Data exported to:\n{filename}\n\n{len(t)} data points written.")

            elif kind == 'cloud':
                # Particle cloud data: t, positions (n_particles x n_timesteps x 3)
                t, positions = self._last_plot_payload
                n_timesteps = len(t)
                n_particles = positions.shape[1] if len(positions.shape) > 2 else 1

                # Determine file format
                is_csv = filename.lower().endswith('.csv')
                delimiter = ',' if is_csv else '\t'

                # Create header
                header_lines = [
                    "# Paul Trap Simulation - Particle Cloud Data",
                    f"# Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"# V_rf = {self.Vrf_var.get():.2f} V",
                    f"# V_dc = {self.Vdc_var.get():.2f} V",
                    f"# Frequency = {self.freq_var.get():.2e} Hz",
                    f"# Duration = {self.duration_var.get():.3e} s",
                    f"# Particles = {n_particles}",
                    f"# Timesteps = {n_timesteps}",
                    "#",
                    "# Columns:",
                    "# time(s), particle_id, x(m), y(m), z(m)"
                ]

                with open(filename, 'w') as f:
                    # Write header
                    for line in header_lines:
                        f.write(line + '\n')

                    # Write data
                    for i in range(n_timesteps):
                        for p in range(n_particles):
                            row = [
                                t[i],
                                p,
                                positions[i, p, 0], positions[i, p, 1], positions[i, p, 2]
                            ]
                            f.write(delimiter.join([
                                f"{row[0]:.12e}",  # time
                                f"{row[1]:d}",      # particle_id
                                f"{row[2]:.12e}",   # x
                                f"{row[3]:.12e}",   # y
                                f"{row[4]:.12e}"    # z
                            ]) + '\n')

                messagebox.showinfo("Export Successful",
                    f"Data exported to:\n{filename}\n\n{n_particles} particles, {n_timesteps} timesteps written.")

            else:
                messagebox.showwarning("Unknown Data Type", f"Cannot export data of type '{kind}'")

        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to export data:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def check_plot_queue(self):
        """Check for plotting updates from simulation thread - THREAD SAFE"""
        try:
            while True:
                plot_data = self.plot_queue.get_nowait()
                
                if plot_data['type'] == 'single':
                    self.update_single_particle_plots(*plot_data['data'])
                elif plot_data['type'] == 'cloud':
                    self.update_cloud_plots(*plot_data['data'])
                elif plot_data['type'] == 'error':
                    messagebox.showerror("Simulation Error", plot_data['message'])
                    self.update_status(f"Error: {plot_data['message']}", 'error')
                elif plot_data['type'] == 'warning':
                    self.update_status(plot_data['message'], 'warning')
                elif plot_data['type'] == 'progress':
                    self.progress_var.set(plot_data['value'])
                    self.progress_label.set(plot_data['label'])
                elif plot_data['type'] == 'auto_grid':
                    self._set_auto_grid_value(plot_data.get('value'))
                
                self.plot_queue.task_done()
        except Empty:
            pass
        finally:
            # Check if simulation finished
            if self.simulation_thread is not None and not self.simulation_thread.is_alive():
                if self.running:
                    self.running = False
                self.on_sim_end()
                self.simulation_thread = None
            self.root.after(100, self.check_plot_queue)
    
    def run_simulation(self):
        #Run sim in separate thread
        try:
            # Create trap simulator
            V_rf = self.Vrf_var.get()
            V_dc = self.Vdc_var.get()
            freq_hz = self.freq_var.get()
            use_numeric = self.use_numeric_mode
            numeric_dir = pathlib.Path(getattr(self, "active_numeric_field_dir", self.numeric_field_dir))
            electrode_program = self._resolve_electrode_program(V_rf, V_dc, freq_hz, numeric_dir)
            props = self._resolve_particle_properties()
            use_secular = self.use_secular_var.get()
            integration_method = self.integration_method.get()
            use_parallel = bool(self.use_parallel_var.get())
            try:
                cache_size = int(self.cache_size_var.get())
            except (TypeError, ValueError):
                cache_size = 50000
            if cache_size <= 0:
                cache_size = 50000

            progress_ranges = {
                "init": (0.0, 5.0),
                "grid": (5.0, 35.0),
                "integrate": (35.0, 90.0),
                "plot": (90.0, 100.0),
            }

            def progress_cb(value, label):
                self.plot_queue.put({'type': 'progress', 'value': value, 'label': label})
                if isinstance(label, str) and label.startswith("Gridfield auto"):
                    parts = label.split()
                    if parts:
                        try:
                            auto_val = int(parts[-1])
                            self.plot_queue.put({'type': 'auto_grid', 'value': auto_val})
                        except (TypeError, ValueError):
                            pass

            progress_cb(progress_ranges["init"][0], "Initializing...")
            stop_check = lambda: self._stop_requested

            # Get beam parameters
            beam_params = None
            if hasattr(self, "beam_panel"):
                try:
                    beam_params = self.beam_panel.get_parameters()
                except Exception as e:
                    print(f"[WARNING] Failed to get beam parameters: {e}")

            self.sim = fields.TrapSimulator(
                V_rf=V_rf,
                V_dc=V_dc,
                Omega=2*np.pi*freq_hz,
                r0=self.r0_var.get(),
                trap_type=self.trap_type.get() if self.trap_type.get() != "Custom (a,b,c)" else None,
                custom_abc=(self.a_var.get(), self.b_var.get(), self.c_var.get()) if self.trap_type.get()=="Custom (a,b,c)" else None,
                particle_radius=self.radius_var.get(),
                particle_density=self.density_var.get(),
                particle_mass=props.get("mass"),
                particle_charge=props.get("charge"),
                qm_ratio=props.get("qm_ratio"),
                stokes_radius=0.0 if self.disable_stokes_var.get() else None,
                pressure_torr=None if self.disable_stokes_var.get() else self.pressure_torr_var.get(),
                enable_gravity=self.gravity_var.get(),
                axial_dc_kappa=0.3 if self.trap_type.get()=="Linear" else None,
                axial_dc_z0=3.0e-3 if self.trap_type.get()=="Linear" else None,
                axial_dc_voltage=V_dc if self.trap_type.get()=="Linear" else None,
                use_numeric=use_numeric,
                numeric_field_dir=str(numeric_dir),
                electrode_program=electrode_program,
                damping_gamma=self.damping_var.get(),
                use_optimized=True,  # ADD THIS LINE to use optimized field
                use_secular=use_secular,
                cache_size=cache_size,
                use_numeric_grid=bool(self.use_numeric_grid_var.get()),
                numeric_grid_points=self.grid_resolution_var.get(),
                adaptive_grid_refinement=bool(self.use_adaptive_grid_var.get()),
                numeric_grid_smoothing=self.grid_smoothing_var.get(),
                beam_params=beam_params,
                progress_callback=progress_cb,
                progress_ranges=progress_ranges,
                stop_check=stop_check,
            )

            
            t_span = (0, self.duration_var.get())
            dt_val = self.dt_var.get()
            n_particles = self.num_var.get()
            charge_kwargs = {}
            
            if self.use_mc_override_var.get():
                charge_c = props.get("charge")
                qm_ratio = props.get("qm_ratio")
                if charge_c is not None:
                    charge_kwargs["charge_c"] = charge_c
                if qm_ratio is not None:
                    charge_kwargs["qm_ratio"] = qm_ratio
                if not charge_kwargs:
                    charge_kwargs["n_charges"] = self.n_charges_var.get()
            elif self.use_qm_override_var.get():
                qm_ratio = props.get("qm_ratio")
                charge_c = props.get("charge")
                if qm_ratio is not None:
                    charge_kwargs["qm_ratio"] = qm_ratio
                if charge_c is not None:
                    charge_kwargs["charge_c"] = charge_c
                if not charge_kwargs:
                    charge_kwargs["n_charges"] = self.n_charges_var.get()
            else:
                charge_kwargs["n_charges"] = self.n_charges_var.get()
            
            if n_particles == 1:
                rng = np.random.default_rng()
                if use_numeric and getattr(self.sim, "numeric", None) is not None:
                    initial_position = self._pick_numeric_start_position(self.sim, rng, numeric_dir=numeric_dir)
                else:
                    r0 = float(self.r0_var.get())
                    base_offset = np.array([0.0, 0.0, 2.0 * r0], dtype=float)
                    jitter_scale = 0.05 * r0
                    if not np.all(np.isfinite(base_offset)):
                        base_offset = np.array([0.0, 0.0, 2.0 * float(self.r0_var.get())], dtype=float)
                        jitter_scale = 0.0
                    initial_position = base_offset + rng.normal(scale=jitter_scale, size=3)
                
                t, pos, vel = self.sim.sim_single_particle(
                    initial_position=initial_position,
                    initial_velocity=np.array([0, 0, 0]),
                    t_span=t_span,
                    **charge_kwargs,
                    use_secular=use_secular,
                    dt=dt_val,
                    use_fixed_step=self.fast_single_var.get(),
                    integration_method=integration_method,
                )
                progress_cb(progress_ranges["plot"][0], "Plotting...")
                self.plot_queue.put({'type': 'single', 'data': (t, pos, vel)})
                stop_reason = getattr(self.sim, "_last_stop_reason", None)
                if stop_reason:
                    self.plot_queue.put({'type': 'warning', 'message': f"Simulation stopped early: {stop_reason}"})
            else:
                cloud_radius = 1e-4
                temp = 300
                
                t, pos = self.sim.sim_particle_cloud(
                    n_particles=n_particles,
                    cloud_radius=cloud_radius,
                    t_span=t_span,
                    temp=temp,
                    **charge_kwargs,
                    use_secular=use_secular,
                    dt=dt_val,
                    integration_method=integration_method,
                    use_parallel=use_parallel,
                )

                progress_cb(progress_ranges["plot"][0], "Plotting...")
                self.plot_queue.put({'type': 'cloud', 'data': (t, pos)})
                stop_reason = getattr(self.sim, "_last_stop_reason", None)
                if stop_reason:
                    self.plot_queue.put({'type': 'warning', 'message': f"Simulation stopped early: {stop_reason}"})
            
            progress_cb(100.0, "Complete")
            
        except Exception as e:
            if "Simulation stopped by user" in str(e):
                self.plot_queue.put({'type': 'warning', 'message': "Simulation stopped by user"})
                self.plot_queue.put({'type': 'progress', 'value': 0, 'label': 'Stopped'})
                return
            import traceback
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            self.plot_queue.put({'type': 'error', 'message': error_msg})
            self.plot_queue.put({'type': 'progress', 'value': 0, 'label': 'Error'})

    def _pick_numeric_start_position(self, sim, rng, numeric_dir=None):
        """
        Pick a good starting position for a particle in the numeric field.
        Uses the trap center estimated from the field structure.
        """
        # Try to get trap center from grid field if available
        grid_field = getattr(sim, 'field_solver', None)
        if grid_field is not None and hasattr(grid_field, 'trap_center_z'):
            trap_z = grid_field.trap_center_z
        else:
            # Default: estimate from field sampling or use 1.5mm (typical for surface traps)
            trap_z = self._estimate_trapping_height(sim, numeric_dir or self.numeric_field_dir)
            if trap_z is None or not np.isfinite(trap_z):
                trap_z = 1.5e-3  # 1.5 mm default

        # Start at trap center with minimal jitter
        position = np.array([0.0, 0.0, trap_z])

        # Very small jitter to keep particle near trap center
        jitter_xy = 5e-6   # 5 μm in x,y
        jitter_z = 2e-6    # 2 μm in z
        position[0] += rng.normal(scale=jitter_xy)
        position[1] += rng.normal(scale=jitter_xy)
        position[2] += rng.normal(scale=jitter_z)

        print(f"[Simulation] Starting position: ({position[0]*1e6:.1f}, {position[1]*1e6:.1f}, {position[2]*1e6:.1f}) μm")
        print(f"[Simulation] Trap center estimated at z = {trap_z*1e6:.0f} μm")
        return position

    def _estimate_trapping_height(self, sim, numeric_dir, elec_bounds=None):
        """
        Estimate trapping height by sampling |E| along the z-axis and picking the minimum.
        This mirrors the numeric sampling approach used in trap depth analysis.
        """
        if elec_bounds is None:
            elec_bounds = self._get_electrode_bounds(numeric_dir)
        if elec_bounds is None:
            return None
        elec_min, elec_max, elec_center = elec_bounds
        z_span = float(elec_max[2] - elec_min[2])
        if not np.isfinite(z_span) or z_span <= 0:
            return None
        surface_like = elec_min[2] >= -0.1 * z_span
        if surface_like:
            z_min = elec_max[2] + max(0.05 * z_span, 1e-5)
            z_max = elec_max[2] + max(2.0 * z_span, 1e-3)
        else:
            z_min = elec_min[2] + 0.1 * z_span
            z_max = elec_max[2] - 0.1 * z_span
        if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
            return None
        zs = np.linspace(z_min, z_max, 41)
        x0 = np.full_like(zs, elec_center[0])
        y0 = np.full_like(zs, elec_center[1])
        try:
            Ex, Ey, Ez = sim.field(x0, y0, zs, 0.0)
        except Exception:
            return None
        E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
        E_mag = np.where(np.isfinite(E_mag), E_mag, np.inf)
        idx = int(np.argmin(E_mag))
        if not np.isfinite(E_mag[idx]) or np.isinf(E_mag[idx]):
            return None
        return float(zs[idx])

    def _find_trap_center_3d(self, sim, numeric_dir):
        """
        Find the trap center (equilibrium position) by searching for minimum |E| in 3D.
        Works for any trap geometry (surface, linear, 3D Paul trap, etc.)
        """
        # Get electrode bounds to determine search region
        elec_bounds = self._get_electrode_bounds(numeric_dir)
        if elec_bounds is None:
            return None

        elec_min, elec_max, elec_center = elec_bounds
        span = elec_max - elec_min

        # Determine if this looks like a surface trap (electrodes mostly at one z level)
        z_span = float(span[2])
        surface_like = (elec_min[2] >= -0.1 * z_span) if z_span > 0 else False

        if surface_like:
            # Surface trap: search above the electrode plane
            x_center = float(elec_center[0])
            y_center = float(elec_center[1])
            z_min = float(elec_max[2]) + max(0.05 * z_span, 1e-5)
            z_max = float(elec_max[2]) + max(2.0 * z_span, 1e-3)
        else:
            # 3D trap: search within the electrode volume
            x_center = float(elec_center[0])
            y_center = float(elec_center[1])
            z_min = float(elec_min[2]) + 0.1 * z_span
            z_max = float(elec_max[2]) - 0.1 * z_span

        # First pass: coarse 1D scan along z to find approximate z
        n_coarse = 21
        zs = np.linspace(z_min, z_max, n_coarse)
        xs = np.full_like(zs, x_center)
        ys = np.full_like(zs, y_center)

        try:
            Ex, Ey, Ez = sim.field(xs, ys, zs, 0.0)
            E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
            E_mag = np.where(np.isfinite(E_mag), E_mag, np.inf)
            z_best_idx = int(np.argmin(E_mag))
            z_best = zs[z_best_idx]
        except Exception:
            # Fallback to center of search region
            z_best = (z_min + z_max) / 2

        # Second pass: refine in x, y around z_best (for non-surface traps)
        if not surface_like:
            # Search in x-y plane at z_best
            x_range = max(0.2 * float(span[0]), 1e-4)
            y_range = max(0.2 * float(span[1]), 1e-4)

            n_xy = 11
            x_search = np.linspace(x_center - x_range, x_center + x_range, n_xy)
            y_search = np.linspace(y_center - y_range, y_center + y_range, n_xy)

            best_E = float('inf')
            best_x, best_y = x_center, y_center

            for xi in x_search:
                xs_line = np.full(n_xy, xi)
                zs_line = np.full(n_xy, z_best)
                try:
                    Ex, Ey, Ez = sim.field(xs_line, y_search, zs_line, 0.0)
                    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
                    for j, E in enumerate(E_mag):
                        if np.isfinite(E) and E < best_E:
                            best_E = E
                            best_x = xi
                            best_y = y_search[j]
                except Exception:
                    continue

            x_center, y_center = best_x, best_y

        return np.array([x_center, y_center, z_best])

    def _pick_numeric_start_position_old_complex(self, sim, rng, numeric_dir=None):
        """DEPRECATED: Overly complex version - kept for reference"""
        centroids = getattr(sim.numeric, "centroids", None)
        nodes = getattr(sim.numeric, "nodes", None)
        points = centroids if centroids is not None and centroids.size else nodes
        if points is None or not points.size:
            return np.zeros(3)

        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = (mins + maxs) / 2.0
        span = maxs - mins
        span_max = float(np.max(span)) if np.all(np.isfinite(span)) else 0.0

        if not np.all(np.isfinite(center)):
            return np.zeros(3)

        max_samples = 2000
        if points.shape[0] > max_samples:
            sample_idx = rng.choice(points.shape[0], size=max_samples, replace=False)
            candidates = points[sample_idx]
        else:
            candidates = points

        if numeric_dir is None:
            numeric_dir = self.numeric_field_dir

        use_electrode_bounds = False
        elec_center = center
        target_z = center[2]
        xy_radius = None
        z_window = None

        electrode_bounds = self._get_electrode_bounds(numeric_dir)
        if electrode_bounds is not None:
            elec_min, elec_max, elec_center = electrode_bounds
            elec_span = elec_max - elec_min
            z_span = max(float(elec_span[2]), 0.05 * float(span[2]), 1e-9)
            z_margin = 0.05 * span[2] if span[2] > 0 else 0.0
            target_z = elec_max[2] + 0.2 * z_span
            if span[2] > 0:
                target_z = min(maxs[2] - z_margin, max(mins[2] + z_margin, target_z))
            z_window = 0.08 * z_span

            xy_span = max(float(elec_span[0]), float(elec_span[1]))
            if not np.isfinite(xy_span) or xy_span <= 0:
                xy_span = max(float(span[0]), float(span[1]))
            xy_radius = 0.12 * xy_span if xy_span > 0 else 0.12 * span_max

            z_mask = np.abs(candidates[:, 2] - target_z) <= z_window
            if np.isfinite(xy_radius) and xy_radius > 0:
                xy_dist2 = np.sum((candidates[:, :2] - elec_center[:2]) ** 2, axis=1)
                mask = z_mask & (xy_dist2 <= xy_radius ** 2)
            else:
                mask = z_mask

            if np.any(mask):
                candidates = candidates[mask]
            use_electrode_bounds = True
        else:
            inner_min = center - 0.3 * span
            inner_max = center + 0.3 * span
            mask = np.all((candidates >= inner_min) & (candidates <= inner_max), axis=1)
            if np.any(mask):
                candidates = candidates[mask]

        t_dc = np.pi / (2.0 * sim.Omega) if np.isfinite(sim.Omega) and sim.Omega != 0.0 else 0.0
        Ex0, Ey0, Ez0 = sim.field(candidates[:, 0], candidates[:, 1], candidates[:, 2], 0.0)
        Exd, Eyd, Ezd = sim.field(candidates[:, 0], candidates[:, 1], candidates[:, 2], t_dc)
        Ex_rf = Ex0 - Exd
        Ey_rf = Ey0 - Eyd
        Ez_rf = Ez0 - Ezd
        E_sq = Ex_rf**2 + Ey_rf**2 + Ez_rf**2
        if not np.all(np.isfinite(E_sq)):
            E_sq = np.where(np.isfinite(E_sq), E_sq, np.inf)

        if use_electrode_bounds and xy_radius is not None and z_window is not None:
            xy_dist2 = np.sum((candidates[:, :2] - elec_center[:2]) ** 2, axis=1)
            z_dist2 = (candidates[:, 2] - target_z) ** 2
            E_scale = np.nanmedian(E_sq) if np.any(np.isfinite(E_sq)) else 1.0
            E_scale = max(E_scale, 1e-30)
            xy_scale = max(xy_radius ** 2, 1e-18)
            z_scale = max(z_window ** 2, 1e-18)
            cost = (E_sq / E_scale) + 12.0 * (xy_dist2 / xy_scale) + (z_dist2 / z_scale)
            pick_idx = int(np.argmin(cost)) if cost.size else 0
        else:
            pick_idx = int(np.argmin(E_sq)) if E_sq.size else 0
        base_offset = candidates[pick_idx].astype(float)

        if use_electrode_bounds and xy_radius is not None and z_window is not None:
            jitter_xy = 0.03 * xy_radius if xy_radius > 0 else 0.0
            jitter_z = 0.08 * z_window if z_window > 0 else 0.0
            initial_position = base_offset.copy()
            if jitter_xy > 0:
                initial_position[:2] += rng.normal(scale=jitter_xy, size=2)
            if jitter_z > 0:
                initial_position[2] += rng.normal(scale=jitter_z, size=1)
            if xy_radius and xy_radius > 0:
                vec_xy = initial_position[:2] - elec_center[:2]
                vec_norm = np.hypot(vec_xy[0], vec_xy[1])
                if vec_norm > xy_radius:
                    initial_position[:2] = elec_center[:2] + (vec_xy / max(vec_norm, 1e-12)) * (0.9 * xy_radius)
        else:
            jitter_scale = 0.002 * span_max if span_max > 0 else 0.0
            initial_position = base_offset + rng.normal(scale=jitter_scale, size=3)

        escape_radius = getattr(sim, "escape_radius", None)
        if escape_radius is not None and escape_radius > 0:
            if np.any(np.abs(initial_position) > escape_radius):
                initial_position = base_offset

        return initial_position

    def _get_electrode_bounds(self, numeric_dir):
        key = str(numeric_dir)
        if key in self._electrode_bounds_cache:
            return self._electrode_bounds_cache[key]
        bounds = self._compute_electrode_bounds_from_mesh(numeric_dir)
        self._electrode_bounds_cache[key] = bounds
        return bounds

    def _compute_electrode_bounds_from_mesh(self, numeric_dir):
        numeric_dir = pathlib.Path(numeric_dir)
        mesh_path = numeric_dir / "mesh.msh"
        meta_path = numeric_dir / "facet_names.json"

        if not mesh_path.exists():
            return None

        unit_scale = 1.0
        electrode_tags = set()
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    unit_scale = meta.get("unit_scale_to_m", 1.0)
                    if "electrodes" in meta:
                        for _, tag in meta["electrodes"].items():
                            electrode_tags.add(tag)
            except Exception:
                unit_scale = 1.0

        try:
            mesh = meshio.read(str(mesh_path))
        except Exception:
            return None

        if "triangle" not in mesh.cells_dict:
            return None

        tris = mesh.cells_dict["triangle"]
        subset_tris = tris

        phys_tags = None
        if "gmsh:physical" in mesh.cell_data_dict:
            if "triangle" in mesh.cell_data_dict.get("gmsh:physical", {}):
                phys_tags = mesh.cell_data_dict["gmsh:physical"]["triangle"]
            else:
                for i, cell_block in enumerate(mesh.cells):
                    if cell_block.type == "triangle":
                        if i < len(mesh.cell_data.get("gmsh:physical", [])):
                            phys_tags = mesh.cell_data["gmsh:physical"][i]
                        break

        if phys_tags is not None and electrode_tags:
            mask = np.isin(phys_tags, list(electrode_tags))
            subset_tris = tris[mask]

        if subset_tris.size == 0:
            return None

        points = mesh.points * unit_scale

        edges = np.vstack([
            subset_tris[:, [0, 1]],
            subset_tris[:, [1, 2]],
            subset_tris[:, [2, 0]],
        ])
        edges_sorted = np.sort(edges, axis=1)
        unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
        boundary_edges = unique_edges[counts == 1]
        if boundary_edges.size == 0:
            boundary_edges = unique_edges

        if boundary_edges.size == 0:
            return None

        edge_points = points[boundary_edges].reshape(-1, 3)
        if not np.all(np.isfinite(edge_points)):
            return None

        elec_min = edge_points.min(axis=0)
        elec_max = edge_points.max(axis=0)
        elec_center = (elec_min + elec_max) / 2.0

        return elec_min, elec_max, elec_center
    
    def _resolve_particle_properties(self):
        """Get particle properties for simulation"""
        props = {}
        if self.use_mc_override_var.get():
            mass = self.mass_var.get()
            charge = self.charge_var.get()
            qm_ratio = self.qm_ratio_var.get() if self.use_qm_override_var.get() else 0.0
            have_mass = mass > 0
            have_charge = charge != 0
            have_qm = qm_ratio > 0

            if have_mass and have_charge:
                props["mass"] = mass
                props["charge"] = charge
                props["qm_ratio"] = charge / mass
            elif have_mass and have_qm:
                props["mass"] = mass
                props["qm_ratio"] = qm_ratio
                props["charge"] = qm_ratio * mass
            elif have_charge and have_qm:
                props["charge"] = charge
                props["qm_ratio"] = qm_ratio
                props["mass"] = charge / qm_ratio
            else:
                raise ValueError("Mass/charge override enabled but insufficient values (need mass+charge or mass+q/m or charge+q/m).")
        elif self.use_qm_override_var.get():
            qm_ratio = self.qm_ratio_var.get()
            if qm_ratio > 0:
                props["qm_ratio"] = qm_ratio
                radius = self.radius_var.get()
                density = self.density_var.get()
                if radius > 0 and density > 0:
                    volume = (4/3) * np.pi * radius**3
                    mass = volume * density
                    props["mass"] = mass
                    props["charge"] = qm_ratio * mass
        return props
    
    def _resolve_electrode_program(self, V_rf, V_dc, freq_hz, numeric_dir):
        """Resolve electrode voltage program"""
        if not self.use_numeric_mode:
            return {}
        return _build_electrode_program_from_numeric(numeric_dir, V_rf, V_dc, freq_hz)

    def check_trap_depth(self):
        """Calculate and display trap depth analysis"""
        try:
            # Get current parameters
            V_rf = self.Vrf_var.get()
            V_dc = self.Vdc_var.get()
            freq_hz = self.freq_var.get()
            Omega = 2 * np.pi * freq_hz

            # Get particle properties - check for mass/charge override
            e = 1.602e-19
            g = 9.81

            if self.use_mc_override_var.get():
                # Use direct mass and charge values
                m = self.mass_var.get()
                Q = self.charge_var.get()

                if m <= 0:
                    messagebox.showwarning("Invalid Mass", "Mass override is enabled but mass is zero or negative!")
                    return
                if Q == 0:
                    messagebox.showwarning("No Charge", "Charge override is enabled but charge is zero!")
                    return
            else:
                # Calculate from radius, density, and number of charges
                particle_radius = self.radius_var.get()
                particle_density = self.density_var.get()
                n_charges = self.n_charges_var.get()

                m = (4/3) * np.pi * particle_radius**3 * particle_density
                Q = n_charges * e

                if Q == 0:
                    messagebox.showwarning("No Charge", "Particle has zero charge! Set 'Elementary charges' > 0.")
                    return

            # Build result message
            result = []
            result.append("=" * 60)
            result.append("TRAP DEPTH ANALYSIS")
            result.append("=" * 60)
            result.append("")
            result.append("Particle Properties:")
            if self.use_mc_override_var.get():
                result.append("  (Using mass/charge override)")
            result.append(f"  Mass: {m:.3e} kg")
            result.append(f"  Charge: {Q:.3e} C ({Q/e:.1f} elementary charges)")
            result.append(f"  q/m ratio: {Q/m:.3e} C/kg")
            result.append("")
            result.append("Trap Parameters:")
            # Display frequency in appropriate units
            if freq_hz >= 1e6:
                result.append(f"  RF frequency: {freq_hz/1e6:.2f} MHz")
            elif freq_hz >= 1e3:
                result.append(f"  RF frequency: {freq_hz/1e3:.2f} kHz")
            else:
                result.append(f"  RF frequency: {freq_hz:.2f} Hz")
            result.append(f"  RF amplitude: {V_rf} V")
            result.append(f"  DC voltage: {V_dc} V")
            result.append("")

            # === MATHIEU STABILITY ANALYSIS ===
            result.append("=" * 60)
            result.append("MATHIEU STABILITY ANALYSIS")
            result.append("=" * 60)
            result.append("")

            # Get characteristic trap size - different approach for numeric vs analytic
            r0 = None
            r0_source = "unknown"

            if self.use_numeric_mode:
                # For CAD/numeric: estimate r0 from field gradient (most accurate)
                # r0 ≈ √(V_rf / (∂E/∂r)) where ∂E/∂r is the field gradient at trap center
                numeric_dir = pathlib.Path(getattr(self, "active_numeric_field_dir", self.numeric_field_dir))

                try:
                    # Create minimal simulator to sample field gradient
                    electrode_program = self._resolve_electrode_program(V_rf, V_dc, freq_hz, numeric_dir)
                    temp_sim_r0 = fields.TrapSimulator(
                        V_rf=V_rf, V_dc=V_dc, Omega=Omega,
                        r0=1e-3,  # Placeholder, not used for numeric
                        particle_radius=1e-9, particle_density=1000, particle_charge=Q,
                        use_numeric=True, numeric_field_dir=str(numeric_dir),
                        electrode_program=electrode_program, use_numeric_grid=False,
                    )

                    # Find trap center first
                    trap_center_r0 = self._find_trap_center_3d(temp_sim_r0, numeric_dir)
                    if trap_center_r0 is None:
                        trap_center_r0 = np.array([0.0, 0.0, 1e-3])

                    # Sample field gradient by finite difference
                    delta = 50e-6  # 50 µm step
                    gradients = []
                    for axis in range(3):
                        pos_plus = trap_center_r0.copy()
                        pos_minus = trap_center_r0.copy()
                        pos_plus[axis] += delta
                        pos_minus[axis] -= delta

                        E_plus = temp_sim_r0.field(
                            np.array([pos_plus[0]]), np.array([pos_plus[1]]), np.array([pos_plus[2]]), 0.0
                        )
                        E_minus = temp_sim_r0.field(
                            np.array([pos_minus[0]]), np.array([pos_minus[1]]), np.array([pos_minus[2]]), 0.0
                        )

                        E_plus_mag = np.sqrt(E_plus[0][0]**2 + E_plus[1][0]**2 + E_plus[2][0]**2)
                        E_minus_mag = np.sqrt(E_minus[0][0]**2 + E_minus[1][0]**2 + E_minus[2][0]**2)
                        grad = abs(E_plus_mag - E_minus_mag) / (2 * delta)
                        if np.isfinite(grad) and grad > 0:
                            gradients.append(grad)

                    if gradients:
                        # Use average gradient to estimate r0
                        # For quadrupole: E = V·r/r₀², so ∂E/∂r = V/r₀², thus r₀ = √(V/(∂E/∂r))
                        avg_gradient = np.mean(gradients)
                        r0 = np.sqrt(V_rf / avg_gradient)
                        r0_source = "field gradient at trap center"
                except Exception as ex:
                    pass

                # Fallback to geometric estimate if gradient method failed
                if r0 is None or r0 <= 0 or not np.isfinite(r0):
                    try:
                        elec_bounds = self._get_electrode_bounds(numeric_dir)
                        if elec_bounds is not None:
                            elec_min, elec_max, elec_center = elec_bounds
                            half_spans = (elec_max - elec_min) / 2
                            r0 = float(np.min(half_spans[half_spans > 1e-6]))
                            r0_source = "electrode geometry (half-span, approximate)"
                    except Exception:
                        pass

                if r0 is None or r0 <= 0 or not np.isfinite(r0):
                    r0 = self.r0_var.get()
                    r0_source = "GUI r0 field (fallback)"
            else:
                # For analytic traps: use the r0 parameter directly
                r0 = self.r0_var.get()
                r0_source = "analytic trap parameter"

            if r0 <= 0:
                r0 = 1e-3  # Default 1mm
                r0_source = "default (1mm)"

            # Calculate Mathieu parameters
            # q = 4·|Q|·V_rf / (m·Ω²·r₀²)  - RF stability parameter
            # a = -8·Q·V_dc / (m·Ω²·r₀²)   - DC stability parameter
            q_mathieu = 4 * abs(Q) * V_rf / (m * Omega**2 * r0**2)
            a_mathieu = -8 * Q * V_dc / (m * Omega**2 * r0**2) if V_dc != 0 else 0.0

            # Stability boundary (approximate)
            q_max = 0.908  # First stability region boundary

            result.append(f"Characteristic size r0: {r0*1e3:.2f} mm")
            result.append(f"  (estimated from: {r0_source})")
            if self.use_numeric_mode and "field gradient" in r0_source:
                result.append("  (r0 = √(V_rf / field_gradient) - accurate for quadrupole-like fields)")
            elif self.use_numeric_mode:
                result.append("  (Note: r0 estimate for CAD may have ±50% uncertainty)")
            result.append(f"Angular frequency Ω: {Omega:.2f} rad/s")
            result.append("")
            result.append("Mathieu parameters:")
            result.append(f"  q = {q_mathieu:.4f}  (stability requires q < {q_max})")
            if V_dc != 0:
                result.append(f"  a = {a_mathieu:.4f}")

            # Add caveat for numeric mode
            if self.use_numeric_mode:
                result.append("")
                result.append("  CAVEAT: Mathieu parameters assume ideal quadrupole fields.")
                result.append("  For CAD/numeric mode, these are approximate estimates only.")
                result.append("  The actual field geometry may deviate significantly from")
                result.append("  quadrupole symmetry, making these parameters less meaningful.")
                result.append("  Consider the pseudopotential depth analysis below as")
                result.append("  a more reliable indicator of trap performance.")

            result.append("")

            # Check stability
            mathieu_stable = True
            if q_mathieu > q_max:
                mathieu_stable = False
                result.append("*** UNSTABLE: q parameter exceeds stability limit! ***")
                result.append("")
                result.append(f"Your q = {q_mathieu:.2f} is {q_mathieu/q_max:.0f}x above the limit.")
                result.append("")
                result.append("The particle CANNOT be trapped in this regime.")
                result.append("The pseudopotential approximation does not apply.")
                result.append("")
                result.append("═" * 60)
                result.append("RECOMMENDATIONS TO ACHIEVE STABILITY")
                result.append("═" * 60)
                result.append("")

                # Calculate required changes for different target q values
                q_safe = 0.4  # Safe middle of stability region
                q_max_safe = 0.7  # Upper limit for robust trapping

                freq_safe = freq_hz * np.sqrt(q_mathieu / q_safe)
                freq_limit = freq_hz * np.sqrt(q_mathieu / q_max)
                v_safe = V_rf * (q_safe / q_mathieu)
                v_limit = V_rf * (q_max / q_mathieu)

                result.append("Priority 1: INCREASE FREQUENCY (Recommended)")
                result.append("-" * 50)
                result.append(f"  Current: {freq_hz:.0f} Hz, Need: {freq_limit:.0f} Hz minimum")
                if freq_limit >= 1e6:
                    result.append(f"           ({freq_limit/1e6:.2f} MHz minimum)")
                elif freq_limit >= 1e3:
                    result.append(f"           ({freq_limit/1e3:.2f} kHz minimum)")
                result.append(f"  Suggested: {freq_safe:.0f} Hz (targets q=0.4, safer)")
                if freq_safe >= 1e6:
                    result.append(f"            ({freq_safe/1e6:.2f} MHz)")
                elif freq_safe >= 1e3:
                    result.append(f"            ({freq_safe/1e3:.2f} kHz)")
                result.append(f"  → Change frequency by {freq_safe/freq_hz:.1f}x")
                result.append("   Keeps voltage same (no electrical changes)")
                result.append("   Improves trap depth (scales with f²)")
                result.append("   Requires higher frequency electronics")
                result.append("")

                result.append("Priority 2: REDUCE RF VOLTAGE")
                result.append("-" * 50)
                result.append(f"  Current: {V_rf} V, Need: ≤ {v_limit:.1f} V minimum")
                result.append(f"  Suggested: {v_safe:.1f} V (targets q=0.4)")
                result.append(f"  → Reduce voltage by {v_safe/V_rf:.2f}x")
                result.append("   Easy to implement (just turn down voltage)")
                result.append("   WEAKENS trap depth by {:.1f}x".format((v_safe/V_rf)**2))
                result.append("   May not overcome gravity!")
                result.append("")

                # Calculate required r0 change
                r0_needed = r0 * np.sqrt(q_mathieu / q_safe)
                result.append("Priority 3: INCREASE r0 (electrode spacing)")
                result.append("-" * 50)
                result.append(f"  Current: {r0*1e3:.2f} mm, Suggested: {r0_needed*1e3:.2f} mm")
                result.append(f"  → Increase r0 by {r0_needed/r0:.2f}x")
                result.append("   Keeps stability (q scales with 1/r0²)")
                result.append("   Requires physical redesign of trap")
                result.append("   Weakens trap depth")
                result.append("")

                # Calculate required charge change
                q_over_m_needed = (Q/m) * (q_safe / q_mathieu)
                result.append("Priority 4: ADJUST PARTICLE PROPERTIES")
                result.append("-" * 50)
                result.append(f"  Current Q/m: {Q/m:.3e} C/kg")
                result.append(f"  Need Q/m ≤ {q_over_m_needed:.3e} C/kg")
                result.append(f"  → Reduce Q/m by {q_safe/q_mathieu:.2f}x")
                result.append("  Options:")
                result.append(f"    • Use heavier particles (mass × {q_mathieu/q_safe:.2f})")
                result.append(f"    • Reduce charge (× {q_safe/q_mathieu:.2f})")
                result.append("   Can work with existing trap hardware")
                result.append("   May limit experimental flexibility")
                result.append("")

                result.append("═" * 60)
                result.append("RECOMMENDED PARAMETER SETS TO TRY")
                result.append("═" * 60)
                result.append("")
                result.append("Option A: Increase frequency (best for trap depth)")
                if freq_safe >= 1e6:
                    result.append(f"  Freq = {freq_safe/1e6:.2f} MHz, Vrf = {V_rf} V")
                elif freq_safe >= 1e3:
                    result.append(f"  Freq = {freq_safe/1e3:.2f} kHz, Vrf = {V_rf} V")
                else:
                    result.append(f"  Freq = {freq_safe:.0f} Hz, Vrf = {V_rf} V")
                result.append(f"  → q = 0.4, trap depth × {(freq_safe/freq_hz)**2:.1f}")
                result.append("")
                result.append("Option B: Reduce voltage (easiest to implement)")
                result.append(f"  Freq = {freq_hz:.0f} Hz, Vrf = {v_safe:.1f} V")
                result.append(f"  → q = 0.4, trap depth × {(v_safe/V_rf)**2:.2f}")
                result.append("   WARNING: Trap will be much weaker!")
                result.append("")
                result.append("Option C: Balanced (moderate changes)")
                freq_balanced = freq_hz * np.sqrt(np.sqrt(q_mathieu / q_safe))  # sqrt to share change
                v_balanced = V_rf * np.sqrt(q_safe / q_mathieu)
                if freq_balanced >= 1e6:
                    result.append(f"  Freq = {freq_balanced/1e6:.2f} MHz, Vrf = {v_balanced:.1f} V")
                elif freq_balanced >= 1e3:
                    result.append(f"  Freq = {freq_balanced/1e3:.2f} kHz, Vrf = {v_balanced:.1f} V")
                else:
                    result.append(f"  Freq = {freq_balanced:.0f} Hz, Vrf = {v_balanced:.1f} V")
                result.append(f"  → q = 0.4, trap depth × {(freq_balanced/freq_hz)**2 * (v_balanced/V_rf)**2:.2f}")
                result.append("")
            elif q_mathieu > 0.7:
                result.append("WARNING: q is high - near edge of stability region")
                result.append("Consider reducing voltage or increasing frequency")
                result.append("for more robust trapping.")
                result.append("")
            elif q_mathieu < 0.1:
                result.append("Note: q is low - trap confinement may be weak")
                result.append("Consider increasing voltage for stronger confinement.")
                result.append("")
            else:
                result.append("STABLE: q parameter is within the stable region.")
                result.append("")

            # Secular frequency estimate (valid only if stable)
            if mathieu_stable and q_mathieu > 0:
                # ω_sec ≈ q·Ω / (2√2) for small q
                omega_sec = q_mathieu * Omega / (2 * np.sqrt(2))
                f_sec = omega_sec / (2 * np.pi)
                result.append(f"Estimated secular frequency: {f_sec:.2f} Hz")
                if f_sec >= 1e3:
                    result.append(f"                            ({f_sec/1e3:.2f} kHz)")
                result.append("")

            # If unstable, still show the rest but with warning
            if not mathieu_stable:
                result.append("=" * 60)
                result.append("NOTE: Trap depth analysis below is NOT VALID")
                result.append("because the trap is in an unstable regime.")
                result.append("=" * 60)
                result.append("")

            # Check if using numeric fields
            if self.use_numeric_mode:
                result.append("Mode: NUMERIC (CAD geometry)")
                result.append("Sampling electric fields from CAD mesh...")
                result.append("")

                # Try to load numeric fields
                try:
                    numeric_dir = pathlib.Path(getattr(self, "active_numeric_field_dir", self.numeric_field_dir))
                    electrode_program = self._resolve_electrode_program(V_rf, V_dc, freq_hz, numeric_dir)

                    # Create temporary simulator to sample fields
                    # For trap depth analysis, we only need mass and charge, not radius/density
                    temp_sim = fields.TrapSimulator(
                        V_rf=V_rf,
                        V_dc=V_dc,
                        Omega=Omega,
                        r0=self.r0_var.get(),
                        trap_type=self.trap_type.get() if self.trap_type.get() != "Custom (a,b,c)" else None,
                        custom_abc=(self.a_var.get(), self.b_var.get(), self.c_var.get()) if self.trap_type.get()=="Custom (a,b,c)" else None,
                        particle_radius=1e-9 if self.use_mc_override_var.get() else self.radius_var.get(),  # Use dummy value for ions
                        particle_density=1000 if self.use_mc_override_var.get() else self.density_var.get(),  # Use dummy value for ions
                        particle_charge=Q,
                        enable_gravity=True,
                        use_numeric=True,
                        numeric_field_dir=str(numeric_dir),
                        electrode_program=electrode_program,
                        use_numeric_grid=False,
                    )

                    # Find trap center by locating minimum |E| (equilibrium position)
                    trap_center = self._find_trap_center_3d(temp_sim, numeric_dir)
                    if trap_center is None:
                        trap_center = np.array([0.0, 0.0, 1.5e-3])  # Default fallback
                        result.append("Could not determine trap center, using default (0, 0, 1.5mm)")
                    else:
                        result.append(f"Trap center (min |E|): ({trap_center[0]*1e6:.0f}, {trap_center[1]*1e6:.0f}, {trap_center[2]*1e6:.0f}) um")
                    result.append("")

                    # Sample in multiple directions from trap center
                    result.append("Sampling pseudopotential around trap center:")
                    result.append("-" * 40)

                    # Sample distances (in meters)
                    sample_distances_um = [50, 100, 200, 500]
                    directions = [
                        ("+X", np.array([1, 0, 0])),
                        ("-X", np.array([-1, 0, 0])),
                        ("+Y", np.array([0, 1, 0])),
                        ("-Y", np.array([0, -1, 0])),
                        ("+Z", np.array([0, 0, 1])),
                        ("-Z", np.array([0, 0, -1])),
                    ]

                    # Calculate pseudopotential at trap center
                    try:
                        Ex0, Ey0, Ez0 = temp_sim.field(
                            np.array([trap_center[0]]),
                            np.array([trap_center[1]]),
                            np.array([trap_center[2]]),
                            0.0
                        )
                        E0_mag = np.sqrt(Ex0[0]**2 + Ey0[0]**2 + Ez0[0]**2)
                        psi_center = (Q**2 / (4 * m * Omega**2)) * (E0_mag**2 / 2.0)
                        result.append(f"At center: |E|={E0_mag:.2e} V/m, Psi={psi_center:.3e} J")
                    except Exception:
                        E0_mag = 0
                        psi_center = 0
                        result.append("Could not sample field at trap center")

                    result.append("")

                    # Track minimum trap depth across all directions
                    min_trap_depth = float('inf')
                    min_direction = None

                    for dir_name, dir_vec in directions:
                        result.append(f"Direction {dir_name}:")
                        for d_um in sample_distances_um:
                            d = d_um * 1e-6
                            pos = trap_center + d * dir_vec
                            try:
                                Ex, Ey, Ez = temp_sim.field(
                                    np.array([pos[0]]),
                                    np.array([pos[1]]),
                                    np.array([pos[2]]),
                                    0.0
                                )
                                E_mag = np.sqrt(Ex[0]**2 + Ey[0]**2 + Ez[0]**2)
                                psi = (Q**2 / (4 * m * Omega**2)) * (E_mag**2 / 2.0)
                                depth = psi - psi_center
                                if depth > 0 and depth < min_trap_depth:
                                    min_trap_depth = depth
                                    min_direction = f"{dir_name} at {d_um}um"
                                result.append(f"  {d_um:4d} um: |E|={E_mag:.2e} V/m, depth={depth:.3e} J")
                            except Exception as ex:
                                result.append(f"  {d_um:4d} um: Error - {str(ex)[:30]}")
                        result.append("")

                    # Gravitational comparison
                    grav_100um = m * g * 100e-6
                    grav_500um = m * g * 500e-6

                    result.append("-" * 40)
                    result.append("Gravitational comparison:")
                    result.append(f"  Grav. potential at 100 um: {grav_100um:.3e} J")
                    result.append(f"  Grav. potential at 500 um: {grav_500um:.3e} J")
                    if min_trap_depth < float('inf'):
                        result.append(f"  Min trap depth found: {min_trap_depth:.3e} J ({min_direction})")
                        ratio = min_trap_depth / grav_500um if grav_500um > 0 else float('inf')
                        result.append(f"  Depth/Gravity ratio (500um): {ratio:.2f}")
                    result.append("")

                    result.append("=" * 60)
                    result.append("VERDICT:")
                    result.append("=" * 60)

                    if min_trap_depth == float('inf') or min_trap_depth <= 0:
                        result.append("Could not determine trap depth.")
                        result.append("The trap may be very weak or sampling failed.")
                    else:
                        ratio = min_trap_depth / grav_500um if grav_500um > 0 else float('inf')
                        if ratio < 0.01:
                            result.append("CRITICAL: Gravity dominates by {:.0f}x!".format(1/ratio))
                            result.append("")
                            result.append("Your trap is FAR too weak. The particle will fall.")
                            result.append("")
                            result.append("═" * 60)
                            result.append("SOLUTIONS TO STRENGTHEN TRAP")
                            result.append("═" * 60)
                            result.append("")

                            # Calculate how much we need to improve
                            target_ratio = 5.0  # Target: trap depth 5x gravity
                            improvement_needed = target_ratio / ratio

                            result.append(f"Target: Make trap depth {target_ratio:.0f}x stronger than gravity")
                            result.append(f"Need to improve trap depth by {improvement_needed:.0f}x")
                            result.append("")

                            if mathieu_stable and q_mathieu < 0.5:
                                # Have room to increase voltage
                                q_margin = 0.7 - q_mathieu  # Use up to q=0.7
                                v_increase_possible = np.sqrt((0.7 / q_mathieu)) if q_mathieu > 0 else float('inf')
                                depth_improvement_voltage = v_increase_possible**2

                                result.append("Option 1: INCREASE RF VOLTAGE (Recommended)")
                                result.append("-" * 50)
                                result.append(f"  Current Vrf: {V_rf} V")
                                result.append(f"  Current q: {q_mathieu:.3f} (stable, can go to ~0.7)")
                                v_new = V_rf * min(v_increase_possible, np.sqrt(improvement_needed))
                                result.append(f"  Suggested Vrf: {v_new:.0f} V")
                                result.append(f"  → Voltage increase: {v_new/V_rf:.1f}x")
                                result.append(f"  → Trap depth increase: {(v_new/V_rf)**2:.1f}x")
                                result.append(f"  → New q: {q_mathieu * (v_new/V_rf):.3f}")
                                if depth_improvement_voltage >= improvement_needed:
                                    result.append("   This WILL achieve stable trapping!")
                                else:
                                    result.append(f"   This only gives {depth_improvement_voltage:.1f}x improvement")
                                    result.append("    Need to also increase frequency or reduce mass")
                                result.append("")

                            # Calculate frequency increase needed
                            freq_for_depth = freq_hz * np.sqrt(improvement_needed)
                            result.append("Option 2: INCREASE FREQUENCY (if voltage limited)")
                            result.append("-" * 50)
                            result.append(f"  Current freq: {freq_hz:.0f} Hz")
                            if freq_for_depth >= 1e6:
                                result.append(f"  Needed freq: {freq_for_depth/1e6:.2f} MHz")
                            elif freq_for_depth >= 1e3:
                                result.append(f"  Needed freq: {freq_for_depth/1e3:.2f} kHz")
                            else:
                                result.append(f"  Needed freq: {freq_for_depth:.0f} Hz")
                            result.append(f"  → Frequency increase: {freq_for_depth/freq_hz:.1f}x")
                            result.append(f"  → Trap depth increase: {(freq_for_depth/freq_hz)**2:.1f}x")
                            result.append(f"  → Mathieu q unchanged (if Vrf constant)")
                            result.append("   Keeps voltage and q parameter same")
                            result.append("   Requires higher frequency electronics")
                            result.append("")

                            # Calculate particle property changes
                            qm_reduction_needed = 1.0 / improvement_needed
                            result.append("Option 3: CHANGE PARTICLE PROPERTIES")
                            result.append("-" * 50)
                            result.append(f"  Current Q/m: {Q/m:.3e} C/kg")
                            result.append(f"  Current mass: {m:.3e} kg")
                            result.append(f"  Current charge: {Q:.3e} C ({Q/e:.0f} elem. charges)")
                            result.append("")
                            result.append(f"  To increase depth {improvement_needed:.0f}x:")
                            result.append(f"    a) Reduce mass to {m/improvement_needed:.3e} kg")
                            result.append(f"       (particle radius → {self.radius_var.get()/improvement_needed**(1/3):.2e} m)")
                            result.append(f"    b) Increase charge to {Q*np.sqrt(improvement_needed):.3e} C")
                            result.append(f"       ({Q*np.sqrt(improvement_needed)/e:.0f} elem. charges)")
                            result.append("  Note: Trap depth ∝ Q²/m")
                            result.append("")

                            # Combined approach
                            result.append("Option 4: COMBINED APPROACH (Balanced)")
                            result.append("-" * 50)
                            # Split improvement between voltage and frequency
                            factor_each = np.sqrt(improvement_needed)  # Each parameter contributes sqrt
                            v_combined = V_rf * factor_each
                            f_combined = freq_hz * factor_each
                            result.append("  Increase both voltage AND frequency moderately:")
                            result.append(f"    Vrf: {V_rf} V → {v_combined:.0f} V ({factor_each:.1f}x)")
                            if f_combined >= 1e6:
                                result.append(f"    Freq: {freq_hz:.0f} Hz → {f_combined/1e6:.2f} MHz ({factor_each:.1f}x)")
                            elif f_combined >= 1e3:
                                result.append(f"    Freq: {freq_hz:.0f} Hz → {f_combined/1e3:.2f} kHz ({factor_each:.1f}x)")
                            else:
                                result.append(f"    Freq: {freq_hz:.0f} Hz → {f_combined:.0f} Hz ({factor_each:.1f}x)")
                            result.append(f"  → Total depth improvement: {improvement_needed:.0f}x")
                            result.append(f"  → New q: {q_mathieu * factor_each:.3f}")
                            result.append("   Moderate changes to each parameter")
                            result.append("   Achieves target depth with stability")
                            result.append("")

                            result.append("Important: Also check CAD geometry units!")
                            result.append("  If your CAD is in mm but loaded as meters,")
                            result.append("  fields will be 1000x too weak.")
                        elif ratio < 0.1:
                            result.append("WARNING: Gravity much stronger than trap!")
                            result.append(f"  Trap/Gravity ratio: {ratio:.3f} (need ≥1 for stability)")
                            result.append("")
                            result.append("Particle will likely drift down and become unstable.")
                            result.append("")

                            improvement = 10.0 / ratio  # Target 10x ratio
                            result.append("Quick fixes to improve by ~{:.0f}x:".format(improvement))
                            result.append("")

                            if mathieu_stable and q_mathieu < 0.6:
                                v_new = V_rf * min(2.0, np.sqrt(improvement))
                                result.append(f"  • Increase Vrf to {v_new:.0f} V")
                                result.append(f"    (currently {V_rf} V, new q={q_mathieu*(v_new/V_rf):.2f})")

                            f_new = freq_hz * np.sqrt(improvement)
                            if f_new >= 1e6:
                                result.append(f"  • Increase frequency to {f_new/1e6:.2f} MHz")
                            elif f_new >= 1e3:
                                result.append(f"  • Increase frequency to {f_new/1e3:.2f} kHz")
                            else:
                                result.append(f"  • Increase frequency to {f_new:.0f} Hz")
                            result.append(f"    (currently {freq_hz:.0f} Hz)")
                            result.append("")
                            result.append(f"  • Or reduce particle mass to {m/improvement:.2e} kg")
                            result.append(f"    (currently {m:.2e} kg)")

                        elif ratio < 1.0:
                            result.append("MARGINAL: Weak confinement")
                            result.append(f"  Trap/Gravity ratio: {ratio:.3f} (need ≥1 minimum, ≥5 recommended)")
                            result.append("")
                            result.append("Particle may be trapped but will drift significantly.")
                            result.append("Recommend strengthening by ~5x for robust trapping:")
                            result.append("")

                            if mathieu_stable and q_mathieu < 0.6:
                                v_rec = V_rf * min(1.5, np.sqrt(5.0/ratio))
                                result.append(f"  • Increase Vrf to {v_rec:.0f} V (currently {V_rf} V)")
                                result.append(f"    → Depth × {(v_rec/V_rf)**2:.1f}, q={q_mathieu*(v_rec/V_rf):.2f}")
                            else:
                                f_rec = freq_hz * np.sqrt(5.0/ratio)
                                if f_rec >= 1e6:
                                    result.append(f"  • Increase freq to {f_rec/1e6:.2f} MHz (currently {freq_hz:.0f} Hz)")
                                elif f_rec >= 1e3:
                                    result.append(f"  • Increase freq to {f_rec/1e3:.2f} kHz (currently {freq_hz:.0f} Hz)")
                                else:
                                    result.append(f"  • Increase freq to {f_rec:.0f} Hz (currently {freq_hz:.0f} Hz)")

                        elif ratio < 5.0:
                            result.append("ACCEPTABLE: Basic confinement")
                            result.append(f"  Trap/Gravity ratio: {ratio:.2f} (acceptable range)")
                            result.append("")
                            result.append("Particle should be trapped but may have large")
                            result.append("oscillations and be sensitive to perturbations.")
                            result.append("")
                            result.append("For more robust trapping (ratio ~10), consider:")

                            factor = np.sqrt(10.0/ratio)
                            if mathieu_stable and q_mathieu < 0.7:
                                result.append(f"  • Increase Vrf by {factor:.1f}x to {V_rf*factor:.0f} V")
                            else:
                                result.append(f"  • Increase frequency by {factor:.1f}x")
                        else:
                            result.append("GOOD: Strong confinement")
                            result.append(f"  Trap/Gravity ratio: {ratio:.1f}")
                            result.append("")
                            result.append("Trap depth significantly exceeds gravitational")
                            result.append("potential. Particle should be stably trapped.")
                            result.append("")
                            if ratio > 20:
                                result.append("Note: Trap is very strong. You could potentially:")
                                result.append(f"  • Reduce voltage to {V_rf/2:.0f} V (save power)")
                                result.append(f"  • Trap heavier particles (up to {m*ratio/5:.2e} kg)")
                                result.append("  while maintaining good confinement (ratio > 5)")

                except Exception as e:
                    result.append(f"Error loading numeric fields: {e}")
                    result.append("")
                    result.append("Make sure you have:")
                    result.append("1. Loaded CAD files")
                    result.append("2. Generated mesh and basis fields")

            else:
                result.append("Mode: ANALYTIC (simplified geometry)")
                result.append("")

                # Analytic estimate - sample around origin (trap center for analytic traps)
                r0 = self.r0_var.get()
                trap_center = np.array([0.0, 0.0, 0.0])

                result.append(f"Trap center: origin (analytic trap)")
                result.append(f"Characteristic size r0: {r0*1e3:.2f} mm")
                result.append("")

                # Sample in multiple directions
                result.append("Sampling pseudopotential around trap center:")
                result.append("-" * 40)

                sample_fractions = [0.1, 0.25, 0.5, 1.0]  # Fractions of r0
                directions = [
                    ("+X", np.array([1, 0, 0])),
                    ("+Y", np.array([0, 1, 0])),
                    ("+Z", np.array([0, 0, 1])),
                ]

                # Get E-field at center
                if self.sim:
                    try:
                        Ex0, Ey0, Ez0 = self.sim.field(
                            np.array([0.0]), np.array([0.0]), np.array([0.0]), 0.0
                        )
                        E0_mag = np.sqrt(Ex0[0]**2 + Ey0[0]**2 + Ez0[0]**2)
                    except Exception:
                        E0_mag = 0
                else:
                    E0_mag = 0

                psi_center = (Q**2 / (4 * m * Omega**2)) * (E0_mag**2 / 2.0)
                result.append(f"At center: |E|={E0_mag:.2e} V/m")
                result.append("")

                min_trap_depth = float('inf')

                for dir_name, dir_vec in directions:
                    result.append(f"Direction {dir_name}:")
                    for frac in sample_fractions:
                        d = frac * r0
                        pos = trap_center + d * dir_vec
                        if self.sim:
                            try:
                                Ex, Ey, Ez = self.sim.field(
                                    np.array([pos[0]]), np.array([pos[1]]), np.array([pos[2]]), 0.0
                                )
                                E_mag = np.sqrt(Ex[0]**2 + Ey[0]**2 + Ez[0]**2)
                                psi = (Q**2 / (4 * m * Omega**2)) * (E_mag**2 / 2.0)
                                depth = psi - psi_center
                                if depth > 0 and depth < min_trap_depth:
                                    min_trap_depth = depth
                                result.append(f"  {frac:.2f}*r0: |E|={E_mag:.2e} V/m, depth={depth:.3e} J")
                            except Exception:
                                # Use analytic formula as fallback
                                E_approx = V_rf * d / r0**2
                                psi = (Q**2 / (4 * m * Omega**2)) * (E_approx**2 / 2.0)
                                result.append(f"  {frac:.2f}*r0: |E|~{E_approx:.2e} V/m (est), depth~{psi:.3e} J")
                                if psi > 0 and psi < min_trap_depth:
                                    min_trap_depth = psi
                        else:
                            # Use analytic formula
                            E_approx = V_rf * d / r0**2
                            psi = (Q**2 / (4 * m * Omega**2)) * (E_approx**2 / 2.0)
                            result.append(f"  {frac:.2f}*r0: |E|~{E_approx:.2e} V/m (est)")
                            if psi > 0 and psi < min_trap_depth:
                                min_trap_depth = psi
                    result.append("")

                # Fallback to formula if sampling failed
                if min_trap_depth == float('inf'):
                    pseudo_depth = Q**2 * V_rf**2 / (4 * m * Omega**2 * r0**2)
                    min_trap_depth = pseudo_depth

                grav_500um = m * g * 500e-6
                ratio = min_trap_depth / grav_500um if grav_500um > 0 else float('inf')

                result.append("-" * 40)
                result.append(f"Estimated trap depth: {min_trap_depth:.3e} J")
                result.append(f"Gravitational potential (500 um): {grav_500um:.3e} J")
                result.append(f"Ratio: {ratio:.2f}")
                result.append("")
                result.append("=" * 60)
                result.append("VERDICT:")
                result.append("=" * 60)

                if ratio < 1.0:
                    result.append("INSUFFICIENT: Trap too weak for stable trapping!")
                    result.append(f"  Trap/Gravity ratio: {ratio:.3f} (need ≥1 minimum)")
                    result.append("")

                    improvement_needed = 5.0 / ratio
                    result.append(f"Need to improve trap depth by ~{improvement_needed:.1f}x")
                    result.append("")

                    if mathieu_stable and q_mathieu < 0.5:
                        v_new = V_rf * min(2.0, np.sqrt(improvement_needed))
                        result.append("RECOMMENDATION: Increase RF voltage")
                        result.append(f"  Current: {V_rf} V → Suggested: {v_new:.0f} V")
                        result.append(f"  This improves depth by {(v_new/V_rf)**2:.1f}x")
                        result.append(f"  New q = {q_mathieu * (v_new/V_rf):.3f} (still stable)")
                    else:
                        result.append("RECOMMENDATION: Increase frequency and voltage together")
                        factor = np.sqrt(np.sqrt(improvement_needed))
                        v_new = V_rf * factor
                        f_new = freq_hz * factor
                        result.append(f"  Voltage: {V_rf} V → {v_new:.0f} V")
                        if f_new >= 1e6:
                            result.append(f"  Frequency: {freq_hz:.0f} Hz → {f_new/1e6:.2f} MHz")
                        elif f_new >= 1e3:
                            result.append(f"  Frequency: {freq_hz:.0f} Hz → {f_new/1e3:.2f} kHz")
                        else:
                            result.append(f"  Frequency: {freq_hz:.0f} Hz → {f_new:.0f} Hz")
                        result.append(f"  This improves depth by {improvement_needed:.1f}x")
                        result.append(f"  Maintains q = {q_mathieu:.3f} (stable)")
                    result.append("")
                    result.append("Alternative: Use particles with higher Q²/m ratio")
                    result.append(f"  (Currently Q²/m = {Q**2/m:.3e})")

                elif ratio < 5.0:
                    result.append("MARGINAL: Weak confinement")
                    result.append(f"  Trap/Gravity ratio: {ratio:.2f} (acceptable but weak)")
                    result.append("")
                    result.append("Particle should trap but may be sensitive to disturbances.")
                    result.append("")
                    result.append("For robust trapping (ratio ~10), consider:")
                    factor_needed = np.sqrt(10.0 / ratio)

                    if mathieu_stable and q_mathieu < 0.7:
                        result.append(f"  • Increase Vrf by {factor_needed:.1f}x to {V_rf*factor_needed:.0f} V")
                        result.append(f"    (q would become {q_mathieu*factor_needed:.3f})")
                    else:
                        result.append(f"  • Increase frequency by {factor_needed:.1f}x")
                        f_improved = freq_hz * factor_needed
                        if f_improved >= 1e6:
                            result.append(f"    ({freq_hz:.0f} Hz → {f_improved/1e6:.2f} MHz)")
                        elif f_improved >= 1e3:
                            result.append(f"    ({freq_hz:.0f} Hz → {f_improved/1e3:.2f} kHz)")

                else:
                    result.append("GOOD: Strong confinement")
                    result.append(f"  Trap/Gravity ratio: {ratio:.1f}")
                    result.append("")
                    result.append("Trap depth is sufficient for stable trapping.")
                    if ratio > 20:
                        result.append("")
                        result.append("Note: Very strong trap. Could consider:")
                        result.append(f"  • Reducing voltage to {V_rf/2:.0f} V (save power)")
                        result.append(f"  • Using heavier particles")

            # Add helpful reference section
            result.append("")
            result.append("=" * 60)
            result.append("QUICK REFERENCE: KEY FORMULAS")
            result.append("=" * 60)
            result.append("")
            result.append("Mathieu Stability Parameter:")
            result.append("  q = 4·|Q|·Vrf / (m·Ω²·r₀²)")
            result.append("  • Must be < 0.908 for stability")
            result.append("  • Optimal range: 0.3 - 0.7")
            result.append("  • q ∝ V/(m·f²)")
            result.append("")
            result.append("Trap Depth (Pseudopotential):")
            result.append("  Ψ ≈ Q²·Vrf² / (4·m·Ω²·r₀²)")
            result.append("  • Trap depth ∝ Q²·V²/(m·f²)")
            result.append("  • Need: Depth > m·g·h (gravitational)")
            result.append("")
            result.append("To improve stability & depth together:")
            result.append("  • Increase f: Keeps q same, increases depth ∝ f²")
            result.append("  • Reduce m or increase Q: Affects both")
            result.append("  • Increase r₀: Improves stability but weakens depth")
            result.append("")
            result.append("Typical parameters:")
            result.append("  • Microspheres: 1-10 kHz, 100-1000 V")
            result.append("  • Ions: 1-10 MHz, 10-1000 V")
            result.append("")

            # Show result in popup
            popup = tk.Toplevel(self.root)
            popup.title("Trap Depth Analysis")
            popup.geometry("700x800")

            # Create scrollable text widget
            text_frame = ttk.Frame(popup)
            text_frame.pack(fill='both', expand=True, padx=10, pady=10)

            scrollbar = ttk.Scrollbar(text_frame)
            scrollbar.pack(side='right', fill='y')

            text = tk.Text(text_frame, wrap='word', font=('Courier', 9), yscrollcommand=scrollbar.set)
            text.pack(side='left', fill='both', expand=True)
            scrollbar.config(command=text.yview)

            text.insert('1.0', '\n'.join(result))
            text.config(state='disabled')

            # Add buttons frame
            button_frame = ttk.Frame(popup)
            button_frame.pack(pady=5)

            def copy_to_clipboard():
                popup.clipboard_clear()
                popup.clipboard_append('\n'.join(result))
                self.update_status("Analysis copied to clipboard", "success")

            ttk.Button(button_frame, text="Copy to Clipboard", command=copy_to_clipboard).pack(side='left', padx=5)
            ttk.Button(button_frame, text="Close", command=popup.destroy).pack(side='left', padx=5)

            self.update_status("Trap depth analysis complete", 'success')

        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze trap depth:\n{str(e)}")
            import traceback
            traceback.print_exc()

    # === Plotting methods ===
    
    def update_single_particle_plots(self, t, positions, velocities):
        """Update plots for single particle simulation"""
        self._last_plot_kind = 'single'
        self._last_plot_payload = (t, positions, velocities)
        
        self._plot_single_particle_to_axes(self.ax3d, self.ax_x, self.ax_y, self.ax_z, t, positions, velocities)
        self.canvas.draw()
    
    def _plot_single_particle_to_axes(self, ax3d, ax_x, ax_y, ax_z, t, positions, velocities):
        """Plot single particle data to given axes"""
        # Clear axes
        ax3d.cla()
        ax_x.cla()
        ax_y.cla()
        ax_z.cla()

        # Convert to appropriate units
        unit_scale = 1e6  # to microns
        unit_label = "μm"

        pos_scaled = positions * unit_scale
        plot_idx = _adaptive_downsample_indices(pos_scaled.shape[0], MAX_TRAJ_POINTS)
        plot_pos = pos_scaled if plot_idx is None else pos_scaled[plot_idx]

        # Store trajectory data for interactive selection
        self._trajectory_data = {
            't': t,
            'positions': pos_scaled,
            'positions_m': positions,  # Original positions in meters (for PyVista)
            'unit_scale': unit_scale,
            'unit_label': unit_label,
            'time_scale': None,
            'time_label': None,
        }

        # Sync to PyVista if open
        self._sync_pyvista_trajectory()
        # Store reference to marker artists (for updating)
        self._selected_point_markers = {
            '3d': None,
            'x': None,
            'y': None,
            'z': None
        }
        self._selected_fft_markers = {
            'x': None,
            'y': None,
            'z': None
        }

        #draw this first so the particle line appears 'inside' or on top depending on z-order
        if self.show_cad_mesh_var.get():
             self._draw_cad_overlay(ax3d, unit_scale_plot=unit_scale)

        # Draw beam marker
        self._draw_beam_marker(ax3d, unit_scale_plot=unit_scale)

        # 3D trajectory
        ax3d.plot(plot_pos[:, 0], plot_pos[:, 1], plot_pos[:, 2],
                 'b-', linewidth=0.5, alpha=0.6)
        ax3d.scatter(pos_scaled[0, 0], pos_scaled[0, 1], pos_scaled[0, 2],
                    c='g', s=100, marker='o', label='Start')
        ax3d.scatter(pos_scaled[-1, 0], pos_scaled[-1, 1], pos_scaled[-1, 2],
                    c='r', s=100, marker='x', label='End')
        
        # Draw trap geometry (only for analytical traps)
        if self.sim and not self.sim.use_numeric:
            r0_scaled = self.sim.r0 * unit_scale
            if self.sim.trap_type == "3D" or "3D" in self.sim.trap_type:
                theta = np.linspace(0, 2*np.pi, 50)
                z_rings = [r0_scaled/np.sqrt(2), -r0_scaled/np.sqrt(2)]
                for z in z_rings:
                    x_ring = r0_scaled * np.cos(theta)
                    y_ring = r0_scaled * np.sin(theta)
                    ax3d.plot(x_ring, y_ring, [z]*len(theta), 'k--', alpha=0.3, label='Trap boundary')
            elif self.sim.trap_type == "Linear":
                z_line = np.linspace(-2*r0_scaled, 2*r0_scaled, 50)
                for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
                    x_center = r0_scaled * np.cos(angle)
                    y_center = r0_scaled * np.sin(angle)
                    ax3d.plot([x_center]*len(z_line), [y_center]*len(z_line), z_line,
                             'k--', alpha=0.5, linewidth=2)

        # Set labels and limits
        max_pos = np.abs(pos_scaled).max()
        if self.sim and not self.sim.use_numeric:
            r0_scaled = self.sim.r0 * unit_scale
            lim = max(max_pos * 1.2, r0_scaled * 1.5)
        else:
            lim = max_pos * 1.2
        ax3d.set_xlim([-lim, lim])
        ax3d.set_ylim([-lim, lim])
        ax3d.set_zlim([-lim, lim])
        ax3d.set_xlabel(f'x ({unit_label})')
        ax3d.set_ylabel(f'y ({unit_label})')
        ax3d.set_zlabel(f'z ({unit_label})')
        ax3d.set_title('Particle Trajectory')
        ax3d.legend()
        
        # Time-series or FFT plots
        if self.show_fft_var.get():
            self._fft_data = {}
            # FFT Mode: Show frequency spectrum
            self._plot_fft_analysis(ax_x, ax_y, ax_z, t, positions)
        else:
            self._fft_data = None
            # Time-domain mode: Show position vs time
            t_series = t
            pos_series = pos_scaled
            time_label = "ms"
            time_scale = 1e3

            duration = t_series[-1] - t_series[0]
            if duration < 1e-6:
                time_scale = 1e9
                time_label = "ns"
            elif duration < 1e-3:
                time_scale = 1e6
                time_label = "us"

            t_plot = t_series * time_scale
            self._trajectory_data['time_scale'] = time_scale
            self._trajectory_data['time_label'] = time_label
            ts_idx = _adaptive_downsample_indices(t_plot.shape[0], MAX_TIME_SERIES_POINTS)
            if ts_idx is None:
                pos_ts = pos_series
            else:
                t_plot = t_plot[ts_idx]
                pos_ts = pos_series[ts_idx]

            ax_x.plot(t_plot, pos_ts[:, 0], 'b-', linewidth=0.8)
            ax_x.set_xlabel(f'Time ({time_label})')
            ax_x.set_ylabel(f'x ({unit_label})')
            ax_x.set_title('X Position vs Time')
            ax_x.grid(True, alpha=0.3)

            ax_y.plot(t_plot, pos_ts[:, 1], 'g-', linewidth=0.8)
            ax_y.set_xlabel(f'Time ({time_label})')
            ax_y.set_ylabel(f'y ({unit_label})')
            ax_y.set_title('Y Position vs Time')
            ax_y.grid(True, alpha=0.3)

            ax_z.plot(t_plot, pos_ts[:, 2], 'r-', linewidth=0.8)
            ax_z.set_xlabel(f'Time ({time_label})')
            ax_z.set_ylabel(f'z ({unit_label})')
            ax_z.set_title('Z Position vs Time')
            ax_z.grid(True, alpha=0.3)

        # Update toolbar's cached view limits so Home button works correctly
        self.toolbar.update()
        self.toolbar.push_current()
        
        # Apply view settings
        ax3d.view_init(elev=self.elev_var.get(), azim=self.azim_var.get())
        self._apply_axis_scale(ax3d)
    
    def update_cloud_plots(self, t, positions):
        """Update plots for particle cloud simulation"""
        self._last_plot_kind = 'cloud'
        self._last_plot_payload = (t, positions)
        
        self._plot_cloud_to_axes(self.ax3d, self.ax_x, self.ax_y, self.ax_z, t, positions)
        self.canvas.draw()
    
    def _plot_cloud_to_axes(self, ax3d, ax_x, ax_y, ax_z, t, positions):
        """Plot particle cloud data to given axes"""
        # Clear axes
        ax3d.cla()
        ax_x.cla()
        ax_y.cla()
        ax_z.cla()
        
        # Convert to microns
        unit_scale = 1e6
        unit_label = "m"
        pos_scaled = positions * unit_scale
        cloud_idx = _adaptive_downsample_indices(pos_scaled.shape[1], MAX_CLOUD_POINTS)
        if cloud_idx is None:
            initial = pos_scaled[0]
            final = pos_scaled[-1]
        else:
            initial = pos_scaled[0, cloud_idx]
            final = pos_scaled[-1, cloud_idx]

        if self.show_cad_mesh_var.get():
             self._draw_cad_overlay(ax3d, unit_scale_plot=unit_scale)

        # Draw beam marker
        self._draw_beam_marker(ax3d, unit_scale_plot=unit_scale)

        # Show initial and final positions
        ax3d.scatter(initial[:, 0], initial[:, 1], initial[:, 2],
                    c='g', alpha=0.6, label='Initial', s=20)
        ax3d.scatter(final[:, 0], final[:, 1], final[:, 2],
                    c='r', alpha=0.6, label='Final', s=20)
        
        # Draw trap geometry (only for analytical traps)
        if self.sim and not self.sim.use_numeric:
            r0_scaled = self.sim.r0 * unit_scale
            if self.sim.trap_type == "3D" or "3D" in self.sim.trap_type:
                theta = np.linspace(0, 2*np.pi, 30)
                z_rings = [r0_scaled/np.sqrt(2), -r0_scaled/np.sqrt(2)]
                for z in z_rings:
                    x_ring = r0_scaled * np.cos(theta)
                    y_ring = r0_scaled * np.sin(theta)
                    ax3d.plot(x_ring, y_ring, [z]*len(theta), 'k--', alpha=0.3, label='Trap boundary')
            elif self.sim.trap_type == "Linear":
                z_line = np.linspace(-2*r0_scaled, 2*r0_scaled, 50)
                for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
                    x_center = r0_scaled * np.cos(angle)
                    y_center = r0_scaled * np.sin(angle)
                    ax3d.plot([x_center]*len(z_line), [y_center]*len(z_line), z_line,
                             'k--', alpha=0.5, linewidth=2)
        
        ax3d.legend()
        ax3d.set_title('Cloud Evolution')
        ax3d.set_xlabel(f'x ({unit_label})')
        ax3d.set_ylabel(f'y ({unit_label})')
        ax3d.set_zlabel(f'z ({unit_label})')
        
        # Position distributions
        for ax, idx, label in zip([ax_x, ax_y, ax_z], [0, 1, 2], ['X', 'Y', 'Z']):
            ax.hist(pos_scaled[-1, :, idx], bins=15, alpha=0.7, color=['b', 'g', 'r'][idx])
            ax.set_title(f'Final {label} Distribution')
            ax.set_xlabel(f'{label} ({unit_label})')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)

        # Update toolbar's cached view limits so Home button works correctly
        self.toolbar.update()
        self.toolbar.push_current()

        # Apply view settings
        ax3d.view_init(elev=self.elev_var.get(), azim=self.azim_var.get())
        self._apply_axis_scale(ax3d)

    def on_cad_mesh_toggle(self):
        # toggle of CAD mesh visualization
        # Trigger a re-plot if we have data
        if self._last_plot_kind and self._last_plot_payload:
            if self._last_plot_kind == 'single':
                self.update_single_particle_plots(*self._last_plot_payload)
            elif self._last_plot_kind == 'cloud':
                self.update_cloud_plots(*self._last_plot_payload)

    def open_pyvista_viewer(self):
        """Open the PyVista 3D viewer with CAD mesh and trajectory."""
        if not PYVISTA_AVAILABLE:
            messagebox.showerror("PyVista Not Available",
                "PyVista is not installed.\nRun: pip install pyvista pyvistaqt")
            return

        # Create viewer if needed
        if self._pyvista_viewer is None:
            self._pyvista_viewer = PyVista3DViewer(title="3D Trap Viewer")

        # Open window
        if not self._pyvista_viewer.is_open():
            self._pyvista_viewer.open()

        # Load CAD mesh if available
        self._sync_pyvista_mesh()

        # Load trajectory if available
        self._sync_pyvista_trajectory()

    def _sync_pyvista_mesh(self):
        """Load the CAD mesh into PyVista viewer."""
        if self._pyvista_viewer is None or not self._pyvista_viewer.is_open():
            return

        mesh_path = self.numeric_field_dir / "mesh.msh"
        meta_path = self.numeric_field_dir / "facet_names.json"

        if not mesh_path.exists():
            return

        # Read metadata for scaling and electrode tags
        unit_scale = 1.0
        electrode_tags = set()

        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    unit_scale = meta.get("unit_scale_to_m", 1.0)
                    if "electrodes" in meta:
                        for name, tag in meta["electrodes"].items():
                            electrode_tags.add(tag)
            except Exception as e:
                print(f"[PyVista] Error reading metadata: {e}")

        # Set display units to match matplotlib plots
        self._pyvista_viewer.set_unit_scale(1e6, "μm")

        # Load mesh
        self._pyvista_viewer.load_mesh(
            mesh_path,
            unit_scale_to_m=unit_scale,
            electrode_tags=electrode_tags if electrode_tags else None
        )

    def _sync_pyvista_trajectory(self):
        """Sync trajectory data to PyVista viewer."""
        if self._pyvista_viewer is None or not self._pyvista_viewer.is_open():
            return

        if not self._trajectory_data:
            self._pyvista_viewer.clear_trajectory()
            return

        # Use positions in meters (not scaled display units)
        positions_m = self._trajectory_data.get('positions_m')
        if positions_m is None or len(positions_m) < 2:
            return

        self._pyvista_viewer.update_trajectory(positions_m, unit_scale_to_m=1.0)

    def _sync_pyvista_marker(self, position_scaled):
        """Update the position marker in PyVista viewer.

        Args:
            position_scaled: Position in display units (microns)
        """
        if self._pyvista_viewer is None or not self._pyvista_viewer.is_open():
            return

        if not self._trajectory_data:
            return

        # Convert from display units back to meters
        unit_scale = self._trajectory_data.get('unit_scale', 1e6)
        position_m = np.array(position_scaled) / unit_scale

        self._pyvista_viewer.update_marker(position_m, unit_scale_to_m=1.0)

    def _on_pv_marker_size_change(self):
        """Handle PyVista marker size slider changes."""
        if self._pyvista_viewer is None or not self._pyvista_viewer.is_open():
            return

        # Update marker sizes
        marker_size = self._pv_marker_size_var.get()
        pos_marker_size = self._pv_pos_marker_size_var.get()

        self._pyvista_viewer.set_marker_sizes(
            start=marker_size,
            end=marker_size,
            position=pos_marker_size
        )

        # Re-sync trajectory to apply new sizes
        self._sync_pyvista_trajectory()

    def on_fft_toggle(self):
        # Toggle FFT frequency analysis view
        if self._last_plot_kind and self._last_plot_payload:
            if self._last_plot_kind == 'single':
                self.update_single_particle_plots(*self._last_plot_payload)
            # Note: FFT only makes sense for single particle trajectories, not clouds

    def _on_plot_click(self, event):
        """Handle click events on time-series plots to highlight point in 3D"""
        # Only handle clicks on time-series plots (not 3D)
        if event.inaxes not in [self.ax_x, self.ax_y, self.ax_z]:
            return

        # Don't interfere with toolbar interactions
        if self.toolbar.mode != '':
            return

        if self.show_fft_var.get():
            self._handle_fft_click(event)
            return

        # Check if we have trajectory data
        if not self._trajectory_data:
            return

        t = self._trajectory_data['t']
        time_scale = self._trajectory_data.get('time_scale') or 1e3

        # Get clicked time from x-coordinate (in ms)
        t_click = event.xdata
        if t_click is None:
            return
        t_click = t_click / time_scale  # Convert to seconds

        # Find nearest time point
        idx = np.argmin(np.abs(t - t_click))

        # Highlight this point across all plots
        self._highlight_trajectory_point(idx)

    def _on_plot_hover(self, event):
        """Handle hover events to show preview of point location"""
        # Only handle hovers on time-series plots
        if event.inaxes not in [self.ax_x, self.ax_y, self.ax_z]:
            # Remove annotations if mouse leaves time-series plots
            self._remove_hover_annotations()
            return

        # Don't show hover during toolbar interactions
        if self.toolbar.mode != '':
            return

        if self.show_fft_var.get():
            self._handle_fft_hover(event)
            return

        # Check if we have trajectory data
        if not self._trajectory_data:
            return

        t = self._trajectory_data['t']
        time_scale = self._trajectory_data.get('time_scale') or 1e3

        # Get hovered time from x-coordinate (in ms)
        t_hover = event.xdata
        if t_hover is None:
            self._remove_hover_annotations()
            return
        t_hover_sec = t_hover / time_scale  # Convert to seconds

        # Find nearest time point
        idx = np.argmin(np.abs(t - t_hover_sec))

        # Show annotation with position
        self._show_hover_annotation(event.inaxes, idx)

    def _highlight_trajectory_point(self, idx):
        """Highlight a specific point on the trajectory across all plots"""
        if not self._trajectory_data:
            return

        t = self._trajectory_data['t']
        positions = self._trajectory_data['positions']
        time_scale = self._trajectory_data.get('time_scale') or 1e3
        time_label = self._trajectory_data.get('time_label') or "ms"

        pos = positions[idx]
        t_plot = t[idx] * time_scale

        # Remove old markers if they exist
        for marker in self._selected_point_markers.values():
            if marker is not None:
                try:
                    marker.remove()
                except:
                    pass

        # Add marker to 3D plot
        marker_3d = self.ax3d.scatter(
            [pos[0]], [pos[1]], [pos[2]],
            c='orange', s=200, marker='1',
            edgecolors='black', linewidths=1.5,
            zorder=10, label=f't={t_plot:.3f} {time_label}'
        )
        self._selected_point_markers['3d'] = marker_3d

        # Add markers to time-series plots
        marker_x = self.ax_x.scatter(
            [t_plot], [pos[0]],
            c='orange', s=100, marker='o',
            edgecolors='black', linewidths=1.5, zorder=10
        )
        self._selected_point_markers['x'] = marker_x

        marker_y = self.ax_y.scatter(
            [t_plot], [pos[1]],
            c='orange', s=100, marker='o',
            edgecolors='black', linewidths=1.5, zorder=10
        )
        self._selected_point_markers['y'] = marker_y

        marker_z = self.ax_z.scatter(
            [t_plot], [pos[2]],
            c='orange', s=100, marker='o',
            edgecolors='black', linewidths=1.5, zorder=10
        )
        self._selected_point_markers['z'] = marker_z

        # Update 3D legend to show selected point info
        self.ax3d.legend()

        # Update PyVista marker if viewer is open
        self._sync_pyvista_marker(pos)

        # Redraw
        self.canvas.draw_idle()

    def _get_fft_series(self, ax):
        axis_map = {self.ax_x: "x", self.ax_y: "y", self.ax_z: "z"}
        key = axis_map.get(ax)
        if not key or not isinstance(self._fft_data, dict):
            return None, None
        series = self._fft_data.get(key)
        if not series:
            return None, None
        return key, series

    def _handle_fft_click(self, event):
        if not isinstance(self._fft_data, dict):
            return
        key, series = self._get_fft_series(event.inaxes)
        if not key:
            return
        freq = series.get("freq")
        psd_db = series.get("psd_db")
        if freq is None or psd_db is None or event.xdata is None:
            return
        idx = int(np.argmin(np.abs(freq - event.xdata)))
        self._highlight_fft_point(event.inaxes, key, idx)

    def _handle_fft_hover(self, event):
        if not isinstance(self._fft_data, dict):
            return
        key, series = self._get_fft_series(event.inaxes)
        if not key:
            return
        freq = series.get("freq")
        psd_db = series.get("psd_db")
        if freq is None or psd_db is None or event.xdata is None:
            self._remove_hover_annotations()
            return
        idx = int(np.argmin(np.abs(freq - event.xdata)))
        self._show_fft_annotation(event.inaxes, key, idx)

    def _highlight_fft_point(self, ax, key, idx):
        series = self._fft_data.get(key) if isinstance(self._fft_data, dict) else None
        if not series:
            return
        freq = series.get("freq")
        psd_db = series.get("psd_db")
        if freq is None or psd_db is None:
            return

        marker = self._selected_fft_markers.get(key)
        if marker is not None:
            try:
                marker.remove()
            except Exception:
                pass

        marker = ax.scatter(
            [freq[idx]], [psd_db[idx]],
            c='orange', s=80, marker='o',
            edgecolors='black', linewidths=1.0, zorder=10
        )
        self._selected_fft_markers[key] = marker
        self.canvas.draw_idle()

    def _show_fft_annotation(self, ax, key, idx):
        series = self._fft_data.get(key) if isinstance(self._fft_data, dict) else None
        if not series:
            return
        freq = series.get("freq")
        psd_db = series.get("psd_db")
        freq_unit = series.get("freq_unit", "MHz")
        if freq is None or psd_db is None:
            return

        text = f'f={freq[idx]:.4f} {freq_unit}\nPSD={psd_db[idx]:.1f} dB'

        if key in self._hover_annotations:
            try:
                self._hover_annotations[key].remove()
            except Exception:
                pass

        annotation = ax.annotate(
            text,
            xy=(freq[idx], psd_db[idx]),
            xytext=(10, 10), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
            fontsize=8,
            zorder=100
        )
        self._hover_annotations[key] = annotation
        self.canvas.draw_idle()

    def _show_hover_annotation(self, ax, idx):
        """Show annotation with position information on hover"""
        if not self._trajectory_data:
            return

        positions = self._trajectory_data['positions']
        unit_label = self._trajectory_data['unit_label']
        time_scale = self._trajectory_data.get('time_scale') or 1e3
        time_label = self._trajectory_data.get('time_label') or "ms"
        t_val = self._trajectory_data['t'][idx] * time_scale

        pos = positions[idx]

        # Determine which axis and create annotation text
        axis_names = {self.ax_x: ('x', 0), self.ax_y: ('y', 1), self.ax_z: ('z', 2)}
        if ax in axis_names:
            axis_name, axis_idx = axis_names[ax]
            text = f't={t_val:.4f} {time_label}\n{axis_name}={pos[axis_idx]:.3f} {unit_label}\n(x,y,z)=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})'

            # Remove old annotation for this axis
            if axis_name in self._hover_annotations:
                try:
                    self._hover_annotations[axis_name].remove()
                except:
                    pass

            # Add new annotation
            annotation = ax.annotate(
                text,
                xy=(t_val, pos[axis_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                fontsize=8,
                zorder=100
            )
            self._hover_annotations[axis_name] = annotation

            self.canvas.draw_idle()

    def _remove_hover_annotations(self):
        """Remove all hover annotations"""
        for annotation in self._hover_annotations.values():
            if annotation is not None:
                try:
                    annotation.remove()
                except:
                    pass
        self._hover_annotations.clear()
        self.canvas.draw_idle()

    def _plot_fft_analysis(self, ax_x, ax_y, ax_z, t, positions):
        """Plot FFT frequency analysis of position data"""

        t = np.asarray(t, dtype=float)
        if t.size < 2:
            return
        dt_samples = np.diff(t)
        dt = float(np.mean(dt_samples))
        if not np.isfinite(dt) or dt <= 0:
            return
        if np.max(np.abs(dt_samples - dt)) > 1e-3 * dt:
            t_uniform = np.linspace(t[0], t[-1], len(t))
        else:
            t_uniform = t
        dt = float(t_uniform[1] - t_uniform[0])
        if not np.isfinite(dt) or dt <= 0:
            return
        fs = 1.0 / dt  # Sampling frequency (Hz)

        # Number of samples
        N = len(t_uniform)
        duration = t_uniform[-1] - t_uniform[0]
        min_plot_hz = 0.0
        if np.isfinite(duration) and duration > 0:
            min_plot_hz = 10.0 / duration

        # Determine frequency unit based on RF frequency or Nyquist frequency
        rf_freq_hz = None
        if self.sim and hasattr(self.sim, 'Omega'):
            rf_freq_hz = self.sim.Omega / (2 * np.pi)

        # Choose frequency scale based on RF frequency or Nyquist
        ref_freq = rf_freq_hz if rf_freq_hz else fs / 2
        if ref_freq >= 1e6:
            freq_scale = 1e6
            freq_unit = "MHz"
        elif ref_freq >= 1e3:
            freq_scale = 1e3
            freq_unit = "kHz"
        else:
            freq_scale = 1.0
            freq_unit = "Hz"

        # For each axis, compute FFT
        for ax, pos_data, label, color in [
            (ax_x, positions[:, 0], 'X', 'b'),
            (ax_y, positions[:, 1], 'Y', 'g'),
            (ax_z, positions[:, 2], 'Z', 'r')
        ]:
            pos_data = np.asarray(pos_data, dtype=float)
            if pos_data.size != N:
                pos_data = np.interp(t_uniform, t, pos_data)
            # Remove low-order trend + mean to suppress low-frequency drift
            trend_order = 2 if N >= 5 else 1
            coeffs = np.polyfit(t_uniform, pos_data, trend_order)
            pos_detrended = pos_data - np.polyval(coeffs, t_uniform)
            pos_centered = pos_detrended - np.mean(pos_detrended)

            # Apply Hanning window to reduce spectral leakage
            window = np.hanning(N)
            pos_windowed = pos_centered * window

            # Compute FFT
            fft_result = np.fft.rfft(pos_windowed)
            freqs = np.fft.rfftfreq(N, dt)

            # Compute power spectral density (magnitude squared)
            window_power = np.sum(window ** 2)
            psd = np.abs(fft_result) ** 2 / max(window_power, 1e-12)

            # Convert to dB scale for better visualization
            psd_db = 10 * np.log10(psd + 1e-20)  # Add small constant to avoid log(0)

            # Plot - exclude DC component (index 0) and very low frequencies
            freq_mask = (freqs > 0) & (freqs >= min_plot_hz)
            if np.count_nonzero(freq_mask) < 2:
                freq_mask = freqs > 0  # At minimum, exclude DC
            if np.any(freq_mask):
                freq_plot = freqs[freq_mask] / freq_scale
                psd_plot = psd_db[freq_mask]
                ax.plot(freq_plot, psd_plot, color=color, linewidth=0.8)
            else:
                freq_plot = freqs[1:] / freq_scale  # Skip DC bin
                psd_plot = psd_db[1:]
                ax.plot(freq_plot, psd_plot, color=color, linewidth=0.8)
            if isinstance(self._fft_data, dict):
                self._fft_data[label.lower()] = {
                    "freq": np.asarray(freq_plot, dtype=float),
                    "psd_db": np.asarray(psd_plot, dtype=float),
                    "freq_unit": freq_unit,
                }
            ax.set_xlabel(f'Frequency ({freq_unit})')
            ax.set_ylabel('Power (dB)')
            ax.set_title(f'{label} Frequency Spectrum')
            ax.grid(True, alpha=0.3, which='both')
            # Set x-limit based on Nyquist frequency in chosen units
            nyquist_scaled = (fs / 2) / freq_scale
            ax.set_xlim(0, nyquist_scaled)

            # Mark RF frequency if available
            if rf_freq_hz:
                rf_freq_scaled = rf_freq_hz / freq_scale
                if rf_freq_scaled < ax.get_xlim()[1]:
                    ax.axvline(rf_freq_scaled, color='red', linestyle='--', alpha=0.5, linewidth=1)
                    ax.text(rf_freq_scaled, ax.get_ylim()[1]*0.95, f'RF: {rf_freq_scaled:.2f} {freq_unit}',
                           rotation=90, va='top', fontsize=8)

    def _load_cad_mesh(self):
        #Load and process the mesh for visualization. Returns list of faces.
        if self._cad_mesh_data is not None:
            return self._cad_mesh_data

        mesh_path = self.numeric_field_dir / "mesh.msh"
        meta_path = self.numeric_field_dir / "facet_names.json"
        
        if not mesh_path.exists():
            self.update_status("Mesh file not found (run Mesh step first)", "warning")
            return None

        try:
            # 1. Read Metadata for Scaling and Physical Groups
            unit_scale = 1.0
            electrode_tags = set()
            
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    unit_scale = meta.get("unit_scale_to_m", 1.0)
                    # Collect all electrode tags to filter plotting
                    if "electrodes" in meta:
                        for name, tag in meta["electrodes"].items():
                            electrode_tags.add(tag)
            
            # 2. Read Mesh
            mesh = meshio.read(str(mesh_path))
            
            # 3. Extract Triangle Cells (Surfaces)
            if "triangle" not in mesh.cells_dict:
                self.update_status("CAD mesh has no triangle cells (surface mesh expected)", "warning")
                print(f"[CAD Mesh] Available cell types: {list(mesh.cells_dict.keys())}")
                return None
            
            tris = mesh.cells_dict["triangle"]
            subset_tris = tris
            
            # 4. Filter by Physical Group (if available) to only show electrodes
            # cell_data_dict['gmsh:physical'] usually aligns with cells
            if "gmsh:physical" in mesh.cell_data_dict:
                # iteration through cell blocks
                phys_tags = None
                
                # Method 1: Try cell_data_dict alignment (meshio >= 5.0)
                if "triangle" in mesh.cell_data_dict.get("gmsh:physical", {}):
                    phys_tags = mesh.cell_data_dict["gmsh:physical"]["triangle"]
                else:
                    # Method 2: Iterate through aligned cell/cell_data lists
                    for i, cell_block in enumerate(mesh.cells):
                        if cell_block.type == "triangle":
                            # Check if this index exists in cell_data
                            if i < len(mesh.cell_data.get("gmsh:physical", [])):
                                phys_tags = mesh.cell_data["gmsh:physical"][i]
                            break
                    
                if phys_tags is not None and electrode_tags:
                    # Filter: Keep only triangles that match an electrode tag
                    # masking
                    mask = np.isin(phys_tags, list(electrode_tags))
                    subset_tris = tris[mask]
            
            if subset_tris.size == 0:
                self.update_status("CAD mesh has no drawable faces after filtering", "warning")
                self._cad_mesh_data = None
                return None

            # 5. Get Vertices and apply ascale
            points = mesh.points * unit_scale

            # 6. Extract boundary edges (outer/inner outlines)
            # For a cleaner wireframe, we want ONLY the silhouette/outline edges
            edges = np.vstack([
                subset_tris[:, [0, 1]],
                subset_tris[:, [1, 2]],
                subset_tris[:, [2, 0]],
            ])
            edges_sorted = np.sort(edges, axis=1)
            unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)

            # Strategy: Only show edges that are on the boundary OR at sharp angles
            # For Paul traps, we typically want the outer frame edges
            boundary_edges = unique_edges[counts == 1]

            wireframe_fallback = False
            if boundary_edges.size == 0:
                # Closed surfaces have no boundary edges - use all unique edges but subsample heavily
                # This happens when the mesh represents solid volumes rather than open surfaces
                boundary_edges = unique_edges
                wireframe_fallback = True
                self.update_status(
                    "CAD mesh: showing subsampled edges (closed surface)",
                    "info",
                )

            max_edges = MAX_CAD_EDGES_WIREFRAME if wireframe_fallback else MAX_CAD_EDGES
            if boundary_edges.shape[0] > max_edges:
                step = (boundary_edges.shape[0] + max_edges - 1) // max_edges
                boundary_edges = boundary_edges[::step]
                self.update_status(
                    f"CAD mesh boundary edges downsampled to {len(boundary_edges)} segments",
                    "info",
                )

            edge_segments = points[boundary_edges]

            # Cache the result: Vertices are now in meters
            self._cad_mesh_data = edge_segments
            self.update_status(f"Loaded CAD mesh: {len(edge_segments)} boundary edges", "success")
            return edge_segments
            
        except Exception as e:
            self.update_status(f"Failed to load CAD mesh: {e}", "error")
            return None

    def _draw_beam_marker(self, ax, unit_scale_plot=1e6):
        """Draw beam position and cross-section marker in 3D view"""
        if not hasattr(self, "beam_panel"):
            return

        beam_params = self.beam_panel.get_parameters()
        if not beam_params.enabled:
            return

        # Map axis to index
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        prop_axis = axis_map[beam_params.propagation_axis]

        # Get the two transverse axes
        all_axes = [0, 1, 2]
        transverse_axes = [i for i in all_axes if i != prop_axis]

        # Convert beam parameters to plot units
        center = [0, 0, 0]
        center[transverse_axes[0]] = beam_params.beam_center[0] * unit_scale_plot
        center[transverse_axes[1]] = beam_params.beam_center[1] * unit_scale_plot
        center[prop_axis] = 0  # Draw at origin along propagation axis

        radius = beam_params.beam_radius * unit_scale_plot

        # Draw circular cross-section perpendicular to propagation
        theta = np.linspace(0, 2*np.pi, 40)
        circle_x = np.zeros(len(theta))
        circle_y = np.zeros(len(theta))
        circle_z = np.zeros(len(theta))

        # Set coordinates based on propagation axis
        if prop_axis == 0:  # Propagating along x
            circle_y = center[1] + radius * np.cos(theta)
            circle_z = center[2] + radius * np.sin(theta)
            circle_x = np.full_like(theta, center[0])
        elif prop_axis == 1:  # Propagating along y
            circle_x = center[0] + radius * np.cos(theta)
            circle_z = center[2] + radius * np.sin(theta)
            circle_y = np.full_like(theta, center[1])
        else:  # Propagating along z
            circle_x = center[0] + radius * np.cos(theta)
            circle_y = center[1] + radius * np.sin(theta)
            circle_z = np.full_like(theta, center[2])

        # Determine color based on beam type
        color = 'cyan' if beam_params.beam_type.value == 'electron' else 'orange'

        # Draw beam cross-section circle
        ax.plot(circle_x, circle_y, circle_z, color=color, linewidth=2,
                alpha=0.7, label=f'Beam ({beam_params.beam_type.value})')

        # Draw crosshairs at center
        cross_size = radius * 0.3
        if prop_axis == 0:  # x-axis
            ax.plot([center[0], center[0]], [center[1]-cross_size, center[1]+cross_size],
                   [center[2], center[2]], color=color, linewidth=1.5, alpha=0.8)
            ax.plot([center[0], center[0]], [center[1], center[1]],
                   [center[2]-cross_size, center[2]+cross_size], color=color, linewidth=1.5, alpha=0.8)
        elif prop_axis == 1:  # y-axis
            ax.plot([center[0]-cross_size, center[0]+cross_size], [center[1], center[1]],
                   [center[2], center[2]], color=color, linewidth=1.5, alpha=0.8)
            ax.plot([center[0], center[0]], [center[1], center[1]],
                   [center[2]-cross_size, center[2]+cross_size], color=color, linewidth=1.5, alpha=0.8)
        else:  # z-axis
            ax.plot([center[0]-cross_size, center[0]+cross_size], [center[1], center[1]],
                   [center[2], center[2]], color=color, linewidth=1.5, alpha=0.8)
            ax.plot([center[0], center[0]], [center[1]-cross_size, center[1]+cross_size],
                   [center[2], center[2]], color=color, linewidth=1.5, alpha=0.8)

        # Draw direction arrow along propagation axis
        arrow_length = radius * 3
        arrow_start = center.copy()
        arrow_end = center.copy()
        arrow_start[prop_axis] = -arrow_length / 2
        arrow_end[prop_axis] = arrow_length / 2

        ax.plot([arrow_start[0], arrow_end[0]],
               [arrow_start[1], arrow_end[1]],
               [arrow_start[2], arrow_end[2]],
               color=color, linewidth=2, alpha=0.5, linestyle='--')

    def _draw_cad_overlay(self, ax, unit_scale_plot=1e6):

        if not self.show_cad_mesh_var.get():
            return

        current_dir = str(self.numeric_field_dir)
        if current_dir not in self._numeric_overlay_cache:
            self._cad_mesh_data = None
            self._numeric_overlay_cache.clear()
            self._numeric_overlay_cache[current_dir] = True

        edges_m = self._load_cad_mesh()
        if edges_m is None or edges_m.size == 0:
            return

        # Convert meters to plot units
        segments = edges_m * unit_scale_plot

        mesh_collection = Line3DCollection(
            segments,
            colors='tab:blue',
            linewidths=0.6,
            alpha=0.9
        )

        ax.add_collection3d(mesh_collection)


class NumericFieldPanel(ttk.Frame):
    #Panel for CAD numeric field management
    
    def __init__(self, parent, on_paths_changed=None):
        super().__init__(parent)
        self.on_paths_changed = on_paths_changed
        
        self.electrodes_dir_var = tk.StringVar(value=str(DEFAULT_ELECTRODES))
        self.numeric_out_var = tk.StringVar(value=str(DEFAULT_NUMERIC_OUT))
        self.pipeline_thread = None
        self.pipeline_queue = Queue()
        self._pipeline_running = False
        
        self.build_panel()
        ensure_project_structure()
        self.update_status()
        self.after(100, self._check_pipeline_queue)
    
    def build_panel(self):
        # Paths
        path_frame = ttk.LabelFrame(self, text="Directories", padding=8)
        path_frame.grid(row=0, column=0, sticky="we", padx=4, pady=4)
        
        ttk.Label(path_frame, text="Electrodes:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(path_frame, textvariable=self.electrodes_dir_var, width=25).grid(row=0, column=1, sticky="we", pady=2)
        ttk.Button(path_frame, text="Browse", command=self.browse_electrodes, width=8).grid(row=0, column=2, padx=4, pady=2)
        
        ttk.Label(path_frame, text="Output:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(path_frame, textvariable=self.numeric_out_var, width=25).grid(row=1, column=1, sticky="we", pady=2)
        ttk.Button(path_frame, text="Browse", command=self.browse_output, width=8).grid(row=1, column=2, padx=4, pady=2)
        
        path_frame.columnconfigure(1, weight=1)
        
        # Actions
        action_frame = ttk.LabelFrame(self, text="Actions", padding=8)
        action_frame.grid(row=1, column=0, sticky="we", padx=4, pady=4)
        
        self.mesh_button = ttk.Button(action_frame, text="Mesh CAD Files", command=self.run_mesh)
        self.mesh_button.pack(fill='x', pady=2)
        self.solve_button = ttk.Button(action_frame, text="Solve Basis Fields", command=self.run_solve)
        self.solve_button.pack(fill='x', pady=2)
        self.clear_cache_button = ttk.Button(action_frame, text="Clear Grid Cache", command=self.clear_grid_cache)
        self.clear_cache_button.pack(fill='x', pady=2)
        self.refresh_button = ttk.Button(action_frame, text="Refresh Status", command=self.update_status)
        self.refresh_button.pack(fill='x', pady=2)
        
        # Status
        status_frame = ttk.LabelFrame(self, text="Status", padding=8)
        status_frame.grid(row=2, column=0, sticky="nsew", padx=4, pady=4)
        self.rowconfigure(2, weight=1)
        
        status_scroll = ttk.Scrollbar(status_frame, orient='vertical')
        self.status_text = tk.Text(status_frame, height=10, width=35, state='disabled',
                                   yscrollcommand=status_scroll.set, wrap='word')
        status_scroll.config(command=self.status_text.yview)
        self.status_text.pack(side='left', fill='both', expand=True)
        status_scroll.pack(side='right', fill='y')
    
    def browse_electrodes(self):
        directory = filedialog.askdirectory(initialdir=self.electrodes_dir_var.get())
        if directory:
            self.electrodes_dir_var.set(directory)
            self.notify_paths_changed()
            self.update_status()
    
    def browse_output(self):
        directory = filedialog.askdirectory(initialdir=self.numeric_out_var.get())
        if directory:
            self.numeric_out_var.set(directory)
            self.notify_paths_changed()
            self.update_status()
    
    def notify_paths_changed(self):
        if self.on_paths_changed:
            electrodes_dir = resolve_project_path(self.electrodes_dir_var.get())
            numeric_out = resolve_project_path(self.numeric_out_var.get())
            self.on_paths_changed((str(electrodes_dir), str(numeric_out)))
    
    def log(self, message):
        self.status_text.configure(state='normal')
        self.status_text.insert('end', message + '\n')
        self.status_text.see('end')
        self.status_text.configure(state='disabled')
    
    def update_status(self):
        self.status_text.configure(state='normal')
        self.status_text.delete('1.0', 'end')
        
        electrodes_dir = resolve_project_path(self.electrodes_dir_var.get())
        numeric_out = resolve_project_path(self.numeric_out_var.get())
        
        # Check electrodes
        present, cad_files, ignored = electrodes_present(electrodes_dir)
        if present:
            self.log(f"Found {len(cad_files)} CAD files:")
            for f in cad_files[:5]:  # Show first 5
                self.log(f" {f.name}")
            if len(cad_files) > 5:
                self.log(f"  ... and {len(cad_files) - 5} more")
        else:
            self.log("[Hint]No CAD files found in electrodes directory")
        
        # Check numeric status
        ready, reason = numeric_ready(numeric_out)
        if ready:
            self.log(f"Numeric fields ready: {reason}")
        else:
            self.log(f"{reason}")

        # Show grid cache status
        try:
            import grid_cache
            stats = grid_cache.get_cache_stats()
            if stats['cached_grids'] > 0:
                self.log(f"\nGrid cache: {stats['cached_grids']} grid(s) cached")
                self.log("(Reusing grids speeds up simulations)")
        except ImportError:
            pass

        self.status_text.configure(state='disabled')
    
    def run_mesh(self):
        self._start_pipeline("mesh")
    
    def run_solve(self):
        self._start_pipeline("solve")

    def clear_grid_cache(self):
        """Clear the global grid cache to free memory or force rebuild"""
        import grid_cache
        stats = grid_cache.get_cache_stats()
        if stats['cached_grids'] > 0:
            grid_cache.clear_cache()
            self.log(f"Cleared {stats['cached_grids']} cached grid(s)")
            self.log("Next simulation will rebuild grid from scratch")
        else:
            self.log("Grid cache is already empty")

    def _set_action_state(self, state):
        for btn in (self.mesh_button, self.solve_button, self.clear_cache_button, self.refresh_button):
            btn.configure(state=state)

    def _start_pipeline(self, step):
        if self._pipeline_running:
            self.log("[Hint] A pipeline step is already running.")
            return
        self._pipeline_running = True
        self._set_action_state("disabled")
        self.log(f"\n--- Running {step.capitalize()} ---")
        electrodes_dir = resolve_project_path(self.electrodes_dir_var.get())
        numeric_out = resolve_project_path(self.numeric_out_var.get())
        self.log(f"[Info] Electrodes dir: {electrodes_dir}")
        self.log(f"[Info] Output dir: {numeric_out}")
        self.pipeline_thread = threading.Thread(
            target=self._pipeline_worker,
            args=(step, electrodes_dir, numeric_out),
            daemon=True
        )
        self.pipeline_thread.start()
    
    def _pipeline_worker(self, step, electrodes_dir, numeric_out):
        def queue_log(message):
            self.pipeline_queue.put({"type": "log", "message": message})
        try:
            success, msg = run_pipeline_step(
                step,
                electrodes_dir,
                numeric_out,
                log_func=queue_log
            )
        except Exception as exc:
            success = False
            msg = str(exc)
        self.pipeline_queue.put({
            "type": "done",
            "step": step,
            "success": success,
            "message": msg,
        })

    def _check_pipeline_queue(self):
        try:
            while True:
                item = self.pipeline_queue.get_nowait()
                if item["type"] == "log":
                    self.log(item["message"])
                elif item["type"] == "done":
                    step = item["step"]
                    if item["success"]:
                        if step == "mesh":
                            self.log("Mesh generation complete")
                        else:
                            self.log("Solve complete")
                    else:
                        self.log(f"[Error] {step.capitalize()} failed: {item['message']}")
                    self.update_status()
                    self._set_action_state("normal")
                    self._pipeline_running = False
                    self.pipeline_thread = None
                self.pipeline_queue.task_done()
        except Empty:
            pass
        finally:
            self.after(100, self._check_pipeline_queue)


def main():
    root = tk.Tk()
    root.geometry("1400x800")
    app = PaulTrapGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()