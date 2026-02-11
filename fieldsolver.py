import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter
import time


#Numba for JIT compilation 
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("[Performance] Numba not available. Install with: pip install numba")
    # Dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


def _eval_program_value(value, t):
    if callable(value):
        return float(value(t))
    return float(value)


def _decompose_voltage_program(trap):
    """ Decompose numeric electrode voltages into DC and RF components based on the program and trap parameters.
        returns: (volts_dc, volts_rf) """
    
    electrodes = list(getattr(trap, "numeric_electrodes", []) or [])
    program = getattr(trap, "electrode_program", {}) or {}
    has_program = bool(program)
    omega = float(getattr(trap, "Omega", 0.0) or 0.0)
    if np.isfinite(omega) and omega != 0.0:
        t_dc = np.pi / (2.0 * omega)
    else:
        t_dc = 0.0
    t_rf = 0.0

    volts_dc = {}
    volts_rf = {}
    missing = []
    for idx, name in enumerate(electrodes):
        if name in program:
            try:
                v_dc = _eval_program_value(program[name], t_dc)
                v_rf = _eval_program_value(program[name], t_rf) - v_dc
            except Exception:
                v_dc = float(getattr(trap, "V_dc", 0.0))
                v_rf = float(getattr(trap, "V_rf", 0.0))
        else:
            if has_program:
                v_dc = 0.0
                v_rf = 0.0
                missing.append(name)
            elif len(electrodes) == 1:
                v_dc = float(getattr(trap, "V_dc", 0.0))
                v_rf = float(getattr(trap, "V_rf", 0.0))
            else:
                sign = 1.0 if (idx % 2 == 0) else -1.0
                v_dc = float(getattr(trap, "V_dc", 0.0))
                v_rf = sign * float(getattr(trap, "V_rf", 0.0))
        volts_dc[name] = v_dc
        volts_rf[name] = v_rf

    if missing and not getattr(trap, "_warned_numeric_program_missing", False):
        print("[numeric] Electrode program missing for:", ", ".join(missing))
        print("[numeric] Missing electrodes set to 0 V.")
        trap._warned_numeric_program_missing = True

    return volts_dc, volts_rf


class solveanalyticfield:
    """Analytic quadrupole field solver for idealized Paul trap geometries (linear and 3D)."""
    def __init__(self, trap):
        self.trap = trap
        
        # Pre-compute coefficients
        self.coeff_rf = trap.field_coeff_rf
        self.coeff_dc = trap.field_coeff_dc
        self.a = trap.a
        self.b = trap.b
        self.c = trap.c
        self.Omega = trap.Omega
        
        # Linear trap parameters
        self.is_linear = (trap.trap_type == "Linear")
        if self.is_linear and trap.axial_dc_U is not None:
            self.axial_coeff = -(2.0 * trap.axial_dc_kappa * trap.axial_dc_U 
                                / (trap.axial_dc_z0**2))
        else:
            self.axial_coeff = 0.0
    
    def field(self, x, y, z, t):
        #field calulations
        # Time-dependent part
        cos_term = np.cos(self.Omega * t)
        
        # DC contribution depends on position; compute each call
        Ex_dc = self.coeff_dc * self.a * x
        Ey_dc = self.coeff_dc * self.b * y
        Ez_dc = self.coeff_dc * self.c * z
        
        # Add RF contribution (vectorized)
        Ex = self.coeff_rf * cos_term * self.a * x + Ex_dc
        Ey = self.coeff_rf * cos_term * self.b * y + Ey_dc
        Ez = self.coeff_rf * cos_term * self.c * z + Ez_dc
        
        # Axial DC for linear traps
        if self.is_linear and self.axial_coeff != 0.0:
            Ez += self.axial_coeff * z
        
        return Ex, Ey, Ez
    
    def secular_gradient(self, x, y, z):
        """Gradient of |E_RF|^2 for secular approximation"""
        E_scale_sq = (self.trap.V_rf / self.trap.r0**2)**2
        
        # grad(|E|^2) = 2 * grad(E_i * E_i) where E_i are RF field components
        # For quadrupole: E_x = -a*V/r0^2 * x, etc.
        # |E|^2 = (a^2*x^2 + b^2*y^2 + c^2*z^2) * (V/r0^2)^2
        # grad(|E|^2) = 2*(V/r0^2)^2 * [a^2*x, b^2*y, c^2*z]
        
        gx = 2 * E_scale_sq * self.a**2 * x
        gy = 2 * E_scale_sq * self.b**2 * y
        gz = 2 * E_scale_sq * self.c**2 * z
        
        return gx, gy, gz


class solvenumericfield:
    """Numeric field solver using pre-computed gradients from the mesh. Optimized for repeated evaluations."""
    def __init__(self, trap):
        self.trap = trap
        #cache the decomposed voltage dictionaries
        self._volts_dc = None
        self._volts_rf = None
        #cache the combined gradient fields
        self._G_elem_dc = None
        self._G_elem_rf = None
        self._PHI_nodes_dc = None
        self._PHI_nodes_rf = None
        self._cache_initialized = False
        self.Omega = trap.Omega
        self._warned_outside_mesh = False
        self._warned_secular_gradient = False
        
        # Performance monitoring
        self.eval_count = 0
        self.total_time = 0.0
    
    def _initialize_cache(self):
        #Decompose voltages into DC and RF components and precompute combined fields.
        if self._cache_initialized:
            return
        
        self._volts_dc, self._volts_rf = _decompose_voltage_program(self.trap)
        
        print("[numeric] Voltage assignment:")
        for name in self.trap.numeric_electrodes:
            print(f"  {name}: DC={self._volts_dc.get(name, 0):.2f}V, RF_amp={self._volts_rf.get(name, 0):.2f}V")
        
        # Precompute combined gradient fields for DC and RF
        n_elems = self.trap.numeric.tets.shape[0] # number of tetrahedral elements
        n_nodes = self.trap.numeric.nodes.shape[0] # number of nodes
        
        self._G_elem_dc = np.zeros((n_elems, 3), dtype=np.float64) # Gradient per element for DC
        self._G_elem_rf = np.zeros((n_elems, 3), dtype=np.float64) # Gradient per element for RF
        self._PHI_nodes_dc = np.zeros(n_nodes, dtype=np.float64) # Potential per node for DC
        self._PHI_nodes_rf = np.zeros(n_nodes, dtype=np.float64) # Potential per node for RF
        
        # Sum up DC contributions
        for name, V_dc in self._volts_dc.items():
            if abs(V_dc) > 0.0 and name in self.trap.numeric.bases:
                self._G_elem_dc += V_dc * self.trap.numeric.bases[name]["grads"]
                self._PHI_nodes_dc += V_dc * self.trap.numeric.bases[name]["phi"]
        
        # Sum up RF contributions
        for name, V_rf in self._volts_rf.items():
            if abs(V_rf) > 0.0 and name in self.trap.numeric.bases:
                self._G_elem_rf += V_rf * self.trap.numeric.bases[name]["grads"]
                self._PHI_nodes_rf += V_rf * self.trap.numeric.bases[name]["phi"]
        
        self._cache_initialized = True
    
    def field(self, x, y, z, t):
        # Calculate electric field at given position and time
        start_time = time.time()
        self.eval_count += 1
        
        # Initialize cache on first call
        self._initialize_cache()
        
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        z_arr = np.asarray(z)
        pts = np.column_stack([
            x_arr.reshape(-1),
            y_arr.reshape(-1),
            z_arr.reshape(-1)
        ]).astype(float)
        
        N = pts.shape[0]
        E = np.zeros((N, 3), dtype=np.float64)
        
        # Combine DC and RF fields with time modulation
        cos_term = np.cos(self.Omega * t)
        G_elem_combined = self._G_elem_dc + cos_term * self._G_elem_rf
        
        # Use optimized evaluation
        if hasattr(self.trap.numeric, 'evaluate_fast'):
            _, E_flat = self.trap.numeric.evaluate_fast(pts, self._volts_dc)
            Ex_dc, Ey_dc, Ez_dc = E_flat[:, 0], E_flat[:, 1], E_flat[:, 2]
            
            _, E_flat = self.trap.numeric.evaluate_fast(pts, self._volts_rf)
            Ex_rf, Ey_rf, Ez_rf = E_flat[:, 0], E_flat[:, 1], E_flat[:, 2]
            
            Ex = Ex_dc + cos_term * Ex_rf
            Ey = Ey_dc + cos_term * Ey_rf
            Ez = Ez_dc + cos_term * Ez_rf
        else:
            # Batch find elements
            element_ids = self.trap.numeric._find_elements_batch(pts)
            
            # Evaluate fields for all valid points
            valid_mask = element_ids >= 0
            valid_elements = element_ids[valid_mask]
            
            # Electric field (constant per element)
            E[valid_mask] = -G_elem_combined[valid_elements]
            outside_count = int(np.sum(~valid_mask))
            if outside_count:
                E[~valid_mask] = np.nan
                if not self._warned_outside_mesh:
                    print(f"[numeric] Warning: {outside_count} point(s) outside mesh; field undefined.")
                    self._warned_outside_mesh = True
            self._last_outside_count = outside_count
            
            Ex = E[:, 0].reshape(x_arr.shape)
            Ey = E[:, 1].reshape(y_arr.shape)
            Ez = E[:, 2].reshape(z_arr.shape)
        
        self.total_time += time.time() - start_time
        
        # Log performance every 1000 evaluations
        if self.eval_count % 1000 == 0:
            avg_time = self.total_time / self.eval_count * 1000  # ms per eval
            print(f"[numeric] Field eval performance: {avg_time:.2f} ms/eval (total: {self.eval_count})")
        
        return Ex, Ey, Ez
    
    def secular_gradient(self, x, y, z):
        #Gradient of |E_RF|^2 for secular approximation
        if not self._warned_secular_gradient:
            print("[numeric] secular_gradient: using finite differences")
            self._warned_secular_gradient = True
        
        # Compute |E_RF|^2 at point
        # Use small displacement for finite difference
        h = 1e-8  # meters
        
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        z_arr = np.asarray(z)
        
        # Evaluate RF field (at t=0 where cos=1, pure RF amplitude)
        Ex0, Ey0, Ez0 = self.field(x_arr, y_arr, z_arr, 0.0)
        E_sq_0 = Ex0**2 + Ey0**2 + Ez0**2
        
        # X gradient
        Ex_px, Ey_px, Ez_px = self.field(x_arr + h, y_arr, z_arr, 0.0)
        E_sq_px = Ex_px**2 + Ey_px**2 + Ez_px**2
        gx = (E_sq_px - E_sq_0) / h
        
        # Y gradient
        Ex_py, Ey_py, Ez_py = self.field(x_arr, y_arr + h, z_arr, 0.0)
        E_sq_py = Ex_py**2 + Ey_py**2 + Ez_py**2
        gy = (E_sq_py - E_sq_0) / h
        
        # Z gradient
        Ex_pz, Ey_pz, Ez_pz = self.field(x_arr, y_arr, z_arr + h, 0.0)
        E_sq_pz = Ex_pz**2 + Ey_pz**2 + Ez_pz**2
        gz = (E_sq_pz - E_sq_0) / h
        
        return gx, gy, gz


# Numba-optimized Coulomb force calculation
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def compute_coulomb_forces_numba(positions, coulomb_coeff, epsilon=1e-12):
        #Optimized Coulomb force calculation using Numba. Returns: F_coulomb array of shape (n_particles, 3)

        n = positions.shape[0]
        F_coulomb = np.zeros((n, 3), dtype=np.float64)
        
        # Parallel loop over particles
        for i in prange(n):
            fx, fy, fz = 0.0, 0.0, 0.0
            
            for j in range(n):
                if i != j:
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dz = positions[i, 2] - positions[j, 2]
                    
                    r2 = dx*dx + dy*dy + dz*dz + epsilon
                    r = np.sqrt(r2)
                    
                    force_mag = coulomb_coeff / r2
                    
                    fx += force_mag * dx / r
                    fy += force_mag * dy / r
                    fz += force_mag * dz / r
            
            F_coulomb[i, 0] = fx
            F_coulomb[i, 1] = fy
            F_coulomb[i, 2] = fz
        
        return F_coulomb
else:
    # Fallback NumPy version
    def compute_coulomb_forces_numba(positions, coulomb_coeff, epsilon=1e-12):
        n = positions.shape[0]
        F_coulomb = np.zeros((n, 3), dtype=np.float64)
        
        dx = positions[:, np.newaxis, 0] - positions[np.newaxis, :, 0]
        dy = positions[:, np.newaxis, 1] - positions[np.newaxis, :, 1]
        dz = positions[:, np.newaxis, 2] - positions[np.newaxis, :, 2]
        
        r = np.sqrt(dx**2 + dy**2 + dz**2 + epsilon)
        np.fill_diagonal(r, 1.0)
        
        F_mag = coulomb_coeff / (r**2)
        np.fill_diagonal(F_mag, 0.0)
        
        F_coulomb[:, 0] = np.sum(F_mag * (dx / r), axis=1)
        F_coulomb[:, 1] = np.sum(F_mag * (dy / r), axis=1)
        F_coulomb[:, 2] = np.sum(F_mag * (dz / r), axis=1)
        
        return F_coulomb


def _extrapolate_nan_nearest(data_3d):
    """
    Replace NaN values with nearest valid neighbor values using distance transform.
    This preserves field structure better than zero-filling.
    """
    mask = ~np.isfinite(data_3d)
    if not np.any(mask):
        return data_3d

    nan_count = np.sum(mask)
    valid_count = np.sum(~mask)

    if valid_count == 0:
        print(f"[GridField] Warning: all values are NaN, setting to 0")
        return np.zeros_like(data_3d)

    # Use distance transform to find indices of nearest valid neighbors
    # This computes distance to nearest False (valid) cell
    indices = distance_transform_edt(mask, return_distances=False, return_indices=True)

    # Use the indices to fill NaN values with nearest valid values
    result = data_3d[tuple(indices)]

    # Only print if significant NaN count
    total = nan_count + valid_count
    if nan_count > total * 0.3:  # More than 30%
        print(f"[GridField] Extrapolated {100*nan_count/total:.1f}% NaN values")

    return result


def _create_adaptive_grid(bounds, base_points, refinement_radii=None, refinement_heights=None, refinement_factor=3):
    """
    Create adaptive grid with higher density near specified radii and heights.

    Args:
        bounds: ((xmin, xmax), (ymin, ymax), (zmin, zmax))
        base_points: base number of grid points per dimension
        refinement_radii: list of (radius, width) tuples for radial refinement (in same units as bounds)
        refinement_heights: list of (height, width) tuples for z refinement
        refinement_factor: how many times denser the refined regions should be

    Returns:
        x_grid, y_grid, z_grid arrays
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds

    if refinement_radii is None and refinement_heights is None:
        # No adaptive refinement, use uniform grid
        return (
            np.linspace(xmin, xmax, base_points),
            np.linspace(ymin, ymax, base_points),
            np.linspace(zmin, zmax, base_points)
        )

    # Create adaptive grids using concatenation of uniform and refined regions
    def make_adaptive_1d(vmin, vmax, base_n, critical_points):
        """Create 1D grid with refinement near critical points."""
        if not critical_points:
            return np.linspace(vmin, vmax, base_n)

        # Start with base grid
        base_spacing = (vmax - vmin) / (base_n - 1)
        fine_spacing = base_spacing / refinement_factor

        # Collect all grid points
        points = set()

        # Add base grid
        for v in np.linspace(vmin, vmax, base_n):
            points.add(v)

        # Add fine grid near critical points
        for (center, width) in critical_points:
            if vmin <= center <= vmax:
                region_min = max(vmin, center - width)
                region_max = min(vmax, center + width)
                n_fine = max(3, int((region_max - region_min) / fine_spacing))
                for v in np.linspace(region_min, region_max, n_fine):
                    points.add(v)

        return np.array(sorted(points))

    # For x and y, critical points are at radius values
    xy_critical = []
    if refinement_radii:
        for (radius, width) in refinement_radii:
            xy_critical.append((radius, width))
            xy_critical.append((-radius, width))

    x_grid = make_adaptive_1d(xmin, xmax, base_points, xy_critical)
    y_grid = make_adaptive_1d(ymin, ymax, base_points, xy_critical)

    # For z, use refinement_heights directly
    z_critical = refinement_heights if refinement_heights else []
    z_grid = make_adaptive_1d(zmin, zmax, base_points, z_critical)

    return x_grid, y_grid, z_grid


class GridInterpolatedField:
    """Pre-compute field on a 3D grid and use interpolation.

    Improvements over uniform grid:
    1. NaN handling: extrapolates from nearest valid neighbors instead of zeroing
    2. Adaptive grid: higher resolution near electrode gaps (optional)
    3. Higher default resolution: 100 instead of 50
    """

    def __init__(self, trap, grid_points=100, bounds=None,
                 adaptive_refinement=True, refinement_radii=None, refinement_heights=None,
                 smoothing_sigma=0.0, smoothing_mode="nearest", eval_batch_size=500000,
                 progress_callback=None, stop_check=None):
        """
        Args:
            trap: TrapSimulator instance
            grid_points: base number of grid points per dimension (default 100)
            bounds: optional custom bounds ((xmin,xmax), (ymin,ymax), (zmin,zmax))
            adaptive_refinement: whether to use adaptive grid near electrode gaps
            refinement_radii: list of (radius, width) in meters for radial refinement
            refinement_heights: list of (height, width) in meters for z refinement
        """
        if grid_points is None:
            grid_points = 0
        if isinstance(grid_points, (float, np.floating)):
            try:
                grid_points = int(grid_points)
            except (TypeError, ValueError):
                grid_points = 0
        if grid_points and grid_points > 0:
            print(f"[GridField] Pre-computing numeric field (base {grid_points} per axis)...")
        else:
            print("[GridField] Pre-computing numeric field (auto resolution)...")

        self.trap = trap
        self.Omega = trap.Omega
        self._warned_outside_grid = False
        self.progress_callback = progress_callback if callable(progress_callback) else None
        self.stop_check = stop_check if callable(stop_check) else None
        try:
            self.smoothing_sigma = float(smoothing_sigma)
        except (TypeError, ValueError):
            self.smoothing_sigma = 0.0
        if not np.isfinite(self.smoothing_sigma) or self.smoothing_sigma < 0:
            self.smoothing_sigma = 0.0
        self.smoothing_mode = smoothing_mode or "nearest"
        self.min_grid_spacing = None
        try:
            self.eval_batch_size = int(eval_batch_size)
        except (TypeError, ValueError):
            self.eval_batch_size = 200000
        if self.eval_batch_size <= 0:
            self.eval_batch_size = 200000

        def check_stop():
            if self.stop_check and self.stop_check():
                raise RuntimeError("Simulation stopped by user")

        # Determine bounds from mesh
        nodes = trap.numeric.nodes
        margin = 1.1  # Slightly tighter margin
        xmin, xmax = nodes[:, 0].min() * margin, nodes[:, 0].max() * margin
        ymin, ymax = nodes[:, 1].min() * margin, nodes[:, 1].max() * margin
        zmin, zmax = nodes[:, 2].min() * margin, nodes[:, 2].max() * margin

        span_x = float(xmax - xmin)
        span_y = float(ymax - ymin)
        span_z = float(zmax - zmin)
        span = max(span_x, span_y, span_z)

        def compute_auto_grid_points(span_val):
            min_points = 60
            max_points = 200
            if not np.isfinite(span_val) or span_val <= 0:
                return 100, None
            target_spacing = span_val / 140.0
            min_spacing = span_val / max_points
            max_spacing = span_val / min_points
            if target_spacing < min_spacing:
                target_spacing = min_spacing
            if target_spacing > max_spacing:
                target_spacing = max_spacing
            points = int(np.clip(np.ceil(span_val / target_spacing), min_points, max_points))
            return points, target_spacing

        auto_points, auto_spacing = compute_auto_grid_points(span)
        self.auto_grid_points = auto_points
        if self.progress_callback:
            self.progress_callback("grid", 0.0, f"Gridfield auto {auto_points}")

        if grid_points <= 0:
            grid_points = auto_points

        # Auto-detect refinement regions for surface traps if not specified
        if adaptive_refinement and refinement_radii is None:
            # Try to infer electrode gap locations from mesh geometry
            r_values = np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2)
            r_min, r_max = r_values.min(), r_values.max()

            # Common surface trap geometry: gaps at inner and outer RF electrode edges
            # Typical gap width is ~100 um, so refine within 200 um of likely boundaries
            gap_width = 200e-6  # 200 um refinement zone

            # Look for clustering in radial distribution to find electrode boundaries
            r_sorted = np.sort(r_values)
            dr = np.diff(r_sorted)

            # Find jumps (potential electrode gaps)
            large_gaps = np.where(dr > (r_max - r_min) / 20)[0]
            refinement_radii = []
            for gap_idx in large_gaps[:5]:  # Limit to 5 most significant gaps
                r_gap = (r_sorted[gap_idx] + r_sorted[gap_idx + 1]) / 2
                if r_gap > 1e-6:  # Skip near-zero radii
                    refinement_radii.append((r_gap, gap_width))

        if adaptive_refinement and refinement_heights is None:
            # For surface traps, refine near the electrode plane (z~0) and expected trap height
            z_min_mesh = nodes[:, 2].min()
            refinement_heights = [
                (z_min_mesh + 100e-6, 100e-6),   # Near electrode surface
                (z_min_mesh + 500e-6, 200e-6),   # Near typical trap height
                (z_min_mesh + 1000e-6, 200e-6),  # Another typical trap height
            ]

        # Create grid (adaptive or uniform)
        if adaptive_refinement and (refinement_radii or refinement_heights):
            self.x_grid, self.y_grid, self.z_grid = _create_adaptive_grid(
                ((xmin, xmax), (ymin, ymax), (zmin, zmax)),
                grid_points,
                refinement_radii=refinement_radii,
                refinement_heights=refinement_heights,
                refinement_factor=4
            )
            total_points = len(self.x_grid) * len(self.y_grid) * len(self.z_grid)
            print(f"[GridField] Building grid: {total_points:,} points")
        else:
            self.x_grid = np.linspace(xmin, xmax, grid_points)
            self.y_grid = np.linspace(ymin, ymax, grid_points)
            self.z_grid = np.linspace(zmin, zmax, grid_points)
            print(f"[GridField] Uniform grid: {grid_points}^3 = {grid_points**3:,} points")

        X, Y, Z = np.meshgrid(self.x_grid, self.y_grid, self.z_grid, indexing='ij')
        shape = (len(self.x_grid), len(self.y_grid), len(self.z_grid))

        if self.progress_callback:
            self.progress_callback("grid", 0.0, "Gridfield precompute")
        check_stop()

        # Pre-compute DC and RF fields
        print("[GridField] Computing fields...")
        volts_dc, volts_rf = _decompose_voltage_program(trap)
        points_flat = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        def chunk_count(total):
            if total <= 0:
                return 0
            return (total + self.eval_batch_size - 1) // self.eval_batch_size

        total_points = points_flat.shape[0]
        dc_active = any(abs(v) > 0.0 for v in volts_dc.values())
        rf_active = any(abs(v) > 0.0 for v in volts_rf.values())
        total_chunks = 0
        if dc_active:
            total_chunks += chunk_count(total_points)
        if rf_active:
            total_chunks += chunk_count(total_points)
        progress_done = 0

        def eval_in_chunks(volts, label):
            nonlocal progress_done
            eval_fast = getattr(trap.numeric, "evaluate_fast", None)
            eval_std = trap.numeric.evaluate
            total = points_flat.shape[0]
            E_out = np.empty((total, 3), dtype=np.float64)
            if total == 0:
                return E_out
            chunks = chunk_count(total)
            fast_threshold = min(20000, self.eval_batch_size)
            for chunk_idx, start in enumerate(range(0, total, self.eval_batch_size), start=1):
                check_stop()
                end = min(total, start + self.eval_batch_size)
                eval_fn = eval_std
                if callable(eval_fast) and (end - start) <= fast_threshold:
                    eval_fn = eval_fast
                _, E_chunk = eval_fn(points_flat[start:end], volts)
                E_out[start:end] = E_chunk
                progress_done += 1
                if self.progress_callback and total_chunks:
                    fraction = progress_done / total_chunks
                    self.progress_callback(
                        "grid",
                        fraction,
                        f"Computing {label} field ({chunk_idx}/{chunks})",
                    )
            return E_out
        if dc_active:
            E_dc = eval_in_chunks(volts_dc, "DC")
        else:
            E_dc = np.zeros((points_flat.shape[0], 3), dtype=np.float64)
        check_stop()

        # Reshape and apply smart NaN handling
        Ex_dc_grid = E_dc[:, 0].reshape(shape)
        Ey_dc_grid = E_dc[:, 1].reshape(shape)
        Ez_dc_grid = E_dc[:, 2].reshape(shape)

        if not np.all(np.isfinite(Ex_dc_grid)):
            Ex_dc_grid = _extrapolate_nan_nearest(Ex_dc_grid)
        if not np.all(np.isfinite(Ey_dc_grid)):
            Ey_dc_grid = _extrapolate_nan_nearest(Ey_dc_grid)
        if not np.all(np.isfinite(Ez_dc_grid)):
            Ez_dc_grid = _extrapolate_nan_nearest(Ez_dc_grid)

        if self.smoothing_sigma > 0.0:
            Ex_dc_grid = gaussian_filter(Ex_dc_grid, self.smoothing_sigma, mode=self.smoothing_mode)
            Ey_dc_grid = gaussian_filter(Ey_dc_grid, self.smoothing_sigma, mode=self.smoothing_mode)
            Ez_dc_grid = gaussian_filter(Ez_dc_grid, self.smoothing_sigma, mode=self.smoothing_mode)

        if rf_active:
            E_rf = eval_in_chunks(volts_rf, "RF")
        else:
            E_rf = np.zeros((points_flat.shape[0], 3), dtype=np.float64)
        check_stop()

        Ex_rf_grid = E_rf[:, 0].reshape(shape)
        Ey_rf_grid = E_rf[:, 1].reshape(shape)
        Ez_rf_grid = E_rf[:, 2].reshape(shape)

        if not np.all(np.isfinite(Ex_rf_grid)):
            Ex_rf_grid = _extrapolate_nan_nearest(Ex_rf_grid)
        if not np.all(np.isfinite(Ey_rf_grid)):
            Ey_rf_grid = _extrapolate_nan_nearest(Ey_rf_grid)
        if not np.all(np.isfinite(Ez_rf_grid)):
            Ez_rf_grid = _extrapolate_nan_nearest(Ez_rf_grid)

        if self.smoothing_sigma > 0.0:
            Ex_rf_grid = gaussian_filter(Ex_rf_grid, self.smoothing_sigma, mode=self.smoothing_mode)
            Ey_rf_grid = gaussian_filter(Ey_rf_grid, self.smoothing_sigma, mode=self.smoothing_mode)
            Ez_rf_grid = gaussian_filter(Ez_rf_grid, self.smoothing_sigma, mode=self.smoothing_mode)

        # Create interpolators with nearest-neighbor extrapolation for out-of-bounds
        print("[GridField] Building interpolators...")
        self.dc_interp = {
            'x': RegularGridInterpolator((self.x_grid, self.y_grid, self.z_grid), Ex_dc_grid, bounds_error=False, fill_value=None),
            'y': RegularGridInterpolator((self.x_grid, self.y_grid, self.z_grid), Ey_dc_grid, bounds_error=False, fill_value=None),
            'z': RegularGridInterpolator((self.x_grid, self.y_grid, self.z_grid), Ez_dc_grid, bounds_error=False, fill_value=None)
        }

        self.rf_interp = {
            'x': RegularGridInterpolator((self.x_grid, self.y_grid, self.z_grid), Ex_rf_grid, bounds_error=False, fill_value=None),
            'y': RegularGridInterpolator((self.x_grid, self.y_grid, self.z_grid), Ey_rf_grid, bounds_error=False, fill_value=None),
            'z': RegularGridInterpolator((self.x_grid, self.y_grid, self.z_grid), Ez_rf_grid, bounds_error=False, fill_value=None)
        }

        # Pre-compute pseudopotential gradients for secular approximation
        print("[GridField] Pre-computing pseudopotential gradients...")
        E_sq = Ex_rf_grid**2 + Ey_rf_grid**2 + Ez_rf_grid**2
        check_stop()

        # Compute gradients with adaptive spacing
        grads = np.gradient(E_sq, self.x_grid, self.y_grid, self.z_grid, edge_order=2)

        self.secular_interp = {
            'x': RegularGridInterpolator((self.x_grid, self.y_grid, self.z_grid), grads[0], bounds_error=False, fill_value=None),
            'y': RegularGridInterpolator((self.x_grid, self.y_grid, self.z_grid), grads[1], bounds_error=False, fill_value=None),
            'z': RegularGridInterpolator((self.x_grid, self.y_grid, self.z_grid), grads[2], bounds_error=False, fill_value=None)
        }

        # Report grid spacing
        if len(self.x_grid) > 1 and len(self.y_grid) > 1 and len(self.z_grid) > 1:
            dx = np.min(np.diff(self.x_grid))
            dy = np.min(np.diff(self.y_grid))
            dz = np.min(np.diff(self.z_grid))
            self.min_grid_spacing = float(np.min([dx, dy, dz]))
        dx = self.min_grid_spacing or 0.0
        print(f"[GridField] Min grid spacing: {dx*1e6:.1f} um")

        # Diagnostic: check field structure along z-axis at (0,0,z)
        ix_center = np.argmin(np.abs(self.x_grid))
        iy_center = np.argmin(np.abs(self.y_grid))

        # Find trap center (minimum of |E_rf| along z-axis at x=0, y=0)
        # Only consider positive z (above electrode surface)
        z_positive_mask = self.z_grid > 100e-6  # Above 100 μm
        z_line = self.z_grid[z_positive_mask]
        if len(z_line) > 0:
            E_rf_line = np.sqrt(
                Ex_rf_grid[ix_center, iy_center, z_positive_mask]**2 +
                Ey_rf_grid[ix_center, iy_center, z_positive_mask]**2 +
                Ez_rf_grid[ix_center, iy_center, z_positive_mask]**2
            )
            # Find minimum |E| (trap center)
            trap_idx = np.argmin(E_rf_line)
            self.trap_center_z = z_line[trap_idx]
        else:
            self.trap_center_z = 1.5e-3  # Default fallback

        print("[GridField] Ready!")
        if self.progress_callback:
            self.progress_callback("grid", 1.0, "Gridfield ready")
    
    def field(self, x, y, z, t):
        # Stack coordinates
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        z_arr = np.asarray(z)
        
        pts = np.column_stack([
            x_arr.ravel(),
            y_arr.ravel(),
            z_arr.ravel()
        ])

        outside_mask = (
            (pts[:, 0] < self.x_grid[0]) | (pts[:, 0] > self.x_grid[-1]) |
            (pts[:, 1] < self.y_grid[0]) | (pts[:, 1] > self.y_grid[-1]) |
            (pts[:, 2] < self.z_grid[0]) | (pts[:, 2] > self.z_grid[-1])
        )
        
        # Interpolate DC components
        Ex_dc = self.dc_interp['x'](pts).reshape(x_arr.shape)
        Ey_dc = self.dc_interp['y'](pts).reshape(y_arr.shape)
        Ez_dc = self.dc_interp['z'](pts).reshape(z_arr.shape)
        
        # Interpolate RF components
        Ex_rf = self.rf_interp['x'](pts).reshape(x_arr.shape)
        Ey_rf = self.rf_interp['y'](pts).reshape(y_arr.shape)
        Ez_rf = self.rf_interp['z'](pts).reshape(z_arr.shape)
        
        # Time modulation
        cos_term = np.cos(self.Omega * t)
        
        Ex = Ex_dc + cos_term * Ex_rf
        Ey = Ey_dc + cos_term * Ey_rf
        Ez = Ez_dc + cos_term * Ez_rf

        if np.any(outside_mask):
            if not self._warned_outside_grid:
                count = int(np.sum(outside_mask))
                print(f"[GridField] Warning: {count} point(s) outside grid; field undefined.")
                self._warned_outside_grid = True
            Ex_flat = Ex.reshape(-1)
            Ey_flat = Ey.reshape(-1)
            Ez_flat = Ez.reshape(-1)
            Ex_flat[outside_mask] = np.nan
            Ey_flat[outside_mask] = np.nan
            Ez_flat[outside_mask] = np.nan
            Ex = Ex_flat.reshape(x_arr.shape)
            Ey = Ey_flat.reshape(x_arr.shape)
            Ez = Ez_flat.reshape(x_arr.shape)

        return Ex, Ey, Ez
    
    def secular_gradient(self, x, y, z):
        #return gradient of |E_RF|^2 at points (x,y,z)
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        z_arr = np.asarray(z)
        
        pts = np.column_stack([
            x_arr.ravel(),
            y_arr.ravel(),
            z_arr.ravel()
        ])
        
        gx = self.secular_interp['x'](pts).reshape(x_arr.shape)
        gy = self.secular_interp['y'](pts).reshape(y_arr.shape)
        gz = self.secular_interp['z'](pts).reshape(z_arr.shape)
        
        return gx, gy, gz


class GridInterpolatedNumericField(GridInterpolatedField):
    #alias for numeric grid interpolation. becuase I had a diff class here and instead of renaming i just made this pass.
    pass


class CachedNumericField:
    """LRU cache wrapper for numeric field evaluations"""
    
    def __init__(self, field_instance, cache_size=50000):
        """
        field_instance: solvenumericfield or GridInterpolatedField instance
        cache_size: maximum number of cached results (points + voltages combinations)
        """
        self.field_instance = field_instance
        self.cache = {}
        self.cache_size = cache_size
        self.hits = 0
        self.misses = 0
        self.total_calls = 0
        
        # For time-dependent fields, we need to cache by time too
        self.time_tolerance = 1e-9  # Treat times within this as equal
        if hasattr(field_instance, "Omega"):
            freq = field_instance.Omega / (2 * np.pi)
            if freq > 1e5:
                self.time_tolerance = 1.0 / (freq * 1000)
                print(f"[Cache] Adjusted time tolerance to {self.time_tolerance:.2e} s for {freq:.0f} Hz")
        
        # Performance monitoring
        self.last_log_time = time.time()
    
    def _make_cache_key(self, points_xyz, voltages, t=None):
        #Create a cache key from points and voltages
        points_arr = np.ascontiguousarray(points_xyz)
        points_hash = hash(points_arr.tobytes())
        points_shape = points_arr.shape
        
        # Create a sorted tuple of voltages
        volt_key = tuple(sorted((k, float(v)) for k, v in voltages.items()))
        
        # Include time if provided
        if t is not None:
            # Quantize time to reduce cache misses for similar times
            t_quantized = round(t / self.time_tolerance) * self.time_tolerance
            return (points_hash, points_shape, volt_key, t_quantized)
        
        return (points_hash, points_shape, volt_key)
    
    def field(self, x, y, z, t):
        #Cached version of field evaluation
        self.total_calls += 1
        
        # Convert inputs to arrays
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        z_arr = np.asarray(z)
        
        # Stack into points array
        points = np.column_stack([
            x_arr.ravel(),
            y_arr.ravel(),
            z_arr.ravel()
        ])
        
        # Get voltages at time t
        if hasattr(self.field_instance, 'trap'):
            voltages = self.field_instance.trap.numeric_voltages(t)
        else:
            # Cant cache properly, fall back
            return self.field_instance.field(x, y, z, t)
        
        # Create cache key
        cache_key = self._make_cache_key(points, voltages, t)
        
        # Check cache
        if cache_key in self.cache:
            self.hits += 1
            Ex, Ey, Ez = self.cache[cache_key]
            return Ex.reshape(x_arr.shape), Ey.reshape(y_arr.shape), Ez.reshape(z_arr.shape)
        
        self.misses += 1
        
        # Compute field
        result = self.field_instance.field(x, y, z, t)
        
        # Manage cache size (simple FIFO eviction)
        if len(self.cache) >= self.cache_size:
            # Remove oldest 10% of entries
            to_remove = list(self.cache.keys())[:self.cache_size // 10]
            for key in to_remove:
                del self.cache[key]
        
        # Store in cache
        self.cache[cache_key] = result
        
        # Log cache performance periodically
        current_time = time.time()
        if current_time - self.last_log_time > 5.0:  # Log every 5 seconds
            hit_rate = self.hits / max(self.total_calls, 1)
            print(f"[Cache] Hit rate: {hit_rate:.1%} ({self.hits}/{self.total_calls}), size: {len(self.cache)}")
            self.last_log_time = current_time
        
        return result
    
    def secular_gradient(self, x, y, z):
        #Cached version of secular gradient
        # This is time-independent, so we can cache it more aggressively
        points = np.column_stack([
            np.asarray(x).reshape(-1),
            np.asarray(y).reshape(-1),
            np.asarray(z).reshape(-1)
        ])
        
        cache_key = self._make_cache_key(points, {"secular_grad": 1.0})
        
        if cache_key in self.cache:
            gx, gy, gz = self.cache[cache_key]
            return gx.reshape(x.shape), gy.reshape(y.shape), gz.reshape(z.shape)
        
        result = self.field_instance.secular_gradient(x, y, z)
        self.cache[cache_key] = result
        return result
    
    def clear_cache(self):
        #Clear the cache
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.total_calls = 0
    
    def get_stats(self):
        #Get cache statistics
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_calls": self.total_calls,
            "hit_rate": self.hits / max(self.total_calls, 1),
            "cache_size": len(self.cache),
            "max_size": self.cache_size
        }

class HighFrequencyCachedNumericField(CachedNumericField):
    #Optimized caching for high-frequency simulations

    def __init__(self, field_instance, cache_size=50000, frequency_hz=1e6):
        super().__init__(field_instance, cache_size)
        self.frequency_hz = float(frequency_hz)
        if self.frequency_hz > 1e5:
            self.time_tolerance = 1.0 / (self.frequency_hz * 100)
            print(f"[HighFreqCache] Time tolerance: {self.time_tolerance:.2e} s")

    def _make_cache_key(self, points_xyz, voltages, t=None):
        #Create a cache key optimized for high-frequency oscillations
        points_arr = np.ascontiguousarray(points_xyz)
        points_hash = hash(points_arr.tobytes())
        points_shape = points_arr.shape
        volt_key = tuple(sorted((k, float(v)) for k, v in voltages.items()))

        if t is not None and self.frequency_hz > 1e5:
            phase = (t * self.frequency_hz * 2 * np.pi) % (2 * np.pi)
            phase_quantized = round(phase / (np.pi / 8))
            return (points_hash, points_shape, volt_key, phase_quantized)

        return super()._make_cache_key(points_xyz, voltages, t)