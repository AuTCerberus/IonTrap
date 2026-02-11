import time
import numpy as np
from scipy.integrate import solve_ivp
from cad_numeric import NumericFieldSKFEM, OptimizedNumericFieldSKFEM
import fieldsolver
import sys
import multiprocessing
from joblib import Parallel, delayed

def _safe_flush():
    stream = sys.stdout or getattr(sys, "__stdout__", None)
    if stream is None:
        return
    try:
        stream.flush()
    except Exception:
        pass

class StopSimulation(Exception):
    pass
            
class TrapSimulator:
    def __init__(self, V_rf, V_dc, Omega, r0, trap_type='3D', custom_abc=None, particle_radius=10e-6, particle_density=2200, particle_mass=None, particle_charge=None,
                qm_ratio=None, axial_dc_kappa=None, axial_dc_z0=None, axial_dc_voltage=None, enable_gravity=True, damping_gamma=0.0, stokes_radius=None,
                pressure_torr=None, use_numeric=False, numeric_field_dir="numeric_out",
                electrode_program=None, use_optimized=True, use_secular=False, cache_size=50000,
                use_numeric_grid=False, numeric_grid_points=100, adaptive_grid_refinement=True,
                numeric_grid_smoothing=0.0, beam_params=None, progress_callback=None, progress_ranges=None,
        stop_check=None):
        print("[TrapSimulator] Initialization started...")
        _safe_flush()
        self.numeric_debug = False
        self.stop_flag = False
        self._last_stop_reason = None

        self._progress_callback = progress_callback if callable(progress_callback) else None
        self._progress_ranges = progress_ranges or {
            "init": (0.0, 5.0),
            "grid": (5.0, 35.0),
            "integrate": (35.0, 90.0),
            "plot": (90.0, 100.0),
        }
        self.stop_check = stop_check if callable(stop_check) else None
        
        #constants
        self.e_charge = 1.602176634e-19 # Elementary charge in Coulombs
        self.eps0 = 8.854e-12   # Vacuum permittivity 
        self.kB = 1.38e-23  # Boltzmann constant
        self.g = 9.81     # Gravitational acceleration

        #Trap parameters
        self.V_rf = V_rf    # Amplitude of the RF voltage
        self.V_dc = V_dc    # Amplitude of the DC voltage
        self.Omega = Omega  # Angular frequency of the RF voltage
        self.f_rf = Omega/(2*np.pi) # RF frequency in Hz
        self.is_high_frequency = self.f_rf > 1e5  # Above 100 kHz
        self.r0 = r0     # Characteristic dimension of the trap
        
        self.axial_dc_kappa = axial_dc_kappa    # Axial DC field geometric factor
        self.axial_dc_z0    = axial_dc_z0   # Axial DC field characteristic length
        self.axial_dc_U     = axial_dc_voltage  # Axial DC voltage
        
        trap_type_aliases = {
            "3D": "3D (hyperbolic)",
            "2D": "Planar (washer)",
            "linear": "Linear",
        }
        if isinstance(trap_type, str):
            key = trap_type.strip()
            trap_type = trap_type_aliases.get(key, trap_type_aliases.get(key.lower(), key))
        self.trap_type = trap_type  # Type of the trap (e.g. '3D (hyperbolic)', 'Planar (washer)', 'Linear')

        # Analytic geometry coefficients
        presets = {
            "3D (hyperbolic)": (1, 1, -2), 
            "Planar (washer)": (1, 1, -2), #yes 3D and planar are same but papers may call them differently
            "Linear": (1, -1, 0),
        }

        if custom_abc is not None:
            self.a, self.b, self.c = custom_abc
        else:
            try:
                self.a, self.b, self.c = presets[trap_type]
            except KeyError:
                raise ValueError(f"Invalid trap type: {trap_type}. Choose from {list(presets.keys())}, or provide custom_abc.")

        #Particle properties
        self.radius = particle_radius
        self.stokes_radius = float(stokes_radius) if stokes_radius is not None else particle_radius
        self.rho = particle_density
        base_mass = (4/3)*np.pi*particle_radius**3*particle_density     # Particle mass

        self.m = float(particle_mass) if particle_mass is not None else base_mass # Particle mass
        self.Q = float(particle_charge) if particle_charge is not None else 0.0 # Particle charge
        self.n_charges = self.Q / self.e_charge if self.e_charge != 0 else 0.0 # Number of elementary charges on the particle

        # Pre-compute constants for field calculation
        self.field_coeff_rf = -self.V_rf/(self.r0**2)
        self.field_coeff_dc = -self.V_dc/(self.r0**2)
        self.coulomb_coeff = self.Q**2/(4*np.pi*self.eps0)
       
        """ Charge-to-mass ratio: fallback to Q/m if qm_ratio not provided """
        if qm_ratio is not None:
            self.qm_ratio = float(qm_ratio)
            if particle_charge is None and self.m not in (None, 0):
                self.Q = self.qm_ratio * self.m
                self.n_charges = self.Q / self.e_charge if self.e_charge != 0 else 0.0
        else:
            if self.m not in (None, 0):
                self.qm_ratio = self.Q / self.m if self.m != 0 else None
        

        #Physics
        self.enable_gravity = enable_gravity    #enable gravity
        self.F_gravity = np.array([0, 0, -self.m * self.g]) if enable_gravity else np.zeros(3) # Gravity force vector

        # Pressure-dependent damping
        self.pressure_torr = pressure_torr  # Background pressure in Torr

        # Stokes drag (only valid for atmospheric pressure!)
        if self.stokes_radius > 0 and pressure_torr is None:
            # Legacy mode: atmospheric pressure Stokes drag
            self.gamma_stokes = 6*np.pi*1.81e-5*self.stokes_radius
            print("[WARNING] Stokes drag enabled with ATMOSPHERIC PRESSURE viscosity.")
            print("[WARNING] This is INCORRECT for Paul traps in ultra-high vacuum!")
            print("[WARNING] For UHV traps: uncheck 'Disable Stokes drag' or set pressure_torr explicitly.")
        elif self.stokes_radius > 0 and pressure_torr is not None:
            # Pressure-dependent damping for buffer gas cooling
            # Use hard-sphere collision model for low pressure
            # gamma = (8/3) * sqrt(pi * m_gas * k_B * T) * n * sigma
            # where n = P/(k_B*T) and sigma = pi*(r_particle + r_gas)^2

            # Assumptions: He gas at 300K
            k_B = 1.380649e-23  # J/K
            T = 300  # K
            m_He = 6.646e-27  # kg (He-4 mass)
            r_He = 140e-12  # m (He atomic radius)

            # Convert pressure to Pa: 1 Torr = 133.322 Pa
            P_Pa = pressure_torr * 133.322

            # Gas number density
            n = P_Pa / (k_B * T)

            # Collision cross-section
            sigma = np.pi * (self.stokes_radius + r_He)**2

            # Mean relative velocity between particle and gas
            v_rel = np.sqrt(8 * k_B * T / (np.pi * m_He))

            # Damping coefficient from momentum transfer
            # gamma = m * nu where nu is collision frequency
            collision_rate = n * sigma * v_rel
            self.gamma_stokes = self.m * collision_rate * (2 * m_He / (self.m + m_He))

            print(f"[INFO] Buffer gas cooling enabled:")
            print(f"       Pressure: {pressure_torr:.2e} Torr")
            print(f"       Collision rate: {collision_rate:.2e} Hz")
            print(f"       Damping rate (gamma/m): {self.gamma_stokes/self.m:.2e} s^-1")
        else:
            self.gamma_stokes = 0.0

        self.damping_gamma = float(damping_gamma) if damping_gamma is not None else 0.0 # Damping coefficient
        self.total_damping = self.damping_gamma + self.gamma_stokes # Total damping coefficient

        #numeric calls
        self.use_numeric = bool(use_numeric)
        self.numeric_field_dir = numeric_field_dir
        self.electrode_program = electrode_program or {} #no empty dict
        self.use_numeric_grid = bool(use_numeric_grid)
        self.adaptive_grid_refinement = bool(adaptive_grid_refinement)
        try:
            self.numeric_grid_points = int(numeric_grid_points)
        except (TypeError, ValueError):
            self.numeric_grid_points = 0
        if self.numeric_grid_points < 0:
            self.numeric_grid_points = 0
        try:
            self.numeric_grid_smoothing = float(numeric_grid_smoothing)
        except (TypeError, ValueError):
            self.numeric_grid_smoothing = 0.0
        if not np.isfinite(self.numeric_grid_smoothing) or self.numeric_grid_smoothing < 0:
            self.numeric_grid_smoothing = 0.0
        self.use_optimized = use_optimized  # Use optimized field class
        self.use_secular_approx = bool(use_secular)
        self.secular_interp = None
        try:
            self.cache_size = int(cache_size)
        except (TypeError, ValueError):
            self.cache_size = 50000
        if self.cache_size <= 0:
            self.cache_size = 50000
        if self.is_high_frequency and self.use_numeric:
            print(f"[TrapSimulator] High-frequency optimization enabled (f={self.f_rf:.0f} Hz)")

        if self.use_numeric: 
            print(f"[TrapSimulator] Numeric field simulation from {self.numeric_field_dir}")
            
            # use optimized field class
            try:
                if self.use_optimized:
                    print("[TrapSimulator] Using OPTIMIZED numeric field")
                    self.numeric = OptimizedNumericFieldSKFEM(
                        self.numeric_field_dir,
                        grid_resolution=50  # Adjust based on memory
                    )
                else:
                    print("[TrapSimulator] Using standard numeric field")
                    self.numeric = NumericFieldSKFEM(self.numeric_field_dir)
            except Exception as e:
                print(f"[TrapSimulator] Warning: Failed to load optimized field: {e}")
                self.numeric = NumericFieldSKFEM(self.numeric_field_dir)
            
            self.numeric_electrodes = list(self.numeric.electrodes)
            print(f"[TrapSimulator] Numeric field electrodes: {self.numeric_electrodes}")
            
            #init base numeric solver
            base_numeric_solver = fieldsolver.solvenumericfield(self)
            
            # Wrap with caching layer
            if self.is_high_frequency:
                cached_solver = fieldsolver.HighFrequencyCachedNumericField(
                    base_numeric_solver,
                    cache_size=self.cache_size,
                    frequency_hz=self.f_rf
                )
            else:
                cached_solver = fieldsolver.CachedNumericField(
                    base_numeric_solver,
                    cache_size=self.cache_size
                )
            
            #optionally wrap in grid for speedup
            if self.use_numeric_grid:
                if self.numeric_grid_points > 0:
                    print(f"[TrapSimulator] Using grid-accelerated numeric field ({self.numeric_grid_points}^3)")
                else:
                    print("[TrapSimulator] Using grid-accelerated numeric field (auto resolution)")

                # Try to reuse cached grid first
                import grid_cache
                grid_solver = grid_cache.get_cached_grid(
                    numeric_field_dir=self.numeric_field_dir,
                    grid_points=self.numeric_grid_points,
                    adaptive_refinement=self.adaptive_grid_refinement,
                    smoothing_sigma=self.numeric_grid_smoothing,
                    omega=self.Omega,
                    v_rf=self.V_rf,
                    v_dc=self.V_dc,
                    electrode_program=self.electrode_program,
                )

                if grid_solver is None:
                    # Build new grid and cache it
                    print("[TrapSimulator] Building new grid ...")
                    grid_solver = fieldsolver.GridInterpolatedNumericField(
                        self,
                        grid_points=self.numeric_grid_points,
                        adaptive_refinement=self.adaptive_grid_refinement,
                        smoothing_sigma=self.numeric_grid_smoothing,
                        progress_callback=self._emit_progress,
                        stop_check=self._should_stop,
                    )
                    grid_cache.cache_grid(
                        numeric_field_dir=self.numeric_field_dir,
                        grid_points=self.numeric_grid_points,
                        adaptive_refinement=self.adaptive_grid_refinement,
                        smoothing_sigma=self.numeric_grid_smoothing,
                        omega=self.Omega,
                        v_rf=self.V_rf,
                        v_dc=self.V_dc,
                        electrode_program=self.electrode_program,
                        grid_instance=grid_solver,
                    )

                # Add caching to grid solver too
                if self.is_high_frequency:
                    self.field_solver = fieldsolver.HighFrequencyCachedNumericField(
                        grid_solver,
                        cache_size=self.cache_size,
                        frequency_hz=self.f_rf
                    )
                else:
                    self.field_solver = fieldsolver.CachedNumericField(
                        grid_solver,
                        cache_size=self.cache_size
                    )
            else:
                print("[TrapSimulator] Using cached direct numeric field evaluation")
                self.field_solver = cached_solver
            
            if self.use_secular_approx and self.is_high_frequency:
                self._precompute_secular_gradients()
        else:
            print("[TrapSimulator] Not using numeric field simulation.")
            self.numeric = None
            self.numeric_electrodes = []
            # Initialize analytic solver
            self.field_solver = fieldsolver.solveanalyticfield(self)

        self.numeric_length_scale = None
        if self.use_numeric and self.numeric is not None:
            try:
                nodes = self.numeric.nodes
                if nodes is not None and nodes.size:
                    bounds = np.nanpercentile(nodes, [5, 95], axis=0)
                    if np.all(np.isfinite(bounds)):
                        span = bounds[1] - bounds[0]
                    else:
                        span = np.nanmax(nodes, axis=0) - np.nanmin(nodes, axis=0)
                    if np.all(np.isfinite(span)):
                        scale = float(np.max(span))
                        if scale > 0:
                            self.numeric_length_scale = scale
            except Exception:
                self.numeric_length_scale = None

        self.grid_min_spacing = None
        if self.use_numeric_grid:
            field_instance = getattr(self.field_solver, "field_instance", None)
            spacing = getattr(field_instance, "min_grid_spacing", None)
            if spacing is not None and np.isfinite(spacing) and spacing > 0:
                self.grid_min_spacing = float(spacing)

        # Guard against runaway trajectories.
        self.escape_radius = None
        if self.use_numeric and self.numeric is not None:
            try:
                nodes = self.numeric.nodes
                if nodes is not None and nodes.size:
                    max_abs = float(np.nanmax(np.abs(nodes)))
                    if np.isfinite(max_abs) and max_abs > 0:
                        self.escape_radius = 1.2 * max_abs #escape radius is 20% larger than max node extent
            except Exception:
                self.escape_radius = None

        # Fallback: use r0 for analytic mode
        if self.escape_radius is None and not self.use_numeric:
            if np.isfinite(self.r0) and self.r0 > 0:
                self.escape_radius = 20.0 * self.r0

        # Final fallback: use a reasonable absolute default
        if self.escape_radius is None:
            self.escape_radius = 0.01  # 10 mm default

        # Initialize particle beam
        from beam_field import ParticleBeam, BeamParameters
        if beam_params is not None:
            if isinstance(beam_params, BeamParameters):
                self.beam_params = beam_params
            else:
                # Assume a dict-like object
                self.beam_params = beam_params
            self.beam = ParticleBeam(self.beam_params)
            # Sync beam RF frequency with trap RF
            self.beam.set_rf_parameters(omega=self.Omega)
            if self.beam_params.enabled:
                print(f"[TrapSimulator] Particle beam enabled: {self.beam_params.beam_type.value}, {self.beam_params.energy:.1f} eV, {self.beam_params.current*1e9:.2f} nA")
        else:
            self.beam_params = BeamParameters(enabled=False)
            self.beam = ParticleBeam(self.beam_params)
            self.beam.set_rf_parameters(omega=self.Omega)

        print("[TrapSimulator] Initialization complete!")
        _safe_flush()

    def numeric_voltages(self, t):
        #Compute numeric electrode voltages for sinusoidal programs
        available = self.numeric_electrodes
        log_once = not getattr(self, "_logged_numeric_once", False)
        log_debug = self.numeric_debug
        cos_term = np.cos(self.Omega * t)
        if log_debug or log_once:
            print(f"[DEBUG] Numeric voltages at t={t:.6e}s (cos={cos_term:.3f})")
            print(f"[DEBUG] Available electrodes: {available}")
        if not available:
            if log_debug or log_once:
                print("[DEBUG] No electrodes available")
            self._logged_numeric_once = True
            return {}

        if not hasattr(self, "_program_dc") or not hasattr(self, "_program_rf"):
            try:
                self._program_dc, self._program_rf = fieldsolver._decompose_voltage_program(self)
            except Exception as exc:
                if log_debug or log_once:
                    print(f"[DEBUG] Voltage program decomposition failed: {exc}")
                self._program_dc, self._program_rf = {}, {}

        out = {}
        for name in available:
            v_dc = float(self._program_dc.get(name, 0.0))
            v_rf = float(self._program_rf.get(name, 0.0))
            out[name] = v_dc + v_rf * cos_term

        if log_debug or log_once:
            print(f"[DEBUG] Numeric voltages: {out}")
        self._logged_numeric_once = True
        return out

    def _precompute_secular_gradients(self):
        #Pre-compute secular gradients for numeric fields
        if not self.use_numeric or self.numeric is None:
            return
        if self.secular_interp is not None:
            return
        field_instance = getattr(self.field_solver, "field_instance", None)
        if isinstance(field_instance, fieldsolver.GridInterpolatedField):
            return
        
        print("[TrapSimulator] Pre-computing secular gradients...")
        
        nodes = getattr(self.numeric, "nodes", None)
        if nodes is None or nodes.size == 0:
            print("[TrapSimulator] Secular precompute skipped: numeric mesh unavailable.")
            return
        
        bounds = np.nanpercentile(nodes, [10, 90], axis=0)
        if not np.all(np.isfinite(bounds)):
            print("[TrapSimulator] Secular precompute skipped: non-finite bounds.")
            return
        
        mins = bounds[0]
        maxs = bounds[1]
        if np.any(maxs <= mins):
            print("[TrapSimulator] Secular precompute skipped: degenerate bounds.")
            return
        
        grid_points = 20
        x_grid = np.linspace(mins[0], maxs[0], grid_points)
        y_grid = np.linspace(mins[1], maxs[1], grid_points)
        z_grid = np.linspace(mins[2], maxs[2], grid_points)
        
        X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        try:
            _, rf_voltages = fieldsolver._decompose_voltage_program(self)
        except Exception:
            rf_voltages = {}
        
        if not rf_voltages:
            print("[TrapSimulator] Secular precompute skipped: no RF voltages detected.")
            return
        
        if hasattr(self.numeric, "evaluate_fast"):
            _, E_rf = self.numeric.evaluate_fast(points, rf_voltages)
        else:
            _, E_rf = self.numeric.evaluate(points, rf_voltages)
        
        E_rf = np.nan_to_num(E_rf, nan=0.0, posinf=0.0, neginf=0.0)
        E_sq = np.sum(E_rf**2, axis=1).reshape(X.shape)
        
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
        dz = z_grid[1] - z_grid[0]
        grads = np.gradient(E_sq, dx, dy, dz, edge_order=2)
        
        from scipy.interpolate import RegularGridInterpolator #
        self.secular_interp = {
            "x": RegularGridInterpolator((x_grid, y_grid, z_grid), grads[0], bounds_error=False, fill_value=0.0),
            "y": RegularGridInterpolator((x_grid, y_grid, z_grid), grads[1], bounds_error=False, fill_value=0.0),
            "z": RegularGridInterpolator((x_grid, y_grid, z_grid), grads[2], bounds_error=False, fill_value=0.0),
        }
        
        print("[TrapSimulator] Secular gradients pre-computed")

    #Set particle charge using elementary charges, Coulombs, or q/m ratio.
    def charge_particle(self, n_charges=None, charge_c=None, qm_ratio=None):
        updated = False
        if charge_c is not None:
            self.Q = float(charge_c)
            self.n_charges = self.Q / self.e_charge if self.e_charge != 0 else 0.0
            updated = True
        elif qm_ratio is not None:
            self.qm_ratio = float(qm_ratio)
            if self.m not in (None, 0):
                self.Q = self.qm_ratio * self.m
                self.n_charges = self.Q / self.e_charge if self.e_charge != 0 else 0.0
            else:
                self.Q = 0.0
                self.n_charges = 0.0
            updated = True
        elif n_charges is not None:
            self.n_charges = float(n_charges)
            self.Q = self.n_charges * self.e_charge
            updated = True

        if updated:
            if self.m not in (None, 0):
                self.qm_ratio = self.Q / self.m if self.m != 0 else self.qm_ratio
            self.coulomb_coeff = self.Q**2/(4*np.pi*self.eps0)

            if self.enable_gravity:
                self.F_gravity = np.array([0, 0, -self.m * self.g])

    def potential(self, x, y , z, t):
        #Solv potential for analytic field
        phi_t = (self.V_rf * np.cos(self.Omega*t) + self.V_dc) * (self.a*x**2 + self.b*y**2 + self.c*z**2) / (2*self.r0**2)
        return phi_t

    def field(self, x, y, z, t):
        # calc electric field at given position and time
        Ex, Ey, Ez = self.field_solver.field(x, y, z, t)
        if self.use_numeric:
            if not (np.all(np.isfinite(Ex)) and np.all(np.isfinite(Ey)) and np.all(np.isfinite(Ez))):
                raise StopSimulation("Particle left numeric mesh; field undefined.")
        return Ex, Ey, Ez

    def _should_stop(self): #stop check function for long-running operations
        if self.stop_flag:
            return True
        if self.stop_check:
            try:
                if self.stop_check():
                    self.stop_flag = True
                    return True
            except Exception as exc:
                self.stop_check = None
                print(f"[TrapSimulator] Stop check failed: {exc}")
        return False

    def _compute_timestep(self, duration, use_secular):
        # compute appropriate integration timestep.
        if use_secular:
            # Secular motion use coarser timestep, but scale to estimated secular frequency.
            if self.use_numeric:
                # For numeric mode: estimate secular frequency from field gradients
                try:
                    # Try to find trap center and estimate field gradient
                    trap_center = np.array([0.0, 0.0, 0.0])
                    if hasattr(self, 'numeric') and self.numeric is not None:
                        # Sample field at trap center
                        delta = 1e-6  # 1 micron
                        Ex_c, Ey_c, Ez_c = self.field(
                            np.array([trap_center[0]]),
                            np.array([trap_center[1]]),
                            np.array([trap_center[2]]),
                            0.0
                        )
                        # Sample along one axis to get field gradient
                        Ex_p, Ey_p, Ez_p = self.field(
                            np.array([trap_center[0] + delta]),
                            np.array([trap_center[1]]),
                            np.array([trap_center[2]]),
                            0.0
                        )
                        dE_dr = (np.sqrt(Ex_p[0]**2 + Ey_p[0]**2 + Ez_p[0]**2) -
                                np.sqrt(Ex_c[0]**2 + Ey_c[0]**2 + Ez_c[0]**2)) / delta

                        if abs(dE_dr) > 1e-6 and self.m > 0 and self.Omega > 0:
                            # Secular frequency estimate: ω_sec ≈ sqrt(Q·dE/dr / m) / Omega
                            secular_freq_est = np.sqrt(abs(self.Q * dE_dr) / self.m) / self.Omega
                            if secular_freq_est > 0 and np.isfinite(secular_freq_est):
                                steps_per_secular = 50.0
                                max_step_secular = 1.0 / (secular_freq_est * steps_per_secular)
                                return min(max_step_secular, duration / 500.0, 1e-5)
                except Exception:
                    pass  # Fall back to conservative default

                # Conservative default for numeric mode
                return min(duration / 500.0, 1e-5)

            # Analytic mode: use r0-based formula
            if self.Omega > 0 and self.m > 0 and self.r0 > 0:
                qm_ratio = abs(self.Q / self.m) if self.m != 0 else 0.0
                geom = max(abs(self.a), abs(self.b), abs(self.c))
                secular_freq_est = (qm_ratio * abs(self.V_rf) * geom) / (np.sqrt(2) * self.r0**2 * self.Omega)
                if secular_freq_est > 0:
                    steps_per_secular = 50.0
                    max_step_secular = 1.0 / (secular_freq_est * steps_per_secular)
                    return min(max_step_secular, duration / 500.0)
            return min(duration / 500.0, 1e-5)
        else:
            # Need to resolve RF oscillations
            steps_per_rf = 40.0
            if self.f_rf > 0:
                max_step_rf = 1.0 / (self.f_rf * steps_per_rf)
            else:
                max_step_rf = duration / 5000.0
            
            # For high frequencies, use fewer steps per cycle
            if self.f_rf > 1e6:  # MHz range
                steps_per_rf = 20.0
                max_step_rf = 1.0 / (self.f_rf * steps_per_rf)
                print(f"[TrapSimulator] Using reduced steps ({steps_per_rf}) for MHz frequency")

            # Also consider particle dynamics: need to resolve fastest particle motion
            if self.m > 0 and self.Q != 0:
                length_scale = None
                if self.use_numeric:
                    length_scale = self.numeric_length_scale
                else:
                    length_scale = self.r0
                if length_scale is None or not np.isfinite(length_scale) or length_scale <= 0:
                    length_scale = None
                if length_scale is not None:
                    E_max_est = abs(self.V_rf) / length_scale
                else:
                    E_max_est = None
                if E_max_est is None or not np.isfinite(E_max_est) or E_max_est <= 0:
                    E_max_est = None
                if E_max_est is not None:
                    a_max_est = abs(self.Q) * E_max_est / self.m
                    if a_max_est > 0:
                        target_dx = 0.01 * length_scale
                        grid_spacing = getattr(self, "grid_min_spacing", None)
                        if grid_spacing is not None and np.isfinite(grid_spacing) and grid_spacing > 0:
                            target_dx = min(target_dx, 0.3 * grid_spacing)
                        max_step_dyn = np.sqrt(2.0 * target_dx / a_max_est)
                        max_step_rf = min(max_step_rf, max_step_dyn)

            max_step_cap = duration / 50000.0
            max_step = min(max_step_rf, max_step_cap)
            if not np.isfinite(max_step) or max_step <= 0:
                max_step = duration / 5000.0

            min_step = 1e-12
            if max_step < min_step:
                print(f"[TrapSimulator] Warning: Computed time step {max_step:.2e} s is very small.")
                print("                  Consider using secular approximation for light particles.")
            return max_step

    def _normalize_integration_method(self, integration_method):
        if not integration_method:
            return "auto"
        label = str(integration_method).strip().lower()
        if label == "auto":
            return "auto"
        if "verlet" in label:
            return "verlet"
        if "rk4" in label or "fixed" in label:
            return "rk4"
        if "rk45" in label or "adaptive" in label:
            return "rk45"
        return "auto"

    def _parse_dt(self, dt):
        if dt is None:
            return None
        try:
            dt_val = float(dt)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(dt_val) or dt_val <= 0:
            return None
        return dt_val

    def _select_integration_method(self, integration_method, use_fixed_step=False):
        method_choice = self._normalize_integration_method(integration_method)
        fixed_method = None
        rk45_only = False
        if method_choice == "rk4":
            fixed_method = "rk4"
        elif method_choice == "verlet":
            fixed_method = "verlet"
        elif method_choice == "rk45":
            rk45_only = True
        elif use_fixed_step:
            fixed_method = "rk4"
        if fixed_method == "verlet" and self.total_damping != 0.0:
            print("[TrapSimulator] Velocity Verlet disabled with damping; using RK4.")
            fixed_method = "rk4"
        return fixed_method, rk45_only

    def _get_solver_tolerances(self, use_secular):
        if self.use_numeric:
            if use_secular:
                rtol, atol = 1e-4, 1e-6
                print("[TrapSimulator] Using relaxed tolerances for secular numeric simulation")
            elif self.is_high_frequency:
                rtol, atol = 1e-4, 1e-6
                print("[TrapSimulator] Using relaxed tolerances for high-frequency simulation")
            else:
                rtol, atol = 1e-6, 1e-8
        else:
            rtol, atol = 1e-8, 1e-10
        return rtol, atol

    def _apply_dt_cap(self, max_step, dt_val, use_secular):
        if dt_val is None:
            return max_step, None
        if dt_val > max_step:
            print(
                "[TrapSimulator] Requested dt exceeds stability limit; "
                f"capping to {max_step:.3e}s."
            )
            dt_val = max_step
        return min(max_step, dt_val), dt_val

    def _solve_ivp_with_fallback(self, eq_func, t_span, state0, t_eval, rtol, atol, max_step, prefer_stiff=False):
        methods = ["BDF", "Radau", "RK45"] if prefer_stiff else ["RK45", "Radau", "BDF"]
        last_message = None
        last_method = None
        for method in methods:
            sol = solve_ivp(
                eq_func,
                t_span,
                state0,
                method=method,
                rtol=rtol,
                atol=atol,
                t_eval=t_eval,
                max_step=max_step,
            )
            if sol.success:
                if method != methods[0]:
                    print(f"[TrapSimulator] Integration fallback: {method}")
                return sol
            last_message = sol.message
            last_method = method
        msg = last_message or "unknown solver failure"
        raise RuntimeError(f"Integration failed ({last_method}): {msg}")

    def _emit_progress(self, phase, fraction=None, label=None):
        cb = getattr(self, "_progress_callback", None)
        if not callable(cb):
            return
        try:
            if fraction is None:
                fraction = 1.0
            start, end = self._progress_ranges.get(phase, (0.0, 100.0))
            value = start + (end - start) * float(fraction)
            if not np.isfinite(value):
                return
            value = max(0.0, min(100.0, value))
            if label is None:
                label = phase
            cb(value, label)
        except Exception as exc:
            self._progress_callback = None
            print(f"[TrapSimulator] Progress callback failed: {exc}")

    def _init_progress(self, t_span):
        self._progress_enabled = True
        self._progress_t_start = float(t_span[0])
        self._progress_t_end = float(t_span[1])
        self._progress_last_wall = time.time()
        self._progress_wall_interval = 2.0
        self._emit_progress("integrate", 0.0, "Integrating...")

    def _maybe_log_progress(self, t):
        if not getattr(self, "_progress_enabled", False):
            return
        now = time.time()
        last_wall = getattr(self, "_progress_last_wall", 0.0)
        if now - last_wall < getattr(self, "_progress_wall_interval", 2.0):
            return
        t_start = getattr(self, "_progress_t_start", 0.0)
        t_end = getattr(self, "_progress_t_end", 0.0)
        duration = t_end - t_start
        pct = 0.0
        if duration > 0:
            pct = 100.0 * (float(t) - t_start) / duration
        if pct > 100.0:
            pct = 100.0
        if pct < 0.0:
            pct = 0.0
        print(f"[TrapSimulator] Progress: {pct:.1f}% (t={float(t):.3e}s)")
        self._emit_progress("integrate", pct / 100.0, f"Integrating... {pct:.0f}%")
        self._progress_last_wall = now

    def _check_state_valid(self, positions, velocities=None):
        if not np.all(np.isfinite(positions)):
            raise StopSimulation("Non-finite position encountered during integration.")
        if velocities is not None and not np.all(np.isfinite(velocities)):
            raise StopSimulation("Non-finite velocity encountered during integration.")
        escape_radius = getattr(self, "escape_radius", None)
        if escape_radius is not None and escape_radius > 0:
            if np.any(np.abs(positions) > escape_radius):
                raise StopSimulation(
                    f"Particle left simulation bounds (|x|>{escape_radius:.3e} m)."
                )

    def _integrate_fixed_step_rk4(self, eq_func, t_span, state0, dt, max_points=10000):
        t0, t1 = float(t_span[0]), float(t_span[1])
        duration = t1 - t0
        if duration <= 0:
            raise ValueError("t_span end time must be greater than start time.")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive for fixed-step integration.")

        n_steps = max(1, int(np.ceil(duration / dt)))
        dt_step = duration / n_steps

        output_stride = max(1, int(np.ceil(n_steps / max_points))) if max_points else 1
        n_outputs = n_steps // output_stride + 1
        if n_steps % output_stride != 0:
            n_outputs += 1

        print(
            "[TrapSimulator] Fixed-step RK4: "
            f"dt={dt_step:.3e}s, steps={n_steps}, output_stride={output_stride}"
        )

        t_out = np.zeros(n_outputs, dtype=float)
        states_out = np.zeros((n_outputs, state0.size), dtype=float)

        t = t0
        state = state0.astype(float, copy=True)
        out_idx = 0
        stop_reason = None

        for step in range(n_steps + 1):
            if step % output_stride == 0 or step == n_steps:
                if out_idx < n_outputs:
                    t_out[out_idx] = t
                    states_out[out_idx] = state
                    out_idx += 1
            if step == n_steps:
                break

            h = dt_step
            try:
                k1 = eq_func(t, state)
                k2 = eq_func(t + 0.5 * h, state + 0.5 * h * k1)
                k3 = eq_func(t + 0.5 * h, state + 0.5 * h * k2)
                k4 = eq_func(t + h, state + h * k3)
            except StopSimulation as exc:
                stop_reason = str(exc)
                break
            state = state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
            try:
                positions = state[::2].reshape(-1, 3)
                velocities = state[1::2].reshape(-1, 3)
                self._check_state_valid(positions, velocities)
            except StopSimulation as exc:
                stop_reason = str(exc)
                break

        if stop_reason:
            self._last_stop_reason = stop_reason

        if out_idx < n_outputs:
            t_out = t_out[:out_idx]
            states_out = states_out[:out_idx]

        return t_out, states_out

    def _integrate_fixed_step_verlet(self, eq_func, t_span, state0, dt, n_steps):
        """Velocity Verlet integrator for position-dependent forces."""
        t0, t1 = float(t_span[0]), float(t_span[1])
        if n_steps < 2:
            n_steps = 2
        t = np.linspace(t0, t1, n_steps)
        dt_step = (t1 - t0) / (n_steps - 1)

        state = state0.copy()
        states = np.zeros((n_steps, len(state0)))
        stop_reason = None
        stop_idx = n_steps

        for i in range(n_steps):
            states[i] = state
            if i == n_steps - 1:
                break

            try:
                accel = eq_func(t[i], state)[1::2]
            except StopSimulation as exc:
                stop_reason = str(exc)
                stop_idx = i + 1
                break
            state[1::2] += 0.5 * dt_step * accel
            state[::2] += dt_step * state[1::2]
            try:
                positions = state[::2].reshape(-1, 3)
                velocities = state[1::2].reshape(-1, 3)
                self._check_state_valid(positions, velocities)
            except StopSimulation as exc:
                stop_reason = str(exc)
                stop_idx = i + 1
                break
            try:
                accel_new = eq_func(t[i + 1], state)[1::2]
            except StopSimulation as exc:
                stop_reason = str(exc)
                stop_idx = i + 1
                break
            state[1::2] += 0.5 * dt_step * accel_new

        if stop_reason:
            self._last_stop_reason = stop_reason
            t = t[:stop_idx]
            states = states[:stop_idx]

        return t, states

    def _integrate_fixed_step_rk4_parallel(self, eq_func, t_span, state0, dt, n_particles):
        #Parallel RK4 for multiple particles
        t0, t1 = float(t_span[0]), float(t_span[1])
        duration = t1 - t0
        if duration <= 0:
            raise ValueError("t_span end time must be greater than start time.")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive for fixed-step integration.")

        n_steps = max(1, int(np.ceil(duration / dt)))
        dt_step = duration / n_steps
        t = np.linspace(t0, t1, n_steps + 1)
        
        # Split by particles
        def integrate_particle(particle_idx):
            state = state0[particle_idx*6:(particle_idx+1)*6].copy()
            states = np.zeros((n_steps + 1, 6))
            
            for i in range(n_steps + 1):
                states[i] = state
                if i == n_steps:
                    break
                k1 = dt_step * eq_func(t[i], state)
                k2 = dt_step * eq_func(t[i] + dt_step/2, state + k1/2)
                k3 = dt_step * eq_func(t[i] + dt_step/2, state + k2/2)
                k4 = dt_step * eq_func(t[i] + dt_step, state + k3)
                state += (k1 + 2*k2 + 2*k3 + k4) / 6
            
            return states
        
        # Parallel integration
        n_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=n_cores)(delayed(integrate_particle)(i) for i in range(n_particles))
        
        # Combine results
        states_combined = np.zeros((n_steps + 1, n_particles * 6))
        for i, result in enumerate(results):
            states_combined[:, i*6:(i+1)*6] = result
        
        return t, states_combined

    def _compute_coulomb_forces(self, positions):
        #Compute Coulomb forces between all particles.

        n_particles = positions.shape[0]
        if n_particles <= 1:
            return np.zeros_like(positions)
        
        # Use optimized 
        return fieldsolver.compute_coulomb_forces_numba(positions, self.coulomb_coeff)

    #vectorized equation of motion
    def eq_of_motion(self, t, state):
        if self._should_stop():
            raise StopSimulation("Simulation stopped by user")
        self._maybe_log_progress(t)
        
        n_particles = len(state) // 6 #particles have 6 state variables: x,y,z,vx,vy,vz
        #reshape state vector into positions and velocities
        positions = state[::2].reshape(n_particles, 3)
        velocities = state[1::2].reshape(n_particles, 3)
        self._check_state_valid(positions, velocities)

        #calc Electric field at each particle position
        Ex, Ey, Ez = self.field(positions[:,0], positions[:,1], positions[:,2], t)
        #Forces
        F_trap = self.Q * np.column_stack((Ex, Ey, Ez)) # Electric force from trap
        F_drag = -self.total_damping * self.m * velocities # Damping force: F = -gamma * m * v
        # Broadcast gravity to match shape (n_particles, 3)
        F_gravity = np.broadcast_to(self.F_gravity, (n_particles, 3))

        F_coulomb = self._compute_coulomb_forces(positions)

        # Beam force (if enabled)
        F_beam = self.beam.calculate_force_vectorized(positions, t, particle_charge=self.Q)

        # Total acceleration
        accelerations = (F_trap + F_drag + F_coulomb + F_gravity + F_beam) / self.m
        if not np.all(np.isfinite(accelerations)):
            raise StopSimulation("Non-finite acceleration encountered during integration.")
        
        derivatives = np.empty_like(state)
        derivatives[::2] = velocities.reshape(-1)
        derivatives[1::2] = accelerations.reshape(-1)
        
        return derivatives
    
    def eq_of_motion_secular(self, t, state):
        #Time-averaged equation of motion.
        
        if self._should_stop():
            raise StopSimulation("Simulation stopped by user")
        self._maybe_log_progress(t)
        
        n_particles = len(state) // 6
        positions = state[::2].reshape(n_particles, 3)
        velocities = state[1::2].reshape(n_particles, 3)
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        self._check_state_valid(positions, velocities)

        # --- 1. Pseudopotential Force (RF) ---
        # F = - grad(Phi) = - (Q^2 / 4m w^2) * grad(|E_RMS|^2)
        # where |E_RMS|^2 = |E_0|^2 / 2 for sinusoidal fields
        prefactor = -(self.Q**2) / (4 * self.m * self.Omega**2)
        
        if self.use_numeric:
            # NUMERIC (CAD): Use the gradient from the grid
            use_interp = self.secular_interp is not None
            if use_interp:
                field_instance = getattr(self.field_solver, "field_instance", None)
                if isinstance(field_instance, fieldsolver.GridInterpolatedField):
                    use_interp = False
            if use_interp:
                pts = np.column_stack((x, y, z))
                gEx = self.secular_interp["x"](pts).reshape(x.shape)
                gEy = self.secular_interp["y"](pts).reshape(x.shape)
                gEz = self.secular_interp["z"](pts).reshape(x.shape)
                fx_sec = prefactor * gEx
                fy_sec = prefactor * gEy
                fz_sec = prefactor * gEz
            elif hasattr(self.field_solver, 'secular_gradient'):
                gEx, gEy, gEz = self.field_solver.secular_gradient(x, y, z)
                fx_sec = prefactor * gEx
                fy_sec = prefactor * gEy
                fz_sec = prefactor * gEz
            else:
                # Numerical gradient fallback
                h = 1e-8
                # Evaluate at t=0 to get RF amplitude (peak value)
                Ex0, Ey0, Ez0 = self.field(x, y, z, 0.0)
                # Use RMS: |E_RMS|^2 = |E_0|^2 / 2 for time-averaged pseudopotential
                E_sq_0 = (Ex0**2 + Ey0**2 + Ez0**2) / 2.0

                Ex_px, Ey_px, Ez_px = self.field(x + h, y, z, 0.0)
                gEx = ((Ex_px**2 + Ey_px**2 + Ez_px**2) / 2.0 - E_sq_0) / h

                Ex_py, Ey_py, Ez_py = self.field(x, y + h, z, 0.0)
                gEy = ((Ex_py**2 + Ey_py**2 + Ez_py**2) / 2.0 - E_sq_0) / h

                Ex_pz, Ey_pz, Ez_pz = self.field(x, y, z + h, 0.0)
                gEz = ((Ex_pz**2 + Ey_pz**2 + Ez_pz**2) / 2.0 - E_sq_0) / h
            
                fx_sec = prefactor * gEx
                fy_sec = prefactor * gEy
                fz_sec = prefactor * gEz
            
            # DC Component (Static)
            # Evaluate field at a time where cos(wt) = 0 to get pure DC
            # (Or use a dedicated DC-only method if you prefer)
            t_dc = np.pi / (2 * self.Omega)
            Ex_dc, Ey_dc, Ez_dc = self.field(x, y, z, t_dc)
            fx_dc, fy_dc, fz_dc = self.Q * Ex_dc, self.Q * Ey_dc, self.Q * Ez_dc

        else:
            # Analytic: For quadrupole, |E|^2 = (V/r0^2)^2 * (a^2*x^2 + b^2*y^2 + c^2*z^2)
            # grad(|E|^2) = 2*(V/r0^2)^2 * [a^2*x, b^2*y, c^2*z]
            # Use RMS: divide by 2 for time-averaged force
            E_scale_sq = (self.V_rf / self.r0**2)**2 / 2.0  # RMS value
            fx_sec = prefactor * E_scale_sq * (2 * self.a**2 * x)
            fy_sec = prefactor * E_scale_sq * (2 * self.b**2 * y)
            fz_sec = prefactor * E_scale_sq * (2 * self.c**2 * z)
            
            fx_dc = self.Q * self.field_coeff_dc * self.a * x
            fy_dc = self.Q * self.field_coeff_dc * self.b * y
            fz_dc = self.Q * self.field_coeff_dc * self.c * z

        # Combine
        F_trap = np.column_stack((fx_dc + fx_sec, fy_dc + fy_sec, fz_dc + fz_sec))
        F_drag = -self.total_damping * self.m * velocities  # F = -gamma * m * v
        # Broadcast gravity to match shape (n_particles, 3)
        F_gravity = np.broadcast_to(self.F_gravity, (n_particles, 3))
        F_coulomb = self._compute_coulomb_forces(positions)

        # Beam force (if enabled)
        F_beam = self.beam.calculate_force_vectorized(positions, t, particle_charge=self.Q)

        accelerations = (F_trap + F_drag + F_coulomb + F_gravity + F_beam) / self.m
        if not np.all(np.isfinite(accelerations)):
            raise StopSimulation("Non-finite acceleration encountered during integration.")
        
        derivatives = np.empty_like(state)
        derivatives[::2] = velocities.reshape(-1)
        derivatives[1::2] = accelerations.reshape(-1)
        
        return derivatives

    def eq_of_motion_secular_optimized(self, t, state):
        return self.eq_of_motion_secular(t, state)

    def sim_single_particle(self, initial_position, initial_velocity, t_span, n_charges = None, charge_c = None, qm_ratio = None, cloud_radius = None, temp = None, use_secular=False, dt=None, use_fixed_step=False, integration_method="auto"):
        self.stop_flag = False
        self._last_stop_reason = None

        if n_charges is None and charge_c is None and qm_ratio is None:
            n_charges = 100 # Default to 100 elementary charges if none provided
        self.charge_particle(n_charges=n_charges, charge_c=charge_c, qm_ratio=qm_ratio)

        #randomized initial conditions if cloud_radius provided, initial velocity if temp provided
        if cloud_radius is not None:
            initial_position = np.random.normal(0, cloud_radius, 3)
        if temp is not None:
            v_th = np.sqrt(self.kB*temp/self.m)
            initial_velocity = np.random.normal(0, v_th, 3)

        #Initialize state vector
        initial_position = np.asarray(initial_position)
        initial_velocity = np.asarray(initial_velocity) 

        state0 = np.zeros(6)
        state0[::2] = initial_position
        state0[1::2] = initial_velocity

        #time steps and RF capping
        duration = t_span[1] - t_span[0]
        if duration <= 0:
            raise ValueError("t_span end time must be greater than start time.") 
        
        dt_val = self._parse_dt(dt)

        max_step = self._compute_timestep(duration, use_secular)
        eq_func = self.eq_of_motion_secular if use_secular else self.eq_of_motion

        max_step, dt_val = self._apply_dt_cap(max_step, dt_val, use_secular)
        
        fixed_method, rk45_only = self._select_integration_method(
            integration_method,
            use_fixed_step=use_fixed_step,
        )

        est_steps = int(np.ceil(duration / max_step)) #estimated number of integration steps
        print(
            "[TrapSimulator] Time stepping: "
            f"duration={duration:.3e}s, max_step={max_step:.3e}s, "
            f"est_steps={est_steps}, use_secular={use_secular}"
        )

        if fixed_method:
            step_dt = dt_val if dt_val is not None else max_step
            if not np.isfinite(step_dt) or step_dt <= 0:
                step_dt = max_step
            self._init_progress(t_span)
            if fixed_method == "rk4":
                t_eval, states = self._integrate_fixed_step_rk4(
                    eq_func,
                    t_span,
                    state0,
                    step_dt,
                    max_points=10000,
                )
            else:
                n_steps = max(2, int(np.ceil(duration / step_dt)) + 1)
                step_dt = duration / max(1, n_steps - 1)
                t_eval, states = self._integrate_fixed_step_verlet(
                    eq_func,
                    t_span,
                    state0,
                    step_dt,
                    n_steps,
                )
            self._progress_enabled = False
        else:
            n_points = int(min(10000, max(1000, est_steps+1))) #limit min/max 1k/10k output points
            t_eval = np.linspace(t_span[0], t_span[1], n_points) #time grid

            rtol, atol = self._get_solver_tolerances(use_secular)

            self._init_progress(t_span)
            try:
                if rk45_only:
                    sol = solve_ivp(
                        eq_func,
                        t_span,
                        state0,
                        method="RK45",
                        rtol=rtol,
                        atol=atol,
                        t_eval=t_eval,
                        max_step=max_step,
                    )
                    if not sol.success:
                        raise RuntimeError(f"Integration failed (RK45): {sol.message}")
                else:
                    sol = self._solve_ivp_with_fallback(
                        eq_func,
                        t_span,
                        state0,
                        t_eval,
                        rtol,
                        atol,
                        max_step,
                        prefer_stiff=use_secular and self.use_numeric,
                    )
                states = sol.y.T
            except StopSimulation as exc:
                self._last_stop_reason = str(exc)
                step_dt = dt_val if dt_val is not None else max_step
                if not np.isfinite(step_dt) or step_dt <= 0:
                    step_dt = max_step
                t_eval, states = self._integrate_fixed_step_rk4(
                    eq_func,
                    t_span,
                    state0,
                    step_dt,
                    max_points=10000,
                )
            finally:
                self._progress_enabled = False

        positions = states[:, ::2]
        velocities = states[:, 1::2]
        stop_reason = self._last_stop_reason or ""
        escape_radius = getattr(self, "escape_radius", None)
        if escape_radius is not None and escape_radius > 0 and "bounds" in stop_reason:
            in_bounds = np.all(np.abs(positions) <= escape_radius, axis=1)
            if np.any(in_bounds):
                positions = positions[in_bounds]
                velocities = velocities[in_bounds]
                t_eval = t_eval[in_bounds]
            else:
                positions = positions[:1]
                velocities = velocities[:1]
                t_eval = t_eval[:1]
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            if self._last_stop_reason:
                finite_mask = np.all(np.isfinite(positions), axis=1) & np.all(np.isfinite(velocities), axis=1)
                if np.any(finite_mask):
                    positions = positions[finite_mask]
                    velocities = velocities[finite_mask]
                    t_eval = t_eval[finite_mask]
                else:
                    raise RuntimeError("Simulation produced non-finite results.")
            else:
                raise RuntimeError("Simulation produced non-finite results.")
        
        return t_eval, positions, velocities
    
    def sim_particle_cloud(self, n_particles, cloud_radius, t_span, n_charges=None, temp=300, charge_c=None, qm_ratio=None, use_secular=False, dt=None, integration_method="auto", use_parallel=True):
        self.stop_flag = False
        self._last_stop_reason = None

        if n_charges is None and charge_c is None and qm_ratio is None:
            n_charges = 100
        self.charge_particle(n_charges=n_charges, charge_c=charge_c, qm_ratio=qm_ratio)
        
        # Initial conditions
        initial_positions = np.random.normal(0, cloud_radius, (n_particles, 3))
        v_th = np.sqrt(self.kB*temp/self.m)
        initial_velocities = np.random.normal(0, v_th, (n_particles, 3))
        
        # Create state vector
        state0 = np.zeros(n_particles * 6)
        for i in range(n_particles):
            state0[i*6:(i+1)*6:2] = initial_positions[i]  # positions at 0,2,4
            state0[i*6+1:(i+1)*6:2] = initial_velocities[i]  # velocities at 1,3,5

        duration = t_span[1] - t_span[0]
        if duration <= 0:
            raise ValueError("t_span end time must be greater than start time.")
        
        dt_val = self._parse_dt(dt)
        max_step = self._compute_timestep(duration, use_secular)
        eq_func = self.eq_of_motion_secular if use_secular else self.eq_of_motion

        max_step, dt_val = self._apply_dt_cap(max_step, dt_val, use_secular)
        
        fixed_method, rk45_only = self._select_integration_method(integration_method)

        est_steps = int(np.ceil(duration / max_step))
        print(
            "[TrapSimulator] Time stepping: "
            f"duration={duration:.3e}s, max_step={max_step:.3e}s, "
            f"est_steps={est_steps}, use_secular={use_secular}"
        )
        try:
            if fixed_method:
                step_dt = dt_val if dt_val is not None else max_step
                if not np.isfinite(step_dt) or step_dt <= 0:
                    step_dt = max_step
                self._init_progress(t_span)
                if fixed_method == "rk4":
                    parallel_ok = (
                        use_parallel
                        and n_particles > 1
                        and self.coulomb_coeff == 0.0
                    )
                    if use_parallel and not parallel_ok and n_particles > 1:
                        print("[TrapSimulator] Parallel processing disabled for interacting particles.")
                    if parallel_ok:
                        try:
                            t_eval, states = self._integrate_fixed_step_rk4_parallel(
                                eq_func,
                                t_span,
                                state0,
                                step_dt,
                                n_particles,
                            )
                        except StopSimulation as exc:
                            self._last_stop_reason = str(exc)
                            t_eval, states = self._integrate_fixed_step_rk4(
                                eq_func,
                                t_span,
                                state0,
                                step_dt,
                                max_points=8000,
                            )
                    else:
                        t_eval, states = self._integrate_fixed_step_rk4(
                            eq_func,
                            t_span,
                            state0,
                            step_dt,
                            max_points=8000,
                        )
                else:
                    n_steps = max(2, int(np.ceil(duration / step_dt)) + 1)
                    step_dt = duration / max(1, n_steps - 1)
                    t_eval, states = self._integrate_fixed_step_verlet(
                        eq_func,
                        t_span,
                        state0,
                        step_dt,
                        n_steps,
                    )
                self._progress_enabled = False
            else:
                n_points = int(min(8000, max(500, est_steps + 1)))
                t_eval = np.linspace(t_span[0], t_span[1], n_points)
                rtol, atol = self._get_solver_tolerances(use_secular)

                self._init_progress(t_span)
                try:
                    if rk45_only:
                        sol = solve_ivp(
                            eq_func,
                            t_span,
                            state0,
                            method="RK45",
                            rtol=rtol,
                            atol=atol,
                            t_eval=t_eval,
                            max_step=max_step,
                        )
                        if not sol.success:
                            raise RuntimeError(f"Integration failed (RK45): {sol.message}")
                    else:
                        sol = self._solve_ivp_with_fallback(
                            eq_func,
                            t_span,
                            state0,
                            t_eval,
                            rtol,
                            atol,
                            max_step,
                            prefer_stiff=use_secular and self.use_numeric,
                        )
                    states = sol.y.T
                except StopSimulation as exc:
                    self._last_stop_reason = str(exc)
                    step_dt = dt_val if dt_val is not None else max_step
                    if not np.isfinite(step_dt) or step_dt <= 0:
                        step_dt = max_step
                    t_eval, states = self._integrate_fixed_step_rk4(
                        eq_func,
                        t_span,
                        state0,
                        step_dt,
                        max_points=8000,
                    )
                finally:
                    self._progress_enabled = False

            positions = states[:, ::2].reshape(len(t_eval), n_particles, 3)
            stop_reason = self._last_stop_reason or ""
            escape_radius = getattr(self, "escape_radius", None)
            if escape_radius is not None and escape_radius > 0 and "bounds" in stop_reason:
                in_bounds = np.all(np.abs(positions) <= escape_radius, axis=(1, 2))
                if np.any(in_bounds):
                    positions = positions[in_bounds]
                    t_eval = t_eval[in_bounds]
                else:
                    positions = positions[:1]
                    t_eval = t_eval[:1]
            if not np.all(np.isfinite(positions)):
                if self._last_stop_reason:
                    finite_mask = np.all(np.isfinite(positions), axis=(1, 2))
                    if np.any(finite_mask):
                        positions = positions[finite_mask]
                        t_eval = t_eval[finite_mask]
                    else:
                        raise RuntimeError("Simulation produced non-finite results.")
                else:
                    raise RuntimeError("Simulation produced non-finite results.")

            return t_eval, positions

        except Exception as e:
            print(f"Error during simulation: {e}")
            raise


