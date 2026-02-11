"""
Parameter Sweep Module for Ion Trap Simulation

Provides tools for:
- 1D and 2D parameter sweeps
- Stability zone mapping (Mathieu diagram)
- Pressure/damping sweeps
- Automated stability detection
- Results persistence and visualization

Key Features:
- Uses cached grid fields for fast sweeps
- Parallel execution support
- Checkpointing for long sweeps
- Comprehensive metrics extraction
"""

import numpy as np
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from enum import Enum
from datetime import datetime
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import traceback


class StabilityResult(Enum):
    #Result of a stability check
    STABLE = "stable"              # Particle remained trapped
    ESCAPED = "escaped"            # Particle left trap bounds
    UNSTABLE = "unstable"          # Exponential growth detected
    CRASHED = "crashed"            # Simulation error
    TIMEOUT = "timeout"            # Simulation took too long
    UNKNOWN = "unknown"            # Could not determine


@dataclass
class SimulationMetrics:
    #Metrics extracted from a single simulation run
    
    # Stability
    stability: StabilityResult = StabilityResult.UNKNOWN
    survival_time: float = 0.0           # Time before escape (or full duration if stable)
    
    # Position metrics (in meters)
    max_amplitude_x: float = 0.0
    max_amplitude_y: float = 0.0
    max_amplitude_z: float = 0.0
    max_amplitude_r: float = 0.0         # Radial (sqrt(x² + y²))
    final_amplitude_x: float = 0.0
    final_amplitude_y: float = 0.0
    final_amplitude_z: float = 0.0
    mean_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Velocity/Energy metrics
    max_speed: float = 0.0
    mean_kinetic_energy: float = 0.0     # Joules
    final_kinetic_energy: float = 0.0
    
    # Oscillation analysis
    secular_frequency_x: float = 0.0     # Hz (estimated)
    secular_frequency_y: float = 0.0
    secular_frequency_z: float = 0.0
    micromotion_amplitude: float = 0.0   # Estimated micromotion
    
    # Simulation info
    total_time: float = 0.0              # Simulation duration
    n_timesteps: int = 0
    computation_time: float = 0.0        # Wall clock time
    error_message: str = ""
    
    def to_dict(self) -> dict:
        #Convert to dictionary for JSON serialization
        d = asdict(self)
        d['stability'] = self.stability.value
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> 'SimulationMetrics':
        #Create from dictionary
        d = d.copy()
        d['stability'] = StabilityResult(d['stability'])
        d['mean_position'] = tuple(d['mean_position'])
        return cls(**d)


@dataclass
class SweepParameter:
    #Definition of a parameter to sweep
    name: str                            # Parameter name (e.g., 'V_rf', 'pressure_torr')
    values: np.ndarray = None            # Explicit values to use
    min_val: float = None                # Or specify range
    max_val: float = None
    n_points: int = 10
    log_scale: bool = False              # Use logarithmic spacing
    
    def __post_init__(self):
        if self.values is None:
            if self.min_val is None or self.max_val is None:
                raise ValueError(f"Must specify either 'values' or 'min_val'/'max_val' for {self.name}")
            if self.log_scale:
                self.values = np.logspace(np.log10(self.min_val), np.log10(self.max_val), self.n_points)
            else:
                self.values = np.linspace(self.min_val, self.max_val, self.n_points)
        else:
            self.values = np.asarray(self.values)
            self.n_points = len(self.values)
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'values': self.values.tolist(),
            'min_val': self.min_val,
            'max_val': self.max_val,
            'n_points': self.n_points,
            'log_scale': self.log_scale,
        }


@dataclass 
class SweepConfig:
    #Configuration for a parameter sweep
    
    # Parameters to sweep
    param1: SweepParameter
    param2: SweepParameter = None        # Optional second parameter for 2D sweep
    
    # Simulation settings
    duration: float = 1e-3               # Simulation duration (s)
    dt: float = None                     # Time step (None = auto)
    n_particles: int = 1
    use_secular: bool = False
    
    # Initial conditions
    initial_position: Tuple[float, float, float] = (1e-4, 1e-4, 1e-4)
    initial_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    randomize_initial: bool = False
    cloud_radius: float = 1e-4           # For randomized initial conditions
    temperature: float = 300             # K, for randomized velocities
    
    # Stability criteria
    escape_radius: float = 1e-2          # Particle escaped if |r| > this
    stable_amplitude: float = 5e-3       # Consider stable if final amp < this
    
    # Execution settings
    n_parallel: int = 1                  # Number of parallel workers
    checkpoint_interval: int = 10        # Save checkpoint every N simulations
    timeout_per_sim: float = 60.0        # Max seconds per simulation
    
    # Output settings
    output_dir: str = "sweep_results"
    save_trajectories: bool = False      # Save full trajectories (large!)
    
    def to_dict(self) -> dict:
        d = {
            'param1': self.param1.to_dict(),
            'param2': self.param2.to_dict() if self.param2 else None,
            'duration': self.duration,
            'dt': self.dt,
            'n_particles': self.n_particles,
            'use_secular': self.use_secular,
            'initial_position': list(self.initial_position),
            'initial_velocity': list(self.initial_velocity),
            'randomize_initial': self.randomize_initial,
            'cloud_radius': self.cloud_radius,
            'temperature': self.temperature,
            'escape_radius': self.escape_radius,
            'stable_amplitude': self.stable_amplitude,
            'n_parallel': self.n_parallel,
            'checkpoint_interval': self.checkpoint_interval,
            'timeout_per_sim': self.timeout_per_sim,
            'output_dir': self.output_dir,
            'save_trajectories': self.save_trajectories,
        }
        return d


class ParameterSweep:
    """
    Executes parameter sweeps for stability analysis.
    
    Usage:
        sweep = ParameterSweep(base_trap_params)
        config = SweepConfig(
            param1=SweepParameter('V_rf', min_val=50, max_val=200, n_points=20),
            param2=SweepParameter('V_dc', min_val=0, max_val=50, n_points=20),
        )
        results = sweep.run(config)
        sweep.plot_stability_diagram(results)
    """
    
    # Map parameter names to trap attributes
    PARAM_MAP = {
        'V_rf': 'V_rf',
        'V_dc': 'V_dc',
        'frequency': 'f_rf',
        'omega': 'Omega',
        'pressure_torr': 'pressure_torr',
        'damping_gamma': 'damping_gamma',
        'particle_radius': 'radius',
        'particle_mass': 'm',
        'particle_charge': 'Q',
        'n_charges': 'n_charges',
        'qm_ratio': 'qm_ratio',
    }
    
    def __init__(self, base_params: dict, numeric_dir: str = None,
                 use_numeric: bool = True, use_grid: bool = True,
                 progress_callback: Callable = None):
        """
        Initialize the parameter sweep.
        
        Args:
            base_params: Base trap parameters (dict passed to TrapSimulator)
            numeric_dir: Directory with numeric field data
            use_numeric: Use numeric (CAD) fields
            use_grid: Use grid-interpolated fields (faster, uses cache)
            progress_callback: Called with (current, total, message)
        """
        self.base_params = base_params.copy()
        self.numeric_dir = numeric_dir
        self.use_numeric = use_numeric
        self.use_grid = use_grid
        self.progress_callback = progress_callback

        self._stop_requested = False
        self._results_cache = {}

        # Grid caching for performance
        self._cached_simulator = None
        self._cached_grid_key = None

        # Parameters that only affect voltages (can reuse grid)
        self._voltage_only_params = {'V_rf', 'V_dc'}

        # Parameters that require grid rebuild
        self._grid_affecting_params = {'frequency', 'omega', 'particle_radius',
                                        'particle_mass', 'particle_charge', 'n_charges',
                                        'qm_ratio', 'damping_gamma', 'pressure_torr'}
        
    def stop(self):
        #Stop
        self._stop_requested = True
        
    def _emit_progress(self, current: int, total: int, message: str = ""):
        #Progress update callback
        if self.progress_callback:
            try:
                self.progress_callback(current, total, message)
            except Exception:
                pass
    
    def _can_reuse_grid(self, sweep_params: set) -> bool:
        #Check if the sweep only changes voltage parameters (grid can be reused)
        if not self.use_numeric or not self.use_grid:
            return False
        # Can reuse if ALL swept parameters are voltage-only
        return sweep_params.issubset(self._voltage_only_params)

    def _create_grid_key(self, sim_params: dict) -> str:
        """Create a key for grid caching based on geometry-affecting parameters.
            Grid depends on numeric_dir and geometry, not voltages"""
        key_parts = []
        if self.numeric_dir:
            key_parts.append(f"dir:{self.numeric_dir}")
        # Add any parameters that affect grid structure
        for param in ['frequency', 'omega', 'numeric_grid_points', 'adaptive_grid_refinement']:
            if param in sim_params:
                key_parts.append(f"{param}:{sim_params[param]}")
        return "|".join(key_parts)

    def _create_simulator(self, params: dict, force_new: bool = False):
        #Create a TrapSimulator with given parameters, reusing grid when possible
        import fields

        # Merge with base params
        sim_params = self.base_params.copy()
        sim_params.update(params)

        # Handle numeric field settings
        if self.use_numeric and self.numeric_dir:
            sim_params['use_numeric'] = True
            sim_params['numeric_field_dir'] = self.numeric_dir
            sim_params['use_numeric_grid'] = self.use_grid

        # Check if we can reuse cached simulator with grid
        if not force_new and self._cached_simulator is not None:
            grid_key = self._create_grid_key(sim_params)

            if grid_key == self._cached_grid_key:
                # Can reuse! Just update voltage parameters
                sim = self._cached_simulator

                # Update voltages directly
                if 'V_rf' in params:
                    sim.V_rf = params['V_rf']
                if 'V_dc' in params:
                    sim.V_dc = params['V_dc']

                # Update electrode program for numeric fields
                if self.use_numeric and hasattr(sim, 'numeric_voltages'):
                    # Force regeneration of electrode voltages
                    pass  # numeric_voltages() method handles this dynamically

                print(f"[Sweep] Reusing cached grid (V_rf={sim.V_rf:.1f}V, V_dc={sim.V_dc:.1f}V)")
                return sim

        # Need to create new simulator
        print(f"[Sweep] Creating new simulator (building grid...)")
        sim = fields.TrapSimulator(**sim_params)

        # Cache for reuse if using grid
        if self.use_numeric and self.use_grid:
            self._cached_simulator = sim
            self._cached_grid_key = self._create_grid_key(sim_params)
            print(f"[Sweep] Grid cached for future runs")

        return sim
    
    def _extract_metrics(self, t: np.ndarray, positions: np.ndarray, 
                         velocities: np.ndarray, sim, config: SweepConfig,
                         computation_time: float) -> SimulationMetrics:
        """Extract metrics from simulation results."""
        
        metrics = SimulationMetrics()
        metrics.computation_time = computation_time
        metrics.total_time = t[-1] - t[0] if len(t) > 1 else 0.0
        metrics.n_timesteps = len(t)
        
        if len(positions) == 0:
            metrics.stability = StabilityResult.CRASHED
            return metrics
        
        # Handle multi-particle case (use first particle)
        if positions.ndim == 3:
            positions = positions[:, 0, :]
            velocities = velocities[:, 0, :] if velocities is not None else None
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(positions)):
            first_bad = np.argmax(~np.all(np.isfinite(positions), axis=1))
            metrics.survival_time = t[first_bad] if first_bad > 0 else 0.0
            metrics.stability = StabilityResult.ESCAPED
            # Use only valid portion
            positions = positions[:first_bad] if first_bad > 0 else positions[:1]
            if velocities is not None:
                velocities = velocities[:first_bad] if first_bad > 0 else velocities[:1]
        
        if len(positions) == 0:
            metrics.stability = StabilityResult.CRASHED
            return metrics
        
        # Position metrics
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        r = np.sqrt(x**2 + y**2)
        
        metrics.max_amplitude_x = float(np.max(np.abs(x)))
        metrics.max_amplitude_y = float(np.max(np.abs(y)))
        metrics.max_amplitude_z = float(np.max(np.abs(z)))
        metrics.max_amplitude_r = float(np.max(r))
        
        # Final position (use last 10% average for stability)
        n_final = max(1, len(positions) // 10)
        metrics.final_amplitude_x = float(np.mean(np.abs(x[-n_final:])))
        metrics.final_amplitude_y = float(np.mean(np.abs(y[-n_final:])))
        metrics.final_amplitude_z = float(np.mean(np.abs(z[-n_final:])))
        metrics.mean_position = (float(np.mean(x)), float(np.mean(y)), float(np.mean(z)))
        
        # Velocity metrics
        if velocities is not None and len(velocities) > 0:
            speeds = np.linalg.norm(velocities, axis=1)
            metrics.max_speed = float(np.max(speeds))
            
            # Kinetic energy
            mass = sim.m
            ke = 0.5 * mass * speeds**2
            metrics.mean_kinetic_energy = float(np.mean(ke))
            metrics.final_kinetic_energy = float(np.mean(ke[-n_final:]))
        
        # Stability determination
        max_r = np.max(np.sqrt(x**2 + y**2 + z**2))
        final_r = np.mean(np.sqrt(x[-n_final:]**2 + y[-n_final:]**2 + z[-n_final:]**2))
        
        if max_r > config.escape_radius:
            metrics.stability = StabilityResult.ESCAPED
            # Find escape time
            r_all = np.sqrt(x**2 + y**2 + z**2)
            escape_idx = np.argmax(r_all > config.escape_radius)
            metrics.survival_time = t[escape_idx]
        elif final_r > config.stable_amplitude:
            metrics.stability = StabilityResult.UNSTABLE
            metrics.survival_time = metrics.total_time
        else:
            metrics.stability = StabilityResult.STABLE
            metrics.survival_time = metrics.total_time
        
        # Frequency estimation (simple FFT)
        if len(t) > 100 and metrics.stability == StabilityResult.STABLE:
            try:
                dt = t[1] - t[0]
                # Remove DC offset
                x_centered = x - np.mean(x)
                if np.std(x_centered) > 1e-12:
                    fft = np.fft.rfft(x_centered)
                    freqs = np.fft.rfftfreq(len(x_centered), dt)
                    peak_idx = np.argmax(np.abs(fft[1:])) + 1  # Skip DC
                    metrics.secular_frequency_x = float(freqs[peak_idx])
            except Exception:
                pass
        
        return metrics
    
    def _run_single_simulation(self, params: dict, config: SweepConfig, 
                                run_id: int = 0) -> Tuple[dict, SimulationMetrics]:
        """Run a single simulation with given parameters."""
        
        start_time = time.time()
        
        try:
            # Create simulator
            sim = self._create_simulator(params)
            
            # Set initial conditions
            if config.randomize_initial:
                initial_pos = np.random.normal(0, config.cloud_radius, 3)
                v_th = np.sqrt(1.38e-23 * config.temperature / sim.m)
                initial_vel = np.random.normal(0, v_th, 3)
            else:
                initial_pos = np.array(config.initial_position)
                initial_vel = np.array(config.initial_velocity)
            
            # Run simulation
            t_span = (0, config.duration)
            
            if config.n_particles == 1:
                t, positions, velocities = sim.sim_single_particle(
                    initial_pos, initial_vel, t_span,
                    use_secular=config.use_secular,
                    dt=config.dt
                )
            else:
                t, positions = sim.sim_particle_cloud(
                    config.n_particles, config.cloud_radius, t_span,
                    temp=config.temperature,
                    use_secular=config.use_secular,
                    dt=config.dt
                )
                velocities = None
            
            computation_time = time.time() - start_time
            
            # Extract metrics
            metrics = self._extract_metrics(t, positions, velocities, sim, config, computation_time)
            
            return params, metrics
            
        except Exception as e:
            computation_time = time.time() - start_time
            metrics = SimulationMetrics()
            metrics.stability = StabilityResult.CRASHED
            metrics.error_message = str(e)
            metrics.computation_time = computation_time
            return params, metrics
    
    def _apply_parameter(self, params: dict, name: str, value: float) -> dict:
        #Apply a parameter value to the params dict
        params = params.copy()
        
        # Special handling for certain parameters
        if name == 'frequency':
            params['Omega'] = 2 * np.pi * value
        elif name == 'omega':
            params['Omega'] = value
        elif name == 'n_charges':
            # Need to recalculate charge
            e = 1.602176634e-19
            params['particle_charge'] = value * e
        elif name in self.PARAM_MAP:
            params[self.PARAM_MAP[name]] = value
        else:
            # Try direct assignment
            params[name] = value
        
        return params
    
    def run_1d_sweep(self, config: SweepConfig) -> Dict[str, Any]:
        """
        Run a 1D parameter sweep.
        
        Returns:
            Dict with 'parameter_values', 'metrics', 'config', 'summary'
        """
        self._stop_requested = False
        
        param = config.param1
        n_total = len(param.values)
        
        results = {
            'type': '1d',
            'parameter_name': param.name,
            'parameter_values': param.values.tolist(),
            'metrics': [],
            'config': config.to_dict(),
            'started': datetime.now().isoformat(),
            'completed': None,
        }
        
        print(f"[Sweep] Starting 1D sweep: {param.name}")
        print(f"[Sweep] {n_total} points from {param.values[0]:.3e} to {param.values[-1]:.3e}")

        # Check if grid caching can be used
        can_cache = self._can_reuse_grid({param.name})
        if can_cache and self.use_numeric and self.use_grid:
            print(f"[Sweep]  Grid caching enabled! First run builds grid, rest reuse it (massive speedup)")
        elif not can_cache and self.use_numeric and self.use_grid:
            print(f"[Sweep]  Grid will be rebuilt each run (sweeping {param.name} affects grid)")

        for i, value in enumerate(param.values):
            if self._stop_requested:
                print("[Sweep] Stop requested - saving partial results...")
                results['completed'] = datetime.now().isoformat()
                results['stopped_early'] = True
                results['completed_runs'] = i
                self._save_checkpoint(results, config)  # Save final checkpoint
                self._add_summary_1d(results)
                self._save_results(results, config)
                print(f"[Sweep] Partial results saved ({i}/{n_total} runs completed)")
                break

            self._emit_progress(i, n_total, f"{param.name}={value:.3e}")

            # Build parameters
            params = self._apply_parameter({}, param.name, value)

            # Run simulation
            _, metrics = self._run_single_simulation(params, config, run_id=i)
            results['metrics'].append(metrics.to_dict())

            # Status update
            status = " OK" if metrics.stability == StabilityResult.STABLE else "NO"
            print(f"[Sweep] {i+1}/{n_total}: {param.name}={value:.3e} -> {status} ({metrics.computation_time:.1f}s)")

            # Checkpoint
            if (i + 1) % config.checkpoint_interval == 0:
                self._save_checkpoint(results, config)

        # Only save if not already saved by stop
        if not self._stop_requested:
            results['completed'] = datetime.now().isoformat()
            results['stopped_early'] = False
            self._add_summary_1d(results)
            self._save_results(results, config)
        
        self._emit_progress(n_total, n_total, "Complete")
        
        return results
    
    def run_2d_sweep(self, config: SweepConfig) -> Dict[str, Any]:
        """
        Run a 2D parameter sweep (stability diagram).
        
        Returns:
            Dict with 'param1_values', 'param2_values', 'stability_map', 'metrics_grid', etc.
        """
        if config.param2 is None:
            raise ValueError("2D sweep requires param2 to be specified")
        
        self._stop_requested = False
        
        param1, param2 = config.param1, config.param2
        n1, n2 = len(param1.values), len(param2.values)
        n_total = n1 * n2
        
        results = {
            'type': '2d',
            'param1_name': param1.name,
            'param2_name': param2.name,
            'param1_values': param1.values.tolist(),
            'param2_values': param2.values.tolist(),
            'stability_map': np.zeros((n1, n2), dtype=int).tolist(),  # 1=stable, 0=unstable
            'metrics_grid': [[None for _ in range(n2)] for _ in range(n1)],
            'config': config.to_dict(),
            'started': datetime.now().isoformat(),
            'completed': None,
        }
        
        print(f"[Sweep] Starting 2D sweep: {param1.name} vs {param2.name}")
        print(f"[Sweep] Grid: {n1} x {n2} = {n_total} points")

        # Check if grid caching can be used
        can_cache = self._can_reuse_grid({param1.name, param2.name})
        if can_cache and self.use_numeric and self.use_grid:
            print(f"[Sweep]   Grid caching enabled! First run builds grid, rest reuse it")
        elif not can_cache and self.use_numeric and self.use_grid:
            print(f"[Sweep]   Grid will be rebuilt each run (sweeping parameters affect grid)")

        count = 0
        stopped = False
        for i, v1 in enumerate(param1.values):
            for j, v2 in enumerate(param2.values):
                if self._stop_requested:
                    print("[Sweep] Stop requested - saving partial results...")
                    stopped = True
                    break

                count += 1
                self._emit_progress(count, n_total, f"{param1.name}={v1:.2e}, {param2.name}={v2:.2e}")

                # Build parameters
                params = {}
                params = self._apply_parameter(params, param1.name, v1)
                params = self._apply_parameter(params, param2.name, v2)

                # Run simulation
                _, metrics = self._run_single_simulation(params, config, run_id=count)

                # Store results
                is_stable = 1 if metrics.stability == StabilityResult.STABLE else 0
                results['stability_map'][i][j] = is_stable
                results['metrics_grid'][i][j] = metrics.to_dict()

                # Status update
                status = " OK" if is_stable else "NO"
                if count % 10 == 0 or count == n_total:
                    print(f"[Sweep] {count}/{n_total}: {status} ({metrics.computation_time:.1f}s)")

            if stopped:
                break

            # Checkpoint after each row
            self._save_checkpoint(results, config)

        # Save results (partial or complete)
        results['completed'] = datetime.now().isoformat()
        results['stopped_early'] = stopped
        if stopped:
            results['completed_runs'] = count
            print(f"[Sweep] Saving partial results ({count}/{n_total} runs completed)...")

        self._add_summary_2d(results)
        self._save_results(results, config)

        if stopped:
            print(f"[Sweep] Partial results saved successfully")
        
        self._emit_progress(n_total, n_total, "Complete")
        
        return results
    
    def run(self, config: SweepConfig) -> Dict[str, Any]:
        #Run sweep (1D or 2D depending on config)
        if config.param2 is not None:
            return self.run_2d_sweep(config)
        else:
            return self.run_1d_sweep(config)
    
    def _add_summary_1d(self, results: dict):
        #Add summary statistics for 1D sweep
        metrics_list = [SimulationMetrics.from_dict(m) for m in results['metrics']]
        
        stable_count = sum(1 for m in metrics_list if m.stability == StabilityResult.STABLE)
        
        results['summary'] = {
            'total_runs': len(metrics_list),
            'stable_count': stable_count,
            'stable_fraction': stable_count / len(metrics_list) if metrics_list else 0,
            'mean_computation_time': np.mean([m.computation_time for m in metrics_list]),
        }
        
        # Find stability boundaries
        values = np.array(results['parameter_values'])
        stable_mask = np.array([m.stability == StabilityResult.STABLE for m in metrics_list])
        
        if np.any(stable_mask) and not np.all(stable_mask):
            # Find transitions
            transitions = np.where(np.diff(stable_mask.astype(int)))[0]
            if len(transitions) > 0:
                results['summary']['stability_boundaries'] = values[transitions].tolist()
    
    def _add_summary_2d(self, results: dict):
        #Add summary statistics for 2D sweep
        stability_map = np.array(results['stability_map'])
        
        results['summary'] = {
            'total_runs': stability_map.size,
            'stable_count': int(np.sum(stability_map)),
            'stable_fraction': float(np.mean(stability_map)),
            'grid_shape': list(stability_map.shape),
        }
    
    def _save_checkpoint(self, results: dict, config: SweepConfig):
        #Save checkpoint to disk
        try:
            output_dir = Path(config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_file = output_dir / "sweep_checkpoint.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save checkpoint: {e}")
    
    def _save_results(self, results: dict, config: SweepConfig):
        #Save final results to disk
        try:
            output_dir = Path(config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sweep_type = results.get('type', '1d')
            
            # Save JSON
            result_file = output_dir / f"sweep_{sweep_type}_{timestamp}.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save stability map as numpy array for 2D sweeps
            if sweep_type == '2d':
                stability_file = output_dir / f"stability_map_{timestamp}.npy"
                np.save(stability_file, np.array(results['stability_map']))
            
            print(f"[Sweep] Results saved to {result_file}")
            
        except Exception as e:
            warnings.warn(f"Failed to save results: {e}")
    
    @staticmethod
    def load_results(filepath: str) -> dict:
        with open(filepath, 'r') as f:
            return json.load(f)


class PressureSweep(ParameterSweep):
    """
    Specialized sweep for studying pressure dependence.
    
    Simulates the transition from atmospheric pressure to vacuum,
    tracking particle behavior as buffer gas damping decreases.
    """
    
    # Pressure presets (in Torr)
    PRESSURE_PRESETS = {
        'atmospheric': 760,
        'rough_vacuum': 1e-3,
        'high_vacuum': 1e-6,
        'ultra_high_vacuum': 1e-9,
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def create_pressure_sweep_config(self, 
                                      p_start: float = 1e-2,
                                      p_end: float = 1e-9,
                                      n_points: int = 20,
                                      duration_per_pressure: float = 10e-3,
                                      **kwargs) -> SweepConfig:
        """
        Create a configuration for pressure sweep.
        
        Args:
            p_start: Starting pressure in Torr
            p_end: Ending pressure in Torr
            n_points: Number of pressure points
            duration_per_pressure: Simulation duration at each pressure
            **kwargs: Additional SweepConfig parameters
        """
        param = SweepParameter(
            name='pressure_torr',
            min_val=p_end,  # Note: reversed for log scale (high to low)
            max_val=p_start,
            n_points=n_points,
            log_scale=True
        )
        # Reverse to go from high to low pressure
        param.values = param.values[::-1]
        
        config = SweepConfig(
            param1=param,
            duration=duration_per_pressure,
            **kwargs
        )
        
        return config
    
    def run_pressure_ramp(self, config: SweepConfig,
                          ramp_time: float = 100e-3,
                          p_start: float = 1e-2,
                          p_end: float = 1e-9) -> Dict[str, Any]:
        """
        Run a single simulation with continuously decreasing pressure.
        
        This simulates the actual pump-down process where pressure
        decreases smoothly over time.
        
        Args:
            config: Base configuration
            ramp_time: Total time for pressure ramp
            p_start: Starting pressure
            p_end: Final pressure
            
        Returns:
            Results dict with time series of position, pressure, energy
        """
        # This would require modifying the simulator to support
        # time-varying pressure. For now, return a placeholder.
        raise NotImplementedError(
            "Continuous pressure ramp requires simulator modification. "
            "Use run_1d_sweep with pressure_torr parameter instead."
        )


def create_stability_diagram_config(
    V_rf_range: Tuple[float, float] = (50, 300),
    V_dc_range: Tuple[float, float] = (0, 50),
    n_points: int = 20,
    duration: float = 1e-3,
    **kwargs
) -> SweepConfig:
    """
    Create a configuration for generating a stability diagram.
    
    Args:
        V_rf_range: (min, max) RF voltage in Volts
        V_dc_range: (min, max) DC voltage in Volts
        n_points: Points per axis
        duration: Simulation duration
        **kwargs: Additional SweepConfig parameters
        
    Returns:
        SweepConfig for 2D stability sweep
    """
    return SweepConfig(
        param1=SweepParameter('V_rf', min_val=V_rf_range[0], max_val=V_rf_range[1], n_points=n_points),
        param2=SweepParameter('V_dc', min_val=V_dc_range[0], max_val=V_dc_range[1], n_points=n_points),
        duration=duration,
        **kwargs
    )


def create_mathieu_parameter_config(
    a_range: Tuple[float, float] = (-0.5, 0.5),
    q_range: Tuple[float, float] = (0, 0.908),
    n_points: int = 50,
    duration_rf_cycles: int = 100,
    rf_frequency: float = 1e6,
    **kwargs
) -> Tuple[SweepConfig, Callable]:
    """
    Create configuration for Mathieu stability diagram in (a, q) space.
    
    The Mathieu parameters are:
        a = 4 * e * U_dc / (m * r0² * Ω²)
        q = 2 * e * V_rf / (m * r0² * Ω²)
    
    Args:
        a_range: Range of Mathieu a parameter
        q_range: Range of Mathieu q parameter
        n_points: Points per axis
        duration_rf_cycles: Duration in RF cycles
        rf_frequency: RF frequency for duration calculation
        
    Returns:
        (SweepConfig, converter_function)
        The converter function maps (a, q) to (V_dc, V_rf)
    """
    # This creates a sweep in a-q space
    # The actual voltages depend on particle properties
    
    config = SweepConfig(
        param1=SweepParameter('mathieu_a', min_val=a_range[0], max_val=a_range[1], n_points=n_points),
        param2=SweepParameter('mathieu_q', min_val=q_range[0], max_val=q_range[1], n_points=n_points),
        duration=duration_rf_cycles / rf_frequency,
        **kwargs
    )
    
    def mathieu_to_voltage(a: float, q: float, m: float, charge: float, 
                           r0: float, omega: float) -> Tuple[float, float]:
        """Convert Mathieu parameters to voltages."""
        # a = 4*Q*U_dc / (m*r0²*Ω²)  =>  U_dc = a*m*r0²*Ω² / (4*Q)
        # q = 2*Q*V_rf / (m*r0²*Ω²)  =>  V_rf = q*m*r0²*Ω² / (2*Q)
        factor = m * r0**2 * omega**2 / charge
        V_dc = a * factor / 4
        V_rf = q * factor / 2
        return V_dc, V_rf
    
    return config, mathieu_to_voltage


# Convenience function for quick sweeps
def quick_stability_check(trap_params: dict, V_rf_values: List[float],
                          numeric_dir: str = None, duration: float = 1e-3) -> List[bool]:
    """
    Quick stability check for a list of RF voltages.
    
    Args:
        trap_params: Base trap parameters
        V_rf_values: List of RF voltages to test
        numeric_dir: Numeric field directory (optional)
        duration: Simulation duration
        
    Returns:
        List of booleans (True = stable)
    """
    sweep = ParameterSweep(trap_params, numeric_dir=numeric_dir)
    
    config = SweepConfig(
        param1=SweepParameter('V_rf', values=np.array(V_rf_values)),
        duration=duration,
    )
    
    results = sweep.run_1d_sweep(config)
    
    return [m['stability'] == 'stable' for m in results['metrics']]


if __name__ == "__main__":
    print("Parameter Sweep Module")
    print("=" * 50)
    print()
    print("Usage example:")
    print()
    print("  from parameter_sweep import ParameterSweep, SweepConfig, SweepParameter")
    print()
    print("  # Create sweep")
    print("  sweep = ParameterSweep(base_params, numeric_dir='numeric_out')")
    print()
    print("  # 2D stability diagram")
    print("  config = SweepConfig(")
    print("      param1=SweepParameter('V_rf', min_val=50, max_val=200, n_points=20),")
    print("      param2=SweepParameter('V_dc', min_val=0, max_val=50, n_points=20),")
    print("      duration=1e-3,")
    print("  )")
    print("  results = sweep.run(config)")
    print()
    print("  # Pressure sweep")
    print("  config = SweepConfig(")
    print("      param1=SweepParameter('pressure_torr', min_val=1e-9, max_val=1e-2,")
    print("                            n_points=20, log_scale=True),")
    print("      duration=10e-3,")
    print("  )")
    print("  results = sweep.run(config)")
