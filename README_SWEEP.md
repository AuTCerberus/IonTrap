# Parameter Sweep System for Ion Trap Simulation

This module provides tools for running parameter sweeps to find stability zones and study how particle behavior changes with different settings.

## Key Features

- **1D Sweeps**: Vary a single parameter (e.g., pressure from atmosphere to vacuum)
- **2D Stability Diagrams**: Map stability zones (e.g., V_rf vs V_dc)
- **Automatic Stability Detection**: Determines if particle stayed trapped
- **Comprehensive Metrics**: Amplitude, energy, frequency, survival time
- **Checkpointing**: Resume interrupted sweeps
- **Visualization**: Stability diagrams, pressure curves, heatmaps
- **Uses Grid Cache**: Fast sweeps when only voltages change

## Files

| File | Purpose |
|------|---------|
| `parameter_sweep.py` | Core sweep engine |
| `sweep_visualization.py` | Plotting functions |
| `sweep_gui.py` | Tkinter GUI panel |

## Quick Start

### 1D Pressure Sweep

```python
from parameter_sweep import ParameterSweep, SweepConfig, SweepParameter

# Base trap parameters
base_params = {
    'V_rf': 100,
    'V_dc': 0,
    'Omega': 2 * np.pi * 1e6,  # 1 MHz
    'r0': 1e-3,
    'particle_radius': 5e-6,
    'particle_density': 2200,
}

# Create sweep
sweep = ParameterSweep(base_params, numeric_dir='numeric_out')

# Configure pressure sweep (high to low)
config = SweepConfig(
    param1=SweepParameter(
        name='pressure_torr',
        min_val=1e-9,      # Ultra-high vacuum
        max_val=1e-2,      # Rough vacuum
        n_points=20,
        log_scale=True     # Logarithmic spacing
    ),
    duration=10e-3,        # 10 ms per pressure
    escape_radius=1e-2,    # Escape if |r| > 1 cm
)

# Run sweep
results = sweep.run(config)

# Visualize
from sweep_visualization import plot_pressure_sweep
fig = plot_pressure_sweep(results)
plt.show()
```

### 2D Stability Diagram (V_rf vs V_dc)

```python
from parameter_sweep import ParameterSweep, SweepConfig, SweepParameter

sweep = ParameterSweep(base_params, numeric_dir='numeric_out')

config = SweepConfig(
    param1=SweepParameter('V_rf', min_val=50, max_val=300, n_points=25),
    param2=SweepParameter('V_dc', min_val=0, max_val=50, n_points=25),
    duration=1e-3,
    escape_radius=1e-2,
)

results = sweep.run(config)  # 625 simulations

# Visualize
from sweep_visualization import plot_stability_diagram
fig = plot_stability_diagram(results)
plt.show()
```

## Available Parameters for Sweeping

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `V_rf` | RF voltage amplitude (V) | 50 - 500 |
| `V_dc` | DC voltage (V) | 0 - 100 |
| `frequency` | RF frequency (Hz) | 1e5 - 1e7 |
| `pressure_torr` | Background pressure (Torr) | 1e-9 - 1e-2 |
| `damping_gamma` | Damping coefficient (1/s) | 0 - 1000 |
| `n_charges` | Number of elementary charges | 10 - 10000 |
| `particle_radius` | Particle radius (m) | 1e-7 - 1e-4 |

## Stability Detection

A simulation is classified as:

| Result | Condition |
|--------|-----------|
| **STABLE** | Particle stayed within bounds AND final amplitude < threshold |
| **ESCAPED** | Particle left escape radius during simulation |
| **UNSTABLE** | Particle stayed in bounds but amplitude growing |
| **CRASHED** | Simulation error (NaN, timeout, etc.) |

## Metrics Extracted

For each simulation, these metrics are recorded:

```python
metrics = {
    # Stability
    'stability': 'stable' | 'escaped' | 'unstable' | 'crashed',
    'survival_time': float,  # Time before escape (or full duration)
    
    # Position (meters)
    'max_amplitude_x': float,
    'max_amplitude_y': float,
    'max_amplitude_z': float,
    'max_amplitude_r': float,  # Radial
    'final_amplitude_x': float,  # Averaged over last 10%
    'final_amplitude_y': float,
    'final_amplitude_z': float,
    
    # Energy (Joules)
    'mean_kinetic_energy': float,
    'final_kinetic_energy': float,
    
    # Frequency (Hz)
    'secular_frequency_x': float,  # Estimated from FFT
    
    # Timing
    'computation_time': float,  # Wall clock seconds
}
```

## Pressure Sweep: Atmosphere to Vacuum

### Physics Background

At higher pressures, buffer gas provides damping:
- **High pressure (>1e-3 Torr)**: Strong damping, particle quickly reaches equilibrium
- **Medium pressure (1e-6 Torr)**: Moderate damping, secular oscillations visible
- **Ultra-high vacuum (<1e-9 Torr)**: Minimal damping, micromotion dominates

### Example Analysis

```python
# Run pressure sweep
results = sweep.run(config)

# Extract metrics vs pressure
pressures = results['parameter_values']
amplitudes = [m['final_amplitude_r'] for m in results['metrics']]
energies = [m['mean_kinetic_energy'] for m in results['metrics']]

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.loglog(pressures, amplitudes)
ax1.set_ylabel('Final Amplitude (m)')

ax2.loglog(pressures, energies)
ax2.set_ylabel('Mean Kinetic Energy (J)')
ax2.set_xlabel('Pressure (Torr)')
```

## Integration with Grid Cache

The sweep system automatically benefits from grid caching:

```python
# First simulation: computes and caches grid (~60 sec)
# Remaining 399 simulations: use cached grid (~2 sec each)

config = SweepConfig(
    param1=SweepParameter('V_rf', min_val=50, max_val=200, n_points=20),
    param2=SweepParameter('pressure_torr', min_val=1e-9, max_val=1e-3, n_points=20),
    ...
)
results = sweep.run(config)  # 400 simulations

# Total time: ~60 + 400*2 = 860 sec (14 min)
# Without cache: 400*60 = 24000 sec (6.7 hours)!
```

## Output Files

Results are saved to the output directory:

```
sweep_results/
├── sweep_1d_20240115_143022.json     # Full results
├── sweep_2d_20240115_150000.json     # Full results
├── stability_map_20240115_150000.npy  # Numpy array (2D only)
└── sweep_checkpoint.json              # Resume interrupted sweep
```

## GUI Integration

Add the sweep panel to your GUI:

```python
from sweep_gui import SweepConfigPanel

# In your GUI setup:
def on_sweep_run(config):
    # Start sweep in background thread
    threading.Thread(target=run_sweep, args=(config,)).start()

def on_sweep_stop():
    sweep.stop()

sweep_panel = SweepConfigPanel(
    parent_frame,
    on_run=on_sweep_run,
    on_stop=on_sweep_stop,
    get_base_params=get_current_trap_params
)
```

## Visualization Functions

```python
from sweep_visualization import (
    plot_1d_sweep,              # 1D parameter vs stability/amplitude
    plot_stability_diagram,      # 2D stability map (binary)
    plot_stability_diagram_contour,  # 2D with contour lines
    plot_pressure_sweep,         # Pressure-specific plots
    plot_metric_heatmap,         # 2D heatmap of any metric
    plot_sweep_comparison,       # Compare multiple sweeps
    create_sweep_report,         # Text summary
    save_figures,                # Save all plots to disk
)
```

## Important Notes

I don't recommend using secular approximation or analytical simulation during sweeps, as they don't have a definded escape radius. 
If you do so anyway, set the escape radius to reasonably small estimate.