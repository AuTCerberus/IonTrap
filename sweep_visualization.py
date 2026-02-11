"""
Visualization Module for Parameter Sweep Results

Provides plotting functions for:
- 1D parameter sweeps
- 2D stability diagrams
- Pressure sweep analysis
- Comparison plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


# Custom colormaps
STABILITY_CMAP = LinearSegmentedColormap.from_list(
    'stability', 
    [(0, '#ff4444'), (0.5, '#ffaa00'), (1, '#44ff44')],  # Red -> Orange -> Green
    N=256
)

STABILITY_BINARY_CMAP = LinearSegmentedColormap.from_list(
    'stability_binary',
    [(0, '#ff6666'), (1, '#66ff66')],  # Red, Green
    N=2
)


def plot_1d_sweep(results: dict, 
                  metric: str = 'stability',
                  ax: plt.Axes = None,
                  show_boundaries: bool = True,
                  **kwargs) -> plt.Figure:
    """
    Plot results of a 1D parameter sweep.
    
    Args:
        results: Results dict from ParameterSweep.run_1d_sweep()
        metric: Which metric to plot ('stability', 'max_amplitude_r', 'survival_time', etc.)
        ax: Matplotlib axes (creates new figure if None)
        show_boundaries: Show stability boundaries
        **kwargs: Additional plot kwargs
        
    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    param_name = results['parameter_name']
    param_values = np.array(results['parameter_values'])
    metrics_list = results['metrics']
    
    if metric == 'stability':
        # Binary stability plot
        stable = np.array([m['stability'] == 'stable' for m in metrics_list])
        colors = ['#66ff66' if s else '#ff6666' for s in stable]
        ax.scatter(param_values, stable.astype(int), c=colors, s=100, **kwargs)
        ax.set_ylabel('Stable')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No', 'Yes'])
        ax.set_ylim(-0.2, 1.2)
        
        # Show stability boundaries
        if show_boundaries and 'summary' in results:
            boundaries = results['summary'].get('stability_boundaries', [])
            for b in boundaries:
                ax.axvline(b, color='orange', linestyle='--', alpha=0.7, label='Boundary')
    else:
        # Numeric metric plot
        values = np.array([m.get(metric, np.nan) for m in metrics_list])
        
        # Color by stability
        colors = ['#66ff66' if m['stability'] == 'stable' else '#ff6666' for m in metrics_list]
        ax.scatter(param_values, values, c=colors, s=50, **kwargs)
        ax.plot(param_values, values, 'k-', alpha=0.3)
        ax.set_ylabel(metric.replace('_', ' ').title())
    
    ax.set_xlabel(param_name.replace('_', ' ').title())
    ax.set_title(f'1D Parameter Sweep: {param_name}')
    ax.grid(True, alpha=0.3)
    
    # Add summary info
    if 'summary' in results:
        summary = results['summary']
        info_text = f"Stable: {summary['stable_count']}/{summary['total_runs']} ({summary['stable_fraction']*100:.1f}%)"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.tight_layout()
    return fig


def plot_stability_diagram(results: dict,
                           ax: plt.Axes = None,
                           cmap: str = None,
                           show_colorbar: bool = True,
                           interpolation: str = 'nearest',
                           **kwargs) -> plt.Figure:
    """
    Plot a 2D stability diagram.
    
    Args:
        results: Results dict from ParameterSweep.run_2d_sweep()
        ax: Matplotlib axes (creates new figure if None)
        cmap: Colormap name (default: binary red/green)
        show_colorbar: Show colorbar
        interpolation: Image interpolation method
        **kwargs: Additional imshow kwargs
        
    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    param1_name = results['param1_name']
    param2_name = results['param2_name']
    param1_values = np.array(results['param1_values'])
    param2_values = np.array(results['param2_values'])
    stability_map = np.array(results['stability_map'])
    
    # Use binary colormap by default
    if cmap is None:
        cmap = STABILITY_BINARY_CMAP
    
    # Plot
    extent = [param2_values[0], param2_values[-1], 
              param1_values[0], param1_values[-1]]
    
    im = ax.imshow(stability_map, origin='lower', aspect='auto',
                   extent=extent, cmap=cmap, interpolation=interpolation,
                   vmin=0, vmax=1, **kwargs)
    
    ax.set_xlabel(param2_name.replace('_', ' ').title())
    ax.set_ylabel(param1_name.replace('_', ' ').title())
    ax.set_title(f'Stability Diagram: {param1_name} vs {param2_name}')
    
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, ticks=[0.25, 0.75])
        cbar.ax.set_yticklabels(['Unstable', 'Stable'])
    
    # Add summary info
    if 'summary' in results:
        summary = results['summary']
        info_text = f"Stable: {summary['stable_fraction']*100:.1f}%"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10, color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    fig.tight_layout()
    return fig


def plot_stability_diagram_contour(results: dict,
                                    ax: plt.Axes = None,
                                    levels: int = 2,
                                    show_points: bool = True,
                                    **kwargs) -> plt.Figure:
    """
    Plot stability diagram with contour lines.
    
    Args:
        results: Results dict from 2D sweep
        ax: Matplotlib axes
        levels: Number of contour levels
        show_points: Show individual simulation points
        **kwargs: Additional contour kwargs
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    param1_values = np.array(results['param1_values'])
    param2_values = np.array(results['param2_values'])
    stability_map = np.array(results['stability_map'])
    
    # Create meshgrid
    P2, P1 = np.meshgrid(param2_values, param1_values)
    
    # Contour plot
    cs = ax.contourf(P2, P1, stability_map, levels=[0, 0.5, 1],
                     colors=['#ffcccc', '#ccffcc'], **kwargs)
    ax.contour(P2, P1, stability_map, levels=[0.5], colors='black', linewidths=2)
    
    # Show individual points
    if show_points:
        for i, v1 in enumerate(param1_values):
            for j, v2 in enumerate(param2_values):
                stable = stability_map[i, j]
                color = '#00aa00' if stable else '#aa0000'
                ax.plot(v2, v1, 'o', color=color, markersize=4, alpha=0.5)
    
    ax.set_xlabel(results['param2_name'].replace('_', ' ').title())
    ax.set_ylabel(results['param1_name'].replace('_', ' ').title())
    ax.set_title('Stability Boundary')
    
    # Legend
    legend_elements = [
        Patch(facecolor='#ccffcc', edgecolor='black', label='Stable'),
        Patch(facecolor='#ffcccc', edgecolor='black', label='Unstable'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    fig.tight_layout()
    return fig


def plot_pressure_sweep(results: dict,
                        metrics: List[str] = None,
                        ax: plt.Axes = None,
                        log_pressure: bool = True,
                        **kwargs) -> plt.Figure:
    """
    Plot results of a pressure sweep.
    
    Args:
        results: Results dict from pressure sweep
        metrics: List of metrics to plot (default: amplitude and energy)
        ax: Matplotlib axes
        log_pressure: Use log scale for pressure axis
        **kwargs: Additional plot kwargs
        
    Returns:
        Matplotlib figure
    """
    if metrics is None:
        metrics = ['final_amplitude_r', 'mean_kinetic_energy']
    
    n_metrics = len(metrics)
    
    if ax is None:
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4*n_metrics), sharex=True)
        if n_metrics == 1:
            axes = [axes]
    else:
        fig = ax.figure
        axes = [ax]
    
    pressure_values = np.array(results['parameter_values'])
    metrics_list = results['metrics']
    
    for i, metric in enumerate(metrics[:len(axes)]):
        ax = axes[i]
        
        values = np.array([m.get(metric, np.nan) for m in metrics_list])
        stable = np.array([m['stability'] == 'stable' for m in metrics_list])
        
        # Plot with stability coloring
        ax.scatter(pressure_values[stable], values[stable], 
                   c='#66ff66', label='Stable', s=50, **kwargs)
        ax.scatter(pressure_values[~stable], values[~stable],
                   c='#ff6666', label='Unstable', s=50, marker='x', **kwargs)
        ax.plot(pressure_values[stable], values[stable], 'g-', alpha=0.5)
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        
        if log_pressure:
            ax.set_xscale('log')
        
        if i == 0:
            ax.legend()
    
    axes[-1].set_xlabel('Pressure (Torr)')
    axes[0].set_title('Pressure Sweep Results')
    
    fig.tight_layout()
    return fig


def plot_metric_heatmap(results: dict,
                        metric: str = 'max_amplitude_r',
                        ax: plt.Axes = None,
                        cmap: str = 'viridis',
                        log_scale: bool = False,
                        mask_unstable: bool = True,
                        **kwargs) -> plt.Figure:
    """
    Plot a heatmap of a specific metric from 2D sweep.
    
    Args:
        results: Results dict from 2D sweep
        metric: Metric to plot
        ax: Matplotlib axes
        cmap: Colormap
        log_scale: Use log scale for color
        mask_unstable: Gray out unstable regions
        **kwargs: Additional imshow kwargs
    """
    if results['type'] != '2d':
        raise ValueError("This function requires 2D sweep results")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    param1_values = np.array(results['param1_values'])
    param2_values = np.array(results['param2_values'])
    metrics_grid = results['metrics_grid']
    stability_map = np.array(results['stability_map'])
    
    # Extract metric values
    n1, n2 = len(param1_values), len(param2_values)
    metric_map = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            m = metrics_grid[i][j]
            if m is not None:
                metric_map[i, j] = m.get(metric, np.nan)
            else:
                metric_map[i, j] = np.nan
    
    # Mask unstable regions
    if mask_unstable:
        metric_map = np.ma.masked_where(stability_map == 0, metric_map)
    
    # Log scale
    if log_scale:
        metric_map = np.log10(np.clip(metric_map, 1e-20, None))
    
    # Plot
    extent = [param2_values[0], param2_values[-1],
              param1_values[0], param1_values[-1]]
    
    im = ax.imshow(metric_map, origin='lower', aspect='auto',
                   extent=extent, cmap=cmap, **kwargs)
    
    ax.set_xlabel(results['param2_name'].replace('_', ' ').title())
    ax.set_ylabel(results['param1_name'].replace('_', ' ').title())
    
    title = metric.replace('_', ' ').title()
    if log_scale:
        title = f'log₁₀({title})'
    ax.set_title(title)
    
    fig.colorbar(im, ax=ax)
    
    # Overlay stability boundary
    ax.contour(param2_values, param1_values, stability_map,
               levels=[0.5], colors='white', linewidths=2, linestyles='--')
    
    fig.tight_layout()
    return fig


def plot_sweep_comparison(results_list: List[dict],
                          labels: List[str] = None,
                          metric: str = 'stability',
                          ax: plt.Axes = None,
                          **kwargs) -> plt.Figure:
    """
    Compare multiple 1D sweep results.
    
    Args:
        results_list: List of results dicts
        labels: Labels for each result set
        metric: Metric to compare
        ax: Matplotlib axes
        **kwargs: Additional plot kwargs
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure
    
    if labels is None:
        labels = [f'Sweep {i+1}' for i in range(len(results_list))]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
    
    for results, label, color in zip(results_list, labels, colors):
        param_values = np.array(results['parameter_values'])
        metrics_list = results['metrics']
        
        if metric == 'stability':
            values = np.array([1 if m['stability'] == 'stable' else 0 for m in metrics_list])
        else:
            values = np.array([m.get(metric, np.nan) for m in metrics_list])
        
        ax.plot(param_values, values, 'o-', color=color, label=label, **kwargs)
    
    ax.set_xlabel(results_list[0]['parameter_name'].replace('_', ' ').title())
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Comparison: {metric}')
    
    fig.tight_layout()
    return fig


def plot_mathieu_stability_diagram(results: dict,
                                    ax: plt.Axes = None,
                                    show_theoretical: bool = True,
                                    **kwargs) -> plt.Figure:
    """
    Plot stability diagram in Mathieu parameter space.
    
    Args:
        results: Results from Mathieu parameter sweep
        ax: Matplotlib axes
        show_theoretical: Overlay theoretical stability boundaries
        **kwargs: Additional plot kwargs
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    # Plot simulation results
    plot_stability_diagram(results, ax=ax, **kwargs)
    
    if show_theoretical:
        # Theoretical Mathieu stability boundaries (first region)
        # These are approximate boundaries for the first stability region
        q_theory = np.linspace(0, 0.908, 100)
        
        # Lower boundary: a ≈ -q²/2 (approximate)
        a_lower = -q_theory**2 / 2
        
        # Upper boundary: a ≈ 1 - q - q²/8 (approximate for small q)
        a_upper = 1 - q_theory - q_theory**2 / 8
        
        # Only plot where valid (inside first stability region)
        valid = (a_upper > a_lower) & (q_theory < 0.908)
        
        ax.plot(q_theory[valid], a_lower[valid], 'k--', linewidth=2, 
                label='Theoretical boundary', alpha=0.7)
        ax.plot(q_theory[valid], a_upper[valid], 'k--', linewidth=2, alpha=0.7)
        
        ax.legend()
    
    ax.set_xlabel('q (Mathieu parameter)')
    ax.set_ylabel('a (Mathieu parameter)')
    ax.set_title('Mathieu Stability Diagram')
    
    return fig


def create_sweep_report(results: dict, output_path: str = None) -> str:
    """
    Generate a text report summarizing sweep results.
    
    Args:
        results: Results dict
        output_path: Optional path to save report
        
    Returns:
        Report text
    """
    lines = []
    lines.append("=" * 60)
    lines.append("PARAMETER SWEEP REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    # Basic info
    lines.append(f"Sweep Type: {results.get('type', 'unknown').upper()}")
    lines.append(f"Started: {results.get('started', 'N/A')}")
    lines.append(f"Completed: {results.get('completed', 'N/A')}")
    lines.append("")
    
    # Parameters
    if results['type'] == '1d':
        lines.append(f"Parameter: {results['parameter_name']}")
        values = results['parameter_values']
        lines.append(f"Range: {values[0]:.3e} to {values[-1]:.3e}")
        lines.append(f"Points: {len(values)}")
    else:
        lines.append(f"Parameter 1: {results['param1_name']}")
        v1 = results['param1_values']
        lines.append(f"  Range: {v1[0]:.3e} to {v1[-1]:.3e} ({len(v1)} points)")
        lines.append(f"Parameter 2: {results['param2_name']}")
        v2 = results['param2_values']
        lines.append(f"  Range: {v2[0]:.3e} to {v2[-1]:.3e} ({len(v2)} points)")
    
    lines.append("")
    
    # Summary
    if 'summary' in results:
        lines.append("-" * 40)
        lines.append("SUMMARY")
        lines.append("-" * 40)
        summary = results['summary']
        lines.append(f"Total Simulations: {summary['total_runs']}")
        lines.append(f"Stable: {summary['stable_count']} ({summary['stable_fraction']*100:.1f}%)")
        
        if 'stability_boundaries' in summary:
            lines.append(f"Stability Boundaries: {summary['stability_boundaries']}")
        
        if 'mean_computation_time' in summary:
            lines.append(f"Mean Computation Time: {summary['mean_computation_time']:.2f} s")
    
    lines.append("")
    lines.append("=" * 60)
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report


def save_figures(results: dict, output_dir: str, prefix: str = "sweep"):
    """
    Save all relevant figures for sweep results.
    
    Args:
        results: Results dict
        output_dir: Output directory
        prefix: Filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if results['type'] == '1d':
        # Stability plot
        fig = plot_1d_sweep(results, metric='stability')
        fig.savefig(output_dir / f"{prefix}_stability.png", dpi=150)
        plt.close(fig)
        
        # Amplitude plot
        fig = plot_1d_sweep(results, metric='max_amplitude_r')
        fig.savefig(output_dir / f"{prefix}_amplitude.png", dpi=150)
        plt.close(fig)
        
    else:  # 2d
        # Stability diagram
        fig = plot_stability_diagram(results)
        fig.savefig(output_dir / f"{prefix}_stability_diagram.png", dpi=150)
        plt.close(fig)
        
        # Contour plot
        fig = plot_stability_diagram_contour(results)
        fig.savefig(output_dir / f"{prefix}_stability_contour.png", dpi=150)
        plt.close(fig)
        
        # Amplitude heatmap
        try:
            fig = plot_metric_heatmap(results, metric='max_amplitude_r')
            fig.savefig(output_dir / f"{prefix}_amplitude_heatmap.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass
    
    # Text report
    report = create_sweep_report(results)
    with open(output_dir / f"{prefix}_report.txt", 'w') as f:
        f.write(report)
    
    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    # Demo with synthetic data
    print("Visualization module for parameter sweeps")
    print("Use with results from ParameterSweep class")
