"""
Global grid cache for reusing expensive GridInterpolatedField instances.

This allows simulations to reuse the same grid when only voltages or other
non-grid parameters change (e.g., beam on/off, damping, initial conditions).
"""

import hashlib
import json
import numpy as np
from typing import Optional, Dict, Any

# Global cache: {cache_key: GridInterpolatedField instance}
_GRID_CACHE: Dict[str, Any] = {}


def _create_cache_key(
    numeric_field_dir: str,
    grid_points: int,
    adaptive_refinement: bool,
    smoothing_sigma: float,
    omega: float,
    v_rf: float,
    v_dc: float,
    electrode_program: dict,
) -> str:
    """
    Create a cache key from grid-affecting parameters.

    Parameters that affect grid values (require rebuild):
    - numeric_field_dir: path to field files
    - grid_points: resolution
    - adaptive_refinement: whether to use adaptive refinement
    - smoothing_sigma: Gaussian smoothing parameter
    - omega: RF frequency (affects time-varying field component)
    - v_rf: RF voltage (affects field magnitudes)
    - v_dc: DC voltage (affects field magnitudes)
    - electrode_program: specific voltage on each electrode

    Parameters that DON'T affect grid (and should allow reuse):
    - Beam settings (beam doesn't affect trap field)
    - Damping coefficients (affects forces, not fields)
    - Particle properties (mass, charge - affect dynamics, not fields)
    - Initial conditions (don't affect fields)
    """
    # Serialize electrode_program in a deterministic way
    # Handle functions by evaluating them at t=0 and t=pi/2 to capture RF behavior
    electrode_program_serialized = ""
    if electrode_program:
        serialized_items = []
        for name, value in sorted(electrode_program.items()):
            if callable(value):
                # Evaluate function at t=0 and t=quarter_period to capture RF amplitude
                try:
                    v_at_0 = value(0.0)
                    quarter_period = (np.pi / 2) / omega if omega > 0 else 0
                    v_at_quarter = value(quarter_period)
                    # Use both values to capture phase and amplitude
                    serialized_items.append((name, f"func({v_at_0:.12e},{v_at_quarter:.12e})"))
                except Exception:
                    # If function evaluation fails, use a generic marker
                    serialized_items.append((name, "func"))
            else:
                # Numeric value - use directly
                serialized_items.append((name, float(value)))
        electrode_program_serialized = str(serialized_items)

    cache_dict = {
        "numeric_field_dir": str(numeric_field_dir),
        "grid_points": int(grid_points),
        "adaptive_refinement": bool(adaptive_refinement),
        "smoothing_sigma": float(smoothing_sigma),
        "omega": float(omega),
        "v_rf": float(v_rf),
        "v_dc": float(v_dc),
        "electrode_program": electrode_program_serialized,
    }

    # Create deterministic hash
    cache_str = json.dumps(cache_dict, sort_keys=True)
    cache_key = hashlib.sha256(cache_str.encode()).hexdigest()[:16]

    return cache_key


def get_cached_grid(
    numeric_field_dir: str,
    grid_points: int,
    adaptive_refinement: bool,
    smoothing_sigma: float,
    omega: float,
    v_rf: float,
    v_dc: float,
    electrode_program: dict,
) -> Optional[Any]:
    """
    Retrieve a cached grid if available.

    Returns:
        GridInterpolatedField instance if cached, None otherwise
    """
    cache_key = _create_cache_key(
        numeric_field_dir, grid_points, adaptive_refinement, smoothing_sigma, omega,
        v_rf, v_dc, electrode_program
    )

    if cache_key in _GRID_CACHE:
        print(f"[GridCache] Reusing cached grid")
        return _GRID_CACHE[cache_key]

    return None


def cache_grid(
    numeric_field_dir: str,
    grid_points: int,
    adaptive_refinement: bool,
    smoothing_sigma: float,
    omega: float,
    v_rf: float,
    v_dc: float,
    electrode_program: dict,
    grid_instance: Any,
) -> None:
    
    #Cache a grid instance for future reuse.
    cache_key = _create_cache_key(
        numeric_field_dir, grid_points, adaptive_refinement, smoothing_sigma, omega,
        v_rf, v_dc, electrode_program
    )

    _GRID_CACHE[cache_key] = grid_instance
    print(f"[GridCache] Grid cached ({len(_GRID_CACHE)} total)")


def clear_cache() -> None:
    """
    Clear all cached grids.
    """
    count = len(_GRID_CACHE)
    _GRID_CACHE.clear()
    print(f"[GridCache] Cleared {count} cached grid(s)")


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.
    """
    return {
        "cached_grids": len(_GRID_CACHE),
        "cache_keys": list(_GRID_CACHE.keys()),
    }
