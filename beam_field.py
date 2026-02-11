import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Callable
from enum import Enum


class BeamType(Enum):
    #Type of particle beam
    ELECTRON = "electron"
    ION = "ion"


class PhaseMode(Enum):
    """Phase relation mode for beam activation"""
    ZERO_CROSSING_POSITIVE = "zero_crossing_positive"  # RF going from - to +
    ZERO_CROSSING_NEGATIVE = "zero_crossing_negative"  # RF going from + to -
    ZERO_CROSSING_BOTH = "zero_crossing_both"          # Any zero crossing
    CONTINUOUS = "continuous"                           # Always on (no phase lock)
    CUSTOM = "custom"                                   # User-defined phase window


@dataclass
class BeamParameters:
    """Parameters defining the particle beam"""
    
    # Beam type
    beam_type: BeamType = BeamType.ELECTRON
    
    # Beam current and energy
    current: float = 1e-9          # Beam current in Amperes
    energy: float = 100.0          # Beam energy in eV
    
    # Beam geometry (assuming beam propagates along one axis)
    propagation_axis: Literal['x', 'y', 'z'] = 'z'
    beam_center: Tuple[float, float] = (0.0, 0.0)  # Center in plane perpendicular to propagation (meters)
    beam_radius: float = 1e-3      # Beam radius in meters (Gaussian 1/e^2 radius)
    
    # Phase-locking parameters
    phase_mode: PhaseMode = PhaseMode.ZERO_CROSSING_BOTH
    phase_window: float = 0.1      # Fraction of RF period when beam is on (0-1)
    custom_phase_start: float = 0.0  # Start phase in radians (for CUSTOM mode)
    custom_phase_end: float = np.pi/10  # End phase in radians (for CUSTOM mode)
    
    # Timing parameters
    delay_start: float = 0.0       # Delay before beam turns on (in s)
    pulse_duration: float = np.inf  # How long beam stays on after delay (in s)
    
    # Interaction model
    interaction_strength: float = 1.0  # Scaling factor for beam-particle interaction
    
    # Enable/disable
    enabled: bool = False


class ParticleBeam:
    """
    Particle beam (electron or ion) that can interact with trapped particles.
    The beam can be phase-locked to the RF drive, activating only during
    specific phases of the RF cycle (e.g., at zero crossings).
    """
    
    # Physical constants
    ELECTRON_CHARGE = 1.602176634e-19  # C
    ELECTRON_MASS = 9.1093837015e-31   # kg
    PROTON_MASS = 1.67262192369e-27    # kg
    EPSILON_0 = 8.8541878128e-12       # F/m
    
    def __init__(self, params: Optional[BeamParameters] = None):
        """
        Initialize the particle beam.
        
        Args:
            params: BeamParameters object defining beam properties
        """
        self.params = params if params is not None else BeamParameters()
        self._rf_omega = None  # Angular frequency (rad/s)
        self._rf_phase_offset = 0.0
        
        # Callback for status updates (for GUI)
        self._status_callback = None
        
    def set_rf_parameters(self, omega: float = None, frequency: float = None, 
                          phase_offset: float = 0.0):
        """
        Set the RF frequency for phase locking.
        
        Args:
            omega: RF angular frequency in rad/s (takes precedence)
            frequency: RF frequency in Hz (converted to omega)
            phase_offset: Phase offset in radians
        """
        if omega is not None:
            self._rf_omega = omega
        elif frequency is not None:
            self._rf_omega = 2 * np.pi * frequency
        self._rf_phase_offset = phase_offset
        
    def set_status_callback(self, callback: Callable):
        #Set a callback for status updates: callback(is_active, phase, delay_remaining)
        self._status_callback = callback
        
    def _get_rf_phase(self, t: float) -> float:
        #Calculate the current RF phase at time t
        if self._rf_omega is None or self._rf_omega == 0:
            return 0.0
        return (self._rf_omega * t + self._rf_phase_offset) % (2 * np.pi)
    
    def is_beam_active(self, t: float) -> bool:
        """
        Determine if the beam should be active at time t.
        
        Takes into account:
        - Delay start
        - Pulse duration
        - Phase locking to RF
        
        Args:
            t: Current simulation time in seconds
            
        Returns:
            True if beam is active, False otherwise
        """
        if not self.params.enabled:
            return False
            
        # Check delay
        if t < self.params.delay_start:
            return False
            
        # Check pulse duration
        time_since_start = t - self.params.delay_start
        if time_since_start > self.params.pulse_duration:
            return False
            
        # Check phase locking
        if self.params.phase_mode == PhaseMode.CONTINUOUS:
            return True
            
        if self._rf_omega is None or self._rf_omega == 0:
            # No RF frequency set, default to continuous
            return True
            
        phase = self._get_rf_phase(t)
        half_window = self.params.phase_window * np.pi  # Convert to radians
        
        if self.params.phase_mode == PhaseMode.ZERO_CROSSING_POSITIVE:
            # Active near phase = 0 (RF going positive)
            return phase < half_window or phase > (2 * np.pi - half_window)
            
        elif self.params.phase_mode == PhaseMode.ZERO_CROSSING_NEGATIVE:
            # Active near phase = pi (RF going negative)
            return abs(phase - np.pi) < half_window
            
        elif self.params.phase_mode == PhaseMode.ZERO_CROSSING_BOTH:
            # Active near both zero crossings
            near_zero = phase < half_window or phase > (2 * np.pi - half_window)
            near_pi = abs(phase - np.pi) < half_window
            return near_zero or near_pi
            
        elif self.params.phase_mode == PhaseMode.CUSTOM:
            # User-defined phase window
            start = self.params.custom_phase_start % (2 * np.pi)
            end = self.params.custom_phase_end % (2 * np.pi)
            if start <= end:
                return start <= phase <= end
            else:
                return phase >= start or phase <= end
                
        return False
    
    def _calculate_beam_intensity(self, position: np.ndarray) -> float:
        """
        Calculate the beam intensity at a given position.
        
        Assumes a Gaussian beam profile perpendicular to propagation axis.
        """
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        prop_axis = axis_map[self.params.propagation_axis]
        transverse_axes = [i for i in range(3) if i != prop_axis]
        center = self.params.beam_center
        
        r_squared = 0.0
        for i, ax in enumerate(transverse_axes):
            r_squared += (position[ax] - center[i])**2
            
        sigma = self.params.beam_radius / 2.0
        intensity = np.exp(-r_squared / (2 * sigma**2))
        
        return intensity
    
    def _calculate_beam_velocity(self) -> float:
        #Calculate the beam particle velocity from energy
        if self.params.beam_type == BeamType.ELECTRON:
            mass = self.ELECTRON_MASS
        else:
            mass = self.PROTON_MASS
            
        energy_joules = self.params.energy * self.ELECTRON_CHARGE
        velocity = np.sqrt(2 * energy_joules / mass)
        return velocity
    
    def _calculate_linear_charge_density(self) -> float:
        #Calculate the linear charge density of the beam
        velocity = self._calculate_beam_velocity()
        return self.params.current / velocity
    
    def calculate_force(self, position: np.ndarray, t: float, 
                        particle_charge: float = None) -> np.ndarray:
        """
        Calculate the force on a trapped particle from the beam.
        
        Args:
            position: 3D position of trapped particle [x, y, z] in meters
            t: Current simulation time in seconds
            particle_charge: Charge of the trapped particle (default: +e)
            
        Returns:
            Force vector [Fx, Fy, Fz] in Newtons
        """
        force = np.zeros(3)
        
        if not self.is_beam_active(t):
            return force
            
        intensity = self._calculate_beam_intensity(position)
        
        if intensity < 1e-10:
            return force
            
        lambda_charge = self._calculate_linear_charge_density()
        
        if self.params.beam_type == BeamType.ELECTRON:
            beam_charge_sign = -1
        else:
            beam_charge_sign = +1
            
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        prop_axis = axis_map[self.params.propagation_axis]
        transverse_axes = [i for i in range(3) if i != prop_axis]
        
        center = self.params.beam_center
        r_vec = np.zeros(3)
        r_squared = 0.0
        for i, ax in enumerate(transverse_axes):
            displacement = position[ax] - center[i]
            r_vec[ax] = displacement
            r_squared += displacement**2
            
        r = np.sqrt(r_squared)
        
        if r < 1e-15:
            return force
            
        E_magnitude = abs(lambda_charge) / (2 * np.pi * self.EPSILON_0 * r)
        E_magnitude *= intensity
        E_magnitude *= self.params.interaction_strength
        
        direction = r_vec / r
        E_field = E_magnitude * beam_charge_sign * direction
        
        if particle_charge is None:
            particle_charge = self.ELECTRON_CHARGE
        
        force = particle_charge * E_field
        
        return force
    
    def calculate_force_vectorized(self, positions: np.ndarray, t: float,
                                    particle_charge: float = None) -> np.ndarray:
        """
        Calculate beam force for multiple particles (vectorized).
        
        Args:
            positions: Nx3 array of positions in meters
            t: Current time in seconds
            particle_charge: Charge of particles (scalar)
            
        Returns:
            Nx3 array of forces in Newtons
        """
        n_particles = positions.shape[0]
        forces = np.zeros((n_particles, 3))
        
        if not self.is_beam_active(t):
            return forces
            
        if particle_charge is None:
            particle_charge = self.ELECTRON_CHARGE
            
        lambda_charge = self._calculate_linear_charge_density()
        
        if self.params.beam_type == BeamType.ELECTRON:
            beam_charge_sign = -1
        else:
            beam_charge_sign = +1
            
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        prop_axis = axis_map[self.params.propagation_axis]
        transverse_axes = [i for i in range(3) if i != prop_axis]
        center = np.array(self.params.beam_center)
        
        r_vec = np.zeros_like(positions)
        r_squared = np.zeros(n_particles)
        for i, ax in enumerate(transverse_axes):
            displacement = positions[:, ax] - center[i]
            r_vec[:, ax] = displacement
            r_squared += displacement**2
            
        r = np.sqrt(r_squared)
        valid = r > 1e-15
        
        sigma = self.params.beam_radius / 2.0
        intensity = np.exp(-r_squared / (2 * sigma**2))
        
        E_magnitude = np.zeros(n_particles)
        E_magnitude[valid] = (abs(lambda_charge) / 
                              (2 * np.pi * self.EPSILON_0 * r[valid]))
        E_magnitude *= intensity * self.params.interaction_strength
        
        direction = np.zeros_like(r_vec)
        direction[valid] = r_vec[valid] / r[valid, np.newaxis]
        
        forces = (particle_charge * beam_charge_sign * E_magnitude[:, np.newaxis] * 
                  direction)
        
        return forces
    
    def get_status(self, t: float) -> dict:
        #Get the current status of the beam
        rf_freq = self._rf_omega / (2 * np.pi) if self._rf_omega else None
        
        return {
            'enabled': self.params.enabled,
            'active': self.is_beam_active(t),
            'time': t,
            'delay_remaining': max(0, self.params.delay_start - t),
            'rf_phase': self._get_rf_phase(t) if self._rf_omega else None,
            'rf_frequency': rf_freq,
            'phase_mode': self.params.phase_mode.value,
            'beam_type': self.params.beam_type.value,
            'current': self.params.current,
            'energy': self.params.energy,
        }


if __name__ == "__main__":
    # Test the beam module
    params = BeamParameters(
        beam_type=BeamType.ELECTRON,
        current=1e-9,
        energy=100.0,
        propagation_axis='z',
        beam_center=(0.0, 0.0),
        beam_radius=1e-3,
        phase_mode=PhaseMode.ZERO_CROSSING_BOTH,
        phase_window=0.1,
        delay_start=1e-3,
        enabled=True
    )
    
    beam = ParticleBeam(params)
    beam.set_rf_parameters(frequency=1e6)
    
    print("Beam activation test:")
    print("=" * 50)
    
    rf_period = 1e-6
    t_base = 1.5e-3
    
    for phase_frac in [0.0, 0.25, 0.5, 0.75]:
        t = t_base + phase_frac * rf_period
        status = beam.get_status(t)
        phase_deg = (status['rf_phase'] or 0) * 180 / np.pi
        print(f"RF phase = {phase_deg:.1f}°: active = {status['active']}")