import tkinter as tk
from tkinter import ttk
import numpy as np
from beam_field import BeamParameters, BeamType, PhaseMode, ParticleBeam


class BeamControlPanel(ttk.Frame):
    """
    Tkinter panel for controlling particle beam parameters.
    
    Usage:
        panel = BeamControlPanel(parent_frame, on_change_callback)
        # Get current beam instance
        beam = panel.get_beam()
        # Update RF parameters
        panel.set_rf_parameters(omega=2*np.pi*1e6)
    """
    
    def __init__(self, parent, on_change=None):
        """
        Initialize the beam control panel.
        
        Args:
            parent: Parent Tkinter widget
            on_change: Callback function called when parameters change
        """
        super().__init__(parent)
        self._beam = ParticleBeam()
        self._on_change = on_change
        self._updating = False  # Prevent recursive updates
        
        self._build_ui()
        self._connect_signals()
        
    def _build_ui(self):
        """Build the user interface."""
        row = 0
        
        # === Enable checkbox with status indicator ===
        enable_frame = ttk.Frame(self)
        enable_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=8)
        row += 1
        
        self.enable_var = tk.BooleanVar(value=False)
        self.enable_check = ttk.Checkbutton(
            enable_frame,
            text="Enable Particle Beam",
            variable=self.enable_var,
            command=self._on_enable_toggle
        )
        self.enable_check.pack(side='left')
        
        self.status_label = ttk.Label(enable_frame, text="●", foreground="gray")
        self.status_label.pack(side='left', padx=10)
        
        self.status_text = ttk.Label(enable_frame, text="Disabled")
        self.status_text.pack(side='left')
        
        # === Beam Type ===
        type_frame = ttk.LabelFrame(self, text="Beam Type", padding=8)
        type_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row += 1
        
        self.beam_type_var = tk.StringVar(value="electron")
        ttk.Radiobutton(type_frame, text="Electron", variable=self.beam_type_var,
                        value="electron", command=self._on_param_change).pack(anchor='w')
        ttk.Radiobutton(type_frame, text="Ion", variable=self.beam_type_var,
                        value="ion", command=self._on_param_change).pack(anchor='w')
        
        # === Beam Parameters ===
        params_frame = ttk.LabelFrame(self, text="Beam Parameters", padding=8)
        params_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row += 1
        
        param_row = 0
        
        # Current
        ttk.Label(params_frame, text="Current (nA):").grid(row=param_row, column=0, sticky="w", pady=2)
        self.current_var = tk.DoubleVar(value=1.0)
        ttk.Entry(params_frame, textvariable=self.current_var, width=12).grid(row=param_row, column=1, pady=2)
        param_row += 1
        
        # Energy
        ttk.Label(params_frame, text="Energy (eV):").grid(row=param_row, column=0, sticky="w", pady=2)
        self.energy_var = tk.DoubleVar(value=100.0)
        ttk.Entry(params_frame, textvariable=self.energy_var, width=12).grid(row=param_row, column=1, pady=2)
        param_row += 1
        
        # Beam radius
        ttk.Label(params_frame, text="Beam Radius (mm):").grid(row=param_row, column=0, sticky="w", pady=2)
        self.radius_var = tk.DoubleVar(value=1.0)
        ttk.Entry(params_frame, textvariable=self.radius_var, width=12).grid(row=param_row, column=1, pady=2)
        param_row += 1
        
        # Propagation axis
        ttk.Label(params_frame, text="Propagation Axis:").grid(row=param_row, column=0, sticky="w", pady=2)
        self.axis_var = tk.StringVar(value="z")
        axis_combo = ttk.Combobox(params_frame, textvariable=self.axis_var,
                                   values=['x', 'y', 'z'], state='readonly', width=10)
        axis_combo.grid(row=param_row, column=1, pady=2)
        axis_combo.bind('<<ComboboxSelected>>', lambda e: self._on_param_change())
        param_row += 1
        
        # Beam center
        ttk.Label(params_frame, text="Beam Center (mm):").grid(row=param_row, column=0, sticky="w", pady=2)
        center_frame = ttk.Frame(params_frame)
        center_frame.grid(row=param_row, column=1, pady=2)
        self.center_x_var = tk.DoubleVar(value=0.0)
        self.center_y_var = tk.DoubleVar(value=0.0)
        ttk.Entry(center_frame, textvariable=self.center_x_var, width=5).pack(side='left', padx=1)
        ttk.Label(center_frame, text=",").pack(side='left')
        ttk.Entry(center_frame, textvariable=self.center_y_var, width=5).pack(side='left', padx=1)
        param_row += 1
        
        # Interaction strength
        ttk.Label(params_frame, text="Interaction Strength:").grid(row=param_row, column=0, sticky="w", pady=2)
        self.strength_var = tk.DoubleVar(value=1.0)
        ttk.Entry(params_frame, textvariable=self.strength_var, width=12).grid(row=param_row, column=1, pady=2)
        
        # === Phase Locking ===
        phase_frame = ttk.LabelFrame(self, text="Phase Locking (RF Sync)", padding=8)
        phase_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row += 1
        
        phase_row = 0
        
        # Phase mode
        ttk.Label(phase_frame, text="Phase Mode:").grid(row=phase_row, column=0, sticky="w", pady=2)
        self.phase_mode_var = tk.StringVar(value="zero_crossing_both")
        phase_combo = ttk.Combobox(
            phase_frame, 
            textvariable=self.phase_mode_var,
            values=[
                "zero_crossing_both",
                "zero_crossing_positive",
                "zero_crossing_negative",
                "continuous",
                "custom"
            ],
            state='readonly',
            width=20
        )
        phase_combo.grid(row=phase_row, column=1, pady=2)
        phase_combo.bind('<<ComboboxSelected>>', self._on_phase_mode_change)
        phase_row += 1
        
        # Phase window
        ttk.Label(phase_frame, text="Phase Window (%):").grid(row=phase_row, column=0, sticky="w", pady=2)
        self.phase_window_var = tk.DoubleVar(value=10.0)
        ttk.Entry(phase_frame, textvariable=self.phase_window_var, width=12).grid(row=phase_row, column=1, pady=2)
        phase_row += 1
        
        # Custom phase range
        ttk.Label(phase_frame, text="Custom Start (°):").grid(row=phase_row, column=0, sticky="w", pady=2)
        self.custom_start_var = tk.DoubleVar(value=0.0)
        self.custom_start_entry = ttk.Entry(phase_frame, textvariable=self.custom_start_var, width=12, state='disabled')
        self.custom_start_entry.grid(row=phase_row, column=1, pady=2)
        phase_row += 1
        
        ttk.Label(phase_frame, text="Custom End (°):").grid(row=phase_row, column=0, sticky="w", pady=2)
        self.custom_end_var = tk.DoubleVar(value=18.0)
        self.custom_end_entry = ttk.Entry(phase_frame, textvariable=self.custom_end_var, width=12, state='disabled')
        self.custom_end_entry.grid(row=phase_row, column=1, pady=2)
        
        # === Timing ===
        timing_frame = ttk.LabelFrame(self, text="Timing", padding=8)
        timing_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row += 1
        
        timing_row = 0
        
        # Delay start
        ttk.Label(timing_frame, text="Delay Start (ms):").grid(row=timing_row, column=0, sticky="w", pady=2)
        self.delay_var = tk.DoubleVar(value=0.0)
        ttk.Entry(timing_frame, textvariable=self.delay_var, width=12).grid(row=timing_row, column=1, pady=2)
        timing_row += 1
        
        # Pulse duration
        ttk.Label(timing_frame, text="Pulse Duration (ms):").grid(row=timing_row, column=0, sticky="w", pady=2)
        self.duration_var = tk.DoubleVar(value=1000.0)
        self.duration_entry = ttk.Entry(timing_frame, textvariable=self.duration_var, width=12)
        self.duration_entry.grid(row=timing_row, column=1, pady=2)
        timing_row += 1
        
        # Infinite duration checkbox
        self.infinite_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            timing_frame,
            text="Infinite Duration",
            variable=self.infinite_var,
            command=self._on_infinite_toggle
        ).grid(row=timing_row, column=0, columnspan=2, sticky="w", pady=2)
        
        # === Status Display ===
        status_frame = ttk.LabelFrame(self, text="Runtime Status", padding=8)
        status_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row += 1
        
        self.runtime_status_var = tk.StringVar(value="Simulation not running")
        ttk.Label(status_frame, textvariable=self.runtime_status_var).pack(anchor='w')
        
        self.rf_phase_var = tk.StringVar(value="RF Phase: --")
        ttk.Label(status_frame, textvariable=self.rf_phase_var).pack(anchor='w')
        
        self.delay_remaining_var = tk.StringVar(value="Delay: --")
        ttk.Label(status_frame, textvariable=self.delay_remaining_var).pack(anchor='w')
        
        # Initialize state
        self._on_infinite_toggle()
        
    def _connect_signals(self):
        """Connect variable traces for auto-update."""
        for var in [self.current_var, self.energy_var, self.radius_var,
                    self.center_x_var, self.center_y_var, self.strength_var,
                    self.phase_window_var, self.custom_start_var, self.custom_end_var,
                    self.delay_var, self.duration_var]:
            var.trace_add('write', lambda *args: self._on_param_change())
            
    def _on_enable_toggle(self):
        """Handle beam enable/disable."""
        enabled = self.enable_var.get()
        if enabled:
            self.status_label.configure(foreground="green")
            self.status_text.configure(text="Enabled")
        else:
            self.status_label.configure(foreground="gray")
            self.status_text.configure(text="Disabled")
        self._on_param_change()
        
    def _on_phase_mode_change(self, event=None):
        """Handle phase mode change."""
        mode = self.phase_mode_var.get()
        custom_enabled = (mode == "custom")
        state = 'normal' if custom_enabled else 'disabled'
        self.custom_start_entry.configure(state=state)
        self.custom_end_entry.configure(state=state)
        self._on_param_change()
        
    def _on_infinite_toggle(self):
        """Handle infinite duration checkbox."""
        if self.infinite_var.get():
            self.duration_entry.configure(state='disabled')
        else:
            self.duration_entry.configure(state='normal')
        self._on_param_change()
        
    def _on_param_change(self):
        """Handle any parameter change."""
        if self._updating:
            return
        self._updating = True
        try:
            params = self.get_parameters()
            self._beam.params = params
            if self._on_change:
                self._on_change(params)
        finally:
            self._updating = False
            
    def get_parameters(self) -> BeamParameters:
        """Get current beam parameters from UI."""
        try:
            duration = np.inf if self.infinite_var.get() else self.duration_var.get() * 1e-3
            
            # Map phase mode string to enum
            phase_mode_map = {
                "zero_crossing_both": PhaseMode.ZERO_CROSSING_BOTH,
                "zero_crossing_positive": PhaseMode.ZERO_CROSSING_POSITIVE,
                "zero_crossing_negative": PhaseMode.ZERO_CROSSING_NEGATIVE,
                "continuous": PhaseMode.CONTINUOUS,
                "custom": PhaseMode.CUSTOM,
            }
            
            return BeamParameters(
                beam_type=BeamType.ELECTRON if self.beam_type_var.get() == "electron" else BeamType.ION,
                current=self.current_var.get() * 1e-9,  # nA to A
                energy=self.energy_var.get(),
                propagation_axis=self.axis_var.get(),
                beam_center=(self.center_x_var.get() * 1e-3, 
                            self.center_y_var.get() * 1e-3),  # mm to m
                beam_radius=self.radius_var.get() * 1e-3,  # mm to m
                phase_mode=phase_mode_map.get(self.phase_mode_var.get(), PhaseMode.ZERO_CROSSING_BOTH),
                phase_window=self.phase_window_var.get() / 100.0,  # % to fraction
                custom_phase_start=np.radians(self.custom_start_var.get()),
                custom_phase_end=np.radians(self.custom_end_var.get()),
                delay_start=self.delay_var.get() * 1e-3,  # ms to s
                pulse_duration=duration,
                interaction_strength=self.strength_var.get(),
                enabled=self.enable_var.get()
            )
        except (tk.TclError, ValueError):
            # Return current params if there's a parse error
            return self._beam.params
    
    def set_parameters(self, params: BeamParameters):
        """Set UI from beam parameters."""
        self._updating = True
        try:
            self.enable_var.set(params.enabled)
            self.beam_type_var.set(params.beam_type.value)
            self.current_var.set(params.current * 1e9)  # A to nA
            self.energy_var.set(params.energy)
            self.radius_var.set(params.beam_radius * 1e3)  # m to mm
            self.axis_var.set(params.propagation_axis)
            self.center_x_var.set(params.beam_center[0] * 1e3)
            self.center_y_var.set(params.beam_center[1] * 1e3)
            self.strength_var.set(params.interaction_strength)
            self.phase_mode_var.set(params.phase_mode.value)
            self.phase_window_var.set(params.phase_window * 100)
            self.custom_start_var.set(np.degrees(params.custom_phase_start))
            self.custom_end_var.set(np.degrees(params.custom_phase_end))
            self.delay_var.set(params.delay_start * 1e3)  # s to ms
            
            if np.isinf(params.pulse_duration):
                self.infinite_var.set(True)
            else:
                self.infinite_var.set(False)
                self.duration_var.set(params.pulse_duration * 1e3)
                
            self._on_enable_toggle()
            self._on_phase_mode_change()
            self._on_infinite_toggle()
        finally:
            self._updating = False
            self._beam.params = params
            
    def update_status(self, t: float, rf_phase: float = None):
        """
        Update the runtime status display.
        
        Args:
            t: Current simulation time in seconds
            rf_phase: Current RF phase in radians (optional)
        """
        params = self._beam.params
        
        if not params.enabled:
            self.runtime_status_var.set("Beam: Disabled")
            self.status_label.configure(foreground="gray")
        elif t < params.delay_start:
            self.runtime_status_var.set("Beam: Waiting (Delay)")
            self.status_label.configure(foreground="orange")
        elif self._beam.is_beam_active(t):
            self.runtime_status_var.set("Beam: ACTIVE")
            self.status_label.configure(foreground="green")
        else:
            self.runtime_status_var.set("Beam: Gated (RF Phase)")
            self.status_label.configure(foreground="yellow")
            
        # Update phase display
        if rf_phase is not None:
            self.rf_phase_var.set(f"RF Phase: {np.degrees(rf_phase):.1f}°")
        else:
            self.rf_phase_var.set("RF Phase: --")
            
        # Update delay display
        delay_remaining = max(0, params.delay_start - t)
        if delay_remaining > 0:
            self.delay_remaining_var.set(f"Delay Remaining: {delay_remaining*1e3:.2f} ms")
        else:
            self.delay_remaining_var.set("Delay: Complete")
            
    def get_beam(self) -> ParticleBeam:
        """Get the ParticleBeam instance."""
        return self._beam
    
    def set_rf_parameters(self, omega: float = None, frequency: float = None,
                          phase_offset: float = 0.0):
        """Set RF parameters for phase locking."""
        self._beam.set_rf_parameters(omega=omega, frequency=frequency, 
                                      phase_offset=phase_offset)


# Demo application
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Beam Control Demo")
    
    def on_change(params):
        print(f"Parameters changed: enabled={params.enabled}, "
              f"current={params.current*1e9:.1f}nA, "
              f"phase_mode={params.phase_mode.value}")
    
    panel = BeamControlPanel(root, on_change=on_change)
    panel.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Set RF frequency
    panel.set_rf_parameters(frequency=1e6)
    
    root.mainloop()