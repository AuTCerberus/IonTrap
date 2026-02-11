"""
Parameter Sweep GUI Panel

Provides a Tkinter interface for configuring and running parameter sweeps.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from pathlib import Path
import threading
import json
from typing import Callable, Optional, Dict, Any
from queue import Queue, Empty


class SweepConfigPanel(ttk.Frame):
    """
    Panel for configuring parameter sweeps.
    """
    
    # Available parameters for sweeping
    SWEEP_PARAMETERS = {
        'V_rf': {'label': 'RF Voltage (V)', 'default_min': 50, 'default_max': 300, 'default_n': 20},
        'V_dc': {'label': 'DC Voltage (V)', 'default_min': 0, 'default_max': 50, 'default_n': 20},
        'frequency': {'label': 'RF Frequency (Hz)', 'default_min': 1e5, 'default_max': 1e7, 'default_n': 20, 'log': True},
        'pressure_torr': {'label': 'Pressure (Torr)', 'default_min': 1e-9, 'default_max': 1e-2, 'default_n': 20, 'log': True},
        'damping_gamma': {'label': 'Damping γ (1/s)', 'default_min': 0, 'default_max': 1000, 'default_n': 20},
        'n_charges': {'label': 'Number of Charges', 'default_min': 10, 'default_max': 1000, 'default_n': 20},
        'particle_radius': {'label': 'Particle Radius (m)', 'default_min': 1e-7, 'default_max': 1e-4, 'default_n': 20, 'log': True},
    }
    
    def __init__(self, parent, 
                 on_run: Callable = None,
                 on_stop: Callable = None,
                 get_base_params: Callable = None):
        """
        Initialize the sweep config panel.
        
        Args:
            parent: Parent widget
            on_run: Callback when Run is clicked, receives (config_dict)
            on_stop: Callback when Stop is clicked
            get_base_params: Callback to get base trap parameters
        """
        super().__init__(parent)
        
        self._on_run = on_run
        self._on_stop = on_stop
        self._get_base_params = get_base_params
        
        self._build_ui()
        self._on_sweep_type_change()
    
    def _build_ui(self):
        """Build the user interface."""
        row = 0
        
        # === Sweep Type ===
        type_frame = ttk.LabelFrame(self, text="Sweep Type", padding=8)
        type_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row += 1
        
        self.sweep_type_var = tk.StringVar(value="1d")
        ttk.Radiobutton(type_frame, text="1D Sweep (single parameter)", 
                        variable=self.sweep_type_var, value="1d",
                        command=self._on_sweep_type_change).pack(anchor='w')
        ttk.Radiobutton(type_frame, text="2D Sweep (stability diagram)",
                        variable=self.sweep_type_var, value="2d",
                        command=self._on_sweep_type_change).pack(anchor='w')
        
        # === Parameter 1 ===
        self.param1_frame = ttk.LabelFrame(self, text="Parameter 1", padding=8)
        self.param1_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row += 1
        
        self._build_param_controls(self.param1_frame, prefix="p1")
        
        # === Parameter 2 (for 2D sweeps) ===
        self.param2_frame = ttk.LabelFrame(self, text="Parameter 2", padding=8)
        self.param2_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row += 1
        
        self._build_param_controls(self.param2_frame, prefix="p2")
        
        # === Simulation Settings ===
        sim_frame = ttk.LabelFrame(self, text="Simulation Settings", padding=8)
        sim_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row += 1
        
        sim_row = 0
        
        ttk.Label(sim_frame, text="Duration (s):").grid(row=sim_row, column=0, sticky='w', pady=2)
        self.duration_var = tk.DoubleVar(value=1e-3)
        ttk.Entry(sim_frame, textvariable=self.duration_var, width=12).grid(row=sim_row, column=1, pady=2)
        sim_row += 1
        
        ttk.Label(sim_frame, text="Escape Radius (m):").grid(row=sim_row, column=0, sticky='w', pady=2)
        self.escape_radius_var = tk.DoubleVar(value=1e-2)
        ttk.Entry(sim_frame, textvariable=self.escape_radius_var, width=12).grid(row=sim_row, column=1, pady=2)
        sim_row += 1
        
        self.use_secular_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sim_frame, text="Use secular approximation (faster)",
                        variable=self.use_secular_var).grid(row=sim_row, column=0, columnspan=2, sticky='w', pady=2)
        sim_row += 1
        
        self.randomize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sim_frame, text="Randomize initial conditions",
                        variable=self.randomize_var).grid(row=sim_row, column=0, columnspan=2, sticky='w', pady=2)
        
        # === Initial Position ===
        init_frame = ttk.LabelFrame(self, text="Initial Position (m)", padding=8)
        init_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row += 1
        
        init_row = ttk.Frame(init_frame)
        init_row.pack(fill='x')
        
        self.init_x_var = tk.DoubleVar(value=1e-4)
        self.init_y_var = tk.DoubleVar(value=1e-4)
        self.init_z_var = tk.DoubleVar(value=1e-4)
        
        ttk.Label(init_row, text="x:").pack(side='left')
        ttk.Entry(init_row, textvariable=self.init_x_var, width=8).pack(side='left', padx=2)
        ttk.Label(init_row, text="y:").pack(side='left')
        ttk.Entry(init_row, textvariable=self.init_y_var, width=8).pack(side='left', padx=2)
        ttk.Label(init_row, text="z:").pack(side='left')
        ttk.Entry(init_row, textvariable=self.init_z_var, width=8).pack(side='left', padx=2)
        
        # === Output Settings ===
        output_frame = ttk.LabelFrame(self, text="Output", padding=8)
        output_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row += 1
        
        ttk.Label(output_frame, text="Output Directory:").pack(anchor='w')
        dir_row = ttk.Frame(output_frame)
        dir_row.pack(fill='x', pady=2)
        
        self.output_dir_var = tk.StringVar(value="sweep_results")
        ttk.Entry(dir_row, textvariable=self.output_dir_var, width=25).pack(side='left', fill='x', expand=True)
        ttk.Button(dir_row, text="Browse", command=self._browse_output, width=8).pack(side='left', padx=4)
        
        # === Progress ===
        progress_frame = ttk.LabelFrame(self, text="Progress", padding=8)
        progress_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row += 1
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                             maximum=100, length=200)
        self.progress_bar.pack(fill='x', pady=2)
        
        self.progress_label_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.progress_label_var).pack(anchor='w')
        
        # === Buttons ===
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=8, pady=8)
        row += 1
        
        self.run_button = ttk.Button(btn_frame, text="Run Sweep", command=self._on_run_click)
        self.run_button.pack(side='left', padx=4)
        
        self.stop_button = ttk.Button(btn_frame, text="Stop", command=self._on_stop_click, state='disabled')
        self.stop_button.pack(side='left', padx=4)
        
        ttk.Button(btn_frame, text="Load Results", command=self._load_results).pack(side='right', padx=4)
    
    def _build_param_controls(self, frame: ttk.Frame, prefix: str):
        """Build controls for a parameter."""
        param_row = 0
        
        # Parameter selection
        ttk.Label(frame, text="Parameter:").grid(row=param_row, column=0, sticky='w', pady=2)
        param_var = tk.StringVar(value=list(self.SWEEP_PARAMETERS.keys())[0])
        param_combo = ttk.Combobox(frame, textvariable=param_var,
                                    values=list(self.SWEEP_PARAMETERS.keys()),
                                    state='readonly', width=18)
        param_combo.grid(row=param_row, column=1, pady=2)
        param_combo.bind('<<ComboboxSelected>>', 
                         lambda e, p=prefix: self._on_param_select(p))
        param_row += 1
        
        # Min value
        ttk.Label(frame, text="Min:").grid(row=param_row, column=0, sticky='w', pady=2)
        min_var = tk.DoubleVar(value=50)
        ttk.Entry(frame, textvariable=min_var, width=12).grid(row=param_row, column=1, pady=2)
        param_row += 1
        
        # Max value
        ttk.Label(frame, text="Max:").grid(row=param_row, column=0, sticky='w', pady=2)
        max_var = tk.DoubleVar(value=300)
        ttk.Entry(frame, textvariable=max_var, width=12).grid(row=param_row, column=1, pady=2)
        param_row += 1
        
        # Number of points
        ttk.Label(frame, text="Points:").grid(row=param_row, column=0, sticky='w', pady=2)
        n_var = tk.IntVar(value=20)
        ttk.Entry(frame, textvariable=n_var, width=12).grid(row=param_row, column=1, pady=2)
        param_row += 1
        
        # Log scale
        log_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Logarithmic scale", 
                        variable=log_var).grid(row=param_row, column=0, columnspan=2, sticky='w', pady=2)
        
        # Store references
        setattr(self, f'{prefix}_param_var', param_var)
        setattr(self, f'{prefix}_min_var', min_var)
        setattr(self, f'{prefix}_max_var', max_var)
        setattr(self, f'{prefix}_n_var', n_var)
        setattr(self, f'{prefix}_log_var', log_var)
    
    def _on_param_select(self, prefix: str):
        """Handle parameter selection change."""
        param_var = getattr(self, f'{prefix}_param_var')
        param_name = param_var.get()
        
        if param_name in self.SWEEP_PARAMETERS:
            info = self.SWEEP_PARAMETERS[param_name]
            getattr(self, f'{prefix}_min_var').set(info['default_min'])
            getattr(self, f'{prefix}_max_var').set(info['default_max'])
            getattr(self, f'{prefix}_n_var').set(info['default_n'])
            getattr(self, f'{prefix}_log_var').set(info.get('log', False))
    
    def _on_sweep_type_change(self):
        """Handle sweep type change."""
        is_2d = self.sweep_type_var.get() == "2d"
        
        # Show/hide parameter 2 frame
        if is_2d:
            self.param2_frame.grid()
            # Set default to V_dc for 2D (V_rf vs V_dc)
            self.p1_param_var.set('V_rf')
            self.p2_param_var.set('V_dc')
            self._on_param_select('p1')
            self._on_param_select('p2')
        else:
            self.param2_frame.grid_remove()
    
    def _browse_output(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(initialdir=self.output_dir_var.get())
        if directory:
            self.output_dir_var.set(directory)
    
    def _on_run_click(self):
        """Handle Run button click."""
        if self._on_run:
            config = self.get_config()
            self._on_run(config)
    
    def _on_stop_click(self):
        """Handle Stop button click."""
        if self._on_stop:
            self._on_stop()
    
    def _load_results(self):
        """Load and display previous results."""
        filepath = filedialog.askopenfilename(
            initialdir=self.output_dir_var.get(),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    results = json.load(f)
                self._show_results(results)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load results:\n{e}")
    
    def _show_results(self, results: dict):
        """Show loaded results (placeholder - integrate with visualization)."""
        # This would open a results viewer window
        messagebox.showinfo("Results Loaded", 
                            f"Loaded {results.get('type', 'unknown')} sweep\n"
                            f"Total runs: {results.get('summary', {}).get('total_runs', 'N/A')}\n"
                            f"Stable: {results.get('summary', {}).get('stable_fraction', 0)*100:.1f}%")
    
    def get_config(self) -> dict:
        """Get the current configuration as a dictionary."""
        is_2d = self.sweep_type_var.get() == "2d"
        
        config = {
            'type': '2d' if is_2d else '1d',
            'param1': {
                'name': self.p1_param_var.get(),
                'min_val': self.p1_min_var.get(),
                'max_val': self.p1_max_var.get(),
                'n_points': self.p1_n_var.get(),
                'log_scale': self.p1_log_var.get(),
            },
            'duration': self.duration_var.get(),
            'escape_radius': self.escape_radius_var.get(),
            'use_secular': self.use_secular_var.get(),
            'randomize_initial': self.randomize_var.get(),
            'initial_position': (self.init_x_var.get(), self.init_y_var.get(), self.init_z_var.get()),
            'output_dir': self.output_dir_var.get(),
        }
        
        if is_2d:
            config['param2'] = {
                'name': self.p2_param_var.get(),
                'min_val': self.p2_min_var.get(),
                'max_val': self.p2_max_var.get(),
                'n_points': self.p2_n_var.get(),
                'log_scale': self.p2_log_var.get(),
            }
        
        return config
    
    def set_running(self, running: bool):
        """Set the running state (updates button states)."""
        if running:
            self.run_button.configure(state='disabled')
            self.stop_button.configure(state='normal')
        else:
            self.run_button.configure(state='normal')
            self.stop_button.configure(state='disabled')
            self.progress_var.set(0)
            self.progress_label_var.set("Ready")
    
    def update_progress(self, current: int, total: int, message: str = ""):
        """Update the progress display."""
        if total > 0:
            pct = 100 * current / total
            self.progress_var.set(pct)
        self.progress_label_var.set(f"{current}/{total}: {message}" if message else f"{current}/{total}")


class SweepResultsViewer(tk.Toplevel):
    """
    Window for viewing sweep results.
    """
    
    def __init__(self, parent, results: dict):
        super().__init__(parent)
        self.title("Sweep Results")
        self.geometry("800x600")
        
        self.results = results
        self._build_ui()
    
    def _build_ui(self):
        """Build the results viewer UI."""
        # Matplotlib canvas for plots
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            from matplotlib.figure import Figure
            import sweep_visualization as viz
            
            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            
            if self.results['type'] == '1d':
                viz.plot_1d_sweep(self.results, ax=ax)
            else:
                viz.plot_stability_diagram(self.results, ax=ax)
            
            canvas = FigureCanvasTkAgg(fig, master=self)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
            toolbar = NavigationToolbar2Tk(canvas, self)
            toolbar.update()
            
        except ImportError as e:
            ttk.Label(self, text=f"Visualization not available:\n{e}").pack(pady=20)
        
        # Summary text
        summary_frame = ttk.LabelFrame(self, text="Summary", padding=8)
        summary_frame.pack(fill='x', padx=8, pady=8)
        
        if 'summary' in self.results:
            summary = self.results['summary']
            ttk.Label(summary_frame, 
                      text=f"Total: {summary.get('total_runs', 'N/A')} | "
                           f"Stable: {summary.get('stable_count', 'N/A')} "
                           f"({summary.get('stable_fraction', 0)*100:.1f}%)").pack()


# Demo
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Parameter Sweep Demo")
    
    def on_run(config):
        print(f"Running sweep with config: {config}")
    
    def on_stop():
        print("Stop requested")
    
    panel = SweepConfigPanel(root, on_run=on_run, on_stop=on_stop)
    panel.pack(fill='both', expand=True, padx=10, pady=10)
    
    root.mainloop()
