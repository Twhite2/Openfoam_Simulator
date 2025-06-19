#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation controls for Openfoam_Simulator application.

This module implements the simulation control panel that allows users to:
- Configure OpenFOAM simulation settings
- Manage simulation execution
- Monitor simulation progress
- Analyze simulation results
- Set up special oil & gas industry simulations (pigging, spill, etc.)
"""

import os
import sys
import time
import threading
import subprocess
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import math

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, 
    QCheckBox, QPushButton, QTabWidget, QToolButton, QFrame,
    QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, 
    QStyledItemDelegate, QProgressBar, QTextEdit, QToolBar,
    QAction, QSizePolicy, QFileDialog, QRadioButton, QButtonGroup,
    QFormLayout, QGridLayout, QStackedWidget, QMessageBox, QListWidget,
    QInputDialog, QListWidgetItem, QMenu, QDialog, QDialogButtonBox,
    QApplication
)
from PyQt5.QtCore import (
    Qt, QSize, QTimer, QProcess, QProcessEnvironment, 
    pyqtSignal, QThread, QMutex, QEventLoop, QDir, QRect
)
from PyQt5.QtGui import (
    QIcon, QColor, QPixmap, QFont, QTextCursor, QPalette,
    QRegExpValidator, QDoubleValidator, QIntValidator, QPen, QPainterPath,
    QPainter
)

# Import utility modules
from ..utils.logger import get_logger
from ..config import get_value, set_value

# Import VTK modules (replacing ParaView)





logger = get_logger(__name__)


class SimulationRunner(QThread):
    """
    Thread for running OpenFOAM simulations.
    
    This class handles the execution of OpenFOAM solvers in a separate thread,
    allowing the UI to remain responsive during long simulations.
    """
    
    # Signals
    progress_update = pyqtSignal(int, float, dict)  # iteration, time, residuals
    status_update = pyqtSignal(str)  # status message
    simulation_finished = pyqtSignal(bool)  # success flag
    log_update = pyqtSignal(str)  # log message
    simulation_memory_exceeded = pyqtSignal()  # Add this new signal
    
    def __init__(self, solver: str, case_dir: str, 
                 parallel: bool = False, processors: int = 4,
                 parent=None):
        """
        Initialize the simulation runner.
        
        Args:
            solver (str): Name of the OpenFOAM solver to use
            case_dir (str): Path to the case directory
            parallel (bool): Whether to run in parallel
            processors (int): Number of processors to use in parallel mode
            parent: Parent object
        """
        super(SimulationRunner, self).__init__(parent)
        
        self.solver = solver
        self.case_dir = case_dir
        self.parallel = parallel
        self.processors = processors
        
        self.process = None
        self.stop_requested = False
        self.mutex = QMutex()
        
        # Track progress
        self.current_iteration = 0
        self.current_time = 0.0
        self.max_iterations = 1000  # Default
        self.residuals = {}
        
        # Add memory protection
        self.max_log_size = 1000000  # Maximum log size to prevent memory issues
        self.last_log = ""
        
        # Add safety flags
        self.is_parsing = False
        self.error_count = 0
        self.max_errors = 10  # Maximum consecutive errors before aborting
    
    def run(self):
        """Run the simulation thread."""
        try:
            # Validate the case before running
            if not self._validate_case_setup():
                self.status_update.emit("Simulation setup has issues. Check log for details.")
                self.simulation_finished.emit(False)
                return
            
            # CRITICAL: Last-minute check for allBoundary in field files
            self._ensure_all_boundary_exists()
            
            # Log start of simulation
            logger.info(f"Starting simulation with solver: {self.solver}")
            self.status_update.emit(f"Starting simulation with solver: {self.solver}")
            
            # Prepare command
            cmd = []
            
            # Check if we're running in parallel
            if self.parallel:
                cmd = ["mpirun", "-np", str(self.processors), self.solver, "-parallel", "-case", self.case_dir]
            else:
                cmd = [self.solver, "-case", self.case_dir]
            
            # Logging
            self.status_update.emit(f"Starting simulation with solver: {self.solver}")
            self.log_update.emit(f"Command: {' '.join(cmd)}")
            
            # Set up environment
            env = QProcessEnvironment.systemEnvironment()
            
            # Check for OpenFOAM environment
            foam_dir = os.environ.get('WM_PROJECT_DIR')
            if foam_dir:
                # OpenFOAM environment is already set up
                self.log_update.emit(f"Using OpenFOAM environment: {foam_dir}")
            else:
                # Try to source OpenFOAM environment
                source_cmd = f"source /opt/openfoam*/etc/bashrc"
                self.log_update.emit(f"Attempting to source OpenFOAM: {source_cmd}")
                subprocess.call(['bash', '-c', source_cmd])
            
            # Create process with timeout protection
            self.process = QProcess()
            self.process.setProcessEnvironment(env)
            self.process.setWorkingDirectory(self.case_dir)
            
            # Connect signals with error handling
            self.process.readyReadStandardOutput.connect(self._safe_read_stdout)
            self.process.readyReadStandardError.connect(self._safe_read_stderr)
            
            # Start the process
            self.process.start(" ".join(cmd))
            
            # Give process time to start
            if not self.process.waitForStarted(5000):
                self.status_update.emit(f"Error: Process failed to start - {self.process.errorString()}")
                self.simulation_finished.emit(False)
                return
                
            # Add periodic memory check
            memory_check_timer = QTimer()
            memory_check_timer.timeout.connect(self._check_memory_usage)
            memory_check_timer.start(10000)  # Check every 10 seconds
            
            # Wait for process to complete or be terminated
            loop = QEventLoop()
            self.process.finished.connect(loop.quit)
            
            # Add timeout to prevent hanging
            timer = QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(loop.quit)
            timer.start(3600000)  # 1 hour max timeout
            
            loop.exec_()
            
            # Clean up timers
            timer.stop()
            memory_check_timer.stop()
            
            # Check if we timed out
            if timer.isActive():
                timer.stop()
            else:
                # Timeout occurred
                self.status_update.emit("Simulation timed out after 1 hour")
                self.stop()
                self.simulation_finished.emit(False)
                return
            
            # Check result
            if self.process.exitCode() == 0 and not self.stop_requested:
                self.status_update.emit("Simulation completed successfully")
                self.simulation_finished.emit(True)

        # Unregister simulation with lock manager

        # Unregister simulation with lock manager

        except Exception as e:
            import traceback
            error_msg = f"Error running simulation: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.status_update.emit(f"Error: {str(e)}")
            self.log_update.emit(error_msg)
            self.simulation_finished.emit(False)
    
    def _safe_read_stdout(self):
        """Read stdout with error protection."""
        try:
            self._read_stdout()
        except Exception as e:
            logger.error(f"Error reading stdout: {e}")
            self.error_count += 1
            if self.error_count > self.max_errors:
                self.status_update.emit("Too many errors reading simulation output. Stopping.")
                self.stop()
    
    def _safe_read_stderr(self):
        """Read stderr with error protection."""
        try:
            self._read_stderr()
        except Exception as e:
            logger.error(f"Error reading stderr: {e}")
            self.error_count += 1
            if self.error_count > self.max_errors:
                self.status_update.emit("Too many errors reading simulation output. Stopping.")
                self.stop()
    
    def _read_stdout(self):
        """Read standard output from the process."""
        if not self.process:
            return
        
        # Read all available output
        output = bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace')
        
        # IMPORTANT: Also print to terminal for debugging
        print(f"[OpenFOAM stdout] {output}", flush=True)
        
        # Update log (with size limitation)
        if len(self.last_log) + len(output) > self.max_log_size:
            # Truncate log if it gets too large
            self.last_log = self.last_log[-(self.max_log_size // 2):] + output
            self.log_update.emit("... log truncated due to size ...\n" + self.last_log)
        else:
            self.last_log += output
            self.log_update.emit(output)
        
        # Parse output for progress information (with mutex protection)
        self.mutex.lock()
        try:
            if not self.is_parsing:
                self.is_parsing = True
                self._parse_output(output)
                self.is_parsing = False
                self.error_count = 0  # Reset error count on success
        finally:
            self.mutex.unlock()
    
    def _parse_output(self, output: str):
        """Parse solver output with enhanced error handling and stall detection."""
        try:
            # Parse iteration and time
            time_match = re.search(r"Time = (\d+\.?\d*)", output)
            if time_match:
                try:
                    self.current_time = float(time_match.group(1))
                except ValueError:
                    logger.warning(f"Invalid time value: {time_match.group(1)}")
            
            # Parse iteration for steady-state solvers
            iter_match = re.search(r"Iteration (\d+)", output)
            if iter_match:
                try:
                    self.current_iteration = int(iter_match.group(1))
                except ValueError:
                    logger.warning(f"Invalid iteration value: {iter_match.group(1)}")
            
            # Parse residuals (pattern varies by solver)
            residual_matches = re.findall(r"Solving for ([^,]+), Initial residual = ([^,]+), Final residual = ([^,]+), No Iterations (\d+)", output)
            if residual_matches:
                for field, initial, final, iterations in residual_matches:
                    field = field.strip()
                    try:
                        self.residuals[field] = float(final)
                    except ValueError:
                        logger.warning(f"Invalid residual value for {field}: {final}")
            
            # For interFoam and other multiphase solvers
            alpha_match = re.search(r"Phase-1 volume fraction = ([^,]+)", output)
            if alpha_match:
                try:
                    self.residuals["alpha.water"] = float(alpha_match.group(1))
                except ValueError:
                    logger.warning(f"Invalid alpha value: {alpha_match.group(1)}")
            
            # Support for oil and gas specific output patterns (based on your analysis)
            # Example: Match phase fractions
            phase_match = re.search(r"Oil phase fraction: ([^,]+), Water phase fraction: ([^,]+)", output)
            if phase_match:
                try:
                    self.residuals["oil.fraction"] = float(phase_match.group(1))
                    self.residuals["water.fraction"] = float(phase_match.group(2))
                except ValueError:
                    logger.warning(f"Invalid phase fraction values")
            
            # NEW: Check for stalled simulation with zero residuals
            self._check_for_stalled_simulation()
            
            # Emit progress update signal
            self.progress_update.emit(self.current_iteration, self.current_time, self.residuals)
        
        except Exception as e:
            import traceback
            logger.error(f"Error parsing output: {e}\n{traceback.format_exc()}")
            self.error_count += 1
    
    def _check_for_stalled_simulation(self):
        """Check if the simulation is making progress or if it's stalled with zero residuals."""
        # If we have several consecutive time steps with all zero residuals,
        # the simulation might be stalled or consuming resources unnecessarily
        zero_residual_count = getattr(self, '_zero_residual_count', 0)
        
        # Check if all current residuals are zero
        all_zeros = True
        for value in self.residuals.values():
            if abs(value) > 1e-10:  # Allow for floating point imprecision
                all_zeros = False
                break
        
        if all_zeros and self.residuals:
            zero_residual_count += 1
            self._zero_residual_count = zero_residual_count
            
            # Log this situation
            if zero_residual_count % 10 == 0:  # Log every 10 occurrences
                msg = f"Warning: {zero_residual_count} consecutive time steps with zero residuals"
                logger.warning(msg)
                self.status_update.emit(msg)
            
            # After too many zero steps, consider stopping
            if zero_residual_count > 50:
                msg = "Stopping simulation: 50 consecutive time steps with zero residuals"
                logger.warning(msg)
                self.status_update.emit(msg)
                self.stop()
                return True
        else:
            # Reset counter if we see non-zero residuals
            self._zero_residual_count = 0
        
        return False
    
    def stop(self):
        """Stop the simulation."""
        self.mutex.lock()
        self.stop_requested = True
        self.mutex.unlock()
        
        if self.process and self.process.state() != QProcess.NotRunning:
            self.process.terminate()
            
            # Give the process a chance to terminate gracefully
            if not self.process.waitForFinished(5000):
                self.process.kill()
    
    def _read_stderr(self):
        """Read standard error from the process."""
        if not self.process:
            return
        
        # Read all available output
        output = bytes(self.process.readAllStandardError()).decode('utf-8', errors='replace')
        
        # IMPORTANT: Also print to terminal for debugging
        print(f"[OpenFOAM stderr] {output}", flush=True)
        
        # Update log
        self.log_update.emit(output)
    
    def _parse_output(self, output: str):
        """
        Parse solver output to extract progress information.
        
        Args:
            output (str): Solver output text
        """
        # Parse iteration and time
        time_match = re.search(r"Time = (\d+\.?\d*)", output)
        if time_match:
            self.current_time = float(time_match.group(1))
        
        # Parse iteration for steady-state solvers
        iter_match = re.search(r"Iteration (\d+)", output)
        if iter_match:
            self.current_iteration = int(iter_match.group(1))
        
        # Parse residuals (pattern varies by solver)
        residual_matches = re.findall(r"Solving for ([^,]+), Initial residual = ([^,]+), Final residual = ([^,]+), No Iterations (\d+)", output)
        if residual_matches:
            for field, initial, final, iterations in residual_matches:
                field = field.strip()
                self.residuals[field] = float(final)
        
        # For interFoam and other multiphase solvers
        alpha_match = re.search(r"Phase-1 volume fraction = ([^,]+)", output)
        if alpha_match:
            self.residuals["alpha.water"] = float(alpha_match.group(1))
        
        # Emit progress update signal
        self.progress_update.emit(self.current_iteration, self.current_time, self.residuals)

    def _check_memory_usage(self):
        """Monitor memory usage of the OpenFOAM process."""
        if not self.process or not self.process.processId():
            return
        
        try:
            # Get process memory info (platform-specific)
            pid = self.process.processId()
            
            # For Linux
            if os.path.exists(f"/proc/{pid}/status"):
                with open(f"/proc/{pid}/status", 'r') as f:
                    for line in f:
                        if line.startswith('VmRSS:'):
                            # Memory in KB
                            memory_kb = int(line.split()[1])
                            memory_mb = memory_kb / 1024
                            
                            # Log high memory usage
                            if memory_mb > 1000:  # If using more than 1GB
                                msg = f"Warning: OpenFOAM using {memory_mb:.1f} MB of memory"
                                logger.warning(msg)
                                self.status_update.emit(msg)
                            
                            # Consider stopping if memory usage is extreme
                            if memory_mb > 4000:  # If using more than 4GB
                                msg = f"Critical: OpenFOAM using excessive memory ({memory_mb:.1f} MB). Stopping."
                                logger.error(msg)
                                self.status_update.emit(msg)
                                self.stop()
                            break
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")

    def _validate_case_setup(self):
        """Validate the OpenFOAM case setup to detect issues before running."""
        try:
            if not self.case_dir:
                self.log_update.emit("ERROR: No case directory specified. Cannot validate setup.")
                return False
                
            self.status_update.emit("Validating case setup...")
            
            # Find the OpenFOAM case directory
            openfoam_dir = self.case_dir
            if os.path.exists(os.path.join(self.case_dir, 'openfoam')):
                openfoam_dir = os.path.join(self.case_dir, 'openfoam')
            
            # Apply comprehensive fixes with explicit case directory
            try:
                from ..openfoam_integration.case_manager import CaseManager
                case_manager = CaseManager(case_dir=openfoam_dir)
                self.log_update.emit("Applying comprehensive boundary condition fixes...")
                if not case_manager.fix_all_boundary_conditions():
                    self.log_update.emit("WARNING: Some boundary condition fixes could not be applied")
            except Exception as e:
                self.log_update.emit(f"Warning: Could not apply automatic fixes: {e}")
            
            # Validate velocity field (U) for non-zero boundary conditions
            U_path = os.path.join(openfoam_dir, "0", "U")
            if os.path.exists(U_path):
                has_flow = self._check_velocity_field(U_path)
                if not has_flow:
                    self.log_update.emit("WARNING: No velocity driving force found. Simulation may result in zero residuals.")
                    return False
            
            # Check pressure field (p) for pressure gradients
            p_path = os.path.join(openfoam_dir, "0", "p")
            if os.path.exists(p_path):
                has_pressure_gradient = self._check_pressure_field(p_path)
                if not has_pressure_gradient:
                    self.log_update.emit("WARNING: No pressure gradient detected. This might result in zero residuals.")
                    # We've already tried to fix it with fix_all_boundary_conditions
            
            return True
        except Exception as e:
            self.log_update.emit(f"Error validating case setup: {e}")
            return False
    
    def _check_velocity_field(self, file_path):
        """Check velocity boundary conditions for trivial setup.
        Returns True if non-zero velocities are found."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check if internal field is all zeros
            internal_match = re.search(r"internalField\s+uniform\s+\((.*?)\)", content)
            if internal_match:
                values = internal_match.group(1).split()
                all_zeros = all(abs(float(v.strip())) < 1e-6 for v in values if v.strip())
                if not all_zeros:
                    return True  # Non-zero internal field is good
            
            # Check boundaries for non-zero velocities
            boundaries = re.findall(r"(inlet|inlets|.*)\s*\n\s*{[^}]*?type\s+fixedValue[^}]*?value\s+uniform\s+\((.*?)\)", content, re.DOTALL)
            for name, values in boundaries:
                values = values.split()
                if any(abs(float(v.strip())) > 1e-6 for v in values if v.strip()):
                    return True  # Found non-zero velocity boundary
            
            return False  # No non-zero velocities found
            
        except Exception as e:
            logger.warning(f"Error checking velocity field: {e}")
            return True  # Assume valid in case of error
    
    def _check_pressure_field(self, file_path):
        """Check pressure boundary conditions for pressure gradients.
        Returns True if pressure gradients are found."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract pressure values from fixed-value boundaries
            pressure_values = set()
            boundaries = re.findall(r"(inlet|outlet|inlets|outlets|.*)\s*\n\s*{[^}]*?type\s+fixedValue[^}]*?value\s+uniform\s+([\d.-]+)", content, re.DOTALL)
            for name, value in boundaries:
                pressure_values.add(float(value.strip()))
            
            # If we have more than one pressure value, we have a gradient
            return len(pressure_values) > 1
            
        except Exception as e:
            logger.warning(f"Error checking pressure field: {e}")
            return True  # Assume valid in case of error

    def _verify_boundary_conditions(self):
        """Verify that all mesh boundaries have corresponding field entries."""
        try:
            from ..openfoam_integration.case_manager import CaseManager
            
            if not self.case_dir:
                self.log_update.emit("ERROR: No case directory specified. Cannot verify boundary conditions.")
                return False
                
            self.status_update.emit("Verifying boundary conditions...")
            self.log_update.emit("Checking and fixing boundary conditions...")
            
            # Create temporary case manager with explicit case directory
            case_manager = CaseManager(case_dir=self.case_dir)
            
            # Apply comprehensive fix to boundary conditions
            if not case_manager.fix_all_boundary_conditions():
                self.log_update.emit("WARNING: Failed to completely fix boundary conditions")
            else:
                self.log_update.emit("Boundary conditions verified and fixed successfully")
                
            return True
        except Exception as e:
            logger.error(f"Error verifying boundary conditions: {e}")
            self.log_update.emit(f"WARNING: Error verifying boundary conditions: {e}")
            return False

    def _ensure_all_boundary_exists(self):
        """
        Final check to ensure allBoundary exists in all field files.
        This is a direct fix applied right before execution.
        """
        try:
            # Check key field files directly
            for field_name in ["p", "U"]:
                field_path = os.path.join(self.case_dir, "0", field_name)
                if os.path.exists(field_path):
                    self.log_update.emit(f"Checking and fixing {field_name} file for proper boundaryField syntax")
                    self._fix_field_file(field_path)
        except Exception as e:
            self.log_update.emit(f"WARNING: Error ensuring allBoundary exists: {e}")
            import traceback
            self.log_update.emit(traceback.format_exc())
    
    def _fix_field_file(self, file_path):
        """Fix the field file to ensure proper boundaryField syntax with allBoundary."""
        try:
            # Read the file
            with open(file_path, 'r') as f:
                content = f.read()
            
            field_name = os.path.basename(file_path)
            
            # Check if file already has a properly formatted boundaryField section
            if 'boundaryField' in content:
                # Make sure boundaryField has proper opening brace
                if not re.search(r'boundaryField\s*{', content):
                    self.log_update.emit(f"FIXING: Adding missing opening brace after boundaryField in {field_name}")
                    # Fix missing opening brace
                    content = content.replace('boundaryField', 'boundaryField\n{')
                
                # Check if allBoundary is already defined
                if 'allBoundary' not in content:
                    # Determine appropriate boundary condition
                    bc_type = "zeroGradient"
                    if field_name == "U":
                        bc_type = "noSlip"
                    
                    self.log_update.emit(f"FIXING: Adding missing allBoundary entry to {field_name}")
                    
                    # Add allBoundary entry before the closing brace of boundaryField
                    all_boundary_entry = f"""
    allBoundary
    {{
        type            {bc_type};
    }}
"""
                    # Find the closing brace of boundaryField
                    parts = content.split('boundaryField')
                    if len(parts) > 1:
                        # Count braces to find the matching closing brace
                        brace_count = 0
                        for i, char in enumerate(parts[1]):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    # Insert the allBoundary entry before this closing brace
                                    new_content = parts[0] + 'boundaryField' + parts[1][:i] + all_boundary_entry + parts[1][i:]
                                    break
                        else:
                            # If we didn't find a matching closing brace, add one
                            new_content = content + '\n}\n'
                            self.log_update.emit(f"FIXING: Adding missing closing brace to boundaryField in {field_name}")
                        
                        # Write the updated content
                        with open(file_path, 'w') as f:
                            f.write(new_content)
                        
                        self.log_update.emit(f"Fixed {field_name} file with proper boundaryField syntax and allBoundary entry")
            else:
                # Create a complete new field file with proper boundaryField section
                self.log_update.emit(f"FIXING: Creating complete boundaryField section in {field_name}")
                self._create_default_field_file(file_path)
                
        except Exception as e:
            self.log_update.emit(f"Error fixing field file {field_name}: {e}")
    
    def _create_default_field_file(self, file_path):
        """Create a default field file with proper structure if missing."""
        field_name = os.path.basename(file_path)
        
        # Default contents based on field type
        if field_name == "p":
            content = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    allBoundary
    {
        type            zeroGradient;
    }
}

// ************************************************************************* //
"""
        elif field_name == "U":
            content = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    allBoundary
    {
        type            noSlip;
    }
}

// ************************************************************************* //
"""
        
        # Write the file
        with open(file_path, 'w') as f:
            f.write(content)
            
        self.log_update.emit(f"Created new default {field_name} file with proper structure")


class ResidualPlot(QWidget):
    """
    Widget for displaying residual plots during simulation.
    
    This is a simple placeholder that can be replaced with an actual plotting
    library implementation (e.g., matplotlib, pyqtgraph).
    """
    
    def __init__(self, parent=None):
        """Initialize the residual plot."""
        super(ResidualPlot, self).__init__(parent)
        
        # Setup basic properties
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Data for plotting
        self.iterations = []
        self.residuals = {}  # Dict of field name to list of residuals
        self.colors = {
            "p": QColor(255, 0, 0),      # Red
            "U": QColor(0, 0, 255),      # Blue
            "k": QColor(0, 255, 0),      # Green
            "epsilon": QColor(255, 165, 0),  # Orange
            "omega": QColor(128, 0, 128),    # Purple
            "alpha.water": QColor(0, 128, 128),  # Teal
        }
        
        # Placeholder message
        self.placeholder_text = "Residual plot will appear here during simulation"
        
        # Flag to indicate if placeholder should be shown
        self.show_placeholder = True
    
    def add_data_point(self, iteration: int, residuals: Dict[str, float]):
        """
        Add a data point to the plot.
        
        Args:
            iteration (int): Current iteration
            residuals (Dict[str, float]): Residuals for each field
        """
        # Update data
        self.iterations.append(iteration)
        
        for field, value in residuals.items():
            if field not in self.residuals:
                self.residuals[field] = []
            self.residuals[field].append(value)
        
        # No longer need to show placeholder
        self.show_placeholder = False
        
        # Trigger repaint
        self.update()
    
    def clear(self):
        """Clear all data."""
        self.iterations = []
        self.residuals = {}
        self.show_placeholder = True
        self.update()
    
    def paintEvent(self, event):
        """Paint the widget with residual plots."""
        super(ResidualPlot, self).paintEvent(event)
        
        # Create a painter
        painter = QPainter(self)
        
        try:
            # Basic setup
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), Qt.white)
            
            if self.show_placeholder:
                # Show placeholder message
                painter.setPen(Qt.gray)
                painter.drawText(self.rect(), Qt.AlignCenter, self.placeholder_text)
                return
            
            # Early exit if no data
            if not self.iterations or not self.residuals:
                return
                
            # Calculate plot area
            margin = 40
            plot_rect = self.rect().adjusted(margin, margin, -margin, -margin)
            
            # Draw axes
            painter.setPen(Qt.black)
            painter.drawLine(
                int(plot_rect.bottomLeft().x()), 
                int(plot_rect.bottomLeft().y()), 
                int(plot_rect.bottomRight().x()), 
                int(plot_rect.bottomRight().y())
            )  # X-axis
            
            painter.drawLine(
                int(plot_rect.bottomLeft().x()), 
                int(plot_rect.bottomLeft().y()), 
                int(plot_rect.topLeft().x()), 
                int(plot_rect.topLeft().y())
            )  # Y-axis
            
            # Skip detailed plotting if not enough iterations
            if len(self.iterations) < 2:
                painter.drawText(plot_rect, Qt.AlignCenter, "Not enough data points yet")
                return
                
            # Calculate max iteration safely
            max_iter = max(self.iterations)
            if max_iter <= 0:
                max_iter = 1  # Prevent division by zero
            
            # Only add X-axis labels if we have meaningful iterations
            if max_iter >= 5:
                step = max(1, max_iter // 5)  # At most 5 labels
                for i in range(0, max_iter + 1, step):
                    x_pos = int(plot_rect.left() + (i * plot_rect.width()) / max_iter)
                    
                    # Draw tick mark
                    painter.drawLine(
                        x_pos, 
                        int(plot_rect.bottom()), 
                        x_pos, 
                        int(plot_rect.bottom() + 5)
                    )
                    
                    # Draw label
                    painter.drawText(
                        QRect(x_pos - 15, int(plot_rect.bottom() + 10), 30, 20),
                        Qt.AlignCenter, 
                        str(i)
                    )
            
            # Find valid residuals (positive values only)
            valid_residuals = {}
            min_res = float('inf')
            max_res = float('-inf')
            
            for field, values in self.residuals.items():
                valid_values = [v for v in values if v > 0]
                if valid_values:
                    valid_residuals[field] = valid_values
                    min_res = min(min_res, min(valid_values))
                    max_res = max(max_res, max(valid_values))
            
            # Skip plotting if no valid residuals found
            if not valid_residuals or min_res >= max_res:
                painter.drawText(plot_rect, Qt.AlignCenter, "No valid residual data")
                return
                
            # Use log scale with safety checks
            try:
                log_min = math.log10(max(1e-10, min_res))
                log_max = math.log10(max(1e-10, max_res))
                
                # Ensure at least 1 order of magnitude
                if abs(log_max - log_min) < 0.1:
                    log_min = log_max - 1
            except:
                # Fallback to simple defaults if log calculation fails
                log_min = -6
                log_max = 0
            
            # Draw Y-axis labels
            y_range = math.ceil(log_max) - math.floor(log_min)
            if y_range > 0:
                step = max(1, y_range // 5)  # At most 5 labels
                
                for i in range(math.floor(log_min), math.ceil(log_max) + 1, step):
                    # Calculate Y position safely
                    if log_max > log_min:
                        y_pos = int(plot_rect.bottom() - ((i - log_min) / (log_max - log_min)) * plot_rect.height())
                    else:
                        y_pos = int(plot_rect.bottom() - (plot_rect.height() / 2))
                    
                    # Draw tick
                    painter.drawLine(
                        int(plot_rect.left() - 5), 
                        y_pos, 
                        int(plot_rect.left()), 
                        y_pos
                    )
                    
                    # Draw label
                    painter.drawText(
                        QRect(int(plot_rect.left() - 35), y_pos - 10, 30, 20),
                        Qt.AlignRight | Qt.AlignVCenter, 
                        f"1e{i}"
                    )
            
            # Plot each residual series
            legend_items = []
            legend_x = int(plot_rect.right() - 100)
            legend_y = int(plot_rect.top() + 10)
            
            for field, values in self.residuals.items():
                # Skip fields with no valid data
                if not field in valid_residuals:
                    continue
                    
                # Get color
                color = self.colors.get(field, QColor(0, 0, 0))
                painter.setPen(QPen(color, 2))
                
                # Add to legend
                legend_items.append(field)
                
                # Prepare points
                path = QPainterPath()
                first_point = True
                
                for i, val in enumerate(values):
                    # Skip invalid values
                    if val <= 0 or i >= len(self.iterations):
                        continue
                    
                    # Calculate X coordinate safely
                    iter_val = self.iterations[i]
                    if max_iter > 0:
                        x_pos = int(plot_rect.left() + (iter_val * plot_rect.width()) / max_iter)
                    else:
                        x_pos = int(plot_rect.left())
                    
                    # Calculate Y coordinate safely
                    try:
                        log_val = math.log10(max(1e-10, val))
                        if log_max > log_min:
                            y_pos = int(plot_rect.bottom() - ((log_val - log_min) / (log_max - log_min)) * plot_rect.height())
                        else:
                            y_pos = int(plot_rect.bottom() - (plot_rect.height() / 2))
                    except:
                        continue
                    
                    # Add point to path
                    if first_point:
                        path.moveTo(x_pos, y_pos)
                        first_point = False
                    else:
                        path.lineTo(x_pos, y_pos)
                
                # Draw the path
                painter.drawPath(path)
            
            # Draw legend
            legend_height = 15
            for i, field in enumerate(legend_items):
                # Get color
                color = self.colors.get(field, QColor(0, 0, 0))
                
                # Draw color line
                painter.setPen(QPen(color, 2))
                painter.drawLine(
                    legend_x, 
                    int(legend_y + i*legend_height), 
                    legend_x + 20, 
                    int(legend_y + i*legend_height)
                )
                
                # Draw field name
                painter.setPen(Qt.black)
                painter.drawText(
                    QRect(legend_x + 25, int(legend_y + i*legend_height - 5), 70, 20),
                    Qt.AlignLeft | Qt.AlignVCenter, 
                    field
                )
                           
        except Exception as e:
            # Log error and show message
            import logging
            logging.getLogger(__name__).error(f"Error plotting residuals: {e}")
            
            # Clear and show error message
            painter.fillRect(self.rect(), Qt.white)
            painter.setPen(Qt.red)
            painter.drawText(self.rect(), Qt.AlignCenter, "Error plotting residuals")
        
        finally:
            # CRITICAL: Always end the painter to avoid Qt errors
            painter.end()


class BoundaryPanel(QWidget):
    """Panel for configuring boundary conditions within simulation controls."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent.main_window if hasattr(parent, 'main_window') else parent
        self.boundary_config = {}
        self.boundary_names = []
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the UI for boundary configuration."""
        layout = QVBoxLayout(self)
        
        # Instructions label
        instructions = QLabel(
            "Configure boundary conditions for OpenFOAM simulation. "
            "Select faces in the viewport to create new boundaries."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Create boundary button
        self.create_boundary_btn = QPushButton("Create New Boundary")
        self.create_boundary_btn.clicked.connect(self.start_face_selection)
        layout.addWidget(self.create_boundary_btn)
        
        # Boundary list
        list_label = QLabel("Available Boundaries:")
        layout.addWidget(list_label)
        
        self.boundary_list = QListWidget()
        self.boundary_list.currentItemChanged.connect(self.boundary_selected)
        layout.addWidget(self.boundary_list)
        
        # Type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Boundary Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["inlet", "outlet", "wall", "symmetryPlane", "empty"])
        self.type_combo.currentTextChanged.connect(self.type_changed)
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)
        
        # Stacked widget for different boundary type configurations
        self.config_stack = QStackedWidget()
        
        # Inlet configuration
        inlet_widget = QWidget()
        inlet_layout = QFormLayout(inlet_widget)
        
        # Velocity
        vel_group = QGroupBox("Velocity")
        vel_form = QFormLayout(vel_group)
        
        self.vel_type = QComboBox()
        self.vel_type.addItems(["fixedValue", "flowRate", "pressureInletOutletVelocity"])
        vel_form.addRow("Type:", self.vel_type)
        
        vel_values = QHBoxLayout()
        
        self.vel_x = QDoubleSpinBox()
        self.vel_x.setRange(-100, 100)
        self.vel_x.setValue(1.0)
        self.vel_x.setDecimals(3)
        self.vel_x.setSuffix(" m/s")
        
        self.vel_y = QDoubleSpinBox()
        self.vel_y.setRange(-100, 100)
        self.vel_y.setValue(0.0)
        self.vel_y.setDecimals(3)
        self.vel_y.setSuffix(" m/s")
        
        self.vel_z = QDoubleSpinBox()
        self.vel_z.setRange(-100, 100)
        self.vel_z.setValue(0.0)
        self.vel_z.setDecimals(3)
        self.vel_z.setSuffix(" m/s")
        
        vel_values.addWidget(QLabel("X:"))
        vel_values.addWidget(self.vel_x)
        vel_values.addWidget(QLabel("Y:"))
        vel_values.addWidget(self.vel_y)
        vel_values.addWidget(QLabel("Z:"))
        vel_values.addWidget(self.vel_z)
        
        vel_form.addRow("Value:", vel_values)
        inlet_layout.addRow(vel_group)
        
        # Pressure
        p_group = QGroupBox("Pressure")
        p_form = QFormLayout(p_group)
        
        self.p_type = QComboBox()
        self.p_type.addItems(["zeroGradient", "totalPressure", "fixedValue"])
        p_form.addRow("Type:", self.p_type)
        
        self.p_value = QDoubleSpinBox()
        self.p_value.setRange(0, 1000000)
        self.p_value.setValue(0)
        self.p_value.setDecimals(1)
        self.p_value.setSuffix(" Pa")
        p_form.addRow("Value:", self.p_value)
        
        inlet_layout.addRow(p_group)
        
        # Temperature
        t_group = QGroupBox("Temperature")
        t_form = QFormLayout(t_group)
        
        self.t_type = QComboBox()
        self.t_type.addItems(["fixedValue", "zeroGradient"])
        t_form.addRow("Type:", self.t_type)
        
        self.t_value = QDoubleSpinBox()
        self.t_value.setRange(0, 1000)
        self.t_value.setValue(300)
        self.t_value.setDecimals(1)
        self.t_value.setSuffix(" K")
        t_form.addRow("Value:", self.t_value)
        
        inlet_layout.addRow(t_group)
        
        self.config_stack.addWidget(inlet_widget)
        
        # Add other boundary type configurations (outlet, wall) similar to boundary_dialog.py
        # ... [Code for other configurations omitted for brevity] ...
        
        layout.addWidget(self.config_stack)
        
        # Apply button for current boundary
        self.apply_btn = QPushButton("Apply Configuration")
        self.apply_btn.clicked.connect(self.apply_to_boundary)
        layout.addWidget(self.apply_btn)
        
        # Status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        # Add stretch to push everything to the top
        layout.addStretch()
    
    def start_face_selection(self):
        """Start face selection for a new boundary"""
        if not hasattr(self.main_window, 'viewport'):
            QMessageBox.warning(self, "Error", "No viewport available for selection")
            return
        
        # Create dialog for boundary configuration
        dialog = QDialog(self)
        dialog.setWindowTitle("New Boundary")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout(dialog)
        
        # Add name and type fields
        form_layout = QFormLayout()
        
        # Name field
        name_edit = QLineEdit("new_boundary")
        form_layout.addRow("Boundary Name:", name_edit)
        
        # Type dropdown
        type_combo = QComboBox()
        type_combo.addItems(["inlet", "outlet", "wall", "symmetry", "empty"])
        form_layout.addRow("Boundary Type:", type_combo)
        
        layout.addLayout(form_layout)
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # Show dialog
        if dialog.exec_() != QDialog.Accepted:
            return
        
        # Get the boundary name and type from the dialog
        boundary_name = name_edit.text().strip()
        boundary_type = type_combo.currentText()
        
        if not boundary_name:
            QMessageBox.warning(self, "Error", "Boundary name cannot be empty")
            return
        
        # Check if boundary already exists
        for i in range(self.boundary_list.count()):
            item = self.boundary_list.item(i)
            if item and item.data(Qt.UserRole) == boundary_name:
                QMessageBox.warning(
                    self,
                    "Duplicate Boundary",
                    f"A boundary named '{boundary_name}' already exists. Please choose a different name."
                )
                return
        
        logger.info(f"Starting face selection for boundary: {boundary_name} (type: {boundary_type})")
        
        # Start the selection in the viewport
        self.status_label.setText(f"Selecting faces for {boundary_type} boundary '{boundary_name}'...")
        success = self.main_window.viewport.start_face_selection(
            boundary_name,
            lambda faces: self._on_face_selection_complete(boundary_name, boundary_type, faces)
        )
        
        if not success:
            self.status_label.setText("Failed to start face selection")
            QMessageBox.warning(self, "Error", "Failed to start face selection mode")
    
    def _on_face_selection_complete(self, boundary_name, boundary_type, face_ids):
        """Handle completion of face selection"""
        if not face_ids:
            logger.warning(f"No faces selected for boundary {boundary_name}")
            self.status_label.setText("No faces selected")
            return
        
        logger.info(f"Face selection complete: {len(face_ids)} faces for {boundary_name} (type: {boundary_type})")
        
        # Store the selected faces
        if not hasattr(self, 'boundary_cell_ids'):
            self.boundary_cell_ids = {}
        self.boundary_cell_ids[boundary_name] = face_ids
        
        # Add to boundary list
        item = QListWidgetItem(boundary_name)
        item.setData(Qt.UserRole, boundary_name)
        self.boundary_list.addItem(item)
        self.boundary_list.setCurrentItem(item)
        
        # Set boundary type
        self.type_combo.setCurrentText(boundary_type)
        
        # Create boundary configuration
        if not hasattr(self, 'boundary_config'):
            self.boundary_config = {}
        
        self.boundary_config[boundary_name] = {
            'type': boundary_type
        }
        
        # Update status
        self.status_label.setText(f"Created {boundary_type} boundary '{boundary_name}' with {len(face_ids)} faces")
        
        # Save configuration
        self._save_boundary_conditions()
    
    def boundary_selected(self, current, previous):
        """Called when a boundary is selected from the list."""
        if not current:
            return
        
        boundary_name = current.data(Qt.UserRole)
        
        # Check if we already have configuration for this boundary
        if hasattr(self, 'boundary_config') and boundary_name in self.boundary_config:
            config = self.boundary_config[boundary_name]
            
            # Set the boundary type
            index = self.type_combo.findText(config.get("type", "wall"))
            if index >= 0:
                self.type_combo.setCurrentIndex(index)
            
            # Set the configuration values based on boundary type
            if config["type"] == "inlet":
                # Set velocity
                if "velocity" in config:
                    self.vel_x.setValue(config["velocity"][0])
                    self.vel_y.setValue(config["velocity"][1])
                    self.vel_z.setValue(config["velocity"][2])
                
                # Set pressure
                if "pressure" in config:
                    self.p_value.setValue(config["pressure"])
                
                # Set temperature
                if "temperature" in config:
                    self.t_value.setValue(config["temperature"])
            
            elif config["type"] == "outlet":
                # Set pressure
                if "pressure" in config:
                    self.p_value.setValue(config["pressure"])
        
        # Highlight this boundary in the viewport
        if hasattr(self, 'main_window') and self.main_window and hasattr(self.main_window, 'viewport'):
            # Check if we have cell IDs for this boundary
            if hasattr(self, 'boundary_cell_ids') and boundary_name in self.boundary_cell_ids:
                cell_ids = self.boundary_cell_ids[boundary_name]
                
                # Debug log to see what type of data we're working with
                logger.debug(f"Cell IDs type: {type(cell_ids)}")
                
                # Process cell_ids to ensure they're in the correct format
                processed_ids = []
                try:
                    # Check if cell_ids is a dictionary (which seems to be the case based on error)
                    if isinstance(cell_ids, dict):
                        # Extract values from dictionary
                        for face_id in cell_ids.values():
                            if isinstance(face_id, (list, tuple)):
                                processed_ids.extend([int(i) for i in face_id])
                            else:
                                processed_ids.append(int(face_id))
                    elif isinstance(cell_ids, (list, tuple)):
                        # Handle list format
                        for cell_id in cell_ids:
                            if hasattr(cell_id, '__iter__') and not isinstance(cell_id, (str, bytes)):
                                processed_ids.append(int(cell_id[0]))
                            else:
                                processed_ids.append(int(cell_id))
                except Exception as e:
                    logger.error(f"Error processing cell IDs: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    processed_ids = []
                    
                if processed_ids:
                    self.status_label.setText(f"Selected boundary '{boundary_name}' with {len(processed_ids)} faces.")
                    try:
                        # Use the processed IDs
                        self.main_window.viewport.highlight_faces(boundary_name, processed_ids)
                    except Exception as e:
                        logger.error(f"Failed to highlight faces: {e}")
                        self.status_label.setText(f"Error highlighting boundary: {str(e)}")
                else:
                    self.status_label.setText(f"Selected boundary '{boundary_name}' (no valid face IDs found)")

    def type_changed(self, new_type):
        """Called when the boundary type is changed."""
        # Set the appropriate configuration widget
        if new_type == "inlet":
            self.config_stack.setCurrentIndex(0)
        elif new_type == "outlet":
            self.config_stack.setCurrentIndex(1)
        elif new_type == "wall":
            self.config_stack.setCurrentIndex(2)
        elif new_type == "symmetryPlane":
            self.config_stack.setCurrentIndex(3)
        elif new_type == "empty":
            self.config_stack.setCurrentIndex(4)

    def apply_to_boundary(self):
        """Apply the current boundary settings to the named boundary"""
        try:
            # Get boundary name from current selection
            current_item = self.boundary_list.currentItem()
            if not current_item:
                QMessageBox.warning(
                    self,
                    "No Boundary Selected",
                    "Please select a boundary to apply configuration."
                )
                return
                
            boundary_name = current_item.data(Qt.UserRole)
            if not boundary_name:
                QMessageBox.warning(
                    self, 
                    "Invalid Selection",
                    "Invalid boundary selection"
                )
                return
                
            logger.info(f"Applying boundary settings to {boundary_name}")
            
            # Determine boundary type from UI
            boundary_type = self.type_combo.currentText()
            
            # Create a configuration dictionary
            config = {
                "name": boundary_name,
                "type": boundary_type
            }
            
            # Add velocity settings based on boundary type
            velocity_type = None
            
            # Try to get the velocity type from the appropriate UI element based on boundary type
            if boundary_type == "inlet":
                if hasattr(self, 'vel_type'):
                    velocity_type = self.vel_type.currentText()
            elif boundary_type == "outlet":
                if hasattr(self, 'vel_type'):
                    velocity_type = self.vel_type.currentText()
            elif boundary_type == "wall":
                if hasattr(self, 'wall_vel_type'):
                    velocity_type = self.wall_vel_type.currentText()
            
            # Set a default velocity type if none was found
            if velocity_type is None:
                velocity_type = "fixedValue" if boundary_type == "inlet" else "noSlip" if boundary_type == "wall" else "zeroGradient"
            
            config["velocity_type"] = velocity_type
            
            # Add velocity values if relevant
            if hasattr(self, 'vel_x') and hasattr(self, 'vel_y') and hasattr(self, 'vel_z'):
                config["velocity_x"] = self.vel_x.value()
                config["velocity_y"] = self.vel_y.value()
                config["velocity_z"] = self.vel_z.value()
            
            # Add pressure settings
            pressure_type = None
            
            # Try to get the pressure type
            if boundary_type == "inlet" and hasattr(self, 'p_type'):
                pressure_type = self.p_type.currentText()
            elif boundary_type == "outlet" and hasattr(self, 'out_p_type'):
                pressure_type = self.out_p_type.currentText()
            
            # Set a default pressure type if none was found
            if pressure_type is None:
                pressure_type = "fixedValue" if boundary_type == "outlet" else "zeroGradient"
            
            config["pressure_type"] = pressure_type
            
            # Add pressure value if relevant
            if boundary_type == "inlet" and hasattr(self, 'p_value'):
                config["pressure_value"] = self.p_value.value()
            elif boundary_type == "outlet" and hasattr(self, 'out_p_value'):
                config["pressure_value"] = self.out_p_value.value()
            
            # Temperature settings (if available)
            temperature_type = None
            
            if hasattr(self, 't_type'):
                temperature_type = self.t_type.currentText()
            
            if temperature_type:
                config["temperature_type"] = temperature_type
                
                if hasattr(self, 't_value'):
                    config["temperature_value"] = self.t_value.value()
            
            # Save the configuration to project
            project_saved = False
            if not hasattr(self, 'boundary_config'):
                self.boundary_config = {}
            
            self.boundary_config[boundary_name] = config
            
            # Save boundary conditions to project
            if self._save_boundary_conditions():
                project_saved = True
            
            # Apply to OpenFOAM case directory if available
            openfoam_saved = False
            case_manager_used = None
            
            # Try to use the case manager directly attached to this object
            if hasattr(self, 'case_manager') and self.case_manager:
                case_manager_used = self.case_manager
                success = self.case_manager.set_boundary_condition(
                    boundary_name, boundary_type, config
                )
                if success:
                    logger.info(f"Applied boundary settings to OpenFOAM case: {config}")
                    openfoam_saved = True
                else:
                    logger.warning(f"Failed to apply boundary settings to OpenFOAM case")
            
            # If that didn't work or wasn't available, try the project's case manager
            elif hasattr(self.main_window, 'project') and hasattr(self.main_window.project, 'case_manager') and self.main_window.project.case_manager:
                case_manager_used = self.main_window.project.case_manager
                success = self.main_window.project.case_manager.set_boundary_condition(
                    boundary_name, boundary_type, config
                )
                if success:
                    logger.info(f"Applied boundary settings to OpenFOAM case via project: {config}")
                    openfoam_saved = True
                else:
                    logger.warning(f"Failed to apply boundary settings to OpenFOAM case via project")
            
            # If neither worked, try the main window's case manager
            elif hasattr(self.main_window, 'case_manager') and self.main_window.case_manager:
                case_manager_used = self.main_window.case_manager
                success = self.main_window.case_manager.set_boundary_condition(
                    boundary_name, boundary_type, config
                )
                if success:
                    logger.info(f"Applied boundary settings to OpenFOAM case via main window: {config}")
                    openfoam_saved = True
                else:
                    logger.warning(f"Failed to apply boundary settings to OpenFOAM case via main window")
            
            # Log diagnostic information to help troubleshoot
            if not openfoam_saved:
                logger.warning(f"Case manager used: {case_manager_used.__class__.__name__ if case_manager_used else 'None'}")
                logger.warning(f"Case directory: {self.case_dir if hasattr(self, 'case_dir') else 'Not set'}")
                if case_manager_used:
                    logger.warning(f"Case manager directory: {case_manager_used.case_directory}")
            
            # Update status message
            if project_saved and openfoam_saved:
                self.status_label.setText(f"Applied configuration to '{boundary_name}' in project and OpenFOAM case")
            elif project_saved:
                if hasattr(self, 'case_dir') and self.case_dir:
                    self.status_label.setText(f"Applied configuration to '{boundary_name}' in project only (OpenFOAM update failed)")
                else:
                    self.status_label.setText(f"Applied configuration to '{boundary_name}' (saved to project only - no case directory set)")
            else:
                self.status_label.setText(f"Failed to apply configuration to '{boundary_name}'")
            
            # Force UI update
            QApplication.processEvents()
            
            return config
        
        except Exception as e:
            logger.error(f"Error applying boundary settings: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.status_label.setText(f"Error: {str(e)}")
            return {}



class SimulationControls(QWidget):
    """
    Simulation control panel for Openfoam_Simulator.
    
    This panel provides controls for configuring and running OpenFOAM simulations,
    with specialized features for oil & gas industry applications.
    """
    
    # Signals
     # Add these signals at the class level (not inside any method)
    status_update = pyqtSignal(str)
    log_update = pyqtSignal(str)
    simulation_started = pyqtSignal()
    simulation_stopped = pyqtSignal()
    simulation_finished = pyqtSignal(bool)  # success flag
    
    def __init__(self, main_window):
        """
        Initialize the simulation controls panel.
        
        Args:
            main_window: The main window
        """
        super().__init__()
        self.main_window = main_window
        self.case_dir = None
        self.is_running = False
        self.solver = None
        self.runner = None
        
        # Initialize boundary conditions dictionary
        self.boundary_conditions = {}
        
        # Set up UI
        self.setup_ui()
        
        # Connect signals
        self._connect_signals()
        
        # Automatically set the case directory if a project is already loaded
        self.auto_set_case_directory()
        
        # Connect to project signals for case directory updates
        if hasattr(self.main_window, 'project_loaded'):
            self.main_window.project_loaded.connect(self.on_project_loaded)
        
        # Load boundary conditions if available
        self._load_boundary_conditions()
        
        # Connect to project signals
        if hasattr(self.main_window, 'project_loaded'):
            self.main_window.project_loaded.connect(self.on_project_loaded)
    
    def on_project_loaded(self, project):
        """Called when a project is loaded."""
        try:
            # Get case directory from the project
            if hasattr(project, 'get_case_directory'):
                case_dir = project.get_case_directory()
                if case_dir and os.path.exists(case_dir):
                    self.case_dir = case_dir
                    if hasattr(self, 'case_dir_edit'):
                        self.case_dir_edit.setText(case_dir)
                    logger.info(f"Connected to OpenFOAM case directory: {case_dir}")
                    
                    # Create case manager for this directory
                    from ..openfoam_integration.case_manager import create_case_manager
                    self.case_manager = create_case_manager(case_dir)
                    
                    # Enable UI elements
                    if hasattr(self, 'setup_button'):
                        self.setup_button.setEnabled(True)
                    if hasattr(self, 'run_button'):
                        self.run_button.setEnabled(True)
            else:
                logger.warning("Project object doesn't have get_case_directory method")
        except Exception as e:
            logger.error(f"Error connecting to case directory: {e}")
    
    def setup_ui(self):
        """Set up the UI elements for the simulation controls panel"""
        # Create main layout
        self.layout = QVBoxLayout(self)
        
        # Create tab widget - use only a single tab widget
        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.solver_tab = QWidget()
        self.parameters_tab = QWidget()
        self.monitoring_tab = QWidget()
        self.boundary_tab = QWidget()
        self.ambient_tab = QWidget()
        
        # Setup tab contents
        self._setup_solver_tab()
        self._setup_parameters_tab()
        self._setup_monitoring_tab()
        self._setup_boundary_tab()
        
        # Setup the ambient region tab
        ambient_layout = QVBoxLayout(self.ambient_tab)
        ambient_layout.addWidget(self._setup_ambient_region_controls())
        
        # Add tabs to widget
        self.tab_widget.addTab(self.boundary_tab, "Boundaries")
        self.tab_widget.addTab(self.ambient_tab, "Ambient Region")
        
        # Add buttons at the bottom
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        
        self.setup_button = QPushButton("Configure Case")  # Change text from "Setup Case" to "Configure Case"
        self.setup_button.setEnabled(False)
        buttons_layout.addWidget(self.setup_button)
        
        self.run_button = QPushButton("Run")
        self.run_button.setEnabled(False)
        buttons_layout.addWidget(self.run_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_button)
        
        self.layout.addWidget(buttons_widget)
        
        # Set layout
        self.setLayout(self.layout)
    
    def _setup_solver_tab(self):
        """Set up the solver configuration tab."""
        solver_widget = QWidget()
        solver_layout = QVBoxLayout(solver_widget)
        
        # Solver selection group
        solver_group = QGroupBox("Solver")
        solver_group_layout = QFormLayout(solver_group)
        
        # Solver combobox
        self.solver_combo = QComboBox()
        self.solver_combo.addItems([
            "simpleFoam",    # Steady-state incompressible
            "pisoFoam",      # Transient incompressible
            "rhoPimpleFoam", # Transient compressible
            "interFoam",     # Two-phase incompressible
            "multiphaseInterFoam", # Multi-phase incompressible
            "reactingFoam"   # Reacting flow
        ])
        solver_group_layout.addRow("Solver:", self.solver_combo)
        
        # Turbulence model
        self.turbulence_combo = QComboBox()
        self.turbulence_combo.addItems([
            "kEpsilon",
            "kOmega",
            "SpalartAllmaras",
            "LES",
            "laminar"
        ])
        solver_group_layout.addRow("Turbulence Model:", self.turbulence_combo)
        
        # Solver settings
        self.steady_radio = QRadioButton("Steady")
        self.transient_radio = QRadioButton("Transient")
        self.steady_radio.setChecked(True)
        
        time_type_layout = QHBoxLayout()
        time_type_layout.addWidget(self.steady_radio)
        time_type_layout.addWidget(self.transient_radio)
        solver_group_layout.addRow("Time:", time_type_layout)
        
        solver_layout.addWidget(solver_group)
        
        # Numerical schemes group
        schemes_group = QGroupBox("Numerical Schemes")
        schemes_layout = QFormLayout(schemes_group)
        
        # Gradient scheme
        self.gradient_combo = QComboBox()
        self.gradient_combo.addItems(["Gauss linear", "leastSquares", "cellLimited Gauss linear 1"])
        schemes_layout.addRow("Gradient:", self.gradient_combo)
        
        # Divergence scheme
        self.divergence_combo = QComboBox()
        self.divergence_combo.addItems(["Gauss linear", "Gauss upwind", "Gauss linearUpwind grad(U)"])
        schemes_layout.addRow("Divergence:", self.divergence_combo)
        
        # Laplacian scheme
        self.laplacian_combo = QComboBox()
        self.laplacian_combo.addItems(["Gauss linear corrected", "Gauss linear limited 0.5"])
        schemes_layout.addRow("Laplacian:", self.laplacian_combo)
        
        # Interpolation scheme
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(["linear", "midPoint"])
        schemes_layout.addRow("Interpolation:", self.interpolation_combo)
        
        solver_layout.addWidget(schemes_group)
        
        # Parallel settings
        parallel_group = QGroupBox("Parallel Execution")
        parallel_layout = QFormLayout(parallel_group)
        
        # Enable parallel checkbox
        self.parallel_check = QCheckBox("Run in parallel")
        parallel_layout.addRow("", self.parallel_check)
        
        # Number of processors
        self.processors_spin = QSpinBox()
        self.processors_spin.setMinimum(1)
        self.processors_spin.setMaximum(64)
        self.processors_spin.setValue(4)
        self.processors_spin.setEnabled(False)  # Disabled until parallel is checked
        parallel_layout.addRow("Processors:", self.processors_spin)
        
        # Connect parallel checkbox to processors spinbox
        self.parallel_check.toggled.connect(self.processors_spin.setEnabled)
        
        # Decomposition method
        self.decomp_combo = QComboBox()
        self.decomp_combo.addItems(["simple", "hierarchical", "scotch"])
        self.decomp_combo.setEnabled(False)
        parallel_layout.addRow("Decomposition:", self.decomp_combo)
        self.parallel_check.toggled.connect(self.decomp_combo.setEnabled)
        
        solver_layout.addWidget(parallel_group)
        
        # Add stretch to push everything to the top
        solver_layout.addStretch()
        
        # Add tab
        self.tab_widget.addTab(solver_widget, "Solver")
    
    def _setup_parameters_tab(self):
        """Set up the simulation parameters tab."""
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)
        
        # Time controls group
        time_group = QGroupBox("Time Controls")
        time_layout = QFormLayout(time_group)
        
        # Start time
        self.start_time_spin = QDoubleSpinBox()
        self.start_time_spin.setDecimals(3)
        self.start_time_spin.setMinimum(0)
        self.start_time_spin.setMaximum(1000)
        self.start_time_spin.setValue(0)
        time_layout.addRow("Start Time:", self.start_time_spin)
        
        # End time
        self.end_time_spin = QDoubleSpinBox()
        self.end_time_spin.setDecimals(3)
        self.end_time_spin.setMinimum(0.001)
        self.end_time_spin.setMaximum(1000000)
        self.end_time_spin.setValue(100)
        time_layout.addRow("End Time:", self.end_time_spin)
        
        # Time step
        self.time_step_spin = QDoubleSpinBox()
        self.time_step_spin.setDecimals(6)
        self.time_step_spin.setMinimum(0.000001)
        self.time_step_spin.setMaximum(1000)
        self.time_step_spin.setValue(0.001)
        time_layout.addRow("Time Step:", self.time_step_spin)
        
        # Max Courant number
        self.max_co_spin = QDoubleSpinBox()
        self.max_co_spin.setDecimals(2)
        self.max_co_spin.setMinimum(0.1)
        self.max_co_spin.setMaximum(10)
        self.max_co_spin.setValue(1.0)
        time_layout.addRow("Max Courant Number:", self.max_co_spin)
        
        # Adjustable time step
        self.adjustable_time_check = QCheckBox()
        time_layout.addRow("Adjustable Time Step:", self.adjustable_time_check)
        
        params_layout.addWidget(time_group)
        
        # Solution controls group
        solution_group = QGroupBox("Solution Controls")
        solution_layout = QFormLayout(solution_group)
        
        # Number of iterations
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setMinimum(1)
        self.iterations_spin.setMaximum(10000)
        self.iterations_spin.setValue(1000)
        solution_layout.addRow("Max Iterations:", self.iterations_spin)
        
        # Convergence tolerance
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setDecimals(10)
        self.tolerance_spin.setMinimum(1e-12)
        self.tolerance_spin.setMaximum(1)
        self.tolerance_spin.setValue(1e-6)
        self.tolerance_spin.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
        solution_layout.addRow("Convergence Tolerance:", self.tolerance_spin)
        
        # Relaxation factors
        self.p_relax_spin = QDoubleSpinBox()
        self.p_relax_spin.setDecimals(2)
        self.p_relax_spin.setMinimum(0.01)
        self.p_relax_spin.setMaximum(1.0)
        self.p_relax_spin.setValue(0.3)
        solution_layout.addRow("p Relaxation:", self.p_relax_spin)
        
        self.u_relax_spin = QDoubleSpinBox()
        self.u_relax_spin.setDecimals(2)
        self.u_relax_spin.setMinimum(0.01)
        self.u_relax_spin.setMaximum(1.0)
        self.u_relax_spin.setValue(0.7)
        solution_layout.addRow("U Relaxation:", self.u_relax_spin)
        
        params_layout.addWidget(solution_group)
        
        # Output controls group
        output_group = QGroupBox("Output Controls")
        output_layout = QFormLayout(output_group)
        
        # Write interval
        self.write_interval_spin = QDoubleSpinBox()
        self.write_interval_spin.setDecimals(3)
        self.write_interval_spin.setMinimum(0.001)
        self.write_interval_spin.setMaximum(1000)
        self.write_interval_spin.setValue(1)
        output_layout.addRow("Write Interval:", self.write_interval_spin)
        
        # Output format
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["binary", "ascii"])
        output_layout.addRow("Output Format:", self.output_format_combo)
        
        # Compression
        self.compression_check = QCheckBox()
        self.compression_check.setChecked(True)
        output_layout.addRow("Compress Results:", self.compression_check)
        
        params_layout.addWidget(output_group)
        
        # Add stretch to push everything to the top
        params_layout.addStretch()
        
        # Add tab
        self.tab_widget.addTab(params_widget, "Parameters")
    
    def _setup_monitoring_tab(self):
        """Set up the simulation monitoring tab."""
        monitor_widget = QWidget()
        monitor_layout = QVBoxLayout(monitor_widget)
        
        # Progress group
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # Progress information
        info_layout = QFormLayout()
        
        self.status_label = QLabel("Not running")
        info_layout.addRow("Status:", self.status_label)
        
        self.iteration_label = QLabel("0")
        info_layout.addRow("Iteration:", self.iteration_label)
        
        self.time_label = QLabel("0.0")
        info_layout.addRow("Time:", self.time_label)
        
        progress_layout.addLayout(info_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        monitor_layout.addWidget(progress_group)
        
        # Residuals group
        residuals_group = QGroupBox("Residuals")
        residuals_layout = QVBoxLayout(residuals_group)
        
        # Residual plot
        self.residual_plot = ResidualPlot()
        residuals_layout.addWidget(self.residual_plot)
        
        # Table of current residuals
        self.residuals_table = QTableWidget(0, 2)
        self.residuals_table.setHorizontalHeaderLabels(["Field", "Residual"])
        self.residuals_table.horizontalHeader().setStretchLastSection(True)
        self.residuals_table.setMaximumHeight(150)
        residuals_layout.addWidget(self.residuals_table)
        
        monitor_layout.addWidget(residuals_group)
        
        # Log output
        log_group = QGroupBox("Solver Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)
        log_layout.addWidget(self.log_text)
        
        monitor_layout.addWidget(log_group)
        
        # Add tab
        self.tab_widget.addTab(monitor_widget, "Monitoring")
    
    def _setup_oil_gas_tab(self):
        """Set up the oil & gas industry-specific tab."""
        oil_gas_widget = QWidget()
        oil_gas_layout = QVBoxLayout(oil_gas_widget)
        
        # Simulation type selection
        type_group = QGroupBox("Simulation Type")
        type_layout = QVBoxLayout(type_group)
        
        # Radio buttons for simulation type
        self.single_phase_radio = QRadioButton("Single Phase Flow")
        self.multi_phase_radio = QRadioButton("Multi-Phase Flow")
        self.pigging_radio = QRadioButton("Pipeline Pigging")
        self.spill_radio = QRadioButton("Spill Quantification")
        
        self.single_phase_radio.setChecked(True)
        
        type_layout.addWidget(self.single_phase_radio)
        type_layout.addWidget(self.multi_phase_radio)
        type_layout.addWidget(self.pigging_radio)
        type_layout.addWidget(self.spill_radio)
        
        # Create button group to manage radio buttons
        self.sim_type_group = QButtonGroup()
        self.sim_type_group.addButton(self.single_phase_radio, 0)
        self.sim_type_group.addButton(self.multi_phase_radio, 1)
        self.sim_type_group.addButton(self.pigging_radio, 2)
        self.sim_type_group.addButton(self.spill_radio, 3)
        
        oil_gas_layout.addWidget(type_group)
        
        # Stacked widget to show different settings depending on simulation type
        self.settings_stack = QStackedWidget()
        
        # 1. Single Phase Flow settings
        single_phase_widget = QWidget()
        single_phase_layout = QFormLayout(single_phase_widget)
        
        # Fluid properties
        self.fluid_combo = QComboBox()
        self.fluid_combo.addItems(["Water", "Oil", "Gas", "Custom"])
        single_phase_layout.addRow("Fluid:", self.fluid_combo)
        
        self.density_spin = QDoubleSpinBox()
        self.density_spin.setRange(0.1, 2000)
        self.density_spin.setDecimals(2)
        self.density_spin.setSingleStep(10.0)
        self.density_spin.setValue(1000)  # Default water density
        single_phase_layout.addRow("Density (kg/m³):", self.density_spin)
        
        self.viscosity_spin = QDoubleSpinBox()
        self.viscosity_spin.setRange(1e-6, 1000.0)
        self.viscosity_spin.setDecimals(6)
        self.viscosity_spin.setSingleStep(0.001)
        self.viscosity_spin.setValue(0.001)  # Default water viscosity
        single_phase_layout.addRow("Viscosity (Pa·s):", self.viscosity_spin)
        
        # Connect fluid combo to property updates
        self.fluid_combo.currentIndexChanged.connect(self._update_fluid_properties)
        
        self.settings_stack.addWidget(single_phase_widget)
        
        # 2. Multi-Phase Flow settings
        multi_phase_widget = QWidget()
        multi_phase_layout = QFormLayout(multi_phase_widget)
        
        # Phase settings
        self.phase_combo = QComboBox()
        self.phase_combo.addItems(["Oil-Water", "Gas-Liquid", "Oil-Water-Gas"])
        multi_phase_layout.addRow("Phases:", self.phase_combo)
        
        # Surface tension
        self.surface_tension_spin = QDoubleSpinBox()
        self.surface_tension_spin.setDecimals(4)
        self.surface_tension_spin.setMinimum(0.0001)
        self.surface_tension_spin.setMaximum(1)
        self.surface_tension_spin.setValue(0.072)  # Water-air surface tension
        multi_phase_layout.addRow("Surface Tension (N/m):", self.surface_tension_spin)
        
        # Phase fractions
        self.phase1_spin = QDoubleSpinBox()
        self.phase1_spin.setDecimals(2)
        self.phase1_spin.setMinimum(0)
        self.phase1_spin.setMaximum(1)
        self.phase1_spin.setValue(0.5)
        multi_phase_layout.addRow("Oil Fraction:", self.phase1_spin)
        
        self.settings_stack.addWidget(multi_phase_widget)
        
        # 3. Pipeline Pigging settings
        pigging_widget = QWidget()
        pigging_layout = QFormLayout(pigging_widget)
        
        # Pig type
        self.pig_type_combo = QComboBox()
        self.pig_type_combo.addItems(["Foam", "Gel", "Sphere", "Cup", "Disc"])
        pigging_layout.addRow("Pig Type:", self.pig_type_combo)
        
        # Pig settings
        self.pig_diameter_spin = QDoubleSpinBox()
        self.pig_diameter_spin.setDecimals(3)
        self.pig_diameter_spin.setMinimum(0.01)
        self.pig_diameter_spin.setMaximum(2)
        self.pig_diameter_spin.setValue(0.15)  # 150mm pig
        pigging_layout.addRow("Pig Diameter (m):", self.pig_diameter_spin)
        
        self.pig_length_spin = QDoubleSpinBox()
        self.pig_length_spin.setDecimals(3)
        self.pig_length_spin.setMinimum(0.01)
        self.pig_length_spin.setMaximum(5)
        self.pig_length_spin.setValue(0.3)  # 300mm pig
        pigging_layout.addRow("Pig Length (m):", self.pig_length_spin)
        
        self.pig_density_spin = QDoubleSpinBox()
        self.pig_density_spin.setDecimals(1)
        self.pig_density_spin.setMinimum(10)
        self.pig_density_spin.setMaximum(2000)
        self.pig_density_spin.setValue(300)  # Typical foam pig density
        pigging_layout.addRow("Pig Density (kg/m³):", self.pig_density_spin)
        
        self.pig_friction_spin = QDoubleSpinBox()
        self.pig_friction_spin.setDecimals(2)
        self.pig_friction_spin.setMinimum(0.01)
        self.pig_friction_spin.setMaximum(1)
        self.pig_friction_spin.setValue(0.3)  # Typical friction coefficient
        pigging_layout.addRow("Friction Coefficient:", self.pig_friction_spin)
        
        self.settings_stack.addWidget(pigging_widget)
        
        # 4. Spill Quantification settings
        spill_widget = QWidget()
        spill_layout = QFormLayout(spill_widget)
        
        # Spill settings
        self.spill_type_combo = QComboBox()
        self.spill_type_combo.addItems(["Surface", "Subsurface", "Jet"])
        spill_layout.addRow("Spill Type:", self.spill_type_combo)
        
        self.fluid_type_combo = QComboBox()
        self.fluid_type_combo.addItems(["Crude Oil", "Diesel", "Gasoline", "Natural Gas"])
        spill_layout.addRow("Fluid Type:", self.fluid_type_combo)
        
        self.spill_rate_spin = QDoubleSpinBox()
        self.spill_rate_spin.setDecimals(2)
        self.spill_rate_spin.setMinimum(0.01)
        self.spill_rate_spin.setMaximum(1000)
        self.spill_rate_spin.setValue(10)  # 10 kg/s
        spill_layout.addRow("Spill Rate (kg/s):", self.spill_rate_spin)
        
        self.spill_duration_spin = QDoubleSpinBox()
        self.spill_duration_spin.setDecimals(1)
        self.spill_duration_spin.setMinimum(1)
        self.spill_duration_spin.setMaximum(86400)  # 24 hours in seconds
        self.spill_duration_spin.setValue(3600)  # 1 hour
        spill_layout.addRow("Duration (s):", self.spill_duration_spin)
        
        self.wind_speed_spin = QDoubleSpinBox()
        self.wind_speed_spin.setDecimals(1)
        self.wind_speed_spin.setMinimum(0)
        self.wind_speed_spin.setMaximum(50)
        self.wind_speed_spin.setValue(5)  # 5 m/s
        spill_layout.addRow("Wind Speed (m/s):", self.wind_speed_spin)
        
        self.settings_stack.addWidget(spill_widget)
        
        # Connect stack to radio buttons
        self.sim_type_group.buttonClicked.connect(
            lambda button: self.settings_stack.setCurrentIndex(self.sim_type_group.id(button))
        )
        
        oil_gas_layout.addWidget(self.settings_stack)
        
        # Additional oil & gas related settings
        templates_group = QGroupBox("Templates")
        templates_layout = QVBoxLayout(templates_group)
        
        # Template selection
        template_form = QFormLayout()
        
        self.template_combo = QComboBox()
        self.template_combo.addItems([
            "None", 
            "Pipeline Flow", 
            "Pipeline Network", 
            "Pipeline Riser", 
            "Subsea Equipment"
        ])
        template_form.addRow("Use Template:", self.template_combo)
        
        templates_layout.addLayout(template_form)
        
        # Load template button
        self.load_template_button = QPushButton("Load Template")
        templates_layout.addWidget(self.load_template_button)
        
        oil_gas_layout.addWidget(templates_group)
        
        # Add stretch to push everything to the top
        oil_gas_layout.addStretch()
        
        # Add tab
        self.tab_widget.addTab(oil_gas_widget, "Oil & Gas")
    
    def _setup_control_buttons(self):
        """Set up the simulation control buttons."""
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        
        # Case directory selection
        case_label = QLabel("Case Directory:")
        buttons_layout.addWidget(case_label)
        
        self.case_path_edit = QLineEdit()
        self.case_path_edit.setReadOnly(True)
        buttons_layout.addWidget(self.case_path_edit, 1)
        
        self.browse_case_button = QToolButton()
        self.browse_case_button.setText("...")
        buttons_layout.addWidget(self.browse_case_button)
        
        # Add some spacing
        buttons_layout.addSpacing(20)
        
        # Control buttons
        self.setup_button = QPushButton("Setup Case")
        buttons_layout.addWidget(self.setup_button)
        
        self.run_button = QPushButton("Run")
        buttons_layout.addWidget(self.run_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_button)
        
        self.layout.addWidget(buttons_widget)
    
    def _connect_signals(self):
        """Connect signals to slots"""
        try:
            # Only connect signals for UI elements that exist
            if hasattr(self, 'browse_case_button'):
                self.browse_case_button.clicked.connect(self._browse_case_directory)
            
            if hasattr(self, 'setup_button'):
                self.setup_button.clicked.connect(self._setup_case)
            
            if hasattr(self, 'run_button'):
                self.run_button.clicked.connect(self._run_simulation)
            
            if hasattr(self, 'stop_button'):
                self.stop_button.clicked.connect(self._stop_simulation)
            
            if hasattr(self, 'solver_combo'):
                self.solver_combo.currentTextChanged.connect(self._on_solver_changed)
            
            if hasattr(self, 'fluid_combo'):
                self.fluid_combo.currentIndexChanged.connect(self._update_fluid_properties)
        except Exception as e:
            logger.error(f"Error connecting signals: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _update_ui_state(self):
        """Update UI state based on current status."""
        is_running = self.is_running
        has_case = bool(self.case_dir and os.path.exists(self.case_dir))
        
        # Update button states
        self.run_button.setEnabled(has_case and not is_running)
        self.stop_button.setEnabled(is_running)
        self.setup_button.setEnabled(has_case and not is_running)
        
        # Update tab states
        self.tab_widget.setTabEnabled(0, not is_running)  # Solver tab
        self.tab_widget.setTabEnabled(1, not is_running)  # Parameters tab
        self.tab_widget.setTabEnabled(2, not is_running)  # Monitoring tab
        self.tab_widget.setTabEnabled(4, not is_running)  # Boundary tab
        
        # If running, ensure monitoring tab is visible
        if is_running:
            self.tab_widget.setCurrentIndex(2)
    
    def _browse_case_directory(self):
        """Browse for case directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select OpenFOAM Case Directory",
            os.path.expanduser("~")
        )
        
        if directory:
            # Check if it's a valid OpenFOAM case directory
            if os.path.exists(os.path.join(directory, "system")) and \
               os.path.exists(os.path.join(directory, "constant")):
                self.case_dir = directory
                self.case_path_edit.setText(directory)
                self._update_ui_state()
            else:
                QMessageBox.warning(
                    self,
                    "Invalid Case Directory",
                    "The selected directory does not appear to be a valid OpenFOAM case.\n"
                    "It should contain 'system' and 'constant' directories."
                )
    
    def _setup_case(self):
        """Set up OpenFOAM case."""
        try:
            logger.info("Setting up OpenFOAM case")
            
            # Get case directory
            if not hasattr(self, 'case_dir') or not self.case_dir:
                logger.error("No case directory specified")
                QMessageBox.warning(self, "Error", "Please select a case directory first.")
                return
            
            # Initialize settings dictionary with safe defaults
            settings = {}
            
            # Add simulation type (if attribute exists)
            if hasattr(self, 'single_phase_radio') and self.single_phase_radio.isChecked():
                settings['simulation_type'] = 'singlePhase'
            elif hasattr(self, 'multi_phase_radio') and self.multi_phase_radio.isChecked():
                settings['simulation_type'] = 'multiPhase'
            else:
                # Default to single phase if radio buttons don't exist
                settings['simulation_type'] = 'singlePhase'
                
            # Add solver type (if attribute exists)
            if hasattr(self, 'steady_radio') and self.steady_radio.isChecked():
                settings['solver_type'] = 'steady'
            elif hasattr(self, 'transient_radio') and self.transient_radio.isChecked():
                settings['solver_type'] = 'transient'
            else:
                # Default to steady if radio buttons don't exist
                settings['solver_type'] = 'steady'
                
            # Add turbulence model (if attribute exists)
            if hasattr(self, 'turbulence_combo'):
                settings['turbulence_model'] = self.turbulence_combo.currentText()
            else:
                settings['turbulence_model'] = 'kEpsilon'  # Default
                
            # Add flow model (if attribute exists)
            if hasattr(self, 'laminar_radio') and self.laminar_radio.isChecked():
                settings['flow_model'] = 'laminar'
            elif hasattr(self, 'turbulent_radio') and self.turbulent_radio.isChecked():
                settings['flow_model'] = 'turbulent'
            else:
                settings['flow_model'] = 'turbulent'  # Default
                
            # Add end time (if attribute exists)
            if hasattr(self, 'end_time_spin'):
                settings['end_time'] = self.end_time_spin.value()
            else:
                settings['end_time'] = 1000.0  # Default
                
            # Add time step (if attribute exists)
            if hasattr(self, 'time_step_spin'):
                settings['time_step'] = self.time_step_spin.value()
            else:
                settings['time_step'] = 0.001  # Default
                
            # Set up the case
            if hasattr(self, 'case_manager') and self.case_manager:
                success = self.case_manager.setup_case(self.case_dir)
            elif hasattr(self.main_window, 'case_manager') and self.main_window.case_manager:
                success = self.main_window.case_manager.setup_case(self.case_dir)
            else:
                # If no case manager is available, try to initialize one
                from ..openfoam_integration.case_manager import CaseManager
                self.case_manager = CaseManager(self.main_window, self.case_dir)
                success = self.case_manager.setup_case()
            
            if success:
                logger.info("Successfully set up case")
                QMessageBox.information(self, "Success", "OpenFOAM case set up successfully.")
                
                # Enable run button
                if hasattr(self, 'run_button'):
                    self.run_button.setEnabled(True)
            else:
                logger.error("Failed to set up case")
                QMessageBox.critical(self, "Error", "Failed to set up OpenFOAM case.")
                
        except Exception as e:
            logger.error(f"Error setting up case: {e}")
            import traceback
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Error setting up case: {str(e)}")
    
    def _generate_control_dict(self):
        """Generate the controlDict file."""
        control_dict_path = os.path.join(self.case_dir, "system", "controlDict")
        
        # Get settings from UI
        start_time = self.start_time_spin.value()
        end_time = self.end_time_spin.value()
        time_step = self.time_step_spin.value()
        write_interval = self.write_interval_spin.value()
        output_format = self.output_format_combo.currentText()
        
        # Create content
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/

FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     {self.solver};

startFrom       startTime;

startTime       {start_time};

stopAt          endTime;

endTime         {end_time};

deltaT          {time_step};

writeControl    runTime;

writeInterval   {write_interval};

purgeWrite      0;

writeFormat     {output_format};

writePrecision  8;

writeCompression {'on' if self.compression_check.isChecked() else 'off'};

timeFormat      general;

timePrecision   6;

runTimeModifiable true;

// Adjustable time step settings
{'adjustTimeStep yes;' if self.adjustable_time_check.isChecked() else '// adjustTimeStep no;'}
{'maxCo ' + str(self.max_co_spin.value()) + ';' if self.adjustable_time_check.isChecked() else ''}

// ************************************************************************* //
"""
        
        # Write to file
        with open(control_dict_path, 'w') as f:
            f.write(content)
    
    def _generate_solver_dict(self):
        """Generate the fvSolution dictionary file."""
        solution_dict_path = os.path.join(self.case_dir, "system", "fvSolution")
        
        # Get settings from UI
        tolerance = self.tolerance_spin.value()
        iterations = self.iterations_spin.value()
        p_relax = self.p_relax_spin.value()
        u_relax = self.u_relax_spin.value()
        
        # Create content (simplified, would need to be adapted for each solver)
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/

FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{{
    p
    {{
        solver          GAMG;
        tolerance       {tolerance};
        relTol          0.01;
        smoother        GaussSeidel;
        nPreSweeps      0;
        nPostSweeps     2;
        cacheAgglomeration on;
        agglomerator    faceAreaPair;
        nCellsInCoarsestLevel 10;
        mergeLevels     1;
        maxIter         100;
    }}

    "(U|k|epsilon|omega|R)"
    {{
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       {tolerance};
        relTol          0.1;
        nSweeps         1;
    }}
}}

SIMPLE
{{
    nNonOrthogonalCorrectors 0;
    consistent      yes;

    residualControl
    {{
        p               {tolerance};
        U               {tolerance};
        "(k|epsilon|omega|R)" {tolerance};
    }}
}}

PISO
{{
    nCorrectors     2;
    nNonOrthogonalCorrectors 1;
    pRefCell        0;
    pRefValue       0;
}}

relaxationFactors
{{
    equations
    {{
        p               {p_relax};
        U               {u_relax};
        k               {u_relax};
        epsilon         {u_relax};
        R               {u_relax};
        nuTilda         {u_relax};
    }}
}}

// ************************************************************************* //
"""
        
        # Write to file
        with open(solution_dict_path, 'w') as f:
            f.write(content)

    def _generate_solution_dict(self):
        """Alias for _generate_solver_dict for backward compatibility."""
        return self._generate_solver_dict()
    
    def _generate_schemes_dict(self):
        """Generate the fvSchemes dictionary file."""
        schemes_dict_path = os.path.join(self.case_dir, "system", "fvSchemes")
        
        # Get settings from UI
        gradient_scheme = self.gradient_combo.currentText()
        divergence_scheme = self.divergence_combo.currentText()
        laplacian_scheme = self.laplacian_combo.currentText()
        interpolation_scheme = self.interpolation_combo.currentText()
        
        # Create content
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/

FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{{
    default         Euler;
}}

gradSchemes
{{
    default         {gradient_scheme};
}}

divSchemes
{{
    default         none;
    div(phi,U)      {divergence_scheme};
    div(phi,k)      {divergence_scheme};
    div(phi,epsilon) {divergence_scheme};
    div(phi,omega)  {divergence_scheme};
    div(phi,R)      {divergence_scheme};
    div(R)          {divergence_scheme};
    div((nuEff*dev2(T(grad(U))))) {divergence_scheme};
}}

laplacianSchemes
{{
    default         {laplacian_scheme};
}}

interpolationSchemes
{{
    default         {interpolation_scheme};
}}

snGradSchemes
{{
    default         corrected;
}}

// ************************************************************************* //
"""
        
        # Write to file
        with open(schemes_dict_path, 'w') as f:
            f.write(content)
    
    def _generate_transport_properties(self):
        """Generate the transportProperties dictionary file."""
        transport_dict_path = os.path.join(self.case_dir, "constant", "transportProperties")
        
        # Create content based on simulation type
        if self.single_phase_radio.isChecked():
            # Single phase properties
            density = self.density_spin.value()
            viscosity = self.viscosity_spin.value()
            
            content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/

FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

transportModel  Newtonian;

// Kinematic viscosity
nu              {viscosity / density};

// Density
rho             {density};

// ************************************************************************* //
"""
        elif self.multi_phase_radio.isChecked():
            # Multi-phase properties
            surface_tension = self.surface_tension_spin.value()
            
            content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/

FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

phases
(
    water
    {{
        transportModel  Newtonian;
        nu              1.0e-6;
        rho             1000;
    }}

    oil
    {{
        transportModel  Newtonian;
        nu              3.0e-5;
        rho             850;
    }}
);

sigma            {surface_tension};

// ************************************************************************* //
"""
        elif self.pigging_radio.isChecked() or self.spill_radio.isChecked():
            # These would require more specialized templates
            # For this example, we'll use a placeholder
            content = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/

FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Specialized properties for advanced simulation
// Would need to be customized based on specific requirements

// ************************************************************************* //
"""
        
        # Write to file
        with open(transport_dict_path, 'w') as f:
            f.write(content)
    
    def _generate_decomp_dict(self):
        """Generate the decomposeParDict file for parallel runs."""
        decomp_dict_path = os.path.join(self.case_dir, "system", "decomposeParDict")
        
        # Get settings from UI
        processors = self.processors_spin.value()
        method = self.decomp_combo.currentText()
        
        # Create content
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/

FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

numberOfSubdomains {processors};

method          {method};

"""
        
        # Add method-specific settings
        if method == "simple":
            content += f"""
simpleCoeffs
{{
    n               ({processors} 1 1);
    delta           0.001;
}}
"""
        elif method == "hierarchical":
            content += f"""
hierarchicalCoeffs
{{
    n               (2 2 1);
    delta           0.001;
    order           xyz;
}}
"""
        
        content += "\n// ************************************************************************* //\n"
        
        # Write to file
        with open(decomp_dict_path, 'w') as f:
            f.write(content)
    
    def _on_solver_changed(self, solver: str):
        """
        Handle solver change event.
        
        Args:
            solver (str): The selected solver
        """
        self.solver = solver
        
        # Update UI based on solver type
        if solver in ["simpleFoam"]:
            # Steady-state solvers
            self.steady_radio.setChecked(True)
            self.time_step_spin.setEnabled(False)
            self.adjustable_time_check.setEnabled(False)
            self.max_co_spin.setEnabled(False)
        else:
            # Transient solvers
            self.transient_radio.setChecked(True)
            self.time_step_spin.setEnabled(True)
            self.adjustable_time_check.setEnabled(True)
            self.max_co_spin.setEnabled(self.adjustable_time_check.isChecked())
        
        # Update multi-phase options
        is_multiphase = solver in ["interFoam", "multiphaseInterFoam"]
        if is_multiphase:
            self.multi_phase_radio.setChecked(True)
            self.settings_stack.setCurrentIndex(1)
    
    def _update_fluid_properties(self, index: int):
        """
        Update fluid properties based on selection.
        
        Args:
            index (int): The index of the selected fluid
        """
        fluid = self.fluid_combo.currentText()
        
        # Set default properties based on fluid type
        if fluid == "Water":
            self.density_spin.setValue(1000)
            self.viscosity_spin.setValue(0.001)
        elif fluid == "Oil":
            self.density_spin.setValue(850)
            self.viscosity_spin.setValue(0.03)
        elif fluid == "Gas":
            self.density_spin.setValue(1.2)
            self.viscosity_spin.setValue(1.8e-5)
        # Custom values are entered by the user
    
    def _run_simulation(self):
        """Run the OpenFOAM simulation with comprehensive setup and memory monitoring."""
        if not self.case_dir:
            QMessageBox.warning(
                self,
                "No Case Directory",
                "Please set up a case directory before running the simulation."
            )
            return
        
        try:
            # === PRE-RUN SETUP AND FIXES ===
            # Fix pressure reference before running
            self._fix_pressure_reference()
            
            # Ensure turbulence properties file exists
            self._ensure_turbulence_properties()
            
            # Ensure turbulence fields exist
            self._ensure_turbulence_fields()
            
            # Apply additional fixes for turbulence fields
            self._fix_turbulence_fields()
            
            # Fix fvOptions file
            self._fix_fv_options()
            
            # Verify and fix boundary conditions
            self._verify_boundary_conditions()
            self._fix_turbulence_boundary_conditions()
            self._fix_field_boundaries_directly()
            self._fix_field_files_complete()
            
            # Add after self._fix_field_files_complete()
            self._fix_numerical_stability()

            self._ensure_thermophysical_properties()

            self._ensure_compressible_fields()
            self._fix_pressure_dimensions()

            # === UI PREPARATION ===
            # Clear previous outputs
            self.residual_plot.clear()
            self.log_text.clear()
            
            # Reset progress indicators
            self.status_label.setText("Running")
            self.iteration_label.setText("0")
            self.time_label.setText("0.0")
            self.progress_bar.setValue(0)
            
            # === START MEMORY MONITORING ===
            self._start_memory_monitor()
            
            # === SET UP SIMULATION PARAMETERS ===
            # Get solver and settings
            solver = self.solver_combo.currentText()
            parallel = self.parallel_check.isChecked()
            processors = self.processors_spin.value() if parallel else 1
            
            # Log simulation parameters
            self.log_update.emit(f"Starting simulation with {solver}")
            self.log_update.emit(f"Case directory: {self.case_dir}")
            self.log_update.emit(f"Parallel mode: {parallel} with {processors} processors")
            
            # === CREATE AND START SIMULATION RUNNER ===
            # Create the simulation runner
            self.runner = SimulationRunner(
                solver=solver,
                case_dir=self.case_dir,
                parallel=parallel,
                processors=processors
            )
            
            # Connect signals
            self.runner.progress_update.connect(self._on_progress_update)
            self.runner.status_update.connect(self._on_status_update)
            self.runner.simulation_finished.connect(self._on_simulation_finished)
            self.runner.log_update.connect(self._on_log_update)
            
            # Add this connection if the signal exists in SimulationRunner
            if hasattr(self.runner, 'simulation_memory_exceeded'):
                self.runner.simulation_memory_exceeded.connect(self._on_memory_exceeded)
            
            # Update UI state
            self.is_running = True
            self._update_ui_state()
            
            # Log the exact command
            cmd = self._prepare_simulation_command()
            self.log_update.emit(f"Running command: {' '.join(cmd)}")
            
            # Start the simulation
            self.runner.start()
            
            # Emit started signal
            self.simulation_started.emit()
            
        except Exception as e:
            logger.error(f"Error starting simulation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            QMessageBox.critical(
                self,
                "Error Starting Simulation",
                f"An error occurred while starting the simulation:\n\n{str(e)}"
            )
            
            # Update UI
            self.is_running = False
            self._update_ui_state()

    def _fix_numerical_stability(self):
        """Fix numerical stability issues for OpenFOAM simulation."""
        try:
            import os
            import re
            
            # Get the solver type
            solver = self.solver_combo.currentText()
            is_compressible = any(s in solver for s in ["rhoPimpleFoam", "rhoSimpleFoam", "reactingFoam"])
            
            self.log_update.emit(f"Setting up numerical schemes for {solver}")
            
            # 1. Fix fvSchemes for more stable schemes
            schemes_path = os.path.join(self.case_dir, "system", "fvSchemes")
            
            # Base schemes applicable to all solvers
            schemes_content = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{
"""
            
            # Adjust ddtSchemes based on solver
            if "pimple" in solver.lower() or "piso" in solver.lower():
                schemes_content += """    default         Euler;
}
"""
            else:
                schemes_content += """    default         steadyState;
}
"""
            
            # Common gradient schemes
            schemes_content += """
gradSchemes
{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1.0;
"""
            
            # Add compressible-specific gradient schemes if needed
            if is_compressible:
                schemes_content += """    grad(e)         cellLimited Gauss linear 1.0;
    grad(h)         cellLimited Gauss linear 1.0;
    grad(T)         cellLimited Gauss linear 1.0;
"""
            
            schemes_content += """}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div(phi,epsilon) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
"""
            
            # Add compressible-specific divergence schemes
            if is_compressible:
                schemes_content += """    div(phi,T)      bounded Gauss upwind;
    div(phi,h)      bounded Gauss upwind;
    div(phi,e)      bounded Gauss upwind;
    div(phi,K)      bounded Gauss upwind;
    div(phi,Ekp)    bounded Gauss upwind;
    div(phid,p)     bounded Gauss upwind;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
"""
            
            schemes_content += """}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
"""
            
            # Add compressible-specific interpolation schemes
            if is_compressible:
                schemes_content += """    interpolate(rho) linear;
"""
            
            schemes_content += """}

snGradSchemes
{
    default         corrected;
}

wallDist
{
    method meshWave;
}

// ************************************************************************* //
"""
            with open(schemes_path, "w") as f:
                f.write(schemes_content)
            self.log_update.emit("Updated fvSchemes for better stability")
            
            # 2. Fix fvSolution for more stable solver settings
            solution_path = os.path.join(self.case_dir, "system", "fvSolution")
            
            # Start with the common solvers section
            solution_content = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-6;
        relTol          0.01;
        smoother        GaussSeidel;
        nPreSweeps      0;
        nPostSweeps     2;
        cacheAgglomeration on;
        agglomerator    faceAreaPair;
        nCellsInCoarsestLevel 10;
        mergeLevels     1;
    }
    
    pFinal
    {
        $p;
        relTol          0.001;
    }

    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-6;
        relTol          0.01;
        nSweeps         2;
    }
    
    UFinal
    {
        $U;
        relTol          0.001;
    }

    k
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-6;
        relTol          0.01;
        nSweeps         2;
    }
    
    kFinal
    {
        $k;
        relTol          0.001;
    }

    omega
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-6;
        relTol          0.01;
        nSweeps         2;
    }
    
    omegaFinal
    {
        $omega;
        relTol          0.001;
    }
"""
            
            # Add compressible-specific solvers if needed
            if is_compressible:
                solution_content += """
    // Compressible flow solvers
    rho
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-8;
        relTol          0.01;
    }
    
    rhoFinal
    {
        $rho;
        relTol          0.001;
    }
    
    T
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-8;
        relTol          0.01;
    }
    
    TFinal
    {
        $T;
        relTol          0.001;
    }
    
    h
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-8;
        relTol          0.01;
    }
    
    hFinal
    {
        $h;
        relTol          0.001;
    }
"""
            
            solution_content += """
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
    
    residualControl
    {
        p               1e-3;
        U               1e-4;
        k               1e-4;
        omega           1e-4;
"""
            
            # Add compressible-specific residual controls
            if is_compressible:
                solution_content += """        T               1e-4;
        h               1e-4;
"""
            
            solution_content += """    }
    
    pRefCell        0;
    pRefValue       0;
}

PISO
{
    nCorrectors     2;
    nNonOrthogonalCorrectors 1;
    pRefCell        0;
    pRefValue       0;
}
"""

            # Add PIMPLE section for transient solvers
            if "pimple" in solver.lower():
                solution_content += """
PIMPLE
{
    nOuterCorrectors    1;
    nCorrectors         2;
    nNonOrthogonalCorrectors 1;
    pRefCell        0;
    pRefValue       0;
    
    // More conservative under-relaxation for compressible solvers
    URF
    {
        p               0.3;
        U               0.7;
        rho             0.05;
        h               0.3;
        T               0.3;
    }

    // To handle potential initial transient instabilities
    transonic       yes;
    consistent      yes;
    correctPhi      yes;
}
"""

            solution_content += """
relaxationFactors
{
    equations
    {
        U               0.7;
        k               0.7;
        omega           0.7;
        epsilon         0.7;
"""

            # Add compressible-specific relaxation factors
            if is_compressible:
                solution_content += """        T               0.5;
        h               0.5;
        rho             0.2;
"""
            
            solution_content += """    }
    
    fields
    {
        p               0.3;
    }
}

// ************************************************************************* //
"""
            with open(solution_path, "w") as f:
                f.write(solution_content)
            self.log_update.emit("Updated fvSolution for better stability")
            
            # 3. Fix initial fields for compressible solvers
            if is_compressible:
                # Set more stable initial values for fields
                # Ensure reasonably high initial pressure to avoid div/0
                p_path = os.path.join(self.case_dir, "0", "p")
                if os.path.exists(p_path):
                    with open(p_path, 'r') as f:
                        p_content = f.read()
                    
                    # Set initial pressure to 1e5 (atmospheric)
                    p_content = re.sub(r'internalField\s+uniform\s+[0-9.e+-]+', 
                                    'internalField   uniform 100000', 
                                    p_content)
                    
                    with open(p_path, 'w') as f:
                        f.write(p_content)
                    self.log_update.emit("Set stable initial pressure for compressible solver")
                
                # Set a reasonable initial T
                T_path = os.path.join(self.case_dir, "0", "T")
                if os.path.exists(T_path):
                    with open(T_path, 'r') as f:
                        T_content = f.read()
                    
                    # Set initial temperature to 300K
                    T_content = re.sub(r'internalField\s+uniform\s+[0-9.e+-]+', 
                                    'internalField   uniform 300', 
                                    T_content)
                    
                    with open(T_path, 'w') as f:
                        f.write(T_content)
                    self.log_update.emit("Set stable initial temperature for compressible solver")
                
                # Fix U (velocity) field - reduce velocity magnitude to avoid divergence
                u_path = os.path.join(self.case_dir, "0", "U")
                if os.path.exists(u_path):
                    with open(u_path, 'r') as f:
                        content = f.read()
                    
                    # Set initial velocity to zero or nearly zero
                    content = re.sub(r'internalField\s+uniform\s+\(\s*[0-9.e+-]+\s+[0-9.e+-]+\s+[0-9.e+-]+\s*\)', 
                                'internalField   uniform (0.001 0 0)', 
                                content)
                    
                    # Reduce inlet velocity if present
                    if "inlet" in content.lower():
                        # Make inlet velocity small but not zero
                        content = re.sub(r'value\s+uniform\s+\(\s*([0-9.e+-]+)\s+([0-9.e+-]+)\s+([0-9.e+-]+)\s*\)', 
                                    r'value       uniform (0.1 0 0)', 
                                    content)
                    
                    with open(u_path, 'w') as f:
                        f.write(content)
                    self.log_update.emit("Set stable initial velocity for compressible solver")
            else:
                # For incompressible solvers, just reduce velocity
                u_path = os.path.join(self.case_dir, "0", "U")
                if os.path.exists(u_path):
                    # Read current U file
                    with open(u_path, 'r') as f:
                        content = f.read()
                    
                    # Reduce inlet velocity if present
                    if "inlet" in content.lower():
                        # Look for patterns like "uniform (1 0 0)" and replace with "uniform (0.1 0 0)"
                        modified_content = re.sub(r'uniform\s*\(\s*([0-9.e+-]+)\s+([0-9.e+-]+)\s+([0-9.e+-]+)\s*\)', 
                                            r'uniform (0.1 0 0)', 
                                            content)
                        
                        # Write back the modified file
                        with open(u_path, 'w') as f:
                            f.write(modified_content)
                        self.log_update.emit("Lowered inlet velocity for better initial stability")
                
            return True
                
        except Exception as e:
            import traceback
            self.log_update.emit(f"Error fixing numerical stability: {str(e)}")
            logger.error(f"Error in _fix_numerical_stability: {e}")
            logger.error(traceback.format_exc())
            return False

            
    
    def _fix_field_files_complete(self):
        """Completely rewrite turbulence field files with correct boundary conditions."""
        try:
            import os
            import re
            
            # Get the actual boundary names from the mesh file
            boundary_file = os.path.join(self.case_dir, "constant", "polyMesh", "boundary")
            if not os.path.exists(boundary_file):
                self.log_update.emit("Error: Cannot find boundary file")
                return
                    
            # Extract boundary names using regex
            with open(boundary_file, 'r') as f:
                content = f.read()
                boundary_matches = re.findall(r'^\s*(\w+)\s*\n\s*{', content, re.MULTILINE)
                boundaries = [b for b in boundary_matches if b not in ['FoamFile']]
                
            self.log_update.emit(f"Found boundaries: {', '.join(boundaries)}")
            
            # Create correct k file
            k_path = os.path.join(self.case_dir, "0", "k")
            if os.path.exists(k_path):
                k_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
    | =========                 |                                                 |
    | \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
    |  \\\\    /   O peration     | Version:  v2312                                 |
    |   \\\\  /    A nd           | Website:  [www.openfoam.com](www.openfoam.com)                      |
    |    \\\\/     M anipulation  |                                                 |
    \\*---------------------------------------------------------------------------*/
    FoamFile
    {{
        version     2.0;
        format      ascii;
        class       volScalarField;
        object      k;
    }}
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    
    dimensions      [0 2 -2 0 0 0 0];
    
    internalField   uniform 0.1;
    
    boundaryField
    {{
    """
                # Add boundary conditions for each boundary
                for boundary in boundaries:
                    if "inlet" in boundary.lower():
                        k_content += f"""    {boundary}
        {{
            type            fixedValue;
            value           uniform 0.1;
        }}
        
    """
                    elif "outlet" in boundary.lower():
                        k_content += f"""    {boundary}
        {{
            type            zeroGradient;
        }}
        
    """
                    elif "wall" in boundary.lower():
                        k_content += f"""    {boundary}
        {{
            type            kqRWallFunction;
            value           uniform 0.1;
        }}
        
    """
                    else:
                        k_content += f"""    {boundary}
        {{
            type            zeroGradient;
        }}
        
    """
                
                k_content += """}
    
    // ************************************************************************* //"""
                
                # Write the file
                with open(k_path, 'w') as f:
                    f.write(k_content)
                    
                self.log_update.emit("Rewrote k field file with correct boundaries")
            
            # Create correct epsilon file
            epsilon_path = os.path.join(self.case_dir, "0", "epsilon")
            if os.path.exists(epsilon_path):
                epsilon_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
    | =========                 |                                                 |
    | \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
    |  \\\\    /   O peration     | Version:  v2312                                 |
    |   \\\\  /    A nd           | Website:  [www.openfoam.com](www.openfoam.com)                      |
    |    \\\\/     M anipulation  |                                                 |
    \\*---------------------------------------------------------------------------*/
    FoamFile
    {{
        version     2.0;
        format      ascii;
        class       volScalarField;
        object      epsilon;
    }}
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    
    dimensions      [0 2 -3 0 0 0 0];
    
    internalField   uniform 0.01;
    
    boundaryField
    {{
    """
                # Add boundary conditions for each boundary
                for boundary in boundaries:
                    if "inlet" in boundary.lower():
                        epsilon_content += f"""    {boundary}
        {{
            type            fixedValue;
            value           uniform 0.01;
        }}
        
    """
                    elif "outlet" in boundary.lower():
                        epsilon_content += f"""    {boundary}
        {{
            type            zeroGradient;
        }}
        
    """
                    elif "wall" in boundary.lower():
                        epsilon_content += f"""    {boundary}
        {{
            type            epsilonWallFunction;
            value           uniform 0.01;
        }}
        
    """
                    else:
                        epsilon_content += f"""    {boundary}
        {{
            type            zeroGradient;
        }}
        
    """
                
                epsilon_content += """}
    
    // ************************************************************************* //"""
                
                # Write the file
                with open(epsilon_path, 'w') as f:
                    f.write(epsilon_content)
                    
                self.log_update.emit("Rewrote epsilon field file with correct boundaries")
                
            # Create correct omega file
            omega_path = os.path.join(self.case_dir, "0", "omega")
            if os.path.exists(omega_path):
                omega_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
    | =========                 |                                                 |
    | \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
    |  \\\\    /   O peration     | Version:  v2312                                 |
    |   \\\\  /    A nd           | Website:  [www.openfoam.com](www.openfoam.com)                      |
    |    \\\\/     M anipulation  |                                                 |
    \\*---------------------------------------------------------------------------*/
    FoamFile
    {{
        version     2.0;
        format      ascii;
        class       volScalarField;
        object      omega;
    }}
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    
    dimensions      [0 0 -1 0 0 0 0];
    
    internalField   uniform 0.1;
    
    boundaryField
    {{
    """
                # Add boundary conditions for each boundary
                for boundary in boundaries:
                    if "inlet" in boundary.lower():
                        omega_content += f"""    {boundary}
        {{
            type            fixedValue;
            value           uniform 0.1;
        }}
        
    """
                    elif "outlet" in boundary.lower():
                        omega_content += f"""    {boundary}
        {{
            type            zeroGradient;
        }}
        
    """
                    elif "wall" in boundary.lower():
                        omega_content += f"""    {boundary}
        {{
            type            omegaWallFunction;
            value           uniform 0.1;
        }}
        
    """
                    else:
                        omega_content += f"""    {boundary}
        {{
            type            zeroGradient;
        }}
        
    """
                
                omega_content += """}
    
    // ************************************************************************* //"""
                
                # Write the file
                with open(omega_path, 'w') as f:
                    f.write(omega_content)
                    
                self.log_update.emit("Rewrote omega field file with correct boundaries")
                
            # Create correct nut file
            nut_path = os.path.join(self.case_dir, "0", "nut")
            if os.path.exists(nut_path):
                nut_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
    | =========                 |                                                 |
    | \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
    |  \\\\    /   O peration     | Version:  v2312                                 |
    |   \\\\  /    A nd           | Website:  [www.openfoam.com](www.openfoam.com)                      |
    |    \\\\/     M anipulation  |                                                 |
    \\*---------------------------------------------------------------------------*/
    FoamFile
    {{
        version     2.0;
        format      ascii;
        class       volScalarField;
        object      nut;
    }}
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    
    dimensions      [0 2 -1 0 0 0 0];
    
    internalField   uniform 0;
    
    boundaryField
    {{
    """
                # Add boundary conditions for each boundary
                for boundary in boundaries:
                    if "wall" in boundary.lower():
                        nut_content += f"""    {boundary}
        {{
            type            nutkWallFunction;
            value           uniform 0;
        }}
        
    """
                    else:
                        nut_content += f"""    {boundary}
        {{
            type            calculated;
            value           uniform 0;
        }}
        
    """
                
                nut_content += """}
    
    // ************************************************************************* //"""
                
                # Write the file
                with open(nut_path, 'w') as f:
                    f.write(nut_content)
                    
                self.log_update.emit("Rewrote nut field file with correct boundaries")
                
        except Exception as e:
            import traceback
            self.log_update.emit(f"Error fixing field files: {str(e)}")
            logger.error(traceback.format_exc())

    
    def _fix_field_boundaries_directly(self):
        """Fix turbulence field boundaries by directly editing the files."""
        try:
            import os
            import re
            
            # Get the actual boundary names from the mesh file
            boundary_file = os.path.join(self.case_dir, "constant", "polyMesh", "boundary")
            if not os.path.exists(boundary_file):
                self.log_update.emit("Error: Cannot find boundary file")
                return
                
            # Extract boundary names using regex (more reliable than dictionary parsing)
            with open(boundary_file, 'r') as f:
                content = f.read()
                # Look for boundary entries in the format "name { ... }"
                boundary_matches = re.findall(r'^\s*(\w+)\s*\n\s*{', content, re.MULTILINE)
                boundaries = [b for b in boundary_matches if b not in ['FoamFile']]
            
            self.log_update.emit(f"Found boundaries: {', '.join(boundaries)}")
            
            # Fields to fix
            fields = ["k", "epsilon", "omega", "nut"]
            
            for field in fields:
                field_path = os.path.join(self.case_dir, "0", field)
                if not os.path.exists(field_path):
                    continue
                    
                self.log_update.emit(f"Fixing {field} boundary conditions")
                
                # Read the file
                with open(field_path, 'r') as f:
                    content = f.read()
                    
                # Extract the boundaryField section
                match = re.search(r'boundaryField\s*\n\s*{([^}]*)}', content, re.DOTALL)
                if not match:
                    continue
                    
                boundary_field = match.group(1)
                
                # Check for each boundary
                new_boundary_field = boundary_field
                for boundary in boundaries:
                    if not re.search(rf'{re.escape(boundary)}\s*\n\s*{{', new_boundary_field, re.DOTALL):
                        # Add the missing boundary condition based on boundary name and field type
                        if "inlet" in boundary.lower():
                            if field == "k":
                                bc = f"""
        {boundary}
        {{
            type            fixedValue;
            value           uniform 0.1;
        }}"""
                            elif field == "epsilon":
                                bc = f"""
        {boundary}
        {{
            type            fixedValue;
                            value           uniform 0.1;
        }}"""
                            elif field == "omega":
                                bc = f"""
        {boundary}
        {{
            type            fixedValue;
            value           uniform 1.0;
        }}"""
                            else:  # nut
                                bc = f"""
        {boundary}
        {{
            type            calculated;
            value           uniform 0;
        }}"""
                        elif "outlet" in boundary.lower():
                            bc = f"""
        {boundary}
        {{
            type            zeroGradient;
        }}"""
                        elif "wall" in boundary.lower():
                            if field == "k":
                                bc = f"""
        {boundary}
        {{
            type            kqRWallFunction;
            value           uniform 0.1;
        }}"""
                            elif field == "epsilon":
                                bc = f"""
        {boundary}
        {{
            type            epsilonWallFunction;
            value           uniform 0.1;
        }}"""
                            elif field == "omega":
                                bc = f"""
        {boundary}
        {{
            type            omegaWallFunction;
            value           uniform 1.0;
        }}"""
                            else:  # nut
                                bc = f"""
        {boundary}
        {{
            type            nutkWallFunction;
            value           uniform 0;
        }}"""
                        else:
                            # Default for any other boundary
                            bc = f"""
        {boundary}
        {{
            type            zeroGradient;
        }}"""
                        
                        new_boundary_field += bc
                        
                # Replace the boundaryField section
                new_content = re.sub(r'boundaryField\s*\n\s*{[^}]*}', f'boundaryField\n{{\n{new_boundary_field}\n}}', content, flags=re.DOTALL)
                
                # Write back to the file
                with open(field_path, 'w') as f:
                    f.write(new_content)
                    
                self.log_update.emit(f"Fixed {field} boundary conditions")
                
        except Exception as e:
            import traceback
            self.log_update.emit(f"Error fixing field boundaries: {str(e)}")
            logger.error(traceback.format_exc())


    def _fix_turbulence_boundary_conditions(self):
        """Fix boundary conditions for turbulence fields (k, epsilon, omega, nut)."""
        try:
            import os
            from ..openfoam_integration.case_manager import CaseDictManager
            
            # Get boundary names from the mesh
            boundary_file = os.path.join(self.case_dir, "constant", "polyMesh", "boundary")
            if not os.path.exists(boundary_file):
                self.log_update.emit("Warning: No boundary file found")
                return
                
            # Try to read boundary names
            dict_manager = CaseDictManager(boundary_file)
            boundaries = list(dict_manager.get_root_keys())
            
            # Fields to check
            fields = ["k", "epsilon", "omega", "nut"]
            
            # Check each field file
            for field in fields:
                field_path = os.path.join(self.case_dir, "0", field)
                if not os.path.exists(field_path):
                    continue
                    
                self.log_update.emit(f"Checking turbulence field: {field}")
                
                # Read file
                field_manager = CaseDictManager(field_path)
                
                # Make sure boundaryField exists
                if not field_manager.has_subdict("boundaryField"):
                    field_manager.add_subdict("boundaryField", {})
                    
                # Check each boundary
                for boundary in boundaries:
                    bc_path = f"boundaryField/{boundary}"
                    
                    if not field_manager.has_subdict(bc_path):
                        # Add appropriate boundary condition based on field type
                        if field == "k":
                            if "wall" in boundary.lower():
                                field_manager.add_subdict(bc_path, {
                                    "type": "kqRWallFunction",
                                    "value": "uniform 0.1"
                                })
                            elif "inlet" in boundary.lower():
                                field_manager.add_subdict(bc_path, {
                                    "type": "fixedValue",
                                    "value": "uniform 0.1"
                                })
                            else:
                                field_manager.add_subdict(bc_path, {
                                    "type": "zeroGradient"
                                })
                        elif field == "epsilon":
                            if "wall" in boundary.lower():
                                field_manager.add_subdict(bc_path, {
                                    "type": "epsilonWallFunction",
                                    "value": "uniform 0.1"
                                })
                            elif "inlet" in boundary.lower():
                                field_manager.add_subdict(bc_path, {
                                    "type": "fixedValue",
                                    "value": "uniform 0.1"
                                })
                            else:
                                field_manager.add_subdict(bc_path, {
                                    "type": "zeroGradient"
                                })
                        elif field == "omega":
                            if "wall" in boundary.lower():
                                field_manager.add_subdict(bc_path, {
                                    "type": "omegaWallFunction",
                                    "value": "uniform 1.0"
                                })
                            elif "inlet" in boundary.lower():
                                field_manager.add_subdict(bc_path, {
                                    "type": "fixedValue",
                                    "value": "uniform 1.0"
                                })
                            else:
                                field_manager.add_subdict(bc_path, {
                                    "type": "zeroGradient"
                                })
                        elif field == "nut":
                            if "wall" in boundary.lower():
                                field_manager.add_subdict(bc_path, {
                                    "type": "nutkWallFunction",
                                    "value": "uniform 0"
                                })
                            else:
                                field_manager.add_subdict(bc_path, {
                                    "type": "calculated",
                                    "value": "uniform 0"
                                })
                
                # Write changes
                field_manager.write()
                self.log_update.emit(f"Fixed {field} boundary conditions")
                
        except Exception as e:
            import traceback
            self.log_update.emit(f"Error fixing turbulence boundary conditions: {str(e)}")
            logger.error(traceback.format_exc())

    def _start_memory_monitor(self):
        """Start a thread to monitor memory usage of the simulation process."""
        self.memory_monitor_active = True
        self.memory_thread = threading.Thread(target=self._monitor_memory_usage)
        self.memory_thread.daemon = True
        self.memory_thread.start()

    def _monitor_memory_usage(self):
        """Monitor memory usage of the process and stop if it exceeds limits."""
        import psutil
        import time
        
        # Get the system memory
        total_memory = psutil.virtual_memory().total
        # Set threshold to 80% of total memory
        memory_threshold = 0.8 * total_memory
        
        self.log_update.emit(f"Memory monitor active: Threshold set to {memory_threshold/1024/1024:.1f} MB")
        
        while self.memory_monitor_active and hasattr(self, 'runner') and self.runner is not None:
            try:
                # If runner is still running
                if self.runner.isRunning():
                    # Get OpenFOAM process object from SimulationRunner if available
                    if hasattr(self.runner, 'process') and self.runner.process is not None:
                        try:
                            process = psutil.Process(self.runner.process.pid)
                            # Get memory info for the process and its children
                            memory_info = process.memory_info().rss
                            for child in process.children(recursive=True):
                                try:
                                    memory_info += child.memory_info().rss
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                            
                            # Log memory usage every 5 seconds
                            if time.time() % 5 < 1:
                                self.log_update.emit(f"Memory usage: {memory_info/1024/1024:.1f} MB")
                            
                            # Check if memory exceeds threshold
                            if memory_info > memory_threshold:
                                self.log_update.emit(f"WARNING: Memory usage exceeded threshold ({memory_info/1024/1024:.1f} MB)")
                                self.log_update.emit("Attempting graceful shutdown of simulation to prevent OS kill...")
                                
                                # Try to send signal to allow process to clean up
                                process.terminate()
                                
                                # Give it 5 seconds to terminate
                                terminated = process.wait(5)
                                if not terminated:
                                    self.log_update.emit("Process did not terminate gracefully, forcing kill...")
                                    process.kill()
                                    
                                self.log_update.emit("Simulation stopped due to high memory usage.")
                                # Emit signal or call handler directly
                                self._on_memory_exceeded()
                                break
                                
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            # Process no longer exists or can't be accessed
                            break
                else:
                    # Runner has stopped
                    break
            except Exception as e:
                self.log_update.emit(f"Error in memory monitor: {e}")
                break
                    
            # Sleep for a short time
            time.sleep(0.5)
        
        self.memory_monitor_active = False

    def _monitor_process_output(self):
        """
        Monitor the output of the simulation process in a non-blocking way.
        This implementation uses QTimer to periodically check for output
        rather than blocking the UI thread.
        """
        # We won't directly use this method for the SimulationRunner approach
        # since the SimulationRunner class handles output reading internally
        # But we'll keep it for compatibility with direct subprocess approaches
        
        if not hasattr(self, 'process') or not self.process:
            logger.warning("No process to monitor")
            return
        
        # Create a timer to periodically read process output without blocking
        self.output_timer = QTimer()
        self.output_timer.timeout.connect(self._read_process_output)
        self.output_timer.start(100)  # Check output every 100ms
        
        # Also create a timer to periodically check if the process is still running
        self.process_timer = QTimer()
        self.process_timer.timeout.connect(self._check_process_status)
        self.process_timer.start(500)  # Check process status every 500ms

    def _read_process_output(self):
        """Read available output from the process without blocking."""
        if not hasattr(self, 'process') or not self.process:
            if hasattr(self, 'output_timer'):
                self.output_timer.stop()
            return
        
        try:
            # Try to read stdout
            if hasattr(self.process, 'stdout') and self.process.stdout:
                import select
                while True:
                    # Use select to check if data is available without blocking
                    readable, _, _ = select.select([self.process.stdout], [], [], 0)
                    if not readable:
                        break  # No data available
                    
                    line = self.process.stdout.readline().decode('utf-8', errors='replace')
                    if not line:
                        break  # End of stream
                    
                    # Process the line
                    self.log_update.emit(line)
                    self._parse_output_line(line)
        
            # Try to read stderr
            if hasattr(self.process, 'stderr') and self.process.stderr:
                import select
                while True:
                    readable, _, _ = select.select([self.process.stderr], [], [], 0)
                    if not readable:
                        break  # No data available
                    
                    line = self.process.stderr.readline().decode('utf-8', errors='replace')
                    if not line:
                        break  # End of stream
                    
                    # Log error output
                    self.log_update.emit(f"[ERROR] {line}")
    
        except Exception as e:
            logger.error(f"Error reading process output: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _check_process_status(self):
        """Check if the process is still running."""
        if not hasattr(self, 'process') or not self.process:
            if hasattr(self, 'process_timer'):
                self.process_timer.stop()
            if hasattr(self, 'output_timer'):
                self.output_timer.stop()
            return
        
        try:
            # Check if process has terminated
            returncode = self.process.poll()
            if returncode is not None:
                # Process has ended
                logger.info(f"Process ended with return code: {returncode}")
                
                # Stop timers
                if hasattr(self, 'process_timer'):
                    self.process_timer.stop()
                if hasattr(self, 'output_timer'):
                    self.output_timer.stop()
                
                # Read any remaining output
                self._read_process_output()
                
                # Update UI
                self.is_running = False
                self._update_ui_state()
                
                # Emit appropriate signal
                if returncode == 0:
                    self.status_label.setText("Completed")
                    self.simulation_finished.emit(True)

        # Unregister simulation with lock manager

        except Exception as e:
            logger.error(f"Error checking process status: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _on_memory_exceeded(self):
        """Handle the case when simulation memory usage exceeds limits."""
        self.status_label.setText("Stopped: Memory limit exceeded")
        QMessageBox.warning(
            self,
            "Memory Usage Warning",
            "The simulation was stopped because it exceeded memory limits.\n\n"
            "Consider using a coarser mesh or reducing the domain size."
        )
        
        # Update UI
        self.is_running = False
        self._update_ui_state()

    def _parse_output_line(self, line):
        """Parse a single line of output to extract progress information."""
        try:
            # Parse iteration and time
            import re
            
            # Parse time
            time_match = re.search(r"Time = (\d+\.?\d*)", line)
            if time_match:
                current_time = float(time_match.group(1))
                self.time_label.setText(f"{current_time:.6f}")
            
            # Parse iteration for steady-state solvers
            iter_match = re.search(r"Iteration (\d+)", line)
            if iter_match:
                iteration = int(iter_match.group(1))
                self.iteration_label.setText(str(iteration))
                
                # Update progress based on max iterations
                if hasattr(self, 'iterations_spin'):
                    max_iterations = self.iterations_spin.value()
                    progress = min(100, int((iteration / max_iterations) * 100))
                    self.progress_bar.setValue(progress)
            
            # Parse residuals
            residual_match = re.search(r"Solving for ([^,]+), Initial residual = ([^,]+), Final residual = ([^,]+)", line)
            if residual_match:
                field = residual_match.group(1).strip()
                final_residual = float(residual_match.group(3))
                
                # Update residuals dict
                if not hasattr(self, 'current_residuals'):
                    self.current_residuals = {}
                self.current_residuals[field] = final_residual
                
                # Parse current iteration if available
                current_iteration = int(self.iteration_label.text()) if self.iteration_label.text().isdigit() else 0
                
                # Update the residual plot
                if hasattr(self, 'residual_plot'):
                    self.residual_plot.add_data_point(current_iteration, self.current_residuals)
    
        except Exception as e:
            # Just log errors without disturbing the UI
            logger.error(f"Error parsing output line: {e}")

    def _prepare_simulation_command(self):
        """Prepare the simulation command based on current settings."""
        solver = self.solver_combo.currentText()
        parallel = self.parallel_check.isChecked()
        processors = self.processors_spin.value() if parallel else 1
        
        if parallel:
            cmd = ["mpirun", "-np", str(processors), solver, "-parallel"]
        else:
            cmd = [solver]
        
        cmd.extend(["-case", self.case_dir])
        return cmd

    def _stop_simulation(self):
        """Stop the running simulation."""
        if self.is_running:
            self.log_update.emit("Stopping simulation...")
            
            # Stop memory monitor
            self.memory_monitor_active = False
            
            # Check if we have a runner and stop it
            if hasattr(self, 'runner') and self.runner is not None:
                if hasattr(self.runner, 'process') and self.runner.process is not None:
                    # Set stop flag on runner
                    self.runner.stop_requested = True
                    
                    # Kill process in runner
                    self.runner.process.terminate()
                    try:
                        # Wait for process to terminate
                        # Can't use wait() directly on QProcess, but we can check its state
                        for _ in range(50):  # 5 second timeout (100ms * 50)
                            if self.runner.process.state() == QProcess.NotRunning:
                                break
                            QApplication.processEvents()
                            time.sleep(0.1)
                        else:
                            # Force kill if it doesn't terminate
                            self.runner.process.kill()
                    except Exception as e:
                        logger.error(f"Error terminating process: {e}")
                        # Force kill
                        self.runner.process.kill()
                
                # Terminate the thread if it's running
                if self.runner.isRunning():
                    self.runner.terminate()
                    self.runner.wait(5000)  # 5 second timeout
            
            # Also try to kill direct process if it exists (backward compatibility)
            if hasattr(self, 'process') and self.process is not None:
                self.process.terminate()
                try:
                    # Wait for process to terminate
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate
                    self.process.kill()
            
            self.is_running = False
            self._update_ui_state()
            self.log_update.emit("Simulation stopped.")
    
    def _on_progress_update(self, iteration: int, time_value: float, residuals: Dict[str, float]):
        """
        Handle progress update from simulation.
        
        Args:
            iteration (int): Current iteration
            time_value (float): Current simulation time
            residuals (Dict[str, float]): Current residuals
        """
        # Update labels
        self.iteration_label.setText(str(iteration))
        self.time_label.setText(f"{time_value:.6f}")
        
        # Update progress bar
        max_iterations = self.iterations_spin.value()
        progress = min(100, int((iteration / max_iterations) * 100))
        self.progress_bar.setValue(progress)
        
        # Update residuals table
        self.residuals_table.setRowCount(len(residuals))
        
        for i, (field, value) in enumerate(residuals.items()):
            # Field name
            field_item = QTableWidgetItem(field)
            self.residuals_table.setItem(i, 0, field_item)
            
            # Residual value
            value_item = QTableWidgetItem(f"{value:.8e}")
            self.residuals_table.setItem(i, 1, value_item)
        
        # Update residual plot
        self.residual_plot.add_data_point(iteration, residuals)
    
    def _on_status_update(self, status: str):
        """
        Handle status update from simulation.
        
        Args:
            status (str): Status message
        """
        self.status_label.setText(status)
    
    def _on_simulation_finished(self, success: bool):
        """
        Handle simulation finished event.
        
        Args:
            success (bool): Whether the simulation completed successfully
        """
        # Update UI
        self.is_running = False
        self._update_ui_state()
        
        if success:
            self.status_label.setText("Completed")
            QMessageBox.information(
                self,
                "Simulation Complete",
                "The simulation has completed successfully."
            )
        else:
            self.status_label.setText("Failed")
            # Don't show error message if stopped by user
            if not self.runner.stop_requested:
                QMessageBox.warning(
                    self,
                    "Simulation Failed",
                    "The simulation failed to complete."
                )
        
        # Clean up runner
        self.runner = None
        
        # Emit finished signal
        self.simulation_finished.emit(success)
    
    def _on_log_update(self, text: str):
        """
        Handle log update from simulation.
        
        Args:
            text (str): Log text
        """
        # Append to log text
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
    
    def set_project(self, project):
        """
        Set the current project.
        
        Args:
            project: The project to set
        """
        self.current_project = project
        
        # Update case directory if available
        if project and hasattr(project, 'case_dir') and project.case_dir:
            self.case_dir = project.case_dir
            self.case_path_edit.setText(self.case_dir)
        
        # Update UI state
        self._update_ui_state()

    def apply_boundaries_to_case(self):
        """Apply configured boundaries to the OpenFOAM case."""
        if not hasattr(self, 'boundary_config') or not self.boundary_config:
            QMessageBox.warning(self, "Warning", "No boundaries have been configured.")
            return False
        
        if not hasattr(self, 'case_dir') or not self.case_dir or not os.path.exists(self.case_dir):
            QMessageBox.warning(self, "Error", "No valid OpenFOAM case directory found.")
            return False
        
        try:
            # Create the case manager
            from ..openfoam_integration.case_manager import CaseManager
            case_manager = CaseManager(self.case_dir)
            
            # First create boundaries from selected faces
            if hasattr(self, 'boundary_cell_ids'):
                for boundary_name, face_ids in self.boundary_cell_ids.items():
                    # Only create boundaries that don't exist yet
                    if boundary_name not in case_manager.dict_manager.get_boundary_names():
                        # Determine patch type based on config
                        patch_type = "patch"  # default
                        if boundary_name in self.boundary_config:
                            config_type = self.boundary_config[boundary_name].get("type", "").lower()
                            if "wall" in config_type:
                                patch_type = "wall"
                            elif "symmetry" in config_type:
                                patch_type = "symmetryPlane"
                        
                        # Create the boundary from faces
                        success = case_manager.create_boundary_from_faces(
                            boundary_name, face_ids, patch_type
                        )
                        
                        if not success:
                            logger.error(f"Failed to create boundary '{boundary_name}' from faces")
                            QMessageBox.warning(self, "Error", f"Failed to create boundary '{boundary_name}' from faces.")
            
            # Now apply the boundary conditions configurations
            success = case_manager.setup_boundary_conditions(self.boundary_config)
            
            if success:
                QMessageBox.information(self, "Success", "Boundary conditions applied successfully to the OpenFOAM case.")
                self.status_label.setText("Boundary conditions applied to case.")
                return True
            else:
                QMessageBox.warning(self, "Error", "Failed to apply boundary conditions to the OpenFOAM case.")
                return False
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            logger.error(f"Error applying boundaries to case: {e}\n{error_msg}")
            QMessageBox.critical(self, "Error", f"Failed to apply boundary conditions: {str(e)}")
            return False

    def _set_all_as_wall(self):
        """Set all unassigned faces as wall boundary"""
        if hasattr(self.main_window, 'viewport'):
            # Ask for confirmation
            confirm = QMessageBox.question(
                self,
                "Set All as Wall",
                "This will set all unassigned faces as wall boundaries. Continue?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if confirm == QMessageBox.Yes:
                try:
                    # Try to set all as wall
                    wall_faces = self.main_window.viewport.set_all_as_wall(force=False)
                    
                    # If we got a tuple result, extract the data
                    if isinstance(wall_faces, tuple) and len(wall_faces) == 2:
                        success, face_ids = wall_faces
                    else:
                        # If just a boolean was returned, try to get face IDs another way
                        success = wall_faces
                        face_ids = None
                    
                    # If we couldn't get face IDs normally, try fallback approaches
                    if not face_ids and hasattr(self.main_window.viewport, 'get_wall_cell_ids'):
                        face_ids = self.main_window.viewport.get_wall_cell_ids()
                        success = face_ids is not None and len(face_ids) > 0
                    
                    # If we still don't have face IDs, try creating them directly from renderer
                    if not face_ids:
                        logger.info("Attempting fallback method to identify wall faces")
                        face_ids = self._generate_fallback_wall_faces()
                        success = face_ids is not None and len(face_ids) > 0
                    
                    if success and face_ids:
                        # Log what we found
                        logger.info(f"Retrieved {len(face_ids)} wall face IDs")
                        
                        # Create the wall boundary
                        wall_name = "wall"
                        self.status_label.setText(f"Creating wall boundary with {len(face_ids)} faces...")
                        
                        # Store these IDs for the boundary
                        if not hasattr(self, 'boundary_cell_ids'):
                            self.boundary_cell_ids = {}
                        self.boundary_cell_ids[wall_name] = face_ids
                        
                        # Add to boundary list if not already there
                        found = False
                        for i in range(self.boundary_list.count()):
                            item = self.boundary_list.item(i)
                            if item and item.data(Qt.UserRole) == wall_name:
                                found = True
                                self.boundary_list.setCurrentItem(item)
                                break
                        
                        if not found:
                            item = QListWidgetItem(wall_name)
                            item.setData(Qt.UserRole, wall_name)
                            self.boundary_list.addItem(item)
                            self.boundary_list.setCurrentItem(item)
                        
                        # Set boundary type to wall
                        self.type_combo.setCurrentText("wall")
                        
                        # Auto-apply the configuration
                        config = self.apply_to_boundary()
                        
                        if config:
                            self.status_label.setText(f"Wall boundary created with {len(face_ids)} faces")
                        else:
                            self.status_label.setText("Wall boundary created but configuration failed")
                    else:
                        # Create placeholder wall boundary anyway to allow manual configuration
                        wall_name = "wall"
                        self.status_label.setText("Creating manual wall boundary (no face IDs available)...")
                        
                        item = QListWidgetItem(wall_name)
                        item.setData(Qt.UserRole, wall_name)
                        self.boundary_list.addItem(item)
                        self.boundary_list.setCurrentItem(item)
                        
                        self.type_combo.setCurrentText("wall")
                        self.status_label.setText("Wall boundary created (no face IDs). Please set in OpenFOAM.")
                        
                        # Try to apply anyway
                        self.apply_to_boundary()
                
                except Exception as e:
                    logger.error(f"Error setting walls: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    self.status_label.setText(f"Error: {str(e)}")
            
    def _generate_fallback_wall_faces(self):
        """Generate fallback wall face IDs when the normal method fails"""
        try:
            # Check if we have access to the viewport
            if not hasattr(self.main_window, 'viewport'):
                return []
                
            viewport = self.main_window.viewport
            
            # Try different approaches to get face information
            face_ids = []
            
            # Approach 1: If we have direct access to the actor's data
            if hasattr(viewport, 'actor') and viewport.actor:
                polydata = viewport.actor.GetMapper().GetInput()
                if polydata:
                    # Get all cell IDs
                    face_ids = list(range(polydata.GetNumberOfCells()))
            
            # Approach 2: If we have a mesh object
            if not face_ids and hasattr(viewport, 'mesh') and viewport.mesh:
                if hasattr(viewport.mesh, 'GetNumberOfCells'):
                    face_ids = list(range(viewport.mesh.GetNumberOfCells()))
            
            # Approach 3: Create a placeholder with ID 0 as a last resort
            if not face_ids:
                face_ids = [0]  # At least one face ID so we can continue
                logger.warning("Using placeholder face ID for wall boundary")
            
            return face_ids
            
        except Exception as e:
            logger.error(f"Error generating fallback wall faces: {e}")
            return []

    def _setup_boundary_tab(self):
        """Set up the boundary conditions tab"""
        boundary_widget = QWidget()
        boundary_layout = QVBoxLayout(boundary_widget)
        
        # Add selection buttons
        buttons_layout = QHBoxLayout()
        
        # Add "all walls" button
        self.all_walls_button = QPushButton("Set All as Wall")
        self.all_walls_button.clicked.connect(self._set_all_as_wall)
        buttons_layout.addWidget(self.all_walls_button)
        
        # Add "select faces" button
        self.select_faces_button = QPushButton("Select Faces for Boundary")
        self.select_faces_button.clicked.connect(self.start_face_selection)
        buttons_layout.addWidget(self.select_faces_button)
        
        boundary_layout.addLayout(buttons_layout)
        
        # Create boundary list
        list_label = QLabel("Available Boundaries:")
        boundary_layout.addWidget(list_label)
        
        self.boundary_list = QListWidget()
        self.boundary_list.currentItemChanged.connect(self.boundary_selected)
        boundary_layout.addWidget(self.boundary_list)
        
        # Add delete button for boundaries
        self.delete_boundary_button = QPushButton("Delete Selected Boundary")
        self.delete_boundary_button.clicked.connect(self._delete_selected_boundary)
        self.delete_boundary_button.setStyleSheet("background-color: #ffaaaa;")
        boundary_layout.addWidget(self.delete_boundary_button)
        
        # Type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Boundary Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["inlet", "outlet", "wall", "symmetryPlane", "empty"])
        self.type_combo.currentTextChanged.connect(self.type_changed)
        type_layout.addWidget(self.type_combo)
        boundary_layout.addLayout(type_layout)
        
        # Stacked widget for different boundary type configurations
        self.config_stack = QStackedWidget()
        
        # Inlet configuration
        inlet_widget = QWidget()
        inlet_layout = QFormLayout(inlet_widget)
        
        # Velocity
        vel_group = QGroupBox("Velocity")
        vel_form = QFormLayout(vel_group)
        
        self.vel_type = QComboBox()
        self.vel_type.addItems(["fixedValue", "flowRate", "pressureInletOutletVelocity"])
        vel_form.addRow("Type:", self.vel_type)
        
        vel_values = QHBoxLayout()
        
        self.vel_x = QDoubleSpinBox()
        self.vel_x.setRange(-100, 100)
        self.vel_x.setValue(1.0)
        self.vel_x.setDecimals(3)
        self.vel_x.setSuffix(" m/s")
        
        self.vel_y = QDoubleSpinBox()
        self.vel_y.setRange(-100, 100)
        self.vel_y.setValue(0.0)
        self.vel_y.setDecimals(3)
        self.vel_y.setSuffix(" m/s")
        
        self.vel_z = QDoubleSpinBox()
        self.vel_z.setRange(-100, 100)
        self.vel_z.setValue(0.0)
        self.vel_z.setDecimals(3)
        self.vel_z.setSuffix(" m/s")
        
        vel_values.addWidget(QLabel("X:"))
        vel_values.addWidget(self.vel_x)
        vel_values.addWidget(QLabel("Y:"))
        vel_values.addWidget(self.vel_y)
        vel_values.addWidget(QLabel("Z:"))
        vel_values.addWidget(self.vel_z)
        
        vel_form.addRow("Value:", vel_values)
        inlet_layout.addRow(vel_group)
        
        # Pressure
        p_group = QGroupBox("Pressure")
        p_form = QFormLayout(p_group)
        
        self.p_type = QComboBox()
        self.p_type.addItems(["zeroGradient", "totalPressure", "fixedValue"])
        p_form.addRow("Type:", self.p_type)
        
        self.p_value = QDoubleSpinBox()
        self.p_value.setRange(0, 1000000)
        self.p_value.setValue(0)
        self.p_value.setDecimals(1)
        self.p_value.setSuffix(" Pa")
        p_form.addRow("Value:", self.p_value)
        
        inlet_layout.addRow(p_group)
        
        # Temperature
        t_group = QGroupBox("Temperature")
        t_form = QFormLayout(t_group)
        
        self.t_type = QComboBox()
        self.t_type.addItems(["fixedValue", "zeroGradient"])
        t_form.addRow("Type:", self.t_type)
        
        self.t_value = QDoubleSpinBox()
        self.t_value.setRange(0, 1000)
        self.t_value.setValue(300)
        self.t_value.setDecimals(1)
        self.t_value.setSuffix(" K")
        t_form.addRow("Value:", self.t_value)
        
        inlet_layout.addRow(t_group)
        
        self.config_stack.addWidget(inlet_widget)
        
        # Add other boundary type configurations (outlet, wall) - simplified
        # Outlet configuration
        outlet_widget = QWidget()
        outlet_layout = QFormLayout(outlet_widget)
        
        # Pressure for outlet
        out_p_group = QGroupBox("Pressure")
        out_p_form = QFormLayout(out_p_group)
        
        self.out_p_type = QComboBox()
        self.out_p_type.addItems(["fixedValue", "totalPressure", "zeroGradient"])
        out_p_form.addRow("Type:", self.out_p_type)
        
        self.out_p_value = QDoubleSpinBox()
        self.out_p_value.setRange(0, 1000000)
        self.out_p_value.setValue(101325)  # standard atmospheric pressure
        self.out_p_value.setDecimals(1)
        self.out_p_value.setSuffix(" Pa")
        out_p_form.addRow("Value:", self.out_p_value)
        
        outlet_layout.addRow(out_p_group)
        self.config_stack.addWidget(outlet_widget)
        
        # Wall configuration
        wall_widget = QWidget()
        wall_layout = QFormLayout(wall_widget)
        
        # Wall velocity
        wall_vel_group = QGroupBox("Velocity")
        wall_vel_form = QFormLayout(wall_vel_group)
        
        self.wall_vel_type = QComboBox()
        self.wall_vel_type.addItems(["noSlip", "slip", "movingWall"])
        wall_vel_form.addRow("Type:", self.wall_vel_type)
        
        wall_layout.addRow(wall_vel_group)
        self.config_stack.addWidget(wall_widget)
        
        # Other boundary types (simplified)
        # Symmetry plane
        symm_widget = QWidget()
        self.config_stack.addWidget(symm_widget)
        
        # Empty
        empty_widget = QWidget()
        self.config_stack.addWidget(empty_widget)
        
        boundary_layout.addWidget(self.config_stack)
        
        # Apply button for current boundary
        self.apply_btn = QPushButton("Apply Configuration")
        self.apply_btn.clicked.connect(self.apply_to_boundary)
        boundary_layout.addWidget(self.apply_btn)
        
        # Status label
        self.status_label = QLabel("")
        boundary_layout.addWidget(self.status_label)
        
        # Add stretch to push everything to the top
        boundary_layout.addStretch()
        
        # Set the layout for the boundary tab
        self.boundary_tab.setLayout(boundary_layout)
    
    def type_changed(self, new_type):
        """Called when the boundary type is changed."""
        # Set the appropriate configuration widget
        if new_type == "inlet":
            self.config_stack.setCurrentIndex(0)
        elif new_type == "outlet":
            self.config_stack.setCurrentIndex(1)
        elif new_type == "wall":
            self.config_stack.setCurrentIndex(2)
        elif new_type == "symmetryPlane":
            self.config_stack.setCurrentIndex(3)
        elif new_type == "empty":
            self.config_stack.setCurrentIndex(4)
    
    def apply_to_boundary(self):
        """Apply the current boundary settings to the named boundary"""
        try:
            # Get boundary name from current selection
            current_item = self.boundary_list.currentItem()
            if not current_item:
                QMessageBox.warning(
                    self,
                    "No Boundary Selected",
                    "Please select a boundary to apply configuration."
                )
                return
                
            boundary_name = current_item.data(Qt.UserRole)
            if not boundary_name:
                QMessageBox.warning(
                    self, 
                    "Invalid Selection",
                    "Invalid boundary selection"
                )
                return
                
            logger.info(f"Applying boundary settings to {boundary_name}")
            
            # Determine boundary type from UI
            boundary_type = self.type_combo.currentText()
            
            # Create a configuration dictionary
            config = {
                "name": boundary_name,
                "type": boundary_type
            }
            
            # Add velocity settings based on boundary type
            velocity_type = None
            
            # Try to get the velocity type from the appropriate UI element based on boundary type
            if boundary_type == "inlet":
                if hasattr(self, 'vel_type'):
                    velocity_type = self.vel_type.currentText()
            elif boundary_type == "outlet":
                if hasattr(self, 'vel_type'):
                    velocity_type = self.vel_type.currentText()
            elif boundary_type == "wall":
                if hasattr(self, 'wall_vel_type'):
                    velocity_type = self.wall_vel_type.currentText()
            
            # Set a default velocity type if none was found
            if velocity_type is None:
                velocity_type = "fixedValue" if boundary_type == "inlet" else "noSlip" if boundary_type == "wall" else "zeroGradient"
            
            config["velocity_type"] = velocity_type
            
            # Add velocity values if relevant
            if hasattr(self, 'vel_x') and hasattr(self, 'vel_y') and hasattr(self, 'vel_z'):
                config["velocity_x"] = self.vel_x.value()
                config["velocity_y"] = self.vel_y.value()
                config["velocity_z"] = self.vel_z.value()
            
            # Add pressure settings
            pressure_type = None
            
            # Try to get the pressure type
            if boundary_type == "inlet" and hasattr(self, 'p_type'):
                pressure_type = self.p_type.currentText()
            elif boundary_type == "outlet" and hasattr(self, 'out_p_type'):
                pressure_type = self.out_p_type.currentText()
            
            # Set a default pressure type if none was found
            if pressure_type is None:
                pressure_type = "fixedValue" if boundary_type == "outlet" else "zeroGradient"
            
            config["pressure_type"] = pressure_type
            
            # Add pressure value if relevant
            if boundary_type == "inlet" and hasattr(self, 'p_value'):
                config["pressure_value"] = self.p_value.value()
            elif boundary_type == "outlet" and hasattr(self, 'out_p_value'):
                config["pressure_value"] = self.out_p_value.value()
            
            # Temperature settings (if available)
            temperature_type = None
            
            if hasattr(self, 't_type'):
                temperature_type = self.t_type.currentText()
            
            if temperature_type:
                config["temperature_type"] = temperature_type
                
                if hasattr(self, 't_value'):
                    config["temperature_value"] = self.t_value.value()
            
            # Save the configuration to project
            project_saved = False
            if not hasattr(self, 'boundary_config'):
                self.boundary_config = {}
            
            self.boundary_config[boundary_name] = config
            
            # Save boundary conditions to project
            if self._save_boundary_conditions():
                project_saved = True
            
            # Apply to OpenFOAM case directory if available
            openfoam_saved = False
            case_manager_used = None
            
            # Try to use the case manager directly attached to this object
            if hasattr(self, 'case_manager') and self.case_manager:
                case_manager_used = self.case_manager
                success = self.case_manager.set_boundary_condition(
                    boundary_name, boundary_type, config
                )
                if success:
                    logger.info(f"Applied boundary settings to OpenFOAM case: {config}")
                    openfoam_saved = True
                else:
                    logger.warning(f"Failed to apply boundary settings to OpenFOAM case")
            
            # If that didn't work or wasn't available, try the project's case manager
            elif hasattr(self.main_window, 'project') and hasattr(self.main_window.project, 'case_manager') and self.main_window.project.case_manager:
                case_manager_used = self.main_window.project.case_manager
                success = self.main_window.project.case_manager.set_boundary_condition(
                    boundary_name, boundary_type, config
                )
                if success:
                    logger.info(f"Applied boundary settings to OpenFOAM case via project: {config}")
                    openfoam_saved = True
                else:
                    logger.warning(f"Failed to apply boundary settings to OpenFOAM case via project")
            
            # If neither worked, try the main window's case manager
            elif hasattr(self.main_window, 'case_manager') and self.main_window.case_manager:
                case_manager_used = self.main_window.case_manager
                success = self.main_window.case_manager.set_boundary_condition(
                    boundary_name, boundary_type, config
                )
                if success:
                    logger.info(f"Applied boundary settings to OpenFOAM case via main window: {config}")
                    openfoam_saved = True
                else:
                    logger.warning(f"Failed to apply boundary settings to OpenFOAM case via main window")
            
            # Log diagnostic information to help troubleshoot
            if not openfoam_saved:
                logger.warning(f"Case manager used: {case_manager_used.__class__.__name__ if case_manager_used else 'None'}")
                logger.warning(f"Case directory: {self.case_dir if hasattr(self, 'case_dir') else 'Not set'}")
                if case_manager_used:
                    logger.warning(f"Case manager directory: {case_manager_used.case_directory}")
            
            # Update status message
            if project_saved and openfoam_saved:
                self.status_label.setText(f"Applied configuration to '{boundary_name}' in project and OpenFOAM case")
            elif project_saved:
                if hasattr(self, 'case_dir') and self.case_dir:
                    self.status_label.setText(f"Applied configuration to '{boundary_name}' in project only (OpenFOAM update failed)")
                else:
                    self.status_label.setText(f"Applied configuration to '{boundary_name}' (saved to project only - no case directory set)")
            else:
                self.status_label.setText(f"Failed to apply configuration to '{boundary_name}'")
            
            # Force UI update
            QApplication.processEvents()
            
            return config
        
        except Exception as e:
            logger.error(f"Error applying boundary settings: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.status_label.setText(f"Error: {str(e)}")
            return {}


    def _setup_ambient_region_controls(self):
        """Add controls for creating and configuring an ambient region"""
        
        # Create a group box for ambient region controls
        ambient_group = QGroupBox("Ambient Region")
        ambient_layout = QVBoxLayout(ambient_group)
        
        # Add explanation label
        info_label = QLabel(
            "An ambient region allows simulation of flow into open space.\n"
            "This is useful for pouring, jets, and external flows."
        )
        info_label.setWordWrap(True)
        ambient_layout.addWidget(info_label)
        
        # Add controls for ambient region size
        size_form = QFormLayout()
        
        self.ambient_x_spin = QDoubleSpinBox()
        self.ambient_x_spin.setRange(1.0, 20.0)
        self.ambient_x_spin.setValue(5.0)
        self.ambient_x_spin.setSuffix("× pipe diameter")
        size_form.addRow("X extent:", self.ambient_x_spin)
        
        self.ambient_y_spin = QDoubleSpinBox()
        self.ambient_y_spin.setRange(1.0, 20.0)
        self.ambient_y_spin.setValue(5.0)
        self.ambient_y_spin.setSuffix("× pipe diameter")
        size_form.addRow("Y extent:", self.ambient_y_spin)
        
        self.ambient_z_spin = QDoubleSpinBox()
        self.ambient_z_spin.setRange(1.0, 20.0)
        self.ambient_z_spin.setValue(10.0)
        self.ambient_z_spin.setSuffix("× pipe diameter")
        size_form.addRow("Z extent:", self.ambient_z_spin)
        
        ambient_layout.addLayout(size_form)
        
        # Mesh refinement options
        refine_form = QFormLayout()
        
        self.interface_refine_spin = QSpinBox()
        self.interface_refine_spin.setRange(1, 5)
        self.interface_refine_spin.setValue(2)
        refine_form.addRow("Interface refinement:", self.interface_refine_spin)
        
        self.grad_refine_check = QCheckBox()
        self.grad_refine_check.setChecked(True)
        refine_form.addRow("Gradual refinement:", self.grad_refine_check)
        
        ambient_layout.addLayout(refine_form)
        
        # Visualization options
        vis_form = QFormLayout()
        
        self.ambient_opacity_spin = QDoubleSpinBox()
        self.ambient_opacity_spin.setRange(0.0, 1.0)
        self.ambient_opacity_spin.setSingleStep(0.1)
        self.ambient_opacity_spin.setValue(0.3)
        vis_form.addRow("Opacity:", self.ambient_opacity_spin)
        
        self.ambient_color_combo = QComboBox()
        self.ambient_color_combo.addItems(["Light Blue", "Light Gray", "Light Green"])
        vis_form.addRow("Color:", self.ambient_color_combo)
        
        ambient_layout.addLayout(vis_form)
        
        # Fluid properties
        fluid_form = QFormLayout()
        
        self.ambient_fluid_combo = QComboBox()
        self.ambient_fluid_combo.addItems(["Air", "Water", "Custom..."])
        fluid_form.addRow("Ambient fluid:", self.ambient_fluid_combo)
        
        ambient_layout.addLayout(fluid_form)
        
        # Create and add ambient region button
        self.create_ambient_btn = QPushButton("Create Ambient Region")
        self.create_ambient_btn.clicked.connect(self._create_ambient_region)
        ambient_layout.addWidget(self.create_ambient_btn)
        
        # Add stretcher to push everything to the top
        ambient_layout.addStretch()
        
        return ambient_group

    def _create_ambient_region(self):
        """Generate an ambient region based on current settings"""
        try:
            # Check if we have a valid mesh to work with
            if not hasattr(self.main_window, 'viewport') or not self.main_window.viewport:
                QMessageBox.warning(self, "Error", "No mesh available to create ambient region.")
                return
            
            # Get ambient region dimensions
            x_extent = self.ambient_x_spin.value()
            y_extent = self.ambient_y_spin.value()
            z_extent = self.ambient_z_spin.value()
            
            # Get refinement settings
            interface_refine = self.interface_refine_spin.value()
            gradual_refine = self.grad_refine_check.isChecked()
            
            # Get ambient fluid
            ambient_fluid = self.ambient_fluid_combo.currentText()
            
            # Get visualization settings
            opacity = self.ambient_opacity_spin.value()
            color_name = self.ambient_color_combo.currentText()
            
            # Create a centered ambient region around the entire mesh
            logger.info("Creating centered ambient region around the entire mesh")
            settings = {
                'x_extent': x_extent,
                'y_extent': y_extent,
                'z_extent': z_extent,
                'interface_refine': interface_refine,
                'gradual_refine': gradual_refine,
                'opacity': opacity,
                'color_name': color_name,
                'fluid': ambient_fluid,
                'centered': True  # Flag to indicate we want a centered ambient region
            }
            
            # Call the viewport method to create and visualize the ambient region
            if hasattr(self.main_window.viewport, 'create_centered_ambient_region'):
                success = self.main_window.viewport.create_centered_ambient_region(settings)
            else:
                # Add a fallback implementation
                self.main_window.viewport.create_centered_ambient_region = self._create_centered_ambient_region_fallback
                success = self.main_window.viewport.create_centered_ambient_region(settings)
            
            if success:
                # Show success message
                QMessageBox.information(
                    self,
                    "Ambient Region Created",
                    f"Centered ambient region created successfully with {ambient_fluid} as the ambient fluid.\n\n"
                    f"The mesh is positioned in the center of the ambient region."
                )
            
        except Exception as e:
            logger.error(f"Error creating ambient region: {e}")
            import traceback
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to create ambient region: {str(e)}")
            
    def _create_centered_ambient_region_fallback(self, settings):
        """Fallback implementation for creating a centered ambient region"""
        try:
            logger.info(f"Using fallback centered ambient region creation")
            
            # Create a simple message about what would happen
            QMessageBox.information(
                self,
                "Centered Ambient Region Simulation",
                f"A centered ambient region would be created with:\n"
                f"- Size: {settings['x_extent']}×{settings['y_extent']}×{settings['z_extent']}\n"
                f"- Ambient Fluid: {settings['fluid']}\n\n"
                f"This is a placeholder - implement create_centered_ambient_region in viewport.py"
            )
            return True
        except Exception as e:
            logger.error(f"Error in fallback centered ambient region creation: {e}")
            return False

    def auto_set_case_directory(self, project_dir=None):
        """
        Automatically set the OpenFOAM case directory.
        
        Args:
            project_dir (str, optional): Base project directory. If not provided,
                                         uses current project directory from main window.
        """
        try:
            # Get base project directory
            if not project_dir:
                if hasattr(self.main_window, 'current_project') and self.main_window.current_project:
                    if hasattr(self.main_window.current_project, 'project_dir'):
                        project_dir = self.main_window.current_project.project_dir
                    elif hasattr(self.main_window.current_project, 'get_project_directory'):
                        project_dir = self.main_window.current_project.get_project_directory()
            
            if not project_dir:
                logger.warning("No project directory available")
                return
                
            # Check if the case directory already exists with either standard format
            openfoam_dir = os.path.join(project_dir, "case", "openfoam")
            if not os.path.exists(openfoam_dir):
                openfoam_dir = os.path.join(project_dir, "case")
                if not os.path.exists(openfoam_dir):
                    # Create the directory if it doesn't exist
                    os.makedirs(openfoam_dir, exist_ok=True)
            
            # Just store the path internally without updating UI
            self.case_dir = openfoam_dir
            
            logger.info(f"OpenFOAM case directory automatically set to: {openfoam_dir}")
            
            # Enable simulation buttons now that we have a case directory
            if hasattr(self, 'setup_button'):
                self.setup_button.setEnabled(True)
            
            # Update case manager 
            from ..openfoam_integration.case_manager import create_case_manager
            self.case_manager = create_case_manager(openfoam_dir)
            
            return openfoam_dir
            
        except Exception as e:
            logger.error(f"Error setting OpenFOAM case directory: {e}")
            return None

    def _connect_signals(self):
        """Connect signals to slots"""
        # Run and stop buttons
        if hasattr(self, 'run_button'):
            # Fix: Changed run_simulation to _run_simulation
            self.run_button.clicked.connect(self._run_simulation)
        if hasattr(self, 'stop_button'):
            # Fix: Changed stop_simulation to _stop_simulation
            self.stop_button.clicked.connect(self._stop_simulation)
        if hasattr(self, 'setup_button'):
            # Fix: Changed setup_case to _setup_case
            self.setup_button.clicked.connect(self._setup_case)
        
        # Listen for project creation/loading events
        if hasattr(self.main_window, 'project_loaded'):
            self.main_window.project_loaded.connect(self.auto_set_case_directory)
            self.main_window.project_loaded.connect(self._load_boundary_conditions)
        
        # Listen for mesh generation events
        if hasattr(self.main_window, 'mesh_generated'):
            self.main_window.mesh_generated.connect(lambda: self.auto_set_case_directory())

    # In the SimulationControls class
    def set_boundary_condition(self, boundary_name, condition_type, values=None):
        """
        Set a boundary condition and save it to the project.
        
        Args:
            boundary_name (str): Name of the boundary
            condition_type (str): Type of boundary condition (inlet, outlet, wall)
            values (dict): Optional parameters for the boundary condition
        
        Returns:
            bool: True if successful, False otherwise
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if values is None:
            values = {}
            
        # Set default values based on condition type
        if condition_type == "inlet":
            if "U" not in values:
                values["U"] = {
                    "type": "fixedValue",
                    "value": "uniform (1 0 0)"  # Default x-direction flow
                }
            if "p" not in values:
                values["p"] = {
                    "type": "zeroGradient"
                }
                
        elif condition_type == "outlet":
            if "U" not in values:
                values["U"] = {
                    "type": "zeroGradient"
                }
            if "p" not in values:
                values["p"] = {
                    "type": "fixedValue",
                    "value": "uniform 0"
                }
                
        elif condition_type == "wall":
            if "U" not in values:
                values["U"] = {
                    "type": "noSlip"
                }
            if "p" not in values:
                values["p"] = {
                    "type": "zeroGradient"
                }
        
        logger.info(f"Setting boundary condition: {boundary_name} as {condition_type}")
        logger.debug(f"Boundary values: {values}")
        
        # Save the boundary condition
        self.boundary_conditions[boundary_name] = {
            "type": condition_type,
            "values": values
        }
        
        # Apply to case directory if available
        if hasattr(self, 'case_manager') and self.case_manager:
            try:
                self.case_manager.apply_boundary_condition(boundary_name, condition_type, values)
                logger.info(f"Applied boundary {boundary_name} to OpenFOAM case")
            except Exception as e:
                logger.error(f"Error applying boundary to OpenFOAM case: {e}")
        
        # Save boundary conditions to project
        saved = self._save_boundary_conditions()
        if saved:
            logger.info(f"Saved boundary condition {boundary_name} to project")
        else:
            logger.warning(f"Failed to save boundary condition {boundary_name} to project")
        
        return True
            
    def _save_boundary_conditions(self):
        """Save boundary conditions to a file in the pFroject directory"""
        try:
            # Create boundary_config if it doesn't exist
            if not hasattr(self, 'boundary_config'):
                self.boundary_config = {}
                
            # Get project directory
            project_dir = None
            if hasattr(self.main_window, 'project_manager'):
                project_dir = self.main_window.project_manager.get_project_directory()
            
            # If no project directory, try to use case directory as fallback
            if not project_dir and hasattr(self, 'case_dir') and self.case_dir:
                project_dir = Path(self.case_dir).parent
                logger.info(f"No project directory found, using parent of case directory: {project_dir}")
            
            if not project_dir:
                logger.warning("No project directory available")
                return False
                
            # Create boundary conditions file
            try:
                bc_file = os.path.join(project_dir, "boundary_conditions.json")
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(bc_file), exist_ok=True)
                
                # Save boundary conditions to file
                with open(bc_file, 'w') as f:
                    json.dump(self.boundary_config, f, indent=4)
                    
                logger.info(f"Saved boundary conditions to {bc_file}")
                return True
            except (IOError, PermissionError) as e:
                logger.error(f"Error writing to boundary conditions file: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving boundary conditions: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _load_boundary_conditions(self):
        """Load boundary conditions from a file in the project directory"""
        try:
            # Get project directory
            if hasattr(self.main_window, 'project_manager'):
                project_dir = self.main_window.project_manager.get_project_directory()
                if not project_dir:
                    logger.warning("No project directory available")
                    return False
                    
                # Check if boundary conditions file exists
                bc_file = os.path.join(project_dir, "boundary_conditions.json")
                if not os.path.exists(bc_file):
                    logger.info("No boundary conditions file found")
                    return False
                    
                # Load boundary conditions from file
                with open(bc_file, 'r') as f:
                    self.boundary_conditions = json.load(f)
                    
                logger.info(f"Loaded boundary conditions from {bc_file}")
                
                # Apply boundary conditions to the case
                if hasattr(self.main_window, 'case_manager'):
                    for boundary_name, condition in self.boundary_conditions.items():
                        self.main_window.case_manager.set_boundary_condition(
                            boundary_name, 
                            condition['type'], 
                            condition.get('values', {})
                        )
                        
                    logger.info(f"Applied {len(self.boundary_conditions)} boundary conditions to case")
                    
                # Update boundary visualization in viewport
                if hasattr(self.main_window, 'viewport'):
                    self.main_window.viewport.update_boundary_visualization()
                    
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error loading boundary conditions: {e}")
            return False
    
    def start_face_selection(self):
        """Start face selection for a new boundary"""
        if not hasattr(self.main_window, 'viewport'):
            QMessageBox.warning(self, "Error", "No viewport available for selection")
            return
        
        # Create dialog for boundary configuration
        dialog = QDialog(self)
        dialog.setWindowTitle("New Boundary")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout(dialog)
        
        # Add name and type fields
        form_layout = QFormLayout()
        
        # Name field
        name_edit = QLineEdit("new_boundary")
        form_layout.addRow("Boundary Name:", name_edit)
        
        # Type dropdown
        type_combo = QComboBox()
        type_combo.addItems(["inlet", "outlet", "wall", "symmetry", "empty"])
        form_layout.addRow("Boundary Type:", type_combo)
        
        layout.addLayout(form_layout)
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # Show dialog
        if dialog.exec_() != QDialog.Accepted:
            return
        
        # Get the boundary name and type from the dialog
        boundary_name = name_edit.text().strip()
        boundary_type = type_combo.currentText()
        
        if not boundary_name:
            QMessageBox.warning(self, "Error", "Boundary name cannot be empty")
            return
        
        # Check if boundary already exists
        for i in range(self.boundary_list.count()):
            item = self.boundary_list.item(i)
            if item and item.data(Qt.UserRole) == boundary_name:
                QMessageBox.warning(
                    self,
                    "Duplicate Boundary",
                    f"A boundary named '{boundary_name}' already exists. Please choose a different name."
                )
                return
        
        logger.info(f"Starting face selection for boundary: {boundary_name} (type: {boundary_type})")
        
        # Start the selection in the viewport
        self.status_label.setText(f"Selecting faces for {boundary_type} boundary '{boundary_name}'...")
        success = self.main_window.viewport.start_face_selection(
            boundary_name,
            lambda faces: self._on_face_selection_complete(boundary_name, boundary_type, faces)
        )
        
        if not success:
            self.status_label.setText("Failed to start face selection")
            QMessageBox.warning(self, "Error", "Failed to start face selection mode")
    
    def _on_face_selection_complete(self, boundary_name, boundary_type, face_ids):
        """Handle completion of face selection"""
        if not face_ids:
            logger.warning(f"No faces selected for boundary {boundary_name}")
            self.status_label.setText("No faces selected")
            return
        
        logger.info(f"Face selection complete: {len(face_ids)} faces for {boundary_name} (type: {boundary_type})")
        
        # Store the selected faces
        if not hasattr(self, 'boundary_cell_ids'):
            self.boundary_cell_ids = {}
        self.boundary_cell_ids[boundary_name] = face_ids
        
        # Add to boundary list
        item = QListWidgetItem(boundary_name)
        item.setData(Qt.UserRole, boundary_name)
        self.boundary_list.addItem(item)
        self.boundary_list.setCurrentItem(item)
        
        # Set boundary type
        self.type_combo.setCurrentText(boundary_type)
        
        # Create boundary configuration
        if not hasattr(self, 'boundary_config'):
            self.boundary_config = {}
        
        self.boundary_config[boundary_name] = {
            'type': boundary_type
        }
        
        # Update status
        self.status_label.setText(f"Created {boundary_type} boundary '{boundary_name}' with {len(face_ids)} faces")
        
        # Save configuration
        self._save_boundary_conditions()
    
    def boundary_selected(self, current, previous):
        """Called when a boundary is selected from the list."""
        if not current:
            return
        
        boundary_name = current.data(Qt.UserRole)
        
        # Check if we already have configuration for this boundary
        if hasattr(self, 'boundary_conditions') and boundary_name in self.boundary_conditions:
            config = self.boundary_conditions[boundary_name]
            
            # Set the boundary type
            index = self.type_combo.findText(config.get("type", "wall"))
            if index >= 0:
                self.type_combo.setCurrentIndex(index)
            
            # Set the configuration values based on boundary type
            if config["type"] == "inlet":
                # Set velocity
                if "velocity" in config:
                    self.vel_x.setValue(config["velocity"][0])
                    self.vel_y.setValue(config["velocity"][1])
                    self.vel_z.setValue(config["velocity"][2])
                
                # Set pressure
                if "pressure" in config:
                    self.p_value.setValue(config["pressure"])
                
                # Set temperature
                if "temperature" in config:
                    self.t_value.setValue(config["temperature"])
            
            elif config["type"] == "outlet":
                # Set pressure
                if "pressure" in config:
                    self.out_p_value.setValue(config["pressure"])
        
        # Highlight this boundary in the viewport
        if hasattr(self, 'main_window') and self.main_window and hasattr(self.main_window, 'viewport'):
            # Check if we have cell IDs for this boundary
            if hasattr(self, 'boundary_cell_ids') and boundary_name in self.boundary_cell_ids:
                cell_ids = self.boundary_cell_ids[boundary_name]
                
                # Debug log to see what type of data we're working with
                logger.debug(f"Cell IDs type: {type(cell_ids)}")
                
                # Process cell_ids to ensure they're in the correct format
                processed_ids = []
                try:
                    # Check if cell_ids is a dictionary (which seems to be the case based on error)
                    if isinstance(cell_ids, dict):
                        # Extract values from dictionary
                        for face_id in cell_ids.values():
                            if isinstance(face_id, (list, tuple)):
                                processed_ids.extend([int(i) for i in face_id])
                            else:
                                processed_ids.append(int(face_id))
                    elif isinstance(cell_ids, (list, tuple)):
                        # Handle list format
                        for cell_id in cell_ids:
                            if hasattr(cell_id, '__iter__') and not isinstance(cell_id, (str, bytes)):
                                processed_ids.append(int(cell_id[0]))
                            else:
                                processed_ids.append(int(cell_id))
                except Exception as e:
                    logger.error(f"Error processing cell IDs: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    processed_ids = []
                    
                if processed_ids:
                    self.status_label.setText(f"Selected boundary '{boundary_name}' with {len(processed_ids)} faces.")
                    try:
                        # Use the processed IDs
                        self.main_window.viewport.highlight_faces(boundary_name, processed_ids)
                    except Exception as e:
                        logger.error(f"Failed to highlight faces: {e}")
                        self.status_label.setText(f"Error highlighting boundary: {str(e)}")
                else:
                    self.status_label.setText(f"Selected boundary '{boundary_name}' (no valid face IDs found)")

    def _delete_selected_boundary(self):
        """Delete the currently selected boundary"""
        current_item = self.boundary_list.currentItem()
        if not current_item:
            QMessageBox.warning(
                self,
                "No Boundary Selected",
                "Please select a boundary to delete."
            )
            return
        
        boundary_name = current_item.data(Qt.UserRole)
        
        # Ask for confirmation
        confirm = QMessageBox.question(
            self,
            "Delete Boundary",
            f"Are you sure you want to delete the boundary '{boundary_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            try:
                logger.info(f"Deleting boundary: {boundary_name}")
                
                # Remove from the boundary list widget
                row = self.boundary_list.row(current_item)
                self.boundary_list.takeItem(row)
                
                # Remove from boundary configuration
                if hasattr(self, 'boundary_config') and boundary_name in self.boundary_config:
                    logger.info(f"Removing {boundary_name} from boundary_config")
                    del self.boundary_config[boundary_name]
                
                # Remove from boundary cell IDs
                if hasattr(self, 'boundary_cell_ids') and boundary_name in self.boundary_cell_ids:
                    logger.info(f"Removing {boundary_name} from boundary_cell_ids")
                    del self.boundary_cell_ids[boundary_name]
                
                # Clear highlighting in viewport
                if hasattr(self.main_window, 'viewport'):
                    logger.info(f"Clearing highlight for {boundary_name} in viewport")
                    # If the viewport has a method to clear highlights
                    if hasattr(self.main_window.viewport, 'clear_highlight'):
                        self.main_window.viewport.clear_highlight(boundary_name)
                    
                    # Force a render update
                    if hasattr(self.main_window.viewport, 'render_window'):
                        logger.info("Forcing render window update")
                        self.main_window.viewport.render_window.Render()
                
                # If we have a case directory and case manager, remove from OpenFOAM case
                if hasattr(self, 'case_dir') and self.case_dir and hasattr(self.main_window, 'case_manager'):
                    try:
                        # Try to remove the boundary from the OpenFOAM case
                        logger.info(f"Removing {boundary_name} from OpenFOAM case")
                        success = self.main_window.case_manager.remove_boundary(boundary_name)
                        if success:
                            logger.info(f"Removed boundary '{boundary_name}' from OpenFOAM case")
                        else:
                            logger.warning(f"Failed to remove boundary '{boundary_name}' from OpenFOAM case")
                    except Exception as e:
                        logger.error(f"Error removing boundary from case: {e}")
                
                # Update status
                self.status_label.setText(f"Deleted boundary '{boundary_name}'")
                
                # Save boundary conditions
                self._save_boundary_conditions()
                
                logger.info(f"Successfully deleted boundary: {boundary_name}")
                
            except Exception as e:
                logger.error(f"Error deleting boundary: {e}")
                import traceback
                logger.error(traceback.format_exc())
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to delete boundary: {str(e)}"
                )

    def setup_case(self):
        """Configure the OpenFOAM case with current UI settings."""
        try:
            logger.info("Configuring OpenFOAM case with current settings")
            
            # Add debug logging to trace the flow
            logger.debug(f"Has case_dir attribute: {hasattr(self, 'case_dir')}")
            if hasattr(self, 'case_dir'):
                logger.debug(f"Current case_dir value: {self.case_dir}")
            
            logger.debug(f"Has main_window attribute: {hasattr(self, 'main_window')}")
            if hasattr(self, 'main_window'):
                logger.debug(f"Has current_project: {hasattr(self.main_window, 'current_project')}")
                if hasattr(self.main_window, 'current_project'):
                    logger.debug(f"Current project filepath: {getattr(self.main_window.current_project, 'filepath', None)}")
            
            # Create a case directory if it doesn't exist yet
            if not hasattr(self, 'case_dir') or not self.case_dir:
                logger.info("No case directory set yet, attempting to create one")
                
                # Check if project has been saved 
                if hasattr(self.main_window, 'current_project') and self.main_window.current_project:
                    project = self.main_window.current_project
                    
                    # If project has a filepath, set case directory based on it
                    if hasattr(project, 'filepath') and project.filepath:
                        project_dir = os.path.dirname(project.filepath)
                        self.case_dir = os.path.join(project_dir, "case")
                        logger.info(f"Created case directory path based on project filepath: {self.case_dir}")
                        
                        # Update project with case directory
                        project.case_dir = self.case_dir
                        
                        # Create the directory if it doesn't exist
                        os.makedirs(self.case_dir, exist_ok=True)
                        
                    # If not saved, prompt user to save first
                    else:
                        logger.info("Project not saved yet, prompting user to save")
                        QMessageBox.information(
                            self,
                            "Save Project",
                            "Please save your project first to create the case directory structure."
                        )
                        
                        if hasattr(self.main_window, 'save_project_as'):
                            self.main_window.save_project_as()
                        else:
                            self.main_window.save_project()
                            
                            # If user canceled or save failed, we can't proceed
                            if not hasattr(project, 'filepath') or not project.filepath:
                                logger.warning("Project save canceled or failed")
                                return
                            
                            # Now get the case directory from the newly saved project
                            project_dir = os.path.dirname(project.filepath)
                            self.case_dir = os.path.join(project_dir, "case")
                            project.case_dir = self.case_dir
                            logger.info(f"Created case directory path after saving: {self.case_dir}")
                            
                            # Create the directory if it doesn't exist
                            os.makedirs(self.case_dir, exist_ok=True)
            
            # After all our attempts, if we still don't have a case directory, show error and return
            if not hasattr(self, 'case_dir') or not self.case_dir:
                logger.error("No case directory specified after attempts to create one")
                QMessageBox.warning(
                    self,
                    "No Case Directory",
                    "Please specify a case directory first."
                )
                return
                
            # Update UI if needed
            if hasattr(self, 'case_dir_edit'):
                self.case_dir_edit.setText(self.case_dir)
            
            logger.info(f"Proceeding with case configuration at: {self.case_dir}")
            
            # Ensure the case manager is set up
            if not hasattr(self, 'case_manager') or not self.case_manager:
                logger.info("Creating new case manager")
                from ..openfoam_integration.case_manager import create_case_manager
                self.case_manager = create_case_manager(self.case_dir)
                logger.info(f"Created new case manager for: {self.case_dir}")
            
            # Create or update the case directory structure
            logger.info("Setting up OpenFOAM case structure")
            if not os.path.exists(os.path.join(self.case_dir, 'system')):
                success = self.case_manager.create_case_structure()
                if not success:
                    logger.error("Failed to create case structure")
                    QMessageBox.critical(self, "Error", "Failed to create OpenFOAM case structure")
                    return
            
            # More robust fix for the reference cell issue in the fvSolution file
            logger.info("Configuring pressure reference cell")
            
            # Get the actual OpenFOAM case directory (might be a subdirectory)
            openfoam_dir = self.case_dir
            if os.path.exists(os.path.join(self.case_dir, 'openfoam')):
                openfoam_dir = os.path.join(self.case_dir, 'openfoam')
                logger.debug(f"Using OpenFOAM subdirectory: {openfoam_dir}")
            
            # Make sure the system directory exists
            system_dir = os.path.join(openfoam_dir, 'system')
            os.makedirs(system_dir, exist_ok=True)
            
            # Check for fvSolution file and create or modify it
            fvSolution_path = os.path.join(system_dir, 'fvSolution')
            logger.debug(f"Checking fvSolution at: {fvSolution_path}")
            
            if os.path.exists(fvSolution_path):
                logger.debug("fvSolution file exists, reading content")
                try:
                    with open(fvSolution_path, 'r') as f:
                        content = f.read()
                    
                    import re
                    
                    # Check if file has SIMPLE section
                    if 'SIMPLE' in content:
                        logger.debug("SIMPLE section found, replacing it with correct version")
                        
                        # Replace the entire SIMPLE section
                        pattern = r'SIMPLE\s*\{[^\}]*\}'
                        replacement = """SIMPLE
{
    nNonOrthogonalCorrectors 0;
    pRefCell    0;
    pRefValue   0;
}"""
                        
                        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                        
                        # If the pattern didn't match, append the section
                        if new_content == content:
                            logger.debug("Pattern did not match, appending SIMPLE section")
                            new_content += f"\n\n{replacement}\n"
                    else:
                        logger.debug("No SIMPLE section found, appending it")
                        # Append the SIMPLE section to the file
                        new_content = content + "\n\n" + """SIMPLE
{
    nNonOrthogonalCorrectors 0;
    pRefCell    0;
    pRefValue   0;
}
"""
                    
                    # Write the updated content back
                    with open(fvSolution_path, 'w') as f:
                        f.write(new_content)
                        
                    logger.info(f"Updated fvSolution file with pressure reference cell settings at {fvSolution_path}")
                    
                except Exception as e:
                    logger.error(f"Error updating fvSolution: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
            else:
                logger.debug("fvSolution file does not exist, creating it")
                # Create a new fvSolution file with all required settings
                with open(fvSolution_path, 'w') as f:
                    f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-6;
        relTol          0.1;
        smoother        GaussSeidel;
    }

    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-5;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    pRefCell    0;
    pRefValue   0;
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
    }
}

// ************************************************************************* //
""")
                logger.info(f"Created new fvSolution file with reference cell settings at {fvSolution_path}")
            
            # Verify the file has been updated correctly
            try:
                if os.path.exists(fvSolution_path):
                    with open(fvSolution_path, 'r') as f:
                        content = f.read()
                    logger.debug(f"Final fvSolution file content (first 500 chars): {content[:500]}...")
                    
                    # Check if pRefCell is in the content
                    if 'pRefCell' in content:
                        logger.info("Verified pRefCell is present in fvSolution file")
                    else:
                        logger.warning("pRefCell still not found in fvSolution file")
                else:
                    logger.warning(f"fvSolution file not found at {fvSolution_path} after attempted creation")
            except Exception as e:
                logger.error(f"Error verifying fvSolution file: {e}")
            
            # Enable the run button now that the case is configured
            self.run_button.setEnabled(True)
            
            # Update status
            if hasattr(self, 'status_label'):
                self.status_label.setText("OpenFOAM case configured successfully")
            
            # Let the user know it worked
            QMessageBox.information(
                self,
                "Case Configured",
                "OpenFOAM case has been configured with current settings.\nYou can now run the simulation."
            )
            
        except Exception as e:
            logger.error(f"Error configuring case: {e}")
            import traceback
            logger.error(traceback.format_exc())
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to configure case: {str(e)}"
            )

    def _ensure_pressure_reference(self, case_dir):
        """Ensure pressure reference cell is set in all OpenFOAM cases."""
        # Find all possible locations of fvSolution
        possible_paths = [
            os.path.join(case_dir, 'system', 'fvSolution'),
            os.path.join(case_dir, 'openfoam', 'system', 'fvSolution')
        ]
        
        # Try each possible path
        for fvSolution_path in possible_paths:
            if os.path.exists(fvSolution_path):
                # Use simple file operations instead of regex
                try:
                    with open(fvSolution_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Look for SIMPLE section
                    simple_section_found = False
                    pref_cell_found = False
                    new_lines = []
                    
                    for line in lines:
                        new_lines.append(line)
                        
                        # Check for SIMPLE section start
                        if 'SIMPLE' in line and '{' in line:
                            simple_section_found = True
                        
                        # Check for pRefCell in the section
                        if simple_section_found and 'pRefCell' in line:
                            pref_cell_found = True
                    
                    # If we found SIMPLE but no pRefCell, add it before the closing brace
                    if simple_section_found and not pref_cell_found:
                        # Find the closing brace of SIMPLE section
                        for i, line in enumerate(new_lines):
                            if '}' in line and simple_section_found:
                                # Insert before this line
                                new_lines.insert(i, '    pRefCell    0;\n')
                                new_lines.insert(i, '    pRefValue   0;\n')
                                break
                    
                    # If we didn't find SIMPLE section at all, add it at the end
                    if not simple_section_found:
                        new_lines.append('\nSIMPLE\n{\n    nNonOrthogonalCorrectors 0;\n    pRefCell    0;\n    pRefValue   0;\n}\n')
                    
                    # Write back
                    with open(fvSolution_path, 'w') as f:
                        f.writelines(new_lines)
                        
                    logger.info(f"Successfully ensured pressure reference cell in {fvSolution_path}")
                    return True
                    
                except Exception as e:
                    logger.error(f"Error updating {fvSolution_path}: {e}")
                    continue
        
        # If we couldn't find or update any existing files, create a new one
        system_dir = os.path.join(case_dir, 'system')
        if not os.path.exists(system_dir):
            system_dir = os.path.join(case_dir, 'openfoam', 'system')
        
        os.makedirs(system_dir, exist_ok=True)
        fvSolution_path = os.path.join(system_dir, 'fvSolution')
        
        try:
            # Create a complete new file with default settings
            with open(fvSolution_path, 'w') as f:
                f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-6;
        relTol          0.1;
        smoother        GaussSeidel;
    }

    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-5;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    pRefCell    0;
    pRefValue   0;
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
    }
}

// ************************************************************************* //
""")
            logger.info(f"Created new fvSolution file with reference cell settings at {fvSolution_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create fvSolution file: {e}")
            return False

    def _fix_pressure_reference(self):
        """Fix pressure reference in fvSolution."""
        try:
            import os
            from ..openfoam_integration.case_manager import CaseDictManager
            
            # Path to fvSolution dictionary
            fv_solution_path = os.path.join(self.case_dir, "system", "fvSolution")
            
            # Ensure system directory exists
            system_dir = os.path.join(self.case_dir, "system")
            if not os.path.exists(system_dir):
                os.makedirs(system_dir)
            
            # Log more info for debugging
            self.log_update.emit(f"Checking fvSolution at: {fv_solution_path}")
            
            # Create fvSolution file if it doesn't exist
            if not os.path.exists(fv_solution_path):
                self.log_update.emit("fvSolution file not found. Creating default.")
                self._generate_solution_dict()
                return
            
            # Get solver settings
            solver = self.solver_combo.currentText()
            
            # Check if file is empty or malformed
            try:
                # Initialize CaseDictManager
                dict_manager = CaseDictManager(fv_solution_path)
                file_contents = dict_manager.get_contents()
                
                if not file_contents or len(file_contents.strip()) < 10:
                    self.log_update.emit("fvSolution file is empty or invalid. Regenerating.")
                    self._generate_solution_dict()
                    return
                    
            except Exception as dict_err:
                self.log_update.emit(f"Error reading fvSolution: {str(dict_err)}. Regenerating.")
                self._generate_solution_dict()
                return
            
            # Now proceed with proper updates
            dict_manager = CaseDictManager(fv_solution_path)
            
            # Add proper PIMPLE/SIMPLE section based on solver
            if "simpleFoam" in solver:
                # Log what we're doing
                self.log_update.emit("Setting up SIMPLE solver settings in fvSolution")
                
                # Check if SIMPLE section exists, create if not
                if not dict_manager.has_subdict("SIMPLE"):
                    self.log_update.emit("Adding SIMPLE dictionary to fvSolution")
                    dict_manager.add_subdict("SIMPLE", {
                        "nNonOrthogonalCorrectors": 0,
                        "consistent": "yes"
                    })
                
                # Get the SIMPLE subdict
                simple_dict = dict_manager.get_subdict("SIMPLE")
                
                # Always explicitly set pRefCell and pRefValue to ensure they exist
                simple_dict["pRefCell"] = 0
                simple_dict["pRefValue"] = 0
                
                # Write back to file
                dict_manager.write()
                self.log_update.emit("SIMPLE settings updated with pRefCell=0, pRefValue=0")
                
            elif "pimpleFoam" in solver or "pimpleDyMFoam" in solver:
                # Log what we're doing
                self.log_update.emit("Setting up PIMPLE solver settings in fvSolution")
                
                # Check if PIMPLE section exists, create if not
                if not dict_manager.has_subdict("PIMPLE"):
                    self.log_update.emit("Adding PIMPLE dictionary to fvSolution")
                    dict_manager.add_subdict("PIMPLE", {
                        "nNonOrthogonalCorrectors": 1,
                        "nCorrectors": 2
                    })
                
                # Get the PIMPLE subdict
                pimple_dict = dict_manager.get_subdict("PIMPLE")
                
                # Always explicitly set pRefCell and pRefValue to ensure they exist
                pimple_dict["pRefCell"] = 0
                pimple_dict["pRefValue"] = 0
                
                # Write back to file
                dict_manager.write()
                self.log_update.emit("PIMPLE settings updated with pRefCell=0, pRefValue=0")
                
            
            elif "pisoFoam" in solver:
                # Log what we're doing
                self.log_update.emit("Setting up PISO solver settings in fvSolution")
                
                # Check if PISO section exists, create if not
                if not dict_manager.has_subdict("PISO"):
                    self.log_update.emit("Adding PISO dictionary to fvSolution")
                    dict_manager.add_subdict("PISO", {
                        "nNonOrthogonalCorrectors": 1,
                        "nCorrectors": 2
                    })
                
                # Get the PISO subdict
                piso_dict = dict_manager.get_subdict("PISO")
                
                # Always explicitly set pRefCell and pRefValue to ensure they exist
                piso_dict["pRefCell"] = 0
                piso_dict["pRefValue"] = 0
                
                # Write back to file
                dict_manager.write()
                self.log_update.emit("PISO settings updated with pRefCell=0, pRefValue=0")
                
            elif "interFoam" in solver:
                # This solver doesn't need a pressure reference cell
                self.log_update.emit("Note: interFoam doesn't require pRefCell/pRefValue")
            
            # Verify the file was properly written
            if os.path.exists(fv_solution_path):
                file_size = os.path.getsize(fv_solution_path)
                self.log_update.emit(f"fvSolution file updated successfully, size: {file_size} bytes")
                
                # Double-check content
                with open(fv_solution_path, 'r') as f:
                    content = f.read()
                    if "simpleFoam" in solver and "pRefCell" not in content:
                        self.log_update.emit("WARNING: pRefCell still not found in fvSolution - regenerating file")
                        self._generate_solution_dict()
            else:
                self.log_update.emit("ERROR: fvSolution file not found after update!")
            
        except Exception as e:
            import traceback
            self.log_update.emit(f"ERROR fixing pressure reference: {str(e)}")
            logger.error(f"Error in _fix_pressure_reference: {e}")
            logger.error(traceback.format_exc())
            
            # As last resort, try direct file writing
            try:
                self._generate_solution_dict()
            except Exception as gen_err:
                self.log_update.emit(f"Failed to generate fvSolution dictionary: {str(gen_err)}")

    def _ensure_turbulence_properties(self):
        """Ensure turbulence properties file exists."""
        try:
            import os
            
            # Path to turbulenceProperties dictionary
            turb_props_path = os.path.join(self.case_dir, "constant", "turbulenceProperties")
            
            # Check if file exists
            if os.path.exists(turb_props_path):
                self.log_update.emit("Found turbulenceProperties file")
                return
            
            # Create constant directory if it doesn't exist
            constant_dir = os.path.join(self.case_dir, "constant")
            if not os.path.exists(constant_dir):
                os.makedirs(constant_dir)
            
            # Default turbulence model based on solver
            solver = self.solver_combo.currentText()
            model = "kOmegaSST"  # Default
            
            # Create content
            content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/

FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

simulationType  RAS;

RAS
{{
    RASModel        {model};
    turbulence      on;
    printCoeffs     on;
}}

// ************************************************************************* //
"""
            
            # Write to file
            with open(turb_props_path, 'w') as f:
                f.write(content)
                
            self.log_update.emit(f"Created turbulenceProperties with {model} model")
            
        except Exception as e:
            import traceback
            self.log_update.emit(f"Warning: Could not create turbulenceProperties: {str(e)}")
            logger.error(traceback.format_exc())

    def _ensure_turbulence_fields(self):
        """Ensure turbulence fields exist in the 0 directory."""
        try:
            import os
            
            # Path to 0 directory
            zero_dir = os.path.join(self.case_dir, "0")
            
            # Check if 0 directory exists
            if not os.path.exists(zero_dir):
                self.log_update.emit("Warning: 0 directory not found")
                return
            
            # Required turbulence fields
            required_fields = ["k", "omega", "epsilon", "nut"]
            
            # Check each field
            for field in required_fields:
                field_path = os.path.join(zero_dir, field)
                
                # Skip if already exists
                if os.path.exists(field_path):
                    continue
                
                # Create the missing field based on type
                if field == "k":
                    self._create_k_field(field_path)
                elif field == "omega":
                    self._create_omega_field(field_path)
                elif field == "epsilon":
                    self._create_epsilon_field(field_path)
                elif field == "nut":
                    self._create_nut_field(field_path)
            
            self.log_update.emit("Turbulence fields checked and created if missing")
            
        except Exception as e:
            import traceback
            self.log_update.emit(f"Warning: Could not ensure turbulence fields: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _fix_turbulence_fields(self):
        """
        Ensure turbulence fields are properly configured for the selected model.
        """
        try:
            # Make sure turbulence properties and fields exist
            self._ensure_turbulence_properties()
            self._ensure_turbulence_fields()
            
            # Get the current turbulence model
            model_path = os.path.join(self.case_dir, "constant", "turbulenceProperties")
            model = "kOmegaSST"  # Default
            
            if os.path.exists(model_path):
                with open(model_path, 'r') as f:
                    content = f.read()
                    match = re.search(r'simulationType\s+(\w+)', content)
                    if match and match.group(1) == "LES":
                        # Handle LES models differently
                        logger.info("LES turbulence model detected")
                        return
                    
                    match = re.search(r'RAS\s*{[^}]*model\s+(\w+)', content, re.DOTALL)
                    if match:
                        model = match.group(1)
            
            # Update fields based on the turbulence model
            if "kEpsilon" in model:
                # Make sure epsilon field exists
                epsilon_path = os.path.join(self.case_dir, "0", "epsilon")
                if not os.path.exists(epsilon_path):
                    self._create_epsilon_field(epsilon_path)
                    
            elif "kOmega" in model:
                # Make sure omega field exists
                omega_path = os.path.join(self.case_dir, "0", "omega")
                if not os.path.exists(omega_path):
                    self._create_omega_field(omega_path)
                    
            # Always make sure k and nut exist for RAS models
            k_path = os.path.join(self.case_dir, "0", "k")
            if not os.path.exists(k_path):
                self._create_k_field(k_path)
                
            nut_path = os.path.join(self.case_dir, "0", "nut")
            if not os.path.exists(nut_path):
                self._create_nut_field(nut_path)
                
        except Exception as e:
            logger.error(f"Error fixing turbulence fields: {e}")
            logger.error(traceback.format_exc())
    
    def _fix_fv_options(self):
        """Fix fvOptions file or create a minimal valid one if needed."""
        try:
            import os
            
            # Path to fvOptions dictionary
            fv_options_path = os.path.join(self.case_dir, "system", "fvOptions")
            
            # Ensure system directory exists
            system_dir = os.path.join(self.case_dir, "system")
            if not os.path.exists(system_dir):
                os.makedirs(system_dir)
            
            # Log the check
            self.log_update.emit(f"Checking fvOptions at: {fv_options_path}")
            
            # Check if cellZones exist in the case
            cell_zones = self._get_available_cell_zones()
            self.log_update.emit(f"Available cell zones: {cell_zones}")
            
            # Create a minimal valid fvOptions file (without any constraints that require cellZones)
            content = """/*--------------------------------*- C++ -*----------------------------------*\\
    | =========                 |                                                 |
    | \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
    |  \\\\    /   O peration     | Version:  v2312                                 |
    |   \\\\  /    A nd           | Website:  [www.openfoam.com](www.openfoam.com)                      |
    |    \\\\/     M anipulation  |                                                 |
    \\*---------------------------------------------------------------------------*/
    FoamFile
    {
        version     2.0;
        format      ascii;
        class       dictionary;
        object      fvOptions;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    
    // This is a minimal fvOptions file with no active options
    
    // ************************************************************************* //
    """
            
            # Write the fixed file
            with open(fv_options_path, "w") as f:
                f.write(content)
            
            self.log_update.emit("Created minimal valid fvOptions file (removed any invalid cellZone references)")
            
        except Exception as e:
            import traceback
            self.log_update.emit(f"Warning: Could not fix fvOptions file: {str(e)}")
            logger.error(f"Error in _fix_fv_options: {e}")
            logger.error(traceback.format_exc())
    
    def _verify_boundary_conditions(self):
        """Verify that all boundary conditions are properly set."""
        try:
            import os
            
            # Set up case manager
            from ..openfoam_integration.case_manager import CaseManager
            case_manager = CaseManager(self.case_dir)
            
            # Get all boundaries
            if hasattr(case_manager, 'get_boundaries'):
                boundaries = case_manager.get_boundaries()
            elif hasattr(case_manager, 'get_boundary_names'):
                boundaries = case_manager.get_boundary_names()
            else:
                # Fallback - try to read boundary file directly
                try:
                    from ..openfoam_integration.case_manager import CaseDictManager
                    boundary_file = os.path.join(self.case_dir, "constant", "polyMesh", "boundary")
                    if os.path.exists(boundary_file):
                        dict_manager = CaseDictManager(boundary_file)
                        boundaries = list(dict_manager.get_root_keys())
                    else:
                        self.log_update.emit("Warning: No boundary file found in the case!")
                        return
                except Exception as inner_e:
                    self.log_update.emit(f"Warning: Cannot read boundaries: {str(inner_e)}")
                    return
            
            # Check if there are any boundaries
            if not boundaries:
                self.log_update.emit("Warning: No boundaries found in the case!")
                return
            
            # Get solver
            solver = self.solver_combo.currentText()
            
            # Check U file
            u_file = os.path.join(self.case_dir, "0", "U")
            if os.path.exists(u_file):
                from ..openfoam_integration.case_manager import CaseDictManager
                u_manager = CaseDictManager(u_file)
                
                # Check if boundaryField exists
                if not u_manager.has_subdict("boundaryField"):
                    u_manager.add_subdict("boundaryField", {})
                
                # Check each boundary
                boundary_field = u_manager.get_subdict("boundaryField")
                for boundary in boundaries:
                    if boundary not in boundary_field:
                        # Add default boundary condition based on name
                        if "inlet" in boundary.lower():
                            u_manager.add_subdict(f"boundaryField/{boundary}", {
                                "type": "fixedValue", 
                                "value": "uniform (1 0 0)"
                            })
                        elif "outlet" in boundary.lower():
                            u_manager.add_subdict(f"boundaryField/{boundary}", {
                                "type": "zeroGradient"
                            })
                        elif "wall" in boundary.lower():
                            u_manager.add_subdict(f"boundaryField/{boundary}", {
                                "type": "noSlip"
                            })
                        else:
                            u_manager.add_subdict(f"boundaryField/{boundary}", {
                                "type": "zeroGradient"
                            })
                
                # Write changes
                u_manager.write()
            
            # For interFoam, check alpha.water file
            if "interFoam" in solver:
                alpha_file = os.path.join(self.case_dir, "0", "alpha.water")
                if os.path.exists(alpha_file):
                    alpha_manager = CaseDictManager(alpha_file)
                    
                    # Check if boundaryField exists
                    if not alpha_manager.has_subdict("boundaryField"):
                        alpha_manager.add_subdict("boundaryField", {})
                    
                    # Check each boundary
                    boundary_field = alpha_manager.get_subdict("boundaryField")
                    for boundary in boundaries:
                        if boundary not in boundary_field:
                            # Add default boundary condition based on name
                            if "inlet" in boundary.lower():
                                alpha_manager.add_subdict(f"boundaryField/{boundary}", {
                                    "type": "fixedValue", 
                                    "value": "uniform 1"
                                })
                            elif "outlet" in boundary.lower():
                                alpha_manager.add_subdict(f"boundaryField/{boundary}", {
                                    "type": "zeroGradient"
                                })
                            elif "wall" in boundary.lower():
                                alpha_manager.add_subdict(f"boundaryField/{boundary}", {
                                    "type": "zeroGradient"
                                })
                            else:
                                alpha_manager.add_subdict(f"boundaryField/{boundary}", {
                                    "type": "zeroGradient"
                                })
                    
                    # Write changes
                    alpha_manager.write()
            
            # Check p or p_rgh file
            p_file = os.path.join(self.case_dir, "0", "p")
            p_rgh_file = os.path.join(self.case_dir, "0", "p_rgh")
            
            pressure_file = p_rgh_file if "interFoam" in solver and os.path.exists(p_rgh_file) else p_file
            
            if os.path.exists(pressure_file):
                p_manager = CaseDictManager(pressure_file)
                
                # Check if boundaryField exists
                if not p_manager.has_subdict("boundaryField"):
                    p_manager.add_subdict("boundaryField", {})
                
                # Check each boundary
                boundary_field = p_manager.get_subdict("boundaryField")
                for boundary in boundaries:
                    if boundary not in boundary_field:
                        # Add default boundary condition based on name
                        if "inlet" in boundary.lower():
                            p_manager.add_subdict(f"boundaryField/{boundary}", {
                                "type": "zeroGradient"
                            })
                        elif "outlet" in boundary.lower():
                            p_manager.add_subdict(f"boundaryField/{boundary}", {
                                "type": "fixedValue",
                                "value": "uniform 0"
                            })
                        elif "wall" in boundary.lower():
                            p_manager.add_subdict(f"boundaryField/{boundary}", {
                                "type": "zeroGradient"
                            })
                        else:
                            p_manager.add_subdict(f"boundaryField/{boundary}", {
                                "type": "zeroGradient"
                            })
                
                # Write changes
                p_manager.write()
            
            self.log_update.emit("Boundary conditions verified and fixed if needed")
            
        except Exception as e:
            import traceback
            self.log_update.emit(f"Warning: Could not verify boundary conditions: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _get_available_cell_zones(self):
        """Get a list of available cell zones in the mesh."""
        try:
            import os
            
            # Path to constant/polyMesh directory
            mesh_dir = os.path.join(self.case_dir, "constant", "polyMesh")
            
            # Check if cellZones file exists
            cell_zones_file = os.path.join(mesh_dir, "cellZones")
            if not os.path.exists(cell_zones_file):
                return []
            
            # Try to parse the cellZones file
            from ..openfoam_integration.case_manager import CaseDictManager
            
            try:
                # Try to use CaseDictManager to parse it
                dict_manager = CaseDictManager(cell_zones_file)
                return list(dict_manager.get_root_keys())
            except:
                # Fallback: simple parsing
                with open(cell_zones_file, 'r') as f:
                    content = f.read()
                    import re
                    zones = re.findall(r'name\s+([^;]+);', content)
                    return [z.strip() for z in zones]
                
        except Exception as e:
            logger.error(f"Error reading cell zones: {e}")
            return []
        

    def _create_epsilon_field(self, file_path):
        """
        Create a default epsilon (turbulent dissipation rate) field file.
        
        Args:
            file_path (str): Path where the epsilon field file should be created
        """
        try:
            # Create epsilon field template
            content = """/*--------------------------------*- C++ -*----------------------------------*\\
    | =========                 |                                                 |
    | \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
    |  \\\\    /   O peration     | Version:  v2312                                 |
    |   \\\\  /    A nd           | Website:  [www.openfoam.com](www.openfoam.com)                      |
    |    \\\\/     M anipulation  |                                                 |
    \\*---------------------------------------------------------------------------*/
    FoamFile
    {
        version     2.0;
        format      ascii;
        class       volScalarField;
        object      epsilon;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    
    dimensions      [0 2 -3 0 0 0 0];
    
    internalField   uniform 0.01;
    
    boundaryField
    {
        allBoundary
        {
            type            epsilon;
            value           uniform 0.01;
        }
        
        #includeEtc "caseDicts/setConstraintTypes"
    }
    
    // ************************************************************************* //
    """
            # Write to file
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(content)
                
            # Update with boundary-specific settings
            self._update_boundary_specific_settings(file_path, "epsilon")
            
            logger.info(f"Created epsilon field file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating epsilon field: {e}")
            return False
    
    def _update_boundary_specific_settings(self, file_path, field_name):
        """
        Update boundary-specific settings for a turbulence field.
        
        Args:
            file_path (str): Path to the field file
            field_name (str): Name of the field (k, epsilon, omega)
        """
        try:
            # Read the file
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Get boundary conditions 
            for boundary_name, bc in self.boundary_conditions.items():
                bc_type = bc.get('type', 'wall')
                
                # Determine values based on boundary type and field
                if bc_type == 'inlet':
                    value = "0.01"  # Default value
                    if field_name == 'k':
                        value = "0.01"  # Default TKE
                    elif field_name == 'omega':
                        value = "10"    # Default omega
                    elif field_name == 'epsilon':
                        value = "0.01"  # Default epsilon
                    
                    # Update inlet boundary with fixed value
                    pattern = rf'({boundary_name}\s*\n\s*{{[^}}]*type\s+\w+;)'
                    replacement = f'\\1\n        {field_name}       {value};'
                    content = re.sub(pattern, replacement, content)
                    
                elif bc_type == 'outlet':
                    # Use zeroGradient for outlets
                    pattern = rf'({boundary_name}\s*\n\s*{{[^}}]*type\s+\w+;)'
                    replacement = f'\\1\n        {field_name}       zeroGradient;'
                    content = re.sub(pattern, replacement, content)
                    
                elif bc_type == 'wall':
                    # Use wall functions for walls
                    if field_name == 'k':
                        bc_value = "kqRWallFunction;"
                    elif field_name == 'omega':
                        bc_value = "omegaWallFunction;"
                    elif field_name == 'epsilon':
                        bc_value = "epsilonWallFunction;"
                    else:
                        bc_value = "zeroGradient;"
                        
                    pattern = rf'({boundary_name}\s*\n\s*{{[^}}]*type\s+\w+;)'
                    replacement = f'\\1\n        {field_name}       {bc_value}'
                    content = re.sub(pattern, replacement, content)
                    
            # Write modified content
            with open(file_path, 'w') as f:
                f.write(content)
                
        except Exception as e:
            logger.error(f"Error updating boundary settings for {field_name}: {e}")
            logger.error(traceback.format_exc())

    def _create_nut_field(self, file_path):
        """
        Create a default nut (turbulent viscosity) field file.
        
        Args:
            file_path (str): Path where the nut field file should be created
        """
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
    | =========                 |                                                 |
    | \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
    |  \\\\    /   O peration     | Version:  v2312                                 |
    |   \\\\  /    A nd           | Website:  www.openfoam.com                      |
    |    \\\\/     M anipulation  |                                                 |
    \\*---------------------------------------------------------------------------*/
    FoamFile
    {
        version     2.0;
        format      ascii;
        class       volScalarField;
        object      nut;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    dimensions      [0 2 -1 0 0 0 0];

    internalField   uniform 0;

    boundaryField
    {
        allBoundary
        {
            type            calculated;
            value           uniform 0;
        }
        
        // Auto-apply wall functions for walls
        ".*[wW]all.*"
        {
            type            nutkWallFunction;
            value           uniform 0;
        }
        
        // Auto-detect inlets
        ".*[iI]nlet.*"
        {
            type            calculated;
            value           uniform 0;
        }
        
        // Auto-detect outlets
        ".*[oO]utlet.*"
        {
            type            calculated;
            value           uniform 0;
        }
    }

    // ************************************************************************* //
    """
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write the file
        with open(file_path, 'w') as f:
            f.write(content)
            
        self.log_update.emit(f"Created new nut (turbulent viscosity) field file")
        
        # Apply boundary-specific settings if available
        try:
            self._update_boundary_specific_settings(file_path, "nut")
        except Exception as e:
            import traceback
            logger.error(f"Error updating boundary settings for nut: {str(e)}")
            logger.error(traceback.format_exc())


    def _create_k_field(self, file_path):
        """
        Create a default k (turbulent kinetic energy) field file.
        
        Args:
            file_path (str): Path where the k field file should be created
        """
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
    | =========                 |                                                 |
    | \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
    |  \\\\    /   O peration     | Version:  v2312                                 |
    |   \\\\  /    A nd           | Website:  www.openfoam.com                      |
    |    \\\\/     M anipulation  |                                                 |
    \\*---------------------------------------------------------------------------*/
    FoamFile
    {
        version     2.0;
        format      ascii;
        class       volScalarField;
        object      k;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    dimensions      [0 2 -2 0 0 0 0];

    internalField   uniform 0.1;

    boundaryField
    {
        allBoundary
        {
            type            kqRWallFunction;
            value           uniform 0.1;
        }
        
        ".*[iI]nlet.*"
        {
            type            fixedValue;
            value           uniform 0.1;
        }
        
        ".*[oO]utlet.*"
        {
            type            zeroGradient;
        }
    }

    // ************************************************************************* //
    """
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write the file
        with open(file_path, 'w') as f:
            f.write(content)
            
        self.log_update.emit(f"Created new k (turbulent kinetic energy) field file")
        
        # Apply boundary-specific settings if available
        try:
            self._update_boundary_specific_settings(file_path, "k")
        except Exception as e:
            import traceback
            logger.error(f"Error updating boundary settings for k: {str(e)}")
            logger.error(traceback.format_exc())


        def _create_omega_field(self, file_path):
            """
            Create a default omega (specific dissipation rate) field file.
            
            Args:
                file_path (str): Path where the omega field file should be created
            """
            content = """/*--------------------------------*- C++ -*----------------------------------*/
        | =========                 |                                                 |
        | \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
        |  \\    /   O peration     | Version:  v2312                                 |
        |   \\  /    A nd           | Website:  www.openfoam.com                      |
        |    \\/     M anipulation  |                                                 |
        \*---------------------------------------------------------------------------*/
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       volScalarField;
            object      omega;
        }
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
        
        dimensions      [0 0 -1 0 0 0 0];
        
        // Use a higher initial value to prevent issues with wall functions
        internalField   uniform 10;
        
        boundaryField
        {
            // Default boundary treatment for all boundaries
            // Using conservative settings to prevent numerical issues
            allBoundary
            {
                type            fixedValue;
                value           uniform 10;
            }
            
            // Auto-detect walls and use robust settings
            ".*[wW]all.*"
            {
                type            omegaWallFunction;
                // Using blended wall function with safer coefficients
                blended         true;
                // Higher value for numerical stability
                value           uniform 10;
            }
            
            // Auto-detect inlets
            ".*[iI]nlet.*"
            {
                type            fixedValue;
                value           uniform 10;
            }
            
            // Auto-detect outlets
            ".*[oO]utlet.*"
            {
                type            zeroGradient;
            }
        }
        
        // ************************************************************************* //
        """
            
            # Create directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write the file
            with open(file_path, 'w') as f:
                f.write(content)
                
            self.log_update.emit(f"Created new omega (specific dissipation rate) field file with robust wall function settings")
            
            # Apply boundary-specific settings if available
            try:
                self._update_boundary_specific_settings_omega(file_path)
            except Exception as e:
                import traceback
                logger.error(f"Error updating boundary settings for omega: {str(e)}")
                logger.error(traceback.format_exc())
    
    def _create_omega_field(self, file_path):
        """
        Create a default omega (specific dissipation rate) field file.
        
        Args:
            file_path (str): Path where the omega field file should be created
        """
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
    | =========                 |                                                 |
    | \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
    |  \\\\    /   O peration     | Version:  v2312                                 |
    |   \\\\  /    A nd           | Website:  www.openfoam.com                      |
    |    \\\\/     M anipulation  |                                                 |
    \\*---------------------------------------------------------------------------*/
    FoamFile
    {
        version     2.0;
        format      ascii;
        class       volScalarField;
        object      omega;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    dimensions      [0 0 -1 0 0 0 0];

    internalField   uniform 0.1;

    boundaryField
    {
        allBoundary
        {
            type            omegaWallFunction;
            value           uniform 0.1;
        }
        
        ".*[iI]nlet.*"
        {
            type            fixedValue;
            value           uniform 0.1;
        }
        
        ".*[oO]utlet.*"
        {
            type            zeroGradient;
        }
    }

    // ************************************************************************* //
    """
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write the file
        with open(file_path, 'w') as f:
            f.write(content)
            
        self.log_update.emit(f"Created new omega (specific dissipation rate) field file")
        
        # Apply boundary-specific settings if available
        try:
            self._update_boundary_specific_settings(file_path, "omega")
        except Exception as e:
            import traceback
            logger.error(f"Error updating boundary settings for omega: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _ensure_thermophysical_properties(self):
        """Ensure thermophysical properties file exists for compressible solvers."""
        try:
            import os
            
            # Get the solver - different solvers need different thermophysical models
            solver = self.solver_combo.currentText()
            
            # Path to thermophysicalProperties dictionary
            thermo_props_path = os.path.join(self.case_dir, "constant", "thermophysicalProperties")
            
            # Create constant directory if it doesn't exist
            constant_dir = os.path.join(self.case_dir, "constant")
            os.makedirs(constant_dir, exist_ok=True)
            
            # Create appropriate thermophysical properties based on solver
            if "rhoPimpleFoam" in solver:
                content = """/*--------------------------------*- C++ -*----------------------------------*\\
    | =========                 |                                                 |
    | \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
    |  \\\\    /   O peration     | Version:  v2312                                 |
    |   \\\\  /    A nd           | Website:  [www.openfoam.com](www.openfoam.com)                      |
    |    \\\\/     M anipulation  |                                                 |
    \\*---------------------------------------------------------------------------*/
    FoamFile
    {
        version     2.0;
        format      ascii;
        class       dictionary;
        object      thermophysicalProperties;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    
    thermoType
    {
        type            hePsiThermo;
        mixture         pureMixture;
        transport       const;
        thermo          hConst;
        equationOfState perfectGas;
        specie          specie;
        energy          sensibleEnthalpy;
    }
    
    mixture
    {
        specie
        {
            molWeight   28.9;
        }
        thermodynamics
        {
            Cp          1004.5;
            Hf          0;
        }
        transport
        {
            mu          1.8e-05;
            Pr          0.7;
        }
    }
    
    // ************************************************************************* //
    """
            elif "reactingFoam" in solver:
                # Different thermophysical setup for reacting flows
                content = """/*--------------------------------*- C++ -*----------------------------------*\\
    | =========                 |                                                 |
    | \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
    |  \\\\    /   O peration     | Version:  v2312                                 |
    |   \\\\  /    A nd           | Website:  [www.openfoam.com](www.openfoam.com)                      |
    |    \\\\/     M anipulation  |                                                 |
    \\*---------------------------------------------------------------------------*/
    FoamFile
    {
        version     2.0;
        format      ascii;
        class       dictionary;
        object      thermophysicalProperties;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    
    thermoType
    {
        type            hePsiThermo;
        mixture         reactingMixture;
        transport       sutherland;
        thermo          janaf;
        energy          sensibleEnthalpy;
        equationOfState perfectGas;
        specie          specie;
    }
    
    // Basic air components for a simple reaction
    species
    (
        O2
        N2
    );
    
    O2
    {
        specie
        {
            molWeight   31.9988;
        }
        thermodynamics
        {
            Tlow            200;
            Thigh           5000;
            Tcommon         1000;
            highCpCoeffs    ( 3.69758 0.00061352 -1.25884e-07 1.77528e-11 -1.13644e-15 -1233.93 3.18917 );
            lowCpCoeffs     ( 3.21294 0.00112749 -5.75615e-07 1.31388e-09 -8.76855e-13 -1005.25 6.03474 );
        }
        transport
        {
            As              1.67212e-06;
            Ts              170.672;
        }
    }
    
    N2
    {
        specie
        {
            molWeight       28.0134;
        }
        thermodynamics
        {
            Tlow            200;
            Thigh           5000;
            Tcommon         1000;
            highCpCoeffs    ( 2.92664 0.00148798 -5.68476e-07 1.0097e-10 -6.75335e-15 -922.798 5.98053 );
            lowCpCoeffs     ( 3.29868 0.00140824 -3.96322e-06 5.64152e-09 -2.44486e-12 -1020.9 3.95037 );
        }
        transport
        {
            As              1.67212e-06;
            Ts              170.672;
        }
    }
    
    // ************************************************************************* //
    """
            else:
                # Generic thermophysical properties for other solvers that may need it
                content = """/*--------------------------------*- C++ -*----------------------------------*\\
    | =========                 |                                                 |
    | \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
    |  \\\\    /   O peration     | Version:  v2312                                 |
    |   \\\\  /    A nd           | Website:  [www.openfoam.com](www.openfoam.com)                      |
    |    \\\\/     M anipulation  |                                                 |
    \\*---------------------------------------------------------------------------*/
    FoamFile
    {
        version     2.0;
        format      ascii;
        class       dictionary;
        object      thermophysicalProperties;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    
    thermoType
    {
        type            hePsiThermo;
        mixture         pureMixture;
        transport       const;
        thermo          hConst;
        equationOfState perfectGas;
        specie          specie;
        energy          sensibleEnthalpy;
    }
    
    mixture
    {
        specie
        {
            molWeight   28.9;
        }
        thermodynamics
        {
            Cp          1004.5;
            Hf          0;
        }
        transport
        {
            mu          1.8e-05;
            Pr          0.7;
        }
    }
    
    // ************************************************************************* //
    """
            
            # Write to file
            with open(thermo_props_path, 'w') as f:
                f.write(content)
                
            self.log_update.emit(f"Created thermophysical properties for {solver}")
            return True
            
        except Exception as e:
            import traceback
            self.log_update.emit(f"Error creating thermophysical properties: {str(e)}")
            logger.error(f"Error in _ensure_thermophysical_properties: {e}")
            logger.error(traceback.format_exc())
            return False

    def _ensure_compressible_fields(self):
        """Ensure fields required for compressible solvers exist."""
        try:
            import os
            import re
            import subprocess
            
            # Get the solver
            solver = self.solver_combo.currentText()
            
            # Check if we need to ensure compressible fields
            if not any(s in solver for s in ["rhoPimpleFoam", "rhoSimpleFoam", "reactingFoam"]):
                return True  # Not a compressible solver, no action needed
                
            self.log_update.emit("Setting up compressible flow fields")
            
            # Ensure the 0 directory exists
            zero_dir = os.path.join(self.case_dir, "0")
            os.makedirs(zero_dir, exist_ok=True)
            
            # Get boundary patches directly from the mesh using OpenFOAM's checkMesh utility
            boundary_patches = []
            
            # First try the polyMesh/boundary file directly
            boundary_file = os.path.join(self.case_dir, "constant", "polyMesh", "boundary")
            if os.path.exists(boundary_file):
                self.log_update.emit("Reading boundary patch names from polyMesh/boundary file")
                with open(boundary_file, 'r') as f:
                    content = f.read()
                
                # Look for boundary entries in the format "patchName { type ... }"
                patches = re.findall(r'(\w+)\s*\n\s*{', content)
                
                if patches:
                    # Filter out any entries that are clearly not boundary patches
                    boundary_patches = [p for p in patches if p not in ["FoamFile", "boundaryField"]]
                    self.log_update.emit(f"Found boundary patches: {', '.join(boundary_patches)}")
            
            # If that doesn't work, try existing field files
            if not boundary_patches:
                for field_name in ["U", "p", "k"]:
                    field_path = os.path.join(zero_dir, field_name)
                    if os.path.exists(field_path):
                        self.log_update.emit(f"Reading boundary patch names from {field_name} file")
                        with open(field_path, 'r') as f:
                            content = f.read()
                        
                        # Extract boundary field section
                        if "boundaryField" in content:
                            boundary_section = content.split("boundaryField")[1]
                            # Find all patch names
                            patches = re.findall(r'(\w+)\s*\n\s*{', boundary_section)
                            
                            # Remove non-patch entries
                            if "allBoundary" in patches:
                                patches.remove("allBoundary")
                            
                            if patches:
                                boundary_patches = patches
                                self.log_update.emit(f"Found boundary patches: {', '.join(boundary_patches)}")
                                break
            
            # If still no patches found, try running checkMesh
            if not boundary_patches:
                try:
                    self.log_update.emit("Trying to determine boundary patches with checkMesh")
                    result = subprocess.run(
                        ["checkMesh", "-case", self.case_dir],
                        capture_output=True, 
                        text=True
                    )
                    
                    if result.stdout:
                        # Look for patch information in checkMesh output
                        for line in result.stdout.split("\n"):
                            if "Patch" in line and "faces" in line:
                                # Lines like "    Patch xmin, faces: 1, type: patch"
                                parts = line.strip().split(',')
                                if len(parts) > 0:
                                    patch_name = parts[0].replace("Patch ", "").strip()
                                    boundary_patches.append(patch_name)
                except Exception as e:
                    self.log_update.emit(f"Error running checkMesh: {str(e)}")
            
            # If still no patches found, use default axis-aligned names that are common in blockMesh
            if not boundary_patches:
                self.log_update.emit("Warning: Could not determine boundary patch names, using defaults")
                boundary_patches = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
            
            # Create alphat field (turbulent thermal diffusivity)
            alphat_path = os.path.join(zero_dir, "alphat")
            
            # Create alphat with the detected boundary patches
            self.log_update.emit(f"Creating alphat field with patches: {', '.join(boundary_patches)}")
            
            # Start building the content
            alphat_header = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      alphat;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [1 -1 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{
"""
            
            # Build the boundary field section with the actual patch names
            boundary_content = ""
            for patch in boundary_patches:
                # Default is calculated type for most boundaries
                if "wall" in patch.lower():
                    # Wall-specific condition
                    boundary_content += f"""    {patch}
    {{
        type            compressible::alphatWallFunction;
        Prt             0.85;
        value           uniform 0;
    }}
    
"""
                else:
                    # Default for other boundaries
                    boundary_content += f"""    {patch}
    {{
        type            calculated;
        value           uniform 0;
    }}
    
"""
            
            # Close the file
            alphat_footer = """}

// ************************************************************************* //
"""
            
            # Combine all parts
            alphat_content = alphat_header + boundary_content + alphat_footer
            
            # Write the file
            with open(alphat_path, 'w') as f:
                f.write(alphat_content)
            
            # Create T field (temperature) if needed
            T_path = os.path.join(zero_dir, "T")
            if not os.path.exists(T_path):
                self.log_update.emit("Creating T field for compressible solver")
                
                # Start building the content
                T_header = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      T;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 0 1 0 0 0];

internalField   uniform 300;

boundaryField
{
"""
                
                # Build the boundary field section with the actual patch names
                boundary_content = ""
                for patch in boundary_patches:
                    if "inlet" in patch.lower():
                        # Inlet-specific condition
                        boundary_content += f"""    {patch}
    {{
        type            fixedValue;
        value           uniform 300;
    }}
    
"""
                    else:
                        # Default for other boundaries
                        boundary_content += f"""    {patch}
    {{
        type            zeroGradient;
    }}
    
"""
                
                # Close the file
                T_footer = """}

// ************************************************************************* //
"""
                
                # Combine all parts
                T_content = T_header + boundary_content + T_footer
                
                # Write the file
                with open(T_path, 'w') as f:
                    f.write(T_content)
            
            # Create p field with compressible settings if not exists
            p_path = os.path.join(zero_dir, "p")
            if os.path.exists(p_path):
                self.log_update.emit("Updating p field for compressible solver")
                # Read existing file
                with open(p_path, 'r') as f:
                    p_content = f.read()
                
                # Check if dimensions are already set for compressible flow
                # For compressible flow, p should have dimensions [1 -1 -2 0 0 0 0]
                if "dimensions      [1 -1 -2 0 0 0 0]" not in p_content:
                    # Replace dimensions line
                    p_content = re.sub(r'dimensions\s+\[[^\]]+\]', 'dimensions      [1 -1 -2 0 0 0 0]', p_content)
                    
                    # Write back
                    with open(p_path, 'w') as f:
                        f.write(p_content)
            
            # Additional fields for reactingFoam if needed
            if "reactingFoam" in solver:
                # Create species fields
                for species in ["O2", "N2"]:
                    species_path = os.path.join(zero_dir, species)
                    if not os.path.exists(species_path):
                        self.log_update.emit(f"Creating {species} field for reacting solver")
                        
                        # Start building the content
                        species_header = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      {species};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 0 0 0 0 0];

internalField   uniform {"0.233" if species == "O2" else "0.767"};

boundaryField
{{
"""
                        
                        # Build the boundary field section with the actual patch names
                        boundary_content = ""
                        for patch in boundary_patches:
                            if "inlet" in patch.lower():
                                # Inlet-specific condition
                                boundary_content += f"""    {patch}
    {{
        type            fixedValue;
        value           uniform {"0.233" if species == "O2" else "0.767"};
    }}
    
"""
                            else:
                                # Default for other boundaries
                                boundary_content += f"""    {patch}
    {{
        type            zeroGradient;
    }}
    
"""
                        
                        # Close the file
                        species_footer = """}}

// ************************************************************************* //
"""
                        
                        # Combine all parts
                        species_content = species_header + boundary_content + species_footer
                        
                        # Write the file
                        with open(species_path, 'w') as f:
                            f.write(species_content)
            
            self.log_update.emit("Successfully created compressible flow fields")
            return True
            
        except Exception as e:
            import traceback
            self.log_update.emit(f"Error creating compressible fields: {str(e)}")
            logger.error(f"Error in _ensure_compressible_fields: {e}")
            logger.error(traceback.format_exc())
            return False
    def _fix_pressure_dimensions(self):
        """
        Fix pressure dimensions based on solver type.
        
        Different OpenFOAM solvers require specific pressure dimensions:
        - Incompressible solvers (simpleFoam, pisoFoam): [0 2 -2 0 0 0 0] (kinematic pressure, p/rho)
        - Compressible solvers (rhoPimpleFoam): [1 -1 -2 0 0 0 0] (dynamic pressure, Pa)
        """
        try:
            import os
            import re
            
            # Get solver type
            solver = self.solver_combo.currentText()
            
            # Determine if it's a compressible solver
            is_compressible = any(s in solver for s in [
                "rhoPimpleFoam", "rhoSimpleFoam", "reactingFoam", 
                "rhoCentralFoam", "dbnsFoam", "sonicFoam"
            ])
            
            # Path to pressure field
            zero_dir = os.path.join(self.case_dir, "0")
            p_path = os.path.join(zero_dir, "p")
            
            if not os.path.exists(p_path):
                self.log_update.emit("Warning: p field not found, cannot fix dimensions")
                return True
            
            # Read the pressure field file
            with open(p_path, 'r') as f:
                p_content = f.read()
            
            # Set correct pressure dimensions based on solver type
            if is_compressible:
                # For compressible flow, p should have dimensions [1 -1 -2 0 0 0 0]
                # This is dynamic pressure with units of Pa (kg/m/s²)
                self.log_update.emit("Setting pressure dimensions for compressible flow")
                p_content = re.sub(r'dimensions\s+\[[^\]]+\]', 
                                  'dimensions      [1 -1 -2 0 0 0 0]', 
                                  p_content)
                
                # Set initial pressure to atmospheric (1e5 Pa) for compressible solvers
                p_content = re.sub(r'internalField\s+uniform\s+[0-9.e+-]+', 
                                  'internalField   uniform 100000', 
                                  p_content)
            else:
                # For incompressible flow, p should have dimensions [0 2 -2 0 0 0 0]
                # This is kinematic pressure (p/rho) with units of m²/s²
                self.log_update.emit("Setting pressure dimensions for incompressible flow")
                p_content = re.sub(r'dimensions\s+\[[^\]]+\]', 
                                  'dimensions      [0 2 -2 0 0 0 0]', 
                                  p_content)
                
                # Use a reasonable value for kinematic pressure in incompressible solvers
                p_content = re.sub(r'internalField\s+uniform\s+[0-9.e+-]+', 
                                  'internalField   uniform 0', 
                                  p_content)
            
            # Write the updated content back to the file
            with open(p_path, 'w') as f:
                f.write(p_content)
            
            self.log_update.emit(f"Successfully set correct pressure dimensions for {solver}")
            return True
        
        except Exception as e:
            import traceback
            self.log_update.emit(f"Error fixing pressure dimensions: {str(e)}")
            logger.error(f"Error in _fix_pressure_dimensions: {e}")
            logger.error(traceback.format_exc())
            return False
