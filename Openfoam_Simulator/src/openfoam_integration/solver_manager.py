#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenFOAM solver manager for Openfoam_Simulator.

This module provides an interface to OpenFOAM solvers, allowing the application to:
- Select appropriate solvers based on simulation type
- Configure solver parameters
- Launch and monitor solver execution
- Process solver outputs
- Provide status updates and error handling

The module acts as a bridge between the UI components and the underlying OpenFOAM installation.
"""

import os
import sys
import subprocess
import logging
import threading
import time
import re
import signal
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Import utility modules
from ..utils.logger import get_logger
from ..utils.progress_tracker import ProgressTracker
from ..config import get_value, set_value

# Import solver-specific modules
from .solvers.singlePhase import configure_single_phase_solver
from .solvers.multiPhase import configure_multi_phase_solver
from .solvers.pigging import configure_pigging_solver
from .solvers.spillModels import configure_spill_solver

# Set up module logger
logger = get_logger(__name__)


class SolverManager:
    """
    Class for managing OpenFOAM solvers.
    
    This class provides methods to configure, launch, monitor, and control
    OpenFOAM solvers for various types of CFD simulations commonly used
    in oil & gas applications.
    """
    
    # Solver types
    SINGLE_PHASE = "singlePhase"
    MULTI_PHASE = "multiPhase"
    PIGGING = "pigging"
    SPILL = "spill"
    
    def __init__(self, case_dir: str = None):
        """
        Initialize the solver manager.
        
        Args:
            case_dir (str, optional): Path to the OpenFOAM case directory.
        """
        self.case_dir = case_dir
        
        # Initialize state variables
        self.solver_process = None
        self.solver_thread = None
        self.monitor_thread = None
        self.is_running = False
        self.is_monitoring = False
        self.stop_requested = False
        
        # Store solver information
        self.solver_type = None
        self.solver_name = None
        self.parallel = False
        self.processors = 1
        
        # Progress tracking
        self.progress_tracker = ProgressTracker()
        self.progress_callbacks = []
        self.completion_callbacks = []
        self.error_callbacks = []
        
        # Results storage
        self.residuals = {}
        self.time_values = []
        self.field_values = {}
        self.solver_output = []
        self.convergence_status = False
        
        # Cache dictionary for solver parameters
        self.solver_params = {}
        
        # Detect OpenFOAM installation
        self._detect_openfoam()
    
    def _detect_openfoam(self):
        """Detect and validate OpenFOAM installation."""
        # Check for WM_PROJECT_DIR environment variable (set when OpenFOAM is sourced)
        self.foam_dir = os.environ.get('WM_PROJECT_DIR')
        if self.foam_dir and os.path.exists(self.foam_dir):
            self.foam_version = os.environ.get('WM_PROJECT_VERSION', 'unknown')
            logger.info(f"Found OpenFOAM installation at {self.foam_dir} (version {self.foam_version})")
        else:
            # Try to find OpenFOAM in standard locations
            std_locations = [
                '/opt/openfoam12',
                '/opt/openfoam-12',
                '/usr/lib/openfoam',
                '/usr/local/lib/openfoam',
            ]
            
            for loc in std_locations:
                if os.path.exists(loc):
                    self.foam_dir = loc
                    # Try to determine version
                    version_file = os.path.join(loc, 'etc', 'bashrc')
                    if os.path.exists(version_file):
                        try:
                            with open(version_file, 'r') as f:
                                content = f.read()
                                version_match = re.search(r'WM_PROJECT_VERSION=(["\'])(.*?)\1', content)
                                if version_match:
                                    self.foam_version = version_match.group(2)
                                else:
                                    self.foam_version = 'unknown'
                        except Exception as e:
                            logger.warning(f"Failed to determine OpenFOAM version: {e}")
                            self.foam_version = 'unknown'
                    else:
                        self.foam_version = 'unknown'
                    
                    logger.info(f"Found OpenFOAM installation at {self.foam_dir} (version {self.foam_version})")
                    break
            else:
                logger.warning("No OpenFOAM installation found. Solvers will not work.")
                self.foam_dir = None
                self.foam_version = None
    
    def set_case_directory(self, case_dir: str):
        """
        Set the OpenFOAM case directory.
        
        Args:
            case_dir (str): Path to the OpenFOAM case directory.
        """
        self.case_dir = case_dir
        logger.info(f"Set case directory to: {case_dir}")
    
    def get_available_solvers(self) -> Dict[str, List[str]]:
        """
        Get list of available OpenFOAM solvers by category.
        
        Returns:
            Dict[str, List[str]]: Dictionary of solver categories and names.
        """
        solvers = {
            self.SINGLE_PHASE: [
                "simpleFoam",       # Steady-state incompressible
                "pisoFoam",         # Transient incompressible
                "pimpleFoam",       # Transient incompressible with PIMPLE algorithm
                "pimpleDyMFoam",    # Transient incompressible with dynamic mesh
                "buoyantBoussinesqPimpleFoam",  # Thermal flows
                "rhoPimpleFoam",    # Transient compressible
                "rhoSimpleFoam"     # Steady-state compressible
            ],
            self.MULTI_PHASE: [
                "interFoam",        # Two-phase immiscible incompressible
                "multiphaseInterFoam",  # N-phase immiscible incompressible
                "compressibleInterFoam",  # Compressible two-phase 
                "twoPhaseEulerFoam",  # Two-phase Eulerian
                "reactingTwoPhaseEulerFoam",  # Reacting two-phase Eulerian
                "driftFluxFoam"     # Drift flux model
            ],
            self.PIGGING: [
                "interFoam",        # Can be adapted for pigging
                "multiphaseInterFoam",  # Better for pigging with multiple phases
                "pimpleDyMFoam"     # For moving boundary (pig) simulations
            ],
            self.SPILL: [
                "interFoam",        # Basic spill model
                "multiphaseInterFoam",  # Multi-component spill
                "reactingMultiphaseInterFoam",  # With chemical reactions
                "shallowWaterFoam", # Surface spill modeling
                "driftFluxFoam"     # For subsea dispersion
            ]
        }
        
        # Verify solver existence if OpenFOAM is found
        if self.foam_dir:
            for category in solvers:
                verified_solvers = []
                for solver in solvers[category]:
                    # Check in standard OpenFOAM bin directory
                    solver_path = os.path.join(self.foam_dir, "platforms", "*", "bin", solver)
                    if list(Path(self.foam_dir).glob(solver_path.replace(self.foam_dir + os.sep, ""))):
                        verified_solvers.append(solver)
                
                solvers[category] = verified_solvers
        
        return solvers
    
    def get_available_turbulence_models(self) -> List[str]:
        """
        Get list of available OpenFOAM turbulence models.
        
        Returns:
            List[str]: List of turbulence model names.
        """
        models = [
            "laminar",
            "kEpsilon",
            "kOmega",
            "kOmegaSST",
            "SpalartAllmaras",
            "realizableKE",
            "RNGkEpsilon",
            "LaunderSharmaKE",
            "LRR",          # Reynolds stress model
            "LES",          # Large Eddy Simulation
            "Smagorinsky",  # LES model
            "dynamicKEqn",  # Dynamic LES model
            "WALE"          # Wall-Adapting Local Eddy-viscosity LES model
        ]
        
        return models
    
    def get_recommended_solver(self, simulation_type: str, is_transient: bool = False, 
                              is_compressible: bool = False) -> str:
        """
        Get recommended solver for the given simulation type.
        
        Args:
            simulation_type (str): Type of simulation.
            is_transient (bool, optional): Whether the simulation is transient.
            is_compressible (bool, optional): Whether the flow is compressible.
            
        Returns:
            str: Name of the recommended solver.
        """
        if simulation_type == self.SINGLE_PHASE:
            if is_compressible:
                return "rhoSimpleFoam" if not is_transient else "rhoPimpleFoam"
            else:
                return "simpleFoam" if not is_transient else "pimpleFoam"
        
        elif simulation_type == self.MULTI_PHASE:
            if is_compressible:
                return "compressibleInterFoam"
            else:
                return "interFoam"
        
        elif simulation_type == self.PIGGING:
            return "multiphaseInterFoam"
        
        elif simulation_type == self.SPILL:
            return "multiphaseInterFoam"
        
        # Default
        return "simpleFoam"
    
    def configure_solver(self, solver_type: str, solver_name: str = None, 
                        parameters: Dict[str, Any] = None, parallel: bool = False,
                        processors: int = 4):
        """
        Configure a solver for the simulation.
        
        Args:
            solver_type (str): Type of solver to configure.
            solver_name (str, optional): Name of the specific solver to use.
            parameters (Dict[str, Any], optional): Dictionary of solver parameters.
            parallel (bool, optional): Whether to run in parallel.
            processors (int, optional): Number of processors for parallel runs.
            
        Returns:
            bool: True if configuration was successful, False otherwise.
        """
        if not self.case_dir:
            logger.error("No case directory set. Cannot configure solver.")
            return False
        
        # Store basic solver information
        self.solver_type = solver_type
        self.parallel = parallel
        self.processors = processors
        
        # If no solver name provided, use recommended one
        if not solver_name:
            is_transient = parameters.get('transient', False) if parameters else False
            is_compressible = parameters.get('compressible', False) if parameters else False
            solver_name = self.get_recommended_solver(solver_type, is_transient, is_compressible)
        
        self.solver_name = solver_name
        
        # Store parameters for later use
        self.solver_params = parameters or {}
        
        # Create necessary directories if they don't exist
        system_dir = os.path.join(self.case_dir, "system")
        constant_dir = os.path.join(self.case_dir, "constant")
        
        os.makedirs(system_dir, exist_ok=True)
        os.makedirs(constant_dir, exist_ok=True)
        
        try:
            # Call appropriate configuration function based on solver type
            if solver_type == self.SINGLE_PHASE:
                success = configure_single_phase_solver(
                    self.case_dir, solver_name, parameters or {})
            
            elif solver_type == self.MULTI_PHASE:
                success = configure_multi_phase_solver(
                    self.case_dir, solver_name, parameters or {})
            
            elif solver_type == self.PIGGING:
                success = configure_pigging_solver(
                    self.case_dir, solver_name, parameters or {})
            
            elif solver_type == self.SPILL:
                success = configure_spill_solver(
                    self.case_dir, solver_name, parameters or {})
            
            else:
                logger.error(f"Unknown solver type: {solver_type}")
                return False
            
            # Configure for parallel execution if needed
            if parallel and success:
                success = self._configure_parallel(processors)
            
            logger.info(f"Solver configuration {'successful' if success else 'failed'}")
            return success
            
        except Exception as e:
            logger.error(f"Error configuring solver: {e}")
            return False
    
    def _configure_parallel(self, processors: int) -> bool:
        """
        Configure case for parallel execution.
        
        Args:
            processors (int): Number of processors to use.
            
        Returns:
            bool: True if configuration was successful, False otherwise.
        """
        try:
            system_dir = os.path.join(self.case_dir, "system")
            decomp_file = os.path.join(system_dir, "decomposeParDict")
            
            decomp_method = get_value('openfoam.decomposition_method', 'scotch')
            
            # Create decomposeParDict file
            with open(decomp_file, 'w') as f:
                f.write('/*--------------------------------*- C++ -*----------------------------------*\\\n')
                f.write('| =========                 |                                                 |\n')
                f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
                f.write('|  \\\\    /   O peration     | Version:  v2312                                 |\n')
                f.write('|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n')
                f.write('|    \\\\/     M anipulation  |                                                 |\n')
                f.write('\\*---------------------------------------------------------------------------*/\n\n')
                
                f.write('FoamFile\n')
                f.write('{\n')
                f.write('    version     2.0;\n')
                f.write('    format      ascii;\n')
                f.write('    class       dictionary;\n')
                f.write('    object      decomposeParDict;\n')
                f.write('}\n')
                f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n')
                
                f.write(f'numberOfSubdomains {processors};\n\n')
                
                f.write(f'method          {decomp_method};\n\n')
                
                # Add method-specific parameters
                if decomp_method == 'simple':
                    # Try to determine a reasonable decomposition
                    import math
                    n = processors
                    nx = max(1, round(math.pow(n, 1/3)))
                    while n % nx != 0 and nx > 1:
                        nx -= 1
                    n2 = n // nx
                    ny = max(1, round(math.sqrt(n2)))
                    while n2 % ny != 0 and ny > 1:
                        ny -= 1
                    nz = n2 // ny
                    
                    f.write('simpleCoeffs\n')
                    f.write('{\n')
                    f.write(f'    n               ({nx} {ny} {nz});\n')
                    f.write('    delta           0.001;\n')
                    f.write('}\n\n')
                
                elif decomp_method == 'hierarchical':
                    f.write('hierarchicalCoeffs\n')
                    f.write('{\n')
                    f.write('    n               (2 2 1);\n')
                    f.write('    delta           0.001;\n')
                    f.write('    order           xyz;\n')
                    f.write('}\n\n')
                
                f.write('// ************************************************************************* //\n')
            
            return True
        
        except Exception as e:
            logger.error(f"Error configuring parallel execution: {e}")
            return False
    
    def run_solver(self, blocking: bool = False) -> bool:
        """
        Run the configured solver.
        
        Args:
            blocking (bool, optional): Whether to wait for the solver to complete.
            
        Returns:
            bool: True if the solver was started successfully, False otherwise.
        """
        if not self.case_dir or not self.solver_name:
            logger.error("Solver not properly configured. Cannot run.")
            return False
        
        if self.is_running:
            logger.warning("Solver is already running.")
            return False
        
        try:
            # Fix the fvSolution file to ensure it works with all solvers
            try:
                from .fix_fvsolution import fix_fv_solution
                logger.info(f"Applying fvSolution fix for solver: {self.solver_name}")
                fix_fv_solution(self.case_dir)
                logger.info("Successfully fixed fvSolution with both SIMPLE and PISO sections")
            except Exception as e:
                logger.warning(f"Could not fix fvSolution file: {e}")
                import traceback
                logger.warning(traceback.format_exc())
            
            # Reset state
            self.is_running = True
            self.stop_requested = False
            self.residuals = {}
            self.time_values = []
            self.field_values = {}
            self.solver_output = []
            self.convergence_status = False
            
            # Build command
            cmd = []
            
            if self.parallel:
                # For parallel execution, need to decompose the case first
                logger.info(f"Decomposing case for {self.processors} processors")
                decompose_cmd = ["decomposePar", "-case", self.case_dir]
                
                try:
                    # Run decomposePar
                    decompose_proc = subprocess.run(
                        decompose_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                        check=True
                    )
                    logger.debug(f"decomposePar output: {decompose_proc.stdout}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error decomposing case: {e}")
                    logger.error(f"decomposePar stderr: {e.stderr}")
                    self.is_running = False
                    return False
                
                # Command for parallel execution
                cmd = [
                    "mpirun", 
                    "-np", str(self.processors), 
                    self.solver_name, 
                    "-parallel", 
                    "-case", self.case_dir
                ]
            else:
                # Command for serial execution
                cmd = [self.solver_name, "-case", self.case_dir]
            
            # Log the command
            logger.info(f"Running solver: {' '.join(cmd)}")
            
            # Start the solver in a new process
            if blocking:
                # Run in this thread and wait for completion
                self.solver_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # Store process output and parse progress
                for line in self.solver_process.stdout:
                    self.solver_output.append(line.strip())
                    self._parse_solver_output(line)
                
                # Wait for process to complete
                return_code = self.solver_process.wait()
                
                # Process completed
                self.is_running = False
                self.convergence_status = (return_code == 0)
                
                # Call completion callbacks
                for callback in self.completion_callbacks:
                    callback(self.convergence_status)
                
                return (return_code == 0)
            
            else:
                # Run in a separate thread
                self.solver_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # Create threads for execution and monitoring
                self.solver_thread = threading.Thread(
                    target=self._solver_thread_func,
                    args=(self.solver_process,),
                    daemon=True
                )
                
                self.monitor_thread = threading.Thread(
                    target=self._monitor_thread_func,
                    daemon=True
                )
                
                # Start threads
                self.solver_thread.start()
                self.is_monitoring = True
                self.monitor_thread.start()
                
                return True
                
        except Exception as e:
            logger.error(f"Error running solver: {e}")
            self.is_running = False
            
            # Call error callbacks
            for callback in self.error_callbacks:
                callback(str(e))
            
            return False
    
    def _solver_thread_func(self, process):
        """
        Thread function for running the solver process.
        
        Args:
            process: Subprocess handle for the solver.
        """
        try:
            # Read and store process output
            for line in process.stdout:
                self.solver_output.append(line.strip())
                self._parse_solver_output(line)
            
            # Wait for process to complete
            return_code = process.wait()
            
            # Process completed
            self.is_running = False
            self.convergence_status = (return_code == 0)
            
            logger.info(f"Solver completed with return code {return_code}")
            
            # Reconstruct the case if it was run in parallel
            if self.parallel and return_code == 0:
                self._reconstruct_case()
            
            # Call completion callbacks
            for callback in self.completion_callbacks:
                callback(self.convergence_status)
                
        except Exception as e:
            logger.error(f"Error in solver thread: {e}")
            self.is_running = False
            
            # Call error callbacks
            for callback in self.error_callbacks:
                callback(str(e))
    
    def _monitor_thread_func(self):
        """Thread function for monitoring the solver progress."""
        try:
            while self.is_running and self.is_monitoring:
                # Calculate overall progress
                progress_info = self._calculate_progress()
                
                # Call progress callbacks
                for callback in self.progress_callbacks:
                    callback(progress_info)
                
                # Sleep to avoid excessive CPU usage
                time.sleep(1)
            
            # Ensure is_monitoring is reset
            self.is_monitoring = False
            
        except Exception as e:
            logger.error(f"Error in monitor thread: {e}")
            self.is_monitoring = False
    
    def _parse_solver_output(self, line: str):
        """
        Parse a line of solver output to extract progress information.
        
        Args:
            line (str): Line of solver output.
        """
        # Look for time step information
        time_match = re.search(r"Time = ([0-9.e+-]+)", line)
        if time_match:
            try:
                current_time = float(time_match.group(1))
                self.time_values.append(current_time)
            except ValueError:
                pass
        
        # Look for residual information
        residual_match = re.search(r"Solving for ([^,]+), Initial residual = ([^,]+), Final residual = ([^,]+), No Iterations ([0-9]+)", line)
        if residual_match:
            field = residual_match.group(1).strip()
            initial = float(residual_match.group(2))
            final = float(residual_match.group(3))
            iterations = int(residual_match.group(4))
            
            if field not in self.residuals:
                self.residuals[field] = []
            
            self.residuals[field].append(final)
        
        # Look for field min/max values
        field_match = re.search(r"min\(([^)]+)\) = ([^,]+), max\(([^)]+)\) = ([^,]+)", line)
        if field_match:
            field = field_match.group(1).strip()
            min_val = float(field_match.group(2))
            max_val = float(field_match.group(4))
            
            if field not in self.field_values:
                self.field_values[field] = []
            
            self.field_values[field].append((min_val, max_val))
        
        # Special patterns for different solvers could be added here
    
    def _calculate_progress(self) -> Dict[str, Any]:
        """
        Calculate the current progress of the solver.
        
        Returns:
            Dict[str, Any]: Dictionary with progress information.
        """
        # Get parameters needed for progress calculation
        max_time = self.solver_params.get('end_time', 1.0)
        steady_state = self.solver_params.get('steady_state', False)
        max_iterations = self.solver_params.get('max_iterations', 1000)
        
        # Calculate progress percentage
        progress = 0.0
        current_time = self.time_values[-1] if self.time_values else 0.0
        current_iteration = len(self.time_values) if steady_state else 0
        
        if steady_state:
            # For steady-state simulations, use iteration count
            progress = min(100.0, (current_iteration / max_iterations) * 100.0)
        else:
            # For transient simulations, use time
            progress = min(100.0, (current_time / max_time) * 100.0)
        
        # Collect latest residuals
        latest_residuals = {}
        for field, values in self.residuals.items():
            if values:
                latest_residuals[field] = values[-1]
        
        # Create progress info dictionary
        progress_info = {
            "progress": progress,
            "time": current_time,
            "iteration": current_iteration,
            "residuals": latest_residuals,
            "fields": {f: v[-1] if v else (0, 0) for f, v in self.field_values.items()}
        }
        
        return progress_info
    
    def _reconstruct_case(self):
        """Reconstruct the case after a parallel run."""
        logger.info("Reconstructing case after parallel run")
        try:
            # Run reconstructPar
            reconstruct_cmd = ["reconstructPar", "-case", self.case_dir]
            subprocess.run(
                reconstruct_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=True
            )
            logger.info("Case reconstruction completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error reconstructing case: {e}")
            logger.error(f"reconstructPar stderr: {e.stderr}")
    
    def stop_solver(self) -> bool:
        """
        Stop the running solver.
        
        Returns:
            bool: True if the solver was stopped successfully, False otherwise.
        """
        if not self.is_running or not self.solver_process:
            logger.warning("No solver running to stop.")
            return False
        
        try:
            logger.info("Stopping solver...")
            
            # Set stop flag
            self.stop_requested = True
            
            # First try a gentle termination
            self.solver_process.terminate()
            
            # Wait for process to terminate
            try:
                self.solver_process.wait(timeout=10)
                logger.info("Solver terminated gracefully.")
            except subprocess.TimeoutExpired:
                # If it doesn't terminate, force kill it
                logger.warning("Solver did not terminate gracefully, forcing kill...")
                self.solver_process.kill()
                self.solver_process.wait()
                logger.info("Solver forcefully killed.")
            
            # Clean up
            self.is_running = False
            
            return True
            
        except Exception as e:
            logger.error(f"Error stopping solver: {e}")
            return False
    
    def register_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Register a callback function for progress updates.
        
        Args:
            callback: Function that takes a progress info dictionary.
        """
        if callback not in self.progress_callbacks:
            self.progress_callbacks.append(callback)
    
    def register_completion_callback(self, callback: Callable[[bool], None]):
        """
        Register a callback function for completion notification.
        
        Args:
            callback: Function that takes a success flag.
        """
        if callback not in self.completion_callbacks:
            self.completion_callbacks.append(callback)
    
    def register_error_callback(self, callback: Callable[[str], None]):
        """
        Register a callback function for error notification.
        
        Args:
            callback: Function that takes an error message.
        """
        if callback not in self.error_callbacks:
            self.error_callbacks.append(callback)
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the simulation results.
        
        Returns:
            Dict[str, Any]: Dictionary with result summary information.
        """
        # Create summary dictionary
        summary = {
            "converged": self.convergence_status,
            "solver": self.solver_name,
            "type": self.solver_type,
            "times": self.time_values,
            "residuals": self.residuals,
        }
        
        # Add final field values if available
        final_fields = {}
        for field, values in self.field_values.items():
            if values:
                final_fields[field] = values[-1]
        
        summary["final_fields"] = final_fields
        
        # Try to extract forces if applicable
        forces = self._extract_forces()
        if forces:
            summary["forces"] = forces
        
        return summary
    
    def _extract_forces(self) -> Dict[str, Any]:
        """
        Extract force data if available.
        
        Returns:
            Dict[str, Any]: Dictionary with force data, or empty dict if none found.
        """
        forces = {}
        
        if not self.case_dir:
            return forces
        
        # Check for forces directory
        forces_dir = os.path.join(self.case_dir, "postProcessing", "forces")
        if not os.path.exists(forces_dir):
            return forces
        
        # Get latest time directory
        time_dirs = sorted([d for d in os.listdir(forces_dir) if os.path.isdir(os.path.join(forces_dir, d))])
        if not time_dirs:
            return forces
        
        latest_time = time_dirs[-1]
        
        # Look for force files
        force_files = {
            "pressure": os.path.join(forces_dir, latest_time, "force_p.dat"),
            "viscous": os.path.join(forces_dir, latest_time, "force_v.dat"),
            "total": os.path.join(forces_dir, latest_time, "force.dat"),
            "pressure_moment": os.path.join(forces_dir, latest_time, "moment_p.dat"),
            "viscous_moment": os.path.join(forces_dir, latest_time, "moment_v.dat"),
            "total_moment": os.path.join(forces_dir, latest_time, "moment.dat")
        }
        
        # Parse each force file
        for force_type, file_path in force_files.items():
            if os.path.exists(file_path):
                try:
                    # Read the file
                    data = []
                    with open(file_path, 'r') as f:
                        for line in f:
                            if line.startswith('#'):
                                continue
                            
                            values = line.strip().split()
                            if len(values) >= 4:  # time, fx, fy, fz
                                data.append([float(v) for v in values[:4]])
                    
                    if data:
                        # Store time series data
                        forces[force_type] = data
                except Exception as e:
                    logger.warning(f"Error parsing force file {file_path}: {e}")
        
        return forces
    
    def save_results(self, output_dir: str = None) -> bool:
        """
        Save the simulation results to a directory.
        
        Args:
            output_dir (str, optional): Directory to save results to.
            
        Returns:
            bool: True if results were saved successfully, False otherwise.
        """
        if not self.case_dir:
            logger.error("No case directory. Cannot save results.")
            return False
        
        # Use case directory if no output directory specified
        if not output_dir:
            output_dir = os.path.join(self.case_dir, "results")
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save summary to JSON
            summary = self.get_results_summary()
            with open(os.path.join(output_dir, "summary.json"), 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save residuals to CSV
            if self.residuals:
                with open(os.path.join(output_dir, "residuals.csv"), 'w') as f:
                    # Write header
                    f.write("Iteration,Time," + ",".join(self.residuals.keys()) + "\n")
                    
                    # Write data rows
                    for i, time in enumerate(self.time_values):
                        row = [str(i+1), str(time)]
                        for field in self.residuals:
                            value = self.residuals[field][i] if i < len(self.residuals[field]) else ""
                            row.append(str(value))
                        f.write(",".join(row) + "\n")
            
            # Save solver output to log file
            with open(os.path.join(output_dir, "solver.log"), 'w') as f:
                f.write("\n".join(self.solver_output))
            
            logger.info(f"Simulation results saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False


# Create a singleton instance
_solver_manager = None

def get_solver_manager() -> SolverManager:
    """
    Get the SolverManager singleton instance.
    
    Returns:
        SolverManager: The SolverManager instance.
    """
    global _solver_manager
    if _solver_manager is None:
        _solver_manager = SolverManager()
    
    return _solver_manager