#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-processing utilities for OpenFOAM simulations in Openfoam_Simulator.

This module provides functions for processing OpenFOAM results after
a simulation completes, including converting to VTK format.
"""

import os
import logging
import threading
from typing import Optional, Dict, Any, Callable

# Import OpenFOAM utilities
from .openfoam_reader import (
    convert_openfoam_to_vtk, 
    is_simulation_running,
    read_openfoam_results_safely
)

logger = logging.getLogger(__name__)

class SimulationPostProcessor:
    """
    Handles post-processing tasks after an OpenFOAM simulation completes.
    
    This class coordinates tasks like converting OpenFOAM data to VTK format,
    generating summary reports, and preparing for visualization.
    """
    
    def __init__(self, case_dir: Optional[str] = None):
        """
        Initialize the post-processor.
        
        Args:
            case_dir (str, optional): OpenFOAM case directory
        """
        self.case_dir = case_dir
        self.processing_thread = None
        self.on_complete_callback = None
        
    def set_case_dir(self, case_dir: str):
        """
        Set the OpenFOAM case directory.
        
        Args:
            case_dir (str): Path to OpenFOAM case directory
        """
        self.case_dir = case_dir
        
    def process_completed_simulation(self, 
                                    on_complete: Optional[Callable[[bool], None]] = None, 
                                    background: bool = True):
        """
        Process a completed simulation by converting results to VTK and preparing for visualization.
        
        Args:
            on_complete (callable, optional): Callback function to call when processing is complete
            background (bool): Whether to run processing in a background thread
            
        Returns:
            bool: True if processing started successfully, False otherwise
        """
        if not self.case_dir or not os.path.exists(self.case_dir):
            logger.error(f"Invalid case directory: {self.case_dir}")
            return False
            
        # Store callback for later
        self.on_complete_callback = on_complete
        
        if background:
            # Run in background thread
            self.processing_thread = threading.Thread(
                target=self._do_process_simulation,
                daemon=True
            )
            self.processing_thread.start()
            return True
        else:
            # Run synchronously
            return self._do_process_simulation()
    
    def _do_process_simulation(self) -> bool:
        """
        Do the actual processing work.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Starting post-processing for case: {self.case_dir}")
        
        try:
            # First check if simulation is still running
            if is_simulation_running(self.case_dir):
                logger.warning("Simulation is still running, cannot post-process")
                if self.on_complete_callback:
                    self.on_complete_callback(False)
                return False
            
            # Step 1: Convert OpenFOAM results to VTK format
            logger.info("Converting OpenFOAM results to VTK format...")
            vtk_success = convert_openfoam_to_vtk(self.case_dir)
            
            if not vtk_success:
                logger.error("Failed to convert OpenFOAM results to VTK")
                if self.on_complete_callback:
                    self.on_complete_callback(False)
                return False
                
            logger.info("OpenFOAM results successfully converted to VTK format")
            
            # Step 2: Ensure boundary patches are also converted to VTK for streamline seeds
            # This is particularly important for inlet boundaries used in streamline visualization
            vtk_dir = os.path.join(self.case_dir, "VTK")
            if not os.path.exists(vtk_dir):
                os.makedirs(vtk_dir, exist_ok=True)
                
            # Check if we need to run a specific boundary extraction
            boundary_pattern = os.path.join(vtk_dir, "*boundary*.vtp")
            inlet_pattern = os.path.join(vtk_dir, "*inlet*.vtp")
            
            import glob
            boundary_files = glob.glob(boundary_pattern)
            inlet_files = glob.glob(inlet_pattern)
            
            if not (boundary_files or inlet_files):
                logger.info("No boundary or inlet VTK files found. Attempting to extract them...")
                try:
                    # Try to run foamToVTK with the -boundary option
                    # This creates separate VTK files for each boundary patch
                    import subprocess
                    import os
                    
                    # Change to case directory
                    original_dir = os.getcwd()
                    os.chdir(self.case_dir)
                    
                    # Run foamToVTK with boundary option
                    cmd = ["foamToVTK", "-boundary"]
                    logger.info(f"Running command: {' '.join(cmd)}")
                    
                    result = subprocess.run(
                        cmd, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        universal_newlines=True
                    )
                    
                    # Change back to original directory
                    os.chdir(original_dir)
                    
                    if result.returncode == 0:
                        logger.info("Successfully extracted boundary patches to VTK format")
                    else:
                        logger.warning(f"Boundary extraction may have failed: {result.stderr}")
                    
                except Exception as e:
                    logger.warning(f"Could not extract boundary patches: {e}")
            
            # Step 3: Generate any additional metadata or reports that might be useful
            # (can be extended in the future)
            
            # Step 4: Trigger callback with success status
            if self.on_complete_callback:
                self.on_complete_callback(True)
                
            return True
            
        except Exception as e:
            import traceback
            logger.error(f"Error during post-processing: {e}\n{traceback.format_exc()}")
            if self.on_complete_callback:
                self.on_complete_callback(False)
            return False

def post_process_simulation(case_dir: str, 
                           on_complete: Optional[Callable[[bool], None]] = None,
                           background: bool = True) -> bool:
    """
    Convenience function to post-process a simulation without creating an instance.
    
    Args:
        case_dir (str): Path to OpenFOAM case directory
        on_complete (callable, optional): Callback function to call when processing is complete
        background (bool): Whether to run processing in a background thread
        
    Returns:
        bool: True if processing started successfully, False otherwise
    """
    processor = SimulationPostProcessor(case_dir)
    return processor.process_completed_simulation(on_complete, background)
