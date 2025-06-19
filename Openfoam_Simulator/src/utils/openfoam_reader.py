#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenFOAM reader utilities for Openfoam_Simulator.

This module provides functions for reading OpenFOAM case results
and converting them to VTK format for visualization.
"""

import os
import re
import glob
import subprocess
import logging
import tempfile
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

def find_latest_time(case_dir: str) -> Optional[str]:
    """
    Find the latest time directory in an OpenFOAM case.
    
    Args:
        case_dir (str): Path to the OpenFOAM case directory
        
    Returns:
        Optional[str]: Path to the latest time directory, or None if not found
    """
    if not os.path.isdir(case_dir):
        logger.error(f"Case directory {case_dir} does not exist")
        return None
        
    # Get all directories that can be converted to float (time directories)
    time_dirs = []
    for item in os.listdir(case_dir):
        path = os.path.join(case_dir, item)
        if os.path.isdir(path):
            try:
                # OpenFOAM uses directory names like "0", "0.5", "1", etc.
                float(item)
                time_dirs.append(item)
            except ValueError:
                # Not a time directory
                pass
                
    if not time_dirs:
        logger.warning(f"No time directories found in {case_dir}")
        return None
        
    # Sort by numeric value and get the latest
    time_dirs.sort(key=lambda x: float(x))
    latest = time_dirs[-1]
    
    logger.info(f"Found latest time directory: {latest}")
    return os.path.join(case_dir, latest)

def has_velocity_field(time_dir: str) -> bool:
    """
    Check if a time directory has a velocity field (U file).
    
    Args:
        time_dir (str): Path to an OpenFOAM time directory
        
    Returns:
        bool: True if velocity field exists
    """
    u_file = os.path.join(time_dir, "U")
    return os.path.isfile(u_file)

def convert_to_vtk(case_dir: str, output_dir: Optional[str] = None) -> Optional[str]:
    """
    Convert OpenFOAM case results to VTK format using foamToVTK.
    
    Args:
        case_dir (str): Path to the OpenFOAM case directory
        output_dir (str, optional): Directory to store VTK files, defaults to case_dir/VTK
        
    Returns:
        Optional[str]: Path to the VTK directory if successful, None otherwise
    """
    if not os.path.isdir(case_dir):
        logger.error(f"Case directory {case_dir} does not exist")
        return None
        
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.join(case_dir, "VTK")
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run foamToVTK utility
    try:
        logger.info(f"Converting OpenFOAM results to VTK format in {output_dir}")
        cmd = ["foamToVTK", "-case", case_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error converting to VTK: {result.stderr}")
            return None
            
        # Check if VTK files were created
        vtk_files = glob.glob(os.path.join(output_dir, "*.vtk"))
        if not vtk_files:
            logger.warning("No VTK files created")
            return None
            
        logger.info(f"Created {len(vtk_files)} VTK files in {output_dir}")
        return output_dir
        
    except Exception as e:
        logger.error(f"Error running foamToVTK: {e}")
        return None

def find_vtk_files(case_dir: str) -> List[str]:
    """
    Find existing VTK files for an OpenFOAM case.
    
    Args:
        case_dir (str): Path to the OpenFOAM case directory
        
    Returns:
        List[str]: List of paths to VTK files, sorted by time
    """
    vtk_dir = os.path.join(case_dir, "VTK")
    if not os.path.isdir(vtk_dir):
        logger.warning(f"VTK directory {vtk_dir} does not exist")
        return []
        
    # Find all VTK files
    vtk_files = glob.glob(os.path.join(vtk_dir, "*.vtk"))
    
    # Sort by time
    def extract_time(filename):
        # Extract time from filename, e.g. "case_100.vtk" -> 100
        match = re.search(r'_(\d+(\.\d+)?)\.vtk$', filename)
        if match:
            return float(match.group(1))
        return 0
        
    vtk_files.sort(key=extract_time)
    return vtk_files

def get_case_vtk_data(case_dir: str) -> Optional[str]:
    """
    Get VTK file for visualization from OpenFOAM case.
    This will check for existing VTK files and convert if needed.
    
    Args:
        case_dir (str): Path to the OpenFOAM case directory
        
    Returns:
        Optional[str]: Path to the latest VTK file, or None if not found
    """
    # First check if we already have VTK files
    vtk_files = find_vtk_files(case_dir)
    
    if vtk_files:
        # Return the latest VTK file
        logger.info(f"Found existing VTK file: {vtk_files[-1]}")
        return vtk_files[-1]
        
    # If no VTK files found, convert the case
    vtk_dir = convert_to_vtk(case_dir)
    
    if not vtk_dir:
        logger.error("Failed to convert OpenFOAM case to VTK")
        return None
        
    # Find VTK files after conversion
    vtk_files = find_vtk_files(case_dir)
    
    if not vtk_files:
        logger.error("No VTK files found after conversion")
        return None
        
    # Return the latest VTK file
    return vtk_files[-1]

def create_vtk_reader(vtk_file: str):
    """
    Create a VTK reader for the given VTK file.
    
    Args:
        vtk_file (str): Path to VTK file
        
    Returns:
        vtk.vtkDataReader: VTK reader for the file
    """
    try:
        import vtk
        
        # Determine reader type based on file extension
        ext = os.path.splitext(vtk_file)[1].lower()
        
        if ext == ".vtk":
            reader = vtk.vtkDataSetReader()
        elif ext == ".vtu":
            reader = vtk.vtkXMLUnstructuredGridReader()
        elif ext == ".vtp":
            reader = vtk.vtkXMLPolyDataReader()
        elif ext == ".vtr":
            reader = vtk.vtkXMLRectilinearGridReader()
        elif ext == ".vts":
            reader = vtk.vtkXMLStructuredGridReader()
        else:
            # Default to generic reader
            reader = vtk.vtkDataSetReader()
            
        reader.SetFileName(vtk_file)
        reader.Update()
        
        return reader
    except Exception as e:
        logger.error(f"Error creating VTK reader: {e}")
        return None

def load_case_for_visualization(case_dir: str):
    """
    Load OpenFOAM case data for visualization.
    Returns a VTK data reader that can be used for visualization.
    
    Args:
        case_dir (str): Path to the OpenFOAM case directory
        
    Returns:
        vtk.vtkAlgorithm: VTK reader for the case data
    """
    # First try using existing VTK files
    vtk_file = get_case_vtk_data(case_dir)
    
    if vtk_file:
        # Create VTK reader
        reader = create_vtk_reader(vtk_file)
        
        if reader:
            logger.info(f"Successfully loaded case data from VTK file: {vtk_file}")
            return reader
    
    # If no VTK files, try converting
    try:
        vtk_dir = convert_to_vtk(case_dir)
        
        if vtk_dir:
            # Find VTK files after conversion
            vtk_files = find_vtk_files(case_dir)
            
            if vtk_files:
                reader = create_vtk_reader(vtk_files[-1])
                if reader:
                    logger.info(f"Successfully loaded case data after conversion: {vtk_files[-1]}")
                    return reader
    except Exception as e:
        logger.warning(f"Error converting to VTK: {e}")
    
    # If conversion fails, try direct approach
    logger.info("Trying direct OpenFOAM data reading...")
    reader = load_openfoam_direct(case_dir)
    
    if reader:
        logger.info("Successfully loaded case data using direct OpenFOAM reading")
        return reader
    
    logger.error("Failed to get VTK data for case")
    return None

def read_openfoam_results(case_dir, velocity_field='U'):
    """
    Read OpenFOAM simulation results directly without converting to VTK.
    
    Args:
        case_dir (str): Path to OpenFOAM case directory
        velocity_field (str): Name of velocity field to look for
    
    Returns:
        tuple: (result_data, reader, source_id) or (None, None, None) if failed
    """
    import vtk
    logger.info(f"Reading OpenFOAM results from: {case_dir}")
    
    try:
        # Step 1: Find the latest time directory
        time_dirs = []
        for item in os.listdir(case_dir):
            item_path = os.path.join(case_dir, item)
            if os.path.isdir(item_path):
                try:
                    # Try to convert to float to check if it's a time directory
                    float(item)
                    time_dirs.append(item)
                except ValueError:
                    # Not a time directory, ignore
                    pass
        
        if not time_dirs:
            logger.warning(f"No time directories found in: {case_dir}")
            return None, None, None
        
        # Sort time directories by numeric value (latest last)
        time_dirs.sort(key=lambda x: float(x))
        latest_time = time_dirs[-1]
        time_dir = os.path.join(case_dir, latest_time)
        logger.info(f"Found latest time directory: {time_dir}")
        
        # Check for velocity field in this directory
        u_file = os.path.join(time_dir, velocity_field)
        
        if not os.path.exists(u_file):
            logger.warning(f"Velocity field file not found: {u_file}")
            return None, None, None
        
        logger.info(f"Found velocity field file: {u_file}")
        
        # Step 2: Try direct OpenFOAM reading with VTK's reader
        if hasattr(vtk, 'vtkOpenFOAMReader'):
            try:
                logger.info("Using vtkOpenFOAMReader to access results directly")
                foam_reader = vtk.vtkOpenFOAMReader()
                foam_reader.SetFileName(os.path.join(case_dir, "system", "controlDict"))
                foam_reader.SetCaseType(0)  # Decomposed case
                foam_reader.UpdateInformation()
                
                # Try to load the correct time step
                if foam_reader.GetTimeValues().GetNumberOfValues() > 0:
                    # Find the closest time to our latest_time
                    latest_time_value = float(latest_time)
                    closest_time_idx = 0
                    min_diff = float('inf')
                    
                    times = foam_reader.GetTimeValues()
                    for i in range(times.GetNumberOfValues()):
                        diff = abs(times.GetValue(i) - latest_time_value)
                        if diff < min_diff:
                            min_diff = diff
                            closest_time_idx = i
                    
                    foam_reader.SetTimeValue(times.GetValue(closest_time_idx))
                    logger.info(f"Reading time step {times.GetValue(closest_time_idx)}")
                
                foam_reader.SetReadZones(1)
                foam_reader.SetReadCellDataArrays(1)
                foam_reader.SetReadPointDataArrays(1)
                foam_reader.Update()
                
                # Generate a source ID
                source_id = f"foam_{os.path.basename(case_dir)}_{latest_time}"
                
                # Get output data and check for velocity field
                output = foam_reader.GetOutput()
                if output.GetPointData().HasArray(velocity_field):
                    logger.info(f"Successfully read velocity field from OpenFOAM case directly")
                    return output, foam_reader, source_id
                else:
                    logger.warning(f"vtkOpenFOAMReader succeeded but velocity field '{velocity_field}' not found")
            except Exception as e:
                import traceback
                logger.error(f"Error using vtkOpenFOAMReader: {e}\n{traceback.format_exc()}")
        else:
            logger.warning("vtkOpenFOAMReader not available - cannot read OpenFOAM case directly")
        
        # Step 3: Check for existing VTK files (but don't create them)
        vtk_dir = os.path.join(case_dir, "VTK")
        if os.path.exists(vtk_dir):
            vtk_pattern = os.path.join(vtk_dir, f"*_{latest_time}.vtk")
            vtk_files = glob.glob(vtk_pattern)
            
            if not vtk_files:
                # Try looking for any VTK files
                vtk_pattern = os.path.join(vtk_dir, "*.vtk")
                vtk_files = glob.glob(vtk_pattern)
                vtk_files.extend(glob.glob(os.path.join(vtk_dir, "*.vtu")))
            
            if vtk_files:
                # Sort by modification time (newest first)
                vtk_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                vtk_file = vtk_files[0]
                
                logger.info(f"Found existing VTK result file: {vtk_file}")
                
                # Create a reader for the VTK file
                if vtk_file.lower().endswith('.vtu'):
                    reader = vtk.vtkXMLUnstructuredGridReader()
                else:
                    reader = vtk.vtkGenericDataObjectReader()
                
                reader.SetFileName(vtk_file)
                reader.Update()
                
                # Check if the VTK file has the velocity field
                output = reader.GetOutput()
                if output.GetPointData().HasArray(velocity_field):
                    source_id = os.path.basename(vtk_file)
                    logger.info(f"Successfully loaded velocity field from existing VTK file: {vtk_file}")
                    return output, reader, source_id
                else:
                    logger.warning(f"VTK file does not contain velocity field: {vtk_file}")
        
        # If we got here, we couldn't read the results
        logger.warning("Could not read OpenFOAM results directly or from existing VTK files")
        return None, None, None
        
    except Exception as e:
        import traceback
        logger.error(f"Error reading OpenFOAM results: {e}\n{traceback.format_exc()}")
        return None, None, None

def convert_openfoam_to_vtk(case_dir):
    """
    Convert OpenFOAM case to VTK format for visualization.
    
    Args:
        case_dir (str): Path to OpenFOAM case directory
        
    Returns:
        bool: True if successful, False otherwise
    """
    
    import os
    import subprocess
    import logging
    import shutil
    import glob
    
    logger = logging.getLogger(__name__)
    
    try:
        # Validate case directory
        if not os.path.exists(case_dir) or not os.path.isdir(case_dir):
            logger.error(f"Invalid case directory: {case_dir}")
            return False
            
        # Check for time directories
        time_dirs = [d for d in os.listdir(case_dir) if os.path.isdir(os.path.join(case_dir, d)) and d.replace('.', '', 1).isdigit()]
        if not time_dirs:
            logger.error(f"No time directories found in case: {case_dir}")
            return False
            
        # Log the time directories found
        logger.info(f"Found time directories: {time_dirs}")
        
        # Create VTK directory if it doesn't exist
        vtk_dir = os.path.join(case_dir, "VTK")
        if not os.path.exists(vtk_dir):
            os.makedirs(vtk_dir, exist_ok=True)
            
        # Store current directory
        original_dir = os.getcwd()
        
        try:
            # Change to case directory for foamToVTK
            os.chdir(case_dir)
            
            # First, try using foamToVTK with enhanced boundary patch preservation
            logger.info("Running foamToVTK with enhanced boundary patch options...")
            
            # Extract boundary patch information from polyMesh/boundary file
            boundary_file = os.path.join(case_dir, "constant", "polyMesh", "boundary")
            patches = []
            if os.path.exists(boundary_file):
                logger.info(f"Found boundary file at {boundary_file}")
                try:
                    with open(boundary_file, 'r') as f:
                        content = f.read()
                        # Basic parsing of OpenFOAM dictionary format
                        import re
                        # Find the number of patches
                        match = re.search(r'\s*(\d+)\s*\(', content)
                        if match:
                            patch_count = int(match.group(1))
                            logger.info(f"Found {patch_count} boundary patches")
                            
                            # Parse patch names - capture patch names between ( and {  
                            patch_matches = re.finditer(r'\s*(\w+)\s*\{[^\{\}]*type\s+(\w+)', content)
                            for match in patch_matches:
                                patch_name = match.group(1)
                                patch_type = match.group(2)
                                patches.append((patch_name, patch_type))
                                logger.info(f"Found patch: {patch_name} of type {patch_type}")
                except Exception as e:
                    logger.error(f"Error parsing boundary file: {e}")
            
            # Save patch info for streamline seeding even if conversion fails
            patch_info_dir = os.path.join(case_dir, "VTK", "boundary_info")
            os.makedirs(patch_info_dir, exist_ok=True)
            
            with open(os.path.join(patch_info_dir, "patches.txt"), 'w') as f:
                for patch_name, patch_type in patches:
                    f.write(f"{patch_name}:{patch_type}\n")
            
            # Extra patch conversion options for comprehensive boundary export
            result = subprocess.run(
                ["foamToVTK", "-boundary", "-patches", "all", "-excludePatches", "()"], 
                cwd=case_dir, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                timeout=120  # 2-minute timeout
            )
            
            if result.returncode != 0:
                logger.warning(f"foamToVTK with enhanced boundary options failed, trying standard boundary flag...")
                logger.warning(f"Error output: {result.stderr.decode('utf-8', errors='replace')}")
                
                # Try with just the boundary flag
                result = subprocess.run(
                    ["foamToVTK", "-boundary"], 
                    cwd=case_dir, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    timeout=120
                )
                
                if result.returncode != 0:
                    logger.warning(f"foamToVTK with standard boundary flag failed, trying without boundary flag...")
                    logger.warning(f"Error output: {result.stderr.decode('utf-8', errors='replace')}")
                    
                    # Try without boundary flag as last resort
                    result = subprocess.run(
                        ["foamToVTK"], 
                        cwd=case_dir, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        timeout=120
                    )
                
                if result.returncode != 0:
                    # Log the error
                    logger.error(f"Error running foamToVTK: {result.stderr.decode('utf-8', errors='replace')}")
                    
                    # Try with absolute path
                    logger.info("Trying with absolute path to foamToVTK...")
                    
                    # Try to find foamToVTK in common locations
                    foam_to_vtk_paths = [
                        "/usr/bin/foamToVTK",
                        "/usr/local/bin/foamToVTK", 
                        "/opt/openfoam*/bin/foamToVTK"
                    ]
                    
                    foam_to_vtk_path = None
                    for path in foam_to_vtk_paths:
                        # Handle wildcards
                        if '*' in path:
                            import glob
                            matches = glob.glob(path)
                            if matches:
                                foam_to_vtk_path = matches[0]
                                break
                        elif os.path.exists(path):
                            foam_to_vtk_path = path
                            break
                    
                    if foam_to_vtk_path:
                        logger.info(f"Found foamToVTK at: {foam_to_vtk_path}")
                        result = subprocess.run(
                            [foam_to_vtk_path], 
                            cwd=case_dir, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE,
                            timeout=120
                        )
                        
                        if result.returncode != 0:
                            logger.error(f"Error running foamToVTK with absolute path: {result.stderr.decode('utf-8', errors='replace')}")
                            return False
                    else:
                        logger.error("Could not find foamToVTK executable")
                        return False
        finally:
            # Restore original directory
            os.chdir(original_dir)
        
        # Check if VTK files were created
        if os.path.exists(vtk_dir):
            # Look for any VTK files (multiple extensions)
            vtk_files = []
            boundary_files = []
            patch_files = {}
            for ext in ['vtk', 'vtu', 'vtp']:
                vtk_files.extend(glob.glob(os.path.join(vtk_dir, f"*.{ext}")))
                vtk_files.extend(glob.glob(os.path.join(case_dir, f"VTK/*.{ext}")))
                # Look specifically for boundary and patch files
                boundary_files.extend(glob.glob(os.path.join(vtk_dir, f"*_boundary.{ext}")))
                for patch_name, _ in patches:
                    matching_files = glob.glob(os.path.join(vtk_dir, f"*{patch_name}.{ext}"))
                    if matching_files:
                        if patch_name not in patch_files:
                            patch_files[patch_name] = []
                        patch_files[patch_name].extend(matching_files)
            
            if vtk_files:
                logger.info(f"Found {len(vtk_files)} VTK files")
                # List first few
                for i, file in enumerate(vtk_files[:5]):
                    logger.info(f"  {i+1}. {os.path.basename(file)}")
                if len(vtk_files) > 5:
                    logger.info(f"  ... and {len(vtk_files) - 5} more files")
                
                # Report on boundary files specifically
                if boundary_files:
                    logger.info(f"Found {len(boundary_files)} boundary VTK files:")
                    for file in boundary_files[:3]:
                        logger.info(f"  - {os.path.basename(file)}")
                    if len(boundary_files) > 3:
                        logger.info(f"  ... and {len(boundary_files) - 3} more boundary files")
                else:
                    logger.warning("No boundary VTK files found - streamline seeding from boundaries may be limited")
                
                # Report on patch-specific files
                if patch_files:
                    logger.info(f"Found patch-specific VTK files for {len(patch_files)} patches:")
                    for patch, files in list(patch_files.items())[:3]:
                        logger.info(f"  - {patch}: {len(files)} files")
                    if len(patch_files) > 3:
                        logger.info(f"  ... and {len(patch_files) - 3} more patches")
                else:
                    logger.warning("No patch-specific VTK files found - this may affect streamline seeding")
                
                # Create a boundary hint file for the visualization system to use for streamline seeding
                hint_file = os.path.join(vtk_dir, "boundary_hints.txt")
                with open(hint_file, 'w') as f:
                    f.write(f"# Boundary information for VTK visualization\n")
                    f.write(f"VTK_BOUNDARY_FILES: {len(boundary_files)}\n")
                    f.write(f"TOTAL_PATCHES: {len(patches)}\n")
                    f.write(f"\n# Patch details\n")
                    for patch_name, patch_type in patches:
                        patch_files = patch_files.get(patch_name, [])
                        f.write(f"PATCH: {patch_name} TYPE: {patch_type} FILES: {len(patch_files)}\n")
                
                logger.info(f"Created boundary hint file at {hint_file}")
                return True
            else:
                logger.warning("No VTK files found after conversion")
                return False
        else:
            logger.error(f"VTK directory not found after conversion: {vtk_dir}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("foamToVTK timed out after 120 seconds")
        return False
    except Exception as e:
        import traceback
        logger.error(f"Error converting OpenFOAM results to VTK: {e}\n{traceback.format_exc()}")
        return False

def is_simulation_running(case_dir):
    """
    Check if an OpenFOAM simulation is currently running.
    
    Args:
        case_dir (str): Path to OpenFOAM case directory
        
    Returns:
        bool: True if simulation is running, False otherwise
    """
    import os
    import glob
    import subprocess
    
    try:
        # Method 1: Check for lock files
        lock_files = glob.glob(os.path.join(case_dir, "*.lock"))
        if lock_files:
            logger.info(f"Found lock files in case directory: {lock_files}")
            return True
            
        # Method 2: Check for write permissions on time directories
        time_dirs = []
        for item in os.listdir(case_dir):
            try:
                float_value = float(item)
                time_path = os.path.join(case_dir, item)
                if os.path.isdir(time_path):
                    time_dirs.append(time_path)
            except ValueError:
                continue
                
        if time_dirs:
            latest_time_dir = max(time_dirs, key=lambda x: float(os.path.basename(x)))
            test_file = os.path.join(latest_time_dir, ".simulation_check")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except (PermissionError, OSError):
                logger.info(f"Cannot write to latest time directory: {latest_time_dir}")
                return True
                
        # Method 3: Check for OpenFOAM processes
        try:
            # Look for simpleFoam processes
            ps_result = subprocess.run(
                ["ps", "-aux"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
            
            output = ps_result.stdout.lower()
            openfoam_processes = ["simpleFoam", "pisoFoam", "pimpleFoam", "interFoam", "foamRun"]
            
            for process in openfoam_processes:
                if process.lower() in output and case_dir.lower() in output:
                    logger.info(f"Found active OpenFOAM process for case: {case_dir}")
                    return True
        except Exception as e:
            logger.warning(f"Error checking for OpenFOAM processes: {e}")
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking if simulation is running: {e}")
        # If we can't determine, assume it's not running
        return False

def read_openfoam_results_safely(case_dir, velocity_field='U'):
    """
    Safely read OpenFOAM results, checking first if a simulation is running.
    
    Args:
        case_dir (str): Path to OpenFOAM case directory
        velocity_field (str): Name of velocity field to look for
        
    Returns:
        tuple: (result_data, reader, source_id) or (None, None, None) if failed
    """
    # First check if simulation is running
    if is_simulation_running(case_dir):
        logger.warning("Cannot read OpenFOAM results: Active simulation detected")
        return None, None, None
        
    # If not running, proceed with reading results
    return read_openfoam_results(case_dir, velocity_field)

def convert_openfoam_to_vtk_safely(case_dir):
    """
    Safely convert OpenFOAM results to VTK format, checking first if a simulation is running.
    
    Args:
        case_dir (str): Path to OpenFOAM case directory
        
    Returns:
        bool: True if successful, False otherwise
    """
    # First check if simulation is running
    if is_simulation_running(case_dir):
        logger.warning("Cannot convert OpenFOAM results: Active simulation detected")
        return False
        
    # If not running, proceed with conversion
    return convert_openfoam_to_vtk(case_dir)

def parse_openfoam_vector_file(file_path):
    """
    Parse an OpenFOAM vector field file (like U) directly.
    
    Args:
        file_path (str): Path to the OpenFOAM field file
        
    Returns:
        Tuple[List[float], List[List[float]]]: (points, vectors) or None if parsing fails
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Simplified parser for OpenFOAM vector files
        # This handles basic OpenFOAM format, but may need to be extended
        # for more complex cases
        
        # Extract internalField
        internal_match = re.search(r'internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;', 
                                   content, re.DOTALL)
        if not internal_match:
            logger.warning("Could not find internalField in vector file")
            return None
            
        # Parse vector values
        vector_count = int(internal_match.group(1))
        vector_data_str = internal_match.group(2)
        
        # Clean up the string - remove newlines and extra whitespace
        vector_data_str = re.sub(r'\s+', ' ', vector_data_str).strip()
        
        # Split into vectors
        vector_items = vector_data_str.split(') (')
        vector_items[0] = vector_items[0].lstrip('(')
        vector_items[-1] = vector_items[-1].rstrip(')')
        
        # Parse vectors from strings like "1 0 0"
        vectors = []
        for item in vector_items:
            components = item.split()
            if len(components) == 3:
                try:
                    vec = [float(components[0]), float(components[1]), float(components[2])]
                    vectors.append(vec)
                except ValueError:
                    logger.warning(f"Could not parse vector: {item}")
        
        # For a proper implementation, we'd also need to get the points
        # from the mesh file, but for this simplified version we'll
        # return the vectors and handle the points separately
        
        logger.info(f"Parsed {len(vectors)} vectors from {file_path}")
        return vectors
    except Exception as e:
        logger.error(f"Error parsing OpenFOAM vector file: {e}")
        return None

def parse_openfoam_points_file(file_path):
    """
    Parse an OpenFOAM points file to get mesh geometry.
    
    Args:
        file_path (str): Path to the OpenFOAM points file
        
    Returns:
        List of 3D points coordinates or None if parsing fails
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Extract the points list
        points_match = re.search(r'(\d+)\s*\((.*?)\)\s*;', content, re.DOTALL)
        if not points_match:
            logger.warning("Could not find points list in points file")
            return None
            
        # Parse point count and data
        point_count = int(points_match.group(1))
        points_data_str = points_match.group(2)
        
        # Clean up the string - remove newlines and extra whitespace
        points_data_str = re.sub(r'\s+', ' ', points_data_str).strip()
        
        # Split into points
        point_items = points_data_str.split(') (')
        point_items[0] = point_items[0].lstrip('(')
        point_items[-1] = point_items[-1].rstrip(')')
        
        # Parse points
        points = []
        for item in point_items:
            components = item.split()
            if len(components) == 3:
                try:
                    point = [float(components[0]), float(components[1]), float(components[2])]
                    points.append(point)
                except ValueError:
                    logger.warning(f"Could not parse point: {item}")
        
        logger.info(f"Parsed {len(points)} points from {file_path}")
        return points
    except Exception as e:
        logger.error(f"Error parsing OpenFOAM points file: {e}")
        return None

def parse_openfoam_cells_file(file_path):
    """
    Parse an OpenFOAM cells file to get cell connectivity.
    This is a simplified version that works with basic cell types.
    
    Args:
        file_path (str): Path to the OpenFOAM cells file
        
    Returns:
        List of cell definitions or None if parsing fails
    """
    try:
        # For a complete implementation, we'd parse the cells file
        # to get the connectivity information. This is complex and
        # beyond the scope of this implementation.
        # 
        # We'll return a placeholder for now
        return None
    except Exception as e:
        logger.error(f"Error parsing OpenFOAM cells file: {e}")
        return None

def create_vtkpolydata_from_raw(case_dir, time_dir=None):
    """
    Create a vtkPolyData object from raw OpenFOAM data.
    
    Args:
        case_dir (str): Path to OpenFOAM case directory
        time_dir (str, optional): Specific time directory to use
        
    Returns:
        vtk.vtkPolyData or None if creation fails
    """
    try:
        import vtk
        import numpy as np
        
        # Find latest time directory if not specified
        if time_dir is None:
            time_dir = find_latest_time(case_dir)
            if not time_dir:
                logger.warning("No time directories found")
                return None
                
        logger.info(f"Found latest time directory: {time_dir}")
                
        # Check if velocity file exists
        u_file = os.path.join(time_dir, "U")
        if not os.path.exists(u_file):
            logger.warning(f"Velocity file not found: {u_file}")
            return None
            
        # Parse velocity data
        vectors = parse_openfoam_vector_file(u_file)
        if not vectors or len(vectors) == 0:
            logger.warning("Failed to parse velocity data")
            return None
            
        # Try to get mesh points
        mesh_points = None
        points_file = os.path.join(case_dir, "constant", "polyMesh", "points")
        if os.path.exists(points_file):
            mesh_points = parse_openfoam_points_file(points_file)
        
        # For a simplified visualization, create a pipe-like structure if mesh points not available
        points = vtk.vtkPoints()
        velocity_data = vtk.vtkDoubleArray()
        velocity_data.SetName("U")  # OpenFOAM velocity field name
        velocity_data.SetNumberOfComponents(3)
        
        if mesh_points and len(mesh_points) == len(vectors):
            # Use actual mesh points if available and they match the vector count
            logger.info(f"Using actual mesh geometry with {len(mesh_points)} points")
            for i, point in enumerate(mesh_points):
                if i < len(vectors):
                    points.InsertNextPoint(point[0], point[1], point[2])
                    velocity_data.InsertNextTuple3(
                        vectors[i][0], vectors[i][1], vectors[i][2]
                    )
        else:
            # Create a pipe-like structure along the x-axis
            logger.info("Creating pipe-like structure for visualization")
            num_points = len(vectors)
            
            # Determine if we have enough points for a reasonable grid
            # For a pipe, we'll create slices along the x-axis
            grid_size = int(np.sqrt(num_points / 20))  # Assuming length is about 20x radius
            grid_size = max(grid_size, 3)  # At least 3 points per slice
            pipe_length = 1.0  # normalized pipe length
            
            # Number of slices along the pipe length
            num_slices = num_points // (grid_size * grid_size)
            num_slices = max(num_slices, 2)  # At least 2 slices
            
            # Create pipe structure
            radius = 0.25  # Pipe radius
            
            # Create points along the pipe
            point_id = 0
            dx = pipe_length / (num_slices - 1)
            
            for slice_idx in range(num_slices):
                x = slice_idx * dx - pipe_length / 2  # Center pipe at origin
                
                # Create circle of points for this slice
                for i in range(grid_size):
                    for j in range(grid_size):
                        # Create points in a circular pattern
                        theta = 2 * np.pi * i / grid_size
                        r = radius * j / (grid_size - 1)
                        
                        y = r * np.cos(theta)
                        z = r * np.sin(theta)
                        
                        if point_id < num_points:
                            points.InsertNextPoint(x, y, z)
                            velocity_data.InsertNextTuple3(
                                vectors[point_id][0],
                                vectors[point_id][1], 
                                vectors[point_id][2]
                            )
                            point_id += 1
        
        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        # Create cells (vertices)
        vertices = vtk.vtkCellArray()
        for i in range(points.GetNumberOfPoints()):
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(i)
        
        polydata.SetVerts(vertices)
        
        # Add velocity data to points
        polydata.GetPointData().AddArray(velocity_data)
        polydata.GetPointData().SetActiveVectors("U")
        
        logger.info(f"Created vtkPolyData with {points.GetNumberOfPoints()} points and velocity data")
        return polydata
    except Exception as e:
        logger.error(f"Error creating vtkPolyData from raw data: {e}")
        return None

def direct_openfoam_to_vtk(case_dir):
    """
    Convert OpenFOAM data to VTK format directly in Python without using foamToVTK.
    
    Args:
        case_dir (str): Path to OpenFOAM case directory
        
    Returns:
        vtk.vtkAlgorithm: VTK reader for the data
    """
    try:
        import vtk
        
        # Create polydata from raw OpenFOAM data
        polydata = create_vtkpolydata_from_raw(case_dir)
        
        if not polydata:
            logger.warning("Failed to create polydata from raw OpenFOAM data")
            return None
            
        # Create a point source for visualization
        source = vtk.vtkPolyDataReader()
        source.ReadFromInputStringOn()
        
        # Write polydata to memory
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(polydata)
        writer.WriteToOutputStringOn()
        writer.Update()
        
        # Read back from memory
        source.SetInputString(writer.GetOutputString())
        source.Update()
        
        logger.info("Successfully created VTK source from raw OpenFOAM data")
        return source
    except Exception as e:
        logger.error(f"Error in direct_openfoam_to_vtk: {e}")
        return None

def load_openfoam_direct(case_dir):
    """
    Load OpenFOAM data directly without foamToVTK utility.
    This is a fallback when foamToVTK is not available.
    
    Args:
        case_dir (str): Path to OpenFOAM case directory
        
    Returns:
        vtk.vtkAlgorithm: VTK source for visualization
    """
    return direct_openfoam_to_vtk(case_dir)

def read_openfoam_mesh(case_dir):
    """
    Read the OpenFOAM mesh directly.
    
    Args:
        case_dir (str): Path to the OpenFOAM case directory
        
    Returns:
        vtk.vtkUnstructuredGrid or None if reading fails
    """
    try:
        import vtk
        
        # Path to mesh files
        constant_polyMesh = os.path.join(case_dir, "constant", "polyMesh")
        points_file = os.path.join(constant_polyMesh, "points")
        faces_file = os.path.join(constant_polyMesh, "faces")
        owner_file = os.path.join(constant_polyMesh, "owner")
        neighbour_file = os.path.join(constant_polyMesh, "neighbour")
        
        # Check if all required files exist
        if not all(os.path.exists(f) for f in [points_file, faces_file, owner_file, neighbour_file]):
            logger.warning(f"Missing mesh files in {constant_polyMesh}")
            return None
            
        # Since reading OpenFOAM mesh files directly is complex, 
        # we'll use foamToVTK for now, but in a more complete implementation
        # we would parse these files directly
        
        return None
    except Exception as e:
        logger.error(f"Error reading OpenFOAM mesh: {e}")
        return None
