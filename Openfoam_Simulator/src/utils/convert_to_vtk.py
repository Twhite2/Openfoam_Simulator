#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert OpenFOAM results to VTK format utility.

This script provides a command-line tool to convert OpenFOAM simulation results
to VTK format for visualization in Openfoam_Simulator or other VTK-based tools.
"""

import os
import sys
import glob
import logging
import argparse
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('convert_to_vtk')

def find_latest_time(case_dir):
    """Find the latest time directory in the OpenFOAM case."""
    time_dirs = []
    
    for item in os.listdir(case_dir):
        path = os.path.join(case_dir, item)
        if os.path.isdir(path):
            try:
                float(item)  # Check if directory name is a number
                time_dirs.append(item)
            except ValueError:
                pass
    
    if not time_dirs:
        return None
        
    # Sort numerically and get the latest
    time_dirs.sort(key=lambda x: float(x))
    return time_dirs[-1]

def check_velocity_field(time_dir):
    """Check if a time directory has a velocity field (U file)."""
    u_file = os.path.join(time_dir, "U")
    return os.path.isfile(u_file)

def convert_openfoam_to_vtk(case_dir, include_boundary=True, force=False):
    """
    Convert OpenFOAM case results to VTK format.
    
    Args:
        case_dir (str): Path to OpenFOAM case directory
        include_boundary (bool): Whether to include boundary patches
        force (bool): Whether to force conversion even if VTK files exist
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Converting OpenFOAM results to VTK format for: {case_dir}")
    
    if not os.path.isdir(case_dir):
        logger.error(f"Case directory {case_dir} does not exist")
        return False
        
    # Check if case has the required simulation data
    latest_time = find_latest_time(case_dir)
    if not latest_time:
        logger.error("No time directories found in the case")
        return False
        
    time_dir = os.path.join(case_dir, latest_time)
    if not check_velocity_field(time_dir):
        logger.error(f"No velocity field found in latest time directory: {time_dir}")
        return False
        
    # Create VTK directory if it doesn't exist
    vtk_dir = os.path.join(case_dir, "VTK")
    if not os.path.exists(vtk_dir):
        os.makedirs(vtk_dir, exist_ok=True)
        logger.info(f"Created VTK directory: {vtk_dir}")
    elif not force and glob.glob(os.path.join(vtk_dir, "*.vtk")):
        logger.info("VTK files already exist. Use --force to overwrite.")
        return True
        
    # Try to run foamToVTK
    try:
        # Change to case directory
        original_dir = os.getcwd()
        os.chdir(case_dir)
        
        # Build command
        cmd = ["foamToVTK"]
        if include_boundary:
            cmd.append("-boundary")
            
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Change back to original directory
        os.chdir(original_dir)
        
        if result.returncode != 0:
            logger.error(f"Error running foamToVTK: {result.stderr}")
            return False
            
        logger.info("OpenFOAM results successfully converted to VTK format")
        
        # Check the created VTK files
        vtk_files = glob.glob(os.path.join(vtk_dir, "*.vtk"))
        vtk_files.extend(glob.glob(os.path.join(vtk_dir, "*.vtu")))
        vtk_files.extend(glob.glob(os.path.join(vtk_dir, "*.vtp")))
        
        logger.info(f"Created {len(vtk_files)} VTK files in {vtk_dir}")
        for i, file in enumerate(vtk_files[:10]):  # Show first 10 files
            logger.info(f"  {i+1}. {os.path.basename(file)}")
            
        if len(vtk_files) > 10:
            logger.info(f"  ... and {len(vtk_files) - 10} more files")
            
        return True
        
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        return False

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Convert OpenFOAM results to VTK format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("case_dir", type=str, 
                        help="Path to OpenFOAM case directory")
    parser.add_argument("--no-boundary", action="store_false", dest="boundary",
                        help="Don't extract boundary patches")
    parser.add_argument("--force", action="store_true",
                        help="Force conversion even if VTK files exist")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Expand path
    case_dir = os.path.abspath(os.path.expanduser(args.case_dir))
    
    # Run conversion
    success = convert_openfoam_to_vtk(case_dir, args.boundary, args.force)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
