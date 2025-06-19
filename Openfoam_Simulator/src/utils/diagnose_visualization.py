#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose visualization issues in Openfoam_Simulator.

This script analyzes OpenFOAM simulation results and VTK files to diagnose
visualization issues, particularly problems with streamline visualization.
"""

import os
import sys
import glob
import logging
import argparse
import subprocess
from pathlib import Path
import vtk

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('diagnose_visualization')

def find_vtk_files(directory):
    """Find all VTK format files in a directory."""
    vtk_files = []
    for ext in ['vtk', 'vtu', 'vtp']:
        vtk_files.extend(glob.glob(os.path.join(directory, f"*.{ext}")))
    return vtk_files

def analyze_vtk_file(vtk_file):
    """Analyze a VTK file for visualization suitability."""
    logger.info(f"Analyzing VTK file: {os.path.basename(vtk_file)}")
    
    try:
        # Create appropriate reader based on file extension
        if vtk_file.lower().endswith('.vtu'):
            reader = vtk.vtkXMLUnstructuredGridReader()
            file_type = "XML Unstructured Grid"
        elif vtk_file.lower().endswith('.vtp'):
            reader = vtk.vtkXMLPolyDataReader()
            file_type = "XML PolyData"
        else:  # .vtk
            reader = vtk.vtkGenericDataObjectReader()
            file_type = "Legacy VTK"
            
        reader.SetFileName(vtk_file)
        reader.Update()
        
        # Get basic file info
        output = reader.GetOutput()
        if not output:
            logger.warning("Reader produced null output")
            return False
            
        num_points = output.GetNumberOfPoints()
        num_cells = output.GetNumberOfCells()
        bounds = output.GetBounds()
        
        logger.info(f"File type: {file_type}")
        logger.info(f"Number of points: {num_points}")
        logger.info(f"Number of cells: {num_cells}")
        logger.info(f"Bounds: {bounds}")
        
        # Check if file has point data
        point_data = output.GetPointData()
        if not point_data:
            logger.warning("File has no point data")
            return False
            
        # Check available arrays
        num_arrays = point_data.GetNumberOfArrays()
        logger.info(f"Number of point data arrays: {num_arrays}")
        
        if num_arrays == 0:
            logger.warning("File has no point data arrays")
            return False
            
        velocity_field = None
        has_velocity = False
        
        logger.info("Point data arrays:")
        for i in range(num_arrays):
            array = point_data.GetArray(i)
            array_name = point_data.GetArrayName(i)
            num_components = array.GetNumberOfComponents()
            
            logger.info(f"  {i+1}. {array_name} ({num_components} components)")
            
            # Check for velocity field (vector with 3 components)
            if array_name in ['U', 'velocity', 'Velocity', 'v', 'vel'] and num_components == 3:
                has_velocity = True
                velocity_field = array_name
                
                # Sample velocity values
                num_tuples = array.GetNumberOfTuples()
                if num_tuples > 0:
                    sample_count = min(5, num_tuples)
                    sample_indices = [int(i * (num_tuples-1) / (sample_count-1)) for i in range(sample_count)]
                    
                    logger.info(f"  Velocity field samples ({velocity_field}):")
                    for idx in sample_indices:
                        v_tuple = [array.GetComponent(idx, i) for i in range(num_components)]
                        logger.info(f"    Point {idx}: {v_tuple}")
                        
                    # Check magnitude range
                    magnitude_sum = 0
                    magnitude_min = float('inf')
                    magnitude_max = 0
                    
                    for idx in range(min(100, num_tuples)):
                        vx = array.GetComponent(idx, 0)
                        vy = array.GetComponent(idx, 1)
                        vz = array.GetComponent(idx, 2)
                        magnitude = (vx*vx + vy*vy + vz*vz) ** 0.5
                        
                        magnitude_sum += magnitude
                        magnitude_min = min(magnitude_min, magnitude)
                        magnitude_max = max(magnitude_max, magnitude)
                    
                    avg_magnitude = magnitude_sum / min(100, num_tuples)
                    logger.info(f"  Velocity magnitude range: {magnitude_min:.6f} to {magnitude_max:.6f}")
                    logger.info(f"  Average velocity magnitude (first 100 points): {avg_magnitude:.6f}")
                    
                    if magnitude_max < 1e-6:
                        logger.warning("  ⚠️ Very low velocity magnitudes detected (near zero)")
                    
        if not has_velocity:
            logger.warning("⚠️ No velocity field found in the file")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error analyzing file: {e}")
        return False

def check_openfoam_case(case_dir):
    """Check OpenFOAM case for visualization suitability."""
    logger.info(f"Checking OpenFOAM case: {case_dir}")
    
    # Check if case directory exists
    if not os.path.isdir(case_dir):
        logger.error(f"Case directory does not exist: {case_dir}")
        return False
        
    # Check for time directories
    time_dirs = []
    for item in os.listdir(case_dir):
        path = os.path.join(case_dir, item)
        if os.path.isdir(path):
            try:
                float(item)  # Check if directory name is a number
                time_dirs.append(path)
            except ValueError:
                pass
                
    if not time_dirs:
        logger.error("No time directories found in the case")
        return False
        
    # Sort and get latest time directory
    time_dirs.sort(key=lambda x: float(os.path.basename(x)))
    latest_time = time_dirs[-1]
    logger.info(f"Latest time directory: {os.path.basename(latest_time)}")
    
    # Check for velocity field
    u_file = os.path.join(latest_time, "U")
    if not os.path.isfile(u_file):
        logger.error("No velocity field (U) found in the latest time directory")
        return False
        
    logger.info("Found velocity field file (U)")
    
    # Check for VTK directory
    vtk_dir = os.path.join(case_dir, "VTK")
    if not os.path.isdir(vtk_dir):
        logger.warning("VTK directory does not exist")
        logger.info("Try running foamToVTK to convert results to VTK format")
        return False
        
    # Check for VTK files
    vtk_files = find_vtk_files(vtk_dir)
    if not vtk_files:
        logger.warning("No VTK files found in the VTK directory")
        logger.info("Try running foamToVTK to convert results to VTK format")
        return False
        
    logger.info(f"Found {len(vtk_files)} VTK files")
    
    # Check if VTK files include boundary patches (important for inlet seeding)
    boundary_files = [f for f in vtk_files if 'boundary' in os.path.basename(f).lower()]
    if not boundary_files:
        logger.warning("No boundary patch VTK files found")
        logger.info("Try running 'foamToVTK -boundary' to extract boundary patches")
    else:
        logger.info(f"Found {len(boundary_files)} boundary patch VTK files")
        
    # Check if there's an inlet patch
    inlet_files = [f for f in vtk_files if 'inlet' in os.path.basename(f).lower()]
    if not inlet_files:
        logger.warning("No inlet patch VTK files found")
        logger.info("For better streamline seeding, ensure inlet patches are properly named")
    else:
        logger.info(f"Found {len(inlet_files)} inlet patch VTK files")
        
    return True

def analyze_case(case_dir):
    """Analyze an OpenFOAM case for visualization issues."""
    # First check the OpenFOAM case
    if not check_openfoam_case(case_dir):
        return False
        
    # Check VTK files
    vtk_dir = os.path.join(case_dir, "VTK")
    if os.path.isdir(vtk_dir):
        vtk_files = find_vtk_files(vtk_dir)
        
        # Analyze internal/volume VTK files first
        volume_files = [f for f in vtk_files if 'boundary' not in os.path.basename(f).lower() 
                                             and 'inlet' not in os.path.basename(f).lower() 
                                             and 'outlet' not in os.path.basename(f).lower()]
        
        if volume_files:
            # Sort by modification time (newest first)
            volume_files.sort(key=os.path.getmtime, reverse=True)
            
            # Analyze the first (newest) volume file
            logger.info("\n=== Analyzing volume VTK file ===")
            analyze_vtk_file(volume_files[0])
        
        # Also analyze inlet files if available
        inlet_files = [f for f in vtk_files if 'inlet' in os.path.basename(f).lower()]
        if inlet_files:
            logger.info("\n=== Analyzing inlet VTK file ===")
            analyze_vtk_file(inlet_files[0])
            
    return True

def convert_and_analyze(case_dir):
    """Convert OpenFOAM results to VTK and analyze them."""
    from convert_to_vtk import convert_openfoam_to_vtk
    
    # Run conversion
    logger.info("Converting OpenFOAM results to VTK format...")
    success = convert_openfoam_to_vtk(case_dir, include_boundary=True, force=True)
    
    if not success:
        logger.error("Failed to convert OpenFOAM results to VTK format")
        return False
        
    # Analyze the case
    logger.info("\n=== Analysis after conversion ===")
    return analyze_case(case_dir)

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Diagnose visualization issues in Openfoam_Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("case_dir", type=str, 
                        help="Path to OpenFOAM case directory")
    parser.add_argument("--convert", action="store_true",
                        help="Convert OpenFOAM results to VTK before analysis")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Expand path
    case_dir = os.path.abspath(os.path.expanduser(args.case_dir))
    
    # Run analysis
    if args.convert:
        success = convert_and_analyze(case_dir)
    else:
        success = analyze_case(case_dir)
    
    if success:
        logger.info("\n=== Visualization Recommendations ===")
        logger.info("1. Ensure VTK files contain non-zero velocity fields")
        logger.info("2. For streamlines, use the inlet as the seed location")
        logger.info("3. If streamlines don't appear, try increasing the line count and width")
        logger.info("4. If using the Openfoam_Simulator, ensure the visualization tab is properly configured")
        logger.info("5. For better visualization, try animating the flow")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
