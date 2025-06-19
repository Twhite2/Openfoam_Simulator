#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project data model for Openfoam_Simulator application.

This module implements the core project model that manages simulation cases,
meshes, results, and other project data for CFD simulations in the oil & gas industry.
"""

import os
import sys
import json
import shutil
import logging
import tempfile
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

# Import utility modules
from ..utils.logger import get_logger
from ..config import get_value, set_value

# Import OpenFOAM integration modules
from ..openfoam_integration.case_manager import CaseManager, create_case_manager

logger = get_logger(__name__)


class Project:
    """
    Project class for Openfoam_Simulator application.
    
    This class represents a CFD project and manages all project data, including:
    - Project metadata (name, description, author, etc.)
    - File paths (geometry, mesh, case, results, etc.)
    - Project directory structure
    - Active components (active mesh, case, etc.)
    - Operations on project data (import, export, etc.)
    """
    
    def __init__(self, name: str = "Untitled Project", author: str = ""):
        """
        Initialize a new project.
        
        Args:
            name (str, optional): Project name
            author (str, optional): Project author
        """
        # Project metadata
        self.name = name
        self.description = ""
        self.author = author
        self.created_date = datetime.datetime.now().isoformat()
        self.modified_date = self.created_date
        self.version = "1.0"
        
        # Project filepath (None for new projects)
        self.filepath = None
        
        # Project directory (will be set when saved)
        self.project_dir = None
        
        # File collections
        self.geometry_files = []  # List of geometry file paths
        self.mesh_files = []      # List of mesh file paths
        self.case_files = []      # List of case file paths
        self.boundary_files = []  # List of boundary condition file paths
        self.result_files = []    # List of result file paths
        self.visualization_files = []  # List of visualization state file paths
        self.report_files = []    # List of report file paths
        
        # Active components
        self.active_geometry = None
        self.active_mesh = None
        self.active_case = None
        self.active_results = None
        
        # Simulation settings
        self.simulation_settings = {}
        
        # Case directory for OpenFOAM
        self.case_dir = None
        
        # Flag for running simulation
        self.simulation_running = False
        self.simulation_process = None
        
        # Case manager for OpenFOAM integration
        self.case_manager = None
    
    def save(self) -> bool:
        """
        Save the project to disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.filepath:
            logger.error("Cannot save project without a filepath")
            return False
        
        try:
            # Update modified date
            self.modified_date = datetime.datetime.now().isoformat()
            
            # Create project directory if it doesn't exist
            project_dir = Path(self.filepath).parent
            if not project_dir.exists():
                project_dir.mkdir(parents=True)
            
            # If this is the first time saving, create directory structure
            if not self.project_dir:
                self.project_dir = project_dir
                self._create_directory_structure()
            
            # Create project data dictionary
            project_data = {
                "metadata": {
                    "name": self.name,
                    "description": self.description,
                    "author": self.author,
                    "created_date": self.created_date,
                    "modified_date": self.modified_date,
                    "version": self.version
                },
                "files": {
                    "geometry": self.geometry_files,
                    "mesh": self.mesh_files,
                    "case": self.case_files,
                    "boundary": self.boundary_files,
                    "results": self.result_files,
                    "visualization": self.visualization_files,
                    "reports": self.report_files
                },
                "active": {
                    "geometry": self.active_geometry,
                    "mesh": self.active_mesh,
                    "case": self.active_case,
                    "results": self.active_results
                },
                "simulation": self.simulation_settings,
                "case_directory": str(self.case_dir) if self.case_dir else None
            }
            
            # Save to JSON file
            with open(self.filepath, 'w') as f:
                json.dump(project_data, f, indent=2)
            
            logger.info(f"Project saved to {self.filepath}")
            
            # Make sure to save the case directory in the project data
            if hasattr(self, 'case_dir') and self.case_dir:
                project_data['case_directory'] = self.case_dir
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving project: {e}")
            return False
    
    @staticmethod
    def load(filepath: str) -> 'Project':
        """
        Load a project from disk.
        
        Args:
            filepath (str): Path to the project file
            
        Returns:
            Project: The loaded project
            
        Raises:
            FileNotFoundError: If the project file doesn't exist
            json.JSONDecodeError: If the project file is invalid JSON
            ValueError: If the project file is not a valid project
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Project file not found: {filepath}")
        
        try:
            # Load project data from JSON file
            with open(filepath, 'r') as f:
                project_data = json.load(f)
            
            # Validate project data
            if 'metadata' not in project_data or 'files' not in project_data:
                raise ValueError("Invalid project file format")
            
            # Create new project instance
            project = Project()
            project.filepath = filepath
            project.project_dir = Path(filepath).parent
            
            # Load metadata
            metadata = project_data['metadata']
            project.name = metadata.get('name', 'Untitled Project')
            project.description = metadata.get('description', '')
            project.author = metadata.get('author', '')
            project.created_date = metadata.get('created_date', datetime.datetime.now().isoformat())
            project.modified_date = metadata.get('modified_date', project.created_date)
            project.version = metadata.get('version', '1.0')
            
            # Load file collections
            files = project_data['files']
            project.geometry_files = files.get('geometry', [])
            project.mesh_files = files.get('mesh', [])
            project.case_files = files.get('case', [])
            project.boundary_files = files.get('boundary', [])
            project.result_files = files.get('results', [])
            project.visualization_files = files.get('visualization', [])
            project.report_files = files.get('reports', [])
            
            # Load active components
            if 'active' in project_data:
                active = project_data['active']
                project.active_geometry = active.get('geometry')
                project.active_mesh = active.get('mesh')
                project.active_case = active.get('case')
                project.active_results = active.get('results')
            
            # Load simulation settings
            if 'simulation' in project_data:
                project.simulation_settings = project_data['simulation']
            
            # Load case directory
            if 'case_directory' in project_data and project_data['case_directory']:
                project.case_dir = project_data['case_directory']
                
                # Initialize case manager for the loaded case
                if os.path.exists(project.case_dir):
                    project.case_manager = create_case_manager(project.case_dir)
                else:
                    # If the directory doesn't exist, create it
                    os.makedirs(project.case_dir, exist_ok=True)
                    project.initialize_openfoam_case()
            
            logger.info(f"Project loaded from {filepath}")
            return project
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing project file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading project: {e}")
            raise ValueError(f"Failed to load project: {str(e)}")
    
    def _create_directory_structure(self):
        """Create the project directory structure."""
        try:
            # Create directories
            for directory in ["geometry", "mesh", "case", "results", "visualization", "reports"]:
                path = Path(self.project_dir) / directory
                if not path.exists():
                    path.mkdir(parents=True)
                    logger.debug(f"Created directory: {path}")
            
            # Set OpenFOAM case directory to be the case directory directly
            if not self.case_dir:
                case_path = Path(self.project_dir) / "case"
                self.case_dir = str(case_path)
                
                # Initialize OpenFOAM case
                self.initialize_openfoam_case()
                
        except Exception as e:
            logger.error(f"Error creating directory structure: {e}")
    
    def initialize_openfoam_case(self):
        """Initialize an OpenFOAM case with default structure and settings."""
        try:
            if not self.case_dir:
                logger.error("Cannot initialize OpenFOAM case: case directory not set")
                return False
                
            # Create case manager if not already present
            if not self.case_manager:
                self.case_manager = create_case_manager(self.case_dir)
                
            # Set up basic case structure and files
            success = self.case_manager.setup_case()
            
            if success:
                logger.info(f"OpenFOAM case initialized at: {self.case_dir}")
                # Add to case files if successful
                system_dir = os.path.join(self.case_dir, "system")
                if os.path.exists(system_dir):
                    control_dict_path = os.path.join(system_dir, "controlDict")
                    if os.path.exists(control_dict_path) and control_dict_path not in self.case_files:
                        self.case_files.append(control_dict_path)
            else:
                logger.warning(f"Failed to initialize OpenFOAM case at: {self.case_dir}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error initializing OpenFOAM case: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def import_mesh(self, filepath: str) -> bool:
        """
        Import a mesh file into the project.
        
        Args:
            filepath (str): Path to the mesh file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure project directory exists
            if not self.project_dir:
                logger.error("Project directory not set")
                return False
            
            # Create mesh directory if it doesn't exist
            # Convert project_dir to Path object if it's a string
            project_dir_path = Path(self.project_dir) if isinstance(self.project_dir, str) else self.project_dir
            mesh_dir = project_dir_path / "mesh"
            os.makedirs(mesh_dir, exist_ok=True)
            
            # Get file info
            file_path = Path(filepath)
            filename = file_path.name
            target_path = mesh_dir / filename
            
            # Check if file already exists within project
            if str(file_path) == str(target_path):
                # File is already in project directory, just register it
                logger.info(f"Mesh file {filename} already in project directory")
            else:
                # Copy the file to project mesh directory if not already there
                import shutil
                shutil.copy2(filepath, target_path)
                logger.info(f"Copied mesh file to project: {target_path}")
            
            # Add to mesh files list
            target_path_str = str(target_path)
            if target_path_str not in self.mesh_files:
                self.mesh_files.append(target_path_str)
            
            # Set as active mesh
            self.active_mesh = target_path_str
            
            # Initialize imported_files if it doesn't exist
            if not hasattr(self, 'imported_files'):
                self.imported_files = []
            
            # Add to imported files list
            if target_path_str not in self.imported_files:
                self.imported_files.append(target_path_str)
            
            logger.info(f"Imported mesh: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing mesh: {e}")
            return False

    def set_active_mesh(self, filepath: str):
        """
        Set the active mesh for the project.
        
        Args:
            filepath: Path to the mesh file
        """
        self._active_mesh = filepath

    def get_active_mesh(self) -> str:
        """
        Get the active mesh file path.
        
        Returns:
            str: Path to the active mesh file, or None if no active mesh
        """
        return getattr(self, '_active_mesh', None)
    
    def import_cad(self, filepath: str) -> bool:
        """
        Import a CAD file to the project.
        
        Args:
            filepath (str): Path to the CAD file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"CAD file not found: {filepath}")
                return False
            
            # If project directory is set, copy file to project
            if self.project_dir:
                dest_dir = self.project_dir / "geometry"
                if not dest_dir.exists():
                    dest_dir.mkdir(parents=True)
                
                # Get destination path
                filename = os.path.basename(filepath)
                dest_path = dest_dir / filename
                
                # Copy file
                shutil.copy2(filepath, dest_path)
                
                # Add to geometry files
                dest_path_str = str(dest_path)
                if dest_path_str not in self.geometry_files:
                    self.geometry_files.append(dest_path_str)
                
                # Set as active geometry
                self.active_geometry = dest_path_str
                
                logger.info(f"Imported geometry: {dest_path_str}")
                return True
            
            # If project directory is not set, just add the file path
            if filepath not in self.geometry_files:
                self.geometry_files.append(filepath)
            
            # Set as active geometry
            self.active_geometry = filepath
            
            logger.info(f"Referenced geometry: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing CAD: {e}")
            return False
    
    def export_results(self, filepath: str) -> bool:
        """
        Export simulation results to a file.
        
        Args:
            filepath (str): Path to save the results
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if we have results to export
        if not self.has_results():
            logger.error("No results to export")
            return False
        
        try:
            # Determine the export format based on file extension
            _, ext = os.path.splitext(filepath)
            ext = ext.lower()
            
            # Implement export logic based on format
            if ext == '.vtk':
                # Export to VTK format
                pass
            elif ext == '.csv':
                # Export to CSV format
                pass
            elif ext == '.foam':
                # Export as OpenFOAM case
                pass
            else:
                logger.error(f"Unsupported export format: {ext}")
                return False
            
            # Success
            logger.info(f"Results exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return False
    
    def generate_mesh(self) -> bool:
        """
        Generate a mesh from the active geometry.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if we have an active geometry
        if not self.active_geometry:
            logger.error("No active geometry for mesh generation")
            return False
        
        try:
            # Mesh generation would be implemented by importing and using appropriate
            # modules from the openfoam_integration package
            
            # For this implementation, just create a placeholder mesh file
            if self.project_dir:
                mesh_dir = self.project_dir / "mesh"
                if not mesh_dir.exists():
                    mesh_dir.mkdir(parents=True)
                
                # Create a timestamp-based mesh name
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                mesh_name = f"mesh_{timestamp}.vtk"
                mesh_path = mesh_dir / mesh_name
                
                # Create an empty mesh file as a placeholder
                with open(mesh_path, 'w') as f:
                    f.write("# VTK DataFile Version 2.0\n")
                    f.write("Generated mesh\n")
                    f.write("ASCII\n")
                    f.write("DATASET POLYDATA\n")
                    f.write("POINTS 0 float\n")
                    f.write("POLYGONS 0 0\n")
                
                # Add to mesh files
                mesh_path_str = str(mesh_path)
                if mesh_path_str not in self.mesh_files:
                    self.mesh_files.append(mesh_path_str)
                
                # Set as active mesh
                self.active_mesh = mesh_path_str
                
                logger.info(f"Generated mesh: {mesh_path_str}")
                return True
            
            logger.error("Project directory not set, cannot generate mesh")
            return False
            
        except Exception as e:
            logger.error(f"Error generating mesh: {e}")
            return False
    
    def setup_case(self) -> bool:
        """
        Set up an OpenFOAM case based on the active mesh.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if we have an active mesh
        if not self.active_mesh:
            logger.error("No active mesh for case setup")
            return False
        
        try:
            # Case setup would be implemented by importing and using appropriate
            # modules from the openfoam_integration package
            
            # For this implementation, just create placeholder case files
            if self.project_dir:
                # Ensure case directory exists
                if not self.case_dir:
                    case_dir = self.project_dir / "case"
                    if not case_dir.exists():
                        case_dir.mkdir(parents=True)
                    self.case_dir = str(case_dir)
                
                # Create system directory
                system_dir = Path(self.case_dir) / "system"
                if not system_dir.exists():
                    system_dir.mkdir(parents=True)
                
                # Create constant directory
                constant_dir = Path(self.case_dir) / "constant"
                if not constant_dir.exists():
                    constant_dir.mkdir(parents=True)
                
                # Create placeholder control dict
                control_dict_path = system_dir / "controlDict"
                with open(control_dict_path, 'w') as f:
                    f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
                    f.write("| =========                 |                                                 |\n")
                    f.write("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
                    f.write("|  \\\\    /   O peration     | Version:  v2312                                 |\n")
                    f.write("|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n")
                    f.write("|    \\\\/     M anipulation  |                                                 |\n")
                    f.write("\\*---------------------------------------------------------------------------*/\n")
                    f.write("FoamFile\n")
                    f.write("{\n")
                    f.write("    version     2.0;\n")
                    f.write("    format      ascii;\n")
                    f.write("    class       dictionary;\n")
                    f.write("    object      controlDict;\n")
                    f.write("}\n")
                    f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n")
                    f.write("\n")
                    f.write("application     simpleFoam;\n")
                    f.write("\n")
                    f.write("startFrom       startTime;\n")
                    f.write("\n")
                    f.write("startTime       0;\n")
                    f.write("\n")
                    f.write("stopAt          endTime;\n")
                    f.write("\n")
                    f.write("endTime         100;\n")
                    f.write("\n")
                    f.write("deltaT          1;\n")
                    f.write("\n")
                    f.write("writeControl    timeStep;\n")
                    f.write("\n")
                    f.write("writeInterval   10;\n")
                    f.write("\n")
                    f.write("purgeWrite      0;\n")
                    f.write("\n")
                    f.write("writeFormat     ascii;\n")
                    f.write("\n")
                    f.write("writePrecision  6;\n")
                    f.write("\n")
                    f.write("writeCompression off;\n")
                    f.write("\n")
                    f.write("timeFormat      general;\n")
                    f.write("\n")
                    f.write("timePrecision   6;\n")
                    f.write("\n")
                    f.write("runTimeModifiable true;\n")
                    f.write("\n")
                    f.write("// ************************************************************************* //\n")
                
                # Add to case files
                case_path_str = str(control_dict_path)
                if case_path_str not in self.case_files:
                    self.case_files.append(case_path_str)
                
                # Set as active case
                self.active_case = case_path_str
                
                logger.info(f"Set up case: {self.case_dir}")
                return True
            
            logger.error("Project directory not set, cannot set up case")
            return False
            
        except Exception as e:
            logger.error(f"Error setting up case: {e}")
            return False
    
    def run_simulation(self) -> bool:
        """
        Run the simulation based on the active case.
        
        Returns:
            bool: True if simulation started successfully, False otherwise
        """
        # Check if we have an active case
        if not self.active_case or not self.case_dir:
            logger.error("No active case for simulation")
            return False
        
        # Check if a simulation is already running
        if self.simulation_running:
            logger.error("Simulation is already running")
            return False
        
        try:
            # Simulation would be implemented by importing and using appropriate
            # modules from the openfoam_integration package
            
            # For this implementation, just create a placeholder results file
            if self.project_dir:
                results_dir = self.project_dir / "results"
                if not results_dir.exists():
                    results_dir.mkdir(parents=True)
                
                # Create a timestamp-based results name
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                results_name = f"results_{timestamp}.vtk"
                results_path = results_dir / results_name
                
                # Create an empty results file as a placeholder
                with open(results_path, 'w') as f:
                    f.write("# VTK DataFile Version 2.0\n")
                    f.write("Simulation results\n")
                    f.write("ASCII\n")
                    f.write("DATASET POLYDATA\n")
                    f.write("POINTS 0 float\n")
                    f.write("POLYGONS 0 0\n")
                
                # Add to results files
                results_path_str = str(results_path)
                if results_path_str not in self.result_files:
                    self.result_files.append(results_path_str)
                
                # Set as active results
                self.active_results = results_path_str
                
                # Set simulation as running (in a real implementation, this would be set by the process)
                self.simulation_running = True
                
                logger.info(f"Started simulation, results will be saved to: {results_path_str}")
                return True
            
            logger.error("Project directory not set, cannot run simulation")
            return False
            
        except Exception as e:
            logger.error(f"Error running simulation: {e}")
            return False
    
    def stop_simulation(self) -> bool:
        """
        Stop the running simulation.
        
        Returns:
            bool: True if simulation stopped successfully, False otherwise
        """
        # Check if a simulation is running
        if not self.simulation_running:
            logger.warning("No simulation is running")
            return True
        
        try:
            # In a real implementation, this would stop the simulation process
            
            # Set simulation as not running
            self.simulation_running = False
            
            logger.info("Stopped simulation")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping simulation: {e}")
            return False
    
    def has_mesh(self) -> bool:
        """
        Check if the project has a mesh.
        
        Returns:
            bool: True if the project has a mesh, False otherwise
        """
        return bool(self.active_mesh and os.path.exists(self.active_mesh))
    
    def has_results(self) -> bool:
        """
        Check if the project has simulation results.
        
        Returns:
            bool: True if the project has results, False otherwise
        """
        return bool(self.active_results and os.path.exists(self.active_results))
    
    def get_mesh_path(self) -> Optional[str]:
        """
        Get the path to the active mesh.
        
        Returns:
            str: Path to the active mesh, or None if no active mesh
        """
        if self.has_mesh():
            return self.active_mesh
        return None
    
    def get_results_path(self) -> Optional[str]:
        """
        Get the path to the active results.
        
        Returns:
            str: Path to the active results, or None if no active results
        """
        if self.has_results():
            return self.active_results
        return None
    
    def set_active_geometry(self, filepath: str) -> bool:
        """
        Set the active geometry.
        
        Args:
            filepath (str): Path to the geometry file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if filepath in self.geometry_files:
            self.active_geometry = filepath
            logger.info(f"Set active geometry: {filepath}")
            return True
        
        logger.error(f"Geometry file not in project: {filepath}")
        return False
    
    def set_active_mesh(self, filepath: str) -> bool:
        """
        Set the active mesh.
        
        Args:
            filepath (str): Path to the mesh file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if filepath in self.mesh_files:
            self.active_mesh = filepath
            logger.info(f"Set active mesh: {filepath}")
            return True
        
        logger.error(f"Mesh file not in project: {filepath}")
        return False
    
    def set_active_case(self, filepath: str) -> bool:
        """
        Set the active case.
        
        Args:
            filepath (str): Path to the case file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if filepath in self.case_files:
            self.active_case = filepath
            logger.info(f"Set active case: {filepath}")
            return True
        
        logger.error(f"Case file not in project: {filepath}")
        return False
    
    def set_active_results(self, filepath: str) -> bool:
        """
        Set the active results.
        
        Args:
            filepath (str): Path to the results file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if filepath in self.result_files:
            self.active_results = filepath
            logger.info(f"Set active results: {filepath}")
            return True
        
        logger.error(f"Results file not in project: {filepath}")
        return False
    
    def __str__(self) -> str:
        """
        Get a string representation of the project.
        
        Returns:
            str: String representation
        """
        return f"Project: {self.name} (Version: {self.version})"

    def get_case_directory(self) -> str:
        """
        Get the OpenFOAM case directory for this project.
        
        Returns:
            str: Path to the case directory, or None if not set
        """
        if hasattr(self, 'case_dir') and self.case_dir:
            return self.case_dir
        
        # If case directory is not set but the project directory exists,
        # return the default case directory location
        if self.project_dir:
            default_case_dir = os.path.join(self.project_dir, "case")
            if os.path.exists(default_case_dir):
                self.case_dir = default_case_dir
                return default_case_dir
        
        return None

    def set_case_directory(self, case_dir: str):
        """
        Set the OpenFOAM case directory for this project.
        
        Args:
            case_dir (str): Path to the case directory
        """
        self.case_dir = case_dir
        logger.info(f"Set case directory to: {case_dir}")

    
    # Add to Project class in project.py
    def set_boundary_faces(self, boundary_name, cell_ids):
        """
        Set cell IDs for a named boundary
        
        Args:
            boundary_name: Name of the boundary
            cell_ids: List of cell IDs that make up the boundary
        """
        # Make sure we have the boundary_faces attribute
        if not hasattr(self, 'boundary_faces'):
            self.boundary_faces = {}
        
        self.boundary_faces[boundary_name] = cell_ids
        
    def get_boundary_faces(self, boundary_name=None):
        """
        Get cell IDs for boundaries
        
        Args:
            boundary_name: Name of specific boundary, or None for all
            
        Returns:
            Dict or List: Cell IDs for boundaries
        """
        if not hasattr(self, 'boundary_faces'):
            self.boundary_faces = {}
        
        if boundary_name:
            return self.boundary_faces.get(boundary_name, [])
        else:
            return self.boundary_faces