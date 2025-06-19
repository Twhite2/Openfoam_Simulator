#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project I/O functionality for Openfoam_Simulator application.

This module handles saving and loading F3D project files, including:
- Project file format handling
- Serialization and deserialization of project data
- Project file versioning
- Project file compression
- Project file encryption (if needed)
- Project file validation
"""

import os
import sys
import json
import time
import shutil
import zipfile
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, TextIO

# Import utility modules
from ..utils.logger import get_logger
from ..config import get_value, set_value, get_config_manager

# Import models
from ..models.project import Project
from ..models.mesh_model import MeshModel
from ..models.flow_model import FlowModel
from ..models.simulation_case import SimulationCase
from ..models.results_model import ResultsModel

logger = get_logger(__name__)

# Current project file format version
CURRENT_FORMAT_VERSION = "1.0.0"

class ProjectFileError(Exception):
    """Exception raised for errors in project file operations."""
    pass


class ProjectIO:
    """
    Class for handling project file I/O operations.
    
    This class provides methods for saving and loading project files,
    handling serialization, compression, and file format versioning.
    """
    
    def __init__(self):
        """Initialize the ProjectIO instance."""
        # Set default file extension
        self.file_extension = ".f3d"
        
        # Set file format version
        self.format_version = CURRENT_FORMAT_VERSION
    
    def save_project(self, project: Project, filepath: str, compress: bool = True) -> bool:
        """
        Save a project to a file.
        
        Args:
            project (Project): Project to save
            filepath (str): Path to save the project to
            compress (bool, optional): Whether to compress the project file
            
        Returns:
            bool: True if saving was successful, False otherwise
            
        Raises:
            ProjectFileError: If an error occurs during saving
        """
        # Ensure filepath has correct extension
        if not filepath.lower().endswith(self.file_extension):
            filepath += self.file_extension
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            if compress:
                # Save as compressed file (zip)
                return self._save_compressed(project, filepath)
            else:
                # Save as plain JSON
                return self._save_plain(project, filepath)
                
        except Exception as e:
            logger.error(f"Error saving project: {e}")
            raise ProjectFileError(f"Failed to save project: {str(e)}")
    
    def _save_plain(self, project: Project, filepath: str) -> bool:
        """
        Save project as plain JSON file.
        
        Args:
            project (Project): Project to save
            filepath (str): Path to save the project to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert project to dictionary
            project_data = self._project_to_dict(project)
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(project_data, f, indent=2)
            
            # Update project's filepath
            project.filepath = filepath
            
            logger.info(f"Project saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving project as plain file: {e}")
            return False
    
    def _save_compressed(self, project: Project, filepath: str) -> bool:
        """
        Save project as compressed zip file.
        
        Args:
            project (Project): Project to save
            filepath (str): Path to save the project to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save project metadata
                project_data = self._project_to_dict(project)
                metadata_path = os.path.join(temp_dir, "project.json")
                with open(metadata_path, 'w') as f:
                    json.dump(project_data, f, indent=2)
                
                # Save accompanying files if project has a directory
                if project.project_dir:
                    self._copy_project_files(project, temp_dir)
                
                # Create zip file
                with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Add metadata file
                    zipf.write(metadata_path, "project.json")
                    
                    # Add other files from temp directory
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            if file == "project.json":
                                continue  # Already added
                            
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, temp_dir)
                            zipf.write(file_path, rel_path)
            
            # Update project's filepath
            project.filepath = filepath
            
            logger.info(f"Project saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving project as compressed file: {e}")
            return False
    
    def _copy_project_files(self, project: Project, target_dir: str):
        """
        Copy project files to a target directory.
        
        Args:
            project (Project): Project containing files
            target_dir (str): Directory to copy files to
        """
        if not project.project_dir:
            return
        
        try:
            # Create files directory
            files_dir = os.path.join(target_dir, "files")
            os.makedirs(files_dir, exist_ok=True)
            
            # Copy mesh files
            if project.mesh_files:
                mesh_dir = os.path.join(files_dir, "mesh")
                os.makedirs(mesh_dir, exist_ok=True)
                
                for file_path in project.mesh_files:
                    if os.path.isfile(file_path):
                        # Copy file to mesh directory
                        shutil.copy2(file_path, mesh_dir)
            
            # Copy geometry files
            if project.geometry_files:
                geom_dir = os.path.join(files_dir, "geometry")
                os.makedirs(geom_dir, exist_ok=True)
                
                for file_path in project.geometry_files:
                    if os.path.isfile(file_path):
                        # Copy file to geometry directory
                        shutil.copy2(file_path, geom_dir)
            
            # Copy case files
            if project.case_files:
                case_dir = os.path.join(files_dir, "case")
                os.makedirs(case_dir, exist_ok=True)
                
                for file_path in project.case_files:
                    if os.path.isfile(file_path):
                        # Copy file to case directory
                        shutil.copy2(file_path, case_dir)
            
            # Copy result files (if not too large)
            max_result_size = get_value('io.max_result_file_size', 50) * 1024 * 1024  # Default 50MB
            
            if project.result_files:
                results_dir = os.path.join(files_dir, "results")
                os.makedirs(results_dir, exist_ok=True)
                
                for file_path in project.result_files:
                    if os.path.isfile(file_path) and os.path.getsize(file_path) <= max_result_size:
                        # Copy file to results directory
                        shutil.copy2(file_path, results_dir)
                    elif os.path.isfile(file_path):
                        logger.warning(f"Skipping large result file: {file_path}")
            
            # Copy visualization files
            if project.visualization_files:
                viz_dir = os.path.join(files_dir, "visualization")
                os.makedirs(viz_dir, exist_ok=True)
                
                for file_path in project.visualization_files:
                    if os.path.isfile(file_path):
                        # Copy file to visualization directory
                        shutil.copy2(file_path, viz_dir)
            
            # Copy report files
            if project.report_files:
                report_dir = os.path.join(files_dir, "reports")
                os.makedirs(report_dir, exist_ok=True)
                
                for file_path in project.report_files:
                    if os.path.isfile(file_path):
                        # Copy file to reports directory
                        shutil.copy2(file_path, report_dir)
            
        except Exception as e:
            logger.warning(f"Error copying project files: {e}")
    
    def _project_to_dict(self, project: Project) -> Dict[str, Any]:
        """
        Convert a project to a dictionary for serialization.
        
        Args:
            project (Project): Project to convert
            
        Returns:
            Dict[str, Any]: Dictionary representation of the project
        """
        data = {
            "format_version": self.format_version,
            "name": project.name,
            "description": project.description,
            "creation_time": project.creation_time,
            "last_modified_time": project.last_modified_time,
            "author": project.author,
            "tags": project.tags,
            "filepath": project.filepath,
            "project_dir": project.project_dir,
            "mesh_files": project.mesh_files,
            "geometry_files": project.geometry_files,
            "case_files": project.case_files,
            "boundary_files": project.boundary_files,
            "result_files": project.result_files,
            "visualization_files": project.visualization_files,
            "report_files": project.report_files,
            "metadata": project.metadata
        }
        
        # Add model data if available
        if project.mesh_model:
            data["mesh_model"] = project.mesh_model.to_dict()
        
        if project.flow_model:
            data["flow_model"] = project.flow_model.to_dict()
        
        if project.simulation_case:
            data["simulation_case"] = project.simulation_case.to_dict()
        
        if project.results_model:
            data["results_model"] = project.results_model.to_dict()
        
        return data
    
    def load_project(self, filepath: str) -> Optional[Project]:
        """
        Load a project from a file.
        
        Args:
            filepath (str): Path to the project file
            
        Returns:
            Optional[Project]: Loaded project or None if loading failed
            
        Raises:
            ProjectFileError: If an error occurs during loading
        """
        if not os.path.isfile(filepath):
            logger.error(f"Project file not found: {filepath}")
            raise ProjectFileError(f"Project file not found: {filepath}")
        
        try:
            # Check if file is a zip file
            if zipfile.is_zipfile(filepath):
                return self._load_compressed(filepath)
            else:
                # Try to load as plain JSON
                return self._load_plain(filepath)
                
        except Exception as e:
            logger.error(f"Error loading project: {e}")
            raise ProjectFileError(f"Failed to load project: {str(e)}")
    
    def _load_plain(self, filepath: str) -> Optional[Project]:
        """
        Load project from plain JSON file.
        
        Args:
            filepath (str): Path to the project file
            
        Returns:
            Optional[Project]: Loaded project or None if loading failed
        """
        try:
            with open(filepath, 'r') as f:
                project_data = json.load(f)
            
            # Check format version
            format_version = project_data.get('format_version', '0.0.0')
            if not self._is_compatible_version(format_version):
                logger.warning(f"Project file version {format_version} may not be compatible with current version {self.format_version}")
            
            # Create project from data
            project = self._dict_to_project(project_data)
            
            # Set filepath
            project.filepath = filepath
            
            logger.info(f"Project loaded from {filepath}")
            return project
            
        except Exception as e:
            logger.error(f"Error loading project from plain file: {e}")
            return None
    
    def _load_compressed(self, filepath: str) -> Optional[Project]:
        """
        Load project from compressed zip file.
        
        Args:
            filepath (str): Path to the project file
            
        Returns:
            Optional[Project]: Loaded project or None if loading failed
        """
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract zip file
                with zipfile.ZipFile(filepath, 'r') as zipf:
                    zipf.extractall(temp_dir)
                
                # Load project metadata
                metadata_path = os.path.join(temp_dir, "project.json")
                if not os.path.isfile(metadata_path):
                    logger.error("Project metadata not found in zip file")
                    return None
                
                with open(metadata_path, 'r') as f:
                    project_data = json.load(f)
                
                # Check format version
                format_version = project_data.get('format_version', '0.0.0')
                if not self._is_compatible_version(format_version):
                    logger.warning(f"Project file version {format_version} may not be compatible with current version {self.format_version}")
                
                # Create project from data
                project = self._dict_to_project(project_data)
                
                # Set filepath
                project.filepath = filepath
                
                # Handle extracted files
                files_dir = os.path.join(temp_dir, "files")
                if os.path.isdir(files_dir):
                    self._process_extracted_files(project, files_dir)
                
                logger.info(f"Project loaded from {filepath}")
                return project
                
        except Exception as e:
            logger.error(f"Error loading project from compressed file: {e}")
            return None
    
    def _process_extracted_files(self, project: Project, files_dir: str):
        """
        Process files extracted from the project archive.
        
        Args:
            project (Project): Project to update with file information
            files_dir (str): Directory containing extracted files
        """
        # Create project directory if needed
        if not project.project_dir:
            # Create a directory for the project files
            project_name = os.path.splitext(os.path.basename(project.filepath))[0]
            project_dir = os.path.join(get_value('paths.runs', 'runs'), project_name)
            os.makedirs(project_dir, exist_ok=True)
            project.project_dir = project_dir
        
        # Process mesh files
        mesh_dir = os.path.join(files_dir, "mesh")
        if os.path.isdir(mesh_dir):
            # Create mesh directory in project
            project_mesh_dir = os.path.join(project.project_dir, "mesh")
            os.makedirs(project_mesh_dir, exist_ok=True)
            
            # Copy mesh files
            mesh_files = []
            for file in os.listdir(mesh_dir):
                src_path = os.path.join(mesh_dir, file)
                dst_path = os.path.join(project_mesh_dir, file)
                shutil.copy2(src_path, dst_path)
                mesh_files.append(dst_path)
            
            # Update project
            project.mesh_files = mesh_files
        
        # Process geometry files
        geom_dir = os.path.join(files_dir, "geometry")
        if os.path.isdir(geom_dir):
            # Create geometry directory in project
            project_geom_dir = os.path.join(project.project_dir, "geometry")
            os.makedirs(project_geom_dir, exist_ok=True)
            
            # Copy geometry files
            geometry_files = []
            for file in os.listdir(geom_dir):
                src_path = os.path.join(geom_dir, file)
                dst_path = os.path.join(project_geom_dir, file)
                shutil.copy2(src_path, dst_path)
                geometry_files.append(dst_path)
            
            # Update project
            project.geometry_files = geometry_files
        
        # Process case files
        case_dir = os.path.join(files_dir, "case")
        if os.path.isdir(case_dir):
            # Create case directory in project
            project_case_dir = os.path.join(project.project_dir, "case")
            os.makedirs(project_case_dir, exist_ok=True)
            
            # Copy case files
            case_files = []
            for file in os.listdir(case_dir):
                src_path = os.path.join(case_dir, file)
                dst_path = os.path.join(project_case_dir, file)
                shutil.copy2(src_path, dst_path)
                case_files.append(dst_path)
            
            # Update project
            project.case_files = case_files
        
        # Similar processing for other file types...
        # Process results files
        results_dir = os.path.join(files_dir, "results")
        if os.path.isdir(results_dir):
            # Create results directory in project
            project_results_dir = os.path.join(project.project_dir, "results")
            os.makedirs(project_results_dir, exist_ok=True)
            
            # Copy results files
            result_files = []
            for file in os.listdir(results_dir):
                src_path = os.path.join(results_dir, file)
                dst_path = os.path.join(project_results_dir, file)
                shutil.copy2(src_path, dst_path)
                result_files.append(dst_path)
            
            # Update project
            project.result_files = result_files
        
        # Process visualization files
        viz_dir = os.path.join(files_dir, "visualization")
        if os.path.isdir(viz_dir):
            # Create visualization directory in project
            project_viz_dir = os.path.join(project.project_dir, "visualization")
            os.makedirs(project_viz_dir, exist_ok=True)
            
            # Copy visualization files
            viz_files = []
            for file in os.listdir(viz_dir):
                src_path = os.path.join(viz_dir, file)
                dst_path = os.path.join(project_viz_dir, file)
                shutil.copy2(src_path, dst_path)
                viz_files.append(dst_path)
            
            # Update project
            project.visualization_files = viz_files
        
        # Process report files
        report_dir = os.path.join(files_dir, "reports")
        if os.path.isdir(report_dir):
            # Create reports directory in project
            project_report_dir = os.path.join(project.project_dir, "reports")
            os.makedirs(project_report_dir, exist_ok=True)
            
            # Copy report files
            report_files = []
            for file in os.listdir(report_dir):
                src_path = os.path.join(report_dir, file)
                dst_path = os.path.join(project_report_dir, file)
                shutil.copy2(src_path, dst_path)
                report_files.append(dst_path)
            
            # Update project
            project.report_files = report_files
    
    def _dict_to_project(self, data: Dict[str, Any]) -> Project:
        """
        Create a project from dictionary data.
        
        Args:
            data (Dict[str, Any]): Dictionary data
            
        Returns:
            Project: Created project
        """
        project = Project()
        
        # Set basic properties
        project.name = data.get('name', 'Untitled Project')
        project.description = data.get('description', '')
        project.creation_time = data.get('creation_time', time.time())
        project.last_modified_time = data.get('last_modified_time', time.time())
        project.author = data.get('author', '')
        project.tags = data.get('tags', [])
        project.filepath = data.get('filepath', '')
        project.project_dir = data.get('project_dir', '')
        
        # Set file lists
        project.mesh_files = data.get('mesh_files', [])
        project.geometry_files = data.get('geometry_files', [])
        project.case_files = data.get('case_files', [])
        project.boundary_files = data.get('boundary_files', [])
        project.result_files = data.get('result_files', [])
        project.visualization_files = data.get('visualization_files', [])
        project.report_files = data.get('report_files', [])
        
        # Set metadata
        project.metadata = data.get('metadata', {})
        
        # Load models if available
        if 'mesh_model' in data and data['mesh_model']:
            try:
                project.mesh_model = MeshModel.from_dict(data['mesh_model'])
            except Exception as e:
                logger.warning(f"Error loading mesh model: {e}")
        
        if 'flow_model' in data and data['flow_model']:
            try:
                project.flow_model = FlowModel.from_dict(data['flow_model'])
            except Exception as e:
                logger.warning(f"Error loading flow model: {e}")
        
        if 'simulation_case' in data and data['simulation_case']:
            try:
                project.simulation_case = SimulationCase.from_dict(data['simulation_case'])
            except Exception as e:
                logger.warning(f"Error loading simulation case: {e}")
        
        if 'results_model' in data and data['results_model']:
            try:
                project.results_model = ResultsModel.from_dict(data['results_model'])
            except Exception as e:
                logger.warning(f"Error loading results model: {e}")
        
        return project
    
    def _is_compatible_version(self, version: str) -> bool:
        """
        Check if a format version is compatible with the current version.
        
        Args:
            version (str): Version to check
            
        Returns:
            bool: True if compatible, False otherwise
        """
        # Parse versions
        try:
            v_parts = [int(x) for x in version.split('.')]
            current_parts = [int(x) for x in self.format_version.split('.')]
            
            # Major version must match for compatibility
            return v_parts[0] == current_parts[0]
            
        except Exception:
            # If version parsing fails, assume incompatible
            return False
    
    def create_backup(self, project: Project) -> str:
        """
        Create a backup of a project file.
        
        Args:
            project (Project): Project to backup
            
        Returns:
            str: Path to backup file
            
        Raises:
            ProjectFileError: If an error occurs during backup
        """
        if not project.filepath:
            raise ProjectFileError("Cannot create backup for unsaved project")
        
        try:
            # Create backup directory if it doesn't exist
            backup_dir = get_value('paths.backups', os.path.join('runs', 'backups'))
            os.makedirs(backup_dir, exist_ok=True)
            
            # Create backup filename
            filename = os.path.basename(project.filepath)
            backup_name = f"{os.path.splitext(filename)[0]}_{int(time.time())}.backup.f3d"
            backup_path = os.path.join(backup_dir, backup_name)
            
            # Save backup
            with open(project.filepath, 'rb') as src:
                with open(backup_path, 'wb') as dst:
                    dst.write(src.read())
            
            logger.info(f"Created backup at {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise ProjectFileError(f"Failed to create backup: {str(e)}")
    
    def restore_from_backup(self, backup_path: str) -> Optional[Project]:
        """
        Restore a project from a backup file.
        
        Args:
            backup_path (str): Path to backup file
            
        Returns:
            Optional[Project]: Restored project or None if restoration failed
            
        Raises:
            ProjectFileError: If an error occurs during restoration
        """
        if not os.path.isfile(backup_path):
            raise ProjectFileError(f"Backup file not found: {backup_path}")
        
        try:
            # Load project from backup
            return self.load_project(backup_path)
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            raise ProjectFileError(f"Failed to restore from backup: {str(e)}")


# Create a singleton instance
_project_io = None

def get_project_io() -> ProjectIO:
    """
    Get the ProjectIO singleton instance.
    
    Returns:
        ProjectIO: The ProjectIO instance
    """
    global _project_io
    if _project_io is None:
        _project_io = ProjectIO()
    
    return _project_io

def save_project(project: Project, filepath: str = None, compress: bool = True) -> bool:
    """
    Save a project to a file.
    
    Args:
        project (Project): Project to save
        filepath (str, optional): Path to save to, if None uses project.filepath
        compress (bool, optional): Whether to compress the project file
        
    Returns:
        bool: True if saving successful, False otherwise
    """
    if filepath is None and not project.filepath:
        logger.error("No filepath specified for project")
        return False
    
    try:
        return get_project_io().save_project(project, filepath or project.filepath, compress)
    except ProjectFileError as e:
        logger.error(f"Error saving project: {e}")
        return False

def load_project(filepath: str) -> Optional[Project]:
    """
    Load a project from a file.
    
    Args:
        filepath (str): Path to the project file
        
    Returns:
        Optional[Project]: Loaded project or None if loading failed
    """
    try:
        return get_project_io().load_project(filepath)
    except ProjectFileError as e:
        logger.error(f"Error loading project: {e}")
        return None

def create_backup(project: Project) -> Optional[str]:
    """
    Create a backup of a project file.
    
    Args:
        project (Project): Project to backup
        
    Returns:
        Optional[str]: Path to backup file or None if backup failed
    """
    try:
        return get_project_io().create_backup(project)
    except ProjectFileError as e:
        logger.error(f"Error creating backup: {e}")
        return None

def restore_from_backup(backup_path: str) -> Optional[Project]:
    """
    Restore a project from a backup file.
    
    Args:
        backup_path (str): Path to backup file
        
    Returns:
        Optional[Project]: Restored project or None if restoration failed
    """
    try:
        return get_project_io().restore_from_backup(backup_path)
    except ProjectFileError as e:
        logger.error(f"Error restoring from backup: {e}")
        return None

def is_valid_project_file(filepath: str) -> bool:
    """
    Check if a file is a valid F3D project file.
    
    Args:
        filepath (str): Path to the file to check
        
    Returns:
        bool: True if file is a valid project file, False otherwise
    """
    if not os.path.isfile(filepath):
        return False
    
    # Check file extension
    if not filepath.lower().endswith('.f3d'):
        return False
    
    # Try to open the file
    try:
        # Check if it's a zip file
        if zipfile.is_zipfile(filepath):
            with zipfile.ZipFile(filepath, 'r') as zipf:
                # Check if project.json exists in the archive
                if "project.json" not in zipf.namelist():
                    return False
                
                # Read project.json
                with zipf.open("project.json") as f:
                    data = json.load(f)
                
                # Check if it has essential fields
                return "format_version" in data and "name" in data
        else:
            # Try to parse as JSON
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Check if it has essential fields
            return "format_version" in data and "name" in data
    
    except Exception as e:
        logger.error(f"Error checking project file: {e}")
        return False