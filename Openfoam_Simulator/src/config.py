#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration management for Openfoam_Simulator

This module handles application configuration, including:
- Loading/saving user settings
- Detecting and configuring OpenFOAM and VTK environments
- Managing paths to resources, templates, and examples
- Defining default configurations for simulation cases
"""

import os
import sys
import yaml
import json
import logging
import platform
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Setup logging
logger = logging.getLogger(__name__)

# Default paths
APP_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = APP_DIR / "data"
TEMPLATES_DIR = APP_DIR / "templates"
EXAMPLES_DIR = APP_DIR / "examples"
RUNS_DIR = APP_DIR / "runs"
USER_CONFIG_FILE = Path.home() / ".openfoam_simulator" / "config.yaml"

# Default configuration
DEFAULT_CONFIG = {
    "app": {
        "name": "Openfoam_Simulator",
        "version": "0.1.0",
        "theme": "dark",
        "language": "en",
        "log_level": "INFO",
        "auto_save": True,
        "autosave_interval_minutes": 5,
        "recent_projects": [],
        "max_recent_projects": 10,
    },
    "paths": {
        "templates": str(TEMPLATES_DIR),
        "examples": str(EXAMPLES_DIR),
        "runs": str(RUNS_DIR),
        "data": str(DATA_DIR),
    },
    "openfoam": {
        "version": "12",
        "solver_parallelism": "auto",  # or a specific number of cores
        "default_solver_settings": {
            "singlePhase": {
                "solver": "simpleFoam",
                "turbulence_model": "kEpsilon",
                "max_iterations": 1000,
                "convergence_tolerance": 1e-6,
            },
            "multiPhase": {
                "solver": "multiphaseInterFoam",
                "max_iterations": 2000,
                "convergence_tolerance": 1e-5,
            },
            "pigging": {
                "solver": "interFoam",
                "max_iterations": 3000,
                "convergence_tolerance": 1e-4,
            },
            "spill": {
                "solver": "multiphaseInterFoam",
                "max_iterations": 2500,
                "convergence_tolerance": 1e-5,
            },
        },
    },
    "vtk": {
        "viewport_settings": {
            "background_color": [0.2, 0.2, 0.2],
            "camera_position": [1.0, 1.0, 1.0],
            "camera_focal_point": [0.0, 0.0, 0.0],
            "camera_view_up": [0.0, 0.0, 1.0],
        },
        "renderer_settings": {
            "use_shadows": False,
            "use_depth_peeling": True,
            "depth_peeling_layers": 4,
            "ambient_light": 0.3,
        },
        "default_colormap": "Cool to Warm",
        "scalar_bar_visibility": True,
        "axes_visibility": True,
        "default_representation": "Surface",
        "edge_visibility": False,
        "point_size": 5,
        "line_width": 1,
    },
    "gui": {
        "window_size": [1280, 800],
        "window_position": [100, 100],
        "viewport_size_ratio": 0.65,  # Viewport takes 65% of window width
        "explorer_width": 250,        # Project explorer panel width
        "properties_width": 300,      # Properties panel width
        "menu_style": "ribbon",       # "ribbon" or "traditional"
        "show_welcome_screen": True,
        "show_tips": True,
    },
    "industry": {
        "units": "metric",  # "metric" or "imperial" or "field"
        "fluid_database_path": "data/fluids.json",
        "material_database_path": "data/materials.json",
        "simulation_defaults": {
            "oil_density": 850,       # kg/m³
            "oil_viscosity": 0.03,    # Pa·s
            "water_density": 1000,    # kg/m³
            "water_viscosity": 0.001, # Pa·s
            "gas_density": 0.8,       # kg/m³
            "gas_viscosity": 1.8e-5,  # Pa·s
            "surface_tension": 0.025, # N/m
            "gravity": [0, 0, -9.81], # m/s²
        }
    }
}


class ConfigManager:
    """
    Class to manage application configuration.
    """
    def __init__(self):
        self.config = DEFAULT_CONFIG.copy()
        self.user_config_path = USER_CONFIG_FILE
        self._setup_directories()
        
    def _setup_directories(self):
        """Ensure required directories exist."""
        for dir_path in [DATA_DIR, RUNS_DIR, Path.home() / ".openfoam_simulator"]:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from user config file. If not found, use defaults.
        """
        try:
            if self.user_config_path.exists():
                with open(self.user_config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    
                # Deep merge with defaults (user config takes precedence)
                self._deep_update(self.config, user_config)
                logger.info(f"Loaded configuration from {self.user_config_path}")
            else:
                logger.info("User configuration not found. Using defaults.")
                self.save_config()  # Create default config file
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration.")
            
        # Update paths based on environment
        self._update_environment_paths()
        return self.config
    
    def _update_environment_paths(self):
        """Update configuration with environment-specific paths."""
        # Check for environment variables that might override paths
        for env_var, config_path in [
            ('F3D_TEMPLATES_DIR', 'paths.templates'),
            ('F3D_EXAMPLES_DIR', 'paths.examples'),
            ('F3D_RUNS_DIR', 'paths.runs'),
            ('F3D_DATA_DIR', 'paths.data'),
        ]:
            if env_var in os.environ:
                self.set_value(config_path, os.environ[env_var])
        
        # Detect OpenFOAM environment
        self._detect_openfoam()
        
        # Detect VTK environment
        self._detect_vtk()
    
    def _detect_openfoam(self):
        """Detect OpenFOAM installation and update configuration."""
        # Check for WM_PROJECT_DIR environment variable (set when OpenFOAM is sourced)
        foam_dir = os.environ.get('WM_PROJECT_DIR')
        if foam_dir:
            self.config['openfoam']['installation_path'] = foam_dir
            version = os.environ.get('WM_PROJECT_VERSION', '').strip()
            if version:
                self.config['openfoam']['version'] = version
            logger.info(f"Detected OpenFOAM installation: {foam_dir}, version: {version}")
        else:
            # Try to find OpenFOAM in standard locations
            standard_paths = [
                '/opt/openfoam12',
                '/usr/lib/openfoam',
                '/opt/openfoam-12',
            ]
            for path in standard_paths:
                if os.path.exists(path):
                    self.config['openfoam']['installation_path'] = path
                    logger.info(f"Found OpenFOAM installation at {path}")
                    break
            else:
                logger.warning("OpenFOAM installation not found. Some features may not work.")
    
    def _detect_vtk(self):
        """Detect VTK installation and update configuration."""
        try:
            # Try to import VTK and get version
            import vtk
            vtk_version = vtk.vtkVersion.GetVTKVersion()
            
            # Store in config
            self.config['vtk']['version'] = vtk_version
            
            # Check if VTK was built with rendering support
            has_rendering = hasattr(vtk, 'vtkRenderWindow')
            has_opengl = hasattr(vtk, 'vtkOpenGLRenderer')
            
            # Check for advanced rendering capabilities
            if has_rendering and has_opengl:
                # Create a test renderer to check capabilities
                renderer = vtk.vtkRenderer()
                has_shadows = hasattr(renderer, 'SetUseShadows')
                has_depth_peeling = hasattr(renderer, 'SetUseDepthPeeling')
                
                # Update config with capabilities
                if has_shadows:
                    self.config['vtk']['renderer_settings']['use_shadows_supported'] = True
                if has_depth_peeling:
                    self.config['vtk']['renderer_settings']['use_depth_peeling_supported'] = True
            
            logger.info(f"Detected VTK version: {vtk_version}")
            
        except ImportError:
            logger.warning("VTK not found. Visualization features will not be available.")
            
        except Exception as e:
            logger.warning(f"Error detecting VTK: {e}")
    
    def save_config(self):
        """Save current configuration to user config file."""
        try:
            # Ensure the directory exists
            self.user_config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.user_config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Configuration saved to {self.user_config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get_value(self, path, default=None):
        """
        Get configuration value at the specified path.
        
        Args:
            path (str): Dot-separated path to the configuration value
            default: Value to return if path doesn't exist
            
        Returns:
            Configuration value or default if not found
        """
        current = self.config
        for key in path.split('.'):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def set_value(self, path, value):
        """
        Set configuration value at the specified path.
        
        Args:
            path (str): Dot-separated path to the configuration value
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        keys = path.split('.')
        current = self.config
        
        # Traverse to the right location
        for i, key in enumerate(keys[:-1]):
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the value
        current[keys[-1]] = value
        return True
    
    def add_recent_project(self, project_path):
        """
        Add a project path to the recent projects list.
        
        Args:
            project_path (str): Path to the project file
        """
        recent = self.config['app']['recent_projects']
        # Remove if already exists
        if project_path in recent:
            recent.remove(project_path)
            
        # Add to the beginning of the list
        recent.insert(0, project_path)
        
        # Trim list to max size
        max_size = self.config['app']['max_recent_projects']
        if len(recent) > max_size:
            recent = recent[:max_size]
            
        self.config['app']['recent_projects'] = recent
        self.save_config()
    
    def _deep_update(self, original, update):
        """
        Recursively update a dictionary.
        
        Args:
            original (dict): Original dictionary to update
            update (dict): Dictionary with updated values
        """
        for key, value in update.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                self._deep_update(original[key], value)
            else:
                original[key] = value


# Create a singleton instance
_config_manager = None

def load_config() -> Dict[str, Any]:
    """
    Load the application configuration.
    
    Returns:
        Dict containing the application configuration
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    return _config_manager.load_config()

def get_config_manager() -> ConfigManager:
    """
    Get the ConfigManager instance.
    
    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    return _config_manager

def get_value(path, default=None):
    """
    Get a configuration value.
    
    Args:
        path (str): Dot-separated path to the configuration value
        default: Value to return if path doesn't exist
        
    Returns:
        Configuration value or default if not found
    """
    return get_config_manager().get_value(path, default)

def set_value(path, value):
    """
    Set a configuration value.
    
    Args:
        path (str): Dot-separated path to the configuration value
        value: Value to set
        
    Returns:
        True if successful, False otherwise
    """
    return get_config_manager().set_value(path, value)

def save_config():
    """
    Save the current configuration.
    
    Returns:
        True if successful, False otherwise
    """
    return get_config_manager().save_config()