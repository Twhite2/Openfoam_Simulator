#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenFOAM case management for Openfoam_Simulator.

This module provides functionality for creating, managing, and manipulating
OpenFOAM cases, including:
- Creating case directory structures
- Setting up boundary conditions
- Configuring case dictionaries
- Managing case templates
- Handling case operations (copy, move, etc.)
- Interfacing with OpenFOAM solvers
"""

import os
import sys
import shutil
import subprocess
import logging
import datetime
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from PyQt5.QtWidgets import QMessageBox

# Import relevant project modules
from ..utils.logger import get_logger
from ..config import get_value, set_value

# Setup logger
logger = get_logger(__name__)


class CaseDirectoryManager:
    """
    Manages OpenFOAM case directory structure and operations.
    
    This class handles the creation and manipulation of the standard OpenFOAM
    case directory structure (0, constant, system) and provides utilities
    for copying, moving, and cleaning case directories.
    """
    
    def __init__(self, case_dir: str):
        """
        Initialize the case directory manager.
        
        Args:
            case_dir (str): Path to the case directory
        """
        self.case_dir = Path(case_dir)
        self.time_dirs: List[str] = []
        self.latest_time: str = "0"
    
    def create_case_structure(self) -> bool:
        """
        Create the standard OpenFOAM case directory structure.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create main case directory if it doesn't exist
            self.case_dir.mkdir(parents=True, exist_ok=True)
            
            # Create standard OpenFOAM directories
            (self.case_dir / "0").mkdir(exist_ok=True)
            (self.case_dir / "constant").mkdir(exist_ok=True)
            (self.case_dir / "system").mkdir(exist_ok=True)
            
            # Create additional subdirectories
            (self.case_dir / "constant" / "polyMesh").mkdir(exist_ok=True)
            (self.case_dir / "constant" / "triSurface").mkdir(exist_ok=True)
            
            # Create log directory
            (self.case_dir / "logs").mkdir(exist_ok=True)
            
            logger.info(f"Created case structure in {self.case_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating case structure: {e}")
            return False
    
    def clean_case(self, preserve_mesh: bool = True, preserve_0_dir: bool = True) -> bool:
        """
        Clean the case by removing solution time directories.
        
        Args:
            preserve_mesh (bool): Whether to preserve mesh files
            preserve_0_dir (bool): Whether to preserve the 0 directory
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get all time directories (directories named with numbers)
            time_pattern = re.compile(r'^\d+(\.\d+)?$')
            
            for item in self.case_dir.iterdir():
                if item.is_dir() and time_pattern.match(item.name):
                    # Skip 0 directory if requested
                    if item.name == "0" and preserve_0_dir:
                        continue
                    
                    # Remove time directory
                    shutil.rmtree(item)
                    logger.debug(f"Removed time directory: {item}")
            
            # Remove processor directories if parallel case
            for item in self.case_dir.iterdir():
                if item.is_dir() and item.name.startswith("processor"):
                    if preserve_mesh:
                        # Remove only time directories in processor directories
                        for subitem in item.iterdir():
                            if subitem.is_dir() and time_pattern.match(subitem.name):
                                if subitem.name == "0" and preserve_0_dir:
                                    continue
                                shutil.rmtree(subitem)
                                logger.debug(f"Removed time directory: {subitem}")
                    else:
                        # Remove entire processor directory
                        shutil.rmtree(item)
                        logger.debug(f"Removed processor directory: {item}")
            
            # Remove log files
            for log_file in (self.case_dir / "logs").glob("*.log"):
                log_file.unlink()
                logger.debug(f"Removed log file: {log_file}")
            
            # Remove other temporary files
            for temp_file in self.case_dir.glob("*.foam"):
                temp_file.unlink()
            
            # Remove decomposition files if not preserving mesh
            if not preserve_mesh:
                decomp_file = self.case_dir / "system" / "decomposeParDict"
                if decomp_file.exists():
                    decomp_file.unlink()
            
            logger.info(f"Cleaned case directory: {self.case_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning case: {e}")
            return False
    
    def find_time_directories(self) -> List[str]:
        """
        Find and sort all time directories in the case.
        
        Returns:
            List[str]: Sorted list of time directory names
        """
        time_dirs = []
        time_pattern = re.compile(r'^\d+(\.\d+)?$')
        
        try:
            for item in self.case_dir.iterdir():
                if item.is_dir() and time_pattern.match(item.name):
                    time_dirs.append(item.name)
            
            # Sort numerically
            time_dirs.sort(key=float)
            self.time_dirs = time_dirs
            
            # Update latest time
            if time_dirs:
                self.latest_time = time_dirs[-1]
            else:
                self.latest_time = "0"
            
            return time_dirs
            
        except Exception as e:
            logger.error(f"Error finding time directories: {e}")
            return []
    
    def copy_time_directory(self, src_time: str, dst_time: str) -> bool:
        """
        Copy a time directory to a new time.
        
        Args:
            src_time (str): Source time directory name
            dst_time (str): Destination time directory name
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            src_dir = self.case_dir / src_time
            dst_dir = self.case_dir / dst_time
            
            if not src_dir.exists():
                logger.error(f"Source time directory does not exist: {src_dir}")
                return False
            
            if dst_dir.exists():
                logger.warning(f"Destination time directory already exists: {dst_dir}")
                return False
            
            # Copy directory
            shutil.copytree(src_dir, dst_dir)
            logger.info(f"Copied time directory: {src_time} to {dst_time}")
            
            # Update time dirs
            self.find_time_directories()
            return True
            
        except Exception as e:
            logger.error(f"Error copying time directory: {e}")
            return False
    
    def get_latest_time(self) -> str:
        """
        Get the latest time directory.
        
        Returns:
            str: Name of the latest time directory
        """
        # Make sure time directories are up to date
        self.find_time_directories()
        return self.latest_time
    
    def check_case_validity(self) -> Tuple[bool, str]:
        """
        Check if the case directory has a valid OpenFOAM structure.
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Check for basic directory structure
        required_dirs = ["0", "system", "constant"]
        for dir_name in required_dirs:
            if not (self.case_dir / dir_name).exists():
                return False, f"Missing required directory: {dir_name}"
        
        # Check for essential system files
        system_files = ["controlDict", "fvSchemes", "fvSolution"]
        for file_name in system_files:
            if not (self.case_dir / "system" / file_name).exists():
                return False, f"Missing required system file: {file_name}"
        
        # Check for mesh
        mesh_present = False
        if (self.case_dir / "constant" / "polyMesh").exists():
            mesh_files = ["points", "faces", "owner", "neighbour"]
            mesh_present = all((self.case_dir / "constant" / "polyMesh" / f).exists() for f in mesh_files)
        
        if not mesh_present:
            return True, "Case is valid but has no mesh"
        
        return True, "Case is valid"


class CaseDictManager:
    """
    Manages OpenFOAM dictionary files in a case.
    
    This class provides functionality for reading, writing, and manipulating
    OpenFOAM dictionary files, including common dictionaries such as
    controlDict, fvSchemes, fvSolution, and boundary condition files.
    """
    
    def __init__(self, case_dir: str):
        """
        Initialize the case dictionary manager.
        
        Args:
            case_dir (str): Path to the case directory
        """
        self.case_dir = Path(case_dir)
    
    def read_dict_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Read an OpenFOAM dictionary file and parse its contents.
        
        Args:
            file_path (Union[str, Path]): Path to the dictionary file
            
        Returns:
            Dict[str, Any]: Parsed dictionary contents
        """
        # This is a simplistic parser for demonstration
        # A full implementation would need a proper OpenFOAM dictionary parser
        
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = self.case_dir / file_path
        
        if not file_path.exists():
            logger.error(f"Dictionary file does not exist: {file_path}")
            return {}
        
        try:
            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Remove comments and preprocess
            content = self._preprocess_dict_content(content)
            
            # Parse dictionary (simplified)
            dict_data = self._parse_dict_content(content)
            
            return dict_data
            
        except Exception as e:
            logger.error(f"Error reading dictionary file {file_path}: {e}")
            return {}
    
    def write_dict_file(self, file_path: Union[str, Path], dict_data: Dict[str, Any], 
                       header: Optional[str] = None) -> bool:
        """
        Write dictionary data to an OpenFOAM dictionary file.
        
        Args:
            file_path (Union[str, Path]): Path to the dictionary file
            dict_data (Dict[str, Any]): Dictionary data to write
            header (Optional[str]): Optional header text for the file
            
        Returns:
            bool: True if successful, False otherwise
        """
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = self.case_dir / file_path
        
        try:
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate default header if not provided
            if header is None:
                header = self._generate_default_header(file_path.name)
            
            # Convert dictionary to OpenFOAM format
            dict_content = self._format_dict_content(dict_data)
            
            # Write file
            with open(file_path, 'w') as f:
                f.write(header)
                f.write(dict_content)
            
            logger.info(f"Wrote dictionary file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing dictionary file {file_path}: {e}")
            return False
    
    def _preprocess_dict_content(self, content: str) -> str:
        """
        Preprocess OpenFOAM dictionary content by removing comments.
        
        Args:
            content (str): Raw dictionary content
            
        Returns:
            str: Preprocessed content
        """
        # Remove C-style comments (/* ... */)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Remove C++-style comments (// ...)
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        
        return content
    
    def _parse_dict_content(self, content: str) -> Dict[str, Any]:
        """
        Parse OpenFOAM dictionary content into a Python dictionary.
        
        Args:
            content (str): Preprocessed dictionary content
            
        Returns:
            Dict[str, Any]: Parsed dictionary
        """
        # NOTE: This is a simplified parser for demonstration purposes
        # A production implementation would need a more robust parser
        
        result = {}
        
        # Simple key-value pattern
        kv_pattern = re.compile(r'(\w+)\s+([^;]+);')
        
        # Find all simple key-value pairs
        for match in kv_pattern.finditer(content):
            key = match.group(1)
            value = match.group(2).strip()
            
            # Try to convert value to appropriate type
            if value.isdigit():
                value = int(value)
            elif re.match(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$', value):
                value = float(value)
            elif value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            
            result[key] = value
        
        # TODO: Handle nested dictionaries, lists, etc.
        
        return result
    
    def _format_dict_content(self, dict_data: Dict[str, Any], indent: int = 0) -> str:
        """
        Format dictionary data as OpenFOAM dictionary content.
        
        Args:
            dict_data (Dict[str, Any]): Dictionary data
            indent (int): Indentation level
            
        Returns:
            str: Formatted dictionary content
        """
        content = ""
        indent_str = " " * indent
        
        for key, value in dict_data.items():
            if isinstance(value, dict):
                # Nested dictionary
                content += f"{indent_str}{key}\n{indent_str}{{\n"
                content += self._format_dict_content(value, indent + 4)
                content += f"{indent_str}}}\n\n"
            elif isinstance(value, list):
                # List
                content += f"{indent_str}{key}\n{indent_str}(\n"
                for item in value:
                    if isinstance(item, dict):
                        content += f"{indent_str}    {{\n"
                        content += self._format_dict_content(item, indent + 8)
                        content += f"{indent_str}    }}\n"
                    else:
                        content += f"{indent_str}    {item}\n"
                content += f"{indent_str});\n\n"
            else:
                # Simple value
                content += f"{indent_str}{key}        {value};\n"
        
        return content
    
    def _generate_default_header(self, file_name: str) -> str:
        """
        Generate a default header for OpenFOAM dictionary files.
        
        Args:
            file_name (str): Name of the file
            
        Returns:
            str: Default header text
        """
        header = f"""/*--------------------------------*- C++ -*----------------------------------*\\
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
    object      {file_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""
        return header
    
    def create_control_dict(self, solver: str, start_time: float = 0.0, end_time: float = 100.0, 
                           delta_t: float = 0.001, write_interval: float = 10.0, 
                           write_format: str = "ascii") -> bool:
        """
        Create and write a controlDict file.
        
        Args:
            solver (str): OpenFOAM solver name
            start_time (float): Simulation start time
            end_time (float): Simulation end time
            delta_t (float): Time step
            write_interval (float): Interval for writing results
            write_format (str): Format for writing results
            
        Returns:
            bool: True if successful, False otherwise
        """
        control_dict = {
            "application": solver,
            "startFrom": "startTime",
            "startTime": start_time,
            "stopAt": "endTime",
            "endTime": end_time,
            "deltaT": delta_t,
            "writeControl": "runTime",
            "writeInterval": write_interval,
            "purgeWrite": 0,
            "writeFormat": write_format,
            "writePrecision": 6,
            "writeCompression": "off",
            "timeFormat": "general",
            "timePrecision": 6,
            "runTimeModifiable": True,
            "functions": {}
        }
        
        return self.write_dict_file("system/controlDict", control_dict)
    
    def create_fv_schemes(self, scheme_type: str = "standard") -> bool:
        """
        Create and write an fvSchemes file.
        
        Args:
            scheme_type (str): Type of schemes to use (e.g., standard, upwind)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Define different scheme sets
        scheme_sets = {
            "standard": {
                "ddtSchemes": {
                    "default": "Euler"
                },
                "gradSchemes": {
                    "default": "Gauss linear",
                    "grad(p)": "Gauss linear"
                },
                "divSchemes": {
                    "default": "none",
                    "div(phi,U)": "Gauss limitedLinearV 1",
                    "div(phi,k)": "Gauss limitedLinear 1",
                    "div(phi,epsilon)": "Gauss limitedLinear 1",
                    "div(phi,omega)": "Gauss limitedLinear 1",
                    "div(phi,R)": "Gauss limitedLinear 1",
                    "div(R)": "Gauss linear",
                    "div((nuEff*dev2(T(grad(U)))))": "Gauss linear"
                },
                "laplacianSchemes": {
                    "default": "Gauss linear corrected"
                },
                "interpolationSchemes": {
                    "default": "linear"
                },
                "snGradSchemes": {
                    "default": "corrected"
                }
            },
            "upwind": {
                "ddtSchemes": {
                    "default": "Euler"
                },
                "gradSchemes": {
                    "default": "Gauss linear"
                },
                "divSchemes": {
                    "default": "none",
                    "div(phi,U)": "Gauss upwind",
                    "div(phi,k)": "Gauss upwind",
                    "div(phi,epsilon)": "Gauss upwind",
                    "div(phi,omega)": "Gauss upwind",
                    "div(phi,R)": "Gauss upwind",
                    "div(R)": "Gauss linear",
                    "div((nuEff*dev2(T(grad(U)))))": "Gauss linear"
                },
                "laplacianSchemes": {
                    "default": "Gauss linear corrected"
                },
                "interpolationSchemes": {
                    "default": "linear"
                },
                "snGradSchemes": {
                    "default": "corrected"
                }
            }
        }
        
        # Use standard schemes if requested type not found
        selected_schemes = scheme_sets.get(scheme_type, scheme_sets["standard"])
        
        return self.write_dict_file("system/fvSchemes", selected_schemes)
    
    def create_fv_solution(self, solver_type: str = "SIMPLE") -> bool:
        """
        Create and write an fvSolution file.
        
        Args:
            solver_type (str): Type of solver to configure
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Define different solver configurations
        solver_configs = {
            "SIMPLE": {
                "solvers": {
                    "p": {
                        "solver": "GAMG",
                        "tolerance": 1e-6,
                        "relTol": 0.01,
                        "smoother": "GaussSeidel",
                        "nPreSweeps": 0,
                        "nPostSweeps": 2,
                        "cacheAgglomeration": True,
                        "nCellsInCoarsestLevel": 10,
                        "agglomerator": "faceAreaPair",
                        "mergeLevels": 1
                    },
                    "(U|k|epsilon|omega|R)": {
                        "solver": "smoothSolver",
                        "smoother": "GaussSeidel",
                        "tolerance": 1e-5,
                        "relTol": 0.1,
                        "nSweeps": 1
                    }
                },
                "SIMPLE": {
                    "nNonOrthogonalCorrectors": 0,
                    "consistent": True,
                    "residualControl": {
                        "p": 1e-4,
                        "U": 1e-4,
                        "(k|epsilon|omega|R)": 1e-4
                    }
                },
                "relaxationFactors": {
                    "equations": {
                        "U": 0.7,
                        "k": 0.7,
                        "epsilon": 0.7,
                        "omega": 0.7,
                        "R": 0.7
                    },
                    "fields": {
                        "p": 0.3
                    }
                }
            },
            "PISO": {
                "solvers": {
                    "p": {
                        "solver": "PCG",
                        "preconditioner": "DIC",
                        "tolerance": 1e-6,
                        "relTol": 0.01
                    },
                    "(U|k|epsilon|omega|R)": {
                        "solver": "smoothSolver",
                        "smoother": "symGaussSeidel",
                        "tolerance": 1e-5,
                        "relTol": 0.1
                    }
                },
                "PISO": {
                    "nCorrectors": 2,
                    "nNonOrthogonalCorrectors": 1,
                    "pRefCell": 0,
                    "pRefValue": 0
                }
            },
            "PIMPLE": {
                "solvers": {
                    "p": {
                        "solver": "GAMG",
                        "tolerance": 1e-6,
                        "relTol": 0.01,
                        "smoother": "GaussSeidel"
                    },
                    "(U|k|epsilon|omega|R)": {
                        "solver": "smoothSolver",
                        "smoother": "symGaussSeidel",
                        "tolerance": 1e-5,
                        "relTol": 0.1
                    }
                },
                "PIMPLE": {
                    "nOuterCorrectors": 1,
                    "nCorrectors": 2,
                    "nNonOrthogonalCorrectors": 1,
                    "pRefCell": 0,
                    "pRefValue": 0
                },
                "relaxationFactors": {
                    "equations": {
                        "U": 0.9,
                        "k": 0.9,
                        "epsilon": 0.9,
                        "omega": 0.9,
                        "R": 0.9
                    }
                }
            }
        }
        
        # Use SIMPLE if requested type not found
        selected_config = solver_configs.get(solver_type, solver_configs["SIMPLE"])
        
        return self.write_dict_file("system/fvSolution", selected_config)
    
    def create_transport_properties(self, fluid_type: str = "water") -> bool:
        """
        Create and write a transportProperties file.
        
        Args:
            fluid_type (str): Type of fluid
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Define fluid properties
        fluid_properties = {
            "water": {
                "transportModel": "Newtonian",
                "nu": 1e-6,  # kinematic viscosity [m^2/s]
                "rho": 1000,  # density [kg/m^3]
            },
            "oil": {
                "transportModel": "Newtonian",
                "nu": 3e-5,  # kinematic viscosity [m^2/s]
                "rho": 850,  # density [kg/m^3]
            },
            "air": {
                "transportModel": "Newtonian",
                "nu": 1.5e-5,  # kinematic viscosity [m^2/s]
                "rho": 1.2,  # density [kg/m^3]
            },
            "multiPhase": {
                "phases": {
                    "water": {
                        "transportModel": "Newtonian",
                        "nu": 1e-6,
                        "rho": 1000
                    },
                    "oil": {
                        "transportModel": "Newtonian",
                        "nu": 3e-5,
                        "rho": 850
                    }
                },
                "sigma": 0.07  # surface tension [N/m]
            }
        }
        
        # Use water if requested type not found
        if fluid_type == "multiPhase":
            transport_props = {
                "phases": [
                    {
                        "name": "water",
                        "transportModel": "Newtonian",
                        "nu": 1e-6,
                        "rho": 1000
                    },
                    {
                        "name": "oil",
                        "transportModel": "Newtonian",
                        "nu": 3e-5,
                        "rho": 850
                    }
                ],
                "sigma": 0.07
            }
        else:
            selected_props = fluid_properties.get(fluid_type, fluid_properties["water"])
            transport_props = selected_props
        
        return self.write_dict_file("constant/transportProperties", transport_props)

    def create_velocity_inlet_bc(self, patch_name: str, time_dir: str = "0", 
                                velocity: List[float] = [0, 0, 0]) -> bool:
        """
        Create velocity inlet boundary condition.
        
        Args:
            patch_name (str): Name of the boundary patch
            time_dir (str): Time directory
            velocity (List[float]): Velocity vector [u, v, w]
            
        Returns:
            bool: True if successful, False otherwise
        """
        # First, check if U file exists and load it
        u_file = self.case_dir / time_dir / "U"
        
        if u_file.exists():
            u_dict = self.read_dict_file(u_file)
        else:
            # Create new U file
            u_dict = {
                "dimensions": "[0 1 -1 0 0 0 0]",
                "internalField": "uniform (0 0 0)",
                "boundaryField": {}
            }
        
        # Ensure boundaryField exists
        if "boundaryField" not in u_dict:
            u_dict["boundaryField"] = {}
        
        # Set boundary condition for the patch
        u_dict["boundaryField"][patch_name] = {
            "type": "fixedValue",
            "value": f"uniform ({velocity[0]} {velocity[1]} {velocity[2]})"
        }
        
        return self.write_dict_file(f"{time_dir}/U", u_dict)
    
    def create_pressure_outlet_bc(self, patch_name: str, time_dir: str = "0", 
                                 pressure: float = 0.0) -> bool:
        """
        Create pressure outlet boundary condition.
        
        Args:
            patch_name (str): Name of the boundary patch
            time_dir (str): Time directory
            pressure (float): Outlet pressure value
            
        Returns:
            bool: True if successful, False otherwise
        """
        # First, check if p file exists and load it
        p_file = self.case_dir / time_dir / "p"
        
        if p_file.exists():
            p_dict = self.read_dict_file(p_file)
        else:
            # Create new p file
            p_dict = {
                "dimensions": "[0 2 -2 0 0 0 0]",
                "internalField": "uniform 0",
                "boundaryField": {}
            }
        
        # Ensure boundaryField exists
        if "boundaryField" not in p_dict:
            p_dict["boundaryField"] = {}
        
        # Set boundary condition for the patch
        p_dict["boundaryField"][patch_name] = {
            "type": "fixedValue",
            "value": f"uniform {pressure}"
        }
        
        return self.write_dict_file(f"{time_dir}/p", p_dict)
    
    def create_wall_bc(self, patch_name: str, time_dir: str = "0", 
                      no_slip: bool = True) -> bool:
        """
        Create wall boundary condition.
        
        Args:
            patch_name (str): Name of the boundary patch
            time_dir (str): Time directory
            no_slip (bool): Whether to use no-slip condition
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Update U file for velocity boundary condition
        u_file = self.case_dir / time_dir / "U"
        
        if u_file.exists():
            u_dict = self.read_dict_file(u_file)
        else:
            # Create new U file
            u_dict = {
                "dimensions": "[0 1 -1 0 0 0 0]",
                "internalField": "uniform (0 0 0)",
                "boundaryField": {}
            }
        
        # Ensure boundaryField exists
        if "boundaryField" not in u_dict:
            u_dict["boundaryField"] = {}
        
        # Set boundary condition for the patch
        if no_slip:
            u_dict["boundaryField"][patch_name] = {
                "type": "noSlip"
            }
        else:
            # Slip wall (zero gradient)
            u_dict["boundaryField"][patch_name] = {
                "type": "slip"
            }
        
        # Update U file
        if not self.write_dict_file(f"{time_dir}/U", u_dict):
            return False
        
        # Update p file for pressure boundary condition
        p_file = self.case_dir / time_dir / "p"
        
        if p_file.exists():
            p_dict = self.read_dict_file(p_file)
        else:
            # Create new p file
            p_dict = {
                "dimensions": "[0 2 -2 0 0 0 0]",
                "internalField": "uniform 0",
                "boundaryField": {}
            }
        
        # Ensure boundaryField exists
        if "boundaryField" not in p_dict:
            p_dict["boundaryField"] = {}
        
        # Set boundary condition for the patch
        p_dict["boundaryField"][patch_name] = {
            "type": "zeroGradient"
        }
        
        # Update p file
        return self.write_dict_file(f"{time_dir}/p", p_dict)
    
    def get_boundary_names(self) -> List[str]:
        """
        Get list of boundary patch names from the mesh.
        
        Returns:
            List[str]: List of boundary patch names
        """
        boundary_file = self.case_dir / "constant" / "polyMesh" / "boundary"
        
        if not boundary_file.exists():
            logger.error("Boundary file does not exist")
            return []
        
        try:
            # Read boundary file
            with open(boundary_file, 'r') as f:
                content = f.read()
            
            # Parse boundary names (simplified)
            boundaries = []
            pattern = re.compile(r'(\w+)\s*\n\s*\{[^{]*?type\s+(\w+);', re.DOTALL)
            
            for match in pattern.finditer(content):
                boundary_name = match.group(1)
                boundary_type = match.group(2)
                
                # Rename auto_walls to walls
                if boundary_name.startswith("auto_walls"):
                    boundary_name = "walls"
                
                boundaries.append(boundary_name)
            
            return boundaries
            
        except Exception as e:
            logger.error(f"Error reading boundary file: {e}")
            return []


class CaseTemplate:
    """
    Manages and applies predefined case templates.
    
    This class provides functionality for applying predefined case templates
    for common oil & gas simulation scenarios.
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize the case template manager.
        
        Args:
            template_dir (Optional[str]): Directory containing templates
        """
        if template_dir:
            self.template_dir = Path(template_dir)
        else:
            # Use default templates directory from config
            self.template_dir = Path(get_value('paths.templates', '')) / "case_templates"
    
    def list_templates(self) -> Dict[str, str]:
        """
        List available templates with descriptions.
        
        Returns:
            Dict[str, str]: Dictionary of template names and descriptions
        """
        templates = {}
        
        try:
            # Look for template directories
            for item in self.template_dir.iterdir():
                if item.is_dir():
                    # Check for description file
                    desc_file = item / "description.txt"
                    if desc_file.exists():
                        with open(desc_file, 'r') as f:
                            description = f.read().strip()
                    else:
                        description = "No description available"
                    
                    templates[item.name] = description
            
            return templates
            
        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            return {}
    
    def apply_template(self, template_name: str, case_dir: str) -> bool:
        """
        Apply a template to a case directory.
        
        Args:
            template_name (str): Name of the template to apply
            case_dir (str): Path to the case directory
            
        Returns:
            bool: True if successful, False otherwise
        """
        template_path = self.template_dir / template_name
        case_path = Path(case_dir)
        
        if not template_path.exists():
            logger.error(f"Template does not exist: {template_path}")
            return False
        
        try:
            # Create case directory if it doesn't exist
            case_path.mkdir(parents=True, exist_ok=True)
            
            # Copy template files to case directory
            self._copy_template_files(template_path, case_path)
            
            logger.info(f"Applied template '{template_name}' to {case_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying template: {e}")
            return False
    
    def _copy_template_files(self, src_dir: Path, dst_dir: Path):
        """
        Copy template files to the case directory.
        
        Args:
            src_dir (Path): Source template directory
            dst_dir (Path): Destination case directory
        """
        # Skip description file and other template metadata
        skip_files = {"description.txt", "template.json"}
        
        for item in src_dir.glob("**/*"):
            # Skip directories (they'll be created as needed)
            if item.is_dir():
                continue
            
            # Skip template metadata files
            if item.name in skip_files:
                continue
            
            # Determine relative path
            rel_path = item.relative_to(src_dir)
            dst_path = dst_dir / rel_path
            
            # Create parent directories if they don't exist
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(item, dst_path)
            logger.debug(f"Copied template file: {rel_path}")
    
    def create_custom_template(self, case_dir: str, template_name: str, 
                             description: str = "") -> bool:
        """
        Create a custom template from an existing case.
        
        Args:
            case_dir (str): Path to the source case directory
            template_name (str): Name for the new template
            description (str): Description of the template
            
        Returns:
            bool: True if successful, False otherwise
        """
        case_path = Path(case_dir)
        template_path = self.template_dir / template_name
        
        if not case_path.exists():
            logger.error(f"Case directory does not exist: {case_path}")
            return False
        
        if template_path.exists():
            logger.error(f"Template already exists: {template_path}")
            return False
        
        try:
            # Create template directory
            template_path.mkdir(parents=True, exist_ok=True)
            
            # Copy case files to template directory
            self._copy_case_to_template(case_path, template_path)
            
            # Create description file
            with open(template_path / "description.txt", 'w') as f:
                f.write(description)
            
            # Create template metadata
            metadata = {
                "name": template_name,
                "description": description,
                "created": datetime.datetime.now().isoformat(),
                "source_case": str(case_path)
            }
            
            with open(template_path / "template.json", 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Created template '{template_name}' from {case_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating template: {e}")
            return False
    
    def _copy_case_to_template(self, src_dir: Path, dst_dir: Path):
        """
        Copy case files to the template directory, excluding result files.
        
        Args:
            src_dir (Path): Source case directory
            dst_dir (Path): Destination template directory
        """
        # Skip time directories except 0, processor directories, and logs
        skip_dirs = {"processor", "logs"}
        time_pattern = re.compile(r'^\d+(\.\d+)?$')
        
        # Copy system, constant, and 0 directories
        for base_dir in ["system", "constant", "0"]:
            src_path = src_dir / base_dir
            dst_path = dst_dir / base_dir
            
            if src_path.exists():
                dst_path.mkdir(parents=True, exist_ok=True)
                
                # Copy all files in the directory
                for item in src_path.glob("**/*"):
                    if item.is_dir():
                        continue
                    
                    rel_path = item.relative_to(src_path)
                    dst_file = dst_path / rel_path
                    
                    # Create parent directories if they don't exist
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(item, dst_file)
                    logger.debug(f"Copied case file: {base_dir}/{rel_path}")


class CaseManager:
    """
    Main class for managing OpenFOAM cases.
    
    This class combines the functionality of directory, dictionary, and template
    management to provide a comprehensive interface for managing OpenFOAM cases.
    """
    
    def __init__(self, main_window=None, case_dir=None):
        """
        Initialize the case manager.
        
        Args:
            main_window: Main window instance
            case_dir: Path to the case directory (optional)
        """
        self.main_window = main_window
        self.case_directory = None
        
        # Initialize attributes
        self.is_running = False
        self.solver_process = None
        
        # Set case directory if provided
        if case_dir:
            self.set_case_directory(case_dir)
    
    def set_case_directory(self, directory):
        """
        Set the case directory.
        
        Args:
            directory: Path to the case directory
            
        Returns:
            bool: True if directory was set successfully
        """
        if directory:
            self.case_directory = Path(directory)
            logger.info(f"Case directory set to: {self.case_directory}")
            return True
        return False
    
    def setup_case(self, directory=None):
        """
        Set up a basic OpenFOAM case structure.
        
        Args:
            directory: Optional directory to set before setup
            
        Returns:
            bool: True if case setup was successful
        """
        try:
            if directory:
                self.set_case_directory(directory)
                
            if not self.case_directory:
                logger.error("Cannot set up case: No case directory specified")
                return False
                
            # Create basic directory structure
            system_dir = self.case_directory / "system"
            constant_dir = self.case_directory / "constant"
            zero_dir = self.case_directory / "0"
            
            system_dir.mkdir(exist_ok=True)
            constant_dir.mkdir(exist_ok=True)
            zero_dir.mkdir(exist_ok=True)
            
            # Create minimal required files
            self._create_control_dict(system_dir)
            self._create_fv_schemes(system_dir)
            self._create_fv_solution(system_dir)
            
            # Create basic transport properties
            self._create_transport_properties(constant_dir)
            
            # Rename auto_walls to walls if a mesh exists
            if (self.case_directory / "constant" / "polyMesh" / "boundary").exists():
                self.rename_auto_walls()
            
            logger.info(f"Successfully set up case in {self.case_directory}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up case: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def set_boundary_condition(self, boundary_name, boundary_type, config):
        """
        Set boundary condition for OpenFOAM case.
        
        Args:
            boundary_name: Name of the boundary
            boundary_type: Type of the boundary (inlet, outlet, wall, etc.)
            config: Configuration dictionary with settings
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.case_directory:
                logger.error("Cannot set boundary condition: No case directory specified")
                return False
                
            # Log the boundary condition being set
            logger.info(f"Setting boundary condition for {boundary_name} as {boundary_type}: {config}")
            
            # Create '0' directory if it doesn't exist
            zero_dir = self.case_directory / "0"
            zero_dir.mkdir(exist_ok=True)
            
            # Update velocity boundary conditions
            velocity_file = zero_dir / "U"
            self._update_velocity_file(velocity_file, boundary_name, boundary_type, config)
            
            # Update pressure boundary conditions
            pressure_file = zero_dir / "p"
            self._update_pressure_file(pressure_file, boundary_name, boundary_type, config)
            
            # Update temperature if needed
            if "temperature_type" in config:
                temp_file = zero_dir / "T"
                self._update_temperature_file(temp_file, boundary_name, boundary_type, config)
            
            return True
                
        except Exception as e:
            logger.error(f"Error setting boundary condition: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        
    def _update_velocity_file(self, file_path, boundary_name, boundary_type, config):
        """Update the velocity boundary condition in the U file."""
        try:
            # If file doesn't exist, create it with default template
            if not file_path.exists():
                self._create_default_velocity_file(file_path)
            
            # Read the existing file
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find the boundary section
            boundary_start = -1
            boundary_end = -1
            in_boundary_section = False
            for i, line in enumerate(lines):
                if "boundaryField" in line:
                    in_boundary_section = True
                    continue
                
                if in_boundary_section and "{" in line:
                    in_boundary_section = False
                    continue
                    
                if boundary_name in line and not in_boundary_section:
                    boundary_start = i
                    # Find the closing brace
                    brace_count = 0
                    for j in range(i, len(lines)):
                        if "{" in lines[j]:
                            brace_count += 1
                        if "}" in lines[j]:
                            brace_count -= 1
                            if brace_count == 0:
                                boundary_end = j
                                break
                    break
            
            # If boundary found, replace it, otherwise append it
            velocity_type = config.get("velocity_type", "fixedValue")
            
            if boundary_type == "inlet":
                if velocity_type == "fixedValue":
                    new_boundary = f"""    {boundary_name}
    {{
        type            {velocity_type};
        value           uniform ({config.get("velocity_x", 0)} {config.get("velocity_y", 0)} {config.get("velocity_z", 0)});
    }}
"""
                elif velocity_type == "flowRate":
                    new_boundary = f"""    {boundary_name}
    {{
        type            flowRateInletVelocity;
        volumetricFlowRate {config.get("flow_rate", 0.001)};
        value           uniform ({config.get("velocity_x", 0)} {config.get("velocity_y", 0)} {config.get("velocity_z", 0)});
    }}
"""
                else:
                    new_boundary = f"""    {boundary_name}
    {{
        type            {velocity_type};
        value           uniform (0 0 0);
    }}
"""
            elif boundary_type == "outlet":
                new_boundary = f"""    {boundary_name}
    {{
        type            zeroGradient;
    }}
"""
            elif boundary_type == "wall":
                wall_type = config.get("velocity_type", "noSlip")
                if wall_type == "movingWall":
                    new_boundary = f"""    {boundary_name}
    {{
        type            movingWallVelocity;
        value           uniform ({config.get("velocity_x", 0)} {config.get("velocity_y", 0)} {config.get("velocity_z", 0)});
    }}
"""
                elif wall_type == "slip":
                    new_boundary = f"""    {boundary_name}
    {{
        type            slip;
    }}
"""
                else:  # noSlip is default
                    new_boundary = f"""    {boundary_name}
    {{
        type            noSlip;
    }}
"""
            elif boundary_type == "symmetryPlane":
                new_boundary = f"""    {boundary_name}
    {{
        type            symmetryPlane;
    }}
"""
            else:  # empty or other types
                new_boundary = f"""    {boundary_name}
    {{
        type            {boundary_type};
    }}
"""
            
            # Update the file
            if boundary_start >= 0 and boundary_end >= 0:
                # Replace existing boundary
                new_lines = lines[:boundary_start] + [new_boundary] + lines[boundary_end+1:]
            else:
                # Find where to add the new boundary (before the final closing brace)
                for i in range(len(lines) - 1, -1, -1):
                    if "}" in lines[i] and not any(c in lines[i] for c in "({["):
                        new_lines = lines[:i] + [new_boundary] + lines[i:]
                        break
                else:
                    # If no proper place found, just append
                    new_lines = lines + [new_boundary]
            
            # Write the updated file
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating velocity file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        
    def _create_default_velocity_file(self, file_path):
        """Create a default velocity (U) file."""
        with open(file_path, 'w') as f:
            f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2212                                 |
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
    // Default boundary condition for all boundaries
    ".*"
    {
        type            noSlip;
    }
}

// ************************************************************************* //
""")
    
    def _create_control_dict(self, system_dir):
        """Create a basic controlDict file."""
        control_dict = system_dir / "controlDict"
        with open(control_dict, 'w') as f:
            f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2212                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     simpleFoam;

startFrom       startTime;

startTime       0;

stopAt          endTime;

endTime         1000;

deltaT          1;

writeControl    timeStep;

writeInterval   100;

purgeWrite      0;

writeFormat     ascii;

writePrecision  6;

writeCompression off;

timeFormat      general;

timePrecision   6;

runTimeModifiable true;

// ************************************************************************* //
""")
    
    def _create_fv_schemes(self, system_dir):
        """Create a basic fvSchemes file."""
        fv_schemes = system_dir / "fvSchemes"
        with open(fv_schemes, 'w') as f:
            f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2212                                 |
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
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
    grad(U)         Gauss linear;
    grad(p)         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss limitedLinearV 1;
    div(phi,k)      bounded Gauss limitedLinear 1;
    div(phi,epsilon) bounded Gauss limitedLinear 1;
    div(phi,omega)  bounded Gauss limitedLinear 1;
    div(phi,R)      bounded Gauss limitedLinear 1;
    div(R)          Gauss linear;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
    laplacian(nuEff,U) Gauss linear corrected;
    laplacian((1|A(U)),p) Gauss linear corrected;
    laplacian(DkEff,k) Gauss linear corrected;
    laplacian(DepsilonEff,epsilon) Gauss linear corrected;
    laplacian(DomegaEff,omega) Gauss linear corrected;
    laplacian(DREff,R) Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

wallDist
{
    method meshWave;
}

// ************************************************************************* //
""")
    
    def _create_fv_solution(self, system_dir):
        """Create a basic fvSolution file."""
        fv_solution = system_dir / "fvSolution"
        with open(fv_solution, 'w') as f:
            f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2212                                 |
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

    "(U|k|epsilon|omega|R)"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-6;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;

    residualControl
    {
        p               1e-4;
        U               1e-4;
        "(k|epsilon|omega|R)" 1e-4;
    }
}

relaxationFactors
{
    equations
    {
        U               0.9;
        k               0.7;
        epsilon         0.7;
        omega           0.7;
        R               0.7;
    }
}

// ************************************************************************* //
""")
    
    def _create_transport_properties(self, constant_dir):
        """Create a basic transportProperties file."""
        transport_properties = constant_dir / "transportProperties"
        with open(transport_properties, 'w') as f:
            f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2212                                 |
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

transportModel  Newtonian;

nu              [0 2 -1 0 0 0 0] 1e-05;

// ************************************************************************* //
""")

    def _update_pressure_file(self, file_path, boundary_name, boundary_type, config):
        """Update the pressure boundary condition in the p file."""
        try:
            # If file doesn't exist, create it with default template
            if not file_path.exists():
                self._create_default_pressure_file(file_path)
            
            # Read the existing file
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find the boundary section
            boundary_start = -1
            boundary_end = -1
            in_boundary_section = False
            for i, line in enumerate(lines):
                if "boundaryField" in line:
                    in_boundary_section = True
                    continue
                
                if in_boundary_section and "{" in line:
                    in_boundary_section = False
                    continue
                    
                if boundary_name in line and not in_boundary_section:
                    boundary_start = i
                    # Find the closing brace
                    brace_count = 0
                    for j in range(i, len(lines)):
                        if "{" in lines[j]:
                            brace_count += 1
                        if "}" in lines[j]:
                            brace_count -= 1
                            if brace_count == 0:
                                boundary_end = j
                                break
                break
            
            # If boundary found, replace it, otherwise append it
            pressure_type = config.get("pressure_type", "zeroGradient")
            
            if boundary_type == "inlet":
                if pressure_type == "fixedValue":
                    new_boundary = f"""    {boundary_name}
    {{
        type            {pressure_type};
        value           uniform {config.get("pressure_value", 0)};
    }}
"""
                else:  # zeroGradient is default for inlet
                    new_boundary = f"""    {boundary_name}
    {{
        type            zeroGradient;
    }}
"""
            elif boundary_type == "outlet":
                if pressure_type == "fixedValue":
                    new_boundary = f"""    {boundary_name}
    {{
        type            {pressure_type};
        value           uniform {config.get("pressure_value", 0)};
    }}
"""
                else:  # totalPressure for outlet
                    new_boundary = f"""    {boundary_name}
    {{
        type            totalPressure;
        p0              uniform {config.get("pressure_value", 0)};
        gamma           1.4;
        value           uniform {config.get("pressure_value", 0)};
    }}
"""
            elif boundary_type == "wall":
                new_boundary = f"""    {boundary_name}
    {{
        type            zeroGradient;
    }}
"""
            elif boundary_type == "symmetryPlane":
                new_boundary = f"""    {boundary_name}
    {{
        type            symmetryPlane;
    }}
"""
            else:  # empty or other types
                new_boundary = f"""    {boundary_name}
    {{
        type            {boundary_type};
    }}
"""
            
            # Update the file
            if boundary_start >= 0 and boundary_end >= 0:
                # Replace existing boundary
                new_lines = lines[:boundary_start] + [new_boundary] + lines[boundary_end+1:]
            else:
                # Find where to add the new boundary (before the final closing brace)
                for i in range(len(lines) - 1, -1, -1):
                    if "}" in lines[i] and not any(c in lines[i] for c in "({["):
                        new_lines = lines[:i] + [new_boundary] + lines[i:]
                        break
                else:
                    # If no proper place found, just append
                    new_lines = lines + [new_boundary]
            
            # Write the updated file
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating pressure file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _create_default_pressure_file(self, file_path):
        """Create a default pressure (p) file."""
        with open(file_path, 'w') as f:
            f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2212                                 |
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
    // Default boundary condition for all boundaries
    ".*"
    {
        type            zeroGradient;
    }
}

// ************************************************************************* //
""")

    def _update_temperature_file(self, file_path, boundary_name, boundary_type, config):
        """Update the temperature boundary condition in the T file."""
        try:
            # If file doesn't exist, create it with default template
            if not file_path.exists():
                self._create_default_temperature_file(file_path)
            
            # Read the existing file
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find the boundary section
            boundary_start = -1
            boundary_end = -1
            in_boundary_section = False
            for i, line in enumerate(lines):
                if "boundaryField" in line:
                    in_boundary_section = True
                    continue
                
                if in_boundary_section and "{" in line:
                    in_boundary_section = False
                    continue
                    
                if boundary_name in line and not in_boundary_section:
                    boundary_start = i
                    # Find the closing brace
                    brace_count = 0
                    for j in range(i, len(lines)):
                        if "{" in lines[j]:
                            brace_count += 1
                        if "}" in lines[j]:
                            brace_count -= 1
                            if brace_count == 0:
                                boundary_end = j
                                break
                break
            
            # If boundary found, replace it, otherwise append it
            temperature_type = config.get("temperature_type", "fixedValue")
            temperature_value = config.get("temperature_value", 300.0)
            
            if temperature_type == "fixedValue":
                new_boundary = f"""    {boundary_name}
    {{
        type            {temperature_type};
        value           uniform {temperature_value};
    }}
"""
            elif temperature_type == "zeroGradient":
                new_boundary = f"""    {boundary_name}
    {{
        type            zeroGradient;
    }}
"""
            elif temperature_type == "inletOutlet":
                new_boundary = f"""    {boundary_name}
    {{
        type            inletOutlet;
        inletValue      uniform {temperature_value};
        value           uniform {temperature_value};
    }}
"""
            else:  # Default
                new_boundary = f"""    {boundary_name}
    {{
        type            {temperature_type};
        value           uniform {temperature_value};
    }}
"""
            
            # Update the file
            if boundary_start >= 0 and boundary_end >= 0:
                # Replace existing boundary
                new_lines = lines[:boundary_start] + [new_boundary] + lines[boundary_end+1:]
            else:
                # Find where to add the new boundary (before the final closing brace)
                for i in range(len(lines) - 1, -1, -1):
                    if "}" in lines[i] and not any(c in lines[i] for c in "({["):
                        new_lines = lines[:i] + [new_boundary] + lines[i:]
                        break
                else:
                    # If no proper place found, just append
                    new_lines = lines + [new_boundary]
            
            # Write the updated file
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating temperature file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _create_default_temperature_file(self, file_path):
        """Create a default temperature (T) file."""
        with open(file_path, 'w') as f:
            f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2212                                 |
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
    // Default boundary condition for all boundaries
    ".*"
    {
        type            zeroGradient;
    }
}

// ************************************************************************* //
""")

    def remove_boundary(self, boundary_name):
        """
        Remove a boundary condition from the OpenFOAM case.
        
        Args:
            boundary_name: Name of the boundary to remove
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.case_directory:
                logger.error("Cannot remove boundary: No case directory specified")
                return False
            
            logger.info(f"Removing boundary '{boundary_name}' from OpenFOAM case")
            
            # Get the 0 directory
            zero_dir = self.case_directory / "0"
            if not zero_dir.exists():
                logger.warning(f"No 0 directory found, nothing to remove")
                return True
            
            # List of field files to process
            field_files = ["U", "p", "T"]
            
            for field_name in field_files:
                field_file = zero_dir / field_name
                if field_file.exists():
                    # Remove the boundary from the file
                    self._remove_boundary_from_file(field_file, boundary_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing boundary: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        
    def _remove_boundary_from_file(self, file_path, boundary_name):
        """Remove a boundary section from a field file."""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find the boundary section
            boundary_start = -1
            boundary_end = -1
            in_boundary_section = False
            for i, line in enumerate(lines):
                if "boundaryField" in line:
                    in_boundary_section = True
                    continue
                
                if in_boundary_section and "{" in line:
                    in_boundary_section = False
                    continue
                    
                if boundary_name in line and not in_boundary_section:
                    boundary_start = i
                    # Find the closing brace
                    brace_count = 0
                    for j in range(i, len(lines)):
                        if "{" in lines[j]:
                            brace_count += 1
                        if "}" in lines[j]:
                            brace_count -= 1
                            if brace_count == 0:
                                boundary_end = j
                                break
                break
            
            if boundary_start >= 0 and boundary_end >= 0:
                # Remove the boundary section
                new_lines = lines[:boundary_start] + lines[boundary_end + 1:]
                
                # Write the updated file
                with open(file_path, 'w') as f:
                    f.writelines(new_lines)
                
                logger.info(f"Removed boundary '{boundary_name}' from {file_path.name}")
                return True
            else:
                logger.info(f"Boundary '{boundary_name}' not found in {file_path.name}")
                return True
            
        except Exception as e:
            logger.error(f"Error removing boundary from file {file_path}: {e}")
            return False

    def rename_auto_walls(self) -> bool:
        """
        Rename 'auto_walls' boundaries to 'walls' in the boundary file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.case_directory:
            logger.error("Cannot rename boundaries: No case directory specified")
            return False
        
        boundary_file = self.case_directory / "constant" / "polyMesh" / "boundary"
        
        if not boundary_file.exists():
            logger.error("Boundary file does not exist")
            return False
        
        try:
            # Read boundary file
            with open(boundary_file, 'r') as f:
                content = f.read()
            
            # Replace all occurrences of auto_walls with walls
            modified_content = re.sub(r'auto_walls', 'walls', content)
            
            # Write the modified content back to the file
            with open(boundary_file, 'w') as f:
                f.write(modified_content)
            
            logger.info("Successfully renamed 'auto_walls' to 'walls' in boundary file")
            return True
            
        except Exception as e:
            logger.error(f"Error renaming boundaries: {e}")
            return False

    def create_centered_ambient_region(self, center, size, name="ambientRegion", settings=None):
        """
        Create a centered ambient region in the OpenFOAM case.
        
        Args:
            center: Tuple/list of (x, y, z) coordinates for the center of the region
            size: Tuple/list of (width, height, depth) for the region
            name: Name of the ambient region
            settings: Dictionary of settings for the region
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.case_directory:
                logger.error("Cannot create ambient region: No case directory specified")
                return False
            
            # Log what we're doing
            logger.info(f"Creating centered ambient region '{name}' at {center} with size {size}")
            
            # Create cellZones directory if it doesn't exist
            constant_dir = self.case_directory / "constant"
            constant_dir.mkdir(exist_ok=True)
            
            cell_zones_dir = constant_dir / "cellZones"
            cell_zones_dir.mkdir(exist_ok=True)
            
            # Extract region parameters
            cx, cy, cz = center
            sx, sy, sz = size
            
            # Calculate bounding box
            x_min = cx - sx/2
            x_max = cx + sx/2
            y_min = cy - sy/2
            y_max = cy + sy/2
            z_min = cz - sz/2
            z_max = cz + sz/2
            
            # Create cellZone file for the ambient region
            cell_zone_file = cell_zones_dir / name
            
            with open(cell_zone_file, 'w') as f:
                f.write(f"""/*--------------------------------*- C++ -*----------------------------------*\\
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
    object      {name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

type            boxToCell;
box             ({x_min} {y_min} {z_min}) ({x_max} {y_max} {z_max});

// ************************************************************************* //
""")

            # Create topoSetDict to use with the topoSet utility
            system_dir = self.case_directory / "system"
            system_dir.mkdir(exist_ok=True)
            
            topo_set_dict = system_dir / "topoSetDict"
            
            with open(topo_set_dict, 'w') as f:
                f.write(f"""/*--------------------------------*- C++ -*----------------------------------*\\
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
    object      topoSetDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

actions
(
    {{
        name            {name};
        type            cellSet;
        action          new;
        source          boxToCell;
        box             ({x_min} {y_min} {z_min}) ({x_max} {y_max} {z_max});
    }}
    
    {{
        name            {name}Zone;
        type            cellZoneSet;
        action          new;
        source          setToCellZone;
        set             {name};
    }}
);

// ************************************************************************* //
""")

            # Create or update fvOptions to use the cell zone for source terms if settings provided
            if settings:
                fv_options = system_dir / "fvOptions"
                
                # Default settings if not provided
                temperature = settings.get('temperature', 300)
                velocity = settings.get('velocity', [0, 0, 0])
                
                with open(fv_options, 'w') as f:
                    f.write(f"""/*--------------------------------*- C++ -*----------------------------------*\\
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
    object      fvOptions;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

{name}Temp
{{
    type            scalarFixedValueConstraint;
    active          true;
    selectionMode   cellZone;
    cellZone        {name}Zone;
    fieldName       T;
    value           {temperature};
}}

{name}Velocity
{{
    type            vectorFixedValueConstraint;
    active          true;
    selectionMode   cellZone;
    cellZone        {name}Zone;
    fieldName       U;
    value           ({velocity[0]} {velocity[1]} {velocity[2]});
}}

// ************************************************************************* //
""")

            logger.info(f"Successfully created ambient region '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error creating ambient region: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def fix_all_boundary_conditions(self):
        """
        Comprehensive fix for all boundary condition issues in the OpenFOAM case.
        Addresses missing boundary entries, pressure gradient, and syntax issues.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.case_directory:
                logger.error("Cannot fix boundary conditions: No case directory specified")
                return False
                
            logger.info("Performing comprehensive fix of boundary conditions")
            
            # Process field files in 0 directory
            import os
            zero_dir = os.path.join(self.case_directory, "0")
            if not os.path.exists(zero_dir):
                logger.warning("No 0 directory found, creating it")
                os.makedirs(zero_dir, exist_ok=True)
            
            # Fix all field files in the 0 directory
            field_files = []
            for file_name in os.listdir(zero_dir):
                file_path = os.path.join(zero_dir, file_name)
                if os.path.isfile(file_path) and file_name not in ['.', '..']:
                    field_files.append(file_path)
                    self._fix_field_file(file_path)
            
            # If p and U don't exist, create them
            required_fields = ["p", "U"]
            for field in required_fields:
                field_path = os.path.join(zero_dir, field)
                if field not in [os.path.basename(f) for f in field_files]:
                    logger.warning(f"Required field {field} not found, creating it")
                    self._create_default_field_file(field_path)
            
            # Check for pressure gradient
            p_path = os.path.join(zero_dir, "p")
            if os.path.exists(p_path):
                has_gradient = self.check_pressure_gradient(p_path)
                if not has_gradient:
                    logger.warning("No pressure gradient detected in p file. This may lead to zero residuals.")
                    # Consider automatically adding a pressure gradient here
            
            # Fix pressure reference cell in fvSolution
            self._fix_pressure_reference_cell()
            
            logger.info("Successfully fixed all boundary conditions")
            return True
            
        except Exception as e:
            logger.error(f"Error fixing boundary conditions: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        
    def _fix_field_file(self, file_path):
        """Fix the field file to ensure proper boundaryField syntax."""
        try:
            # Read the file
            with open(file_path, 'r') as f:
                content = f.read()
            
            field_name = os.path.basename(file_path)
            
            # Check if file already has a boundaryField section
            if 'boundaryField' in content:
                # Make sure boundaryField has proper opening brace
                if not re.search(r'boundaryField\s*{', content):
                    logger.info(f"FIXING: Adding missing opening brace after boundaryField in {field_name}")
                    # Fix missing opening brace
                    content = content.replace('boundaryField', 'boundaryField\n{')
                
                # Check for proper closure - count braces after boundaryField
                parts = content.split('boundaryField')
                if len(parts) > 1:
                    brace_count = 0
                    for char in parts[1]:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                    
                    # If brace count isn't zero, we're missing a closing brace
                    if brace_count > 0:
                        logger.info(f"FIXING: Adding missing closing brace to boundaryField in {field_name}")
                        content += '\n}'
                
                # Handle empty boundaryField section
                if re.search(r'boundaryField\s*{\s*}', content):
                    logger.info(f"FIXING: Empty boundaryField section in {field_name}")
                    
                    # Determine appropriate default boundary condition
                    bc_type = "zeroGradient"
                    if field_name == "U":
                        bc_type = "noSlip"
                    
                    # Replace empty boundaryField with one containing allBoundary
                    content = re.sub(
                        r'boundaryField\s*{\s*}',
                        f'boundaryField\n{{\n    allBoundary\n    {{\n        type            {bc_type};\n    }}\n}}',
                        content
                    )
                
                # Check if allBoundary is already defined
                if 'allBoundary' not in content:
                    # Determine appropriate boundary condition
                    bc_type = "zeroGradient"
                    if field_name == "U":
                        bc_type = "noSlip"
                    
                    logger.info(f"FIXING: Adding missing allBoundary entry to {field_name}")
                    
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
                            logger.info(f"FIXING: Adding missing closing brace to boundaryField in {field_name}")
                        
                        # Write the updated content
                        content = new_content
                
            else:
                # Create a complete boundaryField section
                logger.info(f"FIXING: No boundaryField section found in {field_name}, creating it")
                
                # Find where to insert the boundaryField section (typically after internalField)
                if 'internalField' in content:
                    # Determine appropriate boundary condition
                    bc_type = "zeroGradient"
                    if field_name == "U":
                        bc_type = "noSlip"
                    
                    # Create boundaryField section
                    boundary_field = f"""
boundaryField
{{
    allBoundary
    {{
        type            {bc_type};
    }}
}}
"""
                    # Insert after internalField
                    parts = content.split('internalField')
                    if len(parts) > 1:
                        # Find the end of the internalField statement
                        end_idx = parts[1].find(';')
                        if end_idx != -1:
                            content = parts[0] + 'internalField' + parts[1][:end_idx+1] + boundary_field + parts[1][end_idx+1:]
                else:
                    # If there's no internalField, better to create a whole new file
                    self._create_default_field_file(file_path)
                    return
            
            # Write the updated content
            with open(file_path, 'w') as f:
                f.write(content)
            
            logger.info(f"Fixed {field_name} file with proper boundary field syntax")
            
        except Exception as e:
            logger.error(f"Error fixing field file {os.path.basename(file_path)}: {e}")
        
    def _create_default_field_file(self, file_path):
        """Create a default field file with proper structure."""
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
            
        logger.info(f"Created new default {field_name} file with proper structure")

    def check_pressure_gradient(self, file_path):
        """
        Check if a pressure gradient is defined in the pressure field file.
        Returns True if a gradient is found, False otherwise.
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Look for multiple fixedValue boundaries with different values
            fixed_values = []
            
            # Extract all fixedValue entries with their values
            pattern = r'type\s+fixedValue.*?value\s+uniform\s+([0-9.-]+)'
            matches = re.findall(pattern, content, re.DOTALL)
            
            for match in matches:
                try:
                    value = float(match)
                    fixed_values.append(value)
                except ValueError:
                    pass
            
            # Check if we have at least two different pressure values
            if len(fixed_values) >= 2 and len(set(fixed_values)) >= 2:
                logger.info("Pressure gradient found in pressure field")
                return True
            else:
                logger.warning("No pressure gradient detected. Consider setting different pressures at inlet/outlet")
                return False
                
        except Exception as e:
            logger.error(f"Error checking pressure gradient: {e}")
            return False

    def _fix_pressure_reference_cell(self):
        """Fix pressure reference cell in fvSolution if needed."""
        import os
        system_dir = os.path.join(self.case_directory, "system")
        if not os.path.exists(system_dir):
            os.makedirs(system_dir, exist_ok=True)
        
        fvSolution_path = os.path.join(system_dir, "fvSolution")
        
        # Check if fvSolution file exists
        if os.path.exists(fvSolution_path):
            try:
                # Read the file
                with open(fvSolution_path, 'r') as f:
                    content = f.read()
                
                # Check if PIMPLE solver is used but no reference cell is specified
                if 'PIMPLE' in content and not re.search(r'pRefCell', content):
                    logger.info("Adding pressure reference cell to fvSolution")
                    
                    # Find PIMPLE section
                    pimple_match = re.search(r'(PIMPLE\s*{[^}]*})', content, re.DOTALL)
                    if pimple_match:
                        # Check if the PIMPLE section already has closing brace
                        pimple_content = pimple_match.group(1)
                        if pimple_content.strip().endswith('}'):
                            # Insert before closing brace
                            new_pimple = pimple_content[:-1] + """
    pRefCell        0;
    pRefValue       0;
}"""
                            content = content.replace(pimple_content, new_pimple)
                        else:
                            # Add at the end of PIMPLE section
                            content = content.replace(pimple_content, pimple_content + """
    pRefCell        0;
    pRefValue       0;
}""")
                    
                    # Write the updated content
                    with open(fvSolution_path, 'w') as f:
                        f.write(content)
                        
                    logger.info("Added pressure reference cell to fvSolution")
            except Exception as e:
                logger.error(f"Error fixing pressure reference cell: {e}")


# Factory function to create a case manager
def create_case_manager(case_dir: str) -> CaseManager:
    """
    Create a case manager for the specified case directory.
    
    Args:
        case_dir (str): Path to the case directory
        
    Returns:
        CaseManager: Case manager instance
    """
    return CaseManager(case_dir=case_dir)


# Function to get available templates
def get_available_templates() -> Dict[str, str]:
    """
    Get available case templates.
    
    Returns:
        Dict[str, str]: Dictionary of template names and descriptions
    """
    template_dir = get_value('paths.templates', '')
    template_manager = CaseTemplate(template_dir)
    return template_manager.list_templates()


# Function to detect existing OpenFOAM cases
def find_openfoam_cases(base_dir: str) -> List[str]:
    """
    Find OpenFOAM cases in the specified directory.
    
    Args:
        base_dir (str): Base directory to search in
        
    Returns:
        List[str]: List of paths to OpenFOAM cases
    """
    cases = []
    base_path = Path(base_dir)
    
    try:
        # Look for directories that have system and constant subdirectories
        for item in base_path.iterdir():
            if item.is_dir():
                if (item / "system").exists() and (item / "constant").exists():
                    cases.append(str(item))
                else:
                    # Look one level deeper
                    for subitem in item.iterdir():
                        if subitem.is_dir():
                            if (subitem / "system").exists() and (subitem / "constant").exists():
                                cases.append(str(subitem))
        
        return cases
        
    except Exception as e:
        logger.error(f"Error finding OpenFOAM cases: {e}")
        return []