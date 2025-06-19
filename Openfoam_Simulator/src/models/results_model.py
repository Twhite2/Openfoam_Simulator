#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Results model for Openfoam_Simulator application.

This module defines the data model for storing, managing, and analyzing CFD
simulation results, with a focus on oil & gas industry applications.
"""

import os
import sys
import json
import time
import datetime
import logging
import shutil
import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set, TextIO

import numpy as np
import pandas as pd

# Import utility modules
from ..utils.logger import get_logger
from ..config import get_value, set_value

logger = get_logger(__name__)


class ResultType(Enum):
    """Enumeration of different types of simulation results."""
    
    SINGLE_PHASE = 1    # Results from single-phase simulations
    TWO_PHASE = 2       # Results from two-phase simulations
    THREE_PHASE = 3     # Results from three-phase simulations
    PIGGING = 4         # Results from pigging simulations
    SPILL = 5           # Results from spill simulations
    UNKNOWN = 100       # Unknown result type


class FieldType(Enum):
    """Enumeration of different field types in the results."""
    
    SCALAR = 1          # Scalar field (e.g., pressure, temperature)
    VECTOR = 2          # Vector field (e.g., velocity)
    TENSOR = 3          # Tensor field (e.g., stress tensor)
    VOLSCALAR = 4       # Volume fraction scalar
    DERIVED = 5         # Derived field calculated from other fields
    UNKNOWN = 100       # Unknown field type


class FieldData:
    """Class representing a field data in the simulation results."""
    
    def __init__(self, name: str, field_type: FieldType, dimensions: List[str] = None, 
                units: str = "", description: str = ""):
        """
        Initialize a field data object.
        
        Args:
            name (str): Field name
            field_type (FieldType): Type of field (scalar, vector, etc.)
            dimensions (List[str], optional): Dimensions of the field
            units (str, optional): Units of the field
            description (str, optional): Description of the field
        """
        self.name = name
        self.field_type = field_type
        self.dimensions = dimensions or []
        self.units = units
        self.description = description
        
        # Data is stored as a dictionary mapping time to values
        # For scalar fields, values are arrays
        # For vector fields, values are dictionaries of component arrays
        self.data = {}
        
        # Min/max values for quick access
        self.min_value = None
        self.max_value = None
        
        # Derived field calculation expression
        self.expression = ""
    
    def add_time_data(self, time: float, values: Union[np.ndarray, Dict[str, np.ndarray]]):
        """
        Add data for a specific time.
        
        Args:
            time (float): Simulation time
            values (Union[np.ndarray, Dict[str, np.ndarray]]): Field values
        """
        self.data[time] = values
        
        # Update min/max for scalar fields
        if self.field_type == FieldType.SCALAR or self.field_type == FieldType.VOLSCALAR:
            if isinstance(values, np.ndarray):
                current_min = np.nanmin(values) if values.size > 0 else None
                current_max = np.nanmax(values) if values.size > 0 else None
                
                if self.min_value is None or (current_min is not None and current_min < self.min_value):
                    self.min_value = current_min
                
                if self.max_value is None or (current_max is not None and current_max > self.max_value):
                    self.max_value = current_max
    
    def get_time_data(self, time: float) -> Optional[Union[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Get data for a specific time.
        
        Args:
            time (float): Simulation time
            
        Returns:
            Optional[Union[np.ndarray, Dict[str, np.ndarray]]]: Field values or None if time not found
        """
        return self.data.get(time)
    
    def get_time_points(self) -> List[float]:
        """
        Get list of time points for which data is available.
        
        Returns:
            List[float]: List of time points
        """
        return sorted(self.data.keys())
    
    def get_min_value(self) -> Optional[float]:
        """
        Get minimum value across all time points.
        
        Returns:
            Optional[float]: Minimum value or None if no data
        """
        return self.min_value
    
    def get_max_value(self) -> Optional[float]:
        """
        Get maximum value across all time points.
        
        Returns:
            Optional[float]: Maximum value or None if no data
        """
        return self.max_value
    
    def clear(self):
        """Clear all data."""
        self.data.clear()
        self.min_value = None
        self.max_value = None
    
    def calculate_derived_field(self, expression: str, fields: Dict[str, 'FieldData']):
        """
        Calculate a derived field using an expression.
        
        Args:
            expression (str): Expression to evaluate (e.g., 'p / rho')
            fields (Dict[str, FieldData]): Dictionary of available fields
        
        Returns:
            bool: True if calculation successful, False otherwise
        """
        if self.field_type != FieldType.DERIVED:
            logger.error(f"Can only calculate derived fields, but {self.name} is {self.field_type}")
            return False
        
        self.expression = expression
        
        # Get all time points from input fields
        time_points = set()
        for field in fields.values():
            time_points.update(field.get_time_points())
        time_points = sorted(time_points)
        
        # Clear existing data
        self.clear()
        
        # For each time point, calculate derived field
        for time in time_points:
            try:
                # Create a dictionary of field values at this time
                local_vars = {}
                all_fields_available = True
                for field_name, field in fields.items():
                    field_data = field.get_time_data(time)
                    if field_data is None:
                        all_fields_available = False
                        break
                    
                    if field.field_type == FieldType.VECTOR:
                        # For vector fields, add separate variables for components
                        for component, values in field_data.items():
                            local_vars[f"{field_name}_{component}"] = values
                        
                        # Also add magnitude for vector fields
                        components = list(field_data.values())
                        if all(isinstance(comp, np.ndarray) and comp.shape == components[0].shape 
                              for comp in components):
                            # Calculate magnitude
                            mag = np.sqrt(sum(comp**2 for comp in components))
                            local_vars[f"{field_name}_mag"] = mag
                    else:
                        # For scalar fields, just add the array
                        local_vars[field_name] = field_data
                
                if not all_fields_available:
                    continue
                
                # Add numpy functions to the namespace
                local_vars.update({
                    'np': np,
                    'sin': np.sin,
                    'cos': np.cos,
                    'tan': np.tan,
                    'exp': np.exp,
                    'log': np.log,
                    'sqrt': np.sqrt,
                    'abs': np.abs,
                    'max': np.maximum,
                    'min': np.minimum
                })
                
                # Evaluate the expression
                result = eval(expression, {}, local_vars)
                
                # Add result to data
                self.add_time_data(time, result)
                
            except Exception as e:
                logger.error(f"Error calculating derived field {self.name} at time {time}: {e}")
                continue
        
        return len(self.data) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            'name': self.name,
            'field_type': self.field_type.name,
            'dimensions': self.dimensions,
            'units': self.units,
            'description': self.description,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'expression': self.expression
            # Note: actual data is not serialized here, it's stored separately
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FieldData':
        """
        Create from dictionary representation.
        
        Args:
            data (Dict[str, Any]): Dictionary representation
            
        Returns:
            FieldData: New FieldData object
        """
        field_type = FieldType[data['field_type']] if 'field_type' in data else FieldType.UNKNOWN
        field = cls(
            name=data.get('name', ''),
            field_type=field_type,
            dimensions=data.get('dimensions', []),
            units=data.get('units', ''),
            description=data.get('description', '')
        )
        field.min_value = data.get('min_value')
        field.max_value = data.get('max_value')
        field.expression = data.get('expression', '')
        return field


class ResultsModel:
    """Class representing simulation results from a CFD simulation."""
    
    def __init__(self, name: str = "Simulation Results", result_type: ResultType = ResultType.UNKNOWN):
        """
        Initialize a results model.
        
        Args:
            name (str, optional): Name of the results
            result_type (ResultType, optional): Type of the results
        """
        self.name = name
        self.result_type = result_type
        self.directory = None  # Path to results directory
        
        # Metadata
        self.creation_time = time.time()
        self.last_modified_time = self.creation_time
        self.description = ""
        self.solver = ""
        self.case_name = ""
        self.metadata = {}  # Additional metadata
        
        # Fields
        self.fields = {}  # Dict[str, FieldData]
        
        # Time points
        self.times = []  # List of time values for which data is available
        
        # Mesh information
        self.mesh_info = {
            'nodes': 0,
            'cells': 0,
            'faces': 0,
            'boundaries': {}
        }
        
        # Convergence information
        self.convergence_info = {
            'residuals': {},   # Dict mapping field -> List of residuals
            'iterations': 0,   # Number of iterations
            'converged': False # Whether simulation converged
        }
        
        # Special oil & gas industry results
        self.pressure_drop = None  # Pressure drop across domain
        self.flow_rates = {}       # Dict mapping phase -> flow rate
        self.holdup = {}           # Dict mapping phase -> holdup
        self.pigging_results = {}  # Special results for pigging simulations
        self.spill_results = {}    # Special results for spill simulations
    
    def set_directory(self, directory: str):
        """
        Set the directory containing the results.
        
        Args:
            directory (str): Path to the results directory
        """
        self.directory = directory
    
    def add_field(self, field: FieldData):
        """
        Add a field to the results.
        
        Args:
            field (FieldData): Field to add
        """
        self.fields[field.name] = field
        self._update_times()
        self.last_modified_time = time.time()
    
    def get_field(self, name: str) -> Optional[FieldData]:
        """
        Get a field by name.
        
        Args:
            name (str): Name of the field
            
        Returns:
            Optional[FieldData]: Field data or None if not found
        """
        return self.fields.get(name)
    
    def has_field(self, name: str) -> bool:
        """
        Check if a field exists.
        
        Args:
            name (str): Name of the field
            
        Returns:
            bool: True if field exists, False otherwise
        """
        return name in self.fields
    
    def remove_field(self, name: str) -> bool:
        """
        Remove a field by name.
        
        Args:
            name (str): Name of the field
            
        Returns:
            bool: True if field was removed, False otherwise
        """
        if name in self.fields:
            del self.fields[name]
            self._update_times()
            self.last_modified_time = time.time()
            return True
        return False
    
    def clear_fields(self):
        """Clear all fields."""
        self.fields.clear()
        self.times.clear()
        self.last_modified_time = time.time()
    
    def _update_times(self):
        """Update the list of time points based on available field data."""
        time_points = set()
        for field in self.fields.values():
            time_points.update(field.get_time_points())
        self.times = sorted(time_points)
    
    def get_times(self) -> List[float]:
        """
        Get list of time points for which data is available.
        
        Returns:
            List[float]: List of time points
        """
        return self.times
    
    def get_field_names(self) -> List[str]:
        """
        Get list of field names.
        
        Returns:
            List[str]: List of field names
        """
        return list(self.fields.keys())
    
    def get_field_names_by_type(self, field_type: FieldType) -> List[str]:
        """
        Get list of field names of a specific type.
        
        Args:
            field_type (FieldType): Type of fields to get
            
        Returns:
            List[str]: List of field names
        """
        return [name for name, field in self.fields.items() if field.field_type == field_type]
    
    def add_scalar_field(self, name: str, units: str = "", description: str = ""):
        """
        Add a new scalar field.
        
        Args:
            name (str): Field name
            units (str, optional): Units of the field
            description (str, optional): Description of the field
            
        Returns:
            FieldData: The created field
        """
        field = FieldData(name, FieldType.SCALAR, units=units, description=description)
        self.add_field(field)
        return field
    
    def add_vector_field(self, name: str, dimensions: List[str] = None, 
                         units: str = "", description: str = ""):
        """
        Add a new vector field.
        
        Args:
            name (str): Field name
            dimensions (List[str], optional): Dimensions of the field (e.g., ['x', 'y', 'z'])
            units (str, optional): Units of the field
            description (str, optional): Description of the field
            
        Returns:
            FieldData: The created field
        """
        field = FieldData(name, FieldType.VECTOR, dimensions=dimensions or ['x', 'y', 'z'],
                         units=units, description=description)
        self.add_field(field)
        return field
    
    def add_derived_field(self, name: str, expression: str, units: str = "", description: str = ""):
        """
        Add a new derived field calculated from other fields.
        
        Args:
            name (str): Field name
            expression (str): Expression to evaluate (e.g., 'p / rho')
            units (str, optional): Units of the field
            description (str, optional): Description of the field
            
        Returns:
            FieldData: The created field
        """
        field = FieldData(name, FieldType.DERIVED, units=units, description=description)
        field.expression = expression
        field.calculate_derived_field(expression, self.fields)
        self.add_field(field)
        return field
    
    def set_convergence_info(self, residuals: Dict[str, List[float]], iterations: int, converged: bool):
        """
        Set convergence information.
        
        Args:
            residuals (Dict[str, List[float]]): Residuals for each field
            iterations (int): Number of iterations
            converged (bool): Whether simulation converged
        """
        self.convergence_info = {
            'residuals': residuals,
            'iterations': iterations,
            'converged': converged
        }
        self.last_modified_time = time.time()
    
    def set_mesh_info(self, nodes: int, cells: int, faces: int, boundaries: Dict[str, int]):
        """
        Set mesh information.
        
        Args:
            nodes (int): Number of nodes
            cells (int): Number of cells
            faces (int): Number of faces
            boundaries (Dict[str, int]): Boundary names and cell counts
        """
        self.mesh_info = {
            'nodes': nodes,
            'cells': cells,
            'faces': faces,
            'boundaries': boundaries
        }
        self.last_modified_time = time.time()
    
    def set_oil_gas_results(self, pressure_drop: float = None, 
                           flow_rates: Dict[str, float] = None,
                           holdup: Dict[str, float] = None):
        """
        Set oil & gas specific results.
        
        Args:
            pressure_drop (float, optional): Pressure drop across domain
            flow_rates (Dict[str, float], optional): Flow rates for each phase
            holdup (Dict[str, float], optional): Holdup for each phase
        """
        if pressure_drop is not None:
            self.pressure_drop = pressure_drop
        
        if flow_rates is not None:
            self.flow_rates.update(flow_rates)
        
        if holdup is not None:
            self.holdup.update(holdup)
        
        self.last_modified_time = time.time()
    
    def set_pigging_results(self, pigging_results: Dict[str, Any]):
        """
        Set pigging simulation specific results.
        
        Args:
            pigging_results (Dict[str, Any]): Pigging simulation results
        """
        self.pigging_results = pigging_results
        self.last_modified_time = time.time()
    
    def set_spill_results(self, spill_results: Dict[str, Any]):
        """
        Set spill simulation specific results.
        
        Args:
            spill_results (Dict[str, Any]): Spill simulation results
        """
        self.spill_results = spill_results
        self.last_modified_time = time.time()
    
    def calculate_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for scalar fields.
        
        Returns:
            Dict[str, Dict[str, float]]: Statistics for each scalar field
        """
        statistics = {}
        
        for name, field in self.fields.items():
            if field.field_type == FieldType.SCALAR or field.field_type == FieldType.VOLSCALAR:
                field_stats = {
                    'min': field.get_min_value(),
                    'max': field.get_max_value(),
                    'times': len(field.get_time_points())
                }
                
                # Calculate average over all time points
                if field_stats['times'] > 0:
                    total_avg = 0.0
                    total_std = 0.0
                    count = 0
                    
                    for time in field.get_time_points():
                        values = field.get_time_data(time)
                        if values is not None and isinstance(values, np.ndarray) and values.size > 0:
                            avg = np.nanmean(values)
                            std = np.nanstd(values)
                            total_avg += avg
                            total_std += std
                            count += 1
                    
                    if count > 0:
                        field_stats['avg'] = total_avg / count
                        field_stats['std'] = total_std / count
                
                statistics[name] = field_stats
        
        return statistics
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            'name': self.name,
            'result_type': self.result_type.name,
            'directory': self.directory,
            'creation_time': self.creation_time,
            'last_modified_time': self.last_modified_time,
            'description': self.description,
            'solver': self.solver,
            'case_name': self.case_name,
            'metadata': self.metadata,
            'times': self.times,
            'mesh_info': self.mesh_info,
            'convergence_info': self.convergence_info,
            'pressure_drop': self.pressure_drop,
            'flow_rates': self.flow_rates,
            'holdup': self.holdup,
            'pigging_results': self.pigging_results,
            'spill_results': self.spill_results,
            'fields': {name: field.to_dict() for name, field in self.fields.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResultsModel':
        """
        Create from dictionary representation.
        
        Args:
            data (Dict[str, Any]): Dictionary representation
            
        Returns:
            ResultsModel: New ResultsModel object
        """
        result_type = ResultType[data['result_type']] if 'result_type' in data else ResultType.UNKNOWN
        model = cls(name=data.get('name', ''), result_type=result_type)
        
        model.directory = data.get('directory')
        model.creation_time = data.get('creation_time', time.time())
        model.last_modified_time = data.get('last_modified_time', time.time())
        model.description = data.get('description', '')
        model.solver = data.get('solver', '')
        model.case_name = data.get('case_name', '')
        model.metadata = data.get('metadata', {})
        model.times = data.get('times', [])
        model.mesh_info = data.get('mesh_info', {'nodes': 0, 'cells': 0, 'faces': 0, 'boundaries': {}})
        model.convergence_info = data.get('convergence_info', {'residuals': {}, 'iterations': 0, 'converged': False})
        model.pressure_drop = data.get('pressure_drop')
        model.flow_rates = data.get('flow_rates', {})
        model.holdup = data.get('holdup', {})
        model.pigging_results = data.get('pigging_results', {})
        model.spill_results = data.get('spill_results', {})
        
        # Load fields (but not their data)
        if 'fields' in data:
            for field_name, field_data in data['fields'].items():
                model.fields[field_name] = FieldData.from_dict(field_data)
        
        return model
    
    def save(self, directory: str = None) -> bool:
        """
        Save results to a directory.
        
        Args:
            directory (str, optional): Directory to save to, if None uses self.directory
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        if directory is not None:
            self.directory = directory
        
        if not self.directory:
            logger.error("No directory specified for saving results")
            return False
        
        # Create directory if it doesn't exist
        os.makedirs(self.directory, exist_ok=True)
        
        try:
            # Save metadata
            metadata_file = os.path.join(self.directory, 'metadata.json')
            with open(metadata_file, 'w') as f:
                metadata = self.to_dict()
                
                # Remove field data to keep metadata file small
                for field_data in metadata['fields'].values():
                    field_data.pop('data', None)
                
                json.dump(metadata, f, indent=2)
            
            # Save field data
            fields_dir = os.path.join(self.directory, 'fields')
            os.makedirs(fields_dir, exist_ok=True)
            
            for field_name, field in self.fields.items():
                field_dir = os.path.join(fields_dir, field_name)
                os.makedirs(field_dir, exist_ok=True)
                
                # Save field metadata
                field_metadata_file = os.path.join(field_dir, 'metadata.json')
                with open(field_metadata_file, 'w') as f:
                    json.dump(field.to_dict(), f, indent=2)
                
                # Save field data for each time point
                for time in field.get_time_points():
                    time_data = field.get_time_data(time)
                    if time_data is not None:
                        time_file = os.path.join(field_dir, f'{time:.6f}.npz')
                        
                        if isinstance(time_data, dict):
                            # Vector field
                            np.savez_compressed(time_file, **time_data)
                        else:
                            # Scalar field
                            np.savez_compressed(time_file, data=time_data)
            
            logger.info(f"Results saved to {self.directory}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    @classmethod
    def load(cls, directory: str) -> Optional['ResultsModel']:
        """
        Load results from a directory.
        
        Args:
            directory (str): Directory to load from
            
        Returns:
            Optional[ResultsModel]: Loaded results or None if loading failed
        """
        # Check if directory exists
        if not os.path.isdir(directory):
            logger.error(f"Results directory not found: {directory}")
            return None
        
        try:
            # Load metadata
            metadata_file = os.path.join(directory, 'metadata.json')
            if not os.path.isfile(metadata_file):
                logger.error(f"Metadata file not found in results directory: {metadata_file}")
                return None
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Create model from metadata
            model = cls.from_dict(metadata)
            model.directory = directory
            
            # Load field data
            fields_dir = os.path.join(directory, 'fields')
            if os.path.isdir(fields_dir):
                for field_name, field in model.fields.items():
                    field_dir = os.path.join(fields_dir, field_name)
                    if not os.path.isdir(field_dir):
                        logger.warning(f"Field directory not found: {field_dir}")
                        continue
                    
                    # Find all time files
                    time_files = [f for f in os.listdir(field_dir) if f.endswith('.npz')]
                    for time_file in time_files:
                        try:
                            # Extract time from filename
                            time = float(os.path.splitext(time_file)[0])
                            
                            # Load data
                            time_data_file = os.path.join(field_dir, time_file)
                            npz_data = np.load(time_data_file)
                            
                            if 'data' in npz_data:
                                # Scalar field
                                field.add_time_data(time, npz_data['data'])
                            else:
                                # Vector field
                                field.add_time_data(time, {k: npz_data[k] for k in npz_data.files})
                                
                        except Exception as e:
                            logger.warning(f"Error loading time file {time_file} for field {field_name}: {e}")
                            continue
            
            # Update times
            model._update_times()
            
            logger.info(f"Results loaded from {directory}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return None
    
    def load_openfoam_results(self, case_dir: str, field_names: List[str] = None, 
                             time_values: List[float] = None) -> bool:
        """
        Load OpenFOAM results from a case directory.
        
        Args:
            case_dir (str): Path to OpenFOAM case directory
            field_names (List[str], optional): List of field names to load, if None loads all
            time_values (List[float], optional): List of time values to load, if None loads all
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        if not os.path.isdir(case_dir):
            logger.error(f"OpenFOAM case directory not found: {case_dir}")
            return False
        
        # Store case directory
        self.directory = case_dir
        
        try:
            # Find all time directories
            time_dirs = []
            for item in os.listdir(case_dir):
                try:
                    time_value = float(item)
                    time_dir = os.path.join(case_dir, item)
                    if os.path.isdir(time_dir):
                        time_dirs.append((time_value, time_dir))
                except ValueError:
                    # Not a time directory
                    continue
            
            # Sort by time
            time_dirs.sort()
            
            # Filter by time values if provided
            if time_values is not None:
                time_dirs = [(t, d) for t, d in time_dirs if t in time_values]
            
            # Load metadata from case
            self._load_openfoam_metadata(case_dir)
            
            # Load fields from each time directory
            for time_value, time_dir in time_dirs:
                self._load_openfoam_time_directory(time_value, time_dir, field_names)
            
            # Update times
            self._update_times()
            
            logger.info(f"OpenFOAM results loaded from {case_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading OpenFOAM results: {e}")
            return False
    
    def _load_openfoam_metadata(self, case_dir: str):
        """
        Load metadata from OpenFOAM case.
        
        Args:
            case_dir (str): Path to OpenFOAM case directory
        """
        # Extract case name from directory
        self.case_name = os.path.basename(os.path.normpath(case_dir))
        
        # Check for controlDict to extract solver name
        control_dict_path = os.path.join(case_dir, 'system', 'controlDict')
        if os.path.isfile(control_dict_path):
            try:
                with open(control_dict_path, 'r') as f:
                    for line in f:
                        if 'application' in line and ';' in line:
                            parts = line.strip().split()
                            self.solver = parts[1].rstrip(';')
                            break
            except Exception as e:
                logger.warning(f"Error reading controlDict: {e}")
        
        # Determine result type based on solver
        if self.solver.lower().endswith('foam'):
            if 'interFoam' in self.solver or 'multiphaseInterFoam' in self.solver:
                if 'twoPhase' in self.solver:
                    self.result_type = ResultType.TWO_PHASE
                elif 'multiphase' in self.solver:
                    self.result_type = ResultType.THREE_PHASE
                else:
                    self.result_type = ResultType.TWO_PHASE
            else:
                self.result_type = ResultType.SINGLE_PHASE
        
        # Load mesh information
        self._load_openfoam_mesh_info(case_dir)
    
    def _load_openfoam_mesh_info(self, case_dir: str):
        """
        Load mesh information from OpenFOAM case.
        
        Args:
            case_dir (str): Path to OpenFOAM case directory
        """
        # Check for polyMesh directory in constant or latest time
        mesh_dir = os.path.join(case_dir, 'constant', 'polyMesh')
        if not os.path.isdir(mesh_dir):
            # Try to find in latest time directory
            time_dirs = []
            for item in os.listdir(case_dir):
                try:
                    time_value = float(item)
                    time_dir = os.path.join(case_dir, item)
                    poly_mesh = os.path.join(time_dir, 'polyMesh')
                    if os.path.isdir(time_dir) and os.path.isdir(poly_mesh):
                        time_dirs.append((time_value, poly_mesh))
                except ValueError:
                    continue
            
            if time_dirs:
                # Use mesh from latest time
                time_dirs.sort()
                mesh_dir = time_dirs[-1][1]
        
        if not os.path.isdir(mesh_dir):
            logger.warning(f"Mesh directory not found in OpenFOAM case")
            return
        
        # Read points file
        points_file = os.path.join(mesh_dir, 'points')
        if os.path.isfile(points_file):
            try:
                with open(points_file, 'r') as f:
                    for line in f:
                        if line.strip().isdigit():
                            self.mesh_info['nodes'] = int(line.strip())
                            break
            except Exception as e:
                logger.warning(f"Error reading points file: {e}")
        
        # Read faces file
        faces_file = os.path.join(mesh_dir, 'faces')
        if os.path.isfile(faces_file):
            try:
                with open(faces_file, 'r') as f:
                    for line in f:
                        if line.strip().isdigit():
                            self.mesh_info['faces'] = int(line.strip())
                            break
            except Exception as e:
                logger.warning(f"Error reading faces file: {e}")
        
        # Read cells file
        cells_file = os.path.join(mesh_dir, 'cells')
        if os.path.isfile(cells_file):
            try:
                with open(cells_file, 'r') as f:
                    for line in f:
                        if line.strip().isdigit():
                            self.mesh_info['cells'] = int(line.strip())
                            break
            except Exception as e:
                logger.warning(f"Error reading cells file: {e}")
        
        # Read boundary file
        boundary_file = os.path.join(mesh_dir, 'boundary')
        if os.path.isfile(boundary_file):
            try:
                with open(boundary_file, 'r') as f:
                    content = f.read()
                    # Use regex to find boundary names and nFaces
                    pattern = r'(\w+)\s*\{[^}]*nFaces\s+(\d+);'
                    matches = re.findall(pattern, content)
                    
                    for boundary_name, n_faces in matches:
                        self.mesh_info['boundaries'][boundary_name] = int(n_faces)
            except Exception as e:
                logger.warning(f"Error reading boundary file: {e}")
    
    def _load_openfoam_time_directory(self, time_value: float, time_dir: str, 
                                     field_names: List[str] = None):
        """
        Load fields from an OpenFOAM time directory.
        
        Args:
            time_value (float): Time value
            time_dir (str): Path to time directory
            field_names (List[str], optional): List of field names to load, if None loads all
        """
        # Find field files in the time directory
        field_files = []
        for item in os.listdir(time_dir):
            file_path = os.path.join(time_dir, item)
            if os.path.isfile(file_path) and not item.startswith('.'):
                field_files.append((item, file_path))
        
        # Filter by field names if provided
        if field_names is not None:
            field_files = [(name, path) for name, path in field_files if name in field_names]
        
        # Load each field
        for field_name, field_path in field_files:
            try:
                self._load_openfoam_field(time_value, field_name, field_path)
            except Exception as e:
                logger.warning(f"Error loading field {field_name} at time {time_value}: {e}")
                continue
    
    def _load_openfoam_field(self, time_value: float, field_name: str, field_path: str):
        """
        Load a field from an OpenFOAM field file.
        
        Args:
            time_value (float): Time value
            field_name (str): Field name
            field_path (str): Path to field file
        """
        # Read field header to determine type
        field_type = FieldType.UNKNOWN
        dimensions = []
        units = ""
        
        try:
            with open(field_path, 'r') as f:
                # Check first few lines for field type
                header_lines = [f.readline() for _ in range(20)]
                header = ''.join(header_lines)
                
                # Determine field type
                if 'volScalarField' in header:
                    field_type = FieldType.SCALAR
                elif 'volVectorField' in header:
                    field_type = FieldType.VECTOR
                    dimensions = ['x', 'y', 'z']
                elif 'volTensorField' in header:
                    field_type = FieldType.TENSOR
                elif 'surfaceScalarField' in header:
                    field_type = FieldType.SCALAR
                
                # Extract dimensions
                dim_match = re.search(r'dimensions\s+\[(.*?)\]', header)
                if dim_match:
                    dim_str = dim_match.group(1)
                    # TODO: Parse dimensions to determine units
                
                # For phase fields (alpha)
                if field_name.startswith('alpha.'):
                    field_type = FieldType.VOLSCALAR
        
        except Exception as e:
            logger.warning(f"Error reading field header for {field_name}: {e}")
            return
        
        # Create field if not exists
        if not self.has_field(field_name):
            field = FieldData(field_name, field_type, dimensions, units)
            self.add_field(field)
        else:
            field = self.get_field(field_name)
        
        # Read field data
        try:
            # This is a simplified approach - a full implementation would need to parse
            # OpenFOAM's complex file format more carefully
            with open(field_path, 'r') as f:
                content = f.read()
                
                # Look for internalField section
                internal_field_match = re.search(r'internalField\s+(uniform|nonuniform)\s+(.*?);', 
                                               content, re.DOTALL)
                
                if internal_field_match:
                    field_kind = internal_field_match.group(1)
                    field_data = internal_field_match.group(2)
                    
                    if field_kind == 'uniform':
                        # Uniform field - single value
                        if field_type == FieldType.SCALAR or field_type == FieldType.VOLSCALAR:
                            # Scalar field
                            try:
                                value = float(field_data)
                                # Create array with uniform value, size based on mesh
                                values = np.full(self.mesh_info['cells'], value)
                                field.add_time_data(time_value, values)
                            except ValueError:
                                logger.warning(f"Error parsing uniform scalar value for {field_name}: {field_data}")
                        
                        elif field_type == FieldType.VECTOR:
                            # Vector field
                            try:
                                # Parse vector like (x y z)
                                vector_match = re.match(r'\(\s*(.*?)\s+(.*?)\s+(.*?)\s*\)', field_data)
                                if vector_match:
                                    x, y, z = map(float, vector_match.groups())
                                    # Create arrays with uniform values
                                    values = {
                                        'x': np.full(self.mesh_info['cells'], x),
                                        'y': np.full(self.mesh_info['cells'], y),
                                        'z': np.full(self.mesh_info['cells'], z)
                                    }
                                    field.add_time_data(time_value, values)
                            except ValueError:
                                logger.warning(f"Error parsing uniform vector value for {field_name}: {field_data}")
                    
                    elif field_kind == 'nonuniform':
                        # Nonuniform field - list of values
                        if field_type == FieldType.SCALAR or field_type == FieldType.VOLSCALAR:
                            # Scalar field
                            try:
                                # Find list of values inside ()
                                list_match = re.search(r'\((.*?)\)', field_data, re.DOTALL)
                                if list_match:
                                    values_str = list_match.group(1)
                                    # Parse values, handling scientific notation
                                    values = np.array([float(v) for v in values_str.split()])
                                    field.add_time_data(time_value, values)
                            except Exception as e:
                                logger.warning(f"Error parsing nonuniform scalar values for {field_name}: {e}")
                        
                        elif field_type == FieldType.VECTOR:
                            # Vector field
                            try:
                                # Find list of vectors
                                list_match = re.search(r'\((.*?)\)', field_data, re.DOTALL)
                                if list_match:
                                    vectors_str = list_match.group(1)
                                    # Parse each vector (x y z)
                                    vector_pattern = r'\(\s*(.*?)\s+(.*?)\s+(.*?)\s*\)'
                                    vector_matches = re.findall(vector_pattern, vectors_str)
                                    
                                    # Convert to component arrays
                                    if vector_matches:
                                        x_values = np.array([float(x) for x, _, _ in vector_matches])
                                        y_values = np.array([float(y) for _, y, _ in vector_matches])
                                        z_values = np.array([float(z) for _, _, z in vector_matches])
                                        
                                        values = {
                                            'x': x_values,
                                            'y': y_values,
                                            'z': z_values
                                        }
                                        field.add_time_data(time_value, values)
                            except Exception as e:
                                logger.warning(f"Error parsing nonuniform vector values for {field_name}: {e}")
        
        except Exception as e:
            logger.warning(f"Error loading field data for {field_name}: {e}")
    
    def export_to_vtk(self, output_path: str) -> bool:
        """
        Export results to VTK format.
        
        Args:
            output_path (str): Path to output file or directory
            
        Returns:
            bool: True if export successful, False otherwise
        """
        # This would require PyVTK or another VTK writing library
        # It's beyond the scope of this implementation
        logger.error("VTK export not implemented")
        return False
    
    def export_to_csv(self, output_path: str, field_names: List[str] = None, 
                     time_values: List[float] = None) -> bool:
        """
        Export results to CSV format.
        
        Args:
            output_path (str): Path to output directory
            field_names (List[str], optional): Fields to export, if None exports all scalar fields
            time_values (List[float], optional): Time values to export, if None exports all
            
        Returns:
            bool: True if export successful, False otherwise
        """
        if not self.fields:
            logger.error("No fields to export")
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        try:
            # Determine fields to export
            if field_names is None:
                # Default to all scalar fields
                field_names = self.get_field_names_by_type(FieldType.SCALAR)
                field_names.extend(self.get_field_names_by_type(FieldType.VOLSCALAR))
            
            # Determine time values to export
            if time_values is None:
                time_values = self.get_times()
            
            # Export each field
            for field_name in field_names:
                field = self.get_field(field_name)
                if field is None:
                    logger.warning(f"Field not found: {field_name}")
                    continue
                
                # Skip non-scalar fields
                if field.field_type != FieldType.SCALAR and field.field_type != FieldType.VOLSCALAR:
                    logger.info(f"Skipping non-scalar field: {field_name}")
                    continue
                
                # Create CSV file for this field
                csv_path = os.path.join(output_path, f"{field_name}.csv")
                
                with open(csv_path, 'w') as f:
                    # Write header
                    f.write(f"Cell,{','.join(f'Time_{t:.6f}' for t in time_values)}\n")
                    
                    # Determine number of cells
                    max_cells = 0
                    for time in time_values:
                        values = field.get_time_data(time)
                        if values is not None and isinstance(values, np.ndarray):
                            max_cells = max(max_cells, len(values))
                    
                    # Write data rows
                    for cell_idx in range(max_cells):
                        row = [str(cell_idx)]
                        
                        for time in time_values:
                            values = field.get_time_data(time)
                            if values is not None and isinstance(values, np.ndarray) and cell_idx < len(values):
                                row.append(f"{values[cell_idx]}")
                            else:
                                row.append("")
                        
                        f.write(",".join(row) + "\n")
                
                logger.info(f"Exported field {field_name} to {csv_path}")
            
            # Export summary file
            summary_path = os.path.join(output_path, "summary.csv")
            with open(summary_path, 'w') as f:
                # Write headers
                f.write("Field,Min,Max,Average,StdDev\n")
                
                # Calculate statistics for each field
                statistics = self.calculate_statistics()
                
                for field_name, stats in statistics.items():
                    if field_name in field_names:
                        min_val = stats.get('min', '')
                        max_val = stats.get('max', '')
                        avg_val = stats.get('avg', '')
                        std_val = stats.get('std', '')
                        
                        f.write(f"{field_name},{min_val},{max_val},{avg_val},{std_val}\n")
            
            logger.info(f"Exported summary to {summary_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False
    
    def export_to_paraview(self, output_path: str) -> bool:
        """
        Export results for ParaView visualization.
        
        Args:
            output_path (str): Path to output file (e.g., .foam)
            
        Returns:
            bool: True if export successful, False otherwise
        """
        if not self.directory:
            logger.error("No results directory to export")
            return False
        
        try:
            # ParaView can read OpenFOAM cases directly
            # We just need to create a .foam file pointing to the case
            with open(output_path, 'w') as f:
                f.write(f"// Openfoam_Simulator ParaView Export\n")
                f.write(f"// Original case: {self.directory}\n")
            
            logger.info(f"Exported ParaView file to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to ParaView: {e}")
            return False
    
    def generate_report(self, template_path: str, output_path: str) -> bool:
        """
        Generate a report from the results.
        
        Args:
            template_path (str): Path to report template
            output_path (str): Path to output file
            
        Returns:
            bool: True if report generation successful, False otherwise
        """
        # This would require a template engine like Jinja2
        # It's beyond the scope of this implementation
        logger.error("Report generation not implemented")
        return False


# Factory function to create results model from different sources
def create_results_model(source: str, result_type: ResultType = ResultType.UNKNOWN, 
                        field_names: List[str] = None) -> Optional[ResultsModel]:
    """
    Create a results model from various sources.
    
    Args:
        source (str): Path to source (OpenFOAM case, results directory, etc.)
        result_type (ResultType, optional): Type of results to create
        field_names (List[str], optional): Fields to load, if None loads all
        
    Returns:
        Optional[ResultsModel]: New results model or None if creation failed
    """
    # Check if source exists
    if not os.path.exists(source):
        logger.error(f"Source not found: {source}")
        return None
    
    # Create new model
    model = ResultsModel(result_type=result_type)
    
    # Case 1: Load from existing results directory
    results_meta_path = os.path.join(source, 'metadata.json')
    if os.path.isdir(source) and os.path.isfile(results_meta_path):
        loaded_model = ResultsModel.load(source)
        if loaded_model:
            return loaded_model
    
    # Case 2: Load from OpenFOAM case
    system_path = os.path.join(source, 'system')
    constant_path = os.path.join(source, 'constant')
    if os.path.isdir(source) and os.path.isdir(system_path) and os.path.isdir(constant_path):
        # Looks like an OpenFOAM case
        success = model.load_openfoam_results(source, field_names)
        if success:
            return model
    
    # Failed to create from source
    logger.error(f"Failed to create results model from source: {source}")
    return None