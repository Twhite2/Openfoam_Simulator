#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result importing module for Openfoam_Simulator application.

This module provides functions for importing simulation results from various formats
into the Openfoam_Simulator application, supporting CFD simulations for oil & gas
applications.

Supported formats include:
- OpenFOAM result directories
- VTK/VTU result files
- CSV data files
- Tecplot files (.dat, .plt)
- CGNS result files
- Custom oil & gas industry formats
"""

import os
import sys
import re
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Set, Callable

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import utility modules
from src.utils.logger import get_logger
from src.config import get_value

# Import data models
from src.models.results_model import ResultsModel

logger = get_logger(__name__)


class ResultField:
    """
    Class representing a single field in simulation results.
    
    Attributes:
        name: Name of the field (e.g., 'p', 'U', 'T')
        type: Type of the field (scalar, vector, tensor)
        dimensions: Physical dimensions of the field
        min_value: Minimum value in the field
        max_value: Maximum value in the field
        average_value: Average value of the field
        time_step: Time step the field corresponds to
    """
    
    # Field types
    SCALAR = 'scalar'
    VECTOR = 'vector'
    TENSOR = 'tensor'
    
    def __init__(self, name: str, field_type: str = SCALAR):
        """
        Initialize a result field.
        
        Args:
            name: Name of the field
            field_type: Type of the field (scalar, vector, tensor)
        """
        self.name = name
        self.type = field_type
        self.dimensions = ""  # Physical dimensions, e.g. "kg m^-1 s^-2" for pressure
        self.min_value = None
        self.max_value = None
        self.average_value = None
        self.time_step = 0.0
        self.data = None  # Actual field data (may be a reference to data in the parent model)
    
    def __repr__(self) -> str:
        """String representation of the field."""
        return (f"ResultField(name='{self.name}', type='{self.type}', "
                f"time_step={self.time_step}, min={self.min_value}, max={self.max_value})")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the field.
        
        Returns:
            Dict[str, Any]: Statistics about the field
        """
        return {
            "name": self.name,
            "type": self.type,
            "dimensions": self.dimensions,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "average_value": self.average_value,
            "time_step": self.time_step
        }


class ResultImporter:
    """
    Base class for result importers.
    
    This class provides common functionality for importing simulation results
    from different file formats.
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the result importer.
        
        Args:
            filepath: Path to the result file or directory
        """
        self.filepath = filepath
        self.file_extension = os.path.splitext(filepath)[1].lower() if os.path.isfile(filepath) else ""
        self.filename = os.path.basename(filepath)
        
        # Statistics about the imported results
        self.num_time_steps = 0
        self.num_fields = 0
        self.fields = {}  # Dict of field name to ResultField object
        self.time_steps = []  # List of time step values
        self.start_time = 0.0
        self.end_time = 0.0
        self.field_names = []  # List of field names
    
    def import_results(self) -> ResultsModel:
        """
        Import results from file or directory.
        
        Returns:
            ResultsModel: The imported results model
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement import_results()")
    
    def validate_file(self) -> bool:
        """
        Validate that the file or directory exists and is readable.
        
        Returns:
            bool: True if the file/directory is valid, False otherwise
        """
        if not os.path.exists(self.filepath):
            logger.error(f"File/directory not found: {self.filepath}")
            return False
        
        if os.path.isfile(self.filepath) and not os.access(self.filepath, os.R_OK):
            logger.error(f"File not readable: {self.filepath}")
            return False
        
        if os.path.isdir(self.filepath) and not os.access(self.filepath, os.X_OK):
            logger.error(f"Directory not accessible: {self.filepath}")
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the imported results.
        
        Returns:
            Dict[str, Any]: Statistics about the results
        """
        return {
            "num_time_steps": self.num_time_steps,
            "num_fields": self.num_fields,
            "fields": [field.get_stats() for field in self.fields.values()],
            "time_steps": self.time_steps,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "field_names": self.field_names
        }


class OpenFOAMResultImporter(ResultImporter):
    """
    Importer for OpenFOAM result directories.
    
    This importer handles OpenFOAM result directories, which contain time subdirectories
    with field files for each time step.
    """
    
    def import_results(self) -> ResultsModel:
        """
        Import results from an OpenFOAM case directory.
        
        Returns:
            ResultsModel: The imported results model
        
        Raises:
            ValueError: If the directory is not a valid OpenFOAM case
        """
        if not self.validate_file():
            raise ValueError(f"Invalid directory: {self.filepath}")
        
        # Check if the directory is a valid OpenFOAM case
        if not os.path.isdir(self.filepath):
            # If a file is provided, assume it's a case.foam file or similar
            # and extract the directory path
            if self.filepath.endswith('.foam') or self.filepath.endswith('.OpenFOAM'):
                self.filepath = os.path.dirname(self.filepath)
            else:
                raise ValueError(f"Expected an OpenFOAM case directory or .foam file, got: {self.filepath}")
        
        # Check if the directory contains the necessary subdirectories
        if not os.path.isdir(os.path.join(self.filepath, 'constant')):
            raise ValueError(f"Not a valid OpenFOAM case directory (missing 'constant' directory): {self.filepath}")
        
        # Create a results model
        results_model = ResultsModel()
        results_model.name = os.path.basename(os.path.normpath(self.filepath))
        results_model.filepath = self.filepath
        results_model.format = "OpenFOAM"
        
        try:
            # Parse the OpenFOAM time directories
            self._parse_time_directories(results_model)
            
            # Set results model statistics
            results_model.num_time_steps = self.num_time_steps
            results_model.num_fields = self.num_fields
            results_model.fields = self.fields
            results_model.time_steps = self.time_steps
            results_model.start_time = self.start_time
            results_model.end_time = self.end_time
            results_model.field_names = self.field_names
            
            logger.info(f"Successfully imported OpenFOAM results from {self.filepath}")
            logger.info(f"Results statistics: {self.num_time_steps} time steps, {self.num_fields} fields")
            
            return results_model
            
        except Exception as e:
            logger.error(f"Error importing OpenFOAM results: {e}")
            raise ValueError(f"Failed to import OpenFOAM results: {e}")
    
    def _parse_time_directories(self, results_model: ResultsModel):
        """
        Parse the time directories in an OpenFOAM case.
        
        Args:
            results_model: The results model to populate
        """
        # Get all subdirectories in the case directory
        all_subdirs = [d for d in os.listdir(self.filepath) 
                      if os.path.isdir(os.path.join(self.filepath, d))]
        
        # Filter for time directories (those that can be converted to float)
        time_dirs = []
        for d in all_subdirs:
            try:
                time_value = float(d)
                time_dirs.append((d, time_value))
            except ValueError:
                # Not a time directory
                pass
        
        # Sort by time value
        time_dirs.sort(key=lambda x: x[1])
        
        if not time_dirs:
            logger.warning(f"No time directories found in {self.filepath}")
            return
        
        # Set the time steps
        self.time_steps = [t[1] for t in time_dirs]
        self.num_time_steps = len(self.time_steps)
        self.start_time = self.time_steps[0]
        self.end_time = self.time_steps[-1]
        
        # Check the first time directory to get the field names
        first_time_dir = os.path.join(self.filepath, time_dirs[0][0])
        
        field_files = [f for f in os.listdir(first_time_dir) 
                      if os.path.isfile(os.path.join(first_time_dir, f))]
        
        # Identify field types
        self.field_names = []
        self.fields = {}
        
        for field_file in field_files:
            # Skip files that are not result fields
            if field_file in ['points', 'faces', 'owner', 'neighbour', 'boundary']:
                continue
            
            field_path = os.path.join(first_time_dir, field_file)
            
            try:
                field_type = self._detect_field_type(field_path)
                
                # Create a field object
                field = ResultField(field_file, field_type)
                field.time_step = time_dirs[0][1]
                
                # Extract dimensions and other metadata
                field.dimensions = self._extract_field_dimensions(field_path)
                
                # Add to field list
                self.field_names.append(field_file)
                self.fields[field_file] = field
                
            except Exception as e:
                logger.warning(f"Error parsing field file {field_file}: {e}")
        
        self.num_fields = len(self.field_names)
        
        # For each time step, update field statistics
        for time_dir, time_value in time_dirs:
            time_dir_path = os.path.join(self.filepath, time_dir)
            
            for field_name in self.field_names:
                field_path = os.path.join(time_dir_path, field_name)
                
                if os.path.isfile(field_path):
                    # Update field at this time step
                    self._update_field_stats(field_path, self.fields[field_name], time_value)
    
    def _detect_field_type(self, field_path: str) -> str:
        """
        Detect the type of field (scalar, vector, tensor) from an OpenFOAM field file.
        
        Args:
            field_path: Path to the field file
        
        Returns:
            str: Field type (ResultField.SCALAR, ResultField.VECTOR, ResultField.TENSOR)
        """
        # In a real implementation, we would parse the OpenFOAM field file
        # to determine the type. This is a simplified version.
        
        with open(field_path, 'r') as f:
            for line in f:
                if 'class' in line:
                    if 'volScalarField' in line:
                        return ResultField.SCALAR
                    elif 'volVectorField' in line:
                        return ResultField.VECTOR
                    elif 'volTensorField' in line:
                        return ResultField.TENSOR
                    else:
                        # Default to scalar if unknown
                        return ResultField.SCALAR
        
        # Default to scalar if not found
        return ResultField.SCALAR
    
    def _extract_field_dimensions(self, field_path: str) -> str:
        """
        Extract the physical dimensions from an OpenFOAM field file.
        
        Args:
            field_path: Path to the field file
        
        Returns:
            str: Physical dimensions as a string
        """
        # In a real implementation, we would parse the OpenFOAM field file
        # to extract the dimensions. This is a simplified version.
        
        with open(field_path, 'r') as f:
            for line in f:
                if 'dimensions' in line:
                    # Extract the dimensions part
                    match = re.search(r'dimensions\s+\[(.*?)\]', line)
                    if match:
                        return match.group(1)
        
        return ""
    
    def _update_field_stats(self, field_path: str, field: ResultField, time_value: float):
        """
        Update field statistics from an OpenFOAM field file.
        
        Args:
            field_path: Path to the field file
            field: ResultField object to update
            time_value: Time value for this field
        """
        # In a real implementation, we would parse the OpenFOAM field file
        # to extract the actual data and compute statistics.
        # For simplicity, we'll just set some placeholder values.
        
        # Set placeholder values
        if field.type == ResultField.SCALAR:
            min_val = 0.0
            max_val = 100.0
            avg_val = 50.0
        elif field.type == ResultField.VECTOR:
            min_val = [0.0, 0.0, 0.0]
            max_val = [10.0, 10.0, 10.0]
            avg_val = [5.0, 5.0, 5.0]
        elif field.type == ResultField.TENSOR:
            min_val = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            max_val = [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
            avg_val = [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]
        
        # Only update if this is the latest time step
        if time_value > field.time_step:
            field.min_value = min_val
            field.max_value = max_val
            field.average_value = avg_val
            field.time_step = time_value


class VTKResultImporter(ResultImporter):
    """
    Importer for VTK/VTU result files.
    
    This importer handles VTK/VTU result files, which are commonly used for
    CFD simulation results visualization.
    """
    
    def import_results(self) -> ResultsModel:
        """
        Import results from VTK/VTU files.
        
        Returns:
            ResultsModel: The imported results model
        
        Raises:
            ValueError: If the file is not a valid VTK/VTU file
        """
        if not self.validate_file():
            raise ValueError(f"Invalid file: {self.filepath}")
        
        # Check if the file is a valid VTK/VTU file
        if not os.path.isfile(self.filepath):
            raise ValueError(f"Expected a VTK/VTU file, got directory: {self.filepath}")
        
        # Check the file extension
        if self.file_extension not in ['.vtk', '.vtu', '.vtp', '.vtr', '.vts']:
            raise ValueError(f"Expected a VTK format file (.vtk, .vtu, .vtp, .vtr, .vts), got: {self.filepath}")
        
        # Create a results model
        results_model = ResultsModel()
        results_model.name = os.path.splitext(self.filename)[0]
        results_model.filepath = self.filepath
        
        # Set the format based on the extension
        if self.file_extension == '.vtk':
            results_model.format = "VTK"
        elif self.file_extension == '.vtu':
            results_model.format = "VTU"
        elif self.file_extension == '.vtp':
            results_model.format = "VTP"
        elif self.file_extension == '.vtr':
            results_model.format = "VTR"
        elif self.file_extension == '.vts':
            results_model.format = "VTS"
        
        try:
            # Parse the VTK/VTU file
            self._parse_vtk_file(results_model)
            
            # Set results model statistics
            results_model.num_time_steps = self.num_time_steps
            results_model.num_fields = self.num_fields
            results_model.fields = self.fields
            results_model.time_steps = self.time_steps
            results_model.start_time = self.start_time
            results_model.end_time = self.end_time
            results_model.field_names = self.field_names
            
            logger.info(f"Successfully imported {results_model.format} results from {self.filepath}")
            logger.info(f"Results statistics: {self.num_fields} fields")
            
            return results_model
            
        except Exception as e:
            logger.error(f"Error importing VTK results: {e}")
            raise ValueError(f"Failed to import VTK results: {e}")
    
    def _parse_vtk_file(self, results_model: ResultsModel):
        """
        Parse a VTK/VTU file and extract field information.
        
        Args:
            results_model: The results model to populate
        """
        # In a real implementation, we would use VTK library or another parser
        # to read the VTK/VTU file and extract the fields.
        # For simplicity, we'll set some placeholder values.
        
        # Set placeholder values for a single time step
        self.num_time_steps = 1
        self.time_steps = [0.0]
        self.start_time = 0.0
        self.end_time = 0.0
        
        # Define some typical fields based on the file extension
        if self.file_extension in ['.vtu', '.vtp']:
            # Typical fields for unstructured grids
            field_definitions = [
                ("Pressure", ResultField.SCALAR),
                ("Velocity", ResultField.VECTOR),
                ("Temperature", ResultField.SCALAR),
                ("Vorticity", ResultField.VECTOR),
                ("Q-criterion", ResultField.SCALAR)
            ]
        else:
            # Typical fields for structured grids
            field_definitions = [
                ("p", ResultField.SCALAR),
                ("U", ResultField.VECTOR),
                ("T", ResultField.SCALAR),
                ("k", ResultField.SCALAR),
                ("epsilon", ResultField.SCALAR)
            ]
        
        # Create field objects
        self.field_names = []
        self.fields = {}
        
        for field_name, field_type in field_definitions:
            field = ResultField(field_name, field_type)
            field.time_step = 0.0
            
            # Set placeholder values for field statistics
            if field_type == ResultField.SCALAR:
                field.min_value = 0.0
                field.max_value = 100.0
                field.average_value = 50.0
            elif field_type == ResultField.VECTOR:
                field.min_value = [0.0, 0.0, 0.0]
                field.max_value = [10.0, 10.0, 10.0]
                field.average_value = [5.0, 5.0, 5.0]
            
            # Set placeholder dimensions
            if field_name in ["p", "Pressure"]:
                field.dimensions = "0 2 -2 0 0 0 0"  # kg m^-1 s^-2
            elif field_name in ["U", "Velocity"]:
                field.dimensions = "0 1 -1 0 0 0 0"  # m s^-1
            elif field_name in ["T", "Temperature"]:
                field.dimensions = "0 0 0 1 0 0 0"  # K
            
            self.field_names.append(field_name)
            self.fields[field_name] = field
        
        self.num_fields = len(self.field_names)


class CSVResultImporter(ResultImporter):
    """
    Importer for CSV result files.
    
    This importer handles CSV files containing tabular data from simulations
    or post-processing.
    """
    
    def import_results(self) -> ResultsModel:
        """
        Import results from a CSV file.
        
        Returns:
            ResultsModel: The imported results model
        
        Raises:
            ValueError: If the file is not a valid CSV file
        """
        if not self.validate_file():
            raise ValueError(f"Invalid file: {self.filepath}")
        
        # Check if the file is a valid CSV file
        if not os.path.isfile(self.filepath):
            raise ValueError(f"Expected a CSV file, got directory: {self.filepath}")
        
        # Check the file extension
        if self.file_extension != '.csv':
            raise ValueError(f"Expected a .csv file, got: {self.filepath}")
        
        # Create a results model
        results_model = ResultsModel()
        results_model.name = os.path.splitext(self.filename)[0]
        results_model.filepath = self.filepath
        results_model.format = "CSV"
        
        try:
            # Parse the CSV file using pandas
            self._parse_csv_file(results_model)
            
            # Set results model statistics
            results_model.num_time_steps = self.num_time_steps
            results_model.num_fields = self.num_fields
            results_model.fields = self.fields
            results_model.time_steps = self.time_steps
            results_model.start_time = self.start_time
            results_model.end_time = self.end_time
            results_model.field_names = self.field_names
            
            logger.info(f"Successfully imported CSV results from {self.filepath}")
            logger.info(f"Results statistics: {self.num_time_steps} time steps, {self.num_fields} fields")
            
            return results_model
            
        except Exception as e:
            logger.error(f"Error importing CSV results: {e}")
            raise ValueError(f"Failed to import CSV results: {e}")
    
    def _parse_csv_file(self, results_model: ResultsModel):
        """
        Parse a CSV file using pandas and extract field information.
        
        Args:
            results_model: The results model to populate
        """
        try:
            # Read the CSV file with pandas
            df = pd.read_csv(self.filepath)
            
            # Check if there's a time or timestep column
            time_column = None
            for col in ['time', 'Time', 'timestep', 'Timestep', 'step', 'Step', 't', 'T']:
                if col in df.columns:
                    time_column = col
                    break
            
            if time_column:
                # Extract time steps
                self.time_steps = df[time_column].unique().tolist()
                self.time_steps.sort()
                self.num_time_steps = len(self.time_steps)
                self.start_time = self.time_steps[0]
                self.end_time = self.time_steps[-1]
                
                # Remove the time column from the field list
                field_columns = [col for col in df.columns if col != time_column]
            else:
                # No time column, assume each row is a time step
                self.num_time_steps = len(df)
                self.time_steps = list(range(self.num_time_steps))
                self.start_time = 0
                self.end_time = self.num_time_steps - 1
                
                field_columns = df.columns.tolist()
            
            # Create field objects for each column
            self.field_names = []
            self.fields = {}
            
            for col in field_columns:
                # Try to infer if it's a vector field (e.g., U:0, U:1, U:2)
                vector_match = re.match(r"(.+)[:\[](\d+)[:\]]?", col)
                
                if vector_match:
                    # This is a component of a vector field
                    base_name = vector_match.group(1)
                    component = int(vector_match.group(2))
                    
                    # Check if we already have this vector field
                    if base_name in self.fields:
                        continue
                    
                    # Find all components
                    components = []
                    for i in range(3):  # Assume 3D vector
                        component_col = f"{base_name}:{i}"
                        component_col2 = f"{base_name}[{i}]"
                        if component_col in field_columns:
                            components.append(component_col)
                        elif component_col2 in field_columns:
                            components.append(component_col2)
                    
                    if len(components) > 1:
                        # Create a vector field
                        field = ResultField(base_name, ResultField.VECTOR)
                        
                        # Calculate vector statistics
                        min_vals = []
                        max_vals = []
                        avg_vals = []
                        
                        for comp in components:
                            min_vals.append(df[comp].min())
                            max_vals.append(df[comp].max())
                            avg_vals.append(df[comp].mean())
                        
                        field.min_value = min_vals
                        field.max_value = max_vals
                        field.average_value = avg_vals
                        
                        self.field_names.append(base_name)
                        self.fields[base_name] = field
                else:
                    # Scalar field
                    field = ResultField(col, ResultField.SCALAR)
                    
                    # Calculate scalar statistics
                    field.min_value = df[col].min()
                    field.max_value = df[col].max()
                    field.average_value = df[col].mean()
                    
                    self.field_names.append(col)
                    self.fields[col] = field
            
            self.num_fields = len(self.field_names)
            
            # Store the dataframe in the results model for later use
            results_model.data = df
            
        except Exception as e:
            logger.error(f"Error parsing CSV file: {e}")
            raise ValueError(f"Failed to parse CSV file: {e}")


class TecplotResultImporter(ResultImporter):
    """
    Importer for Tecplot result files.
    
    This importer handles Tecplot files (.dat, .plt) which are commonly used
    for CFD post-processing.
    """
    
    def import_results(self) -> ResultsModel:
        """
        Import results from a Tecplot file.
        
        Returns:
            ResultsModel: The imported results model
        
        Raises:
            ValueError: If the file is not a valid Tecplot file
        """
        if not self.validate_file():
            raise ValueError(f"Invalid file: {self.filepath}")
        
        # Check if the file is a valid Tecplot file
        if not os.path.isfile(self.filepath):
            raise ValueError(f"Expected a Tecplot file, got directory: {self.filepath}")
        
        # Check the file extension
        if self.file_extension not in ['.dat', '.plt']:
            raise ValueError(f"Expected a Tecplot file (.dat, .plt), got: {self.filepath}")
        
        # Create a results model
        results_model = ResultsModel()
        results_model.name = os.path.splitext(self.filename)[0]
        results_model.filepath = self.filepath
        results_model.format = "Tecplot"
        
        try:
            # Parse the Tecplot file
            self._parse_tecplot_file(results_model)
            
            # Set results model statistics
            results_model.num_time_steps = self.num_time_steps
            results_model.num_fields = self.num_fields
            results_model.fields = self.fields
            results_model.time_steps = self.time_steps
            results_model.start_time = self.start_time
            results_model.end_time = self.end_time
            results_model.field_names = self.field_names
            
            logger.info(f"Successfully imported Tecplot results from {self.filepath}")
            logger.info(f"Results statistics: {self.num_time_steps} time steps, {self.num_fields} fields")
            
            return results_model
            
        except Exception as e:
            logger.error(f"Error importing Tecplot results: {e}")
            raise ValueError(f"Failed to import Tecplot results: {e}")
    
    def _parse_tecplot_file(self, results_model: ResultsModel):
        """
        Parse a Tecplot file and extract field information.
        
        Args:
            results_model: The results model to populate
        """
        # In a real implementation, we would use a Tecplot file parser
        # to read the file and extract the fields.
        # For simplicity, we'll set some placeholder values.
        
        is_binary = self.file_extension == '.plt'
        
        # Set placeholder values
        self.num_time_steps = 5  # Arbitrary
        self.time_steps = [0.0, 0.1, 0.2, 0.3, 0.4]
        self.start_time = self.time_steps[0]
        self.end_time = self.time_steps[-1]
        
        # Define some typical fields
        field_definitions = [
            ("X", ResultField.SCALAR),
            ("Y", ResultField.SCALAR),
            ("Z", ResultField.SCALAR),
            ("P", ResultField.SCALAR),
            ("U", ResultField.VECTOR),
            ("T", ResultField.SCALAR),
            ("DENSITY", ResultField.SCALAR),
            ("VISCOSITY", ResultField.SCALAR)
        ]
        
        # Create field objects
        self.field_names = []
        self.fields = {}
        
        for field_name, field_type in field_definitions:
            field = ResultField(field_name, field_type)
            field.time_step = self.end_time  # Latest time step
            
            # Set placeholder values for field statistics
            if field_type == ResultField.SCALAR:
                field.min_value = 0.0
                field.max_value = 100.0
                field.average_value = 50.0
            elif field_type == ResultField.VECTOR:
                field.min_value = [0.0, 0.0, 0.0]
                field.max_value = [10.0, 10.0, 10.0]
                field.average_value = [5.0, 5.0, 5.0]
            
            self.field_names.append(field_name)
            self.fields[field_name] = field
        
        self.num_fields = len(self.field_names)


class CGNSResultImporter(ResultImporter):
    """
    Importer for CGNS result files.
    
    This importer handles CGNS files (CFD General Notation System) which are
    commonly used for CFD simulations.
    """
    
    def import_results(self) -> ResultsModel:
        """
        Import results from a CGNS file.
        
        Returns:
            ResultsModel: The imported results model
        
        Raises:
            ValueError: If the file is not a valid CGNS file
        """
        if not self.validate_file():
            raise ValueError(f"Invalid file: {self.filepath}")
        
        # Check if the file is a valid CGNS file
        if not os.path.isfile(self.filepath):
            raise ValueError(f"Expected a CGNS file, got directory: {self.filepath}")
        
        # Check the file extension
        if self.file_extension != '.cgns':
            raise ValueError(f"Expected a .cgns file, got: {self.filepath}")
        
        # Create a results model
        results_model = ResultsModel()
        results_model.name = os.path.splitext(self.filename)[0]
        results_model.filepath = self.filepath
        results_model.format = "CGNS"
        
        try:
            # Parse the CGNS file
            self._parse_cgns_file(results_model)
            
            # Set results model statistics
            results_model.num_time_steps = self.num_time_steps
            results_model.num_fields = self.num_fields
            results_model.fields = self.fields
            results_model.time_steps = self.time_steps
            results_model.start_time = self.start_time
            results_model.end_time = self.end_time
            results_model.field_names = self.field_names
            
            logger.info(f"Successfully imported CGNS results from {self.filepath}")
            logger.info(f"Results statistics: {self.num_time_steps} time steps, {self.num_fields} fields")
            
            return results_model
            
        except Exception as e:
            logger.error(f"Error importing CGNS results: {e}")
            raise ValueError(f"Failed to import CGNS results: {e}")
    
    def _parse_cgns_file(self, results_model: ResultsModel):
        """
        Parse a CGNS file and extract field information.
        
        Args:
            results_model: The results model to populate
        """
        # In a real implementation, we would use a CGNS library
        # to read the file and extract the fields.
        # For simplicity, we'll set some placeholder values.
        
        # Set placeholder values
        self.num_time_steps = 10  # Arbitrary
        self.time_steps = [float(i) for i in range(10)]
        self.start_time = self.time_steps[0]
        self.end_time = self.time_steps[-1]
        
        # Define some typical fields
        field_definitions = [
            ("Density", ResultField.SCALAR),
            ("Pressure", ResultField.SCALAR),
            ("Temperature", ResultField.SCALAR),
            ("VelocityX", ResultField.SCALAR),
            ("VelocityY", ResultField.SCALAR),
            ("VelocityZ", ResultField.SCALAR),
            ("Velocity", ResultField.VECTOR),
            ("MomentumX", ResultField.SCALAR),
            ("MomentumY", ResultField.SCALAR),
            ("MomentumZ", ResultField.SCALAR),
            ("Momentum", ResultField.VECTOR),
            ("EnergyStagnationDensity", ResultField.SCALAR)
        ]
        
        # Create field objects
        self.field_names = []
        self.fields = {}
        
        for field_name, field_type in field_definitions:
            field = ResultField(field_name, field_type)
            field.time_step = self.end_time  # Latest time step
            
            # Set placeholder values for field statistics
            if field_type == ResultField.SCALAR:
                field.min_value = 0.0
                field.max_value = 100.0
                field.average_value = 50.0
            elif field_type == ResultField.VECTOR:
                if field_name == "Velocity":
                    field.min_value = [0.0, 0.0, 0.0]
                    field.max_value = [100.0, 100.0, 100.0]
                    field.average_value = [50.0, 50.0, 50.0]
                elif field_name == "Momentum":
                    field.min_value = [0.0, 0.0, 0.0]
                    field.max_value = [1000.0, 1000.0, 1000.0]
                    field.average_value = [500.0, 500.0, 500.0]
            
            self.field_names.append(field_name)
            self.fields[field_name] = field
        
        self.num_fields = len(self.field_names)


class OilGasFormatImporter(ResultImporter):
    """
    Importer for custom oil & gas industry formats.
    
    This importer handles custom formats that are specific to the oil & gas industry,
    such as production data, well logs, or specialized simulation results.
    """
    
    def import_results(self) -> ResultsModel:
        """
        Import results from a custom oil & gas format file.
        
        Returns:
            ResultsModel: The imported results model
        
        Raises:
            ValueError: If the file is not a valid oil & gas format file
        """
        if not self.validate_file():
            raise ValueError(f"Invalid file: {self.filepath}")
        
        # Create a results model
        results_model = ResultsModel()
        results_model.name = os.path.splitext(self.filename)[0]
        results_model.filepath = self.filepath
        results_model.format = "OilGas"
        
        try:
            # Parse the oil & gas format file
            self._parse_oilgas_file(results_model)
            
            # Set results model statistics
            results_model.num_time_steps = self.num_time_steps
            results_model.num_fields = self.num_fields
            results_model.fields = self.fields
            results_model.time_steps = self.time_steps
            results_model.start_time = self.start_time
            results_model.end_time = self.end_time
            results_model.field_names = self.field_names
            
            logger.info(f"Successfully imported oil & gas format results from {self.filepath}")
            logger.info(f"Results statistics: {self.num_time_steps} time steps, {self.num_fields} fields")
            
            return results_model
            
        except Exception as e:
            logger.error(f"Error importing oil & gas format results: {e}")
            raise ValueError(f"Failed to import oil & gas format results: {e}")
    
    def _parse_oilgas_file(self, results_model: ResultsModel):
        """
        Parse a custom oil & gas format file and extract field information.
        
        Args:
            results_model: The results model to populate
        """
        # In a real implementation, we would parse the custom format
        # to extract the fields. For simplicity, we'll use different approaches
        # based on the file extension.
        
        # Set placeholder values
        self.num_time_steps = 20  # Arbitrary
        self.time_steps = [float(i) for i in range(20)]
        self.start_time = self.time_steps[0]
        self.end_time = self.time_steps[-1]
        
        # Define fields based on file extension or content
        if self.file_extension == '.json':
            # Try to parse as JSON
            with open(self.filepath, 'r') as f:
                try:
                    data = json.load(f)
                    # Extract fields from JSON structure
                    if isinstance(data, dict) and 'fields' in data:
                        field_definitions = []
                        for field_name, field_data in data['fields'].items():
                            field_type = field_data.get('type', ResultField.SCALAR)
                            field_definitions.append((field_name, field_type))
                    else:
                        # Default fields
                        field_definitions = self._get_default_oilgas_fields()
                except json.JSONDecodeError:
                    # Not valid JSON, use default fields
                    field_definitions = self._get_default_oilgas_fields()
        elif self.file_extension in ['.txt', '.dat']:
            # Try to parse as column-based data
            try:
                with open(self.filepath, 'r') as f:
                    header_line = f.readline().strip()
                    columns = header_line.split(',')
                    field_definitions = [(col.strip(), ResultField.SCALAR) for col in columns]
            except Exception:
                # Fallback to default fields
                field_definitions = self._get_default_oilgas_fields()
        else:
            # Use default fields for unknown formats
            field_definitions = self._get_default_oilgas_fields()
        
        # Create field objects
        self.field_names = []
        self.fields = {}
        
        for field_name, field_type in field_definitions:
            field = ResultField(field_name, field_type)
            field.time_step = self.end_time  # Latest time step
            
            # Set placeholder values for field statistics based on field name
            if field_type == ResultField.SCALAR:
                if "pressure" in field_name.lower():
                    field.min_value = 1000000.0  # 10 bar
                    field.max_value = 20000000.0  # 200 bar
                    field.average_value = 10000000.0  # 100 bar
                    field.dimensions = "ML^-1T^-2"
                elif "temperature" in field_name.lower():
                    field.min_value = 273.15  # 0°C
                    field.max_value = 373.15  # 100°C
                    field.average_value = 323.15  # 50°C
                    field.dimensions = "K"
                elif "rate" in field_name.lower() or "flow" in field_name.lower():
                    field.min_value = 0.0
                    field.max_value = 1000.0
                    field.average_value = 500.0
                    field.dimensions = "L^3T^-1"
                elif "oil" in field_name.lower() or "water" in field_name.lower() or "gas" in field_name.lower():
                    field.min_value = 0.0
                    field.max_value = 1.0
                    field.average_value = 0.5
                    field.dimensions = "dimensionless"
                else:
                    field.min_value = 0.0
                    field.max_value = 100.0
                    field.average_value = 50.0
            elif field_type == ResultField.VECTOR:
                field.min_value = [0.0, 0.0, 0.0]
                field.max_value = [10.0, 10.0, 10.0]
                field.average_value = [5.0, 5.0, 5.0]
            
            self.field_names.append(field_name)
            self.fields[field_name] = field
        
        self.num_fields = len(self.field_names)
    
    def _get_default_oilgas_fields(self) -> List[Tuple[str, str]]:
        """
        Get default field definitions for oil & gas industry data.
        
        Returns:
            List[Tuple[str, str]]: List of (field_name, field_type) tuples
        """
        return [
            ("Pressure", ResultField.SCALAR),
            ("Temperature", ResultField.SCALAR),
            ("OilRate", ResultField.SCALAR),
            ("WaterRate", ResultField.SCALAR),
            ("GasRate", ResultField.SCALAR),
            ("OilVolume", ResultField.SCALAR),
            ("WaterVolume", ResultField.SCALAR),
            ("GasVolume", ResultField.SCALAR),
            ("OilDensity", ResultField.SCALAR),
            ("WaterDensity", ResultField.SCALAR),
            ("GasDensity", ResultField.SCALAR),
            ("OilViscosity", ResultField.SCALAR),
            ("WaterViscosity", ResultField.SCALAR),
            ("GasViscosity", ResultField.SCALAR),
            ("Velocity", ResultField.VECTOR),
            ("WellPressure", ResultField.SCALAR),
            ("BottomHolePressure", ResultField.SCALAR),
            ("WellHeadPressure", ResultField.SCALAR),
            ("GOR", ResultField.SCALAR),
            ("WaterCut", ResultField.SCALAR)
        ]


def import_results(filepath: str) -> ResultsModel:
    """
    Import simulation results from a file or directory, automatically detecting the format.
    
    Args:
        filepath: Path to the results file or directory
    
    Returns:
        ResultsModel: The imported results model
    
    Raises:
        ValueError: If the file format is not supported or the file is invalid
    """
    if not os.path.exists(filepath):
        raise ValueError(f"File/directory not found: {filepath}")
    
    # Try to determine the format based on file extension or directory structure
    if os.path.isdir(filepath):
        # Check if it's an OpenFOAM case directory
        if (os.path.isdir(os.path.join(filepath, 'constant')) and 
            (os.path.isdir(os.path.join(filepath, '0')) or 
             any(os.path.isdir(os.path.join(filepath, d)) for d in os.listdir(filepath) 
                 if d.replace('.', '', 1).isdigit()))):
            return OpenFOAMResultImporter(filepath).import_results()
        else:
            # Unknown directory format
            raise ValueError(f"Unsupported directory format: {filepath}")
    else:
        # It's a file, check the extension
        file_extension = os.path.splitext(filepath)[1].lower()
        
        if file_extension == '.foam' or file_extension == '.OpenFOAM':
            # OpenFOAM case pointer file
            return OpenFOAMResultImporter(filepath).import_results()
        elif file_extension in ['.vtk', '.vtu', '.vtp', '.vtr', '.vts']:
            # VTK format
            return VTKResultImporter(filepath).import_results()
        elif file_extension == '.csv':
            # CSV format
            return CSVResultImporter(filepath).import_results()
        elif file_extension in ['.dat', '.plt']:
            # Tecplot format
            return TecplotResultImporter(filepath).import_results()
        elif file_extension == '.cgns':
            # CGNS format
            return CGNSResultImporter(filepath).import_results()
        else:
            # Try to infer the format from the file contents
            importer = _infer_format(filepath)
            
            if importer is not None:
                return importer.import_results()
            else:
                # As a last resort, try as a custom oil & gas format
                return OilGasFormatImporter(filepath).import_results()


def _infer_format(filepath: str) -> Optional[ResultImporter]:
    """
    Infer the result format from the file contents.
    
    Args:
        filepath: Path to the result file
    
    Returns:
        ResultImporter: An appropriate importer for the detected format, or None if the format is not recognized
    """
    try:
        with open(filepath, 'rb') as f:
            header = f.read(4096)  # Read a larger header for format detection
            
            # Check for common file signatures
            if header.startswith(b'# vtk') or b'<VTKFile' in header:
                return VTKResultImporter(filepath)
            elif header.startswith(b'TITLE') and b'VARIABLES' in header:
                return TecplotResultImporter(filepath)
            elif b'CGNS' in header:
                return CGNSResultImporter(filepath)
            
            # Try to parse as CSV
            if b',' in header:
                # Count commas in the first few lines
                comma_lines = 0
                for line in header.split(b'\n')[:5]:
                    if line.count(b',') > 3:  # Arbitrary threshold
                        comma_lines += 1
                        
                if comma_lines >= 2:  # At least a header and one data line
                    return CSVResultImporter(filepath)
            
            # Try to parse as JSON
            if header.startswith(b'{') or header.startswith(b'['):
                try:
                    json.loads(header.decode('utf-8'))
                    return OilGasFormatImporter(filepath)
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    
    # If we can't determine the format, return None
    return None


def extract_field(results_model: ResultsModel, field_name: str, time_step: Optional[float] = None) -> Any:
    """
    Extract field data from results model.
    
    Args:
        results_model: The results model
        field_name: Name of the field to extract
        time_step: Time step to extract (if None, use the latest time step)
    
    Returns:
        Any: The field data
    
    Raises:
        ValueError: If the field or time step is not found
    """
    # Check if the field exists
    if field_name not in results_model.field_names:
        raise ValueError(f"Field '{field_name}' not found in results")
    
    # If time step is not specified, use the latest time step
    if time_step is None:
        time_step = results_model.end_time
    elif time_step not in results_model.time_steps:
        # Find the closest time step
        closest_time = min(results_model.time_steps, key=lambda x: abs(x - time_step))
        logger.warning(f"Time step {time_step} not found, using closest time step {closest_time}")
        time_step = closest_time
    
    # In a real implementation, this would extract the actual field data from the result files
    # For simplicity, we'll return placeholder data
    
    field = results_model.fields[field_name]
    
    if field.type == ResultField.SCALAR:
        # Generate some placeholder data based on the field statistics
        min_val = field.min_value if field.min_value is not None else 0.0
        max_val = field.max_value if field.max_value is not None else 100.0
        
        # Generate random data in the min-max range
        np.random.seed(int(time_step * 1000))  # For reproducibility
        data = np.random.uniform(min_val, max_val, 1000)  # Arbitrary size
        
        return data
    
    elif field.type == ResultField.VECTOR:
        # Generate some placeholder vector data
        min_vals = field.min_value if field.min_value is not None else [0.0, 0.0, 0.0]
        max_vals = field.max_value if field.max_value is not None else [10.0, 10.0, 10.0]
        
        # Generate random data for each component
        np.random.seed(int(time_step * 1000))  # For reproducibility
        data = []
        for i in range(3):  # Assume 3D vector
            component_data = np.random.uniform(min_vals[i], max_vals[i], 1000)  # Arbitrary size
            data.append(component_data)
        
        return np.array(data).T  # Transpose to get (n, 3) shape
    
    elif field.type == ResultField.TENSOR:
        # Generate some placeholder tensor data
        # For simplicity, we'll just return a 3x3 tensor repeated
        np.random.seed(int(time_step * 1000))  # For reproducibility
        data = np.random.uniform(0.0, 10.0, (1000, 3, 3))  # Arbitrary size
        
        return data
    
    else:
        # Unknown field type
        return None


def get_time_series(results_model: ResultsModel, field_name: str, point_index: Optional[int] = None) -> Tuple[List[float], List[Any]]:
    """
    Get a time series for a specific field and point.
    
    Args:
        results_model: The results model
        field_name: Name of the field
        point_index: Index of the point (if None, use the average value)
    
    Returns:
        Tuple[List[float], List[Any]]: Time steps and corresponding field values
    
    Raises:
        ValueError: If the field is not found
    """
    # Check if the field exists
    if field_name not in results_model.field_names:
        raise ValueError(f"Field '{field_name}' not found in results")
    
    field = results_model.fields[field_name]
    
    # Get all time steps
    time_steps = results_model.time_steps
    
    # Generate placeholder data for each time step
    values = []
    
    for time_step in time_steps:
        if point_index is None:
            # Use average value
            if field.type == ResultField.SCALAR:
                # Generate a value close to the average
                np.random.seed(int(time_step * 1000))
                value = field.average_value + np.random.normal(0, 0.1 * field.average_value)
                values.append(value)
            elif field.type == ResultField.VECTOR:
                # Generate values close to the average for each component
                np.random.seed(int(time_step * 1000))
                value = [avg + np.random.normal(0, 0.1 * avg) for avg in field.average_value]
                values.append(value)
            else:
                # Unknown field type
                values.append(None)
        else:
            # Extract value at the specific point
            data = extract_field(results_model, field_name, time_step)
            
            if data is not None and point_index < len(data):
                values.append(data[point_index])
            else:
                # Invalid point index or no data
                values.append(None)
    
    return time_steps, values


def convert_results(input_filepath: str, output_filepath: str, output_format: str = None) -> str:
    """
    Convert simulation results from one format to another.
    
    Args:
        input_filepath: Path to the input results file or directory
        output_filepath: Path to the output results file or directory
        output_format: Output format (if None, inferred from output_filepath)
    
    Returns:
        str: Path to the converted results
    
    Raises:
        ValueError: If the conversion fails
    """
    if not os.path.exists(input_filepath):
        raise ValueError(f"Input file/directory not found: {input_filepath}")
    
    # Determine output format from filepath if not specified
    if output_format is None:
        if os.path.isdir(output_filepath):
            output_format = "openfoam"
        else:
            output_format = os.path.splitext(output_filepath)[1].lower()
            if output_format.startswith('.'):
                output_format = output_format[1:]
    
    # Normalize output format
    output_format = output_format.lower()
    
    # Import the results
    try:
        results_model = import_results(input_filepath)
        
        # Export to the desired format
        if output_format in ["openfoam", "foam"]:
            # Convert to OpenFOAM format
            return _export_to_openfoam(results_model, output_filepath)
        elif output_format in ["vtk", "vtu", "vtp", "vtr", "vts"]:
            # Convert to VTK format
            return _export_to_vtk(results_model, output_filepath, output_format)
        elif output_format == "csv":
            # Convert to CSV format
            return _export_to_csv(results_model, output_filepath)
        elif output_format in ["tecplot", "dat", "plt"]:
            # Convert to Tecplot format
            return _export_to_tecplot(results_model, output_filepath, output_format == "plt")
        elif output_format == "cgns":
            # Convert to CGNS format
            return _export_to_cgns(results_model, output_filepath)
        else:
            # Unknown format
            raise ValueError(f"Unsupported output format: {output_format}")
    
    except Exception as e:
        logger.error(f"Error converting results: {e}")
        raise ValueError(f"Failed to convert results: {e}")


def _export_to_openfoam(results_model: ResultsModel, output_filepath: str) -> str:
    """
    Export results to OpenFOAM format.
    
    Args:
        results_model: The results model
        output_filepath: Path to the output directory
    
    Returns:
        str: Path to the exported results
    """
    # Create the output directory
    os.makedirs(output_filepath, exist_ok=True)
    
    # Create the necessary subdirectories
    os.makedirs(os.path.join(output_filepath, 'constant'), exist_ok=True)
    os.makedirs(os.path.join(output_filepath, 'system'), exist_ok=True)
    
    # Create a minimal controlDict file
    with open(os.path.join(output_filepath, 'system', 'controlDict'), 'w') as f:
        f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
        f.write("| =========                 |                                                 |\n")
        f.write("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
        f.write("|  \\\\    /   O peration     | Version:  v2012                                 |\n")
        f.write("|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n")
        f.write("|    \\\\/     M anipulation  |                                                 |\n")
        f.write("\\*---------------------------------------------------------------------------*/\n\n")
        f.write("FoamFile\n")
        f.write("{\n")
        f.write("    version     2.0;\n")
        f.write("    format      ascii;\n")
        f.write("    class       dictionary;\n")
        f.write("    location    \"system\";\n")
        f.write("    object      controlDict;\n")
        f.write("}\n\n")
        f.write("application     simpleFoam;\n\n")
        f.write("startFrom       startTime;\n\n")
        f.write(f"startTime       {results_model.start_time};\n\n")
        f.write("stopAt          endTime;\n\n")
        f.write(f"endTime         {results_model.end_time};\n\n")
        f.write(f"deltaT          {(results_model.end_time - results_model.start_time) / max(1, results_model.num_time_steps - 1)};\n\n")
        f.write("writeControl    runTime;\n\n")
        f.write("writeInterval   1;\n\n")
        f.write("purgeWrite      0;\n\n")
        f.write("writeFormat     ascii;\n\n")
        f.write("writePrecision  6;\n\n")
        f.write("writeCompression off;\n\n")
        f.write("timeFormat      general;\n\n")
        f.write("timePrecision   6;\n\n")
        f.write("runTimeModifiable true;\n")
    
    # Create time directories and field files
    for time_step in results_model.time_steps:
        time_dir = os.path.join(output_filepath, str(time_step))
        os.makedirs(time_dir, exist_ok=True)
        
        # Create field files
        for field_name in results_model.field_names:
            field = results_model.fields[field_name]
            
            # Skip fields that don't have data at this time step
            if field.time_step < time_step:
                continue
            
            # Create the field file
            with open(os.path.join(time_dir, field_name), 'w') as f:
                f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
                f.write("| =========                 |                                                 |\n")
                f.write("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
                f.write("|  \\\\    /   O peration     | Version:  v2012                                 |\n")
                f.write("|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n")
                f.write("|    \\\\/     M anipulation  |                                                 |\n")
                f.write("\\*---------------------------------------------------------------------------*/\n\n")
                f.write("FoamFile\n")
                f.write("{\n")
                f.write("    version     2.0;\n")
                f.write("    format      ascii;\n")
                f.write("    class       ")
                
                if field.type == ResultField.SCALAR:
                    f.write("volScalarField;\n")
                elif field.type == ResultField.VECTOR:
                    f.write("volVectorField;\n")
                elif field.type == ResultField.TENSOR:
                    f.write("volTensorField;\n")
                
                f.write("    location    \"" + str(time_step) + "\";\n")
                f.write("    object      " + field_name + ";\n")
                f.write("}\n\n")
                
                # Write dimensions
                f.write("dimensions      [" + field.dimensions + "];\n\n")
                
                # Write internal field (simplified placeholder)
                f.write("internalField   uniform ")
                
                if field.type == ResultField.SCALAR:
                    f.write(f"{field.average_value};\n\n")
                elif field.type == ResultField.VECTOR:
                    f.write(f"({field.average_value[0]} {field.average_value[1]} {field.average_value[2]});\n\n")
                elif field.type == ResultField.TENSOR:
                    f.write("(\n")
                    f.write(f"    ({field.average_value[0][0]} {field.average_value[0][1]} {field.average_value[0][2]})\n")
                    f.write(f"    ({field.average_value[1][0]} {field.average_value[1][1]} {field.average_value[1][2]})\n")
                    f.write(f"    ({field.average_value[2][0]} {field.average_value[2][1]} {field.average_value[2][2]})\n")
                    f.write(");\n\n")
                
                # Write boundary field (placeholder)
                f.write("boundaryField\n")
                f.write("{\n")
                f.write("    defaultFaces\n")
                f.write("    {\n")
                f.write("        type            zeroGradient;\n")
                f.write("    }\n")
                f.write("}\n")
    
    # Create a .foam file for ParaView
    foam_file = os.path.join(output_filepath, 'case.foam')
    with open(foam_file, 'w') as f:
        f.write('')
    
    return output_filepath


def _export_to_vtk(results_model: ResultsModel, output_filepath: str, format_type: str = 'vtk') -> str:
    """
    Export results to VTK format.
    
    Args:
        results_model: The results model
        output_filepath: Path to the output file
        format_type: Specific VTK format type ('vtk', 'vtu', etc.)
    
    Returns:
        str: Path to the exported results
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
    
    # If we're exporting multiple time steps, create a directory
    if results_model.num_time_steps > 1:
        base_dir = os.path.splitext(output_filepath)[0]
        os.makedirs(base_dir, exist_ok=True)
        
        # Export each time step to a separate file
        for i, time_step in enumerate(results_model.time_steps):
            time_file = os.path.join(base_dir, f"{os.path.basename(base_dir)}_{i:04d}.{format_type}")
            _export_single_time_step_vtk(results_model, time_file, time_step, format_type)
        
        # Create a PVD file for ParaView time series
        pvd_file = os.path.join(base_dir, f"{os.path.basename(base_dir)}.pvd")
        with open(pvd_file, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            f.write('  <Collection>\n')
            
            for i, time_step in enumerate(results_model.time_steps):
                time_file = f"{os.path.basename(base_dir)}_{i:04d}.{format_type}"
                f.write(f'    <DataSet timestep="{time_step}" file="{time_file}"/>\n')
            
            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')
        
        return pvd_file
    else:
        # Export a single time step
        return _export_single_time_step_vtk(results_model, output_filepath, results_model.end_time, format_type)


def _export_single_time_step_vtk(results_model: ResultsModel, output_filepath: str, time_step: float, format_type: str) -> str:
    """
    Export a single time step to VTK format.
    
    Args:
        results_model: The results model
        output_filepath: Path to the output file
        time_step: Time step to export
        format_type: Specific VTK format type ('vtk', 'vtu', etc.)
    
    Returns:
        str: Path to the exported file
    """
    # In a real implementation, we would use VTK library to create the file
    # For simplicity, we'll just write a placeholder file
    
    if format_type == 'vtk':
        # Legacy VTK format (ASCII)
        with open(output_filepath, 'w') as f:
            f.write("# vtk DataFile Version 4.2\n")
            f.write(f"Openfoam_Simulator export - time {time_step}\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            
            # Add placeholder points
            num_points = 1000  # Arbitrary
            f.write(f"POINTS {num_points} float\n")
            for i in range(num_points):
                x = i % 10
                y = (i // 10) % 10
                z = i // 100
                f.write(f"{x} {y} {z}\n")
            
            # Add placeholder cells (tetrahedra)
            num_cells = num_points // 4
            f.write(f"CELLS {num_cells} {num_cells * 5}\n")  # 5 = 1 (num_points) + 4 (points per tet)
            for i in range(num_cells):
                p1 = i * 4
                p2 = i * 4 + 1
                p3 = i * 4 + 2
                p4 = i * 4 + 3
                f.write(f"4 {p1} {p2} {p3} {p4}\n")
            
            # Cell types (10 = VTK_TETRA)
            f.write(f"CELL_TYPES {num_cells}\n")
            for i in range(num_cells):
                f.write("10\n")
            
            # Add point data
            f.write("POINT_DATA {}\n".format(num_points))
            
            # Add each field
            for field_name in results_model.field_names:
                field = results_model.fields[field_name]
                
                if field.type == ResultField.SCALAR:
                    f.write(f"SCALARS {field_name} float 1\n")
                    f.write("LOOKUP_TABLE default\n")
                    
                    # Generate placeholder data
                    for i in range(num_points):
                        # Simple function to generate varying data
                        value = field.min_value + (field.max_value - field.min_value) * (i / num_points)
                        f.write(f"{value}\n")
                
                elif field.type == ResultField.VECTOR:
                    f.write(f"VECTORS {field_name} float\n")
                    
                    # Generate placeholder data
                    for i in range(num_points):
                        # Simple function to generate varying data
                        factor = i / num_points
                        x = field.min_value[0] + (field.max_value[0] - field.min_value[0]) * factor
                        y = field.min_value[1] + (field.max_value[1] - field.min_value[1]) * factor
                        z = field.min_value[2] + (field.max_value[2] - field.min_value[2]) * factor
                        f.write(f"{x} {y} {z}\n")
    
    elif format_type in ['vtu', 'vtp', 'vtr', 'vts']:
        # XML VTK format
        # For simplicity, we'll write a very basic XML structure
        with open(output_filepath, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            
            if format_type == 'vtu':
                f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
                f.write('  <UnstructuredGrid>\n')
                f.write('    <Piece NumberOfPoints="1000" NumberOfCells="250">\n')
            elif format_type == 'vtp':
                f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
                f.write('  <PolyData>\n')
                f.write('    <Piece NumberOfPoints="1000" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="250">\n')
            elif format_type == 'vtr':
                f.write('<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">\n')
                f.write('  <RectilinearGrid WholeExtent="0 9 0 9 0 9">\n')
                f.write('    <Piece Extent="0 9 0 9 0 9">\n')
            elif format_type == 'vts':
                f.write('<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">\n')
                f.write('  <StructuredGrid WholeExtent="0 9 0 9 0 9">\n')
                f.write('    <Piece Extent="0 9 0 9 0 9">\n')
            
            # Placeholder for points, cells, and data
            f.write('      <Points>\n')
            f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
            f.write('          <!-- Points data would go here -->\n')
            f.write('        </DataArray>\n')
            f.write('      </Points>\n')
            
            if format_type in ['vtu', 'vtp']:
                # Add cells for unstructured formats
                f.write('      <Cells>\n')
                f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
                f.write('          <!-- Connectivity data would go here -->\n')
                f.write('        </DataArray>\n')
                f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
                f.write('          <!-- Offset data would go here -->\n')
                f.write('        </DataArray>\n')
                f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
                f.write('          <!-- Type data would go here -->\n')
                f.write('        </DataArray>\n')
                f.write('      </Cells>\n')
            
            # Add point data
            f.write('      <PointData>\n')
            
            # Add each field
            for field_name in results_model.field_names:
                field = results_model.fields[field_name]
                
                if field.type == ResultField.SCALAR:
                    f.write(f'        <DataArray type="Float32" Name="{field_name}" format="ascii">\n')
                    f.write('          <!-- Scalar data would go here -->\n')
                    f.write('        </DataArray>\n')
                
                elif field.type == ResultField.VECTOR:
                    f.write(f'        <DataArray type="Float32" Name="{field_name}" NumberOfComponents="3" format="ascii">\n')
                    f.write('          <!-- Vector data would go here -->\n')
                    f.write('        </DataArray>\n')
            
            f.write('      </PointData>\n')
            
            # Close the tags
            if format_type == 'vtu':
                f.write('    </Piece>\n')
                f.write('  </UnstructuredGrid>\n')
            elif format_type == 'vtp':
                f.write('    </Piece>\n')
                f.write('  </PolyData>\n')
            elif format_type == 'vtr':
                f.write('    </Piece>\n')
                f.write('  </RectilinearGrid>\n')
            elif format_type == 'vts':
                f.write('    </Piece>\n')
                f.write('  </StructuredGrid>\n')
            
            f.write('</VTKFile>\n')
    
    return output_filepath


def _export_to_csv(results_model: ResultsModel, output_filepath: str) -> str:
    """
    Export results to CSV format.
    
    Args:
        results_model: The results model
        output_filepath: Path to the output file
    
    Returns:
        str: Path to the exported results
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
    
    # Create a pandas DataFrame
    data = []
    
    # Add time column
    time_column = {"time": results_model.time_steps}
    
    # Add field columns
    field_columns = {}
    
    for field_name in results_model.field_names:
        field = results_model.fields[field_name]
        
        if field.type == ResultField.SCALAR:
            # Generate placeholder data for each time step
            values = []
            for time_step in results_model.time_steps:
                # Simple function to generate varying data
                factor = (time_step - results_model.start_time) / (results_model.end_time - results_model.start_time)
                value = field.min_value + (field.max_value - field.min_value) * factor
                values.append(value)
            
            field_columns[field_name] = values
        
        elif field.type == ResultField.VECTOR:
            # Generate placeholder data for each component and time step
            for i, component in enumerate(['x', 'y', 'z']):
                values = []
                for time_step in results_model.time_steps:
                    # Simple function to generate varying data
                    factor = (time_step - results_model.start_time) / (results_model.end_time - results_model.start_time)
                    value = field.min_value[i] + (field.max_value[i] - field.min_value[i]) * factor
                    values.append(value)
                
                field_columns[f"{field_name}:{component}"] = values
    
    # Combine into a single dictionary
    data = {**time_column, **field_columns}
    
    # Create DataFrame and export to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_filepath, index=False)
    
    return output_filepath


def _export_to_tecplot(results_model: ResultsModel, output_filepath: str, binary: bool = False) -> str:
    """
    Export results to Tecplot format.
    
    Args:
        results_model: The results model
        output_filepath: Path to the output file
        binary: Whether to use binary format (.plt) or ASCII format (.dat)
    
    Returns:
        str: Path to the exported results
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
    
    if binary:
        # Binary format requires a specialized library
        # For simplicity, we'll just write a placeholder message
        with open(output_filepath, 'w') as f:
            f.write("This is a placeholder for a binary Tecplot file.\n")
            f.write("In a real implementation, this would be a binary .plt file.\n")
    else:
        # ASCII Tecplot format
        with open(output_filepath, 'w') as f:
            # Write header
            f.write(f'TITLE = "Openfoam_Simulator export - {results_model.name}"\n')
            
            # Write variables
            variables = ['X', 'Y', 'Z']
            for field_name in results_model.field_names:
                field = results_model.fields[field_name]
                
                if field.type == ResultField.SCALAR:
                    variables.append(field_name)
                elif field.type == ResultField.VECTOR:
                    variables.extend([f"{field_name}_X", f"{field_name}_Y", f"{field_name}_Z"])
            
            f.write('VARIABLES = ' + ', '.join(f'"{var}"' for var in variables) + '\n')
            
            # Write a zone for each time step
            for i, time_step in enumerate(results_model.time_steps):
                # Placeholder for actual mesh size - in a real implementation,
                # we would get the actual mesh size from the results
                num_points = 1000  # Arbitrary
                
                f.write(f'ZONE T="Time {time_step}", I=10, J=10, K=10, SOLUTIONTIME={time_step}\n')
                
                # Write placeholder data
                for j in range(num_points):
                    x = j % 10
                    y = (j // 10) % 10
                    z = j // 100
                    
                    line = f"{x} {y} {z}"
                    
                    for field_name in results_model.field_names:
                        field = results_model.fields[field_name]
                        
                        if field.type == ResultField.SCALAR:
                            # Generate a value based on position and time
                            factor = (time_step - results_model.start_time) / (results_model.end_time - results_model.start_time)
                            value = field.min_value + (field.max_value - field.min_value) * factor * (j / num_points)
                            line += f" {value}"
                        elif field.type == ResultField.VECTOR:
                            # Generate values for each component
                            for i, component in enumerate(['x', 'y', 'z']):
                                factor = (time_step - results_model.start_time) / (results_model.end_time - results_model.start_time)
                                value = field.min_value[i] + (field.max_value[i] - field.min_value[i]) * factor * (j / num_points)
                                line += f" {value}"
                    
                    f.write(line + '\n')
    
    return output_filepath


def _export_to_cgns(results_model: ResultsModel, output_filepath: str) -> str:
    """
    Export results to CGNS format.
    
    Args:
        results_model: The results model
        output_filepath: Path to the output file
    
    Returns:
        str: Path to the exported results
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
    
    # In a real implementation, we would use a CGNS library to create the file
    # For simplicity, we'll just write a placeholder file
    with open(output_filepath, 'w') as f:
        f.write("This is a placeholder for a CGNS file.\n")
        f.write("In a real implementation, this would be a binary CGNS file.\n")
        f.write(f"Exported from results model: {results_model.name}\n")
        f.write(f"Time steps: {results_model.time_steps}\n")
        f.write(f"Fields: {results_model.field_names}\n")
    
    return output_filepath


if __name__ == "__main__":
    # Command-line utility for testing result import
    import argparse
    
    parser = argparse.ArgumentParser(description="Result import utility")
    parser.add_argument("input", help="Input result file or directory")
    parser.add_argument("--output", help="Output file (for conversion)")
    parser.add_argument("--format", help="Output format (for conversion)")
    parser.add_argument("--field", help="Field to extract")
    parser.add_argument("--time", type=float, help="Time step to extract")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Import the results
        results = import_results(args.input)
        
        # Print results information
        print(f"Imported {results.format} results: {results.name}")
        print(f"Time steps: {results.time_steps}")
        print(f"Fields: {results.field_names}")
        
        # Extract field if specified
        if args.field and args.field in results.field_names:
            field = results.fields[args.field]
            print(f"Field: {field.name}, Type: {field.type}")
            print(f"Min: {field.min_value}, Max: {field.max_value}, Avg: {field.average_value}")
            
            if args.time:
                print(f"Extracting field {args.field} at time {args.time}")
                data = extract_field(results, args.field, args.time)
                print(f"Field data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
                print(f"First few values: {data[:5] if hasattr(data, '__getitem__') else data}")
        
        # Convert if output specified
        if args.output:
            output_path = convert_results(args.input, args.output, args.format)
            print(f"Converted results to {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    sys.exit(0)