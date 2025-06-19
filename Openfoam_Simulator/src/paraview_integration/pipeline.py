#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization pipeline manager for Openfoam_Simulator application.

This module manages the visualization pipeline for the application, including
creating, configuring, and connecting visualization filters and data sources.
The pipeline manager handles the flow of data from OpenFOAM results to 
on-screen visualization using pure VTK without ParaView.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# Import utility modules
from ..utils.logger import get_logger
from ..config import get_value

# Import VTK
try:
    import vtk
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False

# Global logger
logger = get_logger(__name__)


def vtk_reader_factory(filepath: str):
    """
    Create a VTK reader based on file extension.
    
    Args:
        filepath (str): Path to the data file
        
    Returns:
        vtkAlgorithm: A VTK reader for the specified file
    """
    # Get file extension
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.vtk':
        reader = vtk.vtkDataSetReader()
    elif ext == '.vtu':
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif ext == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == '.vts':
        reader = vtk.vtkXMLStructuredGridReader()
    elif ext == '.vtr':
        reader = vtk.vtkXMLRectilinearGridReader()
    elif ext == '.stl':
        reader = vtk.vtkSTLReader()
    elif ext == '.obj':
        reader = vtk.vtkOBJReader()
    elif ext == '.ply':
        reader = vtk.vtkPLYReader()
    elif ext in ['.foam', '.OpenFOAM']:
        # For OpenFOAM cases, we need to create a .foam file
        if os.path.isdir(filepath):
            # Create a .foam file for the directory
            foam_file = os.path.join(filepath, 'case.foam')
            with open(foam_file, 'w') as f:
                pass
            filepath = foam_file
            
        # Use vtkOpenFOAMReader if available
        try:
            reader = vtk.vtkOpenFOAMReader()
        except AttributeError:
            logger.warning("vtkOpenFOAMReader not available, using generic reader")
            reader = vtk.vtkGenericDataObjectReader()
    else:
        # Try to use generic reader
        reader = vtk.vtkGenericDataObjectReader()
    
    # Set filename
    reader.SetFileName(filepath)
    
    return reader


def create_lookup_table(colormap_name: str, num_colors: int = 256):
    """
    Create a VTK lookup table based on colormap name.
    
    Args:
        colormap_name (str): Name of the colormap
        num_colors (int): Number of colors in the table
        
    Returns:
        vtkLookupTable: A VTK lookup table
    """
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(num_colors)
    
    if colormap_name == "Rainbow":
        lut.SetHueRange(0.667, 0.0)  # Blue to red
        lut.SetSaturationRange(0.8, 0.8)
        lut.SetValueRange(1.0, 1.0)
    elif colormap_name == "Cool to Warm":
        # Blue to white to red
        for i in range(num_colors):
            t = i / (num_colors - 1)
            if t < 0.5:
                r = 0.0
                g = t * 2
                b = 1.0
            else:
                r = (t - 0.5) * 2
                g = 1.0 - (t - 0.5) * 2
                b = 1.0 - (t - 0.5) * 2
            lut.SetTableValue(i, r, g, b, 1.0)
    elif colormap_name == "Viridis":
        # Approximation of Viridis colormap
        for i in range(num_colors):
            t = i / (num_colors - 1)
            if t < 0.25:
                r = 0.267
                g = 0.004 + t * 4 * 0.996
                b = 0.329 + t * 4 * 0.4
            elif t < 0.5:
                r = 0.267 + (t - 0.25) * 4 * 0.178
                g = 1.0
                b = 0.729 - (t - 0.25) * 4 * 0.356
            elif t < 0.75:
                r = 0.445 + (t - 0.5) * 4 * 0.291
                g = 1.0 - (t - 0.5) * 4 * 0.199
                b = 0.373 - (t - 0.5) * 4 * 0.255
            else:
                r = 0.736 + (t - 0.75) * 4 * 0.264
                g = 0.801 - (t - 0.75) * 4 * 0.801
                b = 0.118 - (t - 0.75) * 4 * 0.118
            lut.SetTableValue(i, r, g, b, 1.0)
    elif colormap_name == "Plasma":
        # Approximation of Plasma colormap
        for i in range(num_colors):
            t = i / (num_colors - 1)
            if t < 0.25:
                r = 0.05 + t * 4 * 0.45
                g = 0.03 + t * 4 * 0.17
                b = 0.54 - t * 4 * 0.16
            elif t < 0.5:
                r = 0.5 + (t - 0.25) * 4 * 0.4
                g = 0.2 + (t - 0.25) * 4 * 0.3
                b = 0.38 - (t - 0.25) * 4 * 0.28
            elif t < 0.75:
                r = 0.9 + (t - 0.5) * 4 * 0.1
                g = 0.5 + (t - 0.5) * 4 * 0.3
                b = 0.1 - (t - 0.5) * 4 * 0.1
            else:
                r = 1.0
                g = 0.8 + (t - 0.75) * 4 * 0.2
                b = 0.0 + (t - 0.75) * 4 * 0.2
            lut.SetTableValue(i, r, g, b, 1.0)
    elif colormap_name == "Jet":
        # Jet colormap (blue-cyan-yellow-red)
        for i in range(num_colors):
            t = i / (num_colors - 1)
            if t < 0.125:
                r = 0.0
                g = 0.0
                b = 0.5 + t * 8 * 0.5
            elif t < 0.375:
                r = 0.0
                g = (t - 0.125) * 4
                b = 1.0
            elif t < 0.625:
                r = (t - 0.375) * 4
                g = 1.0
                b = 1.0 - (t - 0.375) * 4
            elif t < 0.875:
                r = 1.0
                g = 1.0 - (t - 0.625) * 4
                b = 0.0
            else:
                r = 1.0 - (t - 0.875) * 8 * 0.5
                g = 0.0
                b = 0.0
            lut.SetTableValue(i, r, g, b, 1.0)
    elif colormap_name == "Pressure":
        # Custom pressure colormap (blue-white-red)
        for i in range(num_colors):
            t = i / (num_colors - 1)
            if t < 0.5:
                r = t * 2
                g = t * 2
                b = 1.0
            else:
                r = 1.0
                g = 1.0 - (t - 0.5) * 2
                b = 1.0 - (t - 0.5) * 2
            lut.SetTableValue(i, r, g, b, 1.0)
    elif colormap_name == "Velocity":
        # Custom velocity colormap
        for i in range(num_colors):
            t = i / (num_colors - 1)
            if t < 0.25:
                r = 0.0
                g = t * 4
                b = 1.0
            elif t < 0.5:
                r = 0.0
                g = 1.0
                b = 1.0 - (t - 0.25) * 4
            elif t < 0.75:
                r = (t - 0.5) * 4
                g = 1.0
                b = 0.0
            else:
                r = 1.0
                g = 1.0 - (t - 0.75) * 4
                b = 0.0
            lut.SetTableValue(i, r, g, b, 1.0)
    elif colormap_name == "Oil-Water":
        # Custom oil-water interface colormap
        for i in range(num_colors):
            t = i / (num_colors - 1)
            if t < 0.4:
                r = 0.0
                g = 0.0 + t * 2.5 * 0.5
                b = 0.8 - t * 2.5 * 0.2
            elif t < 0.6:
                r = (t - 0.4) * 5 * 0.8
                g = 0.5 + (t - 0.4) * 5 * 0.3
                b = 0.6 - (t - 0.4) * 5 * 0.6
            else:
                r = 0.8 + (t - 0.6) * 2.5 * 0.2
                g = 0.8 - (t - 0.6) * 2.5 * 0.3
                b = 0.0
            lut.SetTableValue(i, r, g, b, 1.0)
    else:
        # Default rainbow
        lut.SetHueRange(0.667, 0.0)
        lut.SetSaturationRange(0.8, 0.8)
        lut.SetValueRange(1.0, 1.0)
    
    lut.Build()
    return lut


class PipelineManager:
    """
    Manager for visualization pipelines using pure VTK.
    
    This class manages the creation and configuration of visualization pipelines,
    handling data flow from sources to sinks through various filters, and
    provides a high-level interface for common visualization operations.
    """
    
    def __init__(self, render_window=None):
        """
        Initialize the pipeline manager.
        
        Args:
            render_window: VTK render window to use for visualization
        """
        self.render_window = render_window
        self.initialized = VTK_AVAILABLE
        
        if not self.initialized:
            logger.error("VTK is not available")
            return
        
        # Create a renderer if render window is provided
        if self.render_window:
            self.renderer = vtk.vtkRenderer()
            self.render_window.AddRenderer(self.renderer)
            
            # Set up default background color
            bg_color = get_value('paraview.viewport_settings.background_color', [0.2, 0.2, 0.2])
            self.renderer.SetBackground(bg_color)
            
            # Set up camera
            self.camera = self.renderer.GetActiveCamera()
            camera_position = get_value('paraview.viewport_settings.camera_position', [1.0, 1.0, 1.0])
            camera_focal_point = get_value('paraview.viewport_settings.camera_focal_point', [0.0, 0.0, 0.0])
            camera_view_up = get_value('paraview.viewport_settings.camera_view_up', [0.0, 0.0, 1.0])
            
            self.camera.SetPosition(camera_position)
            self.camera.SetFocalPoint(camera_focal_point)
            self.camera.SetViewUp(camera_view_up)
        else:
            self.renderer = None
            self.camera = None
        
        # Initialize pipeline data structures
        self.sources = {}  # Dict of pipeline name to source object
        self.filters = {}  # Dict of pipeline name to list of filters
        self.active_pipeline = None
        self.active_source = None
        self.active_filter = None
        self.active_preset = None
        
        # Load presets
        self.presets = self._load_presets()
    
    def _load_presets(self) -> Dict[str, Dict[str, Any]]:
        """
        Load visualization presets from configuration.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of preset configurations
        """
        # This would typically load presets from a configuration file or database
        # For now, we'll just define some basic presets
        presets = {
            "Pipeline Flow": {
                "filters": [
                    {
                        "type": "slice",
                        "parameters": {
                            "normal": [0, 0, 1],
                            "origin": [0, 0, 0]
                        }
                    }
                ],
                "coloring": {
                    "field": "U",
                    "component": "Magnitude",
                    "color_map": "Velocity"
                },
                "representation": "Surface"
            },
            "Oil-Water Interface": {
                "filters": [
                    {
                        "type": "contour",
                        "parameters": {
                            "field": "alpha.water",
                            "values": [0.5]
                        }
                    }
                ],
                "coloring": {
                    "field": "alpha.water",
                    "color_map": "Oil-Water"
                },
                "representation": "Surface"
            },
            "Pressure Distribution": {
                "filters": [],
                "coloring": {
                    "field": "p",
                    "color_map": "Pressure"
                },
                "representation": "Surface"
            },
            "Velocity Streamlines": {
                "filters": [
                    {
                        "type": "streamline",
                        "parameters": {
                            "field": "U",
                            "seed_type": "plane",
                            "max_length": 10.0
                        }
                    }
                ],
                "coloring": {
                    "field": "U",
                    "component": "Magnitude",
                    "color_map": "Velocity"
                },
                "representation": "Surface"
            },
            "Pigging Analysis": {
                "filters": [
                    {
                        "type": "slice",
                        "parameters": {
                            "normal": [0, 0, 1],
                            "origin": [0, 0, 0]
                        }
                    }
                ],
                "coloring": {
                    "field": "alpha.water",
                    "color_map": "Oil-Water"
                },
                "representation": "Surface With Edges"
            }
        }
        
        return presets
    
    def create_pipeline(self, name: str) -> bool:
        """
        Create a new empty pipeline.
        
        Args:
            name (str): Name for the pipeline
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("VTK not initialized")
            return False
        
        if name in self.sources:
            logger.warning(f"Pipeline {name} already exists")
            return False
        
        # Create empty pipeline
        self.sources[name] = None
        self.filters[name] = []
        
        # Set as active pipeline
        self.active_pipeline = name
        self.active_source = None
        self.active_filter = None
        
        logger.info(f"Created pipeline: {name}")
        return True
    
    def load_data(self, filepath: str, pipeline_name: str = None) -> bool:
        """
        Load data into a pipeline.
        
        Args:
            filepath (str): Path to the data file
            pipeline_name (str, optional): Name for the pipeline, if None a name will be generated
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("VTK not initialized")
            return False
        
        try:
            # Generate pipeline name if not provided
            if pipeline_name is None:
                pipeline_name = os.path.basename(filepath)
            
            # Create pipeline if it doesn't exist
            if pipeline_name not in self.sources:
                self.create_pipeline(pipeline_name)
            
            # Create appropriate reader for the file
            reader = vtk_reader_factory(filepath)
            reader.Update()
            
            # Check if reader was successful
            if reader.GetOutput() is None:
                logger.error(f"Failed to read file: {filepath}")
                return False
            
            # Clear any existing actors for this pipeline
            if self.renderer:
                # Remove any existing actors for this pipeline
                self._remove_pipeline_actors(pipeline_name)
            
            # Create mapper and actor for the reader
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputConnection(reader.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            # Add actor to renderer
            if self.renderer:
                self.renderer.AddActor(actor)
            
            # Store source
            self.sources[pipeline_name] = {
                'reader': reader,
                'mapper': mapper,
                'actor': actor,
                'filepath': filepath
            }
            
            # Initialize filters list
            self.filters[pipeline_name] = []
            
            # Set as active pipeline
            self.active_pipeline = pipeline_name
            self.active_source = self.sources[pipeline_name]
            self.active_filter = None
            
            # Reset camera
            if self.renderer:
                self.renderer.ResetCamera()
            
            logger.info(f"Loaded data into pipeline: {pipeline_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def _remove_pipeline_actors(self, pipeline_name: str):
        """
        Remove all actors associated with a pipeline.
        
        Args:
            pipeline_name (str): Name of the pipeline
        """
        if not self.renderer:
            return
        
        # Remove source actor if it exists
        if pipeline_name in self.sources and self.sources[pipeline_name]:
            source = self.sources[pipeline_name]
            if 'actor' in source and source['actor']:
                self.renderer.RemoveActor(source['actor'])
        
        # Remove filter actors
        if pipeline_name in self.filters:
            for filter_info in self.filters[pipeline_name]:
                if 'actor' in filter_info and filter_info['actor']:
                    self.renderer.RemoveActor(filter_info['actor'])
    
    def add_filter(self, filter_type: str, parameters: Dict[str, Any] = None, 
                 pipeline_name: str = None) -> bool:
        """
        Add a filter to a pipeline.
        
        Args:
            filter_type (str): Type of filter to add
            parameters (Dict[str, Any], optional): Parameters for the filter
            pipeline_name (str, optional): Name of the pipeline, if None uses active pipeline
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("VTK not initialized")
            return False
        
        # Use active pipeline if not specified
        if pipeline_name is None:
            if self.active_pipeline is None:
                logger.error("No active pipeline")
                return False
            pipeline_name = self.active_pipeline
        
        # Check if pipeline exists
        if pipeline_name not in self.sources or not self.sources[pipeline_name]:
            logger.error(f"Pipeline {pipeline_name} does not exist or has no source")
            return False
        
        # Get input for the filter
        if self.active_filter is None:
            # Use pipeline source as input
            input_source = self.sources[pipeline_name]['reader']
        else:
            # Use last filter as input
            input_source = self.active_filter['filter']
        
        try:
            # Create appropriate filter based on type
            filter_obj = None
            if filter_type.lower() == 'slice':
                filter_obj = vtk.vtkPlane()
                
                # Set plane parameters
                if parameters:
                    if 'normal' in parameters:
                        filter_obj.SetNormal(parameters['normal'])
                    if 'origin' in parameters:
                        filter_obj.SetOrigin(parameters['origin'])
                
                # Create the actual cutter filter
                cutter = vtk.vtkCutter()
                cutter.SetInputConnection(input_source.GetOutputPort())
                cutter.SetCutFunction(filter_obj)
                
                # Create mapper and actor
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(cutter.GetOutputPort())
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                
                # Add actor to renderer
                if self.renderer:
                    self.renderer.AddActor(actor)
                
                # Store the filter
                filter_info = {
                    'type': filter_type,
                    'filter': cutter,
                    'cut_function': filter_obj,
                    'mapper': mapper,
                    'actor': actor,
                    'parameters': parameters or {}
                }
                
            elif filter_type.lower() == 'contour':
                filter_obj = vtk.vtkContourFilter()
                filter_obj.SetInputConnection(input_source.GetOutputPort())
                
                # Set contour parameters
                if parameters:
                    if 'field' in parameters:
                        # Set the scalar for contouring
                        field_name = parameters['field']
                        filter_obj.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, field_name)
                    
                    if 'values' in parameters and isinstance(parameters['values'], list):
                        for i, value in enumerate(parameters['values']):
                            filter_obj.SetValue(i, value)
                    elif 'value' in parameters:
                        filter_obj.SetValue(0, parameters['value'])
                    else:
                        # Default: single contour at 0.5
                        filter_obj.SetValue(0, 0.5)
                else:
                    # Default: single contour at 0.5
                    filter_obj.SetValue(0, 0.5)
                
                # Create mapper and actor
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(filter_obj.GetOutputPort())
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                
                # Add actor to renderer
                if self.renderer:
                    self.renderer.AddActor(actor)
                
                # Store the filter
                filter_info = {
                    'type': filter_type,
                    'filter': filter_obj,
                    'mapper': mapper,
                    'actor': actor,
                    'parameters': parameters or {}
                }
                
            elif filter_type.lower() == 'clip':
                filter_obj = vtk.vtkClipDataSet()
                filter_obj.SetInputConnection(input_source.GetOutputPort())
                
                # Set clip parameters
                if parameters:
                    if 'function' in parameters and parameters['function'] == 'plane':
                        clip_function = vtk.vtkPlane()
                        
                        if 'normal' in parameters:
                            clip_function.SetNormal(parameters['normal'])
                        if 'origin' in parameters:
                            clip_function.SetOrigin(parameters['origin'])
                        
                        filter_obj.SetClipFunction(clip_function)
                    
                    if 'inside_out' in parameters:
                        filter_obj.SetInsideOut(parameters['inside_out'])
                
                # Create mapper and actor
                mapper = vtk.vtkDataSetMapper()
                mapper.SetInputConnection(filter_obj.GetOutputPort())
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                
                # Add actor to renderer
                if self.renderer:
                    self.renderer.AddActor(actor)
                
                # Store the filter
                filter_info = {
                    'type': filter_type,
                    'filter': filter_obj,
                    'mapper': mapper,
                    'actor': actor,
                    'parameters': parameters or {}
                }
                
            elif filter_type.lower() == 'threshold':
                filter_obj = vtk.vtkThreshold()
                filter_obj.SetInputConnection(input_source.GetOutputPort())
                
                # Set threshold parameters
                if parameters:
                    if 'field' in parameters:
                        # Set the scalar for thresholding
                        field_name = parameters['field']
                        filter_obj.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, field_name)
                    
                    if 'min_value' in parameters:
                        filter_obj.SetLowerThreshold(parameters['min_value'])
                    
                    if 'max_value' in parameters:
                        filter_obj.SetUpperThreshold(parameters['max_value'])
                    
                    # Set threshold function
                    if 'min_value' in parameters and 'max_value' in parameters:
                        filter_obj.ThresholdBetween(parameters['min_value'], parameters['max_value'])
                    elif 'min_value' in parameters:
                        filter_obj.ThresholdByLower(parameters['min_value'])
                    elif 'max_value' in parameters:
                        filter_obj.ThresholdByUpper(parameters['max_value'])
                
                # Create mapper and actor
                mapper = vtk.vtkDataSetMapper()
                mapper.SetInputConnection(filter_obj.GetOutputPort())
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                
                # Add actor to renderer
                if self.renderer:
                    self.renderer.AddActor(actor)
                
                # Store the filter
                filter_info = {
                    'type': filter_type,
                    'filter': filter_obj,
                    'mapper': mapper,
                    'actor': actor,
                    'parameters': parameters or {}
                }
                
            elif filter_type.lower() == 'streamline':
                # First, make sure we have vector data for streamlines
                filter_obj = vtk.vtkStreamTracer()
                filter_obj.SetInputConnection(input_source.GetOutputPort())
                
                # Set streamline parameters
                if parameters:
                    if 'field' in parameters:
                        # Set the vector field for streamlines
                        field_name = parameters['field']
                        filter_obj.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, field_name)
                    
                    # Create seed source
                    seed_type = parameters.get('seed_type', 'plane')
                    if seed_type == 'plane':
                        seed_source = vtk.vtkPlaneSource()
                        seed_source.SetXResolution(10)
                        seed_source.SetYResolution(10)
                        if 'center' in parameters:
                            seed_source.SetCenter(parameters['center'])
                        if 'normal' in parameters:
                            seed_source.SetNormal(parameters['normal'])
                    elif seed_type == 'line':
                        seed_source = vtk.vtkLineSource()
                        if 'point1' in parameters and 'point2' in parameters:
                            seed_source.SetPoint1(parameters['point1'])
                            seed_source.SetPoint2(parameters['point2'])
                        seed_source.SetResolution(20)
                    elif seed_type == 'point':
                        seed_source = vtk.vtkPointSource()
                        if 'center' in parameters:
                            seed_source.SetCenter(parameters['center'])
                        if 'radius' in parameters:
                            seed_source.SetRadius(parameters['radius'])
                        seed_source.SetNumberOfPoints(1)
                    else:
                        # Default to a point source
                        seed_source = vtk.vtkPointSource()
                        seed_source.SetNumberOfPoints(10)
                        seed_source.SetRadius(1.0)
                    
                    # Update seed source
                    seed_source.Update()
                    
                    # Set up the stream tracer
                    filter_obj.SetSourceConnection(seed_source.GetOutputPort())
                    
                    if 'max_length' in parameters:
                        filter_obj.SetMaximumPropagation(parameters['max_length'])
                    
                    if 'integration_direction' in parameters:
                        direction = parameters['integration_direction']
                        if direction == 'forward':
                            filter_obj.SetIntegrationDirectionToForward()
                        elif direction == 'backward':
                            filter_obj.SetIntegrationDirectionToBackward()
                        else:
                            filter_obj.SetIntegrationDirectionToBoth()
                
                # Create mapper and actor
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(filter_obj.GetOutputPort())
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                
                # Streamlines are usually thinner
                actor.GetProperty().SetLineWidth(2.0)
                
                # Add actor to renderer
                if self.renderer:
                    self.renderer.AddActor(actor)
                
                # Store the filter
                filter_info = {
                    'type': filter_type,
                    'filter': filter_obj,
                    'seed_source': seed_source,
                    'mapper': mapper,
                    'actor': actor,
                    'parameters': parameters or {}
                }
                
            else:
                logger.error(f"Unsupported filter type: {filter_type}")
                return False
            
            # Add filter to pipeline
            self.filters[pipeline_name].append(filter_info)
            
            # Update active filter
            self.active_filter = filter_info
            
            # Render
            if self.render_window:
                self.render_window.Render()
            
            logger.info(f"Added {filter_type} filter to pipeline: {pipeline_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding filter: {e}")
            return False
    
    def apply_preset(self, preset_name: str, pipeline_name: str = None) -> bool:
        """
        Apply a visualization preset to a pipeline.
        
        Args:
            preset_name (str): Name of the preset to apply
            pipeline_name (str, optional): Name of the pipeline, if None uses active pipeline
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("VTK not initialized")
            return False
        
        # Use active pipeline if not specified
        if pipeline_name is None:
            if self.active_pipeline is None:
                logger.error("No active pipeline")
                return False
            pipeline_name = self.active_pipeline
        
        # Check if pipeline exists
        if pipeline_name not in self.sources or not self.sources[pipeline_name]:
            logger.error(f"Pipeline {pipeline_name} does not exist or has no source")
            return False
        
        # Check if preset exists
        if preset_name not in self.presets:
            logger.error(f"Preset {preset_name} does not exist")
            return False
        
        try:
            # Get preset configuration
            preset = self.presets[preset_name]
            
            # Remove all existing filters from the pipeline
            self._remove_pipeline_actors(pipeline_name)
            self.filters[pipeline_name] = []
            
            # Set active pipeline and source
            self.active_pipeline = pipeline_name
            self.active_source = self.sources[pipeline_name]
            self.active_filter = None
            
            # Re-add the source actor to the renderer
            if self.renderer and 'actor' in self.active_source:
                self.renderer.AddActor(self.active_source['actor'])
            
            # Add filters from preset
            if 'filters' in preset:
                for filter_config in preset['filters']:
                    filter_type = filter_config['type']
                    parameters = filter_config.get('parameters', {})
                    
                    # Add filter
                    if not self.add_filter(filter_type, parameters, pipeline_name):
                        logger.warning(f"Failed to add {filter_type} filter from preset {preset_name}")
            
            # Apply coloring
            if 'coloring' in preset:
                coloring = preset['coloring']
                field = coloring.get('field')
                component = coloring.get('component')
                color_map = coloring.get('color_map')
                
                if field:
                    # Apply coloring
                    if not self.color_by_field(field, component):
                        logger.warning(f"Failed to apply coloring to field {field} from preset {preset_name}")
                
                if color_map:
                    # Apply color map
                    if not self.set_color_map(color_map):
                        logger.warning(f"Failed to apply color map {color_map} from preset {preset_name}")
            
            # Apply representation
            if 'representation' in preset:
                representation = preset['representation']
                
                # Apply representation
                if not self.set_representation(representation):
                    logger.warning(f"Failed to apply representation {representation} from preset {preset_name}")
            
            # Store active preset
            self.active_preset = preset_name
            
            # Render
            if self.render_window:
                self.render_window.Render()
            
            logger.info(f"Applied preset {preset_name} to pipeline: {pipeline_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying preset: {e}")
            return False
    
    def set_active_pipeline(self, pipeline_name: str) -> bool:
        """
        Set the active pipeline.
        
        Args:
            pipeline_name (str): Name of the pipeline to activate
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if pipeline exists
        if pipeline_name not in self.sources:
            logger.error(f"Pipeline {pipeline_name} does not exist")
            return False
        
        # Set active pipeline
        self.active_pipeline = pipeline_name
        self.active_source = self.sources[pipeline_name]
        
        # Set active filter to the last filter in the pipeline, if any
        if pipeline_name in self.filters and self.filters[pipeline_name]:
            self.active_filter = self.filters[pipeline_name][-1]
        else:
            self.active_filter = None
        
        logger.info(f"Set active pipeline: {pipeline_name}")
        return True
    
    def display_pipeline(self, pipeline_name: str = None) -> bool:
        """
        Display a pipeline in the render window.
        
        Args:
            pipeline_name (str, optional): Name of the pipeline, if None uses active pipeline
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("VTK not initialized")
            return False
        
        # Use active pipeline if not specified
        if pipeline_name is None:
            if self.active_pipeline is None:
                logger.error("No active pipeline")
                return False
            pipeline_name = self.active_pipeline
        
        # Check if pipeline exists
        if pipeline_name not in self.sources or not self.sources[pipeline_name]:
            logger.error(f"Pipeline {pipeline_name} does not exist or has no source")
            return False
        
        try:
            # Hide all other pipelines
            for name, source in self.sources.items():
                if name != pipeline_name and source and 'actor' in source:
                    source['actor'].SetVisibility(False)
                    
                    # Also hide filter actors
                    if name in self.filters:
                        for filter_info in self.filters[name]:
                            if 'actor' in filter_info:
                                filter_info['actor'].SetVisibility(False)
            
            # Show the selected pipeline
            if 'actor' in self.sources[pipeline_name]:
                self.sources[pipeline_name]['actor'].SetVisibility(True)
                
                # Also show filter actors
                if pipeline_name in self.filters:
                    for filter_info in self.filters[pipeline_name]:
                        if 'actor' in filter_info:
                            filter_info['actor'].SetVisibility(True)
            
            # Set active pipeline
            self.set_active_pipeline(pipeline_name)
            
            # Render
            if self.render_window:
                self.render_window.Render()
            
            logger.info(f"Displayed pipeline: {pipeline_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error displaying pipeline: {e}")
            return False
    
    def reset_camera(self) -> bool:
        """
        Reset the camera view.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized or not self.renderer:
            logger.error("VTK not initialized or no renderer")
            return False
        
        try:
            self.renderer.ResetCamera()
            if self.render_window:
                self.render_window.Render()
            
            logger.info("Reset camera view")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting camera: {e}")
            return False
    
    def set_representation(self, representation: str) -> bool:
        """
        Set the representation type for the active pipeline.
        
        Args:
            representation (str): Representation type (Surface, Wireframe, Points, Volume)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("VTK not initialized")
            return False
        
        # Check if we have an active pipeline
        if not self.active_pipeline or not self.sources[self.active_pipeline]:
            logger.error("No active pipeline or source")
            return False
        
        try:
            # Convert representation string to VTK constants
            rep_map = {
                "Surface": vtk.VTK_SURFACE,
                "Wireframe": vtk.VTK_WIREFRAME,
                "Points": vtk.VTK_POINTS,
                "Surface With Edges": vtk.VTK_SURFACE
            }
            
            vtk_rep = rep_map.get(representation, vtk.VTK_SURFACE)
            
            # Set representation for source actor
            source = self.sources[self.active_pipeline]
            if 'actor' in source:
                source['actor'].GetProperty().SetRepresentation(vtk_rep)
                
                # If Surface With Edges, enable edge visibility
                if representation == "Surface With Edges":
                    source['actor'].GetProperty().SetEdgeVisibility(True)
                else:
                    source['actor'].GetProperty().SetEdgeVisibility(False)
            
            # Set representation for filter actors
            if self.active_pipeline in self.filters:
                for filter_info in self.filters[self.active_pipeline]:
                    if 'actor' in filter_info:
                        filter_info['actor'].GetProperty().SetRepresentation(vtk_rep)
                        
                        # If Surface With Edges, enable edge visibility
                        if representation == "Surface With Edges":
                            filter_info['actor'].GetProperty().SetEdgeVisibility(True)
                        else:
                            filter_info['actor'].GetProperty().SetEdgeVisibility(False)
            
            # Render
            if self.render_window:
                self.render_window.Render()
            
            logger.info(f"Set representation to {representation}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting representation: {e}")
            return False
    
    def color_by_field(self, field_name: str, component: str = None) -> bool:
        """
        Color the active pipeline by a field.
        
        Args:
            field_name (str): Name of the field
            component (str, optional): Component for vector fields (X, Y, Z, Magnitude)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("VTK not initialized")
            return False
        
        # Check if we have an active pipeline
        if not self.active_pipeline or not self.sources[self.active_pipeline]:
            logger.error("No active pipeline or source")
            return False
        
        try:
            # Get the active data object
            if self.active_filter is not None:
                # Use the active filter's output
                active_filter = self.active_filter['filter']
                active_filter.Update()
                data_object = active_filter.GetOutput()
                mapper = self.active_filter['mapper']
            else:
                # Use the source's output
                source = self.sources[self.active_pipeline]
                source['reader'].Update()
                data_object = source['reader'].GetOutput()
                mapper = source['mapper']
            
            # If field_name is 'Solid Color', disable scalar coloring
            if field_name.lower() == 'solid color':
                mapper.ScalarVisibilityOff()
                
                # Render
                if self.render_window:
                    self.render_window.Render()
                
                logger.info("Set solid color (disabled scalar coloring)")
                return True
            
            # Check if field exists in data
            point_data = data_object.GetPointData()
            if not point_data:
                logger.error("No point data available")
                return False
            
            # Find the array
            array = point_data.GetArray(field_name)
            if not array:
                logger.error(f"Field not found: {field_name}")
                return False
            
            # Set the array for coloring
            point_data.SetActiveScalars(field_name)
            
            # For vector fields, extract the requested component
            if array.GetNumberOfComponents() > 1:
                if component and component.lower() != 'magnitude':
                    # Extract component (X, Y, Z)
                    component_map = {'x': 0, 'y': 1, 'z': 2}
                    comp_idx = component_map.get(component.lower(), 0)
                    
                    # Create an array extract filter
                    extract = vtk.vtkArrayCalculator()
                    if self.active_filter is not None:
                        extract.SetInputConnection(self.active_filter['filter'].GetOutputPort())
                    else:
                        extract.SetInputConnection(self.sources[self.active_pipeline]['reader'].GetOutputPort())
                    
                    extract.AddScalarVariable("vector", field_name, 0)
                    extract.SetFunction(f"vector.{component.lower()}")
                    extract.SetResultArrayName(f"{field_name}_{component}")
                    extract.Update()
                    
                    # Set output to mapper
                    mapper.SetInputConnection(extract.GetOutputPort())
                    mapper.SetScalarModeToUsePointFieldData()
                    mapper.SelectColorArray(f"{field_name}_{component}")
                    
                    # Store this filter if it's part of the active filter
                    if self.active_filter is not None:
                        self.active_filter['extract_component'] = extract
                
                else:
                    # Use magnitude for vector fields by default
                    calc = vtk.vtkArrayCalculator()
                    if self.active_filter is not None:
                        calc.SetInputConnection(self.active_filter['filter'].GetOutputPort())
                    else:
                        calc.SetInputConnection(self.sources[self.active_pipeline]['reader'].GetOutputPort())
                    
                    calc.AddVectorArrayName(field_name)
                    calc.SetFunction(f"mag({field_name})")
                    calc.SetResultArrayName(f"{field_name}_Magnitude")
                    calc.Update()
                    
                    # Set output to mapper
                    mapper.SetInputConnection(calc.GetOutputPort())
                    mapper.SetScalarModeToUsePointFieldData()
                    mapper.SelectColorArray(f"{field_name}_Magnitude")
                    
                    # Store this filter if it's part of the active filter
                    if self.active_filter is not None:
                        self.active_filter['calc_magnitude'] = calc
            else:
                # Scalar field - use directly
                mapper.SetScalarModeToUsePointFieldData()
                mapper.SelectColorArray(field_name)
            
            # Use a consistent color range for the field
            data_range = array.GetRange()
            mapper.SetScalarRange(data_range)
            
            # Enable scalar visibility
            mapper.ScalarVisibilityOn()
            
            # Render
            if self.render_window:
                self.render_window.Render()
            
            logger.info(f"Set coloring to field: {field_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting color by field: {e}")
            return False
    
    def set_color_map(self, color_map_name: str) -> bool:
        """
        Set the color map for the active pipeline.
        
        Args:
            color_map_name (str): Name of the color map
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("VTK not initialized")
            return False
        
        # Check if we have an active pipeline
        if not self.active_pipeline or not self.sources[self.active_pipeline]:
            logger.error("No active pipeline or source")
            return False
        
        try:
            # Create lookup table
            lut = create_lookup_table(color_map_name)
            
            # Apply to source mapper
            source = self.sources[self.active_pipeline]
            if 'mapper' in source:
                source['mapper'].SetLookupTable(lut)
            
            # Apply to filter mappers
            if self.active_pipeline in self.filters:
                for filter_info in self.filters[self.active_pipeline]:
                    if 'mapper' in filter_info:
                        filter_info['mapper'].SetLookupTable(lut)
            
            # Render
            if self.render_window:
                self.render_window.Render()
            
            logger.info(f"Set color map to {color_map_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting color map: {e}")
            return False
    
    def apply_oil_gas_visualization(self, viz_type: str, **parameters) -> bool:
        """
        Apply oil & gas specific visualization to the active pipeline.
        
        Args:
            viz_type (str): Type of visualization (phase_interface, flow_pattern, etc.)
            **parameters: Additional parameters
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("VTK not initialized")
            return False
        
        # Check if we have an active pipeline
        if not self.active_pipeline or not self.sources[self.active_pipeline]:
            logger.error("No active pipeline or source")
            return False
        
        try:
            # Apply different visualizations based on type
            if viz_type == "phase_interface":
                # Create a contour at 0.5 for the water phase fraction
                return self.add_filter("contour", {
                    "field": "alpha.water",
                    "values": [0.5]
                })
                
            elif viz_type == "flow_pattern":
                # Show velocity with streamlines
                return self.add_filter("streamline", {
                    "field": "U",
                    "seed_type": "plane"
                })
                
            elif viz_type == "velocity_profile":
                # Create a slice and color by velocity
                slice_success = self.add_filter("slice", {
                    "normal": [0, 0, 1],
                    "origin": [0, 0, 0]
                })
                
                if slice_success:
                    return self.color_by_field("U", "Magnitude")
                return False
                
            elif viz_type == "pressure_drop":
                # Color by pressure
                return self.color_by_field("p")
                
            elif viz_type == "pigging":
                # For pigging simulation, show pig location and surrounding fluid
                # This is a simplified implementation
                return self.apply_preset("Pigging Analysis")
                
            elif viz_type == "spill":
                # For spill visualization, show oil-water interface
                return self.apply_preset("Oil-Water Interface")
                
            else:
                logger.error(f"Unsupported visualization type: {viz_type}")
                return False
            
        except Exception as e:
            logger.error(f"Error applying oil & gas visualization: {e}")
            return False
    
    def save_state(self, filepath: str) -> bool:
        """
        Save the current state to a file.
        
        Args:
            filepath (str): Path to save the state
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("VTK not initialized")
            return False
        
        try:
            # Create a state file with basic information about the pipelines
            # This is a simplified implementation as full VTK state saving is complex
            state = {
                "active_pipeline": self.active_pipeline,
                "pipelines": {},
                "version": "1.0"
            }
            
            # Save basic pipeline information
            for name, source in self.sources.items():
                if source:
                    pipeline_info = {
                        "filepath": source.get('filepath', '')
                    }
                    
                    # Save filter information
                    if name in self.filters:
                        filters_info = []
                        for filter_info in self.filters[name]:
                            filter_data = {
                                "type": filter_info['type'],
                                "parameters": filter_info.get('parameters', {})
                            }
                            filters_info.append(filter_data)
                        
                        pipeline_info["filters"] = filters_info
                    
                    state["pipelines"][name] = pipeline_info
            
            # Save to file
            import json
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Saved state to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """
        Load a saved state from a file.
        
        Args:
            filepath (str): Path to the state file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("VTK not initialized")
            return False
        
        try:
            # Load state file
            import json
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Check version
            if 'version' not in state:
                logger.error("Invalid state file: missing version")
                return False
            
            # Clear existing pipelines
            self._clear_all_pipelines()
            
            # Load pipelines
            for name, pipeline_info in state.get('pipelines', {}).items():
                # Load source
                filepath = pipeline_info.get('filepath', '')
                if filepath and os.path.exists(filepath):
                    self.load_data(filepath, name)
                    
                    # Apply filters
                    for filter_info in pipeline_info.get('filters', []):
                        filter_type = filter_info.get('type', '')
                        parameters = filter_info.get('parameters', {})
                        
                        if filter_type:
                            self.add_filter(filter_type, parameters, name)
            
            # Set active pipeline
            active_pipeline = state.get('active_pipeline', '')
            if active_pipeline and active_pipeline in self.sources:
                self.set_active_pipeline(active_pipeline)
                self.display_pipeline(active_pipeline)
            
            logger.info(f"Loaded state from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False
    
    def _clear_all_pipelines(self):
        """Clear all pipelines and associated actors."""
        if self.renderer:
            # Remove all actors
            for name, source in self.sources.items():
                if 'actor' in source:
                    self.renderer.RemoveActor(source['actor'])
                
                if name in self.filters:
                    for filter_info in self.filters[name]:
                        if 'actor' in filter_info:
                            self.renderer.RemoveActor(filter_info['actor'])
        
        # Clear data structures
        self.sources = {}
        self.filters = {}
        self.active_pipeline = None
        self.active_source = None
        self.active_filter = None
    
    def export_image(self, filepath: str, width: int = 1200, height: int = 800) -> bool:
        """
        Export the current view to an image.
        
        Args:
            filepath (str): Path to save the image
            width (int): Width of the image
            height (int): Height of the image
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized or not self.render_window:
            logger.error("VTK not initialized or no render window")
            return False
        
        try:
            # Create a window to image filter
            window_to_image = vtk.vtkWindowToImageFilter()
            window_to_image.SetInput(self.render_window)
            window_to_image.SetScale(1)  # Set the resolution of the output image
            window_to_image.ReadFrontBufferOff()  # Read from the back buffer
            window_to_image.Update()
            
            # Determine file format
            ext = os.path.splitext(filepath)[1].lower()
            
            if ext == '.png':
                writer = vtk.vtkPNGWriter()
            elif ext in ['.jpg', '.jpeg']:
                writer = vtk.vtkJPEGWriter()
            elif ext == '.tif' or ext == '.tiff':
                writer = vtk.vtkTIFFWriter()
            else:
                # Default to PNG
                writer = vtk.vtkPNGWriter()
                if not filepath.lower().endswith('.png'):
                    filepath += '.png'
            
            # Set input and filename
            writer.SetFileName(filepath)
            writer.SetInputConnection(window_to_image.GetOutputPort())
            writer.Write()
            
            logger.info(f"Exported image to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting image: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        # Clear all pipelines
        self._clear_all_pipelines()
        
        # Remove renderer from render window
        if self.render_window and self.renderer:
            self.render_window.RemoveRenderer(self.renderer)
        
        # Clear references
        self.renderer = None
        self.camera = None
        self.render_window = None
        
        logger.info("Pipeline manager resources cleaned up")


# Create singleton instance
_pipeline_manager = None

def get_pipeline_manager(render_window=None):
    """
    Get the singleton pipeline manager instance.
    
    Args:
        render_window: Render window to use
        
    Returns:
        PipelineManager: Pipeline manager instance
    """
    global _pipeline_manager
    if _pipeline_manager is None:
        _pipeline_manager = PipelineManager(render_window)
    elif render_window is not None and _pipeline_manager.render_window != render_window:
        # Update render window
        _pipeline_manager.render_window = render_window
        
        # Re-initialize renderer if needed
        if render_window and not _pipeline_manager.renderer:
            _pipeline_manager.renderer = vtk.vtkRenderer()
            render_window.AddRenderer(_pipeline_manager.renderer)
            
            # Set up camera
            _pipeline_manager.camera = _pipeline_manager.renderer.GetActiveCamera()
    
    return _pipeline_manager

def cleanup_pipeline_manager():
    """Clean up the pipeline manager singleton instance."""
    global _pipeline_manager
    if _pipeline_manager is not None:
        _pipeline_manager.cleanup()
        _pipeline_manager = None