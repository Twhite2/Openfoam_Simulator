#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VTK bridge for Openfoam_Simulator application.

This module provides a bridge between the Openfoam_Simulator application and VTK,
enabling visualization of OpenFOAM results with specialized support for oil & gas 
industry applications. This bridge replaces the ParaView integration with a 
pure VTK implementation.
"""

import os
import sys
import logging
import threading
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

# Import utility modules
from ..utils.logger import get_logger

# Global logger
logger = get_logger(__name__)

# Check if VTK is available
try:
    # Import VTK modules
    import vtk
    from vtk.util import numpy_support
    
    VTK_AVAILABLE = True
    vtk_version = vtk.vtkVersion.GetVTKVersion()
    logger.info(f"VTK version {vtk_version} found")
except ImportError as e:
    logger.warning(f"VTK not available: {e}")
    VTK_AVAILABLE = False


class VTKBridge:
    """
    Bridge between Openfoam_Simulator application and VTK.
    
    This class provides an interface for using VTK functionality from
    within the Openfoam_Simulator application, with specific support for oil & gas
    industry visualizations.
    """
    
    def __init__(self, render_window=None):
        """
        Initialize the VTK bridge.
        
        Args:
            render_window: An existing render window to use (optional)
        """
        self.render_window = render_window
        self.renderer = None
        self.interactor = None
        self.active_source = None
        self.active_mapper = None
        self.active_actor = None
        self.pipeline = {}  # Dictionary of sources in the pipeline
        self.states = {}    # Dictionary of saved states
        self.initialized = False
        
        # Initialize VTK if available
        if VTK_AVAILABLE:
            self.initialize()
    
    def initialize(self) -> bool:
        """
        Initialize VTK and create a default view.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if not VTK_AVAILABLE:
            logger.error("Cannot initialize VTK: VTK is not available")
            return False
        
        try:
            # Create renderer if not already attached to the render window
            if self.render_window is not None:
                # Check if the render window already has a renderer
                if self.render_window.GetRenderers().GetNumberOfItems() > 0:
                    self.renderer = self.render_window.GetRenderers().GetFirstRenderer()
                else:
                    self.renderer = vtk.vtkRenderer()
                    self.render_window.AddRenderer(self.renderer)
                
                # Get interactor if available
                if self.render_window.GetInteractor() is not None:
                    self.interactor = self.render_window.GetInteractor()
            else:
                # Create new render window and renderer
                self.render_window = vtk.vtkRenderWindow()
                self.renderer = vtk.vtkRenderer()
                self.render_window.AddRenderer(self.renderer)
                
                # Create interactor
                self.interactor = vtk.vtkRenderWindowInteractor()
                self.interactor.SetRenderWindow(self.render_window)
                self.interactor.Initialize()
            
            # Apply default settings
            self.apply_default_settings()
            
            # Mark as initialized
            self.initialized = True
            logger.info("VTK bridge initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize VTK: {e}")
            self.initialized = False
            return False
    
    def apply_default_settings(self):
        """Apply default settings to the VTK renderer."""
        if not self.initialized or self.renderer is None:
            return
        
        try:
            # Set up the renderer with defaults
            self.renderer.SetBackground(0.2, 0.2, 0.2)  # Dark gray background
            
            # Add axes for orientation
            axes = vtk.vtkAxesActor()
            axes.SetTotalLength(1.0, 1.0, 1.0)
            axes.SetShaftTypeToCylinder()
            axes.SetXAxisLabelText("X")
            axes.SetYAxisLabelText("Y")
            axes.SetZAxisLabelText("Z")
            
            # Create a widget for the axes
            axes_widget = vtk.vtkOrientationMarkerWidget()
            axes_widget.SetOrientationMarker(axes)
            axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
            axes_widget.SetDefaultRenderer(self.renderer)
            axes_widget.EnabledOn()
            
            if self.interactor:
                axes_widget.SetInteractor(self.interactor)
                axes_widget.On()
            
            # Store the widget to prevent garbage collection
            self.axes_widget = axes_widget
            
            # Set default lighting
            light = vtk.vtkLight()
            light.SetLightTypeToHeadlight()
            light.SetIntensity(0.75)
            self.renderer.AddLight(light)
            
            # Update render window
            if self.render_window is not None:
                self.render_window.Render()
                
        except Exception as e:
            logger.error(f"Error applying default settings: {e}")
    
    def load_data(self, filepath: str, file_type: str = None) -> bool:
        """
        Load data from a file into VTK.
        
        Args:
            filepath (str): Path to the data file
            file_type (str, optional): Type of file ('openfoam', 'vtk', 'stl', etc.)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("VTK bridge not initialized")
            return False
        
        try:
            # Determine file type if not provided
            if file_type is None:
                ext = os.path.splitext(filepath)[1].lower()
                if ext == '.foam' or ext == '.openfoam':
                    file_type = 'openfoam'
                elif ext in ['.vtk', '.vtu', '.vtp', '.vts', '.vtr', '.pvd']:
                    file_type = 'vtk'
                elif ext == '.stl':
                    file_type = 'stl'
                elif ext in ['.obj', '.ply']:
                    file_type = 'geometry'
                else:
                    logger.warning(f"Unknown file extension: {ext}, trying to auto-detect")
            
            # Load data based on file type
            reader = None
            
            if file_type == 'vtk':
                # Legacy VTK file
                reader = vtk.vtkDataSetReader()
                reader.SetFileName(filepath)
                
            elif file_type == 'vtu':
                # XML Unstructured Grid file
                reader = vtk.vtkXMLUnstructuredGridReader()
                reader.SetFileName(filepath)
                
            elif file_type == 'vtp':
                # XML PolyData file
                reader = vtk.vtkXMLPolyDataReader()
                reader.SetFileName(filepath)
                
            elif file_type == 'stl':
                # STL file
                reader = vtk.vtkSTLReader()
                reader.SetFileName(filepath)
                
            elif file_type == 'obj':
                # OBJ file
                reader = vtk.vtkOBJReader()
                reader.SetFileName(filepath)
                
            elif file_type == 'ply':
                # PLY file
                reader = vtk.vtkPLYReader()
                reader.SetFileName(filepath)
                
            elif file_type == 'openfoam':
                # For OpenFOAM, we need a different approach
                # This is a simplified placeholder - real implementation would need to
                # parse OpenFOAM directory structure or use helper libraries
                logger.warning("OpenFOAM support requires either the OpenFOAM reader for VTK or conversion to VTK format")
                logger.warning("Using a placeholder implementation")
                
                # Check if there's a VTK file from OpenFOAM conversion
                vtk_path = filepath + "/VTK/internal.vtk"
                if os.path.exists(vtk_path):
                    reader = vtk.vtkDataSetReader()
                    reader.SetFileName(vtk_path)
                else:
                    logger.error(f"No converted VTK file found in OpenFOAM case directory")
                    return False
                
            else:
                logger.error(f"Unsupported file type: {file_type}")
                return False
            
            # Update reader
            if reader is None:
                logger.error(f"Failed to create reader for {filepath}")
                return False
            
            reader.Update()
            
            # Create mapper and actor
            if file_type in ['stl', 'obj', 'ply', 'vtp']:
                # Create PolyData mapper
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(reader.GetOutputPort())
            else:
                # Create DataSet mapper
                mapper = vtk.vtkDataSetMapper()
                mapper.SetInputConnection(reader.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            # Remove any existing actors
            if self.active_actor:
                self.renderer.RemoveActor(self.active_actor)
            
            # Add actor to renderer
            self.renderer.AddActor(actor)
            
            # Store active objects
            self.active_source = reader
            self.active_mapper = mapper
            self.active_actor = actor
            
            # Add to pipeline
            source_name = os.path.basename(filepath)
            self.pipeline[source_name] = {
                'source': reader,
                'mapper': mapper,
                'actor': actor
            }
            
            # Reset view to fit data
            self.reset_camera()
            
            # Update render window
            if self.render_window is not None:
                self.render_window.Render()
            
            logger.info(f"Successfully loaded data from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data from {filepath}: {e}")
            return False
    
    def create_filter(self, filter_type: str, **parameters) -> Optional[Tuple[Any, Any, Any]]:
        """
        Create a filter in the VTK pipeline.
        
        Args:
            filter_type (str): Type of filter to create
            **parameters: Additional parameters for the filter
            
        Returns:
            Optional[Tuple[Any, Any, Any]]: The created filter, mapper, and actor if successful, None otherwise
        """
        if not self.initialized:
            logger.error("VTK bridge not initialized")
            return None
        
        if self.active_source is None:
            logger.error("No active source to apply filter to")
            return None
        
        try:
            # Create filter based on type
            filter_obj = None
            output_type = "polydata"  # Default output type
            
            if filter_type.lower() == 'slice':
                # Create a slice filter
                filter_obj = vtk.vtkCutter()
                filter_obj.SetInputConnection(self.active_source.GetOutputPort())
                
                # Create a plane to define the cut
                plane = vtk.vtkPlane()
                
                # Set plane parameters
                normal = parameters.get('normal', [0, 0, 1])
                origin = parameters.get('origin', [0, 0, 0])
                
                plane.SetNormal(normal)
                plane.SetOrigin(origin)
                
                # Set the plane as the cut function
                filter_obj.SetCutFunction(plane)
                
            elif filter_type.lower() == 'clip':
                # Create a clip filter
                filter_obj = vtk.vtkClipDataSet()
                filter_obj.SetInputConnection(self.active_source.GetOutputPort())
                
                # Create a plane to define the clip
                plane = vtk.vtkPlane()
                
                # Set plane parameters
                normal = parameters.get('normal', [0, 0, 1])
                origin = parameters.get('origin', [0, 0, 0])
                
                plane.SetNormal(normal)
                plane.SetOrigin(origin)
                
                # Set the plane as the clip function
                filter_obj.SetClipFunction(plane)
                
                # Set inside out parameter
                inside_out = parameters.get('inside_out', False)
                filter_obj.SetInsideOut(inside_out)
                
                output_type = "dataset"
                
            elif filter_type.lower() == 'threshold':
                # Create a threshold filter
                filter_obj = vtk.vtkThreshold()
                filter_obj.SetInputConnection(self.active_source.GetOutputPort())
                
                # Set threshold parameters
                min_value = parameters.get('min_value', 0.0)
                max_value = parameters.get('max_value', 1.0)
                
                filter_obj.ThresholdBetween(min_value, max_value)
                
                # Set field to threshold on
                field = parameters.get('field', '')
                if field:
                    filter_obj.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, field)
                
                output_type = "dataset"
                
            elif filter_type.lower() == 'contour':
                # Create a contour filter
                filter_obj = vtk.vtkContourFilter()
                filter_obj.SetInputConnection(self.active_source.GetOutputPort())
                
                # Set contour parameters
                field = parameters.get('field', '')
                if field:
                    filter_obj.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, field)
                
                # Handle different ways to specify contour values
                if 'values' in parameters:
                    # Use explicitly provided values
                    values = parameters['values']
                    for i, val in enumerate(values):
                        filter_obj.SetValue(i, val)
                elif 'count' in parameters and 'min_value' in parameters and 'max_value' in parameters:
                    # Generate evenly spaced values
                    count = parameters['count']
                    min_val = parameters['min_value']
                    max_val = parameters['max_value']
                    step = (max_val - min_val) / (count - 1) if count > 1 else 0
                    
                    for i in range(count):
                        filter_obj.SetValue(i, min_val + i * step)
                else:
                    # Default to a single contour at 0.0
                    filter_obj.SetValue(0, 0.0)
                
            elif filter_type.lower() == 'streamtracer' or filter_type.lower() == 'streamline':
                # Create a streamline filter
                filter_obj = vtk.vtkStreamTracer()
                filter_obj.SetInputConnection(self.active_source.GetOutputPort())
                
                # Set streamline parameters
                field = parameters.get('field', '')
                if field:
                    filter_obj.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, field)
                
                # Set maximum length
                max_length = parameters.get('max_length', 100.0)
                filter_obj.SetMaximumPropagation(max_length)
                
                # Set integration direction
                filter_obj.SetIntegrationDirectionToForward()
                
                # Set up seed points
                seed_type = parameters.get('seed_type', 'point')
                
                if seed_type.lower() == 'point':
                    # Create a point source for seeding
                    seed = vtk.vtkPointSource()
                    seed.SetNumberOfPoints(1)
                    seed.SetCenter(0, 0, 0)
                    seed.Update()
                    
                    filter_obj.SetSourceConnection(seed.GetOutputPort())
                    
                elif seed_type.lower() == 'line':
                    # Create a line source for seeding
                    seed = vtk.vtkLineSource()
                    seed.SetPoint1(-1, 0, 0)
                    seed.SetPoint2(1, 0, 0)
                    seed.SetResolution(20)
                    seed.Update()
                    
                    filter_obj.SetSourceConnection(seed.GetOutputPort())
                    
                elif seed_type.lower() == 'plane':
                    # Create a plane source for seeding
                    seed = vtk.vtkPlaneSource()
                    seed.SetXResolution(10)
                    seed.SetYResolution(10)
                    seed.Update()
                    
                    filter_obj.SetSourceConnection(seed.GetOutputPort())
                
            elif filter_type.lower() == 'glyph':
                # Create a glyph filter
                filter_obj = vtk.vtkGlyph3D()
                filter_obj.SetInputConnection(self.active_source.GetOutputPort())
                
                # Create arrow source for glyphs
                arrow = vtk.vtkArrowSource()
                arrow.Update()
                
                filter_obj.SetSourceConnection(arrow.GetOutputPort())
                
                # Set vector field for orientation and scaling
                vector_field = parameters.get('vector_field', '')
                if vector_field:
                    filter_obj.SetInputArrayToProcess(1, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, vector_field)
                
                # Set scale mode
                scale_mode = parameters.get('scale_mode', 'scalar')
                if scale_mode.lower() == 'scalar':
                    filter_obj.SetScaleModeToScaleByScalar()
                elif scale_mode.lower() == 'vector':
                    filter_obj.SetScaleModeToScaleByVector()
                else:
                    filter_obj.SetScaleModeToDataScalingOff()
                
                # Set scale factor
                scale_factor = parameters.get('scale_factor', 1.0)
                filter_obj.SetScaleFactor(scale_factor)
                
            else:
                logger.error(f"Unsupported filter type: {filter_type}")
                return None
            
            # Update the filter
            if filter_obj is None:
                logger.error(f"Failed to create {filter_type} filter")
                return None
            
            filter_obj.Update()
            
            # Create appropriate mapper based on output type
            mapper = None
            if output_type == "polydata":
                mapper = vtk.vtkPolyDataMapper()
            else:  # "dataset"
                mapper = vtk.vtkDataSetMapper()
            
            mapper.SetInputConnection(filter_obj.GetOutputPort())
            
            # Create actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            # Remove any existing actors
            if self.active_actor:
                self.renderer.RemoveActor(self.active_actor)
            
            # Add actor to renderer
            self.renderer.AddActor(actor)
            
            # Store active objects
            self.active_source = filter_obj
            self.active_mapper = mapper
            self.active_actor = actor
            
            # Add to pipeline
            filter_name = f"{filter_type}_{len(self.pipeline)}"
            self.pipeline[filter_name] = {
                'source': filter_obj,
                'mapper': mapper,
                'actor': actor
            }
            
            # Update render window
            if self.render_window is not None:
                self.render_window.Render()
            
            logger.info(f"Created {filter_type} filter")
            return (filter_obj, mapper, actor)
            
        except Exception as e:
            logger.error(f"Failed to create {filter_type} filter: {e}")
            return None
    
    def set_representation(self, representation: str) -> bool:
        """
        Set the representation type for the active actor.
        
        Args:
            representation (str): Representation type
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized or self.active_actor is None:
            logger.error("VTK bridge not initialized or no active actor")
            return False
        
        try:
            # Map representation types to VTK properties
            representation_map = {
                'Surface': vtk.VTK_SURFACE,
                'Wireframe': vtk.VTK_WIREFRAME,
                'Points': vtk.VTK_POINTS,
                'Surface With Edges': vtk.VTK_SURFACE  # Will need additional property setting
            }
            
            # Get VTK representation constant
            vtk_rep = representation_map.get(representation, vtk.VTK_SURFACE)
            
            # Set representation property
            self.active_actor.GetProperty().SetRepresentation(vtk_rep)
            
            # Handle "Surface With Edges" special case
            if representation == 'Surface With Edges':
                self.active_actor.GetProperty().EdgeVisibilityOn()
            else:
                self.active_actor.GetProperty().EdgeVisibilityOff()
            
            # Update render window
            if self.render_window is not None:
                self.render_window.Render()
            
            logger.info(f"Set representation to {representation}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set representation: {e}")
            return False
    
    def color_by_field(self, field_name: str, component: str = None) -> bool:
        """
        Color the active actor by a specific field.
        
        Args:
            field_name (str): Name of the field to color by
            component (str, optional): Component for vector fields ('X', 'Y', 'Z', 'Magnitude')
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized or self.active_mapper is None:
            logger.error("VTK bridge not initialized or no active mapper")
            return False
        
        try:
            # Handle solid color case
            if field_name.lower() == 'solid color':
                self.active_mapper.ScalarVisibilityOff()
                return True
            
            # Get output from the active source
            if hasattr(self.active_source, 'GetOutput'):
                data = self.active_source.GetOutput()
            else:
                logger.error("Active source does not have GetOutput method")
                return False
            
            # Check if field exists in point data
            point_data = data.GetPointData()
            cell_data = data.GetCellData()
            
            array = None
            is_point_data = False
            
            # Check point data first
            for i in range(point_data.GetNumberOfArrays()):
                if point_data.GetArrayName(i) == field_name:
                    array = point_data.GetArray(i)
                    is_point_data = True
                    break
            
            # If not found in point data, check cell data
            if array is None:
                for i in range(cell_data.GetNumberOfArrays()):
                    if cell_data.GetArrayName(i) == field_name:
                        array = cell_data.GetArray(i)
                        is_point_data = False
                        break
            
            if array is None:
                logger.warning(f"Field {field_name} not found in data")
                return False
            
            # Handle vector components
            if array.GetNumberOfComponents() > 1 and component:
                # Create a new array for the component
                component_array = vtk.vtkDoubleArray()
                component_array.SetNumberOfComponents(1)
                component_array.SetNumberOfTuples(array.GetNumberOfTuples())
                component_array.SetName(f"{field_name}_{component}")
                
                if component == 'Magnitude':
                    # Calculate magnitude of vector
                    for i in range(array.GetNumberOfTuples()):
                        vector = [array.GetComponent(i, j) for j in range(array.GetNumberOfComponents())]
                        magnitude = sum(v*v for v in vector) ** 0.5
                        component_array.SetValue(i, magnitude)
                elif component in ['X', 'Y', 'Z']:
                    # Get component by index
                    comp_idx = {'X': 0, 'Y': 1, 'Z': 2}.get(component, 0)
                    if comp_idx < array.GetNumberOfComponents():
                        for i in range(array.GetNumberOfTuples()):
                            component_array.SetValue(i, array.GetComponent(i, comp_idx))
                
                # Add the component array to the dataset
                if is_point_data:
                    point_data.AddArray(component_array)
                    point_data.SetActiveScalars(component_array.GetName())
                else:
                    cell_data.AddArray(component_array)
                    cell_data.SetActiveScalars(component_array.GetName())
                
                # Set scalar mode based on point or cell data
                if is_point_data:
                    self.active_mapper.SetScalarModeToUsePointData()
                else:
                    self.active_mapper.SetScalarModeToUseCellData()
                
                # Set the active array
                self.active_mapper.SelectColorArray(component_array.GetName())
                
            else:
                # Single component or no component specified
                # Set active scalars
                if is_point_data:
                    point_data.SetActiveScalars(field_name)
                    self.active_mapper.SetScalarModeToUsePointData()
                else:
                    cell_data.SetActiveScalars(field_name)
                    self.active_mapper.SetScalarModeToUseCellData()
                
                # Set the active array
                self.active_mapper.SelectColorArray(field_name)
            
            # Get data range for scaling
            if array.GetNumberOfComponents() == 1:
                data_range = array.GetRange()
            else:
                # For vector fields, use specified component or magnitude
                if component == 'Magnitude':
                    data_range = array.GetRange(-1)  # -1 means magnitude
                elif component in ['X', 'Y', 'Z']:
                    comp_idx = {'X': 0, 'Y': 1, 'Z': 2}.get(component, 0)
                    data_range = array.GetRange(comp_idx)
                else:
                    data_range = array.GetRange(-1)  # Default to magnitude
            
            # Create lookup table
            lut = vtk.vtkLookupTable()
            lut.SetHueRange(0.667, 0.0)  # Blue to red
            lut.SetNumberOfTableValues(256)
            lut.Build()
            
            # Set lookup table to mapper
            self.active_mapper.SetLookupTable(lut)
            
            # Set scalar range for mapper
            self.active_mapper.SetScalarRange(data_range)
            
            # Enable scalar visibility
            self.active_mapper.ScalarVisibilityOn()
            
            # Add scalar bar (color legend)
            scalar_bar = vtk.vtkScalarBarActor()
            scalar_bar.SetLookupTable(self.active_mapper.GetLookupTable())
            scalar_bar.SetTitle(field_name)
            scalar_bar.SetNumberOfLabels(5)
            scalar_bar.SetWidth(0.1)
            scalar_bar.SetHeight(0.5)
            scalar_bar.SetPosition(0.9, 0.25)
            
            # Store scalar bar for later access
            if hasattr(self, 'scalar_bar') and self.scalar_bar is not None:
                self.renderer.RemoveActor2D(self.scalar_bar)
            
            self.renderer.AddActor2D(scalar_bar)
            self.scalar_bar = scalar_bar
            
            # Update render window
            if self.render_window is not None:
                self.render_window.Render()
            
            logger.info(f"Colored by field {field_name} {component if component else ''}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to color by field {field_name}: {e}")
            return False
    
    def set_color_map(self, color_map_name: str) -> bool:
        """
        Set the color map for the active mapper.
        
        Args:
            color_map_name (str): Name of the color map
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized or self.active_mapper is None:
            logger.error("VTK bridge not initialized or no active mapper")
            return False
        
        try:
            # Check if scalar mapping is enabled
            if not self.active_mapper.GetScalarVisibility():
                logger.warning("Scalar visibility is off, coloring solid")
                return False
            
            # Create lookup table based on color map name
            lut = vtk.vtkLookupTable()
            lut.SetNumberOfTableValues(256)
            
            # Define color maps
            if color_map_name == 'Rainbow':
                lut.SetHueRange(0.667, 0.0)  # Blue to red
            elif color_map_name == 'Cool to Warm':
                lut.SetHueRange(0.667, 0.0)  # Blue to red
                lut.SetSaturationRange(0.8, 0.8)
                lut.SetValueRange(1.0, 1.0)
            elif color_map_name == 'Viridis':
                # Viridis-like colors (approximation)
                for i in range(256):
                    t = i / 255.0
                    r = 0.267 + 0.534 * t
                    g = 0.004 + 0.978 * t - 1.577 * t**2
                    b = 0.329 + 0.036 * t**3
                    lut.SetTableValue(i, r, g, b, 1.0)
            elif color_map_name == 'Plasma':
                # Plasma-like colors (approximation)
                for i in range(256):
                    t = i / 255.0
                    r = 0.05 + 0.95 * t
                    g = 0.0 + 0.85 * t - 0.85 * t**2
                    b = 0.5 - 0.5 * t
                    lut.SetTableValue(i, r, g, b, 1.0)
            elif color_map_name == 'Blues':
                lut.SetHueRange(0.667, 0.667)  # Blue
                lut.SetSaturationRange(0.1, 1.0)
                lut.SetValueRange(1.0, 0.5)
            elif color_map_name == 'Reds':
                lut.SetHueRange(0.0, 0.0)  # Red
                lut.SetSaturationRange(0.1, 1.0)
                lut.SetValueRange(1.0, 0.5)
            elif color_map_name == 'Greens':
                lut.SetHueRange(0.333, 0.333)  # Green
                lut.SetSaturationRange(0.1, 1.0)
                lut.SetValueRange(1.0, 0.5)
            elif color_map_name == 'Jet':
                # Jet-like colors
                for i in range(256):
                    t = i / 255.0
                    if t < 0.125:
                        r, g, b = 0, 0, 0.5 + 4 * t
                    elif t < 0.375:
                        r, g, b = 0, 4 * (t - 0.125), 1
                    elif t < 0.625:
                        r, g, b = 4 * (t - 0.375), 1, 1 - 4 * (t - 0.375)
                    elif t < 0.875:
                        r, g, b = 1, 1 - 4 * (t - 0.625), 0
                    else:
                        r, g, b = 1 - 4 * (t - 0.875), 0, 0
                    lut.SetTableValue(i, r, g, b, 1.0)
            elif color_map_name == 'Black to White':
                lut.SetHueRange(0, 0)
                lut.SetSaturationRange(0, 0)
                lut.SetValueRange(0, 1)
            elif color_map_name == 'Pressure':
                lut.SetHueRange(0.667, 0.0)  # Blue to red
                lut.SetSaturationRange(0.8, 0.8)
                lut.SetValueRange(1.0, 1.0)
            elif color_map_name == 'Velocity':
                # Velocity-specific colors
                for i in range(256):
                    t = i / 255.0
                    if t < 0.25:
                        r, g, b = 0, 0, 0.5 + 2 * t
                    elif t < 0.5:
                        r, g, b = 0, 4 * (t - 0.25), 1
                    elif t < 0.75:
                        r, g, b = 4 * (t - 0.5), 1, 1 - 4 * (t - 0.5)
                    else:
                        r, g, b = 1, 1 - 4 * (t - 0.75), 0
                    lut.SetTableValue(i, r, g, b, 1.0)
            elif color_map_name == 'Oil-Water':
                # Oil-Water specific colors
                for i in range(256):
                    t = i / 255.0
                    if t < 0.5:
                        r, g, b = 0, 0, 0.5 + t
                    else:
                        r, g, b = (t - 0.5) * 2, (t - 0.5) * 1.6, 1 - (t - 0.5) * 2
                    lut.SetTableValue(i, r, g, b, 1.0)
            else:
                # Default rainbow
                lut.SetHueRange(0.667, 0.0)
            
            lut.Build()
            
            # Get current scalar range
            scalar_range = self.active_mapper.GetScalarRange()
            
            # Set lookup table to mapper
            self.active_mapper.SetLookupTable(lut)
            
            # Keep original scalar range
            self.active_mapper.SetScalarRange(scalar_range)
            
            # Update scalar bar if it exists
            if hasattr(self, 'scalar_bar') and self.scalar_bar is not None:
                self.scalar_bar.SetLookupTable(lut)
            
            # Update render window
            if self.render_window is not None:
                self.render_window.Render()
            
            logger.info(f"Set color map to {color_map_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set color map {color_map_name}: {e}")
            return False
    
    def set_solid_color(self, color: List[float]) -> bool:
        """
        Set a solid color for the active actor.
        
        Args:
            color (List[float]): RGB color values [0-1, 0-1, 0-1]
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized or self.active_actor is None:
            logger.error("VTK bridge not initialized or no active actor")
            return False
        
        try:
            # Disable scalar coloring
            if self.active_mapper:
                self.active_mapper.ScalarVisibilityOff()
            
            # Set solid color to actor
            self.active_actor.GetProperty().SetColor(color)
            
            # Remove scalar bar if present
            if hasattr(self, 'scalar_bar') and self.scalar_bar is not None:
                self.renderer.RemoveActor2D(self.scalar_bar)
                self.scalar_bar = None
            
            # Update render window
            if self.render_window is not None:
                self.render_window.Render()
            
            logger.info(f"Set solid color to {color}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set solid color: {e}")
            return False
    
    def set_camera(self, position: List[float] = None, focal_point: List[float] = None,
                  view_up: List[float] = None, view_angle: float = None) -> bool:
        """
        Set camera parameters for the renderer.
        
        Args:
            position (List[float], optional): Camera position [x, y, z]
            focal_point (List[float], optional): Camera focal point [x, y, z]
            view_up (List[float], optional): Camera view up vector [x, y, z]
            view_angle (float, optional): Camera view angle
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized or self.renderer is None:
            logger.error("VTK bridge not initialized or no renderer")
            return False
        
        try:
            # Get the camera
            camera = self.renderer.GetActiveCamera()
            
            # Set camera parameters
            if position is not None:
                camera.SetPosition(position)
            
            if focal_point is not None:
                camera.SetFocalPoint(focal_point)
            
            if view_up is not None:
                camera.SetViewUp(view_up)
            
            if view_angle is not None:
                camera.SetViewAngle(view_angle)
            
            # Update render window
            if self.render_window is not None:
                self.render_window.Render()
            
            logger.info("Camera parameters updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set camera parameters: {e}")
            return False
    
    def reset_camera(self) -> bool:
        """
        Reset the camera to fit the data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized or self.renderer is None:
            logger.error("VTK bridge not initialized or no renderer")
            return False
        
        try:
            # Reset camera
            self.renderer.ResetCamera()
            
            # Update render window
            if self.render_window is not None:
                self.render_window.Render()
            
            logger.info("Camera reset")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset camera: {e}")
            return False
    
    def set_view_properties(self, background_color: List[float] = None,
                         show_axes: bool = None, show_grid: bool = None) -> bool:
        """
        Set view properties for the renderer.
        
        Args:
            background_color (List[float], optional): Background color [r, g, b]
            show_axes (bool, optional): Whether to show axes
            show_grid (bool, optional): Whether to show grid
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized or self.renderer is None:
            logger.error("VTK bridge not initialized or no renderer")
            return False
        
        try:
            # Set background color
            if background_color is not None:
                self.renderer.SetBackground(background_color)
            
            # Show/hide axes
            if show_axes is not None and hasattr(self, 'axes_widget'):
                if show_axes:
                    self.axes_widget.On()
                else:
                    self.axes_widget.Off()
            
            # Show/hide grid
            if show_grid is not None:
                # Check if grid already exists
                grid_exists = hasattr(self, 'grid_actor') and self.grid_actor is not None
                
                if show_grid and not grid_exists:
                    # Create grid
                    grid = vtk.vtkRectilinearGrid()
                    grid.SetDimensions(2, 2, 2)
                    
                    # Create grid actor
                    mapper = vtk.vtkDataSetMapper()
                    mapper.SetInputData(grid)
                    
                    grid_actor = vtk.vtkActor()
                    grid_actor.SetMapper(mapper)
                    grid_actor.GetProperty().SetRepresentationToWireframe()
                    grid_actor.GetProperty().SetColor(0.7, 0.7, 0.7)
                    
                    # Add to renderer
                    self.renderer.AddActor(grid_actor)
                    self.grid_actor = grid_actor
                    
                elif not show_grid and grid_exists:
                    # Remove grid
                    self.renderer.RemoveActor(self.grid_actor)
                    self.grid_actor = None
            
            # Update render window
            if self.render_window is not None:
                self.render_window.Render()
            
            logger.info("View properties updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set view properties: {e}")
            return False
    
    def apply_oil_gas_visualization(self, viz_type: str, **parameters) -> bool:
        """
        Apply oil & gas specific visualization.
        
        Args:
            viz_type (str): Type of visualization (phase_interface, flow_pattern, etc.)
            **parameters: Additional parameters for the visualization
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized or self.active_source is None:
            logger.error("VTK bridge not initialized or no active source")
            return False
        
        try:
            # Apply specific oil & gas visualization
            if viz_type.lower() == 'phase_interface':
                # Create contour at 0.5 value for alpha.water or similar phase field
                field_name = parameters.get('field_name', 'alpha.water')
                
                # Create contour filter
                contour = vtk.vtkContourFilter()
                contour.SetInputConnection(self.active_source.GetOutputPort())
                contour.SetValue(0, 0.5)  # Interface at 0.5 value
                
                # Check if field exists and set it as active
                output = self.active_source.GetOutput()
                point_data = output.GetPointData()
                
                field_exists = False
                for i in range(point_data.GetNumberOfArrays()):
                    if point_data.GetArrayName(i) == field_name:
                        field_exists = True
                        point_data.SetActiveScalars(field_name)
                        break
                
                if not field_exists:
                    logger.warning(f"Field {field_name} not found in data")
                    return False
                
                contour.Update()
                
                # Create mapper and actor
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(contour.GetOutputPort())
                
                # Set up oil-water color scheme
                if 'color' in parameters:
                    # Use specified color
                    color = parameters['color']
                    mapper.ScalarVisibilityOff()
                else:
                    # Blue to brown gradient for oil-water
                    lut = vtk.vtkLookupTable()
                    lut.SetNumberOfTableValues(256)
                    
                    for i in range(256):
                        t = i / 255.0
                        if t < 0.5:
                            r, g, b = 0, 0, 0.5 + t
                        else:
                            r, g, b = (t - 0.5) * 2, (t - 0.5) * 1.6, 1 - (t - 0.5) * 2
                        lut.SetTableValue(i, r, g, b, 1.0)
                    
                    lut.Build()
                    mapper.SetLookupTable(lut)
                    mapper.SetScalarRange(0, 1)
                    mapper.ScalarVisibilityOn()
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                
                # Remove existing actors
                if self.active_actor:
                    self.renderer.RemoveActor(self.active_actor)
                
                # Add new actor
                self.renderer.AddActor(actor)
                
                # Update active objects
                self.active_source = contour
                self.active_mapper = mapper
                self.active_actor = actor
                
                # Add to pipeline
                self.pipeline['phase_interface'] = {
                    'source': contour,
                    'mapper': mapper,
                    'actor': actor
                }
                
            elif viz_type.lower() == 'flow_pattern':
                # This creates streamlines to show flow patterns
                field_name = parameters.get('field_name', 'U')
                
                # Create streamlines
                stream = vtk.vtkStreamTracer()
                stream.SetInputConnection(self.active_source.GetOutputPort())
                
                # Check if field exists
                output = self.active_source.GetOutput()
                point_data = output.GetPointData()
                
                field_exists = False
                for i in range(point_data.GetNumberOfArrays()):
                    if point_data.GetArrayName(i) == field_name:
                        field_exists = True
                        point_data.SetActiveVectors(field_name)
                        break
                
                if not field_exists:
                    logger.warning(f"Vector field {field_name} not found in data")
                    return False
                
                # Set up streamline parameters
                stream.SetMaximumPropagation(parameters.get('max_length', 10.0))
                stream.SetIntegrationDirectionToForward()
                
                # Create source for streamline seeds
                if 'seed_type' in parameters:
                    if parameters['seed_type'].lower() == 'point':
                        point_source = vtk.vtkPointSource()
                        point_source.SetNumberOfPoints(10)
                        point_source.SetCenter(0, 0, 0)
                        point_source.SetRadius(1.0)
                        point_source.Update()
                        
                        stream.SetSourceConnection(point_source.GetOutputPort())
                        
                    elif parameters['seed_type'].lower() == 'line':
                        line_source = vtk.vtkLineSource()
                        line_source.SetPoint1(-10, 0, 0)
                        line_source.SetPoint2(10, 0, 0)
                        line_source.SetResolution(20)
                        line_source.Update()
                        
                        stream.SetSourceConnection(line_source.GetOutputPort())
                        
                    elif parameters['seed_type'].lower() == 'plane':
                        plane_source = vtk.vtkPlaneSource()
                        plane_source.SetOrigin(-5, -5, 0)
                        plane_source.SetPoint1(5, -5, 0)
                        plane_source.SetPoint2(-5, 5, 0)
                        plane_source.SetXResolution(10)
                        plane_source.SetYResolution(10)
                        plane_source.Update()
                        
                        stream.SetSourceConnection(plane_source.GetOutputPort())
                else:
                    # Default to point source
                    point_source = vtk.vtkPointSource()
                    point_source.SetNumberOfPoints(10)
                    point_source.SetCenter(0, 0, 0)
                    point_source.SetRadius(1.0)
                    point_source.Update()
                    
                    stream.SetSourceConnection(point_source.GetOutputPort())
                
                stream.Update()
                
                # Create mapper and actor
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(stream.GetOutputPort())
                
                # Color by velocity magnitude
                lut = vtk.vtkLookupTable()
                lut.SetHueRange(0.667, 0.0)  # Blue to red
                lut.Build()
                
                mapper.SetLookupTable(lut)
                mapper.SetScalarRange(0, 10)  # This should be adjusted based on actual data range
                mapper.ScalarVisibilityOn()
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                
                # Remove existing actors
                if self.active_actor:
                    self.renderer.RemoveActor(self.active_actor)
                
                # Add new actor
                self.renderer.AddActor(actor)
                
                # Update active objects
                self.active_source = stream
                self.active_mapper = mapper
                self.active_actor = actor
                
                # Add to pipeline
                self.pipeline['flow_pattern'] = {
                    'source': stream,
                    'mapper': mapper,
                    'actor': actor
                }
                
            elif viz_type.lower() == 'velocity_profile':
                # Create a slice with vector glyphs to show velocity profile
                field_name = parameters.get('field_name', 'U')
                normal = parameters.get('normal', [0, 0, 1])
                
                # Create a slice filter
                slice_filter = vtk.vtkCutter()
                slice_filter.SetInputConnection(self.active_source.GetOutputPort())
                
                # Create a plane for the slice
                plane = vtk.vtkPlane()
                plane.SetOrigin(0, 0, 0)
                plane.SetNormal(normal)
                slice_filter.SetCutFunction(plane)
                
                slice_filter.Update()
                
                # Check if field exists
                output = self.active_source.GetOutput()
                point_data = output.GetPointData()
                
                field_exists = False
                for i in range(point_data.GetNumberOfArrays()):
                    if point_data.GetArrayName(i) == field_name:
                        field_exists = True
                        point_data.SetActiveVectors(field_name)
                        break
                
                if not field_exists:
                    logger.warning(f"Vector field {field_name} not found in data")
                    return False
                
                # Create glyphs for velocity vectors
                arrow = vtk.vtkArrowSource()
                arrow.Update()
                
                glyph = vtk.vtkGlyph3D()
                glyph.SetInputConnection(slice_filter.GetOutputPort())
                glyph.SetSourceConnection(arrow.GetOutputPort())
                glyph.SetScaleFactor(parameters.get('scale_factor', 0.1))
                glyph.SetScaleModeToScaleByVector()
                glyph.SetVectorModeToUseVector()
                glyph.OrientOn()
                glyph.Update()
                
                # Create mapper and actor for slice
                slice_mapper = vtk.vtkPolyDataMapper()
                slice_mapper.SetInputConnection(slice_filter.GetOutputPort())
                
                # Color by velocity magnitude
                lut = vtk.vtkLookupTable()
                lut.SetHueRange(0.667, 0.0)  # Blue to red
                lut.Build()
                
                slice_mapper.SetLookupTable(lut)
                slice_mapper.SetScalarRange(0, 10)  # Adjust based on data range
                slice_mapper.ScalarVisibilityOn()
                
                slice_actor = vtk.vtkActor()
                slice_actor.SetMapper(slice_mapper)
                
                # Create mapper and actor for glyphs
                glyph_mapper = vtk.vtkPolyDataMapper()
                glyph_mapper.SetInputConnection(glyph.GetOutputPort())
                glyph_mapper.ScalarVisibilityOff()
                
                glyph_actor = vtk.vtkActor()
                glyph_actor.SetMapper(glyph_mapper)
                glyph_actor.GetProperty().SetColor(0, 0, 0)  # Black arrows
                
                # Remove existing actors
                if self.active_actor:
                    self.renderer.RemoveActor(self.active_actor)
                
                # Add new actors
                self.renderer.AddActor(slice_actor)
                self.renderer.AddActor(glyph_actor)
                
                # Update active objects (use slice as active)
                self.active_source = slice_filter
                self.active_mapper = slice_mapper
                self.active_actor = slice_actor
                
                # Add to pipeline
                self.pipeline['velocity_profile_slice'] = {
                    'source': slice_filter,
                    'mapper': slice_mapper,
                    'actor': slice_actor
                }
                
                self.pipeline['velocity_profile_glyph'] = {
                    'source': glyph,
                    'mapper': glyph_mapper,
                    'actor': glyph_actor
                }
                
            elif viz_type.lower() == 'pressure_drop':
                # Create visualization of pressure gradient
                field_name = parameters.get('field_name', 'p')
                
                # This would ideally use a gradient filter, but for now we'll just show pressure
                # Check if field exists
                output = self.active_source.GetOutput()
                point_data = output.GetPointData()
                
                field_exists = False
                for i in range(point_data.GetNumberOfArrays()):
                    if point_data.GetArrayName(i) == field_name:
                        field_exists = True
                        point_data.SetActiveScalars(field_name)
                        break
                
                if not field_exists:
                    logger.warning(f"Field {field_name} not found in data")
                    return False
                
                # Create mapper and actor (keep original source)
                mapper = vtk.vtkDataSetMapper()
                mapper.SetInputConnection(self.active_source.GetOutputPort())
                
                # Color by pressure
                lut = vtk.vtkLookupTable()
                lut.SetHueRange(0.667, 0.0)  # Blue to red
                lut.Build()
                
                field_array = point_data.GetArray(field_name)
                mapper.SetLookupTable(lut)
                mapper.SetScalarRange(field_array.GetRange())
                mapper.ScalarVisibilityOn()
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                
                # Remove existing actors
                if self.active_actor:
                    self.renderer.RemoveActor(self.active_actor)
                
                # Add new actor
                self.renderer.AddActor(actor)
                
                # Update active objects
                self.active_mapper = mapper
                self.active_actor = actor
                
                # Add to pipeline
                self.pipeline['pressure_drop'] = {
                    'source': self.active_source,
                    'mapper': mapper,
                    'actor': actor
                }
                
            elif viz_type.lower() in ['pigging', 'spill']:
                # These require more complex implementations
                logger.warning(f"{viz_type.capitalize()} visualization requires custom setup, not fully implemented")
                return False
                
            else:
                logger.error(f"Unsupported oil & gas visualization type: {viz_type}")
                return False
            
            # Update render window
            if self.render_window is not None:
                self.render_window.Render()
            
            logger.info(f"Applied {viz_type} visualization")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply {viz_type} visualization: {e}")
            return False
    
    def save_state(self, filepath: str) -> bool:
        """
        Save the current VTK state to a file.
        
        This is a simplified implementation that only stores basic state information.
        For a complete state saving, a more complex implementation would be needed.
        
        Args:
            filepath (str): Path to save the state file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("VTK bridge not initialized")
            return False
        
        try:
            # Create a simple state dictionary
            state = {
                "pipeline": list(self.pipeline.keys()),
                "camera": {
                    "position": self.renderer.GetActiveCamera().GetPosition(),
                    "focal_point": self.renderer.GetActiveCamera().GetFocalPoint(),
                    "view_up": self.renderer.GetActiveCamera().GetViewUp(),
                    "view_angle": self.renderer.GetActiveCamera().GetViewAngle()
                },
                "background": self.renderer.GetBackground()
            }
            
            # Save state to file (as a JSON file)
            import json
            
            # Convert numpy arrays/tuples to lists
            def convert_to_serializable(obj):
                if isinstance(obj, (list, tuple, np.ndarray)):
                    return list(obj)
                return obj
            
            # Use a simple dictionary comprehension to convert all values
            state = {k: convert_to_serializable(v) if isinstance(v, dict) else v 
                    for k, v in state.items()}
            
            if isinstance(state["camera"], dict):
                state["camera"] = {k: convert_to_serializable(v) for k, v in state["camera"].items()}
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            # Store in states dictionary
            state_name = os.path.basename(filepath)
            self.states[state_name] = filepath
            
            logger.info(f"Saved simplified state to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save state to {filepath}: {e}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """
        Load a VTK state from a file.
        
        This is a simplified implementation that only loads basic state information.
        For a complete state loading, a more complex implementation would be needed.
        
        Args:
            filepath (str): Path to the state file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("VTK bridge not initialized")
            return False
        
        try:
            # Load state from file
            import json
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Set camera parameters
            if "camera" in state:
                camera = self.renderer.GetActiveCamera()
                camera.SetPosition(state["camera"]["position"])
                camera.SetFocalPoint(state["camera"]["focal_point"])
                camera.SetViewUp(state["camera"]["view_up"])
                camera.SetViewAngle(state["camera"]["view_angle"])
            
            # Set background color
            if "background" in state:
                self.renderer.SetBackground(state["background"])
            
            # Update render window
            if self.render_window is not None:
                self.render_window.Render()
            
            logger.info(f"Loaded simplified state from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state from {filepath}: {e}")
            return False
    
    def export_image(self, filepath: str, width: int = 1200, height: int = 800) -> bool:
        """
        Export the current view to an image file.
        
        Args:
            filepath (str): Path to save the image
            width (int): Image width in pixels
            height (int): Image height in pixels
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized or self.render_window is None:
            logger.error("VTK bridge not initialized or no render window")
            return False
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Save screenshot
            window_to_image = vtk.vtkWindowToImageFilter()
            window_to_image.SetInput(self.render_window)
            window_to_image.SetScale(1)
            window_to_image.SetInputBufferTypeToRGB()
            window_to_image.ReadFrontBufferOff()
            window_to_image.Update()
            
            # Determine writer based on file extension
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
            
            writer.SetFileName(filepath)
            writer.SetInputConnection(window_to_image.GetOutputPort())
            writer.Write()
            
            logger.info(f"Exported image to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export image to {filepath}: {e}")
            return False
    
    def cleanup(self):
        """Clean up VTK resources."""
        if not self.initialized:
            return
        
        try:
            # Remove actors
            if self.renderer:
                self.renderer.RemoveAllViewProps()
            
            # Clear references
            self.pipeline = {}
            self.active_source = None
            self.active_mapper = None
            self.active_actor = None
            
            # Clear states
            self.states = {}
            
            # Turn off any widgets
            if hasattr(self, 'axes_widget') and self.axes_widget is not None:
                self.axes_widget.Off()
                self.axes_widget = None
            
            # Release interactor
            if self.interactor:
                self.interactor.TerminateApp()
                self.interactor = None
            
            # Mark as uninitialized
            self.initialized = False
            
            logger.info("VTK bridge resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up VTK bridge: {e}")


# Create singleton instance
_bridge_instance = None

def get_bridge(render_window=None) -> VTKBridge:
    """
    Get the singleton VTKBridge instance.
    
    Args:
        render_window: An existing render window to use (optional)
        
    Returns:
        VTKBridge: The bridge instance
    """
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = VTKBridge(render_window)
    elif render_window is not None and _bridge_instance.render_window != render_window:
        # Update render window if it changed
        _bridge_instance.render_window = render_window
        if _bridge_instance.initialized and _bridge_instance.renderer is not None:
            # Reconnect renderer to new render window
            _bridge_instance.render_window.AddRenderer(_bridge_instance.renderer)
            if _bridge_instance.render_window is not None:
                _bridge_instance.render_window.Render()
    
    return _bridge_instance

def cleanup_bridge():
    """Clean up the VTKBridge singleton instance."""
    global _bridge_instance
    if _bridge_instance is not None:
        _bridge_instance.cleanup()
        _bridge_instance = None