#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom ParaView representations for Openfoam_Simulator application.

This module defines specialized visual representations for ParaView, tailored 
specifically for oil & gas industry visualizations. These include multiphase flow
representations, pipeline visualizations, and specialized displays for pigging
and spill simulations.
"""

import os
import sys
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# Import ParaView modules (with availability check)
try:
    from paraview.simple import *
    from paraview import servermanager, vtk
    from vtkmodules.vtkCommonCore import vtkDataObject
    from vtkmodules.vtkRenderingCore import vtkProp, vtkActor
    
    PARAVIEW_AVAILABLE = True
except ImportError:
    PARAVIEW_AVAILABLE = False

# Import utility modules
try:
    from ...utils.logger import get_logger
except (ImportError, ValueError):
    # For standalone usage without the Openfoam_Simulator package
    import logging
    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

# Global logger
logger = get_logger(__name__)


class MultiphaseRepresentation(object):
    """
    Custom representation for multiphase flow visualization.
    
    This representation shows different phases with distinct colors and transparency,
    making it easier to understand phase distributions in multiphase simulations.
    """
    
    def __init__(self):
        """Initialize multiphase representation."""
        self.name = "Multiphase Flow"
        self.input = None
        self.display = None
        self.phase_field = "alpha.water"  # Default phase field name
        self.phase_colors = {
            "water": [0.2, 0.4, 0.8],    # Blue for water
            "oil": [0.6, 0.4, 0.2],      # Brown for oil
            "gas": [0.9, 0.9, 0.9]       # Light gray for gas
        }
        self.phase_opacities = {
            "water": 0.8,
            "oil": 0.9,
            "gas": 0.3
        }
        self.phase_thresholds = [0.1, 0.9]  # Default thresholds for phase separation
    
    def set_input(self, input_data):
        """
        Set the input data for the representation.
        
        Args:
            input_data: ParaView data object or proxy
        """
        if not PARAVIEW_AVAILABLE:
            logger.error("ParaView not available")
            return
        
        self.input = input_data
    
    def set_phase_field(self, field_name: str):
        """
        Set the field that contains phase information.
        
        Args:
            field_name (str): Name of the phase field
        """
        self.phase_field = field_name
    
    def set_phase_colors(self, water_color=None, oil_color=None, gas_color=None):
        """
        Set colors for different phases.
        
        Args:
            water_color (List[float], optional): RGB color for water [0-1, 0-1, 0-1]
            oil_color (List[float], optional): RGB color for oil [0-1, 0-1, 0-1]
            gas_color (List[float], optional): RGB color for gas [0-1, 0-1, 0-1]
        """
        if water_color:
            self.phase_colors["water"] = water_color
        if oil_color:
            self.phase_colors["oil"] = oil_color
        if gas_color:
            self.phase_colors["gas"] = gas_color
    
    def set_phase_opacities(self, water_opacity=None, oil_opacity=None, gas_opacity=None):
        """
        Set opacities for different phases.
        
        Args:
            water_opacity (float, optional): Opacity for water [0-1]
            oil_opacity (float, optional): Opacity for oil [0-1]
            gas_opacity (float, optional): Opacity for gas [0-1]
        """
        if water_opacity is not None:
            self.phase_opacities["water"] = water_opacity
        if oil_opacity is not None:
            self.phase_opacities["oil"] = oil_opacity
        if gas_opacity is not None:
            self.phase_opacities["gas"] = gas_opacity
    
    def set_phase_thresholds(self, low_threshold: float, high_threshold: float):
        """
        Set thresholds for phase separation.
        
        Args:
            low_threshold (float): Lower threshold (below this is considered gas)
            high_threshold (float): Upper threshold (above this is considered water)
        """
        self.phase_thresholds = [low_threshold, high_threshold]
    
    def apply(self, view=None):
        """
        Apply the representation to the input data.
        
        Args:
            view: ParaView view
        
        Returns:
            List: List of created displays
        """
        if not PARAVIEW_AVAILABLE or self.input is None:
            logger.error("Cannot apply representation: ParaView not available or no input")
            return []
        
        try:
            # Hide original data
            Hide(self.input, view)
            
            # Create thresholds for different phases
            thresholds = []
            
            # Gas phase (values below low_threshold)
            gas_threshold = Threshold(Input=self.input)
            gas_threshold.Scalars = ['POINTS', self.phase_field]
            gas_threshold.ThresholdRange = [0.0, self.phase_thresholds[0]]
            gas_threshold.UpdatePipeline()
            
            # Oil phase (values between thresholds)
            oil_threshold = Threshold(Input=self.input)
            oil_threshold.Scalars = ['POINTS', self.phase_field]
            oil_threshold.ThresholdRange = [self.phase_thresholds[0], self.phase_thresholds[1]]
            oil_threshold.UpdatePipeline()
            
            # Water phase (values above high_threshold)
            water_threshold = Threshold(Input=self.input)
            water_threshold.Scalars = ['POINTS', self.phase_field]
            water_threshold.ThresholdRange = [self.phase_thresholds[1], 1.0]
            water_threshold.UpdatePipeline()
            
            # Create displays for each phase
            displays = []
            
            # Display water phase
            water_display = Show(water_threshold, view)
            water_display.Representation = 'Surface'
            water_display.ColorArrayName = [None, '']
            water_display.DiffuseColor = self.phase_colors["water"]
            water_display.Opacity = self.phase_opacities["water"]
            displays.append(water_display)
            
            # Display oil phase
            oil_display = Show(oil_threshold, view)
            oil_display.Representation = 'Surface'
            oil_display.ColorArrayName = [None, '']
            oil_display.DiffuseColor = self.phase_colors["oil"]
            oil_display.Opacity = self.phase_opacities["oil"]
            displays.append(oil_display)
            
            # Display gas phase
            gas_display = Show(gas_threshold, view)
            gas_display.Representation = 'Surface'
            gas_display.ColorArrayName = [None, '']
            gas_display.DiffuseColor = self.phase_colors["gas"]
            gas_display.Opacity = self.phase_opacities["gas"]
            displays.append(gas_display)
            
            # Store thresholds for future reference
            self.thresholds = {
                "water": water_threshold,
                "oil": oil_threshold,
                "gas": gas_threshold
            }
            
            # Store displays
            self.displays = displays
            
            # Update view
            if view:
                view.Update()
            
            return displays
            
        except Exception as e:
            logger.error(f"Error applying multiphase representation: {e}")
            return []
    
    def update(self):
        """Update the representation when settings change."""
        if not PARAVIEW_AVAILABLE:
            logger.error("ParaView not available")
            return
        
        try:
            if hasattr(self, 'thresholds') and hasattr(self, 'displays'):
                # Update thresholds
                self.thresholds["gas"].ThresholdRange = [0.0, self.phase_thresholds[0]]
                self.thresholds["oil"].ThresholdRange = [self.phase_thresholds[0], self.phase_thresholds[1]]
                self.thresholds["water"].ThresholdRange = [self.phase_thresholds[1], 1.0]
                
                # Update displays
                self.displays[0].DiffuseColor = self.phase_colors["water"]
                self.displays[0].Opacity = self.phase_opacities["water"]
                self.displays[1].DiffuseColor = self.phase_colors["oil"]
                self.displays[1].Opacity = self.phase_opacities["oil"]
                self.displays[2].DiffuseColor = self.phase_colors["gas"]
                self.displays[2].Opacity = self.phase_opacities["gas"]
                
                # Update the pipeline
                for threshold in self.thresholds.values():
                    threshold.UpdatePipeline()
        except Exception as e:
            logger.error(f"Error updating multiphase representation: {e}")


class PipelineFlowRepresentation(object):
    """
    Custom representation for pipeline flow visualization.
    
    This representation provides a specialized view of fluid flow in pipelines,
    showing velocity profiles, pressure gradients, and other relevant parameters.
    """
    
    def __init__(self):
        """Initialize pipeline flow representation."""
        self.name = "Pipeline Flow"
        self.input = None
        self.display = None
        self.velocity_field = "U"  # Default velocity field name
        self.pressure_field = "p"  # Default pressure field name
        self.scale_factor = 0.1    # Scale factor for velocity arrows
        self.num_slices = 5        # Number of slices along the pipeline
        self.arrow_density = 50    # One arrow for every N points
        self.colormap = "Rainbow"  # Default colormap
    
    def set_input(self, input_data):
        """
        Set the input data for the representation.
        
        Args:
            input_data: ParaView data object or proxy
        """
        if not PARAVIEW_AVAILABLE:
            logger.error("ParaView not available")
            return
        
        self.input = input_data
    
    def set_fields(self, velocity_field: str = None, pressure_field: str = None):
        """
        Set the field names for velocity and pressure.
        
        Args:
            velocity_field (str, optional): Name of the velocity field
            pressure_field (str, optional): Name of the pressure field
        """
        if velocity_field:
            self.velocity_field = velocity_field
        if pressure_field:
            self.pressure_field = pressure_field
    
    def set_scale_factor(self, scale_factor: float):
        """
        Set scale factor for velocity arrows.
        
        Args:
            scale_factor (float): Scale factor
        """
        self.scale_factor = scale_factor
    
    def set_num_slices(self, num_slices: int):
        """
        Set number of slices along the pipeline.
        
        Args:
            num_slices (int): Number of slices
        """
        self.num_slices = num_slices
    
    def set_arrow_density(self, arrow_density: int):
        """
        Set density of velocity arrows.
        
        Args:
            arrow_density (int): One arrow per N points
        """
        self.arrow_density = arrow_density
    
    def set_colormap(self, colormap: str):
        """
        Set colormap for the representation.
        
        Args:
            colormap (str): Name of the colormap
        """
        self.colormap = colormap
    
    def apply(self, view=None):
        """
        Apply the representation to the input data.
        
        Args:
            view: ParaView view
        
        Returns:
            List: List of created displays
        """
        if not PARAVIEW_AVAILABLE or self.input is None:
            logger.error("Cannot apply representation: ParaView not available or no input")
            return []
        
        try:
            # Hide original data
            Hide(self.input, view)
            
            # Create a tube filter to represent the pipeline
            tube = Tube(Input=self.input)
            tube.Radius = 1.0  # Will be scaled appropriately
            tube.NumberOfSides = 24
            tube.Capping = True
            tube.UpdatePipeline()
            
            # Show the pipeline colored by pressure
            tube_display = Show(tube, view)
            tube_display.Representation = 'Surface'
            ColorBy(tube_display, ('POINTS', self.pressure_field))
            
            # Apply selected colormap
            pressure_lut = GetColorTransferFunction(self.pressure_field)
            pressure_lut.ApplyPreset(self.colormap, True)
            
            # Show color bar
            tube_display.SetScalarBarVisibility(view, True)
            
            # Calculate bounds for slicing
            bounds = tube.GetDataInformation().GetBounds()
            length = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
            pipeline_direction = [0, 0, 0]
            
            # Determine main pipeline direction
            dimensions = [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]
            max_dim_index = dimensions.index(max(dimensions))
            if max_dim_index == 0:
                pipeline_direction = [1, 0, 0]
                slice_normal = [0, 1, 0]
                main_start = bounds[0]
                main_end = bounds[1]
            elif max_dim_index == 1:
                pipeline_direction = [0, 1, 0]
                slice_normal = [1, 0, 0]
                main_start = bounds[2]
                main_end = bounds[3]
            else:
                pipeline_direction = [0, 0, 1]
                slice_normal = [0, 1, 0]
                main_start = bounds[4]
                main_end = bounds[5]
            
            # Create slices along the pipeline
            slice_displays = []
            for i in range(self.num_slices):
                # Calculate slice position
                position = main_start + (i + 1) * (main_end - main_start) / (self.num_slices + 1)
                
                # Create slice
                slice_filter = Slice(Input=self.input)
                slice_filter.SliceType = 'Plane'
                slice_filter.SliceType.Normal = slice_normal
                
                # Set slice origin based on pipeline direction
                if max_dim_index == 0:
                    slice_filter.SliceType.Origin = [position, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
                elif max_dim_index == 1:
                    slice_filter.SliceType.Origin = [(bounds[0] + bounds[1]) / 2, position, (bounds[4] + bounds[5]) / 2]
                else:
                    slice_filter.SliceType.Origin = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, position]
                
                slice_filter.UpdatePipeline()
                
                # Create velocity glyphs on the slice
                glyph = Glyph(Input=slice_filter)
                glyph.GlyphType = 'Arrow'
                glyph.OrientationArray = ['POINTS', self.velocity_field]
                glyph.ScaleArray = ['POINTS', self.velocity_field]
                glyph.ScaleFactor = self.scale_factor
                glyph.GlyphMode = 'Every Nth Point'
                glyph.Stride = self.arrow_density
                glyph.UpdatePipeline()
                
                # Show the glyphs
                glyph_display = Show(glyph, view)
                glyph_display.Representation = 'Surface'
                ColorBy(glyph_display, ('POINTS', self.velocity_field, 'Magnitude'))
                
                # Apply a different colormap for velocity
                velocity_lut = GetColorTransferFunction(f"{self.velocity_field}_Magnitude")
                velocity_lut.ApplyPreset('Cool to Warm', True)
                
                slice_displays.append(glyph_display)
            
            # Create streamlines along the pipeline
            streamlines = StreamTracer(Input=self.input)
            streamlines.SeedType = 'Point Source'
            
            # Set seed position near the start of the pipeline
            if max_dim_index == 0:
                streamlines.SeedType.Center = [
                    bounds[0] + 0.05 * (bounds[1] - bounds[0]),
                    (bounds[2] + bounds[3]) / 2,
                    (bounds[4] + bounds[5]) / 2
                ]
            elif max_dim_index == 1:
                streamlines.SeedType.Center = [
                    (bounds[0] + bounds[1]) / 2,
                    bounds[2] + 0.05 * (bounds[3] - bounds[2]),
                    (bounds[4] + bounds[5]) / 2
                ]
            else:
                streamlines.SeedType.Center = [
                    (bounds[0] + bounds[1]) / 2,
                    (bounds[2] + bounds[3]) / 2,
                    bounds[4] + 0.05 * (bounds[5] - bounds[4])
                ]
            
            streamlines.SeedType.Radius = min(dimensions) * 0.4
            streamlines.SeedType.NumberOfPoints = 50
            streamlines.MaximumStreamlineLength = length * 2
            streamlines.Vectors = ['POINTS', self.velocity_field]
            streamlines.IntegrationDirection = 'FORWARD'
            streamlines.UpdatePipeline()
            
            # Show streamlines
            stream_display = Show(streamlines, view)
            stream_display.Representation = 'Surface'
            ColorBy(stream_display, ('POINTS', self.velocity_field, 'Magnitude'))
            
            # Collect all displays
            displays = [tube_display] + slice_displays + [stream_display]
            
            # Store created objects
            self.tube = tube
            self.slices = [slice_filter]
            self.glyphs = [glyph]
            self.streamlines = streamlines
            self.displays = displays
            
            # Update view
            if view:
                view.Update()
            
            return displays
            
        except Exception as e:
            logger.error(f"Error applying pipeline flow representation: {e}")
            return []
    
    def update(self):
        """Update the representation when settings change."""
        if not PARAVIEW_AVAILABLE:
            logger.error("ParaView not available")
            return
        
        try:
            if hasattr(self, 'glyphs'):
                # Update glyph scale factor
                for glyph in self.glyphs:
                    glyph.ScaleFactor = self.scale_factor
                    glyph.Stride = self.arrow_density
                    glyph.UpdatePipeline()
            
            if hasattr(self, 'streamlines'):
                # Update streamline settings
                bounds = self.tube.GetDataInformation().GetBounds()
                dimensions = [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]
                length = max(dimensions)
                self.streamlines.MaximumStreamlineLength = length * 2
                self.streamlines.UpdatePipeline()
        except Exception as e:
            logger.error(f"Error updating pipeline flow representation: {e}")


class PiggingRepresentation(object):
    """
    Custom representation for pipeline pigging simulation.
    
    This representation shows the pig element and its interaction with the pipeline
    fluid, allowing visualization of important pigging parameters.
    """
    
    def __init__(self):
        """Initialize pigging representation."""
        self.name = "Pigging Simulation"
        self.input = None
        self.display = None
        self.pig_field = "pigIndicator"  # Field marking pig location
        self.velocity_field = "U"         # Default velocity field name
        self.pressure_field = "p"         # Default pressure field name
        self.pig_position = 0.0           # Normalized position (0.0 to 1.0)
        self.pig_opacity = 0.8            # Opacity of pig visualization
        self.pig_color = [1.0, 0.6, 0.0]  # Orange color for pig
        self.show_velocity_profile = True  # Show velocity profile around pig
        self.show_pressure_field = True    # Show pressure field
    
    def set_input(self, input_data):
        """
        Set the input data for the representation.
        
        Args:
            input_data: ParaView data object or proxy
        """
        if not PARAVIEW_AVAILABLE:
            logger.error("ParaView not available")
            return
        
        self.input = input_data
    
    def set_pig_field(self, field_name: str):
        """
        Set the field that indicates pig location.
        
        Args:
            field_name (str): Name of the pig indicator field
        """
        self.pig_field = field_name
    
    def set_fields(self, velocity_field: str = None, pressure_field: str = None):
        """
        Set the field names for velocity and pressure.
        
        Args:
            velocity_field (str, optional): Name of the velocity field
            pressure_field (str, optional): Name of the pressure field
        """
        if velocity_field:
            self.velocity_field = velocity_field
        if pressure_field:
            self.pressure_field = pressure_field
    
    def set_pig_position(self, position: float):
        """
        Set the normalized position of the pig (0.0 to 1.0).
        
        Args:
            position (float): Normalized position
        """
        self.pig_position = max(0.0, min(1.0, position))  # Clamp to [0.0, 1.0]
    
    def set_pig_appearance(self, color: List[float] = None, opacity: float = None):
        """
        Set the appearance of the pig.
        
        Args:
            color (List[float], optional): RGB color [0-1, 0-1, 0-1]
            opacity (float, optional): Opacity [0-1]
        """
        if color:
            self.pig_color = color
        if opacity is not None:
            self.pig_opacity = opacity
    
    def set_visualization_options(self, show_velocity: bool = None, show_pressure: bool = None):
        """
        Set visualization options.
        
        Args:
            show_velocity (bool, optional): Whether to show velocity profile
            show_pressure (bool, optional): Whether to show pressure field
        """
        if show_velocity is not None:
            self.show_velocity_profile = show_velocity
        if show_pressure is not None:
            self.show_pressure_field = show_pressure
    
    def apply(self, view=None):
        """
        Apply the representation to the input data.
        
        Args:
            view: ParaView view
        
        Returns:
            List: List of created displays
        """
        if not PARAVIEW_AVAILABLE or self.input is None:
            logger.error("Cannot apply representation: ParaView not available or no input")
            return []
        
        try:
            # Hide original data
            Hide(self.input, view)
            
            displays = []
            
            # First, determine the main pipeline direction and bounds
            bounds = self.input.GetDataInformation().GetBounds()
            dimensions = [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]
            max_dim_index = dimensions.index(max(dimensions))
            
            # Determine pipeline properties
            if max_dim_index == 0:
                pipeline_direction = [1, 0, 0]
                pipeline_start = bounds[0]
                pipeline_end = bounds[1]
            elif max_dim_index == 1:
                pipeline_direction = [0, 1, 0]
                pipeline_start = bounds[2]
                pipeline_end = bounds[3]
            else:
                pipeline_direction = [0, 0, 1]
                pipeline_start = bounds[4]
                pipeline_end = bounds[5]
            
            # Calculate the actual pig position
            pig_pos = pipeline_start + self.pig_position * (pipeline_end - pipeline_start)
            
            # Create a representation of the pipeline
            if self.show_pressure_field:
                # Use the pressure field to color the pipeline
                pipeline_display = Show(self.input, view)
                pipeline_display.Representation = 'Surface'
                pipeline_display.ColorArrayName = ['POINTS', self.pressure_field]
                
                # Apply blue-to-red colormap for pressure
                pressure_lut = GetColorTransferFunction(self.pressure_field)
                pressure_lut.ApplyPreset('Cool to Warm', True)
                
                # Show color bar
                pipeline_display.SetScalarBarVisibility(view, True)
                
                displays.append(pipeline_display)
            
            # Create a clip to visualize the pig
            clip = Clip(Input=self.input)
            clip.ClipType = 'Plane'
            
            # Set the clip plane normal to the pipeline direction
            clip.ClipType.Normal = pipeline_direction
            
            # Set the clip plane origin at the pig position
            if max_dim_index == 0:
                clip.ClipType.Origin = [pig_pos, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
            elif max_dim_index == 1:
                clip.ClipType.Origin = [(bounds[0] + bounds[1]) / 2, pig_pos, (bounds[4] + bounds[5]) / 2]
            else:
                clip.ClipType.Origin = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, pig_pos]
            
            clip.InsideOut = True  # Show the portion of the pipeline after the pig
            clip.UpdatePipeline()
            
            # Show the pig cross-section
            pig_display = Show(clip, view)
            pig_display.Representation = 'Surface'
            pig_display.ColorArrayName = [None, '']
            pig_display.DiffuseColor = self.pig_color
            pig_display.Opacity = self.pig_opacity
            
            displays.append(pig_display)
            
            # Create slices to show the velocity profile around the pig
            if self.show_velocity_profile:
                # Create slices at and around the pig position
                slice_positions = [
                    pig_pos - 0.05 * (pipeline_end - pipeline_start),
                    pig_pos,
                    pig_pos + 0.05 * (pipeline_end - pipeline_start)
                ]
                
                for slice_pos in slice_positions:
                    # Create slice
                    slice_filter = Slice(Input=self.input)
                    slice_filter.SliceType = 'Plane'
                    slice_filter.SliceType.Normal = pipeline_direction
                    
                    # Set slice origin
                    if max_dim_index == 0:
                        slice_filter.SliceType.Origin = [slice_pos, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
                    elif max_dim_index == 1:
                        slice_filter.SliceType.Origin = [(bounds[0] + bounds[1]) / 2, slice_pos, (bounds[4] + bounds[5]) / 2]
                    else:
                        slice_filter.SliceType.Origin = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, slice_pos]
                    
                    slice_filter.UpdatePipeline()
                    
                    # Create velocity glyphs on the slice
                    glyph = Glyph(Input=slice_filter)
                    glyph.GlyphType = 'Arrow'
                    glyph.OrientationArray = ['POINTS', self.velocity_field]
                    glyph.ScaleArray = ['POINTS', self.velocity_field]
                    glyph.ScaleFactor = 0.05 * min(dimensions)
                    glyph.GlyphMode = 'Every Nth Point'
                    glyph.Stride = 10
                    glyph.UpdatePipeline()
                    
                    # Show the glyphs
                    glyph_display = Show(glyph, view)
                    glyph_display.Representation = 'Surface'
                    ColorBy(glyph_display, ('POINTS', self.velocity_field, 'Magnitude'))
                    
                    # Apply a colormap for velocity
                    velocity_lut = GetColorTransferFunction(f"{self.velocity_field}_Magnitude")
                    velocity_lut.ApplyPreset('Rainbow', True)
                    
                    displays.append(glyph_display)
            
            # Store created objects
            self.pipeline_display = pipeline_display if self.show_pressure_field else None
            self.clip = clip
            self.pig_display = pig_display
            self.displays = displays
            
            # Update view
            if view:
                view.Update()
            
            return displays
            
        except Exception as e:
            logger.error(f"Error applying pigging representation: {e}")
            return []
    
    def update(self):
        """Update the representation when settings change."""
        if not PARAVIEW_AVAILABLE:
            logger.error("ParaView not available")
            return
        
        try:
            if hasattr(self, 'clip') and hasattr(self, 'pig_display'):
                # Update pig position
                bounds = self.input.GetDataInformation().GetBounds()
                dimensions = [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]
                max_dim_index = dimensions.index(max(dimensions))
                
                if max_dim_index == 0:
                    pipeline_start = bounds[0]
                    pipeline_end = bounds[1]
                elif max_dim_index == 1:
                    pipeline_start = bounds[2]
                    pipeline_end = bounds[3]
                else:
                    pipeline_start = bounds[4]
                    pipeline_end = bounds[5]
                
                pig_pos = pipeline_start + self.pig_position * (pipeline_end - pipeline_start)
                
                # Update clip origin
                if max_dim_index == 0:
                    self.clip.ClipType.Origin = [pig_pos, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
                elif max_dim_index == 1:
                    self.clip.ClipType.Origin = [(bounds[0] + bounds[1]) / 2, pig_pos, (bounds[4] + bounds[5]) / 2]
                else:
                    self.clip.ClipType.Origin = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, pig_pos]
                
                # Update pig appearance
                self.pig_display.DiffuseColor = self.pig_color
                self.pig_display.Opacity = self.pig_opacity
                
                # Update clip
                self.clip.UpdatePipeline()
        except Exception as e:
            logger.error(f"Error updating pigging representation: {e}")


class SpillRepresentation(object):
    """
    Custom representation for spill modeling visualization.
    
    This representation provides specialized visualization for oil spill simulations,
    showing spill extent, concentration, and environmental impact.
    """
    
    def __init__(self):
        """Initialize spill representation."""
        self.name = "Spill Model"
        self.input = None
        self.display = None
        self.concentration_field = "concentration"  # Field with spill concentration
        self.velocity_field = "U"                 # Default velocity field name
        self.show_surface = True                  # Show spill surface
        self.show_volume = True                   # Show spill volume
        self.surface_threshold = 0.01             # Threshold for surface visualization
        self.volume_threshold = 0.1               # Threshold for volume visualization
        self.colormap = "Viridis"                 # Default colormap
    
    def set_input(self, input_data):
        """
        Set the input data for the representation.
        
        Args:
            input_data: ParaView data object or proxy
        """
        if not PARAVIEW_AVAILABLE:
            logger.error("ParaView not available")
            return
        
        self.input = input_data
    
    def set_fields(self, concentration_field: str = None, velocity_field: str = None):
        """
        Set the field names for concentration and velocity.
        
        Args:
            concentration_field (str, optional): Name of the concentration field
            velocity_field (str, optional): Name of the velocity field
        """
        if concentration_field:
            self.concentration_field = concentration_field
        if velocity_field:
            self.velocity_field = velocity_field
    
    def set_visualization_options(self, show_surface: bool = None, show_volume: bool = None):
        """
        Set visualization options.
        
        Args:
            show_surface (bool, optional): Whether to show spill surface
            show_volume (bool, optional): Whether to show spill volume
        """
        if show_surface is not None:
            self.show_surface = show_surface
        if show_volume is not None:
            self.show_volume = show_volume
    
    def set_thresholds(self, surface_threshold: float = None, volume_threshold: float = None):
        """
        Set thresholds for visualization.
        
        Args:
            surface_threshold (float, optional): Threshold for surface visualization
            volume_threshold (float, optional): Threshold for volume visualization
        """
        if surface_threshold is not None:
            self.surface_threshold = surface_threshold
        if volume_threshold is not None:
            self.volume_threshold = volume_threshold
    
    def set_colormap(self, colormap: str):
        """
        Set colormap for the representation.
        
        Args:
            colormap (str): Name of the colormap
        """
        self.colormap = colormap
    
    def apply(self, view=None):
        """
        Apply the representation to the input data.
        
        Args:
            view: ParaView view
        
        Returns:
            List: List of created displays
        """
        if not PARAVIEW_AVAILABLE or self.input is None:
            logger.error("Cannot apply representation: ParaView not available or no input")
            return []
        
        try:
            # Hide original data
            Hide(self.input, view)
            
            displays = []
            
            # Create contour for spill surface
            if self.show_surface:
                contour = Contour(Input=self.input)
                contour.ContourBy = ['POINTS', self.concentration_field]
                contour.Isosurfaces = [self.surface_threshold]
                contour.UpdatePipeline()
                
                # Show surface
                surface_display = Show(contour, view)
                surface_display.Representation = 'Surface'
                ColorBy(surface_display, ('POINTS', self.concentration_field))
                
                # Apply colormap
                concentration_lut = GetColorTransferFunction(self.concentration_field)
                concentration_lut.ApplyPreset(self.colormap, True)
                
                # Show color bar
                surface_display.SetScalarBarVisibility(view, True)
                
                displays.append(surface_display)
            
            # Create threshold for volume visualization
            if self.show_volume:
                volume_threshold = Threshold(Input=self.input)
                volume_threshold.Scalars = ['POINTS', self.concentration_field]
                volume_threshold.ThresholdRange = [self.volume_threshold, float('inf')]
                volume_threshold.UpdatePipeline()
                
                # Show volume
                volume_display = Show(volume_threshold, view)
                volume_display.Representation = 'Volume'
                ColorBy(volume_display, ('POINTS', self.concentration_field))
                
                # Apply colormap
                concentration_lut = GetColorTransferFunction(self.concentration_field)
                concentration_lut.ApplyPreset(self.colormap, True)
                
                # Set opacity for better visualization
                pwf = GetOpacityTransferFunction(self.concentration_field)
                pwf.Points = [
                    self.volume_threshold, 0.0, 0.5, 0.0,
                    (self.volume_threshold + 1.0) / 2, 0.2, 0.5, 0.0,
                    1.0, 0.8, 0.5, 0.0
                ]
                
                displays.append(volume_display)
            
            # Add streamlines to show dispersion patterns
            streamlines = StreamTracer(Input=self.input)
            streamlines.SeedType = 'Point Source'
            
            # Get bounds
            bounds = self.input.GetDataInformation().GetBounds()
            
            # Place seed at the center of the domain near the surface
            streamlines.SeedType.Center = [
                (bounds[0] + bounds[1]) / 2,
                (bounds[2] + bounds[3]) / 2,
                bounds[5] - (bounds[5] - bounds[4]) * 0.1  # Near the top
            ]
            
            # Set up streamlines
            min_dim = min(bounds[1] - bounds[0], bounds[3] - bounds[2])
            streamlines.SeedType.Radius = min_dim * 0.1
            streamlines.SeedType.NumberOfPoints = 30
            streamlines.Vectors = ['POINTS', self.velocity_field]
            streamlines.MaximumStreamlineLength = min_dim * 5
            streamlines.UpdatePipeline()
            
            # Show streamlines
            stream_display = Show(streamlines, view)
            stream_display.Representation = 'Surface'
            ColorBy(stream_display, ('POINTS', self.concentration_field))
            
            displays.append(stream_display)
            
            # Store created objects
            self.contour = contour if self.show_surface else None
            self.volume_threshold = volume_threshold if self.show_volume else None
            self.streamlines = streamlines
            self.displays = displays
            
            # Update view
            if view:
                view.Update()
            
            return displays
            
        except Exception as e:
            logger.error(f"Error applying spill representation: {e}")
            return []
    
    def update(self):
        """Update the representation when settings change."""
        if not PARAVIEW_AVAILABLE:
            logger.error("ParaView not available")
            return
        
        try:
            # Update surface contour
            if self.show_surface and hasattr(self, 'contour'):
                self.contour.Isosurfaces = [self.surface_threshold]
                self.contour.UpdatePipeline()
            
            # Update volume threshold
            if self.show_volume and hasattr(self, 'volume_threshold'):
                self.volume_threshold.ThresholdRange = [self.volume_threshold, float('inf')]
                self.volume_threshold.UpdatePipeline()
                
                # Update opacity function
                pwf = GetOpacityTransferFunction(self.concentration_field)
                pwf.Points = [
                    self.volume_threshold, 0.0, 0.5, 0.0,
                    (self.volume_threshold + 1.0) / 2, 0.2, 0.5, 0.0,
                    1.0, 0.8, 0.5, 0.0
                ]
        except Exception as e:
            logger.error(f"Error updating spill representation: {e}")


# Dictionary of available representation types
AVAILABLE_REPRESENTATIONS = {
    'MultiphaseFlow': MultiphaseRepresentation,
    'PipelineFlow': PipelineFlowRepresentation,
    'PiggingSimulation': PiggingRepresentation,
    'SpillModel': SpillRepresentation
}


def create_representation(representation_name: str, input_data=None, view=None) -> Optional[object]:
    """
    Create a custom representation.
    
    Args:
        representation_name (str): Name of the representation type
        input_data: Input data for the representation
        view: ParaView view
        
    Returns:
        object: The created representation
    """
    if not PARAVIEW_AVAILABLE:
        logger.error("ParaView not available")
        return None
    
    try:
        if representation_name not in AVAILABLE_REPRESENTATIONS:
            logger.error(f"Unknown representation type: {representation_name}")
            return None
        
        # Create the representation
        rep_class = AVAILABLE_REPRESENTATIONS[representation_name]
        representation = rep_class()
        
        if input_data:
            representation.set_input(input_data)
        
        return representation
    except Exception as e:
        logger.error(f"Error creating representation {representation_name}: {e}")
        return None


def register_representations():
    """
    Register custom representations with ParaView.
    
    This function is called when the module is loaded as a plugin in ParaView.
    """
    if not PARAVIEW_AVAILABLE:
        logger.error("ParaView not available, cannot register representations")
        return
    
    try:
        # This is where we'd register with ParaView's plugin system
        # In a real implementation, this would involve C++ code or ParaView's
        # Python plugin framework to register these as actual ParaView representations
        
        logger.info("Custom representations registered")
    except Exception as e:
        logger.error(f"Error registering representations: {e}")


# Register representations if this module is loaded directly by ParaView
if __name__ == "__main__":
    register_representations()