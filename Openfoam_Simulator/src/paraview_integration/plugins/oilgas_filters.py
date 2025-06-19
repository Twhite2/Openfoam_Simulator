#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom ParaView/VTK filters for oil & gas applications.

This module provides specialized filters for visualizing common oil & gas
simulation results in ParaView. These include:
- Multi-phase flow visualization
- Pipeline flow visualization
- Pigging visualization
- Wellbore visualization
- Spill spread visualization
"""

import os
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

# Import VTK modules
import vtk
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonCore import vtkDataObject
from vtkmodules.vtkCommonDataModel import (
    vtkDataSet, vtkMultiBlockDataSet, vtkPolyData, vtkUnstructuredGrid
)
from vtkmodules.vtkFiltersCore import (
    vtkContourFilter, vtkThreshold, vtkStreamTracer, vtkCutter
)
from vtkmodules.vtkFiltersGeneral import vtkTransformFilter, vtkWarpScalar

# Import ParaView modules
try:
    from paraview.util.vtkAlgorithm import (
        VTKPythonAlgorithmBase, smdomain, smhint, smproperty, smproxy
    )
    from paraview import vtk as para_vtk
    PARAVIEW_AVAILABLE = True
except ImportError:
    # Define mock classes for when ParaView is not available
    class VTKPythonAlgorithmBase:
        pass
    
    def smdomain(*args, **kwargs):
        def decorator(obj):
            return obj
        return decorator
    
    def smhint(*args, **kwargs):
        def decorator(obj):
            return obj
        return decorator
    
    def smproperty(*args, **kwargs):
        def decorator(obj):
            return obj
        return decorator
    
    def smproxy(*args, **kwargs):
        def decorator(obj):
            return obj
        return decorator
    
    PARAVIEW_AVAILABLE = False

# Setup logger
logger = logging.getLogger(__name__)

# Constants
PHASE_THRESHOLD = 0.5  # Default threshold for phase interface detection
DEFAULT_SCALE = 1.0    # Default scale for various filters


@smproxy.filter(name="OilGasPhaseInterfaceFilter", label="Oil & Gas Phase Interface")
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=True)
class PhaseInterfaceFilter(VTKPythonAlgorithmBase):
    """
    Filter to extract and visualize phase interfaces in multi-phase flows.
    
    This filter creates surfaces at the interface between different phases in a
    multi-phase flow simulation, using isocontours of phase fraction values.
    """
    
    @smproperty.stringvector(name="PhaseField", default_values="alpha.water")
    @smdomain.datatype(dataTypes=["vtkDataArray"])
    def SetPhaseField(self, phase_field):
        """
        Set the phase field name.
        
        Args:
            phase_field (str): Name of the phase fraction field
        """
        self._phase_field = phase_field
        self.Modified()
    
    @smproperty.doublevector(name="Threshold", default_values=[0.5])
    @smdomain.doublerange(min=0.0, max=1.0)
    def SetThreshold(self, threshold):
        """
        Set the threshold value for interface extraction.
        
        Args:
            threshold (float): Value between 0 and 1 for iso-surface extraction
        """
        self._threshold = threshold
        self.Modified()
    
    @smproperty.doublevector(name="Smoothing", default_values=[0.0])
    @smdomain.doublerange(min=0.0, max=1.0)
    def SetSmoothing(self, smoothing):
        """
        Set smoothing factor for interface surface.
        
        Args:
            smoothing (float): Smoothing factor (0 = no smoothing, 1 = max smoothing)
        """
        self._smoothing = smoothing
        self.Modified()
    
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, 
            nInputPorts=1, 
            nOutputPorts=1,
            outputType="vtkPolyData"
        )
        self._phase_field = "alpha.water"
        self._threshold = 0.5
        self._smoothing = 0.0
    
    def RequestData(self, request, inInfo, outInfo):
        """
        Execute the filter.
        
        Args:
            request: Pipeline request
            inInfo: Input information
            outInfo: Output information
            
        Returns:
            int: 1 for success, 0 for failure
        """
        # Get input data
        input_data = vtkDataSet.GetData(inInfo[0], 0)
        if not input_data:
            logger.error("No input dataset provided")
            return 0
        
        # Check if phase field exists
        point_data = input_data.GetPointData()
        if not point_data.HasArray(self._phase_field):
            logger.error(f"Phase field '{self._phase_field}' not found in dataset")
            return 0
        
        # Create contour filter for phase interface
        contour = vtkContourFilter()
        contour.SetInputData(input_data)
        contour.SetInputArrayToProcess(
            0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, self._phase_field
        )
        contour.SetValue(0, self._threshold)
        contour.Update()
        
        # Apply smoothing if needed
        if self._smoothing > 0.0:
            smooth = vtk.vtkSmoothPolyDataFilter()
            smooth.SetInputData(contour.GetOutput())
            smooth.SetNumberOfIterations(int(self._smoothing * 100))
            smooth.SetRelaxationFactor(self._smoothing)
            smooth.Update()
            output_data = smooth.GetOutput()
        else:
            output_data = contour.GetOutput()
        
        # Set output
        output = vtkPolyData.GetData(outInfo)
        output.ShallowCopy(output_data)
        
        return 1
    
    def FillInputPortInformation(self, port, info):
        info.Set(vtk.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet")
        return 1


@smproxy.filter(name="OilGasMultiPhaseVolumeFilter", label="Oil & Gas Multi-Phase Volume")
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=True)
class MultiPhaseVolumeFilter(VTKPythonAlgorithmBase):
    """
    Filter to visualize multiple phases in a volume with opacity mapping.
    
    This filter separates different phases in a multi-phase flow simulation
    and assigns appropriate colors and opacities for volume rendering.
    """
    
    @smproperty.stringvector(name="PhaseField", default_values="alpha.water")
    @smdomain.datatype(dataTypes=["vtkDataArray"])
    def SetPhaseField(self, phase_field):
        """
        Set the phase field name.
        
        Args:
            phase_field (str): Name of the phase fraction field
        """
        self._phase_field = phase_field
        self.Modified()
    
    @smproperty.doublevector(name="OilPhaseOpacity", default_values=[0.8])
    @smdomain.doublerange(min=0.0, max=1.0)
    def SetOilPhaseOpacity(self, opacity):
        """
        Set the opacity for the oil phase.
        
        Args:
            opacity (float): Opacity value between 0 and 1
        """
        self._oil_opacity = opacity
        self.Modified()
    
    @smproperty.doublevector(name="WaterPhaseOpacity", default_values=[0.6])
    @smdomain.doublerange(min=0.0, max=1.0)
    def SetWaterPhaseOpacity(self, opacity):
        """
        Set the opacity for the water phase.
        
        Args:
            opacity (float): Opacity value between 0 and 1
        """
        self._water_opacity = opacity
        self.Modified()
    
    @smproperty.doublevector(name="GasPhaseOpacity", default_values=[0.3])
    @smdomain.doublerange(min=0.0, max=1.0)
    def SetGasPhaseOpacity(self, opacity):
        """
        Set the opacity for the gas phase.
        
        Args:
            opacity (float): Opacity value between 0 and 1
        """
        self._gas_opacity = opacity
        self.Modified()
    
    @smproperty.stringvector(name="OilPhaseName", default_values="alpha.oil")
    def SetOilPhaseName(self, name):
        """
        Set the oil phase field name.
        
        Args:
            name (str): Name of the oil phase field
        """
        self._oil_phase_name = name
        self.Modified()
    
    @smproperty.stringvector(name="WaterPhaseName", default_values="alpha.water")
    def SetWaterPhaseName(self, name):
        """
        Set the water phase field name.
        
        Args:
            name (str): Name of the water phase field
        """
        self._water_phase_name = name
        self.Modified()
    
    @smproperty.stringvector(name="GasPhaseName", default_values="alpha.gas")
    def SetGasPhaseName(self, name):
        """
        Set the gas phase field name.
        
        Args:
            name (str): Name of the gas phase field
        """
        self._gas_phase_name = name
        self.Modified()
    
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, 
            nInputPorts=1, 
            nOutputPorts=1,
            outputType="vtkUnstructuredGrid"
        )
        self._phase_field = "alpha.water"
        self._oil_opacity = 0.8
        self._water_opacity = 0.6
        self._gas_opacity = 0.3
        self._oil_phase_name = "alpha.oil"
        self._water_phase_name = "alpha.water"
        self._gas_phase_name = "alpha.gas"
    
    def RequestData(self, request, inInfo, outInfo):
        """
        Execute the filter.
        
        Args:
            request: Pipeline request
            inInfo: Input information
            outInfo: Output information
            
        Returns:
            int: 1 for success, 0 for failure
        """
        # Get input data
        input_data = vtkDataSet.GetData(inInfo[0], 0)
        if not input_data:
            logger.error("No input dataset provided")
            return 0
        
        # Create a new opacity array
        num_points = input_data.GetNumberOfPoints()
        opacity_array = vtk.vtkFloatArray()
        opacity_array.SetName("MultiPhaseOpacity")
        opacity_array.SetNumberOfComponents(1)
        opacity_array.SetNumberOfTuples(num_points)
        
        # Get phase arrays
        point_data = input_data.GetPointData()
        has_oil = point_data.HasArray(self._oil_phase_name)
        has_water = point_data.HasArray(self._water_phase_name)
        has_gas = point_data.HasArray(self._gas_phase_name)
        
        oil_array = point_data.GetArray(self._oil_phase_name) if has_oil else None
        water_array = point_data.GetArray(self._water_phase_name) if has_water else None
        gas_array = point_data.GetArray(self._gas_phase_name) if has_gas else None
        
        # Create phase indicator array
        phase_array = vtk.vtkUnsignedCharArray()
        phase_array.SetName("DominantPhase")
        phase_array.SetNumberOfComponents(1)
        phase_array.SetNumberOfTuples(num_points)
        
        # Calculate opacity and dominant phase for each point
        for i in range(num_points):
            oil_val = oil_array.GetValue(i) if has_oil else 0.0
            water_val = water_array.GetValue(i) if has_water else 0.0
            gas_val = gas_array.GetValue(i) if has_gas else 0.0
            
            # Determine dominant phase (1=oil, 2=water, 3=gas)
            if oil_val >= water_val and oil_val >= gas_val:
                phase = 1
                opacity = self._oil_opacity * oil_val
            elif water_val >= oil_val and water_val >= gas_val:
                phase = 2
                opacity = self._water_opacity * water_val
            else:
                phase = 3
                opacity = self._gas_opacity * gas_val
            
            phase_array.SetValue(i, phase)
            opacity_array.SetValue(i, opacity)
        
        # Add arrays to output
        output = vtkUnstructuredGrid.GetData(outInfo)
        output.ShallowCopy(input_data)
        output.GetPointData().AddArray(opacity_array)
        output.GetPointData().AddArray(phase_array)
        
        return 1
    
    def FillInputPortInformation(self, port, info):
        info.Set(vtk.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet")
        return 1


@smproxy.filter(name="OilGasPiggingFilter", label="Oil & Gas Pigging Visualization")
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=True)
class PiggingFilter(VTKPythonAlgorithmBase):
    """
    Filter to visualize pipeline pigging operations.
    
    This filter enhances the visualization of pipeline pigging simulations
    by creating a visual representation of the pig and its effects on the flow.
    """
    
    @smproperty.doublevector(name="PigPosition", default_values=[0.0])
    @smdomain.doublerange(min=0.0, max=1.0)
    def SetPigPosition(self, position):
        """
        Set the normalized position of the pig along the pipeline.
        
        Args:
            position (float): Position from 0 (start) to 1 (end)
        """
        self._pig_position = position
        self.Modified()
    
    @smproperty.doublevector(name="PigDiameter", default_values=[1.0])
    @smdomain.doublerange(min=0.5, max=1.5)
    def SetPigDiameter(self, diameter):
        """
        Set the diameter ratio of the pig relative to the pipeline.
        
        Args:
            diameter (float): Diameter ratio (1.0 = same as pipeline)
        """
        self._pig_diameter = diameter
        self.Modified()
    
    @smproperty.doublevector(name="PigLength", default_values=[1.0])
    @smdomain.doublerange(min=0.1, max=10.0)
    def SetPigLength(self, length):
        """
        Set the length of the pig relative to pipeline diameter.
        
        Args:
            length (float): Length relative to pipeline diameter
        """
        self._pig_length = length
        self.Modified()
    
    @smproperty.intvector(name="PigType", default_values=[0])
    @smdomain.intrange(min=0, max=5)
    def SetPigType(self, pig_type):
        """
        Set the type of pig.
        
        Args:
            pig_type (int): 0=Foam, 1=Disc, 2=Cup, 3=Sphere, 4=Intelligent, 5=Gel
        """
        self._pig_type = pig_type
        self.Modified()
    
    @smproperty.stringvector(name="FlowAxis", default_values="Z")
    @smdomain.stringlistdomain(strings=["X", "Y", "Z"])
    def SetFlowAxis(self, axis):
        """
        Set the main flow axis.
        
        Args:
            axis (str): Main flow direction (X, Y, or Z)
        """
        self._flow_axis = axis
        self.Modified()
    
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, 
            nInputPorts=1, 
            nOutputPorts=1,
            outputType="vtkMultiBlockDataSet"
        )
        self._pig_position = 0.0
        self._pig_diameter = 1.0
        self._pig_length = 1.0
        self._pig_type = 0
        self._flow_axis = "Z"
    
    def RequestData(self, request, inInfo, outInfo):
        """
        Execute the filter.
        
        Args:
            request: Pipeline request
            inInfo: Input information
            outInfo: Output information
            
        Returns:
            int: 1 for success, 0 for failure
        """
        # Get input data
        input_data = vtkDataSet.GetData(inInfo[0], 0)
        if not input_data:
            logger.error("No input dataset provided")
            return 0
        
        # Get pipeline bounds
        bounds = input_data.GetBounds()
        
        # Determine flow axis index
        axis_map = {"X": 0, "Y": 2, "Z": 4}
        axis_idx = axis_map.get(self._flow_axis, 4)  # Default to Z
        
        # Calculate pipeline dimensions
        pipeline_start = bounds[axis_idx]
        pipeline_end = bounds[axis_idx + 1]
        pipeline_length = pipeline_end - pipeline_start
        
        # Calculate pipeline diameter from non-flow axes
        if self._flow_axis == "X":
            diameter = max(bounds[3] - bounds[2], bounds[5] - bounds[4])
        elif self._flow_axis == "Y":
            diameter = max(bounds[1] - bounds[0], bounds[5] - bounds[4])
        else:  # Z
            diameter = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
        
        # Calculate pig position
        pig_center = pipeline_start + self._pig_position * pipeline_length
        
        # Create pig representation
        pig = self._create_pig(pig_center, diameter, axis_idx // 2)
        
        # Create output multi-block dataset
        output = vtkMultiBlockDataSet.GetData(outInfo)
        output.SetNumberOfBlocks(2)
        
        # Set blocks
        output.SetBlock(0, input_data)  # Original pipeline data
        output.SetBlock(1, pig)         # Pig representation
        
        # Set block names
        output.GetMetaData(0).Set(vtk.vtkCompositeDataSet.NAME(), "Pipeline")
        output.GetMetaData(1).Set(vtk.vtkCompositeDataSet.NAME(), "Pig")
        
        return 1
    
    def _create_pig(self, center_pos, pipeline_diameter, axis):
        """
        Create a visual representation of the pig.
        
        Args:
            center_pos (float): Position along the flow axis
            pipeline_diameter (float): Pipeline diameter
            axis (int): Flow axis index (0=X, 1=Y, 2=Z)
            
        Returns:
            vtkPolyData: Pig representation
        """
        # Calculate pig dimensions
        pig_diameter = pipeline_diameter * self._pig_diameter
        pig_length = pipeline_diameter * self._pig_length
        pig_radius = pig_diameter / 2.0
        
        # Create pig based on type
        if self._pig_type == 3:  # Sphere
            # For sphere pig, use sphere source
            pig_source = vtk.vtkSphereSource()
            pig_source.SetRadius(pig_radius)
            pig_source.SetCenter(0, 0, 0)
            pig_source.SetPhiResolution(20)
            pig_source.SetThetaResolution(20)
        else:
            # For other pigs, use cylinder source
            pig_source = vtk.vtkCylinderSource()
            pig_source.SetRadius(pig_radius)
            pig_source.SetHeight(pig_length)
            pig_source.SetResolution(20)
            pig_source.CappingOn()
        
        pig_source.Update()
        pig = pig_source.GetOutput()
        
        # Transform to proper position and orientation
        transform = vtk.vtkTransform()
        transform.Identity()
        
        # Orient based on flow axis
        if axis == 0:  # X-axis
            if self._pig_type != 3:  # Only rotate non-spherical pigs
                transform.RotateZ(90.0)
            transform.Translate(center_pos, 0, 0)
        elif axis == 1:  # Y-axis
            if self._pig_type != 3:
                transform.RotateX(90.0)
            transform.Translate(0, center_pos, 0)
        else:  # Z-axis
            transform.Translate(0, 0, center_pos)
        
        # Apply transform
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputData(pig)
        transform_filter.SetTransform(transform)
        transform_filter.Update()
        
        # Add a field for pig type
        pig_data = transform_filter.GetOutput()
        type_array = vtk.vtkIntArray()
        type_array.SetName("PigType")
        type_array.SetNumberOfComponents(1)
        type_array.SetNumberOfTuples(pig_data.GetNumberOfPoints())
        type_array.Fill(self._pig_type)
        pig_data.GetPointData().AddArray(type_array)
        
        return pig_data
    
    def FillInputPortInformation(self, port, info):
        info.Set(vtk.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet")
        return 1


@smproxy.filter(name="OilGasWellboreFilter", label="Oil & Gas Wellbore Visualization")
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=True)
class WellboreFilter(VTKPythonAlgorithmBase):
    """
    Filter to enhance wellbore visualization.
    
    This filter provides specialized visualization techniques for wellbore
    simulations, including unwrapping for easier viewing of radial profiles.
    """
    
    @smproperty.doublevector(name="UnwrapAngle", default_values=[360.0])
    @smdomain.doublerange(min=0.0, max=360.0)
    def SetUnwrapAngle(self, angle):
        """
        Set the angle range for unwrapping.
        
        Args:
            angle (float): Angle in degrees (360 = full unwrap)
        """
        self._unwrap_angle = angle
        self.Modified()
    
    @smproperty.doublevector(name="RadialExaggeration", default_values=[1.0])
    @smdomain.doublerange(min=0.1, max=10.0)
    def SetRadialExaggeration(self, scale):
        """
        Set the radial exaggeration factor.
        
        Args:
            scale (float): Scaling factor for radial dimension
        """
        self._radial_exaggeration = scale
        self.Modified()
    
    @smproperty.intvector(name="UnwrapMode", default_values=[0])
    @smdomain.intrange(min=0, max=2)
    def SetUnwrapMode(self, mode):
        """
        Set the unwrap mode.
        
        Args:
            mode (int): 0=None, 1=Cylindrical, 2=Flattened
        """
        self._unwrap_mode = mode
        self.Modified()
    
    @smproperty.stringvector(name="WellboreAxis", default_values="Z")
    @smdomain.stringlistdomain(strings=["X", "Y", "Z"])
    def SetWellboreAxis(self, axis):
        """
        Set the wellbore axis.
        
        Args:
            axis (str): Wellbore axis (X, Y, or Z)
        """
        self._wellbore_axis = axis
        self.Modified()
    
    @smproperty.stringvector(name="DataField", default_values="Pressure")
    @smdomain.datatype(dataTypes=["vtkDataArray"])
    def SetDataField(self, field):
        """
        Set the data field to use for coloring or warping.
        
        Args:
            field (str): Data field name
        """
        self._data_field = field
        self.Modified()
    
    @smproperty.intvector(name="WarpByData", default_values=[0])
    @smdomain.intrange(min=0, max=1)
    def SetWarpByData(self, enable):
        """
        Set whether to warp by data values.
        
        Args:
            enable (int): 0=Disable, 1=Enable
        """
        self._warp_by_data = bool(enable)
        self.Modified()
    
    @smproperty.doublevector(name="WarpFactor", default_values=[1.0])
    @smdomain.doublerange(min=0.0, max=10.0)
    def SetWarpFactor(self, factor):
        """
        Set the warp factor.
        
        Args:
            factor (float): Scale factor for warping
        """
        self._warp_factor = factor
        self.Modified()
    
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, 
            nInputPorts=1, 
            nOutputPorts=1,
            outputType="vtkDataSet"
        )
        self._unwrap_angle = 360.0
        self._radial_exaggeration = 1.0
        self._unwrap_mode = 0
        self._wellbore_axis = "Z"
        self._data_field = "Pressure"
        self._warp_by_data = False
        self._warp_factor = 1.0
    
    def RequestDataObject(self, request, inInfo, outInfo):
        """
        Create the output data object type based on the unwrap mode.
        
        Args:
            request: Pipeline request
            inInfo: Input information
            outInfo: Output information
            
        Returns:
            int: 1 for success
        """
        input_data = vtkDataSet.GetData(inInfo[0], 0)
        if not input_data:
            return 1
        
        output = vtkDataSet.GetData(outInfo)
        if not output or not output.IsA(input_data.GetClassName()):
            output_type = input_data.GetClassName()
            output = vtk.vtkDataObjectTypes.NewDataObject(output_type)
            outInfo.GetInformationObject(0).Set(vtkDataObject.DATA_OBJECT(), output)
        
        return 1
    
    def RequestData(self, request, inInfo, outInfo):
        """
        Execute the filter.
        
        Args:
            request: Pipeline request
            inInfo: Input information
            outInfo: Output information
            
        Returns:
            int: 1 for success, 0 for failure
        """
        # Get input data
        input_data = vtkDataSet.GetData(inInfo[0], 0)
        if not input_data:
            logger.error("No input dataset provided")
            return 0
        
        # Make a deep copy to avoid modifying input
        output_data = input_data.NewInstance()
        output_data.DeepCopy(input_data)
        
        # Apply transform based on unwrap mode
        if self._unwrap_mode > 0:
            # Get axis indices
            if self._wellbore_axis == "X":
                axis_idx = 0
                r1_idx = 1
                r2_idx = 2
            elif self._wellbore_axis == "Y":
                axis_idx = 1
                r1_idx = 0
                r2_idx = 2
            else:  # Z
                axis_idx = 2
                r1_idx = 0
                r2_idx = 1
            
            # Transform points
            num_points = output_data.GetNumberOfPoints()
            for i in range(num_points):
                point = output_data.GetPoint(i)
                
                # Extract coordinates
                x, y, z = point
                coords = [x, y, z]
                
                # Calculate cylindrical coordinates
                axis_val = coords[axis_idx]
                r1 = coords[r1_idx]
                r2 = coords[r2_idx]
                
                # Calculate radius and angle
                radius = (r1**2 + r2**2)**0.5
                angle = np.arctan2(r2, r1)
                
                if self._unwrap_mode == 1:  # Cylindrical unwrap
                    # Scale radius
                    radius *= self._radial_exaggeration
                    
                    # Limit angle to unwrap range
                    unwrap_rad = np.radians(self._unwrap_angle)
                    if angle < 0:
                        angle += 2 * np.pi
                    if angle > unwrap_rad:
                        angle = unwrap_rad
                    
                    # Convert back to Cartesian
                    new_r1 = radius * np.cos(angle)
                    new_r2 = radius * np.sin(angle)
                    
                    coords[r1_idx] = new_r1
                    coords[r2_idx] = new_r2
                
                elif self._unwrap_mode == 2:  # Flattened unwrap
                    # Scale radius
                    radius *= self._radial_exaggeration
                    
                    # Normalize angle to [0, 2π)
                    if angle < 0:
                        angle += 2 * np.pi
                    
                    # Scale angle to target range
                    unwrap_rad = np.radians(self._unwrap_angle)
                    normalized_angle = angle / (2 * np.pi) * unwrap_rad
                    
                    # Convert to "flat" representation
                    new_r1 = normalized_angle
                    new_r2 = radius
                    
                    coords[r1_idx] = new_r1
                    coords[r2_idx] = new_r2
                
                # Update point
                output_data.GetPoints().SetPoint(i, coords)
        
        # Apply warp by data if requested
        if self._warp_by_data and output_data.GetPointData().HasArray(self._data_field):
            # Create a warp scalar filter
            warp = vtkWarpScalar()
            warp.SetInputData(output_data)
            warp.SetScaleFactor(self._warp_factor)
            warp.SetInputArrayToProcess(
                0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, self._data_field
            )
            warp.Update()
            output_data = warp.GetOutput()
        
        # Set output
        output = vtkDataSet.GetData(outInfo)
        output.ShallowCopy(output_data)
        
        return 1
    
    def FillInputPortInformation(self, port, info):
        info.Set(vtk.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet")
        return 1


@smproxy.filter(name="OilGasSpillFilter", label="Oil & Gas Spill Visualization")
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=True)
class SpillFilter(VTKPythonAlgorithmBase):
    """
    Filter to enhance visualization of spill simulations.
    
    This filter provides specialized visualization techniques for oil spill
    simulations, including thickness maps and time-based spreading.
    """
    
    @smproperty.stringvector(name="SpillField", default_values="alpha.oil")
    @smdomain.datatype(dataTypes=["vtkDataArray"])
    def SetSpillField(self, field):
        """
        Set the spill concentration field.
        
        Args:
            field (str): Spill concentration field name
        """
        self._spill_field = field
        self.Modified()
    
    @smproperty.doublevector(name="ConcentrationThreshold", default_values=[0.1])
    @smdomain.doublerange(min=0.0, max=1.0)
    def SetConcentrationThreshold(self, threshold):
        """
        Set the concentration threshold for spill extent.
        
        Args:
            threshold (float): Minimum concentration to consider as spill
        """
        self._concentration_threshold = threshold
        self.Modified()
    
    @smproperty.doublevector(name="ThicknessScale", default_values=[1.0])
    @smdomain.doublerange(min=0.1, max=100.0)
    def SetThicknessScale(self, scale):
        """
        Set the scale factor for thickness visualization.
        
        Args:
            scale (float): Scale factor for thickness
        """
        self._thickness_scale = scale
        self.Modified()
    
    @smproperty.intvector(name="VisualizationMode", default_values=[0])
    @smdomain.intrange(min=0, max=3)
    def SetVisualizationMode(self, mode):
        """
        Set the visualization mode.
        
        Args:
            mode (int): 0=Surface, 1=Volume, 2=Thickness Map, 3=Time-based
        """
        self._visualization_mode = mode
        self.Modified()
    
    @smproperty.stringvector(name="WaterSurfaceField", default_values="zWater")
    @smdomain.datatype(dataTypes=["vtkDataArray"])
    def SetWaterSurfaceField(self, field):
        """
        Set the field representing water surface height.
        
        Args:
            field (str): Water surface height field name
        """
        self._water_surface_field = field
        self.Modified()
    
    @smproperty.doublevector(name="SimulationTime", default_values=[0.0])
    @smdomain.doublerange(min=0.0, max=1e6)
    def SetSimulationTime(self, time):
        """
        Set the simulation time for time-based visualization.
        
        Args:
            time (float): Simulation time
        """
        self._simulation_time = time
        self.Modified()
    
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, 
            nInputPorts=1, 
            nOutputPorts=1,
            outputType="vtkPolyData"
        )
        self._spill_field = "alpha.oil"
        self._concentration_threshold = 0.1
        self._thickness_scale = 1.0
        self._visualization_mode = 0
        self._water_surface_field = "zWater"
        self._simulation_time = 0.0
    
    def RequestData(self, request, inInfo, outInfo):
        """
        Execute the filter.
        
        Args:
            request: Pipeline request
            inInfo: Input information
            outInfo: Output information
            
        Returns:
            int: 1 for success, 0 for failure
        """
        # Get input data
        input_data = vtkDataSet.GetData(inInfo[0], 0)
        if not input_data:
            logger.error("No input dataset provided")
            return 0
        
        # Check if spill field exists
        if not input_data.GetPointData().HasArray(self._spill_field):
            logger.error(f"Spill field '{self._spill_field}' not found in dataset")
            return 0
        
        # Process based on visualization mode
        if self._visualization_mode == 0:  # Surface
            output_data = self._create_surface_visualization(input_data)
        elif self._visualization_mode == 1:  # Volume
            output_data = self._create_volume_visualization(input_data)
        elif self._visualization_mode == 2:  # Thickness Map
            output_data = self._create_thickness_map(input_data)
        elif self._visualization_mode == 3:  # Time-based
            output_data = self._create_time_based_visualization(input_data)
        else:
            logger.error(f"Invalid visualization mode: {self._visualization_mode}")
            return 0
        
        if not output_data:
            logger.error("Failed to create visualization")
            return 0
        
        # Set output
        output = vtkPolyData.GetData(outInfo)
        output.ShallowCopy(output_data)
        
        return 1
    
    def _create_surface_visualization(self, input_data):
        """
        Create a surface visualization of the spill.
        
        Args:
            input_data (vtkDataSet): Input dataset
            
        Returns:
            vtkPolyData: Surface representation
        """
        # Create contour filter
        contour = vtkContourFilter()
        contour.SetInputData(input_data)
        contour.SetInputArrayToProcess(
            0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, self._spill_field
        )
        contour.SetValue(0, self._concentration_threshold)
        contour.Update()
        
        return contour.GetOutput()
    
    def _create_volume_visualization(self, input_data):
        """
        Create a volume visualization of the spill.
        
        Args:
            input_data (vtkDataSet): Input dataset
            
        Returns:
            vtkPolyData: Spill volume representation
        """
        # Threshold to extract spill volume
        threshold = vtkThreshold()
        threshold.SetInputData(input_data)
        threshold.SetInputArrayToProcess(
            0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, self._spill_field
        )
        threshold.ThresholdByUpper(self._concentration_threshold)
        threshold.Update()
        
        # Convert to surface
        surface = vtk.vtkDataSetSurfaceFilter()
        surface.SetInputData(threshold.GetOutput())
        surface.Update()
        
        return surface.GetOutput()
    
    def _create_thickness_map(self, input_data):
        """
        Create a thickness map of the spill.
        
        Args:
            input_data (vtkDataSet): Input dataset
            
        Returns:
            vtkPolyData: Thickness map representation
        """
        # Check if water surface field exists
        has_water_surface = input_data.GetPointData().HasArray(self._water_surface_field)
        
        # Extract top surface
        if has_water_surface:
            # Use water surface field
            # This is a simplified approach - in a real implementation,
            # we would need to project the spill onto the water surface
            
            # For now, just extract a contour at the water surface
            contour = vtkContourFilter()
            contour.SetInputData(input_data)
            contour.SetInputArrayToProcess(
                0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, self._water_surface_field
            )
            # Get average water surface value
            water_array = input_data.GetPointData().GetArray(self._water_surface_field)
            avg_water_level = 0.0
            for i in range(water_array.GetNumberOfTuples()):
                avg_water_level += water_array.GetValue(i)
            avg_water_level /= water_array.GetNumberOfTuples()
            
            contour.SetValue(0, avg_water_level)
            contour.Update()
            surface = contour.GetOutput()
        else:
            # Extract top surface based on Z coordinate
            bounds = input_data.GetBounds()
            top_z = bounds[5]  # Max Z value
            
            plane = vtk.vtkPlane()
            plane.SetOrigin(0, 0, top_z)
            plane.SetNormal(0, 0, 1)
            
            cutter = vtkCutter()
            cutter.SetInputData(input_data)
            cutter.SetCutFunction(plane)
            cutter.Update()
            surface = cutter.GetOutput()
        
        # Add thickness field
        thickness_array = vtk.vtkFloatArray()
        thickness_array.SetName("SpillThickness")
        thickness_array.SetNumberOfComponents(1)
        thickness_array.SetNumberOfTuples(surface.GetNumberOfPoints())
        
        # Calculate thickness based on concentration
        spill_array = surface.GetPointData().GetArray(self._spill_field)
        for i in range(surface.GetNumberOfPoints()):
            concentration = spill_array.GetValue(i)
            thickness = 0.0
            if concentration > self._concentration_threshold:
                # Simple linear model for thickness
                thickness = (concentration - self._concentration_threshold) * self._thickness_scale
            thickness_array.SetValue(i, thickness)
        
        surface.GetPointData().AddArray(thickness_array)
        
        # Warp surface by thickness
        warp = vtkWarpScalar()
        warp.SetInputData(surface)
        warp.SetInputArrayToProcess(
            0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, "SpillThickness"
        )
        warp.SetScaleFactor(1.0)  # Already scaled in the array
        warp.Update()
        
        return warp.GetOutput()
    
    def _create_time_based_visualization(self, input_data):
        """
        Create a time-based visualization of the spill.
        
        Args:
            input_data (vtkDataSet): Input dataset
            
        Returns:
            vtkPolyData: Time-based spill representation
        """
        # For time-based visualization, we need to have time information
        # This is a simplified implementation
        
        # First, extract the spill surface
        surface = self._create_surface_visualization(input_data)
        
        # Create a time-dependent radius field
        time_array = vtk.vtkFloatArray()
        time_array.SetName("SpillAge")
        time_array.SetNumberOfComponents(1)
        time_array.SetNumberOfTuples(surface.GetNumberOfPoints())
        
        # We'll use the distance from a center point as a proxy for time
        # In a real implementation, we would use actual time data from the simulation
        bounds = surface.GetBounds()
        center_x = (bounds[0] + bounds[1]) / 2
        center_y = (bounds[2] + bounds[3]) / 2
        
        for i in range(surface.GetNumberOfPoints()):
            point = surface.GetPoint(i)
            # Calculate distance from center
            dx = point[0] - center_x
            dy = point[1] - center_y
            distance = (dx**2 + dy**2)**0.5
            
            # Normalize distance and scale by simulation time
            max_distance = ((bounds[1] - bounds[0])**2 + (bounds[3] - bounds[2])**2)**0.5 / 2
            normalized_distance = distance / max_distance
            time_value = (1 - normalized_distance) * self._simulation_time
            
            time_array.SetValue(i, time_value)
        
        surface.GetPointData().AddArray(time_array)
        
        return surface
    
    def FillInputPortInformation(self, port, info):
        info.Set(vtk.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet")
        return 1


@smproxy.filter(name="OilGasSeparatorFilter", label="Oil & Gas Separator Visualization")
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=True)
class SeparatorFilter(VTKPythonAlgorithmBase):
    """
    Filter to visualize oil & gas separation equipment.
    
    This filter enhances the visualization of separation equipment like
    gravity separators, cyclones, and other processing equipment.
    """
    
    @smproperty.intvector(name="SeparatorType", default_values=[0])
    @smdomain.intrange(min=0, max=3)
    def SetSeparatorType(self, sep_type):
        """
        Set the separator type.
        
        Args:
            sep_type (int): 0=Gravity, 1=Cyclone, 2=Electrostatic, 3=Filter
        """
        self._separator_type = sep_type
        self.Modified()
    
    @smproperty.stringvector(name="OilPhaseName", default_values="alpha.oil")
    def SetOilPhaseName(self, name):
        """
        Set the oil phase field name.
        
        Args:
            name (str): Name of the oil phase field
        """
        self._oil_phase_name = name
        self.Modified()
    
    @smproperty.stringvector(name="WaterPhaseName", default_values="alpha.water")
    def SetWaterPhaseName(self, name):
        """
        Set the water phase field name.
        
        Args:
            name (str): Name of the water phase field
        """
        self._water_phase_name = name
        self.Modified()
    
    @smproperty.stringvector(name="GasPhaseName", default_values="alpha.gas")
    def SetGasPhaseName(self, name):
        """
        Set the gas phase field name.
        
        Args:
            name (str): Name of the gas phase field
        """
        self._gas_phase_name = name
        self.Modified()
    
    @smproperty.stringvector(name="VelocityField", default_values="U")
    def SetVelocityField(self, name):
        """
        Set the velocity field name.
        
        Args:
            name (str): Name of the velocity field
        """
        self._velocity_field = name
        self.Modified()
    
    @smproperty.intvector(name="StreamlinesDensity", default_values=[50])
    @smdomain.intrange(min=10, max=500)
    def SetStreamlinesDensity(self, density):
        """
        Set the density of streamlines.
        
        Args:
            density (int): Number of streamlines to generate
        """
        self._streamlines_density = density
        self.Modified()
    
    @smproperty.intvector(name="ShowEfficiency", default_values=[0])
    @smdomain.intrange(min=0, max=1)
    def SetShowEfficiency(self, show):
        """
        Set whether to show separation efficiency.
        
        Args:
            show (int): 0=Disable, 1=Enable
        """
        self._show_efficiency = bool(show)
        self.Modified()
    
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, 
            nInputPorts=1, 
            nOutputPorts=1,
            outputType="vtkMultiBlockDataSet"
        )
        self._separator_type = 0
        self._oil_phase_name = "alpha.oil"
        self._water_phase_name = "alpha.water"
        self._gas_phase_name = "alpha.gas"
        self._velocity_field = "U"
        self._streamlines_density = 50
        self._show_efficiency = False
    
    def RequestData(self, request, inInfo, outInfo):
        """
        Execute the filter.
        
        Args:
            request: Pipeline request
            inInfo: Input information
            outInfo: Output information
            
        Returns:
            int: 1 for success, 0 for failure
        """
        # Get input data
        input_data = vtkDataSet.GetData(inInfo[0], 0)
        if not input_data:
            logger.error("No input dataset provided")
            return 0
        
        # Check for required fields
        point_data = input_data.GetPointData()
        has_velocity = point_data.HasArray(self._velocity_field)
        has_oil = point_data.HasArray(self._oil_phase_name)
        has_water = point_data.HasArray(self._water_phase_name)
        has_gas = point_data.HasArray(self._gas_phase_name)
        
        if not has_velocity:
            logger.warning(f"Velocity field '{self._velocity_field}' not found. Streamlines will not be generated.")
        
        if not (has_oil or has_water or has_gas):
            logger.warning("None of the phase fields found. Phase visualization will be limited.")
        
        # Create output multi-block dataset
        output = vtkMultiBlockDataSet.GetData(outInfo)
        output.SetNumberOfBlocks(4)  # Original, Phases, Streamlines, Efficiency
        
        # Block 0: Original dataset
        output.SetBlock(0, input_data)
        output.GetMetaData(0).Set(vtk.vtkCompositeDataSet.NAME(), "Original")
        
        # Block 1: Phase visualization
        if has_oil or has_water or has_gas:
            phase_vis = self._create_phase_visualization(input_data)
            output.SetBlock(1, phase_vis)
            output.GetMetaData(1).Set(vtk.vtkCompositeDataSet.NAME(), "Phases")
        else:
            output.SetBlock(1, None)
        
        # Block 2: Streamlines
        if has_velocity:
            streamlines = self._create_streamlines(input_data)
            output.SetBlock(2, streamlines)
            output.GetMetaData(2).Set(vtk.vtkCompositeDataSet.NAME(), "Streamlines")
        else:
            output.SetBlock(2, None)
        
        # Block 3: Efficiency visualization
        if self._show_efficiency and (has_oil or has_water or has_gas):
            efficiency_vis = self._create_efficiency_visualization(input_data)
            output.SetBlock(3, efficiency_vis)
            output.GetMetaData(3).Set(vtk.vtkCompositeDataSet.NAME(), "Efficiency")
        else:
            output.SetBlock(3, None)
        
        return 1
    
    def _create_phase_visualization(self, input_data):
        """
        Create visualization of phases.
        
        Args:
            input_data (vtkDataSet): Input dataset
            
        Returns:
            vtkPolyData: Phase visualization
        """
        # Extract phases using contours
        point_data = input_data.GetPointData()
        has_oil = point_data.HasArray(self._oil_phase_name)
        has_water = point_data.HasArray(self._water_phase_name)
        has_gas = point_data.HasArray(self._gas_phase_name)
        
        # Create a multi-block dataset for all phases
        phases = vtkMultiBlockDataSet()
        block_count = 0
        
        if has_oil:
            oil_contour = vtkContourFilter()
            oil_contour.SetInputData(input_data)
            oil_contour.SetInputArrayToProcess(
                0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, self._oil_phase_name
            )
            oil_contour.SetValue(0, 0.5)  # Threshold at 50% oil
            oil_contour.Update()
            
            phases.SetNumberOfBlocks(block_count + 1)
            phases.SetBlock(block_count, oil_contour.GetOutput())
            phases.GetMetaData(block_count).Set(vtk.vtkCompositeDataSet.NAME(), "Oil")
            block_count += 1
        
        if has_water:
            water_contour = vtkContourFilter()
            water_contour.SetInputData(input_data)
            water_contour.SetInputArrayToProcess(
                0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, self._water_phase_name
            )
            water_contour.SetValue(0, 0.5)  # Threshold at 50% water
            water_contour.Update()
            
            phases.SetNumberOfBlocks(block_count + 1)
            phases.SetBlock(block_count, water_contour.GetOutput())
            phases.GetMetaData(block_count).Set(vtk.vtkCompositeDataSet.NAME(), "Water")
            block_count += 1
        
        if has_gas:
            gas_contour = vtkContourFilter()
            gas_contour.SetInputData(input_data)
            gas_contour.SetInputArrayToProcess(
                0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, self._gas_phase_name
            )
            gas_contour.SetValue(0, 0.5)  # Threshold at 50% gas
            gas_contour.Update()
            
            phases.SetNumberOfBlocks(block_count + 1)
            phases.SetBlock(block_count, gas_contour.GetOutput())
            phases.GetMetaData(block_count).Set(vtk.vtkCompositeDataSet.NAME(), "Gas")
            block_count += 1
        
        if block_count == 0:
            # Create an empty polydata if no phases
            empty = vtkPolyData()
            phases.SetNumberOfBlocks(1)
            phases.SetBlock(0, empty)
        
        return phases
    
    def _create_streamlines(self, input_data):
        """
        Create streamlines visualization.
        
        Args:
            input_data (vtkDataSet): Input dataset
            
        Returns:
            vtkPolyData: Streamlines visualization
        """
        # Check for velocity field
        if not input_data.GetPointData().HasArray(self._velocity_field):
            return vtkPolyData()
        
        # Create streamlines
        stream_tracer = vtkStreamTracer()
        stream_tracer.SetInputData(input_data)
        stream_tracer.SetInputArrayToProcess(
            0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, self._velocity_field
        )
        
        # Set integration parameters
        stream_tracer.SetIntegrationDirectionToBoth()
        stream_tracer.SetMaximumPropagation(1000)
        stream_tracer.SetIntegrationStepUnit(vtkStreamTracer.CELL_LENGTH_UNIT)
        stream_tracer.SetInitialIntegrationStep(0.05)
        
        # Create seed points
        bounds = input_data.GetBounds()
        seeds = vtk.vtkPointSource()
        seeds.SetNumberOfPoints(self._streamlines_density)
        
        # Adjust seed location based on separator type
        if self._separator_type == 0:  # Gravity separator
            # Seeds in upper part
            seeds.SetCenter(
                (bounds[0] + bounds[1]) / 2,
                (bounds[2] + bounds[3]) / 2,
                bounds[5] - (bounds[5] - bounds[4]) * 0.2
            )
            seeds.SetRadius(min(bounds[1] - bounds[0], bounds[3] - bounds[2]) * 0.4)
        elif self._separator_type == 1:  # Cyclone
            # Seeds near inlet
            seeds.SetCenter(
                bounds[1] - (bounds[1] - bounds[0]) * 0.1,
                (bounds[2] + bounds[3]) / 2,
                (bounds[4] + bounds[5]) / 2
            )
            seeds.SetRadius(min(bounds[1] - bounds[0], bounds[3] - bounds[2]) * 0.3)
        else:
            # Default center seeds
            seeds.SetCenter(
                (bounds[0] + bounds[1]) / 2,
                (bounds[2] + bounds[3]) / 2,
                (bounds[4] + bounds[5]) / 2
            )
            seeds.SetRadius(min(bounds[1] - bounds[0], bounds[3] - bounds[2]) * 0.4)
        
        seeds.SetDistributionToUniform()
        seeds.Update()
        
        stream_tracer.SetSourceConnection(seeds.GetOutputPort())
        stream_tracer.Update()
        
        # Add a field to color by separator type
        streamlines = stream_tracer.GetOutput()
        type_array = vtk.vtkIntArray()
        type_array.SetName("SeparatorType")
        type_array.SetNumberOfComponents(1)
        type_array.SetNumberOfTuples(streamlines.GetNumberOfPoints())
        type_array.Fill(self._separator_type)
        streamlines.GetPointData().AddArray(type_array)
        
        return streamlines
    
    def _create_efficiency_visualization(self, input_data):
        """
        Create visualization of separation efficiency.
        
        Args:
            input_data (vtkDataSet): Input dataset
            
        Returns:
            vtkPolyData: Efficiency visualization
        """
        # This is a simplified implementation
        # In a real implementation, we would analyze the flow and calculate
        # actual separation efficiency based on inlet/outlet compositions
        
        # For now, just create a 2D plot showing the efficiency
        # We'll use a flat surface with a scalar field
        
        # Create a plane
        bounds = input_data.GetBounds()
        plane = vtk.vtkPlaneSource()
        
        # Position the plane based on separator type
        if self._separator_type == 0:  # Gravity separator
            # Position on side
            plane.SetOrigin(
                bounds[0] - (bounds[1] - bounds[0]) * 0.2,
                bounds[2],
                bounds[4]
            )
            plane.SetPoint1(
                bounds[0] - (bounds[1] - bounds[0]) * 0.2,
                bounds[3],
                bounds[4]
            )
            plane.SetPoint2(
                bounds[0] - (bounds[1] - bounds[0]) * 0.2,
                bounds[2],
                bounds[5]
            )
        elif self._separator_type == 1:  # Cyclone
            # Position at bottom
            plane.SetOrigin(
                bounds[0],
                bounds[2],
                bounds[4] - (bounds[5] - bounds[4]) * 0.2
            )
            plane.SetPoint1(
                bounds[1],
                bounds[2],
                bounds[4] - (bounds[5] - bounds[4]) * 0.2
            )
            plane.SetPoint2(
                bounds[0],
                bounds[3],
                bounds[4] - (bounds[5] - bounds[4]) * 0.2
            )
        else:
            # Default position
            plane.SetOrigin(
                bounds[0] - (bounds[1] - bounds[0]) * 0.2,
                bounds[2],
                bounds[4]
            )
            plane.SetPoint1(
                bounds[0] - (bounds[1] - bounds[0]) * 0.2,
                bounds[3],
                bounds[4]
            )
            plane.SetPoint2(
                bounds[0] - (bounds[1] - bounds[0]) * 0.2,
                bounds[2],
                bounds[5]
            )
        
        plane.SetResolution(20, 20)
        plane.Update()
        
        # Create efficiency field
        efficiency = vtk.vtkFloatArray()
        efficiency.SetName("SeparationEfficiency")
        efficiency.SetNumberOfComponents(1)
        efficiency.SetNumberOfTuples(plane.GetOutput().GetNumberOfPoints())
        
        # Simple mock efficiency calculation
        # In a real implementation, this would be based on actual results
        point_data = input_data.GetPointData()
        has_oil = point_data.HasArray(self._oil_phase_name)
        has_water = point_data.HasArray(self._water_phase_name)
        has_gas = point_data.HasArray(self._gas_phase_name)
        
        # Generate a gradient from 0.7 to 0.95 to simulate efficiency
        resolution = plane.GetResolution()
        for j in range(resolution[1] + 1):
            for i in range(resolution[0] + 1):
                idx = j * (resolution[0] + 1) + i
                
                # Simple gradient based on position
                norm_i = i / resolution[0]
                norm_j = j / resolution[1]
                
                # Adjust efficiency based on separator type
                if self._separator_type == 0:  # Gravity separator
                    # Higher efficiency at the bottom for water, top for gas
                    if has_water and has_gas:
                        eff_water = 0.75 + 0.2 * (1 - norm_j)  # Better at bottom
                        eff_gas = 0.75 + 0.2 * norm_j  # Better at top
                        efficiency.SetValue(idx, (eff_water + eff_gas) / 2)
                    else:
                        efficiency.SetValue(idx, 0.75 + 0.2 * norm_i)
                elif self._separator_type == 1:  # Cyclone
                    # Higher efficiency at outside for heavy phases
                    radius = ((norm_i - 0.5)**2 + (norm_j - 0.5)**2)**0.5
                    efficiency.SetValue(idx, 0.75 + 0.2 * radius * 2)
                else:
                    # Default linear gradient
                    efficiency.SetValue(idx, 0.75 + 0.2 * norm_i)
        
        plane.GetOutput().GetPointData().AddArray(efficiency)
        
        # Warp by efficiency to create a 3D plot
        warp = vtkWarpScalar()
        warp.SetInputData(plane.GetOutput())
        warp.SetInputArrayToProcess(
            0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, "SeparationEfficiency"
        )
        warp.SetScaleFactor(min(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) * 0.2)
        warp.Update()
        
        return warp.GetOutput()
    
    def FillInputPortInformation(self, port, info):
        info.Set(vtk.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet")
        return 1


# Register plugins in ParaView
def register_plugins():
    """Register all filter plugins with ParaView."""
    if PARAVIEW_AVAILABLE:
        from paraview.util.vtkAlgorithm import smproxy
        # The smproxy decorator already handles registration
        logger.info("Registered Oil & Gas ParaView filter plugins")
    else:
        logger.warning("ParaView is not available, plugins not registered")

# Auto-register when the module is imported
if PARAVIEW_AVAILABLE:
    register_plugins()