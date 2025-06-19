#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline component definitions for Openfoam_Simulator.

This module provides classes and functions for defining, configuring, and
simulating oil & gas pipeline components including:
- Pipe sections (straight pipes, bends, risers)
- Connectors (elbows, tees, reducers, junctions)
- Valves (gate, ball, check valves)
- Flow control devices
- Instrumentation
- Pipeline accessories (pigs, launchers/receivers)
"""

import math
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# Import utility modules
from ..utils.logger import get_logger
from ..utils.unit_converter import convert_length, convert_pressure

# Import related industry modules
from .oilgas_properties import FluidProperties

logger = get_logger(__name__)


class ComponentType(Enum):
    """Enumeration of pipeline component types."""
    PIPE = "Pipe"
    ELBOW = "Elbow"
    TEE = "Tee"
    CROSS = "Cross"
    REDUCER = "Reducer"
    VALVE = "Valve"
    CHECK_VALVE = "Check Valve"
    PUMP = "Pump"
    COMPRESSOR = "Compressor"
    SEPARATOR = "Separator"
    LAUNCHER = "Pig Launcher"
    RECEIVER = "Pig Receiver"
    METER = "Flow Meter"
    SENSOR = "Sensor"
    RISER = "Riser"
    CUSTOM = "Custom"


class PipeSchedule(Enum):
    """Standard pipe schedules."""
    SCH5 = "5"
    SCH10 = "10"
    SCH20 = "20"
    SCH30 = "30"
    SCH40 = "40"
    SCH60 = "60"
    SCH80 = "80"
    SCH100 = "100"
    SCH120 = "120"
    SCH140 = "140"
    SCH160 = "160"
    STD = "STD"
    XS = "XS"
    XXS = "XXS"
    CUSTOM = "Custom"


class ConnectionType(Enum):
    """Types of pipe connections."""
    FLANGED = "Flanged"
    THREADED = "Threaded"
    WELDED = "Welded"
    MECHANICAL = "Mechanical"
    COMPRESSION = "Compression"
    CUSTOM = "Custom"


class Material(Enum):
    """Common pipeline materials."""
    CARBON_STEEL = "Carbon Steel"
    STAINLESS_STEEL = "Stainless Steel"
    DUPLEX = "Duplex"
    SUPER_DUPLEX = "Super Duplex"
    HDPE = "HDPE"
    PVC = "PVC"
    FIBERGLASS = "Fiberglass"
    COPPER = "Copper"
    ALUMINUM = "Aluminum"
    CONCRETE = "Concrete"
    TITANIUM = "Titanium"
    LINED = "Lined"
    CUSTOM = "Custom"


class ValveType(Enum):
    """Types of valves."""
    GATE = "Gate Valve"
    BALL = "Ball Valve"
    GLOBE = "Globe Valve"
    BUTTERFLY = "Butterfly Valve"
    CHECK = "Check Valve"
    NEEDLE = "Needle Valve"
    PLUG = "Plug Valve"
    DIAPHRAGM = "Diaphragm Valve"
    CONTROL = "Control Valve"
    RELIEF = "Relief Valve"
    CUSTOM = "Custom"


class InstrumentType(Enum):
    """Types of pipeline instruments."""
    FLOW_METER = "Flow Meter"
    PRESSURE_TRANSMITTER = "Pressure Transmitter"
    TEMPERATURE_TRANSMITTER = "Temperature Transmitter"
    LEVEL_TRANSMITTER = "Level Transmitter"
    DENSITY_METER = "Density Meter"
    WATER_CUT_METER = "Water Cut Meter"
    PRESSURE_GAUGE = "Pressure Gauge"
    TEMPERATURE_GAUGE = "Temperature Gauge"
    CUSTOM = "Custom"


# Base class for all pipeline components
class PipelineComponent:
    """Base class for pipeline components."""
    
    def __init__(self, name: str, component_type: ComponentType, 
                 material: Material = Material.CARBON_STEEL):
        """
        Initialize pipeline component.
        
        Args:
            name (str): Component name
            component_type (ComponentType): Type of component
            material (Material, optional): Component material
        """
        self.name = name
        self.component_type = component_type
        self.material = material
        
        # Position and orientation
        self.position = [0.0, 0.0, 0.0]  # x, y, z in meters
        self.orientation = [0.0, 0.0, 0.0]  # rotation around x, y, z in degrees
        
        # Connections to other components
        self.connections = []  # List of connected components
        
        # Physical properties
        self.roughness = 0.0000457  # surface roughness in meters (default for commercial steel)
        self.k_factor = 0.0  # resistance coefficient (default for straight pipe)
        
        # Visualization settings
        self.color = [0.5, 0.5, 0.5]  # RGB, default gray
        self.opacity = 1.0  # Fully opaque by default
        self.visible = True
        
        # Custom properties
        self.custom_properties = {}
    
    def set_position(self, x: float, y: float, z: float):
        """
        Set component position.
        
        Args:
            x (float): X coordinate in meters
            y (float): Y coordinate in meters
            z (float): Z coordinate in meters
        """
        self.position = [x, y, z]
    
    def set_orientation(self, rx: float, ry: float, rz: float):
        """
        Set component orientation.
        
        Args:
            rx (float): Rotation around X axis in degrees
            ry (float): Rotation around Y axis in degrees
            rz (float): Rotation around Z axis in degrees
        """
        self.orientation = [rx, ry, rz]
    
    def connect(self, component: 'PipelineComponent'):
        """
        Connect to another component.
        
        Args:
            component (PipelineComponent): Component to connect to
        """
        if component not in self.connections:
            self.connections.append(component)
        
        # Reciprocal connection
        if self not in component.connections:
            component.connections.append(self)
    
    def disconnect(self, component: 'PipelineComponent'):
        """
        Disconnect from another component.
        
        Args:
            component (PipelineComponent): Component to disconnect from
        """
        if component in self.connections:
            self.connections.remove(component)
        
        # Reciprocal disconnection
        if self in component.connections:
            component.connections.remove(self)
    
    def calculate_pressure_drop(self, flow_rate: float, fluid: FluidProperties, 
                               inlet_pressure: float) -> float:
        """
        Calculate pressure drop across component.
        
        Args:
            flow_rate (float): Flow rate in m³/s
            fluid (FluidProperties): Fluid properties
            inlet_pressure (float): Inlet pressure in Pa
            
        Returns:
            float: Pressure drop in Pa
            
        Raises:
            NotImplementedError: This method should be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement calculate_pressure_drop")
    
    def get_mesh_parameters(self) -> Dict[str, Any]:
        """
        Get parameters for mesh generation.
        
        Returns:
            Dict[str, Any]: Mesh parameters
            
        Raises:
            NotImplementedError: This method should be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement get_mesh_parameters")
    
    def set_property(self, name: str, value: Any):
        """
        Set a custom property.
        
        Args:
            name (str): Property name
            value (Any): Property value
        """
        self.custom_properties[name] = value
    
    def get_property(self, name: str, default: Any = None) -> Any:
        """
        Get a custom property.
        
        Args:
            name (str): Property name
            default (Any, optional): Default value if property doesn't exist
            
        Returns:
            Any: Property value or default
        """
        return self.custom_properties.get(name, default)


class Pipe(PipelineComponent):
    """Class for straight pipe sections."""
    
    def __init__(self, name: str, length: float, diameter: float, 
                 wall_thickness: float = None, 
                 material: Material = Material.CARBON_STEEL,
                 schedule: PipeSchedule = PipeSchedule.SCH40):
        """
        Initialize pipe section.
        
        Args:
            name (str): Pipe name
            length (float): Pipe length in meters
            diameter (float): Pipe inner diameter in meters
            wall_thickness (float, optional): Wall thickness in meters
            material (Material, optional): Pipe material
            schedule (PipeSchedule, optional): Pipe schedule
        """
        super().__init__(name, ComponentType.PIPE, material)
        
        self.length = length
        self.diameter = diameter  # Inner diameter
        self.schedule = schedule
        
        # Set wall thickness based on schedule if not provided
        if wall_thickness is None:
            self.wall_thickness = self._calculate_wall_thickness()
        else:
            self.wall_thickness = wall_thickness
        
        # Calculate outer diameter
        self.outer_diameter = self.diameter + 2 * self.wall_thickness
        
        # Calculate cross-sectional properties
        self.area = math.pi * (self.diameter / 2)**2  # Flow area
        self.perimeter = math.pi * self.diameter  # Wetted perimeter
        
        # Connection types at each end
        self.connection_type_1 = ConnectionType.WELDED
        self.connection_type_2 = ConnectionType.WELDED
        
        # Set typical roughness for material
        self.roughness = self._get_material_roughness()
        
        # Set visualization color based on material
        self.color = self._get_material_color()
    
    def _calculate_wall_thickness(self) -> float:
        """
        Calculate wall thickness based on schedule.
        
        Returns:
            float: Wall thickness in meters
        """
        # Simplified calculation based on schedule and diameter
        # In practice, this would use standard pipe tables
        
        # Convert to inches for standard schedule calculations
        diameter_inch = self.diameter * 39.3701
        
        # Base thickness on schedule (simplified approximation)
        if self.schedule == PipeSchedule.SCH40:
            if diameter_inch <= 2:
                thickness_inch = 0.065 + 0.047 * diameter_inch
            elif diameter_inch <= 6:
                thickness_inch = 0.083 + 0.033 * diameter_inch
            else:
                thickness_inch = 0.143 + 0.023 * diameter_inch
        elif self.schedule == PipeSchedule.SCH80:
            if diameter_inch <= 2:
                thickness_inch = 0.095 + 0.068 * diameter_inch
            elif diameter_inch <= 6:
                thickness_inch = 0.126 + 0.049 * diameter_inch
            else:
                thickness_inch = 0.242 + 0.032 * diameter_inch
        elif self.schedule == PipeSchedule.SCH160:
            if diameter_inch <= 2:
                thickness_inch = 0.153 + 0.108 * diameter_inch
            elif diameter_inch <= 6:
                thickness_inch = 0.206 + 0.078 * diameter_inch
            else:
                thickness_inch = 0.403 + 0.052 * diameter_inch
        else:
            # Default approximation for other schedules
            thickness_inch = 0.1 * diameter_inch
        
        # Convert back to meters
        return thickness_inch / 39.3701
    
    def _get_material_roughness(self) -> float:
        """
        Get typical roughness for material.
        
        Returns:
            float: Surface roughness in meters
        """
        # Typical roughness values for common pipe materials
        roughness_values = {
            Material.CARBON_STEEL: 0.0000457,  # 0.0457 mm
            Material.STAINLESS_STEEL: 0.0000152,  # 0.0152 mm
            Material.DUPLEX: 0.0000152,  # Same as stainless
            Material.SUPER_DUPLEX: 0.0000152,  # Same as stainless
            Material.HDPE: 0.0000015,  # 0.0015 mm
            Material.PVC: 0.0000015,  # 0.0015 mm
            Material.FIBERGLASS: 0.0000010,  # 0.001 mm (very smooth)
            Material.COPPER: 0.0000015,  # 0.0015 mm
            Material.ALUMINUM: 0.0000015,  # 0.0015 mm
            Material.CONCRETE: 0.0003000,  # 0.3 mm (rough)
            Material.TITANIUM: 0.0000152,  # Similar to stainless
            Material.LINED: 0.0000015,  # Depends on lining material
            Material.CUSTOM: 0.0000457,  # Default to carbon steel
        }
        
        return roughness_values.get(self.material, 0.0000457)
    
    def _get_material_color(self) -> List[float]:
        """
        Get visualization color for material.
        
        Returns:
            List[float]: RGB color values (0-1)
        """
        # Define colors for various materials
        color_map = {
            Material.CARBON_STEEL: [0.6, 0.6, 0.6],  # Gray
            Material.STAINLESS_STEEL: [0.8, 0.8, 0.8],  # Light gray
            Material.DUPLEX: [0.7, 0.7, 0.7],  # Mid gray
            Material.SUPER_DUPLEX: [0.75, 0.75, 0.75],  # Mid gray
            Material.HDPE: [0.1, 0.1, 0.7],  # Blue
            Material.PVC: [0.9, 0.9, 0.9],  # White
            Material.FIBERGLASS: [0.8, 0.7, 0.3],  # Tan
            Material.COPPER: [0.85, 0.45, 0.2],  # Copper
            Material.ALUMINUM: [0.75, 0.75, 0.8],  # Light blue-gray
            Material.CONCRETE: [0.65, 0.65, 0.6],  # Gray-tan
            Material.TITANIUM: [0.6, 0.6, 0.7],  # Bluish gray
            Material.LINED: [0.5, 0.7, 0.7],  # Cyan-gray
            Material.CUSTOM: [0.5, 0.5, 0.5],  # Mid gray
        }
        
        return color_map.get(self.material, [0.5, 0.5, 0.5])
    
    def calculate_pressure_drop(self, flow_rate: float, fluid: FluidProperties, 
                               inlet_pressure: float, temperature: float = 293.15) -> float:
        """
        Calculate pressure drop across the pipe using Darcy-Weisbach equation.
        
        Args:
            flow_rate (float): Flow rate in m³/s
            fluid (FluidProperties): Fluid properties
            inlet_pressure (float): Inlet pressure in Pa
            temperature (float, optional): Fluid temperature in K
            
        Returns:
            float: Pressure drop in Pa
        """
        # Calculate fluid properties at given conditions
        density = fluid.density_at_conditions(temperature, inlet_pressure / 1e5)  # Convert Pa to bar
        viscosity = fluid.viscosity_at_conditions(temperature, inlet_pressure / 1e5)
        
        # Calculate flow velocity
        velocity = flow_rate / self.area
        
        # Calculate Reynolds number
        reynolds = density * velocity * self.diameter / viscosity
        
        # Calculate friction factor using Colebrook equation approximation
        if reynolds < 2300:
            # Laminar flow
            friction_factor = 64 / reynolds
        else:
            # Turbulent flow - Haaland equation (approximation of Colebrook)
            term1 = -1.8 * math.log10(6.9/reynolds + (self.roughness/self.diameter/3.7)**1.11)
            friction_factor = 1 / (term1**2)
        
        # Darcy-Weisbach equation
        pressure_drop = friction_factor * (self.length / self.diameter) * (density * velocity**2 / 2)
        
        return pressure_drop
    
    def get_mesh_parameters(self) -> Dict[str, Any]:
        """
        Get parameters for mesh generation.
        
        Returns:
            Dict[str, Any]: Mesh parameters
        """
        # Basic parameters for a cylindrical pipe
        mesh_params = {
            'type': 'cylinder',
            'length': self.length,
            'diameter': self.diameter,
            'position': self.position,
            'orientation': self.orientation,
            'connections': [conn.name for conn in self.connections],
            'material': self.material.value,
            'roughness': self.roughness
        }
        
        return mesh_params


class PipeBend(PipelineComponent):
    """Class for pipe bends and elbows."""
    
    def __init__(self, name: str, diameter: float, radius: float, angle: float = 90.0,
                 wall_thickness: float = None, 
                 material: Material = Material.CARBON_STEEL,
                 schedule: PipeSchedule = PipeSchedule.SCH40):
        """
        Initialize pipe bend.
        
        Args:
            name (str): Bend name
            diameter (float): Pipe inner diameter in meters
            radius (float): Bend radius in meters
            angle (float, optional): Bend angle in degrees
            wall_thickness (float, optional): Wall thickness in meters
            material (Material, optional): Pipe material
            schedule (PipeSchedule, optional): Pipe schedule
        """
        super().__init__(name, ComponentType.ELBOW, material)
        
        self.diameter = diameter  # Inner diameter
        self.radius = radius  # Bend radius
        self.angle = angle  # Bend angle
        self.schedule = schedule
        
        # Set wall thickness based on schedule if not provided
        if wall_thickness is None:
            # Use the same algorithm as for straight pipe
            temp_pipe = Pipe("temp", 1.0, diameter, None, material, schedule)
            self.wall_thickness = temp_pipe.wall_thickness
        else:
            self.wall_thickness = wall_thickness
        
        # Calculate outer diameter
        self.outer_diameter = self.diameter + 2 * self.wall_thickness
        
        # Calculate bend length (arc length)
        self.length = 2 * math.pi * self.radius * (self.angle / 360.0)
        
        # Calculate cross-sectional properties
        self.area = math.pi * (self.diameter / 2)**2  # Flow area
        self.perimeter = math.pi * self.diameter  # Wetted perimeter
        
        # Connection types at each end
        self.connection_type_1 = ConnectionType.WELDED
        self.connection_type_2 = ConnectionType.WELDED
        
        # Calculate k-factor for pressure drop
        self.k_factor = self._calculate_k_factor()
        
        # Set typical roughness for material
        self.roughness = self._get_material_roughness()
        
        # Set visualization color based on material
        self.color = self._get_material_color()
    
    def _calculate_k_factor(self) -> float:
        """
        Calculate resistance coefficient (k-factor) for the bend.
        
        Returns:
            float: K-factor (dimensionless)
        """
        # R/D ratio (bend radius to pipe diameter)
        r_d_ratio = self.radius / self.diameter
        
        # K-factor calculation based on bend angle and R/D ratio
        # Using Crane TP-410 method (simplified)
        
        # Friction factor for 90-degree bend with r/d = 1
        k_90_1 = 20 * 0.0175  # 20 pipe diameters of equivalent length
        
        # Adjust for different r/d ratio
        if r_d_ratio < 1:
            k_90 = k_90_1 * (1/r_d_ratio)**0.65
        else:
            k_90 = k_90_1 * (1/r_d_ratio)**0.5
        
        # Adjust for different bend angle
        k_factor = k_90 * (self.angle / 90.0)**0.5
        
        return k_factor
    
    def _get_material_roughness(self) -> float:
        """
        Get typical roughness for material.
        
        Returns:
            float: Surface roughness in meters
        """
        # Reuse the method from Pipe class
        temp_pipe = Pipe("temp", 1.0, self.diameter, self.wall_thickness, 
                        self.material, self.schedule)
        return temp_pipe._get_material_roughness()
    
    def _get_material_color(self) -> List[float]:
        """
        Get visualization color for material.
        
        Returns:
            List[float]: RGB color values (0-1)
        """
        # Reuse the method from Pipe class
        temp_pipe = Pipe("temp", 1.0, self.diameter, self.wall_thickness, 
                        self.material, self.schedule)
        return temp_pipe._get_material_color()
    
    def calculate_pressure_drop(self, flow_rate: float, fluid: FluidProperties, 
                               inlet_pressure: float, temperature: float = 293.15) -> float:
        """
        Calculate pressure drop across the bend using k-factor method.
        
        Args:
            flow_rate (float): Flow rate in m³/s
            fluid (FluidProperties): Fluid properties
            inlet_pressure (float): Inlet pressure in Pa
            temperature (float, optional): Fluid temperature in K
            
        Returns:
            float: Pressure drop in Pa
        """
        # Calculate fluid density at given conditions
        density = fluid.density_at_conditions(temperature, inlet_pressure / 1e5)  # Convert Pa to bar
        
        # Calculate flow velocity
        velocity = flow_rate / self.area
        
        # Calculate pressure drop using k-factor
        pressure_drop = self.k_factor * (density * velocity**2 / 2)
        
        return pressure_drop
    
    def get_mesh_parameters(self) -> Dict[str, Any]:
        """
        Get parameters for mesh generation.
        
        Returns:
            Dict[str, Any]: Mesh parameters
        """
        # Parameters for a pipe bend
        mesh_params = {
            'type': 'bend',
            'diameter': self.diameter,
            'radius': self.radius,
            'angle': self.angle,
            'position': self.position,
            'orientation': self.orientation,
            'connections': [conn.name for conn in self.connections],
            'material': self.material.value,
            'roughness': self.roughness
        }
        
        return mesh_params


class Tee(PipelineComponent):
    """Class for T-junctions in pipelines."""
    
    def __init__(self, name: str, main_diameter: float, branch_diameter: float = None,
                 wall_thickness: float = None, 
                 material: Material = Material.CARBON_STEEL,
                 schedule: PipeSchedule = PipeSchedule.SCH40):
        """
        Initialize tee junction.
        
        Args:
            name (str): Tee name
            main_diameter (float): Main pipe inner diameter in meters
            branch_diameter (float, optional): Branch pipe inner diameter in meters
            wall_thickness (float, optional): Wall thickness in meters
            material (Material, optional): Pipe material
            schedule (PipeSchedule, optional): Pipe schedule
        """
        super().__init__(name, ComponentType.TEE, material)
        
        self.main_diameter = main_diameter
        self.branch_diameter = branch_diameter if branch_diameter else main_diameter
        self.schedule = schedule
        
        # Set wall thickness based on schedule if not provided
        if wall_thickness is None:
            # Use the same algorithm as for straight pipe
            temp_pipe = Pipe("temp", 1.0, main_diameter, None, material, schedule)
            self.wall_thickness = temp_pipe.wall_thickness
        else:
            self.wall_thickness = wall_thickness
        
        # Calculate outer diameters
        self.main_outer_diameter = self.main_diameter + 2 * self.wall_thickness
        self.branch_outer_diameter = self.branch_diameter + 2 * self.wall_thickness
        
        # Calculate cross-sectional areas
        self.main_area = math.pi * (self.main_diameter / 2)**2
        self.branch_area = math.pi * (self.branch_diameter / 2)**2
        
        # Connection types
        self.connection_type_main_1 = ConnectionType.WELDED
        self.connection_type_main_2 = ConnectionType.WELDED
        self.connection_type_branch = ConnectionType.WELDED
        
        # Flow direction attributes (to be set by simulation)
        self.is_converging = True  # Flow from branch and main into main
        self.flow_ratio = 0.5  # Branch flow / total flow
        
        # Calculate k-factors for pressure drop
        self.k_factors = self._calculate_k_factors()
        
        # Set typical roughness for material
        self.roughness = self._get_material_roughness()
        
        # Set visualization color based on material
        self.color = self._get_material_color()
    
    def _calculate_k_factors(self) -> Dict[str, float]:
        """
        Calculate resistance coefficients (k-factors) for the tee.
        
        Returns:
            Dict[str, float]: K-factors for different flow paths
        """
        # Calculate area ratio
        area_ratio = self.branch_area / self.main_area
        
        # K-factors based on flow direction and Crane TP-410 (simplified)
        k_factors = {
            # Converging flow (from main and branch to main)
            "main_converging": 0.4,  # K for main flow path
            "branch_converging": 1.0,  # K for branch flow path
            
            # Diverging flow (from main to main and branch)
            "main_diverging": 0.3,  # K for main flow path
            "branch_diverging": 1.0,  # K for branch flow path
        }
        
        # Adjust for area ratio
        if area_ratio < 1.0:
            k_factors["branch_converging"] += 0.5 * (1 - area_ratio)**2
            k_factors["branch_diverging"] += 0.5 * (1 - area_ratio)**2
        
        return k_factors
    
    def _get_material_roughness(self) -> float:
        """
        Get typical roughness for material.
        
        Returns:
            float: Surface roughness in meters
        """
        # Reuse the method from Pipe class
        temp_pipe = Pipe("temp", 1.0, self.main_diameter, self.wall_thickness, 
                        self.material, self.schedule)
        return temp_pipe._get_material_roughness()
    
    def _get_material_color(self) -> List[float]:
        """
        Get visualization color for material.
        
        Returns:
            List[float]: RGB color values (0-1)
        """
        # Reuse the method from Pipe class
        temp_pipe = Pipe("temp", 1.0, self.main_diameter, self.wall_thickness, 
                        self.material, self.schedule)
        return temp_pipe._get_material_color()
    
    def calculate_pressure_drop(self, flow_rates: Dict[str, float], fluid: FluidProperties, 
                               inlet_pressure: float, temperature: float = 293.15) -> Dict[str, float]:
        """
        Calculate pressure drops across different paths of the tee.
        
        Args:
            flow_rates (Dict[str, float]): Flow rates in m³/s for each port
                                          ('main_in', 'main_out', 'branch')
            fluid (FluidProperties): Fluid properties
            inlet_pressure (float): Inlet pressure in Pa
            temperature (float, optional): Fluid temperature in K
            
        Returns:
            Dict[str, float]: Pressure drops in Pa for each path
        """
        # Calculate fluid density at given conditions
        density = fluid.density_at_conditions(temperature, inlet_pressure / 1e5)  # Convert Pa to bar
        
        # Determine flow direction (converging or diverging)
        main_in_flow = flow_rates.get('main_in', 0.0)
        main_out_flow = flow_rates.get('main_out', 0.0)
        branch_flow = flow_rates.get('branch', 0.0)
        
        # Check if converging or diverging
        if branch_flow > 0 and main_out_flow > main_in_flow:
            # Converging flow (branch flow goes into main)
            self.is_converging = True
            
            # Calculate velocities
            main_in_velocity = main_in_flow / self.main_area
            main_out_velocity = main_out_flow / self.main_area
            branch_velocity = branch_flow / self.branch_area
            
            # Calculate pressure drops
            main_dp = self.k_factors["main_converging"] * (density * main_out_velocity**2 / 2)
            branch_dp = self.k_factors["branch_converging"] * (density * main_out_velocity**2 / 2)
            
            return {
                'main': main_dp,
                'branch': branch_dp
            }
        else:
            # Diverging flow (main flow goes into branch)
            self.is_converging = False
            
            # Calculate velocities
            main_in_velocity = main_in_flow / self.main_area
            main_out_velocity = main_out_flow / self.main_area
            branch_velocity = branch_flow / self.branch_area
            
            # Calculate flow ratio (branch flow / total flow)
            self.flow_ratio = abs(branch_flow / main_in_flow) if main_in_flow != 0 else 0.5
            
            # Calculate pressure drops
            main_dp = self.k_factors["main_diverging"] * (density * main_in_velocity**2 / 2)
            
            # Branch pressure drop depends on flow ratio
            branch_k = self.k_factors["branch_diverging"] + 0.3 * self.flow_ratio**2
            branch_dp = branch_k * (density * main_in_velocity**2 / 2)
            
            return {
                'main': main_dp,
                'branch': branch_dp
            }
    
    def get_mesh_parameters(self) -> Dict[str, Any]:
        """
        Get parameters for mesh generation.
        
        Returns:
            Dict[str, Any]: Mesh parameters
        """
        # Parameters for a tee junction
        mesh_params = {
            'type': 'tee',
            'main_diameter': self.main_diameter,
            'branch_diameter': self.branch_diameter,
            'position': self.position,
            'orientation': self.orientation,
            'connections': [conn.name for conn in self.connections],
            'material': self.material.value,
            'roughness': self.roughness
        }
        
        return mesh_params


class Reducer(PipelineComponent):
    """Class for pipe reducers/expanders."""
    
    def __init__(self, name: str, inlet_diameter: float, outlet_diameter: float,
                 length: float = None, wall_thickness: float = None, 
                 material: Material = Material.CARBON_STEEL,
                 schedule: PipeSchedule = PipeSchedule.SCH40,
                 is_concentric: bool = True):
        """
        Initialize pipe reducer/expander.
        
        Args:
            name (str): Reducer name
            inlet_diameter (float): Inlet inner diameter in meters
            outlet_diameter (float): Outlet inner diameter in meters
            length (float, optional): Reducer length in meters
            wall_thickness (float, optional): Wall thickness in meters
            material (Material, optional): Pipe material
            schedule (PipeSchedule, optional): Pipe schedule
            is_concentric (bool, optional): Whether reducer is concentric or eccentric
        """
        super().__init__(name, ComponentType.REDUCER, material)
        
        self.inlet_diameter = inlet_diameter
        self.outlet_diameter = outlet_diameter
        self.is_concentric = is_concentric
        self.schedule = schedule
        
        # Auto-calculate length if not provided (typical length)
        if length is None:
            # Typical reducer length is 1-2 times the diameter difference
            self.length = 1.5 * abs(inlet_diameter - outlet_diameter)
            # Ensure minimum length is at least 50mm
            self.length = max(self.length, 0.05)
        else:
            self.length = length
        
        # Calculate taper angle
        self.taper_angle = math.degrees(math.atan(
            abs(inlet_diameter - outlet_diameter) / (2 * self.length)
        ))
        
        # Set wall thickness based on schedule if not provided
        if wall_thickness is None:
            # Use the same algorithm as for straight pipe, using larger diameter
            temp_pipe = Pipe("temp", 1.0, max(inlet_diameter, outlet_diameter), 
                            None, material, schedule)
            self.wall_thickness = temp_pipe.wall_thickness
        else:
            self.wall_thickness = wall_thickness
        
        # Calculate outer diameters
        self.inlet_outer_diameter = self.inlet_diameter + 2 * self.wall_thickness
        self.outlet_outer_diameter = self.outlet_diameter + 2 * self.wall_thickness
        
        # Calculate cross-sectional areas
        self.inlet_area = math.pi * (self.inlet_diameter / 2)**2
        self.outlet_area = math.pi * (self.outlet_diameter / 2)**2
        
        # Calculate area ratio
        self.area_ratio = self.outlet_area / self.inlet_area
        
        # Connection types
        self.connection_type_inlet = ConnectionType.WELDED
        self.connection_type_outlet = ConnectionType.WELDED
        
        # Calculate k-factor for pressure drop
        self.k_factor = self._calculate_k_factor()
        
        # Set typical roughness for material
        self.roughness = self._get_material_roughness()
        
        # Set visualization color based on material
        self.color = self._get_material_color()
    
    def _calculate_k_factor(self) -> float:
        """
        Calculate resistance coefficient (k-factor) for the reducer.
        
        Returns:
            float: K-factor (dimensionless)
        """
        # K-factor calculation based on area ratio and taper angle
        # Using Crane TP-410 method (simplified)
        
        # Calculate area ratio
        area_ratio = self.area_ratio
        
        if area_ratio < 1.0:
            # Contraction (reducer)
            if self.taper_angle <= 15:
                # Gradual contraction
                k_factor = 0.8 * (1 - area_ratio)
            elif self.taper_angle <= 45:
                # Moderate contraction
                k_factor = 0.5 * (1 - area_ratio)
            else:
                # Abrupt contraction
                k_factor = 0.4 * (1 - area_ratio)
        else:
            # Expansion (diffuser)
            if self.taper_angle <= 7:
                # Gradual expansion
                k_factor = 0.8 * (1 - 1/area_ratio)**2
            elif self.taper_angle <= 15:
                # Moderate expansion
                k_factor = 1.0 * (1 - 1/area_ratio)**2
            else:
                # Abrupt expansion
                k_factor = 1.0 * (1 - 1/area_ratio)**2
        
        return k_factor
    
    def _get_material_roughness(self) -> float:
        """
        Get typical roughness for material.
        
        Returns:
            float: Surface roughness in meters
        """
        # Reuse the method from Pipe class
        temp_pipe = Pipe("temp", 1.0, self.inlet_diameter, self.wall_thickness, 
                        self.material, self.schedule)
        return temp_pipe._get_material_roughness()
    
    def _get_material_color(self) -> List[float]:
        """
        Get visualization color for material.
        
        Returns:
            List[float]: RGB color values (0-1)
        """
        # Reuse the method from Pipe class
        temp_pipe = Pipe("temp", 1.0, self.inlet_diameter, self.wall_thickness, 
                        self.material, self.schedule)
        return temp_pipe._get_material_color()
    
    def calculate_pressure_drop(self, flow_rate: float, fluid: FluidProperties, 
                               inlet_pressure: float, temperature: float = 293.15) -> float:
        """
        Calculate pressure drop across the reducer.
        
        Args:
            flow_rate (float): Flow rate in m³/s
            fluid (FluidProperties): Fluid properties
            inlet_pressure (float): Inlet pressure in Pa
            temperature (float, optional): Fluid temperature in K
            
        Returns:
            float: Pressure drop in Pa
        """
        # Calculate fluid density at given conditions
        density = fluid.density_at_conditions(temperature, inlet_pressure / 1e5)  # Convert Pa to bar
        
        # Calculate flow velocity at inlet
        inlet_velocity = flow_rate / self.inlet_area
        
        # Calculate pressure drop using k-factor
        pressure_drop = self.k_factor * (density * inlet_velocity**2 / 2)
        
        # Handle pressure recovery in diffuser
        if self.area_ratio > 1.0:
            # Theoretical pressure recovery in diffuser (simplified)
            ideal_recovery = density * inlet_velocity**2 * (1 - (1/self.area_ratio)**2) / 2
            
            # Actual recovery based on diffuser efficiency
            efficiency = 0.8 - 0.03 * self.taper_angle  # Approximate diffuser efficiency
            efficiency = max(0.3, min(0.8, efficiency))  # Constrain within reasonable range
            
            actual_recovery = efficiency * ideal_recovery
            
            # Net pressure change is loss minus recovery
            pressure_drop -= actual_recovery
        
        return max(0, pressure_drop)  # Ensure non-negative pressure drop
    
    def get_mesh_parameters(self) -> Dict[str, Any]:
        """
        Get parameters for mesh generation.
        
        Returns:
            Dict[str, Any]: Mesh parameters
        """
        # Parameters for a reducer/expander
        mesh_params = {
            'type': 'reducer',
            'inlet_diameter': self.inlet_diameter,
            'outlet_diameter': self.outlet_diameter,
            'length': self.length,
            'is_concentric': self.is_concentric,
            'position': self.position,
            'orientation': self.orientation,
            'connections': [conn.name for conn in self.connections],
            'material': self.material.value,
            'roughness': self.roughness
        }
        
        return mesh_params


class Valve(PipelineComponent):
    """Class for pipeline valves."""
    
    def __init__(self, name: str, diameter: float, valve_type: ValveType,
                 length: float = None, wall_thickness: float = None, 
                 material: Material = Material.CARBON_STEEL,
                 schedule: PipeSchedule = PipeSchedule.SCH40,
                 cv: float = None):
        """
        Initialize valve.
        
        Args:
            name (str): Valve name
            diameter (float): Valve inner diameter in meters
            valve_type (ValveType): Type of valve
            length (float, optional): Valve length in meters
            wall_thickness (float, optional): Wall thickness in meters
            material (Material, optional): Valve material
            schedule (PipeSchedule, optional): Pipe schedule
            cv (float, optional): Flow coefficient (if known)
        """
        super().__init__(name, ComponentType.VALVE, material)
        
        self.diameter = diameter
        self.valve_type = valve_type
        self.schedule = schedule
        self.cv = cv  # Flow coefficient (if provided)
        
        # Auto-calculate length if not provided (typical length)
        if length is None:
            # Typical valve length is 1-3 times the diameter
            length_factors = {
                ValveType.GATE: 1.0,
                ValveType.BALL: 1.5,
                ValveType.GLOBE: 1.8,
                ValveType.BUTTERFLY: 0.5,
                ValveType.CHECK: 1.5,
                ValveType.NEEDLE: 2.0,
                ValveType.PLUG: 1.5,
                ValveType.DIAPHRAGM: 1.5,
                ValveType.CONTROL: 2.0,
                ValveType.RELIEF: 2.0,
                ValveType.CUSTOM: 1.5
            }
            self.length = length_factors.get(valve_type, 1.5) * diameter
        else:
            self.length = length
        
        # Set wall thickness based on schedule if not provided
        if wall_thickness is None:
            # Use the same algorithm as for straight pipe
            temp_pipe = Pipe("temp", 1.0, diameter, None, material, schedule)
            self.wall_thickness = temp_pipe.wall_thickness
        else:
            self.wall_thickness = wall_thickness
        
        # Calculate outer diameter
        self.outer_diameter = self.diameter + 2 * self.wall_thickness
        
        # Calculate cross-sectional area
        self.area = math.pi * (self.diameter / 2)**2
        
        # Connection types
        self.connection_type_inlet = ConnectionType.FLANGED
        self.connection_type_outlet = ConnectionType.FLANGED
        
        # Valve state
        self.opening_percentage = 100.0  # 0-100%
        self.is_open = True
        
        # Calculate k-factor for pressure drop
        self.k_factor = self._calculate_k_factor()
        
        # Set typical roughness for material
        self.roughness = self._get_material_roughness()
        
        # Set visualization color based on valve type
        self.color = self._get_valve_color()
    
    def _calculate_k_factor(self) -> float:
        """
        Calculate resistance coefficient (k-factor) for the valve when fully open.
        
        Returns:
            float: K-factor (dimensionless)
        """
        # Typical k-factor values for different valve types when fully open
        k_factors = {
            ValveType.GATE: 0.2,
            ValveType.BALL: 0.1,
            ValveType.GLOBE: 10.0,
            ValveType.BUTTERFLY: 0.6,
            ValveType.CHECK: 2.0,
            ValveType.NEEDLE: 5.0,
            ValveType.PLUG: 0.8,
            ValveType.DIAPHRAGM: 2.3,
            ValveType.CONTROL: 3.0,
            ValveType.RELIEF: 2.5,
            ValveType.CUSTOM: 1.0
        }
        
        return k_factors.get(self.valve_type, 1.0)
    
    def _get_material_roughness(self) -> float:
        """
        Get typical roughness for material.
        
        Returns:
            float: Surface roughness in meters
        """
        # Reuse the method from Pipe class
        temp_pipe = Pipe("temp", 1.0, self.diameter, self.wall_thickness, 
                        self.material, self.schedule)
        return temp_pipe._get_material_roughness()
    
    def _get_valve_color(self) -> List[float]:
        """
        Get visualization color for valve type.
        
        Returns:
            List[float]: RGB color values (0-1)
        """
        # Define colors for various valve types
        color_map = {
            ValveType.GATE: [0.8, 0.2, 0.2],  # Red
            ValveType.BALL: [0.2, 0.6, 0.9],  # Blue
            ValveType.GLOBE: [0.9, 0.6, 0.2],  # Orange
            ValveType.BUTTERFLY: [0.6, 0.8, 0.2],  # Green-yellow
            ValveType.CHECK: [0.7, 0.3, 0.7],  # Purple
            ValveType.NEEDLE: [0.5, 0.5, 0.8],  # Blue-gray
            ValveType.PLUG: [0.7, 0.7, 0.2],  # Yellow
            ValveType.DIAPHRAGM: [0.4, 0.7, 0.4],  # Green
            ValveType.CONTROL: [0.3, 0.6, 0.8],  # Blue
            ValveType.RELIEF: [1.0, 0.4, 0.4],  # Light red
            ValveType.CUSTOM: [0.5, 0.5, 0.5]   # Gray
        }
        
        return color_map.get(self.valve_type, [0.5, 0.5, 0.5])
    
    def set_opening(self, percentage: float):
        """
        Set valve opening percentage.
        
        Args:
            percentage (float): Opening percentage (0-100%)
            
        Raises:
            ValueError: If percentage is outside the valid range
        """
        if not 0 <= percentage <= 100:
            raise ValueError("Opening percentage must be between 0 and 100")
        
        self.opening_percentage = percentage
        self.is_open = percentage > 0
    
    def calculate_cv_from_k(self) -> float:
        """
        Calculate flow coefficient (Cv) from k-factor.
        
        Returns:
            float: Flow coefficient in US gallons/min
        """
        # Convert k-factor to Cv using formula: Cv = 29.9 * d² / √k
        # Where d is in inches and Cv is in US gallons/min at 1 psi pressure drop
        
        # Convert diameter to inches
        d_inch = self.diameter * 39.3701
        
        # Calculate Cv
        cv = 29.9 * d_inch**2 / math.sqrt(self.k_factor) if self.k_factor > 0 else float('inf')
        
        return cv
    
    def calculate_k_from_cv(self, cv: float) -> float:
        """
        Calculate k-factor from flow coefficient (Cv).
        
        Args:
            cv (float): Flow coefficient in US gallons/min
            
        Returns:
            float: K-factor (dimensionless)
        """
        # Convert Cv to k-factor using formula: k = (29.9 * d²)² / Cv²
        # Where d is in inches and Cv is in US gallons/min at 1 psi pressure drop
        
        # Convert diameter to inches
        d_inch = self.diameter * 39.3701
        
        # Calculate k-factor
        k = (29.9 * d_inch**2)**2 / cv**2 if cv > 0 else float('inf')
        
        return k
    
    def calculate_pressure_drop(self, flow_rate: float, fluid: FluidProperties, 
                               inlet_pressure: float, temperature: float = 293.15) -> float:
        """
        Calculate pressure drop across the valve.
        
        Args:
            flow_rate (float): Flow rate in m³/s
            fluid (FluidProperties): Fluid properties
            inlet_pressure (float): Inlet pressure in Pa
            temperature (float, optional): Fluid temperature in K
            
        Returns:
            float: Pressure drop in Pa
        """
        # Return infinite pressure drop if valve is closed
        if not self.is_open:
            return float('inf')
        
        # Calculate fluid density at given conditions
        density = fluid.density_at_conditions(temperature, inlet_pressure / 1e5)  # Convert Pa to bar
        
        # Calculate flow velocity
        velocity = flow_rate / self.area
        
        # Get base k-factor
        base_k = self.k_factor
        
        # Apply valve opening correction
        if self.opening_percentage < 100:
            # This is a simplified model, different valve types have different
            # flow characteristic curves
            
            # Determine valve characteristics based on type
            if self.valve_type in [ValveType.GATE, ValveType.BALL, ValveType.PLUG]:
                # Equal percentage characteristic
                opening_fraction = self.opening_percentage / 100.0
                # Modified k-factor (increases exponentially as valve closes)
                modified_k = base_k + base_k * (10.0 * (1.0 - opening_fraction))**2.5
            elif self.valve_type in [ValveType.GLOBE, ValveType.NEEDLE, ValveType.CONTROL]:
                # Linear characteristic
                opening_fraction = self.opening_percentage / 100.0
                # Modified k-factor (increases linearly as valve closes)
                modified_k = base_k + base_k * (5.0 / opening_fraction - 5.0)
            elif self.valve_type == ValveType.BUTTERFLY:
                # Modified k-factor based on empirical butterfly valve data
                opening_fraction = self.opening_percentage / 100.0
                if opening_fraction < 0.1:
                    modified_k = base_k * 1000
                else:
                    # Rough approximation
                    modified_k = base_k + base_k * (1.2 / opening_fraction - 1.2)
            else:
                # Default behavior (quick opening characteristic)
                opening_fraction = self.opening_percentage / 100.0
                modified_k = base_k * (1.0 / opening_fraction**2)
            
            # Ensure k-factor is within reasonable bounds
            modified_k = min(modified_k, 1e6)  # Upper limit
            
            # Use modified k-factor
            k_factor = modified_k
        else:
            # Fully open valve
            k_factor = base_k
        
        # Calculate pressure drop using k-factor
        pressure_drop = k_factor * (density * velocity**2 / 2)
        
        return pressure_drop
    
    def get_mesh_parameters(self) -> Dict[str, Any]:
        """
        Get parameters for mesh generation.
        
        Returns:
            Dict[str, Any]: Mesh parameters
        """
        # Parameters for a valve
        mesh_params = {
            'type': 'valve',
            'valve_type': self.valve_type.value,
            'diameter': self.diameter,
            'length': self.length,
            'opening_percentage': self.opening_percentage,
            'position': self.position,
            'orientation': self.orientation,
            'connections': [conn.name for conn in self.connections],
            'material': self.material.value,
        }
        
        return mesh_params


class Riser(PipelineComponent):
    """Class for pipeline risers (vertical sections of pipe)."""
    
    def __init__(self, name: str, height: float, diameter: float, 
                 wall_thickness: float = None, 
                 material: Material = Material.CARBON_STEEL,
                 schedule: PipeSchedule = PipeSchedule.SCH40,
                 is_flexible: bool = False):
        """
        Initialize pipeline riser.
        
        Args:
            name (str): Riser name
            height (float): Riser height in meters
            diameter (float): Pipe inner diameter in meters
            wall_thickness (float, optional): Wall thickness in meters
            material (Material, optional): Pipe material
            schedule (PipeSchedule, optional): Pipe schedule
            is_flexible (bool, optional): Whether riser is flexible
        """
        super().__init__(name, ComponentType.RISER, material)
        
        self.height = height
        self.diameter = diameter
        self.is_flexible = is_flexible
        self.schedule = schedule
        
        # Orientation (risers are vertical)
        self.orientation = [0.0, 0.0, 90.0]  # Vertical orientation
        
        # Set wall thickness based on schedule if not provided
        if wall_thickness is None:
            # Use the same algorithm as for straight pipe
            temp_pipe = Pipe("temp", 1.0, diameter, None, material, schedule)
            self.wall_thickness = temp_pipe.wall_thickness
        else:
            self.wall_thickness = wall_thickness
        
        # Calculate outer diameter
        self.outer_diameter = self.diameter + 2 * self.wall_thickness
        
        # Calculate cross-sectional properties
        self.area = math.pi * (self.diameter / 2)**2  # Flow area
        self.perimeter = math.pi * self.diameter  # Wetted perimeter
        
        # Connection types at each end
        self.connection_type_bottom = ConnectionType.WELDED
        self.connection_type_top = ConnectionType.WELDED
        
        # Riser-specific properties
        self.riser_shape = "Straight"  # Default, could be "Catenary", "Lazy-S", etc.
        self.top_tension = 0.0  # Top tension for tensioned risers
        
        # Set typical roughness for material
        self.roughness = self._get_material_roughness()
        
        # Set visualization color based on material
        self.color = self._get_material_color()
    
    def _get_material_roughness(self) -> float:
        """
        Get typical roughness for material.
        
        Returns:
            float: Surface roughness in meters
        """
        # Reuse the method from Pipe class
        temp_pipe = Pipe("temp", 1.0, self.diameter, self.wall_thickness, 
                        self.material, self.schedule)
        return temp_pipe._get_material_roughness()
    
    def _get_material_color(self) -> List[float]:
        """
        Get visualization color for material.
        
        Returns:
            List[float]: RGB color values (0-1)
        """
        # Reuse the method from Pipe class
        temp_pipe = Pipe("temp", 1.0, self.diameter, self.wall_thickness, 
                        self.material, self.schedule)
        return temp_pipe._get_material_color()
    
    def calculate_pressure_drop(self, flow_rate: float, fluid: FluidProperties, 
                               inlet_pressure: float, temperature: float = 293.15, 
                               flow_direction_up: bool = True) -> float:
        """
        Calculate pressure drop/gain across the riser.
        
        Args:
            flow_rate (float): Flow rate in m³/s
            fluid (FluidProperties): Fluid properties
            inlet_pressure (float): Inlet pressure in Pa
            temperature (float, optional): Fluid temperature in K
            flow_direction_up (bool, optional): Whether flow is upward
            
        Returns:
            float: Pressure drop in Pa (negative for pressure gain)
        """
        # Calculate fluid properties at given conditions
        density = fluid.density_at_conditions(temperature, inlet_pressure / 1e5)  # Convert Pa to bar
        viscosity = fluid.viscosity_at_conditions(temperature, inlet_pressure / 1e5)
        
        # Calculate flow velocity
        velocity = flow_rate / self.area
        
        # Calculate Reynolds number
        reynolds = density * velocity * self.diameter / viscosity
        
        # Calculate friction factor using Colebrook equation approximation
        if reynolds < 2300:
            # Laminar flow
            friction_factor = 64 / reynolds
        else:
            # Turbulent flow - Haaland equation (approximation of Colebrook)
            term1 = -1.8 * math.log10(6.9/reynolds + (self.roughness/self.diameter/3.7)**1.11)
            friction_factor = 1 / (term1**2)
        
        # Frictional pressure drop
        frictional_dp = friction_factor * (self.height / self.diameter) * (density * velocity**2 / 2)
        
        # Hydrostatic pressure difference
        gravity = 9.81  # m/s²
        hydrostatic_dp = density * gravity * self.height
        
        # Total pressure drop (positive for pressure drop, negative for pressure gain)
        if flow_direction_up:
            # Upward flow: friction loss + hydrostatic head
            pressure_drop = frictional_dp + hydrostatic_dp
        else:
            # Downward flow: friction loss - hydrostatic head
            pressure_drop = frictional_dp - hydrostatic_dp
        
        return pressure_drop
    
    def get_mesh_parameters(self) -> Dict[str, Any]:
        """
        Get parameters for mesh generation.
        
        Returns:
            Dict[str, Any]: Mesh parameters
        """
        # Parameters for a riser
        mesh_params = {
            'type': 'riser',
            'height': self.height,
            'diameter': self.diameter,
            'position': self.position,
            'orientation': self.orientation,
            'is_flexible': self.is_flexible,
            'riser_shape': self.riser_shape,
            'connections': [conn.name for conn in self.connections],
            'material': self.material.value,
            'roughness': self.roughness
        }
        
        return mesh_params


class Instrument(PipelineComponent):
    """Class for pipeline instruments such as meters, gauges, and transmitters."""
    
    def __init__(self, name: str, instrument_type: InstrumentType, diameter: float = None,
                 material: Material = Material.STAINLESS_STEEL):
        """
        Initialize pipeline instrument.
        
        Args:
            name (str): Instrument name
            instrument_type (InstrumentType): Type of instrument
            diameter (float, optional): Connection diameter in meters
            material (Material, optional): Instrument material
        """
        super().__init__(name, ComponentType.SENSOR, material)
        
        self.instrument_type = instrument_type
        self.diameter = diameter
        
        # Instrument-specific properties
        self.range_min = 0.0
        self.range_max = 100.0
        self.units = ""
        self.accuracy = 0.01  # 1% accuracy
        
        # Measured value (to be updated by simulation)
        self.current_value = 0.0
        
        # Set appropriate units and ranges based on instrument type
        self._set_instrument_defaults()
        
        # Connection types
        if self.diameter:
            # In-line instrument
            self.is_inline = True
            self.connection_type = ConnectionType.FLANGED
            
            # Calculate cross-sectional area
            self.area = math.pi * (self.diameter / 2)**2
            
            # Set k-factor for pressure drop
            self.k_factor = self._calculate_k_factor()
        else:
            # Offline instrument (e.g., pressure gauge)
            self.is_inline = False
            self.connection_type = ConnectionType.THREADED
        
        # Set visualization color based on instrument type
        self.color = self._get_instrument_color()
    
    def _set_instrument_defaults(self):
        """Set default properties based on instrument type."""
        if self.instrument_type == InstrumentType.FLOW_METER:
            self.units = "m³/h"
            self.range_min = 0.0
            self.range_max = 1000.0
        elif self.instrument_type == InstrumentType.PRESSURE_TRANSMITTER or \
             self.instrument_type == InstrumentType.PRESSURE_GAUGE:
            self.units = "bar"
            self.range_min = 0.0
            self.range_max = 100.0
        elif self.instrument_type == InstrumentType.TEMPERATURE_TRANSMITTER or \
             self.instrument_type == InstrumentType.TEMPERATURE_GAUGE:
            self.units = "°C"
            self.range_min = 0.0
            self.range_max = 200.0
        elif self.instrument_type == InstrumentType.LEVEL_TRANSMITTER:
            self.units = "m"
            self.range_min = 0.0
            self.range_max = 10.0
        elif self.instrument_type == InstrumentType.DENSITY_METER:
            self.units = "kg/m³"
            self.range_min = 500.0
            self.range_max = 2000.0
        elif self.instrument_type == InstrumentType.WATER_CUT_METER:
            self.units = "%"
            self.range_min = 0.0
            self.range_max = 100.0
    
    def _calculate_k_factor(self) -> float:
        """
        Calculate resistance coefficient (k-factor) for the instrument.
        
        Returns:
            float: K-factor (dimensionless)
        """
        # Typical k-factor values for different instrument types
        k_factors = {
            InstrumentType.FLOW_METER: 0.5,        # Typical for full-bore flowmeter
            InstrumentType.PRESSURE_TRANSMITTER: 0.0,  # Typically negligible
            InstrumentType.TEMPERATURE_TRANSMITTER: 0.0,  # Typically negligible
            InstrumentType.LEVEL_TRANSMITTER: 0.0,  # Typically negligible
            InstrumentType.DENSITY_METER: 0.3,     # Depends on design
            InstrumentType.WATER_CUT_METER: 0.3,   # Depends on design
            InstrumentType.PRESSURE_GAUGE: 0.0,    # Typically negligible
            InstrumentType.TEMPERATURE_GAUGE: 0.0,  # Typically negligible
            InstrumentType.CUSTOM: 0.2             # Default for unknown types
        }
        
        return k_factors.get(self.instrument_type, 0.2)
    
    def _get_instrument_color(self) -> List[float]:
        """
        Get visualization color for instrument type.
        
        Returns:
            List[float]: RGB color values (0-1)
        """
        # Define colors for various instrument types
        color_map = {
            InstrumentType.FLOW_METER: [0.2, 0.6, 0.8],  # Blue
            InstrumentType.PRESSURE_TRANSMITTER: [0.8, 0.4, 0.4],  # Red
            InstrumentType.TEMPERATURE_TRANSMITTER: [0.8, 0.6, 0.0],  # Orange
            InstrumentType.LEVEL_TRANSMITTER: [0.4, 0.8, 0.4],  # Green
            InstrumentType.DENSITY_METER: [0.6, 0.4, 0.8],  # Purple
            InstrumentType.WATER_CUT_METER: [0.0, 0.6, 0.6],  # Teal
            InstrumentType.PRESSURE_GAUGE: [0.7, 0.3, 0.3],  # Light red
            InstrumentType.TEMPERATURE_GAUGE: [0.7, 0.5, 0.0],  # Light orange
            InstrumentType.CUSTOM: [0.5, 0.5, 0.5]  # Gray
        }
        
        return color_map.get(self.instrument_type, [0.5, 0.5, 0.5])
    
    def calculate_pressure_drop(self, flow_rate: float, fluid: FluidProperties, 
                               inlet_pressure: float, temperature: float = 293.15) -> float:
        """
        Calculate pressure drop across the instrument.
        
        Args:
            flow_rate (float): Flow rate in m³/s
            fluid (FluidProperties): Fluid properties
            inlet_pressure (float): Inlet pressure in Pa
            temperature (float, optional): Fluid temperature in K
            
        Returns:
            float: Pressure drop in Pa
        """
        # If not inline, there's no pressure drop
        if not self.is_inline:
            return 0.0
        
        # Calculate fluid density at given conditions
        density = fluid.density_at_conditions(temperature, inlet_pressure / 1e5)  # Convert Pa to bar
        
        # Calculate flow velocity
        velocity = flow_rate / self.area
        
        # Calculate pressure drop using k-factor
        pressure_drop = self.k_factor * (density * velocity**2 / 2)
        
        return pressure_drop
    
    def get_mesh_parameters(self) -> Dict[str, Any]:
        """
        Get parameters for mesh generation.
        
        Returns:
            Dict[str, Any]: Mesh parameters
        """
        # Parameters for an instrument
        mesh_params = {
            'type': 'instrument',
            'instrument_type': self.instrument_type.value,
            'is_inline': self.is_inline,
            'diameter': self.diameter,
            'position': self.position,
            'orientation': self.orientation,
            'connections': [conn.name for conn in self.connections],
            'material': self.material.value,
        }
        
        return mesh_params
    
    def set_measurement_range(self, min_value: float, max_value: float, units: str = None):
        """
        Set the measurement range for the instrument.
        
        Args:
            min_value (float): Minimum range value
            max_value (float): Maximum range value
            units (str, optional): Units of measurement
            
        Raises:
            ValueError: If max value is not greater than min value
        """
        if max_value <= min_value:
            raise ValueError("Maximum value must be greater than minimum value")
        
        self.range_min = min_value
        self.range_max = max_value
        
        if units:
            self.units = units
    
    def read_value(self) -> Tuple[float, str]:
        """
        Read the current value from the instrument.
        
        Returns:
            Tuple[float, str]: Current value and units
        """
        return self.current_value, self.units
    
    def update_value(self, value: float):
        """
        Update the current value of the instrument.
        
        Args:
            value (float): New value
        """
        # Constrain value to range
        self.current_value = max(self.range_min, min(self.range_max, value))


class PigLauncher(PipelineComponent):
    """Class for pipeline pig launchers."""
    
    def __init__(self, name: str, pipeline_diameter: float, 
                 barrel_diameter: float = None, 
                 barrel_length: float = None,
                 material: Material = Material.CARBON_STEEL,
                 schedule: PipeSchedule = PipeSchedule.SCH40):
        """
        Initialize pig launcher.
        
        Args:
            name (str): Launcher name
            pipeline_diameter (float): Pipeline inner diameter in meters
            barrel_diameter (float, optional): Barrel inner diameter in meters
            barrel_length (float, optional): Barrel length in meters
            material (Material, optional): Material
            schedule (PipeSchedule, optional): Pipe schedule
        """
        super().__init__(name, ComponentType.LAUNCHER, material)
        
        self.pipeline_diameter = pipeline_diameter
        self.schedule = schedule
        
        # Set barrel diameter if not provided (typically 1.5x pipeline diameter)
        if barrel_diameter is None:
            self.barrel_diameter = pipeline_diameter * 1.5
        else:
            self.barrel_diameter = barrel_diameter
        
        # Set barrel length if not provided (typically 3-5x pipeline diameter)
        if barrel_length is None:
            self.barrel_length = pipeline_diameter * 4.0
        else:
            self.barrel_length = barrel_length
        
        # Calculate wall thickness based on schedule
        temp_pipe = Pipe("temp", 1.0, self.pipeline_diameter, None, material, schedule)
        self.wall_thickness = temp_pipe.wall_thickness
        
        # Calculate cross-sectional area of pipeline
        self.pipeline_area = math.pi * (self.pipeline_diameter / 2)**2
        
        # Calculate cross-sectional area of barrel
        self.barrel_area = math.pi * (self.barrel_diameter / 2)**2
        
        # Calculate k-factor for pressure drop
        self.k_factor = self._calculate_k_factor()
        
        # Launcher-specific properties
        self.has_kicker_line = True
        self.has_pressure_indicator = True
        self.has_drain_valve = True
        self.has_vent_valve = True
        
        # Set visualization color
        self.color = [0.2, 0.7, 0.2]  # Green color for launchers
    
    def _calculate_k_factor(self) -> float:
        """
        Calculate resistance coefficient (k-factor) for the launcher.
        
        Returns:
            float: K-factor (dimensionless)
        """
        # Simplified k-factor based on sudden expansion and contraction
        # Area ratio between barrel and pipeline
        area_ratio = self.pipeline_area / self.barrel_area
        
        # Combined effect of contraction from barrel to pipeline
        k_factor = 0.5 * (1 - area_ratio)
        
        return k_factor
    
    def calculate_pressure_drop(self, flow_rate: float, fluid: FluidProperties, 
                               inlet_pressure: float, temperature: float = 293.15) -> float:
        """
        Calculate pressure drop across the launcher.
        
        Args:
            flow_rate (float): Flow rate in m³/s
            fluid (FluidProperties): Fluid properties
            inlet_pressure (float): Inlet pressure in Pa
            temperature (float, optional): Fluid temperature in K
            
        Returns:
            float: Pressure drop in Pa
        """
        # Calculate fluid density at given conditions
        density = fluid.density_at_conditions(temperature, inlet_pressure / 1e5)  # Convert Pa to bar
        
        # Calculate flow velocity in pipeline
        velocity = flow_rate / self.pipeline_area
        
        # Calculate pressure drop using k-factor
        pressure_drop = self.k_factor * (density * velocity**2 / 2)
        
        return pressure_drop
    
    def get_mesh_parameters(self) -> Dict[str, Any]:
        """
        Get parameters for mesh generation.
        
        Returns:
            Dict[str, Any]: Mesh parameters
        """
        # Parameters for a pig launcher
        mesh_params = {
            'type': 'launcher',
            'pipeline_diameter': self.pipeline_diameter,
            'barrel_diameter': self.barrel_diameter,
            'barrel_length': self.barrel_length,
            'position': self.position,
            'orientation': self.orientation,
            'connections': [conn.name for conn in self.connections],
            'material': self.material.value,
            'has_kicker_line': self.has_kicker_line,
            'has_pressure_indicator': self.has_pressure_indicator,
            'has_drain_valve': self.has_drain_valve,
            'has_vent_valve': self.has_vent_valve
        }
        
        return mesh_params


class PigReceiver(PipelineComponent):
    """Class for pipeline pig receivers."""
    
    def __init__(self, name: str, pipeline_diameter: float, 
                 barrel_diameter: float = None, 
                 barrel_length: float = None,
                 material: Material = Material.CARBON_STEEL,
                 schedule: PipeSchedule = PipeSchedule.SCH40):
        """
        Initialize pig receiver.
        
        Args:
            name (str): Receiver name
            pipeline_diameter (float): Pipeline inner diameter in meters
            barrel_diameter (float, optional): Barrel inner diameter in meters
            barrel_length (float, optional): Barrel length in meters
            material (Material, optional): Material
            schedule (PipeSchedule, optional): Pipe schedule
        """
        super().__init__(name, ComponentType.RECEIVER, material)
        
        self.pipeline_diameter = pipeline_diameter
        self.schedule = schedule
        
        # Set barrel diameter if not provided (typically 1.5x pipeline diameter)
        if barrel_diameter is None:
            self.barrel_diameter = pipeline_diameter * 1.5
        else:
            self.barrel_diameter = barrel_diameter
        
        # Set barrel length if not provided (typically 3-5x pipeline diameter)
        if barrel_length is None:
            self.barrel_length = pipeline_diameter * 4.0
        else:
            self.barrel_length = barrel_length
        
        # Calculate wall thickness based on schedule
        temp_pipe = Pipe("temp", 1.0, self.pipeline_diameter, None, material, schedule)
        self.wall_thickness = temp_pipe.wall_thickness
        
        # Calculate cross-sectional area of pipeline
        self.pipeline_area = math.pi * (self.pipeline_diameter / 2)**2
        
        # Calculate cross-sectional area of barrel
        self.barrel_area = math.pi * (self.barrel_diameter / 2)**2
        
        # Calculate k-factor for pressure drop
        self.k_factor = self._calculate_k_factor()
        
        # Receiver-specific properties
        self.has_bypass_line = True
        self.has_pressure_indicator = True
        self.has_drain_valve = True
        self.has_vent_valve = True
        
        # Set visualization color
        self.color = [0.7, 0.2, 0.2]  # Red color for receivers
    
    def _calculate_k_factor(self) -> float:
        """
        Calculate resistance coefficient (k-factor) for the receiver.
        
        Returns:
            float: K-factor (dimensionless)
        """
        # Simplified k-factor based on sudden expansion and contraction
        # Area ratio between pipeline and barrel
        area_ratio = self.pipeline_area / self.barrel_area
        
        # Effect of expansion from pipeline to barrel
        k_factor = (1 - area_ratio)**2
        
        return k_factor
    
    def calculate_pressure_drop(self, flow_rate: float, fluid: FluidProperties, 
                               inlet_pressure: float, temperature: float = 293.15) -> float:
        """
        Calculate pressure drop across the receiver.
        
        Args:
            flow_rate (float): Flow rate in m³/s
            fluid (FluidProperties): Fluid properties
            inlet_pressure (float): Inlet pressure in Pa
            temperature (float, optional): Fluid temperature in K
            
        Returns:
            float: Pressure drop in Pa
        """
        # Calculate fluid density at given conditions
        density = fluid.density_at_conditions(temperature, inlet_pressure / 1e5)  # Convert Pa to bar
        
        # Calculate flow velocity in pipeline
        velocity = flow_rate / self.pipeline_area
        
        # Calculate pressure drop using k-factor
        pressure_drop = self.k_factor * (density * velocity**2 / 2)
        
        return pressure_drop
    
    def get_mesh_parameters(self) -> Dict[str, Any]:
        """
        Get parameters for mesh generation.
        
        Returns:
            Dict[str, Any]: Mesh parameters
        """
        # Parameters for a pig receiver
        mesh_params = {
            'type': 'receiver',
            'pipeline_diameter': self.pipeline_diameter,
            'barrel_diameter': self.barrel_diameter,
            'barrel_length': self.barrel_length,
            'position': self.position,
            'orientation': self.orientation,
            'connections': [conn.name for conn in self.connections],
            'material': self.material.value,
            'has_bypass_line': self.has_bypass_line,
            'has_pressure_indicator': self.has_pressure_indicator,
            'has_drain_valve': self.has_drain_valve,
            'has_vent_valve': self.has_vent_valve
        }
        
        return mesh_params


# Factory function to create pipeline components
def create_component(component_type: str, name: str, **kwargs) -> PipelineComponent:
    """
    Factory function to create pipeline components.
    
    Args:
        component_type (str): Type of component
        name (str): Component name
        **kwargs: Additional parameters
        
    Returns:
        PipelineComponent: Created component
        
    Raises:
        ValueError: If component type is not recognized
    """
    component_type = component_type.lower()
    
    if component_type == "pipe":
        # Required parameters
        length = kwargs.get("length")
        diameter = kwargs.get("diameter")
        
        if length is None or diameter is None:
            raise ValueError("Pipe requires length and diameter parameters")
        
        # Optional parameters
        wall_thickness = kwargs.get("wall_thickness")
        material = kwargs.get("material", Material.CARBON_STEEL)
        schedule = kwargs.get("schedule", PipeSchedule.SCH40)
        
        return Pipe(name, length, diameter, wall_thickness, material, schedule)
    
    elif component_type == "bend" or component_type == "elbow":
        # Required parameters
        diameter = kwargs.get("diameter")
        radius = kwargs.get("radius")
        
        if diameter is None or radius is None:
            raise ValueError("Bend requires diameter and radius parameters")
        
        # Optional parameters
        angle = kwargs.get("angle", 90.0)
        wall_thickness = kwargs.get("wall_thickness")
        material = kwargs.get("material", Material.CARBON_STEEL)
        schedule = kwargs.get("schedule", PipeSchedule.SCH40)
        
        return PipeBend(name, diameter, radius, angle, wall_thickness, material, schedule)
    
    elif component_type == "tee":
        # Required parameters
        main_diameter = kwargs.get("main_diameter")
        
        if main_diameter is None:
            raise ValueError("Tee requires main_diameter parameter")
        
        # Optional parameters
        branch_diameter = kwargs.get("branch_diameter")
        wall_thickness = kwargs.get("wall_thickness")
        material = kwargs.get("material", Material.CARBON_STEEL)
        schedule = kwargs.get("schedule", PipeSchedule.SCH40)
        
        return Tee(name, main_diameter, branch_diameter, wall_thickness, material, schedule)
    
    elif component_type == "reducer":
        # Required parameters
        inlet_diameter = kwargs.get("inlet_diameter")
        outlet_diameter = kwargs.get("outlet_diameter")
        
        if inlet_diameter is None or outlet_diameter is None:
            raise ValueError("Reducer requires inlet_diameter and outlet_diameter parameters")
        
        # Optional parameters
        length = kwargs.get("length")
        wall_thickness = kwargs.get("wall_thickness")
        material = kwargs.get("material", Material.CARBON_STEEL)
        schedule = kwargs.get("schedule", PipeSchedule.SCH40)
        is_concentric = kwargs.get("is_concentric", True)
        
        return Reducer(name, inlet_diameter, outlet_diameter, length, 
                      wall_thickness, material, schedule, is_concentric)
    
    elif component_type == "valve":
        # Required parameters
        diameter = kwargs.get("diameter")
        
        if diameter is None:
            raise ValueError("Valve requires diameter parameter")
        
        # Optional parameters
        valve_type = kwargs.get("valve_type", ValveType.GATE)
        length = kwargs.get("length")
        wall_thickness = kwargs.get("wall_thickness")
        material = kwargs.get("material", Material.CARBON_STEEL)
        schedule = kwargs.get("schedule", PipeSchedule.SCH40)
        cv = kwargs.get("cv")
        
        return Valve(name, diameter, valve_type, length, wall_thickness, material, schedule, cv)
    
    elif component_type == "riser":
        # Required parameters
        height = kwargs.get("height")
        diameter = kwargs.get("diameter")
        
        if height is None or diameter is None:
            raise ValueError("Riser requires height and diameter parameters")
        
        # Optional parameters
        wall_thickness = kwargs.get("wall_thickness")
        material = kwargs.get("material", Material.CARBON_STEEL)
        schedule = kwargs.get("schedule", PipeSchedule.SCH40)
        is_flexible = kwargs.get("is_flexible", False)
        
        return Riser(name, height, diameter, wall_thickness, material, schedule, is_flexible)
    
    elif component_type == "instrument":
        # Required parameters
        instrument_type = kwargs.get("instrument_type", InstrumentType.FLOW_METER)
        
        # Optional parameters
        diameter = kwargs.get("diameter")  # None for non-inline instruments
        material = kwargs.get("material", Material.STAINLESS_STEEL)
        
        return Instrument(name, instrument_type, diameter, material)
    
    elif component_type == "launcher" or component_type == "pig_launcher":
        # Required parameters
        pipeline_diameter = kwargs.get("pipeline_diameter")
        
        if pipeline_diameter is None:
            raise ValueError("Launcher requires pipeline_diameter parameter")
        
        # Optional parameters
        barrel_diameter = kwargs.get("barrel_diameter")
        barrel_length = kwargs.get("barrel_length")
        material = kwargs.get("material", Material.CARBON_STEEL)
        schedule = kwargs.get("schedule", PipeSchedule.SCH40)
        
        return PigLauncher(name, pipeline_diameter, barrel_diameter, barrel_length, 
                          material, schedule)
    
    elif component_type == "receiver" or component_type == "pig_receiver":
        # Required parameters
        pipeline_diameter = kwargs.get("pipeline_diameter")
        
        if pipeline_diameter is None:
            raise ValueError("Receiver requires pipeline_diameter parameter")
        
        # Optional parameters
        barrel_diameter = kwargs.get("barrel_diameter")
        barrel_length = kwargs.get("barrel_length")
        material = kwargs.get("material", Material.CARBON_STEEL)
        schedule = kwargs.get("schedule", PipeSchedule.SCH40)
        
        return PigReceiver(name, pipeline_diameter, barrel_diameter, barrel_length, 
                          material, schedule)
    
    else:
        raise ValueError(f"Unrecognized component type: {component_type}")


# Utility functions for hydraulic calculations
def calculate_total_pressure_drop(components: List[PipelineComponent], 
                                flow_rate: float, 
                                fluid: FluidProperties,
                                inlet_pressure: float,
                                temperature: float = 293.15) -> float:
    """
    Calculate total pressure drop across a series of components.
    
    Args:
        components (List[PipelineComponent]): List of components in order
        flow_rate (float): Flow rate in m³/s
        fluid (FluidProperties): Fluid properties
        inlet_pressure (float): Inlet pressure in Pa
        temperature (float, optional): Fluid temperature in K
        
    Returns:
        float: Total pressure drop in Pa
    """
    total_dp = 0.0
    current_pressure = inlet_pressure
    
    for component in components:
        # Calculate pressure drop for this component
        if isinstance(component, Tee):
            # Special handling for tees
            # Assuming flow goes from main_in through main_out (ignoring branch)
            flow_rates = {
                'main_in': flow_rate,
                'main_out': flow_rate,
                'branch': 0.0
            }
            dp_dict = component.calculate_pressure_drop(flow_rates, fluid, current_pressure, temperature)
            dp = dp_dict['main']
        elif isinstance(component, Riser):
            # Assume flow direction is upward for risers
            dp = component.calculate_pressure_drop(flow_rate, fluid, current_pressure, temperature, True)
        else:
            # Standard pressure drop calculation
            dp = component.calculate_pressure_drop(flow_rate, fluid, current_pressure, temperature)
        
        # Add to total
        total_dp += dp
        
        # Update current pressure for next component
        current_pressure -= dp
    
    return total_dp


def calculate_equivalent_length(components: List[PipelineComponent], 
                              reference_diameter: float) -> float:
    """
    Calculate equivalent length of a pipeline segment.
    
    Args:
        components (List[PipelineComponent]): List of components
        reference_diameter (float): Reference diameter in meters
        
    Returns:
        float: Equivalent length in meters
    """
    total_length = 0.0
    
    for component in components:
        if isinstance(component, Pipe):
            # Straight pipe
            total_length += component.length
        elif isinstance(component, PipeBend):
            # Convert bend k-factor to equivalent length
            # L_eq = k * D / f, where f is friction factor (assume 0.02 for turbulent flow)
            total_length += (component.k_factor * reference_diameter / 0.02)
        elif hasattr(component, 'k_factor'):
            # Convert component k-factor to equivalent length
            total_length += (component.k_factor * reference_diameter / 0.02)
        
    return total_length


def calculate_pipeline_volume(components: List[PipelineComponent]) -> float:
    """
    Calculate internal volume of a pipeline segment.
    
    Args:
        components (List[PipelineComponent]): List of components
        
    Returns:
        float: Volume in cubic meters
    """
    total_volume = 0.0
    
    for component in components:
        if isinstance(component, Pipe):
            # Straight pipe
            volume = component.area * component.length
            total_volume += volume
        elif isinstance(component, PipeBend):
            # Bend volume
            volume = component.area * component.length
            total_volume += volume
        elif isinstance(component, Tee):
            # Approximate tee volume as main pipe volume plus branch stub
            main_volume = component.main_area * component.main_diameter * 2  # 2D length
            branch_volume = component.branch_area * component.branch_diameter * 0.5  # 0.5D stub
            total_volume += (main_volume + branch_volume)
        elif isinstance(component, Reducer):
            # Approximate reducer volume as average area times length
            avg_area = (component.inlet_area + component.outlet_area) / 2
            volume = avg_area * component.length
            total_volume += volume
        elif isinstance(component, Valve):
            # Approximate valve volume
            volume = component.area * component.length
            total_volume += volume
        elif isinstance(component, Riser):
            # Riser volume
            volume = component.area * component.height
            total_volume += volume
        elif isinstance(component, (PigLauncher, PigReceiver)):
            # Launcher/receiver volume
            pipeline_volume = component.pipeline_area * component.pipeline_diameter * 2  # 2D length
            barrel_volume = component.barrel_area * component.barrel_length
            total_volume += (pipeline_volume + barrel_volume)
        
    return total_volume