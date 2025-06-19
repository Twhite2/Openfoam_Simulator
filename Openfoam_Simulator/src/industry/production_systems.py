#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production systems module for Openfoam_Simulator.

This module provides classes and utilities for modeling oil & gas production systems,
including:
- Wellheads and Christmas trees
- Separators (2-phase and 3-phase)
- Compressors and pumps
- Pressure vessels
- Process equipment
- Manifolds and headers
- Subsea equipment
- Complete production systems

These components can be used to create realistic production facility models
for CFD simulations in the Openfoam_Simulator application.
"""

import os
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

# Import other modules from the package
from ..utils.logger import get_logger
from ..config import get_value

logger = get_logger(__name__)


class ComponentType(Enum):
    """Enum for different types of production system components."""
    WELLHEAD = 1
    CHRISTMAS_TREE = 2
    SEPARATOR_2PHASE = 3
    SEPARATOR_3PHASE = 4
    COMPRESSOR = 5
    PUMP = 6
    PRESSURE_VESSEL = 7
    HEAT_EXCHANGER = 8
    MANIFOLD = 9
    HEADER = 10
    PIPELINE = 11
    VALVE = 12
    SUBSEA_TREE = 13
    SUBSEA_MANIFOLD = 14
    PROCESS_VESSEL = 15
    SLUG_CATCHER = 16
    RISER = 17
    PLATFORM = 18


@dataclass
class FluidProperties:
    """Class to store fluid properties for production system components."""
    name: str
    density: float  # kg/m³
    viscosity: float  # Pa·s
    specific_heat: float  # J/(kg·K)
    thermal_conductivity: float  # W/(m·K)
    compressibility: float = 0.0  # 1/Pa
    pour_point: float = 273.15  # K
    bubble_point: float = 373.15  # K
    wax_content: float = 0.0  # fraction
    asphaltene_content: float = 0.0  # fraction
    h2s_content: float = 0.0  # ppm
    co2_content: float = 0.0  # fraction
    
    @classmethod
    def crude_oil(cls, api_gravity: float = 35.0, temperature: float = 293.15) -> 'FluidProperties':
        """
        Create crude oil properties based on API gravity.
        
        Args:
            api_gravity (float): Oil API gravity
            temperature (float): Temperature in Kelvin
            
        Returns:
            FluidProperties: Crude oil properties
        """
        # Calculate density from API gravity
        density = 141.5 / (api_gravity + 131.5) * 1000  # kg/m³
        
        # Estimate viscosity (simplified model based on API and temperature)
        # Uses a simplified correlation for demonstration
        log_visc = 4.3 - 0.03 * api_gravity - 0.006 * (temperature - 273.15)
        viscosity = 10 ** log_visc / 1000  # Pa·s
        
        # Other properties correlated with API gravity
        specific_heat = 1800 + 3 * api_gravity  # J/(kg·K)
        thermal_conductivity = 0.12 + 0.0003 * api_gravity  # W/(m·K)
        compressibility = (1.5e-9) * (1 + 0.001 * api_gravity)  # 1/Pa
        
        # Pour point decreases with API gravity
        pour_point = 273.15 - 0.5 * api_gravity  # K
        
        # Bubble point increases with API gravity
        bubble_point = 373.15 + api_gravity  # K
        
        # Wax and asphaltene content decreases with API gravity
        wax_content = max(0, 0.15 - 0.003 * api_gravity)
        asphaltene_content = max(0, 0.08 - 0.002 * api_gravity)
        
        return cls(
            name=f"Crude Oil (API {api_gravity})",
            density=density,
            viscosity=viscosity,
            specific_heat=specific_heat,
            thermal_conductivity=thermal_conductivity,
            compressibility=compressibility,
            pour_point=pour_point,
            bubble_point=bubble_point,
            wax_content=wax_content,
            asphaltene_content=asphaltene_content,
            h2s_content=0.0,
            co2_content=0.01
        )
    
    @classmethod
    def natural_gas(cls, specific_gravity: float = 0.6, temperature: float = 293.15) -> 'FluidProperties':
        """
        Create natural gas properties based on specific gravity.
        
        Args:
            specific_gravity (float): Gas specific gravity (relative to air)
            temperature (float): Temperature in Kelvin
            
        Returns:
            FluidProperties: Natural gas properties
        """
        # Calculate density from specific gravity
        density = specific_gravity * 1.225  # kg/m³
        
        # Estimate viscosity (simplified model)
        viscosity = 1.1e-5 + 0.1e-5 * specific_gravity  # Pa·s
        
        # Other properties correlated with specific gravity
        specific_heat = 2200 - 200 * specific_gravity  # J/(kg·K)
        thermal_conductivity = 0.024 + 0.004 * specific_gravity  # W/(m·K)
        compressibility = 4.8e-7  # 1/Pa
        
        return cls(
            name=f"Natural Gas (SG {specific_gravity})",
            density=density,
            viscosity=viscosity,
            specific_heat=specific_heat,
            thermal_conductivity=thermal_conductivity,
            compressibility=compressibility,
            pour_point=100.0,  # K (unrealistic but for completeness)
            bubble_point=115.0,  # K
            wax_content=0.0,
            asphaltene_content=0.0,
            h2s_content=50.0,  # ppm
            co2_content=0.02
        )
    
    @classmethod
    def water(cls, salinity: float = 35000.0, temperature: float = 293.15) -> 'FluidProperties':
        """
        Create water properties based on salinity (ppm).
        
        Args:
            salinity (float): Water salinity in ppm
            temperature (float): Temperature in Kelvin
            
        Returns:
            FluidProperties: Water properties
        """
        # Calculate density adjustment for salinity
        salinity_fraction = salinity / 1e6
        density = 1000.0 + 800.0 * salinity_fraction  # kg/m³
        
        # Temperature effect on viscosity (simplified model)
        viscosity = 0.001 * (1.0 + 0.5 * salinity_fraction) * math.exp(-(temperature - 293.15) / 30)  # Pa·s
        
        return cls(
            name=f"Water (Salinity {salinity} ppm)",
            density=density,
            viscosity=viscosity,
            specific_heat=4200.0 - 500.0 * salinity_fraction,  # J/(kg·K)
            thermal_conductivity=0.6 - 0.1 * salinity_fraction,  # W/(m·K)
            compressibility=4.5e-10,  # 1/Pa
            pour_point=273.15 - 2.0 * salinity_fraction,  # K
            bubble_point=373.15 + 2.0 * salinity_fraction,  # K
            wax_content=0.0,
            asphaltene_content=0.0,
            h2s_content=0.0,
            co2_content=0.0
        )


class ProductionComponent:
    """Base class for all production system components."""
    
    def __init__(self, name: str, component_type: ComponentType, position: Tuple[float, float, float] = (0, 0, 0)):
        """
        Initialize a production component.
        
        Args:
            name (str): Component name
            component_type (ComponentType): Component type
            position (Tuple[float, float, float]): 3D position (x, y, z) in meters
        """
        self.name = name
        self.component_type = component_type
        self.position = position
        self.connected_components = []  # List of connected components
        self.properties = {}  # Dictionary of component-specific properties
        
        # Geometric properties
        self.dimensions = (1.0, 1.0, 1.0)  # Default dimensions in meters
        self.orientation = (0.0, 0.0, 0.0)  # Rotation angles in radians
        
        # Operational properties
        self.pressure = 101325.0  # Pa (default: atmospheric)
        self.temperature = 293.15  # K (default: 20°C)
        self.flow_rate = 0.0  # m³/s
        
        # Material properties
        self.material = "Steel"
        self.wall_thickness = 0.01  # m
        
        # Logging
        logger.info(f"Created {component_type.name} component: {name}")
    
    def connect(self, component: 'ProductionComponent', bidirectional: bool = True) -> None:
        """
        Connect this component to another component.
        
        Args:
            component (ProductionComponent): The component to connect to
            bidirectional (bool): If True, also connect the other component to this one
        """
        if component not in self.connected_components:
            self.connected_components.append(component)
            logger.debug(f"Connected {self.name} to {component.name}")
            
            if bidirectional and self not in component.connected_components:
                component.connect(self, False)
    
    def disconnect(self, component: 'ProductionComponent', bidirectional: bool = True) -> None:
        """
        Disconnect this component from another component.
        
        Args:
            component (ProductionComponent): The component to disconnect from
            bidirectional (bool): If True, also disconnect the other component from this one
        """
        if component in self.connected_components:
            self.connected_components.remove(component)
            logger.debug(f"Disconnected {self.name} from {component.name}")
            
            if bidirectional and self in component.connected_components:
                component.disconnect(self, False)
    
    def set_property(self, key: str, value: Any) -> None:
        """
        Set a component property.
        
        Args:
            key (str): Property name
            value (Any): Property value
        """
        self.properties[key] = value
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """
        Get a component property.
        
        Args:
            key (str): Property name
            default (Any): Default value if property doesn't exist
            
        Returns:
            Any: Property value or default
        """
        return self.properties.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert component to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Component as dictionary
        """
        return {
            "name": self.name,
            "type": self.component_type.name,
            "position": self.position,
            "dimensions": self.dimensions,
            "orientation": self.orientation,
            "pressure": self.pressure,
            "temperature": self.temperature,
            "flow_rate": self.flow_rate,
            "material": self.material,
            "wall_thickness": self.wall_thickness,
            "properties": self.properties,
            "connected_to": [comp.name for comp in self.connected_components]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], component_lookup: Dict[str, 'ProductionComponent'] = None) -> 'ProductionComponent':
        """
        Create component from dictionary.
        
        Args:
            data (Dict[str, Any]): Component data
            component_lookup (Dict[str, ProductionComponent]): Dictionary of existing components
            
        Returns:
            ProductionComponent: New component
        """
        # Create appropriate component type
        component_type = ComponentType[data["type"]]
        if component_type == ComponentType.SEPARATOR_2PHASE:
            component = TwoPhaseSeparator(data["name"], position=tuple(data["position"]))
        elif component_type == ComponentType.SEPARATOR_3PHASE:
            component = ThreePhaseSeparator(data["name"], position=tuple(data["position"]))
        elif component_type == ComponentType.WELLHEAD:
            component = Wellhead(data["name"], position=tuple(data["position"]))
        elif component_type == ComponentType.CHRISTMAS_TREE:
            component = ChristmasTree(data["name"], position=tuple(data["position"]))
        elif component_type == ComponentType.COMPRESSOR:
            component = Compressor(data["name"], position=tuple(data["position"]))
        elif component_type == ComponentType.PUMP:
            component = Pump(data["name"], position=tuple(data["position"]))
        elif component_type == ComponentType.MANIFOLD:
            component = Manifold(data["name"], position=tuple(data["position"]))
        elif component_type == ComponentType.SLUG_CATCHER:
            component = SlugCatcher(data["name"], position=tuple(data["position"]))
        else:
            # Generic component
            component = ProductionComponent(data["name"], component_type, tuple(data["position"]))
        
        # Set properties
        component.dimensions = tuple(data.get("dimensions", (1.0, 1.0, 1.0)))
        component.orientation = tuple(data.get("orientation", (0.0, 0.0, 0.0)))
        component.pressure = data.get("pressure", 101325.0)
        component.temperature = data.get("temperature", 293.15)
        component.flow_rate = data.get("flow_rate", 0.0)
        component.material = data.get("material", "Steel")
        component.wall_thickness = data.get("wall_thickness", 0.01)
        component.properties = data.get("properties", {})
        
        # Connect components if lookup is provided
        if component_lookup is not None:
            for connected_name in data.get("connected_to", []):
                if connected_name in component_lookup:
                    component.connect(component_lookup[connected_name])
        
        return component
    
    def get_mesh_path(self) -> Optional[str]:
        """
        Get path to 3D mesh for this component.
        
        Returns:
            Optional[str]: Path to mesh file or None if not available
        """
        # Check if there's a mesh directory in templates
        templates_dir = get_value('paths.templates')
        if not templates_dir:
            return None
        
        mesh_dir = os.path.join(templates_dir, "mesh_templates", "components")
        if not os.path.exists(mesh_dir):
            return None
        
        # Look for mesh file matching component type
        component_name = self.component_type.name.lower()
        for ext in [".stl", ".obj", ".vtk"]:
            mesh_path = os.path.join(mesh_dir, f"{component_name}{ext}")
            if os.path.exists(mesh_path):
                return mesh_path
        
        return None
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.component_type.name}: {self.name} at {self.position}"


class Wellhead(ProductionComponent):
    """Wellhead component for oil and gas wells."""
    
    def __init__(self, name: str, position: Tuple[float, float, float] = (0, 0, 0),
                well_type: str = "Oil", well_depth: float = 3000.0):
        """
        Initialize a wellhead component.
        
        Args:
            name (str): Component name
            position (Tuple[float, float, float]): 3D position
            well_type (str): Type of well ("Oil", "Gas", "Water", "Injection")
            well_depth (float): Well depth in meters
        """
        super().__init__(name, ComponentType.WELLHEAD, position)
        
        # Specific wellhead properties
        self.properties["well_type"] = well_type
        self.properties["well_depth"] = well_depth
        self.properties["flowing"] = False
        self.properties["wellhead_pressure"] = 5000000.0  # 5 MPa
        self.properties["bottomhole_pressure"] = 20000000.0  # 20 MPa
        self.properties["temperature"] = 348.15  # 75°C
        
        # Default geometry for wellhead
        self.dimensions = (0.5, 0.5, 1.2)  # Typical wellhead size
    
    def start_flow(self, flow_rate: float) -> None:
        """
        Start flow from wellhead.
        
        Args:
            flow_rate (float): Flow rate in m³/s
        """
        self.properties["flowing"] = True
        self.flow_rate = flow_rate
        logger.info(f"Wellhead {self.name} started flowing at {flow_rate} m³/s")
    
    def stop_flow(self) -> None:
        """Stop flow from wellhead."""
        self.properties["flowing"] = False
        self.flow_rate = 0.0
        logger.info(f"Wellhead {self.name} stopped flowing")
    
    def set_well_type(self, well_type: str) -> None:
        """
        Set the well type.
        
        Args:
            well_type (str): Type of well ("Oil", "Gas", "Water", "Injection")
        """
        valid_types = ["Oil", "Gas", "Water", "Injection"]
        if well_type not in valid_types:
            logger.warning(f"Invalid well type: {well_type}. Using 'Oil' instead.")
            well_type = "Oil"
        
        self.properties["well_type"] = well_type


class ChristmasTree(ProductionComponent):
    """Christmas tree component for wellhead flow control."""
    
    def __init__(self, name: str, position: Tuple[float, float, float] = (0, 0, 0),
                is_subsea: bool = False):
        """
        Initialize a Christmas tree component.
        
        Args:
            name (str): Component name
            position (Tuple[float, float, float]): 3D position
            is_subsea (bool): Whether this is a subsea Christmas tree
        """
        super().__init__(name, ComponentType.CHRISTMAS_TREE, position)
        
        # Specific christmas tree properties
        self.properties["is_subsea"] = is_subsea
        self.properties["master_valve_open"] = True
        self.properties["wing_valve_open"] = True
        self.properties["choke_position"] = 100.0  # Percent open
        self.properties["swab_valve_open"] = False
        self.properties["kill_valve_open"] = False
        
        # Default geometry
        self.dimensions = (1.0, 1.0, 2.5)  # Typical X-mas tree size
    
    def set_choke_position(self, position: float) -> None:
        """
        Set the choke valve position.
        
        Args:
            position (float): Valve position (0-100%)
        """
        position = max(0.0, min(100.0, position))  # Clamp to 0-100%
        self.properties["choke_position"] = position
        
        # Calculate flow rate if connected to a wellhead
        for component in self.connected_components:
            if component.component_type == ComponentType.WELLHEAD and component.properties.get("flowing", False):
                original_flow = component.flow_rate
                # Simplified relationship between choke position and flow
                new_flow = original_flow * (position / 100.0) ** 1.5
                self.flow_rate = new_flow
                logger.info(f"Christmas tree {self.name} choke set to {position}%, flow rate: {new_flow} m³/s")
                return
        
        logger.debug(f"Christmas tree {self.name} choke set to {position}%")
    
    def emergency_shutdown(self) -> None:
        """Perform emergency shutdown by closing all valves."""
        self.properties["master_valve_open"] = False
        self.properties["wing_valve_open"] = False
        self.properties["choke_position"] = 0.0
        self.flow_rate = 0.0
        logger.warning(f"Emergency shutdown performed on Christmas tree {self.name}")


class Separator(ProductionComponent):
    """Base class for separator vessels."""
    
    def __init__(self, name: str, component_type: ComponentType, position: Tuple[float, float, float] = (0, 0, 0),
                vessel_volume: float = 10.0, orientation: str = "horizontal"):
        """
        Initialize a separator component.
        
        Args:
            name (str): Component name
            component_type (ComponentType): Component type
            position (Tuple[float, float, float]): 3D position
            vessel_volume (float): Vessel volume in m³
            orientation (str): Vessel orientation ("horizontal" or "vertical")
        """
        super().__init__(name, component_type, position)
        
        # Validate orientation
        if orientation not in ["horizontal", "vertical"]:
            logger.warning(f"Invalid orientation: {orientation}. Using 'horizontal' instead.")
            orientation = "horizontal"
        
        # Common separator properties
        self.properties["vessel_volume"] = vessel_volume
        self.properties["orientation"] = orientation
        self.properties["liquid_level"] = 50.0  # Percent
        self.properties["design_pressure"] = 1000000.0  # 1 MPa
        self.properties["mist_extractor"] = True
        self.properties["heating_enabled"] = False
        self.properties["heating_temperature"] = 323.15  # 50°C
        
        # Calculate dimensions based on volume and orientation
        if orientation == "horizontal":
            # Horizontal vessel: cylindrical with L=3D
            diameter = (4 * vessel_volume / (3 * math.pi)) ** (1/3)
            length = 3 * diameter
            self.dimensions = (length, diameter, diameter)
        else:
            # Vertical vessel: cylindrical with H=4D
            diameter = (vessel_volume / (math.pi)) ** (1/3)
            height = 4 * diameter
            self.dimensions = (diameter, diameter, height)
    
    def set_liquid_level(self, level: float) -> None:
        """
        Set the liquid level in the separator.
        
        Args:
            level (float): Liquid level (0-100%)
        """
        level = max(0.0, min(100.0, level))  # Clamp to 0-100%
        self.properties["liquid_level"] = level
        logger.debug(f"Separator {self.name} liquid level set to {level}%")
        
        # Check for high/low level alarms
        if level > 90.0:
            logger.warning(f"Separator {self.name} high level alarm: {level}%")
        elif level < 10.0:
            logger.warning(f"Separator {self.name} low level alarm: {level}%")
    
    def enable_heating(self, temperature: float) -> None:
        """
        Enable separator heating.
        
        Args:
            temperature (float): Heating temperature in Kelvin
        """
        self.properties["heating_enabled"] = True
        self.properties["heating_temperature"] = temperature
        logger.info(f"Separator {self.name} heating enabled at {temperature} K")
    
    def disable_heating(self) -> None:
        """Disable separator heating."""
        self.properties["heating_enabled"] = False
        logger.info(f"Separator {self.name} heating disabled")
    
    def calculate_residence_time(self, flow_rate: float) -> float:
        """
        Calculate the liquid residence time in the separator.
        
        Args:
            flow_rate (float): Liquid flow rate in m³/s
            
        Returns:
            float: Residence time in seconds
        """
        if flow_rate <= 0.0:
            return float('inf')
        
        # Calculate liquid volume
        vessel_volume = self.properties["vessel_volume"]
        liquid_level = self.properties["liquid_level"] / 100.0
        liquid_volume = vessel_volume * liquid_level
        
        # Calculate residence time
        residence_time = liquid_volume / flow_rate
        
        return residence_time


class TwoPhaseSeparator(Separator):
    """Two-phase separator for gas-liquid separation."""
    
    def __init__(self, name: str, position: Tuple[float, float, float] = (0, 0, 0),
                vessel_volume: float = 10.0, orientation: str = "horizontal"):
        """
        Initialize a two-phase separator.
        
        Args:
            name (str): Component name
            position (Tuple[float, float, float]): 3D position
            vessel_volume (float): Vessel volume in m³
            orientation (str): Vessel orientation ("horizontal" or "vertical")
        """
        super().__init__(name, ComponentType.SEPARATOR_2PHASE, position, vessel_volume, orientation)
        
        # Two-phase specific properties
        self.properties["gas_capacity"] = 50.0  # m³/s
        self.properties["liquid_capacity"] = 0.1  # m³/s
        self.properties["separation_efficiency"] = 95.0  # Percent
        self.properties["gas_outlet_pressure"] = 800000.0  # 0.8 MPa
        self.properties["liquid_outlet_pressure"] = 900000.0  # 0.9 MPa
    
    def calculate_separation_efficiency(self, droplet_size: float) -> float:
        """
        Calculate separation efficiency based on droplet size.
        
        Args:
            droplet_size (float): Liquid droplet size in microns
            
        Returns:
            float: Separation efficiency (0-100%)
        """
        # Simplified model: efficiency increases with droplet size
        if droplet_size < 10.0:
            return 60.0 + 2.0 * droplet_size
        elif droplet_size < 100.0:
            return 80.0 + 0.15 * droplet_size
        else:
            return 95.0 + 0.03 * min(droplet_size - 100.0, 100.0)


class ThreePhaseSeparator(Separator):
    """Three-phase separator for gas-oil-water separation."""
    
    def __init__(self, name: str, position: Tuple[float, float, float] = (0, 0, 0),
                vessel_volume: float = 15.0, orientation: str = "horizontal"):
        """
        Initialize a three-phase separator.
        
        Args:
            name (str): Component name
            position (Tuple[float, float, float]): 3D position
            vessel_volume (float): Vessel volume in m³
            orientation (str): Vessel orientation (always "horizontal" for 3-phase)
        """
        # Three-phase separators are generally horizontal
        if orientation != "horizontal":
            logger.warning("Three-phase separators are typically horizontal. Overriding orientation.")
            orientation = "horizontal"
            
        super().__init__(name, ComponentType.SEPARATOR_3PHASE, position, vessel_volume, orientation)
        
        # Three-phase specific properties
        self.properties["gas_capacity"] = 40.0  # m³/s
        self.properties["oil_capacity"] = 0.08  # m³/s
        self.properties["water_capacity"] = 0.05  # m³/s
        self.properties["oil_water_interface_level"] = 40.0  # Percent of liquid height
        self.properties["separation_efficiency"] = 90.0  # Percent
        self.properties["gas_outlet_pressure"] = 800000.0  # 0.8 MPa
        self.properties["oil_outlet_pressure"] = 850000.0  # 0.85 MPa
        self.properties["water_outlet_pressure"] = 900000.0  # 0.9 MPa
        self.properties["interface_controller_enabled"] = True
        self.properties["weir_installed"] = True
    
    def set_oil_water_interface(self, level: float) -> None:
        """
        Set the oil-water interface level.
        
        Args:
            level (float): Interface level (0-100% of liquid height)
        """
        level = max(10.0, min(90.0, level))  # Keep within reasonable bounds
        self.properties["oil_water_interface_level"] = level
        logger.debug(f"Separator {self.name} oil-water interface set to {level}%")
        
        # Check for interface control issues
        if level < 20.0:
            logger.warning(f"Separator {self.name} oil-water interface too low: {level}%")
        elif level > 80.0:
            logger.warning(f"Separator {self.name} oil-water interface too high: {level}%")
    
    def calculate_water_oil_ratio(self) -> float:
        """
        Calculate water-oil ratio based on interface level.
        
        Returns:
            float: Water-oil ratio
        """
        # Simplified model: interface level correlates with water-oil ratio
        interface_level = self.properties["oil_water_interface_level"]
        liquid_level = self.properties["liquid_level"]
        
        if liquid_level <= 0.0:
            return 0.0
        
        # Water volume as a percentage of total liquid volume
        water_percentage = interface_level / liquid_level * 100.0
        
        # Convert to water-oil ratio
        if water_percentage >= 100.0:
            return float('inf')  # All water
        
        water_oil_ratio = water_percentage / (100.0 - water_percentage)
        return water_oil_ratio


class Compressor(ProductionComponent):
    """Gas compressor component."""
    
    def __init__(self, name: str, position: Tuple[float, float, float] = (0, 0, 0),
                type_: str = "Centrifugal", power: float = 500.0):
        """
        Initialize a compressor component.
        
        Args:
            name (str): Component name
            position (Tuple[float, float, float]): 3D position
            type_ (str): Compressor type ("Centrifugal", "Reciprocating", "Screw")
            power (float): Power rating in kW
        """
        super().__init__(name, ComponentType.COMPRESSOR, position)
        
        # Validate type
        valid_types = ["Centrifugal", "Reciprocating", "Screw"]
        if type_ not in valid_types:
            logger.warning(f"Invalid compressor type: {type_}. Using 'Centrifugal' instead.")
            type_ = "Centrifugal"
        
        # Compressor properties
        self.properties["type"] = type_
        self.properties["power"] = power  # kW
        self.properties["running"] = False
        self.properties["speed"] = 0.0  # RPM
        self.properties["max_speed"] = 3600.0  # RPM
        self.properties["suction_pressure"] = 200000.0  # 0.2 MPa
        self.properties["discharge_pressure"] = 1000000.0  # 1.0 MPa
        self.properties["capacity"] = 10.0  # m³/s
        self.properties["efficiency"] = 85.0  # Percent
        
        # Set dimensions based on power
        volume = 0.1 + power / 100.0  # Approximate size based on power
        self.dimensions = (volume ** (1/3) * 2, volume ** (1/3) * 1.5, volume ** (1/3))
    
    def start(self, speed: float = None) -> None:
        """
        Start the compressor.
        
        Args:
            speed (float, optional): Operating speed in RPM
        """
        self.properties["running"] = True
        
        if speed is None:
            speed = self.properties["max_speed"]
        
        speed = min(speed, self.properties["max_speed"])
        self.properties["speed"] = speed
        
        # Calculate flow rate based on speed
        max_capacity = self.properties["capacity"]
        speed_ratio = speed / self.properties["max_speed"]
        self.flow_rate = max_capacity * speed_ratio
        
        logger.info(f"Compressor {self.name} started at {speed} RPM, flow rate: {self.flow_rate} m³/s")
    
    def stop(self) -> None:
        """Stop the compressor."""
        self.properties["running"] = False
        self.properties["speed"] = 0.0
        self.flow_rate = 0.0
        logger.info(f"Compressor {self.name} stopped")
    
    def calculate_power_consumption(self) -> float:
        """
        Calculate actual power consumption.
        
        Returns:
            float: Power consumption in kW
        """
        if not self.properties["running"]:
            return 0.0
        
        # Simplified power calculation based on speed and pressures
        rated_power = self.properties["power"]
        speed_ratio = self.properties["speed"] / self.properties["max_speed"]
        pressure_ratio = (self.properties["discharge_pressure"] / 
                          self.properties["suction_pressure"])
        
        # Power is proportional to speed³ and affects pressure ratio
        power = rated_power * (speed_ratio ** 3) * math.log(pressure_ratio)
        efficiency = self.properties["efficiency"] / 100.0
        
        return power / efficiency


class Pump(ProductionComponent):
    """Liquid pump component."""
    
    def __init__(self, name: str, position: Tuple[float, float, float] = (0, 0, 0),
                type_: str = "Centrifugal", power: float = 100.0):
        """
        Initialize a pump component.
        
        Args:
            name (str): Component name
            position (Tuple[float, float, float]): 3D position
            type_ (str): Pump type ("Centrifugal", "Positive Displacement", "Submersible")
            power (float): Power rating in kW
        """
        super().__init__(name, ComponentType.PUMP, position)
        
        # Validate type
        valid_types = ["Centrifugal", "Positive Displacement", "Submersible"]
        if type_ not in valid_types:
            logger.warning(f"Invalid pump type: {type_}. Using 'Centrifugal' instead.")
            type_ = "Centrifugal"
        
        # Pump properties
        self.properties["type"] = type_
        self.properties["power"] = power  # kW
        self.properties["running"] = False
        self.properties["speed"] = 0.0  # RPM
        self.properties["max_speed"] = 3600.0  # RPM
        self.properties["suction_pressure"] = 200000.0  # 0.2 MPa
        self.properties["discharge_pressure"] = 1000000.0  # 1.0 MPa
        self.properties["capacity"] = 0.1  # m³/s
        self.properties["head"] = 100.0  # m
        self.properties["efficiency"] = 80.0  # Percent
        self.properties["npsh_required"] = 3.0  # m
        
        # Set dimensions based on power
        volume = 0.02 + power / 500.0  # Approximate size based on power
        self.dimensions = (volume ** (1/3) * 2, volume ** (1/3) * 1, volume ** (1/3))
    
    def start(self, speed: float = None) -> None:
        """
        Start the pump.
        
        Args:
            speed (float, optional): Operating speed in RPM
        """
        self.properties["running"] = True
        
        if speed is None:
            speed = self.properties["max_speed"]
        
        speed = min(speed, self.properties["max_speed"])
        self.properties["speed"] = speed
        
        # Calculate flow rate based on speed (affinity laws)
        max_capacity = self.properties["capacity"]
        speed_ratio = speed / self.properties["max_speed"]
        self.flow_rate = max_capacity * speed_ratio
        
        # Calculate head based on speed (affinity laws)
        max_head = self.properties["head"]
        self.properties["current_head"] = max_head * (speed_ratio ** 2)
        
        logger.info(f"Pump {self.name} started at {speed} RPM, flow rate: {self.flow_rate} m³/s")
    
    def stop(self) -> None:
        """Stop the pump."""
        self.properties["running"] = False
        self.properties["speed"] = 0.0
        self.flow_rate = 0.0
        self.properties["current_head"] = 0.0
        logger.info(f"Pump {self.name} stopped")
    
    def calculate_npsh_available(self, fluid_vapor_pressure: float) -> float:
        """
        Calculate available NPSH (Net Positive Suction Head).
        
        Args:
            fluid_vapor_pressure (float): Fluid vapor pressure in Pa
            
        Returns:
            float: NPSH available in meters
        """
        # NPSH = (P_suction - P_vapor) / (ρ * g) + Z_suction
        suction_pressure = self.properties["suction_pressure"]
        fluid_density = 1000.0  # kg/m³ (assuming water as default)
        gravity = 9.81  # m/s²
        suction_height = 0.0  # m (assuming pump at same level as suction)
        
        npsh_available = ((suction_pressure - fluid_vapor_pressure) / 
                          (fluid_density * gravity)) + suction_height
        
        return npsh_available
    
    def check_cavitation_risk(self, fluid_vapor_pressure: float) -> bool:
        """
        Check if pump is at risk of cavitation.
        
        Args:
            fluid_vapor_pressure (float): Fluid vapor pressure in Pa
            
        Returns:
            bool: True if cavitation risk exists
        """
        npsh_available = self.calculate_npsh_available(fluid_vapor_pressure)
        npsh_required = self.properties["npsh_required"]
        
        # Add safety margin
        margin = 1.3
        
        if npsh_available < (npsh_required * margin):
            logger.warning(f"Pump {self.name} cavitation risk: NPSH_a = {npsh_available:.2f} m, " +
                          f"NPSH_r = {npsh_required:.2f} m")
            return True
        
        return False


class Manifold(ProductionComponent):
    """Manifold component for combining or distributing flows."""
    
    def __init__(self, name: str, position: Tuple[float, float, float] = (0, 0, 0),
                num_inlets: int = 4, num_outlets: int = 1):
        """
        Initialize a manifold component.
        
        Args:
            name (str): Component name
            position (Tuple[float, float, float]): 3D position
            num_inlets (int): Number of inlet connections
            num_outlets (int): Number of outlet connections
        """
        super().__init__(name, ComponentType.MANIFOLD, position)
        
        # Validate numbers
        num_inlets = max(1, min(16, num_inlets))
        num_outlets = max(1, min(8, num_outlets))
        
        # Manifold properties
        self.properties["num_inlets"] = num_inlets
        self.properties["num_outlets"] = num_outlets
        self.properties["inlet_valves_open"] = [True] * num_inlets
        self.properties["outlet_valves_open"] = [True] * num_outlets
        self.properties["design_pressure"] = 15000000.0  # 15 MPa
        self.properties["balanced_flow"] = True
        
        # Set dimensions based on number of connections
        width = 0.5 + (max(num_inlets, num_outlets) / 4) * 0.5
        self.dimensions = (width, width * 0.8, 0.4)
    
    def set_inlet_valve(self, index: int, open_state: bool) -> None:
        """
        Set the state of an inlet valve.
        
        Args:
            index (int): Valve index (0-based)
            open_state (bool): Whether valve is open
        """
        if 0 <= index < len(self.properties["inlet_valves_open"]):
            self.properties["inlet_valves_open"][index] = open_state
            logger.debug(f"Manifold {self.name} inlet valve {index} set to {'open' if open_state else 'closed'}")
            self._update_flow_rates()
        else:
            logger.warning(f"Invalid inlet valve index: {index}")
    
    def set_outlet_valve(self, index: int, open_state: bool) -> None:
        """
        Set the state of an outlet valve.
        
        Args:
            index (int): Valve index (0-based)
            open_state (bool): Whether valve is open
        """
        if 0 <= index < len(self.properties["outlet_valves_open"]):
            self.properties["outlet_valves_open"][index] = open_state
            logger.debug(f"Manifold {self.name} outlet valve {index} set to {'open' if open_state else 'closed'}")
            self._update_flow_rates()
        else:
            logger.warning(f"Invalid outlet valve index: {index}")
    
    def _update_flow_rates(self) -> None:
        """Update flow rates based on valve states."""
        # Count open inlets and outlets
        open_inlets = sum(self.properties["inlet_valves_open"])
        open_outlets = sum(self.properties["outlet_valves_open"])
        
        if open_inlets == 0 or open_outlets == 0:
            self.flow_rate = 0.0
            return
        
        # Get total input flow
        input_flow = 0.0
        for comp in self.connected_components:
            input_flow += getattr(comp, "flow_rate", 0.0)
        
        # Distribute flow to outlets
        self.flow_rate = input_flow / open_outlets
        
        logger.debug(f"Manifold {self.name} flow rate: {self.flow_rate} m³/s")


class SlugCatcher(ProductionComponent):
    """Slug catcher component for handling intermittent flow regimes."""
    
    def __init__(self, name: str, position: Tuple[float, float, float] = (0, 0, 0),
                volume: float = 50.0, type_: str = "Finger"):
        """
        Initialize a slug catcher component.
        
        Args:
            name (str): Component name
            position (Tuple[float, float, float]): 3D position
            volume (float): Total volume capacity in m³
            type_ (str): Slug catcher type ("Finger", "Vessel", "Hybrid")
        """
        super().__init__(name, ComponentType.SLUG_CATCHER, position)
        
        # Validate type
        valid_types = ["Finger", "Vessel", "Hybrid"]
        if type_ not in valid_types:
            logger.warning(f"Invalid slug catcher type: {type_}. Using 'Finger' instead.")
            type_ = "Finger"
        
        # Slug catcher properties
        self.properties["type"] = type_
        self.properties["volume"] = volume  # m³
        self.properties["liquid_level"] = 20.0  # Percent
        self.properties["gas_capacity"] = 30.0  # m³/s
        self.properties["liquid_capacity"] = 0.3  # m³/s
        self.properties["num_fingers"] = 6 if type_ == "Finger" else 0
        self.properties["slug_handling_capacity"] = volume * 0.6  # m³
        self.properties["pressure_drop"] = 50000.0  # 50 kPa
        
        # Set dimensions based on type and volume
        if type_ == "Vessel":
            # Single large vessel
            diameter = (4 * volume / (3 * math.pi)) ** (1/3)
            length = 3 * diameter
            self.dimensions = (length, diameter, diameter)
        elif type_ == "Finger":
            # Multiple parallel pipes
            length = (volume / self.properties["num_fingers"] / math.pi) ** 0.5 * 20
            width = self.properties["num_fingers"] * 0.5
            height = 1.0
            self.dimensions = (length, width, height)
        else:  # Hybrid
            length = (volume / math.pi) ** 0.5 * 4
            width = length / 3
            height = width / 2
            self.dimensions = (length, width, height)
    
    def set_liquid_level(self, level: float) -> None:
        """
        Set the liquid level in the slug catcher.
        
        Args:
            level (float): Liquid level (0-100%)
        """
        level = max(0.0, min(100.0, level))  # Clamp to 0-100%
        self.properties["liquid_level"] = level
        logger.debug(f"Slug catcher {self.name} liquid level set to {level}%")
        
        # Check for high level alarms
        if level > 80.0:
            logger.warning(f"Slug catcher {self.name} high level alarm: {level}%")
    
    def handle_slug(self, slug_volume: float, duration: float) -> Tuple[bool, float]:
        """
        Simulate handling a slug of liquid.
        
        Args:
            slug_volume (float): Volume of liquid slug in m³
            duration (float): Time over which slug arrives in seconds
            
        Returns:
            Tuple[bool, float]: Success flag and overflow volume if any
        """
        # Check if slug exceeds capacity
        available_capacity = (self.properties["volume"] * 
                            (100.0 - self.properties["liquid_level"]) / 100.0)
        
        if slug_volume > available_capacity:
            overflow = slug_volume - available_capacity
            new_level = 100.0
            logger.warning(f"Slug catcher {self.name} overflow: {overflow:.2f} m³")
            success = False
        else:
            overflow = 0.0
            # Calculate new liquid level
            new_level = self.properties["liquid_level"] + (slug_volume / self.properties["volume"]) * 100.0
            success = True
        
        # Update liquid level
        self.set_liquid_level(new_level)
        
        # Calculate instantaneous flow rate
        self.flow_rate = slug_volume / duration
        
        return success, overflow


class ProductionSystem:
    """Complete production system combining multiple components."""
    
    def __init__(self, name: str):
        """
        Initialize a production system.
        
        Args:
            name (str): System name
        """
        self.name = name
        self.components = {}  # Dict of name -> component
        self.properties = {}  # System-level properties
        
        logger.info(f"Created production system: {name}")
    
    def add_component(self, component: ProductionComponent) -> None:
        """
        Add a component to the system.
        
        Args:
            component (ProductionComponent): Component to add
        """
        if component.name in self.components:
            logger.warning(f"Component with name '{component.name}' already exists, overwriting")
        
        self.components[component.name] = component
        logger.debug(f"Added {component.component_type.name} component '{component.name}' to system")
    
    def remove_component(self, component_name: str) -> bool:
        """
        Remove a component from the system.
        
        Args:
            component_name (str): Name of component to remove
            
        Returns:
            bool: True if component was removed
        """
        if component_name in self.components:
            # Disconnect the component from all others
            component = self.components[component_name]
            
            # Make a copy of connected_components to avoid issues while iterating
            connected = component.connected_components.copy()
            for connected_comp in connected:
                component.disconnect(connected_comp)
            
            # Remove the component
            del self.components[component_name]
            logger.debug(f"Removed component '{component_name}' from system")
            return True
        
        logger.warning(f"Component '{component_name}' not found in system")
        return False
    
    def connect_components(self, from_name: str, to_name: str) -> bool:
        """
        Connect two components in the system.
        
        Args:
            from_name (str): Name of source component
            to_name (str): Name of destination component
            
        Returns:
            bool: True if connection was made
        """
        if from_name not in self.components:
            logger.warning(f"Source component '{from_name}' not found in system")
            return False
        
        if to_name not in self.components:
            logger.warning(f"Destination component '{to_name}' not found in system")
            return False
        
        # Connect the components
        from_comp = self.components[from_name]
        to_comp = self.components[to_name]
        
        from_comp.connect(to_comp)
        logger.debug(f"Connected {from_name} to {to_name}")
        
        return True
    
    def disconnect_components(self, from_name: str, to_name: str) -> bool:
        """
        Disconnect two components in the system.
        
        Args:
            from_name (str): Name of source component
            to_name (str): Name of destination component
            
        Returns:
            bool: True if connection was removed
        """
        if from_name not in self.components:
            logger.warning(f"Source component '{from_name}' not found in system")
            return False
        
        if to_name not in self.components:
            logger.warning(f"Destination component '{to_name}' not found in system")
            return False
        
        # Disconnect the components
        from_comp = self.components[from_name]
        to_comp = self.components[to_name]
        
        from_comp.disconnect(to_comp)
        logger.debug(f"Disconnected {from_name} from {to_name}")
        
        return True
    
    def get_component(self, name: str) -> Optional[ProductionComponent]:
        """
        Get a component by name.
        
        Args:
            name (str): Component name
            
        Returns:
            Optional[ProductionComponent]: The component or None if not found
        """
        return self.components.get(name)
    
    def get_components_by_type(self, component_type: ComponentType) -> List[ProductionComponent]:
        """
        Get all components of a specific type.
        
        Args:
            component_type (ComponentType): Type to filter by
            
        Returns:
            List[ProductionComponent]: List of matching components
        """
        return [comp for comp in self.components.values() if comp.component_type == component_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the production system to a dictionary for serialization.
        
        Returns:
            Dict[str, Any]: System as dictionary
        """
        components_dict = {}
        for name, comp in self.components.items():
            components_dict[name] = comp.to_dict()
        
        return {
            "name": self.name,
            "components": components_dict,
            "properties": self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProductionSystem':
        """
        Create a production system from a dictionary.
        
        Args:
            data (Dict[str, Any]): System data
            
        Returns:
            ProductionSystem: New production system
        """
        system = cls(data["name"])
        system.properties = data.get("properties", {})
        
        # First pass: create all components
        components_lookup = {}
        for name, comp_data in data.get("components", {}).items():
            component = ProductionComponent.from_dict(comp_data)
            system.add_component(component)
            components_lookup[name] = component
        
        # Second pass: establish connections
        for name, comp_data in data.get("components", {}).items():
            component = system.components[name]
            
            # Connect to other components
            for other_name in comp_data.get("connected_to", []):
                if other_name in system.components:
                    component.connect(system.components[other_name])
        
        return system
    
    def create_pipeline_system(self) -> None:
        """Create a template pipeline production system."""
        # Create components
        wellhead = Wellhead("Well-1", position=(0, 0, 0), well_type="Oil")
        christmas_tree = ChristmasTree("XmasTree-1", position=(0, 0, 1.5))
        manifold = Manifold("Manifold-1", position=(5, 0, 0), num_inlets=4, num_outlets=1)
        slug_catcher = SlugCatcher("SlugCatcher-1", position=(50, 0, 0), volume=30.0)
        separator = ThreePhaseSeparator("Separator-1", position=(55, 0, 0), vessel_volume=20.0)
        oil_pump = Pump("OilPump-1", position=(60, 5, 0))
        water_pump = Pump("WaterPump-1", position=(60, -5, 0))
        gas_compressor = Compressor("Compressor-1", position=(60, 0, 3))
        
        # Add to system
        self.add_component(wellhead)
        self.add_component(christmas_tree)
        self.add_component(manifold)
        self.add_component(slug_catcher)
        self.add_component(separator)
        self.add_component(oil_pump)
        self.add_component(water_pump)
        self.add_component(gas_compressor)
        
        # Connect components
        self.connect_components("Well-1", "XmasTree-1")
        self.connect_components("XmasTree-1", "Manifold-1")
        self.connect_components("Manifold-1", "SlugCatcher-1")
        self.connect_components("SlugCatcher-1", "Separator-1")
        self.connect_components("Separator-1", "OilPump-1")
        self.connect_components("Separator-1", "WaterPump-1")
        self.connect_components("Separator-1", "Compressor-1")
        
        # Set properties
        wellhead.properties["wellhead_pressure"] = 6000000.0  # 6 MPa
        wellhead.properties["flowing"] = True
        wellhead.flow_rate = 0.05  # 50 L/s
        
        christmas_tree.set_choke_position(80.0)  # 80% open
        
        separator.set_liquid_level(60.0)
        separator.set_oil_water_interface(40.0)
    
    def create_offshore_platform(self) -> None:
        """Create a template offshore platform production system."""
        # Create wellheads (multiple wells)
        for i in range(1, 5):
            wellhead = Wellhead(f"Well-{i}", position=(i*3, 0, 0), 
                              well_type="Oil" if i % 3 != 0 else "Gas")
            christmas_tree = ChristmasTree(f"XmasTree-{i}", position=(i*3, 0, 1.5))
            self.add_component(wellhead)
            self.add_component(christmas_tree)
            self.connect_components(f"Well-{i}", f"XmasTree-{i}")
            
            # Set some wells to be producing
            if i % 2 == 0:
                wellhead.properties["flowing"] = True
                wellhead.flow_rate = 0.03 + (i * 0.01)  # Varying flow rates
                christmas_tree.set_choke_position(70.0 + i*5)
        
        # Create manifolds
        oil_manifold = Manifold("OilManifold", position=(10, 5, 0), num_inlets=3, num_outlets=1)
        gas_manifold = Manifold("GasManifold", position=(10, -5, 0), num_inlets=2, num_outlets=1)
        self.add_component(oil_manifold)
        self.add_component(gas_manifold)
        
        # Connect wellheads to appropriate manifolds
        for i in range(1, 5):
            wellhead = self.get_component(f"Well-{i}")
            if wellhead.properties["well_type"] == "Oil":
                self.connect_components(f"XmasTree-{i}", "OilManifold")
            else:
                self.connect_components(f"XmasTree-{i}", "GasManifold")
        
        # Create separators
        hp_separator = TwoPhaseSeparator("HPSeparator", position=(20, 0, 0), vessel_volume=15.0)
        lp_separator = ThreePhaseSeparator("LPSeparator", position=(30, 0, 0), vessel_volume=25.0)
        self.add_component(hp_separator)
        self.add_component(lp_separator)
        
        # Connect manifolds to HP separator
        self.connect_components("OilManifold", "HPSeparator")
        self.connect_components("GasManifold", "HPSeparator")
        
        # Connect HP separator to LP separator
        self.connect_components("HPSeparator", "LPSeparator")
        
        # Create pumps and compressor
        oil_pump = Pump("OilPump", position=(40, 5, 0))
        water_pump = Pump("WaterPump", position=(40, -5, 0))
        gas_compressor = Compressor("GasCompressor", position=(40, 0, 5))
        self.add_component(oil_pump)
        self.add_component(water_pump)
        self.add_component(gas_compressor)
        
        # Connect LP separator to pumps and compressor
        self.connect_components("LPSeparator", "OilPump")
        self.connect_components("LPSeparator", "WaterPump")
        self.connect_components("LPSeparator", "GasCompressor")
        
        # Set separator levels
        hp_separator.set_liquid_level(50.0)
        lp_separator.set_liquid_level(60.0)
        lp_separator.set_oil_water_interface(35.0)
    
    def simulate_step(self, dt: float = 1.0) -> None:
        """
        Perform a simple simulation step.
        
        Args:
            dt (float): Time step in seconds
        """
        # This would implement a simple flow simulation through the system
        # For a complete simulation, this would integrate with OpenFOAM
        # Here we just illustrate a very basic flow propagation
        
        # Collect all flowing components
        flowing_components = []
        for component in self.components.values():
            if (hasattr(component, "flow_rate") and component.flow_rate > 0.0 and
                component.properties.get("flowing", True)):
                flowing_components.append(component)
        
        # Propagate flows through connections
        for component in flowing_components:
            # Find all downstream components
            for connected in component.connected_components:
                # Skip components that flow into this one
                if component in connected.connected_components:
                    # Check if bidirectional connection or flow from connected to component
                    continue
                
                # Propagate flow to connected component
                if hasattr(connected, "flow_rate"):
                    # Special handling for different component types
                    if connected.component_type == ComponentType.SEPARATOR_2PHASE:
                        # Update separator liquid level based on flow
                        liquid_fraction = 0.4  # Assume 40% liquid by volume
                        liquid_volume = component.flow_rate * dt * liquid_fraction
                        vessel_volume = connected.properties["vessel_volume"]
                        level_change = (liquid_volume / vessel_volume) * 100.0
                        new_level = min(100.0, connected.properties["liquid_level"] + level_change)
                        connected.set_liquid_level(new_level)
                    
                    elif connected.component_type == ComponentType.SEPARATOR_3PHASE:
                        # Handle 3-phase differently
                        liquid_fraction = 0.4  # Assume 40% liquid
                        water_fraction = 0.15  # Assume 15% water in the liquid
                        liquid_volume = component.flow_rate * dt * liquid_fraction
                        water_volume = liquid_volume * (water_fraction / liquid_fraction)
                        vessel_volume = connected.properties["vessel_volume"]
                        
                        # Update liquid level
                        level_change = (liquid_volume / vessel_volume) * 100.0
                        new_level = min(100.0, connected.properties["liquid_level"] + level_change)
                        connected.set_liquid_level(new_level)
                        
                        # Update oil-water interface based on water content
                        interface = connected.properties["oil_water_interface_level"]
                        oil_volume = liquid_volume - water_volume
                        total_liquid = vessel_volume * (connected.properties["liquid_level"] / 100.0)
                        oil_layer = total_liquid * (1.0 - (interface / 100.0))
                        water_layer = total_liquid - oil_layer
                        new_water_layer = water_layer + water_volume
                        new_total = oil_layer + oil_volume + new_water_layer
                        new_interface = (new_water_layer / new_total) * 100.0
                        connected.set_oil_water_interface(new_interface)
                    
                    else:
                        # Default flow propagation
                        connected.flow_rate = component.flow_rate
        
        logger.debug(f"Simulation step completed for {self.name}, dt={dt}s")


# Helper functions to create common production systems

def create_pipeline_system(name: str = "Pipeline System") -> ProductionSystem:
    """
    Create a template pipeline production system.
    
    Args:
        name (str): System name
        
    Returns:
        ProductionSystem: Configured production system
    """
    system = ProductionSystem(name)
    system.create_pipeline_system()
    return system

def create_offshore_platform(name: str = "Offshore Platform") -> ProductionSystem:
    """
    Create a template offshore platform production system.
    
    Args:
        name (str): System name
        
    Returns:
        ProductionSystem: Configured production system
    """
    system = ProductionSystem(name)
    system.create_offshore_platform()
    return system

def load_system_from_file(filepath: str) -> Optional[ProductionSystem]:
    """
    Load a production system from a file.
    
    Args:
        filepath (str): Path to the system file (JSON)
        
    Returns:
        Optional[ProductionSystem]: Loaded system or None if loading failed
    """
    try:
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        system = ProductionSystem.from_dict(data)
        logger.info(f"Loaded production system from {filepath}")
        return system
    except Exception as e:
        logger.error(f"Error loading production system: {e}")
        return None

def save_system_to_file(system: ProductionSystem, filepath: str) -> bool:
    """
    Save a production system to a file.
    
    Args:
        system (ProductionSystem): System to save
        filepath (str): Path to save to (JSON)
        
    Returns:
        bool: True if saving succeeded
    """
    try:
        import json
        data = system.to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved production system to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving production system: {e}")
        return False


# Main function for testing
def main():
    """Test function to demonstrate the module functionality."""
    # Create a simple production system
    system = create_pipeline_system("Test Pipeline")
    
    # Get a component and modify it
    wellhead = system.get_component("Well-1")
    if wellhead:
        wellhead.start_flow(0.1)  # 100 L/s
    
    # Save system to file
    save_system_to_file(system, "test_pipeline.json")
    
    # Load system from file
    loaded_system = load_system_from_file("test_pipeline.json")
    
    # Verify loaded system
    if loaded_system:
        print(f"Loaded system has {len(loaded_system.components)} components")
        
        # Run a simple simulation step
        loaded_system.simulate_step(10.0)  # 10 second time step

if __name__ == "__main__":
    main()