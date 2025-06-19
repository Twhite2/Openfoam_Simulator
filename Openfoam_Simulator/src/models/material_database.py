#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Material database for Openfoam_Simulator application.

This module provides a database of material and fluid properties for oil & gas
CFD simulations, including:
- Physical properties of fluids (oil, water, gas, etc.)
- Physical properties of solid materials (steel, concrete, etc.)
- Methods for calculating temperature and pressure-dependent properties
- Loading and saving material properties from/to files
"""

import os
import sys
import json
import math
import logging
import numpy as np
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

# Import utility modules
from ..utils.logger import get_logger
from ..config import get_value, set_value

logger = get_logger(__name__)


class MaterialType(Enum):
    """Enumeration of material types."""
    
    LIQUID = 1
    GAS = 2
    SOLID = 3
    SLURRY = 4
    UNDEFINED = 99


class PropertyType(Enum):
    """Enumeration of property types."""
    
    CONSTANT = 1     # Constant value
    TABLE = 2        # Tabulated values (e.g., vs temperature)
    EQUATION = 3     # Equation-based (e.g., polynomial)
    CORRELATION = 4  # Empirical correlation
    UNDEFINED = 99


class Material:
    """Class representing a material with physical properties."""
    
    def __init__(self, name: str, material_type: MaterialType = MaterialType.UNDEFINED):
        """
        Initialize a material.
        
        Args:
            name (str): Name of the material
            material_type (MaterialType): Type of material
        """
        self.name = name
        self.material_type = material_type
        self.description = ""
        self.properties = {}  # Dict mapping property name -> Property object
        self.metadata = {}    # Additional metadata
    
    def add_property(self, prop):
        """
        Add a property to the material.
        
        Args:
            prop (Property): Property to add
        """
        self.properties[prop.name] = prop
    
    def get_property(self, name: str):
        """
        Get a property by name.
        
        Args:
            name (str): Name of the property
        
        Returns:
            Property: The property object, or None if not found
        """
        return self.properties.get(name)
    
    def get_property_value(self, name: str, temperature: float = 293.15, 
                          pressure: float = 101325.0) -> Optional[float]:
        """
        Get the value of a property at specified conditions.
        
        Args:
            name (str): Name of the property
            temperature (float): Temperature in K
            pressure (float): Pressure in Pa
        
        Returns:
            Optional[float]: Property value, or None if property not found
        """
        prop = self.get_property(name)
        if prop:
            return prop.get_value(temperature, pressure)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            'name': self.name,
            'material_type': self.material_type.name,
            'description': self.description,
            'properties': {name: prop.to_dict() for name, prop in self.properties.items()},
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Material':
        """
        Create from dictionary representation.
        
        Args:
            data (Dict[str, Any]): Dictionary representation
            
        Returns:
            Material: New Material object
        """
        material_type = MaterialType[data['material_type']] if 'material_type' in data else MaterialType.UNDEFINED
        material = cls(data.get('name', ''), material_type)
        material.description = data.get('description', '')
        material.metadata = data.get('metadata', {})
        
        # Load properties
        if 'properties' in data:
            for prop_name, prop_data in data['properties'].items():
                material.add_property(Property.from_dict(prop_data))
        
        return material


class Property:
    """Class representing a physical property with its dependence on conditions."""
    
    def __init__(self, name: str, value_type: PropertyType = PropertyType.CONSTANT, 
                units: str = "", description: str = ""):
        """
        Initialize a property.
        
        Args:
            name (str): Name of the property
            value_type (PropertyType): Type of property value representation
            units (str): Units of the property
            description (str): Description of the property
        """
        self.name = name
        self.value_type = value_type
        self.units = units
        self.description = description
        
        # The value storage depends on the value_type
        self.constant_value = None
        self.table_values = {}  # Dict mapping (T, P) -> value
        self.equation_coeffs = []  # Coefficients for equation
        self.equation_type = ""    # Type of equation (polynomial, exponential, etc.)
        self.correlation_func = None  # Function for correlation
        self.correlation_name = ""    # Name of correlation
        
        # Temperature and pressure ranges
        self.temp_min = None
        self.temp_max = None
        self.pressure_min = None
        self.pressure_max = None
    
    def set_constant_value(self, value: float):
        """
        Set a constant value for the property.
        
        Args:
            value (float): Constant value
        """
        self.value_type = PropertyType.CONSTANT
        self.constant_value = value
    
    def add_table_value(self, temperature: float, pressure: float, value: float):
        """
        Add a tabulated value at specific temperature and pressure.
        
        Args:
            temperature (float): Temperature in K
            pressure (float): Pressure in Pa
            value (float): Property value
        """
        self.value_type = PropertyType.TABLE
        self.table_values[(temperature, pressure)] = value
        
        # Update ranges
        if self.temp_min is None or temperature < self.temp_min:
            self.temp_min = temperature
        if self.temp_max is None or temperature > self.temp_max:
            self.temp_max = temperature
        if self.pressure_min is None or pressure < self.pressure_min:
            self.pressure_min = pressure
        if self.pressure_max is None or pressure > self.pressure_max:
            self.pressure_max = pressure
    
    def set_equation(self, equation_type: str, coefficients: List[float], 
                    temp_min: float = None, temp_max: float = None,
                    pressure_min: float = None, pressure_max: float = None):
        """
        Set equation-based representation.
        
        Args:
            equation_type (str): Type of equation
            coefficients (List[float]): Coefficients for the equation
            temp_min (float, optional): Minimum valid temperature
            temp_max (float, optional): Maximum valid temperature
            pressure_min (float, optional): Minimum valid pressure
            pressure_max (float, optional): Maximum valid pressure
        """
        self.value_type = PropertyType.EQUATION
        self.equation_type = equation_type
        self.equation_coeffs = coefficients
        
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.pressure_min = pressure_min
        self.pressure_max = pressure_max
    
    def set_correlation(self, correlation_name: str, correlation_func: Callable,
                       temp_min: float = None, temp_max: float = None,
                       pressure_min: float = None, pressure_max: float = None):
        """
        Set correlation-based representation.
        
        Args:
            correlation_name (str): Name of the correlation
            correlation_func (Callable): Function implementing the correlation
            temp_min (float, optional): Minimum valid temperature
            temp_max (float, optional): Maximum valid temperature
            pressure_min (float, optional): Minimum valid pressure
            pressure_max (float, optional): Maximum valid pressure
        """
        self.value_type = PropertyType.CORRELATION
        self.correlation_name = correlation_name
        self.correlation_func = correlation_func
        
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.pressure_min = pressure_min
        self.pressure_max = pressure_max
    
    def get_value(self, temperature: float = 293.15, pressure: float = 101325.0) -> Optional[float]:
        """
        Get property value at specified conditions.
        
        Args:
            temperature (float): Temperature in K
            pressure (float): Pressure in Pa
        
        Returns:
            Optional[float]: Property value, or None if out of range
        """
        # Check if conditions are in range
        if (self.temp_min is not None and temperature < self.temp_min) or \
           (self.temp_max is not None and temperature > self.temp_max) or \
           (self.pressure_min is not None and pressure < self.pressure_min) or \
           (self.pressure_max is not None and pressure > self.pressure_max):
            logger.warning(f"Property {self.name} evaluated outside valid range")
            # Continue evaluation but with warning
        
        # Calculate based on value type
        if self.value_type == PropertyType.CONSTANT:
            return self.constant_value
        
        elif self.value_type == PropertyType.TABLE:
            # Find nearest tabulated value or interpolate
            if (temperature, pressure) in self.table_values:
                return self.table_values[(temperature, pressure)]
            else:
                # Simple bilinear interpolation
                return self._interpolate_table(temperature, pressure)
        
        elif self.value_type == PropertyType.EQUATION:
            # Evaluate equation
            return self._evaluate_equation(temperature, pressure)
        
        elif self.value_type == PropertyType.CORRELATION:
            # Use correlation function
            if self.correlation_func:
                try:
                    return self.correlation_func(temperature, pressure)
                except Exception as e:
                    logger.error(f"Error evaluating correlation for {self.name}: {e}")
                    return None
            return None
        
        return None
    
    def _interpolate_table(self, temperature: float, pressure: float) -> Optional[float]:
        """
        Interpolate property value from tabulated data.
        
        Args:
            temperature (float): Temperature in K
            pressure (float): Pressure in Pa
        
        Returns:
            Optional[float]: Interpolated value, or None if interpolation not possible
        """
        if not self.table_values:
            return None
        
        # Get unique temperature and pressure values
        temps = sorted(set(t for t, _ in self.table_values.keys()))
        pressures = sorted(set(p for _, p in self.table_values.keys()))
        
        # Find bracketing temperatures
        t_lower, t_upper = None, None
        for t in temps:
            if t <= temperature:
                t_lower = t
            if t >= temperature and t_upper is None:
                t_upper = t
        
        # Find bracketing pressures
        p_lower, p_upper = None, None
        for p in pressures:
            if p <= pressure:
                p_lower = p
            if p >= pressure and p_upper is None:
                p_upper = p
        
        # If can't bracket, use nearest value
        if t_lower is None or t_upper is None or p_lower is None or p_upper is None:
            # Find nearest point in table
            nearest_point = min(self.table_values.keys(),
                               key=lambda pt: ((pt[0] - temperature) ** 2 + (pt[1] - pressure) ** 2) ** 0.5)
            return self.table_values[nearest_point]
        
        # If at an exact point, return that value
        if (t_lower, p_lower) in self.table_values and t_lower == temperature and p_lower == pressure:
            return self.table_values[(t_lower, p_lower)]
        
        # Bilinear interpolation
        try:
            # Get the four corner values
            f_ll = self.table_values.get((t_lower, p_lower))
            f_lu = self.table_values.get((t_lower, p_upper))
            f_ul = self.table_values.get((t_upper, p_lower))
            f_uu = self.table_values.get((t_upper, p_upper))
            
            # Check if we have all four corners
            if f_ll is None or f_lu is None or f_ul is None or f_uu is None:
                # Fall back to nearest value
                nearest_point = min(self.table_values.keys(),
                                  key=lambda pt: ((pt[0] - temperature) ** 2 + (pt[1] - pressure) ** 2) ** 0.5)
                return self.table_values[nearest_point]
            
            # Interpolation weights
            t_weight = (temperature - t_lower) / (t_upper - t_lower) if t_upper != t_lower else 0
            p_weight = (pressure - p_lower) / (p_upper - p_lower) if p_upper != p_lower else 0
            
            # Bilinear interpolation formula
            result = (1 - t_weight) * (1 - p_weight) * f_ll + \
                     (1 - t_weight) * p_weight * f_lu + \
                     t_weight * (1 - p_weight) * f_ul + \
                     t_weight * p_weight * f_uu
            
            return result
            
        except Exception as e:
            logger.error(f"Error interpolating table for {self.name}: {e}")
            # Fall back to nearest value
            nearest_point = min(self.table_values.keys(),
                              key=lambda pt: ((pt[0] - temperature) ** 2 + (pt[1] - pressure) ** 2) ** 0.5)
            return self.table_values[nearest_point]
    
    def _evaluate_equation(self, temperature: float, pressure: float) -> Optional[float]:
        """
        Evaluate property using equation.
        
        Args:
            temperature (float): Temperature in K
            pressure (float): Pressure in Pa
        
        Returns:
            Optional[float]: Calculated value, or None if calculation fails
        """
        if not self.equation_coeffs:
            return None
        
        try:
            # Handle different equation types
            if self.equation_type.lower() == 'polynomial_t':
                # Polynomial in temperature: a0 + a1*T + a2*T^2 + ...
                return sum(coef * temperature ** i for i, coef in enumerate(self.equation_coeffs))
            
            elif self.equation_type.lower() == 'polynomial_p':
                # Polynomial in pressure: a0 + a1*P + a2*P^2 + ...
                return sum(coef * pressure ** i for i, coef in enumerate(self.equation_coeffs))
            
            elif self.equation_type.lower() == 'exponential_t':
                # Exponential in temperature: a0 * exp(a1 * T)
                if len(self.equation_coeffs) >= 2:
                    return self.equation_coeffs[0] * math.exp(self.equation_coeffs[1] * temperature)
                return None
            
            elif self.equation_type.lower() == 'power_t':
                # Power law in temperature: a0 * T^a1
                if len(self.equation_coeffs) >= 2:
                    return self.equation_coeffs[0] * temperature ** self.equation_coeffs[1]
                return None
            
            elif self.equation_type.lower() == 'sutherland':
                # Sutherland's law for viscosity: a0 * T^1.5 / (T + a1)
                if len(self.equation_coeffs) >= 2:
                    return self.equation_coeffs[0] * temperature ** 1.5 / (temperature + self.equation_coeffs[1])
                return None
            
            else:
                logger.warning(f"Unknown equation type: {self.equation_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error evaluating equation for {self.name}: {e}")
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        data = {
            'name': self.name,
            'value_type': self.value_type.name,
            'units': self.units,
            'description': self.description,
            'temp_min': self.temp_min,
            'temp_max': self.temp_max,
            'pressure_min': self.pressure_min,
            'pressure_max': self.pressure_max
        }
        
        # Add value-specific data
        if self.value_type == PropertyType.CONSTANT:
            data['constant_value'] = self.constant_value
        
        elif self.value_type == PropertyType.TABLE:
            # Convert table keys to strings for JSON serialization
            data['table_values'] = {f"{t},{p}": v for (t, p), v in self.table_values.items()}
        
        elif self.value_type == PropertyType.EQUATION:
            data['equation_type'] = self.equation_type
            data['equation_coeffs'] = self.equation_coeffs
        
        elif self.value_type == PropertyType.CORRELATION:
            data['correlation_name'] = self.correlation_name
            # Note: correlation_func cannot be serialized
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Property':
        """
        Create from dictionary representation.
        
        Args:
            data (Dict[str, Any]): Dictionary representation
            
        Returns:
            Property: New Property object
        """
        value_type = PropertyType[data['value_type']] if 'value_type' in data else PropertyType.UNDEFINED
        prop = cls(
            name=data.get('name', ''),
            value_type=value_type,
            units=data.get('units', ''),
            description=data.get('description', '')
        )
        
        prop.temp_min = data.get('temp_min')
        prop.temp_max = data.get('temp_max')
        prop.pressure_min = data.get('pressure_min')
        prop.pressure_max = data.get('pressure_max')
        
        # Load value-specific data
        if value_type == PropertyType.CONSTANT and 'constant_value' in data:
            prop.constant_value = data['constant_value']
        
        elif value_type == PropertyType.TABLE and 'table_values' in data:
            # Convert string keys back to tuples
            for key_str, value in data['table_values'].items():
                try:
                    t, p = map(float, key_str.split(','))
                    prop.table_values[(t, p)] = value
                except Exception as e:
                    logger.warning(f"Error parsing table key: {key_str}, {e}")
        
        elif value_type == PropertyType.EQUATION:
            prop.equation_type = data.get('equation_type', '')
            prop.equation_coeffs = data.get('equation_coeffs', [])
        
        elif value_type == PropertyType.CORRELATION:
            prop.correlation_name = data.get('correlation_name', '')
            # Note: correlation_func cannot be deserialized
        
        return prop


class MaterialDatabase:
    """Database of materials and their properties."""
    
    def __init__(self):
        """Initialize the material database."""
        self.materials = {}  # Dict mapping material name -> Material object
        self.loaded = False
    
    def add_material(self, material: Material):
        """
        Add a material to the database.
        
        Args:
            material (Material): Material to add
        """
        self.materials[material.name] = material
    
    def get_material(self, name: str) -> Optional[Material]:
        """
        Get a material by name.
        
        Args:
            name (str): Name of the material
        
        Returns:
            Optional[Material]: Material object, or None if not found
        """
        return self.materials.get(name)
    
    def get_material_names(self) -> List[str]:
        """
        Get list of all material names in the database.
        
        Returns:
            List[str]: List of material names
        """
        return list(self.materials.keys())
    
    def get_materials_by_type(self, material_type: MaterialType) -> List[Material]:
        """
        Get materials of a specific type.
        
        Args:
            material_type (MaterialType): Type of materials to get
        
        Returns:
            List[Material]: List of materials of the specified type
        """
        return [m for m in self.materials.values() if m.material_type == material_type]
    
    def load(self, filepath: str) -> bool:
        """
        Load database from a file.
        
        Args:
            filepath (str): Path to database file
        
        Returns:
            bool: True if loading successful, False otherwise
        """
        if not os.path.isfile(filepath):
            logger.error(f"Material database file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                # Clear existing materials
                self.materials.clear()
                
                # Load materials
                if 'materials' in data:
                    for material_data in data['materials']:
                        material = Material.from_dict(material_data)
                        self.add_material(material)
                
                self.loaded = True
                logger.info(f"Loaded {len(self.materials)} materials from {filepath}")
                return True
                
        except Exception as e:
            logger.error(f"Error loading material database: {e}")
            return False
    
    def save(self, filepath: str) -> bool:
        """
        Save database to a file.
        
        Args:
            filepath (str): Path to database file
        
        Returns:
            bool: True if saving successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                data = {
                    'materials': [m.to_dict() for m in self.materials.values()]
                }
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved {len(self.materials)} materials to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving material database: {e}")
            return False
    
    def init_default_database(self):
        """Initialize database with default materials."""
        # Add standard fluids
        self._add_default_fluids()
        
        # Add standard solids
        self._add_default_solids()
        
        # Mark as loaded
        self.loaded = True
        logger.info(f"Initialized default material database with {len(self.materials)} materials")
    
    def _add_default_fluids(self):
        """Add default fluid materials to the database."""
        # Water
        water = Material("Water", MaterialType.LIQUID)
        water.description = "Standard liquid water properties"
        
        # Density of water (kg/m³)
        density = Property("density", PropertyType.EQUATION, "kg/m³", "Density")
        density.set_equation("polynomial_t", [999.83952, 16.945176, -7.9870401e-3, -46.170461e-6, 105.56302e-9, -280.54253e-12], 
                           273.15, 373.15)
        water.add_property(density)
        
        # Viscosity of water (Pa·s)
        viscosity = Property("viscosity", PropertyType.EQUATION, "Pa·s", "Dynamic viscosity")
        viscosity.set_equation("sutherland", [2.414e-5, 247.8], 273.15, 373.15)
        water.add_property(viscosity)
        
        # Specific heat capacity of water (J/(kg·K))
        cp = Property("specific_heat", PropertyType.CONSTANT, "J/(kg·K)", "Specific heat capacity")
        cp.set_constant_value(4182.0)
        water.add_property(cp)
        
        # Thermal conductivity of water (W/(m·K))
        k = Property("thermal_conductivity", PropertyType.CONSTANT, "W/(m·K)", "Thermal conductivity")
        k.set_constant_value(0.6)
        water.add_property(k)
        
        # Surface tension of water (N/m)
        sigma = Property("surface_tension", PropertyType.EQUATION, "N/m", "Surface tension")
        sigma.set_equation("polynomial_t", [0.07564, -1.39e-4, -3.01e-7], 273.15, 373.15)
        water.add_property(sigma)
        
        self.add_material(water)
        
        # Crude Oil
        crude_oil = Material("Crude Oil", MaterialType.LIQUID)
        crude_oil.description = "Medium crude oil properties"
        
        # Density of crude oil (kg/m³)
        density = Property("density", PropertyType.CONSTANT, "kg/m³", "Density")
        density.set_constant_value(870.0)
        crude_oil.add_property(density)
        
        # Viscosity of crude oil (Pa·s)
        viscosity = Property("viscosity", PropertyType.CONSTANT, "Pa·s", "Dynamic viscosity")
        viscosity.set_constant_value(0.05)  # Medium crude oil
        crude_oil.add_property(viscosity)
        
        # Specific heat capacity of crude oil (J/(kg·K))
        cp = Property("specific_heat", PropertyType.CONSTANT, "J/(kg·K)", "Specific heat capacity")
        cp.set_constant_value(1800.0)
        crude_oil.add_property(cp)
        
        # Thermal conductivity of crude oil (W/(m·K))
        k = Property("thermal_conductivity", PropertyType.CONSTANT, "W/(m·K)", "Thermal conductivity")
        k.set_constant_value(0.12)
        crude_oil.add_property(k)
        
        # Surface tension of crude oil with water (N/m)
        sigma = Property("surface_tension", PropertyType.CONSTANT, "N/m", "Surface tension with water")
        sigma.set_constant_value(0.025)
        crude_oil.add_property(sigma)
        
        self.add_material(crude_oil)
        
        # Natural Gas
        natural_gas = Material("Natural Gas", MaterialType.GAS)
        natural_gas.description = "Natural gas (primarily methane) properties"
        
        # Density of natural gas (kg/m³) at 1 atm, 15°C
        density = Property("density", PropertyType.EQUATION, "kg/m³", "Density")
        density.set_equation("polynomial_t", [1.9020, -0.0026], 273.15, 373.15)
        natural_gas.add_property(density)
        
        # Viscosity of natural gas (Pa·s)
        viscosity = Property("viscosity", PropertyType.EQUATION, "Pa·s", "Dynamic viscosity")
        viscosity.set_equation("sutherland", [1.087e-5, 198.6], 273.15, 373.15)
        natural_gas.add_property(viscosity)
        
        # Specific heat capacity of natural gas (J/(kg·K))
        cp = Property("specific_heat", PropertyType.CONSTANT, "J/(kg·K)", "Specific heat capacity")
        cp.set_constant_value(2200.0)
        natural_gas.add_property(cp)
        
        # Thermal conductivity of natural gas (W/(m·K))
        k = Property("thermal_conductivity", PropertyType.CONSTANT, "W/(m·K)", "Thermal conductivity")
        k.set_constant_value(0.033)
        natural_gas.add_property(k)
        
        self.add_material(natural_gas)
        
        # Diesel
        diesel = Material("Diesel", MaterialType.LIQUID)
        diesel.description = "Diesel fuel properties"
        
        density = Property("density", PropertyType.CONSTANT, "kg/m³", "Density")
        density.set_constant_value(832.0)
        diesel.add_property(density)
        
        viscosity = Property("viscosity", PropertyType.CONSTANT, "Pa·s", "Dynamic viscosity")
        viscosity.set_constant_value(0.0024)
        diesel.add_property(viscosity)
        
        cp = Property("specific_heat", PropertyType.CONSTANT, "J/(kg·K)", "Specific heat capacity")
        cp.set_constant_value(1750.0)
        diesel.add_property(cp)
        
        self.add_material(diesel)
        
        # Glycol
        glycol = Material("Glycol", MaterialType.LIQUID)
        glycol.description = "Ethylene glycol properties (for hydrate inhibition)"
        
        density = Property("density", PropertyType.CONSTANT, "kg/m³", "Density")
        density.set_constant_value(1110.0)
        glycol.add_property(density)
        
        viscosity = Property("viscosity", PropertyType.CONSTANT, "Pa·s", "Dynamic viscosity")
        viscosity.set_constant_value(0.016)
        glycol.add_property(viscosity)
        
        self.add_material(glycol)
    
    def _add_default_solids(self):
        """Add default solid materials to the database."""
        # Carbon Steel
        steel = Material("Carbon Steel", MaterialType.SOLID)
        steel.description = "Standard carbon steel (ASTM A106 Grade B)"
        
        # Density of steel (kg/m³)
        density = Property("density", PropertyType.CONSTANT, "kg/m³", "Density")
        density.set_constant_value(7850.0)
        steel.add_property(density)
        
        # Thermal conductivity of steel (W/(m·K))
        k = Property("thermal_conductivity", PropertyType.CONSTANT, "W/(m·K)", "Thermal conductivity")
        k.set_constant_value(43.0)
        steel.add_property(k)
        
        # Specific heat capacity of steel (J/(kg·K))
        cp = Property("specific_heat", PropertyType.CONSTANT, "J/(kg·K)", "Specific heat capacity")
        cp.set_constant_value(490.0)
        steel.add_property(cp)
        
        # Yield strength of steel (Pa)
        yield_strength = Property("yield_strength", PropertyType.CONSTANT, "Pa", "Yield strength")
        yield_strength.set_constant_value(240e6)  # 240 MPa
        steel.add_property(yield_strength)
        
        # Young's modulus of steel (Pa)
        young = Property("youngs_modulus", PropertyType.CONSTANT, "Pa", "Young's modulus")
        young.set_constant_value(200e9)  # 200 GPa
        steel.add_property(young)
        
        # Poisson's ratio of steel
        poisson = Property("poisson_ratio", PropertyType.CONSTANT, "", "Poisson's ratio")
        poisson.set_constant_value(0.29)
        steel.add_property(poisson)
        
        # Roughness of steel pipe (m)
        roughness = Property("roughness", PropertyType.CONSTANT, "m", "Surface roughness")
        roughness.set_constant_value(4.5e-5)  # 0.045 mm
        steel.add_property(roughness)
        
        self.add_material(steel)
        
        # Stainless Steel
        ss = Material("Stainless Steel", MaterialType.SOLID)
        ss.description = "Stainless steel 316"
        
        density = Property("density", PropertyType.CONSTANT, "kg/m³", "Density")
        density.set_constant_value(8000.0)
        ss.add_property(density)
        
        k = Property("thermal_conductivity", PropertyType.CONSTANT, "W/(m·K)", "Thermal conductivity")
        k.set_constant_value(16.0)
        ss.add_property(k)
        
        cp = Property("specific_heat", PropertyType.CONSTANT, "J/(kg·K)", "Specific heat capacity")
        cp.set_constant_value(500.0)
        ss.add_property(cp)
        
        yield_strength = Property("yield_strength", PropertyType.CONSTANT, "Pa", "Yield strength")
        yield_strength.set_constant_value(290e6)  # 290 MPa
        ss.add_property(yield_strength)
        
        roughness = Property("roughness", PropertyType.CONSTANT, "m", "Surface roughness")
        roughness.set_constant_value(1.5e-5)  # 0.015 mm
        ss.add_property(roughness)
        
        self.add_material(ss)
        
        # Concrete
        concrete = Material("Concrete", MaterialType.SOLID)
        concrete.description = "Standard concrete"
        
        density = Property("density", PropertyType.CONSTANT, "kg/m³", "Density")
        density.set_constant_value(2300.0)
        concrete.add_property(density)
        
        k = Property("thermal_conductivity", PropertyType.CONSTANT, "W/(m·K)", "Thermal conductivity")
        k.set_constant_value(1.4)
        concrete.add_property(k)
        
        cp = Property("specific_heat", PropertyType.CONSTANT, "J/(kg·K)", "Specific heat capacity")
        cp.set_constant_value(880.0)
        concrete.add_property(cp)
        
        self.add_material(concrete)
        
        # Soil
        soil = Material("Soil", MaterialType.SOLID)
        soil.description = "Average soil properties"
        
        density = Property("density", PropertyType.CONSTANT, "kg/m³", "Density")
        density.set_constant_value(1600.0)
        soil.add_property(density)
        
        k = Property("thermal_conductivity", PropertyType.CONSTANT, "W/(m·K)", "Thermal conductivity")
        k.set_constant_value(0.9)
        soil.add_property(k)
        
        self.add_material(soil)


# Create a singleton instance
_database = None

def get_database() -> MaterialDatabase:
    """
    Get the material database singleton instance.
    
    Returns:
        MaterialDatabase: The material database instance
    """
    global _database
    if _database is None:
        _database = MaterialDatabase()
        
        # Try to load from configured file
        database_path = get_value('industry.material_database_path', 'data/materials.json')
        if os.path.isfile(database_path):
            _database.load(database_path)
        else:
            # Initialize with default values
            _database.init_default_database()
            
            # Save to configured path
            if get_value('app.auto_save', True):
                _database.save(database_path)
    
    return _database

def get_material(name: str) -> Optional[Material]:
    """
    Get a material by name from the database.
    
    Args:
        name (str): Name of the material
    
    Returns:
        Optional[Material]: Material object, or None if not found
    """
    return get_database().get_material(name)

def get_property_value(material_name: str, property_name: str, 
                     temperature: float = 293.15, pressure: float = 101325.0) -> Optional[float]:
    """
    Get a property value for a material.
    
    Args:
        material_name (str): Name of the material
        property_name (str): Name of the property
        temperature (float): Temperature in K
        pressure (float): Pressure in Pa
    
    Returns:
        Optional[float]: Property value, or None if not found
    """
    material = get_material(material_name)
    if material:
        return material.get_property_value(property_name, temperature, pressure)
    return None

def save_database(filepath: str = None) -> bool:
    """
    Save the database to a file.
    
    Args:
        filepath (str, optional): Path to save to, if None uses configured path
    
    Returns:
        bool: True if saving successful, False otherwise
    """
    if filepath is None:
        filepath = get_value('industry.material_database_path', 'data/materials.json')
    
    return get_database().save(filepath)

def load_database(filepath: str = None) -> bool:
    """
    Load the database from a file.
    
    Args:
        filepath (str, optional): Path to load from, if None uses configured path
    
    Returns:
        bool: True if loading successful, False otherwise
    """
    if filepath is None:
        filepath = get_value('industry.material_database_path', 'data/materials.json')
    
    return get_database().load(filepath)

def add_custom_material(material: Material) -> bool:
    """
    Add a custom material to the database.
    
    Args:
        material (Material): Material to add
    
    Returns:
        bool: True if successful, False if material already exists
    """
    database = get_database()
    if material.name in database.materials:
        return False
    
    database.add_material(material)
    return True