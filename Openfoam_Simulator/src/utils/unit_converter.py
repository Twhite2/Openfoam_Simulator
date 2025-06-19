#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit converter for Openfoam_Simulator application.

This module provides utilities for converting between different unit systems
commonly used in the oil & gas industry:
- SI/Metric: International System of Units (m, kg, s, K, Pa, etc.)
- Imperial: British Imperial and US Customary Units (ft, lb, °F, psi, etc.)
- Field: Oil & Gas field units (ft, psi, cp, bbl, etc.)

The module supports conversion of various physical quantities:
- Length
- Area
- Volume
- Mass
- Density
- Pressure
- Temperature
- Viscosity
- Flow rate
- Time
- Velocity
- Energy
- Power
- Force
"""

import re
from typing import Dict, Tuple, List, Optional, Union, Any, Callable
from enum import Enum, auto
import math
import logging

# Set up logging
logger = logging.getLogger(__name__)


class UnitSystem(Enum):
    """Enumeration of supported unit systems."""
    SI = auto()  # International System of Units (metric)
    IMPERIAL = auto()  # Imperial units
    FIELD = auto()  # Oil & gas field units


class PhysicalQuantity(Enum):
    """Enumeration of physical quantities."""
    LENGTH = auto()
    AREA = auto()
    VOLUME = auto()
    MASS = auto()
    DENSITY = auto()
    PRESSURE = auto()
    TEMPERATURE = auto()
    VISCOSITY = auto()
    FLOW_RATE = auto()
    TIME = auto()
    VELOCITY = auto()
    ENERGY = auto()
    POWER = auto()
    FORCE = auto()


# Define unit conversion factors (to SI)
LENGTH_CONVERSIONS = {
    "m": 1.0,  # meter (SI)
    "cm": 0.01,  # centimeter
    "mm": 0.001,  # millimeter
    "km": 1000.0,  # kilometer
    "in": 0.0254,  # inch
    "ft": 0.3048,  # foot
    "yd": 0.9144,  # yard
    "mi": 1609.344  # mile
}

AREA_CONVERSIONS = {
    "m2": 1.0,  # square meter (SI)
    "cm2": 0.0001,  # square centimeter
    "mm2": 0.000001,  # square millimeter
    "km2": 1000000.0,  # square kilometer
    "in2": 0.00064516,  # square inch
    "ft2": 0.09290304,  # square foot
    "yd2": 0.83612736,  # square yard
    "ac": 4046.8564224,  # acre
    "ha": 10000.0  # hectare
}

VOLUME_CONVERSIONS = {
    "m3": 1.0,  # cubic meter (SI)
    "cm3": 0.000001,  # cubic centimeter
    "mm3": 1e-9,  # cubic millimeter
    "L": 0.001,  # liter
    "mL": 0.000001,  # milliliter
    "in3": 1.6387064e-5,  # cubic inch
    "ft3": 0.028316846592,  # cubic foot
    "gal": 0.003785411784,  # US gallon
    "bbl": 0.158987294928,  # oil barrel
    "MMscf": 28316.846592  # million standard cubic feet
}

MASS_CONVERSIONS = {
    "kg": 1.0,  # kilogram (SI)
    "g": 0.001,  # gram
    "mg": 0.000001,  # milligram
    "ton": 1000.0,  # metric ton
    "lb": 0.45359237,  # pound
    "oz": 0.028349523125,  # ounce
    "ton_us": 907.18474,  # US short ton
    "ton_uk": 1016.0469088  # UK long ton
}

DENSITY_CONVERSIONS = {
    "kg/m3": 1.0,  # kilogram per cubic meter (SI)
    "g/cm3": 1000.0,  # gram per cubic centimeter
    "g/mL": 1000.0,  # gram per milliliter
    "kg/L": 1000.0,  # kilogram per liter
    "lb/ft3": 16.01846337,  # pound per cubic foot
    "lb/gal": 119.8264273,  # pound per gallon
    "API": None,  # API gravity (special conversion)
    "SG": 1000.0  # specific gravity (relative to water)
}

PRESSURE_CONVERSIONS = {
    "Pa": 1.0,  # pascal (SI)
    "kPa": 1000.0,  # kilopascal
    "MPa": 1000000.0,  # megapascal
    "bar": 100000.0,  # bar
    "atm": 101325.0,  # atmosphere
    "psi": 6894.75729,  # pound per square inch
    "psf": 47.88025898,  # pound per square foot
    "torr": 133.322,  # torr (mmHg)
    "inHg": 3386.389,  # inch of mercury
    "inH2O": 249.08891  # inch of water
}

TEMPERATURE_CONVERSIONS = {
    "K": None,  # kelvin (SI) - special conversion
    "C": None,  # celsius - special conversion
    "F": None,  # fahrenheit - special conversion
    "R": None   # rankine - special conversion
}

VISCOSITY_CONVERSIONS = {
    "Pa*s": 1.0,  # pascal-second (SI)
    "cP": 0.001,  # centipoise
    "P": 0.1,  # poise
    "lb/ft-s": 1.4881639,  # pound per foot-second
    "lb/ft-h": 0.0004133789  # pound per foot-hour
}

FLOW_RATE_CONVERSIONS = {
    "m3/s": 1.0,  # cubic meter per second (SI)
    "m3/h": 1.0/3600.0,  # cubic meter per hour
    "L/s": 0.001,  # liter per second
    "L/min": 1.6666667e-5,  # liter per minute
    "ft3/s": 0.028316846592,  # cubic foot per second
    "ft3/min": 0.00047194744,  # cubic foot per minute (CFM)
    "gal/min": 6.30901964e-5,  # gallon per minute (GPM)
    "bbl/d": 1.84012e-6,  # barrel per day
    "MMscfd": 0.3277413  # million standard cubic feet per day
}

TIME_CONVERSIONS = {
    "s": 1.0,  # second (SI)
    "min": 60.0,  # minute
    "h": 3600.0,  # hour
    "d": 86400.0,  # day
    "w": 604800.0,  # week
    "mo": 2592000.0,  # month (30 days)
    "y": 31536000.0  # year (365 days)
}

VELOCITY_CONVERSIONS = {
    "m/s": 1.0,  # meter per second (SI)
    "km/h": 0.277777778,  # kilometer per hour
    "cm/s": 0.01,  # centimeter per second
    "ft/s": 0.3048,  # foot per second
    "ft/min": 0.00508,  # foot per minute
    "mi/h": 0.44704  # mile per hour (mph)
}

ENERGY_CONVERSIONS = {
    "J": 1.0,  # joule (SI)
    "kJ": 1000.0,  # kilojoule
    "MJ": 1000000.0,  # megajoule
    "cal": 4.184,  # calorie
    "kcal": 4184.0,  # kilocalorie
    "Btu": 1055.05585262,  # British thermal unit
    "ft-lb": 1.3558179483314  # foot-pound
}

POWER_CONVERSIONS = {
    "W": 1.0,  # watt (SI)
    "kW": 1000.0,  # kilowatt
    "MW": 1000000.0,  # megawatt
    "hp": 745.699872,  # horsepower
    "Btu/h": 0.29307107,  # British thermal unit per hour
    "ft-lb/s": 1.3558179483314  # foot-pound per second
}

FORCE_CONVERSIONS = {
    "N": 1.0,  # newton (SI)
    "kN": 1000.0,  # kilonewton
    "lbf": 4.4482216153,  # pound-force
    "kgf": 9.80665,  # kilogram-force
    "dyn": 0.00001  # dyne
}


# Default units for each physical quantity in different unit systems
DEFAULT_UNITS = {
    UnitSystem.SI: {
        PhysicalQuantity.LENGTH: "m",
        PhysicalQuantity.AREA: "m2",
        PhysicalQuantity.VOLUME: "m3",
        PhysicalQuantity.MASS: "kg",
        PhysicalQuantity.DENSITY: "kg/m3",
        PhysicalQuantity.PRESSURE: "Pa",
        PhysicalQuantity.TEMPERATURE: "K",
        PhysicalQuantity.VISCOSITY: "Pa*s",
        PhysicalQuantity.FLOW_RATE: "m3/s",
        PhysicalQuantity.TIME: "s",
        PhysicalQuantity.VELOCITY: "m/s",
        PhysicalQuantity.ENERGY: "J",
        PhysicalQuantity.POWER: "W",
        PhysicalQuantity.FORCE: "N"
    },
    UnitSystem.IMPERIAL: {
        PhysicalQuantity.LENGTH: "ft",
        PhysicalQuantity.AREA: "ft2",
        PhysicalQuantity.VOLUME: "ft3",
        PhysicalQuantity.MASS: "lb",
        PhysicalQuantity.DENSITY: "lb/ft3",
        PhysicalQuantity.PRESSURE: "psi",
        PhysicalQuantity.TEMPERATURE: "F",
        PhysicalQuantity.VISCOSITY: "lb/ft-s",
        PhysicalQuantity.FLOW_RATE: "ft3/min",
        PhysicalQuantity.TIME: "h",
        PhysicalQuantity.VELOCITY: "ft/s",
        PhysicalQuantity.ENERGY: "Btu",
        PhysicalQuantity.POWER: "hp",
        PhysicalQuantity.FORCE: "lbf"
    },
    UnitSystem.FIELD: {
        PhysicalQuantity.LENGTH: "ft",
        PhysicalQuantity.AREA: "ft2",
        PhysicalQuantity.VOLUME: "bbl",
        PhysicalQuantity.MASS: "lb",
        PhysicalQuantity.DENSITY: "lb/ft3",
        PhysicalQuantity.PRESSURE: "psi",
        PhysicalQuantity.TEMPERATURE: "F",
        PhysicalQuantity.VISCOSITY: "cP",
        PhysicalQuantity.FLOW_RATE: "bbl/d",
        PhysicalQuantity.TIME: "d",
        PhysicalQuantity.VELOCITY: "ft/s",
        PhysicalQuantity.ENERGY: "Btu",
        PhysicalQuantity.POWER: "hp",
        PhysicalQuantity.FORCE: "lbf"
    }
}


# Mapping of quantity types to their conversion dictionaries
CONVERSION_MAPS = {
    PhysicalQuantity.LENGTH: LENGTH_CONVERSIONS,
    PhysicalQuantity.AREA: AREA_CONVERSIONS,
    PhysicalQuantity.VOLUME: VOLUME_CONVERSIONS,
    PhysicalQuantity.MASS: MASS_CONVERSIONS,
    PhysicalQuantity.DENSITY: DENSITY_CONVERSIONS,
    PhysicalQuantity.PRESSURE: PRESSURE_CONVERSIONS,
    PhysicalQuantity.TEMPERATURE: TEMPERATURE_CONVERSIONS,
    PhysicalQuantity.VISCOSITY: VISCOSITY_CONVERSIONS,
    PhysicalQuantity.FLOW_RATE: FLOW_RATE_CONVERSIONS,
    PhysicalQuantity.TIME: TIME_CONVERSIONS,
    PhysicalQuantity.VELOCITY: VELOCITY_CONVERSIONS,
    PhysicalQuantity.ENERGY: ENERGY_CONVERSIONS,
    PhysicalQuantity.POWER: POWER_CONVERSIONS,
    PhysicalQuantity.FORCE: FORCE_CONVERSIONS
}


# Special conversion functions for temperature
def kelvin_to_celsius(value):
    """Convert temperature from Kelvin to Celsius."""
    return value - 273.15


def celsius_to_kelvin(value):
    """Convert temperature from Celsius to Kelvin."""
    return value + 273.15


def kelvin_to_fahrenheit(value):
    """Convert temperature from Kelvin to Fahrenheit."""
    return (value * 9/5) - 459.67


def fahrenheit_to_kelvin(value):
    """Convert temperature from Fahrenheit to Kelvin."""
    return (value + 459.67) * 5/9


def kelvin_to_rankine(value):
    """Convert temperature from Kelvin to Rankine."""
    return value * 9/5


def rankine_to_kelvin(value):
    """Convert temperature from Rankine to Kelvin."""
    return value * 5/9


def celsius_to_fahrenheit(value):
    """Convert temperature from Celsius to Fahrenheit."""
    return (value * 9/5) + 32


def fahrenheit_to_celsius(value):
    """Convert temperature from Fahrenheit to Celsius."""
    return (value - 32) * 5/9


# Special conversion functions for API gravity
def density_to_api(density_kg_m3):
    """Convert density in kg/m³ to API gravity."""
    # API gravity = (141.5 / SG) - 131.5
    # SG = density / 999.016 (density of water at 60°F in kg/m³)
    if density_kg_m3 <= 0:
        return 0  # Avoid division by zero
    sg = density_kg_m3 / 999.016
    return (141.5 / sg) - 131.5


def api_to_density(api):
    """Convert API gravity to density in kg/m³."""
    # SG = 141.5 / (api + 131.5)
    # density = SG * 999.016 (density of water at 60°F in kg/m³)
    if api <= 0:
        return 0  # Invalid API gravity
    sg = 141.5 / (api + 131.5)
    return sg * 999.016


class UnitConverter:
    """
    Utility class for converting between different unit systems and formats.
    """
    
    def __init__(self, default_system: UnitSystem = UnitSystem.SI):
        """
        Initialize the unit converter.
        
        Args:
            default_system: The default unit system to use
        """
        self.default_system = default_system
    
    def convert(self, value: float, from_unit: str, to_unit: str, 
               quantity: Optional[PhysicalQuantity] = None) -> float:
        """
        Convert a value from one unit to another.
        
        Args:
            value: The value to convert
            from_unit: The unit to convert from
            to_unit: The unit to convert to
            quantity: The physical quantity (optional, used to determine the conversion
                     method if special handling is needed)
                     
        Returns:
            float: The converted value
        """
        # Handle temperature conversions specially
        if from_unit in TEMPERATURE_CONVERSIONS and to_unit in TEMPERATURE_CONVERSIONS:
            return self._convert_temperature(value, from_unit, to_unit)
        
        # Handle API gravity conversions specially
        if (from_unit == "API" or to_unit == "API") and quantity == PhysicalQuantity.DENSITY:
            return self._convert_api_gravity(value, from_unit, to_unit)
        
        # Standard unit conversion through SI units
        if quantity is None:
            # Try to determine quantity from units
            quantity = self._guess_quantity(from_unit, to_unit)
            if quantity is None:
                raise ValueError(f"Unable to determine physical quantity for units {from_unit} and {to_unit}")
                
        conversion_map = CONVERSION_MAPS[quantity]
        
        if from_unit not in conversion_map:
            raise ValueError(f"Unknown source unit: {from_unit}")
        if to_unit not in conversion_map:
            raise ValueError(f"Unknown target unit: {to_unit}")
            
        # Convert to SI, then to target unit
        si_value = value * conversion_map[from_unit]
        return si_value / conversion_map[to_unit]
    
    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert temperature values between different units.
        
        Args:
            value: The temperature value to convert
            from_unit: The unit to convert from ('K', 'C', 'F', or 'R')
            to_unit: The unit to convert to ('K', 'C', 'F', or 'R')
            
        Returns:
            float: The converted temperature value
        """
        # First convert to Kelvin
        if from_unit == "K":
            kelvin = value
        elif from_unit == "C":
            kelvin = celsius_to_kelvin(value)
        elif from_unit == "F":
            kelvin = fahrenheit_to_kelvin(value)
        elif from_unit == "R":
            kelvin = rankine_to_kelvin(value)
        else:
            raise ValueError(f"Unknown temperature unit: {from_unit}")
        
        # Then convert from Kelvin to target unit
        if to_unit == "K":
            return kelvin
        elif to_unit == "C":
            return kelvin_to_celsius(kelvin)
        elif to_unit == "F":
            return kelvin_to_fahrenheit(kelvin)
        elif to_unit == "R":
            return kelvin_to_rankine(kelvin)
        else:
            raise ValueError(f"Unknown temperature unit: {to_unit}")
    
    def _convert_api_gravity(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert between API gravity and density.
        
        Args:
            value: The value to convert
            from_unit: The unit to convert from
            to_unit: The unit to convert to
            
        Returns:
            float: The converted value
        """
        if from_unit == "API" and to_unit == "kg/m3":
            return api_to_density(value)
        elif from_unit == "kg/m3" and to_unit == "API":
            return density_to_api(value)
        elif from_unit == "API":
            # Convert API to kg/m3, then to target unit
            density = api_to_density(value)
            return self.convert(density, "kg/m3", to_unit, PhysicalQuantity.DENSITY)
        elif to_unit == "API":
            # Convert from source unit to kg/m3, then to API
            density = self.convert(value, from_unit, "kg/m3", PhysicalQuantity.DENSITY)
            return density_to_api(density)
    
    def _guess_quantity(self, from_unit: str, to_unit: str) -> Optional[PhysicalQuantity]:
        """
        Try to guess the physical quantity based on the units.
        
        Args:
            from_unit: The unit to convert from
            to_unit: The unit to convert to
            
        Returns:
            PhysicalQuantity: The guessed physical quantity, or None if not determined
        """
        # Check each conversion map for the units
        for quantity, conversion_map in CONVERSION_MAPS.items():
            if from_unit in conversion_map and to_unit in conversion_map:
                return quantity
        return None
    
    def get_default_unit(self, quantity: PhysicalQuantity, 
                        system: Optional[UnitSystem] = None) -> str:
        """
        Get the default unit for a physical quantity in the specified unit system.
        
        Args:
            quantity: The physical quantity
            system: The unit system (defaults to the converter's default system)
            
        Returns:
            str: The default unit
        """
        if system is None:
            system = self.default_system
            
        return DEFAULT_UNITS[system][quantity]
    
    def set_default_system(self, system: UnitSystem):
        """
        Set the default unit system.
        
        Args:
            system: The unit system to set as default
        """
        self.default_system = system
    
    def format_value(self, value: float, unit: str, precision: int = 3) -> str:
        """
        Format a value with its unit.
        
        Args:
            value: The value to format
            unit: The unit symbol
            precision: Number of decimal places
            
        Returns:
            str: The formatted value with unit
        """
        if abs(value) < 0.001 or abs(value) >= 1000:
            return f"{value:.{precision}e} {unit}"
        else:
            return f"{value:.{precision}f} {unit}"
    
    def parse_value(self, text: str) -> Tuple[float, str]:
        """
        Parse a value with unit from a string.
        
        Args:
            text: Text containing a value and unit
            
        Returns:
            Tuple[float, str]: The parsed value and unit
        """
        # Regular expression to match a number followed by optional whitespace and a unit
        match = re.match(r'^\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*(.*)$', text.strip())
        
        if not match:
            raise ValueError(f"Unable to parse value and unit from: {text}")
            
        value_str, unit = match.groups()
        value = float(value_str)
        unit = unit.strip()
        
        return value, unit
    
    def get_quantity_for_unit(self, unit: str) -> Optional[PhysicalQuantity]:
        """
        Determine the physical quantity for a given unit.
        
        Args:
            unit: The unit to check
            
        Returns:
            PhysicalQuantity: The corresponding physical quantity, or None if not found
        """
        for quantity, conversion_map in CONVERSION_MAPS.items():
            if unit in conversion_map:
                return quantity
        return None
    
    def get_all_units(self, quantity: PhysicalQuantity) -> List[str]:
        """
        Get all available units for a given physical quantity.
        
        Args:
            quantity: The physical quantity
            
        Returns:
            List[str]: List of all available units
        """
        conversion_map = CONVERSION_MAPS[quantity]
        return list(conversion_map.keys())
    
    def is_valid_unit(self, unit: str) -> bool:
        """
        Check if a unit is valid.
        
        Args:
            unit: The unit to check
            
        Returns:
            bool: True if the unit is valid, False otherwise
        """
        for conversion_map in CONVERSION_MAPS.values():
            if unit in conversion_map:
                return True
        return False


# Singleton instance
_converter = None


def get_converter() -> UnitConverter:
    """
    Get the singleton UnitConverter instance.
    
    Returns:
        UnitConverter: The singleton instance
    """
    global _converter
    if _converter is None:
        # Initialize with default system from configuration if available
        from ..config import get_value
        
        # Get unit system from config, default to SI if not found
        units_str = get_value('industry.units', 'metric')
        if units_str == 'imperial':
            system = UnitSystem.IMPERIAL
        elif units_str == 'field':
            system = UnitSystem.FIELD
        else:  # Default to metric/SI
            system = UnitSystem.SI
            
        _converter = UnitConverter(system)
        
    return _converter


def convert(value: float, from_unit: str, to_unit: str, 
           quantity: Optional[PhysicalQuantity] = None) -> float:
    """
    Convenience function to convert a value from one unit to another.
    
    Args:
        value: The value to convert
        from_unit: The unit to convert from
        to_unit: The unit to convert to
        quantity: The physical quantity (optional)
        
    Returns:
        float: The converted value
    """
    return get_converter().convert(value, from_unit, to_unit, quantity)


def format_value(value: float, unit: str, precision: int = 3) -> str:
    """
    Convenience function to format a value with its unit.
    
    Args:
        value: The value to format
        unit: The unit symbol
        precision: Number of decimal places
        
    Returns:
        str: The formatted value with unit
    """
    return get_converter().format_value(value, unit, precision)


def get_default_unit(quantity: PhysicalQuantity, system: Optional[UnitSystem] = None) -> str:
    """
    Convenience function to get the default unit for a physical quantity.
    
    Args:
        quantity: The physical quantity
        system: The unit system (optional)
        
    Returns:
        str: The default unit
    """
    return get_converter().get_default_unit(quantity, system)


# Example usage
if __name__ == "__main__":
    # Convert pressure from psi to bar
    pressure_in_psi = 2000.0
    pressure_in_bar = convert(pressure_in_psi, "psi", "bar", PhysicalQuantity.PRESSURE)
    print(f"{pressure_in_psi} psi = {pressure_in_bar} bar")
    
    # Convert temperature from Fahrenheit to Celsius
    temp_f = 212.0
    temp_c = convert(temp_f, "F", "C")
    print(f"{temp_f} °F = {temp_c} °C")
    
    # Convert flow rate from barrels per day to cubic meters per hour
    flow_bpd = 1000.0
    flow_m3h = convert(flow_bpd, "bbl/d", "m3/h", PhysicalQuantity.FLOW_RATE)
    print(f"{flow_bpd} bbl/d = {flow_m3h} m3/h")
    
    # Format a value with its unit
    formatted = format_value(pressure_in_bar, "bar")
    print(f"Formatted: {formatted}")
    
    # Get the default pressure unit in field units
    field_pressure_unit = get_default_unit(PhysicalQuantity.PRESSURE, UnitSystem.FIELD)
    print(f"Default pressure unit in field units: {field_pressure_unit}")