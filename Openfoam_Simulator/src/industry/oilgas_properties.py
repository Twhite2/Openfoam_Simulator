#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oil & Gas physical properties for Openfoam_Simulator.

This module provides physical property calculations and models for oil and gas fluids
including:
- PVT properties (pressure-volume-temperature relationships)
- Phase behavior
- Viscosity models
- Density correlations
- Thermal properties
- Surface tension models
- Fluid composition models
"""

import math
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union, Any

# Import utility modules
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FluidType(Enum):
    """Enumeration of fluid types."""
    CRUDE_OIL = "Crude Oil"
    CONDENSATE = "Condensate"
    NATURAL_GAS = "Natural Gas"
    WATER = "Water"
    BRINE = "Brine"
    PRODUCED_WATER = "Produced Water"
    DRILLING_MUD = "Drilling Mud"
    DIESEL = "Diesel"
    GASOLINE = "Gasoline"
    JET_FUEL = "Jet Fuel"
    GLYCOL = "Glycol"
    METHANOL = "Methanol"
    CUSTOM = "Custom"


class PVTModel(Enum):
    """Enumeration of PVT models."""
    IDEAL_GAS = "Ideal Gas"
    PENG_ROBINSON = "Peng-Robinson"
    SOAVE_REDLICH_KWONG = "Soave-Redlich-Kwong"
    STANDING = "Standing"
    BEGGS_ROBINSON = "Beggs-Robinson"
    VASQUEZ_BEGGS = "Vasquez-Beggs"
    BLACK_OIL = "Black Oil"
    COMPOSITIONAL = "Compositional"


class ViscosityModel(Enum):
    """Enumeration of viscosity models."""
    CONSTANT = "Constant"
    BEAL = "Beal"
    BEGGS_ROBINSON = "Beggs-Robinson"
    LEE_GONZALEZ_EAKIN = "Lee-Gonzalez-Eakin"
    SUTTON = "Sutton"
    LOHRENZ_BRAY_CLARK = "Lohrenz-Bray-Clark"


class FluidPhase(Enum):
    """Enumeration of fluid phases."""
    LIQUID = "Liquid"
    GAS = "Gas"
    SUPERCRITICAL = "Supercritical"
    SUBCRITICAL = "Subcritical"


class OilAPIGravity:
    """Classification of oil by API gravity."""
    EXTRA_HEAVY = "Extra Heavy Oil"  # < 10°API
    HEAVY = "Heavy Oil"              # 10-22.3°API
    MEDIUM = "Medium Oil"            # 22.3-31.1°API
    LIGHT = "Light Oil"              # 31.1-47°API
    CONDENSATE = "Condensate"        # > 47°API

    @staticmethod
    def classify(api_gravity: float) -> str:
        """
        Classify oil based on API gravity.
        
        Args:
            api_gravity (float): API gravity value
            
        Returns:
            str: Oil classification
        """
        if api_gravity < 10.0:
            return OilAPIGravity.EXTRA_HEAVY
        elif api_gravity < 22.3:
            return OilAPIGravity.HEAVY
        elif api_gravity < 31.1:
            return OilAPIGravity.MEDIUM
        elif api_gravity < 47.0:
            return OilAPIGravity.LIGHT
        else:
            return OilAPIGravity.CONDENSATE


# Standard reference conditions
class ReferenceConditions:
    """Standard reference conditions for oil and gas properties."""
    # Standard temperature and pressure
    STANDARD_TEMPERATURE_C = 15.0
    STANDARD_TEMPERATURE_F = 59.0
    STANDARD_TEMPERATURE_K = 288.15
    STANDARD_PRESSURE_PSI = 14.7
    STANDARD_PRESSURE_KPA = 101.325
    STANDARD_PRESSURE_BAR = 1.01325
    
    # API standard conditions
    API_TEMPERATURE_F = 60.0
    API_PRESSURE_PSI = 14.7
    
    # Critical point of water
    WATER_CRITICAL_TEMPERATURE_C = 374.0
    WATER_CRITICAL_PRESSURE_BAR = 221.2


# Base class for fluid properties
class FluidProperties:
    """Base class for fluid properties."""
    
    def __init__(self, name: str, fluid_type: FluidType = FluidType.CUSTOM):
        """
        Initialize fluid properties.
        
        Args:
            name (str): Name of the fluid
            fluid_type (FluidType): Type of fluid
        """
        self.name = name
        self.fluid_type = fluid_type
        
        # Default properties at standard conditions
        self.reference_temperature = ReferenceConditions.STANDARD_TEMPERATURE_K
        self.reference_pressure = ReferenceConditions.STANDARD_PRESSURE_BAR
        
        # Base physical properties (to be overridden by subclasses)
        self.density = None  # kg/m³
        self.viscosity = None  # Pa·s
        self.specific_heat = None  # J/(kg·K)
        self.thermal_conductivity = None  # W/(m·K)
        self.surface_tension = None  # N/m
        
        # Models (to be set by subclasses)
        self.pvt_model = None
        self.viscosity_model = None
        
        # Custom properties dictionary
        self.custom_properties = {}
    
    def density_at_conditions(self, temperature: float, pressure: float) -> float:
        """
        Calculate density at specified temperature and pressure.
        
        Args:
            temperature (float): Temperature in K
            pressure (float): Pressure in bar
            
        Returns:
            float: Density in kg/m³
            
        Raises:
            NotImplementedError: This method should be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement density_at_conditions")
    
    def viscosity_at_conditions(self, temperature: float, pressure: float) -> float:
        """
        Calculate viscosity at specified temperature and pressure.
        
        Args:
            temperature (float): Temperature in K
            pressure (float): Pressure in bar
            
        Returns:
            float: Viscosity in Pa·s
            
        Raises:
            NotImplementedError: This method should be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement viscosity_at_conditions")
    
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
    
    def get_phase(self, temperature: float, pressure: float) -> FluidPhase:
        """
        Determine the phase of the fluid at given conditions.
        
        Args:
            temperature (float): Temperature in K
            pressure (float): Pressure in bar
            
        Returns:
            FluidPhase: Phase of the fluid
            
        Raises:
            NotImplementedError: This method should be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement get_phase")


class OilProperties(FluidProperties):
    """Class for oil properties."""
    
    def __init__(self, name: str, api_gravity: float, 
                gas_oil_ratio: float = 0.0, 
                fluid_type: FluidType = FluidType.CRUDE_OIL):
        """
        Initialize oil properties.
        
        Args:
            name (str): Name of the oil
            api_gravity (float): API gravity
            gas_oil_ratio (float, optional): Gas-oil ratio in scf/STB
            fluid_type (FluidType, optional): Type of fluid
        """
        super().__init__(name, fluid_type)
        
        self.api_gravity = api_gravity
        self.gas_oil_ratio = gas_oil_ratio  # scf/STB
        
        # Calculate specific gravity from API gravity
        self.specific_gravity = 141.5 / (api_gravity + 131.5)
        
        # Classification
        self.classification = OilAPIGravity.classify(api_gravity)
        
        # Set initial property models
        self.pvt_model = PVTModel.BLACK_OIL
        self.viscosity_model = ViscosityModel.BEGGS_ROBINSON
        
        # Set default properties based on API gravity
        self._set_default_properties()
        
        # Formation volume factor (FVF)
        self.formation_volume_factor = 1.0  # bbl/STB
        
        # Bubble point pressure
        self.bubble_point_pressure = None  # psia
        
        # Compressibility
        self.compressibility = None  # 1/psi
    
    def _set_default_properties(self):
        """Set default properties based on API gravity."""
        # Density calculation using specific gravity
        self.density = self.specific_gravity * 1000.0  # kg/m³
        
        # Dead oil viscosity correlation (Beggs-Robinson)
        # Valid for API gravities between 16 and 58
        temp_f = (self.reference_temperature - 273.15) * 9/5 + 32  # Convert K to °F
        
        # Calculate viscosity at reference temperature
        z = 3.0324 - 0.02023 * self.api_gravity
        y = 10**z
        x = y * (temp_f**-1.163)
        self.viscosity = x * 0.001  # Convert from cP to Pa·s
        
        # Specific heat approximation (modified Einstein correlation)
        self.specific_heat = 1670.0 + 3.40 * self.api_gravity  # J/(kg·K)
        
        # Thermal conductivity approximation
        self.thermal_conductivity = 0.13 - 0.0007 * self.api_gravity  # W/(m·K)
        
        # Surface tension approximation
        self.surface_tension = 0.025  # N/m (oil-water)
    
    def density_at_conditions(self, temperature: float, pressure: float) -> float:
        """
        Calculate oil density at specified temperature and pressure.
        
        Args:
            temperature (float): Temperature in K
            pressure (float): Pressure in bar
            
        Returns:
            float: Density in kg/m³
        """
        # Convert to compatible units
        temp_f = (temperature - 273.15) * 9/5 + 32  # K to °F
        pressure_psi = pressure * 14.5038  # bar to psi
        
        # Thermal expansion coefficient (approximation for light crude)
        alpha = 0.0008  # 1/°F, thermal expansion coefficient
        
        # Compressibility factor (approximation)
        compressibility = 1e-6  # 1/psi
        
        # Reference density at standard conditions
        density_std = self.density
        
        # Density adjustment for temperature and pressure
        # Using the simple relationship: rho = rho_std * (1 + c*(p-p_std) - alpha*(T-T_std))
        temp_ref_f = (self.reference_temperature - 273.15) * 9/5 + 32
        pressure_ref_psi = self.reference_pressure * 14.5038
        
        density = density_std * (1.0 + compressibility * (pressure_psi - pressure_ref_psi) - 
                                alpha * (temp_f - temp_ref_f))
        
        return density
    
    def viscosity_at_conditions(self, temperature: float, pressure: float) -> float:
        """
        Calculate oil viscosity at specified temperature and pressure.
        
        Args:
            temperature (float): Temperature in K
            pressure (float): Pressure in bar
            
        Returns:
            float: Viscosity in Pa·s
        """
        # Convert to compatible units
        temp_f = (temperature - 273.15) * 9/5 + 32  # K to °F
        pressure_psi = pressure * 14.5038  # bar to psi
        
        # Calculate dead oil viscosity using Beggs-Robinson correlation
        z = 3.0324 - 0.02023 * self.api_gravity
        y = 10**z
        x = y * (temp_f**-1.163)
        visc_dead_oil = x  # in cP
        
        # Apply pressure correction if above bubble point
        # This is a simplified approach; in practice, more complex models would be used
        # especially for live oils with high GOR
        if self.bubble_point_pressure is not None and pressure_psi > self.bubble_point_pressure:
            # Simplified pressure correction
            m = 2.6 * pressure_psi**1.187 * math.exp(-11.513 - 8.98e-5 * pressure_psi)
            visc_live_oil = visc_dead_oil * (pressure_psi / self.bubble_point_pressure)**m
        else:
            # Below bubble point, use dead oil viscosity
            visc_live_oil = visc_dead_oil
        
        # Convert from cP to Pa·s
        return visc_live_oil * 0.001
    
    def get_phase(self, temperature: float, pressure: float) -> FluidPhase:
        """
        Determine the phase of the oil at given conditions.
        
        Args:
            temperature (float): Temperature in K
            pressure (float): Pressure in bar
            
        Returns:
            FluidPhase: Phase of the fluid
        """
        # This is a simplified approach for demonstration
        # In practice, this would involve more complex phase behavior calculations
        
        # Convert to compatible units
        pressure_psi = pressure * 14.5038  # bar to psi
        
        # Check against bubble point pressure
        if self.bubble_point_pressure is None:
            # Calculate bubble point using Standing's correlation if not set
            temp_f = (temperature - 273.15) * 9/5 + 32  # K to °F
            gas_gravity = 0.65  # Assumed gas gravity
            self.bubble_point_pressure = 18.2 * ((self.gas_oil_ratio / gas_gravity)**0.83 * 
                                               10**(0.00091 * temp_f - 0.0125 * self.api_gravity) - 1.4)
        
        if pressure_psi < self.bubble_point_pressure:
            return FluidPhase.LIQUID  # Below bubble point, liquid with dissolved gas
        else:
            return FluidPhase.LIQUID  # Above bubble point, undersaturated oil
    
    def calculate_bubble_point(self, temperature: float, gas_gravity: float = 0.65) -> float:
        """
        Calculate bubble point pressure using Standing's correlation.
        
        Args:
            temperature (float): Temperature in K
            gas_gravity (float): Gas specific gravity (air = 1.0)
            
        Returns:
            float: Bubble point pressure in psia
        """
        temp_f = (temperature - 273.15) * 9/5 + 32  # K to °F
        
        # Standing's correlation
        bubble_point = 18.2 * ((self.gas_oil_ratio / gas_gravity)**0.83 * 
                              10**(0.00091 * temp_f - 0.0125 * self.api_gravity) - 1.4)
        
        self.bubble_point_pressure = bubble_point
        return bubble_point
    
    def calculate_formation_volume_factor(self, pressure: float, temperature: float) -> float:
        """
        Calculate oil formation volume factor (FVF) using Standing's correlation.
        
        Args:
            pressure (float): Pressure in bar
            temperature (float): Temperature in K
            
        Returns:
            float: Formation volume factor (bbl/STB)
        """
        # Convert units
        pressure_psi = pressure * 14.5038  # bar to psi
        temp_f = (temperature - 273.15) * 9/5 + 32  # K to °F
        
        # Calculate gas solubility if below bubble point
        rs = self.gas_oil_ratio
        if self.bubble_point_pressure is not None and pressure_psi < self.bubble_point_pressure:
            # Reduced gas solubility below bubble point
            rs = self.gas_oil_ratio * (pressure_psi / self.bubble_point_pressure)
        
        # Gas gravity (assumed)
        gas_gravity = 0.65
        
        # Standing's correlation for FVF
        fvf = 0.9759 + 0.00012 * (rs * (gas_gravity / self.specific_gravity)**0.5 + 
                                1.25 * temp_f)**1.2
        
        self.formation_volume_factor = fvf
        return fvf


class GasProperties(FluidProperties):
    """Class for natural gas properties."""
    
    def __init__(self, name: str, specific_gravity: float, 
                fluid_type: FluidType = FluidType.NATURAL_GAS):
        """
        Initialize gas properties.
        
        Args:
            name (str): Name of the gas
            specific_gravity (float): Gas specific gravity (air = 1.0)
            fluid_type (FluidType, optional): Type of fluid
        """
        super().__init__(name, fluid_type)
        
        self.specific_gravity = specific_gravity
        
        # Critical properties (estimated using correlations)
        self.critical_temperature = None  # K
        self.critical_pressure = None  # bar
        self.pseudo_critical_temperature = None  # K
        self.pseudo_critical_pressure = None  # bar
        
        # Set default property models
        self.pvt_model = PVTModel.PENG_ROBINSON
        self.viscosity_model = ViscosityModel.LEE_GONZALEZ_EAKIN
        
        # Estimate properties based on specific gravity
        self._estimate_critical_properties()
        self._set_default_properties()
        
        # Compressibility factor (z-factor)
        self.z_factor = 1.0
        
        # Composition (for compositional analysis)
        self.composition = {}  # Component name to mole fraction
    
    def _estimate_critical_properties(self):
        """Estimate critical properties based on specific gravity."""
        # Sutton correlations for critical properties
        # Based on gas specific gravity
        self.critical_temperature = 126.66 + 90.7 * self.specific_gravity  # K
        self.critical_pressure = 33.14 + 28.99 * self.specific_gravity  # bar
        
        # Pseudo-critical properties (for sweet gas)
        self.pseudo_critical_temperature = self.critical_temperature  # K
        self.pseudo_critical_pressure = self.critical_pressure  # bar
    
    def _set_default_properties(self):
        """Set default properties based on specific gravity."""
        # Density at standard conditions (ideal gas)
        # ρ = P*MW/(Z*R*T)
        mw = self.specific_gravity * 28.97  # g/mol (molecular weight, air = 28.97)
        r_const = 8.314  # J/(mol·K) (gas constant)
        self.density = (self.reference_pressure * 1e5 * mw) / (self.z_factor * r_const * self.reference_temperature) / 1000
        
        # Viscosity using Lee-Gonzalez-Eakin correlation
        temp_r = self.reference_temperature * 9/5  # K to °R
        self.viscosity = self._lee_gonzalez_eakin_viscosity(self.density, temp_r, mw)
        
        # Specific heat approximation
        self.specific_heat = 2200.0  # J/(kg·K)
        
        # Thermal conductivity approximation
        self.thermal_conductivity = 0.03  # W/(m·K)
        
        # Surface tension is very low for gases
        self.surface_tension = 0.0  # N/m
    
    def _lee_gonzalez_eakin_viscosity(self, density: float, temperature_r: float, 
                                     molecular_weight: float) -> float:
        """
        Calculate gas viscosity using Lee-Gonzalez-Eakin correlation.
        
        Args:
            density (float): Density in kg/m³
            temperature_r (float): Temperature in °R
            molecular_weight (float): Molecular weight in g/mol
            
        Returns:
            float: Viscosity in Pa·s
        """
        # Convert density to g/cc for correlation
        density_gcc = density / 1000.0
        
        # Lee-Gonzalez-Eakin correlation
        k = (9.4 + 0.02 * molecular_weight) * temperature_r**1.5 / (209 + 19 * molecular_weight + temperature_r)
        x = 3.5 + 986 / temperature_r + 0.01 * molecular_weight
        y = 2.4 - 0.2 * x
        
        # Calculate viscosity in cP
        viscosity_cp = 1e-4 * k * math.exp(x * (density_gcc**y))
        
        # Convert to Pa·s
        return viscosity_cp * 0.001
    
    def calculate_z_factor(self, temperature: float, pressure: float) -> float:
        """
        Calculate gas compressibility factor (z-factor) using Standing-Katz correlation.
        
        Args:
            temperature (float): Temperature in K
            pressure (float): Pressure in bar
            
        Returns:
            float: Z-factor (dimensionless)
        """
        # Reduced properties
        if self.pseudo_critical_temperature is None or self.pseudo_critical_pressure is None:
            self._estimate_critical_properties()
            
        t_r = temperature / self.pseudo_critical_temperature
        p_r = pressure / self.pseudo_critical_pressure
        
        # Using simplified approximation
        # In practice, would use Standing-Katz charts or more complex correlations
        if p_r < 1.0:
            # Low pressure approximation
            z = 1.0 - p_r * (0.11 + 0.22 / t_r)
        else:
            # Higher pressure approximation
            A = 1.39 * (t_r - 0.92)**0.5 - 0.36 * t_r - 0.101
            B = (0.62 - 0.23 * t_r) * p_r + (0.066 / (t_r - 0.86) - 0.037) * p_r**2 + 0.32 * p_r**6 / (10**(9 * (t_r - 1)))
            z = A + (1.0 - A) * math.exp(-B) + 0.132 * p_r**4
        
        # Ensure valid range
        z = max(0.2, min(z, 1.5))
        
        self.z_factor = z
        return z
    
    def density_at_conditions(self, temperature: float, pressure: float) -> float:
        """
        Calculate gas density at specified temperature and pressure.
        
        Args:
            temperature (float): Temperature in K
            pressure (float): Pressure in bar
            
        Returns:
            float: Density in kg/m³
        """
        # Calculate z-factor at these conditions
        z = self.calculate_z_factor(temperature, pressure)
        
        # Molecular weight
        mw = self.specific_gravity * 28.97  # g/mol
        
        # Gas density equation: ρ = PM/(ZRT)
        r_const = 8.314  # J/(mol·K)
        density = (pressure * 1e5 * mw) / (z * r_const * temperature) / 1000
        
        return density
    
    def viscosity_at_conditions(self, temperature: float, pressure: float) -> float:
        """
        Calculate gas viscosity at specified temperature and pressure.
        
        Args:
            temperature (float): Temperature in K
            pressure (float): Pressure in bar
            
        Returns:
            float: Viscosity in Pa·s
        """
        # Calculate density at these conditions
        density = self.density_at_conditions(temperature, pressure)
        
        # Convert temperature to Rankine
        temp_r = temperature * 9/5  # K to °R
        
        # Molecular weight
        mw = self.specific_gravity * 28.97  # g/mol
        
        # Calculate viscosity using Lee-Gonzalez-Eakin
        return self._lee_gonzalez_eakin_viscosity(density, temp_r, mw)
    
    def get_phase(self, temperature: float, pressure: float) -> FluidPhase:
        """
        Determine the phase of the gas at given conditions.
        
        Args:
            temperature (float): Temperature in K
            pressure (float): Pressure in bar
            
        Returns:
            FluidPhase: Phase of the fluid
        """
        # Check against critical point
        if self.critical_temperature is None or self.critical_pressure is None:
            self._estimate_critical_properties()
        
        if temperature > self.critical_temperature and pressure > self.critical_pressure:
            return FluidPhase.SUPERCRITICAL
        elif temperature > self.critical_temperature:
            return FluidPhase.GAS  # Supercritical temperature, subcritical pressure
        elif pressure > self.critical_pressure:
            return FluidPhase.LIQUID  # Supercritical pressure, subcritical temperature
        else:
            return FluidPhase.GAS  # Subcritical conditions
    
    def calculate_gas_formation_volume_factor(self, temperature: float, pressure: float) -> float:
        """
        Calculate gas formation volume factor (FVF).
        
        Args:
            temperature (float): Temperature in K
            pressure (float): Pressure in bar
            
        Returns:
            float: Gas FVF in ft³/scf
        """
        # Calculate z-factor
        z = self.calculate_z_factor(temperature, pressure)
        
        # Reference conditions
        p_sc = ReferenceConditions.STANDARD_PRESSURE_BAR  # bar
        t_sc = ReferenceConditions.STANDARD_TEMPERATURE_K  # K
        
        # Gas FVF equation
        # Bg = (0.00504 * z * T) / p  # bbl/scf
        # Convert to ft³/scf
        fvf = 0.0283168 * z * (temperature / t_sc) * (p_sc / pressure)
        
        return fvf
    
    def set_composition(self, composition: Dict[str, float]):
        """
        Set gas composition by component mole fractions.
        
        Args:
            composition (Dict[str, float]): Component name to mole fraction
            
        Raises:
            ValueError: If sum of mole fractions is not close to 1.0
        """
        # Validate composition
        total = sum(composition.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Sum of mole fractions ({total}) must be approximately 1.0")
        
        # Normalize if needed
        if total != 1.0:
            composition = {k: v / total for k, v in composition.items()}
        
        self.composition = composition
        
        # Recalculate properties based on composition
        # This would include critical properties, specific gravity, etc.
        # For simplicity, we're not implementing the full calculations here


class WaterProperties(FluidProperties):
    """Class for water properties."""
    
    def __init__(self, name: str = "Water", 
                salinity: float = 0.0,  # wt% 
                fluid_type: FluidType = FluidType.WATER):
        """
        Initialize water properties.
        
        Args:
            name (str, optional): Name of the water
            salinity (float, optional): Salinity in weight percent
            fluid_type (FluidType, optional): Type of fluid
        """
        super().__init__(name, fluid_type)
        
        self.salinity = salinity  # Weight percent (0-100%)
        
        # Set property models
        self.pvt_model = PVTModel.IDEAL_GAS  # Simplified for water
        self.viscosity_model = ViscosityModel.CONSTANT
        
        # Determine fluid type based on salinity
        if salinity > 1.0:
            self.fluid_type = FluidType.BRINE
        elif salinity > 0.0:
            self.fluid_type = FluidType.PRODUCED_WATER
        
        # Set pure water properties
        self._set_pure_water_properties()
        
        # Adjust for salinity
        self._adjust_for_salinity()
    
    def _set_pure_water_properties(self):
        """Set properties for pure water at reference conditions."""
        # Standard properties at reference conditions
        self.density = 1000.0  # kg/m³
        self.viscosity = 0.001  # Pa·s
        self.specific_heat = 4182.0  # J/(kg·K)
        self.thermal_conductivity = 0.6  # W/(m·K)
        self.surface_tension = 0.072  # N/m
    
    def _adjust_for_salinity(self):
        """Adjust properties based on salinity."""
        if self.salinity <= 0.0:
            return
            
        # Density adjustment for salinity
        # Approximate relation: ρ_brine = ρ_water * (1 + 0.0065 * salinity)
        self.density *= (1.0 + 0.0065 * self.salinity)
        
        # Viscosity adjustment for salinity (simplified)
        # Approximate relation: μ_brine = μ_water * (1 + 0.00214 * salinity)
        self.viscosity *= (1.0 + 0.00214 * self.salinity)
        
        # Specific heat adjustment
        # Approximate relation: cp_brine = cp_water - 4.1868 * salinity
        self.specific_heat -= 4.1868 * self.salinity
        
        # Thermal conductivity adjustment (simplified)
        # Approximate reduction based on salinity
        self.thermal_conductivity *= (1.0 - 0.00124 * self.salinity)
        
        # Surface tension adjustment (simplified)
        # Slight increase with salinity
        self.surface_tension *= (1.0 + 0.0005 * self.salinity)
    
    def density_at_conditions(self, temperature: float, pressure: float) -> float:
        """
        Calculate water density at specified temperature and pressure.
        
        Args:
            temperature (float): Temperature in K
            pressure (float): Pressure in bar
            
        Returns:
            float: Density in kg/m³
        """
        # Convert temperature to °C
        temp_c = temperature - 273.15
        
        # Base density for pure water (IAPWS formulation, simplified)
        # This is a simplified model valid for 0-150°C and up to 1000 bar
        
        # Coefficient table for density calculation
        a1 = -3.65
        a2 = 0.0012
        a3 = -0.000000652
        a4 = -0.00462
        a5 = 0.000112
        a6 = -0.00000048
        a7 = 0.000000000844
        
        # Calculate pure water density
        pure_water_density = 1000.0 * (1 + a1*1e-3*temp_c + a2*1e-3*temp_c**2 + 
                                     a3*1e-3*temp_c**3 + a4*1e-6*pressure + 
                                     a5*1e-6*pressure*temp_c + a6*1e-6*pressure*temp_c**2 + 
                                     a7*1e-6*pressure**2)
        
        # Adjust for salinity
        if self.salinity <= 0.0:
            return pure_water_density
        
        # Salinity effect diminishes slightly with temperature
        salinity_factor = 1.0 + 0.0065 * self.salinity * (1.0 - 0.001 * (temp_c - 15))
        
        return pure_water_density * salinity_factor
    
    def viscosity_at_conditions(self, temperature: float, pressure: float) -> float:
        """
        Calculate water viscosity at specified temperature and pressure.
        
        Args:
            temperature (float): Temperature in K
            pressure (float): Pressure in bar
            
        Returns:
            float: Viscosity in Pa·s
        """
        # Convert temperature to °C
        temp_c = temperature - 273.15
        
        # Base viscosity for pure water (simplified)
        # Valid for 0-100°C at atmospheric pressure
        a = 2.414e-5  # Pa·s
        b = 247.8  # K
        c = 140.0  # K
        
        # Calculate base viscosity using a simplified version of the IAPWS formulation
        pure_water_visc = a * 10**(b / (temperature - c))
        
        # Pressure correction (simplified)
        # Viscosity increases with pressure
        pressure_correction = 1.0 + 0.0001 * (pressure - 1.0)
        
        # Adjust for salinity
        if self.salinity <= 0.0:
            return pure_water_visc * pressure_correction
        
        # Salinity effect on viscosity
        # More pronounced at lower temperatures
        temp_factor = 1.0 - 0.002 * (temp_c - 20)
        salinity_factor = 1.0 + 0.00214 * self.salinity * temp_factor
        
        return pure_water_visc * pressure_correction * salinity_factor
    
    def get_phase(self, temperature: float, pressure: float) -> FluidPhase:
        """
        Determine the phase of water at given conditions.
        
        Args:
            temperature (float): Temperature in K
            pressure (float): Pressure in bar
            
        Returns:
            FluidPhase: Phase of the fluid
        """
        # Critical point of water
        t_crit = ReferenceConditions.WATER_CRITICAL_TEMPERATURE_C + 273.15  # K
        p_crit = ReferenceConditions.WATER_CRITICAL_PRESSURE_BAR  # bar
        
        if temperature > t_crit and pressure > p_crit:
            return FluidPhase.SUPERCRITICAL
        
        # Simple phase diagram for water
        if temperature < 273.15:
            return FluidPhase.SUBCRITICAL  # Solid (ice) - simplified
        elif temperature <= 373.15:
            # Between freezing and boiling (at 1 atm)
            # Need to check against vapor pressure
            t_c = temperature - 273.15
            
            # Antoine equation for vapor pressure (simplified)
            log_p_mmhg = 8.07131 - 1730.63 / (233.426 + t_c)
            p_vapor_bar = 10**log_p_mmhg / 750.061  # Convert mmHg to bar
            
            if pressure < p_vapor_bar:
                return FluidPhase.GAS  # Vapor
            else:
                return FluidPhase.LIQUID  # Liquid
        else:
            # Above boiling at 1 atm, check with vapor pressure
            if pressure < p_crit:
                # Simplified check
                t_c = temperature - 273.15
                
                # Antoine equation (simplified)
                log_p_mmhg = 8.07131 - 1730.63 / (233.426 + t_c)
                p_vapor_bar = 10**log_p_mmhg / 750.061  # Convert mmHg to bar
                
                if pressure < p_vapor_bar:
                    return FluidPhase.GAS
                else:
                    return FluidPhase.LIQUID
            else:
                return FluidPhase.LIQUID  # High pressure, assume liquid


class TwoPhaseProperties:
    """
    Class for calculating properties of two-phase mixtures.
    Used for oil-gas, oil-water, and gas-water mixtures.
    """
    
    def __init__(self, fluid1: FluidProperties, fluid2: FluidProperties, 
                surface_tension: float = None):
        """
        Initialize two-phase properties calculator.
        
        Args:
            fluid1 (FluidProperties): First fluid
            fluid2 (FluidProperties): Second fluid
            surface_tension (float, optional): Surface tension between fluids (N/m)
        """
        self.fluid1 = fluid1
        self.fluid2 = fluid2
        
        # Determine fluid pair type
        self.is_oil_gas = (isinstance(fluid1, OilProperties) and isinstance(fluid2, GasProperties)) or \
                         (isinstance(fluid1, GasProperties) and isinstance(fluid2, OilProperties))
        
        self.is_oil_water = (isinstance(fluid1, OilProperties) and isinstance(fluid2, WaterProperties)) or \
                          (isinstance(fluid1, WaterProperties) and isinstance(fluid2, OilProperties))
        
        self.is_gas_water = (isinstance(fluid1, GasProperties) and isinstance(fluid2, WaterProperties)) or \
                          (isinstance(fluid1, WaterProperties) and isinstance(fluid2, GasProperties))
        
        # Set surface tension if provided, otherwise use default
        if surface_tension is not None:
            self.surface_tension = surface_tension
        elif self.is_oil_gas:
            self.surface_tension = 0.025  # N/m (typical for oil-gas)
        elif self.is_oil_water:
            self.surface_tension = 0.03  # N/m (typical for oil-water)
        elif self.is_gas_water:
            self.surface_tension = 0.072  # N/m (typical for gas-water)
        else:
            self.surface_tension = 0.03  # N/m (default)
    
    def average_density(self, volume_fraction1: float, temperature: float, pressure: float) -> float:
        """
        Calculate average density of the mixture using volume fraction.
        
        Args:
            volume_fraction1 (float): Volume fraction of fluid1 (0-1)
            temperature (float): Temperature in K
            pressure (float): Pressure in bar
            
        Returns:
            float: Average density in kg/m³
        """
        # Ensure volume fraction is in valid range
        volume_fraction1 = max(0.0, min(1.0, volume_fraction1))
        volume_fraction2 = 1.0 - volume_fraction1
        
        # Get densities at conditions
        density1 = self.fluid1.density_at_conditions(temperature, pressure)
        density2 = self.fluid2.density_at_conditions(temperature, pressure)
        
        # Volume-weighted average
        return volume_fraction1 * density1 + volume_fraction2 * density2
    
    def mixture_viscosity(self, volume_fraction1: float, temperature: float, pressure: float) -> float:
        """
        Calculate effective viscosity of the mixture.
        
        Args:
            volume_fraction1 (float): Volume fraction of fluid1 (0-1)
            temperature (float): Temperature in K
            pressure (float): Pressure in bar
            
        Returns:
            float: Effective viscosity in Pa·s
        """
        # Ensure volume fraction is in valid range
        volume_fraction1 = max(0.0, min(1.0, volume_fraction1))
        volume_fraction2 = 1.0 - volume_fraction1
        
        # Get viscosities at conditions
        visc1 = self.fluid1.viscosity_at_conditions(temperature, pressure)
        visc2 = self.fluid2.viscosity_at_conditions(temperature, pressure)
        
        if self.is_oil_gas or self.is_gas_water:
            # For gas-liquid mixtures, use a more complex model
            # Determine which is gas and which is liquid
            if isinstance(self.fluid1, GasProperties):
                gas_visc = visc1
                liquid_visc = visc2
                gas_fraction = volume_fraction1
            else:
                gas_visc = visc2
                liquid_visc = visc1
                gas_fraction = volume_fraction2
            
            # Beggs-Brill correlation (simplified)
            if gas_fraction < 0.001:
                return liquid_visc
            elif gas_fraction > 0.999:
                return gas_visc
            else:
                # Simplified mixture rule for gas-liquid
                y = gas_fraction / (gas_visc**0.5) + (1 - gas_fraction) / (liquid_visc**0.5)
                return 1 / (y**2)
        
        else:
            # For liquid-liquid mixtures (oil-water)
            # Use simple logarithmic mixing rule
            log_visc1 = math.log(visc1)
            log_visc2 = math.log(visc2)
            log_visc_mix = volume_fraction1 * log_visc1 + volume_fraction2 * log_visc2
            return math.exp(log_visc_mix)
    
    def calculate_slip(self, volume_fraction1: float, velocity: float, 
                     pipe_diameter: float, inclination: float = 0.0) -> float:
        """
        Calculate slip velocity between phases.
        
        Args:
            volume_fraction1 (float): Volume fraction of fluid1 (0-1)
            velocity (float): Mixture velocity in m/s
            pipe_diameter (float): Pipe diameter in m
            inclination (float, optional): Pipe inclination in degrees
            
        Returns:
            float: Slip velocity in m/s
        """
        # Simplified slip velocity calculation
        # Positive slip means fluid1 is faster than fluid2
        
        # Ensure we have gas-liquid or oil-water system
        if not (self.is_oil_gas or self.is_gas_water or self.is_oil_water):
            return 0.0
        
        # Convert inclination to radians
        incl_rad = inclination * math.pi / 180.0
        
        # Determine which is the lighter phase
        is_fluid1_lighter = True
        
        # Compare densities at standard conditions for simplicity
        if self.fluid1.density > self.fluid2.density:
            is_fluid1_lighter = False
        
        # Simplified slip calculation
        if self.is_oil_gas or self.is_gas_water:
            # Gas-liquid system
            if (is_fluid1_lighter and isinstance(self.fluid1, GasProperties)) or \
               (not is_fluid1_lighter and isinstance(self.fluid2, GasProperties)):
                # Gas rising through liquid
                slip = 0.3 * math.sqrt(9.81 * pipe_diameter) * (1.0 - volume_fraction1) * math.sin(incl_rad + 0.3)
            else:
                # Invalid configuration (gas should be lighter)
                slip = 0.0
        else:
            # Oil-water system (smaller density difference)
            if volume_fraction1 < 0.1 or volume_fraction1 > 0.9:
                # Low dispersed phase, minimal slip
                slip = 0.0
            else:
                # Calculate based on density difference
                dens1 = self.fluid1.density
                dens2 = self.fluid2.density
                
                # Simplified slip calculation
                slip = 0.1 * math.sqrt(9.81 * pipe_diameter * abs(dens1 - dens2) / max(dens1, dens2)) * math.sin(incl_rad + 0.3)
                
                # Adjust sign based on which phase is lighter
                if (is_fluid1_lighter and dens1 > dens2) or (not is_fluid1_lighter and dens1 < dens2):
                    slip = -slip
        
        return slip
    
    def flow_pattern(self, volume_fraction1: float, velocity: float, 
                   pipe_diameter: float, inclination: float = 0.0) -> str:
        """
        Determine the flow pattern for the two-phase flow.
        
        Args:
            volume_fraction1 (float): Volume fraction of fluid1 (0-1)
            velocity (float): Mixture velocity in m/s
            pipe_diameter (float): Pipe diameter in m
            inclination (float, optional): Pipe inclination in degrees
            
        Returns:
            str: Flow pattern descriptor
        """
        # Only implemented for gas-liquid systems
        if not (self.is_oil_gas or self.is_gas_water):
            if self.is_oil_water:
                # Oil-water flow patterns
                return self._oil_water_flow_pattern(volume_fraction1, velocity, pipe_diameter, inclination)
            return "Undefined"
        
        # Determine gas and liquid properties
        if isinstance(self.fluid1, GasProperties):
            gas = self.fluid1
            liquid = self.fluid2
            gas_fraction = volume_fraction1
        else:
            gas = self.fluid2
            liquid = self.fluid1
            gas_fraction = 1.0 - volume_fraction1
        
        # Calculate superficial velocities
        vsg = velocity * gas_fraction  # Gas superficial velocity
        vsl = velocity * (1.0 - gas_fraction)  # Liquid superficial velocity
        
        # Simplified flow pattern determination
        # Based on Taitel-Dukler approach (horizontal) and Barnea (inclined)
        
        # Convert properties for correlations
        liquid_density = liquid.density  # kg/m³
        gas_density = gas.density  # kg/m³
        liquid_viscosity = liquid.viscosity  # Pa·s
        surface_tension = self.surface_tension  # N/m
        
        # Calculate dimensionless parameters
        froude_number = vsl / math.sqrt(9.81 * pipe_diameter)
        
        # Simplified flow pattern map
        if abs(inclination) <= 10:  # Near horizontal
            if vsg < 0.5:
                if froude_number < 1.0:
                    return "Stratified"
                else:
                    return "Dispersed Bubble"
            elif vsg < 10:
                if vsl < 0.3:
                    return "Stratified Wavy"
                elif vsl < 1.0:
                    return "Slug"
                else:
                    return "Plug"
            else:
                if vsl < 0.3:
                    return "Annular"
                else:
                    return "Annular Mist"
        else:  # Inclined
            if inclination > 0:  # Upward
                if vsg < 0.5:
                    if vsl < 0.1:
                        return "Stratified"
                    else:
                        return "Bubble"
                elif vsg < 10:
                    if vsl < 0.3:
                        return "Churn"
                    else:
                        return "Slug"
                else:
                    return "Annular"
            else:  # Downward
                if vsg < 0.5:
                    if vsl < 0.1:
                        return "Stratified"
                    else:
                        return "Bubble"
                elif vsg < 5:
                    return "Slug"
                else:
                    return "Annular"
    
    def _oil_water_flow_pattern(self, oil_fraction: float, velocity: float, 
                              pipe_diameter: float, inclination: float = 0.0) -> str:
        """
        Determine oil-water flow pattern.
        
        Args:
            oil_fraction (float): Oil volume fraction (0-1)
            velocity (float): Mixture velocity in m/s
            pipe_diameter (float): Pipe diameter in m
            inclination (float): Pipe inclination in degrees
            
        Returns:
            str: Flow pattern descriptor
        """
        # Determine which fluid is oil
        oil = None
        water = None
        if isinstance(self.fluid1, OilProperties):
            oil = self.fluid1
            water = self.fluid2
            actual_oil_fraction = oil_fraction
        else:
            oil = self.fluid2
            water = self.fluid1
            actual_oil_fraction = 1.0 - oil_fraction
        
        # Simplified oil-water flow pattern determination
        # Based on mixture velocity and oil fraction
        
        if abs(inclination) <= 10:  # Near horizontal
            if velocity < 0.5:
                if actual_oil_fraction < 0.2:
                    return "Oil Dispersed in Water"
                elif actual_oil_fraction > 0.8:
                    return "Water Dispersed in Oil"
                else:
                    return "Stratified Oil-Water"
            elif velocity < 1.5:
                if actual_oil_fraction < 0.3:
                    return "Oil Bubbles in Water"
                elif actual_oil_fraction > 0.7:
                    return "Water Bubbles in Oil"
                else:
                    return "Mixed Oil-Water"
            else:
                if actual_oil_fraction < 0.4:
                    return "Dispersed Oil in Water"
                elif actual_oil_fraction > 0.6:
                    return "Dispersed Water in Oil"
                else:
                    return "Oil-Water Emulsion"
        else:  # Inclined
            # Simplified for inclined pipes
            if velocity < 1.0:
                if actual_oil_fraction < 0.3:
                    return "Oil Dispersed in Water"
                elif actual_oil_fraction > 0.7:
                    return "Water Dispersed in Oil"
                else:
                    return "Intermittent Oil-Water"
            else:
                if actual_oil_fraction < 0.4:
                    return "Dispersed Oil in Water"
                elif actual_oil_fraction > 0.6:
                    return "Dispersed Water in Oil"
                else:
                    return "Oil-Water Emulsion"


class ThreePhaseProperties:
    """
    Class for calculating properties of three-phase mixtures.
    Used for oil-water-gas systems.
    """
    
    def __init__(self, oil: OilProperties, water: WaterProperties, gas: GasProperties,
                oil_water_tension: float = None, oil_gas_tension: float = None,
                water_gas_tension: float = None):
        """
        Initialize three-phase properties calculator.
        
        Args:
            oil (OilProperties): Oil phase
            water (WaterProperties): Water phase
            gas (GasProperties): Gas phase
            oil_water_tension (float, optional): Oil-water surface tension (N/m)
            oil_gas_tension (float, optional): Oil-gas surface tension (N/m)
            water_gas_tension (float, optional): Water-gas surface tension (N/m)
        """
        self.oil = oil
        self.water = water
        self.gas = gas
        
        # Set surface tensions or use defaults
        self.oil_water_tension = oil_water_tension or 0.03  # N/m
        self.oil_gas_tension = oil_gas_tension or 0.025  # N/m
        self.water_gas_tension = water_gas_tension or 0.072  # N/m
        
        # Create two-phase property calculators
        self.oil_water = TwoPhaseProperties(oil, water, self.oil_water_tension)
        self.oil_gas = TwoPhaseProperties(oil, gas, self.oil_gas_tension)
        self.water_gas = TwoPhaseProperties(water, gas, self.water_gas_tension)
    
    def average_density(self, oil_fraction: float, water_fraction: float, gas_fraction: float,
                      temperature: float, pressure: float) -> float:
        """
        Calculate average density of the three-phase mixture.
        
        Args:
            oil_fraction (float): Oil volume fraction (0-1)
            water_fraction (float): Water volume fraction (0-1)
            gas_fraction (float): Gas volume fraction (0-1)
            temperature (float): Temperature in K
            pressure (float): Pressure in bar
            
        Returns:
            float: Average density in kg/m³
            
        Raises:
            ValueError: If volume fractions don't sum to approximately 1.0
        """
        # Validate volume fractions
        total = oil_fraction + water_fraction + gas_fraction
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Volume fractions ({total}) must sum to approximately 1.0")
        
        # Normalize if needed
        if total != 1.0:
            factor = 1.0 / total
            oil_fraction *= factor
            water_fraction *= factor
            gas_fraction *= factor
        
        # Get individual densities
        oil_density = self.oil.density_at_conditions(temperature, pressure)
        water_density = self.water.density_at_conditions(temperature, pressure)
        gas_density = self.gas.density_at_conditions(temperature, pressure)
        
        # Volume-weighted average
        return (oil_fraction * oil_density + 
              water_fraction * water_density + 
              gas_fraction * gas_density)
    
    def mixture_viscosity(self, oil_fraction: float, water_fraction: float, gas_fraction: float,
                        temperature: float, pressure: float) -> float:
        """
        Calculate effective viscosity of the three-phase mixture.
        
        Args:
            oil_fraction (float): Oil volume fraction (0-1)
            water_fraction (float): Water volume fraction (0-1)
            gas_fraction (float): Gas volume fraction (0-1)
            temperature (float): Temperature in K
            pressure (float): Pressure in bar
            
        Returns:
            float: Effective viscosity in Pa·s
            
        Raises:
            ValueError: If volume fractions don't sum to approximately 1.0
        """
        # Validate volume fractions
        total = oil_fraction + water_fraction + gas_fraction
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Volume fractions ({total}) must sum to approximately 1.0")
        
        # Normalize if needed
        if total != 1.0:
            factor = 1.0 / total
            oil_fraction *= factor
            water_fraction *= factor
            gas_fraction *= factor
        
        # Get individual viscosities
        oil_visc = self.oil.viscosity_at_conditions(temperature, pressure)
        water_visc = self.water.viscosity_at_conditions(temperature, pressure)
        gas_visc = self.gas.viscosity_at_conditions(temperature, pressure)
        
        # For three-phase mixtures, first combine oil and water, then add gas
        if oil_fraction + water_fraction < 0.001:
            # Almost all gas
            return gas_visc
        
        # Calculate liquid mixture viscosity
        liquid_fraction = oil_fraction + water_fraction
        oil_in_liquid = oil_fraction / liquid_fraction
        water_in_liquid = water_fraction / liquid_fraction
        
        # Calculate liquid mixture viscosity using oil-water calculator
        liquid_visc = self.oil_water.mixture_viscosity(oil_in_liquid, temperature, pressure)
        
        if gas_fraction < 0.001:
            # Almost no gas
            return liquid_visc
        
        # Create temporary two-phase system for liquid-gas
        liquid_gas = TwoPhaseProperties(self.oil, self.gas)  # Using oil object as placeholder
        
        # Override densities and viscosities for the liquid phase
        liquid_density = oil_fraction * self.oil.density + water_fraction * self.water.density
        liquid_gas.fluid1.density = liquid_density
        liquid_gas.fluid1.viscosity = liquid_visc
        
        # Calculate final mixture viscosity
        return liquid_gas.mixture_viscosity(1.0 - gas_fraction, temperature, pressure)
    
    def flow_pattern(self, oil_fraction: float, water_fraction: float, gas_fraction: float,
                   velocity: float, pipe_diameter: float, inclination: float = 0.0) -> str:
        """
        Determine the flow pattern for the three-phase flow.
        
        Args:
            oil_fraction (float): Oil volume fraction (0-1)
            water_fraction (float): Water volume fraction (0-1)
            gas_fraction (float): Gas volume fraction (0-1)
            velocity (float): Mixture velocity in m/s
            pipe_diameter (float): Pipe diameter in m
            inclination (float, optional): Pipe inclination in degrees
            
        Returns:
            str: Flow pattern descriptor
            
        Raises:
            ValueError: If volume fractions don't sum to approximately 1.0
        """
        # Validate volume fractions
        total = oil_fraction + water_fraction + gas_fraction
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Volume fractions ({total}) must sum to approximately 1.0")
        
        # Normalize if needed
        if total != 1.0:
            factor = 1.0 / total
            oil_fraction *= factor
            water_fraction *= factor
            gas_fraction *= factor
        
        # Simplified three-phase flow pattern determination
        # Based on dominant phases and their flow patterns
        
        # Special cases
        if gas_fraction < 0.05:
            # Mostly liquid (oil and water)
            return self.oil_water.flow_pattern(oil_fraction / (oil_fraction + water_fraction), 
                                           velocity, pipe_diameter, inclination)
        
        if oil_fraction + water_fraction < 0.05:
            # Mostly gas
            return "Single Phase Gas"
        
        # Typical three-phase patterns
        liquid_fraction = oil_fraction + water_fraction
        
        # Calculate superficial velocities
        vsg = velocity * gas_fraction  # Gas superficial velocity
        vsl = velocity * liquid_fraction  # Liquid superficial velocity
        
        # Simplified flow pattern map
        if abs(inclination) <= 10:  # Near horizontal
            if vsg < 0.5:
                if liquid_fraction > 0.9:
                    # Oil and water dominant
                    oil_in_liquid = oil_fraction / liquid_fraction
                    if oil_in_liquid < 0.2:
                        return "Oil Bubbles in Water with Gas"
                    elif oil_in_liquid > 0.8:
                        return "Water Bubbles in Oil with Gas"
                    else:
                        return "Stratified Oil-Water with Gas Bubbles"
                else:
                    return "Stratified Three-Phase"
            elif vsg < 10:
                if vsl < 0.3:
                    return "Stratified Wavy Three-Phase"
                elif vsl < 1.0:
                    return "Three-Phase Slug"
                else:
                    return "Three-Phase Plug"
            else:
                if vsl < 0.3:
                    return "Three-Phase Annular"
                else:
                    return "Three-Phase Annular Mist"
        else:  # Inclined
            if inclination > 0:  # Upward
                if vsg < 1.0:
                    return "Three-Phase Bubble"
                elif vsg < 10:
                    return "Three-Phase Slug/Churn"
                else:
                    return "Three-Phase Annular"
            else:  # Downward
                if vsg < 1.0:
                    return "Three-Phase Bubble/Stratified"
                elif vsg < 10:
                    return "Three-Phase Slug"
                else:
                    return "Three-Phase Annular"


# Factory function to create fluid properties
def create_fluid_properties(fluid_type: str, **kwargs) -> FluidProperties:
    """
    Factory function to create fluid properties object based on type.
    
    Args:
        fluid_type (str): Type of fluid
        **kwargs: Additional properties
        
    Returns:
        FluidProperties: Fluid properties object
        
    Raises:
        ValueError: If fluid type is not recognized
    """
    fluid_type = fluid_type.lower()
    
    if fluid_type in ["crude oil", "oil", "condensate"]:
        # Required parameters for oil
        api_gravity = kwargs.get("api_gravity", 35.0)
        gas_oil_ratio = kwargs.get("gas_oil_ratio", 0.0)
        name = kwargs.get("name", "Crude Oil")
        
        return OilProperties(name, api_gravity, gas_oil_ratio)
    
    elif fluid_type in ["natural gas", "gas"]:
        # Required parameters for gas
        specific_gravity = kwargs.get("specific_gravity", 0.65)
        name = kwargs.get("name", "Natural Gas")
        
        return GasProperties(name, specific_gravity)
    
    elif fluid_type in ["water", "brine", "produced water"]:
        # Required parameters for water
        salinity = kwargs.get("salinity", 0.0)
        name = kwargs.get("name", "Water")
        
        return WaterProperties(name, salinity)
    
    else:
        raise ValueError(f"Unrecognized fluid type: {fluid_type}")


# Utility functions for common conversions
def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert temperature between different units.
    
    Args:
        value (float): Temperature value
        from_unit (str): Source unit ('C', 'F', 'K', 'R')
        to_unit (str): Target unit ('C', 'F', 'K', 'R')
        
    Returns:
        float: Converted temperature
        
    Raises:
        ValueError: If units are not recognized
    """
    # Standard temperature in Kelvin
    kelvin = None
    
    # Convert to Kelvin first
    if from_unit.upper() == 'C':
        kelvin = value + 273.15
    elif from_unit.upper() == 'F':
        kelvin = (value - 32) * 5/9 + 273.15
    elif from_unit.upper() == 'K':
        kelvin = value
    elif from_unit.upper() == 'R':
        kelvin = value * 5/9
    else:
        raise ValueError(f"Unknown temperature unit: {from_unit}")
    
    # Convert from Kelvin to target unit
    if to_unit.upper() == 'C':
        return kelvin - 273.15
    elif to_unit.upper() == 'F':
        return (kelvin - 273.15) * 9/5 + 32
    elif to_unit.upper() == 'K':
        return kelvin
    elif to_unit.upper() == 'R':
        return kelvin * 9/5
    else:
        raise ValueError(f"Unknown temperature unit: {to_unit}")


def convert_pressure(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert pressure between different units.
    
    Args:
        value (float): Pressure value
        from_unit (str): Source unit ('pa', 'kpa', 'bar', 'psi', 'atm')
        to_unit (str): Target unit ('pa', 'kpa', 'bar', 'psi', 'atm')
        
    Returns:
        float: Converted pressure
        
    Raises:
        ValueError: If units are not recognized
    """
    # Conversion factors to Pa
    factors = {
        'PA': 1.0,
        'KPA': 1000.0,
        'BAR': 100000.0,
        'PSI': 6894.76,
        'ATM': 101325.0
    }
    
    # Check units
    from_unit = from_unit.upper()
    to_unit = to_unit.upper()
    
    if from_unit not in factors:
        raise ValueError(f"Unknown pressure unit: {from_unit}")
    if to_unit not in factors:
        raise ValueError(f"Unknown pressure unit: {to_unit}")
    
    # Convert to target unit
    return value * factors[from_unit] / factors[to_unit]


def api_to_specific_gravity(api_gravity: float) -> float:
    """
    Convert API gravity to specific gravity.
    
    Args:
        api_gravity (float): API gravity
        
    Returns:
        float: Specific gravity (water = 1.0)
    """
    return 141.5 / (api_gravity + 131.5)


def specific_gravity_to_api(specific_gravity: float) -> float:
    """
    Convert specific gravity to API gravity.
    
    Args:
        specific_gravity (float): Specific gravity (water = 1.0)
        
    Returns:
        float: API gravity
    """
    return (141.5 / specific_gravity) - 131.5


def calculate_z_factor(pr: float, tr: float) -> float:
    """
    Calculate gas compressibility factor (z-factor) using Standing-Katz correlation.
    
    Args:
        pr (float): Reduced pressure (p/p_critical)
        tr (float): Reduced temperature (T/T_critical)
        
    Returns:
        float: Z-factor (dimensionless)
    """
    # This is a simplified correlation for the Standing-Katz z-factor chart
    # Valid for 0.2 < pr < 15 and 1.05 < tr < 3.0
    
    if pr < 0.1:
        return 1.0
    
    a = 1.39 * (tr - 0.92)**0.5 - 0.36 * tr - 0.101
    b = (0.62 - 0.23 * tr) * pr + (0.066 / (tr - 0.86) - 0.037) * pr**2 + 0.32 * pr**6 / (10**(9 * (tr - 1)))
    z = a + (1.0 - a) * math.exp(-b) + 0.132 * pr**4
    
    # Ensure valid range
    z = max(0.2, min(z, 1.5))
    
    return z


def calculate_gas_viscosity(sg: float, temperature: float, pressure: float, z_factor: float = None) -> float:
    """
    Calculate gas viscosity using Lee-Gonzalez-Eakin correlation.
    
    Args:
        sg (float): Gas specific gravity (air = 1.0)
        temperature (float): Temperature in K
        pressure (float): Pressure in bar
        z_factor (float, optional): Gas compressibility factor
        
    Returns:
        float: Gas viscosity in Pa·s
    """
    # Convert temperature to Rankine for correlation
    temp_r = temperature * 9/5  # K to °R
    
    # Molecular weight
    mw = sg * 28.97  # g/mol
    
    # Calculate z-factor if not provided
    if z_factor is None:
        # Estimate critical properties
        t_pc = 168.0 + 325.0 * sg - 12.5 * sg**2  # Pseudo-critical temperature in K
        p_pc = 4.6 + 0.1 * sg**4.0  # Pseudo-critical pressure in MPa
        
        # Convert to bar
        p_pc_bar = p_pc * 10.0
        
        # Calculate reduced properties
        tr = temperature / t_pc
        pr = pressure / p_pc_bar
        
        # Calculate z-factor
        z_factor = calculate_z_factor(pr, tr)
    
    # Calculate gas density in g/cc
    gas_density = pressure * 100000.0 * mw / (z_factor * 8.314 * temperature) / 1000000.0
    
    # Lee-Gonzalez-Eakin correlation
    k = (9.4 + 0.02 * mw) * temp_r**1.5 / (209.0 + 19.0 * mw + temp_r)
    x = 3.5 + 986.0 / temp_r + 0.01 * mw
    y = 2.4 - 0.2 * x
    
    # Calculate viscosity in cP
    viscosity_cp = 1e-4 * k * math.exp(x * (gas_density**y))
    
    # Convert to Pa·s
    return viscosity_cp * 0.001


def oil_viscosity_correlation(api_gravity: float, temperature: float, pressure: float = None, 
                            rs: float = 0.0) -> float:
    """
    Calculate oil viscosity using Beggs-Robinson correlation.
    
    Args:
        api_gravity (float): Oil API gravity
        temperature (float): Temperature in K
        pressure (float, optional): Pressure in bar (for live oil)
        rs (float, optional): Solution gas-oil ratio in scf/STB
        
    Returns:
        float: Oil viscosity in Pa·s
    """
    # Convert temperature to °F for correlation
    temp_f = (temperature - 273.15) * 9/5 + 32
    
    # Beggs-Robinson correlation for dead oil viscosity
    z = 3.0324 - 0.02023 * api_gravity
    y = 10**z
    x = y * (temp_f**-1.163)
    
    # Dead oil viscosity in cP
    dead_oil_visc = x
    
    # If solution gas-oil ratio is provided, adjust for live oil
    if rs > 0:
        # Convert pressure to psi
        if pressure is None:
            pressure = 14.7  # Assume atmospheric
        else:
            pressure = pressure * 14.5038  # bar to psi
        
        # Chew and Connally correction for live oil
        a = 10.715 * (rs + 100)**-0.515
        b = 5.44 * (rs + 150)**-0.338
        live_oil_visc = a * dead_oil_visc**b
    else:
        live_oil_visc = dead_oil_visc
    
    # Convert from cP to Pa·s
    return live_oil_visc * 0.001


def water_viscosity_correlation(temperature: float, salinity: float = 0.0) -> float:
    """
    Calculate water viscosity as a function of temperature and salinity.
    
    Args:
        temperature (float): Temperature in K
        salinity (float, optional): Salinity in weight percent
        
    Returns:
        float: Water viscosity in Pa·s
    """
    # Convert temperature to °C
    temp_c = temperature - 273.15
    
    # Base viscosity for pure water (IAPWS formulation, simplified)
    a = 2.414e-5  # Pa·s
    b = 247.8  # K
    c = 140.0  # K
    
    # Convert temperature to K
    temp_k = temp_c + 273.15
    
    # Calculate pure water viscosity
    pure_water_visc = a * 10**(b / (temp_k - c))
    
    # Apply salinity correction
    if salinity <= 0:
        return pure_water_visc
        
    # McCain correlation for effect of salinity
    # Valid for 0-26% NaCl by weight and 10-350°C
    salinity_factor = 1.0 + 0.00214 * salinity * (1.0 - 0.002 * (temp_c - 20))
    
    return pure_water_visc * salinity_factor


def surface_tension_correlation(fluid1: str, fluid2: str, temperature: float) -> float:
    """
    Estimate surface tension between two fluids.
    
    Args:
        fluid1 (str): First fluid type ('oil', 'water', 'gas')
        fluid2 (str): Second fluid type ('oil', 'water', 'gas')
        temperature (float): Temperature in K
        
    Returns:
        float: Surface tension in N/m
    """
    # Normalize input
    fluid1 = fluid1.lower()
    fluid2 = fluid2.lower()
    
    # Ensure consistent ordering
    if fluid1 > fluid2:
        fluid1, fluid2 = fluid2, fluid1
    
    # Temperature in °C
    temp_c = temperature - 273.15
    
    if fluid1 == 'oil' and fluid2 == 'water':
        # Oil-water
        # Decreases with temperature
        return 0.03 * (1.0 - 0.002 * (temp_c - 20))
    
    elif fluid1 == 'gas' and fluid2 == 'oil':
        # Gas-oil
        # Decreases with temperature
        return 0.025 * (1.0 - 0.0015 * (temp_c - 20))
    
    elif fluid1 == 'gas' and fluid2 == 'water':
        # Gas-water
        # Weinaug-Katz correlation (simplified)
        return 0.0728 * (1.0 - 0.002 * (temp_c - 20)) * (1.0 - temperature / 647.3)**1.256
    
    else:
        # Unknown combination
        return 0.03  # Default value