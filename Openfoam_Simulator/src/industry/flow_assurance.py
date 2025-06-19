#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow assurance utilities for Openfoam_Simulator.

This module provides models and calculations for common flow assurance issues 
in oil & gas pipelines, including:
- Hydrate formation prediction and prevention
- Wax deposition modeling
- Asphaltene precipitation
- Scale formation analysis
- Slugging prediction and mitigation
- Erosion and corrosion modeling
- Thermal management calculations
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from enum import Enum
from dataclasses import dataclass

# Import utility modules
from ..utils.logger import get_logger
from ..utils.unit_converter import convert_temperature, convert_pressure

logger = get_logger(__name__)


class FlowPattern(Enum):
    """Enumeration of multiphase flow patterns in pipelines."""
    STRATIFIED = "Stratified"
    STRATIFIED_WAVY = "Stratified Wavy"
    ANNULAR = "Annular"
    ANNULAR_MIST = "Annular Mist"
    DISPERSED_BUBBLE = "Dispersed Bubble"
    INTERMITTENT = "Intermittent"
    SLUG = "Slug"
    CHURN = "Churn"
    ELONGATED_BUBBLE = "Elongated Bubble"
    BUBBLY = "Bubbly"


class FlowDirection(Enum):
    """Enumeration of flow directions in pipelines."""
    HORIZONTAL = "Horizontal"
    VERTICAL_UP = "Vertical Upward"
    VERTICAL_DOWN = "Vertical Downward"
    INCLINED_UP = "Inclined Upward"
    INCLINED_DOWN = "Inclined Downward"


@dataclass
class FluidProperties:
    """Data class for fluid properties used in flow assurance calculations."""
    density: float  # kg/m³
    viscosity: float  # Pa·s
    specific_heat: float  # J/(kg·K)
    thermal_conductivity: float  # W/(m·K)
    surface_tension: float = 0.02  # N/m
    molecular_weight: float = 100.0  # g/mol
    pour_point: float = 273.15  # K
    wax_content: float = 0.0  # weight fraction
    asphaltene_content: float = 0.0  # weight fraction
    bubble_point: float = 1.0e6  # Pa
    gas_oil_ratio: float = 0.0  # std m³/m³
    water_cut: float = 0.0  # fraction
    salinity: float = 0.0  # ppm
    h2s_content: float = 0.0  # ppm
    co2_content: float = 0.0  # ppm


@dataclass
class PipeProperties:
    """Data class for pipe properties used in flow assurance calculations."""
    diameter: float  # m
    roughness: float  # m
    length: float  # m
    wall_thickness: float  # m
    thermal_conductivity: float  # W/(m·K)
    incline_angle: float = 0.0  # degrees (0 is horizontal, 90 is vertical up)
    insulation_thickness: float = 0.0  # m
    insulation_conductivity: float = 0.05  # W/(m·K)
    burial_depth: float = 0.0  # m (0 means not buried)
    soil_conductivity: float = 1.5  # W/(m·K)


@dataclass
class AmbientConditions:
    """Data class for ambient conditions used in flow assurance calculations."""
    temperature: float  # K
    pressure: float  # Pa
    heat_transfer_coefficient: float = 20.0  # W/(m²·K)
    seawater_temperature: float = 280.0  # K (for subsea pipelines)
    sea_depth: float = 0.0  # m
    current_velocity: float = 0.0  # m/s


class FlowAssuranceCalculator:
    """
    Base class for flow assurance calculations.
    
    Provides common methods for basic flow calculations needed in
    various flow assurance analyses.
    """
    
    def __init__(self):
        """Initialize the flow assurance calculator."""
        self.g = 9.81  # m/s², gravitational acceleration
        
        # Constants for conversions
        self.R = 8.314  # J/(mol·K), universal gas constant
    
    def calculate_reynolds_number(self, 
                                 velocity: float, 
                                 diameter: float, 
                                 density: float, 
                                 viscosity: float) -> float:
        """
        Calculate Reynolds number for a fluid flow.
        
        Args:
            velocity (float): Fluid velocity in m/s
            diameter (float): Pipe diameter in m
            density (float): Fluid density in kg/m³
            viscosity (float): Fluid viscosity in Pa·s
            
        Returns:
            float: Reynolds number (dimensionless)
        """
        if viscosity <= 0:
            raise ValueError("Viscosity must be positive")
        
        return (density * velocity * diameter) / viscosity
    
    def calculate_friction_factor(self, 
                                 reynolds: float, 
                                 roughness: float, 
                                 diameter: float) -> float:
        """
        Calculate Darcy friction factor using Colebrook-White equation.
        
        Args:
            reynolds (float): Reynolds number
            roughness (float): Pipe roughness in m
            diameter (float): Pipe diameter in m
            
        Returns:
            float: Darcy friction factor (dimensionless)
        """
        if reynolds <= 0:
            raise ValueError("Reynolds number must be positive")
        
        if diameter <= 0:
            raise ValueError("Diameter must be positive")
        
        relative_roughness = roughness / diameter
        
        if reynolds < 2300:  # Laminar flow
            return 64.0 / reynolds
        elif reynolds > 4000:  # Turbulent flow
            # Use Haaland equation (explicit approximation of Colebrook-White)
            term1 = -1.8 * math.log10(
                (relative_roughness / 3.7)**1.11 + 6.9 / reynolds
            )
            return (1.0 / term1)**2
        else:  # Transitional flow
            # Interpolate between laminar and turbulent
            f_lam = 64.0 / 2300
            
            # Calculate turbulent factor at Re=4000
            term1_turb = -1.8 * math.log10(
                (relative_roughness / 3.7)**1.11 + 6.9 / 4000
            )
            f_turb = (1.0 / term1_turb)**2
            
            # Linear interpolation
            t = (reynolds - 2300) / (4000 - 2300)
            return f_lam * (1 - t) + f_turb * t
    
    def calculate_pressure_drop(self, 
                              length: float, 
                              diameter: float, 
                              density: float, 
                              velocity: float, 
                              friction_factor: float, 
                              incline_angle: float = 0.0) -> float:
        """
        Calculate pressure drop in a pipe segment.
        
        Args:
            length (float): Pipe length in m
            diameter (float): Pipe diameter in m
            density (float): Fluid density in kg/m³
            velocity (float): Fluid velocity in m/s
            friction_factor (float): Darcy friction factor
            incline_angle (float): Pipe inclination angle in degrees
            
        Returns:
            float: Pressure drop in Pa
        """
        if diameter <= 0:
            raise ValueError("Diameter must be positive")
        
        # Convert angle to radians
        angle_rad = math.radians(incline_angle)
        
        # Friction component
        dp_friction = friction_factor * length * density * velocity**2 / (2 * diameter)
        
        # Elevation component (positive for upward flow)
        dp_elevation = density * self.g * length * math.sin(angle_rad)
        
        # Total pressure drop
        dp_total = dp_friction + dp_elevation
        
        return dp_total
    
    def calculate_flow_rate(self, 
                          diameter: float, 
                          velocity: float) -> float:
        """
        Calculate volumetric flow rate.
        
        Args:
            diameter (float): Pipe diameter in m
            velocity (float): Fluid velocity in m/s
            
        Returns:
            float: Volumetric flow rate in m³/s
        """
        if diameter <= 0:
            raise ValueError("Diameter must be positive")
        
        area = math.pi * (diameter / 2)**2
        return velocity * area
    
    def calculate_velocity(self, 
                         diameter: float, 
                         flow_rate: float) -> float:
        """
        Calculate fluid velocity from flow rate.
        
        Args:
            diameter (float): Pipe diameter in m
            flow_rate (float): Volumetric flow rate in m³/s
            
        Returns:
            float: Fluid velocity in m/s
        """
        if diameter <= 0:
            raise ValueError("Diameter must be positive")
        
        if flow_rate < 0:
            raise ValueError("Flow rate cannot be negative")
        
        area = math.pi * (diameter / 2)**2
        
        if area <= 0:
            raise ValueError("Pipe area must be positive")
        
        return flow_rate / area
    
    def calculate_heat_transfer(self, 
                              length: float, 
                              u_value: float, 
                              t_fluid: float, 
                              t_ambient: float) -> float:
        """
        Calculate heat transfer through pipe wall.
        
        Args:
            length (float): Pipe length in m
            u_value (float): Overall heat transfer coefficient in W/(m²·K)
            t_fluid (float): Fluid temperature in K
            t_ambient (float): Ambient temperature in K
            
        Returns:
            float: Heat transfer rate in W
        """
        if length <= 0:
            raise ValueError("Length must be positive")
        
        # Calculate heat transfer area (circumference * length)
        return u_value * math.pi * length * (t_fluid - t_ambient)
    
    def calculate_overall_heat_transfer_coefficient(self, 
                                                 pipe: PipeProperties, 
                                                 h_internal: float, 
                                                 h_external: float) -> float:
        """
        Calculate overall heat transfer coefficient.
        
        Args:
            pipe (PipeProperties): Pipe properties
            h_internal (float): Internal heat transfer coefficient in W/(m²·K)
            h_external (float): External heat transfer coefficient in W/(m²·K)
            
        Returns:
            float: Overall heat transfer coefficient in W/(m²·K)
        """
        if pipe.diameter <= 0 or pipe.wall_thickness <= 0:
            raise ValueError("Pipe dimensions must be positive")
        
        if h_internal <= 0 or h_external <= 0:
            raise ValueError("Heat transfer coefficients must be positive")
        
        # Inner and outer diameters
        d_i = pipe.diameter
        d_o = d_i + 2 * pipe.wall_thickness
        
        # With insulation
        if pipe.insulation_thickness > 0:
            d_ins = d_o + 2 * pipe.insulation_thickness
            
            # Thermal resistances
            r_internal = 1 / (h_internal * d_i / 2)
            r_pipe = math.log(d_o / d_i) / (2 * math.pi * pipe.thermal_conductivity)
            r_insulation = math.log(d_ins / d_o) / (2 * math.pi * pipe.insulation_conductivity)
            r_external = 1 / (h_external * d_ins / 2)
            
            # Total resistance
            r_total = r_internal + r_pipe + r_insulation + r_external
            
            # Overall heat transfer coefficient (based on inner area)
            u_value = 1 / (r_total * math.pi * d_i)
        
        else:
            # Thermal resistances without insulation
            r_internal = 1 / (h_internal * d_i / 2)
            r_pipe = math.log(d_o / d_i) / (2 * math.pi * pipe.thermal_conductivity)
            r_external = 1 / (h_external * d_o / 2)
            
            # Total resistance
            r_total = r_internal + r_pipe + r_external
            
            # Overall heat transfer coefficient (based on inner area)
            u_value = 1 / (r_total * math.pi * d_i)
        
        return u_value
    
    def calculate_temperature_profile(self, 
                                    inlet_temperature: float, 
                                    ambient_temperature: float, 
                                    pipe: PipeProperties, 
                                    fluid: FluidProperties, 
                                    velocity: float, 
                                    u_value: float,
                                    num_segments: int = 100) -> List[float]:
        """
        Calculate fluid temperature profile along pipeline.
        
        Args:
            inlet_temperature (float): Fluid inlet temperature in K
            ambient_temperature (float): Ambient temperature in K
            pipe (PipeProperties): Pipe properties
            fluid (FluidProperties): Fluid properties
            velocity (float): Fluid velocity in m/s
            u_value (float): Overall heat transfer coefficient in W/(m²·K)
            num_segments (int): Number of segments for calculation
            
        Returns:
            List[float]: Temperature profile along pipeline in K
        """
        if velocity <= 0:
            raise ValueError("Velocity must be positive")
        
        if num_segments <= 0:
            raise ValueError("Number of segments must be positive")
        
        # Calculate flow rate and mass flow rate
        flow_rate = self.calculate_flow_rate(pipe.diameter, velocity)
        mass_flow_rate = flow_rate * fluid.density
        
        # Thermal capacity
        mcp = mass_flow_rate * fluid.specific_heat
        
        # Segment length
        dx = pipe.length / num_segments
        
        # Initialize temperature profile
        temps = [inlet_temperature]
        current_temp = inlet_temperature
        
        # Calculate temperature drop in each segment
        for i in range(num_segments):
            # Heat transfer per segment
            q = u_value * math.pi * pipe.diameter * dx * (current_temp - ambient_temperature)
            
            # Temperature drop
            dt = q / mcp
            current_temp -= dt
            temps.append(current_temp)
        
        return temps


class HydrateCalculator:
    """
    Calculator for hydrate formation prediction and prevention.
    
    Hydrates are ice-like crystalline compounds formed from water and small gas 
    molecules (like methane, ethane, etc.) under high pressure and low temperature
    conditions. Predicting and preventing hydrate formation is critical in oil & gas
    pipelines.
    """
    
    def __init__(self):
        """Initialize the hydrate calculator."""
        # Constants for gas gravity correlation
        self.k1 = 13.7
        self.k2 = 8.49
        self.k3 = 4.26
        self.k4 = 0.68
        
        # Pressure units used in correlations
        self.pressure_unit = "psia"
        self.temperature_unit = "F"
    
    def _convert_temperature(self, temp: float, from_unit: str, to_unit: str) -> float:
        """
        Convert temperature between units.
        
        Args:
            temp (float): Temperature value
            from_unit (str): Current unit ('K', 'C', or 'F')
            to_unit (str): Target unit ('K', 'C', or 'F')
            
        Returns:
            float: Converted temperature
        """
        # Convert to K as intermediate
        if from_unit == "K":
            temp_k = temp
        elif from_unit == "C":
            temp_k = temp + 273.15
        elif from_unit == "F":
            temp_k = (temp + 459.67) * 5/9
        else:
            raise ValueError(f"Unsupported temperature unit: {from_unit}")
        
        # Convert from K to target
        if to_unit == "K":
            return temp_k
        elif to_unit == "C":
            return temp_k - 273.15
        elif to_unit == "F":
            return temp_k * 9/5 - 459.67
        else:
            raise ValueError(f"Unsupported temperature unit: {to_unit}")
    
    def _convert_pressure(self, pressure: float, from_unit: str, to_unit: str) -> float:
        """
        Convert pressure between units.
        
        Args:
            pressure (float): Pressure value
            from_unit (str): Current unit ('Pa', 'kPa', 'bar', 'psi', 'psia')
            to_unit (str): Target unit ('Pa', 'kPa', 'bar', 'psi', 'psia')
            
        Returns:
            float: Converted pressure
        """
        # Convert to Pa as intermediate
        if from_unit == "Pa":
            pressure_pa = pressure
        elif from_unit == "kPa":
            pressure_pa = pressure * 1000
        elif from_unit == "bar":
            pressure_pa = pressure * 100000
        elif from_unit in ["psi", "psia"]:
            pressure_pa = pressure * 6894.76
        else:
            raise ValueError(f"Unsupported pressure unit: {from_unit}")
        
        # Convert from Pa to target
        if to_unit == "Pa":
            return pressure_pa
        elif to_unit == "kPa":
            return pressure_pa / 1000
        elif to_unit == "bar":
            return pressure_pa / 100000
        elif to_unit in ["psi", "psia"]:
            return pressure_pa / 6894.76
        else:
            raise ValueError(f"Unsupported pressure unit: {to_unit}")
    
    def calculate_hydrate_formation_temperature(self, 
                                              pressure: float, 
                                              gas_gravity: float, 
                                              pressure_unit: str = "Pa") -> float:
        """
        Calculate hydrate formation temperature using Hammerschmidt equation.
        
        Args:
            pressure (float): Pressure in specified unit
            gas_gravity (float): Gas specific gravity (relative to air)
            pressure_unit (str): Pressure unit ('Pa', 'kPa', 'bar', 'psi')
            
        Returns:
            float: Hydrate formation temperature in K
        """
        # Convert pressure to psia for correlation
        pressure_psia = self._convert_pressure(pressure, pressure_unit, "psia")
        
        # Check valid pressure range
        if pressure_psia <= 0:
            raise ValueError("Pressure must be positive")
        
        # Calculate hydrate temperature in °F using gas gravity correlation
        # Based on modified Hammerschmidt equation
        log_p = math.log10(pressure_psia)
        
        # Calculate constants based on gas gravity
        a = self.k1 - self.k3 * gas_gravity
        b = self.k2 + self.k4 * gas_gravity
        
        # Hydrate temperature in °F
        hydrate_temp_f = a + b * log_p
        
        # Convert to K
        hydrate_temp_k = self._convert_temperature(hydrate_temp_f, "F", "K")
        
        return hydrate_temp_k
    
    def calculate_hydrate_formation_pressure(self, 
                                           temperature: float, 
                                           gas_gravity: float, 
                                           temperature_unit: str = "K") -> float:
        """
        Calculate hydrate formation pressure using Hammerschmidt equation.
        
        Args:
            temperature (float): Temperature in specified unit
            gas_gravity (float): Gas specific gravity (relative to air)
            temperature_unit (str): Temperature unit ('K', 'C', 'F')
            
        Returns:
            float: Hydrate formation pressure in Pa
        """
        # Convert temperature to °F for correlation
        temperature_f = self._convert_temperature(temperature, temperature_unit, "F")
        
        # Calculate constants based on gas gravity
        a = self.k1 - self.k3 * gas_gravity
        b = self.k2 + self.k4 * gas_gravity
        
        # Hydrate pressure in psia
        hydrate_pressure_psia = 10**((temperature_f - a) / b)
        
        # Convert to Pa
        hydrate_pressure_pa = self._convert_pressure(hydrate_pressure_psia, "psia", "Pa")
        
        return hydrate_pressure_pa
    
    def is_in_hydrate_region(self, 
                            temperature: float, 
                            pressure: float, 
                            gas_gravity: float,
                            temperature_unit: str = "K",
                            pressure_unit: str = "Pa") -> bool:
        """
        Check if conditions are in hydrate formation region.
        
        Args:
            temperature (float): Temperature in specified unit
            pressure (float): Pressure in specified unit
            gas_gravity (float): Gas specific gravity (relative to air)
            temperature_unit (str): Temperature unit ('K', 'C', 'F')
            pressure_unit (str): Pressure unit ('Pa', 'kPa', 'bar', 'psi')
            
        Returns:
            bool: True if conditions are in hydrate formation region
        """
        # Calculate hydrate formation temperature at given pressure
        hydrate_temp = self.calculate_hydrate_formation_temperature(
            pressure, gas_gravity, pressure_unit
        )
        
        # Convert input temperature to K for comparison
        temperature_k = self._convert_temperature(temperature, temperature_unit, "K")
        
        # If actual temperature is below hydrate temperature, hydrates can form
        return temperature_k < hydrate_temp
    
    def calculate_inhibitor_requirement(self, 
                                      temperature: float, 
                                      hydrate_temperature: float, 
                                      inhibitor_type: str = "methanol",
                                      temperature_unit: str = "K") -> float:
        """
        Calculate required inhibitor concentration to prevent hydrate formation.
        
        Args:
            temperature (float): Operating temperature in specified unit
            hydrate_temperature (float): Hydrate formation temperature in specified unit
            inhibitor_type (str): Type of inhibitor ('methanol', 'glycol', 'salt')
            temperature_unit (str): Temperature unit ('K', 'C', 'F')
            
        Returns:
            float: Required inhibitor concentration in weight percent
        """
        # Convert temperatures to °C for calculation
        temperature_c = self._convert_temperature(temperature, temperature_unit, "C")
        hydrate_temp_c = self._convert_temperature(hydrate_temperature, temperature_unit, "C")
        
        # Temperature depression needed
        delta_t = hydrate_temp_c - temperature_c
        
        if delta_t <= 0:
            # No inhibitor needed
            return 0.0
        
        # Constants for different inhibitors (K·kg/g)
        inhibitor_constants = {
            "methanol": 1.2,
            "glycol": 1.8,
            "salt": 1.9
        }
        
        if inhibitor_type not in inhibitor_constants:
            raise ValueError(f"Unsupported inhibitor type: {inhibitor_type}")
        
        k = inhibitor_constants[inhibitor_type]
        
        # Calculate weight percent using modified Hammerschmidt equation
        # wt% = (K * ΔT) / (100 - K * ΔT)
        concentration = (k * delta_t) / (100 - k * delta_t) * 100
        
        return concentration


class WaxDepositionCalculator:
    """
    Calculator for wax deposition modeling and prediction.
    
    Wax deposition occurs when temperature drops below the wax appearance
    temperature (WAT), causing solid wax crystals to form and deposit on
    pipe walls.
    """
    
    def __init__(self):
        """Initialize the wax deposition calculator."""
        # Constants for correlations
        self.diffusion_constant = 1e-9  # m²/s, typical value for wax molecular diffusion
    
    def calculate_wax_appearance_temperature(self, 
                                           oil_properties: FluidProperties) -> float:
        """
        Calculate wax appearance temperature (WAT).
        
        Args:
            oil_properties (FluidProperties): Oil properties
            
        Returns:
            float: Wax appearance temperature in K
        """
        # Simple correlation based on pour point and wax content
        # In reality, more complex models would be used
        pour_point = oil_properties.pour_point
        wax_content = oil_properties.wax_content
        
        if wax_content <= 0:
            return pour_point
        
        # WAT is typically higher than pour point
        wat = pour_point + 10 * wax_content**0.5
        
        return wat
    
    def calculate_deposition_rate(self, 
                                oil_properties: FluidProperties,
                                pipe_properties: PipeProperties,
                                bulk_temperature: float,
                                wall_temperature: float,
                                velocity: float,
                                wat: float = None) -> float:
        """
        Calculate wax deposition rate.
        
        Args:
            oil_properties (FluidProperties): Oil properties
            pipe_properties (PipeProperties): Pipe properties
            bulk_temperature (float): Bulk oil temperature in K
            wall_temperature (float): Pipe wall temperature in K
            velocity (float): Oil velocity in m/s
            wat (float, optional): Wax appearance temperature in K
            
        Returns:
            float: Wax deposition rate in kg/(m²·s)
        """
        # If WAT not provided, calculate it
        if wat is None:
            wat = self.calculate_wax_appearance_temperature(oil_properties)
        
        # Check if wall temperature is below WAT
        if wall_temperature >= wat:
            return 0.0  # No deposition
        
        # Wax concentration at bulk and wall
        # Simplified: concentration proportional to temperature difference between WAT and fluid
        c_bulk = max(0, (wat - bulk_temperature) * oil_properties.wax_content * 0.01)
        c_wall = max(0, (wat - wall_temperature) * oil_properties.wax_content * 0.01)
        
        # Concentration gradient
        delta_c = c_wall - c_bulk
        
        if delta_c <= 0:
            return 0.0  # No deposition
        
        # Calculate Reynolds number
        reynolds = (oil_properties.density * velocity * pipe_properties.diameter) / oil_properties.viscosity
        
        # Calculate Sherwood number (mass transfer)
        # Using Sieder-Tate correlation
        schmidt = oil_properties.viscosity / (oil_properties.density * self.diffusion_constant)
        sherwood = 0.023 * reynolds**0.8 * schmidt**0.33
        
        # Mass transfer coefficient
        k_m = sherwood * self.diffusion_constant / pipe_properties.diameter
        
        # Wax deposition rate
        deposition_rate = k_m * delta_c * oil_properties.density
        
        return deposition_rate
    
    def calculate_deposition_thickness(self, 
                                     deposition_rate: float, 
                                     time: float, 
                                     wax_density: float = 900.0) -> float:
        """
        Calculate wax deposition thickness over time.
        
        Args:
            deposition_rate (float): Wax deposition rate in kg/(m²·s)
            time (float): Time period in seconds
            wax_density (float): Density of deposited wax in kg/m³
            
        Returns:
            float: Wax deposition thickness in m
        """
        if deposition_rate <= 0:
            return 0.0
        
        # Mass of wax deposited per unit area
        mass_per_area = deposition_rate * time
        
        # Thickness = mass / (area * density)
        thickness = mass_per_area / wax_density
        
        return thickness
    
    def calculate_critical_flow_rate(self, 
                                   pipe_properties: PipeProperties, 
                                   oil_properties: FluidProperties,
                                   wat: float = None,
                                   wall_temperature: float = None) -> float:
        """
        Calculate critical flow rate to prevent wax deposition.
        
        Args:
            pipe_properties (PipeProperties): Pipe properties
            oil_properties (FluidProperties): Oil properties
            wat (float, optional): Wax appearance temperature in K
            wall_temperature (float, optional): Pipe wall temperature in K
            
        Returns:
            float: Critical flow rate in m³/s
        """
        if wat is None:
            wat = self.calculate_wax_appearance_temperature(oil_properties)
        
        if wall_temperature is None:
            # Estimate wall temperature
            wall_temperature = wat - 5  # 5K below WAT
        
        # Target Reynolds number to ensure turbulent flow
        target_reynolds = 4000
        
        # Calculate critical velocity
        critical_velocity = (target_reynolds * oil_properties.viscosity) / (oil_properties.density * pipe_properties.diameter)
        
        # Calculate critical flow rate
        area = math.pi * (pipe_properties.diameter / 2)**2
        critical_flow_rate = critical_velocity * area
        
        return critical_flow_rate


class AsphalteneCalculator:
    """
    Calculator for asphaltene precipitation prediction.
    
    Asphaltenes are heavy, polar hydrocarbon components that can precipitate
    from crude oil due to changes in pressure, temperature, or composition.
    """
    
    def __init__(self):
        """Initialize the asphaltene calculator."""
        pass
    
    def calculate_asphaltene_stability(self, 
                                     oil_properties: FluidProperties, 
                                     pressure: float, 
                                     temperature: float) -> float:
        """
        Calculate asphaltene stability index.
        
        Args:
            oil_properties (FluidProperties): Oil properties
            pressure (float): Pressure in Pa
            temperature (float): Temperature in K
            
        Returns:
            float: Asphaltene stability index (>1 means stable)
        """
        # Extract relevant properties
        asphaltene_content = oil_properties.asphaltene_content
        
        if asphaltene_content <= 0:
            return float('inf')  # No asphaltenes, perfectly stable
        
        # Simple correlation based on empirical data
        # In reality, more complex thermodynamic models would be used
        
        # Pressure ratio (relative to bubble point)
        p_ratio = pressure / oil_properties.bubble_point
        
        # Base stability index
        base_stability = 1.0 + 2.0 * (1 - math.exp(-asphaltene_content * 20))
        
        # Pressure effect (decreases stability around bubble point)
        pressure_effect = 1.0 - 0.5 * math.exp(-((p_ratio - 1.0) ** 2) / 0.1)
        
        # Temperature effect (higher temperature generally increases stability)
        temperature_effect = 0.8 + 0.2 * min(1.0, temperature / 373.15)  # Normalized to 100°C
        
        # Combined stability index
        stability_index = base_stability * pressure_effect * temperature_effect
        
        return stability_index
    
    def calculate_precipitation_onset_pressure(self, 
                                             oil_properties: FluidProperties, 
                                             temperature: float) -> float:
        """
        Calculate asphaltene precipitation onset pressure.
        
        Args:
            oil_properties (FluidProperties): Oil properties
            temperature (float): Temperature in K
            
        Returns:
            float: Onset pressure in Pa
        """
        if oil_properties.asphaltene_content <= 0:
            return 0.0  # No asphaltenes, no precipitation
        
        # Correlation based on bubble point and temperature
        # In reality, would use more complex models or lab data
        bubble_point = oil_properties.bubble_point
        
        # Temperature effect (normalized to 100°C)
        temp_factor = 1.0 - 0.3 * min(1.0, temperature / 373.15)
        
        # Asphaltene content effect
        asph_factor = 0.5 + 0.5 * oil_properties.asphaltene_content / 0.1  # Normalized to 10% asphaltenes
        
        # Calculate onset pressure (typically above bubble point)
        onset_pressure = bubble_point * (1.0 + 0.2 * temp_factor * asph_factor)
        
        return onset_pressure
    
    def calculate_precipitation_amount(self, 
                                     oil_properties: FluidProperties, 
                                     pressure: float, 
                                     temperature: float,
                                     onset_pressure: float = None) -> float:
        """
        Calculate asphaltene precipitation amount.
        
        Args:
            oil_properties (FluidProperties): Oil properties
            pressure (float): Pressure in Pa
            temperature (float): Temperature in K
            onset_pressure (float, optional): Onset pressure in Pa
            
        Returns:
            float: Precipitation amount as weight fraction of total asphaltenes
        """
        if oil_properties.asphaltene_content <= 0:
            return 0.0  # No asphaltenes
        
        if onset_pressure is None:
            onset_pressure = self.calculate_precipitation_onset_pressure(oil_properties, temperature)
        
        if pressure >= onset_pressure:
            return 0.0  # Above onset pressure, no precipitation
        
        # Severity of pressure drop below onset
        pressure_ratio = pressure / onset_pressure
        
        # Simple model: precipitation increases as pressure drops below onset
        precipitation_fraction = (1.0 - pressure_ratio)**2 * 0.8  # Max 80% of asphaltenes can precipitate
        
        return precipitation_fraction * oil_properties.asphaltene_content


class ScaleCalculator:
    """
    Calculator for scale formation prediction.
    
    Scale is formed by precipitation of minerals from produced water,
    commonly including calcium carbonate, barium sulfate, and iron compounds.
    """
    
    def __init__(self):
        """Initialize the scale calculator."""
        # Scale type constants
        self.SCALE_CACO3 = "CaCO3"  # Calcium carbonate
        self.SCALE_BASO4 = "BaSO4"  # Barium sulfate
        self.SCALE_SRSO4 = "SrSO4"  # Strontium sulfate
        self.SCALE_CASO4 = "CaSO4"  # Calcium sulfate
        self.SCALE_FECO3 = "FeCO3"  # Iron carbonate
    
    def calculate_scaling_tendency(self, 
                                 water_properties: Dict[str, float], 
                                 temperature: float, 
                                 pressure: float,
                                 scale_type: str = "CaCO3") -> float:
        """
        Calculate scaling tendency for a specific scale type.
        
        Args:
            water_properties (Dict[str, float]): Water properties including ion concentrations
            temperature (float): Temperature in K
            pressure (float): Pressure in Pa
            scale_type (str): Type of scale to evaluate
            
        Returns:
            float: Scaling tendency (>1 indicates potential for scaling)
        """
        # Simple scaling tendency model based on saturation index
        # In reality, complex thermodynamic models would be used
        
        if scale_type == self.SCALE_CACO3:
            # Check if required ions are present
            if "Ca" not in water_properties or "CO3" not in water_properties:
                return 0.0
            
            # Simplified calculation for CaCO3 scaling tendency
            # Increased by high Ca and CO3 concentrations, temperature, and pH
            ca_ion = water_properties.get("Ca", 0)  # mg/L
            co3_ion = water_properties.get("CO3", 0)  # mg/L
            pH = water_properties.get("pH", 7)
            
            # Temperature effect (increases with temperature up to a point)
            temp_c = temperature - 273.15
            temp_factor = 1.0 + 0.005 * max(0, min(80, temp_c) - 20)
            
            # pH effect (increases with pH)
            ph_factor = 1.0 if pH < 7 else 1.0 + 0.2 * (pH - 7)
            
            # Calculate product and compare with temperature-dependent solubility
            ion_product = ca_ion * co3_ion * 1e-6  # Concentration product
            solubility = 5e-5 * (1.0 - 0.01 * min(80, temp_c))  # Simplified solubility
            
            scaling_tendency = (ion_product / solubility) * temp_factor * ph_factor
            
        elif scale_type == self.SCALE_BASO4:
            # Check if required ions are present
            if "Ba" not in water_properties or "SO4" not in water_properties:
                return 0.0
            
            # Simplified calculation for BaSO4 scaling tendency
            ba_ion = water_properties.get("Ba", 0)  # mg/L
            so4_ion = water_properties.get("SO4", 0)  # mg/L
            
            # BaSO4 is less affected by temperature and pH, more by pressure
            # Pressure effect (decreases with pressure)
            pressure_mpa = pressure / 1e6
            pressure_factor = 1.0 - 0.02 * min(20, pressure_mpa)
            
            # Calculate product and compare with solubility
            ion_product = ba_ion * so4_ion * 1e-6
            solubility = 1e-7  # Very low solubility
            
            scaling_tendency = (ion_product / solubility) * pressure_factor
            
        else:
            # For other scale types, implement similar calculations
            # Default to a simple placeholder
            scaling_tendency = 0.5
        
        return scaling_tendency
    
    def predict_scale_formation_rate(self, 
                                   scaling_tendency: float, 
                                   velocity: float, 
                                   temperature: float) -> float:
        """
        Predict scale formation rate based on scaling tendency.
        
        Args:
            scaling_tendency (float): Scaling tendency
            velocity (float): Fluid velocity in m/s
            temperature (float): Temperature in K
            
        Returns:
            float: Scale formation rate in mm/year
        """
        if scaling_tendency <= 1.0:
            return 0.0  # No scaling expected
        
        # Higher scaling tendencies result in faster scaling
        # Velocity affects through mass transfer (higher velocity can either
        # increase deposition or erosion depending on conditions)
        # Temperature generally increases rate
        
        # Base rate from scaling tendency
        base_rate = 0.1 * (scaling_tendency - 1.0)**1.5
        
        # Velocity effect (peaks at moderate velocities)
        vel_factor = 0.2 + 1.6 * velocity / (1.0 + velocity**2)
        
        # Temperature effect (increases with temperature)
        temp_c = temperature - 273.15
        temp_factor = math.exp(0.025 * min(80, temp_c))
        
        # Combined rate
        formation_rate = base_rate * vel_factor * temp_factor
        
        return formation_rate
    
    def calculate_inhibitor_dosage(self, 
                                 scaling_tendency: float, 
                                 water_volume: float, 
                                 inhibitor_efficiency: float = 0.8) -> float:
        """
        Calculate required scale inhibitor dosage.
        
        Args:
            scaling_tendency (float): Scaling tendency
            water_volume (float): Water volume in m³/day
            inhibitor_efficiency (float): Inhibitor efficiency (0-1)
            
        Returns:
            float: Required inhibitor dosage in L/day
        """
        if scaling_tendency <= 1.0:
            return 0.0  # No inhibitor needed
        
        # Base dosage increases with scaling tendency
        base_dosage = 10.0 * (scaling_tendency - 1.0)
        
        # Adjust for water volume
        volume_factor = water_volume / 100.0  # Normalized to 100 m³/day
        
        # Adjust for inhibitor efficiency
        efficiency_factor = 1.0 / max(0.1, inhibitor_efficiency)
        
        # Calculate required dosage
        dosage = base_dosage * volume_factor * efficiency_factor
        
        return dosage


class ErosionCorrosionCalculator:
    """
    Calculator for erosion and corrosion prediction.
    
    Erosion occurs due to particles impacting pipe walls, while
    corrosion is chemical/electrochemical degradation of metal.
    """
    
    def __init__(self):
        """Initialize the erosion and corrosion calculator."""
        # Constants for API RP 14E erosion velocity model
        self.c_factor = 100  # For continuous service
        self.erosion_constant = 0.00003  # m/s per year for carbon steel
    
    def calculate_erosional_velocity(self, 
                                   fluid_density: float, 
                                   is_multiphase: bool = False) -> float:
        """
        Calculate erosional velocity using API RP 14E model.
        
        Args:
            fluid_density (float): Fluid density in kg/m³
            is_multiphase (bool): Whether the flow is multiphase
            
        Returns:
            float: Erosional velocity in m/s
        """
        # Adjust C factor for multiphase flow
        c = self.c_factor
        if is_multiphase:
            c = c / 1.5  # Lower C-factor for multiphase flow
        
        # API RP 14E erosional velocity formula
        # V_e = C / sqrt(ρ) where ρ is density in kg/m³
        erosional_velocity = c / math.sqrt(fluid_density)
        
        return erosional_velocity
    
    def calculate_solid_erosion_rate(self, 
                                   velocity: float, 
                                   particle_size: float, 
                                   particle_concentration: float, 
                                   impact_angle: float,
                                   material_hardness: float = 300) -> float:
        """
        Calculate erosion rate due to solid particles.
        
        Args:
            velocity (float): Fluid velocity in m/s
            particle_size (float): Particle size in mm
            particle_concentration (float): Particle concentration in mg/L
            impact_angle (float): Impact angle in degrees
            material_hardness (float): Material hardness in Brinell (HB)
            
        Returns:
            float: Erosion rate in mm/year
        """
        # Convert angle to radians
        angle_rad = math.radians(impact_angle)
        
        # Adjust erosion constant based on material hardness
        # Softer materials erode faster
        k = self.erosion_constant * (300 / material_hardness)**0.7
        
        # Particle size effect (larger particles cause more erosion)
        size_factor = (particle_size / 0.5)**1.5
        
        # Impact angle effect (different for ductile and brittle materials)
        # This is for ductile materials like carbon steel
        angle_function = math.sin(angle_rad) * (1.0 + 5.0 * math.cos(angle_rad)**2)
        
        # Velocity effect (erosion increases with velocity^n, where n is typically 2-3)
        velocity_exponent = 2.5
        velocity_factor = (velocity / 10.0)**velocity_exponent
        
        # Concentration effect (linear with concentration)
        concentration_factor = particle_concentration / 100.0  # Normalized to 100 mg/L
        
        # Calculate erosion rate
        erosion_rate = k * size_factor * angle_function * velocity_factor * concentration_factor
        
        return erosion_rate
    
    def calculate_co2_corrosion_rate(self, 
                                   temperature: float, 
                                   co2_partial_pressure: float, 
                                   velocity: float, 
                                   ph: float = 6.5) -> float:
        """
        Calculate CO2 corrosion rate using de Waard-Milliams model.
        
        Args:
            temperature (float): Temperature in K
            co2_partial_pressure (float): CO2 partial pressure in bar
            velocity (float): Fluid velocity in m/s
            ph (float): pH of the water phase
            
        Returns:
            float: Corrosion rate in mm/year
        """
        # Convert temperature to °C
        temp_c = temperature - 273.15
        
        # Base rate calculation using de Waard-Milliams equation
        # log(v_corr) = 5.8 - 1710/T + 0.67*log(p_CO2)
        # Where v_corr is in mm/year, T is in K, p_CO2 is in bar
        
        # Temperature component
        temp_component = 1710 / (temperature)
        
        # CO2 pressure component
        pressure_component = 0.67 * math.log10(max(0.001, co2_partial_pressure))
        
        # Base corrosion rate
        log_corrosion_rate = 5.8 - temp_component + pressure_component
        base_rate = 10**log_corrosion_rate
        
        # pH correction (corrosion decreases as pH increases above 3.5)
        ph_factor = 1.0
        if ph > 3.5:
            ph_factor = 1.0 / ((ph - 3.5)**0.2)
        
        # Velocity correction (higher velocity increases corrosion)
        velocity_factor = 1.0 + 0.1 * velocity**0.6
        
        # Calculate final corrosion rate
        corrosion_rate = base_rate * ph_factor * velocity_factor
        
        return corrosion_rate
    
    def calculate_h2s_corrosion_rate(self, 
                                   temperature: float, 
                                   h2s_concentration: float, 
                                   velocity: float,
                                   chloride_concentration: float = 10000) -> float:
        """
        Calculate H2S corrosion rate.
        
        Args:
            temperature (float): Temperature in K
            h2s_concentration (float): H2S concentration in ppm
            velocity (float): Fluid velocity in m/s
            chloride_concentration (float): Chloride concentration in mg/L
            
        Returns:
            float: Corrosion rate in mm/year
        """
        # Convert temperature to °C
        temp_c = temperature - 273.15
        
        # Base rate (empirical correlation)
        # Increases with temperature, H2S concentration, and chlorides
        base_rate = 0.005 * (1 + 0.02 * temp_c)
        
        # H2S effect
        h2s_factor = 1.0 + 0.1 * math.log10(max(1.0, h2s_concentration))
        
        # Velocity effect
        velocity_factor = 1.0 + 0.05 * velocity
        
        # Chloride effect
        chloride_factor = 1.0 + 0.2 * math.log10(max(100.0, chloride_concentration) / 1000.0)
        
        # Calculate H2S corrosion rate
        corrosion_rate = base_rate * h2s_factor * velocity_factor * chloride_factor
        
        return corrosion_rate
    
    def calculate_combined_corrosion_rate(self, 
                                        co2_rate: float, 
                                        h2s_rate: float) -> float:
        """
        Calculate combined corrosion rate from CO2 and H2S.
        
        Args:
            co2_rate (float): CO2 corrosion rate in mm/year
            h2s_rate (float): H2S corrosion rate in mm/year
            
        Returns:
            float: Combined corrosion rate in mm/year
        """
        # Simplified approach: not purely additive as mechanisms interact
        # H2S forms protective films that can reduce CO2 corrosion
        if h2s_rate > co2_rate:
            return h2s_rate * 1.1  # Slightly higher than H2S alone
        else:
            # CO2 dominant, but H2S contributes
            return co2_rate + 0.2 * h2s_rate


class SlugFlowCalculator:
    """
    Calculator for slug flow prediction and analysis.
    
    Slug flow is an intermittent flow pattern in multiphase pipelines
    characterized by alternating liquid slugs and gas pockets.
    """
    
    def __init__(self):
        """Initialize the slug flow calculator."""
        self.g = 9.81  # m/s², gravitational acceleration
    
    def predict_flow_pattern(self, 
                           gas_velocity: float, 
                           liquid_velocity: float, 
                           pipe_diameter: float, 
                           liquid_density: float,
                           gas_density: float, 
                           liquid_viscosity: float,
                           surface_tension: float,
                           pipe_inclination: float = 0.0) -> FlowPattern:
        """
        Predict flow pattern for gas-liquid flow.
        
        Args:
            gas_velocity (float): Superficial gas velocity in m/s
            liquid_velocity (float): Superficial liquid velocity in m/s
            pipe_diameter (float): Pipe diameter in m
            liquid_density (float): Liquid density in kg/m³
            gas_density (float): Gas density in kg/m³
            liquid_viscosity (float): Liquid viscosity in Pa·s
            surface_tension (float): Surface tension in N/m
            pipe_inclination (float): Pipe inclination angle in degrees
            
        Returns:
            FlowPattern: Predicted flow pattern
        """
        # Calculate dimensionless parameters
        # Froude number
        fr_l = liquid_velocity / math.sqrt(self.g * pipe_diameter)
        fr_g = gas_velocity / math.sqrt(self.g * pipe_diameter)
        
        # Reynolds number for liquid
        re_l = (liquid_density * liquid_velocity * pipe_diameter) / liquid_viscosity
        
        # Weber number
        we_g = (gas_density * gas_velocity**2 * pipe_diameter) / surface_tension
        
        # Convert inclination to radians
        incline_rad = math.radians(pipe_inclination)
        
        # Simplified flow pattern map based on Taitel and Dukler
        # The boundaries between flow regimes are complex curves
        # This is a simplified approximation
        
        # Special case for vertical or near-vertical
        if abs(pipe_inclination) > 80:
            # Vertical flow patterns
            if gas_velocity < 0.1:
                return FlowPattern.BUBBLY
            elif gas_velocity < 10 and liquid_velocity > 0.1:
                return FlowPattern.SLUG
            elif gas_velocity > 10 and liquid_velocity > 0.5:
                return FlowPattern.CHURN
            else:
                return FlowPattern.ANNULAR
        
        # Horizontal and inclined flow patterns
        if fr_g < 0.5 and fr_l < 0.5:
            # Low velocities for both phases
            return FlowPattern.STRATIFIED
        
        if fr_g > 0.5 and fr_g < 20 and fr_l < 1.5:
            # Moderate gas velocity, low-moderate liquid velocity
            return FlowPattern.STRATIFIED_WAVY
        
        if fr_g < 5 and fr_l > 0.5 and fr_l < 3:
            # Intermittent flow (slug or elongated bubble)
            if re_l > 2000:
                return FlowPattern.SLUG
            else:
                return FlowPattern.ELONGATED_BUBBLE
        
        if fr_g > 20 and fr_l < 1.0:
            # High gas velocity, low liquid velocity
            if we_g > 20:
                return FlowPattern.ANNULAR_MIST
            else:
                return FlowPattern.ANNULAR
        
        if fr_l > 3:
            # High liquid velocity
            return FlowPattern.DISPERSED_BUBBLE
        
        # Default for cases not clearly falling into above categories
        return FlowPattern.SLUG
    
    def calculate_liquid_holdup(self, 
                              gas_velocity: float, 
                              liquid_velocity: float, 
                              pipe_diameter: float, 
                              pipe_inclination: float = 0.0,
                              flow_pattern: FlowPattern = None) -> float:
        """
        Calculate liquid holdup in multiphase flow.
        
        Args:
            gas_velocity (float): Superficial gas velocity in m/s
            liquid_velocity (float): Superficial liquid velocity in m/s
            pipe_diameter (float): Pipe diameter in m
            pipe_inclination (float): Pipe inclination angle in degrees
            flow_pattern (FlowPattern, optional): Flow pattern
            
        Returns:
            float: Liquid holdup (fraction of pipe volume occupied by liquid)
        """
        # Mixture velocity
        v_m = gas_velocity + liquid_velocity
        
        # No-slip holdup (volumetric ratio)
        lambda_l = liquid_velocity / v_m
        
        # Convert inclination to radians
        incline_rad = math.radians(pipe_inclination)
        
        # Calculate holdup based on flow pattern
        if flow_pattern == FlowPattern.DISPERSED_BUBBLE:
            # In dispersed bubble flow, holdup is close to no-slip value
            return max(0.05, min(0.95, lambda_l * 1.1))
        
        elif flow_pattern in [FlowPattern.STRATIFIED, FlowPattern.STRATIFIED_WAVY]:
            # Stratified flow has lower holdup due to slip
            slip_factor = 1.5 - 0.5 * math.sin(incline_rad)  # Slip increases in downward flow
            return max(0.05, min(0.95, lambda_l / slip_factor))
        
        elif flow_pattern in [FlowPattern.SLUG, FlowPattern.ELONGATED_BUBBLE]:
            # Slug flow has higher holdup
            # Simple correlation based on inclination and velocities
            inclination_factor = 1.0 + 0.2 * math.sin(incline_rad)  # Higher holdup in upward flow
            velocity_ratio = (1 + liquid_velocity) / (1 + gas_velocity)
            return max(0.2, min(0.9, lambda_l * inclination_factor * velocity_ratio**0.25))
        
        elif flow_pattern in [FlowPattern.ANNULAR, FlowPattern.ANNULAR_MIST]:
            # Annular flow has lower holdup
            return max(0.05, min(0.4, lambda_l * 0.9))
        
        else:
            # Default calculation for other flow patterns
            # Using Beggs and Brill correlation (simplified)
            froude_mixture = v_m**2 / (self.g * pipe_diameter)
            
            # Inclination correction
            inclination_factor = 1.0
            if abs(pipe_inclination) > 0:
                inclination_factor = 1.0 + 0.3 * math.sin(abs(incline_rad))
                if pipe_inclination < 0:  # Downward flow
                    inclination_factor = 1.0 / inclination_factor
            
            holdup = lambda_l * inclination_factor * (1 + froude_mixture)**(-0.1)
            
            return max(0.05, min(0.95, holdup))
    
    def calculate_slug_characteristics(self, 
                                     gas_velocity: float, 
                                     liquid_velocity: float, 
                                     pipe_diameter: float, 
                                     liquid_density: float,
                                     gas_density: float, 
                                     pipe_inclination: float = 0.0) -> Dict[str, float]:
        """
        Calculate slug flow characteristics.
        
        Args:
            gas_velocity (float): Superficial gas velocity in m/s
            liquid_velocity (float): Superficial liquid velocity in m/s
            pipe_diameter (float): Pipe diameter in m
            liquid_density (float): Liquid density in kg/m³
            gas_density (float): Gas density in kg/m³
            pipe_inclination (float): Pipe inclination angle in degrees
            
        Returns:
            Dict[str, float]: Slug characteristics
        """
        # Mixture velocity
        v_m = gas_velocity + liquid_velocity
        
        # No-slip holdup
        lambda_l = liquid_velocity / v_m
        
        # Froude number
        fr_m = v_m / math.sqrt(self.g * pipe_diameter)
        
        # Convert inclination to radians
        incline_rad = math.radians(pipe_inclination)
        
        # Calculate slug translational velocity
        # Using Nicklin correlation
        c0 = 1.2  # Distribution coefficient
        slug_velocity = c0 * v_m + 0.35 * math.sqrt(self.g * pipe_diameter)
        
        # Slug frequency
        # Using Gregory and Scott correlation
        if pipe_diameter < 0.05:
            # Small pipes
            slug_frequency = 0.0226 * (lambda_l * fr_m**1.2) / pipe_diameter
        else:
            # Standard correlation
            slug_frequency = 0.0226 * (lambda_l * fr_m**0.539 / pipe_diameter) * \
                             (1.0 + 0.2 * math.sin(incline_rad))
        
        # Limit frequency to realistic values
        slug_frequency = min(2.0, max(0.05, slug_frequency))
        
        # Slug length
        # Using empirical correlations
        if fr_m < 2:
            # Short slugs at low velocities
            slug_length = 12 * pipe_diameter
        else:
            # Longer slugs at higher velocities
            slug_length = (12 + 5 * (fr_m - 2)) * pipe_diameter
        
        # Liquid holdup in slug body
        slug_holdup = min(0.95, max(0.75, lambda_l * 1.2))
        
        # Film holdup
        film_holdup = max(0.05, lambda_l * 0.5)
        
        # Pressure fluctuation amplitude
        # Due to hydrostatic head difference between slug and film
        pressure_fluctuation = (slug_holdup - film_holdup) * liquid_density * self.g * pipe_diameter * math.sin(incline_rad)
        
        # Force on pipe bends (simplified)
        force_on_bend = liquid_density * slug_holdup * slug_velocity**2 * math.pi * (pipe_diameter/2)**2
        
        return {
            "slug_velocity": slug_velocity,
            "slug_frequency": slug_frequency,
            "slug_length": slug_length,
            "slug_holdup": slug_holdup,
            "film_holdup": film_holdup,
            "pressure_fluctuation": pressure_fluctuation,
            "force_on_bend": force_on_bend
        }
    
    def calculate_severe_slugging_potential(self, 
                                          gas_velocity: float, 
                                          liquid_velocity: float, 
                                          riser_height: float, 
                                          riser_diameter: float, 
                                          upstream_inclination: float) -> Dict[str, Any]:
        """
        Calculate potential for severe slugging in riser systems.
        
        Args:
            gas_velocity (float): Superficial gas velocity in m/s
            liquid_velocity (float): Superficial liquid velocity in m/s
            riser_height (float): Riser height in m
            riser_diameter (float): Riser diameter in m
            upstream_inclination (float): Inclination of pipeline upstream of riser
            
        Returns:
            Dict[str, Any]: Severe slugging assessment
        """
        # Calculate critical Froude number for severe slugging
        fr_g = gas_velocity / math.sqrt(self.g * riser_diameter)
        fr_l = liquid_velocity / math.sqrt(self.g * riser_diameter)
        
        # Convert upstream inclination to radians
        upstream_rad = math.radians(upstream_inclination)
        
        # Severe slugging criteria
        # 1. Low gas and liquid velocities (Fr < 1)
        # 2. Downward inclination upstream of riser
        # 3. Sufficient riser height
        
        # Calculate severe slugging potential
        if fr_g < 0.3 and fr_l < 0.5 and upstream_inclination < -2 and riser_height > 10 * riser_diameter:
            severity = "High"
            cycle_period = estimate_slugging_cycle_period(riser_height, gas_velocity, liquid_velocity)
            slug_length = 0.8 * riser_height  # Typically 80% of riser height
        elif fr_g < 0.6 and fr_l < 1.0 and upstream_inclination < 0:
            severity = "Medium"
            cycle_period = estimate_slugging_cycle_period(riser_height, gas_velocity, liquid_velocity) * 0.7
            slug_length = 0.5 * riser_height
        else:
            severity = "Low"
            cycle_period = 0
            slug_length = 0
        
        # Calculate pressure fluctuation amplitude
        if severity != "Low":
            # Estimate pressure fluctuation due to hydrostatic head
            # Pressure increase during slug formation phase
            pressure_fluctuation = 1000 * self.g * slug_length  # Assuming water density
        else:
            pressure_fluctuation = 0
        
        return {
            "severity": severity,
            "cycle_period": cycle_period,
            "slug_length": slug_length,
            "pressure_fluctuation": pressure_fluctuation,
            "mitigation_required": severity != "Low"
        }


def estimate_slugging_cycle_period(riser_height, gas_velocity, liquid_velocity):
    """
    Estimate slugging cycle period for a riser.
    
    Args:
        riser_height (float): Riser height in m
        gas_velocity (float): Superficial gas velocity in m/s
        liquid_velocity (float): Superficial liquid velocity in m/s
        
    Returns:
        float: Estimated cycle period in seconds
    """
    if gas_velocity < 0.001:
        return 9999  # Very long period for extremely low gas rates
    
    # Simplified estimate based on empirical observations
    # Cycle involves four phases: liquid accumulation, slug formation,
    # slug production, and gas blowdown
    
    # Liquid accumulation and slug formation time
    formation_time = riser_height / (2 * liquid_velocity)
    
    # Gas blowdown time
    blowdown_time = riser_height / (5 * gas_velocity)
    
    # Total cycle period
    cycle_period = formation_time + blowdown_time
    
    return min(3600, max(10, cycle_period))  # Limit to reasonable range


class FlowAssurance:
    """
    Main flow assurance class that integrates various calculators.
    
    This class provides a unified interface to access all flow assurance
    calculations and analyses in the Openfoam_Simulator application.
    """
    
    def __init__(self):
        """Initialize the flow assurance module."""
        self.base_calculator = FlowAssuranceCalculator()
        self.hydrate_calculator = HydrateCalculator()
        self.wax_calculator = WaxDepositionCalculator()
        self.asphaltene_calculator = AsphalteneCalculator()
        self.scale_calculator = ScaleCalculator()
        self.erosion_calculator = ErosionCorrosionCalculator()
        self.slug_calculator = SlugFlowCalculator()
        
        logger.info("Flow assurance module initialized")
    
    def analyze_pipeline(self, 
                       fluid_properties: Union[FluidProperties, Dict[str, float]], 
                       pipe_properties: Union[PipeProperties, Dict[str, float]],
                       ambient_conditions: Union[AmbientConditions, Dict[str, float]],
                       flow_rate: float,
                       distance_points: int = 10) -> Dict[str, Any]:
        """
        Perform comprehensive flow assurance analysis for a pipeline.
        
        Args:
            fluid_properties: Fluid properties data
            pipe_properties: Pipe properties data
            ambient_conditions: Ambient conditions data
            flow_rate (float): Flow rate in m³/s
            distance_points (int): Number of points along pipeline for analysis
            
        Returns:
            Dict[str, Any]: Comprehensive flow assurance analysis results
        """
        # Convert dictionary inputs to data classes if needed
        if isinstance(fluid_properties, dict):
            fluid_properties = FluidProperties(**fluid_properties)
        
        if isinstance(pipe_properties, dict):
            pipe_properties = PipeProperties(**pipe_properties)
        
        if isinstance(ambient_conditions, dict):
            ambient_conditions = AmbientConditions(**ambient_conditions)
        
        # Calculate velocity
        velocity = self.base_calculator.calculate_velocity(pipe_properties.diameter, flow_rate)
        
        # Calculate Reynolds number
        reynolds = self.base_calculator.calculate_reynolds_number(
            velocity, pipe_properties.diameter, fluid_properties.density, fluid_properties.viscosity
        )
        
        # Calculate friction factor
        friction_factor = self.base_calculator.calculate_friction_factor(
            reynolds, pipe_properties.roughness, pipe_properties.diameter
        )
        
        # Create distance points along pipeline
        distances = [i * pipe_properties.length / distance_points for i in range(distance_points + 1)]
        
        # Initialize results containers
        pressure_profile = [ambient_conditions.pressure]
        temperature_profile = [ambient_conditions.temperature]
        hydrate_risk_profile = []
        wax_risk_profile = []
        erosion_profile = []
        corrosion_profile = []
        asphaltene_risk_profile = []
        
        # Calculate profiles along pipeline
        current_pressure = ambient_conditions.pressure
        current_temperature = ambient_conditions.temperature
        
        for i in range(1, len(distances)):
            segment_length = distances[i] - distances[i-1]
            
            # Pressure drop in segment
            pressure_drop = self.base_calculator.calculate_pressure_drop(
                segment_length, pipe_properties.diameter, fluid_properties.density, 
                velocity, friction_factor, pipe_properties.incline_angle
            )
            current_pressure -= pressure_drop
            pressure_profile.append(current_pressure)
            
            # Temperature change in segment
            # Calculate heat transfer coefficient
            h_internal = calculate_internal_heat_transfer_coefficient(
                reynolds, pipe_properties.diameter, fluid_properties.thermal_conductivity,
                fluid_properties.specific_heat, fluid_properties.viscosity
            )
            
            u_value = self.base_calculator.calculate_overall_heat_transfer_coefficient(
                pipe_properties, h_internal, ambient_conditions.heat_transfer_coefficient
            )
            
            heat_transfer = self.base_calculator.calculate_heat_transfer(
                segment_length, u_value, current_temperature, ambient_conditions.temperature
            )
            
            # Temperature change from heat transfer
            mass_flow = flow_rate * fluid_properties.density
            specific_heat = fluid_properties.specific_heat
            
            if mass_flow > 0 and specific_heat > 0:
                delta_t = heat_transfer / (mass_flow * specific_heat)
                current_temperature -= delta_t
            
            temperature_profile.append(current_temperature)
            
            # Hydrate risk assessment
            if fluid_properties.gas_oil_ratio > 0:
                # Only relevant for gas-containing fluids
                # Simplified gas gravity calculation
                gas_gravity = 0.6 + 0.1 * (fluid_properties.molecular_weight / 20.0)
                
                # Check if in hydrate region
                hydrate_risk = self.hydrate_calculator.is_in_hydrate_region(
                    current_temperature, current_pressure, gas_gravity
                )
                
                if hydrate_risk:
                    # Calculate hydrate formation temperature
                    hydrate_temperature = self.hydrate_calculator.calculate_hydrate_formation_temperature(
                        current_pressure, gas_gravity
                    )
                    
                    # Calculate temperature margin
                    temp_margin = current_temperature - hydrate_temperature
                    
                    hydrate_risk_profile.append({
                        "position": distances[i],
                        "risk_level": "High" if temp_margin < -5 else "Medium",
                        "hydrate_temperature": hydrate_temperature,
                        "temperature_margin": temp_margin
                    })
                else:
                    hydrate_risk_profile.append({
                        "position": distances[i],
                        "risk_level": "Low",
                        "hydrate_temperature": 0,
                        "temperature_margin": 0
                    })
            
            # Wax deposition risk
            if fluid_properties.wax_content > 0:
                # Calculate wax appearance temperature
                wat = self.wax_calculator.calculate_wax_appearance_temperature(fluid_properties)
                
                # Get wall temperature (simplified)
                wall_temp = current_temperature - 2  # Assume 2K lower than bulk fluid
                
                if wall_temp < wat:
                    # Calculate deposition rate
                    deposition_rate = self.wax_calculator.calculate_deposition_rate(
                        fluid_properties, pipe_properties, current_temperature, 
                        wall_temp, velocity, wat
                    )
                    
                    # Calculate deposition thickness over time (e.g., 30 days)
                    thickness_30days = self.wax_calculator.calculate_deposition_thickness(
                        deposition_rate, 30 * 24 * 3600
                    )
                    
                    wax_risk_profile.append({
                        "position": distances[i],
                        "risk_level": "High" if thickness_30days > 0.005 else "Medium",
                        "wax_appearance_temperature": wat,
                        "temperature_margin": wall_temp - wat,
                        "deposition_rate": deposition_rate,
                        "thickness_30days": thickness_30days
                    })
                else:
                    wax_risk_profile.append({
                        "position": distances[i],
                        "risk_level": "Low",
                        "wax_appearance_temperature": wat,
                        "temperature_margin": wall_temp - wat,
                        "deposition_rate": 0,
                        "thickness_30days": 0
                    })
            
            # Erosion risk
            erosional_velocity = self.erosion_calculator.calculate_erosional_velocity(
                fluid_properties.density, fluid_properties.water_cut > 0
            )
            
            erosion_ratio = velocity / erosional_velocity
            
            erosion_profile.append({
                "position": distances[i],
                "erosion_ratio": erosion_ratio,
                "risk_level": "High" if erosion_ratio > 0.8 else ("Medium" if erosion_ratio > 0.5 else "Low")
            })
            
            # Corrosion risk
            if fluid_properties.co2_content > 0 or fluid_properties.h2s_content > 0:
                # Calculate partial pressures
                co2_pp = current_pressure * (fluid_properties.co2_content / 1e6)  # Convert from ppm to bar
                h2s_pp = current_pressure * (fluid_properties.h2s_content / 1e6)  # Convert from ppm to bar
                
                # Calculate corrosion rates
                co2_rate = self.erosion_calculator.calculate_co2_corrosion_rate(
                    current_temperature, co2_pp * 1e-5, velocity  # Convert to bar
                )
                
                h2s_rate = self.erosion_calculator.calculate_h2s_corrosion_rate(
                    current_temperature, fluid_properties.h2s_content, velocity, 
                    fluid_properties.salinity
                )
                
                total_rate = self.erosion_calculator.calculate_combined_corrosion_rate(
                    co2_rate, h2s_rate
                )
                
                corrosion_profile.append({
                    "position": distances[i],
                    "co2_rate": co2_rate,
                    "h2s_rate": h2s_rate,
                    "total_rate": total_rate,
                    "risk_level": "High" if total_rate > 1.0 else ("Medium" if total_rate > 0.1 else "Low")
                })
            
            # Asphaltene risk
            if fluid_properties.asphaltene_content > 0:
                stability = self.asphaltene_calculator.calculate_asphaltene_stability(
                    fluid_properties, current_pressure, current_temperature
                )
                
                asphaltene_risk_profile.append({
                    "position": distances[i],
                    "stability_index": stability,
                    "risk_level": "High" if stability < 0.7 else ("Medium" if stability < 1.0 else "Low")
                })
        
        # Compile all results
        results = {
            "flow_data": {
                "velocity": velocity,
                "reynolds": reynolds,
                "flow_regime": "Turbulent" if reynolds > 4000 else ("Transitional" if reynolds > 2300 else "Laminar"),
                "friction_factor": friction_factor,
            },
            "profiles": {
                "distance": distances,
                "pressure": pressure_profile,
                "temperature": temperature_profile,
            },
            "risks": {
                "hydrate": hydrate_risk_profile if fluid_properties.gas_oil_ratio > 0 else [],
                "wax": wax_risk_profile if fluid_properties.wax_content > 0 else [],
                "erosion": erosion_profile,
                "corrosion": corrosion_profile if (fluid_properties.co2_content > 0 or fluid_properties.h2s_content > 0) else [],
                "asphaltene": asphaltene_risk_profile if fluid_properties.asphaltene_content > 0 else [],
            }
        }
        
        # Add slug flow analysis if liquid and gas are present
        if fluid_properties.water_cut > 0 or fluid_properties.gas_oil_ratio > 0:
            # Simplified calculation of superficial velocities
            liquid_fraction = fluid_properties.water_cut if fluid_properties.water_cut > 0 else 0.5
            gas_fraction = 1 - liquid_fraction
            
            liquid_velocity = velocity * liquid_fraction
            gas_velocity = velocity * gas_fraction
            
            # Predict flow pattern
            flow_pattern = self.slug_calculator.predict_flow_pattern(
                gas_velocity, liquid_velocity, pipe_properties.diameter,
                fluid_properties.density, 0.5 * fluid_properties.density,  # Simplified gas density
                fluid_properties.viscosity, fluid_properties.surface_tension,
                pipe_properties.incline_angle
            )
            
            results["flow_data"]["flow_pattern"] = flow_pattern.value
            
            # Calculate liquid holdup
            liquid_holdup = self.slug_calculator.calculate_liquid_holdup(
                gas_velocity, liquid_velocity, pipe_properties.diameter,
                pipe_properties.incline_angle, flow_pattern
            )
            
            results["flow_data"]["liquid_holdup"] = liquid_holdup
            
            # Add slug flow characteristics if relevant
            if flow_pattern in [FlowPattern.SLUG, FlowPattern.ELONGATED_BUBBLE]:
                slug_chars = self.slug_calculator.calculate_slug_characteristics(
                    gas_velocity, liquid_velocity, pipe_properties.diameter,
                    fluid_properties.density, 0.5 * fluid_properties.density,
                    pipe_properties.incline_angle
                )
                
                results["flow_data"]["slug_characteristics"] = slug_chars
        
        return results
    
    def generate_hydrate_curve(self, 
                             gas_gravity: float = 0.6, 
                             pressure_range: Tuple[float, float] = (1e5, 2e7),
                             num_points: int = 20) -> Dict[str, List[float]]:
        """
        Generate hydrate formation curve (pressure vs temperature).
        
        Args:
            gas_gravity (float): Gas specific gravity (relative to air)
            pressure_range (Tuple[float, float]): Min and max pressure in Pa
            num_points (int): Number of points on the curve
            
        Returns:
            Dict[str, List[float]]: Pressure and corresponding hydrate temperature
        """
        pressures = []
        temperatures = []
        
        # Generate logarithmically spaced pressure points
        log_min = math.log10(pressure_range[0])
        log_max = math.log10(pressure_range[1])
        log_step = (log_max - log_min) / (num_points - 1)
        
        for i in range(num_points):
            pressure = 10**(log_min + i * log_step)
            temperature = self.hydrate_calculator.calculate_hydrate_formation_temperature(
                pressure, gas_gravity
            )
            
            pressures.append(pressure)
            temperatures.append(temperature)
        
        return {
            "pressure": pressures,
            "temperature": temperatures
        }
    
    def calculate_inhibitor_requirements(self, 
                                       hydrate_curve: Dict[str, List[float]],
                                       operating_temperature: float,
                                       inhibitor_type: str = "methanol") -> Dict[str, List[float]]:
        """
        Calculate inhibitor requirements for hydrate prevention.
        
        Args:
            hydrate_curve (Dict[str, List[float]]): Hydrate formation curve
            operating_temperature (float): Operating temperature in K
            inhibitor_type (str): Type of inhibitor ('methanol', 'glycol', 'salt')
            
        Returns:
            Dict[str, List[float]]: Inhibitor requirements at each pressure point
        """
        pressures = hydrate_curve["pressure"]
        hydrate_temps = hydrate_curve["temperature"]
        
        inhibitor_requirements = []
        
        for pressure, hydrate_temp in zip(pressures, hydrate_temps):
            if operating_temperature >= hydrate_temp:
                # No inhibitor needed
                inhibitor_requirements.append(0.0)
            else:
                # Calculate required inhibitor
                requirement = self.hydrate_calculator.calculate_inhibitor_requirement(
                    operating_temperature, hydrate_temp, inhibitor_type
                )
                
                inhibitor_requirements.append(requirement)
        
        return {
            "pressure": pressures,
            "inhibitor_requirement": inhibitor_requirements
        }


def calculate_internal_heat_transfer_coefficient(reynolds, diameter, fluid_conductivity, 
                                             fluid_specific_heat, fluid_viscosity):
    """
    Calculate internal heat transfer coefficient using Dittus-Boelter correlation.
    
    Args:
        reynolds (float): Reynolds number
        diameter (float): Pipe diameter in m
        fluid_conductivity (float): Fluid thermal conductivity in W/(m·K)
        fluid_specific_heat (float): Fluid specific heat in J/(kg·K)
        fluid_viscosity (float): Fluid viscosity in Pa·s
        
    Returns:
        float: Heat transfer coefficient in W/(m²·K)
    """
    # Calculate Prandtl number
    prandtl = (fluid_specific_heat * fluid_viscosity) / fluid_conductivity
    
    # Dittus-Boelter correlation for turbulent flow
    if reynolds > 4000 and prandtl > 0.7 and prandtl < 160:
        # For cooling (n=0.3)
        nusselt = 0.023 * reynolds**0.8 * prandtl**0.3
        h = nusselt * fluid_conductivity / diameter
        return h
    
    # Laminar flow
    elif reynolds < 2300:
        # Constant Nusselt number for laminar flow
        nusselt = 3.66
        h = nusselt * fluid_conductivity / diameter
        return h
    
    # Transitional flow - interpolate
    else:
        # Calculate at both ends and interpolate
        nu_lam = 3.66
        h_lam = nu_lam * fluid_conductivity / diameter
        
        nu_turb = 0.023 * 4000**0.8 * prandtl**0.3
        h_turb = nu_turb * fluid_conductivity / diameter
        
        # Linear interpolation
        t = (reynolds - 2300) / (4000 - 2300)
        h = h_lam * (1 - t) + h_turb * t
        
        return h