#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spill models and quantification for Openfoam_Simulator.

This module provides classes and functions for modeling and quantifying
spills in the oil & gas industry, including:
- Surface spills on water or land
- Subsurface spills (underwater)
- Jet spills (high pressure releases)

These models are integrated with OpenFOAM for CFD simulations of spill
behavior, spread, and environmental impact assessment.
"""

import os
import math
import numpy as np
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

# Import utility modules
from ..utils.logger import get_logger
from ..utils.unit_converter import convert_units
from ..config import get_value, set_value

# Import OpenFOAM integration modules
from ..openfoam_integration.case_manager import CaseManager
from ..openfoam_integration.solver_manager import SolverManager
from ..openfoam_integration.boundary_conditions import BoundaryCondition
from ..openfoam_integration.transport_models import TransportModel

logger = get_logger(__name__)


class FluidProperties:
    """Class to store and calculate fluid properties relevant to spill modeling."""
    
    def __init__(self, name: str, density: float, viscosity: float, surface_tension: float,
                 vapor_pressure: float, boiling_point: float, pour_point: float = None, 
                 api_gravity: float = None, flash_point: float = None):
        """
        Initialize fluid properties.
        
        Args:
            name (str): Fluid name
            density (float): Density in kg/m³
            viscosity (float): Dynamic viscosity in Pa·s
            surface_tension (float): Surface tension in N/m
            vapor_pressure (float): Vapor pressure in Pa
            boiling_point (float): Boiling point in K
            pour_point (float, optional): Pour point in K
            api_gravity (float, optional): API gravity
            flash_point (float, optional): Flash point in K
        """
        self.name = name
        self.density = density
        self.viscosity = viscosity
        self.surface_tension = surface_tension
        self.vapor_pressure = vapor_pressure
        self.boiling_point = boiling_point
        self.pour_point = pour_point
        self.api_gravity = api_gravity
        self.flash_point = flash_point
        
        # Derived properties
        self.kinematic_viscosity = self.viscosity / self.density
    
    @classmethod
    def from_database(cls, fluid_name: str) -> 'FluidProperties':
        """
        Create a FluidProperties instance from the database.
        
        Args:
            fluid_name (str): Name of the fluid to look up
            
        Returns:
            FluidProperties: Instance with properties from database
            
        Raises:
            ValueError: If fluid is not found in database
        """
        # Load fluid database
        database_path = get_value('industry.fluid_database_path', 'data/fluids.json')
        database_path = os.path.join(os.path.dirname(__file__), '..', '..', database_path)
        
        try:
            with open(database_path, 'r') as f:
                fluids_data = json.load(f)
            
            if fluid_name not in fluids_data:
                raise ValueError(f"Fluid '{fluid_name}' not found in database")
            
            fluid_data = fluids_data[fluid_name]
            
            return cls(
                name=fluid_name,
                density=fluid_data.get('density', 0.0),
                viscosity=fluid_data.get('viscosity', 0.0),
                surface_tension=fluid_data.get('surface_tension', 0.0),
                vapor_pressure=fluid_data.get('vapor_pressure', 0.0),
                boiling_point=fluid_data.get('boiling_point', 0.0),
                pour_point=fluid_data.get('pour_point'),
                api_gravity=fluid_data.get('api_gravity'),
                flash_point=fluid_data.get('flash_point')
            )
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading fluid database: {e}")
            
            # Return default properties for common fluids
            default_properties = {
                'Crude Oil': cls(
                    name='Crude Oil',
                    density=850.0,
                    viscosity=0.05,
                    surface_tension=0.025,
                    vapor_pressure=30000.0,
                    boiling_point=633.0,
                    pour_point=270.0,
                    api_gravity=35.0,
                    flash_point=335.0
                ),
                'Diesel': cls(
                    name='Diesel',
                    density=820.0,
                    viscosity=0.003,
                    surface_tension=0.023,
                    vapor_pressure=1000.0,
                    boiling_point=553.0,
                    pour_point=260.0,
                    api_gravity=38.0,
                    flash_point=330.0
                ),
                'Gasoline': cls(
                    name='Gasoline',
                    density=750.0,
                    viscosity=0.0005,
                    surface_tension=0.022,
                    vapor_pressure=55000.0,
                    boiling_point=373.0,
                    api_gravity=60.0,
                    flash_point=233.0
                ),
                'Natural Gas': cls(
                    name='Natural Gas',
                    density=0.8,
                    viscosity=1.1e-5,
                    surface_tension=0.0,
                    vapor_pressure=101325.0,
                    boiling_point=111.0,
                    api_gravity=None,
                    flash_point=None
                )
            }
            
            if fluid_name in default_properties:
                return default_properties[fluid_name]
            
            # Return default crude oil properties if specific fluid not found
            logger.warning(f"Using default properties for {fluid_name}")
            return default_properties['Crude Oil']
    
    def adjust_for_temperature(self, temperature: float):
        """
        Adjust properties for a given temperature.
        
        Args:
            temperature (float): Temperature in K
        """
        # Simple temperature corrections for density and viscosity
        # (These are approximations and would be more complex in a real implementation)
        
        # For density: ρ(T) = ρ(T₀) * (1 - β * (T - T₀))
        # where β is the thermal expansion coefficient
        # Approximate β as 0.001 /K for oils
        beta = 0.001
        reference_temp = 293.15  # 20°C in K
        self.density = self.density * (1 - beta * (temperature - reference_temp))
        
        # For viscosity: μ(T) = μ(T₀) * exp(E/R * (1/T - 1/T₀))
        # where E is the activation energy and R is the gas constant
        # Approximate E/R as 2000 K for oils
        activation_energy_over_r = 2000
        self.viscosity = self.viscosity * math.exp(
            activation_energy_over_r * (1/temperature - 1/reference_temp)
        )
        
        # Update derived properties
        self.kinematic_viscosity = self.viscosity / self.density
        
        # For surface tension: σ(T) = σ(T₀) - γ * (T - T₀)
        # where γ is temperature coefficient of surface tension
        # Approximate γ as 0.0001 N/(m·K) for oils
        gamma = 0.0001
        self.surface_tension = max(0.001, self.surface_tension - gamma * (temperature - reference_temp))
        
        # Vapor pressure using Clausius-Clapeyron relation
        # ln(P₂/P₁) = (ΔH_vap/R)*(1/T₁ - 1/T₂)
        # ΔH_vap/R is approximately 5000 K for hydrocarbons
        if self.vapor_pressure > 0 and self.boiling_point > 0:
            heat_of_vap_over_r = 5000
            self.vapor_pressure = self.vapor_pressure * math.exp(
                heat_of_vap_over_r * (1/reference_temp - 1/temperature)
            )


class EnvironmentProperties:
    """Class to store and calculate environmental properties for spill modeling."""
    
    def __init__(self, env_type: str, temperature: float, wind_speed: float = 0.0, 
                 water_temperature: float = None, current_speed: float = 0.0, 
                 wave_height: float = 0.0, salinity: float = 0.0, 
                 soil_type: str = None, permeability: float = None):
        """
        Initialize environment properties.
        
        Args:
            env_type (str): Environment type (e.g., "Water (Ocean)", "Land (Soil)")
            temperature (float): Ambient temperature in K
            wind_speed (float): Wind speed in m/s
            water_temperature (float, optional): Water temperature in K
            current_speed (float, optional): Water current speed in m/s
            wave_height (float, optional): Wave height in m
            salinity (float, optional): Water salinity in ppt
            soil_type (str, optional): Soil type for land spills
            permeability (float, optional): Soil permeability in m²
        """
        self.env_type = env_type
        self.temperature = temperature
        self.wind_speed = wind_speed
        
        # Water-specific properties
        self.is_water = env_type.startswith("Water")
        self.water_temperature = water_temperature if water_temperature is not None else temperature
        self.current_speed = current_speed if self.is_water else 0.0
        self.wave_height = wave_height if self.is_water else 0.0
        self.salinity = salinity if self.is_water else 0.0
        
        # Water density and viscosity (adjusted for temperature and salinity)
        if self.is_water:
            # Approximate density of water: ρ = 1000 - 0.2 * (T - 277)
            self.water_density = 1000.0 - 0.2 * (self.water_temperature - 277.0)
            # Add salinity effect: ρ = ρ + 0.8 * salinity
            self.water_density += 0.8 * self.salinity
            
            # Approximate viscosity of water: μ = 1.8e-3 * exp(-0.03 * (T - 277))
            self.water_viscosity = 1.8e-3 * math.exp(-0.03 * (self.water_temperature - 277.0))
        else:
            self.water_density = 0.0
            self.water_viscosity = 0.0
        
        # Land-specific properties
        self.is_land = env_type.startswith("Land")
        self.soil_type = soil_type if self.is_land else None
        self.permeability = permeability if self.is_land and permeability is not None else 0.0
    
    @property
    def beaufort_scale(self) -> int:
        """
        Get Beaufort scale number for the wind speed.
        
        Returns:
            int: Beaufort scale number (0-12)
        """
        beaufort_thresholds = [0.5, 1.5, 3.3, 5.5, 8.0, 10.8, 13.9, 17.2, 20.7, 24.5, 28.4, 32.6]
        for i, threshold in enumerate(beaufort_thresholds):
            if self.wind_speed < threshold:
                return i
        return 12  # Hurricane
    
    def get_sea_state(self) -> str:
        """
        Get the sea state description.
        
        Returns:
            str: Sea state description
        """
        if not self.is_water:
            return "N/A (Land environment)"
        
        # Douglas Sea Scale
        if self.wave_height < 0.1:
            return "Calm (glassy)"
        elif self.wave_height < 0.5:
            return "Calm (rippled)"
        elif self.wave_height < 1.25:
            return "Smooth"
        elif self.wave_height < 2.5:
            return "Slight"
        elif self.wave_height < 4.0:
            return "Moderate"
        elif self.wave_height < 6.0:
            return "Rough"
        elif self.wave_height < 9.0:
            return "Very rough"
        elif self.wave_height < 14.0:
            return "High"
        else:
            return "Very high"


class SpillModel:
    """Base class for spill models."""
    
    def __init__(self, spill_type: str, fluid: str, rate: float, total_mass: float, 
                 opening_diameter: float, temperature: float, pressure: float,
                 environment: EnvironmentProperties):
        """
        Initialize the spill model.
        
        Args:
            spill_type (str): Type of spill (Surface, Subsurface, Jet)
            fluid (str): Type of fluid
            rate (float): Spill rate in kg/s
            total_mass (float): Total mass in kg
            opening_diameter (float): Opening diameter in m
            temperature (float): Fluid temperature in K
            pressure (float): Fluid pressure in Pa
            environment (EnvironmentProperties): Environmental properties
        """
        self.spill_type = spill_type
        self.fluid_name = fluid
        self.spill_rate = rate
        self.total_mass = total_mass
        self.opening_diameter = opening_diameter
        self.temperature = temperature
        self.pressure = pressure
        self.environment = environment
        
        # Calculate duration
        self.duration = total_mass / rate if rate > 0 else float('inf')
        
        # Get fluid properties (adjusted for temperature)
        self.fluid = FluidProperties.from_database(fluid)
        self.fluid.adjust_for_temperature(temperature)
        
        # Results storage
        self.results = {}
        
        # OpenFOAM case manager
        self.case_manager = None
    
    def setup_case(self, case_dir: str):
        """
        Set up the OpenFOAM case for this spill model.
        
        Args:
            case_dir (str): Directory for the OpenFOAM case
            
        Returns:
            CaseManager: The configured case manager
        """
        # Create case manager
        self.case_manager = CaseManager(case_dir)
        
        # Common setup steps
        self.case_manager.create_case()
        
        # Set up model-specific configuration
        self._setup_case_specific()
        
        return self.case_manager
    
    def _setup_case_specific(self):
        """
        Set up model-specific configuration.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _setup_case_specific")
    
    def run_simulation(self, parallel: bool = False, processors: int = 4):
        """
        Run the simulation.
        
        Args:
            parallel (bool): Whether to run in parallel
            processors (int): Number of processors for parallel run
            
        Returns:
            bool: True if simulation completed successfully, False otherwise
        """
        if not self.case_manager:
            raise ValueError("Case must be set up before running simulation")
        
        # Create solver manager
        solver_manager = SolverManager(self.case_manager)
        
        # Run the solver
        success = solver_manager.run_solver(parallel=parallel, processors=processors)
        
        # Process results if successful
        if success:
            self._process_results()
        
        return success
    
    def _process_results(self):
        """
        Process simulation results.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _process_results")
    
    def calculate_area(self, time: float) -> float:
        """
        Calculate spill area at a given time.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            float: Spill area in m²
        """
        raise NotImplementedError("Subclasses must implement calculate_area")
    
    def calculate_volume(self, time: float) -> float:
        """
        Calculate spill volume at a given time.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            float: Spill volume in m³
        """
        if time <= 0:
            return 0.0
        
        # Basic calculation (to be overridden by subclasses)
        actual_time = min(time, self.duration)
        return (self.spill_rate * actual_time) / self.fluid.density
    
    def calculate_thickness(self, time: float) -> float:
        """
        Calculate spill thickness at a given time.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            float: Spill thickness in m
        """
        area = self.calculate_area(time)
        if area <= 0:
            return 0.0
        
        volume = self.calculate_volume(time)
        return volume / area
    
    def calculate_evaporation(self, time: float) -> float:
        """
        Calculate evaporated mass at a given time.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            float: Evaporated mass in kg
        """
        # Basic implementation - to be overridden by subclasses
        return 0.0
    
    def calculate_dispersion(self, time: float) -> float:
        """
        Calculate dispersed mass at a given time.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            float: Dispersed mass in kg
        """
        # Basic implementation - to be overridden by subclasses
        return 0.0
    
    def calculate_dissolution(self, time: float) -> float:
        """
        Calculate dissolved mass at a given time.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            float: Dissolved mass in kg
        """
        # Basic implementation - to be overridden by subclasses
        return 0.0
    
    def calculate_remaining_mass(self, time: float) -> float:
        """
        Calculate remaining mass at a given time.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            float: Remaining mass in kg
        """
        if time <= 0:
            return 0.0
        
        # Calculate released mass
        actual_time = min(time, self.duration)
        released_mass = self.spill_rate * actual_time
        
        # Calculate losses
        evaporated_mass = self.calculate_evaporation(time)
        dispersed_mass = self.calculate_dispersion(time)
        dissolved_mass = self.calculate_dissolution(time)
        
        # Calculate remaining mass
        remaining_mass = released_mass - evaporated_mass - dispersed_mass - dissolved_mass
        
        # Ensure non-negative
        return max(0.0, remaining_mass)
    
    def get_summary(self, time: float) -> Dict[str, Any]:
        """
        Get a summary of the spill at a given time.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            Dict[str, Any]: Spill summary
        """
        # Calculate various quantities
        released_mass = min(time, self.duration) * self.spill_rate
        volume = self.calculate_volume(time)
        area = self.calculate_area(time)
        thickness = self.calculate_thickness(time)
        evaporated_mass = self.calculate_evaporation(time)
        dispersed_mass = self.calculate_dispersion(time)
        dissolved_mass = self.calculate_dissolution(time)
        remaining_mass = self.calculate_remaining_mass(time)
        
        # Calculate percentages
        if released_mass > 0:
            evaporated_pct = (evaporated_mass / released_mass) * 100
            dispersed_pct = (dispersed_mass / released_mass) * 100
            dissolved_pct = (dissolved_mass / released_mass) * 100
            remaining_pct = (remaining_mass / released_mass) * 100
        else:
            evaporated_pct = dispersed_pct = dissolved_pct = remaining_pct = 0.0
        
        # Return summary
        return {
            "time": time,
            "released_mass": released_mass,
            "released_volume": volume,
            "area": area,
            "thickness": thickness,
            "evaporated_mass": evaporated_mass,
            "evaporated_percentage": evaporated_pct,
            "dispersed_mass": dispersed_mass,
            "dispersed_percentage": dispersed_pct,
            "dissolved_mass": dissolved_mass,
            "dissolved_percentage": dissolved_pct,
            "remaining_mass": remaining_mass,
            "remaining_percentage": remaining_pct
        }
    
    def export_results(self, output_file: str):
        """
        Export results to a file.
        
        Args:
            output_file (str): Output file path
        """
        # Ensure we have results
        if not self.results:
            logger.warning("No results to export")
            return
        
        # Export to JSON
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            logger.info(f"Results exported to {output_file}")
        
        except Exception as e:
            logger.error(f"Error exporting results: {e}")


class SurfaceSpillModel(SpillModel):
    """
    Model for surface spills on water or land.
    
    This model uses the semi-empirical approach for surface spreading
    and weathering of oil spills on water or land.
    """
    
    def __init__(self, fluid: str, rate: float, total_mass: float, 
                 opening_diameter: float, temperature: float, pressure: float,
                 environment: EnvironmentProperties):
        """
        Initialize the surface spill model.
        
        Args:
            fluid (str): Type of fluid
            rate (float): Spill rate in kg/s
            total_mass (float): Total mass in kg
            opening_diameter (float): Opening diameter in m
            temperature (float): Fluid temperature in K
            pressure (float): Fluid pressure in Pa
            environment (EnvironmentProperties): Environmental properties
        """
        super().__init__("Surface", fluid, rate, total_mass, 
                         opening_diameter, temperature, pressure, environment)
        
        # Initialize spreading coefficients (Fay's equations)
        self.k1 = 1.14  # Gravity-inertial spreading coefficient
        self.k2 = 1.45  # Gravity-viscous spreading coefficient
        
        # Evaporation parameters
        self.evaporation_coefficient = 0.0025  # m/s, mass transfer coefficient
        
        # Dispersion parameters
        self.dispersion_coefficient = 0.0  # To be set based on environment
        
        # Set dispersion coefficient based on environment
        if self.environment.is_water:
            # Delvigne & Sweeney dispersion model
            # C_disp = 0.11 * D_a * wind_speed^2
            # where D_a is a sea state dependent constant
            sea_state_factor = min(1.0, 0.1 * self.environment.wave_height)
            self.dispersion_coefficient = 0.11 * sea_state_factor * (self.environment.wind_speed ** 2)
    
    def _setup_case_specific(self):
        """Set up surface spill specific configuration."""
        if not self.case_manager:
            raise ValueError("Case manager not initialized")
        
        # Set solver based on environment
        if self.environment.is_water:
            solver = "multiphaseInterFoam"
        else:
            solver = "interFoam"
        
        # Set up basic case structure
        self.case_manager.set_solver(solver)
        
        # Set up transport properties
        transport_model = TransportModel(self.case_manager)
        
        # Add phases
        if self.environment.is_water:
            # Water, oil, and air
            transport_model.add_phase("water", self.environment.water_density, self.environment.water_viscosity)
            transport_model.add_phase(self.fluid_name.lower(), self.fluid.density, self.fluid.viscosity)
            transport_model.add_phase("air", 1.2, 1.8e-5)
            
            # Set surface tensions
            transport_model.set_surface_tension("water", self.fluid_name.lower(), self.fluid.surface_tension)
            transport_model.set_surface_tension("water", "air", 0.072)
            transport_model.set_surface_tension(self.fluid_name.lower(), "air", 0.025)
        else:
            # Oil and air
            transport_model.add_phase(self.fluid_name.lower(), self.fluid.density, self.fluid.viscosity)
            transport_model.add_phase("air", 1.2, 1.8e-5)
            
            # Set surface tension
            transport_model.set_surface_tension(self.fluid_name.lower(), "air", self.fluid.surface_tension)
        
        # Write transport properties
        transport_model.write()
        
        # Set up domain and mesh
        # (In a full implementation, this would be more complex with appropriate domain size)
        if self.environment.is_water:
            # Water surface domain
            self.case_manager.create_box_mesh(
                x_min=-50.0, x_max=50.0, 
                y_min=-50.0, y_max=50.0, 
                z_min=-5.0, z_max=5.0,
                x_cells=50, y_cells=50, z_cells=20
            )
        else:
            # Land surface domain
            self.case_manager.create_box_mesh(
                x_min=-20.0, x_max=20.0, 
                y_min=-20.0, y_max=20.0, 
                z_min=0.0, z_max=2.0,
                x_cells=40, y_cells=40, z_cells=10
            )
        
        # Set up boundary conditions
        if self.environment.is_water:
            # Initialize water/air interface at z=0
            self.case_manager.set_field_initialization("alpha.water", f"pos().z <= 0 ? 1 : 0")
            
            # Initialize source region for oil
            source_condition = (
                f"pos().z > 0 && pos().z < 0.2 && "
                f"mag(vector(pos().x, pos().y, 0)) < {self.opening_diameter/2} ? 1 : 0"
            )
            self.case_manager.set_field_initialization(f"alpha.{self.fluid_name.lower()}", source_condition)
            
            # Set up inlet patch for continuous source
            inlet_patch = BoundaryCondition("inlet", "patch")
            inlet_patch.set_location(f"pos().z > 0 && pos().z < 0.2 && mag(vector(pos().x, pos().y, 0)) < {self.opening_diameter/2}")
            
            # Set velocity at inlet
            inlet_velocity = self.spill_rate / (self.fluid.density * math.pi * (self.opening_diameter/2)**2)
            inlet_patch.set_velocity([0, 0, -inlet_velocity])
            
            # Set alpha fields at inlet
            inlet_patch.set_field(f"alpha.{self.fluid_name.lower()}", 1.0)
            inlet_patch.set_field("alpha.water", 0.0)
            inlet_patch.set_field("alpha.air", 0.0)
            
            # Add patch to case
            self.case_manager.add_boundary_condition(inlet_patch)
            
            # Set up wind and current as velocity boundary conditions
            if self.environment.wind_speed > 0:
                top_patch = BoundaryCondition("atmosphere", "patch")
                top_patch.set_location("pos().z >= 5.0")
                top_patch.set_velocity([self.environment.wind_speed, 0, 0])
                self.case_manager.add_boundary_condition(top_patch)
            
            if self.environment.current_speed > 0:
                side_patch = BoundaryCondition("inlet_current", "patch")
                side_patch.set_location("pos().x <= -50.0 && pos().z <= 0")
                side_patch.set_velocity([self.environment.current_speed, 0, 0])
                self.case_manager.add_boundary_condition(side_patch)
        else:
            # Land spill setup
            # Initialize source region for oil
            source_condition = (
                f"pos().z > 0 && pos().z < 0.1 && "
                f"mag(vector(pos().x, pos().y, 0)) < {self.opening_diameter/2} ? 1 : 0"
            )
            self.case_manager.set_field_initialization(f"alpha.{self.fluid_name.lower()}", source_condition)
            
            # Set up inlet patch for continuous source
            inlet_patch = BoundaryCondition("inlet", "patch")
            inlet_patch.set_location(f"pos().z > 0 && pos().z < 0.1 && mag(vector(pos().x, pos().y, 0)) < {self.opening_diameter/2}")
            
            # Set velocity at inlet
            inlet_velocity = self.spill_rate / (self.fluid.density * math.pi * (self.opening_diameter/2)**2)
            inlet_patch.set_velocity([0, 0, 0])  # Initial velocity zero for land spill
            inlet_patch.set_pressure(self.pressure)  # Set pressure to drive flow
            
            # Set alpha fields at inlet
            inlet_patch.set_field(f"alpha.{self.fluid_name.lower()}", 1.0)
            inlet_patch.set_field("alpha.air", 0.0)
            
            # Add patch to case
            self.case_manager.add_boundary_condition(inlet_patch)
        
        # Write case files
        self.case_manager.write_case()
    
    def calculate_area(self, time: float) -> float:
        """
        Calculate spill area using Fay's spreading equations.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            float: Spill area in m²
        """
        if time <= 0:
            return 0.0
        
        # Calculate volume at current time
        volume = self.calculate_volume(time)
        if volume <= 0:
            return 0.0
        
        if self.environment.is_water:
            # Water surface spill
            # Use Fay's gravity-inertial spreading for early phase
            # A = k1 * (g * Δρ/ρ * V²)^(1/3) * t^(2/3)
            
            # Then transition to gravity-viscous spreading
            # A = k2 * (g * Δρ/ρ * V³)^(1/4) * (1/ν)^(1/2) * t^(1/2)
            
            # Calculate density difference ratio
            density_diff_ratio = abs(self.fluid.density - self.environment.water_density) / self.environment.water_density
            
            # Early phase (gravity-inertial)
            if time < 60:  # First minute
                area_gi = math.pi * (self.k1 * (9.81 * density_diff_ratio * volume**2)**(1/3) * time**(2/3))**2
                
                # Limit to minimum area based on initial release
                min_area = math.pi * (self.opening_diameter / 2)**2
                return max(min_area, area_gi)
            else:
                # Later phase (gravity-viscous)
                area_gv = math.pi * (self.k2 * (9.81 * density_diff_ratio * volume**3)**(1/4) * 
                                    (1/self.environment.water_viscosity)**(1/2) * time**(1/2))**2
                
                # Limit to minimum area from initial phase
                min_area = math.pi * (self.k1 * (9.81 * density_diff_ratio * volume**2)**(1/3) * 60**(2/3))**2
                return max(min_area, area_gv)
        else:
            # Land surface spill
            # Spreading is much more limited on land
            # Use a simple pool spreading model with permeability
            
            # For impermeable surface (like concrete)
            if self.environment.permeability < 1e-12:
                # Simple gravity-driven spreading with friction
                # Approximated as A = k * V^(2/3) * t^(1/3)
                k_land = 0.5  # spreading coefficient for land (much slower than water)
                
                area = math.pi * (k_land * volume**(2/3) * time**(1/3))**2
                
                # Limit to minimum area based on initial release
                min_area = math.pi * (self.opening_diameter / 2)**2
                return max(min_area, area)
            else:
                # For permeable surface (like soil)
                # Model both horizontal spreading and vertical infiltration
                # Approximated as A = k * V^(2/3) * t^(1/4) * (1/permeability)^(1/4)
                k_perm = 0.3  # spreading coefficient for permeable land
                
                area = math.pi * (k_perm * volume**(2/3) * time**(1/4) * 
                                (1/self.environment.permeability)**(1/4))**2
                
                # Limit to minimum area based on initial release
                min_area = math.pi * (self.opening_diameter / 2)**2
                return max(min_area, area)
    
    def calculate_evaporation(self, time: float) -> float:
        """
        Calculate evaporated mass using pseudo-component approach.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            float: Evaporated mass in kg
        """
        if time <= 0:
            return 0.0
        
        # Calculate volume at current time
        volume = self.calculate_volume(time)
        if volume <= 0:
            return 0.0
        
        # Calculate area at current time
        area = self.calculate_area(time)
        
        # Calculate evaporation based on fluid properties
        # Use a simplified model based on vapor pressure, temperature, and wind speed
        # Mass evaporated = K_evap * A * (P_v/RT) * t
        # where K_evap is the mass transfer coefficient, A is area,
        # P_v is vapor pressure, R is gas constant, T is temperature
        
        # Adjust evaporation coefficient based on wind speed
        # Wind increases evaporation rate
        wind_factor = 1.0 + 0.1 * self.environment.wind_speed
        
        # Adjust for temperature
        # Higher temperature increases evaporation rate
        temp_factor = math.exp(0.03 * (self.temperature - 293.15))
        
        # Gas constant
        R = 8.314
        
        # Calculate evaporation rate (kg/s)
        evap_rate = (self.evaporation_coefficient * wind_factor * temp_factor * 
                    area * (self.fluid.vapor_pressure / (R * self.temperature)))
        
        # Calculate total evaporated mass
        total_evaporated = evap_rate * time
        
        # Limit evaporation to available mass
        released_mass = min(time, self.duration) * self.spill_rate
        return min(total_evaporated, 0.3 * released_mass)  # Limit to 30% of released mass
    
    def calculate_dispersion(self, time: float) -> float:
        """
        Calculate dispersed mass for water surface spills.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            float: Dispersed mass in kg
        """
        if time <= 0 or not self.environment.is_water:
            return 0.0
        
        # Calculate volume at current time
        volume = self.calculate_volume(time)
        if volume <= 0:
            return 0.0
        
        # Calculate area at current time
        area = self.calculate_area(time)
        
        # Calculate thickness
        thickness = volume / area
        
        # Dispersion only occurs with sufficient wave action
        if self.environment.wave_height < 0.1:
            return 0.0
        
        # Simplified Delvigne and Sweeney model
        # Fraction dispersed per hour = C * (wind_speed^2) * (wave_height^0.5) / thickness
        # where C is a dispersion coefficient
        
        # Adjust coefficient based on oil properties
        # Lighter oils disperse more easily
        if self.fluid.density < 850:  # Light crude or products
            dispersion_factor = 0.0005
        else:  # Heavy crude
            dispersion_factor = 0.0002
        
        # Calculate fraction dispersed per second
        fraction_per_second = (dispersion_factor * 
                              (self.environment.wind_speed**2) * 
                              (self.environment.wave_height**0.5) / 
                              max(0.001, thickness)) / 3600  # Convert from per hour to per second
        
        # Calculate total dispersed mass
        total_dispersed = min(time, self.duration) * self.spill_rate * fraction_per_second * time
        
        # Limit dispersion to available mass
        released_mass = min(time, self.duration) * self.spill_rate
        evaporated_mass = self.calculate_evaporation(time)
        available_mass = released_mass - evaporated_mass
        
        return min(total_dispersed, available_mass)
    
    def calculate_dissolution(self, time: float) -> float:
        """
        Calculate dissolved mass for water surface spills.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            float: Dissolved mass in kg
        """
        if time <= 0 or not self.environment.is_water:
            return 0.0
        
        # Calculate volume at current time
        volume = self.calculate_volume(time)
        if volume <= 0:
            return 0.0
        
        # Calculate area at current time
        area = self.calculate_area(time)
        
        # Dissolution is a minor weathering process for most oils
        # Use a simple first-order model
        # Rate = K_dissolution * A * water_solubility
        # where K_dissolution is the mass transfer coefficient
        
        # Estimate water solubility based on oil type (in kg/m³)
        if self.fluid_name == "Crude Oil":
            solubility = 0.005
        elif self.fluid_name == "Diesel":
            solubility = 0.002
        elif self.fluid_name == "Gasoline":
            solubility = 0.05
        else:
            solubility = 0.001
        
        # Mass transfer coefficient (m/s)
        k_dissolution = 1e-6
        
        # Calculate dissolution rate (kg/s)
        dissolution_rate = k_dissolution * area * solubility
        
        # Calculate total dissolved mass
        total_dissolved = dissolution_rate * time
        
        # Limit dissolution to available mass
        released_mass = min(time, self.duration) * self.spill_rate
        evaporated_mass = self.calculate_evaporation(time)
        dispersed_mass = self.calculate_dispersion(time)
        available_mass = released_mass - evaporated_mass - dispersed_mass
        
        return min(total_dissolved, 0.01 * available_mass)  # Limit to 1% of available mass
    
    def _process_results(self):
        """Process OpenFOAM simulation results for surface spill."""
        if not self.case_manager:
            logger.warning("No case manager to process results")
            return
        
        # Get list of time directories
        time_dirs = self.case_manager.get_time_directories()
        
        # Process each time directory
        results = {
            "times": [],
            "areas": [],
            "volumes": [],
            "thicknesses": [],
            "velocities": [],
            "positions": []
        }
        
        for time_dir in time_dirs:
            time = float(os.path.basename(time_dir))
            
            # Read alpha field to determine spill location and extent
            alpha_field = self.case_manager.read_field(f"alpha.{self.fluid_name.lower()}", time_dir)
            
            if alpha_field is None:
                continue
            
            # Calculate spill extent, volume, etc.
            # This would be complex in a real implementation, simplifying here
            area = self.calculate_area(time)
            volume = self.calculate_volume(time)
            thickness = self.calculate_thickness(time)
            
            # Store results
            results["times"].append(time)
            results["areas"].append(area)
            results["volumes"].append(volume)
            results["thicknesses"].append(thickness)
            
            # Additional results from simulation
            # These would be extracted from the actual OpenFOAM results
            results["velocities"].append([0, 0, 0])  # Placeholder
            results["positions"].append([0, 0, 0])  # Placeholder
        
        # Store in results dictionary
        self.results["simulation"] = results
        
        # Add weathering results
        self.results["weathering"] = {
            "times": results["times"],
            "evaporation": [self.calculate_evaporation(t) for t in results["times"]],
            "dispersion": [self.calculate_dispersion(t) for t in results["times"]],
            "dissolution": [self.calculate_dissolution(t) for t in results["times"]],
            "remaining": [self.calculate_remaining_mass(t) for t in results["times"]]
        }


class SubsurfaceSpillModel(SpillModel):
    """
    Model for subsurface (underwater) spills.
    
    This model accounts for underwater release dynamics including:
    - Buoyant plume behavior
    - Underwater spreading
    - Dissolution and droplet formation
    """
    
    def __init__(self, fluid: str, rate: float, total_mass: float, 
                 opening_diameter: float, temperature: float, pressure: float,
                 environment: EnvironmentProperties, depth: float):
        """
        Initialize the subsurface spill model.
        
        Args:
            fluid (str): Type of fluid
            rate (float): Spill rate in kg/s
            total_mass (float): Total mass in kg
            opening_diameter (float): Opening diameter in m
            temperature (float): Fluid temperature in K
            pressure (float): Fluid pressure in Pa
            environment (EnvironmentProperties): Environmental properties
            depth (float): Release depth in m
        """
        super().__init__("Subsurface", fluid, rate, total_mass, 
                         opening_diameter, temperature, pressure, environment)
        
        # Underwater release parameters
        self.depth = depth
        
        # Plume model parameters
        self.entrainment_coefficient = 0.1  # Typical value for buoyant plumes
        
        # Terminal velocity calculation
        # For oil droplets, use modified Stokes law
        # v_t = (2/9) * g * R² * (ρ_w - ρ_o) / μ_w
        # where R is droplet radius, ρ_w is water density,
        # ρ_o is oil density, μ_w is water viscosity
        droplet_radius = self.opening_diameter / 4  # Assume initial droplet size
        self.terminal_velocity = ((2/9) * 9.81 * droplet_radius**2 * 
                                 (self.environment.water_density - self.fluid.density) / 
                                 self.environment.water_viscosity)
        
        # Calculate rise time to surface
        self.rise_time = self.depth / abs(self.terminal_velocity) if self.terminal_velocity != 0 else float('inf')
        
        # Dissolution model parameters
        # Mass transfer coefficient depends on relative velocity
        relative_velocity = max(0.1, abs(self.terminal_velocity))
        self.dissolution_coefficient = 1e-6 * (1 + 0.1 * relative_velocity)
    
    def _setup_case_specific(self):
        """Set up subsurface spill specific configuration."""
        if not self.case_manager:
            raise ValueError("Case manager not initialized")
        
        # Set solver for underwater release
        solver = "multiphaseInterFoam"
        
        # Set up basic case structure
        self.case_manager.set_solver(solver)
        
        # Set up transport properties
        transport_model = TransportModel(self.case_manager)
        
        # Add phases
        transport_model.add_phase("water", self.environment.water_density, self.environment.water_viscosity)
        transport_model.add_phase(self.fluid_name.lower(), self.fluid.density, self.fluid.viscosity)
        transport_model.add_phase("air", 1.2, 1.8e-5)
        
        # Set surface tensions
        transport_model.set_surface_tension("water", self.fluid_name.lower(), self.fluid.surface_tension)
        transport_model.set_surface_tension("water", "air", 0.072)
        transport_model.set_surface_tension(self.fluid_name.lower(), "air", 0.025)
        
        # Write transport properties
        transport_model.write()
        
        # Set up domain and mesh
        # Create domain with underwater release point and water surface
        self.case_manager.create_box_mesh(
            x_min=-50.0, x_max=50.0, 
            y_min=-50.0, y_max=50.0, 
            z_min=-self.depth-10.0, z_max=10.0,
            x_cells=50, y_cells=50, z_cells=int(60 * (self.depth + 20) / 100)
        )
        
        # Initialize water/air interface at z=0
        self.case_manager.set_field_initialization("alpha.water", f"pos().z <= 0 ? 1 : 0")
        
        # Initialize source region for oil
        source_condition = (
            f"pos().z < {-self.depth + 0.5} && pos().z > {-self.depth - 0.5} && "
            f"mag(vector(pos().x, pos().y, 0)) < {self.opening_diameter/2} ? 1 : 0"
        )
        self.case_manager.set_field_initialization(f"alpha.{self.fluid_name.lower()}", source_condition)
        
        # Set up inlet patch for continuous source
        inlet_patch = BoundaryCondition("inlet", "patch")
        inlet_patch.set_location(
            f"pos().z < {-self.depth + 0.5} && pos().z > {-self.depth - 0.5} && "
            f"mag(vector(pos().x, pos().y, 0)) < {self.opening_diameter/2}"
        )
        
        # Set velocity at inlet
        inlet_velocity = self.spill_rate / (self.fluid.density * math.pi * (self.opening_diameter/2)**2)
        inlet_patch.set_velocity([0, 0, 0])  # Initial velocity zero, pressure-driven
        inlet_patch.set_pressure(self.pressure)  # Set pressure to drive flow
        
        # Set alpha fields at inlet
        inlet_patch.set_field(f"alpha.{self.fluid_name.lower()}", 1.0)
        inlet_patch.set_field("alpha.water", 0.0)
        inlet_patch.set_field("alpha.air", 0.0)
        
        # Add patch to case
        self.case_manager.add_boundary_condition(inlet_patch)
        
        # Set up water current as velocity boundary conditions
        if self.environment.current_speed > 0:
            side_patch = BoundaryCondition("inlet_current", "patch")
            side_patch.set_location("pos().x <= -50.0 && pos().z <= 0")
            side_patch.set_velocity([self.environment.current_speed, 0, 0])
            self.case_manager.add_boundary_condition(side_patch)
        
        # Set up wind for atmosphere
        if self.environment.wind_speed > 0:
            top_patch = BoundaryCondition("atmosphere", "patch")
            top_patch.set_location("pos().z >= 10.0")
            top_patch.set_velocity([self.environment.wind_speed, 0, 0])
            self.case_manager.add_boundary_condition(top_patch)
        
        # Write case files
        self.case_manager.write_case()
    
    def calculate_area(self, time: float) -> float:
        """
        Calculate area of the surface slick.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            float: Surface area in m²
        """
        if time <= 0:
            return 0.0
        
        # No surface slick until oil reaches the surface
        if time < self.rise_time:
            return 0.0
        
        # Time since first oil reached the surface
        surface_time = time - self.rise_time
        
        # Calculate volume that has reached the surface
        if surface_time <= 0:
            surface_volume = 0.0
        elif surface_time < self.duration:
            surface_volume = (self.spill_rate * surface_time) / self.fluid.density
        else:
            surface_volume = (self.spill_rate * self.duration) / self.fluid.density
        
        if surface_volume <= 0:
            return 0.0
        
        # Once at the surface, use Fay's spreading equations (from SurfaceSpillModel)
        # But with reduced coefficients due to droplet dynamics
        k1 = 0.5  # Reduced gravity-inertial spreading coefficient
        k2 = 0.7  # Reduced gravity-viscous spreading coefficient
        
        # Calculate density difference ratio
        density_diff_ratio = abs(self.fluid.density - self.environment.water_density) / self.environment.water_density
        
        # Early phase (gravity-inertial)
        if surface_time < 60:  # First minute
            area_gi = math.pi * (k1 * (9.81 * density_diff_ratio * surface_volume**2)**(1/3) * surface_time**(2/3))**2
            
            # Limit to minimum area based on initial release
            min_area = math.pi * (self.opening_diameter / 2)**2
            return max(min_area, area_gi)
        else:
            # Later phase (gravity-viscous)
            area_gv = math.pi * (k2 * (9.81 * density_diff_ratio * surface_volume**3)**(1/4) * 
                                (1/self.environment.water_viscosity)**(1/2) * surface_time**(1/2))**2
            
            # Limit to minimum area from initial phase
            min_area = math.pi * (k1 * (9.81 * density_diff_ratio * surface_volume**2)**(1/3) * 60**(2/3))**2
            return max(min_area, area_gv)
    
    def calculate_plume_radius(self, z: float) -> float:
        """
        Calculate plume radius at a given depth.
        
        Args:
            z (float): Depth coordinate (negative below surface)
            
        Returns:
            float: Plume radius in m
        """
        # Simple plume growth model
        # r = r₀ + α * |z - z₀|
        # where r₀ is initial radius, α is entrainment coefficient,
        # z is current depth, z₀ is release depth
        
        if z < -self.depth:
            # Below release point
            return self.opening_diameter / 2
        elif z > 0:
            # Above water surface
            return 0.0
        else:
            # Between release point and surface
            return self.opening_diameter / 2 + self.entrainment_coefficient * abs(z + self.depth)
    
    def calculate_plume_velocity(self, z: float) -> float:
        """
        Calculate plume velocity at a given depth.
        
        Args:
            z (float): Depth coordinate (negative below surface)
            
        Returns:
            float: Plume velocity in m/s (positive upward)
        """
        if z < -self.depth:
            # Below release point
            return 0.0
        elif z > 0:
            # Above water surface
            return 0.0
        
        # Calculate initial velocity
        initial_velocity = self.spill_rate / (self.fluid.density * math.pi * (self.opening_diameter/2)**2)
        
        # Calculate buoyancy-driven velocity
        # For plume, velocity decreases with distance due to entrainment
        # v = v₀ * (r₀/r)² + v_buoyancy
        
        r0 = self.opening_diameter / 2
        r = self.calculate_plume_radius(z)
        
        # Buoyancy-driven velocity
        # v_buoyancy = √(2 * g * Δρ/ρ * h)
        # where h is distance from release point
        density_diff_ratio = abs(self.fluid.density - self.environment.water_density) / self.environment.water_density
        buoyancy_velocity = math.sqrt(2 * 9.81 * density_diff_ratio * abs(z + self.depth))
        
        # Combine initial and buoyancy velocities
        if r > 0:
            velocity = initial_velocity * (r0/r)**2 + buoyancy_velocity
        else:
            velocity = 0.0
        
        # Adjust sign based on buoyancy
        if self.fluid.density < self.environment.water_density:
            return velocity  # Oil rises
        else:
            return -velocity  # Oil sinks
    
    def calculate_dissolution(self, time: float) -> float:
        """
        Calculate dissolved mass for subsurface spill.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            float: Dissolved mass in kg
        """
        if time <= 0:
            return 0.0
        
        # For underwater release, dissolution is more significant
        # Calculate mass released so far
        released_mass = min(time, self.duration) * self.spill_rate
        
        # Estimate total surface area of droplets
        # Assume droplet size distribution with mean radius = opening_diameter/4
        mean_droplet_radius = self.opening_diameter / 4
        droplet_volume = (4/3) * math.pi * mean_droplet_radius**3
        num_droplets = (released_mass / self.fluid.density) / droplet_volume
        total_droplet_area = num_droplets * 4 * math.pi * mean_droplet_radius**2
        
        # Estimate water-soluble fraction of oil
        # This varies by oil type
        if self.fluid_name == "Crude Oil":
            soluble_fraction = 0.01  # 1%
        elif self.fluid_name == "Diesel":
            soluble_fraction = 0.005  # 0.5%
        elif self.fluid_name == "Gasoline":
            soluble_fraction = 0.03  # 3%
        else:
            soluble_fraction = 0.01  # Default 1%
        
        # Calculate dissolution based on first-order mass transfer
        # dm/dt = k * A * C_sat
        # where k is mass transfer coefficient, A is area,
        # C_sat is saturation concentration
        
        # Estimated saturation concentration (kg/m³)
        if self.fluid_name == "Crude Oil":
            saturation_conc = 0.005
        elif self.fluid_name == "Diesel":
            saturation_conc = 0.002
        elif self.fluid_name == "Gasoline":
            saturation_conc = 0.05
        else:
            saturation_conc = 0.001
        
        # Calculate max potential dissolution
        max_dissolution = released_mass * soluble_fraction
        
        # Calculate actual dissolution based on time, area, and coefficient
        dissolution_rate = self.dissolution_coefficient * total_droplet_area * saturation_conc
        actual_dissolution = dissolution_rate * min(time, self.rise_time)
        
        # Return the lesser of the two values
        return min(actual_dissolution, max_dissolution)
    
    def calculate_evaporation(self, time: float) -> float:
        """
        Calculate evaporated mass for subsurface spill.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            float: Evaporated mass in kg
        """
        if time <= 0:
            return 0.0
        
        # No evaporation until oil reaches the surface
        if time < self.rise_time:
            return 0.0
        
        # Time since first oil reached the surface
        surface_time = time - self.rise_time
        
        # Calculate volume that has reached the surface
        if surface_time <= 0:
            surface_volume = 0.0
        elif surface_time < self.duration:
            surface_volume = (self.spill_rate * surface_time) / self.fluid.density
        else:
            surface_volume = (self.spill_rate * self.duration) / self.fluid.density
        
        if surface_volume <= 0:
            return 0.0
        
        # Calculate area at current time
        area = self.calculate_area(time)
        
        # Calculate evaporation based on fluid properties
        # Use a simplified model based on vapor pressure, temperature, and wind speed
        # Mass evaporated = K_evap * A * (P_v/RT) * t
        
        # Adjust evaporation coefficient based on wind speed
        # Wind increases evaporation rate
        evaporation_coefficient = 0.0025  # Base coefficient (m/s)
        wind_factor = 1.0 + 0.1 * self.environment.wind_speed
        
        # Adjust for temperature
        # Higher temperature increases evaporation rate
        temp_factor = math.exp(0.03 * (self.temperature - 293.15))
        
        # Gas constant
        R = 8.314
        
        # Calculate evaporation rate (kg/s)
        evap_rate = (evaporation_coefficient * wind_factor * temp_factor * 
                    area * (self.fluid.vapor_pressure / (R * self.temperature)))
        
        # Calculate total evaporated mass
        total_evaporated = evap_rate * surface_time
        
        # Limit evaporation to available mass at surface
        released_mass = min(time, self.duration) * self.spill_rate
        dissolution = self.calculate_dissolution(time)
        available_mass = released_mass - dissolution
        
        return min(total_evaporated, 0.3 * available_mass)  # Limit to 30% of available mass
    
    def calculate_volume(self, time: float) -> float:
        """
        Calculate spill volume at the surface.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            float: Spill volume in m³
        """
        if time <= 0:
            return 0.0
        
        # No surface volume until oil reaches the surface
        if time < self.rise_time:
            return 0.0
        
        # Time since first oil reached the surface
        surface_time = time - self.rise_time
        
        # Calculate volume that has reached the surface
        if surface_time <= 0:
            surface_volume = 0.0
        elif surface_time < self.duration:
            surface_volume = (self.spill_rate * surface_time) / self.fluid.density
        else:
            surface_volume = (self.spill_rate * self.duration) / self.fluid.density
        
        # Adjust for losses (dissolution occurs during rise and evaporation after surfacing)
        dissolution = self.calculate_dissolution(time)
        evaporation = self.calculate_evaporation(time)
        
        # Calculate net volume
        net_volume = surface_volume - (dissolution + evaporation) / self.fluid.density
        
        # Ensure non-negative
        return max(0.0, net_volume)
    
    def calculate_thickness(self, time: float) -> float:
        """
        Calculate spill thickness at the surface.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            float: Spill thickness in m
        """
        if time <= self.rise_time:
            return 0.0
        
        area = self.calculate_area(time)
        if area <= 0:
            return 0.0
        
        volume = self.calculate_volume(time)
        return volume / area
    
    def calculate_dispersion(self, time: float) -> float:
        """
        Calculate dispersed mass from the surface.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            float: Dispersed mass in kg
        """
        if time <= self.rise_time:
            return 0.0
        
        # Time since first oil reached the surface
        surface_time = time - self.rise_time
        
        # Calculate thickness
        thickness = self.calculate_thickness(time)
        if thickness <= 0:
            return 0.0
        
        # Dispersion only occurs with sufficient wave action
        if self.environment.wave_height < 0.1:
            return 0.0
        
        # Simplified Delvigne and Sweeney model
        # Fraction dispersed per hour = C * (wind_speed^2) * (wave_height^0.5) / thickness
        
        # Adjust coefficient based on oil properties
        if self.fluid.density < 850:  # Light crude or products
            dispersion_factor = 0.0005
        else:  # Heavy crude
            dispersion_factor = 0.0002
        
        # Calculate fraction dispersed per second
        fraction_per_second = (dispersion_factor * 
                              (self.environment.wind_speed**2) * 
                              (self.environment.wave_height**0.5) / 
                              max(0.001, thickness)) / 3600  # Convert from per hour to per second
        
        # Calculate surface mass
        surface_mass = self.fluid.density * self.calculate_volume(time)
        
        # Calculate total dispersed mass
        total_dispersed = surface_mass * fraction_per_second * surface_time
        
        # Limit dispersion to available mass
        released_mass = min(time, self.duration) * self.spill_rate
        evaporated_mass = self.calculate_evaporation(time)
        dissolved_mass = self.calculate_dissolution(time)
        available_mass = released_mass - evaporated_mass - dissolved_mass
        
        return min(total_dispersed, available_mass)
    
    def _process_results(self):
        """Process OpenFOAM simulation results for subsurface spill."""
        if not self.case_manager:
            logger.warning("No case manager to process results")
            return
        
        # Get list of time directories
        time_dirs = self.case_manager.get_time_directories()
        
        # Process each time directory
        results = {
            "times": [],
            "surface_areas": [],
            "surface_volumes": [],
            "surface_thicknesses": [],
            "plume_data": []
        }
        
        for time_dir in time_dirs:
            time = float(os.path.basename(time_dir))
            
            # Read alpha field to determine oil distribution
            alpha_field = self.case_manager.read_field(f"alpha.{self.fluid_name.lower()}", time_dir)
            
            if alpha_field is None:
                continue
            
            # Calculate surface metrics
            surface_area = self.calculate_area(time)
            surface_volume = self.calculate_volume(time)
            surface_thickness = self.calculate_thickness(time)
            
            # Store results
            results["times"].append(time)
            results["surface_areas"].append(surface_area)
            results["surface_volumes"].append(surface_volume)
            results["surface_thicknesses"].append(surface_thickness)
            
            # Plume analysis would extract data from the simulation
            # For now, use our analytical models
            plume_data = {
                "centerline_z": np.linspace(-self.depth, 0, 20),
                "radius": [self.calculate_plume_radius(z) for z in np.linspace(-self.depth, 0, 20)],
                "velocity": [self.calculate_plume_velocity(z) for z in np.linspace(-self.depth, 0, 20)]
            }
            
            results["plume_data"].append(plume_data)
        
        # Store in results dictionary
        self.results["simulation"] = results
        
        # Add weathering results
        self.results["weathering"] = {
            "times": results["times"],
            "dissolution": [self.calculate_dissolution(t) for t in results["times"]],
            "evaporation": [self.calculate_evaporation(t) for t in results["times"]],
            "dispersion": [self.calculate_dispersion(t) for t in results["times"]],
            "remaining": [self.calculate_remaining_mass(t) for t in results["times"]]
        }


class JetSpillModel(SpillModel):
    """
    Model for high-pressure jet spills.
    
    This model accounts for:
    - Initial jet dynamics
    - Atomization and droplet formation
    - Trajectory and deposition
    """
    
    def __init__(self, fluid: str, rate: float, total_mass: float, 
                 opening_diameter: float, temperature: float, pressure: float,
                 environment: EnvironmentProperties, height: float = 0.0, 
                 angle: float = 0.0, direction: float = 0.0):
        """
        Initialize the jet spill model.
        
        Args:
            fluid (str): Type of fluid
            rate (float): Spill rate in kg/s
            total_mass (float): Total mass in kg
            opening_diameter (float): Opening diameter in m
            temperature (float): Fluid temperature in K
            pressure (float): Fluid pressure in Pa
            environment (EnvironmentProperties): Environmental properties
            height (float): Release height above ground/water in m
            angle (float): Release angle from horizontal in degrees
            direction (float): Release direction (azimuth) in degrees
        """
        super().__init__("Jet", fluid, rate, total_mass, 
                         opening_diameter, temperature, pressure, environment)
        
        # Jet-specific parameters
        self.height = height
        self.angle = angle  # In degrees
        self.direction = direction  # In degrees
        
        # Calculate initial velocity from pressure
        # v = √(2 * ΔP / ρ)
        # where ΔP is pressure difference, ρ is fluid density
        delta_p = self.pressure - 101325  # Pressure difference from atmospheric
        self.initial_velocity = math.sqrt(2 * delta_p / self.fluid.density)
        
        # Calculate jet parameters
        # Angle in radians
        angle_rad = math.radians(self.angle)
        
        # Initial velocity components
        self.v_x = self.initial_velocity * math.cos(angle_rad) * math.cos(math.radians(self.direction))
        self.v_y = self.initial_velocity * math.cos(angle_rad) * math.sin(math.radians(self.direction))
        self.v_z = self.initial_velocity * math.sin(angle_rad)
        
        # Calculate trajectory parameters
        # Maximum height and range for simple ballistic trajectory
        if self.v_z > 0:
            self.max_height = self.height + (self.v_z**2) / (2 * 9.81)
        else:
            self.max_height = self.height
        
        # Range (horizontal distance) for ballistic trajectory
        # Simplified by ignoring air resistance
        if self.environment.is_water and self.height > 0:
            # For jets over water, account for impact and then spreading
            # Time to reach water surface
            if self.v_z < 0:
                # Initial downward velocity
                time_to_surface = (-self.v_z + math.sqrt(self.v_z**2 + 2 * 9.81 * self.height)) / 9.81
            elif self.v_z > 0:
                # Initial upward velocity
                # Time to reach max height and then fall to surface
                time_up = self.v_z / 9.81
                time_down = math.sqrt(2 * (self.max_height - self.height) / 9.81)
                time_to_surface = time_up + time_down
            else:
                # Horizontal velocity only
                time_to_surface = math.sqrt(2 * self.height / 9.81)
            
            horizontal_speed = math.sqrt(self.v_x**2 + self.v_y**2)
            self.range = horizontal_speed * time_to_surface
        elif self.environment.is_land or self.height <= 0:
            # For jets on land or at ground level
            if self.v_z >= 0:
                # Upward or horizontal initial velocity
                time_up = self.v_z / 9.81
                time_down = math.sqrt(2 * (self.max_height - self.height) / 9.81)
                time_total = time_up + time_down
                horizontal_speed = math.sqrt(self.v_x**2 + self.v_y**2)
                self.range = horizontal_speed * time_total
            else:
                # Downward initial velocity
                time_to_ground = (-self.v_z + math.sqrt(self.v_z**2 + 2 * 9.81 * self.height)) / 9.81
                horizontal_speed = math.sqrt(self.v_x**2 + self.v_y**2)
                self.range = horizontal_speed * time_to_ground
        
        # Droplet size distribution
        # For jet breakup, use a correlation based on Weber number
        # Weber number = ρ * v² * d / σ
        # where ρ is density, v is velocity, d is diameter, σ is surface tension
        weber_number = (self.fluid.density * self.initial_velocity**2 * 
                        self.opening_diameter / self.fluid.surface_tension)
        
        # Sauter mean diameter (SMD) for droplets
        # SMD ∝ d * We^(-0.6)
        self.droplet_smd = self.opening_diameter * (weber_number ** -0.6)
        
        # Set evaporation parameters based on droplet size and velocity
        evap_enhancement = 1 + 0.1 * self.initial_velocity
        self.evaporation_coefficient = 0.0025 * evap_enhancement
    
    def _setup_case_specific(self):
        """Set up jet spill specific configuration."""
        if not self.case_manager:
            raise ValueError("Case manager not initialized")
        
        # Set solver based on environment
        if self.environment.is_water:
            solver = "multiphaseInterFoam"
        else:
            solver = "interFoam"
        
        # Set up basic case structure
        self.case_manager.set_solver(solver)
        
        # Set up transport properties
        transport_model = TransportModel(self.case_manager)
        
        # Add phases
        if self.environment.is_water:
            # Water, oil, and air
            transport_model.add_phase("water", self.environment.water_density, self.environment.water_viscosity)
            transport_model.add_phase(self.fluid_name.lower(), self.fluid.density, self.fluid.viscosity)
            transport_model.add_phase("air", 1.2, 1.8e-5)
            
            # Set surface tensions
            transport_model.set_surface_tension("water", self.fluid_name.lower(), self.fluid.surface_tension)
            transport_model.set_surface_tension("water", "air", 0.072)
            transport_model.set_surface_tension(self.fluid_name.lower(), "air", 0.025)
        else:
            # Oil and air
            transport_model.add_phase(self.fluid_name.lower(), self.fluid.density, self.fluid.viscosity)
            transport_model.add_phase("air", 1.2, 1.8e-5)
            
            # Set surface tension
            transport_model.set_surface_tension(self.fluid_name.lower(), "air", self.fluid.surface_tension)
        
        # Write transport properties
        transport_model.write()
        
        # Set up domain and mesh
        # Domain needs to be large enough to capture the jet trajectory
        # Estimate domain size based on range and max height
        x_min = -10.0
        x_max = max(100.0, self.range * 1.5) * math.cos(math.radians(self.direction))
        y_min = -10.0
        y_max = max(100.0, self.range * 1.5) * math.sin(math.radians(self.direction))
        z_min = -5.0 if self.environment.is_water else 0.0
        z_max = max(10.0, self.max_height * 1.5)
        
        # Create mesh with refinement near the jet source and trajectory
        self.case_manager.create_box_mesh(
            x_min=x_min, x_max=x_max, 
            y_min=y_min, y_max=y_max, 
            z_min=z_min, z_max=z_max,
            x_cells=int(max(50, (x_max - x_min) / 2)),
            y_cells=int(max(50, (y_max - y_min) / 2)),
            z_cells=int(max(50, (z_max - z_min) / 0.5))
        )
        
        # Set up initial fields
        if self.environment.is_water:
            # Initialize water/air interface at z=0
            self.case_manager.set_field_initialization("alpha.water", f"pos().z <= 0 ? 1 : 0")
        
        # Set up source for jet
        # Position based on height and angle
        source_center = [0, 0, self.height]
        source_direction = [
            math.cos(math.radians(self.angle)) * math.cos(math.radians(self.direction)),
            math.cos(math.radians(self.angle)) * math.sin(math.radians(self.direction)),
            math.sin(math.radians(self.angle))
        ]
        
        # Create inlet patch for jet source
        inlet_patch = BoundaryCondition("inlet", "patch")
        inlet_patch.set_location(
            f"mag(vector(pos().x - {source_center[0]}, pos().y - {source_center[1]}, pos().z - {source_center[2]})) < {self.opening_diameter/2}"
        )
        
        # Set velocity at inlet
        inlet_patch.set_velocity(
            [self.v_x, self.v_y, self.v_z]
        )
        
        # Set alpha fields at inlet
        inlet_patch.set_field(f"alpha.{self.fluid_name.lower()}", 1.0)
        if self.environment.is_water:
            inlet_patch.set_field("alpha.water", 0.0)
        inlet_patch.set_field("alpha.air", 0.0)
        
        # Add patch to case
        self.case_manager.add_boundary_condition(inlet_patch)
        
        # Set up environmental conditions
        if self.environment.is_water:
            # Set up water current
            if self.environment.current_speed > 0:
                current_patch = BoundaryCondition("inlet_current", "patch")
                current_patch.set_location(f"pos().x <= {x_min} && pos().z <= 0")
                current_patch.set_velocity([self.environment.current_speed, 0, 0])
                self.case_manager.add_boundary_condition(current_patch)
        
        # Set up wind
        if self.environment.wind_speed > 0:
            wind_patch = BoundaryCondition("atmosphere", "patch")
            wind_patch.set_location(f"pos().z >= {z_max}")
            wind_patch.set_velocity([self.environment.wind_speed, 0, 0])
            self.case_manager.add_boundary_condition(wind_patch)
        
        # Write case files
        self.case_manager.write_case()
    
    def calculate_trajectory(self, time: float) -> List[float]:
        """
        Calculate position at a given time along the jet trajectory.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            List[float]: [x, y, z] position in m
        """
        if time <= 0:
            return [0, 0, self.height]
        
        # Simple ballistic trajectory
        # x = x₀ + v_x * t
        # y = y₀ + v_y * t
        # z = z₀ + v_z * t - 0.5 * g * t²
        
        x = self.v_x * time
        y = self.v_y * time
        z = self.height + self.v_z * time - 0.5 * 9.81 * time**2
        
        # If we hit the ground/water, position stays at the surface
        if z < (0 if self.environment.is_water else 0):
            # Calculate time of impact
            if self.v_z < 0:
                # Initial downward velocity
                t_impact = (-self.v_z + math.sqrt(self.v_z**2 + 2 * 9.81 * self.height)) / 9.81
            elif self.v_z > 0:
                # Initial upward velocity
                # Time to reach max height and then fall to surface
                t_up = self.v_z / 9.81
                t_down = math.sqrt(2 * (self.max_height - self.height) / 9.81)
                t_impact = t_up + t_down
            else:
                # Horizontal velocity only
                t_impact = math.sqrt(2 * self.height / 9.81)
            
            # Position at impact
            x = self.v_x * t_impact
            y = self.v_y * t_impact
            z = 0 if self.environment.is_water else 0
        
        return [x, y, z]
    
    def calculate_area(self, time: float) -> float:
        """
        Calculate affected area at a given time.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            float: Affected area in m²
        """
        if time <= 0:
            return 0.0
        
        # Calculate volume at current time
        volume = self.calculate_volume(time)
        if volume <= 0:
            return 0.0
        
        # Calculate trajectory endpoint
        endpoint = self.calculate_trajectory(min(time, self.duration))
        
        # Check if we've hit the ground/water
        if endpoint[2] <= 0:
            # After impact, use spreading models similar to surface spill
            # Time since impact
            if self.v_z < 0:
                # Initial downward velocity
                t_impact = (-self.v_z + math.sqrt(self.v_z**2 + 2 * 9.81 * self.height)) / 9.81
            elif self.v_z > 0:
                # Initial upward velocity
                t_up = self.v_z / 9.81
                t_down = math.sqrt(2 * (self.max_height - self.height) / 9.81)
                t_impact = t_up + t_down
            else:
                # Horizontal velocity only
                t_impact = math.sqrt(2 * self.height / 9.81)
            
            time_since_impact = time - t_impact
            
            if time_since_impact <= 0:
                # Not yet impacted
                # Area is approximated as a line of droplets with width based on breakup
                length = math.sqrt(endpoint[0]**2 + endpoint[1]**2)
                width = 10 * self.opening_diameter  # Approximate jet width after breakup
                return length * width
            else:
                if self.environment.is_water:
                    # Water surface spreading
                    # Use Fay's model with reduced coefficients due to initial momentum
                    k1 = 0.5  # Reduced spreading coefficient
                    
                    # Calculate density difference ratio
                    density_diff_ratio = abs(self.fluid.density - self.environment.water_density) / self.environment.water_density
                    
                    # Use gravity-inertial spreading equation
                    area = math.pi * (k1 * (9.81 * density_diff_ratio * volume**2)**(1/3) * time_since_impact**(2/3))**2
                    
                    # Ensure minimum area
                    min_area = math.pi * (5 * self.opening_diameter)**2
                    return max(min_area, area)
                else:
                    # Land spreading
                    # Much more limited spreading on land
                    if self.environment.permeability < 1e-12:
                        # Impermeable surface (like concrete)
                        # Simple gravity-driven spreading with friction
                        k_land = 0.5  # Spreading coefficient for land
                        
                        area = math.pi * (k_land * volume**(2/3) * time_since_impact**(1/3))**2
                        
                        # Ensure minimum area
                        min_area = math.pi * (5 * self.opening_diameter)**2
                        return max(min_area, area)
                    else:
                        # Permeable surface (like soil)
                        # Model both horizontal spreading and vertical infiltration
                        k_perm = 0.3  # Spreading coefficient for permeable land
                        
                        area = math.pi * (k_perm * volume**(2/3) * time_since_impact**(1/4) * 
                                        (1/self.environment.permeability)**(1/4))**2
                        
                        # Ensure minimum area
                        min_area = math.pi * (5 * self.opening_diameter)**2
                        return max(min_area, area)
        else:
            # Still in air - area is approximated as a line of droplets with width based on breakup
            length = math.sqrt(endpoint[0]**2 + endpoint[1]**2 + (endpoint[2] - self.height)**2)
            width = 10 * self.opening_diameter  # Approximate jet width after breakup
            return length * width
    
    def calculate_volume(self, time: float) -> float:
        """
        Calculate spill volume at a given time.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            float: Spill volume in m³
        """
        if time <= 0:
            return 0.0
        
        # Basic calculation of released volume
        actual_time = min(time, self.duration)
        released_volume = (self.spill_rate * actual_time) / self.fluid.density
        
        # Adjust for evaporation which is significant for atomized jet
        evaporated_mass = self.calculate_evaporation(time)
        
        # Calculate net volume
        net_volume = released_volume - evaporated_mass / self.fluid.density
        
        # Ensure non-negative
        return max(0.0, net_volume)
    
    def calculate_evaporation(self, time: float) -> float:
        """
        Calculate evaporated mass for jet spill.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            float: Evaporated mass in kg
        """
        if time <= 0:
            return 0.0
        
        # For jets, evaporation is enhanced due to atomization
        # Use a simplified model based on droplet size distribution
        
        # Calculate released mass
        released_mass = min(time, self.duration) * self.spill_rate
        
        # Estimate surface area of droplets
        # A = 6 * V / d_smd
        # where V is volume, d_smd is Sauter mean diameter
        volume = released_mass / self.fluid.density
        droplet_area = 6 * volume / self.droplet_smd
        
        # Adjust evaporation coefficient based on wind speed
        wind_factor = 1.0 + 0.1 * self.environment.wind_speed
        
        # Adjust for temperature
        temp_factor = math.exp(0.03 * (self.temperature - 293.15))
        
        # Gas constant
        R = 8.314
        
        # Calculate evaporation rate (kg/s)
        evap_rate = (self.evaporation_coefficient * wind_factor * temp_factor * 
                    droplet_area * (self.fluid.vapor_pressure / (R * self.temperature)))
        
        # Calculate total evaporated mass
        total_evaporated = evap_rate * time
        
        # For highly volatile fluids (like gasoline), enhance evaporation
        if self.fluid_name == "Gasoline":
            evaporation_factor = 0.5  # 50% evaporation limit
        elif self.fluid_name == "Diesel":
            evaporation_factor = 0.2  # 20% evaporation limit
        elif self.fluid_name == "Crude Oil":
            evaporation_factor = 0.3  # 30% evaporation limit
        else:
            evaporation_factor = 0.25  # 25% evaporation limit for other fluids
        
        # Limit evaporation to a percentage of released mass
        max_evaporation = released_mass * evaporation_factor
        
        return min(total_evaporated, max_evaporation)
    
    def calculate_dispersion(self, time: float) -> float:
        """
        Calculate dispersed mass for jet spill.
        
        Args:
            time (float): Time in seconds
            
        Returns:
            float: Dispersed mass in kg
        """
        if time <= 0 or not self.environment.is_water:
            return 0.0
        
        # Calculate trajectory endpoint
        endpoint = self.calculate_trajectory(min(time, self.duration))
        
        # Only consider dispersion after impact with water
        if endpoint[2] > 0:
            return 0.0
        
        # Calculate impact time
        if self.v_z < 0:
            # Initial downward velocity
            t_impact = (-self.v_z + math.sqrt(self.v_z**2 + 2 * 9.81 * self.height)) / 9.81
        elif self.v_z > 0:
            # Initial upward velocity
            t_up = self.v_z / 9.81
            t_down = math.sqrt(2 * (self.max_height - self.height) / 9.81)
            t_impact = t_up + t_down
        else:
            # Horizontal velocity only
            t_impact = math.sqrt(2 * self.height / 9.81)
        
        time_since_impact = time - t_impact
        if time_since_impact <= 0:
            return 0.0
        
        # For jet hitting water surface, initial dispersion is significant
        # Estimate fraction dispersed on impact based on height and velocity
        impact_speed = math.sqrt((self.v_x**2 + self.v_y**2) + 
                                (self.v_z - 9.81 * t_impact)**2)
        
        # Higher impact speed means more initial dispersion
        if impact_speed < 1.0:
            initial_dispersion_fraction = 0.05  # 5%
        elif impact_speed < 5.0:
            initial_dispersion_fraction = 0.15  # 15%
        else:
            initial_dispersion_fraction = 0.25  # 25%
        
        # Calculate mass released before impact
        mass_at_impact = min(t_impact, self.duration) * self.spill_rate
        
        # Calculate initial dispersion
        initial_dispersion = mass_at_impact * initial_dispersion_fraction
        
        # Calculate continued dispersion using similar model to surface spill
        # Calculate thickness for remaining surface oil
        area = self.calculate_area(time)
        if area <= 0:
            return initial_dispersion
        
        thickness = self.calculate_thickness(time)
        
        # Dispersion requires wave action
        if self.environment.wave_height < 0.1:
            return initial_dispersion
        
        # Simplified Delvigne and Sweeney model
        dispersion_factor = 0.0005 if self.fluid.density < 850 else 0.0002
        
        # Calculate fraction dispersed per second
        fraction_per_second = (dispersion_factor * 
                              (self.environment.wind_speed**2) * 
                              (self.environment.wave_height**0.5) / 
                              max(0.001, thickness)) / 3600  # Convert from per hour to per second
        
        # Calculate mass released after impact
        if time <= self.duration:
            mass_after_impact = self.spill_rate * (time - t_impact)
        else:
            mass_after_impact = self.spill_rate * (self.duration - t_impact)
        
        mass_after_impact = max(0, mass_after_impact)
        
        # Calculate surface mass available for continued dispersion
        evaporated_mass = self.calculate_evaporation(time)
        surface_mass = mass_after_impact - evaporated_mass
        
        # Calculate continued dispersion
        continued_dispersion = surface_mass * fraction_per_second * time_since_impact
        
        # Total dispersion is sum of initial and continued dispersion
        total_dispersion = initial_dispersion + continued_dispersion
        
        # Limit to available mass
        released_mass = min(time, self.duration) * self.spill_rate
        evaporated_mass = self.calculate_evaporation(time)
        available_mass = released_mass - evaporated_mass
        
        return min(total_dispersion, available_mass)
    
    def _process_results(self):
        """Process OpenFOAM simulation results for jet spill."""
        if not self.case_manager:
            logger.warning("No case manager to process results")
            return
        
        # Get list of time directories
        time_dirs = self.case_manager.get_time_directories()
        
        # Process each time directory
        results = {
            "times": [],
            "areas": [],
            "volumes": [],
            "trajectories": [],
            "impact_points": []
        }
        
        for time_dir in time_dirs:
            time = float(os.path.basename(time_dir))
            
            # Read alpha field to determine spill distribution
            alpha_field = self.case_manager.read_field(f"alpha.{self.fluid_name.lower()}", time_dir)
            
            if alpha_field is None:
                continue
            
            # Calculate area and volume
            area = self.calculate_area(time)
            volume = self.calculate_volume(time)
            
            # Calculate trajectory point at this time
            trajectory = self.calculate_trajectory(time)
            
            # Store results
            results["times"].append(time)
            results["areas"].append(area)
            results["volumes"].append(volume)
            results["trajectories"].append(trajectory)
            
            # Calculate impact point (if applicable)
            endpoint = self.calculate_trajectory(min(time, self.duration))
            if endpoint[2] <= 0:
                results["impact_points"].append([endpoint[0], endpoint[1], 0])
            else:
                results["impact_points"].append(None)
        
        # Store in results dictionary
        self.results["simulation"] = results
        
        # Add weathering results
        self.results["weathering"] = {
            "times": results["times"],
            "evaporation": [self.calculate_evaporation(t) for t in results["times"]],
            "dispersion": [self.calculate_dispersion(t) for t in results["times"]],
            "remaining": [self.calculate_remaining_mass(t) for t in results["times"]]
        }


class SpillModelFactory:
    """
    Factory class for creating spill models.
    
    This class provides methods to create the appropriate spill model
    based on the spill type and parameters.
    """
    
    @staticmethod
    def create_model(spill_type: str, fluid: str, rate: float, total_mass: float, 
                     opening_diameter: float, temperature: float, pressure: float,
                     environment: EnvironmentProperties, **kwargs) -> SpillModel:
        """
        Create a spill model.
        
        Args:
            spill_type (str): Type of spill (Surface, Subsurface, Jet)
            fluid (str): Type of fluid
            rate (float): Spill rate in kg/s
            total_mass (float): Total mass in kg
            opening_diameter (float): Opening diameter in m
            temperature (float): Fluid temperature in K
            pressure (float): Fluid pressure in Pa
            environment (EnvironmentProperties): Environmental properties
            **kwargs: Additional parameters for specific spill types
            
        Returns:
            SpillModel: The created spill model
            
        Raises:
            ValueError: If spill type is unknown
        """
        if spill_type.lower() == "surface":
            return SurfaceSpillModel(
                fluid, rate, total_mass, opening_diameter, 
                temperature, pressure, environment
            )
        elif spill_type.lower() == "subsurface":
            # Check for required parameter
            if "depth" not in kwargs:
                raise ValueError("Subsurface spill model requires 'depth' parameter")
            
            return SubsurfaceSpillModel(
                fluid, rate, total_mass, opening_diameter, 
                temperature, pressure, environment, kwargs["depth"]
            )
        elif spill_type.lower() == "jet":
            # Extract optional parameters
            height = kwargs.get("height", 0.0)
            angle = kwargs.get("angle", 0.0)
            direction = kwargs.get("direction", 0.0)
            
            return JetSpillModel(
                fluid, rate, total_mass, opening_diameter, 
                temperature, pressure, environment,
                height, angle, direction
            )
        else:
            raise ValueError(f"Unknown spill type: {spill_type}")


def create_spill_model(
        spill_type: str, 
        fluid: str, 
        rate: float, 
        total_mass: float,
        opening_diameter: float, 
        temperature: float, 
        pressure: float,
        env_type: str, 
        ambient_temperature: float, 
        wind_speed: float = 0.0,
        water_temperature: float = None, 
        current_speed: float = 0.0, 
        wave_height: float = 0.0,
        salinity: float = 0.0, 
        soil_type: str = None, 
        permeability: float = None,
        **kwargs) -> SpillModel:
    """
    Create a spill model with parameters.
    
    Args:
        spill_type (str): Type of spill (Surface, Subsurface, Jet)
        fluid (str): Type of fluid
        rate (float): Spill rate in kg/s
        total_mass (float): Total mass in kg
        opening_diameter (float): Opening diameter in m
        temperature (float): Fluid temperature in K
        pressure (float): Fluid pressure in Pa
        env_type (str): Environment type (e.g., "Water (Ocean)", "Land (Soil)")
        ambient_temperature (float): Ambient temperature in K
        wind_speed (float, optional): Wind speed in m/s
        water_temperature (float, optional): Water temperature in K
        current_speed (float, optional): Water current speed in m/s
        wave_height (float, optional): Wave height in m
        salinity (float, optional): Water salinity in ppt
        soil_type (str, optional): Soil type for land spills
        permeability (float, optional): Soil permeability in m²
        **kwargs: Additional parameters for specific spill types
        
    Returns:
        SpillModel: The created spill model
    """
    # Create environment properties
    environment = EnvironmentProperties(
        env_type=env_type,
        temperature=ambient_temperature,
        wind_speed=wind_speed,
        water_temperature=water_temperature,
        current_speed=current_speed,
        wave_height=wave_height,
        salinity=salinity,
        soil_type=soil_type,
        permeability=permeability
    )
    
    # Create spill model using factory
    return SpillModelFactory.create_model(
        spill_type, fluid, rate, total_mass, opening_diameter, 
        temperature, pressure, environment, **kwargs
    )


# Entry point for command-line usage
if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Oil & Gas Spill Modeling")
    
    # Add arguments
    parser.add_argument("--type", choices=["surface", "subsurface", "jet"], required=True,
                       help="Type of spill")
    parser.add_argument("--fluid", default="Crude Oil",
                       help="Fluid type (Crude Oil, Diesel, Gasoline, Natural Gas)")
    parser.add_argument("--rate", type=float, required=True,
                       help="Spill rate in kg/s")
    parser.add_argument("--mass", type=float, required=True,
                       help="Total spill mass in kg")
    parser.add_argument("--diameter", type=float, required=True,
                       help="Opening diameter in m")
    parser.add_argument("--temperature", type=float, default=293.15,
                       help="Fluid temperature in K")
    parser.add_argument("--pressure", type=float, default=101325,
                       help="Fluid pressure in Pa")
    parser.add_argument("--environment", choices=["water-ocean", "water-river", "water-lake", 
                                               "land-soil", "land-concrete"], 
                       default="water-ocean", help="Environment type")
    parser.add_argument("--ambient-temp", type=float, default=293.15,
                       help="Ambient temperature in K")
    parser.add_argument("--wind-speed", type=float, default=5.0,
                       help="Wind speed in m/s")
    parser.add_argument("--depth", type=float, default=10.0,
                       help="Release depth for subsurface spill in m")
    parser.add_argument("--height", type=float, default=0.0,
                       help="Release height for jet spill in m")
    parser.add_argument("--angle", type=float, default=0.0,
                       help="Release angle for jet spill in degrees")
    parser.add_argument("--direction", type=float, default=0.0,
                       help="Release direction for jet spill in degrees")
    parser.add_argument("--output", default="spill_results.json",
                       help="Output file for results")
    parser.add_argument("--case-dir", default="./spill_case",
                       help="Directory for OpenFOAM case")
    parser.add_argument("--run-simulation", action="store_true",
                       help="Run OpenFOAM simulation")
    parser.add_argument("--parallel", action="store_true",
                       help="Run simulation in parallel")
    parser.add_argument("--processors", type=int, default=4,
                       help="Number of processors for parallel run")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Map environment string to env_type
    env_map = {
        "water-ocean": "Water (Ocean)",
        "water-river": "Water (River)",
        "water-lake": "Water (Lake)",
        "land-soil": "Land (Soil)",
        "land-concrete": "Land (Concrete)"
    }
    env_type = env_map[args.environment]
    
    # Create kwargs for specific spill types
    kwargs = {}
    if args.type == "subsurface":
        kwargs["depth"] = args.depth
    elif args.type == "jet":
        kwargs["height"] = args.height
        kwargs["angle"] = args.angle
        kwargs["direction"] = args.direction
    
    # Create spill model
    model = create_spill_model(
        spill_type=args.type,
        fluid=args.fluid,
        rate=args.rate,
        total_mass=args.mass,
        opening_diameter=args.diameter,
        temperature=args.temperature,
        pressure=args.pressure,
        env_type=env_type,
        ambient_temperature=args.ambient_temp,
        wind_speed=args.wind_speed,
        **kwargs
    )
    
    # Set up case
    model.setup_case(args.case_dir)
    
    # Run simulation if requested
    if args.run_simulation:
        success = model.run_simulation(args.parallel, args.processors)
        if not success:
            logger.error("Simulation failed")
    
    # Calculate and print summary
    times = [0, 60, 300, 600, 1800, 3600, 7200, 14400, 28800, 86400]  # 0s to 24h
    print("\nSpill Summary:")
    print("--------------")
    for t in times:
        if t == 0:
            continue  # Skip t=0
        
        summary = model.get_summary(t)
        hours = t / 3600
        print(f"\nTime: {hours:.2f} hours")
        print(f"Released Mass: {summary['released_mass']:.2f} kg")
        print(f"Area: {summary['area']:.2f} m²")
        print(f"Thickness: {summary['thickness']*1000:.2f} mm")
        print(f"Evaporated: {summary['evaporated_percentage']:.1f}%")
        print(f"Dispersed: {summary['dispersed_percentage']:.1f}%")
        print(f"Remaining: {summary['remaining_percentage']:.1f}%")
    
    # Export results
    model.export_results(args.output)
    print(f"\nResults exported to {args.output}")