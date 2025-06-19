#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow simulation model for Openfoam_Simulator application.

This module implements the flow model that represents the fluid flow simulation
configuration and results for CFD simulations in the oil & gas industry.
"""

import os
import sys
import json
import logging
import datetime
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set

# Import utility modules
from ..utils.logger import get_logger
from ..config import get_value, set_value

logger = get_logger(__name__)


class PhaseProperties:
    """Helper class to store phase properties for multiphase flows."""
    
    def __init__(self, name: str, fluid_type: str, 
                 density: float, viscosity: float, 
                 volume_fraction: float = 1.0,
                 specific_heat: float = None,
                 thermal_conductivity: float = None,
                 color: List[float] = None):
        """
        Initialize phase properties.
        
        Args:
            name (str): Phase name
            fluid_type (str): Type of fluid ("water", "oil", "gas", etc.)
            density (float): Density in kg/m³
            viscosity (float): Dynamic viscosity in Pa·s
            volume_fraction (float): Initial volume fraction
            specific_heat (float, optional): Specific heat in J/(kg·K)
            thermal_conductivity (float, optional): Thermal conductivity in W/(m·K)
            color (List[float], optional): RGB color for visualization
        """
        self.name = name
        self.fluid_type = fluid_type
        self.density = density
        self.viscosity = viscosity
        self.volume_fraction = volume_fraction
        self.specific_heat = specific_heat
        self.thermal_conductivity = thermal_conductivity
        self.color = color or [0.5, 0.5, 0.5]  # Default gray
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "fluid_type": self.fluid_type,
            "density": self.density,
            "viscosity": self.viscosity,
            "volume_fraction": self.volume_fraction,
            "specific_heat": self.specific_heat,
            "thermal_conductivity": self.thermal_conductivity,
            "color": self.color
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhaseProperties':
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            fluid_type=data.get("fluid_type", "custom"),
            density=data.get("density", 1000.0),
            viscosity=data.get("viscosity", 0.001),
            volume_fraction=data.get("volume_fraction", 1.0),
            specific_heat=data.get("specific_heat"),
            thermal_conductivity=data.get("thermal_conductivity"),
            color=data.get("color")
        )
    
    def get_kinematic_viscosity(self) -> float:
        """
        Calculate kinematic viscosity.
        
        Returns:
            float: Kinematic viscosity in m²/s
        """
        return self.viscosity / self.density


class InterfaceProperties:
    """Helper class to store interface properties for multiphase flows."""
    
    def __init__(self, phase1: str, phase2: str, 
                 surface_tension: float,
                 contact_angle: float = 90.0):
        """
        Initialize interface properties.
        
        Args:
            phase1 (str): Name of first phase
            phase2 (str): Name of second phase
            surface_tension (float): Surface tension in N/m
            contact_angle (float): Contact angle in degrees
        """
        self.phase1 = phase1
        self.phase2 = phase2
        self.surface_tension = surface_tension
        self.contact_angle = contact_angle
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "phase1": self.phase1,
            "phase2": self.phase2,
            "surface_tension": self.surface_tension,
            "contact_angle": self.contact_angle
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InterfaceProperties':
        """Create from dictionary."""
        return cls(
            phase1=data.get("phase1", ""),
            phase2=data.get("phase2", ""),
            surface_tension=data.get("surface_tension", 0.072),
            contact_angle=data.get("contact_angle", 90.0)
        )


class BoundaryCondition:
    """Helper class to store boundary conditions for flow models."""
    
    def __init__(self, name: str, type_name: str, 
                 velocity: List[float] = None,
                 pressure: float = None,
                 temperature: float = None,
                 volume_fractions: Dict[str, float] = None,
                 turbulence_intensity: float = None,
                 turbulence_length_scale: float = None):
        """
        Initialize boundary condition.
        
        Args:
            name (str): Boundary name
            type_name (str): Type of boundary ("inlet", "outlet", "wall", etc.)
            velocity (List[float], optional): Velocity vector [u, v, w]
            pressure (float, optional): Pressure value
            temperature (float, optional): Temperature value
            volume_fractions (Dict[str, float], optional): Volume fractions for each phase
            turbulence_intensity (float, optional): Turbulence intensity
            turbulence_length_scale (float, optional): Turbulence length scale
        """
        self.name = name
        self.type_name = type_name
        self.velocity = velocity or [0.0, 0.0, 0.0]
        self.pressure = pressure
        self.temperature = temperature
        self.volume_fractions = volume_fractions or {}
        self.turbulence_intensity = turbulence_intensity
        self.turbulence_length_scale = turbulence_length_scale
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type_name": self.type_name,
            "velocity": self.velocity,
            "pressure": self.pressure,
            "temperature": self.temperature,
            "volume_fractions": self.volume_fractions,
            "turbulence_intensity": self.turbulence_intensity,
            "turbulence_length_scale": self.turbulence_length_scale
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BoundaryCondition':
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            type_name=data.get("type_name", "wall"),
            velocity=data.get("velocity"),
            pressure=data.get("pressure"),
            temperature=data.get("temperature"),
            volume_fractions=data.get("volume_fractions"),
            turbulence_intensity=data.get("turbulence_intensity"),
            turbulence_length_scale=data.get("turbulence_length_scale")
        )


class FlowResults:
    """Helper class to store flow simulation results."""
    
    def __init__(self, num_timesteps: int = 0,
                 times: List[float] = None,
                 field_names: List[str] = None,
                 converged: bool = False,
                 residuals: Dict[str, List[float]] = None,
                 result_files: List[str] = None):
        """
        Initialize flow results.
        
        Args:
            num_timesteps (int): Number of time steps
            times (List[float], optional): Time values
            field_names (List[str], optional): Names of result fields
            converged (bool): Whether the simulation converged
            residuals (Dict[str, List[float]], optional): Residual history for each field
            result_files (List[str], optional): Paths to result files
        """
        self.num_timesteps = num_timesteps
        self.times = times or []
        self.field_names = field_names or []
        self.converged = converged
        self.residuals = residuals or {}
        self.result_files = result_files or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_timesteps": self.num_timesteps,
            "times": self.times,
            "field_names": self.field_names,
            "converged": self.converged,
            "residuals": self.residuals,
            "result_files": self.result_files
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FlowResults':
        """Create from dictionary."""
        return cls(
            num_timesteps=data.get("num_timesteps", 0),
            times=data.get("times"),
            field_names=data.get("field_names"),
            converged=data.get("converged", False),
            residuals=data.get("residuals"),
            result_files=data.get("result_files")
        )


class FlowModel:
    """
    Flow model class for Openfoam_Simulator application.
    
    This class represents a fluid flow simulation and provides functionality for:
    - Configuring different types of flows (single-phase, multi-phase)
    - Setting up flow properties and boundary conditions
    - Managing special oil & gas industry flows (pigging, spill simulation)
    - Post-processing and analyzing flow results
    """
    
    def __init__(self, name: str = "Untitled Flow", mesh_path: str = None, case_dir: str = None):
        """
        Initialize a new flow model.
        
        Args:
            name (str, optional): Flow model name
            mesh_path (str, optional): Path to the mesh file
            case_dir (str, optional): Directory for case files
        """
        # Flow metadata
        self.name = name
        self.description = ""
        self.created_date = datetime.datetime.now().isoformat()
        self.modified_date = self.created_date
        
        # Paths
        self.mesh_path = mesh_path
        self.case_dir = case_dir
        
        # Flow characteristics
        self.flow_type = "single_phase"  # single_phase, two_phase, three_phase, pigging, spill
        self.flow_regime = "turbulent"   # laminar, transitional, turbulent
        self.time_dependent = False
        
        # Reference values
        self.reference_pressure = 101325.0    # Pa
        self.reference_temperature = 293.15   # K
        self.gravity = [0.0, 0.0, -9.81]      # m/s²
        
        # Physical models
        self.turbulence_model = "k-epsilon"  # k-epsilon, k-omega, etc.
        self.turbulence_wall_function = "standard"
        self.heat_transfer = False
        self.buoyancy = False
        
        # Phases - dictionary of PhaseProperties objects
        self.phases = {}
        
        # Interfaces - list of InterfaceProperties objects
        self.interfaces = []
        
        # Boundary conditions - dictionary of BoundaryCondition objects
        self.boundary_conditions = {}
        
        # Initial conditions
        self.initial_conditions = {}  # Dictionary of field values
        
        # Solver settings
        self.max_time = 1000.0  # Maximum simulation time (s)
        self.time_step = 0.1     # Time step (s)
        self.adjustable_time_step = False
        self.max_courant_number = 1.0
        self.write_interval = 10.0  # Time interval for writing results
        
        # Numerical settings
        self.convergence_tolerance = 1e-5
        self.max_iterations = 1000
        self.relaxation_factors = {
            "p": 0.3,
            "U": 0.7,
            "T": 0.7,
            "k": 0.7,
            "epsilon": 0.7,
            "omega": 0.7
        }
        
        # Special models for oil & gas industry
        self.pigging_properties = {}
        self.spill_properties = {}
        
        # Results
        self.results = FlowResults()
        
        # Parallel settings
        self.parallel = False
        self.n_processors = 4
    
    def setup_single_phase_flow(self, fluid_type: str = "water", 
                               inlet_velocity: float = 1.0,
                               inlet_pressure: float = 101325.0,
                               outlet_pressure: float = 101325.0) -> bool:
        """
        Set up a single-phase flow simulation.
        
        Args:
            fluid_type (str): Type of fluid ("water", "oil", "gas", "custom")
            inlet_velocity (float): Inlet velocity in m/s
            inlet_pressure (float): Inlet pressure in Pa
            outlet_pressure (float): Outlet pressure in Pa
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Set flow type
            self.flow_type = "single_phase"
            self.time_dependent = False
            
            # Clear existing phases and add the single phase
            self.phases = {}
            
            # Create phase properties based on fluid type
            fluid_props = self._get_default_fluid_properties(fluid_type)
            
            # Add the phase
            self.phases["fluid"] = PhaseProperties(
                name="fluid",
                fluid_type=fluid_type,
                density=fluid_props["density"],
                viscosity=fluid_props["viscosity"],
                volume_fraction=1.0,
                specific_heat=fluid_props.get("specific_heat"),
                thermal_conductivity=fluid_props.get("thermal_conductivity")
            )
            
            # Set up default boundary conditions
            self._setup_default_boundaries(inlet_velocity, inlet_pressure, outlet_pressure)
            
            # Set simulation parameters appropriate for the flow
            if fluid_type == "water" or fluid_type == "oil":
                self.reference_pressure = 101325.0  # 1 atm
                self.turbulence_model = "k-epsilon"
            elif fluid_type == "gas" or fluid_type == "air":
                self.reference_pressure = 101325.0  # 1 atm
                self.turbulence_model = "k-omega"
            
            # Choose appropriate solver
            if self.flow_regime == "laminar":
                self.turbulence_model = "laminar"
            
            # Update metadata
            self.modified_date = datetime.datetime.now().isoformat()
            
            logger.info(f"Set up single-phase flow with {fluid_type} fluid")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up single-phase flow: {e}")
            return False
    
    def setup_two_phase_flow(self, flow_type: str = "Oil-Water",
                            phase1_type: str = "water", 
                            phase2_type: str = "oil",
                            phase1_fraction: float = 0.7,
                            phase2_fraction: float = 0.3,
                            surface_tension: float = 0.025) -> bool:
        """
        Set up a two-phase flow simulation.
        
        Args:
            flow_type (str): Type of two-phase flow ("Oil-Water", "Gas-Liquid")
            phase1_type (str): Type of first phase ("water", "oil", "gas")
            phase2_type (str): Type of second phase ("water", "oil", "gas")
            phase1_fraction (float): Volume fraction of first phase
            phase2_fraction (float): Volume fraction of second phase
            surface_tension (float): Surface tension between phases in N/m
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Set flow type
            self.flow_type = "two_phase"
            self.time_dependent = True
            
            # Clear existing phases and interfaces
            self.phases = {}
            self.interfaces = []
            
            # Get properties for each phase
            phase1_props = self._get_default_fluid_properties(phase1_type)
            phase2_props = self._get_default_fluid_properties(phase2_type)
            
            # Add the phases
            self.phases["phase1"] = PhaseProperties(
                name="phase1",
                fluid_type=phase1_type,
                density=phase1_props["density"],
                viscosity=phase1_props["viscosity"],
                volume_fraction=phase1_fraction,
                specific_heat=phase1_props.get("specific_heat"),
                thermal_conductivity=phase1_props.get("thermal_conductivity")
            )
            
            self.phases["phase2"] = PhaseProperties(
                name="phase2",
                fluid_type=phase2_type,
                density=phase2_props["density"],
                viscosity=phase2_props["viscosity"],
                volume_fraction=phase2_fraction,
                specific_heat=phase2_props.get("specific_heat"),
                thermal_conductivity=phase2_props.get("thermal_conductivity")
            )
            
            # Add interface properties
            self.interfaces.append(InterfaceProperties(
                phase1="phase1",
                phase2="phase2",
                surface_tension=surface_tension,
                contact_angle=90.0  # Default neutral wetting
            ))
            
            # Set up default boundary conditions
            inlet_velocity = 1.0  # Default
            self._setup_default_boundaries(inlet_velocity, None, None, is_multiphase=True)
            
            # Set simulation parameters appropriate for the flow
            self.reference_pressure = 101325.0  # 1 atm
            
            if flow_type == "Oil-Water":
                self.turbulence_model = "k-epsilon"
                self.gravity = [0.0, 0.0, -9.81]  # Important for stratification
                self.buoyancy = True
            elif flow_type == "Gas-Liquid":
                self.turbulence_model = "k-omega"
                self.gravity = [0.0, 0.0, -9.81]
                self.buoyancy = True
            
            # Appropriate numerical settings for multiphase
            self.time_step = 0.001  # Smaller time step for VOF
            self.adjustable_time_step = True
            self.max_courant_number = 0.5
            self.convergence_tolerance = 1e-6
            
            # Update metadata
            self.modified_date = datetime.datetime.now().isoformat()
            
            logger.info(f"Set up two-phase flow with {flow_type} flow")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up two-phase flow: {e}")
            return False
    
    def setup_three_phase_flow(self, water_fraction: float = 0.4,
                              oil_fraction: float = 0.4,
                              gas_fraction: float = 0.2) -> bool:
        """
        Set up a three-phase flow simulation.
        
        Args:
            water_fraction (float): Volume fraction of water
            oil_fraction (float): Volume fraction of oil
            gas_fraction (float): Volume fraction of gas
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Set flow type
            self.flow_type = "three_phase"
            self.time_dependent = True
            
            # Clear existing phases and interfaces
            self.phases = {}
            self.interfaces = []
            
            # Get properties for each phase
            water_props = self._get_default_fluid_properties("water")
            oil_props = self._get_default_fluid_properties("oil")
            gas_props = self._get_default_fluid_properties("gas")
            
            # Add the phases
            self.phases["water"] = PhaseProperties(
                name="water",
                fluid_type="water",
                density=water_props["density"],
                viscosity=water_props["viscosity"],
                volume_fraction=water_fraction,
                specific_heat=water_props.get("specific_heat"),
                thermal_conductivity=water_props.get("thermal_conductivity"),
                color=[0.0, 0.0, 1.0]  # Blue
            )
            
            self.phases["oil"] = PhaseProperties(
                name="oil",
                fluid_type="oil",
                density=oil_props["density"],
                viscosity=oil_props["viscosity"],
                volume_fraction=oil_fraction,
                specific_heat=oil_props.get("specific_heat"),
                thermal_conductivity=oil_props.get("thermal_conductivity"),
                color=[0.6, 0.4, 0.1]  # Brown
            )
            
            self.phases["gas"] = PhaseProperties(
                name="gas",
                fluid_type="gas",
                density=gas_props["density"],
                viscosity=gas_props["viscosity"],
                volume_fraction=gas_fraction,
                specific_heat=gas_props.get("specific_heat"),
                thermal_conductivity=gas_props.get("thermal_conductivity"),
                color=[0.8, 0.8, 0.8]  # Light gray
            )
            
            # Add interface properties
            self.interfaces.append(InterfaceProperties(
                phase1="water",
                phase2="oil",
                surface_tension=0.025,  # Oil-water
                contact_angle=100.0     # Slightly oil-wetting
            ))
            
            self.interfaces.append(InterfaceProperties(
                phase1="water",
                phase2="gas",
                surface_tension=0.072,  # Water-air
                contact_angle=110.0     # Hydrophobic
            ))
            
            self.interfaces.append(InterfaceProperties(
                phase1="oil",
                phase2="gas",
                surface_tension=0.023,  # Oil-air
                contact_angle=80.0      # Slightly oil-wetting
            ))
            
            # Set up default boundary conditions
            inlet_velocity = 1.0  # Default
            self._setup_default_boundaries(inlet_velocity, None, None, is_multiphase=True)
            
            # Set simulation parameters appropriate for three-phase flow
            self.reference_pressure = 101325.0  # 1 atm
            self.turbulence_model = "k-epsilon"
            self.gravity = [0.0, 0.0, -9.81]
            self.buoyancy = True
            
            # Appropriate numerical settings for multiphase
            self.time_step = 0.0005  # Even smaller time step for three-phase
            self.adjustable_time_step = True
            self.max_courant_number = 0.3
            self.convergence_tolerance = 1e-6
            
            # Update metadata
            self.modified_date = datetime.datetime.now().isoformat()
            
            logger.info("Set up three-phase flow simulation")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up three-phase flow: {e}")
            return False
    
    def setup_pigging_simulation(self, pipe_diameter: float = 0.1, 
                                pig_type: str = "Foam",
                                driving_fluid: str = "Water") -> bool:
        """
        Set up a pipeline pigging simulation.
        
        Args:
            pipe_diameter (float): Pipe diameter in meters
            pig_type (str): Type of pig ("Foam", "Disc", "Cup", "Sphere")
            driving_fluid (str): Type of driving fluid ("Water", "Oil", "Gas")
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Set flow type
            self.flow_type = "pigging"
            self.time_dependent = True
            
            # Clear existing phases and interfaces
            self.phases = {}
            self.interfaces = []
            
            # Set up a two-phase flow for pigging simulation
            # (driving fluid and displaced fluid)
            driving_fluid_type = driving_fluid.lower()
            displaced_fluid_type = "oil"  # Default to oil as displaced fluid
            
            # Get properties for each phase
            driving_props = self._get_default_fluid_properties(driving_fluid_type)
            displaced_props = self._get_default_fluid_properties(displaced_fluid_type)
            
            # Add the phases
            self.phases["driving_fluid"] = PhaseProperties(
                name="driving_fluid",
                fluid_type=driving_fluid_type,
                density=driving_props["density"],
                viscosity=driving_props["viscosity"],
                volume_fraction=0.5,  # Initial value
                specific_heat=driving_props.get("specific_heat"),
                thermal_conductivity=driving_props.get("thermal_conductivity")
            )
            
            self.phases["displaced_fluid"] = PhaseProperties(
                name="displaced_fluid",
                fluid_type=displaced_fluid_type,
                density=displaced_props["density"],
                viscosity=displaced_props["viscosity"],
                volume_fraction=0.5,  # Initial value
                specific_heat=displaced_props.get("specific_heat"),
                thermal_conductivity=displaced_props.get("thermal_conductivity")
            )
            
            # Add interface properties
            self.interfaces.append(InterfaceProperties(
                phase1="driving_fluid",
                phase2="displaced_fluid",
                surface_tension=0.025,  # Water-oil default
                contact_angle=90.0
            ))
            
            # Set up pigging properties
            pig_density_map = {
                "Foam": 250.0,
                "Disc": 950.0,
                "Cup": 900.0,
                "Sphere": 800.0,
                "Intelligent": 1200.0,
                "Gel": 1050.0
            }
            
            pig_friction_map = {
                "Foam": 0.3,
                "Disc": 0.25,
                "Cup": 0.35,
                "Sphere": 0.2,
                "Intelligent": 0.15,
                "Gel": 0.4
            }
            
            # Calculate appropriate pig dimensions based on pipe diameter
            if pig_type == "Foam":
                pig_diameter = pipe_diameter * 1.05  # 5% oversized
                pig_length = pipe_diameter * 2.0
                bypass_flow = 0.08  # 8%
            elif pig_type == "Disc":
                pig_diameter = pipe_diameter * 1.02  # 2% oversized
                pig_length = pipe_diameter * 1.5
                bypass_flow = 0.03  # 3%
            elif pig_type == "Cup":
                pig_diameter = pipe_diameter * 1.03  # 3% oversized
                pig_length = pipe_diameter * 2.5
                bypass_flow = 0.02  # 2%
            elif pig_type == "Sphere":
                pig_diameter = pipe_diameter * 1.04  # 4% oversized
                pig_length = pipe_diameter  # Same as diameter for sphere
                bypass_flow = 0.05  # 5%
            elif pig_type == "Intelligent":
                pig_diameter = pipe_diameter * 0.98  # 2% undersized
                pig_length = pipe_diameter * 4.0
                bypass_flow = 0.01  # 1%
            elif pig_type == "Gel":
                pig_diameter = pipe_diameter  # Same as pipeline
                pig_length = pipe_diameter * 3.0
                bypass_flow = 0.005  # 0.5%
            else:
                # Default for unknown types
                pig_diameter = pipe_diameter * 1.03
                pig_length = pipe_diameter * 2.0
                bypass_flow = 0.05  # 5%
            
            # Store pigging properties
            self.pigging_properties = {
                "pipe_diameter": pipe_diameter,
                "pig_type": pig_type,
                "pig_diameter": pig_diameter,
                "pig_length": pig_length,
                "pig_density": pig_density_map.get(pig_type, 300.0),
                "pig_friction": pig_friction_map.get(pig_type, 0.3),
                "bypass_flow": bypass_flow,
                "driving_fluid_type": driving_fluid_type,
                "initial_position": 0.0,  # At the start of the pipeline
                "travel_distance": 100.0,  # Default pipeline length
                "pig_velocity": 0.0,  # Initial velocity
                "differential_pressure": 0.0  # Initial differential pressure
            }
            
            # Set up default boundary conditions
            inlet_velocity = 1.0  # Default
            self._setup_default_boundaries(inlet_velocity, None, None, is_multiphase=True)
            
            # Set simulation parameters appropriate for pigging
            self.reference_pressure = 101325.0  # 1 atm
            self.turbulence_model = "k-epsilon"
            self.gravity = [0.0, 0.0, -9.81]
            
            # Appropriate numerical settings for pigging
            self.time_step = 0.001
            self.adjustable_time_step = True
            self.max_courant_number = 0.5
            self.convergence_tolerance = 1e-6
            
            # Update metadata
            self.modified_date = datetime.datetime.now().isoformat()
            
            logger.info(f"Set up pigging simulation with {pig_type} pig")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up pigging simulation: {e}")
            return False
    
    def setup_spill_simulation(self, spill_type: str = "Surface",
                              fluid_type: str = "Crude Oil",
                              environment: str = "Water (Ocean)") -> bool:
        """
        Set up a spill simulation.
        
        Args:
            spill_type (str): Type of spill ("Surface", "Subsurface", "Jet")
            fluid_type (str): Type of spilled fluid ("Crude Oil", "Diesel", etc.)
            environment (str): Environment type ("Water (Ocean)", "Land (Soil)", etc.)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Set flow type
            self.flow_type = "spill"
            self.time_dependent = True
            
            # Clear existing phases and interfaces
            self.phases = {}
            self.interfaces = []
            
            # Determine fluid properties
            if fluid_type.lower() == "crude oil":
                spill_fluid_type = "crude_oil"
            elif fluid_type.lower() == "diesel":
                spill_fluid_type = "diesel"
            elif fluid_type.lower() == "gasoline":
                spill_fluid_type = "gasoline"
            else:
                spill_fluid_type = "oil"  # Default
            
            # Determine environment properties
            if environment.startswith("Water"):
                environment_fluid_type = "water"
            else:
                environment_fluid_type = "air"  # For land environments
            
            # Get properties for each phase
            spill_props = self._get_default_fluid_properties(spill_fluid_type)
            env_props = self._get_default_fluid_properties(environment_fluid_type)
            
            # Add the phases
            self.phases["spill_fluid"] = PhaseProperties(
                name="spill_fluid",
                fluid_type=spill_fluid_type,
                density=spill_props["density"],
                viscosity=spill_props["viscosity"],
                volume_fraction=0.1,  # Initial concentration
                specific_heat=spill_props.get("specific_heat"),
                thermal_conductivity=spill_props.get("thermal_conductivity")
            )
            
            self.phases["environment"] = PhaseProperties(
                name="environment",
                fluid_type=environment_fluid_type,
                density=env_props["density"],
                viscosity=env_props["viscosity"],
                volume_fraction=0.9,  # Initial concentration
                specific_heat=env_props.get("specific_heat"),
                thermal_conductivity=env_props.get("thermal_conductivity")
            )
            
            # Add interface properties
            surface_tension = 0.025  # Default oil-water
            if environment_fluid_type == "air":
                surface_tension = 0.023  # Oil-air
            
            self.interfaces.append(InterfaceProperties(
                phase1="spill_fluid",
                phase2="environment",
                surface_tension=surface_tension,
                contact_angle=90.0  # Neutral wetting
            ))
            
            # Set up spill properties
            self.spill_properties = {
                "spill_type": spill_type,
                "fluid_type": fluid_type,
                "environment": environment,
                "spill_rate": 10.0,  # kg/s
                "total_mass": 1000.0,  # kg
                "duration": 100.0,  # s
                "opening_diameter": 0.05,  # m
                "temperature": 293.15,  # K
                "pressure": 500000.0,  # Pa
                "wind_speed": 5.0,  # m/s
                "current_speed": 0.5,  # m/s
                "domain_size": 1000.0,  # m
                "weathering": True,
                "dispersion": True,
                "evaporation": True
            }
            
            # Set up default boundary conditions for spill
            self._setup_spill_boundaries(spill_type, environment.startswith("Water"))
            
            # Set simulation parameters appropriate for spill modeling
            self.reference_pressure = 101325.0  # 1 atm
            self.reference_temperature = 293.15  # 20°C
            
            if environment.startswith("Water"):
                self.turbulence_model = "k-epsilon"
            else:
                self.turbulence_model = "k-omega"  # Better for atmospheric flows
            
            self.gravity = [0.0, 0.0, -9.81]
            self.buoyancy = True
            
            # Enable heat transfer for evaporation modeling
            if self.spill_properties["evaporation"]:
                self.heat_transfer = True
            
            # Numerical settings for spill simulation
            self.time_step = 0.05  # Larger time step for environmental scale
            self.adjustable_time_step = True
            self.max_courant_number = 0.8
            self.max_time = 86400.0  # 24 hours in seconds
            self.write_interval = 600.0  # Every 10 minutes
            
            # Update metadata
            self.modified_date = datetime.datetime.now().isoformat()
            
            logger.info(f"Set up spill simulation with {fluid_type} in {environment}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up spill simulation: {e}")
            return False
    
    def add_boundary_condition(self, name: str, type_name: str, 
                              velocity: List[float] = None,
                              pressure: float = None,
                              temperature: float = None,
                              volume_fractions: Dict[str, float] = None) -> bool:
        """
        Add a boundary condition to the flow model.
        
        Args:
            name (str): Boundary name
            type_name (str): Type of boundary ("inlet", "outlet", "wall", etc.)
            velocity (List[float], optional): Velocity vector [u, v, w]
            pressure (float, optional): Pressure value
            temperature (float, optional): Temperature value
            volume_fractions (Dict[str, float], optional): Volume fractions for each phase
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create the boundary condition
            bc = BoundaryCondition(
                name=name,
                type_name=type_name,
                velocity=velocity,
                pressure=pressure,
                temperature=temperature,
                volume_fractions=volume_fractions
            )
            
            # Add to the dictionary
            self.boundary_conditions[name] = bc
            
            # Update metadata
            self.modified_date = datetime.datetime.now().isoformat()
            
            logger.info(f"Added boundary condition: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding boundary condition: {e}")
            return False
    
    def set_turbulence_model(self, model_name: str) -> bool:
        """
        Set the turbulence model for the flow simulation.
        
        Args:
            model_name (str): Name of the turbulence model
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            valid_models = [
                "laminar", 
                "k-epsilon", 
                "k-omega", 
                "SpalartAllmaras", 
                "LES", 
                "RNG k-epsilon", 
                "Realizable k-epsilon"
            ]
            
            if model_name not in valid_models:
                logger.warning(f"Unknown turbulence model: {model_name}. Using k-epsilon instead.")
                model_name = "k-epsilon"
            
            self.turbulence_model = model_name
            
            # Update metadata
            self.modified_date = datetime.datetime.now().isoformat()
            
            logger.info(f"Set turbulence model to {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting turbulence model: {e}")
            return False
    
    def set_initial_conditions(self, field_name: str, value: Any) -> bool:
        """
        Set an initial condition for a field.
        
        Args:
            field_name (str): Name of the field
            value (Any): Value to set
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.initial_conditions[field_name] = value
            
            # Update metadata
            self.modified_date = datetime.datetime.now().isoformat()
            
            logger.info(f"Set initial condition for {field_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting initial condition: {e}")
            return False
    
    def load_results(self, results_dir: str) -> bool:
        """
        Load simulation results from a directory.
        
        Args:
            results_dir (str): Path to the results directory
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if the directory exists
            if not os.path.exists(results_dir):
                logger.error(f"Results directory not found: {results_dir}")
                return False
            
            # Find result files
            result_files = []
            for root, _, files in os.walk(results_dir):
                for file in files:
                    if file.endswith('.vtk') or file.endswith('.vtu') or file.endswith('.foam'):
                        result_files.append(os.path.join(root, file))
            
            if not result_files:
                logger.error(f"No result files found in {results_dir}")
                return False
            
            # Extract time steps and field information
            # (This would be a more complex implementation in practice)
            # Here we're just setting placeholder values
            
            # Placeholder for extracted times
            times = []
            for i in range(10):  # Assuming 10 time steps
                times.append(i * 0.1)  # 0.1s intervals
            
            # Placeholder for field names
            field_names = ["p", "U", "k", "epsilon"]
            if len(self.phases) > 1:
                field_names.append("alpha.water")
                field_names.append("alpha.oil")
            
            # Placeholder for residuals
            residuals = {}
            for field in ["p", "Ux", "Uy", "Uz", "k", "epsilon"]:
                residuals[field] = [1e-2 * (0.5 ** i) for i in range(10)]  # Decreasing values
            
            # Create results object
            self.results = FlowResults(
                num_timesteps=len(times),
                times=times,
                field_names=field_names,
                converged=True,  # Placeholder
                residuals=residuals,
                result_files=result_files
            )
            
            # Update metadata
            self.modified_date = datetime.datetime.now().isoformat()
            
            logger.info(f"Loaded results from {results_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return False
    
    def get_field_range(self, field_name: str, time_step: int = -1) -> Tuple[float, float]:
        """
        Get the range of values for a field at a specific time step.
        
        Args:
            field_name (str): Name of the field
            time_step (int): Time step index (-1 for latest)
            
        Returns:
            Tuple[float, float]: Min and max values
        """
        if not self.results.result_files:
            logger.error("No results available")
            return (0.0, 1.0)  # Default range
        
        try:
            # In a real implementation, this would load data from the result files
            # and extract the actual range for the field
            
            # Return placeholder ranges based on field name
            if field_name == "p":
                return (0.0, 200000.0)  # Pressure range in Pa
            elif field_name.startswith("U"):
                return (0.0, 10.0)  # Velocity range in m/s
            elif field_name == "k":
                return (0.0, 1.0)  # Turbulent kinetic energy
            elif field_name == "epsilon":
                return (0.0, 1.0)  # Turbulent dissipation rate
            elif field_name.startswith("alpha."):
                return (0.0, 1.0)  # Phase fraction range
            elif field_name == "T":
                return (273.15, 323.15)  # Temperature range
            else:
                return (0.0, 1.0)  # Default range
            
        except Exception as e:
            logger.error(f"Error getting field range: {e}")
            return (0.0, 1.0)  # Default range
    
    def analyze_flow_pattern(self) -> Dict[str, Any]:
        """
        Analyze the flow pattern in multiphase flow.
        
        Returns:
            Dict[str, Any]: Flow pattern analysis results
        """
        if self.flow_type not in ["two_phase", "three_phase"]:
            logger.error("Flow pattern analysis is only applicable to multiphase flows")
            return {}
        
        if not self.results.result_files:
            logger.error("No results available for flow pattern analysis")
            return {}
        
        try:
            # In a real implementation, this would analyze the flow pattern
            # from the simulation results using appropriate algorithms
            
            # Return placeholder analysis
            if self.flow_type == "two_phase":
                if list(self.phases.values())[0].fluid_type in ["water", "oil"] and \
                   list(self.phases.values())[1].fluid_type in ["water", "oil"]:
                    # Oil-water flow
                    return {
                        "pattern": "stratified",
                        "primary_phase": "oil",
                        "secondary_phase": "water",
                        "interface_area": 0.5,
                        "water_holdup": 0.4
                    }
                else:
                    # Gas-liquid flow
                    return {
                        "pattern": "slug",
                        "primary_phase": "gas",
                        "secondary_phase": "liquid",
                        "slug_frequency": 0.5,
                        "slug_length": 0.2,
                        "gas_void_fraction": 0.6
                    }
            else:  # three_phase
                return {
                    "pattern": "stratified",
                    "top_phase": "gas",
                    "middle_phase": "oil",
                    "bottom_phase": "water",
                    "water_holdup": 0.3,
                    "oil_holdup": 0.3,
                    "gas_void_fraction": 0.4
                }
            
        except Exception as e:
            logger.error(f"Error analyzing flow pattern: {e}")
            return {}
    
    def analyze_pigging_performance(self) -> Dict[str, Any]:
        """
        Analyze the performance of a pigging operation.
        
        Returns:
            Dict[str, Any]: Pigging performance analysis results
        """
        if self.flow_type != "pigging":
            logger.error("Pigging performance analysis is only applicable to pigging simulations")
            return {}
        
        if not self.results.result_files:
            logger.error("No results available for pigging performance analysis")
            return {}
        
        try:
            # In a real implementation, this would analyze the pigging performance
            # from the simulation results using appropriate algorithms
            
            # Return placeholder analysis
            return {
                "pig_type": self.pigging_properties["pig_type"],
                "average_velocity": 0.8,  # m/s
                "cleaning_efficiency": 0.85,  # 85%
                "pressure_drop": 200000.0,  # 2 bar
                "bypass_flow_rate": self.pigging_properties["bypass_flow"] * 100,  # %
                "travel_distance": 95.0,  # m
                "travel_time": 120.0,  # s
                "removed_debris": 50.0,  # kg
                "efficiency_rating": "Good"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing pigging performance: {e}")
            return {}
    
    def analyze_spill_impact(self) -> Dict[str, Any]:
        """
        Analyze the impact of a spill.
        
        Returns:
            Dict[str, Any]: Spill impact analysis results
        """
        if self.flow_type != "spill":
            logger.error("Spill impact analysis is only applicable to spill simulations")
            return {}
        
        if not self.results.result_files:
            logger.error("No results available for spill impact analysis")
            return {}
        
        try:
            # In a real implementation, this would analyze the spill impact
            # from the simulation results using appropriate algorithms
            
            # Return placeholder analysis
            environment = self.spill_properties["environment"]
            fluid_type = self.spill_properties["fluid_type"]
            spill_type = self.spill_properties["spill_type"]
            
            if environment.startswith("Water"):
                return {
                    "spill_type": spill_type,
                    "fluid_type": fluid_type,
                    "environment": environment,
                    "affected_area": 15000.0,  # m²
                    "maximum_spread": 150.0,  # m
                    "thickness_range": [0.001, 0.01],  # m
                    "evaporated_fraction": 0.15,
                    "dissolved_fraction": 0.05,
                    "dispersed_fraction": 0.2,
                    "settled_fraction": 0.1,
                    "remaining_fraction": 0.5,
                    "shore_arrival_time": 8.5,  # hours
                    "shore_impact_length": 500.0,  # m
                    "impact_severity": "Moderate"
                }
            else:  # Land
                return {
                    "spill_type": spill_type,
                    "fluid_type": fluid_type,
                    "environment": environment,
                    "affected_area": 5000.0,  # m²
                    "maximum_spread": 80.0,  # m
                    "penetration_depth": 0.3,  # m
                    "evaporated_fraction": 0.2,
                    "infiltrated_fraction": 0.4,
                    "remaining_fraction": 0.4,
                    "groundwater_risk": "Medium",
                    "cleanup_difficulty": "High",
                    "impact_severity": "Moderate"
                }
            
        except Exception as e:
            logger.error(f"Error analyzing spill impact: {e}")
            return {}
    
    def _setup_default_boundaries(self, inlet_velocity: float = 1.0,
                                inlet_pressure: float = None,
                                outlet_pressure: float = None,
                                is_multiphase: bool = False):
        """
        Set up default boundary conditions.
        
        Args:
            inlet_velocity (float): Inlet velocity
            inlet_pressure (float): Inlet pressure
            outlet_pressure (float): Outlet pressure
            is_multiphase (bool): Whether the flow is multiphase
        """
        # Clear existing boundary conditions
        self.boundary_conditions = {}
        
        # Set default values if not provided
        if inlet_pressure is None:
            inlet_pressure = self.reference_pressure
        if outlet_pressure is None:
            outlet_pressure = self.reference_pressure
        
        # Set up standard boundaries
        if is_multiphase:
            # Volume fractions for each phase
            volume_fractions = {}
            for name, phase in self.phases.items():
                volume_fractions[name] = phase.volume_fraction
            
            # Inlet
            self.add_boundary_condition(
                name="inlet",
                type_name="inlet",
                velocity=[inlet_velocity, 0.0, 0.0],
                pressure=inlet_pressure,
                temperature=self.reference_temperature,
                volume_fractions=volume_fractions
            )
            
            # Outlet
            self.add_boundary_condition(
                name="outlet",
                type_name="outlet",
                pressure=outlet_pressure
            )
            
            # Walls
            self.add_boundary_condition(
                name="walls",
                type_name="wall",
                velocity=[0.0, 0.0, 0.0]  # No-slip condition
            )
            
        else:
            # Single phase boundaries
            
            # Inlet
            self.add_boundary_condition(
                name="inlet",
                type_name="inlet",
                velocity=[inlet_velocity, 0.0, 0.0],
                pressure=inlet_pressure,
                temperature=self.reference_temperature
            )
            
            # Outlet
            self.add_boundary_condition(
                name="outlet",
                type_name="outlet",
                pressure=outlet_pressure
            )
            
            # Walls
            self.add_boundary_condition(
                name="walls",
                type_name="wall",
                velocity=[0.0, 0.0, 0.0]  # No-slip condition
            )
    
    def _setup_spill_boundaries(self, spill_type: str, is_water_environment: bool):
        """
        Set up boundary conditions for a spill simulation.
        
        Args:
            spill_type (str): Type of spill
            is_water_environment (bool): Whether the environment is water
        """
        # Clear existing boundary conditions
        self.boundary_conditions = {}
        
        # Volume fractions for the spill source
        volume_fractions = {
            "spill_fluid": 1.0,
            "environment": 0.0
        }
        
        # Set up boundaries based on spill type
        if spill_type == "Surface":
            # Surface spill - patch on top boundary
            self.add_boundary_condition(
                name="spill_source",
                type_name="patch",
                velocity=[0.0, 0.0, 0.0],  # Initial velocity
                temperature=self.reference_temperature,
                volume_fractions=volume_fractions
            )
            
            # Environment boundary conditions
            if is_water_environment:
                # Water current
                self.add_boundary_condition(
                    name="water_inlet",
                    type_name="inlet",
                    velocity=[0.5, 0.0, 0.0],  # Current velocity
                    temperature=288.15  # 15°C
                )
            else:
                # Air/wind
                self.add_boundary_condition(
                    name="air_inlet",
                    type_name="inlet",
                    velocity=[5.0, 0.0, 0.0],  # Wind velocity
                    temperature=293.15  # 20°C
                )
            
        elif spill_type == "Subsurface":
            # Subsurface spill - source at depth
            self.add_boundary_condition(
                name="spill_source",
                type_name="patch",
                velocity=[0.0, 0.0, 2.0],  # Initial upward velocity
                temperature=self.reference_temperature,
                volume_fractions=volume_fractions
            )
            
            # Environment boundary conditions
            if is_water_environment:
                # Water current
                self.add_boundary_condition(
                    name="water_inlet",
                    type_name="inlet",
                    velocity=[0.5, 0.0, 0.0],  # Current velocity
                    temperature=288.15  # 15°C
                )
            
        elif spill_type == "Jet":
            # Jet spill - high velocity source
            self.add_boundary_condition(
                name="spill_source",
                type_name="patch",
                velocity=[5.0, 0.0, 0.0],  # Initial jet velocity
                pressure=500000.0,  # 5 bar
                temperature=self.reference_temperature,
                volume_fractions=volume_fractions
            )
            
            # Environment boundary conditions
            if is_water_environment:
                # Water current
                self.add_boundary_condition(
                    name="water_inlet",
                    type_name="inlet",
                    velocity=[0.5, 0.0, 0.0],  # Current velocity
                    temperature=288.15  # 15°C
                )
            else:
                # Air/wind
                self.add_boundary_condition(
                    name="air_inlet",
                    type_name="inlet",
                    velocity=[5.0, 0.0, 0.0],  # Wind velocity
                    temperature=293.15  # 20°C
                )
        
        # Common boundaries
        
        # Outlet/pressure boundary
        self.add_boundary_condition(
            name="outlet",
            type_name="outlet",
            pressure=self.reference_pressure
        )
        
        # Bottom boundary (ground or seabed)
        self.add_boundary_condition(
            name="bottom",
            type_name="wall",
            velocity=[0.0, 0.0, 0.0]  # No-slip condition
        )
        
        # Side boundaries
        self.add_boundary_condition(
            name="sides",
            type_name="patch",  # or "wall" depending on simulation needs
            velocity=[0.0, 0.0, 0.0]
        )
        
        # Top boundary (atmosphere)
        self.add_boundary_condition(
            name="top",
            type_name="patch",
            pressure=self.reference_pressure
        )
    
    def _get_default_fluid_properties(self, fluid_type: str) -> Dict[str, Any]:
        """
        Get default properties for a fluid type.
        
        Args:
            fluid_type (str): Type of fluid
            
        Returns:
            Dict[str, Any]: Dictionary of fluid properties
        """
        properties = {
            # Default to water if unknown fluid type
            "density": 1000.0,        # kg/m³
            "viscosity": 0.001,       # Pa·s
            "specific_heat": 4182.0,  # J/(kg·K)
            "thermal_conductivity": 0.6  # W/(m·K)
        }
        
        # Set specific fluid properties
        if fluid_type == "water":
            pass  # Already set to water
        
        elif fluid_type == "oil":
            properties["density"] = 850.0
            properties["viscosity"] = 0.03
            properties["specific_heat"] = 1800.0
            properties["thermal_conductivity"] = 0.15
        
        elif fluid_type == "crude_oil":
            properties["density"] = 900.0
            properties["viscosity"] = 0.05
            properties["specific_heat"] = 1900.0
            properties["thermal_conductivity"] = 0.12
        
        elif fluid_type == "gas" or fluid_type == "natural_gas":
            properties["density"] = 0.8
            properties["viscosity"] = 1.8e-5
            properties["specific_heat"] = 2200.0
            properties["thermal_conductivity"] = 0.026
        
        elif fluid_type == "air":
            properties["density"] = 1.2
            properties["viscosity"] = 1.8e-5
            properties["specific_heat"] = 1005.0
            properties["thermal_conductivity"] = 0.025
        
        elif fluid_type == "diesel":
            properties["density"] = 830.0
            properties["viscosity"] = 0.0024
            properties["specific_heat"] = 2100.0
            properties["thermal_conductivity"] = 0.12
        
        elif fluid_type == "gasoline":
            properties["density"] = 750.0
            properties["viscosity"] = 0.0006
            properties["specific_heat"] = 2000.0
            properties["thermal_conductivity"] = 0.11
        
        return properties
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the flow model to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the flow model
        """
        # Convert phases and interfaces to dictionaries
        phases_dict = {}
        for name, phase in self.phases.items():
            phases_dict[name] = phase.to_dict()
        
        interfaces_list = []
        for interface in self.interfaces:
            interfaces_list.append(interface.to_dict())
        
        # Convert boundary conditions to dictionaries
        boundary_dict = {}
        for name, bc in self.boundary_conditions.items():
            boundary_dict[name] = bc.to_dict()
        
        # Create dictionary
        return {
            "name": self.name,
            "description": self.description,
            "created_date": self.created_date,
            "modified_date": self.modified_date,
            "mesh_path": self.mesh_path,
            "case_dir": self.case_dir,
            "flow_type": self.flow_type,
            "flow_regime": self.flow_regime,
            "time_dependent": self.time_dependent,
            "reference_pressure": self.reference_pressure,
            "reference_temperature": self.reference_temperature,
            "gravity": self.gravity,
            "turbulence_model": self.turbulence_model,
            "turbulence_wall_function": self.turbulence_wall_function,
            "heat_transfer": self.heat_transfer,
            "buoyancy": self.buoyancy,
            "phases": phases_dict,
            "interfaces": interfaces_list,
            "boundary_conditions": boundary_dict,
            "initial_conditions": self.initial_conditions,
            "max_time": self.max_time,
            "time_step": self.time_step,
            "adjustable_time_step": self.adjustable_time_step,
            "max_courant_number": self.max_courant_number,
            "write_interval": self.write_interval,
            "convergence_tolerance": self.convergence_tolerance,
            "max_iterations": self.max_iterations,
            "relaxation_factors": self.relaxation_factors,
            "pigging_properties": self.pigging_properties,
            "spill_properties": self.spill_properties,
            "results": self.results.to_dict(),
            "parallel": self.parallel,
            "n_processors": self.n_processors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FlowModel':
        """
        Create a flow model from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary data
            
        Returns:
            FlowModel: New flow model
        """
        flow_model = cls(
            name=data.get("name", "Untitled Flow"),
            mesh_path=data.get("mesh_path"),
            case_dir=data.get("case_dir")
        )
        
        # Set basic properties
        flow_model.description = data.get("description", "")
        flow_model.created_date = data.get("created_date", flow_model.created_date)
        flow_model.modified_date = data.get("modified_date", flow_model.modified_date)
        flow_model.flow_type = data.get("flow_type", "single_phase")
        flow_model.flow_regime = data.get("flow_regime", "turbulent")
        flow_model.time_dependent = data.get("time_dependent", False)
        flow_model.reference_pressure = data.get("reference_pressure", 101325.0)
        flow_model.reference_temperature = data.get("reference_temperature", 293.15)
        flow_model.gravity = data.get("gravity", [0.0, 0.0, -9.81])
        flow_model.turbulence_model = data.get("turbulence_model", "k-epsilon")
        flow_model.turbulence_wall_function = data.get("turbulence_wall_function", "standard")
        flow_model.heat_transfer = data.get("heat_transfer", False)
        flow_model.buoyancy = data.get("buoyancy", False)
        
        # Load phases
        phases_dict = data.get("phases", {})
        for name, phase_data in phases_dict.items():
            flow_model.phases[name] = PhaseProperties.from_dict(phase_data)
        
        # Load interfaces
        interfaces_list = data.get("interfaces", [])
        for interface_data in interfaces_list:
            flow_model.interfaces.append(InterfaceProperties.from_dict(interface_data))
        
        # Load boundary conditions
        boundary_dict = data.get("boundary_conditions", {})
        for name, bc_data in boundary_dict.items():
            flow_model.boundary_conditions[name] = BoundaryCondition.from_dict(bc_data)
        
        # Set other properties
        flow_model.initial_conditions = data.get("initial_conditions", {})
        flow_model.max_time = data.get("max_time", 1000.0)
        flow_model.time_step = data.get("time_step", 0.1)
        flow_model.adjustable_time_step = data.get("adjustable_time_step", False)
        flow_model.max_courant_number = data.get("max_courant_number", 1.0)
        flow_model.write_interval = data.get("write_interval", 10.0)
        flow_model.convergence_tolerance = data.get("convergence_tolerance", 1e-5)
        flow_model.max_iterations = data.get("max_iterations", 1000)
        flow_model.relaxation_factors = data.get("relaxation_factors", flow_model.relaxation_factors)
        flow_model.pigging_properties = data.get("pigging_properties", {})
        flow_model.spill_properties = data.get("spill_properties", {})
        
        # Load results
        results_data = data.get("results", {})
        if results_data:
            flow_model.results = FlowResults.from_dict(results_data)
        
        # Set parallel settings
        flow_model.parallel = data.get("parallel", False)
        flow_model.n_processors = data.get("n_processors", 4)
        
        return flow_model
    
    def save(self, filepath: str) -> bool:
        """
        Save the flow model to a JSON file.
        
        Args:
            filepath (str): Path to save the file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert to dictionary
            data = self.to_dict()
            
            # Save to JSON
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved flow model to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving flow model: {e}")
            return False
    
    @classmethod
    def load(cls, filepath: str) -> 'FlowModel':
        """
        Load a flow model from a JSON file.
        
        Args:
            filepath (str): Path to the file
            
        Returns:
            FlowModel: Loaded flow model
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file is invalid JSON
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Flow model file not found: {filepath}")
        
        try:
            # Load from JSON
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Create from dictionary
            flow_model = cls.from_dict(data)
            
            logger.info(f"Loaded flow model from {filepath}")
            return flow_model
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing flow model file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading flow model: {e}")
            raise ValueError(f"Failed to load flow model: {str(e)}")