#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pigging simulation models for Openfoam_Simulator.

This module provides physical models and calculations for simulating pipeline 
pigging operations in oil & gas applications. It includes models for different
types of pigs, their dynamics in pipelines, and associated physics for:
- Pig motion dynamics
- Fluid-pig interaction
- Cleaning efficiency calculation
- Differential pressure modeling
- Bypass flow calculations
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Import logging functionality
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BasePig:
    """
    Base class for pig simulation models.
    
    This serves as the parent class for all specific pig types and provides
    common attributes and methods for pig simulation.
    """
    
    def __init__(self, 
                 diameter: float, 
                 length: float, 
                 density: float, 
                 friction_coef: float,
                 bypass_fraction: float,
                 pipeline_diameter: float):
        """
        Initialize the base pig model.
        
        Args:
            diameter (float): Pig diameter in meters
            length (float): Pig length in meters
            density (float): Pig material density in kg/m³
            friction_coef (float): Friction coefficient between pig and pipe wall
            bypass_fraction (float): Fraction of flow that bypasses the pig (0-1)
            pipeline_diameter (float): Pipeline inner diameter in meters
        """
        self.diameter = diameter
        self.length = length
        self.density = density
        self.friction_coef = friction_coef
        self.bypass_fraction = bypass_fraction
        self.pipeline_diameter = pipeline_diameter
        
        # Derived properties
        self.cross_section = math.pi * (diameter / 2) ** 2  # m²
        self.volume = self.cross_section * length  # m³
        self.mass = self.volume * density  # kg
        self.pipeline_area = math.pi * (pipeline_diameter / 2) ** 2  # m²
        self.sealing_efficiency = 1.0 - bypass_fraction
        
        # State variables
        self.position = 0.0  # Position along pipeline (m)
        self.velocity = 0.0  # Pig velocity (m/s)
        self.acceleration = 0.0  # Pig acceleration (m/s²)
        self.differential_pressure = 0.0  # Pressure difference across pig (Pa)
        self.force_balance = {}  # Dict of forces acting on pig (N)
        
        # Initialize logging
        logger.info(f"Initialized {self.__class__.__name__} with diameter={diameter}m, length={length}m")
    
    def update_position(self, time_step: float):
        """
        Update pig position based on current velocity and time step.
        
        Args:
            time_step (float): Simulation time step in seconds
            
        Returns:
            float: New position in meters
        """
        self.position += self.velocity * time_step
        return self.position
    
    def calculate_velocity(self, flow_rate: float) -> float:
        """
        Calculate pig velocity based on flow rate and bypass.
        
        Args:
            flow_rate (float): Total flow rate in m³/s
            
        Returns:
            float: Pig velocity in m/s
        """
        # Effective flow rate (accounting for bypass)
        effective_flow_rate = flow_rate * (1.0 - self.bypass_fraction)
        
        # Velocity = Flow rate / Pipeline cross-sectional area
        self.velocity = effective_flow_rate / self.pipeline_area
        
        return self.velocity
    
    def calculate_differential_pressure(self, 
                                      fluid_density: float, 
                                      fluid_viscosity: float,
                                      flow_rate: float,
                                      pipeline_roughness: float) -> float:
        """
        Calculate pressure differential across the pig.
        
        Args:
            fluid_density (float): Density of driving fluid in kg/m³
            fluid_viscosity (float): Viscosity of driving fluid in Pa·s
            flow_rate (float): Flow rate in m³/s
            pipeline_roughness (float): Pipeline roughness in m
            
        Returns:
            float: Pressure differential in Pa
        """
        # Reynolds number for the flow
        fluid_velocity = flow_rate / self.pipeline_area
        reynolds = (fluid_density * fluid_velocity * self.pipeline_diameter) / fluid_viscosity
        
        # Friction factor (Colebrook-White equation approximation)
        if reynolds > 4000:  # Turbulent flow
            relative_roughness = pipeline_roughness / self.pipeline_diameter
            # Haaland equation (approximation of Colebrook-White)
            friction_factor = (-1.8 * math.log10((relative_roughness/3.7)**1.11 + 6.9/reynolds))**(-2)
        else:  # Laminar flow
            friction_factor = 64 / reynolds
        
        # Friction pressure loss in the pipe
        pipe_friction_loss = (friction_factor * fluid_density * fluid_velocity**2 * self.length) / (2 * self.pipeline_diameter)
        
        # Pressure required to overcome pig friction with pipe wall
        wall_friction_force = self.friction_coef * self.mass * 9.81  # F = μ·m·g
        wall_friction_pressure = wall_friction_force / self.cross_section
        
        # Total differential pressure
        self.differential_pressure = pipe_friction_loss + wall_friction_pressure
        
        # Force balance
        self.force_balance = {
            "flow_force": self.differential_pressure * self.cross_section,
            "wall_friction": -wall_friction_force,
            "net_force": (self.differential_pressure * self.cross_section) - wall_friction_force
        }
        
        # Calculate acceleration
        self.acceleration = self.force_balance["net_force"] / self.mass
        
        return self.differential_pressure
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current state of the pig.
        
        Returns:
            Dict[str, Any]: Dictionary with pig state variables
        """
        return {
            "position": self.position,
            "velocity": self.velocity, 
            "acceleration": self.acceleration,
            "differential_pressure": self.differential_pressure,
            "force_balance": self.force_balance
        }
    
    def get_properties(self) -> Dict[str, Any]:
        """
        Get physical properties of the pig.
        
        Returns:
            Dict[str, Any]: Dictionary with pig properties
        """
        return {
            "diameter": self.diameter,
            "length": self.length,
            "density": self.density,
            "friction_coef": self.friction_coef,
            "bypass_fraction": self.bypass_fraction,
            "cross_section": self.cross_section,
            "volume": self.volume,
            "mass": self.mass,
            "sealing_efficiency": self.sealing_efficiency
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert pig model to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Serializable dictionary of pig properties
        """
        # Combine properties and state
        result = {
            "type": self.__class__.__name__,
            "properties": self.get_properties(),
            "state": self.get_state()
        }
        return result


class FoamPig(BasePig):
    """
    Foam pig model for pipeline cleaning operations.
    
    Foam pigs are commonly used for cleaning, batching, or as proving pigs.
    They have medium sealing efficiency and medium to high friction.
    """
    
    def __init__(self, 
                 diameter: float, 
                 length: float, 
                 density: float = 250.0, 
                 friction_coef: float = 0.3,
                 bypass_fraction: float = 0.08,
                 pipeline_diameter: float = 0.1016):  # Default 4 inch pipe
        """
        Initialize a foam pig model.
        
        Args:
            diameter (float): Pig diameter in meters (typically 1.05 * pipeline diameter)
            length (float): Pig length in meters
            density (float): Pig material density in kg/m³, default 250 kg/m³
            friction_coef (float): Friction coefficient between pig and pipe wall
            bypass_fraction (float): Fraction of flow that bypasses the pig (0-1)
            pipeline_diameter (float): Pipeline inner diameter in meters
        """
        super().__init__(diameter, length, density, friction_coef, bypass_fraction, pipeline_diameter)
        
        # Foam-specific properties
        self.compressibility = 5e-7  # 1/Pa, foam compressibility
        self.wear_rate = 1e-6  # m/m, wear rate per meter traveled
        
        # Initialize with some wearing, since foam pigs are not perfectly new
        self.wear_factor = 0.02  # 2% initial wear
    
    def calculate_cleaning_efficiency(self, 
                                     debris_density: float = 800.0,
                                     pipe_wall_condition: float = 0.8) -> float:
        """
        Calculate cleaning efficiency of the foam pig.
        
        Args:
            debris_density (float): Density of debris in kg/m³
            pipe_wall_condition (float): Pipe wall condition factor (0-1, 1 is clean)
            
        Returns:
            float: Cleaning efficiency (0-1)
        """
        # Basic model for cleaning efficiency based on:
        # - Sealing efficiency (how well pig contacts pipe wall)
        # - Wear factor (higher wear reduces cleaning)
        # - Pig velocity (moderate velocities are better for cleaning)
        # - Pipe wall condition (harder to clean dirty pipes)
        
        # Velocity factor (peaks at 1.5 m/s)
        vel_factor = 1.0 - 0.5 * abs(self.velocity - 1.5) / 1.5
        vel_factor = max(0.3, min(1.0, vel_factor))
        
        # Calculated cleaning efficiency
        efficiency = (
            self.sealing_efficiency * 
            (1.0 - self.wear_factor) * 
            vel_factor * 
            pipe_wall_condition
        )
        
        return max(0.0, min(1.0, efficiency))
    
    def update_wear(self, distance_traveled: float):
        """
        Update wear factor based on distance traveled.
        
        Args:
            distance_traveled (float): Distance traveled in meters
            
        Returns:
            float: Updated wear factor
        """
        self.wear_factor += self.wear_rate * distance_traveled
        self.wear_factor = min(1.0, self.wear_factor)  # Cap at 100% wear
        
        # Update diameter due to wear
        diameter_reduction = self.wear_factor * 0.05 * self.diameter  # Up to 5% reduction
        self.diameter -= diameter_reduction
        
        # Update bypass fraction as wear increases
        self.bypass_fraction += 0.5 * self.wear_factor  # Bypass increases with wear
        self.bypass_fraction = min(0.95, self.bypass_fraction)  # Max 95% bypass
        
        # Update other properties
        self.cross_section = math.pi * (self.diameter / 2) ** 2
        self.volume = self.cross_section * self.length
        self.mass = self.volume * self.density
        self.sealing_efficiency = 1.0 - self.bypass_fraction
        
        return self.wear_factor


class CupPig(BasePig):
    """
    Cup pig model for pipeline separation and wiping operations.
    
    Cup pigs have cups or discs that provide better sealing than foam pigs.
    They are used for product separation, liquid removal, and more effective cleaning.
    """
    
    def __init__(self, 
                 diameter: float, 
                 length: float, 
                 density: float = 900.0, 
                 friction_coef: float = 0.35,
                 bypass_fraction: float = 0.02,
                 pipeline_diameter: float = 0.1016,
                 num_cups: int = 3):
        """
        Initialize a cup pig model.
        
        Args:
            diameter (float): Pig diameter in meters
            length (float): Pig length in meters
            density (float): Pig material density in kg/m³
            friction_coef (float): Friction coefficient between pig and pipe wall
            bypass_fraction (float): Fraction of flow that bypasses the pig (0-1)
            pipeline_diameter (float): Pipeline inner diameter in meters
            num_cups (int): Number of cups on the pig
        """
        super().__init__(diameter, length, density, friction_coef, bypass_fraction, pipeline_diameter)
        
        # Cup-specific properties
        self.num_cups = num_cups
        self.cup_flexibility = 0.85  # Flexibility factor of cups (0-1)
        self.wear_rate = 5e-7  # m/m, slower wear rate than foam
        self.wear_factor = 0.0  # 0% initial wear
    
    def calculate_differential_pressure(self, 
                                       fluid_density: float, 
                                       fluid_viscosity: float,
                                       flow_rate: float,
                                       pipeline_roughness: float) -> float:
        """
        Calculate pressure differential across the cup pig.
        
        Args:
            fluid_density (float): Density of driving fluid in kg/m³
            fluid_viscosity (float): Viscosity of driving fluid in Pa·s
            flow_rate (float): Flow rate in m³/s
            pipeline_roughness (float): Pipeline roughness in m
            
        Returns:
            float: Pressure differential in Pa
        """
        # Use base calculation
        base_diff_pressure = super().calculate_differential_pressure(
            fluid_density, fluid_viscosity, flow_rate, pipeline_roughness
        )
        
        # Cup pigs have higher differential pressure due to better sealing
        # Scale by number of cups and their sealing capability
        cup_factor = (1 + 0.2 * self.num_cups) * (1 - self.wear_factor)
        
        # Pipeline bend effects - cups have more pressure loss in bends
        bend_factor = 1.0  # Default to 1.0 (no bend)
        
        # Final differential pressure
        self.differential_pressure = base_diff_pressure * cup_factor * bend_factor
        
        # Update force balance
        self.force_balance["cup_sealing_force"] = self.differential_pressure * 0.2 * self.num_cups * self.cross_section
        self.force_balance["flow_force"] = self.differential_pressure * self.cross_section
        self.force_balance["wall_friction"] = -self.friction_coef * self.mass * 9.81
        self.force_balance["net_force"] = (self.force_balance["flow_force"] + 
                                         self.force_balance["cup_sealing_force"] + 
                                         self.force_balance["wall_friction"])
        
        # Calculate acceleration
        self.acceleration = self.force_balance["net_force"] / self.mass
        
        return self.differential_pressure
    
    def calculate_cleaning_efficiency(self, 
                                    debris_density: float = 800.0,
                                    pipe_wall_condition: float = 0.8) -> float:
        """
        Calculate cleaning efficiency of the cup pig.
        
        Args:
            debris_density (float): Density of debris in kg/m³
            pipe_wall_condition (float): Pipe wall condition factor (0-1, 1 is clean)
            
        Returns:
            float: Cleaning efficiency (0-1)
        """
        # Cup pigs have higher cleaning efficiency than foam pigs
        # Factors: number of cups, cup flexibility, sealing efficiency, velocity
        
        # Velocity factor (peaks at 2.0 m/s)
        vel_factor = 1.0 - 0.4 * abs(self.velocity - 2.0) / 2.0
        vel_factor = max(0.4, min(1.0, vel_factor))
        
        # Cup cleaning factor
        cup_factor = 0.6 + 0.4 * (self.num_cups / 5)  # 5 cups is max effective
        cup_factor = min(1.0, cup_factor)
        
        # Calculated cleaning efficiency
        efficiency = (
            self.sealing_efficiency * 
            (1.0 - self.wear_factor) * 
            vel_factor * 
            cup_factor * 
            self.cup_flexibility * 
            pipe_wall_condition
        )
        
        return max(0.0, min(1.0, efficiency))
    
    def update_wear(self, distance_traveled: float):
        """
        Update wear factor based on distance traveled.
        
        Args:
            distance_traveled (float): Distance traveled in meters
            
        Returns:
            float: Updated wear factor
        """
        # Cup pigs wear more in dirty pipelines or at high velocities
        velocity_wear_multiplier = 1.0 + max(0, (self.velocity - 3.0)) / 2.0
        
        # Update wear factor
        self.wear_factor += self.wear_rate * distance_traveled * velocity_wear_multiplier
        self.wear_factor = min(1.0, self.wear_factor)  # Cap at 100% wear
        
        # Update bypass fraction as wear increases
        self.bypass_fraction += 0.3 * self.wear_factor
        self.bypass_fraction = min(0.80, self.bypass_fraction)  # Max 80% bypass
        
        # Update other properties
        self.sealing_efficiency = 1.0 - self.bypass_fraction
        
        return self.wear_factor


class SpherePig(BasePig):
    """
    Sphere pig model for pipeline batching and gauging operations.
    
    Sphere pigs are simple, highly flexible pigs used primarily for batching,
    wiping, and light cleaning. They can navigate tight bends easily.
    """
    
    def __init__(self, 
                 diameter: float, 
                 density: float = 800.0, 
                 friction_coef: float = 0.2,
                 bypass_fraction: float = 0.05,
                 pipeline_diameter: float = 0.1016):
        """
        Initialize a sphere pig model.
        
        Args:
            diameter (float): Pig diameter in meters (sphere, so length = diameter)
            density (float): Pig material density in kg/m³
            friction_coef (float): Friction coefficient between pig and pipe wall
            bypass_fraction (float): Fraction of flow that bypasses the pig (0-1)
            pipeline_diameter (float): Pipeline inner diameter in meters
        """
        # For a sphere, length = diameter
        super().__init__(diameter, diameter, density, friction_coef, bypass_fraction, pipeline_diameter)
        
        # Sphere-specific properties
        self.elasticity = 0.9  # Elasticity factor (0-1)
        self.rotational_speed = 0.0  # rad/s
        self.wear_rate = 3e-7  # m/m, moderate wear rate
        self.wear_factor = 0.0  # 0% initial wear
    
    def calculate_velocity(self, flow_rate: float) -> float:
        """
        Calculate sphere pig velocity based on flow rate and bypass.
        Spheres may rotate as they move, affecting velocity.
        
        Args:
            flow_rate (float): Total flow rate in m³/s
            
        Returns:
            float: Sphere velocity in m/s
        """
        # Call parent method to get base velocity
        base_velocity = super().calculate_velocity(flow_rate)
        
        # Sphere pigs roll as they move - rotational effects on velocity
        # Higher flow rates cause more rolling, which increases velocity slightly
        # but also increases bypass
        roll_factor = 1.0 + 0.05 * min(5.0, base_velocity) / 5.0
        
        # Set rotational speed - roughly flow velocity converted to rad/s
        # assuming sphere diameter as the moment arm
        fluid_velocity = flow_rate / self.pipeline_area
        self.rotational_speed = 2 * fluid_velocity / self.diameter
        
        # Adjusted velocity
        self.velocity = base_velocity * roll_factor
        
        return self.velocity
    
    def calculate_differential_pressure(self, 
                                      fluid_density: float, 
                                      fluid_viscosity: float,
                                      flow_rate: float,
                                      pipeline_roughness: float) -> float:
        """
        Calculate pressure differential across the sphere pig.
        
        Args:
            fluid_density (float): Density of driving fluid in kg/m³
            fluid_viscosity (float): Viscosity of driving fluid in Pa·s
            flow_rate (float): Flow rate in m³/s
            pipeline_roughness (float): Pipeline roughness in m
            
        Returns:
            float: Pressure differential in Pa
        """
        # Sphere pigs have lower differential pressure due to their shape
        # and higher bypass flow
        
        # Get base calculation
        base_diff_pressure = super().calculate_differential_pressure(
            fluid_density, fluid_viscosity, flow_rate, pipeline_roughness
        )
        
        # Sphere shape factor - spheres have less surface contact and less drag
        sphere_factor = 0.85 - (0.1 * self.wear_factor)
        
        # Elasticity factor - more elastic spheres deform to pipe better
        elasticity_factor = 0.9 + (0.2 * self.elasticity)
        
        # Final differential pressure
        self.differential_pressure = base_diff_pressure * sphere_factor * elasticity_factor
        
        # Update force balance
        self.force_balance["flow_force"] = self.differential_pressure * self.cross_section
        self.force_balance["wall_friction"] = -self.friction_coef * self.mass * 9.81
        
        # Additional rolling friction (reduced compared to sliding)
        rolling_friction = -0.5 * self.friction_coef * self.mass * 9.81
        self.force_balance["rolling_friction"] = rolling_friction
        
        self.force_balance["net_force"] = (self.force_balance["flow_force"] + 
                                        self.force_balance["wall_friction"] +
                                        self.force_balance["rolling_friction"])
        
        # Calculate acceleration
        self.acceleration = self.force_balance["net_force"] / self.mass
        
        return self.differential_pressure
    
    def calculate_cleaning_efficiency(self, 
                                    debris_density: float = 800.0,
                                    pipe_wall_condition: float = 0.8) -> float:
        """
        Calculate cleaning efficiency of the sphere pig.
        
        Args:
            debris_density (float): Density of debris in kg/m³
            pipe_wall_condition (float): Pipe wall condition factor (0-1, 1 is clean)
            
        Returns:
            float: Cleaning efficiency (0-1)
        """
        # Sphere pigs have lower cleaning efficiency than cup or foam pigs
        # Benefit is navigation through tight bends
        
        # Velocity factor (peaks at 2.5 m/s)
        vel_factor = 1.0 - 0.3 * abs(self.velocity - 2.5) / 2.5
        vel_factor = max(0.5, min(1.0, vel_factor))
        
        # Elasticity factor - more elastic spheres clean better
        elasticity_factor = 0.7 + 0.3 * self.elasticity
        
        # Rotation factor - rotation helps cleaning
        rotation_factor = 0.8 + 0.2 * min(1.0, self.rotational_speed / 10.0)
        
        # Calculated cleaning efficiency (lower than other pig types)
        efficiency = (
            0.7 *  # Base efficiency multiplier for spheres (lower than cups/discs)
            self.sealing_efficiency * 
            (1.0 - self.wear_factor) * 
            vel_factor * 
            elasticity_factor *
            rotation_factor *
            pipe_wall_condition
        )
        
        return max(0.0, min(1.0, efficiency))


class IntelligentPig(BasePig):
    """
    Intelligent pig model for pipeline inspection operations.
    
    Intelligent pigs are equipped with sensors and data recording capabilities
    for pipeline inspection. They measure wall thickness, detect cracks, and
    identify other pipeline defects.
    """
    
    def __init__(self, 
                 diameter: float, 
                 length: float, 
                 density: float = 1200.0, 
                 friction_coef: float = 0.15,
                 bypass_fraction: float = 0.01,
                 pipeline_diameter: float = 0.1016):
        """
        Initialize an intelligent pig model.
        
        Args:
            diameter (float): Pig diameter in meters
            length (float): Pig length in meters
            density (float): Pig material density in kg/m³
            friction_coef (float): Friction coefficient between pig and pipe wall
            bypass_fraction (float): Fraction of flow that bypasses the pig (0-1)
            pipeline_diameter (float): Pipeline inner diameter in meters
        """
        super().__init__(diameter, length, density, friction_coef, bypass_fraction, pipeline_diameter)
        
        # Intelligent pig specific properties
        self.battery_life = 100.0  # hours
        self.battery_remaining = 100.0  # hours
        self.data_storage = 500.0  # GB
        self.data_used = 0.0  # GB
        self.sensor_accuracy = 0.95  # base accuracy
        self.sensor_types = ["MFL", "Caliper", "IMU", "Odometer"]
        self.inspection_results = {}
        
        # Power consumption rates (per hour)
        self.power_rates = {
            "idle": 0.1,  # hours of battery per hour
            "operating": 1.0,  # hours of battery per hour
            "high_resolution": 2.0  # hours of battery per hour
        }
        
        # Data generation rates (GB per km)
        self.data_rates = {
            "low_resolution": 0.5,  # GB per kilometer
            "medium_resolution": 2.0,  # GB per kilometer
            "high_resolution": 8.0   # GB per kilometer
        }
        
        self.resolution_mode = "medium_resolution"
        self.power_mode = "operating"
    
    def calculate_velocity(self, flow_rate: float) -> float:
        """
        Calculate intelligent pig velocity considering operating constraints.
        
        Intelligent pigs have preferred velocity ranges for optimal sensor operation.
        
        Args:
            flow_rate (float): Total flow rate in m³/s
            
        Returns:
            float: Pig velocity in m/s
        """
        # Calculate base velocity
        base_velocity = super().calculate_velocity(flow_rate)
        
        # Check if velocity is in optimal range (0.5 - 3.0 m/s typically)
        if base_velocity < 0.5:
            logger.warning(f"Intelligent pig velocity ({base_velocity:.2f} m/s) below minimum recommended (0.5 m/s)")
            # Low velocity - sensors can compensate but data quality suffers
            self.sensor_accuracy = 0.85
        elif base_velocity > 3.0:
            logger.warning(f"Intelligent pig velocity ({base_velocity:.2f} m/s) above maximum recommended (3.0 m/s)")
            # High velocity - sensors may not capture data accurately
            self.sensor_accuracy = 0.75
        else:
            # Optimal velocity range
            self.sensor_accuracy = 0.95
        
        self.velocity = base_velocity
        return self.velocity
    
    def update_battery_and_storage(self, time_elapsed: float, distance_traveled: float):
        """
        Update battery life and data storage based on operation time and distance.
        
        Args:
            time_elapsed (float): Time elapsed in hours
            distance_traveled (float): Distance traveled in kilometers
            
        Returns:
            Tuple[float, float]: Remaining battery life (hours) and data storage (GB)
        """
        # Update battery
        battery_consumed = time_elapsed * self.power_rates[self.power_mode]
        self.battery_remaining -= battery_consumed
        self.battery_remaining = max(0.0, self.battery_remaining)
        
        # Update data storage
        data_generated = distance_traveled * self.data_rates[self.resolution_mode]
        self.data_used += data_generated
        
        # Check for low battery or storage
        if self.battery_remaining < 10.0:
            logger.warning(f"Intelligent pig battery low: {self.battery_remaining:.1f} hours remaining")
        
        if self.data_used > self.data_storage * 0.9:
            logger.warning(f"Intelligent pig storage nearly full: {self.data_used:.1f}/{self.data_storage:.1f} GB used")
            
            # Automatically reduce resolution if storage is critically low
            if self.data_used > self.data_storage * 0.95 and self.resolution_mode != "low_resolution":
                logger.warning("Automatically reducing to low resolution due to critical storage")
                self.resolution_mode = "low_resolution"
        
        return (self.battery_remaining, self.data_storage - self.data_used)
    
    def record_inspection_data(self, 
                             pipeline_position: float, 
                             wall_thickness: float = 10.0,
                             defect_present: bool = False,
                             defect_type: str = None,
                             defect_size: float = 0.0):
        """
        Record pipeline inspection data at the current position.
        
        Args:
            pipeline_position (float): Position along pipeline in meters
            wall_thickness (float): Pipe wall thickness in mm
            defect_present (bool): Whether a defect is present
            defect_type (str): Type of defect if present
            defect_size (float): Size of defect in mm
            
        Returns:
            Dict: The recorded inspection data
        """
        # Only record if battery and storage are available
        if self.battery_remaining <= 0:
            logger.error("Cannot record inspection data: Battery depleted")
            return None
        
        if self.data_used >= self.data_storage:
            logger.error("Cannot record inspection data: Storage full")
            return None
        
        # Apply sensor accuracy to measurements
        accuracy_factor = self.sensor_accuracy * (0.95 + 0.1 * np.random.random())
        measured_thickness = wall_thickness * accuracy_factor
        
        # Add some random noise to measurements
        noise_factor = 0.02 * (2 * np.random.random() - 1)  # ±2% noise
        measured_thickness *= (1 + noise_factor)
        
        # Determine if defect is detected (based on accuracy and defect size)
        defect_detected = False
        defect_data = None
        
        if defect_present:
            # Probability of detection depends on defect size and sensor accuracy
            detection_probability = min(1.0, self.sensor_accuracy * (defect_size / 2.0))
            defect_detected = np.random.random() < detection_probability
            
            if defect_detected:
                measured_size = defect_size * (0.9 + 0.2 * np.random.random())  # Some measurement error
                defect_data = {
                    "type": defect_type,
                    "measured_size": measured_size,
                    "confidence": detection_probability * 100  # Confidence percentage
                }
        
        # Create inspection record
        inspection_data = {
            "position": pipeline_position,
            "wall_thickness": measured_thickness,
            "defect_detected": defect_detected,
            "defect_data": defect_data,
            "timestamp": time.time(),
            "battery_remaining": self.battery_remaining,
            "data_storage_remaining": self.data_storage - self.data_used,
            "resolution_mode": self.resolution_mode
        }
        
        # Save to inspection results
        self.inspection_results[pipeline_position] = inspection_data
        
        return inspection_data
    
    def set_resolution_mode(self, mode: str):
        """
        Set the sensor resolution mode.
        
        Args:
            mode (str): Resolution mode ('low_resolution', 'medium_resolution', 'high_resolution')
            
        Returns:
            str: The selected mode
        """
        if mode in self.data_rates:
            self.resolution_mode = mode
            logger.info(f"Set intelligent pig resolution mode to {mode}")
            
            # Adjust power mode based on resolution
            if mode == "high_resolution":
                self.power_mode = "high_resolution"
            else:
                self.power_mode = "operating"
                
            return mode
        else:
            logger.warning(f"Unknown resolution mode: {mode}")
            return self.resolution_mode
    
    def get_inspection_summary(self) -> Dict[str, Any]:
        """
        Get a summary of inspection results.
        
        Returns:
            Dict[str, Any]: Summary statistics of inspection data
        """
        if not self.inspection_results:
            return {"status": "No inspection data recorded"}
        
        total_points = len(self.inspection_results)
        defects_found = sum(1 for data in self.inspection_results.values() if data["defect_detected"])
        avg_wall_thickness = sum(data["wall_thickness"] for data in self.inspection_results.values()) / total_points
        
        # Get min wall thickness and its position
        min_thickness = float('inf')
        min_thickness_position = 0
        
        for position, data in self.inspection_results.items():
            if data["wall_thickness"] < min_thickness:
                min_thickness = data["wall_thickness"]
                min_thickness_position = position
        
        return {
            "status": "Inspection data available",
            "total_inspection_points": total_points,
            "pipeline_length_inspected": max(self.inspection_results.keys()) - min(self.inspection_results.keys()),
            "defects_found": defects_found,
            "average_wall_thickness": avg_wall_thickness,
            "minimum_wall_thickness": min_thickness,
            "minimum_thickness_position": min_thickness_position,
            "inspection_resolution": self.resolution_mode,
            "battery_remaining": self.battery_remaining,
            "storage_remaining": self.data_storage - self.data_used
        }


class GelPig(BasePig):
    """
    Gel pig model for specialized applications.
    
    Gel pigs are used for specialized applications like dewatering, chemical
    treatment application, or removing very soft debris. They can change shape
    to navigate complex geometries.
    """
    
    def __init__(self, 
                 diameter: float, 
                 length: float, 
                 density: float = 1050.0, 
                 friction_coef: float = 0.4,
                 bypass_fraction: float = 0.005,
                 pipeline_diameter: float = 0.1016,
                 gel_type: str = "standard"):
        """
        Initialize a gel pig model.
        
        Args:
            diameter (float): Pig diameter in meters
            length (float): Pig length in meters
            density (float): Pig material density in kg/m³
            friction_coef (float): Friction coefficient between pig and pipe wall
            bypass_fraction (float): Fraction of flow that bypasses the pig (0-1)
            pipeline_diameter (float): Pipeline inner diameter in meters
            gel_type (str): Type of gel ('standard', 'chemical', 'solvent')
        """
        super().__init__(diameter, length, density, friction_coef, bypass_fraction, pipeline_diameter)
        
        # Gel pig specific properties
        self.gel_type = gel_type
        self.deformation_factor = 0.95  # How much the gel can deform (0-1)
        self.consistency = 0.85  # Gel consistency (0-1)
        self.chemical_concentration = 1.0  # For chemical treatment gels (0-1)
        self.solvent_strength = 1.0  # For solvent gels (0-1)
        
        # Set properties based on gel type
        if gel_type == "chemical":
            self.chemical_concentration = 0.9
            self.density = 1100.0
            self.friction_coef = 0.35
        elif gel_type == "solvent":
            self.solvent_strength = 0.95
            self.density = 950.0
            self.friction_coef = 0.3
        
        # Gel pigs deplete as they travel
        self.depletion_rate = 0.0001  # % per meter
        self.depletion_factor = 0.0  # 0% initial depletion
    
    def calculate_velocity(self, flow_rate: float) -> float:
        """
        Calculate gel pig velocity.
        
        Gel pigs may move slower than the flow due to wall friction and
        their semi-solid nature.
        
        Args:
            flow_rate (float): Total flow rate in m³/s
            
        Returns:
            float: Pig velocity in m/s
        """
        # Calculate base velocity
        base_velocity = super().calculate_velocity(flow_rate)
        
        # Gel pigs move slightly slower than the flow due to drag
        gel_factor = 0.9 * (1 - 0.3 * self.depletion_factor)
        
        self.velocity = base_velocity * gel_factor
        return self.velocity
    
    def calculate_differential_pressure(self, 
                                      fluid_density: float, 
                                      fluid_viscosity: float,
                                      flow_rate: float,
                                      pipeline_roughness: float) -> float:
        """
        Calculate pressure differential across the gel pig.
        
        Args:
            fluid_density (float): Density of driving fluid in kg/m³
            fluid_viscosity (float): Viscosity of driving fluid in Pa·s
            flow_rate (float): Flow rate in m³/s
            pipeline_roughness (float): Pipeline roughness in m
            
        Returns:
            float: Pressure differential in Pa
        """
        # Get base calculation
        base_diff_pressure = super().calculate_differential_pressure(
            fluid_density, fluid_viscosity, flow_rate, pipeline_roughness
        )
        
        # Gel creates more flow resistance initially, but less as it depletes
        gel_factor = 1.2 * (1 - 0.5 * self.depletion_factor)
        
        # Add consistency impact
        consistency_factor = 0.8 + 0.4 * self.consistency
        
        # Final differential pressure
        self.differential_pressure = base_diff_pressure * gel_factor * consistency_factor
        
        # Update force balance
        self.force_balance["flow_force"] = self.differential_pressure * self.cross_section
        self.force_balance["wall_friction"] = -self.friction_coef * self.mass * 9.81
        self.force_balance["net_force"] = (self.force_balance["flow_force"] + 
                                         self.force_balance["wall_friction"])
        
        # Calculate acceleration
        self.acceleration = self.force_balance["net_force"] / self.mass
        
        return self.differential_pressure
    
    def calculate_chemical_application(self, 
                                     surface_area: float,
                                     application_rate: float = 0.001) -> float:
        """
        Calculate chemical application for chemical gel pigs.
        
        Args:
            surface_area (float): Pipeline surface area covered in m²
            application_rate (float): Base chemical application rate in kg/m²
            
        Returns:
            float: Chemical amount applied in kg
        """
        if self.gel_type != "chemical":
            return 0.0
            
        # Calculate chemical application
        effective_rate = application_rate * self.chemical_concentration * (1 - self.depletion_factor)
        chemical_applied = surface_area * effective_rate
        
        # Update gel properties
        new_depletion = chemical_applied / (self.mass * 0.2)  # Assume 20% of gel mass is chemical
        self.depletion_factor += new_depletion
        self.depletion_factor = min(1.0, self.depletion_factor)
        
        # Update bypass fraction as gel depletes
        self.bypass_fraction += 0.1 * new_depletion
        self.bypass_fraction = min(0.9, self.bypass_fraction)
        
        # Update other properties
        self.sealing_efficiency = 1.0 - self.bypass_fraction
        
        return chemical_applied
    
    def calculate_cleaning_efficiency(self, 
                                    debris_density: float = 800.0,
                                    pipe_wall_condition: float = 0.8) -> float:
        """
        Calculate cleaning efficiency of the gel pig.
        
        Args:
            debris_density (float): Density of debris in kg/m³
            pipe_wall_condition (float): Pipe wall condition factor (0-1, 1 is clean)
            
        Returns:
            float: Cleaning efficiency (0-1)
        """
        # Different cleaning mechanism than mechanical pigs
        # Gel pigs clean by chemical action and complete wall contact
        
        # Base cleaning factor depends on gel type
        if self.gel_type == "standard":
            base_cleaning = 0.6
        elif self.gel_type == "chemical":
            base_cleaning = 0.8 * self.chemical_concentration
        elif self.gel_type == "solvent":
            base_cleaning = 0.85 * self.solvent_strength
        else:
            base_cleaning = 0.5
        
        # Velocity factor - gel pigs work best at moderate velocities
        vel_factor = 1.0 - 0.4 * abs(self.velocity - 1.0) / 1.0
        vel_factor = max(0.5, min(1.0, vel_factor))
        
        # Consistency and depletion effects
        consistency_factor = 0.7 + 0.3 * self.consistency
        depletion_effect = 1.0 - 0.9 * self.depletion_factor  # Depleted gel cleans poorly
        
        # Calculated cleaning efficiency
        efficiency = (
            base_cleaning * 
            self.sealing_efficiency * 
            vel_factor * 
            consistency_factor * 
            depletion_effect * 
            pipe_wall_condition
        )
        
        return max(0.0, min(1.0, efficiency))
    
    def update_depletion(self, distance_traveled: float):
        """
        Update gel depletion based on distance traveled.
        
        Args:
            distance_traveled (float): Distance traveled in meters
            
        Returns:
            float: Updated depletion factor
        """
        # Update depletion factor
        self.depletion_factor += self.depletion_rate * distance_traveled
        self.depletion_factor = min(1.0, self.depletion_factor)
        
        # Update bypass fraction as gel depletes
        original_bypass = 0.005  # Initial bypass fraction
        self.bypass_fraction = original_bypass + (0.9 - original_bypass) * self.depletion_factor
        
        # Update mass and density as gel depletes
        self.density = 1050.0 * (1 - 0.2 * self.depletion_factor)  # Density decreases
        self.mass = self.volume * self.density
        
        # Update other properties
        self.sealing_efficiency = 1.0 - self.bypass_fraction
        
        return self.depletion_factor


class DualDiameterPipeline:
    """
    Model for pipelines with changing diameters.
    
    This class simulates how pigs behave when transitioning between pipeline 
    sections with different diameters, which is a common challenge in pigging 
    operations.
    """
    
    def __init__(self):
        """Initialize dual diameter pipeline model."""
        self.sections = []  # List of (start_position, end_position, diameter) tuples
        
    def add_section(self, start_position: float, end_position: float, diameter: float):
        """
        Add a pipeline section with specified diameter.
        
        Args:
            start_position (float): Start position in meters
            end_position (float): End position in meters
            diameter (float): Pipeline diameter in meters
            
        Returns:
            List: Updated sections list
        """
        self.sections.append((start_position, end_position, diameter))
        # Sort sections by start position
        self.sections.sort(key=lambda x: x[0])
        return self.sections
    
    def get_diameter_at_position(self, position: float) -> float:
        """
        Get pipeline diameter at a specific position.
        
        Args:
            position (float): Position along pipeline in meters
            
        Returns:
            float: Pipeline diameter in meters
        """
        for start, end, diameter in self.sections:
            if start <= position <= end:
                return diameter
        
        # If position is out of any defined section, return the last section's diameter
        if self.sections and position > self.sections[-1][1]:
            return self.sections[-1][2]
        
        # Default diameter if no sections defined
        return 0.1  # Default 100mm
    
    def is_transition_point(self, position: float, tolerance: float = 0.1) -> bool:
        """
        Check if a position is at a diameter transition point.
        
        Args:
            position (float): Position along pipeline in meters
            tolerance (float): Distance tolerance in meters
            
        Returns:
            bool: True if position is at a transition point
        """
        for i, (start, end, _) in enumerate(self.sections):
            # Check if position is within tolerance of section boundary
            if abs(position - start) < tolerance and i > 0:
                return True
            if abs(position - end) < tolerance and i < len(self.sections) - 1:
                return True
        return False
    
    def transition_forces(self, 
                        pig: BasePig, 
                        position: float, 
                        flow_rate: float,
                        fluid_density: float) -> Dict[str, float]:
        """
        Calculate forces acting on pig at diameter transitions.
        
        Args:
            pig (BasePig): The pig object
            position (float): Position along pipeline in meters
            flow_rate (float): Flow rate in m³/s
            fluid_density (float): Fluid density in kg/m³
            
        Returns:
            Dict[str, float]: Forces acting on pig at transition
        """
        forces = {"expansion_force": 0.0, "contraction_force": 0.0, "net_force": 0.0}
        
        # Check if we're at a transition point
        if not self.is_transition_point(position):
            return forces
        
        # Find the transition
        for i, (start, end, diameter) in enumerate(self.sections):
            # Check if we're at the end of this section
            if abs(position - end) < 0.1 and i < len(self.sections) - 1:
                # Get next section diameter
                next_diameter = self.sections[i+1][2]
                
                # Calculate transition forces
                if next_diameter > diameter:
                    # Expansion
                    area_ratio = (next_diameter / diameter) ** 2
                    
                    # Flow velocity at current section
                    velocity = flow_rate / (math.pi * (diameter / 2) ** 2)
                    
                    # Pressure drop at expansion
                    pressure_drop = 0.5 * fluid_density * velocity**2 * (1 - 1/area_ratio)
                    
                    # Force due to expansion
                    expansion_force = pressure_drop * math.pi * (pig.diameter / 2) ** 2
                    forces["expansion_force"] = expansion_force
                    forces["net_force"] = expansion_force
                    
                elif next_diameter < diameter:
                    # Contraction
                    area_ratio = (next_diameter / diameter) ** 2
                    
                    # Flow velocity at current section
                    velocity = flow_rate / (math.pi * (diameter / 2) ** 2)
                    
                    # Increase in velocity at contraction
                    velocity_increase = velocity * (1/area_ratio - 1)
                    
                    # Additional force needed to accelerate pig
                    contraction_force = pig.mass * velocity_increase / 0.1  # Assume transition occurs over 0.1m
                    forces["contraction_force"] = contraction_force
                    forces["net_force"] = contraction_force
                
                # Only consider one transition at a time
                break
                
            # Check if we're at the start of this section
            elif abs(position - start) < 0.1 and i > 0:
                # Get previous section diameter
                prev_diameter = self.sections[i-1][2]
                
                # Calculate transition forces (similar logic as above)
                if diameter > prev_diameter:
                    # Expansion from previous to current
                    area_ratio = (diameter / prev_diameter) ** 2
                    
                    # Flow velocity at previous section
                    velocity = flow_rate / (math.pi * (prev_diameter / 2) ** 2)
                    
                    # Pressure drop at expansion
                    pressure_drop = 0.5 * fluid_density * velocity**2 * (1 - 1/area_ratio)
                    
                    # Force due to expansion
                    expansion_force = pressure_drop * math.pi * (pig.diameter / 2) ** 2
                    forces["expansion_force"] = expansion_force
                    forces["net_force"] = expansion_force
                    
                elif diameter < prev_diameter:
                    # Contraction from previous to current
                    area_ratio = (diameter / prev_diameter) ** 2
                    
                    # Flow velocity at previous section
                    velocity = flow_rate / (math.pi * (prev_diameter / 2) ** 2)
                    
                    # Increase in velocity at contraction
                    velocity_increase = velocity * (1/area_ratio - 1)
                    
                    # Additional force needed to accelerate pig
                    contraction_force = pig.mass * velocity_increase / 0.1
                    forces["contraction_force"] = contraction_force
                    forces["net_force"] = contraction_force
                
                # Only consider one transition at a time
                break
        
        return forces


class PiggingSimulation:
    """
    Main pigging simulation class that coordinates the behavior of pigs
    in pipelines under different operating conditions.
    """
    
    def __init__(self, pipeline_length: float = 1000.0, pipeline_diameter: float = 0.1016):
        """
        Initialize the pigging simulation.
        
        Args:
            pipeline_length (float): Length of pipeline in meters
            pipeline_diameter (float): Diameter of pipeline in meters
        """
        self.pipeline_length = pipeline_length
        self.pipeline_diameter = pipeline_diameter
        self.pipeline = DualDiameterPipeline()
        
        # Add default section covering the entire pipeline
        self.pipeline.add_section(0.0, pipeline_length, pipeline_diameter)
        
        # Simulation state
        self.time = 0.0  # Current simulation time
        self.time_step = 0.1  # Simulation time step in seconds
        self.pigs = []  # List of pigs in the pipeline
        self.flow_rate = 0.0  # Current flow rate in m³/s
        
        # Fluid properties
        self.fluid_properties = {
            "density": 1000.0,  # kg/m³
            "viscosity": 0.001,  # Pa·s
            "temperature": 293.15,  # K
        }
        
        # Pipeline properties
        self.pipeline_properties = {
            "roughness": 0.00005,  # m
            "wall_condition": 0.8,  # 0-1 scale (1 is clean)
            "debris_density": 900.0,  # kg/m³
            "elevation_profile": [],  # List of (position, elevation) tuples
            "bend_points": [],  # List of (position, bend_angle) tuples
        }
        
        # Simulation results
        self.results = {
            "pig_positions": [],  # List of pig positions over time
            "pig_velocities": [],  # List of pig velocities over time
            "differential_pressures": [],  # List of pressure differentials over time
            "cleaning_efficiencies": [],  # List of cleaning efficiencies over time
            "time_stamps": [],  # List of time stamps
            "pig_travel_times": [],  # Time taken by pigs to travel pipeline
            "debris_removal": [],  # Amount of debris removed
            "energy_consumption": [],  # Energy used for pigging
        }
        
        logger.info(f"Initialized pigging simulation with {pipeline_length}m pipeline, {pipeline_diameter}m diameter")
    
    def add_pig(self, pig_type: str, 
               diameter: float, 
               length: float = None,
               density: float = None,
               friction_coef: float = None,
               bypass_fraction: float = None,
               position: float = 0.0,
               **kwargs) -> BasePig:
        """
        Add a pig to the simulation.
        
        Args:
            pig_type (str): Type of pig ('foam', 'cup', 'sphere', 'intelligent', 'gel')
            diameter (float): Pig diameter in meters
            length (float, optional): Pig length in meters
            density (float, optional): Pig density in kg/m³
            friction_coef (float, optional): Friction coefficient
            bypass_fraction (float, optional): Bypass flow fraction
            position (float, optional): Initial position along pipeline
            **kwargs: Additional arguments specific to pig type
            
        Returns:
            BasePig: The created pig object
        """
        # Use defaults if not specified
        if length is None and pig_type != "sphere":
            length = 0.3  # Default 30cm length
        
        if pig_type.lower() == "foam":
            if density is None: density = 250.0
            if friction_coef is None: friction_coef = 0.3
            if bypass_fraction is None: bypass_fraction = 0.08
            
            pig = FoamPig(diameter, length, density, friction_coef, bypass_fraction, self.pipeline_diameter)
            
        elif pig_type.lower() == "cup":
            if density is None: density = 900.0
            if friction_coef is None: friction_coef = 0.35
            if bypass_fraction is None: bypass_fraction = 0.02
            
            num_cups = kwargs.get('num_cups', 3)
            pig = CupPig(diameter, length, density, friction_coef, bypass_fraction, 
                         self.pipeline_diameter, num_cups)
            
        elif pig_type.lower() == "sphere":
            if density is None: density = 800.0
            if friction_coef is None: friction_coef = 0.2
            if bypass_fraction is None: bypass_fraction = 0.05
            
            pig = SpherePig(diameter, density, friction_coef, bypass_fraction, self.pipeline_diameter)
            
        elif pig_type.lower() == "intelligent":
            if density is None: density = 1200.0
            if friction_coef is None: friction_coef = 0.15
            if bypass_fraction is None: bypass_fraction = 0.01
            
            pig = IntelligentPig(diameter, length, density, friction_coef, bypass_fraction, self.pipeline_diameter)
            
        elif pig_type.lower() == "gel":
            if density is None: density = 1050.0
            if friction_coef is None: friction_coef = 0.4
            if bypass_fraction is None: bypass_fraction = 0.005
            
            gel_type = kwargs.get('gel_type', 'standard')
            pig = GelPig(diameter, length, density, friction_coef, bypass_fraction, 
                         self.pipeline_diameter, gel_type)
            
        else:
            logger.warning(f"Unknown pig type: {pig_type}, using BasePig")
            pig = BasePig(diameter, length or 0.3, density or 900.0, 
                         friction_coef or 0.3, bypass_fraction or 0.05, self.pipeline_diameter)
        
        # Set initial position
        pig.position = position
        
        # Add to pig list
        self.pigs.append(pig)
        
        logger.info(f"Added {pig_type} pig at position {position}m")
        return pig
    
    def set_flow_rate(self, flow_rate: float):
        """
        Set the flow rate for the simulation.
        
        Args:
            flow_rate (float): Flow rate in m³/s
            
        Returns:
            float: The set flow rate
        """
        self.flow_rate = flow_rate
        logger.info(f"Set flow rate to {flow_rate} m³/s")
        return flow_rate
    
    def set_fluid_properties(self, 
                           density: float = None, 
                           viscosity: float = None,
                           temperature: float = None):
        """
        Set fluid properties for the simulation.
        
        Args:
            density (float, optional): Fluid density in kg/m³
            viscosity (float, optional): Fluid viscosity in Pa·s
            temperature (float, optional): Fluid temperature in K
            
        Returns:
            dict: Updated fluid properties
        """
        if density is not None:
            self.fluid_properties["density"] = density
        
        if viscosity is not None:
            self.fluid_properties["viscosity"] = viscosity
        
        if temperature is not None:
            self.fluid_properties["temperature"] = temperature
        
        logger.info(f"Updated fluid properties: {self.fluid_properties}")
        return self.fluid_properties
    
    def add_pipeline_section(self, start_position: float, end_position: float, diameter: float):
        """
        Add a pipeline section with a specified diameter.
        
        Args:
            start_position (float): Start position in meters
            end_position (float): End position in meters
            diameter (float): Pipeline diameter in meters
            
        Returns:
            List: Updated pipeline sections
        """
        return self.pipeline.add_section(start_position, end_position, diameter)
    
    def add_elevation_point(self, position: float, elevation: float):
        """
        Add an elevation point to the pipeline profile.
        
        Args:
            position (float): Position along pipeline in meters
            elevation (float): Elevation in meters
            
        Returns:
            List: Updated elevation profile
        """
        self.pipeline_properties["elevation_profile"].append((position, elevation))
        # Sort by position
        self.pipeline_properties["elevation_profile"].sort(key=lambda x: x[0])
        return self.pipeline_properties["elevation_profile"]
    
    def add_bend_point(self, position: float, bend_angle: float):
        """
        Add a bend point to the pipeline.
        
        Args:
            position (float): Position along pipeline in meters
            bend_angle (float): Bend angle in degrees
            
        Returns:
            List: Updated bend points
        """
        self.pipeline_properties["bend_points"].append((position, bend_angle))
        # Sort by position
        self.pipeline_properties["bend_points"].sort(key=lambda x: x[0])
        return self.pipeline_properties["bend_points"]
    
    def get_elevation_at_position(self, position: float) -> float:
        """
        Get elevation at a specific position along pipeline.
        
        Args:
            position (float): Position along pipeline in meters
            
        Returns:
            float: Elevation in meters
        """
        profile = self.pipeline_properties["elevation_profile"]
        if not profile:
            return 0.0  # Flat pipeline
        
        # If position is before first point or after last point
        if position <= profile[0][0]:
            return profile[0][1]
        if position >= profile[-1][0]:
            return profile[-1][1]
        
        # Find surrounding points and interpolate
        for i in range(len(profile) - 1):
            pos1, elev1 = profile[i]
            pos2, elev2 = profile[i + 1]
            
            if pos1 <= position <= pos2:
                # Linear interpolation
                t = (position - pos1) / (pos2 - pos1)
                return elev1 + t * (elev2 - elev1)
        
        return 0.0  # Fallback
    
    def get_elevation_gradient(self, position: float) -> float:
        """
        Get elevation gradient at a specific position.
        
        Args:
            position (float): Position along pipeline in meters
            
        Returns:
            float: Elevation gradient (rise/run)
        """
        profile = self.pipeline_properties["elevation_profile"]
        if not profile or len(profile) < 2:
            return 0.0  # Flat pipeline
        
        # If position is before first point or after last point
        if position <= profile[0][0]:
            return 0.0
        if position >= profile[-1][0]:
            return 0.0
        
        # Find surrounding points and calculate gradient
        for i in range(len(profile) - 1):
            pos1, elev1 = profile[i]
            pos2, elev2 = profile[i + 1]
            
            if pos1 <= position <= pos2:
                # Gradient = rise / run
                return (elev2 - elev1) / (pos2 - pos1)
        
        return 0.0  # Fallback
    
    def is_at_bend(self, position: float, tolerance: float = 1.0) -> Tuple[bool, float]:
        """
        Check if position is at a pipeline bend.
        
        Args:
            position (float): Position along pipeline in meters
            tolerance (float): Distance tolerance in meters
            
        Returns:
            Tuple[bool, float]: (is_at_bend, bend_angle)
        """
        bends = self.pipeline_properties["bend_points"]
        
        for bend_pos, bend_angle in bends:
            if abs(position - bend_pos) < tolerance:
                return True, bend_angle
        
        return False, 0.0
    
    def run_step(self):
        """
        Run a single simulation step.
        
        Returns:
            Dict: Updated simulation state
        """
        # Skip if no pigs or flow rate
        if not self.pigs or self.flow_rate <= 0:
            self.time += self.time_step
            return {"time": self.time, "pigs": []}
        
        # Process each pig
        pig_states = []
        
        for pig in self.pigs:
            # Get pipeline diameter at pig position
            local_diameter = self.pipeline.get_diameter_at_position(pig.position)
            
            # Update pig diameter if needed (for transition modeling)
            if local_diameter != pig.pipeline_diameter:
                pig.pipeline_diameter = local_diameter
                pig.pipeline_area = math.pi * (local_diameter / 2) ** 2
            
            # Calculate pig velocity
            pig.calculate_velocity(self.flow_rate)
            
            # Check for bends
            is_at_bend, bend_angle = self.is_at_bend(pig.position)
            
            # Apply bend effects if applicable
            if is_at_bend:
                # Reduce velocity at bends
                bend_factor = max(0.5, 1.0 - 0.005 * bend_angle)  # Max 50% reduction at 90° bend
                pig.velocity *= bend_factor
                
                logger.info(f"Pig at {pig.position:.1f}m encountering {bend_angle}° bend: velocity reduced by {(1-bend_factor)*100:.1f}%")
            
            # Apply elevation effects
            gradient = self.get_elevation_gradient(pig.position)
            if abs(gradient) > 0.01:  # More than 1% grade
                # Gravitational acceleration component
                g_component = 9.81 * gradient  # m/s²
                
                # Calculate a simple correction to velocity
                buoyancy_factor = 1.0 - pig.density / self.fluid_properties["density"]
                elevation_effect = g_component * buoyancy_factor * 0.1  # Scale factor
                
                # Apply to velocity (slower uphill, faster downhill)
                pig.velocity -= elevation_effect
                
                # Ensure velocity doesn't go negative or exceed flow velocity
                pig.velocity = max(0, min(pig.velocity, 
                                         self.flow_rate / pig.pipeline_area))
            
            # Calculate differential pressure
            pig.calculate_differential_pressure(
                self.fluid_properties["density"],
                self.fluid_properties["viscosity"],
                self.flow_rate,
                self.pipeline_properties["roughness"]
            )
            
            # Check for diameter transitions
            transition_forces = self.pipeline.transition_forces(
                pig, pig.position, self.flow_rate, self.fluid_properties["density"]
            )
            
            if any(abs(f) > 0.1 for f in transition_forces.values()):
                # Apply transition effects
                pig.force_balance.update(transition_forces)
                pig.force_balance["net_force"] = (
                    pig.force_balance.get("flow_force", 0) + 
                    pig.force_balance.get("wall_friction", 0) + 
                    transition_forces.get("net_force", 0)
                )
                
                # Recalculate acceleration
                pig.acceleration = pig.force_balance["net_force"] / pig.mass
                
                logger.info(f"Pig at {pig.position:.1f}m experiencing diameter transition: additional force {transition_forces['net_force']:.1f}N")
            
            # Calculate cleaning efficiency
            cleaning_efficiency = 0.0
            
            # Different pigs have different cleaning behaviors
            if isinstance(pig, FoamPig):
                cleaning_efficiency = pig.calculate_cleaning_efficiency(
                    self.pipeline_properties["debris_density"],
                    self.pipeline_properties["wall_condition"]
                )
                # Update wear based on distance traveled
                distance = pig.velocity * self.time_step
                pig.update_wear(distance)
                
            elif isinstance(pig, CupPig):
                cleaning_efficiency = pig.calculate_cleaning_efficiency(
                    self.pipeline_properties["debris_density"],
                    self.pipeline_properties["wall_condition"]
                )
                
            elif isinstance(pig, SpherePig):
                cleaning_efficiency = pig.calculate_cleaning_efficiency(
                    self.pipeline_properties["debris_density"],
                    self.pipeline_properties["wall_condition"]
                )
                
            elif isinstance(pig, GelPig):
                cleaning_efficiency = pig.calculate_cleaning_efficiency(
                    self.pipeline_properties["debris_density"],
                    self.pipeline_properties["wall_condition"]
                )
                # Update gel depletion
                distance = pig.velocity * self.time_step
                pig.update_depletion(distance)
                
            elif isinstance(pig, IntelligentPig):
                # Intelligent pigs collect data rather than clean
                # Update battery life and data storage
                time_elapsed = self.time_step / 3600.0  # convert to hours
                distance_traveled = pig.velocity * self.time_step / 1000.0  # convert to km
                pig.update_battery_and_storage(time_elapsed, distance_traveled)
                
                # Record simulated inspection data at regular intervals
                if int(pig.position) % 10 == 0:  # Every 10 meters
                    # Simulate a random wall thickness with occasional defects
                    normal_thickness = 10.0  # mm
                    random_factor = 0.05 * (2 * np.random.random() - 1)  # ±5% variation
                    thickness = normal_thickness * (1 + random_factor)
                    
                    # Small chance of a defect
                    defect_present = np.random.random() < 0.02  # 2% chance
                    defect_type = None
                    defect_size = 0.0
                    
                    if defect_present:
                        defect_types = ["corrosion", "crack", "dent", "manufacturing defect"]
                        defect_type = defect_types[np.random.randint(0, len(defect_types))]
                        defect_size = 1.0 + 4.0 * np.random.random()  # 1-5mm defect
                    
                    pig.record_inspection_data(
                        pig.position, thickness, defect_present, defect_type, defect_size
                    )
            
            # Update pig position
            old_position = pig.position
            pig.update_position(self.time_step)
            
            # Check if pig has reached the end of the pipeline
            if pig.position >= self.pipeline_length:
                logger.info(f"Pig has reached the end of the pipeline at time {self.time:.1f}s")
                pig.position = self.pipeline_length  # Cap at pipeline length
                pig.velocity = 0.0
                
                # Record travel time
                self.results["pig_travel_times"].append(self.time)
            
            # Record pig state
            pig_state = {
                "type": type(pig).__name__,
                "position": pig.position,
                "velocity": pig.velocity,
                "differential_pressure": pig.differential_pressure,
                "cleaning_efficiency": cleaning_efficiency,
                "distance_traveled": pig.position - old_position
            }
            
            pig_states.append(pig_state)
            
            # Update simulation results
            self.results["pig_positions"].append(pig.position)
            self.results["pig_velocities"].append(pig.velocity)
            self.results["differential_pressures"].append(pig.differential_pressure)
            self.results["cleaning_efficiencies"].append(cleaning_efficiency)
            
            # Calculate energy used
            energy = pig.differential_pressure * self.flow_rate * self.time_step
            self.results["energy_consumption"].append(energy)
            
            # Calculate debris removal if applicable
            if cleaning_efficiency > 0:
                # Simple model: debris removal proportional to cleaning efficiency, 
                # velocity, time step, and pipe circumference
                pipe_circum = math.pi * local_diameter
                cleaned_area = pipe_circum * pig.velocity * self.time_step
                debris_thickness = 0.0001  # 0.1mm debris thickness
                debris_volume = cleaned_area * debris_thickness * cleaning_efficiency
                debris_mass = debris_volume * self.pipeline_properties["debris_density"]
                self.results["debris_removal"].append(debris_mass)
            else:
                self.results["debris_removal"].append(0.0)
        
        # Record time stamps
        self.results["time_stamps"].append(self.time)
        
        # Increment time
        self.time += self.time_step
        
        return {
            "time": self.time,
            "pigs": pig_states
        }
    
    def run_simulation(self, duration: float, callback: Callable = None):
        """
        Run the simulation for a specified duration.
        
        Args:
            duration (float): Duration to simulate in seconds
            callback (Callable, optional): Function to call after each step
            
        Returns:
            Dict: Simulation results
        """
        start_time = self.time
        end_time = start_time + duration
        
        # Clear results if starting from zero
        if start_time == 0:
            self.results = {
                "pig_positions": [],
                "pig_velocities": [],
                "differential_pressures": [],
                "cleaning_efficiencies": [],
                "time_stamps": [],
                "pig_travel_times": [],
                "debris_removal": [],
                "energy_consumption": [],
            }
        
        step_count = 0
        logger.info(f"Starting simulation for {duration}s duration")
        
        while self.time < end_time:
            # Run simulation step
            step_result = self.run_step()
            step_count += 1
            
            # Call callback if provided
            if callback and callable(callback):
                callback(step_result)
            
            # Check if all pigs have reached the end
            all_pigs_finished = all(pig.position >= self.pipeline_length for pig in self.pigs)
            if all_pigs_finished:
                logger.info(f"All pigs have reached the end of the pipeline at time {self.time:.1f}s")
                break
                
            # Log progress periodically
            if step_count % 100 == 0:
                logger.info(f"Simulation time: {self.time:.1f}s, Progress: {(self.time - start_time) / duration * 100:.1f}%")
        
        logger.info(f"Simulation completed: {step_count} steps, {self.time - start_time:.1f}s simulated")
        
        # Generate summary
        summary = self.get_simulation_summary()
        
        return {
            "results": self.results,
            "summary": summary
        }
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of simulation results.
        
        Returns:
            Dict[str, Any]: Simulation summary
        """
        if not self.results["time_stamps"]:
            return {"status": "No simulation data available"}
        
        # Calculate summary statistics
        avg_velocity = sum(self.results["pig_velocities"]) / max(1, len(self.results["pig_velocities"]))
        avg_pressure = sum(self.results["differential_pressures"]) / max(1, len(self.results["differential_pressures"]))
        avg_cleaning = sum(self.results["cleaning_efficiencies"]) / max(1, len(self.results["cleaning_efficiencies"]))
        total_debris = sum(self.results["debris_removal"])
        total_energy = sum(self.results["energy_consumption"])
        
        # Travel time
        if self.results["pig_travel_times"]:
            travel_time = self.results["pig_travel_times"][0]
        else:
            travel_time = None
        
        # Get final pig positions
        final_positions = []
        for pig in self.pigs:
            final_positions.append({
                "type": type(pig).__name__,
                "position": pig.position,
                "velocity": pig.velocity,
                "completed": pig.position >= self.pipeline_length
            })
        
        # Create summary
        summary = {
            "status": "Simulation data available",
            "time_simulated": self.time,
            "average_velocity": avg_velocity,
            "average_differential_pressure": avg_pressure,
            "average_cleaning_efficiency": avg_cleaning,
            "total_debris_removed": total_debris,
            "total_energy_consumed": total_energy,
            "pig_travel_time": travel_time,
            "final_pig_positions": final_positions,
        }
        
        # Add intelligent pig data if available
        for pig in self.pigs:
            if isinstance(pig, IntelligentPig):
                inspection_summary = pig.get_inspection_summary()
                summary["intelligent_pig_data"] = inspection_summary
        
        return summary
    
    def export_results(self, filepath: str) -> bool:
        """
        Export simulation results to CSV file.
        
        Args:
            filepath (str): File path to export to
            
        Returns:
            bool: Success status
        """
        try:
            import csv
            
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = ['time', 'position', 'velocity', 'differential_pressure', 
                             'cleaning_efficiency', 'debris_removal', 'energy']
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for i, t in enumerate(self.results["time_stamps"]):
                    if i < len(self.results["pig_positions"]):
                        writer.writerow({
                            'time': t,
                            'position': self.results["pig_positions"][i],
                            'velocity': self.results["pig_velocities"][i],
                            'differential_pressure': self.results["differential_pressures"][i],
                            'cleaning_efficiency': self.results["cleaning_efficiencies"][i],
                            'debris_removal': self.results["debris_removal"][i],
                            'energy': self.results["energy_consumption"][i]
                        })
            
            logger.info(f"Results exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return False
    
    def reset(self):
        """
        Reset the simulation to initial state.
        
        Returns:
            Dict: Initial state
        """
        self.time = 0.0
        self.pigs = []
        
        # Reset results
        self.results = {
            "pig_positions": [],
            "pig_velocities": [],
            "differential_pressures": [],
            "cleaning_efficiencies": [],
            "time_stamps": [],
            "pig_travel_times": [],
            "debris_removal": [],
            "energy_consumption": [],
        }
        
        logger.info("Simulation reset to initial state")
        
        return {
            "time": self.time,
            "pigs": []
        }