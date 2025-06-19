#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boundary conditions module for Openfoam_Simulator OpenFOAM integration.

This module provides functionality to:
- Create and manage boundary conditions for OpenFOAM simulations
- Support common boundary types used in oil & gas CFD applications
- Generate OpenFOAM-compatible boundary condition dictionary entries
- Apply boundary conditions to specific fields and regions
- Support custom industry-specific boundary conditions

The module is designed to work with the OpenFOAM case structure and provides
a high-level interface for setting up complex boundary conditions.
"""

import os
import re
import math
import json
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path

# Import utility modules
from ..utils.logger import get_logger
from ..config import get_value, set_value

# Set up module logger
logger = get_logger(__name__)


class BoundaryCondition:
    """Base class for all boundary conditions."""
    
    def __init__(self, name: str, type: str, patch_type: str = "patch"):
        """
        Initialize a boundary condition.
        
        Args:
            name (str): Name of the boundary patch
            type (str): OpenFOAM boundary condition type
            patch_type (str): OpenFOAM patch type (e.g., patch, wall, etc.)
        """
        self.name = name
        self.type = type
        self.patch_type = patch_type
        self.parameters = {}
    
    def add_parameter(self, key: str, value: Any):
        """
        Add a parameter to the boundary condition.
        
        Args:
            key (str): Parameter name
            value (Any): Parameter value
        """
        self.parameters[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the boundary condition to a dictionary format.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the boundary condition
        """
        bc_dict = {
            "name": self.name,
            "type": self.type,
            "patch_type": self.patch_type,
            "parameters": self.parameters
        }
        return bc_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BoundaryCondition':
        """
        Create a boundary condition from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing boundary condition data
            
        Returns:
            BoundaryCondition: Created boundary condition
        """
        bc = cls(data["name"], data["type"], data["patch_type"])
        bc.parameters = data.get("parameters", {})
        return bc
    
    def format_for_openfoam(self, field_name: str) -> str:
        """
        Format the boundary condition for inclusion in an OpenFOAM dictionary.
        
        Args:
            field_name (str): Name of the field this boundary condition applies to
            
        Returns:
            str: Formatted boundary condition entry
        """
        lines = [f"\t{self.name}",
                 "\t{",
                 f"\t\ttype\t\t\t{self.type};"]
        
        # Add parameters
        for key, value in self.parameters.items():
            if isinstance(value, str):
                lines.append(f"\t\t{key}\t\t\t{value};")
            elif isinstance(value, (int, float)):
                lines.append(f"\t\t{key}\t\t\t{value};")
            elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value):
                # Vector, tensor, or list of values
                if len(value) == 3:  # Assume it's a vector
                    vec_str = f"({value[0]} {value[1]} {value[2]})"
                    lines.append(f"\t\t{key}\t\t\t{vec_str};")
                elif len(value) == 9:  # Assume it's a tensor
                    tensor_str = f"({value[0]} {value[1]} {value[2]} {value[3]} {value[4]} {value[5]} {value[6]} {value[7]} {value[8]})"
                    lines.append(f"\t\t{key}\t\t\t{tensor_str};")
                else:  # Generic list
                    list_str = f"({' '.join(str(x) for x in value)})"
                    lines.append(f"\t\t{key}\t\t\t{list_str};")
            elif isinstance(value, dict):
                # Nested dictionary (e.g., for tabulated data)
                lines.append(f"\t\t{key}")
                lines.append("\t\t{")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (list, tuple)):
                        sub_str = f"({' '.join(str(x) for x in sub_value)})"
                        lines.append(f"\t\t\t{sub_key}\t\t{sub_str};")
                    else:
                        lines.append(f"\t\t\t{sub_key}\t\t{sub_value};")
                lines.append("\t\t}")
        
        lines.append("\t}")
        
        return '\n'.join(lines)


class VelocityBC(BoundaryCondition):
    """Boundary condition for velocity field."""
    
    def __init__(self, name: str, patch_type: str = "patch"):
        """
        Initialize a velocity boundary condition.
        
        Args:
            name (str): Name of the boundary patch
            patch_type (str): OpenFOAM patch type
        """
        super().__init__(name, "fixedValue", patch_type)
        self.add_parameter("value", "uniform (0 0 0)")
    
    def set_value(self, velocity: Union[Tuple[float, float, float], List[float]]):
        """
        Set the velocity value.
        
        Args:
            velocity: Velocity vector (x, y, z)
        """
        self.add_parameter("value", f"uniform ({velocity[0]} {velocity[1]} {velocity[2]})")
    
    def set_profile(self, profile_type: str, parameters: Dict[str, Any]):
        """
        Set a velocity profile.
        
        Args:
            profile_type (str): Type of profile (e.g., parabolic, turbulent)
            parameters (Dict[str, Any]): Profile parameters
        """
        self.type = "codedFixedValue"
        self.add_parameter("name", f"{profile_type}Profile")
        self.add_parameter("code", self._generate_profile_code(profile_type, parameters))
    
    def _generate_profile_code(self, profile_type: str, parameters: Dict[str, Any]) -> str:
        """
        Generate OpenFOAM code for the velocity profile.
        
        Args:
            profile_type (str): Type of profile
            parameters (Dict[str, Any]): Profile parameters
            
        Returns:
            str: Generated code
        """
        if profile_type == "parabolic":
            # Parabolic profile for pipe flow
            max_velocity = parameters.get("max_velocity", 1.0)
            radius = parameters.get("radius", 0.1)
            axis = parameters.get("axis", (0, 0, 1))
            
            # Generate code for parabolic profile
            code = f"""#include "codeStream.H"

            codeStream
            {{
                code
                {{
                    const vector axis({axis[0]}, {axis[1]}, {axis[2]});
                    const scalar maxVel = {max_velocity};
                    const scalar radius = {radius};
                    
                    const fvPatch& patch = this->patch();
                    const vectorField& Cf = patch.Cf();
                    vectorField& field = *this;
                    
                    // Calculate local coordinates
                    vector centerPoint(0, 0, 0);
                    forAll(Cf, i)
                    {{
                        centerPoint += Cf[i];
                    }}
                    centerPoint /= Cf.size();
                    
                    // Calculate profile
                    forAll(Cf, i)
                    {{
                        const vector r = Cf[i] - centerPoint;
                        const scalar rMag = mag(r - (r & axis)*axis);
                        const scalar u = maxVel * (1.0 - sqr(rMag/radius));
                        field[i] = axis * max(0.0, u);
                    }}
                }}
            }};
            """
            return code
            
        elif profile_type == "turbulent":
            # Turbulent profile (power law)
            bulk_velocity = parameters.get("bulk_velocity", 1.0)
            radius = parameters.get("radius", 0.1)
            exponent = parameters.get("exponent", 1/7)  # 1/7 power law
            axis = parameters.get("axis", (0, 0, 1))
            
            # Generate code for turbulent profile
            code = f"""#include "codeStream.H"

            codeStream
            {{
                code
                {{
                    const vector axis({axis[0]}, {axis[1]}, {axis[2]});
                    const scalar bulkVel = {bulk_velocity};
                    const scalar radius = {radius};
                    const scalar n = {exponent};
                    
                    const fvPatch& patch = this->patch();
                    const vectorField& Cf = patch.Cf();
                    vectorField& field = *this;
                    
                    // Calculate local coordinates
                    vector centerPoint(0, 0, 0);
                    forAll(Cf, i)
                    {{
                        centerPoint += Cf[i];
                    }}
                    centerPoint /= Cf.size();
                    
                    // Calculate profile
                    forAll(Cf, i)
                    {{
                        const vector r = Cf[i] - centerPoint;
                        const scalar rMag = mag(r - (r & axis)*axis);
                        const scalar u = bulkVel * pow(1.0 - rMag/radius, n);
                        field[i] = axis * max(0.0, u);
                    }}
                }}
            }};
            """
            return code
            
        else:
            # Default to uniform profile
            velocity = parameters.get("velocity", (0, 0, 0))
            return f"return tensor(0, 0, 0, 0, 0, 0, 0, 0, 0);"


class PressureBC(BoundaryCondition):
    """Boundary condition for pressure field."""
    
    def __init__(self, name: str, patch_type: str = "patch"):
        """
        Initialize a pressure boundary condition.
        
        Args:
            name (str): Name of the boundary patch
            patch_type (str): OpenFOAM patch type
        """
        super().__init__(name, "fixedValue", patch_type)
        self.add_parameter("value", "uniform 0")
    
    def set_value(self, pressure: float):
        """
        Set the pressure value.
        
        Args:
            pressure (float): Pressure value
        """
        self.add_parameter("value", f"uniform {pressure}")
    
    def set_gradient(self, gradient: float = 0.0):
        """
        Set a fixed pressure gradient.
        
        Args:
            gradient (float): Pressure gradient value
        """
        self.type = "fixedGradient"
        self.add_parameter("gradient", f"uniform {gradient}")
    
    def set_outlet(self, reference_pressure: float = 0.0):
        """
        Set as a pressure outlet.
        
        Args:
            reference_pressure (float): Reference pressure value
        """
        self.type = "fixedValue" if reference_pressure != 0.0 else "zeroGradient"
        if reference_pressure != 0.0:
            self.add_parameter("value", f"uniform {reference_pressure}")


class WallBC(BoundaryCondition):
    """Wall boundary condition for various fields."""
    
    def __init__(self, name: str):
        """
        Initialize a wall boundary condition.
        
        Args:
            name (str): Name of the wall patch
        """
        super().__init__(name, "noSlip", "wall")
    
    def set_velocity_condition(self, condition: str = "noSlip", 
                              moving_wall_velocity: Tuple[float, float, float] = None):
        """
        Set the velocity condition at the wall.
        
        Args:
            condition (str): Wall condition (noSlip, slip, movingWall)
            moving_wall_velocity: Velocity for moving wall
        """
        if condition == "noSlip":
            self.type = "noSlip"
        elif condition == "slip":
            self.type = "slip"
        elif condition == "movingWall" and moving_wall_velocity is not None:
            self.type = "fixedValue"
            self.add_parameter("value", f"uniform ({moving_wall_velocity[0]} {moving_wall_velocity[1]} {moving_wall_velocity[2]})")
    
    def set_thermal_condition(self, condition: str = "zeroGradient", 
                             temperature: float = None,
                             heat_flux: float = None):
        """
        Set the thermal condition at the wall.
        
        Args:
            condition (str): Thermal condition (zeroGradient, fixedValue, fixedGradient)
            temperature (float, optional): Fixed temperature value
            heat_flux (float, optional): Fixed heat flux value
        """
        self.thermal_type = condition
        
        if condition == "fixedValue" and temperature is not None:
            self.thermal_parameters = {"value": f"uniform {temperature}"}
        elif condition == "fixedGradient" and heat_flux is not None:
            self.thermal_parameters = {"gradient": f"uniform {heat_flux}"}
        else:
            self.thermal_parameters = {}
    
    def format_for_openfoam(self, field_name: str) -> str:
        """
        Format the boundary condition for inclusion in an OpenFOAM dictionary.
        
        Args:
            field_name (str): Name of the field this boundary condition applies to
            
        Returns:
            str: Formatted boundary condition entry
        """
        # Different handling based on field name
        if field_name == "U":
            # Velocity field
            return super().format_for_openfoam(field_name)
        elif field_name == "T" or field_name == "h" or field_name == "e":
            # Temperature or energy field
            if hasattr(self, 'thermal_type'):
                # Create temporary BC for thermal
                thermal_bc = BoundaryCondition(self.name, self.thermal_type, "wall")
                for key, value in getattr(self, 'thermal_parameters', {}).items():
                    thermal_bc.add_parameter(key, value)
                return thermal_bc.format_for_openfoam(field_name)
            else:
                # Default to zero gradient if not specified
                thermal_bc = BoundaryCondition(self.name, "zeroGradient", "wall")
                return thermal_bc.format_for_openfoam(field_name)
        elif field_name == "p" or field_name == "p_rgh":
            # Pressure field - default to zero gradient for walls
            pressure_bc = BoundaryCondition(self.name, "zeroGradient", "wall")
            return pressure_bc.format_for_openfoam(field_name)
        elif field_name.startswith("alpha."):
            # Phase fraction - default to zero gradient for walls
            alpha_bc = BoundaryCondition(self.name, "zeroGradient", "wall")
            return alpha_bc.format_for_openfoam(field_name)
        elif field_name == "k" or field_name == "epsilon" or field_name == "omega":
            # Turbulence fields - use wall functions
            turb_bc = BoundaryCondition(self.name, f"kqRWallFunction" if field_name == "k" else 
                                        "epsilonWallFunction" if field_name == "epsilon" else
                                        "omegaWallFunction", "wall")
            turb_bc.add_parameter("value", "uniform 0.1")
            return turb_bc.format_for_openfoam(field_name)
        else:
            # Default to zero gradient for unknown fields
            default_bc = BoundaryCondition(self.name, "zeroGradient", "wall")
            return default_bc.format_for_openfoam(field_name)


class InletBC(BoundaryCondition):
    """Inlet boundary condition for various fields."""
    
    def __init__(self, name: str):
        """
        Initialize an inlet boundary condition.
        
        Args:
            name (str): Name of the inlet patch
        """
        super().__init__(name, "fixedValue", "patch")
    
    def set_velocity(self, velocity_type: str = "uniform", 
                    value: Union[float, Tuple[float, float, float]] = None,
                    **kwargs):
        """
        Set the velocity at the inlet.
        
        Args:
            velocity_type (str): Type of velocity specification
            value: Velocity value
            **kwargs: Additional parameters for specific velocity types
        """
        if velocity_type == "uniform" and isinstance(value, (list, tuple)):
            # Uniform velocity vector
            self.velocity_type = "fixedValue"
            self.velocity_parameters = {"value": f"uniform ({value[0]} {value[1]} {value[2]})"}
        
        elif velocity_type == "flowRate":
            # Volumetric flow rate
            flow_rate = value if isinstance(value, (int, float)) else kwargs.get("flow_rate", 1.0)
            area = kwargs.get("area", 1.0)
            direction = kwargs.get("direction", (0, 0, 1))
            
            # Calculate velocity magnitude
            velocity_mag = flow_rate / area
            
            # Normalize direction vector
            dir_mag = math.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
            if dir_mag > 0:
                normalized_dir = (direction[0]/dir_mag, direction[1]/dir_mag, direction[2]/dir_mag)
            else:
                normalized_dir = (0, 0, 1)  # Default to z-direction
            
            # Calculate velocity vector
            velocity = (
                normalized_dir[0] * velocity_mag,
                normalized_dir[1] * velocity_mag,
                normalized_dir[2] * velocity_mag
            )
            
            self.velocity_type = "fixedValue"
            self.velocity_parameters = {"value": f"uniform ({velocity[0]} {velocity[1]} {velocity[2]})"}
        
        elif velocity_type == "profile":
            # Velocity profile
            profile_type = kwargs.get("profile_type", "parabolic")
            profile_params = kwargs.get("profile_params", {})
            
            self.velocity_type = "codedFixedValue"
            self.velocity_parameters = {
                "name": f"{profile_type}VelocityProfile",
                "code": self._generate_velocity_profile_code(profile_type, profile_params)
            }
        
        elif velocity_type == "massFlowRate":
            # Mass flow rate with density
            mass_flow = value if isinstance(value, (int, float)) else kwargs.get("mass_flow", 1.0)
            density = kwargs.get("density", 1000.0)  # Default to water
            area = kwargs.get("area", 1.0)
            direction = kwargs.get("direction", (0, 0, 1))
            
            # Calculate velocity magnitude
            velocity_mag = mass_flow / (density * area)
            
            # Normalize direction vector
            dir_mag = math.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
            if dir_mag > 0:
                normalized_dir = (direction[0]/dir_mag, direction[1]/dir_mag, direction[2]/dir_mag)
            else:
                normalized_dir = (0, 0, 1)  # Default to z-direction
            
            # Calculate velocity vector
            velocity = (
                normalized_dir[0] * velocity_mag,
                normalized_dir[1] * velocity_mag,
                normalized_dir[2] * velocity_mag
            )
            
            self.velocity_type = "fixedValue"
            self.velocity_parameters = {"value": f"uniform ({velocity[0]} {velocity[1]} {velocity[2]})"}
    
    def set_pressure(self, pressure_type: str = "zeroGradient", value: float = None):
        """
        Set the pressure at the inlet.
        
        Args:
            pressure_type (str): Type of pressure specification
            value (float, optional): Pressure value for fixedValue
        """
        self.pressure_type = pressure_type
        
        if pressure_type == "fixedValue" and value is not None:
            self.pressure_parameters = {"value": f"uniform {value}"}
        else:
            self.pressure_parameters = {}
    
    def set_temperature(self, temperature: float):
        """
        Set the temperature at the inlet.
        
        Args:
            temperature (float): Temperature value
        """
        self.temperature_type = "fixedValue"
        self.temperature_parameters = {"value": f"uniform {temperature}"}
    
    def set_phase_fraction(self, phase_name: str, fraction: float):
        """
        Set the phase fraction at the inlet for multiphase simulations.
        
        Args:
            phase_name (str): Name of the phase
            fraction (float): Volume fraction (0-1)
        """
        setattr(self, f"alpha_{phase_name}_type", "fixedValue")
        setattr(self, f"alpha_{phase_name}_parameters", {"value": f"uniform {fraction}"})
    
    def set_turbulence(self, k: float = 0.1, epsilon: float = 0.01, 
                      intensity: float = None, length_scale: float = None):
        """
        Set the turbulence properties at the inlet.
        
        Args:
            k (float): Turbulent kinetic energy
            epsilon (float): Turbulent dissipation rate
            intensity (float, optional): Turbulence intensity (0-1)
            length_scale (float, optional): Turbulent length scale
        """
        # Store basic values
        self.k_type = "fixedValue"
        self.epsilon_type = "fixedValue"
        
        # Calculate from intensity and length scale if provided
        if intensity is not None and length_scale is not None:
            # Assuming a reference velocity of 1 m/s for simplicity
            # This should be replaced with actual inlet velocity for accuracy
            ref_velocity = 1.0
            
            # Calculate k from intensity
            k = 1.5 * (intensity * ref_velocity)**2
            
            # Calculate epsilon from k and length scale using C_mu = 0.09
            c_mu = 0.09
            epsilon = c_mu**0.75 * k**1.5 / length_scale
        
        self.k_parameters = {"value": f"uniform {k}"}
        self.epsilon_parameters = {"value": f"uniform {epsilon}"}
    
    def _generate_velocity_profile_code(self, profile_type: str, parameters: Dict[str, Any]) -> str:
        """
        Generate OpenFOAM code for the velocity profile.
        
        Args:
            profile_type (str): Type of profile
            parameters (Dict[str, Any]): Profile parameters
            
        Returns:
            str: Generated code
        """
        if profile_type == "parabolic":
            max_velocity = parameters.get("max_velocity", 1.0)
            radius = parameters.get("radius", 0.1)
            axis = parameters.get("axis", (0, 0, 1))
            
            # Code for parabolic profile
            # (Same as in VelocityBC._generate_profile_code)
            code = f"""#include "codeStream.H"

            codeStream
            {{
                code
                {{
                    const vector axis({axis[0]}, {axis[1]}, {axis[2]});
                    const scalar maxVel = {max_velocity};
                    const scalar radius = {radius};
                    
                    const fvPatch& patch = this->patch();
                    const vectorField& Cf = patch.Cf();
                    vectorField& field = *this;
                    
                    // Calculate local coordinates
                    vector centerPoint(0, 0, 0);
                    forAll(Cf, i)
                    {{
                        centerPoint += Cf[i];
                    }}
                    centerPoint /= Cf.size();
                    
                    // Calculate profile
                    forAll(Cf, i)
                    {{
                        const vector r = Cf[i] - centerPoint;
                        const scalar rMag = mag(r - (r & axis)*axis);
                        const scalar u = maxVel * (1.0 - sqr(rMag/radius));
                        field[i] = axis * max(0.0, u);
                    }}
                }}
            }};
            """
            return code
        else:
            return """#include "codeStream.H"

            codeStream
            {
                code
                {
                    // Default to uniform profile
                    const fvPatch& patch = this->patch();
                    vectorField& field = *this;
                    field = vector(0, 0, 1);
                }
            };"""
    
    def format_for_openfoam(self, field_name: str) -> str:
        """
        Format the boundary condition for inclusion in an OpenFOAM dictionary.
        
        Args:
            field_name (str): Name of the field this boundary condition applies to
            
        Returns:
            str: Formatted boundary condition entry
        """
        # Different handling based on field name
        if field_name == "U":
            # Velocity field
            if hasattr(self, 'velocity_type'):
                velocity_bc = BoundaryCondition(self.name, self.velocity_type, "patch")
                for key, value in getattr(self, 'velocity_parameters', {}).items():
                    velocity_bc.add_parameter(key, value)
                return velocity_bc.format_for_openfoam(field_name)
            else:
                # Default to fixed zero if not specified
                velocity_bc = BoundaryCondition(self.name, "fixedValue", "patch")
                velocity_bc.add_parameter("value", "uniform (0 0 0)")
                return velocity_bc.format_for_openfoam(field_name)
                
        elif field_name == "p" or field_name == "p_rgh":
            # Pressure field
            if hasattr(self, 'pressure_type'):
                pressure_bc = BoundaryCondition(self.name, self.pressure_type, "patch")
                for key, value in getattr(self, 'pressure_parameters', {}).items():
                    pressure_bc.add_parameter(key, value)
                return pressure_bc.format_for_openfoam(field_name)
            else:
                # Default to zero gradient for inlets
                pressure_bc = BoundaryCondition(self.name, "zeroGradient", "patch")
                return pressure_bc.format_for_openfoam(field_name)
                
        elif field_name == "T" or field_name == "h" or field_name == "e":
            # Temperature or energy field
            if hasattr(self, 'temperature_type'):
                temp_bc = BoundaryCondition(self.name, self.temperature_type, "patch")
                for key, value in getattr(self, 'temperature_parameters', {}).items():
                    temp_bc.add_parameter(key, value)
                return temp_bc.format_for_openfoam(field_name)
            else:
                # Default to a reference temperature if not specified
                temp_bc = BoundaryCondition(self.name, "fixedValue", "patch")
                temp_bc.add_parameter("value", "uniform 300")  # 300K default
                return temp_bc.format_for_openfoam(field_name)
                
        elif field_name.startswith("alpha."):
            # Phase fraction
            phase_name = field_name.split('.')[1]
            if hasattr(self, f'alpha_{phase_name}_type'):
                alpha_bc = BoundaryCondition(self.name, getattr(self, f'alpha_{phase_name}_type'), "patch")
                for key, value in getattr(self, f'alpha_{phase_name}_parameters', {}).items():
                    alpha_bc.add_parameter(key, value)
                return alpha_bc.format_for_openfoam(field_name)
            else:
                # Default to 0 or 1 based on primary phase
                value = 1.0 if phase_name == "water" else 0.0
                alpha_bc = BoundaryCondition(self.name, "fixedValue", "patch")
                alpha_bc.add_parameter("value", f"uniform {value}")
                return alpha_bc.format_for_openfoam(field_name)
                
        elif field_name == "k":
            # Turbulent kinetic energy
            if hasattr(self, 'k_type'):
                k_bc = BoundaryCondition(self.name, self.k_type, "patch")
                for key, value in getattr(self, 'k_parameters', {}).items():
                    k_bc.add_parameter(key, value)
                return k_bc.format_for_openfoam(field_name)
            else:
                # Default value for k
                k_bc = BoundaryCondition(self.name, "fixedValue", "patch")
                k_bc.add_parameter("value", "uniform 0.1")
                return k_bc.format_for_openfoam(field_name)
                
        elif field_name == "epsilon":
            # Turbulent dissipation rate
            if hasattr(self, 'epsilon_type'):
                epsilon_bc = BoundaryCondition(self.name, self.epsilon_type, "patch")
                for key, value in getattr(self, 'epsilon_parameters', {}).items():
                    epsilon_bc.add_parameter(key, value)
                return epsilon_bc.format_for_openfoam(field_name)
            else:
                # Default value for epsilon
                epsilon_bc = BoundaryCondition(self.name, "fixedValue", "patch")
                epsilon_bc.add_parameter("value", "uniform 0.01")
                return epsilon_bc.format_for_openfoam(field_name)
                
        elif field_name == "omega":
            # Specific dissipation rate for k-omega models
            if hasattr(self, 'k_parameters') and hasattr(self, 'epsilon_parameters'):
                # Calculate omega from k and epsilon using k-omega relationship
                # omega = epsilon / (C_mu * k), where C_mu = 0.09
                k_value = float(getattr(self, 'k_parameters', {}).get("value", "uniform 0.1").split()[-1])
                epsilon_value = float(getattr(self, 'epsilon_parameters', {}).get("value", "uniform 0.01").split()[-1])
                c_mu = 0.09
                omega_value = epsilon_value / (c_mu * k_value)
                
                omega_bc = BoundaryCondition(self.name, "fixedValue", "patch")
                omega_bc.add_parameter("value", f"uniform {omega_value}")
                return omega_bc.format_for_openfoam(field_name)
            else:
                # Default value for omega
                omega_bc = BoundaryCondition(self.name, "fixedValue", "patch")
                omega_bc.add_parameter("value", "uniform 1.0")
                return omega_bc.format_for_openfoam(field_name)
        else:
            # Default handling for other fields
            default_bc = BoundaryCondition(self.name, "fixedValue", "patch")
            default_bc.add_parameter("value", "uniform 0")
            return default_bc.format_for_openfoam(field_name)


class OutletBC(BoundaryCondition):
    """Outlet boundary condition for various fields."""
    
    def __init__(self, name: str):
        """
        Initialize an outlet boundary condition.
        
        Args:
            name (str): Name of the outlet patch
        """
        super().__init__(name, "zeroGradient", "patch")
    
    def set_pressure(self, pressure_type: str = "fixedValue", value: float = 0.0):
        """
        Set the pressure at the outlet.
        
        Args:
            pressure_type (str): Type of pressure specification
            value (float): Pressure value for fixedValue
        """
        self.pressure_type = pressure_type
        
        if pressure_type == "fixedValue":
            self.pressure_parameters = {"value": f"uniform {value}"}
        else:
            self.pressure_parameters = {}
    
    def set_velocity(self, velocity_type: str = "zeroGradient"):
        """
        Set the velocity condition at the outlet.
        
        Args:
            velocity_type (str): Type of velocity specification
        """
        self.velocity_type = velocity_type
        self.velocity_parameters = {}
        
        if velocity_type == "inletOutlet":
            # Special handling for inletOutlet condition
            # This switches between zeroGradient and fixedValue based on flow direction
            self.velocity_parameters = {
                "inletValue": "uniform (0 0 0)",
                "value": "uniform (0 0 0)"
            }
    
    def set_phase_fraction(self, phase_name: str, outlet_type: str = "zeroGradient"):
        """
        Set the phase fraction condition at the outlet for multiphase simulations.
        
        Args:
            phase_name (str): Name of the phase
            outlet_type (str): Type of outlet condition
        """
        setattr(self, f"alpha_{phase_name}_type", outlet_type)
        
        if outlet_type == "inletOutlet":
            setattr(self, f"alpha_{phase_name}_parameters", {
                "inletValue": "uniform 0",
                "value": "uniform 0"
            })
        else:
            setattr(self, f"alpha_{phase_name}_parameters", {})
    
    def format_for_openfoam(self, field_name: str) -> str:
        """
        Format the boundary condition for inclusion in an OpenFOAM dictionary.
        
        Args:
            field_name (str): Name of the field this boundary condition applies to
            
        Returns:
            str: Formatted boundary condition entry
        """
        # Different handling based on field name
        if field_name == "U":
            # Velocity field
            if hasattr(self, 'velocity_type'):
                velocity_bc = BoundaryCondition(self.name, self.velocity_type, "patch")
                for key, value in getattr(self, 'velocity_parameters', {}).items():
                    velocity_bc.add_parameter(key, value)
                return velocity_bc.format_for_openfoam(field_name)
            else:
                # Default to zero gradient
                velocity_bc = BoundaryCondition(self.name, "zeroGradient", "patch")
                return velocity_bc.format_for_openfoam(field_name)
                
        elif field_name == "p" or field_name == "p_rgh":
            # Pressure field
            if hasattr(self, 'pressure_type'):
                pressure_bc = BoundaryCondition(self.name, self.pressure_type, "patch")
                for key, value in getattr(self, 'pressure_parameters', {}).items():
                    pressure_bc.add_parameter(key, value)
                return pressure_bc.format_for_openfoam(field_name)
            else:
                # Default to fixed value for outlets
                pressure_bc = BoundaryCondition(self.name, "fixedValue", "patch")
                pressure_bc.add_parameter("value", "uniform 0")
                return pressure_bc.format_for_openfoam(field_name)
                
        elif field_name.startswith("alpha."):
            # Phase fraction
            phase_name = field_name.split('.')[1]
            if hasattr(self, f'alpha_{phase_name}_type'):
                alpha_bc = BoundaryCondition(self.name, getattr(self, f'alpha_{phase_name}_type'), "patch")
                for key, value in getattr(self, f'alpha_{phase_name}_parameters', {}).items():
                    alpha_bc.add_parameter(key, value)
                return alpha_bc.format_for_openfoam(field_name)
            else:
                # Default to zero gradient
                alpha_bc = BoundaryCondition(self.name, "zeroGradient", "patch")
                return alpha_bc.format_for_openfoam(field_name)
                
        elif field_name in ["k", "epsilon", "omega", "nuTilda"]:
            # Turbulence fields
            turb_bc = BoundaryCondition(self.name, "zeroGradient", "patch")
            return turb_bc.format_for_openfoam(field_name)
                
        elif field_name in ["T", "h", "e"]:
            # Temperature/energy fields
            temp_bc = BoundaryCondition(self.name, "zeroGradient", "patch")
            return temp_bc.format_for_openfoam(field_name)
        
        else:
            # Default handling for other fields
            default_bc = BoundaryCondition(self.name, "zeroGradient", "patch")
            return default_bc.format_for_openfoam(field_name)


class SymmetryBC(BoundaryCondition):
    """Symmetry boundary condition."""
    
    def __init__(self, name: str):
        """
        Initialize a symmetry boundary condition.
        
        Args:
            name (str): Name of the symmetry patch
        """
        super().__init__(name, "symmetry", "symmetry")
    
    def format_for_openfoam(self, field_name: str) -> str:
        """
        Format the boundary condition for inclusion in an OpenFOAM dictionary.
        
        Args:
            field_name (str): Name of the field this boundary condition applies to
            
        Returns:
            str: Formatted boundary condition entry
        """
        # Symmetry is the same for all fields
        return f"\t{self.name}\n\t{{\n\t\ttype\t\t\t{self.type};\n\t}}"


class CyclicBC(BoundaryCondition):
    """Cyclic (periodic) boundary condition."""
    
    def __init__(self, name: str, neighbor_patch: str):
        """
        Initialize a cyclic boundary condition.
        
        Args:
            name (str): Name of the cyclic patch
            neighbor_patch (str): Name of the matching neighbor patch
        """
        super().__init__(name, "cyclic", "cyclic")
        self.add_parameter("neighbourPatch", neighbor_patch)
    
    def format_for_openfoam(self, field_name: str) -> str:
        """
        Format the boundary condition for inclusion in an OpenFOAM dictionary.
        
        Args:
            field_name (str): Name of the field this boundary condition applies to
            
        Returns:
            str: Formatted boundary condition entry
        """
        # Cyclic is the same for all fields
        return super().format_for_openfoam(field_name)


class BoundaryManager:
    """
    Class to manage boundary conditions for OpenFOAM cases.
    
    This class provides methods to create, modify, and write boundary conditions
    for OpenFOAM simulations.
    """
    
    def __init__(self, case_dir: str = None):
        """
        Initialize the boundary manager.
        
        Args:
            case_dir (str, optional): Path to the OpenFOAM case directory
        """
        self.case_dir = case_dir
        self.boundaries = {}  # Dictionary of boundary conditions by name
        self.field_defaults = {}  # Default boundary conditions by field
    
    def set_case_directory(self, case_dir: str):
        """
        Set the OpenFOAM case directory.
        
        Args:
            case_dir (str): Path to the OpenFOAM case directory
        """
        self.case_dir = case_dir
    
    def add_boundary(self, boundary: BoundaryCondition):
        """
        Add a boundary condition.
        
        Args:
            boundary (BoundaryCondition): Boundary condition to add
        """
        self.boundaries[boundary.name] = boundary
    
    def get_boundary(self, name: str) -> Optional[BoundaryCondition]:
        """
        Get a boundary condition by name.
        
        Args:
            name (str): Name of the boundary condition
            
        Returns:
            Optional[BoundaryCondition]: The boundary condition, or None if not found
        """
        return self.boundaries.get(name)
    
    def set_field_default(self, field_name: str, boundary_type: str):
        """
        Set a default boundary condition type for a field.
        
        Args:
            field_name (str): Name of the field
            boundary_type (str): Default boundary condition type
        """
        self.field_defaults[field_name] = boundary_type
    
    def create_inlet(self, name: str, **kwargs) -> InletBC:
        """
        Create and add an inlet boundary condition.
        
        Args:
            name (str): Name of the inlet patch
            **kwargs: Additional parameters for the inlet
            
        Returns:
            InletBC: The created inlet boundary condition
        """
        inlet = InletBC(name)
        
        # Set velocity if provided
        if "velocity" in kwargs:
            inlet.set_velocity("uniform", kwargs["velocity"])
        elif "flow_rate" in kwargs:
            inlet.set_velocity("flowRate", kwargs["flow_rate"],
                              area=kwargs.get("area", 1.0),
                              direction=kwargs.get("direction", (0, 0, 1)))
        elif "velocity_profile" in kwargs:
            profile_type = kwargs.get("profile_type", "parabolic")
            profile_params = kwargs.get("profile_params", {})
            inlet.set_velocity("profile", None, profile_type=profile_type, profile_params=profile_params)
        
        # Set pressure if provided
        if "pressure" in kwargs:
            inlet.set_pressure("fixedValue", kwargs["pressure"])
        
        # Set temperature if provided
        if "temperature" in kwargs:
            inlet.set_temperature(kwargs["temperature"])
        
        # Set turbulence if provided
        if "k" in kwargs and "epsilon" in kwargs:
            inlet.set_turbulence(kwargs["k"], kwargs["epsilon"])
        elif "turbulence_intensity" in kwargs and "turbulence_length_scale" in kwargs:
            inlet.set_turbulence(intensity=kwargs["turbulence_intensity"], length_scale=kwargs["turbulence_length_scale"])
        
        # Set phase fractions for multiphase simulations
        if "phases" in kwargs:
            for phase_name, fraction in kwargs["phases"].items():
                inlet.set_phase_fraction(phase_name, fraction)
        
        # Add to boundaries dictionary
        self.add_boundary(inlet)
        return inlet
    
    def create_outlet(self, name: str, **kwargs) -> OutletBC:
        """
        Create and add an outlet boundary condition.
        
        Args:
            name (str): Name of the outlet patch
            **kwargs: Additional parameters for the outlet
            
        Returns:
            OutletBC: The created outlet boundary condition
        """
        outlet = OutletBC(name)
        
        # Set pressure if provided
        if "pressure" in kwargs:
            outlet.set_pressure("fixedValue", kwargs["pressure"])
        else:
            # Default to fixed value of 0
            outlet.set_pressure()
        
        # Set velocity condition if provided
        if "velocity_type" in kwargs:
            outlet.set_velocity(kwargs["velocity_type"])
        
        # Set phase fractions for multiphase simulations
        if "phases" in kwargs:
            for phase_name, outlet_type in kwargs["phases"].items():
                outlet.set_phase_fraction(phase_name, outlet_type)
        
        # Add to boundaries dictionary
        self.add_boundary(outlet)
        return outlet
    
    def create_wall(self, name: str, **kwargs) -> WallBC:
        """
        Create and add a wall boundary condition.
        
        Args:
            name (str): Name of the wall patch
            **kwargs: Additional parameters for the wall
            
        Returns:
            WallBC: The created wall boundary condition
        """
        wall = WallBC(name)
        
        # Set velocity condition if provided
        if "velocity_condition" in kwargs:
            if kwargs["velocity_condition"] == "movingWall" and "velocity" in kwargs:
                wall.set_velocity_condition("movingWall", kwargs["velocity"])
            else:
                wall.set_velocity_condition(kwargs["velocity_condition"])
        
        # Set thermal condition if provided
        if "thermal_condition" in kwargs:
            if kwargs["thermal_condition"] == "fixedValue" and "temperature" in kwargs:
                wall.set_thermal_condition("fixedValue", kwargs["temperature"])
            elif kwargs["thermal_condition"] == "fixedGradient" and "heat_flux" in kwargs:
                wall.set_thermal_condition("fixedGradient", None, kwargs["heat_flux"])
            else:
                wall.set_thermal_condition(kwargs["thermal_condition"])
        
        # Add to boundaries dictionary
        self.add_boundary(wall)
        return wall
    
    def create_symmetry(self, name: str) -> SymmetryBC:
        """
        Create and add a symmetry boundary condition.
        
        Args:
            name (str): Name of the symmetry patch
            
        Returns:
            SymmetryBC: The created symmetry boundary condition
        """
        symmetry = SymmetryBC(name)
        self.add_boundary(symmetry)
        return symmetry
    
    def create_cyclic(self, name: str, neighbor_patch: str) -> CyclicBC:
        """
        Create and add a cyclic boundary condition.
        
        Args:
            name (str): Name of the cyclic patch
            neighbor_patch (str): Name of the matching neighbor patch
            
        Returns:
            CyclicBC: The created cyclic boundary condition
        """
        cyclic = CyclicBC(name, neighbor_patch)
        self.add_boundary(cyclic)
        return cyclic
    
    def create_custom_boundary(self, name: str, bc_type: str, patch_type: str, **kwargs) -> BoundaryCondition:
        """
        Create and add a custom boundary condition.
        
        Args:
            name (str): Name of the boundary patch
            bc_type (str): OpenFOAM boundary condition type
            patch_type (str): OpenFOAM patch type
            **kwargs: Additional parameters for the boundary condition
            
        Returns:
            BoundaryCondition: The created boundary condition
        """
        boundary = BoundaryCondition(name, bc_type, patch_type)
        
        # Add parameters from kwargs
        for key, value in kwargs.items():
            boundary.add_parameter(key, value)
        
        self.add_boundary(boundary)
        return boundary
    
    def load_from_case(self) -> bool:
        """
        Load boundary conditions from the OpenFOAM case.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.case_dir:
            logger.error("No case directory set. Cannot load boundaries.")
            return False
        
        try:
            # Clear existing boundaries
            self.boundaries = {}
            
            # Load from constant/polyMesh/boundary file
            boundary_file = os.path.join(self.case_dir, "constant", "polyMesh", "boundary")
            if not os.path.exists(boundary_file):
                logger.error(f"Boundary file not found: {boundary_file}")
                return False
            
            # Parse boundary file
            with open(boundary_file, 'r') as f:
                content = f.read()
            
            # Extract boundary entries
            # This is a simplified parser and might not handle all cases correctly
            boundary_pattern = r'(\w+)\s*\n\s*{\s*\n\s*type\s+(\w+);\s*\n(?:[^}]*\n)*\s*}'
            matches = re.findall(boundary_pattern, content)
            
            for name, patch_type in matches:
                # Create appropriate boundary condition based on patch type
                if patch_type == "wall":
                    self.add_boundary(WallBC(name))
                elif patch_type == "patch":
                    # Check if it's an inlet or outlet by looking at field files
                    if self._is_inlet(name):
                        self.add_boundary(InletBC(name))
                    else:
                        self.add_boundary(OutletBC(name))
                elif patch_type == "symmetry":
                    self.add_boundary(SymmetryBC(name))
                elif patch_type == "cyclic":
                    # Need to find neighbor patch
                    neighbor_pattern = rf'{name}\s*\n\s*{{[^}}]*neighbourPatch\s+(\w+);[^}}]*}}'
                    neighbor_match = re.search(neighbor_pattern, content)
                    neighbor = neighbor_match.group(1) if neighbor_match else "unknown"
                    self.add_boundary(CyclicBC(name, neighbor))
                else:
                    # Generic boundary condition
                    self.add_boundary(BoundaryCondition(name, "zeroGradient", patch_type))
            
            # Load field-specific boundary conditions
            self._load_field_conditions()
            
            logger.info(f"Loaded {len(self.boundaries)} boundary conditions from case")
            return True
            
        except Exception as e:
            logger.error(f"Error loading boundary conditions: {e}")
            return False
    
    def _is_inlet(self, patch_name: str) -> bool:
        """
        Check if a patch is likely an inlet by examining velocity boundary conditions.
        
        Args:
            patch_name (str): Name of the patch to check
            
        Returns:
            bool: True if the patch appears to be an inlet, False otherwise
        """
        # Look in the U file for fixedValue conditions
        u_file = os.path.join(self.case_dir, "0", "U")
        if not os.path.exists(u_file):
            return False
        
        try:
            with open(u_file, 'r') as f:
                content = f.read()
            
            # Check for fixedValue on this patch
            pattern = rf'{patch_name}\s*\n\s*{{\s*\n\s*type\s+fixedValue;'
            return re.search(pattern, content) is not None
            
        except Exception:
            return False
    
    def _load_field_conditions(self):
        """
        Load field-specific boundary conditions from the case.
        """
        # Look in the 0 directory for field files
        zero_dir = os.path.join(self.case_dir, "0")
        if not os.path.exists(zero_dir):
            return
        
        # Common fields to check
        fields = ["U", "p", "p_rgh", "T", "k", "epsilon", "omega"]
        
        # Add alpha fields for multiphase
        alpha_files = [f for f in os.listdir(zero_dir) if f.startswith("alpha.")]
        fields.extend(alpha_files)
        
        # Process each field
        for field in fields:
            field_file = os.path.join(zero_dir, field)
            if not os.path.exists(field_file):
                continue
            
            try:
                with open(field_file, 'r') as f:
                    content = f.read()
                
                # Extract boundary entries for this field
                # This is a simplified parser and might not handle all cases correctly
                for boundary_name, boundary in self.boundaries.items():
                    # Look for this boundary in the field file
                    pattern = rf'{boundary_name}\s*\n\s*{{\s*\n\s*type\s+(\w+);([^}}]*)}}'
                    match = re.search(pattern, content)
                    
                    if match:
                        bc_type = match.group(1)
                        bc_content = match.group(2)
                        
                        # Store field-specific information in the boundary object
                        # This depends on the specific boundary condition classes
                        if isinstance(boundary, InletBC) and field == "U":
                            boundary.velocity_type = bc_type
                            # Parse parameters
                            value_match = re.search(r'value\s+uniform\s+\(([^)]+)\);', bc_content)
                            if value_match:
                                value_str = value_match.group(1)
                                boundary.velocity_parameters = {"value": f"uniform ({value_str})"}
                        
                        elif isinstance(boundary, OutletBC) and field == "p":
                            boundary.pressure_type = bc_type
                            # Parse parameters
                            value_match = re.search(r'value\s+uniform\s+([^;]+);', bc_content)
                            if value_match:
                                value_str = value_match.group(1)
                                boundary.pressure_parameters = {"value": f"uniform {value_str}"}
                        
                        # Add more field-specific parsing as needed
                        
            except Exception as e:
                logger.warning(f"Error parsing field {field}: {e}")
    
    def write_to_case(self, fields: List[str]) -> bool:
        """
        Write boundary conditions to the OpenFOAM case.
        
        Args:
            fields (List[str]): List of field names to write boundary conditions for
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.case_dir:
            logger.error("No case directory set. Cannot write boundaries.")
            return False
        
        try:
            # Ensure the 0 directory exists
            zero_dir = os.path.join(self.case_dir, "0")
            os.makedirs(zero_dir, exist_ok=True)
            
            # Write boundary conditions for each field
            for field in fields:
                self._write_field_boundary_conditions(field)
            
            logger.info(f"Wrote boundary conditions for {len(fields)} fields")
            return True
            
        except Exception as e:
            logger.error(f"Error writing boundary conditions: {e}")
            return False
    
    def _write_field_boundary_conditions(self, field_name: str):
        """
        Write boundary conditions for a specific field.
        
        Args:
            field_name (str): Name of the field
        """
        # Field file path
        field_file = os.path.join(self.case_dir, "0", field_name)
        
        # Check if the file exists
        file_exists = os.path.exists(field_file)
        
        # Read existing file or create a template
        if file_exists:
            with open(field_file, 'r') as f:
                content = f.read()
            
            # Extract dimensions and internalField
            dimensions_match = re.search(r'dimensions\s+\[([^]]+)\];', content)
            dimensions = dimensions_match.group(1) if dimensions_match else "0 0 0 0 0 0 0"
            
            internal_field_match = re.search(r'internalField\s+(\w+)\s+([^;]+);', content)
            if internal_field_match:
                internal_type = internal_field_match.group(1)
                internal_value = internal_field_match.group(2)
                internal_field = f"{internal_type} {internal_value}"
            else:
                # Default internal field
                if field_name == "U":
                    internal_field = "uniform (0 0 0)"
                elif field_name in ["p", "p_rgh"]:
                    internal_field = "uniform 0"
                elif field_name == "T":
                    internal_field = "uniform 300"
                elif field_name == "k":
                    internal_field = "uniform 0.1"
                elif field_name == "epsilon":
                    internal_field = "uniform 0.01"
                elif field_name == "omega":
                    internal_field = "uniform 1.0"
                elif field_name.startswith("alpha."):
                    phase = field_name.split(".")[1]
                    internal_field = f"uniform {1.0 if phase == 'water' else 0.0}"
                else:
                    internal_field = "uniform 0"
        else:
            # Create default dimensions and internalField
            if field_name == "U":
                dimensions = "0 1 -1 0 0 0 0"  # m/s
                internal_field = "uniform (0 0 0)"
            elif field_name in ["p", "p_rgh"]:
                dimensions = "0 2 -2 0 0 0 0"  # m²/s²
                internal_field = "uniform 0"
            elif field_name == "T":
                dimensions = "0 0 0 1 0 0 0"  # K
                internal_field = "uniform 300"
            elif field_name == "k":
                dimensions = "0 2 -2 0 0 0 0"  # m²/s²
                internal_field = "uniform 0.1"
            elif field_name == "epsilon":
                dimensions = "0 2 -3 0 0 0 0"  # m²/s³
                internal_field = "uniform 0.01"
            elif field_name == "omega":
                dimensions = "0 0 -1 0 0 0 0"  # 1/s
                internal_field = "uniform 1.0"
            elif field_name.startswith("alpha."):
                dimensions = "0 0 0 0 0 0 0"  # dimensionless
                phase = field_name.split(".")[1]
                internal_field = f"uniform {1.0 if phase == 'water' else 0.0}"
            else:
                dimensions = "0 0 0 0 0 0 0"  # dimensionless
                internal_field = "uniform 0"
        
        # Create the field file content
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/

FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      {field_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""
        
        # Set appropriate class for vector fields
        if field_name == "U":
            content = content.replace("volScalarField", "volVectorField")
        
        # Add dimensions and internalField
        content += f"dimensions      [{dimensions}];\n\n"
        content += f"internalField   {internal_field};\n\n"
        
        # Add boundary conditions
        content += "boundaryField\n{\n"
        
        # Add each boundary condition
        for name, bc in self.boundaries.items():
            content += bc.format_for_openfoam(field_name) + "\n\n"
        
        # Close the boundaryField dictionary
        content += "}\n\n"
        content += "// ************************************************************************* //\n"
        
        # Write the file
        with open(field_file, 'w') as f:
            f.write(content)


# Create a singleton instance
_boundary_manager = None

def get_boundary_manager():
    """
    Get the BoundaryManager singleton instance.
    
    Returns:
        BoundaryManager: The BoundaryManager instance
    """
    global _boundary_manager
    if _boundary_manager is None:
        _boundary_manager = BoundaryManager()
    
    return _boundary_manager