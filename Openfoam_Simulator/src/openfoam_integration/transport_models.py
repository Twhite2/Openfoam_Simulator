#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transport models configuration for Openfoam_Simulator OpenFOAM integration.

This module provides classes and functions to generate and configure
transport properties for OpenFOAM simulations, including:
- Newtonian and non-Newtonian fluid models
- Single and multi-phase transport properties
- Temperature-dependent properties
- Special models for oil & gas applications

The module integrates with the material database to provide physical
properties for common fluids in the oil & gas industry.
"""

import os
import sys
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

# Add parent directory to path to allow imports from other modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from models.material_database import MaterialDatabase
except ImportError:
    # Placeholder for when importing from the actual module is not possible
    class MaterialDatabase:
        """Placeholder for MaterialDatabase if import fails."""
        @staticmethod
        def get_fluid_properties(name: str) -> Dict[str, Any]:
            """Get properties of a fluid by name."""
            # Default properties for common fluids
            fluids = {
                "water": {
                    "density": 1000.0,
                    "viscosity": 0.001,
                    "specific_heat": 4182.0,
                    "thermal_conductivity": 0.6
                },
                "oil": {
                    "density": 850.0,
                    "viscosity": 0.03,
                    "specific_heat": 1800.0,
                    "thermal_conductivity": 0.15
                },
                "natural_gas": {
                    "density": 0.8,
                    "viscosity": 1.8e-5,
                    "specific_heat": 2200.0,
                    "thermal_conductivity": 0.026
                },
                "air": {
                    "density": 1.2,
                    "viscosity": 1.8e-5,
                    "specific_heat": 1005.0,
                    "thermal_conductivity": 0.025
                }
            }
            
            # Default to water if fluid not found
            return fluids.get(name.lower(), fluids["water"])

# Import logger
try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    # Default logger if import fails
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)


class TransportModel:
    """
    Base class for transport models.
    
    This class provides a common interface for all transport models
    and includes methods for writing OpenFOAM dictionary entries.
    """
    
    def __init__(self, name: str = "default"):
        """
        Initialize the transport model.
        
        Args:
            name (str): Name of the transport model
        """
        self.name = name
        self.properties = {}
    
    def generate_dict_entry(self) -> str:
        """
        Generate OpenFOAM dictionary entry for this transport model.
        
        Returns:
            str: OpenFOAM dictionary entry as string
        """
        # To be implemented by subclasses
        return ""
    
    def write_to_file(self, file_path: Union[str, Path]) -> bool:
        """
        Write the transport model to a file.
        
        Args:
            file_path (Union[str, Path]): Path to the file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(file_path, 'w') as f:
                f.write(self.generate_dict_entry())
            return True
        except Exception as e:
            logger.error(f"Error writing transport model to file: {e}")
            return False
    
    @staticmethod
    def create_dictionary_header() -> str:
        """
        Create OpenFOAM dictionary file header.
        
        Returns:
            str: OpenFOAM dictionary header
        """
        return """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/

FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""


class NewtonianTransport(TransportModel):
    """
    Newtonian transport model for single-phase flows.
    
    This class represents the standard Newtonian transport model where
    the viscosity is constant and independent of shear rate.
    """
    
    def __init__(self, name: str = "Newtonian", density: float = 1000.0, 
                viscosity: float = 1e-3, temperature: float = 293.15):
        """
        Initialize the Newtonian transport model.
        
        Args:
            name (str): Name of the transport model
            density (float): Fluid density in kg/m³
            viscosity (float): Dynamic viscosity in Pa·s
            temperature (float): Temperature in K
        """
        super().__init__(name)
        self.properties = {
            "density": density,
            "viscosity": viscosity,
            "temperature": temperature
        }
    
    def generate_dict_entry(self) -> str:
        """
        Generate OpenFOAM dictionary entry for Newtonian transport.
        
        Returns:
            str: OpenFOAM dictionary entry
        """
        # Calculate kinematic viscosity (nu) from dynamic viscosity and density
        nu = self.properties["viscosity"] / self.properties["density"]
        
        entry = self.create_dictionary_header()
        entry += f"""
transportModel  {self.name};

// Kinematic viscosity [m^2/s]
nu              {nu:.8e};

// Density [kg/m^3]
rho             {self.properties["density"]:.2f};

// Reference temperature [K]
TRef            {self.properties["temperature"]:.2f};

// ************************************************************************* //
"""
        return entry


class NonNewtonianTransport(TransportModel):
    """
    Non-Newtonian transport model for flows with shear-dependent viscosity.
    
    This class represents various non-Newtonian models such as:
    - Power law
    - Cross model
    - Carreau model
    - Herschel-Bulkley
    Commonly used for heavy oil and polymer flows.
    """
    
    # Non-Newtonian model types
    POWER_LAW = "powerLaw"
    CROSS = "crossPowerLaw"
    CARREAU = "Carreau"
    HERSCHEL_BULKLEY = "HerschelBulkley"
    
    def __init__(self, model_type: str = POWER_LAW, density: float = 1000.0, 
                k: float = 1.0, n: float = 0.8, temperature: float = 293.15,
                yield_stress: float = 0.0, viscosity_inf: float = 1e-4, 
                viscosity_0: float = 1.0, critical_stress: float = 0.0):
        """
        Initialize the non-Newtonian transport model.
        
        Args:
            model_type (str): Type of non-Newtonian model
            density (float): Fluid density in kg/m³
            k (float): Consistency index (Pa·s^n for power law)
            n (float): Power law index
            temperature (float): Temperature in K
            yield_stress (float): Yield stress for Herschel-Bulkley (Pa)
            viscosity_inf (float): Infinite shear viscosity (Pa·s)
            viscosity_0 (float): Zero shear viscosity (Pa·s)
            critical_stress (float): Critical stress (Pa)
        """
        super().__init__("nonNewtonian")
        self.model_type = model_type
        self.properties = {
            "density": density,
            "k": k,
            "n": n,
            "temperature": temperature,
            "yield_stress": yield_stress,
            "viscosity_inf": viscosity_inf,
            "viscosity_0": viscosity_0,
            "critical_stress": critical_stress
        }
    
    def generate_dict_entry(self) -> str:
        """
        Generate OpenFOAM dictionary entry for non-Newtonian transport.
        
        Returns:
            str: OpenFOAM dictionary entry
        """
        entry = self.create_dictionary_header()
        entry += f"""
transportModel  {self.name};

{self.name}Coeffs
{{
    model       {self.model_type};
    rho         {self.properties["density"]:.2f};
"""
        
        # Add model-specific parameters
        if self.model_type == self.POWER_LAW:
            entry += f"""
    k           {self.properties["k"]:.8e};
    n           {self.properties["n"]:.8e};
"""
        elif self.model_type == self.CROSS:
            entry += f"""
    nu0         {self.properties["viscosity_0"]:.8e};
    nuInf       {self.properties["viscosity_inf"]:.8e};
    m           {self.properties["k"]:.8e};
    n           {self.properties["n"]:.8e};
"""
        elif self.model_type == self.CARREAU:
            entry += f"""
    nu0         {self.properties["viscosity_0"]:.8e};
    nuInf       {self.properties["viscosity_inf"]:.8e};
    lambda      {self.properties["k"]:.8e};
    n           {self.properties["n"]:.8e};
"""
        elif self.model_type == self.HERSCHEL_BULKLEY:
            entry += f"""
    k           {self.properties["k"]:.8e};
    n           {self.properties["n"]:.8e};
    tau0        {self.properties["yield_stress"]:.8e};
    nu0         {self.properties["viscosity_0"]:.8e};
"""
        
        entry += f"""
}}

// Reference temperature [K]
TRef            {self.properties["temperature"]:.2f};

// ************************************************************************* //
"""
        return entry


class MultiPhaseTransport(TransportModel):
    """
    Multi-phase transport model for immiscible fluids.
    
    This class represents transport models for multi-phase flows like:
    - Oil-water flows
    - Gas-liquid flows
    - Three-phase flows (gas-oil-water)
    Includes interface properties like surface tension.
    """
    
    def __init__(self, phases: List[Dict[str, Any]], phase_names: Optional[List[str]] = None):
        """
        Initialize the multi-phase transport model.
        
        Args:
            phases (List[Dict[str, Any]]): List of phase properties
            phase_names (Optional[List[str]]): List of phase names
        """
        super().__init__("multiphase")
        
        self.phases = phases
        
        # Generate phase names if not provided
        if phase_names is None:
            self.phase_names = [f"phase{i+1}" for i in range(len(phases))]
        else:
            self.phase_names = phase_names[:len(phases)]
            if len(phase_names) < len(phases):
                # Add generic names for any additional phases
                self.phase_names.extend([f"phase{i+1+len(phase_names)}" for i in range(len(phases) - len(phase_names))])
        
        # Interface properties for each phase pair
        self.interface_properties = {}
    
    def set_interface_property(self, phase1: str, phase2: str, 
                              property_name: str, value: float) -> None:
        """
        Set interface property between two phases.
        
        Args:
            phase1 (str): First phase name
            phase2 (str): Second phase name
            property_name (str): Property name (e.g., "sigma" for surface tension)
            value (float): Property value
        """
        # Sort phase names to ensure consistent key
        key = tuple(sorted([phase1, phase2]))
        
        if key not in self.interface_properties:
            self.interface_properties[key] = {}
        
        self.interface_properties[key][property_name] = value
    
    def set_surface_tension(self, phase1: str, phase2: str, 
                           sigma: float) -> None:
        """
        Set surface tension between two phases.
        
        Args:
            phase1 (str): First phase name
            phase2 (str): Second phase name
            sigma (float): Surface tension in N/m
        """
        self.set_interface_property(phase1, phase2, "sigma", sigma)
    
    def generate_dict_entry(self) -> str:
        """
        Generate OpenFOAM dictionary entry for multi-phase transport.
        
        Returns:
            str: OpenFOAM dictionary entry
        """
        entry = self.create_dictionary_header()
        
        # Add phases section
        entry += "phases\n(\n"
        
        for i, (phase_name, phase) in enumerate(zip(self.phase_names, self.phases)):
            model_type = phase.get("transportModel", "Newtonian")
            entry += f"    {phase_name}\n    {{\n"
            entry += f"        transportModel  {model_type};\n"
            
            # Add density and viscosity
            density = phase.get("density", 1000.0)
            viscosity = phase.get("viscosity", 0.001)
            nu = viscosity / density
            
            entry += f"        nu              {nu:.8e};\n"
            entry += f"        rho             {density:.2f};\n"
            
            # Add any additional properties
            for key, value in phase.items():
                if key not in ["transportModel", "density", "viscosity"]:
                    if isinstance(value, float):
                        entry += f"        {key}             {value:.8e};\n"
                    else:
                        entry += f"        {key}             {value};\n"
            
            entry += "    }\n\n"
        
        entry += ");\n\n"
        
        # Add interface properties
        for (phase1, phase2), properties in self.interface_properties.items():
            for property_name, value in properties.items():
                entry += f"{property_name}"
                if len(self.phases) > 2:
                    # For more than 2 phases, specify the phases
                    entry += f".{phase1}.{phase2}"
                entry += f"            {value:.8e};\n"
        
        entry += "\n// ************************************************************************* //\n"
        return entry


class ThermalTransport(TransportModel):
    """
    Transport model with temperature-dependent properties.
    
    This class represents transport models where fluid properties
    (viscosity, density, etc.) depend on temperature, which is important
    for flows with significant temperature variations.
    """
    
    def __init__(self, density_ref: float = 1000.0, viscosity_ref: float = 1e-3,
                thermal_conductivity: float = 0.6, specific_heat: float = 4200.0,
                temperature_ref: float = 293.15, thermal_expansion: float = 2e-4,
                viscosity_model: str = "constant"):
        """
        Initialize the thermal transport model.
        
        Args:
            density_ref (float): Reference density in kg/m³
            viscosity_ref (float): Reference dynamic viscosity in Pa·s
            thermal_conductivity (float): Thermal conductivity in W/(m·K)
            specific_heat (float): Specific heat capacity in J/(kg·K)
            temperature_ref (float): Reference temperature in K
            thermal_expansion (float): Thermal expansion coefficient in 1/K
            viscosity_model (str): Viscosity temperature model
        """
        super().__init__("thermal")
        self.properties = {
            "density_ref": density_ref,
            "viscosity_ref": viscosity_ref,
            "thermal_conductivity": thermal_conductivity,
            "specific_heat": specific_heat,
            "temperature_ref": temperature_ref,
            "thermal_expansion": thermal_expansion,
            "viscosity_model": viscosity_model
        }
        
        # Coefficients for viscosity model
        self.viscosity_coeffs = []
    
    def set_viscosity_model(self, model: str, coefficients: List[float]) -> None:
        """
        Set viscosity temperature model.
        
        Args:
            model (str): Model type (constant, polynomial, Sutherland, etc.)
            coefficients (List[float]): Model coefficients
        """
        self.properties["viscosity_model"] = model
        self.viscosity_coeffs = coefficients
    
    def generate_dict_entry(self) -> str:
        """
        Generate OpenFOAM dictionary entry for thermal transport.
        
        Returns:
            str: OpenFOAM dictionary entry
        """
        # Calculate kinematic viscosity
        nu_ref = self.properties["viscosity_ref"] / self.properties["density_ref"]
        
        entry = self.create_dictionary_header()
        entry += f"""
transportModel  Newtonian;

// Thermal properties
DT              {self.properties["thermal_conductivity"] / (self.properties["density_ref"] * self.properties["specific_heat"]):.8e};
Pr              {self.properties["viscosity_ref"] * self.properties["specific_heat"] / self.properties["thermal_conductivity"]:.4f};
Prt             0.85;

// Kinematic viscosity
nu              {nu_ref:.8e};

// Density
rho             {self.properties["density_ref"]:.2f};

// Reference values
TRef            {self.properties["temperature_ref"]:.2f};
beta            {self.properties["thermal_expansion"]:.8e};

// Specific heat capacity
Cp              {self.properties["specific_heat"]:.2f};
"""
        
        # Add viscosity model if not constant
        if self.properties["viscosity_model"] != "constant" and self.viscosity_coeffs:
            entry += f"""
// Temperature dependent viscosity model
viscosityModel  {self.properties["viscosity_model"]};
"""
            
            if self.properties["viscosity_model"] == "polynomial":
                entry += "viscosityCoeffs\n{\n"
                for i, coeff in enumerate(self.viscosity_coeffs):
                    entry += f"    a{i}          {coeff:.8e};\n"
                entry += "}\n"
            elif self.properties["viscosity_model"] == "Sutherland":
                entry += f"""
viscosityCoeffs
{{
    As          {self.viscosity_coeffs[0]:.8e};
    Ts          {self.viscosity_coeffs[1]:.2f};
}}
"""
        
        entry += "\n// ************************************************************************* //\n"
        return entry


class OilGasTransport(TransportModel):
    """
    Specialized transport model for oil & gas applications.
    
    This class includes industry-specific models and properties:
    - PVT (Pressure-Volume-Temperature) relationships
    - Black oil model
    - Compositional model
    - Asphaltene and wax precipitation
    """
    
    # Model types
    BLACK_OIL = "blackOil"
    COMPOSITIONAL = "compositional"
    
    def __init__(self, model_type: str = BLACK_OIL, oil_density: float = 850.0, 
                oil_viscosity: float = 0.03, gas_density: float = 0.8,
                gas_viscosity: float = 1.8e-5, water_density: float = 1000.0,
                water_viscosity: float = 0.001, temperature: float = 293.15,
                pressure: float = 101325.0):
        """
        Initialize the oil & gas transport model.
        
        Args:
            model_type (str): Type of oil & gas model
            oil_density (float): Oil density in kg/m³
            oil_viscosity (float): Oil viscosity in Pa·s
            gas_density (float): Gas density in kg/m³
            gas_viscosity (float): Gas viscosity in Pa·s
            water_density (float): Water density in kg/m³
            water_viscosity (float): Water viscosity in Pa·s
            temperature (float): Temperature in K
            pressure (float): Pressure in Pa
        """
        super().__init__("oilGas")
        self.model_type = model_type
        self.properties = {
            "oil_density": oil_density,
            "oil_viscosity": oil_viscosity,
            "gas_density": gas_density,
            "gas_viscosity": gas_viscosity,
            "water_density": water_density,
            "water_viscosity": water_viscosity,
            "temperature": temperature,
            "pressure": pressure
        }
        
        # PVT properties
        self.pvt_properties = {}
        
        # Surface tension values
        self.surface_tension_ow = 0.025  # Oil-water
        self.surface_tension_go = 0.023  # Gas-oil
        self.surface_tension_gw = 0.072  # Gas-water
    
    def set_pvt_property(self, property_name: str, value: Any) -> None:
        """
        Set a PVT property.
        
        Args:
            property_name (str): Property name
            value (Any): Property value
        """
        self.pvt_properties[property_name] = value
    
    def set_black_oil_properties(self, gas_oil_ratio: float, 
                                oil_formation_volume_factor: float,
                                gas_formation_volume_factor: float,
                                gas_solubility: float) -> None:
        """
        Set black oil model properties.
        
        Args:
            gas_oil_ratio (float): Gas-oil ratio in m³/m³
            oil_formation_volume_factor (float): Oil formation volume factor
            gas_formation_volume_factor (float): Gas formation volume factor
            gas_solubility (float): Gas solubility in oil
        """
        self.pvt_properties.update({
            "GOR": gas_oil_ratio,
            "Bo": oil_formation_volume_factor,
            "Bg": gas_formation_volume_factor,
            "Rs": gas_solubility
        })
    
    def set_surface_tensions(self, oil_water: float = 0.025, 
                            gas_oil: float = 0.023, 
                            gas_water: float = 0.072) -> None:
        """
        Set surface tension values.
        
        Args:
            oil_water (float): Oil-water surface tension in N/m
            gas_oil (float): Gas-oil surface tension in N/m
            gas_water (float): Gas-water surface tension in N/m
        """
        self.surface_tension_ow = oil_water
        self.surface_tension_go = gas_oil
        self.surface_tension_gw = gas_water
    
    def generate_dict_entry(self) -> str:
        """
        Generate OpenFOAM dictionary entry for oil & gas transport.
        
        Returns:
            str: OpenFOAM dictionary entry
        """
        entry = self.create_dictionary_header()
        entry += f"""
// Oil & Gas Transport Properties

// Model type
transportModel  {self.model_type};

phases
(
    oil
    {{
        transportModel  Newtonian;
        nu              {self.properties["oil_viscosity"] / self.properties["oil_density"]:.8e};
        rho             {self.properties["oil_density"]:.2f};
    }}

    water
    {{
        transportModel  Newtonian;
        nu              {self.properties["water_viscosity"] / self.properties["water_density"]:.8e};
        rho             {self.properties["water_density"]:.2f};
    }}

    gas
    {{
        transportModel  Newtonian;
        nu              {self.properties["gas_viscosity"] / self.properties["gas_density"]:.8e};
        rho             {self.properties["gas_density"]:.2f};
    }}
);

// Surface tensions between phases
sigma.oil.water    {self.surface_tension_ow:.8e};
sigma.gas.oil      {self.surface_tension_go:.8e};
sigma.gas.water    {self.surface_tension_gw:.8e};

// Reference pressure and temperature
pRef                {self.properties["pressure"]:.2f};
TRef                {self.properties["temperature"]:.2f};
"""
        
        # Add PVT properties if model is black oil
        if self.model_type == self.BLACK_OIL and self.pvt_properties:
            entry += "\n// Black Oil PVT Properties\n"
            entry += "blackOilCoeffs\n{\n"
            
            for key, value in self.pvt_properties.items():
                if isinstance(value, float):
                    entry += f"    {key}         {value:.8e};\n"
                else:
                    entry += f"    {key}         {value};\n"
            
            entry += "}\n"
        
        # Add compositional properties
        elif self.model_type == self.COMPOSITIONAL:
            entry += "\n// Compositional Model Properties\n"
            entry += "compositionalCoeffs\n{\n"
            entry += "    // Component properties would be defined here\n"
            entry += "}\n"
        
        entry += "\n// ************************************************************************* //\n"
        return entry


class PiggingTransport(TransportModel):
    """
    Transport model for pipeline pigging simulations.
    
    This class represents transport models for simulating pipeline
    pigs moving through a fluid-filled pipeline, including:
    - Pig-fluid interaction
    - Bypass flow
    - Friction and drag
    """
    
    def __init__(self, fluid_density: float = 1000.0, fluid_viscosity: float = 0.001,
                pig_density: float = 300.0, pig_diameter: float = 0.1, 
                pipeline_diameter: float = 0.1016, bypass_fraction: float = 0.05,
                friction_coefficient: float = 0.3):
        """
        Initialize the pigging transport model.
        
        Args:
            fluid_density (float): Fluid density in kg/m³
            fluid_viscosity (float): Fluid viscosity in Pa·s
            pig_density (float): Pig material density in kg/m³
            pig_diameter (float): Pig diameter in m
            pipeline_diameter (float): Pipeline diameter in m
            bypass_fraction (float): Fraction of flow bypassing the pig
            friction_coefficient (float): Friction coefficient between pig and pipe
        """
        super().__init__("pigging")
        self.properties = {
            "fluid_density": fluid_density,
            "fluid_viscosity": fluid_viscosity,
            "pig_density": pig_density,
            "pig_diameter": pig_diameter,
            "pipeline_diameter": pipeline_diameter,
            "bypass_fraction": bypass_fraction,
            "friction_coefficient": friction_coefficient
        }
    
    def generate_dict_entry(self) -> str:
        """
        Generate OpenFOAM dictionary entry for pigging transport.
        
        Returns:
            str: OpenFOAM dictionary entry
        """
        # Calculate kinematic viscosity
        nu = self.properties["fluid_viscosity"] / self.properties["fluid_density"]
        
        entry = self.create_dictionary_header()
        entry += f"""
// Pigging Transport Properties

// Fluid properties
transportModel  Newtonian;
nu              {nu:.8e};
rho             {self.properties["fluid_density"]:.2f};

// Pig properties
piggingCoeffs
{{
    pigDensity       {self.properties["pig_density"]:.2f};
    pigDiameter      {self.properties["pig_diameter"]:.6f};
    pipelineDiameter {self.properties["pipeline_diameter"]:.6f};
    bypassFraction   {self.properties["bypass_fraction"]:.6f};
    frictionCoeff    {self.properties["friction_coefficient"]:.6f};
    
    // Differential pressure across the pig [Pa]
    deltaPCoeff      100.0;
    
    // Added mass coefficient
    addedMassCoeff   0.5;
}}

// ************************************************************************* //
"""
        return entry


class SpillTransport(TransportModel):
    """
    Transport model for spill simulations.
    
    This class represents transport models for simulating the spread
    and transport of spilled fluids, including:
    - Spreading on surfaces
    - Evaporation and weathering
    - Environmental effects
    """
    
    # Spill types
    SURFACE_SPILL = "surfaceSpill"
    SUBSURFACE_SPILL = "subsurfaceSpill"
    JET_SPILL = "jetSpill"
    
    # Fluid types
    CRUDE_OIL = "crudeOil"
    DIESEL = "diesel"
    GASOLINE = "gasoline"
    NATURAL_GAS = "naturalGas"
    
    def __init__(self, spill_type: str = SURFACE_SPILL, fluid_type: str = CRUDE_OIL,
                fluid_density: float = 850.0, fluid_viscosity: float = 0.03,
                surface_tension: float = 0.025, evaporation_rate: float = 0.0001,
                environment_type: str = "water", wind_speed: float = 5.0,
                current_speed: float = 0.5, temperature: float = 293.15):
        """
        Initialize the spill transport model.
        
        Args:
            spill_type (str): Type of spill
            fluid_type (str): Type of fluid
            fluid_density (float): Fluid density in kg/m³
            fluid_viscosity (float): Fluid viscosity in Pa·s
            surface_tension (float): Surface tension in N/m
            evaporation_rate (float): Evaporation rate in kg/(m²·s)
            environment_type (str): Environment type (water, soil, etc.)
            wind_speed (float): Wind speed in m/s
            current_speed (float): Current speed in m/s (for water environment)
            temperature (float): Temperature in K
        """
        super().__init__("spill")
        self.spill_type = spill_type
        self.fluid_type = fluid_type
        self.environment_type = environment_type
        self.properties = {
            "fluid_density": fluid_density,
            "fluid_viscosity": fluid_viscosity,
            "surface_tension": surface_tension,
            "evaporation_rate": evaporation_rate,
            "wind_speed": wind_speed,
            "current_speed": current_speed,
            "temperature": temperature
        }
        
        # Additional fluid properties
        self.fluid_properties = {}
    
    def set_fluid_properties(self, properties: Dict[str, Any]) -> None:
        """
        Set additional fluid properties.
        
        Args:
            properties (Dict[str, Any]): Dictionary of fluid properties
        """
        self.fluid_properties.update(properties)
    
    def generate_dict_entry(self) -> str:
        """
        Generate OpenFOAM dictionary entry for spill transport.
        
        Returns:
            str: OpenFOAM dictionary entry
        """
        # Calculate kinematic viscosity
        nu = self.properties["fluid_viscosity"] / self.properties["fluid_density"]
        
        entry = self.create_dictionary_header()
        entry += f"""
// Spill Transport Properties

// Model type
transportModel  Newtonian;
spill
{{
    type            {self.spill_type};
    fluid           {self.fluid_type};
    environment     {self.environment_type};
}}

// Basic fluid properties
nu              {nu:.8e};
rho             {self.properties["fluid_density"]:.2f};
sigma           {self.properties["surface_tension"]:.8e};

// Environmental conditions
environmentalCoeffs
{{
    temperature     {self.properties["temperature"]:.2f};
    windSpeed       {self.properties["wind_speed"]:.2f};
    currentSpeed    {self.properties["current_speed"]:.2f};
}}

// Spill behavior models
spillModels
{{
    // Whether to include weathering
    weathering      on;
    
    // Evaporation model
    evaporation     on;
    evaporationRate {self.properties["evaporation_rate"]:.8e};
    
    // Dispersion model
    dispersion      on;
    
    // Emulsification model
    emulsification  on;
}}
"""
        
        # Add additional fluid properties if any
        if self.fluid_properties:
            entry += "\n// Additional fluid properties\n"
            entry += "fluidProperties\n{\n"
            
            for key, value in self.fluid_properties.items():
                if isinstance(value, float):
                    entry += f"    {key}         {value:.8e};\n"
                else:
                    entry += f"    {key}         {value};\n"
            
            entry += "}\n"
        
        entry += "\n// ************************************************************************* //\n"
        return entry


def create_transport_properties(case_dir: Union[str, Path], model_type: str, **kwargs) -> bool:
    """
    Create transportProperties dictionary file for a specific model type.
    
    Args:
        case_dir (Union[str, Path]): OpenFOAM case directory
        model_type (str): Type of transport model
        **kwargs: Additional arguments for the specific model
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Ensure case_dir is a Path object
    if isinstance(case_dir, str):
        case_dir = Path(case_dir)
    
    # Create constant directory if it doesn't exist
    constant_dir = case_dir / "constant"
    constant_dir.mkdir(parents=True, exist_ok=True)
    
    # Path to transportProperties file
    transport_file = constant_dir / "transportProperties"
    
    # Create appropriate transport model based on type
    model = None
    
    if model_type == "newtonian":
        model = NewtonianTransport(
            density=kwargs.get("density", 1000.0),
            viscosity=kwargs.get("viscosity", 0.001),
            temperature=kwargs.get("temperature", 293.15)
        )
    
    elif model_type == "nonNewtonian":
        model = NonNewtonianTransport(
            model_type=kwargs.get("nonNewtonian_type", NonNewtonianTransport.POWER_LAW),
            density=kwargs.get("density", 1000.0),
            k=kwargs.get("k", 1.0),
            n=kwargs.get("n", 0.8),
            temperature=kwargs.get("temperature", 293.15),
            yield_stress=kwargs.get("yield_stress", 0.0),
            viscosity_inf=kwargs.get("viscosity_inf", 1e-4),
            viscosity_0=kwargs.get("viscosity_0", 1.0),
            critical_stress=kwargs.get("critical_stress", 0.0)
        )
    
    elif model_type == "multiphase":
        # Extract phases from kwargs
        phases = kwargs.get("phases", [])
        phase_names = kwargs.get("phase_names", None)
        
        if not phases:
            # Default to oil-water system
            phases = [
                {
                    "transportModel": "Newtonian",
                    "density": 1000.0,
                    "viscosity": 0.001
                },
                {
                    "transportModel": "Newtonian",
                    "density": 850.0,
                    "viscosity": 0.03
                }
            ]
            phase_names = ["water", "oil"]
        
        model = MultiPhaseTransport(phases, phase_names)
        
        # Set surface tension
        if len(phases) == 2:
            model.set_surface_tension(phase_names[0], phase_names[1], 
                                     kwargs.get("surface_tension", 0.025))
        elif len(phases) > 2:
            # For three phase, set all interface tensions
            model.set_surface_tension(phase_names[0], phase_names[1], 
                                     kwargs.get("surface_tension_01", 0.025))
            model.set_surface_tension(phase_names[0], phase_names[2], 
                                     kwargs.get("surface_tension_02", 0.072))
            model.set_surface_tension(phase_names[1], phase_names[2], 
                                     kwargs.get("surface_tension_12", 0.023))
    
    elif model_type == "thermal":
        model = ThermalTransport(
            density_ref=kwargs.get("density", 1000.0),
            viscosity_ref=kwargs.get("viscosity", 0.001),
            thermal_conductivity=kwargs.get("thermal_conductivity", 0.6),
            specific_heat=kwargs.get("specific_heat", 4200.0),
            temperature_ref=kwargs.get("temperature", 293.15),
            thermal_expansion=kwargs.get("thermal_expansion", 2e-4)
        )
        
        # Set viscosity model if provided
        visc_model = kwargs.get("viscosity_model")
        visc_coeffs = kwargs.get("viscosity_coeffs")
        if visc_model and visc_coeffs:
            model.set_viscosity_model(visc_model, visc_coeffs)
    
    elif model_type == "oilGas":
        model = OilGasTransport(
            model_type=kwargs.get("oilGas_type", OilGasTransport.BLACK_OIL),
            oil_density=kwargs.get("oil_density", 850.0),
            oil_viscosity=kwargs.get("oil_viscosity", 0.03),
            gas_density=kwargs.get("gas_density", 0.8),
            gas_viscosity=kwargs.get("gas_viscosity", 1.8e-5),
            water_density=kwargs.get("water_density", 1000.0),
            water_viscosity=kwargs.get("water_viscosity", 0.001),
            temperature=kwargs.get("temperature", 293.15),
            pressure=kwargs.get("pressure", 101325.0)
        )
        
        # Set surface tensions if provided
        if all(k in kwargs for k in ["surface_tension_ow", "surface_tension_go", "surface_tension_gw"]):
            model.set_surface_tensions(
                kwargs["surface_tension_ow"],
                kwargs["surface_tension_go"],
                kwargs["surface_tension_gw"]
            )
        
        # Set black oil properties if provided
        if all(k in kwargs for k in ["GOR", "Bo", "Bg", "Rs"]):
            model.set_black_oil_properties(
                kwargs["GOR"],
                kwargs["Bo"],
                kwargs["Bg"],
                kwargs["Rs"]
            )
    
    elif model_type == "pigging":
        model = PiggingTransport(
            fluid_density=kwargs.get("fluid_density", 1000.0),
            fluid_viscosity=kwargs.get("fluid_viscosity", 0.001),
            pig_density=kwargs.get("pig_density", 300.0),
            pig_diameter=kwargs.get("pig_diameter", 0.1),
            pipeline_diameter=kwargs.get("pipeline_diameter", 0.1016),
            bypass_fraction=kwargs.get("bypass_fraction", 0.05),
            friction_coefficient=kwargs.get("friction_coefficient", 0.3)
        )
    
    elif model_type == "spill":
        model = SpillTransport(
            spill_type=kwargs.get("spill_type", SpillTransport.SURFACE_SPILL),
            fluid_type=kwargs.get("fluid_type", SpillTransport.CRUDE_OIL),
            fluid_density=kwargs.get("fluid_density", 850.0),
            fluid_viscosity=kwargs.get("fluid_viscosity", 0.03),
            surface_tension=kwargs.get("surface_tension", 0.025),
            evaporation_rate=kwargs.get("evaporation_rate", 0.0001),
            environment_type=kwargs.get("environment_type", "water"),
            wind_speed=kwargs.get("wind_speed", 5.0),
            current_speed=kwargs.get("current_speed", 0.5),
            temperature=kwargs.get("temperature", 293.15)
        )
        
        # Set additional fluid properties if provided
        fluid_props = kwargs.get("fluid_properties")
        if fluid_props:
            model.set_fluid_properties(fluid_props)
    
    else:
        logger.error(f"Unknown transport model type: {model_type}")
        return False
    
    # Write model to file
    if model:
        return model.write_to_file(transport_file)
    
    return False


def create_from_fluid_database(case_dir: Union[str, Path], fluid_name: str, 
                              model_type: str = "newtonian", **kwargs) -> bool:
    """
    Create transport properties from fluid database.
    
    Args:
        case_dir (Union[str, Path]): OpenFOAM case directory
        fluid_name (str): Name of the fluid in the database
        model_type (str): Type of transport model
        **kwargs: Additional arguments for the specific model
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get fluid properties from database
        fluid_props = MaterialDatabase.get_fluid_properties(fluid_name)
        
        # Merge with kwargs (kwargs take precedence)
        props = {**fluid_props, **kwargs}
        
        # Create transport properties
        return create_transport_properties(case_dir, model_type, **props)
        
    except Exception as e:
        logger.error(f"Error creating transport properties from fluid database: {e}")
        return False


# Example usage
if __name__ == "__main__":
    print("OpenFOAM Transport Models Module")
    
    # Example: Create a Newtonian transport model
    newtonian = NewtonianTransport(density=998.0, viscosity=1.0e-3)
    print(newtonian.generate_dict_entry())
    
    # Example: Create a multiphase model for oil-water flow
    phases = [
        {"density": 998.0, "viscosity": 1.0e-3},  # Water
        {"density": 850.0, "viscosity": 3.0e-2}   # Oil
    ]
    multiphase = MultiPhaseTransport(phases, ["water", "oil"])
    multiphase.set_surface_tension("water", "oil", 0.025)
    print(multiphase.generate_dict_entry())