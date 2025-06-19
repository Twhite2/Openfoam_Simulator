#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spill modeling and simulation for Openfoam_Simulator.

This module provides tools for setting up and running OpenFOAM simulations
of oil and gas spills, focusing on environmental impact assessment and 
response planning in the oil & gas industry.

The module handles various spill types and environmental conditions:
- Surface spills (on water or land)
- Subsurface spills (underwater releases)
- Jet releases (high-pressure releases)
- Different fluid types (crude oil, diesel, gasoline, natural gas)
- Weathering processes (evaporation, dispersion, dissolution)
- Environmental factors (wind, current, temperature, terrain)
"""

import os
import sys
import math
import logging
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum, auto

# Import utilities
from ...utils.logger import get_logger
from ...config import get_value

# Import OpenFOAM integration modules
from ..case_manager import CaseManager
from ..solver_manager import SolverManager
from ..boundary_conditions import BoundaryConditionsManager
from ..transport_models import TransportModelsManager
from ..mesh_generator import MeshGenerator

# Industry-specific imports
from ...industry.spill_models import SpillType, FluidType, EnvironmentType, WeatheringModel

logger = get_logger(__name__)


class SpillSimulationType(Enum):
    """Enumeration of spill simulation types."""
    SURFACE = auto()  # Surface spill (on water or land)
    SUBSURFACE = auto()  # Underwater/underground release
    JET = auto()  # High-pressure jet release


class FluidProperties:
    """Class to store physical properties of spilled fluids."""
    
    def __init__(self, 
                 name: str,
                 density: float,
                 viscosity: float, 
                 surface_tension: float,
                 evaporation_rate: float,
                 solubility: float,
                 vapor_pressure: float,
                 flash_point: float):
        """
        Initialize the fluid properties.
        
        Args:
            name (str): Name of the fluid
            density (float): Density in kg/m³
            viscosity (float): Dynamic viscosity in Pa·s
            surface_tension (float): Surface tension in N/m
            evaporation_rate (float): Evaporation rate coefficient
            solubility (float): Solubility in water in kg/m³
            vapor_pressure (float): Vapor pressure in Pa
            flash_point (float): Flash point in K
        """
        self.name = name
        self.density = density
        self.viscosity = viscosity
        self.surface_tension = surface_tension
        self.evaporation_rate = evaporation_rate
        self.solubility = solubility
        self.vapor_pressure = vapor_pressure
        self.flash_point = flash_point


class SpillSimulation:
    """
    Class for managing and executing spill simulations in OpenFOAM.
    
    This class handles the complex multiphase flow simulations for 
    environmental spill modeling, incorporating weathering processes
    and environmental factors.
    """
    
    # Common fluid properties presets
    FLUID_PRESETS = {
        "Crude Oil": FluidProperties(
            name="Crude Oil",
            density=850.0,
            viscosity=0.05,
            surface_tension=0.025,
            evaporation_rate=2.5e-6,
            solubility=0.005,
            vapor_pressure=2000.0,
            flash_point=333.15  # 60°C
        ),
        "Diesel": FluidProperties(
            name="Diesel",
            density=830.0,
            viscosity=0.003,
            surface_tension=0.023,
            evaporation_rate=5.0e-6,
            solubility=0.0005,
            vapor_pressure=1000.0,
            flash_point=328.15  # 55°C
        ),
        "Gasoline": FluidProperties(
            name="Gasoline",
            density=750.0,
            viscosity=0.0005,
            surface_tension=0.022,
            evaporation_rate=4.0e-5,
            solubility=0.002,
            vapor_pressure=40000.0,
            flash_point=233.15  # -40°C
        ),
        "Natural Gas": FluidProperties(
            name="Natural Gas",
            density=0.8,
            viscosity=1.1e-5,
            surface_tension=0.0,  # Gas, no surface tension
            evaporation_rate=1.0,  # Already gaseous
            solubility=0.0002,
            vapor_pressure=101325.0,  # Atmospheric pressure
            flash_point=111.15  # -162°C (methane)
        )
    }
    
    def __init__(self, case_dir: str, spill_parameters: Dict[str, Any], 
                environment_parameters: Dict[str, Any],
                simulation_parameters: Dict[str, Any]):
        """
        Initialize the spill simulation.
        
        Args:
            case_dir (str): Path to the OpenFOAM case directory
            spill_parameters (Dict[str, Any]): Parameters defining the spill
            environment_parameters (Dict[str, Any]): Parameters defining the environment
            simulation_parameters (Dict[str, Any]): Parameters defining the simulation
        """
        self.case_dir = Path(case_dir)
        self.spill_params = spill_parameters
        self.env_params = environment_parameters
        self.sim_params = simulation_parameters
        
        # Initialize managers
        self.case_manager = CaseManager(case_dir)
        self.solver_manager = SolverManager(case_dir)
        self.bc_manager = BoundaryConditionsManager(case_dir)
        self.transport_manager = TransportModelsManager(case_dir)
        
        # Determine spill type
        spill_type_str = self.spill_params.get("type", "Surface").lower()
        if spill_type_str == "subsurface":
            self.spill_type = SpillSimulationType.SUBSURFACE
        elif spill_type_str == "jet":
            self.spill_type = SpillSimulationType.JET
        else:
            self.spill_type = SpillSimulationType.SURFACE
        
        # Get fluid properties
        fluid_name = self.spill_params.get("fluid", "Crude Oil")
        self.fluid_props = self.FLUID_PRESETS.get(fluid_name, self.FLUID_PRESETS["Crude Oil"])
        
        # Set appropriate solver based on spill type
        if self.spill_type == SpillSimulationType.SURFACE:
            # Use interFoam with VOF approach for surface spills
            self.solver_name = "interFoam"
        elif self.spill_type == SpillSimulationType.SUBSURFACE:
            # Use multiphaseInterFoam for subsurface spills with multiple phases
            self.solver_name = "multiphaseInterFoam"
        elif self.spill_type == SpillSimulationType.JET:
            # Use compressible solver for high-pressure jets
            if fluid_name == "Natural Gas":
                self.solver_name = "rhoCentralFoam"
            else:
                self.solver_name = "interFoam"
        
        # Initialize state flags
        self.mesh_generated = False
        self.case_setup = False
    
    def generate_mesh(self) -> bool:
        """
        Generate the mesh for the spill simulation.
        
        This creates a specialized mesh appropriate for the spill type
        and environmental conditions.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create mesh generator
            mesh_generator = MeshGenerator(self.case_dir)
            
            # Determine domain parameters
            domain_size = self.sim_params.get("domain_size", 1000.0)  # Default 1km
            
            # Set up domain based on environment type
            env_type = self.env_params.get("type", "Water (Ocean)")
            
            if env_type.startswith("Water"):
                # Water environment (ocean, river, lake)
                if self.spill_type == SpillSimulationType.SURFACE:
                    # Surface spill on water
                    depth = domain_size / 5.0  # Water depth - 1/5 of domain
                    mesh_generator.setup_water_surface_mesh(
                        length=domain_size,
                        width=domain_size,
                        depth=depth,
                        cells_x=100,
                        cells_y=100,
                        cells_z=50,
                        spill_location=(domain_size/2, domain_size/2, 0),
                        refinement_regions=[
                            {
                                "type": "box",
                                "min": (domain_size/2 - domain_size/4, domain_size/2 - domain_size/4, -depth/10),
                                "max": (domain_size/2 + domain_size/4, domain_size/2 + domain_size/4, depth/10),
                                "level": 2
                            }
                        ]
                    )
                
                elif self.spill_type == SpillSimulationType.SUBSURFACE:
                    # Subsurface spill in water
                    depth = domain_size / 2.0  # Water depth - 1/2 of domain
                    # Release depth - 1/2 of water depth
                    release_depth = self.spill_params.get("release_depth", depth/2)
                    
                    mesh_generator.setup_water_volume_mesh(
                        length=domain_size,
                        width=domain_size,
                        depth=depth,
                        cells_x=80,
                        cells_y=80,
                        cells_z=40,
                        spill_location=(domain_size/2, domain_size/2, -release_depth),
                        refinement_regions=[
                            {
                                "type": "sphere",
                                "center": (domain_size/2, domain_size/2, -release_depth),
                                "radius": domain_size/10,
                                "level": 2
                            }
                        ]
                    )
                
                elif self.spill_type == SpillSimulationType.JET:
                    # Jet release in water
                    depth = domain_size / 2.0
                    release_depth = self.spill_params.get("release_depth", depth/2)
                    opening_diameter = self.spill_params.get("opening_diameter", 0.05)
                    
                    mesh_generator.setup_jet_mesh(
                        length=domain_size,
                        width=domain_size,
                        depth=depth,
                        cells_x=80,
                        cells_y=80,
                        cells_z=40,
                        jet_origin=(domain_size/2, domain_size/2, -release_depth),
                        jet_direction=(0, 0, 1),  # Upward
                        jet_diameter=opening_diameter,
                        refinement_levels=3
                    )
            
            else:
                # Land environment
                if self.spill_type == SpillSimulationType.SURFACE:
                    # Surface spill on land
                    height = domain_size / 10.0  # Atmosphere height
                    
                    mesh_generator.setup_terrain_mesh(
                        length=domain_size,
                        width=domain_size,
                        height=height,
                        cells_x=100,
                        cells_y=100,
                        cells_z=30,
                        spill_location=(domain_size/2, domain_size/2, 0),
                        terrain_file=self.env_params.get("terrain_file", None),
                        refinement_regions=[
                            {
                                "type": "box",
                                "min": (domain_size/2 - domain_size/4, domain_size/2 - domain_size/4, 0),
                                "max": (domain_size/2 + domain_size/4, domain_size/2 + domain_size/4, height/5),
                                "level": 2
                            }
                        ]
                    )
                
                elif self.spill_type == SpillSimulationType.JET:
                    # Jet release on land
                    height = domain_size / 5.0
                    opening_diameter = self.spill_params.get("opening_diameter", 0.05)
                    
                    mesh_generator.setup_jet_mesh(
                        length=domain_size,
                        width=domain_size,
                        depth=height,
                        cells_x=80,
                        cells_y=80,
                        cells_z=40,
                        jet_origin=(domain_size/2, domain_size/2, 0.1),
                        jet_direction=(0, 0, 1),  # Upward
                        jet_diameter=opening_diameter,
                        refinement_levels=3
                    )
            
            # Generate the mesh
            success = mesh_generator.generate()
            if success:
                logger.info(f"Mesh generation successful for spill simulation in {self.case_dir}")
                self.mesh_generated = True
                return True
            else:
                logger.error(f"Mesh generation failed for spill simulation in {self.case_dir}")
                return False
            
        except Exception as e:
            logger.error(f"Error generating mesh for spill simulation: {e}")
            return False
    
    def setup_case(self) -> bool:
        """
        Set up the OpenFOAM case for spill simulation.
        
        This configures all necessary files for the simulation, including:
        - Control parameters (controlDict)
        - Numerical schemes (fvSchemes)
        - Solution settings (fvSolution)
        - Transport properties
        - Boundary conditions
        - Turbulence model settings
        - Weathering models (if enabled)
        - Initialization fields
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if mesh exists
            if not self.mesh_generated:
                logger.warning("Mesh has not been generated yet. Attempting to generate mesh...")
                if not self.generate_mesh():
                    return False
            
            # Set up case structure
            self.case_manager.setup_base_case_structure()
            
            # Set up controlDict
            self._setup_control_dict()
            
            # Set up schemes
            self._setup_schemes()
            
            # Set up solution settings
            self._setup_solution()
            
            # Set up transport properties
            self._setup_transport_properties()
            
            # Set up turbulence modeling
            self._setup_turbulence_model()
            
            # Set up boundary conditions
            self._setup_boundary_conditions()
            
            # Set up weathering models if enabled
            if self.sim_params.get("weathering", True):
                self._setup_weathering_models()
            
            # Set up initial fields
            self._setup_initial_fields()
            
            # Set up decomposition dictionary for parallel runs
            self._setup_decomposition()
            
            logger.info(f"Case setup successful for spill simulation in {self.case_dir}")
            self.case_setup = True
            return True
            
        except Exception as e:
            logger.error(f"Error setting up case for spill simulation: {e}")
            return False
    
    def _setup_control_dict(self):
        """Set up the controlDict file for time control and I/O settings."""
        # Get simulation duration
        sim_duration = self.sim_params.get("duration", 3600.0)  # Default 1 hour
        
        # Determine appropriate time step based on spill type and environment
        if self.spill_type == SpillSimulationType.JET:
            # Smaller time step for jet releases
            init_dt = 0.0001
        elif self.spill_type == SpillSimulationType.SURFACE:
            # Medium time step for surface spills
            init_dt = 0.01
        else:
            # Moderate time step for subsurface spills
            init_dt = 0.001
        
        # Prepare function objects for specialized analysis
        function_objects = {}
        
        # Add weathering model tracking if enabled
        if self.sim_params.get("weathering", True):
            function_objects["weatheringAnalysis"] = {
                "type": "coded",
                "libs": ["libutilityFunctionObjects.so"],
                "name": "weatheringAnalysis",
                "writeControl": "timeStep",
                "writeInterval": 10,
                "active": True,
                "code": self._get_weathering_analysis_code()
            }
        
        # Add surface area calculation for surface spills
        if self.spill_type == SpillSimulationType.SURFACE:
            function_objects["surfaceAreaCalculation"] = {
                "type": "interfaceHeight",
                "libs": ["libfieldFunctionObjects.so"],
                "writeControl": "timeStep",
                "writeInterval": 10,
                "alpha": "alpha.water",
                "interfaces": ["internalFace"],
                "interpolationScheme": "pointMVC"
            }
        
        # Set up control dictionary
        control_dict = {
            "application": self.solver_name,
            "startFrom": "startTime",
            "startTime": 0,
            "stopAt": "endTime",
            "endTime": sim_duration,
            "deltaT": init_dt,
            "writeControl": "adjustableRunTime",
            "writeInterval": sim_duration / 100,  # 100 output intervals
            "purgeWrite": 0,
            "writeFormat": "binary",
            "writePrecision": 6,
            "writeCompression": "on",
            "timeFormat": "general",
            "timePrecision": 6,
            "runTimeModifiable": True,
            "adjustTimeStep": True,
            "maxCo": 0.5,  # Conservative Courant number for stability
            "maxAlphaCo": 0.5,
            "functions": function_objects
        }
        
        self.case_manager.write_control_dict(control_dict)
    
    def _get_weathering_analysis_code(self) -> str:
        """
        Get the code for weathering analysis function object.
        
        Returns:
            str: The C++ code for the weathering analysis
        """
        # This is a placeholder for the actual C++ code that would be used in OpenFOAM
        # In a real implementation, this would contain actual OpenFOAM compatible C++ code
        code = """
            #include "fvCFD.H"
            #include "volFields.H"
            #include "surfaceFields.H"
            
            // Execute at each timestep
            codeExecute
            {
                const volScalarField& alpha = mesh().lookupObject<volScalarField>("alpha.water");
                
                // Calculate total spill volume
                scalar spillVolume = 0.0;
                scalar evaporatedVolume = 0.0;
                scalar dispersedVolume = 0.0;
                
                forAll(alpha, cellI)
                {
                    spillVolume += (1.0 - alpha[cellI]) * mesh().V()[cellI];
                }
                
                // Calculate evaporation and dispersion based on models
                
                // Log the results
                Info<< "Spill Analysis at t = " << mesh().time().value() << nl
                    << "  Total spill volume: " << spillVolume << nl
                    << "  Evaporated volume: " << evaporatedVolume << nl
                    << "  Dispersed volume: " << dispersedVolume << nl
                    << endl;
                
                // Write to a CSV file for post-processing
                if (Pstream::master())
                {
                    std::ofstream file;
                    file.open("spillAnalysis.csv", std::ios::app);
                    file << mesh().time().value() << ","
                         << spillVolume << ","
                         << evaporatedVolume << ","
                         << dispersedVolume << "\\n";
                    file.close();
                }
            }
        """
        return code
    
    def _setup_schemes(self):
        """Set up the numerical schemes for the simulation."""
        # Adjust schemes based on the solver type
        if self.solver_name == "interFoam" or self.solver_name == "multiphaseInterFoam":
            # VOF schemes for liquid spills
            schemes = {
                "ddtSchemes": {
                    "default": "Euler"
                },
                "gradSchemes": {
                    "default": "Gauss linear",
                    "grad(U)": "cellLimited Gauss linear 1",
                    "grad(alpha.water)": "cellLimited Gauss linear 1"
                },
                "divSchemes": {
                    "default": "none",
                    "div(rhoPhi,U)": "Gauss linearUpwind grad(U)",
                    "div(phi,alpha)": "Gauss vanLeer",
                    "div(phirb,alpha)": "Gauss linear",
                    "div(phi,k)": "Gauss upwind",
                    "div(phi,epsilon)": "Gauss upwind",
                    "div(((rho*nuEff)*dev2(T(grad(U)))))": "Gauss linear"
                },
                "laplacianSchemes": {
                    "default": "Gauss linear corrected"
                },
                "interpolationSchemes": {
                    "default": "linear"
                },
                "snGradSchemes": {
                    "default": "corrected"
                }
            }
        elif self.solver_name == "rhoCentralFoam":
            # Compressible schemes for gas spills
            schemes = {
                "ddtSchemes": {
                    "default": "Euler"
                },
                "gradSchemes": {
                    "default": "Gauss linear"
                },
                "divSchemes": {
                    "default": "none",
                    "div(tauMC)": "Gauss linear",
                    "div(phi,T)": "Gauss upwind",
                    "div(phiv,p)": "Gauss upwind",
                    "div(phiv,U)": "Gauss upwind"
                },
                "laplacianSchemes": {
                    "default": "Gauss linear corrected"
                },
                "interpolationSchemes": {
                    "default": "linear",
                    "reconstruct(rho)": "vanLeer",
                    "reconstruct(U)": "vanLeerV",
                    "reconstruct(T)": "vanLeer"
                },
                "snGradSchemes": {
                    "default": "corrected"
                }
            }
        
        self.case_manager.write_fv_schemes(schemes)
    
    def _setup_solution(self):
        """Set up the solution control settings."""
        # Adjust solution settings based on the solver type
        if self.solver_name == "interFoam" or self.solver_name == "multiphaseInterFoam":
            # Multiphase solution settings
            solution = {
                "solvers": {
                    "alpha.water.*": {
                        "nAlphaCorr": 2,
                        "nAlphaSubCycles": 1,
                        "alphaOuterCorrectors": True,
                        "cAlpha": 1
                    },
                    "pcorr.*": {
                        "solver": "PCG",
                        "preconditioner": "DIC",
                        "tolerance": 1e-05,
                        "relTol": 0
                    },
                    "p_rgh.*": {
                        "solver": "PCG",
                        "preconditioner": "DIC",
                        "tolerance": 1e-07,
                        "relTol": 0.05
                    },
                    "p_rghFinal": {
                        "solver": "PCG",
                        "preconditioner": "DIC",
                        "tolerance": 1e-07,
                        "relTol": 0
                    },
                    "U.*": {
                        "solver": "smoothSolver",
                        "smoother": "symGaussSeidel",
                        "tolerance": 1e-07,
                        "relTol": 0.1
                    },
                    "UFinal": {
                        "solver": "smoothSolver",
                        "smoother": "symGaussSeidel",
                        "tolerance": 1e-07,
                        "relTol": 0
                    },
                    "k.*": {
                        "solver": "smoothSolver",
                        "smoother": "symGaussSeidel",
                        "tolerance": 1e-07,
                        "relTol": 0.1
                    },
                    "epsilon.*": {
                        "solver": "smoothSolver",
                        "smoother": "symGaussSeidel",
                        "tolerance": 1e-07,
                        "relTol": 0.1
                    }
                },
                "PIMPLE": {
                    "momentumPredictor": True,
                    "nOuterCorrectors": 2,
                    "nCorrectors": 2,
                    "nNonOrthogonalCorrectors": 1
                },
                "relaxationFactors": {
                    "equations": {
                        "U": 0.7,
                        "k": 0.7,
                        "epsilon": 0.7
                    }
                }
            }
        else:
            # Compressible solver settings
            solution = {
                "solvers": {
                    "rho": {
                        "solver": "PCG",
                        "preconditioner": "DIC",
                        "tolerance": 1e-05,
                        "relTol": 0.1
                    },
                    "U": {
                        "solver": "smoothSolver",
                        "smoother": "symGaussSeidel",
                        "tolerance": 1e-05,
                        "relTol": 0.1
                    },
                    "e": {
                        "solver": "smoothSolver",
                        "smoother": "symGaussSeidel",
                        "tolerance": 1e-05,
                        "relTol": 0.1
                    },
                    "k": {
                        "solver": "smoothSolver",
                        "smoother": "symGaussSeidel",
                        "tolerance": 1e-05,
                        "relTol": 0.1
                    },
                    "epsilon": {
                        "solver": "smoothSolver",
                        "smoother": "symGaussSeidel",
                        "tolerance": 1e-05,
                        "relTol": 0.1
                    }
                }
            }
        
        self.case_manager.write_fv_solution(solution)
    
    def _setup_transport_properties(self):
        """Set up the transport properties for the simulation."""
        # Get fluid properties
        fluid_properties = self.fluid_props
        
        # For multiphase simulations
        if self.solver_name == "interFoam" or self.solver_name == "multiphaseInterFoam":
            # Create content for transportProperties file
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
    class       dictionary;
    object      transportProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

phases (
    water
    {{
        transportModel  Newtonian;
        nu              1.0e-6;
        rho             1000;
    }}

    oil
    {{
        transportModel  Newtonian;
        nu              {fluid_properties.viscosity / fluid_properties.density};
        rho             {fluid_properties.density};
    }}
);

sigma           {fluid_properties.surface_tension};

// Weathering model properties
weatheringProperties
{{
    evaporationModel      constant;
    evaporationRate       {fluid_properties.evaporation_rate};
    
    dispersionModel       constant;
    dispersionRate        1e-5;
    
    waterTemperature      {self.env_params.get("water_temperature", 288.15)};
    airTemperature        {self.env_params.get("ambient_temperature", 293.15)};
}}

// ************************************************************************* //
"""
        elif self.solver_name == "rhoCentralFoam":
            # For compressible gas simulations
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
    class       dictionary;
    object      transportProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

transportModel  Newtonian;

nu              {fluid_properties.viscosity / fluid_properties.density};
rho             {fluid_properties.density};

// Thermophysical properties
thermoType
{{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleInternalEnergy;
}}

mixture
{{
    specie
    {{
        molWeight       16.04;   // Methane
    }}
    thermodynamics
    {{
        Cp              2220;
        Hf              0;
    }}
    transport
    {{
        mu              {fluid_properties.viscosity};
        Pr              0.7;
    }}
}}

// ************************************************************************* //
"""
        
        # Write to file
        transport_file = self.case_dir / "constant" / "transportProperties"
        with open(transport_file, 'w') as f:
            f.write(content)
    
    def _setup_turbulence_model(self):
        """Set up the turbulence model settings."""
        # Create turbulenceProperties file
        os.makedirs(self.case_dir / "constant", exist_ok=True)
        
        model = self.sim_params.get("turbulence_model", "kEpsilon")
        
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
    class       dictionary;
    object      turbulenceProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

simulationType  RAS;

RAS
{{
    RASModel        {model};
    
    turbulence      on;
    
    printCoeffs     on;
}}

// ************************************************************************* //
"""
        
        # Write the file
        turbulence_file = self.case_dir / "constant" / "turbulenceProperties"
        with open(turbulence_file, 'w') as f:
            f.write(content)
    
    def _setup_weathering_models(self):
        """Set up weathering models for the spill simulation."""
        # This would configure specific weathering models if available
        # For now, we'll create a basic structure
        
        # Create directory for weathering models
        weathering_dir = self.case_dir / "constant" / "weathering"
        os.makedirs(weathering_dir, exist_ok=True)
        
        # Setup weathering models based on simulation parameters
        enable_evaporation = self.sim_params.get("evaporation", True)
        enable_dispersion = self.sim_params.get("dispersion", True)
        
        # Evaporation model
        if enable_evaporation:
            evap_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
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
    class       dictionary;
    object      evaporationProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

evaporationModel constant;

constantCoeffs
{{
    rate            {self.fluid_props.evaporation_rate};
    activationTemp  {self.fluid_props.flash_point};
    vaporPressure   {self.fluid_props.vapor_pressure};
}}

// ************************************************************************* //
"""
            
            # Write evaporation model file
            evap_file = weathering_dir / "evaporationProperties"
            with open(evap_file, 'w') as f:
                f.write(evap_content)
        
        # Dispersion model
        if enable_dispersion:
            disp_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
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
    class       dictionary;
    object      dispersionProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dispersionModel  windWave;

windWaveCoeffs
{{
    windSpeed       {self.env_params.get("wind_speed", 5.0)};
    waveHeight      0.5;
    waveFrequency   0.1;
    oilViscosity    {self.fluid_props.viscosity};
    oilDensity      {self.fluid_props.density};
    waterViscosity  1.0e-3;
    waterDensity    1000.0;
    interfaceTension {self.fluid_props.surface_tension};
}}

// ************************************************************************* //
"""
            
            # Write dispersion model file
            disp_file = weathering_dir / "dispersionProperties"
            with open(disp_file, 'w') as f:
                f.write(disp_content)
    
    def _setup_boundary_conditions(self):
        """Set up the boundary conditions for the simulation."""
        # Set up the boundary conditions based on simulation type and environment
        # This is a simplified approach; real implementation would be more complex
        
        # Determine boundary names based on environment
        env_type = self.env_params.get("type", "Water (Ocean)")
        
        if env_type.startswith("Water"):
            # Water environment
            if self.spill_type == SpillSimulationType.SURFACE:
                # Surface spill on water
                boundaries = {
                    "atmosphere": {
                        "type": "patch"
                    },
                    "water": {
                        "type": "wall"
                    },
                    "sides": {
                        "type": "patch"
                    },
                    "spillSource": {
                        "type": "patch"
                    }
                }
            elif self.spill_type == SpillSimulationType.SUBSURFACE:
                # Subsurface spill in water
                boundaries = {
                    "atmosphere": {
                        "type": "patch"
                    },
                    "bottom": {
                        "type": "wall"
                    },
                    "sides": {
                        "type": "patch"
                    },
                    "spillSource": {
                        "type": "patch"
                    }
                }
            elif self.spill_type == SpillSimulationType.JET:
                # Jet release in water
                boundaries = {
                    "atmosphere": {
                        "type": "patch"
                    },
                    "bottom": {
                        "type": "wall"
                    },
                    "sides": {
                        "type": "patch"
                    },
                    "jetInlet": {
                        "type": "patch"
                    }
                }
        else:
            # Land environment
            if self.spill_type == SpillSimulationType.SURFACE:
                # Surface spill on land
                boundaries = {
                    "atmosphere": {
                        "type": "patch"
                    },
                    "ground": {
                        "type": "wall"
                    },
                    "sides": {
                        "type": "patch"
                    },
                    "spillSource": {
                        "type": "patch"
                    }
                }
            elif self.spill_type == SpillSimulationType.JET:
                # Jet release on land
                boundaries = {
                    "atmosphere": {
                        "type": "patch"
                    },
                    "ground": {
                        "type": "wall"
                    },
                    "sides": {
                        "type": "patch"
                    },
                    "jetInlet": {
                        "type": "patch"
                    }
                }
        
        # Define boundary conditions for velocity
        velocity_bcs = {}
        pressure_bcs = {}
        alpha_bcs = {}
        
        # Setup for each boundary
        for name, props in boundaries.items():
            if name == "spillSource" or name == "jetInlet":
                # Spill source - inflow boundary
                spill_rate = self.spill_params.get("rate", 10.0)  # kg/s
                opening_diam = self.spill_params.get("opening_diameter", 0.05)  # m
                opening_area = np.pi * (opening_diam/2)**2
                
                # Calculate velocity based on density and area
                inflow_vel = spill_rate / (self.fluid_props.density * opening_area)
                
                # Set boundary conditions
                if self.spill_type == SpillSimulationType.JET:
                    # High velocity jet
                    velocity_bcs[name] = {
                        "type": "fixedValue",
                        "value": f"uniform (0 0 {inflow_vel})"
                    }
                else:
                    # Surface or subsurface spill
                    velocity_bcs[name] = {
                        "type": "fixedValue",
                        "value": f"uniform (0 0 {inflow_vel/10})"  # Lower velocity
                    }
                
                pressure_bcs[name] = {
                    "type": "zeroGradient"
                }
                
                alpha_bcs[name] = {
                    "type": "fixedValue",
                    "value": "uniform 0"  # Oil/gas phase
                }
            
            elif name == "atmosphere":
                # Atmosphere boundary - open to air
                velocity_bcs[name] = {
                    "type": "pressureInletOutletVelocity",
                    "value": "uniform (0 0 0)"
                }
                
                pressure_bcs[name] = {
                    "type": "totalPressure",
                    "p0": "uniform 0",
                    "U": "U",
                    "phi": "phi",
                    "rho": "rho",
                    "psi": "none"
                }
                
                alpha_bcs[name] = {
                    "type": "inletOutlet",
                    "inletValue": "uniform 1",  # Water phase
                    "value": "uniform 1"
                }
            
            elif name == "sides":
                # Side boundaries - open
                velocity_bcs[name] = {
                    "type": "pressureInletOutletVelocity",
                    "value": "uniform (0 0 0)"
                }
                
                pressure_bcs[name] = {
                    "type": "fixedValue",
                    "value": "uniform 0"
                }
                
                alpha_bcs[name] = {
                    "type": "zeroGradient"
                }
            
            else:
                # Wall boundaries
                velocity_bcs[name] = {
                    "type": "noSlip"
                }
                
                pressure_bcs[name] = {
                    "type": "zeroGradient"
                }
                
                alpha_bcs[name] = {
                    "type": "zeroGradient"
                }
        
        # Set up boundary conditions via manager
        if self.solver_name == "interFoam" or self.solver_name == "multiphaseInterFoam":
            self.bc_manager.set_boundary_conditions("U", velocity_bcs)
            self.bc_manager.set_boundary_conditions("p_rgh", pressure_bcs)
            self.bc_manager.set_boundary_conditions("alpha.water", alpha_bcs)
        else:
            # For compressible solver
            self.bc_manager.set_boundary_conditions("U", velocity_bcs)
            self.bc_manager.set_boundary_conditions("p", pressure_bcs)
            
            # Temperature boundary conditions
            temp_bcs = {}
            for name in boundaries.keys():
                if name == "spillSource" or name == "jetInlet":
                    temp_bcs[name] = {
                        "type": "fixedValue",
                        "value": f"uniform {self.spill_params.get('temperature', 293.15)}"
                    }
                else:
                    temp_bcs[name] = {
                        "type": "zeroGradient"
                    }
            
            self.bc_manager.set_boundary_conditions("T", temp_bcs)
    
    def _setup_initial_fields(self):
        """Set up the initial fields for the simulation."""
        # Create 0 directory
        os.makedirs(self.case_dir / "0", exist_ok=True)
        
        # Initialize fields based on solver type
        if self.solver_name == "interFoam" or self.solver_name == "multiphaseInterFoam":
            # For VOF simulations
            
            # Initialize velocity field
            # Include environmental effects like wind and currents
            wind_speed = self.env_params.get("wind_speed", 5.0)
            current_speed = self.env_params.get("current_speed", 0.5)
            
            # Apply only to water environments
            if self.env_params.get("type", "").startswith("Water"):
                initial_velocity = f"uniform ({current_speed} 0 0)"
            else:
                initial_velocity = "uniform (0 0 0)"
            
            self.bc_manager.initialize_field(
                "U", 
                initial_velocity,
                {"internalField": True}
            )
            
            # Initialize pressure field
            self.bc_manager.initialize_field(
                "p_rgh", 
                "uniform 0",
                {"internalField": True}
            )
            
            # Initialize alpha field (1 = water, 0 = oil/gas)
            self.bc_manager.initialize_field(
                "alpha.water", 
                "uniform 1",  # Start with all water
                {"internalField": True}
            )
            
            # Initialize turbulence fields if using turbulence model
            if self.sim_params.get("turbulence_model", "kEpsilon") != "laminar":
                # Estimate turbulence values based on environment
                if self.env_params.get("type", "").startswith("Water"):
                    velocity = current_speed
                else:
                    velocity = wind_speed
                
                intensity = 0.05  # 5% turbulence intensity
                length_scale = 0.1  # 10cm length scale
                
                k = 1.5 * (velocity * intensity)**2
                epsilon = 0.09**0.75 * k**1.5 / length_scale
                
                self.bc_manager.initialize_field(
                    "k", 
                    f"uniform {k}",
                    {"internalField": True}
                )
                
                self.bc_manager.initialize_field(
                    "epsilon", 
                    f"uniform {epsilon}",
                    {"internalField": True}
                )
        
        else:
            # For compressible gas simulations
            
            # Initialize velocity field
            self.bc_manager.initialize_field(
                "U", 
                "uniform (0 0 0)",
                {"internalField": True}
            )
            
            # Initialize pressure field
            # Use atmospheric pressure
            self.bc_manager.initialize_field(
                "p", 
                "uniform 101325",
                {"internalField": True}
            )
            
            # Initialize temperature field
            # Use ambient temperature
            ambient_temp = self.env_params.get("ambient_temperature", 293.15)
            self.bc_manager.initialize_field(
                "T", 
                f"uniform {ambient_temp}",
                {"internalField": True}
            )
    
    def _setup_decomposition(self):
        """Set up the domain decomposition for parallel runs."""
        # Default values
        n_processors = get_value('openfoam.solver_parallelism', 4)
        if n_processors == 'auto':
            # Detect number of cores
            import multiprocessing
            n_processors = multiprocessing.cpu_count()
        
        # Create decomposeParDict
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
    class       dictionary;
    object      decomposeParDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

numberOfSubdomains {n_processors};

method          scotch;

scotchCoeffs
{{
    processorWeights ( {' '.join(['1'] * n_processors)} );
}}

// ************************************************************************* //
"""
        
        # Write to file
        decomp_file = self.case_dir / "system" / "decomposeParDict"
        with open(decomp_file, 'w') as f:
            f.write(content)
    
    def run_simulation(self, parallel: bool = True) -> bool:
        """
        Run the spill simulation.
        
        Args:
            parallel (bool): Whether to run in parallel
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if case is set up
        if not self.case_setup:
            logger.warning("Case has not been set up yet. Attempting to set up case...")
            if not self.setup_case():
                return False
        
        # Run the simulation
        if parallel:
            # First decompose the domain
            self.solver_manager.run_utility("decomposePar", parallel=False)
            
            # Run the solver in parallel
            success = self.solver_manager.run_solver(self.solver_name, parallel=True)
            
            # Reconstruct the results if successful
            if success:
                self.solver_manager.run_utility("reconstructPar", parallel=False)
                
            return success
        else:
            # Run the solver in serial
            return self.solver_manager.run_solver(self.solver_name, parallel=False)
    
    def post_process(self) -> bool:
        """
        Run post-processing operations on the simulation results.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Run paraFoam to generate visualization files
            self.solver_manager.run_utility("paraFoam", ["-touch"], parallel=False)
            
            # Generate spill contour visualization
            if self.spill_type == SpillSimulationType.SURFACE:
                # Surface contour extraction
                self._generate_surface_contours()
            
            # Generate summary report
            self._generate_report()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
            return False
    
    def _generate_surface_contours(self):
        """Generate surface contours for the spill visualization."""
        # This would create a script to extract surface contours using ParaView
        # For now, just create a placeholder script
        
        script_content = """
import os
import sys
from paraview.simple import *

# Load the case
openfoam_file = 'case.foam'
reader = OpenFOAMReader(FileName=openfoam_file)
reader.MeshRegions = ['internalMesh']
reader.CellArrays = ['alpha.water']

# Get the time steps
timesteps = reader.TimestepValues

# Create a slice at the surface
slice1 = Slice(Input=reader)
slice1.SliceType = 'Plane'
slice1.SliceType.Origin = [0, 0, 0]
slice1.SliceType.Normal = [0, 0, 1]

# Create a contour of the oil-water interface
contour1 = Contour(Input=slice1)
contour1.ContourBy = ['POINTS', 'alpha.water']
contour1.Isosurfaces = [0.5]

# Extract the contour at each time step
for i, time in enumerate(timesteps):
    if i % 10 == 0:  # Every 10th step
        # Set the time
        reader.UpdatePipeline(time)
        contour1.UpdatePipeline()
        
        # Save the contour
        writer = CreateWriter(f'contours/contour_{i:04d}.vtp', contour1)
        writer.UpdatePipeline()

# Create a final visualization
Show(contour1)
Render()
SaveScreenshot('final_contour.png')
"""
        
        # Create directory for contours
        contours_dir = self.case_dir / "postProcessing" / "contours"
        os.makedirs(contours_dir, exist_ok=True)
        
        # Write script
        script_file = self.case_dir / "postProcessing" / "extract_contours.py"
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        logger.info(f"Surface contour extraction script created at {script_file}")
    
    def _generate_report(self):
        """Generate a summary report of the spill simulation."""
        # Create a basic report with summary information
        report_content = f"""# Spill Simulation Report

## Simulation Overview
- Spill Type: {self.spill_type.name}
- Fluid: {self.fluid_props.name}
- Environment: {self.env_params.get("type", "Water (Ocean)")}
- Duration: {self.sim_params.get("duration", 3600.0)} seconds

## Spill Parameters
- Spill Rate: {self.spill_params.get("rate", 10.0)} kg/s
- Total Mass: {self.spill_params.get("total_mass", 1000.0)} kg
- Opening Diameter: {self.spill_params.get("opening_diameter", 0.05)} m
- Fluid Temperature: {self.spill_params.get("temperature", 293.15)} K
- Fluid Pressure: {self.spill_params.get("pressure", 101325.0)} Pa

## Environmental Parameters
- Wind Speed: {self.env_params.get("wind_speed", 5.0)} m/s
- Current Speed: {self.env_params.get("current_speed", 0.5)} m/s
- Water Temperature: {self.env_params.get("water_temperature", 288.15)} K
- Ambient Temperature: {self.env_params.get("ambient_temperature", 293.15)} K

## Simulation Parameters
- Domain Size: {self.sim_params.get("domain_size", 1000.0)} m
- Weathering Models: {"Enabled" if self.sim_params.get("weathering", True) else "Disabled"}
- Evaporation: {"Enabled" if self.sim_params.get("evaporation", True) else "Disabled"}
- Dispersion: {"Enabled" if self.sim_params.get("dispersion", True) else "Disabled"}

## Results Summary
- Maximum Spill Area: [Calculated from simulation]
- Total Evaporated: [Calculated from simulation]
- Total Dispersed: [Calculated from simulation]
- Remaining Mass: [Calculated from simulation]

## Visualization
- Surface contours are available in the postProcessing/contours directory
- ParaView state file available at case.foam
"""
        
        # Create directory for report
        report_dir = self.case_dir / "postProcessing"
        os.makedirs(report_dir, exist_ok=True)
        
        # Write report file
        report_file = report_dir / "spill_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Simulation report generated at {report_file}")


def create_spill_simulation(case_dir: str, spill_parameters: Dict[str, Any], 
                          environment_parameters: Dict[str, Any],
                          simulation_parameters: Dict[str, Any]) -> SpillSimulation:
    """
    Create a spill simulation case.
    
    Args:
        case_dir (str): Path to the OpenFOAM case directory
        spill_parameters (Dict[str, Any]): Parameters defining the spill
        environment_parameters (Dict[str, Any]): Parameters defining the environment
        simulation_parameters (Dict[str, Any]): Parameters defining the simulation
        
    Returns:
        SpillSimulation: The created simulation object
    """
    # Create the simulation object
    sim = SpillSimulation(case_dir, spill_parameters, environment_parameters, 
                         simulation_parameters)
    
    # Return the simulation object
    return sim


def run_from_template(case_dir: str, template_name: str, parameters: Dict[str, Any]) -> bool:
    """
    Set up and run a spill simulation from a template.
    
    Args:
        case_dir (str): Path to the OpenFOAM case directory
        template_name (str): Name of the template to use
        parameters (Dict[str, Any]): Parameters to customize the template
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get template path from configuration
        templates_dir = get_value('paths.templates', '')
        template_path = Path(templates_dir) / "case_templates" / "spill" / template_name
        
        # Check if template exists
        if not template_path.exists():
            logger.error(f"Template {template_name} does not exist at {template_path}")
            return False
        
        # Create case directory if it doesn't exist
        os.makedirs(case_dir, exist_ok=True)
        
        # Copy template to case directory
        for item in template_path.glob('*'):
            if item.is_dir():
                shutil.copytree(item, Path(case_dir) / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, Path(case_dir) / item.name)
        
        # Extract parameters into the format needed by SpillSimulation
        spill_params = parameters.get('spill', {})
        env_params = parameters.get('environment', {})
        sim_params = parameters.get('simulation', {})
        
        # Create simulation object
        sim = SpillSimulation(case_dir, spill_params, env_params, sim_params)
        
        # Run simulation
        if sim.setup_case():
            success = sim.run_simulation(parallel=True)
            if success:
                sim.post_process()
            return success
        else:
            return False
            
    except Exception as e:
        logger.error(f"Error running from template: {e}")
        return False


def analyze_results(case_dir: str) -> Dict[str, Any]:
    """
    Analyze the results of a spill simulation.
    
    Args:
        case_dir (str): Path to the OpenFOAM case directory
        
    Returns:
        Dict[str, Any]: Analysis results
    """
    try:
        from ...utils.field_calculator import FieldCalculator
        
        # Create field calculator
        calculator = FieldCalculator(case_dir)
        
        # Extract basic statistics
        results = {}
        
        # Load alpha field to calculate spill volume and area
        # For surface spills, calculate spread area over time
        alpha_stats = calculator.compute_field_statistics("alpha.water")
        
        # Calculate volumes
        total_volume = calculator.calculate_domain_volume()
        water_volume = calculator.calculate_phase_volume("alpha.water", 0.5, 1.0)
        oil_volume = total_volume - water_volume
        
        # For surface spills, calculate surface area
        surface_area = calculator.calculate_interface_area("alpha.water", 0.5)
        
        # Calculate maximum extent
        max_extent = calculator.calculate_max_extent("alpha.water", 0.5)
        
        # Add to results
        results["oil_volume"] = oil_volume
        results["surface_area"] = surface_area
        results["max_extent"] = max_extent
        
        # If weathering models were enabled, extract weathering data
        weathering_data = calculator.extract_weathering_data()
        if weathering_data:
            results["weathering"] = weathering_data
        
        # Extract time series data
        time_series = calculator.extract_time_series_data()
        results["time_series"] = time_series
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing results: {e}")
        return {"error": str(e)}