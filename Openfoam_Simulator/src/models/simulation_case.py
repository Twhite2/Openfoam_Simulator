#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation case data model for Openfoam_Simulator application.

This module implements the simulation case model that represents a specific
OpenFOAM simulation configuration with all associated settings for
CFD simulations in the oil & gas industry.
"""

import os
import sys
import json
import shutil
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

# Import utility modules
from ..utils.logger import get_logger
from ..config import get_value, set_value

logger = get_logger(__name__)


class SimulationCase:
    """
    Simulation case class for Openfoam_Simulator application.
    
    This class represents a specific OpenFOAM simulation configuration and manages:
    - Solver settings (type, time steps, convergence criteria, etc.)
    - Physical model configuration (turbulence model, multiphase settings, etc.)
    - Boundary conditions
    - Initial conditions
    - Numerical schemes
    - Output settings
    """
    
    def __init__(self, name: str = "Untitled Case", mesh_path: str = None, case_dir: str = None):
        """
        Initialize a new simulation case.
        
        Args:
            name (str, optional): Case name
            mesh_path (str, optional): Path to the mesh file
            case_dir (str, optional): Directory for case files
        """
        # Case metadata
        self.name = name
        self.description = ""
        self.created_date = datetime.datetime.now().isoformat()
        self.modified_date = self.created_date
        
        # Case directory
        self.case_dir = case_dir
        
        # Mesh information
        self.mesh_path = mesh_path
        self.mesh_type = None  # "blockMesh", "snappyHexMesh", "external", etc.
        
        # General settings
        self.solver_type = "simpleFoam"  # Default to steady-state incompressible
        self.time_dependent = False
        self.multiphase = False
        self.multiphase_type = None  # "VOF", "Euler", etc.
        self.number_of_phases = 1
        
        # Time settings
        self.start_time = 0.0
        self.end_time = 100.0
        self.time_step = 0.01
        self.adjustable_time_step = False
        self.max_courant_number = 1.0
        self.write_interval = 10
        self.write_format = "ascii"  # "ascii" or "binary"
        
        # Physical models
        self.turbulence_model = "kEpsilon"
        self.turbulence_wall_function = "standard"
        self.gravity = [0.0, 0.0, -9.81]  # Default gravity in m/s²
        self.reference_pressure = 101325.0  # Atmospheric pressure in Pa
        self.reference_temperature = 293.15  # 20°C in K
        
        # Phase properties - dictionary of phase properties
        # Keys are phase names, values are dictionaries of properties
        self.phase_properties = {}
        
        # Interface properties (for multiphase cases)
        self.interface_properties = {}
        
        # Domain dimensions
        self.domain_min = None  # [x_min, y_min, z_min]
        self.domain_max = None  # [x_max, y_max, z_max]
        
        # Boundary conditions - dictionary of boundary conditions
        # Keys are boundary names, values are dictionaries of condition specifications
        self.boundary_conditions = {}
        
        # Initial conditions - dictionary of initial field values
        self.initial_conditions = {}
        
        # Numerical schemes
        self.gradient_scheme = "Gauss linear"
        self.divergence_scheme = "Gauss upwind"
        self.laplacian_scheme = "Gauss linear corrected"
        self.interpolation_scheme = "linear"
        
        # Solution controls
        self.p_reference_cell = 0
        self.p_reference_value = 0.0
        self.convergence_tolerance = 1e-5
        self.max_iterations = 1000
        self.relaxation_factors = {
            "p": 0.3,
            "U": 0.7,
            "k": 0.7,
            "epsilon": 0.7,
            "omega": 0.7
        }
        
        # Parallel settings
        self.parallel = False
        self.n_processors = 4
        self.decomposition_method = "scotch"
        
        # Special models for oil & gas
        self.pigging_simulation = False
        self.pigging_properties = {}
        
        self.spill_simulation = False
        self.spill_properties = {}
        
        # Runtime status
        self.is_running = False
        self.current_time = 0.0
        self.current_iteration = 0
        self.residuals = {}
    
    def setup_single_phase_flow(self, fluid_type: str = "water", 
                                turbulence_model: str = "kEpsilon", 
                                inlet_velocity: float = 1.0,
                                inlet_pressure: float = 101325.0,
                                outlet_pressure: float = 101325.0) -> bool:
        """
        Configure the case for single-phase flow simulation.
        
        Args:
            fluid_type (str): Type of fluid ("water", "oil", "gas", "custom")
            turbulence_model (str): Turbulence model to use
            inlet_velocity (float): Inlet velocity in m/s
            inlet_pressure (float): Inlet pressure in Pa
            outlet_pressure (float): Outlet pressure in Pa
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Set case parameters for single-phase flow
            self.multiphase = False
            self.number_of_phases = 1
            self.solver_type = "simpleFoam"
            
            # Set turbulence model
            self.turbulence_model = turbulence_model
            
            # Set up fluid properties
            fluid_props = self._get_default_fluid_properties(fluid_type)
            self.phase_properties = {"fluid": fluid_props}
            
            # Set up boundary conditions for inlet and outlet
            # (This is a simplification - actual boundary setup would be more complex)
            self.boundary_conditions = {
                "inlet": {
                    "type": "inlet",
                    "velocity": [inlet_velocity, 0.0, 0.0],
                    "pressure": inlet_pressure
                },
                "outlet": {
                    "type": "outlet",
                    "pressure": outlet_pressure
                },
                "walls": {
                    "type": "wall",
                    "velocity": [0.0, 0.0, 0.0]  # No-slip condition
                }
            }
            
            # Update metadata
            self.modified_date = datetime.datetime.now().isoformat()
            
            logger.info(f"Set up single-phase flow case with {fluid_type} fluid")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up single phase flow: {e}")
            return False
    
    def setup_two_phase_flow(self, flow_type: str = "Oil-Water",
                             phase1_type: str = "water", 
                             phase2_type: str = "oil",
                             phase1_fraction: float = 0.7,
                             phase2_fraction: float = 0.3,
                             surface_tension: float = 0.025) -> bool:
        """
        Configure the case for two-phase flow simulation.
        
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
            # Set case parameters for two-phase flow
            self.multiphase = True
            self.multiphase_type = "VOF"  # Volume of Fluid method
            self.number_of_phases = 2
            self.solver_type = "interFoam"
            
            # Adjust for time-dependent simulation
            self.time_dependent = True
            
            # Set up phase properties
            phase1_props = self._get_default_fluid_properties(phase1_type)
            phase2_props = self._get_default_fluid_properties(phase2_type)
            
            # Add volume fractions
            phase1_props["fraction"] = phase1_fraction
            phase2_props["fraction"] = phase2_fraction
            
            self.phase_properties = {
                "phase1": phase1_props,
                "phase2": phase2_props
            }
            
            # Set up interface properties
            self.interface_properties = {
                "surface_tension": surface_tension,
                "interface_compression": True,
                "contact_angle": 90.0  # Neutral wetting in degrees
            }
            
            # Update metadata
            self.modified_date = datetime.datetime.now().isoformat()
            
            logger.info(f"Set up two-phase flow case with {flow_type} flow")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up two-phase flow: {e}")
            return False
    
    def setup_three_phase_flow(self) -> bool:
        """
        Configure the case for three-phase flow simulation.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Set case parameters for three-phase flow
            self.multiphase = True
            self.multiphase_type = "multiPhaseEulerFoam"
            self.number_of_phases = 3
            self.solver_type = "multiphaseInterFoam"
            
            # Adjust for time-dependent simulation
            self.time_dependent = True
            
            # Set up phase properties
            water_props = self._get_default_fluid_properties("water")
            oil_props = self._get_default_fluid_properties("oil")
            gas_props = self._get_default_fluid_properties("gas")
            
            # Add volume fractions
            water_props["fraction"] = 0.4
            oil_props["fraction"] = 0.4
            gas_props["fraction"] = 0.2
            
            self.phase_properties = {
                "water": water_props,
                "oil": oil_props,
                "gas": gas_props
            }
            
            # Set up interface properties
            self.interface_properties = {
                "surface_tension": {
                    "water-oil": 0.025,
                    "water-gas": 0.072,
                    "oil-gas": 0.023
                },
                "interface_compression": True
            }
            
            # Update metadata
            self.modified_date = datetime.datetime.now().isoformat()
            
            logger.info("Set up three-phase flow case")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up three-phase flow: {e}")
            return False
    
    def setup_pigging_simulation(self, pipe_diameter: float = 0.1, 
                                 pig_type: str = "Foam",
                                 driving_fluid: str = "Water") -> bool:
        """
        Configure the case for pipeline pigging simulation.
        
        Args:
            pipe_diameter (float): Pipe diameter in meters
            pig_type (str): Type of pig ("Foam", "Disc", "Cup", "Sphere")
            driving_fluid (str): Type of driving fluid ("Water", "Oil", "Gas")
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Set case parameters for pigging simulation
            self.multiphase = True
            self.multiphase_type = "VOF"
            self.number_of_phases = 2
            self.solver_type = "interFoam"
            self.pigging_simulation = True
            
            # Adjust for time-dependent simulation
            self.time_dependent = True
            
            # Set up pigging properties
            self.pigging_properties = {
                "pipe_diameter": pipe_diameter,
                "pig_type": pig_type,
                "pig_diameter": pipe_diameter * 1.05,  # 5% oversized
                "pig_length": pipe_diameter * 2.0,
                "pig_density": 300.0,  # kg/m³
                "pig_friction": 0.3,
                "bypass_flow": 0.05,  # 5% bypass
                "driving_fluid": driving_fluid,
                "initial_position": 0.0  # At the start of the pipeline
            }
            
            # Set up phase properties
            driving_fluid_props = self._get_default_fluid_properties(driving_fluid.lower())
            displaced_fluid_props = self._get_default_fluid_properties("oil")  # Default to oil as displaced fluid
            
            self.phase_properties = {
                "driving_fluid": driving_fluid_props,
                "displaced_fluid": displaced_fluid_props
            }
            
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
        Configure the case for spill simulation.
        
        Args:
            spill_type (str): Type of spill ("Surface", "Subsurface", "Jet")
            fluid_type (str): Type of spilled fluid ("Crude Oil", "Diesel", etc.)
            environment (str): Environment type ("Water (Ocean)", "Land (Soil)", etc.)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Set case parameters for spill simulation
            self.multiphase = True
            self.multiphase_type = "VOF"
            self.number_of_phases = 2
            self.solver_type = "interFoam"
            self.spill_simulation = True
            
            # Adjust for time-dependent simulation
            self.time_dependent = True
            
            # Set up spill properties
            self.spill_properties = {
                "spill_type": spill_type,
                "fluid_type": fluid_type,
                "environment": environment,
                "spill_rate": 10.0,  # kg/s
                "total_mass": 1000.0,  # kg
                "duration": 100.0,  # s
                "temperature": 293.15,  # K
                "wind_speed": 5.0,  # m/s
                "current_speed": 0.5,  # m/s (for water environments)
                "weathering": True,
                "dispersion": True,
                "evaporation": True
            }
            
            # Set up phase properties
            spilled_fluid_props = self._get_default_fluid_properties(fluid_type.lower().replace(" ", "_"))
            
            if environment.startswith("Water"):
                environment_props = self._get_default_fluid_properties("water")
            else:
                # For land environments, use air as the second phase
                environment_props = self._get_default_fluid_properties("air")
            
            self.phase_properties = {
                "spilled_fluid": spilled_fluid_props,
                "environment": environment_props
            }
            
            # Update metadata
            self.modified_date = datetime.datetime.now().isoformat()
            
            logger.info(f"Set up spill simulation with {fluid_type} in {environment}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up spill simulation: {e}")
            return False
    
    def write_case_files(self) -> bool:
        """
        Write OpenFOAM case files to the case directory.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.case_dir:
            logger.error("Case directory not set")
            return False
        
        try:
            # Create case directory structure
            self._create_case_directory_structure()
            
            # Write controlDict
            self._write_control_dict()
            
            # Write fvSchemes
            self._write_fv_schemes()
            
            # Write fvSolution
            self._write_fv_solution()
            
            # Write transportProperties
            self._write_transport_properties()
            
            # Write boundary conditions
            self._write_boundary_conditions()
            
            # Write initial conditions
            self._write_initial_conditions()
            
            # Write decomposeParDict if parallel
            if self.parallel:
                self._write_decompose_par_dict()
            
            # Write special model files if needed
            if self.pigging_simulation:
                self._write_pigging_files()
                
            if self.spill_simulation:
                self._write_spill_files()
            
            logger.info(f"Case files written to {self.case_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing case files: {e}")
            return False
    
    def _create_case_directory_structure(self):
        """Create OpenFOAM case directory structure."""
        case_dir = Path(self.case_dir)
        
        # Create main directories
        system_dir = case_dir / "system"
        constant_dir = case_dir / "constant"
        time0_dir = case_dir / "0"
        
        # Create directories if they don't exist
        for directory in [system_dir, constant_dir, time0_dir]:
            if not directory.exists():
                directory.mkdir(parents=True)
                logger.debug(f"Created directory: {directory}")
        
        # Create subdirectories for specific purposes
        if self.parallel:
            processor_dir = case_dir / "processor0"
            if not processor_dir.exists():
                processor_dir.mkdir(parents=True)
        
        # Create polyMesh directory for mesh files
        polymesh_dir = constant_dir / "polyMesh"
        if not polymesh_dir.exists():
            polymesh_dir.mkdir(parents=True)
    
    def _write_control_dict(self):
        """Write the controlDict file."""
        system_dir = Path(self.case_dir) / "system"
        control_dict_path = system_dir / "controlDict"
        
        with open(control_dict_path, 'w') as f:
            f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
            f.write("| =========                 |                                                 |\n")
            f.write("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
            f.write("|  \\\\    /   O peration     | Version:  v2312                                 |\n")
            f.write("|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n")
            f.write("|    \\\\/     M anipulation  |                                                 |\n")
            f.write("\\*---------------------------------------------------------------------------*/\n")
            f.write("FoamFile\n")
            f.write("{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       dictionary;\n")
            f.write("    object      controlDict;\n")
            f.write("}\n")
            f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
            
            f.write(f"application     {self.solver_type};\n\n")
            
            f.write("startFrom       startTime;\n\n")
            
            f.write(f"startTime       {self.start_time};\n\n")
            
            f.write("stopAt          endTime;\n\n")
            
            f.write(f"endTime         {self.end_time};\n\n")
            
            f.write(f"deltaT          {self.time_step};\n\n")
            
            f.write("writeControl    runTime;\n\n")
            
            f.write(f"writeInterval   {self.write_interval};\n\n")
            
            f.write("purgeWrite      0;\n\n")
            
            f.write(f"writeFormat     {self.write_format};\n\n")
            
            f.write("writePrecision  8;\n\n")
            
            f.write("writeCompression off;\n\n")
            
            f.write("timeFormat      general;\n\n")
            
            f.write("timePrecision   6;\n\n")
            
            f.write("runTimeModifiable true;\n\n")
            
            # Add adjustable time step settings if enabled
            if self.adjustable_time_step:
                f.write("adjustTimeStep  yes;\n")
                f.write(f"maxCo           {self.max_courant_number};\n\n")
            
            f.write("// ************************************************************************* //\n")
    
    def _write_fv_schemes(self):
        """Write the fvSchemes file."""
        system_dir = Path(self.case_dir) / "system"
        fv_schemes_path = system_dir / "fvSchemes"
        
        with open(fv_schemes_path, 'w') as f:
            f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
            f.write("| =========                 |                                                 |\n")
            f.write("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
            f.write("|  \\\\    /   O peration     | Version:  v2312                                 |\n")
            f.write("|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n")
            f.write("|    \\\\/     M anipulation  |                                                 |\n")
            f.write("\\*---------------------------------------------------------------------------*/\n")
            f.write("FoamFile\n")
            f.write("{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       dictionary;\n")
            f.write("    object      fvSchemes;\n")
            f.write("}\n")
            f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
            
            # ddtSchemes (time derivative)
            if self.time_dependent:
                f.write("ddtSchemes\n")
                f.write("{\n")
                f.write("    default         Euler;\n")
                f.write("}\n\n")
            else:
                f.write("ddtSchemes\n")
                f.write("{\n")
                f.write("    default         steadyState;\n")
                f.write("}\n\n")
            
            # gradSchemes
            f.write("gradSchemes\n")
            f.write("{\n")
            f.write(f"    default         {self.gradient_scheme};\n")
            f.write("}\n\n")
            
            # divSchemes
            f.write("divSchemes\n")
            f.write("{\n")
            f.write("    default         none;\n")
            
            # Add specific schemes based on solver type
            if self.solver_type.startswith("simple") or self.solver_type.startswith("piso"):
                f.write(f"    div(phi,U)      {self.divergence_scheme};\n")
                f.write(f"    div(phi,k)      {self.divergence_scheme};\n")
                f.write(f"    div(phi,epsilon) {self.divergence_scheme};\n")
                f.write(f"    div(phi,omega)  {self.divergence_scheme};\n")
                f.write(f"    div(phi,R)      {self.divergence_scheme};\n")
                f.write(f"    div(R)          {self.divergence_scheme};\n")
                f.write(f"    div((nuEff*dev2(T(grad(U))))) {self.divergence_scheme};\n")
            
            # Additional schemes for multiphase solvers
            if self.multiphase:
                f.write(f"    div(phi,alpha)  {self.divergence_scheme};\n")
                f.write(f"    div(phirb,alpha) {self.divergence_scheme};\n")
            
            f.write("}\n\n")
            
            # laplacianSchemes
            f.write("laplacianSchemes\n")
            f.write("{\n")
            f.write(f"    default         {self.laplacian_scheme};\n")
            f.write("}\n\n")
            
            # interpolationSchemes
            f.write("interpolationSchemes\n")
            f.write("{\n")
            f.write(f"    default         {self.interpolation_scheme};\n")
            f.write("}\n\n")
            
            # snGradSchemes
            f.write("snGradSchemes\n")
            f.write("{\n")
            f.write("    default         corrected;\n")
            f.write("}\n\n")
            
            f.write("// ************************************************************************* //\n")
    
    def _write_fv_solution(self):
        """Write the fvSolution file."""
        system_dir = Path(self.case_dir) / "system"
        fv_solution_path = system_dir / "fvSolution"
        
        with open(fv_solution_path, 'w') as f:
            f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
            f.write("| =========                 |                                                 |\n")
            f.write("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
            f.write("|  \\\\    /   O peration     | Version:  v2312                                 |\n")
            f.write("|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n")
            f.write("|    \\\\/     M anipulation  |                                                 |\n")
            f.write("\\*---------------------------------------------------------------------------*/\n")
            f.write("FoamFile\n")
            f.write("{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       dictionary;\n")
            f.write("    object      fvSolution;\n")
            f.write("}\n")
            f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
            
            # solvers section
            f.write("solvers\n")
            f.write("{\n")
            
            # Pressure solver
            f.write("    p\n")
            f.write("    {\n")
            f.write("        solver          GAMG;\n")
            f.write(f"        tolerance       {self.convergence_tolerance};\n")
            f.write("        relTol          0.01;\n")
            f.write("        smoother        GaussSeidel;\n")
            f.write("        nPreSweeps      0;\n")
            f.write("        nPostSweeps     2;\n")
            f.write("        cacheAgglomeration on;\n")
            f.write("        agglomerator    faceAreaPair;\n")
            f.write("        nCellsInCoarsestLevel 10;\n")
            f.write("        mergeLevels     1;\n")
            f.write("        maxIter         100;\n")
            f.write("    }\n\n")
            
            # Velocity, turbulence, and other fields
            f.write("    \"(U|k|epsilon|omega|R)\"\n")
            f.write("    {\n")
            f.write("        solver          smoothSolver;\n")
            f.write("        smoother        GaussSeidel;\n")
            f.write(f"        tolerance       {self.convergence_tolerance};\n")
            f.write("        relTol          0.1;\n")
            f.write("        nSweeps         1;\n")
            f.write("    }\n")
            
            # Add alpha solver for multiphase cases
            if self.multiphase:
                f.write("\n    alpha\n")
                f.write("    {\n")
                f.write("        solver          smoothSolver;\n")
                f.write("        smoother        GaussSeidel;\n")
                f.write(f"        tolerance       {self.convergence_tolerance};\n")
                f.write("        relTol          0.1;\n")
                f.write("        nSweeps         1;\n")
                f.write("    }\n")
            
            f.write("}\n\n")
            
            # SIMPLE algorithm settings for steady-state solvers
            if self.solver_type.startswith("simple"):
                f.write("SIMPLE\n")
                f.write("{\n")
                f.write("    nNonOrthogonalCorrectors 0;\n")
                f.write("    consistent      yes;\n")
                f.write("\n")
                f.write("    residualControl\n")
                f.write("    {\n")
                f.write(f"        p               {self.convergence_tolerance};\n")
                f.write(f"        U               {self.convergence_tolerance};\n")
                f.write(f"        \"(k|epsilon|omega|R)\" {self.convergence_tolerance};\n")
                f.write("    }\n")
                f.write("}\n\n")
            
            # PISO algorithm settings for transient solvers
            if self.solver_type.startswith("piso") or self.solver_type == "interFoam":
                f.write("PISO\n")
                f.write("{\n")
                f.write("    nCorrectors     2;\n")
                f.write("    nNonOrthogonalCorrectors 1;\n")
                f.write(f"    pRefCell        {self.p_reference_cell};\n")
                f.write(f"    pRefValue       {self.p_reference_value};\n")
                f.write("}\n\n")
            
            # PIMPLE algorithm for interFoam and similar solvers
            if self.solver_type == "interFoam" or self.solver_type == "multiphaseInterFoam":
                f.write("PIMPLE\n")
                f.write("{\n")
                f.write("    nCorrectors     3;\n")
                f.write("    nNonOrthogonalCorrectors 1;\n")
                f.write(f"    pRefCell        {self.p_reference_cell};\n")
                f.write(f"    pRefValue       {self.p_reference_value};\n")
                f.write("}\n\n")
            
            # Relaxation factors
            f.write("relaxationFactors\n")
            f.write("{\n")
            f.write("    equations\n")
            f.write("    {\n")
            
            for field, factor in self.relaxation_factors.items():
                f.write(f"        {field}               {factor};\n")
            
            f.write("    }\n")
            f.write("}\n\n")
            
            f.write("// ************************************************************************* //\n")
    
    def _write_transport_properties(self):
        """Write the transportProperties file."""
        constant_dir = Path(self.case_dir) / "constant"
        transport_path = constant_dir / "transportProperties"
        
        with open(transport_path, 'w') as f:
            f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
            f.write("| =========                 |                                                 |\n")
            f.write("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
            f.write("|  \\\\    /   O peration     | Version:  v2312                                 |\n")
            f.write("|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n")
            f.write("|    \\\\/     M anipulation  |                                                 |\n")
            f.write("\\*---------------------------------------------------------------------------*/\n")
            f.write("FoamFile\n")
            f.write("{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       dictionary;\n")
            f.write("    object      transportProperties;\n")
            f.write("}\n")
            f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
            
            # Single phase properties
            if not self.multiphase:
                # Get properties of the first phase
                props = self._get_first_phase_properties()
                
                f.write("transportModel  Newtonian;\n\n")
                
                # Kinematic viscosity
                nu = props['viscosity'] / props['density']
                f.write(f"nu              {nu};\n\n")
                
                # Density
                f.write(f"rho             {props['density']};\n\n")
            
            # Multiphase properties
            else:
                # For VOF method (interFoam)
                if self.multiphase_type == "VOF":
                    f.write("phases\n")
                    f.write("(\n")
                    
                    # Add each phase
                    for name, props in self.phase_properties.items():
                        phase_name = name.lower()
                        f.write(f"    {phase_name}\n")
                        f.write("    {\n")
                        f.write("        transportModel  Newtonian;\n")
                        nu = props['viscosity'] / props['density']
                        f.write(f"        nu              {nu};\n")
                        f.write(f"        rho             {props['density']};\n")
                        f.write("    }\n")
                    
                    f.write(");\n\n")
                    
                    # Add surface tension for interface
                    if 'surface_tension' in self.interface_properties:
                        st = self.interface_properties['surface_tension']
                        f.write(f"sigma            {st};\n\n")
            
            f.write("// ************************************************************************* //\n")
    
    def _write_boundary_conditions(self):
        """Write the boundary condition files."""
        # This would create the necessary 0/ directory files for U, p, k, epsilon, etc.
        # This is a simplified version that would need to be expanded for real use
        time0_dir = Path(self.case_dir) / "0"
        
        # Create velocity file
        u_path = time0_dir / "U"
        with open(u_path, 'w') as f:
            f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
            f.write("| =========                 |                                                 |\n")
            f.write("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
            f.write("|  \\\\    /   O peration     | Version:  v2312                                 |\n")
            f.write("|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n")
            f.write("|    \\\\/     M anipulation  |                                                 |\n")
            f.write("\\*---------------------------------------------------------------------------*/\n")
            f.write("FoamFile\n")
            f.write("{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       volVectorField;\n")
            f.write("    object      U;\n")
            f.write("}\n")
            f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
            
            # Default internal field
            f.write("dimensions      [0 1 -1 0 0 0 0];\n\n")
            f.write("internalField   uniform (0 0 0);\n\n")
            
            # Boundary fields
            f.write("boundaryField\n")
            f.write("{\n")
            
            # Add boundary conditions for each boundary
            for name, conditions in self.boundary_conditions.items():
                f.write(f"    {name}\n")
                f.write("    {\n")
                
                # Set type-specific conditions
                if conditions['type'] == 'inlet':
                    f.write("        type            fixedValue;\n")
                    velocity = conditions.get('velocity', [0, 0, 0])
                    f.write(f"        value           uniform ({velocity[0]} {velocity[1]} {velocity[2]});\n")
                
                elif conditions['type'] == 'outlet':
                    f.write("        type            zeroGradient;\n")
                
                elif conditions['type'] == 'wall':
                    f.write("        type            noSlip;\n")
                
                f.write("    }\n")
            
            f.write("}\n\n")
            
            f.write("// ************************************************************************* //\n")
        
        # Create pressure file (similar approach for other fields)
        p_path = time0_dir / "p"
        with open(p_path, 'w') as f:
            f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
            f.write("| =========                 |                                                 |\n")
            f.write("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
            f.write("|  \\\\    /   O peration     | Version:  v2312                                 |\n")
            f.write("|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n")
            f.write("|    \\\\/     M anipulation  |                                                 |\n")
            f.write("\\*---------------------------------------------------------------------------*/\n")
            f.write("FoamFile\n")
            f.write("{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       volScalarField;\n")
            f.write("    object      p;\n")
            f.write("}\n")
            f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
            
            # Default internal field
            f.write("dimensions      [0 2 -2 0 0 0 0];\n\n")
            f.write(f"internalField   uniform {self.reference_pressure};\n\n")
            
            # Boundary fields
            f.write("boundaryField\n")
            f.write("{\n")
            
            # Add boundary conditions for each boundary
            for name, conditions in self.boundary_conditions.items():
                f.write(f"    {name}\n")
                f.write("    {\n")
                
                # Set type-specific conditions
                if conditions['type'] == 'inlet':
                    f.write("        type            zeroGradient;\n")
                
                elif conditions['type'] == 'outlet':
                    f.write("        type            fixedValue;\n")
                    pressure = conditions.get('pressure', self.reference_pressure)
                    f.write(f"        value           uniform {pressure};\n")
                
                elif conditions['type'] == 'wall':
                    f.write("        type            zeroGradient;\n")
                
                f.write("    }\n")
            
            f.write("}\n\n")
            
            f.write("// ************************************************************************* //\n")
    
    def _write_initial_conditions(self):
        """Write the initial condition files."""
        # This would be similar to boundary conditions but for the internal field
        # In a real implementation, this would be more complex
        pass
    
    def _write_decompose_par_dict(self):
        """Write the decomposeParDict file for parallel runs."""
        system_dir = Path(self.case_dir) / "system"
        decomp_path = system_dir / "decomposeParDict"
        
        with open(decomp_path, 'w') as f:
            f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
            f.write("| =========                 |                                                 |\n")
            f.write("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
            f.write("|  \\\\    /   O peration     | Version:  v2312                                 |\n")
            f.write("|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n")
            f.write("|    \\\\/     M anipulation  |                                                 |\n")
            f.write("\\*---------------------------------------------------------------------------*/\n")
            f.write("FoamFile\n")
            f.write("{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       dictionary;\n")
            f.write("    object      decomposeParDict;\n")
            f.write("}\n")
            f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
            
            f.write(f"numberOfSubdomains {self.n_processors};\n\n")
            
            f.write(f"method          {self.decomposition_method};\n\n")
            
            # Add method-specific settings
            if self.decomposition_method == "simple":
                f.write("simpleCoeffs\n")
                f.write("{\n")
                f.write(f"    n               ({self.n_processors} 1 1);\n")
                f.write("    delta           0.001;\n")
                f.write("}\n\n")
            
            elif self.decomposition_method == "hierarchical":
                f.write("hierarchicalCoeffs\n")
                f.write("{\n")
                # Calculate a reasonable decomposition
                n = [1, 1, 1]
                remaining = self.n_processors
                for i in range(3):
                    while remaining % 2 == 0 and n[i] < 16:
                        n[i] *= 2
                        remaining //= 2
                    if remaining > 1:
                        n[i] *= remaining
                        remaining = 1
                
                f.write(f"    n               ({n[0]} {n[1]} {n[2]});\n")
                f.write("    delta           0.001;\n")
                f.write("    order           xyz;\n")
                f.write("}\n\n")
            
            f.write("// ************************************************************************* //\n")
    
    def _write_pigging_files(self):
        """Write special files for pigging simulation."""
        # This would create additional files needed for pigging simulation
        # In a real implementation, this would be more complex
        pass
    
    def _write_spill_files(self):
        """Write special files for spill simulation."""
        # This would create additional files needed for spill simulation
        # In a real implementation, this would be more complex
        pass
    
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
            "thermal_conductivity": 0.6,  # W/(m·K)
            "temperature": 293.15     # K
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
    
    def _get_first_phase_properties(self) -> Dict[str, Any]:
        """
        Get properties of the first phase.
        
        Returns:
            Dict[str, Any]: Dictionary of fluid properties
        """
        if self.phase_properties:
            # Return the first phase found
            for name, props in self.phase_properties.items():
                return props
        
        # Default to water if no phases defined
        return self._get_default_fluid_properties("water")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the simulation case to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the case
        """
        # Create a dictionary from all instance variables
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationCase':
        """
        Create a SimulationCase from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary data
            
        Returns:
            SimulationCase: New instance created from data
        """
        case = cls()
        
        # Update instance variables from dictionary
        for key, value in data.items():
            if hasattr(case, key):
                setattr(case, key, value)
        
        return case
    
    def save(self, filepath: str) -> bool:
        """
        Save the simulation case to a JSON file.
        
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
            
            logger.info(f"Saved simulation case to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving simulation case: {e}")
            return False
    
    @classmethod
    def load(cls, filepath: str) -> 'SimulationCase':
        """
        Load a simulation case from a JSON file.
        
        Args:
            filepath (str): Path to the file
            
        Returns:
            SimulationCase: Loaded simulation case
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file is invalid JSON
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Case file not found: {filepath}")
        
        try:
            # Load from JSON
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Create from dictionary
            case = cls.from_dict(data)
            
            logger.info(f"Loaded simulation case from {filepath}")
            return case
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing case file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading simulation case: {e}")
            raise ValueError(f"Failed to load simulation case: {str(e)}")