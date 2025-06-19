#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single phase flow solver setup for Openfoam_Simulator.

This module provides classes and functions to set up and configure
OpenFOAM solvers for single phase flow simulations, including:
- Incompressible flow solvers (simpleFoam, pisoFoam)
- Compressible flow solvers (rhoPimpleFoam)
- Turbulence model configuration
- Solver control settings
- Case initialization functions
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

# Add parent directory to path to allow imports from other modules
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import OpenFOAM integration modules
try:
    from openfoam_integration.case_manager import CaseManager
    from openfoam_integration.boundary_conditions import BoundaryCondition
    from openfoam_integration.transport_models import NewtonianTransport, NonNewtonianTransport
except ImportError:
    # Placeholder imports for standalone usage
    CaseManager = object
    BoundaryCondition = object

# Import utility modules
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


class SinglePhaseSolver:
    """
    Base class for single phase flow solvers.
    
    This class provides the common interface and functionality
    for all single phase flow solvers in OpenFOAM.
    """
    
    # Solver types
    SIMPLE_FOAM = "simpleFoam"  # Steady-state incompressible
    PISO_FOAM = "pisoFoam"      # Transient incompressible
    RHO_SIMPLE_FOAM = "rhoSimpleFoam"  # Steady-state compressible
    RHO_PIMPLE_FOAM = "rhoPimpleFoam"  # Transient compressible
    
    # Turbulence models
    LAMINAR = "laminar"
    K_EPSILON = "kEpsilon"
    K_OMEGA = "kOmega"
    K_OMEGA_SST = "kOmegaSST"
    SPALART_ALLMARAS = "SpalartAllmaras"
    LES_SMAGORINSKY = "Smagorinsky"
    
    def __init__(self, case_dir: Union[str, Path], solver_type: str = SIMPLE_FOAM):
        """
        Initialize the single phase solver.
        
        Args:
            case_dir (Union[str, Path]): OpenFOAM case directory
            solver_type (str): Type of OpenFOAM solver
        """
        # Convert case_dir to Path object if it's a string
        if isinstance(case_dir, str):
            self.case_dir = Path(case_dir)
        else:
            self.case_dir = case_dir
        
        self.solver_type = solver_type
        
        # Default solver settings
        self.settings = {
            "steady_state": solver_type in [self.SIMPLE_FOAM, self.RHO_SIMPLE_FOAM],
            "compressible": solver_type in [self.RHO_SIMPLE_FOAM, self.RHO_PIMPLE_FOAM],
            "turbulence_model": self.K_EPSILON,
            "max_iterations": 1000,
            "convergence_tolerance": 1e-5,
            "start_time": 0,
            "end_time": 1000,
            "delta_t": 0.001,
            "write_interval": 100,
            "purge_write": 0,
            "write_format": "binary",
            "run_time_modifiable": True,
            "max_co": 1.0
        }
        
        # Solver-specific settings
        if solver_type == self.SIMPLE_FOAM:
            self.settings.update({
                "n_non_orthogonal_correctors": 0,
                "p_reference_cell": 0,
                "p_reference_value": 0,
                "solver_relaxation_factors": {
                    "p": 0.3,
                    "U": 0.7,
                    "k": 0.7,
                    "epsilon": 0.7,
                    "omega": 0.7
                }
            })
        elif solver_type == self.PISO_FOAM:
            self.settings.update({
                "n_correctors": 2,
                "n_non_orthogonal_correctors": 1,
                "p_reference_cell": 0,
                "p_reference_value": 0
            })
        
        # Transport properties
        self.transport_model = "Newtonian"
        self.transport_properties = {
            "density": 1000.0,
            "viscosity": 0.001,
            "temperature": 293.15
        }
        
        # Boundary conditions
        self.boundary_conditions = {}
        
        # Numerical schemes
        self.schemes = {
            "default_gradient_scheme": "Gauss linear",
            "default_divergence_scheme": "Gauss linear",
            "default_laplacian_scheme": "Gauss linear corrected",
            "default_interpolation_scheme": "linear"
        }
        
        # Solution control
        self.solution_control = {
            "p_solver": "GAMG",
            "p_tolerance": 1e-6,
            "p_relative_tolerance": 0.01,
            "U_solver": "smoothSolver",
            "U_tolerance": 1e-6,
            "U_relative_tolerance": 0.1,
            "turbulence_solver": "smoothSolver",
            "turbulence_tolerance": 1e-6,
            "turbulence_relative_tolerance": 0.1
        }
        
        # Initial conditions
        self.initial_conditions = {
            "p": 0,
            "U": [0, 0, 0],
            "k": 0.1,
            "epsilon": 0.01,
            "omega": 0.1,
            "nut": 0
        }
    
    def set_transport_properties(self, model: str = "Newtonian", **kwargs):
        """
        Set transport properties for the fluid.
        
        Args:
            model (str): Transport model type ("Newtonian" or "nonNewtonian")
            **kwargs: Additional properties (density, viscosity, etc.)
        """
        self.transport_model = model
        
        # Update transport properties with provided values
        for key, value in kwargs.items():
            if key in self.transport_properties or key not in self.settings:
                self.transport_properties[key] = value
    
    def set_turbulence_model(self, model: str = K_EPSILON):
        """
        Set the turbulence model.
        
        Args:
            model (str): Name of the turbulence model
        """
        if model in [self.LAMINAR, self.K_EPSILON, self.K_OMEGA, 
                     self.K_OMEGA_SST, self.SPALART_ALLMARAS, self.LES_SMAGORINSKY]:
            self.settings["turbulence_model"] = model
        else:
            logger.warning(f"Unknown turbulence model: {model}. Using default: {self.settings['turbulence_model']}")
    
    def set_solver_settings(self, **kwargs):
        """
        Set solver settings.
        
        Args:
            **kwargs: Solver settings to update
        """
        # Update settings with provided values
        for key, value in kwargs.items():
            if key in self.settings:
                self.settings[key] = value
            else:
                logger.warning(f"Unknown solver setting: {key}")
    
    def add_boundary_condition(self, name: str, bc_type: str, patch_type: str, **kwargs):
        """
        Add a boundary condition.
        
        Args:
            name (str): Name of the boundary
            bc_type (str): Type of boundary condition
            patch_type (str): Type of patch
            **kwargs: Additional boundary condition parameters
        """
        self.boundary_conditions[name] = {
            "type": bc_type,
            "patch_type": patch_type,
            "properties": kwargs
        }
    
    def set_schemes(self, **kwargs):
        """
        Set numerical schemes.
        
        Args:
            **kwargs: Scheme settings to update
        """
        # Update schemes with provided values
        for key, value in kwargs.items():
            self.schemes[key] = value
    
    def set_solution_control(self, **kwargs):
        """
        Set solution control parameters.
        
        Args:
            **kwargs: Solution control parameters to update
        """
        # Update solution control parameters with provided values
        for key, value in kwargs.items():
            self.solution_control[key] = value
    
    def set_initial_conditions(self, **kwargs):
        """
        Set initial conditions.
        
        Args:
            **kwargs: Initial conditions to update
        """
        # Update initial conditions with provided values
        for key, value in kwargs.items():
            self.initial_conditions[key] = value
    
    def generate_transport_properties(self) -> str:
        """
        Generate transportProperties dictionary content.
        
        Returns:
            str: Content of transportProperties dictionary
        """
        # Create appropriate transport model
        if self.transport_model == "Newtonian":
            transport = NewtonianTransport(
                density=self.transport_properties.get("density", 1000.0),
                viscosity=self.transport_properties.get("viscosity", 0.001),
                temperature=self.transport_properties.get("temperature", 293.15)
            )
            return transport.generate_dict_entry()
        
        elif self.transport_model == "nonNewtonian":
            non_newtonian_type = self.transport_properties.get("nonNewtonian_type", "powerLaw")
            transport = NonNewtonianTransport(
                model_type=non_newtonian_type,
                density=self.transport_properties.get("density", 1000.0),
                k=self.transport_properties.get("k", 1.0),
                n=self.transport_properties.get("n", 0.8),
                temperature=self.transport_properties.get("temperature", 293.15),
                yield_stress=self.transport_properties.get("yield_stress", 0.0),
                viscosity_inf=self.transport_properties.get("viscosity_inf", 1e-4),
                viscosity_0=self.transport_properties.get("viscosity_0", 1.0)
            )
            return transport.generate_dict_entry()
        
        else:
            logger.warning(f"Unknown transport model: {self.transport_model}. Using Newtonian.")
            transport = NewtonianTransport()
            return transport.generate_dict_entry()
    
    def generate_turbulence_properties(self) -> str:
        """
        Generate turbulenceProperties dictionary content.
        
        Returns:
            str: Content of turbulenceProperties dictionary
        """
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
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
    object      turbulenceProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""
        
        if self.settings["turbulence_model"] == self.LAMINAR:
            content += """simulationType  laminar;
"""
        else:
            content += f"""simulationType  RAS;

RAS
{{
    RASModel        {self.settings["turbulence_model"]};
    turbulence      on;
    printCoeffs     on;
}}
"""
        
        content += """
// ************************************************************************* //
"""
        return content
    
    def generate_control_dict(self) -> str:
        """
        Generate controlDict dictionary content.
        
        Returns:
            str: Content of controlDict dictionary
        """
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
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
    object      controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""
        
        # Add application (solver type)
        content += f"application     {self.solver_type};\n\n"
        
        # Add time control
        content += f"""startFrom       startTime;
startTime       {self.settings['start_time']};
stopAt          endTime;
endTime         {self.settings['end_time']};
deltaT          {self.settings['delta_t']};
writeControl    {"runTime" if self.settings['steady_state'] else "timeStep"};
writeInterval   {self.settings['write_interval']};
purgeWrite      {self.settings['purge_write']};
writeFormat     {self.settings['write_format']};
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable {"yes" if self.settings['run_time_modifiable'] else "no"};
"""
        
        # Add adjustTimeStep for transient simulations
        if not self.settings["steady_state"]:
            content += f"""
// Time step adjustment settings
adjustTimeStep  yes;
maxCo           {self.settings['max_co']};
"""
        
        content += """
functions
{
    // Add any function objects here
}

// ************************************************************************* //
"""
        return content
    
    def generate_fv_schemes(self) -> str:
        """
        Generate fvSchemes dictionary content.
        
        Returns:
            str: Content of fvSchemes dictionary
        """
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
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
    object      fvSchemes;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{
    default         %s;
}

gradSchemes
{
    default         %s;
    grad(p)         %s;
}

divSchemes
{
    default         none;
    div(phi,U)      %s;
""" % (
        "steady-state" if self.settings["steady_state"] else "Euler",
        self.schemes.get("default_gradient_scheme", "Gauss linear"),
        self.schemes.get("p_gradient_scheme", self.schemes.get("default_gradient_scheme", "Gauss linear")),
        self.schemes.get("div_phi_u_scheme", "Gauss upwind")
    )
        
        # Add turbulence schemes if not laminar
        if self.settings["turbulence_model"] != self.LAMINAR:
            if self.settings["turbulence_model"] in [self.K_EPSILON, self.K_OMEGA, self.K_OMEGA_SST]:
                content += f"""    div(phi,k)      {self.schemes.get("div_phi_k_scheme", "Gauss upwind")};
"""
                
                if self.settings["turbulence_model"] == self.K_EPSILON:
                    content += f"""    div(phi,epsilon) {self.schemes.get("div_phi_epsilon_scheme", "Gauss upwind")};
"""
                elif self.settings["turbulence_model"] in [self.K_OMEGA, self.K_OMEGA_SST]:
                    content += f"""    div(phi,omega)  {self.schemes.get("div_phi_omega_scheme", "Gauss upwind")};
"""
            
            elif self.settings["turbulence_model"] == self.SPALART_ALLMARAS:
                content += f"""    div(phi,nuTilda) {self.schemes.get("div_phi_nutilda_scheme", "Gauss upwind")};
"""
        
        # Add div(div(phi,U)) for second-order schemes
        if "div_phi_u_scheme" in self.schemes and "bounded" not in self.schemes["div_phi_u_scheme"]:
            content += f"""    div((nuEff*dev2(T(grad(U))))) {self.schemes.get("div_dev_scheme", "Gauss linear")};
"""
        
        content += f"""}}

laplacianSchemes
{{
    default         {self.schemes.get("default_laplacian_scheme", "Gauss linear corrected")};
}}

interpolationSchemes
{{
    default         {self.schemes.get("default_interpolation_scheme", "linear")};
}}

snGradSchemes
{{
    default         {self.schemes.get("default_sn_grad_scheme", "corrected")};
}}

// ************************************************************************* //
"""
        return content
    
    def generate_fv_solution(self) -> str:
        """
        Generate fvSolution dictionary content.
        
        Returns:
            str: Content of fvSolution dictionary
        """
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
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
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{
    p
    {
        solver          %s;
        tolerance       %g;
        relTol          %g;
""" % (
        self.solution_control.get("p_solver", "GAMG"),
        self.solution_control.get("p_tolerance", 1e-6),
        self.solution_control.get("p_relative_tolerance", 0.01)
    )
        
        # Add GAMG-specific settings
        if self.solution_control.get("p_solver", "GAMG") == "GAMG":
            content += """        smoother        GaussSeidel;
        nPreSweeps      0;
        nPostSweeps     2;
        cacheAgglomeration on;
        agglomerator    faceAreaPair;
        nCellsInCoarsestLevel 10;
        mergeLevels     1;
"""
            
        content += """    }

    U
    {
        solver          %s;
        smoother        %s;
        tolerance       %g;
        relTol          %g;
        nSweeps         %d;
    }
""" % (
        self.solution_control.get("U_solver", "smoothSolver"),
        self.solution_control.get("U_smoother", "GaussSeidel"),
        self.solution_control.get("U_tolerance", 1e-6),
        self.solution_control.get("U_relative_tolerance", 0.1),
        self.solution_control.get("U_sweeps", 1)
    )
        
        # Add turbulence solvers if not laminar
        if self.settings["turbulence_model"] != self.LAMINAR:
            if self.settings["turbulence_model"] in [self.K_EPSILON, self.K_OMEGA, self.K_OMEGA_SST]:
                if self.settings["turbulence_model"] == self.K_EPSILON:
                    vars_string = "(k|epsilon)"
                else:
                    vars_string = "(k|omega)"
            elif self.settings["turbulence_model"] == self.SPALART_ALLMARAS:
                vars_string = "nuTilda"
            else:
                vars_string = "(k|epsilon|omega|nuTilda)"
            
            content += """
    %s
    {
        solver          %s;
        smoother        %s;
        tolerance       %g;
        relTol          %g;
        nSweeps         %d;
    }
""" % (
        vars_string,
        self.solution_control.get("turbulence_solver", "smoothSolver"),
        self.solution_control.get("turbulence_smoother", "GaussSeidel"),
        self.solution_control.get("turbulence_tolerance", 1e-6),
        self.solution_control.get("turbulence_relative_tolerance", 0.1),
        self.solution_control.get("turbulence_sweeps", 1)
    )
        
        # Add SIMPLE/PISO/PIMPLE algorithm settings
        if self.solver_type == self.SIMPLE_FOAM or self.solver_type == self.RHO_SIMPLE_FOAM:
            content += """
SIMPLE
{
    nNonOrthogonalCorrectors %d;
    consistent      yes;

    residualControl
    {
        p               %g;
        U               %g;
""" % (
        self.settings.get("n_non_orthogonal_correctors", 0),
        self.settings.get("convergence_tolerance", 1e-5),
        self.settings.get("convergence_tolerance", 1e-5)
    )
            
            # Add turbulence residual control if not laminar
            if self.settings["turbulence_model"] != self.LAMINAR:
                if self.settings["turbulence_model"] in [self.K_EPSILON, self.K_OMEGA, self.K_OMEGA_SST]:
                    content += f"        \"(k|epsilon|omega)\" {self.settings.get('convergence_tolerance', 1e-5)};\n"
                elif self.settings["turbulence_model"] == self.SPALART_ALLMARAS:
                    content += f"        nuTilda         {self.settings.get('convergence_tolerance', 1e-5)};\n"
                
            content += """    }
}

relaxationFactors
{
    equations
    {
        p               %g;
        U               %g;
""" % (
        self.settings.get("solver_relaxation_factors", {}).get("p", 0.3),
        self.settings.get("solver_relaxation_factors", {}).get("U", 0.7)
    )
            
            # Add turbulence relaxation factors if not laminar
            if self.settings["turbulence_model"] != self.LAMINAR:
                if self.settings["turbulence_model"] == self.K_EPSILON:
                    content += f"""        k               {self.settings.get("solver_relaxation_factors", {}).get("k", 0.7)};
        epsilon         {self.settings.get("solver_relaxation_factors", {}).get("epsilon", 0.7)};
"""
                elif self.settings["turbulence_model"] in [self.K_OMEGA, self.K_OMEGA_SST]:
                    content += f"""        k               {self.settings.get("solver_relaxation_factors", {}).get("k", 0.7)};
        omega           {self.settings.get("solver_relaxation_factors", {}).get("omega", 0.7)};
"""
                elif self.settings["turbulence_model"] == self.SPALART_ALLMARAS:
                    content += f"""        nuTilda         {self.settings.get("solver_relaxation_factors", {}).get("nuTilda", 0.7)};
"""
            
            content += """    }
}
"""
        else:  # PISO or PIMPLE
            content += """
PISO
{
    nCorrectors     %d;
    nNonOrthogonalCorrectors %d;
    pRefCell        %d;
    pRefValue       %g;
}
""" % (
        self.settings.get("n_correctors", 2),
        self.settings.get("n_non_orthogonal_correctors", 1),
        self.settings.get("p_reference_cell", 0),
        self.settings.get("p_reference_value", 0)
    )
        
        content += """
// ************************************************************************* //
"""
        return content
    
    def generate_boundary_conditions(self) -> Dict[str, str]:
        """
        Generate boundary condition files content.
        
        Returns:
            Dict[str, str]: Dictionary mapping field names to file contents
        """
        # Maps of field names by solver type
        field_maps = {
            self.SIMPLE_FOAM: ["p", "U"],
            self.PISO_FOAM: ["p", "U"],
            self.RHO_SIMPLE_FOAM: ["p", "U", "T"],
            self.RHO_PIMPLE_FOAM: ["p", "U", "T"]
        }
        
        # Add turbulence fields based on model
        if self.settings["turbulence_model"] != self.LAMINAR:
            if self.settings["turbulence_model"] == self.K_EPSILON:
                for solver in field_maps:
                    field_maps[solver].extend(["k", "epsilon", "nut"])
            elif self.settings["turbulence_model"] in [self.K_OMEGA, self.K_OMEGA_SST]:
                for solver in field_maps:
                    field_maps[solver].extend(["k", "omega", "nut"])
            elif self.settings["turbulence_model"] == self.SPALART_ALLMARAS:
                for solver in field_maps:
                    field_maps[solver].extend(["nuTilda", "nut"])
            elif self.settings["turbulence_model"] == self.LES_SMAGORINSKY:
                for solver in field_maps:
                    field_maps[solver].extend(["k", "nut"])
        
        # Get fields for current solver
        fields = field_maps.get(self.solver_type, ["p", "U"])
        
        # Create boundary condition files for each field
        bc_files = {}
        for field in fields:
            bc_files[field] = self._generate_field_file(field)
        
        return bc_files
    
    def _generate_field_file(self, field_name: str) -> str:
        """
        Generate boundary condition file for a specific field.
        
        Args:
            field_name (str): Name of the field (p, U, k, etc.)
            
        Returns:
            str: Content of the field file
        """
        header = f"""/*--------------------------------*- C++ -*----------------------------------*\\
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
    class       volScalar{"Vector" if field_name == "U" else "Field"};
    object      {field_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [{self._get_field_dimensions(field_name)}];

internalField   uniform {self._format_internal_field_value(self.initial_conditions.get(field_name, 0))};

boundaryField
{{
"""
        
        # Process boundary conditions
        for boundary_name, bc_info in self.boundary_conditions.items():
            header += f"    {boundary_name}\n    {{\n"
            
            # Get field-specific boundary condition or use default
            if field_name in bc_info.get("properties", {}):
                bc_type = bc_info["properties"][field_name].get("type", "zeroGradient")
                header += f"        type            {bc_type};\n"
                
                # Add additional properties if present
                for prop_name, prop_value in bc_info["properties"][field_name].items():
                    if prop_name != "type":
                        header += f"        {prop_name}            {self._format_value(prop_value)};\n"
            else:
                # Use default boundary condition based on patch type and field
                header += f"        type            {self._get_default_bc_type(bc_info['patch_type'], field_name)};\n"
            
            header += "    }\n"
        
        header += """
}

// ************************************************************************* //
"""
        return header
    
    def _get_field_dimensions(self, field_name: str) -> str:
        """
        Get the dimensions string for a field.
        
        Args:
            field_name (str): Name of the field
            
        Returns:
            str: Dimensions string in OpenFOAM format
        """
        dimensions = {
            "p": "0 2 -2 0 0 0 0",  # Pressure: m²/s²
            "U": "0 1 -1 0 0 0 0",  # Velocity: m/s
            "k": "0 2 -2 0 0 0 0",  # Turbulent kinetic energy: m²/s²
            "epsilon": "0 2 -3 0 0 0 0",  # Dissipation rate: m²/s³
            "omega": "0 0 -1 0 0 0 0",  # Specific dissipation rate: 1/s
            "nut": "0 2 -1 0 0 0 0",  # Turbulent viscosity: m²/s
            "nuTilda": "0 2 -1 0 0 0 0",  # Spalart-Allmaras variable: m²/s
            "T": "0 0 0 1 0 0 0"  # Temperature: K
        }
        
        return dimensions.get(field_name, "0 0 0 0 0 0 0")
    
    def _get_default_bc_type(self, patch_type: str, field_name: str) -> str:
        """
        Get the default boundary condition type for a patch.
        
        Args:
            patch_type (str): Type of patch (wall, inlet, outlet, etc.)
            field_name (str): Name of the field
            
        Returns:
            str: Default boundary condition type
        """
        if patch_type == "wall":
            if field_name == "U":
                return "noSlip"
            elif field_name in ["k", "epsilon", "omega"]:
                return "kqRWallFunction"
            elif field_name == "nut":
                return "nutkWallFunction"
            elif field_name == "T":
                return "zeroGradient"
            else:
                return "zeroGradient"
        
        elif patch_type == "inlet":
            if field_name == "U":
                return "fixedValue"
            elif field_name == "p":
                return "zeroGradient"
            elif field_name in ["k", "epsilon", "omega"]:
                return "fixedValue"
            elif field_name == "nut":
                return "calculated"
            elif field_name == "T":
                return "fixedValue"
            else:
                return "zeroGradient"
        
        elif patch_type == "outlet":
            if field_name == "U":
                return "zeroGradient"
            elif field_name == "p":
                return "fixedValue"
            elif field_name in ["k", "epsilon", "omega"]:
                return "zeroGradient"
            elif field_name == "nut":
                return "calculated"
            elif field_name == "T":
                return "zeroGradient"
            else:
                return "zeroGradient"
        
        elif patch_type == "symmetry":
            return "symmetry"
        
        elif patch_type == "cyclic":
            return "cyclic"
        
        else:
            return "zeroGradient"
    
    def _format_internal_field_value(self, value) -> str:
        """
        Format internal field value for OpenFOAM.
        
        Args:
            value: Value to format
            
        Returns:
            str: Formatted value
        """
        return self._format_value(value)
    
    def _format_value(self, value) -> str:
        """
        Format a value for OpenFOAM.
        
        Args:
            value: Value to format
            
        Returns:
            str: Formatted value
        """
        if isinstance(value, (list, tuple)):
            if len(value) == 3:  # Vector
                return f"({value[0]} {value[1]} {value[2]})"
            elif len(value) == 6:  # Tensor
                return f"({value[0]} {value[1]} {value[2]} {value[3]} {value[4]} {value[5]})"
            else:
                return f"{value}"  # Default to string representation
        
        elif isinstance(value, bool):
            return "on" if value else "off"
        
        elif isinstance(value, (int, float)):
            return f"{value}"
        
        else:
            return f"{value}"
    
    def setup_case(self) -> bool:
        """
        Set up the OpenFOAM case.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create required directories
            system_dir = self.case_dir / "system"
            constant_dir = self.case_dir / "constant"
            time_dir = self.case_dir / "0"
            
            system_dir.mkdir(parents=True, exist_ok=True)
            constant_dir.mkdir(parents=True, exist_ok=True)
            time_dir.mkdir(parents=True, exist_ok=True)
            
            # Create controlDict
            with open(system_dir / "controlDict", "w") as f:
                f.write(self.generate_control_dict())
            
            # Create fvSchemes
            with open(system_dir / "fvSchemes", "w") as f:
                f.write(self.generate_fv_schemes())
            
            # Create fvSolution
            with open(system_dir / "fvSolution", "w") as f:
                f.write(self.generate_fv_solution())
            
            # Create transportProperties
            with open(constant_dir / "transportProperties", "w") as f:
                f.write(self.generate_transport_properties())
            
            # Create turbulenceProperties
            with open(constant_dir / "turbulenceProperties", "w") as f:
                f.write(self.generate_turbulence_properties())
            
            # Create boundary conditions
            bc_files = self.generate_boundary_conditions()
            for field_name, content in bc_files.items():
                with open(time_dir / field_name, "w") as f:
                    f.write(content)
            
            logger.info(f"Case setup completed successfully: {self.case_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up case: {e}")
            return False
    
    def run_simulation(self, parallel: bool = False, n_processors: int = 4) -> bool:
        """
        Run the OpenFOAM simulation.
        
        Args:
            parallel (bool): Whether to run in parallel
            n_processors (int): Number of processors for parallel run
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create run directory
            cmd = []
            
            # Add decomposition step if running in parallel
            if parallel:
                # Create decomposeParDict
                decompose_dict = f"""/*--------------------------------*- C++ -*----------------------------------*\\
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

// ************************************************************************* //
"""
                with open(self.case_dir / "system" / "decomposeParDict", "w") as f:
                    f.write(decompose_dict)
                
                # Decompose the case
                cmd.append(f"cd {self.case_dir} && decomposePar -force")
                
                # Run in parallel
                cmd.append(f"cd {self.case_dir} && mpirun -np {n_processors} {self.solver_type} -parallel")
                
                # Reconstruct the case
                cmd.append(f"cd {self.case_dir} && reconstructPar")
            else:
                # Run in serial
                cmd.append(f"cd {self.case_dir} && {self.solver_type}")
            
            # Run the commands
            for command in cmd:
                logger.info(f"Running command: {command}")
                result = os.system(command)
                if result != 0:
                    logger.error(f"Command failed with exit code {result}: {command}")
                    return False
            
            logger.info(f"Simulation completed successfully: {self.case_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error running simulation: {e}")
            return False


def create_pipe_flow_case(case_dir: Union[str, Path], diameter: float = 0.1, 
                         length: float = 1.0, inlet_velocity: float = 1.0,
                         fluid_density: float = 1000.0, fluid_viscosity: float = 0.001,
                         turbulence_model: str = SinglePhaseSolver.K_EPSILON) -> SinglePhaseSolver:
    """
    Create a pipe flow case.
    
    Args:
        case_dir (Union[str, Path]): Case directory
        diameter (float): Pipe diameter in meters
        length (float): Pipe length in meters
        inlet_velocity (float): Inlet velocity in m/s
        fluid_density (float): Fluid density in kg/m³
        fluid_viscosity (float): Fluid viscosity in Pa·s
        turbulence_model (str): Turbulence model
        
    Returns:
        SinglePhaseSolver: Configured solver
    """
    # Create solver
    solver = SinglePhaseSolver(case_dir, SinglePhaseSolver.SIMPLE_FOAM)
    
    # Set transport properties
    solver.set_transport_properties("Newtonian", density=fluid_density, viscosity=fluid_viscosity)
    
    # Set turbulence model
    solver.set_turbulence_model(turbulence_model)
    
    # Set solver settings
    solver.set_solver_settings(
        steady_state=True,
        max_iterations=1000,
        convergence_tolerance=1e-5,
        write_interval=100
    )
    
    # Calculate Reynolds number to determine default schemes
    reynolds = (fluid_density * inlet_velocity * diameter) / fluid_viscosity
    
    # Set schemes
    if reynolds > 2300:  # Turbulent
        solver.set_schemes(
            default_gradient_scheme="Gauss linear",
            div_phi_u_scheme="Gauss linearUpwind grad(U)",
            div_phi_k_scheme="Gauss upwind",
            div_phi_epsilon_scheme="Gauss upwind"
        )
    else:  # Laminar
        solver.set_schemes(
            default_gradient_scheme="Gauss linear",
            div_phi_u_scheme="Gauss linear"
        )
    
    # Set boundary conditions
    solver.add_boundary_condition(
        "inlet",
        "patch",
        "inlet",
        U={"type": "fixedValue", "value": [inlet_velocity, 0, 0]},
        p={"type": "zeroGradient"},
        k={"type": "fixedValue", "value": 0.01},
        epsilon={"type": "fixedValue", "value": 0.01}
    )
    
    solver.add_boundary_condition(
        "outlet",
        "patch",
        "outlet",
        U={"type": "zeroGradient"},
        p={"type": "fixedValue", "value": 0},
        k={"type": "zeroGradient"},
        epsilon={"type": "zeroGradient"}
    )
    
    solver.add_boundary_condition(
        "wall",
        "wall",
        "wall",
        U={"type": "noSlip"},
        p={"type": "zeroGradient"},
        k={"type": "kqRWallFunction", "value": 0.01},
        epsilon={"type": "epsilonWallFunction", "value": 0.01}
    )
    
    # Set initial conditions
    solver.set_initial_conditions(
        p=0,
        U=[inlet_velocity, 0, 0],
        k=0.01,
        epsilon=0.01
    )
    
    return solver


def create_channel_flow_case(case_dir: Union[str, Path], height: float = 0.1, 
                            width: float = 0.1, length: float = 1.0, 
                            inlet_velocity: float = 1.0, transient: bool = False,
                            fluid_density: float = 1000.0, fluid_viscosity: float = 0.001,
                            turbulence_model: str = SinglePhaseSolver.K_EPSILON) -> SinglePhaseSolver:
    """
    Create a channel flow case.
    
    Args:
        case_dir (Union[str, Path]): Case directory
        height (float): Channel height in meters
        width (float): Channel width in meters
        length (float): Channel length in meters
        inlet_velocity (float): Inlet velocity in m/s
        transient (bool): Whether to use transient solver
        fluid_density (float): Fluid density in kg/m³
        fluid_viscosity (float): Fluid viscosity in Pa·s
        turbulence_model (str): Turbulence model
        
    Returns:
        SinglePhaseSolver: Configured solver
    """
    # Create solver
    solver_type = SinglePhaseSolver.PISO_FOAM if transient else SinglePhaseSolver.SIMPLE_FOAM
    solver = SinglePhaseSolver(case_dir, solver_type)
    
    # Set transport properties
    solver.set_transport_properties("Newtonian", density=fluid_density, viscosity=fluid_viscosity)
    
    # Set turbulence model
    solver.set_turbulence_model(turbulence_model)
    
    # Set solver settings
    if transient:
        solver.set_solver_settings(
            steady_state=False,
            start_time=0,
            end_time=10,
            delta_t=0.001,
            write_interval=10,
            max_co=1.0
        )
    else:
        solver.set_solver_settings(
            steady_state=True,
            max_iterations=1000,
            convergence_tolerance=1e-5,
            write_interval=100
        )
    
    # Calculate Reynolds number to determine default schemes
    hydraulic_diameter = (2 * height * width) / (height + width)
    reynolds = (fluid_density * inlet_velocity * hydraulic_diameter) / fluid_viscosity
    
    # Set schemes
    if reynolds > 2300:  # Turbulent
        solver.set_schemes(
            default_gradient_scheme="Gauss linear",
            div_phi_u_scheme="Gauss linearUpwind grad(U)",
            div_phi_k_scheme="Gauss upwind",
            div_phi_epsilon_scheme="Gauss upwind"
        )
    else:  # Laminar
        solver.set_schemes(
            default_gradient_scheme="Gauss linear",
            div_phi_u_scheme="Gauss linear"
        )
    
    # Set boundary conditions
    solver.add_boundary_condition(
        "inlet",
        "patch",
        "inlet",
        U={"type": "fixedValue", "value": [inlet_velocity, 0, 0]},
        p={"type": "zeroGradient"},
        k={"type": "fixedValue", "value": 0.01},
        epsilon={"type": "fixedValue", "value": 0.01}
    )
    
    solver.add_boundary_condition(
        "outlet",
        "patch",
        "outlet",
        U={"type": "zeroGradient"},
        p={"type": "fixedValue", "value": 0},
        k={"type": "zeroGradient"},
        epsilon={"type": "zeroGradient"}
    )
    
    solver.add_boundary_condition(
        "topWall",
        "wall",
        "wall",
        U={"type": "noSlip"},
        p={"type": "zeroGradient"},
        k={"type": "kqRWallFunction", "value": 0.01},
        epsilon={"type": "epsilonWallFunction", "value": 0.01}
    )
    
    solver.add_boundary_condition(
        "bottomWall",
        "wall",
        "wall",
        U={"type": "noSlip"},
        p={"type": "zeroGradient"},
        k={"type": "kqRWallFunction", "value": 0.01},
        epsilon={"type": "epsilonWallFunction", "value": 0.01}
    )
    
    solver.add_boundary_condition(
        "leftWall",
        "wall",
        "wall",
        U={"type": "noSlip"},
        p={"type": "zeroGradient"},
        k={"type": "kqRWallFunction", "value": 0.01},
        epsilon={"type": "epsilonWallFunction", "value": 0.01}
    )
    
    solver.add_boundary_condition(
        "rightWall",
        "wall",
        "wall",
        U={"type": "noSlip"},
        p={"type": "zeroGradient"},
        k={"type": "kqRWallFunction", "value": 0.01},
        epsilon={"type": "epsilonWallFunction", "value": 0.01}
    )
    
    # Set initial conditions
    solver.set_initial_conditions(
        p=0,
        U=[inlet_velocity, 0, 0],
        k=0.01,
        epsilon=0.01
    )
    
    return solver


def create_oilgas_flow_case(case_dir: Union[str, Path], diameter: float = 0.1, 
                          length: float = 1.0, inlet_velocity: float = 1.0,
                          oil_density: float = 850.0, oil_viscosity: float = 0.03,
                          temperature: float = 293.15, pressure: float = 101325.0,
                          solver_type: str = SinglePhaseSolver.SIMPLE_FOAM) -> SinglePhaseSolver:
    """
    Create an oil & gas single phase flow case.
    
    Args:
        case_dir (Union[str, Path]): Case directory
        diameter (float): Pipe diameter in meters
        length (float): Pipe length in meters
        inlet_velocity (float): Inlet velocity in m/s
        oil_density (float): Oil density in kg/m³
        oil_viscosity (float): Oil viscosity in Pa·s
        temperature (float): Temperature in K
        pressure (float): Pressure in Pa
        solver_type (str): Solver type
        
    Returns:
        SinglePhaseSolver: Configured solver
    """
    # Create solver
    solver = SinglePhaseSolver(case_dir, solver_type)
    
    # Set transport properties
    solver.set_transport_properties("Newtonian", density=oil_density, viscosity=oil_viscosity, temperature=temperature)
    
    # Set turbulence model - oil flows are usually turbulent
    solver.set_turbulence_model(SinglePhaseSolver.K_EPSILON)
    
    # Set solver settings
    solver.set_solver_settings(
        steady_state=(solver_type == SinglePhaseSolver.SIMPLE_FOAM),
        max_iterations=2000,
        convergence_tolerance=1e-6,
        write_interval=100,
        max_co=0.9
    )
    
    # Set schemes
    solver.set_schemes(
        default_gradient_scheme="Gauss linear",
        div_phi_u_scheme="Gauss linearUpwind grad(U)",
        div_phi_k_scheme="Gauss upwind",
        div_phi_epsilon_scheme="Gauss upwind"
    )
    
    # Set boundary conditions
    solver.add_boundary_condition(
        "inlet",
        "patch",
        "inlet",
        U={"type": "fixedValue", "value": [inlet_velocity, 0, 0]},
        p={"type": "zeroGradient"},
        k={"type": "fixedValue", "value": 0.01},
        epsilon={"type": "fixedValue", "value": 0.01},
        T={"type": "fixedValue", "value": temperature}
    )
    
    solver.add_boundary_condition(
        "outlet",
        "patch",
        "outlet",
        U={"type": "zeroGradient"},
        p={"type": "fixedValue", "value": pressure},
        k={"type": "zeroGradient"},
        epsilon={"type": "zeroGradient"},
        T={"type": "zeroGradient"}
    )
    
    solver.add_boundary_condition(
        "wall",
        "wall",
        "wall",
        U={"type": "noSlip"},
        p={"type": "zeroGradient"},
        k={"type": "kqRWallFunction", "value": 0.01},
        epsilon={"type": "epsilonWallFunction", "value": 0.01},
        T={"type": "fixedValue", "value": temperature}
    )
    
    # Set initial conditions
    solver.set_initial_conditions(
        p=pressure,
        U=[inlet_velocity, 0, 0],
        k=0.01,
        epsilon=0.01,
        T=temperature
    )
    
    return solver


def create_from_template(case_dir: Union[str, Path], template_dir: Union[str, Path], 
                        parameters: Dict[str, Any]) -> SinglePhaseSolver:
    """
    Create a case from a template.
    
    Args:
        case_dir (Union[str, Path]): Target case directory
        template_dir (Union[str, Path]): Template directory
        parameters (Dict[str, Any]): Parameters to customize the template
        
    Returns:
        SinglePhaseSolver: Configured solver
    """
    # Convert to Path objects
    if isinstance(case_dir, str):
        case_dir = Path(case_dir)
    if isinstance(template_dir, str):
        template_dir = Path(template_dir)
    
    try:
        # Create case directory if it doesn't exist
        case_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy template files
        for src_path in template_dir.glob("**/*"):
            if src_path.is_file():
                rel_path = src_path.relative_to(template_dir)
                dst_path = case_dir / rel_path
                
                # Create parent directories if they don't exist
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Read template content
                with open(src_path, "r") as f:
                    content = f.read()
                
                # Replace placeholders with parameter values
                for key, value in parameters.items():
                    placeholder = f"@{key}@"
                    if placeholder in content:
                        content = content.replace(placeholder, str(value))
                
                # Write customized content
                with open(dst_path, "w") as f:
                    f.write(content)
        
        # Create solver from case directory
        solver = SinglePhaseSolver(case_dir)
        
        # Load solver settings from case directory if they exist
        if (case_dir / "system" / "controlDict").exists():
            # TODO: Parse OpenFOAM files to load settings
            # This would require a more complex parser for OpenFOAM dictionary files
            pass
        
        return solver
        
    except Exception as e:
        logger.error(f"Error creating case from template: {e}")
        return SinglePhaseSolver(case_dir)


# Example usage
if __name__ == "__main__":
    print("OpenFOAM Single Phase Solver Module")
    
    # Example: Create a pipe flow case
    case_dir = Path("./pipe_flow_case")
    solver = create_pipe_flow_case(case_dir, diameter=0.1, length=1.0, inlet_velocity=1.0)
    
    # Setup and run the case
    success = solver.setup_case()
    if success:
        print(f"Case setup successfully: {case_dir}")
        
        # Uncomment to run the simulation
        # success = solver.run_simulation()
        # if success:
        #     print(f"Simulation completed successfully: {case_dir}")
        # else:
        #     print(f"Simulation failed: {case_dir}")
    else:
        print(f"Case setup failed: {case_dir}")