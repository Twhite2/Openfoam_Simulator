#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-phase flow solver setup for Openfoam_Simulator.

This module provides classes and functions to set up and configure
OpenFOAM solvers for multi-phase flow simulations, including:
- Two-phase flows (interFoam)
- Multi-phase flows (multiphaseInterFoam)
- Volume of Fluid (VOF) method configuration
- Interface capturing and compression settings
- Phase properties and interface physics
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
    from openfoam_integration.transport_models import MultiPhaseTransport
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


class MultiPhaseSolver:
    """
    Base class for multi-phase flow solvers.
    
    This class provides the common interface and functionality
    for all multi-phase flow solvers in OpenFOAM.
    """
    
    # Solver types
    INTER_FOAM = "interFoam"                # Two-phase incompressible
    MULTIPHASE_INTER_FOAM = "multiphaseInterFoam"  # Multi-phase incompressible
    COMPRESSIBLE_INTER_FOAM = "compressibleInterFoam"  # Two-phase compressible
    TWO_PHASE_EULER_FOAM = "twoPhaseEulerFoam"  # Two-phase Eulerian
    REACTIVE_INTER_FOAM = "reactingMultiphaseInterFoam"  # Reacting multi-phase
    
    # Turbulence models
    LAMINAR = "laminar"
    K_EPSILON = "kEpsilon"
    K_OMEGA = "kOmega"
    K_OMEGA_SST = "kOmegaSST"
    MIXTURE_K_EPSILON = "mixtureKEpsilon"
    
    def __init__(self, case_dir: Union[str, Path], solver_type: str = INTER_FOAM):
        """
        Initialize the multi-phase solver.
        
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
            "compressible": solver_type == self.COMPRESSIBLE_INTER_FOAM,
            "turbulence_model": self.K_EPSILON,
            "max_iterations": 2000,
            "convergence_tolerance": 1e-6,
            "start_time": 0,
            "end_time": 1,
            "delta_t": 0.001,
            "write_interval": 0.01,
            "purge_write": 0,
            "write_format": "binary",
            "run_time_modifiable": True,
            "max_co": 0.5,  # Lower Courant number for multi-phase
            "max_alpha_co": 0.2,  # Alpha Courant number for interface
            "interface_compression": 1.0,  # Interface compression factor
            "p_reference_cell": 0,
            "p_reference_value": 0
        }
        
        # Phase names and properties
        self.phases = []  # List of phase names
        self.phase_properties = {}  # Dict of phase properties
        
        # Interface properties for each phase pair
        self.interface_properties = {}
        
        # Boundary conditions
        self.boundary_conditions = {}
        
        # Numerical schemes
        self.schemes = {
            "default_gradient_scheme": "Gauss linear",
            "default_divergence_scheme": "Gauss linear",
            "default_laplacian_scheme": "Gauss linear corrected",
            "default_interpolation_scheme": "linear",
            "alpha_divergence_scheme": "Gauss vanLeer",
            "alpha_interface_compression_scheme": "Gauss interfaceCompression phi (alpha.water alpha.air)"
        }
        
        # Solution control
        self.solution_control = {
            "p_rgh_solver": "PCG",
            "p_rgh_preconditioner": "DIC",
            "p_rgh_tolerance": 1e-7,
            "p_rgh_relative_tolerance": 0.01,
            "p_final_tolerance": 1e-7,
            "p_correctors": 3,
            "non_orthogonal_correctors": 1,
            "phase_correctors": 1,
            "relaxation_factor_p_rgh": 0.3,
            "relaxation_factor_U": 0.7,
            "relaxation_factor_alpha": 0.3,
            "relaxation_factor_k": 0.7,
            "relaxation_factor_epsilon": 0.7,
            "momentumPredictor": True,
            "nOuterCorrectors": 1,
            "nCorrectors": 3
        }
        
        # Initial conditions
        self.initial_conditions = {
            "p_rgh": 0,
            "U": [0, 0, 0],
            "k": 0.1,
            "epsilon": 0.01,
            "omega": 0.1,
            "nut": 0
        }
        
        # For now, default to water-air two-phase system
        self.add_phase("water", density=1000.0, viscosity=0.001, volume_fraction=0.0)
        self.add_phase("air", density=1.2, viscosity=1.8e-5, volume_fraction=1.0)
    
    def add_phase(self, name: str, density: float, viscosity: float, volume_fraction: float = 0.0,
                  transport_model: str = "Newtonian", **kwargs):
        """
        Add a phase to the multi-phase system.
        
        Args:
            name (str): Name of the phase
            density (float): Density of the phase in kg/m³
            viscosity (float): Dynamic viscosity of the phase in Pa·s
            volume_fraction (float): Initial volume fraction of the phase
            transport_model (str): Transport model for the phase
            **kwargs: Additional phase properties
        """
        # Add phase name to list if not already there
        if name not in self.phases:
            self.phases.append(name)
        
        # Set phase properties
        self.phase_properties[name] = {
            "density": density,
            "viscosity": viscosity,
            "transportModel": transport_model,
            "initial_volume_fraction": volume_fraction,
            **kwargs
        }
        
        # Add volume fraction initial condition
        field_name = f"alpha.{name}"
        self.initial_conditions[field_name] = volume_fraction
    
    def set_interface_property(self, phase1: str, phase2: str, property_name: str, value: float):
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
    
    def set_surface_tension(self, phase1: str, phase2: str, sigma: float):
        """
        Set surface tension between two phases.
        
        Args:
            phase1 (str): First phase name
            phase2 (str): Second phase name
            sigma (float): Surface tension in N/m
        """
        self.set_interface_property(phase1, phase2, "sigma", sigma)
    
    def set_turbulence_model(self, model: str = K_EPSILON):
        """
        Set the turbulence model.
        
        Args:
            model (str): Name of the turbulence model
        """
        if model in [self.LAMINAR, self.K_EPSILON, self.K_OMEGA, 
                     self.K_OMEGA_SST, self.MIXTURE_K_EPSILON]:
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
        # Create phase property objects for multi-phase transport
        phases_list = []
        phase_names = []
        
        for name in self.phases:
            props = self.phase_properties[name]
            phases_list.append({
                "transportModel": props.get("transportModel", "Newtonian"),
                "density": props.get("density", 1000.0),
                "viscosity": props.get("viscosity", 0.001)
            })
            phase_names.append(name)
        
        # Create transport model
        transport = MultiPhaseTransport(phases_list, phase_names)
        
        # Add interface properties
        for (phase1, phase2), properties in self.interface_properties.items():
            for property_name, value in properties.items():
                transport.set_interface_property(phase1, phase2, property_name, value)
        
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
            # For multi-phase, need to decide between per-phase or mixture turbulence
            if self.settings["turbulence_model"] == self.MIXTURE_K_EPSILON:
                content += f"""simulationType  RAS;

RAS
{{
    RASModel        mixtureKEpsilon;
    turbulence      on;
    printCoeffs     on;
}}
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
    
    def generate_phase_field_dict(self) -> str:
        """
        Generate phaseFieldProperties dictionary for multi-phase solvers.
        
        Returns:
            str: Content of phaseFieldProperties dictionary
        """
        if self.solver_type not in [self.MULTIPHASE_INTER_FOAM, self.REACTIVE_INTER_FOAM]:
            # Only needed for multi-phase solvers with more than 2 phases
            return ""
        
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
    object      phaseProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

type    multiphaseInterFoam;

phases  ("""
        
        # Add phase names
        for name in self.phases:
            content += f" {name}"
        
        content += " );\n\n"
        
        # Add phase pairs for surface tension
        if self.phases and len(self.phases) > 1:
            content += "phasePairs\n(\n"
            
            # Generate all pairs
            for i, phase1 in enumerate(self.phases):
                for phase2 in self.phases[i+1:]:
                    pair = tuple(sorted([phase1, phase2]))
                    
                    content += f"    ({phase1} {phase2})\n    {{\n"
                    
                    # Add surface tension if defined
                    if pair in self.interface_properties and "sigma" in self.interface_properties[pair]:
                        sigma = self.interface_properties[pair]["sigma"]
                        content += f"        sigma           {sigma:.6e};\n"
                    else:
                        # Default surface tension
                        content += f"        sigma           0.07;\n"
                    
                    content += "    }\n"
            
            content += ");\n"
        
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
writeControl    adjustableRunTime;
writeInterval   {self.settings['write_interval']};
purgeWrite      {self.settings['purge_write']};
writeFormat     {self.settings['write_format']};
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable {"yes" if self.settings['run_time_modifiable'] else "no"};

// Time step adjustment settings
adjustTimeStep  yes;
maxCo           {self.settings['max_co']};
maxAlphaCo      {self.settings['max_alpha_co']};
maxDeltaT       1;

"""
        
        # Add function objects for postprocessing
        content += """functions
{
    // Function objects for phase fraction statistics, forces, etc.
    volumeFractions
    {
        type            volFieldValue;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
        log             true;
        writeFields     false;
        operation       volIntegrate;
        fields          (alpha.*);
    }
    
    // Force on patches
    forces
    {
        type            forces;
        libs            (forces);
        writeControl    writeTime;
        patches         (wall);
        rho             rho;
        log             true;
    }
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
    default         Euler;
}

gradSchemes
{
    default         %s;
    grad(p_rgh)     %s;
}

divSchemes
{
    default         none;
    div(phi,U)      %s;
""" % (
        self.schemes.get("default_gradient_scheme", "Gauss linear"),
        self.schemes.get("p_rgh_gradient_scheme", self.schemes.get("default_gradient_scheme", "Gauss linear")),
        self.schemes.get("div_phi_u_scheme", "Gauss upwind")
    )
        
        # Add phase fraction schemes for VOF method
        alpha_scheme = self.schemes.get("alpha_divergence_scheme", "Gauss vanLeer")
        compression_scheme = self.schemes.get("alpha_interface_compression_scheme", 
                                             f"Gauss interfaceCompression phi (alpha.{self.phases[0]} alpha.{self.phases[1]})")
        
        if len(self.phases) == 2:
            # Two-phase simulation
            content += f"""    div(phi,alpha.{self.phases[0]})      {alpha_scheme};
    div(phir,alpha.{self.phases[0]})     {compression_scheme};
"""
        else:
            # Multi-phase simulation
            for phase in self.phases:
                content += f"""    div(phi,alpha.{phase})      {alpha_scheme};
    div(phir,alpha.{phase})     {alpha_scheme};
"""
            
        # Add turbulence schemes if not laminar
        if self.settings["turbulence_model"] != self.LAMINAR:
            if self.settings["turbulence_model"] in [self.K_EPSILON, self.K_OMEGA, self.K_OMEGA_SST, self.MIXTURE_K_EPSILON]:
                content += f"""    div(phi,k)      {self.schemes.get("div_phi_k_scheme", "Gauss upwind")};
"""
                
                if self.settings["turbulence_model"] in [self.K_EPSILON, self.MIXTURE_K_EPSILON]:
                    content += f"""    div(phi,epsilon) {self.schemes.get("div_phi_epsilon_scheme", "Gauss upwind")};
"""
                elif self.settings["turbulence_model"] in [self.K_OMEGA, self.K_OMEGA_SST]:
                    content += f"""    div(phi,omega)  {self.schemes.get("div_phi_omega_scheme", "Gauss upwind")};
"""
        
        # Add additional schemes for multiphase solver
        content += f"""    div(((rho*nuEff)*dev2(T(grad(U)))))    {self.schemes.get("div_dev_scheme", "Gauss linear")};
}}

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

// For interface compression
fluxRequired
{{
    default         no;
    p_rgh;"""

        # Add alpha fields to fluxRequired
        for phase in self.phases:
            content += f"""
    alpha.{phase};"""
        
        content += """
}

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
    "alpha.*"
    {
        nAlphaCorr      %d;
        nAlphaSubCycles %d;
        cAlpha          %g;
        
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0;
    }

    "pcorr.*"
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-5;
        relTol          0;
    }

    p_rgh
    {
        solver          %s;
        preconditioner  %s;
        tolerance       %g;
        relTol          %g;
    }

    p_rghFinal
    {
        $p_rgh;
        tolerance       %g;
        relTol          0;
    }

    "(U|k|epsilon|omega)"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       %g;
        relTol          %g;
        nSweeps         %d;
    }
}

PIMPLE
{
    momentumPredictor   %s;
    nOuterCorrectors    %d;
    nCorrectors         %d;
    nNonOrthogonalCorrectors %d;
    pRefCell            %d;
    pRefValue           %g;
}

relaxationFactors
{
    equations
    {
        ".*"            1;
    }
}

// ************************************************************************* //
""" % (
        self.solution_control.get("phase_correctors", 1),
        self.solution_control.get("phase_subcycles", 3),
        self.settings.get("interface_compression", 1.0),
        self.solution_control.get("p_rgh_solver", "PCG"),
        self.solution_control.get("p_rgh_preconditioner", "DIC"),
        self.solution_control.get("p_rgh_tolerance", 1e-7),
        self.solution_control.get("p_rgh_relative_tolerance", 0.01),
        self.solution_control.get("p_final_tolerance", 1e-7),
        self.solution_control.get("turbulence_tolerance", 1e-6),
        self.solution_control.get("turbulence_relative_tolerance", 0.1),
        self.solution_control.get("turbulence_sweeps", 1),
        "yes" if self.solution_control.get("momentumPredictor", True) else "no",
        self.solution_control.get("nOuterCorrectors", 1),
        self.solution_control.get("nCorrectors", 3),
        self.solution_control.get("non_orthogonal_correctors", 1),
        self.settings.get("p_reference_cell", 0),
        self.settings.get("p_reference_value", 0)
    )
        
        return content
    
    def generate_boundary_conditions(self) -> Dict[str, str]:
        """
        Generate boundary condition files content.
        
        Returns:
            Dict[str, str]: Dictionary mapping field names to file contents
        """
        # Base fields for all VOF solvers
        fields = ["p_rgh", "U"]
        
        # Add phase fields
        for phase in self.phases:
            fields.append(f"alpha.{phase}")
        
        # Add turbulence fields based on model
        if self.settings["turbulence_model"] != self.LAMINAR:
            if self.settings["turbulence_model"] in [self.K_EPSILON, self.MIXTURE_K_EPSILON]:
                fields.extend(["k", "epsilon", "nut"])
            elif self.settings["turbulence_model"] in [self.K_OMEGA, self.K_OMEGA_SST]:
                fields.extend(["k", "omega", "nut"])
        
        # Create boundary condition files for each field
        bc_files = {}
        for field in fields:
            bc_files[field] = self._generate_field_file(field)
        
        return bc_files
    
    def _generate_field_file(self, field_name: str) -> str:
        """
        Generate boundary condition file for a specific field.
        
        Args:
            field_name (str): Name of the field (p_rgh, U, alpha.*, etc.)
            
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
            "p_rgh": "0 2 -2 0 0 0 0",  # Reduced pressure: m²/s²
            "U": "0 1 -1 0 0 0 0",  # Velocity: m/s
            "k": "0 2 -2 0 0 0 0",  # Turbulent kinetic energy: m²/s²
            "epsilon": "0 2 -3 0 0 0 0",  # Dissipation rate: m²/s³
            "omega": "0 0 -1 0 0 0 0",  # Specific dissipation rate: 1/s
            "nut": "0 2 -1 0 0 0 0",  # Turbulent viscosity: m²/s
            "T": "0 0 0 1 0 0 0"  # Temperature: K
        }
        
        # Handle alpha fields
        if field_name.startswith("alpha."):
            return "0 0 0 0 0 0 0"  # Volume fraction: dimensionless
        
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
            elif field_name.startswith("alpha."):
                return "zeroGradient"
            elif field_name == "p_rgh":
                return "fixedFluxPressure"
            elif field_name in ["k", "epsilon", "omega"]:
                return "kqRWallFunction"
            elif field_name == "nut":
                return "nutkWallFunction"
            else:
                return "zeroGradient"
        
        elif patch_type == "inlet":
            if field_name == "U":
                return "fixedValue"
            elif field_name.startswith("alpha."):
                return "fixedValue"
            elif field_name == "p_rgh":
                return "fixedFluxPressure"
            elif field_name in ["k", "epsilon", "omega"]:
                return "fixedValue"
            elif field_name == "nut":
                return "calculated"
            else:
                return "zeroGradient"
        
        elif patch_type == "outlet":
            if field_name == "U":
                return "zeroGradient"
            elif field_name.startswith("alpha."):
                return "zeroGradient"
            elif field_name == "p_rgh":
                return "fixedValue"
            elif field_name in ["k", "epsilon", "omega"]:
                return "zeroGradient"
            elif field_name == "nut":
                return "calculated"
            else:
                return "zeroGradient"
        
        elif patch_type == "atmosphere":
            if field_name == "U":
                return "pressureInletOutletVelocity"
            elif field_name.startswith("alpha."):
                return "inletOutlet"
            elif field_name == "p_rgh":
                return "totalPressure"
            elif field_name in ["k", "epsilon", "omega"]:
                return "inletOutlet"
            elif field_name == "nut":
                return "calculated"
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
            
            # Create phase properties for multiphase solvers
            if self.solver_type in [self.MULTIPHASE_INTER_FOAM, self.REACTIVE_INTER_FOAM]:
                with open(constant_dir / "phaseProperties", "w") as f:
                    f.write(self.generate_phase_field_dict())
            
            # Add g (gravity) file
            with open(constant_dir / "g", "w") as f:
                f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
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
    class       uniformDimensionedVectorField;
    object      g;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -2 0 0 0 0];
value           (0 0 -9.81);

// ************************************************************************* //
""")
            
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


def create_oil_water_case(case_dir: Union[str, Path], domain_type: str = "dam",
                          oil_density: float = 850.0, oil_viscosity: float = 0.03,
                          water_density: float = 1000.0, water_viscosity: float = 0.001,
                          surface_tension: float = 0.025) -> MultiPhaseSolver:
    """
    Create an oil-water two-phase case.
    
    Args:
        case_dir (Union[str, Path]): Case directory
        domain_type (str): Type of domain ("dam", "pipe", "tank")
        oil_density (float): Oil density in kg/m³
        oil_viscosity (float): Oil viscosity in Pa·s
        water_density (float): Water density in kg/m³
        water_viscosity (float): Water viscosity in Pa·s
        surface_tension (float): Surface tension in N/m
        
    Returns:
        MultiPhaseSolver: Configured solver
    """
    # Create solver
    solver = MultiPhaseSolver(case_dir, MultiPhaseSolver.INTER_FOAM)
    
    # Set up phases
    solver.phases = []  # Clear default phases
    solver.add_phase("water", water_density, water_viscosity, 0.0)
    solver.add_phase("oil", oil_density, oil_viscosity, 1.0)
    
    # Set surface tension
    solver.set_surface_tension("water", "oil", surface_tension)
    
    # Set turbulence model
    solver.set_turbulence_model(MultiPhaseSolver.K_EPSILON)
    
    # Set solver settings
    solver.set_solver_settings(
        end_time=5.0,
        delta_t=0.001,
        write_interval=0.05,
        max_co=0.5,
        max_alpha_co=0.25,
        interface_compression=1.0
    )
    
    # Set up boundary conditions based on domain type
    if domain_type == "dam":
        # Dam break case
        solver.add_boundary_condition(
            "left",
            "wall",
            "wall",
            **{
                "U": {"type": "noSlip"},
                "p_rgh": {"type": "fixedFluxPressure"},
                "alpha.water": {"type": "zeroGradient"},
                "alpha.oil": {"type": "zeroGradient"},
                "k": {"type": "kqRWallFunction", "value": 0.01},
                "epsilon": {"type": "epsilonWallFunction", "value": 0.01}
            }
        )
        
        solver.add_boundary_condition(
            "right",
            "wall",
            "wall",
            **{
                "U": {"type": "noSlip"},
                "p_rgh": {"type": "fixedFluxPressure"},
                "alpha.water": {"type": "zeroGradient"},
                "alpha.oil": {"type": "zeroGradient"},
                "k": {"type": "kqRWallFunction", "value": 0.01},
                "epsilon": {"type": "epsilonWallFunction", "value": 0.01}
            }
        )
        
        solver.add_boundary_condition(
            "bottom",
            "wall",
            "wall",
            **{
                "U": {"type": "noSlip"},
                "p_rgh": {"type": "fixedFluxPressure"},
                "alpha.water": {"type": "zeroGradient"},
                "alpha.oil": {"type": "zeroGradient"},
                "k": {"type": "kqRWallFunction", "value": 0.01},
                "epsilon": {"type": "epsilonWallFunction", "value": 0.01}
            }
        )
        
        solver.add_boundary_condition(
            "top",
            "patch",
            "atmosphere",
            **{
                "U": {"type": "pressureInletOutletVelocity", "value": [0, 0, 0]},
                "p_rgh": {"type": "totalPressure", "p0": 0},
                "alpha.water": {"type": "inletOutlet", "inletValue": 0, "value": 0},
                "alpha.oil": {"type": "inletOutlet", "inletValue": 0, "value": 0},
                "k": {"type": "inletOutlet", "inletValue": 0.01, "value": 0.01},
                "epsilon": {"type": "inletOutlet", "inletValue": 0.01, "value": 0.01}
            }
        )
        
    elif domain_type == "pipe":
        # Pipe flow case
        solver.add_boundary_condition(
            "inlet",
            "patch",
            "inlet",
            **{
                "U": {"type": "fixedValue", "value": [1, 0, 0]},
                "p_rgh": {"type": "fixedFluxPressure"},
                "alpha.water": {"type": "fixedValue", "value": 0.5},
                "alpha.oil": {"type": "fixedValue", "value": 0.5},
                "k": {"type": "fixedValue", "value": 0.01},
                "epsilon": {"type": "fixedValue", "value": 0.01}
            }
        )
        
        solver.add_boundary_condition(
            "outlet",
            "patch",
            "outlet",
            **{
                "U": {"type": "zeroGradient"},
                "p_rgh": {"type": "fixedValue", "value": 0},
                "alpha.water": {"type": "zeroGradient"},
                "alpha.oil": {"type": "zeroGradient"},
                "k": {"type": "zeroGradient"},
                "epsilon": {"type": "zeroGradient"}
            }
        )
        
        solver.add_boundary_condition(
            "wall",
            "wall",
            "wall",
            **{
                "U": {"type": "noSlip"},
                "p_rgh": {"type": "fixedFluxPressure"},
                "alpha.water": {"type": "zeroGradient"},
                "alpha.oil": {"type": "zeroGradient"},
                "k": {"type": "kqRWallFunction", "value": 0.01},
                "epsilon": {"type": "epsilonWallFunction", "value": 0.01}
            }
        )
    
    elif domain_type == "tank":
        # Tank with water-oil interface
        solver.add_boundary_condition(
            "walls",
            "wall",
            "wall",
            **{
                "U": {"type": "noSlip"},
                "p_rgh": {"type": "fixedFluxPressure"},
                "alpha.water": {"type": "zeroGradient"},
                "alpha.oil": {"type": "zeroGradient"},
                "k": {"type": "kqRWallFunction", "value": 0.01},
                "epsilon": {"type": "epsilonWallFunction", "value": 0.01}
            }
        )
        
        solver.add_boundary_condition(
            "top",
            "patch",
            "atmosphere",
            **{
                "U": {"type": "pressureInletOutletVelocity", "value": [0, 0, 0]},
                "p_rgh": {"type": "totalPressure", "p0": 0},
                "alpha.water": {"type": "inletOutlet", "inletValue": 0, "value": 0},
                "alpha.oil": {"type": "inletOutlet", "inletValue": 0, "value": 0},
                "k": {"type": "inletOutlet", "inletValue": 0.01, "value": 0.01},
                "epsilon": {"type": "inletOutlet", "inletValue": 0.01, "value": 0.01}
            }
        )
        
        solver.add_boundary_condition(
            "inlet",
            "patch",
            "inlet",
            **{
                "U": {"type": "fixedValue", "value": [0, 0, -0.1]},
                "p_rgh": {"type": "fixedFluxPressure"},
                "alpha.water": {"type": "fixedValue", "value": 1.0},
                "alpha.oil": {"type": "fixedValue", "value": 0.0},
                "k": {"type": "fixedValue", "value": 0.01},
                "epsilon": {"type": "fixedValue", "value": 0.01}
            }
        )
        
        solver.add_boundary_condition(
            "outlet",
            "patch",
            "outlet",
            **{
                "U": {"type": "zeroGradient"},
                "p_rgh": {"type": "fixedValue", "value": 0},
                "alpha.water": {"type": "zeroGradient"},
                "alpha.oil": {"type": "zeroGradient"},
                "k": {"type": "zeroGradient"},
                "epsilon": {"type": "zeroGradient"}
            }
        )
    
    # Set initial conditions for dam break
    if domain_type == "dam":
        # These will be overridden by the setFields utility
        solver.set_initial_conditions(
            p_rgh=0,
            U=[0, 0, 0],
            k=0.01,
            epsilon=0.01,
            **{
                "alpha.water": 0.0,
                "alpha.oil": 1.0
            }
        )
        # Create setFieldsDict to initialize the phases
        system_dir = Path(case_dir) / "system"
        system_dir.mkdir(parents=True, exist_ok=True)
        
        with open(system_dir / "setFieldsDict", "w") as f:
            f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
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
    object      setFieldsDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defaultFieldValues
(
    volScalarFieldValue alpha.water 0
    volScalarFieldValue alpha.oil 1
);

regions
(
    boxToCell
    {
        box (0 0 0) (0.1 1 0.5);
        fieldValues
        (
            volScalarFieldValue alpha.water 1
            volScalarFieldValue alpha.oil 0
        );
    }
);

// ************************************************************************* //
""")
    
    return solver


def create_gas_liquid_case(case_dir: Union[str, Path], domain_type: str = "bubble",
                           liquid_density: float = 1000.0, liquid_viscosity: float = 0.001,
                           gas_density: float = 1.0, gas_viscosity: float = 1.8e-5,
                           surface_tension: float = 0.072) -> MultiPhaseSolver:
    """
    Create a gas-liquid two-phase case.
    
    Args:
        case_dir (Union[str, Path]): Case directory
        domain_type (str): Type of domain ("bubble", "pipe", "column")
        liquid_density (float): Liquid density in kg/m³
        liquid_viscosity (float): Liquid viscosity in Pa·s
        gas_density (float): Gas density in kg/m³
        gas_viscosity (float): Gas viscosity in Pa·s
        surface_tension (float): Surface tension in N/m
        
    Returns:
        MultiPhaseSolver: Configured solver
    """
    # Create solver
    solver = MultiPhaseSolver(case_dir, MultiPhaseSolver.INTER_FOAM)
    
    # Set up phases
    solver.phases = []  # Clear default phases
    solver.add_phase("liquid", liquid_density, liquid_viscosity, 1.0)
    solver.add_phase("gas", gas_density, gas_viscosity, 0.0)
    
    # Set surface tension
    solver.set_surface_tension("liquid", "gas", surface_tension)
    
    # Set turbulence model
    solver.set_turbulence_model(MultiPhaseSolver.K_EPSILON)
    
    # Set solver settings
    solver.set_solver_settings(
        end_time=10.0,
        delta_t=0.001,
        write_interval=0.1,
        max_co=0.5,
        max_alpha_co=0.25,
        interface_compression=1.0
    )
    
    # Set up boundary conditions based on domain type
    if domain_type == "bubble":
        # Single bubble rising in liquid
        solver.add_boundary_condition(
            "walls",
            "wall",
            "wall",
            **{
                "U": {"type": "noSlip"},
                "p_rgh": {"type": "fixedFluxPressure"},
                "alpha.liquid": {"type": "zeroGradient"},
                "alpha.gas": {"type": "zeroGradient"},
                "k": {"type": "kqRWallFunction", "value": 0.01},
                "epsilon": {"type": "epsilonWallFunction", "value": 0.01}
            }
        )
        
        # Create setFieldsDict to initialize the bubble
        system_dir = Path(case_dir) / "system"
        system_dir.mkdir(parents=True, exist_ok=True)
        
        with open(system_dir / "setFieldsDict", "w") as f:
            f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
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
    object      setFieldsDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defaultFieldValues
(
    volScalarFieldValue alpha.liquid 1
    volScalarFieldValue alpha.gas 0
);

regions
(
    sphereToCell
    {
        centre (0.05 0.05 0.05);
        radius 0.02;
        fieldValues
        (
            volScalarFieldValue alpha.liquid 0
            volScalarFieldValue alpha.gas 1
        );
    }
);

// ************************************************************************* //
""")
    
    elif domain_type == "pipe":
        # Gas-liquid pipe flow
        solver.add_boundary_condition(
            "inlet",
            "patch",
            "inlet",
            **{
                "U": {"type": "fixedValue", "value": [1, 0, 0]},
                "p_rgh": {"type": "fixedFluxPressure"},
                "alpha.liquid": {"type": "fixedValue", "value": 0.5},
                "alpha.gas": {"type": "fixedValue", "value": 0.5},
                "k": {"type": "fixedValue", "value": 0.01},
                "epsilon": {"type": "fixedValue", "value": 0.01}
            }
        )
        
        solver.add_boundary_condition(
            "outlet",
            "patch",
            "outlet",
            **{
                "U": {"type": "zeroGradient"},
                "p_rgh": {"type": "fixedValue", "value": 0},
                "alpha.liquid": {"type": "zeroGradient"},
                "alpha.gas": {"type": "zeroGradient"},
                "k": {"type": "zeroGradient"},
                "epsilon": {"type": "zeroGradient"}
            }
        )
        
        solver.add_boundary_condition(
            "wall",
            "wall",
            "wall",
            **{
                "U": {"type": "noSlip"},
                "p_rgh": {"type": "fixedFluxPressure"},
                "alpha.liquid": {"type": "zeroGradient"},
                "alpha.gas": {"type": "zeroGradient"},
                "k": {"type": "kqRWallFunction", "value": 0.01},
                "epsilon": {"type": "epsilonWallFunction", "value": 0.01}
            }
        )
    
    elif domain_type == "column":
        # Bubble column
        solver.add_boundary_condition(
            "walls",
            "wall",
            "wall",
            **{
                "U": {"type": "noSlip"},
                "p_rgh": {"type": "fixedFluxPressure"},
                "alpha.liquid": {"type": "zeroGradient"},
                "alpha.gas": {"type": "zeroGradient"},
                "k": {"type": "kqRWallFunction", "value": 0.01},
                "epsilon": {"type": "epsilonWallFunction", "value": 0.01}
            }
        )
        
        solver.add_boundary_condition(
            "inlet",
            "patch",
            "inlet",
            **{
                "U": {"type": "fixedValue", "value": [0, 0, 0.1]},
                "p_rgh": {"type": "fixedFluxPressure"},
                "alpha.liquid": {"type": "fixedValue", "value": 0.0},
                "alpha.gas": {"type": "fixedValue", "value": 1.0},
                "k": {"type": "fixedValue", "value": 0.01},
                "epsilon": {"type": "fixedValue", "value": 0.01}
            }
        )
        
        solver.add_boundary_condition(
            "outlet",
            "patch",
            "atmosphere",
            **{
                "U": {"type": "pressureInletOutletVelocity", "value": [0, 0, 0]},
                "p_rgh": {"type": "totalPressure", "p0": 0},
                "alpha.liquid": {"type": "inletOutlet", "inletValue": 0, "value": 0},
                "alpha.gas": {"type": "inletOutlet", "inletValue": 1, "value": 1},
                "k": {"type": "inletOutlet", "inletValue": 0.01, "value": 0.01},
                "epsilon": {"type": "inletOutlet", "inletValue": 0.01, "value": 0.01}
            }
        )
    
    return solver


def create_three_phase_case(case_dir: Union[str, Path], domain_type: str = "separator",
                           oil_density: float = 850.0, oil_viscosity: float = 0.03,
                           water_density: float = 1000.0, water_viscosity: float = 0.001,
                           gas_density: float = 1.0, gas_viscosity: float = 1.8e-5,
                           sigma_ow: float = 0.025, sigma_og: float = 0.023, 
                           sigma_wg: float = 0.072) -> MultiPhaseSolver:
    """
    Create a three-phase (oil-water-gas) case.
    
    Args:
        case_dir (Union[str, Path]): Case directory
        domain_type (str): Type of domain ("separator", "pipeline", "wellbore")
        oil_density (float): Oil density in kg/m³
        oil_viscosity (float): Oil viscosity in Pa·s
        water_density (float): Water density in kg/m³
        water_viscosity (float): Water viscosity in Pa·s
        gas_density (float): Gas density in kg/m³
        gas_viscosity (float): Gas viscosity in Pa·s
        sigma_ow (float): Oil-water surface tension in N/m
        sigma_og (float): Oil-gas surface tension in N/m
        sigma_wg (float): Water-gas surface tension in N/m
        
    Returns:
        MultiPhaseSolver: Configured solver
    """
    # Create solver
    solver = MultiPhaseSolver(case_dir, MultiPhaseSolver.MULTIPHASE_INTER_FOAM)
    
    # Set up phases
    solver.phases = []  # Clear default phases
    solver.add_phase("water", water_density, water_viscosity, 0.0)
    solver.add_phase("oil", oil_density, oil_viscosity, 0.0)
    solver.add_phase("gas", gas_density, gas_viscosity, 1.0)
    
    # Set surface tension values
    solver.set_surface_tension("oil", "water", sigma_ow)
    solver.set_surface_tension("oil", "gas", sigma_og)
    solver.set_surface_tension("water", "gas", sigma_wg)
    
    # Set turbulence model
    solver.set_turbulence_model(MultiPhaseSolver.K_EPSILON)
    
    # Set solver settings
    solver.set_solver_settings(
        end_time=10.0,
        delta_t=0.001,
        write_interval=0.1,
        max_co=0.5,
        max_alpha_co=0.25,
        interface_compression=1.0
    )
    
    # Set up boundary conditions based on domain type
    if domain_type == "separator":
        # Oil-water-gas gravity separator
        solver.add_boundary_condition(
            "walls",
            "wall",
            "wall",
            **{
                "U": {"type": "noSlip"},
                "p_rgh": {"type": "fixedFluxPressure"},
                "alpha.water": {"type": "zeroGradient"},
                "alpha.oil": {"type": "zeroGradient"},
                "alpha.gas": {"type": "zeroGradient"},
                "k": {"type": "kqRWallFunction", "value": 0.01},
                "epsilon": {"type": "epsilonWallFunction", "value": 0.01}
            }
        )
        
        solver.add_boundary_condition(
            "inlet",
            "patch",
            "inlet",
            **{
                "U": {"type": "fixedValue", "value": [0.5, 0, 0]},
                "p_rgh": {"type": "fixedFluxPressure"},
                "alpha.water": {"type": "fixedValue", "value": 0.3},
                "alpha.oil": {"type": "fixedValue", "value": 0.3},
                "alpha.gas": {"type": "fixedValue", "value": 0.4},
                "k": {"type": "fixedValue", "value": 0.01},
                "epsilon": {"type": "fixedValue", "value": 0.01}
            }
        )
        
        solver.add_boundary_condition(
            "gasOutlet",
            "patch",
            "outlet",
            **{
                "U": {"type": "zeroGradient"},
                "p_rgh": {"type": "fixedValue", "value": 0},
                "alpha.water": {"type": "zeroGradient"},
                "alpha.oil": {"type": "zeroGradient"},
                "alpha.gas": {"type": "zeroGradient"},
                "k": {"type": "zeroGradient"},
                "epsilon": {"type": "zeroGradient"}
            }
        )
        
        solver.add_boundary_condition(
            "oilOutlet",
            "patch",
            "outlet",
            **{
                "U": {"type": "zeroGradient"},
                "p_rgh": {"type": "fixedValue", "value": 0},
                "alpha.water": {"type": "zeroGradient"},
                "alpha.oil": {"type": "zeroGradient"},
                "alpha.gas": {"type": "zeroGradient"},
                "k": {"type": "zeroGradient"},
                "epsilon": {"type": "zeroGradient"}
            }
        )
        
        solver.add_boundary_condition(
            "waterOutlet",
            "patch",
            "outlet",
            **{
                "U": {"type": "zeroGradient"},
                "p_rgh": {"type": "fixedValue", "value": 0},
                "alpha.water": {"type": "zeroGradient"},
                "alpha.oil": {"type": "zeroGradient"},
                "alpha.gas": {"type": "zeroGradient"},
                "k": {"type": "zeroGradient"},
                "epsilon": {"type": "zeroGradient"}
            }
        )
        
    elif domain_type == "pipeline":
        # Three-phase pipeline flow
        solver.add_boundary_condition(
            "inlet",
            "patch",
            "inlet",
            **{
                "U": {"type": "fixedValue", "value": [1, 0, 0]},
                "p_rgh": {"type": "fixedFluxPressure"},
                "alpha.water": {"type": "fixedValue", "value": 0.3},
                "alpha.oil": {"type": "fixedValue", "value": 0.3},
                "alpha.gas": {"type": "fixedValue", "value": 0.4},
                "k": {"type": "fixedValue", "value": 0.01},
                "epsilon": {"type": "fixedValue", "value": 0.01}
            }
        )
        
        solver.add_boundary_condition(
            "outlet",
            "patch",
            "outlet",
            **{
                "U": {"type": "zeroGradient"},
                "p_rgh": {"type": "fixedValue", "value": 0},
                "alpha.water": {"type": "zeroGradient"},
                "alpha.oil": {"type": "zeroGradient"},
                "alpha.gas": {"type": "zeroGradient"},
                "k": {"type": "zeroGradient"},
                "epsilon": {"type": "zeroGradient"}
            }
        )
        
        solver.add_boundary_condition(
            "wall",
            "wall",
            "wall",
            **{
                "U": {"type": "noSlip"},
                "p_rgh": {"type": "fixedFluxPressure"},
                "alpha.water": {"type": "zeroGradient"},
                "alpha.oil": {"type": "zeroGradient"},
                "alpha.gas": {"type": "zeroGradient"},
                "k": {"type": "kqRWallFunction", "value": 0.01},
                "epsilon": {"type": "epsilonWallFunction", "value": 0.01}
            }
        )
    
    elif domain_type == "wellbore":
        # Wellbore with three-phase flow
        solver.add_boundary_condition(
            "walls",
            "wall",
            "wall",
            **{
                "U": {"type": "noSlip"},
                "p_rgh": {"type": "fixedFluxPressure"},
                "alpha.water": {"type": "zeroGradient"},
                "alpha.oil": {"type": "zeroGradient"},
                "alpha.gas": {"type": "zeroGradient"},
                "k": {"type": "kqRWallFunction", "value": 0.01},
                "epsilon": {"type": "epsilonWallFunction", "value": 0.01}
            }
        )
        
        solver.add_boundary_condition(
            "inlet",
            "patch",
            "inlet",
            **{
                "U": {"type": "fixedValue", "value": [0, 0, 2.0]},
                "p_rgh": {"type": "fixedFluxPressure"},
                "alpha.water": {"type": "fixedValue", "value": 0.4},
                "alpha.oil": {"type": "fixedValue", "value": 0.4},
                "alpha.gas": {"type": "fixedValue", "value": 0.2},
                "k": {"type": "fixedValue", "value": 0.01},
                "epsilon": {"type": "fixedValue", "value": 0.01}
            }
        )
        
        solver.add_boundary_condition(
            "outlet",
            "patch",
            "outlet",
            **{
                "U": {"type": "zeroGradient"},
                "p_rgh": {"type": "fixedValue", "value": 0},
                "alpha.water": {"type": "zeroGradient"},
                "alpha.oil": {"type": "zeroGradient"},
                "alpha.gas": {"type": "zeroGradient"},
                "k": {"type": "zeroGradient"},
                "epsilon": {"type": "zeroGradient"}
            }
        )
    
    return solver


def create_from_template(case_dir: Union[str, Path], template_dir: Union[str, Path], 
                        parameters: Dict[str, Any]) -> MultiPhaseSolver:
    """
    Create a case from a template.
    
    Args:
        case_dir (Union[str, Path]): Target case directory
        template_dir (Union[str, Path]): Template directory
        parameters (Dict[str, Any]): Parameters to customize the template
        
    Returns:
        MultiPhaseSolver: Configured solver
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
        
        # Create solver object for the case
        # Detect solver type from constant/transportProperties or system/controlDict
        solver_type = MultiPhaseSolver.INTER_FOAM  # Default
        
        control_dict_path = case_dir / "system" / "controlDict"
        if control_dict_path.exists():
            with open(control_dict_path, "r") as f:
                content = f.read()
                for solver in [MultiPhaseSolver.INTER_FOAM, MultiPhaseSolver.MULTIPHASE_INTER_FOAM,
                              MultiPhaseSolver.COMPRESSIBLE_INTER_FOAM, MultiPhaseSolver.TWO_PHASE_EULER_FOAM]:
                    if f"application     {solver};" in content:
                        solver_type = solver
                        break
        
        solver = MultiPhaseSolver(case_dir, solver_type)
        
        # Extract phase information if available
        transport_props_path = case_dir / "constant" / "transportProperties"
        if transport_props_path.exists():
            # This would require a more complex parser for OpenFOAM dictionary files
            # For now, we just create a basic solver instance
            pass
        
        return solver
        
    except Exception as e:
        logger.error(f"Error creating case from template: {e}")
        return MultiPhaseSolver(case_dir)


# Example usage
if __name__ == "__main__":
    print("OpenFOAM Multi-Phase Solver Module")
    
    # Example: Create an oil-water case
    case_dir = Path("./oil_water_case")
    solver = create_oil_water_case(case_dir, domain_type="dam",
                                  oil_density=850.0, oil_viscosity=0.03,
                                  water_density=1000.0, water_viscosity=0.001,
                                  surface_tension=0.025)
    
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