#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline pigging simulation solver for Openfoam_Simulator.

This module provides tools for setting up and running OpenFOAM simulations
of pipeline pigging operations, which are commonly used in the oil & gas
industry for cleaning, inspection, and maintenance of pipelines.

The module handles the specialized physics of pig-fluid interactions, including:
- Multiphase flow around the pig
- Pig movement dynamics
- Bypass flow modeling
- Friction and contact modeling
- Debris accumulation (for cleaning pigs)
"""

import os
import sys
import math
import logging
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

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
from ...industry.pigging_models import PigType, PigModel, PigParameters
from ...industry.pipeline_components import PipelineParameters

logger = get_logger(__name__)


class PiggingSimulation:
    """
    Class for managing and executing pipeline pigging simulations in OpenFOAM.
    
    This class handles the complex multiphase flow simulation with a moving pig
    interface, typically using the interFoam solver with dynamic mesh capabilities
    and custom boundary conditions.
    """
    
    def __init__(self, case_dir: str, pig_parameters: Dict[str, Any], 
                pipeline_parameters: Dict[str, Any], operation_parameters: Dict[str, Any]):
        """
        Initialize the pigging simulation.
        
        Args:
            case_dir (str): Path to the OpenFOAM case directory
            pig_parameters (Dict[str, Any]): Parameters defining the pig properties
            pipeline_parameters (Dict[str, Any]): Parameters defining the pipeline
            operation_parameters (Dict[str, Any]): Parameters defining the operation
        """
        self.case_dir = Path(case_dir)
        self.pig_params = pig_parameters
        self.pipeline_params = pipeline_parameters
        self.operation_params = operation_parameters
        
        # Initialize managers
        self.case_manager = CaseManager(case_dir)
        self.solver_manager = SolverManager(case_dir)
        self.bc_manager = BoundaryConditionsManager(case_dir)
        self.transport_manager = TransportModelsManager(case_dir)
        
        # Initialize specialized pig model based on type
        self.pig_model = self._create_pig_model()
        
        # Set defaults
        self.solver_name = "interFoam"  # Default solver for pigging sims
        self.mesh_generated = False
        self.case_setup = False
    
    def _create_pig_model(self) -> PigModel:
        """
        Create the appropriate pig model based on type.
        
        Returns:
            PigModel: The pig model instance
        """
        # Convert dict to proper PigParameters object
        pig_type_str = self.pig_params.get("type", "Foam")
        
        # Convert string type to enum
        pig_type = getattr(PigType, pig_type_str.upper(), PigType.FOAM)
        
        # Create parameters object
        params = PigParameters(
            pig_type=pig_type,
            diameter=self.pig_params.get("diameter", 0.1),
            length=self.pig_params.get("length", 0.2),
            density=self.pig_params.get("density", 300.0),
            friction_coefficient=self.pig_params.get("friction", 0.3),
            bypass_ratio=self.pig_params.get("bypass", 0.05)
        )
        
        # Create and return the appropriate model
        return PigModel(params)
    
    def generate_mesh(self) -> bool:
        """
        Generate the mesh for the pigging simulation.
        
        This creates a specialized mesh with refinement in the pig region
        and appropriate boundary conditions.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create mesh generator
            mesh_generator = MeshGenerator(self.case_dir)
            
            # Set up pipeline geometry
            diameter = self.pipeline_params.get("diameter", 0.1016)  # 4 inches default
            length = self.pipeline_params.get("length", 1000.0)
            
            # We'll create a simplified 2D axisymmetric or full 3D mesh depending on needs
            is_3d = True  # Default to 3D for accuracy
            
            # Check if we can use 2D axisymmetric (faster simulation)
            if (self.pig_model.parameters.pig_type != PigType.SPHERE and
                'geometry_simplification' in self.operation_params and
                self.operation_params['geometry_simplification'] == '2D_axisymmetric'):
                is_3d = False
            
            # Initial position of the pig
            initial_position = self.operation_params.get("initial_position", 0.0)
            pig_location = initial_position * length
            
            # Set up base mesh parameters
            if is_3d:
                mesh_generator.setup_pipeline_mesh_3d(
                    diameter=diameter,
                    length=length,
                    cells_radial=20,  # Adjust mesh density as needed
                    cells_axial=int(length / (diameter/2)),  # Maintain reasonable aspect ratio
                    cells_circumferential=36,  # 10-degree segments
                    pig_location=pig_location,
                    pig_diameter=self.pig_params.get("diameter", diameter * 1.05),
                    pig_length=self.pig_params.get("length", diameter * 2.0)
                )
            else:
                # 2D axisymmetric
                mesh_generator.setup_pipeline_mesh_2d_axisymmetric(
                    diameter=diameter,
                    length=length,
                    cells_radial=20,
                    cells_axial=int(length / (diameter/5)),
                    pig_location=pig_location,
                    pig_diameter=self.pig_params.get("diameter", diameter * 1.05),
                    pig_length=self.pig_params.get("length", diameter * 2.0)
                )
            
            # Generate the mesh
            success = mesh_generator.generate()
            if success:
                logger.info(f"Mesh generation successful for pigging simulation in {self.case_dir}")
                self.mesh_generated = True
                return True
            else:
                logger.error(f"Mesh generation failed for pigging simulation in {self.case_dir}")
                return False
            
        except Exception as e:
            logger.error(f"Error generating mesh for pigging simulation: {e}")
            return False
    
    def setup_case(self) -> bool:
        """
        Set up the OpenFOAM case for pigging simulation.
        
        This configures all necessary files for the simulation, including:
        - Control parameters (controlDict)
        - Numerical schemes (fvSchemes)
        - Solution settings (fvSolution)
        - Transport properties
        - Boundary conditions
        - Dynamic mesh settings
        - Turbulence model settings
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
            
            # Set up dynamic mesh dictionary
            self._setup_dynamic_mesh()
            
            # Set up boundary conditions
            self._setup_boundary_conditions()
            
            # Set up initial fields
            self._setup_initial_fields()
            
            # Set up decomposition dictionary for parallel runs
            self._setup_decomposition()
            
            logger.info(f"Case setup successful for pigging simulation in {self.case_dir}")
            self.case_setup = True
            return True
            
        except Exception as e:
            logger.error(f"Error setting up case for pigging simulation: {e}")
            return False
    
    def _setup_control_dict(self):
        """Set up the controlDict file for time control and I/O settings."""
        # Set up control dictionary with appropriate settings for pigging
        end_time = self._estimate_simulation_time()
        
        control_dict = {
            "application": self.solver_name,
            "startFrom": "startTime",
            "startTime": 0,
            "stopAt": "endTime",
            "endTime": end_time,
            "deltaT": 0.001,  # Start with small time step
            "writeControl": "adjustableRunTime",
            "writeInterval": end_time / 100,  # 100 output intervals
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
            "libs": ["libdynamicMeshDict.so", "libwaves.so"]  # Required libraries
        }
        
        self.case_manager.write_control_dict(control_dict)
    
    def _estimate_simulation_time(self) -> float:
        """
        Estimate the total simulation time needed.
        
        Returns:
            float: Estimated simulation time in seconds
        """
        # Get pipeline length and flow rate
        length = self.pipeline_params.get("length", 1000.0)
        diameter = self.pipeline_params.get("diameter", 0.1016)
        flow_rate = self.operation_params.get("flow_rate", 0.5)  # m³/s
        
        # Calculate cross-sectional area and average velocity
        area = math.pi * (diameter/2)**2
        velocity = flow_rate / area
        
        # Estimate time to travel the pipeline
        travel_time = length / velocity
        
        # Add safety factor to ensure simulation captures entire journey
        safety_factor = 1.5
        return travel_time * safety_factor
    
    def _setup_schemes(self):
        """Set up the numerical schemes for the simulation."""
        # Set up schemes dictionary with appropriate settings for pigging
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
        
        self.case_manager.write_fv_schemes(schemes)
    
    def _setup_solution(self):
        """Set up the solution control settings."""
        # Set up solution dictionary with appropriate settings for pigging
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
        
        self.case_manager.write_fv_solution(solution)
    
    def _setup_transport_properties(self):
        """Set up the transport properties for the simulation."""
        # Determine fluids based on driving fluid
        driving_fluid = self.operation_params.get("driving_fluid", "Water")
        
        # Set up properties based on driving fluid
        if driving_fluid == "Water":
            fluid1_properties = {
                "transportModel": "Newtonian",
                "nu": 1e-6,
                "rho": 1000
            }
            fluid2_properties = {
                "transportModel": "Newtonian",
                "nu": 1.8e-5,
                "rho": 1.2
            }
            fluid1_name = "water"
            fluid2_name = "air"
            surface_tension = 0.072
            
        elif driving_fluid == "Oil":
            fluid1_properties = {
                "transportModel": "Newtonian",
                "nu": 3.5e-5,
                "rho": 850
            }
            fluid2_properties = {
                "transportModel": "Newtonian",
                "nu": 1.8e-5,
                "rho": 1.2
            }
            fluid1_name = "oil"
            fluid2_name = "air"
            surface_tension = 0.035
            
        elif driving_fluid == "Natural Gas" or driving_fluid == "Air":
            fluid1_properties = {
                "transportModel": "Newtonian",
                "nu": 1.8e-5,
                "rho": 1.2
            }
            fluid2_properties = {
                "transportModel": "Newtonian",
                "nu": 1e-6,
                "rho": 1000
            }
            fluid1_name = "gas"
            fluid2_name = "water"
            surface_tension = 0.072
        
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
    {fluid1_name}
    {{
        transportModel  {fluid1_properties['transportModel']};
        nu              {fluid1_properties['nu']};
        rho             {fluid1_properties['rho']};
    }}

    {fluid2_name}
    {{
        transportModel  {fluid2_properties['transportModel']};
        nu              {fluid2_properties['nu']};
        rho             {fluid2_properties['rho']};
    }}
);

sigma           {surface_tension};

// ************************************************************************* //
"""
        
        # Write to file
        transport_file = self.case_dir / "constant" / "transportProperties"
        with open(transport_file, 'w') as f:
            f.write(content)
    
    def _setup_dynamic_mesh(self):
        """Set up the dynamic mesh dictionary for pig movement."""
        # Create pig movement specific dynamics
        motion_type = "solidBodyMotion" if self.operation_params.get("pig_movement_model", "prescribed") == "prescribed" else "sixDoFRigidBodyMotion"
        
        if motion_type == "solidBodyMotion":
            # Prescribed motion based on calculated velocity
            diameter = self.pipeline_params.get("diameter", 0.1016)
            area = math.pi * (diameter/2)**2
            flow_rate = self.operation_params.get("flow_rate", 0.5)  # m³/s
            velocity = flow_rate / area
            
            # Account for bypass flow
            bypass_ratio = self.pig_params.get("bypass", 0.05)
            pig_velocity = velocity * (1 - bypass_ratio)
            
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
    object      dynamicMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dynamicFvMesh   dynamicMotionSolverFvMesh;

motionSolverLibs (
    "libfvMotionSolvers.so"
    "librigidBodyMotion.so"
);

solver          displacementLaplacian;

diffusivity     inverseDistance (pig);

// Pig motion
frozenDiffusion  false;

displacementLaplacianCoeffs
{{
    // Diffusivity field
    diffusivity     inverseDistance (pig);
}}

// Motion control for pig
bodies
{{
    pig
    {{
        type            rigidBody;
        parent          root;
        centreOfMass    (0 0 0);
        mass            {self.pig_params.get('density', 300.0) * math.pi * (self.pig_params.get('diameter', 0.1)/2)**2 * self.pig_params.get('length', 0.2)};
        inertia         (1 0 0 0 1 0 0 0 1);
        transform       (1 0 0 0 1 0 0 0 1) (0 0 0);
        joint
        {{
            type            composite;
            joints
            (
                {{
                    type        floating;
                }}
            );
        }}
        patches         (pig);
        innerDistance   0.05;
        outerDistance   0.15;
    }}
}}

// Pig motion
motion
{{
    type            solidBodyMotion;
    solidBodyMotionFunction linearMotion;
    linearMotionCoeffs
    {{
        velocity        ({pig_velocity} 0 0);
    }}
}}

// ************************************************************************* //
"""
        else:
            # 6DoF motion based on forces
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
    object      dynamicMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dynamicFvMesh   dynamicMotionSolverFvMesh;

motionSolverLibs (
    "libfvMotionSolvers.so"
    "librigidBodyMeshMotion.so"
);

solver          rigidBodyMotion;

rigidBodyMotionCoeffs
{{
    report          on;
    solver
    {{
        type Newmark;
    }}

    bodies
    {{
        pig
        {{
            type            rigidBody;
            parent          root;
            centreOfMass    (0 0 0);
            mass            {self.pig_params.get('density', 300.0) * math.pi * (self.pig_params.get('diameter', 0.1)/2)**2 * self.pig_params.get('length', 0.2)};
            inertia         (1 0 0 0 1 0 0 0 1);
            transform       (1 0 0 0 1 0 0 0 1) (0 0 0);
            joint
            {{
                type            composite;
                joints
                (
                    {{
                        type        floating;
                    }}
                );
            }}
            patches         (pig);
            innerDistance   0.05;
            outerDistance   0.15;
            restraints
            {{
                radialMotion
                {{
                    type    linearDamper;
                    axis    (0 1 0); // Constrain to Y axis
                    coeff   1000; // Damping coefficient
                }}
                verticalMotion
                {{
                    type    linearDamper;
                    axis    (0 0 1); // Constrain to Z axis
                    coeff   1000; // Damping coefficient
                }}
            }}
        }}
    }}
}}

// ************************************************************************* //
"""
        
        # Write to file
        dynamic_mesh_file = self.case_dir / "constant" / "dynamicMeshDict"
        with open(dynamic_mesh_file, 'w') as f:
            f.write(content)
    
    def _setup_boundary_conditions(self):
        """Set up the boundary conditions for the simulation."""
        # Set up the boundary conditions based on operation parameters
        # Typical boundaries: inlet, outlet, walls, pig (moving boundary)
        
        # Get flow parameters
        diameter = self.pipeline_params.get("diameter", 0.1016)
        area = math.pi * (diameter/2)**2
        flow_rate = self.operation_params.get("flow_rate", 0.5)  # m³/s
        velocity = flow_rate / area
        driving_fluid = self.operation_params.get("driving_fluid", "Water")
        
        # Velocity boundary conditions
        velocity_bcs = {
            "inlet": {
                "type": "fixedValue",
                "value": f"uniform ({velocity} 0 0)"
            },
            "outlet": {
                "type": "zeroGradient"
            },
            "walls": {
                "type": "noSlip"
            },
            "pig": {
                "type": "movingWallVelocity",
                "value": "uniform (0 0 0)"
            }
        }
        
        # Pressure boundary conditions
        pressure_bcs = {
            "inlet": {
                "type": "zeroGradient"
            },
            "outlet": {
                "type": "fixedValue",
                "value": "uniform 0"
            },
            "walls": {
                "type": "zeroGradient"
            },
            "pig": {
                "type": "zeroGradient"
            }
        }
        
        # Phase fraction boundary conditions
        if driving_fluid == "Water" or driving_fluid == "Oil":
            # Liquid-driven
            alpha_bcs = {
                "inlet": {
                    "type": "fixedValue",
                    "value": "uniform 1"  # Driving fluid
                },
                "outlet": {
                    "type": "zeroGradient"
                },
                "walls": {
                    "type": "zeroGradient"
                },
                "pig": {
                    "type": "zeroGradient"
                }
            }
        else:
            # Gas-driven
            alpha_bcs = {
                "inlet": {
                    "type": "fixedValue",
                    "value": "uniform 0"  # Gas = alpha 0
                },
                "outlet": {
                    "type": "zeroGradient"
                },
                "walls": {
                    "type": "zeroGradient"
                },
                "pig": {
                    "type": "zeroGradient"
                }
            }
        
        # Set up boundary conditions via manager
        self.bc_manager.set_boundary_conditions("U", velocity_bcs)
        self.bc_manager.set_boundary_conditions("p_rgh", pressure_bcs)
        self.bc_manager.set_boundary_conditions("alpha.water", alpha_bcs)
        
        # Turbulence boundary conditions will be set if using turbulence model
        if self.operation_params.get("turbulence_model", "kEpsilon") != "laminar":
            # Turbulence intensity and length scale approach
            intensity = 0.05  # 5%
            length_scale = 0.07 * diameter  # Rule of thumb for pipes
            
            # Calculated derived values
            k = 1.5 * (velocity * intensity)**2
            epsilon = 0.09**0.75 * k**1.5 / length_scale
            
            # k boundary conditions
            k_bcs = {
                "inlet": {
                    "type": "fixedValue",
                    "value": f"uniform {k}"
                },
                "outlet": {
                    "type": "zeroGradient"
                },
                "walls": {
                    "type": "kqRWallFunction",
                    "value": "uniform 0"
                },
                "pig": {
                    "type": "kqRWallFunction",
                    "value": "uniform 0"
                }
            }
            
            # epsilon boundary conditions
            epsilon_bcs = {
                "inlet": {
                    "type": "fixedValue",
                    "value": f"uniform {epsilon}"
                },
                "outlet": {
                    "type": "zeroGradient"
                },
                "walls": {
                    "type": "epsilonWallFunction",
                    "value": f"uniform {epsilon}"
                },
                "pig": {
                    "type": "epsilonWallFunction",
                    "value": f"uniform {epsilon}"
                }
            }
            
            # Set up turbulence boundary conditions
            self.bc_manager.set_boundary_conditions("k", k_bcs)
            self.bc_manager.set_boundary_conditions("epsilon", epsilon_bcs)
    
    def _setup_initial_fields(self):
        """Set up the initial fields for the simulation."""
        # Set up initial fields based on operation parameters
        # Typical fields: U, p_rgh, alpha.water, turbulence (if needed)
        
        # Get flow parameters
        diameter = self.pipeline_params.get("diameter", 0.1016)
        area = math.pi * (diameter/2)**2
        flow_rate = self.operation_params.get("flow_rate", 0.5)  # m³/s
        velocity = flow_rate / area
        driving_fluid = self.operation_params.get("driving_fluid", "Water")
        initial_position = self.operation_params.get("initial_position", 0.0)
        
        # Create 0 directory if needed
        os.makedirs(self.case_dir / "0", exist_ok=True)
        
        # Initial velocity field
        self.bc_manager.initialize_field(
            "U", 
            f"uniform ({velocity} 0 0)",  # Initial velocity
            self.case_manager.get_boundary_patches()
        )
        
        # Initial pressure field
        self.bc_manager.initialize_field(
            "p_rgh", 
            "uniform 0",  # Initial pressure
            self.case_manager.get_boundary_patches()
        )
        
        # Initial phase fraction field based on driving fluid
        if driving_fluid == "Water" or driving_fluid == "Oil":
            # Liquid-driven
            # Initial configuration:
            # - Behind pig: liquid (alpha=1)
            # - In front of pig: air/gas (alpha=0)
            
            # We'll need to set this up with a conditional field setup
            # For now, initialize with a placeholder that should be replaced
            # with a more sophisticated setup in a full implementation
            alpha_water_init = "uniform 0.5"  # Placeholder
            
            # Proper implementation would use setFieldsDict to set up regions
            # based on pig position
        else:
            # Gas-driven
            # Initial configuration:
            # - Behind pig: gas (alpha=0)
            # - In front of pig: liquid (alpha=1)
            
            # Same placeholder approach
            alpha_water_init = "uniform 0.5"  # Placeholder
        
        self.bc_manager.initialize_field(
            "alpha.water", 
            alpha_water_init,
            self.case_manager.get_boundary_patches()
        )
        
        # Initialize turbulence fields if needed
        if self.operation_params.get("turbulence_model", "kEpsilon") != "laminar":
            # Turbulence intensity and length scale approach
            intensity = 0.05  # 5%
            length_scale = 0.07 * diameter  # Rule of thumb for pipes
            
            # Calculate derived values
            k = 1.5 * (velocity * intensity)**2
            epsilon = 0.09**0.75 * k**1.5 / length_scale
            
            # Initialize k and epsilon fields
            self.bc_manager.initialize_field(
                "k", 
                f"uniform {k}",
                self.case_manager.get_boundary_patches()
            )
            
            self.bc_manager.initialize_field(
                "epsilon", 
                f"uniform {epsilon}",
                self.case_manager.get_boundary_patches()
            )
        
        # Set up setFieldsDict for proper phase initialization
        self._create_set_fields_dict()
    
    def _create_set_fields_dict(self):
        """Create the setFieldsDict file for initializing fields."""
        # This dictionary is used to set up regions of fluids
        # based on the pig position
        
        # Get parameters
        diameter = self.pipeline_params.get("diameter", 0.1016)
        length = self.pipeline_params.get("length", 1000.0)
        initial_position = self.operation_params.get("initial_position", 0.0)
        pig_location = initial_position * length
        pig_length = self.pig_params.get("length", diameter * 2.0)
        driving_fluid = self.operation_params.get("driving_fluid", "Water")
        
        # Determine alpha values based on driving fluid
        if driving_fluid == "Water" or driving_fluid == "Oil":
            # Liquid-driven
            behind_pig_alpha = 1.0
            ahead_pig_alpha = 0.0
        else:
            # Gas-driven
            behind_pig_alpha = 0.0
            ahead_pig_alpha = 1.0
        
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
    object      setFieldsDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defaultFieldValues
(
    volScalarFieldValue alpha.water {ahead_pig_alpha}
);

regions
(
    // Region behind the pig
    boxToCell
    {{
        box (0 -1 -1) ({pig_location} 1 1);
        fieldValues
        (
            volScalarFieldValue alpha.water {behind_pig_alpha}
        );
    }}
);

// ************************************************************************* //
"""
        
        # Write to file
        set_fields_file = self.case_dir / "system" / "setFieldsDict"
        with open(set_fields_file, 'w') as f:
            f.write(content)
    
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

// ************************************************************************* //
"""
        
        # Write to file
        decomp_file = self.case_dir / "system" / "decomposeParDict"
        with open(decomp_file, 'w') as f:
            f.write(content)
    
    def run_simulation(self, parallel: bool = True) -> bool:
        """
        Run the pigging simulation.
        
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
        
        # Initialize fields using setFields
        self.solver_manager.run_utility("setFields", parallel=False)
        
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
            # Create OpenFOAM function object setup for post-processing
            post_process_file = self.case_dir / "system" / "controlDict.functions"
            with open(post_process_file, 'w') as f:
                f.write("""
// Field averaging
fieldAverage
{
    type            fieldAverage;
    libs            (fieldFunctionObjects);
    writeControl    writeTime;
    restartOnRestart    false;
    restartOnOutput     false;
    
    fields
    (
        U
        {
            mean        on;
            prime2Mean  on;
            base        time;
        }
        
        p_rgh
        {
            mean        on;
            prime2Mean  on;
            base        time;
        }
        
        alpha.water
        {
            mean        on;
            prime2Mean  on;
            base        time;
        }
    );
}

// Forces and coefficients
forces
{
    type            forces;
    libs            (forces);
    writeControl    writeTime;
    
    patches         (pig);
    
    CofR            (0 0 0);
    
    log             true;
}

// Pig position tracker
pigPositionTracker
{
    type            writeCellCentres;
    libs            (fieldFunctionObjects);
    writeControl    writeTime;
    
    select          cellSet;
    cellSet         pig;
}
                """)
            
            # Append to controlDict
            with open(self.case_dir / "system" / "controlDict", 'a') as f:
                f.write(f"""
// Include function objects
#include "controlDict.functions"
                """)
            
            # Run paraFoam to generate visualization files
            self.solver_manager.run_utility("paraFoam", ["-touch"], parallel=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
            return False


def create_pigging_simulation(case_dir: str, pig_parameters: Dict[str, Any], 
                             pipeline_parameters: Dict[str, Any], 
                             operation_parameters: Dict[str, Any]) -> PiggingSimulation:
    """
    Create a pigging simulation case.
    
    Args:
        case_dir (str): Path to the OpenFOAM case directory
        pig_parameters (Dict[str, Any]): Parameters defining the pig properties
        pipeline_parameters (Dict[str, Any]): Parameters defining the pipeline
        operation_parameters (Dict[str, Any]): Parameters defining the operation
        
    Returns:
        PiggingSimulation: The created simulation object
    """
    # Create the simulation object
    sim = PiggingSimulation(case_dir, pig_parameters, pipeline_parameters, operation_parameters)
    
    # Return the simulation object
    return sim


def run_from_template(case_dir: str, template_name: str, parameters: Dict[str, Any]) -> bool:
    """
    Set up and run a pigging simulation from a template.
    
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
        template_path = Path(templates_dir) / "case_templates" / "pigging" / template_name
        
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
        
        # Extract parameters into the format needed by PiggingSimulation
        pig_params = parameters.get('pig', {})
        pipeline_params = parameters.get('pipeline', {})
        operation_params = parameters.get('operation', {})
        
        # Create simulation object
        sim = PiggingSimulation(case_dir, pig_params, pipeline_params, operation_params)
        
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
    Analyze the results of a pigging simulation.
    
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
        
        # Pig position over time
        pig_position = calculator.extract_moving_object_position("pig")
        results["pig_position"] = pig_position
        
        # Velocity field statistics
        velocity_stats = calculator.compute_field_statistics("U")
        results["velocity"] = velocity_stats
        
        # Pressure field statistics
        pressure_stats = calculator.compute_field_statistics("p_rgh")
        results["pressure"] = pressure_stats
        
        # Phase fraction statistics
        alpha_stats = calculator.compute_field_statistics("alpha.water")
        results["phase_fraction"] = alpha_stats
        
        # Forces on the pig
        forces = calculator.extract_forces("pig")
        results["forces"] = forces
        
        # Drag coefficient
        drag_coefficient = calculator.compute_drag_coefficient("pig")
        results["drag_coefficient"] = drag_coefficient
        
        # Pressure drop across the pig
        pressure_drop = calculator.compute_pressure_drop("inlet", "outlet")
        results["pressure_drop"] = pressure_drop
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing results: {e}")
        return {"error": str(e)}