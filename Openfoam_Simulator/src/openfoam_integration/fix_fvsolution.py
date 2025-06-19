#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to fix fvSolution files for OpenFOAM cases.
Ensures that PISO and SIMPLE sections have proper reference cell settings
and that all required solver entries for pisoFoam are included.
"""

import os
import re
import argparse
import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_fv_solution(case_dir):
    """
    Fix the fvSolution file to ensure it has both SIMPLE and PISO sections
    with proper reference cell settings and all required solver entries.
    
    Args:
        case_dir (str): Path to the OpenFOAM case directory
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get the system directory
        system_dir = os.path.join(case_dir, 'system')
        if not os.path.exists(system_dir):
            logger.info(f"Creating system directory at {system_dir}")
            os.makedirs(system_dir, exist_ok=True)
        
        # Path to fvSolution file
        fv_solution_path = os.path.join(system_dir, 'fvSolution')
        
        # Always create a fresh, correctly formatted file
        logger.info(f"Creating new fvSolution at {fv_solution_path}")
        with open(fv_solution_path, 'w') as f:
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
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-6;
        relTol          0.1;
        smoother        GaussSeidel;
    }

    pFinal
    {
        $p;
        relTol          0.01;
    }

    "(U|k|epsilon|omega|R)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-6;
        relTol          0.1;
    }

    "(U|k|epsilon|omega|R)Final"
    {
        $U;
        relTol          0.01;
    }
}

// PISO section for pisoFoam - MUST include pRefCell and pRefValue
PISO
{
    nCorrectors     2;
    nNonOrthogonalCorrectors 1;
    pRefCell        0;
    pRefValue       0;
}

// SIMPLE section for simpleFoam - MUST include pRefCell and pRefValue 
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
    pRefCell        0;
    pRefValue       0;
    
    residualControl
    {
        p               1e-4;
        U               1e-4;
        "(k|epsilon|omega|R)" 1e-3;
    }
}

relaxationFactors
{
    equations
    {
        ".*"            0.9;
    }
    fields
    {
        "p.*"           0.3;
    }
}

// ************************************************************************* //
""")
        
        logger.info(f"Successfully created fvSolution at {fv_solution_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing fvSolution: {e}")
        return False


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description='Fix fvSolution file for OpenFOAM cases')
    parser.add_argument('case_dir', help='Path to the OpenFOAM case directory')
    args = parser.parse_args()
    
    if fix_fv_solution(args.case_dir):
        logger.info(f"Successfully fixed fvSolution for case: {args.case_dir}")
        return 0
    else:
        logger.error(f"Failed to fix fvSolution for case: {args.case_dir}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
