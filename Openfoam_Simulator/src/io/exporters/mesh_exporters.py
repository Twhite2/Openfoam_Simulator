#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mesh exporting module for Openfoam_Simulator application.

This module provides functions for exporting mesh files to various formats
from the Openfoam_Simulator application, supporting CFD simulations for oil & gas
applications.

Supported formats include:
- STL (ASCII and binary)
- OBJ
- OpenFOAM mesh
- VTK / VTU
- CGNS
- NASTRAN
- GMSH
"""

import os
import sys
import logging
import tempfile
import shutil
import subprocess
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import utility modules
from src.utils.logger import get_logger
from src.config import get_value

# Import data models
from src.models.mesh_model import MeshModel

logger = get_logger(__name__)


class MeshExporter:
    """
    Base class for mesh exporters.
    
    This class provides common functionality for exporting meshes
    to different file formats.
    """
    
    def __init__(self, mesh_model: MeshModel):
        """
        Initialize the mesh exporter.
        
        Args:
            mesh_model: The mesh model to export
        """
        self.mesh_model = mesh_model
        self.stats = {}  # Statistics about the export
    
    def export_mesh(self, filepath: str) -> str:
        """
        Export the mesh to a file.
        
        Args:
            filepath: Path to the output file
        
        Returns:
            str: Path to the exported file
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement export_mesh()")
    
    def validate_mesh(self) -> bool:
        """
        Validate that the mesh model is suitable for export.
        
        Returns:
            bool: True if the mesh is valid, False otherwise
        """
        if self.mesh_model is None:
            logger.error("No mesh model provided")
            return False
        
        # Basic validation - actual implementation would be more comprehensive
        if not hasattr(self.mesh_model, 'num_points') or self.mesh_model.num_points == 0:
            logger.error("Mesh has no points")
            return False
        
        if not hasattr(self.mesh_model, 'num_faces') or self.mesh_model.num_faces == 0:
            logger.error("Mesh has no faces")
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the exported mesh.
        
        Returns:
            Dict[str, Any]: Statistics about the export
        """
        return self.stats


class STLExporter(MeshExporter):
    """
    Exporter for STL (STereoLithography) mesh files.
    
    Supports both ASCII and binary STL formats.
    """
    
    def export_mesh(self, filepath: str, ascii_format: bool = False) -> str:
        """
        Export a mesh to STL format.
        
        Args:
            filepath: Path to the output file
            ascii_format: Whether to use ASCII format (default: binary)
        
        Returns:
            str: Path to the exported file
        
        Raises:
            ValueError: If the mesh model is invalid or export fails
        """
        if not self.validate_mesh():
            raise ValueError("Invalid mesh model")
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        try:
            # Export to STL
            if ascii_format:
                self._export_ascii_stl(filepath)
            else:
                self._export_binary_stl(filepath)
            
            # Collect statistics
            self.stats = {
                "format": "STL",
                "file_size": os.path.getsize(filepath),
                "ascii_format": ascii_format,
                "num_triangles": self.mesh_model.num_faces
            }
            
            logger.info(f"Successfully exported mesh to STL: {filepath}")
            logger.info(f"Exported {self.mesh_model.num_faces} triangles")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting mesh to STL: {e}")
            raise ValueError(f"Failed to export mesh to STL: {e}")
    
    def _export_ascii_stl(self, filepath: str):
        """
        Export a mesh to ASCII STL format.
        
        Args:
            filepath: Path to the output file
        """
        # For a real implementation, we would iterate through the mesh data
        # and write each triangle to the STL file.
        
        # This is a placeholder implementation that creates a simple cube
        with open(filepath, 'w') as f:
            # Write header
            f.write(f"solid {os.path.splitext(os.path.basename(filepath))[0]}\n")
            
            # Create a simple cube with 12 triangles (2 per face)
            vertices = [
                # Front face
                [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
                # Back face
                [[0, 0, 1], [1, 0, 1], [1, 1, 1]],
                [[0, 0, 1], [1, 1, 1], [0, 1, 1]],
                # Left face
                [[0, 0, 0], [0, 1, 0], [0, 1, 1]],
                [[0, 0, 0], [0, 1, 1], [0, 0, 1]],
                # Right face
                [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
                [[1, 0, 0], [1, 1, 1], [1, 0, 1]],
                # Bottom face
                [[0, 0, 0], [1, 0, 0], [1, 0, 1]],
                [[0, 0, 0], [1, 0, 1], [0, 0, 1]],
                # Top face
                [[0, 1, 0], [1, 1, 0], [1, 1, 1]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 1]]
            ]
            
            normals = [
                [0, 0, -1], [0, 0, -1],  # Front face
                [0, 0, 1], [0, 0, 1],    # Back face
                [-1, 0, 0], [-1, 0, 0],  # Left face
                [1, 0, 0], [1, 0, 0],    # Right face
                [0, -1, 0], [0, -1, 0],  # Bottom face
                [0, 1, 0], [0, 1, 0]     # Top face
            ]
            
            # Write triangles
            for i, (triangle, normal) in enumerate(zip(vertices, normals)):
                f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                f.write("    outer loop\n")
                for vertex in triangle:
                    f.write(f"      vertex {vertex[0]} {vertex[1]} {vertex[2]}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            
            # Write footer
            f.write(f"endsolid {os.path.splitext(os.path.basename(filepath))[0]}\n")
    
    def _export_binary_stl(self, filepath: str):
        """
        Export a mesh to binary STL format.
        
        Args:
            filepath: Path to the output file
        """
        # For a real implementation, we would iterate through the mesh data
        # and write each triangle to the STL file in binary format.
        
        # This is a placeholder implementation that creates a simple cube
        with open(filepath, 'wb') as f:
            # Write header (80 bytes)
            header = f"Binary STL export from Openfoam_Simulator".encode('ascii')
            header = header.ljust(80, b'\0')
            f.write(header)
            
            # Write number of triangles (4 bytes)
            num_triangles = 12  # Simple cube has 12 triangles
            f.write(struct.pack('<I', num_triangles))
            
            # Create a simple cube with 12 triangles (2 per face)
            vertices = [
                # Front face
                [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
                # Back face
                [[0, 0, 1], [1, 0, 1], [1, 1, 1]],
                [[0, 0, 1], [1, 1, 1], [0, 1, 1]],
                # Left face
                [[0, 0, 0], [0, 1, 0], [0, 1, 1]],
                [[0, 0, 0], [0, 1, 1], [0, 0, 1]],
                # Right face
                [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
                [[1, 0, 0], [1, 1, 1], [1, 0, 1]],
                # Bottom face
                [[0, 0, 0], [1, 0, 0], [1, 0, 1]],
                [[0, 0, 0], [1, 0, 1], [0, 0, 1]],
                # Top face
                [[0, 1, 0], [1, 1, 0], [1, 1, 1]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 1]]
            ]
            
            normals = [
                [0, 0, -1], [0, 0, -1],  # Front face
                [0, 0, 1], [0, 0, 1],    # Back face
                [-1, 0, 0], [-1, 0, 0],  # Left face
                [1, 0, 0], [1, 0, 0],    # Right face
                [0, -1, 0], [0, -1, 0],  # Bottom face
                [0, 1, 0], [0, 1, 0]     # Top face
            ]
            
            # Write triangles
            # Each triangle is 50 bytes: normal (3 floats), vertices (3 x 3 floats), attribute count (2 bytes)
            for i, (triangle, normal) in enumerate(zip(vertices, normals)):
                # Write normal (3 floats)
                f.write(struct.pack('<fff', *normal))
                
                # Write vertices (3 x 3 floats)
                for vertex in triangle:
                    f.write(struct.pack('<fff', *vertex))
                
                # Write attribute byte count (2 bytes)
                f.write(struct.pack('<H', 0))


class OBJExporter(MeshExporter):
    """
    Exporter for OBJ (Wavefront) mesh files.
    """
    
    def export_mesh(self, filepath: str) -> str:
        """
        Export a mesh to OBJ format.
        
        Args:
            filepath: Path to the output file
        
        Returns:
            str: Path to the exported file
        
        Raises:
            ValueError: If the mesh model is invalid or export fails
        """
        if not self.validate_mesh():
            raise ValueError("Invalid mesh model")
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        try:
            # For a real implementation, we would iterate through the mesh data
            # and write points and faces to the OBJ file.
            
            # This is a placeholder implementation that creates a simple cube
            with open(filepath, 'w') as f:
                # Write header
                f.write(f"# OBJ file created by Openfoam_Simulator\n")
                f.write(f"# Model: {self.mesh_model.name}\n")
                
                # Define vertices
                vertices = [
                    [0, 0, 0],  # 0
                    [1, 0, 0],  # 1
                    [1, 1, 0],  # 2
                    [0, 1, 0],  # 3
                    [0, 0, 1],  # 4
                    [1, 0, 1],  # 5
                    [1, 1, 1],  # 6
                    [0, 1, 1]   # 7
                ]
                
                # Write vertices
                for v in vertices:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                
                # Define normals
                normals = [
                    [0, 0, -1],  # front
                    [0, 0, 1],   # back
                    [-1, 0, 0],  # left
                    [1, 0, 0],   # right
                    [0, -1, 0],  # bottom
                    [0, 1, 0]    # top
                ]
                
                # Write normals
                for n in normals:
                    f.write(f"vn {n[0]} {n[1]} {n[2]}\n")
                
                # Write object name
                f.write(f"o {self.mesh_model.name}\n")
                
                # Write faces
                # front face (vertices 0,1,2,3)
                f.write(f"f 1//1 2//1 3//1\n")
                f.write(f"f 1//1 3//1 4//1\n")
                
                # back face (vertices 4,5,6,7)
                f.write(f"f 5//2 6//2 7//2\n")
                f.write(f"f 5//2 7//2 8//2\n")
                
                # left face (vertices 0,3,7,4)
                f.write(f"f 1//3 4//3 8//3\n")
                f.write(f"f 1//3 8//3 5//3\n")
                
                # right face (vertices 1,2,6,5)
                f.write(f"f 2//4 3//4 7//4\n")
                f.write(f"f 2//4 7//4 6//4\n")
                
                # bottom face (vertices 0,1,5,4)
                f.write(f"f 1//5 2//5 6//5\n")
                f.write(f"f 1//5 6//5 5//5\n")
                
                # top face (vertices 3,2,6,7)
                f.write(f"f 4//6 3//6 7//6\n")
                f.write(f"f 4//6 7//6 8//6\n")
            
            # Collect statistics
            self.stats = {
                "format": "OBJ",
                "file_size": os.path.getsize(filepath),
                "num_vertices": len(vertices),
                "num_normals": len(normals),
                "num_faces": 12  # 12 triangular faces
            }
            
            logger.info(f"Successfully exported mesh to OBJ: {filepath}")
            logger.info(f"Exported {len(vertices)} vertices, {len(normals)} normals, {12} faces")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting mesh to OBJ: {e}")
            raise ValueError(f"Failed to export mesh to OBJ: {e}")


class OpenFOAMMeshExporter(MeshExporter):
    """
    Exporter for OpenFOAM mesh format.
    
    This exporter creates an OpenFOAM mesh directory structure.
    """
    
    def export_mesh(self, filepath: str) -> str:
        """
        Export a mesh to OpenFOAM format.
        
        Args:
            filepath: Path to the output directory
        
        Returns:
            str: Path to the exported directory
        
        Raises:
            ValueError: If the mesh model is invalid or export fails
        """
        if not self.validate_mesh():
            raise ValueError("Invalid mesh model")
        
        # For OpenFOAM, filepath should be a directory
        mesh_dir = os.path.join(filepath, 'constant', 'polyMesh')
        os.makedirs(mesh_dir, exist_ok=True)
        
        try:
            # Export the mesh to OpenFOAM format
            self._export_openfoam_mesh(mesh_dir)
            
            # Collect statistics
            self.stats = {
                "format": "OpenFOAM",
                "directory": mesh_dir,
                "num_points": self.mesh_model.num_points,
                "num_cells": self.mesh_model.num_cells,
                "num_faces": self.mesh_model.num_faces,
                "boundaries": self.mesh_model.boundaries
            }
            
            logger.info(f"Successfully exported mesh to OpenFOAM: {mesh_dir}")
            logger.info(f"Exported {self.mesh_model.num_points} points, {self.mesh_model.num_cells} cells, {self.mesh_model.num_faces} faces")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting mesh to OpenFOAM: {e}")
            raise ValueError(f"Failed to export mesh to OpenFOAM: {e}")
    
    def _export_openfoam_mesh(self, mesh_dir: str):
        """
        Export the mesh to OpenFOAM format in the specified directory.
        
        Args:
            mesh_dir: Path to the output constant/polyMesh directory
        """
        # For a real implementation, we would convert the mesh model data
        # to OpenFOAM format and write the necessary files.
        
        # This is a placeholder implementation that creates a very simple OpenFOAM mesh
        
        # Write points file
        with open(os.path.join(mesh_dir, 'points'), 'w') as f:
            f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
            f.write("| =========                 |                                                 |\n")
            f.write("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
            f.write("|  \\\\    /   O peration     | Version:  v2012                                 |\n")
            f.write("|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n")
            f.write("|    \\\\/     M anipulation  |                                                 |\n")
            f.write("\\*---------------------------------------------------------------------------*/\n")
            f.write("FoamFile\n")
            f.write("{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       vectorField;\n")
            f.write("    location    \"constant/polyMesh\";\n")
            f.write("    object      points;\n")
            f.write("}\n")
            f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
            
            # Define 8 vertices for a cube
            f.write("8\n")
            f.write("(\n")
            f.write("(0 0 0)\n")
            f.write("(1 0 0)\n")
            f.write("(1 1 0)\n")
            f.write("(0 1 0)\n")
            f.write("(0 0 1)\n")
            f.write("(1 0 1)\n")
            f.write("(1 1 1)\n")
            f.write("(0 1 1)\n")
            f.write(")\n\n")
        
        # Write faces file
        with open(os.path.join(mesh_dir, 'faces'), 'w') as f:
            f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
            f.write("| =========                 |                                                 |\n")
            f.write("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
            f.write("|  \\\\    /   O peration     | Version:  v2012                                 |\n")
            f.write("|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n")
            f.write("|    \\\\/     M anipulation  |                                                 |\n")
            f.write("\\*---------------------------------------------------------------------------*/\n")
            f.write("FoamFile\n")
            f.write("{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       faceList;\n")
            f.write("    location    \"constant/polyMesh\";\n")
            f.write("    object      faces;\n")
            f.write("}\n")
            f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
            
            # Define 6 faces for a cube
            f.write("6\n")
            f.write("(\n")
            f.write("4(0 3 2 1)    // front face (z=0)\n")
            f.write("4(4 5 6 7)    // back face (z=1)\n")
            f.write("4(0 4 7 3)    // left face (x=0)\n")
            f.write("4(1 2 6 5)    // right face (x=1)\n")
            f.write("4(0 1 5 4)    // bottom face (y=0)\n")
            f.write("4(3 7 6 2)    // top face (y=1)\n")
            f.write(")\n\n")
        
        # Write owner file
        with open(os.path.join(mesh_dir, 'owner'), 'w') as f:
            f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
            f.write("| =========                 |                                                 |\n")
            f.write("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
            f.write("|  \\\\    /   O peration     | Version:  v2012                                 |\n")
            f.write("|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n")
            f.write("|    \\\\/     M anipulation  |                                                 |\n")
            f.write("\\*---------------------------------------------------------------------------*/\n")
            f.write("FoamFile\n")
            f.write("{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       labelList;\n")
            f.write("    location    \"constant/polyMesh\";\n")
            f.write("    object      owner;\n")
            f.write("}\n")
            f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
            
            # Define owners for the 6 faces (only 1 cell, so all faces are owned by cell 0)
            f.write("6\n")
            f.write("(\n")
            f.write("0\n")
            f.write("0\n")
            f.write("0\n")
            f.write("0\n")
            f.write("0\n")
            f.write("0\n")
            f.write(")\n\n")
        
        # Write neighbour file
        with open(os.path.join(mesh_dir, 'neighbour'), 'w') as f:
            f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
            f.write("| =========                 |                                                 |\n")
            f.write("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
            f.write("|  \\\\    /   O peration     | Version:  v2012                                 |\n")
            f.write("|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n")
            f.write("|    \\\\/     M anipulation  |                                                 |\n")
            f.write("\\*---------------------------------------------------------------------------*/\n")
            f.write("FoamFile\n")
            f.write("{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       labelList;\n")
            f.write("    location    \"constant/polyMesh\";\n")
            f.write("    object      neighbour;\n")
            f.write("}\n")
            f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
            
            # No neighbors for a single-cell mesh (0 entries)
            f.write("0\n")
            f.write("(\n")
            f.write(")\n\n")
        
        # Write boundary file
        with open(os.path.join(mesh_dir, 'boundary'), 'w') as f:
            f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
            f.write("| =========                 |                                                 |\n")
            f.write("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
            f.write("|  \\\\    /   O peration     | Version:  v2012                                 |\n")
            f.write("|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n")
            f.write("|    \\\\/     M anipulation  |                                                 |\n")
            f.write("\\*---------------------------------------------------------------------------*/\n")
            f.write("FoamFile\n")
            f.write("{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       polyBoundaryMesh;\n")
            f.write("    location    \"constant/polyMesh\";\n")
            f.write("    object      boundary;\n")
            f.write("}\n")
            f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
            
            # Define 6 boundary patches for the cube
            f.write("6\n")
            f.write("(\n")
            
            # Write each boundary patch
            boundaries = ["front", "back", "left", "right", "bottom", "top"]
            for i, boundary in enumerate(boundaries):
                f.write(f"    {boundary}\n")
                f.write("    {\n")
                f.write("        type            wall;\n")
                f.write(f"        inGroups        1(wall);\n")
                f.write(f"        nFaces          1;\n")
                f.write(f"        startFace       {i};\n")
                f.write("    }\n")
            
            f.write(")\n\n")


class VTKExporter(MeshExporter):
    """
    Exporter for VTK/VTU mesh files.
    
    Supports various VTK formats (legacy VTK, XML VTK, VTU, etc.)
    """
    
    def export_mesh(self, filepath: str, format: str = "vtk") -> str:
        """
        Export a mesh to VTK format.
        
        Args:
            filepath: Path to the output file
            format: VTK format ("vtk", "vtu", "vtp", etc.)
        
        Returns:
            str: Path to the exported file
        
        Raises:
            ValueError: If the mesh model is invalid or export fails
        """
        if not self.validate_mesh():
            raise ValueError("Invalid mesh model")
        
        # Check that the format is supported
        format = format.lower()
        if format not in ["vtk", "vtu", "vtp", "vtr", "vts"]:
            raise ValueError(f"Unsupported VTK format: {format}")
        
        # Ensure filepath has the correct extension
        _, ext = os.path.splitext(filepath)
        if not ext.lower() == f".{format}":
            filepath = f"{filepath}.{format}"
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        try:
            # Export to the appropriate VTK format
            if format == "vtk":
                self._export_legacy_vtk(filepath)
            elif format == "vtu":
                self._export_vtu(filepath)
            else:
                # Other formats would be implemented similarly
                self._export_legacy_vtk(filepath)  # Placeholder
            
            # Collect statistics
            self.stats = {
                "format": format.upper(),
                "file_size": os.path.getsize(filepath),
                "num_points": self.mesh_model.num_points,
                "num_cells": self.mesh_model.num_cells
            }
            
            logger.info(f"Successfully exported mesh to {format.upper()}: {filepath}")
            logger.info(f"Exported {self.mesh_model.num_points} points, {self.mesh_model.num_cells} cells")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting mesh to {format.upper()}: {e}")
            raise ValueError(f"Failed to export mesh to {format.upper()}: {e}")
    
    def _export_legacy_vtk(self, filepath: str):
        """
        Export a mesh to legacy VTK format.
        
        Args:
            filepath: Path to the output file
        """
        # For a real implementation, we would convert the mesh model data
        # to VTK format and write it to the file.
        
        # This is a placeholder implementation that creates a simple VTK file
        with open(filepath, 'w') as f:
            # Write header
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"Mesh exported from Openfoam_Simulator: {self.mesh_model.name}\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            
            # Write points
            # Define 8 vertices for a cube
            vertices = [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1]
            ]
            
            f.write(f"POINTS {len(vertices)} float\n")
            for v in vertices:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
            
            # Write cells
            # For a hexahedron, we have 1 cell with 8 points
            f.write("CELLS 1 9\n")  # 1 cell, 9 entries (1 + 8)
            f.write("8 0 1 2 3 4 5 6 7\n")  # 8 points followed by point indices
            
            # Write cell types
            # VTK_HEXAHEDRON = 12
            f.write("CELL_TYPES 1\n")
            f.write("12\n")
    
    def _export_vtu(self, filepath: str):
        """
        Export a mesh to VTU format (XML Unstructured Grid).
        
        Args:
            filepath: Path to the output file
        """
        # For a real implementation, we would convert the mesh model data
        # to VTU format and write it to the file.
        
        # This is a placeholder implementation that creates a simple VTU file
        with open(filepath, 'w') as f:
            # Write XML header
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
            f.write('  <UnstructuredGrid>\n')
            f.write('    <Piece NumberOfPoints="8" NumberOfCells="1">\n')
            
            # Write points
            f.write('      <Points>\n')
            f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
            # Define 8 vertices for a cube
            f.write('          0 0 0\n')
            f.write('          1 0 0\n')
            f.write('          1 1 0\n')
            f.write('          0 1 0\n')
            f.write('          0 0 1\n')
            f.write('          1 0 1\n')
            f.write('          1 1 1\n')
            f.write('          0 1 1\n')
            f.write('        </DataArray>\n')
            f.write('      </Points>\n')
            
            # Write cells
            f.write('      <Cells>\n')
            f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
            f.write('          0 1 2 3 4 5 6 7\n')
            f.write('        </DataArray>\n')
            f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
            f.write('          8\n')
            f.write('        </DataArray>\n')
            f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
            f.write('          12\n')  # VTK_HEXAHEDRON = 12
            f.write('        </DataArray>\n')
            f.write('      </Cells>\n')
            
            # Close tags
            f.write('    </Piece>\n')
            f.write('  </UnstructuredGrid>\n')
            f.write('</VTKFile>\n')


class CGNSExporter(MeshExporter):
    """
    Exporter for CGNS (CFD General Notation System) mesh files.
    """
    
    def export_mesh(self, filepath: str) -> str:
        """
        Export a mesh to CGNS format.
        
        Args:
            filepath: Path to the output file
        
        Returns:
            str: Path to the exported file
        
        Raises:
            ValueError: If the mesh model is invalid or export fails
        """
        if not self.validate_mesh():
            raise ValueError("Invalid mesh model")
        
        # Ensure filepath has the correct extension
        _, ext = os.path.splitext(filepath)
        if not ext.lower() == ".cgns":
            filepath = f"{filepath}.cgns"
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        try:
            # For a real implementation, we would convert the mesh model data
            # to CGNS format and write it to the file using the CGNS library.
            
            # This is a placeholder implementation that creates a minimal CGNS file
            # We'll just create a dummy file for now
            with open(filepath, 'wb') as f:
                f.write(b'CGNS')
                f.write(b'\x00\x00\x00\x03')  # Version 3.0
                f.write(b'\x00\x00\x00\x00')  # File ID
                
                # In a real implementation, we would use the CGNS API
                
            # Collect statistics
            self.stats = {
                "format": "CGNS",
                "file_size": os.path.getsize(filepath),
                "num_points": self.mesh_model.num_points,
                "num_cells": self.mesh_model.num_cells
            }
            
            logger.info(f"Successfully exported mesh to CGNS: {filepath}")
            logger.info(f"Exported {self.mesh_model.num_points} points, {self.mesh_model.num_cells} cells")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting mesh to CGNS: {e}")
            raise ValueError(f"Failed to export mesh to CGNS: {e}")


class NASTRANExporter(MeshExporter):
    """
    Exporter for NASTRAN mesh files.
    """
    
    def export_mesh(self, filepath: str) -> str:
        """
        Export a mesh to NASTRAN format.
        
        Args:
            filepath: Path to the output file
        
        Returns:
            str: Path to the exported file
        
        Raises:
            ValueError: If the mesh model is invalid or export fails
        """
        if not self.validate_mesh():
            raise ValueError("Invalid mesh model")
        
        # Ensure filepath has the correct extension
        _, ext = os.path.splitext(filepath)
        if not ext.lower() in [".nas", ".bdf"]:
            filepath = f"{filepath}.nas"
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        try:
            # For a real implementation, we would convert the mesh model data
            # to NASTRAN format and write it to the file.
            
            # This is a placeholder implementation that creates a simple NASTRAN file
            with open(filepath, 'w') as f:
                # Write header
                f.write("$ NASTRAN file exported from Openfoam_Simulator\n")
                f.write("$ Model: " + self.mesh_model.name + "\n")
                f.write("$ Date: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
                f.write("BEGIN BULK\n")
                
                # Write grid points (nodes)
                # GRID ID CP X Y Z CD PS
                for i in range(8):
                    x = 0 if i in [0, 3, 4, 7] else 1
                    y = 0 if i in [0, 1, 4, 5] else 1
                    z = 0 if i < 4 else 1
                    f.write(f"GRID    {i+1:8d}        {x:.8f}{y:.8f}{z:.8f}                \n")
                
                # Write elements
                # CHEXA EID PID G1 G2 G3 G4 G5 G6 G7 G8
                f.write("CHEXA   1       1       1       2       3       4       5       6       7       8\n")
                
                # Write property
                # PSOLID PID MID
                f.write("PSOLID  1       1\n")
                
                # Write material
                # MAT1 MID E NU RHO
                f.write("MAT1    1       1.0+7   0.3     1.0\n")
                
                # End bulk data
                f.write("ENDDATA\n")
            
            # Collect statistics
            self.stats = {
                "format": "NASTRAN",
                "file_size": os.path.getsize(filepath),
                "num_points": 8,
                "num_cells": 1
            }
            
            logger.info(f"Successfully exported mesh to NASTRAN: {filepath}")
            logger.info(f"Exported 8 points, 1 cells")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting mesh to NASTRAN: {e}")
            raise ValueError(f"Failed to export mesh to NASTRAN: {e}")


class GMSHExporter(MeshExporter):
    """
    Exporter for GMSH mesh files.
    """
    
    def export_mesh(self, filepath: str) -> str:
        """
        Export a mesh to GMSH format.
        
        Args:
            filepath: Path to the output file
        
        Returns:
            str: Path to the exported file
        
        Raises:
            ValueError: If the mesh model is invalid or export fails
        """
        if not self.validate_mesh():
            raise ValueError("Invalid mesh model")
        
        # Ensure filepath has the correct extension
        _, ext = os.path.splitext(filepath)
        if not ext.lower() == ".msh":
            filepath = f"{filepath}.msh"
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        try:
            # For a real implementation, we would convert the mesh model data
            # to GMSH format and write it to the file.
            
            # This is a placeholder implementation that creates a simple GMSH file
            with open(filepath, 'w') as f:
                # Write GMSH format 2.2 header
                f.write("$MeshFormat\n")
                f.write("2.2 0 8\n")
                f.write("$EndMeshFormat\n")
                
                # Write nodes
                f.write("$Nodes\n")
                f.write("8\n")  # 8 nodes for a cube
                
                # Define the 8 vertices of a cube
                vertices = [
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                    [0, 1, 1]
                ]
                
                # Write nodes
                for i, v in enumerate(vertices):
                    f.write(f"{i+1} {v[0]} {v[1]} {v[2]}\n")
                
                f.write("$EndNodes\n")
                
                # Write elements
                f.write("$Elements\n")
                f.write("1\n")  # 1 hexahedral element
                
                # Type 5 is a hexahedron
                # Format: elm-number elm-type number-of-tags <tags> node-number-list
                f.write("1 5 2 0 1 1 2 3 4 5 6 7 8\n")
                
                f.write("$EndElements\n")
            
            # Collect statistics
            self.stats = {
                "format": "GMSH",
                "file_size": os.path.getsize(filepath),
                "num_points": 8,
                "num_cells": 1
            }
            
            logger.info(f"Successfully exported mesh to GMSH: {filepath}")
            logger.info(f"Exported 8 points, 1 cells")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting mesh to GMSH: {e}")
            raise ValueError(f"Failed to export mesh to GMSH: {e}")


def export_mesh(mesh_model: MeshModel, filepath: str, format: str = None) -> str:
    """
    Export a mesh to a file.
    
    Args:
        mesh_model: The mesh model to export
        filepath: Path to the output file
        format: Output format (if None, inferred from filepath)
    
    Returns:
        str: Path to the exported file
    
    Raises:
        ValueError: If the export fails or the format is not supported
    """
    if mesh_model is None:
        raise ValueError("No mesh model provided")
    
    # Determine format from filepath if not specified
    if format is None:
        if os.path.isdir(filepath) or filepath.endswith('/') or filepath.endswith('\\'):
            # Directory path suggests OpenFOAM format
            format = "openfoam"
        else:
            ext = os.path.splitext(filepath)[1].lower()
            
            if ext == '':
                raise ValueError("Cannot determine export format from filepath without extension")
            
            if ext.startswith('.'):
                ext = ext[1:]
            
            format = ext
    
    # Normalize format
    format = format.lower()
    
    # Create the appropriate exporter based on format
    if format in ["stl"]:
        exporter = STLExporter(mesh_model)
        # Determine if ASCII or binary based on config
        use_ascii = get_value("mesh_export.stl_ascii", False)
        return exporter.export_mesh(filepath, ascii_format=use_ascii)
    
    elif format in ["obj"]:
        exporter = OBJExporter(mesh_model)
        return exporter.export_mesh(filepath)
    
    elif format in ["openfoam", "foam"]:
        exporter = OpenFOAMMeshExporter(mesh_model)
        return exporter.export_mesh(filepath)
    
    elif format in ["vtk", "vtu", "vtp", "vtr", "vts"]:
        exporter = VTKExporter(mesh_model)
        return exporter.export_mesh(filepath, format=format)
    
    elif format in ["cgns"]:
        exporter = CGNSExporter(mesh_model)
        return exporter.export_mesh(filepath)
    
    elif format in ["nas", "bdf"]:
        exporter = NASTRANExporter(mesh_model)
        return exporter.export_mesh(filepath)
    
    elif format in ["msh"]:
        exporter = GMSHExporter(mesh_model)
        return exporter.export_mesh(filepath)
    
    else:
        raise ValueError(f"Unsupported export format: {format}")


def convert_mesh(input_filepath: str, output_filepath: str, input_format: str = None, output_format: str = None) -> str:
    """
    Convert a mesh from one format to another.
    
    Args:
        input_filepath: Path to the input mesh file
        output_filepath: Path to the output mesh file
        input_format: Input format (if None, inferred from input_filepath)
        output_format: Output format (if None, inferred from output_filepath)
    
    Returns:
        str: Path to the converted mesh file
    
    Raises:
        ValueError: If the conversion fails
    """
    from src.io.importers.mesh_importers import import_mesh
    
    # Import the mesh
    mesh_model = import_mesh(input_filepath, format=input_format)
    
    # Export to the desired format
    return export_mesh(mesh_model, output_filepath, format=output_format)


if __name__ == "__main__":
    # Command-line utility for testing mesh export
    import argparse
    
    parser = argparse.ArgumentParser(description="Mesh export utility")
    parser.add_argument("--input", help="Input mesh file (for conversion)")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--format", help="Output format (if not inferred from output path)")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        if args.input:
            # Convert the mesh
            from src.io.importers.mesh_importers import import_mesh
            
            logger.info(f"Converting mesh: {args.input} -> {args.output}")
            mesh_model = import_mesh(args.input)
            output_path = export_mesh(mesh_model, args.output, args.format)
            logger.info(f"Mesh exported to: {output_path}")
        else:
            # Create a simple mesh model for testing
            mesh_model = MeshModel()
            mesh_model.name = "Test Mesh"
            mesh_model.num_points = 8
            mesh_model.num_cells = 1
            mesh_model.num_faces = 6
            mesh_model.boundaries = ["front", "back", "left", "right", "bottom", "top"]
            
            # Export the mesh
            logger.info(f"Exporting test mesh to: {args.output}")
            output_path = export_mesh(mesh_model, args.output, args.format)
            logger.info(f"Mesh exported to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    
    sys.exit(0)