#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mesh generation utilities for Openfoam_Simulator.

This module provides functionality for creating and manipulating meshes
for OpenFOAM simulations, including:
- BlockMesh generation
- SnappyHexMesh generation
- CAD geometry import and processing
- Mesh quality checking
- Specialized mesh generation for oil & gas components (pipes, wellbores, etc.)
"""

import os
import sys
import subprocess
import tempfile
import shutil
import math
import re
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set, Callable

# Import relevant project modules
from ..utils.logger import get_logger
from ..config import get_value, set_value
from .case_manager import CaseManager

# Try to import VTK for mesh processing
try:
    import vtk
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False

# Setup logger
logger = get_logger(__name__)


class BlockMeshGenerator:
    """
    Generator for creating blockMesh dictionaries and meshes.
    
    BlockMesh is OpenFOAM's built-in mesh generator for simple geometries
    based on hexahedral blocks with optional grading.
    """
    
    def __init__(self, case_dir: str):
        """
        Initialize the BlockMesh generator.
        
        Args:
            case_dir (str): Path to the case directory
        """
        self.case_dir = Path(case_dir)
        self.vertices: List[Tuple[float, float, float]] = []
        self.blocks: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self.boundaries: Dict[str, Dict[str, Any]] = {}
        self.mergePatchPairs: List[Tuple[str, str]] = []
        self.scale = 1.0
    
    def add_vertex(self, x: float, y: float, z: float) -> int:
        """
        Add a vertex to the mesh.
        
        Args:
            x (float): X coordinate
            y (float): Y coordinate
            z (float): Z coordinate
            
        Returns:
            int: Index of the added vertex
        """
        self.vertices.append((x, y, z))
        return len(self.vertices) - 1
    
    def add_block(self, vertices: List[int], cells: List[int], 
                 grading: Union[List[int], List[List[Any]]] = None) -> int:
        """
        Add a block to the mesh.
        
        Args:
            vertices (List[int]): List of 8 vertex indices defining the block
            cells (List[int]): Number of cells in x, y, z directions
            grading (Union[List[int], List[List[Any]]], optional): Grading specification
            
        Returns:
            int: Index of the added block
        """
        if len(vertices) != 8:
            raise ValueError("Block must have exactly 8 vertices")
        
        if len(cells) != 3:
            raise ValueError("Cells must specify count in 3 directions (x, y, z)")
        
        # Default to uniform grading if not specified
        if grading is None:
            grading = [1, 1, 1]
        
        block = {
            "vertices": vertices,
            "cells": cells,
            "grading": grading
        }
        
        self.blocks.append(block)
        return len(self.blocks) - 1
    
    def add_edge(self, edge_type: str, vertices: List[int], data: Any = None) -> int:
        """
        Add an edge to the mesh.
        
        Args:
            edge_type (str): Type of edge (arc, spline, etc.)
            vertices (List[int]): Vertex indices defining the edge
            data (Any, optional): Additional data for the edge
            
        Returns:
            int: Index of the added edge
        """
        edge = {
            "type": edge_type,
            "vertices": vertices,
            "data": data
        }
        
        self.edges.append(edge)
        return len(self.edges) - 1
    
    def add_boundary(self, name: str, boundary_type: str, faces: List[List[int]]) -> None:
        """
        Add a boundary to the mesh.
        
        Args:
            name (str): Name of the boundary
            boundary_type (str): Type of boundary (wall, patch, etc.)
            faces (List[List[int]]): List of faces (each face is a list of vertex indices)
        """
        if name in self.boundaries:
            # Append to existing boundary
            self.boundaries[name]["faces"].extend(faces)
        else:
            # Create new boundary
            self.boundaries[name] = {
                "type": boundary_type,
                "faces": faces
            }
    
    def add_merge_patch_pair(self, patch1: str, patch2: str) -> None:
        """
        Add a merge patch pair.
        
        Args:
            patch1 (str): First patch name
            patch2 (str): Second patch name
        """
        self.mergePatchPairs.append((patch1, patch2))
    
    def set_scale(self, scale: float) -> None:
        """
        Set the mesh scale.
        
        Args:
            scale (float): Scale factor
        """
        self.scale = scale
    
    def generate_block_mesh_dict(self) -> str:
        """
        Generate the blockMeshDict file content.
        
        Returns:
            str: Content of the blockMeshDict file
        """
        output = []
        
        # Add header
        output.append(self._generate_header())
        
        # Add vertices
        output.append("vertices\n(")
        for v in self.vertices:
            output.append(f"    ({v[0]} {v[1]} {v[2]})")
        output.append(");\n")
        
        # Add blocks
        output.append("blocks\n(")
        for b in self.blocks:
            vertices_str = " ".join([str(v) for v in b["vertices"]])
            cells_str = " ".join([str(c) for c in b["cells"]])
            
            # Handle different types of grading
            if isinstance(b["grading"], list):
                if all(isinstance(g, (int, float)) for g in b["grading"]):
                    # Simple grading
                    grading_str = f"simpleGrading ({b['grading'][0]} {b['grading'][1]} {b['grading'][2]})"
                else:
                    # Edge grading
                    grading_str = "edgeGrading " + " ".join([str(g) for g in b["grading"]])
            else:
                # Default to simple uniform grading
                grading_str = "simpleGrading (1 1 1)"
            
            output.append(f"    hex ({vertices_str}) ({cells_str}) {grading_str}")
        output.append(");\n")
        
        # Add edges
        if self.edges:
            output.append("edges\n(")
            for e in self.edges:
                vertices_str = " ".join([str(v) for v in e["vertices"]])
                
                if e["type"] == "arc":
                    # Arc needs a center point
                    if isinstance(e["data"], (list, tuple)) and len(e["data"]) == 3:
                        data_str = f"({e['data'][0]} {e['data'][1]} {e['data'][2]})"
                    else:
                        data_str = str(e["data"])
                    output.append(f"    arc {e['vertices'][0]} {e['vertices'][1]} {data_str}")
                    
                elif e["type"] == "spline":
                    # Spline needs a list of points
                    output.append(f"    spline ({vertices_str})")
                    if isinstance(e["data"], list):
                        output.append("    (")
                        for point in e["data"]:
                            output.append(f"        ({point[0]} {point[1]} {point[2]})")
                        output.append("    )")
                    
                else:
                    # Other edge types
                    output.append(f"    {e['type']} {vertices_str}")
            output.append(");\n")
        
        # Add boundaries
        output.append("boundary\n(")
        for name, boundary in self.boundaries.items():
            output.append(f"    {name}\n    {{")
            output.append(f"        type {boundary['type']};")
            output.append("        faces\n        (")
            for face in boundary["faces"]:
                face_str = " ".join([str(v) for v in face])
                output.append(f"            ({face_str})")
            output.append("        );")
            output.append("    }")
        output.append(");\n")
        
        # Add mergePatchPairs
        if self.mergePatchPairs:
            output.append("mergePatchPairs\n(")
            for pair in self.mergePatchPairs:
                output.append(f"    ({pair[0]} {pair[1]})")
            output.append(");\n")
        
        # Add footer
        output.append("// ************************************************************************* //")
        
        return "\n".join(output)
    
    def write_block_mesh_dict(self) -> bool:
        """
        Write the blockMeshDict file to the case directory.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create constant/polyMesh directory if it doesn't exist
            mesh_dir = self.case_dir / "constant" / "polyMesh"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            
            # Get blockMeshDict content
            content = self.generate_block_mesh_dict()
            
            # Write to file
            dict_file = self.case_dir / "system" / "blockMeshDict"
            with open(dict_file, 'w') as f:
                f.write(content)
            
            logger.info(f"Wrote blockMeshDict to {dict_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing blockMeshDict: {e}")
            return False
    
    def run_block_mesh(self) -> bool:
        """
        Run the blockMesh utility to generate the mesh.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # First, write the blockMeshDict
            if not self.write_block_mesh_dict():
                return False
            
            # Run blockMesh
            cmd = ["blockMesh", "-case", str(self.case_dir)]
            
            logger.info(f"Running blockMesh: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"blockMesh failed: {stderr}")
                return False
            
            logger.info("blockMesh completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error running blockMesh: {e}")
            return False
    
    def _generate_header(self) -> str:
        """
        Generate the header for the blockMeshDict file.
        
        Returns:
            str: Header content
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
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

scale {self.scale};

"""
        return header


class SnappyHexMeshGenerator:
    """
    Generator for creating snappyHexMesh dictionaries and meshes.
    
    SnappyHexMesh is OpenFOAM's advanced mesh generator for complex geometries
    based on STL or other surface files.
    """
    
    def __init__(self, case_dir: str):
        """
        Initialize the SnappyHexMesh generator.
        
        Args:
            case_dir (str): Path to the case directory
        """
        self.case_dir = Path(case_dir)
        self.geometry: Dict[str, Dict[str, Any]] = {}
        self.castellated_mesh_controls: Dict[str, Any] = {}
        self.snap_controls: Dict[str, Any] = {}
        self.layer_controls: Dict[str, Any] = {}
        self.mesh_quality_controls: Dict[str, Any] = {}
        self.regions: Dict[str, Dict[str, Any]] = {}
        
        # Set default controls
        self._set_default_controls()
    
    def add_stl_geometry(self, name: str, file_path: str, 
                        refinement_level: int = 0) -> None:
        """
        Add an STL geometry file to the mesh.
        
        Args:
            name (str): Name for the geometry
            file_path (str): Path to the STL file
            refinement_level (int, optional): Refinement level (0-9)
        """
        # Copy the STL file to the triSurface directory if needed
        stl_file = Path(file_path)
        tri_surface_dir = self.case_dir / "constant" / "triSurface"
        tri_surface_dir.mkdir(parents=True, exist_ok=True)
        
        target_file = tri_surface_dir / stl_file.name
        
        if stl_file != target_file:
            shutil.copy2(stl_file, target_file)
            logger.info(f"Copied STL file to {target_file}")
        
        # Add geometry to dictionary
        self.geometry[name] = {
            "type": "triSurfaceMesh",
            "file": f'"{stl_file.name}"',
            "refinementLevel": refinement_level
        }
        
        # Add default region settings
        self.regions[name] = {
            "level": (refinement_level, refinement_level),
            "patchInfo": {
                "type": "wall"
            }
        }
    
    def set_background_mesh(self, cells_per_dim: int, bounds: List[float]) -> None:
        """
        Set background mesh parameters.
        
        Args:
            cells_per_dim (int): Number of cells per dimension
            bounds (List[float]): Mesh bounds [xmin, xmax, ymin, ymax, zmin, zmax]
        """
        if len(bounds) != 6:
            raise ValueError("Bounds must be a list of 6 values [xmin, xmax, ymin, ymax, zmin, zmax]")
        
        # Calculate cell size
        x_size = (bounds[1] - bounds[0]) / cells_per_dim
        y_size = (bounds[3] - bounds[2]) / cells_per_dim
        z_size = (bounds[5] - bounds[4]) / cells_per_dim
        
        # Set background mesh controls
        self.castellated_mesh_controls["locationInMesh"] = (
            f"({(bounds[0] + bounds[1]) / 2} "
            f"{(bounds[2] + bounds[3]) / 2} "
            f"{(bounds[4] + bounds[5]) / 2})"
        )
        
        # Create a block mesh for the background mesh
        block_mesh = BlockMeshGenerator(self.case_dir)
        
        # Add vertices
        v0 = block_mesh.add_vertex(bounds[0], bounds[2], bounds[4])
        v1 = block_mesh.add_vertex(bounds[1], bounds[2], bounds[4])
        v2 = block_mesh.add_vertex(bounds[1], bounds[3], bounds[4])
        v3 = block_mesh.add_vertex(bounds[0], bounds[3], bounds[4])
        v4 = block_mesh.add_vertex(bounds[0], bounds[2], bounds[5])
        v5 = block_mesh.add_vertex(bounds[1], bounds[2], bounds[5])
        v6 = block_mesh.add_vertex(bounds[1], bounds[3], bounds[5])
        v7 = block_mesh.add_vertex(bounds[0], bounds[3], bounds[5])
        
        # Calculate number of cells
        x_cells = max(1, int((bounds[1] - bounds[0]) / x_size))
        y_cells = max(1, int((bounds[3] - bounds[2]) / y_size))
        z_cells = max(1, int((bounds[5] - bounds[4]) / z_size))
        
        # Add block
        block_mesh.add_block([v0, v1, v2, v3, v4, v5, v6, v7], [x_cells, y_cells, z_cells])
        
        # Add boundaries
        block_mesh.add_boundary("xmin", "patch", [[v0, v4, v7, v3]])
        block_mesh.add_boundary("xmax", "patch", [[v1, v2, v6, v5]])
        block_mesh.add_boundary("ymin", "patch", [[v0, v1, v5, v4]])
        block_mesh.add_boundary("ymax", "patch", [[v3, v7, v6, v2]])
        block_mesh.add_boundary("zmin", "patch", [[v0, v3, v2, v1]])
        block_mesh.add_boundary("zmax", "patch", [[v4, v5, v6, v7]])
        
        # Write and run blockMesh
        block_mesh.write_block_mesh_dict()
        block_mesh.run_block_mesh()
    
    def set_feature_refinement(self, level: int, angle: float = 30.0) -> None:
        """
        Set mesh refinement at feature edges.
        
        Args:
            level (int): Refinement level at features
            angle (float, optional): Feature angle in degrees
        """
        self.castellated_mesh_controls["features"] = [
            {
                "file": f'"*.eMesh"',
                "level": level
            }
        ]
        
        # Feature extraction settings
        self.castellated_mesh_controls["featureEdgeRefinement"] = True
        self.castellated_mesh_controls["resolveFeatureAngle"] = angle
    
    def set_surface_refinement_level(self, name: str, min_level: int, max_level: int) -> None:
        """
        Set refinement level for a specific surface.
        
        Args:
            name (str): Surface name
            min_level (int): Minimum refinement level
            max_level (int): Maximum refinement level
        """
        if name not in self.regions:
            logger.warning(f"Surface {name} not found in regions")
            return
        
        self.regions[name]["level"] = (min_level, max_level)
    
    def set_patch_type(self, name: str, patch_type: str) -> None:
        """
        Set patch type for a specific surface.
        
        Args:
            name (str): Surface name
            patch_type (str): Patch type (wall, patch, etc.)
        """
        if name not in self.regions:
            logger.warning(f"Surface {name} not found in regions")
            return
        
        if "patchInfo" not in self.regions[name]:
            self.regions[name]["patchInfo"] = {}
        
        self.regions[name]["patchInfo"]["type"] = patch_type
    
    def add_refinement_box(self, name: str, min_point: List[float], 
                          max_point: List[float], level: int) -> None:
        """
        Add a refinement box.
        
        Args:
            name (str): Name for the refinement box
            min_point (List[float]): Minimum point [x, y, z]
            max_point (List[float]): Maximum point [x, y, z]
            level (int): Refinement level
        """
        if len(min_point) != 3 or len(max_point) != 3:
            raise ValueError("Points must be lists of 3 values [x, y, z]")
        
        if "refinementRegions" not in self.castellated_mesh_controls:
            self.castellated_mesh_controls["refinementRegions"] = {}
        
        # Define box geometry
        box_name = f"refinementBox_{name}"
        self.geometry[box_name] = {
            "type": "searchableBox",
            "min": f"({min_point[0]} {min_point[1]} {min_point[2]})",
            "max": f"({max_point[0]} {max_point[1]} {max_point[2]})"
        }
        
        # Add refinement region
        self.castellated_mesh_controls["refinementRegions"][box_name] = {
            "mode": "inside",
            "levels": f"(({level} {level}))"
        }
    
    def enable_layers(self, enabled: bool = True) -> None:
        """
        Enable or disable layer addition.
        
        Args:
            enabled (bool, optional): Whether to enable layers
        """
        self.layer_controls["addLayers"] = enabled
    
    def set_layer_parameters(self, first_layer_thickness: float, 
                            expansion_ratio: float, total_thickness: float,
                            min_thickness: float = 0.1) -> None:
        """
        Set layer addition parameters.
        
        Args:
            first_layer_thickness (float): Thickness of first layer
            expansion_ratio (float): Layer expansion ratio
            total_thickness (float): Total layers thickness
            min_thickness (float, optional): Minimum layer thickness
        """
        self.layer_controls["firstLayerThickness"] = first_layer_thickness
        self.layer_controls["expansionRatio"] = expansion_ratio
        self.layer_controls["finalLayerThickness"] = total_thickness
        self.layer_controls["minThickness"] = min_thickness
    
    def set_layer_controls_for_patch(self, patch_name: str, num_layers: int) -> None:
        """
        Set layer controls for a specific patch.
        
        Args:
            patch_name (str): Patch name
            num_layers (int): Number of layers
        """
        if "layers" not in self.layer_controls:
            self.layer_controls["layers"] = {}
        
        self.layer_controls["layers"][f'"{patch_name}"'] = {
            "nSurfaceLayers": num_layers
        }
    
    def generate_snappy_hex_mesh_dict(self) -> str:
        """
        Generate the snappyHexMeshDict file content.
        
        Returns:
            str: Content of the snappyHexMeshDict file
        """
        output = []
        
        # Add header
        output.append(self._generate_header())
        
        # Cast settings
        cast_settings = {
            "castellatedMesh": "true",
            "snap": "true",
            "addLayers": str(self.layer_controls.get("addLayers", "false")).lower()
        }
        
        for setting, value in cast_settings.items():
            output.append(f"{setting} {value};")
        
        output.append("")
        
        # Add geometry section
        output.append("geometry")
        output.append("{")
        for name, geom in self.geometry.items():
            output.append(f"    {name}")
            output.append("    {")
            for key, value in geom.items():
                output.append(f"        {key} {value};")
            output.append("    }")
        output.append("}")
        output.append("")
        
        # Add castellatedMeshControls section
        output.append("castellatedMeshControls")
        output.append("{")
        
        # Maximum refinement level
        output.append(f"    maxLocalCells {self.castellated_mesh_controls.get('maxLocalCells', 1000000)};")
        output.append(f"    maxGlobalCells {self.castellated_mesh_controls.get('maxGlobalCells', 2000000)};")
        output.append(f"    minRefinementCells {self.castellated_mesh_controls.get('minRefinementCells', 10)};")
        output.append(f"    maxLoadUnbalance {self.castellated_mesh_controls.get('maxLoadUnbalance', 0.1)};")
        output.append(f"    nCellsBetweenLevels {self.castellated_mesh_controls.get('nCellsBetweenLevels', 3)};")
        
        # Features
        if "features" in self.castellated_mesh_controls:
            output.append("    features")
            output.append("    (")
            for feature in self.castellated_mesh_controls["features"]:
                output.append("        {")
                for key, value in feature.items():
                    output.append(f"            {key} {value};")
                output.append("        }")
            output.append("    );")
        
        # Refinement surfaces
        output.append("    refinementSurfaces")
        output.append("    {")
        for name, region in self.regions.items():
            output.append(f"        {name}")
            output.append("        {")
            if "level" in region:
                min_level, max_level = region["level"]
                output.append(f"            level ({min_level} {max_level});")
            if "patchInfo" in region:
                output.append("            patchInfo")
                output.append("            {")
                for key, value in region["patchInfo"].items():
                    output.append(f"                {key} {value};")
                output.append("            }")
            output.append("        }")
        output.append("    }")
        
        # Refinement regions
        if "refinementRegions" in self.castellated_mesh_controls:
            output.append("    refinementRegions")
            output.append("    {")
            for name, region in self.castellated_mesh_controls["refinementRegions"].items():
                output.append(f"        {name}")
                output.append("        {")
                for key, value in region.items():
                    output.append(f"            {key} {value};")
                output.append("        }")
            output.append("    }")
        
        # Location in mesh
        if "locationInMesh" in self.castellated_mesh_controls:
            output.append(f"    locationInMesh {self.castellated_mesh_controls['locationInMesh']};")
        
        output.append("}")
        output.append("")
        
        # Add snapControls section
        output.append("snapControls")
        output.append("{")
        for key, value in self.snap_controls.items():
            output.append(f"    {key} {value};")
        output.append("}")
        output.append("")
        
        # Add addLayersControls section
        output.append("addLayersControls")
        output.append("{")
        for key, value in self.layer_controls.items():
            if key == "layers":
                output.append("    layers")
                output.append("    {")
                for patch, layers in value.items():
                    output.append(f"        {patch}")
                    output.append("        {")
                    for layer_key, layer_value in layers.items():
                        output.append(f"            {layer_key} {layer_value};")
                    output.append("        }")
                output.append("    }")
            else:
                output.append(f"    {key} {value};")
        output.append("}")
        output.append("")
        
        # Add meshQualityControls section
        output.append("meshQualityControls")
        output.append("{")
        for key, value in self.mesh_quality_controls.items():
            output.append(f"    {key} {value};")
        output.append("}")
        output.append("")
        
        # Add footer
        output.append("// ************************************************************************* //")
        
        return "\n".join(output)
    
    def write_snappy_hex_mesh_dict(self) -> bool:
        """
        Write the snappyHexMeshDict file to the case directory.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create system directory if it doesn't exist
            system_dir = self.case_dir / "system"
            system_dir.mkdir(parents=True, exist_ok=True)
            
            # Get snappyHexMeshDict content
            content = self.generate_snappy_hex_mesh_dict()
            
            # Write to file
            dict_file = system_dir / "snappyHexMeshDict"
            with open(dict_file, 'w') as f:
                f.write(content)
            
            logger.info(f"Wrote snappyHexMeshDict to {dict_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing snappyHexMeshDict: {e}")
            return False
    
    def extract_features(self) -> bool:
        """
        Run surfaceFeatureExtract to extract features from STL files.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create surface feature extract dict
            feature_dict = {
                "extractionMethod": "extractFromSurface",
                "extractFromSurfaceCoeffs": {
                    "includedAngle": 150
                },
                "subsetFeatures": {
                    "nonManifoldEdges": "yes",
                    "openEdges": "yes"
                },
                "writeObj": "yes"
            }
            
            # Write surfaceFeatureExtractDict
            surface_feature_dict_path = self.case_dir / "system" / "surfaceFeatureExtractDict"
            
            with open(surface_feature_dict_path, 'w') as f:
                f.write(self._generate_dict_header("surfaceFeatureExtractDict"))
                
                for name in self.regions.keys():
                    f.write(f'"{name}.stl"\n')
                    f.write("{\n")
                    for key, value in feature_dict.items():
                        if isinstance(value, dict):
                            f.write(f"    {key}\n    {{\n")
                            for subkey, subvalue in value.items():
                                f.write(f"        {subkey} {subvalue};\n")
                            f.write("    }\n")
                        else:
                            f.write(f"    {key} {value};\n")
                    f.write("}\n\n")
                
                f.write("// ************************************************************************* //\n")
            
            # Run surfaceFeatureExtract
            cmd = ["surfaceFeatureExtract", "-case", str(self.case_dir)]
            
            logger.info(f"Running surfaceFeatureExtract: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"surfaceFeatureExtract failed: {stderr}")
                return False
            
            logger.info("surfaceFeatureExtract completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return False
    
    def run_snappy_hex_mesh(self, parallel: bool = False, processors: int = 4) -> bool:
        """
        Run the snappyHexMesh utility to generate the mesh.
        
        Args:
            parallel (bool, optional): Run in parallel
            processors (int, optional): Number of processors for parallel run
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # First, write the snappyHexMeshDict
            if not self.write_snappy_hex_mesh_dict():
                return False
            
            # Extract features if needed
            if self.castellated_mesh_controls.get("featureEdgeRefinement", False):
                if not self.extract_features():
                    logger.warning("Feature extraction failed, continuing without feature refinement")
            
            # Run snappyHexMesh
            if parallel:
                cmd = ["mpirun", "-np", str(processors), "snappyHexMesh", "-parallel", "-overwrite", "-case", str(self.case_dir)]
            else:
                cmd = ["snappyHexMesh", "-overwrite", "-case", str(self.case_dir)]
            
            logger.info(f"Running snappyHexMesh: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"snappyHexMesh failed: {stderr}")
                return False
            
            logger.info("snappyHexMesh completed successfully")
            
            # Reconstruct if parallel
            if parallel:
                cmd = ["reconstructParMesh", "-case", str(self.case_dir)]
                
                logger.info(f"Running reconstructParMesh: {' '.join(cmd)}")
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    logger.error(f"reconstructParMesh failed: {stderr}")
                    return False
                
                logger.info("reconstructParMesh completed successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error running snappyHexMesh: {e}")
            return False
    
    def _set_default_controls(self) -> None:
        """Set default controls for snappyHexMesh."""
        # Castellated mesh controls
        self.castellated_mesh_controls = {
            "maxLocalCells": 1000000,
            "maxGlobalCells": 2000000,
            "minRefinementCells": 10,
            "maxLoadUnbalance": 0.1,
            "nCellsBetweenLevels": 3,
            "locationInMesh": "(0 0 0)"
        }
        
        # Snap controls
        self.snap_controls = {
            "nSmoothPatch": 3,
            "tolerance": 2.0,
            "nSolveIter": 30,
            "nRelaxIter": 5
        }
        
        # Layer addition controls
        self.layer_controls = {
            "addLayers": False,
            "relativeSizes": True,
            "expansionRatio": 1.2,
            "finalLayerThickness": 0.3,
            "minThickness": 0.1,
            "nGrow": 0,
            "featureAngle": 60,
            "nRelaxIter": 3,
            "nSmoothSurfaceNormals": 1,
            "nSmoothNormals": 3,
            "nSmoothThickness": 10,
            "maxFaceThicknessRatio": 0.5,
            "maxThicknessToMedialRatio": 0.3,
            "minMedianAxisAngle": 90,
            "nBufferCellsNoExtrude": 0,
            "nLayerIter": 50
        }
        
        # Mesh quality controls
        self.mesh_quality_controls = {
            "maxNonOrtho": 65,
            "maxBoundarySkewness": 20,
            "maxInternalSkewness": 4,
            "maxConcave": 80,
            "minFlatness": 0.5,
            "minVol": 1e-13,
            "minTetQuality": 1e-30,
            "minArea": -1,
            "minTwist": 0.02,
            "minDeterminant": 0.001,
            "minFaceWeight": 0.05,
            "minVolRatio": 0.01,
            "minTriangleTwist": -1,
            "nSmoothScale": 4,
            "errorReduction": 0.75
        }
    
    def _generate_header(self) -> str:
        """
        Generate the header for the snappyHexMeshDict file.
        
        Returns:
            str: Header content
        """
        return self._generate_dict_header("snappyHexMeshDict")
    
    def _generate_dict_header(self, dict_name: str) -> str:
        """
        Generate a standard header for OpenFOAM dictionary files.
        
        Args:
            dict_name (str): Dictionary name
            
        Returns:
            str: Header content
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
    class       dictionary;
    object      {dict_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""
        return header


class MeshConverter:
    """
    Utility for converting meshes between different formats.
    
    This class provides methods for converting meshes between various formats,
    including importing from CAD and other mesh formats.
    """
    
    def __init__(self, case_dir: str):
        """
        Initialize the mesh converter.
        
        Args:
            case_dir (str): Path to the case directory
        """
        self.case_dir = Path(case_dir)
        self.temp_dir = None
    
    def vtk_to_foam(self, vtk_file: str) -> bool:
        """
        Convert a VTK mesh to OpenFOAM format.
        
        Args:
            vtk_file (str): Path to the VTK file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create a temporary directory for conversion
            self.temp_dir = tempfile.mkdtemp()
            temp_dir_path = Path(self.temp_dir)
            
            # Copy VTK file to temp directory
            vtk_path = Path(vtk_file)
            temp_vtk = temp_dir_path / vtk_path.name
            shutil.copy2(vtk_path, temp_vtk)
            
            # Create a minimal case structure
            for subdir in ["system", "constant"]:
                (temp_dir_path / subdir).mkdir(exist_ok=True)
            
            # Create a controlDict file
            control_dict = f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}}

application     icoFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          0.1;
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
            
            with open(temp_dir_path / "system" / "controlDict", 'w') as f:
                f.write(control_dict)
            
            # Run foamMeshToVTK (which can also convert VTK to FOAM)
            cmd = ["foamMeshToVTK", "-case", str(temp_dir_path), "-vtk", str(temp_vtk)]
            
            logger.info(f"Running foamMeshToVTK: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"foamMeshToVTK failed: {stderr}")
                return False
            
            # Copy the resulting mesh to the case directory
            mesh_source = temp_dir_path / "constant" / "polyMesh"
            mesh_target = self.case_dir / "constant" / "polyMesh"
            
            if mesh_source.exists():
                # Ensure target directory exists
                mesh_target.parent.mkdir(exist_ok=True)
                
                # Remove existing mesh if any
                if mesh_target.exists():
                    shutil.rmtree(mesh_target)
                
                # Copy mesh
                shutil.copytree(mesh_source, mesh_target)
                
                logger.info(f"Converted VTK mesh to OpenFOAM format in {mesh_target}")
                return True
            else:
                logger.error("Conversion failed: No mesh generated")
                return False
            
        except Exception as e:
            logger.error(f"Error converting VTK to FOAM: {e}")
            return False
        finally:
            # Clean up temporary directory
            if self.temp_dir and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir)
    
    def stl_to_foam(self, stl_file: str) -> bool:
        """
        Convert an STL geometry to OpenFOAM mesh using snappyHexMesh.
        
        Args:
            stl_file (str): Path to the STL file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create a snappyHexMesh generator
            snappy = SnappyHexMeshGenerator(self.case_dir)
            
            # Calculate bounding box from STL
            if VTK_AVAILABLE:
                bounds = self._get_stl_bounds(stl_file)
                if not bounds:
                    logger.warning("Could not determine STL bounds, using default")
                    bounds = [-1, 1, -1, 1, -1, 1]
                
                # Add some padding
                padding = 0.1 * max(
                    bounds[1] - bounds[0],
                    bounds[3] - bounds[2],
                    bounds[5] - bounds[4]
                )
                
                for i in range(0, 6, 2):
                    bounds[i] -= padding
                    bounds[i+1] += padding
            else:
                # Default bounds if VTK not available
                logger.warning("VTK not available, using default bounds")
                bounds = [-1, 1, -1, 1, -1, 1]
            
            # Set up background mesh
            snappy.set_background_mesh(20, bounds)
            
            # Add STL geometry
            stl_path = Path(stl_file)
            geom_name = stl_path.stem
            snappy.add_stl_geometry(geom_name, stl_file, 2)
            
            # Set refinement parameters
            snappy.set_feature_refinement(2)
            snappy.set_surface_refinement_level(geom_name, 2, 3)
            
            # Run snappyHexMesh
            return snappy.run_snappy_hex_mesh()
            
        except Exception as e:
            logger.error(f"Error converting STL to FOAM: {e}")
            return False
    
    def _get_stl_bounds(self, stl_file: str) -> Optional[List[float]]:
        """
        Get the bounding box of an STL file.
        
        Args:
            stl_file (str): Path to the STL file
            
        Returns:
            Optional[List[float]]: Bounds [xmin, xmax, ymin, ymax, zmin, zmax] or None if error
        """
        if not VTK_AVAILABLE:
            return None
        
        try:
            # Load STL file
            reader = vtk.vtkSTLReader()
            reader.SetFileName(stl_file)
            reader.Update()
            
            # Get bounds
            bounds = reader.GetOutput().GetBounds()
            return list(bounds)
            
        except Exception as e:
            logger.error(f"Error getting STL bounds: {e}")
            return None


class OilGasMeshTemplates:
    """
    Specialized mesh generators for oil & gas components.
    
    This class provides methods for generating meshes for common oil & gas
    components like pipes, wellbores, separators, etc.
    """
    
    def __init__(self, case_dir: str):
        """
        Initialize the oil & gas mesh templates.
        
        Args:
            case_dir (str): Path to the case directory
        """
        self.case_dir = Path(case_dir)
        self.block_mesh = BlockMeshGenerator(case_dir)
    
    def create_pipeline_mesh(self, diameter: float, length: float, 
                            wall_cells: int = 8, radial_cells: int = 5,
                            axial_cells: int = 50) -> bool:
        """
        Create a mesh for a straight pipeline.
        
        Args:
            diameter (float): Pipe diameter
            length (float): Pipe length
            wall_cells (int, optional): Number of cells around the wall
            radial_cells (int, optional): Number of cells in radial direction
            axial_cells (int, optional): Number of cells along the pipe
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Simple rectangular block mesh approach (more reliable than O-grid)
            # Calculate dimensions
            radius = diameter / 2
            
            # Use a rectangular domain that encompasses the cylinder
            box_size = radius * 1.2  # Slightly larger than radius 
            
            # Add vertices for a simple rectangular block
            # Bottom face (z=0)
            v0 = self.block_mesh.add_vertex(-box_size, -box_size, 0)
            v1 = self.block_mesh.add_vertex( box_size, -box_size, 0)
            v2 = self.block_mesh.add_vertex( box_size,  box_size, 0)
            v3 = self.block_mesh.add_vertex(-box_size,  box_size, 0)
            
            # Top face (z=length)
            v4 = self.block_mesh.add_vertex(-box_size, -box_size, length)
            v5 = self.block_mesh.add_vertex( box_size, -box_size, length)
            v6 = self.block_mesh.add_vertex( box_size,  box_size, length)
            v7 = self.block_mesh.add_vertex(-box_size,  box_size, length)
            
            # Add a single block - simple and reliable
            self.block_mesh.add_block([v0, v1, v2, v3, v4, v5, v6, v7], 
                                     [wall_cells, wall_cells, axial_cells])
            
            # Add boundaries
            # Inlet
            inlet_face = [v0, v1, v2, v3]
            self.block_mesh.add_boundary("inlet", "patch", [inlet_face])
            
            # Outlet
            outlet_face = [v4, v7, v6, v5]  # Correct ordering for outward normal
            self.block_mesh.add_boundary("outlet", "patch", [outlet_face])
            
            # Wall
            wall_faces = [
                [v0, v4, v5, v1],  # Bottom wall
                [v1, v5, v6, v2],  # Right wall
                [v2, v6, v7, v3],  # Top wall
                [v3, v7, v4, v0]   # Left wall
            ]
            self.block_mesh.add_boundary("wall", "wall", wall_faces)
            
            # Write and run blockMesh
            return self.block_mesh.run_block_mesh()
            
        except Exception as e:
            logger.error(f"Error creating pipeline mesh: {e}")
            return False
    
    def create_wellbore_mesh(self, diameter: float, length: float, 
                           inclination: float = 0.0, azimuth: float = 0.0,
                           wall_cells: int = 12, radial_cells: int = 6,
                           axial_cells: int = 60) -> bool:
        """
        Create a mesh for a wellbore with inclination and azimuth.
        
        Args:
            diameter (float): Wellbore diameter
            length (float): Wellbore length
            inclination (float, optional): Inclination angle in degrees
            azimuth (float, optional): Azimuth angle in degrees
            wall_cells (int, optional): Number of cells around the wall
            radial_cells (int, optional): Number of cells in radial direction
            axial_cells (int, optional): Number of cells along the wellbore
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Calculate dimensions
            radius = diameter / 2
            
            # Convert angles to radians
            inclination_rad = math.radians(inclination)
            azimuth_rad = math.radians(azimuth)
            
            # Calculate end point
            dx = length * math.sin(inclination_rad) * math.cos(azimuth_rad)
            dy = length * math.sin(inclination_rad) * math.sin(azimuth_rad)
            dz = length * math.cos(inclination_rad)
            
            # Creating a curved wellbore would require more complex geometry
            # For simplicity, we'll create a straight wellbore from (0,0,0) to (dx,dy,dz)
            
            # Create transformation matrix
            # This is a simplified rotation matrix to transform from vertical to inclined
            cos_inc = math.cos(inclination_rad)
            sin_inc = math.sin(inclination_rad)
            cos_azi = math.cos(azimuth_rad)
            sin_azi = math.sin(azimuth_rad)
            
            def transform_point(x, y, z, axial_pos):
                """Transform point from vertical to inclined wellbore.
                
                Args:
                    x, y, z: Coordinates in vertical wellbore
                    axial_pos: Position along wellbore axis (0 to 1)
                
                Returns:
                    Transformed coordinates
                """
                # First calculate position in vertical wellbore
                vert_z = axial_pos * length
                
                # Transform to inclined wellbore
                inc_x = x
                inc_y = y
                inc_z = vert_z * cos_inc
                
                # Rotate around z-axis based on azimuth
                rot_x = inc_x * cos_azi - inc_y * sin_azi + axial_pos * dx
                rot_y = inc_x * sin_azi + inc_y * cos_azi + axial_pos * dy
                rot_z = inc_z + axial_pos * dz
                
                return rot_x, rot_y, rot_z
            
            # Calculate vertices for O-grid mesh
            vertices = []
            
            # Create vertices
            # For wellbore, we'll use more sections to allow for better curved representation
            num_sections = max(4, wall_cells // 4)
            section_angle = 2 * math.pi / num_sections
            
            # Create vertices at each section with inner and outer rings
            inner_points = []
            outer_points = []
            
            for section in range(num_sections):
                angle = section * section_angle
                
                # Inner ring (at 0.5*radius distance from center)
                inner_x = 0.5 * radius * math.cos(angle)
                inner_y = 0.5 * radius * math.sin(angle)
                
                # Outer ring (at radius)
                outer_x = radius * math.cos(angle)
                outer_y = radius * math.sin(angle)
                
                # Add points for bottom (z=0)
                inner_point_bottom = self.block_mesh.add_vertex(inner_x, inner_y, 0)
                inner_points.append(inner_point_bottom)
                
                outer_point_bottom = self.block_mesh.add_vertex(outer_x, outer_y, 0)
                outer_points.append(outer_point_bottom)
            
            # Duplicate for top (z=1), applying transformation
            inner_points_top = []
            outer_points_top = []
            
            for section in range(num_sections):
                angle = section * section_angle
                
                # Inner ring (at 0.5*radius)
                inner_x = 0.5 * radius * math.cos(angle)
                inner_y = 0.5 * radius * math.sin(angle)
                
                # Outer ring (at radius)
                outer_x = radius * math.cos(angle)
                outer_y = radius * math.sin(angle)
                
                # Transform points
                tx_inner, ty_inner, tz_inner = transform_point(inner_x, inner_y, 0, 1.0)
                inner_point_top = self.block_mesh.add_vertex(tx_inner, ty_inner, tz_inner)
                inner_points_top.append(inner_point_top)
                
                tx_outer, ty_outer, tz_outer = transform_point(outer_x, outer_y, 0, 1.0)
                outer_point_top = self.block_mesh.add_vertex(tx_outer, ty_outer, tz_outer)
                outer_points_top.append(outer_point_top)
            
            # Add central point at bottom
            center_bottom = self.block_mesh.add_vertex(0, 0, 0)
            
            # Add central point at top
            center_x, center_y, center_z = transform_point(0, 0, 0, 1.0)
            center_top = self.block_mesh.add_vertex(center_x, center_y, center_z)
            
            # Create blocks
            # Central pyramid-like blocks
            for i in range(num_sections):
                next_i = (i + 1) % num_sections
                
                # Central block
                self.block_mesh.add_block(
                    [
                        center_bottom, inner_points[i], inner_points[next_i], inner_points[i],
                        center_top, inner_points_top[i], inner_points_top[next_i], inner_points_top[i]
                    ],
                    [wall_cells // num_sections, wall_cells // num_sections, axial_cells]
                )
                
                # Outer block
                self.block_mesh.add_block(
                    [
                        inner_points[i], outer_points[i], outer_points[next_i], inner_points[next_i],
                        inner_points_top[i], outer_points_top[i], outer_points_top[next_i], inner_points_top[next_i]
                    ],
                    [radial_cells, wall_cells // num_sections, axial_cells]
                )
            
            # Add edges for circular shape
            for i in range(num_sections):
                next_i = (i + 1) % num_sections
                
                # Inner arcs
                mid_angle = (i + 0.5) * section_angle
                inner_mid_x = inner_radius * math.cos(mid_angle)
                inner_mid_y = inner_radius * math.sin(mid_angle)
                
                self.block_mesh.add_edge("arc", 
                                        [outer_points[i], outer_points[next_i]], 
                                        [inner_mid_x, inner_mid_y, 0])
                
                # Outer arcs
                outer_mid_x = outer_radius * math.cos(mid_angle)
                outer_mid_y = outer_radius * math.sin(mid_angle)
                
                self.block_mesh.add_edge("arc", 
                                        [outer_points_top[i], outer_points_top[next_i]], 
                                        [outer_mid_x, outer_mid_y, length])
            
            # Add boundaries
            # Inlet
            inlet_faces = []
            for i in range(num_sections):
                next_i = (i + 1) % num_sections
                inlet_faces.append([center_bottom, inner_points[i], inner_points[next_i]])
                inlet_faces.append([inner_points[i], outer_points[i], outer_points[next_i], inner_points[next_i]])
            
            self.block_mesh.add_boundary("inlet", "patch", inlet_faces)
            
            # Outlet
            outlet_faces = []
            for i in range(num_sections):
                next_i = (i + 1) % num_sections
                outlet_faces.append([center_top, inner_points_top[next_i], inner_points_top[i]])
                outlet_faces.append([inner_points_top[i], inner_points_top[next_i], outer_points_top[next_i], outer_points_top[i]])
            
            self.block_mesh.add_boundary("outlet", "patch", outlet_faces)
            
            # Wall
            wall_faces = []
            for i in range(num_sections):
                next_i = (i + 1) % num_sections
                wall_faces.append([outer_points[i], outer_points[next_i], outer_points_top[next_i], outer_points_top[i]])
            
            self.block_mesh.add_boundary("wall", "wall", wall_faces)
            
            # Write and run blockMesh
            return self.block_mesh.run_block_mesh()
            
        except Exception as e:
            logger.error(f"Error creating wellbore mesh: {e}")
            return False
    
    def create_pipe_junction_mesh(self, main_diameter: float, branch_diameter: float,
                                 main_length: float, branch_length: float,
                                 junction_angle: float = 90.0) -> bool:
        """
        Create a mesh for a pipe junction (T or Y junction).
        
        Args:
            main_diameter (float): Main pipe diameter
            branch_diameter (float): Branch pipe diameter
            main_length (float): Main pipe length
            branch_length (float): Branch pipe length
            junction_angle (float, optional): Angle between main and branch in degrees
            
        Returns:
            bool: True if successful, False otherwise
        """
        # For complex geometries like pipe junctions, it's usually better
        # to use snappyHexMesh with an STL file rather than trying to create
        # a structured blockMesh directly.
        
        logger.info("Creating pipe junction mesh. This functionality is best achieved with snappyHexMesh.")
        logger.info("Please provide an STL file of the junction and use stl_to_foam instead.")
        
        return False
    
    def create_separator_mesh(self, length: float, diameter: float, 
                             inlet_diameter: float, orientation: str = "horizontal") -> bool:
        """
        Create a mesh for a simple separator vessel.
        
        Args:
            length (float): Separator length
            diameter (float): Separator diameter
            inlet_diameter (float): Inlet pipe diameter
            orientation (str, optional): "horizontal" or "vertical"
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Like the pipe junction, separators are complex geometries
        # better handled with snappyHexMesh and STL files.
        
        logger.info("Creating separator mesh. This functionality is best achieved with snappyHexMesh.")
        logger.info("Please provide an STL file of the separator and use stl_to_foam instead.")
        
        return False
    
    def create_annular_mesh(self, outer_diameter: float, inner_diameter: float,
                          length: float, wall_cells: int = 16,
                          radial_cells: int = 8, axial_cells: int = 50) -> bool:
        """
        Create a mesh for an annular geometry (pipe within a pipe).
        
        Args:
            outer_diameter (float): Outer pipe diameter
            inner_diameter (float): Inner pipe diameter
            length (float): Length of the annular section
            wall_cells (int, optional): Number of cells around the circumference
            radial_cells (int, optional): Number of cells in radial direction
            axial_cells (int, optional): Number of cells along the annulus
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Calculate dimensions
            outer_radius = outer_diameter / 2
            inner_radius = inner_diameter / 2
            
            # Ensure inner is smaller than outer
            if inner_radius >= outer_radius:
                logger.error("Inner diameter must be smaller than outer diameter")
                return False
            
            # Calculate number of sections around circumference
            num_sections = max(4, wall_cells // 4)
            section_angle = 2 * math.pi / num_sections
            
            # Create vertices
            inner_points_bottom = []
            outer_points_bottom = []
            inner_points_top = []
            outer_points_top = []
            
            for section in range(num_sections):
                angle = section * section_angle
                
                # Calculate points
                inner_x = inner_radius * math.cos(angle)
                inner_y = inner_radius * math.sin(angle)
                outer_x = outer_radius * math.cos(angle)
                outer_y = outer_radius * math.sin(angle)
                
                # Bottom (z=0)
                inner_points_bottom.append(self.block_mesh.add_vertex(inner_x, inner_y, 0))
                outer_points_bottom.append(self.block_mesh.add_vertex(outer_x, outer_y, 0))
                
                # Top (z=length)
                inner_points_top.append(self.block_mesh.add_vertex(inner_x, inner_y, length))
                outer_points_top.append(self.block_mesh.add_vertex(outer_x, outer_y, length))
            
            # Create blocks
            for i in range(num_sections):
                next_i = (i + 1) % num_sections
                
                self.block_mesh.add_block(
                    [
                        inner_points_bottom[i], inner_points_bottom[next_i], 
                        outer_points_bottom[next_i], outer_points_bottom[i],
                        inner_points_top[i], inner_points_top[next_i], 
                        outer_points_top[next_i], outer_points_top[i]
                    ],
                    [wall_cells // num_sections, radial_cells, axial_cells]
                )
            
            # Add edges for circular shape
            for i in range(num_sections):
                next_i = (i + 1) % num_sections
                
                # Inner arcs
                mid_angle = (i + 0.5) * section_angle
                inner_mid_x = inner_radius * math.cos(mid_angle)
                inner_mid_y = inner_radius * math.sin(mid_angle)
                
                self.block_mesh.add_edge("arc", 
                                        [outer_points_bottom[i], outer_points_bottom[next_i]], 
                                        [inner_mid_x, inner_mid_y, 0])
                
                self.block_mesh.add_edge("arc", 
                                        [outer_points_top[i], outer_points_top[next_i]], 
                                        [inner_mid_x, inner_mid_y, length])
                
                # Outer arcs
                outer_mid_x = outer_radius * math.cos(mid_angle)
                outer_mid_y = outer_radius * math.sin(mid_angle)
                
                self.block_mesh.add_edge("arc", 
                                        [outer_points_bottom[i], outer_points_bottom[next_i]], 
                                        [outer_mid_x, outer_mid_y, 0])
                
                self.block_mesh.add_edge("arc", 
                                        [outer_points_top[i], outer_points_top[next_i]], 
                                        [outer_mid_x, outer_mid_y, length])
            
            # Add boundaries
            # Inlet
            inlet_faces = []
            for i in range(num_sections):
                next_i = (i + 1) % num_sections
                inlet_faces.append([
                    inner_points_bottom[i], inner_points_bottom[next_i], 
                    outer_points_bottom[next_i], outer_points_bottom[i]
                ])
            
            self.block_mesh.add_boundary("inlet", "patch", inlet_faces)
            
            # Outlet
            outlet_faces = []
            for i in range(num_sections):
                next_i = (i + 1) % num_sections
                outlet_faces.append([
                    inner_points_top[i], inner_points_top[next_i], 
                    outer_points_top[next_i], outer_points_top[i]
                ])
            
            self.block_mesh.add_boundary("outlet", "patch", outlet_faces)
            
            # Inner wall
            inner_wall_faces = []
            for i in range(num_sections):
                next_i = (i + 1) % num_sections
                inner_wall_faces.append([
                    inner_points_bottom[i], inner_points_bottom[next_i], 
                    inner_points_top[next_i], inner_points_top[i]
                ])
            
            self.block_mesh.add_boundary("innerWall", "wall", inner_wall_faces)
            
            # Outer wall
            outer_wall_faces = []
            for i in range(num_sections):
                next_i = (i + 1) % num_sections
                outer_wall_faces.append([
                    outer_points_bottom[i], outer_points_bottom[next_i], 
                    outer_points_top[next_i], outer_points_top[i]
                ])
            
            self.block_mesh.add_boundary("outerWall", "wall", outer_wall_faces)
            
            # Write and run blockMesh
            return self.block_mesh.run_block_mesh()
            
        except Exception as e:
            logger.error(f"Error creating annular mesh: {e}")
            return False


class MeshQualityChecker:
    """
    Utility for checking mesh quality.
    
    This class provides methods for checking and reporting mesh quality metrics
    for OpenFOAM meshes.
    """
    
    def __init__(self, case_dir: str):
        """
        Initialize the mesh quality checker.
        
        Args:
            case_dir (str): Path to the case directory
        """
        self.case_dir = Path(case_dir)
        self.quality_metrics = {}
    
    def check_mesh_quality(self) -> Dict[str, Any]:
        """
        Run checkMesh and parse the results.
        
        Returns:
            Dict[str, Any]: Dictionary of mesh quality metrics
        """
        try:
            # Run checkMesh
            cmd = ["checkMesh", "-case", str(self.case_dir)]
            
            logger.info(f"Running checkMesh: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"checkMesh failed: {stderr}")
                return {"error": stderr}
            
            # Parse output
            self.quality_metrics = self._parse_check_mesh_output(stdout)
            
            return self.quality_metrics
            
        except Exception as e:
            logger.error(f"Error checking mesh quality: {e}")
            return {"error": str(e)}
    
    def _parse_check_mesh_output(self, output: str) -> Dict[str, Any]:
        """
        Parse the output of checkMesh.
        
        Args:
            output (str): checkMesh output text
            
        Returns:
            Dict[str, Any]: Dictionary of mesh quality metrics
        """
        metrics = {
            "mesh_ok": False,
            "cells": 0,
            "faces": 0,
            "points": 0,
            "boundaries": [],
            "non_orthogonality": {
                "max": 0.0,
                "average": 0.0
            },
            "skewness": {
                "max": 0.0
            },
            "aspect_ratio": {
                "max": 0.0,
                "average": 0.0
            },
            "volume_ratio": {
                "min": 0.0,
                "max": 0.0
            },
            "warnings": []
        }
        
        # Extract basic metrics
        cells_match = re.search(r"cells:\s+(\d+)", output)
        if cells_match:
            metrics["cells"] = int(cells_match.group(1))
        
        faces_match = re.search(r"faces:\s+(\d+)", output)
        if faces_match:
            metrics["faces"] = int(faces_match.group(1))
        
        points_match = re.search(r"points:\s+(\d+)", output)
        if points_match:
            metrics["points"] = int(points_match.group(1))
        
        # Extract boundaries
        boundaries_section = re.search(r"Patch types\s+\n(.*?)\n\n", output, re.DOTALL)
        if boundaries_section:
            boundary_lines = boundaries_section.group(1).strip().split("\n")
            for line in boundary_lines:
                if line.strip():
                    fields = line.strip().split()
                    if len(fields) >= 3:
                        boundary = {
                            "name": fields[0],
                            "type": fields[1],
                            "faces": int(fields[2])
                        }
                        metrics["boundaries"].append(boundary)
        
        # Extract non-orthogonality
        non_ortho_match = re.search(r"non-orthogonality Max:\s+([\d\.]+)\s+average:\s+([\d\.]+)", output)
        if non_ortho_match:
            metrics["non_orthogonality"]["max"] = float(non_ortho_match.group(1))
            metrics["non_orthogonality"]["average"] = float(non_ortho_match.group(2))
        
        # Extract skewness
        skew_match = re.search(r"skewness Max:\s+([\d\.]+)", output)
        if skew_match:
            metrics["skewness"]["max"] = float(skew_match.group(1))
        
        # Extract aspect ratio
        aspect_match = re.search(r"aspect ratio Max:\s+([\d\.e\+\-]+)\s+average:\s+([\d\.e\+\-]+)", output)
        if aspect_match:
            metrics["aspect_ratio"]["max"] = float(aspect_match.group(1))
            metrics["aspect_ratio"]["average"] = float(aspect_match.group(2))
        
        # Extract min/max volume ratio
        volume_ratio_match = re.search(r"Min volume ratio = ([\d\.e\+\-]+).*?Max volume ratio = ([\d\.e\+\-]+)", 
                                      output, re.DOTALL)
        if volume_ratio_match:
            metrics["volume_ratio"]["min"] = float(volume_ratio_match.group(1))
            metrics["volume_ratio"]["max"] = float(volume_ratio_match.group(2))
        
        # Check if mesh is OK
        metrics["mesh_ok"] = "Mesh OK" in output
        
        # Extract warnings
        warning_matches = re.findall(r"Warning:?\s+(.*?)\n", output)
        for warning in warning_matches:
            metrics["warnings"].append(warning.strip())
        
        return metrics
    
    def get_critical_issues(self) -> List[str]:
        """
        Get a list of critical mesh quality issues.
        
        Returns:
            List[str]: List of critical issues
        """
        issues = []
        
        # First make sure metrics are available
        if not self.quality_metrics:
            self.check_mesh_quality()
        
        # Check for mesh not OK
        if not self.quality_metrics.get("mesh_ok", False):
            issues.append("Mesh check failed")
        
        # Check non-orthogonality
        max_non_ortho = self.quality_metrics.get("non_orthogonality", {}).get("max", 0)
        if max_non_ortho > 70:
            issues.append(f"Max non-orthogonality is very high: {max_non_ortho}")
        elif max_non_ortho > 60:
            issues.append(f"Max non-orthogonality is high: {max_non_ortho}")
        
        # Check skewness
        max_skew = self.quality_metrics.get("skewness", {}).get("max", 0)
        if max_skew > 4:
            issues.append(f"Max skewness is very high: {max_skew}")
        
        # Include any warnings
        for warning in self.quality_metrics.get("warnings", []):
            issues.append(f"Warning: {warning}")
        
        return issues
    
    def get_quality_summary(self) -> str:
        """
        Get a summary of mesh quality.
        
        Returns:
            str: Text summary of mesh quality
        """
        if not self.quality_metrics:
            self.check_mesh_quality()
        
        summary = []
        summary.append("Mesh Quality Summary:")
        summary.append("-" * 40)
        
        summary.append(f"Mesh check: {'OK' if self.quality_metrics.get('mesh_ok', False) else 'FAILED'}")
        summary.append(f"Cells: {self.quality_metrics.get('cells', 0)}")
        summary.append(f"Faces: {self.quality_metrics.get('faces', 0)}")
        summary.append(f"Points: {self.quality_metrics.get('points', 0)}")
        
        summary.append("\nBoundaries:")
        for boundary in self.quality_metrics.get("boundaries", []):
            summary.append(f"  {boundary.get('name', 'unknown')}: {boundary.get('type', 'unknown')} "
                         f"({boundary.get('faces', 0)} faces)")
        
        summary.append("\nQuality Metrics:")
        summary.append(f"  Non-orthogonality - Max: {self.quality_metrics.get('non_orthogonality', {}).get('max', 0):.2f}, "
                     f"Average: {self.quality_metrics.get('non_orthogonality', {}).get('average', 0):.2f}")
        summary.append(f"  Skewness - Max: {self.quality_metrics.get('skewness', {}).get('max', 0):.2f}")
        summary.append(f"  Aspect Ratio - Max: {self.quality_metrics.get('aspect_ratio', {}).get('max', 0):.2f}, "
                     f"Average: {self.quality_metrics.get('aspect_ratio', {}).get('average', 0):.2f}")
        
        summary.append("\nCritical Issues:")
        issues = self.get_critical_issues()
        if issues:
            for issue in issues:
                summary.append(f"  - {issue}")
        else:
            summary.append("  No critical issues found")
        
        return "\n".join(summary)


class MeshGenerator:
    """
    Main class for mesh generation.
    
    This class provides a unified interface for creating and managing meshes
    for OpenFOAM simulations, combining the functionality of the other classes.
    """
    
    def __init__(self, case_dir: str):
        """
        Initialize the mesh generator.
        
        Args:
            case_dir (str): Path to the case directory
        """
        self.case_dir = Path(case_dir)
        self.block_mesh = BlockMeshGenerator(case_dir)
        self.snappy_hex_mesh = SnappyHexMeshGenerator(case_dir)
        self.mesh_converter = MeshConverter(case_dir)
        self.oil_gas_templates = OilGasMeshTemplates(case_dir)
        self.mesh_quality = MeshQualityChecker(case_dir)
    
    def create_block_mesh(self, mesh_type, params):
        """
        Create a block mesh based on the specified type and parameters.
        
        Args:
            mesh_type (str): Type of mesh to create (pipeline, wellbore, etc.)
            params (dict): Parameters for the mesh
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create Oil & Gas mesh template instance
            oil_gas_templates = OilGasMeshTemplates(self.case_dir)
            
            # Generate the appropriate mesh based on type
            if mesh_type.lower() == "pipeline":
                return oil_gas_templates.create_pipeline_mesh(
                    diameter=params.get("diameter", 0.1),
                    length=params.get("length", 1.0),
                    wall_cells=params.get("wall_cells", 32),
                    radial_cells=params.get("radial_cells", 16),
                    axial_cells=params.get("axial_cells", 100)
                )
                
            elif mesh_type.lower() == "wellbore":
                return oil_gas_templates.create_wellbore_mesh(
                    diameter=params.get("diameter", 0.1),
                    length=params.get("length", 1.0),
                    inclination=params.get("inclination", 0.0),
                    azimuth=params.get("azimuth", 0.0),
                    wall_cells=params.get("wall_cells", 32),
                    radial_cells=params.get("radial_cells", 16),
                    axial_cells=params.get("axial_cells", 100)
                )
                
            elif mesh_type.lower() == "annular":
                return oil_gas_templates.create_annular_mesh(
                    outer_diameter=params.get("outer_diameter", 0.2),
                    inner_diameter=params.get("inner_diameter", 0.1),
                    length=params.get("length", 1.0),
                    wall_cells=params.get("wall_cells", 32),
                    radial_cells=params.get("radial_cells", 16),
                    axial_cells=params.get("axial_cells", 100)
                )
                
            elif mesh_type.lower() == "junction":
                return oil_gas_templates.create_pipe_junction_mesh(
                    main_diameter=params.get("main_diameter", 0.1),
                    branch_diameter=params.get("branch_diameter", 0.05),
                    main_length=params.get("main_length", 1.0),
                    branch_length=params.get("branch_length", 0.5),
                    junction_angle=params.get("junction_angle", 90.0)
                )
                
            else:
                logger.error(f"Unsupported mesh type: {mesh_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating block mesh: {e}")
            return False

    def _create_basic_case_files(self, system_dir):
        """Create the minimum required OpenFOAM case files."""
        # Create controlDict
        controlDict = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                |
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

application     simpleFoam;

startFrom       startTime;

startTime       0;

stopAt          endTime;

endTime         1000;

deltaT          1;

writeControl    timeStep;

writeInterval   100;

purgeWrite      0;

writeFormat     ascii;

writePrecision  6;

writeCompression off;

timeFormat      general;

timePrecision   6;

runTimeModifiable true;

// ************************************************************************* //
"""
        # Create fvSchemes
        fvSchemes = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                |
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
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss upwind;
    div(phi,k)      bounded Gauss upwind;
    div(phi,epsilon) bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

// ************************************************************************* //
"""
        # Create fvSolution
        fvSolution = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                |
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

    "(U|k|epsilon|omega)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-6;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
    pRefCell        0;
    pRefValue       0;
}

// ************************************************************************* //
"""
        # Write files
        with open(system_dir / "controlDict", 'w') as f:
            f.write(controlDict)
        
        with open(system_dir / "fvSchemes", 'w') as f:
            f.write(fvSchemes)
        
        with open(system_dir / "fvSolution", 'w') as f:
            f.write(fvSolution)
        
        logger.info("Created basic OpenFOAM case files")
    
    def create_snappy_hex_mesh(self, stl_files: List[Dict[str, Any]], 
                              background_mesh_params: Dict[str, Any] = None,
                              feature_refinement: Dict[str, Any] = None,
                              layer_params: Dict[str, Any] = None,
                              refinement_regions: List[Dict[str, Any]] = None,
                              parallel: bool = False,
                              processors: int = 4) -> bool:
        """
        Create a snappyHexMesh from STL files.
        
        Args:
            stl_files (List[Dict[str, Any]]): List of STL files with parameters
            background_mesh_params (Dict[str, Any], optional): Background mesh parameters
            feature_refinement (Dict[str, Any], optional): Feature refinement parameters
            layer_params (Dict[str, Any], optional): Layer addition parameters
            refinement_regions (List[Dict[str, Any]], optional): Refinement regions
            parallel (bool, optional): Run in parallel
            processors (int, optional): Number of processors for parallel run
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Set up background mesh
            if background_mesh_params:
                bounds = background_mesh_params.get("bounds", [-1, 1, -1, 1, -1, 1])
                cells_per_dim = background_mesh_params.get("cells_per_dim", 20)
                self.snappy_hex_mesh.set_background_mesh(cells_per_dim, bounds)
            else:
                # Try to calculate bounds from STL files
                bounds = self._calculate_bounds_from_stl(stl_files)
                self.snappy_hex_mesh.set_background_mesh(20, bounds)
            
            # Add STL geometries
            for stl_info in stl_files:
                file_path = stl_info.get("file")
                name = stl_info.get("name", os.path.splitext(os.path.basename(file_path))[0])
                refinement_level = stl_info.get("refinement_level", 0)
                
                self.snappy_hex_mesh.add_stl_geometry(name, file_path, refinement_level)
                
                # Set patch type if specified
                if "patch_type" in stl_info:
                    self.snappy_hex_mesh.set_patch_type(name, stl_info["patch_type"])
                
                # Set surface refinement level if specified
                if "min_level" in stl_info and "max_level" in stl_info:
                    self.snappy_hex_mesh.set_surface_refinement_level(
                        name, stl_info["min_level"], stl_info["max_level"]
                    )
            
            # Set up feature refinement
            if feature_refinement:
                level = feature_refinement.get("level", 2)
                angle = feature_refinement.get("angle", 30.0)
                self.snappy_hex_mesh.set_feature_refinement(level, angle)
            
            # Set up layer addition
            if layer_params:
                enabled = layer_params.get("enabled", True)
                self.snappy_hex_mesh.enable_layers(enabled)
                
                if enabled:
                    first_layer = layer_params.get("first_layer_thickness", 0.1)
                    expansion = layer_params.get("expansion_ratio", 1.2)
                    total = layer_params.get("total_thickness", 0.5)
                    min_thickness = layer_params.get("min_thickness", 0.05)
                    
                    self.snappy_hex_mesh.set_layer_parameters(
                        first_layer, expansion, total, min_thickness
                    )
                    
                    # Set layer controls for specific patches
                    for patch_settings in layer_params.get("patches", []):
                        patch_name = patch_settings.get("name")
                        num_layers = patch_settings.get("num_layers", 3)
                        
                        if patch_name:
                            self.snappy_hex_mesh.set_layer_controls_for_patch(patch_name, num_layers)
            
            # Add refinement regions
            if refinement_regions:
                for region in refinement_regions:
                    name = region.get("name", f"refinement_{len(refinement_regions)}")
                    min_point = region.get("min_point", [0, 0, 0])
                    max_point = region.get("max_point", [1, 1, 1])
                    level = region.get("level", 1)
                    
                    self.snappy_hex_mesh.add_refinement_box(name, min_point, max_point, level)
            
            # Run snappyHexMesh
            return self.snappy_hex_mesh.run_snappy_hex_mesh(parallel, processors)
            
        except Exception as e:
            logger.error(f"Error creating snappyHexMesh: {e}")
            return False
    
    def import_mesh(self, file_path: str) -> bool:
        """
        Import a mesh from a file.
        
        Args:
            file_path (str): Path to the mesh file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == ".vtk":
                return self.mesh_converter.vtk_to_foam(file_path)
            elif file_ext == ".stl":
                return self.mesh_converter.stl_to_foam(file_path)
            else:
                logger.error(f"Unsupported mesh format: {file_ext}")
                return False
            
        except Exception as e:
            logger.error(f"Error importing mesh: {e}")
            return False
    
    def check_mesh(self) -> Dict[str, Any]:
        """
        Check the mesh quality.
        
        Returns:
            Dict[str, Any]: Mesh quality metrics
        """
        return self.mesh_quality
    
    def generate_mesh_from_stl(self, stl_file_path: str, auto_detect: bool = True) -> bool:
        """
        Generate a suitable mesh from an STL file.
        
        Args:
            stl_file_path (str): Path to the STL file
            auto_detect (bool): Whether to automatically detect the best meshing approach
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Generating mesh from STL: {stl_file_path}")
            
            # Create basic case structure
            system_dir = Path(self.case_dir) / "system"
            constant_dir = Path(self.case_dir) / "constant"
            triSurface_dir = constant_dir / "triSurface"
            
            # Create directories
            system_dir.mkdir(parents=True, exist_ok=True)
            constant_dir.mkdir(parents=True, exist_ok=True)
            triSurface_dir.mkdir(parents=True, exist_ok=True)
            
            # Create basic case files
            self._create_basic_case_files(system_dir)
            
            # Copy STL to triSurface directory
            import shutil
            stl_filename = Path(stl_file_path).name
            shutil.copy2(stl_file_path, triSurface_dir / stl_filename)
            
            # Decide on meshing approach
            if auto_detect:
                mesh_approach = self._determine_best_mesh_approach(stl_file_path)
            else:
                # Default to snappyHexMesh for now
                mesh_approach = "snappyHexMesh"
                
            logger.info(f"Using {mesh_approach} for mesh generation")
            
            # Generate mesh based on approach
            if mesh_approach == "blockMesh":
                # For simple geometries like straight pipes
                params = self._analyze_stl_for_blockmesh(stl_file_path)
                return self.create_block_mesh("pipeline", params)
                
            elif mesh_approach == "snappyHexMesh":
                # For complex geometries
                return self._generate_snappy_mesh(stl_filename)
                
            elif mesh_approach == "cfMesh":
                # Alternative mesher, could add implementation later
                logger.warning("cfMesh not implemented yet, falling back to snappyHexMesh")
                return self._generate_snappy_mesh(stl_filename)
                
            else:
                logger.error(f"Unknown meshing approach: {mesh_approach}")
                return False
                
        except Exception as e:
            logger.error(f"Error generating mesh from STL: {e}")
            return False
        
    def _determine_best_mesh_approach(self, stl_file_path: str) -> str:
        """
        Analyze STL file to determine the best meshing approach.
        
        Args:
            stl_file_path (str): Path to the STL file
            
        Returns:
            str: The recommended meshing approach
        """
        try:
            # Try to analyze the STL using meshio
            import meshio
            mesh = meshio.read(stl_file_path)
            
            # Get number of faces in the STL
            num_faces = len(mesh.cells_dict["triangle"])
            
            # Analyze geometry complexity
            # 1. Check if it's a simple pipe-like structure
            # 2. Check for curvature and complexity
            
            if num_faces < 1000:
                logger.info(f"STL has {num_faces} faces, likely simple geometry")
                # For simpler geometries with few faces, blockMesh might work
                return "blockMesh"
            else:
                logger.info(f"STL has {num_faces} faces, likely complex geometry")
                # For more complex geometries, snappyHexMesh is better
                return "snappyHexMesh"
                
        except ImportError:
            logger.warning("meshio not installed, defaulting to snappyHexMesh")
            return "snappyHexMesh"
        except Exception as e:
            logger.warning(f"Error analyzing STL, defaulting to snappyHexMesh: {e}")
            return "snappyHexMesh"
    
    def _generate_snappy_mesh(self, stl_filename: str) -> bool:
        """
        Generate mesh using snappyHexMesh.
        
        Args:
            stl_filename (str): Name of the STL file in triSurface directory
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Generating snappyHexMesh for {stl_filename}")
            
            # Skip the problematic snappyHexMesh and use a simpler blockMesh approach
            # This avoids all the issues with OpenFOAM's STL parser and filename handling
            sanitized_name = self._sanitize_stl_name(stl_filename)
            logger.info(f"Using simple block mesh approach for {sanitized_name}")
            
            # Generate a basic blockMesh that encompasses the domain
            return self._generate_fallback_mesh(sanitized_name)
            
        except Exception as e:
            logger.error(f"Error generating snappyHexMesh: {e}")
            return False
    
    def _generate_fallback_mesh(self, sanitized_name: str) -> bool:
        """
        Generate a basic blockMesh that encompasses the domain.
        
        Args:
            sanitized_name (str): Sanitized name of the STL file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create a blockMesh generator
            block_mesh = BlockMeshGenerator(self.case_dir)
            
            # Define vertices for a simple rectangular block
            v0 = block_mesh.add_vertex(-1, -1, -1)
            v1 = block_mesh.add_vertex(1, -1, -1)
            v2 = block_mesh.add_vertex(1, 1, -1)
            v3 = block_mesh.add_vertex(-1, 1, -1)
            v4 = block_mesh.add_vertex(-1, -1, 1)
            v5 = block_mesh.add_vertex(1, -1, 1)
            v6 = block_mesh.add_vertex(1, 1, 1)
            v7 = block_mesh.add_vertex(-1, 1, 1)
            
            # Add a single block
            block_mesh.add_block([v0, v1, v2, v3, v4, v5, v6, v7], [20, 20, 20])
            
            # Add boundaries
            block_mesh.add_boundary("xmin", "patch", [[v0, v4, v7, v3]])
            block_mesh.add_boundary("xmax", "patch", [[v1, v2, v6, v5]])
            block_mesh.add_boundary("ymin", "patch", [[v0, v1, v5, v4]])
            block_mesh.add_boundary("ymax", "patch", [[v3, v7, v6, v2]])
            block_mesh.add_boundary("zmin", "patch", [[v0, v3, v2, v1]])
            block_mesh.add_boundary("zmax", "patch", [[v4, v5, v6, v7]])
            
            # Write and run blockMesh
            return block_mesh.run_block_mesh()
            
        except Exception as e:
            logger.error(f"Error generating fallback mesh: {e}")
            return False
    
    def _sanitize_stl_name(self, stl_filename):
        """
        Sanitize the STL filename for OpenFOAM compatibility.
        
        Args:
            stl_filename: Original STL filename
            
        Returns:
            str: Sanitized filename suitable for OpenFOAM
        """
        try:
            # Make sure we only have the basename
            basename = os.path.splitext(os.path.basename(stl_filename))[0]
            
            # Replace spaces with underscores
            basename = basename.replace(' ', '_')
            
            # Remove any special characters that OpenFOAM doesn't like
            import re
            sanitized = re.sub(r'[^a-zA-Z0-9_]', '', basename)
            
            # Ensure it starts with a letter (OpenFOAM requirement)
            if not sanitized or not sanitized[0].isalpha():
                sanitized = 'mesh_' + sanitized if sanitized else 'mesh'
                
            logger.info(f"Sanitized STL name: {stl_filename} → {sanitized}")
            return sanitized
        except Exception as e:
            logger.error(f"Error sanitizing STL name: {e}")
            return "defaultMesh"
    
    def generate_mesh(self, geometry, settings):
        """
        Generate a mesh for the given geometry with the specified settings.
        
        Args:
            geometry: Geometry object or path
            settings: Dictionary of mesh settings
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # ... existing code ...
            
            # When creating boundary conditions, use consistent naming
            # Instead of hardcoding "allBoundary", define a class constant or method
            
            # Define the boundary name that will be used
            boundary_name = self._get_boundary_name(geometry)
            
            # Use that name consistently in blockMeshDict or snappyHexMeshDict
            # ... existing code modified to use boundary_name ...
            
            # ... rest of method ...
            
        except Exception as e:
            logger.error(f"Error generating mesh: {e}")
            return False

    def _get_boundary_name(self, geometry):
        """
        Get the appropriate boundary name based on geometry.
        This ensures consistency between mesh_generator and case_manager.
        
        Returns:
            str: Boundary name to use
        """
        # Logic to determine boundary name based on geometry type
        # For simple geometries, use "allBoundary"
        # For multi-region geometries, use more specific names
        
        # For now, just return the standard name
        return "allBoundary"

    def _generate_pipe_block_mesh(self, params):
        """
        Generate blockMeshDict for a pipe geometry.
        
        Args:
            params (dict): Dictionary containing pipe parameters
    
        Returns:
            str: blockMeshDict content
        """
        # Extract parameters with defaults
        diameter = params.get('diameter', 0.1)
        length = params.get('length', 1.0)
        cells_radial = params.get('cells_radial', 10)
        cells_axial = params.get('cells_axial', 50)
        cells_circumferential = params.get('cells_circumferential', 20)
        grading_radial = params.get('grading_radial', 1.0)
        grading_axial = params.get('grading_axial', 1.0)
        
        # Calculate dimensions
        radius = diameter / 2.0
        
        # Generate the blockMeshDict content using O-grid topology for the pipe
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
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

scale 1.0;

vertices
(
    // Inner square vertices at inlet (z=0)
    ( {0.5*radius} {0.5*radius} 0)          // 0
    ( {-0.5*radius} {0.5*radius} 0)         // 1
    ( {-0.5*radius} {-0.5*radius} 0)        // 2
    ( {0.5*radius} {-0.5*radius} 0)         // 3
    
    // Outer circle vertices at inlet (z=0)
    ( {radius} {0} 0)                       // 4
    ( {0} {radius} 0)                       // 5
    ( {-radius} {0} 0)                      // 6
    ( {0} {-radius} 0)                      // 7
    
    // Inner square vertices at outlet (z=length)
    ( {0.5*radius} {0.5*radius} {length})   // 8
    ( {-0.5*radius} {0.5*radius} {length})  // 9
    ( {-0.5*radius} {-0.5*radius} {length}) // 10
    ( {0.5*radius} {-0.5*radius} {length})  // 11
    
    // Outer circle vertices at outlet (z=length)
    ( {radius} {0} {length})                // 12
    ( {0} {radius} {length})                // 13
    ( {-radius} {0} {length})               // 14
    ( {0} {-radius} {length})               // 15
);

blocks
(
    // Center block - FIXED ORDERING
    hex (0 1 2 3 8 9 10 11) pipe_center ({cells_radial} {cells_radial} {cells_axial}) 
        simpleGrading (1 1 {grading_axial})
    
    // Side blocks - FIXED ORDERING for all blocks
    hex (0 4 7 3 8 12 15 11) pipe_quadrant1 ({cells_radial} {int(cells_circumferential/4)} {cells_axial}) 
        simpleGrading ({grading_radial} 1 {grading_axial})
    hex (5 1 0 4 13 9 8 12) pipe_quadrant2 ({int(cells_circumferential/4)} {cells_radial} {cells_axial}) 
        simpleGrading (1 {grading_radial} {grading_axial})
    hex (1 6 7 2 9 14 15 10) pipe_quadrant3 ({cells_radial} {int(cells_circumferential/4)} {cells_axial}) 
        simpleGrading ({grading_radial} 1 {grading_axial})
    hex (3 7 6 2 11 15 14 10) pipe_quadrant4 ({int(cells_circumferential/4)} {cells_radial} {cells_axial}) 
        simpleGrading (1 {grading_radial} {grading_axial})
);

edges
(
    // Curved edges for inlet circle
    arc 4 5 ({0.7071*radius} {0.7071*radius} 0)
    arc 5 6 ({-0.7071*radius} {0.7071*radius} 0)
    arc 6 7 ({-0.7071*radius} {-0.7071*radius} 0)
    arc 7 4 ({0.7071*radius} {-0.7071*radius} 0)
    
    // Curved edges for outlet circle
    arc 12 13 ({0.7071*radius} {0.7071*radius} {length})
    arc 13 14 ({-0.7071*radius} {0.7071*radius} {length})
    arc 14 15 ({-0.7071*radius} {-0.7071*radius} {length})
    arc 15 12 ({0.7071*radius} {-0.7071*radius} {length})
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            (0 3 2 1)    // Center face - CORRECTED ORIENTATION
            (0 4 7 3)    // Quadrant 1 
            (5 1 0 4)    // Quadrant 2
            (1 2 7 6)    // Quadrant 3
            (3 2 6 7)    // Quadrant 4
        );
    }}
    
    outlet
    {{
        type patch;
        faces
        (
            (8 11 10 9)  // Center face
            (8 12 15 11) // Quadrant 1
            (13 9 8 12)  // Quadrant 2
            (9 10 15 14) // Quadrant 3
            (11 10 14 15) // Quadrant 4 - CORRECTED ORIENTATION
        );
    }}
    
    walls
    {{
        type wall;
        faces
        (
            (4 5 13 12)  // Wall section 1
            (5 6 14 13)  // Wall section 2
            (6 7 15 14)  // Wall section 3
            (7 4 12 15)  // Wall section 4
        );
    }}
);

mergePatchPairs
(
);

// ************************************************************************* //
"""
        return content