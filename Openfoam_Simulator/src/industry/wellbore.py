#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wellbore modeling for Openfoam_Simulator.

This module provides classes and functions for modeling wellbores in the oil & gas
industry, including:
- Wellbore geometry definition
- Fluid flow in wellbores
- Heat transfer in wellbores
- Production and injection operations
- Well integrity analysis
"""

import os
import math
import numpy as np
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum, auto

# Import utility modules
from ..utils.logger import get_logger
from ..utils.unit_converter import convert_units
from ..config import get_value, set_value

# Import OpenFOAM integration modules
from ..openfoam_integration.case_manager import CaseManager
from ..openfoam_integration.solver_manager import SolverManager
from ..openfoam_integration.mesh_generator import MeshGenerator
from ..openfoam_integration.boundary_conditions import BoundaryCondition
from ..openfoam_integration.transport_models import TransportModel

# Import industry-specific modules
from .oilgas_properties import FluidProperties, MixtureProperties, PVTProperties

logger = get_logger(__name__)


class WellboreFlowRegime(Enum):
    """Enumeration of wellbore flow regimes."""
    SINGLE_PHASE = auto()
    BUBBLY = auto()
    SLUG = auto()
    CHURN = auto()
    ANNULAR = auto()
    MIST = auto()
    TRANSITION = auto()


class CasingType(Enum):
    """Enumeration of wellbore casing types."""
    CONDUCTOR = auto()
    SURFACE = auto()
    INTERMEDIATE = auto()
    PRODUCTION = auto()
    LINER = auto()


class Perforation:
    """Class representing a well perforation."""
    
    def __init__(self, md_top: float, md_bottom: float, azimuth: float,
                 phase_angle: float, diameter: float, density: float):
        """
        Initialize a perforation.
        
        Args:
            md_top (float): Top measured depth in m
            md_bottom (float): Bottom measured depth in m
            azimuth (float): Azimuth angle in degrees
            phase_angle (float): Phase angle in degrees
            diameter (float): Perforation diameter in m
            density (float): Perforation density in shots/m
        """
        self.md_top = md_top
        self.md_bottom = md_bottom
        self.azimuth = azimuth
        self.phase_angle = phase_angle
        self.diameter = diameter
        self.density = density  # Shots per meter
        
        # Calculate number of perforations
        interval_length = md_bottom - md_top
        self.num_perforations = int(interval_length * density)
    
    def get_positions(self) -> List[float]:
        """
        Get positions of perforations along the wellbore.
        
        Returns:
            List[float]: List of measured depths for each perforation
        """
        if self.num_perforations <= 0:
            return []
        
        # Distribute perforations evenly along the interval
        return np.linspace(self.md_top, self.md_bottom, self.num_perforations).tolist()
    
    def get_flow_area(self) -> float:
        """
        Calculate total flow area of the perforations.
        
        Returns:
            float: Total flow area in m²
        """
        single_area = math.pi * (self.diameter/2)**2
        return single_area * self.num_perforations


class CasingSection:
    """Class representing a wellbore casing section."""
    
    def __init__(self, casing_type: CasingType, md_top: float, md_bottom: float,
                 od: float, id: float, weight: float, grade: str, 
                 connections: str = None, material: str = "Steel"):
        """
        Initialize a casing section.
        
        Args:
            casing_type (CasingType): Type of casing
            md_top (float): Top measured depth in m
            md_bottom (float): Bottom measured depth in m
            od (float): Outer diameter in m
            id (float): Inner diameter in m
            weight (float): Weight in kg/m
            grade (str): Steel grade
            connections (str, optional): Connection type
            material (str, optional): Casing material
        """
        self.casing_type = casing_type
        self.md_top = md_top
        self.md_bottom = md_bottom
        self.od = od
        self.id = id
        self.weight = weight
        self.grade = grade
        self.connections = connections
        self.material = material
        
        # Calculate properties
        self.wall_thickness = (od - id) / 2
        self.length = md_bottom - md_top
        self.volume = math.pi * ((od/2)**2 - (id/2)**2) * self.length
        self.inner_area = math.pi * (id/2)**2
    
    def get_burst_pressure(self) -> float:
        """
        Calculate burst pressure rating.
        
        Returns:
            float: Burst pressure in Pa
        """
        # Barlow's formula: P = 2 * S * t / D
        # where S is yield strength, t is wall thickness, D is OD
        
        # Get yield strength based on grade
        # Values in psi, converted to Pa
        yield_strengths = {
            "H40": 40000 * 6894.76,  # psi to Pa
            "J55": 55000 * 6894.76,
            "K55": 55000 * 6894.76,
            "N80": 80000 * 6894.76,
            "L80": 80000 * 6894.76,
            "C90": 90000 * 6894.76,
            "C95": 95000 * 6894.76,
            "P110": 110000 * 6894.76,
            "Q125": 125000 * 6894.76,
            "V150": 150000 * 6894.76
        }
        
        yield_strength = yield_strengths.get(self.grade, 55000 * 6894.76)  # Default to J55
        
        # Safety factor
        safety_factor = 1.1
        
        # Calculate burst pressure
        burst_pressure = 2 * yield_strength * self.wall_thickness / self.od / safety_factor
        
        return burst_pressure
    
    def get_collapse_pressure(self) -> float:
        """
        Calculate collapse pressure rating.
        
        Returns:
            float: Collapse pressure in Pa
        """
        # Simplified calculation based on OD/t ratio
        odt_ratio = self.od / self.wall_thickness
        
        # Get yield strength based on grade
        yield_strengths = {
            "H40": 40000 * 6894.76,  # psi to Pa
            "J55": 55000 * 6894.76,
            "K55": 55000 * 6894.76,
            "N80": 80000 * 6894.76,
            "L80": 80000 * 6894.76,
            "C90": 90000 * 6894.76,
            "C95": 95000 * 6894.76,
            "P110": 110000 * 6894.76,
            "Q125": 125000 * 6894.76,
            "V150": 150000 * 6894.76
        }
        
        yield_strength = yield_strengths.get(self.grade, 55000 * 6894.76)  # Default to J55
        
        # Simplified formula based on elastoplastic collapse
        # Different formulas apply based on OD/t ratio
        if odt_ratio < 10:
            # Yield strength collapse
            collapse_pressure = 2 * yield_strength * ((self.od - self.id) / self.od)
        else:
            # Plastic collapse
            collapse_pressure = yield_strength * (self.wall_thickness / self.od) * (
                1 - (self.wall_thickness / self.od)) 
        
        # Safety factor
        safety_factor = 1.125
        
        return collapse_pressure / safety_factor


class TubingSection:
    """Class representing a wellbore tubing section."""
    
    def __init__(self, md_top: float, md_bottom: float, od: float, id: float,
                 weight: float, grade: str, connections: str = None,
                 material: str = "Steel", roughness: float = 0.0000457):
        """
        Initialize a tubing section.
        
        Args:
            md_top (float): Top measured depth in m
            md_bottom (float): Bottom measured depth in m
            od (float): Outer diameter in m
            id (float): Inner diameter in m
            weight (float): Weight in kg/m
            grade (str): Steel grade
            connections (str, optional): Connection type
            material (str, optional): Tubing material
            roughness (float, optional): Internal roughness in m
        """
        self.md_top = md_top
        self.md_bottom = md_bottom
        self.od = od
        self.id = id
        self.weight = weight
        self.grade = grade
        self.connections = connections
        self.material = material
        self.roughness = roughness
        
        # Calculate properties
        self.wall_thickness = (od - id) / 2
        self.length = md_bottom - md_top
        self.volume = math.pi * ((od/2)**2 - (id/2)**2) * self.length
        self.inner_area = math.pi * (id/2)**2
        self.hydraulic_diameter = id


class WellTrajectory:
    """Class representing a wellbore trajectory."""
    
    def __init__(self, name: str, surface_x: float, surface_y: float, kb_elevation: float):
        """
        Initialize a well trajectory.
        
        Args:
            name (str): Well name
            surface_x (float): Surface X coordinate in m
            surface_y (float): Surface Y coordinate in m
            kb_elevation (float): Kelly bushing elevation in m
        """
        self.name = name
        self.surface_x = surface_x
        self.surface_y = surface_y
        self.kb_elevation = kb_elevation
        
        # Survey data: MD, Inc, Azim
        self.survey_data = []
        
        # Calculated points: MD, TVD, X, Y
        self.points = []
    
    def add_survey_point(self, md: float, inclination: float, azimuth: float):
        """
        Add a survey point to the trajectory.
        
        Args:
            md (float): Measured depth in m
            inclination (float): Inclination in degrees (0 = vertical, 90 = horizontal)
            azimuth (float): Azimuth in degrees (0 = North, 90 = East)
        """
        self.survey_data.append((md, inclination, azimuth))
        self._calculate_trajectory()
    
    def load_from_file(self, filepath: str):
        """
        Load survey data from file.
        
        Args:
            filepath (str): Path to survey file
        """
        try:
            with open(filepath, 'r') as f:
                # Skip header
                next(f)
                
                # Read survey points
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        md = float(parts[0])
                        inclination = float(parts[1])
                        azimuth = float(parts[2])
                        self.survey_data.append((md, inclination, azimuth))
            
            # Calculate trajectory
            self._calculate_trajectory()
            
        except Exception as e:
            logger.error(f"Error loading survey data: {e}")
    
    def _calculate_trajectory(self):
        """Calculate wellbore trajectory using minimum curvature method."""
        if not self.survey_data:
            return
        
        # Sort survey data by MD
        self.survey_data.sort(key=lambda x: x[0])
        
        # Reset points
        self.points = []
        
        # Add surface point
        self.points.append((0, 0, self.surface_x, self.surface_y))
        
        # Calculate trajectory using minimum curvature method
        for i in range(1, len(self.survey_data)):
            md1, inc1, az1 = self.survey_data[i-1]
            md2, inc2, az2 = self.survey_data[i]
            
            # Convert to radians
            inc1_rad = math.radians(inc1)
            az1_rad = math.radians(az1)
            inc2_rad = math.radians(inc2)
            az2_rad = math.radians(az2)
            
            # Calculate dog leg severity
            dog_leg = math.acos(math.cos(inc1_rad) * math.cos(inc2_rad) + 
                              math.sin(inc1_rad) * math.sin(inc2_rad) * 
                              math.cos(az2_rad - az1_rad))
            
            # Calculate ratio factor
            if dog_leg < 0.0001:
                rf = 1.0
            else:
                rf = 2 * math.tan(dog_leg/2) / dog_leg
            
            # Calculate position changes
            md_delta = md2 - md1
            dx = md_delta/2 * (math.sin(inc1_rad) * math.sin(az1_rad) + 
                             math.sin(inc2_rad) * math.sin(az2_rad)) * rf
            dy = md_delta/2 * (math.sin(inc1_rad) * math.cos(az1_rad) + 
                             math.sin(inc2_rad) * math.cos(az2_rad)) * rf
            dz = md_delta/2 * (math.cos(inc1_rad) + math.cos(inc2_rad)) * rf
            
            # Get previous TVD, X, Y
            prev_md, prev_tvd, prev_x, prev_y = self.points[-1]
            
            # Calculate new position
            tvd = prev_tvd + dz
            x = prev_x + dx
            y = prev_y + dy
            
            # Add point
            self.points.append((md2, tvd, x, y))
    
    def get_position(self, md: float) -> Tuple[float, float, float]:
        """
        Get position at a given measured depth.
        
        Args:
            md (float): Measured depth in m
            
        Returns:
            Tuple[float, float, float]: (TVD, X, Y) coordinates
        """
        if not self.points:
            return (md, 0, 0)  # Vertical well if no survey data
        
        if md <= 0:
            return (0, self.surface_x, self.surface_y)
        
        # Find points that bracket the desired MD
        for i in range(1, len(self.points)):
            md1, tvd1, x1, y1 = self.points[i-1]
            md2, tvd2, x2, y2 = self.points[i]
            
            if md1 <= md <= md2:
                # Interpolate
                ratio = (md - md1) / (md2 - md1) if md2 != md1 else 0
                tvd = tvd1 + ratio * (tvd2 - tvd1)
                x = x1 + ratio * (x2 - x1)
                y = y1 + ratio * (y2 - y1)
                return (tvd, x, y)
        
        # If MD is beyond the last point, extrapolate using the last segment
        last_md, last_tvd, last_x, last_y = self.points[-1]
        if len(self.points) > 1:
            prev_md, prev_tvd, prev_x, prev_y = self.points[-2]
            
            # Calculate rates of change
            md_delta = last_md - prev_md
            tvd_delta = last_tvd - prev_tvd
            x_delta = last_x - prev_x
            y_delta = last_y - prev_y
            
            if md_delta > 0:
                # Extrapolate
                ratio = (md - last_md) / md_delta
                tvd = last_tvd + ratio * tvd_delta
                x = last_x + ratio * x_delta
                y = last_y + ratio * y_delta
                return (tvd, x, y)
        
        # Default to extending vertically if no other data
        return (last_tvd + (md - last_md), last_x, last_y)
    
    def get_inclination(self, md: float) -> float:
        """
        Get inclination at a given measured depth.
        
        Args:
            md (float): Measured depth in m
            
        Returns:
            float: Inclination in degrees
        """
        if not self.survey_data:
            return 0.0  # Vertical well if no survey data
        
        # Find points that bracket the desired MD
        for i in range(1, len(self.survey_data)):
            md1, inc1, _ = self.survey_data[i-1]
            md2, inc2, _ = self.survey_data[i]
            
            if md1 <= md <= md2:
                # Interpolate
                ratio = (md - md1) / (md2 - md1) if md2 != md1 else 0
                return inc1 + ratio * (inc2 - inc1)
        
        # If MD is beyond the last point, return the last inclination
        if self.survey_data:
            return self.survey_data[-1][1]
        
        return 0.0
    
    def get_azimuth(self, md: float) -> float:
        """
        Get azimuth at a given measured depth.
        
        Args:
            md (float): Measured depth in m
            
        Returns:
            float: Azimuth in degrees
        """
        if not self.survey_data:
            return 0.0  # North by default if no survey data
        
        # Find points that bracket the desired MD
        for i in range(1, len(self.survey_data)):
            md1, _, az1 = self.survey_data[i-1]
            md2, _, az2 = self.survey_data[i]
            
            if md1 <= md <= md2:
                # Interpolate
                ratio = (md - md1) / (md2 - md1) if md2 != md1 else 0
                return az1 + ratio * (az2 - az1)
        
        # If MD is beyond the last point, return the last azimuth
        if self.survey_data:
            return self.survey_data[-1][2]
        
        return 0.0
    
    def get_dogleg_severity(self, md: float) -> float:
        """
        Get dogleg severity at a given measured depth.
        
        Args:
            md (float): Measured depth in m
            
        Returns:
            float: Dogleg severity in degrees per 30m
        """
        if len(self.survey_data) < 2:
            return 0.0
        
        # Find points that bracket the desired MD
        for i in range(1, len(self.survey_data)):
            md1, inc1, az1 = self.survey_data[i-1]
            md2, inc2, az2 = self.survey_data[i]
            
            if md1 <= md <= md2:
                # Convert to radians
                inc1_rad = math.radians(inc1)
                az1_rad = math.radians(az1)
                inc2_rad = math.radians(inc2)
                az2_rad = math.radians(az2)
                
                # Calculate dog leg angle
                dog_leg = math.acos(math.cos(inc1_rad) * math.cos(inc2_rad) + 
                                  math.sin(inc1_rad) * math.sin(inc2_rad) * 
                                  math.cos(az2_rad - az1_rad))
                
                # Convert to degrees and normalize to 30m
                dog_leg_deg = math.degrees(dog_leg)
                md_diff = md2 - md1
                if md_diff > 0:
                    return dog_leg_deg * 30 / md_diff
                else:
                    return 0.0
        
        return 0.0  # Default to zero if MD is out of range


class Wellbore:
    """Class representing a complete wellbore."""
    
    def __init__(self, name: str, trajectory: WellTrajectory = None):
        """
        Initialize a wellbore.
        
        Args:
            name (str): Well name
            trajectory (WellTrajectory, optional): Well trajectory
        """
        self.name = name
        self.trajectory = trajectory or WellTrajectory(name, 0, 0, 0)
        
        # Wellbore components
        self.casing_sections = []
        self.tubing_sections = []
        self.perforations = []
        
        # Calculated properties
        self.total_depth = 0.0
    
    def add_casing_section(self, casing_section: CasingSection):
        """
        Add a casing section to the wellbore.
        
        Args:
            casing_section (CasingSection): Casing section to add
        """
        self.casing_sections.append(casing_section)
        self.casing_sections.sort(key=lambda x: x.md_top)
        self.total_depth = max(self.total_depth, casing_section.md_bottom)
    
    def add_tubing_section(self, tubing_section: TubingSection):
        """
        Add a tubing section to the wellbore.
        
        Args:
            tubing_section (TubingSection): Tubing section to add
        """
        self.tubing_sections.append(tubing_section)
        self.tubing_sections.sort(key=lambda x: x.md_top)
        self.total_depth = max(self.total_depth, tubing_section.md_bottom)
    
    def add_perforation(self, perforation: Perforation):
        """
        Add a perforation to the wellbore.
        
        Args:
            perforation (Perforation): Perforation to add
        """
        self.perforations.append(perforation)
        self.perforations.sort(key=lambda x: x.md_top)
    
    def get_diameter_profile(self) -> Dict[str, List[Tuple[float, float]]]:
        """
        Get diameter profile along the wellbore.
        
        Returns:
            Dict[str, List[Tuple[float, float]]]: Dictionary of 
                profile lists (MD, Diameter) for each component type
        """
        profiles = {
            "casing_od": [],
            "casing_id": [],
            "tubing_od": [],
            "tubing_id": [],
        }
        
        # Process casing sections
        for section in self.casing_sections:
            profiles["casing_od"].append((section.md_top, section.od))
            profiles["casing_od"].append((section.md_bottom, section.od))
            
            profiles["casing_id"].append((section.md_top, section.id))
            profiles["casing_id"].append((section.md_bottom, section.id))
        
        # Process tubing sections
        for section in self.tubing_sections:
            profiles["tubing_od"].append((section.md_top, section.od))
            profiles["tubing_od"].append((section.md_bottom, section.od))
            
            profiles["tubing_id"].append((section.md_top, section.id))
            profiles["tubing_id"].append((section.md_bottom, section.id))
        
        # Sort profiles by MD
        for key in profiles:
            profiles[key].sort(key=lambda x: x[0])
        
        return profiles
    
    def get_csg_at_depth(self, md: float) -> Optional[CasingSection]:
        """
        Get the casing section at a given depth.
        
        Args:
            md (float): Measured depth in m
            
        Returns:
            Optional[CasingSection]: Casing section at the given depth, or None
        """
        for section in self.casing_sections:
            if section.md_top <= md <= section.md_bottom:
                return section
        return None
    
    def get_tbg_at_depth(self, md: float) -> Optional[TubingSection]:
        """
        Get the tubing section at a given depth.
        
        Args:
            md (float): Measured depth in m
            
        Returns:
            Optional[TubingSection]: Tubing section at the given depth, or None
        """
        for section in self.tubing_sections:
            if section.md_top <= md <= section.md_bottom:
                return section
        return None
    
    def has_perforations_at_depth(self, md: float) -> bool:
        """
        Check if there are perforations at a given depth.
        
        Args:
            md (float): Measured depth in m
            
        Returns:
            bool: True if there are perforations at the given depth
        """
        for perf in self.perforations:
            if perf.md_top <= md <= perf.md_bottom:
                return True
        return False
    
    def get_annular_area(self, md: float) -> float:
        """
        Calculate annular area at a given depth.
        
        Args:
            md (float): Measured depth in m
            
        Returns:
            float: Annular area in m²
        """
        csg = self.get_csg_at_depth(md)
        tbg = self.get_tbg_at_depth(md)
        
        if not csg:
            return 0.0
        
        if not tbg:
            # No tubing, full casing area
            return math.pi * (csg.id/2)**2
        
        # Annular area
        return math.pi * ((csg.id/2)**2 - (tbg.od/2)**2)
    
    def get_flow_path_geometry(self, md: float) -> Dict[str, float]:
        """
        Get flow path geometry at a given depth.
        
        Args:
            md (float): Measured depth in m
            
        Returns:
            Dict[str, float]: Dictionary of flow path geometry parameters
        """
        csg = self.get_csg_at_depth(md)
        tbg = self.get_tbg_at_depth(md)
        
        result = {
            "tubing_area": 0.0,
            "annular_area": 0.0,
            "tubing_hydraulic_diameter": 0.0,
            "annular_hydraulic_diameter": 0.0,
            "tubing_roughness": 0.0,
            "annular_roughness": 0.0,
        }
        
        if tbg:
            result["tubing_area"] = math.pi * (tbg.id/2)**2
            result["tubing_hydraulic_diameter"] = tbg.id
            result["tubing_roughness"] = tbg.roughness
        
        if csg and tbg:
            # Annular area
            result["annular_area"] = math.pi * ((csg.id/2)**2 - (tbg.od/2)**2)
            
            # Hydraulic diameter = 4 * (flow area) / (wetted perimeter)
            # For annulus: Dh = Dcsg_id - Dtbg_od
            result["annular_hydraulic_diameter"] = csg.id - tbg.od
            
            # Effective roughness for annulus (simplified)
            result["annular_roughness"] = (csg.id * 0.0000457 + tbg.od * tbg.roughness) / (csg.id + tbg.od)
        
        return result
    
    def get_3d_mesh_geometry(self) -> Dict[str, np.ndarray]:
        """
        Generate 3D mesh geometry for the wellbore.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary of mesh geometry arrays
        """
        # Calculate trajectory points for mesh
        if not self.trajectory.points:
            return {}
        
        # Create a denser set of points along the trajectory
        md_values = np.linspace(0, self.total_depth, max(100, int(self.total_depth / 10)))
        
        # Initialize arrays
        tvd_values = np.zeros_like(md_values)
        x_values = np.zeros_like(md_values)
        y_values = np.zeros_like(md_values)
        
        # Get positions along trajectory
        for i, md in enumerate(md_values):
            tvd, x, y = self.trajectory.get_position(md)
            tvd_values[i] = tvd
            x_values[i] = x
            y_values[i] = y
        
        # Get diameters along the wellbore
        diameter_profiles = self.get_diameter_profile()
        
        casing_id_values = np.zeros_like(md_values)
        tubing_od_values = np.zeros_like(md_values)
        tubing_id_values = np.zeros_like(md_values)
        
        # Interpolate diameters
        for i, md in enumerate(md_values):
            csg = self.get_csg_at_depth(md)
            tbg = self.get_tbg_at_depth(md)
            
            casing_id_values[i] = csg.id if csg else 0.0
            tubing_od_values[i] = tbg.od if tbg else 0.0
            tubing_id_values[i] = tbg.id if tbg else 0.0
        
        # Create perforation points
        perf_positions = []
        for perf in self.perforations:
            for md in perf.get_positions():
                tvd, x, y = self.trajectory.get_position(md)
                perf_positions.append((md, tvd, x, y, perf.diameter))
        
        return {
            "md": md_values,
            "tvd": tvd_values,
            "x": x_values,
            "y": y_values,
            "casing_id": casing_id_values,
            "tubing_od": tubing_od_values,
            "tubing_id": tubing_id_values,
            "perforations": np.array(perf_positions) if perf_positions else np.empty((0, 5))
        }
    
    def export_to_json(self, filepath: str):
        """
        Export wellbore to JSON.
        
        Args:
            filepath (str): Output file path
        """
        # Create dictionary representation
        wellbore_dict = {
            "name": self.name,
            "trajectory": {
                "name": self.trajectory.name,
                "surface_x": self.trajectory.surface_x,
                "surface_y": self.trajectory.surface_y,
                "kb_elevation": self.trajectory.kb_elevation,
                "survey_data": self.trajectory.survey_data
            },
            "casing_sections": [
                {
                    "casing_type": section.casing_type.name,
                    "md_top": section.md_top,
                    "md_bottom": section.md_bottom,
                    "od": section.od,
                    "id": section.id,
                    "weight": section.weight,
                    "grade": section.grade,
                    "connections": section.connections,
                    "material": section.material
                }
                for section in self.casing_sections
            ],
            "tubing_sections": [
                {
                    "md_top": section.md_top,
                    "md_bottom": section.md_bottom,
                    "od": section.od,
                    "id": section.id,
                    "weight": section.weight,
                    "grade": section.grade,
                    "connections": section.connections,
                    "material": section.material,
                    "roughness": section.roughness
                }
                for section in self.tubing_sections
            ],
            "perforations": [
                {
                    "md_top": perf.md_top,
                    "md_bottom": perf.md_bottom,
                    "azimuth": perf.azimuth,
                    "phase_angle": perf.phase_angle,
                    "diameter": perf.diameter,
                    "density": perf.density
                }
                for perf in self.perforations
            ],
            "total_depth": self.total_depth
        }
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(wellbore_dict, f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'Wellbore':
        """
        Load wellbore from JSON.
        
        Args:
            filepath (str): Input file path
            
        Returns:
            Wellbore: Loaded wellbore object
        """
        with open(filepath, 'r') as f:
            wellbore_dict = json.load(f)
        
        # Create trajectory
        traj_dict = wellbore_dict.get("trajectory", {})
        trajectory = WellTrajectory(
            name=traj_dict.get("name", ""),
            surface_x=traj_dict.get("surface_x", 0.0),
            surface_y=traj_dict.get("surface_y", 0.0),
            kb_elevation=traj_dict.get("kb_elevation", 0.0)
        )
        
        # Add survey data
        for md, inc, az in traj_dict.get("survey_data", []):
            trajectory.add_survey_point(md, inc, az)
        
        # Create wellbore
        wellbore = cls(wellbore_dict.get("name", ""), trajectory)
        
        # Add casing sections
        for csg_dict in wellbore_dict.get("casing_sections", []):
            # Parse casing type
            casing_type_name = csg_dict.get("casing_type", "PRODUCTION")
            casing_type = getattr(CasingType, casing_type_name, CasingType.PRODUCTION)
            
            casing_section = CasingSection(
                casing_type=casing_type,
                md_top=csg_dict.get("md_top", 0.0),
                md_bottom=csg_dict.get("md_bottom", 0.0),
                od=csg_dict.get("od", 0.0),
                id=csg_dict.get("id", 0.0),
                weight=csg_dict.get("weight", 0.0),
                grade=csg_dict.get("grade", ""),
                connections=csg_dict.get("connections"),
                material=csg_dict.get("material", "Steel")
            )
            wellbore.add_casing_section(casing_section)
        
        # Add tubing sections
        for tbg_dict in wellbore_dict.get("tubing_sections", []):
            tubing_section = TubingSection(
                md_top=tbg_dict.get("md_top", 0.0),
                md_bottom=tbg_dict.get("md_bottom", 0.0),
                od=tbg_dict.get("od", 0.0),
                id=tbg_dict.get("id", 0.0),
                weight=tbg_dict.get("weight", 0.0),
                grade=tbg_dict.get("grade", ""),
                connections=tbg_dict.get("connections"),
                material=tbg_dict.get("material", "Steel"),
                roughness=tbg_dict.get("roughness", 0.0000457)
            )
            wellbore.add_tubing_section(tubing_section)
        
        # Add perforations
        for perf_dict in wellbore_dict.get("perforations", []):
            perforation = Perforation(
                md_top=perf_dict.get("md_top", 0.0),
                md_bottom=perf_dict.get("md_bottom", 0.0),
                azimuth=perf_dict.get("azimuth", 0.0),
                phase_angle=perf_dict.get("phase_angle", 0.0),
                diameter=perf_dict.get("diameter", 0.0),
                density=perf_dict.get("density", 0.0)
            )
            wellbore.add_perforation(perforation)
        
        return wellbore


class WellboreFluidProperties:
    """Class representing fluid properties in a wellbore."""
    
    def __init__(self, temperature_profile: List[Tuple[float, float]], 
                 pressure_profile: List[Tuple[float, float]],
                 fluid_props: FluidProperties = None, 
                 pvt_model: PVTProperties = None):
        """
        Initialize wellbore fluid properties.
        
        Args:
            temperature_profile (List[Tuple[float, float]]): List of (MD, temperature)
            pressure_profile (List[Tuple[float, float]]): List of (MD, pressure)
            fluid_props (FluidProperties, optional): Base fluid properties
            pvt_model (PVTProperties, optional): PVT model for fluid
        """
        self.temperature_profile = temperature_profile
        self.pressure_profile = pressure_profile
        self.fluid_props = fluid_props
        self.pvt_model = pvt_model
    
    def get_temperature(self, md: float) -> float:
        """
        Get temperature at a given depth.
        
        Args:
            md (float): Measured depth in m
            
        Returns:
            float: Temperature in K
        """
        if not self.temperature_profile:
            return 293.15  # Default to 20°C
        
        # Find points that bracket the desired MD
        for i in range(1, len(self.temperature_profile)):
            md1, t1 = self.temperature_profile[i-1]
            md2, t2 = self.temperature_profile[i]
            
            if md1 <= md <= md2:
                # Interpolate
                ratio = (md - md1) / (md2 - md1) if md2 != md1 else 0
                return t1 + ratio * (t2 - t1)
        
        # If MD is beyond the range, use the closest point
        if md < self.temperature_profile[0][0]:
            return self.temperature_profile[0][1]
        else:
            return self.temperature_profile[-1][1]
    
    def get_pressure(self, md: float) -> float:
        """
        Get pressure at a given depth.
        
        Args:
            md (float): Measured depth in m
            
        Returns:
            float: Pressure in Pa
        """
        if not self.pressure_profile:
            return 101325 + 9.81 * 1000 * md  # Simple hydrostatic
        
        # Find points that bracket the desired MD
        for i in range(1, len(self.pressure_profile)):
            md1, p1 = self.pressure_profile[i-1]
            md2, p2 = self.pressure_profile[i]
            
            if md1 <= md <= md2:
                # Interpolate
                ratio = (md - md1) / (md2 - md1) if md2 != md1 else 0
                return p1 + ratio * (p2 - p1)
        
        # If MD is beyond the range, use the closest point
        if md < self.pressure_profile[0][0]:
            return self.pressure_profile[0][1]
        else:
            return self.pressure_profile[-1][1]
    
    def get_fluid_props_at_depth(self, md: float) -> Dict[str, float]:
        """
        Get fluid properties at a given depth.
        
        Args:
            md (float): Measured depth in m
            
        Returns:
            Dict[str, float]: Fluid properties at the given depth
        """
        # Get pressure and temperature
        pressure = self.get_pressure(md)
        temperature = self.get_temperature(md)
        
        # If PVT model is available, use it
        if self.pvt_model:
            return self.pvt_model.get_properties(pressure, temperature)
        
        # If base fluid properties are available, use simple correlations
        if self.fluid_props:
            # Simple pressure and temperature corrections
            density = self.fluid_props.density * (1 + 1e-7 * (pressure - 101325)) * (1 - 5e-4 * (temperature - 293.15))
            viscosity = self.fluid_props.viscosity * math.exp(-0.02 * (temperature - 293.15)) * (1 + 1e-8 * (pressure - 101325))
            
            return {
                "density": density,
                "viscosity": viscosity,
                "formation_volume_factor": 1.0,
                "gas_oil_ratio": 0.0,
                "bubble_point": 0.0
            }
        
        # Default values
        return {
            "density": 1000.0,  # kg/m³
            "viscosity": 0.001,  # Pa·s
            "formation_volume_factor": 1.0,
            "gas_oil_ratio": 0.0,
            "bubble_point": 0.0
        }


class WellboreFlow:
    """Class for modeling flow in a wellbore."""
    
    def __init__(self, wellbore: Wellbore, fluid_props: WellboreFluidProperties = None):
        """
        Initialize wellbore flow model.
        
        Args:
            wellbore (Wellbore): Wellbore geometry
            fluid_props (WellboreFluidProperties, optional): Fluid properties
        """
        self.wellbore = wellbore
        self.fluid_props = fluid_props or WellboreFluidProperties([], [])
        
        # Flow properties
        self.flow_rate = 0.0  # m³/s
        self.wc = 0.0  # Water cut (fraction)
        self.gor = 0.0  # Gas-oil ratio (m³/m³)
        self.dp_friction = 0.0  # Friction pressure drop (Pa)
        self.velocity_profile = []  # (MD, velocity)
        
        # Flow direction
        self.is_production = True  # True for production, False for injection
    
    def set_production_rate(self, oil_rate: float, wc: float, gor: float):
        """
        Set production rates.
        
        Args:
            oil_rate (float): Oil rate in m³/s
            wc (float): Water cut (fraction)
            gor (float): Gas-oil ratio (m³/m³)
        """
        self.flow_rate = oil_rate
        self.wc = wc
        self.gor = gor
        self.is_production = True
    
    def set_injection_rate(self, fluid_rate: float):
        """
        Set injection rate.
        
        Args:
            fluid_rate (float): Injection rate in m³/s
        """
        self.flow_rate = fluid_rate
        self.is_production = False
    
    def _calculate_friction_factor(self, reynolds: float, roughness_ratio: float) -> float:
        """
        Calculate friction factor using Colebrook correlation.
        
        Args:
            reynolds (float): Reynolds number
            roughness_ratio (float): Relative roughness (roughness/diameter)
            
        Returns:
            float: Friction factor
        """
        if reynolds < 2000:
            # Laminar flow
            return 64 / reynolds
        elif reynolds > 4000:
            # Turbulent flow: use Colebrook correlation
            # Approximate solution using Chen correlation
            a = -2 * math.log10(roughness_ratio/3.7 + 2.51/(reynolds * (1e-7 + roughness_ratio/3.7)))
            return 1 / (a**2)
        else:
            # Transition region
            # Linear interpolation
            f_2000 = 64 / 2000
            roughness_ratio_limited = max(1e-6, roughness_ratio)
            a = -2 * math.log10(roughness_ratio_limited/3.7 + 2.51/(4000 * (1e-7 + roughness_ratio_limited/3.7)))
            f_4000 = 1 / (a**2)
            
            ratio = (reynolds - 2000) / 2000
            return f_2000 + ratio * (f_4000 - f_2000)
    
    def _calculate_mixture_properties(self, md: float, pressure: float, temperature: float) -> Dict[str, float]:
        """
        Calculate multiphase mixture properties.
        
        Args:
            md (float): Measured depth in m
            pressure (float): Pressure in Pa
            temperature (float): Temperature in K
            
        Returns:
            Dict[str, float]: Mixture properties
        """
        # Get base fluid properties
        props = self.fluid_props.get_fluid_props_at_depth(md)
        
        # For simplicity, use simple correlations for multiphase mixtures
        # In a full model, this would use proper multiphase flow models
        
        # Simple mixture density
        if self.is_production:
            # Production (oil, water, gas)
            # Get phase densities
            oil_density = props.get("density", 850.0)  # kg/m³
            water_density = 1000.0  # kg/m³
            
            # Gas density using ideal gas law (simplified)
            gas_density = 1.2 * (pressure / 101325) * (293.15 / temperature)  # kg/m³
            
            # Calculate volume fractions
            total_volume = 1.0
            water_volume = self.wc * total_volume
            gas_volume = self.gor * (total_volume - water_volume)
            oil_volume = total_volume - water_volume - gas_volume
            
            # Calculate mixture density
            if total_volume > 0:
                mixture_density = (oil_volume * oil_density + water_volume * water_density + gas_volume * gas_density) / total_volume
            else:
                mixture_density = oil_density
            
            # Calculate mixture viscosity (simplified)
            oil_viscosity = props.get("viscosity", 0.01)  # Pa·s
            water_viscosity = 0.001  # Pa·s
            gas_viscosity = 1.8e-5  # Pa·s
            
            # Log mean viscosity
            if oil_volume > 0 and water_volume > 0 and gas_volume > 0:
                liquid_volume = oil_volume + water_volume
                liquid_viscosity = (oil_volume * oil_viscosity + water_volume * water_viscosity) / liquid_volume
                
                # Two-phase viscosity using Dukler correlation (simplified)
                mixture_viscosity = liquid_viscosity * liquid_volume / total_volume + gas_viscosity * gas_volume / total_volume
            else:
                mixture_viscosity = oil_viscosity
        else:
            # Injection (single phase)
            mixture_density = props.get("density", 1000.0)
            mixture_viscosity = props.get("viscosity", 0.001)
        
        return {
            "density": mixture_density,
            "viscosity": mixture_viscosity,
        }
    
    def calculate_flow_profile(self, steps: int = 50) -> Dict[str, List[float]]:
        """
        Calculate flow profile along the wellbore.
        
        Args:
            steps (int, optional): Number of calculation steps
            
        Returns:
            Dict[str, List[float]]: Flow profile data
        """
        # Initialize arrays
        md_values = np.linspace(0, self.wellbore.total_depth, steps)
        pressure_values = np.zeros_like(md_values)
        temperature_values = np.zeros_like(md_values)
        velocity_values = np.zeros_like(md_values)
        reynolds_values = np.zeros_like(md_values)
        density_values = np.zeros_like(md_values)
        viscosity_values = np.zeros_like(md_values)
        
        # Direction for calculation
        if self.is_production:
            # Production: calculate from bottom to top
            md_values = md_values[::-1]
        
        # Define bottomhole or wellhead pressure
        if self.is_production:
            # Start with bottomhole pressure
            pressure_values[0] = self.fluid_props.get_pressure(self.wellbore.total_depth)
        else:
            # Start with wellhead pressure
            pressure_values[0] = self.fluid_props.get_pressure(0)
        
        # Loop through depths
        for i in range(len(md_values)):
            md = md_values[i]
            
            # Get temperature from profile
            temperature_values[i] = self.fluid_props.get_temperature(md)
            
            # Get flow geometry
            geo = self.wellbore.get_flow_path_geometry(md)
            
            if i > 0:
                # Get the prevouse pressure
                p_prev = pressure_values[i-1]
                
                # Get the previous and current depths
                md_prev = md_values[i-1]
                
                # Get depths for gravity calculation
                tvd_prev, _, _ = self.wellbore.trajectory.get_position(md_prev)
                tvd, _, _ = self.wellbore.trajectory.get_position(md)
                
                # Calculate fluid properties at average conditions
                p_avg = p_prev
                t_avg = (temperature_values[i-1] + temperature_values[i]) / 2
                md_avg = (md_prev + md) / 2
                
                # Get mixture properties
                mixture = self._calculate_mixture_properties(md_avg, p_avg, t_avg)
                density = mixture["density"]
                viscosity = mixture["viscosity"]
                
                # Store density and viscosity
                density_values[i] = density
                viscosity_values[i] = viscosity
                
                # Calculate flow area
                area = geo["tubing_area"] if self.is_production else geo["tubing_area"]
                
                # Calculate velocity
                if area > 0:
                    velocity = self.flow_rate / area
                else:
                    velocity = 0.0
                
                # Store velocity
                velocity_values[i] = velocity
                
                # Calculate Reynolds number
                d_h = geo["tubing_hydraulic_diameter"] if self.is_production else geo["tubing_hydraulic_diameter"]
                if d_h > 0 and viscosity > 0:
                    reynolds = density * abs(velocity) * d_h / viscosity
                else:
                    reynolds = 0.0
                
                # Store Reynolds number
                reynolds_values[i] = reynolds
                
                # Calculate hydrostatic pressure change
                dp_hydrostatic = density * 9.81 * (tvd - tvd_prev)
                
                # Calculate friction pressure change
                if d_h > 0 and reynolds > 0:
                    roughness = geo["tubing_roughness"] if self.is_production else geo["tubing_roughness"]
                    roughness_ratio = roughness / d_h
                    
                    friction_factor = self._calculate_friction_factor(reynolds, roughness_ratio)
                    
                    dmd = abs(md - md_prev)
                    dp_friction = friction_factor * density * velocity**2 * dmd / (2 * d_h)
                else:
                    dp_friction = 0.0
                
                # Calculate new pressure
                if self.is_production:
                    # Production: pressure decreases upward
                    # Friction acts against flow, decreasing pressure
                    pressure_values[i] = p_prev - dp_hydrostatic - dp_friction
                else:
                    # Injection: pressure decreases downward
                    # Friction acts against flow, decreasing pressure
                    # Hydrostatic pressure increases downward
                    pressure_values[i] = p_prev + dp_hydrostatic - dp_friction
            else:
                # For first point, use fluid properties at the starting point
                mixture = self._calculate_mixture_properties(md, pressure_values[0], temperature_values[0])
                density_values[i] = mixture["density"]
                viscosity_values[i] = mixture["viscosity"]
                
                # Calculate flow area and velocity for first point
                geo = self.wellbore.get_flow_path_geometry(md)
                area = geo["tubing_area"] if self.is_production else geo["tubing_area"]
                
                if area > 0:
                    velocity_values[i] = self.flow_rate / area
                else:
                    velocity_values[i] = 0.0
                
                # Calculate Reynolds number for first point
                d_h = geo["tubing_hydraulic_diameter"] if self.is_production else geo["tubing_hydraulic_diameter"]
                if d_h > 0 and mixture["viscosity"] > 0:
                    reynolds_values[i] = density_values[i] * abs(velocity_values[i]) * d_h / mixture["viscosity"]
                else:
                    reynolds_values[i] = 0.0
        
        # If production, reverse arrays back to increasing depth
        if self.is_production:
            md_values = md_values[::-1]
            pressure_values = pressure_values[::-1]
            temperature_values = temperature_values[::-1]
            velocity_values = velocity_values[::-1]
            reynolds_values = reynolds_values[::-1]
            density_values = density_values[::-1]
            viscosity_values = viscosity_values[::-1]
        
        return {
            "md": md_values.tolist(),
            "pressure": pressure_values.tolist(),
            "temperature": temperature_values.tolist(),
            "velocity": velocity_values.tolist(),
            "reynolds": reynolds_values.tolist(),
            "density": density_values.tolist(),
            "viscosity": viscosity_values.tolist()
        }
    
    def calculate_flow_regime(self, velocity: float, density: float, viscosity: float, 
                            diameter: float) -> WellboreFlowRegime:
        """
        Calculate flow regime for multiphase flow.
        
        Args:
            velocity (float): Flow velocity in m/s
            density (float): Fluid density in kg/m³
            viscosity (float): Fluid viscosity in Pa·s
            diameter (float): Pipe diameter in m
            
        Returns:
            WellboreFlowRegime: Flow regime
        """
        # Calculate Reynolds number
        reynolds = density * abs(velocity) * diameter / viscosity
        
        # For single phase flow, only laminar/turbulent
        if self.gor < 0.001 and self.wc < 0.001:
            # Essentially oil only
            if reynolds < 2300:
                return WellboreFlowRegime.SINGLE_PHASE  # Laminar
            else:
                return WellboreFlowRegime.SINGLE_PHASE  # Turbulent
        
        # For multiphase flow, use simplified flow regime map
        # Based on superficial velocities
        
        # Calculate superficial velocities (simplified)
        # In a full model, use proper PVT correlations for volume factors
        total_volume = 1.0
        water_volume = self.wc * total_volume
        gas_volume = self.gor * (total_volume - water_volume)
        oil_volume = total_volume - water_volume - gas_volume
        
        liquid_volume = oil_volume + water_volume
        
        # Calculate velocities
        if liquid_volume > 0 and gas_volume > 0:
            area = math.pi * (diameter/2)**2
            liquid_velocity = self.flow_rate * liquid_volume / total_volume / area
            gas_velocity = self.flow_rate * gas_volume / total_volume / area
            
            # Simple flow regime map
            if gas_velocity < 1.0:
                if liquid_velocity < 0.3:
                    return WellboreFlowRegime.BUBBLY
                else:
                    return WellboreFlowRegime.SLUG
            elif gas_velocity < 10.0:
                return WellboreFlowRegime.CHURN
            else:
                if liquid_velocity < 0.3:
                    return WellboreFlowRegime.MIST
                else:
                    return WellboreFlowRegime.ANNULAR
        else:
            return WellboreFlowRegime.SINGLE_PHASE


class WellboreCFDModel:
    """Class for setting up CFD simulations of wellbore flows."""
    
    def __init__(self, wellbore: Wellbore, wellbore_flow: WellboreFlow):
        """
        Initialize wellbore CFD model.
        
        Args:
            wellbore (Wellbore): Wellbore geometry
            wellbore_flow (WellboreFlow): Wellbore flow properties
        """
        self.wellbore = wellbore
        self.wellbore_flow = wellbore_flow
        self.case_manager = None
    
    def setup_case(self, case_dir: str, section_start: float = 0.0, section_end: float = None):
        """
        Set up OpenFOAM case for wellbore flow simulation.
        
        Args:
            case_dir (str): Directory for the OpenFOAM case
            section_start (float, optional): Start MD for the section to model
            section_end (float, optional): End MD for the section to model
            
        Returns:
            CaseManager: The configured case manager
        """
        # Create case manager
        self.case_manager = CaseManager(case_dir)
        
        # Set end depth if not specified
        if section_end is None:
            section_end = self.wellbore.total_depth
        
        # Get segment trajectory
        segment_points = []
        md_values = np.linspace(section_start, section_end, 100)
        for md in md_values:
            tvd, x, y = self.wellbore.trajectory.get_position(md)
            segment_points.append((md, tvd, x, y))
        
        # Create mesh generator
        mesh_generator = MeshGenerator(self.case_manager)
        
        # Get diameter at section start and end
        start_geo = self.wellbore.get_flow_path_geometry(section_start)
        end_geo = self.wellbore.get_flow_path_geometry(section_end)
        
        start_tubing_id = 0.0
        end_tubing_id = 0.0
        
        if "tubing_hydraulic_diameter" in start_geo:
            start_tubing_id = start_geo["tubing_hydraulic_diameter"]
        
        if "tubing_hydraulic_diameter" in end_geo:
            end_tubing_id = end_geo["tubing_hydraulic_diameter"]
        
        # Generate wellbore mesh
        mesh_generator.create_wellbore_mesh(
            trajectory_points=segment_points,
            tubing_diameter=[(section_start, start_tubing_id), (section_end, end_tubing_id)],
            cells_across_diameter=20,
            cells_along_wellbore=500
        )
        
        # Determine solver based on flow conditions
        if self.wellbore_flow.gor > 0.01 or self.wellbore_flow.wc > 0.01:
            # Multiphase flow
            solver = "multiphaseInterFoam"
        else:
            # Single phase flow
            solver = "simpleFoam"
        
        # Set up basic case structure
        self.case_manager.set_solver(solver)
        
        # Set up transport properties
        transport_model = TransportModel(self.case_manager)
        
        # Calculate mixture properties
        if solver == "multiphaseInterFoam":
            # Add phases
            # Oil phase
            oil_props = self.wellbore_flow.fluid_props.get_fluid_props_at_depth((section_start + section_end) / 2)
            transport_model.add_phase("oil", oil_props.get("density", 850.0), oil_props.get("viscosity", 0.01))
            
            # Water phase if needed
            if self.wellbore_flow.wc > 0.01:
                transport_model.add_phase("water", 1000.0, 0.001)
                transport_model.set_surface_tension("oil", "water", 0.02)
            
            # Gas phase if needed
            if self.wellbore_flow.gor > 0.01:
                # Estimate gas density at average pressure and temperature
                avg_md = (section_start + section_end) / 2
                pressure = self.wellbore_flow.fluid_props.get_pressure(avg_md)
                temperature = self.wellbore_flow.fluid_props.get_temperature(avg_md)
                
                # Gas density using ideal gas law (simplified)
                gas_density = 1.2 * (pressure / 101325) * (293.15 / temperature)
                
                transport_model.add_phase("gas", gas_density, 1.8e-5)
                transport_model.set_surface_tension("oil", "gas", 0.02)
                
                if self.wellbore_flow.wc > 0.01:
                    transport_model.set_surface_tension("water", "gas", 0.07)
        else:
            # Single phase
            oil_props = self.wellbore_flow.fluid_props.get_fluid_props_at_depth((section_start + section_end) / 2)
            transport_model.add_phase("oil", oil_props.get("density", 850.0), oil_props.get("viscosity", 0.01))
        
        # Write transport properties
        transport_model.write()
        
        # Set up boundary conditions
        # Inlet boundary
        inlet_patch = BoundaryCondition("inlet", "patch")
        
        if self.wellbore_flow.is_production:
            # Inlet at bottom of wellbore
            inlet_patch.set_location(f"pos().z < {-section_end + 0.1}")
        else:
            # Inlet at top of wellbore
            inlet_patch.set_location(f"pos().z > {-section_start - 0.1}")
        
        # Calculate flow velocity
        geo = self.wellbore.get_flow_path_geometry(section_start if self.wellbore_flow.is_production else section_end)
        area = geo["tubing_area"]
        
        if area > 0:
            velocity_magnitude = self.wellbore_flow.flow_rate / area
        else:
            velocity_magnitude = 1.0  # Default
        
        # Set velocity direction based on production/injection
        if self.wellbore_flow.is_production:
            # Flow direction from bottom to top
            inlet_patch.set_velocity([0, 0, velocity_magnitude])
        else:
            # Flow direction from top to bottom
            inlet_patch.set_velocity([0, 0, -velocity_magnitude])
        
        # Set phase fractions for multiphase flow
        if solver == "multiphaseInterFoam":
            # Calculate volume fractions
            total_volume = 1.0
            water_volume = self.wellbore_flow.wc * total_volume
            gas_volume = self.wellbore_flow.gor * (total_volume - water_volume)
            oil_volume = total_volume - water_volume - gas_volume
            
            # Set fractions
            inlet_patch.set_field("alpha.oil", oil_volume / total_volume)
            
            if self.wellbore_flow.wc > 0.01:
                inlet_patch.set_field("alpha.water", water_volume / total_volume)
            
            if self.wellbore_flow.gor > 0.01:
                inlet_patch.set_field("alpha.gas", gas_volume / total_volume)
        
        # Add inlet patch
        self.case_manager.add_boundary_condition(inlet_patch)
        
        # Outlet boundary
        outlet_patch = BoundaryCondition("outlet", "patch")
        
        if self.wellbore_flow.is_production:
            # Outlet at top of wellbore
            outlet_patch.set_location(f"pos().z > {-section_start - 0.1}")
            outlet_pressure = self.wellbore_flow.fluid_props.get_pressure(section_start)
        else:
            # Outlet at bottom of wellbore
            outlet_patch.set_location(f"pos().z < {-section_end + 0.1}")
            outlet_pressure = self.wellbore_flow.fluid_props.get_pressure(section_end)
        
        # Set pressure at outlet
        outlet_patch.set_pressure(outlet_pressure)
        
        # Add outlet patch
        self.case_manager.add_boundary_condition(outlet_patch)
        
        # Add wall boundary for tubing
        wall_patch = BoundaryCondition("wall", "wall")
        wall_patch.set_location("!(pos().z > -" + str(section_start) + " || pos().z < -" + str(section_end) + ")")
        
        # Set no-slip condition
        wall_patch.set_velocity([0, 0, 0])
        
        # Add wall patch
        self.case_manager.add_boundary_condition(wall_patch)
        
        # Set up perforation boundaries if present
        perforations_in_section = []
        for perf in self.wellbore.perforations:
            if section_start <= perf.md_top <= section_end or section_start <= perf.md_bottom <= section_end:
                perforations_in_section.append(perf)
        
        for i, perf in enumerate(perforations_in_section):
            # Create a patch for each perforation interval
            perf_patch = BoundaryCondition(f"perforation_{i}", "patch")
            
            # Set location based on perforation interval
            md_top = max(perf.md_top, section_start)
            md_bottom = min(perf.md_bottom, section_end)
            
            tvd_top, x_top, y_top = self.wellbore.trajectory.get_position(md_top)
            tvd_bottom, x_bottom, y_bottom = self.wellbore.trajectory.get_position(md_bottom)
            
            # Simplified location (in practice would need more precise geometry)
            perf_patch.set_location(
                f"pos().z >= {-md_bottom} && pos().z <= {-md_top} && mag(pos() - vector({x_top}, {y_top}, {-md_top})) < {perf.diameter/2 + 0.05}"
            )
            
            # Set conditions based on flow direction
            if self.wellbore_flow.is_production:
                # Inflow from reservoir
                inflow_velocity = 0.1  # m/s (simplified)
                perf_patch.set_velocity([inflow_velocity, 0, 0])  # Simplified direction
                
                # Set phase fractions for multiphase
                if solver == "multiphaseInterFoam":
                    perf_patch.set_field("alpha.oil", 1.0)  # Assume pure oil from reservoir
            else:
                # Outflow to reservoir
                perf_patch.set_pressure(self.wellbore_flow.fluid_props.get_pressure(md_top))
            
            # Add perforation patch
            self.case_manager.add_boundary_condition(perf_patch)
        
        # Write case files
        self.case_manager.write_case()
        
        return self.case_manager
    
    def run_simulation(self, parallel: bool = False, processors: int = 4):
        """
        Run the simulation.
        
        Args:
            parallel (bool): Whether to run in parallel
            processors (int): Number of processors for parallel run
            
        Returns:
            bool: True if simulation completed successfully, False otherwise
        """
        if not self.case_manager:
            raise ValueError("Case must be set up before running simulation")
        
        # Create solver manager
        solver_manager = SolverManager(self.case_manager)
        
        # Run the solver
        return solver_manager.run_solver(parallel=parallel, processors=processors)
    
    def extract_results(self) -> Dict[str, Any]:
        """
        Extract results from the simulation.
        
        Returns:
            Dict[str, Any]: Dictionary of results
        """
        if not self.case_manager:
            raise ValueError("Case must be set up and run before extracting results")
        
        # Get list of time directories
        time_dirs = self.case_manager.get_time_directories()
        if not time_dirs:
            return {}
        
        # Get the last time directory
        last_time_dir = time_dirs[-1]
        
        # Extract results
        results = {
            "time": float(os.path.basename(last_time_dir)),
            "velocity": {},
            "pressure": {},
            "phase_fractions": {}
        }
        
        # Read velocity field
        velocity_field = self.case_manager.read_field("U", last_time_dir)
        if velocity_field is not None:
            results["velocity"] = velocity_field
        
        # Read pressure field
        pressure_field = self.case_manager.read_field("p", last_time_dir)
        if pressure_field is not None:
            results["pressure"] = pressure_field
        
        # Read phase fractions if multiphase
        alpha_fields = []
        for phase in ["oil", "water", "gas"]:
            alpha_field = self.case_manager.read_field(f"alpha.{phase}", last_time_dir)
            if alpha_field is not None:
                alpha_fields.append((phase, alpha_field))
        
        results["phase_fractions"] = {phase: field for phase, field in alpha_fields}
        
        return results


# Entry point for command-line usage
if __name__ == "__main__":
    # Example usage
    print("Wellbore Modeling Module")
    
    # Create a simple vertical wellbore
    trajectory = WellTrajectory("Test Well", 0, 0, 0)
    trajectory.add_survey_point(0, 0, 0)
    trajectory.add_survey_point(1000, 0, 0)
    
    wellbore = Wellbore("Test Well", trajectory)
    
    # Add casing
    surface_casing = CasingSection(
        casing_type=CasingType.SURFACE,
        md_top=0,
        md_bottom=500,
        od=0.244,  # 9 5/8"
        id=0.226,
        weight=47.0,
        grade="J55"
    )
    
    production_casing = CasingSection(
        casing_type=CasingType.PRODUCTION,
        md_top=0,
        md_bottom=1000,
        od=0.178,  # 7"
        id=0.166,
        weight=29.0,
        grade="J55"
    )
    
    wellbore.add_casing_section(surface_casing)
    wellbore.add_casing_section(production_casing)
    
    # Add tubing
    tubing = TubingSection(
        md_top=0,
        md_bottom=1000,
        od=0.089,  # 3 1/2"
        id=0.076,
        weight=9.3,
        grade="J55"
    )
    
    wellbore.add_tubing_section(tubing)
    
    # Add perforations
    perf = Perforation(
        md_top=950,
        md_bottom=980,
        azimuth=0,
        phase_angle=0,
        diameter=0.01,
        density=12  # 12 shots per meter
    )
    
    wellbore.add_perforation(perf)
    
    # Print wellbore information
    print(f"Wellbore: {wellbore.name}")
    print(f"Total depth: {wellbore.total_depth} m")
    print(f"Number of casing sections: {len(wellbore.casing_sections)}")
    print(f"Number of tubing sections: {len(wellbore.tubing_sections)}")
    print(f"Number of perforations: {len(wellbore.perforations)}")
    
    # Export wellbore to JSON
    wellbore.export_to_json("test_wellbore.json")
    print("Wellbore exported to test_wellbore.json")
    
    # Create fluid properties
    temp_points = [(0, 293.15), (1000, 343.15)]  # 20°C to 70°C
    press_points = [(0, 1e6), (1000, 2e7)]  # 10 bar to 200 bar
    
    fluid_props = WellboreFluidProperties(temp_points, press_points)
    
    # Create flow model
    flow_model = WellboreFlow(wellbore, fluid_props)
    flow_model.set_production_rate(0.01, 0.2, 50)  # 10 L/s, 20% water, 50 m³/m³ GOR
    
    # Calculate flow profile
    profile = flow_model.calculate_flow_profile()
    
    print("\nFlow Profile:")
    print(f"Wellhead pressure: {profile['pressure'][0]/1e6:.2f} MPa")
    print(f"Bottomhole pressure: {profile['pressure'][-1]/1e6:.2f} MPa")
    print(f"Maximum velocity: {max(profile['velocity']):.2f} m/s")