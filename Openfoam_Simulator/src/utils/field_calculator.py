#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Field calculator for Openfoam_Simulator application.

This module provides utilities for performing calculations on scalar and vector
fields from CFD simulations, with a focus on oil & gas industry applications.
It supports:
- Basic field operations (gradient, divergence, curl)
- Statistical analysis of fields
- Derived quantities calculation
- Flow properties calculation
- Integration and averaging
- Specialized calculations for oil & gas applications
"""

import numpy as np
import math
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, TypeVar
import logging

# Import project modules
from .logger import get_logger
from .unit_converter import convert, PhysicalQuantity

# Set up logging
logger = get_logger(__name__)

# Type definitions for clarity
Scalar = float
Vector = Union[List[float], Tuple[float, float, float], np.ndarray]
ScalarField = np.ndarray  # 3D array of scalar values
VectorField = np.ndarray  # 4D array, where the last dimension is 3 (x,y,z)
Mesh = Any  # Placeholder for mesh representation

# Constants
GRAVITATIONAL_ACCELERATION = 9.81  # m/s²
STANDARD_TEMPERATURE = 288.15  # K (15°C)
STANDARD_PRESSURE = 101325.0  # Pa
GAS_CONSTANT = 8.31446  # J/(mol·K)


class FieldType(Enum):
    """Enumeration of field types."""
    SCALAR = auto()
    VECTOR = auto()
    TENSOR = auto()


class BoundaryCondition(Enum):
    """Enumeration of boundary condition types."""
    ZERO_GRADIENT = auto()
    FIXED_VALUE = auto()
    SYMMETRY = auto()
    CYCLIC = auto()
    WALL = auto()
    EMPTY = auto()


class FieldCalculator:
    """
    Calculator for operations on scalar and vector fields.
    """
    
    def __init__(self, mesh: Optional[Mesh] = None):
        """
        Initialize the field calculator.
        
        Args:
            mesh: The computational mesh (optional)
        """
        self.mesh = mesh
        self.cell_volumes = None
        self.face_areas = None
        self.cell_centers = None
        
        # Initialize field properties if mesh is provided
        if mesh is not None:
            self._initialize_mesh_properties()
            
    def _initialize_mesh_properties(self):
        """Initialize geometric properties from the mesh."""
        # This would extract cell volumes, face areas, centers, etc.
        # from the provided mesh data structure
        logger.debug("Initializing mesh properties")
        
        # Placeholder implementation - in a real application,
        # these would be calculated from the actual mesh
        try:
            # Example logic to extract properties from mesh
            # self.cell_volumes = extract_cell_volumes(self.mesh)
            # self.face_areas = extract_face_areas(self.mesh)
            # self.cell_centers = extract_cell_centers(self.mesh)
            pass
        except Exception as e:
            logger.error(f"Error initializing mesh properties: {e}")
            
    def set_mesh(self, mesh: Mesh):
        """
        Set or update the computational mesh.
        
        Args:
            mesh: The new computational mesh
        """
        self.mesh = mesh
        self._initialize_mesh_properties()
    
    # Basic field operations
    
    def gradient(self, field: ScalarField) -> VectorField:
        """
        Compute the gradient of a scalar field.
        
        Args:
            field: The scalar field
            
        Returns:
            VectorField: The gradient vector field
        """
        # Check if mesh is available
        if self.mesh is None:
            raise ValueError("Mesh not set. Call set_mesh() first.")
            
        logger.debug("Computing gradient of scalar field")
        
        # Placeholder for actual gradient calculation
        # This would use the mesh information to compute an accurate gradient
        # For now, use numpy's gradient as a simplified approximation
        grad_x, grad_y, grad_z = np.gradient(field)
        
        # Stack the components to form a vector field
        return np.stack([grad_x, grad_y, grad_z], axis=-1)
    
    def divergence(self, vector_field: VectorField) -> ScalarField:
        """
        Compute the divergence of a vector field.
        
        Args:
            vector_field: The vector field (shape [..., 3])
            
        Returns:
            ScalarField: The divergence scalar field
        """
        # Check if mesh is available
        if self.mesh is None:
            raise ValueError("Mesh not set. Call set_mesh() first.")
            
        logger.debug("Computing divergence of vector field")
        
        # Extract components
        u = vector_field[..., 0]
        v = vector_field[..., 1]
        w = vector_field[..., 2]
        
        # Compute partial derivatives
        du_dx, _, _ = np.gradient(u)
        _, dv_dy, _ = np.gradient(v)
        _, _, dw_dz = np.gradient(w)
        
        # Return divergence
        return du_dx + dv_dy + dw_dz
    
    def curl(self, vector_field: VectorField) -> VectorField:
        """
        Compute the curl of a vector field.
        
        Args:
            vector_field: The vector field (shape [..., 3])
            
        Returns:
            VectorField: The curl vector field
        """
        # Check if mesh is available
        if self.mesh is None:
            raise ValueError("Mesh not set. Call set_mesh() first.")
            
        logger.debug("Computing curl of vector field")
        
        # Extract components
        u = vector_field[..., 0]
        v = vector_field[..., 1]
        w = vector_field[..., 2]
        
        # Compute partial derivatives
        _, du_dy, du_dz = np.gradient(u)
        dv_dx, _, dv_dz = np.gradient(v)
        dw_dx, dw_dy, _ = np.gradient(w)
        
        # Compute curl components
        curl_x = dw_dy - dv_dz
        curl_y = du_dz - dw_dx
        curl_z = dv_dx - du_dy
        
        # Stack the components to form a vector field
        return np.stack([curl_x, curl_y, curl_z], axis=-1)
    
    def laplacian(self, field: ScalarField) -> ScalarField:
        """
        Compute the Laplacian of a scalar field.
        
        Args:
            field: The scalar field
            
        Returns:
            ScalarField: The Laplacian scalar field
        """
        # Check if mesh is available
        if self.mesh is None:
            raise ValueError("Mesh not set. Call set_mesh() first.")
            
        logger.debug("Computing Laplacian of scalar field")
        
        # Compute gradient and then divergence of gradient
        grad_field = self.gradient(field)
        return self.divergence(grad_field)
    
    def interpolate_to_points(self, field: Union[ScalarField, VectorField], 
                           points: np.ndarray) -> np.ndarray:
        """
        Interpolate field values to arbitrary points.
        
        Args:
            field: The field to interpolate
            points: Array of point coordinates, shape (n_points, 3)
            
        Returns:
            np.ndarray: Interpolated field values at the given points
        """
        # Check if mesh is available
        if self.mesh is None:
            raise ValueError("Mesh not set. Call set_mesh() first.")
            
        logger.debug(f"Interpolating field to {len(points)} points")
        
        # Placeholder for actual interpolation
        # This would use mesh-specific interpolation methods
        # For a structured grid, could use scipy.interpolate.RegularGridInterpolator
        
        # Return dummy values for now
        if len(field.shape) == 3:  # Scalar field
            return np.zeros(len(points))
        else:  # Vector field
            return np.zeros((len(points), 3))
    
    # Statistical functions
    
    def min_value(self, field: Union[ScalarField, VectorField]) -> Union[Scalar, Vector]:
        """
        Find the minimum value in a field.
        
        Args:
            field: The field to analyze
            
        Returns:
            Scalar or Vector: The minimum value
        """
        logger.debug("Computing minimum field value")
        
        if len(field.shape) == 3:  # Scalar field
            return np.min(field)
        else:  # Vector field
            # For vector field, return min magnitude and corresponding vector
            magnitudes = np.linalg.norm(field, axis=-1)
            min_idx = np.unravel_index(np.argmin(magnitudes), magnitudes.shape)
            return field[min_idx]
    
    def max_value(self, field: Union[ScalarField, VectorField]) -> Union[Scalar, Vector]:
        """
        Find the maximum value in a field.
        
        Args:
            field: The field to analyze
            
        Returns:
            Scalar or Vector: The maximum value
        """
        logger.debug("Computing maximum field value")
        
        if len(field.shape) == 3:  # Scalar field
            return np.max(field)
        else:  # Vector field
            # For vector field, return max magnitude and corresponding vector
            magnitudes = np.linalg.norm(field, axis=-1)
            max_idx = np.unravel_index(np.argmax(magnitudes), magnitudes.shape)
            return field[max_idx]
    
    def average(self, field: Union[ScalarField, VectorField], 
               weighted: bool = True) -> Union[Scalar, Vector]:
        """
        Compute the average of a field.
        
        Args:
            field: The field to analyze
            weighted: Whether to weight by cell volumes (True) or use simple average (False)
            
        Returns:
            Scalar or Vector: The average value
        """
        logger.debug("Computing field average")
        
        if weighted and self.cell_volumes is not None:
            # Volume-weighted average
            if len(field.shape) == 3:  # Scalar field
                total = np.sum(field * self.cell_volumes)
                return total / np.sum(self.cell_volumes)
            else:  # Vector field
                total = np.zeros(3)
                for i in range(3):
                    total[i] = np.sum(field[..., i] * self.cell_volumes)
                return total / np.sum(self.cell_volumes)
        else:
            # Simple average
            if len(field.shape) == 3:  # Scalar field
                return np.mean(field)
            else:  # Vector field
                return np.mean(field, axis=(0, 1, 2))
    
    def rms(self, field: Union[ScalarField, VectorField]) -> Scalar:
        """
        Compute the root mean square of a field.
        
        Args:
            field: The field to analyze
            
        Returns:
            Scalar: The RMS value
        """
        logger.debug("Computing field RMS")
        
        if len(field.shape) == 3:  # Scalar field
            return np.sqrt(np.mean(np.square(field)))
        else:  # Vector field
            # For vector field, compute RMS of magnitude
            magnitudes = np.linalg.norm(field, axis=-1)
            return np.sqrt(np.mean(np.square(magnitudes)))
    
    def histogram(self, field: ScalarField, bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute histogram of a scalar field.
        
        Args:
            field: The scalar field to analyze
            bins: Number of histogram bins
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: The bin edges and histogram values
        """
        logger.debug(f"Computing field histogram with {bins} bins")
        
        if len(field.shape) != 3:  # Must be a scalar field
            raise ValueError("Histogram calculation requires a scalar field")
            
        # Flatten the field and compute histogram
        flat_field = field.flatten()
        hist, bin_edges = np.histogram(flat_field, bins=bins)
        
        return bin_edges, hist
    
    # Integration functions
    
    def integrate(self, field: ScalarField) -> Scalar:
        """
        Integrate a scalar field over the domain.
        
        Args:
            field: The scalar field to integrate
            
        Returns:
            Scalar: The integrated value
        """
        # Check if mesh is available
        if self.mesh is None or self.cell_volumes is None:
            raise ValueError("Mesh not set or cell volumes not available.")
            
        logger.debug("Integrating scalar field over domain")
        
        # Multiply field by cell volumes and sum
        return np.sum(field * self.cell_volumes)
    
    def surface_integral(self, field: Union[ScalarField, VectorField], 
                        boundary_name: str) -> Union[Scalar, Vector]:
        """
        Integrate a field over a boundary surface.
        
        Args:
            field: The field to integrate
            boundary_name: Name of the boundary surface
            
        Returns:
            Scalar or Vector: The integrated value
        """
        # Check if mesh is available
        if self.mesh is None:
            raise ValueError("Mesh not set. Call set_mesh() first.")
            
        logger.debug(f"Integrating field over boundary {boundary_name}")
        
        # Placeholder for actual boundary integration
        # This would need to:
        # 1. Identify faces belonging to the named boundary
        # 2. Get face areas and field values at faces
        # 3. Multiply and sum
        
        # Return dummy value for now
        if len(field.shape) == 3:  # Scalar field
            return 0.0
        else:  # Vector field
            return np.zeros(3)
    
    def mass_flow_rate(self, velocity_field: VectorField, 
                     density_field: ScalarField, 
                     boundary_name: str) -> Scalar:
        """
        Calculate mass flow rate through a boundary.
        
        Args:
            velocity_field: The velocity vector field
            density_field: The density scalar field
            boundary_name: Name of the boundary surface
            
        Returns:
            Scalar: The mass flow rate (kg/s)
        """
        # Check if mesh is available
        if self.mesh is None:
            raise ValueError("Mesh not set. Call set_mesh() first.")
            
        logger.debug(f"Calculating mass flow rate through boundary {boundary_name}")
        
        # Placeholder for actual mass flow calculation
        # This would:
        # 1. Interpolate velocity and density to boundary faces
        # 2. Compute dot product of velocity with face normal vectors
        # 3. Multiply by density and face area
        # 4. Sum over all boundary faces
        
        # Return dummy value for now
        return 0.0
    
    # Oil & Gas specific calculations
    
    def reynolds_number(self, velocity_field: VectorField, 
                       density_field: ScalarField,
                       viscosity_field: ScalarField, 
                       characteristic_length: float) -> ScalarField:
        """
        Calculate Reynolds number field.
        
        Args:
            velocity_field: The velocity vector field
            density_field: The density scalar field
            viscosity_field: The viscosity scalar field
            characteristic_length: Characteristic length scale
            
        Returns:
            ScalarField: The Reynolds number field
        """
        logger.debug("Calculating Reynolds number field")
        
        # Compute velocity magnitude
        velocity_magnitude = np.linalg.norm(velocity_field, axis=-1)
        
        # Reynolds number = (rho * V * L) / mu
        return (density_field * velocity_magnitude * characteristic_length) / viscosity_field
    
    def pressure_drop(self, pressure_field: ScalarField, 
                    point1: Vector, point2: Vector) -> Scalar:
        """
        Calculate pressure drop between two points.
        
        Args:
            pressure_field: The pressure scalar field
            point1: First point coordinates
            point2: Second point coordinates
            
        Returns:
            Scalar: The pressure drop (p1 - p2)
        """
        # Check if mesh is available
        if self.mesh is None:
            raise ValueError("Mesh not set. Call set_mesh() first.")
            
        logger.debug(f"Calculating pressure drop between points {point1} and {point2}")
        
        # Interpolate pressure to the specified points
        p1 = self.interpolate_to_points(pressure_field, np.array([point1]))[0]
        p2 = self.interpolate_to_points(pressure_field, np.array([point2]))[0]
        
        # Return pressure drop
        return p1 - p2
    
    def darcy_friction_factor(self, reynolds_number: Union[Scalar, ScalarField],
                            relative_roughness: float) -> Union[Scalar, ScalarField]:
        """
        Calculate Darcy friction factor for pipe flow.
        
        Args:
            reynolds_number: The Reynolds number
            relative_roughness: Relative roughness (e/D)
            
        Returns:
            Scalar or ScalarField: The Darcy friction factor
        """
        logger.debug("Calculating Darcy friction factor")
        
        # Check flow regime
        if np.any(reynolds_number < 2300):  # Laminar flow
            # Laminar flow: f = 64/Re
            if isinstance(reynolds_number, np.ndarray):
                f = np.zeros_like(reynolds_number)
                laminar = reynolds_number < 2300
                f[laminar] = 64.0 / reynolds_number[laminar]
            else:
                f = 64.0 / reynolds_number
                
            # Handle transitional and turbulent flow if needed
            if isinstance(reynolds_number, np.ndarray) and np.any(reynolds_number >= 2300):
                # Use Colebrook equation for turbulent flow
                turbulent = reynolds_number >= 2300
                # Iterative solution (simplified)
                f_turb = np.ones_like(reynolds_number[turbulent]) * 0.02  # Initial guess
                for _ in range(10):  # Max 10 iterations
                    f_turb = 1.0 / (-2.0 * np.log10(relative_roughness/3.7 + 2.51/(reynolds_number[turbulent]*np.sqrt(f_turb))))**2
                f[turbulent] = f_turb
                
            return f
        else:  # Turbulent flow only
            # Use Colebrook equation for turbulent flow
            # This requires an iterative solution (simplified)
            f = 0.02  # Initial guess
            for _ in range(10):  # Max 10 iterations
                f = 1.0 / (-2.0 * np.log10(relative_roughness/3.7 + 2.51/(reynolds_number*np.sqrt(f))))**2
            return f
    
    def phase_fraction_gradient(self, phase_fraction: ScalarField) -> VectorField:
        """
        Calculate gradient of phase fraction for interface tracking.
        
        Args:
            phase_fraction: The phase fraction scalar field (0-1)
            
        Returns:
            VectorField: The phase fraction gradient
        """
        logger.debug("Calculating phase fraction gradient")
        
        # Compute gradient
        return self.gradient(phase_fraction)
    
    def interface_curvature(self, phase_fraction: ScalarField) -> ScalarField:
        """
        Calculate interface curvature from phase fraction field.
        
        Args:
            phase_fraction: The phase fraction scalar field (0-1)
            
        Returns:
            ScalarField: The interface curvature field
        """
        logger.debug("Calculating interface curvature")
        
        # Compute gradient of phase fraction
        grad_alpha = self.gradient(phase_fraction)
        
        # Normalize gradient to get interface normal
        grad_magnitude = np.linalg.norm(grad_alpha, axis=-1)
        # Avoid division by zero
        mask = grad_magnitude > 1e-10
        interface_normal = np.zeros_like(grad_alpha)
        interface_normal[mask] = grad_alpha[mask] / grad_magnitude[mask, np.newaxis]
        
        # Compute divergence of normal for curvature
        return -self.divergence(interface_normal)
    
    def liquid_holdup(self, liquid_fraction: ScalarField, 
                    gas_velocity: VectorField, 
                    liquid_velocity: VectorField) -> ScalarField:
        """
        Calculate liquid holdup for multiphase pipe flow.
        
        Args:
            liquid_fraction: The liquid phase fraction field
            gas_velocity: The gas phase velocity field
            liquid_velocity: The liquid phase velocity field
            
        Returns:
            ScalarField: The liquid holdup field
        """
        logger.debug("Calculating liquid holdup")
        
        # Compute velocity magnitudes
        gas_vel_mag = np.linalg.norm(gas_velocity, axis=-1)
        liquid_vel_mag = np.linalg.norm(liquid_velocity, axis=-1)
        
        # Compute mixture velocity
        mixture_velocity = gas_vel_mag * (1.0 - liquid_fraction) + liquid_vel_mag * liquid_fraction
        
        # Compute slip ratio
        slip_ratio = np.zeros_like(liquid_fraction)
        mask = liquid_vel_mag > 1e-10
        slip_ratio[mask] = gas_vel_mag[mask] / liquid_vel_mag[mask]
        
        # Compute holdup
        holdup = liquid_fraction * (1.0 + (1.0 - liquid_fraction) * (slip_ratio - 1.0))
        return holdup
    
    def pressure_gradient_components(self, pressure_field: ScalarField, 
                                   velocity_field: VectorField,
                                   density_field: ScalarField,
                                   viscosity_field: ScalarField,
                                   pipe_diameter: float, 
                                   inclination: float = 0.0) -> Dict[str, ScalarField]:
        """
        Decompose pressure gradient into friction, gravity, and acceleration components.
        
        Args:
            pressure_field: The pressure scalar field
            velocity_field: The velocity vector field
            density_field: The density scalar field
            viscosity_field: The viscosity scalar field
            pipe_diameter: Pipe diameter in meters
            inclination: Pipe inclination angle in degrees
            
        Returns:
            Dict[str, ScalarField]: Dictionary with pressure gradient components
        """
        logger.debug("Calculating pressure gradient components")
        
        # Convert inclination to radians
        inclination_rad = inclination * math.pi / 180.0
        
        # Calculate total pressure gradient
        total_grad = self.gradient(pressure_field)
        
        # Calculate friction component
        reynolds = self.reynolds_number(
            velocity_field, density_field, viscosity_field, pipe_diameter)
        friction_factor = self.darcy_friction_factor(reynolds, 0.0001)  # Assume smooth pipe
        
        velocity_magnitude = np.linalg.norm(velocity_field, axis=-1)
        # dp/dx due to friction = -f * rho * v^2 / (2 * D)
        friction_grad = friction_factor * density_field * np.square(velocity_magnitude) / (2 * pipe_diameter)
        
        # Create direction vector based on velocity
        velocity_direction = np.zeros_like(velocity_field)
        mask = velocity_magnitude > 1e-10
        velocity_direction[mask] = velocity_field[mask] / velocity_magnitude[mask, np.newaxis]
        
        # Apply to get friction gradient vector
        friction_grad_vector = -friction_grad[..., np.newaxis] * velocity_direction
        
        # Calculate gravity component
        # dp/dx due to gravity = rho * g * sin(inclination)
        gravity_grad = density_field * GRAVITATIONAL_ACCELERATION * math.sin(inclination_rad)
        
        # Gravity acts in the z-direction
        gravity_direction = np.zeros_like(velocity_field)
        gravity_direction[..., 2] = 1.0  # z-component
        
        gravity_grad_vector = gravity_grad[..., np.newaxis] * gravity_direction
        
        # Calculate acceleration component (residual)
        accel_grad_vector = total_grad - friction_grad_vector - gravity_grad_vector
        
        # Package results
        return {
            "total": total_grad,
            "friction": friction_grad_vector,
            "gravity": gravity_grad_vector,
            "acceleration": accel_grad_vector
        }
    
    def erosion_rate(self, velocity_field: VectorField, 
                   particle_concentration: ScalarField,
                   particle_diameter: float,
                   particle_density: float,
                   target_material: str = "carbon_steel") -> ScalarField:
        """
        Calculate erosion rate based on velocity and particle properties.
        
        Args:
            velocity_field: The velocity vector field
            particle_concentration: Particle concentration field
            particle_diameter: Particle diameter in meters
            particle_density: Particle density in kg/m³
            target_material: Target material type
            
        Returns:
            ScalarField: The erosion rate field
        """
        logger.debug(f"Calculating erosion rate for {target_material}")
        
        # Material properties (simplified model parameters)
        material_params = {
            "carbon_steel": {"k": 2.0e-9, "n": 2.6},
            "stainless_steel": {"k": 5.5e-10, "n": 2.6},
            "aluminum": {"k": 5.0e-9, "n": 2.3},
            "copper": {"k": 1.5e-9, "n": 2.3}
        }
        
        if target_material not in material_params:
            raise ValueError(f"Unknown material: {target_material}")
            
        k = material_params[target_material]["k"]
        n = material_params[target_material]["n"]
        
        # Calculate velocity magnitude
        velocity_magnitude = np.linalg.norm(velocity_field, axis=-1)
        
        # Calculate erosion rate using simplified power-law model:
        # E = k * C * rho_p * V^n * d_p
        erosion_rate = k * particle_concentration * particle_density * \
                      np.power(velocity_magnitude, n) * particle_diameter
                      
        return erosion_rate
    
    def flow_pattern_identification(self, liquid_fraction: ScalarField,
                                  gas_velocity: VectorField,
                                  liquid_velocity: VectorField,
                                  pipe_diameter: float) -> np.ndarray:
        """
        Identify multiphase flow patterns.
        
        Args:
            liquid_fraction: The liquid phase fraction field
            gas_velocity: The gas phase velocity field
            liquid_velocity: The liquid phase velocity field
            pipe_diameter: Pipe diameter in meters
            
        Returns:
            np.ndarray: Flow pattern classification field
                        (0: Bubbly, 1: Slug, 2: Churn, 3: Annular)
        """
        logger.debug("Identifying multiphase flow patterns")
        
        # Initialize flow pattern field (same shape as liquid_fraction)
        # 0: Bubbly, 1: Slug, 2: Churn, 3: Annular
        flow_pattern = np.zeros_like(liquid_fraction, dtype=np.int32)
        
        # Calculate superficial velocities
        gas_vel_mag = np.linalg.norm(gas_velocity, axis=-1)
        liquid_vel_mag = np.linalg.norm(liquid_velocity, axis=-1)
        
        # Calculate superficial velocities
        j_g = gas_vel_mag * (1.0 - liquid_fraction)  # Gas superficial velocity
        j_l = liquid_vel_mag * liquid_fraction       # Liquid superficial velocity
        
        # Simplified flow pattern map based on superficial velocities
        # This is a basic implementation and would need to be refined
        # for accurate flow pattern prediction
        
        # Bubbly flow
        bubbly = (j_l > 0.5) & (j_g < 0.5)
        flow_pattern[bubbly] = 0
        
        # Slug flow
        slug = (j_l > 0.1) & (j_l <= 0.5) & (j_g >= 0.5) & (j_g < 10)
        flow_pattern[slug] = 1
        
        # Churn flow
        churn = (j_l > 0.1) & (j_g >= 10) & (j_g < 20)
        flow_pattern[churn] = 2
        
        # Annular flow
        annular = (j_g >= 20) | ((j_l <= 0.1) & (j_g >= 0.5))
        flow_pattern[annular] = 3
        
        return flow_pattern
    
    def pigging_efficiency(self, liquid_film_thickness: ScalarField,
                         pig_diameter: float,
                         pipe_diameter: float) -> ScalarField:
        """
        Calculate pigging efficiency based on liquid film thickness.
        
        Args:
            liquid_film_thickness: Liquid film thickness field
            pig_diameter: Pig diameter in meters
            pipe_diameter: Pipe diameter in meters
            
        Returns:
            ScalarField: The pigging efficiency field (0-1)
        """
        logger.debug("Calculating pigging efficiency")
        
        # Calculate clearance between pig and pipe wall
        clearance = (pipe_diameter - pig_diameter) / 2
        
        # Calculate efficiency based on film thickness and clearance
        # Efficiency = max(0, min(1, (film_thickness - clearance) / film_thickness))
        efficiency = np.zeros_like(liquid_film_thickness)
        mask = liquid_film_thickness > 0
        efficiency[mask] = np.maximum(0, np.minimum(1, 
                          (liquid_film_thickness[mask] - clearance) / liquid_film_thickness[mask]))
        
        return efficiency
    
    # Utility methods
    
    def extract_isosurface(self, field: ScalarField, 
                         isovalue: float) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract an isosurface from a scalar field.
        
        Args:
            field: The scalar field
            isovalue: The isovalue to extract
            
        Returns:
            Dict or None: Dictionary with vertices and faces for the isosurface,
                         or None if the mesh is not available
        """
        # This is a placeholder for actual isosurface extraction
        # In a real implementation, this would use something like marching cubes
        # and would return vertices and faces for the isosurface
        logger.debug(f"Extracting isosurface at value {isovalue}")
        
        # A simplified version might look like:
        # from skimage import measure
        # verts, faces, _, _ = measure.marching_cubes(field, isovalue)
        # return {"vertices": verts, "faces": faces}
        
        return None
    
    def compute_streamlines(self, vector_field: VectorField, 
                          start_points: np.ndarray,
                          max_steps: int = 1000,
                          step_size: float = 0.01) -> List[np.ndarray]:
        """
        Compute streamlines through a vector field.
        
        Args:
            vector_field: The vector field
            start_points: Starting points for streamlines, shape (n_points, 3)
            max_steps: Maximum number of steps
            step_size: Integration step size
            
        Returns:
            List[np.ndarray]: List of streamline point arrays
        """
        # Check if mesh is available
        if self.mesh is None:
            raise ValueError("Mesh not set. Call set_mesh() first.")
            
        logger.debug(f"Computing {len(start_points)} streamlines")
        
        # Placeholder for actual streamline computation
        # This would use a numerical integration method (e.g., RK4)
        # to trace streamlines through the vector field
        
        # Return dummy streamlines for now
        streamlines = []
        for point in start_points:
            # Create a dummy straight line
            line = np.zeros((10, 3))
            line[:, 0] = point[0] + np.linspace(0, 1, 10)
            line[:, 1] = point[1]
            line[:, 2] = point[2]
            streamlines.append(line)
            
        return streamlines


# Convenience functions

def compute_vorticity(velocity_field: VectorField, mesh: Optional[Mesh] = None) -> VectorField:
    """
    Compute vorticity (curl of velocity) for a velocity field.
    
    Args:
        velocity_field: The velocity vector field
        mesh: Optional mesh for more accurate calculation
        
    Returns:
        VectorField: The vorticity field
    """
    calculator = FieldCalculator(mesh)
    return calculator.curl(velocity_field)


def compute_q_criterion(velocity_field: VectorField, mesh: Optional[Mesh] = None) -> ScalarField:
    """
    Compute Q-criterion for vortex identification.
    
    Args:
        velocity_field: The velocity vector field
        mesh: Optional mesh for more accurate calculation
        
    Returns:
        ScalarField: The Q-criterion field
    """
    # Check dimensions of velocity field
    if len(velocity_field.shape) != 4 or velocity_field.shape[-1] != 3:
        raise ValueError("Velocity field must have shape (..., 3)")
    
    # Create calculator
    calculator = FieldCalculator(mesh)
    
    # Extract velocity components
    u = velocity_field[..., 0]
    v = velocity_field[..., 1]
    w = velocity_field[..., 2]
    
    # Compute velocity gradients
    du_dx, du_dy, du_dz = np.gradient(u)
    dv_dx, dv_dy, dv_dz = np.gradient(v)
    dw_dx, dw_dy, dw_dz = np.gradient(w)
    
    # Compute rate-of-strain tensor S and vorticity tensor Ω
    # S = 0.5 * (∇u + (∇u)ᵀ)
    # Ω = 0.5 * (∇u - (∇u)ᵀ)
    
    # Compute tensor norms (Frobenius norm)
    # |Ω|² = tr(ΩΩᵀ)
    omega_squared = 0.5 * (
        (dw_dy - dv_dz)**2 + (du_dz - dw_dx)**2 + (dv_dx - du_dy)**2
    )
    
    # |S|² = tr(SSᵀ)
    strain_squared = 0.5 * (
        (du_dx)**2 + (dv_dy)**2 + (dw_dz)**2 +
        0.5 * ((du_dy + dv_dx)**2 + (du_dz + dw_dx)**2 + (dv_dz + dw_dy)**2)
    )
    
    # Q = 0.5 * (|Ω|² - |S|²)
    q_criterion = 0.5 * (omega_squared - strain_squared)
    
    return q_criterion


def compute_pressure_coefficient(pressure_field: ScalarField,
                              reference_pressure: float,
                              velocity_field: VectorField,
                              density: float) -> ScalarField:
    """
    Compute pressure coefficient.
    
    Args:
        pressure_field: The pressure scalar field
        reference_pressure: Reference pressure (e.g., free-stream)
        velocity_field: The velocity vector field
        density: Fluid density
        
    Returns:
        ScalarField: The pressure coefficient field
    """
    # Compute velocity magnitude
    velocity_magnitude = np.linalg.norm(velocity_field, axis=-1)
    
    # Dynamic pressure = 0.5 * rho * V²
    dynamic_pressure = 0.5 * density * np.square(velocity_magnitude)
    
    # Cp = (p - p_ref) / q
    # Avoid division by zero
    mask = dynamic_pressure > 1e-10
    cp = np.zeros_like(pressure_field)
    cp[mask] = (pressure_field[mask] - reference_pressure) / dynamic_pressure[mask]
    
    return cp


def identify_shocks(mach_field: ScalarField,
                  pressure_field: ScalarField,
                  threshold: float = 0.5) -> np.ndarray:
    """
    Identify shock waves in compressible flow.
    
    Args:
        mach_field: The Mach number scalar field
        pressure_field: The pressure scalar field
        threshold: Pressure gradient threshold for shock detection
        
    Returns:
        np.ndarray: Boolean array indicating shock locations
    """
    # Compute pressure gradient magnitude
    px, py, pz = np.gradient(pressure_field)
    pressure_grad_mag = np.sqrt(px**2 + py**2 + pz**2)
    
    # Normalize by local pressure for better thresholding
    normalized_grad = pressure_grad_mag / np.maximum(1e-10, pressure_field)
    
    # Identify potential shock regions
    shock_candidates = normalized_grad > threshold
    
    # Only consider supersonic regions
    shocks = shock_candidates & (mach_field > 1.0)
    
    return shocks


# Example usage
if __name__ == "__main__":
    # Create simple 3D fields for testing
    nx, ny, nz = 50, 50, 50
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create a scalar field (pressure)
    pressure = 101325 + 1000 * (X**2 + Y**2 + Z**2)
    
    # Create a vector field (velocity)
    u = -Y
    v = X
    w = 0.1 * Z
    velocity = np.stack([u, v, w], axis=-1)
    
    # Create a field calculator
    calculator = FieldCalculator()
    
    # Compute gradient of pressure
    pressure_grad = calculator.gradient(pressure)
    print(f"Pressure gradient shape: {pressure_grad.shape}")
    
    # Compute curl of velocity (vorticity)
    vorticity = calculator.curl(velocity)
    print(f"Vorticity shape: {vorticity.shape}")
    
    # Compute average pressure
    avg_pressure = calculator.average(pressure)
    print(f"Average pressure: {avg_pressure:.2f} Pa")
    
    # Compute Q-criterion
    q = compute_q_criterion(velocity)
    print(f"Q-criterion shape: {q.shape}")
    
    # Find maximum vorticity magnitude
    vorticity_mag = np.linalg.norm(vorticity, axis=-1)
    max_vorticity = np.max(vorticity_mag)
    print(f"Maximum vorticity magnitude: {max_vorticity:.2f} 1/s")