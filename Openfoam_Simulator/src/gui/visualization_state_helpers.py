#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper methods for saving and restoring visualization state in Openfoam_Simulator.

This module provides functions to interact with simulation_controls, viewport,
and other UI components to properly save and restore their state when reopening projects.
"""

import os
import json
import logging
from pathlib import Path

# Setup logger
logger = logging.getLogger(__name__)

class VisualizationStateHelpers:
    """
    Provides helper methods for saving and restoring visualization state in Openfoam_Simulator.
    """
    
    @staticmethod
    def save_boundary_conditions(simulation_controls):
        """
        Extract boundary conditions from simulation controls.
        
        Args:
            simulation_controls: The simulation controls panel
            
        Returns:
            dict: Boundary condition settings or None if not available
        """
        try:
            # Check if boundary condition specific methods exist
            if hasattr(simulation_controls, 'boundary_panel') and hasattr(simulation_controls.boundary_panel, 'get_boundary_values'):
                return simulation_controls.boundary_panel.get_boundary_values()
            
            # Try to access boundary_conditions attribute directly
            if hasattr(simulation_controls, 'boundary_conditions'):
                return simulation_controls.boundary_conditions
            
            # Try to access via OpenFOAM case manager (more common approach)
            if hasattr(simulation_controls, 'case_manager') and simulation_controls.case_manager:
                if hasattr(simulation_controls.case_manager, 'get_boundary_conditions'):
                    return simulation_controls.case_manager.get_boundary_conditions()
                    
                # Also check if boundary_conditions is in case_manager
                if hasattr(simulation_controls.case_manager, 'boundary_conditions'):
                    return simulation_controls.case_manager.boundary_conditions
            
            # Check for boundary data in boundary panels
            boundary_data = {}
            
            # Look for common boundary UI panel naming patterns
            for attr_name in dir(simulation_controls):
                if 'boundary' in attr_name.lower() and hasattr(simulation_controls, attr_name):
                    panel = getattr(simulation_controls, attr_name)
                    # If it's a panel with values, try to extract
                    if hasattr(panel, 'get_values'):
                        boundary_data[attr_name] = panel.get_values()
            
            if boundary_data:
                return boundary_data
                
            # Last resort: Check for boundary_ prefixed variables directly
            boundary_data = {}
            for attr_name in dir(simulation_controls):
                if attr_name.startswith('boundary_') and not callable(getattr(simulation_controls, attr_name)):
                    boundary_data[attr_name] = getattr(simulation_controls, attr_name)
            
            return boundary_data if boundary_data else None
            
        except Exception as e:
            logger.error(f"Error extracting boundary conditions: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def restore_boundary_conditions(simulation_controls, boundary_data):
        """
        Restore boundary conditions to simulation controls.
        
        Args:
            simulation_controls: The simulation controls panel
            boundary_data: Boundary condition data to restore
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not boundary_data:
                return False
                
            # Check if boundary condition specific methods exist
            if hasattr(simulation_controls, 'boundary_panel') and hasattr(simulation_controls.boundary_panel, 'set_boundary_values'):
                simulation_controls.boundary_panel.set_boundary_values(boundary_data)
                return True
            
            # Try to access boundary_conditions attribute directly
            if hasattr(simulation_controls, 'boundary_conditions'):
                simulation_controls.boundary_conditions = boundary_data
                
                # If there's a refresh or update method, call it
                if hasattr(simulation_controls, 'update_boundary_ui'):
                    simulation_controls.update_boundary_ui()
                    
                return True
            
            # Try to access via OpenFOAM case manager (more common approach)
            if hasattr(simulation_controls, 'case_manager') and simulation_controls.case_manager:
                if hasattr(simulation_controls.case_manager, 'set_boundary_conditions'):
                    simulation_controls.case_manager.set_boundary_conditions(boundary_data)
                    return True
                    
                # Also check if boundary_conditions is in case_manager
                if hasattr(simulation_controls.case_manager, 'boundary_conditions'):
                    simulation_controls.case_manager.boundary_conditions = boundary_data
                    
                    # If there's a refresh method
                    if hasattr(simulation_controls.case_manager, 'update_boundary_files'):
                        simulation_controls.case_manager.update_boundary_files()
                        
                    return True
            
            # Check for boundary data in boundary panels
            success = False
            
            # Look for common boundary UI panel naming patterns
            for attr_name in dir(simulation_controls):
                if 'boundary' in attr_name.lower() and hasattr(simulation_controls, attr_name):
                    panel = getattr(simulation_controls, attr_name)
                    # If it's a panel with values, try to extract
                    if attr_name in boundary_data and hasattr(panel, 'set_values'):
                        panel.set_values(boundary_data[attr_name])
                        success = True
            
            if success:
                return True
                
            # Last resort: Directly set boundary_ prefixed variables
            for key, value in boundary_data.items():
                if hasattr(simulation_controls, key):
                    setattr(simulation_controls, key, value)
                    success = True
            
            return success
            
        except Exception as e:
            logger.error(f"Error restoring boundary conditions: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    @staticmethod
    def save_solver_parameters(simulation_controls):
        """
        Extract solver parameters from simulation controls.
        
        Args:
            simulation_controls: The simulation controls panel
            
        Returns:
            dict: Solver parameter settings or None if not available
        """
        try:
            # Check if solver parameter specific methods exist
            if hasattr(simulation_controls, 'solver_panel') and hasattr(simulation_controls.solver_panel, 'get_solver_values'):
                return simulation_controls.solver_panel.get_solver_values()
            
            # Try to directly access solver parameters
            if hasattr(simulation_controls, 'solver_parameters'):
                return simulation_controls.solver_parameters
            
            # Try to get solver settings from case manager
            if hasattr(simulation_controls, 'case_manager') and simulation_controls.case_manager:
                if hasattr(simulation_controls.case_manager, 'get_solver_settings'):
                    return simulation_controls.case_manager.get_solver_settings()
                
                # Check for system/controlDict getters
                if hasattr(simulation_controls.case_manager, 'get_control_dict'):
                    return simulation_controls.case_manager.get_control_dict()
                
            # Generic approach to save any existing solver parameters
            solver_data = {}
            
            # Look for specific solver widgets
            for attr_name in dir(simulation_controls):
                # Check for any solver UI component
                if 'solver' in attr_name.lower() and hasattr(simulation_controls, attr_name):
                    panel = getattr(simulation_controls, attr_name)
                    # If it has a get_values method, use it
                    if hasattr(panel, 'get_values'):
                        solver_data[attr_name] = panel.get_values()
            
            # Common solver parameter attributes
            solver_attrs = [
                'solver_type', 'turbulence_model', 'time_step', 'end_time',
                'viscosity_model', 'pressure_dimensions', 'start_time',
                'max_iterations', 'write_interval', 'delta_t'
            ]
            
            for attr_name in solver_attrs:
                if hasattr(simulation_controls, attr_name):
                    solver_data[attr_name] = getattr(simulation_controls, attr_name)
            
            return solver_data if solver_data else None
            
        except Exception as e:
            logger.error(f"Error extracting solver parameters: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def restore_solver_parameters(simulation_controls, solver_data):
        """
        Restore solver parameters to simulation controls.
        
        Args:
            simulation_controls: The simulation controls panel
            solver_data: Solver parameter data to restore
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not solver_data:
                return False
                
            # Check if solver parameter specific methods exist
            if hasattr(simulation_controls, 'solver_panel') and hasattr(simulation_controls.solver_panel, 'set_solver_values'):
                simulation_controls.solver_panel.set_solver_values(solver_data)
                return True
            
            # Try to directly set solver parameters
            if hasattr(simulation_controls, 'solver_parameters'):
                simulation_controls.solver_parameters = solver_data
                return True
                
            # Generic approach to restore any solver parameters
            for key, value in solver_data.items():
                if hasattr(simulation_controls, key):
                    setattr(simulation_controls, key, value)
            
            # If there's a UI update method, call it
            if hasattr(simulation_controls, 'update_solver_ui'):
                simulation_controls.update_solver_ui()
            
            return True
            
        except Exception as e:
            logger.error(f"Error restoring solver parameters: {str(e)}")
            return False
    
    @staticmethod
    def save_ambient_settings(simulation_controls):
        """
        Extract ambient region settings from simulation controls.
        
        Args:
            simulation_controls: The simulation controls panel
            
        Returns:
            dict: Ambient region settings or None if not available
        """
        try:
            # Check if ambient region specific methods exist
            if hasattr(simulation_controls, 'ambient_panel') and hasattr(simulation_controls.ambient_panel, 'get_ambient_values'):
                return simulation_controls.ambient_panel.get_ambient_values()
            
            # Try to directly access ambient settings
            if hasattr(simulation_controls, 'ambient_settings'):
                return simulation_controls.ambient_settings
                
            # Generic approach to save any existing ambient settings
            ambient_data = {}
            # Common ambient setting attributes
            ambient_attrs = [
                'fluid_type', 'ambient_temperature', 'ambient_pressure',
                'fluid_color', 'fluid_opacity'
            ]
            
            for attr_name in ambient_attrs:
                if hasattr(simulation_controls, attr_name):
                    ambient_data[attr_name] = getattr(simulation_controls, attr_name)
            
            return ambient_data if ambient_data else None
            
        except Exception as e:
            logger.error(f"Error extracting ambient settings: {str(e)}")
            return None
    
    @staticmethod
    def restore_ambient_settings(simulation_controls, ambient_data):
        """
        Restore ambient region settings to simulation controls.
        
        Args:
            simulation_controls: The simulation controls panel
            ambient_data: Ambient region data to restore
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not ambient_data:
                return False
                
            # Check if ambient region specific methods exist
            if hasattr(simulation_controls, 'ambient_panel') and hasattr(simulation_controls.ambient_panel, 'set_ambient_values'):
                simulation_controls.ambient_panel.set_ambient_values(ambient_data)
                return True
            
            # Try to directly set ambient settings
            if hasattr(simulation_controls, 'ambient_settings'):
                simulation_controls.ambient_settings = ambient_data
                return True
                
            # Generic approach to restore any ambient settings
            for key, value in ambient_data.items():
                if hasattr(simulation_controls, key):
                    setattr(simulation_controls, key, value)
            
            # If there's a UI update method, call it
            if hasattr(simulation_controls, 'update_ambient_ui'):
                simulation_controls.update_ambient_ui()
            
            return True
            
        except Exception as e:
            logger.error(f"Error restoring ambient settings: {str(e)}")
            return False

    @staticmethod
    def save_camera_settings(viewport):
        """
        Extract camera settings from the viewport.
        
        Args:
            viewport: The VTK viewport
            
        Returns:
            dict: Camera settings or None if not available
        """
        try:
            if not viewport or not hasattr(viewport, 'renderer') or not viewport.renderer:
                return None
            
            camera = viewport.renderer.GetActiveCamera()
            if not camera:
                return None
            
            # Save camera settings
            camera_data = {
                'position': list(camera.GetPosition()),
                'focal_point': list(camera.GetFocalPoint()),
                'view_up': list(camera.GetViewUp()),
                'clipping_range': list(camera.GetClippingRange())
            }
            
            return camera_data
        
        except Exception as e:
            logger.error(f"Error extracting camera settings: {str(e)}")
            return None
    
    @staticmethod
    def restore_camera_settings(viewport, camera_data):
        """
        Restore camera settings to the viewport.
        
        Args:
            viewport: The VTK viewport
            camera_data: Camera settings to restore
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not viewport or not hasattr(viewport, 'renderer') or not viewport.renderer:
                return False
            
            if not camera_data:
                return False
            
            camera = viewport.renderer.GetActiveCamera()
            if not camera:
                return False
            
            # Restore camera settings
            if 'position' in camera_data:
                camera.SetPosition(*camera_data['position'])
            
            if 'focal_point' in camera_data:
                camera.SetFocalPoint(*camera_data['focal_point'])
            
            if 'view_up' in camera_data:
                camera.SetViewUp(*camera_data['view_up'])
            
            if 'clipping_range' in camera_data:
                camera.SetClippingRange(*camera_data['clipping_range'])
            
            # Render the scene with new camera settings
            if hasattr(viewport, 'render_window') and viewport.render_window:
                viewport.render_window.Render()
            
            return True
        
        except Exception as e:
            logger.error(f"Error restoring camera settings: {str(e)}")
            return False

    @staticmethod
    def save_pipeline_model_properties(viewport):
        """
        Save pipeline model rendering properties such as opacity, visibility and color.
        
        Args:
            viewport: The VTK viewport
            
        Returns:
            dict: Model properties or None if not available
        """
        try:
            if not viewport:
                return None
                
            model_properties = {}
            
            # Check for actor collection
            if hasattr(viewport, 'renderer') and viewport.renderer:
                # Get all actors in the renderer
                actors = viewport.renderer.GetActors()
                actors.InitTraversal()
                
                actor_count = actors.GetNumberOfItems()
                logger.info(f"Found {actor_count} actors in renderer")
                
                # Store properties for each actor
                actor_index = 0
                actor = actors.GetNextActor()
                while actor:
                    try:
                        # Get actor properties
                        props = actor.GetProperty()
                        
                        # Check if this actor has a name or user data
                        if hasattr(actor, "GetObjectName") and callable(getattr(actor, "GetObjectName")):
                            actor_name = actor.GetObjectName()
                        else:
                            actor_name = f"actor_{actor_index}"
                        
                        # Store properties
                        model_properties[actor_name] = {
                            "opacity": props.GetOpacity(),
                            "color": list(props.GetColor()),
                            "visible": actor.GetVisibility(),
                            "representation": props.GetRepresentation()
                        }
                        
                        actor_index += 1
                        actor = actors.GetNextActor()
                    except Exception as inner_e:
                        logger.error(f"Error saving actor {actor_index} properties: {str(inner_e)}")
                        actor_index += 1
                        actor = actors.GetNextActor()
            
            # Also check for pipeline actors list if it exists
            if hasattr(viewport, 'pipeline_actors') and viewport.pipeline_actors:
                for i, actor in enumerate(viewport.pipeline_actors):
                    try:
                        props = actor.GetProperty()
                        model_properties[f"pipeline_{i}"] = {
                            "opacity": props.GetOpacity(),
                            "color": list(props.GetColor()),
                            "visible": actor.GetVisibility(),
                            "representation": props.GetRepresentation()
                        }
                    except Exception as inner_e:
                        logger.error(f"Error saving pipeline actor {i} properties: {str(inner_e)}")
            
            # Check for any STL model actors
            if hasattr(viewport, 'stl_actors') and viewport.stl_actors:
                for stl_file, actor in viewport.stl_actors.items():
                    try:
                        props = actor.GetProperty()
                        model_properties[f"stl_{os.path.basename(stl_file)}"] = {
                            "opacity": props.GetOpacity(),
                            "color": list(props.GetColor()),
                            "visible": actor.GetVisibility(),
                            "representation": props.GetRepresentation()
                        }
                    except Exception as inner_e:
                        logger.error(f"Error saving STL actor properties for {stl_file}: {str(inner_e)}")
            
            # Check for boundary actors
            if hasattr(viewport, 'boundary_actors') and viewport.boundary_actors:
                for boundary_name, actor in viewport.boundary_actors.items():
                    try:
                        props = actor.GetProperty()
                        model_properties[f"boundary_{boundary_name}"] = {
                            "opacity": props.GetOpacity(),
                            "color": list(props.GetColor()),
                            "visible": actor.GetVisibility(),
                            "representation": props.GetRepresentation()
                        }
                    except Exception as inner_e:
                        logger.error(f"Error saving boundary actor properties for {boundary_name}: {str(inner_e)}")
            
            return model_properties
            
        except Exception as e:
            logger.error(f"Error saving pipeline model properties: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def restore_pipeline_model_properties(viewport, model_properties):
        """
        Restore pipeline model rendering properties such as opacity, visibility and color.
        
        Args:
            viewport: The VTK viewport
            model_properties: Model properties to restore
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not viewport or not model_properties:
                return False
                
            success = False
            
            # Check for actor collection
            if hasattr(viewport, 'renderer') and viewport.renderer:
                # Get all actors in the renderer
                actors = viewport.renderer.GetActors()
                actors.InitTraversal()
                
                actor_count = actors.GetNumberOfItems()
                logger.info(f"Found {actor_count} actors in renderer for property restoration")
                
                # Restore properties for each actor if we have a matching entry
                actor_index = 0
                actor = actors.GetNextActor()
                while actor:
                    try:
                        # Check if this actor has a name or user data
                        if hasattr(actor, "GetObjectName") and callable(getattr(actor, "GetObjectName")):
                            actor_name = actor.GetObjectName()
                        else:
                            actor_name = f"actor_{actor_index}"
                        
                        # See if we have properties for this actor
                        if actor_name in model_properties:
                            props = model_properties[actor_name]
                            actor_props = actor.GetProperty()
                            
                            # Apply properties
                            if "opacity" in props:
                                actor_props.SetOpacity(props["opacity"])
                            if "color" in props:
                                actor_props.SetColor(*props["color"])
                            if "visible" in props:
                                actor.SetVisibility(props["visible"])
                            if "representation" in props:
                                actor_props.SetRepresentation(props["representation"])
                                
                            success = True
                        
                        actor_index += 1
                        actor = actors.GetNextActor()
                    except Exception as inner_e:
                        logger.error(f"Error restoring actor {actor_index} properties: {str(inner_e)}")
                        actor_index += 1
                        actor = actors.GetNextActor()
            
            # Also check for pipeline actors list if it exists
            if hasattr(viewport, 'pipeline_actors') and viewport.pipeline_actors:
                for i, actor in enumerate(viewport.pipeline_actors):
                    try:
                        actor_name = f"pipeline_{i}"
                        if actor_name in model_properties:
                            props = model_properties[actor_name]
                            actor_props = actor.GetProperty()
                            
                            # Apply properties
                            if "opacity" in props:
                                actor_props.SetOpacity(props["opacity"])
                            if "color" in props:
                                actor_props.SetColor(*props["color"])
                            if "visible" in props:
                                actor.SetVisibility(props["visible"])
                            if "representation" in props:
                                actor_props.SetRepresentation(props["representation"])
                                
                            success = True
                    except Exception as inner_e:
                        logger.error(f"Error restoring pipeline actor {i} properties: {str(inner_e)}")
            
            # Check for any STL model actors
            if hasattr(viewport, 'stl_actors') and viewport.stl_actors:
                for stl_file, actor in viewport.stl_actors.items():
                    try:
                        actor_name = f"stl_{os.path.basename(stl_file)}"
                        if actor_name in model_properties:
                            props = model_properties[actor_name]
                            actor_props = actor.GetProperty()
                            
                            # Apply properties
                            if "opacity" in props:
                                actor_props.SetOpacity(props["opacity"])
                            if "color" in props:
                                actor_props.SetColor(*props["color"])
                            if "visible" in props:
                                actor.SetVisibility(props["visible"])
                            if "representation" in props:
                                actor_props.SetRepresentation(props["representation"])
                                
                            success = True
                    except Exception as inner_e:
                        logger.error(f"Error restoring STL actor properties for {stl_file}: {str(inner_e)}")
            
            # Check for boundary actors
            if hasattr(viewport, 'boundary_actors') and viewport.boundary_actors:
                for boundary_name, actor in viewport.boundary_actors.items():
                    try:
                        actor_name = f"boundary_{boundary_name}"
                        if actor_name in model_properties:
                            props = model_properties[actor_name]
                            actor_props = actor.GetProperty()
                            
                            # Apply properties
                            if "opacity" in props:
                                actor_props.SetOpacity(props["opacity"])
                            if "color" in props:
                                actor_props.SetColor(*props["color"])
                            if "visible" in props:
                                actor.SetVisibility(props["visible"])
                            if "representation" in props:
                                actor_props.SetRepresentation(props["representation"])
                                
                            success = True
                    except Exception as inner_e:
                        logger.error(f"Error restoring boundary actor properties for {boundary_name}: {str(inner_e)}")
            
            # Render the changes
            if success and hasattr(viewport, 'render_window') and viewport.render_window:
                viewport.render_window.Render()
            
            return success
            
        except Exception as e:
            logger.error(f"Error restoring pipeline model properties: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
