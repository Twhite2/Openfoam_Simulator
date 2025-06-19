#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization state module for Openfoam_Simulator.

This module handles saving and restoring the complete visualization state
for projects, ensuring boundary conditions, solver parameters, and view settings
are preserved between sessions.
"""

import os
import json
import logging
from pathlib import Path

# Setup logger
logger = logging.getLogger(__name__)

class VisualizationState:
    """
    Handles saving and loading of visualization state for Openfoam_Simulator projects.
    
    This includes boundary conditions, solver parameters, ambient region settings,
    and viewport camera configuration.
    """
    
    @staticmethod
    def save_state(project, main_window):
        """
        Save the complete visualization state to a project.
        
        Args:
            project: The project object
            main_window: The main window containing visualization controls
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create visualization state directory if it doesn't exist
            viz_dir = os.path.join(project.project_dir, "visualization")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Define state file path
            state_file = os.path.join(viz_dir, "visualization_state.json")
            
            # Collect state from all UI components
            state = {
                "ambient_region": VisualizationState._collect_ambient_region_state(main_window),
                "boundary_conditions": VisualizationState._collect_boundary_condition_state(main_window),
                "solver_parameters": VisualizationState._collect_solver_parameters(main_window),
                "viewport": VisualizationState._collect_viewport_state(main_window),
                "visualization_controls": VisualizationState._collect_visualization_controls_state(main_window)
            }
            
            # Save state to file
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=4)
                
            logger.info(f"Visualization state saved to {state_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving visualization state: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    @staticmethod
    def load_state(project, main_window):
        """
        Load the complete visualization state for a project.
        
        Args:
            project: The project object
            main_window: The main window containing visualization controls
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check for visualization state file
            state_file = os.path.join(project.project_dir, "visualization", "visualization_state.json")
            if not os.path.exists(state_file):
                logger.info(f"No visualization state file found at {state_file}")
                return False
            
            # Load state from file
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Apply state to all UI components
            VisualizationState._apply_ambient_region_state(state.get("ambient_region", {}), main_window)
            VisualizationState._apply_boundary_condition_state(state.get("boundary_conditions", {}), main_window)
            VisualizationState._apply_solver_parameters(state.get("solver_parameters", {}), main_window)
            VisualizationState._apply_viewport_state(state.get("viewport", {}), main_window)
            VisualizationState._apply_visualization_controls_state(state.get("visualization_controls", {}), main_window)
            
            logger.info(f"Visualization state loaded from {state_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading visualization state: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    @staticmethod
    def _collect_ambient_region_state(main_window):
        """Collect ambient region settings from the UI."""
        state = {
            "enabled": False,
            "settings": {}
        }
        
        # Get ambient region tab
        ambient_tab = getattr(main_window, "ambient_region_tab", None)
        if not ambient_tab:
            return state
        
        # Check if ambient region is created
        ambient_region_created = False
        if hasattr(ambient_tab, "is_region_created"):
            ambient_region_created = ambient_tab.is_region_created
        
        state["enabled"] = ambient_region_created
        
        # Collect settings from UI components
        settings = {}
        if hasattr(ambient_tab, "fluid_combo"):
            settings["fluid"] = ambient_tab.fluid_combo.currentText()
        
        if hasattr(ambient_tab, "color_combo"):
            settings["color"] = ambient_tab.color_combo.currentText()
            
        if hasattr(ambient_tab, "opacity_spin"):
            settings["opacity"] = ambient_tab.opacity_spin.value()
            
        # Collect all numeric fields
        for attr_name in dir(ambient_tab):
            if attr_name.endswith("_spin") and attr_name != "opacity_spin":
                attr = getattr(ambient_tab, attr_name, None)
                if attr and hasattr(attr, "value"):
                    # Extract parameter name from spin box name
                    param_name = attr_name.replace("_spin", "")
                    settings[param_name] = attr.value()
            elif attr_name.endswith("_check"):
                attr = getattr(ambient_tab, attr_name, None)
                if attr and hasattr(attr, "isChecked"):
                    param_name = attr_name.replace("_check", "")
                    settings[param_name] = attr.isChecked()
        
        state["settings"] = settings
        return state
    
    @staticmethod
    def _collect_boundary_condition_state(main_window):
        """Collect boundary condition settings from the UI."""
        state = {}
        
        # Get boundary conditions dictionary
        boundary_conditions = getattr(main_window, "boundary_conditions", {})
        if not boundary_conditions:
            return state
            
        # Collect settings from each boundary condition widget
        for name, widget in boundary_conditions.items():
            if not widget:
                continue
                
            # Check if widget has get_values method
            if hasattr(widget, "get_values"):
                try:
                    values = widget.get_values()
                    state[name] = values
                except Exception as e:
                    logger.warning(f"Error getting values for boundary {name}: {str(e)}")
                    continue
            
            # If no get_values method, try to extract values from UI components
            else:
                bc_state = {}
                
                # Collect from combo boxes
                for attr_name in dir(widget):
                    if attr_name.endswith("_combo"):
                        attr = getattr(widget, attr_name, None)
                        if attr and hasattr(attr, "currentText"):
                            param_name = attr_name.replace("_combo", "")
                            bc_state[param_name] = attr.currentText()
                
                # Collect from spin boxes
                for attr_name in dir(widget):
                    if attr_name.endswith("_spin"):
                        attr = getattr(widget, attr_name, None)
                        if attr and hasattr(attr, "value"):
                            param_name = attr_name.replace("_spin", "")
                            bc_state[param_name] = attr.value()
                
                # Collect from check boxes
                for attr_name in dir(widget):
                    if attr_name.endswith("_check"):
                        attr = getattr(widget, attr_name, None)
                        if attr and hasattr(attr, "isChecked"):
                            param_name = attr_name.replace("_check", "")
                            bc_state[param_name] = attr.isChecked()
                
                # Add to state if we found values
                if bc_state:
                    state[name] = bc_state
        
        return state
    
    @staticmethod
    def _collect_solver_parameters(main_window):
        """Collect solver parameters from the UI."""
        state = {}
        
        # Get simulation controls
        sim_controls = getattr(main_window, "simulation_controls", None)
        if not sim_controls:
            return state
        
        # Collect solver type
        if hasattr(sim_controls, "solver_combo"):
            state["solver"] = sim_controls.solver_combo.currentText()
        
        # Collect parallel settings
        if hasattr(sim_controls, "parallel_check"):
            state["parallel"] = sim_controls.parallel_check.isChecked()
            
        if hasattr(sim_controls, "processors_spin"):
            state["processors"] = sim_controls.processors_spin.value()
        
        # Collect time settings
        if hasattr(sim_controls, "start_time_spin"):
            state["start_time"] = sim_controls.start_time_spin.value()
            
        if hasattr(sim_controls, "end_time_spin"):
            state["end_time"] = sim_controls.end_time_spin.value()
            
        if hasattr(sim_controls, "delta_t_spin"):
            state["delta_t"] = sim_controls.delta_t_spin.value()
            
        if hasattr(sim_controls, "write_interval_spin"):
            state["write_interval"] = sim_controls.write_interval_spin.value()
        
        # Collect scheme settings if available
        schemes = {}
        for scheme_type in ["div", "grad", "laplacian", "interpolation"]:
            scheme_combo = getattr(sim_controls, f"{scheme_type}_scheme_combo", None)
            if scheme_combo:
                schemes[scheme_type] = scheme_combo.currentText()
        
        if schemes:
            state["schemes"] = schemes
        
        return state
    
    @staticmethod
    def _collect_viewport_state(main_window):
        """Collect viewport state from the UI."""
        state = {}
        
        # Get viewport
        viewport = getattr(main_window, "viewport", None)
        if not viewport:
            return state
        
        # Collect camera position if method exists
        if hasattr(viewport, "get_camera_state"):
            try:
                state["camera"] = viewport.get_camera_state()
            except Exception as e:
                logger.warning(f"Error getting camera state: {str(e)}")
        
        # If no dedicated method, try direct attribute access
        elif hasattr(viewport, "camera"):
            camera = viewport.camera
            if hasattr(camera, "GetPosition") and hasattr(camera, "GetFocalPoint"):
                try:
                    state["camera"] = {
                        "position": camera.GetPosition(),
                        "focal_point": camera.GetFocalPoint(),
                        "view_up": camera.GetViewUp() if hasattr(camera, "GetViewUp") else [0, 1, 0]
                    }
                except Exception as e:
                    logger.warning(f"Error accessing camera attributes: {str(e)}")
        
        return state
    
    @staticmethod
    def _collect_visualization_controls_state(main_window):
        """Collect visualization controls state from the UI."""
        state = {}
        
        # Get visualization controls
        viz_controls = getattr(main_window, "visualization_controls", None)
        if not viz_controls:
            return state
        
        # Collect representation
        if hasattr(viz_controls, "representation_combo"):
            state["representation"] = viz_controls.representation_combo.currentText()
        
        # Collect coloring
        if hasattr(viz_controls, "coloring_combo"):
            state["coloring"] = viz_controls.coloring_combo.currentText()
            
        # Collect field component if applicable
        if hasattr(viz_controls, "component_combo"):
            state["component"] = viz_controls.component_combo.currentText()
        
        # Collect colormap
        if hasattr(viz_controls, "colormap_combo"):
            state["colormap"] = viz_controls.colormap_combo.currentText()
        
        # Collect scalar range
        scalar_range = []
        if hasattr(viz_controls, "min_value_spin") and hasattr(viz_controls, "max_value_spin"):
            scalar_range = [viz_controls.min_value_spin.value(), viz_controls.max_value_spin.value()]
            state["scalar_range"] = scalar_range
        
        # Collect other visualization options
        options = {}
        
        # Point size
        if hasattr(viz_controls, "point_size_spin"):
            options["point_size"] = viz_controls.point_size_spin.value()
        
        # Line width
        if hasattr(viz_controls, "line_width_spin"):
            options["line_width"] = viz_controls.line_width_spin.value()
            
        # Opacity
        if hasattr(viz_controls, "opacity_spin"):
            options["opacity"] = viz_controls.opacity_spin.value()
        
        # Show edges
        if hasattr(viz_controls, "show_edges_check"):
            options["show_edges"] = viz_controls.show_edges_check.isChecked()
            
        # Collect any additional checkboxes
        for attr_name in dir(viz_controls):
            if attr_name.endswith("_check") and attr_name != "show_edges_check":
                attr = getattr(viz_controls, attr_name, None)
                if attr and hasattr(attr, "isChecked"):
                    param_name = attr_name.replace("_check", "")
                    options[param_name] = attr.isChecked()
        
        if options:
            state["options"] = options
        
        return state
    
    @staticmethod
    def _apply_ambient_region_state(state, main_window):
        """Apply saved ambient region state to the UI."""
        if not state:
            return
            
        # Get ambient region tab
        ambient_tab = getattr(main_window, "ambient_region_tab", None)
        if not ambient_tab:
            return
        
        # Apply settings to UI components
        settings = state.get("settings", {})
        
        # Set fluid type
        if "fluid" in settings and hasattr(ambient_tab, "fluid_combo"):
            index = ambient_tab.fluid_combo.findText(settings["fluid"])
            if index >= 0:
                ambient_tab.fluid_combo.setCurrentIndex(index)
        
        # Set color
        if "color" in settings and hasattr(ambient_tab, "color_combo"):
            index = ambient_tab.color_combo.findText(settings["color"])
            if index >= 0:
                ambient_tab.color_combo.setCurrentIndex(index)
        
        # Set opacity
        if "opacity" in settings and hasattr(ambient_tab, "opacity_spin"):
            ambient_tab.opacity_spin.setValue(settings["opacity"])
        
        # Set other numeric values
        for param, value in settings.items():
            if param not in ["fluid", "color", "opacity"]:
                # Try to find matching spin box
                spin_attr = getattr(ambient_tab, f"{param}_spin", None)
                if spin_attr and hasattr(spin_attr, "setValue"):
                    spin_attr.setValue(value)
                
                # Try to find matching check box
                check_attr = getattr(ambient_tab, f"{param}_check", None)
                if check_attr and hasattr(check_attr, "setChecked"):
                    check_attr.setChecked(value)
        
        # Create the ambient region if it was enabled
        if state.get("enabled", False) and hasattr(ambient_tab, "create_region_button"):
            ambient_tab.create_region_button.click()
    
    @staticmethod
    def _apply_boundary_condition_state(state, main_window):
        """Apply saved boundary condition state to the UI."""
        if not state:
            return
            
        # Get boundary conditions dictionary
        boundary_conditions = getattr(main_window, "boundary_conditions", {})
        if not boundary_conditions:
            return
        
        # Apply saved state to each boundary condition widget
        for name, values in state.items():
            if name not in boundary_conditions:
                continue
                
            widget = boundary_conditions[name]
            if not widget:
                continue
            
            # Use set_values method if available
            if hasattr(widget, "set_values"):
                try:
                    widget.set_values(values)
                    continue
                except Exception as e:
                    logger.warning(f"Error setting values for boundary {name}: {str(e)}")
            
            # Otherwise set values directly on UI components
            for param, value in values.items():
                # Try combo box
                combo_attr = getattr(widget, f"{param}_combo", None)
                if combo_attr and hasattr(combo_attr, "findText") and hasattr(combo_attr, "setCurrentIndex"):
                    index = combo_attr.findText(value)
                    if index >= 0:
                        combo_attr.setCurrentIndex(index)
                    continue
                
                # Try spin box
                spin_attr = getattr(widget, f"{param}_spin", None)
                if spin_attr and hasattr(spin_attr, "setValue"):
                    spin_attr.setValue(value)
                    continue
                
                # Try check box
                check_attr = getattr(widget, f"{param}_check", None)
                if check_attr and hasattr(check_attr, "setChecked"):
                    check_attr.setChecked(value)
    
    @staticmethod
    def _apply_solver_parameters(state, main_window):
        """Apply saved solver parameters to the UI."""
        if not state:
            return
            
        # Get simulation controls
        sim_controls = getattr(main_window, "simulation_controls", None)
        if not sim_controls:
            return
        
        # Set solver type
        if "solver" in state and hasattr(sim_controls, "solver_combo"):
            index = sim_controls.solver_combo.findText(state["solver"])
            if index >= 0:
                sim_controls.solver_combo.setCurrentIndex(index)
        
        # Set parallel settings
        if "parallel" in state and hasattr(sim_controls, "parallel_check"):
            sim_controls.parallel_check.setChecked(state["parallel"])
            
        if "processors" in state and hasattr(sim_controls, "processors_spin"):
            sim_controls.processors_spin.setValue(state["processors"])
        
        # Set time settings
        if "start_time" in state and hasattr(sim_controls, "start_time_spin"):
            sim_controls.start_time_spin.setValue(state["start_time"])
            
        if "end_time" in state and hasattr(sim_controls, "end_time_spin"):
            sim_controls.end_time_spin.setValue(state["end_time"])
            
        if "delta_t" in state and hasattr(sim_controls, "delta_t_spin"):
            sim_controls.delta_t_spin.setValue(state["delta_t"])
            
        if "write_interval" in state and hasattr(sim_controls, "write_interval_spin"):
            sim_controls.write_interval_spin.setValue(state["write_interval"])
        
        # Set scheme settings
        schemes = state.get("schemes", {})
        for scheme_type, scheme_value in schemes.items():
            scheme_combo = getattr(sim_controls, f"{scheme_type}_scheme_combo", None)
            if scheme_combo:
                index = scheme_combo.findText(scheme_value)
                if index >= 0:
                    scheme_combo.setCurrentIndex(index)
    
    @staticmethod
    def _apply_viewport_state(state, main_window):
        """Apply saved viewport state to the UI."""
        if not state:
            return
            
        # Get viewport
        viewport = getattr(main_window, "viewport", None)
        if not viewport:
            return
        
        # Apply camera position if method exists
        camera_state = state.get("camera", {})
        if not camera_state:
            return
            
        if hasattr(viewport, "set_camera_state"):
            try:
                viewport.set_camera_state(camera_state)
            except Exception as e:
                logger.warning(f"Error setting camera state: {str(e)}")
        
        # If no dedicated method, try direct attribute access
        elif hasattr(viewport, "camera"):
            camera = viewport.camera
            if (hasattr(camera, "SetPosition") and hasattr(camera, "SetFocalPoint") and
                "position" in camera_state and "focal_point" in camera_state):
                try:
                    camera.SetPosition(*camera_state["position"])
                    camera.SetFocalPoint(*camera_state["focal_point"])
                    if "view_up" in camera_state:
                        camera.SetViewUp(*camera_state["view_up"])
                    viewport.renderer.ResetCameraClippingRange()
                    viewport.render_window.Render()
                except Exception as e:
                    logger.warning(f"Error setting camera attributes: {str(e)}")
    
    @staticmethod
    def _apply_visualization_controls_state(state, main_window):
        """Apply saved visualization controls state to the UI."""
        if not state:
            return
            
        # Get visualization controls
        viz_controls = getattr(main_window, "visualization_controls", None)
        if not viz_controls:
            return
        
        # Set representation
        if "representation" in state and hasattr(viz_controls, "representation_combo"):
            index = viz_controls.representation_combo.findText(state["representation"])
            if index >= 0:
                viz_controls.representation_combo.setCurrentIndex(index)
        
        # Set coloring
        if "coloring" in state and hasattr(viz_controls, "coloring_combo"):
            index = viz_controls.coloring_combo.findText(state["coloring"])
            if index >= 0:
                viz_controls.coloring_combo.setCurrentIndex(index)
        
        # Set field component if applicable
        if "component" in state and hasattr(viz_controls, "component_combo"):
            index = viz_controls.component_combo.findText(state["component"])
            if index >= 0:
                viz_controls.component_combo.setCurrentIndex(index)
        
        # Set colormap
        if "colormap" in state and hasattr(viz_controls, "colormap_combo"):
            index = viz_controls.colormap_combo.findText(state["colormap"])
            if index >= 0:
                viz_controls.colormap_combo.setCurrentIndex(index)
        
        # Set scalar range
        scalar_range = state.get("scalar_range", [])
        if len(scalar_range) == 2:
            if hasattr(viz_controls, "min_value_spin"):
                viz_controls.min_value_spin.setValue(scalar_range[0])
            if hasattr(viz_controls, "max_value_spin"):
                viz_controls.max_value_spin.setValue(scalar_range[1])
        
        # Set visualization options
        options = state.get("options", {})
        for param, value in options.items():
            # Point size
            if param == "point_size" and hasattr(viz_controls, "point_size_spin"):
                viz_controls.point_size_spin.setValue(value)
                
            # Line width
            elif param == "line_width" and hasattr(viz_controls, "line_width_spin"):
                viz_controls.line_width_spin.setValue(value)
                
            # Opacity
            elif param == "opacity" and hasattr(viz_controls, "opacity_spin"):
                viz_controls.opacity_spin.setValue(value)
                
            # Show edges
            elif param == "show_edges" and hasattr(viz_controls, "show_edges_check"):
                viz_controls.show_edges_check.setChecked(value)
                
            # Other checkboxes
            else:
                check_attr = getattr(viz_controls, f"{param}_check", None)
                if check_attr and hasattr(check_attr, "setChecked"):
                    check_attr.setChecked(value)
