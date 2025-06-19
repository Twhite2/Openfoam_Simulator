#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boundary conditions dialog for Openfoam_Simulator.

This module provides a dialog for configuring boundary conditions
for OpenFOAM simulations.
"""

import os
import logging
from typing import Dict, Any, List
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QComboBox, QTabWidget, QWidget, QFormLayout, QLineEdit,
                           QDoubleSpinBox, QCheckBox, QListWidget, QListWidgetItem,
                           QGroupBox, QMessageBox, QStackedWidget, QInputDialog)
from PyQt5.QtCore import Qt

from ..utils.logger import get_logger

# Setup logger
logger = get_logger(__name__)

class BoundaryDialog(QDialog):
    """
    Dialog for configuring boundary conditions.
    """
    
    def __init__(self, parent=None, boundary_names=None):
        super().__init__(parent)
        self.boundary_names = boundary_names or []
        self.boundary_config = {}
        
        self.setWindowTitle("OpenFOAM Boundary Conditions")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the dialog UI."""
        main_layout = QVBoxLayout()
        
        # Boundary list on the left
        boundaries_group = QGroupBox("Available Boundaries")
        boundaries_layout = QVBoxLayout()
        
        self.boundary_list = QListWidget()
        for name in self.boundary_names:
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, name)
            self.boundary_list.addItem(item)
        
        self.boundary_list.currentItemChanged.connect(self.boundary_selected)
        boundaries_layout.addWidget(self.boundary_list)
        boundaries_group.setLayout(boundaries_layout)

         # Add Create Boundary button next to the boundary list
        create_boundary_btn = QPushButton("Create New Boundary")
        create_boundary_btn.clicked.connect(self.start_face_selection)
        boundaries_layout.addWidget(create_boundary_btn)
        
        # Configuration area on the right
        config_layout = QVBoxLayout()
        
        # Type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Boundary Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["inlet", "outlet", "wall", "symmetryPlane", "empty"])
        self.type_combo.currentTextChanged.connect(self.type_changed)
        type_layout.addWidget(self.type_combo)
        config_layout.addLayout(type_layout)
        
        # Stacked widget for different boundary type configurations
        self.config_stack = QStackedWidget()
        
        # Inlet configuration
        inlet_widget = QWidget()
        inlet_layout = QFormLayout()
        
        # Velocity
        vel_group = QGroupBox("Velocity")
        vel_form = QFormLayout()
        
        self.vel_type = QComboBox()
        self.vel_type.addItems(["fixedValue", "flowRate", "pressureInletOutletVelocity"])
        vel_form.addRow("Type:", self.vel_type)
        
        self.vel_x = QDoubleSpinBox()
        self.vel_x.setRange(-100, 100)
        self.vel_x.setValue(1.0)
        self.vel_x.setDecimals(3)
        self.vel_x.setSuffix(" m/s")
        
        self.vel_y = QDoubleSpinBox()
        self.vel_y.setRange(-100, 100)
        self.vel_y.setValue(0.0)
        self.vel_y.setDecimals(3)
        self.vel_y.setSuffix(" m/s")
        
        self.vel_z = QDoubleSpinBox()
        self.vel_z.setRange(-100, 100)
        self.vel_z.setValue(0.0)
        self.vel_z.setDecimals(3)
        self.vel_z.setSuffix(" m/s")
        
        vel_components = QHBoxLayout()
        vel_components.addWidget(QLabel("X:"))
        vel_components.addWidget(self.vel_x)
        vel_components.addWidget(QLabel("Y:"))
        vel_components.addWidget(self.vel_y)
        vel_components.addWidget(QLabel("Z:"))
        vel_components.addWidget(self.vel_z)
        
        vel_form.addRow("Value:", vel_components)
        vel_group.setLayout(vel_form)
        inlet_layout.addRow(vel_group)
        
        # Pressure
        p_group = QGroupBox("Pressure")
        p_form = QFormLayout()
        
        self.p_type = QComboBox()
        self.p_type.addItems(["zeroGradient", "totalPressure", "fixedValue"])
        p_form.addRow("Type:", self.p_type)
        
        self.p_value = QDoubleSpinBox()
        self.p_value.setRange(0, 1000000)
        self.p_value.setValue(0)
        self.p_value.setDecimals(1)
        self.p_value.setSuffix(" Pa")
        p_form.addRow("Value:", self.p_value)
        
        p_group.setLayout(p_form)
        inlet_layout.addRow(p_group)
        
        # Temperature
        t_group = QGroupBox("Temperature")
        t_form = QFormLayout()
        
        self.t_type = QComboBox()
        self.t_type.addItems(["fixedValue", "zeroGradient"])
        t_form.addRow("Type:", self.t_type)
        
        self.t_value = QDoubleSpinBox()
        self.t_value.setRange(0, 1000)
        self.t_value.setValue(300)
        self.t_value.setDecimals(1)
        self.t_value.setSuffix(" K")
        t_form.addRow("Value:", self.t_value)
        
        t_group.setLayout(t_form)
        inlet_layout.addRow(t_group)
        
        inlet_widget.setLayout(inlet_layout)
        self.config_stack.addWidget(inlet_widget)
        
        # Outlet configuration
        outlet_widget = QWidget()
        outlet_layout = QFormLayout()
        
        # Pressure
        out_p_group = QGroupBox("Pressure")
        out_p_form = QFormLayout()
        
        self.out_p_type = QComboBox()
        self.out_p_type.addItems(["fixedValue", "zeroGradient", "totalPressure"])
        out_p_form.addRow("Type:", self.out_p_type)
        
        self.out_p_value = QDoubleSpinBox()
        self.out_p_value.setRange(0, 1000000)
        self.out_p_value.setValue(0)
        self.out_p_value.setDecimals(1)
        self.out_p_value.setSuffix(" Pa")
        out_p_form.addRow("Value:", self.out_p_value)
        
        out_p_group.setLayout(out_p_form)
        outlet_layout.addRow(out_p_group)
        
        # Velocity
        out_vel_group = QGroupBox("Velocity")
        out_vel_form = QFormLayout()
        
        self.out_vel_type = QComboBox()
        self.out_vel_type.addItems(["inletOutlet", "zeroGradient"])
        out_vel_form.addRow("Type:", self.out_vel_type)
        
        out_vel_group.setLayout(out_vel_form)
        outlet_layout.addRow(out_vel_group)
        
        outlet_widget.setLayout(outlet_layout)
        self.config_stack.addWidget(outlet_widget)
        
        # Wall configuration
        wall_widget = QWidget()
        wall_layout = QFormLayout()
        
        # Velocity
        wall_vel_group = QGroupBox("Velocity")
        wall_vel_form = QFormLayout()
        
        self.wall_vel_type = QComboBox()
        self.wall_vel_type.addItems(["noSlip", "slip", "movingWallVelocity"])
        wall_vel_form.addRow("Type:", self.wall_vel_type)
        
        wall_vel_group.setLayout(wall_vel_form)
        wall_layout.addRow(wall_vel_group)
        
        # Temperature
        wall_t_group = QGroupBox("Temperature")
        wall_t_form = QFormLayout()
        
        self.wall_t_type = QComboBox()
        self.wall_t_type.addItems(["zeroGradient", "fixedValue"])
        wall_t_form.addRow("Type:", self.wall_t_type)
        
        self.wall_t_value = QDoubleSpinBox()
        self.wall_t_value.setRange(0, 1000)
        self.wall_t_value.setValue(300)
        self.wall_t_value.setDecimals(1)
        self.wall_t_value.setSuffix(" K")
        wall_t_form.addRow("Value:", self.wall_t_value)
        
        wall_t_group.setLayout(wall_t_form)
        wall_layout.addRow(wall_t_group)
        
        wall_widget.setLayout(wall_layout)
        self.config_stack.addWidget(wall_widget)
        
        # Empty configuration
        empty_widget = QWidget()
        empty_layout = QVBoxLayout()
        empty_layout.addWidget(QLabel("No configuration needed for this boundary type."))
        empty_widget.setLayout(empty_layout)
        self.config_stack.addWidget(empty_widget)
        
        # Symmetry configuration
        symmetry_widget = QWidget()
        symmetry_layout = QVBoxLayout()
        symmetry_layout.addWidget(QLabel("No configuration needed for this boundary type."))
        symmetry_widget.setLayout(symmetry_layout)
        self.config_stack.addWidget(symmetry_widget)
        
        config_layout.addWidget(self.config_stack)
        
        # Apply button for current boundary
        apply_btn = QPushButton("Apply to Boundary")
        apply_btn.clicked.connect(self.apply_to_boundary)
        config_layout.addWidget(apply_btn)
        
        # Horizontal layout for boundaries list and configuration
        h_layout = QHBoxLayout()
        h_layout.addWidget(boundaries_group, 1)
        h_layout.addLayout(config_layout, 2)
        
        main_layout.addLayout(h_layout)
        
        # Dialog buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
    
    def boundary_selected(self, current, previous):
        """Called when a boundary is selected from the list."""
        if not current:
            return
        
        boundary_name = current.data(Qt.UserRole)
        
        # Check if we already have configuration for this boundary
        if boundary_name in self.boundary_config:
            config = self.boundary_config[boundary_name]
            
            # Set the boundary type
            index = self.type_combo.findText(config.get("type", "wall"))
            if index >= 0:
                self.type_combo.setCurrentIndex(index)
                
            # Set the configuration values based on boundary type
            if config["type"] == "inlet":
                # Set velocity
                if "velocity" in config:
                    self.vel_x.setValue(config["velocity"][0])
                    self.vel_y.setValue(config["velocity"][1])
                    self.vel_z.setValue(config["velocity"][2])
                
                # Set pressure
                if "pressure" in config:
                    self.p_value.setValue(config["pressure"])
                
                # Set temperature
                if "temperature" in config:
                    self.t_value.setValue(config["temperature"])
            
            elif config["type"] == "outlet":
                # Set pressure
                if "pressure" in config:
                    self.out_p_value.setValue(config["pressure"])
    
    def type_changed(self, new_type):
        """Called when the boundary type is changed."""
        # Set the appropriate configuration widget
        if new_type == "inlet":
            self.config_stack.setCurrentIndex(0)
        elif new_type == "outlet":
            self.config_stack.setCurrentIndex(1)
        elif new_type == "wall":
            self.config_stack.setCurrentIndex(2)
        elif new_type == "empty":
            self.config_stack.setCurrentIndex(3)
        elif new_type == "symmetryPlane":
            self.config_stack.setCurrentIndex(4)
    
    def apply_to_boundary(self):
        """Apply the current configuration to the selected boundary."""
        current_item = self.boundary_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a boundary first.")
            return
        
        boundary_name = current_item.data(Qt.UserRole)
        boundary_type = self.type_combo.currentText()
        
        # Build configuration based on boundary type
        config = {"type": boundary_type}
        
        if boundary_type == "inlet":
            # Get velocity
            config["velocity"] = [
                self.vel_x.value(),
                self.vel_y.value(),
                self.vel_z.value()
            ]
            config["velocity_type"] = self.vel_type.currentText()
            
            # Get pressure
            config["pressure_type"] = self.p_type.currentText()
            config["pressure"] = self.p_value.value()
            
            # Get temperature
            config["temperature_type"] = self.t_type.currentText()
            config["temperature"] = self.t_value.value()
            
        elif boundary_type == "outlet":
            # Get pressure
            config["pressure_type"] = self.out_p_type.currentText()
            config["pressure"] = self.out_p_value.value()
            
            # Get velocity
            config["velocity_type"] = self.out_vel_type.currentText()
            
        elif boundary_type == "wall":
            # Get velocity
            config["velocity_type"] = self.wall_vel_type.currentText()
            
            # Get temperature
            config["temperature_type"] = self.wall_t_type.currentText()
            config["temperature"] = self.wall_t_value.value()
        
        # Save to config dictionary
        self.boundary_config[boundary_name] = config
        
        # Mark the item as configured by adding an asterisk
        if not current_item.text().endswith("*"):
            current_item.setText(f"{boundary_name} *")
        
        QMessageBox.information(self, "Success", f"Boundary '{boundary_name}' configured as {boundary_type}.")
    
    def get_boundary_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the boundary configuration.
        
        Returns:
            Dict[str, Dict[str, Any]]: The boundary configuration dictionary
        """
        return self.boundary_config
    
    # New method in boundary_dialog.py
    def start_face_selection(self):
        """Start face selection mode in the viewport"""
        # Ask for boundary name first
        boundary_name, ok = QInputDialog.getText(
            self, "New Boundary", "Enter boundary name (e.g., inlet, outlet, wall):"
        )
        
        if not ok or not boundary_name:
            return
            
        # Set viewport to selection mode and connect to callback
        main_window = self.parent()
        if hasattr(main_window, 'viewport') and main_window.viewport:
            # Tell user what to do
            QMessageBox.information(
                self,
                "Face Selection",
                f"Click on faces to select them for the new '{boundary_name}' boundary.\n"
                "Press ESC when finished selecting."
            )
            
            # Start selection mode in viewport
            main_window.viewport.start_face_selection(boundary_name, self.selection_complete)
            
            # Minimize dialog while selecting
            self.setWindowState(Qt.WindowMinimized)
    
    # Add to BoundaryDialog class
    def selection_complete(self, boundary_name, cell_ids):
        """
        Handle completion of face selection
        
        Args:
            boundary_name: Name of the new boundary
            cell_ids: List of selected cell IDs
        """
        # Restore window from minimized state
        self.setWindowState(Qt.WindowActive)
        
        # If no cells selected, abort
        if not cell_ids:
            QMessageBox.warning(
                self,
                "Selection Empty",
                "No faces were selected. Boundary not created."
            )
            return
        
        # Add new boundary to list
        item = QListWidgetItem(boundary_name)
        item.setData(Qt.UserRole, boundary_name)
        self.boundary_list.addItem(item)
        
        # Store the cell IDs for this boundary
        if not hasattr(self, 'boundary_cell_ids'):
            self.boundary_cell_ids = {}
        self.boundary_cell_ids[boundary_name] = cell_ids
        
        # Select the new boundary in list
        self.boundary_list.setCurrentItem(item)
        
        # Show success message with color recommendation
        color_info = ""
        if boundary_name.lower() == "inlet":
            color_info = "\nInlet boundaries are typically colored blue."
        elif boundary_name.lower() == "outlet":
            color_info = "\nOutlet boundaries are typically colored red."
        elif boundary_name.lower() == "wall":
            color_info = "\nWall boundaries are typically colored gray."
        
        QMessageBox.information(
            self,
            "Boundary Created",
            f"New boundary '{boundary_name}' created with {len(cell_ids)} faces.{color_info}"
        )
