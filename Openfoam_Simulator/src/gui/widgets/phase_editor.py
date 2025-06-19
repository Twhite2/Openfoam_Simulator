#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase editor widget for Openfoam_Simulator application.

This module implements a widget for editing multiple fluid phases used in
multiphase CFD simulations, with specific focus on oil & gas applications.
It allows configuration of:
- Individual phase properties (density, viscosity, etc.)
- Phase interactions (surface tension, diffusion coefficients)
- Phase transitions and reactions
- Visualization properties for each phase
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QCheckBox, QRadioButton,
    QButtonGroup, QGroupBox, QTabWidget, QSpinBox, QDoubleSpinBox,
    QPushButton, QFrame, QScrollArea, QSizePolicy, QSlider,
    QToolTip, QMenu, QAction, QColorDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QToolButton,
    QListWidget, QListWidgetItem, QDialog, QDialogButtonBox,
    QInputDialog, QFileDialog
)
from PyQt5.QtCore import (
    Qt, QSize, pyqtSignal, QTimer, QPoint, QSettings, QRegExp, 
    QRectF
)
from PyQt5.QtGui import (
    QIcon, QPixmap, QColor, QPalette, QFont, QPainter, QBrush,
    QRegExpValidator, QIntValidator, QDoubleValidator, QLinearGradient
)

# Import utility modules
from ...utils.logger import get_logger
from ...config import get_value, set_value

logger = get_logger(__name__)


class PhaseProperties(QGroupBox):
    """
    Widget for configuring properties of a single phase.
    
    This widget provides controls for setting basic properties of a fluid phase,
    including physical properties and visualization settings.
    """
    
    # Signal emitted when properties change
    properties_changed = pyqtSignal(str, dict)
    
    def __init__(self, phase_name: str = "Phase 1", parent=None):
        """
        Initialize the phase properties widget.
        
        Args:
            phase_name (str): Name of the phase
            parent: Parent widget
        """
        super(PhaseProperties, self).__init__(phase_name, parent)
        
        # Store phase name
        self.phase_name = phase_name
        
        # Store phase properties
        self.properties = {
            "name": phase_name,
            "type": "liquid",
            "density": 1000.0,
            "viscosity": 0.001,
            "specific_heat": 4200.0,
            "thermal_conductivity": 0.6,
            "molar_weight": 18.0,
            "compressibility": 4.5e-10,
            "color": QColor(0, 120, 255, 150)
        }
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        self.layout = QFormLayout(self)
        self.layout.setContentsMargins(10, 20, 10, 10)
        self.layout.setSpacing(8)
        
        # Phase name
        self.name_edit = QLineEdit(self.phase_name)
        self.layout.addRow("Name:", self.name_edit)
        
        # Phase type
        self.type_combo = QComboBox()
        self.type_combo.addItems(["liquid", "gas", "solid"])
        self.type_combo.setCurrentText(self.properties["type"])
        self.layout.addRow("Type:", self.type_combo)
        
        # Density
        self.density_spin = QDoubleSpinBox()
        self.density_spin.setRange(0.01, 20000.0)
        self.density_spin.setDecimals(2)
        self.density_spin.setSingleStep(10.0)
        self.density_spin.setValue(self.properties["density"])
        self.density_spin.setSuffix(" kg/m³")
        self.layout.addRow("Density:", self.density_spin)
        
        # Viscosity
        self.viscosity_spin = QDoubleSpinBox()
        self.viscosity_spin.setRange(1e-7, 1000.0)
        self.viscosity_spin.setDecimals(6)
        self.viscosity_spin.setSingleStep(0.001)
        self.viscosity_spin.setValue(self.properties["viscosity"])
        self.viscosity_spin.setSuffix(" Pa·s")
        self.layout.addRow("Viscosity:", self.viscosity_spin)
        
        # Specific heat
        self.specific_heat_spin = QDoubleSpinBox()
        self.specific_heat_spin.setRange(1.0, 10000.0)
        self.specific_heat_spin.setDecimals(1)
        self.specific_heat_spin.setSingleStep(100.0)
        self.specific_heat_spin.setValue(self.properties["specific_heat"])
        self.specific_heat_spin.setSuffix(" J/(kg·K)")
        self.layout.addRow("Specific Heat:", self.specific_heat_spin)
        
        # Thermal conductivity
        self.thermal_cond_spin = QDoubleSpinBox()
        self.thermal_cond_spin.setRange(0.001, 1000.0)
        self.thermal_cond_spin.setDecimals(3)
        self.thermal_cond_spin.setSingleStep(0.01)
        self.thermal_cond_spin.setValue(self.properties["thermal_conductivity"])
        self.thermal_cond_spin.setSuffix(" W/(m·K)")
        self.layout.addRow("Thermal Conductivity:", self.thermal_cond_spin)
        
        # Molar weight
        self.molar_weight_spin = QDoubleSpinBox()
        self.molar_weight_spin.setRange(1.0, 1000.0)
        self.molar_weight_spin.setDecimals(2)
        self.molar_weight_spin.setSingleStep(1.0)
        self.molar_weight_spin.setValue(self.properties["molar_weight"])
        self.molar_weight_spin.setSuffix(" g/mol")
        self.layout.addRow("Molar Weight:", self.molar_weight_spin)
        
        # Compressibility
        self.compressibility_spin = QDoubleSpinBox()
        self.compressibility_spin.setRange(0.0, 1.0)
        self.compressibility_spin.setDecimals(15)
        self.compressibility_spin.setSingleStep(1e-10)
        self.compressibility_spin.setValue(self.properties["compressibility"])
        self.compressibility_spin.setSuffix(" 1/Pa")
        self.compressibility_spin.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
        self.layout.addRow("Compressibility:", self.compressibility_spin)
        
        # Color picker
        color_layout = QHBoxLayout()
        
        self.color_preview = QFrame()
        self.color_preview.setFixedSize(24, 24)
        self.color_preview.setFrameShape(QFrame.Box)
        self.color_preview.setFrameShadow(QFrame.Plain)
        self.color_preview.setAutoFillBackground(True)
        self._update_color_preview()
        color_layout.addWidget(self.color_preview)
        
        self.color_button = QPushButton("Select...")
        self.color_button.setFixedWidth(80)
        color_layout.addWidget(self.color_button)
        
        color_layout.addStretch()
        
        self.layout.addRow("Color:", color_layout)
        
        # Load from database button
        self.load_button = QPushButton("Load from Database...")
        self.layout.addRow("", self.load_button)
        
    def _connect_signals(self):
        """Connect signals to slots."""
        # Connect all inputs to property_changed slot
        self.name_edit.textChanged.connect(self._on_name_changed)
        self.type_combo.currentTextChanged.connect(self._on_property_changed)
        self.density_spin.valueChanged.connect(self._on_property_changed)
        self.viscosity_spin.valueChanged.connect(self._on_property_changed)
        self.specific_heat_spin.valueChanged.connect(self._on_property_changed)
        self.thermal_cond_spin.valueChanged.connect(self._on_property_changed)
        self.molar_weight_spin.valueChanged.connect(self._on_property_changed)
        self.compressibility_spin.valueChanged.connect(self._on_property_changed)
        
        # Connect color button
        self.color_button.clicked.connect(self._on_color_button_clicked)
        
        # Connect load button
        self.load_button.clicked.connect(self._on_load_from_database)
        
    def _update_color_preview(self):
        """Update the color preview frame."""
        palette = self.color_preview.palette()
        palette.setColor(QPalette.Window, self.properties["color"])
        self.color_preview.setPalette(palette)
        
    def _on_name_changed(self, name: str):
        """
        Handle phase name change.
        
        Args:
            name (str): New phase name
        """
        # Update properties
        self.phase_name = name
        self.properties["name"] = name
        
        # Update group box title
        self.setTitle(name)
        
        # Emit signal
        self.properties_changed.emit(name, self.properties)
        
    def _on_property_changed(self):
        """Handle property value changes."""
        # Update properties
        self.properties["type"] = self.type_combo.currentText()
        self.properties["density"] = self.density_spin.value()
        self.properties["viscosity"] = self.viscosity_spin.value()
        self.properties["specific_heat"] = self.specific_heat_spin.value()
        self.properties["thermal_conductivity"] = self.thermal_cond_spin.value()
        self.properties["molar_weight"] = self.molar_weight_spin.value()
        self.properties["compressibility"] = self.compressibility_spin.value()
        
        # Emit signal
        self.properties_changed.emit(self.phase_name, self.properties)
        
    def _on_color_button_clicked(self):
        """Handle color button click."""
        color = QColorDialog.getColor(
            self.properties["color"],
            self,
            "Select Phase Color",
            QColorDialog.ShowAlphaChannel
        )
        
        if color.isValid():
            # Update color
            self.properties["color"] = color
            
            # Update preview
            self._update_color_preview()
            
            # Emit signal
            self.properties_changed.emit(self.phase_name, self.properties)
            
    def _on_load_from_database(self):
        """Handle load from database button click."""
        # This would typically display a dialog with a list of fluids from a database
        # For now, we'll simulate this with a simple combo box dialog
        
        # Create and configure the dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Fluid")
        dialog.setMinimumWidth(300)
        
        # Fluid selection
        layout = QVBoxLayout(dialog)
        
        fluid_label = QLabel("Select a fluid from the database:")
        layout.addWidget(fluid_label)
        
        fluid_combo = QComboBox()
        fluid_combo.addItems([
            "Water (Standard)",
            "Air (Standard)",
            "Crude Oil (Light)",
            "Crude Oil (Medium)",
            "Crude Oil (Heavy)",
            "Natural Gas",
            "Diesel",
            "Gasoline",
            "Methane",
            "Ethane",
            "Propane",
            "n-Butane",
            "Carbon Dioxide",
            "Nitrogen",
            "Hydrogen Sulfide",
            "Brine (3.5% NaCl)"
        ])
        layout.addWidget(fluid_combo)
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            # Get selected fluid
            fluid = fluid_combo.currentText()
            
            # Set properties based on selected fluid
            if fluid == "Water (Standard)":
                self.properties["name"] = "Water"
                self.properties["type"] = "liquid"
                self.properties["density"] = 998.0
                self.properties["viscosity"] = 0.001
                self.properties["specific_heat"] = 4182.0
                self.properties["thermal_conductivity"] = 0.6
                self.properties["molar_weight"] = 18.01528
                self.properties["compressibility"] = 4.5e-10
                self.properties["color"] = QColor(0, 120, 255, 180)
                
            elif fluid == "Air (Standard)":
                self.properties["name"] = "Air"
                self.properties["type"] = "gas"
                self.properties["density"] = 1.225
                self.properties["viscosity"] = 1.81e-5
                self.properties["specific_heat"] = 1005.0
                self.properties["thermal_conductivity"] = 0.024
                self.properties["molar_weight"] = 28.97
                self.properties["compressibility"] = 1.0
                self.properties["color"] = QColor(220, 220, 220, 100)
                
            elif "Crude Oil" in fluid:
                self.properties["name"] = fluid
                self.properties["type"] = "liquid"
                
                if "Light" in fluid:
                    self.properties["density"] = 830.0
                    self.properties["viscosity"] = 0.005
                elif "Medium" in fluid:
                    self.properties["density"] = 870.0
                    self.properties["viscosity"] = 0.05
                else:  # Heavy
                    self.properties["density"] = 930.0
                    self.properties["viscosity"] = 0.5
                    
                self.properties["specific_heat"] = 2000.0
                self.properties["thermal_conductivity"] = 0.12
                self.properties["molar_weight"] = 250.0
                self.properties["compressibility"] = 1e-9
                self.properties["color"] = QColor(150, 75, 0, 200)
                
            elif fluid == "Natural Gas":
                self.properties["name"] = "Natural Gas"
                self.properties["type"] = "gas"
                self.properties["density"] = 0.8
                self.properties["viscosity"] = 1.1e-5
                self.properties["specific_heat"] = 2220.0
                self.properties["thermal_conductivity"] = 0.033
                self.properties["molar_weight"] = 18.0
                self.properties["compressibility"] = 1.0
                self.properties["color"] = QColor(200, 200, 150, 100)
                
            # Update UI
            self.name_edit.setText(self.properties["name"])
            self.type_combo.setCurrentText(self.properties["type"])
            self.density_spin.setValue(self.properties["density"])
            self.viscosity_spin.setValue(self.properties["viscosity"])
            self.specific_heat_spin.setValue(self.properties["specific_heat"])
            self.thermal_cond_spin.setValue(self.properties["thermal_conductivity"])
            self.molar_weight_spin.setValue(self.properties["molar_weight"])
            self.compressibility_spin.setValue(self.properties["compressibility"])
            self._update_color_preview()
            
            # Emit signal
            self.properties_changed.emit(self.phase_name, self.properties)
            
    def get_properties(self) -> Dict[str, Any]:
        """
        Get the current phase properties.
        
        Returns:
            Dict[str, Any]: Phase properties
        """
        return self.properties
    
    def set_properties(self, properties: Dict[str, Any]):
        """
        Set the phase properties.
        
        Args:
            properties (Dict[str, Any]): Properties to set
        """
        # Update stored properties
        self.properties.update(properties)
        
        # Update UI
        if "name" in properties:
            self.phase_name = properties["name"]
            self.name_edit.setText(properties["name"])
            self.setTitle(properties["name"])
            
        if "type" in properties:
            self.type_combo.setCurrentText(properties["type"])
            
        if "density" in properties:
            self.density_spin.setValue(properties["density"])
            
        if "viscosity" in properties:
            self.viscosity_spin.setValue(properties["viscosity"])
            
        if "specific_heat" in properties:
            self.specific_heat_spin.setValue(properties["specific_heat"])
            
        if "thermal_conductivity" in properties:
            self.thermal_cond_spin.setValue(properties["thermal_conductivity"])
            
        if "molar_weight" in properties:
            self.molar_weight_spin.setValue(properties["molar_weight"])
            
        if "compressibility" in properties:
            self.compressibility_spin.setValue(properties["compressibility"])
            
        if "color" in properties and isinstance(properties["color"], QColor):
            self._update_color_preview()


class PhaseInteraction(QGroupBox):
    """
    Widget for configuring interaction between two phases.
    
    This widget provides controls for setting properties of the interaction
    between two fluid phases, such as surface tension and diffusion coefficients.
    """
    
    # Signal emitted when properties change
    properties_changed = pyqtSignal(str, str, dict)
    
    def __init__(self, phase1_name: str = "Phase 1", phase2_name: str = "Phase 2", parent=None):
        """
        Initialize the phase interaction widget.
        
        Args:
            phase1_name (str): Name of first phase
            phase2_name (str): Name of second phase
            parent: Parent widget
        """
        title = f"Interaction: {phase1_name} - {phase2_name}"
        super(PhaseInteraction, self).__init__(title, parent)
        
        # Store phase names
        self.phase1_name = phase1_name
        self.phase2_name = phase2_name
        
        # Store interaction properties
        self.properties = {
            "surface_tension": 0.072,
            "contact_angle": 90.0,
            "diffusion_coefficient": 1e-9,
            "heat_transfer_coefficient": 0.1,
            "enable_mass_transfer": False,
            "enable_heat_transfer": True
        }
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        self.layout = QFormLayout(self)
        self.layout.setContentsMargins(10, 20, 10, 10)
        self.layout.setSpacing(8)
        
        # Surface tension
        self.surface_tension_spin = QDoubleSpinBox()
        self.surface_tension_spin.setRange(0.0, 1.0)
        self.surface_tension_spin.setDecimals(5)
        self.surface_tension_spin.setSingleStep(0.001)
        self.surface_tension_spin.setValue(self.properties["surface_tension"])
        self.surface_tension_spin.setSuffix(" N/m")
        self.layout.addRow("Surface Tension:", self.surface_tension_spin)
        
        # Contact angle
        self.contact_angle_spin = QDoubleSpinBox()
        self.contact_angle_spin.setRange(0.0, 180.0)
        self.contact_angle_spin.setDecimals(1)
        self.contact_angle_spin.setSingleStep(1.0)
        self.contact_angle_spin.setValue(self.properties["contact_angle"])
        self.contact_angle_spin.setSuffix("°")
        self.layout.addRow("Contact Angle:", self.contact_angle_spin)
        
        # Diffusion coefficient
        self.diffusion_spin = QDoubleSpinBox()
        self.diffusion_spin.setRange(0.0, 1.0)
        self.diffusion_spin.setDecimals(12)
        self.diffusion_spin.setSingleStep(1e-9)
        self.diffusion_spin.setValue(self.properties["diffusion_coefficient"])
        self.diffusion_spin.setSuffix(" m²/s")
        self.diffusion_spin.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
        self.layout.addRow("Diffusion Coefficient:", self.diffusion_spin)
        
        # Heat transfer coefficient
        self.heat_transfer_spin = QDoubleSpinBox()
        self.heat_transfer_spin.setRange(0.0, 1000.0)
        self.heat_transfer_spin.setDecimals(3)
        self.heat_transfer_spin.setSingleStep(0.1)
        self.heat_transfer_spin.setValue(self.properties["heat_transfer_coefficient"])
        self.heat_transfer_spin.setSuffix(" W/(m²·K)")
        self.layout.addRow("Heat Transfer Coefficient:", self.heat_transfer_spin)
        
        # Enable mass transfer
        self.mass_transfer_check = QCheckBox()
        self.mass_transfer_check.setChecked(self.properties["enable_mass_transfer"])
        self.layout.addRow("Enable Mass Transfer:", self.mass_transfer_check)
        
        # Enable heat transfer
        self.heat_transfer_check = QCheckBox()
        self.heat_transfer_check.setChecked(self.properties["enable_heat_transfer"])
        self.layout.addRow("Enable Heat Transfer:", self.heat_transfer_check)
        
    def _connect_signals(self):
        """Connect signals to slots."""
        # Connect all inputs to property_changed slot
        self.surface_tension_spin.valueChanged.connect(self._on_property_changed)
        self.contact_angle_spin.valueChanged.connect(self._on_property_changed)
        self.diffusion_spin.valueChanged.connect(self._on_property_changed)
        self.heat_transfer_spin.valueChanged.connect(self._on_property_changed)
        self.mass_transfer_check.toggled.connect(self._on_property_changed)
        self.heat_transfer_check.toggled.connect(self._on_property_changed)
        
    def _on_property_changed(self):
        """Handle property value changes."""
        # Update properties
        self.properties["surface_tension"] = self.surface_tension_spin.value()
        self.properties["contact_angle"] = self.contact_angle_spin.value()
        self.properties["diffusion_coefficient"] = self.diffusion_spin.value()
        self.properties["heat_transfer_coefficient"] = self.heat_transfer_spin.value()
        self.properties["enable_mass_transfer"] = self.mass_transfer_check.isChecked()
        self.properties["enable_heat_transfer"] = self.heat_transfer_check.isChecked()
        
        # Emit signal
        self.properties_changed.emit(self.phase1_name, self.phase2_name, self.properties)
        
    def get_properties(self) -> Dict[str, Any]:
        """
        Get the current interaction properties.
        
        Returns:
            Dict[str, Any]: Interaction properties
        """
        return self.properties
    
    def set_properties(self, properties: Dict[str, Any]):
        """
        Set the interaction properties.
        
        Args:
            properties (Dict[str, Any]): Properties to set
        """
        # Update stored properties
        self.properties.update(properties)
        
        # Update UI
        if "surface_tension" in properties:
            self.surface_tension_spin.setValue(properties["surface_tension"])
            
        if "contact_angle" in properties:
            self.contact_angle_spin.setValue(properties["contact_angle"])
            
        if "diffusion_coefficient" in properties:
            self.diffusion_spin.setValue(properties["diffusion_coefficient"])
            
        if "heat_transfer_coefficient" in properties:
            self.heat_transfer_spin.setValue(properties["heat_transfer_coefficient"])
            
        if "enable_mass_transfer" in properties:
            self.mass_transfer_check.setChecked(properties["enable_mass_transfer"])
            
        if "enable_heat_transfer" in properties:
            self.heat_transfer_check.setChecked(properties["enable_heat_transfer"])
    
    def update_phase_names(self, phase1_name: str, phase2_name: str):
        """
        Update the phase names.
        
        Args:
            phase1_name (str): New name for first phase
            phase2_name (str): New name for second phase
        """
        # Update names
        self.phase1_name = phase1_name
        self.phase2_name = phase2_name
        
        # Update title
        self.setTitle(f"Interaction: {phase1_name} - {phase2_name}")


class PhaseEditor(QWidget):
    """
    Main widget for editing multiple fluid phases.
    
    This widget provides a comprehensive interface for configuring multiple
    fluid phases and their interactions for multiphase CFD simulations.
    """
    
    # Signal emitted when phase configuration changes
    configuration_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """
        Initialize the phase editor widget.
        
        Args:
            parent: Parent widget
        """
        super(PhaseEditor, self).__init__(parent)
        
        # Store phase configuration
        self.phases = {}  # Dictionary of phase name -> properties
        self.interactions = {}  # Dictionary of (phase1, phase2) -> properties
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
        
        # Add initial phases
        self._add_default_phases()
        
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(10)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        
        # 1. Phases tab
        phases_widget = QWidget()
        phases_layout = QVBoxLayout(phases_widget)
        
        # Phase list and control buttons
        phase_list_layout = QHBoxLayout()
        
        # Phase list
        phase_list_group = QGroupBox("Phases")
        phase_list_inner = QVBoxLayout(phase_list_group)
        
        self.phase_list = QListWidget()
        self.phase_list.setMaximumWidth(200)
        self.phase_list.setSelectionMode(QListWidget.SingleSelection)
        phase_list_inner.addWidget(self.phase_list)
        
        # Phase control buttons
        phase_buttons_layout = QHBoxLayout()
        
        self.add_phase_button = QPushButton("Add")
        phase_buttons_layout.addWidget(self.add_phase_button)
        
        self.remove_phase_button = QPushButton("Remove")
        phase_buttons_layout.addWidget(self.remove_phase_button)
        
        phase_list_inner.addLayout(phase_buttons_layout)
        
        phase_list_layout.addWidget(phase_list_group)
        
        # Phase properties (will be added dynamically)
        self.phase_properties_scroll = QScrollArea()
        self.phase_properties_scroll.setWidgetResizable(True)
        self.phase_properties_scroll.setMinimumWidth(400)
        
        self.phase_properties_container = QWidget()
        self.phase_properties_layout = QVBoxLayout(self.phase_properties_container)
        self.phase_properties_layout.setAlignment(Qt.AlignTop)
        
        self.phase_properties_scroll.setWidget(self.phase_properties_container)
        
        phase_list_layout.addWidget(self.phase_properties_scroll, 1)
        
        phases_layout.addLayout(phase_list_layout)
        
        self.tab_widget.addTab(phases_widget, "Phases")
        
        # 2. Interactions tab
        interactions_widget = QScrollArea()
        interactions_widget.setWidgetResizable(True)
        
        self.interactions_container = QWidget()
        self.interactions_layout = QVBoxLayout(self.interactions_container)
        self.interactions_layout.setAlignment(Qt.AlignTop)
        
        interactions_widget.setWidget(self.interactions_container)
        
        self.tab_widget.addTab(interactions_widget, "Interactions")
        
        # Add tab widget to main layout
        self.layout.addWidget(self.tab_widget)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        self.load_button = QPushButton("Load Configuration")
        buttons_layout.addWidget(self.load_button)
        
        self.save_button = QPushButton("Save Configuration")
        buttons_layout.addWidget(self.save_button)
        
        self.apply_button = QPushButton("Apply Configuration")
        buttons_layout.addWidget(self.apply_button)
        
        self.layout.addLayout(buttons_layout)
        
    def _connect_signals(self):
        """Connect signals to slots."""
        # Connect phase list selection
        self.phase_list.currentItemChanged.connect(self._on_phase_selected)
        
        # Connect phase control buttons
        self.add_phase_button.clicked.connect(self._on_add_phase)
        self.remove_phase_button.clicked.connect(self._on_remove_phase)
        
        # Connect action buttons
        self.load_button.clicked.connect(self._on_load_configuration)
        self.save_button.clicked.connect(self._on_save_configuration)
        self.apply_button.clicked.connect(self._on_apply_configuration)
        
    def _add_default_phases(self):
        """Add default phases."""
        # Add water phase
        water_properties = {
            "name": "Water",
            "type": "liquid",
            "density": 998.0,
            "viscosity": 0.001,
            "specific_heat": 4182.0,
            "thermal_conductivity": 0.6,
            "molar_weight": 18.01528,
            "compressibility": 4.5e-10,
            "color": QColor(0, 120, 255, 180)
        }
        self._add_phase("Water", water_properties)
        
        # Add air phase
        air_properties = {
            "name": "Air",
            "type": "gas",
            "density": 1.225,
            "viscosity": 1.81e-5,
            "specific_heat": 1005.0,
            "thermal_conductivity": 0.024,
            "molar_weight": 28.97,
            "compressibility": 1.0,
            "color": QColor(220, 220, 220, 100)
        }
        self._add_phase("Air", air_properties)
        
        # Add interaction
        interaction_properties = {
            "surface_tension": 0.072,
            "contact_angle": 90.0,
            "diffusion_coefficient": 1e-9,
            "heat_transfer_coefficient": 0.1,
            "enable_mass_transfer": False,
            "enable_heat_transfer": True
        }
        self._add_interaction("Water", "Air", interaction_properties)
        
        # Select first phase
        self.phase_list.setCurrentRow(0)
        
    def _add_phase(self, name: str, properties: Dict[str, Any] = None):
        """
        Add a new phase.
        
        Args:
            name (str): Phase name
            properties (Dict[str, Any], optional): Phase properties
        """
        # Ensure unique name
        if name in self.phases:
            i = 1
            while f"{name} {i}" in self.phases:
                i += 1
            name = f"{name} {i}"
            
        # Create default properties if not provided
        if properties is None:
            properties = {
                "name": name,
                "type": "liquid",
                "density": 1000.0,
                "viscosity": 0.001,
                "specific_heat": 4200.0,
                "thermal_conductivity": 0.6,
                "molar_weight": 18.0,
                "compressibility": 4.5e-10,
                "color": QColor(0, 120, 255, 150)
            }
        
        # Store phase
        self.phases[name] = properties
        
        # Add to list
        self.phase_list.addItem(name)
        
        # Create properties widget
        phase_properties = PhaseProperties(name)
        phase_properties.set_properties(properties)
        phase_properties.properties_changed.connect(self._on_phase_properties_changed)
        
        # Store widget reference
        properties["widget"] = phase_properties
        
        # Add to properties layout
        self.phase_properties_layout.addWidget(phase_properties)
        phase_properties.hide()  # Hide initially
        
        # Add interactions with existing phases
        for other_name in self.phases:
            if other_name != name:
                self._add_interaction(name, other_name)
                
        # Update phase list selection
        for i in range(self.phase_list.count()):
            if self.phase_list.item(i).text() == name:
                self.phase_list.setCurrentRow(i)
                break
        
    def _remove_phase(self, name: str):
        """
        Remove a phase.
        
        Args:
            name (str): Phase name
        """
        if name not in self.phases:
            return
            
        # Remove phase
        phase_widget = self.phases[name].get("widget")
        if phase_widget:
            self.phase_properties_layout.removeWidget(phase_widget)
            phase_widget.deleteLater()
            
        del self.phases[name]
        
        # Remove from list
        for i in range(self.phase_list.count()):
            if self.phase_list.item(i).text() == name:
                self.phase_list.takeItem(i)
                break
                
        # Remove interactions
        interactions_to_remove = []
        for (phase1, phase2) in self.interactions:
            if phase1 == name or phase2 == name:
                interactions_to_remove.append((phase1, phase2))
                
                # Remove widget
                interaction_widget = self.interactions[(phase1, phase2)].get("widget")
                if interaction_widget:
                    self.interactions_layout.removeWidget(interaction_widget)
                    interaction_widget.deleteLater()
                    
        for interaction in interactions_to_remove:
            del self.interactions[interaction]
            
        # Select first phase if available
        if self.phase_list.count() > 0:
            self.phase_list.setCurrentRow(0)
            
    def _add_interaction(self, phase1_name: str, phase2_name: str, properties: Dict[str, Any] = None):
        """
        Add an interaction between two phases.
        
        Args:
            phase1_name (str): First phase name
            phase2_name (str): Second phase name
            properties (Dict[str, Any], optional): Interaction properties
        """
        # Sort phase names to ensure consistent key
        if phase1_name > phase2_name:
            phase1_name, phase2_name = phase2_name, phase1_name
            
        # Check if interaction already exists
        if (phase1_name, phase2_name) in self.interactions:
            return
            
        # Create default properties if not provided
        if properties is None:
            properties = {
                "surface_tension": 0.072,
                "contact_angle": 90.0,
                "diffusion_coefficient": 1e-9,
                "heat_transfer_coefficient": 0.1,
                "enable_mass_transfer": False,
                "enable_heat_transfer": True
            }
            
        # Store interaction
        self.interactions[(phase1_name, phase2_name)] = properties
        
        # Create interaction widget
        interaction_widget = PhaseInteraction(phase1_name, phase2_name)
        interaction_widget.set_properties(properties)
        interaction_widget.properties_changed.connect(self._on_interaction_properties_changed)
        
        # Store widget reference
        properties["widget"] = interaction_widget
        
        # Add to interactions layout
        self.interactions_layout.addWidget(interaction_widget)
        
    def _on_phase_selected(self, current, previous):
        """
        Handle phase selection in the list.
        
        Args:
            current: Current selected item
            previous: Previously selected item
        """
        # Hide all phase property widgets
        for name, properties in self.phases.items():
            widget = properties.get("widget")
            if widget:
                widget.hide()
                
        # Show selected phase properties
        if current:
            name = current.text()
            widget = self.phases[name].get("widget")
            if widget:
                widget.show()
                
    def _on_add_phase(self):
        """Handle add phase button click."""
        # Prompt for name
        name, ok = QInputDialog.getText(
            self,
            "Add Phase",
            "Enter name for new phase:"
        )
        
        if ok and name:
            self._add_phase(name)
            
    def _on_remove_phase(self):
        """Handle remove phase button click."""
        current_item = self.phase_list.currentItem()
        if current_item:
            name = current_item.text()
            
            # Confirm removal
            reply = QMessageBox.question(
                self,
                "Remove Phase",
                f"Are you sure you want to remove phase '{name}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self._remove_phase(name)
                
    def _on_phase_properties_changed(self, name: str, properties: Dict[str, Any]):
        """
        Handle phase property changes.
        
        Args:
            name (str): Phase name
            properties (Dict[str, Any]): New phase properties
        """
        # Check if name has changed
        old_name = None
        for n in list(self.phases.keys()):
            if self.phases[n].get("widget") == properties["widget"]:
                if n != name:
                    old_name = n
                break
                
        if old_name:
            # Update phase list
            for i in range(self.phase_list.count()):
                if self.phase_list.item(i).text() == old_name:
                    self.phase_list.item(i).setText(name)
                    break
                    
            # Update interactions
            interactions_to_update = []
            for (phase1, phase2) in self.interactions:
                if phase1 == old_name or phase2 == old_name:
                    interactions_to_update.append((phase1, phase2))
                    
            for (phase1, phase2) in interactions_to_update:
                # Create new key
                new_phase1 = name if phase1 == old_name else phase1
                new_phase2 = name if phase2 == old_name else phase2
                
                # Sort to ensure consistent key
                if new_phase1 > new_phase2:
                    new_phase1, new_phase2 = new_phase2, new_phase1
                    
                # Only update if the new key doesn't exist yet
                if (new_phase1, new_phase2) not in self.interactions:
                    # Move interaction
                    self.interactions[(new_phase1, new_phase2)] = self.interactions[(phase1, phase2)]
                    del self.interactions[(phase1, phase2)]
                    
                    # Update widget
                    widget = self.interactions[(new_phase1, new_phase2)].get("widget")
                    if widget:
                        widget.update_phase_names(new_phase1, new_phase2)
                        
            # Update phase dictionary
            self.phases[name] = self.phases[old_name]
            del self.phases[old_name]
            
        # Update properties
        self.phases[name].update(properties)
        
        # Emit configuration changed signal
        self.configuration_changed.emit(self.get_configuration())
        
    def _on_interaction_properties_changed(self, phase1_name: str, phase2_name: str, properties: Dict[str, Any]):
        """
        Handle interaction property changes.
        
        Args:
            phase1_name (str): First phase name
            phase2_name (str): Second phase name
            properties (Dict[str, Any]): New interaction properties
        """
        # Sort phase names to ensure consistent key
        if phase1_name > phase2_name:
            phase1_name, phase2_name = phase2_name, phase1_name
            
        # Update properties
        self.interactions[(phase1_name, phase2_name)].update(properties)
        
        # Emit configuration changed signal
        self.configuration_changed.emit(self.get_configuration())
        
    def _on_load_configuration(self):
        """Handle load configuration button click."""
        # Show file dialog
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Phase Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not filepath:
            return
            
        try:
            # Load configuration
            import json
            with open(filepath, 'r') as f:
                config = json.load(f)
                
            # Apply configuration
            self.set_configuration(config)
            
            QMessageBox.information(
                self,
                "Configuration Loaded",
                f"Phase configuration loaded from {filepath}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Configuration",
                f"An error occurred while loading the configuration:\n\n{str(e)}"
            )
            
    def _on_save_configuration(self):
        """Handle save configuration button click."""
        # Show file dialog
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Phase Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not filepath:
            return
            
        # Ensure .json extension
        if not filepath.lower().endswith('.json'):
            filepath += '.json'
            
        try:
            # Get configuration
            config = self.get_configuration()
            
            # Convert QColor objects to serializable format
            for phase_name, phase in config["phases"].items():
                if "color" in phase and isinstance(phase["color"], QColor):
                    phase["color"] = {
                        "r": phase["color"].red(),
                        "g": phase["color"].green(),
                        "b": phase["color"].blue(),
                        "a": phase["color"].alpha()
                    }
                    
            # Save configuration
            import json
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=4)
                
            QMessageBox.information(
                self,
                "Configuration Saved",
                f"Phase configuration saved to {filepath}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Saving Configuration",
                f"An error occurred while saving the configuration:\n\n{str(e)}"
            )
            
    def _on_apply_configuration(self):
        """Handle apply configuration button click."""
        # Emit configuration changed signal
        self.configuration_changed.emit(self.get_configuration())
        
        QMessageBox.information(
            self,
            "Configuration Applied",
            "Phase configuration has been applied to the simulation."
        )
        
    def get_configuration(self) -> Dict[str, Any]:
        """
        Get the current phase configuration.
        
        Returns:
            Dict[str, Any]: Phase configuration
        """
        # Create configuration dictionary
        config = {
            "phases": {},
            "interactions": {}
        }
        
        # Add phases
        for name, properties in self.phases.items():
            # Create a copy of properties without the widget
            phase_props = properties.copy()
            if "widget" in phase_props:
                del phase_props["widget"]
                
            config["phases"][name] = phase_props
            
        # Add interactions
        for (phase1, phase2), properties in self.interactions.items():
            # Create a copy of properties without the widget
            interaction_props = properties.copy()
            if "widget" in interaction_props:
                del interaction_props["widget"]
                
            config["interactions"][(phase1, phase2)] = interaction_props
            
        return config
        
    def set_configuration(self, config: Dict[str, Any]):
        """
        Set the phase configuration.
        
        Args:
            config (Dict[str, Any]): Phase configuration
        """
        # Clear existing configuration
        self._clear_configuration()
        
        # Add phases
        for name, properties in config.get("phases", {}).items():
            # Convert color if needed
            if "color" in properties and isinstance(properties["color"], dict):
                color_dict = properties["color"]
                properties["color"] = QColor(
                    color_dict.get("r", 0),
                    color_dict.get("g", 0),
                    color_dict.get("b", 0),
                    color_dict.get("a", 255)
                )
                
            self._add_phase(name, properties)
            
        # Add interactions
        for interaction_key, properties in config.get("interactions", {}).items():
            # Convert key if needed
            if isinstance(interaction_key, str):
                # Try to parse string key like "phase1,phase2"
                parts = interaction_key.split(",")
                if len(parts) == 2:
                    phase1, phase2 = parts
                else:
                    continue
            else:
                # Assume tuple key
                phase1, phase2 = interaction_key
                
            self._add_interaction(phase1, phase2, properties)
            
        # Select first phase if available
        if self.phase_list.count() > 0:
            self.phase_list.setCurrentRow(0)
            
    def _clear_configuration(self):
        """Clear the current configuration."""
        # Clear phases
        for name in list(self.phases.keys()):
            self._remove_phase(name)
            
        # Clear interactions
        for key in list(self.interactions.keys()):
            phase1, phase2 = key
            interaction_widget = self.interactions[key].get("widget")
            if interaction_widget:
                self.interactions_layout.removeWidget(interaction_widget)
                interaction_widget.deleteLater()
                
        self.interactions.clear()