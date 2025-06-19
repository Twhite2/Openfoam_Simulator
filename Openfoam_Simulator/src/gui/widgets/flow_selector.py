#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow selector widget for Openfoam_Simulator application.

This widget provides a UI for selecting and configuring different types of flows
for CFD simulations in the oil & gas industry, including:
- Single phase flows (gas, oil, water)
- Multi-phase flows (oil-water, gas-liquid, three-phase)
- Special cases like pigging and spill simulations
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QCheckBox, QRadioButton,
    QButtonGroup, QGroupBox, QTabWidget, QSpinBox, QDoubleSpinBox,
    QPushButton, QFrame, QScrollArea, QSizePolicy, QSlider,
    QToolTip, QMenu, QAction, QColorDialog
)
from PyQt5.QtCore import (
    Qt, QSize, pyqtSignal, QTimer, QPoint, QSettings
)
from PyQt5.QtGui import (
    QIcon, QPixmap, QColor, QPalette, QFont, QPainter, QBrush
)

# Import utility modules
from ...utils.logger import get_logger
from ...config import get_value, set_value

logger = get_logger(__name__)


class FluidProperties:
    """Class to store physical properties of fluids."""
    
    def __init__(self, name: str = "Water", 
                density: float = 1000.0, 
                viscosity: float = 0.001,
                specific_heat: float = 4182.0,
                thermal_conductivity: float = 0.6,
                color: QColor = QColor(0, 0, 255, 150)):
        """
        Initialize fluid properties.
        
        Args:
            name (str): Fluid name
            density (float): Density in kg/m³
            viscosity (float): Dynamic viscosity in Pa·s
            specific_heat (float): Specific heat in J/(kg·K)
            thermal_conductivity (float): Thermal conductivity in W/(m·K)
            color (QColor): Color for visualization
        """
        self.name = name
        self.density = density
        self.viscosity = viscosity
        self.specific_heat = specific_heat
        self.thermal_conductivity = thermal_conductivity
        self.color = color


class PhaseProperties(QGroupBox):
    """
    Widget for configuring properties of a single phase.
    
    This widget allows setting fluid type, volume fraction, and physical properties
    for a single phase in a multi-phase flow simulation.
    """
    
    # Signal emitted when properties are changed
    properties_changed = pyqtSignal(str, dict)
    
    def __init__(self, title: str = "Phase", parent=None):
        """
        Initialize the phase properties widget.
        
        Args:
            title (str): Title for the group box
            parent: Parent widget
        """
        super(PhaseProperties, self).__init__(title, parent)
        
        # Default fluid properties
        self.fluids = {
            "Water": FluidProperties("Water", 1000.0, 0.001, 4182.0, 0.6, QColor(0, 0, 255, 150)),
            "Oil": FluidProperties("Oil", 850.0, 0.03, 1800.0, 0.15, QColor(150, 75, 0, 150)),
            "Crude Oil": FluidProperties("Crude Oil", 900.0, 0.05, 1900.0, 0.12, QColor(50, 25, 0, 150)),
            "Natural Gas": FluidProperties("Natural Gas", 0.8, 1.8e-5, 2200.0, 0.026, QColor(200, 200, 200, 100)),
            "Air": FluidProperties("Air", 1.2, 1.8e-5, 1005.0, 0.025, QColor(200, 200, 250, 50)),
            "Methane": FluidProperties("Methane", 0.66, 1.1e-5, 2220.0, 0.033, QColor(220, 220, 180, 100)),
            "Custom": FluidProperties("Custom", 1000.0, 0.001, 4000.0, 0.5, QColor(150, 150, 150, 150))
        }
        
        # Current fluid
        self.current_fluid = "Water"
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
        
        # Initialize fields with default values
        self._update_fields()
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        self.layout = QFormLayout(self)
        self.layout.setContentsMargins(10, 15, 10, 10)
        self.layout.setSpacing(8)
        
        # Fluid type
        self.fluid_combo = QComboBox()
        self.fluid_combo.addItems(list(self.fluids.keys()))
        self.layout.addRow("Fluid:", self.fluid_combo)
        
        # Volume fraction
        self.fraction_spin = QDoubleSpinBox()
        self.fraction_spin.setRange(0.0, 1.0)
        self.fraction_spin.setDecimals(3)
        self.fraction_spin.setSingleStep(0.01)
        self.fraction_spin.setValue(1.0)
        self.layout.addRow("Volume Fraction:", self.fraction_spin)
        
        # Density
        self.density_spin = QDoubleSpinBox()
        self.density_spin.setRange(0.1, 10000.0)
        self.density_spin.setDecimals(2)
        self.density_spin.setSingleStep(10.0)
        self.density_spin.setSuffix(" kg/m³")
        self.layout.addRow("Density:", self.density_spin)
        
        # Viscosity
        self.viscosity_spin = QDoubleSpinBox()
        self.viscosity_spin.setRange(1e-6, 1000.0)
        self.viscosity_spin.setDecimals(6)
        self.viscosity_spin.setSingleStep(0.001)
        self.viscosity_spin.setSuffix(" Pa·s")
        self.layout.addRow("Viscosity:", self.viscosity_spin)
        
        # Specific heat
        self.specific_heat_spin = QDoubleSpinBox()
        self.specific_heat_spin.setRange(1.0, 10000.0)
        self.specific_heat_spin.setDecimals(1)
        self.specific_heat_spin.setSingleStep(100.0)
        self.specific_heat_spin.setSuffix(" J/(kg·K)")
        self.layout.addRow("Specific Heat:", self.specific_heat_spin)
        
        # Thermal conductivity
        self.thermal_cond_spin = QDoubleSpinBox()
        self.thermal_cond_spin.setRange(0.001, 1000.0)
        self.thermal_cond_spin.setDecimals(3)
        self.thermal_cond_spin.setSingleStep(0.01)
        self.thermal_cond_spin.setSuffix(" W/(m·K)")
        self.layout.addRow("Thermal Conductivity:", self.thermal_cond_spin)
        
        # Color for visualization
        self.color_button = QPushButton("Select...")
        self.color_button.setFixedWidth(80)
        self.layout.addRow("Color:", self.color_button)
        
        # Temperature
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 1000.0)
        self.temperature_spin.setDecimals(1)
        self.temperature_spin.setSingleStep(10.0)
        self.temperature_spin.setValue(293.15)  # 20°C in Kelvin
        self.temperature_spin.setSuffix(" K")
        self.layout.addRow("Temperature:", self.temperature_spin)
    
    def _connect_signals(self):
        """Connect signals to slots."""
        # Connect fluid combo
        self.fluid_combo.currentTextChanged.connect(self._on_fluid_changed)
        
        # Connect input fields for custom values
        self.density_spin.valueChanged.connect(self._on_property_changed)
        self.viscosity_spin.valueChanged.connect(self._on_property_changed)
        self.specific_heat_spin.valueChanged.connect(self._on_property_changed)
        self.thermal_cond_spin.valueChanged.connect(self._on_property_changed)
        self.fraction_spin.valueChanged.connect(self._on_property_changed)
        self.temperature_spin.valueChanged.connect(self._on_property_changed)
        
        # Connect color button
        self.color_button.clicked.connect(self._on_color_button_clicked)
    
    def _update_fields(self):
        """Update fields based on current fluid."""
        fluid = self.fluids[self.current_fluid]
        
        # Update spin boxes with fluid properties
        self.density_spin.setValue(fluid.density)
        self.viscosity_spin.setValue(fluid.viscosity)
        self.specific_heat_spin.setValue(fluid.specific_heat)
        self.thermal_cond_spin.setValue(fluid.thermal_conductivity)
        
        # Update color button
        self._update_color_button()
        
        # Emit signal
        self._on_property_changed()
    
    def _update_color_button(self):
        """Update color button with current color."""
        color = self.fluids[self.current_fluid].color
        
        # Create a 16x16 pixmap with the color
        pixmap = QPixmap(16, 16)
        pixmap.fill(color)
        
        # Set as button icon
        self.color_button.setIcon(QIcon(pixmap))
    
    def _on_fluid_changed(self, fluid_name: str):
        """
        Handle fluid type change.
        
        Args:
            fluid_name (str): Name of the selected fluid
        """
        # Update current fluid
        self.current_fluid = fluid_name
        
        # Enable/disable property fields
        is_custom = (fluid_name == "Custom")
        self.density_spin.setReadOnly(not is_custom)
        self.viscosity_spin.setReadOnly(not is_custom)
        self.specific_heat_spin.setReadOnly(not is_custom)
        self.thermal_cond_spin.setReadOnly(not is_custom)
        
        # Update fields
        self._update_fields()
    
    def _on_property_changed(self):
        """Handle property value change."""
        # Update fluid properties if custom
        if self.current_fluid == "Custom":
            fluid = self.fluids["Custom"]
            fluid.density = self.density_spin.value()
            fluid.viscosity = self.viscosity_spin.value()
            fluid.specific_heat = self.specific_heat_spin.value()
            fluid.thermal_conductivity = self.thermal_cond_spin.value()
        
        # Emit signal with current properties
        properties = {
            "fluid": self.current_fluid,
            "fraction": self.fraction_spin.value(),
            "density": self.density_spin.value(),
            "viscosity": self.viscosity_spin.value(),
            "specific_heat": self.specific_heat_spin.value(),
            "thermal_conductivity": self.thermal_cond_spin.value(),
            "temperature": self.temperature_spin.value(),
            "color": self.fluids[self.current_fluid].color
        }
        
        self.properties_changed.emit(self.title(), properties)
    
    def _on_color_button_clicked(self):
        """Handle color button click."""
        # Get current color
        current_color = self.fluids[self.current_fluid].color
        
        # Show color dialog
        color = QColorDialog.getColor(
            current_color, 
            self, 
            "Select Color",
            QColorDialog.ShowAlphaChannel
        )
        
        if color.isValid():
            # Update fluid color
            self.fluids[self.current_fluid].color = color
            
            # Update color button
            self._update_color_button()
            
            # Emit signal
            self._on_property_changed()
    
    def get_properties(self) -> Dict[str, Any]:
        """
        Get current phase properties.
        
        Returns:
            Dict[str, Any]: Current properties
        """
        return {
            "fluid": self.current_fluid,
            "fraction": self.fraction_spin.value(),
            "density": self.density_spin.value(),
            "viscosity": self.viscosity_spin.value(),
            "specific_heat": self.specific_heat_spin.value(),
            "thermal_conductivity": self.thermal_cond_spin.value(),
            "temperature": self.temperature_spin.value(),
            "color": self.fluids[self.current_fluid].color
        }
    
    def set_properties(self, properties: Dict[str, Any]):
        """
        Set phase properties.
        
        Args:
            properties (Dict[str, Any]): Properties to set
        """
        # Set fluid type if provided
        if "fluid" in properties and properties["fluid"] in self.fluids:
            self.fluid_combo.setCurrentText(properties["fluid"])
            self.current_fluid = properties["fluid"]
        
        # Set individual properties
        if "fraction" in properties:
            self.fraction_spin.setValue(properties["fraction"])
        
        if "density" in properties:
            self.density_spin.setValue(properties["density"])
        
        if "viscosity" in properties:
            self.viscosity_spin.setValue(properties["viscosity"])
        
        if "specific_heat" in properties:
            self.specific_heat_spin.setValue(properties["specific_heat"])
        
        if "thermal_conductivity" in properties:
            self.thermal_cond_spin.setValue(properties["thermal_conductivity"])
        
        if "temperature" in properties:
            self.temperature_spin.setValue(properties["temperature"])
        
        if "color" in properties and isinstance(properties["color"], QColor):
            self.fluids[self.current_fluid].color = properties["color"]
            self._update_color_button()
        
        # Emit signal
        self._on_property_changed()


class TwoPhaseWidget(QWidget):
    """
    Widget for configuring a two-phase flow.
    
    This widget provides controls for setting up a two-phase flow simulation,
    such as oil-water or gas-liquid flows commonly found in oil & gas applications.
    """
    
    # Signal emitted when properties are changed
    properties_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """
        Initialize the two-phase flow widget.
        
        Args:
            parent: Parent widget
        """
        super(TwoPhaseWidget, self).__init__(parent)
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(10)
        
        # Flow type selection
        flow_type_group = QGroupBox("Flow Type")
        flow_type_layout = QVBoxLayout(flow_type_group)
        
        self.oil_water_radio = QRadioButton("Oil-Water")
        self.gas_liquid_radio = QRadioButton("Gas-Liquid")
        
        flow_type_layout.addWidget(self.oil_water_radio)
        flow_type_layout.addWidget(self.gas_liquid_radio)
        
        # Create button group
        self.flow_type_group = QButtonGroup()
        self.flow_type_group.addButton(self.oil_water_radio, 0)
        self.flow_type_group.addButton(self.gas_liquid_radio, 1)
        
        # Default selection
        self.oil_water_radio.setChecked(True)
        
        self.layout.addWidget(flow_type_group)
        
        # Interface physics
        interface_group = QGroupBox("Interface Physics")
        interface_layout = QFormLayout(interface_group)
        
        # Surface tension
        self.surface_tension_spin = QDoubleSpinBox()
        self.surface_tension_spin.setRange(0.001, 1.0)
        self.surface_tension_spin.setDecimals(3)
        self.surface_tension_spin.setSingleStep(0.001)
        self.surface_tension_spin.setValue(0.072)  # Default for water-air
        self.surface_tension_spin.setSuffix(" N/m")
        interface_layout.addRow("Surface Tension:", self.surface_tension_spin)
        
        # Interface thickness
        self.interface_thickness_spin = QDoubleSpinBox()
        self.interface_thickness_spin.setRange(0.0, 0.1)
        self.interface_thickness_spin.setDecimals(4)
        self.interface_thickness_spin.setSingleStep(0.001)
        self.interface_thickness_spin.setValue(0.0015)
        self.interface_thickness_spin.setSuffix(" m")
        interface_layout.addRow("Interface Thickness:", self.interface_thickness_spin)
        
        # Contact angle (for wall interaction)
        self.contact_angle_spin = QDoubleSpinBox()
        self.contact_angle_spin.setRange(0.0, 180.0)
        self.contact_angle_spin.setDecimals(1)
        self.contact_angle_spin.setSingleStep(5.0)
        self.contact_angle_spin.setValue(90.0)
        self.contact_angle_spin.setSuffix("°")
        interface_layout.addRow("Contact Angle:", self.contact_angle_spin)
        
        self.layout.addWidget(interface_group)
        
        # Phase properties
        self.phase1_properties = PhaseProperties("Phase 1")
        self.phase2_properties = PhaseProperties("Phase 2")
        
        # Initialize phase 2
        self.phase2_properties.fluid_combo.setCurrentText("Oil")
        self.phase2_properties.fraction_spin.setValue(0.3)
        
        self.layout.addWidget(self.phase1_properties)
        self.layout.addWidget(self.phase2_properties)
        
        # Add stretch to push widgets to the top
        self.layout.addStretch()
    
    def _connect_signals(self):
        """Connect signals to slots."""
        # Connect flow type radio buttons
        self.flow_type_group.buttonClicked.connect(self._on_flow_type_changed)
        
        # Connect interface physics inputs
        self.surface_tension_spin.valueChanged.connect(self._on_property_changed)
        self.interface_thickness_spin.valueChanged.connect(self._on_property_changed)
        self.contact_angle_spin.valueChanged.connect(self._on_property_changed)
        
        # Connect phase properties
        self.phase1_properties.properties_changed.connect(self._on_phase_changed)
        self.phase2_properties.properties_changed.connect(self._on_phase_changed)
    
    def _on_flow_type_changed(self, button):
        """
        Handle flow type change.
        
        Args:
            button: The selected radio button
        """
        # Update phases based on flow type
        if button == self.oil_water_radio:
            # Oil-water flow
            if self.phase1_properties.current_fluid == "Natural Gas" or self.phase1_properties.current_fluid == "Air":
                self.phase1_properties.fluid_combo.setCurrentText("Water")
            
            if self.phase2_properties.current_fluid == "Natural Gas" or self.phase2_properties.current_fluid == "Air":
                self.phase2_properties.fluid_combo.setCurrentText("Oil")
            
            # Update surface tension
            self.surface_tension_spin.setValue(0.025)  # Oil-water
            
        elif button == self.gas_liquid_radio:
            # Gas-liquid flow
            if self.phase1_properties.current_fluid != "Natural Gas" and self.phase1_properties.current_fluid != "Air":
                self.phase1_properties.fluid_combo.setCurrentText("Water")
            
            if self.phase2_properties.current_fluid != "Natural Gas" and self.phase2_properties.current_fluid != "Air":
                self.phase2_properties.fluid_combo.setCurrentText("Natural Gas")
            
            # Update surface tension
            self.surface_tension_spin.setValue(0.072)  # Water-air
        
        # Emit signal
        self._on_property_changed()
    
    def _on_phase_changed(self, phase_name: str, phase_properties: Dict[str, Any]):
        """
        Handle phase property change.
        
        Args:
            phase_name (str): Name of the phase
            phase_properties (Dict[str, Any]): Phase properties
        """
        # Update phase fractions to ensure they sum to 1.0
        if phase_name == "Phase 1":
            # If phase 1 fraction changed, adjust phase 2
            phase1_fraction = phase_properties["fraction"]
            if phase1_fraction > 1.0:
                phase1_fraction = 1.0
                self.phase1_properties.fraction_spin.setValue(phase1_fraction)
            
            phase2_fraction = 1.0 - phase1_fraction
            # Block signals to avoid recursive updates
            self.phase2_properties.fraction_spin.blockSignals(True)
            self.phase2_properties.fraction_spin.setValue(phase2_fraction)
            self.phase2_properties.fraction_spin.blockSignals(False)
            
        elif phase_name == "Phase 2":
            # If phase 2 fraction changed, adjust phase 1
            phase2_fraction = phase_properties["fraction"]
            if phase2_fraction > 1.0:
                phase2_fraction = 1.0
                self.phase2_properties.fraction_spin.setValue(phase2_fraction)
            
            phase1_fraction = 1.0 - phase2_fraction
            # Block signals to avoid recursive updates
            self.phase1_properties.fraction_spin.blockSignals(True)
            self.phase1_properties.fraction_spin.setValue(phase1_fraction)
            self.phase1_properties.fraction_spin.blockSignals(False)
        
        # Emit signal with updated properties
        self._on_property_changed()
    
    def _on_property_changed(self):
        """Handle property value change."""
        # Get flow type
        flow_type = "Oil-Water" if self.oil_water_radio.isChecked() else "Gas-Liquid"
        
        # Get interface properties
        interface_props = {
            "surface_tension": self.surface_tension_spin.value(),
            "interface_thickness": self.interface_thickness_spin.value(),
            "contact_angle": self.contact_angle_spin.value()
        }
        
        # Get phase properties
        phase1_props = self.phase1_properties.get_properties()
        phase2_props = self.phase2_properties.get_properties()
        
        # Combine into a single properties dictionary
        properties = {
            "flow_type": flow_type,
            "interface": interface_props,
            "phase1": phase1_props,
            "phase2": phase2_props
        }
        
        # Emit signal
        self.properties_changed.emit(properties)
    
    def get_properties(self) -> Dict[str, Any]:
        """
        Get current two-phase flow properties.
        
        Returns:
            Dict[str, Any]: Current properties
        """
        flow_type = "Oil-Water" if self.oil_water_radio.isChecked() else "Gas-Liquid"
        
        interface_props = {
            "surface_tension": self.surface_tension_spin.value(),
            "interface_thickness": self.interface_thickness_spin.value(),
            "contact_angle": self.contact_angle_spin.value()
        }
        
        phase1_props = self.phase1_properties.get_properties()
        phase2_props = self.phase2_properties.get_properties()
        
        return {
            "flow_type": flow_type,
            "interface": interface_props,
            "phase1": phase1_props,
            "phase2": phase2_props
        }
    
    def set_properties(self, properties: Dict[str, Any]):
        """
        Set two-phase flow properties.
        
        Args:
            properties (Dict[str, Any]): Properties to set
        """
        # Set flow type if provided
        if "flow_type" in properties:
            if properties["flow_type"] == "Oil-Water":
                self.oil_water_radio.setChecked(True)
            elif properties["flow_type"] == "Gas-Liquid":
                self.gas_liquid_radio.setChecked(True)
        
        # Set interface properties
        if "interface" in properties:
            interface = properties["interface"]
            if "surface_tension" in interface:
                self.surface_tension_spin.setValue(interface["surface_tension"])
            if "interface_thickness" in interface:
                self.interface_thickness_spin.setValue(interface["interface_thickness"])
            if "contact_angle" in interface:
                self.contact_angle_spin.setValue(interface["contact_angle"])
        
        # Set phase properties
        if "phase1" in properties:
            self.phase1_properties.set_properties(properties["phase1"])
        
        if "phase2" in properties:
            self.phase2_properties.set_properties(properties["phase2"])


class ThreePhaseWidget(QWidget):
    """
    Widget for configuring a three-phase flow.
    
    This widget provides controls for setting up a three-phase flow simulation,
    typically oil-water-gas flows in oil & gas applications.
    """
    
    # Signal emitted when properties are changed
    properties_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """
        Initialize the three-phase flow widget.
        
        Args:
            parent: Parent widget
        """
        super(ThreePhaseWidget, self).__init__(parent)
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(10)
        
        # Interface physics
        interface_group = QGroupBox("Interface Physics")
        interface_layout = QGridLayout(interface_group)
        
        # Labels for clarity
        interface_layout.addWidget(QLabel("Property"), 0, 0)
        interface_layout.addWidget(QLabel("Oil-Water"), 0, 1)
        interface_layout.addWidget(QLabel("Gas-Oil"), 0, 2)
        interface_layout.addWidget(QLabel("Gas-Water"), 0, 3)
        
        # Surface tension
        interface_layout.addWidget(QLabel("Surface Tension (N/m):"), 1, 0)
        
        self.surface_tension_ow_spin = QDoubleSpinBox()
        self.surface_tension_ow_spin.setRange(0.001, 1.0)
        self.surface_tension_ow_spin.setDecimals(3)
        self.surface_tension_ow_spin.setSingleStep(0.001)
        self.surface_tension_ow_spin.setValue(0.025)  # Default for oil-water
        interface_layout.addWidget(self.surface_tension_ow_spin, 1, 1)
        
        self.surface_tension_go_spin = QDoubleSpinBox()
        self.surface_tension_go_spin.setRange(0.001, 1.0)
        self.surface_tension_go_spin.setDecimals(3)
        self.surface_tension_go_spin.setSingleStep(0.001)
        self.surface_tension_go_spin.setValue(0.023)  # Default for gas-oil
        interface_layout.addWidget(self.surface_tension_go_spin, 1, 2)
        
        self.surface_tension_gw_spin = QDoubleSpinBox()
        self.surface_tension_gw_spin.setRange(0.001, 1.0)
        self.surface_tension_gw_spin.setDecimals(3)
        self.surface_tension_gw_spin.setSingleStep(0.001)
        self.surface_tension_gw_spin.setValue(0.072)  # Default for gas-water
        interface_layout.addWidget(self.surface_tension_gw_spin, 1, 3)
        
        # Interface thickness
        interface_layout.addWidget(QLabel("Interface Thickness (m):"), 2, 0)
        
        self.thickness_ow_spin = QDoubleSpinBox()
        self.thickness_ow_spin.setRange(0.0, 0.1)
        self.thickness_ow_spin.setDecimals(4)
        self.thickness_ow_spin.setSingleStep(0.001)
        self.thickness_ow_spin.setValue(0.0015)
        interface_layout.addWidget(self.thickness_ow_spin, 2, 1)
        
        self.thickness_go_spin = QDoubleSpinBox()
        self.thickness_go_spin.setRange(0.0, 0.1)
        self.thickness_go_spin.setDecimals(4)
        self.thickness_go_spin.setSingleStep(0.001)
        self.thickness_go_spin.setValue(0.0015)
        interface_layout.addWidget(self.thickness_go_spin, 2, 2)
        
        self.thickness_gw_spin = QDoubleSpinBox()
        self.thickness_gw_spin.setRange(0.0, 0.1)
        self.thickness_gw_spin.setDecimals(4)
        self.thickness_gw_spin.setSingleStep(0.001)
        self.thickness_gw_spin.setValue(0.0015)
        interface_layout.addWidget(self.thickness_gw_spin, 2, 3)
        
        self.layout.addWidget(interface_group)
        
        # Phase properties
        self.water_properties = PhaseProperties("Water Phase")
        self.water_properties.fluid_combo.setCurrentText("Water")
        self.water_properties.fraction_spin.setValue(0.4)
        
        self.oil_properties = PhaseProperties("Oil Phase")
        self.oil_properties.fluid_combo.setCurrentText("Oil")
        self.oil_properties.fraction_spin.setValue(0.4)
        
        self.gas_properties = PhaseProperties("Gas Phase")
        self.gas_properties.fluid_combo.setCurrentText("Natural Gas")
        self.gas_properties.fraction_spin.setValue(0.2)
        
        self.layout.addWidget(self.water_properties)
        self.layout.addWidget(self.oil_properties)
        self.layout.addWidget(self.gas_properties)
        
        # Add stretch to push widgets to the top
        self.layout.addStretch()
    
    def _connect_signals(self):
        """Connect signals to slots."""
        # Connect interface physics inputs
        self.surface_tension_ow_spin.valueChanged.connect(self._on_property_changed)
        self.surface_tension_go_spin.valueChanged.connect(self._on_property_changed)
        self.surface_tension_gw_spin.valueChanged.connect(self._on_property_changed)
        self.thickness_ow_spin.valueChanged.connect(self._on_property_changed)
        self.thickness_go_spin.valueChanged.connect(self._on_property_changed)
        self.thickness_gw_spin.valueChanged.connect(self._on_property_changed)
        
        # Connect phase properties
        self.water_properties.properties_changed.connect(self._on_phase_changed)
        self.oil_properties.properties_changed.connect(self._on_phase_changed)
        self.gas_properties.properties_changed.connect(self._on_phase_changed)
    
    def _on_phase_changed(self, phase_name: str, phase_properties: Dict[str, Any]):
        """
        Handle phase property change.
        
        Args:
            phase_name (str): Name of the phase
            phase_properties (Dict[str, Any]): Phase properties
        """
        # Update phase fractions to ensure they sum to 1.0
        water_fraction = self.water_properties.fraction_spin.value()
        oil_fraction = self.oil_properties.fraction_spin.value()
        gas_fraction = self.gas_properties.fraction_spin.value()
        
        # Calculate total
        total = water_fraction + oil_fraction + gas_fraction
        
        if total > 0:
            # Normalize to ensure sum is 1.0
            if phase_name == "Water Phase":
                # If water changed, adjust oil and gas proportionally
                if water_fraction > 1.0:
                    water_fraction = 1.0
                    self.water_properties.fraction_spin.setValue(water_fraction)
                
                remainder = 1.0 - water_fraction
                if remainder > 0:
                    old_sum = oil_fraction + gas_fraction
                    if old_sum > 0:
                        # Distribute remainder proportionally
                        new_oil = remainder * (oil_fraction / old_sum)
                        new_gas = remainder * (gas_fraction / old_sum)
                    else:
                        # Equal distribution
                        new_oil = remainder / 2
                        new_gas = remainder / 2
                    
                    # Update with signals blocked
                    self.oil_properties.fraction_spin.blockSignals(True)
                    self.gas_properties.fraction_spin.blockSignals(True)
                    self.oil_properties.fraction_spin.setValue(new_oil)
                    self.gas_properties.fraction_spin.setValue(new_gas)
                    self.oil_properties.fraction_spin.blockSignals(False)
                    self.gas_properties.fraction_spin.blockSignals(False)
                else:
                    # All water
                    self.oil_properties.fraction_spin.blockSignals(True)
                    self.gas_properties.fraction_spin.blockSignals(True)
                    self.oil_properties.fraction_spin.setValue(0.0)
                    self.gas_properties.fraction_spin.setValue(0.0)
                    self.oil_properties.fraction_spin.blockSignals(False)
                    self.gas_properties.fraction_spin.blockSignals(False)
                
            elif phase_name == "Oil Phase":
                # If oil changed, adjust water and gas proportionally
                if oil_fraction > 1.0:
                    oil_fraction = 1.0
                    self.oil_properties.fraction_spin.setValue(oil_fraction)
                
                remainder = 1.0 - oil_fraction
                if remainder > 0:
                    old_sum = water_fraction + gas_fraction
                    if old_sum > 0:
                        # Distribute remainder proportionally
                        new_water = remainder * (water_fraction / old_sum)
                        new_gas = remainder * (gas_fraction / old_sum)
                    else:
                        # Equal distribution
                        new_water = remainder / 2
                        new_gas = remainder / 2
                    
                    # Update with signals blocked
                    self.water_properties.fraction_spin.blockSignals(True)
                    self.gas_properties.fraction_spin.blockSignals(True)
                    self.water_properties.fraction_spin.setValue(new_water)
                    self.gas_properties.fraction_spin.setValue(new_gas)
                    self.water_properties.fraction_spin.blockSignals(False)
                    self.gas_properties.fraction_spin.blockSignals(False)
                else:
                    # All oil
                    self.water_properties.fraction_spin.blockSignals(True)
                    self.gas_properties.fraction_spin.blockSignals(True)
                    self.water_properties.fraction_spin.setValue(0.0)
                    self.gas_properties.fraction_spin.setValue(0.0)
                    self.water_properties.fraction_spin.blockSignals(False)
                    self.gas_properties.fraction_spin.blockSignals(False)
                
            elif phase_name == "Gas Phase":
                # If gas changed, adjust water and oil proportionally
                if gas_fraction > 1.0:
                    gas_fraction = 1.0
                    self.gas_properties.fraction_spin.setValue(gas_fraction)
                
                remainder = 1.0 - gas_fraction
                if remainder > 0:
                    old_sum = water_fraction + oil_fraction
                    if old_sum > 0:
                        # Distribute remainder proportionally
                        new_water = remainder * (water_fraction / old_sum)
                        new_oil = remainder * (oil_fraction / old_sum)
                    else:
                        # Equal distribution
                        new_water = remainder / 2
                        new_oil = remainder / 2
                    
                    # Update with signals blocked
                    self.water_properties.fraction_spin.blockSignals(True)
                    self.oil_properties.fraction_spin.blockSignals(True)
                    self.water_properties.fraction_spin.setValue(new_water)
                    self.oil_properties.fraction_spin.setValue(new_oil)
                    self.water_properties.fraction_spin.blockSignals(False)
                    self.oil_properties.fraction_spin.blockSignals(False)
                else:
                    # All gas
                    self.water_properties.fraction_spin.blockSignals(True)
                    self.oil_properties.fraction_spin.blockSignals(True)
                    self.water_properties.fraction_spin.setValue(0.0)
                    self.oil_properties.fraction_spin.setValue(0.0)
                    self.water_properties.fraction_spin.blockSignals(False)
                    self.oil_properties.fraction_spin.blockSignals(False)
        
        # Emit signal with updated properties
        self._on_property_changed()
    
    def _on_property_changed(self):
        """Handle property value change."""
        # Get interface properties
        interface_props = {
            "surface_tension": {
                "oil_water": self.surface_tension_ow_spin.value(),
                "gas_oil": self.surface_tension_go_spin.value(),
                "gas_water": self.surface_tension_gw_spin.value()
            },
            "interface_thickness": {
                "oil_water": self.thickness_ow_spin.value(),
                "gas_oil": self.thickness_go_spin.value(),
                "gas_water": self.thickness_gw_spin.value()
            }
        }
        
        # Get phase properties
        water_props = self.water_properties.get_properties()
        oil_props = self.oil_properties.get_properties()
        gas_props = self.gas_properties.get_properties()
        
        # Combine into a single properties dictionary
        properties = {
            "interface": interface_props,
            "water": water_props,
            "oil": oil_props,
            "gas": gas_props
        }
        
        # Emit signal
        self.properties_changed.emit(properties)
    
    def get_properties(self) -> Dict[str, Any]:
        """
        Get current three-phase flow properties.
        
        Returns:
            Dict[str, Any]: Current properties
        """
        interface_props = {
            "surface_tension": {
                "oil_water": self.surface_tension_ow_spin.value(),
                "gas_oil": self.surface_tension_go_spin.value(),
                "gas_water": self.surface_tension_gw_spin.value()
            },
            "interface_thickness": {
                "oil_water": self.thickness_ow_spin.value(),
                "gas_oil": self.thickness_go_spin.value(),
                "gas_water": self.thickness_gw_spin.value()
            }
        }
        
        water_props = self.water_properties.get_properties()
        oil_props = self.oil_properties.get_properties()
        gas_props = self.gas_properties.get_properties()
        
        return {
            "interface": interface_props,
            "water": water_props,
            "oil": oil_props,
            "gas": gas_props
        }
    
    def set_properties(self, properties: Dict[str, Any]):
        """
        Set three-phase flow properties.
        
        Args:
            properties (Dict[str, Any]): Properties to set
        """
        # Set interface properties
        if "interface" in properties:
            interface = properties["interface"]
            
            if "surface_tension" in interface:
                st = interface["surface_tension"]
                if "oil_water" in st:
                    self.surface_tension_ow_spin.setValue(st["oil_water"])
                if "gas_oil" in st:
                    self.surface_tension_go_spin.setValue(st["gas_oil"])
                if "gas_water" in st:
                    self.surface_tension_gw_spin.setValue(st["gas_water"])
            
            if "interface_thickness" in interface:
                it = interface["interface_thickness"]
                if "oil_water" in it:
                    self.thickness_ow_spin.setValue(it["oil_water"])
                if "gas_oil" in it:
                    self.thickness_go_spin.setValue(it["gas_oil"])
                if "gas_water" in it:
                    self.thickness_gw_spin.setValue(it["gas_water"])
        
        # Set phase properties
        if "water" in properties:
            self.water_properties.set_properties(properties["water"])
        
        if "oil" in properties:
            self.oil_properties.set_properties(properties["oil"])
        
        if "gas" in properties:
            self.gas_properties.set_properties(properties["gas"])


class SinglePhaseWidget(QWidget):
    """
    Widget for configuring a single-phase flow.
    
    This widget provides controls for setting up a single-phase flow simulation,
    with focus on the fluid properties and flow conditions.
    """
    
    # Signal emitted when properties are changed
    properties_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """
        Initialize the single-phase flow widget.
        
        Args:
            parent: Parent widget
        """
        super(SinglePhaseWidget, self).__init__(parent)
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(10)
        
        # Flow conditions group
        flow_group = QGroupBox("Flow Conditions")
        flow_layout = QFormLayout(flow_group)
        
        # Flow regime
        self.flow_regime_combo = QComboBox()
        self.flow_regime_combo.addItems([
            "Laminar",
            "Transitional",
            "Turbulent"
        ])
        self.flow_regime_combo.setCurrentIndex(2)  # Default to turbulent
        flow_layout.addRow("Flow Regime:", self.flow_regime_combo)
        
        # Inlet velocity
        self.velocity_spin = QDoubleSpinBox()
        self.velocity_spin.setRange(0.0, 100.0)
        self.velocity_spin.setDecimals(2)
        self.velocity_spin.setSingleStep(0.1)
        self.velocity_spin.setValue(1.0)
        self.velocity_spin.setSuffix(" m/s")
        flow_layout.addRow("Inlet Velocity:", self.velocity_spin)
        
        # Inlet pressure
        self.pressure_spin = QDoubleSpinBox()
        self.pressure_spin.setRange(0.0, 1e7)
        self.pressure_spin.setDecimals(0)
        self.pressure_spin.setSingleStep(1000)
        self.pressure_spin.setValue(101325)  # Atmospheric pressure
        self.pressure_spin.setSuffix(" Pa")
        flow_layout.addRow("Inlet Pressure:", self.pressure_spin)
        
        # Outlet pressure
        self.outlet_pressure_spin = QDoubleSpinBox()
        self.outlet_pressure_spin.setRange(0.0, 1e7)
        self.outlet_pressure_spin.setDecimals(0)
        self.outlet_pressure_spin.setSingleStep(1000)
        self.outlet_pressure_spin.setValue(101325)  # Atmospheric pressure
        self.outlet_pressure_spin.setSuffix(" Pa")
        flow_layout.addRow("Outlet Pressure:", self.outlet_pressure_spin)
        
        # Reynolds number display
        self.reynolds_label = QLabel("0")
        flow_layout.addRow("Reynolds Number:", self.reynolds_label)
        
        self.layout.addWidget(flow_group)
        
        # Turbulence model (only shown for turbulent flow)
        self.turbulence_group = QGroupBox("Turbulence Model")
        turbulence_layout = QFormLayout(self.turbulence_group)
        
        # Turbulence model
        self.turbulence_combo = QComboBox()
        self.turbulence_combo.addItems([
            "k-epsilon",
            "k-omega",
            "Spalart-Allmaras",
            "LES",
            "DNS"
        ])
        turbulence_layout.addRow("Model:", self.turbulence_combo)
        
        # Near-wall treatment
        self.wall_treatment_combo = QComboBox()
        self.wall_treatment_combo.addItems([
            "Wall Functions",
            "Enhanced Wall Treatment",
            "Low-Re Approach"
        ])
        turbulence_layout.addRow("Wall Treatment:", self.wall_treatment_combo)
        
        # Turbulence intensity
        self.turbulence_intensity_spin = QDoubleSpinBox()
        self.turbulence_intensity_spin.setRange(0.0, 1.0)
        self.turbulence_intensity_spin.setDecimals(2)
        self.turbulence_intensity_spin.setSingleStep(0.01)
        self.turbulence_intensity_spin.setValue(0.05)  # 5% default
        self.turbulence_intensity_spin.setSuffix(" (5%)")
        
        def format_percentage(value):
            return f" ({value*100:.0f}%)"
        
        self.turbulence_intensity_spin.valueChanged.connect(
            lambda v: self.turbulence_intensity_spin.setSuffix(format_percentage(v))
        )
        
        turbulence_layout.addRow("Turbulence Intensity:", self.turbulence_intensity_spin)
        
        self.layout.addWidget(self.turbulence_group)
        
        # Phase properties
        self.phase_properties = PhaseProperties("Fluid Properties")
        self.layout.addWidget(self.phase_properties)
        
        # Add stretch to push widgets to the top
        self.layout.addStretch()
        
        # Update initial Reynolds number
        self._update_reynolds_number()
    
    def _connect_signals(self):
        """Connect signals to slots."""
        # Connect flow regime combo
        self.flow_regime_combo.currentTextChanged.connect(self._on_flow_regime_changed)
        
        # Connect flow condition inputs
        self.velocity_spin.valueChanged.connect(self._on_property_changed)
        self.pressure_spin.valueChanged.connect(self._on_property_changed)
        self.outlet_pressure_spin.valueChanged.connect(self._on_property_changed)
        
        # Connect turbulence model inputs
        self.turbulence_combo.currentTextChanged.connect(self._on_property_changed)
        self.wall_treatment_combo.currentTextChanged.connect(self._on_property_changed)
        self.turbulence_intensity_spin.valueChanged.connect(self._on_property_changed)
        
        # Connect phase properties
        self.phase_properties.properties_changed.connect(self._on_phase_changed)
    
    def _on_flow_regime_changed(self, regime: str):
        """
        Handle flow regime change.
        
        Args:
            regime (str): The selected flow regime
        """
        # Show/hide turbulence settings based on regime
        is_turbulent = (regime == "Turbulent" or regime == "Transitional")
        self.turbulence_group.setVisible(is_turbulent)
        
        # Emit signal
        self._on_property_changed()
    
    def _on_phase_changed(self, phase_name: str, phase_properties: Dict[str, Any]):
        """
        Handle phase property change.
        
        Args:
            phase_name (str): Name of the phase
            phase_properties (Dict[str, Any]): Phase properties
        """
        # Update Reynolds number
        self._update_reynolds_number()
        
        # Emit signal with updated properties
        self._on_property_changed()
    
    def _update_reynolds_number(self):
        """Update the Reynolds number display."""
        # Get fluid properties
        density = self.phase_properties.density_spin.value()
        viscosity = self.phase_properties.viscosity_spin.value()
        velocity = self.velocity_spin.value()
        
        # Calculate Reynolds number
        # Re = (density * velocity * characteristic_length) / viscosity
        # Use 1m as a characteristic length for simplicity
        characteristic_length = 0.1  # 10cm pipe diameter
        if viscosity > 0:
            reynolds = (density * velocity * characteristic_length) / viscosity
            self.reynolds_label.setText(f"{reynolds:.2e}")
            
            # Update flow regime automatically
            if reynolds < 2300:
                self.flow_regime_combo.setCurrentText("Laminar")
            elif reynolds < 4000:
                self.flow_regime_combo.setCurrentText("Transitional")
            else:
                self.flow_regime_combo.setCurrentText("Turbulent")
    
    def _on_property_changed(self):
        """Handle property value change."""
        # Update Reynolds number
        self._update_reynolds_number()
        
        # Get flow properties
        flow_props = {
            "regime": self.flow_regime_combo.currentText(),
            "velocity": self.velocity_spin.value(),
            "inlet_pressure": self.pressure_spin.value(),
            "outlet_pressure": self.outlet_pressure_spin.value(),
            "reynolds": float(self.reynolds_label.text())
        }
        
        # Get turbulence properties if applicable
        turbulence_props = {}
        if self.flow_regime_combo.currentText() in ["Turbulent", "Transitional"]:
            turbulence_props = {
                "model": self.turbulence_combo.currentText(),
                "wall_treatment": self.wall_treatment_combo.currentText(),
                "intensity": self.turbulence_intensity_spin.value()
            }
        
        # Get phase properties
        phase_props = self.phase_properties.get_properties()
        
        # Combine into a single properties dictionary
        properties = {
            "flow": flow_props,
            "turbulence": turbulence_props,
            "fluid": phase_props
        }
        
        # Emit signal
        self.properties_changed.emit(properties)
    
    def get_properties(self) -> Dict[str, Any]:
        """
        Get current single-phase flow properties.
        
        Returns:
            Dict[str, Any]: Current properties
        """
        flow_props = {
            "regime": self.flow_regime_combo.currentText(),
            "velocity": self.velocity_spin.value(),
            "inlet_pressure": self.pressure_spin.value(),
            "outlet_pressure": self.outlet_pressure_spin.value(),
            "reynolds": float(self.reynolds_label.text())
        }
        
        turbulence_props = {}
        if self.flow_regime_combo.currentText() in ["Turbulent", "Transitional"]:
            turbulence_props = {
                "model": self.turbulence_combo.currentText(),
                "wall_treatment": self.wall_treatment_combo.currentText(),
                "intensity": self.turbulence_intensity_spin.value()
            }
        
        phase_props = self.phase_properties.get_properties()
        
        return {
            "flow": flow_props,
            "turbulence": turbulence_props,
            "fluid": phase_props
        }
    
    def set_properties(self, properties: Dict[str, Any]):
        """
        Set single-phase flow properties.
        
        Args:
            properties (Dict[str, Any]): Properties to set
        """
        # Set flow properties
        if "flow" in properties:
            flow = properties["flow"]
            if "regime" in flow:
                self.flow_regime_combo.setCurrentText(flow["regime"])
            if "velocity" in flow:
                self.velocity_spin.setValue(flow["velocity"])
            if "inlet_pressure" in flow:
                self.pressure_spin.setValue(flow["inlet_pressure"])
            if "outlet_pressure" in flow:
                self.outlet_pressure_spin.setValue(flow["outlet_pressure"])
        
        # Set turbulence properties
        if "turbulence" in properties:
            turbulence = properties["turbulence"]
            if "model" in turbulence:
                self.turbulence_combo.setCurrentText(turbulence["model"])
            if "wall_treatment" in turbulence:
                self.wall_treatment_combo.setCurrentText(turbulence["wall_treatment"])
            if "intensity" in turbulence:
                self.turbulence_intensity_spin.setValue(turbulence["intensity"])
        
        # Set fluid properties
        if "fluid" in properties:
            self.phase_properties.set_properties(properties["fluid"])
        
        # Update Reynolds number
        self._update_reynolds_number()


class PiggingWidget(QWidget):
    """
    Widget for configuring a pipeline pigging simulation.
    
    This widget provides controls for setting up a pigging simulation,
    a common operation in oil & gas pipeline maintenance.
    """
    
    # Signal emitted when properties are changed
    properties_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """
        Initialize the pigging widget.
        
        Args:
            parent: Parent widget
        """
        super(PiggingWidget, self).__init__(parent)
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(10)
        
        # Pig configuration group
        pig_group = QGroupBox("Pig Configuration")
        pig_layout = QFormLayout(pig_group)
        
        # Pig type
        self.pig_type_combo = QComboBox()
        self.pig_type_combo.addItems([
            "Foam",
            "Disc",
            "Cup",
            "Sphere",
            "Intelligent",
            "Gel"
        ])
        pig_layout.addRow("Pig Type:", self.pig_type_combo)
        
        # Pig diameter
        self.pig_diameter_spin = QDoubleSpinBox()
        self.pig_diameter_spin.setRange(0.01, 2.0)
        self.pig_diameter_spin.setDecimals(3)
        self.pig_diameter_spin.setSingleStep(0.001)
        self.pig_diameter_spin.setValue(0.1)  # 100mm default
        self.pig_diameter_spin.setSuffix(" m")
        pig_layout.addRow("Pig Diameter:", self.pig_diameter_spin)
        
        # Pig length
        self.pig_length_spin = QDoubleSpinBox()
        self.pig_length_spin.setRange(0.01, 5.0)
        self.pig_length_spin.setDecimals(3)
        self.pig_length_spin.setSingleStep(0.01)
        self.pig_length_spin.setValue(0.2)  # 200mm default
        self.pig_length_spin.setSuffix(" m")
        pig_layout.addRow("Pig Length:", self.pig_length_spin)
        
        # Pig density
        self.pig_density_spin = QDoubleSpinBox()
        self.pig_density_spin.setRange(10.0, 2000.0)
        self.pig_density_spin.setDecimals(1)
        self.pig_density_spin.setSingleStep(10.0)
        self.pig_density_spin.setValue(300.0)  # 300 kg/m³ default
        self.pig_density_spin.setSuffix(" kg/m³")
        pig_layout.addRow("Pig Density:", self.pig_density_spin)
        
        # Friction coefficient
        self.friction_spin = QDoubleSpinBox()
        self.friction_spin.setRange(0.01, 1.0)
        self.friction_spin.setDecimals(2)
        self.friction_spin.setSingleStep(0.01)
        self.friction_spin.setValue(0.3)  # Default
        pig_layout.addRow("Friction Coefficient:", self.friction_spin)
        
        # Bypass flow (percentage)
        self.bypass_spin = QDoubleSpinBox()
        self.bypass_spin.setRange(0.0, 100.0)
        self.bypass_spin.setDecimals(1)
        self.bypass_spin.setSingleStep(1.0)
        self.bypass_spin.setValue(5.0)  # 5% default
        self.bypass_spin.setSuffix(" %")
        pig_layout.addRow("Bypass Flow:", self.bypass_spin)
        
        self.layout.addWidget(pig_group)
        
        # Pipeline configuration group
        pipeline_group = QGroupBox("Pipeline Configuration")
        pipeline_layout = QFormLayout(pipeline_group)
        
        # Pipeline diameter
        self.pipeline_diameter_spin = QDoubleSpinBox()
        self.pipeline_diameter_spin.setRange(0.05, 3.0)
        self.pipeline_diameter_spin.setDecimals(3)
        self.pipeline_diameter_spin.setSingleStep(0.01)
        self.pipeline_diameter_spin.setValue(0.1016)  # 4 inches (101.6mm)
        self.pipeline_diameter_spin.setSuffix(" m")
        pipeline_layout.addRow("Pipeline Diameter:", self.pipeline_diameter_spin)
        
        # Pipeline length
        self.pipeline_length_spin = QDoubleSpinBox()
        self.pipeline_length_spin.setRange(1.0, 50000.0)
        self.pipeline_length_spin.setDecimals(1)
        self.pipeline_length_spin.setSingleStep(100.0)
        self.pipeline_length_spin.setValue(1000.0)  # 1km default
        self.pipeline_length_spin.setSuffix(" m")
        pipeline_layout.addRow("Pipeline Length:", self.pipeline_length_spin)
        
        # Pipeline roughness
        self.roughness_spin = QDoubleSpinBox()
        self.roughness_spin.setRange(0.0, 0.01)
        self.roughness_spin.setDecimals(5)
        self.roughness_spin.setSingleStep(0.00001)
        self.roughness_spin.setValue(0.00005)  # 0.05mm default
        self.roughness_spin.setSuffix(" m")
        pipeline_layout.addRow("Roughness:", self.roughness_spin)
        
        self.layout.addWidget(pipeline_group)
        
        # Operation settings group
        operation_group = QGroupBox("Operation Settings")
        operation_layout = QFormLayout(operation_group)
        
        # Driving fluid type
        self.driving_fluid_combo = QComboBox()
        self.driving_fluid_combo.addItems([
            "Water",
            "Oil",
            "Natural Gas",
            "Air"
        ])
        operation_layout.addRow("Driving Fluid:", self.driving_fluid_combo)
        
        # Flow rate
        self.flow_rate_spin = QDoubleSpinBox()
        self.flow_rate_spin.setRange(0.001, 10.0)
        self.flow_rate_spin.setDecimals(3)
        self.flow_rate_spin.setSingleStep(0.1)
        self.flow_rate_spin.setValue(0.5)  # Default
        self.flow_rate_spin.setSuffix(" m³/s")
        operation_layout.addRow("Flow Rate:", self.flow_rate_spin)
        
        # Initial position
        self.initial_position_spin = QDoubleSpinBox()
        self.initial_position_spin.setRange(0.0, 1.0)
        self.initial_position_spin.setDecimals(2)
        self.initial_position_spin.setSingleStep(0.1)
        self.initial_position_spin.setValue(0.0)  # Start of pipeline
        self.initial_position_spin.setSuffix(" (0 = start, 1 = end)")
        operation_layout.addRow("Initial Position:", self.initial_position_spin)
        
        # Receiving trap
        self.receiving_trap_check = QCheckBox()
        self.receiving_trap_check.setChecked(True)
        operation_layout.addRow("Receiving Trap:", self.receiving_trap_check)
        
        self.layout.addWidget(operation_group)
        
        # Add stretch to push widgets to the top
        self.layout.addStretch()
    
    def _connect_signals(self):
        """Connect signals to slots."""
        # Connect all inputs to property changed
        self.pig_type_combo.currentTextChanged.connect(self._on_property_changed)
        self.pig_diameter_spin.valueChanged.connect(self._on_property_changed)
        self.pig_length_spin.valueChanged.connect(self._on_property_changed)
        self.pig_density_spin.valueChanged.connect(self._on_property_changed)
        self.friction_spin.valueChanged.connect(self._on_property_changed)
        self.bypass_spin.valueChanged.connect(self._on_property_changed)
        
        self.pipeline_diameter_spin.valueChanged.connect(self._on_property_changed)
        self.pipeline_length_spin.valueChanged.connect(self._on_property_changed)
        self.roughness_spin.valueChanged.connect(self._on_property_changed)
        
        self.driving_fluid_combo.currentTextChanged.connect(self._on_property_changed)
        self.flow_rate_spin.valueChanged.connect(self._on_property_changed)
        self.initial_position_spin.valueChanged.connect(self._on_property_changed)
        self.receiving_trap_check.toggled.connect(self._on_property_changed)
        
        # Connect pig type to default diameter ratio
        self.pig_type_combo.currentTextChanged.connect(self._update_pig_parameters)
        
        # Connect pipeline diameter to update pig diameter
        self.pipeline_diameter_spin.valueChanged.connect(self._update_pig_diameter)
    
    def _update_pig_parameters(self, pig_type: str):
        """
        Update pig parameters based on type.
        
        Args:
            pig_type (str): The selected pig type
        """
        # Set default values based on pig type
        pipe_diameter = self.pipeline_diameter_spin.value()
        
        if pig_type == "Foam":
            self.pig_diameter_spin.setValue(pipe_diameter * 1.05)  # 5% oversized
            self.pig_length_spin.setValue(pipe_diameter * 2.0)
            self.pig_density_spin.setValue(250.0)
            self.friction_spin.setValue(0.3)
            self.bypass_spin.setValue(8.0)
        
        elif pig_type == "Disc":
            self.pig_diameter_spin.setValue(pipe_diameter * 1.02)  # 2% oversized
            self.pig_length_spin.setValue(pipe_diameter * 1.5)
            self.pig_density_spin.setValue(950.0)
            self.friction_spin.setValue(0.25)
            self.bypass_spin.setValue(3.0)
        
        elif pig_type == "Cup":
            self.pig_diameter_spin.setValue(pipe_diameter * 1.03)  # 3% oversized
            self.pig_length_spin.setValue(pipe_diameter * 2.5)
            self.pig_density_spin.setValue(900.0)
            self.friction_spin.setValue(0.35)
            self.bypass_spin.setValue(2.0)
        
        elif pig_type == "Sphere":
            self.pig_diameter_spin.setValue(pipe_diameter * 1.04)  # 4% oversized
            self.pig_length_spin.setValue(pipe_diameter)  # Same as diameter
            self.pig_density_spin.setValue(800.0)
            self.friction_spin.setValue(0.2)
            self.bypass_spin.setValue(5.0)
        
        elif pig_type == "Intelligent":
            self.pig_diameter_spin.setValue(pipe_diameter * 0.98)  # 2% undersized
            self.pig_length_spin.setValue(pipe_diameter * 4.0)
            self.pig_density_spin.setValue(1200.0)
            self.friction_spin.setValue(0.15)
            self.bypass_spin.setValue(1.0)
        
        elif pig_type == "Gel":
            self.pig_diameter_spin.setValue(pipe_diameter)  # Same as pipeline
            self.pig_length_spin.setValue(pipe_diameter * 3.0)
            self.pig_density_spin.setValue(1050.0)
            self.friction_spin.setValue(0.4)
            self.bypass_spin.setValue(0.5)
    
    def _update_pig_diameter(self, pipe_diameter: float):
        """
        Update pig diameter based on pipeline diameter.
        
        Args:
            pipe_diameter (float): The pipeline diameter
        """
        # Get current pig type
        pig_type = self.pig_type_combo.currentText()
        
        # Set diameter based on pig type
        if pig_type == "Foam":
            self.pig_diameter_spin.setValue(pipe_diameter * 1.05)
        elif pig_type == "Disc":
            self.pig_diameter_spin.setValue(pipe_diameter * 1.02)
        elif pig_type == "Cup":
            self.pig_diameter_spin.setValue(pipe_diameter * 1.03)
        elif pig_type == "Sphere":
            self.pig_diameter_spin.setValue(pipe_diameter * 1.04)
        elif pig_type == "Intelligent":
            self.pig_diameter_spin.setValue(pipe_diameter * 0.98)
        elif pig_type == "Gel":
            self.pig_diameter_spin.setValue(pipe_diameter)
    
    def _on_property_changed(self):
        """Handle property value change."""
        # Get pig properties
        pig_props = {
            "type": self.pig_type_combo.currentText(),
            "diameter": self.pig_diameter_spin.value(),
            "length": self.pig_length_spin.value(),
            "density": self.pig_density_spin.value(),
            "friction": self.friction_spin.value(),
            "bypass": self.bypass_spin.value() / 100.0  # Convert to fraction
        }
        
        # Get pipeline properties
        pipeline_props = {
            "diameter": self.pipeline_diameter_spin.value(),
            "length": self.pipeline_length_spin.value(),
            "roughness": self.roughness_spin.value()
        }
        
        # Get operation properties
        operation_props = {
            "driving_fluid": self.driving_fluid_combo.currentText(),
            "flow_rate": self.flow_rate_spin.value(),
            "initial_position": self.initial_position_spin.value(),
            "receiving_trap": self.receiving_trap_check.isChecked()
        }
        
        # Combine into a single properties dictionary
        properties = {
            "pig": pig_props,
            "pipeline": pipeline_props,
            "operation": operation_props
        }
        
        # Emit signal
        self.properties_changed.emit(properties)
    
    def get_properties(self) -> Dict[str, Any]:
        """
        Get current pigging simulation properties.
        
        Returns:
            Dict[str, Any]: Current properties
        """
        pig_props = {
            "type": self.pig_type_combo.currentText(),
            "diameter": self.pig_diameter_spin.value(),
            "length": self.pig_length_spin.value(),
            "density": self.pig_density_spin.value(),
            "friction": self.friction_spin.value(),
            "bypass": self.bypass_spin.value() / 100.0
        }
        
        pipeline_props = {
            "diameter": self.pipeline_diameter_spin.value(),
            "length": self.pipeline_length_spin.value(),
            "roughness": self.roughness_spin.value()
        }
        
        operation_props = {
            "driving_fluid": self.driving_fluid_combo.currentText(),
            "flow_rate": self.flow_rate_spin.value(),
            "initial_position": self.initial_position_spin.value(),
            "receiving_trap": self.receiving_trap_check.isChecked()
        }
        
        return {
            "pig": pig_props,
            "pipeline": pipeline_props,
            "operation": operation_props
        }
    
    def set_properties(self, properties: Dict[str, Any]):
        """
        Set pigging simulation properties.
        
        Args:
            properties (Dict[str, Any]): Properties to set
        """
        # Set pig properties
        if "pig" in properties:
            pig = properties["pig"]
            if "type" in pig:
                self.pig_type_combo.setCurrentText(pig["type"])
            if "diameter" in pig:
                self.pig_diameter_spin.setValue(pig["diameter"])
            if "length" in pig:
                self.pig_length_spin.setValue(pig["length"])
            if "density" in pig:
                self.pig_density_spin.setValue(pig["density"])
            if "friction" in pig:
                self.friction_spin.setValue(pig["friction"])
            if "bypass" in pig:
                self.bypass_spin.setValue(pig["bypass"] * 100.0)
        
        # Set pipeline properties
        if "pipeline" in properties:
            pipeline = properties["pipeline"]
            if "diameter" in pipeline:
                self.pipeline_diameter_spin.setValue(pipeline["diameter"])
            if "length" in pipeline:
                self.pipeline_length_spin.setValue(pipeline["length"])
            if "roughness" in pipeline:
                self.roughness_spin.setValue(pipeline["roughness"])
        
        # Set operation properties
        if "operation" in properties:
            operation = properties["operation"]
            if "driving_fluid" in operation:
                self.driving_fluid_combo.setCurrentText(operation["driving_fluid"])
            if "flow_rate" in operation:
                self.flow_rate_spin.setValue(operation["flow_rate"])
            if "initial_position" in operation:
                self.initial_position_spin.setValue(operation["initial_position"])
            if "receiving_trap" in operation:
                self.receiving_trap_check.setChecked(operation["receiving_trap"])


class SpillWidget(QWidget):
    """
    Widget for configuring a spill simulation.
    
    This widget provides controls for setting up a spill simulation,
    which is critical for environmental risk assessment in the oil & gas industry.
    """
    
    # Signal emitted when properties are changed
    properties_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """
        Initialize the spill widget.
        
        Args:
            parent: Parent widget
        """
        super(SpillWidget, self).__init__(parent)
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(10)
        
        # Spill configuration group
        spill_group = QGroupBox("Spill Configuration")
        spill_layout = QFormLayout(spill_group)
        
        # Spill type
        self.spill_type_combo = QComboBox()
        self.spill_type_combo.addItems([
            "Surface",
            "Subsurface",
            "Jet"
        ])
        spill_layout.addRow("Spill Type:", self.spill_type_combo)
        
        # Fluid type
        self.fluid_type_combo = QComboBox()
        self.fluid_type_combo.addItems([
            "Crude Oil",
            "Diesel",
            "Gasoline",
            "Natural Gas"
        ])
        spill_layout.addRow("Fluid Type:", self.fluid_type_combo)
        
        # Spill rate
        self.spill_rate_spin = QDoubleSpinBox()
        self.spill_rate_spin.setRange(0.01, 1000.0)
        self.spill_rate_spin.setDecimals(2)
        self.spill_rate_spin.setSingleStep(1.0)
        self.spill_rate_spin.setValue(10.0)  # Default
        self.spill_rate_spin.setSuffix(" kg/s")
        spill_layout.addRow("Spill Rate:", self.spill_rate_spin)
        
        # Total mass
        self.total_mass_spin = QDoubleSpinBox()
        self.total_mass_spin.setRange(1.0, 1000000.0)
        self.total_mass_spin.setDecimals(0)
        self.total_mass_spin.setSingleStep(100.0)
        self.total_mass_spin.setValue(1000.0)  # Default
        self.total_mass_spin.setSuffix(" kg")
        spill_layout.addRow("Total Mass:", self.total_mass_spin)
        
        # Duration (calculated automatically)
        self.duration_label = QLabel("100.0 s")
        spill_layout.addRow("Duration:", self.duration_label)
        
        # Source opening
        self.opening_diameter_spin = QDoubleSpinBox()
        self.opening_diameter_spin.setRange(0.001, 1.0)
        self.opening_diameter_spin.setDecimals(3)
        self.opening_diameter_spin.setSingleStep(0.01)
        self.opening_diameter_spin.setValue(0.05)  # Default
        self.opening_diameter_spin.setSuffix(" m")
        spill_layout.addRow("Opening Diameter:", self.opening_diameter_spin)
        
        # Temperature
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(253.15, 373.15)  # -20°C to 100°C
        self.temperature_spin.setDecimals(1)
        self.temperature_spin.setSingleStep(5.0)
        self.temperature_spin.setValue(293.15)  # 20°C
        self.temperature_spin.setSuffix(" K")
        spill_layout.addRow("Temperature:", self.temperature_spin)
        
        # Pressure
        self.pressure_spin = QDoubleSpinBox()
        self.pressure_spin.setRange(100000.0, 20000000.0)
        self.pressure_spin.setDecimals(0)
        self.pressure_spin.setSingleStep(100000.0)
        self.pressure_spin.setValue(500000.0)  # 5 bar
        self.pressure_spin.setSuffix(" Pa")
        spill_layout.addRow("Pressure:", self.pressure_spin)
        
        self.layout.addWidget(spill_group)
        
        # Environment group
        env_group = QGroupBox("Environment")
        env_layout = QFormLayout(env_group)
        
        # Environment type
        self.env_type_combo = QComboBox()
        self.env_type_combo.addItems([
            "Water (Ocean)",
            "Water (River)",
            "Water (Lake)",
            "Land (Soil)",
            "Land (Concrete)"
        ])
        env_layout.addRow("Environment:", self.env_type_combo)
        
        # Water temperature
        self.water_temp_spin = QDoubleSpinBox()
        self.water_temp_spin.setRange(273.15, 313.15)  # 0°C to 40°C
        self.water_temp_spin.setDecimals(1)
        self.water_temp_spin.setSingleStep(1.0)
        self.water_temp_spin.setValue(288.15)  # 15°C
        self.water_temp_spin.setSuffix(" K")
        env_layout.addRow("Water Temperature:", self.water_temp_spin)
        
        # Wind speed
        self.wind_speed_spin = QDoubleSpinBox()
        self.wind_speed_spin.setRange(0.0, 30.0)
        self.wind_speed_spin.setDecimals(1)
        self.wind_speed_spin.setSingleStep(1.0)
        self.wind_speed_spin.setValue(5.0)  # Default
        self.wind_speed_spin.setSuffix(" m/s")
        env_layout.addRow("Wind Speed:", self.wind_speed_spin)
        
        # Current speed (for water environments)
        self.current_speed_spin = QDoubleSpinBox()
        self.current_speed_spin.setRange(0.0, 5.0)
        self.current_speed_spin.setDecimals(2)
        self.current_speed_spin.setSingleStep(0.1)
        self.current_speed_spin.setValue(0.5)  # Default
        self.current_speed_spin.setSuffix(" m/s")
        env_layout.addRow("Current Speed:", self.current_speed_spin)
        
        # Ambient temperature
        self.ambient_temp_spin = QDoubleSpinBox()
        self.ambient_temp_spin.setRange(253.15, 323.15)  # -20°C to 50°C
        self.ambient_temp_spin.setDecimals(1)
        self.ambient_temp_spin.setSingleStep(1.0)
        self.ambient_temp_spin.setValue(293.15)  # 20°C
        self.ambient_temp_spin.setSuffix(" K")
        env_layout.addRow("Ambient Temperature:", self.ambient_temp_spin)
        
        self.layout.addWidget(env_group)
        
        # Simulation settings group
        sim_group = QGroupBox("Simulation Settings")
        sim_layout = QFormLayout(sim_group)
        
        # Simulation duration
        self.sim_duration_spin = QDoubleSpinBox()
        self.sim_duration_spin.setRange(1.0, 86400.0)  # 1s to 24h
        self.sim_duration_spin.setDecimals(0)
        self.sim_duration_spin.setSingleStep(60.0)
        self.sim_duration_spin.setValue(3600.0)  # 1h default
        self.sim_duration_spin.setSuffix(" s")
        sim_layout.addRow("Simulation Duration:", self.sim_duration_spin)
        
        # Domain size
        self.domain_size_spin = QDoubleSpinBox()
        self.domain_size_spin.setRange(10.0, 10000.0)
        self.domain_size_spin.setDecimals(0)
        self.domain_size_spin.setSingleStep(100.0)
        self.domain_size_spin.setValue(1000.0)  # Default
        self.domain_size_spin.setSuffix(" m")
        sim_layout.addRow("Domain Size:", self.domain_size_spin)
        
        # Weathering model
        self.weathering_check = QCheckBox()
        self.weathering_check.setChecked(True)
        sim_layout.addRow("Weathering Model:", self.weathering_check)
        
        # Dispersion model
        self.dispersion_check = QCheckBox()
        self.dispersion_check.setChecked(True)
        sim_layout.addRow("Dispersion Model:", self.dispersion_check)
        
        # Evaporation model
        self.evaporation_check = QCheckBox()
        self.evaporation_check.setChecked(True)
        sim_layout.addRow("Evaporation Model:", self.evaporation_check)
        
        self.layout.addWidget(sim_group)
        
        # Connect spill rate and total mass to update duration
        self.spill_rate_spin.valueChanged.connect(self._update_duration)
        self.total_mass_spin.valueChanged.connect(self._update_duration)
        
        # Update duration initially
        self._update_duration()
        
        # Add stretch to push widgets to the top
        self.layout.addStretch()
    
    def _connect_signals(self):
        """Connect signals to slots."""
        # Connect all inputs to property changed
        self.spill_type_combo.currentTextChanged.connect(self._on_property_changed)
        self.fluid_type_combo.currentTextChanged.connect(self._on_property_changed)
        self.spill_rate_spin.valueChanged.connect(self._on_property_changed)
        self.total_mass_spin.valueChanged.connect(self._on_property_changed)
        self.opening_diameter_spin.valueChanged.connect(self._on_property_changed)
        self.temperature_spin.valueChanged.connect(self._on_property_changed)
        self.pressure_spin.valueChanged.connect(self._on_property_changed)
        
        self.env_type_combo.currentTextChanged.connect(self._on_env_type_changed)
        self.water_temp_spin.valueChanged.connect(self._on_property_changed)
        self.wind_speed_spin.valueChanged.connect(self._on_property_changed)
        self.current_speed_spin.valueChanged.connect(self._on_property_changed)
        self.ambient_temp_spin.valueChanged.connect(self._on_property_changed)
        
        self.sim_duration_spin.valueChanged.connect(self._on_property_changed)
        self.domain_size_spin.valueChanged.connect(self._on_property_changed)
        self.weathering_check.toggled.connect(self._on_property_changed)
        self.dispersion_check.toggled.connect(self._on_property_changed)
        self.evaporation_check.toggled.connect(self._on_property_changed)
    
    def _update_duration(self):
        """Update spill duration based on rate and total mass."""
        rate = self.spill_rate_spin.value()
        mass = self.total_mass_spin.value()
        
        if rate > 0:
            duration = mass / rate
            self.duration_label.setText(f"{duration:.1f} s")
        else:
            self.duration_label.setText("∞")
    
    def _on_env_type_changed(self, env_type: str):
        """
        Handle environment type change.
        
        Args:
            env_type (str): The selected environment type
        """
        # Enable/disable water temperature and current speed based on environment
        is_water = env_type.startswith("Water")
        self.water_temp_spin.setEnabled(is_water)
        self.current_speed_spin.setEnabled(is_water)
        
        # Emit signal
        self._on_property_changed()
    
    def _on_property_changed(self):
        """Handle property value change."""
        # Get spill properties
        spill_props = {
            "type": self.spill_type_combo.currentText(),
            "fluid": self.fluid_type_combo.currentText(),
            "rate": self.spill_rate_spin.value(),
            "total_mass": self.total_mass_spin.value(),
            "duration": float(self.duration_label.text().split()[0]),
            "opening_diameter": self.opening_diameter_spin.value(),
            "temperature": self.temperature_spin.value(),
            "pressure": self.pressure_spin.value()
        }
        
        # Get environment properties
        env_props = {
            "type": self.env_type_combo.currentText(),
            "water_temperature": self.water_temp_spin.value(),
            "wind_speed": self.wind_speed_spin.value(),
            "current_speed": self.current_speed_spin.value(),
            "ambient_temperature": self.ambient_temp_spin.value()
        }
        
        # Get simulation properties
        sim_props = {
            "duration": self.sim_duration_spin.value(),
            "domain_size": self.domain_size_spin.value(),
            "weathering": self.weathering_check.isChecked(),
            "dispersion": self.dispersion_check.isChecked(),
            "evaporation": self.evaporation_check.isChecked()
        }
        
        # Combine into a single properties dictionary
        properties = {
            "spill": spill_props,
            "environment": env_props,
            "simulation": sim_props
        }
        
        # Emit signal
        self.properties_changed.emit(properties)
    
    def get_properties(self) -> Dict[str, Any]:
        """
        Get current spill simulation properties.
        
        Returns:
            Dict[str, Any]: Current properties
        """
        spill_props = {
            "type": self.spill_type_combo.currentText(),
            "fluid": self.fluid_type_combo.currentText(),
            "rate": self.spill_rate_spin.value(),
            "total_mass": self.total_mass_spin.value(),
            "duration": float(self.duration_label.text().split()[0]),
            "opening_diameter": self.opening_diameter_spin.value(),
            "temperature": self.temperature_spin.value(),
            "pressure": self.pressure_spin.value()
        }
        
        env_props = {
            "type": self.env_type_combo.currentText(),
            "water_temperature": self.water_temp_spin.value(),
            "wind_speed": self.wind_speed_spin.value(),
            "current_speed": self.current_speed_spin.value(),
            "ambient_temperature": self.ambient_temp_spin.value()
        }
        
        sim_props = {
            "duration": self.sim_duration_spin.value(),
            "domain_size": self.domain_size_spin.value(),
            "weathering": self.weathering_check.isChecked(),
            "dispersion": self.dispersion_check.isChecked(),
            "evaporation": self.evaporation_check.isChecked()
        }
        
        return {
            "spill": spill_props,
            "environment": env_props,
            "simulation": sim_props
        }
    
    def set_properties(self, properties: Dict[str, Any]):
        """
        Set spill simulation properties.
        
        Args:
            properties (Dict[str, Any]): Properties to set
        """
        # Set spill properties
        if "spill" in properties:
            spill = properties["spill"]
            if "type" in spill:
                self.spill_type_combo.setCurrentText(spill["type"])
            if "fluid" in spill:
                self.fluid_type_combo.setCurrentText(spill["fluid"])
            if "rate" in spill:
                self.spill_rate_spin.setValue(spill["rate"])
            if "total_mass" in spill:
                self.total_mass_spin.setValue(spill["total_mass"])
            if "opening_diameter" in spill:
                self.opening_diameter_spin.setValue(spill["opening_diameter"])
            if "temperature" in spill:
                self.temperature_spin.setValue(spill["temperature"])
            if "pressure" in spill:
                self.pressure_spin.setValue(spill["pressure"])
        
        # Set environment properties
        if "environment" in properties:
            env = properties["environment"]
            if "type" in env:
                self.env_type_combo.setCurrentText(env["type"])
            if "water_temperature" in env:
                self.water_temp_spin.setValue(env["water_temperature"])
            if "wind_speed" in env:
                self.wind_speed_spin.setValue(env["wind_speed"])
            if "current_speed" in env:
                self.current_speed_spin.setValue(env["current_speed"])
            if "ambient_temperature" in env:
                self.ambient_temp_spin.setValue(env["ambient_temperature"])
        
        # Set simulation properties
        if "simulation" in properties:
            sim = properties["simulation"]
            if "duration" in sim:
                self.sim_duration_spin.setValue(sim["duration"])
            if "domain_size" in sim:
                self.domain_size_spin.setValue(sim["domain_size"])
            if "weathering" in sim:
                self.weathering_check.setChecked(sim["weathering"])
            if "dispersion" in sim:
                self.dispersion_check.setChecked(sim["dispersion"])
            if "evaporation" in sim:
                self.evaporation_check.setChecked(sim["evaporation"])
        
        # Update duration
        self._update_duration()


class FlowSelector(QWidget):
    """
    Flow selector widget for CFD simulations in the oil & gas industry.
    
    This widget provides a comprehensive interface for selecting and configuring
    different types of flows for CFD simulations, including single phase, multi-phase,
    pigging simulations, and spill quantification.
    """
    
    # Signal emitted when flow configuration changes
    flow_changed = pyqtSignal(str, dict)
    
    def __init__(self, parent=None):
        """
        Initialize the flow selector widget.
        
        Args:
            parent: Parent widget
        """
        super(FlowSelector, self).__init__(parent)
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(10)
        
        # Flow type selection group
        self.flow_type_group = QGroupBox("Flow Type")
        flow_type_layout = QHBoxLayout(self.flow_type_group)
        
        # Radio buttons for flow type
        self.single_phase_radio = QRadioButton("Single Phase")
        self.two_phase_radio = QRadioButton("Two Phase")
        self.three_phase_radio = QRadioButton("Three Phase")
        self.pigging_radio = QRadioButton("Pigging")
        self.spill_radio = QRadioButton("Spill")
        
        # Default selection
        self.single_phase_radio.setChecked(True)
        
        # Add to layout
        flow_type_layout.addWidget(self.single_phase_radio)
        flow_type_layout.addWidget(self.two_phase_radio)
        flow_type_layout.addWidget(self.three_phase_radio)
        flow_type_layout.addWidget(self.pigging_radio)
        flow_type_layout.addWidget(self.spill_radio)
        
        # Create button group
        self.flow_type_button_group = QButtonGroup()
        self.flow_type_button_group.addButton(self.single_phase_radio, 0)
        self.flow_type_button_group.addButton(self.two_phase_radio, 1)
        self.flow_type_button_group.addButton(self.three_phase_radio, 2)
        self.flow_type_button_group.addButton(self.pigging_radio, 3)
        self.flow_type_button_group.addButton(self.spill_radio, 4)
        
        self.layout.addWidget(self.flow_type_group)
        
        # Stacked widget to show different settings depending on flow type
        self.flow_settings_stack = QStackedWidget()
        
        # Create widgets for each flow type
        self.single_phase_widget = SinglePhaseWidget()
        self.two_phase_widget = TwoPhaseWidget()
        self.three_phase_widget = ThreePhaseWidget()
        self.pigging_widget = PiggingWidget()
        self.spill_widget = SpillWidget()
        
        # Add widgets to stack
        self.flow_settings_stack.addWidget(self.single_phase_widget)
        self.flow_settings_stack.addWidget(self.two_phase_widget)
        self.flow_settings_stack.addWidget(self.three_phase_widget)
        self.flow_settings_stack.addWidget(self.pigging_widget)
        self.flow_settings_stack.addWidget(self.spill_widget)
        
        # Add stacked widget to layout
        self.layout.addWidget(self.flow_settings_stack)
        
        # Connect flow type radio buttons to stack index
        self.flow_type_button_group.buttonClicked.connect(
            lambda button: self.flow_settings_stack.setCurrentIndex(self.flow_type_button_group.id(button))
        )
        
        # Add preset selection
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Presets:"))
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "None",
            "Pipeline Flow",
            "Wellbore Flow",
            "Production Separator",
            "Process Vessel",
            "Slug Catcher",
            "Multiphase Riser",
            "Gas Lift Well"
        ])
        preset_layout.addWidget(self.preset_combo)
        
        self.load_preset_button = QPushButton("Load")
        preset_layout.addWidget(self.load_preset_button)
        
        self.save_preset_button = QPushButton("Save")
        preset_layout.addWidget(self.save_preset_button)
        
        self.layout.addLayout(preset_layout)
        
        # Add apply button
        self.apply_button = QPushButton("Apply Flow Configuration")
        self.apply_button.setFixedHeight(30)
        self.layout.addWidget(self.apply_button)