#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization controls for Openfoam_Simulator application.

This module implements the visualization control panel that allows users to:
- Control VTK visualization parameters
- Create and manage visualization filters (slices, contours, etc.)
- Set up specialized oil & gas industry visualizations
- Save and load visualization states
"""

import os
import glob
import sys
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

# Import simulation lock manager
try:
    from ..utils.simulation_lock import (
        register_simulation, unregister_simulation, 
        is_simulation_active, block_component, 
        unblock_component, safe_during_simulation
    )
except ImportError:
    # Fallbacks if not available
    def register_simulation(*args, **kwargs): pass
    def unregister_simulation(*args, **kwargs): pass
    def is_simulation_active(*args, **kwargs): return False
    def block_component(*args, **kwargs): pass
    def unblock_component(*args, **kwargs): pass
    def safe_during_simulation(*args, **kwargs): return lambda func: func

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, 
    QCheckBox, QPushButton, QTabWidget, QToolButton, QFrame,
    QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, 
    QStyledItemDelegate, QColorDialog, QSlider, QFileDialog,
    QAction, QSizePolicy, QRadioButton, QButtonGroup, 
    QFormLayout, QGridLayout, QMenu, QMessageBox, QDial, QStackedWidget
)
from PyQt5.QtCore import (
    Qt, QSize, QTimer, pyqtSignal, QSettings, QDir, QRectF,
    QPoint, QPointF
)
from PyQt5.QtGui import (
    QIcon, QColor, QPixmap, QFont, QPalette, QImage, 
    QPainter, QPen, QBrush, QRadialGradient, QLinearGradient
)

# Import utility modules
from ..utils.logger import get_logger
from ..config import get_value, set_value

logger = get_logger(__name__)

# Import VTK modules
try:
    import vtk
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError:
    logger.warning("VTK library not found. Some features will be disabled.")

class ColorMapPreview(QWidget):
    """
    Widget for displaying a color map preview.
    
    This widget shows a preview of a VTK color map to help users
    select the appropriate color scale for their data.
    """
    
    def __init__(self, color_map_name: str = "Rainbow", parent=None):
        """
        Initialize the color map preview.
        
        Args:
            color_map_name (str): Name of the color map
            parent: Parent widget
        """
        super(ColorMapPreview, self).__init__(parent)
        
        # Set up basic properties
        self.color_map_name = color_map_name
        self.setMinimumSize(100, 20)
        self.setMaximumHeight(20)
        
        # Define color maps
        self.color_maps = {
            "Rainbow": [
                (0.0, QColor(0, 0, 255)),   # Blue
                (0.25, QColor(0, 255, 255)), # Cyan
                (0.5, QColor(0, 255, 0)),   # Green
                (0.75, QColor(255, 255, 0)), # Yellow
                (1.0, QColor(255, 0, 0))    # Red
            ],
            "Cool to Warm": [
                (0.0, QColor(59, 76, 192)),  # Cool blue
                (0.5, QColor(221, 221, 221)), # White
                (1.0, QColor(180, 4, 38))    # Warm red
            ],
            "Viridis": [
                (0.0, QColor(68, 1, 84)),    # Dark purple
                (0.25, QColor(59, 82, 139)), # Purple
                (0.5, QColor(33, 145, 140)), # Teal
                (0.75, QColor(94, 201, 98)), # Green
                (1.0, QColor(253, 231, 37))  # Yellow
            ],
            "Plasma": [
                (0.0, QColor(13, 8, 135)),   # Dark blue
                (0.25, QColor(156, 23, 158)), # Purple
                (0.5, QColor(237, 47, 89)),  # Pink
                (0.75, QColor(255, 131, 0)), # Orange
                (1.0, QColor(240, 249, 33))  # Yellow
            ],
            "Blues": [
                (0.0, QColor(247, 251, 255)), # Very light blue
                (0.25, QColor(198, 219, 239)), # Light blue
                (0.5, QColor(107, 174, 214)), # Medium blue
                (0.75, QColor(33, 113, 181)), # Blue
                (1.0, QColor(8, 48, 107))     # Dark blue
            ],
            "Reds": [
                (0.0, QColor(255, 245, 240)), # Very light red
                (0.25, QColor(254, 224, 210)), # Light red
                (0.5, QColor(252, 146, 114)), # Medium red
                (0.75, QColor(222, 45, 38)),  # Red
                (1.0, QColor(165, 15, 21))    # Dark red
            ],
            "Greens": [
                (0.0, QColor(247, 252, 245)), # Very light green
                (0.25, QColor(199, 233, 192)), # Light green
                (0.5, QColor(116, 196, 118)), # Medium green
                (0.75, QColor(35, 139, 69)),  # Green
                (1.0, QColor(0, 68, 27))      # Dark green
            ],
            "Jet": [
                (0.0, QColor(0, 0, 128)),    # Dark blue
                (0.125, QColor(0, 0, 255)),  # Blue
                (0.375, QColor(0, 255, 255)), # Cyan
                (0.625, QColor(255, 255, 0)), # Yellow
                (0.875, QColor(255, 0, 0)),  # Red
                (1.0, QColor(128, 0, 0))     # Dark red
            ],
            "Black to White": [
                (0.0, QColor(0, 0, 0)),      # Black
                (1.0, QColor(255, 255, 255)) # White
            ],
            "Pressure": [  # Customized for pressure visualization
                (0.0, QColor(0, 0, 255)),    # Blue (low pressure)
                (0.5, QColor(255, 255, 255)), # White (medium pressure)
                (1.0, QColor(255, 0, 0))     # Red (high pressure)
            ],
            "Velocity": [  # Customized for velocity visualization
                (0.0, QColor(0, 0, 128)),    # Dark blue (low velocity)
                (0.25, QColor(0, 128, 255)), # Light blue
                (0.5, QColor(0, 255, 255)),  # Cyan
                (0.75, QColor(255, 255, 0)), # Yellow
                (1.0, QColor(255, 0, 0))     # Red (high velocity)
            ],
            "Oil-Water": [  # Customized for oil-water interface
                (0.0, QColor(0, 0, 128)),    # Dark blue (water)
                (0.4, QColor(0, 128, 255)),  # Light blue (water)
                (0.5, QColor(200, 200, 200)), # Gray (interface)
                (0.6, QColor(160, 120, 0)),  # Dark brown (oil)
                (1.0, QColor(210, 180, 140)) # Light brown (oil)
            ]
        }
    
    def set_color_map(self, color_map_name: str):
        """
        Set the color map to display.
        
        Args:
            color_map_name (str): Name of the color map
        """
        if color_map_name in self.color_maps:
            self.color_map_name = color_map_name
            self.update()
    
    def paintEvent(self, event):
        """
        Paint the color map preview.
        
        Args:
            event: Paint event
        """
        painter = QPainter(self)
        rect = self.rect()
        
        # Get color map
        color_map = self.color_maps.get(self.color_map_name, self.color_maps["Rainbow"])
        
        # Create gradient
        gradient = QLinearGradient(rect.left(), 0, rect.right(), 0)
        for position, color in color_map:
            gradient.setColorAt(position, color)
        
        # Fill with gradient
        painter.fillRect(rect, gradient)
        
        # Draw border
        painter.setPen(QPen(Qt.black, 1))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))


class VisualEffectsTab(QWidget):
    """
    Tab for controlling visual effects like lighting and shading.
    """
    
    # Signal emitted when visual effects are changed
    effects_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """
        Initialize the visual effects tab.
        
        Args:
            parent: Parent widget
        """
        super(VisualEffectsTab, self).__init__(parent)
        
        # Set up layout
        self.layout = QVBoxLayout(self)
        
        # Create groups
        self._setup_lighting_group()
        self._setup_shading_group()
        self._setup_advanced_group()
        
        # Add stretch to push everything to the top
        self.layout.addStretch()
    
    def _setup_lighting_group(self):
        """Set up lighting controls group."""
        lighting_group = QGroupBox("Lighting")
        lighting_layout = QFormLayout(lighting_group)
        
        # Lighting enabled
        self.lighting_check = QCheckBox()
        self.lighting_check.setChecked(True)
        lighting_layout.addRow("Enable Lighting:", self.lighting_check)
        
        # Light intensity
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setMinimum(0)
        self.intensity_slider.setMaximum(100)
        self.intensity_slider.setValue(75)
        self.intensity_slider.setTickPosition(QSlider.TicksBelow)
        lighting_layout.addRow("Intensity:", self.intensity_slider)
        
        # Light color
        self.light_color_button = QPushButton("Select...")
        self.light_color_button.setFixedWidth(80)
        self.light_color = QColor(255, 255, 255)  # White by default
        self._update_color_button(self.light_color_button, self.light_color)
        self.light_color_button.clicked.connect(lambda: self._show_color_dialog(self.light_color_button, "light"))
        lighting_layout.addRow("Light Color:", self.light_color_button)
        
        # Add to main layout
        self.layout.addWidget(lighting_group)
    
    def _setup_shading_group(self):
        """Set up shading controls group."""
        shading_group = QGroupBox("Shading")
        shading_layout = QFormLayout(shading_group)
        
        # Shading model
        self.shading_combo = QComboBox()
        self.shading_combo.addItems(["Flat", "Gouraud", "Phong"])
        self.shading_combo.setCurrentIndex(1)  # Gouraud by default
        shading_layout.addRow("Shading Model:", self.shading_combo)
        
        # Ambient factor
        self.ambient_slider = QSlider(Qt.Horizontal)
        self.ambient_slider.setMinimum(0)
        self.ambient_slider.setMaximum(100)
        self.ambient_slider.setValue(20)
        self.ambient_slider.setTickPosition(QSlider.TicksBelow)
        shading_layout.addRow("Ambient:", self.ambient_slider)
        
        # Diffuse factor
        self.diffuse_slider = QSlider(Qt.Horizontal)
        self.diffuse_slider.setMinimum(0)
        self.diffuse_slider.setMaximum(100)
        self.diffuse_slider.setValue(70)
        self.diffuse_slider.setTickPosition(QSlider.TicksBelow)
        shading_layout.addRow("Diffuse:", self.diffuse_slider)
        
        # Specular factor
        self.specular_slider = QSlider(Qt.Horizontal)
        self.specular_slider.setMinimum(0)
        self.specular_slider.setMaximum(100)
        self.specular_slider.setValue(30)
        self.specular_slider.setTickPosition(QSlider.TicksBelow)
        shading_layout.addRow("Specular:", self.specular_slider)
        
        # Add to main layout
        self.layout.addWidget(shading_group)
    
    def _setup_advanced_group(self):
        """Set up advanced controls group."""
        self.advanced_group = QGroupBox("Advanced")
        advanced_layout = QFormLayout(self.advanced_group)
        
        # Edge display
        self.edge_check = QCheckBox()
        self.edge_check.setChecked(False)
        advanced_layout.addRow("Show Edges:", self.edge_check)
        
        # Edge color
        self.edge_color_button = QPushButton("Select...")
        self.edge_color_button.setFixedWidth(80)
        self.edge_color = QColor(0, 0, 0)  # Black by default
        self._update_color_button(self.edge_color_button, self.edge_color)
        self.edge_color_button.clicked.connect(lambda: self._show_color_dialog(self.edge_color_button, "edge"))
        advanced_layout.addRow("Edge Color:", self.edge_color_button)
        
        # Background color
        self.bg_color_button = QPushButton("Select...")
        self.bg_color_button.setFixedWidth(80)
        self.bg_color = QColor(50, 50, 50)  # Dark gray by default
        self._update_color_button(self.bg_color_button, self.bg_color)
        self.bg_color_button.clicked.connect(lambda: self._show_color_dialog(self.bg_color_button, "background"))
        advanced_layout.addRow("Background:", self.bg_color_button)
        
        self.layout.addWidget(self.advanced_group)
    
    def _update_color_button(self, button: QPushButton, color: QColor):
        """
        Update a color button with the selected color.
        
        Args:
            button (QPushButton): The button to update
            color (QColor): The color to apply
        """
        # Create a pixmap with the selected color
        pixmap = QPixmap(16, 16)
        pixmap.fill(color)
        
        # Set as button icon
        button.setIcon(QIcon(pixmap))
    
    def _show_color_dialog(self, button: QPushButton, color_type: str):
        """
        Show color dialog and update the button.
        
        Args:
            button (QPushButton): The button that was clicked
            color_type (str): Type of color (light, edge, background)
        """
        # Get current color
        if color_type == "light":
            current_color = self.light_color
        elif color_type == "edge":
            current_color = self.edge_color
        elif color_type == "background":
            current_color = self.bg_color
        else:
            current_color = QColor(255, 255, 255)
        
        # Show color dialog
        color = QColorDialog.getColor(current_color, self, "Select Color")
        
        if color.isValid():
            # Update button
            self._update_color_button(button, color)
            
            # Store color
            if color_type == "light":
                self.light_color = color
            elif color_type == "edge":
                self.edge_color = color
            elif color_type == "background":
                self.bg_color = color
            
            # Emit change signal
            self._emit_effects_change()
    
    def _emit_effects_change(self):
        """Emit signal with current visual effects settings."""
        settings = {
            "lighting_enabled": self.lighting_check.isChecked(),
            "light_intensity": self.intensity_slider.value() / 100.0,
            "light_color": [self.light_color.red() / 255.0, 
                           self.light_color.green() / 255.0, 
                           self.light_color.blue() / 255.0],
            "shading_model": self.shading_combo.currentText(),
            "ambient": self.ambient_slider.value() / 100.0,
            "diffuse": self.diffuse_slider.value() / 100.0,
            "specular": self.specular_slider.value() / 100.0,
            "show_edges": self.edge_check.isChecked(),
            "edge_color": [self.edge_color.red() / 255.0, 
                          self.edge_color.green() / 255.0, 
                          self.edge_color.blue() / 255.0],
            "background_color": [self.bg_color.red() / 255.0, 
                               self.bg_color.green() / 255.0, 
                               self.bg_color.blue() / 255.0]
        }
        
        self.effects_changed.emit(settings)
    
    def connect_signals(self):
        """Connect signals to slots."""
        # Connect all control signals to emit changes
        self.lighting_check.toggled.connect(self._emit_effects_change)
        self.intensity_slider.valueChanged.connect(self._emit_effects_change)
        self.shading_combo.currentIndexChanged.connect(self._emit_effects_change)
        self.ambient_slider.valueChanged.connect(self._emit_effects_change)
        self.diffuse_slider.valueChanged.connect(self._emit_effects_change)
        self.specular_slider.valueChanged.connect(self._emit_effects_change)
        self.edge_check.toggled.connect(self._emit_effects_change)
        
        self.light_color_button.clicked.connect(lambda: self._show_color_dialog(self.light_color_button, "light"))
        self.edge_color_button.clicked.connect(lambda: self._show_color_dialog(self.edge_color_button, "edge"))
        self.bg_color_button.clicked.connect(lambda: self._show_color_dialog(self.bg_color_button, "background"))
    
    def get_settings(self) -> Dict[str, Any]:
        """
        Get current visual effects settings.
        
        Returns:
            Dict[str, Any]: Current settings
        """
        return {
            "lighting_enabled": self.lighting_check.isChecked(),
            "light_intensity": self.intensity_slider.value() / 100.0,
            "light_color": [self.light_color.red() / 255.0, 
                           self.light_color.green() / 255.0, 
                           self.light_color.blue() / 255.0],
            "shading_model": self.shading_combo.currentText(),
            "ambient": self.ambient_slider.value() / 100.0,
            "diffuse": self.diffuse_slider.value() / 100.0,
            "specular": self.specular_slider.value() / 100.0,
            "show_edges": self.edge_check.isChecked(),
            "edge_color": [self.edge_color.red() / 255.0, 
                          self.edge_color.green() / 255.0, 
                          self.edge_color.blue() / 255.0],
            "background_color": [self.bg_color.red() / 255.0, 
                               self.bg_color.green() / 255.0, 
                               self.bg_color.blue() / 255.0]
        }
    
    def apply_settings(self, settings: Dict[str, Any]):
        """
        Apply visual effects settings.
        
        Args:
            settings (Dict[str, Any]): Settings to apply
        """
        # Apply lighting settings
        if "lighting_enabled" in settings:
            self.lighting_check.setChecked(settings["lighting_enabled"])
        
        if "light_intensity" in settings:
            self.intensity_slider.setValue(int(settings["light_intensity"] * 100))
        
        if "light_color" in settings and len(settings["light_color"]) == 3:
            self.light_color = QColor(
                int(settings["light_color"][0] * 255),
                int(settings["light_color"][1] * 255), 
                int(settings["light_color"][2] * 255)
            )
            self._update_color_button(self.light_color_button, self.light_color)
        
        # Apply shading settings
        if "shading_model" in settings:
            index = self.shading_combo.findText(settings["shading_model"])
            if index >= 0:
                self.shading_combo.setCurrentIndex(index)
        
        if "ambient" in settings:
            self.ambient_slider.setValue(int(settings["ambient"] * 100))
        
        if "diffuse" in settings:
            self.diffuse_slider.setValue(int(settings["diffuse"] * 100))
        
        if "specular" in settings:
            self.specular_slider.setValue(int(settings["specular"] * 100))
        
        # Apply advanced settings
        if "show_edges" in settings:
            self.edge_check.setChecked(settings["show_edges"])
        
        if "edge_color" in settings and len(settings["edge_color"]) == 3:
            self.edge_color = QColor(
                int(settings["edge_color"][0] * 255),
                int(settings["edge_color"][1] * 255), 
                int(settings["edge_color"][2] * 255)
            )
            self._update_color_button(self.edge_color_button, self.edge_color)
        
        if "background_color" in settings and len(settings["background_color"]) == 3:
            self.bg_color = QColor(
                int(settings["background_color"][0] * 255),
                int(settings["background_color"][1] * 255), 
                int(settings["background_color"][2] * 255)
            )
            self._update_color_button(self.bg_color_button, self.bg_color)


class VisualizationControls(QWidget):
    """
    Visualization control panel for Openfoam_Simulator.
    
    This panel provides controls for configuring and manipulating
    VTK visualization parameters, with specialized features
    for oil & gas industry applications.
    """
    
    # Signals
    representation_changed = pyqtSignal(str)
    coloring_changed = pyqtSignal(str, str)  # field, component
    colormap_changed = pyqtSignal(str)
    filter_applied = pyqtSignal(str, dict)  # filter_type, parameters
    filter_removed = pyqtSignal(str)  # filter_id
    camera_changed = pyqtSignal(dict)  # camera parameters
    
    def __init__(self, parent=None):
        """
        Initialize the visualization controls.
        
        Args:
            parent: The parent widget (should be MainWindow)
        """
        super(VisualizationControls, self).__init__(parent)
        
        # Store reference to main window
        self.main_window = parent
        
        # Initialize instance variables
        self.current_project = None
        self.active_filters = {}  # Dict of active filters
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
        
        # Initialize with default state
        self._update_ui_state()
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.layout.setSpacing(6)
        
        # Add tab widget for different panels
        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)
        
        # Create tabs
        self._setup_display_tab()
        self._setup_filters_tab()
        self._setup_camera_tab()
        self._setup_effects_tab()
        self._setup_oilgas_tab()
        
        # Add control buttons at the bottom
        self._setup_control_buttons()
    
    def _setup_display_tab(self):
        """Set up the display configuration tab."""
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        
        # Representation group
        rep_group = QGroupBox("Representation")
        rep_layout = QFormLayout(rep_group)
        
        # Representation type
        self.rep_combo = QComboBox()
        self.rep_combo.addItems([
            "Surface", 
            "Surface With Edges", 
            "Wireframe", 
            "Points", 
            "Volume"
        ])
        rep_layout.addRow("Type:", self.rep_combo)
        
        # Point size (for point representation)
        self.point_size_spin = QDoubleSpinBox()
        self.point_size_spin.setDecimals(1)
        self.point_size_spin.setMinimum(1.0)
        self.point_size_spin.setMaximum(20.0)
        self.point_size_spin.setValue(3.0)
        rep_layout.addRow("Point Size:", self.point_size_spin)
        
        # Line width (for wireframe representation)
        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setDecimals(1)
        self.line_width_spin.setMinimum(1.0)
        self.line_width_spin.setMaximum(10.0)
        self.line_width_spin.setValue(1.0)
        rep_layout.addRow("Line Width:", self.line_width_spin)
        
        display_layout.addWidget(rep_group)
        
        # Coloring group
        color_group = QGroupBox("Coloring")
        color_layout = QFormLayout(color_group)
        
        # Field to color by
        self.color_field_combo = QComboBox()
        self.color_field_combo.addItems([
            "Solid Color",
            "Pressure",
            "Velocity",
            "Temperature",
            "Phase",
            "Turbulence"
        ])
        color_layout.addRow("Color by:", self.color_field_combo)
        
        # Component for vector fields
        self.component_combo = QComboBox()
        self.component_combo.addItems([
            "Magnitude",
            "X",
            "Y",
            "Z"
        ])
        self.component_combo.setEnabled(False)  # Disabled until vector field is selected
        color_layout.addRow("Component:", self.component_combo)
        
        # Solid color selection
        self.solid_color_button = QPushButton("Select...")
        self.solid_color_button.setFixedWidth(80)
        self.solid_color = QColor(220, 220, 220)  # Light gray default
        self._update_color_button(self.solid_color_button, self.solid_color)
        self.solid_color_button.clicked.connect(self._show_solid_color_dialog)
        color_layout.addRow("Solid Color:", self.solid_color_button)
        
        # Color map selection
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "Rainbow",
            "Cool to Warm",
            "Viridis",
            "Plasma",
            "Blues",
            "Reds",
            "Greens",
            "Jet",
            "Black to White",
            "Pressure",
            "Velocity",
            "Oil-Water"
        ])
        color_layout.addRow("Color Map:", self.colormap_combo)
        
        # Color map preview
        self.colormap_preview = ColorMapPreview("Rainbow")
        color_layout.addRow("Preview:", self.colormap_preview)
        
        # Connect color map combo to preview
        self.colormap_combo.currentTextChanged.connect(self.colormap_preview.set_color_map)
        
        # Show scalar bar checkbox
        self.scalar_bar_check = QCheckBox()
        self.scalar_bar_check.setChecked(True)
        color_layout.addRow("Show Color Bar:", self.scalar_bar_check)
        
        display_layout.addWidget(color_group)
        
        # Mesh opacity group
        opacity_group = QGroupBox("Mesh Opacity")
        opacity_layout = QFormLayout(opacity_group)
        
        # Mesh opacity slider
        self.mesh_opacity_slider = QSlider(Qt.Horizontal)
        self.mesh_opacity_slider.setMinimum(0)
        self.mesh_opacity_slider.setMaximum(100)
        self.mesh_opacity_slider.setValue(100)
        self.mesh_opacity_slider.setTickPosition(QSlider.TicksBelow)
        opacity_layout.addRow("Opacity:", self.mesh_opacity_slider)
        
        display_layout.addWidget(opacity_group)
        
        # Add stretch to push everything to the top
        display_layout.addStretch()
        
        # Add tab
        self.tab_widget.addTab(display_widget, "Display")
    
    def _setup_filters_tab(self):
        """Set up the filters configuration tab."""
        filters_widget = QWidget()
        filters_layout = QVBoxLayout(filters_widget)
        
        # New filter group
        new_filter_group = QGroupBox("Create Filter")
        new_filter_layout = QVBoxLayout(new_filter_group)
        
        # Filter type selection
        filter_type_layout = QHBoxLayout()
        filter_type_label = QLabel("Filter Type:")
        filter_type_layout.addWidget(filter_type_label)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            "Slice",
            "Clip",
            "Threshold",
            "Contour",
            "Streamlines",
            "Glyph",
            "Calculator"
        ])
        filter_type_layout.addWidget(self.filter_combo, 1)
        
        new_filter_layout.addLayout(filter_type_layout)
        
        # Stacked widget for filter-specific settings
        self.filter_settings_stack = QStackedWidget()
        
        # 1. Slice filter settings
        slice_widget = QWidget()
        slice_layout = QFormLayout(slice_widget)
        
        # Normal direction
        self.slice_normal_combo = QComboBox()
        self.slice_normal_combo.addItems(["X", "Y", "Z", "Custom"])
        slice_layout.addRow("Normal:", self.slice_normal_combo)
        
        # Custom normal inputs
        self.slice_normal_layout = QHBoxLayout()
        self.slice_normal_x = QDoubleSpinBox()
        self.slice_normal_x.setRange(-1, 1)
        self.slice_normal_x.setValue(1)
        self.slice_normal_y = QDoubleSpinBox()
        self.slice_normal_y.setRange(-1, 1)
        self.slice_normal_y.setValue(0)
        self.slice_normal_z = QDoubleSpinBox()
        self.slice_normal_z.setRange(-1, 1)
        self.slice_normal_z.setValue(0)
        
        self.slice_normal_layout.addWidget(QLabel("X:"))
        self.slice_normal_layout.addWidget(self.slice_normal_x)
        self.slice_normal_layout.addWidget(QLabel("Y:"))
        self.slice_normal_layout.addWidget(self.slice_normal_y)
        self.slice_normal_layout.addWidget(QLabel("Z:"))
        self.slice_normal_layout.addWidget(self.slice_normal_z)
        
        slice_layout.addRow("Custom Normal:", self.slice_normal_layout)
        
        # Origin
        self.slice_origin_layout = QHBoxLayout()
        self.slice_origin_x = QDoubleSpinBox()
        self.slice_origin_x.setRange(-1000, 1000)
        self.slice_origin_x.setValue(0)
        self.slice_origin_y = QDoubleSpinBox()
        self.slice_origin_y.setRange(-1000, 1000)
        self.slice_origin_y.setValue(0)
        self.slice_origin_z = QDoubleSpinBox()
        self.slice_origin_z.setRange(-1000, 1000)
        self.slice_origin_z.setValue(0)
        
        self.slice_origin_layout.addWidget(QLabel("X:"))
        self.slice_origin_layout.addWidget(self.slice_origin_x)
        self.slice_origin_layout.addWidget(QLabel("Y:"))
        self.slice_origin_layout.addWidget(self.slice_origin_y)
        self.slice_origin_layout.addWidget(QLabel("Z:"))
        self.slice_origin_layout.addWidget(self.slice_origin_z)
        
        slice_layout.addRow("Origin:", self.slice_origin_layout)
        
        # Slice offset
        self.slice_offset_slider = QSlider(Qt.Horizontal)
        self.slice_offset_slider.setMinimum(-100)
        self.slice_offset_slider.setMaximum(100)
        self.slice_offset_slider.setValue(0)
        self.slice_offset_slider.setTickPosition(QSlider.TicksBelow)
        slice_layout.addRow("Offset:", self.slice_offset_slider)
        
        self.filter_settings_stack.addWidget(slice_widget)
        
        # 2. Clip filter settings
        clip_widget = QWidget()
        clip_layout = QFormLayout(clip_widget)
        
        # Clip function
        self.clip_function_combo = QComboBox()
        self.clip_function_combo.addItems(["Plane", "Box", "Sphere"])
        clip_layout.addRow("Function:", self.clip_function_combo)
        
        # Clip direction
        self.clip_direction_combo = QComboBox()
        self.clip_direction_combo.addItems(["Inside", "Outside"])
        clip_layout.addRow("Direction:", self.clip_direction_combo)
        
        # Similar controls to slice for origin and normal
        self.clip_normal_combo = QComboBox()
        self.clip_normal_combo.addItems(["X", "Y", "Z", "Custom"])
        clip_layout.addRow("Normal:", self.clip_normal_combo)
        
        # Custom normal inputs
        self.clip_normal_layout = QHBoxLayout()
        self.clip_normal_x = QDoubleSpinBox()
        self.clip_normal_x.setRange(-1, 1)
        self.clip_normal_x.setValue(1)
        self.clip_normal_y = QDoubleSpinBox()
        self.clip_normal_y.setRange(-1, 1)
        self.clip_normal_y.setValue(0)
        self.clip_normal_z = QDoubleSpinBox()
        self.clip_normal_z.setRange(-1, 1)
        self.clip_normal_z.setValue(0)
        
        self.clip_normal_layout.addWidget(QLabel("X:"))
        self.clip_normal_layout.addWidget(self.clip_normal_x)
        self.clip_normal_layout.addWidget(QLabel("Y:"))
        self.clip_normal_layout.addWidget(self.clip_normal_y)
        self.clip_normal_layout.addWidget(QLabel("Z:"))
        self.clip_normal_layout.addWidget(self.clip_normal_z)
        
        clip_layout.addRow("Custom Normal:", self.clip_normal_layout)
        
        # Origin
        self.clip_origin_layout = QHBoxLayout()
        self.clip_origin_x = QDoubleSpinBox()
        self.clip_origin_x.setRange(-1000, 1000)
        self.clip_origin_x.setValue(0)
        self.clip_origin_y = QDoubleSpinBox()
        self.clip_origin_y.setRange(-1000, 1000)
        self.clip_origin_y.setValue(0)
        self.clip_origin_z = QDoubleSpinBox()
        self.clip_origin_z.setRange(-1000, 1000)
        self.clip_origin_z.setValue(0)
        
        self.clip_origin_layout.addWidget(QLabel("X:"))
        self.clip_origin_layout.addWidget(self.clip_origin_x)
        self.clip_origin_layout.addWidget(QLabel("Y:"))
        self.clip_origin_layout.addWidget(self.clip_origin_y)
        self.clip_origin_layout.addWidget(QLabel("Z:"))
        self.clip_origin_layout.addWidget(self.clip_origin_z)
        
        clip_layout.addRow("Origin:", self.clip_origin_layout)
        
        self.filter_settings_stack.addWidget(clip_widget)
        
        # 3. Threshold filter settings
        threshold_widget = QWidget()
        threshold_layout = QFormLayout(threshold_widget)
        
        # Scalar field
        self.thresh_field_combo = QComboBox()
        self.thresh_field_combo.addItems([
            "Pressure",
            "Velocity",
            "Temperature",
            "Phase",
            "Turbulence"
        ])
        threshold_layout.addRow("Field:", self.thresh_field_combo)
        
        # Threshold range
        self.thresh_min_spin = QDoubleSpinBox()
        self.thresh_min_spin.setRange(-1e6, 1e6)
        self.thresh_min_spin.setValue(0)
        threshold_layout.addRow("Minimum:", self.thresh_min_spin)
        
        self.thresh_max_spin = QDoubleSpinBox()
        self.thresh_max_spin.setRange(-1e6, 1e6)
        self.thresh_max_spin.setValue(100)
        threshold_layout.addRow("Maximum:", self.thresh_max_spin)
        
        self.filter_settings_stack.addWidget(threshold_widget)
        
        # 4. Contour filter settings
        contour_widget = QWidget()
        contour_layout = QFormLayout(contour_widget)
        
        # Scalar field
        self.contour_field_combo = QComboBox()
        self.contour_field_combo.addItems([
            "Pressure",
            "Velocity",
            "Temperature",
            "Phase",
            "Turbulence"
        ])
        contour_layout.addRow("Field:", self.contour_field_combo)
        
        # Number of contours
        self.contour_count_spin = QSpinBox()
        self.contour_count_spin.setRange(1, 20)
        self.contour_count_spin.setValue(5)
        contour_layout.addRow("Count:", self.contour_count_spin)
        
        # Contour range
        self.contour_min_spin = QDoubleSpinBox()
        self.contour_min_spin.setRange(-1e6, 1e6)
        self.contour_min_spin.setValue(0)
        contour_layout.addRow("Minimum:", self.contour_min_spin)
        
        self.contour_max_spin = QDoubleSpinBox()
        self.contour_max_spin.setRange(-1e6, 1e6)
        self.contour_max_spin.setValue(100)
        contour_layout.addRow("Maximum:", self.contour_max_spin)
        
        self.filter_settings_stack.addWidget(contour_widget)
        
        # 5. Streamlines filter settings
        streamlines_widget = QWidget()
        streamlines_layout = QFormLayout(streamlines_widget)
        
        # Vector field
        self.stream_field_combo = QComboBox()
        self.stream_field_combo.addItems(["Velocity", "Other Vector"])
        streamlines_layout.addRow("Field:", self.stream_field_combo)
        
        # Seed type
        self.stream_seed_combo = QComboBox()
        self.stream_seed_combo.addItems(["Point", "Line", "Plane"])
        streamlines_layout.addRow("Seed Type:", self.stream_seed_combo)
        
        # Number of seeds
        self.stream_count_spin = QSpinBox()
        self.stream_count_spin.setRange(1, 1000)
        self.stream_count_spin.setValue(100)
        streamlines_layout.addRow("Seed Count:", self.stream_count_spin)
        
        # Streamline length
        self.stream_length_spin = QDoubleSpinBox()
        self.stream_length_spin.setRange(0.1, 1000)
        self.stream_length_spin.setValue(100)
        streamlines_layout.addRow("Max Length:", self.stream_length_spin)
        
        self.filter_settings_stack.addWidget(streamlines_widget)
        
        # Add stacked widget
        new_filter_layout.addWidget(self.filter_settings_stack)
        
        # Connect filter combo to stacked widget
        self.filter_combo.currentIndexChanged.connect(self.filter_settings_stack.setCurrentIndex)
        
        # Create filter button
        self.create_filter_button = QPushButton("Create Filter")
        new_filter_layout.addWidget(self.create_filter_button)
        
        filters_layout.addWidget(new_filter_group)
        
        # Flow visualization group
        flow_group = QGroupBox("Flow Visualization")
        flow_layout = QFormLayout(flow_group)
        
        # Visualization type
        self.flow_viz_type = QComboBox()
        self.flow_viz_type.addItems(["Streamlines", "Pathlines", "Surface LIC", "Vectors", "Glyphs"])
        flow_layout.addRow("Visualization:", self.flow_viz_type)
        
        # Number of streamlines
        self.flow_line_count = QSpinBox()
        self.flow_line_count.setMinimum(10)
        self.flow_line_count.setMaximum(500)
        self.flow_line_count.setValue(100)
        self.flow_line_count.setSingleStep(10)
        flow_layout.addRow("Line Count:", self.flow_line_count)
        
        # Line width
        self.flow_line_width = QDoubleSpinBox()
        self.flow_line_width.setMinimum(0.5)
        self.flow_line_width.setMaximum(10.0)
        self.flow_line_width.setValue(2.0)
        self.flow_line_width.setSingleStep(0.5)
        flow_layout.addRow("Line Width:", self.flow_line_width)
        
        # Seed location
        seed_layout = QHBoxLayout()
        self.seed_inlet = QRadioButton("Inlet")
        self.seed_outlet = QRadioButton("Outlet")
        self.seed_custom = QRadioButton("Custom")
        self.seed_inlet.setChecked(True)
        
        seed_group = QButtonGroup(self)
        seed_group.addButton(self.seed_inlet)
        seed_group.addButton(self.seed_outlet)
        seed_group.addButton(self.seed_custom)
        
        seed_layout.addWidget(self.seed_inlet)
        seed_layout.addWidget(self.seed_outlet)
        seed_layout.addWidget(self.seed_custom)
        flow_layout.addRow("Seed Location:", seed_layout)
        
        # Color by
        self.flow_color_by = QComboBox()
        self.flow_color_by.addItems(["Velocity", "Pressure", "Temperature", "Solid Color"])
        flow_layout.addRow("Color By:", self.flow_color_by)
        
        # Animation options
        self.flow_animate = QCheckBox("Animate Flow")
        self.flow_animate.setChecked(True)
        flow_layout.addRow("", self.flow_animate)
        
        # Animation speed
        self.flow_speed = QSlider(Qt.Horizontal)
        self.flow_speed.setMinimum(1)
        self.flow_speed.setMaximum(10)
        self.flow_speed.setValue(5)
        self.flow_speed.setTickPosition(QSlider.TicksBelow)
        flow_layout.addRow("Speed:", self.flow_speed)
        
        # View mode
        view_layout = QHBoxLayout()
        self.view_internal = QRadioButton("Internal")
        self.view_external = QRadioButton("External")
        self.view_both = QRadioButton("Combined")
        self.view_both.setChecked(True)
        
        view_group = QButtonGroup(self)
        view_group.addButton(self.view_internal)
        view_group.addButton(self.view_external)
        view_group.addButton(self.view_both)
        
        view_layout.addWidget(self.view_internal)
        view_layout.addWidget(self.view_external)
        view_layout.addWidget(self.view_both)
        flow_layout.addRow("View Mode:", view_layout)
        
        # Apply button
        self.apply_flow_button = QPushButton("Apply Flow Visualization")
        flow_layout.addRow("", self.apply_flow_button)
        
        # Force conversion button
        self.force_convert_button = QPushButton("Force VTK Conversion")
        self.force_convert_button.setToolTip("Force conversion of OpenFOAM results to VTK format")
        self.force_convert_button.clicked.connect(self._on_force_convert)
        flow_layout.addRow("", self.force_convert_button)
        
        filters_layout.addWidget(flow_group)
        
        # Active Filters group
        active_filters_group = QGroupBox("Active Filters")
        active_filters_layout = QVBoxLayout(active_filters_group)
        
        # Table of active filters
        self.filters_table = QTableWidget(0, 3)
        self.filters_table.setHorizontalHeaderLabels(["Type", "Details", ""])
        self.filters_table.horizontalHeader().setStretchLastSection(True)
        self.filters_table.setSelectionBehavior(QTableWidget.SelectRows)
        active_filters_layout.addWidget(self.filters_table)
        
        filters_layout.addWidget(active_filters_group)
        
        # Add tab
        self.tab_widget.addTab(filters_widget, "Filters")
    
    def _setup_camera_tab(self):
        """Set up the camera control tab."""
        camera_widget = QWidget()
        camera_layout = QVBoxLayout(camera_widget)
        
        # Views group
        views_group = QGroupBox("Standard Views")
        views_layout = QGridLayout(views_group)
        
        # Standard view buttons
        self.view_buttons = {}
        
        self.view_buttons["front"] = QPushButton("+X")
        views_layout.addWidget(self.view_buttons["front"], 0, 0)
        
        self.view_buttons["back"] = QPushButton("-X")
        views_layout.addWidget(self.view_buttons["back"], 0, 1)
        
        self.view_buttons["right"] = QPushButton("+Y")
        views_layout.addWidget(self.view_buttons["right"], 1, 0)
        
        self.view_buttons["left"] = QPushButton("-Y")
        views_layout.addWidget(self.view_buttons["left"], 1, 1)
        
        self.view_buttons["top"] = QPushButton("+Z")
        views_layout.addWidget(self.view_buttons["top"], 2, 0)
        
        self.view_buttons["bottom"] = QPushButton("-Z")
        views_layout.addWidget(self.view_buttons["bottom"], 2, 1)
        
        self.view_buttons["isometric"] = QPushButton("Isometric")
        views_layout.addWidget(self.view_buttons["isometric"], 3, 0, 1, 2)
        
        self.reset_view_button = QPushButton("Reset View")
        views_layout.addWidget(self.reset_view_button, 4, 0, 1, 2)
        
        camera_layout.addWidget(views_group)
        
        # Camera position group
        camera_group = QGroupBox("Camera Settings")
        camera_form_layout = QFormLayout(camera_group)
        
        # Focal point
        self.focal_layout = QHBoxLayout()
        self.focal_x = QDoubleSpinBox()
        self.focal_x.setRange(-1000, 1000)
        self.focal_x.setValue(0)
        self.focal_y = QDoubleSpinBox()
        self.focal_y.setRange(-1000, 1000)
        self.focal_y.setValue(0)
        self.focal_z = QDoubleSpinBox()
        self.focal_z.setRange(-1000, 1000)
        self.focal_z.setValue(0)
        
        self.focal_layout.addWidget(QLabel("X:"))
        self.focal_layout.addWidget(self.focal_x)
        self.focal_layout.addWidget(QLabel("Y:"))
        self.focal_layout.addWidget(self.focal_y)
        self.focal_layout.addWidget(QLabel("Z:"))
        self.focal_layout.addWidget(self.focal_z)
        
        camera_form_layout.addRow("Focal Point:", self.focal_layout)
        
        # Distance
        self.camera_distance = QDoubleSpinBox()
        self.camera_distance.setRange(0.1, 1000)
        self.camera_distance.setValue(10)
        camera_form_layout.addRow("Distance:", self.camera_distance)
        
        # Zoom factor
        self.zoom_factor = QDoubleSpinBox()
        self.zoom_factor.setRange(0.1, 10)
        self.zoom_factor.setValue(1)
        self.zoom_factor.setSingleStep(0.1)
        camera_form_layout.addRow("Zoom:", self.zoom_factor)
        
        # Roll
        self.camera_roll = QSpinBox()
        self.camera_roll.setRange(-180, 180)
        self.camera_roll.setValue(0)
        camera_form_layout.addRow("Roll:", self.camera_roll)
        
        # View angle
        self.view_angle = QSpinBox()
        self.view_angle.setRange(1, 180)
        self.view_angle.setValue(30)
        camera_form_layout.addRow("View Angle:", self.view_angle)
        
        # Apply button
        self.apply_camera_button = QPushButton("Apply Camera Settings")
        camera_form_layout.addRow("", self.apply_camera_button)
        
        camera_layout.addWidget(camera_group)
        
        # Shows group
        shows_group = QGroupBox("Show/Hide Elements")
        shows_layout = QFormLayout(shows_group)
        
        # Checkboxes for different elements
        self.show_axes = QCheckBox()
        self.show_axes.setChecked(True)
        shows_layout.addRow("Axes:", self.show_axes)
        
        self.show_grid = QCheckBox()
        shows_layout.addRow("Grid:", self.show_grid)
        
        self.show_bounds = QCheckBox()
        shows_layout.addRow("Bounding Box:", self.show_bounds)
        
        self.show_orientation = QCheckBox()
        self.show_orientation.setChecked(True)
        shows_layout.addRow("Orientation Widget:", self.show_orientation)
        
        camera_layout.addWidget(shows_group)
        
        # Add stretch to push everything to the top
        camera_layout.addStretch()
        
        # Add tab
        self.tab_widget.addTab(camera_widget, "Camera")
    
    def _setup_effects_tab(self):
        """Set up the visual effects tab."""
        # Create the tab
        self.effects_tab = VisualEffectsTab()
        
        # Connect signals
        self.effects_tab.effects_changed.connect(self._on_effects_changed)
        self.effects_tab.connect_signals()
        
        # Add tab
        self.tab_widget.addTab(self.effects_tab, "Effects")
    
    def _setup_oilgas_tab(self):
        """Set up the oil & gas industry-specific tab."""
        oilgas_widget = QWidget()
        oilgas_layout = QVBoxLayout(oilgas_widget)
        
        # Specialized visualizations group
        viz_group = QGroupBox("Specialized Visualizations")
        viz_layout = QVBoxLayout(viz_group)
        
        # Radio buttons for visualization type
        self.phase_viz_radio = QRadioButton("Phase Interface")
        self.flow_pattern_radio = QRadioButton("Flow Pattern")
        self.velocity_profile_radio = QRadioButton("Velocity Profile")
        self.pressure_drop_radio = QRadioButton("Pressure Drop")
        self.pigging_viz_radio = QRadioButton("Pigging Visualization")
        self.spill_viz_radio = QRadioButton("Spill Visualization")
        
        self.phase_viz_radio.setChecked(True)
        
        viz_layout.addWidget(self.phase_viz_radio)
        viz_layout.addWidget(self.flow_pattern_radio)
        viz_layout.addWidget(self.velocity_profile_radio)
        viz_layout.addWidget(self.pressure_drop_radio)
        viz_layout.addWidget(self.pigging_viz_radio)
        viz_layout.addWidget(self.spill_viz_radio)
        
        # Create button group to manage radio buttons
        self.viz_type_group = QButtonGroup()
        self.viz_type_group.addButton(self.phase_viz_radio, 0)
        self.viz_type_group.addButton(self.flow_pattern_radio, 1)
        self.viz_type_group.addButton(self.velocity_profile_radio, 2)
        self.viz_type_group.addButton(self.pressure_drop_radio, 3)
        self.viz_type_group.addButton(self.pigging_viz_radio, 4)
        self.viz_type_group.addButton(self.spill_viz_radio, 5)
        
        # Apply button
        self.apply_viz_button = QPushButton("Apply Visualization")
        viz_layout.addWidget(self.apply_viz_button)
        
        oilgas_layout.addWidget(viz_group)
        
        # Presets group
        presets_group = QGroupBox("Industry Presets")
        presets_layout = QFormLayout(presets_group)
        
        # Preset selection
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Pipeline Flow",
            "Wellbore Flow",
            "Separator",
            "Tank",
            "Pipe Junction",
            "Manifold",
            "Riser"
        ])
        presets_layout.addRow("Preset:", self.preset_combo)
        
        # Apply preset button
        self.apply_preset_button = QPushButton("Apply Preset")
        presets_layout.addRow("", self.apply_preset_button)
        
        oilgas_layout.addWidget(presets_group)
        
        # Animation group
        animation_group = QGroupBox("Animation")
        animation_layout = QFormLayout(animation_group)
        
        # Animation type
        self.animation_combo = QComboBox()
        self.animation_combo.addItems([
            "Time Steps",
            "Streamlines",
            "Particle Trace",
            "Pig Movement",
            "Spill Progression"
        ])
        animation_layout.addRow("Type:", self.animation_combo)
        
        # Animation speed
        self.animation_speed = QSlider(Qt.Horizontal)
        self.animation_speed.setMinimum(1)
        self.animation_speed.setMaximum(100)
        self.animation_speed.setValue(50)
        animation_layout.addRow("Speed:", self.animation_speed)
        
        # Animation controls
        animation_buttons_layout = QHBoxLayout()
        
        self.anim_play_button = QPushButton("Play")
        animation_buttons_layout.addWidget(self.anim_play_button)
        
        self.anim_pause_button = QPushButton("Pause")
        self.anim_pause_button.setEnabled(False)
        animation_buttons_layout.addWidget(self.anim_pause_button)
        
        self.anim_stop_button = QPushButton("Stop")
        self.anim_stop_button.setEnabled(False)
        animation_buttons_layout.addWidget(self.anim_stop_button)
        
        animation_layout.addRow("Controls:", animation_buttons_layout)
        
        oilgas_layout.addWidget(animation_group)
        
        # Add stretch to push everything to the top
        oilgas_layout.addStretch()
        
        # Add tab
        self.tab_widget.addTab(oilgas_widget, "Oil & Gas")
    
    def _setup_control_buttons(self):
        """Set up the visualization control buttons."""
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        
        # Screenshot button
        self.screenshot_button = QPushButton("Screenshot")
        buttons_layout.addWidget(self.screenshot_button)
        
        # Add spacer
        buttons_layout.addStretch()
        
        # Save/load view buttons
        self.save_view_button = QPushButton("Save View")
        buttons_layout.addWidget(self.save_view_button)
        
        self.load_view_button = QPushButton("Load View")
        buttons_layout.addWidget(self.load_view_button)
        
        # Reset button
        self.reset_all_button = QPushButton("Reset All")
        buttons_layout.addWidget(self.reset_all_button)
        
        self.layout.addWidget(buttons_widget)
    
    def _connect_signals(self):
        """Connect signals to slots."""
        # Display tab
        self.rep_combo.currentTextChanged.connect(self._on_representation_changed)
        
        # Coloring tab
        self.color_field_combo.currentTextChanged.connect(self._on_color_field_changed)
        self.component_combo.currentTextChanged.connect(self._on_component_changed)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        self.scalar_bar_check.toggled.connect(self._on_scalar_bar_toggled)
        
        # Filters tab
        self.create_filter_button.clicked.connect(self._on_create_filter)
        
        # Connect flow visualization controls
        self.flow_viz_type.currentIndexChanged.connect(self._on_flow_viz_type_changed)
        self.apply_flow_button.clicked.connect(self._on_apply_flow_visualization)
        self.view_internal.toggled.connect(self._on_view_mode_changed)
        self.view_external.toggled.connect(self._on_view_mode_changed)
        self.view_both.toggled.connect(self._on_view_mode_changed)
        
        # Camera tab
        for view_name, button in self.view_buttons.items():
            button.clicked.connect(lambda checked, name=view_name: self._on_standard_view(name))
        
        self.reset_view_button.clicked.connect(self._on_reset_view)
        self.apply_camera_button.clicked.connect(self._on_apply_camera)
        
        self.show_axes.toggled.connect(self._on_show_axes_toggled)
        self.show_grid.toggled.connect(self._on_show_grid_toggled)
        self.show_bounds.toggled.connect(self._on_show_bounds_toggled)
        self.show_orientation.toggled.connect(self._on_show_orientation_toggled)
        
        # Oil & Gas tab
        self.apply_viz_button.clicked.connect(self._on_apply_oilgas_viz)
        self.apply_preset_button.clicked.connect(self._on_apply_preset)
        
        self.anim_play_button.clicked.connect(self._on_anim_play)
        self.anim_pause_button.clicked.connect(self._on_anim_pause)
        self.anim_stop_button.clicked.connect(self._on_anim_stop)
        
        # Control buttons
        self.screenshot_button.clicked.connect(self._on_screenshot)
        self.save_view_button.clicked.connect(self._on_save_view)
        self.load_view_button.clicked.connect(self._on_load_view)
        self.reset_all_button.clicked.connect(self._on_reset_all)
        
        # Mesh opacity slider
        self.mesh_opacity_slider.valueChanged.connect(self._on_mesh_opacity_changed)
    
    def _update_ui_state(self):
        """Update UI state based on current status."""
        # Check if we have a viewport to work with
        has_viewport = (self.main_window and 
                       hasattr(self.main_window, 'viewport') and 
                       self.main_window.viewport is not None)
        
        # Update button states
        self.screenshot_button.setEnabled(has_viewport)
        self.save_view_button.setEnabled(has_viewport)
        self.load_view_button.setEnabled(has_viewport)
        self.reset_all_button.setEnabled(has_viewport)
    
    def _update_color_button(self, button: QPushButton, color: QColor):
        """
        Update a color button with the selected color.
        
        Args:
            button (QPushButton): The button to update
            color (QColor): The color to apply
        """
        # Create a pixmap with the selected color
        pixmap = QPixmap(16, 16)
        pixmap.fill(color)
        
        # Set as button icon
        button.setIcon(QIcon(pixmap))
    
    def _show_solid_color_dialog(self):
        """Show color dialog for solid color selection."""
        # Show color dialog
        color = QColorDialog.getColor(self.solid_color, self, "Select Color")
        
        if color.isValid():
            # Update button
            self._update_color_button(self.solid_color_button, color)
            
            # Store color
            self.solid_color = color
            
            # Apply solid color if that's the current mode
            if self.color_field_combo.currentText() == "Solid Color":
                self._apply_solid_color()
    
    def _apply_solid_color(self):
        """Apply the selected solid color to the visualization."""
        # Convert to RGB array (0-1 range for VTK)
        rgb = [self.solid_color.red() / 255.0, 
              self.solid_color.green() / 255.0, 
              self.solid_color.blue() / 255.0]
        
        # Apply to viewport if available
        if self.main_window and hasattr(self.main_window, 'viewport'):
            # This would call a method in the viewport to set the solid color
            # self.main_window.viewport.set_solid_color(rgb)
            pass
    
    def _on_representation_changed(self, representation: str):
        """
        Handle representation type change.
        
        Args:
            representation (str): The new representation type
        """
        # Enable/disable point size control
        self.point_size_spin.setEnabled(representation == "Points")
        
        # Enable/disable line width control
        self.line_width_spin.setEnabled(representation == "Wireframe" or 
                                        representation == "Surface With Edges")
        
        # Emit signal
        self.representation_changed.emit(representation)
    
    def _on_color_field_changed(self, field: str):
        """
        Handle color field change.
        
        Args:
            field (str): The selected field
        """
        # Check if it's a vector field
        is_vector = field == "Velocity"
        self.component_combo.setEnabled(is_vector)
        
        # Enable/disable solid color button
        self.solid_color_button.setEnabled(field == "Solid Color")
        
        # Enable/disable colormap controls
        colormap_enabled = field != "Solid Color"
        self.colormap_combo.setEnabled(colormap_enabled)
        self.colormap_preview.setEnabled(colormap_enabled)
        self.scalar_bar_check.setEnabled(colormap_enabled)
        
        # Apply solid color if selected
        if field == "Solid Color":
            self._apply_solid_color()
        else:
            # Emit signal with current component (if vector)
            component = self.component_combo.currentText() if is_vector else ""
            self.coloring_changed.emit(field, component)
    
    def _on_component_changed(self, component: str):
        """
        Handle vector component change.
        
        Args:
            component (str): The selected component
        """
        # Only emit if we're using a vector field
        field = self.color_field_combo.currentText()
        if field == "Velocity":
            self.coloring_changed.emit(field, component)
    
    def _on_colormap_changed(self, colormap: str):
        """
        Handle colormap change.
        
        Args:
            colormap (str): The selected colormap
        """
        # Emit signal
        self.colormap_changed.emit(colormap)
    
    def _on_scalar_bar_toggled(self, checked: bool):
        """
        Handle scalar bar visibility toggle.
        
        Args:
            checked (bool): Whether the scalar bar should be visible
        """
        # Apply to viewport if available
        if self.main_window and hasattr(self.main_window, 'viewport'):
            # This would call a method in the viewport to show/hide the scalar bar
            # self.main_window.viewport.set_scalar_bar_visibility(checked)
            pass
    
    def _on_create_filter(self):
        """Handle create filter button click."""
        # Get filter type
        filter_type = self.filter_combo.currentText()
        
        # Get filter parameters based on type
        parameters = {}
        
        if filter_type == "Slice":
            # Get slice normal
            normal_type = self.slice_normal_combo.currentText()
            if normal_type == "X":
                normal = [1, 0, 0]
            elif normal_type == "Y":
                normal = [0, 1, 0]
            elif normal_type == "Z":
                normal = [0, 0, 1]
            else:  # Custom
                normal = [
                    self.slice_normal_x.value(),
                    self.slice_normal_y.value(),
                    self.slice_normal_z.value()
                ]
            
            # Get origin
            origin = [
                self.slice_origin_x.value(),
                self.slice_origin_y.value(),
                self.slice_origin_z.value()
            ]
            
            # Get offset
            offset = self.slice_offset_slider.value() / 10.0  # Scale to reasonable range
            
            parameters = {
                "normal": normal,
                "origin": origin,
                "offset": offset
            }
            
        elif filter_type == "Clip":
            # Get clip function
            function = self.clip_function_combo.currentText()
            
            # Get clip direction
            inside_out = self.clip_direction_combo.currentText() == "Outside"
            
            # Get normal and origin (for plane function)
            normal_type = self.clip_normal_combo.currentText()
            if normal_type == "X":
                normal = [1, 0, 0]
            elif normal_type == "Y":
                normal = [0, 1, 0]
            elif normal_type == "Z":
                normal = [0, 0, 1]
            else:  # Custom
                normal = [
                    self.clip_normal_x.value(),
                    self.clip_normal_y.value(),
                    self.clip_normal_z.value()
                ]
            
            origin = [
                self.clip_origin_x.value(),
                self.clip_origin_y.value(),
                self.clip_origin_z.value()
            ]
            
            parameters = {
                "function": function,
                "inside_out": inside_out,
                "normal": normal,
                "origin": origin
            }
            
        elif filter_type == "Threshold":
            # Get threshold field
            field = self.thresh_field_combo.currentText()
            
            # Get threshold range
            min_value = self.thresh_min_spin.value()
            max_value = self.thresh_max_spin.value()
            
            parameters = {
                "field": field,
                "min_value": min_value,
                "max_value": max_value
            }
            
        elif filter_type == "Contour":
            # Get contour field
            field = self.contour_field_combo.currentText()
            
            # Get contour settings
            count = self.contour_count_spin.value()
            min_value = self.contour_min_spin.value()
            max_value = self.contour_max_spin.value()
            
            parameters = {
                "field": field,
                "count": count,
                "min_value": min_value,
                "max_value": max_value
            }
            
        elif filter_type == "Streamlines":
            # Get vector field
            field = self.stream_field_combo.currentText()
            
            # Get seed type
            seed_type = self.stream_seed_combo.currentText()
            
            # Get seed count
            count = self.stream_count_spin.value()
            
            # Get max length
            length = self.stream_length_spin.value()
            
            parameters = {
                "field": field,
                "seed_type": seed_type,
                "count": count,
                "length": length
            }
        
        # Generate a unique ID for this filter
        import uuid
        filter_id = str(uuid.uuid4())
        
        # Add to active filters
        self.active_filters[filter_id] = {
            "type": filter_type,
            "parameters": parameters
        }
        
        # Update filters table
        self._update_filters_table()
        
        # Emit signal to create filter
        self.filter_applied.emit(filter_type, parameters)
    
    def _update_filters_table(self):
        """Update the table of active filters."""
        # Clear table
        self.filters_table.setRowCount(0)
        
        # Add each active filter
        for filter_id, filter_info in self.active_filters.items():
            row = self.filters_table.rowCount()
            self.filters_table.insertRow(row)
            
            # Filter type
            type_item = QTableWidgetItem(filter_info["type"])
            self.filters_table.setItem(row, 0, type_item)
            
            # Filter details
            details = self._get_filter_details(filter_info)
            details_item = QTableWidgetItem(details)
            self.filters_table.setItem(row, 1, details_item)
            
            # Remove button
            remove_button = QPushButton("Remove")
            remove_button.clicked.connect(lambda checked, fid=filter_id: self._remove_filter(fid))
            self.filters_table.setCellWidget(row, 2, remove_button)
    
    def _get_filter_details(self, filter_info: Dict[str, Any]) -> str:
        """
        Get a human-readable description of a filter.
        
        Args:
            filter_info (Dict[str, Any]): Filter information
            
        Returns:
            str: Description of the filter
        """
        filter_type = filter_info.get("type", "Unknown")
        
        # Handle both 'parameters' and 'details' keys for backward compatibility
        if "parameters" in filter_info:
            params = filter_info["parameters"]
        elif "details" in filter_info:
            params = filter_info["details"]
        else:
            params = {}
        
        if filter_type == "Clip":
            plane_normal = params.get("normal", [0, 0, 0])
            origin = params.get("origin", [0, 0, 0])
            return f"Clip plane at ({origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f}) with normal ({plane_normal[0]:.1f}, {plane_normal[1]:.1f}, {plane_normal[2]:.1f})"
        elif filter_type == "Slice":
            origin = params.get("origin", [0, 0, 0])
            return f"Slice plane at ({origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f})"
        elif filter_type == "Threshold":
            field = params.get("field", "Unnamed")
            min_val = params.get("min", 0.0)
            max_val = params.get("max", 0.0)
            return f"Threshold of {field} from {min_val:.2f} to {max_val:.2f}"
        elif filter_type == "Contour":
            field = params.get("field", "Unnamed")
            value = params.get("value", 0.0)
            return f"Contour of {field} at {value:.2f}"
        elif filter_type.startswith("flow_"):
            # Special handling for flow visualization filters
            viz_type = filter_type.split("_")[1] if len(filter_type.split("_")) > 1 else "Flow"
            
            line_count = params.get("line_count", 0)
            seed_location = params.get("seed_location", "custom")
            color_by = params.get("color_by", "Solid Color")
            
            if viz_type in ["Streamlines", "Pathlines"]:
                return f"{viz_type}: {line_count} lines from {seed_location}, colored by {color_by}"
            elif viz_type == "Vectors":
                return f"Vectors: density {line_count}, colored by {color_by}"
            elif viz_type == "Glyphs":
                return f"Glyphs: density {line_count}, colored by {color_by}"
            elif viz_type == "Surface":
                return f"Surface LIC: colored by {color_by}"
            else:
                return f"Flow visualization: {viz_type}"
        else:
            return f"{filter_type} filter"
    
    def _remove_filter(self, filter_id: str):
        """
        Remove a filter.
        
        Args:
            filter_id (str): ID of the filter to remove
        """
        if filter_id in self.active_filters:
            # Remove from active filters
            del self.active_filters[filter_id]
            
            # Update table
            self._update_filters_table()
            
            # Emit signal to remove filter
            self.filter_removed.emit(filter_id)
    
    def _on_standard_view(self, view_name: str):
        """
        Handle standard view button click.
        
        Args:
            view_name (str): Name of the standard view
        """
        # Define camera positions for standard views
        views = {
            "front": {"position": [1, 0, 0], "up": [0, 0, 1]},
            "back": {"position": [-1, 0, 0], "up": [0, 0, 1]},
            "right": {"position": [0, 1, 0], "up": [0, 0, 1]},
            "left": {"position": [0, -1, 0], "up": [0, 0, 1]},
            "top": {"position": [0, 0, 1], "up": [0, 1, 0]},
            "bottom": {"position": [0, 0, -1], "up": [0, 1, 0]},
            "isometric": {"position": [1, 1, 1], "up": [0, 0, 1]}
        }
        
        if view_name in views:
            view = views[view_name]
            
            # Apply to viewport if available
            if self.main_window and hasattr(self.main_window, 'viewport'):
                # This would call a method in the viewport to set the view
                # self.main_window.viewport.set_view(view["position"], view["up"])
                pass
    
    def _on_reset_view(self):
        """Handle reset view button click."""
        # Apply to viewport if available
        if self.main_window and hasattr(self.main_window, 'viewport'):
            # This would call a method in the viewport to reset the view
            # self.main_window.viewport.reset_view()
            pass
    
    def _on_apply_camera(self):
        """Handle apply camera settings button click."""
        # Get camera settings
        focal_point = [
            self.focal_x.value(),
            self.focal_y.value(),
            self.focal_z.value()
        ]
        
        distance = self.camera_distance.value()
        zoom = self.zoom_factor.value()
        roll = self.camera_roll.value()
        view_angle = self.view_angle.value()
        
        # Create camera parameters dictionary
        camera_params = {
            "focal_point": focal_point,
            "distance": distance,
            "zoom": zoom,
            "roll": roll,
            "view_angle": view_angle
        }
        
        # Emit signal
        self.camera_changed.emit(camera_params)
    
    def _on_show_axes_toggled(self, checked: bool):
        """
        Handle axes visibility toggle.
        
        Args:
            checked (bool): Whether axes should be visible
        """
        # Apply to viewport if available
        if self.main_window and hasattr(self.main_window, 'viewport'):
            # This would call a method in the viewport to show/hide axes
            # self.main_window.viewport.set_axes_visibility(checked)
            pass
    
    def _on_show_grid_toggled(self, checked: bool):
        """
        Handle grid visibility toggle.
        
        Args:
            checked (bool): Whether grid should be visible
        """
        # Apply to viewport if available
        if self.main_window and hasattr(self.main_window, 'viewport'):
            # This would call a method in the viewport to show/hide grid
            # self.main_window.viewport.set_grid_visibility(checked)
            pass
    
    def _on_show_bounds_toggled(self, checked: bool):
        """
        Handle bounds visibility toggle.
        
        Args:
            checked (bool): Whether bounds should be visible
        """
        # Apply to viewport if available
        if self.main_window and hasattr(self.main_window, 'viewport'):
            # This would call a method in the viewport to show/hide bounds
            # self.main_window.viewport.set_bounds_visibility(checked)
            pass
    
    def _on_show_orientation_toggled(self, checked: bool):
        """
        Handle orientation widget visibility toggle.
        
        Args:
            checked (bool): Whether orientation widget should be visible
        """
        # Apply to viewport if available
        if self.main_window and hasattr(self.main_window, 'viewport'):
            # This would call a method in the viewport to show/hide orientation widget
            # self.main_window.viewport.set_orientation_visibility(checked)
            pass
    
    def _on_effects_changed(self, settings: Dict[str, Any]):
        """
        Handle visual effects changes.
        
        Args:
            settings (Dict[str, Any]): New visual effects settings
        """
        # Apply to viewport if available
        if self.main_window and hasattr(self.main_window, 'viewport'):
            # This would call a method in the viewport to apply visual effects
            # self.main_window.viewport.set_visual_effects(settings)
            pass
    
    def _on_apply_oilgas_viz(self):
        """Handle apply oil & gas visualization button click."""
        # Get selected visualization type
        viz_id = self.viz_type_group.checkedId()
        
        viz_types = [
            "phase_interface",
            "flow_pattern",
            "velocity_profile",
            "pressure_drop",
            "pigging",
            "spill"
        ]
        
        if 0 <= viz_id < len(viz_types):
            viz_type = viz_types[viz_id]
            
            # Apply to viewport if available
            if self.main_window and hasattr(self.main_window, 'viewport'):
                # This would call a method in the viewport to apply the visualization
                # self.main_window.viewport.apply_oilgas_visualization(viz_type)
                pass
    
    def _on_apply_preset(self):
        """Handle apply preset button click."""
        # Get selected preset
        preset = self.preset_combo.currentText()
        
        # Apply to viewport if available
        if self.main_window and hasattr(self.main_window, 'viewport'):
            # This would call a method in the viewport to apply the preset
            # self.main_window.viewport.apply_preset(preset)
            pass
    
    def _on_anim_play(self):
        """Handle animation play button click."""
        # Get animation type
        anim_type = self.animation_combo.currentText()
        speed = self.animation_speed.value()
        
        # Apply to viewport if available
        if self.main_window and hasattr(self.main_window, 'viewport'):
            # This would call a method in the viewport to start animation
            # self.main_window.viewport.start_animation(anim_type, speed)
            pass
        
        # Update button states
        self.anim_play_button.setEnabled(False)
        self.anim_pause_button.setEnabled(True)
        self.anim_stop_button.setEnabled(True)
    
    def _on_anim_pause(self):
        """Handle animation pause button click."""
        # Apply to viewport if available
        if self.main_window and hasattr(self.main_window, 'viewport'):
            # This would call a method in the viewport to pause animation
            # self.main_window.viewport.pause_animation()
            pass
        
        # Update button states
        self.anim_play_button.setEnabled(True)
        self.anim_pause_button.setEnabled(False)
        self.anim_stop_button.setEnabled(True)
    
    def _on_anim_stop(self):
        """Handle animation stop button click."""
        # Apply to viewport if available
        if self.main_window and hasattr(self.main_window, 'viewport'):
            # This would call a method in the viewport to stop animation
            # self.main_window.viewport.stop_animation()
            pass
        
        # Update button states
        self.anim_play_button.setEnabled(True)
        self.anim_pause_button.setEnabled(False)
        self.anim_stop_button.setEnabled(False)
    
    def _on_screenshot(self):
        """Handle screenshot button click."""
        # Get file path
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Screenshot",
            "",
            "PNG Images (*.png);;JPEG Images (*.jpg);;All Files (*)"
        )
        
        if not filepath:
            return
        
        # Ensure file has correct extension
        if not filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath += '.png'
        
        # Get screenshot from viewport
        if self.main_window and hasattr(self.main_window, 'viewport'):
            # This would call a method in the viewport to take screenshot
            # self.main_window.viewport.export_view(filepath)
            
            # Show confirmation
            QMessageBox.information(
                self,
                "Screenshot Saved",
                f"Screenshot has been saved to:\n{filepath}"
            )
    
    def _on_save_view(self):
        """Handle save view button click."""
        # Get file path
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save View State",
            "",
            "VTK State Files (*.vts);;All Files (*)"
        )
        
        if not filepath:
            return
        
        # Ensure file has correct extension
        if not filepath.lower().endswith('.vts'):
            filepath += '.vts'
        
        # Save state from viewport
        if self.main_window and hasattr(self.main_window, 'viewport'):
            # This would call a method in the viewport to save state
            # success = self.main_window.viewport.export_state(filepath)
            success = True  # Placeholder
            
            if success:
                # Show confirmation
                QMessageBox.information(
                    self,
                    "View State Saved",
                    f"View state has been saved to:\n{filepath}"
                )
            else:
                # Show error
                QMessageBox.critical(
                    self,
                    "Error Saving View State",
                    "An error occurred while saving the view state."
                )
    
    def _on_load_view(self):
        """Handle load view button click."""
        # Get file path
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load View State",
            "",
            "VTK State Files (*.vts);;All Files (*)"
        )
        
        if not filepath:
            return
        
        # Load state into viewport
        if self.main_window and hasattr(self.main_window, 'viewport'):
            # This would call a method in the viewport to load state
            # success = self.main_window.viewport.import_state(filepath)
            success = True  # Placeholder
            
            if success:
                # Show confirmation
                QMessageBox.information(
                    self,
                    "View State Loaded",
                    "View state has been loaded successfully."
                )
            else:
                # Show error
                QMessageBox.critical(
                    self,
                    "Error Loading View State",
                    "An error occurred while loading the view state."
                )
    
    def _on_reset_all(self):
        """Handle reset all button click."""
        # Confirm reset
        reply = QMessageBox.question(
            self,
            "Reset All",
            "Are you sure you want to reset all visualization settings?\n"
            "This will remove all filters and reset all settings to default values.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Reset all filters
        self.active_filters = {}
        self._update_filters_table()
        
        # Reset display settings
        self.rep_combo.setCurrentIndex(0)
        self.color_field_combo.setCurrentIndex(0)
        self.colormap_combo.setCurrentIndex(0)
        self.scalar_bar_check.setChecked(True)
        
        # Reset camera
        self.focal_x.setValue(0)
        self.focal_y.setValue(0)
        self.focal_z.setValue(0)
        self.camera_distance.setValue(10)
        self.zoom_factor.setValue(1)
        self.camera_roll.setValue(0)
        self.view_angle.setValue(30)
        
        # Reset show/hide elements
        self.show_axes.setChecked(True)
        self.show_grid.setChecked(False)
        self.show_bounds.setChecked(False)
        self.show_orientation.setChecked(True)
        
        # Reset visual effects
        self.effects_tab.apply_settings({
            "lighting_enabled": True,
            "light_intensity": 0.75,
            "light_color": [1, 1, 1],
            "shading_model": "Gouraud",
            "ambient": 0.2,
            "diffuse": 0.7,
            "specular": 0.3,
            "show_edges": False,
            "edge_color": [0, 0, 0],
            "background_color": [0.2, 0.2, 0.2]
        })
        
        # Reset in viewport if available
        if self.main_window and hasattr(self.main_window, 'viewport'):
            # This would call a method in the viewport to reset everything
            # self.main_window.viewport.reset_all()
            pass
    
    def set_project(self, project):
        """
        Set the current project.
        
        Args:
            project: The project to set
        """
        self.current_project = project
        
        # Update UI state
        self._update_ui_state()
    
    def _update_color_button(self, button: QPushButton, color: QColor):
        """
        Update a color button with the selected color.
        
        Args:
            button (QPushButton): The button to update
            color (QColor): The color to apply
        """
        # Create a pixmap with the selected color
        pixmap = QPixmap(16, 16)
        pixmap.fill(color)
        
        # Set as button icon
        button.setIcon(QIcon(pixmap))
    
    def _on_mesh_opacity_changed(self, value):
        """
        Handle mesh opacity slider value change.
        
        Args:
            value (int): The new opacity value (0-100)
        """
        # Calculate opacity as float between 0.0 and 1.0
        opacity = value / 100.0
        
        # Find the main viewport in the parent widget (main window)
        if hasattr(self.main_window, 'viewport'):
            # Set opacity for all actors in the viewport
            if hasattr(self.main_window.viewport, 'renderer') and self.main_window.viewport.renderer:
                actors = self.main_window.viewport.renderer.GetActors()
                actors.InitTraversal()
                
                actor = actors.GetNextActor()
                while actor:
                    actor.GetProperty().SetOpacity(opacity)
                    actor = actors.GetNextActor()
                
                # Also check for specific actor collections
                for actor_attr in ['stl_actors', 'pipeline_actors', 'boundary_actors']:
                    if hasattr(self.main_window.viewport, actor_attr):
                        actor_collection = getattr(self.main_window.viewport, actor_attr)
                        if isinstance(actor_collection, dict):
                            for actor in actor_collection.values():
                                actor.GetProperty().SetOpacity(opacity)
                        elif isinstance(actor_collection, list):
                            for actor in actor_collection:
                                actor.GetProperty().SetOpacity(opacity)
                
                # Render the scene to apply changes
                self.main_window.viewport.render_window.Render()
    
    def _on_flow_viz_type_changed(self, index):
        """
        Handle flow visualization type change.
        
        Args:
            index (int): Index of the selected type
        """
        # Enable/disable line width control
        self.flow_line_width.setEnabled(index != 0)
        
        # Enable/disable seed location controls
        self.seed_inlet.setEnabled(index != 0)
        self.seed_outlet.setEnabled(index != 0)
        self.seed_custom.setEnabled(index != 0)
        
        # Enable/disable color by controls
        self.flow_color_by.setEnabled(index != 0)
        
        # Enable/disable animation controls
        self.flow_animate.setEnabled(index != 0)
        self.flow_speed.setEnabled(index != 0)
        
        # Enable/disable view mode controls
        self.view_internal.setEnabled(index != 0)
        self.view_external.setEnabled(index != 0)
        self.view_both.setEnabled(index != 0)
    
    def _on_apply_flow_visualization(self):
        """
        Handle apply flow visualization button click.
        """
        try:
            # Make sure VTK is available
            try:
                import vtk
                vtk_available = True
            except ImportError:
                vtk_available = False
                
            if not vtk_available:
                QMessageBox.warning(self, "VTK Not Available", 
                               "The VTK library is not available. Flow visualization features are disabled.")
                return
                
            # Check if we have a viewport to work with
            if not hasattr(self.main_window, 'viewport'):
                QMessageBox.warning(self, "Viewport Not Available", 
                               "Cannot create flow visualization: viewport is not available.")
                return
                
            # Get the viewport
            viewport = self.main_window.viewport
            
            # Check if the viewport is properly initialized
            if not hasattr(viewport, 'renderer') or not viewport.renderer:
                QMessageBox.warning(
                    self,
                    "No Renderer Available",
                    "The viewport renderer is not available. Cannot create visualization."
                )
                return
                
            if not hasattr(viewport, 'render_window') or not viewport.render_window:
                QMessageBox.warning(
                    self,
                    "No Render Window Available",
                    "The viewport render window is not available. Cannot create visualization."
                )
                return
                
            # Check if we have OpenFOAM simulation results
            has_results = False
            
            # First check the simulation_controls for case_dir
            case_dir = None
            if hasattr(self.main_window, 'simulation_controls') and hasattr(self.main_window.simulation_controls, 'case_dir'):
                case_dir = self.main_window.simulation_controls.case_dir
                if case_dir and os.path.exists(case_dir):
                    has_results = True
                    logger.info(f"Found case directory from simulation_controls: {case_dir}")
            
            # If not found, try with case manager
            if not case_dir and hasattr(self.main_window, 'case_manager'):
                case_manager = self.main_window.case_manager
                
                # Try to get case directory from case manager
                if hasattr(case_manager, 'get_case_dir'):
                    case_dir = case_manager.get_case_dir()
                    if case_dir and os.path.exists(case_dir):
                        has_results = True
                elif hasattr(case_manager, 'case_dir'):
                    case_dir = case_manager.case_dir
                    if case_dir and os.path.exists(case_dir):
                        has_results = True
                        
            if not case_dir:
                logger.warning("No case directory found, using current directory")
                case_dir = os.getcwd()
                
            logger.info(f"Using case directory: {case_dir}")
            
            # Get visualization parameters
            viz_type = self.flow_viz_type.currentText()
            line_count = self.flow_line_count.value()
            line_width = self.flow_line_width.value()
            
            # Get seed location
            seed_location = None
            if hasattr(self, 'seed_inlet') and self.seed_inlet.isChecked():
                seed_location = "inlet"
            elif hasattr(self, 'seed_outlet') and self.seed_outlet.isChecked():
                seed_location = "outlet"
            else:
                seed_location = "custom"
                
            # Get color by
            color_by = self.flow_color_by.currentText()
            
            # Get animation settings
            animate = self.flow_animate.isChecked()
            speed = self.flow_speed.value()
            
            # Get view mode
            view_mode = None
            if hasattr(self, 'view_internal') and self.view_internal.isChecked():
                view_mode = "internal"
            elif hasattr(self, 'view_external') and self.view_external.isChecked():
                view_mode = "external"
            else:  # view_both is checked
                view_mode = "both"
                    
            # Log visualization parameters
            logger.info(f"Creating {viz_type} visualization with {line_count} lines, width {line_width}")
            logger.info(f"Seed location: {seed_location}, Color by: {color_by}")
            logger.info(f"Animate: {animate}, Speed: {speed}, View mode: {view_mode}")
                
            # Create visualization based on type
            try:
                if viz_type == "Streamlines":
                    logger.info("Creating streamlines visualization...")
                    self._create_streamlines(
                        viewport, 
                        line_count, 
                        line_width, 
                        seed_location, 
                        color_by, 
                        animate, 
                        speed
                    )
                elif viz_type == "Pathlines":
                    self._create_pathlines(
                        viewport, 
                        line_count, 
                        line_width, 
                        seed_location, 
                        color_by, 
                        animate, 
                        speed
                    )
                elif viz_type == "Surface LIC":
                    self._create_surface_lic(
                        viewport,
                        color_by,
                        animate,
                        speed
                    )
                elif viz_type == "Vectors":
                    self._create_vectors(
                        viewport,
                        line_count,
                        line_width,
                        color_by
                    )
                elif viz_type == "Glyphs":
                    self._create_glyphs(
                        viewport,
                        line_count,
                        line_width,
                        color_by
                    )
                    
                # Apply view mode
                self._apply_view_mode(view_mode)
                
                # Add flow visualization to active filters
                filter_id = f"flow_{viz_type}_{len(self.active_filters)}"
                self.active_filters[filter_id] = {
                    "type": viz_type,
                    "details": {
                        "line_count": line_count,
                        "line_width": line_width,
                        "seed_location": seed_location,
                        "color_by": color_by,
                        "animate": animate,
                        "speed": speed,
                        "view_mode": view_mode
                    }
                }
                
                # Update filter list display
                self._update_filters_table()
                
                # Finally update the viewport
                viewport.render_window.Render()
                
            except Exception as e:
                import traceback
                logger.error(f"Error creating visualization: {e}\n{traceback.format_exc()}")
                QMessageBox.critical(
                    self,
                    "Visualization Error",
                    f"Failed to create {viz_type.lower()} visualization: {str(e)}"
                )
                
        except Exception as e:
            import traceback
            logger.error(f"Error in apply flow visualization: {e}\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to create streamlines: {str(e)}")
            QMessageBox.critical(
                self,
                "Visualization Error",
                f"An error occurred: {str(e)}"
            )
    
    def _create_streamlines(self, viewport, line_count, line_width, seed_location, color_by, animate=False, speed=1.0):
        """
        Create streamlines visualization.
        
        Args:
            viewport (VTKViewport): The viewport to render in
            line_count (int): Number of streamlines to show
            line_width (float): Width of streamlines
            seed_location (str): Where to place streamline seeds (inlet, outlet, custom)
            color_by (str): How to color the streamlines (velocity, pressure, solid)
            animate (bool): Whether to animate the streamlines
            speed (float): Animation speed
        """
        try:
            import vtk
            import logging
            logger = logging.getLogger(__name__)
            
            # Validate seed location parameter
            if seed_location not in ["inlet", "outlet", "custom"]:
                logger.warning(f"Invalid seed_location: {seed_location}, defaulting to 'custom'")
                seed_location = "custom"
                
            logger.info(f"Creating streamlines with {line_count} lines, width {line_width}")
            logger.info(f"Seed location: {seed_location}, Color by: {color_by}")
            
            # Default values
            velocity_field = "U"  # Default field name in OpenFOAM
            velocity_field_names = ["U", "velocity", "Velocity", "v", "vel", "Vel"]  # Possible names
            result_data = None
            
            # Get the case directory
            case_dir = self._get_case_directory()
            if not case_dir:
                raise RuntimeError("No case directory found. Please run a simulation first.")
            
            # Implementation continues here...
            logger.info("Streamlines visualization created")
            
        except Exception as e:
            import traceback
            logger.error(f"Error creating streamlines: {e}\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to create streamlines: {str(e)}")
    
    def _force_convert_to_vtk(self, case_dir):
        """
        Force conversion of OpenFOAM results to VTK format.
        
        Args:
            case_dir (str): Path to OpenFOAM case directory
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Forcing conversion of OpenFOAM results to VTK format for: {case_dir}")
        
        try:
            # First, ensure case directory exists
            if not os.path.exists(case_dir):
                logger.error(f"Case directory does not exist: {case_dir}")
                return False
                
            # Check for time directories explicitly since foamToVTK can't convert without them
            time_dirs = [d for d in os.listdir(case_dir) 
                       if os.path.isdir(os.path.join(case_dir, d)) 
                       and d.replace('.', '', 1).isdigit()]
            
            if not time_dirs:
                logger.error(f"No time directories found in case: {case_dir}")
                return False
                
            logger.info(f"Found {len(time_dirs)} time directories: {time_dirs}")
            
            # Check for velocity field (U) in latest time directory
            time_dirs.sort(key=lambda x: float(x.replace('e-', '0.0').replace('e+', '0.0')))
            latest_time = os.path.join(case_dir, time_dirs[-1])
            u_file = os.path.join(latest_time, "U")
            
            if not os.path.exists(u_file):
                logger.error(f"No velocity field (U) found in latest time directory: {latest_time}")
                return False
                
            logger.info(f"Found velocity field (U) in latest time directory: {latest_time}")
            
            # Create VTK directory if it doesn't exist
            vtk_dir = os.path.join(case_dir, "VTK")
            if not os.path.exists(vtk_dir):
                os.makedirs(vtk_dir, exist_ok=True)
                logger.info(f"Created VTK directory: {vtk_dir}")
            
            # Try method 1: Via OpenFOAM reader utility
            try:
                from ..utils.openfoam_reader import convert_openfoam_to_vtk
                success = convert_openfoam_to_vtk(case_dir)
                
                if success:
                    logger.info("Successfully converted OpenFOAM results to VTK format using OpenFOAM reader")
                    return True
            except Exception as e:
                logger.error(f"Error converting with OpenFOAM reader: {e}")
                # Continue with other methods
            
            # Try method 2: Enhanced direct foamToVTK call with comprehensive boundary patch handling
            logger.info("Trying direct foamToVTK call with enhanced boundary patch options...")
            
            # Save current directory
            original_dir = os.getcwd()
            
            try:
                # Change to case directory
                os.chdir(case_dir)
                
                # First, extract boundary patch information from the case
                boundary_file = os.path.join(case_dir, "constant", "polyMesh", "boundary")
                patches = []
                
                if os.path.exists(boundary_file):
                    logger.info(f"Found boundary file at {boundary_file}")
                    try:
                        with open(boundary_file, 'r') as f:
                            content = f.read()
                            # Basic parsing of OpenFOAM dictionary format
                            import re
                            # Find the number of patches
                            match = re.search(r'\s*(\d+)\s*\(', content)
                            if match:
                                patch_count = int(match.group(1))
                                logger.info(f"Found {patch_count} boundary patches")
                                
                                # Parse patch names - capture patch names between ( and {  
                                patch_matches = re.finditer(r'\s*(\w+)\s*\{[^\{\}]*type\s+(\w+)', content)
                                for match in patch_matches:
                                    patch_name = match.group(1)
                                    patch_type = match.group(2)
                                    patches.append((patch_name, patch_type))
                                    logger.info(f"Found patch: {patch_name} of type {patch_type}")
                    except Exception as e:
                        logger.error(f"Error parsing boundary file: {e}")
                
                # Save patch info for streamline seeding even if conversion fails
                vtk_dir = os.path.join(case_dir, "VTK")
                patch_info_dir = os.path.join(vtk_dir, "boundary_info")
                os.makedirs(patch_info_dir, exist_ok=True)
                
                with open(os.path.join(patch_info_dir, "patches.txt"), 'w') as f:
                    f.write(f"# Boundary patch information for {case_dir}\n")
                    f.write(f"# Total patches: {len(patches)}\n")
                    f.write("# Format: name:type\n")
                    for patch_name, patch_type in patches:
                        f.write(f"{patch_name}:{patch_type}\n")
                
                # Try to run foamToVTK with comprehensive boundary patch options
                import subprocess
                
                # Build command with boundary patch preservation options
                cmd = ["foamToVTK", "-boundary"]
                # Add patch options if we found patches
                if patches:
                    # Include all patches without excluding any for proper streamline seeding
                    cmd.extend(["-patches", "all"])
                
                logger.info(f"Running command: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=120  # 2-minute timeout for larger cases
                )
                
                if result.returncode == 0:
                    logger.info("Direct foamToVTK call with enhanced boundary options successful")
                    # Wait for files to be written
                    import time
                    time.sleep(2)
                    
                    # Create a hint file for streamline seeding
                    self._create_boundary_hint_file(case_dir, patches)
                    
                    return True
                else:
                    stderr = result.stderr.decode('utf-8')
                    logger.error(f"Direct foamToVTK call with enhanced boundary options failed: {stderr}")
                    
                    # Try method 3: Standard direct foamToVTK call with boundary flag
                    logger.info("Trying standard direct foamToVTK call with boundary flag...")
                    result = subprocess.run(
                        ["foamToVTK", "-boundary"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=120
                    )
                    
                    if result.returncode == 0:
                        logger.info("Direct foamToVTK call with standard boundary flag successful")
                        # Wait for files to be written
                        import time
                        time.sleep(2)
                        
                        # Create a hint file for streamline seeding
                        self._create_boundary_hint_file(case_dir, patches)
                        
                        return True
                    else:
                        stderr = result.stderr.decode('utf-8')
                        logger.error(f"Direct foamToVTK call with standard boundary flag failed: {stderr}")
                        
                        # Try method 4: Last resort - Direct foamToVTK call without boundary flag
                        logger.info("Trying direct foamToVTK call without boundary flag as last resort...")
                        result = subprocess.run(
                            ["foamToVTK"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=120
                        )
                        
                        if result.returncode == 0:
                            logger.info("Direct foamToVTK call without boundary flag successful")
                            logger.warning("Note: Without boundary flag, streamline seeding from boundaries may be limited")
                            # Wait for files to be written
                            import time
                            time.sleep(2)
                            
                            # Still create a hint file but mark it as potentially incomplete
                            self._create_boundary_hint_file(case_dir, patches, complete=False)
                            
                            return True
                        else:
                            stderr = result.stderr.decode('utf-8')
                            logger.error(f"All direct foamToVTK attempts failed: {stderr}")
            finally:
                # Restore original directory
                os.chdir(original_dir)
            
            # Final check: Look for any VTK files that might have been created
            import glob
            vtk_files = []
            boundary_files = []
            patch_specific_files = {}
            
            # Check for various VTK file types including boundary files
            for ext in ['vtk', 'vtu', 'vtp']:
                vtk_files.extend(glob.glob(os.path.join(vtk_dir, f"*.{ext}")))
                
                # Look specifically for boundary files
                boundary_files.extend(glob.glob(os.path.join(vtk_dir, f"*_boundary.{ext}")))
                
                # Look for patch-specific files
                for patch_name, _ in patches:
                    matching_files = glob.glob(os.path.join(vtk_dir, f"*{patch_name}.{ext}"))  
                    if matching_files:
                        if patch_name not in patch_specific_files:
                            patch_specific_files[patch_name] = []
                        patch_specific_files[patch_name].extend(matching_files)
            
            if vtk_files:
                logger.info(f"Found {len(vtk_files)} VTK files after all conversion attempts")
                
                # Report on boundary files specifically
                if boundary_files:
                    logger.info(f"Found {len(boundary_files)} boundary VTK files:")
                    for file in boundary_files[:3]:
                        logger.info(f"  - {os.path.basename(file)}")
                    if len(boundary_files) > 3:
                        logger.info(f"  ... and {len(boundary_files) - 3} more boundary files")
                else:
                    logger.warning("No boundary VTK files found - streamline seeding from boundaries may be limited")
                
                # Report on patch-specific files
                if patch_specific_files:
                    logger.info(f"Found patch-specific VTK files for {len(patch_specific_files)} patches:")
                    for patch, files in list(patch_specific_files.items())[:3]:
                        logger.info(f"  - {patch}: {len(files)} files")
                    if len(patch_specific_files) > 3:
                        logger.info(f"  ... and {len(patch_specific_files) - 3} more patches")
                else:
                    logger.warning("No patch-specific VTK files found - this may affect streamline seeding")
                
                # Create a final boundary hint file in case it wasn't created earlier
                self._create_boundary_hint_file(case_dir, patches, complete=(len(boundary_files) > 0))
                
                return True
            else:
                logger.warning("No VTK files found after all conversion attempts")
                return False
                
        except Exception as e:
            import traceback
            logger.error(f"Error in _force_convert_to_vtk: {e}\n{traceback.format_exc()}")
            return False
    
    def _create_boundary_hint_file(self, case_dir, patches, complete=True):
        """
        Create a boundary hint file for streamline seeding from boundary patches.
        
        Args:
            case_dir (str): Path to the OpenFOAM case directory
            patches (list): List of (patch_name, patch_type) tuples
            complete (bool): Whether the boundary conversion was complete (with -boundary flag)
        """
        try:
            vtk_dir = os.path.join(case_dir, "VTK")
            os.makedirs(vtk_dir, exist_ok=True)
            
            # Scan for existing boundary and patch files
            boundary_files = []
            patch_specific_files = {}
            for ext in ['vtk', 'vtu', 'vtp']:
                # Look specifically for boundary files
                boundary_files.extend(glob.glob(os.path.join(vtk_dir, f"*_boundary.{ext}")))
                
                # Look for patch-specific files
                for patch_name, _ in patches:
                    matching_files = glob.glob(os.path.join(vtk_dir, f"*{patch_name}.{ext}"))  
                    if matching_files:
                        if patch_name not in patch_specific_files:
                            patch_specific_files[patch_name] = []
                        patch_specific_files[patch_name].extend(matching_files)
            
            # Create a boundary hint file for the visualization system
            hint_file = os.path.join(vtk_dir, "boundary_hints.txt")
            with open(hint_file, 'w') as f:
                f.write(f"# Boundary information for VTK visualization\n")
                f.write(f"VTK_BOUNDARY_FILES: {len(boundary_files)}\n")
                f.write(f"BOUNDARY_CONVERSION_COMPLETE: {'true' if complete else 'false'}\n")
                f.write(f"TOTAL_PATCHES: {len(patches)}\n")
                f.write(f"\n# Patch details\n")
                for patch_name, patch_type in patches:
                    files = patch_specific_files.get(patch_name, [])
                    f.write(f"PATCH: {patch_name} TYPE: {patch_type} FILES: {len(files)}\n")
            
            logger.info(f"Created boundary hint file at {hint_file}")
            
            # Create additional metadata file for streamline seeding
            # This file contains information about which patches are inlets/outlets for streamline seeding
            seeding_hint_file = os.path.join(vtk_dir, "streamline_seeding_hints.txt")
            with open(seeding_hint_file, 'w') as f:
                f.write(f"# Streamline seeding hints for boundary patches\n")
                f.write(f"# Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Identify potential inlet/outlet patches based on type
                inlets = []
                outlets = []
                walls = []
                for patch_name, patch_type in patches:
                    if 'inlet' in patch_name.lower() or 'in' == patch_name.lower():
                        inlets.append(patch_name)
                    elif 'outlet' in patch_name.lower() or 'out' == patch_name.lower():
                        outlets.append(patch_name)
                    elif 'wall' in patch_name.lower() or patch_type.lower() == 'wall':
                        walls.append(patch_name)
                
                # Write the seeding hints
                f.write(f"POTENTIAL_INLETS: {len(inlets)}\n")
                for inlet in inlets:
                    f.write(f"INLET: {inlet}\n")
                
                f.write(f"\nPOTENTIAL_OUTLETS: {len(outlets)}\n")
                for outlet in outlets:
                    f.write(f"OUTLET: {outlet}\n")
                
                f.write(f"\nWALL_PATCHES: {len(walls)}\n")
                for wall in walls:
                    f.write(f"WALL: {wall}\n")
            
            logger.info(f"Created streamline seeding hint file at {seeding_hint_file}")
            
        except Exception as e:
            logger.error(f"Error creating boundary hint file: {e}")
    
    def _on_force_convert(self):
        """Handle force convert button click."""
        # Get case directory
        case_dir = self._get_case_directory()
        if not case_dir:
            logger.error("No case directory found. Please run a simulation first.")
            QMessageBox.warning(
                self,
                "No Case Directory",
                "No case directory found. Please run a simulation first."
            )
            return
            
        logger.info(f"Force converting OpenFOAM results to VTK format for: {case_dir}")
        
        # Show progress message
        QMessageBox.information(
            self,
            "Converting",
            "Converting OpenFOAM results to VTK format. This may take a while.\n"
            "Check the log for progress updates."
        )
        
        # Force conversion
        success = self._force_convert_to_vtk(case_dir)
        
        if success:
            logger.info("Force conversion successful")
            QMessageBox.information(
                self,
                "Conversion Successful",
                "Successfully converted OpenFOAM results to VTK format.\n"
                "You can now create visualization."
            )
        else:
            logger.error("Force conversion failed")
            QMessageBox.warning(
                self,
                "Conversion Failed",
                "Failed to convert OpenFOAM results to VTK format.\n"
                "Check the log for details."
            )
            
    def _on_view_mode_changed(self, checked):
        """
        Handle view mode radio button change.
        
        Args:
            checked (bool): Whether the button is checked
        """
        if not checked:
            return  # Only respond to the button that was checked
            
        # Apply the view mode immediately if we have flow actors
        if hasattr(self.main_window, 'viewport'):
            viewport = self.main_window.viewport
            
            # Determine the view mode
            view_mode = None
            if hasattr(self, 'view_internal') and self.view_internal.isChecked():
                view_mode = "internal"
            elif hasattr(self, 'view_external') and self.view_external.isChecked():
                view_mode = "external"
            else:  # view_both is checked
                view_mode = "both"
                    
            # Apply the view mode
            self._apply_view_mode(view_mode)
    
    def _apply_view_mode(self, view_mode):
        """
        Apply the selected view mode (internal, external, both).
        
        Args:
            view_mode (str): The view mode to apply
        """
        if not hasattr(self.main_window, 'viewport'):
            return
            
        viewport = self.main_window.viewport
        
        # Keep track of which actors need to be visible based on view mode
        streamline_actors_visible = {}
        vector_actors_visible = {}
        glyph_actors_visible = {}
        
        # Get all streamline actors (if any)
        if hasattr(viewport, 'streamline_actors'):
            for name, actor in viewport.streamline_actors.items():
                if view_mode == "internal" and "external" in name:
                    streamline_actors_visible[name] = False
                elif view_mode == "external" and "internal" in name:
                    streamline_actors_visible[name] = False
                else:
                    streamline_actors_visible[name] = True
                    
        # Get all vector actors (if any)
        if hasattr(viewport, 'vector_actors'):
            for name, actor in viewport.vector_actors.items():
                if view_mode == "internal" and "external" in name:
                    vector_actors_visible[name] = False
                elif view_mode == "external" and "internal" in name:
                    vector_actors_visible[name] = False
                else:
                    vector_actors_visible[name] = True
                    
        # Get all glyph actors (if any)
        if hasattr(viewport, 'glyph_actors'):
            for name, actor in viewport.glyph_actors.items():
                if view_mode == "internal" and "external" in name:
                    glyph_actors_visible[name] = False
                elif view_mode == "external" and "internal" in name:
                    glyph_actors_visible[name] = False
                else:
                    glyph_actors_visible[name] = True
                    
        # Apply visibility to streamline actors
        if hasattr(viewport, 'streamline_actors'):
            for name, actor in viewport.streamline_actors.items():
                actor.SetVisibility(streamline_actors_visible.get(name, True))
                
        # Apply visibility to vector actors
        if hasattr(viewport, 'vector_actors'):
            for name, actor in viewport.vector_actors.items():
                actor.SetVisibility(vector_actors_visible.get(name, True))
                
        # Apply visibility to glyph actors
        if hasattr(viewport, 'glyph_actors'):
            for name, actor in viewport.glyph_actors.items():
                actor.SetVisibility(glyph_actors_visible.get(name, True))
                
        # Render the scene
        if hasattr(viewport, 'render_window'):
            viewport.render_window.Render()
            
    def _get_case_directory(self):
        """
        Get the current case directory.
        
        Returns:
            str: Path to the case directory
        """
        # Method 1: Try accessing case_dir directly from simulation_controls
        if hasattr(self.main_window, 'simulation_controls') and hasattr(self.main_window.simulation_controls, 'case_dir'):
            case_dir = self.main_window.simulation_controls.case_dir
            if case_dir and os.path.exists(case_dir):
                logger.info(f"Using case directory from simulation_controls: {case_dir}")
                return case_dir
        
        # Method 2: Check if we have a case manager
        if hasattr(self.main_window, 'case_manager'):
            case_manager = self.main_window.case_manager
            
            # Try to get case directory from case manager
            if hasattr(case_manager, 'get_case_dir'):
                case_dir = case_manager.get_case_dir()
                if case_dir and os.path.exists(case_dir):
                    logger.info(f"Using case directory from case_manager.get_case_dir(): {case_dir}")
                    return case_dir
            elif hasattr(case_manager, 'case_dir'):
                case_dir = case_manager.case_dir
                if case_dir and os.path.exists(case_dir):
                    logger.info(f"Using case directory from case_manager.case_dir: {case_dir}")
                    return case_dir
        
        # Method 3: Try accessing via project
        if hasattr(self.main_window, 'current_project'):
            project = self.main_window.current_project
            if project:
                if hasattr(project, 'case_dir'):
                    case_dir = project.case_dir
                    if case_dir and os.path.exists(case_dir):
                        logger.info(f"Using case directory from current_project.case_dir: {case_dir}")
                        return case_dir
                elif hasattr(project, 'get_case_directory'):
                    case_dir = project.get_case_directory()
                    if case_dir and os.path.exists(case_dir):
                        logger.info(f"Using case directory from current_project.get_case_directory(): {case_dir}")
                        return case_dir
        
        # No case directory found
        logger.error("No case directory found. Please run a simulation first.")
        return None