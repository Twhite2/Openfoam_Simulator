#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mesh properties widget for Openfoam_Simulator application.

This widget provides an interface for viewing and configuring mesh properties
for CFD simulations in the oil & gas industry, including:
- Mesh statistics (cells, faces, points)
- Mesh quality metrics
- Mesh refinement controls
- Boundary conditions assignment
- Block mesh generation parameters
- Snappy Hex Mesh configuration
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
    QToolTip, QMenu, QAction, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QMessageBox, QToolButton, QSplitter
)
from PyQt5.QtCore import (
    Qt, QSize, pyqtSignal, QTimer, QPoint, QSettings, QRegExp
)
from PyQt5.QtGui import (
    QIcon, QPixmap, QColor, QPalette, QFont, QPainter, QBrush,
    QRegExpValidator, QIntValidator, QDoubleValidator
)

# Import utility modules
from ...utils.logger import get_logger
from ...config import get_value, set_value

logger = get_logger(__name__)


class MeshStatisticsWidget(QWidget):
    """
    Widget for displaying mesh statistics.
    
    This widget shows statistics about a mesh, including cell count,
    face count, point count, and quality metrics.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the mesh statistics widget.
        
        Args:
            parent: Parent widget
        """
        super(MeshStatisticsWidget, self).__init__(parent)
        
        # Setup UI
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(10)
        
        # Cell statistics group
        cell_group = QGroupBox("Cell Statistics")
        cell_layout = QFormLayout(cell_group)
        
        # Cell count
        self.cell_count_label = QLabel("0")
        cell_layout.addRow("Total Cells:", self.cell_count_label)
        
        # Hex cells
        self.hex_count_label = QLabel("0")
        cell_layout.addRow("Hexahedral Cells:", self.hex_count_label)
        
        # Tet cells
        self.tet_count_label = QLabel("0")
        cell_layout.addRow("Tetrahedral Cells:", self.tet_count_label)
        
        # Poly cells
        self.poly_count_label = QLabel("0")
        cell_layout.addRow("Polyhedral Cells:", self.poly_count_label)
        
        # Prism cells
        self.prism_count_label = QLabel("0")
        cell_layout.addRow("Prism Cells:", self.prism_count_label)
        
        # Pyramid cells
        self.pyramid_count_label = QLabel("0")
        cell_layout.addRow("Pyramid Cells:", self.pyramid_count_label)
        
        self.layout.addWidget(cell_group)
        
        # Point and face statistics
        point_face_group = QGroupBox("Points and Faces")
        point_face_layout = QFormLayout(point_face_group)
        
        # Point count
        self.point_count_label = QLabel("0")
        point_face_layout.addRow("Total Points:", self.point_count_label)
        
        # Face count
        self.face_count_label = QLabel("0")
        point_face_layout.addRow("Total Faces:", self.face_count_label)
        
        # Internal faces
        self.internal_faces_label = QLabel("0")
        point_face_layout.addRow("Internal Faces:", self.internal_faces_label)
        
        # Boundary faces
        self.boundary_faces_label = QLabel("0")
        point_face_layout.addRow("Boundary Faces:", self.boundary_faces_label)
        
        # Boundary count
        self.boundary_count_label = QLabel("0")
        point_face_layout.addRow("Number of Boundaries:", self.boundary_count_label)
        
        self.layout.addWidget(point_face_group)
        
        # Quality metrics group
        quality_group = QGroupBox("Mesh Quality Metrics")
        quality_layout = QFormLayout(quality_group)
        
        # Min non-orthogonality
        self.min_ortho_label = QLabel("0.0")
        quality_layout.addRow("Min Non-Orthogonality:", self.min_ortho_label)
        
        # Max non-orthogonality
        self.max_ortho_label = QLabel("0.0")
        quality_layout.addRow("Max Non-Orthogonality:", self.max_ortho_label)
        
        # Average non-orthogonality
        self.avg_ortho_label = QLabel("0.0")
        quality_layout.addRow("Avg Non-Orthogonality:", self.avg_ortho_label)
        
        # Max aspect ratio
        self.max_aspect_label = QLabel("0.0")
        quality_layout.addRow("Max Aspect Ratio:", self.max_aspect_label)
        
        # Average aspect ratio
        self.avg_aspect_label = QLabel("0.0")
        quality_layout.addRow("Avg Aspect Ratio:", self.avg_aspect_label)
        
        # Min volume
        self.min_volume_label = QLabel("0.0")
        quality_layout.addRow("Min Cell Volume:", self.min_volume_label)
        
        # Max skewness
        self.max_skewness_label = QLabel("0.0")
        quality_layout.addRow("Max Skewness:", self.max_skewness_label)
        
        self.layout.addWidget(quality_group)
        
        # Add stretch to push everything to the top
        self.layout.addStretch()
    
    def set_statistics(self, stats: Dict[str, Any]):
        """
        Set the statistics to display.
        
        Args:
            stats (Dict[str, Any]): Dictionary of mesh statistics
        """
        # Cell statistics
        if "cell_count" in stats:
            self.cell_count_label.setText(str(stats["cell_count"]))
        if "hex_count" in stats:
            self.hex_count_label.setText(str(stats["hex_count"]))
        if "tet_count" in stats:
            self.tet_count_label.setText(str(stats["tet_count"]))
        if "poly_count" in stats:
            self.poly_count_label.setText(str(stats["poly_count"]))
        if "prism_count" in stats:
            self.prism_count_label.setText(str(stats["prism_count"]))
        if "pyramid_count" in stats:
            self.pyramid_count_label.setText(str(stats["pyramid_count"]))
        
        # Point and face statistics
        if "point_count" in stats:
            self.point_count_label.setText(str(stats["point_count"]))
        if "face_count" in stats:
            self.face_count_label.setText(str(stats["face_count"]))
        if "internal_faces" in stats:
            self.internal_faces_label.setText(str(stats["internal_faces"]))
        if "boundary_faces" in stats:
            self.boundary_faces_label.setText(str(stats["boundary_faces"]))
        if "boundary_count" in stats:
            self.boundary_count_label.setText(str(stats["boundary_count"]))
        
        # Quality metrics
        if "min_orthogonality" in stats:
            self.min_ortho_label.setText(f"{stats['min_orthogonality']:.2f}")
        if "max_orthogonality" in stats:
            self.max_ortho_label.setText(f"{stats['max_orthogonality']:.2f}")
        if "avg_orthogonality" in stats:
            self.avg_ortho_label.setText(f"{stats['avg_orthogonality']:.2f}")
        if "max_aspect_ratio" in stats:
            self.max_aspect_label.setText(f"{stats['max_aspect_ratio']:.2f}")
        if "avg_aspect_ratio" in stats:
            self.avg_aspect_label.setText(f"{stats['avg_aspect_ratio']:.2f}")
        if "min_volume" in stats:
            self.min_volume_label.setText(f"{stats['min_volume']:.6e}")
        if "max_skewness" in stats:
            self.max_skewness_label.setText(f"{stats['max_skewness']:.2f}")
    
    def clear(self):
        """Clear all statistics."""
        # Cell statistics
        self.cell_count_label.setText("0")
        self.hex_count_label.setText("0")
        self.tet_count_label.setText("0")
        self.poly_count_label.setText("0")
        self.prism_count_label.setText("0")
        self.pyramid_count_label.setText("0")
        
        # Point and face statistics
        self.point_count_label.setText("0")
        self.face_count_label.setText("0")
        self.internal_faces_label.setText("0")
        self.boundary_faces_label.setText("0")
        self.boundary_count_label.setText("0")
        
        # Quality metrics
        self.min_ortho_label.setText("0.0")
        self.max_ortho_label.setText("0.0")
        self.avg_ortho_label.setText("0.0")
        self.max_aspect_label.setText("0.0")
        self.avg_aspect_label.setText("0.0")
        self.min_volume_label.setText("0.0")
        self.max_skewness_label.setText("0.0")


class BoundaryWidget(QWidget):
    """
    Widget for configuring mesh boundaries.
    
    This widget provides controls for viewing and editing mesh boundaries
    and their properties, such as patch type and settings.
    """
    
    # Signal emitted when boundary properties are changed
    boundary_changed = pyqtSignal(str, dict)
    
    def __init__(self, parent=None):
        """
        Initialize the boundary widget.
        
        Args:
            parent: Parent widget
        """
        super(BoundaryWidget, self).__init__(parent)
        
        # Store boundary data
        self.boundaries = {}  # Dict of boundary name -> properties
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(10)
        
        # Boundary list group
        boundaries_group = QGroupBox("Boundaries")
        boundaries_layout = QVBoxLayout(boundaries_group)
        
        # Table widget for boundaries
        self.boundary_table = QTableWidget(0, 3)
        self.boundary_table.setHorizontalHeaderLabels(["Name", "Type", "Face Count"])
        self.boundary_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.boundary_table.setSelectionMode(QTableWidget.SingleSelection)
        header = self.boundary_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        
        boundaries_layout.addWidget(self.boundary_table)
        
        # Boundary controls
        controls_layout = QHBoxLayout()
        
        self.add_boundary_button = QPushButton("Add")
        controls_layout.addWidget(self.add_boundary_button)
        
        self.remove_boundary_button = QPushButton("Remove")
        controls_layout.addWidget(self.remove_boundary_button)
        
        self.rename_boundary_button = QPushButton("Rename")
        controls_layout.addWidget(self.rename_boundary_button)
        
        boundaries_layout.addLayout(controls_layout)
        
        self.layout.addWidget(boundaries_group)
        
        # Boundary properties group
        properties_group = QGroupBox("Boundary Properties")
        properties_layout = QFormLayout(properties_group)
        
        # Boundary type
        self.type_combo = QComboBox()
        self.type_combo.addItems([
            "patch",          # Generic patch
            "wall",           # Wall boundary
            "symmetryPlane",  # Symmetry plane
            "empty",          # Empty patch (2D simulation)
            "wedge",          # Wedge patch (axisymmetric)
            "cyclic",         # Cyclic boundary
            "processor",      # Processor boundary (parallel)
            "inlet",          # Inlet boundary
            "outlet",         # Outlet boundary
            "atmosphere"      # Atmosphere boundary
        ])
        properties_layout.addRow("Type:", self.type_combo)
        
        # Physical type (for specialized boundary conditions)
        self.physical_type_combo = QComboBox()
        self.physical_type_combo.addItems([
            "default",       # Default (no special treatment)
            "inlet",         # Inlet boundary condition
            "outlet",        # Outlet boundary condition
            "wall",          # Wall boundary condition
            "symmetry",      # Symmetry boundary condition
            "cyclic",        # Cyclic boundary condition
            "atmosphere",    # Atmosphere boundary condition
            "interface"      # Interface (for multi-region)
        ])
        properties_layout.addRow("Physical Type:", self.physical_type_combo)
        
        # Face count (read-only)
        self.face_count_label = QLabel("0")
        properties_layout.addRow("Face Count:", self.face_count_label)
        
        # Boundary group (for grouping related boundaries)
        self.group_edit = QLineEdit()
        properties_layout.addRow("Group:", self.group_edit)
        
        # Apply button
        self.apply_button = QPushButton("Apply Changes")
        properties_layout.addRow("", self.apply_button)
        
        # Disable until a boundary is selected
        self._enable_properties(False)
        
        self.layout.addWidget(properties_group)
        
        # Auto-detect boundaries button
        self.auto_detect_button = QPushButton("Auto-Detect Boundaries")
        self.layout.addWidget(self.auto_detect_button)
        
        # Add stretch to push everything to the top
        self.layout.addStretch()
    
    def _connect_signals(self):
        """Connect signals to slots."""
        # Boundary table selection
        self.boundary_table.itemSelectionChanged.connect(self._on_boundary_selected)
        
        # Property changes
        self.apply_button.clicked.connect(self._on_apply_changes)
        
        # Boundary management
        self.add_boundary_button.clicked.connect(self._on_add_boundary)
        self.remove_boundary_button.clicked.connect(self._on_remove_boundary)
        self.rename_boundary_button.clicked.connect(self._on_rename_boundary)
        
        # Auto-detect
        self.auto_detect_button.clicked.connect(self._on_auto_detect)
    
    def _enable_properties(self, enable: bool):
        """
        Enable or disable property controls.
        
        Args:
            enable (bool): Whether to enable the controls
        """
        self.type_combo.setEnabled(enable)
        self.physical_type_combo.setEnabled(enable)
        self.group_edit.setEnabled(enable)
        self.apply_button.setEnabled(enable)
    
    def _on_boundary_selected(self):
        """Handle boundary selection in the table."""
        selected_rows = self.boundary_table.selectionModel().selectedRows()
        
        if selected_rows:
            # Get the boundary name from the first column
            row = selected_rows[0].row()
            boundary_name = self.boundary_table.item(row, 0).text()
            
            # Get boundary properties
            if boundary_name in self.boundaries:
                boundary = self.boundaries[boundary_name]
                
                # Update property controls
                self.type_combo.setCurrentText(boundary.get("type", "patch"))
                self.physical_type_combo.setCurrentText(boundary.get("physical_type", "default"))
                self.face_count_label.setText(str(boundary.get("face_count", 0)))
                self.group_edit.setText(boundary.get("group", ""))
                
                # Enable controls
                self._enable_properties(True)
            else:
                # Clear and disable controls
                self._enable_properties(False)
        else:
            # Clear and disable controls
            self._enable_properties(False)
    
    def _on_apply_changes(self):
        """Apply changes to the selected boundary."""
        selected_rows = self.boundary_table.selectionModel().selectedRows()
        
        if selected_rows:
            # Get the boundary name from the first column
            row = selected_rows[0].row()
            boundary_name = self.boundary_table.item(row, 0).text()
            
            # Update boundary properties
            if boundary_name in self.boundaries:
                boundary = self.boundaries[boundary_name]
                
                # Get values from controls
                boundary["type"] = self.type_combo.currentText()
                boundary["physical_type"] = self.physical_type_combo.currentText()
                boundary["group"] = self.group_edit.text()
                
                # Update table
                self.boundary_table.item(row, 1).setText(boundary["type"])
                
                # Emit signal
                self.boundary_changed.emit(boundary_name, boundary)
    
    def _on_add_boundary(self):
        """Add a new boundary."""
        # Prompt for name
        boundary_name, ok = QInputDialog.getText(
            self, "Add Boundary", "Boundary Name:"
        )
        
        if ok and boundary_name:
            # Check if name already exists
            if boundary_name in self.boundaries:
                QMessageBox.warning(
                    self, "Duplicate Name", 
                    f"A boundary named '{boundary_name}' already exists."
                )
                return
            
            # Create new boundary with default properties
            self.boundaries[boundary_name] = {
                "type": "patch",
                "physical_type": "default",
                "face_count": 0,
                "group": ""
            }
            
            # Add to table
            row = self.boundary_table.rowCount()
            self.boundary_table.insertRow(row)
            self.boundary_table.setItem(row, 0, QTableWidgetItem(boundary_name))
            self.boundary_table.setItem(row, 1, QTableWidgetItem("patch"))
            self.boundary_table.setItem(row, 2, QTableWidgetItem("0"))
            
            # Select new row
            self.boundary_table.selectRow(row)
    
    def _on_remove_boundary(self):
        """Remove the selected boundary."""
        selected_rows = self.boundary_table.selectionModel().selectedRows()
        
        if selected_rows:
            # Get the boundary name from the first column
            row = selected_rows[0].row()
            boundary_name = self.boundary_table.item(row, 0).text()
            
            # Confirm deletion
            reply = QMessageBox.question(
                self, "Confirm Deletion",
                f"Are you sure you want to remove boundary '{boundary_name}'?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Remove from dictionary
                if boundary_name in self.boundaries:
                    del self.boundaries[boundary_name]
                
                # Remove from table
                self.boundary_table.removeRow(row)
    
    def _on_rename_boundary(self):
        """Rename the selected boundary."""
        selected_rows = self.boundary_table.selectionModel().selectedRows()
        
        if selected_rows:
            # Get the boundary name from the first column
            row = selected_rows[0].row()
            old_name = self.boundary_table.item(row, 0).text()
            
            # Prompt for new name
            new_name, ok = QInputDialog.getText(
                self, "Rename Boundary", "New Name:", 
                text=old_name
            )
            
            if ok and new_name and new_name != old_name:
                # Check if name already exists
                if new_name in self.boundaries:
                    QMessageBox.warning(
                        self, "Duplicate Name", 
                        f"A boundary named '{new_name}' already exists."
                    )
                    return
                
                # Update dictionary
                if old_name in self.boundaries:
                    self.boundaries[new_name] = self.boundaries[old_name]
                    del self.boundaries[old_name]
                
                # Update table
                self.boundary_table.item(row, 0).setText(new_name)
    
    def _on_auto_detect(self):
        """Auto-detect boundaries from the mesh."""
        # This would typically call a function to analyze the mesh
        # and detect boundaries automatically. Here we just show a
        # placeholder message.
        QMessageBox.information(
            self, "Auto-Detect", 
            "This would automatically detect boundaries from the mesh.\n"
            "Implementation would depend on the specific mesh format and tools."
        )
    
    def set_boundaries(self, boundaries: Dict[str, Dict[str, Any]]):
        """
        Set the boundaries to display.
        
        Args:
            boundaries (Dict[str, Dict[str, Any]]): Dictionary of boundary properties
        """
        # Store boundaries
        self.boundaries = boundaries
        
        # Clear table
        self.boundary_table.setRowCount(0)
        
        # Add boundaries to table
        for name, props in boundaries.items():
            row = self.boundary_table.rowCount()
            self.boundary_table.insertRow(row)
            self.boundary_table.setItem(row, 0, QTableWidgetItem(name))
            self.boundary_table.setItem(row, 1, QTableWidgetItem(props.get("type", "patch")))
            self.boundary_table.setItem(row, 2, QTableWidgetItem(str(props.get("face_count", 0))))
        
        # Clear selection
        self.boundary_table.clearSelection()
        self._enable_properties(False)
    
    def get_boundaries(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current boundary configuration.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of boundary properties
        """
        return self.boundaries
    
    def clear(self):
        """Clear all boundaries."""
        self.boundaries = {}
        self.boundary_table.setRowCount(0)
        self._enable_properties(False)


class BlockMeshWidget(QWidget):
    """
    Widget for configuring OpenFOAM blockMesh mesh generation.
    
    This widget provides controls for setting up blockMesh parameters,
    which is used to generate simple structured meshes in OpenFOAM.
    """
    
    # Signal emitted when parameters are changed
    parameters_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """
        Initialize the blockMesh widget.
        
        Args:
            parent: Parent widget
        """
        super(BlockMeshWidget, self).__init__(parent)
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(10)
        
        # Geometry group
        geometry_group = QGroupBox("Geometry")
        geometry_layout = QVBoxLayout(geometry_group)
        
        # Geometry type selection
        self.geometry_type_group = QButtonGroup()
        
        # Radio buttons for different geometry types
        radio_layout = QHBoxLayout()
        
        self.box_radio = QRadioButton("Box")
        self.box_radio.setChecked(True)
        self.geometry_type_group.addButton(self.box_radio, 0)
        radio_layout.addWidget(self.box_radio)
        
        self.cylinder_radio = QRadioButton("Cylinder")
        self.geometry_type_group.addButton(self.cylinder_radio, 1)
        radio_layout.addWidget(self.cylinder_radio)
        
        self.pipe_radio = QRadioButton("Pipe")
        self.geometry_type_group.addButton(self.pipe_radio, 2)
        radio_layout.addWidget(self.pipe_radio)
        
        self.custom_radio = QRadioButton("Custom")
        self.geometry_type_group.addButton(self.custom_radio, 3)
        radio_layout.addWidget(self.custom_radio)
        
        geometry_layout.addLayout(radio_layout)
        
        # Stacked widget for different geometry type parameters
        self.geometry_stack = QStackedWidget()
        
        # 1. Box geometry
        box_widget = QWidget()
        box_layout = QFormLayout(box_widget)
        
        # Box dimensions
        box_dim_layout = QGridLayout()
        
        box_dim_layout.addWidget(QLabel("X:"), 0, 0)
        self.box_x_min = QDoubleSpinBox()
        self.box_x_min.setRange(-1000, 1000)
        self.box_x_min.setValue(0)
        self.box_x_min.setSuffix(" m")
        box_dim_layout.addWidget(self.box_x_min, 0, 1)
        
        box_dim_layout.addWidget(QLabel("to"), 0, 2)
        self.box_x_max = QDoubleSpinBox()
        self.box_x_max.setRange(-1000, 1000)
        self.box_x_max.setValue(1)
        self.box_x_max.setSuffix(" m")
        box_dim_layout.addWidget(self.box_x_max, 0, 3)
        
        box_dim_layout.addWidget(QLabel("Y:"), 1, 0)
        self.box_y_min = QDoubleSpinBox()
        self.box_y_min.setRange(-1000, 1000)
        self.box_y_min.setValue(0)
        self.box_y_min.setSuffix(" m")
        box_dim_layout.addWidget(self.box_y_min, 1, 1)
        
        box_dim_layout.addWidget(QLabel("to"), 1, 2)
        self.box_y_max = QDoubleSpinBox()
        self.box_y_max.setRange(-1000, 1000)
        self.box_y_max.setValue(1)
        self.box_y_max.setSuffix(" m")
        box_dim_layout.addWidget(self.box_y_max, 1, 3)
        
        box_dim_layout.addWidget(QLabel("Z:"), 2, 0)
        self.box_z_min = QDoubleSpinBox()
        self.box_z_min.setRange(-1000, 1000)
        self.box_z_min.setValue(0)
        self.box_z_min.setSuffix(" m")
        box_dim_layout.addWidget(self.box_z_min, 2, 1)
        
        box_dim_layout.addWidget(QLabel("to"), 2, 2)
        self.box_z_max = QDoubleSpinBox()
        self.box_z_max.setRange(-1000, 1000)
        self.box_z_max.setValue(1)
        self.box_z_max.setSuffix(" m")
        box_dim_layout.addWidget(self.box_z_max, 2, 3)
        
        box_layout.addRow("Dimensions:", box_dim_layout)
        
        # Cell counts
        cell_count_layout = QGridLayout()
        
        cell_count_layout.addWidget(QLabel("X:"), 0, 0)
        self.box_cells_x = QSpinBox()
        self.box_cells_x.setRange(1, 1000)
        self.box_cells_x.setValue(20)
        cell_count_layout.addWidget(self.box_cells_x, 0, 1)
        
        cell_count_layout.addWidget(QLabel("Y:"), 1, 0)
        self.box_cells_y = QSpinBox()
        self.box_cells_y.setRange(1, 1000)
        self.box_cells_y.setValue(20)
        cell_count_layout.addWidget(self.box_cells_y, 1, 1)
        
        cell_count_layout.addWidget(QLabel("Z:"), 2, 0)
        self.box_cells_z = QSpinBox()
        self.box_cells_z.setRange(1, 1000)
        self.box_cells_z.setValue(20)
        cell_count_layout.addWidget(self.box_cells_z, 2, 1)
        
        box_layout.addRow("Cell Count:", cell_count_layout)
        
        # Grading
        grading_layout = QGridLayout()
        
        grading_layout.addWidget(QLabel("X:"), 0, 0)
        self.box_grading_x = QDoubleSpinBox()
        self.box_grading_x.setRange(0.01, 100)
        self.box_grading_x.setValue(1)
        grading_layout.addWidget(self.box_grading_x, 0, 1)
        
        grading_layout.addWidget(QLabel("Y:"), 1, 0)
        self.box_grading_y = QDoubleSpinBox()
        self.box_grading_y.setRange(0.01, 100)
        self.box_grading_y.setValue(1)
        grading_layout.addWidget(self.box_grading_y, 1, 1)
        
        grading_layout.addWidget(QLabel("Z:"), 2, 0)
        self.box_grading_z = QDoubleSpinBox()
        self.box_grading_z.setRange(0.01, 100)
        self.box_grading_z.setValue(1)
        grading_layout.addWidget(self.box_grading_z, 2, 1)
        
        box_layout.addRow("Grading:", grading_layout)
        
        self.geometry_stack.addWidget(box_widget)
        
        # 2. Cylinder geometry
        cylinder_widget = QWidget()
        cylinder_layout = QFormLayout(cylinder_widget)
        
        # Cylinder parameters
        self.cylinder_radius = QDoubleSpinBox()
        self.cylinder_radius.setRange(0.001, 1000)
        self.cylinder_radius.setValue(0.5)
        self.cylinder_radius.setSuffix(" m")
        cylinder_layout.addRow("Radius:", self.cylinder_radius)
        
        self.cylinder_length = QDoubleSpinBox()
        self.cylinder_length.setRange(0.001, 1000)
        self.cylinder_length.setValue(1)
        self.cylinder_length.setSuffix(" m")
        cylinder_layout.addRow("Length:", self.cylinder_length)
        
        # Cell counts for cylinder
        cyl_cell_layout = QGridLayout()
        
        cyl_cell_layout.addWidget(QLabel("Radial:"), 0, 0)
        self.cylinder_cells_r = QSpinBox()
        self.cylinder_cells_r.setRange(1, 1000)
        self.cylinder_cells_r.setValue(10)
        cyl_cell_layout.addWidget(self.cylinder_cells_r, 0, 1)
        
        cyl_cell_layout.addWidget(QLabel("Circumferential:"), 1, 0)
        self.cylinder_cells_circ = QSpinBox()
        self.cylinder_cells_circ.setRange(4, 1000)
        self.cylinder_cells_circ.setValue(20)
        cyl_cell_layout.addWidget(self.cylinder_cells_circ, 1, 1)
        
        cyl_cell_layout.addWidget(QLabel("Axial:"), 2, 0)
        self.cylinder_cells_axial = QSpinBox()
        self.cylinder_cells_axial.setRange(1, 1000)
        self.cylinder_cells_axial.setValue(20)
        cyl_cell_layout.addWidget(self.cylinder_cells_axial, 2, 1)
        
        cylinder_layout.addRow("Cell Count:", cyl_cell_layout)
        
        # Grading for cylinder
        cyl_grading_layout = QGridLayout()
        
        cyl_grading_layout.addWidget(QLabel("Radial:"), 0, 0)
        self.cylinder_grading_r = QDoubleSpinBox()
        self.cylinder_grading_r.setRange(0.01, 100)
        self.cylinder_grading_r.setValue(1)
        cyl_grading_layout.addWidget(self.cylinder_grading_r, 0, 1)
        
        cyl_grading_layout.addWidget(QLabel("Axial:"), 1, 0)
        self.cylinder_grading_axial = QDoubleSpinBox()
        self.cylinder_grading_axial.setRange(0.01, 100)
        self.cylinder_grading_axial.setValue(1)
        cyl_grading_layout.addWidget(self.cylinder_grading_axial, 1, 1)
        
        cylinder_layout.addRow("Grading:", cyl_grading_layout)
        
        self.geometry_stack.addWidget(cylinder_widget)
        
        # 3. Pipe geometry
        pipe_widget = QWidget()
        pipe_layout = QFormLayout(pipe_widget)
        
        # Pipe parameters
        self.pipe_inner_radius = QDoubleSpinBox()
        self.pipe_inner_radius.setRange(0.001, 1000)
        self.pipe_inner_radius.setValue(0.25)
        self.pipe_inner_radius.setSuffix(" m")
        pipe_layout.addRow("Inner Radius:", self.pipe_inner_radius)
        
        self.pipe_outer_radius = QDoubleSpinBox()
        self.pipe_outer_radius.setRange(0.001, 1000)
        self.pipe_outer_radius.setValue(0.5)
        self.pipe_outer_radius.setSuffix(" m")
        pipe_layout.addRow("Outer Radius:", self.pipe_outer_radius)
        
        self.pipe_length = QDoubleSpinBox()
        self.pipe_length.setRange(0.001, 1000)
        self.pipe_length.setValue(1)
        self.pipe_length.setSuffix(" m")
        pipe_layout.addRow("Length:", self.pipe_length)
        
        # Cell counts for pipe
        pipe_cell_layout = QGridLayout()
        
        pipe_cell_layout.addWidget(QLabel("Radial:"), 0, 0)
        self.pipe_cells_r = QSpinBox()
        self.pipe_cells_r.setRange(1, 1000)
        self.pipe_cells_r.setValue(10)
        pipe_cell_layout.addWidget(self.pipe_cells_r, 0, 1)
        
        pipe_cell_layout.addWidget(QLabel("Circumferential:"), 1, 0)
        self.pipe_cells_circ = QSpinBox()
        self.pipe_cells_circ.setRange(4, 1000)
        self.pipe_cells_circ.setValue(20)
        pipe_cell_layout.addWidget(self.pipe_cells_circ, 1, 1)
        
        pipe_cell_layout.addWidget(QLabel("Axial:"), 2, 0)
        self.pipe_cells_axial = QSpinBox()
        self.pipe_cells_axial.setRange(1, 1000)
        self.pipe_cells_axial.setValue(20)
        pipe_cell_layout.addWidget(self.pipe_cells_axial, 2, 1)
        
        pipe_layout.addRow("Cell Count:", pipe_cell_layout)
        
        # Grading for pipe
        pipe_grading_layout = QGridLayout()
        
        pipe_grading_layout.addWidget(QLabel("Radial:"), 0, 0)
        self.pipe_grading_r = QDoubleSpinBox()
        self.pipe_grading_r.setRange(0.01, 100)
        self.pipe_grading_r.setValue(1)
        pipe_grading_layout.addWidget(self.pipe_grading_r, 0, 1)
        
        pipe_grading_layout.addWidget(QLabel("Axial:"), 1, 0)
        self.pipe_grading_axial = QDoubleSpinBox()
        self.pipe_grading_axial.setRange(0.01, 100)
        self.pipe_grading_axial.setValue(1)
        pipe_grading_layout.addWidget(self.pipe_grading_axial, 1, 1)
        
        pipe_layout.addRow("Grading:", pipe_grading_layout)
        
        self.geometry_stack.addWidget(pipe_widget)
        
        # 4. Custom geometry (uses a text editor for blockMeshDict)
        custom_widget = QWidget()
        custom_layout = QVBoxLayout(custom_widget)
        
        custom_info = QLabel(
            "Enter custom blockMeshDict content below. This will be used as-is."
        )
        custom_info.setWordWrap(True)
        custom_layout.addWidget(custom_info)
        
        self.custom_editor = QPlainTextEdit()
        self.custom_editor.setFont(QFont("Courier New", 10))
        self.custom_editor.setMinimumHeight(200)
        
        # Default template
        self.custom_editor.setPlainText("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

scale   1;

vertices
(
    (0 0 0)
    (1 0 0)
    (1 1 0)
    (0 1 0)
    (0 0 1)
    (1 0 1)
    (1 1 1)
    (0 1 1)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (20 20 20) simpleGrading (1 1 1)
);

boundary
(
    xmin
    {
        type patch;
        faces
        (
            (0 3 7 4)
        );
    }
    xmax
    {
        type patch;
        faces
        (
            (1 5 6 2)
        );
    }
    ymin
    {
        type patch;
        faces
        (
            (0 4 5 1)
        );
    }
    ymax
    {
        type patch;
        faces
        (
            (3 2 6 7)
        );
    }
    zmin
    {
        type patch;
        faces
        (
            (0 1 2 3)
        );
    }
    zmax
    {
        type patch;
        faces
        (
            (4 7 6 5)
        );
    }
);
""")
        
        custom_layout.addWidget(self.custom_editor)
        
        self.geometry_stack.addWidget(custom_widget)
        
        geometry_layout.addWidget(self.geometry_stack)
        
        self.layout.addWidget(geometry_group)
        
        # Connect geometry type to stack
        self.geometry_type_group.buttonClicked.connect(
            lambda button: self.geometry_stack.setCurrentIndex(self.geometry_type_group.id(button))
        )
        
        # Add separate controls for common boundary names
        boundary_group = QGroupBox("Boundary Names")
        boundary_layout = QGridLayout(boundary_group)
        
        # These fields let users customize the boundary names
        boundary_layout.addWidget(QLabel("X Min:"), 0, 0)
        self.boundary_xmin = QLineEdit("xmin")
        boundary_layout.addWidget(self.boundary_xmin, 0, 1)
        
        boundary_layout.addWidget(QLabel("X Max:"), 0, 2)
        self.boundary_xmax = QLineEdit("xmax")
        boundary_layout.addWidget(self.boundary_xmax, 0, 3)
        
        boundary_layout.addWidget(QLabel("Y Min:"), 1, 0)
        self.boundary_ymin = QLineEdit("ymin")
        boundary_layout.addWidget(self.boundary_ymin, 1, 1)
        
        boundary_layout.addWidget(QLabel("Y Max:"), 1, 2)
        self.boundary_ymax = QLineEdit("ymax")
        boundary_layout.addWidget(self.boundary_ymax, 1, 3)
        
        boundary_layout.addWidget(QLabel("Z Min:"), 2, 0)
        self.boundary_zmin = QLineEdit("zmin")
        boundary_layout.addWidget(self.boundary_zmin, 2, 1)
        
        boundary_layout.addWidget(QLabel("Z Max:"), 2, 2)
        self.boundary_zmax = QLineEdit("zmax")
        boundary_layout.addWidget(self.boundary_zmax, 2, 3)
        
        # For cylinder and pipe
        boundary_layout.addWidget(QLabel("Inner:"), 3, 0)
        self.boundary_inner = QLineEdit("inner")
        boundary_layout.addWidget(self.boundary_inner, 3, 1)
        
        boundary_layout.addWidget(QLabel("Outer:"), 3, 2)
        self.boundary_outer = QLineEdit("outer")
        boundary_layout.addWidget(self.boundary_outer, 3, 3)
        
        self.layout.addWidget(boundary_group)
        
        # Mesh options group
        options_group = QGroupBox("Mesh Options")
        options_layout = QFormLayout(options_group)
        
        # Merge tolerance
        self.merge_tolerance = QDoubleSpinBox()
        self.merge_tolerance.setRange(1e-10, 1)
        self.merge_tolerance.setDecimals(10)
        self.merge_tolerance.setValue(1e-6)
        self.merge_tolerance.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
        options_layout.addRow("Merge Tolerance:", self.merge_tolerance)
        
        # Scale
        self.scale = QDoubleSpinBox()
        self.scale.setRange(1e-10, 1000)
        self.scale.setValue(1)
        options_layout.addRow("Scale:", self.scale)
        
        # Generate button
        self.generate_button = QPushButton("Generate blockMesh")
        options_layout.addRow("", self.generate_button)
        
        self.layout.addWidget(options_group)
        
        # Add stretch to push everything to the top
        self.layout.addStretch()
    
    def _connect_signals(self):
        """Connect signals to slots."""
        # Connect all inputs to parameter_changed slot
        # Box parameters
        self.box_x_min.valueChanged.connect(self._on_parameter_changed)
        self.box_x_max.valueChanged.connect(self._on_parameter_changed)
        self.box_y_min.valueChanged.connect(self._on_parameter_changed)
        self.box_y_max.valueChanged.connect(self._on_parameter_changed)
        self.box_z_min.valueChanged.connect(self._on_parameter_changed)
        self.box_z_max.valueChanged.connect(self._on_parameter_changed)
        self.box_cells_x.valueChanged.connect(self._on_parameter_changed)
        self.box_cells_y.valueChanged.connect(self._on_parameter_changed)
        self.box_cells_z.valueChanged.connect(self._on_parameter_changed)
        self.box_grading_x.valueChanged.connect(self._on_parameter_changed)
        self.box_grading_y.valueChanged.connect(self._on_parameter_changed)
        self.box_grading_z.valueChanged.connect(self._on_parameter_changed)
        
        # Cylinder parameters
        self.cylinder_radius.valueChanged.connect(self._on_parameter_changed)
        self.cylinder_length.valueChanged.connect(self._on_parameter_changed)
        self.cylinder_cells_r.valueChanged.connect(self._on_parameter_changed)
        self.cylinder_cells_circ.valueChanged.connect(self._on_parameter_changed)
        self.cylinder_cells_axial.valueChanged.connect(self._on_parameter_changed)
        self.cylinder_grading_r.valueChanged.connect(self._on_parameter_changed)
        self.cylinder_grading_axial.valueChanged.connect(self._on_parameter_changed)
        
        # Pipe parameters
        self.pipe_inner_radius.valueChanged.connect(self._on_parameter_changed)
        self.pipe_outer_radius.valueChanged.connect(self._on_parameter_changed)
        self.pipe_length.valueChanged.connect(self._on_parameter_changed)
        self.pipe_cells_r.valueChanged.connect(self._on_parameter_changed)
        self.pipe_cells_circ.valueChanged.connect(self._on_parameter_changed)
        self.pipe_cells_axial.valueChanged.connect(self._on_parameter_changed)
        self.pipe_grading_r.valueChanged.connect(self._on_parameter_changed)
        self.pipe_grading_axial.valueChanged.connect(self._on_parameter_changed)
        
        # Custom editor
        self.custom_editor.textChanged.connect(self._on_parameter_changed)
        
        # Boundary names
        self.boundary_xmin.textChanged.connect(self._on_parameter_changed)
        self.boundary_xmax.textChanged.connect(self._on_parameter_changed)
        self.boundary_ymin.textChanged.connect(self._on_parameter_changed)
        self.boundary_ymax.textChanged.connect(self._on_parameter_changed)
        self.boundary_zmin.textChanged.connect(self._on_parameter_changed)
        self.boundary_zmax.textChanged.connect(self._on_parameter_changed)
        self.boundary_inner.textChanged.connect(self._on_parameter_changed)
        self.boundary_outer.textChanged.connect(self._on_parameter_changed)
        
        # Options
        self.merge_tolerance.valueChanged.connect(self._on_parameter_changed)
        self.scale.valueChanged.connect(self._on_parameter_changed)
        
        # Generate button
        self.generate_button.clicked.connect(self._on_generate)
    
    def _on_parameter_changed(self):
        """Handle parameter change events."""
        # Emit signal with current parameters
        self.parameters_changed.emit(self.get_parameters())
    
    def _on_generate(self):
        """Handle generate button click."""
        # This would typically call an OpenFOAM blockMesh generation function.
        # Here we just show a placeholder message.
        QMessageBox.information(
            self, "Generate blockMesh", 
            "This would generate a mesh using blockMesh.\n"
            "Implementation would call the appropriate OpenFOAM functions."
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current blockMesh parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of parameters
        """
        geometry_type = self.geometry_type_group.checkedId()
        
        # Common parameters
        params = {
            "geometry_type": ["box", "cylinder", "pipe", "custom"][geometry_type],
            "scale": self.scale.value(),
            "merge_tolerance": self.merge_tolerance.value(),
            "boundaries": {
                "xmin": self.boundary_xmin.text(),
                "xmax": self.boundary_xmax.text(),
                "ymin": self.boundary_ymin.text(),
                "ymax": self.boundary_ymax.text(),
                "zmin": self.boundary_zmin.text(),
                "zmax": self.boundary_zmax.text(),
                "inner": self.boundary_inner.text(),
                "outer": self.boundary_outer.text()
            }
        }
        
        # Geometry-specific parameters
        if geometry_type == 0:  # Box
            params.update({
                "box": {
                    "dimensions": {
                        "x": [self.box_x_min.value(), self.box_x_max.value()],
                        "y": [self.box_y_min.value(), self.box_y_max.value()],
                        "z": [self.box_z_min.value(), self.box_z_max.value()]
                    },
                    "cells": {
                        "x": self.box_cells_x.value(),
                        "y": self.box_cells_y.value(),
                        "z": self.box_cells_z.value()
                    },
                    "grading": {
                        "x": self.box_grading_x.value(),
                        "y": self.box_grading_y.value(),
                        "z": self.box_grading_z.value()
                    }
                }
            })
        elif geometry_type == 1:  # Cylinder
            params.update({
                "cylinder": {
                    "radius": self.cylinder_radius.value(),
                    "length": self.cylinder_length.value(),
                    "cells": {
                        "radial": self.cylinder_cells_r.value(),
                        "circumferential": self.cylinder_cells_circ.value(),
                        "axial": self.cylinder_cells_axial.value()
                    },
                    "grading": {
                        "radial": self.cylinder_grading_r.value(),
                        "axial": self.cylinder_grading_axial.value()
                    }
                }
            })
        elif geometry_type == 2:  # Pipe
            params.update({
                "pipe": {
                    "inner_radius": self.pipe_inner_radius.value(),
                    "outer_radius": self.pipe_outer_radius.value(),
                    "length": self.pipe_length.value(),
                    "cells": {
                        "radial": self.pipe_cells_r.value(),
                        "circumferential": self.pipe_cells_circ.value(),
                        "axial": self.pipe_cells_axial.value()
                    },
                    "grading": {
                        "radial": self.pipe_grading_r.value(),
                        "axial": self.pipe_grading_axial.value()
                    }
                }
            })
        elif geometry_type == 3:  # Custom
            params.update({
                "custom": {
                    "blockMeshDict": self.custom_editor.toPlainText()
                }
            })
        
        return params
    
    def set_parameters(self, params: Dict[str, Any]):
        """
        Set the blockMesh parameters.
        
        Args:
            params (Dict[str, Any]): Parameters to set
        """
        # Set geometry type
        if "geometry_type" in params:
            if params["geometry_type"] == "box":
                self.box_radio.setChecked(True)
                self.geometry_stack.setCurrentIndex(0)
            elif params["geometry_type"] == "cylinder":
                self.cylinder_radio.setChecked(True)
                self.geometry_stack.setCurrentIndex(1)
            elif params["geometry_type"] == "pipe":
                self.pipe_radio.setChecked(True)
                self.geometry_stack.setCurrentIndex(2)
            elif params["geometry_type"] == "custom":
                self.custom_radio.setChecked(True)
                self.geometry_stack.setCurrentIndex(3)
        
        # Set common parameters
        if "scale" in params:
            self.scale.setValue(params["scale"])
        if "merge_tolerance" in params:
            self.merge_tolerance.setValue(params["merge_tolerance"])
        
        # Set boundary names
        if "boundaries" in params:
            boundaries = params["boundaries"]
            if "xmin" in boundaries:
                self.boundary_xmin.setText(boundaries["xmin"])
            if "xmax" in boundaries:
                self.boundary_xmax.setText(boundaries["xmax"])
            if "ymin" in boundaries:
                self.boundary_ymin.setText(boundaries["ymin"])
            if "ymax" in boundaries:
                self.boundary_ymax.setText(boundaries["ymax"])
            if "zmin" in boundaries:
                self.boundary_zmin.setText(boundaries["zmin"])
            if "zmax" in boundaries:
                self.boundary_zmax.setText(boundaries["zmax"])
            if "inner" in boundaries:
                self.boundary_inner.setText(boundaries["inner"])
            if "outer" in boundaries:
                self.boundary_outer.setText(boundaries["outer"])
        
        # Set geometry-specific parameters
        if "box" in params:
            box = params["box"]
            if "dimensions" in box:
                dim = box["dimensions"]
                if "x" in dim and len(dim["x"]) == 2:
                    self.box_x_min.setValue(dim["x"][0])
                    self.box_x_max.setValue(dim["x"][1])
                if "y" in dim and len(dim["y"]) == 2:
                    self.box_y_min.setValue(dim["y"][0])
                    self.box_y_max.setValue(dim["y"][1])
                if "z" in dim and len(dim["z"]) == 2:
                    self.box_z_min.setValue(dim["z"][0])
                    self.box_z_max.setValue(dim["z"][1])
            if "cells" in box:
                cells = box["cells"]
                if "x" in cells:
                    self.box_cells_x.setValue(cells["x"])
                if "y" in cells:
                    self.box_cells_y.setValue(cells["y"])
                if "z" in cells:
                    self.box_cells_z.setValue(cells["z"])
            if "grading" in box:
                grading = box["grading"]
                if "x" in grading:
                    self.box_grading_x.setValue(grading["x"])
                if "y" in grading:
                    self.box_grading_y.setValue(grading["y"])
                if "z" in grading:
                    self.box_grading_z.setValue(grading["z"])
        
        elif "cylinder" in params:
            cylinder = params["cylinder"]
            if "radius" in cylinder:
                self.cylinder_radius.setValue(cylinder["radius"])
            if "length" in cylinder:
                self.cylinder_length.setValue(cylinder["length"])
            if "cells" in cylinder:
                cells = cylinder["cells"]
                if "radial" in cells:
                    self.cylinder_cells_r.setValue(cells["radial"])
                if "circumferential" in cells:
                    self.cylinder_cells_circ.setValue(cells["circumferential"])
                if "axial" in cells:
                    self.cylinder_cells_axial.setValue(cells["axial"])
            if "grading" in cylinder:
                grading = cylinder["grading"]
                if "radial" in grading:
                    self.cylinder_grading_r.setValue(grading["radial"])
                if "axial" in grading:
                    self.cylinder_grading_axial.setValue(grading["axial"])
        
        elif "pipe" in params:
            pipe = params["pipe"]
            if "inner_radius" in pipe:
                self.pipe_inner_radius.setValue(pipe["inner_radius"])
            if "outer_radius" in pipe:
                self.pipe_outer_radius.setValue(pipe["outer_radius"])
            if "length" in pipe:
                self.pipe_length.setValue(pipe["length"])
            if "cells" in pipe:
                cells = pipe["cells"]
                if "radial" in cells:
                    self.pipe_cells_r.setValue(cells["radial"])
                if "circumferential" in cells:
                    self.pipe_cells_circ.setValue(cells["circumferential"])
                if "axial" in cells:
                    self.pipe_cells_axial.setValue(cells["axial"])
            if "grading" in pipe:
                grading = pipe["grading"]
                if "radial" in grading:
                    self.pipe_grading_r.setValue(grading["radial"])
                if "axial" in grading:
                    self.pipe_grading_axial.setValue(grading["axial"])
        
        elif "custom" in params:
            custom = params["custom"]
            if "blockMeshDict" in custom:
                self.custom_editor.setPlainText(custom["blockMeshDict"])


class SnappyHexMeshWidget(QWidget):
    """
    Widget for configuring OpenFOAM snappyHexMesh mesh generation.
    
    This widget provides controls for setting up snappyHexMesh parameters,
    which is used to generate complex meshes from STL geometry in OpenFOAM.
    """
    
    # Signal emitted when parameters are changed
    parameters_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """
        Initialize the snappyHexMesh widget.
        
        Args:
            parent: Parent widget
        """
        super(SnappyHexMeshWidget, self).__init__(parent)
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(10)
        
        # Tab widget for different snappyHexMesh sections
        self.tab_widget = QTabWidget()
        
        # 1. Geometry tab
        geometry_widget = QWidget()
        geometry_layout = QVBoxLayout(geometry_widget)
        
        # STL file selection
        stl_group = QGroupBox("Geometry Files")
        stl_layout = QVBoxLayout(stl_group)
        
        # Table for STL files
        self.stl_table = QTableWidget(0, 3)
        self.stl_table.setHorizontalHeaderLabels(["Name", "File", "Type"])
        self.stl_table.setSelectionBehavior(QTableWidget.SelectRows)
        header = self.stl_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        
        stl_layout.addWidget(self.stl_table)
        
        # STL controls
        stl_controls_layout = QHBoxLayout()
        
        self.add_stl_button = QPushButton("Add")
        stl_controls_layout.addWidget(self.add_stl_button)
        
        self.remove_stl_button = QPushButton("Remove")
        stl_controls_layout.addWidget(self.remove_stl_button)
        
        stl_layout.addLayout(stl_controls_layout)
        
        geometry_layout.addWidget(stl_group)
        
        # Add stretch to push everything to the top
        geometry_layout.addStretch()
        
        self.tab_widget.addTab(geometry_widget, "Geometry")
        
        # 2. Castellated Mesh Controls tab
        castellated_widget = QWidget()
        castellated_layout = QVBoxLayout(castellated_widget)
        
        # Basic controls
        basic_group = QGroupBox("Basic Controls")
        basic_layout = QFormLayout(basic_group)
        
        # Enable castellatedMesh
        self.castellated_enabled_check = QCheckBox()
        self.castellated_enabled_check.setChecked(True)
        basic_layout.addRow("Enable Castellated Mesh:", self.castellated_enabled_check)
        
        # Max local cells
        self.max_local_cells_spin = QSpinBox()
        self.max_local_cells_spin.setRange(1000, 1000000000)
        self.max_local_cells_spin.setValue(1000000)
        self.max_local_cells_spin.setSingleStep(10000)
        basic_layout.addRow("Max Local Cells:", self.max_local_cells_spin)
        
        # Max global cells
        self.max_global_cells_spin = QSpinBox()
        self.max_global_cells_spin.setRange(1000, 2000000000)
        self.max_global_cells_spin.setValue(20000000)
        self.max_global_cells_spin.setSingleStep(1000000)
        basic_layout.addRow("Max Global Cells:", self.max_global_cells_spin)
        
        # Min refinement cells
        self.min_refinement_cells_spin = QSpinBox()
        self.min_refinement_cells_spin.setRange(0, 1000000)
        self.min_refinement_cells_spin.setValue(10)
        basic_layout.addRow("Min Refinement Cells:", self.min_refinement_cells_spin)
        
        # Max load balance
        self.max_load_balance_spin = QDoubleSpinBox()
        self.max_load_balance_spin.setRange(0, 1)
        self.max_load_balance_spin.setValue(0.1)
        self.max_load_balance_spin.setSingleStep(0.01)
        basic_layout.addRow("Max Load Imbalance:", self.max_load_balance_spin)
        
        # Resolve feature angle
        self.resolve_feature_angle_spin = QDoubleSpinBox()
        self.resolve_feature_angle_spin.setRange(0, 180)
        self.resolve_feature_angle_spin.setValue(30)
        self.resolve_feature_angle_spin.setSuffix("°")
        basic_layout.addRow("Feature Angle:", self.resolve_feature_angle_spin)
        
        castellated_layout.addWidget(basic_group)
        
        # Refinement levels
        refinement_group = QGroupBox("Refinement Controls")
        refinement_layout = QFormLayout(refinement_group)
        
        # Level 0 size
        self.level0_size_spin = QDoubleSpinBox()
        self.level0_size_spin.setRange(0.001, 1000)
        self.level0_size_spin.setValue(1)
        self.level0_size_spin.setSuffix(" m")
        refinement_layout.addRow("Level 0 Size:", self.level0_size_spin)
        
        # Max surface level
        self.max_surface_level_spin = QSpinBox()
        self.max_surface_level_spin.setRange(0, 10)
        self.max_surface_level_spin.setValue(4)
        refinement_layout.addRow("Max Surface Level:", self.max_surface_level_spin)
        
        # Max cell size
        self.max_cell_size_spin = QDoubleSpinBox()
        self.max_cell_size_spin.setRange(0.001, 1000)
        self.max_cell_size_spin.setValue(1)
        self.max_cell_size_spin.setSuffix(" m")
        refinement_layout.addRow("Max Cell Size:", self.max_cell_size_spin)
        
        # Min cell size
        self.min_cell_size_spin = QDoubleSpinBox()
        self.min_cell_size_spin.setRange(0.0001, 100)
        self.min_cell_size_spin.setValue(0.05)
        self.min_cell_size_spin.setSuffix(" m")
        refinement_layout.addRow("Min Cell Size:", self.min_cell_size_spin)
        
        castellated_layout.addWidget(refinement_group)
        
        # Add stretch to push everything to the top
        castellated_layout.addStretch()
        
        self.tab_widget.addTab(castellated_widget, "Castellated Mesh")
        
        # 3. Snap Controls tab
        snap_widget = QWidget()
        snap_layout = QVBoxLayout(snap_widget)
        
        # Basic controls
        snap_group = QGroupBox("Snap Controls")
        snap_form_layout = QFormLayout(snap_group)
        
        # Enable snap
        self.snap_enabled_check = QCheckBox()
        self.snap_enabled_check.setChecked(True)
        snap_form_layout.addRow("Enable Snap:", self.snap_enabled_check)
        
        # Tolerance
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0.001, 10)
        self.tolerance_spin.setValue(1)
        snap_form_layout.addRow("Tolerance:", self.tolerance_spin)
        
        # Number of iterations
        self.n_snap_iterations_spin = QSpinBox()
        self.n_snap_iterations_spin.setRange(1, 100)
        self.n_snap_iterations_spin.setValue(10)
        snap_form_layout.addRow("Iterations:", self.n_snap_iterations_spin)
        
        # Number of smoothing iterations
        self.n_smooth_iterations_spin = QSpinBox()
        self.n_smooth_iterations_spin.setRange(0, 100)
        self.n_smooth_iterations_spin.setValue(5)
        snap_form_layout.addRow("Smoothing Iterations:", self.n_smooth_iterations_spin)
        
        # Feature snap
        self.feature_snap_check = QCheckBox()
        self.feature_snap_check.setChecked(True)
        snap_form_layout.addRow("Feature Snap:", self.feature_snap_check)
        
        # Feature angle
        self.feature_angle_spin = QDoubleSpinBox()
        self.feature_angle_spin.setRange(0, 180)
        self.feature_angle_spin.setValue(30)
        self.feature_angle_spin.setSuffix("°")
        snap_form_layout.addRow("Feature Angle:", self.feature_angle_spin)
        
        snap_layout.addWidget(snap_group)
        
        # Add stretch to push everything to the top
        snap_layout.addStretch()
        
        self.tab_widget.addTab(snap_widget, "Snap")
        
        # 4. Layer Addition Controls tab
        layer_widget = QWidget()
        layer_layout = QVBoxLayout(layer_widget)
        
        # Basic controls
        layer_group = QGroupBox("Layer Controls")
        layer_form_layout = QFormLayout(layer_group)
        
        # Enable layers
        self.layers_enabled_check = QCheckBox()
        self.layers_enabled_check.setChecked(True)
        layer_form_layout.addRow("Enable Layers:", self.layers_enabled_check)
        
        # Relative sizes
        self.relative_sizes_check = QCheckBox()
        self.relative_sizes_check.setChecked(True)
        layer_form_layout.addRow("Relative Sizes:", self.relative_sizes_check)
        
        # Expansion ratio
        self.expansion_ratio_spin = QDoubleSpinBox()
        self.expansion_ratio_spin.setRange(1, 2)
        self.expansion_ratio_spin.setValue(1.25)
        self.expansion_ratio_spin.setSingleStep(0.01)
        layer_form_layout.addRow("Expansion Ratio:", self.expansion_ratio_spin)
        
        # Final layer thickness
        self.final_layer_thickness_spin = QDoubleSpinBox()
        self.final_layer_thickness_spin.setRange(0.001, 1)
        self.final_layer_thickness_spin.setValue(0.5)
        layer_form_layout.addRow("Final Layer Thickness:", self.final_layer_thickness_spin)
        
        # Min thickness
        self.min_thickness_spin = QDoubleSpinBox()
        self.min_thickness_spin.setRange(0.0001, 1)
        self.min_thickness_spin.setValue(0.1)
        layer_form_layout.addRow("Min Thickness:", self.min_thickness_spin)
        
        # Number of layers
        self.n_layers_spin = QSpinBox()
        self.n_layers_spin.setRange(0, 100)
        self.n_layers_spin.setValue(5)
        layer_form_layout.addRow("Number of Layers:", self.n_layers_spin)
        
        # Layer feature angle
        self.layer_feature_angle_spin = QDoubleSpinBox()
        self.layer_feature_angle_spin.setRange(0, 180)
        self.layer_feature_angle_spin.setValue(30)
        self.layer_feature_angle_spin.setSuffix("°")
        layer_form_layout.addRow("Feature Angle:", self.layer_feature_angle_spin)
        
        layer_layout.addWidget(layer_group)
        
        # Add stretch to push everything to the top
        layer_layout.addStretch()
        
        self.tab_widget.addTab(layer_widget, "Layers")
        
        # 5. Mesh Quality Controls tab
        quality_widget = QWidget()
        quality_layout = QVBoxLayout(quality_widget)
        
        # Quality controls
        quality_group = QGroupBox("Mesh Quality Controls")
        quality_form_layout = QFormLayout(quality_group)
        
        # Max non-orthogonality
        self.max_non_ortho_spin = QDoubleSpinBox()
        self.max_non_ortho_spin.setRange(0, 180)
        self.max_non_ortho_spin.setValue(70)
        quality_form_layout.addRow("Max Non-Orthogonality:", self.max_non_ortho_spin)
        
        # Max skewness
        self.max_skewness_spin = QDoubleSpinBox()
        self.max_skewness_spin.setRange(0, 10)
        self.max_skewness_spin.setValue(4)
        quality_form_layout.addRow("Max Skewness:", self.max_skewness_spin)
        
        # Min volume
        self.min_volume_spin = QDoubleSpinBox()
        self.min_volume_spin.setRange(-1, 1)
        self.min_volume_spin.setValue(1e-13)
        self.min_volume_spin.setDecimals(15)
        self.min_volume_spin.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
        quality_form_layout.addRow("Min Volume:", self.min_volume_spin)
        
        # Min face area
        self.min_area_spin = QDoubleSpinBox()
        self.min_area_spin.setRange(-1, 1)
        self.min_area_spin.setValue(1e-10)
        self.min_area_spin.setDecimals(12)
        self.min_area_spin.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
        quality_form_layout.addRow("Min Face Area:", self.min_area_spin)
        
        # Min twist
        self.min_twist_spin = QDoubleSpinBox()
        self.min_twist_spin.setRange(-1, 1)
        self.min_twist_spin.setValue(0.05)
        self.min_twist_spin.setSingleStep(0.01)
        quality_form_layout.addRow("Min Twist:", self.min_twist_spin)
        
        # Min determinant
        self.min_determinant_spin = QDoubleSpinBox()
        self.min_determinant_spin.setRange(0, 1)
        self.min_determinant_spin.setValue(0.001)
        self.min_determinant_spin.setSingleStep(0.001)
        quality_form_layout.addRow("Min Determinant:", self.min_determinant_spin)
        
        quality_layout.addWidget(quality_group)
        
        # Add stretch to push everything to the top
        quality_layout.addStretch()
        
        self.tab_widget.addTab(quality_widget, "Mesh Quality")
        
        self.layout.addWidget(self.tab_widget)
        
        # Generate button
        self.generate_button = QPushButton("Generate snappyHexMesh")
        self.layout.addWidget(self.generate_button)
    
    def _connect_signals(self):
        """Connect signals to slots."""
        # STL file controls
        self.add_stl_button.clicked.connect(self._on_add_stl)
        self.remove_stl_button.clicked.connect(self._on_remove_stl)
        
        # Castellated mesh controls
        self.castellated_enabled_check.toggled.connect(self._on_parameter_changed)
        self.max_local_cells_spin.valueChanged.connect(self._on_parameter_changed)
        self.max_global_cells_spin.valueChanged.connect(self._on_parameter_changed)
        self.min_refinement_cells_spin.valueChanged.connect(self._on_parameter_changed)
        self.max_load_balance_spin.valueChanged.connect(self._on_parameter_changed)
        self.resolve_feature_angle_spin.valueChanged.connect(self._on_parameter_changed)
        
        self.level0_size_spin.valueChanged.connect(self._on_parameter_changed)
        self.max_surface_level_spin.valueChanged.connect(self._on_parameter_changed)
        self.max_cell_size_spin.valueChanged.connect(self._on_parameter_changed)
        self.min_cell_size_spin.valueChanged.connect(self._on_parameter_changed)
        
        # Snap controls
        self.snap_enabled_check.toggled.connect(self._on_parameter_changed)
        self.tolerance_spin.valueChanged.connect(self._on_parameter_changed)
        self.n_snap_iterations_spin.valueChanged.connect(self._on_parameter_changed)
        self.n_smooth_iterations_spin.valueChanged.connect(self._on_parameter_changed)
        self.feature_snap_check.toggled.connect(self._on_parameter_changed)
        self.feature_angle_spin.valueChanged.connect(self._on_parameter_changed)
        
        # Layer controls
        self.layers_enabled_check.toggled.connect(self._on_parameter_changed)
        self.relative_sizes_check.toggled.connect(self._on_parameter_changed)
        self.expansion_ratio_spin.valueChanged.connect(self._on_parameter_changed)
        self.final_layer_thickness_spin.valueChanged.connect(self._on_parameter_changed)
        self.min_thickness_spin.valueChanged.connect(self._on_parameter_changed)
        self.n_layers_spin.valueChanged.connect(self._on_parameter_changed)
        self.layer_feature_angle_spin.valueChanged.connect(self._on_parameter_changed)
        
        # Quality controls
        self.max_non_ortho_spin.valueChanged.connect(self._on_parameter_changed)
        self.max_skewness_spin.valueChanged.connect(self._on_parameter_changed)
        self.min_volume_spin.valueChanged.connect(self._on_parameter_changed)
        self.min_area_spin.valueChanged.connect(self._on_parameter_changed)
        self.min_twist_spin.valueChanged.connect(self._on_parameter_changed)
        self.min_determinant_spin.valueChanged.connect(self._on_parameter_changed)
        
        # Generate button
        self.generate_button.clicked.connect(self._on_generate)
    
    def _on_add_stl(self):
        """Add a new STL file."""
        # Open file dialog to select STL file
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select STL File",
            "",
            "STL Files (*.stl);;All Files (*)"
        )
        
        if not filepath:
            return
        
        # Get a name for the STL region
        filename = os.path.basename(filepath)
        default_name = os.path.splitext(filename)[0]
        
        name, ok = QInputDialog.getText(
            self,
            "STL Region Name",
            "Enter a name for this STL region:",
            text=default_name
        )
        
        if not ok or not name:
            return
        
        # Get region type
        region_type, ok = QInputDialog.getItem(
            self,
            "Region Type",
            "Select the region type:",
            ["wall", "patch", "symmetry", "empty"],
            0,
            False
        )
        
        if not ok:
            return
        
        # Add to table
        row = self.stl_table.rowCount()
        self.stl_table.insertRow(row)
        self.stl_table.setItem(row, 0, QTableWidgetItem(name))
        self.stl_table.setItem(row, 1, QTableWidgetItem(filepath))
        self.stl_table.setItem(row, 2, QTableWidgetItem(region_type))
        
        # Emit parameter changed signal
        self._on_parameter_changed()
    
    def _on_remove_stl(self):
        """Remove the selected STL file."""
        selected_rows = self.stl_table.selectionModel().selectedRows()
        
        if selected_rows:
            # Remove the row
            self.stl_table.removeRow(selected_rows[0].row())
            
            # Emit parameter changed signal
            self._on_parameter_changed()
    
    def _on_parameter_changed(self):
        """Handle parameter change events."""
        # Emit signal with current parameters
        self.parameters_changed.emit(self.get_parameters())
    
    def _on_generate(self):
        """Handle generate button click."""
        # This would typically call an OpenFOAM snappyHexMesh generation function.
        # Here we just show a placeholder message.
        QMessageBox.information(
            self, "Generate snappyHexMesh", 
            "This would generate a mesh using snappyHexMesh.\n"
            "Implementation would call the appropriate OpenFOAM functions."
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current snappyHexMesh parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of parameters
        """
        # Get STL files
        stl_files = []
        for row in range(self.stl_table.rowCount()):
            name = self.stl_table.item(row, 0).text()
            filepath = self.stl_table.item(row, 1).text()
            region_type = self.stl_table.item(row, 2).text()
            
            stl_files.append({
                "name": name,
                "file": filepath,
                "type": region_type
            })
        
        # Create parameters dictionary
        params = {
            "geometry": {
                "stl_files": stl_files
            },
            "castellatedMesh": {
                "enabled": self.castellated_enabled_check.isChecked(),
                "maxLocalCells": self.max_local_cells_spin.value(),
                "maxGlobalCells": self.max_global_cells_spin.value(),
                "minRefinementCells": self.min_refinement_cells_spin.value(),
                "maxLoadUnbalance": self.max_load_balance_spin.value(),
                "resolveFeatureAngle": self.resolve_feature_angle_spin.value(),
                "level0Size": self.level0_size_spin.value(),
                "maxSurfaceLevel": self.max_surface_level_spin.value(),
                "maxCellSize": self.max_cell_size_spin.value(),
                "minCellSize": self.min_cell_size_spin.value()
            },
            "snap": {
                "enabled": self.snap_enabled_check.isChecked(),
                "tolerance": self.tolerance_spin.value(),
                "nSmoothPatch": self.n_smooth_iterations_spin.value(),
                "nSolveIter": self.n_snap_iterations_spin.value(),
                "nRelaxIter": 5,
                "featureSnap": self.feature_snap_check.isChecked(),
                "implicitFeatureSnap": True,
                "explicitFeatureSnap": False,
                "multiRegionFeatureSnap": False,
                "nFeatureSnapIter": 10,
                "featureEdgeAngle": self.feature_angle_spin.value()
            },
            "addLayers": {
                "enabled": self.layers_enabled_check.isChecked(),
                "relativeSizes": self.relative_sizes_check.isChecked(),
                "expansionRatio": self.expansion_ratio_spin.value(),
                "finalLayerThickness": self.final_layer_thickness_spin.value(),
                "minThickness": self.min_thickness_spin.value(),
                "nGrow": 0,
                "featureAngle": self.layer_feature_angle_spin.value(),
                "nRelaxIter": 5,
                "nSmoothSurfaceNormals": 1,
                "nSmoothNormals": 3,
                "nSmoothThickness": 10,
                "maxFaceThicknessRatio": 0.5,
                "maxThicknessToMedialRatio": 0.3,
                "minMedianAxisAngle": 90,
                "nBufferCellsNoExtrude": 0,
                "nLayerIter": 50,
                "nLayers": self.n_layers_spin.value()
            },
            "meshQualityControls": {
                "maxNonOrtho": self.max_non_ortho_spin.value(),
                "maxBoundarySkewness": self.max_skewness_spin.value(),
                "maxInternalSkewness": 4.0,
                "maxConcave": 80.0,
                "minVol": self.min_volume_spin.value(),
                "minTetQuality": 1e-30,
                "minArea": self.min_area_spin.value(),
                "minTwist": self.min_twist_spin.value(),
                "minDeterminant": self.min_determinant_spin.value(),
                "minFaceWeight": 0.05,
                "minVolRatio": 0.01,
                "minTriangleTwist": -1,
                "nSmoothScale": 4,
                "errorReduction": 0.75
            }
        }
        
        return params
    
    def set_parameters(self, params: Dict[str, Any]):
        """
        Set the snappyHexMesh parameters.
        
        Args:
            params (Dict[str, Any]): Parameters to set
        """
        # Set STL files
        if "geometry" in params and "stl_files" in params["geometry"]:
            # Clear existing files
            self.stl_table.setRowCount(0)
            
            # Add each file
            for stl in params["geometry"]["stl_files"]:
                row = self.stl_table.rowCount()
                self.stl_table.insertRow(row)
                self.stl_table.setItem(row, 0, QTableWidgetItem(stl["name"]))
                self.stl_table.setItem(row, 1, QTableWidgetItem(stl["file"]))
                self.stl_table.setItem(row, 2, QTableWidgetItem(stl["type"]))
        
        # Set castellatedMesh parameters
        if "castellatedMesh" in params:
            c = params["castellatedMesh"]
            if "enabled" in c:
                self.castellated_enabled_check.setChecked(c["enabled"])
            if "maxLocalCells" in c:
                self.max_local_cells_spin.setValue(c["maxLocalCells"])
            if "maxGlobalCells" in c:
                self.max_global_cells_spin.setValue(c["maxGlobalCells"])
            if "minRefinementCells" in c:
                self.min_refinement_cells_spin.setValue(c["minRefinementCells"])
            if "maxLoadUnbalance" in c:
                self.max_load_balance_spin.setValue(c["maxLoadUnbalance"])
            if "resolveFeatureAngle" in c:
                self.resolve_feature_angle_spin.setValue(c["resolveFeatureAngle"])
            if "level0Size" in c:
                self.level0_size_spin.setValue(c["level0Size"])
            if "maxSurfaceLevel" in c:
                self.max_surface_level_spin.setValue(c["maxSurfaceLevel"])
            if "maxCellSize" in c:
                self.max_cell_size_spin.setValue(c["maxCellSize"])
            if "minCellSize" in c:
                self.min_cell_size_spin.setValue(c["minCellSize"])
        
        # Set snap parameters
        if "snap" in params:
            s = params["snap"]
            if "enabled" in s:
                self.snap_enabled_check.setChecked(s["enabled"])
            if "tolerance" in s:
                self.tolerance_spin.setValue(s["tolerance"])
            if "nSmoothPatch" in s:
                self.n_smooth_iterations_spin.setValue(s["nSmoothPatch"])
            if "nSolveIter" in s:
                self.n_snap_iterations_spin.setValue(s["nSolveIter"])
            if "featureSnap" in s:
                self.feature_snap_check.setChecked(s["featureSnap"])
            if "featureEdgeAngle" in s:
                self.feature_angle_spin.setValue(s["featureEdgeAngle"])
        
        # Set layer parameters
        if "addLayers" in params:
            l = params["addLayers"]
            if "enabled" in l:
                self.layers_enabled_check.setChecked(l["enabled"])
            if "relativeSizes" in l:
                self.relative_sizes_check.setChecked(l["relativeSizes"])
            if "expansionRatio" in l:
                self.expansion_ratio_spin.setValue(l["expansionRatio"])
            if "finalLayerThickness" in l:
                self.final_layer_thickness_spin.setValue(l["finalLayerThickness"])
            if "minThickness" in l:
                self.min_thickness_spin.setValue(l["minThickness"])
            if "nLayers" in l:
                self.n_layers_spin.setValue(l["nLayers"])
            if "featureAngle" in l:
                self.layer_feature_angle_spin.setValue(l["featureAngle"])
        
        # Set quality parameters
        if "meshQualityControls" in params:
            q = params["meshQualityControls"]
            if "maxNonOrtho" in q:
                self.max_non_ortho_spin.setValue(q["maxNonOrtho"])
            if "maxBoundarySkewness" in q:
                self.max_skewness_spin.setValue(q["maxBoundarySkewness"])
            if "minVol" in q:
                self.min_volume_spin.setValue(q["minVol"])
            if "minArea" in q:
                self.min_area_spin.setValue(q["minArea"])
            if "minTwist" in q:
                self.min_twist_spin.setValue(q["minTwist"])
            if "minDeterminant" in q:
                self.min_determinant_spin.setValue(q["minDeterminant"])


class MeshPropertiesWidget(QWidget):
    """
    Main mesh properties widget for Openfoam_Simulator application.
    
    This widget provides a comprehensive interface for viewing and configuring
    mesh properties, including mesh statistics, boundary conditions, and mesh
    generation parameters.
    """
    
    # Signal emitted when mesh properties are changed
    properties_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """
        Initialize the mesh properties widget.
        
        Args:
            parent: Parent widget
        """
        super(MeshPropertiesWidget, self).__init__(parent)
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Tab widget for different sections
        self.tab_widget = QTabWidget()
        
        # 1. Statistics tab
        self.statistics_widget = MeshStatisticsWidget()
        self.tab_widget.addTab(self.statistics_widget, "Statistics")
        
        # 2. Boundaries tab
        self.boundary_widget = BoundaryWidget()
        self.tab_widget.addTab(self.boundary_widget, "Boundaries")
        
        # 3. Mesh Generation tab with tabs for different mesh generators
        mesh_gen_widget = QWidget()
        mesh_gen_layout = QVBoxLayout(mesh_gen_widget)
        mesh_gen_layout.setContentsMargins(0, 0, 0, 0)
        
        # Mesh type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Mesh Generator:"))
        
        self.mesh_type_combo = QComboBox()
        self.mesh_type_combo.addItems(["blockMesh", "snappyHexMesh"])
        type_layout.addWidget(self.mesh_type_combo)
        
        mesh_gen_layout.addLayout(type_layout)
        
        # Create a stacked widget for different mesh generators
        self.mesh_gen_stack = QStackedWidget()
        
        # Add blockMesh widget
        self.block_mesh_widget = BlockMeshWidget()
        self.mesh_gen_stack.addWidget(self.block_mesh_widget)
        
        # Add snappyHexMesh widget
        self.snappy_hex_mesh_widget = SnappyHexMeshWidget()
        self.mesh_gen_stack.addWidget(self.snappy_hex_mesh_widget)
        
        # Connect combo box to stack
        self.mesh_type_combo.currentIndexChanged.connect(self.mesh_gen_stack.setCurrentIndex)
        
        mesh_gen_layout.addWidget(self.mesh_gen_stack)
        
        self.tab_widget.addTab(mesh_gen_widget, "Mesh Generation")
        
        # Add tab widget to main layout
        self.layout.addWidget(self.tab_widget)
        
        # Add buttons at the bottom
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(5, 5, 5, 5)
        
        self.import_mesh_button = QPushButton("Import Mesh")
        button_layout.addWidget(self.import_mesh_button)
        
        self.export_mesh_button = QPushButton("Export Mesh")
        button_layout.addWidget(self.export_mesh_button)
        
        self.check_mesh_button = QPushButton("Check Mesh")
        button_layout.addWidget(self.check_mesh_button)
        
        self.layout.addLayout(button_layout)
    
    def _connect_signals(self):
        """Connect signals to slots."""
        # Boundary widget signals
        self.boundary_widget.boundary_changed.connect(self._on_boundary_changed)
        
        # Mesh generation widgets
        self.block_mesh_widget.parameters_changed.connect(self._on_block_mesh_changed)
        self.snappy_hex_mesh_widget.parameters_changed.connect(self._on_snappy_hex_changed)
        
        # Button signals
        self.import_mesh_button.clicked.connect(self._on_import_mesh)
        self.export_mesh_button.clicked.connect(self._on_export_mesh)
        self.check_mesh_button.clicked.connect(self._on_check_mesh)
        
    def _on_boundary_changed(self, boundary_name: str, properties: Dict[str, Any]):
        """
        Handle boundary property changes.
        
        Args:
            boundary_name (str): Name of the boundary
            properties (Dict[str, Any]): Boundary properties
        """
        # Update mesh data
        if "boundaries" not in self.mesh_data:
            self.mesh_data["boundaries"] = {}
        
        self.mesh_data["boundaries"][boundary_name] = properties
        
        # Emit signal
        self.properties_changed.emit(self.mesh_data)
    
    def _on_block_mesh_changed(self, parameters: Dict[str, Any]):
        """
        Handle blockMesh parameter changes.
        
        Args:
            parameters (Dict[str, Any]): blockMesh parameters
        """
        # Update mesh data
        self.mesh_data["blockMesh"] = parameters
        
        # Emit signal
        self.properties_changed.emit(self.mesh_data)
    
    def _on_snappy_hex_changed(self, parameters: Dict[str, Any]):
        """
        Handle snappyHexMesh parameter changes.
        
        Args:
            parameters (Dict[str, Any]): snappyHexMesh parameters
        """
        # Update mesh data
        self.mesh_data["snappyHexMesh"] = parameters
        
        # Emit signal
        self.properties_changed.emit(self.mesh_data)
    
    def _on_import_mesh(self):
        """Handle import mesh button click."""
        # Show file dialog
        filepath, filter_str = QFileDialog.getOpenFileName(
            self,
            "Import Mesh",
            "",
            "OpenFOAM Mesh (*.foam);;STL Files (*.stl);;VTK Files (*.vtk *.vtu);;All Files (*)"
        )
        
        if not filepath:
            return
        
        # Determine mesh type
        if filepath.lower().endswith('.stl'):
            mesh_type = 'STL'
        elif filepath.lower().endswith(('.vtk', '.vtu')):
            mesh_type = 'VTK'
        elif filepath.lower().endswith('.foam') or os.path.isdir(filepath):
            mesh_type = 'OpenFOAM'
        else:
            mesh_type = 'Unknown'
        
        # Store mesh file and type
        self.mesh_file = filepath
        self.mesh_type = mesh_type
        
        # This would typically load the mesh and update the UI
        # For now, we just show a placeholder message
        QMessageBox.information(
            self,
            "Mesh Imported",
            f"Mesh file '{os.path.basename(filepath)}' of type {mesh_type} has been imported.\n"
            "This would load the mesh geometry and update the UI with mesh statistics."
        )
        
        # Load some placeholder statistics
        self._load_placeholder_statistics()
        
        # Emit signal
        self.mesh_imported.emit(filepath)
    
    def _on_export_mesh(self):
        """Handle export mesh button click."""
        if not self.mesh_file:
            QMessageBox.warning(
                self,
                "No Mesh",
                "No mesh is currently loaded to export."
            )
            return
        
        # Show file dialog
        filepath, filter_str = QFileDialog.getSaveFileName(
            self,
            "Export Mesh",
            "",
            "OpenFOAM Mesh (*.foam);;STL Files (*.stl);;VTK Files (*.vtk);;All Files (*)"
        )
        
        if not filepath:
            return
        
        # This would typically save the mesh
        # For now, we just show a placeholder message
        QMessageBox.information(
            self,
            "Mesh Exported",
            f"Mesh has been exported to '{filepath}'.\n"
            "This would save the mesh in the selected format."
        )
        
        # Emit signal
        self.mesh_exported.emit(filepath)
    
    def _on_check_mesh(self):
        """Handle check mesh button click."""
        if not self.mesh_file:
            QMessageBox.warning(
                self,
                "No Mesh",
                "No mesh is currently loaded to check."
            )
            return
        
        # This would typically run checkMesh or a similar tool
        # For now, we just show a placeholder message
        QMessageBox.information(
            self,
            "Mesh Check",
            "This would run OpenFOAM's checkMesh utility on the current mesh.\n"
            "Results would be shown in the statistics tab."
        )
        
        # Emit signal - in a real implementation, this would send actual results
        self.mesh_checked.emit(True, "Mesh check completed successfully.")
    
    def _load_placeholder_statistics(self):
        """Load placeholder mesh statistics."""
        # This is just for demonstration purposes
        # In a real implementation, this would load actual mesh statistics
        stats = {
            "cell_count": 125000,
            "hex_count": 125000,
            "tet_count": 0,
            "poly_count": 0,
            "prism_count": 0,
            "pyramid_count": 0,
            "point_count": 132651,
            "face_count": 381250,
            "internal_faces": 375000,
            "boundary_faces": 6250,
            "boundary_count": 6,
            "min_orthogonality": 0.01,
            "max_orthogonality": 45.3,
            "avg_orthogonality": 12.7,
            "max_aspect_ratio": 12.5,
            "avg_aspect_ratio": 3.2,
            "min_volume": 1.23e-8,
            "max_skewness": 0.82
        }
        
        # Update statistics widget
        self.statistics_widget.set_statistics(stats)
        
        # Set up placeholder boundaries
        boundaries = {
            "inlet": {
                "type": "patch",
                "physical_type": "inlet",
                "face_count": 100,
                "group": "inlets"
            },
            "outlet": {
                "type": "patch",
                "physical_type": "outlet",
                "face_count": 100,
                "group": "outlets"
            },
            "walls": {
                "type": "wall",
                "physical_type": "wall",
                "face_count": 6000,
                "group": "walls"
            },
            "symmetry": {
                "type": "symmetryPlane",
                "physical_type": "symmetry",
                "face_count": 50,
                "group": ""
            }
        }
        
        # Update boundaries widget
        self.boundary_widget.set_boundaries(boundaries)
    
    def set_mesh(self, mesh_file: str, mesh_data: Dict[str, Any] = None):
        """
        Set the current mesh.
        
        Args:
            mesh_file (str): Path to the mesh file
            mesh_data (Dict[str, Any], optional): Additional mesh data
        """
        # Store mesh file
        self.mesh_file = mesh_file
        
        # Determine mesh type
        if mesh_file.lower().endswith('.stl'):
            self.mesh_type = 'STL'
        elif mesh_file.lower().endswith(('.vtk', '.vtu')):
            self.mesh_type = 'VTK'
        elif mesh_file.lower().endswith('.foam') or os.path.isdir(mesh_file):
            self.mesh_type = 'OpenFOAM'
        else:
            self.mesh_type = 'Unknown'
        
        # Store mesh data
        if mesh_data:
            self.mesh_data = mesh_data
        else:
            self.mesh_data = {}
        
        # Load mesh statistics
        self._load_mesh_statistics()
        
        # Load mesh boundaries
        self._load_mesh_boundaries()
        
        # Update mesh generation settings
        self._load_mesh_generation_settings()
    
    def _load_mesh_statistics(self):
        """Load mesh statistics from the mesh file."""
        # In a real implementation, this would parse the mesh file
        # For now, we just load placeholder statistics
        self._load_placeholder_statistics()
    
    def _load_mesh_boundaries(self):
        """Load mesh boundaries from the mesh file."""
        # In a real implementation, this would parse the mesh file
        # For now, we just use the placeholder boundaries from _load_placeholder_statistics
        pass
    
    def _load_mesh_generation_settings(self):
        """Load mesh generation settings."""
        # In a real implementation, this would load settings from the mesh data
        # For now, we just use default settings
        
        # If mesh data contains blockMesh settings
        if "blockMesh" in self.mesh_data:
            self.block_mesh_widget.set_parameters(self.mesh_data["blockMesh"])
            self.mesh_type_combo.setCurrentIndex(0)  # Select blockMesh
        
        # If mesh data contains snappyHexMesh settings
        if "snappyHexMesh" in self.mesh_data:
            self.snappy_hex_mesh_widget.set_parameters(self.mesh_data["snappyHexMesh"])
            self.mesh_type_combo.setCurrentIndex(1)  # Select snappyHexMesh
    
    def get_mesh_properties(self) -> Dict[str, Any]:
        """
        Get current mesh properties.
        
        Returns:
            Dict[str, Any]: Dictionary of mesh properties
        """
        return self.mesh_data
    
    def clear(self):
        """Clear all mesh data."""
        # Clear mesh file and type
        self.mesh_file = None
        self.mesh_type = None
        self.mesh_data = {}
        
        # Clear statistics
        self.statistics_widget.clear()
        
        # Clear boundaries
        self.boundary_widget.clear()
        
        # Reset mesh generation settings
        # This would depend on the specific implementation of these widgets