#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ribbon menu for Openfoam_Simulator application.

This module implements a modern ribbon-style menu interface similar to those
found in applications like Microsoft Office. The ribbon is organized into
tabs containing categorized commands for the application.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

from PyQt5.QtWidgets import (
    QWidget, QTabWidget, QToolButton, QLabel, QFrame, QHBoxLayout,
    QVBoxLayout, QGridLayout, QSizePolicy, QComboBox, QAction,
    QMenu, QPushButton, QToolBar, QSpacerItem
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QEvent
from PyQt5.QtGui import QIcon, QPixmap, QColor, QPalette, QFont

# Import utility modules
from ..utils.logger import get_logger
from ..config import get_value

logger = get_logger(__name__)

# Constants
ICON_SIZE = QSize(32, 32)
SMALL_ICON_SIZE = QSize(16, 16)
BUTTON_SIZE = QSize(100, 80)  # Change from 80 to 100
SMALL_BUTTON_SIZE = QSize(90, 22)  # Change from 60 to 90

class RibbonTab(QWidget):
    """
    A tab in the ribbon menu containing groups of commands.
    """
    
    def __init__(self, name: str, parent=None):
        """
        Initialize the ribbon tab.
        
        Args:
            name (str): The name of the tab
            parent: The parent widget
        """
        super(RibbonTab, self).__init__(parent)
        self.name = name
        
        # Setup UI
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the UI for the ribbon tab."""
        # Create main layout
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.layout.setSpacing(5)
        
        # Style the tab
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#f0f0f0"))  # Light gray background
        self.setPalette(palette)
    
    def add_group(self, name: str) -> 'RibbonGroup':
        """
        Add a group to the ribbon tab.
        
        Args:
            name (str): The name of the group
            
        Returns:
            RibbonGroup: The created group
        """
        group = RibbonGroup(name, self)
        self.layout.addWidget(group)
        return group
    
    def add_spacer(self):
        """Add a spacer to the ribbon tab to push groups to the left."""
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.layout.addItem(spacer)


class RibbonGroup(QFrame):
    """
    A group of related commands in a ribbon tab.
    """
    
    def __init__(self, name: str, parent=None):
        """
        Initialize the ribbon group.
        
        Args:
            name (str): The name of the group
            parent: The parent widget
        """
        super(RibbonGroup, self).__init__(parent)
        self.name = name
        
        # Setup UI
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the UI for the ribbon group."""
        # Create main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(4, 2, 4, 2)
        self.main_layout.setSpacing(0)
        
        # Content layout for buttons
        self.content_layout = QGridLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(2)
        self.main_layout.addLayout(self.content_layout, 1)
        
        # Label for group name
        self.label = QLabel(self.name)
        self.label.setAlignment(Qt.AlignCenter)
        font = self.label.font()
        font.setPointSize(8)
        self.label.setFont(font)
        self.main_layout.addWidget(self.label)
        
        # Style the group
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#f8f8f8"))  # Slightly lighter gray
        self.setPalette(palette)
        
        # Add a frame to create a visual separation
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
    
    def add_large_button(self, row: int, col: int, icon: str, text: str, action: QAction = None, 
                        tooltip: str = None, menu: QMenu = None) -> QToolButton:
        """
        Add a large button to the group.
        
        Args:
            row (int): The row in the grid layout
            col (int): The column in the grid layout
            icon (str): The path to the icon or icon name
            text (str): The text to display
            action (QAction, optional): The action to connect to the button
            tooltip (str, optional): The tooltip to display
            menu (QMenu, optional): A menu to attach to the button
            
        Returns:
            QToolButton: The created button
        """
        button = QToolButton(self)
        button.setIcon(self._get_icon(icon))
        button.setIconSize(ICON_SIZE)
        button.setText(text)
        button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        button.setFixedSize(BUTTON_SIZE)
        
        if action:
            button.setDefaultAction(action)
        if tooltip:
            button.setToolTip(tooltip)
        if menu:
            button.setMenu(menu)
            button.setPopupMode(QToolButton.MenuButtonPopup)
        
        self.content_layout.addWidget(button, row, col, 2, 1)
        return button
    
    def add_small_button(self, row: int, col: int, icon: str, text: str, action: QAction = None, 
                         tooltip: str = None) -> QToolButton:
        """
        Add a small button to the group.
        
        Args:
            row (int): The row in the grid layout
            col (int): The column in the grid layout
            icon (str): The path to the icon or icon name
            text (str): The text to display
            action (QAction, optional): The action to connect to the button
            tooltip (str, optional): The tooltip to display
            
        Returns:
            QToolButton: The created button
        """
        button = QToolButton(self)
        button.setIcon(self._get_icon(icon))
        button.setIconSize(SMALL_ICON_SIZE)
        button.setText(text)
        button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        button.setFixedSize(SMALL_BUTTON_SIZE)
        
        if action:
            button.setDefaultAction(action)
        if tooltip:
            button.setToolTip(tooltip)
        
        self.content_layout.addWidget(button, row, col)
        return button
    
    def add_combo_box(self, row: int, col: int, items: List[str], 
                     current_index: int = 0, tooltip: str = None,
                     on_change: Callable = None) -> QComboBox:
        """
        Add a combo box to the group.
        
        Args:
            row (int): The row in the grid layout
            col (int): The column in the grid layout
            items (List[str]): The items to add to the combo box
            current_index (int, optional): The initial selected index
            tooltip (str, optional): The tooltip to display
            on_change (Callable, optional): Function to call when selection changes
            
        Returns:
            QComboBox: The created combo box
        """
        combo = QComboBox(self)
        combo.addItems(items)
        combo.setCurrentIndex(current_index)
        
        if tooltip:
            combo.setToolTip(tooltip)
        if on_change:
            combo.currentIndexChanged.connect(on_change)
        
        self.content_layout.addWidget(combo, row, col)
        return combo
    
    def add_separator(self, row: int, col: int) -> QFrame:
        """
        Add a vertical separator to the group.
        
        Args:
            row (int): The row in the grid layout
            col (int): The column in the grid layout
            
        Returns:
            QFrame: The created separator
        """
        separator = QFrame(self)
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        
        self.content_layout.addWidget(separator, row, col, 2, 1)
        return separator
    
    def _get_icon(self, icon: str) -> QIcon:
        """
        Get a QIcon from a path or name.
        
        Args:
            icon (str): The path to the icon or icon name
            
        Returns:
            QIcon: The icon
        """
        # Check if it's a path
        if os.path.exists(icon):
            return QIcon(icon)
        
        # Check in resources directory
        resources_dir = Path(__file__).parent.parent.parent / 'resources' / 'icons'
        icon_path = resources_dir / f"{icon}.png"
        
        if icon_path.exists():
            return QIcon(str(icon_path))
        
        # Try to load from system theme
        return QIcon.fromTheme(icon, QIcon())


class RibbonMenu(QTabWidget):
    """
    A ribbon menu containing tabs with groups of commands.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the ribbon menu.
        
        Args:
            parent: The parent widget (should be MainWindow)
        """
        super(RibbonMenu, self).__init__(parent)
        self.main_window = parent
        
        # Setup UI
        self._setup_ui()
        self._create_tabs()
    
    def _setup_ui(self):
        """Set up the UI for the ribbon menu."""
        # Set properties
        self.setTabPosition(QTabWidget.North)
        self.setDocumentMode(True)
        self.setContentsMargins(0, 0, 0, 0)
        
        # Set fixed height for the ribbon
        self.setFixedHeight(150)
        
        # Style the ribbon
        self.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background: #f0f0f0;
            }
            
            QTabBar::tab {
                background: #e0e0e0;
                border: 1px solid #c0c0c0;
                border-bottom: none;
                padding: 5px 12px;  /* Increase padding */
                margin-right: 2px;  /* Increase margin */
                min-width: 80px;    /* Add minimum width */
            }
            
            QTabBar::tab:selected {
                background: #f0f0f0;
                border-bottom: 1px solid #f0f0f0;
            }
            
            QTabBar::tab:hover {
                background: #f8f8f8;
            }
        """)
    
    def _create_tabs(self):
        """Create the tabs for the ribbon menu."""
        # Home tab
        self._create_home_tab()
        
        # Geometry tab
        self._create_geometry_tab()
        
        # Simulation tab
        self._create_simulation_tab()
        
        # Visualization tab
        self._create_visualization_tab()
        
        # Tools tab
        self._create_tools_tab()
        
        # Help tab
        self._create_help_tab()
    
    def _create_home_tab(self):
        """Create the Home tab."""
        tab = RibbonTab("Home")
        
        # File group
        file_group = tab.add_group("File")
        file_group.add_large_button(0, 0, "new", "New", self.main_window.action_new_project, 
                                   "Create a new project")
        file_group.add_large_button(0, 1, "open", "Open", self.main_window.action_open_project, 
                                   "Open an existing project")
        file_group.add_large_button(0, 2, "save", "Save", self.main_window.action_save_project, 
                                   "Save the current project")
        
        save_menu = QMenu()
        save_menu.addAction(self.main_window.action_save_project)
        save_menu.addAction(self.main_window.action_save_project_as)
        
        # Create recent files menu
        self.recent_menu = QMenu("Recent Files")
        self._update_recent_files_menu()
        save_menu.addMenu(self.recent_menu)
        
        file_group.add_large_button(0, 2, "save", "Save", action=None, menu=save_menu)
        
        # Import/Export group
        io_group = tab.add_group("Import/Export")
        io_group.add_small_button(0, 0, "mesh", "Import Mesh", self.main_window.action_import_mesh)
        io_group.add_small_button(1, 0, "cad", "Import CAD", self.main_window.action_import_cad)
        io_group.add_small_button(0, 1, "export", "Export Results", self.main_window.action_export_results)
        
        # View group
        view_group = tab.add_group("View")
        view_group.add_small_button(0, 0, "explorer", "Project Explorer", 
                                 self.main_window.action_toggle_project_explorer)
        view_group.add_small_button(1, 0, "properties", "Properties", 
                                 self.main_window.action_toggle_property_editor)
        view_group.add_small_button(0, 1, "simulation", "Simulation", 
                                 self.main_window.action_toggle_simulation_controls)
        view_group.add_small_button(1, 1, "visualization", "Visualization", 
                                 self.main_window.action_toggle_visualization_controls)
        
        # Add spacer to push everything to the left
        tab.add_spacer()
        
        self.addTab(tab, "Home")
    
    def _create_geometry_tab(self):
        """Create the Geometry tab."""
        tab = RibbonTab("Geometry")
        
        # Mesh group
        mesh_group = tab.add_group("Mesh")
        mesh_group.add_large_button(0, 0, "generate_mesh", "Generate\nMesh", 
                                  self.main_window.action_generate_mesh)
        
        mesh_operations = ["Edit Mesh", "Refine Mesh", "Smooth Mesh"]
        mesh_group.add_combo_box(0, 1, mesh_operations, tooltip="Mesh Operations")
        
        # CAD group
        cad_group = tab.add_group("CAD")
        cad_group.add_small_button(0, 0, "import_stl", "Import STL", 
                                tooltip="Import STL file")
        cad_group.add_small_button(1, 0, "import_step", "Import STEP", 
                                tooltip="Import STEP file")
        
        # Geometry tools group
        geom_tools_group = tab.add_group("Geometry Tools")
        geom_tools_group.add_small_button(0, 0, "boolean", "Boolean Op", 
                                      tooltip="Boolean operations")
        geom_tools_group.add_small_button(1, 0, "transform", "Transform", 
                                      tooltip="Transform geometry")
        geom_tools_group.add_small_button(0, 1, "measure", "Measure", 
                                      tooltip="Measure geometry")
        
        # Add spacer
        tab.add_spacer()
        
        self.addTab(tab, "Geometry")
    
    def _create_simulation_tab(self):
        """Create the Simulation tab."""
        tab = RibbonTab("Simulation")
        
        # Setup group
        setup_group = tab.add_group("Setup")
        setup_group.add_large_button(0, 0, "setup_case", "Setup\nCase", 
                                   self.main_window.action_setup_case)
        
        # Physics group
        physics_group = tab.add_group("Physics")
        flow_types = ["Single Phase", "Multi Phase", "Pigging", "Spill"]
        physics_group.add_combo_box(0, 0, flow_types, tooltip="Flow Type")
        
        turbulence_models = ["k-epsilon", "k-omega", "Spalart-Allmaras", "LES", "Laminar"]
        physics_group.add_combo_box(1, 0, turbulence_models, tooltip="Turbulence Model")
        
        # Run group
        run_group = tab.add_group("Run")
        run_group.add_large_button(0, 0, "run", "Run\nSimulation", 
                                 self.main_window.action_run_simulation)
        run_group.add_large_button(0, 1, "stop", "Stop\nSimulation", 
                                 self.main_window.action_stop_simulation)
        
        # Parameters group
        params_group = tab.add_group("Parameters")
        params_group.add_small_button(0, 0, "boundary", "Boundary\nConditions", 
                                    tooltip="Set boundary conditions")
        params_group.add_small_button(1, 0, "material", "Material\nProperties", 
                                    tooltip="Set material properties")
        
        # Find this line or something similar:
        boundary_button = params_group.add_small_button(0, 0, "boundary", "Boundary")
        
        # Add this line right after:
        boundary_button.clicked.connect(self.main_window.create_boundary_dialog)
        
        # Add spacer
        tab.add_spacer()
        
        self.addTab(tab, "Simulation")
    
    def _create_visualization_tab(self):
        """Create the Visualization tab."""
        tab = RibbonTab("Visualization")
        
        # View group
        view_group = tab.add_group("View")
        view_group.add_small_button(0, 0, "reset_view", "Reset\nView", 
                                  tooltip="Reset view to default")
        view_group.add_small_button(1, 0, "fit_to_screen", "Fit to\nScreen", 
                                  tooltip="Fit view to screen")
        
        # Representation group - Using VTK terminology
        rep_group = tab.add_group("Representation")
        rep_types = ["Surface", "Wireframe", "Points", "Volume"]
        rep_group.add_combo_box(0, 0, rep_types, tooltip="Representation Type")
        
        # Coloring group - Using VTK terminology
        color_group = tab.add_group("Coloring")
        color_vars = ["Pressure", "Velocity", "Temperature", "Phase", "Solid"]
        color_group.add_combo_box(0, 0, color_vars, tooltip="Color by")
        
        # Using VTK-compatible color maps
        color_maps = ["Rainbow", "Cool to Warm", "Viridis", "Plasma", "Jet"]
        color_group.add_combo_box(1, 0, color_maps, tooltip="Color map")
        
        # Filters group - VTK filters
        filters_group = tab.add_group("Filters")
        filters_group.add_small_button(0, 0, "slice", "Slice", 
                                    tooltip="Create a slice plane")
        filters_group.add_small_button(1, 0, "clip", "Clip", 
                                    tooltip="Clip the geometry")
        filters_group.add_small_button(0, 1, "contour", "Contour", 
                                    tooltip="Create contours")
        filters_group.add_small_button(1, 1, "streamlines", "Streamlines", 
                                    tooltip="Create streamlines")
        
        # Add spacer
        tab.add_spacer()
        
        self.addTab(tab, "Visualization")
    
    def _create_tools_tab(self):
        """Create the Tools tab."""
        tab = RibbonTab("Tools")
        
        # Analysis group
        analysis_group = tab.add_group("Analysis")
        analysis_group.add_small_button(0, 0, "calculator", "Calculator", 
                                     tooltip="Field calculator")
        analysis_group.add_small_button(1, 0, "integrate", "Integrate", 
                                     tooltip="Integrate field values")
        analysis_group.add_small_button(0, 1, "extract", "Extract", 
                                     tooltip="Extract data")
        
        # Industry group
        industry_group = tab.add_group("Industry")
        industry_group.add_small_button(0, 0, "oil_props", "Oil\nProperties", 
                                     tooltip="Oil properties database")
        industry_group.add_small_button(1, 0, "pigging", "Pigging\nAnalysis", 
                                     tooltip="Pigging analysis tools")
        industry_group.add_small_button(0, 1, "spill", "Spill\nModel", 
                                     tooltip="Spill modeling tools")
        
        # Reports group
        reports_group = tab.add_group("Reports")
        reports_group.add_small_button(0, 0, "generate_report", "Generate\nReport", 
                                    tooltip="Generate report")
        reports_group.add_small_button(1, 0, "export_report", "Export\nReport", 
                                    tooltip="Export report")
        
        # Settings group
        settings_group = tab.add_group("Settings")
        settings_group.add_small_button(0, 0, "preferences", "Preferences", 
                                      tooltip="Application preferences")
        settings_group.add_small_button(1, 0, "units", "Units", 
                                      tooltip="Change units system")
        
        # Add spacer
        tab.add_spacer()
        
        self.addTab(tab, "Tools")
    
    def _create_help_tab(self):
        """Create the Help tab."""
        tab = RibbonTab("Help")
        
        # Documentation group
        docs_group = tab.add_group("Documentation")
        docs_group.add_large_button(0, 0, "help", "Help", 
                                  self.main_window.action_help)
        docs_group.add_small_button(0, 1, "tutorials", "Tutorials", 
                                  tooltip="View tutorials")
        docs_group.add_small_button(1, 1, "examples", "Examples", 
                                  tooltip="View examples")
        
        # Support group
        support_group = tab.add_group("Support")
        support_group.add_small_button(0, 0, "bug_report", "Report\nBug", 
                                    tooltip="Report a bug")
        support_group.add_small_button(1, 0, "feature_request", "Feature\nRequest", 
                                    tooltip="Request a feature")
        
        # About group
        about_group = tab.add_group("About")
        about_group.add_large_button(0, 0, "about", "About", 
                                   self.main_window.action_about)
        
        # Add spacer
        tab.add_spacer()
        
        self.addTab(tab, "Help")
    
    def _update_recent_files_menu(self):
        """Update the recent files menu."""
        self.recent_menu.clear()
        
        # Get recent projects from config
        from ..config import get_value
        recent_projects = get_value('app.recent_projects', [])
        
        if not recent_projects:
            action = self.recent_menu.addAction("No recent files")
            action.setEnabled(False)
            return
        
        for project_path in recent_projects:
            action = self.recent_menu.addAction(os.path.basename(project_path))
            action.setData(project_path)
            action.triggered.connect(self._open_recent_project)
    
    def _open_recent_project(self):
        """Open a recent project."""
        action = self.sender()
        if action and action.data():
            project_path = action.data()
            if os.path.exists(project_path):
                # Forward to main window's open project function
                if hasattr(self.main_window, 'open_recent_project'):
                    self.main_window.open_recent_project(project_path)
                else:
                    # Fall back to standard open function
                    self.main_window.open_project(project_path)
            else:
                # File doesn't exist anymore
                from ..config import get_config_manager
                config_manager = get_config_manager()
                recent_projects = get_value('app.recent_projects', [])
                if project_path in recent_projects:
                    recent_projects.remove(project_path)
                    config_manager.set_value('app.recent_projects', recent_projects)
                    config_manager.save_config()
                
                self._update_recent_files_menu()
    
    def update_recent_files(self):
        """Update the recent files menu."""
        self._update_recent_files_menu()