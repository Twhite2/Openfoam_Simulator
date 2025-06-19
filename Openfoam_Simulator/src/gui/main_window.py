#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main window for Openfoam_Simulator application.

This module implements the main application window using PyQt5, integrating
the VTK viewport, ribbon menu, and various control panels.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from PyQt5.QtWidgets import (
    QMainWindow, QDockWidget, QAction, QMessageBox, QFileDialog,
    QApplication, QSplitter, QVBoxLayout, QHBoxLayout, QWidget,
    QToolBar, QStatusBar, QLabel, QFrame, QSizePolicy, QDialog, QProgressDialog,
    QRadioButton, QButtonGroup
)
from PyQt5.QtCore import Qt, QSize, QSettings, QTimer, pyqtSignal, QEvent
from PyQt5.QtGui import QIcon, QKeySequence, QCloseEvent, QResizeEvent
import json

# Import other Openfoam_Simulator modules
from ..config import load_config, get_value, set_value, save_config
from .ribbon_menu import RibbonMenu
from .viewport import VTKViewport
from .project_explorer import ProjectExplorer
from .property_editor import PropertyEditor
from .simulation_controls import SimulationControls
from .visualization_controls import VisualizationControls
from .boundary_dialog import BoundaryDialog
from .visualization_state_helpers import VisualizationStateHelpers

# Import utility modules
from ..utils.logger import get_logger
from ..models.project import Project

logger = get_logger(__name__)


class MainWindow(QMainWindow):
    """
    Main application window for Openfoam_Simulator.
    
    Integrates all UI components including the VTK viewport, ribbon menu,
    project explorer, and various control panels.
    """
    
    # Signals
    project_loaded = pyqtSignal(object)
    project_saved = pyqtSignal(object)
    view_changed = pyqtSignal()
    
    def __init__(self, config=None):
        """
        Initialize the main window.
        
        Args:
            config (dict, optional): Application configuration. If None, will load from config module.
        """
        super(MainWindow, self).__init__()
        
        # Load configuration if not provided
        self.config = config or load_config()
        
        # Initialize instance variables
        self.current_project = None
        self.modified = False
        self.autosave_timer = None
        
        # Initialize case manager (without a directory yet)
        from ..openfoam_integration.case_manager import CaseManager
        self.case_manager = CaseManager(main_window=self)
        
        # Set up the UI
        self._setup_actions()
        self._setup_ui()
        self._setup_menus()
        self._setup_dock_widgets()
        self._setup_status_bar()
        self._setup_autosave()
        self._restore_window_state()
        
        # Apply initial configuration
        self._apply_settings()
        
        # Show welcome screen if enabled
        if get_value('gui.show_welcome_screen', True):
            self._show_welcome_screen()
        
        # Connect SimulationControls signals
        self.simulation_controls.status_update.connect(self.update_status)
        self.simulation_controls.log_update.connect(self.update_log)
        self.simulation_controls.simulation_started.connect(self.on_simulation_started)
        self.simulation_controls.simulation_finished.connect(self.on_simulation_finished)
        
        logger.info("Main window initialized")
    
    def _setup_ui(self):
        """Set up the main window UI elements."""
        # Set window properties
        self.setWindowTitle(f"{get_value('app.name', 'Openfoam_Simulator')} - Oil & Gas CFD Visualization")
        self.setWindowIcon(QIcon(str(Path(__file__).parent.parent.parent / 'resources' / 'icons' / 'app_icon.png')))
        
        # Set window geometry from configuration
        window_size = get_value('gui.window_size', [1280, 800])
        window_position = get_value('gui.window_position', [100, 100])
        self.resize(window_size[0], window_size[1])
        self.move(window_position[0], window_position[1])

        # Add global stylesheet for better readability
        self.setStyleSheet("""
            QWidget {
                font-size: 10pt;
            }
            QToolButton {
                min-width: 80px;
                padding: 4px;
                margin: 2px;
            }
            QDockWidget::title {
                padding: 5px;
                background-color: #e0e0e0;
            }
            QTabBar::tab {
                min-width: 80px;
                padding: 8px;
            }
            QToolBar {
                spacing: 5px;
            }
            QPushButton {
                padding: 5px;
                min-width: 80px;
            }
            QComboBox {
                min-width: 150px;
                padding: 4px;
            }
            QFormLayout {
                spacing: 8px;
            }
        """)
        
        # Create central widget with layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create main layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Create ribbon menu (or traditional menu based on config)
        menu_style = get_value('gui.menu_style', 'ribbon')
        if menu_style == 'ribbon':
            self.ribbon_menu = RibbonMenu(self)
            self.main_layout.addWidget(self.ribbon_menu)
        
        # Create content layout (viewport and controls)
        self.content_layout = QHBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)
        self.main_layout.addLayout(self.content_layout, 1)
        
        # Create viewport
        self.viewport = VTKViewport(self)
        self.content_layout.addWidget(self.viewport, 1)
        self.viewport.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Enable drag and drop
        self.setAcceptDrops(True)
    
    def _setup_actions(self):
        """Set up the actions for the application."""
        # File actions
        self.action_new_project = QAction("New Project", self)
        self.action_new_project.setShortcut(QKeySequence.New)
        self.action_new_project.triggered.connect(self.new_project)
        
        self.action_open_project = QAction("Open Project", self)
        self.action_open_project.setShortcut(QKeySequence.Open)
        self.action_open_project.triggered.connect(self.open_project)
        
        self.action_save_project = QAction("Save Project", self)
        self.action_save_project.setShortcut(QKeySequence.Save)
        self.action_save_project.triggered.connect(self.save_project)
        
        self.action_save_project_as = QAction("Save Project As...", self)
        self.action_save_project_as.setShortcut(QKeySequence.SaveAs)
        self.action_save_project_as.triggered.connect(self.save_project_as)
        
        self.action_import_mesh = QAction("Import Mesh", self)
        self.action_import_mesh.triggered.connect(self.import_mesh)
        
        self.action_import_cad = QAction("Import CAD", self)
        self.action_import_cad.triggered.connect(self.import_cad)
        
        self.action_export_results = QAction("Export Results", self)
        self.action_export_results.triggered.connect(self.export_results)
        
        self.action_exit = QAction("Exit", self)
        self.action_exit.setShortcut(QKeySequence.Quit)
        self.action_exit.triggered.connect(self.close)
        
        # View actions
        self.action_toggle_project_explorer = QAction("Project Explorer", self)
        self.action_toggle_project_explorer.setCheckable(True)
        self.action_toggle_project_explorer.setChecked(True)
        
        self.action_toggle_property_editor = QAction("Property Editor", self)
        self.action_toggle_property_editor.setCheckable(True)
        self.action_toggle_property_editor.setChecked(True)
        
        self.action_toggle_simulation_controls = QAction("Simulation Controls", self)
        self.action_toggle_simulation_controls.setCheckable(True)
        self.action_toggle_simulation_controls.setChecked(True)
        
        self.action_toggle_visualization_controls = QAction("Visualization Controls", self)
        self.action_toggle_visualization_controls.setCheckable(True)
        self.action_toggle_visualization_controls.setChecked(True)
        
        # OpenFOAM actions
        self.action_generate_mesh = QAction("Generate Mesh", self)
        self.action_generate_mesh.triggered.connect(self.on_generate_mesh)
        
        self.action_setup_case = QAction("Setup Case", self)
        self.action_setup_case.triggered.connect(self.setup_case)

        self.boundary_button = QAction("Boundary")
        self.boundary_button.triggered.connect(self.create_boundary_dialog)
        
        self.action_run_simulation = QAction("Run Simulation", self)
        self.action_run_simulation.triggered.connect(self.run_simulation)
        
        self.action_stop_simulation = QAction("Stop Simulation", self)
        self.action_stop_simulation.triggered.connect(self.stop_simulation)
        self.action_stop_simulation.setEnabled(False)
        
        # Help actions
        self.action_about = QAction("About", self)
        self.action_about.triggered.connect(self.show_about_dialog)
        
        self.action_help = QAction("Help", self)
        self.action_help.setShortcut(QKeySequence.HelpContents)
        self.action_help.triggered.connect(self.show_help)
    
    def _setup_menus(self):
        """Set up the traditional menu bar (used if ribbon is disabled)."""
        menu_style = get_value('gui.menu_style', 'ribbon')
        
        if menu_style != 'ribbon':
            # File menu
            file_menu = self.menuBar().addMenu("&File")
            file_menu.addAction(self.action_new_project)
            file_menu.addAction(self.action_open_project)
            file_menu.addAction(self.action_save_project)
            file_menu.addAction(self.action_save_project_as)
            file_menu.addSeparator()
            file_menu.addAction(self.action_import_mesh)
            file_menu.addAction(self.action_import_cad)
            file_menu.addAction(self.action_export_results)
            file_menu.addSeparator()
            file_menu.addAction(self.action_exit)
            
            # View menu
            view_menu = self.menuBar().addMenu("&View")
            view_menu.addAction(self.action_toggle_project_explorer)
            view_menu.addAction(self.action_toggle_property_editor)
            view_menu.addAction(self.action_toggle_simulation_controls)
            view_menu.addAction(self.action_toggle_visualization_controls)
            
            # Simulation menu
            simulation_menu = self.menuBar().addMenu("&Simulation")
            simulation_menu.addAction(self.action_generate_mesh)
            simulation_menu.addAction(self.action_setup_case)
            simulation_menu.addAction(self.action_run_simulation)
            simulation_menu.addAction(self.action_stop_simulation)
            
            # Help menu
            help_menu = self.menuBar().addMenu("&Help")
            help_menu.addAction(self.action_help)
            help_menu.addAction(self.action_about)
    
    def _setup_dock_widgets(self):
        """Set up the dock widgets for various panels."""
        # Project Explorer dock
        self.project_explorer_dock = QDockWidget("Project Explorer", self)
        self.project_explorer_dock.setObjectName("project_explorer_dock")
        self.project_explorer = ProjectExplorer(self)
        self.project_explorer_dock.setWidget(self.project_explorer)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.project_explorer_dock)
        self.action_toggle_project_explorer.triggered.connect(self.project_explorer_dock.setVisible)
        self.project_explorer_dock.visibilityChanged.connect(
            lambda visible: self.action_toggle_project_explorer.setChecked(visible))
        # Add these lines after creating each dock widget
        self.project_explorer_dock.setMinimumWidth(250)
        
        # Property Editor dock
        self.property_editor_dock = QDockWidget("Property Editor", self)
        self.property_editor_dock.setObjectName("property_editor_dock")
        self.property_editor = PropertyEditor(self)
        self.property_editor_dock.setWidget(self.property_editor)
        self.addDockWidget(Qt.RightDockWidgetArea, self.property_editor_dock)
        self.action_toggle_property_editor.triggered.connect(self.property_editor_dock.setVisible)
        self.property_editor_dock.visibilityChanged.connect(
            lambda visible: self.action_toggle_property_editor.setChecked(visible))
        # Add these lines after creating each dock widget
        self.property_editor_dock.setMinimumWidth(300)
        
        # Simulation Controls dock
        self.simulation_controls_dock = QDockWidget("Simulation Controls", self)
        self.simulation_controls_dock.setObjectName("simulation_controls_dock")
        self.simulation_controls = SimulationControls(self)
        self.simulation_controls_dock.setWidget(self.simulation_controls)
        self.addDockWidget(Qt.RightDockWidgetArea, self.simulation_controls_dock)
        self.action_toggle_simulation_controls.triggered.connect(self.simulation_controls_dock.setVisible)
        self.simulation_controls_dock.visibilityChanged.connect(
            lambda visible: self.action_toggle_simulation_controls.setChecked(visible))
        # Add these lines after creating each dock widget
        self.simulation_controls_dock.setMinimumWidth(300)
        
        # Visualization Controls dock
        self.visualization_controls_dock = QDockWidget("Visualization Controls", self)
        self.visualization_controls_dock.setObjectName("visualization_controls_dock")
        self.visualization_controls = VisualizationControls(self)
        self.visualization_controls_dock.setWidget(self.visualization_controls)
        self.addDockWidget(Qt.RightDockWidgetArea, self.visualization_controls_dock)
        self.action_toggle_visualization_controls.triggered.connect(self.visualization_controls_dock.setVisible)
        self.visualization_controls_dock.visibilityChanged.connect(
            lambda visible: self.action_toggle_visualization_controls.setChecked(visible))
        # Add these lines after creating each dock widget
        self.visualization_controls_dock.setMinimumWidth(300)
        
        # Tab the right docks
        self.tabifyDockWidget(self.property_editor_dock, self.simulation_controls_dock)
        self.tabifyDockWidget(self.simulation_controls_dock, self.visualization_controls_dock)
        
        # Set dock widget sizes from configuration
        explorer_width = get_value('gui.explorer_width', 250)
        properties_width = get_value('gui.properties_width', 300)
        
        self.resizeDocks([self.project_explorer_dock], [explorer_width], Qt.Horizontal)
        self.resizeDocks([self.property_editor_dock], [properties_width], Qt.Horizontal)
    
    def _setup_status_bar(self):
        """Set up the status bar."""
        self.status_bar = self.statusBar()
        
        # Add status labels
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label, 1)
        
        # Add OpenFOAM version label
        openfoam_version = get_value('openfoam.version', 'Unknown')
        self.openfoam_version_label = QLabel(f"OpenFOAM {openfoam_version}")
        self.status_bar.addPermanentWidget(self.openfoam_version_label)
        
        # Add VTK version label
        vtk_version = get_value('vtk.version', 'Unknown')
        self.vtk_version_label = QLabel(f"VTK {vtk_version}")
        self.status_bar.addPermanentWidget(self.vtk_version_label)
        
        # Add simulation status indicator
        self.simulation_status_label = QLabel("No simulation running")
        self.status_bar.addPermanentWidget(self.simulation_status_label)
    
    def _setup_autosave(self):
        """Set up autosave functionality."""
        autosave_enabled = get_value('app.auto_save', True)
        if autosave_enabled:
            interval_minutes = get_value('app.autosave_interval_minutes', 5)
            self.autosave_timer = QTimer(self)
            self.autosave_timer.timeout.connect(self._autosave)
            self.autosave_timer.start(interval_minutes * 60 * 1000)  # Convert minutes to milliseconds
    
    def _restore_window_state(self):
        """Restore the window state from settings."""
        settings = QSettings("Openfoam_Simulator", "MainWindow")
        
        if settings.contains("geometry"):
            self.restoreGeometry(settings.value("geometry"))
        
        if settings.contains("windowState"):
            self.restoreState(settings.value("windowState"))
    
    def _apply_settings(self):
        """Apply settings from configuration."""
        # Set theme
        theme = get_value('app.theme', 'dark')
        if theme == 'dark':
            # Apply dark theme stylesheet (this would be loaded from a file in practice)
            pass
        
        # Apply VTK viewport settings
        if hasattr(self, 'viewport') and self.viewport:
            # Apply background color
            bg_color = get_value('vtk.viewport_settings.background_color', [0.2, 0.2, 0.2])
            self.viewport.set_background_color(bg_color)
            
            # Apply camera settings
            camera_position = get_value('vtk.viewport_settings.camera_position', [1.0, 1.0, 1.0])
            camera_focal_point = get_value('vtk.viewport_settings.camera_focal_point', [0.0, 0.0, 0.0])
            camera_view_up = get_value('vtk.viewport_settings.camera_view_up', [0.0, 0.0, 1.0])
            self.viewport.set_camera_position(camera_position, camera_focal_point, camera_view_up)
            
            # Apply renderer settings
            use_shadows = get_value('vtk.renderer_settings.use_shadows', False)
            use_depth_peeling = get_value('vtk.renderer_settings.use_depth_peeling', True)
            depth_peeling_layers = get_value('vtk.renderer_settings.depth_peeling_layers', 4)
            ambient_light = get_value('vtk.renderer_settings.ambient_light', 0.3)
            
            self.viewport.set_renderer_settings(
                use_shadows=use_shadows,
                use_depth_peeling=use_depth_peeling,
                depth_peeling_layers=depth_peeling_layers,
                ambient_light=ambient_light
            )
            
            # Apply axes visibility
            axes_visibility = get_value('vtk.axes_visibility', True)
            self.viewport.set_axes_visibility(axes_visibility)
    
    def _show_welcome_screen(self):
        """Show the welcome screen."""
        # This would be implemented to show a welcome dialog or splash screen
        logger.info("Welcome screen would be shown here")
    
    def _update_window_title(self):
        """Update the window title based on the current project."""
        app_name = get_value('app.name', 'Openfoam_Simulator')
        
        if self.current_project:
            project_name = os.path.basename(self.current_project.filepath) if self.current_project.filepath else "Untitled"
            modified_indicator = "*" if self.modified else ""
            self.setWindowTitle(f"{app_name} - {project_name}{modified_indicator}")
        else:
            self.setWindowTitle(f"{app_name} - Oil & Gas CFD Visualization")
    
    def _autosave(self):
        """Autosave the current project if modified."""
        if self.current_project and self.modified and self.current_project.filepath:
            self.save_project()
            logger.info(f"Autosaved project: {self.current_project.filepath}")
    
    def set_modified(self, modified=True):
        """
        Set the modified state of the current project.
        
        Args:
            modified (bool): Whether the project has been modified
        """
        self.modified = modified
        self._update_window_title()
    
    def confirm_save(self) -> bool:
        """
        Confirm with the user whether to save the current project if modified.
        
        Returns:
            bool: True if operation should continue, False if cancelled
        """
        if not self.modified:
            return True
        
        # Ask user if they want to save changes
        reply = QMessageBox.question(
            self, 
            "Save Changes",
            "The current project has unsaved changes. Would you like to save them?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
        )
        
        if reply == QMessageBox.Save:
            return self.save_project()
        elif reply == QMessageBox.Cancel:
            return False
        
        # Discard
        return True
    
    def new_project(self):
        """Create a new project."""
        if not self.confirm_save():
            return
        
        self.current_project = Project()
        self.set_modified(False)
        
        # Update UI components
        self.project_explorer.set_project(self.current_project)
        self.property_editor.clear()
        self.viewport.reset_view()
        
        # Don't create any temporary directories - wait until the user saves the project
        self.current_project.project_dir = None
        self.current_project.case_dir = None
        
        # Create placeholder for get_case_directory method
        def get_case_directory():
            if hasattr(self.current_project, 'case_dir') and self.current_project.case_dir:
                return self.current_project.case_dir
            return None
        
        self.current_project.get_case_directory = get_case_directory
        
        # Emit signal AFTER setup is complete
        self.project_loaded.emit(self.current_project)
        
        # Update simulation controls - disable buttons until project is saved
        if hasattr(self, 'simulation_controls'):
            if hasattr(self.simulation_controls, 'case_dir_edit'):
                self.simulation_controls.case_dir_edit.setText("Please save project first")
            
            # Disable simulation controls until project is saved
            if hasattr(self.simulation_controls, 'setup_button'):
                self.simulation_controls.setup_button.setEnabled(False)
        
        logger.info("New project created")
        self.status_label.setText("New project created - Please save the project to enable simulation controls")
    
    def open_project(self):
        """Open an existing project."""
        if not self.confirm_save():
            return
        
        # Show file dialog
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            "F3D Project Files (*.f3d);;All Files (*)"
        )
        
        if not filepath:
            return
        
        try:
            # Load project
            project = Project.load(filepath)
            self.current_project = project
            self.set_modified(False)
            
            # Check for existing meshes in the project's mesh folder
            self._detect_existing_meshes()
            
            # Update UI components
            self.project_explorer.set_project(project)
            self.property_editor.clear()
            self.viewport.load_project(project)
            
            # Load visualization state (boundary conditions, solver parameters, etc.)
            self._load_visualization_state()
            
            # Add to recent projects
            from ..config import get_config_manager
            get_config_manager().add_recent_project(filepath)
            
            # Emit signal
            self.project_loaded.emit(project)
            
            logger.info(f"Project opened: {filepath}")
            self.status_label.setText(f"Project opened: {os.path.basename(filepath)}")
            
        except Exception as e:
            logger.error(f"Error opening project: {e}")
            QMessageBox.critical(
                self,
                "Error Opening Project",
                f"An error occurred while opening the project:\n\n{str(e)}"
            )
    
    def _detect_existing_meshes(self):
        """Detect existing meshes in the project's mesh folder and set them as imported/active."""
        if not self.current_project or not self.current_project.project_dir:
            logger.info("No current project or project directory, skipping mesh detection")
            return
        
        # Initialize attributes if not present
        if not hasattr(self.current_project, 'imported_files'):
            self.current_project.imported_files = []
        
        if not hasattr(self.current_project, 'mesh_files'):
            self.current_project.mesh_files = []
        
        logger.info(f"Starting STL file detection for project: {self.current_project.project_dir}")
        
        # Search in multiple potential locations for STL files
        search_dirs = [
            Path(self.current_project.project_dir) / "mesh",
            Path(self.current_project.project_dir),
            Path(self.current_project.project_dir) / "case" / "openfoam" / "constant" / "triSurface",
            Path(self.current_project.project_dir) / "case" / "constant" / "triSurface"
        ]
        
        # Additionally, check if there's a case directory set in the simulation controls
        if hasattr(self, 'simulation_controls') and hasattr(self.simulation_controls, 'case_dir') and self.simulation_controls.case_dir:
            case_dir = Path(self.simulation_controls.case_dir)
            search_dirs.append(case_dir / "constant" / "triSurface")
        
        # Add more common locations where files might be
        if hasattr(self.current_project, 'get_case_directory') and self.current_project.get_case_directory():
            case_dir = Path(self.current_project.get_case_directory())
            search_dirs.append(case_dir / "constant" / "triSurface")
        
        # Also check for STL files directly in case directory
        if hasattr(self.current_project, 'get_case_directory') and self.current_project.get_case_directory():
            search_dirs.append(Path(self.current_project.get_case_directory()))
        
        # Log all search directories for debugging
        for directory in search_dirs:
            logger.info(f"Checking for STL files in: {directory}")
            if directory.exists() and directory.is_dir():
                logger.info(f"Directory exists: {directory}")
            else:
                logger.info(f"Directory does not exist: {directory}")
        
        stl_files = []
        for directory in search_dirs:
            if directory.exists() and directory.is_dir():
                found_files = list(directory.glob("*.stl")) + list(directory.glob("*.STL"))
                if found_files:
                    logger.info(f"Found {len(found_files)} STL files in {directory}: {[f.name for f in found_files]}")
                stl_files.extend(found_files)
        
        # If there are STL files, add them to imported_files and mesh_files
        for stl_file in stl_files:
            stl_path = str(stl_file)
            if stl_path not in self.current_project.imported_files:
                self.current_project.imported_files.append(stl_path)
                logger.info(f"Added to imported_files: {stl_path}")
            
            if stl_path not in self.current_project.mesh_files:
                self.current_project.mesh_files.append(stl_path)
                logger.info(f"Added to mesh_files: {stl_path}")
        
        # Set active mesh if not set and there are meshes available
        if (not hasattr(self.current_project, 'active_mesh') or not self.current_project.active_mesh) and self.current_project.imported_files:
            self.current_project.active_mesh = self.current_project.imported_files[0]
            logger.info(f"Set active mesh to: {self.current_project.active_mesh}")
        
        # Log current state after detection
        logger.info(f"After detection - imported_files: {self.current_project.imported_files}")
        logger.info(f"After detection - active_mesh: {getattr(self.current_project, 'active_mesh', None)}")
    
    def _load_visualization_state(self):
        """
        Load visualization state including boundary conditions, ambient settings, and solver parameters.
        
        This restores all visualization settings when reopening a project.
        """
        try:
            if not self.current_project or not self.current_project.project_dir:
                logger.warning("Cannot load visualization state: no project or project directory")
                return False
            
            # Import helper methods
            from .visualization_state_helpers import VisualizationStateHelpers
                
            # Check for visualization state file
            state_file = os.path.join(self.current_project.project_dir, "visualization", "visualization_state.json")
            if not os.path.exists(state_file):
                logger.info(f"No visualization state file found at {state_file}")
                return False
            
            # Load state from file
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # 1. Restore viewport camera settings
            if 'camera' in state and hasattr(self, 'viewport'):
                VisualizationStateHelpers.restore_camera_settings(self.viewport, state['camera'])
                
                # Also restore pipeline model properties (colors, opacity, etc.)
                if 'model_properties' in state:
                    logger.info("Restoring pipeline model properties...")
                    VisualizationStateHelpers.restore_pipeline_model_properties(self.viewport, state['model_properties'])
            
            # 2. Restore boundary conditions
            if 'boundary_conditions' in state and hasattr(self, 'simulation_controls'):
                logger.info("Restoring boundary conditions...")
                VisualizationStateHelpers.restore_boundary_conditions(self.simulation_controls, state['boundary_conditions'])
            
            # 3. Restore solver parameters
            if 'solver_parameters' in state and hasattr(self, 'simulation_controls'):
                logger.info("Restoring solver parameters...")
                VisualizationStateHelpers.restore_solver_parameters(self.simulation_controls, state['solver_parameters'])
            
            # 4. Restore ambient region settings
            if 'ambient_settings' in state and hasattr(self, 'simulation_controls'):
                logger.info("Restoring ambient region settings...")
                VisualizationStateHelpers.restore_ambient_settings(self.simulation_controls, state['ambient_settings'])
            
            logger.info(f"Visualization state loaded from {state_file}")
            
            # Force a full render refresh
            if hasattr(self, 'viewport') and hasattr(self.viewport, 'render_window'):
                self.viewport.render_window.Render()
                
            return True
            
        except Exception as e:
            logger.error(f"Error loading visualization state: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def save_project(self) -> bool:
        """
        Save the current project.
        
        Returns:
            bool: True if project was saved, False otherwise
        """
        if not self.current_project:
            return False
        
        # If project has no filepath yet, prompt for one
        if not self.current_project.filepath:
            return self.save_project_as()
        
        try:
            # Save visualization state before saving project
            self._save_visualization_state()
            
            # Save project
            self.current_project.save()
            self.set_modified(False)
            
            # Check if case directory is set after saving
            if hasattr(self.current_project, 'case_dir') and self.current_project.case_dir:
                # Initialize case manager if needed
                from ..openfoam_integration.case_manager import create_case_manager
                if not hasattr(self, 'case_manager'):
                    self.case_manager = create_case_manager(self.current_project.case_dir)
                else:
                    self.case_manager.set_case_directory(self.current_project.case_dir)
                
                # Update simulation controls
                if hasattr(self, 'simulation_controls'):
                    self.simulation_controls.case_dir = self.current_project.case_dir
                    if hasattr(self.simulation_controls, 'case_dir_edit'):
                        self.simulation_controls.case_dir_edit.setText(self.current_project.case_dir)
                    # Enable buttons
                    if hasattr(self.simulation_controls, 'setup_button'):
                        self.simulation_controls.setup_button.setEnabled(True)
                    if hasattr(self.simulation_controls, 'run_button'):
                        self.simulation_controls.run_button.setEnabled(True)
                    
                    # Create case manager in simulation controls if needed
                    from ..openfoam_integration.case_manager import create_case_manager
                    if not hasattr(self.simulation_controls, 'case_manager'):
                        self.simulation_controls.case_manager = create_case_manager(self.current_project.case_dir)
                    elif hasattr(self.simulation_controls, 'case_manager'):
                        self.simulation_controls.case_manager.set_case_directory(self.current_project.case_dir)
            
            # Emit signal
            self.project_saved.emit(self.current_project)
            
            logger.info(f"Project saved: {self.current_project.filepath}")
            self.status_label.setText(f"Project saved: {os.path.basename(self.current_project.filepath)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving project: {e}")
            QMessageBox.critical(
                self,
                "Error Saving Project",
                f"An error occurred while saving the project:\n\n{str(e)}"
            )
            return False
    
    def save_project_as(self) -> bool:
        """
        Save the current project with a new filename.
        
        Returns:
            bool: True if project was saved, False otherwise
        """
        if not self.current_project:
            return False
        
        # Show file dialog
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            "",
            "F3D Project Files (*.f3d);;All Files (*)"
        )
        
        if not filepath:
            return False
        
        # Add .f3d extension if not present
        if not filepath.lower().endswith('.f3d'):
            filepath += '.f3d'
        
        # Set filepath and save
        self.current_project.filepath = filepath
        return self.save_project()
    
    def import_mesh(self):
        """Import a mesh file."""
        if not self.current_project:
            QMessageBox.warning(
                self,
                "No Project",
                "Please create or open a project first."
            )
            return
        
        # Show file dialog
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Import Mesh",
            "",
            "Mesh Files (*.stl *.obj *.vtk *.vtu);;All Files (*)"
        )
        
        if not filepath:
            return
        
        try:
            # Import mesh
            logger.info(f"Importing mesh file: {filepath}")
            self.status_label.setText(f"Importing mesh: {os.path.basename(filepath)}")
            
            # Import the mesh to the project
            success = self.current_project.import_mesh(filepath)
            
            if success:
                self.set_modified(True)
                
                # Update UI
                self.project_explorer.refresh()
                
                # Get the actual mesh object from the project
                mesh = self.current_project.get_active_mesh()
                
                # Load the mesh in the viewport directly from the file path
                logger.info(f"Loading mesh into viewport: {filepath}")
                self.viewport.load_mesh(filepath)
                
                self.status_label.setText(f"Mesh imported: {os.path.basename(filepath)}")
            else:
                logger.error("Failed to import mesh to project")
                
        except Exception as e:
            logger.error(f"Error importing mesh: {e}")
            QMessageBox.critical(
                self,
                "Error Importing Mesh",
                f"An error occurred while importing the mesh:\n\n{str(e)}"
            )
    
    def import_cad(self):
        """Import a CAD file."""
        if not self.current_project:
            self.new_project()
        
        # Show file dialog
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Import CAD",
            "",
            "STL Files (*.stl);;STEP Files (*.step *.stp);;OBJ Files (*.obj);;VTK Files (*.vtk);;All Files (*)"
        )
        
        if not filepath:
            return
        
        try:
            # Import CAD (implementation would depend on specific module)
            logger.info(f"Importing CAD: {filepath}")
            self.status_label.setText(f"Importing CAD: {os.path.basename(filepath)}")
            
            # This is a placeholder - actual implementation would be in the appropriate module
            self.current_project.import_cad(filepath)
            self.set_modified(True)
            
            # Update UI
            self.project_explorer.refresh()
            self.viewport.update_view()
            
        except Exception as e:
            logger.error(f"Error importing CAD: {e}")
            QMessageBox.critical(
                self,
                "Error Importing CAD",
                f"An error occurred while importing the CAD file:\n\n{str(e)}"
            )
    
    def export_results(self):
        """Export simulation results."""
        if not self.current_project or not self.current_project.has_results():
            QMessageBox.warning(
                self,
                "No Results",
                "There are no simulation results to export."
            )
            return
        
        # Show file dialog
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "",
            "VTK Files (*.vtk);;CSV Files (*.csv);;All Files (*)"
        )
        
        if not filepath:
            return
        
        try:
            # Export results (implementation would depend on specific module)
            logger.info(f"Exporting results: {filepath}")
            self.status_label.setText(f"Exporting results: {os.path.basename(filepath)}")
            
            # This is a placeholder - actual implementation would be in the appropriate module
            self.current_project.export_results(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            QMessageBox.critical(
                self,
                "Error Exporting Results",
                f"An error occurred while exporting the results:\n\n{str(e)}"
            )
    
    def on_generate_mesh(self):
        """Handle Generate Mesh button click."""
        from PyQt5.QtWidgets import QProgressDialog, QMessageBox, QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QRadioButton, QButtonGroup
        from PyQt5.QtCore import Qt
        from ..openfoam_integration.mesh_generator import MeshGenerator

        if not self.current_project:
            QMessageBox.warning(self, "Warning", "Please create a project first.")
            return

        try:
            # Force mesh detection every time to ensure latest STL files are found
            self._detect_existing_meshes()
            
            # If there are still no meshes, check if there might be an STL visible in the viewport
            # and add it to our imported files
            if (not hasattr(self.current_project, 'imported_files') or not self.current_project.imported_files) and hasattr(self, 'viewport'):
                # Try to get mesh from viewport if it exists
                if hasattr(self.viewport, 'current_mesh_path') and self.viewport.current_mesh_path:
                    stl_path = self.viewport.current_mesh_path
                    if os.path.exists(stl_path) and stl_path.lower().endswith('.stl'):
                        if not hasattr(self.current_project, 'imported_files'):
                            self.current_project.imported_files = []
                        self.current_project.imported_files.append(stl_path)
                        logger.info(f"Added viewport mesh to imported files: {stl_path}")
            
            # Last resort - look for any STL in the project directory
            if not hasattr(self.current_project, 'imported_files') or not self.current_project.imported_files:
                # Look for any STL files in the project directory
                project_dir = Path(self.current_project.project_dir)
                stl_files = list(project_dir.glob("**/*.stl")) + list(project_dir.glob("**/*.STL"))
                
                if stl_files:
                    logger.info(f"Found STL files in project directory: {[f.name for f in stl_files]}")
                    if not hasattr(self.current_project, 'imported_files'):
                        self.current_project.imported_files = []
                    
                    for stl_file in stl_files:
                        stl_path = str(stl_file)
                        if stl_path not in self.current_project.imported_files:
                            self.current_project.imported_files.append(stl_path)
                            logger.info(f"Added project STL to imported files: {stl_path}")
            
            # If there are still no meshes, ask the user to import one
            if not hasattr(self.current_project, 'imported_files') or not self.current_project.imported_files:
                # Show a more detailed error message
                error_msg = "No STL files were found for this project. Please import an STL file first.\n\n"
                error_msg += "Project directory: " + str(self.current_project.project_dir)
                
                if hasattr(self.current_project, 'get_case_directory') and self.current_project.get_case_directory():
                    error_msg += "\nCase directory: " + str(self.current_project.get_case_directory())
                
                QMessageBox.warning(self, "Warning", error_msg)
                return
            
            # Set the imported STL as active geometry if not already set
            if not hasattr(self.current_project, 'active_mesh') or not self.current_project.active_mesh:
                self.current_project.active_mesh = self.current_project.imported_files[0]
                logger.info(f"Set active geometry to: {self.current_project.active_mesh}")
            
            # Get STL file path
            stl_file_path = self.current_project.active_mesh
            
            # Verify that the STL file exists
            if not os.path.exists(stl_file_path):
                logger.error(f"STL file not found: {stl_file_path}")
                QMessageBox.warning(self, "Warning", f"STL file not found: {stl_file_path}\nPlease import a valid STL file.")
                return
            
            # Create mesh strategy selection dialog
            class MeshStrategyDialog(QDialog):
                def __init__(self, parent=None):
                    super().__init__(parent)
                    self.setWindowTitle("Select Mesh Generation Strategy")
                    self.setMinimumWidth(400)
                    
                    layout = QVBoxLayout()
                    
                    # Add description
                    layout.addWidget(QLabel("Select mesh generation strategy for your geometry:"))
                    
                    # Add radio buttons
                    self.auto_detect = QRadioButton("Auto-detect (recommended)")
                    self.block_mesh = QRadioButton("blockMesh - for simple pipe geometries")
                    self.snappy_hex_mesh = QRadioButton("snappyHexMesh - for complex geometries")
                    
                    # Group radio buttons
                    self.button_group = QButtonGroup()
                    self.button_group.addButton(self.auto_detect)
                    self.button_group.addButton(self.block_mesh)
                    self.button_group.addButton(self.snappy_hex_mesh)
                    
                    # Set default
                    self.auto_detect.setChecked(True)
                    
                    # Add to layout
                    layout.addWidget(self.auto_detect)
                    layout.addWidget(self.block_mesh)
                    layout.addWidget(self.snappy_hex_mesh)
                    
                    # Add description for current STL
                    layout.addWidget(QLabel(f"Current STL: {Path(stl_file_path).name}"))
                    
                    # Add buttons
                    button_layout = QHBoxLayout()
                    self.ok_button = QPushButton("OK")
                    self.cancel_button = QPushButton("Cancel")
                    
                    self.ok_button.clicked.connect(self.accept)
                    self.cancel_button.clicked.connect(self.reject)
                    
                    button_layout.addWidget(self.ok_button)
                    button_layout.addWidget(self.cancel_button)
                    
                    layout.addLayout(button_layout)
                    
                    self.setLayout(layout)
                    
                def get_strategy(self):
                    if self.auto_detect.isChecked():
                        return "auto"
                    elif self.block_mesh.isChecked():
                        return "blockMesh"
                    elif self.snappy_hex_mesh.isChecked():
                        return "snappyHexMesh"
                    else:
                        return "auto"
            
            # Show dialog
            dialog = MeshStrategyDialog(self)
            if dialog.exec_() != QDialog.Accepted:
                return
            
            # Get selected strategy
            strategy = dialog.get_strategy()
            
            # Get case directory
            if not self.current_project.project_dir:
                QMessageBox.warning(self, "Warning", "Project directory not set.")
                return
            
            case_dir = self.current_project.get_case_directory()
            if not case_dir:
                # Create case directory inside project directory
                case_dir = os.path.join(self.current_project.project_dir, "case")
                os.makedirs(case_dir, exist_ok=True)
                self.current_project.set_case_directory(case_dir)
            
            logger.info(f"Using case directory: {case_dir}")
            
            # Create mesh generator instance
            mesh_generator = MeshGenerator(case_dir)

            # Show progress dialog
            progress = QProgressDialog("Generating mesh...", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            # Generate the mesh based on strategy
            if strategy == "auto":
                success = mesh_generator.generate_mesh_from_stl(stl_file_path, auto_detect=True)
            elif strategy == "blockMesh":
                params = mesh_generator._analyze_stl_for_blockmesh(stl_file_path)
                success = mesh_generator.create_block_mesh("pipeline", params)
            elif strategy == "snappyHexMesh":
                success = mesh_generator.generate_mesh_from_stl(stl_file_path, auto_detect=False)
            else:
                success = False
                logger.error(f"Unknown mesh strategy: {strategy}")

            progress.close()

            if success:
                QMessageBox.information(self, "Success", "Mesh generation completed successfully!")
                # Update the viewport to show the generated mesh
                self.viewport.load_openfoam_mesh(case_dir)
                # Update project explorer
                self.project_explorer.refresh()
            else:
                QMessageBox.critical(self, "Error", "Failed to generate mesh. Check the log for details.")

        except Exception as e:
            logger.error(f"Error generating mesh: {e}")
            QMessageBox.critical(self, "Error", f"Error generating mesh: {str(e)}")
    
    def setup_case(self):
        """Set up an OpenFOAM case for the current project."""
        if not self.current_project:
            QMessageBox.warning(
                self,
                "No Project",
                "Please create or open a project first."
            )
            return
        
        try:
            # Setup case (implementation would depend on specific module)
            logger.info("Setting up case")
            self.status_label.setText("Setting up case...")
            
            # This is a placeholder - actual implementation would be in the appropriate module
            self.current_project.setup_case()
            self.set_modified(True)
            
            # Update UI
            self.project_explorer.refresh()
            
            self.status_label.setText("Case setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up case: {e}")
            QMessageBox.critical(
                self,
                "Error Setting Up Case",
                f"An error occurred while setting up the case:\n\n{str(e)}"
            )
    
    def run_simulation(self):
        """Run the simulation for the current project."""
        if not self.current_project:
            QMessageBox.warning(
                self,
                "No Project",
                "Please create or open a project first."
            )
            return
        
        try:
            # Run simulation (implementation would depend on specific module)
            logger.info("Running simulation")
            self.status_label.setText("Running simulation...")
            self.simulation_status_label.setText("Simulation running")
            
            # This is a placeholder - actual implementation would be in the appropriate module
            self.current_project.run_simulation()
            self.set_modified(True)
            
            # Update UI
            self.action_run_simulation.setEnabled(False)
            self.action_stop_simulation.setEnabled(True)
            
        except Exception as e:
            logger.error(f"Error running simulation: {e}")
            QMessageBox.critical(
                self,
                "Error Running Simulation",
                f"An error occurred while running the simulation:\n\n{str(e)}"
            )
    
    def stop_simulation(self):
        """Stop the running simulation."""
        try:
            # Stop simulation (implementation would depend on specific module)
            logger.info("Stopping simulation")
            self.status_label.setText("Stopping simulation...")
            
            # This is a placeholder - actual implementation would be in the appropriate module
            self.current_project.stop_simulation()
            
            # Update UI
            self.action_run_simulation.setEnabled(True)
            self.action_stop_simulation.setEnabled(False)
            self.simulation_status_label.setText("Simulation stopped")
            self.status_label.setText("Simulation stopped")
            
        except Exception as e:
            logger.error(f"Error stopping simulation: {e}")
            QMessageBox.critical(
                self,
                "Error Stopping Simulation",
                f"An error occurred while stopping the simulation:\n\n{str(e)}"
            )
    
    def simulation_finished(self, success):
        """
        Handle simulation completion.
        
        Args:
            success (bool): Whether the simulation completed successfully
        """
        # Update UI
        self.action_run_simulation.setEnabled(True)
        self.action_stop_simulation.setEnabled(False)
        
        if success:
            self.simulation_status_label.setText("Simulation completed")
            self.status_label.setText("Simulation completed successfully")
            
            # Load results
            self.viewport.load_results(self.current_project)
        else:
            self.simulation_status_label.setText("Simulation failed")
            self.status_label.setText("Simulation failed")

    def create_boundary_dialog(self):
        """Create dialog for setting boundary conditions"""
        try:
            # Instead of calling show_simulation_controls(), use the simulation_controls attribute
            if hasattr(self, 'simulation_controls') and self.simulation_controls:
                # Show simulation controls panel
                self.simulation_controls.setVisible(True)
                # Switch to boundary tab
                if hasattr(self.simulation_controls, 'tab_widget'):
                    tab_index = self.simulation_controls.tab_widget.indexOf(self.simulation_controls.boundary_tab)
                    if tab_index >= 0:
                        self.simulation_controls.tab_widget.setCurrentIndex(tab_index)
            else:
                # Create new simulation controls if not exist
                self.setup_simulation_controls()
                # Make sure it's visible and on the boundary tab
                self.simulation_controls.setVisible(True)
                if hasattr(self.simulation_controls, 'tab_widget'):
                    tab_index = self.simulation_controls.tab_widget.indexOf(self.simulation_controls.boundary_tab)
                    if tab_index >= 0:
                        self.simulation_controls.tab_widget.setCurrentIndex(tab_index)
        except Exception as e:
            logger.error(f"Error showing boundary dialog: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def show_about_dialog(self):
        """Show the about dialog."""
        app_name = get_value('app.name', 'Openfoam_Simulator')
        app_version = get_value('app.version', '0.1.0')
        vtk_version = get_value('vtk.version', 'Unknown')
        
        QMessageBox.about(
            self,
            f"About {app_name}",
            f"<h2>{app_name} v{app_version}</h2>"
            "<p>Oil & Gas CFD Visualization System</p>"
            f"<p>Built with PyQt and VTK {vtk_version}</p>"
            "<p>&copy; 2025 Openfoam_Simulator Team</p>"
        )
    
    def show_help(self):
        """Show the help documentation."""
        # This would open the help documentation, either locally or online
        logger.info("Help documentation would be shown here")
    
    def check_environment(self):
        """Check if all required environment variables and dependencies are available"""
        # Check OpenFOAM environment
        if "WM_PROJECT_DIR" not in os.environ:
            logger.warning("OpenFOAM environment not detected. Some features may not work correctly.")
        
        # Check VTK
        try:
            import vtk
            vtk_version = vtk.vtkVersion.GetVTKVersion()
            logger.info(f"VTK version: {vtk_version}")
            
            # Check for OpenGL rendering support
            try:
                vtkOpenGLRW = vtk.vtkRenderWindow()
                if hasattr(vtkOpenGLRW, "ReportCapabilities"):
                    logger.info("VTK OpenGL capabilities available")
                else:
                    logger.warning("VTK may not have full OpenGL support")
                
                # Check if we have advanced rendering features
                renderer = vtk.vtkRenderer()
                if hasattr(renderer, "SetUseShadows"):
                    logger.info("VTK shadow rendering is available")
                if hasattr(renderer, "SetUseDepthPeeling"):
                    logger.info("VTK depth peeling is available")
                
            except Exception as e:
                logger.warning(f"Error checking VTK rendering capabilities: {e}")
                
        except ImportError:
            logger.error("VTK not found. Visualization features will not be available.")
        except Exception as e:
            logger.error(f"Error checking VTK: {e}")
    
    def closeEvent(self, event):
        """
        Handle window close event.
        
        Args:
            event: Close event
        """
        # Save window state
        settings = QSettings("Openfoam_Simulator", "MainWindow")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        
        # Save window position and size to config
        set_value('gui.window_size', [self.width(), self.height()])
        set_value('gui.window_position', [self.x(), self.y()])
        
        save_config()
        
        # Check for unsaved changes
        if self.modified:
            if not self.confirm_save():
                event.ignore()
                return
        
        # Clean up resources
        if hasattr(self, 'autosave_timer') and self.autosave_timer:
            self.autosave_timer.stop()
        
        self.viewport.cleanup()
        
        # Accept the close event
        event.accept()
    
    def resizeEvent(self, event: QResizeEvent):
        """
        Handle the window resize event.
        
        Args:
            event (QResizeEvent): The resize event
        """
        super().resizeEvent(event)
        
        # Update viewport size if needed
        if hasattr(self, 'viewport') and self.viewport:
            self.viewport.update_size()
    
    def dragEnterEvent(self, event: QEvent):
        """
        Handle drag enter event for drag and drop.
        
        Args:
            event (QEvent): The drag enter event
        """
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QEvent):
        """
        Handle drop event for drag and drop.
        
        Args:
            event (QEvent): The drop event
        """
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            filepath = url.toLocalFile()
            
            # Check file extension to decide what to do
            if filepath.lower().endswith('.f3d'):
                # Open project
                if self.confirm_save():
                    try:
                        project = Project.load(filepath)
                        self.current_project = project
                        self.set_modified(False)
                        
                        # Update UI components
                        self.project_explorer.set_project(project)
                        self.property_editor.clear()
                        self.viewport.load_project(project)
                        
                        # Emit signal
                        self.project_loaded.emit(project)
                        
                        logger.info(f"Project opened (drag and drop): {filepath}")
                    except Exception as e:
                        logger.error(f"Error opening project (drag and drop): {e}")
                        QMessageBox.critical(
                            self,
                            "Error Opening Project",
                            f"An error occurred while opening the project:\n\n{str(e)}"
                        )
            elif filepath.lower().endswith(('.stl', '.obj', '.step', '.stp', '.vtk', '.vtu')):
                # Import mesh/CAD
                if not self.current_project:
                    self.new_project()
                
                try:
                    logger.info(f"Importing file (drag and drop): {filepath}")
                    
                    # Detect file type and import accordingly
                    if filepath.lower().endswith(('.stl', '.obj', '.vtk', '.vtu')):
                        # Don't use the success variable to call load_mesh
                        success = self.current_project.import_mesh(filepath)
                        if success:
                            # Directly use the original filepath 
                            self.viewport.load_mesh(filepath)
                    else:
                        self.current_project.import_cad(filepath)
                    
                    self.set_modified(True)
                    
                    # Update UI
                    self.project_explorer.refresh()
                    
                except Exception as e:
                    logger.error(f"Error importing file (drag and drop): {e}")
                    QMessageBox.critical(
                        self,
                        "Error Importing File",
                        f"An error occurred while importing the file:\n\n{str(e)}"
                    )

    def setup_boundary_conditions(self):
        """
        Set up boundary conditions for the current simulation.
        """
        try:
            # Check if we have a valid project
            if not hasattr(self, 'current_project') or not self.current_project:
                QMessageBox.warning(self, "Error", "No active project found.")
                return
            
            # Enable boundary selection mode in viewport
            if hasattr(self, 'viewport') and self.viewport:
                self.viewport.enable_boundary_selection()
                
                # Reset any previous selections
                self.boundary_selections = {}
                
                # Show instructions
                QMessageBox.information(
                    self,
                    "Boundary Selection",
                    "Hover over mesh faces and click to select boundaries.\n\n"
                    "- Inlet faces will be colored blue\n"
                    "- Outlet faces will be colored red\n"
                    "- Wall faces will be colored gray\n\n"
                    "Click 'Apply Boundaries' in the toolbar when finished."
                )
                
                # Add a button to finalize and apply the boundary conditions
                if not hasattr(self, 'apply_boundaries_action'):
                    self.apply_boundaries_action = QAction("Apply Boundaries", self)
                    self.apply_boundaries_action.triggered.connect(self.apply_boundary_selections)
                    self.toolbar.addAction(self.apply_boundaries_action)
                    
                # Make the apply button visible
                self.apply_boundaries_action.setVisible(True)
            else:
                QMessageBox.warning(self, "Error", "Viewport not available.")
                
        except Exception as e:
            logging.error(f"Error in setup_boundary_conditions: {e}")
            QMessageBox.critical(self, "Error", f"Failed to set up boundary conditions: {str(e)}")

    def apply_boundary_selections(self):
        """Apply the selected boundary conditions"""
        try:
            if not hasattr(self, 'boundary_selections') or not self.boundary_selections:
                QMessageBox.warning(self, "Warning", "No boundaries have been selected.")
                return
            
            # Disable selection mode
            if hasattr(self, 'viewport') and self.viewport:
                self.viewport.disable_boundary_selection()
            
            # Hide apply button
            if hasattr(self, 'apply_boundaries_action'):
                self.apply_boundaries_action.setVisible(False)
            
            # Get case directory
            case_dir = None
            if hasattr(self.current_project, 'get_case_directory'):
                case_dir = self.current_project.get_case_directory()
            elif hasattr(self.current_project, 'case_dir'):
                case_dir = self.current_project.case_dir
            
            if not case_dir or not os.path.exists(case_dir):
                QMessageBox.warning(self, "Error", "No valid case directory found.")
                return
            
            # Create case manager
            from ..openfoam_integration.case_manager import CaseManager
            case_manager = CaseManager(case_dir)
            
            # Convert selections to boundary config format
            bc_config = {}
            for boundary_type, cell_ids in self.boundary_selections.items():
                if not cell_ids:
                    continue
                
                # Create a boundary name based on type
                boundary_name = f"{boundary_type}_{len(cell_ids)}_cells"
                
                bc_config[boundary_name] = {
                    "type": boundary_type,
                    "cells": cell_ids
                }
                
                # Add default values based on type
                if boundary_type == "inlet":
                    bc_config[boundary_name]["velocity"] = [1.0, 0.0, 0.0]  # Default x-direction
                    bc_config[boundary_name]["velocity_type"] = "fixedValue"
                    bc_config[boundary_name]["pressure_type"] = "zeroGradient"
                elif boundary_type == "outlet":
                    bc_config[boundary_name]["pressure"] = 0.0
                    bc_config[boundary_name]["pressure_type"] = "fixedValue"
                    bc_config[boundary_name]["velocity_type"] = "zeroGradient"
                elif boundary_type == "wall":
                    bc_config[boundary_name]["velocity_type"] = "noSlip"
                
            # Apply boundary conditions
            success = case_manager.setup_boundary_conditions(bc_config)
            
            if success:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Boundary conditions applied successfully for {len(bc_config)} boundaries."
                )
            else:
                QMessageBox.warning(self, "Error", "Failed to apply boundary conditions.")
            
        except Exception as e:
            logging.error(f"Error applying boundary selections: {e}")
            QMessageBox.critical(self, "Error", f"Failed to apply boundary conditions: {str(e)}")

    def boundary_selected(self, cell_id, boundary_type):
        """
        Handle boundary selection from viewport.
        
        Args:
            cell_id: ID of the selected cell
            boundary_type: Type of boundary (inlet, outlet, wall)
        """
        try:
            logging.info(f"Boundary selected: cell {cell_id} as {boundary_type}")
            
            # Update status
            self.status_label.setText(f"Selected {boundary_type} at cell {cell_id}")
            
            # Store boundary information
            if not hasattr(self, 'boundary_selections'):
                self.boundary_selections = {}
            
            if boundary_type not in self.boundary_selections:
                self.boundary_selections[boundary_type] = []
            
            self.boundary_selections[boundary_type].append(cell_id)
            
            # Show a small notification
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Boundary Selected",
                f"Selected {boundary_type} boundary at cell {cell_id}.\n\nClick on more cells to continue selection."
            )
            
        except Exception as e:
            logging.error(f"Error handling boundary selection: {e}")

    def update_status(self, message):
        """Update status bar with message."""
        self.statusBar().showMessage(message)
    
    def update_log(self, message):
        """Update log with message."""
        if hasattr(self, 'log_view'):
            self.log_view.append(message)
        
    def on_simulation_started(self):
        """Handle simulation started event."""
        self.update_status("Simulation started")
        # Update UI as needed
    
    def on_simulation_finished(self, success):
        """Handle simulation finished event."""
        if success:
            self.update_status("Simulation completed successfully")
        else:
            self.update_status("Simulation failed or was stopped")
        # Update UI as needed

    def _save_visualization_state(self):
        """
        Save visualization state including boundary conditions, ambient settings, and solver parameters.
        
        This ensures that when reopening the project, all visualization settings are restored properly.
        """
        try:
            if not self.current_project or not self.current_project.project_dir:
                logger.warning("Cannot save visualization state: no project or project directory")
                return False
                
            # Import helper methods
            from .visualization_state_helpers import VisualizationStateHelpers
                
            # Create visualization state directory if it doesn't exist
            viz_dir = os.path.join(self.current_project.project_dir, "visualization")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Define visualization state file
            state_file = os.path.join(viz_dir, "visualization_state.json")
            
            # Collect visualization state
            state = {}
            
            # 1. Save viewport camera settings
            if hasattr(self, 'viewport'):
                state['camera'] = VisualizationStateHelpers.save_camera_settings(self.viewport)
                
                # Also save pipeline model properties (colors, opacity, etc.)
                state['model_properties'] = VisualizationStateHelpers.save_pipeline_model_properties(self.viewport)
            
            # 2. Save boundary conditions
            if hasattr(self, 'simulation_controls'):
                state['boundary_conditions'] = VisualizationStateHelpers.save_boundary_conditions(self.simulation_controls)
            
            # 3. Save solver parameters
            if hasattr(self, 'simulation_controls'):
                state['solver_parameters'] = VisualizationStateHelpers.save_solver_parameters(self.simulation_controls)
            
            # 4. Save ambient region settings
            if hasattr(self, 'simulation_controls'):
                state['ambient_settings'] = VisualizationStateHelpers.save_ambient_settings(self.simulation_controls)
            
            # Write to file
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Visualization state saved to {state_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving visualization state: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False