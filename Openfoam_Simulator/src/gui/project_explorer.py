#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project explorer for Openfoam_Simulator application.

This module implements a tree-based project explorer that allows users
to navigate and manage the components of a CFD project, including:
- Geometry and mesh files
- Case setup files
- Simulation results
- Reports and exports
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeView, QAbstractItemView,
    QMenu, QAction, QFileDialog, QMessageBox, QPushButton, QToolBar,
    QLabel, QSplitter, QFrame, QHeaderView, QToolButton, QSizePolicy
)
from PyQt5.QtCore import (
    Qt, QSize, QModelIndex, QPoint, pyqtSignal, QItemSelectionModel,
    QSortFilterProxyModel
)
from PyQt5.QtGui import (
    QIcon, QStandardItemModel, QStandardItem, QFont, QBrush, QColor,
    QDragEnterEvent, QDropEvent, QKeySequence
)

# Import utility modules
from ..utils.logger import get_logger
from ..config import get_value, set_value

logger = get_logger(__name__)


class ProjectItem(QStandardItem):
    """
    Custom QStandardItem for project items.
    
    This class extends QStandardItem to add metadata specific to project items
    such as item type, file path, and other relevant information.
    """
    
    # Item types
    PROJECT = 0
    GEOMETRY = 1
    MESH = 2
    CASE = 3
    BOUNDARIES = 4
    RESULTS = 5
    VISUALIZATION = 6
    REPORT = 7
    FOLDER = 8
    FILE = 9
    
    def __init__(self, name: str, item_type: int, filepath: str = None, icon_name: str = None):
        """
        Initialize a project item.
        
        Args:
            name (str): The display name of the item
            item_type (int): The type of the item (use ProjectItem constants)
            filepath (str, optional): The file path associated with the item
            icon_name (str, optional): The name of the icon to use
        """
        super(ProjectItem, self).__init__(name)
        
        self.item_type = item_type
        self.filepath = filepath
        
        # Set icon based on item type or provided icon name
        if icon_name:
            self.setIcon(self._get_icon(icon_name))
        else:
            self.setIcon(self._get_default_icon())
        
        # Set tooltip to show file path
        if filepath:
            self.setToolTip(filepath)
    
    def _get_default_icon(self) -> QIcon:
        """
        Get the default icon for this item type.
        
        Returns:
            QIcon: The icon for this item type
        """
        icon_name = {
            self.PROJECT: "project",
            self.GEOMETRY: "geometry",
            self.MESH: "mesh",
            self.CASE: "case",
            self.BOUNDARIES: "boundary",
            self.RESULTS: "results",
            self.VISUALIZATION: "visualization",
            self.REPORT: "report",
            self.FOLDER: "folder",
            self.FILE: "file"
        }.get(self.item_type, "file")
        
        return self._get_icon(icon_name)
    
    def _get_icon(self, icon_name: str) -> QIcon:
        """
        Get an icon by name, checking the application resources.
        
        Args:
            icon_name (str): The name of the icon to retrieve
            
        Returns:
            QIcon: The requested icon
        """
        # Check in resources directory
        resources_dir = Path(__file__).parent.parent.parent / 'resources' / 'icons'
        icon_path = resources_dir / f"{icon_name}.png"
        
        if icon_path.exists():
            return QIcon(str(icon_path))
        
        # Fall back to system theme icons
        return QIcon.fromTheme(icon_name, QIcon())


class ProjectExplorer(QWidget):
    """
    Project explorer widget for navigating project components.
    
    Provides a tree view for exploring and managing project files and components, 
    with context menus for common operations.
    """
    
    # Signals
    item_selected = pyqtSignal(ProjectItem)
    item_double_clicked = pyqtSignal(ProjectItem)
    
    def __init__(self, parent=None):
        """
        Initialize the project explorer.
        
        Args:
            parent: The parent widget (should be MainWindow)
        """
        super(ProjectExplorer, self).__init__(parent)
        
        # Store reference to main window
        self.main_window = parent
        
        # Initialize instance variables
        self.current_project = None
        self.model = None
        self.proxy_model = None
        self.project_root_item = None
        
        # Setup UI
        self._setup_ui()
        self._create_actions()
        
        # Show empty project initially
        self._create_empty_model()
        self._connect_signals()
        
        # Show empty project initially
        
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Toolbar for common actions
        self.toolbar = QToolBar()
        self.toolbar.setIconSize(QSize(16, 16))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.layout.addWidget(self.toolbar)
        
        # Tree view
        self.tree_view = QTreeView()
        self.tree_view.setHeaderHidden(True)
        self.tree_view.setEditTriggers(QAbstractItemView.EditKeyPressed | QAbstractItemView.DoubleClicked)
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.setDragEnabled(True)
        self.tree_view.setAcceptDrops(True)
        self.tree_view.setDropIndicatorShown(True)
        self.tree_view.setDragDropMode(QAbstractItemView.InternalMove)
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree_view.setAnimated(True)
        self.layout.addWidget(self.tree_view)
    
    def _create_actions(self):
        """Create actions for toolbar and context menus."""
        # Refresh action
        self.action_refresh = QAction(self._get_icon("refresh"), "Refresh", self)
        self.action_refresh.setToolTip("Refresh project view")
        self.action_refresh.triggered.connect(self.refresh)
        self.toolbar.addAction(self.action_refresh)
        
        # Expand all action
        self.action_expand_all = QAction(self._get_icon("expand"), "Expand All", self)
        self.action_expand_all.setToolTip("Expand all items")
        self.action_expand_all.triggered.connect(self.tree_view.expandAll)
        self.toolbar.addAction(self.action_expand_all)
        
        # Collapse all action
        self.action_collapse_all = QAction(self._get_icon("collapse"), "Collapse All", self)
        self.action_collapse_all.setToolTip("Collapse all items")
        self.action_collapse_all.triggered.connect(self.tree_view.collapseAll)
        self.toolbar.addAction(self.action_collapse_all)
        
        # Separator
        self.toolbar.addSeparator()
        
        # Add folder action
        self.action_add_folder = QAction(self._get_icon("folder_add"), "Add Folder", self)
        self.action_add_folder.setToolTip("Add a new folder")
        self.action_add_folder.triggered.connect(self._add_folder)
        self.toolbar.addAction(self.action_add_folder)
        
        # Import file action
        self.action_import_file = QAction(self._get_icon("import"), "Import File", self)
        self.action_import_file.setToolTip("Import a file to the project")
        self.action_import_file.triggered.connect(self._import_file)
        self.toolbar.addAction(self.action_import_file)
        
        # Add separator
        self.toolbar.addSeparator()
        
        # Filter field - could be added here if needed
    
    def _connect_signals(self):
        """Connect signals to slots."""
        # Connect tree view signals
        self.tree_view.customContextMenuRequested.connect(self._show_context_menu)
        self.tree_view.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.tree_view.doubleClicked.connect(self._on_item_double_clicked)
    
    def _create_empty_model(self):
        """Create an empty model for the tree view."""
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(["Project"])
        
        # Create proxy model for filtering
        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setSourceModel(self.model)
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.tree_view.setModel(self.proxy_model)
        
        # Create empty project root
        self.project_root_item = ProjectItem("No Project", ProjectItem.PROJECT)
        self.model.appendRow(self.project_root_item)
        
        # Expand root
        self.tree_view.expand(self.proxy_model.mapFromSource(self.model.indexFromItem(self.project_root_item)))
    
    def set_project(self, project):
        """
        Set the current project and update the model.
        
        Args:
            project: The project to display
        """
        self.current_project = project
        
        # Clear the model
        self.model.clear()
        self.model.setHorizontalHeaderLabels(["Project"])
        
        if not project:
            # Create empty project root
            self.project_root_item = ProjectItem("No Project", ProjectItem.PROJECT)
            self.model.appendRow(self.project_root_item)
            return
        
        # Create project root
        project_name = "Untitled Project"
        if project.filepath:
            project_name = os.path.basename(project.filepath)
            if project_name.lower().endswith('.f3d'):
                project_name = project_name[:-4]
        
        self.project_root_item = ProjectItem(project_name, ProjectItem.PROJECT, project.filepath)
        self.model.appendRow(self.project_root_item)
        
        # Create standard project folders
        self._create_project_structure()
        
        # Populate with project data
        self._populate_project_data()
        
        # Expand root
        self.tree_view.expand(self.proxy_model.mapFromSource(self.model.indexFromItem(self.project_root_item)))
    
    def _create_project_structure(self):
        """Create the standard project structure in the tree."""
        # Geometry folder
        geometry_item = ProjectItem("Geometry", ProjectItem.FOLDER, None, "geometry")
        self.project_root_item.appendRow(geometry_item)
        
        # Mesh folder
        mesh_item = ProjectItem("Mesh", ProjectItem.FOLDER, None, "mesh")
        self.project_root_item.appendRow(mesh_item)
        
        # Case folder
        case_item = ProjectItem("Case", ProjectItem.FOLDER, None, "case")
        self.project_root_item.appendRow(case_item)
        
        # Create boundary conditions under case
        boundaries_item = ProjectItem("Boundary Conditions", ProjectItem.FOLDER, None, "boundary")
        case_item.appendRow(boundaries_item)
        
        # Results folder
        results_item = ProjectItem("Results", ProjectItem.FOLDER, None, "results")
        self.project_root_item.appendRow(results_item)
        
        # Visualization folder
        viz_item = ProjectItem("Visualization", ProjectItem.FOLDER, None, "visualization")
        self.project_root_item.appendRow(viz_item)
        
        # Reports folder
        reports_item = ProjectItem("Reports", ProjectItem.FOLDER, None, "report")
        self.project_root_item.appendRow(reports_item)
    
    def _populate_project_data(self):
        """Populate the tree with actual project data."""
        if not self.current_project:
            return
        
        try:
            # Populate geometry files
            if hasattr(self.current_project, 'geometry_files') and self.current_project.geometry_files:
                geometry_folder = self._find_item("Geometry", self.project_root_item)
                if geometry_folder:
                    for filepath in self.current_project.geometry_files:
                        name = os.path.basename(filepath)
                        item = ProjectItem(name, ProjectItem.GEOMETRY, filepath)
                        geometry_folder.appendRow(item)
            
            # Populate mesh files
            if hasattr(self.current_project, 'mesh_files') and self.current_project.mesh_files:
                mesh_folder = self._find_item("Mesh", self.project_root_item)
                if mesh_folder:
                    for filepath in self.current_project.mesh_files:
                        name = os.path.basename(filepath)
                        item = ProjectItem(name, ProjectItem.MESH, filepath)
                        mesh_folder.appendRow(item)
            
            # Populate case files
            if hasattr(self.current_project, 'case_files') and self.current_project.case_files:
                case_folder = self._find_item("Case", self.project_root_item)
                if case_folder:
                    for filepath in self.current_project.case_files:
                        name = os.path.basename(filepath)
                        item = ProjectItem(name, ProjectItem.CASE, filepath)
                        case_folder.appendRow(item)
            
            # Populate boundary conditions
            if hasattr(self.current_project, 'boundary_files') and self.current_project.boundary_files:
                case_folder = self._find_item("Case", self.project_root_item)
                if case_folder:
                    boundary_folder = self._find_item("Boundary Conditions", case_folder)
                    if boundary_folder:
                        for filepath in self.current_project.boundary_files:
                            name = os.path.basename(filepath)
                            item = ProjectItem(name, ProjectItem.BOUNDARIES, filepath)
                            boundary_folder.appendRow(item)
            
            # Populate results
            if hasattr(self.current_project, 'result_files') and self.current_project.result_files:
                results_folder = self._find_item("Results", self.project_root_item)
                if results_folder:
                    for filepath in self.current_project.result_files:
                        name = os.path.basename(filepath)
                        item = ProjectItem(name, ProjectItem.RESULTS, filepath)
                        results_folder.appendRow(item)
            
            # Populate visualization states
            if hasattr(self.current_project, 'visualization_files') and self.current_project.visualization_files:
                viz_folder = self._find_item("Visualization", self.project_root_item)
                if viz_folder:
                    for filepath in self.current_project.visualization_files:
                        name = os.path.basename(filepath)
                        item = ProjectItem(name, ProjectItem.VISUALIZATION, filepath)
                        viz_folder.appendRow(item)
            
            # Populate reports
            if hasattr(self.current_project, 'report_files') and self.current_project.report_files:
                reports_folder = self._find_item("Reports", self.project_root_item)
                if reports_folder:
                    for filepath in self.current_project.report_files:
                        name = os.path.basename(filepath)
                        item = ProjectItem(name, ProjectItem.REPORT, filepath)
                        reports_folder.appendRow(item)
                        
        except Exception as e:
            logger.error(f"Error populating project data: {e}")
    
    def _find_item(self, name: str, parent_item: QStandardItem) -> Optional[QStandardItem]:
        """
        Find a child item by name under a parent item.
        
        Args:
            name (str): The name of the item to find
            parent_item (QStandardItem): The parent item to search under
            
        Returns:
            QStandardItem: The found item, or None if not found
        """
        for row in range(parent_item.rowCount()):
            child = parent_item.child(row)
            if child and child.text() == name:
                return child
        return None
    
    def refresh(self):
        """Refresh the project view."""
        # Store current selection
        selected_indexes = self.tree_view.selectionModel().selectedIndexes()
        selected_items = []
        for index in selected_indexes:
            source_index = self.proxy_model.mapToSource(index)
            item = self.model.itemFromIndex(source_index)
            if item:
                selected_items.append((item.text(), item.parent().text() if item.parent() else None))
        
        # Repopulate the project
        if self.current_project:
            self.set_project(self.current_project)
        
        # Restore selection if possible
        if selected_items:
            selection_model = self.tree_view.selectionModel()
            selection_model.clearSelection()
            
            for item_text, parent_text in selected_items:
                for row in range(self.model.rowCount()):
                    root_item = self.model.item(row)
                    if parent_text is None and root_item.text() == item_text:
                        # Root level item
                        index = self.proxy_model.mapFromSource(self.model.indexFromItem(root_item))
                        selection_model.select(index, QItemSelectionModel.Select)
                    else:
                        # Child item
                        self._select_child_by_name(root_item, item_text, parent_text, selection_model)
    
    def _select_child_by_name(self, parent_item: QStandardItem, item_text: str, 
                             parent_text: str, selection_model: QItemSelectionModel):
        """
        Recursively select a child item by name.
        
        Args:
            parent_item (QStandardItem): The parent item to search under
            item_text (str): The text of the item to select
            parent_text (str): The text of the parent of the item to select
            selection_model (QItemSelectionModel): The selection model to use for selection
        """
        if parent_item.text() == parent_text:
            for row in range(parent_item.rowCount()):
                child = parent_item.child(row)
                if child and child.text() == item_text:
                    index = self.proxy_model.mapFromSource(self.model.indexFromItem(child))
                    selection_model.select(index, QItemSelectionModel.Select)
                    return
        
        # Recursively search children
        for row in range(parent_item.rowCount()):
            child = parent_item.child(row)
            if child:
                self._select_child_by_name(child, item_text, parent_text, selection_model)
    
    def _get_icon(self, icon_name: str) -> QIcon:
        """
        Get an icon by name, checking the application resources.
        
        Args:
            icon_name (str): The name of the icon to retrieve
            
        Returns:
            QIcon: The requested icon
        """
        # Check in resources directory
        resources_dir = Path(__file__).parent.parent.parent / 'resources' / 'icons'
        icon_path = resources_dir / f"{icon_name}.png"
        
        if icon_path.exists():
            return QIcon(str(icon_path))
        
        # Fall back to system theme icons
        return QIcon.fromTheme(icon_name, QIcon())
    
    def _add_folder(self):
        """Add a new folder to the selected location in the project."""
        # Get selected item
        selected = self.tree_view.selectionModel().currentIndex()
        if not selected.isValid():
            parent_item = self.project_root_item
        else:
            source_index = self.proxy_model.mapToSource(selected)
            item = self.model.itemFromIndex(source_index)
            if item.item_type == ProjectItem.FILE:
                parent_item = item.parent()
            else:
                parent_item = item
        
        # Create new folder
        new_folder = ProjectItem("New Folder", ProjectItem.FOLDER)
        parent_item.appendRow(new_folder)
        
        # Start editing to allow user to rename
        new_index = self.proxy_model.mapFromSource(self.model.indexFromItem(new_folder))
        self.tree_view.setCurrentIndex(new_index)
        self.tree_view.edit(new_index)
    
    def _import_file(self):
        """Import a file to the project."""
        # Get selected item to determine target folder
        selected = self.tree_view.selectionModel().currentIndex()
        if not selected.isValid():
            target_folder = self.project_root_item
        else:
            source_index = self.proxy_model.mapToSource(selected)
            item = self.model.itemFromIndex(source_index)
            if item.item_type == ProjectItem.FILE:
                target_folder = item.parent()
            else:
                target_folder = item
        
        # Show file dialog
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Import File",
            "",
            "All Files (*)"
        )
        
        if not filepath:
            return
        
        # Determine file type based on extension
        file_ext = os.path.splitext(filepath)[1].lower()
        if file_ext in ['.stl', '.obj', '.step', '.stp', '.iges', '.igs']:
            item_type = ProjectItem.GEOMETRY
        elif file_ext in ['.vtk', '.vtu', '.vtp', '.vts', '.vtr', '.blockMesh', '.snappyHexMesh']:
            item_type = ProjectItem.MESH
        elif file_ext in ['.foam', '.OpenFOAM']:
            item_type = ProjectItem.CASE
        elif file_ext in ['.csv', '.dat', '.txt']:
            item_type = ProjectItem.RESULTS
        elif file_ext in ['.pvsm']:
            item_type = ProjectItem.VISUALIZATION
        elif file_ext in ['.pdf', '.html', '.docx']:
            item_type = ProjectItem.REPORT
        else:
            item_type = ProjectItem.FILE
        
        # Create file item
        filename = os.path.basename(filepath)
        file_item = ProjectItem(filename, item_type, filepath)
        
        # Add to target folder
        target_folder.appendRow(file_item)
        
        # Auto-move to appropriate folder based on type
        if target_folder == self.project_root_item:
            self._auto_categorize_file(file_item)
    
    def _auto_categorize_file(self, file_item: ProjectItem):
        """
        Move file items to the appropriate category folder based on their type.
        
        Args:
            file_item (ProjectItem): The file item to categorize
        """
        target_folder_name = {
            ProjectItem.GEOMETRY: "Geometry",
            ProjectItem.MESH: "Mesh",
            ProjectItem.CASE: "Case",
            ProjectItem.BOUNDARIES: "Boundary Conditions",
            ProjectItem.RESULTS: "Results",
            ProjectItem.VISUALIZATION: "Visualization",
            ProjectItem.REPORT: "Reports"
        }.get(file_item.item_type)
        
        if not target_folder_name:
            return
        
        # Find target folder
        if target_folder_name == "Boundary Conditions":
            case_folder = self._find_item("Case", self.project_root_item)
            if case_folder:
                target_folder = self._find_item(target_folder_name, case_folder)
            else:
                return
        else:
            target_folder = self._find_item(target_folder_name, self.project_root_item)
        
        if not target_folder:
            return
        
        # Clone the item
        new_item = ProjectItem(
            file_item.text(),
            file_item.item_type,
            file_item.filepath,
        )
        
        # Add to target folder
        target_folder.appendRow(new_item)
        
        # Remove from root
        self.project_root_item.removeRow(file_item.row())
    
    def _show_context_menu(self, pos: QPoint):
        """
        Show context menu for the item at the given position.
        
        Args:
            pos (QPoint): The position at which to show the menu
        """
        # Get the item under the cursor
        index = self.tree_view.indexAt(pos)
        if not index.isValid():
            return
        
        source_index = self.proxy_model.mapToSource(index)
        item = self.model.itemFromIndex(source_index)
        
        menu = QMenu(self)
        
        # Common actions
        menu.addAction(self.action_refresh)
        
        if item.item_type == ProjectItem.FOLDER:
            # Folder actions
            menu.addSeparator()
            menu.addAction(self.action_add_folder)
            menu.addAction(self.action_import_file)
            
            # Add expand/collapse actions
            menu.addSeparator()
            menu.addAction(self.action_expand_all)
            menu.addAction(self.action_collapse_all)
            
            # Add rename action
            rename_action = QAction("Rename", self)
            rename_action.triggered.connect(lambda: self.tree_view.edit(index))
            menu.addAction(rename_action)
            
            # Add delete action if not a standard folder
            if item.text() not in ["Geometry", "Mesh", "Case", "Boundary Conditions", 
                                  "Results", "Visualization", "Reports"]:
                menu.addSeparator()
                delete_action = QAction("Delete", self)
                delete_action.triggered.connect(lambda: self._delete_item(item))
                menu.addAction(delete_action)
                
        elif item.item_type in [ProjectItem.FILE, ProjectItem.GEOMETRY, ProjectItem.MESH, 
                               ProjectItem.CASE, ProjectItem.BOUNDARIES, ProjectItem.RESULTS,
                               ProjectItem.VISUALIZATION, ProjectItem.REPORT]:
            # File actions
            menu.addSeparator()
            
            # Open action
            open_action = QAction("Open", self)
            open_action.triggered.connect(lambda: self._open_item(item))
            menu.addAction(open_action)
            
            # Add specific actions based on file type
            if item.item_type == ProjectItem.GEOMETRY:
                # Geometry actions
                menu.addAction("Generate Mesh", lambda: self._generate_mesh_from_geometry(item))
            
            elif item.item_type == ProjectItem.MESH:
                # Mesh actions
                menu.addAction("Setup Case", lambda: self._setup_case_from_mesh(item))
            
            elif item.item_type == ProjectItem.CASE:
                # Case actions
                menu.addAction("Run Simulation", lambda: self._run_simulation_from_case(item))
            
            elif item.item_type == ProjectItem.RESULTS:
                # Results actions
                menu.addAction("Visualize", lambda: self._visualize_results(item))
                menu.addAction("Generate Report", lambda: self._generate_report_from_results(item))
            
            # Common file actions
            menu.addSeparator()
            menu.addAction("Rename", lambda: self.tree_view.edit(index))
            menu.addAction("Delete", lambda: self._delete_item(item))
        
        # Show the menu
        menu.exec_(self.tree_view.viewport().mapToGlobal(pos))
    
    def _open_item(self, item: ProjectItem):
        """
        Open the selected item.
        
        Args:
            item (ProjectItem): The item to open
        """
        if not item.filepath or not os.path.exists(item.filepath):
            QMessageBox.warning(self, "File Not Found", 
                              f"The file {item.filepath} does not exist.")
            return
        
        try:
            # Handle different file types
            if item.item_type == ProjectItem.GEOMETRY:
                # Open in viewport
                if self.main_window and hasattr(self.main_window, 'viewport'):
                    logger.info(f"Opening geometry file: {item.filepath}")
                    # This would call a method in the viewport to load the geometry
                    # self.main_window.viewport.load_geometry(item.filepath)
            
            elif item.item_type == ProjectItem.MESH:
                # Open in viewport
                if self.main_window and hasattr(self.main_window, 'viewport'):
                    logger.info(f"Opening mesh file: {item.filepath}")
                    # This would call a method in the viewport to load the mesh
                    # self.main_window.viewport.load_mesh(item.filepath)
            
            elif item.item_type == ProjectItem.CASE:
                # Open case settings
                logger.info(f"Opening case file: {item.filepath}")
                # This would open a case editor dialog
            
            elif item.item_type == ProjectItem.RESULTS:
                # Open in viewport
                if self.main_window and hasattr(self.main_window, 'viewport'):
                    logger.info(f"Opening results file: {item.filepath}")
                    # This would call a method in the viewport to load the results
                    # self.main_window.viewport.load_results(item.filepath)
            
            elif item.item_type == ProjectItem.VISUALIZATION:
                # Load visualization state
                if self.main_window and hasattr(self.main_window, 'viewport'):
                    logger.info(f"Loading visualization state: {item.filepath}")
                    # This would call a method in the viewport to load the visualization state
                    # self.main_window.viewport.load_state(item.filepath)
            
            elif item.item_type == ProjectItem.REPORT:
                # Open report in external viewer
                logger.info(f"Opening report: {item.filepath}")
                import webbrowser
                webbrowser.open(item.filepath)
            
            else:
                # Open with default application
                logger.info(f"Opening file with default application: {item.filepath}")
                import webbrowser
                webbrowser.open(item.filepath)
        
        except Exception as e:
            logger.error(f"Error opening item: {e}")
            QMessageBox.critical(self, "Error Opening Item", 
                               f"An error occurred while opening the item:\n\n{str(e)}")
    
    def _delete_item(self, item: ProjectItem):
        """
        Delete the selected item.
        
        Args:
            item (ProjectItem): The item to delete
        """
        # Confirm deletion
        reply = QMessageBox.question(
            self, 
            "Confirm Deletion",
            f"Are you sure you want to delete '{item.text()}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Remove from model
        if item.parent():
            item.parent().removeRow(item.row())
        else:
            self.model.removeRow(item.row())
        
        # Update project data
        if self.current_project and item.filepath:
            # Remove from appropriate list based on type
            if item.item_type == ProjectItem.GEOMETRY and hasattr(self.current_project, 'geometry_files'):
                if item.filepath in self.current_project.geometry_files:
                    self.current_project.geometry_files.remove(item.filepath)
            
            elif item.item_type == ProjectItem.MESH and hasattr(self.current_project, 'mesh_files'):
                if item.filepath in self.current_project.mesh_files:
                    self.current_project.mesh_files.remove(item.filepath)
            
            elif item.item_type == ProjectItem.CASE and hasattr(self.current_project, 'case_files'):
                if item.filepath in self.current_project.case_files:
                    self.current_project.case_files.remove(item.filepath)
            
            elif item.item_type == ProjectItem.BOUNDARIES and hasattr(self.current_project, 'boundary_files'):
                if item.filepath in self.current_project.boundary_files:
                    self.current_project.boundary_files.remove(item.filepath)
            
            elif item.item_type == ProjectItem.RESULTS and hasattr(self.current_project, 'result_files'):
                if item.filepath in self.current_project.result_files:
                    self.current_project.result_files.remove(item.filepath)
            
            elif item.item_type == ProjectItem.VISUALIZATION and hasattr(self.current_project, 'visualization_files'):
                if item.filepath in self.current_project.visualization_files:
                    self.current_project.visualization_files.remove(item.filepath)
            
            elif item.item_type == ProjectItem.REPORT and hasattr(self.current_project, 'report_files'):
                if item.filepath in self.current_project.report_files:
                    self.current_project.report_files.remove(item.filepath)
            
            # Mark project as modified
            if self.main_window:
                self.main_window.set_modified(True)
    
    def _generate_mesh_from_geometry(self, item: ProjectItem):
        """
        Generate a mesh from a geometry file.
        
        Args:
            item (ProjectItem): The geometry item
        """
        if not self.main_window:
            return
        
        # Update project data
        if self.current_project and hasattr(self.current_project, 'set_active_geometry'):
            self.current_project.set_active_geometry(item.filepath)
        
        # Call main window's generate mesh function
        if hasattr(self.main_window, 'generate_mesh'):
            self.main_window.generate_mesh()
    
    def _setup_case_from_mesh(self, item: ProjectItem):
        """
        Set up a case from a mesh file.
        
        Args:
            item (ProjectItem): The mesh item
        """
        if not self.main_window:
            return
        
        # Update project data
        if self.current_project and hasattr(self.current_project, 'set_active_mesh'):
            self.current_project.set_active_mesh(item.filepath)
        
        # Call main window's setup case function
        if hasattr(self.main_window, 'setup_case'):
            self.main_window.setup_case()
    
    def _run_simulation_from_case(self, item: ProjectItem):
        """
        Run a simulation from a case file.
        
        Args:
            item (ProjectItem): The case item
        """
        if not self.main_window:
            return
        
        # Update project data
        if self.current_project and hasattr(self.current_project, 'set_active_case'):
            self.current_project.set_active_case(item.filepath)
        
        # Call main window's run simulation function
        if hasattr(self.main_window, 'run_simulation'):
            self.main_window.run_simulation()
    
    def _visualize_results(self, item: ProjectItem):
        """
        Visualize simulation results.
        
        Args:
            item (ProjectItem): The results item
        """
        if not self.main_window or not hasattr(self.main_window, 'viewport'):
            return
        
        # Load results in viewport
        if item.filepath and os.path.exists(item.filepath):
            logger.info(f"Visualizing results: {item.filepath}")
            # This would call a method in the viewport to load the results
            # self.main_window.viewport.load_results(item.filepath)
    
    def _generate_report_from_results(self, item: ProjectItem):
        """
        Generate a report from simulation results.
        
        Args:
            item (ProjectItem): The results item
        """
        if not self.main_window:
            return
        
        # Update project data
        if self.current_project and hasattr(self.current_project, 'set_active_results'):
            self.current_project.set_active_results(item.filepath)
        
        # Call appropriate function to generate report
        # This would open a report generator dialog or call a report generation function
        logger.info(f"Generating report from results: {item.filepath}")
    
    def _on_selection_changed(self, selected, deselected):
        """
        Handle selection changes in the tree view.
        
        Args:
            selected: The selected items
            deselected: The deselected items
        """
        indexes = selected.indexes()
        if not indexes:
            return
        
        # Get the selected item
        index = indexes[0]
        source_index = self.proxy_model.mapToSource(index)
        item = self.model.itemFromIndex(source_index)
        
        # Emit signal with selected item
        self.item_selected.emit(item)
    
    def _on_item_double_clicked(self, index):
        """
        Handle double-click on an item.
        
        Args:
            index: The index of the clicked item
        """
        if not index.isValid():
            return
        
        source_index = self.proxy_model.mapToSource(index)
        item = self.model.itemFromIndex(source_index)
        
        # Emit signal
        self.item_double_clicked.emit(item)
        
        # Open the item if it's a file
        if item.item_type not in [ProjectItem.FOLDER, ProjectItem.PROJECT]:
            self._open_item(item)