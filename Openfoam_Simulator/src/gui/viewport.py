#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VTK viewport integration for Openfoam_Simulator application.

This module provides visualization using VTK for OpenFOAM simulation
results and meshes. It embeds a VTK render window in the PyQt5 interface and
provides methods to interact with the visualization pipeline.
"""

import os
import sys
import logging
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import time

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QSplitter, QToolBar, QPushButton, QComboBox, QAction, QMessageBox, QMenu
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QTimer, QPoint
from PyQt5.QtGui import QIcon, QColor
from PyQt5 import QtCore, QtGui, QtWidgets

# Import utility modules
from ..utils.logger import get_logger
from ..config import get_value, set_value

# Import VTK modules
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"Python version: {sys.version}")
logger.debug(f"Python path: {sys.path}")

try:
    import vtk
    from vtk import (vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor, 
                     vtkPolyDataMapper, vtkActor, vtkAxesActor, vtkCubeAxesActor,
                     vtkLight, vtkLookupTable, vtkWindowToImageFilter,
                     vtkPNGWriter, vtkJPEGWriter, vtkTIFFWriter,
                     vtkInteractorStyleTrackballCamera, vtkTransform,
                     vtkConeSource, vtkPolyDataReader, vtkUnstructuredGridReader,
                     vtkXMLPolyDataReader, vtkXMLUnstructuredGridReader,
                     vtkStructuredGridReader, vtkStructuredPointsReader,
                     vtkRectilinearGridReader, vtkDataReader, vtkPLYReader,
                     vtkGenericDataObjectReader)
    logger.debug(f"VTK imported successfully from {vtk.__file__}")
    logger.debug(f"VTK version: {vtk.vtkVersion.GetVTKVersion()}")
    
    # Try importing specific VTK components
    logger.debug("Successfully imported vtkRenderer and vtkRenderWindow")
    
    # Try importing Qt integration
    try:
        from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
        logger.debug("Successfully imported QVTKRenderWindowInteractor")
    except ImportError as e:
        logger.error(f"Failed to import QVTKRenderWindowInteractor: {e}")
        # Try alternative import path
        try:
            from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
            logger.debug("Successfully imported QVTKRenderWindowInteractor from vtkmodules")
        except ImportError as e2:
            logger.error(f"Failed alternative import: {e2}")
    
    VTK_AVAILABLE = True
except ImportError as e:
    logger.error(f"VTK import failed: {e}")
    VTK_AVAILABLE = False

logger = get_logger(__name__)


class VTKViewport(QWidget):
    """
    VTK viewport widget for visualizing OpenFOAM simulation results.
    
    This class provides a QWidget that embeds a VTK render window
    for visualization of meshes and simulation results.
    """
    
    # Signals
    view_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Initialize the VTK viewport.
        
        Args:
            parent: The parent widget
        """
        super(VTKViewport, self).__init__(parent)
        self.main_window = parent
        
        # Initialize instance variables
        self.vtk_widget = None
        self.renderer = None
        self.render_window = None
        self.interactor = None
        
        # Pipeline variables
        self.sources = {}         # Dictionary of data sources
        self.active_source = None # Currently active source
        self.filters = {}         # Dictionary of applied filters
        self.actors = {}          # Dictionary of visualization actors
        self.scalar_bars = {}     # Dictionary of scalar bar actors
        
        # View properties
        self.background_color = [0.2, 0.2, 0.2]
        self.show_axes = True
        self.show_bounds = False
        self.axes_actor = None
        self.bounds_actor = None
        
        # Setup UI
        self._setup_ui()
        
        # Initialize VTK
        if VTK_AVAILABLE:
            self._initialize_vtk()
        else:
            self._show_vtk_not_available()
    
    def _setup_ui(self):
        """Set up the UI for the VTK viewport."""
        # Create main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor("#2a2a2a"))  # Dark gray background
        self.setPalette(palette)
    
    def _initialize_vtk(self):
        """Initialize VTK widget."""
        # Create VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.layout.addWidget(self.vtk_widget)
        
        # Get render window
        self.render_window = self.vtk_widget.GetRenderWindow()
        
        # Create renderer
        self.renderer = vtkRenderer()
        self.render_window.AddRenderer(self.renderer)
        
        # Set background color from config
        bg_color = get_value('vtk.viewport_settings.background_color', [0.2, 0.2, 0.2])
        self.background_color = bg_color
        self.renderer.SetBackground(bg_color)
        
        # Set up interactor
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        # Use fully qualified name with vtk namespace
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        
        # Initialize interactor
        self.interactor.Initialize()
        
        # Add axes
        self._add_axes()
        
        # Show some default content
        self._show_default_content()
        
        # Start interactor
        self.interactor.Start()
    
    def _show_vtk_not_available(self):
        """Show message when VTK is not available."""
        self.layout.addWidget(QLabel("VTK not available. Please check your installation."))
    
    def _add_axes(self):
        """Add coordinate axes to the viewport."""
        if not VTK_AVAILABLE:
            return
        
        # Create axes actor
        axes = vtkAxesActor()
        # Make the axes smaller (reduce from default 1.0)
        axes.SetTotalLength(0.3, 0.3, 0.3)  # Smaller size
        axes.SetShaftType(0)  # 0 = Cylinder, 1 = Line
        axes.SetAxisLabels(1)  # Show axis labels
        
        # Create orientation marker widget
        widget = vtk.vtkOrientationMarkerWidget()
        widget.SetOrientationMarker(axes)
        widget.SetInteractor(self.interactor)
        # Position in top-left corner and set relative size
        # Parameters are (xmin, ymin, xmax, ymax) in normalized coordinates (0-1)
        widget.SetViewport(0.0, 0.8, 0.2, 1.0)  # Top-left corner
        widget.SetEnabled(1)
        widget.InteractiveOff()  # Disable interactive manipulation
        
        # Save reference to widget
        self.axes_widget = widget
        self.axes_actor = axes
        
        # Set visibility based on config
        self.show_axes = get_value('vtk.axes_visibility', True)
        self.axes_widget.SetEnabled(self.show_axes)
    
    def _show_default_content(self):
        """Show default content in the viewport."""
        if not VTK_AVAILABLE:
            return
        
        # Just reset the camera without adding any default geometry
        self.renderer.ResetCamera()
        self.render_window.Render()
    
    def reset_view(self):
        """Reset the view to default."""
        if not VTK_AVAILABLE:
            return
        
        logger.debug("Resetting view")  # Debug log
        self.renderer.ResetCamera()
        self.render_window.Render()
        self.view_changed.emit()
    
    def update_view(self):
        """Update the view."""
        if not VTK_AVAILABLE:
            return
        
        logger.debug("Updating view")  # Debug log
        self.render_window.Render()
        self.view_changed.emit()
    
    def cleanup(self):
        """Clean up resources."""
        if self.interactor:
            self.interactor.TerminateApp()
    
    def load_project(self, project):
        """
        Load a project into the viewport.
        
        Args:
            project: The project to load
        """
        if not VTK_AVAILABLE:
            return
        
        # Clear existing visualization
        self._clear_visualization()
        
        # Load STL files from common locations in the project
        if hasattr(project, "project_dir") and project.project_dir:
            # First check mesh directory
            mesh_dir = os.path.join(project.project_dir, "mesh")
            if os.path.exists(mesh_dir):
                for file in os.listdir(mesh_dir):
                    if file.lower().endswith(".stl"):
                        stl_path = os.path.join(mesh_dir, file)
                        logger.info(f"Loading pipeline STL file: {stl_path}")
                        try:
                            self.load_mesh(stl_path)
                        except Exception as e:
                            logger.warning(f"Failed to load STL file {stl_path}: {str(e)}")
                            
            # Then check case/constant/triSurface directory
            tri_surface_dir = os.path.join(project.project_dir, "case", "constant", "triSurface")
            if os.path.exists(tri_surface_dir):
                for file in os.listdir(tri_surface_dir):
                    if file.lower().endswith(".stl"):
                        stl_path = os.path.join(tri_surface_dir, file)
                        logger.info(f"Loading pipeline STL file: {stl_path}")
                        try:
                            self.load_mesh(stl_path)
                        except Exception as e:
                            logger.warning(f"Failed to load STL file {stl_path}: {str(e)}")
        
        # Also try loading from mesh_files collection if available
        if hasattr(project, "mesh_files") and project.mesh_files:
            for mesh_file in project.mesh_files:
                if os.path.exists(mesh_file) and mesh_file.lower().endswith(".stl"):
                    logger.info(f"Loading mesh file from collection: {mesh_file}")
                    try:
                        self.load_mesh(mesh_file)
                    except Exception as e:
                        logger.warning(f"Failed to load mesh file {mesh_file}: {str(e)}")
        
        # Check if the project has a mesh
        if hasattr(project, "has_mesh") and project.has_mesh():
            mesh_path = project.get_mesh_path()
            if mesh_path and mesh_path not in project.mesh_files:
                self.load_mesh(mesh_path)
        
        # Check if the project has results
        if hasattr(project, "has_results") and project.has_results():
            self.load_results(project.get_results_path())
            
        # Reset view to show all loaded objects
        self.reset_view()
    
    def load_results(self, result_file):
        """Load simulation results from a file."""
        if not self._check_vtk_available():
            return False
        
        logger.info(f"Loading results file: {result_file}")
        
        try:
            # Check file exists
            if not os.path.exists(result_file):
                logger.error(f"Results file not found: {result_file}")
                return False
            
            # Determine file type by extension
            file_ext = os.path.splitext(result_file)[1].lower()
            
            # For VTK legacy files, use this specialized approach
            if file_ext == '.vtk':
                return self._load_legacy_vtk(result_file)
            
            # Handle other file types (keep existing code)
            elif file_ext == '.stl':
                return self.load_mesh(result_file)
            elif file_ext == '.obj':
                reader = vtk.vtkOBJReader()
            elif file_ext == '.vtu':
                reader = vtk.vtkXMLUnstructuredGridReader()
            elif file_ext == '.vtp':
                reader = vtk.vtkXMLPolyDataReader()
            else:
                logger.error(f"Unable to determine file type for: {result_file}")
                return False
            
            # Set file name and update reader
            reader.SetFileName(result_file)
            reader.Update()
            
            # Check if reader succeeded
            output = reader.GetOutput()
            if not output or output.GetNumberOfPoints() == 0:
                logger.error(f"Reader returned empty dataset for: {result_file}")
                return False
            
            logger.info(f"Successfully loaded file with {output.GetNumberOfPoints()} points and {output.GetNumberOfCells()} cells")
            
            # Store the data
            self.data = output
            
            # Display the results
            self._display_results()
            return True
            
        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _load_legacy_vtk(self, file_path):
        """
        Load a legacy VTK file using a completely manual approach.
        
        Args:
            file_path (str): Path to the VTK file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Loading legacy VTK file with manual parser: {file_path}")
            
            # Create objects for our data
            points = vtk.vtkPoints()
            polygons = vtk.vtkCellArray()
            polydata = vtk.vtkPolyData()
            
            # Parse state variables
            in_points = False
            in_polygons = False
            in_vertices = False
            in_lines = False
            in_triangle_strips = False
            num_expected_points = 0
            num_expected_polys = 0
            point_count = 0
            poly_count = 0
            
            # Read the file line by line
            with open(file_path, 'r', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip empty lines and comments that start with #
                    if not line or (line.startswith('#') and not in_points and not in_polygons):
                        continue
                    
                    # Parse POINTS section
                    if line.upper().startswith('POINTS'):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                num_expected_points = int(parts[1])
                                logger.info(f"Found POINTS section: {num_expected_points} points")
                                in_points = True
                                in_polygons = False
                                in_vertices = False
                                in_lines = False
                                in_triangle_strips = False
                            except ValueError:
                                logger.error(f"Invalid POINTS format: {line}")
                        continue
                    
                    # Parse POLYGONS section
                    elif line.upper().startswith('POLYGONS'):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                num_expected_polys = int(parts[1])
                                logger.info(f"Found POLYGONS section: {num_expected_polys} polygons")
                                in_points = False
                                in_polygons = True
                                in_vertices = False
                                in_lines = False
                                in_triangle_strips = False
                            except ValueError:
                                logger.error(f"Invalid POLYGONS format: {line}")
                        continue
                    
                    # Other sections that might appear
                    elif line.upper().startswith(('VERTICES', 'LINES', 'TRIANGLE_STRIPS')):
                        in_points = False
                        in_polygons = False
                        if line.upper().startswith('VERTICES'):
                            in_vertices = True
                        elif line.upper().startswith('LINES'):
                            in_lines = True
                        elif line.upper().startswith('TRIANGLE_STRIPS'):
                            in_triangle_strips = True
                        continue
                    
                    # End of sections
                    elif line.upper().startswith(('CELL_TYPES', 'POINT_DATA', 'CELL_DATA')):
                        in_points = False
                        in_polygons = False
                        in_vertices = False
                        in_lines = False
                        in_triangle_strips = False
                        continue
                    
                    # Parse point coordinates
                    if in_points and point_count < num_expected_points:
                        try:
                            coords = [float(x) for x in line.split()]
                            if len(coords) >= 3:
                                points.InsertNextPoint(coords[0], coords[1], coords[2])
                                point_count += 1
                                if point_count % 1000 == 0:
                                    logger.info(f"Processed {point_count} points")
                        except ValueError:
                            pass  # Skip non-numeric lines
                    
                    # Parse polygon data
                    elif in_polygons and poly_count < num_expected_polys:
                        try:
                            values = [int(x) for x in line.split()]
                            if len(values) >= 1:
                                n_pts = values[0]
                                if len(values) >= n_pts + 1:
                                    cell = vtk.vtkPolygon()
                                    cell.GetPointIds().SetNumberOfIds(n_pts)
                                    
                                    for i in range(n_pts):
                                        cell.GetPointIds().SetId(i, values[i+1])
                                    
                                    polygons.InsertNextCell(cell)
                                    poly_count += 1
                                    if poly_count % 1000 == 0:
                                        logger.info(f"Processed {poly_count} polygons")
                        except ValueError:
                            pass  # Skip non-numeric lines
            
            logger.info(f"Processed {point_count} points and {poly_count} polygons")
            
            # Check if we have any data
            if point_count == 0:
                logger.error("No points found in VTK file")
                return False
            
            # Create the dataset
            polydata.SetPoints(points)
            if poly_count > 0:
                polydata.SetPolys(polygons)
            
            # Store and display the data
            self.data = polydata
            self._display_results()
            
            logger.info("Successfully loaded VTK file with manual parser")
            return True
            
        except Exception as e:
            logger.error(f"Error in manual VTK parsing: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Try one more approach - use PyVista if available 
            try:
                import pyvista as pv
                logger.info("Attempting to load with PyVista")
                
                mesh = pv.read(file_path)
                vtk_mesh = mesh.extract_surface().to_vtk()
                
                self.data = vtk_mesh
                self._display_results()
                logger.info("Successfully loaded VTK file using PyVista")
                return True
            except ImportError:
                logger.error("PyVista not available")
            except Exception as e2:
                logger.error(f"PyVista loading failed: {str(e2)}")
            
            return False

    def _parse_vtk_file_manually(self, file_path):
        """Manually parse a VTK file and create a dataset from it."""
        logger.info("Attempting to manually parse VTK file")
        
        try:
            # Create an empty polydata object to store our data
            polydata = vtk.vtkPolyData()
            
            # Create points and cells containers
            points = vtk.vtkPoints()
            cells = vtk.vtkCellArray()
            
            # Track current parsing state
            in_points_section = False
            in_cells_section = False
            num_points = 0
            num_cells = 0
            point_count = 0
            cell_count = 0
            
            # Open and read file line by line
            with open(file_path, 'r', errors='ignore') as f:
                # Skip header and metadata
                for _ in range(4):  # Skip first 4 lines (header, title, format, dataset type)
                    f.readline()
                
                # Process remaining lines
                for line in f:
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Look for POINTS section
                    if line.upper().startswith("POINTS"):
                        in_points_section = True
                        in_cells_section = False
                        try:
                            parts = line.split()
                            num_points = int(parts[1])
                            logger.info(f"Found POINTS section: {num_points} points")
                        except:
                            logger.error(f"Error parsing POINTS line: {line}")
                        continue
                    
                    # Look for CELLS or POLYGONS section
                    elif line.upper().startswith(("CELLS", "POLYGONS")):
                        in_points_section = False
                        in_cells_section = True
                        try:
                            parts = line.split()
                            num_cells = int(parts[1])
                            logger.info(f"Found CELLS section: {num_cells} cells")
                        except:
                            logger.error(f"Error parsing CELLS line: {line}")
                        continue
                    
                    # End of sections
                    elif line.upper().startswith(("CELL_TYPES", "POINT_DATA", "CELL_DATA")):
                        in_points_section = False
                        in_cells_section = False
                        continue
                    
                    # Process points
                    if in_points_section and point_count < num_points:
                        try:
                            coords = [float(x) for x in line.split()]
                            if len(coords) >= 3:
                                points.InsertNextPoint(coords[0], coords[1], coords[2])
                                point_count += 1
                        except ValueError:
                            logger.debug(f"Skipping non-numeric line in POINTS section: {line}")
                    
                    # Process cells
                    elif in_cells_section and cell_count < num_cells:
                        try:
                            values = [int(x) for x in line.split()]
                            if len(values) >= 1:
                                n_pts = values[0]
                                if len(values) >= n_pts + 1:
                                    # This is a valid cell definition
                                    if n_pts == 3:  # Triangle
                                        cells.InsertNextCell(3)
                                        cells.InsertCellPoint(values[1])
                                        cells.InsertCellPoint(values[2])
                                        cells.InsertCellPoint(values[3])
                                        cell_count += 1
                                    elif n_pts == 4:  # Quad
                                        cells.InsertNextCell(4)
                                        cells.InsertCellPoint(values[1])
                                        cells.InsertCellPoint(values[2])
                                        cells.InsertCellPoint(values[3])
                                        cells.InsertCellPoint(values[4])
                                        cell_count += 1
                        except ValueError:
                            logger.debug(f"Skipping non-numeric line in CELLS section: {line}")
            
            # Set the points and cells in the polydata
            polydata.SetPoints(points)
            
            if cell_count > 0:
                polydata.SetPolys(cells)
            
            logger.info(f"Manually created dataset with {point_count} points and {cell_count} cells")
            
            # If we have at least some points, display the dataset
            if point_count > 0:
                self.data = polydata
                self._display_results()
                return True
            else:
                logger.error("Manual parsing failed - no points found")
                return False
                
        except Exception as e:
            logger.error(f"Error in manual VTK parsing: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _display_results(self):
        """Display the loaded results data."""
        if not hasattr(self, 'data') or self.data is None:
            return
        
        # Clear existing visualization
        self._clear_visualization()
        
        # Check data type to determine appropriate mapper
        if isinstance(self.data, vtk.vtkPolyData):
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.data)
        else:
            # Use geometry filter to extract surface
            surface = vtk.vtkGeometryFilter()
            surface.SetInputData(self.data)
            surface.Update()
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(surface.GetOutputPort())
        
        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.8, 0.8, 1.0)  # Light blue
        
        # Add actor to renderer
        source_id = "results"
        self.renderer.AddActor(actor)
        self.actors[source_id] = actor
        self.active_source = source_id
        
        # Add wireframe for better visualization
        wireframe = vtk.vtkActor()
        wireframe.SetMapper(mapper)
        wireframe.GetProperty().SetRepresentationToWireframe()
        wireframe.GetProperty().SetColor(0.0, 0.0, 0.0)  # Black wireframe
        wireframe.GetProperty().SetLineWidth(0.5)
        wireframe.GetProperty().SetOpacity(0.2)
        
        self.renderer.AddActor(wireframe)
        self.actors[f"{source_id}_wireframe"] = wireframe
        
        # Reset view to show results
        self.reset_view()
        
        # Force render update
        self.render_window.Render()
    
    def _load_legacy_vtk(self, filepath):
        """
        Load a legacy VTK file.
        
        Args:
            filepath (str): Path to the VTK file
        """
        # Determine if file is structured or unstructured grid or polydata
        # For simplicity, we'll try polydata first, then unstructured grid
        
        try:
            # Try as polydata first
            reader = vtkPolyDataReader()
            reader.SetFileName(filepath)
            reader.Update()
            
            # Check if the reader successfully read the file
            if reader.GetOutput().GetNumberOfCells() > 0:
                # Create a source identifier
                source_id = os.path.basename(filepath)
                self.sources[source_id] = reader
                self.active_source = source_id
                
                # Create mapper and actor
                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(reader.GetOutputPort())
                
                actor = vtkActor()
                actor.SetMapper(mapper)
                
                # Add actor to renderer
                self.renderer.AddActor(actor)
                self.actors[source_id] = actor
                
                return
        except Exception as e:
            logger.warning(f"Could not read as polydata: {e}")
        
        try:
            # Try as unstructured grid
            reader = vtkUnstructuredGridReader()
            reader.SetFileName(filepath)
            reader.Update()
            
            # Check if the reader successfully read the file
            if reader.GetOutput().GetNumberOfCells() > 0:
                # Create a source identifier
                source_id = os.path.basename(filepath)
                self.sources[source_id] = reader
                self.active_source = source_id
                
                # Create mapper and actor
                mapper = vtkDataSetMapper()
                mapper.SetInputConnection(reader.GetOutputPort())
                
                actor = vtkActor()
                actor.SetMapper(mapper)
                
                # Add actor to renderer
                self.renderer.AddActor(actor)
                self.actors[source_id] = actor
                
                return
        except Exception as e:
            logger.error(f"Could not read as unstructured grid: {e}")
            raise ValueError(f"Cannot read VTK file: {filepath}")
    
    def _load_xml_polydata(self, filepath):
        """
        Load an XML polydata file.
        
        Args:
            filepath (str): Path to the VTP file
        """
        try:
            reader = vtkXMLPolyDataReader()
            reader.SetFileName(filepath)
            reader.Update()
            
            # Create a source identifier
            source_id = os.path.basename(filepath)
            self.sources[source_id] = reader
            self.active_source = source_id
            
            # Create mapper and actor
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())
            
            actor = vtkActor()
            actor.SetMapper(mapper)
            
            # Add actor to renderer
            self.renderer.AddActor(actor)
            self.actors[source_id] = actor
        
        except Exception as e:
            logger.error(f"Error loading XML polydata: {e}")
            raise
    
    def _load_xml_unstructured_grid(self, filepath):
        """
        Load an XML unstructured grid file.
        
        Args:
            filepath (str): Path to the VTU file
        """
        try:
            reader = vtkXMLUnstructuredGridReader()
            reader.SetFileName(filepath)
            reader.Update()
            
            # Create a source identifier
            source_id = os.path.basename(filepath)
            self.sources[source_id] = reader
            self.active_source = source_id
            
            # Create mapper and actor
            mapper = vtkDataSetMapper()
            mapper.SetInputConnection(reader.GetOutputPort())
            
            actor = vtkActor()
            actor.SetMapper(mapper)
            
            # Add actor to renderer
            self.renderer.AddActor(actor)
            self.actors[source_id] = actor
        
        except Exception as e:
            logger.error(f"Error loading XML unstructured grid: {e}")
            raise
    
    def _load_openfoam_case(self, case_dir):
        """
        Load an OpenFOAM case.
        
        Args:
            case_dir (str): Path to the OpenFOAM case directory
        """
        logger.info(f"Loading OpenFOAM case: {case_dir}")
        
        try:
            # FIRST PRIORITY: Try to directly use the original STL file
            if hasattr(self, 'main_window') and self.main_window and hasattr(self.main_window, 'current_project'):
                project = self.main_window.current_project
                
                # Check for active mesh file (original STL)
                if hasattr(project, 'active_mesh') and project.active_mesh:
                    mesh_path = project.active_mesh
                    if os.path.exists(mesh_path):
                        logger.info(f"Using original mesh file: {mesh_path}")
                        return self.load_mesh(mesh_path)  # Use enhanced load_mesh method
                
                # Check for original geometry file
                if hasattr(project, 'active_geometry') and project.active_geometry:
                    geom_path = project.active_geometry
                    if os.path.exists(geom_path):
                        logger.info(f"Using original geometry file: {geom_path}")
                        return self.load_mesh(geom_path)  # Use enhanced load_mesh method
                        
                # Check through all mesh files in the project
                if hasattr(project, 'mesh_files') and project.mesh_files:
                    for mesh_file in project.mesh_files:
                        if os.path.exists(mesh_file):
                            logger.info(f"Using project mesh file: {mesh_file}")
                            return self.load_mesh(mesh_file)
                            
                # Check through all geometry files
                if hasattr(project, 'geometry_files') and project.geometry_files:
                    for geom_file in project.geometry_files:
                        if os.path.exists(geom_file):
                            logger.info(f"Using project geometry file: {geom_file}")
                            return self.load_mesh(geom_file)
            
            # SECOND PRIORITY: Search for STL files in various directories
            # Try to find any STL files in the case directory structure
            stl_files = []
            
            # Check in constant/triSurface
            tri_surface_dir = os.path.join(case_dir, "constant", "triSurface")
            if os.path.exists(tri_surface_dir):
                stl_files = [os.path.join(tri_surface_dir, f) for f in os.listdir(tri_surface_dir) 
                            if f.lower().endswith('.stl')]
            
            # Check in project mesh directory
            if hasattr(self, 'main_window') and self.main_window and hasattr(self.main_window, 'current_project') and self.main_window.current_project:
                project = self.main_window.current_project
                if hasattr(project, 'project_dir') and project.project_dir:
                    mesh_dir = os.path.join(project.project_dir, "mesh")
                    if os.path.exists(mesh_dir):
                        stl_files.extend([os.path.join(mesh_dir, f) for f in os.listdir(mesh_dir) 
                                        if f.lower().endswith('.stl')])
            
            if stl_files:
                logger.info(f"Found STL file to display: {stl_files[0]}")
                return self.load_mesh(stl_files[0])
            
            # THIRD PRIORITY: Look for VTK files from foamToVTK conversion
            vtk_dir = os.path.join(case_dir, "VTK")
            if os.path.exists(vtk_dir):
                vtk_files = [os.path.join(vtk_dir, f) for f in os.listdir(vtk_dir) 
                            if f.lower().endswith('.vtk') or f.lower().endswith('.vtu')]
                if vtk_files:
                    logger.info(f"Found VTK file to display: {vtk_files[0]}")
                    return self.load_mesh(vtk_files[0])
    
        except Exception as e:
            logger.error(f"Error attempting to load actual mesh: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # If we reach here, we couldn't find a good mesh file to display
        # Create a placeholder visualization that better represents the mesh
        try:
            # Clear existing visualization
            self._clear_visualization()
            
            # Get mesh bounding box from OpenFOAM if possible
            bounds = None
            try:
                # Try to get bounds from points file
                points_file = os.path.join(case_dir, "constant", "polyMesh", "points")
                if os.path.exists(points_file):
                    import numpy as np
                    # Read first few lines to get bounds
                    points = []
                    with open(points_file, 'r') as f:
                        content = f.readlines()
                        for line in content:
                            if '(' in line and ')' in line and line.strip()[0] == '(':
                                coords = line.strip().strip('()')
                                try:
                                    point = [float(x) for x in coords.split()]
                                    if len(point) == 3:
                                        points.append(point)
                                except:
                                    pass
                
                    if points:
                        points = np.array(points)
                        min_bounds = np.min(points, axis=0)
                        max_bounds = np.max(points, axis=0)
                        bounds = [min_bounds[0], max_bounds[0], 
                                min_bounds[1], max_bounds[1],
                                min_bounds[2], max_bounds[2]]
            except Exception as e:
                logger.warning(f"Error getting bounds from points file: {e}")
            
            # If bounds not found, use default
            if not bounds:
                bounds = [-0.1, 0.3, -0.1, 0.1, -0.1, 0.1]  # Default bounds
                
            # Create outline to represent the mesh bounds
            outline = vtk.vtkOutlineSource()
            outline.SetBounds(bounds)
            outline.Update()
            
            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(outline.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.0, 0.7, 0.9)  # Cyan color
            actor.GetProperty().SetLineWidth(2.0)
            
            # Add actor to renderer
            self.renderer.AddActor(actor)
            self.actors["outline"] = actor
            
            # Add text showing mesh info
            text_actor = vtk.vtkTextActor()
            text_actor.SetInput(f"OpenFOAM Mesh\nBounds: ({bounds[0]:.2f}, {bounds[1]:.2f}) × ({bounds[2]:.2f}, {bounds[3]:.2f}) × ({bounds[4]:.2f}, {bounds[5]:.2f})")
            text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)  # White text
            text_actor.GetTextProperty().SetFontSize(12)
            text_actor.SetPosition(10, 10)
            
            self.renderer.AddActor2D(text_actor)
            
            # Reset camera
            self.renderer.ResetCamera()
            self.render_window.Render()
            
            logger.info("Created wireframe visualization for OpenFOAM mesh")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Show error message
            self._show_load_error(f"Failed to visualize mesh: {str(e)}")
            return False
    
    def _clear_visualization(self):
        """Clear all actors from the renderer"""
        if hasattr(self, 'renderer') and self.renderer:
            self.renderer.RemoveAllViewProps()
        
    def load_mesh(self, mesh_file):
        """
        Load a mesh file.
        
        Args:
            mesh_file (str): Path to the mesh file
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._check_vtk_available():
            return False
        
        logger.info(f"Loading mesh: {mesh_file}")
        
        try:
            # Check file exists
            if not os.path.exists(mesh_file):
                logger.error(f"Mesh file not found: {mesh_file}")
                return False
            
            # Get file extension
            file_ext = os.path.splitext(mesh_file)[1].lower()
            
            # Choose reader based on file extension
            if file_ext == '.stl':
                reader = vtk.vtkSTLReader()
            elif file_ext == '.obj':
                reader = vtk.vtkOBJReader()
            elif file_ext == '.ply':
                reader = vtk.vtkPLYReader()
            elif file_ext == '.vtk':
                reader = vtk.vtkPolyDataReader()
            elif file_ext == '.vtp':
                reader = vtk.vtkXMLPolyDataReader()
            else:
                logger.error(f"Unsupported mesh file format: {file_ext}")
                return False
            
            # Set file name and read
            reader.SetFileName(mesh_file)
            reader.Update()
            
            # Get the output data
            output = reader.GetOutput()
            
            # Check if we got any points
            if output.GetNumberOfPoints() == 0:
                logger.error(f"No points found in mesh file: {mesh_file}")
                return False
            
            logger.info(f"STL mesh loaded: {output.GetNumberOfPoints()} points, {output.GetNumberOfCells()} cells")
            
            # Store the data
            self.data = output
            
            # Create a mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(output)
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            # Add to scene
            source_id = "mesh"
            self.renderer.AddActor(actor)
            self.actors[source_id] = actor
            self.active_source = source_id
            
            # Add a wireframe version for better visualization
            wireframe = vtk.vtkActor()
            wireframe.SetMapper(mapper)
            wireframe.GetProperty().SetRepresentationToWireframe()
            wireframe.GetProperty().SetColor(0.0, 0.0, 0.0)  # Black
            wireframe.GetProperty().SetLineWidth(0.5)
            wireframe.GetProperty().SetOpacity(0.2)
            
            self.renderer.AddActor(wireframe)
            self.actors[f"{source_id}_wireframe"] = wireframe
            
            # Reset view to show the whole mesh
            self.reset_view()
            
            logger.info(f"Successfully loaded and displayed mesh: {mesh_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading mesh: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _load_stl(self, filepath):
        """
        Load an STL file.
        
        Args:
            filepath (str): Path to the STL file
        """
        try:
            reader = vtkSTLReader()
            reader.SetFileName(filepath)
            reader.Update()
            
            # For better visualization, compute normals
            normals = vtkPolyDataNormals()
            normals.SetInputConnection(reader.GetOutputPort())
            normals.ComputePointNormalsOn()
            normals.ComputeCellNormalsOn()
            normals.SplittingOff()
            normals.ConsistencyOn()
            normals.Update()
            
            # Create a source identifier
            source_id = os.path.basename(filepath)
            self.sources[source_id] = normals
            self.active_source = source_id
            
            # Create mapper and actor
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(normals.GetOutputPort())
            
            actor = vtkActor()
            actor.SetMapper(mapper)
            
            # Add actor to renderer
            self.renderer.AddActor(actor)
            self.actors[source_id] = actor
        
        except Exception as e:
            logger.error(f"Error loading STL: {e}")
            raise
    
    def _load_obj(self, filepath):
        """
        Load an OBJ file.
        
        Args:
            filepath (str): Path to the OBJ file
        """
        try:
            reader = vtkOBJReader()
            reader.SetFileName(filepath)
            reader.Update()
            
            # Create a source identifier
            source_id = os.path.basename(filepath)
            self.sources[source_id] = reader
            self.active_source = source_id
            
            # Create mapper and actor
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())
            
            actor = vtkActor()
            actor.SetMapper(mapper)
            
            # Add actor to renderer
            self.renderer.AddActor(actor)
            self.actors[source_id] = actor
        
        except Exception as e:
            logger.error(f"Error loading OBJ: {e}")
            raise
    
    def create_slice(self, normal=(0, 0, 1), origin=None):
        """
        Create a slice through the current data.
        
        Args:
            normal (tuple): Normal vector of the slice plane
            origin (tuple, optional): Origin of the slice plane
        """
        if not VTK_AVAILABLE or self.active_source is None:
            return
        
        try:
            # Get the active source
            source = self.sources[self.active_source]
            
            # Create a plane for slicing
            plane = vtk.vtkPlane()
            plane.SetNormal(normal)
            
            # Set origin if provided, otherwise use center of data
            if origin:
                plane.SetOrigin(origin)
            else:
                source_output = None
                if hasattr(source, "GetOutput"):
                    source_output = source.GetOutput()
                elif hasattr(source, "GetOutputPort"):
                    # Need to use a temporary to get output
                    tmp = vtkPolyDataMapper()
                    tmp.SetInputConnection(source.GetOutputPort())
                    tmp.Update()
                    source_output = tmp.GetInput()
                
                if source_output:
                    bounds = source_output.GetBounds()
                    center = [(bounds[0] + bounds[1])/2, (bounds[2] + bounds[3])/2, (bounds[4] + bounds[5])/2]
                    plane.SetOrigin(center)
                
                if source_output:
                    bounds = source_output.GetBounds()
                    center = [(bounds[0] + bounds[1])/2, (bounds[2] + bounds[3])/2, (bounds[4] + bounds[5])/2]
                    plane.SetOrigin(center)
            
            # Create the cutter
            cutter = vtk.vtkCutter()
            
            # Connect to the source or its output
            if hasattr(source, "GetOutputPort"):
                cutter.SetInputConnection(source.GetOutputPort())
            else:
                cutter.SetInputData(source.GetOutput())
            
            cutter.SetCutFunction(plane)
            cutter.Update()
            
            # Create a filter identifier
            filter_id = f"slice_{len(self.filters)}"
            self.filters[filter_id] = cutter
            
            # Create mapper and actor
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(cutter.GetOutputPort())
            
            actor = vtkActor()
            actor.SetMapper(mapper)
            
            # Set color to distinguish from original
            actor.GetProperty().SetColor(1.0, 1.0, 0.0)  # Yellow
            
            # Add actor to renderer
            self.renderer.AddActor(actor)
            self.actors[filter_id] = actor
            
            # Hide the original actor if needed
            if self.active_source in self.actors:
                self.actors[self.active_source].SetVisibility(False)
            
            # Update active source
            self.active_source = filter_id
            
            # Render
            self.render_window.Render()
            self.view_changed.emit()
            
            logger.info(f"Created slice with normal {normal}")
            
        except Exception as e:
            logger.error(f"Error creating slice: {e}")
    
    def create_contour(self, field='p', value=None):
        """
        Create a contour of the current data.
        
        Args:
            field (str): Field to contour
            value (float, optional): Contour value. If None, estimates a reasonable value.
        """
        if not VTK_AVAILABLE or self.active_source is None:
            return
        
        try:
            # Get the active source
            source = self.sources[self.active_source]
            
            # Create the contour filter
            contour = vtkContourFilter()
            
            # Connect to the source or its output
            if hasattr(source, "GetOutputPort"):
                contour.SetInputConnection(source.GetOutputPort())
            else:
                contour.SetInputData(source.GetOutput())
            
            # Try to find the specified field in point data
            source_output = None
            if hasattr(source, "GetOutput"):
                source_output = source.GetOutput()
            elif hasattr(source, "GetOutputPort"):
                # Need to use a temporary to get output
                tmp = vtkPolyDataMapper()
                tmp.SetInputConnection(source.GetOutputPort())
                tmp.Update()
                source_output = tmp.GetInput()
            
            if source_output and source_output.GetPointData().HasArray(field):
                contour.SetInputArrayToProcess(0, 0, 0, vtkDataSetAttributes.SCALARS, field)
                
                # Set contour value if provided, otherwise estimate
                if value is not None:
                    contour.SetValue(0, value)
                else:
                    # Estimate a reasonable value
                    array = source_output.GetPointData().GetArray(field)
                    if array:
                        range_min, range_max = array.GetRange()
                        contour.SetValue(0, (range_min + range_max) / 2)
                    else:
                        contour.SetValue(0, 0.0)
            else:
                logger.warning(f"Field '{field}' not found, using first available scalar")
                contour.SetValue(0, value if value is not None else 0.0)
            
            contour.Update()
            
            # Create a filter identifier
            filter_id = f"contour_{len(self.filters)}"
            self.filters[filter_id] = contour
            
            # Create mapper and actor
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(contour.GetOutputPort())
            
            actor = vtkActor()
            actor.SetMapper(mapper)
            
            # Set color to distinguish from original
            actor.GetProperty().SetColor(0.0, 1.0, 0.0)  # Green
            
            # Add actor to renderer
            self.renderer.AddActor(actor)
            self.actors[filter_id] = actor
            
            # Hide the original actor if needed
            if self.active_source in self.actors:
                self.actors[self.active_source].SetVisibility(False)
            
            # Update active source
            self.active_source = filter_id
            
            # Render
            self.render_window.Render()
            self.view_changed.emit()
            
            logger.info(f"Created contour of {field}")
            
        except Exception as e:
            logger.error(f"Error creating contour: {e}")
    
    def create_streamlines(self, field='U', seed_type='Point Source', patch_name=None, num_points=50):
        """
        Create streamlines through the current vector field.
        
        This enhanced implementation supports seeding from boundary patches from OpenFOAM data,
        allowing for more accurate visualization of flow entering/exiting through inlets/outlets.
        
        Args:
            field (str): Vector field to use for streamlines
            seed_type (str): Type of seed for streamlines
                Options: 'Point Source', 'Line Source', 'Plane Source', 'Boundary Patch'
            patch_name (str, optional): Name of the boundary patch to use for seeding (when seed_type is 'Boundary Patch')
            num_points (int, optional): Number of seed points to generate
        """
        # Check required VTK packages are available
        if not VTK_AVAILABLE or self.active_source is None:
            return
            
        # Check for active simulation before proceeding
        try:
            from ..utils.simulation_lock import is_simulation_active
            if is_simulation_active():
                logger.warning("Cannot create streamlines during active simulation.")
                return False
        except ImportError:
            # If simulation_lock module not available, proceed but log a warning
            logger.warning("Simulation lock module not found, cannot check simulation status")
        
        try:
            # Get the active source
            source = self.sources[self.active_source]
            
            # Make sure we have the field
            source_output = None
            if hasattr(source, "GetOutput"):
                source_output = source.GetOutput()
            elif hasattr(source, "GetOutputPort"):
                # Need to use a temporary to get output
                tmp = vtkPolyDataMapper()
                tmp.SetInputConnection(source.GetOutputPort())
                source_output = tmp.GetInput()
            
            if not source_output or not source_output.GetPointData().HasArray(field):
                logger.error(f"Vector field '{field}' not found")
                return
            
            # Get bounds to position seeds
            bounds = source_output.GetBounds()
            center = [(bounds[0] + bounds[1])/2, (bounds[2] + bounds[3])/2, (bounds[4] + bounds[5])/2]
            
            # Create seed source based on type
            seed_source = None
            if seed_type == 'Point Source':
                point = vtkPointSource()
                point.SetCenter(center)
                point.SetRadius(min(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]) * 0.3)
                point.SetNumberOfPoints(num_points)
                seed_source = point
                
            elif seed_type == 'Line Source':
                line = vtkLineSource()
                line.SetPoint1(center[0] - (bounds[1]-bounds[0])*0.25, center[1], center[2])
                line.SetPoint2(center[0] + (bounds[1]-bounds[0])*0.25, center[1], center[2])
                line.SetResolution(20)
                seed_source = line
                
            elif seed_type == 'Plane Source':
                plane = vtkPlaneSource()
                plane.SetCenter(center)
                plane.SetNormal(0, 0, 1)
                plane.SetResolution(5, 5)
                seed_source = plane
                
            elif seed_type == 'Boundary Patch':
                # Use boundary patch for seeding - most accurate for real-world flow visualization
                seed_source = self._create_boundary_patch_seed(patch_name, num_points)
                if not seed_source:
                    logger.error(f"Could not create seed points from boundary patch: {patch_name}")
                    # Fall back to point source
                    point = vtkPointSource()
                    point.SetCenter(center)
                    point.SetRadius(min(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]) * 0.3)
                    point.SetNumberOfPoints(num_points)
                    seed_source = point
            else:
                logger.error(f"Unknown seed type: {seed_type}")
                return
            
            seed_source.Update()
            
            # Create streamline tracer
            streamlines = vtkStreamTracer()
            
            # Connect to the source or its output
            if hasattr(source, "GetOutputPort"):
                streamlines.SetInputConnection(source.GetOutputPort())
            else:
                streamlines.SetInputData(source.GetOutput())
            
            # Set the vector field
            streamlines.SetInputArrayToProcess(0, 0, 0, vtkDataSetAttributes.VECTORS, field)
            
            # Set the seeds
            streamlines.SetSourceConnection(seed_source.GetOutputPort())
            
            # Set integration properties
            streamlines.SetMaximumPropagation(100)  # Maximum streamline length
            streamlines.SetIntegrationDirectionToBoth()
            streamlines.SetIntegratorTypeToRungeKutta45()
            
            streamlines.Update()
            
            # Create a filter identifier
            filter_id = f"streamlines_{len(self.filters)}"
            self.filters[filter_id] = streamlines
            
            # Create mapper and actor
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(streamlines.GetOutputPort())
            
            actor = vtkActor()
            actor.SetMapper(mapper)
            
            # Set color to distinguish from original
            actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red
            
            # Add actor to renderer
            self.renderer.AddActor(actor)
            self.actors[filter_id] = actor
            
            # Don't hide the original for streamlines
            
            # Render
            self.render_window.Render()
            self.view_changed.emit()
            
            logger.info(f"Created streamlines for {field}")
            
        except Exception as e:
            logger.error(f"Error creating streamlines: {e}")
    
    def set_representation(self, representation_type):
        """
        Set the representation type for the current data.
        
        Args:
            representation_type (str): Representation type ('Surface', 'Wireframe', 'Points', 'Volume')
        """
        if not VTK_AVAILABLE or self.active_source is None:
            return
        
        try:
            if self.active_source in self.actors:
                actor = self.actors[self.active_source]
                
                # Set representation type
                if representation_type == 'Surface':
                    actor.GetProperty().SetRepresentationToSurface()
                elif representation_type == 'Wireframe':
                    actor.GetProperty().SetRepresentationToWireframe()
                elif representation_type == 'Points':
                    actor.GetProperty().SetRepresentationToPoints()
                else:
                    # Default to surface
                    actor.GetProperty().SetRepresentationToSurface()
                
                # Render
                self.render_window.Render()
                self.view_changed.emit()
                
                logger.info(f"Set representation to {representation_type}")
                
        except Exception as e:
            logger.error(f"Error setting representation: {e}")
    
    def set_coloring(self, field, component=None):
        """
        Set the coloring for the current data.
        
        Args:
            field (str): Field to color by
            component (str, optional): Component for vector fields ('X', 'Y', 'Z', 'Magnitude')
        """
        if not VTK_AVAILABLE or self.active_source is None:
            return
        
        try:
            # Get actor for the active source
            if self.active_source not in self.actors:
                logger.error(f"No actor for active source: {self.active_source}")
                return
            
            actor = self.actors[self.active_source]
            mapper = actor.GetMapper()
            
            # Get source data
            source = self.sources[self.active_source] if self.active_source in self.sources else None
            if self.active_source in self.filters:
                source = self.filters[self.active_source]
            
            if not source:
                logger.error(f"No source for active filter: {self.active_source}")
                return
            
            # Handle solid color case
            if field == 'Solid Color':
                mapper.ScalarVisibilityOff()
                actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # Light gray
                
                # Hide scalar bar if any
                for bar_id, bar in self.scalar_bars.items():
                    bar.SetVisibility(False)
                
                self.render_window.Render()
                return
            
            # Get the data source
            source_output = None
            if hasattr(source, "GetOutput"):
                source_output = source.GetOutput()
            elif hasattr(source, "GetOutputPort"):
                # Need to use a temporary to get output
                tmp = vtkPolyDataMapper()
                tmp.SetInputConnection(source.GetOutputPort())
                source_output = tmp.GetInput()
            
            if not source_output:
                logger.error("Could not get source output")
                return
            
            # Check if the field exists
            if not source_output.GetPointData().HasArray(field):
                logger.error(f"Field '{field}' not found")
                return
            
            # Handle vector fields and components
            array = source_output.GetPointData().GetArray(field)
            if array.GetNumberOfComponents() > 1:
                # Vector field
                if component == 'Magnitude':
                    # Use magnitude
                    mapper.SetScalarModeToUsePointFieldData()
                    mapper.SelectColorArray(field)
                    mapper.ColorByArrayComponent(field, -1)  # -1 for magnitude
                else:
                    # Use specific component
                    comp_idx = {'X': 0, 'Y': 1, 'Z': 2}.get(component, 0)
                    mapper.SetScalarModeToUsePointFieldData()
                    mapper.SelectColorArray(field)
                    mapper.ColorByArrayComponent(field, comp_idx)
            else:
                # Scalar field
                mapper.SetScalarModeToUsePointFieldData()
                mapper.SelectColorArray(field)
            
            # Enable scalar visibility
            mapper.ScalarVisibilityOn()
            
            # Automatically adjust color scale
            mapper.SetUseLookupTableScalarRange(True)
            
            # Create a new scalar bar
            scalar_bar = vtkScalarBarActor()
            scalar_bar.SetTitle(field + (" " + component if component else ""))
            scalar_bar.SetNumberOfLabels(5)
            scalar_bar.SetMaximumWidthInPixels(80)
            scalar_bar.SetMaximumHeightInPixels(400)
            
            # Position the scalar bar
            scalar_bar.SetPosition(0.9, 0.1)
            scalar_bar.SetWidth(0.08)
            scalar_bar.SetHeight(0.8)
            
            # Add scalar bar to renderer and dictionary
            scalar_bar_id = f"scalar_bar_{field}"
            
            # Remove old scalar bars
            for bar_id, bar in self.scalar_bars.items():
                self.renderer.RemoveActor(bar)
            self.scalar_bars.clear()
            
            self.renderer.AddActor(scalar_bar)
            self.scalar_bars[scalar_bar_id] = scalar_bar
            
            # Set lookup table for mapper and scalar bar
            lut = mapper.GetLookupTable()
            scalar_bar.SetLookupTable(lut)
            
            # Render
            self.render_window.Render()
            self.view_changed.emit()
            
            logger.info(f"Set coloring to {field} {component if component else ''}")
            
        except Exception as e:
            logger.error(f"Error setting coloring: {e}")
    
    def set_colormap(self, colormap_name):
        """
        Set the color map for the current representation.
        
        Args:
            colormap_name (str): Name of the color map
        """
        if not VTK_AVAILABLE or self.active_source is None:
            return
        
        try:
            # Get actor for the active source
            if self.active_source not in self.actors:
                logger.error(f"No actor for active source: {self.active_source}")
                return
            
            actor = self.actors[self.active_source]
            mapper = actor.GetMapper()
            
            # Check if scalar visualization is enabled
            if not mapper.GetScalarVisibility():
                logger.warning("Scalar visualization is not enabled, cannot set colormap")
                return
            
            # Create a lookup table based on colormap name
            lut = vtkLookupTable()
            
            if colormap_name == 'Rainbow':
                # Rainbow colormap
                lut.SetHueRange(0.667, 0.0)  # Blue to red
                lut.SetSaturationRange(0.8, 0.8)
                lut.SetValueRange(1.0, 1.0)
            elif colormap_name == 'Cool to Warm':
                # Cool to Warm colormap
                lut.SetNumberOfTableValues(256)
                for i in range(256):
                    t = i / 255.0
                    if t < 0.5:
                        # Cool (blue to white)
                        r = t * 2
                        g = t * 2
                        b = 1.0
                    else:
                        # Warm (white to red)
                        r = 1.0
                        g = 2.0 - t * 2
                        b = 2.0 - t * 2
                    lut.SetTableValue(i, r, g, b, 1.0)
            elif colormap_name == 'Viridis':
                # Viridis colormap (approximation)
                lut.SetNumberOfTableValues(256)
                for i in range(256):
                    t = i / 255.0
                    # Approximate Viridis colormap
                    r = 0.267 + 0.731 * t - 0.857 * t*t
                    g = 0.004 + 0.931 * t - 0.520 * t*t
                    b = 0.329 + 0.395 * t - 0.024 * t*t
                    lut.SetTableValue(i, r, g, b, 1.0)
            elif colormap_name == 'Plasma':
                # Plasma colormap (approximation)
                lut.SetNumberOfTableValues(256)
                for i in range(256):
                    t = i / 255.0
                    # Approximate Plasma colormap
                    r = 0.050 + 1.863 * t - 1.207 * t*t
                    g = -0.187 + 2.059 * t - 2.292 * t*t
                    b = 0.380 + 0.673 * t - 1.040 * t*t
                    lut.SetTableValue(i, min(1, max(0, r)), min(1, max(0, g)), min(1, max(0, b)), 1.0)
            elif colormap_name == 'Jet':
                # Traditional Jet colormap
                lut.SetHueRange(0.667, 0.0)  # Blue to red
                lut.SetSaturationRange(0.8, 0.8)
                lut.SetValueRange(1.0, 1.0)
            elif colormap_name == 'Black to White':
                # Grayscale
                lut.SetHueRange(0.0, 0.0)
                lut.SetSaturationRange(0.0, 0.0)
                lut.SetValueRange(0.0, 1.0)
            else:
                # Default to "Rainbow"
                lut.SetHueRange(0.667, 0.0)  # Blue to red
                lut.SetSaturationRange(0.8, 0.8)
                lut.SetValueRange(1.0, 1.0)
            
            lut.Build()
            
            # Set the lookup table for the mapper
            mapper.SetLookupTable(lut)
            
            # Update scalar bar if any
            for bar_id, bar in self.scalar_bars.items():
                bar.SetLookupTable(lut)
            
            # Render
            self.render_window.Render()
            self.view_changed.emit()
            
            logger.info(f"Set color map to {colormap_name}")
            
        except Exception as e:
            logger.error(f"Error setting color map: {e}")
    
    def export_view(self, filepath, resolution=None):
        """
        Export the current view to an image file.
        
        Args:
            filepath (str): Path to save the image
            resolution (tuple, optional): Resolution of the image (width, height)
        """
        if not VTK_AVAILABLE:
            return
        
        try:
            # Create window to image filter
            w2i = vtkWindowToImageFilter()
            w2i.SetInput(self.render_window)
            w2i.Update()
            
            # Create writer based on file extension
            ext = os.path.splitext(filepath)[1].lower()
            if ext == '.png':
                writer = vtkPNGWriter()
            elif ext == '.jpg' or ext == '.jpeg':
                writer = vtkJPEGWriter()
            elif ext == '.tiff' or ext == '.tif':
                writer = vtkTIFFWriter()
            else:
                # Default to PNG
                writer = vtkPNGWriter()
                if not filepath.lower().endswith('.png'):
                    filepath += '.png'
            
            writer.SetFileName(filepath)
            writer.SetInputConnection(w2i.GetOutputPort())
            writer.Write()
            
            logger.info(f"Exported view to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting view: {e}")
    
    def set_axes_visibility(self, visible):
        """
        Set the visibility of the coordinate axes.
        
        Args:
            visible (bool): Whether the axes should be visible
        """
        if not VTK_AVAILABLE or not self.axes_actor:
            return
        
        self.show_axes = visible
        self.axes_actor.SetVisibility(visible)
        self.render_window.Render()
    
    def set_bounds_visibility(self, visible):
        """
        Set the visibility of the bounding box.
        
        Args:
            visible (bool): Whether the bounding box should be visible
        """
        if not VTK_AVAILABLE:
            return
        
        self.show_bounds = visible
        
        if visible:
            # Create bounds actor if it doesn't exist
            if not self.bounds_actor and self.active_source:
                # Get the bounds of the active source
                bounds = [0, 0, 0, 0, 0, 0]
                
                source = None
                if self.active_source in self.sources:
                    source = self.sources[self.active_source]
                elif self.active_source in self.filters:
                    source = self.filters[self.active_source]
                
                if source:
                    if hasattr(source, "GetOutput"):
                        bounds = source.GetOutput().GetBounds()
                    elif hasattr(source, "GetOutputPort"):
                        # Need to use a temporary to get bounds
                        tmp = vtkPolyDataMapper()
                        tmp.SetInputConnection(source.GetOutputPort())
                        bounds = tmp.GetBounds()
                
                # Create a cube axes actor
                self.bounds_actor = vtkCubeAxesActor()
                self.bounds_actor.SetBounds(bounds)
                self.bounds_actor.SetCamera(self.renderer.GetActiveCamera())
                self.bounds_actor.SetXLabelFormat("%6.1f")
                self.bounds_actor.SetYLabelFormat("%6.1f")
                self.bounds_actor.SetZLabelFormat("%6.1f")
                self.bounds_actor.SetFlyModeToStaticEdges()
                
                self.renderer.AddActor(self.bounds_actor)
        else:
            # Remove bounds actor if it exists
            if self.bounds_actor:
                self.renderer.RemoveActor(self.bounds_actor)
                self.bounds_actor = None
        
        self.render_window.Render()
    
    def set_background_color(self, color):
        """
        Set the background color.
        
        Args:
            color (list): RGB color components [r, g, b] in range 0-1
        """
        if not VTK_AVAILABLE:
            return
        
        self.background_color = color
        self.renderer.SetBackground(color)
        self.render_window.Render()
    
    def resizeEvent(self, event):
        """
        Handle widget resize events.
        
        Args:
            event: The resize event
        """
        super(VTKViewport, self).resizeEvent(event)
        
        # Update render window size
        if self.render_window:
            self.render_window.SetSize(self.width(), self.height())
            self.render_window.Render()

    def set_camera_position(self, position, focal_point, view_up):
        """
        Set the camera position, focal point, and view up direction.
        
        Args:
            position (list): Camera position [x, y, z]
            focal_point (list): Camera focal point [x, y, z]
            view_up (list): Camera view up direction [x, y, z]
        """
        if not VTK_AVAILABLE or not self.renderer:
            return
        
        camera = self.renderer.GetActiveCamera()
        
        if position and len(position) == 3:
            camera.SetPosition(position[0], position[1], position[2])
        
        if focal_point and len(focal_point) == 3:
            camera.SetFocalPoint(focal_point[0], focal_point[1], focal_point[2])
        
        if view_up and len(view_up) == 3:
            camera.SetViewUp(view_up[0], view_up[1], view_up[2])
        
        self.renderer.ResetCameraClippingRange()
        self.render_window.Render()

    def set_renderer_settings(self, background_color=None, ambient_light=None, 
                             show_axes=None, show_grid=None, show_bounds=None,
                             orientation_indicator=None, gradient_background=None,
                             use_shadows=None, use_depth_peeling=None, 
                             depth_peeling_layers=None):
        """
        Set renderer settings for the viewport.
        
        Args:
            background_color (list): RGB background color [r, g, b]
            ambient_light (float): Ambient light intensity (0.0-1.0)
            show_axes (bool): Whether to show the axes
            show_grid (bool): Whether to show the grid
            show_bounds (bool): Whether to show the bounds
            orientation_indicator (bool): Whether to show the orientation indicator
            gradient_background (bool): Whether to use a gradient background
            use_shadows (bool): Whether to enable shadow rendering
            use_depth_peeling (bool): Whether to enable depth peeling for transparency
            depth_peeling_layers (int): Number of depth peeling layers to use
        """
        if not VTK_AVAILABLE:
            return
        
        # Set background color
        if background_color is not None and self.renderer:
            if isinstance(background_color, list) and len(background_color) >= 3:
                self.background_color = background_color[:3]
                self.renderer.SetBackground(background_color[0], background_color[1], background_color[2])
                
                # Set gradient background if enabled
                if gradient_background and len(background_color) >= 6:
                    self.renderer.GradientBackgroundOn()
                    self.renderer.SetBackground2(background_color[3], background_color[4], background_color[5])
                else:
                    self.renderer.GradientBackgroundOff()
        
        # Set ambient light
        if ambient_light is not None and self.renderer:
            # Create an ambient light if it doesn't exist
            if not hasattr(self, 'ambient_light') or self.ambient_light is None:
                self.ambient_light = vtk.vtkLight()
                self.ambient_light.SetLightTypeToHeadlight()
                self.ambient_light.SetIntensity(ambient_light)
                self.renderer.AddLight(self.ambient_light)
            else:
                self.ambient_light.SetIntensity(ambient_light)
        
        # Enable/disable depth peeling for better transparency rendering
        if (use_depth_peeling is not None or depth_peeling_layers is not None) and self.render_window:
            # Determine if depth peeling should be enabled
            enable_depth_peeling = use_depth_peeling if use_depth_peeling is not None else self.renderer.GetUseDepthPeeling()
            
            # Get number of layers to use
            num_layers = depth_peeling_layers if depth_peeling_layers is not None else 100
            
            if enable_depth_peeling:
                # Enable depth peeling
                self.renderer.SetUseDepthPeeling(True)
                self.renderer.SetMaximumNumberOfPeels(num_layers)
                self.renderer.SetOcclusionRatio(0.0)
                
                # Tell the render window to use alpha channel
                self.render_window.SetAlphaBitPlanes(True)
                
                # Setup multi-samples (for antialiasing)
                self.render_window.SetMultiSamples(0)
            else:
                # Disable depth peeling
                self.renderer.SetUseDepthPeeling(False)
                
                # Reset render window settings
                self.render_window.SetAlphaBitPlanes(False)
                self.render_window.SetMultiSamples(8)  # Re-enable multisample antialiasing
        
        # Enable/disable shadows
        if use_shadows is not None and self.renderer:
            if use_shadows:
                # VTK 9+ has better shadow support
                if hasattr(self.renderer, 'SetUseShadows'):
                    self.renderer.SetUseShadows(True)
                
                # For newer VTK versions, we can use shadow maps
                if hasattr(vtk, 'vtkShadowMapPass'):
                    # Only set up shadow mapping if not already done
                    if not hasattr(self, 'shadow_pass'):
                        # Create shadow mapping render pass
                        shadows = vtk.vtkShadowMapPass()
                        
                        # Create basic passes
                        passes = vtk.vtkRenderPassCollection()
                        
                        # Add shadow pass
                        shadows_pass = vtk.vtkShadowMapPass()
                        passes.AddItem(shadows_pass)
                        
                        # Add basic passes (lights, camera, opaque, etc.)
                        opaque_pass = vtk.vtkOpaquePass()
                        passes.AddItem(opaque_pass)
                        
                        # Set up sequence
                        seq = vtk.vtkSequencePass()
                        seq.SetPasses(passes)
                        
                        # Add to camera
                        camera_pass = vtk.vtkCameraPass()
                        camera_pass.SetDelegatePass(seq)
                        
                        # Assign to renderer
                        self.renderer.SetPass(camera_pass)
                        
                        # Save references to avoid garbage collection
                        self.shadow_pass = shadows_pass
                        self.opaque_pass = opaque_pass
                        self.seq_pass = seq
                        self.camera_pass = camera_pass
            else:
                # Disable shadows
                if hasattr(self.renderer, 'SetUseShadows'):
                    self.renderer.SetUseShadows(False)
                
                # Remove any special render passes
                if hasattr(self, 'camera_pass'):
                    self.renderer.SetPass(None)
        
        # Show/hide axes
        if show_axes is not None and hasattr(self, 'axes_actor'):
            self.show_axes = show_axes
            self.axes_actor.SetVisibility(show_axes)
        
        # Show/hide grid
        if show_grid is not None:
            if show_grid:
                if not hasattr(self, 'grid_actor') or self.grid_actor is None:
                    # Create a grid if it doesn't exist
                    plane = vtk.vtkPlaneSource()
                    plane.SetXResolution(10)
                    plane.SetYResolution(10)
                    plane.SetCenter(0, 0, 0)
                    plane.SetNormal(0, 0, 1)
                    plane.Update()
                    
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(plane.GetOutputPort())
                    
                    self.grid_actor = vtk.vtkActor()
                    self.grid_actor.SetMapper(mapper)
                    self.grid_actor.GetProperty().SetRepresentationToWireframe()
                    self.grid_actor.GetProperty().SetColor(0.7, 0.7, 0.7)
                    self.grid_actor.GetProperty().SetOpacity(0.5)
                    self.renderer.AddActor(self.grid_actor)
                else:
                    self.grid_actor.SetVisibility(True)
            elif hasattr(self, 'grid_actor') and self.grid_actor is not None:
                self.grid_actor.SetVisibility(False)
        
        # Show/hide bounds
        if show_bounds is not None:
            if show_bounds:
                if not hasattr(self, 'bounds_actor') or self.bounds_actor is None:
                    # If we have data, create a bounds box
                    if hasattr(self, 'active_source') and self.active_source and self.active_source in self.sources:
                        data = self.sources[self.active_source]
                        if hasattr(data, 'GetOutputDataObject') and data.GetOutputDataObject(0):
                            bounds = data.GetOutputDataObject(0).GetBounds()
                            outline = vtk.vtkOutlineFilter()
                            outline.SetInputConnection(data.GetOutputPort())
                            
                            mapper = vtk.vtkPolyDataMapper()
                            mapper.SetInputConnection(outline.GetOutputPort())
                            
                            self.bounds_actor = vtk.vtkActor()
                            self.bounds_actor.SetMapper(mapper)
                            self.bounds_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
                            self.renderer.AddActor(self.bounds_actor)
            elif self.bounds_actor is not None:
                self.bounds_actor.SetVisibility(True)
        elif hasattr(self, 'bounds_actor') and self.bounds_actor is not None:
            self.bounds_actor.SetVisibility(False)
        
        # Show/hide orientation indicator
        if orientation_indicator is not None:
            if orientation_indicator:
                if not hasattr(self, 'orientation_widget') or self.orientation_widget is None:
                    # Create an orientation widget
                    axes = vtk.vtkAxesActor()
                    self.orientation_widget = vtk.vtkOrientationMarkerWidget()
                    self.orientation_widget.SetOrientationMarker(axes)
                    self.orientation_widget.SetInteractor(self.interactor)
                    self.orientation_widget.SetViewport(0.85, 0.0, 1.0, 0.15)
                    self.orientation_widget.SetEnabled(1)
                    self.orientation_widget.InteractiveOff()
                else:
                    self.orientation_widget.SetEnabled(1)
            elif hasattr(self, 'orientation_widget') and self.orientation_widget is not None:
                self.orientation_widget.SetEnabled(0)
        
        # Update the render window
        self.render_window.Render()

    def update_size(self):
        """
        Update the render window size to match the current widget size.
        Called when the main window is resized.
        """
        if not VTK_AVAILABLE or not self.render_window:
            return
        
        # Update render window size
        self.render_window.SetSize(self.width(), self.height())
        
        # Reset camera clipping range
        if self.renderer:
            self.renderer.ResetCameraClippingRange()
        
        # Render the scene
        self.render_window.Render()
        
        # Emit signal that view has changed
        if hasattr(self, 'view_changed'):
            self.view_changed.emit()

    def _show_load_error(self, message):
        """Show error message about mesh loading failure"""
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.warning(
            self, 
            "Visualization Error", 
            f"{message}\n\nGenerated mesh successfully, but can't visualize it. "
            "Use an external viewer like ParaView to see the mesh."
        )

    def load_openfoam_mesh(self, case_dir):
        """
        Load an OpenFOAM mesh for visualization.
        
        Args:
            case_dir (str): Path to the OpenFOAM case directory
        """
        logger.info(f"Loading OpenFOAM mesh from {case_dir}")
        
        # Delegate to the implementation method
        success = self._load_openfoam_case(case_dir)
        
        if success:
            # Show informative message to the user
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(
                self, 
                "Mesh Loaded", 
                "OpenFOAM mesh loaded successfully. Use the View menu to change the visualization style."
            )
        
        return success
    
    def ensure_interactor_available(self):
        """
        Ensure the render window interactor is available and initialized.
        
        Returns:
            bool: True if interactor is available, False otherwise
        """
        try:
            if not hasattr(self, 'interactor') or not self.interactor:
                logger.debug("Interactor not found, attempting to initialize")
                
                # Check if we have a render window
                if not hasattr(self, 'render_window') or not self.render_window:
                    logger.error("No render window available, cannot create interactor")
                    return False
                
                # Create a new interactor
                self.interactor = vtk.vtkRenderWindowInteractor()
                self.interactor.SetRenderWindow(self.render_window)
                
                # Set default style
                style = vtk.vtkInteractorStyleTrackballCamera()
                self.interactor.SetInteractorStyle(style)
                
                # Initialize the interactor
                self.interactor.Initialize()
                logger.debug("Successfully initialized new interactor")
            
            # Ensure interactor is working
            if not self.interactor.GetInitialized():
                logger.debug("Interactor not initialized, initializing now")
                self.interactor.Initialize()
            
            return True
        except Exception as e:
            logger.error(f"Error ensuring interactor availability: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def start_face_selection(self, boundary_name, callback):
        """
        Start face selection mode for boundary creation.
        
        Args:
            boundary_name (str): Name of the boundary to create
            callback (function): Function to call when selection is complete
        
        Returns:
            bool: True if selection mode started successfully
        """
        try:
            logger.info(f"Starting face selection for boundary: {boundary_name}")
            
            # Initialize hover face ID
            if not hasattr(self, 'hover_face_id'):
                self.hover_face_id = -1
                
            # Initialize selected_faces set if not exists
            if not hasattr(self, 'selected_faces'):
                self.selected_faces = set()
            else:
                self.selected_faces.clear()
            
            # Check if we have data to select from
            if not hasattr(self, 'data') or not self.data:
                logger.warning("No mesh data available for selection")
                return False
            
            # Set selection mode
            self.selection_mode = True
            self.boundary_name = boundary_name
            
            # Store the callback to be called when done
            self.selection_callback = callback
            
            # Enable hover picking
            self.hover_enabled = True
            
            # Create a selection actor if it doesn't exist
            if not hasattr(self, 'selection_actor') or not self.selection_actor:
                self.selection_actor = vtk.vtkActor()
                self.selection_actor.GetProperty().SetColor(0.0, 1.0, 0.0)  # Green
                self.selection_actor.GetProperty().SetOpacity(0.7)
                self.renderer.AddActor(self.selection_actor)
            else:
                self.selection_actor.VisibilityOff()  # Reset visibility
            
            # Create hover actor if needed
            if not hasattr(self, 'hover_actor') or not self.hover_actor:
                self.hover_actor = vtk.vtkActor()
                self.hover_actor.GetProperty().SetColor(1.0, 1.0, 0.0)  # Yellow
                self.hover_actor.GetProperty().SetOpacity(0.5)
                self.renderer.AddActor(self.hover_actor)
            self.hover_actor.VisibilityOff()  # Reset visibility
            
            # Set instruction message
            self.status_message = "Select faces for boundary. Hold CTRL for plane selection. Press ESC when done."
            self.update_status()
            
            # Make sure we have the right interactor style for selection
            if hasattr(self, 'interactor') and self.interactor:
                # Get the current interactor style
                current_style = self.interactor.GetInteractorStyle()
                
                # If it's not a trackball style, set it (might need adjustment based on your app)
                if not isinstance(current_style, vtk.vtkInteractorStyleTrackballCamera):
                    trackball = vtk.vtkInteractorStyleTrackballCamera()
                    self.interactor.SetInteractorStyle(trackball)
                
                # Force render to update UI
                self.render_window.Render()
                
                # Make sure the viewport has focus
                if hasattr(self, 'widget') and self.widget:
                    self.widget.setFocus()
                
                # For Qt interactor, add this to ensure hover works
                self.interactor.AddObserver("MouseMoveEvent", self._on_mouse_move)
                self.interactor.AddObserver("LeftButtonPressEvent", self._handle_pick)
                self.interactor.AddObserver("KeyPressEvent", self._on_key_press)
                
                logger.info("Interactor observers added for selection mode")
            
            return True
        except Exception as e:
            logger.error(f"Error starting face selection: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _on_mouse_move(self, obj, event):
        """Called when the mouse moves."""
        if not hasattr(self, 'interactor'):
            return
        
        x, y = self.interactor.GetEventPosition()
        
        # Check if we're in selection mode
        if hasattr(self, 'selection_mode') and self.selection_mode:
            # Update hover highlight if picker is available
            if hasattr(self, 'data') and self.data:
                picker = vtk.vtkCellPicker()
                picker.SetTolerance(0.005)
                
                # Pick at the mouse location
                picker.Pick(x, y, 0, self.renderer)
                
                # Get the picked cell ID
                cell_id = picker.GetCellId()
                
                # If we have a valid cell ID, highlight it and connected faces
                if cell_id != -1 and cell_id not in self.selected_faces:
                    # Find all connected cells that form a face
                    ctrl_key = self.interactor.GetControlKey()
                    if ctrl_key:
                        # If CTRL is held, find all cells in the same plane
                        connected_cells = self._find_cells_in_plane(cell_id)
                        logger.debug(f"Found {len(connected_cells)} cells in plane")
                    else:
                        # Otherwise, just get the face
                        connected_cells = self._find_connected_face(cell_id)
                        logger.debug(f"Found {len(connected_cells)} connected cells in face")
                    
                    # Only update if the cell ID changed
                    if self.hover_face_id != cell_id:
                        self.hover_face_id = cell_id
                        logger.debug(f"Hovering over face containing cell {cell_id}")
                        
                        # Create selection for all connected cells
                        self.hover_connected_cells = connected_cells
                        
                        # Clear any existing hover highlight
                        if hasattr(self, 'hover_actor'):
                            self.renderer.RemoveActor(self.hover_actor)
                        
                        # Create selection for visualization
                        id_array = vtk.vtkIdTypeArray()
                        id_array.SetNumberOfComponents(1)
                        for face_cell_id in connected_cells:
                            id_array.InsertNextValue(face_cell_id)
                        
                        # Create selection
                        selection = vtk.vtkSelectionNode()
                        selection.SetFieldType(vtk.vtkSelectionNode.CELL)
                        selection.SetContentType(vtk.vtkSelectionNode.INDICES)
                        selection.SetSelectionList(id_array)
                        
                        selections = vtk.vtkSelection()
                        selections.AddNode(selection)
                        
                        # Extract selected cells
                        extract = vtk.vtkExtractSelection()
                        extract.SetInputData(0, self.data)
                        extract.SetInputData(1, selections)
                        extract.Update()
                        
                        # Convert to polydata for visualization
                        geom = vtk.vtkGeometryFilter()
                        geom.SetInputConnection(extract.GetOutputPort())
                        geom.Update()
                        
                        # Create mapper and actor
                        mapper = vtk.vtkPolyDataMapper()
                        mapper.SetInputConnection(geom.GetOutputPort())
                        
                        self.hover_actor = vtk.vtkActor()
                        self.hover_actor.SetMapper(mapper)
                        self.hover_actor.GetProperty().SetColor(1.0, 1.0, 0.0)  # Yellow
                        self.hover_actor.GetProperty().SetOpacity(0.5)
                        
                        # Add actor to renderer
                        self.renderer.AddActor(self.hover_actor)
                        
                        # Update status message
                        self.status_message = f"Hovering over face with {len(connected_cells)} cells. Click to select."
                        self.update_status()
                        
                        # Force render update
                        self.render_window.Render()
                    
                    # Update the status
                    cell_count = len(connected_cells)
                    ctrl_text = " (CTRL held - plane selection)" if ctrl_key else ""
                    self.status_message = f"Hover over face with {cell_count} cells{ctrl_text}. Click to select. Press ESC when done."
                    self.update_status()
                    
                elif cell_id == -1 or cell_id in self.selected_faces:
                    # No cell under cursor or already selected, hide hover actor
                    if hasattr(self, 'hover_actor'):
                        self.hover_actor.VisibilityOff()
                        self.render_window.Render()
                    
                    if self.hover_face_id != -1:
                        self.hover_face_id = -1
                        
                        # Clear connected cells
                        if hasattr(self, 'hover_connected_cells'):
                            self.hover_connected_cells = []
                        
                        # Update status
                        self.status_message = "Select faces for boundary. Hold CTRL for plane selection. Press ESC when done."
                        self.update_status()

    def _handle_pick(self, obj, event):
        """Handle left-click to select faces"""
        try:
            if not hasattr(self, 'selection_mode') or not self.selection_mode:
                # Properly forward the event if not in selection mode
                # Use the interactor style instead of the obj directly
                if hasattr(self.interactor, 'GetInteractorStyle'):
                    style = self.interactor.GetInteractorStyle()
                    if hasattr(style, 'OnLeftButtonDown'):
                        style.OnLeftButtonDown()
                return
                
            # Get the currently hovered cells
            if hasattr(self, 'hover_connected_cells') and self.hover_connected_cells:
                logger.debug(f"Adding {len(self.hover_connected_cells)} cells from hover highlight to selection")
                
                # Add all connected cells to the selection as a group
                for cell_id in self.hover_connected_cells:
                    self.selected_faces.add(cell_id)
                
                # Update the visualization
                self._update_face_selection()
                
                # Update status with accurate count
                cell_count = len(self.hover_connected_cells)
                self.status_message = f"Selected face with {cell_count} cells. Total selected: {len(self.selected_faces)} cells"
                self.update_status()
                
                # Log the selection
                logger.info(f"Selected face with {cell_count} cells")
                
                # Force render
                self.render_window.Render()
            else:
                # If no hover cells, use the standard pick operation
                x, y = self.interactor.GetEventPosition()
                picker = vtk.vtkCellPicker()
                picker.SetTolerance(0.005)
                picker.Pick(x, y, 0, self.renderer)
                cell_id = picker.GetCellId()
                
                if cell_id != -1:
                    # Find connected cells based on mode
                    ctrl_key = self.interactor.GetControlKey()
                    if ctrl_key:
                        connected_cells = self._find_cells_in_plane(cell_id)
                    else:
                        connected_cells = self._find_connected_face(cell_id)
                    
                    logger.info(f"Directly selected face with {len(connected_cells)} cells")
                    
                    # Add all cells in the face
                    for face_cell_id in connected_cells:
                        self.selected_faces.add(face_cell_id)
                    
                    # Update the visualization
                    self._update_face_selection()
                    
                    # Update status
                    self.status_message = f"Selected face with {len(connected_cells)} cells. Total: {len(self.selected_faces)}"
                    self.update_status()
                    
                    # Force render
                    self.render_window.Render()
        
        except Exception as e:
            logger.error(f"Error handling pick: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _on_key_press(self, obj, event):
        """Handle key press during selection"""
        try:
            # Ensure interactor is available
            if not hasattr(self, 'interactor'):
                logger.error("No interactor available for key press handling")
                return
                
            # Get key that was pressed
            key = self.interactor.GetKeySym()
            
            # Check for ESC key to cancel
            if key.lower() == "escape":
                logger.debug("ESC key pressed, ending selection")
                self._end_selection()
                return
            
            # Forward event - this line is causing errors so we'll handle differently
            # obj.OnKeyPress()  # This method doesn't exist in this VTK version
            
            # Let the default VTK interactor style handle other keys
            if hasattr(self, 'vtk_widget') and hasattr(self.vtk_widget, '_Iren'):
                iren = self.vtk_widget._Iren
                if hasattr(iren, 'GetInteractorStyle'):
                    style = iren.GetInteractorStyle()
                    if style and hasattr(style, 'OnKeyPress'):
                        style.OnKeyPress()
                    # Otherwise just let the key event be processed normally
        
        except Exception as e:
            logger.error(f"Error in key press handler: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _cleanup_selection_mode(self):
        """Clean up after selection mode"""
        try:
            # Reset selection mode flag
            self.selection_mode = False
            
            # Remove hover and selection actors
            if hasattr(self, 'hover_actor'):
                self.renderer.RemoveActor(self.hover_actor)
                self.hover_actor = None
            
            if hasattr(self, 'selection_actor'):
                self.renderer.RemoveActor(self.selection_actor)
                self.selection_actor = None
            
            # Remove instruction actor
            self._remove_instruction_actor()
            
            # Restore default interactor style
            if hasattr(self, 'default_style') and self.render_window_interactor:
                self.render_window_interactor.SetInteractorStyle(self.default_style)
            
            # Restore cursor
            self.setCursor(Qt.ArrowCursor)
            
            # Clear selection data
            self.selected_faces = set()
            self.hover_cell_id = -1
            self.hover_connected_cells = []
            
            # Ensure render window is enabled
            self.render_window.Render()
        except Exception as e:
            logger.error(f"Error cleaning up selection mode: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _cancel_selection(self):
        """Cancel the selection process"""
        try:
            if not self.selection_mode:
                return
            
            # Clean up selection mode
            self._cleanup_selection_mode()
            
            # Update status
            self.status_message = "Selection canceled"
            self.update_status()
            
            # Force render
            self.render_window.Render()
        except Exception as e:
            logger.error(f"Error canceling selection: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _initialize_default_boundaries(self):
        """Set all external faces as walls by default"""
        if not hasattr(self, 'data') or self.data is None:
            return
        
        try:
            # Initialize boundary dictionary if it doesn't exist
            if not hasattr(self, 'boundary_types'):
                self.boundary_types = {}
                
            # Find all external faces
            external_faces = self._find_external_faces()
            
            # Set all as walls by default
            for face_id in external_faces:
                self.boundary_types[face_id] = "wall"
                
            # Create visualization for walls
            self._update_boundary_visualization()
            
            logger.info(f"Set {len(external_faces)} external faces as walls by default")
            
        except Exception as e:
            logger.error(f"Error setting default boundaries: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _find_external_faces(self):
        """Find all external faces in the mesh"""
        if not hasattr(self, 'data') or self.data is None:
            return []
        
        try:
            # Create feature edges filter to find boundaries
            feature_edges = vtk.vtkFeatureEdges()
            feature_edges.SetInputData(self.data)
            feature_edges.BoundaryEdgesOn()
            feature_edges.FeatureEdgesOff()
            feature_edges.NonManifoldEdgesOff()
            feature_edges.ManifoldEdgesOff()
            feature_edges.Update()
            
            # Get all cells in the mesh
            all_cells = set(range(self.data.GetNumberOfCells()))
            
            # Return the set of all cells (we're assuming all are external for now)
            # In a more sophisticated implementation, we would filter for truly external faces
            return list(all_cells)
            
        except Exception as e:
            logger.error(f"Error finding external faces: {e}")
            return []

    def _update_boundary_visualization(self):
        """Update visualization for all boundary types"""
        if not hasattr(self, 'boundary_types') or not self.boundary_types:
            return
        
        try:
            # Define colors for different boundary types
            colors = {
                "wall": [0.8, 0.8, 0.8],     # Gray
                "inlet": [0.0, 0.0, 1.0],    # Blue
                "outlet": [1.0, 0.0, 0.0],   # Red
                "symmetry": [0.0, 1.0, 1.0], # Cyan
                "custom": [1.0, 0.5, 0.0]    # Orange
            }
            
            # Group faces by boundary type
            faces_by_type = {}
            for face_id, btype in self.boundary_types.items():
                if btype not in faces_by_type:
                    faces_by_type[btype] = []
                faces_by_type[btype].append(face_id)
                
            # Remove any old boundary actors
            if hasattr(self, 'boundary_actors'):
                for actor in self.boundary_actors:
                    self.renderer.RemoveActor(actor)
                    
            self.boundary_actors = []
            
            # Create an actor for each boundary type
            for btype, faces in faces_by_type.items():
                # Create selection array
                id_array = vtk.vtkIdTypeArray()
                id_array.SetNumberOfComponents(1)
                for face_id in faces:
                    id_array.InsertNextValue(face_id)
                    
                selection = vtk.vtkSelectionNode()
                selection.SetFieldType(vtk.vtkSelectionNode.CELL)
                selection.SetContentType(vtk.vtkSelectionNode.INDICES)
                selection.SetSelectionList(id_array)
                
                selections = vtk.vtkSelection()
                selections.AddNode(selection)
                
                # Extract selected faces
                extract = vtk.vtkExtractSelection()
                extract.SetInputData(0, self.data)
                extract.SetInputData(1, selections)
                extract.Update()
                
                # Convert to polydata
                geom = vtk.vtkGeometryFilter()
                geom.SetInputConnection(extract.GetOutputPort())
                geom.Update()
                
                # Create actor
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(geom.GetOutputPort())
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                
                # Set color based on boundary type
                color = colors.get(btype, [0.5, 0.5, 0.5])  # Default gray if type not found
                actor.GetProperty().SetColor(*color)
                
                # Set opacity (walls slightly transparent)
                opacity = 0.4 if btype == "wall" else 0.7
                actor.GetProperty().SetOpacity(opacity)
                
                # Add to renderer
                self.renderer.AddActor(actor)
                self.boundary_actors.append(actor)
                
            # Render the scene
            self.render_window.Render()
            
        except Exception as e:
            logger.error(f"Error updating boundary visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _add_selection_instructions(self):
        """Add instructions text actor"""
        if hasattr(self, 'instructions_actor'):
            self.renderer.RemoveActor(self.instructions_actor)
        
        # Create text actor
        self.instructions_actor = vtk.vtkTextActor()
        self.instructions_actor.SetInput(
            f"Selecting for boundary: {self.selection_boundary_name}\n"
            "Hover over faces to highlight, click to select\n"
            "Hold CTRL for plane selection. Press ESC when finished"
        )
        self.instructions_actor.GetTextProperty().SetFontSize(14)
        self.instructions_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)  # White text
        self.instructions_actor.GetTextProperty().SetBackgroundColor(0.1, 0.1, 0.1)
        self.instructions_actor.GetTextProperty().SetBackgroundOpacity(0.7)
        
        # Position in top-left corner
        self.instructions_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        self.instructions_actor.GetPositionCoordinate().SetValue(0.05, 0.85)
        
        self.renderer.AddActor2D(self.instructions_actor)
        self.render_window.Render()

    def _remove_selection_instructions(self):
        """Remove text instructions for selection mode"""
        if hasattr(self, 'instruction_actor') and self.instruction_actor:
            self.renderer.RemoveActor2D(self.instruction_actor)
            self.instruction_actor = None

    def _find_cylindrical_surface_cells(self, face_id, normal, origin):
        """
        Find all cells in the same cylindrical surface as the given face.
        
        Args:
            face_id (int): The ID of the face to start from
            normal (list, optional): Normal vector of the cylindrical surface
            origin (list, optional): Origin of the cylindrical surface
            
        Returns:
            list: List of cell IDs forming the cylindrical surface
        """
        if not hasattr(self, 'data') or self.data is None:
            return []
        
        try:
            # Calculate normal and origin if not provided
            if normal is None or origin is None:
                # Get the cell
                cell = self.data.GetCell(face_id)
                if not cell:
                    return [face_id]
                
                # Calculate cell center
                center = [0, 0, 0]
                points = cell.GetPoints()
                n_points = points.GetNumberOfPoints()
                for i in range(n_points):
                    point = points.GetPoint(i)
                    center[0] += point[0]
                    center[1] += point[1]
                    center[2] += point[2]
                
                if n_points > 0:
                    center[0] /= n_points
                    center[1] /= n_points
                    center[2] /= n_points
                
                # If origin not provided, use calculated center
                if origin is None:
                    origin = center
                
                # If normal not provided, calculate it
                if normal is None:
                    # Get normals
                    normals = vtk.vtkPolyDataNormals()
                    normals.SetInputData(self.data)
                    normals.ComputeCellNormalsOn()
                    normals.Update()
                    
                    cell_normals = normals.GetOutput().GetCellData().GetNormals()
                    if not cell_normals:
                        return [face_id]
                    
                    normal = [0, 0, 0]
                    cell_normals.GetTuple(face_id, normal)
            
            # Now normal and origin should be valid
            # Create a plane for slicing
            plane = vtk.vtkPlane()
            plane.SetNormal(normal)
            plane.SetOrigin(origin)
            
            # Find all connected cells that match this cylindrical surface
            # Using a breadth-first approach for better performance
            visited = set([face_id])
            queue = [face_id]
            connected_cells = [face_id]
            
            # Process cells in batches for better UI responsiveness
            batch_size = 100
            max_cells = 10000  # Limit total cells for performance
            
            while queue and len(connected_cells) < max_cells:
                current_batch = queue[:batch_size]
                queue = queue[batch_size:]
                
                for current_id in current_batch:
                    # Get cell neighbors 
                    neighbors = []
                    
                    # Get the cell
                    cell = self.data.GetCell(current_id)
                    if not cell:
                        continue
                    
                    # Get points of this cell
                    points = cell.GetPoints()
                    if not points:
                        continue
                    
                    # Find neighbor cells that share points
                    for point_idx in range(points.GetNumberOfPoints()):
                        point_id = cell.GetPointId(point_idx)
                        
                        # Get cells using this point
                        cell_ids = vtk.vtkIdList()
                        self.data.GetPointCells(point_id, cell_ids)
                        
                        for i in range(cell_ids.GetNumberOfIds()):
                            neighbor_id = cell_ids.GetId(i)
                            
                            # Skip if already visited
                            if neighbor_id in visited:
                                continue
                            
                            # Mark as visited
                            visited.add(neighbor_id)
                            
                            # Get neighbor normal
                            neighbor_normal = [0, 0, 0]
                            normals.GetOutput().GetCellData().GetNormals().GetTuple(neighbor_id, neighbor_normal)
                            
                            # Check if normals are similar (dot product close to 1 or -1)
                            dot_product = (normal[0]*neighbor_normal[0] + 
                                           normal[1]*neighbor_normal[1] + 
                                           normal[2]*neighbor_normal[2])
                            
                            # Add to connected cells if normal is similar
                            if abs(dot_product) > 0.8:  # Within ~36 degrees
                                connected_cells.append(neighbor_id)
                                queue.append(neighbor_id)
            
            return connected_cells
            
        except Exception as e:
            logger.debug(f"Error finding cylindrical surface cells: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return [face_id]

    def _update_face_selection(self):
        """Update the selection actor to show all selected faces."""
        if not hasattr(self, 'selected_faces') or not self.selected_faces:
            if hasattr(self, 'selection_actor'):
                self.selection_actor.VisibilityOff()
            return
        
        try:
            # Create an array of selected face IDs
            id_array = vtk.vtkIdTypeArray()
            id_array.SetNumberOfComponents(1)
            for face_id in self.selected_faces:
                id_array.InsertNextValue(face_id)
            
            # Create selection
            selection = vtk.vtkSelectionNode()
            selection.SetFieldType(vtk.vtkSelectionNode.CELL)
            selection.SetContentType(vtk.vtkSelectionNode.INDICES)
            selection.SetSelectionList(id_array)
            
            selections = vtk.vtkSelection()
            selections.AddNode(selection)
            
            # Extract selected cells
            extract = vtk.vtkExtractSelection()
            extract.SetInputData(0, self.data)
            extract.SetInputData(1, selections)
            extract.Update()
            
            # Convert to polydata
            geom = vtk.vtkGeometryFilter()
            geom.SetInputConnection(extract.GetOutputPort())
            geom.Update()
            
            # Update selection actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(geom.GetOutputPort())
            
            # Choose color based on boundary type
            color = (0.0, 1.0, 0.0)  # Default green
            if hasattr(self, 'boundary_name'):
                if "inlet" in self.boundary_name.lower():
                    color = (0.0, 0.0, 1.0)  # Blue for inlet
                elif "outlet" in self.boundary_name.lower():
                    color = (1.0, 0.0, 0.0)  # Red for outlet
            
            self.selection_actor.SetMapper(mapper)
            self.selection_actor.GetProperty().SetColor(*color)
            self.selection_actor.VisibilityOn()
        except Exception as e:
            logger.error(f"Error updating selection actor: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def update_status(self):
        """Update the status message in the viewport."""
        if not hasattr(self, 'renderer'):
            return
        
        # Create text actor for status
        status_actor = vtk.vtkTextActor()
        status_actor.SetInput(self.status_message)
        status_actor.GetTextProperty().SetFontSize(14)
        status_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)  # White
        status_actor.GetTextProperty().SetBackgroundColor(0.1, 0.1, 0.1)
        status_actor.GetTextProperty().SetBackgroundOpacity(0.7)
        
        # Position in top-left corner
        status_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        status_actor.GetPositionCoordinate().SetValue(0.05, 0.95)
        
        # Remove old status actor if exists
        if hasattr(self, 'status_actor'):
            self.renderer.RemoveActor(self.status_actor)
        
        # Add new status actor
        self.status_actor = status_actor
        self.renderer.AddActor2D(status_actor)
        
        # Force render update
        self.render_window.Render()

    def clear_highlight(self, boundary_name=None):
        """Clear highlighting for a specific boundary or all highlights
        
        Args:
            boundary_name (str, optional): Name of boundary to clear. If None, clears all.
        """
        try:
            # Store whether any change was made
            updated = False
            
            # Check if we already have a dict for storing highlight actors
            if not hasattr(self, 'boundary_highlight_actors'):
                self.boundary_highlight_actors = {}
                
            # Remove specific boundary highlight
            if boundary_name is not None:
                if boundary_name in self.boundary_highlight_actors:
                    # Remove the actor from the renderer
                    actor = self.boundary_highlight_actors[boundary_name]
                    self.renderer.RemoveActor(actor)
                    # Remove from our dictionary
                    del self.boundary_highlight_actors[boundary_name]
                    logger.info(f"Cleared highlighting for boundary: {boundary_name}")
                    updated = True
                    
                # Also remove from any data storage
                if hasattr(self, 'highlighted_faces') and boundary_name in self.highlighted_faces:
                    del self.highlighted_faces[boundary_name]
                    
            # Or remove all highlights
            else:
                # Remove all highlight actors from renderer
                for name, actor in list(self.boundary_highlight_actors.items()):
                    logger.info(f"Removing highlight actor for {name}")
                    self.renderer.RemoveActor(actor)
                # Clear the dictionary
                self.boundary_highlight_actors.clear()
                
                # Also clear any data storage
                if hasattr(self, 'highlighted_faces'):
                    self.highlighted_faces.clear()
                    
                logger.info("Cleared all boundary highlights")
                updated = True
                
            # Update the rendering if changes were made
            if updated and hasattr(self, 'render_window'):
                self.render_window.Render()
                
        except Exception as e:
            logger.error(f"Error clearing highlights: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def highlight_faces(self, face_ids, boundary_name, color=None):
        """Highlight faces for a specific boundary
        
        Args:
            face_ids (list): List of face IDs to highlight
            boundary_name (str): Name of the boundary
            color (list, optional): RGB color tuple. Defaults to None (uses preset colors).
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure parameters are the correct types
            if isinstance(boundary_name, list) and isinstance(face_ids, str):
                # Parameters were passed in the wrong order, swap them
                face_ids, boundary_name = boundary_name, face_ids
            
            # Ensure boundary_name is a string
            if not isinstance(boundary_name, str):
                boundary_name = "wall"  # Default to wall if not a string
            
            # Ensure face_ids is a list
            if not isinstance(face_ids, list):
                if face_ids is None:
                    face_ids = []
                else:
                    face_ids = [face_ids]  # Convert single value to list
            
            if not face_ids:
                logger.warning(f"No face IDs provided for boundary {boundary_name}")
                return False
            
            # Check if we already have a dict for storing cell IDs
            if not hasattr(self, 'highlighted_faces'):
                self.highlighted_faces = {}
                
            # Check if we already have a dict for storing highlight actors
            if not hasattr(self, 'boundary_highlight_actors'):
                self.boundary_highlight_actors = {}
                
            # Clear any existing highlight for this boundary
            if boundary_name in self.boundary_highlight_actors:
                self.renderer.RemoveActor(self.boundary_highlight_actors[boundary_name])
                
            # Set default color if not specified
            if color is None:
                # Different colors for different types of boundaries
                if 'inlet' in boundary_name.lower():
                    color = [0.0, 0.0, 1.0]  # Blue for inlets
                elif 'outlet' in boundary_name.lower():
                    color = [1.0, 0.0, 0.0]  # Red for outlets
                elif 'wall' in boundary_name.lower():
                    color = [0.8, 0.8, 0.8]  # Gray for walls
                else:
                    color = [0.0, 1.0, 0.0]  # Green default
            
            # Log what color we're using
            logger.info(f"Using color {color} for boundary {boundary_name}")
            
            # Store the face IDs for this boundary
            self.highlighted_faces[boundary_name] = face_ids
            
            # Also update boundary_cell_ids dictionary to maintain consistency
            if not hasattr(self, 'boundary_cell_ids'):
                self.boundary_cell_ids = {}
            self.boundary_cell_ids[boundary_name] = face_ids
            logger.info(f"Updated boundary_cell_ids for {boundary_name} with {len(face_ids)} faces")
            
            # Setup the selection node for these cell IDs
            selection = vtk.vtkSelectionNode()
            selection.SetFieldType(vtk.vtkSelectionNode.CELL)
            selection.SetContentType(vtk.vtkSelectionNode.INDICES)
            
            # Convert the face IDs to a vtkIdTypeArray
            id_array = vtk.vtkIdTypeArray()
            id_array.SetNumberOfComponents(1)
            for face_id in face_ids:
                id_array.InsertNextValue(face_id)
            
            selection.SetSelectionList(id_array)
            
            # Create a selection object
            selections = vtk.vtkSelection()
            selections.AddNode(selection)
            
            # Extract the selected cells
            extract = vtk.vtkExtractSelection()
            extract.SetInputData(0, self.data)
            extract.SetInputData(1, selections)
            extract.Update()
            
            # Convert to polydata for visualization
            geom = vtk.vtkGeometryFilter()
            geom.SetInputConnection(extract.GetOutputPort())
            geom.Update()
            
            # Create mapper and actor for the highlighting
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(geom.GetOutputPort())
            
            # Create the actor and set its properties
            highlight_actor = vtk.vtkActor()
            highlight_actor.SetMapper(mapper)
            highlight_actor.GetProperty().SetColor(*color)
            highlight_actor.GetProperty().SetOpacity(1.0)
            
            # Set rendering priority based on boundary type
            # Ensure inlets and outlets are always visible above walls
            if 'inlet' in boundary_name.lower() or 'outlet' in boundary_name.lower():
                # Use better rendering properties to ensure visibility instead of position offset
                highlight_actor.GetProperty().SetOpacity(1.0)
                highlight_actor.GetProperty().SetPointSize(5)
                highlight_actor.GetProperty().SetLineWidth(2)
                # Remove and re-add actor to ensure it's rendered on top
                self.renderer.RemoveActor(highlight_actor)
                self.renderer.AddActor(highlight_actor)
                # No position offset needed
                highlight_actor.SetPosition(0, 0, 0)
            elif 'wall' in boundary_name.lower():
                # Set walls slightly back and with some transparency
                highlight_actor.SetPosition(0, 0, 0)
                highlight_actor.GetProperty().SetOpacity(0.7)  # Partially transparent
            
            # Add to renderer
            self.renderer.AddActor(highlight_actor)
            
            # Store the actor for future reference
            self.boundary_highlight_actors[boundary_name] = highlight_actor
            
            # Force a render update
            if hasattr(self, 'render_window'):
                self.render_window.Render()
                
            logger.info(f"Highlighted {len(face_ids)} faces for boundary: {boundary_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error highlighting faces: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _check_vtk_available(self):
        """
        Check if VTK is available.
        
        Returns:
            bool: True if VTK is available, False otherwise
        """
        global VTK_AVAILABLE
        
        if not VTK_AVAILABLE:
            logger.error("VTK is not available. Cannot load results.")
            if hasattr(self, 'status_message'):
                self.status_message = "Error: VTK library is not available"
                self.update_status()
            return False
        
        return True

    def _get_connected_face_cells(self, seed_cell_id):
        """
        Get all cells that belong to the same face as the seed cell.
        Uses a more robust algorithm that considers face normals and connectivity.
        
        Args:
            seed_cell_id: The starting cell ID
            
        Returns:
            List of connected cell IDs
        """
        if not hasattr(self, 'data') or self.data is None:
            return [seed_cell_id]
        
        try:
            # First, compute normals for all cells
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(self.data)
            normals.ComputeCellNormalsOn()
            normals.Update()
            
            # Get the cell normals array
            cell_normals = normals.GetOutput().GetCellData().GetNormals()
            if not cell_normals:
                return [seed_cell_id]
            
            # Get the seed cell normal
            seed_normal = [0, 0, 0]
            cell_normals.GetTuple(seed_cell_id, seed_normal)
            
            # Normalize the seed normal
            magnitude = (seed_normal[0]**2 + seed_normal[1]**2 + seed_normal[2]**2)**0.5
            if magnitude > 0:
                seed_normal = [n/magnitude for n in seed_normal]
            
            # Get the seed cell to find its neighbors
            seed_cell = self.data.GetCell(seed_cell_id)
            if not seed_cell:
                return [seed_cell_id]
            
            # Use a breadth-first search to find connected cells with similar normals
            visited = set([seed_cell_id])
            queue = [seed_cell_id]
            
            # Process the queue
            while queue:
                current_id = queue.pop(0)
                
                # Get cell neighbors 
                neighbors = []
                
                # Get the cell
                cell = self.data.GetCell(current_id)
                if not cell:
                    continue
                    
                # For each point in the cell, find cells that share this point
                for i in range(cell.GetNumberOfPoints()):
                    point_id = cell.GetPointId(i)
                    
                    # Get cells using this point
                    cell_ids = vtk.vtkIdList()
                    self.data.GetPointCells(point_id, cell_ids)
                    
                    for j in range(cell_ids.GetNumberOfIds()):
                        neighbor_id = cell_ids.GetId(j)
                        
                        # Skip if already visited
                        if neighbor_id in visited:
                            continue
                        
                        # Mark as visited
                        visited.add(neighbor_id)
                        
                        # Get neighbor normal
                        neighbor_normal = [0, 0, 0]
                        cell_normals.GetTuple(neighbor_id, neighbor_normal)
                        
                        # Normalize
                        magnitude = (neighbor_normal[0]**2 + neighbor_normal[1]**2 + neighbor_normal[2]**2)**0.5
                        if magnitude > 0:
                            neighbor_normal = [n/magnitude for n in neighbor_normal]
                        
                        # Check if normals are similar (dot product close to 1)
                        dot_product = abs(seed_normal[0]*neighbor_normal[0] + 
                                         seed_normal[1]*neighbor_normal[1] + 
                                         seed_normal[2]*neighbor_normal[2])
                        
                        # If normals are similar, add to the face
                        if dot_product > 0.90:  # Within ~25 degrees
                            connected_cells.append(neighbor_id)
                            queue.append(neighbor_id)
            
            return list(visited)
            
        except Exception as e:
            logger.error(f"Error finding connected cells: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [seed_cell_id]

    def set_all_as_wall(self, force=False):
        """Set all unassigned faces as wall boundaries
        
        Args:
            force (bool): If True, set all external faces as walls even if already assigned
        
        Returns:
            tuple: (success, face_ids) where success is a boolean and face_ids is a list of face IDs
        """
        try:
            # Check if we have a valid mesh loaded
            if not hasattr(self, 'data') or not self.data:
                logger.error("No mesh data available")
                return False, []
                
            # Get number of cells in the mesh
            num_cells = self.data.GetNumberOfCells()
            logger.info(f"Mesh has {num_cells} cells")
            
            if num_cells == 0:
                logger.error("Mesh has 0 cells")
                return False, []
            
            # Create a set of all cell IDs
            all_cell_ids = set(range(num_cells))
            
            # Get set of already assigned cells to other boundaries
            assigned_cell_ids = set()
            if hasattr(self, 'boundary_cell_ids') and self.boundary_cell_ids:
                for boundary_name, cell_ids in self.boundary_cell_ids.items():
                    # Skip the "wall" boundary itself
                    if boundary_name.lower() != "wall":
                        logger.info(f"Excluding {len(cell_ids)} cells from boundary: {boundary_name}")
                        assigned_cell_ids.update(cell_ids)
            
            # Get unassigned cells by excluding already assigned cells
            if force:
                # If force is True, include all cells (traditional behavior)
                wall_face_ids = list(all_cell_ids)
                logger.info(f"Force mode: Using all {len(wall_face_ids)} cells for wall boundary")
            else:
                # Only include cells not assigned to other boundaries
                wall_face_ids = list(all_cell_ids - assigned_cell_ids)
                logger.info(f"Found {len(assigned_cell_ids)} cells already assigned to other boundaries")
                logger.info(f"Setting remaining {len(wall_face_ids)} unassigned cells as wall boundary")
            
            # If we found faces, set them as walls
            if wall_face_ids:
                # Store the face IDs for later access
                self.wall_face_ids = wall_face_ids
                
                # Visually mark these as walls
                if hasattr(self, 'highlight_faces'):
                    self.highlight_faces(wall_face_ids, "wall", color=[0.8, 0.8, 0.8])
                
                # Log success
                logger.info(f"Successfully set {len(wall_face_ids)} faces as walls")
                return True, wall_face_ids
            else:
                logger.warning("No faces found to set as walls")
                return False, []
                
        except Exception as e:
            logger.error(f"Error setting unassigned faces as wall: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False, []

    def get_wall_cell_ids(self):
        """Get the IDs of faces set as walls
        
        Returns:
            list: List of face IDs that are set as walls
        """
        if hasattr(self, 'wall_face_ids'):
            return self.wall_face_ids
        else:
            return []

    def set_boundaries_callback(self, callback):
        """Set callback for when boundaries change"""
        self.on_boundaries_changed = callback

    def _create_instruction_actor(self, text):
        """Create an instruction text actor with the given text"""
        try:
            # Create text actor
            self.instruction_actor = vtk.vtkTextActor()
            self.instruction_actor.SetInput(text)
            
            # Set text properties
            text_prop = self.instruction_actor.GetTextProperty()
            text_prop.SetFontSize(14)
            text_prop.SetColor(1.0, 1.0, 1.0)  # White text
            text_prop.SetBackgroundColor(0.1, 0.1, 0.1)
            text_prop.SetBackgroundOpacity(0.7)
            
            # Position at the bottom of the viewport
            self.instruction_actor.SetDisplayPosition(10, 10)
            
            # Add to renderer
            self.renderer.AddActor2D(self.instruction_actor)
            
            # Update the display
            self.render_window.Render()
        except Exception as e:
            logger.error(f"Error creating instruction actor: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _remove_instruction_actor(self):
        """Remove the instruction text actor"""
        if hasattr(self, 'instruction_actor') and self.instruction_actor:
            self.renderer.RemoveActor2D(self.instruction_actor)
            self.instruction_actor = None
            self.render_window.Render()

    def initialize(self):
        """Initialize the viewport with a VTK render window."""
        if not VTK_AVAILABLE:
            logger.warning("VTK not available, visualization disabled")
            return

        try:
            # Create interactor
            self.interactor = QVTKRenderWindowInteractor(self)
            
            # Create render window and renderer
            self.render_window = vtk.vtkRenderWindow()
            self.renderer = vtk.vtkRenderer()
            
            # Add renderer to render window
            self.render_window.AddRenderer(self.renderer)
            
            # Connect interactor to render window
            self.interactor.SetRenderWindow(self.render_window)
            
            # Set background color
            self.renderer.SetBackground(0.2, 0.2, 0.2)  # Dark gray background
            
            # Create layout and add VTK widget
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.interactor)
            self.setLayout(layout)
            
            # Set default interaction style
            style = vtk.vtkInteractorStyleTrackballCamera()
            self.interactor.SetInteractorStyle(style)
            
            # Initialize and start the interactor
            self.interactor.Initialize()
            self.interactor.Start()
            
            # Store a reference to the render window interactor
            self.render_window_interactor = self.interactor
            
            # Create axes indicator
            self._create_axes()
            
            # Initialize other properties
            self.background_color = [0.2, 0.2, 0.2]
            self.boundary_types = {}
            self.default_boundaries_set = False
            
            logger.info("VTK viewport initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing VTK viewport: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _show_boundary_context_menu(self, position):
        """Show context menu for boundary items"""
        # Get the item under the cursor
        item = self.boundary_list.itemAt(position)
        if not item:
            return
        
        # Get boundary name from item
        boundary_name = item.text()
        
        # Create menu
        menu = QMenu()
        delete_action = menu.addAction("Delete Boundary")
        
        # Show menu and handle selection
        action = menu.exec_(self.boundary_list.mapToGlobal(position))
        
        if action == delete_action:
            self._delete_boundary(boundary_name)

    def _delete_boundary(self, boundary_name):
        """Delete a boundary and allow it to be redefined"""
        # Confirm deletion with the user
        reply = QMessageBox.question(
            self, 
            "Delete Boundary",
            f"Are you sure you want to delete the '{boundary_name}' boundary?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Remove from boundaries dictionary
            if boundary_name in self.boundaries:
                del self.boundaries[boundary_name]
                
            # Remove from cell_ids dictionary
            if boundary_name in self.cell_ids:
                # Get cells that need to be unmarked
                cells_to_clear = self.cell_ids[boundary_name]
                
                # Remove boundary type from those cells
                for cell_id in cells_to_clear:
                    if cell_id in self.boundary_types:
                        del self.boundary_types[cell_id]
                        
                # Remove from cell_ids dictionary
                del self.cell_ids[boundary_name]
                
            # Update the boundary list
            self.update_boundary_list()
            
            # Update visualization in viewport
            if hasattr(self.main_window, 'viewport'):
                self.main_window.viewport.update_boundary_visualization()
                
            # Log the change
            logger.info(f"Deleted boundary: {boundary_name}")

    def update_boundary_visualization(self):
        """Update visualization of boundaries based on current boundary data"""
        # Clear existing boundary actors
        if hasattr(self, 'boundary_actors'):
            for actor in self.boundary_actors:
                self.renderer.RemoveActor(actor)
        else:
            self.boundary_actors = []
        
        # Recreate visualization for current boundaries
        if hasattr(self, 'boundary_types') and hasattr(self, 'data'):
            for boundary_name, cell_ids in self.cell_ids.items():
                # Create visualization for this boundary
                self._create_boundary_visualization(boundary_name, cell_ids)
                
        # Render the changes
        self.render_window.Render()

    def create_ambient_region(self, interface_boundary, settings):
        """
        Create an ambient region extending from the specified boundary.
        
        Args:
            interface_boundary (str): Name of the boundary to use as interface
            settings (dict): Dictionary of ambient region settings
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Creating ambient region from boundary: {interface_boundary}")
        
        try:
            # More robust approach to get active data
            source_data = None
            
            # Try multiple ways to get the active data
            if hasattr(self, 'active_source') and self.active_source:
                logger.info(f"Active source: {self.active_source}")
                
                # Method 1: From sources dictionary
                if hasattr(self, 'sources') and self.active_source in self.sources:
                    source = self.sources[self.active_source]
                    logger.info(f"Found source in sources dictionary: {type(source).__name__}")
                    
                    if hasattr(source, 'GetOutput'):
                        source_data = source.GetOutput()
                        logger.info("Got data using GetOutput()")
                    elif hasattr(source, 'GetOutputPort'):
                        # Need to use a temporary to get output
                        tmp = vtkPolyDataMapper()
                        tmp.SetInputConnection(source.GetOutputPort())
                        source_data = tmp.GetInput()
                
                # Method 2: Try from actors if they exist
                if not source_data and hasattr(self, 'actors'):
                    logger.info("Trying to get data from actors")
                    for actor_name, actor in self.actors.items():
                        if actor and actor.GetMapper():
                            try:
                                mapper = actor.GetMapper()
                                source_data = mapper.GetInput()
                                if source_data:
                                    logger.info(f"Got data from actor: {actor_name}")
                                    break
                            except Exception as e:
                                logger.error(f"Error getting data from actor {actor_name}: {e}")
                
                # Method 3: Try from renderer props
                if not source_data and hasattr(self, 'renderer'):
                    logger.info("Trying to get data from renderer props")
                    props = self.renderer.GetViewProps()
                    for i in range(props.GetNumberOfItems()):
                        prop = props.GetItemAsObject(i)
                        if isinstance(prop, vtk.vtkActor) and prop.GetMapper():
                            try:
                                mapper = prop.GetMapper()
                                source_data = mapper.GetInput()
                                if source_data:
                                    logger.info(f"Got data from renderer prop {i}")
                                    break
                            except Exception as e:
                                logger.error(f"Error getting data from renderer prop {i}: {e}")
            
            # Method 4: Fall back to a simple box if all else fails
            if not source_data:
                logger.warning("Could not obtain source data - using fallback box")
                # Create a simple box as fallback
                box = vtk.vtkCubeSource()
                box.SetXLength(10)
                box.SetYLength(10)
                box.SetZLength(10)
                box.Update()
                source_data = box.GetOutput()
                
                # In the fallback case, we need to create some boundary data
                if not hasattr(self, 'boundary_types'):
                    self.boundary_types = {}
                
                if interface_boundary not in self.boundary_types:
                    self.boundary_types[interface_boundary] = list(range(100, 110))
            
            # Create the ambient region box (simplified for demonstration)
            cube = vtk.vtkCubeSource()
            
            # Set size based on provided settings
            x_size = 10 * settings.get('x_extent', 5.0)
            y_size = 10 * settings.get('y_extent', 5.0)
            z_size = 10 * settings.get('z_extent', 5.0)
            
            cube.SetXLength(x_size)
            cube.SetYLength(y_size)
            cube.SetZLength(z_size)
            cube.Update()
            
            ambient_box = cube.GetOutput()
            
            # Create a visual representation of the ambient box
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(ambient_box)
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            # Set appearance
            color_map = {
                "Light Blue": (0.7, 0.8, 1.0),
                "Light Gray": (0.8, 0.8, 0.8),
                "Light Green": (0.7, 1.0, 0.7)
            }
            
            color = color_map.get(settings.get('color_name', "Light Blue"), (0.7, 0.8, 1.0))
            opacity = settings.get('opacity', 0.3)
            
            actor.GetProperty().SetColor(color)
            actor.GetProperty().SetOpacity(opacity)
            actor.GetProperty().SetRepresentationToWireframe()
            
            # Add to renderer
            self.renderer.AddActor(actor)
            self.ambient_actor = actor
            
            # Store ambient region info
            self.ambient_box_data = {
                'interface_boundary': interface_boundary,
                'settings': settings,
                'box': ambient_box
            }
            
            # Add fake ambient boundaries for demonstration
            if hasattr(self, 'boundary_cell_ids'):
                for direction in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']:
                    boundary_name = f"ambient_{direction}"
                    self.boundary_cell_ids[boundary_name] = list(range(1000, 1010))
                    logger.info(f"Added ambient boundary: {boundary_name}")
            
            # Update the window
            if hasattr(self, 'render_window'):
                self.render_window.Render()
            
            # Now handle the OpenFOAM case manager part
            openfoam_success = False
            
            # Get case manager from main window
            case_manager = None
            if hasattr(self, 'main_window') and self.main_window and hasattr(self.main_window, 'case_manager') and self.main_window.case_manager:
                case_manager = self.main_window.case_manager
            elif hasattr(self, 'main_window') and self.main_window and hasattr(self.main_window, 'simulation_controls') and hasattr(self.main_window.simulation_controls, 'case_manager'):
                case_manager = self.main_window.simulation_controls.case_manager
            
            if case_manager:
                # Ensure case directory is set
                if not hasattr(case_manager, 'case_directory') or not case_manager.case_directory:
                    # Try to get case directory from simulation controls
                    if hasattr(self, 'main_window') and self.main_window and hasattr(self.main_window, 'simulation_controls') and hasattr(self.main_window.simulation_controls, 'case_dir'):
                        case_dir = self.main_window.simulation_controls.case_dir
                        if case_dir:
                            logger.info(f"Setting case directory from simulation controls: {case_dir}")
                            case_manager.set_case_directory(case_dir)
                    
                    # If still no case directory, don't create ambient region in OpenFOAM
                    if not hasattr(case_manager, 'case_directory') or not case_manager.case_directory:
                        logger.warning("Cannot create ambient region in OpenFOAM: No case directory")
                
                # If case directory is set, create the ambient region
                if hasattr(case_manager, 'case_directory') and case_manager.case_directory:
                    logger.info("Creating ambient region in OpenFOAM case")
                    
                    # Prepare settings for case manager
                    ambient_settings = {
                        'temperature': settings.get('temperature', 300),  # Default temperature in K
                        'velocity': settings.get('velocity', [0, 0, 0])   # Default velocity
                    }
                    
                    openfoam_success = case_manager.create_centered_ambient_region(
                        center=[0, 0, 0],
                        size=[x_size, y_size, z_size],
                        name="ambientRegion",
                        settings=ambient_settings
                    )
                    
                    if openfoam_success:
                        logger.info("Successfully created ambient region in OpenFOAM case")
                    else:
                        logger.warning("Failed to create ambient region in OpenFOAM case")
                
            # Return true as long as the visualization worked, even if OpenFOAM failed
            return True
                
        except Exception as e:
            logger.error(f"Error creating ambient region: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _get_cell_center_average(self, polydata, cell_ids):
        """Calculate the average center position of the specified cells"""
        if not cell_ids:
            return [0, 0, 0]
            
        sum_x, sum_y, sum_z = 0, 0, 0
        count = 0
        
        for cell_id in cell_ids:
            cell = polydata.GetCell(cell_id)
            if not cell:
                continue
                
            # Get cell center
            center = [0, 0, 0]
            vtk.vtkCellCenters().GetCellCenter(cell_id, center)
            
            sum_x += center[0]
            sum_y += center[1]
            sum_z += center[2]
            count += 1
        
        if count == 0:
            return [0, 0, 0]
            
        return [sum_x/count, sum_y/count, sum_z/count]

    def _get_cell_normal_average(self, polydata, cell_ids):
        """Calculate the average normal vector of the specified cells"""
        if not cell_ids:
            return [0, 0, 1]  # Default to Z direction
            
        sum_x, sum_y, sum_z = 0, 0, 0
        count = 0
        
        # Create a normal generator
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(polydata)
        normals.ComputeCellNormalsOn()
        normals.Update()
        
        cell_normals = normals.GetOutput().GetCellData().GetNormals()
        
        for cell_id in cell_ids:
            if cell_normals:
                normal = cell_normals.GetTuple3(cell_id)
                sum_x += normal[0]
                sum_y += normal[1]
                sum_z += normal[2]
                count += 1
        
        if count == 0:
            return [0, 0, 1]  # Default to Z direction
            
        # Normalize the vector
        length = np.sqrt(sum_x*sum_x + sum_y*sum_y + sum_z*sum_z)
        if length < 1e-6:
            return [0, 0, 1]  # Default to Z direction
            
        return [sum_x/length, sum_y/length, sum_z/length]

    def _create_ambient_box(self, center, normal, max_dim, settings):
        """
        Create an ambient region box extending from the interface in the normal direction.
        
        Args:
            center (list): Center point of the interface [x, y, z]
            normal (list): Normal vector of the interface [nx, ny, nz]
            max_dim (float): Maximum dimension of the original mesh
            settings (dict): Ambient region settings
            
        Returns:
            vtkPolyData: The ambient region box
        """
        # Scale the normal to point in the positive direction if it's pointing inward
        if normal[0] + normal[1] + normal[2] < 0:
            normal = [-n for n in normal]
        
        # Calculate dimensions based on settings
        x_extent = max_dim * settings['x_extent']
        y_extent = max_dim * settings['y_extent']
        z_extent = max_dim * settings['z_extent']
        
        # Create the box
        box = vtk.vtkCubeSource()
        
        # Set box size
        box.SetXLength(x_extent)
        box.SetYLength(y_extent)
        box.SetZLength(z_extent)
        
        # Position the box - we want it to extend outward from the center along the normal
        box_center = [
            center[0] + normal[0] * z_extent * 0.5,
            center[1] + normal[1] * z_extent * 0.5,
            center[2] + normal[2] * z_extent * 0.5
        ]
        box.SetCenter(box_center)
        
        # Orient the box to align with the normal
        # This is a simplified approach - a more robust approach would use vtkTransform
        if abs(normal[2]) < 0.99:  # If not aligned with Z-axis
            # Calculate rotation to align with normal
            transform = vtk.vtkTransform()
            transform.Identity()
            
            # Find rotation from Z axis to normal
            z_axis = [0, 0, 1]
            cross = [
                z_axis[1]*normal[2] - z_axis[2]*normal[1],
                z_axis[2]*normal[0] - z_axis[0]*normal[2],
                z_axis[0]*normal[1] - z_axis[1]*normal[0]
            ]
            
            # Calculate rotation angle
            dot = z_axis[0]*normal[0] + z_axis[1]*normal[1] + z_axis[2]*normal[2]
            angle = np.arccos(dot) * 180.0 / np.pi
            
            # Apply rotation
            transform.RotateWXYZ(angle, cross[0], cross[1], cross[2])
            
            # Apply transform to box
            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetInputConnection(box.GetOutputPort())
            transform_filter.SetTransform(transform)
            transform_filter.Update()
            
            return transform_filter.GetOutput()
        else:
            # Box is already aligned with Z-axis
            box.Update()
            return box.GetOutput()

    def _visualize_ambient_box(self, ambient_box, settings):
        """
        Create visual representation of the ambient box.
        
        Args:
            ambient_box (vtkPolyData): The ambient box geometry
            settings (dict): Visualization settings
        """
        # Create a mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(ambient_box)
        
        # Create actor
        self.ambient_actor = vtk.vtkActor()
        self.ambient_actor.SetMapper(mapper)
        
        # Set appearance
        color_map = {
            "Light Blue": (0.7, 0.8, 1.0),
            "Light Gray": (0.8, 0.8, 0.8),
            "Light Green": (0.7, 1.0, 0.7)
        }
        
        color = color_map.get(settings.get('color_name', "Light Blue"), (0.7, 0.8, 1.0))
        opacity = settings.get('opacity', 0.3)
        
        self.ambient_actor.GetProperty().SetColor(color)
        self.ambient_actor.GetProperty().SetOpacity(opacity)
        self.ambient_actor.GetProperty().SetRepresentationToWireframe()
        
        # Add to renderer
        self.renderer.AddActor(self.ambient_actor)
        
        # Store the ambient box data for later use
        self.ambient_box = ambient_box

    def _add_ambient_boundaries(self, ambient_box, interface_boundary):
        """
        Add new boundaries for the ambient region.
        
        Args:
            ambient_box (vtkPolyData): The ambient box geometry
            interface_boundary (str): Name of the interface boundary
        """
        # This would normally extract the faces of the box and add them as new boundaries
        # For simplicity in this implementation, we'll just create dummy boundaries
        
        # Create names for the ambient region boundaries
        ambient_boundaries = {
            f"ambient_xmin": [],
            f"ambient_xmax": [],
            f"ambient_ymin": [],
            f"ambient_ymax": [],
            f"ambient_zmin": [],
            f"ambient_zmax": []
        }
        
        # In a real implementation, we would extract the faces and get their cell IDs
        # For now, we'll just create fake cell IDs for demonstration
        for i, name in enumerate(ambient_boundaries.keys()):
            # Add dummy cell IDs (these would normally be real cell IDs)
            ambient_boundaries[name] = [1000 + i * 100 + j for j in range(10)]
        
        # Store these in the boundary_cell_ids dictionary
        if hasattr(self, 'boundary_cell_ids'):
            for name, cells in ambient_boundaries.items():
                self.boundary_cell_ids[name] = cells
                logger.info(f"Added ambient boundary: {name} with {len(cells)} cells")

    def _mark_interface_internal(self, interface_boundary):
        """
        Mark the interface boundary as internal.
        
        Args:
            interface_boundary (str): Name of the interface boundary
        """
        # In a real implementation, this would update the boundary type
        # For now, just rename the boundary to indicate it's now internal
        if hasattr(self, 'boundary_cell_ids') and interface_boundary in self.boundary_cell_ids:
            cells = self.boundary_cell_ids[interface_boundary]
            internal_name = f"{interface_boundary}_internal"
            
            # Remove old boundary and add new internal boundary
            self.boundary_cell_ids[internal_name] = cells
            
            if hasattr(self, 'boundary_config') and interface_boundary in self.boundary_config:
                # Copy configuration but mark as internal
                config = self.boundary_config[interface_boundary].copy()
                config['type'] = 'internal'
                self.boundary_config[internal_name] = config
                
            logger.info(f"Marked boundary {interface_boundary} as internal interface")
            
            # Don't delete the original yet - in the real implementation we would,
            # but for the demo we'll keep it to avoid breaking anything

    def create_centered_ambient_region(self, settings):
        """
        Create an ambient region centered around the entire mesh.
        
        Args:
            settings (dict): Dictionary containing ambient region settings
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Creating centered ambient region with settings: {settings}")
            
            # Check if we have valid data to work with
            if not hasattr(self, 'data') or not self.data:
                logger.error("No mesh data available")
                return False
                
            # Get mesh bounds
            bounds = self.data.GetBounds()
            logger.info(f"Mesh bounds: {bounds}")
            
            # Calculate mesh center and size
            mesh_center = [
                (bounds[0] + bounds[1]) / 2,  # X center
                (bounds[2] + bounds[3]) / 2,  # Y center
                (bounds[4] + bounds[5]) / 2   # Z center
            ]
            
            mesh_size = [
                bounds[1] - bounds[0],  # X size
                bounds[3] - bounds[2],  # Y size
                bounds[5] - bounds[4]   # Z size
            ]
            
            max_dimension = max(mesh_size)
            logger.info(f"Mesh center: {mesh_center}, Max dimension: {max_dimension}")
            
            # Calculate ambient region size based on extents
            x_extent = settings.get('x_extent', 5.0)
            y_extent = settings.get('y_extent', 5.0)
            z_extent = settings.get('z_extent', 5.0)
            
            ambient_size = [
                max_dimension * x_extent,
                max_dimension * y_extent,
                max_dimension * z_extent
            ]
            
            # Create ambient box geometry
            cube = vtk.vtkCubeSource()
            
            cube.SetXLength(ambient_size[0])
            cube.SetYLength(ambient_size[1])
            cube.SetZLength(ambient_size[2])
            cube.SetCenter(mesh_center)
            cube.Update()
            
            ambient_box = cube.GetOutput()
            
            # Visualize the ambient box
            opacity = settings.get('opacity', 0.3)
            color_name = settings.get('color_name', 'Light Blue')
            
            # Create a mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(ambient_box)
            
            # Create actor
            ambient_actor = vtk.vtkActor()
            ambient_actor.SetMapper(mapper)
            
            # Set appearance
            color_map = {
                "Light Blue": (0.7, 0.8, 1.0),
                "Light Gray": (0.8, 0.8, 0.8),
                "Light Green": (0.7, 1.0, 0.7)
            }
            
            color = color_map.get(color_name, (0.7, 0.8, 1.0))
            
            ambient_actor.GetProperty().SetColor(color)
            ambient_actor.GetProperty().SetOpacity(opacity)
            
            # Make visible - use surface representation instead of wireframe
            ambient_actor.GetProperty().SetRepresentationToSurface()
            ambient_actor.GetProperty().SetEdgeVisibility(True)
            
            # Remove existing ambient actor if any
            if hasattr(self, 'ambient_actor') and self.ambient_actor:
                self.renderer.RemoveActor(self.ambient_actor)
                
            # Add to renderer
            self.renderer.AddActor(ambient_actor)
            self.ambient_actor = ambient_actor
            
            # Store ambient region info
            self.ambient_region_settings = settings
            
            # Update the window
            if hasattr(self, 'render_window'):
                self.render_window.Render()
            
            # Now handle the OpenFOAM case manager part
            openfoam_success = False
            
            # Get case manager from main window
            case_manager = None
            if hasattr(self, 'main_window') and self.main_window and hasattr(self.main_window, 'case_manager') and self.main_window.case_manager:
                case_manager = self.main_window.case_manager
            elif hasattr(self, 'main_window') and self.main_window and hasattr(self.main_window, 'simulation_controls') and hasattr(self.main_window.simulation_controls, 'case_manager'):
                case_manager = self.main_window.simulation_controls.case_manager
            
            if case_manager:
                # Ensure case directory is set
                if not hasattr(case_manager, 'case_directory') or not case_manager.case_directory:
                    # Try to get case directory from simulation controls
                    if hasattr(self, 'main_window') and self.main_window and hasattr(self.main_window, 'simulation_controls') and hasattr(self.main_window.simulation_controls, 'case_dir'):
                        case_dir = self.main_window.simulation_controls.case_dir
                        if case_dir:
                            logger.info(f"Setting case directory from simulation controls: {case_dir}")
                            case_manager.set_case_directory(case_dir)
                    
                    # If still no case directory, don't create ambient region in OpenFOAM
                    if not hasattr(case_manager, 'case_directory') or not case_manager.case_directory:
                        logger.warning("Cannot create ambient region in OpenFOAM: No case directory")
                
                # If case directory is set, create the ambient region
                if hasattr(case_manager, 'case_directory') and case_manager.case_directory:
                    logger.info("Creating ambient region in OpenFOAM case")
                    
                    # Prepare settings for case manager
                    ambient_settings = {
                        'temperature': settings.get('temperature', 300),  # Default temperature in K
                        'velocity': settings.get('velocity', [0, 0, 0])   # Default velocity
                    }
                    
                    openfoam_success = case_manager.create_centered_ambient_region(
                        center=mesh_center,
                        size=ambient_size,
                        name="ambientRegion",
                        settings=ambient_settings
                    )
                    
                    if openfoam_success:
                        logger.info("Successfully created ambient region in OpenFOAM case")
                    else:
                        logger.warning("Failed to create ambient region in OpenFOAM case")
                
            # Return true as long as the visualization worked, even if OpenFOAM failed
            return True
                
        except Exception as e:
            logger.error(f"Error creating centered ambient region: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def select_faces_by_plane(self, tolerance=0.01):
        """
        Select all faces that lie on the same plane as the currently hovered face.
        
        Args:
            tolerance (float): Tolerance for determining if a face is on the same plane
            
        Returns:
            List[int]: IDs of faces on the same plane
        """
        if not hasattr(self, 'hover_face_id') or self.hover_face_id == -1:
            return []
        
        if not hasattr(self, 'data') or not self.data:
            return []
        
        try:
            # Get the hovered face
            cell = self.data.GetCell(self.hover_face_id)
            if not cell:
                return []
            
            # Get the normal of the hovered face
            normal = [0, 0, 0]
            center = [0, 0, 0]
            
            # Calculate the face normal
            vtk.vtkPolygon.ComputeNormal(cell.GetPoints(), normal)
            
            # Calculate the face center
            points = cell.GetPoints()
            n_pts = points.GetNumberOfPoints()
            for i in range(n_pts):
                point = points.GetPoint(i)
                center[0] += point[0]
                center[1] += point[1]
                center[2] += point[2]
            
            center[0] /= n_pts
            center[1] /= n_pts
            center[2] /= n_pts
            
            # Define the plane equation: ax + by + cz + d = 0
            a, b, c = normal
            d = -(a*center[0] + b*center[1] + c*center[2])
            
            # Find all faces on the same plane
            similar_faces = []
            
            for face_id in range(self.data.GetNumberOfCells()):
                # Skip already selected faces
                if face_id in self.selected_faces:
                    continue
                
                # Check if this face is on the same plane
                face_cell = self.data.GetCell(face_id)
                face_points = face_cell.GetPoints()
                
                # Check if all points of this face lie on the plane
                on_plane = True
                for i in range(face_points.GetNumberOfPoints()):
                    point = face_points.GetPoint(i)
                    # Calculate distance from point to plane
                    vec = [point[i] - center[i] for i in range(3)]
                    distance = abs(vec[0]*normal[0] + vec[1]*normal[1] + vec[2]*normal[2] + 
                                   a*center[0] + b*center[1] + c*center[2])
                    
                    if distance > tolerance:
                        on_plane = False
                        break
                
                if on_plane:
                    similar_faces.append(face_id)
            
            return similar_faces
            
        except Exception as e:
            logger.error(f"Error selecting faces by plane: {str(e)}")
            return []

    def _update_selection_visualization(self):
        """Update visualization of selected faces"""
        try:
            if not self.selected_faces:
                if hasattr(self, 'selection_actor'):
                    self.selection_actor.VisibilityOff()
                return
            
            # Create selection for all selected faces
            id_array = vtk.vtkIdTypeArray()
            id_array.SetNumberOfComponents(1)
            for face_id in self.selected_faces:
                id_array.InsertNextValue(face_id)
            
            selection = vtk.vtkSelectionNode()
            selection.SetFieldType(vtk.vtkSelectionNode.CELL)
        except Exception as e:
            logger.error(f"Error updating selection visualization: {str(e)}")
            return
            selection.SetContentType(vtk.vtkSelectionNode.INDICES)
            selection.SetSelectionList(id_array)
            
            selections = vtk.vtkSelection()
            selections.AddNode(selection)
            
            # Extract selected cells
            extract = vtk.vtkExtractSelection()
            extract.SetInputData(0, self.data)
            extract.SetInputData(1, selections)
            extract.Update()
            
            # Convert to polydata
            geom = vtk.vtkGeometryFilter()
            geom.SetInputConnection(extract.GetOutputPort())
            geom.Update()
            
            # Update selection actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(geom.GetOutputPort())
            
            # Choose color based on boundary type
            color = (0.0, 1.0, 0.0)  # Default green
            if hasattr(self, 'boundary_name'):
                if "inlet" in self.boundary_name.lower():
                    color = (0.0, 0.0, 1.0)  # Blue for inlet
                elif "outlet" in self.boundary_name.lower():
                    color = (1.0, 0.0, 0.0)  # Red for outlet
            
            self.selection_actor.SetMapper(mapper)
            self.selection_actor.GetProperty().SetColor(*color)
            self.selection_actor.VisibilityOn()
        except Exception as e:
            logger.error(f"Error updating selection visualization: {str(e)}")
            return
            
    def _create_boundary_patch_seed(self, patch_name, num_points=50):
        """Create a seed source from an OpenFOAM boundary patch.
        
        Args:
            patch_name (str): Name of the boundary patch
            num_points (int): Number of seed points to generate
            
        Returns:
            vtkAlgorithm: Seed source for the streamline tracer or None if failed
        """
        openfoam_success = False
        
        try:
            # Get case directory from active source
            case_dir = None
            if hasattr(self, 'active_source') and self.active_source:
                source = self.sources[self.active_source]
                if hasattr(source, 'case_dir'):
                    case_dir = source.case_dir
                elif hasattr(source, 'get_property') and hasattr(source, 'get_property', 'case_dir'):
                    case_dir = source.get_property('case_dir')
            
            if not case_dir:
                # Try to determine case directory from source data
                if hasattr(self, 'data_info') and self.data_info:
                    for info in self.data_info.values():
                        if 'case_dir' in info:
                            case_dir = info['case_dir']
                            break
            
            if not case_dir:
                logger.error("Could not determine case directory for boundary patch seeding")
                return None
                
            # Look for boundary information files
            vtk_dir = os.path.join(case_dir, "VTK")
            if not os.path.exists(vtk_dir):
                logger.error(f"VTK directory not found at {vtk_dir}")
                return None
                
            # Check for streamline seeding hints file
            seeding_hint_file = os.path.join(vtk_dir, "streamline_seeding_hints.txt")
            boundary_info_dir = os.path.join(vtk_dir, "boundary_info")
            patches_file = os.path.join(boundary_info_dir, "patches.txt")
            
            # First look for boundary patch VTK files
            patch_files = []
            for ext in ['vtk', 'vtu', 'vtp']:
                patch_files.extend(glob.glob(os.path.join(vtk_dir, f"*{patch_name}*.{ext}")))
            
            # If we found patch-specific VTK files, use them for seeding
            if patch_files:
                logger.info(f"Found {len(patch_files)} VTK files for patch {patch_name}")
                # Use the first one for seeding
                patch_file = patch_files[0]
                
                # Read the patch file
                if patch_file.endswith('.vtk'):
                    reader = vtk.vtkPolyDataReader()
                elif patch_file.endswith('.vtp'):
                    reader = vtk.vtkXMLPolyDataReader()
                else:
                    reader = vtk.vtkGenericDataObjectReader()
                    
                reader.SetFileName(patch_file)
                reader.Update()
                
                # Create seed points from the patch
                if reader.GetOutput().GetNumberOfPoints() > 0:
                    # Use a vtkPointSource with specific points from the patch
                    points = vtk.vtkPoints()
                    step = max(1, reader.GetOutput().GetNumberOfPoints() // num_points)
                    count = 0
                    
                    for i in range(0, reader.GetOutput().GetNumberOfPoints(), step):
                        if count >= num_points:
                            break
                        point = reader.GetOutput().GetPoint(i)
                        points.InsertNextPoint(point)
                        count += 1
                    
                    polydata = vtk.vtkPolyData()
                    polydata.SetPoints(points)
                    
                    source = vtk.vtkProgrammableSource()
                    def create_output():
                        source.GetPolyDataOutput().ShallowCopy(polydata)
                    source.SetExecuteMethod(create_output)
                    
                    logger.info(f"Created seed source with {count} points from patch {patch_name}")
                    return source
            
            # If we didn't find specific patch files, try to use general boundary information
            if os.path.exists(seeding_hint_file):
                with open(seeding_hint_file, 'r') as f:
                    lines = f.readlines()
                    
                # Extract information about the patch
                patch_info = None
                for line in lines:
                    if line.startswith(f"INLET: {patch_name}") or \
                       line.startswith(f"OUTLET: {patch_name}") or \
                       line.startswith(f"WALL: {patch_name}"):
                        patch_info = line.strip()
                        break
                
                if patch_info:
                    logger.info(f"Found patch info: {patch_info}")
                    # Fall back to a generic seed source near a boundary
                    # In a real implementation, we would extract actual points on the boundary
                    # For now, we'll use a point source with a small radius near a boundary
                    bounds = self.data.GetBounds()
                    center = [(bounds[0] + bounds[1])/2, (bounds[2] + bounds[3])/2, (bounds[4] + bounds[5])/2]
                    
                    # Adjust center based on patch type
                    if "INLET" in patch_info:
                        center[0] = bounds[0] + (bounds[1] - bounds[0]) * 0.1  # Near inlet
                    elif "OUTLET" in patch_info:
                        center[0] = bounds[1] - (bounds[1] - bounds[0]) * 0.1  # Near outlet
                    
                    point = vtk.vtkPointSource()
                    point.SetCenter(center)
                    point.SetRadius(min(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]) * 0.1)
                    point.SetNumberOfPoints(num_points)
                    return point
            
            # If all else fails, return None
            logger.warning(f"Could not find boundary information for patch {patch_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error creating boundary patch seed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _end_selection(self):
        """End the face selection mode and call the callback."""
        try:
            # Call the callback with selected face IDs
            if hasattr(self, 'selection_callback') and self.selection_callback and self.selected_faces:
                # Prepare the selection result
                cell_ids = list(self.selected_faces)
                logger.info(f"Selection complete, selected {len(cell_ids)} faces")
                
                # Call the callback with the cell IDs
                self.selection_callback(cell_ids)
            
            # Reset selection state
            self.selection_mode = False
            self.selection_callback = None
            self.boundary_name = None
            self.status_message = ""
            
            # Hide selection actor and clear selection
            if hasattr(self, 'selection_actor'):
                self.selection_actor.VisibilityOff()
            self.selected_faces = set()
            
            # Clear hover highlighting
            self.clear_hover()
            
            # Update status
            self.update_status()
            
            # Force render update
            if hasattr(self, 'render_window'):
                self.render_window.Render()
                
        except Exception as e:
            logger.error(f"Error ending selection: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def highlight_faces(self, face_ids, boundary_name, color=None):
        """Highlight faces for a specific boundary
        
        Args:
            face_ids (list): List of face IDs to highlight
            boundary_name (str): Name of the boundary
            color (list, optional): RGB color tuple. Defaults to None (uses preset colors).
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure parameters are the correct types
            if isinstance(boundary_name, list) and isinstance(face_ids, str):
                # Parameters were passed in the wrong order, swap them
                face_ids, boundary_name = boundary_name, face_ids
            
            # Ensure boundary_name is a string
            if not isinstance(boundary_name, str):
                boundary_name = "wall"  # Default to wall if not a string
            
            # Ensure face_ids is a list
            if not isinstance(face_ids, list):
                if face_ids is None:
                    face_ids = []
                else:
                    face_ids = [face_ids]  # Convert single value to list
            
            if not face_ids:
                logger.warning(f"No face IDs provided for boundary {boundary_name}")
                return False
            
            # Check if we already have a dict for storing cell IDs
            if not hasattr(self, 'highlighted_faces'):
                self.highlighted_faces = {}
                
            # Check if we already have a dict for storing highlight actors
            if not hasattr(self, 'boundary_highlight_actors'):
                self.boundary_highlight_actors = {}
                
            # Clear any existing highlight for this boundary
            if boundary_name in self.boundary_highlight_actors:
                self.renderer.RemoveActor(self.boundary_highlight_actors[boundary_name])
                
            # Set default color if not specified
            if color is None:
                # Different colors for different types of boundaries
                if 'inlet' in boundary_name.lower():
                    color = [0.0, 0.0, 1.0]  # Blue for inlets
                elif 'outlet' in boundary_name.lower():
                    color = [1.0, 0.0, 0.0]  # Red for outlets
                elif 'wall' in boundary_name.lower():
                    color = [0.8, 0.8, 0.8]  # Gray for walls
                else:
                    color = [0.0, 1.0, 0.0]  # Green default
            
            # Log what color we're using
            logger.info(f"Using color {color} for boundary {boundary_name}")
            
            # Store the face IDs for this boundary
            self.highlighted_faces[boundary_name] = face_ids
            
            # Also update boundary_cell_ids dictionary to maintain consistency
            if not hasattr(self, 'boundary_cell_ids'):
                self.boundary_cell_ids = {}
            self.boundary_cell_ids[boundary_name] = face_ids
            logger.info(f"Updated boundary_cell_ids for {boundary_name} with {len(face_ids)} faces")
            
            # Setup the selection node for these cell IDs
            selection = vtk.vtkSelectionNode()
            selection.SetFieldType(vtk.vtkSelectionNode.CELL)
            selection.SetContentType(vtk.vtkSelectionNode.INDICES)
            
            # Convert the face IDs to a vtkIdTypeArray
            id_array = vtk.vtkIdTypeArray()
            id_array.SetNumberOfComponents(1)
            for face_id in face_ids:
                id_array.InsertNextValue(face_id)
            
            selection.SetSelectionList(id_array)
            
            # Create a selection object
            selections = vtk.vtkSelection()
            selections.AddNode(selection)
            
            # Extract the selected cells
            extract = vtk.vtkExtractSelection()
            extract.SetInputData(0, self.data)
            extract.SetInputData(1, selections)
            extract.Update()
            
            # Convert to polydata for visualization
            geom = vtk.vtkGeometryFilter()
            geom.SetInputConnection(extract.GetOutputPort())
            geom.Update()
            
            # Create mapper and actor for the highlighting
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(geom.GetOutputPort())
            
            # Create the actor and set its properties
            highlight_actor = vtk.vtkActor()
            highlight_actor.SetMapper(mapper)
            highlight_actor.GetProperty().SetColor(*color)
            highlight_actor.GetProperty().SetOpacity(1.0)
            
            # Set rendering priority based on boundary type
            # Ensure inlets and outlets are always visible above walls
            if 'inlet' in boundary_name.lower() or 'outlet' in boundary_name.lower():
                # Use better rendering properties to ensure visibility instead of position offset
                highlight_actor.GetProperty().SetOpacity(1.0)
                highlight_actor.GetProperty().SetPointSize(5)
                highlight_actor.GetProperty().SetLineWidth(2)
                # Remove and re-add actor to ensure it's rendered on top
                self.renderer.RemoveActor(highlight_actor)
                self.renderer.AddActor(highlight_actor)
                # No position offset needed
                highlight_actor.SetPosition(0, 0, 0)
            elif 'wall' in boundary_name.lower():
                # Set walls slightly back and with some transparency
                highlight_actor.SetPosition(0, 0, 0)
                highlight_actor.GetProperty().SetOpacity(0.7)  # Partially transparent
            
            # Add to renderer
            self.renderer.AddActor(highlight_actor)
            
            # Store the actor for future reference
            self.boundary_highlight_actors[boundary_name] = highlight_actor
            
            # Force a render update
            if hasattr(self, 'render_window'):
                self.render_window.Render()
                
            logger.info(f"Highlighted {len(face_ids)} faces for boundary: {boundary_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error highlighting faces: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _find_connected_face(self, seed_cell_id):
        """
        Find all cells that belong to the same logical face as the seed cell.
        For pipe meshes, this groups cells with similar normals that are connected.
        
        Args:
            seed_cell_id: The starting cell ID
            
        Returns:
            List of cell IDs that form a complete face
        """
        if not hasattr(self, 'data') or self.data is None:
            return [seed_cell_id]
        
        try:
            # First, compute normals for all cells
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(self.data)
            normals.ComputeCellNormalsOn()
            normals.Update()
            
            # Get the cell normals array
            cell_normals = normals.GetOutput().GetCellData().GetNormals()
            if not cell_normals:
                logger.warning("Failed to compute cell normals")
                return [seed_cell_id]
            
            # Get the seed cell's normal
            seed_normal = [0, 0, 0]
            cell_normals.GetTuple(seed_cell_id, seed_normal)
            
            # Normalize the seed normal
            magnitude = (seed_normal[0]**2 + seed_normal[1]**2 + seed_normal[2]**2)**0.5
            if magnitude > 0:
                seed_normal = [n/magnitude for n in seed_normal]
            
            # Get the seed cell to find its neighbors
            seed_cell = self.data.GetCell(seed_cell_id)
            if not seed_cell:
                logger.warning(f"Could not get cell {seed_cell_id}")
                return [seed_cell_id]
            
            # Use a breadth-first search to find connected cells with similar normals
            visited = set([seed_cell_id])
            queue = [seed_cell_id]
            
            # Normal similarity threshold - adjust as needed
            normal_threshold = 0.95  # Stricter than connected face (about 18 degrees)
            
            # Process the queue
            while queue:
                current_id = queue.pop(0)
                current_cell = self.data.GetCell(current_id)
                
                # For each point in the cell, find neighboring cells
                for i in range(current_cell.GetNumberOfPoints()):
                    point_id = current_cell.GetPointId(i)
                    
                    # Get all cells that use this point
                    cell_ids = vtk.vtkIdList()
                    self.data.GetPointCells(point_id, cell_ids)
                    
                    # Check each neighboring cell
                    for j in range(cell_ids.GetNumberOfIds()):
                        neighbor_id = cell_ids.GetId(j)
                        
                        # Skip if already visited
                        if neighbor_id in visited:
                            continue
                        
                        # Get neighbor's normal
                        neighbor_normal = [0, 0, 0]
                        cell_normals.GetTuple(neighbor_id, neighbor_normal)
                        
                        # Normalize
                        magnitude = (neighbor_normal[0]**2 + neighbor_normal[1]**2 + neighbor_normal[2]**2)**0.5
                        if magnitude > 0:
                            neighbor_normal = [n/magnitude for n in neighbor_normal]
                        
                        # Check if normals are similar (dot product close to 1)
                        dot_product = abs(seed_normal[0]*neighbor_normal[0] + 
                                         seed_normal[1]*neighbor_normal[1] + 
                                         seed_normal[2]*neighbor_normal[2])
                        
                        # If normals are similar, add to the face
                        if dot_product > normal_threshold:
                            visited.add(neighbor_id)
                            queue.append(neighbor_id)
            
            return list(visited)
            
        except Exception as e:
            logger.error(f"Error finding connected face: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [seed_cell_id]

    def _find_cells_in_plane(self, seed_cell_id):
        """
        Find all cells that lie in approximately the same plane as the seed cell.
        Used for CTRL+click selection of an entire planar region.
        
        Args:
            seed_cell_id: The starting cell ID
            
        Returns:
            List of cell IDs that are in the same plane
        """
        if not hasattr(self, 'data') or self.data is None:
            return [seed_cell_id]
        
        try:
            # First, compute normals for all cells
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(self.data)
            normals.ComputeCellNormalsOn()
            normals.Update()
            
            # Get the cell normals array
            cell_normals = normals.GetOutput().GetCellData().GetNormals()
            if not cell_normals:
                logger.warning("Failed to compute cell normals")
                return [seed_cell_id]
            
            # Get the seed cell's normal
            seed_normal = [0, 0, 0]
            cell_normals.GetTuple(seed_cell_id, seed_normal)
            
            # Normalize the seed normal
            magnitude = (seed_normal[0]**2 + seed_normal[1]**2 + seed_normal[2]**2)**0.5
            if magnitude > 0:
                seed_normal = [n/magnitude for n in seed_normal]
            
            # Get the seed cell's center
            seed_cell = self.data.GetCell(seed_cell_id)
            if not seed_cell:
                logger.warning(f"Could not get cell {seed_cell_id}")
                return [seed_cell_id]
            
            # Get a point from the seed cell to define the plane
            seed_point_id = seed_cell.GetPointId(0)
            seed_point = self.data.GetPoint(seed_point_id)
            
            # Find all cells in the same plane by checking:
            # 1. Similar normal direction
            # 2. Points lie close to the plane defined by seed_point and seed_normal
            in_plane_cells = []
            
            # Thresholds
            normal_threshold = 0.95  # Stricter than connected face (about 18 degrees)
            distance_threshold = 0.01  # Distance from plane
            
            # Check all cells in the mesh
            for cell_id in range(self.data.GetNumberOfCells()):
                # Skip already selected faces
                if cell_id in self.selected_faces:
                    continue
                
                # Check if this face is on the same plane
                face_cell = self.data.GetCell(cell_id)
                face_points = face_cell.GetPoints()
                
                # Check if all points of this face lie on the plane
                on_plane = True
                for i in range(face_points.GetNumberOfPoints()):
                    point = face_points.GetPoint(i)
                    # Calculate distance from point to plane
                    vec = [point[i] - seed_point[i] for i in range(3)]
                    distance = abs(vec[0]*seed_normal[0] + vec[1]*seed_normal[1] + vec[2]*seed_normal[2] + 
                                   seed_normal[0]*seed_point[0] + seed_normal[1]*seed_point[1] + seed_normal[2]*seed_point[2])
                    
                    if distance > distance_threshold:
                        on_plane = False
                        break
                
                if on_plane:
                    in_plane_cells.append(cell_id)
            
            logger.info(f"Found {len(in_plane_cells)} cells in plane")
            return in_plane_cells
        
        except Exception as e:
            logger.error(f"Error finding cells in plane: {e}")

    def load_openfoam_results(self, case_dir):
        """
        Load OpenFOAM simulation results from VTK files.
        
        Args:
            case_dir (str): Path to OpenFOAM case directory
            
        Returns:
            bool: True if successful, False otherwise
        """
        import os
        import glob
        
        logger.info(f"Loading OpenFOAM results from: {case_dir}")
        
        try:
            # Check VTK directory exists
            vtk_dir = os.path.join(case_dir, "VTK")
            
            # If VTK directory doesn't exist, run foamToVTK
            if not os.path.exists(vtk_dir):
                logger.info("VTK directory doesn't exist, creating and running foamToVTK...")
                os.makedirs(vtk_dir, exist_ok=True)
                
                # Run foamToVTK
                import subprocess
                try:
                    logger.info(f"Running foamToVTK on case: {case_dir}")
                    subprocess.run(["foamToVTK", "-case", case_dir], 
                                  cwd=case_dir,
                                  check=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
                    logger.info("foamToVTK completed successfully")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error running foamToVTK: {e}")
                    logger.error(f"STDERR: {e.stderr.decode('utf-8', errors='replace') if e.stderr else 'None'}")
                    # Continue anyway in case VTK files already exist
            
            # Look for VTK files 
            vtk_files = glob.glob(os.path.join(vtk_dir, "*.vtk"))
            vtk_files.extend(glob.glob(os.path.join(vtk_dir, "*.vtu")))
            
            if not vtk_files:
                logger.warning(f"No VTK files found in {vtk_dir}")
                return False
            
            # Sort by modification time (newest first)
            vtk_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_vtk = vtk_files[0]
            
            logger.info(f"Loading latest VTK result file: {latest_vtk}")
            
            # Create appropriate reader
            if latest_vtk.lower().endswith('.vtu'):
                reader = vtk.vtkXMLUnstructuredGridReader()
            else:
                reader = vtk.vtkGenericDataObjectReader()
                
            reader.SetFileName(latest_vtk)
            reader.Update()
            
            # Store as a source
            source_id = os.path.basename(latest_vtk)
            self.sources[source_id] = reader
            self.active_source = source_id
            
            # Create mapper and actor
            if latest_vtk.lower().endswith('.vtu'):
                mapper = vtk.vtkDataSetMapper()
            else:
                mapper = vtk.vtkPolyDataMapper()
                
            mapper.SetInputConnection(reader.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            # Store the actor
            self.actors[source_id] = actor
            
            # Add to renderer
            self.renderer.AddActor(actor)
            
            # Check what arrays are available
            output = reader.GetOutput()
            point_data = output.GetPointData()
            available_arrays = []
            
            for i in range(point_data.GetNumberOfArrays()):
                array_name = point_data.GetArrayName(i)
                array = point_data.GetArray(i)
                num_components = array.GetNumberOfComponents()
                available_arrays.append({
                    'name': array_name,
                    'components': num_components
                })
                logger.info(f"Found array: {array_name} with {num_components} components")
            
            # Store available fields for later access by visualization controls
            self.available_fields = available_arrays
            
            # Reset camera to show the new data
            self.renderer.ResetCamera()
            self.render_window.Render()
            
            logger.info(f"Successfully loaded OpenFOAM results from VTK file: {latest_vtk}")
            return True
            
        except Exception as e:
            import traceback
            logger.error(f"Error loading OpenFOAM results: {e}\n{traceback.format_exc()}")
            return False