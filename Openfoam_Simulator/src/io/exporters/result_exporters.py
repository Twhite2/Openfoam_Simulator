#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result exporters for Openfoam_Simulator application.

This module provides exporters for OpenFOAM simulation results to various formats,
with specialized exporters for oil & gas industry data. These exporters allow
saving simulation results to common file formats for sharing, further analysis,
or documentation.
"""

import os
import sys
import logging
import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

# Import utility modules
from ...utils.logger import get_logger
from ...utils.unit_converter import convert_units
from ...models.results_model import ResultsModel

# Try to import ParaView modules
try:
    # For ParaView 5.7+
    from paraview.simple import *
    paraview_available = True
except ImportError:
    paraview_available = False
    pass

logger = get_logger(__name__)


class BaseExporter:
    """Base class for all result exporters."""
    
    def __init__(self, results_model: Optional['ResultsModel'] = None):
        """
        Initialize the base exporter.
        
        Args:
            results_model: Results model containing the data to export
        """
        self.results_model = results_model
    
    def export(self, filepath: str, **kwargs) -> bool:
        """
        Export results to file.
        
        Args:
            filepath: Path to export the results to
            **kwargs: Additional export options
            
        Returns:
            bool: True if export succeeded, False otherwise
        """
        raise NotImplementedError("Subclasses must implement export method")
    
    def set_results_model(self, results_model: 'ResultsModel'):
        """
        Set the results model to export.
        
        Args:
            results_model: Results model containing the data to export
        """
        self.results_model = results_model
    
    def _ensure_directory_exists(self, filepath: str):
        """
        Ensure the directory for the output file exists.
        
        Args:
            filepath: Path to create directory for
        """
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)


class VTKExporter(BaseExporter):
    """Exporter for VTK file formats (.vtk, .vtu, .vtp)."""
    
    def export(self, filepath: str, **kwargs) -> bool:
        """
        Export results to VTK format.
        
        Args:
            filepath: Path to export the results to
            **kwargs: Additional export options including:
                format: Format to use ('vtk', 'vtu', or 'vtp')
                fields: List of field names to export
                time_step: Time step to export (None for all)
                
        Returns:
            bool: True if export succeeded, False otherwise
        """
        if not self.results_model:
            logger.error("No results model provided for export")
            return False
        
        try:
            # Get export options
            format_type = kwargs.get('format', 'vtk').lower()
            fields = kwargs.get('fields', None)  # None means all fields
            time_step = kwargs.get('time_step', None)  # None means all time steps
            
            # Ensure file has correct extension
            if not filepath.lower().endswith(f'.{format_type}'):
                filepath = f"{filepath}.{format_type}"
            
            self._ensure_directory_exists(filepath)
            
            # If ParaView is available, use it for VTK export
            if paraview_available:
                return self._export_with_paraview(filepath, format_type, fields, time_step)
            else:
                # Fallback to direct VTK file writing using results model data
                return self._export_direct(filepath, format_type, fields, time_step)
            
        except Exception as e:
            logger.error(f"Error exporting to VTK: {e}")
            return False
    
    def _export_with_paraview(self, filepath: str, format_type: str, 
                             fields: Optional[List[str]], 
                             time_step: Optional[float]) -> bool:
        """
        Export results using ParaView.
        
        Args:
            filepath: Path to export the results to
            format_type: Format to use ('vtk', 'vtu', or 'vtp')
            fields: List of field names to export
            time_step: Time step to export
            
        Returns:
            bool: True if export succeeded, False otherwise
        """
        try:
            # Get case directory from results model
            case_dir = self.results_model.get_case_directory()
            if not case_dir:
                logger.error("No case directory available in results model")
                return False
            
            # Create .foam file for ParaView if it doesn't exist
            foam_file = os.path.join(case_dir, 'case.foam')
            if not os.path.exists(foam_file):
                with open(foam_file, 'w') as f:
                    f.write('')
            
            # Load the OpenFOAM case
            reader = OpenFOAMReader(FileName=foam_file)
            
            # Configure reader based on fields to export
            if fields:
                # Enable only the specified fields
                reader.CellArrays = fields
            
            # Set time step if specified
            if time_step is not None:
                # Find closest time step
                reader.UpdatePipeline()
                available_times = reader.TimestepValues
                if available_times:
                    closest_time = min(available_times, key=lambda x: abs(x - time_step))
                    reader.UpdatePipeline(closest_time)
            
            # Create writer based on format
            if format_type == 'vtk':
                writer = CreateWriter(filepath, reader)
            elif format_type == 'vtu':
                writer = XMLUnstructuredGridWriter(Input=reader, FileName=filepath)
            elif format_type == 'vtp':
                # For surface exports, we need to extract surface first
                surface = ExtractSurface(Input=reader)
                writer = XMLPolyDataWriter(Input=surface, FileName=filepath)
            else:
                logger.error(f"Unsupported VTK format: {format_type}")
                return False
            
            # Write the file
            writer.UpdatePipeline()
            
            logger.info(f"Successfully exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting with ParaView: {e}")
            return False
    
    def _export_direct(self, filepath: str, format_type: str, 
                      fields: Optional[List[str]], 
                      time_step: Optional[float]) -> bool:
        """
        Export results directly using VTK file format.
        
        Args:
            filepath: Path to export the results to
            format_type: Format to use ('vtk', 'vtu', or 'vtp')
            fields: List of field names to export
            time_step: Time step to export
            
        Returns:
            bool: True if export succeeded, False otherwise
        """
        try:
            # Try to import VTK
            try:
                import vtk
            except ImportError:
                logger.error("VTK is not available for direct export")
                return False
            
            # Get the data from results model
            if time_step is None:
                # Get latest time step if not specified
                time_steps = self.results_model.get_time_steps()
                if not time_steps:
                    logger.error("No time steps available in results model")
                    return False
                time_step = time_steps[-1]
            
            # Get mesh data
            mesh_data = self.results_model.get_mesh_data()
            if not mesh_data:
                logger.error("No mesh data available in results model")
                return False
            
            # Get field data
            if fields is None:
                # Get all available fields
                fields = self.results_model.get_field_names()
            
            field_data = {}
            for field in fields:
                field_data[field] = self.results_model.get_field_data(field, time_step)
            
            # Create VTK grid based on mesh type
            mesh_type = self.results_model.get_mesh_type()
            
            if mesh_type == 'structured':
                # Create structured grid
                grid = vtk.vtkStructuredGrid()
                # Set dimensions and points (implementation depends on mesh data format)
                # ...
            elif mesh_type == 'unstructured':
                # Create unstructured grid
                grid = vtk.vtkUnstructuredGrid()
                # Set cells and points (implementation depends on mesh data format)
                # ...
            elif mesh_type == 'polydata':
                # Create poly data
                grid = vtk.vtkPolyData()
                # Set polygons and points (implementation depends on mesh data format)
                # ...
            else:
                logger.error(f"Unsupported mesh type: {mesh_type}")
                return False
            
            # Add field data to grid
            for field_name, data in field_data.items():
                # Implementation depends on data format from results model
                
            
                # Create writer based on format
                if format_type == 'vtk':
                    writer = vtk.vtkDataSetWriter()
                elif format_type == 'vtu':
                    writer = vtk.vtkXMLUnstructuredGridWriter()
                elif format_type == 'vtp':
                    writer = vtk.vtkXMLPolyDataWriter()
                else:
                    logger.error(f"Unsupported VTK format: {format_type}")
                    return False
            
            # Set input and filename
            writer.SetInputData(grid)
            writer.SetFileName(filepath)
            
            # Write the file
            writer.Write()
            
            logger.info(f"Successfully exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error in direct VTK export: {e}")
            return False


class CSVExporter(BaseExporter):
    """Exporter for CSV format."""
    
    def export(self, filepath: str, **kwargs) -> bool:
        """
        Export results to CSV format.
        
        Args:
            filepath: Path to export the results to
            **kwargs: Additional export options including:
                fields: List of field names to export
                time_step: Time step to export (None for all)
                points: List of points to export (None for all)
                
        Returns:
            bool: True if export succeeded, False otherwise
        """
        if not self.results_model:
            logger.error("No results model provided for export")
            return False
        
        try:
            # Get export options
            fields = kwargs.get('fields', None)  # None means all fields
            time_step = kwargs.get('time_step', None)  # None means all time steps
            points = kwargs.get('points', None)  # None means all points
            
            # Ensure file has correct extension
            if not filepath.lower().endswith('.csv'):
                filepath = f"{filepath}.csv"
            
            self._ensure_directory_exists(filepath)
            
            # Get available fields if not specified
            if fields is None:
                fields = self.results_model.get_field_names()
            
            # Get available time steps if not specified
            if time_step is None:
                time_steps = self.results_model.get_time_steps()
            else:
                time_steps = [time_step]
            
            # Get point coordinates if specified
            if points is not None:
                point_data = {}
                for point in points:
                    point_data[tuple(point)] = {}
                    for field in fields:
                        for step in time_steps:
                            # Get field value at specific point and time
                            value = self.results_model.get_field_at_point(field, point, step)
                            if step not in point_data[tuple(point)]:
                                point_data[tuple(point)][step] = {}
                            point_data[tuple(point)][step][field] = value
                
                # Write point data to CSV
                with open(filepath, 'w', newline='') as csvfile:
                    fieldnames = ['Point', 'Time'] + fields
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for point, time_data in point_data.items():
                        for step, field_data in time_data.items():
                            row = {'Point': str(point), 'Time': step}
                            row.update(field_data)
                            writer.writerow(row)
            
            else:
                # Export field data for all points or cells
                for field in fields:
                    field_filepath = filepath
                    if len(fields) > 1:
                        # Create separate files for each field if multiple fields
                        base, ext = os.path.splitext(filepath)
                        field_filepath = f"{base}_{field}{ext}"
                    
                    with open(field_filepath, 'w', newline='') as csvfile:
                        fieldnames = ['X', 'Y', 'Z', 'Time', field]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        
                        for step in time_steps:
                            # Get field data for this time step
                            field_data = self.results_model.get_field_data(field, step)
                            coordinates = self.results_model.get_coordinates()
                            
                            if field_data is None or coordinates is None:
                                logger.warning(f"No data for field {field} at time {step}")
                                continue
                            
                            # Write data for each point
                            for i, (x, y, z) in enumerate(coordinates):
                                if i < len(field_data):
                                    row = {
                                        'X': x,
                                        'Y': y,
                                        'Z': z,
                                        'Time': step,
                                        field: field_data[i]
                                    }
                                    writer.writerow(row)
            
            logger.info(f"Successfully exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False


class JSONExporter(BaseExporter):
    """Exporter for JSON format."""
    
    def export(self, filepath: str, **kwargs) -> bool:
        """
        Export results to JSON format.
        
        Args:
            filepath: Path to export the results to
            **kwargs: Additional export options including:
                fields: List of field names to export
                time_step: Time step to export (None for all)
                formatted: Whether to format the JSON output (default: True)
                
        Returns:
            bool: True if export succeeded, False otherwise
        """
        if not self.results_model:
            logger.error("No results model provided for export")
            return False
        
        try:
            # Get export options
            fields = kwargs.get('fields', None)  # None means all fields
            time_step = kwargs.get('time_step', None)  # None means all time steps
            formatted = kwargs.get('formatted', True)  # Whether to format the JSON output
            
            # Ensure file has correct extension
            if not filepath.lower().endswith('.json'):
                filepath = f"{filepath}.json"
            
            self._ensure_directory_exists(filepath)
            
            # Get available fields if not specified
            if fields is None:
                fields = self.results_model.get_field_names()
            
            # Get available time steps if not specified
            if time_step is None:
                time_steps = self.results_model.get_time_steps()
            else:
                time_steps = [time_step]
            
            # Create JSON data structure
            json_data = {
                "metadata": {
                    "case_name": self.results_model.get_case_name(),
                    "export_time": self.results_model.get_current_time(),
                    "fields": fields,
                    "time_steps": time_steps
                },
                "data": {}
            }
            
            # Get mesh data if available
            mesh_type = self.results_model.get_mesh_type()
            if mesh_type:
                json_data["metadata"]["mesh_type"] = mesh_type
            
            coordinates = self.results_model.get_coordinates()
            if coordinates is not None:
                json_data["metadata"]["num_points"] = len(coordinates)
                
                # Only include coordinates if not too large
                if len(coordinates) <= 10000:  # Arbitrary limit to avoid huge files
                    json_data["coordinates"] = coordinates.tolist() if isinstance(coordinates, np.ndarray) else coordinates
            
            # Add field data for each time step
            for step in time_steps:
                json_data["data"][str(step)] = {}
                
                for field in fields:
                    field_data = self.results_model.get_field_data(field, step)
                    if field_data is not None:
                        # Convert to list if numpy array
                        if isinstance(field_data, np.ndarray):
                            field_data = field_data.tolist()
                        
                        # Check if data is too large
                        if isinstance(field_data, list) and len(field_data) > 10000:
                            # Store summary statistics instead of full data
                            if all(isinstance(x, (int, float)) for x in field_data):
                                json_data["data"][str(step)][field] = {
                                    "min": min(field_data),
                                    "max": max(field_data),
                                    "mean": sum(field_data) / len(field_data),
                                    "size": len(field_data),
                                    "truncated": True
                                }
                            else:
                                json_data["data"][str(step)][field] = {
                                    "size": len(field_data),
                                    "truncated": True
                                }
                        else:
                            json_data["data"][str(step)][field] = field_data
            
            # Write to file
            with open(filepath, 'w') as f:
                if formatted:
                    json.dump(json_data, f, indent=2)
                else:
                    json.dump(json_data, f)
            
            logger.info(f"Successfully exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return False


class ParaViewStateExporter(BaseExporter):
    """Exporter for ParaView state files (.pvsm)."""
    
    def export(self, filepath: str, **kwargs) -> bool:
        """
        Export ParaView visualization state to .pvsm file.
        
        Args:
            filepath: Path to export the state to
            **kwargs: Additional export options
                view: ParaView view object to save (if None, uses active view)
                sources: List of ParaView source objects to include
                
        Returns:
            bool: True if export succeeded, False otherwise
        """
        if not paraview_available:
            logger.error("ParaView is not available for state export")
            return False
        
        try:
            # Get export options
            view = kwargs.get('view', None)
            sources = kwargs.get('sources', None)
            
            # Ensure file has correct extension
            if not filepath.lower().endswith('.pvsm'):
                filepath = f"{filepath}.pvsm"
            
            self._ensure_directory_exists(filepath)
            
            # Export state
            SaveState(filepath)
            
            logger.info(f"Successfully exported ParaView state to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting ParaView state: {e}")
            return False


class ScreenshotExporter(BaseExporter):
    """Exporter for screenshots of visualization (.png, .jpg)."""
    
    def export(self, filepath: str, **kwargs) -> bool:
        """
        Export screenshot of current visualization.
        
        Args:
            filepath: Path to export the screenshot to
            **kwargs: Additional export options including:
                view: ParaView view object to capture (if None, uses active view)
                width: Image width in pixels (default: 1920)
                height: Image height in pixels (default: 1080)
                format: Image format ('png', 'jpg', etc.)
                
        Returns:
            bool: True if export succeeded, False otherwise
        """
        if not paraview_available:
            logger.error("ParaView is not available for screenshot export")
            return False
        
        try:
            # Get export options
            view = kwargs.get('view', GetActiveView())
            width = kwargs.get('width', 1920)
            height = kwargs.get('height', 1080)
            image_format = kwargs.get('format', 'png').lower()
            
            # Ensure file has correct extension
            if not filepath.lower().endswith(f'.{image_format}'):
                filepath = f"{filepath}.{image_format}"
            
            self._ensure_directory_exists(filepath)
            
            # Set image size
            view.ViewSize = [width, height]
            
            # Save screenshot
            SaveScreenshot(filepath, view, ImageResolution=[width, height])
            
            logger.info(f"Successfully exported screenshot to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting screenshot: {e}")
            return False


class OilGasReportExporter(BaseExporter):
    """Specialized exporter for oil & gas industry reports."""
    
    def export(self, filepath: str, **kwargs) -> bool:
        """
        Export oil & gas specific analysis report.
        
        Args:
            filepath: Path to export the report to
            **kwargs: Additional export options including:
                report_type: Type of report ('flow', 'pigging', 'spill')
                template: Path to HTML template (optional)
                include_images: Whether to include images in report (default: True)
                
        Returns:
            bool: True if export succeeded, False otherwise
        """
        if not self.results_model:
            logger.error("No results model provided for export")
            return False
        
        try:
            # Get export options
            report_type = kwargs.get('report_type', 'flow')
            template_path = kwargs.get('template', None)
            include_images = kwargs.get('include_images', True)
            
            # Ensure file has correct extension
            if not filepath.lower().endswith('.html'):
                filepath = f"{filepath}.html"
            
            self._ensure_directory_exists(filepath)
            
            # If no template is provided, use default templates
            if template_path is None:
                # Get default template based on report type
                app_dir = Path(__file__).parent.parent.parent.parent
                templates_dir = app_dir / "templates" / "report_templates"
                
                if report_type == 'flow':
                    template_path = templates_dir / "flow_report.html"
                elif report_type == 'pigging':
                    template_path = templates_dir / "pigging_report.html"
                elif report_type == 'spill':
                    template_path = templates_dir / "spill_report.html"
                else:
                    logger.error(f"Unsupported report type: {report_type}")
                    return False
            
            # Check if template exists
            if not os.path.exists(template_path):
                logger.error(f"Template not found: {template_path}")
                return False
            
            # Read template
            with open(template_path, 'r') as f:
                template = f.read()
            
            # Generate image directory if including images
            image_dir = None
            if include_images:
                image_dir = os.path.splitext(filepath)[0] + "_images"
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)
            
            # Generate report content based on report type
            if report_type == 'flow':
                content = self._generate_flow_report(image_dir)
            elif report_type == 'pigging':
                content = self._generate_pigging_report(image_dir)
            elif report_type == 'spill':
                content = self._generate_spill_report(image_dir)
            else:
                logger.error(f"Unsupported report type: {report_type}")
                return False
            
            # Insert content into template
            for key, value in content.items():
                template = template.replace(f"{{{{ {key} }}}}", value)
            
            # Write report to file
            with open(filepath, 'w') as f:
                f.write(template)
            
            logger.info(f"Successfully exported report to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            return False
    
    def _generate_flow_report(self, image_dir: Optional[str]) -> Dict[str, str]:
        """
        Generate content for flow simulation report.
        
        Args:
            image_dir: Directory to save images to (None if not including images)
            
        Returns:
            Dict[str, str]: Dictionary of content to insert into template
        """
        # Generate flow report content
        content = {
            "title": f"Flow Simulation Report: {self.results_model.get_case_name()}",
            "date": self.results_model.get_current_time(),
            "summary": "This report summarizes the results of a flow simulation.",
            "simulation_parameters": self._format_simulation_parameters(),
            "results_summary": self._format_results_summary(),
            "charts": "",
            "conclusions": "Analysis of the simulation results shows..."
        }
        
        # Generate images if requested
        if image_dir is not None and paraview_available:
            # Generate velocity profile
            vel_image_path = os.path.join(image_dir, "velocity_profile.png")
            self._generate_velocity_profile(vel_image_path)
            content["charts"] += f'<div class="chart"><img src="{os.path.basename(image_dir)}/velocity_profile.png" alt="Velocity Profile"><p>Figure 1: Velocity Profile</p></div>'
            
            # Generate pressure contour
            pressure_image_path = os.path.join(image_dir, "pressure_contour.png")
            self._generate_pressure_contour(pressure_image_path)
            content["charts"] += f'<div class="chart"><img src="{os.path.basename(image_dir)}/pressure_contour.png" alt="Pressure Contour"><p>Figure 2: Pressure Contour</p></div>'
        
        return content
    
    def _generate_pigging_report(self, image_dir: Optional[str]) -> Dict[str, str]:
        """
        Generate content for pigging simulation report.
        
        Args:
            image_dir: Directory to save images to (None if not including images)
            
        Returns:
            Dict[str, str]: Dictionary of content to insert into template
        """
        # Generate pigging report content
        content = {
            "title": f"Pigging Simulation Report: {self.results_model.get_case_name()}",
            "date": self.results_model.get_current_time(),
            "summary": "This report summarizes the results of a pipeline pigging simulation.",
            "pig_parameters": self._format_pig_parameters(),
            "pipeline_parameters": self._format_pipeline_parameters(),
            "results_summary": self._format_pigging_results(),
            "charts": "",
            "conclusions": "Analysis of the pig run shows..."
        }
        
        # Generate images if requested
        if image_dir is not None and paraview_available:
            # Generate pig trajectory
            traj_image_path = os.path.join(image_dir, "pig_trajectory.png")
            self._generate_pig_trajectory(traj_image_path)
            content["charts"] += f'<div class="chart"><img src="{os.path.basename(image_dir)}/pig_trajectory.png" alt="Pig Trajectory"><p>Figure 1: Pig Trajectory</p></div>'
            
            # Generate pressure drop
            pressure_image_path = os.path.join(image_dir, "pressure_drop.png")
            self._generate_pressure_drop(pressure_image_path)
            content["charts"] += f'<div class="chart"><img src="{os.path.basename(image_dir)}/pressure_drop.png" alt="Pressure Drop"><p>Figure 2: Pressure Drop</p></div>'
        
        return content
    
    def _generate_spill_report(self, image_dir: Optional[str]) -> Dict[str, str]:
        """
        Generate content for spill simulation report.
        
        Args:
            image_dir: Directory to save images to (None if not including images)
            
        Returns:
            Dict[str, str]: Dictionary of content to insert into template
        """
        # Generate spill report content
        content = {
            "title": f"Spill Simulation Report: {self.results_model.get_case_name()}",
            "date": self.results_model.get_current_time(),
            "summary": "This report summarizes the results of a spill simulation.",
            "spill_parameters": self._format_spill_parameters(),
            "environmental_parameters": self._format_environmental_parameters(),
            "results_summary": self._format_spill_results(),
            "charts": "",
            "conclusions": "Analysis of the spill simulation shows..."
        }
        
        # Generate images if requested
        if image_dir is not None and paraview_available:
            # Generate spill contour
            contour_image_path = os.path.join(image_dir, "spill_contour.png")
            self._generate_spill_contour(contour_image_path)
            content["charts"] += f'<div class="chart"><img src="{os.path.basename(image_dir)}/spill_contour.png" alt="Spill Contour"><p>Figure 1: Spill Contour</p></div>'
            
            # Generate spill volume chart
            volume_image_path = os.path.join(image_dir, "spill_volume.png")
            self._generate_spill_volume(volume_image_path)
            content["charts"] += f'<div class="chart"><img src="{os.path.basename(image_dir)}/spill_volume.png" alt="Spill Volume"><p>Figure 2: Spill Volume Over Time</p></div>'
        
        return content
    
    def _format_simulation_parameters(self) -> str:
        """Format simulation parameters into HTML."""
        # Get simulation parameters from results model
        params = self.results_model.get_simulation_parameters()
        if not params:
            return "<p>No simulation parameters available.</p>"
        
        html = "<table class='params-table'>"
        html += "<tr><th>Parameter</th><th>Value</th></tr>"
        
        for key, value in params.items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"
        
        html += "</table>"
        return html
    
    def _format_results_summary(self) -> str:
        """Format results summary into HTML."""
        # Get results summary from results model
        summary = self.results_model.get_results_summary()
        if not summary:
            return "<p>No results summary available.</p>"
        
        html = "<table class='results-table'>"
        html += "<tr><th>Parameter</th><th>Minimum</th><th>Maximum</th><th>Average</th></tr>"
        
        for field, stats in summary.items():
            html += f"<tr><td>{field}</td><td>{stats.get('min', 'N/A')}</td><td>{stats.get('max', 'N/A')}</td><td>{stats.get('avg', 'N/A')}</td></tr>"
        
        html += "</table>"
        return html
    
    def _format_pig_parameters(self) -> str:
        """Format pig parameters into HTML."""
        # Get pig parameters from results model
        params = self.results_model.get_pig_parameters()
        if not params:
            return "<p>No pig parameters available.</p>"
        
        html = "<table class='params-table'>"
        html += "<tr><th>Parameter</th><th>Value</th></tr>"
        
        for key, value in params.items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"
        
        html += "</table>"
        return html
    
    def _format_pipeline_parameters(self) -> str:
        """Format pipeline parameters into HTML."""
        # Get pipeline parameters from results model
        params = self.results_model.get_pipeline_parameters()
        if not params:
            return "<p>No pipeline parameters available.</p>"
        
        html = "<table class='params-table'>"
        html += "<tr><th>Parameter</th><th>Value</th></tr>"
        
        for key, value in params.items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"
        
        html += "</table>"
        return html
    
    def _format_pigging_results(self) -> str:
        """Format pigging results into HTML."""
        # Get pigging results from results model
        results = self.results_model.get_pigging_results()
        if not results:
            return "<p>No pigging results available.</p>"
        
        html = "<table class='results-table'>"
        html += "<tr><th>Parameter</th><th>Value</th></tr>"
        
        for key, value in results.items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"
        
        html += "</table>"
        return html
    
    def _format_spill_parameters(self) -> str:
        """Format spill parameters into HTML."""
        # Get spill parameters from results model
        params = self.results_model.get_spill_parameters()
        if not params:
            return "<p>No spill parameters available.</p>"
        
        html = "<table class='params-table'>"
        html += "<tr><th>Parameter</th><th>Value</th></tr>"
        
        for key, value in params.items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"
        
        html += "</table>"
        return html
    
    def _format_environmental_parameters(self) -> str:
        """Format environmental parameters into HTML."""
        # Get environmental parameters from results model
        params = self.results_model.get_environmental_parameters()
        if not params:
            return "<p>No environmental parameters available.</p>"
        
        html = "<table class='params-table'>"
        html += "<tr><th>Parameter</th><th>Value</th></tr>"
        
        for key, value in params.items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"
        
        html += "</table>"
        return html
    
    def _format_spill_results(self) -> str:
        """Format spill results into HTML."""
        # Get spill results from results model
        results = self.results_model.get_spill_results()
        if not results:
            return "<p>No spill results available.</p>"
        
        html = "<table class='results-table'>"
        html += "<tr><th>Parameter</th><th>Value</th></tr>"
        
        for key, value in results.items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"
        
        html += "</table>"
        return html
    
    # Methods to generate visualization images for reports
    def _generate_velocity_profile(self, filepath: str):
        """Generate velocity profile image."""
        if not paraview_available:
            return
        
        # Implementation would use ParaView to generate custom visualizations
        # and save them to the specified filepath
        pass
    
    def _generate_pressure_contour(self, filepath: str):
        """Generate pressure contour image."""
        if not paraview_available:
            return
        
        # ParaView implementation
        pass
    
    def _generate_pig_trajectory(self, filepath: str):
        """Generate pig trajectory image."""
        if not paraview_available:
            return
        
        # ParaView implementation
        pass
    
    def _generate_pressure_drop(self, filepath: str):
        """Generate pressure drop image."""
        if not paraview_available:
            return
        
        # ParaView implementation
        pass
    
    def _generate_spill_contour(self, filepath: str):
        """Generate spill contour image."""
        if not paraview_available:
            return
        
        # ParaView implementation
        pass
    
    def _generate_spill_volume(self, filepath: str):
        """Generate spill volume chart."""
        if not paraview_available:
            return
        
        # ParaView implementation
        pass


# Factory function to create appropriate exporter
def create_exporter(format_type: str, results_model: Optional['ResultsModel'] = None) -> BaseExporter:
    """
    Create an exporter for the specified format.
    
    Args:
        format_type: Format to export to ('vtk', 'csv', 'json', etc.)
        results_model: Results model containing the data to export
        
    Returns:
        BaseExporter: An exporter instance for the specified format
        
    Raises:
        ValueError: If format_type is not supported
    """
    format_type = format_type.lower()
    
    if format_type in ['vtk', 'vtu', 'vtp']:
        return VTKExporter(results_model)
    elif format_type == 'csv':
        return CSVExporter(results_model)
    elif format_type == 'json':
        return JSONExporter(results_model)
    elif format_type == 'pvsm':
        return ParaViewStateExporter(results_model)
    elif format_type in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
        return ScreenshotExporter(results_model)
    elif format_type == 'html':
        return OilGasReportExporter(results_model)
    else:
        raise ValueError(f"Unsupported export format: {format_type}")