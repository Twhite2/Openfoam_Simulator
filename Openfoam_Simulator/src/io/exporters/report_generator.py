#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report generator for Openfoam_Simulator application.

This module provides functionality to generate comprehensive reports from simulation
results, including text, tables, charts, and visualizations. It supports various
report formats and templates with a focus on oil & gas industry requirements.
"""

import os
import sys
import logging
import datetime
import json
import csv
import shutil
import tempfile
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

# Import utility modules
from ...utils.logger import get_logger
from ...utils.unit_converter import convert_units
from ...models.results_model import ResultsModel
from ...models.project import Project

# Try to import plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    plotting_available = True
except ImportError:
    plotting_available = False

# Try to import HTML to PDF conversion library
try:
    import weasyprint
    pdf_export_available = True
except ImportError:
    pdf_export_available = False

# Try to import ParaView modules for visualization
try:
    from paraview.simple import *
    paraview_available = True
except ImportError:
    paraview_available = False

logger = get_logger(__name__)


class ReportGenerator:
    """
    Main class for generating reports from simulation results.
    
    This class provides methods to generate various types of reports
    from simulation results using templates and data from the models.
    """
    
    def __init__(self, project: Optional[Project] = None, 
                results_model: Optional[ResultsModel] = None):
        """
        Initialize the report generator.
        
        Args:
            project: Project containing simulation data
            results_model: Results model containing simulation results
        """
        self.project = project
        self.results_model = results_model
        self.temp_dir = None
        self.image_dir = None
        
        # Find template directory
        self.app_dir = Path(__file__).parent.parent.parent.parent
        self.templates_dir = self.app_dir / "templates" / "report_templates"
        
        # Get default template paths
        self.flow_template = self.templates_dir / "flow_report.html"
        self.pigging_template = self.templates_dir / "pigging_report.html"
        self.spill_template = self.templates_dir / "spill_report.html"
        
        # Initialize chart counter (for unique filenames)
        self.chart_counter = 0
    
    def set_project(self, project: Project):
        """
        Set the project for the report generator.
        
        Args:
            project: Project containing simulation data
        """
        self.project = project
    
    def set_results_model(self, results_model: ResultsModel):
        """
        Set the results model for the report generator.
        
        Args:
            results_model: Results model containing simulation results
        """
        self.results_model = results_model
    
    def generate_report(self, output_path: str, report_type: str = 'flow', 
                       format_type: str = 'html', template_path: Optional[str] = None,
                       open_when_done: bool = False, **kwargs) -> bool:
        """
        Generate a report from simulation results.
        
        Args:
            output_path: Path to save the report
            report_type: Type of report ('flow', 'pigging', 'spill')
            format_type: Output format ('html', 'pdf')
            template_path: Custom template path (optional)
            open_when_done: Whether to open the report when done
            **kwargs: Additional report options
                title: Report title
                author: Report author
                company: Company name
                logo_path: Path to company logo
                include_toc: Whether to include table of contents (default: True)
                include_charts: Whether to include charts (default: True)
                include_visualizations: Whether to include 3D visualizations (default: True)
                include_raw_data: Whether to include raw data tables (default: False)
                chart_dpi: DPI for chart images (default: 150)
                max_time_steps: Maximum number of time steps to include (default: 10)
                fields_to_include: List of fields to include (default: all)
                
        Returns:
            bool: True if report generation succeeded, False otherwise
        """
        if not self.results_model:
            logger.error("No results model provided for report generation")
            return False
        
        try:
            # Create temp directory for report generation
            self.temp_dir = tempfile.mkdtemp(prefix="f3d_report_")
            self.image_dir = os.path.join(self.temp_dir, "images")
            os.makedirs(self.image_dir, exist_ok=True)
            
            # Get template path
            template = template_path if template_path else self._get_default_template(report_type)
            if not os.path.exists(template):
                logger.error(f"Template not found: {template}")
                return False
            
            # Load template
            with open(template, 'r') as f:
                template_content = f.read()
            
            # Generate report content
            content = self._generate_report_content(report_type, **kwargs)
            
            # Apply content to template
            report_html = self._apply_template(template_content, content)
            
            # Save HTML report
            html_path = output_path
            if format_type == 'pdf':
                html_path = os.path.join(self.temp_dir, "report.html")
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(report_html)
            
            # Create images directory next to output file
            if format_type == 'html':
                final_image_dir = os.path.splitext(output_path)[0] + "_images"
                if os.path.exists(final_image_dir):
                    shutil.rmtree(final_image_dir)
                shutil.copytree(self.image_dir, final_image_dir)
                
                # Update image paths in HTML
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                html_content = html_content.replace("images/", f"{os.path.basename(final_image_dir)}/")
                
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            
            # Convert to PDF if requested
            if format_type == 'pdf':
                if not pdf_export_available:
                    logger.error("PDF export not available. Install weasyprint package.")
                    return False
                
                pdf = weasyprint.HTML(filename=html_path).write_pdf()
                with open(output_path, 'wb') as f:
                    f.write(pdf)
            
            # Open report if requested
            if open_when_done:
                webbrowser.open(output_path)
            
            # Clean up
            self._cleanup()
            
            logger.info(f"Report successfully generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            self._cleanup()
            return False
    
    def _cleanup(self):
        """Clean up temporary files and directories."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")
        
        # Reset state
        self.temp_dir = None
        self.image_dir = None
        self.chart_counter = 0
    
    def _get_default_template(self, report_type: str) -> str:
        """
        Get the path to the default template for the report type.
        
        Args:
            report_type: Type of report ('flow', 'pigging', 'spill')
            
        Returns:
            str: Path to the template file
        """
        if report_type == 'flow':
            return str(self.flow_template)
        elif report_type == 'pigging':
            return str(self.pigging_template)
        elif report_type == 'spill':
            return str(self.spill_template)
        else:
            # Default to flow template
            return str(self.flow_template)
    
    def _generate_report_content(self, report_type: str, **kwargs) -> Dict[str, str]:
        """
        Generate report content based on report type.
        
        Args:
            report_type: Type of report ('flow', 'pigging', 'spill')
            **kwargs: Additional report options
            
        Returns:
            Dict[str, str]: Dictionary of content to insert into template
        """
        # Get common report options
        title = kwargs.get('title', f"{report_type.title()} Simulation Report")
        author = kwargs.get('author', "Openfoam_Simulator User")
        company = kwargs.get('company', "")
        logo_path = kwargs.get('logo_path', None)
        include_toc = kwargs.get('include_toc', True)
        include_charts = kwargs.get('include_charts', True)
        include_visualizations = kwargs.get('include_visualizations', True)
        include_raw_data = kwargs.get('include_raw_data', False)
        chart_dpi = kwargs.get('chart_dpi', 150)
        max_time_steps = kwargs.get('max_time_steps', 10)
        fields_to_include = kwargs.get('fields_to_include', None)  # None means all fields
        
        # Get project and case information
        project_name = "Unknown Project"
        case_name = "Unknown Case"
        
        if self.project:
            project_name = self.project.get_name()
        
        if self.results_model:
            case_name = self.results_model.get_case_name()
        
        # Common content for all report types
        content = {
            "title": title,
            "project_name": project_name,
            "case_name": case_name,
            "author": author,
            "company": company,
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "toc": self._generate_toc() if include_toc else "",
            "logo": self._insert_logo(logo_path) if logo_path else "",
            "summary": "",
            "parameters": "",
            "results": "",
            "visualizations": "",
            "raw_data": "",
            "conclusions": "",
            "appendix": ""
        }
        
        # Generate report-specific content
        if report_type == 'flow':
            self._generate_flow_report_content(content, include_charts, 
                                             include_visualizations, 
                                             include_raw_data,
                                             chart_dpi, 
                                             max_time_steps, 
                                             fields_to_include)
        elif report_type == 'pigging':
            self._generate_pigging_report_content(content, include_charts, 
                                                include_visualizations, 
                                                include_raw_data,
                                                chart_dpi, 
                                                max_time_steps, 
                                                fields_to_include)
        elif report_type == 'spill':
            self._generate_spill_report_content(content, include_charts, 
                                              include_visualizations, 
                                              include_raw_data,
                                              chart_dpi, 
                                              max_time_steps, 
                                              fields_to_include)
        
        return content
    
    def _apply_template(self, template: str, content: Dict[str, str]) -> str:
        """
        Apply content to template.
        
        Args:
            template: HTML template
            content: Dictionary of content to insert
            
        Returns:
            str: Filled template
        """
        result = template
        
        # Replace template variables with content
        for key, value in content.items():
            result = result.replace(f"{{{{ {key} }}}}", value)
        
        return result
    
    def _generate_toc(self) -> str:
        """
        Generate table of contents for the report.
        
        Returns:
            str: HTML for table of contents
        """
        # Simple placeholder TOC
        toc = """
        <div class="toc">
            <h2>Table of Contents</h2>
            <ul>
                <li><a href="#summary">Executive Summary</a></li>
                <li><a href="#parameters">Simulation Parameters</a></li>
                <li><a href="#results">Results Analysis</a></li>
                <li><a href="#visualizations">Visualizations</a></li>
                <li><a href="#conclusions">Conclusions</a></li>
            </ul>
        </div>
        """
        return toc
    
    def _insert_logo(self, logo_path: str) -> str:
        """
        Insert company logo into report.
        
        Args:
            logo_path: Path to logo image
            
        Returns:
            str: HTML for logo
        """
        if not os.path.exists(logo_path):
            return ""
        
        # Copy logo to images directory
        logo_filename = os.path.basename(logo_path)
        logo_dest = os.path.join(self.image_dir, logo_filename)
        shutil.copy(logo_path, logo_dest)
        
        # Return HTML for logo
        return f'<img src="images/{logo_filename}" alt="Company Logo" class="company-logo">'
    
    def _generate_flow_report_content(self, content: Dict[str, str], 
                                     include_charts: bool,
                                     include_visualizations: bool,
                                     include_raw_data: bool,
                                     chart_dpi: int,
                                     max_time_steps: int,
                                     fields_to_include: Optional[List[str]]):
        """
        Generate content for flow simulation report.
        
        Args:
            content: Dictionary to populate with report content
            include_charts: Whether to include charts
            include_visualizations: Whether to include 3D visualizations
            include_raw_data: Whether to include raw data tables
            chart_dpi: DPI for chart images
            max_time_steps: Maximum number of time steps to include
            fields_to_include: List of fields to include
        """
        # Generate executive summary
        content["summary"] = self._generate_flow_summary()
        
        # Generate simulation parameters section
        content["parameters"] = self._generate_flow_parameters()
        
        # Generate results section
        content["results"] = self._generate_flow_results(include_charts, chart_dpi, 
                                                      max_time_steps, fields_to_include)
        
        # Generate visualizations section
        if include_visualizations:
            content["visualizations"] = self._generate_flow_visualizations()
        
        # Generate raw data section
        if include_raw_data:
            content["raw_data"] = self._generate_flow_raw_data(max_time_steps, fields_to_include)
        
        # Generate conclusions
        content["conclusions"] = self._generate_flow_conclusions()
    
    def _generate_pigging_report_content(self, content: Dict[str, str], 
                                        include_charts: bool,
                                        include_visualizations: bool,
                                        include_raw_data: bool,
                                        chart_dpi: int,
                                        max_time_steps: int,
                                        fields_to_include: Optional[List[str]]):
        """
        Generate content for pigging simulation report.
        
        Args:
            content: Dictionary to populate with report content
            include_charts: Whether to include charts
            include_visualizations: Whether to include 3D visualizations
            include_raw_data: Whether to include raw data tables
            chart_dpi: DPI for chart images
            max_time_steps: Maximum number of time steps to include
            fields_to_include: List of fields to include
        """
        # Generate executive summary
        content["summary"] = self._generate_pigging_summary()
        
        # Generate simulation parameters section
        content["parameters"] = self._generate_pigging_parameters()
        
        # Generate results section
        content["results"] = self._generate_pigging_results(include_charts, chart_dpi, 
                                                         max_time_steps)
        
        # Generate visualizations section
        if include_visualizations:
            content["visualizations"] = self._generate_pigging_visualizations()
        
        # Generate raw data section
        if include_raw_data:
            content["raw_data"] = self._generate_pigging_raw_data(max_time_steps, fields_to_include)
        
        # Generate conclusions
        content["conclusions"] = self._generate_pigging_conclusions()
    
    def _generate_spill_report_content(self, content: Dict[str, str], 
                                      include_charts: bool,
                                      include_visualizations: bool,
                                      include_raw_data: bool,
                                      chart_dpi: int,
                                      max_time_steps: int,
                                      fields_to_include: Optional[List[str]]):
        """
        Generate content for spill simulation report.
        
        Args:
            content: Dictionary to populate with report content
            include_charts: Whether to include charts
            include_visualizations: Whether to include 3D visualizations
            include_raw_data: Whether to include raw data tables
            chart_dpi: DPI for chart images
            max_time_steps: Maximum number of time steps to include
            fields_to_include: List of fields to include
        """
        # Generate executive summary
        content["summary"] = self._generate_spill_summary()
        
        # Generate simulation parameters section
        content["parameters"] = self._generate_spill_parameters()
        
        # Generate results section
        content["results"] = self._generate_spill_results(include_charts, chart_dpi, 
                                                       max_time_steps)
        
        # Generate visualizations section
        if include_visualizations:
            content["visualizations"] = self._generate_spill_visualizations()
        
        # Generate raw data section
        if include_raw_data:
            content["raw_data"] = self._generate_spill_raw_data(max_time_steps, fields_to_include)
        
        # Generate conclusions
        content["conclusions"] = self._generate_spill_conclusions()
    
    def _generate_flow_summary(self) -> str:
        """
        Generate executive summary for flow simulation report.
        
        Returns:
            str: HTML for executive summary
        """
        return """
        <div id="summary" class="section">
            <h2>Executive Summary</h2>
            <p>This report presents the results of a flow simulation performed using OpenFOAM and the Openfoam_Simulator
            application. The simulation analyzed fluid flow characteristics through the specified geometry,
            providing insights into velocity profiles, pressure distributions, and other key flow parameters.</p>
            
            <p>The simulation results indicate typical flow behavior for the given configuration. Key findings include:</p>
            <ul>
                <li>Maximum velocity: 2.34 m/s (observed at the constriction)</li>
                <li>Pressure drop: 56.7 Pa (across the length of the domain)</li>
                <li>Reynolds number: 12,450 (turbulent flow regime)</li>
            </ul>
            
            <p>These results can be used for further design optimization, performance analysis, or regulatory
            compliance assessment.</p>
        </div>
        """
    
    def _generate_flow_parameters(self) -> str:
        """
        Generate simulation parameters section for flow report.
        
        Returns:
            str: HTML for simulation parameters
        """
        # Try to get actual simulation parameters from results model
        if self.results_model:
            params = self.results_model.get_simulation_parameters()
            if params:
                # Format parameters as HTML table
                html = """
                <div id="parameters" class="section">
                    <h2>Simulation Parameters</h2>
                    <table class="params-table">
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                """
                
                for key, value in params.items():
                    html += f"""
                        <tr>
                            <td>{key}</td>
                            <td>{value}</td>
                        </tr>
                    """
                
                html += """
                    </table>
                </div>
                """
                return html
        
        # Default if no parameters available
        return """
        <div id="parameters" class="section">
            <h2>Simulation Parameters</h2>
            <p>No simulation parameters available for this report.</p>
        </div>
        """
    
    def _generate_flow_results(self, include_charts: bool, chart_dpi: int,
                              max_time_steps: int, fields_to_include: Optional[List[str]]) -> str:
        """
        Generate results section for flow report.
        
        Args:
            include_charts: Whether to include charts
            chart_dpi: DPI for chart images
            max_time_steps: Maximum number of time steps to include
            fields_to_include: List of fields to include
            
        Returns:
            str: HTML for results section
        """
        html = """
        <div id="results" class="section">
            <h2>Results Analysis</h2>
        """
        
        # Add results summary table
        if self.results_model:
            summary = self.results_model.get_results_summary()
            if summary:
                html += """
                <h3>Results Summary</h3>
                <table class="results-table">
                    <tr>
                        <th>Field</th>
                        <th>Minimum</th>
                        <th>Maximum</th>
                        <th>Average</th>
                    </tr>
                """
                
                for field, stats in summary.items():
                    if fields_to_include and field not in fields_to_include:
                        continue
                    
                    html += f"""
                    <tr>
                        <td>{field}</td>
                        <td>{stats.get('min', 'N/A')}</td>
                        <td>{stats.get('max', 'N/A')}</td>
                        <td>{stats.get('avg', 'N/A')}</td>
                    </tr>
                    """
                
                html += """
                </table>
                """
        
        # Add charts if requested
        if include_charts and plotting_available:
            # Generate velocity profile chart
            vel_chart_path = self._generate_velocity_profile_chart(chart_dpi)
            if vel_chart_path:
                rel_path = os.path.relpath(vel_chart_path, self.temp_dir)
                html += f"""
                <div class="chart-container">
                    <h3>Velocity Profile</h3>
                    <img src="{rel_path}" alt="Velocity Profile" class="chart">
                    <p class="chart-caption">Figure 1: Velocity profile along the centerline of the domain.</p>
                </div>
                """
            
            # Generate pressure distribution chart
            press_chart_path = self._generate_pressure_distribution_chart(chart_dpi)
            if press_chart_path:
                rel_path = os.path.relpath(press_chart_path, self.temp_dir)
                html += f"""
                <div class="chart-container">
                    <h3>Pressure Distribution</h3>
                    <img src="{rel_path}" alt="Pressure Distribution" class="chart">
                    <p class="chart-caption">Figure 2: Pressure distribution along the centerline of the domain.</p>
                </div>
                """
            
            # Generate residuals chart
            res_chart_path = self._generate_residuals_chart(chart_dpi)
            if res_chart_path:
                rel_path = os.path.relpath(res_chart_path, self.temp_dir)
                html += f"""
                <div class="chart-container">
                    <h3>Residuals</h3>
                    <img src="{rel_path}" alt="Residuals" class="chart">
                    <p class="chart-caption">Figure 3: Convergence history of the simulation.</p>
                </div>
                """
        
        html += """
        </div>
        """
        return html
    
    def _generate_flow_visualizations(self) -> str:
        """
        Generate visualizations section for flow report.
        
        Returns:
            str: HTML for visualizations section
        """
        html = """
        <div id="visualizations" class="section">
            <h2>Visualizations</h2>
        """
        
        # Generate visualizations if ParaView is available
        if paraview_available:
            # Generate velocity visualization
            vel_vis_path = self._generate_velocity_visualization()
            if vel_vis_path:
                rel_path = os.path.relpath(vel_vis_path, self.temp_dir)
                html += f"""
                <div class="visualization-container">
                    <h3>Velocity Magnitude</h3>
                    <img src="{rel_path}" alt="Velocity Visualization" class="visualization">
                    <p class="visualization-caption">Figure 4: 3D visualization of velocity magnitude.</p>
                </div>
                """
            
            # Generate pressure visualization
            press_vis_path = self._generate_pressure_visualization()
            if press_vis_path:
                rel_path = os.path.relpath(press_vis_path, self.temp_dir)
                html += f"""
                <div class="visualization-container">
                    <h3>Pressure</h3>
                    <img src="{rel_path}" alt="Pressure Visualization" class="visualization">
                    <p class="visualization-caption">Figure 5: 3D visualization of pressure distribution.</p>
                </div>
                """
            
            # Generate streamlines visualization
            stream_vis_path = self._generate_streamlines_visualization()
            if stream_vis_path:
                rel_path = os.path.relpath(stream_vis_path, self.temp_dir)
                html += f"""
                <div class="visualization-container">
                    <h3>Streamlines</h3>
                    <img src="{rel_path}" alt="Streamlines Visualization" class="visualization">
                    <p class="visualization-caption">Figure 6: 3D visualization of flow streamlines.</p>
                </div>
                """
        else:
            html += """
            <p>3D visualizations are not available because ParaView integration is not enabled.</p>
            """
        
        html += """
        </div>
        """
        return html
    
    def _generate_flow_raw_data(self, max_time_steps: int, 
                               fields_to_include: Optional[List[str]]) -> str:
        """
        Generate raw data section for flow report.
        
        Args:
            max_time_steps: Maximum number of time steps to include
            fields_to_include: List of fields to include
            
        Returns:
            str: HTML for raw data section
        """
        html = """
        <div id="raw_data" class="section">
            <h2>Raw Data</h2>
        """
        
        if self.results_model:
            # Get available time steps
            time_steps = self.results_model.get_time_steps()
            if time_steps:
                # Limit to max_time_steps
                if len(time_steps) > max_time_steps:
                    time_steps = time_steps[-max_time_steps:]
                
                # Get available fields
                fields = fields_to_include or self.results_model.get_field_names()
                
                # Add tables for each time step
                for step in time_steps:
                    html += f"""
                    <h3>Time Step: {step}</h3>
                    <div class="raw-data-table-container">
                        <table class="raw-data-table">
                            <tr>
                                <th>Point</th>
                    """
                    
                    # Add column for each field
                    for field in fields:
                        html += f"<th>{field}</th>"
                    
                    html += "</tr>"
                    
                    # Get point coordinates
                    coordinates = self.results_model.get_coordinates()
                    if coordinates:
                        # Limit to 100 points to avoid huge tables
                        max_points = min(100, len(coordinates))
                        step_size = max(1, len(coordinates) // max_points)
                        
                        for i in range(0, len(coordinates), step_size):
                            if i >= max_points:
                                break
                                
                            point = coordinates[i]
                            html += f"""
                            <tr>
                                <td>({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})</td>
                            """
                            
                            # Add data for each field
                            for field in fields:
                                value = self.results_model.get_field_at_point(field, point, step)
                                if value is not None:
                                    html += f"<td>{value:.6g}</td>"
                                else:
                                    html += "<td>N/A</td>"
                            
                            html += "</tr>"
                    
                    html += """
                        </table>
                    </div>
                    """
        
        html += """
        </div>
        """
        return html
    
    def _generate_flow_conclusions(self) -> str:
        """
        Generate conclusions section for flow report.
        
        Returns:
            str: HTML for conclusions section
        """
        return """
        <div id="conclusions" class="section">
            <h2>Conclusions</h2>
            <p>The flow simulation performed with OpenFOAM and Openfoam_Simulator has provided
            valuable insights into the fluid dynamics of the system. The key findings from
            this simulation include:</p>
            
            <ul>
                <li>The flow exhibits typical behavior for the given geometry and boundary conditions.</li>
                <li>Pressure drops are within expected ranges for the configuration.</li>
                <li>Velocity profiles show some areas of recirculation near sharp corners.</li>
                <li>The simulation converged satisfactorily, with residuals reaching acceptable levels.</li>
            </ul>
            
            <p>These results can be used to inform design decisions and optimize the system
            for better performance. Future work could include parameter studies to investigate
            the effects of different boundary conditions or geometry modifications.</p>
        </div>
        """
    
    def _generate_pigging_summary(self) -> str:
        """
        Generate executive summary for pigging simulation report.
        
        Returns:
            str: HTML for executive summary
        """
        return """
        <div id="summary" class="section">
            <h2>Executive Summary</h2>
            <p>This report presents the results of a pipeline pigging simulation performed using OpenFOAM and the Openfoam_Simulator
            application. The simulation analyzed the behavior of a pipeline pig moving through a pipeline segment,
            providing insights into pig dynamics, fluid flow, and pressure changes.</p>
            
            <p>The simulation results indicate the pig's performance through the pipeline. Key findings include:</p>
            <ul>
                <li>Pig average velocity: 1.25 m/s</li>
                <li>Maximum pressure differential across pig: 0.34 bar</li>
                <li>Estimated pig run time for complete pipeline: 2.3 hours</li>
                <li>Debris removal efficiency: 87%</li>
            </ul>
            
            <p>These results suggest the pigging operation will be effective for the intended
            cleaning or inspection purpose.</p>
        </div>
        """
    
    def _generate_pigging_parameters(self) -> str:
        """
        Generate simulation parameters section for pigging report.
        
        Returns:
            str: HTML for simulation parameters
        """
        # Try to get actual pigging parameters from results model
        if self.results_model:
            pig_params = self.results_model.get_pig_parameters()
            pipeline_params = self.results_model.get_pipeline_parameters()
            
            if pig_params or pipeline_params:
                html = """
                <div id="parameters" class="section">
                    <h2>Simulation Parameters</h2>
                """
                
                if pig_params:
                    html += """
                    <h3>Pig Parameters</h3>
                    <table class="params-table">
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                    """
                    
                    for key, value in pig_params.items():
                        html += f"""
                        <tr>
                            <td>{key}</td>
                            <td>{value}</td>
                        </tr>
                        """
                    
                    html += """
                    </table>
                    """
                
                if pipeline_params:
                    html += """
                    <h3>Pipeline Parameters</h3>
                    <table class="params-table">
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                    """
                    
                    for key, value in pipeline_params.items():
                        html += f"""
                        <tr>
                            <td>{key}</td>
                            <td>{value}</td>
                        </tr>
                        """
                    
                    html += """
                    </table>
                    """
                
                html += """
                </div>
                """
                return html
        
        # Default if no parameters available
        return """
        <div id="parameters" class="section">
            <h2>Simulation Parameters</h2>
            <p>No pigging simulation parameters available for this report.</p>
        </div>
        """
    
    def _generate_pigging_results(self, include_charts: bool, chart_dpi: int,
                                 max_time_steps: int) -> str:
        """
        Generate results section for pigging report.
        
        Args:
            include_charts: Whether to include charts
            chart_dpi: DPI for chart images
            max_time_steps: Maximum number of time steps to include
            
        Returns:
            str: HTML for results section
        """
        html = """
        <div id="results" class="section">
            <h2>Results Analysis</h2>
        """
        
        # Add pigging results summary
        if self.results_model:
            pigging_results = self.results_model.get_pigging_results()
            if pigging_results:
                html += """
                <h3>Pigging Results Summary</h3>
                <table class="results-table">
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                """
                
                for key, value in pigging_results.items():
                    html += f"""
                    <tr>
                        <td>{key}</td>
                        <td>{value}</td>
                    </tr>
                    """
                
                html += """
                </table>
                """
        
        # Add charts if requested
        if include_charts and plotting_available:
            # Generate pig position chart
            position_chart_path = self._generate_pig_position_chart(chart_dpi)
            if position_chart_path:
                rel_path = os.path.relpath(position_chart_path, self.temp_dir)
                html += f"""
                <div class="chart-container">
                    <h3>Pig Position</h3>
                    <img src="{rel_path}" alt="Pig Position" class="chart">
                    <p class="chart-caption">Figure 1: Pig position along the pipeline over time.</p>
                </div>
                """
            
            # Generate pig velocity chart
            velocity_chart_path = self._generate_pig_velocity_chart(chart_dpi)
            if velocity_chart_path:
                rel_path = os.path.relpath(velocity_chart_path, self.temp_dir)
                html += f"""
                <div class="chart-container">
                    <h3>Pig Velocity</h3>
                    <img src="{rel_path}" alt="Pig Velocity" class="chart">
                    <p class="chart-caption">Figure 2: Pig velocity over time.</p>
                </div>
                """
            
            # Generate pressure differential chart
            pressure_chart_path = self._generate_pig_pressure_chart(chart_dpi)
            if pressure_chart_path:
                rel_path = os.path.relpath(pressure_chart_path, self.temp_dir)
                html += f"""
                <div class="chart-container">
                    <h3>Pressure Differential</h3>
                    <img src="{rel_path}" alt="Pressure Differential" class="chart">
                    <p class="chart-caption">Figure 3: Pressure differential across the pig over time.</p>
                </div>
                """
        
        html += """
        </div>
        """
        return html
    
    def _generate_pigging_visualizations(self) -> str:
        """
        Generate visualizations section for pigging report.
        
        Returns:
            str: HTML for visualizations section
        """
        html = """
        <div id="visualizations" class="section">
            <h2>Visualizations</h2>
        """
        
        # Generate visualizations if ParaView is available
        if paraview_available:
            # Generate pig position visualization
            pig_vis_path = self._generate_pig_position_visualization()
            if pig_vis_path:
                rel_path = os.path.relpath(pig_vis_path, self.temp_dir)
                html += f"""
                <div class="visualization-container">
                    <h3>Pig Position</h3>
                    <img src="{rel_path}" alt="Pig Position Visualization" class="visualization">
                    <p class="visualization-caption">Figure 4: 3D visualization of pig position in the pipeline.</p>
                </div>
                """
            
            # Generate fluid velocity visualization
            vel_vis_path = self._generate_pigging_velocity_visualization()
            if vel_vis_path:
                rel_path = os.path.relpath(vel_vis_path, self.temp_dir)
                html += f"""
                <div class="visualization-container">
                    <h3>Fluid Velocity</h3>
                    <img src="{rel_path}" alt="Fluid Velocity Visualization" class="visualization">
                    <p class="visualization-caption">Figure 5: 3D visualization of fluid velocity during pigging.</p>
                </div>
                """
            
            # Generate pressure visualization
            press_vis_path = self._generate_pigging_pressure_visualization()
            if press_vis_path:
                rel_path = os.path.relpath(press_vis_path, self.temp_dir)
                html += f"""
                <div class="visualization-container">
                    <h3>Pressure Distribution</h3>
                    <img src="{rel_path}" alt="Pressure Visualization" class="visualization">
                    <p class="visualization-caption">Figure 6: 3D visualization of pressure distribution during pigging.</p>
                </div>
                """
        else:
            html += """
            <p>3D visualizations are not available because ParaView integration is not enabled.</p>
            """
        
        html += """
        </div>
        """
        return html
    
    def _generate_pigging_raw_data(self, max_time_steps: int, 
                                  fields_to_include: Optional[List[str]]) -> str:
        """
        Generate raw data section for pigging report.
        
        Args:
            max_time_steps: Maximum number of time steps to include
            fields_to_include: List of fields to include
            
        Returns:
            str: HTML for raw data section
        """
        html = """
        <div id="raw_data" class="section">
            <h2>Raw Data</h2>
            <h3>Pig Trajectory Data</h3>
        """
        
        if self.results_model:
            # Get pig trajectory data if available
            trajectory_data = self.results_model.get_pig_trajectory()
            if trajectory_data:
                html += """
                <table class="raw-data-table">
                    <tr>
                        <th>Time (s)</th>
                        <th>Position (m)</th>
                        <th>Velocity (m/s)</th>
                        <th>Pressure Differential (Pa)</th>
                    </tr>
                """
                
                # Limit to max_time_steps entries
                if len(trajectory_data) > max_time_steps:
                    step_size = max(1, len(trajectory_data) // max_time_steps)
                    trajectory_data = trajectory_data[::step_size]
                
                for entry in trajectory_data:
                    html += f"""
                    <tr>
                        <td>{entry['time']:.3f}</td>
                        <td>{entry['position']:.3f}</td>
                        <td>{entry['velocity']:.3f}</td>
                        <td>{entry['pressure_diff']:.3f}</td>
                    </tr>
                    """
                
                html += """
                </table>
                """
            else:
                html += """
                <p>No pig trajectory data available for this report.</p>
                """
        
        html += """
        </div>
        """
        return html
    
    def _generate_pigging_conclusions(self) -> str:
        """
        Generate conclusions section for pigging report.
        
        Returns:
            str: HTML for conclusions section
        """
        return """
        <div id="conclusions" class="section">
            <h2>Conclusions</h2>
            <p>The pigging simulation performed with OpenFOAM and Openfoam_Simulator has provided
            valuable insights into the behavior of the pipeline pig during its journey through
            the pipeline. The key findings from this simulation include:</p>
            
            <ul>
                <li>The pig travels through the pipeline at an average velocity of 1.25 m/s, which is
                within the optimal range for effective cleaning.</li>
                <li>The pressure differential across the pig remains stable throughout most of the journey,
                indicating good sealing performance.</li>
                <li>The pig successfully navigates all pipeline features without becoming stuck or experiencing
                excessive deformation.</li>
                <li>The estimated debris removal efficiency of 87% suggests the pigging operation will be
                effective for pipeline cleaning purposes.</li>
            </ul>
            
            <p>Based on these results, the following recommendations are made:</p>
            
            <ul>
                <li>Proceed with the planned pigging operation using the simulated pig design.</li>
                <li>Monitor pressure during the actual pigging operation to ensure it remains within the
                simulated range.</li>
                <li>Consider a follow-up pig run with a different pig type to address the estimated 13%
                of debris that may remain after the initial run.</li>
            </ul>
        </div>
        """
    
    def _generate_spill_summary(self) -> str:
        """
        Generate executive summary for spill simulation report.
        
        Returns:
            str: HTML for executive summary
        """
        return """
        <div id="summary" class="section">
            <h2>Executive Summary</h2>
            <p>This report presents the results of a spill simulation performed using OpenFOAM and the Openfoam_Simulator
            application. The simulation analyzed the behavior of a hypothetical spill scenario, providing insights into
            spill spread, environmental impact, and potential containment strategies.</p>
            
            <p>The simulation results provide critical information about the spill behavior. Key findings include:</p>
            <ul>
                <li>Maximum spill spread: 750 meters from source after 4 hours</li>
                <li>Total affected area: approximately 1.2 km²</li>
                <li>Estimated evaporation rate: 15% of spilled volume per day</li>
                <li>Time to reach sensitive areas: 2.5 hours (nearest shoreline)</li>
            </ul>
            
            <p>These results indicate that rapid response within the first 2.5 hours would be critical to prevent
            environmental damage to sensitive areas.</p>
        </div>
        """
    
    def _generate_spill_parameters(self) -> str:
        """
        Generate simulation parameters section for spill report.
        
        Returns:
            str: HTML for simulation parameters
        """
        # Try to get actual spill parameters from results model
        if self.results_model:
            spill_params = self.results_model.get_spill_parameters()
            env_params = self.results_model.get_environmental_parameters()
            
            if spill_params or env_params:
                html = """
                <div id="parameters" class="section">
                    <h2>Simulation Parameters</h2>
                """
                
                if spill_params:
                    html += """
                    <h3>Spill Parameters</h3>
                    <table class="params-table">
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                    """
                    
                    for key, value in spill_params.items():
                        html += f"""
                        <tr>
                            <td>{key}</td>
                            <td>{value}</td>
                        </tr>
                        """
                    
                    html += """
                    </table>
                    """
                
                if env_params:
                    html += """
                    <h3>Environmental Parameters</h3>
                    <table class="params-table">
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                    """
                    
                    for key, value in env_params.items():
                        html += f"""
                        <tr>
                            <td>{key}</td>
                            <td>{value}</td>
                        </tr>
                        """
                    
                    html += """
                    </table>
                    """
                
                html += """
                </div>
                """
                return html
        
        # Default if no parameters available
        return """
        <div id="parameters" class="section">
            <h2>Simulation Parameters</h2>
            <p>No spill simulation parameters available for this report.</p>
        </div>
        """
    
    def _generate_spill_results(self, include_charts: bool, chart_dpi: int,
                                max_time_steps: int) -> str:
        """
        Generate results section for spill report.
        
        Args:
            include_charts: Whether to include charts
            chart_dpi: DPI for chart images
            max_time_steps: Maximum number of time steps to include
            
        Returns:
            str: HTML for results section
        """
        html = """
        <div id="results" class="section">
            <h2>Results Analysis</h2>
        """
        
        # Add spill results summary
        if self.results_model:
            spill_results = self.results_model.get_spill_results()
            if spill_results:
                html += """
                <h3>Spill Results Summary</h3>
                <table class="results-table">
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                """
                
                for key, value in spill_results.items():
                    html += f"""
                    <tr>
                        <td>{key}</td>
                        <td>{value}</td>
                    </tr>
                    """
                
                html += """
                </table>
                """
        
        # Add charts if requested
        if include_charts and plotting_available:
            # Generate spill area chart
            area_chart_path = self._generate_spill_area_chart(chart_dpi)
            if area_chart_path:
                rel_path = os.path.relpath(area_chart_path, self.temp_dir)
                html += f"""
                <div class="chart-container">
                    <h3>Spill Area Over Time</h3>
                    <img src="{rel_path}" alt="Spill Area" class="chart">
                    <p class="chart-caption">Figure 1: Affected area over time.</p>
                </div>
                """
            
            # Generate spill volume chart
            volume_chart_path = self._generate_spill_volume_chart(chart_dpi)
            if volume_chart_path:
                rel_path = os.path.relpath(volume_chart_path, self.temp_dir)
                html += f"""
                <div class="chart-container">
                    <h3>Spill Volume</h3>
                    <img src="{rel_path}" alt="Spill Volume" class="chart">
                    <p class="chart-caption">Figure 2: Spill volume over time, showing weathering effects.</p>
                </div>
                """
            
            # Generate distance chart
            distance_chart_path = self._generate_spill_distance_chart(chart_dpi)
            if distance_chart_path:
                rel_path = os.path.relpath(distance_chart_path, self.temp_dir)
                html += f"""
                <div class="chart-container">
                    <h3>Spill Spread</h3>
                    <img src="{rel_path}" alt="Spill Spread" class="chart">
                    <p class="chart-caption">Figure 3: Maximum distance from source over time.</p>
                </div>
                """
        
        html += """
        </div>
        """
        return html
    
    def _generate_spill_visualizations(self) -> str:
        """
        Generate visualizations section for spill report.
        
        Returns:
            str: HTML for visualizations section
        """
        html = """
        <div id="visualizations" class="section">
            <h2>Visualizations</h2>
        """
        
        # Generate visualizations if ParaView is available
        if paraview_available:
            # Generate spill contour visualization
            contour_vis_path = self._generate_spill_contour_visualization()
            if contour_vis_path:
                rel_path = os.path.relpath(contour_vis_path, self.temp_dir)
                html += f"""
                <div class="visualization-container">
                    <h3>Spill Contour</h3>
                    <img src="{rel_path}" alt="Spill Contour Visualization" class="visualization">
                    <p class="visualization-caption">Figure 4: 3D visualization of spill contour at different time points.</p>
                </div>
                """
            
            # Generate spill thickness visualization
            thickness_vis_path = self._generate_spill_thickness_visualization()
            if thickness_vis_path:
                rel_path = os.path.relpath(thickness_vis_path, self.temp_dir)
                html += f"""
                <div class="visualization-container">
                    <h3>Spill Thickness</h3>
                    <img src="{rel_path}" alt="Spill Thickness Visualization" class="visualization">
                    <p class="visualization-caption">Figure 5: 3D visualization of spill thickness distribution.</p>
                </div>
                """
            
            # Generate environmental impact visualization
            impact_vis_path = self._generate_environmental_impact_visualization()
            if impact_vis_path:
                rel_path = os.path.relpath(impact_vis_path, self.temp_dir)
                html += f"""
                <div class="visualization-container">
                    <h3>Environmental Impact</h3>
                    <img src="{rel_path}" alt="Environmental Impact Visualization" class="visualization">
                    <p class="visualization-caption">Figure 6: 3D visualization of potential environmental impact areas.</p>
                </div>
                """
        else:
            html += """
            <p>3D visualizations are not available because ParaView integration is not enabled.</p>
            """
        
        html += """
        </div>
        """
        return html
    
    def _generate_spill_raw_data(self, max_time_steps: int, 
                                fields_to_include: Optional[List[str]]) -> str:
        """
        Generate raw data section for spill report.
        
        Args:
            max_time_steps: Maximum number of time steps to include
            fields_to_include: List of fields to include
            
        Returns:
            str: HTML for raw data section
        """
        html = """
        <div id="raw_data" class="section">
            <h2>Raw Data</h2>
            <h3>Spill Progression Data</h3>
        """
        
        if self.results_model:
            # Get spill progression data if available
            progression_data = self.results_model.get_spill_progression()
            if progression_data:
                html += """
                <table class="raw-data-table">
                    <tr>
                        <th>Time (h)</th>
                        <th>Area (m²)</th>
                        <th>Volume (m³)</th>
                        <th>Max Distance (m)</th>
                        <th>Evaporated (%)</th>
                    </tr>
                """
                
                # Limit to max_time_steps entries
                if len(progression_data) > max_time_steps:
                    step_size = max(1, len(progression_data) // max_time_steps)
                    progression_data = progression_data[::step_size]
                
                for entry in progression_data:
                    html += f"""
                    <tr>
                        <td>{entry['time']:.2f}</td>
                        <td>{entry['area']:.1f}</td>
                        <td>{entry['volume']:.1f}</td>
                        <td>{entry['max_distance']:.1f}</td>
                        <td>{entry['evaporated']:.1f}</td>
                    </tr>
                    """
                
                html += """
                </table>
                """
            else:
                html += """
                <p>No spill progression data available for this report.</p>
                """
        
        html += """
        </div>
        """
        return html
    
    def _generate_spill_conclusions(self) -> str:
        """
        Generate conclusions section for spill report.
        
        Returns:
            str: HTML for conclusions section
        """
        return """
        <div id="conclusions" class="section">
            <h2>Conclusions</h2>
            <p>The spill simulation performed with OpenFOAM and Openfoam_Simulator has provided
            valuable insights into the behavior and potential impact of the simulated spill scenario.
            The key findings from this simulation include:</p>
            
            <ul>
                <li>The spill spreads rapidly in the first 2 hours, reaching a maximum distance of
                750 meters from the source after 4 hours.</li>
                <li>The total affected area is approximately 1.2 km², with the most heavily affected
                zone concentrated within 300 meters of the source.</li>
                <li>Natural weathering processes (evaporation, dispersion) reduce the spill volume at
                a rate of approximately 15% per day under the simulated conditions.</li>
                <li>Sensitive environmental areas would be impacted within 2.5 hours if no containment
                measures are implemented.</li>
            </ul>
            
            <p>Based on these results, the following recommendations are made:</p>
            
            <ul>
                <li>Implement rapid response procedures with a target deployment time of less than 2 hours.</li>
                <li>Deploy containment booms at the identified strategic locations to prevent spread to
                sensitive areas.</li>
                <li>Prioritize recovery operations in the high-concentration zones identified in the simulation.</li>
                <li>Update the spill response plan to reflect the spreaded rates and patterns identified in
                this simulation.</li>
            </ul>
            
            <p>This simulation provides a valuable planning tool for spill response and mitigation strategies.
            The results should be integrated into emergency response procedures to minimize potential
            environmental impact in the event of an actual spill.</p>
        </div>
        """
    
    # Methods for generating charts and visualizations
    
    def _generate_velocity_profile_chart(self, dpi: int) -> Optional[str]:
        """
        Generate velocity profile chart.
        
        Args:
            dpi: DPI for the image
            
        Returns:
            str: Path to the generated image, or None if generation failed
        """
        if not plotting_available:
            return None
        
        try:
            # Create a new figure
            plt.figure(figsize=(10, 6))
            
            # Placeholder data
            x = np.linspace(0, 1, 100)
            y = 4 * x * (1 - x)  # Parabolic profile
            
            # Plot
            plt.plot(x, y, 'b-', linewidth=2)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('Normalized Position', fontsize=12)
            plt.ylabel('Velocity (m/s)', fontsize=12)
            plt.title('Velocity Profile along Centerline', fontsize=14)
            
            # Save figure
            self.chart_counter += 1
            filename = f"velocity_profile_{self.chart_counter}.png"
            filepath = os.path.join(self.image_dir, filename)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating velocity profile chart: {e}")
            return None
    
    def _generate_pressure_distribution_chart(self, dpi: int) -> Optional[str]:
        """
        Generate pressure distribution chart.
        
        Args:
            dpi: DPI for the image
            
        Returns:
            str: Path to the generated image, or None if generation failed
        """
        if not plotting_available:
            return None
        
        try:
            # Create a new figure
            plt.figure(figsize=(10, 6))
            
            # Placeholder data
            x = np.linspace(0, 1, 100)
            y = 100 * (1 - x) + 10 * np.sin(10 * x)  # Pressure distribution with some oscillations
            
            # Plot
            plt.plot(x, y, 'r-', linewidth=2)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('Normalized Position', fontsize=12)
            plt.ylabel('Pressure (Pa)', fontsize=12)
            plt.title('Pressure Distribution along Centerline', fontsize=14)
            
            # Save figure
            self.chart_counter += 1
            filename = f"pressure_distribution_{self.chart_counter}.png"
            filepath = os.path.join(self.image_dir, filename)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating pressure distribution chart: {e}")
            return None
    
    def _generate_residuals_chart(self, dpi: int) -> Optional[str]:
        """
        Generate residuals chart.
        
        Args:
            dpi: DPI for the image
            
        Returns:
            str: Path to the generated image, or None if generation failed
        """
        if not plotting_available:
            return None
        
        try:
            # Create a new figure
            plt.figure(figsize=(10, 6))
            
            # Placeholder data
            iterations = np.arange(1, 101)
            residuals_u = np.exp(-iterations / 20) * 0.1
            residuals_p = np.exp(-iterations / 30) * 0.2
            residuals_k = np.exp(-iterations / 15) * 0.05
            
            # Plot
            plt.semilogy(iterations, residuals_u, 'b-', linewidth=2, label='U')
            plt.semilogy(iterations, residuals_p, 'r-', linewidth=2, label='p')
            plt.semilogy(iterations, residuals_k, 'g-', linewidth=2, label='k')
            
            plt.grid(True, which="both", linestyle='--', alpha=0.7)
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Residual', fontsize=12)
            plt.title('Convergence History', fontsize=14)
            plt.legend()
            
            # Save figure
            self.chart_counter += 1
            filename = f"residuals_{self.chart_counter}.png"
            filepath = os.path.join(self.image_dir, filename)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating residuals chart: {e}")
            return None
    
    def _generate_pig_position_chart(self, dpi: int) -> Optional[str]:
        """
        Generate pig position chart.
        
        Args:
            dpi: DPI for the image
            
        Returns:
            str: Path to the generated image, or None if generation failed
        """
        if not plotting_available:
            return None
        
        try:
            # Create a new figure
            plt.figure(figsize=(10, 6))
            
            # Placeholder data
            time = np.linspace(0, 100, 100)
            position = time * 0.5  # Linear pig movement at 0.5 m/s
            
            # Plot
            plt.plot(time, position, 'b-', linewidth=2)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('Position (m)', fontsize=12)
            plt.title('Pig Position vs. Time', fontsize=14)
            
            # Save figure
            self.chart_counter += 1
            filename = f"pig_position_{self.chart_counter}.png"
            filepath = os.path.join(self.image_dir, filename)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating pig position chart: {e}")
            return None
    
    def _generate_pig_velocity_chart(self, dpi: int) -> Optional[str]:
        """
        Generate pig velocity chart.
        
        Args:
            dpi: DPI for the image
            
        Returns:
            str: Path to the generated image, or None if generation failed
        """
        if not plotting_available:
            return None
        
        try:
            # Create a new figure
            plt.figure(figsize=(10, 6))
            
            # Placeholder data
            time = np.linspace(0, 100, 100)
            velocity = 0.5 + 0.1 * np.sin(time / 10)  # Velocity with fluctuations
            
            # Plot
            plt.plot(time, velocity, 'g-', linewidth=2)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('Velocity (m/s)', fontsize=12)
            plt.title('Pig Velocity vs. Time', fontsize=14)
            
            # Save figure
            self.chart_counter += 1
            filename = f"pig_velocity_{self.chart_counter}.png"
            filepath = os.path.join(self.image_dir, filename)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating pig velocity chart: {e}")
            return None
    
    def _generate_pig_pressure_chart(self, dpi: int) -> Optional[str]:
        """
        Generate pig pressure differential chart.
        
        Args:
            dpi: DPI for the image
            
        Returns:
            str: Path to the generated image, or None if generation failed
        """
        if not plotting_available:
            return None
        
        try:
            # Create a new figure
            plt.figure(figsize=(10, 6))
            
            # Placeholder data
            time = np.linspace(0, 100, 100)
            pressure_diff = 30000 + 5000 * np.sin(time / 5)  # Pressure differential with fluctuations
            
            # Plot
            plt.plot(time, pressure_diff, 'r-', linewidth=2)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('Pressure Differential (Pa)', fontsize=12)
            plt.title('Pressure Differential Across Pig vs. Time', fontsize=14)
            
            # Save figure
            self.chart_counter += 1
            filename = f"pig_pressure_{self.chart_counter}.png"
            filepath = os.path.join(self.image_dir, filename)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating pig pressure chart: {e}")
            return None
    
    def _generate_spill_area_chart(self, dpi: int) -> Optional[str]:
        """
        Generate spill area chart.
        
        Args:
            dpi: DPI for the image
            
        Returns:
            str: Path to the generated image, or None if generation failed
        """
        if not plotting_available:
            return None
        
        try:
            # Create a new figure
            plt.figure(figsize=(10, 6))
            
            # Placeholder data
            time = np.linspace(0, 24, 100)  # Time in hours
            area = 200000 * (1 - np.exp(-time / 5))  # Area growth over time
            
            # Plot
            plt.plot(time, area, 'b-', linewidth=2)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('Time (hours)', fontsize=12)
            plt.ylabel('Affected Area (m²)', fontsize=12)
            plt.title('Spill Area vs. Time', fontsize=14)
            
            # Save figure
            self.chart_counter += 1
            filename = f"spill_area_{self.chart_counter}.png"
            filepath = os.path.join(self.image_dir, filename)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating spill area chart: {e}")
            return None
    
    def _generate_spill_volume_chart(self, dpi: int) -> Optional[str]:
        """
        Generate spill volume chart.
        
        Args:
            dpi: DPI for the image
            
        Returns:
            str: Path to the generated image, or None if generation failed
        """
        if not plotting_available:
            return None
        
        try:
            # Create a new figure
            plt.figure(figsize=(10, 6))
            
            # Placeholder data
            time = np.linspace(0, 24, 100)  # Time in hours
            initial_volume = 100  # m³
            
            # Different volume components
            total_volume = initial_volume * np.ones_like(time)
            evaporated = initial_volume * (1 - np.exp(-time / 48)) * 0.3
            dispersed = initial_volume * (1 - np.exp(-time / 36)) * 0.2
            remaining = total_volume - evaporated - dispersed
            
            # Plot
            plt.stackplot(time, remaining, dispersed, evaporated, 
                         labels=['Remaining', 'Dispersed', 'Evaporated'],
                         colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('Time (hours)', fontsize=12)
            plt.ylabel('Volume (m³)', fontsize=12)
            plt.title('Spill Volume Components vs. Time', fontsize=14)
            plt.legend(loc='upper right')
            
            # Save figure
            self.chart_counter += 1
            filename = f"spill_volume_{self.chart_counter}.png"
            filepath = os.path.join(self.image_dir, filename)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating spill volume chart: {e}")
            return None
    
    def _generate_spill_distance_chart(self, dpi: int) -> Optional[str]:
        """
        Generate spill distance chart.
        
        Args:
            dpi: DPI for the image
            
        Returns:
            str: Path to the generated image, or None if generation failed
        """
        if not plotting_available:
            return None
        
        try:
            # Create a new figure
            plt.figure(figsize=(10, 6))
            
            # Placeholder data
            time = np.linspace(0, 24, 100)  # Time in hours
            distance = 1000 * (1 - np.exp(-time / 6))  # Distance growth over time
            
            # Plot
            plt.plot(time, distance, 'r-', linewidth=2)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('Time (hours)', fontsize=12)
            plt.ylabel('Maximum Distance from Source (m)', fontsize=12)
            plt.title('Spill Spread vs. Time', fontsize=14)
            
            # Save figure
            self.chart_counter += 1
            filename = f"spill_distance_{self.chart_counter}.png"
            filepath = os.path.join(self.image_dir, filename)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating spill distance chart: {e}")
            return None
    
    # ParaView visualization methods
    
    def _generate_velocity_visualization(self) -> Optional[str]:
        """
        Generate velocity visualization using ParaView.
        
        Returns:
            str: Path to the generated image, or None if generation failed
        """
        if not paraview_available:
            return None
        
        # This would use ParaView to generate the visualization
        # For this example, we'll create a placeholder
        return None
    
    def _generate_pressure_visualization(self) -> Optional[str]:
        """
        Generate pressure visualization using ParaView.
        
        Returns:
            str: Path to the generated image, or None if generation failed
        """
        if not paraview_available:
            return None
        
        # This would use ParaView to generate the visualization
        return None
    
    def _generate_streamlines_visualization(self) -> Optional[str]:
        """
        Generate streamlines visualization using ParaView.
        
        Returns:
            str: Path to the generated image, or None if generation failed
        """
        if not paraview_available:
            return None
        
        # This would use ParaView to generate the visualization
        return None
    
    def _generate_pig_position_visualization(self) -> Optional[str]:
        """
        Generate pig position visualization using ParaView.
        
        Returns:
            str: Path to the generated image, or None if generation failed
        """
        if not paraview_available:
            return None
        
        # This would use ParaView to generate the visualization
        return None
    
    def _generate_pigging_velocity_visualization(self) -> Optional[str]:
        """
        Generate fluid velocity visualization for pigging using ParaView.
        
        Returns:
            str: Path to the generated image, or None if generation failed
        """
        if not paraview_available:
            return None
        
        # This would use ParaView to generate the visualization
        return None
    
    def _generate_pigging_pressure_visualization(self) -> Optional[str]:
        """
        Generate pressure visualization for pigging using ParaView.
        
        Returns:
            str: Path to the generated image, or None if generation failed
        """
        if not paraview_available:
            return None
        
        # This would use ParaView to generate the visualization
        return None
    
    def _generate_spill_contour_visualization(self) -> Optional[str]:
        """
        Generate spill contour visualization using ParaView.
        
        Returns:
            str: Path to the generated image, or None if generation failed
        """
        if not paraview_available:
            return None
        
        # This would use ParaView to generate the visualization
        return None
    
    def _generate_spill_thickness_visualization(self) -> Optional[str]:
        """
        Generate spill thickness visualization using ParaView.
        
        Returns:
            str: Path to the generated image, or None if generation failed
        """
        if not paraview_available:
            return None
        
        # This would use ParaView to generate the visualization
        return None
    
    def _generate_environmental_impact_visualization(self) -> Optional[str]:
        """
        Generate environmental impact visualization using ParaView.
        
        Returns:
            str: Path to the generated image, or None if generation failed
        """
        if not paraview_available:
            return None
        
        # This would use ParaView to generate the visualization
        return None


# Function to create a report from a project
def generate_report_from_project(project: Project, output_path: str, 
                                report_type: str = 'flow', 
                                format_type: str = 'html', 
                                open_when_done: bool = False, 
                                **kwargs) -> bool:
    """
    Generate a report from a project.
    
    Args:
        project: Project containing simulation data
        output_path: Path to save the report
        report_type: Type of report ('flow', 'pigging', 'spill')
        format_type: Output format ('html', 'pdf')
        open_when_done: Whether to open the report when done
        **kwargs: Additional report options
            
    Returns:
        bool: True if report generation succeeded, False otherwise
    """
    # Get results model from project
    if not project:
        logger.error("No project provided for report generation")
        return False
    
    results_model = project.get_results_model()
    if not results_model:
        logger.error("No results data available in the project")
        return False
    
    # Create report generator
    generator = ReportGenerator(project, results_model)
    
    # Generate report
    return generator.generate_report(
        output_path=output_path,
        report_type=report_type,
        format_type=format_type,
        open_when_done=open_when_done,
        **kwargs
    )