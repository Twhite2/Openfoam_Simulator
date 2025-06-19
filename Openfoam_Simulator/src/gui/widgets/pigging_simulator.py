#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pigging simulator widget for Openfoam_Simulator application.

This widget provides interactive visualization and simulation of pipeline pigging
operations, which are crucial for pipeline maintenance in the oil & gas industry.
Features include:
- Interactive pig movement visualization
- Real-time calculation of forces on the pig
- Analysis of debris removal efficiency
- Visualization of multiphase flow around the pig
- Prediction of pig velocity and differential pressure
"""

import os
import sys
import math
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QSlider, QGroupBox, QTabWidget, QSplitter,
    QFrame, QProgressBar, QToolButton, QFileDialog, QMessageBox,
    QSizePolicy
)
from PyQt5.QtCore import (
    Qt, QSize, QTimer, QThread, pyqtSignal, QPropertyAnimation,
    QEasingCurve, QRectF, QPoint, QPointF, QSettings
)
from PyQt5.QtGui import (
    QIcon, QPixmap, QPainter, QPen, QBrush, QColor, QFont,
    QRadialGradient, QLinearGradient, QPainterPath, QPolygonF
)

# Import utility modules
from ...utils.logger import get_logger
from ...config import get_value, set_value

logger = get_logger(__name__)


class PipelineView(QWidget):
    """
    Visual representation of a pipeline with a pig for simulation.
    
    This widget renders a 2D cross-section and longitudinal view of a pipeline
    with a pig, showing fluid flow, debris, and other relevant features.
    """
    
    # Signal emitted when the user interacts with the visualization
    interaction = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """
        Initialize the pipeline view.
        
        Args:
            parent: Parent widget
        """
        super(PipelineView, self).__init__(parent)
        
        # Pipeline properties
        self.pipeline_length = 1000.0  # Pipeline length in meters
        self.pipeline_diameter = 0.3  # Pipeline diameter in meters
        self.pig_position = 0.0  # Current pig position (0.0 to 1.0)
        self.pig_length = 0.3  # Pig length in meters
        self.pig_diameter = 0.29  # Pig diameter in meters
        
        # View settings
        self.view_width_meters = 5.0  # Width of visible section in meters
        self.zoom_level = 1.0  # Zoom level
        self.view_center = 0.0  # Center position of the view (0.0 to 1.0)
        
        # Animation settings
        self.animation_speed = 1.0  # Speed multiplier
        self.is_animating = False
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._update_animation)
        
        # Simulation data
        self.flow_velocity = 1.0  # Fluid velocity in m/s
        self.pressure_upstream = 500000.0  # Upstream pressure in Pa
        self.pressure_downstream = 100000.0  # Downstream pressure in Pa
        self.debris_density = 0.2  # Debris density (0.0 to 1.0)
        self.pig_bypass_flow = 0.05  # Bypass flow fraction
        
        # Drawing settings
        self.colors = {
            "pipeline": QColor(100, 100, 100),
            "background": QColor(240, 240, 240),
            "pig": QColor(255, 165, 0),
            "liquid": QColor(65, 105, 225, 150),
            "gas": QColor(220, 220, 220, 100),
            "oil": QColor(139, 69, 19, 180),
            "debris": QColor(101, 67, 33, 200),
            "pressure_high": QColor(255, 0, 0, 150),
            "pressure_low": QColor(0, 0, 255, 150),
            "text": QColor(0, 0, 0),
            "grid": QColor(200, 200, 200)
        }
        
        # Set up widget properties
        self.setMinimumHeight(200)
        self.setMinimumWidth(400)
        self.setSizePolicy(
            QSizePolicy.Expanding, 
            QSizePolicy.Expanding
        )
        
        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)
        
        # Initialize animation timer
        self.last_update_time = time.time()
    
    def set_pipeline_properties(self, properties: Dict[str, Any]):
        """
        Set pipeline properties.
        
        Args:
            properties (Dict[str, Any]): Pipeline properties
        """
        if "length" in properties:
            self.pipeline_length = properties["length"]
        if "diameter" in properties:
            self.pipeline_diameter = properties["diameter"]
        if "pig_length" in properties:
            self.pig_length = properties["pig_length"]
        if "pig_diameter" in properties:
            self.pig_diameter = properties["pig_diameter"]
        
        # Update the view
        self.update()
    
    def set_pig_position(self, position: float):
        """
        Set pig position along the pipeline.
        
        Args:
            position (float): Position from 0.0 (start) to 1.0 (end)
        """
        self.pig_position = max(0.0, min(1.0, position))
        
        # Update view center if pig is outside of view
        pig_pos_meters = self.pig_position * self.pipeline_length
        view_start = (self.view_center - self.view_width_meters / 2)
        view_end = (self.view_center + self.view_width_meters / 2)
        
        if pig_pos_meters < view_start or pig_pos_meters > view_end:
            self.view_center = pig_pos_meters
        
        # Update the view
        self.update()
    
    def set_simulation_data(self, data: Dict[str, Any]):
        """
        Set simulation data for visualization.
        
        Args:
            data (Dict[str, Any]): Simulation data
        """
        if "flow_velocity" in data:
            self.flow_velocity = data["flow_velocity"]
        if "pressure_upstream" in data:
            self.pressure_upstream = data["pressure_upstream"]
        if "pressure_downstream" in data:
            self.pressure_downstream = data["pressure_downstream"]
        if "debris_density" in data:
            self.debris_density = data["debris_density"]
        if "pig_bypass_flow" in data:
            self.pig_bypass_flow = data["pig_bypass_flow"]
        
        # Update the view
        self.update()
    
    def start_animation(self, speed: float = 1.0):
        """
        Start pig movement animation.
        
        Args:
            speed (float): Animation speed multiplier
        """
        self.animation_speed = speed
        self.is_animating = True
        self.last_update_time = time.time()
        self.animation_timer.start(16)  # ~60 FPS
    
    def pause_animation(self):
        """Pause pig movement animation."""
        self.is_animating = False
        self.animation_timer.stop()
    
    def stop_animation(self):
        """Stop pig movement animation and reset position."""
        self.is_animating = False
        self.animation_timer.stop()
        self.set_pig_position(0.0)
    
    def _update_animation(self):
        """Update pig position for animation."""
        if not self.is_animating:
            return
        
        # Calculate time delta
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Calculate new position
        delta_position = (self.flow_velocity / self.pipeline_length) * delta_time * self.animation_speed
        new_position = self.pig_position + delta_position
        
        # Check if animation is complete
        if new_position >= 1.0:
            new_position = 1.0
            self.pause_animation()
        
        # Update position
        self.set_pig_position(new_position)
    
    def zoom_in(self):
        """Zoom in view."""
        self.zoom_level = min(5.0, self.zoom_level * 1.2)
        self.view_width_meters = 5.0 / self.zoom_level
        self.update()
    
    def zoom_out(self):
        """Zoom out view."""
        self.zoom_level = max(0.2, self.zoom_level / 1.2)
        self.view_width_meters = 5.0 / self.zoom_level
        self.update()
    
    def reset_view(self):
        """Reset view to default."""
        self.zoom_level = 1.0
        self.view_width_meters = 5.0
        self.view_center = self.pig_position * self.pipeline_length
        self.update()
    
    def paintEvent(self, event):
        """
        Paint the pipeline visualization.
        
        Args:
            event: Paint event
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        
        # Fill background
        painter.fillRect(0, 0, width, height, self.colors["background"])
        
        # Calculate the visible portion of the pipeline
        view_start_meters = max(0, self.view_center - self.view_width_meters / 2)
        view_end_meters = min(self.pipeline_length, self.view_center + self.view_width_meters / 2)
        
        # Calculate scale factors
        meters_to_pixels_x = width / (view_end_meters - view_start_meters)
        
        # Top section (longitudinal view) - 70% of height
        longview_height = int(height * 0.7)
        longview_rect = QRectF(0, 0, width, longview_height)
        
        # Draw longitudinal view
        self._draw_longitudinal_view(painter, longview_rect, view_start_meters, view_end_meters, meters_to_pixels_x)
        
        # Bottom section (cross-section view) - 30% of height
        crossview_height = height - longview_height
        crossview_rect = QRectF(0, longview_height, width, crossview_height)
        
        # Draw cross-section view
        self._draw_cross_section_view(painter, crossview_rect)
        
        # Draw measurement grid and labels
        self._draw_grid_and_labels(painter, QRectF(0, 0, width, height), view_start_meters, view_end_meters)
    
    def _draw_longitudinal_view(self, painter: QPainter, rect: QRectF, 
                                view_start: float, view_end: float, scale_factor: float):
        """
        Draw longitudinal view of pipeline.
        
        Args:
            painter (QPainter): QPainter instance
            rect (QRectF): Drawing rectangle
            view_start (float): Start position in meters
            view_end (float): End position in meters
            scale_factor (float): Scale factor from meters to pixels
        """
        # Draw pipeline outer wall
        pipe_top = rect.top() + rect.height() * 0.2
        pipe_bottom = rect.top() + rect.height() * 0.8
        pipe_height = pipe_bottom - pipe_top
        
        # Draw pipeline section
        pipeline_rect = QRectF(rect.left(), pipe_top, rect.width(), pipe_height)
        pipeline_brush = QBrush(self.colors["pipeline"])
        painter.fillRect(pipeline_rect, pipeline_brush)
        
        # Draw pressure gradient
        pressure_gradient = QLinearGradient(rect.left(), 0, rect.right(), 0)
        pressure_gradient.setColorAt(0, self.colors["pressure_high"])
        pressure_gradient.setColorAt(1, self.colors["pressure_low"])
        pipeline_inner_rect = QRectF(
            pipeline_rect.left() + 2, 
            pipeline_rect.top() + 2, 
            pipeline_rect.width() - 4, 
            pipeline_rect.height() - 4
        )
        painter.fillRect(pipeline_inner_rect, pressure_gradient)
        
        # Draw debris (increasing towards the pig)
        if self.pig_position > 0.01:
            # Calculate pig position in viewport
            pig_pos_meters = self.pig_position * self.pipeline_length
            if view_start <= pig_pos_meters <= view_end:
                pig_x = (pig_pos_meters - view_start) * scale_factor
                
                # Draw debris with increasing density towards the pig
                debris_gradient = QLinearGradient(0, 0, pig_x, 0)
                debris_gradient.setColorAt(0, QColor(0, 0, 0, 0))  # Transparent at start
                debris_gradient.setColorAt(1, self.colors["debris"])  # Debris color at pig
                
                # Only fill the part of pipe before the pig
                debris_rect = QRectF(
                    pipeline_rect.left(), 
                    pipeline_rect.bottom() - pipeline_rect.height() * 0.3, 
                    pig_x, 
                    pipeline_rect.height() * 0.3
                )
                painter.fillRect(debris_rect, debris_gradient)
        
        # Draw pig
        pig_pos_meters = self.pig_position * self.pipeline_length
        if view_start <= pig_pos_meters <= view_end:
            pig_x = (pig_pos_meters - view_start) * scale_factor
            pig_length_pixels = self.pig_length * scale_factor
            
            pig_rect = QRectF(
                pig_x - pig_length_pixels / 2, 
                pipe_top + 2, 
                pig_length_pixels, 
                pipe_height - 4
            )
            
            pig_brush = QBrush(self.colors["pig"])
            painter.setBrush(pig_brush)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(pig_rect, 5, 5)
            
            # Draw pig details
            painter.setPen(QPen(Qt.black, 1))
            # Draw sealing disc lines
            disc_count = 3
            for i in range(disc_count):
                disc_x = pig_rect.left() + (i + 1) * pig_rect.width() / (disc_count + 1)
                painter.drawLine(QPointF(disc_x, pig_rect.top()), 
                                QPointF(disc_x, pig_rect.bottom()))
            
            # Draw bypassing flow
            if self.pig_bypass_flow > 0.001:
                painter.setPen(QPen(self.colors["liquid"], 2, Qt.DashLine))
                bypass_y = pig_rect.top() + pig_rect.height() * 0.3
                painter.drawLine(QPointF(pig_rect.left(), bypass_y), 
                                QPointF(pig_rect.right(), bypass_y))
                
                # Draw flow arrows
                arrow_size = 4
                arrow_count = max(1, int(self.pig_bypass_flow * 10))
                for i in range(arrow_count):
                    arrow_x = pig_rect.left() + (i + 1) * pig_rect.width() / (arrow_count + 1)
                    painter.drawLine(QPointF(arrow_x, bypass_y - arrow_size), 
                                    QPointF(arrow_x, bypass_y + arrow_size))
                    painter.drawLine(QPointF(arrow_x, bypass_y + arrow_size), 
                                    QPointF(arrow_x + arrow_size, bypass_y))
                    painter.drawLine(QPointF(arrow_x, bypass_y + arrow_size), 
                                    QPointF(arrow_x - arrow_size, bypass_y))
        
        # Draw flow arrows
        painter.setPen(QPen(self.colors["liquid"], 1))
        arrow_spacing = 50  # pixels
        arrow_size = 8
        arrow_count = int(rect.width() / arrow_spacing)
        arrow_y = pipe_top + pipe_height / 2
        
        for i in range(arrow_count):
            arrow_x = rect.left() + i * arrow_spacing + arrow_spacing / 2
            
            # Skip if arrow would be inside the pig
            if pig_x - pig_length_pixels / 2 <= arrow_x <= pig_x + pig_length_pixels / 2:
                continue
            
            # Draw arrow line
            painter.drawLine(QPointF(arrow_x - arrow_size, arrow_y), 
                            QPointF(arrow_x + arrow_size, arrow_y))
            
            # Draw arrow head
            painter.drawLine(QPointF(arrow_x + arrow_size, arrow_y), 
                            QPointF(arrow_x, arrow_y - arrow_size / 2))
            painter.drawLine(QPointF(arrow_x + arrow_size, arrow_y), 
                            QPointF(arrow_x, arrow_y + arrow_size / 2))
    
    def _draw_cross_section_view(self, painter: QPainter, rect: QRectF):
        """
        Draw cross-section view of pipeline.
        
        Args:
            painter (QPainter): QPainter instance
            rect (QRectF): Drawing rectangle
        """
        # Calculate center and radius
        center_x = rect.left() + rect.width() / 2
        center_y = rect.top() + rect.height() / 2
        outer_radius = min(rect.width(), rect.height()) * 0.45
        inner_radius = outer_radius * 0.95
        
        # Draw pipeline outer circle
        painter.setPen(QPen(self.colors["pipeline"], 2))
        painter.setBrush(QBrush(self.colors["pipeline"]))
        painter.drawEllipse(QPointF(center_x, center_y), outer_radius, outer_radius)
        
        # Draw pipeline inner circle (fluid section)
        painter.setPen(Qt.NoPen)
        
        # Draw mixed fluid
        if self.pig_position > 0.01 and self.pig_position < 0.99:
            # Show a mix of fluids
            if self.pig_position < 0.5:
                # More oil/debris before pig
                bottom_height = inner_radius * 2 * (0.2 + self.debris_density * 0.3)
                painter.setBrush(QBrush(self.colors["debris"]))
                painter.drawChord(
                    QRectF(center_x - inner_radius, center_y - inner_radius, 
                           inner_radius * 2, inner_radius * 2),
                    180 * 16, 180 * 16
                )
                
                # Draw liquid above debris
                painter.setBrush(QBrush(self.colors["liquid"]))
                painter.drawChord(
                    QRectF(center_x - inner_radius, center_y - inner_radius, 
                           inner_radius * 2, inner_radius * 2),
                    0 * 16, 180 * 16
                )
            else:
                # Mostly liquid after pig has passed
                painter.setBrush(QBrush(self.colors["liquid"]))
                painter.drawEllipse(QPointF(center_x, center_y), inner_radius, inner_radius)
                
                # Some remaining debris
                remaining_debris = self.debris_density * 0.1 * (1.0 - self.pig_position)
                if remaining_debris > 0.01:
                    painter.setBrush(QBrush(self.colors["debris"]))
                    debris_height = inner_radius * 2 * remaining_debris
                    painter.drawRect(
                        QRectF(center_x - inner_radius, center_y + inner_radius - debris_height, 
                               inner_radius * 2, debris_height)
                    )
        else:
            # Clean fluid if pig hasn't moved or has completed the pipeline
            painter.setBrush(QBrush(self.colors["liquid"]))
            painter.drawEllipse(QPointF(center_x, center_y), inner_radius, inner_radius)
        
        # If we're looking at a position where the pig is located, draw it
        pig_pos_normalized = self.pig_position
        pig_length_normalized = self.pig_length / self.pipeline_length
        
        view_center_normalized = self.view_center / self.pipeline_length
        view_width_normalized = self.view_width_meters / self.pipeline_length
        
        view_start_normalized = view_center_normalized - view_width_normalized / 2
        view_end_normalized = view_center_normalized + view_width_normalized / 2
        
        if (view_start_normalized <= pig_pos_normalized <= view_end_normalized):
            # Draw the pig cross-section (orange circle)
            pig_radius = inner_radius * (self.pig_diameter / self.pipeline_diameter)
            painter.setBrush(QBrush(self.colors["pig"]))
            painter.setPen(QPen(Qt.black, 1))
            painter.drawEllipse(QPointF(center_x, center_y), pig_radius, pig_radius)
            
            # Draw pig details
            # Concentric circles
            painter.setPen(QPen(Qt.black, 1, Qt.DashLine))
            for i in range(1, 3):
                circle_radius = pig_radius * (1 - i * 0.25)
                painter.drawEllipse(QPointF(center_x, center_y), circle_radius, circle_radius)
    
    def _draw_grid_and_labels(self, painter: QPainter, rect: QRectF, 
                             view_start: float, view_end: float):
        """
        Draw grid lines and labels.
        
        Args:
            painter (QPainter): QPainter instance
            rect (QRectF): Drawing rectangle
            view_start (float): Start position in meters
            view_end (float): End position in meters
        """
        # Set up font and pens
        painter.setFont(QFont("Arial", 8))
        painter.setPen(QPen(self.colors["grid"], 1, Qt.DotLine))
        
        # Draw horizontal grid lines in longitudinal view
        longview_height = int(rect.height() * 0.7)
        pipe_top = rect.top() + longview_height * 0.2
        pipe_bottom = rect.top() + longview_height * 0.8
        
        painter.drawLine(QPointF(rect.left(), pipe_top), QPointF(rect.right(), pipe_top))
        painter.drawLine(QPointF(rect.left(), pipe_bottom), QPointF(rect.right(), pipe_bottom))
        
        # Draw vertical grid lines (distance markers)
        visible_length = view_end - view_start
        marker_interval = self._get_appropriate_interval(visible_length)
        first_marker = math.ceil(view_start / marker_interval) * marker_interval
        
        painter.setPen(QPen(self.colors["text"], 1))
        for distance in np.arange(first_marker, view_end, marker_interval):
            x_pos = rect.left() + (distance - view_start) / visible_length * rect.width()
            
            # Draw line
            painter.setPen(QPen(self.colors["grid"], 1, Qt.DotLine))
            painter.drawLine(QPointF(x_pos, rect.top()), QPointF(x_pos, longview_height))
            
            # Draw label
            painter.setPen(QPen(self.colors["text"], 1))
            label_text = f"{distance:.1f}m"
            painter.drawText(QPointF(x_pos - 15, longview_height - 5), label_text)
        
        # Draw position marker for pig
        pig_pos_meters = self.pig_position * self.pipeline_length
        if view_start <= pig_pos_meters <= view_end:
            x_pos = rect.left() + (pig_pos_meters - view_start) / visible_length * rect.width()
            painter.setPen(QPen(self.colors["pig"], 2))
            painter.drawLine(QPointF(x_pos, rect.top()), QPointF(x_pos, longview_height))
            
            # Draw label
            painter.drawText(QPointF(x_pos - 20, rect.top() + 15), f"{pig_pos_meters:.1f}m")
        
        # Draw status and metrics
        painter.setPen(QPen(self.colors["text"], 1))
        status_text = f"Pig Position: {self.pig_position * 100:.1f}% | " \
                     f"Flow: {self.flow_velocity:.2f} m/s | " \
                     f"ΔP: {(self.pressure_upstream - self.pressure_downstream) / 1000:.1f} kPa"
        painter.drawText(QPointF(rect.left() + 10, rect.top() + 15), status_text)
    
    def _get_appropriate_interval(self, visible_length: float) -> float:
        """
        Get appropriate interval for distance markers.
        
        Args:
            visible_length (float): Visible length in meters
            
        Returns:
            float: Appropriate interval in meters
        """
        if visible_length < 1:
            return 0.1
        elif visible_length < 5:
            return 0.5
        elif visible_length < 10:
            return 1.0
        elif visible_length < 50:
            return 5.0
        elif visible_length < 100:
            return 10.0
        elif visible_length < 500:
            return 50.0
        else:
            return 100.0
    
    def mousePressEvent(self, event):
        """
        Handle mouse press event.
        
        Args:
            event: Mouse event
        """
        # Calculate pipeline area
        longview_height = int(self.height() * 0.7)
        pipe_top = longview_height * 0.2
        pipe_bottom = longview_height * 0.8
        
        # Check if click is in pipeline area
        if pipe_top <= event.y() <= pipe_bottom:
            # Calculate position in pipeline
            view_start_meters = max(0, self.view_center - self.view_width_meters / 2)
            view_end_meters = min(self.pipeline_length, self.view_center + self.view_width_meters / 2)
            visible_length = view_end_meters - view_start_meters
            
            click_pos_normalized = event.x() / self.width()
            click_pos_meters = view_start_meters + click_pos_normalized * visible_length
            click_pos = click_pos_meters / self.pipeline_length
            
            # Update pig position
            self.set_pig_position(click_pos)
            
            # Emit interaction signal
            self.interaction.emit({"action": "set_position", "position": click_pos})
    
    def wheelEvent(self, event):
        """
        Handle mouse wheel event for zooming.
        
        Args:
            event: Wheel event
        """
        delta = event.angleDelta().y()
        
        if delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()


class PiggingSimulator(QWidget):
    """
    Widget for simulating pipeline pigging operations.
    
    This widget provides a comprehensive interface for configuring and
    running pigging simulations, with visualization and analysis features
    specifically designed for oil & gas pipeline maintenance operations.
    """
    
    # Signal emitted when simulation state changes
    simulation_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """
        Initialize the pigging simulator widget.
        
        Args:
            parent: Parent widget
        """
        super(PiggingSimulator, self).__init__(parent)
        
        # State variables
        self.is_simulating = False
        self.current_simulation_time = 0.0
        self.simulation_time_step = 0.1  # seconds
        self.simulation_speed = 1.0  # real-time by default
        
        # Set up default values
        self.pipeline_properties = {
            "length": 1000.0,  # meters
            "diameter": 0.3,    # meters
            "roughness": 0.00005,  # meters
            "inclination": 0.0   # degrees (0 = horizontal)
        }
        
        self.pig_properties = {
            "type": "Cup",
            "diameter": 0.29,   # meters
            "length": 0.3,      # meters
            "mass": 10.0,       # kg
            "friction": 0.3,    # friction coefficient
            "bypass": 0.05      # bypass fraction
        }
        
        self.fluid_properties = {
            "type": "Oil",
            "density": 850.0,   # kg/m³
            "viscosity": 0.03,  # Pa·s
            "flow_rate": 0.2,   # m³/s
            "debris_density": 0.2  # fraction (0.0 to 1.0)
        }
        
        # Simulation results
        self.simulation_results = {
            "pig_velocity": 0.0,       # m/s
            "pressure_drop": 0.0,      # Pa
            "cleaning_efficiency": 0.0,  # 0.0 to 1.0
            "pig_differential_pressure": 0.0,  # Pa
            "completion_time": 0.0     # seconds
        }
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
        
        # Initialize visualization with default values
        self._update_visualization()
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(10)
        
        # Title and description
        title_label = QLabel("Pipeline Pigging Simulator")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.layout.addWidget(title_label)
        
        desc_label = QLabel(
            "Simulate pipeline pigging operations for cleaning, "
            "inspection, and maintenance in oil and gas pipelines."
        )
        desc_label.setWordWrap(True)
        self.layout.addWidget(desc_label)
        
        # Create tabs
        self.tab_widget = QTabWidget()
        
        # Visualization tab
        visualization_tab = QWidget()
        viz_layout = QVBoxLayout(visualization_tab)
        
        # Pipeline visualization
        self.pipeline_view = PipelineView()
        viz_layout.addWidget(self.pipeline_view)
        
        # Visualization controls
        viz_controls = QHBoxLayout()
        
        # Zoom controls
        zoom_group = QWidget()
        zoom_layout = QHBoxLayout(zoom_group)
        zoom_layout.setContentsMargins(0, 0, 0, 0)
        
        zoom_in_button = QPushButton("+")
        zoom_in_button.setFixedSize(30, 30)
        zoom_in_button.clicked.connect(self.pipeline_view.zoom_in)
        zoom_layout.addWidget(zoom_in_button)
        
        zoom_out_button = QPushButton("-")
        zoom_out_button.setFixedSize(30, 30)
        zoom_out_button.clicked.connect(self.pipeline_view.zoom_out)
        zoom_layout.addWidget(zoom_out_button)
        
        reset_view_button = QPushButton("Reset View")
        reset_view_button.clicked.connect(self.pipeline_view.reset_view)
        zoom_layout.addWidget(reset_view_button)
        
        viz_controls.addWidget(zoom_group)
        
        # Animation controls
        anim_group = QWidget()
        anim_layout = QHBoxLayout(anim_group)
        anim_layout.setContentsMargins(0, 0, 0, 0)
        
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._start_simulation)
        anim_layout.addWidget(self.play_button)
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self._pause_simulation)
        self.pause_button.setEnabled(False)
        anim_layout.addWidget(self.pause_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_simulation)
        self.stop_button.setEnabled(False)
        anim_layout.addWidget(self.stop_button)
        
        # Simulation speed
        anim_layout.addWidget(QLabel("Speed:"))
        
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "5x", "10x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        anim_layout.addWidget(self.speed_combo)
        
        viz_controls.addWidget(anim_group)
        viz_layout.addLayout(viz_controls)
        
        # Simulation progress
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Simulation Progress:"))
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.time_label = QLabel("Time: 0.0s")
        progress_layout.addWidget(self.time_label)
        
        viz_layout.addLayout(progress_layout)
        
        # Results display
        results_group = QGroupBox("Simulation Results")
        results_layout = QFormLayout(results_group)
        
        self.pig_velocity_label = QLabel("0.0 m/s")
        results_layout.addRow("Pig Velocity:", self.pig_velocity_label)
        
        self.pressure_drop_label = QLabel("0.0 kPa")
        results_layout.addRow("Pressure Drop:", self.pressure_drop_label)
        
        self.cleaning_efficiency_label = QLabel("0.0%")
        results_layout.addRow("Cleaning Efficiency:", self.cleaning_efficiency_label)
        
        self.completion_time_label = QLabel("N/A")
        results_layout.addRow("Estimated Completion Time:", self.completion_time_label)
        
        viz_layout.addWidget(results_group)
        
        self.tab_widget.addTab(visualization_tab, "Visualization")
        
        # Configuration tab
        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        
        # Create a splitter for configuration panels
        config_splitter = QSplitter(Qt.Horizontal)
        
        # Pipeline configuration
        pipeline_group = QGroupBox("Pipeline Configuration")
        pipeline_layout = QFormLayout(pipeline_group)
        
        self.pipeline_length_spin = QDoubleSpinBox()
        self.pipeline_length_spin.setRange(10, 10000)
        self.pipeline_length_spin.setDecimals(1)
        self.pipeline_length_spin.setSingleStep(100)
        self.pipeline_length_spin.setValue(self.pipeline_properties["length"])
        self.pipeline_length_spin.setSuffix(" m")
        pipeline_layout.addRow("Length:", self.pipeline_length_spin)
        
        self.pipeline_diameter_spin = QDoubleSpinBox()
        self.pipeline_diameter_spin.setRange(0.05, 2.0)
        self.pipeline_diameter_spin.setDecimals(3)
        self.pipeline_diameter_spin.setSingleStep(0.01)
        self.pipeline_diameter_spin.setValue(self.pipeline_properties["diameter"])
        self.pipeline_diameter_spin.setSuffix(" m")
        pipeline_layout.addRow("Diameter:", self.pipeline_diameter_spin)
        
        self.pipeline_roughness_spin = QDoubleSpinBox()
        self.pipeline_roughness_spin.setRange(0.00001, 0.01)
        self.pipeline_roughness_spin.setDecimals(5)
        self.pipeline_roughness_spin.setSingleStep(0.00001)
        self.pipeline_roughness_spin.setValue(self.pipeline_properties["roughness"])
        self.pipeline_roughness_spin.setSuffix(" m")
        pipeline_layout.addRow("Roughness:", self.pipeline_roughness_spin)
        
        self.pipeline_inclination_spin = QDoubleSpinBox()
        self.pipeline_inclination_spin.setRange(-45, 45)
        self.pipeline_inclination_spin.setDecimals(1)
        self.pipeline_inclination_spin.setSingleStep(1)
        self.pipeline_inclination_spin.setValue(self.pipeline_properties["inclination"])
        self.pipeline_inclination_spin.setSuffix("°")
        pipeline_layout.addRow("Inclination:", self.pipeline_inclination_spin)
        
        config_splitter.addWidget(pipeline_group)
        
        # Pig configuration
        pig_group = QGroupBox("Pig Configuration")
        pig_layout = QFormLayout(pig_group)
        
        self.pig_type_combo = QComboBox()
        self.pig_type_combo.addItems(["Foam", "Disc", "Cup", "Sphere", "Intelligent", "Gel"])
        self.pig_type_combo.setCurrentText(self.pig_properties["type"])
        pig_layout.addRow("Type:", self.pig_type_combo)
        
        self.pig_diameter_spin = QDoubleSpinBox()
        self.pig_diameter_spin.setRange(0.01, 2.0)
        self.pig_diameter_spin.setDecimals(3)
        self.pig_diameter_spin.setSingleStep(0.001)
        self.pig_diameter_spin.setValue(self.pig_properties["diameter"])
        self.pig_diameter_spin.setSuffix(" m")
        pig_layout.addRow("Diameter:", self.pig_diameter_spin)
        
        self.pig_length_spin = QDoubleSpinBox()
        self.pig_length_spin.setRange(0.01, 5.0)
        self.pig_length_spin.setDecimals(3)
        self.pig_length_spin.setSingleStep(0.01)
        self.pig_length_spin.setValue(self.pig_properties["length"])
        self.pig_length_spin.setSuffix(" m")
        pig_layout.addRow("Length:", self.pig_length_spin)
        
        self.pig_mass_spin = QDoubleSpinBox()
        self.pig_mass_spin.setRange(0.1, 500.0)
        self.pig_mass_spin.setDecimals(1)
        self.pig_mass_spin.setSingleStep(1.0)
        self.pig_mass_spin.setValue(self.pig_properties["mass"])
        self.pig_mass_spin.setSuffix(" kg")
        pig_layout.addRow("Mass:", self.pig_mass_spin)
        
        self.pig_friction_spin = QDoubleSpinBox()
        self.pig_friction_spin.setRange(0.01, 1.0)
        self.pig_friction_spin.setDecimals(2)
        self.pig_friction_spin.setSingleStep(0.01)
        self.pig_friction_spin.setValue(self.pig_properties["friction"])
        pig_layout.addRow("Friction Coefficient:", self.pig_friction_spin)
        
        self.pig_bypass_spin = QDoubleSpinBox()
        self.pig_bypass_spin.setRange(0.0, 0.5)
        self.pig_bypass_spin.setDecimals(2)
        self.pig_bypass_spin.setSingleStep(0.01)
        self.pig_bypass_spin.setValue(self.pig_properties["bypass"])
        self.pig_bypass_spin.setSuffix(" (fraction)")
        pig_layout.addRow("Bypass Flow:", self.pig_bypass_spin)
        
        config_splitter.addWidget(pig_group)
        
        # Fluid configuration
        fluid_group = QGroupBox("Fluid Configuration")
        fluid_layout = QFormLayout(fluid_group)
        
        self.fluid_type_combo = QComboBox()
        self.fluid_type_combo.addItems(["Water", "Oil", "Crude Oil", "Natural Gas"])
        self.fluid_type_combo.setCurrentText(self.fluid_properties["type"])
        fluid_layout.addRow("Type:", self.fluid_type_combo)
        
        self.fluid_density_spin = QDoubleSpinBox()
        self.fluid_density_spin.setRange(0.1, 2000.0)
        self.fluid_density_spin.setDecimals(1)
        self.fluid_density_spin.setSingleStep(10.0)
        self.fluid_density_spin.setValue(self.fluid_properties["density"])
        self.fluid_density_spin.setSuffix(" kg/m³")
        fluid_layout.addRow("Density:", self.fluid_density_spin)
        
        self.fluid_viscosity_spin = QDoubleSpinBox()
        self.fluid_viscosity_spin.setRange(0.0001, 1.0)
        self.fluid_viscosity_spin.setDecimals(4)
        self.fluid_viscosity_spin.setSingleStep(0.001)
        self.fluid_viscosity_spin.setValue(self.fluid_properties["viscosity"])
        self.fluid_viscosity_spin.setSuffix(" Pa·s")
        fluid_layout.addRow("Viscosity:", self.fluid_viscosity_spin)
        
        self.fluid_flow_rate_spin = QDoubleSpinBox()
        self.fluid_flow_rate_spin.setRange(0.01, 10.0)
        self.fluid_flow_rate_spin.setDecimals(2)
        self.fluid_flow_rate_spin.setSingleStep(0.1)
        self.fluid_flow_rate_spin.setValue(self.fluid_properties["flow_rate"])
        self.fluid_flow_rate_spin.setSuffix(" m³/s")
        fluid_layout.addRow("Flow Rate:", self.fluid_flow_rate_spin)
        
        self.debris_density_spin = QDoubleSpinBox()
        self.debris_density_spin.setRange(0.0, 1.0)
        self.debris_density_spin.setDecimals(2)
        self.debris_density_spin.setSingleStep(0.05)
        self.debris_density_spin.setValue(self.fluid_properties["debris_density"])
        self.debris_density_spin.setSuffix(" (fraction)")
        fluid_layout.addRow("Debris Density:", self.debris_density_spin)
        
        config_splitter.addWidget(fluid_group)
        
        # Balance the splitter sections
        config_splitter.setSizes([1, 1, 1])
        
        config_layout.addWidget(config_splitter)
        
        # Preset selection and buttons
        preset_layout = QHBoxLayout()
        
        preset_layout.addWidget(QLabel("Presets:"))
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Custom",
            "Standard Oil Pipeline",
            "Gas Pipeline",
            "Offshore Pipeline", 
            "Small Diameter Pipeline",
            "Large Diameter Pipeline"
        ])
        preset_layout.addWidget(self.preset_combo)
        
        preset_load_button = QPushButton("Load")
        preset_load_button.clicked.connect(self._load_preset)
        preset_layout.addWidget(preset_load_button)
        
        preset_save_button = QPushButton("Save")
        preset_save_button.clicked.connect(self._save_preset)
        preset_layout.addWidget(preset_save_button)
        
        preset_layout.addStretch()
        
        # Apply configuration button
        apply_button = QPushButton("Apply Configuration")
        apply_button.clicked.connect(self._apply_configuration)
        preset_layout.addWidget(apply_button)
        
        config_layout.addLayout(preset_layout)
        
        self.tab_widget.addTab(config_tab, "Configuration")
        
        # Analysis tab
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        
        # Tabs for different analyses
        analysis_tabs = QTabWidget()
        
        # Forces tab
        forces_tab = QWidget()
        forces_layout = QVBoxLayout(forces_tab)
        forces_layout.addWidget(QLabel("Pig Force Analysis"))
        
        # Placeholder for force analysis visualization
        forces_placeholder = QFrame()
        forces_placeholder.setFrameShape(QFrame.StyledPanel)
        forces_placeholder.setMinimumHeight(200)
        forces_layout.addWidget(forces_placeholder)
        
        # Force breakdown
        forces_group = QGroupBox("Force Breakdown")
        forces_group_layout = QFormLayout(forces_group)
        
        self.driving_force_label = QLabel("0.0 N")
        forces_group_layout.addRow("Driving Force:", self.driving_force_label)
        
        self.friction_force_label = QLabel("0.0 N")
        forces_group_layout.addRow("Friction Force:", self.friction_force_label)
        
        self.bypass_force_label = QLabel("0.0 N")
        forces_group_layout.addRow("Bypass Effect:", self.bypass_force_label)
        
        self.gravity_force_label = QLabel("0.0 N")
        forces_group_layout.addRow("Gravity Component:", self.gravity_force_label)
        
        self.net_force_label = QLabel("0.0 N")
        forces_group_layout.addRow("Net Force:", self.net_force_label)
        
        forces_layout.addWidget(forces_group)
        forces_layout.addStretch()
        
        analysis_tabs.addTab(forces_tab, "Force Analysis")
        
        # Efficiency tab
        efficiency_tab = QWidget()
        efficiency_layout = QVBoxLayout(efficiency_tab)
        efficiency_layout.addWidget(QLabel("Cleaning Efficiency Analysis"))
        
        # Placeholder for efficiency analysis visualization
        efficiency_placeholder = QFrame()
        efficiency_placeholder.setFrameShape(QFrame.StyledPanel)
        efficiency_placeholder.setMinimumHeight(200)
        efficiency_layout.addWidget(efficiency_placeholder)
        
        # Efficiency details
        efficiency_group = QGroupBox("Cleaning Efficiency Details")
        efficiency_group_layout = QFormLayout(efficiency_group)
        
        self.debris_removal_label = QLabel("0.0%")
        efficiency_group_layout.addRow("Debris Removal Rate:", self.debris_removal_label)
        
        self.residual_debris_label = QLabel("0.0%")
        efficiency_group_layout.addRow("Residual Debris:", self.residual_debris_label)
        
        self.bypass_efficiency_label = QLabel("0.0%")
        efficiency_group_layout.addRow("Bypass Efficiency Impact:", self.bypass_efficiency_label)
        
        self.overall_efficiency_label = QLabel("0.0%")
        efficiency_group_layout.addRow("Overall Cleaning Efficiency:", self.overall_efficiency_label)
        
        efficiency_layout.addWidget(efficiency_group)
        efficiency_layout.addStretch()
        
        analysis_tabs.addTab(efficiency_tab, "Efficiency Analysis")
        
        # Pressure tab
        pressure_tab = QWidget()
        pressure_layout = QVBoxLayout(pressure_tab)
        pressure_layout.addWidget(QLabel("Pressure Profile Analysis"))
        
        # Placeholder for pressure analysis visualization
        pressure_placeholder = QFrame()
        pressure_placeholder.setFrameShape(QFrame.StyledPanel)
        pressure_placeholder.setMinimumHeight(200)
        pressure_layout.addWidget(pressure_placeholder)
        
        # Pressure details
        pressure_group = QGroupBox("Pressure Details")
        pressure_group_layout = QFormLayout(pressure_group)
        
        self.inlet_pressure_label = QLabel("0.0 kPa")
        pressure_group_layout.addRow("Inlet Pressure:", self.inlet_pressure_label)
        
        self.outlet_pressure_label = QLabel("0.0 kPa")
        pressure_group_layout.addRow("Outlet Pressure:", self.outlet_pressure_label)
        
        self.pig_differential_label = QLabel("0.0 kPa")
        pressure_group_layout.addRow("Differential Across Pig:", self.pig_differential_label)
        
        self.friction_loss_label = QLabel("0.0 kPa")
        pressure_group_layout.addRow("Friction Loss:", self.friction_loss_label)
        
        pressure_layout.addWidget(pressure_group)
        pressure_layout.addStretch()
        
        analysis_tabs.addTab(pressure_tab, "Pressure Analysis")
        
        analysis_layout.addWidget(analysis_tabs)
        
        # Export and reporting section
        export_group = QGroupBox("Export and Reporting")
        export_layout = QHBoxLayout(export_group)
        
        export_layout.addWidget(QLabel("Export Results:"))
        
        export_data_button = QPushButton("Export Data")
        export_data_button.clicked.connect(self._export_data)
        export_layout.addWidget(export_data_button)
        
        export_report_button = QPushButton("Generate Report")
        export_report_button.clicked.connect(self._generate_report)
        export_layout.addWidget(export_report_button)
        
        export_layout.addStretch()
        
        analysis_layout.addWidget(export_group)
        
        self.tab_widget.addTab(analysis_tab, "Analysis")
        
        self.layout.addWidget(self.tab_widget)
    
    def _connect_signals(self):
        """Connect signals to slots."""
        # Connect pipeline view signals
        self.pipeline_view.interaction.connect(self._handle_view_interaction)
        
        # Connect configuration controls
        self.pig_type_combo.currentTextChanged.connect(self._on_pig_type_changed)
        self.fluid_type_combo.currentTextChanged.connect(self._on_fluid_type_changed)
        self.pipeline_diameter_spin.valueChanged.connect(self._on_pipeline_diameter_changed)
        
        # Connect fluid flow rate to update pig velocity calculation
        self.fluid_flow_rate_spin.valueChanged.connect(self._update_pig_velocity)
    
    def _handle_view_interaction(self, interaction_data: Dict[str, Any]):
        """
        Handle interactions with the pipeline view.
        
        Args:
            interaction_data (Dict[str, Any]): Interaction data
        """
        if interaction_data.get("action") == "set_position":
            # User has manually set pig position
            position = interaction_data.get("position", 0.0)
            self._update_simulation_time(position)
        
        # Update UI to reflect new state
        self._update_ui_state()
    
    def _update_simulation_time(self, position: float):
        """
        Update simulation time based on pig position.
        
        Args:
            position (float): Pig position from 0.0 to 1.0
        """
        # Calculate time based on position and pig velocity
        if self.simulation_results["pig_velocity"] > 0:
            self.current_simulation_time = position * self.pipeline_properties["length"] / self.simulation_results["pig_velocity"]
        else:
            self.current_simulation_time = 0.0
        
        # Update time label
        self.time_label.setText(f"Time: {self.current_simulation_time:.1f}s")
        
        # Update progress bar
        self.progress_bar.setValue(int(position * 100))
    
    def _on_pig_type_changed(self, pig_type: str):
        """
        Update pig properties based on selected type.
        
        Args:
            pig_type (str): Selected pig type
        """
        # Set default values based on pig type
        pipe_diameter = self.pipeline_diameter_spin.value()
        
        if pig_type == "Foam":
            self.pig_diameter_spin.setValue(pipe_diameter * 1.05)  # 5% oversized
            self.pig_length_spin.setValue(pipe_diameter * 2.0)
            self.pig_mass_spin.setValue(pipe_diameter * pipe_diameter * 2.0 * 500)  # Approx mass
            self.pig_friction_spin.setValue(0.3)
            self.pig_bypass_spin.setValue(0.08)
        
        elif pig_type == "Disc":
            self.pig_diameter_spin.setValue(pipe_diameter * 1.02)  # 2% oversized
            self.pig_length_spin.setValue(pipe_diameter * 1.5)
            self.pig_mass_spin.setValue(pipe_diameter * pipe_diameter * 1.5 * 800)  # Approx mass
            self.pig_friction_spin.setValue(0.25)
            self.pig_bypass_spin.setValue(0.03)
        
        elif pig_type == "Cup":
            self.pig_diameter_spin.setValue(pipe_diameter * 1.03)  # 3% oversized
            self.pig_length_spin.setValue(pipe_diameter * 2.5)
            self.pig_mass_spin.setValue(pipe_diameter * pipe_diameter * 2.5 * 700)  # Approx mass
            self.pig_friction_spin.setValue(0.35)
            self.pig_bypass_spin.setValue(0.02)
        
        elif pig_type == "Sphere":
            self.pig_diameter_spin.setValue(pipe_diameter * 1.04)  # 4% oversized
            self.pig_length_spin.setValue(pipe_diameter)  # Same as diameter
            self.pig_mass_spin.setValue(pipe_diameter * pipe_diameter * pipe_diameter * 500)  # Approx mass
            self.pig_friction_spin.setValue(0.2)
            self.pig_bypass_spin.setValue(0.05)
        
        elif pig_type == "Intelligent":
            self.pig_diameter_spin.setValue(pipe_diameter * 0.98)  # 2% undersized
            self.pig_length_spin.setValue(pipe_diameter * 4.0)
            self.pig_mass_spin.setValue(pipe_diameter * pipe_diameter * 4.0 * 1200)  # Approx mass
            self.pig_friction_spin.setValue(0.15)
            self.pig_bypass_spin.setValue(0.01)
        
        elif pig_type == "Gel":
            self.pig_diameter_spin.setValue(pipe_diameter)  # Same as pipeline
            self.pig_length_spin.setValue(pipe_diameter * 3.0)
            self.pig_mass_spin.setValue(pipe_diameter * pipe_diameter * 3.0 * 1000)  # Approx mass
            self.pig_friction_spin.setValue(0.4)
            self.pig_bypass_spin.setValue(0.005)
    
    def _on_fluid_type_changed(self, fluid_type: str):
        """
        Update fluid properties based on selected type.
        
        Args:
            fluid_type (str): Selected fluid type
        """
        # Set default values based on fluid type
        if fluid_type == "Water":
            self.fluid_density_spin.setValue(1000.0)
            self.fluid_viscosity_spin.setValue(0.001)
        elif fluid_type == "Oil":
            self.fluid_density_spin.setValue(850.0)
            self.fluid_viscosity_spin.setValue(0.03)
        elif fluid_type == "Crude Oil":
            self.fluid_density_spin.setValue(900.0)
            self.fluid_viscosity_spin.setValue(0.05)
        elif fluid_type == "Natural Gas":
            self.fluid_density_spin.setValue(100.0)  # Compressed gas
            self.fluid_viscosity_spin.setValue(0.00002)
    
    def _on_pipeline_diameter_changed(self, diameter: float):
        """
        Update pig diameter based on pipeline diameter.
        
        Args:
            diameter (float): Pipeline diameter
        """
        # Update pig diameter based on current pig type
        self._on_pig_type_changed(self.pig_type_combo.currentText())
    
    def _on_speed_changed(self, speed_text: str):
        """
        Update simulation speed.
        
        Args:
            speed_text (str): Speed multiplier text (e.g., "2x")
        """
        # Extract numerical value
        speed = float(speed_text.replace("x", ""))
        self.simulation_speed = speed
        
        # Update animation speed if running
        if self.is_simulating:
            self.pipeline_view.animation_speed = speed
    
    def _apply_configuration(self):
        """Apply current configuration to the simulation."""
        # Update pipeline properties
        self.pipeline_properties = {
            "length": self.pipeline_length_spin.value(),
            "diameter": self.pipeline_diameter_spin.value(),
            "roughness": self.pipeline_roughness_spin.value(),
            "inclination": self.pipeline_inclination_spin.value()
        }
        
        # Update pig properties
        self.pig_properties = {
            "type": self.pig_type_combo.currentText(),
            "diameter": self.pig_diameter_spin.value(),
            "length": self.pig_length_spin.value(),
            "mass": self.pig_mass_spin.value(),
            "friction": self.pig_friction_spin.value(),
            "bypass": self.pig_bypass_spin.value()
        }
        
        # Update fluid properties
        self.fluid_properties = {
            "type": self.fluid_type_combo.currentText(),
            "density": self.fluid_density_spin.value(),
            "viscosity": self.fluid_viscosity_spin.value(),
            "flow_rate": self.fluid_flow_rate_spin.value(),
            "debris_density": self.debris_density_spin.value()
        }
        
        # Update visualization
        self._update_visualization()
        
        # Calculate simulation results
        self._calculate_simulation_results()
        
        # Update UI
        self._update_ui_state()
        
        # Show confirmation to user
        QMessageBox.information(
            self,
            "Configuration Applied",
            "Configuration has been applied to the simulation."
        )
    
    def _update_visualization(self):
        """Update the pipeline visualization with current configuration."""
        # Update pipeline view properties
        self.pipeline_view.set_pipeline_properties({
            "length": self.pipeline_properties["length"],
            "diameter": self.pipeline_properties["diameter"],
            "pig_length": self.pig_properties["length"],
            "pig_diameter": self.pig_properties["diameter"]
        })
        
        # Calculate flow velocity from flow rate and diameter
        pipe_area = math.pi * (self.pipeline_properties["diameter"] / 2) ** 2
        flow_velocity = self.fluid_properties["flow_rate"] / pipe_area
        
        # Calculate pressures
        # Simplified calculation - in reality would be more complex
        density = self.fluid_properties["density"]
        length = self.pipeline_properties["length"]
        diameter = self.pipeline_properties["diameter"]
        roughness = self.pipeline_properties["roughness"]
        viscosity = self.fluid_properties["viscosity"]
        
        # Reynolds number
        reynolds = density * flow_velocity * diameter / viscosity
        
        # Friction factor (Colebrook-White approximation)
        if reynolds > 4000:  # Turbulent flow
            friction_factor = 0.25 / (math.log10(roughness / (3.7 * diameter) + 5.74 / (reynolds ** 0.9))) ** 2
        else:  # Laminar flow
            friction_factor = 64 / reynolds
        
        # Pressure drop due to friction
        pressure_drop = friction_factor * (length / diameter) * (density * flow_velocity ** 2) / 2
        
        # Set base pressures (simplified)
        base_pressure = 500000  # 5 bar
        upstream_pressure = base_pressure + pressure_drop
        downstream_pressure = base_pressure
        
        # Update simulation data in the visualization
        self.pipeline_view.set_simulation_data({
            "flow_velocity": flow_velocity,
            "pressure_upstream": upstream_pressure,
            "pressure_downstream": downstream_pressure,
            "debris_density": self.fluid_properties["debris_density"],
            "pig_bypass_flow": self.pig_properties["bypass"]
        })
    
    def _update_pig_velocity(self):
        """Update pig velocity calculation when flow rate changes."""
        # Calculate flow velocity from flow rate and diameter
        pipe_area = math.pi * (self.pipeline_properties["diameter"] / 2) ** 2
        flow_velocity = self.fluid_properties["flow_rate"] / pipe_area
        
        # Simplified pig velocity calculation (assuming bypass flow)
        bypass_fraction = self.pig_properties["bypass"]
        pig_velocity = flow_velocity * (1 - bypass_fraction)
        
        # Update display
        self.pig_velocity_label.setText(f"{pig_velocity:.2f} m/s")
        
        # Update completion time
        if pig_velocity > 0:
            completion_time = self.pipeline_properties["length"] / pig_velocity
            hours = int(completion_time / 3600)
            minutes = int((completion_time % 3600) / 60)
            seconds = int(completion_time % 60)
            self.completion_time_label.setText(f"{hours}h {minutes}m {seconds}s")
        else:
            self.completion_time_label.setText("N/A")
    
    def _calculate_simulation_results(self):
        """Calculate simulation results based on current configuration."""
        # Calculate flow velocity
        pipe_area = math.pi * (self.pipeline_properties["diameter"] / 2) ** 2
        flow_velocity = self.fluid_properties["flow_rate"] / pipe_area
        
        # Simplified pig velocity calculation
        bypass_fraction = self.pig_properties["bypass"]
        pig_velocity = flow_velocity * (1 - bypass_fraction)
        
        # Calculate pressure drop
        # Simplified calculation - in reality would be more complex
        density = self.fluid_properties["density"]
        length = self.pipeline_properties["length"]
        diameter = self.pipeline_properties["diameter"]
        roughness = self.pipeline_properties["roughness"]
        viscosity = self.fluid_properties["viscosity"]
        
        # Reynolds number
        reynolds = density * flow_velocity * diameter / viscosity
        
        # Friction factor (Colebrook-White approximation)
        if reynolds > 4000:  # Turbulent flow
            friction_factor = 0.25 / (math.log10(roughness / (3.7 * diameter) + 5.74 / (reynolds ** 0.9))) ** 2
        else:  # Laminar flow
            friction_factor = 64 / reynolds
        
        # Pressure drop due to friction
        pressure_drop = friction_factor * (length / diameter) * (density * flow_velocity ** 2) / 2
        
        # Pig differential pressure (simplified)
        # Takes into account pressure drop across the pig due to the bypass flow
        pig_differential_pressure = pressure_drop * (1 - bypass_fraction) * 2
        
        # Cleaning efficiency (simplified model)
        # Based on pig type, bypass, and velocity
        base_efficiency = 0.0
        if self.pig_properties["type"] == "Foam":
            base_efficiency = 0.7
        elif self.pig_properties["type"] == "Disc":
            base_efficiency = 0.85
        elif self.pig_properties["type"] == "Cup":
            base_efficiency = 0.9
        elif self.pig_properties["type"] == "Sphere":
            base_efficiency = 0.65
        elif self.pig_properties["type"] == "Intelligent":
            base_efficiency = 0.8
        elif self.pig_properties["type"] == "Gel":
            base_efficiency = 0.95
        
        # Adjust for bypass flow (more bypass = less efficient)
        bypass_factor = 1 - bypass_fraction * 2  # Penalty for bypass
        
        # Adjust for velocity (sweet spot around 1-3 m/s)
        velocity_factor = 1.0
        if pig_velocity < 1.0:
            velocity_factor = 0.8 + 0.2 * pig_velocity  # Penalty for low velocity
        elif pig_velocity > 3.0:
            velocity_factor = 1.0 - 0.05 * (pig_velocity - 3.0)  # Penalty for high velocity
        
        cleaning_efficiency = base_efficiency * bypass_factor * velocity_factor
        cleaning_efficiency = max(0.0, min(1.0, cleaning_efficiency))  # Clamp to [0, 1]
        
        # Calculate completion time
        if pig_velocity > 0:
            completion_time = length / pig_velocity
        else:
            completion_time = float('inf')
        
        # Store results
        self.simulation_results = {
            "pig_velocity": pig_velocity,
            "pressure_drop": pressure_drop,
            "cleaning_efficiency": cleaning_efficiency,
            "pig_differential_pressure": pig_differential_pressure,
            "completion_time": completion_time
        }
        
        # Update UI with results
        self._update_results_display()
        self._update_analysis_display()
    
    def _update_results_display(self):
        """Update the results display with current simulation results."""
        # Update simulation results display
        self.pig_velocity_label.setText(f"{self.simulation_results['pig_velocity']:.2f} m/s")
        self.pressure_drop_label.setText(f"{self.simulation_results['pressure_drop'] / 1000:.1f} kPa")
        self.cleaning_efficiency_label.setText(f"{self.simulation_results['cleaning_efficiency'] * 100:.1f}%")
        
        # Format completion time
        completion_time = self.simulation_results['completion_time']
        if completion_time != float('inf'):
            hours = int(completion_time / 3600)
            minutes = int((completion_time % 3600) / 60)
            seconds = int(completion_time % 60)
            self.completion_time_label.setText(f"{hours}h {minutes}m {seconds}s")
        else:
            self.completion_time_label.setText("N/A")
    
    def _update_analysis_display(self):
        """Update the analysis displays with current simulation results."""
        # Calculate forces on the pig
        # Simplified calculations
        flow_rate = self.fluid_properties["flow_rate"]
        density = self.fluid_properties["density"]
        pig_area = math.pi * (self.pig_properties["diameter"] / 2) ** 2
        pipe_area = math.pi * (self.pipeline_properties["diameter"] / 2) ** 2
        differential_pressure = self.simulation_results["pig_differential_pressure"]
        
        # Driving force from fluid flow
        driving_force = differential_pressure * pig_area
        
        # Friction force
        friction_force = self.pig_properties["friction"] * self.pig_properties["mass"] * 9.81
        
        # Force due to bypass flow
        bypass_force = -differential_pressure * pipe_area * self.pig_properties["bypass"]
        
        # Gravity component (if inclined)
        inclination_rad = math.radians(self.pipeline_properties["inclination"])
        gravity_force = -self.pig_properties["mass"] * 9.81 * math.sin(inclination_rad)
        
        # Net force
        net_force = driving_force + bypass_force - friction_force + gravity_force
        
        # Update force analysis labels
        self.driving_force_label.setText(f"{driving_force:.1f} N")
        self.friction_force_label.setText(f"{friction_force:.1f} N")
        self.bypass_force_label.setText(f"{bypass_force:.1f} N")
        self.gravity_force_label.setText(f"{gravity_force:.1f} N")
        self.net_force_label.setText(f"{net_force:.1f} N")
        
        # Update efficiency analysis
        self.debris_removal_label.setText(f"{self.simulation_results['cleaning_efficiency'] * 100:.1f}%")
        residual_debris = 100 - (self.simulation_results['cleaning_efficiency'] * 100)
        self.residual_debris_label.setText(f"{residual_debris:.1f}%")
        
        # Bypass impact (simplified)
        bypass_impact = self.pig_properties["bypass"] * 100 * 2  # Double the impact for demonstration
        self.bypass_efficiency_label.setText(f"-{bypass_impact:.1f}%")
        self.overall_efficiency_label.setText(f"{self.simulation_results['cleaning_efficiency'] * 100:.1f}%")
        
        # Update pressure analysis
        pipe_pressure_drop = self.simulation_results["pressure_drop"]
        base_pressure = 500000  # 5 bar base pressure
        inlet_pressure = base_pressure + pipe_pressure_drop
        outlet_pressure = base_pressure
        
        self.inlet_pressure_label.setText(f"{inlet_pressure / 1000:.1f} kPa")
        self.outlet_pressure_label.setText(f"{outlet_pressure / 1000:.1f} kPa")
        self.pig_differential_label.setText(f"{self.simulation_results['pig_differential_pressure'] / 1000:.1f} kPa")
        self.friction_loss_label.setText(f"{pipe_pressure_drop / 1000:.1f} kPa")
    
    def _update_ui_state(self):
        """Update UI elements based on current state."""
        # Update button states based on simulation status
        self.play_button.setEnabled(not self.is_simulating)
        self.pause_button.setEnabled(self.is_simulating)
        self.stop_button.setEnabled(self.is_simulating or self.pipeline_view.pig_position > 0.01)
    
    def _start_simulation(self):
        """Start the pigging simulation."""
        # Update state
        self.is_simulating = True
        self._update_ui_state()
        
        # Calculate results first
        self._calculate_simulation_results()
        
        # Start animation with selected speed
        self.pipeline_view.start_animation(self.simulation_speed)
    
    def _pause_simulation(self):
        """Pause the pigging simulation."""
        # Update state
        self.is_simulating = False
        self._update_ui_state()
        
        # Pause animation
        self.pipeline_view.pause_animation()
    
    def _stop_simulation(self):
        """Stop the pigging simulation and reset."""
        # Update state
        self.is_simulating = False
        self._update_ui_state()
        
        # Stop animation and reset position
        self.pipeline_view.stop_animation()
        
        # Reset simulation time and progress
        self.current_simulation_time = 0.0
        self.time_label.setText("Time: 0.0s")
        self.progress_bar.setValue(0)
    
    def _load_preset(self):
        """Load selected preset configuration."""
        preset_name = self.preset_combo.currentText()
        
        if preset_name == "Custom":
            return
        
        # Define presets
        presets = {
            "Standard Oil Pipeline": {
                "pipeline": {
                    "length": 1000.0,
                    "diameter": 0.3048,  # 12 inches
                    "roughness": 0.00005,
                    "inclination": 0.0
                },
                "pig": {
                    "type": "Cup",
                    "diameter": 0.3048 * 1.03,
                    "length": 0.3048 * 2.5,
                    "mass": 25.0,
                    "friction": 0.35,
                    "bypass": 0.02
                },
                "fluid": {
                    "type": "Oil",
                    "density": 850.0,
                    "viscosity": 0.03,
                    "flow_rate": 0.2,
                    "debris_density": 0.2
                }
            },
            "Gas Pipeline": {
                "pipeline": {
                    "length": 5000.0,
                    "diameter": 0.508,  # 20 inches
                    "roughness": 0.00002,
                    "inclination": 0.0
                },
                "pig": {
                    "type": "Foam",
                    "diameter": 0.508 * 1.05,
                    "length": 0.508 * 2.0,
                    "mass": 20.0,
                    "friction": 0.3,
                    "bypass": 0.08
                },
                "fluid": {
                    "type": "Natural Gas",
                    "density": 100.0,
                    "viscosity": 0.00002,
                    "flow_rate": 1.0,
                    "debris_density": 0.1
                }
            },
            "Offshore Pipeline": {
                "pipeline": {
                    "length": 2000.0,
                    "diameter": 0.4064,  # 16 inches
                    "roughness": 0.00004,
                    "inclination": -2.0  # Slightly downward
                },
                "pig": {
                    "type": "Disc",
                    "diameter": 0.4064 * 1.02,
                    "length": 0.4064 * 1.5,
                    "mass": 40.0,
                    "friction": 0.25,
                    "bypass": 0.03
                },
                "fluid": {
                    "type": "Crude Oil",
                    "density": 900.0,
                    "viscosity": 0.05,
                    "flow_rate": 0.3,
                    "debris_density": 0.3
                }
            },
            "Small Diameter Pipeline": {
                "pipeline": {
                    "length": 500.0,
                    "diameter": 0.1016,  # 4 inches
                    "roughness": 0.00005,
                    "inclination": 0.0
                },
                "pig": {
                    "type": "Sphere",
                    "diameter": 0.1016 * 1.04,
                    "length": 0.1016,
                    "mass": 2.0,
                    "friction": 0.2,
                    "bypass": 0.05
                },
                "fluid": {
                    "type": "Water",
                    "density": 1000.0,
                    "viscosity": 0.001,
                    "flow_rate": 0.01,
                    "debris_density": 0.15
                }
            },
            "Large Diameter Pipeline": {
                "pipeline": {
                    "length": 3000.0,
                    "diameter": 0.9144,  # 36 inches
                    "roughness": 0.00007,
                    "inclination": 1.0  # Slightly upward
                },
                "pig": {
                    "type": "Intelligent",
                    "diameter": 0.9144 * 0.98,
                    "length": 0.9144 * 4.0,
                    "mass": 200.0,
                    "friction": 0.15,
                    "bypass": 0.01
                },
                "fluid": {
                    "type": "Oil",
                    "density": 850.0,
                    "viscosity": 0.025,
                    "flow_rate": 1.5,
                    "debris_density": 0.25
                }
            }
        }
        
        # Apply preset if it exists
        if preset_name in presets:
            preset = presets[preset_name]
            
            # Pipeline properties
            if "pipeline" in preset:
                pipeline = preset["pipeline"]
                self.pipeline_length_spin.setValue(pipeline["length"])
                self.pipeline_diameter_spin.setValue(pipeline["diameter"])
                self.pipeline_roughness_spin.setValue(pipeline["roughness"])
                self.pipeline_inclination_spin.setValue(pipeline["inclination"])
            
            # Pig properties
            if "pig" in preset:
                pig = preset["pig"]
                self.pig_type_combo.setCurrentText(pig["type"])
                self.pig_diameter_spin.setValue(pig["diameter"])
                self.pig_length_spin.setValue(pig["length"])
                self.pig_mass_spin.setValue(pig["mass"])
                self.pig_friction_spin.setValue(pig["friction"])
                self.pig_bypass_spin.setValue(pig["bypass"])
            
            # Fluid properties
            if "fluid" in preset:
                fluid = preset["fluid"]
                self.fluid_type_combo.setCurrentText(fluid["type"])
                self.fluid_density_spin.setValue(fluid["density"])
                self.fluid_viscosity_spin.setValue(fluid["viscosity"])
                self.fluid_flow_rate_spin.setValue(fluid["flow_rate"])
                self.debris_density_spin.setValue(fluid["debris_density"])
            
            # Apply configuration
            self._apply_configuration()
            
            # Show confirmation
            QMessageBox.information(
                self,
                "Preset Loaded",
                f"The '{preset_name}' preset has been loaded successfully."
            )
    
    def _save_preset(self):
        """Save current configuration as a custom preset."""
        # Not implemented - would need to save to settings or file
        QMessageBox.information(
            self,
            "Save Preset",
            "This feature is not yet implemented. Custom presets would be saved to a configuration file."
        )
    
    def _export_data(self):
        """Export simulation data to a file."""
        # Get file path
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Simulation Data",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not filepath:
            return
        
        # Ensure file has .csv extension
        if not filepath.lower().endswith('.csv'):
            filepath += '.csv'
        
        try:
            # Create CSV content
            csv_content = "Parameter,Value,Unit\n"
            
            # Pipeline properties
            csv_content += f"Pipeline Length,{self.pipeline_properties['length']},m\n"
            csv_content += f"Pipeline Diameter,{self.pipeline_properties['diameter']},m\n"
            csv_content += f"Pipeline Roughness,{self.pipeline_properties['roughness']},m\n"
            csv_content += f"Pipeline Inclination,{self.pipeline_properties['inclination']},degrees\n"
            
            # Pig properties
            csv_content += f"Pig Type,{self.pig_properties['type']},\n"
            csv_content += f"Pig Diameter,{self.pig_properties['diameter']},m\n"
            csv_content += f"Pig Length,{self.pig_properties['length']},m\n"
            csv_content += f"Pig Mass,{self.pig_properties['mass']},kg\n"
            csv_content += f"Friction Coefficient,{self.pig_properties['friction']},\n"
            csv_content += f"Bypass Fraction,{self.pig_properties['bypass']},\n"
            
            # Fluid properties
            csv_content += f"Fluid Type,{self.fluid_properties['type']},\n"
            csv_content += f"Fluid Density,{self.fluid_properties['density']},kg/m³\n"
            csv_content += f"Fluid Viscosity,{self.fluid_properties['viscosity']},Pa·s\n"
            csv_content += f"Flow Rate,{self.fluid_properties['flow_rate']},m³/s\n"
            csv_content += f"Debris Density,{self.fluid_properties['debris_density']},\n"
            
            # Simulation results
            csv_content += f"Pig Velocity,{self.simulation_results['pig_velocity']},m/s\n"
            csv_content += f"Pressure Drop,{self.simulation_results['pressure_drop']},Pa\n"
            csv_content += f"Cleaning Efficiency,{self.simulation_results['cleaning_efficiency'] * 100},%\n"
            csv_content += f"Differential Pressure,{self.simulation_results['pig_differential_pressure']},Pa\n"
            csv_content += f"Completion Time,{self.simulation_results['completion_time']},s\n"
            
            # Force analysis
            driving_force = float(self.driving_force_label.text().split()[0])
            friction_force = float(self.friction_force_label.text().split()[0])
            bypass_force = float(self.bypass_force_label.text().split()[0])
            gravity_force = float(self.gravity_force_label.text().split()[0])
            net_force = float(self.net_force_label.text().split()[0])
            
            csv_content += f"Driving Force,{driving_force},N\n"
            csv_content += f"Friction Force,{friction_force},N\n"
            csv_content += f"Bypass Force,{bypass_force},N\n"
            csv_content += f"Gravity Force,{gravity_force},N\n"
            csv_content += f"Net Force,{net_force},N\n"
            
            # Write to file
            with open(filepath, 'w') as f:
                f.write(csv_content)
            
            # Show confirmation
            QMessageBox.information(
                self,
                "Export Successful",
                f"Simulation data has been exported to:\n{filepath}"
            )
            
        except Exception as e:
            # Show error
            QMessageBox.critical(
                self,
                "Export Failed",
                f"An error occurred while exporting data:\n{str(e)}"
            )
    
    def _generate_report(self):
        """Generate a report from simulation results."""
        # Get file path
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Report",
            "",
            "HTML Files (*.html);;All Files (*)"
        )
        
        if not filepath:
            return
        
        # Ensure file has .html extension
        if not filepath.lower().endswith('.html'):
            filepath += '.html'
        
        try:
            # Create HTML content
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Pigging Simulation Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #2c3e50; }
                    h2 { color: #3498db; }
                    table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                    .summary { background-color: #e8f4f8; padding: 15px; border-radius: 5px; }
                    .highlight { color: #e74c3c; font-weight: bold; }
                </style>
            </head>
            <body>
                <h1>Pipeline Pigging Simulation Report</h1>
                <p>Generated on """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
                
                <div class="summary">
                    <h2>Executive Summary</h2>
                    <p>This report presents the results of a pipeline pigging simulation for a """ + str(self.pipeline_properties["length"]) + """ meter long, """ + str(self.pipeline_properties["diameter"]) + """ meter diameter pipeline using a """ + self.pig_properties["type"] + """ pig.</p>
                    <p>The simulation predicts a pig velocity of <span class="highlight">""" + f"{self.simulation_results['pig_velocity']:.2f}" + """ m/s</span> with an estimated cleaning efficiency of <span class="highlight">""" + f"{self.simulation_results['cleaning_efficiency'] * 100:.1f}" + """%</span>.</p>
                    <p>Estimated completion time: <span class="highlight">""" + self.completion_time_label.text() + """</span></p>
                </div>
                
                <h2>Pipeline Configuration</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th><th>Unit</th></tr>
                    <tr><td>Length</td><td>""" + str(self.pipeline_properties["length"]) + """</td><td>m</td></tr>
                    <tr><td>Diameter</td><td>""" + str(self.pipeline_properties["diameter"]) + """</td><td>m</td></tr>
                    <tr><td>Roughness</td><td>""" + str(self.pipeline_properties["roughness"]) + """</td><td>m</td></tr>
                    <tr><td>Inclination</td><td>""" + str(self.pipeline_properties["inclination"]) + """</td><td>degrees</td></tr>
                </table>
                
                <h2>Pig Configuration</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th><th>Unit</th></tr>
                    <tr><td>Type</td><td>""" + self.pig_properties["type"] + """</td><td></td></tr>
                    <tr><td>Diameter</td><td>""" + str(self.pig_properties["diameter"]) + """</td><td>m</td></tr>
                    <tr><td>Length</td><td>""" + str(self.pig_properties["length"]) + """</td><td>m</td></tr>
                    <tr><td>Mass</td><td>""" + str(self.pig_properties["mass"]) + """</td><td>kg</td></tr>
                    <tr><td>Friction Coefficient</td><td>""" + str(self.pig_properties["friction"]) + """</td><td></td></tr>
                    <tr><td>Bypass Fraction</td><td>""" + str(self.pig_properties["bypass"]) + """</td><td></td></tr>
                </table>
                
                <h2>Fluid Configuration</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th><th>Unit</th></tr>
                    <tr><td>Type</td><td>""" + self.fluid_properties["type"] + """</td><td></td></tr>
                    <tr><td>Density</td><td>""" + str(self.fluid_properties["density"]) + """</td><td>kg/m³</td></tr>
                    <tr><td>Viscosity</td><td>""" + str(self.fluid_properties["viscosity"]) + """</td><td>Pa·s</td></tr>
                    <tr><td>Flow Rate</td><td>""" + str(self.fluid_properties["flow_rate"]) + """</td><td>m³/s</td></tr>
                    <tr><td>Debris Density</td><td>""" + str(self.fluid_properties["debris_density"]) + """</td><td></td></tr>
                </table>
                
                <h2>Simulation Results</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th><th>Unit</th></tr>
                    <tr><td>Pig Velocity</td><td>""" + f"{self.simulation_results['pig_velocity']:.2f}" + """</td><td>m/s</td></tr>
                    <tr><td>Pressure Drop</td><td>""" + f"{self.simulation_results['pressure_drop'] / 1000:.1f}" + """</td><td>kPa</td></tr>
                    <tr><td>Cleaning Efficiency</td><td>""" + f"{self.simulation_results['cleaning_efficiency'] * 100:.1f}" + """</td><td>%</td></tr>
                    <tr><td>Differential Pressure</td><td>""" + f"{self.simulation_results['pig_differential_pressure'] / 1000:.1f}" + """</td><td>kPa</td></tr>
                    <tr><td>Completion Time</td><td>""" + self.completion_time_label.text() + """</td><td></td></tr>
                </table>
                
                <h2>Force Analysis</h2>
                <table>
                    <tr><th>Force Component</th><th>Value</th><th>Unit</th></tr>
                    <tr><td>Driving Force</td><td>""" + self.driving_force_label.text() + """</td><td></td></tr>
                    <tr><td>Friction Force</td><td>""" + self.friction_force_label.text() + """</td><td></td></tr>
                    <tr><td>Bypass Effect</td><td>""" + self.bypass_force_label.text() + """</td><td></td></tr>
                    <tr><td>Gravity Component</td><td>""" + self.gravity_force_label.text() + """</td><td></td></tr>
                    <tr><td>Net Force</td><td>""" + self.net_force_label.text() + """</td><td></td></tr>
                </table>
                
                <h2>Recommendations</h2>
                <p>Based on the simulation results, the following recommendations are provided:</p>
                <ul>
            """
            
            # Add recommendations based on cleaning efficiency
            cleaning_efficiency = self.simulation_results['cleaning_efficiency'] * 100
            if cleaning_efficiency < 70:
                html_content += """
                    <li>Consider using a different pig type with better sealing capabilities, such as a Cup or Disc pig.</li>
                    <li>Reduce the bypass flow by using a pig with a slightly larger diameter.</li>
                    <li>Run multiple pigs in sequence to improve overall cleaning efficiency.</li>
                    <li>Decrease flow rate to reduce bypass around the pig.</li>
                """
            elif cleaning_efficiency < 85:
                html_content += """
                    <li>The cleaning efficiency is acceptable but could be improved.</li>
                    <li>Consider minor adjustments to the pig configuration or flow rate.</li>
                    <li>Monitor the pipeline for residual debris after pigging.</li>
                """
            else:
                html_content += """
                    <li>The cleaning efficiency is excellent with the current configuration.</li>
                    <li>Maintain the current pigging schedule and parameters.</li>
                """
            
            # Add recommendations based on pig velocity
            pig_velocity = self.simulation_results['pig_velocity']
            if pig_velocity < 0.5:
                html_content += """
                    <li>The pig velocity is too low, which may lead to pig stalling.</li>
                    <li>Increase the flow rate to achieve a minimum velocity of 0.5 m/s.</li>
                """
            elif pig_velocity > 5.0:
                html_content += """
                    <li>The pig velocity is high, which may reduce cleaning efficiency and increase wear.</li>
                    <li>Consider reducing the flow rate to achieve a velocity between 1-3 m/s.</li>
                """
            
            # Close the HTML content
            html_content += """
                </ul>
                
                <h2>Conclusion</h2>
                <p>The simulated pigging operation is predicted to complete in """ + self.completion_time_label.text() + """ with a cleaning efficiency of """ + f"{cleaning_efficiency:.1f}" + """%.</p>
                <p>Regular pigging operations are recommended to maintain pipeline efficiency and integrity.</p>
                
                <p><em>Note: This report was generated automatically by the Openfoam_Simulator Pigging Simulator module.</em></p>
            </body>
            </html>
            """
            
            # Write to file
            with open(filepath, 'w') as f:
                f.write(html_content)
            
            # Show confirmation
            QMessageBox.information(
                self,
                "Report Generated",
                f"Simulation report has been saved to:\n{filepath}"
            )
            
        except Exception as e:
            # Show error
            QMessageBox.critical(
                self,
                "Report Generation Failed",
                f"An error occurred while generating the report:\n{str(e)}"
            )