#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spill calculator widget for Openfoam_Simulator application.

This widget provides simulation and visualization of oil and gas spills
for environmental risk assessment, including:
- Surface and subsurface spill modeling
- Weathering processes simulation
- Trajectory prediction
- Environmental impact assessment
- Volume estimation and spread rate calculation
- Containment planning assistance
"""

import os
import sys
import math
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QSlider, QGroupBox, QTabWidget, QSplitter,
    QFrame, QProgressBar, QToolButton, QFileDialog, QMessageBox,
    QSizePolicy, QRadioButton, QButtonGroup, QDateTimeEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QScrollArea
)
from PyQt5.QtCore import (
    Qt, QSize, QTimer, QThread, pyqtSignal, QPropertyAnimation,
    QEasingCurve, QRectF, QPoint, QPointF, QSettings, QDateTime
)
from PyQt5.QtGui import (
    QIcon, QPixmap, QPainter, QPen, QBrush, QColor, QFont,
    QRadialGradient, QLinearGradient, QPainterPath, QPolygonF,
    QTransform
)

# Import utility modules
from ...utils.logger import get_logger
from ...config import get_value, set_value

logger = get_logger(__name__)


class SpillMapView(QWidget):
    """
    Visual representation of a spill and its spread over time.
    
    This widget renders a 2D map view of a spill event, showing the spread
    pattern, affected areas, and key environmental features.
    """
    
    # Signal emitted when the user interacts with the visualization
    interaction = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """
        Initialize the spill map view.
        
        Args:
            parent: Parent widget
        """
        super(SpillMapView, self).__init__(parent)
        
        # Spill properties
        self.spill_type = "Surface"  # Surface, Subsurface, or Jet
        self.spill_location = (0.0, 0.0)  # (x, y) coordinates in meters
        self.spill_volume = 100.0  # m³
        self.spill_rate = 10.0  # m³/h
        self.spill_duration = 10.0  # hours
        self.spill_age = 0.0  # hours since spill start
        
        # Fluid properties
        self.fluid_type = "Crude Oil"
        self.fluid_density = 900.0  # kg/m³
        self.fluid_viscosity = 0.05  # Pa·s
        self.pour_point = 263.15  # K (-10°C)
        self.api_gravity = 28.0  # API gravity
        
        # Environmental conditions
        self.wind_speed = 5.0  # m/s
        self.wind_direction = 45.0  # degrees (0 = North, clockwise)
        self.current_speed = 0.5  # m/s
        self.current_direction = 30.0  # degrees (0 = North, clockwise)
        self.temperature = 293.15  # K (20°C)
        self.wave_height = 0.5  # m
        self.shoreline_proximity = 5000.0  # m
        
        # Visualization settings
        self.map_scale = 200.0  # pixels per km
        self.map_center = (0.0, 0.0)  # (x, y) coordinates in meters
        self.show_grid = True
        self.show_wind = True
        self.show_current = True
        self.show_shoreline = True
        self.show_legend = True
        
        # Animation settings
        self.animation_speed = 1.0  # Speed multiplier
        self.is_animating = False
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._update_animation)
        
        # Calculated spill properties
        self.spill_radius = 0.0  # m
        self.spill_area = 0.0  # m²
        self.spill_thickness = 0.0  # mm
        self.evaporated_fraction = 0.0  # 0.0 to 1.0
        self.dispersed_fraction = 0.0  # 0.0 to 1.0
        self.emulsified_fraction = 0.0  # 0.0 to 1.0
        self.remaining_volume = self.spill_volume  # m³
        
        # Additional spill contours for time progression
        self.spill_contours = []  # List of (time, radius, color) tuples
        
        # Drawing settings
        self.colors = {
            "background": QColor(230, 240, 255),  # Light blue background
            "grid": QColor(200, 200, 200),
            "spill": QColor(139, 69, 19, 180),    # Brown with transparency
            "evaporated": QColor(200, 200, 200, 120),  # Gray with transparency
            "dispersed": QColor(65, 105, 225, 120),   # Blue with transparency
            "emulsified": QColor(210, 105, 30, 180),  # Brown-red with transparency
            "shoreline": QColor(194, 178, 128),      # Sand color
            "water": QColor(65, 105, 225, 100),      # Blue with transparency
            "text": QColor(0, 0, 0),
            "arrow": QColor(0, 0, 0)
        }
        
        # Set up widget properties
        self.setMinimumHeight(300)
        self.setMinimumWidth(400)
        self.setSizePolicy(
            QSizePolicy.Expanding, 
            QSizePolicy.Expanding
        )
        
        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)
        
        # Initialize animation timer
        self.last_update_time = time.time()
        
        # Calculate initial spill properties
        self._calculate_spill_properties()
    
    def set_spill_properties(self, properties: Dict[str, Any]):
        """
        Set spill properties.
        
        Args:
            properties (Dict[str, Any]): Spill properties
        """
        if "type" in properties:
            self.spill_type = properties["type"]
        if "location" in properties:
            self.spill_location = properties["location"]
        if "volume" in properties:
            self.spill_volume = properties["volume"]
        if "rate" in properties:
            self.spill_rate = properties["rate"]
        if "duration" in properties:
            self.spill_duration = properties["duration"]
        if "age" in properties:
            self.spill_age = properties["age"]
        
        # Update spill calculations
        self._calculate_spill_properties()
        
        # Update the view
        self.update()
    
    def set_fluid_properties(self, properties: Dict[str, Any]):
        """
        Set fluid properties.
        
        Args:
            properties (Dict[str, Any]): Fluid properties
        """
        if "type" in properties:
            self.fluid_type = properties["type"]
        if "density" in properties:
            self.fluid_density = properties["density"]
        if "viscosity" in properties:
            self.fluid_viscosity = properties["viscosity"]
        if "pour_point" in properties:
            self.pour_point = properties["pour_point"]
        if "api_gravity" in properties:
            self.api_gravity = properties["api_gravity"]
        
        # Update spill calculations
        self._calculate_spill_properties()
        
        # Update the view
        self.update()
    
    def set_environmental_conditions(self, conditions: Dict[str, Any]):
        """
        Set environmental conditions.
        
        Args:
            conditions (Dict[str, Any]): Environmental conditions
        """
        if "wind_speed" in conditions:
            self.wind_speed = conditions["wind_speed"]
        if "wind_direction" in conditions:
            self.wind_direction = conditions["wind_direction"]
        if "current_speed" in conditions:
            self.current_speed = conditions["current_speed"]
        if "current_direction" in conditions:
            self.current_direction = conditions["current_direction"]
        if "temperature" in conditions:
            self.temperature = conditions["temperature"]
        if "wave_height" in conditions:
            self.wave_height = conditions["wave_height"]
        if "shoreline_proximity" in conditions:
            self.shoreline_proximity = conditions["shoreline_proximity"]
        
        # Update spill calculations
        self._calculate_spill_properties()
        
        # Update the view
        self.update()
    
    def _calculate_spill_properties(self):
        """Calculate spill properties based on current settings."""
        # Calculate the volume of spill at current age
        if self.spill_age <= self.spill_duration:
            current_volume = self.spill_rate * self.spill_age
        else:
            current_volume = self.spill_rate * self.spill_duration
        
        current_volume = min(current_volume, self.spill_volume)
        
        # Calculate weathering (simplified models)
        # Evaporation model - depends on temperature, wind speed, and API gravity
        # Higher API gravity = more volatile = more evaporation
        evap_factor = 0.01 * (self.temperature - 273.15) * self.wind_speed * self.api_gravity / 30.0
        self.evaporated_fraction = min(0.8, evap_factor * self.spill_age / 24.0)  # Max 80% evaporation
        
        # Dispersion model - depends on wave height, temperature, and viscosity
        disp_factor = 0.005 * self.wave_height * (self.temperature - 273.15) / self.fluid_viscosity
        self.dispersed_fraction = min(0.4, disp_factor * self.spill_age / 24.0)  # Max 40% dispersion
        
        # Emulsification model - depends on wave height and temperature
        emul_factor = 0.01 * self.wave_height * (293.15 / self.temperature)
        self.emulsified_fraction = min(0.8, emul_factor * self.spill_age / 12.0)  # Max 80% emulsification
        
        # Calculate remaining volume (accounting for all processes)
        total_loss_fraction = self.evaporated_fraction + (1 - self.evaporated_fraction) * self.dispersed_fraction
        self.remaining_volume = current_volume * (1 - total_loss_fraction)
        
        # Emulsified volume is larger due to water content
        emulsified_volume = self.remaining_volume * self.emulsified_fraction * 3.0  # Water uptake increases volume
        effective_volume = self.remaining_volume * (1 - self.emulsified_fraction) + emulsified_volume
        
        # Calculate surface area and thickness (for surface spills)
        if self.spill_type == "Surface":
            # Calculate spreading radius using Fay's formula (simplified)
            # R = k * (V^(1/3)) * t^(1/4) where k is a constant, V is volume, t is time in hours
            k = 1.5  # Simplified constant combining gravity, density difference, etc.
            
            # Modify spreading based on wind and current
            combined_transport_factor = ((self.wind_speed * 0.03) ** 2 + (self.current_speed) ** 2) ** 0.5
            transport_factor = 1.0 + combined_transport_factor
            
            # Calculate spill radius
            if self.spill_age > 0:
                self.spill_radius = k * (effective_volume ** (1/3)) * (self.spill_age ** (1/4)) * transport_factor
            else:
                self.spill_radius = 10.0  # Initial radius
            
            # Calculate area from radius
            self.spill_area = math.pi * (self.spill_radius ** 2)
            
            # Calculate thickness in mm
            if self.spill_area > 0:
                self.spill_thickness = (effective_volume / self.spill_area) * 1000  # Convert m to mm
            else:
                self.spill_thickness = 0.0
        
        elif self.spill_type == "Subsurface":
            # For subsurface spills, model a rising plume
            # This is a simplified model - real models would be more complex
            self.spill_radius = 0.5 * (effective_volume ** (1/3)) * (self.spill_age ** 0.5)
            self.spill_area = math.pi * (self.spill_radius ** 2)
            self.spill_thickness = 1.0  # Not really applicable for subsurface
        
        else:  # Jet
            # For jet spills, model a narrow, directional spread
            self.spill_radius = 0.3 * (effective_volume ** (1/3)) * (self.spill_age ** 0.7)
            self.spill_area = math.pi * (self.spill_radius ** 2)
            self.spill_thickness = 2.0  # Thicker due to jet dynamics
        
        # Calculate travel distance and direction due to wind and current
        # Wind has ~3% effect on surface oil movement
        wind_effect = self.wind_speed * 0.03 * self.spill_age  # Distance in meters
        wind_x = wind_effect * math.sin(math.radians(self.wind_direction))
        wind_y = wind_effect * math.cos(math.radians(self.wind_direction))
        
        # Current has 100% effect
        current_effect = self.current_speed * self.spill_age  # Distance in meters
        current_x = current_effect * math.sin(math.radians(self.current_direction))
        current_y = current_effect * math.cos(math.radians(self.current_direction))
        
        # Combine effects
        total_x = wind_x + current_x
        total_y = wind_y + current_y
        
        # Update spill center location
        x, y = self.spill_location
        self.map_center = (x + total_x, y + total_y)
        
        # Update spill contours for time progression
        self.spill_contours = []
        for t in [0.2, 0.4, 0.6, 0.8, 1.0]:
            age = self.spill_age * t
            if age > 0:
                radius = k * (effective_volume ** (1/3)) * (age ** (1/4)) * transport_factor
                alpha = int(255 * (0.2 + t * 0.8))  # Increasing opacity with time
                color = QColor(139, 69, 19, alpha)
                offset_x = wind_x * t + current_x * t
                offset_y = wind_y * t + current_y * t
                self.spill_contours.append((age, radius, color, offset_x, offset_y))
    
    def set_spill_age(self, age: float):
        """
        Set spill age in hours.
        
        Args:
            age (float): Spill age in hours
        """
        self.spill_age = max(0.0, age)
        
        # Update calculations
        self._calculate_spill_properties()
        
        # Update the view
        self.update()
    
    def start_animation(self, speed: float = 1.0):
        """
        Start spill spread animation.
        
        Args:
            speed (float): Animation speed multiplier
        """
        self.animation_speed = speed
        self.is_animating = True
        self.last_update_time = time.time()
        self.animation_timer.start(50)  # 20 fps
    
    def pause_animation(self):
        """Pause spill spread animation."""
        self.is_animating = False
        self.animation_timer.stop()
    
    def stop_animation(self):
        """Stop spill spread animation and reset."""
        self.is_animating = False
        self.animation_timer.stop()
        self.set_spill_age(0.0)
    
    def _update_animation(self):
        """Update spill age for animation."""
        if not self.is_animating:
            return
        
        # Calculate time delta
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Convert real seconds to simulation hours
        delta_hours = delta_time * self.animation_speed
        new_age = self.spill_age + delta_hours
        
        # Update age
        self.set_spill_age(new_age)
        
        # Check if animation is complete
        if new_age >= 72.0:  # 3 days limit
            self.pause_animation()
    
    def zoom_in(self):
        """Zoom in view."""
        self.map_scale *= 1.2
        self.update()
    
    def zoom_out(self):
        """Zoom out view."""
        self.map_scale /= 1.2
        self.update()
    
    def reset_view(self):
        """Reset view to default."""
        self.map_scale = 200.0  # pixels per km
        self.map_center = self.spill_location
        self.update()
    
    def paintEvent(self, event):
        """
        Paint the spill visualization.
        
        Args:
            event: Paint event
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        
        # Fill background (water)
        painter.fillRect(0, 0, width, height, self.colors["background"])
        
        # Calculate scale factors and offsets
        pixels_per_meter = self.map_scale / 1000.0
        center_x, center_y = self.map_center
        
        # Transform coordinates to center the map
        painter.translate(width / 2, height / 2)
        painter.scale(pixels_per_meter, pixels_per_meter)
        painter.translate(-center_x, -center_y)
        
        # Draw grid if enabled
        if self.show_grid:
            self._draw_grid(painter, width, height, pixels_per_meter, center_x, center_y)
        
        # Draw shoreline if enabled
        if self.show_shoreline:
            self._draw_shoreline(painter, width, height, pixels_per_meter, center_x, center_y)
        
        # Draw spill contours
        self._draw_spill_contours(painter)
        
        # Draw wind and current indicators if enabled
        if self.show_wind:
            self._draw_wind_indicator(painter, width, height, pixels_per_meter, center_x, center_y)
        
        if self.show_current:
            self._draw_current_indicator(painter, width, height, pixels_per_meter, center_x, center_y)
        
        # Reset transformation
        painter.resetTransform()
        
        # Draw legend if enabled
        if self.show_legend:
            self._draw_legend(painter, width, height)
        
        # Draw information overlay
        self._draw_info_overlay(painter, width, height)
    
    def _draw_grid(self, painter, width, height, pixels_per_meter, center_x, center_y):
        """
        Draw coordinate grid.
        
        Args:
            painter: QPainter instance
            width: Widget width
            height: Widget height
            pixels_per_meter: Scale factor
            center_x: Center X coordinate
            center_y: Center Y coordinate
        """
        # Calculate grid spacing based on zoom level
        if pixels_per_meter >= 0.1:  # Close zoom
            grid_spacing = 100  # 100m
        elif pixels_per_meter >= 0.01:  # Medium zoom
            grid_spacing = 1000  # 1km
        else:  # Far zoom
            grid_spacing = 10000  # 10km
        
        # Calculate visible range
        visible_width = width / pixels_per_meter
        visible_height = height / pixels_per_meter
        
        min_x = center_x - visible_width / 2
        max_x = center_x + visible_width / 2
        min_y = center_y - visible_height / 2
        max_y = center_y + visible_height / 2
        
        # Adjust to grid multiples
        min_x = math.floor(min_x / grid_spacing) * grid_spacing
        max_x = math.ceil(max_x / grid_spacing) * grid_spacing
        min_y = math.floor(min_y / grid_spacing) * grid_spacing
        max_y = math.ceil(max_y / grid_spacing) * grid_spacing
        
        # Draw grid lines
        painter.setPen(QPen(self.colors["grid"], 1 / pixels_per_meter))
        
        # Draw vertical lines
        for x in np.arange(min_x, max_x + grid_spacing, grid_spacing):
            painter.drawLine(QPointF(x, min_y), QPointF(x, max_y))
        
        # Draw horizontal lines
        for y in np.arange(min_y, max_y + grid_spacing, grid_spacing):
            painter.drawLine(QPointF(min_x, y), QPointF(max_x, y))
        
        # Draw coordinate labels
        painter.setPen(QPen(self.colors["text"], 1 / pixels_per_meter))
        font = painter.font()
        font.setPointSizeF(10 / pixels_per_meter)
        painter.setFont(font)
        
        # Vertical lines labels
        for x in np.arange(min_x, max_x + grid_spacing, grid_spacing):
            painter.drawText(QPointF(x + 5 / pixels_per_meter, min_y + 20 / pixels_per_meter), f"{x:.0f}m")
        
        # Horizontal lines labels
        for y in np.arange(min_y, max_y + grid_spacing, grid_spacing):
            painter.drawText(QPointF(min_x + 5 / pixels_per_meter, y + 5 / pixels_per_meter), f"{y:.0f}m")
    
    def _draw_shoreline(self, painter, width, height, pixels_per_meter, center_x, center_y):
        """
        Draw shoreline features.
        
        Args:
            painter: QPainter instance
            width: Widget width
            height: Widget height
            pixels_per_meter: Scale factor
            center_x: Center X coordinate
            center_y: Center Y coordinate
        """
        # Calculate shoreline position (simplified as a straight line)
        # In a real application, this would use GIS data for coastlines
        shoreline_y = center_y + self.shoreline_proximity
        
        # Calculate visible range
        visible_width = width / pixels_per_meter
        min_x = center_x - visible_width / 2
        max_x = center_x + visible_width / 2
        
        # Draw shoreline
        painter.setPen(QPen(self.colors["shoreline"], 3 / pixels_per_meter))
        painter.drawLine(QPointF(min_x, shoreline_y), QPointF(max_x, shoreline_y))
        
        # Fill land area
        visible_height = height / pixels_per_meter
        max_y = center_y + visible_height / 2
        
        land_rect = QRectF(min_x, shoreline_y, max_x - min_x, max_y - shoreline_y)
        painter.fillRect(land_rect, self.colors["shoreline"])
        
        # Draw shoreline label
        painter.setPen(QPen(self.colors["text"], 1 / pixels_per_meter))
        font = painter.font()
        font.setPointSizeF(12 / pixels_per_meter)
        painter.setFont(font)
        painter.drawText(QPointF(center_x, shoreline_y + 30 / pixels_per_meter), "Shoreline")
    
    def _draw_spill_contours(self, painter):
        """
        Draw spill contours showing spread over time.
        
        Args:
            painter: QPainter instance
        """
        # Draw historical contours (less opacity)
        for age, radius, color, offset_x, offset_y in self.spill_contours:
            x, y = self.spill_location
            center = QPointF(x + offset_x, y + offset_y)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawEllipse(center, radius, radius)
        
        # Draw current spill circle
        x, y = self.spill_location
        center = QPointF(x, y)
        
        # Calculate travel distance and direction due to wind and current
        wind_effect = self.wind_speed * 0.03 * self.spill_age  # Distance in meters
        wind_x = wind_effect * math.sin(math.radians(self.wind_direction))
        wind_y = wind_effect * math.cos(math.radians(self.wind_direction))
        
        current_effect = self.current_speed * self.spill_age  # Distance in meters
        current_x = current_effect * math.sin(math.radians(self.current_direction))
        current_y = current_effect * math.cos(math.radians(self.current_direction))
        
        # Combine effects
        total_x = wind_x + current_x
        total_y = wind_y + current_y
        
        # Current spill center
        current_center = QPointF(x + total_x, y + total_y)
        
        # Draw main spill body
        painter.setPen(Qt.NoPen)
        
        # Draw evaporated portion if significant
        if self.evaporated_fraction > 0.05:
            evap_radius = self.spill_radius * 1.2  # Slightly larger to indicate dispersion
            painter.setBrush(QBrush(self.colors["evaporated"]))
            painter.drawEllipse(current_center, evap_radius, evap_radius)
        
        # Draw dispersed portion if significant
        if self.dispersed_fraction > 0.05:
            disp_radius = self.spill_radius * 1.1  # Slightly larger than main body
            painter.setBrush(QBrush(self.colors["dispersed"]))
            painter.drawEllipse(current_center, disp_radius, disp_radius)
        
        # Draw main spill body
        # Adjust color based on emulsification
        if self.emulsified_fraction > 0.5:
            spill_color = self.colors["emulsified"]
        else:
            spill_color = self.colors["spill"]
        
        painter.setBrush(QBrush(spill_color))
        painter.drawEllipse(current_center, self.spill_radius, self.spill_radius)
        
        # Draw spill source marker
        painter.setPen(QPen(Qt.black, 2))
        painter.setBrush(QBrush(Qt.red))
        painter.drawEllipse(center, 5, 5)
        
        # Draw a line from source to current center
        painter.setPen(QPen(Qt.red, 1, Qt.DashLine))
        painter.drawLine(center, current_center)
    
    def _draw_wind_indicator(self, painter, width, height, pixels_per_meter, center_x, center_y):
        """
        Draw wind direction and speed indicator.
        
        Args:
            painter: QPainter instance
            width: Widget width
            height: Widget height
            pixels_per_meter: Scale factor
            center_x: Center X coordinate
            center_y: Center Y coordinate
        """
        # Calculate position for the indicator (top right)
        indicator_x = center_x + (width / pixels_per_meter) * 0.4
        indicator_y = center_y - (height / pixels_per_meter) * 0.4
        
        # Calculate arrow properties
        arrow_length = 50 / pixels_per_meter
        arrow_width = 15 / pixels_per_meter
        
        # Draw wind arrow (arrow points in the direction the wind is blowing towards)
        self._draw_direction_arrow(painter, indicator_x, indicator_y, self.wind_direction, 
                                  arrow_length, arrow_width, QColor(100, 100, 255), "Wind")
    
    def _draw_current_indicator(self, painter, width, height, pixels_per_meter, center_x, center_y):
        """
        Draw current direction and speed indicator.
        
        Args:
            painter: QPainter instance
            width: Widget width
            height: Widget height
            pixels_per_meter: Scale factor
            center_x: Center X coordinate
            center_y: Center Y coordinate
        """
        # Calculate position for the indicator (bottom right)
        indicator_x = center_x + (width / pixels_per_meter) * 0.4
        indicator_y = center_y + (height / pixels_per_meter) * 0.4
        
        # Calculate arrow properties
        arrow_length = 50 / pixels_per_meter
        arrow_width = 15 / pixels_per_meter
        
        # Draw current arrow
        self._draw_direction_arrow(painter, indicator_x, indicator_y, self.current_direction, 
                                  arrow_length, arrow_width, QColor(0, 100, 255), "Current")
    
    def _draw_direction_arrow(self, painter, x, y, direction, length, width, color, label):
        """
        Draw a direction arrow.
        
        Args:
            painter: QPainter instance
            x: X coordinate of arrow base
            y: Y coordinate of arrow base
            direction: Direction in degrees (0 = North, clockwise)
            length: Arrow length
            width: Arrow width
            color: Arrow color
            label: Arrow label
        """
        # Calculate end point
        end_x = x + length * math.sin(math.radians(direction))
        end_y = y + length * math.cos(math.radians(direction))
        
        # Calculate arrow head points
        arrow_angle = math.radians(direction)
        head_length = length * 0.3
        
        head_1_x = end_x - head_length * math.sin(arrow_angle + math.pi / 6)
        head_1_y = end_y - head_length * math.cos(arrow_angle + math.pi / 6)
        
        head_2_x = end_x - head_length * math.sin(arrow_angle - math.pi / 6)
        head_2_y = end_y - head_length * math.cos(arrow_angle - math.pi / 6)
        
        # Draw arrow line
        painter.setPen(QPen(color, width / 3))
        painter.drawLine(QPointF(x, y), QPointF(end_x, end_y))
        
        # Draw arrow head
        arrow_head = QPolygonF()
        arrow_head.append(QPointF(end_x, end_y))
        arrow_head.append(QPointF(head_1_x, head_1_y))
        arrow_head.append(QPointF(head_2_x, head_2_y))
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawPolygon(arrow_head)
        
        # Draw label
        painter.setPen(QPen(self.colors["text"], width / 6))
        font = painter.font()
        font.setPointSizeF(width)
        painter.setFont(font)
        
        # Adjust label position based on direction
        label_x = x - width * 2
        label_y = y - width * 2
        
        # Add speed to label
        speed = self.wind_speed if label == "Wind" else self.current_speed
        full_label = f"{label}: {speed:.1f} m/s"
        
        painter.drawText(QPointF(label_x, label_y), full_label)
    
    def _draw_legend(self, painter, width, height):
        """
        Draw map legend.
        
        Args:
            painter: QPainter instance
            width: Widget width
            height: Widget height
        """
        # Set up legend properties
        legend_width = 180
        legend_height = 150
        legend_x = width - legend_width - 10
        legend_y = 10
        
        # Draw legend background
        legend_rect = QRectF(legend_x, legend_y, legend_width, legend_height)
        painter.fillRect(legend_rect, QColor(255, 255, 255, 200))
        painter.setPen(QPen(Qt.black, 1))
        painter.drawRect(legend_rect)
        
        # Draw legend title
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(QRectF(legend_x + 5, legend_y + 5, legend_width - 10, 20), "Legend")
        
        # Draw legend items
        painter.setFont(QFont("Arial", 9))
        
        items = [
            ("Spill Source", QColor(255, 0, 0)),
            ("Current Spill", self.colors["spill"]),
            ("Evaporated Oil", self.colors["evaporated"]),
            ("Dispersed Oil", self.colors["dispersed"]),
            ("Emulsified Oil", self.colors["emulsified"]),
            ("Shoreline", self.colors["shoreline"])
        ]
        
        for i, (label, color) in enumerate(items):
            y_pos = legend_y + 30 + i * 20
            
            # Draw color box
            color_rect = QRectF(legend_x + 5, y_pos, 15, 15)
            painter.fillRect(color_rect, color)
            painter.setPen(QPen(Qt.black, 1))
            painter.drawRect(color_rect)
            
            # Draw label
            painter.drawText(QRectF(legend_x + 25, y_pos, legend_width - 30, 15), label)
    
    def _draw_info_overlay(self, painter, width, height):
        """
        Draw information overlay with spill details.
        
        Args:
            painter: QPainter instance
            width: Widget width
            height: Widget height
        """
        # Set up info box properties
        info_width = 300
        info_height = 130
        info_x = 10
        info_y = height - info_height - 10
        
        # Draw info background
        info_rect = QRectF(info_x, info_y, info_width, info_height)
        painter.fillRect(info_rect, QColor(255, 255, 255, 200))
        painter.setPen(QPen(Qt.black, 1))
        painter.drawRect(info_rect)
        
        # Draw info title
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(QRectF(info_x + 5, info_y + 5, info_width - 10, 20), "Spill Information")
        
        # Draw info items
        painter.setFont(QFont("Arial", 9))
        
        y_offset = info_y + 30
        line_height = 16
        
        info_items = [
            f"Type: {self.spill_type}",
            f"Age: {self.spill_age:.1f} hours",
            f"Radius: {self.spill_radius:.1f} m",
            f"Area: {self.spill_area / 10000:.2f} hectares",
            f"Thickness: {self.spill_thickness:.2f} mm",
            f"Remaining Volume: {self.remaining_volume:.1f} m³ ({self.remaining_volume / self.spill_volume * 100:.1f}%)"
        ]
        
        for i, text in enumerate(info_items):
            painter.drawText(QRectF(info_x + 5, y_offset + i * line_height, info_width - 10, line_height), text)
    
    def mousePressEvent(self, event):
        """
        Handle mouse press event.
        
        Args:
            event: Mouse event
        """
        # Convert screen coordinates to map coordinates
        width = self.width()
        height = self.height()
        pixels_per_meter = self.map_scale / 1000.0
        center_x, center_y = self.map_center
        
        # Calculate map coordinates
        map_x = center_x + (event.x() - width / 2) / pixels_per_meter
        map_y = center_y + (event.y() - height / 2) / pixels_per_meter
        
        # Emit interaction signal with coordinates
        self.interaction.emit({
            "action": "click",
            "position": (map_x, map_y)
        })
    
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


class WeatheringGraph(QWidget):
    """
    Widget for displaying weathering processes over time.
    
    This widget shows graphs of evaporation, dispersion, emulsification,
    and remaining oil volume over time.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the weathering graph.
        
        Args:
            parent: Parent widget
        """
        super(WeatheringGraph, self).__init__(parent)
        
        # Data for plotting
        self.times = []  # Hours
        self.evaporated = []  # Fraction
        self.dispersed = []  # Fraction
        self.emulsified = []  # Fraction
        self.remaining = []  # Fraction
        
        # Set up widget properties
        self.setMinimumHeight(200)
        self.setMinimumWidth(400)
        self.setSizePolicy(
            QSizePolicy.Expanding, 
            QSizePolicy.Expanding
        )
        
        # Colors for different processes
        self.colors = {
            "evaporated": QColor(200, 200, 200),
            "dispersed": QColor(65, 105, 225),
            "emulsified": QColor(210, 105, 30),
            "remaining": QColor(139, 69, 19)
        }
    
    def add_data_point(self, time: float, evap: float, disp: float, emul: float, remain: float):
        """
        Add a data point to the graph.
        
        Args:
            time (float): Time in hours
            evap (float): Evaporated fraction (0-1)
            disp (float): Dispersed fraction (0-1)
            emul (float): Emulsified fraction (0-1)
            remain (float): Remaining fraction (0-1)
        """
        self.times.append(time)
        self.evaporated.append(evap)
        self.dispersed.append(disp)
        self.emulsified.append(emul)
        self.remaining.append(remain)
        
        # Trigger repaint
        self.update()
    
    def clear(self):
        """Clear all data points."""
        self.times = []
        self.evaporated = []
        self.dispersed = []
        self.emulsified = []
        self.remaining = []
        self.update()
    
    def paintEvent(self, event):
        """
        Paint the weathering graph.
        
        Args:
            event: Paint event
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        
        # Fill background
        painter.fillRect(0, 0, width, height, Qt.white)
        
        # Check if we have data to plot
        if not self.times:
            # No data, draw placeholder
            painter.setPen(Qt.gray)
            painter.drawText(QRectF(0, 0, width, height), Qt.AlignCenter, 
                           "No data to display. Start simulation to see weathering processes.")
            return
        
        # Calculate graph area
        margin = 40
        graph_rect = QRectF(margin, margin, width - 2 * margin, height - 2 * margin)
        
        # Draw axes
        painter.setPen(QPen(Qt.black, 1))
        painter.drawLine(graph_rect.bottomLeft(), graph_rect.bottomRight())  # X-axis
        painter.drawLine(graph_rect.bottomLeft(), graph_rect.topLeft())      # Y-axis
        
        # Draw axis labels
        painter.setFont(QFont("Arial", 9))
        
        # X-axis label
        painter.drawText(
            QRectF(graph_rect.center().x() - 50, graph_rect.bottom() + 5, 100, 20),
            Qt.AlignCenter, "Time (hours)"
        )
        
        # Y-axis label
        painter.save()
        painter.translate(graph_rect.left() - 25, graph_rect.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-50, 0, 100, 20), Qt.AlignCenter, "Fraction")
        painter.restore()
        
        # Draw X-axis ticks and labels
        max_time = max(self.times)
        time_step = max(1, round(max_time / 5))  # Calculate appropriate step
        
        for t in range(0, int(max_time) + time_step, time_step):
            x = graph_rect.left() + (t / max_time) * graph_rect.width()
            
            # Draw tick
            painter.drawLine(QPointF(x, graph_rect.bottom()), QPointF(x, graph_rect.bottom() + 5))
            
            # Draw label
            painter.drawText(
                QRectF(x - 15, graph_rect.bottom() + 5, 30, 15),
                Qt.AlignCenter, str(t)
            )
        
        # Draw Y-axis ticks and labels
        for i in range(0, 11, 2):
            y_val = i / 10.0
            y = graph_rect.bottom() - y_val * graph_rect.height()
            
            # Draw tick
            painter.drawLine(QPointF(graph_rect.left() - 5, y), QPointF(graph_rect.left(), y))
            
            # Draw label
            painter.drawText(
                QRectF(graph_rect.left() - 35, y - 7, 30, 15),
                Qt.AlignRight | Qt.AlignVCenter, f"{y_val:.1f}"
            )
        
        # Draw grid lines
        painter.setPen(QPen(QColor(200, 200, 200), 1, Qt.DotLine))
        
        # Horizontal grid lines
        for i in range(0, 11, 2):
            y_val = i / 10.0
            y = graph_rect.bottom() - y_val * graph_rect.height()
            painter.drawLine(QPointF(graph_rect.left(), y), QPointF(graph_rect.right(), y))
        
        # Vertical grid lines
        for t in range(0, int(max_time) + time_step, time_step):
            x = graph_rect.left() + (t / max_time) * graph_rect.width()
            painter.drawLine(QPointF(x, graph_rect.top()), QPointF(x, graph_rect.bottom()))
        
        # Draw data lines
        self._draw_data_line(painter, graph_rect, self.times, self.evaporated, self.colors["evaporated"], "Evaporated")
        self._draw_data_line(painter, graph_rect, self.times, self.dispersed, self.colors["dispersed"], "Dispersed")
        self._draw_data_line(painter, graph_rect, self.times, self.emulsified, self.colors["emulsified"], "Emulsified")
        self._draw_data_line(painter, graph_rect, self.times, self.remaining, self.colors["remaining"], "Remaining")
        
        # Draw legend
        self._draw_legend(painter, width, height)
    
    def _draw_data_line(self, painter, rect, x_data, y_data, color, label):
        """
        Draw a data line on the graph.
        
        Args:
            painter: QPainter instance
            rect: Graph rectangle
            x_data: X-axis data (time)
            y_data: Y-axis data (fraction)
            color: Line color
            label: Line label
        """
        if not x_data or not y_data:
            return
        
        max_time = max(x_data)
        
        # Create a list of points
        points = []
        for i in range(len(x_data)):
            x = rect.left() + (x_data[i] / max_time) * rect.width()
            y = rect.bottom() - y_data[i] * rect.height()
            points.append(QPointF(x, y))
        
        # Draw the line
        painter.setPen(QPen(color, 2))
        for i in range(1, len(points)):
            painter.drawLine(points[i-1], points[i])
    
    def _draw_legend(self, painter, width, height):
        """
        Draw the graph legend.
        
        Args:
            painter: QPainter instance
            width: Widget width
            height: Widget height
        """
        # Set up legend properties
        legend_width = 100
        legend_height = 90
        legend_x = width - legend_width - 10
        legend_y = 10
        
        # Draw legend background
        legend_rect = QRectF(legend_x, legend_y, legend_width, legend_height)
        painter.fillRect(legend_rect, QColor(255, 255, 255, 200))
        painter.setPen(QPen(Qt.black, 1))
        painter.drawRect(legend_rect)
        
        # Draw legend items
        painter.setFont(QFont("Arial", 9))
        
        items = [
            ("Evaporated", self.colors["evaporated"]),
            ("Dispersed", self.colors["dispersed"]),
            ("Emulsified", self.colors["emulsified"]),
            ("Remaining", self.colors["remaining"])
        ]
        
        for i, (label, color) in enumerate(items):
            y_pos = legend_y + 10 + i * 20
            
            # Draw color line
            painter.setPen(QPen(color, 2))
            painter.drawLine(QPointF(legend_x + 5, y_pos + 7), QPointF(legend_x + 25, y_pos + 7))
            
            # Draw label
            painter.setPen(QPen(Qt.black, 1))
            painter.drawText(QRectF(legend_x + 30, y_pos, legend_width - 35, 15), label)


class SpillCalculator(QWidget):
    """
    Widget for simulating and analyzing oil and gas spills.
    
    This widget provides comprehensive tools for spill modeling, trajectory
    prediction, weathering simulation, and environmental impact assessment.
    """
    
    # Signal emitted when simulation state changes
    simulation_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """
        Initialize the spill calculator widget.
        
        Args:
            parent: Parent widget
        """
        super(SpillCalculator, self).__init__(parent)
        
        # State variables
        self.is_simulating = False
        self.current_simulation_time = 0.0  # hours
        self.simulation_time_step = 0.1  # hours
        self.simulation_speed = 1.0  # Real-time multiplier
        
        # Set up default values
        self.spill_properties = {
            "type": "Surface",
            "location": (0.0, 0.0),
            "volume": 100.0,  # m³
            "rate": 10.0,  # m³/h
            "duration": 10.0,  # hours
            "age": 0.0  # hours
        }
        
        self.fluid_properties = {
            "type": "Crude Oil",
            "density": 900.0,  # kg/m³
            "viscosity": 0.05,  # Pa·s
            "pour_point": 263.15,  # K (-10°C)
            "api_gravity": 28.0  # API gravity
        }
        
        self.environmental_conditions = {
            "wind_speed": 5.0,  # m/s
            "wind_direction": 45.0,  # degrees (0 = North, clockwise)
            "current_speed": 0.5,  # m/s
            "current_direction": 30.0,  # degrees (0 = North, clockwise)
            "temperature": 293.15,  # K (20°C)
            "wave_height": 0.5,  # m
            "shoreline_proximity": 5000.0  # m
        }
        
        # Simulation results
        self.simulation_results = {
            "max_radius": 0.0,  # m
            "max_area": 0.0,  # m²
            "shoreline_impact_time": float('inf'),  # hours
            "evaporated_fraction": 0.0,  # 0.0 to 1.0
            "dispersed_fraction": 0.0,  # 0.0 to 1.0
            "emulsified_fraction": 0.0,  # 0.0 to 1.0
            "remaining_fraction": 1.0,  # 0.0 to 1.0
            "remaining_volume": 0.0,  # m³
            "shoreline_length_affected": 0.0  # m
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
        title_label = QLabel("Spill Calculator")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.layout.addWidget(title_label)
        
        desc_label = QLabel(
            "Simulate oil and gas spills to assess environmental impact, "
            "predict spill trajectory, and calculate required response resources."
        )
        desc_label.setWordWrap(True)
        self.layout.addWidget(desc_label)
        
        # Create tabs
        self.tab_widget = QTabWidget()
        
        # Visualization tab
        visualization_tab = QWidget()
        viz_layout = QVBoxLayout(visualization_tab)
        
        # Spill visualization
        self.spill_map_view = SpillMapView()
        viz_layout.addWidget(self.spill_map_view)
        
        # Visualization controls
        viz_controls = QHBoxLayout()
        
        # Zoom controls
        zoom_group = QWidget()
        zoom_layout = QHBoxLayout(zoom_group)
        zoom_layout.setContentsMargins(0, 0, 0, 0)
        
        zoom_in_button = QPushButton("+")
        zoom_in_button.setFixedSize(30, 30)
        zoom_in_button.clicked.connect(self.spill_map_view.zoom_in)
        zoom_layout.addWidget(zoom_in_button)
        
        zoom_out_button = QPushButton("-")
        zoom_out_button.setFixedSize(30, 30)
        zoom_out_button.clicked.connect(self.spill_map_view.zoom_out)
        zoom_layout.addWidget(zoom_out_button)
        
        reset_view_button = QPushButton("Reset View")
        reset_view_button.clicked.connect(self.spill_map_view.reset_view)
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
        
        self.stop_button = QPushButton("Reset")
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
        
        # Display options
        display_group = QWidget()
        display_layout = QHBoxLayout(display_group)
        display_layout.setContentsMargins(0, 0, 0, 0)
        
        display_layout.addWidget(QLabel("Show:"))
        
        self.show_grid_check = QCheckBox("Grid")
        self.show_grid_check.setChecked(True)
        self.show_grid_check.toggled.connect(self._on_show_grid_toggled)
        display_layout.addWidget(self.show_grid_check)
        
        self.show_wind_check = QCheckBox("Wind")
        self.show_wind_check.setChecked(True)
        self.show_wind_check.toggled.connect(self._on_show_wind_toggled)
        display_layout.addWidget(self.show_wind_check)
        
        self.show_current_check = QCheckBox("Current")
        self.show_current_check.setChecked(True)
        self.show_current_check.toggled.connect(self._on_show_current_toggled)
        display_layout.addWidget(self.show_current_check)
        
        self.show_shoreline_check = QCheckBox("Shoreline")
        self.show_shoreline_check.setChecked(True)
        self.show_shoreline_check.toggled.connect(self._on_show_shoreline_toggled)
        display_layout.addWidget(self.show_shoreline_check)
        
        viz_controls.addWidget(display_group)
        
        viz_layout.addLayout(viz_controls)
        
        # Simulation progress
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Simulation Time:"))
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(72)  # 72 hours (3 days)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.time_label = QLabel("0.0 hours")
        progress_layout.addWidget(self.time_label)
        
        viz_layout.addLayout(progress_layout)
        
        # Weathering graph
        self.weathering_graph = WeatheringGraph()
        viz_layout.addWidget(self.weathering_graph)
        
        self.tab_widget.addTab(visualization_tab, "Visualization")
        
        # Configuration tab
        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        
        # Create a splitter for configuration panels
        config_splitter = QSplitter(Qt.Horizontal)
        
        # Spill configuration
        spill_group = QGroupBox("Spill Configuration")
        spill_layout = QFormLayout(spill_group)
        
        self.spill_type_combo = QComboBox()
        self.spill_type_combo.addItems(["Surface", "Subsurface", "Jet"])
        self.spill_type_combo.setCurrentText(self.spill_properties["type"])
        spill_layout.addRow("Spill Type:", self.spill_type_combo)
        
        # Date and time
        self.spill_datetime = QDateTimeEdit(QDateTime.currentDateTime())
        self.spill_datetime.setCalendarPopup(True)
        spill_layout.addRow("Date & Time:", self.spill_datetime)
        
        # Location
        location_layout = QHBoxLayout()
        
        self.location_x_spin = QDoubleSpinBox()
        self.location_x_spin.setRange(-10000.0, 10000.0)
        self.location_x_spin.setDecimals(1)
        self.location_x_spin.setValue(self.spill_properties["location"][0])
        self.location_x_spin.setSuffix(" m")
        location_layout.addWidget(QLabel("X:"))
        location_layout.addWidget(self.location_x_spin)
        
        self.location_y_spin = QDoubleSpinBox()
        self.location_y_spin.setRange(-10000.0, 10000.0)
        self.location_y_spin.setDecimals(1)
        self.location_y_spin.setValue(self.spill_properties["location"][1])
        self.location_y_spin.setSuffix(" m")
        location_layout.addWidget(QLabel("Y:"))
        location_layout.addWidget(self.location_y_spin)
        
        spill_layout.addRow("Location:", location_layout)
        
        # Volume
        self.volume_spin = QDoubleSpinBox()
        self.volume_spin.setRange(0.1, 10000.0)
        self.volume_spin.setDecimals(1)
        self.volume_spin.setValue(self.spill_properties["volume"])
        self.volume_spin.setSuffix(" m³")
        spill_layout.addRow("Total Volume:", self.volume_spin)
        
        # Rate
        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setRange(0.1, 1000.0)
        self.rate_spin.setDecimals(1)
        self.rate_spin.setValue(self.spill_properties["rate"])
        self.rate_spin.setSuffix(" m³/h")
        spill_layout.addRow("Spill Rate:", self.rate_spin)
        
        # Duration
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 168.0)  # Up to 1 week
        self.duration_spin.setDecimals(1)
        self.duration_spin.setValue(self.spill_properties["duration"])
        self.duration_spin.setSuffix(" hours")
        spill_layout.addRow("Spill Duration:", self.duration_spin)
        
        config_splitter.addWidget(spill_group)
        
        # Fluid configuration
        fluid_group = QGroupBox("Fluid Properties")
        fluid_layout = QFormLayout(fluid_group)
        
        self.fluid_type_combo = QComboBox()
        self.fluid_type_combo.addItems(["Crude Oil", "Diesel", "Gasoline", "Jet Fuel", "Bunker Fuel", "Natural Gas"])
        self.fluid_type_combo.setCurrentText(self.fluid_properties["type"])
        fluid_layout.addRow("Fluid Type:", self.fluid_type_combo)
        
        self.density_spin = QDoubleSpinBox()
        self.density_spin.setRange(500.0, 1500.0)
        self.density_spin.setDecimals(1)
        self.density_spin.setValue(self.fluid_properties["density"])
        self.density_spin.setSuffix(" kg/m³")
        fluid_layout.addRow("Density:", self.density_spin)
        
        self.viscosity_spin = QDoubleSpinBox()
        self.viscosity_spin.setRange(0.001, 1.0)
        self.viscosity_spin.setDecimals(3)
        self.viscosity_spin.setValue(self.fluid_properties["viscosity"])
        self.viscosity_spin.setSuffix(" Pa·s")
        fluid_layout.addRow("Viscosity:", self.viscosity_spin)
        
        self.pour_point_spin = QDoubleSpinBox()
        self.pour_point_spin.setRange(233.15, 313.15)  # -40°C to 40°C
        self.pour_point_spin.setDecimals(1)
        self.pour_point_spin.setValue(self.fluid_properties["pour_point"])
        self.pour_point_spin.setSuffix(" K")
        fluid_layout.addRow("Pour Point:", self.pour_point_spin)
        
        self.api_gravity_spin = QDoubleSpinBox()
        self.api_gravity_spin.setRange(10.0, 50.0)
        self.api_gravity_spin.setDecimals(1)
        self.api_gravity_spin.setValue(self.fluid_properties["api_gravity"])
        self.api_gravity_spin.setSuffix(" °API")
        fluid_layout.addRow("API Gravity:", self.api_gravity_spin)
        
        config_splitter.addWidget(fluid_group)
        
        # Environmental configuration
        env_group = QGroupBox("Environmental Conditions")
        env_layout = QFormLayout(env_group)
        
        self.wind_speed_spin = QDoubleSpinBox()
        self.wind_speed_spin.setRange(0.0, 30.0)
        self.wind_speed_spin.setDecimals(1)
        self.wind_speed_spin.setValue(self.environmental_conditions["wind_speed"])
        self.wind_speed_spin.setSuffix(" m/s")
        env_layout.addRow("Wind Speed:", self.wind_speed_spin)
        
        self.wind_direction_spin = QDoubleSpinBox()
        self.wind_direction_spin.setRange(0.0, 359.9)
        self.wind_direction_spin.setDecimals(1)
        self.wind_direction_spin.setValue(self.environmental_conditions["wind_direction"])
        self.wind_direction_spin.setSuffix("°")
        env_layout.addRow("Wind Direction:", self.wind_direction_spin)
        
        self.current_speed_spin = QDoubleSpinBox()
        self.current_speed_spin.setRange(0.0, 5.0)
        self.current_speed_spin.setDecimals(2)
        self.current_speed_spin.setValue(self.environmental_conditions["current_speed"])
        self.current_speed_spin.setSuffix(" m/s")
        env_layout.addRow("Current Speed:", self.current_speed_spin)
        
        self.current_direction_spin = QDoubleSpinBox()
        self.current_direction_spin.setRange(0.0, 359.9)
        self.current_direction_spin.setDecimals(1)
        self.current_direction_spin.setValue(self.environmental_conditions["current_direction"])
        self.current_direction_spin.setSuffix("°")
        env_layout.addRow("Current Direction:", self.current_direction_spin)
        
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(263.15, 313.15)  # -10°C to 40°C
        self.temperature_spin.setDecimals(1)
        self.temperature_spin.setValue(self.environmental_conditions["temperature"])
        self.temperature_spin.setSuffix(" K")
        env_layout.addRow("Temperature:", self.temperature_spin)
        
        self.wave_height_spin = QDoubleSpinBox()
        self.wave_height_spin.setRange(0.0, 10.0)
        self.wave_height_spin.setDecimals(1)
        self.wave_height_spin.setValue(self.environmental_conditions["wave_height"])
        self.wave_height_spin.setSuffix(" m")
        env_layout.addRow("Wave Height:", self.wave_height_spin)
        
        self.shoreline_proximity_spin = QDoubleSpinBox()
        self.shoreline_proximity_spin.setRange(100.0, 50000.0)
        self.shoreline_proximity_spin.setDecimals(0)
        self.shoreline_proximity_spin.setValue(self.environmental_conditions["shoreline_proximity"])
        self.shoreline_proximity_spin.setSuffix(" m")
        env_layout.addRow("Shoreline Proximity:", self.shoreline_proximity_spin)
        
        config_splitter.addWidget(env_group)
        
        # Balance the splitter sections
        config_splitter.setSizes([1, 1, 1])
        
        config_layout.addWidget(config_splitter)
        
        # Preset selection and buttons
        preset_layout = QHBoxLayout()
        
        preset_layout.addWidget(QLabel("Presets:"))
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Custom",
            "Small Oil Spill (Offshore)",
            "Medium Oil Spill (Nearshore)",
            "Large Oil Spill (Coastal)",
            "Pipeline Rupture",
            "Tanker Accident",
            "Platform Blowout"
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
        
        # Results tab
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        
        # Tabs for different analyses
        results_tabs = QTabWidget()
        
        # Trajectory tab
        trajectory_tab = QWidget()
        trajectory_layout = QVBoxLayout(trajectory_tab)
        
        trajectory_layout.addWidget(QLabel("Spill Trajectory Analysis"))
        
        # Trajectory results table
        self.trajectory_table = QTableWidget(10, 3)
        self.trajectory_table.setHorizontalHeaderLabels(["Time (h)", "Distance (km)", "Direction"])
        self.trajectory_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        trajectory_layout.addWidget(self.trajectory_table)
        
        # Impact summary
        impact_group = QGroupBox("Impact Summary")
        impact_layout = QFormLayout(impact_group)
        
        self.max_distance_label = QLabel("0.0 km")
        impact_layout.addRow("Maximum Distance:", self.max_distance_label)
        
        self.travel_direction_label = QLabel("N/A")
        impact_layout.addRow("Travel Direction:", self.travel_direction_label)
        
        self.shoreline_impact_label = QLabel("N/A")
        impact_layout.addRow("Shoreline Impact Time:", self.shoreline_impact_label)
        
        self.shoreline_length_label = QLabel("0.0 km")
        impact_layout.addRow("Shoreline Length Affected:", self.shoreline_length_label)
        
        trajectory_layout.addWidget(impact_group)
        
        results_tabs.addTab(trajectory_tab, "Trajectory")
        
        # Weathering tab
        weathering_tab = QWidget()
        weathering_layout = QVBoxLayout(weathering_tab)
        weathering_layout.addWidget(QLabel("Oil Weathering Analysis"))
        
        # Weathering results table
        self.weathering_table = QTableWidget(10, 5)
        self.weathering_table.setHorizontalHeaderLabels([
            "Time (h)", "Evaporated (%)", "Dispersed (%)", "Emulsified (%)", "Remaining (%)"
        ])
        self.weathering_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        weathering_layout.addWidget(self.weathering_table)
        
        # Weathering summary
        weathering_group = QGroupBox("Weathering Summary")
        weathering_sum_layout = QFormLayout(weathering_group)
        
        self.max_evap_label = QLabel("0.0%")
        weathering_sum_layout.addRow("Maximum Evaporation:", self.max_evap_label)
        
        self.max_disp_label = QLabel("0.0%")
        weathering_sum_layout.addRow("Maximum Dispersion:", self.max_disp_label)
        
        self.max_emul_label = QLabel("0.0%")
        weathering_sum_layout.addRow("Maximum Emulsification:", self.max_emul_label)
        
        self.min_remain_label = QLabel("100.0%")
        weathering_sum_layout.addRow("Minimum Remaining:", self.min_remain_label)
        
        self.half_life_label = QLabel("N/A")
        weathering_sum_layout.addRow("Spill Half-life:", self.half_life_label)
        
        weathering_layout.addWidget(weathering_group)
        
        results_tabs.addTab(weathering_tab, "Weathering")
        
        # Response tab
        response_tab = QWidget()
        response_layout = QVBoxLayout(response_tab)
        response_layout.addWidget(QLabel("Response Planning Analysis"))
        
        # Response requirements
        response_group = QGroupBox("Response Requirements")
        response_layout_inner = QFormLayout(response_group)
        
        self.boom_length_label = QLabel("0.0 m")
        response_layout_inner.addRow("Containment Boom Required:", self.boom_length_label)
        
        self.skimmer_capacity_label = QLabel("0.0 m³/h")
        response_layout_inner.addRow("Skimmer Capacity Required:", self.skimmer_capacity_label)
        
        self.storage_capacity_label = QLabel("0.0 m³")
        response_layout_inner.addRow("Storage Capacity Required:", self.storage_capacity_label)
        
        self.dispersant_volume_label = QLabel("0.0 m³")
        response_layout_inner.addRow("Dispersant Volume (if used):", self.dispersant_volume_label)
        
        self.personnel_label = QLabel("0")
        response_layout_inner.addRow("Estimated Personnel Required:", self.personnel_label)
        
        response_layout.addWidget(response_group)
        
        # Time windows
        window_group = QGroupBox("Response Time Windows")
        window_layout = QFormLayout(window_group)
        
        self.mechanical_window_label = QLabel("N/A")
        window_layout.addRow("Mechanical Recovery Window:", self.mechanical_window_label)
        
        self.dispersant_window_label = QLabel("N/A")
        window_layout.addRow("Dispersant Application Window:", self.dispersant_window_label)
        
        self.in_situ_window_label = QLabel("N/A")
        window_layout.addRow("In-situ Burning Window:", self.in_situ_window_label)
        
        self.shoreline_window_label = QLabel("N/A")
        window_layout.addRow("Shoreline Protection Window:", self.shoreline_window_label)
        
        response_layout.addWidget(window_group)
        
        results_tabs.addTab(response_tab, "Response")
        
        results_layout.addWidget(results_tabs)
        
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
        
        results_layout.addWidget(export_group)
        
        self.tab_widget.addTab(results_tab, "Results")
        
        self.layout.addWidget(self.tab_widget)
    
    def _connect_signals(self):
        """Connect signals to slots."""
        # Connect spill map view signals
        self.spill_map_view.interaction.connect(self._handle_view_interaction)
        
        # Connect configuration controls
        self.spill_type_combo.currentTextChanged.connect(self._on_spill_type_changed)
        self.fluid_type_combo.currentTextChanged.connect(self._on_fluid_type_changed)
        
        # Connect location spin boxes
        self.location_x_spin.valueChanged.connect(self._on_location_changed)
        self.location_y_spin.valueChanged.connect(self._on_location_changed)
        
        # Connect volume, rate and duration spin boxes
        self.volume_spin.valueChanged.connect(self._on_volume_changed)
        self.rate_spin.valueChanged.connect(self._on_rate_changed)
        self.duration_spin.valueChanged.connect(self._on_duration_changed)
    
    def _handle_view_interaction(self, interaction_data: Dict[str, Any]):
        """
        Handle interactions with the spill map view.
        
        Args:
            interaction_data (Dict[str, Any]): Interaction data
        """
        if interaction_data.get("action") == "click":
            # User has clicked on the map
            position = interaction_data.get("position", (0.0, 0.0))
            
            # Update location spin boxes
            self.location_x_spin.setValue(position[0])
            self.location_y_spin.setValue(position[1])
            
            # Update spill location
            self.spill_properties["location"] = position
            self._update_visualization()
        
        # Update UI to reflect new state
        self._update_ui_state()
    
    def _on_show_grid_toggled(self, checked: bool):
        """
        Handle grid visibility toggle.
        
        Args:
            checked (bool): Whether to show the grid
        """
        self.spill_map_view.show_grid = checked
        self.spill_map_view.update()
    
    def _on_show_wind_toggled(self, checked: bool):
        """
        Handle wind indicator visibility toggle.
        
        Args:
            checked (bool): Whether to show the wind indicator
        """
        self.spill_map_view.show_wind = checked
        self.spill_map_view.update()
    
    def _on_show_current_toggled(self, checked: bool):
        """
        Handle current indicator visibility toggle.
        
        Args:
            checked (bool): Whether to show the current indicator
        """
        self.spill_map_view.show_current = checked
        self.spill_map_view.update()
    
    def _on_show_shoreline_toggled(self, checked: bool):
        """
        Handle shoreline visibility toggle.
        
        Args:
            checked (bool): Whether to show the shoreline
        """
        self.spill_map_view.show_shoreline = checked
        self.spill_map_view.update()
    
    def _on_spill_type_changed(self, spill_type: str):
        """
        Handle spill type change.
        
        Args:
            spill_type (str): The new spill type
        """
        self.spill_properties["type"] = spill_type
    
    def _on_fluid_type_changed(self, fluid_type: str):
        """
        Handle fluid type change.
        
        Args:
            fluid_type (str): The new fluid type
        """
        # Set default values based on fluid type
        if fluid_type == "Crude Oil":
            self.density_spin.setValue(900.0)
            self.viscosity_spin.setValue(0.05)
            self.pour_point_spin.setValue(263.15)  # -10°C
            self.api_gravity_spin.setValue(28.0)
        elif fluid_type == "Diesel":
            self.density_spin.setValue(850.0)
            self.viscosity_spin.setValue(0.003)
            self.pour_point_spin.setValue(253.15)  # -20°C
            self.api_gravity_spin.setValue(35.0)
        elif fluid_type == "Gasoline":
            self.density_spin.setValue(750.0)
            self.viscosity_spin.setValue(0.0005)
            self.pour_point_spin.setValue(233.15)  # -40°C
            self.api_gravity_spin.setValue(45.0)
        elif fluid_type == "Jet Fuel":
            self.density_spin.setValue(800.0)
            self.viscosity_spin.setValue(0.001)
            self.pour_point_spin.setValue(243.15)  # -30°C
            self.api_gravity_spin.setValue(40.0)
        elif fluid_type == "Bunker Fuel":
            self.density_spin.setValue(990.0)
            self.viscosity_spin.setValue(0.5)
            self.pour_point_spin.setValue(283.15)  # 10°C
            self.api_gravity_spin.setValue(15.0)
        elif fluid_type == "Natural Gas":
            self.density_spin.setValue(500.0)  # Liquefied
            self.viscosity_spin.setValue(0.0001)
            self.pour_point_spin.setValue(173.15)  # -100°C (Not applicable but needed)
            self.api_gravity_spin.setValue(50.0)  # Not applicable but needed
        
        self.fluid_properties["type"] = fluid_type
    
    def _on_location_changed(self):
        """Handle spill location change."""
        x = self.location_x_spin.value()
        y = self.location_y_spin.value()
        self.spill_properties["location"] = (x, y)
    
    def _on_volume_changed(self):
        """Handle spill volume change."""
        volume = self.volume_spin.value()
        self.spill_properties["volume"] = volume
        
        # Ensure duration is consistent with volume and rate
        if self.rate_spin.value() > 0:
            max_duration = volume / self.rate_spin.value()
            if self.duration_spin.value() > max_duration:
                self.duration_spin.setValue(max_duration)
    
    def _on_rate_changed(self):
        """Handle spill rate change."""
        rate = self.rate_spin.value()
        self.spill_properties["rate"] = rate
        
        # Ensure duration is consistent with volume and rate
        if rate > 0:
            max_duration = self.volume_spin.value() / rate
            if self.duration_spin.value() > max_duration:
                self.duration_spin.setValue(max_duration)
    
    def _on_duration_changed(self):
        """Handle spill duration change."""
        duration = self.duration_spin.value()
        self.spill_properties["duration"] = duration
        
        # Ensure volume is consistent with duration and rate
        min_volume = duration * self.rate_spin.value()
        if self.volume_spin.value() < min_volume:
            self.volume_spin.setValue(min_volume)
    
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
            self.spill_map_view.animation_speed = speed
    
    def _apply_configuration(self):
        """Apply current configuration to the simulation."""
        # Update spill properties
        self.spill_properties = {
            "type": self.spill_type_combo.currentText(),
            "location": (self.location_x_spin.value(), self.location_y_spin.value()),
            "volume": self.volume_spin.value(),
            "rate": self.rate_spin.value(),
            "duration": self.duration_spin.value(),
            "age": 0.0  # Reset age to 0
        }
        
        # Update fluid properties
        self.fluid_properties = {
            "type": self.fluid_type_combo.currentText(),
            "density": self.density_spin.value(),
            "viscosity": self.viscosity_spin.value(),
            "pour_point": self.pour_point_spin.value(),
            "api_gravity": self.api_gravity_spin.value()
        }
        
        # Update environmental conditions
        self.environmental_conditions = {
            "wind_speed": self.wind_speed_spin.value(),
            "wind_direction": self.wind_direction_spin.value(),
            "current_speed": self.current_speed_spin.value(),
            "current_direction": self.current_direction_spin.value(),
            "temperature": self.temperature_spin.value(),
            "wave_height": self.wave_height_spin.value(),
            "shoreline_proximity": self.shoreline_proximity_spin.value()
        }
        
        # Update visualization
        self._update_visualization()
        
        # Calculate simulation results
        self._calculate_simulation_results()
        
        # Update UI
        self._update_ui_state()
        
        # Clear weathering graph
        self.weathering_graph.clear()
        
        # Show confirmation to user
        QMessageBox.information(
            self,
            "Configuration Applied",
            "Configuration has been applied to the simulation."
        )
    
    def _update_visualization(self):
        """Update the visualization with current configuration."""
        # Update spill map view properties
        self.spill_map_view.set_spill_properties(self.spill_properties)
        self.spill_map_view.set_fluid_properties(self.fluid_properties)
        self.spill_map_view.set_environmental_conditions(self.environmental_conditions)
    
    def _calculate_simulation_results(self):
        """Calculate simulation results based on current configuration."""
        # Most of this calculation is done in the SpillMapView class
        # Here we just need to collect the relevant values for reporting
        
        # Get values from spill map view
        self.simulation_results["max_radius"] = self.spill_map_view.spill_radius
        self.simulation_results["max_area"] = self.spill_map_view.spill_area
        self.simulation_results["evaporated_fraction"] = self.spill_map_view.evaporated_fraction
        self.simulation_results["dispersed_fraction"] = self.spill_map_view.dispersed_fraction
        self.simulation_results["emulsified_fraction"] = self.spill_map_view.emulsified_fraction
        
        # Calculate remaining fraction
        total_loss_fraction = self.spill_map_view.evaporated_fraction + (
            1 - self.spill_map_view.evaporated_fraction
        ) * self.spill_map_view.dispersed_fraction
        
        self.simulation_results["remaining_fraction"] = 1.0 - total_loss_fraction
        self.simulation_results["remaining_volume"] = self.spill_properties["volume"] * self.simulation_results["remaining_fraction"]
        
        # Calculate shoreline impact time (simplified)
        shoreline_proximity = self.environmental_conditions["shoreline_proximity"]
        wind_effect = self.environmental_conditions["wind_speed"] * 0.03  # 3% of wind speed
        current_effect = self.environmental_conditions["current_speed"]
        
        # Calculate combined transport velocity toward shoreline
        # This is a simplified calculation assuming wind and current are perfectly aligned
        # In a real model, we'd need to consider the vector components
        wind_direction_rad = math.radians(self.environmental_conditions["wind_direction"])
        current_direction_rad = math.radians(self.environmental_conditions["current_direction"])
        
        # Calculate transport speed toward positive y (simplified)
        wind_y = wind_effect * math.cos(wind_direction_rad)
        current_y = current_effect * math.cos(current_direction_rad)
        
        total_speed_toward_shore = wind_y + current_y
        
        if total_speed_toward_shore > 0:
            self.simulation_results["shoreline_impact_time"] = shoreline_proximity / total_speed_toward_shore / 3600.0  # convert to hours
        else:
            self.simulation_results["shoreline_impact_time"] = float('inf')
        
        # Calculate shoreline length affected (simplified)
        # This would depend on spill size, shoreline distance, and other factors
        # For this simple model, we assume it's roughly related to the square root of the area
        if self.simulation_results["shoreline_impact_time"] < float('inf'):
            self.simulation_results["shoreline_length_affected"] = math.sqrt(self.simulation_results["max_area"]) * 2.0
        else:
            self.simulation_results["shoreline_length_affected"] = 0.0
        
        # Update tables and labels with results
        self._update_results_tables()
        self._update_results_labels()
    
    def _update_results_tables(self):
        """Update the results tables with simulation results."""
        # Populate trajectory table
        self.trajectory_table.clearContents()
        
        # Calculate trajectory at different times
        times = [1, 3, 6, 12, 24, 36, 48, 60, 72]  # hours
        
        wind_speed = self.environmental_conditions["wind_speed"]
        wind_direction = self.environmental_conditions["wind_direction"]
        current_speed = self.environmental_conditions["current_speed"]
        current_direction = self.environmental_conditions["current_direction"]
        
        for i, time in enumerate(times):
            if i >= self.trajectory_table.rowCount():
                break
                
            # Calculate distance and direction
            wind_effect = wind_speed * 0.03 * time * 3600  # in meters
            wind_x = wind_effect * math.sin(math.radians(wind_direction))
            wind_y = wind_effect * math.cos(math.radians(wind_direction))
            
            current_effect = current_speed * time * 3600  # in meters
            current_x = current_effect * math.sin(math.radians(current_direction))
            current_y = current_effect * math.cos(math.radians(current_direction))
            
            total_x = wind_x + current_x
            total_y = wind_y + current_y
            
            distance = math.sqrt(total_x ** 2 + total_y ** 2)
            
            if distance > 0:
                # Calculate direction
                direction_rad = math.atan2(total_x, total_y)
                direction_deg = math.degrees(direction_rad)
                if direction_deg < 0:
                    direction_deg += 360
                
                # Convert to cardinal direction
                cardinal = self._degrees_to_cardinal(direction_deg)
                
                # Add to table
                self.trajectory_table.setItem(i, 0, QTableWidgetItem(str(time)))
                self.trajectory_table.setItem(i, 1, QTableWidgetItem(f"{distance / 1000:.2f}"))
                self.trajectory_table.setItem(i, 2, QTableWidgetItem(cardinal))
        
        # Populate weathering table
        self.weathering_table.clearContents()
        
        # Simulate weathering processes for different times
        for i, time in enumerate(times):
            if i >= self.weathering_table.rowCount():
                break
                
            # Calculate weathering (simplified models)
            # Evaporation model
            temperature = self.environmental_conditions["temperature"]
            wind_speed = self.environmental_conditions["wind_speed"]
            api_gravity = self.fluid_properties["api_gravity"]
            
            evap_factor = 0.01 * (temperature - 273.15) * wind_speed * api_gravity / 30.0
            evaporated = min(0.8, evap_factor * time / 24.0)  # max 80% evaporation
            
            # Dispersion model
            wave_height = self.environmental_conditions["wave_height"]
            viscosity = self.fluid_properties["viscosity"]
            
            disp_factor = 0.005 * wave_height * (temperature - 273.15) / viscosity
            dispersed = min(0.4, disp_factor * time / 24.0)  # max 40% dispersion
            
            # Emulsification model
            emul_factor = 0.01 * wave_height * (293.15 / temperature)
            emulsified = min(0.8, emul_factor * time / 12.0)  # max 80% emulsification
            
        
            # Calculate total loss and remaining fraction
            total_loss = evaporated + (1 - evaporated) * dispersed
            remaining = 1.0 - total_loss
            
            # Add to table
            self.weathering_table.setItem(i, 0, QTableWidgetItem(str(time)))
            self.weathering_table.setItem(i, 1, QTableWidgetItem(f"{evaporated * 100:.1f}"))
            self.weathering_table.setItem(i, 2, QTableWidgetItem(f"{dispersed * 100:.1f}"))
            self.weathering_table.setItem(i, 3, QTableWidgetItem(f"{emulsified * 100:.1f}"))
            self.weathering_table.setItem(i, 4, QTableWidgetItem(f"{remaining * 100:.1f}"))
            
            # Add data point to weathering graph
            self.weathering_graph.add_data_point(time, evaporated, dispersed, emulsified, remaining)
    
    def _update_results_labels(self):
        """Update the results labels with simulation results."""
        # Trajectory results
        max_time = 72  # 3 days
        
        # Calculate maximum distance after 3 days
        wind_effect = self.environmental_conditions["wind_speed"] * 0.03 * max_time * 3600  # in meters
        wind_x = wind_effect * math.sin(math.radians(self.environmental_conditions["wind_direction"]))
        wind_y = wind_effect * math.cos(math.radians(self.environmental_conditions["wind_direction"]))
        
        current_effect = self.environmental_conditions["current_speed"] * max_time * 3600  # in meters
        current_x = current_effect * math.sin(math.radians(self.environmental_conditions["current_direction"]))
        current_y = current_effect * math.cos(math.radians(self.environmental_conditions["current_direction"]))
        
        total_x = wind_x + current_x
        total_y = wind_y + current_y
        
        max_distance = math.sqrt(total_x ** 2 + total_y ** 2)
        
        self.max_distance_label.setText(f"{max_distance / 1000:.2f} km")
        
        # Calculate overall direction
        if max_distance > 0:
            direction_rad = math.atan2(total_x, total_y)
            direction_deg = math.degrees(direction_rad)
            if direction_deg < 0:
                direction_deg += 360
            
            cardinal = self._degrees_to_cardinal(direction_deg)
            self.travel_direction_label.setText(f"{direction_deg:.1f}° ({cardinal})")
        else:
            self.travel_direction_label.setText("N/A")
        
        # Shoreline impact
        impact_time = self.simulation_results["shoreline_impact_time"]
        if impact_time < float('inf'):
            self.shoreline_impact_label.setText(f"{impact_time:.1f} hours")
        else:
            self.shoreline_impact_label.setText("No impact detected")
        
        shoreline_length = self.simulation_results["shoreline_length_affected"]
        self.shoreline_length_label.setText(f"{shoreline_length / 1000:.2f} km")
        
        # Weathering results
        max_evap = 0.0
        max_disp = 0.0
        max_emul = 0.0
        min_remain = 1.0
        half_life = float('inf')
        
        # Find maximum weathering values and half-life
        if self.weathering_graph.times:
            max_evap = max(self.weathering_graph.evaporated)
            max_disp = max(self.weathering_graph.dispersed)
            max_emul = max(self.weathering_graph.emulsified)
            min_remain = min(self.weathering_graph.remaining)
            
            # Find half-life (time when remaining volume is 50% of original)
            for i, remain in enumerate(self.weathering_graph.remaining):
                if remain <= 0.5:
                    if i > 0:
                        # Linear interpolation to find more precise half-life
                        t1 = self.weathering_graph.times[i-1]
                        t2 = self.weathering_graph.times[i]
                        r1 = self.weathering_graph.remaining[i-1]
                        r2 = remain
                        
                        half_life = t1 + (t2 - t1) * (0.5 - r1) / (r2 - r1)
                    else:
                        half_life = self.weathering_graph.times[i]
                    break
        
        self.max_evap_label.setText(f"{max_evap * 100:.1f}%")
        self.max_disp_label.setText(f"{max_disp * 100:.1f}%")
        self.max_emul_label.setText(f"{max_emul * 100:.1f}%")
        self.min_remain_label.setText(f"{min_remain * 100:.1f}%")
        
        if half_life < float('inf'):
            self.half_life_label.setText(f"{half_life:.1f} hours")
        else:
            self.half_life_label.setText("N/A")
        
        # Response requirements
        # Calculate containment boom length (simplified)
        spill_radius = self.simulation_results["max_radius"]
        boom_length = 2 * math.pi * spill_radius * 1.5  # 1.5x the perimeter for safety
        self.boom_length_label.setText(f"{boom_length:.1f} m")
        
        # Calculate skimmer capacity (simplified)
        volume = self.spill_properties["volume"]
        skimmer_capacity = volume / 48  # Recover in 48 hours
        self.skimmer_capacity_label.setText(f"{skimmer_capacity:.1f} m³/h")
        
        # Calculate storage capacity (simplified)
        # Account for water in emulsion (3x volume)
        emulsified_volume = volume * min_remain * max_emul * 3.0
        storage_capacity = volume * min_remain * (1 - max_emul) + emulsified_volume
        self.storage_capacity_label.setText(f"{storage_capacity:.1f} m³")
        
        # Calculate dispersant volume (if applicable)
        dispersant_ratio = 1/20  # Typically 1:20 dispersant:oil ratio
        dispersant_volume = volume * min_remain * 0.5 * dispersant_ratio  # Assume 50% of remaining oil is treated
        self.dispersant_volume_label.setText(f"{dispersant_volume:.1f} m³")
        
        # Estimate personnel required (very simplified)
        personnel = math.ceil(boom_length / 100) * 2 + math.ceil(skimmer_capacity / 5) * 2 + 5
        self.personnel_label.setText(str(personnel))
        
        # Response time windows
        mechanical_window_end = 48  # hours (typical for mechanical recovery)
        if impact_time < mechanical_window_end:
            self.mechanical_window_label.setText(f"0 - {min(impact_time, mechanical_window_end):.1f} hours")
        else:
            self.mechanical_window_label.setText(f"0 - {mechanical_window_end} hours")
        
        dispersant_window_end = 24  # hours (typical for dispersant application)
        if impact_time < dispersant_window_end:
            self.dispersant_window_label.setText(f"0 - {min(impact_time, dispersant_window_end):.1f} hours")
        else:
            self.dispersant_window_label.setText(f"0 - {dispersant_window_end} hours")
        
        in_situ_window_end = 24  # hours (typical for in-situ burning)
        if impact_time < in_situ_window_end:
            self.in_situ_window_label.setText(f"0 - {min(impact_time, in_situ_window_end):.1f} hours")
        else:
            self.in_situ_window_label.setText(f"0 - {in_situ_window_end} hours")
        
        # Shoreline protection window
        if impact_time < float('inf'):
            prep_time = max(0, impact_time - 6)  # Assume 6 hours needed for preparation
            self.shoreline_window_label.setText(f"0 - {prep_time:.1f} hours")
        else:
            self.shoreline_window_label.setText("Not required")
    
    def _degrees_to_cardinal(self, degrees: float) -> str:
        """
        Convert degrees to cardinal direction.
        
        Args:
            degrees (float): Direction in degrees
            
        Returns:
            str: Cardinal direction
        """
        directions = [
            "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
        ]
        
        index = round(degrees / 22.5) % 16
        return directions[index]
    
    def _start_simulation(self):
        """Start the spill simulation."""
        # Update state
        self.is_simulating = True
        
        # Reset simulation time
        self.current_simulation_time = 0.0
        
        # Update spill age
        self.spill_properties["age"] = 0.0
        self.spill_map_view.set_spill_age(0.0)
        
        # Update progress bar
        self.progress_bar.setValue(0)
        self.time_label.setText("0.0 hours")
        
        # Start animation in map view
        self.spill_map_view.start_animation(self.simulation_speed)
        
        # Clear and reset weathering graph
        self.weathering_graph.clear()
        
        # Enable/disable controls
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        
        # Set up timer for simulation updates
        self._setup_simulation_timer()
    
    def _pause_simulation(self):
        """Pause the spill simulation."""
        # Update state
        self.is_simulating = False
        
        # Pause animation in map view
        self.spill_map_view.pause_animation()
        
        # Enable/disable controls
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(True)
    
    def _stop_simulation(self):
        """Stop and reset the spill simulation."""
        # Update state
        self.is_simulating = False
        
        # Stop animation in map view
        self.spill_map_view.stop_animation()
        
        # Reset simulation time
        self.current_simulation_time = 0.0
        
        # Update progress bar
        self.progress_bar.setValue(0)
        self.time_label.setText("0.0 hours")
        
        # Enable/disable controls
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        
        # Clear weathering graph
        self.weathering_graph.clear()
    
    def _setup_simulation_timer(self):
        """Set up timer for simulation updates."""
        # Create timer if it doesn't exist
        if not hasattr(self, 'simulation_timer'):
            self.simulation_timer = QTimer(self)
            self.simulation_timer.timeout.connect(self._update_simulation)
        
        # Start timer - update every second
        self.simulation_timer.start(1000)
    
    def _update_simulation(self):
        """Update simulation state based on timer."""
        if not self.is_simulating:
            return
        
        # Get current time from map view
        self.current_simulation_time = self.spill_map_view.spill_age
        
        # Update progress bar
        self.progress_bar.setValue(min(72, int(self.current_simulation_time)))
        self.time_label.setText(f"{self.current_simulation_time:.1f} hours")
        
        # Collect data for weathering graph
        if self.current_simulation_time % 1.0 < 0.1:  # Add data point approximately every hour
            self.weathering_graph.add_data_point(
                self.current_simulation_time,
                self.spill_map_view.evaporated_fraction,
                self.spill_map_view.dispersed_fraction,
                self.spill_map_view.emulsified_fraction,
                1.0 - self.spill_map_view.evaporated_fraction - (
                    1.0 - self.spill_map_view.evaporated_fraction
                ) * self.spill_map_view.dispersed_fraction
            )
        
        # Check if simulation should end
        if self.current_simulation_time >= 72.0:  # 3 days
            self._pause_simulation()
            
            # Calculate final results
            self._calculate_simulation_results()
            
            # Show results tab
            self.tab_widget.setCurrentIndex(2)  # Results tab
    
    def _load_preset(self):
        """Load selected preset configuration."""
        preset = self.preset_combo.currentText()
        
        if preset == "Small Oil Spill (Offshore)":
            # Set spill properties
            self.spill_type_combo.setCurrentText("Surface")
            self.volume_spin.setValue(10.0)
            self.rate_spin.setValue(5.0)
            self.duration_spin.setValue(2.0)
            
            # Set fluid properties
            self.fluid_type_combo.setCurrentText("Crude Oil")
            
            # Set environmental conditions
            self.wind_speed_spin.setValue(7.0)
            self.wind_direction_spin.setValue(45.0)
            self.current_speed_spin.setValue(0.3)
            self.current_direction_spin.setValue(30.0)
            self.wave_height_spin.setValue(1.0)
            self.shoreline_proximity_spin.setValue(20000.0)
            
        elif preset == "Medium Oil Spill (Nearshore)":
            # Set spill properties
            self.spill_type_combo.setCurrentText("Surface")
            self.volume_spin.setValue(100.0)
            self.rate_spin.setValue(20.0)
            self.duration_spin.setValue(5.0)
            
            # Set fluid properties
            self.fluid_type_combo.setCurrentText("Crude Oil")
            
            # Set environmental conditions
            self.wind_speed_spin.setValue(5.0)
            self.wind_direction_spin.setValue(30.0)
            self.current_speed_spin.setValue(0.5)
            self.current_direction_spin.setValue(45.0)
            self.wave_height_spin.setValue(0.8)
            self.shoreline_proximity_spin.setValue(5000.0)
            
        elif preset == "Large Oil Spill (Coastal)":
            # Set spill properties
            self.spill_type_combo.setCurrentText("Surface")
            self.volume_spin.setValue(1000.0)
            self.rate_spin.setValue(50.0)
            self.duration_spin.setValue(20.0)
            
            # Set fluid properties
            self.fluid_type_combo.setCurrentText("Crude Oil")
            
            # Set environmental conditions
            self.wind_speed_spin.setValue(10.0)
            self.wind_direction_spin.setValue(180.0)
            self.current_speed_spin.setValue(0.7)
            self.current_direction_spin.setValue(190.0)
            self.wave_height_spin.setValue(1.5)
            self.shoreline_proximity_spin.setValue(1000.0)
            
        elif preset == "Pipeline Rupture":
            # Set spill properties
            self.spill_type_combo.setCurrentText("Jet")
            self.volume_spin.setValue(500.0)
            self.rate_spin.setValue(100.0)
            self.duration_spin.setValue(5.0)
            
            # Set fluid properties
            self.fluid_type_combo.setCurrentText("Crude Oil")
            self.viscosity_spin.setValue(0.1)
            
            # Set environmental conditions
            self.wind_speed_spin.setValue(3.0)
            self.wind_direction_spin.setValue(90.0)
            self.current_speed_spin.setValue(0.2)
            self.current_direction_spin.setValue(90.0)
            self.wave_height_spin.setValue(0.3)
            self.shoreline_proximity_spin.setValue(8000.0)
            
        elif preset == "Tanker Accident":
            # Set spill properties
            self.spill_type_combo.setCurrentText("Surface")
            self.volume_spin.setValue(5000.0)
            self.rate_spin.setValue(200.0)
            self.duration_spin.setValue(25.0)
            
            # Set fluid properties
            self.fluid_type_combo.setCurrentText("Bunker Fuel")
            
            # Set environmental conditions
            self.wind_speed_spin.setValue(12.0)
            self.wind_direction_spin.setValue(220.0)
            self.current_speed_spin.setValue(1.0)
            self.current_direction_spin.setValue(200.0)
            self.wave_height_spin.setValue(2.5)
            self.shoreline_proximity_spin.setValue(3000.0)
            
        elif preset == "Platform Blowout":
            # Set spill properties
            self.spill_type_combo.setCurrentText("Subsurface")
            self.volume_spin.setValue(10000.0)
            self.rate_spin.setValue(500.0)
            self.duration_spin.setValue(20.0)
            
            # Set fluid properties
            self.fluid_type_combo.setCurrentText("Crude Oil")
            self.api_gravity_spin.setValue(35.0)
            
            # Set environmental conditions
            self.wind_speed_spin.setValue(8.0)
            self.wind_direction_spin.setValue(150.0)
            self.current_speed_spin.setValue(0.8)
            self.current_direction_spin.setValue(140.0)
            self.wave_height_spin.setValue(1.8)
            self.shoreline_proximity_spin.setValue(15000.0)
        
        # Apply the preset configuration
        self._apply_configuration()
    
    def _save_preset(self):
        """Save current configuration as a custom preset."""
        # Ask user for preset name
        from PyQt5.QtWidgets import QInputDialog
        
        preset_name, ok = QInputDialog.getText(
            self,
            "Save Preset",
            "Enter a name for this preset configuration:"
        )
        
        if ok and preset_name:
            # Add to presets combo box if not already present
            if self.preset_combo.findText(preset_name) == -1:
                self.preset_combo.addItem(preset_name)
            
            # Set as current preset
            self.preset_combo.setCurrentText(preset_name)
            
            # Save preset to settings (would be expanded in real implementation)
            QMessageBox.information(
                self,
                "Preset Saved",
                f"Preset '{preset_name}' has been saved."
            )
    
    def _export_data(self):
        """Export simulation data to CSV file."""
        # Get file path
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Data",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not filepath:
            return
        
        # Ensure file has .csv extension
        if not filepath.lower().endswith('.csv'):
            filepath += '.csv'
        
        try:
            # Open file for writing
            with open(filepath, 'w') as f:
                # Write header
                f.write("Time (h),Spill Radius (m),Spill Area (m²),Evaporated (%),Dispersed (%),Emulsified (%),Remaining (%)\n")
                
                # Write data
                for i in range(len(self.weathering_graph.times)):
                    time = self.weathering_graph.times[i]
                    evap = self.weathering_graph.evaporated[i] * 100
                    disp = self.weathering_graph.dispersed[i] * 100
                    emul = self.weathering_graph.emulsified[i] * 100
                    remain = self.weathering_graph.remaining[i] * 100
                    
                    # Calculate radius and area at this time (simplified)
                    k = 1.5  # Same constant used in spill map view
                    volume = self.spill_properties["volume"] * self.weathering_graph.remaining[i]
                    emulsified_volume = volume * self.weathering_graph.emulsified[i] * 3.0
                    effective_volume = volume * (1 - self.weathering_graph.emulsified[i]) + emulsified_volume
                    
                    radius = k * (effective_volume ** (1/3)) * (time ** (1/4))
                    area = math.pi * (radius ** 2)
                    
                    f.write(f"{time:.1f},{radius:.1f},{area:.1f},{evap:.1f},{disp:.1f},{emul:.1f},{remain:.1f}\n")
            
            QMessageBox.information(
                self,
                "Data Exported",
                f"Simulation data has been exported to:\n{filepath}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Exporting Data",
                f"An error occurred while exporting data:\n{str(e)}"
            )
    
    def _generate_report(self):
        """Generate a simulation report."""
        # Get file path
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Generate Report",
            "",
            "HTML Files (*.html);;All Files (*)"
        )
        
        if not filepath:
            return
        
        # Ensure file has .html extension
        if not filepath.lower().endswith('.html'):
            filepath += '.html'
        
        try:
            # Format date and time
            report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create report HTML
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Spill Simulation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .section {{ margin-bottom: 30px; }}
    </style>
</head>
<body>
    <h1>Oil Spill Simulation Report</h1>
    <p>Report Generated: {report_date}</p>
    
    <div class="section">
        <h2>Spill Information</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Spill Type</td><td>{self.spill_properties["type"]}</td></tr>
            <tr><td>Total Volume</td><td>{self.spill_properties["volume"]:.1f} m³</td></tr>
            <tr><td>Spill Rate</td><td>{self.spill_properties["rate"]:.1f} m³/h</td></tr>
            <tr><td>Spill Duration</td><td>{self.spill_properties["duration"]:.1f} hours</td></tr>
            <tr><td>Fluid Type</td><td>{self.fluid_properties["type"]}</td></tr>
            <tr><td>API Gravity</td><td>{self.fluid_properties["api_gravity"]:.1f} °API</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Environmental Conditions</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Wind Speed</td><td>{self.environmental_conditions["wind_speed"]:.1f} m/s</td></tr>
            <tr><td>Wind Direction</td><td>{self.environmental_conditions["wind_direction"]:.1f}° ({self._degrees_to_cardinal(self.environmental_conditions["wind_direction"])})</td></tr>
            <tr><td>Current Speed</td><td>{self.environmental_conditions["current_speed"]:.1f} m/s</td></tr>
            <tr><td>Current Direction</td><td>{self.environmental_conditions["current_direction"]:.1f}° ({self._degrees_to_cardinal(self.environmental_conditions["current_direction"])})</td></tr>
            <tr><td>Temperature</td><td>{self.environmental_conditions["temperature"]:.1f} K ({self.environmental_conditions["temperature"] - 273.15:.1f}°C)</td></tr>
            <tr><td>Wave Height</td><td>{self.environmental_conditions["wave_height"]:.1f} m</td></tr>
            <tr><td>Shoreline Proximity</td><td>{self.environmental_conditions["shoreline_proximity"]:.1f} m</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Simulation Results</h2>
        <h3>Trajectory Summary</h3>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Maximum Distance (72h)</td><td>{self.max_distance_label.text()}</td></tr>
            <tr><td>Travel Direction</td><td>{self.travel_direction_label.text()}</td></tr>
            <tr><td>Shoreline Impact Time</td><td>{self.shoreline_impact_label.text()}</td></tr>
            <tr><td>Shoreline Length Affected</td><td>{self.shoreline_length_label.text()}</td></tr>
        </table>
        
        <h3>Weathering Summary</h3>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Maximum Evaporation</td><td>{self.max_evap_label.text()}</td></tr>
            <tr><td>Maximum Dispersion</td><td>{self.max_disp_label.text()}</td></tr>
            <tr><td>Maximum Emulsification</td><td>{self.max_emul_label.text()}</td></tr>
            <tr><td>Minimum Remaining</td><td>{self.min_remain_label.text()}</td></tr>
            <tr><td>Spill Half-life</td><td>{self.half_life_label.text()}</td></tr>
        </table>
        
        <h3>Response Requirements</h3>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Containment Boom Required</td><td>{self.boom_length_label.text()}</td></tr>
            <tr><td>Skimmer Capacity Required</td><td>{self.skimmer_capacity_label.text()}</td></tr>
            <tr><td>Storage Capacity Required</td><td>{self.storage_capacity_label.text()}</td></tr>
            <tr><td>Dispersant Volume (if used)</td><td>{self.dispersant_volume_label.text()}</td></tr>
            <tr><td>Estimated Personnel Required</td><td>{self.personnel_label.text()}</td></tr>
        </table>
        
        <h3>Response Time Windows</h3>
        <table>
            <tr><th>Response Type</th><th>Time Window</th></tr>
            <tr><td>Mechanical Recovery</td><td>{self.mechanical_window_label.text()}</td></tr>
            <tr><td>Dispersant Application</td><td>{self.dispersant_window_label.text()}</td></tr>
            <tr><td>In-situ Burning</td><td>{self.in_situ_window_label.text()}</td></tr>
            <tr><td>Shoreline Protection</td><td>{self.shoreline_window_label.text()}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Disclaimer</h2>
        <p>This simulation uses simplified models and should be used for planning purposes only. Real-world spill behavior may differ based on additional factors not accounted for in this simulation.</p>
    </div>
</body>
</html>"""
            
            # Write to file
            with open(filepath, 'w') as f:
                f.write(html)
            
            QMessageBox.information(
                self,
                "Report Generated",
                f"Simulation report has been generated at:\n{filepath}"
            )
            
            # Try to open the report in default browser
            import webbrowser
            webbrowser.open(filepath)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Generating Report",
                f"An error occurred while generating report:\n{str(e)}"
            )
    
    def _update_ui_state(self):
        """Update UI state based on current simulation state."""
        # Enable/disable controls based on simulation state
        self.play_button.setEnabled(not self.is_simulating)
        self.pause_button.setEnabled(self.is_simulating)
        self.stop_button.setEnabled(self.is_simulating or self.current_simulation_time > 0)
        
        # Disable configuration controls during simulation
        is_configurable = not self.is_simulating
        
        # Configuration tab widgets
        for tab_index in range(1, self.tab_widget.count()):
            widget = self.tab_widget.widget(tab_index)
            if hasattr(widget, 'setEnabled'):
                widget.setEnabled(is_configurable)