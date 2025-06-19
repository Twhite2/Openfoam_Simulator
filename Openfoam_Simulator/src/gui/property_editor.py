#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Property editor for Openfoam_Simulator application.

This module implements a property editor panel that allows users to view and edit
properties of selected items in the project, including:
- Geometry properties
- Mesh settings
- Material properties
- Boundary conditions
- Solver settings
- Visualization parameters
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeView, QHeaderView, 
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, 
    QCheckBox, QPushButton, QColorDialog, QToolButton, QFrame,
    QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, 
    QStyledItemDelegate, QItemDelegate, QAbstractItemView, 
    QGridLayout, QFormLayout, QGroupBox, QTabWidget, QMenu,
    QAction, QSizePolicy, QToolBar
)
from PyQt5.QtCore import (
    Qt, QSize, QModelIndex, QVariant, pyqtSignal, QObject,
    QAbstractTableModel, QSortFilterProxyModel, QRegExp
)
from PyQt5.QtGui import (
    QIcon, QColor, QBrush, QFont, QPalette, QValidator, QRegExpValidator
)

# Import project explorer for item types
from .project_explorer import ProjectItem

# Import utility modules
from ..utils.logger import get_logger
from ..config import get_value, set_value

logger = get_logger(__name__)


class PropertyItem:
    """
    Represents a property with name, value, type, and other metadata.
    """
    
    # Property types
    STRING = 0
    INTEGER = 1
    FLOAT = 2
    BOOLEAN = 3
    COLOR = 4
    ENUM = 5
    VECTOR = 6
    FILE = 7
    
    def __init__(self, name: str, value: Any, prop_type: int, category: str = "General",
                 read_only: bool = False, options: List[str] = None, 
                 min_value: Any = None, max_value: Any = None,
                 description: str = None, unit: str = None,
                 callback: Callable = None):
        """
        Initialize a property item.
        
        Args:
            name (str): Property name
            value (Any): Property value
            prop_type (int): Property type (use PropertyItem constants)
            category (str): Property category for grouping
            read_only (bool): Whether the property is read-only
            options (List[str]): Options for ENUM type
            min_value (Any): Minimum value for numeric types
            max_value (Any): Maximum value for numeric types
            description (str): Property description for tooltip
            unit (str): Unit for the property (displayed after value)
            callback (Callable): Function to call when property changes
        """
        self.name = name
        self.value = value
        self.prop_type = prop_type
        self.category = category
        self.read_only = read_only
        self.options = options or []
        self.min_value = min_value
        self.max_value = max_value
        self.description = description
        self.unit = unit
        self.callback = callback
        
        # Default validator
        self.validator = None
        
        # Set up validator based on type
        if prop_type == self.INTEGER:
            self.validator = QRegExpValidator(QRegExp("-?[0-9]+"))
        elif prop_type == self.FLOAT:
            self.validator = QRegExpValidator(QRegExp("-?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?"))
        
    def formatted_value(self) -> str:
        """
        Get a formatted string representation of the value.
        
        Returns:
            str: The formatted value
        """
        if self.value is None:
            return ""
        
        if self.prop_type == self.BOOLEAN:
            return "True" if self.value else "False"
        elif self.prop_type == self.COLOR:
            # Return color as hex string
            if isinstance(self.value, QColor):
                return self.value.name()
            elif isinstance(self.value, (list, tuple)) and len(self.value) >= 3:
                color = QColor(int(self.value[0] * 255), 
                              int(self.value[1] * 255), 
                              int(self.value[2] * 255))
                return color.name()
            else:
                return str(self.value)
        elif self.prop_type == self.VECTOR:
            # Format vector as comma-separated values
            if isinstance(self.value, (list, tuple)):
                return ", ".join(str(x) for x in self.value)
            else:
                return str(self.value)
        elif self.unit and self.prop_type in [self.INTEGER, self.FLOAT]:
            # Add unit to numeric values
            return f"{self.value} {self.unit}"
        else:
            return str(self.value)
    
    def set_value(self, value: Any):
        """
        Set the property value.
        
        Args:
            value (Any): The new value
        """
        if self.read_only:
            return
        
        # Convert value to appropriate type
        if self.prop_type == self.INTEGER:
            try:
                value = int(value)
                # Check against min/max
                if self.min_value is not None and value < self.min_value:
                    value = self.min_value
                if self.max_value is not None and value > self.max_value:
                    value = self.max_value
            except (ValueError, TypeError):
                return
        elif self.prop_type == self.FLOAT:
            try:
                value = float(value)
                # Check against min/max
                if self.min_value is not None and value < self.min_value:
                    value = self.min_value
                if self.max_value is not None and value > self.max_value:
                    value = self.max_value
            except (ValueError, TypeError):
                return
        elif self.prop_type == self.BOOLEAN:
            if isinstance(value, str):
                value = value.lower() in ['true', 'yes', '1', 'y', 't']
            else:
                value = bool(value)
        elif self.prop_type == self.COLOR:
            # Handle different color formats
            if isinstance(value, str):
                # Assume hex color
                value = QColor(value)
            elif isinstance(value, (list, tuple)) and len(value) >= 3:
                # RGB or RGBA
                if all(0 <= x <= 1 for x in value[:3]):
                    # Normalized RGB
                    value = QColor(int(value[0] * 255), 
                                  int(value[1] * 255), 
                                  int(value[2] * 255))
                else:
                    # Integer RGB
                    value = QColor(*value[:3])
            else:
                # Unsupported format
                return
        elif self.prop_type == self.VECTOR:
            # Parse vector from string
            if isinstance(value, str):
                try:
                    # Split by comma and convert to float
                    value = [float(x.strip()) for x in value.split(",")]
                except (ValueError, TypeError):
                    return
        
        # Update the value
        self.value = value
        
        # Call the callback if provided
        if self.callback:
            self.callback(self.name, value)


class PropertyModel(QAbstractTableModel):
    """
    Model for property editor table view.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the property model.
        
        Args:
            parent: The parent widget
        """
        super(PropertyModel, self).__init__(parent)
        
        # Property items organized by category
        self.categories = {}  # Dict[str, List[PropertyItem]]
        self.all_properties = []  # List[PropertyItem]
        
        # Column definitions
        self.columns = ["Property", "Value"]
    
    def clear(self):
        """Clear all properties."""
        self.beginResetModel()
        self.categories = {}
        self.all_properties = []
        self.endResetModel()
    
    def add_property(self, prop: PropertyItem):
        """
        Add a property to the model.
        
        Args:
            prop (PropertyItem): The property to add
        """
        # Ensure category exists
        if prop.category not in self.categories:
            self.categories[prop.category] = []
        
        # Check if property already exists
        for i, existing_prop in enumerate(self.all_properties):
            if existing_prop.name == prop.name and existing_prop.category == prop.category:
                # Update existing property
                self.beginResetModel()
                self.all_properties[i] = prop
                self.categories[prop.category][self.categories[prop.category].index(existing_prop)] = prop
                self.endResetModel()
                return
        
        # Add new property
        self.beginInsertRows(QModelIndex(), len(self.all_properties), len(self.all_properties))
        self.all_properties.append(prop)
        self.categories[prop.category].append(prop)
        self.endInsertRows()
    
    def add_properties(self, props: List[PropertyItem]):
        """
        Add multiple properties to the model.
        
        Args:
            props (List[PropertyItem]): The properties to add
        """
        self.beginResetModel()
        for prop in props:
            # Ensure category exists
            if prop.category not in self.categories:
                self.categories[prop.category] = []
            
            # Check if property already exists
            existing_index = -1
            for i, existing_prop in enumerate(self.all_properties):
                if existing_prop.name == prop.name and existing_prop.category == prop.category:
                    existing_index = i
                    break
            
            if existing_index >= 0:
                # Update existing property
                self.all_properties[existing_index] = prop
                self.categories[prop.category][self.categories[prop.category].index(
                    self.all_properties[existing_index])] = prop
            else:
                # Add new property
                self.all_properties.append(prop)
                self.categories[prop.category].append(prop)
        self.endResetModel()
    
    def rowCount(self, parent=QModelIndex()):
        """
        Get the number of rows.
        
        Returns:
            int: The number of rows
        """
        if parent.isValid():
            return 0
        return len(self.all_properties)
    
    def columnCount(self, parent=QModelIndex()):
        """
        Get the number of columns.
        
        Returns:
            int: The number of columns
        """
        if parent.isValid():
            return 0
        return len(self.columns)
    
    def data(self, index, role=Qt.DisplayRole):
        """
        Get data for a given index and role.
        
        Args:
            index (QModelIndex): The index
            role (int): The role
            
        Returns:
            QVariant: The data
        """
        if not index.isValid() or index.row() >= len(self.all_properties):
            return QVariant()
        
        prop = self.all_properties[index.row()]
        
        if role == Qt.DisplayRole:
            # Display role - show property name or formatted value
            if index.column() == 0:
                return prop.name
            elif index.column() == 1:
                return prop.formatted_value()
        
        elif role == Qt.EditRole:
            # Edit role - raw value for editor
            if index.column() == 1 and not prop.read_only:
                return prop.value
        
        elif role == Qt.ToolTipRole:
            # Tooltip role - show description
            if prop.description:
                return prop.description
        
        elif role == Qt.BackgroundRole:
            # Background role - color for color properties
            if index.column() == 1 and prop.prop_type == PropertyItem.COLOR:
                if isinstance(prop.value, QColor):
                    return QBrush(prop.value)
                elif isinstance(prop.value, (list, tuple)) and len(prop.value) >= 3:
                    return QBrush(QColor(int(prop.value[0] * 255), 
                                        int(prop.value[1] * 255), 
                                        int(prop.value[2] * 255)))
        
        elif role == Qt.TextAlignmentRole:
            # Alignment role
            if index.column() == 0:
                return Qt.AlignLeft | Qt.AlignVCenter
            else:
                return Qt.AlignRight | Qt.AlignVCenter
        
        elif role == Qt.ForegroundRole:
            # Foreground role - gray out read-only properties
            if prop.read_only:
                return QBrush(QColor("#707070"))
        
        return QVariant()
    
    def setData(self, index, value, role=Qt.EditRole):
        """
        Set data for a given index and role.
        
        Args:
            index (QModelIndex): The index
            value: The value to set
            role (int): The role
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not index.isValid() or index.row() >= len(self.all_properties) or index.column() != 1:
            return False
        
        prop = self.all_properties[index.row()]
        
        if prop.read_only:
            return False
        
        if role == Qt.EditRole:
            # Update the property value
            prop.set_value(value)
            
            # Emit data changed signal
            self.dataChanged.emit(index, index)
            return True
        
        return False
    
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """
        Get header data.
        
        Args:
            section (int): The section
            orientation (Qt.Orientation): The orientation
            role (int): The role
            
        Returns:
            QVariant: The header data
        """
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            if 0 <= section < len(self.columns):
                return self.columns[section]
        
        return QVariant()
    
    def flags(self, index):
        """
        Get flags for a given index.
        
        Args:
            index (QModelIndex): The index
            
        Returns:
            Qt.ItemFlags: The flags
        """
        if not index.isValid():
            return Qt.NoItemFlags
        
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        
        # Only value column is editable, and only if not read-only
        if index.column() == 1 and not self.all_properties[index.row()].read_only:
            flags |= Qt.ItemIsEditable
        
        return flags
    
    def get_property(self, row: int) -> Optional[PropertyItem]:
        """
        Get the property at the specified row.
        
        Args:
            row (int): The row
            
        Returns:
            PropertyItem: The property, or None if not found
        """
        if 0 <= row < len(self.all_properties):
            return self.all_properties[row]
        return None
    
    def get_property_by_name(self, name: str, category: str = None) -> Optional[PropertyItem]:
        """
        Get a property by name and optional category.
        
        Args:
            name (str): The property name
            category (str, optional): The category
            
        Returns:
            PropertyItem: The property, or None if not found
        """
        for prop in self.all_properties:
            if prop.name == name and (category is None or prop.category == category):
                return prop
        return None
    
    def update_property(self, name: str, value: Any, category: str = None):
        """
        Update a property by name and optional category.
        
        Args:
            name (str): The property name
            value: The new value
            category (str, optional): The category
            
        Returns:
            bool: True if successful, False otherwise
        """
        for i, prop in enumerate(self.all_properties):
            if prop.name == name and (category is None or prop.category == category):
                prop.set_value(value)
                index = self.index(i, 1)
                self.dataChanged.emit(index, index)
                return True
        return False


class PropertyDelegate(QStyledItemDelegate):
    """
    Custom delegate for property editing.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the property delegate.
        
        Args:
            parent: The parent widget
        """
        super(PropertyDelegate, self).__init__(parent)
    
    def createEditor(self, parent, option, index):
        """
        Create an editor for a given index.
        
        Args:
            parent: The parent widget
            option: The style option
            index (QModelIndex): The index
            
        Returns:
            QWidget: The editor widget
        """
        if not index.isValid() or index.column() != 1:
            return super().createEditor(parent, option, index)
        
        model = index.model()
        if hasattr(model, 'sourceModel'):
            source_model = model.sourceModel()
            source_index = model.mapToSource(index)
        else:
            source_model = model
            source_index = index
        
        prop = source_model.get_property(source_index.row())
        if not prop or prop.read_only:
            return None
        
        if prop.prop_type == PropertyItem.STRING:
            # Text editor for strings
            editor = QLineEdit(parent)
            if prop.validator:
                editor.setValidator(prop.validator)
            return editor
        
        elif prop.prop_type == PropertyItem.INTEGER:
            # Spin box for integers
            editor = QSpinBox(parent)
            if prop.min_value is not None:
                editor.setMinimum(prop.min_value)
            else:
                editor.setMinimum(-1000000)
            
            if prop.max_value is not None:
                editor.setMaximum(prop.max_value)
            else:
                editor.setMaximum(1000000)
            
            return editor
        
        elif prop.prop_type == PropertyItem.FLOAT:
            # Double spin box for floats
            editor = QDoubleSpinBox(parent)
            editor.setDecimals(6)
            
            if prop.min_value is not None:
                editor.setMinimum(prop.min_value)
            else:
                editor.setMinimum(-1000000.0)
            
            if prop.max_value is not None:
                editor.setMaximum(prop.max_value)
            else:
                editor.setMaximum(1000000.0)
            
            return editor
        
        elif prop.prop_type == PropertyItem.BOOLEAN:
            # Combo box for booleans
            editor = QComboBox(parent)
            editor.addItems(["False", "True"])
            return editor
        
        elif prop.prop_type == PropertyItem.ENUM:
            # Combo box for enums
            editor = QComboBox(parent)
            editor.addItems(prop.options)
            return editor
        
        elif prop.prop_type == PropertyItem.COLOR:
            # Button to show color dialog
            editor = QToolButton(parent)
            editor.setText("...")
            editor.clicked.connect(lambda: self._show_color_dialog(editor, prop))
            return editor
        
        elif prop.prop_type == PropertyItem.VECTOR:
            # Line edit for vectors
            editor = QLineEdit(parent)
            return editor
        
        elif prop.prop_type == PropertyItem.FILE:
            # Button to show file dialog
            editor = QToolButton(parent)
            editor.setText("...")
            editor.clicked.connect(lambda: self._show_file_dialog(editor, prop))
            return editor
        
        return super().createEditor(parent, option, index)
    
    def setEditorData(self, editor, index):
        """
        Set editor data for a given index.
        
        Args:
            editor: The editor widget
            index (QModelIndex): The index
        """
        if not index.isValid() or index.column() != 1:
            super().setEditorData(editor, index)
            return
        
        model = index.model()
        if hasattr(model, 'sourceModel'):
            source_model = model.sourceModel()
            source_index = model.mapToSource(index)
        else:
            source_model = model
            source_index = index
        
        prop = source_model.get_property(source_index.row())
        if not prop:
            super().setEditorData(editor, index)
            return
        
        value = prop.value
        
        if prop.prop_type == PropertyItem.STRING:
            # Set text for string editor
            editor.setText(str(value) if value is not None else "")
        
        elif prop.prop_type == PropertyItem.INTEGER:
            # Set value for integer spin box
            editor.setValue(int(value) if value is not None else 0)
        
        elif prop.prop_type == PropertyItem.FLOAT:
            # Set value for float spin box
            editor.setValue(float(value) if value is not None else 0.0)
        
        elif prop.prop_type == PropertyItem.BOOLEAN:
            # Set index for boolean combo box
            editor.setCurrentIndex(1 if value else 0)
        
        elif prop.prop_type == PropertyItem.ENUM:
            # Set index for enum combo box
            if value in prop.options:
                editor.setCurrentIndex(prop.options.index(value))
            elif isinstance(value, int) and 0 <= value < len(prop.options):
                editor.setCurrentIndex(value)
            else:
                editor.setCurrentIndex(0)
        
        elif prop.prop_type == PropertyItem.COLOR:
            # Set background color for color button
            if isinstance(value, QColor):
                color = value
            elif isinstance(value, (list, tuple)) and len(value) >= 3:
                color = QColor(int(value[0] * 255), 
                              int(value[1] * 255), 
                              int(value[2] * 255))
            else:
                color = QColor(0, 0, 0)
            
            palette = editor.palette()
            palette.setColor(QPalette.Button, color)
            editor.setPalette(palette)
        
        elif prop.prop_type == PropertyItem.VECTOR:
            # Set text for vector editor
            if isinstance(value, (list, tuple)):
                editor.setText(", ".join(str(x) for x in value))
            else:
                editor.setText(str(value) if value is not None else "")
        
        elif prop.prop_type == PropertyItem.FILE:
            # No specific setup for file button
            pass
        
        else:
            super().setEditorData(editor, index)
    
    def setModelData(self, editor, model, index):
        """
        Set model data from editor for a given index.
        
        Args:
            editor: The editor widget
            model: The model
            index (QModelIndex): The index
        """
        if not index.isValid() or index.column() != 1:
            super().setModelData(editor, model, index)
            return
        
        if hasattr(model, 'sourceModel'):
            source_model = model.sourceModel()
            source_index = model.mapToSource(index)
        else:
            source_model = model
            source_index = index
        
        prop = source_model.get_property(source_index.row())
        if not prop:
            super().setModelData(editor, model, index)
            return
        
        if prop.prop_type == PropertyItem.STRING:
            # Get text from string editor
            model.setData(index, editor.text())
        
        elif prop.prop_type == PropertyItem.INTEGER:
            # Get value from integer spin box
            model.setData(index, editor.value())
        
        elif prop.prop_type == PropertyItem.FLOAT:
            # Get value from float spin box
            model.setData(index, editor.value())
        
        elif prop.prop_type == PropertyItem.BOOLEAN:
            # Get value from boolean combo box
            model.setData(index, editor.currentIndex() == 1)
        
        elif prop.prop_type == PropertyItem.ENUM:
            # Get value from enum combo box
            model.setData(index, prop.options[editor.currentIndex()])
        
        elif prop.prop_type == PropertyItem.COLOR:
            # Color is set in _show_color_dialog
            pass
        
        elif prop.prop_type == PropertyItem.VECTOR:
            # Get text from vector editor
            model.setData(index, editor.text())
        
        elif prop.prop_type == PropertyItem.FILE:
            # File is set in _show_file_dialog
            pass
        
        else:
            super().setModelData(editor, model, index)
    
    def updateEditorGeometry(self, editor, option, index):
        """
        Update editor geometry.
        
        Args:
            editor: The editor widget
            option: The style option
            index (QModelIndex): The index
        """
        editor.setGeometry(option.rect)
    
    def _show_color_dialog(self, button, prop):
        """
        Show color dialog for color properties.
        
        Args:
            button: The button that triggered the dialog
            prop (PropertyItem): The property
        """
        if not prop or prop.read_only:
            return
        
        # Get current color
        current_color = None
        if isinstance(prop.value, QColor):
            current_color = prop.value
        elif isinstance(prop.value, (list, tuple)) and len(prop.value) >= 3:
            current_color = QColor(int(prop.value[0] * 255), 
                                  int(prop.value[1] * 255), 
                                  int(prop.value[2] * 255))
        
        # Show color dialog
        color = QColorDialog.getColor(current_color or QColor(0, 0, 0), button.parent())
        if color.isValid():
            # Update property
            prop.set_value(color)
            
            # Update button color
            palette = button.palette()
            palette.setColor(QPalette.Button, color)
            button.setPalette(palette)
            
            # Notify model
            model = self.parent().model()
            for row, p in enumerate(model.all_properties):
                if p is prop:
                    index = model.index(row, 1)
                    model.dataChanged.emit(index, index)
                    break
    
    def _show_file_dialog(self, button, prop):
        """
        Show file dialog for file properties.
        
        Args:
            button: The button that triggered the dialog
            prop (PropertyItem): The property
        """
        if not prop or prop.read_only:
            return
        
        # Show file dialog
        filepath, _ = QFileDialog.getOpenFileName(
            button.parent(),
            "Select File",
            os.path.dirname(prop.value) if prop.value else "",
            "All Files (*)"
        )
        
        if filepath:
            # Update property
            prop.set_value(filepath)
            
            # Notify model
            model = self.parent().model()
            for row, p in enumerate(model.all_properties):
                if p is prop:
                    index = model.index(row, 1)
                    model.dataChanged.emit(index, index)
                    break


class PropertyEditor(QWidget):
    """
    Property editor widget.
    
    This widget displays and allows editing of properties for the selected item
    in the project.
    """
    
    # Signals
    property_changed = pyqtSignal(str, object)
    
    def __init__(self, parent=None):
        """
        Initialize the property editor.
        
        Args:
            parent: The parent widget
        """
        super(PropertyEditor, self).__init__(parent)
        
        # Store reference to main window
        self.main_window = parent
        
        # Initialize instance variables
        self.current_item = None
        self.model = PropertyModel(self)
        self.proxy_model = QSortFilterProxyModel(self)
        self.proxy_model.setSourceModel(self.model)
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Toolbar with common actions
        self.toolbar = QToolBar()
        self.toolbar.setIconSize(QSize(16, 16))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.layout.addWidget(self.toolbar)
        
        # Search box
        self.search_layout = QHBoxLayout()
        self.search_layout.setContentsMargins(4, 4, 4, 4)
        self.search_layout.setSpacing(4)
        
        self.search_label = QLabel("Filter:")
        self.search_layout.addWidget(self.search_label)
        
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search properties...")
        self.search_layout.addWidget(self.search_box)
        
        self.layout.addLayout(self.search_layout)
        
        # Table view
        self.table_view = QTreeView()
        self.table_view.setRootIsDecorated(False)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_view.setEditTriggers(QAbstractItemView.DoubleClicked | 
                                        QAbstractItemView.EditKeyPressed | 
                                        QAbstractItemView.AnyKeyPressed)
        self.table_view.setModel(self.proxy_model)
        
        # Set up header
        header = self.table_view.header()
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setStretchLastSection(True)
        
        # Set up delegate
        self.delegate = PropertyDelegate(self.table_view)
        self.table_view.setItemDelegate(self.delegate)
        
        self.layout.addWidget(self.table_view)
        
        # Category filter
        self.category_frame = QFrame()
        self.category_layout = QHBoxLayout(self.category_frame)
        self.category_layout.setContentsMargins(4, 4, 4, 4)
        self.category_layout.setSpacing(4)
        
        self.category_label = QLabel("Category:")
        self.category_layout.addWidget(self.category_label)
        
        self.category_combo = QComboBox()
        self.category_combo.addItem("All")
        self.category_layout.addWidget(self.category_combo)
        
        self.layout.addWidget(self.category_frame)
        
        # Add actions to toolbar
        self.action_reset = QAction(self._get_icon("reset"), "Reset", self)
        self.action_reset.setToolTip("Reset to default values")
        self.toolbar.addAction(self.action_reset)
        
        self.action_expand_all = QAction(self._get_icon("expand"), "Expand All", self)
        self.action_expand_all.setToolTip("Expand all categories")
        self.toolbar.addAction(self.action_expand_all)
        
        self.action_collapse_all = QAction(self._get_icon("collapse"), "Collapse All", self)
        self.action_collapse_all.setToolTip("Collapse all categories")
        self.toolbar.addAction(self.action_collapse_all)
    
    def _connect_signals(self):
        """Connect signals to slots."""
        # Connect search box
        self.search_box.textChanged.connect(self._filter_changed)
        
        # Connect category filter
        self.category_combo.currentTextChanged.connect(self._category_changed)
        
        # Connect actions
        self.action_reset.triggered.connect(self._reset_properties)
        self.action_expand_all.triggered.connect(self.table_view.expandAll)
        self.action_collapse_all.triggered.connect(self.table_view.collapseAll)
    
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
    
    def _filter_changed(self, text: str):
        """
        Update the filter when search text changes.
        
        Args:
            text (str): The new filter text
        """
        self.proxy_model.setFilterRegExp(QRegExp(text, Qt.CaseInsensitive))
    
    def _category_changed(self, category: str):
        """
        Update the filter when category changes.
        
        Args:
            category (str): The new category
        """
        if category == "All":
            # Show all categories
            self.proxy_model.setFilterRegExp("")
        else:
            # Filter by category
            # This would require a custom proxy model that filters on category
            # For now, we can just use the search box to filter by category
            self.search_box.setText(category)
    
    def _reset_properties(self):
        """Reset properties to default values."""
        # This would depend on the specific implementation
        pass
    
    def clear(self):
        """Clear all properties."""
        self.current_item = None
        self.model.clear()
        self.category_combo.clear()
        self.category_combo.addItem("All")
    
    def set_item(self, item):
        """
        Set the current item and update properties.
        
        Args:
            item: The item to display properties for
        """
        if item == self.current_item:
            return
        
        self.current_item = item
        
        # Clear existing properties
        self.model.clear()
        
        # Update properties based on item type
        if hasattr(item, 'item_type'):
            self._load_item_properties(item)
        else:
            # Generic item with no specific type
            self._load_generic_properties(item)
        
        # Update category filter
        self.category_combo.clear()
        self.category_combo.addItem("All")
        for category in sorted(self.model.categories.keys()):
            self.category_combo.addItem(category)
    
    def _load_item_properties(self, item):
        """
        Load properties for a specific item type.
        
        Args:
            item: The item to load properties for
        """
        # Project item with type information
        if item.item_type == ProjectItem.GEOMETRY:
            self._load_geometry_properties(item)
        elif item.item_type == ProjectItem.MESH:
            self._load_mesh_properties(item)
        elif item.item_type == ProjectItem.CASE:
            self._load_case_properties(item)
        elif item.item_type == ProjectItem.BOUNDARIES:
            self._load_boundary_properties(item)
        elif item.item_type == ProjectItem.RESULTS:
            self._load_results_properties(item)
        elif item.item_type == ProjectItem.VISUALIZATION:
            self._load_visualization_properties(item)
        elif item.item_type == ProjectItem.PROJECT:
            self._load_project_properties(item)
        else:
            # Default to file properties
            self._load_file_properties(item)
    
    def _load_generic_properties(self, item):
        """
        Load properties for a generic item.
        
        Args:
            item: The item to load properties for
        """
        # Add basic properties
        if hasattr(item, 'name'):
            self.model.add_property(PropertyItem(
                "Name", item.name, PropertyItem.STRING, "General", True))
        
        if hasattr(item, 'filepath') and item.filepath:
            self.model.add_property(PropertyItem(
                "File Path", item.filepath, PropertyItem.FILE, "General", True))
    
    def _load_file_properties(self, item):
        """
        Load properties for a file item.
        
        Args:
            item: The file item to load properties for
        """
        # Add basic file properties
        self.model.add_property(PropertyItem(
            "Name", item.text(), PropertyItem.STRING, "General", True))
        
        if hasattr(item, 'filepath') and item.filepath:
            self.model.add_property(PropertyItem(
                "File Path", item.filepath, PropertyItem.FILE, "General", True))
            
            # Add file metadata
            if os.path.exists(item.filepath):
                stat = os.stat(item.filepath)
                self.model.add_property(PropertyItem(
                    "Size", f"{stat.st_size / 1024:.2f} KB", PropertyItem.STRING, "File Info", True))
                
                import datetime
                modified_time = datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                self.model.add_property(PropertyItem(
                    "Modified", modified_time, PropertyItem.STRING, "File Info", True))
                
                created_time = datetime.datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
                self.model.add_property(PropertyItem(
                    "Created", created_time, PropertyItem.STRING, "File Info", True))
    
    def _load_geometry_properties(self, item):
        """
        Load properties for a geometry item.
        
        Args:
            item: The geometry item to load properties for
        """
        # Load basic file properties
        self._load_file_properties(item)
        
        # Add geometry-specific properties
        # These would be populated from the actual geometry data
        
        # Example properties
        self.model.add_property(PropertyItem(
            "Type", "STL", PropertyItem.STRING, "Geometry", True))
        
        self.model.add_property(PropertyItem(
            "Vertices", 1024, PropertyItem.INTEGER, "Geometry", True))
        
        self.model.add_property(PropertyItem(
            "Faces", 2048, PropertyItem.INTEGER, "Geometry", True))
        
        self.model.add_property(PropertyItem(
            "Normals", "Computed", PropertyItem.STRING, "Geometry", True))
        
        self.model.add_property(PropertyItem(
            "Volume", 1.25, PropertyItem.FLOAT, "Dimensions", True, unit="m³"))
        
        self.model.add_property(PropertyItem(
            "Surface Area", 10.5, PropertyItem.FLOAT, "Dimensions", True, unit="m²"))
        
        self.model.add_property(PropertyItem(
            "Bounding Box", [0, 0, 0, 10, 10, 10], PropertyItem.VECTOR, "Dimensions", True))
    
    def _load_mesh_properties(self, item):
        """
        Load properties for a mesh item.
        
        Args:
            item: The mesh item to load properties for
        """
        # Load basic file properties
        self._load_file_properties(item)
        
        # Add mesh-specific properties
        # These would be populated from the actual mesh data
        
        # Example properties
        self.model.add_property(PropertyItem(
            "Type", "Block Mesh", PropertyItem.STRING, "Mesh", True))
        
        self.model.add_property(PropertyItem(
            "Cells", 100000, PropertyItem.INTEGER, "Mesh", True))
        
        self.model.add_property(PropertyItem(
            "Points", 120000, PropertyItem.INTEGER, "Mesh", True))
        
        self.model.add_property(PropertyItem(
            "Faces", 350000, PropertyItem.INTEGER, "Mesh", True))
        
        self.model.add_property(PropertyItem(
            "Boundaries", 6, PropertyItem.INTEGER, "Mesh", True))
        
        self.model.add_property(PropertyItem(
            "Min Quality", 0.85, PropertyItem.FLOAT, "Quality", True))
        
        self.model.add_property(PropertyItem(
            "Max Skewness", 0.25, PropertyItem.FLOAT, "Quality", True))
        
        self.model.add_property(PropertyItem(
            "Min Volume", 1e-6, PropertyItem.FLOAT, "Quality", True, unit="m³"))
    
    def _load_case_properties(self, item):
        """
        Load properties for a case item.
        
        Args:
            item: The case item to load properties for
        """
        # Load basic file properties
        self._load_file_properties(item)
        
        # Add case-specific properties
        self.model.add_property(PropertyItem(
            "Solver", "simpleFoam", PropertyItem.ENUM, "Solver", False,
            options=["simpleFoam", "pisoFoam", "interFoam", "reactingFoam", "multiphaseInterFoam"]))
        
        self.model.add_property(PropertyItem(
            "Turbulence Model", "kEpsilon", PropertyItem.ENUM, "Solver", False,
            options=["kEpsilon", "kOmega", "SpalartAllmaras", "LES", "laminar"]))
        
        self.model.add_property(PropertyItem(
            "Start Time", 0.0, PropertyItem.FLOAT, "Time", False, unit="s"))
        
        self.model.add_property(PropertyItem(
            "End Time", 100.0, PropertyItem.FLOAT, "Time", False, unit="s"))
        
        self.model.add_property(PropertyItem(
            "Time Step", 0.001, PropertyItem.FLOAT, "Time", False, unit="s"))
        
        self.model.add_property(PropertyItem(
            "Write Interval", 10, PropertyItem.INTEGER, "Output", False))
        
        self.model.add_property(PropertyItem(
            "Write Format", "binary", PropertyItem.ENUM, "Output", False,
            options=["binary", "ascii"]))
    
    def _load_boundary_properties(self, item):
        """
        Load properties for a boundary condition item.
        
        Args:
            item: The boundary condition item to load properties for
        """
        # Load basic file properties
        self._load_file_properties(item)
        
        # Add boundary-specific properties
        self.model.add_property(PropertyItem(
            "Type", "wall", PropertyItem.ENUM, "Boundary", False,
            options=["wall", "inlet", "outlet", "symmetry", "patch"]))
        
        self.model.add_property(PropertyItem(
            "Pressure", 101325, PropertyItem.FLOAT, "Conditions", False, unit="Pa"))
        
        self.model.add_property(PropertyItem(
            "Velocity", [0, 0, 0], PropertyItem.VECTOR, "Conditions", False))
        
        self.model.add_property(PropertyItem(
            "Temperature", 300, PropertyItem.FLOAT, "Conditions", False, unit="K"))
        
        self.model.add_property(PropertyItem(
            "No Slip", True, PropertyItem.BOOLEAN, "Conditions", False))
    
    def _load_results_properties(self, item):
        """
        Load properties for a results item.
        
        Args:
            item: The results item to load properties for
        """
        # Load basic file properties
        self._load_file_properties(item)
        
        # Add results-specific properties
        self.model.add_property(PropertyItem(
            "Time Steps", 100, PropertyItem.INTEGER, "Results", True))
        
        self.model.add_property(PropertyItem(
            "Variables", ["p", "U", "T", "k", "epsilon"], PropertyItem.STRING, "Results", True))
        
        self.model.add_property(PropertyItem(
            "Converged", True, PropertyItem.BOOLEAN, "Results", True))
        
        self.model.add_property(PropertyItem(
            "Max Residual", 1e-6, PropertyItem.FLOAT, "Convergence", True))
        
        self.model.add_property(PropertyItem(
            "Max Courant Number", 0.95, PropertyItem.FLOAT, "Convergence", True))
    
    def _load_visualization_properties(self, item):
        """
        Load properties for a visualization item.
        
        Args:
            item: The visualization item to load properties for
        """
        # Load basic file properties
        self._load_file_properties(item)
        
        # Add visualization-specific properties
        self.model.add_property(PropertyItem(
            "Representation", "Surface", PropertyItem.ENUM, "Display", False,
            options=["Surface", "Wireframe", "Points", "Volume"]))
        
        self.model.add_property(PropertyItem(
            "Coloring", "Pressure", PropertyItem.ENUM, "Display", False,
            options=["Pressure", "Velocity", "Temperature", "Solid Color"]))
        
        self.model.add_property(PropertyItem(
            "Color Map", "Rainbow", PropertyItem.ENUM, "Display", False,
            options=["Rainbow", "Cool to Warm", "Viridis", "Plasma", "Jet"]))
        
        self.model.add_property(PropertyItem(
            "Background Color", [0.2, 0.2, 0.2], PropertyItem.COLOR, "Display", False))
        
        self.model.add_property(PropertyItem(
            "Show Axes", True, PropertyItem.BOOLEAN, "Display", False))
        
        self.model.add_property(PropertyItem(
            "Show Color Bar", True, PropertyItem.BOOLEAN, "Display", False))
    
    def _load_project_properties(self, item):
        """
        Load properties for a project item.
        
        Args:
            item: The project item to load properties for
        """
        # Load basic file properties
        self._load_file_properties(item)
        
        # Add project-specific properties
        self.model.add_property(PropertyItem(
            "Created", "2025-01-01", PropertyItem.STRING, "Project", True))
        
        self.model.add_property(PropertyItem(
            "Modified", "2025-01-15", PropertyItem.STRING, "Project", True))
        
        self.model.add_property(PropertyItem(
            "Version", "1.0", PropertyItem.STRING, "Project", True))
        
        self.model.add_property(PropertyItem(
            "Author", "User", PropertyItem.STRING, "Project", False))
        
        self.model.add_property(PropertyItem(
            "Description", "Oil and gas CFD project", PropertyItem.STRING, "Project", False))
        
        # Industry-specific properties
        self.model.add_property(PropertyItem(
            "Units", "metric", PropertyItem.ENUM, "Industry", False,
            options=["metric", "imperial", "field"]))
        
        self.model.add_property(PropertyItem(
            "Fluid Type", "Oil", PropertyItem.ENUM, "Industry", False,
            options=["Oil", "Water", "Gas", "Multiphase"]))
        
        self.model.add_property(PropertyItem(
            "Viscosity", 0.03, PropertyItem.FLOAT, "Fluid Properties", False, unit="Pa·s"))
        
        self.model.add_property(PropertyItem(
            "Density", 850, PropertyItem.FLOAT, "Fluid Properties", False, unit="kg/m³"))