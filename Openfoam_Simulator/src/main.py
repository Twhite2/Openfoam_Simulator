#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Openfoam_Simulator - Oil & Gas CFD Visualization System
Main application entry point
"""

import sys
import os
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QCoreApplication, Qt
    import vtk
    import vtkmodules.vtkRenderingOpenGL2
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError as e:
    print(f"Critical import error: {e}")
    print("Please make sure all dependencies are installed")
    sys.exit(1)

# Import Openfoam_Simulator modules
try:
    from src.gui.main_window import MainWindow
    from src.utils.logger import setup_logger
    from src.config import load_config
except ImportError as e:
    print(f"Openfoam_Simulator module import error: {e}")
    print("Please make sure the application is properly installed")
    sys.exit(1)

def check_environment():
    """Check if all required environment variables and dependencies are available"""
    # Check OpenFOAM environment
    if "WM_PROJECT_DIR" not in os.environ:
        logging.warning("OpenFOAM environment not detected. Some features may not work correctly.")
    
    # Check VTK version
    vtk_version = vtk.vtkVersion.GetVTKVersion()
    logging.info(f"VTK version: {vtk_version}")
    
    # Check for VTK rendering capabilities
    try:
        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        
        # Check if OpenGL backend is available
        renderWindow.SetOffScreenRendering(1)
        if renderWindow.SupportsOpenGL():
            logging.info("VTK OpenGL rendering is available")
        else:
            logging.warning("VTK OpenGL rendering may not be fully supported")
    except Exception as e:
        logging.warning(f"Error testing VTK rendering: {e}")
    
    # Check for GPU
    vtkOpenGLRW = vtk.vtkRenderWindow()
    if hasattr(vtkOpenGLRW, "ReportCapabilities"):
        logging.info("VTK OpenGL capabilities available")
    else:
        logging.warning("VTK may not have full OpenGL support")

def main():
    """Main application entry point"""
    # Setup logger
    setup_logger()
    
    # Load configuration
    config = load_config()
    
    # Enable high DPI scaling
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Openfoam_Simulator")
    app.setOrganizationName("Openfoam_Simulator Team")
    
    # Check environment
    check_environment()
    
    # Create main window
    main_window = MainWindow(config)
    main_window.show()
    
    # Enter Qt event loop
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())