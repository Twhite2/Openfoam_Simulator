# Openfoam_Simulator

A comprehensive CFD visualization and simulation system for OpenFOAM, designed for oil & gas industry applications.

## Overview

Openfoam_Simulator provides an intuitive interface for setting up, running, and visualizing OpenFOAM simulations. The application integrates OpenFOAM's powerful computational fluid dynamics capabilities with advanced visualization tools based on VTK and ParaView.

## Features

- Direct integration with OpenFOAM for CFD simulations
- Advanced 3D visualization of simulation results
- Support for complex geometry and mesh handling
- Customizable boundary conditions and material properties
- Streamlined workflow for oil & gas industry simulations
- Simulation state persistence and project management
- ParaView integration for advanced post-processing

## Requirements

- OpenFOAM (environment variable WM_PROJECT_DIR must be set)
- Python 3.7+
- PyQt5
- VTK with OpenGL2 backend
- ParaView (for advanced visualization features)

## Installation

Clone the repository:

```bash
git clone https://github.com/Twhite2/Openfoam_Simulator.git
cd Openfoam_Simulator
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python src/main.py
```

Or use Docker:

```bash
docker build -t openfoam_simulator -f docker/Dockerfile.fixed .
docker run -it --rm -p 6080:6080 openfoam_simulator
```

Then access the application through a web browser at http://localhost:6080

## License

© 2025 Openfoam_Simulator Team
