"""
VTK utility functions for Openfoam_Simulator.
Provides helper functions for working with VTK datasets.
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from ..utils.logger import get_logger

# Set up logger
logger = get_logger(__name__)

# Check if VTK is available
try:
    import vtk
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    logger.error("VTK not available - visualization features will be disabled")

# Disable checking for numpy array memory layout - needed for some VTK operations
vtk.vtkObject.GlobalWarningDisplayOff()

def read_vtk_file(filepath: str) -> Optional[vtk.vtkDataSet]:
    """
    Read a VTK file and return the dataset.
    
    Args:
        filepath (str): Path to VTK file
        
    Returns:
        vtk.vtkDataSet: VTK dataset or None if error
    """
    if not VTK_AVAILABLE:
        logger.error("VTK not available")
        return None
    
    try:
        # Check if file exists
        if not os.path.isfile(filepath):
            logger.error(f"File not found: {filepath}")
            return None
        
        # Determine file type and reader
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        
        if ext == '.vtk':
            reader = vtk.vtkDataSetReader()
        elif ext == '.vtp':
            reader = vtk.vtkXMLPolyDataReader()
        elif ext == '.vtu':
            reader = vtk.vtkXMLUnstructuredGridReader()
        elif ext == '.vts':
            reader = vtk.vtkXMLStructuredGridReader()
        elif ext == '.vtr':
            reader = vtk.vtkXMLRectilinearGridReader()
        elif ext == '.vti':
            reader = vtk.vtkXMLImageDataReader()
        elif ext == '.foam' or ext == '.openfoam':
            reader = vtk.vtkOpenFOAMReader()
        else:
            logger.error(f"Unsupported file format: {ext}")
            return None
        
        # Read file
        reader.SetFileName(filepath)
        reader.Update()
        
        return reader.GetOutput()
    except Exception as e:
        logger.error(f"Error reading VTK file: {e}")
        return None

def extract_block(multiblock_dataset, block_index: int) -> Optional[vtk.vtkDataSet]:
    """
    Extract a block from a multiblock dataset.
    
    Args:
        multiblock_dataset: VTK multiblock dataset
        block_index (int): Index of the block to extract
        
    Returns:
        vtk.vtkDataSet: Extracted block
    """
    if not VTK_AVAILABLE:
        logger.error("VTK not available")
        return None
    
    try:
        if not isinstance(multiblock_dataset, vtk.vtkMultiBlockDataSet):
            logger.warning("Input is not a multiblock dataset")
            return None
        
        if block_index >= multiblock_dataset.GetNumberOfBlocks():
            logger.warning(f"Block index {block_index} out of range")
            return None
        
        return multiblock_dataset.GetBlock(block_index)
    except Exception as e:
        logger.error(f"Error extracting block: {e}")
        return None

def compute_contour(dataset, field_name: str, values: List[float]) -> Optional[vtk.vtkPolyData]:
    """
    Compute contours of a scalar field.
    
    Args:
        dataset: VTK dataset
        field_name (str): Name of the scalar field
        values (List[float]): List of contour values
        
    Returns:
        vtk.vtkPolyData: Contour as polydata
    """
    if not VTK_AVAILABLE:
        logger.error("VTK not available")
        return None
    
    try:
        # Check if field exists
        point_data = dataset.GetPointData()
        if not point_data.GetArray(field_name):
            logger.warning(f"Field {field_name} not found in point data")
            return None
        
        # Create contour filter
        contour = vtk.vtkContourFilter()
        contour.SetInputData(dataset)
        contour.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, field_name)
        
        # Add contour values
        for i, value in enumerate(values):
            contour.SetValue(i, value)
        
        contour.Update()
        
        return contour.GetOutput()
    except Exception as e:
        logger.error(f"Error computing contour: {e}")
        return None

def slice_with_plane(dataset, origin: List[float], normal: List[float]) -> Optional[vtk.vtkPolyData]:
    """
    Create a slice through a dataset using a plane.
    
    Args:
        dataset: VTK dataset
        origin (List[float]): Origin point of the slice plane
        normal (List[float]): Normal vector of the slice plane
        
    Returns:
        vtk.vtkPolyData: Slice as polydata
    """
    if not VTK_AVAILABLE:
        logger.error("VTK not available")
        return None
    
    try:
        # Create a plane
        plane = vtk.vtkPlane()
        plane.SetOrigin(origin[0], origin[1], origin[2])
        plane.SetNormal(normal[0], normal[1], normal[2])
        
        # Create cutter
        cutter = vtk.vtkCutter()
        cutter.SetInputData(dataset)
        cutter.SetCutFunction(plane)
        cutter.Update()
        
        return cutter.GetOutput()
    except Exception as e:
        logger.error(f"Error creating slice: {e}")
        return None

def compute_streamlines(dataset, vector_field: str, seeds: np.ndarray,
                     max_time: float = 100.0, step_size: float = 0.1) -> Optional[vtk.vtkPolyData]:
    """
    Compute streamlines from a vector field.
    
    Args:
        dataset: VTK dataset
        vector_field (str): Name of the vector field
        seeds (np.ndarray): Array of seed points with shape (n, 3)
        max_time (float): Maximum streamline integration time
        step_size (float): Integration step size
        
    Returns:
        vtk.vtkPolyData: Streamlines as polydata
    """
    if not VTK_AVAILABLE:
        logger.error("VTK not available")
        return None
    
    try:
        # Check if field exists
        point_data = dataset.GetPointData()
        if not point_data.GetArray(vector_field):
            logger.warning(f"Vector field {vector_field} not found in point data")
            return None
        
        # Create seed points
        points = vtk.vtkPoints()
        for seed in seeds:
            points.InsertNextPoint(seed[0], seed[1], seed[2])
        
        seed_data = vtk.vtkPolyData()
        seed_data.SetPoints(points)
        
        # Create streamline filter
        streamlines = vtk.vtkStreamTracer()
        streamlines.SetInputData(dataset)
        streamlines.SetSourceData(seed_data)
        streamlines.SetMaximumPropagation(max_time)
        streamlines.SetInitialIntegrationStep(step_size)
        streamlines.SetIntegrationDirectionToBoth()
        streamlines.SetIntegratorTypeToRungeKutta45()
        streamlines.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, vector_field)
        streamlines.Update()
        
        return streamlines.GetOutput()
    except Exception as e:
        logger.error(f"Error computing streamlines: {e}")
        return None


def create_glyphs(dataset, vector_field: str, scale_factor: float = 1.0, 
                mask_pts: int = 50) -> Optional[vtk.vtkPolyData]:
    """
    Create glyphs from a vector field.
    
    Args:
        dataset: VTK dataset
        vector_field (str): Name of the vector field
        scale_factor (float): Scaling factor for the glyphs
        mask_pts (int): Use only 1 out of every mask_pts points
        
    Returns:
        vtk.vtkPolyData: Glyph dataset
    """
    if not VTK_AVAILABLE:
        logger.error("VTK not available")
        return None
    
    try:
        # Check if field exists
        point_data = dataset.GetPointData()
        if not point_data.GetArray(vector_field):
            logger.warning(f"Vector field {vector_field} not found in point data")
            return None
        
        # Create mask points filter to reduce number of glyphs
        mask_points = vtk.vtkMaskPoints()
        mask_points.SetInputData(dataset)
        mask_points.SetOnRatio(mask_pts)
        mask_points.RandomModeOn()
        mask_points.Update()
        
        # Create arrow source
        arrow = vtk.vtkArrowSource()
        arrow.SetTipResolution(16)
        arrow.SetShaftResolution(16)
        arrow.SetTipLength(0.3)
        arrow.SetTipRadius(0.1)
        arrow.SetShaftRadius(0.03)
        arrow.Update()
        
        # Create glyph filter
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(mask_points.GetOutput())
        glyph.SetSourceData(arrow.GetOutput())
        glyph.SetVectorModeToUseVector()
        glyph.SetScaleModeToScaleByVector()
        glyph.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, vector_field)
        glyph.SetScaleFactor(scale_factor)
        glyph.OrientOn()
        glyph.Update()
        
        return glyph.GetOutput()
    except Exception as e:
        logger.error(f"Error creating glyphs: {e}")
        return None


def compute_phase_interface(dataset, phase_field: str, threshold: float = 0.5) -> Optional[vtk.vtkPolyData]:
    """
    Compute the interface between phases in a multiphase flow.
    
    Args:
        dataset: VTK dataset
        phase_field (str): Name of the phase fraction field (e.g., 'alpha.water')
        threshold (float): Phase fraction threshold for the interface
        
    Returns:
        vtk.vtkPolyData: Interface surface
    """
    if not VTK_AVAILABLE:
        logger.error("VTK not available")
        return None
    
    try:
        # Use contour filter to extract the interface at the specified threshold
        return compute_contour(dataset, phase_field, [threshold])
    except Exception as e:
        logger.error(f"Error computing phase interface: {e}")
        return None


def compute_interface_area(interface_surface) -> float:
    """
    Compute the area of an interface surface.
    
    Args:
        interface_surface: VTK polydata representing the interface
        
    Returns:
        float: Interface area
    """
    if not VTK_AVAILABLE:
        logger.error("VTK not available")
        return 0.0
    
    try:
        # Create mass properties filter
        mass_properties = vtk.vtkMassProperties()
        mass_properties.SetInputData(interface_surface)
        mass_properties.Update()
        
        return mass_properties.GetSurfaceArea()
    except Exception as e:
        logger.error(f"Error computing interface area: {e}")
        return 0.0


def compute_field_on_slice(dataset, slice_origin: List[float], slice_normal: List[float], 
                         field_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a field on a slice through the dataset.
    
    Args:
        dataset: VTK dataset
        slice_origin (List[float]): Origin point of the slice
        slice_normal (List[float]): Normal vector of the slice
        field_name (str): Name of the field to extract
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Points and field values on the slice
    """
    if not VTK_AVAILABLE:
        logger.error("VTK not available")
        return np.array([]), np.array([])
    
    try:
        # Create slice
        slice_data = slice_with_plane(dataset, slice_origin, slice_normal)
        
        if slice_data is None:
            return np.array([]), np.array([])
        
        # Extract points
        points = vtk_to_numpy(slice_data.GetPoints().GetData())
        
        # Extract field data
        point_data = slice_data.GetPointData()
        field_array = point_data.GetArray(field_name)
        
        if field_array is None:
            logger.warning(f"Field {field_name} not found on slice")
            return points, np.array([])
        
        field_values = vtk_to_numpy(field_array)
        
        return points, field_values
    except Exception as e:
        logger.error(f"Error computing field on slice: {e}")
        return np.array([]), np.array([])


def resample_to_image(dataset, dimensions: List[int] = [50, 50, 50]) -> Optional[vtk.vtkImageData]:
    """
    Resample a dataset to a regular grid (vtkImageData).
    
    Args:
        dataset: VTK dataset
        dimensions (List[int]): Dimensions of the output image [nx, ny, nz]
        
    Returns:
        vtk.vtkImageData: Resampled image data
    """
    if not VTK_AVAILABLE:
        logger.error("VTK not available")
        return None
    
    try:
        # Get dataset bounds
        bounds = dataset.GetBounds()
        
        # Create resampling filter
        resample = vtk.vtkResampleToImage()
        resample.SetInputData(dataset)
        resample.SetSamplingDimensions(dimensions)
        resample.SetUseInputBounds(True)
        resample.Update()
        
        return resample.GetOutput()
    except Exception as e:
        logger.error(f"Error resampling to image: {e}")
        return None


def extract_block_by_name(multiblock_dataset, block_name: str) -> Optional[vtk.vtkDataSet]:
    """
    Extract a block from a multiblock dataset by name.
    
    Args:
        multiblock_dataset: VTK multiblock dataset
        block_name (str): Name of the block to extract
        
    Returns:
        vtk.vtkDataSet: Extracted block
    """
    if not VTK_AVAILABLE:
        logger.error("VTK not available")
        return None
    
    try:
        if not isinstance(multiblock_dataset, vtk.vtkMultiBlockDataSet):
            logger.warning("Input is not a multiblock dataset")
            return None
        
        # Iterate through blocks to find by name
        for i in range(multiblock_dataset.GetNumberOfBlocks()):
            if multiblock_dataset.GetMetaData(i) and multiblock_dataset.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME()):
                name = multiblock_dataset.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME())
                if name == block_name:
                    return multiblock_dataset.GetBlock(i)
        
        logger.warning(f"Block '{block_name}' not found")
        return None
    except Exception as e:
        logger.error(f"Error extracting block by name: {e}")
        return None


def extract_boundary(dataset, boundary_name: str) -> Optional[vtk.vtkPolyData]:
    """
    Extract a specific boundary from an OpenFOAM dataset.
    
    Args:
        dataset: VTK multiblock dataset from OpenFOAM
        boundary_name (str): Name of the boundary to extract
        
    Returns:
        vtk.vtkPolyData: Extracted boundary
    """
    if not VTK_AVAILABLE:
        logger.error("VTK not available")
        return None
    
    try:
        # Extract boundary block (typically in block 1 for OpenFOAM readers)
        if not isinstance(dataset, vtk.vtkMultiBlockDataSet) or dataset.GetNumberOfBlocks() < 2:
            logger.warning("Input is not a valid OpenFOAM dataset")
            return None
        
        boundaries = dataset.GetBlock(1)
        if not boundaries:
            logger.warning("No boundaries found in dataset")
            return None
        
        # Extract the specific boundary by name
        return extract_block_by_name(boundaries, boundary_name)
    except Exception as e:
        logger.error(f"Error extracting boundary: {e}")
        return None


def compute_flow_rate(dataset, boundary_name: str, velocity_field: str = 'U') -> float:
    """
    Compute the flow rate through a boundary.
    
    Args:
        dataset: VTK multiblock dataset from OpenFOAM
        boundary_name (str): Name of the boundary to compute flow rate through
        velocity_field (str): Name of the velocity field
        
    Returns:
        float: Flow rate through the boundary
    """
    if not VTK_AVAILABLE:
        logger.error("VTK not available")
        return 0.0
    
    try:
        # Extract the boundary
        boundary = extract_boundary(dataset, boundary_name)
        if boundary is None:
            return 0.0
        
        # Check for velocity field
        point_data = boundary.GetPointData()
        velocity_array = point_data.GetArray(velocity_field)
        
        if velocity_array is None:
            logger.warning(f"Velocity field '{velocity_field}' not found on boundary")
            return 0.0
        
        # Compute surface normals
        normal_generator = vtk.vtkPolyDataNormals()
        normal_generator.SetInputData(boundary)
        normal_generator.ComputeCellNormalsOn()
        normal_generator.Update()
        
        boundary_with_normals = normal_generator.GetOutput()
        
        # Get cell normals and velocities
        cell_normals = vtk_to_numpy(boundary_with_normals.GetCellData().GetNormals())
        
        # Integrate velocity over the boundary
        flow_rate = 0.0
        
        # Create cell centers filter to get velocity at cell centers
        cell_centers = vtk.vtkCellCenters()
        cell_centers.SetInputData(boundary_with_normals)
        cell_centers.Update()
        
        centers_with_velocity = cell_centers.GetOutput()
        
        # Get velocity at cell centers
        velocities = vtk_to_numpy(centers_with_velocity.GetPointData().GetArray(velocity_field))
        
        # Compute cell areas
        area_filter = vtk.vtkMeshQuality()
        area_filter.SetInputData(boundary_with_normals)
        area_filter.SetTriangleQualityMeasureToArea()
        area_filter.Update()
        
        areas = vtk_to_numpy(area_filter.GetOutput().GetCellData().GetArray("Quality"))
        
        # Calculate flow rate as sum of (velocity dot normal) * area
        for i in range(len(areas)):
            dot_product = np.dot(velocities[i], cell_normals[i])
            flow_rate += dot_product * areas[i]
        
        return flow_rate
    except Exception as e:
        logger.error(f"Error computing flow rate: {e}")
        return 0.0


def append_arrays(dataset, arrays: List[Tuple[str, np.ndarray, int]]) -> Optional[vtk.vtkDataSet]:
    """
    Append new arrays to a dataset.
    
    Args:
        dataset: VTK dataset
        arrays: List of tuples (name, data, num_components)
        
    Returns:
        vtk.vtkDataSet: Dataset with appended arrays
    """
    if not VTK_AVAILABLE:
        logger.error("VTK not available")
        return None
    
    try:
        # Create a deep copy of the dataset
        new_dataset = vtk.vtkDataSet.SafeDownCast(dataset.NewInstance())
        new_dataset.DeepCopy(dataset)
        
        # Add each array
        for name, data, num_components in arrays:
            # Reshape data if necessary
            if num_components > 1:
                if data.ndim == 1:
                    data = data.reshape(-1, num_components)
                data_flat = data.flatten()
            else:
                data_flat = data
            
            # Create VTK array
            vtk_array = numpy_to_vtk(data_flat, name=name, deep=True)
            vtk_array.SetNumberOfComponents(num_components)
            
            # Add to point data
            if len(data) == new_dataset.GetNumberOfPoints():
                new_dataset.GetPointData().AddArray(vtk_array)
            # Add to cell data
            elif len(data) == new_dataset.GetNumberOfCells():
                new_dataset.GetCellData().AddArray(vtk_array)
            else:
                logger.warning(f"Array '{name}' size doesn't match points or cells")
        
        return new_dataset
    except Exception as e:
        logger.error(f"Error appending arrays: {e}")
        return None


def write_vtk(dataset, filepath: str) -> bool:
    """
    Write a dataset to a VTK file.
    
    Args:
        dataset: VTK dataset
        filepath (str): Output file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not VTK_AVAILABLE:
        logger.error("VTK not available")
        return False
    
    try:
        # Determine writer type based on extension
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.vtp':
            writer = vtk.vtkXMLPolyDataWriter()
        elif ext == '.vtu':
            writer = vtk.vtkXMLUnstructuredGridWriter()
        elif ext == '.vts':
            writer = vtk.vtkXMLStructuredGridWriter()
        elif ext == '.vtr':
            writer = vtk.vtkXMLRectilinearGridWriter()
        elif ext == '.vti':
            writer = vtk.vtkXMLImageDataWriter()
        elif ext == '.vtk':
            writer = vtk.vtkDataSetWriter()
        else:
            logger.warning(f"Unsupported file extension: {ext}")
            return False
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Set up writer
        writer.SetFileName(filepath)
        writer.SetInputData(dataset)
        writer.SetDataModeToAscii()  # Use ASCII format for readability
        
        # Write file
        writer.Update()
        
        logger.info(f"Dataset written to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error writing VTK file: {e}")
        return False


def convert_openfoam_to_vtk(foam_case: str, output_dir: str, timestep: str = 'latest') -> bool:
    """
    Convert an OpenFOAM case to VTK files.
    
    Args:
        foam_case (str): Path to OpenFOAM case directory
        output_dir (str): Directory to write VTK files
        timestep (str): Timestep to convert ('latest' or a specific time)
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not VTK_AVAILABLE:
        logger.error("VTK not available")
        return False
    
    try:
        # Make sure the foam case exists
        if not os.path.isdir(foam_case):
            logger.error(f"OpenFOAM case directory not found: {foam_case}")
            return False
        
        # Create a fake .foam file for the reader
        foam_file = os.path.join(foam_case, "case.foam")
        with open(foam_file, "w") as f:
            pass
        
        # Create OpenFOAM reader
        reader = vtk.vtkOpenFOAMReader()
        reader.SetFileName(foam_file)
        reader.Update()
        
        # Set which time steps to read
        if timestep == 'latest':
            reader.SetTimeValue(reader.GetTimeValues().GetMaxNorm())
        else:
            try:
                time_value = float(timestep)
                # Find closest time
                times = reader.GetTimeValues()
                closest_time = times.GetValue(0)
                closest_diff = abs(time_value - closest_time)
                
                for i in range(1, times.GetNumberOfTuples()):
                    time = times.GetValue(i)
                    diff = abs(time_value - time)
                    if diff < closest_diff:
                        closest_time = time
                        closest_diff = diff
                
                reader.SetTimeValue(closest_time)
            except ValueError:
                logger.warning(f"Invalid timestep: {timestep}, using latest")
                reader.SetTimeValue(reader.GetTimeValues().GetMaxNorm())
        
        # Read all fields and patches
        reader.SetDecomposePolyhedra(True)
        reader.ReadAllVariablesOn()
        reader.Update()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the dataset
        dataset = reader.GetOutput()
        
        # Write internal mesh
        internal_mesh = extract_block(dataset, 0)
        if internal_mesh:
            write_vtk(internal_mesh, os.path.join(output_dir, "internalMesh.vtu"))
        
        # Write boundaries
        boundaries = dataset.GetBlock(1)
        if boundaries:
            for i in range(boundaries.GetNumberOfBlocks()):
                if boundaries.GetMetaData(i) and boundaries.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME()):
                    boundary_name = boundaries.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME())
                    boundary = boundaries.GetBlock(i)
                    if boundary:
                        write_vtk(boundary, os.path.join(output_dir, f"{boundary_name}.vtp"))
        
        # Clean up temporary .foam file
        os.remove(foam_file)
        
        logger.info(f"OpenFOAM case converted to VTK files in {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Error converting OpenFOAM to VTK: {e}")
        return False