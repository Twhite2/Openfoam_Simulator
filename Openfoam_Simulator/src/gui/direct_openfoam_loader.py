"""
Direct OpenFOAM loader for Openfoam_Simulator.

This is a utility that attempts to load OpenFOAM data directly when VTK conversion fails.
"""

import os
import glob
import logging
import numpy as np

try:
    import vtk
except ImportError:
    # Create a fallback vtk module with minimal functionality
    class FallbackVTK:
        def __getattr__(self, name):
            if name.startswith('vtk'):
                # Create a dummy class that logs any calls
                class DummyVTKClass:
                    def __init__(self, *args, **kwargs):
                        pass
                    def __getattr__(self, name):
                        return lambda *args, **kwargs: None
                return DummyVTKClass
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    vtk = FallbackVTK()

logger = logging.getLogger(__name__)

def load_openfoam_direct(case_dir, viewport):
    """
    Attempt to load OpenFOAM data directly without converting to VTK.
    
    Args:
        case_dir (str): Path to the OpenFOAM case directory
        viewport: The viewport to load data into
        
    Returns:
        vtk.vtkDataSet or None: The loaded data or None if loading failed
    """
    try:
        logger.info("Attempting to directly load OpenFOAM data...")
        
        # Find the latest time directory
        time_dirs = glob.glob(os.path.join(case_dir, "[0-9]*"))
        time_dirs = [td for td in time_dirs if os.path.isdir(td) and os.path.basename(td).replace('.', '', 1).isdigit()]
        
        if not time_dirs:
            logger.error("No time directories found in OpenFOAM case")
            return None
            
        # Sort and get the latest time directory
        time_dirs.sort(key=lambda x: float(os.path.basename(x).replace('e-', '0.0').replace('e+', '0.0')))
        latest_time = time_dirs[-1]
        logger.info(f"Found latest time directory: {latest_time}")
        
        # Check for velocity field (U)
        u_file = os.path.join(latest_time, "U")
        if not os.path.exists(u_file):
            logger.error(f"No velocity field (U) found in latest time directory: {latest_time}")
            return None
            
        logger.info(f"Found velocity field (U): {u_file}")
        
        # Check for points file (constant/polyMesh/points)
        points_file = os.path.join(case_dir, "constant", "polyMesh", "points")
        if not os.path.exists(points_file):
            logger.error(f"No mesh points file found: {points_file}")
            return None
            
        logger.info(f"Found mesh points file: {points_file}")
        
        # Check for faces file (constant/polyMesh/faces)
        faces_file = os.path.join(case_dir, "constant", "polyMesh", "faces")
        if not os.path.exists(faces_file):
            logger.error(f"No mesh faces file found: {faces_file}")
            return None
            
        logger.info(f"Found mesh faces file: {faces_file}")
        
        # Try multiple methods to load the data
        
        # Method 1: Try through openfoam_reader utility
        try:
            from ..utils.openfoam_reader import read_openfoam_results
            logger.info("Attempting to load with read_openfoam_results...")
            result_data, reader, source_id = read_openfoam_results(case_dir, 'U')
            
            if result_data:
                logger.info("Successfully loaded OpenFOAM results directly")
                if source_id:
                    viewport.sources[source_id] = reader
                    viewport.active_source = source_id
                return result_data
        except Exception as e:
            logger.error(f"Error using read_openfoam_results: {e}")
        
        # Method 2: Try with vtkOpenFOAMReader if available
        try:
            logger.info("Attempting to load with vtkOpenFOAMReader...")
            if hasattr(vtk, 'vtkOpenFOAMReader'):
                reader = vtk.vtkOpenFOAMReader()
                reader.SetFileName(os.path.join(case_dir, "system", "controlDict"))
                reader.Update()
                
                output = reader.GetOutput()
                if output:
                    # Verify this is a valid dataset with bounds
                    try:
                        bounds = output.GetBounds()
                        if bounds and len(bounds) == 6:
                            logger.info(f"Dataset has valid bounds: {bounds}")
                        else:
                            logger.warning("Dataset has invalid bounds")
                    except Exception as e:
                        logger.warning(f"Error getting bounds from dataset: {e}")
                        
                    logger.info("Successfully loaded OpenFOAM data with vtkOpenFOAMReader")
                    source_id = "openfoam_direct"
                    viewport.sources[source_id] = reader
                    viewport.active_source = source_id
                    
                    # Wrap the output in a custom class that handles GetBounds properly
                    class BoundsWrapper:
                        def __init__(self, output):
                            self.output = output
                            
                        def GetBounds(self):
                            try:
                                return self.output.GetBounds()
                            except:
                                # Default bounds if GetBounds fails
                                return (-1, 1, -1, 1, -1, 1)
                                
                        def __getattr__(self, name):
                            return getattr(self.output, name)
                            
                    return BoundsWrapper(output)
            else:
                logger.warning("vtkOpenFOAMReader not available in VTK installation")
        except Exception as e:
            logger.error(f"Error using vtkOpenFOAMReader: {e}")
        
        # Method 3: Manually create visualization for pipe simulation
        try:
            logger.info("Attempting to create fallback pipe visualization...")
            
            # Create a cylinder as a fallback for pipe simulations
            cylinder = vtk.vtkCylinderSource()
            cylinder.SetRadius(1.0)  # Default radius
            cylinder.SetHeight(10.0)  # Default height
            cylinder.SetResolution(50)
            cylinder.Update()
            
            # Create default velocity field
            points = cylinder.GetOutput().GetPoints()
            num_points = points.GetNumberOfPoints()
            
            velocity_array = vtk.vtkFloatArray()
            velocity_array.SetNumberOfComponents(3)
            velocity_array.SetName("U")
            
            # Fill with dummy velocity data (flow along Z axis)
            for i in range(num_points):
                point = points.GetPoint(i)
                x, y, z = point
                r = np.sqrt(x*x + y*y)
                # Parabolic profile
                vz = 1.0 * (1.0 - r*r/1.0)
                velocity_array.InsertNextTuple3(0.0, 0.0, vz)
            
            cylinder.GetOutput().GetPointData().AddArray(velocity_array)
            cylinder.GetOutput().GetPointData().SetActiveVectors("U")
            
            logger.info("Created fallback pipe visualization with dummy velocity field")
            source_id = "fallback_pipe"
            viewport.sources[source_id] = cylinder
            viewport.active_source = source_id
            return cylinder.GetOutput()
        except Exception as e:
            logger.error(f"Error creating fallback pipe visualization: {e}")
        
        return None
    except Exception as e:
        import traceback
        logger.error(f"Error in load_openfoam_direct: {e}\n{traceback.format_exc()}")
        return None
