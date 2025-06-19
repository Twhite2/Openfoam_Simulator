"""
Integration instructions for adding visualization state persistence to Openfoam_Simulator.

This will ensure that boundary conditions, solver parameters, ambient regions, and other
visualization settings are properly restored when reopening projects.
"""

# Step 1: Add these imports near the top of main_window.py
from ..models.visualization_state import VisualizationState

# Step 2: Modify the save_project method in main_window.py to save visualization state
def enhanced_save_project(self, filepath=None):
    """
    Save the current project to a file, including visualization state.
    
    Args:
        filepath (str, optional): Path to save the project to
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Original save project code here...
    
    # This should be added at the end of your save_project method:
    # Save visualization state
    if self.project and self.project.project_dir:
        VisualizationState.save_state(self.project, self)
        
    return True

# Step 3: Modify the open_project method in main_window.py to restore visualization state
def enhanced_open_project(self, filepath):
    """
    Open a project from a file and restore visualization state.
    
    Args:
        filepath (str): Path to the project file
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Original open project code here...
    # This typically loads the project and sets self.project
    
    # This should be added after the project is loaded but before the end of your open_project method:
    # Load visualization state
    if self.project and self.project.project_dir:
        VisualizationState.load_state(self.project, self)
        
    return True
