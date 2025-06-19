"""
Integration code to connect visualization state saving/loading with the Project class.
This patch will be applied to the main Project class to add visualization state persistence.
"""

# Add this import near the top of project.py
# from .visualization_state import VisualizationState

def patch_save_method(self, main_window) -> bool:
    """
    Enhanced save method that also saves visualization state.
    
    Args:
        main_window: Reference to the main window for accessing UI components
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Call original save method
    if not self.save():
        return False
    
    # Save visualization state
    from .visualization_state import VisualizationState
    return VisualizationState.save_state(self, main_window)

def patch_load_with_state(project, main_window):
    """
    Apply visualization state after loading a project.
    
    Args:
        project: The loaded project
        main_window: Reference to the main window for accessing UI components
        
    Returns:
        The project with visualization state applied
    """
    if not project:
        return project
    
    # Load visualization state
    from .visualization_state import VisualizationState
    VisualizationState.load_state(project, main_window)
    
    return project
