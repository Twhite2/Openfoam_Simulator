"""
This file contains the missing view mode methods for the VisualizationControls class.
Copy these methods back into visualization_controls.py to fix the error.
"""

def _on_view_mode_changed(self, checked):
    """
    Handle view mode radio button change.
    
    Args:
        checked (bool): Whether the button is checked
    """
    if not checked:
        return  # Only respond to the button that was checked
        
    # Apply the view mode immediately if we have flow actors
    if hasattr(self.main_window, 'viewport'):
        viewport = self.main_window.viewport
        
        # Determine the view mode
        view_mode = None
        if self.view_internal.isChecked():
            view_mode = "internal"
        elif self.view_external.isChecked():
            view_mode = "external"
        else:  # view_both is checked
            view_mode = "both"
                
        # Apply the view mode
        self._apply_view_mode(view_mode)

def _apply_view_mode(self, view_mode):
    """
    Apply the selected view mode (internal, external, both).
    
    Args:
        view_mode (str): The view mode to apply
    """
    if not hasattr(self.main_window, 'viewport'):
        return
        
    viewport = self.main_window.viewport
    
    # Keep track of which actors need to be visible based on view mode
    streamline_actors_visible = {}
    vector_actors_visible = {}
    glyph_actors_visible = {}
    
    # Get all streamline actors (if any)
    if hasattr(viewport, 'streamline_actors'):
        for name, actor in viewport.streamline_actors.items():
            if view_mode == "internal" and "external" in name:
                streamline_actors_visible[name] = False
            elif view_mode == "external" and "internal" in name:
                streamline_actors_visible[name] = False
            else:
                streamline_actors_visible[name] = True
                
    # Get all vector actors (if any)
    if hasattr(viewport, 'vector_actors'):
        for name, actor in viewport.vector_actors.items():
            if view_mode == "internal" and "external" in name:
                vector_actors_visible[name] = False
            elif view_mode == "external" and "internal" in name:
                vector_actors_visible[name] = False
            else:
                vector_actors_visible[name] = True
                
    # Get all glyph actors (if any)
    if hasattr(viewport, 'glyph_actors'):
        for name, actor in viewport.glyph_actors.items():
            if view_mode == "internal" and "external" in name:
                glyph_actors_visible[name] = False
            elif view_mode == "external" and "internal" in name:
                glyph_actors_visible[name] = False
            else:
                glyph_actors_visible[name] = True
                
    # Apply visibility to streamline actors
    if hasattr(viewport, 'streamline_actors'):
        for name, actor in viewport.streamline_actors.items():
            actor.SetVisibility(streamline_actors_visible.get(name, True))
            
    # Apply visibility to vector actors
    if hasattr(viewport, 'vector_actors'):
        for name, actor in viewport.vector_actors.items():
            actor.SetVisibility(vector_actors_visible.get(name, True))
            
    # Apply visibility to glyph actors
    if hasattr(viewport, 'glyph_actors'):
        for name, actor in viewport.glyph_actors.items():
            actor.SetVisibility(glyph_actors_visible.get(name, True))
            
    # Render the scene
    if hasattr(viewport, 'render_window'):
        viewport.render_window.Render()
