# Import simulation lock manager
try:
    from ..utils.simulation_lock import (
        register_simulation, unregister_simulation, 
        is_simulation_active, block_component, 
        unblock_component, safe_during_simulation
    )
except ImportError:
    # Fallbacks if not available
    def register_simulation(*args, **kwargs): pass
    def unregister_simulation(*args, **kwargs): pass
    def is_simulation_active(*args, **kwargs): return False
    def block_component(*args, **kwargs): pass
    def unblock_component(*args, **kwargs): pass
    def safe_during_simulation(*args, **kwargs): return lambda func: func
