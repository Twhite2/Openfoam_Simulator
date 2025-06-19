#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation lock mechanism for Openfoam_Simulator.

This module provides a global lock to prevent any part of the application
from accessing OpenFOAM case files during an active simulation, which
can cause segmentation faults and other crashes.
"""

import os
import time
import threading
import logging
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

class SimulationLockManager:
    """
    A manager for handling simulation locks to prevent file access conflicts.
    
    This is a singleton class that provides a global mechanism to track
    active OpenFOAM simulations and prevent other components from 
    accessing case files during simulation, which can cause crashes.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the lock manager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the simulation lock manager."""
        self._active_simulations: Dict[str, dict] = {}  # case_dir -> info
        self._case_locks: Dict[str, threading.RLock] = {}  # case_dir -> lock
        self._blocked_components: Set[str] = set()  # Components blocked from accessing cases
    
    def register_simulation(self, case_dir: str, solver: str, pid: Optional[int] = None) -> bool:
        """
        Register an active simulation.
        
        Args:
            case_dir (str): Path to the case directory
            solver (str): Name of the OpenFOAM solver
            pid (int, optional): Process ID of the simulation
            
        Returns:
            bool: True if registered successfully, False if already registered
        """
        with self._lock:
            if case_dir in self._active_simulations:
                return False
            
            self._active_simulations[case_dir] = {
                'solver': solver,
                'pid': pid,
                'start_time': time.time(),
                'status': 'running'
            }
            
            self._case_locks[case_dir] = threading.RLock()
            logger.info(f"Registered active simulation in {case_dir} with solver {solver}")
            return True
    
    def unregister_simulation(self, case_dir: str) -> bool:
        """
        Unregister an active simulation.
        
        Args:
            case_dir (str): Path to the case directory
            
        Returns:
            bool: True if unregistered successfully, False if not registered
        """
        with self._lock:
            if case_dir not in self._active_simulations:
                return False
            
            del self._active_simulations[case_dir]
            if case_dir in self._case_locks:
                del self._case_locks[case_dir]
            
            logger.info(f"Unregistered simulation in {case_dir}")
            return True
    
    def is_simulation_active(self, case_dir: Optional[str] = None) -> bool:
        """
        Check if any simulation is active, or if a simulation is active for a specific case.
        
        Args:
            case_dir (str, optional): Path to the case directory. 
                If None, checks if any simulation is active.
                
        Returns:
            bool: True if a simulation is active, False otherwise
        """
        try:
            # Use a separate lock to avoid deadlocks
            manager = get_lock_manager()
            
            # If no case_dir specified, check if any simulation is active
            if case_dir is None:
                # Check without acquiring the lock first for speed
                if not manager._active_simulations:
                    return False
                    
                # Double-check with lock
                with manager._lock:
                    return len(manager._active_simulations) > 0
            
            # Check specific case directory
            # Check without acquiring lock first for speed
            if case_dir not in manager._active_simulations:
                return False
                
            # Double-check with lock
            with manager._lock:
                return case_dir in manager._active_simulations
        except Exception as e:
            # If anything goes wrong, assume no simulation is active
            # This prevents crashes if the lock manager is not initialized properly
            logger.error(f"Error checking simulation status: {e}")
            return False
    
    def block_component(self, component_name: str) -> None:
        """
        Block a component from accessing case files during simulation.
        
        Args:
            component_name (str): Name of the component to block
        """
        with self._lock:
            self._blocked_components.add(component_name)
            logger.info(f"Blocked component {component_name} from accessing case files during simulation")
    
    def unblock_component(self, component_name: str) -> None:
        """
        Unblock a component from accessing case files during simulation.
        
        Args:
            component_name (str): Name of the component to unblock
        """
        with self._lock:
            if component_name in self._blocked_components:
                self._blocked_components.remove(component_name)
                logger.info(f"Unblocked component {component_name}")
    
    def is_component_blocked(self, component_name: str) -> bool:
        """
        Check if a component is blocked from accessing case files.
        
        Args:
            component_name (str): Name of the component to check
            
        Returns:
            bool: True if the component is blocked, False otherwise
        """
        with self._lock:
            return component_name in self._blocked_components
    
    def acquire_case_lock(self, case_dir: str, component_name: str, timeout: float = 0.5) -> bool:
        """
        Attempt to acquire a lock on a case directory.
        
        Args:
            case_dir (str): Path to the case directory
            component_name (str): Name of the component requesting the lock
            timeout (float): Timeout in seconds
            
        Returns:
            bool: True if lock acquired, False otherwise
        """
        # If simulation is active and component is blocked, don't even try
        if self.is_simulation_active(case_dir) and self.is_component_blocked(component_name):
            logger.warning(f"Component {component_name} blocked from accessing {case_dir} during active simulation")
            return False
        
        # If no lock exists for this case, create one
        with self._lock:
            if case_dir not in self._case_locks:
                self._case_locks[case_dir] = threading.RLock()
        
        # Try to acquire the lock
        lock = self._case_locks.get(case_dir)
        if lock is None:
            return False
        
        acquired = lock.acquire(blocking=True, timeout=timeout)
        if acquired:
            logger.debug(f"Component {component_name} acquired lock on {case_dir}")
        else:
            logger.warning(f"Component {component_name} failed to acquire lock on {case_dir}")
        
        return acquired
    
    def release_case_lock(self, case_dir: str, component_name: str) -> bool:
        """
        Release a lock on a case directory.
        
        Args:
            case_dir (str): Path to the case directory
            component_name (str): Name of the component releasing the lock
            
        Returns:
            bool: True if lock released, False otherwise
        """
        lock = self._case_locks.get(case_dir)
        if lock is None:
            return False
        
        try:
            lock.release()
            logger.debug(f"Component {component_name} released lock on {case_dir}")
            return True
        except RuntimeError:
            logger.warning(f"Component {component_name} attempted to release lock it doesn't hold on {case_dir}")
            return False
    
    def get_all_active_simulations(self) -> Dict[str, dict]:
        """
        Get all active simulations.
        
        Returns:
            Dict[str, dict]: Dictionary of active simulations, keyed by case directory
        """
        with self._lock:
            return self._active_simulations.copy()


# Global instance accessor functions
def get_lock_manager() -> SimulationLockManager:
    """Get the singleton instance of the simulation lock manager."""
    return SimulationLockManager.get_instance()

def register_simulation(case_dir: str, solver: str, pid: Optional[int] = None) -> bool:
    """Register an active simulation."""
    return get_lock_manager().register_simulation(case_dir, solver, pid)

def unregister_simulation(case_dir: str) -> bool:
    """Unregister an active simulation."""
    return get_lock_manager().unregister_simulation(case_dir)

def is_simulation_active(case_dir: Optional[str] = None) -> bool:
    """Check if any simulation is active, or if a simulation is active for a specific case."""
    return get_lock_manager().is_simulation_active(case_dir)

def block_component(component_name: str) -> None:
    """Block a component from accessing case files during simulation."""
    get_lock_manager().block_component(component_name)

def unblock_component(component_name: str) -> None:
    """Unblock a component from accessing case files during simulation."""
    get_lock_manager().unblock_component(component_name)

def is_component_blocked(component_name: str) -> bool:
    """Check if a component is blocked from accessing case files."""
    return get_lock_manager().is_component_blocked(component_name)

def safe_during_simulation(component_name: str):
    """
    Decorator to make a function safe during simulation.
    This will prevent the function from executing if a simulation is active.
    
    Args:
        component_name (str): Name of the component this function belongs to
        
    Returns:
        callable: Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # For methods, args[0] is self
            self_obj = args[0] if args else None
            case_dir = None
            
            # Try to get case_dir from self
            if self_obj:
                if hasattr(self_obj, 'case_dir') and self_obj.case_dir:
                    case_dir = self_obj.case_dir
                elif hasattr(self_obj, 'main_window'):
                    main_window = self_obj.main_window
                    if hasattr(main_window, 'current_project') and main_window.current_project:
                        if hasattr(main_window.current_project, 'case_dir'):
                            case_dir = main_window.current_project.case_dir
            
            # If no case_dir found, can't check if simulation is active
            if not case_dir:
                return func(*args, **kwargs)
            
            # Check if component is blocked during active simulation
            if is_simulation_active(case_dir) and is_component_blocked(component_name):
                logger.warning(f"Function {func.__name__} in {component_name} blocked during active simulation")
                return None
            
            # Execute function if not blocked
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator
