#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Progress tracking utilities for Openfoam_Simulator application.

This module provides utilities for tracking and reporting progress
of long-running operations, with support for both console and GUI progress
reporting, nested sub-tasks, time estimation, and cancellation.
"""

import time
import threading
import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
from abc import ABC, abstractmethod
import logging

# Import project modules
from .logger import get_logger

# Set up logging
logger = get_logger(__name__)


class ProgressState(Enum):
    """Enumeration of progress states."""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    CANCELLED = auto()
    ERROR = auto()


class ProgressObserver(ABC):
    """Abstract base class for progress observers."""
    
    @abstractmethod
    def update(self, progress: float, message: str, state: ProgressState) -> None:
        """
        Update the observer with current progress.
        
        Args:
            progress: Current progress (0.0 to 1.0)
            message: Progress message
            state: Current progress state
        """
        pass


class ConsoleProgressObserver(ProgressObserver):
    """Progress observer that prints to the console."""
    
    def __init__(self, use_bar: bool = True, bar_width: int = 50):
        """
        Initialize the console progress observer.
        
        Args:
            use_bar: Whether to show a progress bar
            bar_width: Width of the progress bar in characters
        """
        self.use_bar = use_bar
        self.bar_width = bar_width
        self.last_progress = -1
        
    def update(self, progress: float, message: str, state: ProgressState) -> None:
        """
        Update progress in the console.
        
        Args:
            progress: Current progress (0.0 to 1.0)
            message: Progress message
            state: Current progress state
        """
        # Only update if progress has changed significantly or state has changed
        if (progress - self.last_progress) < 0.01 and progress < 1.0:
            return
            
        self.last_progress = progress
        
        if self.use_bar:
            # Create progress bar
            bar_length = int(progress * self.bar_width)
            bar = "[" + "=" * bar_length + " " * (self.bar_width - bar_length) + "]"
            
            # Print progress
            print(f"\r{bar} {progress*100:.1f}% - {message}", end="")
            
            # Add newline if completed or cancelled
            if state in (ProgressState.COMPLETED, ProgressState.CANCELLED, ProgressState.ERROR):
                print()
        else:
            # Print simple progress message
            print(f"{progress*100:.1f}% - {message}")


class LoggingProgressObserver(ProgressObserver):
    """Progress observer that writes to the logger."""
    
    def __init__(self, log_level: int = logging.INFO, 
                min_interval: float = 1.0):
        """
        Initialize the logging progress observer.
        
        Args:
            log_level: Logging level
            min_interval: Minimum time interval between log entries (seconds)
        """
        self.log_level = log_level
        self.min_interval = min_interval
        self.last_log_time = 0
        
    def update(self, progress: float, message: str, state: ProgressState) -> None:
        """
        Update progress in the log.
        
        Args:
            progress: Current progress (0.0 to 1.0)
            message: Progress message
            state: Current progress state
        """
        current_time = time.time()
        
        # Only log if sufficient time has passed or state has changed
        if (current_time - self.last_log_time < self.min_interval and 
            state == ProgressState.RUNNING and
            progress < 1.0):
            return
            
        self.last_log_time = current_time
        
        # Log the progress
        log_message = f"Progress: {progress*100:.1f}% - {message} - State: {state.name}"
        logger.log(self.log_level, log_message)


class PyQtProgressObserver(ProgressObserver):
    """Progress observer that updates a PyQt progress bar and label."""
    
    def __init__(self, progress_bar=None, label=None, status_bar=None):
        """
        Initialize the PyQt progress observer.
        
        Args:
            progress_bar: PyQt progress bar widget
            label: PyQt label widget
            status_bar: PyQt status bar widget
        """
        self.progress_bar = progress_bar
        self.label = label
        self.status_bar = status_bar
        
    def update(self, progress: float, message: str, state: ProgressState) -> None:
        """
        Update progress in PyQt widgets.
        
        Args:
            progress: Current progress (0.0 to 1.0)
            message: Progress message
            state: Current progress state
        """
        try:
            # Convert progress to percentage (0-100)
            percentage = int(progress * 100)
            
            # Update progress bar
            if self.progress_bar:
                self.progress_bar.setValue(percentage)
                
                # Set color based on state
                if state == ProgressState.ERROR:
                    self.progress_bar.setStyleSheet("QProgressBar { color: white; background-color: #f0f0f0; } QProgressBar::chunk { background-color: #e74c3c; }")
                elif state == ProgressState.CANCELLED:
                    self.progress_bar.setStyleSheet("QProgressBar { color: white; background-color: #f0f0f0; } QProgressBar::chunk { background-color: #f39c12; }")
                elif state == ProgressState.PAUSED:
                    self.progress_bar.setStyleSheet("QProgressBar { color: white; background-color: #f0f0f0; } QProgressBar::chunk { background-color: #3498db; }")
                else:
                    self.progress_bar.setStyleSheet("QProgressBar { color: white; background-color: #f0f0f0; } QProgressBar::chunk { background-color: #2ecc71; }")
            
            # Update label
            if self.label:
                if state == ProgressState.RUNNING:
                    self.label.setText(f"{message} ({percentage}%)")
                else:
                    self.label.setText(f"{message} - {state.name}")
            
            # Update status bar
            if self.status_bar:
                self.status_bar.showMessage(f"{message} - {percentage}%")
        
        except Exception as e:
            logger.error(f"Error updating PyQt progress widgets: {e}")


class ProgressTracker:
    """
    Class for tracking progress of operations.
    
    Provides methods for updating and reporting progress, with support for
    nested sub-tasks, time estimation, and cancellation.
    """
    
    def __init__(self, task_name: str, parent: Optional['ProgressTracker'] = None,
                total_steps: int = 100, weight: float = 1.0):
        """
        Initialize the progress tracker.
        
        Args:
            task_name: Name of the task being tracked
            parent: Parent progress tracker for nested tasks
            total_steps: Total number of steps in the task
            weight: Weight of this task relative to siblings
        """
        self.task_name = task_name
        self.parent = parent
        self.total_steps = max(1, total_steps)
        self.current_step = 0
        self.weight = max(0.0001, weight)
        self.state = ProgressState.IDLE
        
        # Time tracking
        self.start_time = None
        self.elapsed_time = 0.0
        self.estimated_time_remaining = None
        
        # Sub-tasks
        self.sub_tasks = []
        self.sub_task_weights = []
        
        # Observers
        self.observers = []
        
        # Thread safety
        self.lock = threading.RLock()
        
        # If this is a sub-task, register with parent
        if self.parent:
            self.parent.add_sub_task(self)
    
    def add_observer(self, observer: ProgressObserver) -> None:
        """
        Add a progress observer.
        
        Args:
            observer: The observer to add
        """
        with self.lock:
            self.observers.append(observer)
    
    def remove_observer(self, observer: ProgressObserver) -> None:
        """
        Remove a progress observer.
        
        Args:
            observer: The observer to remove
        """
        with self.lock:
            if observer in self.observers:
                self.observers.remove(observer)
    
    def add_sub_task(self, sub_task: 'ProgressTracker') -> None:
        """
        Add a sub-task.
        
        Args:
            sub_task: The sub-task to add
        """
        with self.lock:
            self.sub_tasks.append(sub_task)
            self.sub_task_weights.append(sub_task.weight)
    
    def start(self) -> None:
        """Start the progress tracking."""
        with self.lock:
            self.start_time = time.time()
            self.state = ProgressState.RUNNING
            self.current_step = 0
            self.elapsed_time = 0.0
            self.estimated_time_remaining = None
            
            # Notify observers
            self._notify_observers()
    
    def update(self, step: Optional[int] = None, message: str = "") -> None:
        """
        Update progress.
        
        Args:
            step: Current step (if None, increment by 1)
            message: Progress message
        """
        with self.lock:
            if self.state not in (ProgressState.RUNNING, ProgressState.PAUSED):
                return
                
            # Update state
            self.state = ProgressState.RUNNING
            
            # Update step
            if step is not None:
                self.current_step = min(max(0, step), self.total_steps)
            else:
                self.current_step = min(self.current_step + 1, self.total_steps)
            
            # Update time estimates
            current_time = time.time()
            if self.start_time:
                self.elapsed_time = current_time - self.start_time
                
                # Estimate time remaining
                if self.current_step > 0:
                    time_per_step = self.elapsed_time / self.current_step
                    steps_remaining = self.total_steps - self.current_step
                    self.estimated_time_remaining = time_per_step * steps_remaining
            
            # Notify observers
            self._notify_observers(message)
            
            # Propagate to parent
            if self.parent:
                self.parent._update_from_sub_tasks()
    
    def _update_from_sub_tasks(self) -> None:
        """Update progress based on sub-tasks progress."""
        with self.lock:
            if not self.sub_tasks:
                return
                
            # Calculate weighted average progress
            total_weight = sum(self.sub_task_weights)
            if total_weight > 0:
                progress = 0.0
                for i, sub_task in enumerate(self.sub_tasks):
                    sub_progress = sub_task.get_progress()
                    progress += (sub_progress * self.sub_task_weights[i] / total_weight)
                
                # Update current step
                self.current_step = int(progress * self.total_steps)
                
                # Notify observers
                self._notify_observers()
                
                # Propagate to parent
                if self.parent:
                    self.parent._update_from_sub_tasks()
    
    def get_progress(self) -> float:
        """
        Get current progress as a fraction.
        
        Returns:
            float: Current progress (0.0 to 1.0)
        """
        with self.lock:
            return self.current_step / self.total_steps
    
    def complete(self, message: str = "Completed") -> None:
        """
        Mark the task as completed.
        
        Args:
            message: Completion message
        """
        with self.lock:
            self.current_step = self.total_steps
            self.state = ProgressState.COMPLETED
            
            # Update time
            if self.start_time:
                self.elapsed_time = time.time() - self.start_time
                self.estimated_time_remaining = 0.0
            
            # Notify observers
            self._notify_observers(message)
            
            # Propagate to parent
            if self.parent:
                self.parent._update_from_sub_tasks()
    
    def cancel(self, message: str = "Cancelled") -> None:
        """
        Cancel the task.
        
        Args:
            message: Cancellation message
        """
        with self.lock:
            self.state = ProgressState.CANCELLED
            
            # Update time
            if self.start_time:
                self.elapsed_time = time.time() - self.start_time
                self.estimated_time_remaining = None
            
            # Notify observers
            self._notify_observers(message)
            
            # Propagate to sub-tasks
            for sub_task in self.sub_tasks:
                sub_task.cancel(message)
            
            # Propagate to parent
            if self.parent:
                self.parent._update_from_sub_tasks()
    
    def error(self, message: str = "Error") -> None:
        """
        Mark the task as errored.
        
        Args:
            message: Error message
        """
        with self.lock:
            self.state = ProgressState.ERROR
            
            # Update time
            if self.start_time:
                self.elapsed_time = time.time() - self.start_time
                self.estimated_time_remaining = None
            
            # Notify observers
            self._notify_observers(message)
            
            # Propagate to parent
            if self.parent:
                self.parent._update_from_sub_tasks()
    
    def pause(self, message: str = "Paused") -> None:
        """
        Pause the task.
        
        Args:
            message: Pause message
        """
        with self.lock:
            if self.state == ProgressState.RUNNING:
                self.state = ProgressState.PAUSED
                
                # Notify observers
                self._notify_observers(message)
                
                # Propagate to sub-tasks
                for sub_task in self.sub_tasks:
                    sub_task.pause(message)
    
    def resume(self, message: str = "Resumed") -> None:
        """
        Resume the paused task.
        
        Args:
            message: Resume message
        """
        with self.lock:
            if self.state == ProgressState.PAUSED:
                self.state = ProgressState.RUNNING
                
                # Notify observers
                self._notify_observers(message)
                
                # Propagate to sub-tasks
                for sub_task in self.sub_tasks:
                    sub_task.resume(message)
    
    def is_cancelled(self) -> bool:
        """
        Check if the task is cancelled.
        
        Returns:
            bool: True if the task is cancelled
        """
        with self.lock:
            return self.state == ProgressState.CANCELLED
    
    def get_elapsed_time(self) -> float:
        """
        Get elapsed time in seconds.
        
        Returns:
            float: Elapsed time
        """
        with self.lock:
            if self.start_time and self.state == ProgressState.RUNNING:
                return time.time() - self.start_time
            return self.elapsed_time
    
    def get_estimated_time_remaining(self) -> Optional[float]:
        """
        Get estimated time remaining in seconds.
        
        Returns:
            float or None: Estimated time remaining, or None if not available
        """
        with self.lock:
            return self.estimated_time_remaining
    
    def get_formatted_progress(self) -> str:
        """
        Get formatted progress string.
        
        Returns:
            str: Formatted progress string
        """
        with self.lock:
            progress = self.get_progress() * 100
            return f"{progress:.1f}%"
    
    def get_formatted_elapsed_time(self) -> str:
        """
        Get formatted elapsed time string.
        
        Returns:
            str: Formatted elapsed time string
        """
        with self.lock:
            elapsed = self.get_elapsed_time()
            return self._format_time_duration(elapsed)
    
    def get_formatted_estimated_time_remaining(self) -> str:
        """
        Get formatted estimated time remaining string.
        
        Returns:
            str: Formatted time remaining string
        """
        with self.lock:
            remaining = self.get_estimated_time_remaining()
            if remaining is None:
                return "Unknown"
            return self._format_time_duration(remaining)
    
    def _format_time_duration(self, seconds: float) -> str:
        """
        Format time duration as a string.
        
        Args:
            seconds: Time duration in seconds
            
        Returns:
            str: Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.1f} s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} min"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} h"
    
    def _notify_observers(self, message: str = "") -> None:
        """
        Notify all observers of the current progress.
        
        Args:
            message: Progress message
        """
        progress = self.get_progress()
        for observer in self.observers:
            try:
                observer.update(progress, message, self.state)
            except Exception as e:
                logger.error(f"Error notifying observer: {e}")


class ProgressContext:
    """
    Context manager for progress tracking.
    
    Automatically starts and completes a progress tracker when used
    in a with statement.
    """
    
    def __init__(self, tracker: ProgressTracker, message: str = ""):
        """
        Initialize the progress context.
        
        Args:
            tracker: The progress tracker to use
            message: Initial progress message
        """
        self.tracker = tracker
        self.message = message
    
    def __enter__(self) -> ProgressTracker:
        """
        Enter the context.
        
        Returns:
            ProgressTracker: The progress tracker
        """
        # Start tracking
        self.tracker.start()
        if self.message:
            self.tracker.update(message=self.message)
        return self.tracker
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the context.
        
        Args:
            exc_type: Exception type, if any
            exc_val: Exception value, if any
            exc_tb: Exception traceback, if any
        """
        # Check if an exception occurred
        if exc_type is not None:
            self.tracker.error(f"Error: {exc_val}")
        else:
            self.tracker.complete()


def create_progress_tracker(task_name: str, total_steps: int = 100,
                          console: bool = True, logger: bool = True,
                          log_level: int = logging.INFO) -> ProgressTracker:
    """
    Create a progress tracker with common observers.
    
    Args:
        task_name: Name of the task
        total_steps: Total number of steps
        console: Whether to add a console observer
        logger: Whether to add a logger observer
        log_level: Logging level for logger observer
        
    Returns:
        ProgressTracker: The configured progress tracker
    """
    # Create tracker
    tracker = ProgressTracker(task_name, total_steps=total_steps)
    
    # Add observers
    if console:
        tracker.add_observer(ConsoleProgressObserver())
    if logger:
        tracker.add_observer(LoggingProgressObserver(log_level=log_level))
    
    return tracker


def create_pyqt_progress_tracker(task_name: str, total_steps: int = 100,
                               progress_bar=None, label=None, status_bar=None,
                               add_console: bool = False,
                               add_logger: bool = True) -> ProgressTracker:
    """
    Create a progress tracker with PyQt observer.
    
    Args:
        task_name: Name of the task
        total_steps: Total number of steps
        progress_bar: PyQt progress bar widget
        label: PyQt label widget
        status_bar: PyQt status bar widget
        add_console: Whether to add a console observer
        add_logger: Whether to add a logger observer
        
    Returns:
        ProgressTracker: The configured progress tracker
    """
    # Create tracker
    tracker = ProgressTracker(task_name, total_steps=total_steps)
    
    # Add PyQt observer
    tracker.add_observer(PyQtProgressObserver(progress_bar, label, status_bar))
    
    # Add other observers if requested
    if add_console:
        tracker.add_observer(ConsoleProgressObserver())
    if add_logger:
        tracker.add_observer(LoggingProgressObserver())
    
    return tracker


def create_sub_task(parent_tracker: ProgressTracker, name: str, 
                   total_steps: int = 100, weight: float = 1.0) -> ProgressTracker:
    """
    Create a sub-task for a parent tracker.
    
    Args:
        parent_tracker: The parent progress tracker
        name: Name of the sub-task
        total_steps: Total number of steps
        weight: Weight of this task relative to siblings
        
    Returns:
        ProgressTracker: The sub-task tracker
    """
    return ProgressTracker(name, parent=parent_tracker, total_steps=total_steps, weight=weight)


# Example usage
if __name__ == "__main__":
    # Create a progress tracker
    tracker = create_progress_tracker("Example Task", total_steps=100)
    
    # Use with context manager
    with ProgressContext(tracker, "Starting task"):
        # Create sub-tasks for different phases
        prepare_tracker = create_sub_task(tracker, "Preparation", total_steps=20, weight=1.0)
        process_tracker = create_sub_task(tracker, "Processing", total_steps=60, weight=3.0)
        finalize_tracker = create_sub_task(tracker, "Finalization", total_steps=20, weight=1.0)
        
        # Phase 1: Preparation
        with ProgressContext(prepare_tracker, "Preparing data"):
            for i in range(20):
                # Simulate work
                time.sleep(0.1)
                prepare_tracker.update(message=f"Preparing item {i+1}/20")
        
        # Phase 2: Processing
        with ProgressContext(process_tracker, "Processing data"):
            for i in range(60):
                # Simulate work
                time.sleep(0.1)
                process_tracker.update(message=f"Processing item {i+1}/60")
                
                # Show how to handle cancellation (for demonstration)
                if i == 30:
                    # Uncomment to test cancellation
                    # process_tracker.cancel("User cancelled")
                    # break
                    pass
        
        # Phase 3: Finalization
        with ProgressContext(finalize_tracker, "Finalizing results"):
            for i in range(20):
                # Simulate work
                time.sleep(0.1)
                finalize_tracker.update(message=f"Finalizing item {i+1}/20")
    
    print(f"Task completed in {tracker.get_formatted_elapsed_time()}")