"""
Performance Timer Utility

Ensures training runs complete within the 10-minute constraint.

Usage:
    with Timer("Training"):
        train_model()
    
    # Or as decorator:
    @timed(max_seconds=600)
    def train_model():
        ...
"""

import time
from functools import wraps
from typing import Callable, Optional


class Timer:
    """
    Context manager for timing code blocks.
    
    Usage:
        with Timer("My operation", max_seconds=60):
            expensive_operation()
    """
    
    def __init__(self, name: str = "Operation", max_seconds: float = 600):
        """
        Initialize timer.
        
        Args:
            name: Description of what's being timed
            max_seconds: Maximum allowed time (default: 600 = 10 minutes)
        """
        self.name = name
        self.max_seconds = max_seconds
        self.elapsed = 0.0
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        
        if self.elapsed < self.max_seconds:
            status = "✅"
            suffix = ""
        else:
            status = "⚠️"
            suffix = f" (exceeded {self.max_seconds}s limit!)"
        
        print(f"{status} {self.name}: {self.elapsed:.2f}s{suffix}")
    
    def get_elapsed(self) -> float:
        """Get elapsed time while still running."""
        if self.start_time is None:
            return 0.0
        return time.perf_counter() - self.start_time


def timed(max_seconds: float = 600, name: Optional[str] = None):
    """
    Decorator to time function execution.
    
    Args:
        max_seconds: Maximum allowed time
        name: Custom name (defaults to function name)
    
    Usage:
        @timed(max_seconds=60)
        def train_epoch():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name or func.__name__
            with Timer(timer_name, max_seconds):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class ProgressTimer:
    """
    Timer that shows progress during long operations.
    
    Usage:
        timer = ProgressTimer(total_steps=100, name="Training")
        for i in range(100):
            do_step()
            timer.step()
        timer.finish()
    """
    
    def __init__(self, total_steps: int, name: str = "Progress", 
                 max_seconds: float = 600, report_every: int = 10):
        self.total_steps = total_steps
        self.name = name
        self.max_seconds = max_seconds
        self.report_every = report_every
        
        self.current_step = 0
        self.start_time = time.perf_counter()
    
    def step(self):
        """Record completion of one step."""
        self.current_step += 1
        
        if self.current_step % self.report_every == 0 or self.current_step == self.total_steps:
            elapsed = time.perf_counter() - self.start_time
            progress = self.current_step / self.total_steps
            
            # Estimate remaining time
            if progress > 0:
                estimated_total = elapsed / progress
                remaining = estimated_total - elapsed
            else:
                remaining = 0
            
            bar_width = 30
            filled = int(bar_width * progress)
            bar = "█" * filled + "░" * (bar_width - filled)
            
            print(f"\r{self.name}: [{bar}] {progress*100:.1f}% | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s", end="")
    
    def finish(self):
        """Mark completion and print final stats."""
        elapsed = time.perf_counter() - self.start_time
        
        print()  # New line after progress bar
        
        if elapsed < self.max_seconds:
            print(f"✅ {self.name} completed in {elapsed:.2f}s")
        else:
            print(f"⚠️ {self.name} completed in {elapsed:.2f}s (exceeded {self.max_seconds}s limit!)")


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


if __name__ == "__main__":
    print("Testing timer utilities...\n")
    
    # Test 1: Basic Timer
    print("Test 1: Basic Timer")
    with Timer("Sleep test", max_seconds=1):
        time.sleep(0.5)
    
    # Test 2: Decorator
    print("\nTest 2: Decorated function")
    
    @timed(max_seconds=1)
    def slow_function():
        time.sleep(0.3)
        return "done"
    
    result = slow_function()
    print(f"Result: {result}")
    
    # Test 3: Progress Timer
    print("\nTest 3: Progress Timer")
    timer = ProgressTimer(total_steps=50, name="Demo", report_every=10)
    for i in range(50):
        time.sleep(0.02)  # Simulate work
        timer.step()
    timer.finish()
    
    # Test 4: Format time
    print("\nTest 4: Time formatting")
    print(f"30 seconds: {format_time(30)}")
    print(f"90 seconds: {format_time(90)}")
    print(f"3700 seconds: {format_time(3700)}")
    
    print("\n✅ All timer tests passed!")
