"""
Beat Timeline - A class for managing beat times with dynamic playback rate scaling.

This module provides a flexible timeline for beat-based events that can be scaled
to match a target cadence or playback rate.
"""

import numpy as np
from typing import Optional, Tuple


class BeatTimeline:
    """
    Manages a timeline of beat events with dynamic playback rate scaling.
    
    This class stores the original beat times from a beat tracker and provides
    methods to scale them according to a playback rate, find the current beat index,
    and calculate the next beat time.
    
    The scaling preserves the first beat position and scales all intervals by 1/rate.
    """
    
    def __init__(self, beat_times: np.ndarray):
        """
        Initialize the beat timeline with a sequence of beat times.
        
        Args:
            beat_times: Array of beat times in seconds (must be monotonically increasing)
        """
        if not isinstance(beat_times, np.ndarray) or len(beat_times) == 0:
            self._original_beats = np.array([], dtype=np.float64)
        else:
            # Store as float64 for precision in calculations
            self._original_beats = beat_times.astype(np.float64)
            
        self._rate: float = 1.0
        self._scaled_beats: Optional[np.ndarray] = None
        
    @property
    def original_beats(self) -> np.ndarray:
        """Get the original (unscaled) beat times."""
        return self._original_beats
        
    @property
    def scaled_beats(self) -> np.ndarray:
        """Get the scaled beat times based on current playback rate."""
        if self._scaled_beats is None:
            self._scaled_beats = self._compute_scaled_beats()
        return self._scaled_beats
        
    @property
    def rate(self) -> float:
        """Get the current playback rate."""
        return self._rate
        
    @rate.setter
    def rate(self, value: float):
        """
        Set the playback rate and invalidate cached scaled beats.
        
        Args:
            value: New playback rate (must be > 0)
        
        Raises:
            ValueError: If rate is <= 0
        """
        if value <= 0:
            raise ValueError("Playback rate must be positive")
            
        if value != self._rate:
            self._rate = value
            self._scaled_beats = None  # Invalidate cache
    
    def _compute_scaled_beats(self) -> np.ndarray:
        """
        Compute scaled beat times based on current rate.
        
        The scaling preserves the first beat position and scales all intervals by 1/rate.
        
        Returns:
            np.ndarray: Scaled beat times
        """
        if len(self._original_beats) == 0:
            return np.array([], dtype=np.float64)
            
        # Keep first beat fixed, scale intervals by 1/rate
        first_beat = self._original_beats[0]
        
        if len(self._original_beats) == 1:
            return np.array([first_beat], dtype=np.float64)
            
        intervals = np.diff(self._original_beats)
        scaled_intervals = intervals / self._rate
        
        # Construct scaled beats by cumulative sum of scaled intervals
        scaled_beats = np.zeros_like(self._original_beats)
        scaled_beats[0] = first_beat
        scaled_beats[1:] = first_beat + np.cumsum(scaled_intervals)
        
        return scaled_beats
    
    def index_at(self, time: float) -> int:
        """
        Find the index of the beat at or immediately before the given time.
        
        Args:
            time: Time position in seconds
            
        Returns:
            int: Index of the beat at or before the given time,
                 or -1 if time is before the first beat or no beats exist
        """
        if len(self.scaled_beats) == 0 or time < self.scaled_beats[0]:
            return -1
            
        # Binary search for the largest index where scaled_beats[index] <= time
        idx = np.searchsorted(self.scaled_beats, time, side='right') - 1
        return max(0, idx)
    
    def next_time_after(self, time: float) -> Tuple[float, int]:
        """
        Find the next beat time and index after the given time.
        
        Args:
            time: Current time position in seconds
            
        Returns:
            Tuple[float, int]: (next_beat_time, next_beat_index)
                               If no next beat exists, returns (float('inf'), -1)
        """
        if len(self.scaled_beats) == 0:
            return float('inf'), -1
            
        # Find the index of the first beat strictly after the given time
        idx = np.searchsorted(self.scaled_beats, time, side='right')
        
        if idx >= len(self.scaled_beats):
            return float('inf'), -1
            
        return self.scaled_beats[idx], idx
    
    def current_beat_index(self, time: float) -> int:
        """
        Get the current beat index at the given time.
        
        Args:
            time: Current time position in seconds
            
        Returns:
            int: Current beat index (1-based), or 0 if before first beat
        """
        idx = self.index_at(time)
        return idx + 1 if idx >= 0 else 0
    
    def beat_count(self) -> int:
        """
        Get the total number of beats in the timeline.
        
        Returns:
            int: Number of beats
        """
        return len(self._original_beats)
