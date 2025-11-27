"""
High-precision timer module for accurate music synchronization

This module implements microsecond-level timing precision for musical beat tracking
using performance counters and adaptive sleep algorithms. Key techniques include:
Implements microsecond-level timing precision using performance counters and adaptive sleep algorithms,
with thread-safe callbacks for accurate musical beat tracking.
"""

import time
import threading
from typing import Optional, Callable, Tuple, List
from dataclasses import dataclass

@dataclass
class BeatEvent:
    """Beat event data structure"""
    measure: int
    beat: int
    time_stamp: float
    is_strong_beat: bool

class PrecisionTimer:
    """High-precision timer for music synchronization
    
    Provides microsecond-level accuracy for beat tracking and synchronization.
    Uses performance counter as the authoritative time source.
    """
    
    def __init__(self):
        """Initialize precision timer"""
        self.tempo: float = 120.0    # Current BPM
        self.time_signature: Tuple[int, int] = (4, 4)  # (numerator, denominator)
        self.total_measures: int = 0  # Total number of measures in the piece
        
        # Internal timing state
        self.is_running: bool = False
        self.is_paused: bool = False
        self.base_time: float = 0.0
        self._beat_interval: float = 0.5  # Seconds per beat
        self._pause_position: float = 0.0
        self._last_beat_time: float = 0.0
        
        # Threading
        self._timer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Beat callbacks
        self._callbacks: List[Callable[[BeatEvent], None]] = []
        
        self._update_beat_interval()
    
    def _update_beat_interval(self):
        """Update beat interval when tempo changes
        
        Note: Always uses quarter-note tempo for MIDI playback compatibility.
        Time signature adjustments are handled separately in beat callback logic.
        """
        self._beat_interval = 60.0 / self.tempo
    
    @staticmethod
    def calculate_musical_beat_multiplier(time_signature: Tuple[int, int]) -> float:
        """Calculate the multiplier to convert quarter-note tempo to musical beat tempo
        
        Args:
            time_signature: Time signature tuple (numerator, denominator)
        
        Returns:
            float: Multiplier for converting quarter-note BPM to musical beat BPM
        """
        numerator, denominator = time_signature
        
        # Determine the musical beat unit based on time signature patterns
        if denominator == 8:
            # Eighth-note based time signatures
            if numerator in [6, 9, 12]:
                # Compound time signatures (6/8, 9/8, 12/8)
                # Musical beat = dotted quarter note = 1.5 quarter notes
                # So musical beat tempo = quarter_note_tempo / 1.5
                return 1.0 / 1.5
            elif numerator in [2, 3, 4]:
                # Simple time signatures (2/8, 3/8, 4/8)
                # Musical beat = eighth note = 0.5 quarter notes  
                # So musical beat tempo = quarter_note_tempo / 0.5
                return 1.0 / 0.5
            else:
                # Other 8th-note time signatures - default to eighth note beat
                return 1.0 / 0.5
        
        elif denominator == 16:
            # Sixteenth-note based time signatures  
            if numerator in [6, 9, 12]:
                # Compound sixteenth time (6/16, 9/16, 12/16) 
                # Musical beat = dotted eighth note = 0.75 quarter notes
                return 1.0 / 0.75
            else:
                # Simple sixteenth time - musical beat = sixteenth note = 0.25 quarter notes
                return 1.0 / 0.25
        
        elif denominator == 2:
            # Half-note based time signatures (2/2, 3/2, 4/2)
            # Musical beat = half note = 2 quarter notes
            return 1.0 / 2.0
        
        else:
            # Quarter-note based time signatures (4/4, 3/4, 2/4) and others
            # Musical beat = quarter note = 1 quarter note (no change)
            return 1.0
    
    def _calculate_musical_beat_multiplier(self) -> float:
        """Calculate the multiplier to convert quarter-note tempo to musical beat tempo
        
        Returns:
            float: Multiplier for converting quarter-note BPM to musical beat BPM
        """
        return self.calculate_musical_beat_multiplier(self.time_signature)
    
    @staticmethod
    def calculate_musical_beats_per_measure(numerator: int, denominator: int) -> int:
        """Calculate the actual number of musical beats per measure
        
        This is different from the numerator for compound time signatures.
        For example, 6/8 has 2 musical beats per measure (dotted quarters),
        not 6 beats as the numerator suggests.
        
        Args:
            numerator: Time signature numerator
            denominator: Time signature denominator
            
        Returns:
            int: Number of musical beats per measure
        """
        if denominator == 8:
            # Eighth-note based time signatures
            if numerator in [6, 9, 12]:
                # Compound time signatures (6/8, 9/8, 12/8)
                # Musical beats are dotted quarter notes
                # 6/8 = 2 dotted quarters, 9/8 = 3 dotted quarters, 12/8 = 4 dotted quarters
                return numerator // 3
            elif numerator in [2, 3, 4]:
                # Simple time signatures (2/8, 3/8, 4/8)
                # Musical beats are eighth notes - use numerator
                return numerator
            else:
                # Other 8th-note time signatures - default to eighth note beats
                return numerator
        
        elif denominator == 16:
            # Sixteenth-note based time signatures  
            if numerator in [6, 9, 12]:
                # Compound sixteenth time (6/16, 9/16, 12/16) 
                # Musical beats are dotted eighth notes
                return numerator // 3
            else:
                # Simple sixteenth time - musical beats are sixteenth notes
                return numerator
        
        elif denominator == 2:
            # Half-note based time signatures (2/2, 3/2, 4/2)
            # Musical beats are half notes - use numerator
            return numerator
        
        else:
            # Quarter-note based time signatures (4/4, 3/4, 2/4) and others
            # Musical beats are quarter notes - use numerator
            return numerator

    def _calculate_musical_beats_per_measure(self, numerator: int, denominator: int) -> int:
        """Calculate the actual number of musical beats per measure (instance method wrapper)
        
        Delegates to the static method for consistency.
        """
        return self.calculate_musical_beats_per_measure(numerator, denominator)
    
    @staticmethod
    def convert_midi_tempo_to_musical_tempo(midi_tempo: float, time_signature: Tuple[int, int]) -> float:
        """Convert MIDI tempo (quarter-note based) to musical tempo based on time signature
        
        Args:
            midi_tempo: MIDI tempo in quarter notes per minute
            time_signature: Time signature tuple (numerator, denominator)
            
        Returns:
            float: Musical tempo that matches actual beat frequency
        """
        multiplier = PrecisionTimer.calculate_musical_beat_multiplier(time_signature)
        return midi_tempo * multiplier
    
    @staticmethod
    def convert_musical_tempo_to_midi_tempo(musical_tempo: float, time_signature: Tuple[int, int]) -> float:
        """Convert musical tempo back to MIDI tempo (quarter-note based)
        
        Args:
            musical_tempo: Musical tempo that users hear/set
            time_signature: Time signature tuple (numerator, denominator)
            
        Returns:
            float: MIDI tempo in quarter notes per minute
        """
        multiplier = PrecisionTimer.calculate_musical_beat_multiplier(time_signature)
        return musical_tempo / multiplier
    
    def set_tempo(self, bpm: float):
        """Set tempo with immediate effect and improved continuity guarantee
        
        Args:
            bpm: Beats per minute
        """
        if bpm == self.tempo:
            return  # No change needed
        
        old_tempo = self.tempo
        self.tempo = max(20.0, min(300.0, bpm))
        old_beat_interval = self._beat_interval
        self._update_beat_interval()
        
        # Enhanced continuity preservation for running timer
        if self.is_running and not self.is_paused:
            # Use higher precision calculation to minimize drift
            current_time_high_precision = time.perf_counter()
            elapsed_time = current_time_high_precision - self.base_time
            
            # Calculate exact beat position with old tempo
            exact_beat_position = elapsed_time / old_beat_interval
            
            # Recalculate base_time to maintain exact same beat position
            target_elapsed_time = exact_beat_position * self._beat_interval
            self.base_time = current_time_high_precision - target_elapsed_time
            
            print(f"Tempo change: {old_tempo:.1f}→{self.tempo:.1f} BPM, continuity preserved at beat {exact_beat_position:.6f}")
        
        # Notify about precision drift monitoring
        if hasattr(self, '_drift_monitor_enabled') and self._drift_monitor_enabled:
            self._tempo_change_count += 1
    
    def set_time_signature(self, numerator: int, denominator: int):
        """Set time signature for beat position calculations
        
        Args:
            numerator: Beats per measure
            denominator: Note value per beat
        
        Note: This affects beat position calculations but not the core timing interval,
        which remains quarter-note based for MIDI playback compatibility.
        """
        old_signature = self.time_signature
        self.time_signature = (numerator, denominator)
        
        # Log the change if running
        if self.is_running or self.is_paused:
            print(f"MIDI engine time signature changed: {old_signature[0]}/{old_signature[1]} → {numerator}/{denominator}")
            print(f"Note: MIDI playback timing remains quarter-note based, time signature affects beat callbacks only")
    
    def set_total_measures(self, total_measures: int):
        """Set total number of measures in the piece
        
        Args:
            total_measures: Total measures count
        """
        self.total_measures = total_measures
        print(f"Total measures set to: {total_measures}")
    
    def start(self, start_position: float = 0.0):
        """Start high-precision timer with precise fractional beat handling
        
        Args:
            start_position: Starting position in beats (can be fractional)
        """
        if self.is_running:
            return
        
        # Calculate the base_time normally
        self.base_time = time.perf_counter() - (start_position * self._beat_interval)
        self.is_running = True
        self.is_paused = False
        self._stop_event.clear()
        
        # Log the precise starting position (for debugging)
        print(f"[PrecisionTimer] Starting at position {start_position:.3f} beats")
        print(f"  - Fractional part: {start_position - int(start_position):.3f} beats")
        print(f"  - Calculating exact timing for fractional adjustment")
        
        # Add an audible verification that the fractional part is being applied
        # This will help confirm if the adjustment is working
        fractional_part = start_position - int(start_position)
        if 0.001 < fractional_part < 0.999:  # If there's a meaningful fractional part
            print(f"  - VERIFICATION: Fractional position {fractional_part:.3f} will shift timing")
        
        # Store initial position as special attribute for the timer thread
        # This tells the thread where we're starting (with the fractional component)
        self._initial_position = start_position
        
        # Start high-precision timer thread
        self._timer_thread = threading.Thread(target=self._precision_loop, daemon=True)
        self._timer_thread.start()
    
    def pause(self):
        """Pause timer while maintaining position"""
        if not self.is_running or self.is_paused:
            return
        
        # Calculate current position BEFORE setting paused state to avoid circular dependency
        elapsed_seconds = time.perf_counter() - self.base_time
        current_position = elapsed_seconds / self._beat_interval
        
        # Now set paused state and store position
        self.is_paused = True
        self._pause_position = current_position
    
    def resume(self):
        """Resume timer from paused position"""
        if not self.is_running or not self.is_paused:
            return
        
        # Restore position: convert beats back to seconds for base_time calculation
        pause_time_seconds = self._pause_position * self._beat_interval
        self.base_time = time.perf_counter() - pause_time_seconds
        self.is_paused = False
    
    def stop(self):
        """Stop timer"""
        self.is_running = False
        self.is_paused = False
        self._stop_event.set()
        
        if self._timer_thread and self._timer_thread.is_alive():
            self._timer_thread.join(timeout=0.1)
    
    def get_precise_time(self) -> float:
        """Get current precise time in beats
        
        Returns:
            float: Current time in beats from start
        """
        if not self.is_running:
            return 0.0
        
        if self.is_paused:
            return self._pause_position
        
        elapsed_seconds = time.perf_counter() - self.base_time
        return elapsed_seconds / self._beat_interval
    
    def get_beat_position(self) -> BeatEvent:
        """Get current beat position
        
        Returns:
            BeatEvent: Current beat information
        """
        current_time = self.get_precise_time()
        
        numerator, denominator = self.time_signature
        
        # Calculate actual musical beats per measure based on time signature
        # This is different from the numerator for compound time signatures
        musical_beats_per_measure = self._calculate_musical_beats_per_measure(numerator, denominator)
        
        # Convert quarter-note time to musical beat time
        musical_beat_multiplier = self._calculate_musical_beat_multiplier()
        musical_time = current_time * musical_beat_multiplier
        
        # Calculate measure and beat position
        measure = int(musical_time // musical_beats_per_measure)
        beat = int(musical_time % musical_beats_per_measure)
        
        is_strong_beat = (beat == 0)
        
        return BeatEvent(measure, beat, current_time, is_strong_beat)
    
    def add_beat_callback(self, callback: Callable[[BeatEvent], None]):
        """Add callback for beat events
        
        Args:
            callback: Function to call on each beat
        """
        self._callbacks.append(callback)
    
    def remove_beat_callback(self, callback: Callable[[BeatEvent], None]):
        """Remove beat callback
        
        Args:
            callback: Function to remove
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def _precision_loop(self):
        """High-precision timer loop with time-signature-aware beat callbacks
        
        Uses adaptive timing to maintain microsecond-level accuracy.
        Fires beat callbacks at time-signature-appropriate intervals.
        """
        import math
        
        # Initialize tracking variables
        last_musical_beat = -1  # Track last processed musical beat
        
        # Calculate musical beat multiplier for callback timing
        musical_beat_multiplier = self._calculate_musical_beat_multiplier()
        
        # Handle the initial fractional position (if any)
        if hasattr(self, '_initial_position'):
            initial_position = self._initial_position
            
            # Calculate the initial musical beat we're starting at
            initial_musical_time = initial_position * musical_beat_multiplier
            # Calculate the next integer musical beat
            next_musical_beat = math.ceil(initial_musical_time)
            
            # Check if we're starting at a fractional position (not exactly on a beat boundary)
            if initial_musical_time != next_musical_beat and next_musical_beat > math.floor(initial_musical_time):
                # Calculate time needed to reach the next integer musical beat
                time_to_next_beat = (next_musical_beat - initial_musical_time) / musical_beat_multiplier
                wait_seconds = time_to_next_beat * self._beat_interval
                
                # Only wait if there's a significant fraction to wait for
                if wait_seconds > 0.001:  # More than 1ms
                    print(f"  - Starting at fractional position {initial_musical_time:.3f} musical beats")
                    print(f"  - Waiting {wait_seconds:.3f}s to reach next beat boundary at {next_musical_beat}")
                    print(f"  - VERIFICATION: Beat timing will shift by {wait_seconds:.3f}s due to fractional position")
                    time.sleep(wait_seconds)
                else:
                    print(f"  - Fractional position too small, no timing adjustment needed")
            
            # Initialize last_musical_beat based on where we are now
            current_time = self.get_precise_time()
            current_musical_time = current_time * musical_beat_multiplier
            last_musical_beat = math.floor(current_musical_time)
            
            # Remove the attribute to avoid recalculating on resume
            delattr(self, '_initial_position')
        
        # Main timer loop
        while not self._stop_event.is_set() and self.is_running:
            if self.is_paused:
                time.sleep(0.001)
                continue
            
            # Get current time in quarter-note beats (for MIDI playback)
            current_quarter_note_time = self.get_precise_time()
            
            # Calculate musical beat time for metronome callbacks
            current_musical_time = current_quarter_note_time * musical_beat_multiplier
            current_musical_beat = math.floor(current_musical_time)  # Use floor for consistent negative handling
            
            # Fire beat callbacks at musical beat intervals (not quarter-note intervals)
            if current_musical_beat > last_musical_beat:
                # Create beat event with quarter-note timing (for position info)
                # but fire at musical beat intervals
                beat_event = self.get_beat_position()
                
                # Fire callbacks for this musical beat
                for callback in self._callbacks:
                    try:
                        callback(beat_event)
                    except Exception as e:
                        print(f"Error in beat callback: {e}")
                
                # Update last processed musical beat
                last_musical_beat = current_musical_beat
            
            # Adaptive sleep for optimal CPU usage vs accuracy
            # Base sleep timing on musical beats for responsive metronome
            time_to_next_musical_beat = (current_musical_beat + 1) - current_musical_time
            musical_beat_interval = self._beat_interval / musical_beat_multiplier
            time_to_next_musical_beat_seconds = time_to_next_musical_beat * musical_beat_interval
            
            if time_to_next_musical_beat_seconds > 0.1:
                time.sleep(0.01)  # Longer sleep when far from next beat
            elif time_to_next_musical_beat_seconds > 0.01:
                time.sleep(0.001)  # Shorter sleep when approaching beat
            else:
                time.sleep(0.0001)  # Microsecond precision near beat
    
    def sync_to_position(self, position_beats: float):
        """Synchronize timer to specific position with enhanced precision
        
        Args:
            position_beats: Target position in beats
        """
        if not self.is_running:
            return
        
        # High-precision synchronization
        current_time = time.perf_counter()
        target_elapsed_time = position_beats * self._beat_interval
        self.base_time = current_time - target_elapsed_time
        
        print(f"Timer synchronized to beat position {position_beats:.6f}")
    
    def enable_drift_monitoring(self, enabled: bool = True):
        """Enable drift monitoring for debugging synchronization issues
        
        Args:
            enabled: Whether to enable drift monitoring
        """
        self._drift_monitor_enabled = enabled
        if enabled:
            self._tempo_change_count = 0
            self._drift_history = []
            print("PrecisionTimer drift monitoring enabled")
        else:
            print("PrecisionTimer drift monitoring disabled")
    
    def get_synchronization_status(self) -> dict:
        """Get detailed synchronization status for debugging
        
        Returns:
            dict: Synchronization status information
        """
        current_time = self.get_precise_time()
        system_time = time.perf_counter()
        
        status = {
            'timer_running': self.is_running,
            'timer_paused': self.is_paused,
            'current_beats': current_time,
            'base_time': self.base_time,
            'system_time': system_time,
            'beat_interval': self._beat_interval,
            'tempo': self.tempo,
            'time_signature': self.time_signature
        }
        
        if hasattr(self, '_drift_monitor_enabled') and self._drift_monitor_enabled:
            status['tempo_changes'] = getattr(self, '_tempo_change_count', 0)
            status['drift_monitoring'] = True
        
        return status
    
    def get_seconds_per_beat(self) -> float:
        """Get current seconds per beat
        
        Returns:
            float: Seconds per beat
        """
        return self._beat_interval
    
    def get_position_info(self) -> dict:
        """Get comprehensive position information
        
        Returns:
            dict: Position information including beats, measure, seconds
        """
        if not self.is_running and not self.is_paused:
            return {
                'beats': 0.0,
                'measure': 0,
                'beat': 0,
                'is_strong_beat': True,
                'tempo': self.tempo,
                'seconds': 0.0,
                'total_measures': self.total_measures
            }
        
        beat_event = self.get_beat_position()
        current_beats = self.get_precise_time()
        current_seconds = current_beats * self._beat_interval
        
        return {
            'beats': current_beats,
            'measure': beat_event.measure,
            'beat': beat_event.beat,
            'is_strong_beat': beat_event.is_strong_beat,
            'tempo': self.tempo,
            'seconds': current_seconds,
            'total_measures': self.total_measures
        } 