import time
import statistics
import threading
import numpy as np
from typing import Dict, List, Tuple, Optional
from .metronome import BaseMetronome
from .precision_timer import PrecisionTimer
from .playback_mode import PlaybackMode
from .beat_timeline import BeatTimeline

class RASTherapyMetronome(BaseMetronome):
    """Simplified metronome for RAS (Rhythmic Auditory Stimulation) therapy
    
    Features:
    - Step-synchronized beat cueing (each beat sound = one step)
    - No strong/weak beat distinction (only one sound type)
    - Time signature-based beat pattern only (no note density rules)
    - Ultra-low latency synthesized audio
    """
    
    def __init__(self, engine):
        """Initialize RAS therapy metronome
        
        Args:
            engine: MIDI engine instance with precision timer
        """
        super().__init__()
        self.engine = engine
        self._beat_callback_registered = False
        self._last_cue_beat = -1  # Track last processed beat to avoid duplicates
        
        # Beat cueing configuration
        self.cue_pattern: List[bool] = []  # Which beats in measure to cue
        self.beats_per_measure: int = 4
        self.step_frequency: float = 60.0  # Steps per minute
        self.RAS_cue: float = 60.0  # same as step frequency
        
        # Beat track for performance MIDIs (DYNAMIC mode only)
        self.beat_times: Optional[np.ndarray] = None
        self.beat_timeline: Optional[BeatTimeline] = None
        self.playback_rate: float = 1.0
        self.current_beat_index: int = 0
        
        # Performance monitoring
        self._latency_monitor_enabled = False
        self._cue_timestamps = []
        self._max_timestamp_history = 100
        
        # Initialize with only strong beat sound (no weak beats)
        self._init_ras_sound()
        
        # Print initialization info
        latency_info = self.get_latency_info()
        print(f"RAS Therapy Metronome initialized with {latency_info['theoretical_latency_ms']:.1f}ms latency")
    
    def _init_ras_sound(self):
        """Initialize RAS-specific sound (only one sound type)"""
        # Use only one sound for RAS (no weak beat distinction)
        self.sound_type = "click"  # Default to click sound
        self._create_sound_objects()
        
        # Set volume to ensure consistent step cueing
        self.set_volume(1.0)
    
    def _convert_midi_tempo_to_musical_tempo(self, midi_tempo: float) -> float:
        """Convert MIDI tempo (quarter-note based) to musical tempo based on time signature
            Uses the centralized conversion logic from PrecisionTimer for consistency.
            Args:
                midi_tempo: MIDI tempo in quarter notes per minute
                
            Returns:
                float: Musical tempo that matches actual beat frequency
        """
        if not self.engine.midi_file:
            return midi_tempo  # No conversion if no file loaded
        
        metadata = self.engine.get_metadata()
        time_sig = metadata.get('time_signature', {})
        
        if not isinstance(time_sig, dict) or 'numerator' not in time_sig or 'denominator' not in time_sig:
            return midi_tempo  # No conversion if no time signature info
        
        # Use centralized conversion logic from PrecisionTimer
        time_signature_tuple = (time_sig['numerator'], time_sig['denominator'])
        self.RAS_cue = PrecisionTimer.convert_midi_tempo_to_musical_tempo(midi_tempo, time_signature_tuple)
        return self.RAS_cue

    def _calculate_cue_pattern(self, numerator: int, denominator: int) -> List[bool]:
        """Calculate beat cueing pattern based on time signature using PrecisionTimer's beat rules,
            Uses PrecisionTimer.calculate_musical_beats_per_measure() for consistency.
            For example, 6/8 has 2 musical beats (dotted quarters), not 6 beats.
            
            Args:
                numerator: Time signature numerator
                denominator: Time signature denominator
                
            Returns:
                List[bool]: Boolean array indicating which beats to cue
        """
        # Use PrecisionTimer's static method for consistent musical beat calculation
        musical_beats = PrecisionTimer.calculate_musical_beats_per_measure(numerator, denominator)
        
        # Unified rule: Each musical beat corresponds to one step
        pattern = [True] * musical_beats
        
        if musical_beats != numerator:
            print(f"  Note: Using {musical_beats} musical beats instead of {numerator} subdivided beats")
        
        return pattern
    
    def _calculate_step_frequency(self, musical_tempo: float, cue_pattern: List[bool]) -> float:
        """Calculate step frequency based on musical tempo (simplified)
        
        Since musical tempo equals cadence in our RAS system design,
        step frequency directly equals musical tempo.
        
        Args:
            musical_tempo: Musical tempo in BPM (already converted from MIDI tempo)
            cue_pattern: Beat cueing pattern (for validation only)
            
        Returns:
            float: Step frequency in steps per minute
        """
        # Verify 1:1 cueing pattern (every beat has a cue)
        cues_per_measure = sum(cue_pattern) if cue_pattern else 0
        beats_per_measure = len(cue_pattern) if cue_pattern else 0
        
        if cues_per_measure == 0:
            return 0.0
        
        # In RAS therapy: musical tempo = cadence directly
        step_frequency = musical_tempo
        print(f"Step frequency: {step_frequency:.1f} steps/min; {cues_per_measure} cues per measure")
        
        return step_frequency
    
    def _is_playback_finished(self) -> bool:
        """Check if playback has reached or passed the end of the MIDI file.
        
        Returns:
            bool: True if playback has finished, False otherwise
        """
        try:
            # If timer is not running, consider playback finished
            if not hasattr(self.engine, 'precision_timer') or not self.engine.precision_timer.is_running:
                return True
            
            # If no MIDI file is loaded, cannot determine
            if not hasattr(self.engine, 'midi_file') or self.engine.midi_file is None:
                return False
            
            # Get current playback position
            position_info = self.engine.precision_timer.get_position_info()
            current_sec = position_info.get('seconds', 0.0)
            
            # Get total duration of MIDI file
            total_sec = float(self.engine.midi_file.get_end_time())
            
            # Use small epsilon to account for timing precision
            epsilon = 0.02  # 20ms tolerance
            return current_sec >= (total_sec - epsilon)
            
        except Exception:
            # On any error, don't force stop (conservative approach)
            return False
    
    def start(self):
        """Start RAS therapy metronome with time signature-based pattern"""
        if not hasattr(self.engine, 'precision_timer'):
            print("Error: Engine does not have precision timer")
            return
        
        # Get time signature and tempo from current section if available, otherwise from MIDI file
        if self.engine.midi_file:
            # Check if we're in section mode and have current section
            if hasattr(self.engine, 'current_section') and self.engine.current_section:
                section = self.engine.current_section
                time_sig = section.get('time_signature', {'numerator': 4, 'denominator': 4})
                midi_tempo = section.get('tempo', 120.0)
                # print(f"RAS: Using section-specific parameters from '{section.get('label', 'Current Section')}'")
            else:
                # Fallback to global metadata
                metadata = self.engine.get_metadata()
                time_sig = metadata.get('time_signature', {})
                midi_tempo = getattr(self.engine, 'tempo', 120.0)
            
            if isinstance(time_sig, dict) and 'numerator' in time_sig and 'denominator' in time_sig:
                numerator = time_sig['numerator']
                denominator = time_sig['denominator']
                
                # Convert MIDI tempo to musical tempo for step frequency calculation
                tempo = self._convert_midi_tempo_to_musical_tempo(midi_tempo)
                
                # Calculate cue pattern (simplified - no note density check)
                self.cue_pattern = self._calculate_cue_pattern(numerator, denominator)
                self.beats_per_measure = numerator
                
                # Calculate step frequency
                self.step_frequency = self._calculate_step_frequency(tempo, self.cue_pattern)
                
                # Set time signature for base metronome
                self.set_beat(numerator, denominator)
                
        # Register precision beat callback
        if not self._beat_callback_registered:
            self.engine.precision_timer.add_beat_callback(self._ras_beat_callback)
            self._beat_callback_registered = True
        
        # CRITICAL: Activate metronome so callbacks will play sounds
        self.is_active = True
            
    def stop(self):
        """Stop RAS therapy metronome"""
        # Deactivate metronome first
        self.is_active = False
        
        # Unregister beat callback
        if self._beat_callback_registered and hasattr(self.engine, 'precision_timer'):
            self.engine.precision_timer.remove_beat_callback(self._ras_beat_callback)
            self._beat_callback_registered = False
        
        self._last_cue_beat = -1  # Reset beat tracking
        self._cue_timestamps.clear()  # Clear latency monitoring data
    
    def _ras_beat_callback(self, beat_event):
        """RAS therapy beat callback with time signature-based cueing pattern
        
        Args:
            beat_event: BeatEvent with precise timing information
        """
        # Only play if metronome is active and engine is playing
        if not self.is_active or not self.engine.is_playing:
            return
        
        # Check if playback has finished and stop metronome automatically
        if self._is_playback_finished():
            self.stop()
            return

        # Use metadata-based timing for both STANDARD and DYNAMIC modes
        try:
            # Check if this beat should be cued according to pattern
            if self.cue_pattern and beat_event.beat < len(self.cue_pattern):
                should_cue = self.cue_pattern[beat_event.beat]

                if should_cue:
                    # Check if we've already processed this beat to avoid duplicates
                    current_beat_id = beat_event.measure * self.beats_per_measure + beat_event.beat

                    if current_beat_id != self._last_cue_beat:
                        # Record timestamp for latency monitoring
                        if self._latency_monitor_enabled:
                            cue_time = time.perf_counter()
                            self._cue_timestamps.append(cue_time)

                            # Keep only recent timestamps
                            if len(self._cue_timestamps) > self._max_timestamp_history:
                                self._cue_timestamps = self._cue_timestamps[-self._max_timestamp_history:]

                        # Play the cue (always using the single beat sound)
                        self.beat_sound.play()
                        self._last_cue_beat = current_beat_id

        except Exception as e:
            print(f"Error in RAS therapy metronome: {e}")
    
    def get_cueing_info(self) -> dict:
        """Get current cueing configuration info for RAS therapy
        
        Returns:
            dict: Information about current cueing mode, step frequency, and configuration
        """
        cues_per_measure = sum(self.cue_pattern) if self.cue_pattern else 0
        
        info = {
            'time_signature': f"{self.beats_per_measure}/{self.beat_denominator}",
            'cueing_mode': "step-synchronized",
            'cue_pattern': self.cue_pattern,
            'cues_per_measure': cues_per_measure,
            'step_frequency': self.step_frequency,
            'audio_system': 'synthesized_single_sound'
        }
        
        # Add playback mode info
        if hasattr(self.engine, 'session_state'):
            info['playback_mode'] = str(self.engine.session_state.mode)
            
            if self.engine.session_state.mode == PlaybackMode.DYNAMIC:
                if self.beat_times is not None:
                    info['beat_track_count'] = len(self.beat_times)
                else:
                    info['beat_track_count'] = 0
            elif self.engine.session_state.mode == PlaybackMode.STANDARD:
                # This space is intentionally left blank as the old timing_offset_sec is removed
                pass
        
        # Add latency information
        latency_info = self.get_latency_info()
        info.update(latency_info)
        
        if hasattr(self.engine, 'tempo'):
            info['musical_tempo'] = self.engine.tempo
            
            # Simple rule description
            rule_applied = f"time-signature-based ({cues_per_measure} cues/measure)"
            
            info['rule_applied'] = rule_applied
            info['description'] = (f"Step-synchronized cueing: {self.step_frequency:.1f} steps/min "
                                 f"(Rule: {rule_applied}, "
                                 f"Tempo: {self.engine.tempo:.1f} BPM, "
                                 f"Latency: {latency_info['theoretical_latency_ms']:.1f}ms)")
        else:
            info['description'] = f"Step-synchronized cueing ({latency_info['theoretical_latency_ms']:.1f}ms latency)"
        
        return info
    
    def toggle(self):
        """Toggle RAS therapy metronome on/off"""
        self.is_active = not self.is_active
        
        # if self.is_active:
        #     cueing_info = self.get_cueing_info()
        #     print(f"RAS therapy metronome enabled: {cueing_info['description']}")
    
    def enable_latency_monitoring(self, enabled: bool = True):
        """Enable/disable latency monitoring for performance analysis
        
        Args:
            enabled: Whether to enable latency monitoring
        """
        self._latency_monitor_enabled = enabled
        if enabled:
            self._cue_timestamps.clear()
        else:
            print("RAS metronome latency monitoring disabled")
    
    def get_latency_statistics(self) -> dict:
        """Get latency statistics from recent cues
        
        Returns:
            dict: Latency statistics
        """
        if not self._cue_timestamps or len(self._cue_timestamps) < 2:
            return {'error': 'Insufficient data for latency analysis'}
        
        # Calculate inter-cue intervals
        intervals = []
        for i in range(1, len(self._cue_timestamps)):
            interval = self._cue_timestamps[i] - self._cue_timestamps[i-1]
            intervals.append(interval * 1000)  # Convert to ms
        
        if not intervals:
            return {'error': 'No interval data available'}
        
        return {
            'total_cues': len(self._cue_timestamps),
            'mean_interval_ms': statistics.mean(intervals),
            'median_interval_ms': statistics.median(intervals),
            'stdev_interval_ms': statistics.stdev(intervals) if len(intervals) > 1 else 0,
            'min_interval_ms': min(intervals),
            'max_interval_ms': max(intervals),
            'step_frequency': self.step_frequency,
            'theoretical_latency_ms': self.get_latency_info()['theoretical_latency_ms']
        }
    
    def update_configuration(self):
        """Update RAS configuration when MIDI file or settings change (section-aware)"""
        if self.engine.midi_file:            
            # Check if we're in section mode and have current section
            if hasattr(self.engine, 'current_section') and self.engine.current_section:
                section = self.engine.current_section
                time_sig = section.get('time_signature', {'numerator': 4, 'denominator': 4})
                midi_tempo = section.get('tempo', 120.0)
            else:
                # Fallback to global metadata
                metadata = self.engine.get_metadata()
                time_sig = metadata.get('time_signature', {})
                midi_tempo = getattr(self.engine, 'tempo', 120.0)
                # print("RAS: Updating with global file parameters")
            
            # Convert MIDI tempo to musical tempo for step frequency calculation
            musical_tempo = self._convert_midi_tempo_to_musical_tempo(midi_tempo)
            
            if isinstance(time_sig, dict) and 'numerator' in time_sig and 'denominator' in time_sig:
                numerator = time_sig['numerator']
                denominator = time_sig['denominator']
                
                # calculate cue pattern for RAS beat cueing
                self.cue_pattern = self._calculate_cue_pattern(numerator, denominator)
                self.beats_per_measure = numerator
                
                # Update step frequency
                self.step_frequency = self._calculate_step_frequency(musical_tempo, self.cue_pattern)
                
                # print(f"RAS configuration updated:")
                # print(f"  Time signature: {numerator}/{denominator}")
                # print(f"  Musical tempo: {musical_tempo:.1f} BPM")
                # print(f"  Step frequency: {self.step_frequency:.1f} steps/min")
                
                # If currently active, restart to apply changes
                was_active = self.is_active
                if was_active:
                    self.stop()
                    self.start()
    
    def set_playback_rate(self, rate: float):
        """Set playback rate for beat timeline scaling
        
        Args:
            rate: Playback rate factor (1.0 = normal speed)
        """
        if rate <= 0:
            print(f"Invalid playback rate: {rate}")
            return
            
        self.playback_rate = rate
        
        # Update beat timeline if we have one
        if self.beat_timeline:
            self.beat_timeline.rate = rate
            print(f"Beat timeline rate set to {rate:.3f}")
        
    def set_beat_track(self, beat_times: np.ndarray):
        """Set beat track for performance MIDIs (DYNAMIC mode only)
        
        Args:
            beat_times: Array of beat times in seconds
        """
        if beat_times is None or len(beat_times) == 0:
            print("Cannot set empty beat track")
            return
            
        self.beat_times = beat_times.copy()
        
        # Create or update beat timeline
        self.beat_timeline = BeatTimeline(beat_times)
        
        # Apply current playback rate if not 1.0
        if self.playback_rate != 1.0:
            self.beat_timeline.rate = self.playback_rate
            
        print(f"Beat track set with {len(beat_times)} beats")
        
    def clear_beat_track(self):
        """Clear the beat track and return to standard mode"""
        self.beat_times = None
        self.beat_timeline = None
        self.current_beat_index = 0
        print("Beat track cleared")
    
    def get_current_beat_number(self) -> int:
        """Get the current beat number (1-based) for display in UI
        
        This method calculates the current beat in real-time based on the 
        current playback position, ensuring accurate display even between callbacks.
        
        Returns:
            int: Current beat number (1-based), or 0 if not available
        """
        if self.engine.session_state.mode == PlaybackMode.DYNAMIC and self.beat_timeline:
            # Get current playback time from precision timer
            if hasattr(self.engine, 'precision_timer') and self.engine.precision_timer.is_running:
                position_info = self.engine.precision_timer.get_position_info()
                playback_time = position_info['seconds']
                
                # Find the beat at or immediately before the current time
                current_idx = self.beat_timeline.index_at(playback_time)
                
                # Update the stored index for other methods to use
                if current_idx >= 0:
                    self.current_beat_index = current_idx
                    # Return 1-based index for display
                    return current_idx + 1
                
            # Fallback to stored index if we can't calculate
            return self.current_beat_index + 1 if self.current_beat_index >= 0 else 0
        return 0