import threading
import time
from typing import List, Tuple, Optional
import mido
from .track_activity_monitor import BeatBasedActivityMonitor

class EventScheduler:
    """Event scheduler class responsible for precise MIDI event scheduling
    
    Updated to use PrecisionTimer as the authoritative time source
    to eliminate dual clock domain synchronization drift issues.
    
    Now includes integrated beat-based track activity monitoring for stable visualization.
    """
    
    def __init__(self, engine):
        """Initialize event scheduler
        
        Args:
            engine: MIDI engine instance
        """
        self.engine = engine
        self.scheduler_thread: Optional[threading.Thread] = None
        self.should_stop = threading.Event()
        self.events: List[Tuple[float, mido.Message, int]] = []  # [(time, message, track_index), ...]
        
        # Beat-based track activity monitoring for visualization
        self.activity_monitor = BeatBasedActivityMonitor()
        
        # Beat tracking for activity monitoring
        self._last_beat_number = -1
        
        # Anacrusis offset for event time adjustment (beats)
        self.timing_offset_beats = 0.0
            
    def start(self):
        """Start scheduler thread using PrecisionTimer as the authoritative time source"""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            return
        
        # Reset beat tracking to ensure activity monitoring starts from current position
        if hasattr(self.engine, 'precision_timer') and self.engine.precision_timer.is_running:
            current_beat = self.engine.precision_timer.get_precise_time()
            self._last_beat_number = int(current_beat) - 1  # Ensure first beat triggers update
        else:
            # If timer not running yet, reset to -1 (will be set when loop starts)
            self._last_beat_number = -1
        
        self.should_stop.clear()
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
    
    def stop(self):
        """Stop scheduler thread"""
        self.should_stop.set()
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=1.0)

    def initialize_activity_monitoring(self, tracks_info: List[dict]):
        """Initialize activity monitoring for loaded MIDI file
        
        Args:
            tracks_info: List of track information from engine metadata
        """
        # Pass MIDI file to activity monitor for preprocessing
        midi_file = self.engine.midi_file if hasattr(self.engine, 'midi_file') else None
        self.activity_monitor.initialize_tracks(tracks_info, midi_file)
        
        # Set current tempo in activity monitor
        if hasattr(self.engine, 'tempo'):
            self.activity_monitor.set_tempo(self.engine.tempo)
        
        # print(f"EventScheduler: Beat-based activity monitoring initialized for {len(tracks_info)} tracks")
    
    def add_activity_callback(self, callback):
        """Add callback for track activity changes
        
        Args:
            callback: Callback function for activity updates
        """
        self.activity_monitor.add_callback(callback)
    
    def remove_activity_callback(self, callback):
        """Remove callback for track activity changes
        
        Args:
            callback: Callback function to remove
        """
        self.activity_monitor.remove_callback(callback)
    
    def get_track_activity(self, track_id: int):
        """Get current activity state for a track
        
        Args:
            track_id: Track index
            
        Returns:
            TrackActivityState or None
        """
        return self.activity_monitor.get_track_activity(track_id)
    
    def get_all_track_activities(self):
        """Get activity states for all tracks
        
        Returns:
            Dictionary mapping track_id to TrackActivityState
        """
        return self.activity_monitor.get_all_activities()
    
    def prepare_section(self, section=None):
        """Prepare scheduler for a specific section but don't start it yet
        
        This method sets up the section boundaries without starting the scheduler thread.
        It allows decoupling the section preparation from the actual playback start.
        """
        if section is None:
            return
        self.current_section = section
        # Just prepare but don't start thread yet
    
    def set_timing_offset(self, offset_beats: float):
        """Set timing offset for MIDI event time adjustment
        
        In the single-clock architecture, this offset is applied to all MIDI event times:
        adjusted_event_time = original_event_time + offset_beats
        
        Args:
            offset_beats: Anacrusis offset in beats (positive value delays MIDI events)
        """
        old_offset = self.timing_offset_beats
        self.timing_offset_beats = offset_beats
        
        
    def start_section(self, section=None):
        """Start scheduler for a specific section (by time signature)
        
        LEGACY METHOD: Kept for backward compatibility
        """
        if section is None:
            self.start()
            return
        self.current_section = section
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            return
        self.should_stop.clear()
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()

    def _scheduler_loop(self):
        """Main scheduler loop using PrecisionTimer as the single authoritative time source
        
        In the single-clock architecture:
        - All timing queries come from PrecisionTimer.get_precise_time()
        - MIDI event times are adjusted: adjusted_time = original_time + timing_offset_beats
        - This creates a delay period at start where only metronome plays (when offset > 0)
        - Perfect synchronization is maintained because both metronome and MIDI use the same clock
        """
        if not self.engine.midi_file:
            return
        
        self._prepare_events()
        
        # Verify PrecisionTimer availability
        if not hasattr(self.engine, 'precision_timer'):
            print("Error: Engine does not have PrecisionTimer - cannot schedule events")
            return
        
        # Get current position from PrecisionTimer (single clock source)
        current_beat = self.engine.precision_timer.get_precise_time()
        
        # Always reset to current_beat - 1 to handle section switching (beat can go backwards)
        current_beat_number = int(current_beat)
        self._last_beat_number = current_beat_number - 1
        
        event_index = 0
        
        # Main event scheduling loop
        while not self.should_stop.is_set() and event_index < len(self.events):
            # Wait for resume if paused
            if not self.engine.is_playing:
                time.sleep(0.1)
                continue
            
            # Verify precision timer is running
            if not self.engine.precision_timer.is_running:
                time.sleep(0.01)
                continue
            
            event_time, msg, track_idx = self.events[event_index]
            
            # Get current beat from PrecisionTimer (single clock source)
            current_beat = self.engine.precision_timer.get_precise_time()
            
            # even if no events are due yet, drive activity monitor by beats
            # to keep UI responsive during long waits or initial offsets.
            self._handle_beat_monitoring(current_beat)
            
            # Apply anacrusis offset to event time
            adjusted_event_time = event_time + self.timing_offset_beats
            
            # Calculate wait time
            wait_time_beats = adjusted_event_time - current_beat
            seconds_per_beat = 60.0 / self.engine.tempo
            wait_time_seconds = wait_time_beats * seconds_per_beat
            
            # RACE CONDITION FIX: For early events (first 10), use generous tolerance
            # Thread startup delay may cause events to appear "slightly expired" (e.g., -10ms to -50ms)
            # Instead of skipping them, treat them as "due now" and play them immediately
            if event_index < 10 and wait_time_seconds < 0:
                # Early event that appears expired - check if it's within tolerance
                if wait_time_seconds >= -0.5:  # Within 500ms - treat as "due now" due to startup delay
                    # Don't skip - treat as immediately due and play it
                    wait_time_seconds = 0.0
                else:
                    # Really expired (>500ms late) - skip it
                    print(f"[EventScheduler] ⚠️ Skipping early event {event_index} (expired by {abs(wait_time_seconds)*1000:.1f}ms): "
                          f"event_time={adjusted_event_time:.6f}, current={current_beat:.6f}")
                    event_index += 1
                    continue
            
            # Handle expired events (more than 100ms late) for normal events
            if wait_time_seconds < -0.1:
                event_index += 1
                continue
            
            # Sleep if event is in the future (cap at 100ms to allow responsive tempo changes)
            if wait_time_seconds > 0:
                time.sleep(min(wait_time_seconds, 0.1))
                continue  # Re-check timing after sleep
            
            # Event is due now (within 100ms window)
            # Double-check timing after sleep to handle tempo changes
            current_beat = self.engine.precision_timer.get_precise_time()
            adjusted_event_time = event_time + self.timing_offset_beats
            wait_time_beats = adjusted_event_time - current_beat
            
            # Process event if it's due (within 10ms tolerance)
            if wait_time_beats <= 0.01:
                # Beat-based activity monitoring
                self._handle_beat_monitoring(current_beat)
                
                # Process the MIDI event
                if not self.should_stop.is_set() and self.engine.is_playing:
                    self._process_event(msg, track_idx)
                    event_index += 1

    def _handle_beat_monitoring(self, current_beats: float):
        """Handle beat-based activity monitoring
        
        Args:
            current_beats: Current playback position in beats
        """
        current_beat_number = int(current_beats)
        
        # Check if we've crossed a beat boundary
        if current_beat_number > self._last_beat_number:
            # Update activity monitor for new beat
            self.activity_monitor.on_beat_event(current_beat_number)
            self._last_beat_number = current_beat_number
    
    def _prepare_events(self):
        """Prepare event list, filtered by section if set"""
        self.events = []
        section = getattr(self, 'current_section', None)
        for i, track in enumerate(self.engine.midi_file.tracks):
            track_time = 0
            for msg in track:
                track_time += msg.time / self.engine.midi_file.ticks_per_beat
                # Only process non-meta messages
                if not msg.is_meta:
                    # if section is set, only include events in section range
                    if section is not None:
                        start_beat = section.get('start_beat', 0.0)
                        end_beat = section.get('end_beat', None)
                        if end_beat is not None:
                            if not (start_beat <= track_time < end_beat):
                                continue
                        else:
                            if track_time < start_beat:
                                continue
                    self.events.append((track_time, msg, i))
        # sort by time
        self.events.sort(key=lambda x: x[0])
    
    def _process_event(self, msg, track_idx):
        """Process MIDI event
        
        Args:
            msg: MIDI message
            track_idx: Track index
        """
        # Note: Activity monitoring is now handled by beat events, not individual MIDI events
        # This eliminates the per-event processing overhead and provides stable visualization
        
        # Check mute status for actual audio playback
        if track_idx in self.engine.muted_tracks:
            return
        
        # Handle note on/off events
        if msg.type == 'note_on' and msg.velocity > 0:
            self.engine._send_note_on(msg.note, msg.velocity, msg.channel, track_idx)
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            self.engine._send_note_off(msg.note, msg.channel)
        else:
            # Send other MIDI messages directly
            self.engine._send_midi_message(msg)
    
    def reset_activity_monitoring(self):
        """Reset activity monitoring state (called when loading new file)"""
        self.activity_monitor.reset()
        self._last_beat_number = -1
        # print("EventScheduler: Beat-based activity monitoring reset")
    
    def set_tempo(self, tempo: float):
        """Update tempo in activity monitor
        
        Args:
            tempo: New tempo in BPM
        """
        self.activity_monitor.set_tempo(tempo)
    
    def get_activity_statistics(self):
        """Get activity monitoring statistics for debugging
        
        Returns:
            Dictionary with monitoring statistics
        """
        return self.activity_monitor.get_statistics()