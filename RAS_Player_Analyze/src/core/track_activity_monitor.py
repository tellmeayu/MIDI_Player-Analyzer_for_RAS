from dataclasses import dataclass, field
from typing import Dict, Set, List, Callable, Optional
import time
import mido

@dataclass
class TrackActivityState:
    """Track activity state data structure
    
    Simplified state for beat-based detection system.
    """
    track_id: int
    current_velocity: int = 0  # Current velocity for this beat (0-127)
    is_active: bool = False  # Whether track has notes in current beat
    beat_number: int = 0  # Current beat number for tracking


@dataclass
class BeatNoteInfo:
    """Information about notes in a specific beat"""
    track_id: int
    beat_number: int
    first_note_velocity: int = 0  # Velocity of first note in this beat
    has_notes: bool = False  # Whether this beat has any notes
    notes_count: int = 0  # Number of notes in this beat


class BeatBasedActivityMonitor:
    """Beat-based track activity monitor
    
    This monitor preprocesses MIDI events by beats and updates track visualization
    at beat intervals rather than on every MIDI event. This provides:
    - More stable visualization (no flickering)
    - Lower computational overhead
    - Musical rhythm-aligned updates
    - Cleaner detection logic
    """
    
    def __init__(self):
        """Initialize beat-based activity monitor"""
        self.track_states: Dict[int, TrackActivityState] = {}
        self.callbacks: List[Callable[[int, TrackActivityState], None]] = []
        
        # Beat-based data structures
        self.beat_data: Dict[int, Dict[int, BeatNoteInfo]] = {}  # beat_number -> {track_id: BeatNoteInfo}
        self.total_beats: int = 0
        self.current_beat: int = 0
        
        # Tempo and timing info
        self.current_tempo: float = 120.0
        self.ticks_per_beat: int = 480
        
        # Track information
        self.monitored_tracks: Set[int] = set()
        
        # Active notes tracking for cross-beat notes handling
        self.active_notes: Dict[int, Dict[int, int]] = {}  # track_id -> {note_number: velocity}
    
    def initialize_tracks(self, tracks_info: List[Dict], midi_file=None):
        """Initialize monitoring for all tracks and preprocess MIDI events
        
        Args:
            tracks_info: List of track information dictionaries from engine metadata
            midi_file: MIDI file object for preprocessing
        """
        # Clear existing states
        self.track_states.clear()
        self.beat_data.clear()
        self.monitored_tracks.clear()
        self.active_notes.clear()
        
        # Initialize track states for tracks with note events
        for track in tracks_info:
            track_id = track['index']
            if track.get('has_note_events', False):
                self.track_states[track_id] = TrackActivityState(track_id=track_id)
                self.monitored_tracks.add(track_id)
                self.active_notes[track_id] = {}
        
        # Preprocess MIDI file if provided
        if midi_file:
            self._preprocess_midi_events(midi_file)
    
    def _preprocess_midi_events(self, midi_file):
        """Preprocess MIDI events by beats for efficient runtime lookup
        
        Uses a timeline-based approach to accurately handle long notes and overlapping notes.
        
        Args:
            midi_file: MIDI file object
        """
        
        # Get timing information
        self.ticks_per_beat = midi_file.ticks_per_beat
        
        # Calculate total beats in the file
        file_length_seconds = midi_file.length
        beats_per_second = self.current_tempo / 60.0
        self.total_beats = int(file_length_seconds * beats_per_second) + 1
        
        # Process each track
        for track_idx, track in enumerate(midi_file.tracks):
            if track_idx not in self.monitored_tracks:
                continue
            
            # Collect all events with precise timing
            events = []  # [(beat_time, event_type, note, velocity), ...]
            track_time_beats = 0.0
            
            for msg in track:
                # Accumulate time in beats
                track_time_beats += msg.time / self.ticks_per_beat
                
                # Process note events
                if msg.type == 'note_on' and msg.velocity > 0:
                    events.append((track_time_beats, 'note_on', msg.note, msg.velocity))
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    events.append((track_time_beats, 'note_off', msg.note, 0))
            
            # Sort events by time
            events.sort(key=lambda x: x[0])
            
            # Process events to determine activity for each beat
            self._process_track_timeline(track_idx, events)        
        # Debug: Print summary of preprocessing
        # total_beat_activities = sum(len(beats) for beats in self.beat_data.values())
        # print(f"BeatBasedActivityMonitor: Found {total_beat_activities} beat activities across all tracks")
        
    def _process_track_timeline(self, track_id: int, events: list):
        """Process track events using timeline approach for accurate long note handling
        
        Args:
            track_id: Track index
            events: List of (beat_time, event_type, note, velocity) sorted by beat_time
        """
        if not events:
            return
        
        # For each beat, determine what notes are active during that beat
        for beat in range(self.total_beats):
            beat_start = float(beat)
            beat_end = float(beat + 1)
            
            active_notes_this_beat = {}  # note_number -> velocity
            
            # Check all events to see what notes are active during this beat
            for i, (event_time, event_type, note, velocity) in enumerate(events):
                if event_type == 'note_on':
                    # Find when this note ends
                    note_end_time = None
                    for j in range(i + 1, len(events)):
                        end_time, end_type, end_note, _ = events[j]
                        if end_type == 'note_off' and end_note == note:
                            note_end_time = end_time
                            break
                    
                    # If no note_off found, assume note lasts until end of track
                    if note_end_time is None:
                        note_end_time = float(self.total_beats)
                    
                    # Check if this note is active during current beat
                    if event_time < beat_end and note_end_time > beat_start:
                        # Note is active during this beat
                        active_notes_this_beat[note] = velocity
            
            # Set activity for this beat based on active notes
            if active_notes_this_beat:
                max_velocity = max(active_notes_this_beat.values())
                self._ensure_beat_has_activity(track_id, beat, max_velocity)
                
    def _add_note_to_beat(self, track_id: int, beat_number: int, velocity: int):
        """Add note information to specific beat
        
        Args:
            track_id: Track index
            beat_number: Beat number
            velocity: Note velocity
        """
        if beat_number not in self.beat_data:
            self.beat_data[beat_number] = {}
        
        if track_id not in self.beat_data[beat_number]:
            self.beat_data[beat_number][track_id] = BeatNoteInfo(
                track_id=track_id,
                beat_number=beat_number
            )
        
        beat_info = self.beat_data[beat_number][track_id]
        
        # If this is the first note in this beat, record its velocity
        if not beat_info.has_notes:
            beat_info.first_note_velocity = velocity
            beat_info.has_notes = True
        
        beat_info.notes_count += 1
    
    def _ensure_beat_has_activity(self, track_id: int, beat_number: int, velocity: int):
        """Ensure a beat shows activity, handling velocity priority for overlapping notes
        
        Args:
            track_id: Track index
            beat_number: Beat number
            velocity: Note velocity to use
        """
        if beat_number >= self.total_beats:
            return  # Don't go beyond file length
            
        if beat_number not in self.beat_data:
            self.beat_data[beat_number] = {}
        
        if track_id not in self.beat_data[beat_number]:
            # Create new activity for this beat
            self.beat_data[beat_number][track_id] = BeatNoteInfo(
                track_id=track_id,
                beat_number=beat_number,
                first_note_velocity=velocity,
                has_notes=True,
                notes_count=1
            )
        else:
            # Beat already has activity - use higher velocity if this note is louder
            existing_info = self.beat_data[beat_number][track_id]
            if velocity > existing_info.first_note_velocity:
                existing_info.first_note_velocity = velocity
            existing_info.notes_count += 1
    
    def set_tempo(self, tempo: float):
        """Update tempo for beat timing calculations
        
        Args:
            tempo: New tempo in BPM
        """
        self.current_tempo = tempo

    def on_beat_event(self, beat_number: int):
        """Handle beat event - update track activities based on preprocessed data
        
        Args:
            beat_number: Current beat number (0-based)
        """
        self.current_beat = beat_number
        
        # Update all monitored tracks for this beat
        for track_id in self.monitored_tracks:
            self._update_track_for_beat(track_id, beat_number)
    
    def _update_track_for_beat(self, track_id: int, beat_number: int):
        """Update specific track activity for current beat
        
        Args:
            track_id: Track index
            beat_number: Beat number
        """
        track_state = self.track_states.get(track_id)
        if not track_state:
            return
        
        # Get beat data for this track
        beat_info = None
        if beat_number in self.beat_data and track_id in self.beat_data[beat_number]:
            beat_info = self.beat_data[beat_number][track_id]
        
        # Update track state based on beat data
        old_velocity = track_state.current_velocity
        old_active = track_state.is_active
        
        if beat_info and beat_info.has_notes:
            # Beat has notes - show activity
            track_state.current_velocity = beat_info.first_note_velocity
            track_state.is_active = True
        else:
            # Beat has no notes - show silence
            track_state.current_velocity = 0
            track_state.is_active = False
        
        track_state.beat_number = beat_number
        
        # Always trigger callbacks to ensure GUI updates
        # This ensures all tracks are updated on each beat, providing stable visualization
        self._trigger_callbacks(track_id, track_state)
    
    def get_track_activity(self, track_id: int) -> Optional[TrackActivityState]:
        """Get current activity state for a track
        
        Args:
            track_id: Track index
            
        Returns:
            TrackActivityState or None if track not monitored
        """
        return self.track_states.get(track_id)
    
    def get_all_activities(self) -> Dict[int, TrackActivityState]:
        """Get activity states for all monitored tracks
        
        Returns:
            Dictionary mapping track_id to TrackActivityState
        """
        return self.track_states.copy()
    
    def add_callback(self, callback: Callable[[int, TrackActivityState], None]):
        """Add callback for track activity changes
        
        Args:
            callback: Function to call when track activity changes
                     Signature: callback(track_id: int, state: TrackActivityState)
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[int, TrackActivityState], None]):
        """Remove callback for track activity changes
        
        Args:
            callback: Callback function to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _trigger_callbacks(self, track_id: int, state: TrackActivityState):
        """Trigger all registered callbacks
        
        Args:
            track_id: Track index
            state: Current track activity state
        """
        for callback in self.callbacks:
            try:
                callback(track_id, state)
            except Exception as e:
                print(f"Error in track activity callback: {e}")
    
    def reset(self):
        """Reset all track states (called when loading new file)"""
        for state in self.track_states.values():
            state.current_velocity = 0
            state.is_active = False
            state.beat_number = 0
        
        self.current_beat = 0
    
    def get_beat_info(self, beat_number: int) -> Dict[int, BeatNoteInfo]:
        """Get beat information for debugging/analysis
        
        Args:
            beat_number: Beat number to query
            
        Returns:
            Dictionary mapping track_id to BeatNoteInfo for the specified beat
        """
        return self.beat_data.get(beat_number, {})
    
    def get_statistics(self) -> Dict:
        """Get monitor statistics for debugging
        
        Returns:
            Dictionary with monitoring statistics
        """
        active_beats = len(self.beat_data)
        total_activities = sum(len(beats) for beats in self.beat_data.values())
        
        return {
            'total_beats': self.total_beats,
            'active_beats': active_beats,
            'total_activities': total_activities,
            'monitored_tracks': len(self.monitored_tracks),
            'current_beat': self.current_beat,
            'current_tempo': self.current_tempo
        }


# Maintain backward compatibility by aliasing the new class
TrackActivityMonitor = BeatBasedActivityMonitor 