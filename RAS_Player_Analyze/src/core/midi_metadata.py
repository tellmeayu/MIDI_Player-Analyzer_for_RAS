"""MIDI file metadata analysis.

This module provides comprehensive analysis of MIDI file metadata, including:
- Tempo analysis with note activity weighting
- Time signature detection
- Track information extraction
- Dynamic tempo and time signature changes
"""

from typing import Dict, Optional, List
import numpy as np
import mido
from .gm_instruments import GM_INSTRUMENT_NAMES

class MidiMetadataAnalyzer:
    """Analyzes MIDI files for metadata and musical characteristics"""
    
    # Define reasonable tempo range for musical content
    MIN_REASONABLE_TEMPO = 30.0   # Minimum reasonable BPM
    MAX_REASONABLE_TEMPO = 200.0  # Maximum reasonable BPM
    
    def __init__(self):
        self._cached_metadata: Dict = {}  # Cache by file path
        self._tempo_init_time_tolerance = 1.5
    
    def analyze_file(self, midi_file: mido.MidiFile, current_tempo: float = 120.0) -> Dict:
        """Analyze MIDI file metadata with intelligent tempo analysis
            Args:
                midi_file: MIDI file object
                current_tempo: Current engine tempo (default 120.0)
                
            Returns:
                Dict containing comprehensive metadata analysis
        """
        # Basic metadata structure
        metadata = {
            'format': midi_file.type,
            'tracks': len(midi_file.tracks),
            'ticks_per_beat': midi_file.ticks_per_beat,
            'tempo': current_tempo,
            'length': midi_file.length,  # Length (seconds)
            'time_signature': {'numerator': 4, 'denominator': 4},  # Default 4/4 time
            'key_signature': None,    # Default empty
            'title': None  # MIDI file title/name
        }
        
        # Add formatted MIDI type information with hint
        metadata['midi_type_info'] = self._format_midi_type_info(midi_file.type)
        
        # Analysis state - use mutable containers for initial values
        time_signature_changes = []
        initial_time_signature = [None]  # Use list to make it mutable
        tempo_changes = []
        initial_tempo = [None]  # Use list to make it mutable
        note_events = []  # (time_beats, note_on/off, track_idx)
        
        # Analyze each track
        for track_idx, track in enumerate(midi_file.tracks):
            current_time = 0.0  # Track position in beats
            
            for msg in track:
                # Update time position
                current_time += msg.time / midi_file.ticks_per_beat
                
                # Extract metadata from messages
                self._process_midi_message(msg, current_time, metadata,
                                        time_signature_changes, tempo_changes,
                                        note_events, track_idx,
                                        initial_time_signature, initial_tempo)
        
        # Process collected data
        metadata = self._process_time_signatures(metadata, time_signature_changes, initial_time_signature[0])
        metadata = self._analyze_tempo_data(metadata, tempo_changes, note_events)
        metadata = self._extract_track_info(metadata, midi_file)
        metadata = self._calculate_total_measures(metadata)

        # --- SECTION DETECTION ---
        metadata['sections'] = self._detect_sections(
            midi_file,
            time_signature_changes,
            tempo_changes,
            metadata['ticks_per_beat'],
            metadata.get('total_measures', 0),
            metadata.get('length', 0.0),
            note_events
        )
        
        return metadata
    
    def _process_midi_message(self, msg, current_time: float, metadata: Dict,
                            time_signature_changes: List, tempo_changes: List,
                            note_events: List, track_idx: int,
                            initial_time_signature: List,
                            initial_tempo: List):
        """Process individual MIDI message for metadata extraction"""
        if msg.type == 'track_name':
            if metadata['title'] is None and msg.name.strip():
                metadata['title'] = msg.name.strip()
        
        elif msg.type == 'sequence_name':
            if msg.name.strip():
                metadata['title'] = msg.name.strip()
        
        elif msg.type == 'text':
            if metadata['title'] is None and msg.text.strip():
                text = msg.text.strip()
                if any(keyword in text.lower() for keyword in 
                      ['symphony', 'sonata', 'concerto', 'op.', 'opus', 'no.', 'minor', 'major']):
                    metadata['title'] = text
        
        elif msg.type == 'time_signature':
            time_sig_info = {
                'time_beats': current_time,
                'numerator': msg.numerator,
                'denominator': msg.denominator,
                'clocks_per_click': msg.clocks_per_click,
                'notated_32nd_notes_per_beat': msg.notated_32nd_notes_per_beat,
                'text': f"{msg.numerator}/{msg.denominator}"
            }
            
            if initial_time_signature[0] is None:
                initial_time_signature[0] = time_sig_info.copy()
                # print(f"Initial time signature detected: {time_sig_info['text']} at time {current_time:.2f}")
            
            time_signature_changes.append(time_sig_info)
        
        elif msg.type == 'set_tempo':
            tempo_bpm = mido.tempo2bpm(msg.tempo)
            tempo_info = {
                'time_beats': current_time,
                'time_seconds': current_time * (60.0 / 120.0),
                'tempo_bpm': tempo_bpm,
                'microseconds_per_beat': msg.tempo
            }
            
            if initial_tempo[0] is None:
                initial_tempo[0] = tempo_info.copy()
                # print(f"Initial tempo detected: {tempo_bpm:.1f} BPM at time {current_time:.2f}")
            
            tempo_changes.append(tempo_info)
        
        elif msg.type == 'key_signature':
            metadata['key_signature'] = msg.key
        
        elif msg.type == 'note_on' and msg.velocity > 0:
            note_events.append((current_time, 'note_on', track_idx))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            note_events.append((current_time, 'note_off', track_idx))
    
    def _process_time_signatures(self, metadata: Dict, time_signature_changes: List,
                               initial_time_signature: Optional[Dict]) -> Dict:
        """Process collected time signature data with intelligent change detection"""
        if initial_time_signature:
            metadata['time_signature'] = initial_time_signature
        else:
            metadata['time_signature'] = {'numerator': 4, 'denominator': 4, 'text': '4/4'}
        
        metadata['time_signature_changes'] = time_signature_changes
        
        # Intelligent detection: only consider it "dynamic" if time signatures actually change
        unique_time_signatures = self._get_unique_time_signatures(time_signature_changes)
        
        if len(unique_time_signatures) > 1:
            metadata['has_dynamic_time_signatures'] = True
            signatures_text = " → ".join([ts['text'] for ts in unique_time_signatures[:8]])
            if len(unique_time_signatures) > 8:
                signatures_text += f" ... (+{len(unique_time_signatures) - 8} more)"
            metadata['dynamic_signatures_preview'] = signatures_text
            
            # Report the improvement
            total_events = len(time_signature_changes)
            unique_count = len(unique_time_signatures)
            print(f"Dynamic time signatures detected: {signatures_text}")
            if total_events > unique_count:
                print(f"  Filtered {total_events - unique_count} redundant time signature events")
        else:
            metadata['has_dynamic_time_signatures'] = False
            
        return metadata
    
    def _get_unique_time_signatures(self, time_signature_changes: List) -> List[Dict]:
        """Extract unique time signatures in chronological order
        
        Args:
            time_signature_changes: List of time signature change events
            
        Returns:
            List[Dict]: Unique time signatures in order of first appearance
        """
        if not time_signature_changes:
            return []
        
        unique_signatures = []
        seen_signatures = set()
        
        for ts_change in time_signature_changes:
            # Create a signature key based on numerator and denominator
            signature_key = (ts_change['numerator'], ts_change['denominator'])
            
            if signature_key not in seen_signatures:
                seen_signatures.add(signature_key)
                unique_signatures.append(ts_change)
        
        return unique_signatures
    
    def _analyze_tempo_data(self, metadata: Dict, tempo_changes: List,
                          note_events: List) -> Dict:
        """Analyze tempo data with note activity weighting"""
        
        metadata['tempo_changes'] = tempo_changes
        
        # Flag indicating if the MIDI has explicit tempo events
        metadata['has_tempo_events'] = len(tempo_changes) > 0
        
        if not tempo_changes:
            metadata['has_dynamic_tempo'] = False
            metadata['tempo_type'] = 'default'
            metadata['initial_tempo'] = 120.0
            metadata['average_tempo'] = 120.0
            print("No tempo information found - using default 120 BPM")
            return metadata
        
        else:
            metadata['initial_tempo'] = tempo_changes[0]['tempo_bpm']
            initial_tempo = metadata['initial_tempo']

            if len(tempo_changes) == 1:
                if not self._is_reasonable_tempo(initial_tempo):
                    print(f"⚠️  Single tempo {initial_tempo:.1f} BPM is unreasonable, using fallback")
                    return self._handle_no_active_tempos(metadata)
            
                metadata['has_dynamic_tempo'] = False
                metadata['tempo_type'] = 'static'
                metadata['average_tempo'] = initial_tempo
                return metadata
            else: 
                return self._analyze_multiple_tempos(metadata, tempo_changes, note_events)
        
    def _handle_single_tempo(self, metadata: Dict, initial_tempo: float) -> Dict:
        """Handle case with single tempo throughout"""
        
        # Check if the single tempo is reasonable
        if not self._is_reasonable_tempo(initial_tempo):
            print(f"⚠️  Single tempo {initial_tempo:.1f} BPM is unreasonable, using fallback")
            return self._handle_no_active_tempos(metadata)
        
        metadata['has_dynamic_tempo'] = False
        metadata['tempo_type'] = 'static'
        metadata['average_tempo'] = initial_tempo
        return metadata
    
    def _analyze_multiple_tempos(self, metadata: Dict, tempo_changes: List,
                               note_events: List) -> Dict:
        """Analyze multiple tempo changes with note activity weighting for average calculation"""
        metadata['has_dynamic_tempo'] = True
        metadata['tempo_type'] = 'dynamic'
        
        weighted_tempo_sum = 0.0
        total_weight = 0.0
        filtered_out_count = 0
                
        for i, tempo_change in enumerate(tempo_changes):
            tempo_info = self._analyze_tempo_period(
                tempo_change, tempo_changes, i, note_events
            )
            
            # Only include reasonable tempos with notes for average calculation
            if tempo_info['notes_in_period'] > 0:
                if self._is_reasonable_tempo(tempo_info['tempo_bpm']):
                    weighted_tempo_sum += tempo_info['tempo_bpm'] * tempo_info['duration_seconds']
                    total_weight += tempo_info['duration_seconds']
                else:
                    filtered_out_count += 1
        
        if total_weight > 0:
            average_tempo = weighted_tempo_sum / total_weight
            metadata['average_tempo'] = average_tempo
            return metadata
        
        return self._handle_no_active_tempos(metadata)
    
    def _analyze_tempo_period(self, tempo_change: Dict, tempo_changes: List,
                            index: int, note_events: List) -> Dict:
        """Analyze individual tempo period"""
        tempo_start_time = tempo_change['time_beats']
        tempo_bpm = tempo_change['tempo_bpm']
        
        if index < len(tempo_changes) - 1:
            tempo_end_time = tempo_changes[index + 1]['time_beats']
        else:
            last_note_time = note_events[-1][0] if note_events else tempo_start_time
            tempo_end_time = last_note_time + 4  # Add buffer
        
        notes_in_period = sum(
            1 for time, event_type, _ in note_events
            if tempo_start_time <= time < tempo_end_time and event_type == 'note_on'
        )
        
        duration_beats = tempo_end_time - tempo_start_time
        duration_seconds = duration_beats * (60.0 / tempo_bpm)
        
        # print(f"  Tempo {tempo_bpm:.1f} BPM: {duration_beats:.1f} beats "
        #       f"({duration_seconds:.1f}s), {notes_in_period} notes")
        
        return {
            'tempo_bpm': tempo_bpm,
            'duration_beats': duration_beats,
            'duration_seconds': duration_seconds,
            'notes_in_period': notes_in_period
        }
    
    def _handle_no_active_tempos(self, metadata: Dict) -> Dict:
        """Handle case when no active tempos remain after filtering"""
        print("⚠️  No reasonable active tempos found - using fallback")
        
        # Use a reasonable fallback tempo
        fallback_tempo = 120.0
        metadata['initial_tempo'] = fallback_tempo
        metadata['average_tempo'] = fallback_tempo
        metadata['tempo_type'] = 'fallback'
        metadata['has_dynamic_tempo'] = False
        
        return metadata
    
    def _extract_track_info(self, metadata: Dict, midi_file: mido.MidiFile) -> Dict:
        """Extract information about MIDI tracks"""
        track_info = []
        
        for i, track in enumerate(midi_file.tracks):
            track_data = self._analyze_track(track, i)
            if track_data['has_note_events']:
                track_info.append(track_data)
        
        metadata['tracks_info'] = track_info
        return metadata
    
    def _analyze_track(self, track: mido.MidiTrack, index: int) -> Dict:
        """Analyze individual MIDI track"""
        track_name = None
        instrument = None
        events_count = len(track)
        has_note_events = False
        
        for msg in track:
            if msg.type == 'track_name':
                track_name = msg.name
            elif msg.type == 'program_change':
                instrument = msg.program
            elif msg.type == 'note_on' and msg.velocity > 0:
                has_note_events = True
        
        track_data = {
            'index': index,
            'name': track_name or f"Track {index}",
            'instrument': instrument,
            'events_count': events_count,
            'has_note_events': has_note_events
        }
        
        if instrument is not None:
            instrument_name = GM_INSTRUMENT_NAMES.get(instrument, "Unknown Instrument")
            track_data['instrument_display'] = f"{instrument} ({instrument_name})"
        
        return track_data
    
    def _calculate_total_measures(self, metadata: Dict) -> Dict:
        """Calculate total measures based on length and time signature"""
        practical_tempo = metadata.get('initial_tempo', 120.0)
        length_beats = metadata['length'] * (practical_tempo / 60.0)
        beats_per_measure = metadata['time_signature']['numerator']
        
        total_measures = int(length_beats / beats_per_measure)
        if length_beats % beats_per_measure > 0:
            total_measures += 1
        
        metadata['total_measures'] = total_measures
        return metadata

    def _is_reasonable_tempo(self, tempo_bpm: float) -> bool:
        """Check if a tempo is reasonable within the defined range"""
        return self.MIN_REASONABLE_TEMPO <= tempo_bpm <= self.MAX_REASONABLE_TEMPO 

    def _format_midi_type_info(self, midi_type: int) -> Optional[str]:
        """Format MIDI type information with hint for non-Type 1 files
        
        Args:
            midi_type: MIDI file type (0, 1, or 2)
            
        Returns:
            Optional[str]: Formatted MIDI type information, None if Type 1 (default)
        """
        if midi_type == 1:
            # Type 1 is the default/standard type, no need to show hint
            return None
        elif midi_type == 0:
            return "MIDI Type 0 (single track format)"
        elif midi_type == 2:
            return "MIDI Type 2 (multiple independent sequences)"
        else:
            return f"MIDI Type {midi_type} (non-standard format)" 
        
    def _detect_sections(self, midi_file, time_signature_changes, tempo_changes, ticks_per_beat, total_measures, midi_length_sec, note_events):
        """Detect musical sections based on time signature changes, with practical initial tempo
        Args:
            midi_file: mido.MidiFile
            time_signature_changes: list of time signature change dicts
            tempo_changes: list of tempo change dicts
            ticks_per_beat: int
            total_measures: int
            midi_length_sec: float
            note_events: list of note events [(time_beats, 'note_on'/'note_off', track_idx)]
        Returns:
            List of section dicts,
            each with start/end beat, measure, time signature, practical initial tempo, etc.
        """
        # use only time signature changes for section boundaries
        ts_events = sorted(time_signature_changes, key=lambda x: x['time_beats'])
        # --- extract all marker/text events with their beat positions ---
        marker_events = [] # list of (beat, text)
        for track in midi_file.tracks:
            current_time = 0.0
            for msg in track:
                current_time += msg.time / ticks_per_beat
                if msg.type == 'marker' and hasattr(msg, 'texg'):
                    marker_events.append((current_time, msg.text.strip()))
                elif msg.type == 'text' and hasattr(msg, 'text'):
                    marker_events.append((current_time, msg.text.strip()))
        # if no time signature changes, signle section
        if not ts_events:
            return [{
                'index': 0,
                'label': 'Section 1',
                'start_beat': 0.0,
                'end_beat': None,
                'start_measure': 1,
                'end_measure': total_measures,
                'time_signature': {'numerator': 4, 'denominator': 4, 'text': '4/4'},
                'tempo': 120.0,
                'marker': ''
            }]
        # Helper: get tempo at a given beat
        def get_tempo_at_beat(beat):
            tempo = 120.0
            for t in tempo_changes:
                if t['time_beats'] <= beat:
                    tempo = t['tempo_bpm']
                else:
                    break
            return tempo

        def get_practical_tempo_at_beat(section_start_beat, section_end_beat, tempo_changes, note_events):
            """Find practical initial tempo for a specific section by 
                finding the timing of first note, then find the closest tempo before or at that time
                Args:
                    section_start_beat: float - section start time in beats
                    section_end_beat: float or None - section end time in beats
                    tempo_changes: list of tempo change dicts
                    note_events: list of note events
                
                Returns:
                    float - practical initial tempo for this section
            """
            
            # Convert to numpy array and extract note_on times with proper float type
            note_events_array = np.array(note_events, dtype=object)
            if len(note_events_array) == 0:
                tempo_at_start = get_tempo_at_beat(section_start_beat)
                print(f"Section {section_start_beat:.2f}-{section_end_beat}: No note events, using tempo at start: {tempo_at_start:.1f} BPM")
                return tempo_at_start
            
            # Filter note_on events and convert times to float
            note_on_mask = note_events_array[:, 1] == 'note_on'
            if not np.any(note_on_mask):
                tempo_at_start = get_tempo_at_beat(section_start_beat)
                print(f"Section {section_start_beat:.2f}-{section_end_beat}: No note_on events, using tempo at start: {tempo_at_start:.1f} BPM")
                return tempo_at_start
            
            noteON_times = note_events_array[note_on_mask][:, 0].astype(float)
            
            # Find notes within section boundaries
            potential_first_notes = noteON_times[noteON_times >= section_start_beat]
            if section_end_beat is not None:
                potential_first_notes = potential_first_notes[potential_first_notes < section_end_beat]
            
            # Handle case where no notes found in section
            if len(potential_first_notes) == 0:
                tempo_at_start = get_tempo_at_beat(section_start_beat)
                print(f"Section {section_start_beat:.2f}-{section_end_beat}: No notes found within section, using tempo at start: {tempo_at_start:.1f} BPM")
                return tempo_at_start
            
            # Find the first note in the section
            section_first_note_time = potential_first_notes.min()
            # print(f"Section {section_start_beat:.2f}-{section_end_beat}: First note at beat {section_first_note_time:.2f}")
            
            # Handle empty tempo_changes list
            if not tempo_changes:
                print(f"Section {section_start_beat:.2f}-{section_end_beat}: No tempo changes, using default 120 BPM")
                return 120.0
            
            # Find the closest tempo at or before the first note
            tempo_times = np.array([t['time_beats'] for t in tempo_changes])
            tempo_values = np.array([t['tempo_bpm'] for t in tempo_changes])
            valid_mask = tempo_times <= (section_first_note_time + self._tempo_init_time_tolerance)

            if np.any(valid_mask):
                # Find the latest tempo change before or at the first note
                valid_tempo_times = tempo_times[valid_mask]
                # print(f"last valid tempo times: {valid_tempo_times[-1]}")
                valid_tempo_values = tempo_values[valid_mask]
                # print(f"valid tempo values: {valid_tempo_values[-1]}")
                nearest_tempo_idx = valid_tempo_times.argmax()
                practical_first_tempo = valid_tempo_values[nearest_tempo_idx]
                closest_tempo_time = valid_tempo_times[nearest_tempo_idx]
                # print(f"Section {section_start_beat:.2f}-{section_end_beat}: Using tempo {practical_first_tempo:.1f} BPM from beat {closest_tempo_time:.2f}")
                return practical_first_tempo
            else:
                # No tempo change before first note, use tempo at section start
                tempo_at_start = get_tempo_at_beat(section_start_beat)
                # print(f"Section {section_start_beat:.2f}-{section_end_beat}: No tempo before first note, using tempo at start: {tempo_at_start:.1f} BPM")
                return tempo_at_start

        sections = []
        last_measure = 1
        section_idx = 0
        for i, ts in enumerate(ts_events):  # detect sections by checking meter change point
            # For the first section, start at beat 0; for subsequent sections, start at the time signature change
            if i == 0:
                start_beat = 0.0
                start_measure = 1
            else:
                start_beat = ts['time_beats']
                # Calculate start measure based on previous sections
                prev_section = sections[-1]
                start_measure = prev_section['end_measure'] + 1
            
            time_sig = ts.copy()
            beats_per_measure = time_sig.get('numerator', 4)
            # end beat is next time signature change, or None for last section
            if i + 1 < len(ts_events):
                end_beat = ts_events[i+1]['time_beats']
            else:
                end_beat = None
            # calculate measures in this section
            if end_beat is not None:
                beats_in_section = end_beat - start_beat
                measures = int(round(beats_in_section / beats_per_measure))
                end_measure = start_measure + measures -1
            else:
                # last section: estimate from file length and tempo
                if midi_file.length > 0:
                    tempo = get_tempo_at_beat(start_beat)
                    total_beats = midi_file.length * (tempo / 60.0)
                    beats_in_section = total_beats - start_beat
                    measures = int(round(beats_in_section / beats_per_measure))
                    end_measure = start_measure + measures -1
                else:
                    measures = total_measures - start_measure + 1
                    end_measure = total_measures
            # Find practical initial tempo for this section based on note activity
            tempo = get_practical_tempo_at_beat(start_beat, end_beat, tempo_changes, note_events)
            # find marker at this section's start beat (exact match or within small tolerance)
            marker_text = ''
            for marker_beat, marker in marker_events:
                if abs(marker_beat - start_beat) < 1e-3:
                    marker_text = marker
                    break
            section = {
                'index': section_idx,
                'label': f'Section {section_idx+1}',
                'start_beat': start_beat,
                'end_beat': end_beat,
                'start_measure': start_measure,
                'end_measure': end_measure,
                'measures': measures,
                'time_signature': time_sig,
                'tempo': tempo,
                'marker': marker_text,
            }
            sections.append(section)
            section_idx += 1
        # filter out sections with fewer than 16 measures
        filtered_sections = [s for s in sections if s['measures'] >= 16]
        # re-index and relabel
        for idx, s in enumerate(filtered_sections):
            s['index'] = idx
            s['label'] = f'Section {idx+1}'
        return filtered_sections

