"""
MIDI processing utilities for rhythm analysis.
Handles onset extraction, collapsing simultaneous events, filtering, and metadata retrieval.
"""

from typing import List, Optional, Tuple
import numpy as np
import pretty_midi


class MIDIProcessor:
    """Extract, filter, and collapse MIDI onset events."""
    
    @staticmethod
    def extract_onsets(
        pm: pretty_midi.PrettyMIDI,
        include_drums: bool = False,
        scope: str = 'global'
    ) -> np.ndarray:
        """
        Extract note onset times from MIDI.
        
        Args:
            pm: PrettyMIDI object.
            include_drums: If False, exclude drum tracks.
            scope: Analysis scope (currently only 'global' implemented).
        
        Returns:
            Sorted array of note onset times in seconds.
        """
        onsets = []
        
        for instr in pm.instruments:
            # Skip drum tracks if requested
            if not include_drums and instr.is_drum:
                continue
            
            # Collect onset times
            for note in instr.notes:
                onsets.append(note.start)
        
        if len(onsets) == 0:
            return np.array([], dtype=float)
        
        return np.sort(np.array(onsets, dtype=float))
    
    @staticmethod
    def extract_onsets_with_duration(
        pm: pretty_midi.PrettyMIDI,
        include_drums: bool = False,
        scope: str = 'global'
    ) -> List[Tuple[float, float]]:
        """
        Extract note onsets with their durations.
        
        Args:
            pm: PrettyMIDI object.
            include_drums: If False, exclude drum tracks.
            scope: Analysis scope (currently only 'global' implemented).
        
        Returns:
            List of (start_time, duration) tuples, sorted by start time.
        """
        notes = []
        
        for instr in pm.instruments:
            # Skip drum tracks if requested
            if not include_drums and instr.is_drum:
                continue
            
            # Collect onset times with durations
            for note in instr.notes:
                duration = note.end - note.start
                notes.append((note.start, duration))
        
        # Sort by start time
        notes.sort(key=lambda x: x[0])
        return notes
    
    @staticmethod
    def collapse_simultaneous(
        onsets: np.ndarray,
        epsilon_sec: float = 0.01
    ) -> np.ndarray:
        """
        Merge onsets that occur within epsilon threshold (simultaneous notes as chords).
        
        Args:
            onsets: Sorted array of onset times in seconds.
            epsilon_sec: Tolerance window for simultaneity in seconds.
        
        Returns:
            Collapsed array of onset times (chords collapsed to single event).
        
        Example:
            >>> onsets = np.array([0.0, 0.001, 0.5, 1.0])
            >>> collapse_simultaneous(onsets, 0.01)
            array([0.   , 0.5  , 1.   ])
        """
        if len(onsets) == 0:
            return np.array([], dtype=float)
        
        collapsed = [onsets[0]]
        
        for t in onsets[1:]:
            if t - collapsed[-1] > epsilon_sec:
                collapsed.append(t)
        
        return np.array(collapsed, dtype=float)
    
    @staticmethod
    def filter_iois(
        iois: np.ndarray,
        min_ioi_sec: float = 0.02,
        max_ioi_sec: Optional[float] = None
    ) -> np.ndarray:
        """
        Filter out extreme inter-onset intervals (very short notes).
        
        Args:
            iois: Array of inter-onset intervals in seconds.
            min_ioi_sec: Minimum IOI to keep (seconds).
            max_ioi_sec: Maximum IOI to keep (seconds). None = no upper limit.
        
        Returns:
            Filtered IOI array.
        
        Example:
            >>> iois = np.array([0.01, 0.1, 0.2, 10.0])
            >>> filter_iois(iois, min_ioi_sec=0.02, max_ioi_sec=1.0)
            array([0.1, 0.2])
        """
        mask = iois >= min_ioi_sec
        
        if max_ioi_sec is not None:
            mask &= iois <= max_ioi_sec
        
        return iois[mask]
    
    @staticmethod
    def get_tempo_map(pm: pretty_midi.PrettyMIDI) -> List[Tuple[float, float]]:
        """
        Get tempo change map from MIDI.
        
        Args:
            pm: PrettyMIDI object.
        
        Returns:
            List of (time_seconds, bpm) tuples sorted by time.
        """
        times, bpms = pm.get_tempo_changes()
        
        if len(times) == 0:
            # Default to 120 BPM if no explicit tempo
            return [(0.0, 120.0)]
        
        return list(zip(times, bpms))
    
    @staticmethod
    def get_time_signature_map(pm: pretty_midi.PrettyMIDI) -> List[Tuple[float, int, int]]:
        """
        Get time signature change map from MIDI.
        
        Args:
            pm: PrettyMIDI object.
        
        Returns:
            List of (time_seconds, numerator, denominator) tuples sorted by time.
        """
        ts_list = []
        
        if pm.time_signature_changes:
            for ts in pm.time_signature_changes:
                ts_list.append((ts.time, ts.numerator, ts.denominator))
        else:
            # Default to 4/4 if no explicit time signature
            ts_list = [(0.0, 4, 4)]
        
        return ts_list
    
    @staticmethod
    def get_end_time(pm: pretty_midi.PrettyMIDI) -> float:
        """
        Get end time of MIDI piece.
        
        Args:
            pm: PrettyMIDI object.
        
        Returns:
            End time in seconds.
        """
        return pm.get_end_time()
    
    @staticmethod
    def get_duration_seconds(pm: pretty_midi.PrettyMIDI) -> float:
        """
        Get total duration of MIDI piece.
        
        Args:
            pm: PrettyMIDI object.
        
        Returns:
            Duration in seconds.
        """
        return pm.get_end_time()
    
    @staticmethod
    def compute_iois(onsets: np.ndarray) -> np.ndarray:
        """
        Compute inter-onset intervals from onset times.
        
        Args:
            onsets: Sorted array of onset times in seconds.
        
        Returns:
            Array of inter-onset intervals (differences between consecutive onsets).
        
        Example:
            >>> onsets = np.array([0.0, 0.5, 1.0, 1.5])
            >>> compute_iois(onsets)
            array([0.5, 0.5, 0.5])
        """
        if len(onsets) < 2:
            return np.array([], dtype=float)
        
        return np.diff(onsets)
    
    @staticmethod
    def extract_valid_section(
        pm: pretty_midi.PrettyMIDI,
        min_duration_sec: float = 15.0
    ) -> pretty_midi.PrettyMIDI:
        """
        Extract the first valid musical section from MIDI.
        
        For MIDIs with multiple time signatures, extracts the first section
        that lasts at least min_duration_sec to skip short quantization artifacts.
        For MIDIs with a single time signature, returns the original MIDI unchanged.
        
        Args:
            pm: PrettyMIDI object to process.
            min_duration_sec: Minimum duration in seconds for a valid section.
        
        Returns:
            PrettyMIDI object containing the valid section, or original if single
            time signature or no valid section found.
        """
        # Get time signature map
        ts_map = MIDIProcessor.get_time_signature_map(pm)
        
        # If only one time signature, return original
        if len(ts_map) == 1:
            return pm
        
        # If multiple time signatures, find first valid section
        end_time = MIDIProcessor.get_end_time(pm)
        tempo_map = MIDIProcessor.get_tempo_map(pm)
        
        # Find first section with sufficient duration
        for i, (t_ts, num, den) in enumerate(ts_map):
            # Calculate duration of this section
            if i + 1 < len(ts_map):
                next_ts_time = ts_map[i + 1][0]
                duration = next_ts_time - t_ts
            else:
                duration = end_time - t_ts
            
            # If this section meets minimum duration, extract it
            if duration >= min_duration_sec:
                section_end = t_ts + duration if i + 1 < len(ts_map) else end_time
                
                # Create new PrettyMIDI object
                pm_extracted = pretty_midi.PrettyMIDI()
                
                # Copy instruments with filtered notes
                for instr in pm.instruments:
                    new_instr = pretty_midi.Instrument(
                        program=instr.program,
                        is_drum=instr.is_drum,
                        name=instr.name
                    )
                    
                    # Filter notes in the valid section
                    for note in instr.notes:
                        if t_ts <= note.start < section_end:
                            # Adjust note times relative to section start
                            adjusted_note = pretty_midi.Note(
                                velocity=note.velocity,
                                pitch=note.pitch,
                                start=note.start - t_ts,
                                end=note.end - t_ts
                            )
                            new_instr.notes.append(adjusted_note)
                    
                    if len(new_instr.notes) > 0:
                        pm_extracted.instruments.append(new_instr)
                
                # Set time signature
                pm_extracted.time_signature_changes = [
                    pretty_midi.TimeSignature(num, den, 0.0)
                ]
                
                # Set tempo (find tempo active at section start)
                active_tempo = 120.0  # default
                for t_tempo, bpm in tempo_map:
                    if t_tempo <= t_ts:
                        active_tempo = bpm
                    else:
                        break
                
                pm_extracted.tempo_changes = [
                    pretty_midi.TempoChange(active_tempo, 0.0)
                ]
                
                return pm_extracted
        
        # If no section meets minimum duration, return original
        return pm