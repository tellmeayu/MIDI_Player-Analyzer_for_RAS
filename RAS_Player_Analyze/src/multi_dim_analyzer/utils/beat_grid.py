"""
Beat grid generation utilities.
Provides helpers for musical beat grids (consist with the playback system).
"""

import numpy as np
import pretty_midi

from core.precision_timer import PrecisionTimer
from .midi_processor import MIDIProcessor


class DeterministicBeatGrid:
    """Generate beat and bar boundary timestamps using PrettyMIDI helpers."""
    
    @staticmethod
    def generate(
        pm: pretty_midi.PrettyMIDI,
        start_time: float = 0.0
    ) -> np.ndarray:
        """
        Generate beat times (seconds) from MIDI metadata.

        Args:
            pm: PrettyMIDI object (which incl. tempo/time signature metadata).
            start_time: Start time in seconds (default 0.0). Set later to
                skip pickup/anacrusis if desired.

        Returns:
            Array of beat times in seconds, sorted chronologically.
        """
        end_time = pm.get_end_time()
        if end_time <= start_time:
            return np.array([], dtype=float)

        beats = pm.get_beats(start_time=start_time)
        return np.array(beats, dtype=float)
    
    @staticmethod
    def generate_musical_beats(
        pm: pretty_midi.PrettyMIDI,
        start_time: float = 0.0
    ) -> np.ndarray:
        """
        Generate beat times using musical beat interpretation.

        For compound meters (6/8, 9/8, 12/8), dotted quarter notes are treated
        as the beat unit. This matches the playback system's behavior where
        each musical beat represents the perceived pulse rather than the raw
        MIDI metadata beat.

        Args:
            pm: PrettyMIDI object.
            start_time: Start time in seconds (default 0.0).

        Returns:
            Array of musical beat times in seconds.
        """
        end_time = pm.get_end_time()
        if end_time <= start_time:
            return np.array([], dtype=float)

        ts_map = MIDIProcessor.get_time_signature_map(pm)
        tempo_map = MIDIProcessor.get_tempo_map(pm)

        # Collect all change points (tempo + time signature) within range
        change_points = {start_time, end_time}
        change_points.update(
            t for t, _, _ in ts_map if start_time <= t < end_time
        )
        change_points.update(
            t for t, _ in tempo_map if start_time <= t < end_time
        )
        sorted_points = sorted(change_points)

        beats = []
        next_beat_time = start_time

        for idx in range(len(sorted_points) - 1):
            seg_start = sorted_points[idx]
            seg_end = sorted_points[idx + 1]

            if seg_end <= seg_start or seg_start >= end_time:
                continue

            if next_beat_time < seg_start:
                next_beat_time = seg_start

            active_ts = DeterministicBeatGrid._get_active_time_signature(ts_map, seg_start)
            active_tempo = DeterministicBeatGrid._get_active_tempo(tempo_map, seg_start)

            musical_tempo = PrecisionTimer.convert_midi_tempo_to_musical_tempo(
                active_tempo, active_ts
            )
            if musical_tempo <= 0:
                continue

            beat_interval = 60.0 / musical_tempo
            if beat_interval <= 0:
                continue

            segment_end = min(seg_end, end_time)

            while next_beat_time < segment_end:
                if next_beat_time >= start_time:
                    beats.append(next_beat_time)
                next_beat_time += beat_interval

        return np.array(beats, dtype=float)
    
    @staticmethod
    def generate_bar_boundaries(
        pm: pretty_midi.PrettyMIDI,
        start_time: float = 0.0
    ) -> np.ndarray:
        """
        Generate bar (measure) boundaries from tempo and time signature.
        
        Useful for predictability analysis and segmentation.

        Args:
            pm: PrettyMIDI object.
            start_time: Start time in seconds (default 0.0). 

        Returns:
            Array of bar boundary (downbeat) times in seconds.
        """
        end_time = pm.get_end_time()
        if end_time <= start_time:
            return np.array([], dtype=float)

        downbeats = pm.get_downbeats(start_time=start_time)
        return np.array(downbeats, dtype=float)
    
    @staticmethod
    def get_beats_in_range(
        beats: np.ndarray,
        start_sec: float,
        end_sec: float
    ) -> np.ndarray:
        """
        Extract beats within a time range.
        
        Args:
            beats: Array of beat times.
            start_sec: Start time in seconds (inclusive).
            end_sec: End time in seconds (exclusive).
        
        Returns:
            Subset of beats within the range.
        """
        mask = (beats >= start_sec) & (beats < end_sec)
        return beats[mask]
    
    @staticmethod
    def count_beats(
        beats: np.ndarray,
        start_sec: float,
        end_sec: float
    ) -> int:
        """
        Count beats in a time range.
        
        Args:
            beats: Array of beat times.
            start_sec: Start time in seconds (inclusive).
            end_sec: End time in seconds (exclusive).
        
        Returns:
            Number of beats in range.
        """
        return len(DeterministicBeatGrid.get_beats_in_range(beats, start_sec, end_sec))

    @staticmethod
    def _get_active_time_signature(
        ts_map: list,
        time_sec: float
    ) -> tuple:
        """Return the active time signature at the given time."""
        active = (4, 4)
        for change_time, num, den in ts_map:
            if change_time <= time_sec:
                active = (num, den)
            else:
                break
        return active

    @staticmethod
    def _get_active_tempo(
        tempo_map: list,
        time_sec: float
    ) -> float:
        """Return the active MIDI tempo (BPM) at the given time."""
        active_tempo = 120.0
        for change_time, bpm in tempo_map:
            if change_time <= time_sec:
                active_tempo = bpm
            else:
                break
        return active_tempo
