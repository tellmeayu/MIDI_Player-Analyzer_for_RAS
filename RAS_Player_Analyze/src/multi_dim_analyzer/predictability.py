"""
Predictability Analyzer (Dimension II).

Measures metrical conformance using a weighted combination of macro-level beat
coverage and micro-level weighted alignment analysis. High predictability
indicates notes align with metrical positions; low predictability indicates
irregular or syncopated patterns.

Theory:
  - Filter out very short notes (ornaments) using duration threshold
  - Macro layer: Beat coverage - percentage of beats that have aligned notes
    (all beats treated equally, no metrical hierarchy). Off-beat notes receive
    distance-based penalties. Tolerance is dynamically calculated based on tempo
    (1/4 beat duration, 25% of beat interval).
  - Micro layer: Weighted alignment analysis per bar - notes aligned to strong
    metrical positions (higher weights) contribute more to the score. Uses 16th
    note grid with metrical weight vectors.
  - Weighted fusion: final = macro_weight * macro_score + (1 - macro_weight) * micro_score
    (macro layer typically dominant, default 0.8)

Result: Score in [0, 1] where:
  - 1.0: Perfect metrical alignment (all beats covered, notes on strong positions)
  - 0.5: Moderate alignment
  - 0.0: Poor alignment (many missing beats, notes off-beat or on weak positions)
"""

from typing import Optional, Tuple
import numpy as np
import pretty_midi

from core.precision_timer import PrecisionTimer
from .config import PredictabilityConfig
from .utils.midi_processor import MIDIProcessor
from .utils.beat_grid import DeterministicBeatGrid


class PredictabilityAnalyzer:
    """Compute Predictability (Dimension II) from MIDI using beat coverage and weighted alignment."""
    
    def __init__(self, config: PredictabilityConfig):
        """
        Initialize analyzer with configuration.
        
        Args:
            config: PredictabilityConfig with parameters for analysis.
        """
        self.config = config
    
    def analyze(self, pm: pretty_midi.PrettyMIDI) -> Optional[float]:
        """
        Compute normalized predictability score [0, 1] using weighted macro/micro approach.
        
        Args:
            pm: PrettyMIDI object with note events and metadata.
        
        Returns:
            Float in [0, 1] representing predictability, or None if insufficient data.
        """
        score, _ = self.analyze_with_diagnostics(pm)
        return score
    
    def analyze_with_diagnostics(
        self,
        pm: pretty_midi.PrettyMIDI
    ) -> Tuple[Optional[float], dict]:
        """
        Analyze predictability with detailed diagnostics using weighted macro/micro approach.
        
        Args:
            pm: PrettyMIDI object.
        
        Returns:
            Tuple of (score, diagnostics_dict).
            
        Diagnostics include:
            - final_score: Combined macro/micro predictability score
            - macro_score: Macro-level beat coverage score (with off-beat penalties)
            - micro_score: Micro-level weighted alignment score
            - macro_weight: Weight used for macro layer
            - beats_covered: Number of beats with aligned notes
            - total_beats: Total number of beats analyzed
            - coverage_ratio: beats_covered / total_beats
            - off_beat_penalty: Total penalty from off-beat notes
            - on_beat_notes: Number of notes within beat tolerance
            - off_beat_notes: Number of notes outside beat tolerance (penalized)
            - total_filtered_notes: Total notes after duration filtering
            - alignment_score: Total weighted alignment score (micro layer)
            - max_alignment_score: Maximum possible alignment score (micro layer)
            - total_bars: Total number of bars
            - bars_processed: Number of bars with notes
            - bars_skipped: Number of empty bars
        """
        # 1. Get bar and beat boundaries
        bars = DeterministicBeatGrid.generate_bar_boundaries(pm)
        if len(bars) < self.config.min_bars_required + 1:
            return None, {'error': 'insufficient_bars'}
        beats = DeterministicBeatGrid.generate_musical_beats(pm)
        print(f"all {len(beats)} beats in {len(bars)} bars.")
        ts_map = MIDIProcessor.get_time_signature_map(pm)

        if hasattr(self.config, 'max_duration_sec') and self.config.max_duration_sec:
            total_duration = pm.get_end_time()
            max_duration_sec = self.config.max_duration_sec
            
            if total_duration > max_duration_sec:
                # find the first boundary that exceeds max duration
                bar_cutoff_idx = np.searchsorted(bars, max_duration_sec, side='right')

                if bar_cutoff_idx < len(bars):
                    actual_max_time = bars[bar_cutoff_idx]
                else:
                    actual_max_time = total_duration
                
                # slice beats and bars
                beats = beats[beats < actual_max_time]
                bars = bars[bars <= actual_max_time]
                print(f" Sliced to {len(beats)} beats in {len(bars)} bars, dur: {actual_max_time} (Original {total_duration}).")

                slice_end_time = actual_max_time
            else:
                slice_end_time = None
        else:
            slice_end_time = None

        # 2. Extract onsets with duration and filter short notes
        onsets_with_duration = MIDIProcessor.extract_onsets_with_duration(
            pm, include_drums=False
        )
        if slice_end_time is not None:
            onsets_with_duration = [
                (onset, duration) for onset, duration in onsets_with_duration if onset < slice_end_time
            ]

        if not onsets_with_duration:
            return 0.0, {'note': 'No valid notes'}
            
        else:
            onsets_with_duration = np.array(onsets_with_duration)
            filtered_notes = onsets_with_duration[onsets_with_duration[:,1] > self.config.note_duration_threshold_sec]

            if not filtered_notes.any():
                return 0.0, {'note': 'No valid notes after duration filtering'}

            onsets = filtered_notes[:, 0]

        # --- 3. Macro-level Analysis (Beat Coverage) ---
        total_notes_macro = len(onsets)
        on_beat_notes = 0  # For diagnostics
        off_beat_notes = 0  # For diagnostics
        beats_with_aligned_notes = set()
        off_beat_penalty = 0.0
        total_beats = len(beats)

        if total_notes_macro > 0 and len(beats) > 0:
            # Calculate dynamic tolerance based on musical beat interval
            tolerance = self._calculate_macro_beat_tolerance(pm, beats)
            
            # First pass: find aligned notes and calculate off-beat penalties
            for onset_time in onsets:
                min_dist = np.min(np.abs(beats - onset_time))
                nearest_beat_idx = np.argmin(np.abs(beats - onset_time))
                
                if min_dist <= tolerance:
                    # On-beat: note aligns within tolerance
                    beats_with_aligned_notes.add(nearest_beat_idx)
                    on_beat_notes += 1
                else:
                    # Off-beat: distance-based penalty
                    off_beat_notes += 1
                    max_penalty_dist = tolerance * 2.0
                    penalty_factor = min(1.0, (min_dist - tolerance) / 
                                        (max_penalty_dist - tolerance))
                    off_beat_penalty += penalty_factor

            # Coverage score: beats with notes / total beats
            coverage_score = len(beats_with_aligned_notes) / len(beats) if len(beats) > 0 else 0.0

            # Apply off-beat penalty (reduce score based on off-beat notes)
            # Max 30% reduction
            penalty_reduction = min(0.3, off_beat_penalty / max(len(onsets), 1))
            macro_score = max(0.0, coverage_score - penalty_reduction)
        else:
            macro_score = 1.0  # Perfectly predictable if no notes or no beats
        
        # --- 4. Micro-level Analysis (Weighted Alignment) ---
        onsets_collapsed = MIDIProcessor.collapse_simultaneous(onsets, epsilon_sec=0.01)
        
        total_alignment_score = 0.0
        total_max_alignment_score = 0.0
        bars_processed = 0
        bars_skipped = 0
        
        for i in range(len(bars) - 1):
            bar_start = bars[i]
            bar_end = bars[i + 1]
            
            # Get time signature for this bar
            time_sig = self._get_time_signature_at_time(ts_map, bar_start)
            
            # Get grid resolution
            resolution = self._get_grid_resolution(time_sig)
            
            # Build weight vector (metrical weights for 16th note grid)
            weights = self._build_weight_vector(time_sig, resolution, self.config.style)
            
            # Extract onsets in bar range
            onsets_in_bar = onsets_collapsed[
                (onsets_collapsed >= bar_start) & (onsets_collapsed < bar_end)
            ]
            
            if len(onsets_in_bar) == 0:
                bars_skipped += 1
                continue
            
            # Populate rhythm vector (quantize notes to grid positions)
            rhythm_vec = self._populate_rhythm_vector(
                onsets_in_bar, bar_start, bar_end, resolution, self.config.chord_handling
            )
            
            # Calculate alignment: sum of weights at note positions
            alignment_score = float(np.sum(weights[rhythm_vec > 0]))
            
            # Maximum alignment: sum of all weights (all positions filled)
            max_alignment_score = float(np.sum(weights))
            
            total_alignment_score += alignment_score
            total_max_alignment_score += max_alignment_score
            bars_processed += 1
        
        # Normalized alignment score
        if total_max_alignment_score > 0:
            micro_score = total_alignment_score / total_max_alignment_score
        else:
            micro_score = 1.0  # Perfectly predictable if no bars processed
        
        micro_score = float(np.clip(micro_score, 0.0, 1.0))
        
        # --- 5. Combine Scores ---
        if micro_score == 0.0:
            final_score = macro_score
            macro_w = 1.0
        else:
            macro_w = self.config.macro_weight
            final_score = (macro_score * macro_w) + (micro_score * (1.0 - macro_w))
        final_score = float(np.clip(final_score, 0.0, 1.0))
        
        diagnostics = {
            'final_score': final_score,
            'macro_score': float(macro_score),
            'micro_score': micro_score,
            'macro_weight': macro_w,
            # Macro layer diagnostics
            'beats_covered': len(beats_with_aligned_notes),
            'total_beats': total_beats,
            'coverage_ratio': len(beats_with_aligned_notes) / total_beats if total_beats > 0 else 0.0,
            'off_beat_penalty': float(off_beat_penalty),
            'on_beat_notes': on_beat_notes,
            'off_beat_notes': off_beat_notes,
            'total_filtered_notes': total_notes_macro,
            # Micro layer diagnostics
            'alignment_score': float(total_alignment_score),
            'max_alignment_score': float(total_max_alignment_score),
            'total_bars': len(bars) - 1,
            'bars_processed': bars_processed,
            'bars_skipped': bars_skipped,
            'resolution_mode': self.config.resolution_mode,
            'chord_handling': self.config.chord_handling,
            'style': self.config.style,
        }
        
        return final_score, diagnostics
    
    def _calculate_macro_beat_tolerance(
        self,
        pm: pretty_midi.PrettyMIDI,
        beats: np.ndarray
    ) -> float:
        """
        Calculate dynamic beat tolerance based on musical beat interval.

        Prefers the median interval of the musical beat grid; falls back to
        tempo/time signature derived calculations when insufficient beats exist.
        """
        beat_interval = None

        if len(beats) >= 2:
            intervals = np.diff(beats)
            positive_intervals = intervals[intervals > 0]
            if positive_intervals.size > 0:
                beat_interval = float(np.median(positive_intervals))

        if beat_interval is None or beat_interval <= 0:
            tempo_map = MIDIProcessor.get_tempo_map(pm)
            midi_tempo = tempo_map[0][1] if tempo_map and tempo_map[0][1] > 0 else 120.0
            ts_map = MIDIProcessor.get_time_signature_map(pm)
            time_sig = self._get_time_signature_at_time(ts_map, 0.0)
            musical_tempo = PrecisionTimer.convert_midi_tempo_to_musical_tempo(
                midi_tempo,
                time_sig
            )
            beat_interval = 60.0 / musical_tempo if musical_tempo > 0 else 0.5

        dynamic_tolerance = beat_interval * 0.25  # 1/4 beat
        return float(max(0.05, min(0.2, dynamic_tolerance)))
    
    def _get_time_signature_at_time(
        self,
        ts_map: list,
        time: float
    ) -> Tuple[int, int]:
        """
        Get time signature active at given time.
        
        Args:
            ts_map: List of (time, numerator, denominator) tuples.
            time: Time in seconds.
        
        Returns:
            Tuple of (numerator, denominator).
        """
        # Find the last time signature change before or at this time
        active_ts = (4, 4)  # default
        for t_ts, num, den in ts_map:
            if t_ts <= time:
                active_ts = (num, den)
            else:
                break
        return active_ts
    
    def _get_grid_resolution(self, time_sig: Tuple[int, int]) -> int:
        """
        Get grid resolution (slots per bar) for given time signature.
        
        Args:
            time_sig: Tuple of (numerator, denominator).
        
        Returns:
            Number of slots per bar.
        """
        num, den = time_sig
        
        if self.config.resolution_mode == 'fixed':
            return self.config.fixed_resolution if self.config.fixed_resolution is not None else 16
        
        # Auto mode: determine resolution based on time signature
        if den == 4:  # Simple meters
            return 4 * num  # Sixteenth-note grid
        elif den == 8:  # Compound meters
            if num == 6:
                return 12  # Eighth-note triplet resolution
            elif num == 9:
                return 18
            elif num == 12:
                return 24
            else:
                return 4 * num  # Fallback
        else:
            # Fallback for unknown signatures
            return 4 * num
    
    def _build_weight_vector(
        self,
        time_sig: Tuple[int, int],
        resolution: int,
        style: str
    ) -> np.ndarray:
        """
        Build metrical weight vector for given time signature and resolution.
        
        Uses hardcoded patterns for common time signatures.
        
        Args:
            time_sig: Tuple of (numerator, denominator).
            resolution: Number of slots per bar.
            style: Weight style ('neutral', 'swing', 'folk') - currently only 'neutral' implemented.
        
        Returns:
            Array of weights, one per slot.
        """
        num, den = time_sig
        
        # Hardcoded patterns for common time signatures
        if (num, den) == (4, 4) and resolution == 16:
            weights = np.array([5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1], dtype=float)
        elif (num, den) == (3, 4) and resolution == 12:
            weights = np.array([4, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1], dtype=float)
        elif (num, den) == (2, 4) and resolution == 8:
            weights = np.array([5, 1, 2, 1, 4, 1, 2, 1], dtype=float)
        elif (num, den) == (6, 8) and resolution == 12:
            weights = np.array([4, 2, 2, 3, 2, 2, 4, 2, 2, 3, 2, 2], dtype=float)
        elif (num, den) == (9, 8) and resolution == 18:
            weights = np.array([5, 2, 2, 3, 2, 2, 4, 2, 2, 3, 2, 2, 4, 2, 2, 3, 2, 2], dtype=float)
        elif (num, den) == (12, 8) and resolution == 24:
            weights = np.array([5, 2, 2, 3, 2, 2, 4, 2, 2, 3, 2, 2, 5, 2, 2, 3, 2, 2, 4, 2, 2, 3, 2, 2], dtype=float)
        else:
            # Fallback: uniform weights
            weights = np.ones(resolution, dtype=float)
        
        return weights
    
    def _populate_rhythm_vector(
        self,
        onsets: np.ndarray,
        bar_start: float,
        bar_end: float,
        resolution: int,
        chord_handling: str
    ) -> np.ndarray:
        """
        Populate rhythm vector by quantizing onsets to grid slots.
        
        Args:
            onsets: Array of onset times in seconds (within bar range).
            bar_start: Bar start time in seconds.
            bar_end: Bar end time in seconds.
            resolution: Number of slots per bar.
            chord_handling: 'binary' or 'stacked'.
        
        Returns:
            Binary or stacked rhythm vector.
        """
        rhythm_vec = np.zeros(resolution, dtype=int)
        
        if len(onsets) == 0:
            return rhythm_vec
        
        slot_duration = (bar_end - bar_start) / resolution
        
        for onset in onsets:
            rel_time = onset - bar_start
            slot_idx = int(np.floor(rel_time / slot_duration))
            slot_idx = min(slot_idx, resolution - 1)  # Clamp to valid range
            
            if chord_handling == 'binary':
                rhythm_vec[slot_idx] = 1
            else:  # stacked
                rhythm_vec[slot_idx] += 1
        
        return rhythm_vec
    

