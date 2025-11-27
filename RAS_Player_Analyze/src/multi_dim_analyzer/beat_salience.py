"""
Beat Salience Analyzer (Dimension III) — MIDI-only.

Measures the local prominence (salience) of beats using a MIDI piano-roll
onset envelope and a deterministic beat grid derived from MIDI metadata.

Score ∈ [0, 1], where higher indicates beats stand out more strongly
from their local off-beat context.
"""

from typing import Optional, List
import numpy as np
import pretty_midi

from .config import BeatSalienceConfig
from .utils.beat_grid import DeterministicBeatGrid
from .utils.midi_processor import MIDIProcessor


class BeatSalienceAnalyzer:
    """Compute Beat Salience (local onset energy contrast) from MIDI only."""

    def __init__(self, config: BeatSalienceConfig):
        """
        Initialize analyzer with configuration.

        Args:
            config: BeatSalienceConfig with parameters for analysis.
        """
        self.config = config

    def analyze(self, pm: pretty_midi.PrettyMIDI) -> Optional[float]:
        """
        Compute beat salience score [0, 1] using MIDI-only methods.

        Args:
            pm: PrettyMIDI object.

        Returns:
            Float in [0, 1] representing beat salience, or None if insufficient data.
        """
        # 1) Deterministic beat grid
        beats = DeterministicBeatGrid.generate(pm)
        
        # Slice to max_duration_sec if specified
        actual_max_time = None
        if self.config.max_duration_sec is not None:
            total_duration = pm.get_end_time()
            max_duration_sec = self.config.max_duration_sec
            
            if total_duration > max_duration_sec:
                # Find the first beat that exceeds max duration
                beat_cutoff_idx = np.searchsorted(beats, max_duration_sec, side='right')
                
                if beat_cutoff_idx < len(beats):
                    actual_max_time = beats[beat_cutoff_idx]
                else:
                    actual_max_time = total_duration
                
                # Slice beats
                beats = beats[beats < actual_max_time]
                print(f"Beat Salience: Sliced to {len(beats)} beats, dur: {actual_max_time:.2f}s (Original {total_duration:.2f}s).")
        
        if len(beats) < self.config.min_beats_required:
            return None

        # Determine analysis time range for dynamic delta_window_sec calculation
        analysis_time_range = min(60.0, pm.get_end_time())
        if actual_max_time is not None:
            analysis_time_range = min(analysis_time_range, actual_max_time)
        
        # Compute dynamic delta_window_sec if not specified
        delta_window_sec = self.config.delta_window_sec
        if delta_window_sec is None:
            delta_window_sec = self._compute_half_beat_duration(pm, analysis_time_range)
            # print(f"Beat Salience: Using dynamic delta_window_sec = {delta_window_sec:.4f}s (half-beat duration)")
        else:
            print(f"Beat Salience: Using fixed delta_window_sec = {delta_window_sec:.4f}s")

        # 2) Piano-roll onset envelope (positive frame differences of energy)
        onset_env = self._compute_onset_envelope_piano_roll(
            pm, self.config.frame_hop_sec, self.config.use_velocity_weighted_proxy
        )
        if onset_env.size == 0:
            return None
        
        # Slice onset_env to match sliced beats if applicable
        if actual_max_time is not None:
            # Calculate the frame index corresponding to actual_max_time
            max_frame_idx = int(round(actual_max_time / max(self.config.frame_hop_sec, 1e-6)))
            # Slice onset_env to match the sliced duration
            onset_env = onset_env[:max_frame_idx + 1]  # +1 to include the boundary frame

        # 3) Map beats to frame indices
        beat_frames = self._beats_to_frames(beats)
        if beat_frames.size == 0:
            return None

        # 4) Compute local salience per beat
        saliences: List[float] = []
        for b in beat_frames:
            if 0 <= b < len(onset_env):
                s = self._compute_local_salience(onset_env, b, delta_window_sec=delta_window_sec)
                if s is not None:
                    if self.config.max_ratio_clip is not None:
                        s = min(s, self.config.max_ratio_clip)
                    saliences.append(s)

        if len(saliences) < self.config.min_beats_required:
            return None

        # 5) Aggregate and normalize
        raw_score = float(np.mean(saliences))
        score = self._normalize_score(raw_score)
        
        # Debug output if verbose
        if self.config.verbose:
            saliences_arr = np.array(saliences)
            print(f"\n=== Beat Salience Debug Info ===")
            print(f"Salience statistics:")
            print(f"  Count: {len(saliences)}")
            print(f"  Min: {np.min(saliences_arr):.4f}")
            print(f"  Max: {np.max(saliences_arr):.4f}")
            print(f"  Mean: {np.mean(saliences_arr):.4f}")
            print(f"  Median: {np.median(saliences_arr):.4f}")
            print(f"  Std: {np.std(saliences_arr):.4f}")
            print(f"  25th percentile: {np.percentile(saliences_arr, 25):.4f}")
            print(f"  75th percentile: {np.percentile(saliences_arr, 75):.4f}")
            print(f"\nOnset envelope statistics:")
            print(f"  Mean: {np.mean(onset_env):.4f}")
            print(f"  Std: {np.std(onset_env):.4f}")
            print(f"  25th percentile: {np.percentile(onset_env, 25):.4f}")
            print(f"  50th percentile: {np.percentile(onset_env, 50):.4f}")
            print(f"  75th percentile: {np.percentile(onset_env, 75):.4f}")
            print(f"  99th percentile: {np.percentile(onset_env, 99):.4f}")
            print(f"\nScore normalization:")
            print(f"  Raw score (mean salience): {raw_score:.4f}")
            print(f"  Normalization method: {self.config.normalization_method}")
            print(f"  Normalized score: {score:.4f}")
            print(f"================================\n")
        
        return score

    def _compute_onset_envelope_piano_roll(
        self,
        pm: pretty_midi.PrettyMIDI,
        frame_hop_sec: float,
        velocity_weighted: bool = False,
    ) -> np.ndarray:
        """
        Compute a MIDI-only onset envelope from piano-roll energy differences.

        Args:
            pm: PrettyMIDI object.
            frame_hop_sec: Frame hop in seconds (e.g., 0.01).
            velocity_weighted: If True, weight energy by velocity; otherwise sum binary.

        Returns:
            Onset envelope normalized to ~[0, 1] via 99th percentile scaling.
        """
        fs = max(1, int(round(1.0 / max(frame_hop_sec, 1e-4))))
        piano_roll = pm.get_piano_roll(fs=fs)  # (128, T)
        if piano_roll.size == 0:
            return np.array([], dtype=float)

        if velocity_weighted:
            # Use raw velocities (0-127) accumulated per frame
            energy = np.sum(piano_roll, axis=0)
        else:
            # Binary presence per pitch per frame
            energy = np.sum((piano_roll > 0).astype(np.float32), axis=0)

        # Positive frame differences to approximate onset strength (MIDI-flux)
        flux = np.maximum(np.diff(energy, prepend=energy[0]), 0.0)

        # Robust normalization by 99th percentile to handle heavy tails
        denom = np.percentile(flux, 99) + 1e-9
        onset_env = np.clip(flux / denom, 0.0, 1.0)
        return onset_env.astype(np.float32)

    def _beats_to_frames(self, beats: np.ndarray) -> np.ndarray:
        """Convert beat times to onset envelope frame indices."""
        h = max(self.config.frame_hop_sec, 1e-6)
        return np.round(beats / h).astype(int)
    
    def _get_average_tempo_in_range(
        self,
        pm: pretty_midi.PrettyMIDI,
        start_time: float,
        end_time: float
    ) -> float:
        """
        Get weighted average tempo in a time range, filtering invalid initial tempo.
        
        Args:
            pm: PrettyMIDI object.
            start_time: Start time in seconds (typically 0.0).
            end_time: End time in seconds (typically 60.0 or actual duration).
        
        Returns:
            Average BPM in the specified range.
        """
        # Get tempo map
        tempo_map = MIDIProcessor.get_tempo_map(pm)
        
        # Filter out tempo changes after end_time
        tempo_map = [(t, bpm) for t, bpm in tempo_map if t < end_time]

        if len(tempo_map) == 0:
            return 120.0  # Default BPM
        
        # Filter invalid initial tempo (if first and second are < 1 second apart)
        if len(tempo_map) >= 2:
            first_time, _ = tempo_map[0]
            second_time, _ = tempo_map[1]
            if second_time - first_time < 1.0:
                tempo_map = tempo_map[1:]  # Skip first tempo
        
        if len(tempo_map) == 0:
            return 120.0  # Default BPM
        
        # Calculate weighted average tempo
        # Each tempo segment contributes: bpm * duration
        total_weighted_bpm = 0.0
        total_duration = 0.0
        
        for i, (t, bpm) in enumerate(tempo_map):
            # Determine segment end time
            if i + 1 < len(tempo_map):
                segment_end = min(tempo_map[i + 1][0], end_time)
            else:
                segment_end = end_time
            
            # Segment start time (clamped to start_time)
            segment_start = max(t, start_time)
            
            # Segment duration
            segment_duration = max(0.0, segment_end - segment_start)
            
            if segment_duration > 0:
                total_weighted_bpm += bpm * segment_duration
                total_duration += segment_duration
        
        # Return weighted average, or default if no valid segments
        if total_duration > 0:
            return total_weighted_bpm / total_duration
        else:
            return 120.0  # Default BPM
    
    def _compute_half_beat_duration(
        self,
        pm: pretty_midi.PrettyMIDI,
        analysis_time_range: float
    ) -> float:
        """
        Compute half-beat duration based on average tempo and time signature.
        
        Args:
            pm: PrettyMIDI object.
            analysis_time_range: Time range to analyze (seconds, typically 60.0).
        
        Returns:
            Half-beat duration in seconds.
        """
        # Get average tempo in the analysis range
        avg_bpm = self._get_average_tempo_in_range(pm, 0.0, analysis_time_range)
        
        # Get time signature (use first one in the range, or default 4/4)
        ts_map = MIDIProcessor.get_time_signature_map(pm)
        # Find the time signature active at the start of analysis
        num, den = 4, 4  # Default
        for t, n, d in ts_map:
            if t <= analysis_time_range:
                num, den = n, d
            else:
                break
        
        # Calculate half-beat duration
        # For time signature num/den, one beat = (60/BPM) * (4/den)
        # Half beat = (60/BPM) * (4/den) * 0.5
        sec_per_beat = (60.0 / avg_bpm) * (4.0 / den)
        half_beat_duration = sec_per_beat * 0.5
        
        return half_beat_duration

    def _compute_local_salience(
        self,
        onset_env: np.ndarray,
        beat_frame: int,
        delta_window_sec: Optional[float] = None,
    ) -> Optional[float]:
        """
        Compute on-beat vs local off-beat energy ratio for a single beat.

        Args:
            onset_env: Onset envelope (frames).
            beat_frame: Index of beat frame.
            delta_window_sec: Window size in seconds. If None, uses config value.

        Returns:
            Salience ratio (>=0) or None if insufficient off-beat context.
        """
        # Use provided delta_window_sec or fall back to config
        window_sec = delta_window_sec if delta_window_sec is not None else self.config.delta_window_sec
        if window_sec is None:
            # Fallback to default if both are None
            window_sec = 0.2
        
        # Window radii in frames
        R = int(round(window_sec / max(self.config.frame_hop_sec, 1e-6)))
        E = int(round(self.config.exclude_self_radius / max(self.config.frame_hop_sec, 1e-6)))

        left = max(0, beat_frame - R)
        right = min(len(onset_env), beat_frame + R + 1)

        # Off-beat frames excluding immediate vicinity
        offbeat_idx = [
            j for j in range(left, right)
            if j != beat_frame and abs(j - beat_frame) > E
        ]

        # If no context, try one expansion
        if len(offbeat_idx) == 0:
            left = max(0, beat_frame - 2 * R)
            right = min(len(onset_env), beat_frame + 2 * R + 1)
            offbeat_idx = [
                j for j in range(left, right)
                if j != beat_frame and abs(j - beat_frame) > E
            ]
            if len(offbeat_idx) == 0:
                return None

        on_energy = float(onset_env[beat_frame])
        offbeat_values = onset_env[offbeat_idx]
        off_mean = float(np.mean(offbeat_values))
        
        # Use more robust off-beat mean: take max of mean and 25th percentile
        # This prevents extreme ratios when off_mean is very small
        if len(offbeat_values) > 0:
            off_percentile_25 = float(np.percentile(offbeat_values, 25))
            min_off_mean = max(off_mean, off_percentile_25, self.config.epsilon)
        else:
            min_off_mean = max(off_mean, self.config.epsilon)
        
        # Compute salience ratio with robust denominator
        salience = on_energy / min_off_mean
        return salience
    
    def _normalize_score(self, raw_score: float) -> float:
        """
        Normalize raw salience score to [0, 1] range using specified method.
        
        Avoids extreme values by ensuring minimum score threshold.
        
        Args:
            raw_score: Raw average salience ratio (>= 0).
        
        Returns:
            Normalized score in [0, 1], with minimum of min_score to avoid extreme values.
        """
        if raw_score < 0:
            return self.config.min_score
        
        method = self.config.normalization_method.lower()
        min_score = self.config.min_score
        
        if method == "clip":
            # Simple clipping (legacy behavior)
            normalized = float(np.clip(raw_score, 0.0, 1.0))
            # Ensure minimum score to avoid extreme values
            if normalized < min_score:
                normalized = min_score
        
        elif method == "log":
            # Logarithmic compression: log(1 + score) / log(1 + max_expected)
            # Assuming max_expected salience ratio is around 10
            max_expected = 10.0
            if raw_score > 0:
                normalized = np.log1p(raw_score) / np.log1p(max_expected)
                normalized = float(np.clip(normalized, 0.0, 1.0))
            else:
                normalized = min_score
            # Ensure minimum score threshold
            if normalized < min_score:
                normalized = min_score
        
        elif method == "sigmoid":
            # Sigmoid-like compression: maps [0, inf] to [0, 1] with expanded discrimination
            # For score <= 1.0: linear mapping to [0, 0.65] (elevated from [0, 0.5] to address overall low scores)
            # For score > 1.0: steeper sigmoid compression to [0.65, 1.0) (expanded range)
            if raw_score <= 1.0:
                # Linear mapping: [0, 1] -> [0, 0.65] (elevated to address overall low scores)
                # For very small scores, use smooth mapping to avoid 0.0
                if raw_score == 0.0:
                    normalized = min_score
                elif raw_score < 0.1:
                    # Smooth mapping for small scores: [0, 0.1] -> [min_score, 0.12]
                    normalized = min_score + (0.12 - min_score) * (raw_score / 0.1)
                else:
                    # Normal linear mapping for [0.1, 1.0] -> [0.12, 0.65]
                    normalized = 0.12 + (raw_score - 0.1) * (0.65 - 0.12) / (1.0 - 0.1)
            else:
                # Steeper sigmoid compression: [1, inf] -> [0.65, 1.0)
                # Using steeper curve: 0.65 + 0.35 * (1 - 1/(1 + 0.7*log(score)))
                # This maps: 1 -> 0.65, 2 -> ~0.78, 3 -> ~0.85, 5 -> ~0.92, 10 -> ~0.97
                # Provides better discrimination across the [1, 5] range (max_ratio_clip=5.0)
                # while maintaining elevated scores to address overall low score issue
                log_score = np.log(raw_score)
                normalized = 0.65 + 0.35 * (1.0 - 1.0 / (1.0 + 0.7 * log_score))
            
            normalized = float(np.clip(normalized, 0.0, 1.0))
        
        else:
            # Default to sigmoid if unknown method (same as sigmoid method above)
            if raw_score <= 1.0:
                if raw_score == 0.0:
                    normalized = min_score
                elif raw_score < 0.1:
                    normalized = min_score + (0.12 - min_score) * (raw_score / 0.1)
                else:
                    normalized = 0.12 + (raw_score - 0.1) * (0.65 - 0.12) / (1.0 - 0.1)
            else:
                log_score = np.log(raw_score)
                normalized = 0.65 + 0.35 * (1.0 - 1.0 / (1.0 + 0.7 * log_score))
            normalized = float(np.clip(normalized, 0.0, 1.0))
        
        # Ensure minimum score threshold to avoid extreme values
        if normalized < min_score:
            normalized = min_score
        
        return float(np.clip(normalized, min_score, 1.0))


