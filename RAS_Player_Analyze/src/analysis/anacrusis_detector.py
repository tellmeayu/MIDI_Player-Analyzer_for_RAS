#!/usr/bin/env python3
"""
Anacrusis Detection Module

This module specializes in detecting anacrusis (pickup beats) in musical audio.
Anacrusis detection is crucial for:
- MIDI timing correction
- Musical structure analysis
- Proper beat numbering
- Performance analysis

Main class:
- AnacrusisDetector: Class-based detector with MIDI and audio support

Created: September 2025
Refactored: September 2025
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import librosa

# Try to import madmom - gracefully handle if unavailable (Python 3.11 compatibility issue)
try:
    import madmom
    from pydub import AudioSegment
    MADMOM_AVAILABLE = True
except (ImportError, RuntimeError, OSError) as e:
    MADMOM_AVAILABLE = False
    MADMOM_ERROR = str(e)

# Debug mode: Enable detailed logging and simple detection mode
# Set environment variable: ANACRUSIS_DEBUG=1 for logging
# Set environment variable: ANACRUSIS_SIMPLE=1 to bypass pattern validation
DEBUG_MODE = os.environ.get('ANACRUSIS_DEBUG', '0') == '1'
SIMPLE_MODE = os.environ.get('ANACRUSIS_SIMPLE', '0') == '1'

def _log(message: str):
    """Log debug messages if debug mode is enabled."""
    if DEBUG_MODE:
        print(f"[ANACRUSIS DEBUG] {message}")


class AnacrusisDetector:
    """
    Detects anacrusis (pickup measures) from MIDI files for timing correction.
    
    This detector is specifically designed for MIDI timing correction and finding 
    the first true downbeat. It synthesizes a 15-second audio excerpt from the MIDI
    and analyzes it for anacrusis detection.
    """

    def __init__(
        self,
        search_window_beats: Optional[int] = None,
        detection_mode: str = "prob",
        sample_rate: int = 22050,
    ):
        """
        Initialize the AnacrusisDetector.

    Args:
            search_window_beats: Number of beats to search for first downbeat.
                If None, will be set to meter * 2.
            detection_mode: Detection strategy ("prob" for probability-based).
                Currently only "prob" is implemented.
            sample_rate: Target sample rate for MIDI synthesis.
        """
        self.search_window_beats = search_window_beats
        self.detection_mode = detection_mode
        self.sample_rate = int(sample_rate)

        # Check madmom availability
        if not MADMOM_AVAILABLE:
            print(f"WARNING: madmom library unavailable - anacrusis detection disabled")
            print(f"Reason: {MADMOM_ERROR}")

        # Log debug/simple mode status
        if DEBUG_MODE:
            print(f"[ANACRUSIS DEBUG] Debug mode enabled")
        if SIMPLE_MODE:
            print(f"[ANACRUSIS SIMPLE] Simple detection mode (no pattern validation)")

        # Validate detection mode
        if self.detection_mode not in ["prob"]:
            raise ValueError(f"Unsupported detection_mode: {self.detection_mode}")

    def detect_anacrusis(
        self,
        midi_path: str,
        sf2_path: str,
        meter: List[int],
    ) -> Dict:
        """
            Detect anacrusis from a pre-trimmed MIDI file for timing correction.

            This is the main method that encapsulates the complete anacrusis detection
            process: audio synthesis → beat/downbeat analysis. The input MIDI is assumed
            to be pre-trimmed to start at t=0.

            Args:
                midi_path: Path to the pre-trimmed MIDI file.
                sf2_path: Path to SoundFont (.sf2) for synthesis.
                meter: The meter to assume for bar tracking (e.g., [3] for 3/4).

        Returns:
            Dictionary containing anacrusis detection results for MIDI correction:
            {
                'beats': np.ndarray,                    # Beat times
                'downbeats': np.ndarray,                # (N,2) time, beat_number pairs
                'first_downbeat_time': float | None,    # First true downbeat time
                'anacrusis_beats': int,                 # Raw anacrusis count
                'effective_anacrusis_beats': int,       # Modulo meter
                'downbeat_probabilities': np.ndarray,   # (N,2) time, probability pairs
                'analysis_window': float,               # Duration analyzed (15.0s)
                'meter': int,                          # Meter value used
                'total_beats_detected': int,           # Total beats found
                'midi_file': str,                      # Source MIDI path
                'strategy': str,                       # Detection strategy used
                'warnings': List[str],                 # Analysis warnings
                'errors': List[str],                   # Error messages if any
            }
        """
        # Check if madmom is available
        if not MADMOM_AVAILABLE:
            return self._create_error_result(
                f"madmom library unavailable (Python 3.11 incompatibility). "
                f"Please install Python 3.9-3.10 or use madmom-py3.10-compat fork. "
                f"Error: {MADMOM_ERROR}",
                meter, midi_path
            )

        # Synthesize audio from the pre-trimmed MIDI clip (starts at 0.0)
        try:
            from .preprocessor import load_midi, synthesize_wav
            
            # Load the pre-trimmed MIDI (already starts at t=0)
            trimmed_midi = load_midi(midi_path)

            first_note_time = None
            try:
                # Use the efficient get_onsets() to find the first note's start time
                onsets = trimmed_midi.get_onsets()
                if len(onsets) > 0:
                    first_note_time = onsets[0]
            except Exception as e:
                # If finding the time fails, we can still proceed without filtering
                print(f"Could not determine first note time: {e}")

            # Synthesize the entire clip
            wav_path = synthesize_wav(
                trimmed_midi, 
                sf2_path=sf2_path, 
                sr=self.sample_rate, 
                suffix="anacrusis"
            )
        
        except Exception as e:
            return self._create_error_result(f"MIDI synthesis failed: {e}", meter, midi_path)

        # Analyze the synthesized audio
        result = self._detect_anacrusis_from_audio_path(wav_path, meter, first_note_time)
        result["midi_file"] = midi_path

        return result

    def _detect_anacrusis_from_audio_path(
        self,
        audio_path: str,
        meter: List[int],
        first_note_time: Optional[float] = None
    ) -> Dict:
        """
        Core anacrusis detection logic for an audio file.

        Args:
            audio_path: Path to audio file.
            meter: Meter specification.

        Returns:
            Dictionary with detection results.
        """
        if not os.path.exists(audio_path):
            return self._create_error_result(f"Audio file not found: {audio_path}", meter)

        if not meter or len(meter) == 0:
            return self._create_error_result("Invalid meter specification", meter)

        meter_value = meter[0]
        search_bar = 3 # default to search three measures
        warnings = []
        errors = []

        try:
            # Audio Cleaning (a short fade-in at the begining)
            try:
                fade_in_duration = 100  # 100ms fadein
                audio = AudioSegment.from_wav(audio_path)
                cleaned_audio = audio.fade_in(fade_in_duration)

                # overwrite original file with cleaned version
                cleaned_audio.export(audio_path, format="wav")
            except Exception as audio_clean_err:
                warnings.append(f"Audio cleaning with pydub failed: {audio_clean_err}")
                
            beat_proc_start = time.time()
            # Load the audio signal 
            processed_signal = madmom.audio.signal.Signal(audio_path, sample_rate=self.sample_rate, num_channels=1)
            analysis_duration = len(processed_signal) / processed_signal.sample_rate

            # STAGE 1: Beat tracking
            beat_act = madmom.features.beats.RNNBeatProcessor()(processed_signal)
            beat_proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
            all_beats = beat_proc(beat_act)

            # estimate tempo to set search window
            try:
                tempo_proc = madmom.features.tempo.TempoEstimationProcessor(fps=100)
                tempi = tempo_proc(beat_act)
                # Use the first (most likely) tempo est
                estimated_tempo = tempi[0][0] if (tempi is not None and len(tempi)>0) else 120.0
                print(f"[DETECTOR] Estimated tempo: {estimated_tempo:.2f} BPM")
                
                # determine search window based on tempo
                if estimated_tempo < 60:
                    search_bar = 5
                else:
                    search_bar = 3
            except Exception as tempo_err:
                warnings.append(f"Tempo estimation failed: {tempo_err}. Using default.")
                search_bar = 3

            if all_beats.size == 0:
                return self._create_error_result("No beats detected", meter, "", warnings)

            # STAGE 2: Downbeat probability analysis
            bar_proc = madmom.features.downbeats.RNNBarProcessor()
            downbeat_probs = bar_proc((processed_signal, all_beats))

            # Calculate RMS from madmom signal
            hop_length = processed_signal.sample_rate // 100  # For 100 FPS
            frame_length = hop_length * 2

            rms = librosa.feature.rms(y=processed_signal, frame_length=frame_length, hop_length=hop_length)[0]
            # Normalize to 0-1
            rms_normalized = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-6)

            # Create new array with rms (time, downbeat_prob, rms_score)
            beat_frames = librosa.time_to_frames(downbeat_probs[:, 0], sr=processed_signal.sample_rate, hop_length=hop_length)
            beat_frames = np.minimum(beat_frames, len(rms_normalized) - 1)
            rms_scores = rms_normalized[beat_frames]
            combined_probs = np.c_[downbeat_probs, rms_scores]

            if first_note_time is not None and len(combined_probs) > 0:
                delta = 0.1    # allow 100ms tolerance window
                filter_time = first_note_time - delta
                original_count = len(combined_probs)
                combined_probs = combined_probs[combined_probs[:, 0] >= filter_time]
                
            # formatted = np.array2string(combined_probs[:10], formatter={'float_kind': lambda x: f"{x:.3f}"})
            # print(f"combined probs:\n{formatted}")

            # CRITICAL: Validate array shape to prevent memory corruption
            if combined_probs.ndim != 2 or combined_probs.shape[1] != 3:
                return self._create_error_result(
                    f"Invalid combined_probs shape: {combined_probs.shape}",
                    meter, "", warnings
                )
            print(f"madmom beat process duration: {time.time()-beat_proc_start}")

            # Determine search window (3 measure as default)
            search_window = self.search_window_beats
            if search_window is None:
                search_window = meter_value * search_bar

            # Never exceed available probabilities
            search_window = min(len(combined_probs), int(search_window))

            if search_window == 0:
                return self._create_error_result("No beats in search window", meter, "", warnings)
            
            # Find first downbeat - pattern-based validation
            first_downbeat_idx, confidence = self._find_first_downbeat_with_pattern_validation(
                    combined_probs, meter_value, search_window
                )

            # Fall back to no anacrusis if confidence is very low
            if confidence < 0.1:
                warnings.append("Low downbeat confidence - assuming no anacrusis")
                first_downbeat_idx = 0
                print("Low confidence - defaulting to no anacrusis")
            
            original_candidate_idx = int(first_downbeat_idx) 
            # If chosen downbeat is in a later bar, step back by full measure to the earlies
            if first_downbeat_idx > 0 and meter_value > 0:
                earliest_idx = int(first_downbeat_idx % meter_value)
                if earliest_idx != first_downbeat_idx:
                    print(f"[DETECTOR] Adjusting downbeat idx {first_downbeat_idx} -> earliest {earliest_idx} within clip")
                first_downbeat_idx = earliest_idx

            first_downbeat_time = combined_probs[first_downbeat_idx, 0]
            anacrusis_beats = original_candidate_idx
            # Calculate effective anacrusis (modulo meter)
            effective_anacrusis_beats = anacrusis_beats % meter_value
            if anacrusis_beats > 0 and effective_anacrusis_beats == 0:
                effective_anacrusis_beats = 0
                warnings.append("Anacrusis spans full bars, treating as no anacrusis")

            # Generate beat numbering
            num_beats = len(all_beats)
            beat_numbers = self._generate_beat_numbering(
                num_beats, anacrusis_beats, effective_anacrusis_beats, meter_value
            )

            # Combine times and beat numbers
            corrected_downbeats = np.column_stack((
                all_beats.astype(np.float32),
                beat_numbers.astype(np.int32)
            ))

            return {
                'beats': all_beats.astype(np.float32),
                'downbeats': corrected_downbeats,
                'first_downbeat_time': float(first_downbeat_time),
                'downbeat_probabilities': downbeat_probs.astype(np.float32),
                'analysis_window': float(analysis_duration),
                'meter': int(meter_value),
                'total_beats_detected': int(num_beats),
                'strategy': self.detection_mode,
                'search_window_used': int(search_window),
                'warnings': warnings,
                'errors': errors,
            }

        except Exception as e:
            print(f"❌ Exception caught: {type(e).__name__}: {e}")
            import traceback
            if DEBUG_MODE:
                traceback.print_exc()
            return self._create_error_result(f"Analysis failed: {e}", meter, "", warnings)

    def  _generate_beat_numbering(
        self,
        num_beats: int,
        anacrusis_beats: int,
        effective_anacrusis_beats: int,
        meter_value: int,
    ) -> np.ndarray:
        """Generate proper beat numbering accounting for anacrusis."""
        if effective_anacrusis_beats > 0:
            # Calculate anacrusis beat numbers
            anacrusis_start_beat = meter_value - effective_anacrusis_beats + 1
            anacrusis_beat_numbers = np.arange(anacrusis_start_beat, meter_value + 1)
            
            # Calculate main sequence beat numbers
            main_beats_count = num_beats - effective_anacrusis_beats
            main_beat_numbers = (np.arange(main_beats_count) % meter_value) + 1
            
            # Combine and trim to exact length
            beat_numbers = np.concatenate((anacrusis_beat_numbers, main_beat_numbers))
            # Trim to ensure beat_numbers matches the expected number of beats
            beat_numbers = beat_numbers[:num_beats]     
        else:
            # No effective anacrusis, re-center around identified downbeat
            beat_numbers = (np.arange(num_beats) - anacrusis_beats) % meter_value + 1

        return beat_numbers

    def _find_first_downbeat_with_pattern_validation(
        self,
        downbeat_data: np.ndarray,  # should be 3-colum array with rms
        meter_value: int,
        search_window: int,
    ) -> Tuple[int, float]:
        """Find first downbeat by validating pattern consistency.

        Instead of simply choosing the beat with highest probability,
        this method tests multiple candidates and validates that the
        downbeat pattern repeats consistently every meter_value beats.

        Args:
            downbeat_probs: Array of (time, probability) pairs from RNNBarProcessor
            meter_value: Beats per measure (e.g., 3 for 3/4, 4 for 4/4)
            search_window: Maximum number of beats to search

        Returns:
            Tuple of (first_downbeat_index, confidence_score)
        """
        print(f"Pattern validation: meter={meter_value}, search_window={search_window}")

        # Get top K candidates in search window, with low-threshold filtering
        # We test multiple candidates, not just the single maximum
        top_k = min(3, search_window)
        candidate_probs = downbeat_data[:search_window, 1]

        # Handle NaN values
        valid_indices = ~np.isnan(candidate_probs)
        if not np.any(valid_indices):
            return 0, 0.0  # No valid probabilities, assume no anacrusis

        valid_probs = candidate_probs[valid_indices]
        valid_idx_map = np.where(valid_indices)[0]
        if len(valid_probs) == 0:
            return 0, 0.0

        # Low relative thresholding tailored for classical (often low confidence)
        max_prob = float(np.nanmax(valid_probs)) if len(valid_probs) else 0.0
        # Keep threshold low to not miss candidates; also apply absolute floor
        rel_ratio = 0.4  # can be tuned; lower means more permissive
        abs_floor = 0.1
        thr = max(max_prob * rel_ratio, abs_floor)
        mask = (candidate_probs >= thr) & valid_indices
        prelim_indices = np.where(mask)[0]

        if len(prelim_indices) > 0:
            # Sort prelim candidates by prob desc and take top_k
            probs_for_prelim = candidate_probs[prelim_indices]
            order = np.argsort(probs_for_prelim)[::-1]
            candidate_indices = prelim_indices[order[:top_k]]
        else:
            # Fallback: plain top-k from all valid
            sorted_indices = np.argsort(valid_probs)[::-1]
            num_candidates = min(top_k, len(sorted_indices))
            candidate_indices = valid_idx_map[sorted_indices[:num_candidates]]

        best_score = -1.0
        best_candidate_idx = 0

        # CRITICAL: Convert to list [int] to avoid numpy scalar indexing issues
        for candidate_idx in candidate_indices.tolist():
            candidate_idx = int(candidate_idx)
            expected_pos = [candidate_idx + i * meter_value for i in range(5)]
            valid_pos = [p for p in expected_pos if p < len(downbeat_data)]

            # Need at least 2 expected downbeats for reliable validation
            if len(valid_pos) < 3:
                print("[Patter Validation] not enough data for reliable validation")
                # fall back to single point score
                madmom_prob = downbeat_data[candidate_idx, 1]
                rms_score = downbeat_data[candidate_idx, 2]

                # Define weights for combining scores
                w_madmom = 0.9
                w_rms = 0.1

                consistency_score = (madmom_prob * w_madmom) + (rms_score * w_rms)
            else:
                # pattern validation
                pattern_data = downbeat_data[valid_pos]
                madmom_probs = pattern_data[:, 1]
                rms_scores = pattern_data[:, 2]

                mean_madmom_prob = np.mean(madmom_probs)
                std_madmom_prob = np.std(madmom_probs)

                mean_rms_score = np.mean(rms_scores)

                # Madmom consistency: high mean, low std
                madmom_consistency = mean_madmom_prob * (1.0 - min(std_madmom_prob, 0.5))

                # Define weights for combining scores
                w_madmom = 0.7
                w_rms = 0.3

                # Combined consistency score
                consistency_score = (madmom_consistency * w_madmom) + (mean_rms_score * w_rms)
            
            # Apply position penalty
            penalty_strength = 0.2
            position_penalty = 1.0 - (float(candidate_idx) / float(search_window)) * penalty_strength
            total_score = float(consistency_score * position_penalty)

            if not np.isnan(total_score) and total_score > best_score:
                best_score = float(total_score)
                best_candidate_idx = int(candidate_idx)

        return best_candidate_idx, best_score

    def _create_error_result(
        self,
        error_message: str,
        meter: List[int],
        midi_path: str = "",
        warnings: Optional[List[str]] = None,
    ) -> Dict:
        """Create a standardized error result."""
        meter_value = meter[0] if meter and len(meter) > 0 else 4
        
        return {
            'beats': np.array([], dtype=np.float32),
            'downbeats': np.array([], dtype=np.float32).reshape(0, 2),
            'first_downbeat_time': None,
            'anacrusis_beats': 0,
            'effective_anacrusis_beats': 0,
            'downbeat_probabilities': np.array([], dtype=np.float32).reshape(0, 2),
            'analysis_window': 0.0,
            'meter': int(meter_value),
            'total_beats_detected': 0,
            'midi_file': midi_path,
            'strategy': self.detection_mode,
            'warnings': warnings or [],
            'errors': [error_message],
        }

    def validate(self, result: Dict, expected_meter: int) -> Dict:
        """
        Validate the anacrusis detection result and provide quality metrics.
        
        Args:
            result: Result from detect_from_audio() or detect_from_midi()
            expected_meter: Expected meter (e.g., 3 for 3/4)
            
        Returns:
            Validation report with quality metrics
        """
        validation = {
            'valid': True,
            'warnings': [],
            'metrics': {},
            'quality_score': 0.0
        }
        
        # Check if anacrusis is reasonable
        anacrusis_beats = result.get('anacrusis_beats', 0)
        if anacrusis_beats >= expected_meter:
            validation['warnings'].append(f"Anacrusis ({anacrusis_beats}) >= meter ({expected_meter})")
        
        # Check beat detection quality
        total_beats = result.get('total_beats_detected', 0)
        analysis_window = result.get('analysis_window', 0)
        
        if total_beats > 0 and analysis_window > 0:
            avg_beat_interval = analysis_window / total_beats
            validation['metrics']['avg_beat_interval'] = avg_beat_interval
            validation['metrics']['estimated_tempo'] = 60.0 / avg_beat_interval
            
            # Quality score based on beat regularity and downbeat confidence
            downbeat_probs = result.get('downbeat_probabilities', np.array([]))
            if len(downbeat_probs) > 0:
                max_prob = np.nanmax(downbeat_probs[:, 1])
                avg_prob = np.nanmean(downbeat_probs[:, 1])
                validation['metrics']['max_downbeat_prob'] = float(max_prob)
                validation['metrics']['avg_downbeat_prob'] = float(avg_prob)
                validation['quality_score'] = (max_prob + avg_prob) / 2.0
        
        if validation['quality_score'] < 0.1:
            validation['warnings'].append("Low confidence in downbeat detection")
            validation['valid'] = False
        
        return validation
