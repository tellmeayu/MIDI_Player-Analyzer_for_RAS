"""Madmom-driven meter estimator (class-based) integrated with preprocessor.

This module provides a class `MeterEstimator` which estimates musical meter
from audio using madmom's downbeat tracking. It supports two entry points:
- estimate_from_audio: analyze an audio file directly
- estimate_from_midi: preprocess a MIDI (via `preprocessor`) to audio, then analyze
"""

from __future__ import annotations

# Standard libraries
import os
import shutil
from typing import Dict, List, Optional

# Third-party libraries
import numpy as np
import madmom

# Local modules
try:
    from ...analysis.preprocessor import prepare_wav_from_midi
except ImportError:
    from analysis.preprocessor import prepare_wav_from_midi


class MeterEstimator:
    """Estimate meter using madmom downbeat tracking.

    Methods expose clear entry points for analyzing either a prepared audio
    file or a MIDI path (which will be rendered to audio using the shared
    preprocessor pipeline).
    """

    def __init__(self, sample_rate: int = 22050) -> None:
        """Initialize the estimator.

        Args:
            sample_rate: Target sample rate for audio synthesis when analyzing
                from MIDI.
        """
        self.sample_rate = int(sample_rate)

    # ------------------------------- Public API ------------------------------

    def estimate_from_audio(
        self,
        audio_path: str,
        meters_to_try: Optional[List[List[int]]] = None,
        duration_limit_sec: Optional[float] = 20.0,
        ) -> Dict[str, object]:
        """Estimate meter directly from an audio file.
        
        Args:
            audio_path: Path to audio file (.wav).
            meters_to_try: List of beat-per-bar hypotheses, e.g., [[2], [3], [4]].
            duration_limit_sec: Minimum analysis duration; if the audio is
                longer, it may be truncated to this length for efficiency.
            
        Returns:
            Dict containing downbeat analysis results per meter, and summary:
            {
              'estimated_meter': str,
              'confidence': float,
              'raw_confidence': float,
              'confidence_scores': Dict[str, float],
              'all_results': Dict[str, Dict],
              'tempo': float  # Estimated tempo in BPM
            }
        """
        results = self._analyze_downbeats(
            audio_path=audio_path,
            meters_to_try=meters_to_try,
            duration_limit_sec=duration_limit_sec,
        )

        best_meter = self._get_best_meter_from_results(results)
        confidence_scores = {
            meter: result["confidence"] for meter, result in results.items()
        }
        raw_confidence = (
            confidence_scores.get(best_meter, 0.0) if best_meter != "N/A" else 0.0
        )
        enhanced_confidence = self._calculate_relative_confidence(confidence_scores)

        # Calculate tempo from best meter's beat tracking results
        tempo = 120.0  # default
        if best_meter != "N/A" and best_meter in results:
            best_result = results[best_meter]
            all_beats = best_result.get("all_beats")
            if all_beats is not None and isinstance(all_beats, np.ndarray):
                tempo = self._calculate_tempo_from_beats(all_beats)

        return {
            "estimated_meter": best_meter,
            "confidence": enhanced_confidence,
            "raw_confidence": raw_confidence,
            "confidence_scores": confidence_scores,
            "all_results": results,
            "tempo": tempo,
        }

    def estimate_from_midi(
        self,
        midi_path: str,
        sf2_path: str,
        music_type: str = "classical",
        meters_to_try: Optional[List[List[int]]] = None,
        ) -> Dict[str, object]:
        """Estimate meter from a MIDI path via shared preprocessing.

        The MIDI is rendered to a temporary WAV using the shared preprocessor
        pipeline. A short, strategy-dependent duration is used to accelerate
        analysis.
        
        Args:
            midi_path: Path to MIDI file.
            sf2_path: Path to SoundFont (.sf2) used for synthesis.
            music_type: One of {"classical", "popular"} to select duration.
            meters_to_try: List of beat-per-bar hypotheses, e.g., [[2], [3], [4]].
            
        Returns:
            Same result schema as `estimate_from_audio`, with additional fields:
            {
              'midi_file': str,
              'strategy': str,
              'tempo': float  # Estimated tempo in BPM
            }
        """
        duration_s, strategy = self._select_duration_by_music_type(music_type)

        # Render a temporary audio file for analysis
        wav_path = prepare_wav_from_midi(
            midi_path=midi_path,
            sf2_path=sf2_path,
            duration_s=duration_s,
            sr=self.sample_rate,
            include_drums_for_onset=False,
            suffix="madmom",
        )

        analysis = self.estimate_from_audio(
            audio_path=wav_path,
            meters_to_try=meters_to_try,
            duration_limit_sec=duration_s,
        )

        analysis.update({
            "midi_file": midi_path,
            "strategy": strategy,
        })
        return analysis

    # ------------------------------ Internals --------------------------------

    @staticmethod
    def _ensure_ffmpeg_in_path() -> None:
        """Ensure ffmpeg is accessible in PATH for madmom.
        
        This method checks if ffmpeg is in PATH, and if not, adds common
        brew installation paths to the environment PATH.
        """
        # Check if ffmpeg is already accessible
        if shutil.which('ffmpeg'):
            return
        
        # Common brew installation paths
        brew_paths = [
            '/usr/local/bin/ffmpeg',     # Intel Mac or older installations
            '/opt/homebrew/bin/ffmpeg',   # Apple Silicon (M1/M2/M3)
        ]
        
        # Find ffmpeg and add its directory to PATH
        for ffmpeg_path in brew_paths:
            if os.path.exists(ffmpeg_path):
                bin_dir = os.path.dirname(ffmpeg_path)
                current_path = os.environ.get('PATH', '')
                if bin_dir not in current_path:
                    os.environ['PATH'] = f"{bin_dir}:{current_path}"
                return

    def _analyze_downbeats(
        self,
        audio_path: str,
        meters_to_try: Optional[List[List[int]]],
        duration_limit_sec: Optional[float],
        ) -> Dict[str, Dict[str, object]]:
        """Run madmom RNN+DBN to analyze downbeats for multiple meters.

        Args:
            audio_path: Path to audio.
            meters_to_try: Candidate meters; defaults to [[2], [3], [4]].
            duration_limit_sec: Minimum duration to analyze; if audio is longer,
                it may be truncated to this duration.

        Returns:
            Dict keyed by meter string (e.g., "3/4"), each value contains:
            {
              'all_beats': np.ndarray(shape=(N, 2)),  # time, beat_number
              'downbeats': np.ndarray(shape=(M,)),    # downbeat times
              'confidence': float
            }
        """
        # Ensure ffmpeg is accessible before using madmom
        self._ensure_ffmpeg_in_path()
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if meters_to_try is None:
            meters_to_try = [[2], [3], [4]]

        signal = madmom.audio.signal.Signal(audio_path)
        total_duration = len(signal) / signal.sample_rate # type: ignore

        min_analysis_duration = 20.0
        effective_duration = (
            duration_limit_sec if duration_limit_sec is not None else total_duration
        )
        if (
            effective_duration < min_analysis_duration
            and total_duration >= min_analysis_duration
        ):
            effective_duration = min_analysis_duration

        if effective_duration < total_duration:
            num_samples = int(effective_duration * signal.sample_rate) # type: ignore
            processed_signal = signal[:num_samples]
        else:
            processed_signal = signal

        proc_rnn = madmom.features.downbeats.RNNDownBeatProcessor()
        activations = proc_rnn(processed_signal)

        fps = 100  # fixed by madmom processor
        downbeat_activations = activations[:, 1]

        analysis_results: Dict[str, Dict[str, object]] = {}
        for meter in meters_to_try:
            meter_str = f"{meter[0]}/4"
            proc_dbn = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
                beats_per_bar=meter, fps=fps
            )
            all_beats = proc_dbn(activations)

            confidence = 0.0
            actual_downbeat_times = np.array([])
            if all_beats.size > 0:
                actual_downbeat_times = all_beats[all_beats[:, 1] == 1][:, 0]
                if actual_downbeat_times.size > 0:
                    downbeat_frames = (actual_downbeat_times * fps).astype(int)
                    downbeat_frames = downbeat_frames[
                        downbeat_frames < len(downbeat_activations)
                    ]
                    confidence = float(
                        np.mean(downbeat_activations[downbeat_frames])
                    )

            analysis_results[meter_str] = {
                "all_beats": all_beats,
                "downbeats": actual_downbeat_times,
                "confidence": float(confidence),
            }

        return analysis_results

    @staticmethod
    def _calculate_tempo_from_beats(all_beats: np.ndarray) -> float:
        """Calculate tempo from beat times.
        
        Args:
            all_beats: Array of shape (N, 2) where [:, 0] are beat times in seconds
                       and [:, 1] are beat numbers.
        
        Returns:
            Estimated tempo in BPM
        """
        if all_beats.size == 0 or len(all_beats) < 2:
            return 120.0  # default
        
        beat_times = all_beats[:, 0]
        intervals = np.diff(beat_times)
        # Filter out unrealistic intervals (too fast or too slow)
        # 0.1s = 600 BPM (too fast), 2.0s = 30 BPM (too slow)
        valid_intervals = intervals[(intervals > 0.1) & (intervals < 2.0)]
        if len(valid_intervals) == 0:
            return 120.0
        
        avg_interval = np.mean(valid_intervals)
        tempo = 60.0 / avg_interval
        return float(np.clip(tempo, 30.0, 300.0))  # Reasonable range

    @staticmethod
    def _get_best_meter_from_results(results: Dict[str, Dict[str, object]]) -> str:
        """Select the meter with the highest confidence."""
        if not results:
            return "N/A"
        return max(results.keys(), key=lambda m: results[m]["confidence"])  # type: ignore[index]

    @staticmethod
    def _calculate_relative_confidence(
        confidence_scores: Dict[str, float]
        ) -> float:
        """Compute enhanced confidence considering relative margins and spread.

        Args:
            confidence_scores: Map of meter string to confidence value.

        Returns:
            A float in [0.0, 1.0].
        """
        if not confidence_scores:
            return 0.0

        scores = list(confidence_scores.values())
        if all(score == 0.0 for score in scores):
            return 0.0

        if len(scores) == 1:
            return scores[0]

        scores_sorted = sorted(scores, reverse=True)
        best_score = scores_sorted[0]
        second_best = scores_sorted[1] if len(scores_sorted) > 1 else 0.0

        if best_score <= 0.0:
            return 0.0

        relative_margin = (
            (best_score - second_best) / best_score if second_best > 0 else 1.0
        )
        avg_score = float(np.mean(scores))
        score_std = float(np.std(scores)) if len(scores) > 1 else 0.0

        relative_confidence = 0.8 * best_score + 0.15 * relative_margin + 0.05 * (
            min(score_std / avg_score, 1.0) if avg_score > 0 else 0.0
        )
        return float(min(max(relative_confidence, 0.0), 1.0))

    @staticmethod
    def _select_duration_by_music_type(music_type: str) -> (float, str): # type: ignore
        """Return (duration_seconds, strategy_description) by music type."""
        mt = (music_type or "").strip().lower()
        if mt in {"popular", "pop", "modern"}:
            return 20.0, "Popular music: 20s from first note"
        return 30.0, "Classical music: 30s from first note"


__all__ = ["MeterEstimator"]


