"""Perceptual tempo and beat feel analyzer using MIDI-derived features.

This module provides `TempogramAnalyzer` for determining if a MIDI file has a
duple or triple feel. It is designed as the fast, MIDI-only path for batch
analysis, avoiding audio synthesis by default.

Core Logic:
- Load and merge MIDI instruments via `preprocessor`.
- Truncate to a fixed analysis window from t=0 (no time shifting).
- Compute a chroma-flux onset envelope directly from MIDI notes.
- Analyze the onset envelope using a tempogram and autocorrelation to score
  duple vs. triple rhythmic patterns.
- An optional, on-demand `rescore_with_audio` method is provided for cases
  where higher accuracy is needed, which uses the preprocessor for synthesis.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import librosa
import numpy as np
import pretty_midi

try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

try:
    from ...analysis.preprocessor import (
        load_midi,
        merge_instruments_by_name,
        prepare_audio_from_midi,
        synthesize,
        trim_pm,
    )
except ImportError:
    from analysis.preprocessor import (
        load_midi,
        merge_instruments_by_name,
        prepare_audio_from_midi,
        synthesize,
        trim_pm,
    )


class TempogramAnalyzer:
    """Analyzes MIDI for duple/triple feel using tempogram and autocorrelation."""

    def __init__(
        self,
        max_analysis_duration: float = 69.6,
        tempogram_window_duration: float = 8.7,
        duple_sensitivity: float = 1.05,
        triple_sensitivity: float = 1.02,  # Added: Enhance triple sensitivity
        music_type: str = "classical",
        ):
        """Initializes the TempogramAnalyzer.

        Args:
            max_analysis_duration: Max duration in seconds to analyze from t=0.
            tempogram_window_duration: Length of the tempogram window in seconds.
            duple_sensitivity: Multiplier to boost duple meter detection.
            triple_sensitivity: Multiplier to boost triple meter detection.
            music_type: 'classical' or 'pop', for setting confidence thresholds.
        """
        self.max_analysis_duration = max_analysis_duration
        self.tempogram_window_duration = tempogram_window_duration
        self.duple_sensitivity = duple_sensitivity
        self.triple_sensitivity = triple_sensitivity  
        self.music_type = music_type.lower()
        if self.music_type not in ["classical", "pop"]:
            self.music_type = "classical"

        # Caches for a single analysis run
        self._tempogram: Optional[np.ndarray] = None
        self._tempi: Optional[np.ndarray] = None
        self._onset_env: Optional[np.ndarray] = None
        self._onset_sr: Optional[int] = None
        self._pm_window: Optional[pretty_midi.PrettyMIDI] = None
        self._prepared_midi_path: Optional[str] = None

    def _prepare_midi_window(self, midi_path: str) -> Optional[pretty_midi.PrettyMIDI]:
        """
        Loads, merges, and trims a MIDI file according to the analyzer's
        specific settings (t=0 start, no time shifting) and caches the result.
        This is the single source of truth for the analysis window.
        """
        # If the requested path is already prepared and cached, return it instantly.
        if self._prepared_midi_path == midi_path and self._pm_window is not None:
            return self._pm_window

        # Reset all caches if the file path is new.
        self._tempogram = None
        self._tempi = None
        self._onset_env = None
        self._onset_sr = None

        # 1. Preprocessing using the shared preprocessor
        try:
            pm = load_midi(midi_path)
            pm_merged = merge_instruments_by_name(pm)
        except Exception as e:
            # MIDI loading failed, return default result
            pass
            self._pm_window = None
            self._prepared_midi_path = None
            return None

        # 2. Determine analysis duration based on this module's specific logic
        full_duration = pm_merged.get_end_time()
        min_required = self.tempogram_window_duration * 3
        if full_duration < min_required:
            analysis_duration = full_duration
        else:
            analysis_duration = min(self.max_analysis_duration, full_duration)

        # 3. Create the analysis window and cache it
        self._pm_window = trim_pm(
            pm_merged,
            start_s=0.0,
            duration_s=analysis_duration,
            shift_to_zero=False,
        )
        self._prepared_midi_path = midi_path
        
        return self._pm_window

    def analyze(self, midi_path: str) -> Dict:
        """
        Performs the main MIDI-only analysis to determine beat feel.

        This is the default, fast-path method for batch processing.

        Args:
            midi_path: Path to the MIDI file.

        Returns:
            A dictionary containing the analysis results.
        """
        # 1. Prepare the analysis window (uses cache if available)
        pm_window = self._prepare_midi_window(midi_path)

        # If preparation failed or MIDI has no notes, return default.
        if pm_window is None or not any(i.notes for i in pm_window.instruments):
            return self._default_result()

        # 2. Feature Extraction
        self._calculate_and_cache_features(pm_window)
        if self._tempogram is None or self._tempi is None:
            return self._default_result()

        # 3. Tempo and Clarity Analysis
        global_tempogram = np.mean(self._tempogram, axis=1)
        valid_mask = (self._tempi >= 30) & (self._tempi <= 300)
        if not np.any(valid_mask):
            return self._default_result()

        strong_tempi, strong_energies, clarity = self._get_strong_tempi_and_clarity(
            global_tempogram[valid_mask], self._tempi[valid_mask]
        )
        if not strong_tempi:
            return self._default_result()

        # 4. Ensemble Scoring (Tempogram + Autocorrelation)
        ratio_results = self._score_from_tempogram(strong_tempi)
        autocorr_results = self._score_from_autocorrelation(strong_tempi[0])
        
        # 5. Combine and Classify
        final_scores = self._combine_scores(ratio_results, autocorr_results)
        final_results = self._classify_from_scores(
            final_scores["duple"], final_scores["triple"], clarity
        )

        return {
            **final_results,
            "clarity": clarity,
            "analytical_tempo": strong_tempi[0],
            "dominant_tempi": strong_tempi,
            "dominant_energies": strong_energies,
            "music_type": self.music_type,
            "audio_validation_performed": False,
        }

    def _calculate_and_cache_features(self, pm: pretty_midi.PrettyMIDI):
        """Computes and caches the onset envelope and tempogram for a MIDI object."""
        self._onset_sr = 200  # High temporal fidelity with low overhead
        self._onset_env = self._compute_midi_onset_envelope(pm, fs=self._onset_sr)

        win_length = int(self.tempogram_window_duration * self._onset_sr)
        
        # Use Fourier tempogram for robust tempo estimation
        self._tempogram = np.abs(librosa.feature.fourier_tempogram(
            onset_envelope=self._onset_env,
            sr=self._onset_sr,
            win_length=win_length,
        ))
        # Get corresponding tempo values for each bin
        self._tempi = librosa.fft_frequencies(sr=self._onset_sr, n_fft=win_length) * 60

    @staticmethod
    def _compute_midi_onset_envelope(pm: pretty_midi.PrettyMIDI, fs: int) -> np.ndarray:
        """Computes a chroma-flux onset envelope from a PrettyMIDI object."""
        if not pm.instruments or not any(i.notes for i in pm.instruments):
            return np.array([])
            
        chroma = pm.get_chroma(fs=fs)
        chroma_flux = np.sum(np.maximum(0, np.diff(chroma, axis=1)), axis=0)
        onset_env = np.pad(chroma_flux, (1, 0), "constant")

        if len(onset_env) > 3:
            window = np.array([0.25, 0.5, 0.25], dtype=np.float32)
            onset_env = np.convolve(onset_env, window, mode="same")

        max_val = np.max(onset_env)
        if max_val > 0:
            onset_env /= max_val
        return onset_env.astype(np.float32)

    @staticmethod
    def _get_strong_tempi_and_clarity(
        global_tempogram, tempi, top_n=3
        ) -> (list, list, float): # type: ignore
        """Finds dominant tempi and calculates a clarity score."""
        if global_tempogram.size == 0:
            return [], [], 0.0

        peak_indices, energies = [], []
        temp_tg = global_tempogram.copy()
        min_separation_bins = max(1, len(temp_tg) // 20)

        for _ in range(top_n):
            peak_idx = np.argmax(temp_tg)
            peak_energy = temp_tg[peak_idx]
            if peak_energy < 0.2 * np.max(global_tempogram):
                break
            peak_indices.append(peak_idx)
            energies.append(peak_energy)
            start = max(0, peak_idx - min_separation_bins)
            end = min(len(temp_tg), peak_idx + min_separation_bins + 1)
            temp_tg[start:end] = 0

        strong_tempi = [tempi[idx] for idx in peak_indices]
        
        # Clarity based on the prominence of the primary peak
        if len(energies) > 1:
            clarity = (energies[0] - energies[1]) / energies[0] if energies[0] > 0 else 0.0
        elif len(energies) == 1:
            clarity = 1.0
        else:
            clarity = 0.0

        return strong_tempi, energies, clarity

    def _score_from_tempogram(self, dominant_tempi: list) -> Dict[str, float]:
        """Scores duple/triple feel by sampling tempogram at harmonic ratios."""
        if self._tempogram is None or not dominant_tempi:
            return {"duple": 0.0, "triple": 0.0}

        global_tempogram = np.mean(self._tempogram, axis=1)
        duple_factors = [2, 4, 1, 1/2, 1/4, 1.5]  
        triple_factors = [3, 1, 1/3, 1/1.5, 9]    

        duple_energy = 0
        triple_energy = 0

        for tempo in dominant_tempi:
            for factor in duple_factors:
                target_bpm = tempo * factor
                idx = np.argmin(np.abs(self._tempi - target_bpm))
                if factor in [2, 1/2]:                              
                    weight_D = 1.1
                elif factor in [1.5, 1/4]:
                    weight_D = 1.05
                else:
                    weight_D = 1.0
                duple_energy += global_tempogram[idx] * weight_D

            for factor in triple_factors:
                target_bpm = tempo * factor
                idx = np.argmin(np.abs(self._tempi - target_bpm))
                if factor in [3, 1/3]:                               # Reset weights (13:17)
                    weight_T = 1.2  # Increase 3-beat weights (14:08)
                elif factor == 9:
                    weight_T = 1.05
                elif factor == 1/1.5:
                    weight_T = 0.8
                else:
                    weight_T = 1.0
                triple_energy += global_tempogram[idx] * weight_T
        
        return {"duple": duple_energy, "triple": triple_energy}

    def _score_from_autocorrelation(self, dominant_tempo: float) -> Dict[str, float]:
        """Scores duple/triple feel using autocorrelation of the onset envelope."""
        if self._onset_env is None or self._onset_sr is None:
            return {"duple": 0.0, "triple": 0.0}
            
        autocorr = librosa.autocorrelate(self._onset_env, max_size=int(self._onset_sr * 8))
        
        period_sec = 60.0 / dominant_tempo
        lag_window = int(self._onset_sr * 0.1)

        def get_max_autocorr_at_lag(lag_beats):
            center = int(lag_beats * period_sec * self._onset_sr)
            start = max(0, center - lag_window)
            end = min(len(autocorr), center + lag_window)
            return np.max(autocorr[start:end]) if start < end else 0.0

        triple_score = max(get_max_autocorr_at_lag(3), get_max_autocorr_at_lag(6) * 0.9)
        duple_score = max(get_max_autocorr_at_lag(2), get_max_autocorr_at_lag(4))
        
        return {"duple": duple_score, "triple": triple_score}

    @staticmethod
    def _combine_scores(
        ratio_results: Dict, autocorr_results: Dict
    ) -> Dict[str, float]:
        """Combines scores from two methods to reward agreement."""
        def normalize(scores):
            total = scores["duple"] + scores["triple"]
            return (scores["duple"] / total, scores["triple"] / total) if total > 1e-9 else (0.5, 0.5)

        duple_r, triple_r = normalize(ratio_results)
        duple_a, triple_a = normalize(autocorr_results)

        return {"duple": duple_r * duple_a, "triple": triple_r * triple_a}

    def _classify_from_scores(
        self, duple_score: float, triple_score: float, clarity: float = 1.0
    ) -> Dict:
        """Classifies meter based on final duple and triple scores.
        
        Args:
            duple_score: Normalized duple score from combined methods.
            triple_score: Normalized triple score from combined methods.
            clarity: Tempo clarity score (0.0-1.0), indicating how clear the
                dominant tempo is. Higher values mean more reliable analysis.
        
        Returns:
            Dict with classification, confidence (for the chosen classification),
            and raw scores.
        """
        final_duple = duple_score * self.duple_sensitivity
        final_triple = triple_score * self.triple_sensitivity
        total = final_duple + final_triple
        
        if total < 1e-9:
            return {
                "classification": "ambiguous",
                "confidence": 0.0,
                "duple_score": 0.0,
                "triple_score": 0.0,
            }
        
        duple_likelihood = final_duple / total
        triple_likelihood = final_triple / total
        w_likelihood = 0.95
        w_clarity = 0.05

        # Directly compare two likelihoods, select the higher one
        # If the difference is too small (within ambiguous range), classify as ambiguous
        # weighted confidence of likelihood and clarity
        if triple_likelihood > 0.5 and triple_likelihood > duple_likelihood:
            classification = "triple"
            confidence = triple_likelihood * w_likelihood + clarity * w_clarity

        elif duple_likelihood > 0.5 and duple_likelihood > triple_likelihood:
            classification = "duple"
            confidence = duple_likelihood * w_likelihood + clarity * w_clarity

        else:
            classification = "ambiguous"
            confidence = min(duple_likelihood, triple_likelihood) * w_likelihood + clarity * w_clarity

        return {
            "classification": classification,
            "confidence": float(min(max(confidence, 0.0), 1.0)),  # Ensure within [0, 1] range
            "duple_score": final_duple,
            "triple_score": final_triple,
        }

    def _default_result(self) -> Dict:
        """Returns a default result for ambiguous or un-analyzable files."""
        return {
            "classification": "ambiguous",
            "confidence": 0.0,
            "duple_score": 0.0,
            "triple_score": 0.0,
            "clarity": 0.0,
            "analytical_tempo": 0.0,
            "dominant_tempi": [],
            "dominant_energies": [],
            "music_type": self.music_type,
            "audio_validation_performed": False,
        }

    def rescore_with_audio(self, midi_path: str, sf2_path: str) -> Dict:
        """
        Re-scores the analysis using synthesized audio for higher accuracy.

        This method uses the same prepared MIDI data as the `analyze` method,
        ensuring consistency, and synthesizes it to re-run the analysis.

        Args:
            midi_path: Path to the MIDI file to analyze.
            sf2_path: Path to the SoundFont file for audio synthesis.

        Returns:
            A dictionary with the audio-based analysis results.
        """
        # 1. Prepare the exact same MIDI window (will be cached from `analyze`)
        pm_window = self._prepare_midi_window(midi_path)
        if pm_window is None or not any(i.notes for i in pm_window.instruments):
            return {**self._default_result(), "audio_validation_performed": True}

        # 2. Synthesize audio from the prepared MIDI window
        try:
            sr = 22050  # Standard sample rate for audio analysis
            audio = synthesize(pm_window, sf2_path, sr=sr)
        except Exception as e:
            # Audio synthesis failed, continue with MIDI-only analysis
            pass
            return {**self._default_result(), "audio_validation_performed": True}

        # 3. Extract features from audio
        onset_env_audio = librosa.onset.onset_strength(y=audio, sr=sr)
        
        # --- FIX: Use a more robust tempo estimation for audio ---
        # Estimate a single, reliable tempo from the audio onset envelope
        estimated_tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env_audio, sr=sr)
        if not estimated_tempo.size:
            return {**self._default_result(), "audio_validation_performed": True}
        
        strong_tempi = [estimated_tempo[0]]
        strong_energies = [] # Not directly comparable, can be omitted for audio
        clarity = 1.0 # Assume high clarity for a single dominant tempo

        win_length = int(self.tempogram_window_duration * (sr / 512)) # 512 is hop
        tempogram_audio = np.abs(librosa.feature.fourier_tempogram(
            onset_envelope=onset_env_audio, sr=sr, win_length=win_length
        ))
        tempi_audio = librosa.fft_frequencies(sr=sr, n_fft=win_length) * 60

        # 4. Ensemble Scoring on Audio Features
        # We need to re-cache audio-derived features for scoring methods
        self._tempogram = tempogram_audio
        self._tempi = tempi_audio
        self._onset_env = onset_env_audio
        self._onset_sr = sr / 512 # Effective sample rate of onset envelope

        ratio_results = self._score_from_tempogram(strong_tempi)
        autocorr_results = self._score_from_autocorrelation(strong_tempi[0])
        
        # 5. Combine and Classify
        final_scores = self._combine_scores(ratio_results, autocorr_results)
        final_results = self._classify_from_scores(
            final_scores["duple"], final_scores["triple"], clarity
        )

        return {
            **final_results,
            "clarity": clarity,
            "analytical_tempo": strong_tempi[0],
            "dominant_tempi": strong_tempi,
            "dominant_energies": strong_energies,
            "music_type": self.music_type,
            "audio_validation_performed": True,
        }

    def plot_tempogram(self, midi_path: str, output_path: Optional[str] = None):
        """
        Generates and displays or saves a visualization of the tempogram.

        This method will run the analysis if it hasn't been run yet for the
        specified MIDI file.

        Args:
            midi_path: Path to the MIDI file to analyze.
            output_path: Optional path to save the plot image. If None,
                         displays the plot interactively.
        """
        if not _MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for plotting. Please install it.")

        # 1. Run analysis to ensure features are computed and cached
        results = self.analyze(midi_path)
        
        # 2. Check if analysis was successful
        if self._tempogram is None or self._tempi is None or self._onset_sr is None:
            return

        # 3. Get the duration from the prepared midi window for correct x-axis limit
        if self._pm_window is None:
            return
        analysis_duration = self._pm_window.get_end_time()

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 4. Display the tempogram using librosa.display.specshow
        img = librosa.display.specshow(
            self._tempogram,
            sr=self._onset_sr,
            hop_length=1,  # Matches our feature extraction hop length
            x_axis='time',
            y_axis='tempo',
            ax=ax
        )
        fig.colorbar(img, ax=ax, label='Energy')

        # 5. Overlay dominant tempi found during analysis
        dominant_tempi = results.get('dominant_tempi', [])
        if dominant_tempi:
            ax.hlines(
                dominant_tempi,
                0,
                analysis_duration,
                colors='lime',
                linestyles='--',
                label='Dominant Tempi'
            )
            ax.legend(loc='upper right')

        # 6. Add labels and title
        ax.set_title(f"Tempogram - {os.path.basename(midi_path)}")
        ax.set_ylabel("Tempo (BPM)")
        
        # 7. Save or show the plot
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            # Plot saved successfully
            pass
        else:
            plt.show()
        
        plt.close(fig)
