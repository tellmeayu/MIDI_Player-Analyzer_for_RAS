# beat_tracker.py

import os
import numpy as np
import librosa
from typing import List, Tuple, Optional
from collections import defaultdict
from .preprocessor import (
    load_midi,
    merge_instruments_by_name,
    first_note_time,
    prepare_audio_from_midi,
)

# Add a conditional import for visualization to keep it optional
try:
    import matplotlib.pyplot as plt
    import librosa.display
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

# Import for saving audio file
try:
    import soundfile as sf
    _SOUNDFILE_AVAILABLE = True
except ImportError:
    _SOUNDFILE_AVAILABLE = False

class MidiBeatTracker:
    """
    A highly modular MIDI beat tracker.

    This class encapsulates the entire process of extracting precise beat times from a MIDI file.
    It follows the best practice of "symbolic-to-audio":
    1. Load and parse the MIDI file using pretty_midi.
    2. Synthesize the MIDI into an in-memory audio waveform using pretty_midi.
    3. Perform beat tracking analysis on the audio waveform using librosa.

    The main output is a list of beat time points (in seconds) and the estimated tempo.
    """

    def __init__(self, midi_path: str, soundfont_path: str):
        """
        Initialize the beat tracker.

        Args:
            midi_path (str): Path to the input MIDI file.
            soundfont_path (str): Path to the SoundFont (.sf2) file for audio synthesis.
        
        Raises:
            FileNotFoundError: If the MIDI or SoundFont file does not exist.
        """
        if not os.path.exists(midi_path):
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")
        if not os.path.exists(soundfont_path):
            raise FileNotFoundError(f"SoundFont file not found: {soundfont_path}")

        self.midi_path = midi_path
        self.soundfont_path = soundfont_path
        
        # Analysis caches
        self._beat_times: Optional[np.ndarray] = None
        self._estimated_tempo: Optional[float] = None
        self._beat_regularity: Optional[dict] = None
        self._beat_frames: Optional[np.ndarray] = None

    def _get_audio_data(self, sample_rate: int = 22050) -> Tuple[np.ndarray, int]:
        """Prepare audio for the whole piece (from first note to end)."""
        pm = merge_instruments_by_name(load_midi(self.midi_path))
        t0 = first_note_time(pm)
        total = pm.get_end_time()
        duration = max(0.0, total - t0)
        if duration <= 0:
            return np.array([], dtype=np.float32), sample_rate
        audio, sr = prepare_audio_from_midi(
            self.midi_path, self.soundfont_path, duration_s=duration, sr=sample_rate
        )
        return audio, sr

    def _analyze_beats_from_audio(self) -> Tuple[np.ndarray, float]:
        """
        Core analysis step: Track beats from the synthesized audio.

        Returns: Tuple[np.ndarray, float]: Beat times and estimated tempo.
        """
        audio_data, sr = self._get_audio_data()
        
        # Use librosa for beat tracking and then convert beat frame indices to seconds
        tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # Cache both beat_frames and beat_times
        self._beat_frames = beat_frames
        self._beat_times = beat_times
        return beat_times, float(tempo)

    def get_beat_times(self, force_recompute: bool = False) -> np.ndarray:
        """
        Retrieve all beat times (in seconds).
        The result will be cached, and repeated calls to this method will return the cached result,
        unless force_recompute is specified.

        Args:
            force_recompute (bool): If True, forces recomputation, ignoring the cache.

        Returns:
            np.ndarray: A NumPy array containing all beat times (in seconds).
        """
        if self._beat_times is None or force_recompute:
            self._beat_times, self._estimated_tempo = self._analyze_beats_from_audio()
        
        # Ensure the return value is not None
        return self._beat_times if self._beat_times is not None else np.array([])

    def get_estimated_tempo(self, force_recompute: bool = False, max_duration: Optional[float] = None) -> float:
        """
        Retrieve the estimated global tempo (BPM).

        Args:
            force_recompute (bool): If True, forces recomputation.
            max_duration (Optional[float]): If specified, only analyze the first max_duration seconds.

        Returns:
            float: The estimated BPM tempo.
        """
        if self._estimated_tempo is None or force_recompute:
            self.get_beat_times(force_recompute=True, max_duration=max_duration) # Ensure beats have been analyzed
            
        return self._estimated_tempo if self._estimated_tempo is not None else 0.0

    def get_beat_regularity(self, force_recompute: bool = False) -> dict:
        """
        Analyzes the regularity of beat times by calculating inter-beat intervals (IBIs)
        and their statistical properties.

        Args:
            force_recompute (bool): If True, forces re-computation.

        Returns:
            dict: A dictionary containing:
                - 'mean_ibi': The mean inter-beat interval in seconds.
                - 'std_ibi': The standard deviation of inter-beat intervals.
                - 'cv_ibi': The coefficient of variation of inter-beat intervals
                            (std_ibi / mean_ibi), indicating regularity (lower is more regular).
        """
        if self._beat_regularity is None or force_recompute:
            # This will trigger beat computation if not already done
            beat_times = self.get_beat_times(force_recompute)

            if len(beat_times) < 2:
                self._beat_regularity = {'mean_ibi': 0.0, 'std_ibi': 0.0, 'cv_ibi': 0.0}
            else:
                # Calculate inter-beat intervals (IBI)
                ibis = np.diff(beat_times)
                mean_ibi = float(np.mean(ibis))
                std_ibi = float(np.std(ibis))
                cv_ibi = (std_ibi / mean_ibi) if mean_ibi != 0 else 0.0

                self._beat_regularity = {
                    'mean_ibi': mean_ibi,
                    'std_ibi': std_ibi,
                    'cv_ibi': cv_ibi
                }
        
        # Ensure the return is not None for type hinting
        return self._beat_regularity if self._beat_regularity is not None else {}

    def save_merged_midi(self, output_path: str):
        """
        Save the MIDI data with merged instruments to a new .mid file.

        This is useful for verifying the merged result in a DAW (such as Logic Pro).

        Args:
            output_path (str): Path to the output MIDI file.
        """
        # Merge using preprocessor and write
        pm = merge_instruments_by_name(load_midi(self.midi_path))
        try:
            pm.write(output_path)
        except Exception as e:
            raise RuntimeError(f"Error writing MIDI file to {output_path}: {e}")

    def save_audio(self, output_path: str, max_duration: Optional[float] = None):
        """
        Save the synthesized audio into a new .wav file.
        If output_path is not specified, automatically generates a filename in the script directory's 'outputs' subfolder.

        Args:
            output_path: WAV file output path
            max_duration: if specified, only synthesize the first max_duration seconds.
        """
        if not _SOUNDFILE_AVAILABLE:
            raise ImportError("soundfile library not be installed")
        
        if output_path is None:
            # Get script directory
            script_dir = os.path.dirname(__file__)
            output_dir = os.path.join(script_dir, 'outputs')
            os.makedirs(output_dir, exist_ok=True)

            # Generate filename: based on MIDI filename + audio type + truncation  duration (if any)
            midi_base = os.path.splitext(os.path.basename(self.midi_path))[0]
            duration_suffix = f"_{int(max_duration)}s" if max_duration else ""
            output_path = os.path.join(output_dir, f"{midi_base}_audio{duration_suffix}.wav")
        
        audio_data, sr = self._get_audio_data()
        
        try:
            sf.write(output_path, audio_data, sr)
        except Exception as e:
            raise RuntimeError(f"Error writing audio file to {output_path}: {e}")

    def generate_full_click_track(
        self,
        click_freq: float = 1000.0,
        click_strength: float = 0.8,
        sample_rate: int = 22050
    ) -> Tuple[np.ndarray, int]:
        """Generate full-length audio with click overlay.

        Synthesizes the entire MIDI file to audio and overlays click transients
        at detected beat times. This is used for the audio-based player.

        Args:
            click_freq: Frequency of click sound in Hz (default: 1000.0)
            click_strength: Amplitude of clicks relative to music (default: 0.8)
            sample_rate: Audio sample rate (default: 22050 Hz)

        Returns:
            Tuple of (mixed_audio, sample_rate) where mixed_audio is a numpy array
            containing the music with click overlay, normalized to prevent clipping.
        """
        # Get beat times and full audio
        beat_times = self.get_beat_times()
        audio, sr = self._get_audio_data(sample_rate=sample_rate)

        if audio is None or len(audio) == 0:
            return np.array([], dtype=np.float32), sr

        # Generate click track for all beats
        clicks = librosa.clicks(times=beat_times, sr=sr, click_freq=click_freq, length=len(audio))
        
        # Amplify click audio for better audibility
        click_amplification = 2.5  # Amplify clicks by 2.5x
        clicks_amplified = clicks * click_amplification
        
        # Apply additional click strength multiplier
        clicks_final = clicks_amplified * click_strength

        # Mix clicks into audio
        if audio.ndim == 1:
            mix = audio.astype(np.float32) + clicks_final.astype(np.float32)
        else:
            # Handle stereo (tile clicks to match audio channels)
            clicks_st = np.tile(clicks_final.astype(np.float32)[:, None], (1, audio.shape[1]))
            mix = audio.astype(np.float32) + clicks_st

        # Normalize to avoid clipping while preserving click audibility
        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak > 1.0:
            mix = mix / peak

        return mix, sr

    def generate_click_previews(
        self,
        click_freq: float = 1000.0,
        click_strength: float = 0.9,
        ) -> dict:
        """Generate three short preview segments with click overlays.

        Returns three segments mixed with clicks at detected beat times:
        - begin: first 10s from the start (first note)
        - middle: 6s around the middle of the piece
        - end: last 6s near the end

        Returns:
            dict with keys 'begin', 'middle', 'end', each containing:
            {
              'audio': np.ndarray,
              'sr': int,
              'start_time': float,
              'duration': float,
            }
        """
        # Ensure beat times are computed and cache audio once
        beat_times = self.get_beat_times()
        audio, sr = self._get_audio_data()
        if audio is None or len(audio) == 0:
            return {
                'begin': {'audio': np.array([], dtype=np.float32), 'sr': sr, 'start_time': 0.0, 'duration': 0.0},
                'middle': {'audio': np.array([], dtype=np.float32), 'sr': sr, 'start_time': 0.0, 'duration': 0.0},
                'end': {'audio': np.array([], dtype=np.float32), 'sr': sr, 'start_time': 0.0, 'duration': 0.0},
            }

        total_sec = len(audio) / sr

        def _segment(start_s: float, duration_s: float) -> Tuple[np.ndarray, float, float]:
            start_s = max(0.0, min(start_s, max(0.0, total_sec - 1e-6)))
            duration_s = max(0.0, min(duration_s, max(0.0, total_sec - start_s)))
            s0 = int(round(start_s * sr))
            s_len = int(round(duration_s * sr))
            s1 = min(len(audio), s0 + s_len)
            seg = audio[s0:s1]

            # Select beats within segment and make times relative to segment start
            mask = (beat_times >= start_s) & (beat_times < (start_s + duration_s))
            rel_times = (beat_times[mask] - start_s)

            # Generate click track
            clicks = librosa.clicks(times=rel_times, sr=sr, click_freq=click_freq, length=len(seg))
            
            # Amplify click audio for better audibility (same as full track)
            click_amplification = 2.5  # Amplify clicks by 2.5x
            clicks_amplified = clicks * click_amplification
            
            # Apply additional click strength multiplier
            clicks_final = clicks_amplified * click_strength

            # Mix clicks into audio
            if seg.ndim == 1:
                mix = seg.astype(np.float32) + clicks_final.astype(np.float32)
            else:
                clicks_st = np.tile(clicks_final.astype(np.float32)[:, None], (1, seg.shape[1]))
                mix = seg.astype(np.float32) + clicks_st

            # Normalize softly to avoid clipping
            peak = float(np.max(np.abs(mix))) if mix.size else 0.0
            if peak > 1.0:
                mix = mix / peak

            return mix, start_s, duration_s

        # Define segments
        begin_dur = min(10.0, total_sec)
        begin_audio, begin_start, begin_d = _segment(2.0, begin_dur)

        mid_center = total_sec / 2.0
        mid_start = max(0.0, min(mid_center - 3.0, max(0.0, total_sec - 6.0)))
        middle_audio, middle_start, middle_d = _segment(mid_start, min(6.0, total_sec - mid_start))

        end_start = max(0.0, total_sec - 6.0)
        end_audio, end_s, end_d = _segment(end_start, min(6.0, total_sec - end_start))

        return {
            'begin': {'audio': begin_audio, 'sr': sr, 'start_time': begin_start, 'duration': begin_d},
            'middle': {'audio': middle_audio, 'sr': sr, 'start_time': middle_start, 'duration': middle_d},
            'end': {'audio': end_audio, 'sr': sr, 'start_time': end_s, 'duration': end_d},
        }

