"""
Beat Density Analyzer (Dimension I).

Measures the "busyness" or textural crowdedness of music relative to its 
underlying pulse. Computed as normalized Notes-Per-Beat (NPB) score [0, 1].

Theory:
  - Partition MIDI into beat intervals
  - Count note onsets per beat
  - Trim extreme outliers (intro/outro noise)
  - Use median as robust NPB estimate
  - Apply sigmoid normalization with cap at 8 NPB
  
Result: Score in [0, 1]
"""

from typing import Optional, Tuple
import numpy as np
import pretty_midi

from core.precision_timer import PrecisionTimer
from .config import BeatDensityConfig
from .utils.midi_processor import MIDIProcessor
from .utils.beat_grid import DeterministicBeatGrid


class BeatDensityAnalyzer:
    """Compute Beat Density (Dimension I) from MIDI."""
    
    def __init__(self, config: BeatDensityConfig):
        """
        Initialize analyzer with configuration.
        
        Args:
            config: BeatDensityConfig with parameters for analysis.
        """
        self.config = config
    
    def analyze(self, pm: pretty_midi.PrettyMIDI) -> Optional[float]:
        """
        Compute normalized beat density score [0, 1].
        
        Args:
            pm: PrettyMIDI object with note events and metadata.
        
        Returns:
            Float in [0, 1] representing beat density, or None if insufficient data.
        """
        # 1. Get musical beat grid
        beats = DeterministicBeatGrid.generate_musical_beats(pm)
        if len(beats) < self.config.min_beats_required + 1:
            return None
        
        # 2. Extract onsets
        onsets = MIDIProcessor.extract_onsets(pm, self.config.include_drums)
        if len(onsets) == 0:
            return 0.0
        
        # 3. Collapse simultaneous notes (chords) using a dynamic or fixed epsilon
        epsilon = (self._calculate_dynamic_epsilon(pm)
                   if self.config.use_dynamic_epsilon
                   else self.config.simul_epsilon_sec)
        onsets = MIDIProcessor.collapse_simultaneous(
            onsets, 
            epsilon
        )
        
        # 4. Count onsets per beat interval
        counts = self._count_onsets_per_beat(onsets, beats)
        if len(counts) < self.config.min_beats_required:
            return None
        
        # 5. Trim outliers
        counts_trimmed = self._trim_outliers(counts)
        
        # 6. Compute robust NPB (Notes-Per-Beat)
        npb_robust = np.median(counts_trimmed) if len(counts_trimmed) > 0 else 0.0
        
        # 7. Sigmoid normalization
        bd_score = self._sigmoid_normalize(npb_robust)
        
        return float(bd_score)
    
    def _calculate_dynamic_epsilon(self, pm: pretty_midi.PrettyMIDI) -> float:
        """
        Calculate a dynamic epsilon for collapsing simultaneous notes based on tempo.
        
        Args:
            pm: PrettyMIDI object.
        
        Returns:
            Epsilon value in seconds.
        """
        # Get musical tempo at start (compound meters -> dotted quarter beats)
        tempo_map = MIDIProcessor.get_tempo_map(pm)
        tempo = tempo_map[0][1] if tempo_map and tempo_map[0][1] > 0 else 120.0
        ts_map = MIDIProcessor.get_time_signature_map(pm)
        time_sig = DeterministicBeatGrid._get_active_time_signature(ts_map, 0.0)
        musical_tempo = PrecisionTimer.convert_midi_tempo_to_musical_tempo(tempo, time_sig)
        beat_interval = 60.0 / musical_tempo if musical_tempo > 0 else 0.5
        
        # Determine epsilon as a percentage of the beat interval
        effective_tempo = musical_tempo if musical_tempo > 0 else tempo
        
        if effective_tempo >= 180:
            epsilon_ratio = 0.05  # 5% for very fast music
        elif effective_tempo >= 120:
            epsilon_ratio = 0.03  # 3% for moderate to fast music
        else:
            epsilon_ratio = 0.02  # 2% for slow music
            
        dynamic_epsilon = beat_interval * epsilon_ratio
        
        # Clamp the epsilon to a reasonable range [0.01s, 0.05s]
        # Use the configured epsilon as the minimum floor
        min_epsilon = self.config.simul_epsilon_sec
        return max(min_epsilon, min(0.05, dynamic_epsilon))

    def analyze_with_diagnostics(
        self, 
        pm: pretty_midi.PrettyMIDI
    ) -> Tuple[Optional[float], dict]:
        """
        Analyze beat density with detailed diagnostics.
        
        Args:
            pm: PrettyMIDI object.
        
        Returns:
            Tuple of (score, diagnostics_dict).
            
        Diagnostics include:
            - raw_npb: Median NPB before normalization
            - mean_npb: Mean NPB (before trimming)
            - min_count, max_count: Range of per-beat counts
            - trim_range: (lower_percentile, upper_percentile)
            - total_onsets: Count of all note onsets
            - total_beats: Count of beat intervals analyzed
        """
        # Get musical beat grid
        beats = DeterministicBeatGrid.generate_musical_beats(pm)
        if len(beats) < self.config.min_beats_required + 1:
            return None, {'error': 'insufficient_beats'}
        
        # Extract onsets
        onsets = MIDIProcessor.extract_onsets(pm, self.config.include_drums)
        if len(onsets) == 0:
            return 0.0, {
                'total_onsets': 0,
                'raw_npb': 0.0,
                'mean_npb': 0.0,
                'min_count': 0,
                'max_count': 0,
                'note': 'No onsets found'
            }
        
        # Collapse simultaneous using a dynamic or fixed epsilon
        epsilon = (self._calculate_dynamic_epsilon(pm)
                   if self.config.use_dynamic_epsilon
                   else self.config.simul_epsilon_sec)
        onsets = MIDIProcessor.collapse_simultaneous(
            onsets,
            epsilon
        )
        
        # Count per beat
        counts = self._count_onsets_per_beat(onsets, beats)
        if len(counts) < self.config.min_beats_required:
            return None, {'error': 'insufficient_beats_with_onsets'}
        
        # Compute diagnostics before trimming
        mean_npb = np.mean(counts)
        min_count = np.min(counts)
        max_count = np.max(counts)
        
        # Trim outliers
        counts_trimmed = self._trim_outliers(counts)
        
        # Compute NPB and score
        npb_robust = np.median(counts_trimmed) if len(counts_trimmed) > 0 else 0.0
        bd_score = self._sigmoid_normalize(npb_robust)
        
        diagnostics = {
            'total_onsets': len(onsets),
            'total_beats': len(counts),
            'raw_npb': float(npb_robust),
            'min_count': int(min_count),
            'max_count': int(max_count),
            'dynamic_epsilon': float(epsilon)
        }
        
        return float(bd_score), diagnostics
    
    def _count_onsets_per_beat(
        self, 
        onsets: np.ndarray, 
        beats: np.ndarray
    ) -> np.ndarray:
        """
        Count note onsets in each beat interval [b_i, b_{i+1}).
        
        Args:
            onsets: Sorted array of onset times in seconds.
            beats: Array of beat times in seconds.
        
        Returns:
            Array of counts, one per beat interval.
        """
        
        # safety check
        if len(beats) < 2 or len(onsets) == 0:
            return np.zeros(max(len(beats)-1, 0), dtype=int)
        if not (np.all(np.diff(beats) >= 0) and np.all(np.diff(onsets) >= 0)):
            raise ValueError("beats and onsets must be sorted in ascending order.")

        idx = np.searchsorted(onsets, beats, side='left')
        counts = np.diff(idx)
        
        return counts.astype(int)
            
    def _trim_outliers(self, counts: np.ndarray) -> np.ndarray:
        """
        Remove extreme per-beat counts (outliers).
        
        Outlier removal handles intro/outro noise and unusual bars.
        
        Args:
            counts: Array of per-beat counts.
        
        Returns:
            Trimmed array of counts.
        """
        if len(counts) == 0:
            return counts
        
        if self.config.trim_method == 'percentile':
            lower_p, upper_p = self.config.trim_percentiles
            lower_bound = np.percentile(counts, lower_p)
            upper_bound = np.percentile(counts, upper_p)
            trimmed = counts[(counts >= lower_bound) & (counts <= upper_bound)]
        
        elif self.config.trim_method == 'iqr':
            q1 = np.percentile(counts, 25)
            q3 = np.percentile(counts, 75)
            iqr = q3 - q1
            lower_bound = q1 - self.config.iqr_k * iqr
            upper_bound = q3 + self.config.iqr_k * iqr
            trimmed = counts[(counts >= lower_bound) & (counts <= upper_bound)]
        
        else:
            trimmed = counts
        
        # Fallback to untrimmed if everything was removed
        if len(trimmed) == 0:
            # Silently return untrimmed counts if all were removed
            return counts
        
        return trimmed
    
    def _sigmoid_normalize(self, npb: float) -> float:
        """
        Normalize NPB to [0, 1] using sigmoid function.
        
        Maps raw Notes-Per-Beat to perceptually smooth [0, 1] range with:
        - Exact 0 at NPB=0
        - Exact 1 at NPB >= bd_cap_npb
        - Higher resolution around sigmoid_center (typical textures)
        
        Args:
            npb: Raw Notes-Per-Beat value (unbounded).
        
        Returns:
            Normalized score in [0, 1].
        """
        # Clip to valid range [0, bd_cap_npb]
        npb_clipped = np.clip(npb, 0.0, self.config.bd_cap_npb)
        
        # Sigmoid function: sigmoid(x) = 1 / (1 + exp(-alpha * (x - center)))
        # Centered at sigmoid_center with steepness alpha
        alpha = self.config.sigmoid_alpha
        center = self.config.sigmoid_center
        
        # Avoid overflow in exp
        exponent = -alpha * (npb_clipped - center)
        exponent = np.clip(exponent, -500, 500)
        
        # Sigmoid normalization
        bd_norm = 1.0 / (1.0 + np.exp(exponent))
        
        # Clamp to [0, 1] to handle numerical noise
        bd_norm = np.clip(bd_norm, 0.0, 1.0)
        
        return float(bd_norm)
    
    def _sigmoid_normalize_vectorized(self, npb_array: np.ndarray) -> np.ndarray:
        """
        Vectorized sigmoid normalization for array of NPB values.
        
        Args:
            npb_array: Array of raw NPB values.
        
        Returns:
            Array of normalized scores in [0, 1].
        """
        npb_clipped = np.clip(npb_array, 0.0, self.config.bd_cap_npb)
        alpha = self.config.sigmoid_alpha
        center = self.config.sigmoid_center
        
        exponent = -alpha * (npb_clipped - center)
        exponent = np.clip(exponent, -500, 500)
        
        bd_norm = 1.0 / (1.0 + np.exp(exponent))
        bd_norm = np.clip(bd_norm, 0.0, 1.0)
        
        return bd_norm
