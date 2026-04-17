"""
Rhythmic Uniformity Analyzer (Dimension IV).

Measures the uniformity of inter-onset intervals (rhythmic consistency).
Computed as inverse normalized Pairwise Variability Index (nPVI) [0, 1].

Theory:
  - Extract all note onsets
  - Compute inter-onset intervals (IOIs)
  - Filter IOIs by duration range
  - Calculate normalized Pairwise Variability Index (nPVI)
  - Apply exponential decay mapping to get uniformity score [0, 1]
  
Result: Score in [0, 1] where:
  - 0.0: Highly syncopated/variable rhythm (high nPVI)
  - 0.5: Mixed rhythm character (moderate nPVI)
  - 1.0: Perfectly uniform rhythm (nPVI ≈ 0)

Note on nPVI calculation:
  nPVI is a measure of local inter-onset interval variability, defined as:
  nPVI = (1/(N-1)) * Σ |IOI_k - IOI_{k-1}| / ((IOI_k + IOI_{k-1})/2) * 100
  
  Where IOI_k are inter-onset intervals and N is the number of IOIs.
  The formula calculates the average of the absolute difference between successive
  IOIs, normalized by their mean.
  nPVI values typically range from 0 (uniform) to ~200.
"""

from typing import Optional, Tuple
import numpy as np
import pretty_midi

from .config import RhythmicUniformityConfig
from .utils.midi_processor import MIDIProcessor


class RhythmicUniformityAnalyzer:
    """Compute Rhythmic Uniformity (Dimension IV) from MIDI."""
    
    def __init__(self, config: RhythmicUniformityConfig):
        """
        Initialize analyzer with configuration.
        
        Args:
            config: RhythmicUniformityConfig with parameters for analysis.
        """
        self.config = config
    
    def analyze(self, pm: pretty_midi.PrettyMIDI) -> Optional[float]:
        """
        Compute normalized rhythmic uniformity score [0, 1].
        
        Args:
            pm: PrettyMIDI object with note events.
        
        Returns:
            Float in [0, 1] representing rhythmic uniformity, or None if insufficient data.
        """
        # 1. Extract onsets
        onsets = MIDIProcessor.extract_onsets(pm)
        if len(onsets) < self.config.min_events_required:
            return None
        
        # 2. Collapse simultaneous notes (chords)
        onsets = MIDIProcessor.collapse_simultaneous(
            onsets,
            self.config.collapse_epsilon_sec
        )
        
        if len(onsets) < self.config.min_events_required:
            return None
        
        # 3. Compute inter-onset intervals
        iois = MIDIProcessor.compute_iois(onsets)
        
        # 4. Filter IOIs by range
        iois_filtered = MIDIProcessor.filter_iois(
            iois,
            self.config.min_ioi_sec,
            self.config.max_ioi_sec
        )
        
        if len(iois_filtered) < 1:  # Need at least 2 IOIs for nPVI
            return None
        
        # 5. Compute nPVI
        npvi = self._compute_npvi(iois_filtered)
        
        # 6. Map nPVI to uniformity score [0, 1]
        uniformity_score = self._npvi_to_uniformity(npvi)
        
        return float(uniformity_score)
    
    def analyze_with_diagnostics(
        self,
        pm: pretty_midi.PrettyMIDI
    ) -> Tuple[Optional[float], dict]:
        """
        Analyze rhythmic uniformity with detailed diagnostics.
        
        Args:
            pm: PrettyMIDI object.
        
        Returns:
            Tuple of (score, diagnostics_dict).
            
        Diagnostics include:
            - npvi_raw: Raw nPVI value (0-100 scale)
            - mean_ioi: Mean inter-onset interval (seconds)
            - std_ioi: Standard deviation of IOIs
            - min_ioi, max_ioi: Range of filtered IOIs
            - ioi_count: Number of IOI pairs used
            - cv_ioi: Coefficient of variation
        """
        # Extract onsets
        onsets = MIDIProcessor.extract_onsets(pm, include_drums=True)
        if len(onsets) < self.config.min_events_required:
            return None, {'error': 'insufficient_events'}
        
        # Collapse simultaneous
        onsets = MIDIProcessor.collapse_simultaneous(
            onsets,
            self.config.collapse_epsilon_sec
        )
        
        if len(onsets) < self.config.min_events_required:
            return None, {'error': 'insufficient_events_after_collapse'}
        
        # Compute IOIs
        iois = MIDIProcessor.compute_iois(onsets)
        
        # Filter IOIs
        iois_filtered = MIDIProcessor.filter_iois(
            iois,
            self.config.min_ioi_sec,
            self.config.max_ioi_sec
        )
        
        if len(iois_filtered) < 1:
            return None, {'error': 'insufficient_iois'}
        
        # Compute diagnostics before mapping
        npvi_raw = self._compute_npvi(iois_filtered)
        mean_ioi = float(np.mean(iois_filtered))
        std_ioi = float(np.std(iois_filtered))
        min_ioi = float(np.min(iois_filtered))
        max_ioi = float(np.max(iois_filtered))
        cv_ioi = std_ioi / mean_ioi if mean_ioi > 0 else 0.0
        
        uniformity_score = self._npvi_to_uniformity(npvi_raw)
        
        diagnostics = {
            'npvi_raw': float(npvi_raw),
            'mean_ioi': mean_ioi,
            'std_ioi': std_ioi,
            'min_ioi': min_ioi,
            'max_ioi': max_ioi,
            'ioi_count': len(iois_filtered),
            'cv_ioi': cv_ioi,
            'k_decay': self.config.k,
            'total_onsets': len(onsets),
        }
        
        return float(uniformity_score), diagnostics
    
    def _compute_npvi(self, iois: np.ndarray) -> float:
        """
        Compute normalized Pairwise Variability Index (nPVI).
        
        nPVI measures the normalized variability between consecutive IOIs.
        Higher values indicate more syncopation/variability.
        
        Args:
            iois: Array of inter-onset intervals (seconds).
        
        Returns:
            nPVI value on approximate 0-100 scale (0=uniform, 100=highly variable).
        """
        if len(iois) < 2:
            return 0.0
        
        # Compute pairwise differences normalized by their mean
        # nPVI = 100 * (1/(n-1)) * sum(|d_k - d_{k-1}| / ((d_k + d_{k-1})/2))
        diffs = np.abs(np.diff(iois))
        sums = iois[:-1] + iois[1:]
        
        # Avoid division by zero
        valid_indices = sums > 1e-10
        
        if not np.any(valid_indices):
            return 0.0
            
        # The original formula divides by the mean of the pair, so we multiply the sum by 2
        # and then divide by the number of pairs.
        # This is equivalent to mean(|a-b|/((a+b)/2))
        npvi = np.mean(diffs[valid_indices] / (sums[valid_indices] / 2.0)) * 100.0
        
        return float(npvi)
    
    def _npvi_to_uniformity(self, npvi: float) -> float:
        """
        Map nPVI to uniformity score [0, 1] using piecewise decay mapping.
        
        - nPVI [0, 20]: High uniformity, slow decay.
        - nPVI [20, 55]: Main discrimination range, rapid decay.
        - nPVI > 55: Low uniformity, saturated.
        
        Args:
            npvi: Raw nPVI value (0-100 scale).
        
        Returns:
            Uniformity score in [0, 1] where 1=perfectly uniform.
        """
        # Clamp to a practical maximum, e.g., 150, as nPVI can exceed 100
        npvi_clipped = np.clip(npvi, 0.0, 150.0)
        
        # 1. Highly Uniform zone: nPVI in [0, 20] -> Score in [1.0, 0.7]
        if npvi_clipped <= 20.0:
          return 1.0 - (npvi_clipped / 20.0) * 0.3
          
        # 2. Main discrimination zone: nPVI in (20, 55] -> Score in (0.7, 0.2]
        elif npvi_clipped <= 55.0:
          progress = (npvi_clipped - 20.0) / (55.0 - 20.0)  # normalize to [0,1]

          # map the normalized progress to the score range [0.7,0.2] using a simpler power function instead of exponential
          # y = start - (start - end) * p^alpha, where alpha < 1
          alpha = 0.6
          curved_progress = progress ** alpha

          start_score = 0.7
          end_score = 0.2
          
          return start_score - curved_progress * (start_score - end_score)

        # 3. saturation zone (highly variable): nPVI > 55 -> score in [0.2, 0,0]
        else:
          k = 0.15 # adjustable param for exponential decay
          return 0.3 * np.exp(-k * (npvi_clipped - 55))
    
    def _npvi_to_uniformity_vectorized(self, npvi_array: np.ndarray) -> np.ndarray:
        """
        Vectorized nPVI to uniformity mapping.
        
        Args:
            npvi_array: Array of nPVI values.
        
        Returns:
            Array of uniformity scores in [0, 1].
        """
        npvi_clipped = np.clip(npvi_array, 0.0, 100.0)
        uniformity = np.exp(-self.config.k * npvi_clipped)
        uniformity = np.clip(uniformity, 0.0, 1.0)
        
        return uniformity
