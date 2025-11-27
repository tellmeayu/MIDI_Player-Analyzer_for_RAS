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
  nPVI = (1/N) * Σ |IOI_i - IOI_{i-1}| / (IOI_i + IOI_{i-1}) * 100
  
  Where IOI_i are inter-onset intervals and N is the number of pairs.
  nPVI values typically range from 0 (uniform) to 100 (highly variable).
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
        
        # Compute pairwise differences normalized by sum
        differences = []
        for i in range(len(iois) - 1):
            ioi_curr = iois[i]
            ioi_next = iois[i + 1]
            
            # Avoid division by very small values
            denominator = ioi_curr + ioi_next
            if denominator > 1e-10:
                diff = abs(ioi_next - ioi_curr) / denominator
                differences.append(diff)
        
        if len(differences) == 0:
            return 0.0
        
        # nPVI is the mean relative difference, scaled to ~0-100 range
        npvi = np.mean(differences) * 100.0
        
        return float(npvi)
    
    def _npvi_to_uniformity(self, npvi: float) -> float:
        """
        Map nPVI to uniformity score [0, 1] using exponential decay.
        
        Higher nPVI → lower uniformity score.
        Mapping: uniformity = exp(-k * nPVI)
        
        Args:
            npvi: Raw nPVI value (0-100 scale).
        
        Returns:
            Uniformity score in [0, 1] where 1=perfectly uniform.
        """
        # Clamp to reasonable range
        npvi_clipped = np.clip(npvi, 0.0, 100.0)
        
        # Exponential decay mapping
        k = self.config.k
        uniformity = np.exp(-k * npvi_clipped)
        
        # Clamp to [0, 1] to handle numerical noise
        uniformity = np.clip(uniformity, 0.0, 1.0)
        
        return float(uniformity)
    
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
