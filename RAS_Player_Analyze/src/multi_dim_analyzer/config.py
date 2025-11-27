"""
Configuration module for Multi-Dimensional Rhythm Analyzer.

Provides dataclass configurations for all four dimensions with sensible defaults.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class BeatDensityConfig:
    """Configuration for Beat Density (Dimension I).
    
    Measures the "busyness" or textural crowdedness relative to underlying pulse.
    """
    use_dynamic_epsilon: bool = True
    """Whether to use tempo-based dynamic epsilon for collapsing simultaneous onsets."""

    simul_epsilon_sec: float = 0.015
    """Threshold (seconds) for collapsing simultaneous onsets."""
    
    include_drums: bool = False
    """Whether to include drum tracks in analysis."""
    
    beat_grid_source: Literal['deterministic', 'estimated'] = 'deterministic'
    """Source of beat grid: deterministic (MIDI metadata) or estimated (audio-based)."""
    
    min_beats_required: int = 4
    """Minimum number of beats required for valid analysis."""
    
    trim_method: Literal['percentile', 'iqr'] = 'percentile'
    """Method for outlier trimming: percentile or interquartile range."""
    
    trim_percentiles: tuple = (5, 95)
    """Percentile range for outlier trimming (min, max)."""
    
    iqr_k: float = 1.5
    """IQR multiplier for outlier detection (when trim_method='iqr')."""
    
    bd_cap_npb: float = 8.0
    """Upper cap for Notes-Per-Beat before sigmoid saturation."""
    
    sigmoid_alpha: float = 0.9
    """Sigmoid steepness parameter."""
    
    sigmoid_center: float = 4.0
    """Sigmoid center (inflection point) in NPB units."""


@dataclass
class RhythmicUniformityConfig:
    """Configuration for Rhythmic Uniformity (Dimension IV).
    
    Measures inverse normalized Pairwise Variability Index (nPVI).
    """
    collapse_epsilon_sec: float = 0.01
    """Threshold (seconds) for collapsing simultaneous onsets."""
    
    min_ioi_sec: float = 0.02
    """Minimum inter-onset interval (seconds) to include."""
    
    max_ioi_sec: Optional[float] = None
    """Maximum inter-onset interval (seconds) to include. None = no upper limit."""
    
    min_events_required: int = 4
    """Minimum number of events required for valid analysis."""
    
    k: float = 0.015
    """Decay constant for exponential mapping: uniformity = exp(-k * nPVI)."""
    
    beat_relative: bool = False
    """Whether to normalize IOIs relative to beat duration."""


@dataclass
class BeatSalienceConfig:
    """Configuration for Beat Salience (Dimension III).
    
    Measures local onset energy contrast around beat positions.
    """
    frame_hop_sec: float = 0.02
    """Time hop for frame-based analysis (seconds)."""
        
    delta_window_sec: Optional[float] = None
    """Window size around each beat for computing local salience (seconds).
    
    If None, dynamically computed as half-beat duration based on average tempo
    and time signature in the first 60 seconds (or actual duration if shorter).
    If set to a value, uses fixed window size (legacy behavior).
    """
    
    exclude_self_radius: float = 0.03
    """Radius around beat to exclude from off-beat comparison (seconds)."""
    
    min_beats_required: int = 4
    """Minimum number of beats required for valid analysis."""
    
    epsilon: float = 1e-6
    """Small constant to avoid division by zero."""
    
    max_ratio_clip: Optional[float] = 5.0
    """Maximum ratio clip for salience values. None = no clipping."""
    
    use_velocity_weighted_proxy: bool = False
    """Whether to use velocity-weighted onset proxy instead of binary."""
    
    max_duration_sec: Optional[float] = 90.0
    """Maximum duration (seconds) to analyze. If MIDI is longer, slices at nearest beat boundary.
    If None, analyzes entire piece. Default: 60.0 seconds."""
    
    normalization_method: str = "sigmoid"
    """Score normalization method: 'clip' (simple clipping), 'sigmoid' (sigmoid compression), 
    or 'log' (logarithmic compression). Default: 'sigmoid'."""
    
    min_score: float = 0.05
    """Minimum score threshold to avoid extreme values. Even if raw salience is 0, 
    the final score will be at least this value. Default: 0.05."""
    
    verbose: bool = False
    """Whether to print detailed debug information during analysis."""
    # MIDI-only mode; onset envelope derived from piano roll.


@dataclass
class PredictabilityConfig:
    """Configuration for Predictability (Dimension II).
    
    Measures inverse Toussaint Syncopation Index (metrical conformance).
    """
    resolution_mode: Literal['auto', 'fixed'] = 'auto'
    """Grid resolution mode: 'auto' (adaptive per time sig) or 'fixed' (global)."""
    
    min_bars_required: int = 6
    """Minimum number of bars required for valid analysis."""
    
    style: Literal['neutral', 'swing', 'folk'] = 'neutral'
    """Weight vector style: 'neutral' (classical), 'swing', or 'folk'."""
    
    fixed_resolution: Optional[int] = None
    """Grid size when resolution_mode='fixed'. If None, defaults to 16."""
    
    chord_handling: Literal['binary', 'stacked'] = 'binary'
    """How to handle simultaneous onsets: 'binary' (=1 event) or 'stacked' (count all)."""
    
    note_duration_threshold_sec: float = 0.05
    """Minimum note duration (seconds) to include. Filters out very short grace notes."""
    
    macro_beat_tolerance_sec: float = 0.1
    """Tolerance (seconds) around a beat to consider a note as 'on-beat' for macro analysis."""
    
    macro_weight: float = 0.8
    """The weight of the macro-level predictability score in the final combined score [0, 1].
    
    Macro-level (beat adherence) is the primary indicator of metrical predictability,
    while micro-level (syncopation) provides fine-grained rhythmic pattern analysis.
    Default 0.8 gives primary weight to macro-level analysis.
    """

    max_duration_sec: Optional[float] = 80.0
    """Maximum duration (seconds) to analyze. If MIDI is longer, slices at nearest bar boundary.
    If None, analyzes entire piece. Default: 90.0 seconds."""


@dataclass
class AnalyzerConfig:
    """Master configuration for all four dimensions."""
    
    beat_density: BeatDensityConfig = field(default_factory=BeatDensityConfig)
    """Configuration for Dimension I: Beat Density."""
    
    rhythmic_uniformity: RhythmicUniformityConfig = field(default_factory=RhythmicUniformityConfig)
    """Configuration for Dimension IV: Rhythmic Uniformity."""
    
    beat_salience: BeatSalienceConfig = field(default_factory=BeatSalienceConfig)
    """Configuration for Dimension III: Beat Salience."""
    
    predictability: PredictabilityConfig = field(default_factory=PredictabilityConfig)
    """Configuration for Dimension II: Predictability."""
        
    verbose: bool = False
    """Enable verbose logging and diagnostics."""
    
    def __repr__(self) -> str:
        """Pretty-print configuration."""
        lines = [
            "AnalyzerConfig(",
            f"  beat_density={self.beat_density},",
            f"  rhythmic_uniformity={self.rhythmic_uniformity},",
            f"  beat_salience={self.beat_salience},",
            f"  predictability={self.predictability},",
            f"  verbose={self.verbose}",
            ")",
        ]
        return "\n".join(lines)
