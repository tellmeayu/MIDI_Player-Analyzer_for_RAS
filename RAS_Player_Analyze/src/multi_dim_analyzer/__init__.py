"""
Multi-Dimensional Rhythm Analyzer for MIDI files.

A comprehensive rhythm analysis system that evaluates music across four distinct dimensions:
1. Beat Density - Textural busyness (notes per beat)
2. Predictability - Metrical conformance (inverse syncopation)
3. Beat Salience - Perceptual prominence of beat positions
4. Rhythmic Uniformity - Inter-onset interval consistency (inverse nPVI)

Each dimension is normalized to [0, 1] for intuitive visualization (e.g., radar charts).
"""

from .config import (
    AnalyzerConfig,
    BeatDensityConfig,
    RhythmicUniformityConfig,
    BeatSalienceConfig,
    PredictabilityConfig,
)

__version__ = "0.1.0"
__author__ = "Rhythm Analysis Team"

__all__ = [
    "AnalyzerConfig",
    "BeatDensityConfig",
    "RhythmicUniformityConfig",
    "BeatSalienceConfig",
    "PredictabilityConfig",
]
