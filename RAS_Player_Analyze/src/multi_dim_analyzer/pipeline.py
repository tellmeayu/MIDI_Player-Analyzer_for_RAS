"""
Pipeline orchestration for 4D rhythm analysis.

Coordinates the four analyzers (Beat Density, Predictability, Beat Salience,
Rhythmic Uniformity) over a PrettyMIDI input and returns structured results.
"""

from dataclasses import dataclass
from typing import Optional, Dict

import pretty_midi

from .beat_density import BeatDensityAnalyzer
from .predictability import PredictabilityAnalyzer
from .beat_salience import BeatSalienceAnalyzer
from .rhythmic_uniformity import RhythmicUniformityAnalyzer
from .config import AnalyzerConfig
from .plotting import plot_radar_chart


@dataclass
class AnalysisResult:
    """Output for 4-dimensional analysis."""
    beat_density: Optional[float]
    predictability: Optional[float]
    beat_salience: Optional[float]
    rhythmic_uniformity: Optional[float]
    error_messages: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "beat_density": self.beat_density,
            "predictability": self.predictability,
            "beat_salience": self.beat_salience,
            "rhythmic_uniformity": self.rhythmic_uniformity,
        }


class MultiDimensionalAnalyzer:
    """Orchestrates analysis across all four dimensions."""

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.config = config or AnalyzerConfig()
        self.bd_analyzer = BeatDensityAnalyzer(self.config.beat_density)
        self.ru_analyzer = RhythmicUniformityAnalyzer(self.config.rhythmic_uniformity)
        self.bs_analyzer = BeatSalienceAnalyzer(self.config.beat_salience)
        self.pred_analyzer = PredictabilityAnalyzer(self.config.predictability)

    def analyze(self, midi_path: str) -> AnalysisResult:
        """
        Full analysis pipeline for a single MIDI file.

        Args:
            midi_path: Path to a MIDI file.
        """
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            return AnalysisResult(
                beat_density=None,
                predictability=None,
                beat_salience=None,
                rhythmic_uniformity=None,
                error_messages={"midi_load": str(e)},
            )

        errors: Dict[str, str] = {}

        # Dimension I: Beat Density
        try:
            bd = self.bd_analyzer.analyze(pm)
        except Exception as e:
            bd = None
            errors["beat_density"] = str(e)

        # Dimension IV: Rhythmic Uniformity
        try:
            ru = self.ru_analyzer.analyze(pm)
        except Exception as e:
            ru = None
            errors["rhythmic_uniformity"] = str(e)

        # Dimension II: Predictability
        try:
            pred = self.pred_analyzer.analyze(pm)
        except Exception as e:
            pred = None
            errors["predictability"] = str(e)

        # Dimension III: Beat Salience (MIDI-only)
        try:
            bs = self.bs_analyzer.analyze(pm)
        except Exception as e:
            bs = None
            errors["beat_salience"] = str(e)

        return AnalysisResult(
            beat_density=bd,
            predictability=pred,
            beat_salience=bs,
            rhythmic_uniformity=ru,
            error_messages=errors or None,
        )

    def plot(
        self,
        result: AnalysisResult,
        piece_name: Optional[str] = None,
        title: str = "Rhythm Analysis"
    ) -> None:
        """
        Plot a radar chart for the analysis result.

        Args:
            result: AnalysisResult from analyze() method.
            piece_name: Optional name for the piece (for legend/labeling).
                      If None, defaults to "Piece".
            title: Title for the plot.
        """
        if piece_name is None:
            piece_name = "Piece"
        plot_radar_chart(result, piece_name, title)


