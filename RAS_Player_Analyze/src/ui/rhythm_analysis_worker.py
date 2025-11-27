"""Qt Worker for multi-dimensional rhythm analysis.

This module provides a Qt-based worker for running 4D rhythm analysis
in a background thread.
"""

from PyQt5.QtCore import QObject, pyqtSignal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multi_dim_analyzer.pipeline import AnalysisResult


class RhythmAnalysisWorker(QObject):
    """
    Worker for running multi-dimensional rhythm analysis in a background thread.

    This worker uses the moveToThread pattern (QObject + QThread) for
    proper signal/slot communication.

    Signals:
        progress(int): Progress percentage (0-100)
        finished(AnalysisResult): Analysis results
        error(str): Error message
    """

    # Signals
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)  # AnalysisResult
    error = pyqtSignal(str)

    def __init__(self, midi_path: str):
        """
        Initialize the worker.

        Args:
            midi_path: Path to the MIDI file
        """
        super().__init__()
        self.midi_path = midi_path

    def run(self):
        """
        Run the multi-dimensional rhythm analysis.

        This method will be called when the worker is started.
        It runs in a separate thread and emits signals to report progress.
        """
        try:
            # Import here to avoid circular imports
            from multi_dim_analyzer.pipeline import MultiDimensionalAnalyzer

            # Report progress at start
            self.progress.emit(10)

            # Initialize analyzer
            analyzer = MultiDimensionalAnalyzer()
            self.progress.emit(30)

            # Run analysis for each dimension
            # Dimension I: Beat Density
            self.progress.emit(40)
            result = analyzer.analyze(self.midi_path)
            self.progress.emit(90)

            # Report completion
            self.progress.emit(100)
            self.finished.emit(result)

        except Exception as e:
            # Catch any unexpected exceptions
            import traceback
            error_msg = f"Error during rhythm analysis: {str(e)}"
            print(f"RhythmAnalysisWorker error: {error_msg}")
            traceback.print_exc()
            self.error.emit(error_msg)

