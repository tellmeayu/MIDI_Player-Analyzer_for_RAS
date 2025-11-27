"""Qt Worker for beat tracking analysis.

This module provides a Qt-based worker for running beat tracking analysis
in a background thread with cancellation support.
"""

from PyQt5.QtCore import QObject, pyqtSignal
import threading
from typing import Optional


class BeatTrackingWorker(QObject):
    """
    Worker for running beat tracking analysis in a background thread.

    This worker uses the moveToThread pattern (QObject + QThread) for
    better control over cancellation and cleanup.

    Signals:
        progress(int): Progress percentage (0-100)
        finished(dict): Analysis results dictionary
        error(str): Error message
        cancelled(): Emitted when analysis is cancelled
    """

    # Signals
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, midi_path: str, sf2_path: Optional[str] = None):
        """
        Initialize the worker.

        Args:
            midi_path: Path to the MIDI file
            sf2_path: Path to SoundFont file for synthesis
        """
        super().__init__()
        self.midi_path = midi_path
        self.sf2_path = sf2_path
        self._cancel_flag = threading.Event()

    def run(self):
        """
        Run the beat tracking analysis.

        This method will be called when the worker is started.
        It runs in a separate thread and emits signals to report progress.

        SAFETY: This analysis uses pm.fluidsynth() which creates a temporary
        FluidSynth instance. The GUI should stop MIDI playback before calling this.
        """
        try:
            # Check for cancellation before starting
            if self._cancel_flag.is_set():
                self.cancelled.emit()
                return

            # SAFETY: Log that FluidSynth synthesis is starting
            # (should never conflict due to GUI stopping playback first, but defensive logging)
            print("BeatTrackingWorker: Starting analysis (FluidSynth synthesis)")

            # Import here to avoid circular imports
            from analysis.beat_tracking_service import generate_beat_track

            # Report progress at start
            self.progress.emit(10)

            # Check for cancellation
            if self._cancel_flag.is_set():
                self.cancelled.emit()
                return

            # Run the analysis
            # Note: The actual analysis doesn't support incremental progress,
            # so we simulate progress at key points
            self.progress.emit(30)

            result = generate_beat_track(self.midi_path, self.sf2_path)

            # Check for cancellation after analysis
            if self._cancel_flag.is_set():
                self.cancelled.emit()
                return

            self.progress.emit(90)

            # Check for errors in result
            if result.get('errors'):
                error_msg = '; '.join(result['errors'])
                self.error.emit(error_msg)
                return

            # Report completion
            self.progress.emit(100)
            self.finished.emit(result)

        except Exception as e:
            # Catch any unexpected exceptions
            self.error.emit(f"Unexpected error during beat tracking: {str(e)}")

    def cancel(self):
        """
        Cancel the running analysis.

        This sets a flag that the run() method checks periodically.
        The cancellation is cooperative and may not be immediate.
        """
        self._cancel_flag.set()

    def is_cancelled(self) -> bool:
        """
        Check if cancellation has been requested.

        Returns:
            True if cancel() has been called
        """
        return self._cancel_flag.is_set()
