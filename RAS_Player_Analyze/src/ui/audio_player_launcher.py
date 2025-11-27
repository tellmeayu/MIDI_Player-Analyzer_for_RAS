"""
Audio Player Launcher - Entry point for audio player child process
Provides process-level isolation for the audio-based player
"""
import sys
from PyQt5.QtWidgets import QApplication


def launch_audio_player_process(midi_path: str, sf2_path: str):
    """Entry point for audio player child process.

    This function runs in a separate process, providing OS-level isolation:
    - Separate memory space
    - Independent FluidSynth instance
    - Separate audio device handles
    - No shared state with main process

    Args:
        midi_path: Path to MIDI file
        sf2_path: Path to SoundFont file
    """
    # Import here (not at module level) to avoid issues with multiprocessing
    from ui.audio_player_window import AudioPlayerWindow

    # Create QApplication for child process
    app = QApplication(sys.argv)

    # Create and show window
    window = AudioPlayerWindow(midi_path, sf2_path)
    window.show()

    # Run event loop (blocks until window closes)
    sys.exit(app.exec_())
