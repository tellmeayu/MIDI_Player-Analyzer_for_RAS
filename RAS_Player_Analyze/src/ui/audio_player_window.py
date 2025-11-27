"""Audio Player Window for cadence-based playback.

This module provides a separate window for audio-based playback with
real-time cadence control and time-stretching.
"""

import numpy as np
import librosa
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QLabel, QSpinBox, QGroupBox, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QEvent
from PyQt5.QtGui import QMouseEvent
from typing import Optional

try:
    import sounddevice as sd
    _SOUNDDEVICE_AVAILABLE = True
except ImportError:
    _SOUNDDEVICE_AVAILABLE = False


class SeekProgressBar(QProgressBar):
    """Custom progress bar for seeking that's immune to interference from other sliders."""
    
    seekPressed = pyqtSignal()
    seekReleased = pyqtSignal(float)  # Emits position ratio (0-1)
    seekMoved = pyqtSignal(float)     # Emits position ratio (0-1) during dragging
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(0, 1000)  # Use 0-1000 for smooth seeking
        self.setValue(0)
        self.setTextVisible(False)
        self.setMinimumHeight(15)
        
        # State tracking
        self.dragging = False
        self.setMouseTracking(True)
        
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press on the progress bar."""
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.seekPressed.emit()
            self._updatePositionFromMouse(event)
            
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release on the progress bar."""
        if event.button() == Qt.LeftButton and self.dragging:
            self.dragging = False
            ratio = self.value() / 1000.0
            self.seekReleased.emit(ratio)
            
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse movement on the progress bar."""
        if self.dragging:
            self._updatePositionFromMouse(event)
            
    def _updatePositionFromMouse(self, event: QMouseEvent):
        """Update position based on mouse position."""
        width = self.width()
        if width > 0:
            x = max(0, min(event.x(), width))
            ratio = x / width
            self.setValue(int(ratio * 1000))
            self.seekMoved.emit(ratio)


class AudioPlayerWindow(QMainWindow):
    """Audio player window with cadence-based time-stretching.

    This window loads preprocessed audio artifacts (music + clicks) from cache
    and provides real-time cadence control via time-stretching.
    """

    closed = pyqtSignal()  # Emitted when window is closed

    def __init__(self, midi_path: str, sf2_path: str, parent=None):
        """Initialize the audio player window.

        Args:
            midi_path: Path to MIDI file
            sf2_path: Path to SoundFont file
            parent: Parent widget
        """
        super().__init__(parent)

        if not _SOUNDDEVICE_AVAILABLE:
            raise ImportError("sounddevice library is required for audio playback")

        self.midi_path = midi_path
        self.sf2_path = sf2_path
        self.setWindowTitle("Audio Player - RAS Therapy")
        self.setMinimumSize(600, 400)

        # Audio state
        self.original_audio: Optional[np.ndarray] = None
        self.stretched_audio: Optional[np.ndarray] = None
        self.sample_rate: int = 22050
        self.beat_times: Optional[np.ndarray] = None
        self.estimated_tempo: float = 120.0
        self.current_cadence: float = 120.0
        self.current_stretch_ratio: float = 1.0

        # Playback state
        self.is_playing: bool = False
        self.is_paused: bool = False
        self.playback_position: int = 0  # Position in samples
        self.stream: Optional[sd.OutputStream] = None
        
        # UI state flags
        self._updating_progress: bool = False
        self._was_playing_before_seek: bool = False
        self._was_paused_before_seek: bool = False

        # Timer for UI updates
        self.ui_update_timer = QTimer()
        self.ui_update_timer.timeout.connect(self._update_ui_state)
        self.ui_update_timer.setInterval(100)  # 10 Hz update rate (reduced from 20 Hz)

        # Load or generate audio artifacts (self-contained)
        self._load_or_generate_artifacts()

        # Setup UI
        self._init_ui()

        # Start UI update timer
        self.ui_update_timer.start()

    def _load_or_generate_artifacts(self):
        """Load audio artifacts from cache, or generate them if not cached.

        This method makes the audio player completely self-contained:
        1. Check if cache exists
        2. If no cache: run beat tracking → synthesize audio → cache
        3. Load from cache
        """
        from core.audio_cache import get_cache
        from analysis.beat_tracker_basic import MidiBeatTracker

        cache = get_cache()

        # Compute cache key from file paths
        cache_key, exists = cache.get_or_create_key(
            midi_path=self.midi_path,
            sf2_path=self.sf2_path,
            sample_rate=22050
        )

        if not exists:
            print(f"Audio Player: No cache found, generating artifacts...")

            # Step 1: Beat tracking
            print(f"Audio Player: Running beat tracking...")
            tracker = MidiBeatTracker(self.midi_path, self.sf2_path)
            beat_times = tracker.get_beat_times()
            estimated_tempo = tracker.get_estimated_tempo()
            regularity = tracker.get_beat_regularity()

            if beat_times is None or len(beat_times) == 0:
                raise RuntimeError("Beat tracking failed: No beats detected")

            print(f"Audio Player: Detected {len(beat_times)} beats at {estimated_tempo:.1f} BPM")

            # Step 2: Synthesize audio with click overlay
            print(f"Audio Player: Synthesizing audio with click overlay...")
            mixed_audio, sr = tracker.generate_full_click_track(sample_rate=22050)

            if mixed_audio is None or len(mixed_audio) == 0:
                raise RuntimeError("Audio synthesis failed")

            print(f"Audio Player: Synthesized {len(mixed_audio)} samples @ {sr} Hz")

            # Step 3: Cache artifacts
            print(f"Audio Player: Caching artifacts...")
            cache_metadata = {
                'estimated_tempo_bpm': estimated_tempo,
                'regularity': regularity,
                'total_beats': len(beat_times),
                'sample_rate': sr
            }

            cache.cache_audio_artifact(
                midi_path=self.midi_path,
                sf2_path=self.sf2_path,
                mixed_audio=mixed_audio,
                sample_rate=sr,
                beat_times=beat_times,
                metadata=cache_metadata
            )

            print(f"Audio Player: Artifacts cached successfully")

        # Load from cache
        print(f"Audio Player: Loading artifacts from cache...")
        artifacts = cache.get_cached_artifact(cache_key)

        if artifacts is None:
            raise RuntimeError(f"Failed to load cached artifacts")

        self.original_audio = artifacts['mixed_audio']
        self.sample_rate = artifacts['sample_rate']
        self.beat_times = artifacts['beat_times']

        metadata = artifacts['metadata']
        self.estimated_tempo = metadata.get('estimated_tempo_bpm', 120.0)
        self.current_cadence = self.estimated_tempo

        # Initialize stretched audio (no stretch initially)
        self.stretched_audio = self.original_audio.copy()
        self.current_stretch_ratio = 1.0

        print(f"Audio Player: Loaded {len(self.original_audio)} samples @ {self.sample_rate} Hz")
        print(f"Audio Player: Estimated tempo: {self.estimated_tempo:.1f} BPM")
        print(f"Audio Player: Total beats: {len(self.beat_times)}")

    def _init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # === Tempo/Cadence Control ===
        tempo_group = QGroupBox("Cadence Control")
        tempo_layout = QVBoxLayout()

        # First row: Cadence input and info
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Target Cadence (steps/min):"))

        self.cadence_spinbox = QSpinBox()
        self.cadence_spinbox.setRange(20, 180)
        self.cadence_spinbox.setValue(int(self.current_cadence))
        self.cadence_spinbox.setSuffix(" steps/min")
        # Remove immediate change connection - only update on confirm
        input_layout.addWidget(self.cadence_spinbox)

        self.tempo_label = QLabel(f"Original: {self.estimated_tempo:.1f} BPM")
        input_layout.addWidget(self.tempo_label)

        self.stretch_label = QLabel(f"Stretch: {self.current_stretch_ratio:.3f}x")
        input_layout.addWidget(self.stretch_label)

        input_layout.addStretch()
        tempo_layout.addLayout(input_layout)

        # Second row: Confirm button
        confirm_layout = QHBoxLayout()
        self.confirm_cadence_button = QPushButton("Apply Cadence Change")
        self.confirm_cadence_button.clicked.connect(self._on_confirm_cadence)
        self.confirm_cadence_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 5px; }")
        confirm_layout.addWidget(self.confirm_cadence_button)
        confirm_layout.addStretch()
        tempo_layout.addLayout(confirm_layout)

        tempo_group.setLayout(tempo_layout)
        main_layout.addWidget(tempo_group)

        # === Playback Controls ===
        controls_layout = QHBoxLayout()

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._on_play_clicked)
        controls_layout.addWidget(self.play_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self._on_pause_clicked)
        self.pause_button.setEnabled(False)
        controls_layout.addWidget(self.pause_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.stop_button)

        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        # === Progress Bar ===
        progress_layout = QVBoxLayout()

        # Use our custom progress bar instead of a slider
        self.progress_bar = SeekProgressBar()
        self.progress_bar.setStyleSheet("QProgressBar { background-color: #E0E0E0; border: 1px solid #AAAAAA; border-radius: 3px; } QProgressBar::chunk { background-color: #4CAF50; }")
        self.progress_bar.seekPressed.connect(self._on_seek_pressed)
        self.progress_bar.seekReleased.connect(self._on_seek_released)
        self.progress_bar.seekMoved.connect(self._on_seek_moved)
        progress_layout.addWidget(self.progress_bar)

        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.time_label)

        main_layout.addLayout(progress_layout)

        # === Beat Info ===
        beat_info_layout = QHBoxLayout()

        self.beat_label = QLabel("Beat: 0 / 0")
        beat_info_layout.addWidget(self.beat_label)

        beat_info_layout.addStretch()
        main_layout.addLayout(beat_info_layout)

        # === Volume Control ===
        volume_layout = QHBoxLayout()

        volume_layout.addWidget(QLabel("Volume:"))

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)
        self.volume_slider.setMaximumWidth(200)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        # Add a small delay to prevent rapid updates
        self.volume_slider.setTracking(True)
        volume_layout.addWidget(self.volume_slider)

        self.volume_label = QLabel("70%")
        volume_layout.addWidget(self.volume_label)

        volume_layout.addStretch()
        main_layout.addLayout(volume_layout)

        main_layout.addStretch()

    def _on_confirm_cadence(self):
        """Handle cadence confirmation button click."""
        new_cadence = float(self.cadence_spinbox.value())
        
        if abs(new_cadence - self.current_cadence) < 0.1:
            return  # No significant change
        
        # Store previous state
        was_playing = self.is_playing
        was_paused = self.is_paused
        
        # Pause playback first
        if was_playing:
            self._pause_playback()
        
        # Update cadence
        self.current_cadence = new_cadence
        
        # Calculate stretch ratio
        self.current_stretch_ratio = new_cadence / self.estimated_tempo
        
        # Update label
        self.stretch_label.setText(f"Stretch: {self.current_stretch_ratio:.3f}x")
        
        # Apply time-stretching
        self._apply_time_stretch()
        
        # Resume playback if it was playing
        if was_playing:
            self._resume_playback()
        elif was_paused:
            # Keep paused state
            self.is_paused = True
            self.is_playing = False


    def _apply_time_stretch(self):
        """Apply time-stretching to audio based on current stretch ratio."""
        if self.original_audio is None:
            return

        print(f"Audio Player: Applying time-stretch with ratio {self.current_stretch_ratio:.3f}...")

        # Apply time-stretch using librosa
        # Note: librosa.effects.time_stretch uses rate parameter (inverse of ratio)
        # rate > 1.0 = faster, rate < 1.0 = slower
        # But we want stretch_ratio where > 1.0 = faster
        # So we use rate = stretch_ratio

        self.stretched_audio = librosa.effects.time_stretch(
            y=self.original_audio,
            rate=self.current_stretch_ratio
        )

        print(f"Audio Player: Stretched audio from {len(self.original_audio)} to {len(self.stretched_audio)} samples")

    def _on_play_clicked(self):
        """Handle Play button click."""
        if self.is_paused:
            # Resume from pause
            self._resume_playback()
        else:
            # Start from beginning or current position
            self._start_playback()

        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)

    def _on_pause_clicked(self):
        """Handle Pause button click."""
        self._pause_playback()

        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)

    def _on_stop_clicked(self):
        """Handle Stop button click."""
        self.is_playing = False
        self.is_paused = False
        self.playback_position = 0

        self._stop_playback()

        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)

    def _pause_playback(self):
        """Pause audio playback while preserving position."""
        if self.stream is not None:
            try:
                self.stream.stop()
                # Don't close the stream - just stop it
            except Exception as e:
                print(f"Audio Player: Error pausing stream: {e}")
        
        self.is_playing = False
        self.is_paused = True

    def _resume_playback(self):
        """Resume audio playback from current position."""
        if self.stream is not None:
            try:
                self.stream.start()
                self.is_playing = True
                self.is_paused = False
            except Exception as e:
                print(f"Audio Player: Error resuming stream: {e}")
                # If resume fails, restart the stream
                self._start_playback()

    def _start_playback(self):
        """Start audio playback from current position."""
        if self.stretched_audio is None:
            return

        self.is_playing = True

        # Create output stream
        def audio_callback(outdata, frames, time_info, status):
            if status:
                print(f"Audio Player: Stream status: {status}")

            if not self.is_playing:
                outdata.fill(0)
                return

            # Get audio chunk
            start_idx = self.playback_position
            end_idx = start_idx + frames

            if start_idx >= len(self.stretched_audio):
                # Reached end
                outdata.fill(0)
                self.is_playing = False
                return

            # Copy audio data
            chunk = self.stretched_audio[start_idx:end_idx]

            # Handle stereo/mono
            if chunk.ndim == 1:
                # Mono to stereo
                outdata[:len(chunk), 0] = chunk
                outdata[:len(chunk), 1] = chunk
            else:
                outdata[:len(chunk)] = chunk

            # Fill remaining frames with silence if needed
            if len(chunk) < frames:
                outdata[len(chunk):].fill(0)
                self.is_playing = False

            # Update position
            self.playback_position = end_idx

            # Apply volume
            volume = self.volume_slider.value() / 100.0
            outdata *= volume

        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                callback=audio_callback,
                blocksize=2048
            )
            self.stream.start()
        except Exception as e:
            print(f"Audio Player: Failed to start playback: {e}")
            self.is_playing = False

    def _stop_playback(self):
        """Stop audio playback."""
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Audio Player: Error stopping stream: {e}")
            finally:
                self.stream = None

        self.is_playing = False

    def _on_seek_pressed(self):
        """Handle seek progress bar pressed."""
        self._updating_progress = True
        # Store current state before seeking
        self._was_playing_before_seek = self.is_playing
        self._was_paused_before_seek = self.is_paused
        
        # Pause playback during seeking
        if self.is_playing:
            self._pause_playback()

    def _on_seek_released(self, ratio: float):
        """Handle seek progress bar released.
        
        Args:
            ratio: Position ratio (0-1) from the progress bar
        """
        # Update playback position
        if self.stretched_audio is None:
            self._updating_progress = False
            return

        self.playback_position = int(ratio * len(self.stretched_audio))
        
        self._updating_progress = False

        # Resume playback if it was playing before seek
        if self._was_playing_before_seek:
            self._start_playback()
        elif self._was_paused_before_seek:
            # Keep paused state
            self.is_paused = True
            self.is_playing = False

    def _on_seek_moved(self, ratio: float):
        """Handle seek progress bar movement during dragging.
        
        Args:
            ratio: Position ratio (0-1) from the progress bar
        """
        if self.stretched_audio is None:
            return
        
        # Update playback position in real-time during dragging
        self.playback_position = int(ratio * len(self.stretched_audio))
        
        # Update time label in real-time for better feedback
        current_time_s = self.playback_position / self.sample_rate
        total_time_s = len(self.stretched_audio) / self.sample_rate
        current_min = int(current_time_s // 60)
        current_sec = int(current_time_s % 60)
        total_min = int(total_time_s // 60)
        total_sec = int(total_time_s % 60)
        self.time_label.setText(f"{current_min}:{current_sec:02d} / {total_min}:{total_sec:02d}")

    def _on_volume_changed(self, value: int):
        """Handle volume slider change."""
        self.volume_label.setText(f"{value}%")

    def _update_ui_state(self):
        """Update UI elements based on current playback state."""
        if self.stretched_audio is None:
            return

        # Update progress bar (if not being dragged and audio is playing)
        if (not self.progress_bar.dragging and 
            not self._updating_progress and 
            self.is_playing and 
            len(self.stretched_audio) > 0):
            ratio = self.playback_position / len(self.stretched_audio)
            new_value = int(ratio * 1000)
            # Only update if the value has actually changed to prevent unnecessary redraws
            if abs(new_value - self.progress_bar.value()) > 1:
                self._updating_progress = True
                self.progress_bar.setValue(new_value)
                self._updating_progress = False

        # Update time label
        current_time_s = self.playback_position / self.sample_rate
        total_time_s = len(self.stretched_audio) / self.sample_rate

        current_min = int(current_time_s // 60)
        current_sec = int(current_time_s % 60)
        total_min = int(total_time_s // 60)
        total_sec = int(total_time_s % 60)

        self.time_label.setText(f"{current_min}:{current_sec:02d} / {total_min}:{total_sec:02d}")

        # Update beat number
        if self.beat_times is not None:
            # Find current beat based on playback position in original audio
            # Need to map stretched position back to original time
            original_time_s = current_time_s / self.current_stretch_ratio

            # Find nearest beat
            beat_idx = np.searchsorted(self.beat_times, original_time_s)
            total_beats = len(self.beat_times)

            self.beat_label.setText(f"Beat: {beat_idx} / {total_beats}")

        # Check if playback finished
        if self.is_playing and self.playback_position >= len(self.stretched_audio):
            self.is_playing = False
            self.is_paused = False
            self._stop_playback()

            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)

    def closeEvent(self, event):
        """Handle window close event."""
        # Stop playback
        self._stop_playback()

        # Stop UI update timer
        self.ui_update_timer.stop()

        # Emit closed signal
        self.closed.emit()

        event.accept()
