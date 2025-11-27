from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QSlider, QLabel, QFileDialog, QListWidget, QListWidgetItem, 
    QCheckBox, QGroupBox, QGridLayout, QSpinBox, QFrame, QAction, QMenu,
    QDialog, QScrollArea, QPushButton, QComboBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon
import os
import time
from functools import partial

# Import separated UI components
from ui.dialogs import LoadingDialog
from ui.file_info_display import FileInfoDisplay
from ui.playback_controls import PlaybackControls
from ui.track_visualization import TrackVisualizationWidget
from ui.analysis_dialogs import BeatTrackingDialog
from .menu_manager import MenuManager

# Import core components with correct path
from core.metronome import SOUND_TYPE_CLICK, SOUND_TYPE_BEEP, SOUND_TYPE_WOOD, SOUND_TYPE_TICK
from core.ras_therapy_metronome import RASTherapyMetronome
from core.precision_timer import PrecisionTimer
from core.playback_mode import PlaybackMode
from .utilities import handle_file_loaded, _calculate_cadence_from_section_tempo

class MainWindow(QMainWindow):
    """Main window class"""
    
    def __init__(self, engine, scheduler):
        """Initialize main window
            Args:
                engine: MIDI engine instance
                scheduler: Event scheduler instance
        """
        super().__init__()
        
        self.engine = engine
        self.scheduler = scheduler
        self.beat_hint = RASTherapyMetronome(engine)
        self.loading_dialog = LoadingDialog(self)
        
        # Analysis dialogs
        self.beat_tracking_dialog = BeatTrackingDialog(self)
        
        # Soundfont path for analysis
        self.sf2_path = "resources/soundfonts/FluidR3_GM.sf2"
        
        self.setWindowTitle("RAS MIDI Analyzer & Player")
        self.setMinimumSize(900, 800)
        self.resize(1000, 920)
        
        # Set window icon if available - prefer high-resolution PNG for best quality
        hd_png_path = os.path.join(os.path.dirname(__file__), "ras_midi_icon_hd.png")
        if os.path.exists(hd_png_path):
            self.setWindowIcon(QIcon(hd_png_path))
        else:
            # Try regular PNG
            png_path = os.path.join(os.path.dirname(__file__), "ras_midi_icon.png")
            if os.path.exists(png_path):
                self.setWindowIcon(QIcon(png_path))
            else:
                # Fallback to ICO file
                ico_path = os.path.join(os.path.dirname(__file__), "ras_midi_icon.ico")
                if os.path.exists(ico_path):
                    self.setWindowIcon(QIcon(ico_path))
        
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(100)  # Update UI every 100ms
        
        # Track tempo adjustment state for accumulative percentage changes
        self.base_tempo_for_adjustment = 120.0
        self.accumulated_percentage_change = 0.0  # Accumulated percentage change
        
        # Track cadence adjustment state for RAS therapy
        self.base_cadence_for_adjustment = 60.0  # Default 60 steps/min
        self.accumulated_cadence_percentage_change = 0.0  # Accumulated percentage change for cadence
        self.RAS_cue = 60.0  # Default RAS cue for RAS therapy
        
        # Track visualization system
        self.track_visualization_enabled = True

        # Audio player state (Phase 1F - Process-level Isolation)
        self.audio_player_process = None  # Track audio player child process

        # Setup menu using MenuManager
        self.menu_manager = MenuManager(self)
        self.menu_manager.setup_menu()
        self.setup_ui()
        
        # Setup activity monitoring callback
        self.scheduler.add_activity_callback(self.on_track_activity_changed)
    
    def setup_ui(self):
        """Setup UI components"""
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # File information area - using separated component
        self.file_info_display = FileInfoDisplay()
        main_layout.addWidget(self.file_info_display)

        # Playback controls area - using separated component (includes section selection)
        self.playback_controls = PlaybackControls()
        
        # Connect playback control signals
        self.playback_controls.file_loaded.connect(partial(handle_file_loaded, self))
        self.playback_controls.play_requested.connect(self.play)
        self.playback_controls.pause_requested.connect(self.pause)
        self.playback_controls.stop_requested.connect(self.stop)
        self.playback_controls.midi_volume_changed.connect(self.set_midi_volume)
        
        # Connect analysis signals
        self.playback_controls.generate_beat_track_requested.connect(self.on_generate_beat_track)
        self.playback_controls.revert_beat_track_requested.connect(self.on_revert_beat_track)
        self.playback_controls.revert_anacrusis_requested.connect(self.on_revert_anacrusis)
        self.playback_controls.fine_adjustment_changed.connect(self.on_fine_adjustment_changed)

        # Connect audio player signal (Phase 1F)
        self.playback_controls.audio_player_requested.connect(self.on_open_audio_player)

        # Connect analysis dialog signals
        self.beat_tracking_dialog.cancelled.connect(self.on_beat_tracking_cancelled)
        
        # Connect section control signals
        self.playback_controls.section_selected.connect(self.update_section_info_label)
        self.playback_controls.section_confirm_requested.connect(self.confirm_and_play_section)
        
        # Set maximum height for playback controls to reduce space usage
        self.playback_controls.setMaximumHeight(260)  # Limit height to reduce space
        main_layout.addWidget(self.playback_controls)
        
        # Set initial MIDI volume to match the slider default (35%)
        self.set_midi_volume(35)
        
        # Enhanced Beat hint control area
        beat_hint_group = QGroupBox("RAS Therapy Beat Cue")
        beat_hint_layout = QVBoxLayout()
        
        # Beat hint toggle
        beat_hint_toggle_layout = QHBoxLayout()
        self.beat_hint_checkbox = QCheckBox("Enable Beat Cue")
        self.beat_hint_checkbox.toggled.connect(self.toggle_beat_hint)
        
        beat_hint_toggle_layout.addWidget(self.beat_hint_checkbox)
        beat_hint_toggle_layout.addStretch()
        
        beat_hint_layout.addLayout(beat_hint_toggle_layout)
        
        # Cadence control with confirmation (for RAS therapy) - all controls in one row
        cadence_layout = QHBoxLayout()
        cadence_layout.addWidget(QLabel("Cadence:"))
        
        # Replace tempo spinbox with cadence spinbox for precise input
        self.cadence_spinbox = QSpinBox()
        self.cadence_spinbox.setRange(1, 999)  # Allow any positive cadence value
        self.cadence_spinbox.setValue(60)  # Default 60 steps/min
        self.cadence_spinbox.setSuffix(" steps/min")
        self.cadence_spinbox.setToolTip("Enter desired cadence (any positive value)")
        
        # Fix focus behavior - clear focus when editing is finished
        self.cadence_spinbox.setFocusPolicy(Qt.ClickFocus)  # type: ignore # Only get focus when clicked
        
        # Connect Enter key to apply cadence change
        self.cadence_spinbox.editingFinished.connect(self.apply_cadence_change)
        
        # Add value change detection for visual feedback
        self.cadence_spinbox.valueChanged.connect(self.on_cadence_input_changed)
        
        # Quick cadence adjustment buttons
        self.cadence_decrease_button = QPushButton("-5%")
        self.cadence_decrease_button.clicked.connect(self.decrease_cadence_5_percent)
        self.cadence_decrease_button.setToolTip("Decrease cadence by 5%")
        self.cadence_decrease_button.setMaximumWidth(50)
        
        self.cadence_increase_button = QPushButton("+5%")
        self.cadence_increase_button.clicked.connect(self.increase_cadence_5_percent)
        self.cadence_increase_button.setToolTip("Increase cadence by 5%")
        self.cadence_increase_button.setMaximumWidth(50)
        
        # Add confirmation button
        self.cadence_confirm_button = QPushButton("Apply Cadence")
        self.cadence_confirm_button.clicked.connect(self.apply_cadence_change)
        self.cadence_confirm_button.setToolTip("Click to apply the cadence change")
        
        # Add some spacing before sound controls
        cadence_layout.addWidget(self.cadence_spinbox)
        cadence_layout.addWidget(self.cadence_decrease_button)
        cadence_layout.addWidget(self.cadence_increase_button)
        cadence_layout.addWidget(self.cadence_confirm_button)
        cadence_layout.addSpacing(7) 
        
        # Sound selection dropdown (compact)
        cadence_layout.addWidget(QLabel("Sound:"))
        self.sound_combo = QComboBox()
        self.sound_combo.addItem("Click", SOUND_TYPE_CLICK)
        self.sound_combo.addItem("Beep", SOUND_TYPE_BEEP)
        self.sound_combo.addItem("Wood", SOUND_TYPE_WOOD)
        self.sound_combo.addItem("Tick", SOUND_TYPE_TICK)
        self.sound_combo.currentIndexChanged.connect(self.change_metronome_sound)
        self.sound_combo.setToolTip("Select metronome sound type")
        cadence_layout.addWidget(self.sound_combo)
        
        cadence_layout.addSpacing(7)
        
        # Volume control (improved)
        cadence_layout.addWidget(QLabel("Volume:"))
        
        self.volume_slider = QSlider(Qt.Horizontal) # type: ignore
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(60)  # Default to 60% volume
        self.volume_slider.valueChanged.connect(self.set_volume)
        self.volume_slider.setMaximumWidth(150)  # Increased from 100 to 130
        self.volume_slider.setMinimumWidth(110)  # Increased from 80 to 110
        
        self.volume_label = QLabel("60%")  # Default to 60% volume
        self.volume_label.setMinimumWidth(35)  # Fixed width for percentage display
        
        cadence_layout.addWidget(self.volume_slider)
        cadence_layout.addWidget(self.volume_label)
        cadence_layout.addStretch()  # Push everything to the left
        
        beat_hint_layout.addLayout(cadence_layout)
        
        # Set initial beat hint volume to 60%
        self.beat_hint.set_volume(0.6)

        # Set initial sound type to CLICK (default)
        self.beat_hint.set_sound_type(SOUND_TYPE_CLICK)
        
        beat_hint_group.setLayout(beat_hint_layout)
        main_layout.addWidget(beat_hint_group)
        
        # Track visualization area
        tracks_group = QGroupBox("Multi-track Control: click on track to mute/unmute")
        tracks_layout = QVBoxLayout()
        
        # Create scroll area for track visualization
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # type: ignore
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded) # type: ignore
        scroll_area.setFrameShape(QFrame.NoFrame)  # Remove frame for cleaner look
        scroll_area.setMinimumHeight(140)  # Slightly reduced minimum height
        scroll_area.setMaximumHeight(400)  # Reduced maximum for more compact layout
        
        # Create track visualization widget
        self.track_visualization = TrackVisualizationWidget()
        self.track_visualization.track_clicked_callback = self.track_clicked_from_visualization
        
        # Set the track visualization as the scroll area's widget
        scroll_area.setWidget(self.track_visualization)
        
        # Add scroll area to tracks layout instead of direct widget
        tracks_layout.addWidget(scroll_area)
        
        tracks_group.setLayout(tracks_layout)
        main_layout.addWidget(tracks_group)
        
        self.setCentralWidget(central_widget)
        
        # Install event filter to handle clicks outside tempo spinbox
        self.installEventFilter(self)
    
    def eventFilter(self, obj, event):
        """Event filter to handle focus management"""
        from PyQt5.QtCore import QEvent, Qt as QtCore_Qt
        from PyQt5.QtGui import QMouseEvent, QKeyEvent
        
        if event.type() == QEvent.MouseButtonPress:
            if isinstance(event, QMouseEvent):
                # Clear focus from tempo spinbox when clicking elsewhere
                if self.cadence_spinbox.hasFocus():
                    clicked_widget = self.childAt(event.pos())
                    if clicked_widget != self.cadence_spinbox and not self.cadence_spinbox.isAncestorOf(clicked_widget):
                        self.cadence_spinbox.clearFocus()
        
        elif event.type() == QEvent.KeyPress:
            if isinstance(event, QKeyEvent):
                # Clear focus when Escape is pressed
                if event.key() == QtCore_Qt.Key_Escape and self.cadence_spinbox.hasFocus():
                    self.cadence_spinbox.clearFocus()
        
        return super().eventFilter(obj, event)
    
    def toggle_track_visualization(self, enabled):
        """Toggle track visualization on/off
        Args:
            enabled: Whether visualization is enabled
        """
        self.track_visualization_enabled = enabled
        self.track_visualization.setVisible(enabled)
    
    def set_volume(self, volume):
        """Set beat hint volume
        
        Args:
            volume: Volume value (0-100)
        """
        volume_float = volume / 100.0
        self.beat_hint.set_volume(volume_float)
        self.volume_label.setText(f"{volume}%")
    
    def set_midi_volume(self, volume):
        """Set MIDI music volume
        
        Args:
            volume: Volume value (0-100)
        """
        volume_float = volume / 100.0
        if hasattr(self.engine, 'set_volume'):
            self.engine.set_volume(volume_float)
        # Update the label in the playback controls component
        self.playback_controls.midi_volume_label.setText(f"{volume}%")
    
    def play(self):
        """Play MIDI file from selected section"""
        # set the scheduler on the engine so play() can use it
        self.engine.scheduler = self.scheduler
        # get the selected section
        section = None
        if hasattr(self, 'sections') and self.sections:
            section_index = self.playback_controls.get_selected_section_index()
            section = self.sections[section_index]
        
        # Make sure the cadence display is synced with the actual tempo before playing
        # (especially important when starting playback after adjusting cadence while stopped)
        self.sync_cadence_with_current_section()
        
        # Use unified play method with section parameter
        self.engine.play(section)
        
        # start precision beat hint if enabled
        if self.beat_hint_checkbox.isChecked():
            self.beat_hint.start()
    
    def pause(self):
        """Pause playbook"""
        # Stop beat hint first
        self.beat_hint.stop()
        self.engine.pause()
    
    def stop(self):
        """Stop playback"""
        # Stop beat hint first to avoid interference
        self.beat_hint.stop()
        self.engine.stop()
        self.scheduler.stop()
        # Ensure all notes are stopped
        self.engine.all_notes_off()
        # Reset position display using the separated component
        self.file_info_display.update_stopped_state(self.engine)
    
    def apply_cadence_change(self):
        """Apply cadence change - converts cadence directly to MIDI tempo"""
        new_cadence = self.cadence_spinbox.value()  # This is already an integer
        
        # In RAS system: Cadence = Musical Tempo (1:1 relationship)
        musical_tempo = float(new_cadence) # float for calculation in precision timer
        
        # Convert musical tempo to MIDI tempo for engine, using current time signature
        section_idx = self.playback_controls.get_selected_section_index()
        section_ts = self.sections[section_idx].get('time_signature', {})
        section_ts_tuple = (section_ts.get('numerator','4'), section_ts.get('denominator','4'))
        midi_tempo = PrecisionTimer.convert_musical_tempo_to_midi_tempo(musical_tempo, section_ts_tuple)
            
        # Apply the calculated MIDI tempo to the engine
        self.engine.set_tempo(midi_tempo)

        # Store the user's preferred tempo in session state for persistence
        self.engine.session_state.user_preferred_tempo = midi_tempo
        
        # If we have a current section, also update its tempo so it persists when restarted
        if self.engine.current_section:
            self.engine.current_section['tempo'] = midi_tempo

        # update playback musical tempo label, cadence equals to musical tempo in RAS system
        self.playback_controls.update_section_tempo(new_cadence)
        
        # Notify scheduler to update activity monitor tempo
        if hasattr(self.scheduler, 'set_tempo'):
            self.scheduler.set_tempo(midi_tempo)
        
        # Reset cadence adjustment tracking for next round of percentage changes
        self.base_cadence_for_adjustment = float(new_cadence)
        self.accumulated_cadence_percentage_change = 0.0
        
        # Reset button style after applying
        self.cadence_confirm_button.setStyleSheet("")
        self.cadence_confirm_button.setText("Apply Cadence")
    
    def on_cadence_input_changed(self):
        """Handle cadence input change for visual feedback"""
        if not self.engine.midi_file:
            # Don't show preview when no file is loaded
            self.cadence_confirm_button.setStyleSheet("")
            self.cadence_confirm_button.setText("Apply Cadence")
            return
            
        # Get current cadence from RAS metronome
        current_cadence = round(self.beat_hint.step_frequency)
        input_cadence = self.cadence_spinbox.value()
        
        # Only show button highlight if input differs from current
        if abs(input_cadence - current_cadence) > 0:
            # Highlight button to show there's a pending change
            self.cadence_confirm_button.setStyleSheet("background-color: #4CAF50; font-weight: bold;")
            self.cadence_confirm_button.setText("Apply Cadence ✓")
        else:
            # Reset button style if value matches current cadence
            self.cadence_confirm_button.setStyleSheet("")
            self.cadence_confirm_button.setText("Apply Cadence")
    
    def decrease_cadence_5_percent(self):
        """Decrease cadence by 5% of base cadence (accumulative)"""
        self.accumulated_cadence_percentage_change -= 5.0
        # Calculate new cadence based on base cadence and accumulated percentage
        new_cadence = self.base_cadence_for_adjustment * (1 + self.accumulated_cadence_percentage_change / 100.0)
        # Round to nearest integer and ensure within bounds
        new_cadence_int = max(1, round(new_cadence))
        # Update spinbox value (this will trigger visual feedback)
        self.cadence_spinbox.setValue(new_cadence_int)
           
    def increase_cadence_5_percent(self):
        """Increase cadence by 5% of base cadence (accumulative)"""
        self.accumulated_cadence_percentage_change += 5.0
        # Calculate new cadence based on base cadence and accumulated percentage
        new_cadence = self.base_cadence_for_adjustment * (1 + self.accumulated_cadence_percentage_change / 100.0)        
        # Round to nearest integer and ensure within bounds
        new_cadence_int = max(1, round(new_cadence))        
        # Update spinbox value (this will trigger visual feedback)
        self.cadence_spinbox.setValue(new_cadence_int)
            
    def toggle_beat_hint(self, checked):
        """Toggle beat hint on/off
        Args:
            checked: Whether checkbox is checked
        """
        self.beat_hint.is_active = checked
        
        if checked and self.engine.is_playing:
            self.beat_hint.start()
        else:
            self.beat_hint.stop()
    
    def track_clicked_from_visualization(self, track_id):
        """Handle track click from visualization widget        
        Args:
            track_id: Track index that was clicked
        """
        self.engine.mute_track(track_id)
        self.update_track_mute_states()
    
    def update_track_visualization(self):
        """Update track visualization with current tracks"""
        self.track_visualization.clear_tracks()
        
        if not self.engine.midi_file:
            return
        
        metadata = self.engine.get_metadata()
        tracks_info = metadata.get('tracks_info', [])
        
        for track in tracks_info:
            track_id = track['index']
            track_name = track['name']
            instrument_display = track.get('instrument_display')
            
            # Add track to visualization
            self.track_visualization.add_track(track_id, track_name, instrument_display)
        
        # Finalize layout
        self.track_visualization.finalize_layout()       
        # Update mute states
        self.update_track_mute_states()        
    
    def update_track_mute_states(self):
        """Update visual mute states for all tracks"""
        for track_id in self.track_visualization.track_widgets.keys():
            is_muted = track_id in self.engine.muted_tracks
            self.track_visualization.set_track_muted(track_id, is_muted)
    
    def on_track_activity_changed(self, track_id, activity_state):
        """Callback for track activity changes
        
        Args:
            track_id: Track index
            activity_state: TrackActivityState object
        """
        if self.track_visualization_enabled:
            self.track_visualization.update_track_activity(track_id, activity_state)
    
    def update_ui(self):
        """Periodically update UI with precision timing information"""
        # Update playback position using precision timer
        if self.engine.midi_file:
            # Update playback mode UI (analysis buttons visibility)
            self.playback_controls.update_playback_mode(self.engine)
            
            if self.engine.is_playing or self.engine.is_paused:
                # Get high-precision timing information
                timing_info = self.engine.get_timing_info()
                # Update playback info using the separated component
                self.file_info_display.update_playback_info(timing_info)
            
            # --- update playback controls with section-local progress ---
            section = None
            if hasattr(self, 'sections') and self.sections:
                section_index = self.playback_controls.get_selected_section_index()
                section = self.sections[section_index]
            
            if section:
                # Get section-specific playback information from engine
                section_info = self.engine.get_section_playback_info()
                
                if section_info.get('has_section', False):
                    # Calculate section-local measure from section-relative beats
                    section_position_beats = section_info['section_position_beats']
                    
                    # Get time signature from current section
                    time_sig = section.get('time_signature', {})
                    beats_per_measure = time_sig.get('numerator', 4)
                    
                    # Calculate section-local measure (1-based)
                    current_section_measure = int(section_position_beats // beats_per_measure) + 1
                    total_section_measures = section.get('measures', 1)
                    
                    # Ensure current measure doesn't exceed total measures
                    current_section_measure = min(current_section_measure, total_section_measures)
                    
                    self.playback_controls.update_section_progress(current_section_measure, total_section_measures)
                    
                    # Get musical beat position from precision timer (aligned with RAS cues)
                    beat_event = self.engine.precision_timer.get_beat_position()
                    
                    # Get musical beats per measure for display
                    numerator = time_sig.get('numerator', 4)
                    denominator = time_sig.get('denominator', 4)
                    musical_beats_per_measure = self.engine.precision_timer.calculate_musical_beats_per_measure(numerator, denominator)
                    
                    # Adjust the displayed beat to account for anacrusis offset
                    # This makes beat 1 land on the first true downbeat of the music
                    if hasattr(self.engine.session_state, 'has_anacrusis_correction') and self.engine.session_state.has_anacrusis_correction:
                        # Calculate adjusted beat position (still 0-based)
                        offset_beats_total = self.engine.session_state.anacrusis_offset_beats
                        
                        # Calculate adjusted measure and beat
                        total_beats = beat_event.measure * musical_beats_per_measure + beat_event.beat
                        adjusted_total_beats = total_beats - offset_beats_total
                        
                        # Ensure we don't have negative beats
                        adjusted_total_beats = max(0, adjusted_total_beats)
                        
                        # Recalculate measure and beat
                        adjusted_measure = int(adjusted_total_beats // musical_beats_per_measure)
                        adjusted_beat = int(adjusted_total_beats % musical_beats_per_measure)
                        
                        # Use the adjusted beat position
                        musical_beat_in_measure = adjusted_beat + 1  # Convert to 1-based
                    else:
                        # No anacrusis correction, use raw beat position
                        musical_beat_in_measure = beat_event.beat + 1  # Convert to 1-based
                    
                    # Use musical_beat_in_measure to determine if it's a strong beat
                    # In most time signatures, beat 1 is the strong beat
                    is_strong_beat = (musical_beat_in_measure == 1)
                    self.playback_controls.update_section_beat(musical_beat_in_measure, musical_beats_per_measure, is_strong_beat)
                    
                else:    # No section playback info available
                    self.playback_controls.update_section_progress('-', '-')
                    self.playback_controls.update_section_beat('-', '-')

            else:
                self.playback_controls.update_section_progress('-', '-')
                self.playback_controls.update_section_beat('-', '-')
                
        else:
            # Reset displays when not playing - use the separated component
            self.file_info_display.update_stopped_state(self.engine)
            self.playback_controls.update_section_progress('-', '-')
            self.playback_controls.update_section_beat('-', '-')
            
    def closeEvent(self, event):
        """Window close event
        
        Args:
            event: Close event
        """
        self.stop()
        
        # Clean up activity monitoring callback
        if hasattr(self.scheduler, 'remove_activity_callback'):
            self.scheduler.remove_activity_callback(self.on_track_activity_changed)
        
        event.accept()
    
    def change_metronome_sound(self, index):
        """Change metronome sound type
        
        Args:
            index: Selected combo box index
        """
        sound_type = self.sound_combo.itemData(index)
        if hasattr(self.beat_hint, 'set_sound_type'):
            self.beat_hint.set_sound_type(sound_type)
    
    def sync_cadence_with_current_section(self):
        """Synchronize cadence display with current playing section's tempo or user's preferred tempo"""
        if not hasattr(self, 'sections') or not self.sections:
            return
        
        # Get current section
        section_index = self.playback_controls.get_selected_section_index()
        if 0 <= section_index < len(self.sections):
            current_section = self.sections[section_index]
            section_time_signature = current_section.get('time_signature', {'numerator': 4, 'denominator': 4})
            
            # Check for user's preferred tempo first
            if hasattr(self.engine, 'session_state') and self.engine.session_state.user_preferred_tempo > 0:
                tempo_to_use = self.engine.session_state.user_preferred_tempo
            else:
                # Otherwise use section's default tempo
                tempo_to_use = current_section.get('tempo', 120.0)
            
            # Calculate cadence from the chosen tempo
            cadence = _calculate_cadence_from_section_tempo(tempo_to_use, section_time_signature)
            cadence_rounded = int(round(cadence))
            
            # Update cadence spinbox with the appropriate cadence
            self.cadence_spinbox.setValue(cadence_rounded)
            self.playback_controls.update_section_tempo(cadence_rounded)
            
            # Reset cadence adjustment tracking for new section
            self.base_cadence_for_adjustment = float(cadence_rounded)
            self.accumulated_cadence_percentage_change = 0.0
            
            # Reset button visual state
            self.cadence_confirm_button.setStyleSheet("")
            self.cadence_confirm_button.setText("Apply Cadence")
            
            section_label = current_section.get('label', f'Section {section_index + 1}')

    def load_file(self):
        """Load MIDI file - kept for MENU compatibility, delegates to playback controls"""
        self.playback_controls.load_file()
        
    def on_generate_beat_track(self):
        """Handle beat track generation request (with audio artifact caching)."""
        if not self.engine.midi_file or not hasattr(self.engine, 'session_state'):
            return

        # Check if we're in DYNAMIC mode
        if self.engine.session_state.mode != PlaybackMode.DYNAMIC:
            return

        # CRITICAL: Stop MIDI playback to avoid FluidSynth conflicts
        # Beat tracking uses pm.fluidsynth() which creates a temporary FluidSynth instance.
        # Running this while the main FluidSynth is active causes audio driver conflicts.
        was_playing = self.engine.is_playing
        was_paused = self.engine.is_paused
        if was_playing or was_paused:
            self.stop()
            # Brief pause to ensure FluidSynth cleanup completes
            time.sleep(0.1)

        # Show progress dialog
        self.beat_tracking_dialog.start()

        # Create worker and thread using Worker + QThread pattern
        from ui.beat_tracking_worker import BeatTrackingWorker
        from PyQt5.QtCore import QThread

        self.beat_tracking_worker = BeatTrackingWorker(
            self.engine.midi_file.filename,
            self.sf2_path
        )
        self.beat_tracking_thread = QThread()

        # Move worker to thread
        self.beat_tracking_worker.moveToThread(self.beat_tracking_thread)

        # Connect signals
        self.beat_tracking_thread.started.connect(self.beat_tracking_worker.run)
        self.beat_tracking_worker.finished.connect(self.on_beat_tracking_finished)
        self.beat_tracking_worker.error.connect(self.on_beat_tracking_error)
        self.beat_tracking_worker.progress.connect(self.on_beat_tracking_progress)
        self.beat_tracking_worker.cancelled.connect(self.on_beat_tracking_cancelled_worker)

        # Cleanup when done
        self.beat_tracking_worker.finished.connect(self.cleanup_beat_tracking_thread)
        self.beat_tracking_worker.error.connect(self.cleanup_beat_tracking_thread)
        self.beat_tracking_worker.cancelled.connect(self.cleanup_beat_tracking_thread)

        # Start thread
        self.beat_tracking_thread.start()

    def on_beat_tracking_finished(self, result):
        """Handle beat tracking completion (analysis only)."""
        # Extract data from result dictionary
        beat_times = result.get('beat_times')
        tempo = result.get('estimated_tempo_bpm', 0.0)
        regularity = result.get('regularity', {})
        total_beats = result.get('total_beats', 0)

        if beat_times is not None and len(beat_times) > 0:
            # Update session state
            self.engine.session_state.has_beat_track = True
            self.engine.session_state.beat_times = beat_times
            self.engine.session_state.estimated_tempo_bpm = tempo
            self.engine.session_state.beat_regularity = regularity
            
            # Set beat track in metronome
            self.beat_hint.set_beat_track(beat_times)
            
            # Calculate and set playback rate (initially 1.0 since target = estimated)
            playback_rate = 1.0
            self.beat_hint.set_playback_rate(playback_rate)
            
            # Update cadence to the estimated tempo
            # Convert musical tempo to MIDI tempo if needed
            time_sig = self.engine.precision_timer.time_signature
            midi_tempo = self.engine.precision_timer.convert_musical_tempo_to_midi_tempo(tempo, time_sig)
            self.engine.set_tempo(midi_tempo)
            
            # Update UI cadence controls to show the estimated tempo
            # The cadence spinbox is in the main window, not playback controls
            estimated_cadence = int(round(tempo))
            self.cadence_spinbox.setValue(estimated_cadence)
            # Note: RAS cue label is hidden in DYNAMIC mode, no need to update it
            
            # Reset base cadence for 5% adjustment buttons to use the estimated tempo
            self.base_cadence_for_adjustment = float(estimated_cadence)
            self.accumulated_cadence_percentage_change = 0.0
            print(f"Dynamic mode: Base cadence reset to {estimated_cadence} BPM for adjustment buttons")
            
            # Update beat regularity display
            self.playback_controls.update_beat_regularity(regularity)

            # Show Audio Player button after successful beat tracking
            self.playback_controls.show_audio_player_button()

            # Update UI
            message = f"Detected {total_beats} beats at {tempo:.1f} BPM"
            self.beat_tracking_dialog.complete(True, message)
            self.playback_controls.update_playback_mode(self.engine)
            
        else:
            # Update UI
            self.beat_tracking_dialog.complete(False, "Beat tracking failed: No beats detected")
            
    def on_beat_tracking_cancelled(self):
        """Handle beat tracking cancellation (old method for dialog cancel button)."""
        # Request cancellation from worker
        if hasattr(self, 'beat_tracking_worker') and self.beat_tracking_worker is not None:
            self.beat_tracking_worker.cancel()

    def on_beat_tracking_cancelled_worker(self):
        """Handle beat tracking cancellation from worker."""
        self.beat_tracking_dialog.complete(False, "Beat tracking cancelled")

    def on_beat_tracking_error(self, error_msg: str):
        """Handle beat tracking error."""
        self.beat_tracking_dialog.complete(False, f"Beat tracking error: {error_msg}")

    def on_beat_tracking_progress(self, percent: int):
        """Handle beat tracking progress updates."""
        # Update progress in dialog if it has progress reporting
        if hasattr(self.beat_tracking_dialog, 'update_progress'):
            self.beat_tracking_dialog.update_progress(percent)

    def cleanup_beat_tracking_thread(self):
        """Clean up beat tracking worker and thread after completion."""
        if hasattr(self, 'beat_tracking_thread') and self.beat_tracking_thread is not None and self.beat_tracking_thread.isRunning():
            self.beat_tracking_thread.quit()
            self.beat_tracking_thread.wait(1000)  # Wait up to 1 second

        # Clear references
        self.beat_tracking_worker = None
        self.beat_tracking_thread = None

    def on_rhythm_analysis_finished(self, result):
        """Handle rhythm analysis completion."""
        import os
        from ui.rhythm_analysis_dialog import RhythmAnalysisDialog
        
        # Extract filename from current MIDI file for piece name
        if self.engine.midi_file and hasattr(self.engine.midi_file, 'filename'):
            piece_name = os.path.splitext(os.path.basename(self.engine.midi_file.filename))[0]
        else:
            piece_name = "Unknown Piece"
        
        # Create and show dialog
        dialog = RhythmAnalysisDialog(result, piece_name, self)
        dialog.exec_()
        
        # Update status bar
        self.statusBar().showMessage("Rhythm analysis complete", 3000)

    def on_rhythm_analysis_error(self, error_msg: str):
        """Handle rhythm analysis error."""
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(
            self,
            "Rhythm Analysis Error",
            f"An error occurred during rhythm analysis:\n\n{error_msg}"
        )
        self.statusBar().showMessage("Rhythm analysis failed", 3000)

    def on_open_audio_player(self):
        """Launch audio player in separate process (process-level isolation)."""
        if not self.engine.midi_file:
            print("Audio Player: No MIDI file loaded")
            return

        # Import multiprocessing
        import multiprocessing

        try:
            # Import launcher module
            from ui.audio_player_launcher import launch_audio_player_process

            # Spawn separate process for audio player
            # This provides true OS-level isolation:
            # - Separate memory space
            # - Independent FluidSynth instance
            # - Separate audio device handles
            # - No shared state with main process
            self.audio_player_process = multiprocessing.Process(
                target=launch_audio_player_process,
                args=(self.engine.midi_file.filename, self.sf2_path),
                daemon=True  # Clean up automatically when main process exits
            )
            self.audio_player_process.start()
        except Exception as e:
            print(f"Audio Player: Failed to launch process: {e}")
            self.audio_player_process = None
            
    def on_revert_beat_track(self):
        """Revert beat track and return to standard mode."""
        if not hasattr(self.engine, 'session_state'):
            return
            
        # Stop playback if active
        was_playing = self.engine.is_playing
        was_paused = self.engine.is_paused
        if was_playing or was_paused:
            self.stop()
            
        # Clear beat track
        self.beat_hint.clear_beat_track()
        
        # Update session state
        self.engine.session_state.has_beat_track = False
        self.engine.session_state.mode = PlaybackMode.STANDARD
        
        # Update UI
        self.playback_controls.update_playback_mode(self.engine)
    
    def on_revert_anacrusis(self):
        """Revert anacrusis correction by resetting offset to zero."""
        if not hasattr(self.engine, 'session_state'):
            return
            
        print("[GUI] Reverting timing correction...")
        
        # Reset values in session state
        if hasattr(self.engine, 'session_state'):
            self.engine.session_state.has_anacrusis_correction = False
            self.engine.session_state.anacrusis_offset_beats = 0.0
            self.engine.session_state.timing_correction = 0.0
            self.engine.session_state.fine_adjustment = 0.0
        
        # Update EventScheduler's offset to zero
        if self.engine.scheduler:
            self.engine.scheduler.set_timing_offset(0.0)
            print("  - EventScheduler offset reset to 0.0 beats")

        self.playback_controls.timing_correction_label.setVisible(False)
        self.playback_controls.fine_adjustment_frame.setVisible(False)

        # If playing, restart playback to apply the reset
        if self.engine.is_playing:
            print("  - Stopping playback and remove offset...")
            self.engine.stop()
            
        # Reset fine adjustment UI state completely
        self.playback_controls.reset_fine_adjustment()
        
        # Update UI
        self.playback_controls.update_playback_mode(self.engine)
        
        print(f"[GUI] Timing correction reverted. All timing offsets reset to zero.")
    
    def on_apply_anacrusis_correction(self, detection_result: dict):
        """Apply timing correction from detection result
        
        Args:
            detection_result: Dictionary containing detection results from anacrusis detector
        """
        if not hasattr(self.engine, 'session_state'):
            return
            
        print("[GUI] Applying anacrusis correction...")
        
        # Extract timing correction from detection result
        # The detection result contains first_downbeat_time and other info        
        # Calculate timing correction using the grid-snapping logic from midi_engine
        timing_correction_beats = self.engine.compute_beat_offset(detection_result)[0]
        
        # Store in session state
        self.engine.session_state.has_anacrusis_correction = True
        self.engine.session_state.timing_correction = timing_correction_beats
        self.engine.session_state.fine_adjustment = 0.0  # Start with no fine adjustment
        
        # Calculate total offset for scheduler (timing correction + fine adjustment)
        total_offset_beats = timing_correction_beats + 0.0
        
        # Update EventScheduler's offset
        if self.engine.scheduler:
            self.engine.scheduler.set_timing_offset(total_offset_beats)
            print(f"  - EventScheduler offset set to {total_offset_beats:.2f} beats")
        
        # Calculate timing offset in seconds for display
        if hasattr(self.engine, 'tempo') and self.engine.tempo > 0:
            seconds_per_beat = 60.0 / self.engine.tempo
            offset_seconds = total_offset_beats * seconds_per_beat
            self.engine.session_state.timing_offset_sec = offset_seconds
            print(f"  - Timing offset: {offset_seconds:.3f}s")
        
        # Update UI to show timing correction
        self.playback_controls.update_anacrusis_display(
            timing_correction_beats, 0.0  # timing_correction, fine_adjustment
        )
        
        # If playing, stop the playback
        if self.engine.is_playing:
            print("  - Stopping playback to apply timing correction...")
            self.engine.stop()
            
        # Update UI
        self.playback_controls.update_playback_mode(self.engine)
        
        print(f"[GUI] Timing correction applied: {timing_correction_beats:+.2f} beats")
    
    def on_fine_adjustment_changed(self, adjustment_value):
        """Handle fine adjustment changes - Only called when user clicks confirm button
        
        Args:
            adjustment_value: Adjustment value in beats (-3.0 to +3.0)
        """
        if not hasattr(self.engine, 'session_state'):
            return
            
        # Update the fine adjustment value in the session state
        self.engine.session_state.fine_adjustment = adjustment_value
        
        # Calculate total effective timing correction
        base_correction = self.engine.session_state.timing_correction
        effective_correction = base_correction + adjustment_value
        
        # IMPORTANT: Update the anacrusis_offset_beats with the effective total
        # This is what's actually used by the playback engine
        self.engine.session_state.anacrusis_offset_beats = effective_correction
        
        # Update the label to show the confirmed value
        self.playback_controls.update_timing_correction_label(
            base_correction, adjustment_value
        )
        
        # Calculate and update timing offset in seconds
        if hasattr(self.engine, 'tempo') and self.engine.tempo > 0:
            seconds_per_beat = 60.0 / self.engine.tempo
            offset_seconds = effective_correction * seconds_per_beat
            self.engine.session_state.timing_offset_sec = offset_seconds


            print(f"[GUI]  - New timing offset: {offset_seconds:.3f}s")
        
        # If the player is running, restart it to apply the new timing offset
        was_playing = False
        if self.engine.is_playing:
            was_playing = True
            self.stop()
            
        # Resume playback after a brief delay
        if was_playing:
            QTimer.singleShot(100, self.play)

    def update_section_info_label(self):
        """Update section info label for the pending section"""
        if hasattr(self, 'sections') and self.sections:
            self.playback_controls.update_section_info_label(self.sections)

    def confirm_and_play_section(self):
        """Confirm the selected section and start playback."""
        if not hasattr(self, 'sections') or not self.sections:
            return
        
        selected_index = self.playback_controls.get_selected_section_index()
        pending_index = self.playback_controls.get_pending_section_index()
        
        if selected_index != pending_index:
            # User wants to switch to a different section
            old_section = self.sections[selected_index]
            new_section = self.sections[pending_index]
            old_label = old_section.get('label', f'Section {selected_index + 1}')
            new_label = new_section.get('label', f'Section {pending_index + 1}')            
            # Stop current playback if it's active
            if self.engine.is_playing or self.engine.is_paused:
                self.stop()
            
            self.sync_cadence_with_current_section()            
            self.playback_controls.update_current_section_status(self.sections)
            self.play()
            
        else:
            # Stop current playback if it's active (restart)
            if self.engine.is_playing or self.engine.is_paused:
                self.stop()
            
            # Ensure cadence is synchronized with current section
            self.sync_cadence_with_current_section()
            self.playback_controls.update_current_section_status(self.sections)
            self.play()
