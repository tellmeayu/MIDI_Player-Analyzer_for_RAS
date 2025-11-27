from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QFileDialog, QComboBox,
    QToolTip, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication
import os

from core.playback_mode import PlaybackMode

class PlaybackControls(QGroupBox):
    """Widget for playback controls including buttons, section selection, and volume control"""
    
    # Signals for communication with main window
    file_loaded = pyqtSignal(str)  # Emitted when file is loaded
    play_requested = pyqtSignal()
    pause_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    midi_volume_changed = pyqtSignal(int)
    
    # New signals for section control
    section_selected = pyqtSignal(int)  # Emitted when section dropdown changes
    section_confirm_requested = pyqtSignal()  # Emitted when confirm button is clicked
    
    # Signals for analysis features
    generate_beat_track_requested = pyqtSignal()  # Emitted when beat track generation is requested
    revert_beat_track_requested = pyqtSignal()  # Emitted when beat track reversion is requested
    revert_anacrusis_requested = pyqtSignal()  # Emitted when anacrusis correction reversion is requested
    fine_adjustment_changed = pyqtSignal(float)  # Emitted when fine adjustment value changes
    audio_player_requested = pyqtSignal()  # Emitted when audio player window is requested (Phase 1F)
    
    def __init__(self, parent=None):
        """Initialize playback controls widget
        
        Args:
            parent: Parent widget
        """
        super().__init__("Playback Controls", parent)
        
        # Section state management
        self.selected_section_index = 0
        self.pending_section_index = 0
        
        # Fine adjustment state
        self.current_base_pickup_beats = 0.0  # Store the base pickup beats for preview
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the playback controls UI"""
        layout = QVBoxLayout()
        layout.setSpacing(8)  # Reduced from 15 to 8 - smaller vertical spacing between sections
        layout.setContentsMargins(15, 15, 15, 20)  # Reduced top and bottom margins from (15, 20, 15, 25)
        
        # --- SECTION SELECTION UI ---
        section_layout = QHBoxLayout()
        section_label = QLabel("Selected:")
        self.section_combo = QComboBox()
        self.section_combo.setMinimumWidth(220)
        self.section_combo.currentIndexChanged.connect(self.on_section_selected)
        self.section_info_label = QLabel("")
        self.section_info_label.setMinimumWidth(200)
        
        # Add confirm and play button for section selection
        self.section_confirm_button = QPushButton("Confirm and Play")
        self.section_confirm_button.clicked.connect(self.on_section_confirm_clicked)
        self.section_confirm_button.setMinimumWidth(120)
        self.section_confirm_button.setToolTip("Stop current playback and start playing the selected section")
        self.section_confirm_button.setStyleSheet("background-color: #4CAF50; font-weight: bold; color: white;")
        
        section_layout.addWidget(section_label)
        section_layout.addWidget(self.section_combo)
        section_layout.addWidget(self.section_info_label)
        section_layout.addWidget(self.section_confirm_button)
        section_layout.addStretch(1)
        layout.addLayout(section_layout)
        
        # --- PLAYBACK CONTROLS ---
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(7)  # Reduced from 9 to 7 - smaller spacing between element groups
        
        # Playback buttons group
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(5)  # Reduced from 7 to 5 - smaller spacing between buttons
        
        self.load_button = QPushButton("Open")
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        
        # Make buttons slightly smaller to save space
        for button in [self.load_button, self.play_button, self.pause_button, self.stop_button]:
            button.setMinimumHeight(28)  # Reduced from 32 to 28 - shorter buttons
        
        # Connect button signals
        self.load_button.clicked.connect(self.load_file)
        self.play_button.clicked.connect(self.play_requested.emit)
        self.pause_button.clicked.connect(self.pause_requested.emit)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        
        buttons_layout.addWidget(self.load_button)
        buttons_layout.addWidget(self.play_button)
        buttons_layout.addWidget(self.pause_button)
        buttons_layout.addWidget(self.stop_button)
        
        # Add buttons group to main controls layout
        controls_layout.addLayout(buttons_layout)

        # --- Sectional Beat, Bar, Tempo ---
        # create a layout for the sectional labels
        sectional_labels_layout = QHBoxLayout()
        sectional_labels_layout.setSpacing(5)  # Reduced spacing between labels
        
        # Add small spacing after buttons
        sectional_labels_layout.addSpacing(5)

        self.section_beat_label = QLabel("Beat: -")
        self.section_beat_label.setFixedWidth(66)
        self.section_beat_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.section_beat_label.setStyleSheet("color: #2E7D32; font-weight: bold;")
        sectional_labels_layout.addWidget(self.section_beat_label)

        # Add consistent spacing between labels
        sectional_labels_layout.addSpacing(4)

        self.section_bar_progress_label = QLabel("Bar: -/-")
        self.section_bar_progress_label.setFixedWidth(120)
        self.section_bar_progress_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.section_bar_progress_label.setStyleSheet("color: #2E7D32; font-weight: bold;")
        sectional_labels_layout.addWidget(self.section_bar_progress_label)

        # Add consistent spacing between labels
        sectional_labels_layout.addSpacing(9)

        # Add total beats label for DYNAMIC mode (initially hidden)
        self.total_beats_label = QLabel("Total: -")
        self.total_beats_label.setFixedWidth(70)
        self.total_beats_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.total_beats_label.setStyleSheet("color: #1565C0;")
        self.total_beats_label.setVisible(False)  # Initially hidden
        sectional_labels_layout.addWidget(self.total_beats_label)
        
        # Add estimated tempo label for DYNAMIC mode (initially hidden)
        self.est_tempo_label = QLabel("Est: - BPM")
        self.est_tempo_label.setFixedWidth(90)
        self.est_tempo_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.est_tempo_label.setStyleSheet("color: #1565C0;")
        self.est_tempo_label.setVisible(False)  # Initially hidden
        sectional_labels_layout.addWidget(self.est_tempo_label)

        # Add consistent spacing between labels
        sectional_labels_layout.addSpacing(4)

        self.section_tempo_label = QLabel("RAS cue: - ")
        self.section_tempo_label.setFixedWidth(130)
        self.section_tempo_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.section_tempo_label.setStyleSheet("color: #2E7D32; font-weight: bold;")
        sectional_labels_layout.addWidget(self.section_tempo_label)

        # Add the sectional labels layout to the main controls layout with adjusted spacing
        controls_layout.addLayout(sectional_labels_layout)
        
        controls_layout.addStretch()  # Add stretch to push the sectional labels to the left and create space before the MIDI volume control
        # MIDI music volume control
        volume_label = QLabel("MIDI Volume:")
        volume_label.setMinimumHeight(28)  # Reduced from 32 to match button height
        self.midi_volume_slider = QSlider(Qt.Horizontal)
        self.midi_volume_slider.setRange(0, 100)
        self.midi_volume_slider.setValue(35)  # Default to 35% volume
        self.midi_volume_slider.valueChanged.connect(self.on_midi_volume_changed)
        self.midi_volume_slider.setMinimumWidth(120)  # Increased minimum width
        self.midi_volume_slider.setMaximumWidth(200)  # Increased maximum width
        self.midi_volume_slider.setToolTip(
            "This controls the FluidSynth internal gain.\n" +
            "For best audio quality, keep this moderate (30-60%)\n" +
            "and use your computer's volume control for overall loudness.\n\n" +
            "Higher values may cause distortion."
        )
        
        self.midi_volume_label = QLabel("35%")
        self.midi_volume_label.setMinimumWidth(40)  # Fixed width for percentage
        
        # Adjusting the layout to add space between the volume control and the previous labels
        volume_control_layout = QHBoxLayout()
        volume_control_layout.addWidget(volume_label)
        volume_control_layout.addWidget(self.midi_volume_slider)
        volume_control_layout.addWidget(self.midi_volume_label)
        volume_control_layout.addStretch()  # Adding stretch to push the volume control to the right
        
        controls_layout.addLayout(volume_control_layout)
        
        # Add the controls layout to the main layout
        layout.addLayout(controls_layout)
        
        # --- BEAT REGULARITY DISPLAY (moved to analysis controls row) ---
        self.regularity_frame = QFrame()
        self.regularity_frame.setFrameShape(QFrame.StyledPanel)
        self.regularity_frame.setFrameShadow(QFrame.Raised)
        self.regularity_frame.setStyleSheet("background-color: #F5F5F5; border-radius: 4px; padding: 4px;")
        self.regularity_frame.setVisible(False)
        
        regularity_frame_layout = QVBoxLayout(self.regularity_frame)
        regularity_frame_layout.setContentsMargins(8, 6, 8, 6)
        regularity_frame_layout.setSpacing(4)
        
        # First row: Title and status
        regularity_title_row = QHBoxLayout()
        regularity_title_row.setSpacing(10)
        
        self.regularity_title_label = QLabel("Beat Regularity:")
        self.regularity_title_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        regularity_title_row.addWidget(self.regularity_title_label)
        
        self.regularity_status_label = QLabel("Status: -")
        self.regularity_status_label.setStyleSheet("font-size: 11px;")
        regularity_title_row.addWidget(self.regularity_status_label)
        regularity_title_row.addStretch()
        
        regularity_frame_layout.addLayout(regularity_title_row)
        
        # Second row: Detailed statistics
        self.regularity_details_label = QLabel("IBI CV: - · Std: - · Mean IBI: -")
        self.regularity_details_label.setStyleSheet("font-size: 10px; color: #555;")
        self.regularity_details_label.setWordWrap(True)
        regularity_frame_layout.addWidget(self.regularity_details_label)

        # --- CURRENT PLAYING SECTION STATUS (separate row) ---
        current_section_layout = QHBoxLayout()
        current_section_layout.setSpacing(5)  # Add some spacing control
        self.current_section_status_label = QLabel("Currently Playing: -")
        self.current_section_status_label.setStyleSheet(
            "font-weight: bold; color: #2E7D32; background-color: #E8F5E8; "
            "padding: 4px 10px; border-radius: 4px; font-size: 12px; "  # Reduced padding from 6px 12px to 4px 10px
        )
        self.current_section_status_label.setAlignment(Qt.AlignLeft)
        current_section_layout.addWidget(self.current_section_status_label)
        current_section_layout.addStretch(1)
        layout.addLayout(current_section_layout)
        
        # --- ANALYSIS CONTROLS ---
        analysis_layout = QHBoxLayout()
        analysis_layout.setSpacing(7)
        
        # Beat tracking button (for DYNAMIC mode)
        self.beat_track_button = QPushButton("Generate Beat Track")
        self.beat_track_button.setStyleSheet(
            "background-color: #FFC107; font-weight: bold; padding: 4px 10px;"
        )
        self.beat_track_button.setToolTip("Generate beat track for performance MIDI (no tempo metadata)")
        self.beat_track_button.clicked.connect(self.generate_beat_track_requested.emit)
        self.beat_track_button.setVisible(False)  # Initially hidden
        
        # Beat track revert button
        self.revert_beat_track_button = QPushButton("Clear Beat Track")
        self.revert_beat_track_button.setStyleSheet(
            "background-color: #F44336; color: white; font-weight: bold; padding: 4px 10px;"
        )
        self.revert_beat_track_button.setToolTip("Clear the beat track and return to standard mode")
        self.revert_beat_track_button.clicked.connect(self.revert_beat_track_requested.emit)
        self.revert_beat_track_button.setVisible(False)  # Initially hidden

        # Audio player button (for DYNAMIC mode after beat tracking, Phase 1F)
        self.audio_player_button = QPushButton("Open Audio Player")
        self.audio_player_button.setStyleSheet(
            "background-color: #9C27B0; color: white; font-weight: bold; padding: 4px 10px;"
        )
        self.audio_player_button.setToolTip("Open audio-based player with cadence control and time-stretching")
        self.audio_player_button.clicked.connect(self.audio_player_requested.emit)
        self.audio_player_button.setVisible(False)  # Initially hidden
        
        anacrusis_info_layout = QHBoxLayout()
        self.timing_correction_label = QLabel("Correction: +0.00 beats")
        self.timing_correction_label.setStyleSheet(
            "background-color: #B2DFDB; color: #004D40; font-weight: bold; padding: 3px 8px; border-radius: 4px; margin: 2px;"
        )
        self.timing_correction_label.setMinimumWidth(160)
        self.timing_correction_label.setFixedHeight(32)
        self.timing_correction_label.setVisible(False)
        self.timing_correction_label.setToolTip("Timing Correction: calculated offset to align MIDI with performance.")

        anacrusis_info_layout.addWidget(self.timing_correction_label)
        anacrusis_info_layout.addStretch()

        analysis_layout.addLayout(anacrusis_info_layout)

        # Anacrusis revert button
        self.revert_anacrusis_button = QPushButton("Reset Timing")
        self.revert_anacrusis_button.setStyleSheet(
            "background-color: #F44336; color: white; font-weight: bold; padding: 4px 10px;"
        )
        self.revert_anacrusis_button.setToolTip("Reset metronome timing offset")
        self.revert_anacrusis_button.clicked.connect(self.revert_anacrusis_requested.emit)
        self.revert_anacrusis_button.setVisible(False)  # Initially hidden
        
        # Fine adjustment controls - original clean design
        self.fine_adjustment_frame = QFrame()
        self.fine_adjustment_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.fine_adjustment_frame.setStyleSheet("background-color: #E0F2F1; border-radius: 4px; margin: 2px;")
        self.fine_adjustment_frame.setVisible(False)  # Initially hidden
        self.fine_adjustment_frame.setToolTip(
            "Fine-tune the timing correction for precise MIDI alignment.\n"
            "• + values delay MIDI events\n"
            "• - values advance MIDI events\n"
            "• Click 'Apply' to confirm, or 'Reset' to return to auto-detected timing"
        )
        
        # Simple single-row layout
        fine_adjustment_layout = QHBoxLayout(self.fine_adjustment_frame)
        fine_adjustment_layout.setContentsMargins(5, 3, 5, 3)
        fine_adjustment_layout.setSpacing(3)
        
        # Label
        fine_adjustment_label = QLabel("Adjust:")
        fine_adjustment_label.setStyleSheet("font-size: 10px; background-color: transparent; font-weight: bold;")
        fine_adjustment_label.setFixedHeight(26)
        
        # Create adjustment buttons
        self.fine_minus_quarter = QPushButton("-¼")
        self.fine_minus_quarter.setFixedSize(30, 26)
        self.fine_minus_quarter.clicked.connect(lambda: self.apply_fine_adjustment(-0.25))
        self.fine_minus_quarter.setToolTip("Move ¼ beat earlier")
        self.fine_minus_quarter.setStyleSheet("padding: 1px; font-size: 10px; background-color: #FFCDD2;")
        
        self.fine_minus_eighth = QPushButton("-⅛")
        self.fine_minus_eighth.setFixedSize(30, 26)
        self.fine_minus_eighth.clicked.connect(lambda: self.apply_fine_adjustment(-0.125))
        self.fine_minus_eighth.setToolTip("Move ⅛ beat earlier")
        self.fine_minus_eighth.setStyleSheet("padding: 1px; font-size: 10px; background-color: #FFE0B2;")

        self.fine_minus_sixteenth = QPushButton("-¹/₁₆")
        self.fine_minus_sixteenth.setFixedSize(30, 26)
        self.fine_minus_sixteenth.clicked.connect(lambda: self.apply_fine_adjustment(-0.0625))
        self.fine_minus_sixteenth.setToolTip("Move ¹/₁₆ beat earlier")
        self.fine_minus_sixteenth.setStyleSheet("padding: 1px; font-size: 10px; background-color: #FFF9C4;")
        
        # Value display
        self.fine_value_label = QLabel("0.000")
        self.fine_value_label.setFixedSize(40, 26)
        self.fine_value_label.setAlignment(Qt.AlignCenter)
        self.fine_value_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        
        # Positive adjustment buttons
        self.fine_plus_sixteenth = QPushButton("+¹/₁₆")
        self.fine_plus_sixteenth.setFixedSize(30, 26)
        self.fine_plus_sixteenth.clicked.connect(lambda: self.apply_fine_adjustment(0.0625))
        self.fine_plus_sixteenth.setToolTip("Move ¹/₁₆ beat later")
        self.fine_plus_sixteenth.setStyleSheet("padding: 1px; font-size: 10px; background-color: #F1F8E9;")

        self.fine_plus_eighth = QPushButton("+⅛")
        self.fine_plus_eighth.setFixedSize(30, 26)
        self.fine_plus_eighth.clicked.connect(lambda: self.apply_fine_adjustment(0.125))
        self.fine_plus_eighth.setToolTip("Move ⅛ beat later")
        self.fine_plus_eighth.setStyleSheet("padding: 1px; font-size: 10px; background-color: #C8E6C9;")
        
        self.fine_plus_quarter = QPushButton("+¼")
        self.fine_plus_quarter.setFixedSize(30, 26)
        self.fine_plus_quarter.clicked.connect(lambda: self.apply_fine_adjustment(0.25))
        self.fine_plus_quarter.setToolTip("Move ¼ beat later")
        self.fine_plus_quarter.setStyleSheet("padding: 1px; font-size: 10px; background-color: #B2DFDB;")
        
        # Reset button
        self.fine_reset = QPushButton("Reset")
        self.fine_reset.setFixedSize(40, 26)
        self.fine_reset.clicked.connect(self.reset_fine_adjustment)
        self.fine_reset.setToolTip("Reset all adjustment")
        self.fine_reset.setStyleSheet("padding: 1px; font-size: 10px; background-color: #E0E0E0;")
        
        # Apply button
        self.fine_confirm_button = QPushButton("Apply")
        self.fine_confirm_button.setFixedSize(45, 26)
        self.fine_confirm_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 1px; font-size: 10px;")
        self.fine_confirm_button.setToolTip("Apply the adjustment value to the beats")
        self.fine_confirm_button.clicked.connect(self.confirm_fine_adjustment)
        
        # Add all adjustment buttons to a single row layout
        fine_adjustment_layout.addWidget(fine_adjustment_label)
        fine_adjustment_layout.addWidget(self.fine_minus_quarter)
        fine_adjustment_layout.addWidget(self.fine_minus_eighth)
        fine_adjustment_layout.addWidget(self.fine_minus_sixteenth)
        fine_adjustment_layout.addWidget(self.fine_value_label)
        fine_adjustment_layout.addWidget(self.fine_plus_sixteenth)
        fine_adjustment_layout.addWidget(self.fine_plus_eighth)
        fine_adjustment_layout.addWidget(self.fine_plus_quarter)
        fine_adjustment_layout.addWidget(self.fine_reset)
        fine_adjustment_layout.addWidget(self.fine_confirm_button)
        fine_adjustment_layout.addStretch(1)
        
        analysis_layout.addWidget(self.beat_track_button)
        analysis_layout.addWidget(self.revert_beat_track_button)
        analysis_layout.addWidget(self.audio_player_button)
        analysis_layout.addWidget(self.revert_anacrusis_button)
        analysis_layout.addWidget(self.fine_adjustment_frame)
        
        # Add spacing before regularity frame
        analysis_layout.addSpacing(15)
        
        # Add regularity frame to the same row as other analysis controls
        analysis_layout.addWidget(self.regularity_frame)
        analysis_layout.addStretch(1)
        
        layout.addLayout(analysis_layout)
        
        # Reduce stretch space at bottom
        layout.addStretch(0)  # Reduced from 1 to 0 to minimize extra space
        
        self.setLayout(layout)
    
    def load_file(self):
        """Load MIDI file using file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select MIDI File", "", "MIDI Files (*.mid *.midi)"
        )
        
        if file_path:
            # print(f"PlaybackControls: File selected: {os.path.basename(file_path)}")
            self.file_loaded.emit(file_path)
    
    def on_midi_volume_changed(self, volume):
        """Handle MIDI volume slider change
        
        Args:
            volume: Volume value (0-100)
        """
        self.midi_volume_label.setText(f"{volume}%")
        self.midi_volume_changed.emit(volume)
    
    def set_midi_volume(self, volume):
        """Set MIDI volume slider value
        
        Args:
            volume: Volume value (0-100)
        """
        self.midi_volume_slider.setValue(volume)
        self.midi_volume_label.setText(f"{volume}%") 

    def update_section_progress(self, current_measure, total_measures):
        """Update section progress display"""
        self.section_bar_progress_label.setText(f"Progress: {current_measure}/{total_measures}")
    
    def update_section_beat(self, current_beat, beats_per_measure, is_strong_beat=None):
        """Update section beat display"""
        # use same style as beat position in metadata panel
        if current_beat == '-' or beats_per_measure == '-':
            self.section_beat_label.setText("Beat: -")
            return
        # If is_strong_beat is not provided, default to strong for beat 1
        if is_strong_beat is None:
            is_strong_beat = (current_beat == 1)
        indicator = "●" if is_strong_beat else "○"
        self.section_beat_label.setText(f"Beat:  {current_beat} {indicator}")
        
    def update_total_beats(self, total_beats):
        """Update total beats display for DYNAMIC mode
        
        Args:
            total_beats: Total number of beats in the track
        """
        if total_beats <= 0:
            self.total_beats_label.setText("Total: -")
        else:
            self.total_beats_label.setText(f"Total: {total_beats}")
            
    def update_estimated_tempo(self, tempo):
        """Update estimated tempo display for DYNAMIC mode
        
        Args:
            tempo: Estimated tempo in BPM
        """
        if tempo <= 0:
            self.est_tempo_label.setText("Est: - BPM")
        else:
            self.est_tempo_label.setText(f"Est: {tempo:.1f} BPM")
            
    def update_beat_regularity(self, regularity_data):
        """Update beat regularity display
        
        Args:
            regularity_data: Dictionary with cv_ibi, std_ibi, mean_ibi
        """
        if regularity_data is None:
            self.regularity_status_label.setText("Status: -")
            self.regularity_details_label.setText("IBI CV: - · Std: -  · Mean IBI: -")
            self.regularity_frame.setStyleSheet("background-color: #F5F5F5; border-radius: 4px; padding: 4px;")
            return
            
        cv = regularity_data.get('cv_ibi', 0)
        std_s = regularity_data.get('std_ibi', 0) 
        mean_s = regularity_data.get('mean_ibi', 0) 

        # Calculate BPM from mean IBI
        bpm = 60.0 / mean_s if mean_s > 0 else 0
        
        # Format details string
        details = f"IBI CV: {cv:.3f} · Std: {std_s:.3f} s · Mean IBI: {mean_s:.3f} s"
        self.regularity_details_label.setText(details)
        
        # Set status and color based on CV
        if cv <= 0.02:
            status = "Stable — Suitable for RAS training"
            color = "#E8F5E9"  # Light green
            text_color = "#2E7D32"  # Dark green
        elif cv <= 0.04:
            status = "Marginal — Use with caution for RAS"
            color = "#FFF8E1"  # Light amber
            text_color = "#F57F17"  # Dark amber
        else:
            status = "Irregular — NOT recommended for RAS"
            color = "#FFEBEE"  # Light red
            text_color = "#C62828"  # Dark red
            
        self.regularity_status_label.setText(f"Status: {status}")
        self.regularity_status_label.setStyleSheet(f"color: {text_color}; font-weight: bold;")
        self.regularity_frame.setStyleSheet(f"background-color: {color}; border-radius: 4px; padding: 4px;")

    def update_section_tempo(self, music_tempo):
        """Update section tempo display to show current musical tempo (cadence)"""
        if music_tempo is None or music_tempo == '-':
            self.section_tempo_label.setText("RAS cue: - ")
        else:
            self.section_tempo_label.setText(f"RAS cue: {music_tempo} BPM")  # Use musical note symbol to indicate musical tempo

    # --- SECTION CONTROL METHODS --- 
    def populate_sections(self, sections):
        """Populate section combo box with available sections
        
        Args:
            sections: List of section dictionaries
        """
        self.section_combo.clear()
        for section in sections:
            label = section.get('label', f"Section {section.get('index', 0)+1}")
            ts = section.get('time_signature', {}).get('text', '')
            tempo = section.get('tempo', '')
            measures = section.get('measures', '')
            tempo_int = int(round(tempo)) if tempo != '' else tempo
            self.section_combo.addItem(f"{label}: {ts}, {tempo_int} BPM, {measures} bars")
        
        # Reset indices
        self.selected_section_index = 0
        self.pending_section_index = 0
        self.section_combo.setCurrentIndex(0)
        
        # Update current section status display
        self.update_current_section_status(sections)
        # Reset confirm button to normal state
        self.section_confirm_button.setText("Confirm and Play")
        self.section_confirm_button.setStyleSheet("background-color: #4CAF50; font-weight: bold; color: white;")
        self.section_confirm_button.setToolTip("Start playing the current section")
    
    def update_section_info_label(self, sections):
        """Update section info label for the pending section
            Args:
                sections: List of section dictionaries
        """
        if sections and 0 <= self.pending_section_index < len(sections):
            section = sections[self.pending_section_index]
            label = section.get('label', '')
            ts = section.get('time_signature', {}).get('text', '')
            ts_html = f'<span style="color: #2E7D32; font-weight: bold;">{ts}</span>'  # highlight time signature
            self.section_info_label.setTextFormat(Qt.RichText)
            self.section_info_label.setText(f"{label}: {ts_html}")
        else:
            self.section_info_label.setText("")
    
    def on_section_selected(self, idx):
        """Handle section dropdown selection change"""
        self.pending_section_index = idx
        print(f"Section preview: {idx}")
        
        # Emit signal to main window for info label update
        self.section_selected.emit(idx)
        
        # Update button appearance based on whether this is a change
        if self.selected_section_index != self.pending_section_index:
            # Section changed - highlight confirm button
            self.section_confirm_button.setText("Confirm and Play ✓")
            self.section_confirm_button.setStyleSheet("background-color: #FF9800; font-weight: bold; color: white;")
            self.section_confirm_button.setToolTip(f"Stop current playback and switch to Section {idx + 1}")
        else:
            # Same section - normal button appearance
            self.section_confirm_button.setText("Confirm and Play")
            self.section_confirm_button.setStyleSheet("background-color: #4CAF50; font-weight: bold; color: white;")
            self.section_confirm_button.setToolTip("Start playing the current section")
    
    def on_section_confirm_clicked(self):
        """Handle section confirm button click"""
        # Update selected index
        self.selected_section_index = self.pending_section_index
        
        # Reset button appearance
        self.section_confirm_button.setText("Confirm and Play")
        self.section_confirm_button.setStyleSheet("background-color: #4CAF50; font-weight: bold; color: white;")
        self.section_confirm_button.setToolTip("Start playing the current section")
        
        # Emit signal to main window
        self.section_confirm_requested.emit()
    
    def get_selected_section_index(self):
        """Get the currently selected section index"""
        return self.selected_section_index
    
    def get_pending_section_index(self):
        """Get the pending section index"""
        return self.pending_section_index 

    def update_current_section_status(self, sections):
        """Update current section status label
        
        Args:
            sections: List of section dictionaries
        """
        if not sections or self.selected_section_index >= len(sections):
            self.current_section_status_label.setText("Currently Playing: -")
            return
            
        current_section = sections[self.selected_section_index]
        label = current_section.get('label', f'Section {self.selected_section_index + 1}')
        ts = current_section.get('time_signature', {}).get('text', '')
        tempo = current_section.get('tempo', '')
        measures = current_section.get('measures', '')
        
        tempo_int = int(round(tempo)) if tempo != '' else tempo
        status_text = f"Currently Playing: {label}"
        self.current_section_status_label.setText(status_text)
        
    def apply_fine_adjustment(self, adjustment_value, is_reset=False):
        """Apply a fine adjustment value to the UI display and preview it immediately
        
        Args:
            adjustment_value: Value to adjust by or set to (if is_reset is True)
            is_reset: If True, sets the value directly, otherwise adds to current value
        """
        try:
            current_value = float(self.fine_value_label.text())
        except ValueError:
            current_value = 0.0
        
        if is_reset:
            new_value = adjustment_value
        else:
            new_value = current_value + adjustment_value
            
        # Limit adjustment range to -3.0 to +3.0 beats
        new_value = max(-3.0, min(3.0, new_value))
        
        # Update display with 3 decimal precision
        self.fine_value_label.setText(f"{new_value:.3f}")
        
        print(f"Fine adjustment: {new_value:.3f}")  # For debugging
        
    def confirm_fine_adjustment(self):
        """Confirm and apply fine adjustment value"""
        # Get the current displayed value
        try:
            adjustment_value = float(self.fine_value_label.text())
        except ValueError:
            adjustment_value = 0.0
            
        # Send signal with current displayed adjustment value
        print(f"Fine adjustment confirmed: {adjustment_value:.2f}")
        self.fine_adjustment_changed.emit(adjustment_value)
        
        # Store the original style and text for later restoration
        original_style = self.fine_confirm_button.styleSheet()
        original_text = self.fine_confirm_button.text()
        
        # Just briefly change the background color for feedback
        self.fine_confirm_button.setStyleSheet("background-color: #8BC34A; color: white; font-weight: bold; padding: 1px; font-size: 10px;")
        
        # Restore original button appearance after a short time
        QTimer.singleShot(1000, lambda: self.fine_confirm_button.setStyleSheet(original_style))

    def update_anacrusis_display(self, timing_correction: float, fine_adjustment: float):
        """
        Updates timing correction display.
        
        Args:
            timing_correction: The calculated fractional timing correction.
            fine_adjustment: The user-applied fine adjustment.
        """
        # Update timing correction label (showing base correction plus user's adjustment)
        total_correction = timing_correction + fine_adjustment
        self.timing_correction_label.setText(f"Correction: {total_correction:+.2f} beats")
        self.timing_correction_label.setVisible(True)

        # Update fine adjustment display
        self.fine_value_label.setText(f"{fine_adjustment:.2f}")
        self.fine_adjustment_frame.setVisible(True)

        QApplication.processEvents()

    def reset_fine_adjustment(self):
        """Reset fine adjustment to 0 and apply it immediately."""
        # Set the internal value to 0
        self.fine_adjustment_value = 0.0
        
        # Update the display
        self.fine_value_label.setText("0.000")
        
        # Immediately confirm the change
        self.confirm_fine_adjustment()
        
        print("Fine adjustment reset and applied.")
    
    def update_timing_correction_label(self, timing_correction, fine_adjustment=0.0):
        """Update the timing correction label with the applied timing correction information

        Args:
            timing_correction: Timing correction value applied
            fine_adjustment: Fine adjustment value in fractions of a beat, or None to keep current display
        """
        # If fine_adjustment is None, keep the current display value (don't overwrite user's preview)
        if fine_adjustment is None:
            # Don't update the display or text - just make sure widgets are visible
            self.timing_correction_label.setVisible(True)
            self.fine_adjustment_frame.setVisible(True)
            return

        # Update the display with the provided fine adjustment value
        self.fine_value_label.setText(f"{fine_adjustment:.2f}")

        # Show fine adjustment if non-zero
        if fine_adjustment != 0:
            total = timing_correction + fine_adjustment
            text = f"Correction: {timing_correction:+.2f} beats ({fine_adjustment:+.2f})"
        else:
            text = f"Correction: {timing_correction:+.2f} beats"

        self.timing_correction_label.setText(text)
        self.timing_correction_label.setVisible(True)
        self.fine_adjustment_frame.setVisible(True)
        # Force update
        QApplication.processEvents()
    
    def update_playback_mode(self, engine):
        """Update UI based on playback mode
        
        Args:
            engine: MIDI engine instance
        """
        if not hasattr(engine, 'session_state'):
            # Hide all analysis buttons if no session state
            self.beat_track_button.setVisible(False)
            self.revert_beat_track_button.setVisible(False)
            self.audio_player_button.setVisible(False)
            self.revert_anacrusis_button.setVisible(False)
            self.fine_adjustment_frame.setVisible(False)
            self.total_beats_label.setVisible(False)
            self.est_tempo_label.setVisible(False)
            self.section_beat_label.setVisible(True)
            self.section_bar_progress_label.setVisible(True)
            self.section_tempo_label.setVisible(True)  # Show RAS cue by default
            self.regularity_frame.setVisible(False)
            return
            
        # Get playback mode
        mode = engine.session_state.mode
        
        if mode == PlaybackMode.DYNAMIC:
            # Show beat track button, hide anacrusis buttons
            self.revert_anacrusis_button.setVisible(False)
            
            # Hide standard mode beat/bar indicators and RAS cue, show dynamic mode labels instead
            self.section_beat_label.setVisible(False)
            self.section_bar_progress_label.setVisible(False)
            self.section_tempo_label.setVisible(False)  # Hide RAS cue in DYNAMIC mode
            self.total_beats_label.setVisible(True)
            self.est_tempo_label.setVisible(True)
            
            # Update total beats and estimated tempo if available
            if engine.session_state.has_beat_track:
                if engine.session_state.beat_times is not None:
                    self.update_total_beats(len(engine.session_state.beat_times))
                if engine.session_state.estimated_tempo_bpm > 0:
                    self.update_estimated_tempo(engine.session_state.estimated_tempo_bpm)
            
            # Show regularity frame if we have beat track
            if engine.session_state.has_beat_track and engine.session_state.beat_regularity is not None:
                self.regularity_frame.setVisible(True)
                self.update_beat_regularity(engine.session_state.beat_regularity)
            else:
                self.regularity_frame.setVisible(False)
            
            # Show appropriate beat track buttons
            if engine.session_state.has_beat_track:
                # Beat track generated - show revert button and audio player button, hide generate button
                self.beat_track_button.setVisible(False)
                self.revert_beat_track_button.setVisible(True)
                self.revert_beat_track_button.setEnabled(True)
                self.audio_player_button.setVisible(True)
            else:
                # No beat track - show generate button, hide revert and audio player buttons
                self.beat_track_button.setVisible(True)
                self.beat_track_button.setEnabled(True)
                self.beat_track_button.setText("Generate Beat Track")
                self.revert_beat_track_button.setVisible(False)
                self.audio_player_button.setVisible(False)
                
        elif mode == PlaybackMode.STANDARD:
            # Hide beat track and audio player buttons (audio player is for DYNAMIC mode only)
            self.beat_track_button.setVisible(False)
            self.revert_beat_track_button.setVisible(False)
            self.audio_player_button.setVisible(False)
            
            # Show standard mode beat/bar indicators and RAS cue, hide dynamic mode labels and regularity
            self.section_beat_label.setVisible(True)
            self.total_beats_label.setVisible(False)
            self.est_tempo_label.setVisible(False)
            self.section_bar_progress_label.setVisible(True)
            self.section_tempo_label.setVisible(True)  # Show RAS cue in STANDARD mode
            self.regularity_frame.setVisible(False)
            
            # Anacrusis detection now handled by standalone tool from menu
            # Hide anacrusis button
            
            # Check for anacrusis correction in session state
            if hasattr(engine.session_state, 'has_anacrusis_correction') and engine.session_state.has_anacrusis_correction:
                # Anacrusis correction applied - show revert button and timing correction label
                self.revert_anacrusis_button.setVisible(True)
                self.revert_anacrusis_button.setEnabled(True)

                # Update timing correction label ONLY if the values have changed from session state
                # This prevents overwriting the user's preview when they're adjusting
                if hasattr(engine.session_state, 'timing_correction'):
                    # Check if we need to update based on session state fine adjustment
                    session_fine_adj = 0.0
                    if hasattr(engine.session_state, 'fine_adjustment'):
                        session_fine_adj = engine.session_state.fine_adjustment

                    # Only update if the current display doesn't match what's been CONFIRMED in session state
                    # This also avoids updating when the user is actively adjusting
                    try:
                        current_display_value = float(self.fine_value_label.text())
                    except:
                        current_display_value = 0.0

                    # Only call update_timing_correction_label if display shows a CONFIRMED value
                    # (i.e., matches session state), not a preview value
                    if current_display_value == session_fine_adj:
                        # Display matches confirmed value - it's safe to refresh
                        self.update_timing_correction_label(
                            engine.session_state.timing_correction,
                            session_fine_adj
                        )
                    else:
                        # User is previewing an adjustment - DON'T overwrite it
                        # Just ensure widgets are visible
                        self.timing_correction_label.setVisible(True)
                        self.fine_adjustment_frame.setVisible(True)
                else:
                    # No timing_correction attribute - hide controls
                    self.timing_correction_label.setVisible(False)
                    self.fine_adjustment_frame.setVisible(False)
            else:
                # No anacrusis correction - hide revert button and timing correction label
                self.revert_anacrusis_button.setVisible(False)
                self.timing_correction_label.setVisible(False)
                self.fine_adjustment_frame.setVisible(False)
        else:
            # Hide all analysis buttons for unknown mode
            self.beat_track_button.setVisible(False)
            self.revert_beat_track_button.setVisible(False)
            self.audio_player_button.setVisible(False)
            self.revert_anacrusis_button.setVisible(False)
            self.section_beat_label.setVisible(True)
            self.section_bar_progress_label.setVisible(True)
            self.section_tempo_label.setVisible(True)  # Show RAS cue in unknown mode
            self.regularity_frame.setVisible(False)

    def show_audio_player_button(self):
        """Show the Audio Player button after audio artifacts are cached (Phase 1F)."""
        self.audio_player_button.setVisible(True)
        # print("Playback Controls: Audio Player button shown")