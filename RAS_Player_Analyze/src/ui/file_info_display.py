from PyQt5.QtWidgets import QGroupBox, QGridLayout, QLabel
import os
from PyQt5.QtCore import Qt
# from core.precision_timer import PrecisionTimer

class FileInfoDisplay(QGroupBox):
    """Widget for displaying MIDI file metadata and playback information"""
    
    def __init__(self, parent=None):
        """Initialize file info display widget
        
        Args:
            parent: Parent widget
        """
        super().__init__("File Information (metadata)", parent)
        
        # Store metadata tempo info for use during playback
        self._metadata_tempo_info = ""
        self._current_engine = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the file information display UI"""
        layout = QGridLayout()
        
        # File information labels
        self.file_name_label = QLabel("<b>File:</b> Not loaded")
        self.file_title_label = QLabel("<b>Title:</b> -")  
        self.file_tracks_label = QLabel("<b>Tracks:</b> -")
        self.file_resolution_label = QLabel("<b>Resolution:</b> -")
        self.file_tempo_label = QLabel("<b>MIDI Tempo:</b> -")
        self.file_time_signature_label = QLabel("<b>Time Sig:</b> -")
        self.file_key_signature_label = QLabel("<b>Key:</b> -")
        self.file_length_label = QLabel("<b>Duration:</b> -")
        self.total_bars_label = QLabel("<b>Total bars:</b> -")
        
        
        # Playback mode warning banner (for DYNAMIC mode)
        self.playback_mode_banner = QLabel("")
        self.playback_mode_banner.setStyleSheet(
            "background-color: #FFF3CD; color: #856404; font-weight: bold; "
            "padding: 5px; border-radius: 3px; border: 1px solid #FFEEBA; "
            "font-size: 12px; margin: 5px 0px;"
        )
        self.playback_mode_banner.setWordWrap(True)
        self.playback_mode_banner.setAlignment(Qt.AlignCenter)
        self.playback_mode_banner.hide()  # Initially hidden

        self.file_title_label.setWordWrap(True)
        self.file_name_label.setWordWrap(True)

        # Layout the file information in a grid
        layout.addWidget(self.file_name_label, 0, 0, 1, 3)  
        layout.addWidget(self.file_title_label, 1, 0, 1, 3)
        layout.addWidget(self.file_time_signature_label, 2, 0, 1, 3)
        layout.addWidget(self.file_tempo_label, 3, 0, 1, 3)
        layout.addWidget(self.file_key_signature_label, 0, 3)
        layout.addWidget(self.file_resolution_label, 1, 3)
        layout.addWidget(self.file_length_label, 2, 3)
        layout.addWidget(self.total_bars_label, 3, 3)
        
        
        # Add playback mode banner spanning both columns
        layout.addWidget(self.playback_mode_banner, 6, 0, 1, 4)

        self.setLayout(layout)
    
    def update_metadata(self, engine):
        """Update metadata display with information from engine
        
        Args:
            engine: MIDI engine instance
        """
        if not engine.midi_file:
            self.clear_display()
            return
        
        # Store engine reference for tempo updates
        self._current_engine = engine
        
        metadata = engine.get_metadata()
        file_name = os.path.basename(engine.midi_file.filename)
        
        self.file_name_label.setText(f"<b>File:</b> {file_name}")
        self.file_title_label.setText(f"<b>Title:</b> {metadata.get('title', '-')}")
        self.file_tracks_label.setText(f"<b>Tracks:</b> {metadata.get('tracks', '-')}")
        self.file_resolution_label.setText(f"<b>Resolution:</b> {metadata.get('ticks_per_beat', '-')} ticks/beat")
        
        
        # Display playback mode banner if in DYNAMIC mode
        if hasattr(engine, 'session_state'):
            from core.playback_mode import PlaybackMode

            if engine.session_state.mode == PlaybackMode.DYNAMIC:
                warnings = engine.session_state.warnings
                warning_text = ", ".join(warnings) if warnings else "Unknown reason"

                # Customize message based on the reason for dynamic mode
                if "Default metadata detected" in warning_text:
                    banner_text = (f"Suspicious metadata. Click 'Generate Beat Track' for accurate analysis.")
                elif "Type 0 MIDI file" in warning_text:
                    banner_text = (f"Untrustworthy metadata of type 0. Click 'Generate Beat Track' for accurate analysis.")
                else:
                    banner_text = (f"Performance MIDI Detected. Click 'Generate Beat Track' to enable.")

                self.playback_mode_banner.setText(banner_text)
                self.playback_mode_banner.show()
            else:
                self.playback_mode_banner.hide()
        else:
            self.playback_mode_banner.hide()
        
        # Prepare metadata tempo info
        if not engine.session_state.has_tempo_metadata:
            # No actual tempo metadata
            self._metadata_tempo_info = ""
            self.file_tempo_label.setText(f"<b>MIDI Tempo:</b> -")
            self.file_tempo_label.setToolTip("No tempo found in metadata.")
        elif metadata.get('has_dynamic_tempo', False):
            # Dynamic tempo - get raw MIDI values for reference only
            initial_tempo_midi = metadata.get('initial_tempo', 120)
            average_tempo_midi = metadata.get('average_tempo', 120)
            
            # Store raw MIDI tempo info for reference
            self._metadata_tempo_info = f"({int(initial_tempo_midi)}; {int(average_tempo_midi)})"

            self.file_tempo_label.setText(f"<b>MIDI Tempo:</b> Initial - {int(initial_tempo_midi)};  Average - {int(average_tempo_midi)}")
            self.file_tempo_label.setToolTip("Dynamic tempo found in MIDI metadata, " 
                                             "showing initial tempo and weighted average."
                                             )
        else:
            # Static tempo
            tempo_value_midi = metadata.get('initial_tempo') or metadata.get('tempo', '-')
            if tempo_value_midi != '-':
                # Store raw MIDI tempo info for reference (never changes)
                self._metadata_tempo_info = f"({int(tempo_value_midi)})"
                
                self.file_tempo_label.setText(f"<b>MIDI Tempo:</b> {int(tempo_value_midi)} BPM")
                self.file_tempo_label.setToolTip("Fixed tempo found from MIDI metadata")
            else:
                self._metadata_tempo_info = ""
                self.file_tempo_label.setText(f"<b>MIDI Tempo:</b> -")
                self.file_tempo_label.setToolTip("No tempo found in metadata.")
        
        # Time signature display
        time_signature = metadata.get('time_signature', {})
        if isinstance(time_signature, dict) and 'text' in time_signature:
            # Check if this is actual metadata or a default value
            if engine.session_state.has_time_signature_metadata:
                # Show initial time signature with dynamic indicator
                time_sig_text = time_signature['text']
                if metadata.get('has_dynamic_time_signatures', False):
                    # Add indicator for dynamic time signatures
                    dynamic_preview = metadata.get('dynamic_signatures_preview', '')
                    time_sig_text += f" (Multi meter found: {dynamic_preview}. Showing selected sections)"
                self.file_time_signature_label.setText(f"<b>Time Signature:</b> {time_sig_text}")
            else:
                # No actual time signature metadata
                self.file_time_signature_label.setText(f"<b>Time Signature:</b> -")
        else:
            self.file_time_signature_label.setText(f"<b>Time Signature:</b> -")
        
        # Key signature and length
        key_signature = metadata.get('key_signature')
        # Only display key signature if it was explicitly found in the MIDI
        if key_signature and key_signature != 'C major':  # Don't show default C major
            self.file_key_signature_label.setText(f"<b>Key Signature:</b> {key_signature}")
        else:
            self.file_key_signature_label.setText(f"<b>Key Signature:</b> -")
        
        length_seconds = metadata.get('length', 0)
        minutes = int(length_seconds // 60)
        seconds = int(length_seconds % 60)
        self.file_length_label.setText(f"<b>Length est.:</b> {minutes}:{seconds:02d}")
        
        # display total bars (sum of all section measures)
        sections = metadata.get('sections', [])
        total_bar_count = sum(s.get('measures',0) for s in sections)
        self.total_bars_label.setText(f"<b>Total Bars:</b> {total_bar_count}")
    
    def update_playback_info(self, timing_info):
        """Update playback position and timing information
        
        Args:
            timing_info: Dictionary containing timing information from engine
        """
        if timing_info.get('precision_active', False):
            # Display precise position
            position_seconds = timing_info['seconds']
            minutes = int(position_seconds // 60)
            seconds = int(position_seconds % 60)
            milliseconds = int((position_seconds % 1) * 1000)
            self.file_length_label.setText(f"<b>Position:</b> {minutes}:{seconds:02d}.{milliseconds:03d}")
            
        else:
            # Fallback to regular timing
            position_seconds = timing_info.get('position_seconds', 0)
            minutes = int(position_seconds // 60)
            seconds = int(position_seconds % 60)
            self.file_length_label.setText(f"<b>Position:</b> {minutes}:{seconds:02d}")
    
    def update_stopped_state(self, engine):
        """Update display when playback is stopped
        
        Args:
            engine: MIDI engine instance
        """
        if engine.midi_file:
            metadata = engine.get_metadata()
            length_seconds = metadata.get('length', 0)
            minutes = int(length_seconds // 60)
            seconds = int(length_seconds % 60)
            self.file_length_label.setText(f"<b>Length est.:</b> {minutes}:{seconds:02d}")
        else:
            pass
    
    def clear_display(self):
        """Clear all display labels to default state"""
        self.file_name_label.setText("<b>File:</b> Not loaded")
        self.file_title_label.setText("<b>Title:</b> -")
        self.file_tracks_label.setText("<b>Tracks:</b> -")
        self.file_resolution_label.setText("<b>Resolution:</b> -")
        self.file_tempo_label.setText("<b>MIDI Tempo:</b> -")
        self.file_time_signature_label.setText("<b>Time Signature:</b> -")
        self.file_key_signature_label.setText("<b>Key Signature:</b> -")
        self.file_length_label.setText("<b>Length est.:</b> -")
        # self.total_bars_label.setText("<b>Total Bars:</b> -")
        
        
        # Clear stored tempo info
        self._metadata_tempo_info = ""
        self._current_engine = None
        
        self.file_tempo_label.setToolTip("")
