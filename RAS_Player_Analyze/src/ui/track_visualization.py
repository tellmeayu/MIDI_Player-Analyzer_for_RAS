from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QColor, QFont, QPen
from typing import Dict, Optional

from core.track_activity_monitor import TrackActivityState

def velocity_to_orange_saturation_color(velocity: int) -> QColor:
    """Convert MIDI velocity to orange saturation color
    
    Uses unified orange hue with saturation variation to represent velocity intensity:
    - velocity = 0: Light gray (silent state)
    - Low velocity: Light orange (low saturation)
    - High velocity: Deep orange (high saturation)
    
    Args:
        velocity: MIDI velocity (0-127)
        
    Returns:
        QColor object representing the velocity
    """
    if velocity == 0:
        return QColor(192, 192, 192)  # Light gray - silent state
    
    # Map velocity (1-127) to 4 saturation levels
    if velocity <= 31:  # Gentle playing
        return QColor(255, 212, 170)  # Light orange #FFD4AA
    elif velocity <= 63:  # Medium intensity
        return QColor(255, 170, 85)   # Medium orange #FFAA55
    elif velocity <= 95:  # Strong intensity
        return QColor(255, 128, 0)    # Orange #FF8000
    else:  # Maximum intensity (96-127)
        return QColor(255, 85, 0)     # Deep orange #FF5500


def get_velocity_color_description(velocity: int) -> str:
    """Get velocity color description for accessibility
    
    Args:
        velocity: MIDI velocity (0-127)
        
    Returns:
        String description of the color/intensity
    """
    if velocity == 0:
        return "Silent (Light Gray)"
    elif velocity <= 31:
        return "Gentle Playing (Light Orange)"
    elif velocity <= 63:
        return "Medium Intensity (Medium Orange)"
    elif velocity <= 95:
        return "Strong Intensity (Orange)"
    else:
        return "Maximum Intensity (Deep Orange)"


class TrackActivityBar(QWidget):
    """Single track activity visualization square
    
    Displays real-time activity for one MIDI track as a colored square
    that changes color based on note velocity and shows silence with gray color.
    """
    
    def __init__(self, track_id: int, track_name: str, parent=None):
        """Initialize track activity square
        
        Args:
            track_id: Track index
            track_name: Display name for the track
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.track_id = track_id
        self.track_name = track_name
        self.current_velocity = 0
        self.is_active = False
        
        # Visual configuration - increased size for better visibility
        self.square_size = 12  # Increased from 12 to 16 pixels
        self.setFixedHeight(self.square_size)
        self.setFixedWidth(self.square_size)
        
        # Color caching for performance
        self._last_velocity = -1
        self._cached_color = None
        
        # Tooltip for accessibility
        self.setToolTip(f"Track {track_id}: {track_name}")
    
    def update_activity(self, velocity: int, is_active: bool):
        """Update activity state and redraw
        
        Args:
            velocity: Current velocity (0-127)
            is_active: Whether track has active notes
        """
        # Only update if state actually changed
        if velocity != self.current_velocity or is_active != self.is_active:
            self.current_velocity = velocity
            self.is_active = is_active
            
            # Update tooltip with current state
            state_desc = get_velocity_color_description(velocity)
            self.setToolTip(f"Track {self.track_id}: {self.track_name}\nState: {state_desc}")
            
            # Trigger repaint
            self.update()
    
    def paintEvent(self, event):
        """Custom paint event for drawing the activity square
        
        Args:
            event: Paint event
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get color (with caching for performance)
        if self.current_velocity != self._last_velocity:
            self._cached_color = velocity_to_orange_saturation_color(self.current_velocity)
            self._last_velocity = self.current_velocity
        
        # Draw simple filled square without border or decorations
        square_rect = QRect(0, 0, self.square_size, self.square_size)
        painter.fillRect(square_rect, self._cached_color)


class TrackVisualizationWidget(QWidget):
    """Enhanced track list widget with real-time activity visualization
    
    Replaces the simple track list with a visualization that shows track names
    on the left and activity bars on the right, with separator lines between tracks.
    """
    
    def __init__(self, parent=None):
        """Initialize track visualization widget
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Layout setup with more compact spacing
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)  # Reduced margins
        self.main_layout.setSpacing(2)  # Reduced spacing between tracks
        
        # Storage for track components
        self.track_widgets: Dict[int, Dict] = {}  # track_id -> {'name_label': QLabel, 'activity_bar': TrackActivityBar, 'frame': QFrame}
        
        # Callback for track click events - set by parent
        self.track_clicked_callback = None
        
        # No tracks initially
        self.setMinimumHeight(100)
    
    def clear_tracks(self):
        """Clear all track widgets"""
        # Clear layout
        while self.main_layout.count():
            child = self.main_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Clear storage
        self.track_widgets.clear()
    
    def add_track(self, track_id: int, track_name: str, instrument_display: Optional[str] = None):
        """Add a track to the visualization
        
        Args:
            track_id: Track index
            track_name: Track name
            instrument_display: Optional instrument display string
        """
        # Create main frame for this track (simple frame without extra styling)
        track_frame = QFrame()
        track_frame.setFrameStyle(QFrame.NoFrame)  # Remove frame borders
        track_frame.setStyleSheet("""
            QFrame {
                background-color: white;
            }
            QFrame:hover {
                background-color: #F5F5F5;
            }
        """)
        track_frame.setFixedHeight(36)  # Slightly reduced height for more compact layout
        track_frame.setCursor(Qt.PointingHandCursor)  # Show it's clickable
        
        # Layout for this track (horizontal: activity square on left, name on right)
        track_layout = QHBoxLayout(track_frame)
        track_layout.setContentsMargins(8, 6, 8, 6)  # Reduced vertical padding
        track_layout.setSpacing(12)  # Reduced spacing between square and text
        
        # Activity visualization square (left side) - larger for better visibility
        activity_bar = TrackActivityBar(track_id, track_name)
        track_layout.addWidget(activity_bar, 0)  # Fixed size, no stretch
        
        # Track name label (right side)
        display_text = track_name
        if instrument_display:
            display_text += f" - {instrument_display}"
        
        name_label = QLabel(display_text)
        name_label.setFont(QFont("Arial", 12))  # Slightly reduced font size
        name_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        name_label.setWordWrap(True)  # Allow text wrapping for long names
        track_layout.addWidget(name_label, 1)  # Stretch to take available space
        
        # Store components
        self.track_widgets[track_id] = {
            'name_label': name_label,
            'activity_bar': activity_bar,
            'frame': track_frame
        }
        
        # Add to main layout
        self.main_layout.addWidget(track_frame)
        
        # Set up click handling for mute/unmute functionality
        track_frame.mousePressEvent = lambda event, tid=track_id: self._on_track_clicked(tid, event)
        
        # Also make the name label clickable
        name_label.mousePressEvent = lambda event, tid=track_id: self._on_track_clicked(tid, event)
    
    def update_track_activity(self, track_id: int, activity_state: TrackActivityState):
        """Update activity visualization for a track
        
        Args:
            track_id: Track index
            activity_state: Current activity state
        """
        if track_id in self.track_widgets:
            activity_bar = self.track_widgets[track_id]['activity_bar']
            activity_bar.update_activity(activity_state.current_velocity, activity_state.is_active)
    
    def set_track_muted(self, track_id: int, is_muted: bool):
        """Update visual state for muted/unmuted track
        
        Args:
            track_id: Track index
            is_muted: Whether track is muted
        """
        if track_id in self.track_widgets:
            name_label = self.track_widgets[track_id]['name_label']
            frame = self.track_widgets[track_id]['frame']
            
            if is_muted:
                name_label.setStyleSheet("color: #888888; font-style: italic;")
                frame.setStyleSheet("""
                    QFrame {
                        background-color: #F8F8F8;
                    }
                    QFrame:hover {
                        background-color: #F0F0F0;
                    }
                """)
            else:
                name_label.setStyleSheet("color: #000000; font-style: normal;")
                frame.setStyleSheet("""
                    QFrame {
                        background-color: white;
                    }
                    QFrame:hover {
                        background-color: #F5F5F5;
                    }
                """)
    
    def _on_track_clicked(self, track_id: int, event):
        """Handle track click for mute/unmute
        
        Args:
            track_id: Track index
            event: Mouse event
        """
        # Call the callback if it's set
        if self.track_clicked_callback:
            self.track_clicked_callback(track_id)
    
    def finalize_layout(self):
        """Finalize the layout after all tracks are added"""
        # Add stretch at the end to push all tracks to the top
        self.main_layout.addStretch()
        
        # Note: Height is now handled by QScrollArea in the parent widget
        # No need to set minimum height here as scroll area will manage it 