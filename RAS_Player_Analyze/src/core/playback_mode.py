"""Playback mode definitions for the MIDI player.

This module defines the different playback modes available in the MIDI player:
- STANDARD: Traditional grid-based playback with fixed tempo and meter
- DYNAMIC: Beat-tracked playback for performance MIDIs without metadata
"""

from enum import Enum, auto

class PlaybackMode(Enum):
    """Enum defining the available playback modes."""
    
    STANDARD = auto()  # Traditional grid-based playback with fixed tempo and meter
    DYNAMIC = auto()   # Beat-tracked playback for performance MIDIs without metadata
    
    def __str__(self):
        """Return a user-friendly string representation."""
        return self.name.capitalize()
