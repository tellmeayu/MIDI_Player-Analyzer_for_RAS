"""Player session state management.

This module provides a class to manage the state of a player session, including:
- Playback mode (STANDARD or DYNAMIC)
- Timing offset for anacrusis correction
- Beat track for performance MIDIs
"""

from typing import Optional, List
import numpy as np
from .playback_mode import PlaybackMode

class PlayerSessionState:
    """Manages the state of a player session."""
    
    def __init__(self):
        """Initialize a new player session state."""
        self.mode: PlaybackMode = PlaybackMode.STANDARD
        self.timing_offset_sec: float = 0.0
        self.has_beat_track: bool = False
        self.beat_times: Optional[np.ndarray] = None
        self.estimated_tempo_bpm: float = 0.0
        self.beat_regularity: Optional[dict] = None
        self.has_tempo_metadata: bool = True
        self.has_time_signature_metadata: bool = True
        self.warnings: List[str] = []
        self.anacrusis_offset_beats: float = 0.0
        self.pickup_beats: float = 0.0  # Store number of pickup beats for user display
        self.fine_adjustment: float = 0.0  # Store fine adjustment in fractions of a beat (-0.75 to +0.75)
        self.has_anacrusis_correction: bool = False
        self.user_preferred_tempo: float = 0.0  # Store user's preferred tempo (0.0 means use default)
        
    def reset(self):
        """Reset the session state to defaults."""
        self.__init__()
        
    def __str__(self):
        """Return a string representation of the session state."""
        status = []
        status.append(f"Mode: {self.mode}")
        
        if self.mode == PlaybackMode.STANDARD:
            if self.timing_offset_sec != 0.0:
                status.append(f"Timing offset: {self.timing_offset_sec:.3f}s")
            if self.has_anacrusis_correction and self.pickup_beats > 0:
                # Include fine adjustment in display if it exists
                if self.fine_adjustment != 0.0:
                    status.append(f"Pickup beats: {self.pickup_beats:.2f} + {self.fine_adjustment:+.2f}")
                else:
                    status.append(f"Pickup beats: {self.pickup_beats:.2f}")
        elif self.mode == PlaybackMode.DYNAMIC:
            if self.has_beat_track:
                if self.beat_times is not None:
                    status.append(f"Beat track: {len(self.beat_times)} beats")
                    if self.estimated_tempo_bpm > 0:
                        status.append(f"Estimated tempo: {self.estimated_tempo_bpm:.1f} BPM")
                else:
                    status.append("Beat track: Empty")
            else:
                status.append("Beat track: Not generated")
                
            if self.beat_regularity is not None:
                cv = self.beat_regularity.get('cv_ibi', 0)
                if cv <= 0.02:
                    status.append("Regularity: Stable")
                elif cv <= 0.04:
                    status.append("Regularity: Marginal")
                else:
                    status.append("Regularity: Irregular")
                
        if self.warnings:
            status.append(f"Warnings: {len(self.warnings)}")
            
        return ", ".join(status)
