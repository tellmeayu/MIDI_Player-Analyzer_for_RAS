"""Beat tracking analysis service.

This module provides a pure Python service for beat tracking analysis,
independent of Qt for better testability and reusability.
"""

from typing import Dict, Optional
import numpy as np
from .beat_tracker_basic import MidiBeatTracker


def generate_beat_track(midi_path: str, sf2_path: Optional[str] = None) -> Dict:
    """
    Generate beat track from MIDI file using audio-based analysis.

    This is a pure Python function with no Qt dependencies, making it
    suitable for unit testing and use in non-GUI contexts.

    Args:
        midi_path: Path to the MIDI file
        sf2_path: Path to SoundFont file for synthesis (optional)

    Returns:
        Dictionary containing:
            - 'beat_times': numpy array of beat times in seconds
            - 'estimated_tempo_bpm': estimated tempo in BPM
            - 'regularity': dict with 'mean_ibi', 'std_ibi', 'cv_ibi'
            - 'total_beats': number of beats detected
            - 'errors': list of error messages (empty if successful)
            - 'warnings': list of warning messages

    Example:
        >>> result = generate_beat_track('song.mid', 'soundfont.sf2')
        >>> if not result['errors']:
        >>>     print(f"Detected {result['total_beats']} beats")
        >>>     print(f"Tempo: {result['estimated_tempo_bpm']:.1f} BPM")
    """
    result = {
        'beat_times': None,
        'estimated_tempo_bpm': 0.0,
        'regularity': {'mean_ibi': 0.0, 'std_ibi': 0.0, 'cv_ibi': 0.0},
        'total_beats': 0,
        'errors': [],
        'warnings': []
    }

    try:
        # Create beat tracker
        tracker = MidiBeatTracker(midi_path, sf2_path)

        # Get beat times
        beat_times = tracker.get_beat_times()

        if beat_times is None or len(beat_times) == 0:
            result['errors'].append("No beats detected in the MIDI file")
            return result

        # Get estimated tempo
        tempo = tracker.get_estimated_tempo()

        # Get beat regularity
        regularity = tracker.get_beat_regularity()

        # Populate result
        result['beat_times'] = beat_times
        result['estimated_tempo_bpm'] = float(tempo)
        result['regularity'] = regularity
        result['total_beats'] = len(beat_times)

        # Add warnings for irregular beats
        cv_ibi = regularity.get('cv_ibi', 0.0)
        if cv_ibi > 0.04:
            result['warnings'].append(
                f"Beat regularity is irregular (CV: {cv_ibi:.3f}). "
                "Not recommended for RAS therapy."
            )
        elif cv_ibi > 0.02:
            result['warnings'].append(
                f"Beat regularity is marginal (CV: {cv_ibi:.3f}). "
                "Use with caution for RAS therapy."
            )

    except FileNotFoundError as e:
        result['errors'].append(f"File not found: {str(e)}")
    except Exception as e:
        result['errors'].append(f"Beat tracking failed: {str(e)}")

    return result


