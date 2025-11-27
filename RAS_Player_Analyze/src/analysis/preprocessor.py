"""Unified MIDI preprocessing and audio synthesis utilities for analyzers.

This module centralizes common symbolic processing steps used across analyzers
in the analysis package to ensure consistent behavior and reduce duplication.

Pipeline (typical):
 1) load_midi → 2) merge_instruments_by_name → 3) first_note_time →
 4) trim_pm (optionally shift_to_zero) → 5) synthesize/synthesize_wav

The functions are stateless; a simple in-memory cache is provided for
audio synthesis results keyed by deterministic content digests.

Note on tempo changes: pretty_midi encodes note times in absolute seconds;
FluidSynth rendering uses those absolute times, so tempo-change copying is
not required for accurate audio output. Time signatures are copied for
downstream analysis compatibility.
"""

from __future__ import annotations

import copy
import hashlib
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pretty_midi
from scipy.io import wavfile
import mido


# ------------------------------ Hashing utils ------------------------------ #


def _hash_bytes(data: bytes) -> str:
    """Return a short hex digest for given bytes.

    Args:
        data: Raw bytes to hash.

    Returns:
        A short (12 chars) hex digest.
    """
    return hashlib.sha256(data).hexdigest()[:12]


def hash_pm(pm: pretty_midi.PrettyMIDI) -> str:
    """Compute a deterministic content hash for a PrettyMIDI object.

    The hash is based on instrument meta (name, program, is_drum) and all note
    tuples (pitch, start, end, velocity) in chronological order. Control and
    pitch-bend events are not included to keep hashing fast and stable.

    Args:
        pm: PrettyMIDI object.

    Returns:
        A short hex digest string.
    """
    parts: List[bytes] = []
    parts.append(str(pm.resolution).encode("utf-8"))
    # Include time signature changes for determinism when trimming/merging
    for ts in getattr(pm, "time_signature_changes", []) or []:
        parts.append(f"TS:{ts.numerator}/{ts.denominator}@{ts.time:.6f}".encode("utf-8"))
    for inst in pm.instruments:
        parts.append(
            f"I:{inst.name}|{inst.program}|{int(inst.is_drum)}".encode("utf-8")
        )
        for note in inst.notes:
            parts.append(
                f"N:{note.pitch}|{note.start:.6f}|{note.end:.6f}|{note.velocity}".encode(
                    "utf-8"
                )
            )
    return _hash_bytes(b"|".join(parts))


# ------------------------------ Load / Merge ------------------------------- #


def load_midi(midi_path: str) -> pretty_midi.PrettyMIDI:
    """Load a MIDI file.

    Args:
        midi_path: Path to MIDI file (.mid/.midi).

    Returns:
        Loaded PrettyMIDI.

    Raises:
        FileNotFoundError: If the path does not exist.
        Exception: On parse errors.
    """
    if not os.path.exists(midi_path):
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")
    return pretty_midi.PrettyMIDI(midi_path)


def merge_instruments_by_name(pm: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
    """Merge instruments by effective name to avoid fragmentation.

    - If instrument.name is empty, use a stable fallback: "Program {program}".
    - Preserve first-seen program per name for non-drum parts.
    - Mark merged instrument as drum if any source instrument is drum.

    Args:
        pm: Source PrettyMIDI.

    Returns:
        A new PrettyMIDI with merged instruments and copied meta events.
    """
    merged = pretty_midi.PrettyMIDI(resolution=pm.resolution)

    # Copy meta events that are safe/meaningful for downstream analysis
    for ts in getattr(pm, "time_signature_changes", []) or []:
        merged.time_signature_changes.append(ts)

    # Group notes by effective name
    grouped_notes: Dict[str, List[pretty_midi.Note]] = {}
    program_by_name: Dict[str, int] = {}
    is_drum_by_name: Dict[str, bool] = {}

    for inst in pm.instruments:
        name = inst.name.strip() if inst.name and inst.name.strip() else f"Program {inst.program}"
        if name not in grouped_notes:
            grouped_notes[name] = []
            program_by_name[name] = inst.program
            is_drum_by_name[name] = bool(inst.is_drum)
        else:
            # Once drum, always drum for this group
            is_drum_by_name[name] = is_drum_by_name[name] or bool(inst.is_drum)
        grouped_notes[name].extend(inst.notes)

    for name, notes in grouped_notes.items():
        new_inst = pretty_midi.Instrument(
            program=program_by_name[name], is_drum=is_drum_by_name[name], name=name
        )
        # Maintain original order
        new_inst.notes.extend(sorted(notes, key=lambda n: (n.start, n.end, n.pitch)))
        merged.instruments.append(new_inst)

    return merged


# ----------------------------- Detection / Trim ---------------------------- #


def first_note_time(pm: pretty_midi.PrettyMIDI, include_drums: bool = False) -> float:
    """Return the earliest note-on time across instruments.

    Args:
        pm: PrettyMIDI object.
        include_drums: Whether to include drum tracks when searching.

    Returns:
        Earliest note start time in seconds; 0.0 if no notes found.
    """
    t0 = float("inf")
    for inst in pm.instruments:
        if not include_drums and inst.is_drum:
            continue
        if inst.notes:
            # notes are sorted in pretty_midi
            t0 = min(t0, inst.notes[0].start)
    return 0.0 if t0 == float("inf") else float(t0)


def _shift_and_clip_time(value: float, start_s: float, duration_s: float, shift: bool) -> Optional[float]:
    """Shift and clip a timestamp to the target window.

    Args:
        value: Original time in seconds.
        start_s: Window start (inclusive).
        duration_s: Window duration.
        shift: If True, subtract start_s.

    Returns:
        New time within [0, duration_s] if inside window, else None.
    """
    end_s = start_s + duration_s
    if value < start_s or value >= end_s:
        return None
    return (value - start_s) if shift else value


def trim_pm(
    pm: pretty_midi.PrettyMIDI,
    start_s: float,
    duration_s: float,
    shift_to_zero: bool = True,
    ) -> pretty_midi.PrettyMIDI:
    """Trim a PrettyMIDI to a [start_s, start_s+duration_s) window and shift times.

    Notes are included if they start within the window. Note ends are clipped to
    the window end if necessary. Control changes and pitch bends are preserved
    within the window. Time signatures are copied and shifted as needed. Tempo
    changes are intentionally not rewritten (audio rendering uses absolute note times).

    Args:
        pm: Source PrettyMIDI.
        start_s: Start time (seconds, inclusive).
        duration_s: Window duration (seconds).
        shift_to_zero: If True, shift all retained events so the window start is 0.

    Returns:
        A new PrettyMIDI object representing the trimmed content.
    """
    trimmed = pretty_midi.PrettyMIDI(resolution=pm.resolution)

    # Copy time signatures in window
    added_ts = False
    for ts in getattr(pm, "time_signature_changes", []) or []:
        new_time = _shift_and_clip_time(ts.time, start_s, duration_s, shift_to_zero)
        if new_time is not None:
            trimmed.time_signature_changes.append(
                pretty_midi.TimeSignature(ts.numerator, ts.denominator, new_time)
            )
            added_ts = True

    # Ensure an initial time signature at t=0 if shifting and none in window
    if shift_to_zero and not added_ts and getattr(pm, "time_signature_changes", None):
        first_ts = pm.time_signature_changes[0]
        trimmed.time_signature_changes.append(
            pretty_midi.TimeSignature(first_ts.numerator, first_ts.denominator, 0.0)
        )

    # Instruments
    for src in pm.instruments:
        dst = pretty_midi.Instrument(program=src.program, is_drum=src.is_drum, name=src.name)

        # Notes: keep those starting in window, clip end
        for note in src.notes:
            new_start = _shift_and_clip_time(note.start, start_s, duration_s, shift_to_zero)
            if new_start is None:
                continue
            new_end = note.end
            if new_end > start_s + duration_s:
                new_end = start_s + duration_s
            if shift_to_zero:
                new_end = max(new_start, new_end - start_s)
            dst.notes.append(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=float(new_start),
                    end=float(new_end),
                )
            )

        # Control changes
        for cc in src.control_changes:
            new_time = _shift_and_clip_time(cc.time, start_s, duration_s, shift_to_zero)
            if new_time is None:
                continue
            dst.control_changes.append(
                pretty_midi.ControlChange(number=cc.number, value=cc.value, time=float(new_time))
            )

        # Pitch bends
        for pb in src.pitch_bends:
            new_time = _shift_and_clip_time(pb.time, start_s, duration_s, shift_to_zero)
            if new_time is None:
                continue
            dst.pitch_bends.append(
                pretty_midi.PitchBend(pitch=pb.pitch, time=float(new_time))
            )

        if dst.notes or dst.control_changes or dst.pitch_bends:
            trimmed.instruments.append(dst)

    return trimmed


# ------------------------------- Synthesis -------------------------------- #


_AUDIO_CACHE: Dict[Tuple[str, float, float, int, str], np.ndarray] = {}
_WAV_CACHE: Dict[Tuple[str, float, float, int, str, str], str] = {}


def synthesize(
    pm: pretty_midi.PrettyMIDI,
    sf2_path: str,
    sr: int = 22050,
    cache_key_hint: Optional[str] = None,
    ) -> np.ndarray:
    """Render MIDI to audio waveform using FluidSynth via pretty_midi.

    Args:
        pm: PrettyMIDI to render.
        sf2_path: Path to SoundFont (.sf2).
        sr: Sample rate for rendering.
        cache_key_hint: Optional external digest to reuse across sessions.

    Returns:
        Audio waveform as numpy array (float32), mono or stereo per backend.
    """
    if not os.path.exists(sf2_path):
        raise FileNotFoundError(f"SoundFont not found: {sf2_path}")

    digest = cache_key_hint or hash_pm(pm)
    key = (digest, 0.0, 0.0, int(sr), os.path.basename(sf2_path))
    if key in _AUDIO_CACHE:
        return _AUDIO_CACHE[key]

    audio = pm.fluidsynth(fs=sr, sf2_path=sf2_path)
    # pretty_midi returns float64; cast to float32 for consistency/size
    audio = audio.astype(np.float32, copy=False)
    _AUDIO_CACHE[key] = audio
    return audio


def synthesize_wav(
    pm: pretty_midi.PrettyMIDI,
    sf2_path: str,
    sr: int = 22050,
    suffix: str = "",
    cache_key_hint: Optional[str] = None,
    ) -> str:
    """Render to a temporary WAV file and return its path.

    The file name includes a short digest to aid reuse across repeated calls.

    Args:
        pm: PrettyMIDI to render.
        sf2_path: Path to SoundFont (.sf2).
        sr: Sample rate for rendering.
        suffix: Additional string to differentiate contexts.
        cache_key_hint: Optional external digest to reuse across sessions.

    Returns:
        Absolute path to the temporary WAV file.
    """
    if not os.path.exists(sf2_path):
        raise FileNotFoundError(f"SoundFont not found: {sf2_path}")

    digest = cache_key_hint or hash_pm(pm)
    wav_key = (digest, 0.0, 0.0, int(sr), os.path.basename(sf2_path), suffix)
    if wav_key in _WAV_CACHE and os.path.exists(_WAV_CACHE[wav_key]):
        return _WAV_CACHE[wav_key]

    audio = synthesize(pm, sf2_path, sr=sr, cache_key_hint=digest)

    # Write to a stable-named temp file
    tmp_dir = tempfile.gettempdir()
    base_name = f"midi_render_{digest}{('_' + suffix) if suffix else ''}_{sr}Hz.wav"
    tmp_path = os.path.join(tmp_dir, base_name)
    wavfile.write(tmp_path, sr, audio)
    _WAV_CACHE[wav_key] = tmp_path
    return tmp_path


# ---------------------------- Convenience API ----------------------------- #


def prepare_audio_from_midi(
    midi_path: str,
    sf2_path: str,
    duration_s: float,
    sr: int = 22050,
    include_drums_for_onset: bool = False,
    ) -> Tuple[np.ndarray, int]:
    """End-to-end: load → merge → first note → trim window → render audio.

    Args:
        midi_path: Path to MIDI file.
        sf2_path: SoundFont path.
        duration_s: Window duration starting at first note.
        sr: Sample rate.
        include_drums_for_onset: Whether to include drums when finding first note.

    Returns:
        Tuple of (audio array, sample rate).
    """
    pm0 = load_midi(midi_path)
    pm1 = merge_instruments_by_name(pm0)
    t0 = first_note_time(pm1, include_drums=include_drums_for_onset)
    pm2 = trim_pm(pm1, start_s=t0, duration_s=duration_s, shift_to_zero=True)
    digest = _hash_bytes((midi_path + f"|{t0:.3f}|{duration_s:.3f}").encode("utf-8"))
    audio = synthesize(pm2, sf2_path=sf2_path, sr=sr, cache_key_hint=digest)
    return audio, sr


def prepare_wav_from_midi(
    midi_path: str,
    sf2_path: str,
    duration_s: float,
    sr: int = 22050,
    include_drums_for_onset: bool = False,
    suffix: str = "",
    ) -> str:
    """End-to-end: load → merge → first note → trim window → render to WAV.

    Args:
        midi_path: Path to MIDI file.
        sf2_path: SoundFont path.
        duration_s: Window duration starting at first note.
        sr: Sample rate.
        include_drums_for_onset: Whether to include drums when finding first note.
        suffix: Extra suffix for the temp filename.

    Returns:
        Absolute path to the temporary WAV file.
    """
    pm0 = load_midi(midi_path)
    pm1 = merge_instruments_by_name(pm0)
    t0 = first_note_time(pm1, include_drums=include_drums_for_onset)
    pm2 = trim_pm(pm1, start_s=t0, duration_s=duration_s, shift_to_zero=True)
    digest = _hash_bytes((midi_path + f"|{t0:.3f}|{duration_s:.3f}").encode("utf-8"))
    return synthesize_wav(pm2, sf2_path=sf2_path, sr=sr, suffix=suffix, cache_key_hint=digest)


# ------------------------------- Anacrusis Detection ------------------------------- #

def create_detector_clip(midi_path: str, start_beat: float) -> str:
    """Create a trimmed MIDI clip for anacrusis detection.

    Converts start_beat to absolute seconds, extracts a 15-second window,
    and returns a path to the trimmed MIDI file. The trimmed MIDI will have
    correct note timings for audio synthesis; FluidSynth renders based on
    absolute note times (seconds), not MIDI tempo information.

    Args:
        midi_path: Path to source MIDI file.
        start_beat: Starting beat number for the clip.

    Returns:
        Path to a temporary trimmed MIDI file.

    Raises:
        IOError: If MIDI file cannot be loaded or written.
        ValueError: If the MIDI file has no beats.
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        raise IOError(f"Failed to load MIDI file: {midi_path}") from e

    # Convert start_beat to absolute seconds
    beat_times = pm.get_beats()
    if len(beat_times) == 0:
        raise ValueError("No beats found in MIDI file.")
    
    # Interpolate beat position if start_beat is fractional
    if start_beat >= len(beat_times):
        start_time_s = beat_times[-1]
    else:
        lower_idx = int(start_beat)
        frac = start_beat - lower_idx
        if frac > 0 and lower_idx + 1 < len(beat_times):
            start_time_s = beat_times[lower_idx] + frac * (beat_times[lower_idx + 1] - beat_times[lower_idx])
        else:
            start_time_s = beat_times[lower_idx]

    # Use trim_pm to handle all MIDI manipulation
    # trim_pm preserves CC and pitch-bends, copies time signatures, and shifts to t=0
    trimmed_pm = trim_pm(
        pm,
        start_s=start_time_s,
        duration_s=20.0,
        shift_to_zero=True,
    )

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        temp_path = f.name

    try:
        trimmed_pm.write(temp_path)
    except Exception as e:
        raise IOError(f"Failed to write trimmed MIDI file to {temp_path}") from e

    return temp_path


# ------------------------------- Data Classes ------------------------------ #

@dataclass
class PreprocessSummary:
    """Summary of preprocessing steps for logging or UI.

    Attributes:
        midi_path: Source MIDI path.
        merged_instruments: Number of instruments after merging.
        window_start: Start time used for trimming.
        window_duration: Duration used for trimming.
        time_signatures: Count of time signature changes retained.
        digest: Content digest of the trimmed PM.
    """

    midi_path: str
    merged_instruments: int
    window_start: float
    window_duration: float
    time_signatures: int
    digest: str


def summarize_preprocess(
    midi_path: str, pm_merged: pretty_midi.PrettyMIDI, pm_trimmed: pretty_midi.PrettyMIDI, start_s: float, duration_s: float
    ) -> PreprocessSummary:
    """Create a human-readable summary of the preprocessing result."""
    return PreprocessSummary(
        midi_path=midi_path,
        merged_instruments=len(pm_merged.instruments),
        window_start=float(start_s),
        window_duration=float(duration_s),
        time_signatures=len(getattr(pm_trimmed, "time_signature_changes", []) or []),
        digest=hash_pm(pm_trimmed),
        )
