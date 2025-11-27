import mido
import rtmidi
import threading
import time

from typing import Dict, List, Set, Optional, Tuple
import os
import numpy as np
from .precision_timer import PrecisionTimer
from .midi_metadata import MidiMetadataAnalyzer
from .playback_mode import PlaybackMode
from .player_session_state import PlayerSessionState

try:
    import fluidsynth
    FLUIDSYNTH_AVAILABLE = True
except ImportError:
    FLUIDSYNTH_AVAILABLE = False
    print("Warning: FluidSynth not found, will use MIDI output port instead of audio output")

class MidiEngine:
    """MIDI engine class responsible for MIDI file parsing, playback control and event handling

    FluidSynth Singleton Pattern:
    ==============================
    This class maintains a single FluidSynth instance for realtime MIDI synthesis.
    To prevent audio corruption and resource conflicts:

    - **Beat tracking analysis MUST NOT run during playback**
      Beat tracking uses pm.fluidsynth() which creates a temporary FluidSynth instance.
      Running this while the main FluidSynth is active causes audio driver conflicts.
      The GUI stops playback automatically before starting beat tracking.

    - **Audio player runs in separate process (process-level isolation)**
      The AudioPlayerWindow uses multiprocessing to run in a completely separate OS process,
      ensuring it has its own FluidSynth instance and audio device handles.

    - **Only ONE FluidSynth instance should be active in main process at any time**
      The cleanup_fluidsynth() method ensures proper resource release before:
      - Loading a new MIDI file
      - Reinitializing FluidSynth
      - Application shutdown

    - **Thread safety via audio_lock**
      All FluidSynth operations are protected by a threading.RLock to prevent
      concurrent access from multiple threads.
    """

    def __init__(self):
        """Initialize MIDI engine"""
        self.midi_file: Optional[mido.MidiFile] = None
        self.current_position: float = 0  # Current playback position (seconds)
        self.tempo: float = 120.0  # Default BPM
        self.is_playing: bool = False
        self.is_paused: bool = False
        self.active_notes: Dict[int, List[Tuple[int, int]]] = {}  # Track active notes {track_idx: [(note, channel), ...]}
        self.muted_tracks: Set[int] = set()  # Set of muted track indices
        
        # Metadata analyzer
        self.metadata_analyzer = MidiMetadataAnalyzer()
        self._cached_metadata: Optional[Dict] = None
        
        # High-precision timing system
        self.precision_timer = PrecisionTimer()
        self._playback_start_time: float = 0.0

        # Event scheduler 
        self.scheduler = None
        
        # Section playback state management (always in section mode now)
        self.current_section: Optional[Dict] = None
        self._section_start_beat: float = 0.0  # Section start position in beats
        self._section_end_beat: Optional[float] = None  # Section end position in beats
        
        # Player session state (playback mode, timing offset, beat track)
        self.session_state = PlayerSessionState()
        
        # Anacrusis correction (pickup beats)
        # In single-clock architecture: this offset delays MIDI events, creating
        # a silence period at start where only the metronome plays
        self.session_state.anacrusis_offset_beats = 0.0
        
        # Thread safety lock to protect FluidSynth and MIDI output concurrent access
        self.audio_lock = threading.RLock()
        
        self.midi_out = None
        self.available_ports = []
        self._init_midi_output()
        
        self.fs = None
        self.sf_id = None  # Explicitly initialize to None
        self._init_fluidsynth()
    
    def _init_midi_output(self):
        """Initialize MIDI output device"""
        try:
            self.midi_out = rtmidi.MidiOut()
            self.available_ports = self.midi_out.get_ports()
            
            # If ports available, open first one
            if self.available_ports:
                self.midi_out.open_port(0)
                print(f"Connected to MIDI output device: {self.available_ports[0]}")
            else:
                # If no available ports, create virtual port
                self.midi_out.open_virtual_port("MIDI Player Virtual Output")
                print("Created virtual MIDI output port")
        except Exception as e:
            print(f"Error initializing MIDI output: {e}")
            self.midi_out = None
    
    def _init_fluidsynth(self):
        """Initialize FluidSynth"""
        if not FLUIDSYNTH_AVAILABLE:
            print("[MidiEngine] FluidSynth not available!")
            return

        # Cleanup existing instance first to prevent resource leaks
        if hasattr(self, 'fs') and self.fs is not None:
            self.cleanup_fluidsynth()

        # Explicitly reset sf_id
        self.sf_id = None

        try:
            # Initialize with 22050 Hz to match analysis modules sample rate
            self.fs = fluidsynth.Synth(samplerate=22050.0)
            self.fs.start()
            
            # Try to load SoundFont
            soundfont_paths = [
                os.path.expanduser("~/.fluidsynth/default_sound_font.sf2"),  # User directory
                "/usr/share/sounds/sf2/FluidR3_GM.sf2",  # Linux system directory
                "/usr/local/share/fluidsynth/soundfonts/default.sf2",  # macOS Homebrew directory
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "resources", "FluidR3_GM.sf2"),  # Project resources directory
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "resources", "soundfonts", "FluidR3_GM.sf2")  # Project soundfonts directory
            ]
            
            loaded = False
            for sf_path in soundfont_paths:
                if os.path.exists(sf_path):
                    self.sf_id = self.fs.sfload(sf_path)
                    if self.sf_id != -1 and self.sf_id is not None:
                        # Mute briefly to avoid bursts on init
                        self.fs.setting('synth.gain', 0.0)
                        time.sleep(0.05)  # Brief delay to ensure mute is applied
                        
                        # Send all-notes-off to all channels immediately after loading SoundFont
                        # This prevents any residual notes from playing
                        for channel in range(16):
                            try:
                                self.fs.cc(channel, 123, 0)  # All notes off
                                self.fs.cc(channel, 120, 0)  # All sound off
                            except Exception:
                                pass
                        
                        time.sleep(0.05)  # Additional delay to ensure commands are processed
                        self.fs.setting('synth.gain', 1.0)  # Restore gain after cleanup
                        loaded = True
                        break
                    else:
                        print(f"[MidiEngine] ⚠️ SoundFont load returned invalid ID: {self.sf_id}")
            
            if not loaded:
                print("[MidiEngine] ❌ ERROR: SoundFont file not found, cannot generate audio")
                print(f"[MidiEngine]    Searched paths:")
                for sf_path in soundfont_paths:
                    exists = "✓" if os.path.exists(sf_path) else "✗"
                    print(f"[MidiEngine]      {exists} {sf_path}")
                self.cleanup_fluidsynth()
        except Exception as e:
            print(f"[MidiEngine] ❌ ERROR initializing FluidSynth: {e}")
            import traceback
            traceback.print_exc()
            self.cleanup_fluidsynth()

    def ensure_fs_ready(self) -> bool:
        """Ensure FluidSynth and SoundFont are ready; attempt one reinit if needed.

        Returns:
            bool: True if a valid synth and soundfont are available.
        """
        if not FLUIDSYNTH_AVAILABLE:
            return False
        with self.audio_lock:
            if self.fs is not None and self.sf_id is not None and self.sf_id != -1:
                return True
            # Try one re-initialization
            try:
                self.cleanup_fluidsynth()
                time.sleep(0.05)
                self._init_fluidsynth()
            except Exception as e:
                print(f"[MidiEngine] FluidSynth reinit error: {e}")
            return self.fs is not None and self.sf_id is not None and self.sf_id != -1

    def cleanup_fluidsynth(self):
        """Explicitly cleanup FluidSynth resources.

        This method ensures clean resource release before:
        - Loading a new MIDI file
        - Reinitializing FluidSynth
        - Application shutdown

        IMPORTANT: This prevents resource leaks and audio driver conflicts.
        """
        with self.audio_lock:
            if self.fs:
                try:
                    self.fs.delete()
                except Exception as e:
                    print(f"[MidiEngine] ⚠️ Warning: FluidSynth cleanup error: {e}")
                finally:
                    self.fs = None
                    self.sf_id = None

    def set_scheduler(self, scheduler):
        """Register the event scheduler for synchronized tempo changes
        
        Args:
            scheduler, EventScheduler instance to coordinate with
        """
        self.scheduler = scheduler

    def set_volume(self, volume: float):
        """Set MIDI music volume
    
        Args:
            volume: Volume level (0.0-1.0)
        """
        volume = max(0.0, min(1.0, volume))  # Clamp to valid range
        
        with self.audio_lock:
            # Set FluidSynth gain using the correct API
            if self.fs and self.sf_id is not None:
                fluidsynth_gain = volume * 1.5
                self.fs.setting('synth.gain', fluidsynth_gain)
            
            # Also send CC7 (Main Volume) to all channels for MIDI output
            if self.midi_out:
                midi_volume = int(volume * 127)  # Convert to MIDI range (0-127)
                for channel in range(16):
                    msg = mido.Message('control_change', channel=channel, control=7, value=midi_volume)
                    self._send_midi_message_unsafe(msg)

    def load_file(self, file_path: str, progress_callback=None) -> bool:
        """Load MIDI file with comprehensive error handling and state management
        
        Args:
            file_path: Path to MIDI file
            progress_callback: Optional callback for progress updates
            
        Returns:
            bool: True if file loaded successfully
        """
        
        try:
            print(f"Loading new MIDI file")
            
            if progress_callback:
                progress_callback("Initializing MIDI parser...")
            
            # === COMPLETE STATE RESET BEFORE LOADING NEW FILE ===
            # 1. Stop all playback activities immediately
            self.is_playing = False
            self.is_paused = False

            # ensure scheduler (from last playback) is fully stopped
            if self.scheduler:
                self.scheduler.stop()
                # Wait for scheduler thread to actually finish
                time.sleep(0.1)  # 100ms to ensure thread termination

            # 2. Stop and reset precision timer completely
            if hasattr(self, 'precision_timer'):
                self.precision_timer.stop()

            # 3. Stop all notes and clear audio state FIRST (while FluidSynth is still active)
            self.all_notes_off()
            time.sleep(0.1)  # 100ms delay to ensure all notes are fully stopped

            # 4. CRITICAL: Cleanup FluidSynth after stopping all notes
            # This ensures no resource leaks and prevents audio driver conflicts
            self.cleanup_fluidsynth()
            time.sleep(0.1)  # 100ms delay to ensure complete cleanup and driver release

            # 4.5. Reinitialize FluidSynth for the new file
            # This recreates a fresh synthesizer instance ready for playback
            self._init_fluidsynth()
            
            # 4.6. CRITICAL: After reinitialization, ensure all notes are off again
            # This prevents any residual notes from the previous file from playing
            # on the new FluidSynth instance
            if self.fs is not None and self.sf_id is not None and self.sf_id != -1:
                self.all_notes_off()
                time.sleep(0.05)  # 50ms delay to ensure cleanup is complete

            # 5. Clear track muting state - reset to unmuted
            if self.muted_tracks:
                # print(f"Clearing muted tracks: {list(self.muted_tracks)}")
                self.muted_tracks.clear()

            # 6. Reset and clear
            self.current_position = 0
            self._cached_metadata = None
            self.active_notes.clear()
            
            # Reset session state
            self.session_state.reset()
            
            # Ensure user_preferred_tempo is reset for new files
            self.session_state.user_preferred_tempo = 0.0
         
            if progress_callback:
                progress_callback("Reading MIDI file structure...")
            
            # Load the new MIDI file
            self.midi_file = mido.MidiFile(file_path)
            
            if progress_callback:
                progress_callback("Analyzing MIDI metadata...")
            
            # Get metadata with intelligent tempo analysis
            metadata = self.get_metadata()
            
            if progress_callback:
                progress_callback("Detecting valid initial tempo...")
            
            # Check for tempo metadata
            has_tempo = metadata.get('has_tempo_events', False)
            self.session_state.has_tempo_metadata = has_tempo
            
            # Use practical initial tempo from intelligent analysis
            self.tempo = metadata.get('initial_tempo', 120.0)
            
            # Initialize precision timer with defaults
            self.precision_timer.set_tempo(self.tempo)
            
            if progress_callback:
                progress_callback("Registering time signatures...")

            # Check for time signature metadata
            time_sig = metadata.get('time_signature', {})
            has_time_sig = (isinstance(time_sig, dict) and 
                           'numerator' in time_sig and 
                           'denominator' in time_sig)
            self.session_state.has_time_signature_metadata = has_time_sig
            
            # Set time signature for timing system
            if has_time_sig:
                self.precision_timer.set_time_signature(time_sig['numerator'], time_sig['denominator'])
            else:
                self.precision_timer.set_time_signature(4, 4)  # Default to 4/4
            
            # Determine playback mode based on metadata
            # Rule 1: No metadata at all → dynamic mode
            if not has_tempo or not has_time_sig:
                self.session_state.mode = PlaybackMode.DYNAMIC
                if not has_tempo:
                    self.session_state.warnings.append("No tempo metadata found")
                if not has_time_sig:
                    self.session_state.warnings.append("No time signature metadata found")
                print(f"⚠️ Switching to {self.session_state.mode} mode due to missing metadata")

            # Rule 2: Only one tempo data at 120 BPM with no key signature → dynamic mode
            elif (not metadata.get('key_signature') and
                  len(metadata.get('tempo_changes', [])) == 1 and
                  self.tempo == 120.0
                  ):
                self.session_state.mode = PlaybackMode.DYNAMIC
                self.session_state.warnings.append("suspicious MIDI data")

            else:
                self.session_state.mode = PlaybackMode.STANDARD
            
            # Clear cached metadata to force reload
            self._cached_metadata = None

            if progress_callback:
                progress_callback("Processing track information and section analysis...")

            # Display file loading summary
            print(f" File loaded: {os.path.basename(file_path)}")
            
            # Add warnings if any
            if self.session_state.warnings:
                print(f"   Warnings: {', '.join(self.session_state.warnings)}")
            
            if progress_callback:
                progress_callback("Almost ready...")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading MIDI file: {e}")
            # Even if loading fails, ensure we're in a clean state
            self.is_playing = False
            self.is_paused = False
            self.current_position = 0
            self.all_notes_off()
            self.muted_tracks.clear()
            self.active_notes.clear()
            if hasattr(self, 'precision_timer'):
                self.precision_timer.stop()
            return False
    
    def play(self, section=None):
        """Play MIDI file starting from the specified section (unified playback method)
        
        Args:
            section: Section dict to play, or None to play from current position
        """
        if not self.midi_file:
            return
        
        # Ensure synthesizer is ready before starting playback (best-effort)
        _fs_ok = self.ensure_fs_ready()
        if not _fs_ok:
            print("[MidiEngine] ⚠️ WARNING: FluidSynth not ready, continuing with MIDI out only (no audio synthesis)")
        
        # If resuming from pause, maintain current section context
        if self.is_paused:
            self.is_paused = False
            self.is_playing = True
            
            # If user has set a preferred tempo while paused, apply it
            if self.session_state.user_preferred_tempo > 0:
                self.tempo = self.session_state.user_preferred_tempo
                self.precision_timer.set_tempo(self.tempo)
            
            # Resume precision timer first (authoritative time source)
            self.precision_timer.resume()
            
            # Sync engine position to precision timer position immediately
            position_info = self.precision_timer.get_position_info()
            self.current_position = position_info['seconds']
            
            # Verify section boundaries if current section has an end
            if self.current_section and self._section_end_beat:
                current_beats = position_info['beats']
                if current_beats >= self._section_end_beat:
                    # print(f"Resume: Reached end of section at beat {current_beats:.2f}")
                    self.stop()
                    return
            
            return
        
        # Fresh start - initialize section playback
        if section is not None:
            self._setup_section_playback(section)
        else:
            # No section specified - get default first section
            metadata = self.get_metadata()
            sections = metadata.get('sections', [])
            if sections:
                self._setup_section_playback(sections[0])
            else:
                print("No sections available, cannot start playback")
                return
        
        # Start playback
        self.is_playing = True
        self.is_paused = False
        
        # Start precision timer at section start position (standard position, no offset)
        self.precision_timer.start(self._section_start_beat)
        
        # Immediately sync engine position to precision timer
        position_info = self.precision_timer.get_position_info()
        self.current_position = position_info['seconds']
        
        # Set timing offset in scheduler and start it
        # MIDI events will be delayed/advanced by offset amount
        if self.scheduler:
            offset = self.session_state.anacrusis_offset_beats
            self.scheduler.set_timing_offset(offset)
            self.scheduler.start()
            
    def _setup_section_playback(self, section: Dict):
        """Setup engine for section-specific playback parameters
        
        Args:
            section: Section dictionary with playback parameters
        """
        # Store current section and boundaries
        self.current_section = section
        self._section_start_beat = section.get('start_beat', 0.0)
        self._section_end_beat = section.get('end_beat')  # May be None for last section
        
        # Configure engine for section-specific parameters
        time_sig = section.get('time_signature', {'numerator': 4, 'denominator': 4})
        
        # Set time signature for timing system
        self.precision_timer.set_time_signature(
            time_sig.get('numerator', 4), 
            time_sig.get('denominator', 4)
        )
        
        # Get default section tempo first (we'll need it for logging even if we don't use it)
        section_tempo = section.get('tempo', 120.0)
        
        # Check for user preferred tempo in session state
        if hasattr(self, 'session_state') and self.session_state.user_preferred_tempo > 0:
            # Use the user's preferred tempo
            self.tempo = self.session_state.user_preferred_tempo
        else:
            # Otherwise use the section's default tempo
            self.tempo = section_tempo
        
        # Apply tempo to precision timer
        self.precision_timer.set_tempo(self.tempo)
        
        # Set current position to section start (in seconds)
        self.current_position = (self._section_start_beat * 60.0) / self.tempo
        
        # Log section setup
        section_label = section.get('label', 'Unknown Section')
        end_info = f" to {self._section_end_beat:.2f}" if self._section_end_beat else " (to end)"
        print(f"Setup section playback: '{section_label}' from beat {self._section_start_beat:.2f}{end_info}")
        
        # Tell scheduler to play only this section, but don't apply position yet
        # The position will be set independently in the play() method to account for anacrusis
        if self.scheduler:
            self.scheduler.prepare_section(section)  # Only prepare, don't start

    def pause(self):
        """Pause playback with section-aware synchronized timing"""
        if self.is_playing:
            self.is_playing = False
            self.is_paused = True
            
            # Pause precision timer first (authoritative source)
            self.precision_timer.pause()
            
            # Sync engine position to precision timer position immediately
            position_info = self.precision_timer.get_position_info()
            self.current_position = position_info['seconds']
            
            self.all_notes_off()  # Stop all notes

    def stop(self):
        """Stop playback and reset position"""
        self.is_playing = False
        self.is_paused = False
        
        # Stop precision timer first
        self.precision_timer.stop()
        
        # Reset position to current section start (or zero if no section)
        if self.current_section:
            self.current_position = (self._section_start_beat * 60.0) / self.tempo           
        else:
            self.current_position = 0
        
        self.all_notes_off()  # Stop all notes
    
    def update_anacrusis_offset(self, offset_beats: float):
        """Update anacrusis offset (can be adjusted during playback)
        
        In the single-clock architecture, updating the offset during playback requires
        restarting playback to apply the new event time adjustments correctly.
        
        Args:
            offset_beats: New anacrusis offset in beats (positive value delays MIDI events)
        """
        old_offset = self.session_state.anacrusis_offset_beats
        self.session_state.anacrusis_offset_beats = max(0.0, offset_beats)
        
        if self.scheduler:
            self.scheduler.set_timing_offset(offset_beats)
            
        # Log the update (only if significant change)
        if abs(old_offset - offset_beats) > 0.001 and self.is_playing:
            print(f"[MidiEngine] ⚠️ Anacrusis offset changed during playback: {old_offset:.3f} -> {offset_beats:.3f} beats (restart may be needed)")
    
    def compute_beat_offset(self, detection_result: Dict) -> Tuple[float, Dict]:
        """
            Computes the precise timing correction offset in beats.
            It compares the detected first downbeat's time with its theoretical grid position.

        Args:
            detection_result: Output from AnacrusisDetector.

        Returns:
            A tuple of (timing_correction_beats, reason_dict).
        """
        # 1. Get necessary data and compute average beat interval
        beats = detection_result.get('beats', [])
        if beats is None or len(beats) < 2:
            return 0.0, {"reason": "insufficient beats"}
        
        avg_beat_interval = np.mean(np.diff(beats))  # seconds per beat
        if avg_beat_interval <= 0:
            return 0.0, {"reason": "invalid beat interval"}
        
        first_downbeat_time = detection_result.get('first_downbeat_time')
        if first_downbeat_time is None:
            return 0.0, {"reason": "no downbeat time in result"}
        
        # effective_anacrusis_beats = detection_result.get('effective_anacrusis_beats', 0)
        meter = detection_result.get('meter', 4)
        
        # 2. Determine the THEORETICAL grid position of the first downbeat
        bar_duration_sec = meter * avg_beat_interval

        # Calculate how many full bars fit into the detected time, round to nearest whole number
        num_bars = round(first_downbeat_time / bar_duration_sec)
        ideal_downbeat_time = num_bars * bar_duration_sec
        
        # 3. calculate timing correction
        # (the difference between the detected reality and the theoretical ideal.)
        time_delta_seconds = ideal_downbeat_time - first_downbeat_time
        timing_correction_beats = time_delta_seconds / avg_beat_interval
        print(f"raw offset calculated: {timing_correction_beats}")
        
        # 4. Sanity checks, ignore tiny offset which's from audio analysis noise
        if abs(timing_correction_beats) < 0.2:
            return 0.0, {"reason": "tiny offset ignored", "calculated_offset": timing_correction_beats}
        
        # 5. snap offset to the nearest beat grid (fractional, current resolution: 1/4 beat)
        original_offset = timing_correction_beats
        quantization_grid = 0.25    # snap to 1/4 beat
        snapped_offset = round(timing_correction_beats / quantization_grid) * quantization_grid 

        # Handle offsets larger than one measure
        if abs(snapped_offset) > meter:
            # wrap offset into the range of a single bar
            final_offset = snapped_offset % meter
            # re-snap to grid
            final_offset = round(final_offset / quantization_grid) * quantization_grid
            
            return final_offset, {
                "reason": "offset_wrapped_and_calculated",
                "ideal_downbeat_time": ideal_downbeat_time,
                "detected_downbeat_time": first_downbeat_time,
                "original_offset": original_offset,
                "snapped_offset": snapped_offset,
                "final_offset": final_offset
            }
    
        # if not an extreme offset, snapped offset is the final value
        final_offset = snapped_offset
        return final_offset, {
            "reason": "calculated",
            "ideal_downbeat_time": ideal_downbeat_time,
            "detected_downbeat_time": first_downbeat_time,
            "original offset": original_offset,
            "snapped offset": final_offset
        }
    
    def set_tempo(self, bpm: float):
        """Set playback tempo with atomic synchronization
        
        Args:
            bpm: Beats per minute
        """
        old_tempo = self.tempo
        self.tempo = max(20.0, min(300.0, bpm))  # Limit BPM range to 20-300
        
        # For DYNAMIC mode, calculate and apply playback rate
        if hasattr(self, 'session_state') and self.session_state.mode == PlaybackMode.DYNAMIC:
            if hasattr(self, 'beat_hint') and hasattr(self.beat_hint, 'set_playback_rate'):
                # If we have an estimated tempo from beat tracking
                if hasattr(self.session_state, 'estimated_tempo_bpm') and self.session_state.estimated_tempo_bpm > 0:
                    # Calculate playback rate as ratio of target tempo to estimated tempo
                    playback_rate = self.tempo / self.session_state.estimated_tempo_bpm
                    # Apply to metronome's beat timeline
                    self.beat_hint.set_playback_rate(playback_rate)
                    print(f"Dynamic mode: playback rate set to {playback_rate:.3f} (target: {self.tempo:.1f} BPM, estimated: {self.session_state.estimated_tempo_bpm:.1f} BPM)")
        
        # Update tempo in PrecisionTimer (single clock source)
        if self.is_playing or self.is_paused:
            # PrecisionTimer handles tempo continuity automatically
            self.precision_timer.set_tempo(self.tempo)
            
            # Immediately sync engine position to new tempo
            new_position_info = self.precision_timer.get_position_info()
            self.current_position = new_position_info['seconds']
            
            # No need to notify EventScheduler - it queries PrecisionTimer directly
        else:
            # Not playing - just update tempo
            self.precision_timer.set_tempo(self.tempo)
            # print(f"Tempo set to {self.tempo} BPM")
    
    def mute_track(self, track_index: int):
        """Mute or unmute specified track
        
        Args:
            track_index: Track index
        """
        if track_index in self.muted_tracks:
            self.muted_tracks.remove(track_index)
        else:
            self.muted_tracks.add(track_index)
            # Stop all active notes on this track
            if track_index in self.active_notes:
                for note, channel in self.active_notes[track_index]:
                    self._send_note_off(note, channel)
                self.active_notes[track_index] = []
    
    def _send_note_on(self, note: int, velocity: int, channel: int, track_idx: int):
        """Send note on message
        
        Args:
            note: Note number
            velocity: Velocity
            channel: MIDI channel
            track_idx: Track index
        """
        if track_idx in self.muted_tracks:
            return
        
        with self.audio_lock:  # Thread safety protection
            msg = mido.Message('note_on', note=note, velocity=velocity, channel=channel)
            self._send_midi_message_unsafe(msg)
            
            # Play note using FluidSynth
            if self.fs is not None and self.sf_id is not None and self.sf_id != -1:
                try:
                    self.fs.noteon(channel, note, velocity)
                except Exception as e:
                    print(f"[MidiEngine] ⚠️ Error sending note to FluidSynth: {e}")
            
            # Record active note
            if track_idx not in self.active_notes:
                self.active_notes[track_idx] = []
            self.active_notes[track_idx].append((note, channel))
    
    def _send_note_off(self, note: int, channel: int):
        """Send note off message
        
        Args:
            note: Note number
            channel: MIDI channel
        """
        with self.audio_lock:  # Thread safety protection
            msg = mido.Message('note_off', note=note, velocity=0, channel=channel)
            self._send_midi_message_unsafe(msg)
            
            # Stop note using FluidSynth
            if self.fs and self.sf_id is not None:
                self.fs.noteoff(channel, note)
            
            # Remove from active notes list
            self._remove_active_note(note, channel)
    
    def _remove_active_note(self, note: int, channel: int):
        """Remove specified note from active notes list
        
        Args:
            note: Note number
            channel: MIDI channel
        """
        for track_idx in list(self.active_notes.keys()):
            note_list = self.active_notes[track_idx]
            # Remove matching notes
            self.active_notes[track_idx] = [
                (n, c) for n, c in note_list if not (n == note and c == channel)
            ]
            # If track has no active notes, remove empty list
            if not self.active_notes[track_idx]:
                del self.active_notes[track_idx]
    
    def _send_midi_message(self, msg: mido.Message):
        """Send MIDI message (thread-safe version)
        
        Args:
            msg: MIDI message
        """
        with self.audio_lock:
            self._send_midi_message_unsafe(msg)
    
    def _send_midi_message_unsafe(self, msg: mido.Message):
        """Send MIDI message (not thread-safe, internal use)
        
        Args:
            msg: MIDI message
        """
        if self.midi_out:
            try:
                self.midi_out.send_message(msg.bytes())
            except Exception as e:
                print(f"Error sending MIDI message: {e}")
        
        # Handle other types of MIDI messages (like controller changes)
        if self.fs and self.sf_id is not None and not msg.is_meta:
            if msg.type == 'control_change':
                self.fs.cc(msg.channel, msg.control, msg.value)
            elif msg.type == 'program_change':
                self.fs.program_change(msg.channel, msg.program)
            elif msg.type == 'pitchwheel':
                self.fs.pitch_bend(msg.channel, msg.pitch)
    
    def all_notes_off(self):
        """Stop all notes"""
        with self.audio_lock:  # Thread safety protection
            if not self.midi_out and not self.fs:
                return
            
            # 1. Send CC 123 (all notes off) to all 16 MIDI channels
            for channel in range(16):
                msg = mido.Message('control_change', channel=channel, control=123, value=0)
                self._send_midi_message_unsafe(msg)
                
                # Stop all notes using FluidSynth
                if self.fs and self.sf_id is not None:
                    self.fs.cc(channel, 123, 0)
            
            # 2. Additionally send CC 120 (All Sound Off) for immediate silence
            for channel in range(16):
                msg = mido.Message('control_change', channel=channel, control=120, value=0)
                self._send_midi_message_unsafe(msg)
                
                if self.fs and self.sf_id is not None:
                    self.fs.cc(channel, 120, 0)
            
            # 3. Force note-off for all possible MIDI notes (0-127)
            if self.fs and self.sf_id is not None:
                for channel in range(16):
                    for note in range(128):
                        self.fs.noteoff(channel, note)

            # Clear active notes tracking
            self.active_notes.clear()
    
    def __del__(self):
        """Destructor to ensure resource cleanup"""
        self.all_notes_off()
        if self.midi_out:
            self.midi_out.close_port()
        
        # Close FluidSynth
        if self.fs:
            self.fs.delete()
    
    def get_metadata(self) -> Dict:
        """Get MIDI file metadata with intelligent tempo analysis
        
        Returns:
            Dict: Metadata dictionary with track information, tempo, etc.
        """
        if not self.midi_file:
            return {}
        
        # Use cached metadata if available
        if self._cached_metadata:
            return self._cached_metadata
        
        # Generate new metadata using analyzer
        self._cached_metadata = self.metadata_analyzer.analyze_file(
            self.midi_file, 
            current_tempo=self.tempo
        )
        
        return self._cached_metadata
    
    def get_precise_position(self) -> float:
        """Get high-precision current position in seconds
        
        Returns:
            float: Current position in seconds from PrecisionTimer
        """
        if not self.is_playing and not self.is_paused:
            return self.current_position
        
        # Always get position from authoritative PrecisionTimer
        position_info = self.precision_timer.get_position_info()
        
        # Keep engine position synchronized
        self.current_position = position_info['seconds']
        
        return position_info['seconds']

    def calculate_beat_position(self) -> tuple:
        """Calculate current measure and beat position using high-precision timer
        
        Returns:
            tuple: (measure number, beat position)
        """
        if not self.midi_file or (not self.is_playing and not self.is_paused):
            return (0, 0)
        
        # Use precision timer for accurate beat calculation
        beat_event = self.precision_timer.get_beat_position()
        return (beat_event.measure, beat_event.beat)

    def get_timing_info(self) -> dict:
        """Get comprehensive timing information
        
        Returns:
            dict: Timing information including beats, measures, precision data
        """
        if not self.precision_timer.is_running and not self.precision_timer.is_paused:
            return {
                'beats': 0.0,
                'measure': 0,
                'beat': 0,
                'is_strong_beat': True,
                'tempo': self.tempo,
                'seconds': self.current_position,
                'precision_active': False
            }
        
        position_info = self.precision_timer.get_position_info()
        position_info['precision_active'] = True
        return position_info
    
    def get_section_playback_info(self) -> Dict:
        """Get section-specific playback information
        
        Returns:
            Dict containing section playback state and timing
        """
        if not self.current_section:
            return {
                'has_section': False,
                'section': None,
                'section_position_beats': 0.0,
                'section_progress': 0.0
            }
        
        # Get current timing from precision timer
        position_info = self.precision_timer.get_position_info()
        current_beats = position_info['beats']
        
        # Calculate section-relative position
        section_position_beats = current_beats - self._section_start_beat
        
        # Calculate section progress (if end beat is known)
        section_progress = 0.0
        if self._section_end_beat:
            section_duration = self._section_end_beat - self._section_start_beat
            if section_duration > 0:
                section_progress = section_position_beats / section_duration
                section_progress = max(0.0, min(1.0, section_progress))  # Clamp to [0,1]
        
        return {
            'has_section': True,
            'section': self.current_section,
            'section_position_beats': section_position_beats,
            'section_progress': section_progress,
            'section_start_beat': self._section_start_beat,
            'section_end_beat': self._section_end_beat,
            'global_position_beats': current_beats
        }