"""
Utility function for file loading logic from MainWindow
"""
import os
from PyQt5.QtWidgets import QApplication
from core.precision_timer import PrecisionTimer
from core.playback_mode import PlaybackMode


def handle_file_loaded(main_window, file_path):
    """Handle file loading from playback controls (refactored from MainWindow)
    Args:
        main_window: Instance of MainWindow
        file_path: Path to the MIDI file to load
    """
    print(f"[GUI] Loading: {os.path.basename(file_path)}")
    main_window.loading_dialog.show_loading()
    QApplication.processEvents()
    try:
        # === STOP ALL ACTIVE COMPONENTS BEFORE LOADING ===
        if main_window.engine.is_playing or main_window.engine.is_paused:
            main_window.stop()  # This will stop engine, scheduler, and beat hint
        if hasattr(main_window, 'scheduler'):
            main_window.scheduler.stop()
        if hasattr(main_window, 'beat_hint'):
            main_window.beat_hint.stop()
        main_window.track_visualization.clear_tracks()
        main_window.scheduler.reset_activity_monitoring()
        # === LOAD NEW FILE ===
        # Pass the loading dialog's update_status method as progress callback
        if main_window.engine.load_file(file_path, progress_callback=main_window.loading_dialog.update_status):
            # metadata = main_window.engine.get_metadata()
            main_window.file_info_display.update_metadata(main_window.engine)
            metadata = main_window.engine.get_metadata()
            main_window.sections = metadata.get('sections', [])
            main_window.playback_controls.populate_sections(main_window.sections)
            main_window.update_section_info_label()
            tracks_info = metadata.get('tracks_info', [])
            main_window.scheduler.initialize_activity_monitoring(tracks_info)
            main_window.update_track_visualization()
            engine_tempo = int(main_window.engine.tempo)
            time_sig = metadata.get('time_signature', {})
            
            # Initialize cadence from first section if available (section-aware initialization)
            if main_window.sections:
                first_section = main_window.sections[0]
                section_tempo = first_section.get('tempo', 120.0)
                section_time_signature = first_section.get('time_signature', {'numerator': 4, 'denominator': 4})
                initial_cadence = _calculate_cadence_from_section_tempo(section_tempo, section_time_signature)
            else:
                initial_cadence = _calculate_cadence_from_tempo(engine_tempo, time_sig)
                print(f"[GUI] Using global cadence initialization (no sections)")
            initial_cadence_rounded = int(round(initial_cadence))
            main_window.playback_controls.update_section_tempo(initial_cadence_rounded)
            main_window.cadence_spinbox.setValue(initial_cadence_rounded)
            print(f"GUI: Cadence initialized to {initial_cadence_rounded} steps/min")
            main_window.base_cadence_for_adjustment = float(initial_cadence_rounded)
            main_window.accumulated_cadence_percentage_change = 0.0
            main_window.cadence_confirm_button.setStyleSheet("")
            main_window.cadence_confirm_button.setText("Apply Cadence")
            if isinstance(time_sig, dict) and 'numerator' in time_sig and 'denominator' in time_sig:
                main_window.beat_hint.set_beat(time_sig['numerator'], time_sig['denominator'])
                print(f"GUI: RAS beat hint initialized with {time_sig['numerator']}/{time_sig['denominator']} time signature")
            else:
                main_window.beat_hint.set_beat(4, 4)
                print("GUI: RAS beat hint initialized with default 4/4 time signature")
            if hasattr(main_window.beat_hint, 'update_configuration'):
                main_window.beat_hint.update_configuration()
            current_volume = main_window.playback_controls.midi_volume_slider.value()
            main_window.set_midi_volume(current_volume)
            print(f"GUI: Restored MIDI volume to {current_volume}%")
            
            # Update UI for playback mode
            main_window.playback_controls.update_playback_mode(main_window.engine)
            
            # Update RAS controls visibility based on mode
            if hasattr(main_window, '_update_ras_controls_visibility'):
                main_window._update_ras_controls_visibility()
            
            # Reset dynamic mode display in DYNAMIC mode
            if main_window.engine.session_state.mode == PlaybackMode.DYNAMIC:
                main_window.playback_controls.update_total_beats(0)
                main_window.playback_controls.update_estimated_tempo(0)
            
            main_window.loading_dialog.hide_loading()
        else:
            print("❌ GUI: Failed to load new file")
            main_window.loading_dialog.hide_loading()
    except Exception as e:
        print(f"❌ GUI: Error during file loading: {e}")
        main_window.loading_dialog.hide_loading()
        raise

 
def _calculate_cadence_from_tempo(midi_tempo_bpm, time_signature):
    """Calculate cadence from MIDI tempo (convert to musical tempo first)
        Args:
            midi_tempo_bpm (float): MIDI tempo (in quarter-note BPM).
            time_signature (dict): Dict with 'numerator' and 'denominator'.
            
        Returns:
            float: Cadence in steps per minute (matches musical tempo).
    """

    if not isinstance(time_signature, dict) or 'numerator' not in time_signature or 'denominator' not in time_signature:
        return midi_tempo_bpm  # No conversion if no time signature info
    
    time_signature_tuple = (time_signature['numerator'], time_signature['denominator'])
    musical_tempo = PrecisionTimer.convert_midi_tempo_to_musical_tempo(midi_tempo_bpm, time_signature_tuple)
    cadence = musical_tempo         # Cadence matches musical tempo (1:1 relationship)
    
    return cadence


def _calculate_cadence_from_section_tempo(section_tempo, section_time_signature):
    """Calculate cadence from section-specific tempo and time signature
        Args:
            section_tempo: Section tempo in MIDI BPM
            section_time_signature: Section time signature dict
        Returns:
            float: Cadence in steps per minute (matches musical tempo)
    """

    # Convert section MIDI tempo to musical tempo using section time signature
    if isinstance(section_time_signature, dict) and 'numerator' in section_time_signature and 'denominator' in section_time_signature:
        time_signature_tuple = (section_time_signature['numerator'], section_time_signature['denominator'])
        musical_tempo = PrecisionTimer.convert_midi_tempo_to_musical_tempo(section_tempo, time_signature_tuple)
        cadence = musical_tempo         # Cadence matches musical tempo (1:1 relationship)
        
        numerator = section_time_signature['numerator']
        denominator = section_time_signature['denominator']
        # print(f"Section cadence conversion: {section_tempo:.1f} MIDI BPM → {musical_tempo:.1f} musical BPM = {cadence:.1f} steps/min")
    else:
        # Fallback to direct conversion if no time signature info
        cadence = section_tempo
        print(f"Section cadence conversion (fallback): {section_tempo:.1f} BPM → {cadence:.1f} steps/min")
    
    return cadence
    