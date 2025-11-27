from PyQt5.QtWidgets import QAction
from PyQt5.QtCore import QObject, pyqtSignal
import os
import sys
import subprocess
import tempfile
import time
import json
from analysis.preprocessor import create_detector_clip
from batch_filter.ui.batch_analyzer_window import BatchAnalyzerWindow


class FileResultHandler(QObject):
    """Handle anacrusis detection results via JSON file polling for thread safety and stability"""
    result_received = pyqtSignal(dict)  # Signal to emit result dict

    def __init__(self, result_path, timeout=120):
        super().__init__()
        self.result_path = result_path
        self.timeout = timeout

    def monitor_file(self):
        """Monitor for result file (runs in background thread)"""
        start_time = time.time()

        try:
            while time.time() - start_time < self.timeout:
                if not os.path.exists(self.result_path) or os.path.getsize(self.result_path) == 0:
                    time.sleep(0.2)
                    continue
                # Read the result
                try:
                    with open(self.result_path, 'r', encoding='utf-8') as f:
                        result = json.load(f)

                    # Delete the file immediately to prevent stale reads
                    try:
                        os.remove(self.result_path)
                    except Exception as e:
                        print(f"Warning: Could not delete result file: {e}")

                    # Process the result
                    if isinstance(result, dict):
                        if not result.get("ok", True):
                            # Error occurred
                            error = result.get("error", "Unknown error")
                            print(f"Detection error: {error}")
                            return

                        if result.get("apply"):
                            # User clicked Apply
                            # Emit the detection result dict for beat-based offset calculation
                            detection_result = result.get("result", result)  # Fallback to whole if no "result" key
                            self.result_received.emit(detection_result)
                        else:
                            # User declined to apply
                            pass
                    else:
                        pass  # Invalid result format, ignore
                    return

                except json.JSONDecodeError as e:
                    print(f"Error parsing result JSON (retrying): {e}")
                    time.sleep(0.2)
                    continue
                except Exception as e:
                    print(f"Error reading result file: {e}")
                    import traceback
                    traceback.print_exc()
                    return

                # Sleep briefly before checking again
                # time.sleep(0.2)

            # Timeout reached
            print(f"Timeout after {self.timeout} seconds - no result file")

        except Exception as e:
            print(f"Error in monitor_file: {e}")
            import traceback
            traceback.print_exc()
        finally:
            pass  # File monitoring completed

class MenuManager:
    """Manages the application menu bar"""
    
    def __init__(self, main_window):
        """Initialize menu manager
        
        Args:
            main_window: Reference to the main window instance
        """
        self.main_window = main_window
    
    def setup_menu(self):
        """Setup menu bar"""
        menubar = self.main_window.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        open_action = QAction("Open", self.main_window)
        open_action.triggered.connect(self.main_window.load_file)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        
        # Batch Analyze Library (Phase 2 - Process Isolation)
        batch_analyze_action = QAction("Batch Analyze Library...", self.main_window)
        batch_analyze_action.triggered.connect(self._on_batch_analyze)
        file_menu.addAction(batch_analyze_action)
        
        file_menu.addSeparator()
        exit_action = QAction("Exit", self.main_window)
        exit_action.triggered.connect(self.main_window.close)
        file_menu.addAction(exit_action)
        
        # Playback menu
        playback_menu = menubar.addMenu("Playback")
        play_action = QAction("Play", self.main_window)
        play_action.triggered.connect(self.main_window.play)
        playback_menu.addAction(play_action)
        pause_action = QAction("Pause", self.main_window)
        pause_action.triggered.connect(self.main_window.pause)
        playback_menu.addAction(pause_action)
        stop_action = QAction("Stop", self.main_window)
        stop_action.triggered.connect(self.main_window.stop)
        playback_menu.addAction(stop_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        # Add standalone Timing Correction (runs in separate process)
        detect_action = QAction("Timing Correction...", self.main_window)
        def _run_anacrusis_tool():
            
            # Resolve paths
            src_dir = os.path.dirname(os.path.dirname(__file__))  # .../src
            project_root = os.path.dirname(src_dir)

            midi_path = None
            meter_num = 4
            start_beat = 0.0  # Default to global start
            if getattr(self.main_window.engine, 'midi_file', None):
                midi_path = self.main_window.engine.midi_file.filename
            
            # Get section-aware start beat
            try:
                metadata = self.main_window.engine.get_metadata()
                ts = metadata.get('time_signature', {})
                if isinstance(ts, dict) and 'numerator' in ts:
                    meter_num = int(ts['numerator'])
                
                # Get current section start beat if a section is active
                current_section = getattr(self.main_window, 'sections', [])
                if current_section and hasattr(self.main_window, 'playback_controls'):
                    section_index = self.main_window.playback_controls.get_selected_section_index()
                    if 0 <= section_index < len(current_section):
                        selected_section = current_section[section_index]
                        start_beat = selected_section.get('start_beat', 0.0)

                        # Use section-specific time signature if available
                        section_ts = selected_section.get('time_signature', {})
                        if isinstance(section_ts, dict) and 'numerator' in section_ts:
                            meter_num = int(section_ts['numerator'])

            except Exception as e:
                print(f"Error getting section info: {e}")
                start_beat = 0.0

            if not midi_path:
                return

            if not os.path.isabs(midi_path):
                midi_path = os.path.abspath(os.path.join(project_root, midi_path))
            sf2_path = self.main_window.sf2_path
            if not os.path.isabs(sf2_path):
                sf2_path = os.path.abspath(os.path.join(project_root, sf2_path))

            # Create trimmed clip
            try:
                trimmed_path = create_detector_clip(midi_path, start_beat)
                midi_to_use = trimmed_path
            except Exception as e:
                print(f"Failed to create trimmed clip: {e}, falling back to full MIDI")
                midi_to_use = midi_path

            # Generate unique temporary JSON file path for results
            # result_fd, result_path = tempfile.mkstemp(suffix=".json", prefix="anacrusis_result_")
            # os.close(result_fd)  # Close the file descriptor, keep only the path
            result_path = os.path.join(
                tempfile.gettempdir(),
                f"anacrusis_result_{int(time.time()*1000)}.json"
            )

            # Build env with PYTHONPATH
            env = os.environ.copy()
            py_path = env.get("PYTHONPATH", "")
            if src_dir not in py_path.split(os.pathsep):
                env["PYTHONPATH"] = (src_dir + (os.pathsep + py_path if py_path else ""))

            # Launch the standalone tool window as a module
            cmd = [
                sys.executable,
                "-m", "ui.anacrusis_tool_window",
                "--midi", midi_to_use,
                "--sf2", sf2_path,
                "--meter", str(meter_num),
                "--out", result_path,  # Required output path
            ]
            try:
                # Launch the tool subprocess
                process = subprocess.Popen(
                    cmd,
                    cwd=project_root,
                    env=env,
                )

                # Create result handler and connect signal to main window method
                result_handler = FileResultHandler(result_path, timeout=30)
                result_handler.result_received.connect(self.main_window.on_apply_anacrusis_correction)

                # Store reference to prevent garbage collection
                self._result_handler = result_handler

                # Start monitoring in a separate thread
                from threading import Thread
                Thread(target=result_handler.monitor_file, daemon=True).start()

            except Exception as e:
                print(f"Error launching anacrusis tool: {e}")

        detect_action.triggered.connect(_run_anacrusis_tool)
        tools_menu.addAction(detect_action)
        
        # Add Rhythm Analyzer
        rhythm_analyzer_action = QAction("Rhythm Analyzer...", self.main_window)
        rhythm_analyzer_action.triggered.connect(self._on_rhythm_analyzer)
        tools_menu.addAction(rhythm_analyzer_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        visualization_action = QAction("Enable Track Visualization", self.main_window)
        visualization_action.setCheckable(True)
        visualization_action.setChecked(True)
        visualization_action.triggered.connect(self.main_window.toggle_track_visualization)
        view_menu.addAction(visualization_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self.main_window)
        help_menu.addAction(about_action)

    def _on_batch_analyze(self):
        """Open the batch analyzer window (Phase 2 - Process Isolation)
        
        This opens an independent UI window that manages a separate subprocess
        for batch MIDI analysis. The window and subprocess are completely isolated
        from the main application's playback engine.
        """
        try:
            # Create batch analyzer window (runs in main process as UI container)
            self.batch_window = BatchAnalyzerWindow(parent=self.main_window)
            
            # Connect file selection signal to main window's file loading
            self.batch_window.file_selected.connect(self._on_batch_file_selected)
            
            self.batch_window.show()
            
            # Store reference in main window to prevent garbage collection
            self.main_window.batch_analyzer_window = self.batch_window
            
        except Exception as e:
            print(f"Error opening batch analyzer: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_batch_file_selected(self, filepath):
        """Handle file selection from batch analyzer window.
        
        Args:
            filepath: Full path to the selected MIDI file
        """
        try:
            # Use the same file loading mechanism as the main UI
            from ui.utilities import handle_file_loaded
            
            # Load the file using the main window's engine
            handle_file_loaded(self.main_window, filepath)
            
            # Bring main window to front for better user experience
            self.main_window.raise_()
            self.main_window.activateWindow()
            
            # Show success message in status bar
            self.main_window.statusBar().showMessage(
                f"Loaded from batch analysis: {os.path.basename(filepath)}", 
                3000
            )
            
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(
                self.main_window, 
                "Load Error", 
                f"Failed to load file from batch analysis:\n{filepath}\n\nError: {e}"
            )
    
    def _on_rhythm_analyzer(self):
        """Handle Rhythm Analyzer menu action.
        
        Starts multi-dimensional rhythm analysis in a background thread.
        """
        # Check if a MIDI file is loaded
        if not self.main_window.engine.midi_file:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(
                self.main_window,
                "No File Loaded",
                "Please load a MIDI file before running rhythm analysis."
            )
            return
        
        # Get MIDI file path
        midi_path = self.main_window.engine.midi_file.filename
        
        # Show status message
        self.main_window.statusBar().showMessage("Starting rhythm analysis...", 2000)
        
        # Create worker and thread
        from ui.rhythm_analysis_worker import RhythmAnalysisWorker
        from PyQt5.QtCore import QThread
        
        self.rhythm_analysis_worker = RhythmAnalysisWorker(midi_path)
        self.rhythm_analysis_thread = QThread()
        
        # Move worker to thread
        self.rhythm_analysis_worker.moveToThread(self.rhythm_analysis_thread)
        
        # Connect signals
        self.rhythm_analysis_thread.started.connect(self.rhythm_analysis_worker.run)
        self.rhythm_analysis_worker.finished.connect(self.main_window.on_rhythm_analysis_finished)
        self.rhythm_analysis_worker.error.connect(self.main_window.on_rhythm_analysis_error)
        
        # Cleanup when done
        self.rhythm_analysis_worker.finished.connect(self.rhythm_analysis_thread.quit)
        self.rhythm_analysis_worker.error.connect(self.rhythm_analysis_thread.quit)
        self.rhythm_analysis_thread.finished.connect(self.rhythm_analysis_thread.deleteLater)
        
        # Store reference in main window to prevent garbage collection
        self.main_window.rhythm_analysis_worker = self.rhythm_analysis_worker
        self.main_window.rhythm_analysis_thread = self.rhythm_analysis_thread
        
        # Start thread
        self.rhythm_analysis_thread.start()