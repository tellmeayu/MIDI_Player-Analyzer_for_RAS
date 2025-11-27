#!/usr/bin/env python3
"""
*Note: the anacrusis detection functionality is now officially reconstructed as "Timing Correction".
Anacrusis pickup beats will not be shown as part of analysis result anymore.
Due to its late redesign, multiple modules are still referencing the old name but it won't affect the user-end.*

Standalone Timing Correction Tool Window

Runs in a separate process. Loads the MIDI and runs detection, then shows a
minimal window with results and two actions:
- Apply: Print the result to stdout and exit with code 0
- Close: Exit without applying (code 1)

The parent process can capture stdout to get the result directly.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional
import numpy as np

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QThread, pyqtSignal


def run_detection(midi_path: str, sf2_path: str, meter_num: int) -> Dict[str, Any]:
    # Ensure src on sys.path
    here = os.path.abspath(os.path.dirname(__file__))
    src_dir = os.path.abspath(os.path.join(here, os.pardir))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from analysis.anacrusis_detector import AnacrusisDetector  # child process only

    detector = AnacrusisDetector()
    result = detector.detect_anacrusis(midi_path, sf2_path, [meter_num])
    return result


def _json_serialize(obj):
    """JSON serializer function that handles numpy arrays."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class DetectionWorker(QThread):
    """Background thread for anacrusis detection."""
    
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, midi_path: str, sf2_path: str, meter_num: int):
        super().__init__()
        self.midi_path = midi_path
        self.sf2_path = sf2_path
        self.meter_num = meter_num
    
    def run(self):
        try:
            result = run_detection(self.midi_path, self.sf2_path, self.meter_num)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ToolWindow(QWidget):
    def __init__(self, midi_path: str, sf2_path: str, meter_num: int, out_path: str):
        super().__init__()
        self.setWindowTitle("Anacrusis Detector")
        self.setMinimumWidth(420)

        self.midi_path = midi_path
        self.sf2_path = sf2_path
        self.meter_num = meter_num
        self.out_path = out_path

        layout = QVBoxLayout()


        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Fake progress messages
        self.progress_messages = [
            "Starting anacrusis detector...",
            "Initializing detection module...",
            "Loading MIDI file...",
            "Parsing MIDI structure...",
            "Extracting MIDI metadata...",
            "Creating sliced MIDI copy...",
            "Preparing audio synthesis engine...",
            "Synthesizing audio from MIDI...",
            "Loading synthesized audio into memory...",
            "Starting rhythm analysis...",
            "Detecting beat probabilities...",
            "Refining beat positions...",
            "Detecting downbeat probabilities...",
            "Refining downbeat positions...",
            "Validating rhythmic patterns...",
            "Identifying potential pickup beats...",
            "Evaluating consistency of downbeat patterns...",
            "Calculating confidence scores for candidates...",
            "Selecting the most likely pickup beat...",
            "Finalizing detection results...",
            "Writing results to output file...",
            "Detection complete. Preparing to display results...",
            "Ready for user action: Apply or Close."
        ]
        self.progress_index = 0
        from PyQt5.QtCore import QTimer
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self._update_progress_message)
        self.progress_timer.start(400)  # update every 2 seconds
        self._update_progress_message()

        self.details_label = QLabel("")
        self.details_label.setAlignment(Qt.AlignCenter)
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)

        self.apply_button = QPushButton("Apply")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.on_apply)
        layout.addWidget(self.apply_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.on_close)
        layout.addWidget(self.close_button)

        self.setLayout(layout)

        # Show window immediately
        self.show()

        # Initialize detection state
        # Note: we no longer rely on "effective" pickup beats to drive UI logic.
        # The tool window only shows detector outputs as reference; final timing
        # correction is computed and applied by the main application.
        self._first_downbeat_time = 0.0

        # Start detection in background thread
        self.detection_worker = DetectionWorker(midi_path, sf2_path, meter_num)
        self.detection_worker.finished.connect(self._on_detection_finished)
        self.detection_worker.error.connect(self._on_detection_error)

        # Start detection after short delay to ensure UI is responsive
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self._start_detection)

    def _update_progress_message(self):
        if self.progress_index < len(self.progress_messages):
            self.status_label.setText(self.progress_messages[self.progress_index])
            self.progress_index += 1
        else:
            # Loop last message until real result arrives
            self.status_label.setText(self.progress_messages[-1])

    def _start_detection(self):
        """Start detection in background thread."""
        self.detection_worker.start()
    
    def _on_detection_finished(self, result):
        """Handle successful detection completion."""
        # Stop fake progress when real result is ready
        self.progress_timer.stop()
        
        if result.get('errors'):
            print(f"[TOOL WINDOW] Detection failed with errors: {result.get('errors')}")
            self.status_label.setText("Detection failed")
            self.details_label.setText("\n".join(result.get('errors', [])))
            self.apply_button.setEnabled(False)
            # Write error result to file
            error_payload = {
                "ok": False,
                "apply": False,
                "error": "\n".join(result.get('errors', []))
            }
            self._write_result(error_payload)
            return

        fdb = result.get('first_downbeat_time')
        try:
            # store as float for consistent formatting
            self._first_downbeat_time = float(fdb) if fdb is not None else 0.0
        except Exception:
            self._first_downbeat_time = 0.0

        self.result = result  # Store the full detection result
        print(f"[TOOL WINDOW] Detection result: first_downbeat={self._first_downbeat_time}")

        # Display unified information. Do NOT let pickup-beats drive UI logic.
        self.status_label.setText("Detection complete")
        # Show only the detector's primary output (first downbeat).
        info_lines = [f"First downbeat (detected): {self._first_downbeat_time:.3f} s"]
        info_lines.append("Press Apply to calculate beats offset.")
        self.details_label.setText("\n".join(info_lines))

        # Always enable Apply so user may send detector output to main app for final calc
        self.apply_button.setEnabled(True)
    
    def _on_detection_error(self, error_message):
        """Handle detection error."""
        print(f"[TOOL WINDOW] Exception in detection: {error_message}")
        import traceback
        traceback.print_exc()
        self.progress_timer.stop()
        self.status_label.setText("Detection error")
        self.details_label.setText(f"Error: {error_message}")
        self.apply_button.setEnabled(False)
        # Write error result to file
        error_payload = {
            "ok": False,
            "apply": False,
            "error": f"Detection error: {error_message}"
        }
        self._write_result(error_payload)

    def _write_result(self, payload: dict):
        """Write result to JSON file atomically

        Args:
            payload: Dictionary containing result data
        """
        try:
            with open(self.out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=_json_serialize)
        except Exception as e:
            print(f"[TOOL WINDOW] Error writing result file: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    def on_apply(self):
        """Apply button clicked - send result and exit"""
        result = dict(self.result) if isinstance(self.result, dict) else {}

        payload = {
            "ok": True,
            "apply": True,
            "result": result
        }

        self._write_result(payload)

        # Exit with success code (0)
        # print("[TOOL WINDOW] Calling app.exit(0)")
        self.close()
        QApplication.instance().exit(0)
        
    def on_close(self):
        """Close without applying - write result and exit"""

        payload = {
            "ok": True,
            "apply": False,
        }

        self._write_result(payload)

        # Exit with code 1 to indicate no action taken
        # print("[TOOL WINDOW] Calling app.exit(1)")
        self.close()
        QApplication.instance().exit(1)

    def closeEvent(self, event):
        """Handle window close event (X button) - same as Close button"""
        
        # Clean up detection worker
        if hasattr(self, 'detection_worker') and self.detection_worker.isRunning():
            self.detection_worker.terminate()
            self.detection_worker.wait(1000)  # Wait up to 1 second
        
        # If we haven't written a result yet, write a "not applied" result
        if not os.path.exists(self.out_path):
            payload = {
                "ok": True,
                "apply": False,
            }
            self._write_result(payload)
        event.accept()


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone Anacrusis Detector Tool")
    parser.add_argument("--midi", required=True)
    parser.add_argument("--sf2", required=True)
    parser.add_argument("--meter", required=True, type=int)
    parser.add_argument("--out", required=True, help="Path to output JSON file for results")

    args = parser.parse_args()

    app = QApplication(sys.argv)
    w = ToolWindow(args.midi, args.sf2, args.meter, args.out)
    w.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())


