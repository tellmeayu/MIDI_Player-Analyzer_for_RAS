"""Batch Analyzer Window - Independent UI for Batch Analysis.

This module provides a standalone QMainWindow for batch MIDI analysis,
with real-time progress display and result viewing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QComboBox, QProgressBar, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QHeaderView, QAbstractItemView
)
from PyQt5.QtCore import Qt

try:
    from ..cache import LibraryManager
except ImportError:
    from src.batch_filter.cache import LibraryManager


class BatchAnalyzerWindow(QMainWindow):
    """Standalone QMainWindow for batch MIDI analysis."""
    
    # Signal emitted when user double-clicks a file in the results table
    file_selected = pyqtSignal(str)  # Parameter: full file path
    
    def __init__(self, parent=None):
        """Initialize batch analyzer window.
        
        Args:
            parent: Parent widget (usually MainWindow)
        """
        super().__init__(parent)
        self.setWindowTitle("Batch Analyzer - Library Management")
        self.setGeometry(100, 100, 1200, 800)
        
        # Internal state
        self.selected_folder: Optional[str] = None
        self.selected_file: Optional[str] = None
        self.analysis_mode: str = "folder"  # "folder" or "single"
        self.batch_process: Optional[subprocess.Popen] = None
        self.progress_file: Optional[str] = None
        self.result_file: Optional[str] = None
        self.progress_timer: Optional[QTimer] = None
        self.library_manager: Optional[LibraryManager] = None
        self.results_data: dict = {}
        
        # Animation state
        self.animation_timer: Optional[QTimer] = None
        self.animation_dots: str = ""
        self.base_progress_text: str = ""
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Input Selection Panel
        input_group = QGroupBox("Input Selection")
        input_layout = QVBoxLayout()

        # Single input row for both folder and file selection
        input_row_layout = QHBoxLayout()
        input_row_layout.addWidget(QLabel("MIDI Select:"))

        self.input_label = QLineEdit()
        self.input_label.setPlaceholderText("No input selected")
        self.input_label.setReadOnly(True)
        input_row_layout.addWidget(self.input_label)

        self.browse_folder_button = QPushButton("Browse Folder")
        self.browse_folder_button.clicked.connect(self.select_folder)
        input_row_layout.addWidget(self.browse_folder_button)

        self.browse_file_button = QPushButton("Select File")
        self.browse_file_button.clicked.connect(self.select_file)
        input_row_layout.addWidget(self.browse_file_button)

        input_layout.addLayout(input_row_layout)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Analysis Options Panel
        options_group = QGroupBox("Analysis Options")
        options_layout = QHBoxLayout()
        
        # Music Type selection
        options_layout.addWidget(QLabel("Music Type:"))
        self.music_type_combo = QComboBox()
        self.music_type_combo.addItems(["Classical", "Pop"])
        options_layout.addWidget(self.music_type_combo)
        
        options_layout.addSpacing(20)
        
        # Analysis Depth selection
        options_layout.addWidget(QLabel("Analysis Depth:"))
        self.depth_combo = QComboBox()
        self.depth_combo.addItems(["Fast", "Deep"])
        self.depth_combo.setCurrentIndex(0)  # Default to Fast
        options_layout.addWidget(self.depth_combo)
        
        # Add stretch to push buttons to the right
        options_layout.addStretch()
        
        # Control Buttons (right-aligned)
        self.start_button = QPushButton("Start Analysis")
        self.start_button.clicked.connect(self.start_analysis)
        options_layout.addWidget(self.start_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_analysis)
        self.cancel_button.setEnabled(False)
        options_layout.addWidget(self.cancel_button)
        
        options_layout.addSpacing(10)
        
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        options_layout.addWidget(self.export_button)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_analysis)
        options_layout.addWidget(self.reset_button)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Progress Panel
        progress_group = QGroupBox("Analysis Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_label = QLabel("Ready")
        self.progress_label.setStyleSheet("color: #2E8B57; font-weight: bold;")
        progress_layout.addWidget(self.progress_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Results Section
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout()
        
        # Filter info label
        self.filter_info_label = QLabel("")
        self.filter_info_label.setStyleSheet("color: #666; font-style: italic;")
        results_layout.addWidget(self.filter_info_label)
        
        # Two-column layout for results tables
        tables_layout = QHBoxLayout()
        
        # Left table: Complete Metadata Files
        left_table_group = QGroupBox("Complete Metadata Files")
        left_table_layout = QVBoxLayout()
        
        self.metadata_table = QTableWidget()
        self.metadata_table.setColumnCount(4)
        self.metadata_table.setHorizontalHeaderLabels([
            "File",
            "Category", 
            "Meter",
            "Tempo"
        ])
        
        # Configure table properties
        self._configure_table(self.metadata_table)
        left_table_layout.addWidget(self.metadata_table)
        left_table_group.setLayout(left_table_layout)
        tables_layout.addWidget(left_table_group)
        
        # Right table: Analyzed Files
        right_table_group = QGroupBox("Analyzed Files")
        right_table_layout = QVBoxLayout()
        
        self.analyzed_table = QTableWidget()
        self.analyzed_table.setColumnCount(5)
        self.analyzed_table.setHorizontalHeaderLabels([
            "File",
            "Category",
            "Meter est.", 
            "Tempo est.",
            "Confidence"
        ])
        
        # Configure table properties
        self._configure_table(self.analyzed_table)
        right_table_layout.addWidget(self.analyzed_table)
        right_table_group.setLayout(right_table_layout)
        tables_layout.addWidget(right_table_group)
        
        results_layout.addLayout(tables_layout)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
    
    def _configure_table(self, table: QTableWidget):
        """Configure table properties for consistent appearance."""
        # Make table read-only (not editable)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        
        # Allow row selection but not cell editing
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        
        # Set alternating row colors for better readability
        table.setAlternatingRowColors(True)
        
        # Set column widths
        header = table.horizontalHeader()
        for i in range(table.columnCount()):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        
        # Connect double-click signal to file selection handler
        table.doubleClicked.connect(self._on_file_double_clicked)
    
    def select_folder(self):
        """Open folder selection dialog."""
        folder = QFileDialog.getExistingDirectory(self, "Select MIDI Folder")
        if folder:
            self.selected_folder = folder
            self.selected_file = None  # Clear file selection
            self.input_label.setText(os.path.basename(folder))
            self.analysis_mode = "folder"

            # Count MIDI files
            midi_count = sum(1 for _ in Path(folder).rglob("*.mid*") if Path(_).is_file())
            self.progress_label.setText(f"Ready ({midi_count} MIDI files found)")

    def select_file(self):
        """Open file selection dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select MIDI File", "", "MIDI Files (*.mid *.midi);;All Files (*)"
        )
        if file_path:
            self.selected_file = file_path
            self.selected_folder = None  # Clear folder selection
            self.input_label.setText(os.path.basename(file_path))
            self.analysis_mode = "single"

            self.progress_label.setText("Ready (1 MIDI file selected)")
    
    def start_progress_animation(self, base_text: str):
        """Start the simple dot animation for progress indication."""
        self.base_progress_text = base_text
        self.animation_dots = ""
        
        if self.animation_timer:
            self.animation_timer.stop()
        
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._update_animation)
        self.animation_timer.start(300)  # Update every 300ms
    
    def stop_progress_animation(self):
        """Stop the progress animation."""
        if self.animation_timer:
            self.animation_timer.stop()
            self.animation_timer = None
    
    def _update_animation(self):
        """Update the animation dots."""
        if len(self.animation_dots) >= 6:
            self.animation_dots = ""
        else:
            self.animation_dots += "."
        
        self.progress_label.setText(f"{self.base_progress_text}{self.animation_dots}")
    
    def start_analysis(self):
        """Start batch analysis."""
        if not self.selected_folder and not self.selected_file:
            QMessageBox.warning(self, "Warning", "Please select a folder or file first")
            return

        # Determine input path based on mode
        input_path = self.selected_folder if self.analysis_mode == "folder" else self.selected_file

        # Generate temporary file paths
        progress_fd, self.progress_file = tempfile.mkstemp(suffix=".json", prefix="batch_progress_")
        os.close(progress_fd)  # Close the file descriptor, keep only the path
        result_fd, self.result_file = tempfile.mkstemp(suffix=".json", prefix="batch_result_")
        os.close(result_fd)  # Close the file descriptor, keep only the path

        # Build CLI command
        cmd = [
            sys.executable,
            "-m", "src.batch_filter.cli.batch_analyzer_cli",
            "--input", input_path,
            "--music-type", self.music_type_combo.currentText().lower(),
            "--analysis-depth", self.depth_combo.currentText().lower(),
            "--progress", self.progress_file,
            "--output", self.result_file,
        ]
        
        try:
            # Start subprocess
            self.batch_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            # Update UI
            self.start_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.browse_folder_button.setEnabled(False)
            self.browse_file_button.setEnabled(False)
            self.start_progress_animation("Analysis running")
            self.filter_info_label.setText("")
            self.metadata_table.setRowCount(0)
            self.analyzed_table.setRowCount(0)
            
            # Start progress polling
            self.progress_timer = QTimer()
            self.progress_timer.timeout.connect(self.poll_progress)
            self.progress_timer.start(500)  # Poll every 500ms
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start analysis: {e}")
    
    def poll_progress(self):
        """Poll progress file and update UI."""
        try:
            # Check if progress file exists
            if not os.path.exists(self.progress_file):
                return
            
            # Read progress
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
            
            # Ensure progress is a dictionary
            if not isinstance(progress, dict):
                print(f"ERROR: Progress file contains {type(progress)} instead of dict: {progress}")
                return
            
            current = progress.get('current', 0)
            total = progress.get('total', 1)
            
            # Update progress display with color
            self.progress_label.setStyleSheet("color: #FF8C00; font-weight: bold;")
            self.base_progress_text = f"Processing: {current}/{total}"
            
            # Check if subprocess is done
            poll_result = self.batch_process.poll()
            if poll_result is not None:
                # Process has finished
                self.progress_timer.stop()
                self.on_analysis_complete()
        
        except json.JSONDecodeError:
            pass  # File still being written
        except Exception as e:
            print(f"Error polling progress: {e}")
    
    def on_analysis_complete(self):
        """Handle analysis completion."""
        # Stop animation
        self.stop_progress_animation()
        
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.browse_folder_button.setEnabled(True)
        self.browse_file_button.setEnabled(True)
        self.export_button.setEnabled(True)
        
        try:
            # Read results
            with open(self.result_file, 'r') as f:
                self.results_data = json.load(f)
            
            # Ensure results_data is a dictionary
            if not isinstance(self.results_data, dict):
                print(f"ERROR: Results file contains {type(self.results_data)} instead of dict: {self.results_data}")
                self.progress_label.setText("Error: Invalid results format")
                self.progress_label.setStyleSheet("color: #DC143C; font-weight: bold;")
                return
            
            # Display results
            self.display_results(self.results_data)
            
            # Update label with success color
            total = self.results_data.get('statistics', {}).get('analyzed', 0)
            self.progress_label.setText(f"Complete! Analyzed {total} files")
            self.progress_label.setStyleSheet("color: #228B22; font-weight: bold;")
            
            # Cache results
            if self.selected_folder:
                self._cache_results()
        
        except Exception as e:
            self.progress_label.setText(f"Error: {e}")
            self.progress_label.setStyleSheet("color: #DC143C; font-weight: bold;")
        
        finally:
            # Clean up temporary files
            self._cleanup_temp_files()
    
    def display_results(self, results_data: dict):
        """Display results in two-column table layout."""
        try:
            # Clear both tables
            self.metadata_table.setRowCount(0)
            self.analyzed_table.setRowCount(0)
            
            results = results_data.get('results', [])
            if not isinstance(results, list):
                print(f"ERROR: results is {type(results)} instead of list: {results}")
                return
            
            # Count files by status
            metadata_files = []
            analyzed_files = []
            corrupted_count = 0
            
            for result in results:
                if not isinstance(result, dict):
                    print(f"ERROR: result item is {type(result)} instead of dict: {result}")
                    continue
                
                status = result.get('status', 'unknown')
                if status == 'complete_metadata':
                    metadata_files.append(result)
                elif status == 'analyzed':
                    analyzed_files.append(result)
                elif status == 'corrupted':
                    corrupted_count += 1
            
            # Update filter info label
            if corrupted_count > 0:
                self.filter_info_label.setText(f"Filtered out {corrupted_count} corrupted/invalid file(s)")
            else:
                self.filter_info_label.setText("")
            
            # Populate metadata table
            for result in metadata_files:
                self._add_result_to_table(self.metadata_table, result)
            
            # Populate analyzed table
            for result in analyzed_files:
                self._add_result_to_table(self.analyzed_table, result)
                
        except Exception as e:
            print(f"ERROR in display_results: {e}")
            import traceback
            traceback.print_exc()
    
    def _add_result_to_table(self, table: QTableWidget, result: dict):
        """Add a result to the specified table."""
        row = table.rowCount()
        table.insertRow(row)
        
        # File name
        filename = os.path.basename(result.get('filename', ''))
        table.setItem(row, 0, QTableWidgetItem(filename))
        
        # Analysis results
        analysis = result.get('analysis')
        if analysis and isinstance(analysis, dict):
            table.setItem(row, 1, QTableWidgetItem(
                analysis.get('classification', 'N/A')
            ))
            table.setItem(row, 2, QTableWidgetItem(
                analysis.get('detected_meter', 'N/A')
            ))
            table.setItem(row, 3, QTableWidgetItem(
                f"{analysis.get('analytical_tempo', 0):.0f}"
            ))
            
            # Add confidence column for analyzed files table
            if table == self.analyzed_table:
                confidence = analysis.get('confidence', 0)
                try:
                    confidence_val = float(confidence) if confidence != 'N/A' else 0.0
                    table.setItem(row, 4, QTableWidgetItem(
                        f"{confidence_val:.3f}"
                    ))
                except (ValueError, TypeError):
                    table.setItem(row, 4, QTableWidgetItem('N/A'))
        else:
            # No analysis data available
            table.setItem(row, 1, QTableWidgetItem('N/A'))
            table.setItem(row, 2, QTableWidgetItem('N/A'))
            table.setItem(row, 3, QTableWidgetItem('N/A'))
            
            # Add confidence column for analyzed files table
            if table == self.analyzed_table:
                table.setItem(row, 4, QTableWidgetItem('N/A'))
    
    def _cache_results(self):
        """Cache analysis results to database."""
        try:
            if not self.library_manager:
                self.library_manager = LibraryManager()
            
            results = self.results_data.get('results', [])
            if not isinstance(results, list):
                print(f"ERROR: Cannot cache results - results is {type(results)} instead of list")
                return
                
            for result in results:
                if not isinstance(result, dict):
                    print(f"ERROR: Skipping result - item is {type(result)} instead of dict")
                    continue
                    
                analysis = result.get('analysis')
                if not analysis:
                    continue  # Skip if no analysis data
                    
                if not isinstance(analysis, dict):
                    print(f"ERROR: Skipping result - analysis is {type(analysis)} instead of dict")
                    continue
                    
                filepath = result.get('filepath')
                if not filepath:
                    continue
                    
                # Add file info
                analysis['file_info'] = result.get('file_info', {})
                
                self.library_manager.cache_result(filepath, analysis)
        
        except Exception as e:
            print(f"Warning: Failed to cache results: {e}")
            import traceback
            traceback.print_exc()
    
    def cancel_analysis(self):
        """Cancel running analysis."""
        if self.batch_process:
            # Terminate gracefully
            self.batch_process.terminate()
            
            # Wait a bit, then force kill if needed
            try:
                self.batch_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.batch_process.kill()
        
        if self.progress_timer:
            self.progress_timer.stop()
        
        # Stop animation
        self.stop_progress_animation()
        
        # Update UI
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.browse_folder_button.setEnabled(True)
        self.browse_file_button.setEnabled(True)
        self.progress_label.setText("Cancelled by user")
        self.progress_label.setStyleSheet("color: #FF6347; font-weight: bold;")
        
        # Clean up temporary files
        self._cleanup_temp_files()
    
    def export_results(self):
        """Export results to file."""
        export_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "",
            "CSV Files (*.csv);;JSON Files (*.json)"
        )
        
        if not export_path:
            return
        
        try:
            if export_path.endswith('.csv'):
                self._export_to_csv(export_path)
            else:
                self._export_to_json(export_path)
            
            QMessageBox.information(self, "Success", f"Results exported to {export_path}")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export: {e}")
    
    def _export_to_csv(self, path: str):
        """Export results to CSV."""
        import csv
        
        results = self.results_data.get('results', [])
        
        with open(path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'File Name', 'Rhythmic Feel', 'Detected Meter',
                'Tempo (BPM)', 'Confidence', 'Status'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                analysis = result.get('analysis', {})
                writer.writerow({
                    'File Name': os.path.basename(result.get('filename', '')),
                    'Rhythmic Feel': analysis.get('classification', 'N/A'),
                    'Detected Meter': analysis.get('detected_meter', 'N/A'),
                    'Tempo (BPM)': f"{analysis.get('analytical_tempo', 0):.0f}",
                    'Confidence': f"{analysis.get('confidence', 0):.3f}",
                    'Status': result.get('status', 'unknown'),
                })
    
    def _export_to_json(self, path: str):
        """Export results to JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.results_data, f, indent=2, ensure_ascii=False)
    
    def _cleanup_temp_files(self):
        """Clean up temporary files used for batch analysis."""
        try:
            if hasattr(self, 'progress_file') and self.progress_file and os.path.exists(self.progress_file):
                os.remove(self.progress_file)
                self.progress_file = None
        except Exception as e:
            print(f"Warning: Failed to clean up progress file: {e}")
        
        try:
            if hasattr(self, 'result_file') and self.result_file and os.path.exists(self.result_file):
                os.remove(self.result_file)
                self.result_file = None
        except Exception as e:
            print(f"Warning: Failed to clean up result file: {e}")
    
    def reset_analysis(self):
        """Reset analysis state."""
        # Stop animation
        self.stop_progress_animation()
        
        self.progress_label.setText("Ready")
        self.progress_label.setStyleSheet("color: #2E8B57; font-weight: bold;")
        self.input_label.setText("")
        self.filter_info_label.setText("")
        self.metadata_table.setRowCount(0)
        self.analyzed_table.setRowCount(0)
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.results_data = {}
        
        # Clean up any remaining temporary files
        self._cleanup_temp_files()
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.batch_process:
            self.batch_process.terminate()
        
        if self.progress_timer:
            self.progress_timer.stop()
        
        # Stop animation
        self.stop_progress_animation()
        
        if self.library_manager:
            self.library_manager.close()
        
        # Clean up temporary files
        self._cleanup_temp_files()
        
        event.accept()
    
    def _on_file_double_clicked(self, index):
        """Handle table double-click event to load selected file."""
        # Get the table that was clicked
        sender = self.sender()
        if not isinstance(sender, QTableWidget):
            return
        
        # Get row number
        row = index.row()
        
        # Get file path from results data
        filepath = self._get_filepath_from_row(sender, row)
        
        if filepath and os.path.exists(filepath):
            # Emit signal to load the file
            self.file_selected.emit(filepath)
        else:
            QMessageBox.warning(self, "File Not Found", 
                              f"File does not exist: {filepath}")
    
    def _get_filepath_from_row(self, table, row):
        """Get full file path from table row.
        
        Args:
            table: The table widget that was clicked
            row: Row index
            
        Returns:
            Full file path or None if not found
        """
        # Get filename from first column
        filename_item = table.item(row, 0)
        if not filename_item:
            return None
        
        filename = filename_item.text()
        
        # Find corresponding result in results_data
        results = self.results_data.get('results', [])
        for result in results:
            if os.path.basename(result.get('filename', '')) == filename:
                # Return the full filepath, preferring 'filepath' over 'filename'
                return result.get('filepath') or result.get('filename')
        
        return None
