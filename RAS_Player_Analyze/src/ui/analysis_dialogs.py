"""Analysis dialogs for the MIDI player.

This module provides dialog mainly for Beat tracking progress dialog
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, 
    QPushButton, QDialogButtonBox, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

class BeatTrackingDialog(QDialog):
    """Dialog for beat tracking progress."""
    
    # Signal emitted when the user cancels the operation
    cancelled = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize the dialog.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Beat Tracking")
        self.setMinimumWidth(400)
        self.setModal(True)
        
        # Set up the UI
        self.setup_ui()
        
        # Initialize progress
        self.progress = 0
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress)
        
    def setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout()
        
        # Status label
        self.status_label = QLabel("Analyzing beats in MIDI file...")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Info label
        self.info_label = QLabel(
            "This may take a few moments. Beat tracking is analyzing the audio "
            "to detect beats for metronome synchronization."
        )
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)
        
        # Cancel button
        button_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        button_box.rejected.connect(self.on_cancel)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
    def start(self):
        """Start the progress animation."""
        self.progress = 0
        self.progress_bar.setValue(0)
        self.progress_timer.start(100)  # Update every 100ms
        self.show()
        
    def update_progress(self, percent=None):
        """Update the progress bar with progress information.
        
        Args:
            percent: Progress percentage (0-100). If None, uses simulated progress.
        """
        if percent is not None:
            # Use actual progress value
            self.progress = min(100, max(0, int(percent)))
            self.progress_bar.setValue(self.progress)
        else:
            # Simulated progress for backward compatibility
            if self.progress < 95:  # Max out at 95% until complete
                self.progress += 1
                self.progress_bar.setValue(self.progress)
            
    def set_status(self, status):
        """Update the status label.
        
        Args:
            status: Status text to display
        """
        self.status_label.setText(status)
        
    def on_cancel(self):
        """Handle cancel button click."""
        reply = QMessageBox.question(
            self,
            "Cancel Beat Tracking",
            "Are you sure you want to cancel beat tracking?\n\n"
            "The metronome will not be available in dynamic mode without beat tracking.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.progress_timer.stop()
            self.cancelled.emit()
            self.reject()
            
    def complete(self, success=True, message=None):
        """Complete the operation.
        
        Args:
            success: Whether the operation was successful
            message: Optional message to display
        """
        self.progress_timer.stop()
        
        if success:
            self.progress_bar.setValue(100)
            self.status_label.setText("Beat tracking complete!")
            if message:
                self.info_label.setText(message)
        else:
            self.status_label.setText("Beat tracking failed")
            if message:
                self.info_label.setText(message)
                
        # Change to OK button
        button_box = self.findChild(QDialogButtonBox)
        if button_box:
            button_box.clear()
            button_box.addButton(QDialogButtonBox.Ok)
            button_box.accepted.connect(self.accept)
