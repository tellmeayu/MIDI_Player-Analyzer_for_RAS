from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QApplication, QDesktopWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import time

class LoadingDialog(QDialog):
    """Dynamic loading dialog to show detailed progress during MIDI file preprocessing"""
    
    def __init__(self, parent=None):
        """Initialize loading dialog
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Configure dialog
        self.setWindowTitle("Loading MIDI File")
        self.setModal(True)
        self.setFixedSize(450, 150)
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        
        # Center the dialog on screen
        screen = QApplication.desktop().screenGeometry()
        dialog_size = self.geometry()
        x = (screen.width() - dialog_size.width()) // 2
        y = (screen.height() - dialog_size.height()) // 2
        self.move(x, y)
        
        # Setup layout
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title - main message
        title_label = QLabel("Processing MIDI File...")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Dynamic status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setFont(QFont("Arial", 11))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #1976D2; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        # Initialize timers
        self.start_time = None
        
        # Auto-close timer
        self.auto_close_timer = QTimer(self)
        self.auto_close_timer.setSingleShot(True)
        self.auto_close_timer.timeout.connect(self.accept)
        
        # Timer intervals
        self.min_display_time = 1.0
        self.max_display_time = 8.0
    
    def show_loading(self):
        """Show the loading dialog"""
        # print("Starting loading dialog...")  # Debug print
        self.start_time = time.time()
        
        # Set initial status
        self.status_label.setText("Initializing...")
        
        # Start auto-close timer
        self.auto_close_timer.start(int(self.max_display_time * 1000))
        
        # Show dialog
        self.show()
        QApplication.processEvents()  # Ensure UI updates
    
    def update_status(self, message: str):
        """Update the status message
        
        Args:
            message: New status message to display
        """
        self.status_label.setText(message)
        QApplication.processEvents()  # Ensure UI updates
    
    def hide_loading(self):
        """Hide the loading dialog, respecting minimum display time"""
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed < self.min_display_time:
                remaining = int((self.min_display_time - elapsed) * 1000)
                QTimer.singleShot(remaining, self.accept)
            else:
                self.accept()
        else:
            self.accept()
    
    def set_status(self, message):
        """Manually set a specific status message
        
        Args:
            message: Status message to display
        """
        self.status_label.setText(message)
        QApplication.processEvents()  # Ensure UI updates 