import sys
import os
import multiprocessing
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

# Import core modules with correct paths
from core.midi_engine import MidiEngine
from core.event_scheduler import EventScheduler
from ui.gui import MainWindow

def main():
    """Main program entry point"""
    # Enable multiprocessing support for frozen executables (Windows/macOS)
    multiprocessing.freeze_support()

    # Create application
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("RAS MIDI Player")
    app.setApplicationDisplayName("RAS MIDI Analyzer & Player")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("NeuroApp RAS")
    
    # Set application icon with high-quality preference
    icon_loaded = False
    
    # Get the project root directory (parent of src)
    project_root = os.path.dirname(os.path.dirname(__file__))
    resources_dir = os.path.join(project_root, "resources")
    
    # Try high-resolution PNG first for best quality 
    hd_png_path = os.path.join(resources_dir, "ras_midi_icon_hd.png")
    if os.path.exists(hd_png_path):
        app.setWindowIcon(QIcon(hd_png_path))
        icon_loaded = True
    else:
        # Try regular PNG
        png_path = os.path.join(resources_dir, "ras_midi_icon.png")
        if os.path.exists(png_path):
            app.setWindowIcon(QIcon(png_path))
            icon_loaded = True
        else:
            # Fallback to ICO file
            ico_path = os.path.join(resources_dir, "ras_midi_icon.ico")
            if os.path.exists(ico_path):
                app.setWindowIcon(QIcon(ico_path))
                print(f"✅ ICO Application icon loaded: {ico_path}")
                icon_loaded = True
    
    if not icon_loaded:
        print("⚠️ No icon file found - using default system icon")
        print(f"   Searched in: {resources_dir}")
    
    # Initialize components
    engine = MidiEngine()
    scheduler = EventScheduler(engine)
    
    # Register scheduler back-reference in engine (for playback control)
    engine.set_scheduler(scheduler)

    # Create main window
    window = MainWindow(engine, scheduler)
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()