"""Dialog for displaying multi-dimensional rhythm analysis results.

This module provides a dialog that displays the four dimension scores
and an embedded radar chart visualization.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDialogButtonBox, QMessageBox, QGroupBox, QFormLayout, QWidget, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multi_dim_analyzer.pipeline import AnalysisResult

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


class RhythmAnalysisDialog(QDialog):
    """Dialog for displaying rhythm analysis results with radar chart."""

    def __init__(self, result: "AnalysisResult", piece_name: str, parent=None):
        """
        Initialize the dialog.

        Args:
            result: AnalysisResult from the analysis
            piece_name: Name of the piece (filename)
            parent: Parent widget
        """
        super().__init__(parent)
        self.result = result
        self.piece_name = piece_name
        self.setWindowTitle(f"Rhythm Analysis - {piece_name}")
        self.setMinimumSize(900, 700)
        self.setModal(True)

        # Set up the UI
        self.setup_ui()

    def _get_density_desc(self) -> str:
        """Get description for Beat Density score."""
        score = self.result.beat_density
        if score is None:
            return "N/A"
        if score < 0.3:
            return "Sparse - Light texture; effective unless the cadence is very slow."
        if score < 0.8:
            return "Moderate - A balanced and common rhythmic texture."
        return "Dense - A fast, busy rhythm with many notes per beat."

    def _get_predictability_desc(self) -> str:
        """Get description for Predictability score."""
        score = self.result.predictability
        if score is None:
            return "N/A"
        if score < 0.6:
            return "Syncopated - Complex, off-beat rhythm creating a groovy feel."
        if score < 0.8:
            return "Balanced - A mix of on-beat rhythms with some variation."
        return "Predictable - Simple, on-beat rhythm emphasizing the pulse."

    def _get_salience_desc(self) -> str:
        """Get description for Beat Salience score."""
        score = self.result.beat_salience
        if score is None:
            return "N/A"
        if score < 0.3:
            return "Ambiguous - The beat is weak or unclear, creating a floating feel."
        if score < 0.5:
            return "Moderate - Not strongly emphasized beats though, typically becoming clearer with metronome cues."
        return "Strong - A prominent, powerfully articulated pulse."

    def _get_uniformity_desc(self) -> str:
        """Get description for Rhythmic Uniformity score."""
        score = self.result.rhythmic_uniformity
        if score is None:
            return "N/A"
        if score < 0.4:
            return "Varied - Rhythm patterns are diverse and constantly changing."
        if score < 0.7:
            return "Moderate - Contains some repeating figures but also includes variation."
        return "Repetitive - Built on consistent, repeating patterns."

    def setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout()

        # Main content area (scores + chart)
        content_layout = QHBoxLayout()

        # Left panel: Container for scores and profile
        left_panel = QWidget()
        left_panel.setMaximumWidth(280)  # Fixed width for left panel
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)

        # Scores Group
        scores_group = QGroupBox("Dimension Scores")
        scores_layout = QVBoxLayout()
        scores_layout.setSpacing(8)

        # Add filename section (vertical layout: label on top, filename below)
        file_section = QVBoxLayout()
        file_section.setSpacing(4)  # Small spacing between label and filename
        file_label = QLabel("File:")
        file_label.setAlignment(Qt.AlignLeft)
        file_value = QLabel(self._wrap_filename(self.piece_name))
        file_value.setWordWrap(True)
        file_value.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        # Allow full panel width and enable wrapping even for long paths
        file_value.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        file_value.setMaximumWidth(260)
        file_section.addWidget(file_label)
        file_section.addWidget(file_value)
        file_section.setAlignment(Qt.AlignLeft)
        scores_layout.addLayout(file_section)
        
        # Add spacing between filename section and scores
        scores_layout.addSpacing(6)

        # Format scores with labels (all left-aligned)
        dimension_labels = {
            'beat_density': 'Beat Density',
            'predictability': 'Predictability',
            'beat_salience': 'Beat Salience',
            'rhythmic_uniformity': 'Rhythmic Uniformity'
        }

        for key, label in dimension_labels.items():
            score = getattr(self.result, key, None)
            if score is not None:
                score_text = f"{score:.3f}"
            else:
                score_text = "N/A"
            
            row_layout = QHBoxLayout()
            label_widget = QLabel(f"{label}:")
            label_widget.setAlignment(Qt.AlignLeft)
            label_widget.setMinimumWidth(100)  # Fixed width for label
            value_widget = QLabel(score_text)
            value_widget.setAlignment(Qt.AlignLeft)
            value_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            # Make score numbers bold
            font = value_widget.font()
            font.setBold(True)
            value_widget.setFont(font)
            row_layout.addWidget(label_widget)
            row_layout.addWidget(value_widget, stretch=1)
            row_layout.setAlignment(Qt.AlignLeft)
            scores_layout.addLayout(row_layout)

        scores_group.setLayout(scores_layout)
        left_panel_layout.addWidget(scores_group)

        # Rhythmic Profile Group
        profile_group = QGroupBox("Rhythmic Profile")
        profile_layout = QVBoxLayout()
        profile_layout.setSpacing(9)  # Increased spacing between dimensions

        # Build descriptions with conditional highlight flags
        bd_score = self.result.beat_density
        pred_score = self.result.predictability
        bs_score = self.result.beat_salience
        ru_score = self.result.rhythmic_uniformity

        density_highlight = False
        if bd_score is not None:
            if bd_score >= 0.8 or bd_score < 0.3:
                density_highlight = True

        items = [
            ("■ Density", self._get_density_desc(), density_highlight),
            ("■ Predictability", self._get_predictability_desc(), (pred_score is not None and pred_score < 0.6)),
            ("■ Salience", self._get_salience_desc(), (bs_score is not None and bs_score < 0.3)),
            ("■ Uniformity", self._get_uniformity_desc(), (ru_score is not None and ru_score < 0.4)),
        ]

        for i, (label, desc, highlight) in enumerate(items):
            # Add spacing before each dimension except the first one
            if i > 0:
                profile_layout.addSpacing(9)
            
            # Vertical layout for each dimension (label on top, description below)
            dimension_layout = QVBoxLayout()
            dimension_layout.setSpacing(4)  # Small spacing between label and description
            
            label_widget = QLabel(f"{label}:")
            label_widget.setAlignment(Qt.AlignLeft)
            
            desc_label = QLabel(desc)
            desc_label.setWordWrap(True)
            desc_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            # Use full panel width and allow wrapping across the entire column
            desc_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            desc_label.setMaximumWidth(260)
            # Set default color to dark gray, or red if highlighted
            if highlight:
                # Use a readable red tone for highlighted dimensions
                desc_label.setStyleSheet("color: #C62828;")
            else:
                # Default dark gray color to differentiate from labels
                desc_label.setStyleSheet("color: #615E5E;")
            
            dimension_layout.addWidget(label_widget)
            dimension_layout.addWidget(desc_label)
            dimension_layout.setAlignment(Qt.AlignLeft)
            
            profile_layout.addLayout(dimension_layout)

        profile_group.setLayout(profile_layout)
        left_panel_layout.addWidget(profile_group)

        left_panel_layout.addStretch()
        content_layout.addWidget(left_panel)

        # Right panel: Radar chart
        if _MATPLOTLIB_AVAILABLE:
            try:
                from multi_dim_analyzer.plotting import create_radar_figure

                # Create figure with fixed title (no filename)
                fig = create_radar_figure(
                    self.result,
                    self.piece_name,
                    "Rhythm Analysis"
                )

                # Embed in Qt canvas
                canvas = FigureCanvasQTAgg(fig)
                content_layout.addWidget(canvas)
                # Explicitly set stretch so right panel gets more width
                content_layout.setStretch(0, 1)  # left panel
                content_layout.setStretch(1, 3)  # right (chart)
            except Exception as e:
                # Fallback if chart creation fails
                error_label = QLabel(f"Could not generate radar chart:\n{str(e)}")
                error_label.setAlignment(Qt.AlignCenter)
                content_layout.addWidget(error_label)
        else:
            error_label = QLabel("Matplotlib not available.\nRadar chart cannot be displayed.")
            error_label.setAlignment(Qt.AlignCenter)
            content_layout.addWidget(error_label)

        layout.addLayout(content_layout)

        # Error messages (if any)
        if self.result.error_messages:
            error_group = QGroupBox("Warnings")
            error_layout = QVBoxLayout()
            for dimension, error_msg in self.result.error_messages.items():
                error_label = QLabel(f"<b>{dimension}:</b> {error_msg}")
                error_label.setWordWrap(True)
                error_layout.addWidget(error_label)
            error_group.setLayout(error_layout)
            layout.addWidget(error_group)

        # OK button
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def _wrap_filename(self, text: str) -> str:
        """
        Insert zero-width spaces to allow wrapping of long filenames/paths.
        This enables breaking at common separators and punctuation even without spaces.
        """
        if not isinstance(text, str):
            return str(text)
        zwsp = "\u200B"
        # Allow break after path and word separators
        chars = ['/', '\\\\', '_', '-', '.', ' ']
        for ch in chars:
            text = text.replace(ch, ch + zwsp)
        return text

