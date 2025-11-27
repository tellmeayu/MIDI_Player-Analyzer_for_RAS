"""
Radar chart plotting module for 4D rhythm analysis visualization.

Provides functionality to create radar charts from AnalysisResult objects
with a color bank system for random color selection.
"""

import numpy as np
import random
import warnings
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import AnalysisResult


# Color bank: curated palette of visually distinct colors suitable for radar charts
COLOR_BANK = [
    '#4ECDC4',  # Turquoise
    '#45B7D1',  # Sky blue
    '#74B9FF',  # Bright blue
    '#00D2D3',  # Cyan
    '#0984E3',  # Deep blue
    '#5F27CD',  # Indigo blue
    '#00CEC9',  # Dark turquoise
    '#96CEB4',  # Mint green
    '#FFEAA7',  # Soft yellow
    '#00B894',  # Emerald green
    '#FDCB6E',  # Golden yellow
    '#A8E6CF',  # Pale green
    '#B8E994',  # Light green
    '#6C5CE7',  # Purple
    '#A29BFE',  # Lavender
    '#E056FD',  # Magenta
    '#DDA0DD',  # Plum
    '#DA70D6',  # Orchid
    '#EE82EE',  # Violet
    '#9370DB',  # Medium purple
    '#95A5A6',  # Blue gray
    '#34495E',  # Dark slate blue
    '#5D6D7E',  # Slate blue gray
]


def get_random_color() -> str:
    """
    Randomly select a color from the color bank.

    Returns:
        A hex color code string.
    """
    return random.choice(COLOR_BANK)


def create_radar_figure(
    result: "AnalysisResult",
    piece_name: str = "Piece",
    title: str = "Rhythm Analysis"
):
    """
    Create a radar chart figure for rhythm analysis with four dimensions.

    This function returns a matplotlib Figure object that can be embedded
    in Qt widgets or saved to file, without displaying it immediately.

    Args:
        result: AnalysisResult containing the four dimension scores.
        piece_name: Name for the piece (used in legend/labeling).
        title: Title for the plot.

    Returns:
        matplotlib.figure.Figure: The created figure object.
    """
    import matplotlib.pyplot as plt

    # Extract scores and handle None values
    scores = [
        result.beat_density,
        result.predictability,
        result.beat_salience,
        result.rhythmic_uniformity,
    ]
    
    dimension_names = ['beat_density', 'predictability', 'beat_salience', 'rhythmic_uniformity']
    missing_dims = []
    
    # Replace None values with 0.0 and collect warnings
    processed_scores = []
    for i, score in enumerate(scores):
        if score is None:
            processed_scores.append(0.0)
            missing_dims.append(dimension_names[i])
        else:
            processed_scores.append(float(score))
    
    # Issue warning if any dimensions are missing
    if missing_dims:
        warnings.warn(
            f"Missing dimension scores for: {', '.join(missing_dims)}. "
            f"Plotting with 0.0 for these dimensions.",
            UserWarning
        )
    
    # Number of variables
    categories = ['Density', 'Predictability', 'Salience', 'Uniformity']
    num_vars = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle
    
    # Initialize the spider plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Get random color from color bank
    color = get_random_color()
    
    # Complete the loop for the plot
    values = processed_scores.copy()
    values += values[:1]
    
    # Plot data and fill area
    ax.plot(angles, values, 'o-', linewidth=2, color=color, alpha=0.8)
    ax.fill(angles, values, color=color, alpha=0.25)
    
    # Fix axis to go in the right order and start at 9 o'clock
    # Offset by 180 degrees so first dimension (Density) is at 9 o'clock
    ax.set_theta_offset(np.pi)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    # Place category labels slightly outside the circle to avoid overlap (version-safe)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    # Increase padding between the circle and labels (works across matplotlib versions)
    ax.tick_params(axis='x', pad=14)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], color="grey", size=8)
    ax.set_ylim(0, 1)
    
    # Add title (increased y value to create more space between title and chart)
    ax.set_title(title, y=1.15, fontsize=14, pad=20)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig


def plot_radar_chart(
    result: "AnalysisResult",
    piece_name: str = "Piece",
    title: str = "Rhythm Analysis"
) -> None:
    """
    Create and display a radar chart for rhythm analysis with four dimensions.

    This function creates the chart and immediately displays it using plt.show().
    For embedding in Qt widgets, use create_radar_figure() instead.

    Args:
        result: AnalysisResult containing the four dimension scores.
        piece_name: Name for the piece (used in legend/labeling).
        title: Title for the plot.
    """
    import matplotlib.pyplot as plt
    fig = create_radar_figure(result, piece_name, title)
    plt.show()

