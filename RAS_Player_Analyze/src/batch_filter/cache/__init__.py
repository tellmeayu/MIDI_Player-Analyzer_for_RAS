"""Caching layer for batch analysis results.

This package provides persistent caching using SQLite to avoid re-analyzing
unchanged MIDI files.
"""

from .library_manager import LibraryManager

__all__ = ['LibraryManager']
