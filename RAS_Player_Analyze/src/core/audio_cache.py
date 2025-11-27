"""Audio artifact cache manager.

This module provides caching for preprocessed audio artifacts (mixed audio + clicks)
generated during beat tracking analysis. Artifacts are keyed by content hash to
enable reuse across sessions.
"""

import os
import json
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

try:
    import soundfile as sf
    _SOUNDFILE_AVAILABLE = True
except ImportError:
    _SOUNDFILE_AVAILABLE = False


class AudioCache:
    """Manages caching of audio artifacts for the audio-based player.

    Artifacts include:
    - Mixed audio (music + click overlay) as WAV
    - Beat times as numpy array
    - Metadata (tempo, regularity, etc.) as JSON

    Cache keys are derived from MIDI file hash + SoundFont hash + analysis params.
    """

    def __init__(self, cache_dir: Optional[str] = None, max_size_mb: int = 500):
        """Initialize the audio cache.

        Args:
            cache_dir: Directory for cache storage. If None, uses ~/.player_analyzer/audio_cache
            max_size_mb: Maximum cache size in megabytes (default: 500 MB)
        """
        if cache_dir is None:
            cache_dir = os.path.join(Path.home(), '.player_analyzer', 'audio_cache')

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024

        # Initialize index file
        self.index_file = self.cache_dir / 'index.json'
        self._load_index()

    def _load_index(self):
        """Load the cache index from disk."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {}

    def _save_index(self):
        """Save the cache index to disk."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)

    def _compute_cache_key(self, midi_path: str, sf2_path: Optional[str],
                           sample_rate: int = 22050) -> str:
        """Compute cache key from file hashes and analysis parameters.

        Args:
            midi_path: Path to MIDI file
            sf2_path: Path to SoundFont file (optional)
            sample_rate: Audio sample rate

        Returns:
            SHA256 hash string as cache key
        """
        hasher = hashlib.sha256()

        # Hash MIDI file content
        with open(midi_path, 'rb') as f:
            hasher.update(f.read())

        # Hash SoundFont if provided
        if sf2_path and os.path.exists(sf2_path):
            with open(sf2_path, 'rb') as f:
                hasher.update(f.read())

        # Hash analysis parameters
        hasher.update(str(sample_rate).encode())

        return hasher.hexdigest()

    def has_cache(self, midi_path: str, sf2_path: Optional[str],
                  sample_rate: int = 22050) -> bool:
        """Check if cached artifacts exist for the given inputs.

        Args:
            midi_path: Path to MIDI file
            sf2_path: Path to SoundFont file
            sample_rate: Audio sample rate

        Returns:
            True if cache exists and is valid
        """
        cache_key = self._compute_cache_key(midi_path, sf2_path, sample_rate)

        if cache_key not in self.index:
            return False

        # Verify all files exist
        entry = self.index[cache_key]
        cache_subdir = self.cache_dir / cache_key

        required_files = [
            cache_subdir / 'mixed_audio.wav',
            cache_subdir / 'beat_times.npy',
            cache_subdir / 'metadata.json'
        ]

        return all(f.exists() for f in required_files)

    def cache_audio_artifact(self, midi_path: str, sf2_path: Optional[str],
                            mixed_audio: np.ndarray, sample_rate: int,
                            beat_times: np.ndarray, metadata: Dict) -> str:
        """Cache audio artifacts for later retrieval.

        Args:
            midi_path: Path to MIDI file
            sf2_path: Path to SoundFont file
            mixed_audio: Audio array (music + clicks)
            sample_rate: Audio sample rate
            beat_times: Array of beat times in seconds
            metadata: Dictionary with tempo, regularity, etc.

        Returns:
            Cache key string
        """
        if not _SOUNDFILE_AVAILABLE:
            raise ImportError("soundfile library is required for audio caching")

        # Compute cache key
        cache_key = self._compute_cache_key(midi_path, sf2_path, sample_rate)

        # Create cache subdirectory
        cache_subdir = self.cache_dir / cache_key
        cache_subdir.mkdir(exist_ok=True)

        # Save mixed audio as WAV
        audio_path = cache_subdir / 'mixed_audio.wav'
        sf.write(str(audio_path), mixed_audio, sample_rate)

        # Save beat times as numpy array
        beat_times_path = cache_subdir / 'beat_times.npy'
        np.save(str(beat_times_path), beat_times)

        # Save metadata as JSON
        metadata_path = cache_subdir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            # Convert numpy types to native Python for JSON serialization
            serializable_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_metadata[key] = float(value)
                elif isinstance(value, dict):
                    # Handle nested dicts (like regularity)
                    serializable_metadata[key] = {
                        k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                        for k, v in value.items()
                    }
                else:
                    serializable_metadata[key] = value

            json.dump(serializable_metadata, f, indent=2)

        # Update index
        self.index[cache_key] = {
            'midi_path': midi_path,
            'sf2_path': sf2_path,
            'sample_rate': sample_rate,
            'timestamp': os.path.getmtime(midi_path),
            'size_bytes': audio_path.stat().st_size
        }
        self._save_index()

        # Check cache size and evict if necessary
        self._enforce_size_limit()

        return cache_key

    def get_cached_artifact(self, cache_key: str) -> Optional[Dict]:
        """Retrieve cached artifacts by key.

        Args:
            cache_key: Cache key string

        Returns:
            Dictionary with 'mixed_audio', 'sample_rate', 'beat_times', 'metadata'
            or None if not found
        """
        if not _SOUNDFILE_AVAILABLE:
            raise ImportError("soundfile library is required for audio caching")

        if cache_key not in self.index:
            return None

        cache_subdir = self.cache_dir / cache_key

        try:
            # Load mixed audio
            audio_path = cache_subdir / 'mixed_audio.wav'
            mixed_audio, sample_rate = sf.read(str(audio_path))

            # Load beat times
            beat_times_path = cache_subdir / 'beat_times.npy'
            beat_times = np.load(str(beat_times_path))

            # Load metadata
            metadata_path = cache_subdir / 'metadata.json'
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            return {
                'mixed_audio': mixed_audio,
                'sample_rate': sample_rate,
                'beat_times': beat_times,
                'metadata': metadata
            }

        except Exception as e:
            print(f"Warning: Failed to load cached artifact {cache_key}: {e}")
            return None

    def get_or_create_key(self, midi_path: str, sf2_path: Optional[str],
                          sample_rate: int = 22050) -> Tuple[str, bool]:
        """Get cache key and whether it exists.

        Args:
            midi_path: Path to MIDI file
            sf2_path: Path to SoundFont file
            sample_rate: Audio sample rate

        Returns:
            Tuple of (cache_key, exists)
        """
        cache_key = self._compute_cache_key(midi_path, sf2_path, sample_rate)
        exists = self.has_cache(midi_path, sf2_path, sample_rate)
        return cache_key, exists

    def _get_cache_size(self) -> int:
        """Calculate total cache size in bytes."""
        total_size = 0
        for cache_key in self.index:
            cache_subdir = self.cache_dir / cache_key
            if cache_subdir.exists():
                for file in cache_subdir.iterdir():
                    if file.is_file():
                        total_size += file.stat().st_size
        return total_size

    def _enforce_size_limit(self):
        """Remove oldest cache entries if size exceeds limit."""
        current_size = self._get_cache_size()

        if current_size <= self.max_size_bytes:
            return

        # Sort entries by timestamp (oldest first)
        sorted_entries = sorted(
            self.index.items(),
            key=lambda x: x[1].get('timestamp', 0)
        )

        # Remove oldest entries until under limit
        for cache_key, entry in sorted_entries:
            if current_size <= self.max_size_bytes:
                break

            cache_subdir = self.cache_dir / cache_key
            if cache_subdir.exists():
                # Calculate size before removal
                entry_size = sum(
                    f.stat().st_size for f in cache_subdir.iterdir() if f.is_file()
                )

                # Remove directory
                shutil.rmtree(cache_subdir)
                current_size -= entry_size

                # Remove from index
                del self.index[cache_key]
                print(f"Cache: Evicted {cache_key} ({entry_size / 1024 / 1024:.1f} MB)")

        self._save_index()

    def clear_cache(self):
        """Remove all cached artifacts."""
        for cache_key in list(self.index.keys()):
            cache_subdir = self.cache_dir / cache_key
            if cache_subdir.exists():
                shutil.rmtree(cache_subdir)

        self.index = {}
        self._save_index()
        print("Cache: Cleared all artifacts")


# Global cache instance
_cache_instance = None


def get_cache() -> AudioCache:
    """Get the global audio cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = AudioCache()
    return _cache_instance
