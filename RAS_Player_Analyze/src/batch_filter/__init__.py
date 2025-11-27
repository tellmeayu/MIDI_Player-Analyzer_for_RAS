"""Batch Processing Module for MIDI Library Analysis.

This package provides a complete batch analysis system with process-level
isolation, JSON-based IPC, and persistent caching.

## Architecture

- **core/**: Core analysis algorithms (TempogramAnalyzer, MeterEstimator)
- **cli/**: Command-line service (independent subprocess)
- **cache/**: Persistent result caching (SQLite database)
- **ui/**: PyQt5-based user interface (independent window)

## Usage

### Via UI (Recommended)
```python
from src.batch_filter.ui import BatchAnalyzerWindow

window = BatchAnalyzerWindow()
window.show()
```

### Via CLI (Direct)
```bash
python -m src.batch_filter.cli.batch_analyzer_cli \
    --input /path/to/midi/folder \
    --music-type classical \
    --analysis-depth fast \
    --progress /tmp/progress.json \
    --output /tmp/result.json
```

### Via Library Manager (Caching)
```python
from src.batch_filter.cache import LibraryManager

manager = LibraryManager()
uncached_files = manager.get_uncached_files('/path/to/folder')
manager.cache_result(filepath, analysis_result)
```

## Key Features

✅ Process-level isolation (no impact on main program)
✅ JSON file-based IPC (robust, debuggable)
✅ SQLite caching (avoid re-analyzing unchanged files)
✅ Real-time progress reporting (500ms polling)
✅ Independent PyQt5 window (responsive UI)
✅ Atomic file operations (no corruption)
"""

from .core import BatchProcessor, TempogramAnalyzer, MeterEstimator
from .cache import LibraryManager
from .cli import main as cli_main

__all__ = [
    'BatchProcessor',
    'TempogramAnalyzer',
    'MeterEstimator',
    'LibraryManager',
    'cli_main',
]
