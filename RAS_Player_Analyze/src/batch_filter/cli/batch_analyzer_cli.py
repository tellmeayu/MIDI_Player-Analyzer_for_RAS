"""Batch Analyzer CLI Service.

Standalone Python CLI that scans MIDI directory, runs multi-stage analysis,
and writes progress/results to JSON files for IPC with the main UI process.

This module is designed to run as an independent subprocess, completely
isolated from the main MIDI playback engine.

Usage:
    python -m src.batch_filter.cli.batch_analyzer_cli \\
        --input /path/to/midi/folder \\
        --music-type classical \\
        --analysis-depth fast \\
        --progress /tmp/progress.json \\
        --output /tmp/result.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    from ..core import BatchProcessor
except ImportError:
    from src.batch_filter.core import BatchProcessor


class ProgressReporter:
    """Handles atomic progress reporting via JSON files."""
    
    def __init__(self, progress_file: str):
        """Initialize progress reporter.
        
        Args:
            progress_file: Path to progress JSON file
        """
        self.progress_file = progress_file
    
    def report(self, current: int, total: int, filename: str, status: str = "in_progress") -> None:
        """Report progress atomically.
        
        Args:
            current: Current file index (1-based)
            total: Total number of files
            filename: Name of current file
            status: Status message
        """
        progress = {
            "current": current,
            "total": total,
            "filename": filename,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        # Atomic write: write to temp file first, then rename
        try:
            temp_fd, temp_path = tempfile.mkstemp(suffix='.json', dir=os.path.dirname(self.progress_file) or '.')
            with os.fdopen(temp_fd, 'w') as f:
                json.dump(progress, f)
            os.replace(temp_path, self.progress_file)
        except Exception as e:
            print(f"Warning: Failed to write progress file: {e}")


class BatchAnalyzerCLI:
    """CLI service for batch MIDI analysis."""
    
    def __init__(
        self,
        input_folder: str,
        music_type: str = "classical",
        analysis_depth: str = "fast",
        progress_file: str = None,
        output_file: str = None,
    ):
        """Initialize batch analyzer CLI.
        
        Args:
            input_folder: Path to folder containing MIDI files
            music_type: Music type ('classical' or 'pop')
            analysis_depth: Analysis depth ('fast' or 'deep')
            progress_file: Path to progress JSON file
            output_file: Path to output results JSON file
        """
        self.input_folder = input_folder
        self.music_type = music_type
        self.analysis_depth = analysis_depth
        self.progress_file = progress_file
        self.output_file = output_file
        
        # Initialize reporter
        self.reporter = ProgressReporter(progress_file) if progress_file else None
        
        # Initialize processor
        self.processor = BatchProcessor(
            music_type=music_type,
            analysis_depth=analysis_depth
        )
    
    def run(self) -> Dict[str, Any]:
        """Run the batch analysis.
        
        Returns:
            Dictionary with analysis results
        """
        try:
            # 1. Scan folder/file for MIDI files
            print(f"Scanning input: {self.input_folder}")
            midi_files = self.processor.find_midi_files(self.input_folder)
            print(f"Found {len(midi_files)} MIDI files")
            
            if not midi_files:
                print("No MIDI files found")
                return {
                    "status": "completed",
                    "total": 0,
                    "results": []
                }
            
            # 2. Report initial progress
            if self.reporter:
                self.reporter.report(0, len(midi_files), "", "starting")
            
            # 3. Validate files
            print(f"Validating {len(midi_files)} files...")
            categorized = self.processor.validate_files(midi_files)
            
            # 4. Analyze files
            files_to_analyze = categorized['needs_analysis']
            print(f"Analyzing {len(files_to_analyze)} files...")
            self.processor.analyze_files(files_to_analyze)
            
            # 5. Save results
            print(f"Saving results...")
            csv_path, json_path = self.processor.save_results(os.path.dirname(self.output_file) or ".")
            
            # 6. Prepare final output
            result = {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "input_folder": self.input_folder,
                "music_type": self.music_type,
                "analysis_depth": self.analysis_depth,
                "total": len(midi_files),
                "statistics": self.processor.statistics,
                "results": self.processor.results,
                "csv_file": csv_path,
                "json_file": json_path,
            }
            
            # 7. Save final results
            if self.output_file:
                with open(self.output_file, 'w') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
                print(f"Results saved to: {self.output_file}")
            
            # 8. Report completion
            if self.reporter:
                self.reporter.report(len(midi_files), len(midi_files), "", "completed")
            
            return result
            
        except Exception as e:
            print(f"Error: {e}")
            raise


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Batch MIDI analysis service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze entire folder
  python -m src.batch_filter.cli.batch_analyzer_cli \\
      --input /path/to/midi/folder \\
      --music-type classical \\
      --analysis-depth fast \\
      --progress /tmp/progress.json \\
      --output /tmp/result.json

  # Analyze single file
  python -m src.batch_filter.cli.batch_analyzer_cli \\
      --input /path/to/single/file.mid \\
      --music-type classical \\
      --analysis-depth deep \\
      --progress /tmp/progress.json \\
      --output /tmp/result.json
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input folder path or single MIDI file path (required)'
    )
    parser.add_argument(
        '--music-type',
        default='classical',
        choices=['classical', 'pop'],
        help='Music type for analysis (default: classical)'
    )
    parser.add_argument(
        '--analysis-depth',
        default='fast',
        choices=['fast', 'deep'],
        help='Analysis depth level (default: fast)'
    )
    parser.add_argument(
        '--progress',
        required=True,
        help='Progress JSON file path (required, for polling)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output results JSON path (required)'
    )
    
    args = parser.parse_args()
    
    try:
        # Create and run CLI
        cli = BatchAnalyzerCLI(
            input_folder=args.input,
            music_type=args.music_type,
            analysis_depth=args.analysis_depth,
            progress_file=args.progress,
            output_file=args.output,
        )
        
        result = cli.run()
        
        print(f"Batch analysis completed successfully")
        print(f"Statistics: {result['statistics']}")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
