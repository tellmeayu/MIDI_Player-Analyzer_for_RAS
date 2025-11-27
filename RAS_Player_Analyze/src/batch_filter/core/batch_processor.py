"""Batch processor for MIDI files.

This module provides batch processing capabilities for analyzing MIDI files
using either TempogramAnalyzer (fast mode, MIDI only) or MeterEstimator (deep mode, synth audio)
to determine duple/triple meter classification.

The processor includes a three-stage validation system:
1. Structural integrity check (file format, size, parseability)
2. Metadata completeness check (time signature and tempo metadata)
3. Playability validation (instruments, notes, duration)

Files with complete metadata are skipped, while valid files without
complete metadata are analyzed using the selected analysis method.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pretty_midi

try:
    from .tempogram_analyzer import TempogramAnalyzer
except ImportError:
    # When running as script, use absolute imports
    from batch_filter.core.tempogram_analyzer import TempogramAnalyzer

try:
    from .meter_estimator import MeterEstimator
except ImportError:
    # When running as script, use absolute imports
    from batch_filter.core.meter_estimator import MeterEstimator


class MidiFileValidator:
    """
    A three-stage MIDI file validator for batch processing. 
        1. Structural Integrity:
            - Verifies file existence, extension, and size constraints.
            - Attempts to parse the file using pretty_midi.
            - Checks for minimum duration and presence of notes/instruments.
        2. Metadata Completeness:
            - Determines MIDI file type (Type 0 or Type 1+).
            - Extracts time signature and tempo changes.
            - If both time signature and tempo metadata are present (Type 1+), file is considered complete.
        3. Playability Validation:
            - Ensures at least one instrument contains notes.
            - Confirms valid duration.
        - 'corrupted': File is invalid or unreadable, with reason and file info.
        - 'complete_metadata': File has sufficient metadata (time signature and tempo), with file info.
        - 'needs_analysis': File is valid but lacks complete metadata, with file info.
    """

    def __init__(
        self,
        min_duration_sec: float = 15.0,
        max_file_size_mb: float = 50.0,
        min_file_size_kb: float = 1.0,
    ):
        """Initialize validator with validation parameters.
        
        Args:
            min_duration_sec: Minimum file duration in seconds
            max_file_size_mb: Maximum file size in MB
            min_file_size_kb: Minimum file size in KB
        """
        self.min_duration_sec = min_duration_sec
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.min_file_size = min_file_size_kb * 1024
    
    def validate_file(self, midi_path: str) -> Tuple[str, str, Dict[str, Any]]:
        """Validate a single MIDI file through three stages.
        
        Args:
            midi_path: Path to MIDI file
            
        Returns:
            Tuple of (status, reason, file_info)
            Status: 'corrupted', 'complete_metadata', 'needs_analysis'
        """
        file_info = {
            'filepath': midi_path,
            'filename': os.path.basename(midi_path),
            'file_size_mb': 0.0,
            'duration_sec': 0.0,
            'total_notes': 0,
            'instruments': 0,
            'midi_type': None,
            'has_time_signature': False,
            'has_tempo': False,
            'time_signatures': [],
            'tempo_changes': [],
        }
        
        # Stage 1: Structural Integrity
        try:
            # Check file exists and is readable
            if not os.path.exists(midi_path):
                return 'corrupted', 'File does not exist', file_info
                
            if not midi_path.lower().endswith(('.mid', '.midi')):
                return 'corrupted', 'Not a MIDI file', file_info
            
            # Check file size
            file_size = os.path.getsize(midi_path)
            file_info['file_size_mb'] = file_size / (1024 * 1024)
            
            if file_size < self.min_file_size:
                return 'corrupted', f'File too small ({file_size} bytes)', file_info
                
            if file_size > self.max_file_size:
                return 'corrupted', f'File too large ({file_size / (1024*1024):.1f}MB)', file_info
            
            # Attempt to parse with pretty_midi
            pm = pretty_midi.PrettyMIDI(midi_path)
            
            # Check duration
            duration = pm.get_end_time()
            file_info['duration_sec'] = duration
            
            if duration < self.min_duration_sec:
                return 'corrupted', f'File too short ({duration:.1f}s < {self.min_duration_sec}s)', file_info
            
            # Check has instruments with notes
            total_notes = sum(len(inst.notes) for inst in pm.instruments)
            file_info['total_notes'] = total_notes
            file_info['instruments'] = len(pm.instruments)
            
            if total_notes == 0:
                return 'corrupted', 'No notes found', file_info
                
            if not any(inst.notes for inst in pm.instruments):
                return 'corrupted', 'No valid instruments with notes', file_info
                
        except Exception as e:
            print(f"  DEBUG: Structural integrity failed for {os.path.basename(midi_path)}: {e}")
            return 'corrupted', f'Parse error: {str(e)[:100]}', file_info
        
        # Stage 2: Metadata Completeness Check
        midi_file = None  # Store for potential reuse in instrument validation
        try:
            # Check MIDI type first - Type 0 files have unreliable metadata
            # Access the underlying MIDI file type
            try:
                import mido
                midi_file = mido.MidiFile(midi_path)
                midi_type = midi_file.type
            except Exception:
                # Fallback: assume Type 1 if we can't determine
                midi_type = 1
            file_info['midi_type'] = midi_type
            
            # For Type 0 MIDI files, always analyze regardless of metadata
            if midi_type == 0:
                return 'needs_analysis', 'Type 0 MIDI - metadata unreliable, needs analysis', file_info
            
            # Check time signature changes
            time_signatures = []
            if hasattr(pm, 'time_signature_changes') and pm.time_signature_changes:
                for ts in pm.time_signature_changes:
                    time_signatures.append({
                        'numerator': ts.numerator,
                        'denominator': ts.denominator,
                        'time': ts.time
                    })
                file_info['has_time_signature'] = True
                file_info['time_signatures'] = time_signatures
            
            # Check tempo changes
            tempo_changes = []
            try:
                tempo_times, tempo_values = pm.get_tempo_changes()
                if len(tempo_times) > 0 and len(tempo_values) > 0:
                    for i, (time, tempo) in enumerate(zip(tempo_times, tempo_values)):
                        tempo_changes.append({
                            'time': time,
                            'tempo': tempo
                        })
                    file_info['has_tempo'] = True
                    file_info['tempo_changes'] = tempo_changes
            except Exception:
                # Some files may not have tempo information
                pass
            
            # If both time signature AND tempo exist, skip analysis (only for Type 1+ files)
            if file_info['has_time_signature'] and file_info['has_tempo']:
                print(f"  DEBUG: {os.path.basename(midi_path)} has complete metadata (ts: {file_info['has_time_signature']}, tempo: {file_info['has_tempo']}, type: {midi_type})")
                return 'complete_metadata', 'Has complete metadata (time signature and tempo)', file_info
            else:
                print(f"  DEBUG: {os.path.basename(midi_path)} needs analysis (ts: {file_info['has_time_signature']}, tempo: {file_info['has_tempo']}, type: {midi_type})")
                
        except Exception as e:
            return 'corrupted', f'Metadata check error: {str(e)[:100]}', file_info
        
        # Stage 3: Playability Validation
        try:
            # Verify at least one non-empty instrument exists
            valid_instruments = [inst for inst in pm.instruments if inst.notes]
            if not valid_instruments:
                return 'corrupted', 'No valid instruments with notes', file_info
            
            # Ensure get_end_time() returns valid duration
            if duration <= 0:
                return 'corrupted', 'Invalid duration', file_info
                
        except Exception as e:
            return 'corrupted', f'Playability check error: {str(e)[:100]}', file_info
        
        # Stage 4: Instrument Validation
        try:
            # Use the already loaded mido file from metadata check
            if midi_file is None:
                import mido
                midi_file = mido.MidiFile(midi_path)

            # Perform lightweight instrument validation directly
            is_valid, invalid_instruments = self._validate_instruments_lightweight(midi_file)

            if not is_valid:
                reasons = []
                for invalid in invalid_instruments:
                    reasons.append(f"{invalid['track']}: {invalid['reason']}")
                reason_text = "; ".join(reasons[:3])  # Limit to first 3 reasons
                if len(reasons) > 3:
                    reason_text += f" ... (+{len(reasons) - 3} more)"
                return 'corrupted', f'Invalid instruments: {reason_text}', file_info

        except Exception as e:
            return 'corrupted', f'Instrument validation error: {str(e)[:100]}', file_info
        
        # All checks passed, needs analysis
        return 'needs_analysis', 'Valid file without complete metadata', file_info

    def _validate_instruments_lightweight(self, midi_file) -> Tuple[bool, List[Dict]]:
        """
        Lightweight instrument validation for batch processing.
        Only checks bank/program combinations without full metadata analysis.

        Args:
            midi_file: mido.MidiFile object

        Returns:
            tuple: (is_valid, invalid_instruments_list)
                - is_valid: bool, True if all instruments are valid
                - invalid_instruments_list: list of dicts with invalid instrument details
        """
        invalid_instruments = []

        for track in midi_file.tracks:
            current_bank = 0  # Default bank
            current_program = 0  # Default program

            for msg in track:
                if msg.type == 'control_change' and msg.control == 0:  # Bank select MSB
                    current_bank = msg.value
                elif msg.type == 'control_change' and msg.control == 32:  # Bank select LSB
                    current_bank = (current_bank << 7) + msg.value
                elif msg.type == 'program_change':
                    current_program = msg.program

                    # Check if this bank/program combination is valid
                    if current_bank == 0 and current_program not in self._get_gm_instrument_names():
                        invalid_instruments.append({
                            'track': track.name if hasattr(track, 'name') and track.name else f'Track {midi_file.tracks.index(track)}',
                            'bank': current_bank,
                            'program': current_program,
                            'reason': f'Invalid GM program {current_program}'
                        })
                    elif current_bank != 0:
                        # Non-zero banks are considered invalid for basic GM compatibility
                        invalid_instruments.append({
                            'track': track.name if hasattr(track, 'name') and track.name else f'Track {midi_file.tracks.index(track)}',
                            'bank': current_bank,
                            'program': current_program,
                            'reason': f'Non-GM bank {current_bank}'
                        })

        is_valid = len(invalid_instruments) == 0
        return is_valid, invalid_instruments

    def _get_gm_instrument_names(self) -> Dict[int, str]:
        """Get GM instrument names mapping for validation."""
        # Import here to avoid circular imports
        try:
            from ...core.gm_instruments import GM_INSTRUMENT_NAMES
            return GM_INSTRUMENT_NAMES
        except ImportError:
            # Fallback: return basic GM program range (0-127)
            return {i: f"Program {i}" for i in range(128)}


class BatchProcessor:
    """Batch processor for MIDI file analysis with dual analysis modes."""
    
    def __init__(
        self,
        music_type: str = "classical",
        analysis_depth: str = "fast",
        duple_sensitivity: float = 1.05,
        max_analysis_duration: float = 69.6,
        tempogram_window_duration: float = 8.7,
    ):
        """Initialize batch processor.
        
        Args:
            music_type: Music type for analysis ('classical' or 'pop')
            analysis_depth: Analysis depth ('fast' or 'deep')
            duple_sensitivity: Sensitivity for duple meter detection
            max_analysis_duration: Maximum duration to analyze
            tempogram_window_duration: Window duration for tempogram analysis
        """
        self.music_type = music_type
        self.analysis_depth = analysis_depth
        self.validator = MidiFileValidator()
        
        # Initialize analyzers based on depth
        if analysis_depth == "deep":
            # Deep analysis uses MeterEstimator (madmom-based)
            self.meter_estimator = MeterEstimator()
            self.analyzer = None  # Not used in deep mode
            print(f"DEBUG: Initialized deep analysis mode with MeterEstimator")
        else:
            # Fast analysis uses TempogramAnalyzer
            self.analyzer = TempogramAnalyzer(
                max_analysis_duration=max_analysis_duration,
                tempogram_window_duration=tempogram_window_duration,
                duple_sensitivity=duple_sensitivity,
                triple_sensitivity=1.02,
                music_type=music_type,
            )
            self.meter_estimator = None  # Not used in fast mode
            print(f"DEBUG: Initialized fast analysis mode with TempogramAnalyzer")
        
        # Results storage
        self.results = []
        self.statistics = {
            'total_files': 0,
            'corrupted': 0,
            'complete_metadata': 0,
            'analyzed': 0,
            'analysis_failed': 0,
        }
    
    def find_midi_files(self, input_path: str) -> List[str]:
        """Find MIDI files in folder or return single file.
        
        Args:
            input_path: Path to folder to search or single MIDI file
            
        Returns:
            List of MIDI file paths
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Path does not exist: {input_path}")
        
        # If it's a file, check if it's a MIDI file and return it
        if input_path.is_file():
            if input_path.suffix.lower() in ['.mid', '.midi']:
                return [str(input_path)]
            else:
                raise ValueError(f"Not a MIDI file: {input_path}")
        
        # If it's a directory, find all MIDI files recursively
        midi_files = []
        for file_path in input_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.mid', '.midi']:
                midi_files.append(str(file_path))
        
        return sorted(midi_files)
    
    def validate_files(self, midi_files: List[str]) -> Dict[str, List[str]]:
        """Validate all MIDI files and categorize them.
        
        Args:
            midi_files: List of MIDI file paths
            
        Returns:
            Dictionary with categorized file lists
        """
        print(f"Validating {len(midi_files)} MIDI files...")
        
        categorized = {
            'corrupted': [],
            'complete_metadata': [],
            'needs_analysis': [],
        }
        
        for i, midi_file in enumerate(midi_files, 1):
            print(f"[{i}/{len(midi_files)}] Validating: {os.path.basename(midi_file)}")
            
            status, reason, file_info = self.validator.validate_file(midi_file)
            
            # Store result
            result = {
                'filepath': midi_file,
                'filename': os.path.basename(midi_file),
                'status': status,
                'reason': reason,
                'file_info': file_info,
                'analysis': None,
                'error': None,
            }
            
            # For complete metadata files, extract meter information for display
            if status == 'complete_metadata':
                analysis_from_metadata = self._extract_analysis_from_metadata(file_info)
                if analysis_from_metadata:
                    result['analysis'] = analysis_from_metadata
            
            self.results.append(result)
            
            categorized[status].append(midi_file)
            
            if status == 'corrupted':
                print(f"  ❌ {reason}")
            elif status == 'complete_metadata':
                print(f"  ⏭️  {reason}")
            else:
                print(f"  ✅ {reason}")
        
        # Update statistics
        self.statistics['total_files'] = len(midi_files)
        self.statistics['corrupted'] = len(categorized['corrupted'])
        self.statistics['complete_metadata'] = len(categorized['complete_metadata'])
        
        return categorized
    
    def _extract_analysis_from_metadata(self, file_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract analysis-like information from complete metadata.
        
        Args:
            file_info: File information dictionary from validation
            
        Returns:
            Analysis-like dictionary with classification, detected_meter, and analytical_tempo
        """
        try:
            # Get time signatures
            time_signatures = file_info.get('time_signatures', [])
            if not time_signatures:
                return None
            
            # Use the first time signature (or most common if multiple)
            # For simplicity, use the first one
            ts = time_signatures[0]
            numerator = ts['numerator']
            denominator = ts['denominator']
            
            # Determine classification based on numerator
            if numerator in [2, 4, 6, 12]:
                classification = 'duple'
            elif numerator in [3, 9]:
                classification = 'triple'
            else:
                classification = 'irregular'
            
            # Format detected meter
            detected_meter = f"{numerator}/{denominator}"
            
            # Get tempo (use first tempo change)
            tempo_changes = file_info.get('tempo_changes', [])
            analytical_tempo = 120.0  # default
            if tempo_changes:
                analytical_tempo = tempo_changes[0]['tempo']
            
            # Create analysis-like result
            analysis = {
                'classification': classification,
                'detected_meter': detected_meter,
                'analytical_tempo': analytical_tempo,
                'confidence': 1.0,  # Metadata is definitive
                'source': 'metadata'  # Indicate this came from metadata, not analysis
            }
            
            return analysis
            
        except Exception as e:
            print(f"  DEBUG: Failed to extract analysis from metadata: {e}")
            return None
    
    def analyze_files(self, files_to_analyze: List[str]) -> None:
        """Analyze files that need analysis.
        
        Args:
            files_to_analyze: List of MIDI file paths to analyze
        """
        if not files_to_analyze:
            print("No files need analysis.")
            return
        
        print(f"\nAnalyzing {len(files_to_analyze)} files in {self.analysis_depth} mode...")
        
        for i, midi_file in enumerate(files_to_analyze, 1):
            print(f"[{i}/{len(files_to_analyze)}] Analyzing: {os.path.basename(midi_file)}")
            
            try:
                if self.analysis_depth == "deep":
                    # Use MeterEstimator for deep analysis
                    analysis_result = self._analyze_with_meter_estimator(midi_file)
                else:
                    # Use TempogramAnalyzer for fast analysis
                    analysis_result = self._analyze_with_tempogram(midi_file)
                
                # Find the corresponding result entry and update it
                for result in self.results:
                    if result['filepath'] == midi_file:
                        result['analysis'] = analysis_result
                        result['status'] = 'analyzed'
                        print(f"  DEBUG: Updated result for {os.path.basename(midi_file)}")
                        break
                else:
                    print(f"  DEBUG: Could not find result entry for {os.path.basename(midi_file)}")
                
                # Print analysis result
                classification = analysis_result.get('classification', 'unknown')
                confidence = analysis_result.get('confidence', 0.0)
                tempo = analysis_result.get('analytical_tempo', 0.0)
                print(f"  📊 {classification} (confidence: {confidence:.3f}, tempo: {tempo:.1f} BPM)")
                
                self.statistics['analyzed'] += 1
                
            except Exception as e:
                print(f"  ❌ Analysis failed: {str(e)[:100]}")
                
                # Update result with error
                for result in self.results:
                    if result['filepath'] == midi_file:
                        result['error'] = str(e)
                        result['status'] = 'analysis_failed'
                        break
                
                self.statistics['analysis_failed'] += 1
    
    def _analyze_with_tempogram(self, midi_file: str) -> Dict[str, Any]:
        """Analyze a single file using TempogramAnalyzer (fast mode).
        
        Args:
            midi_file: Path to MIDI file
            
        Returns:
            Analysis result dictionary
        """
        # Run tempogram analysis
        analysis_result = self.analyzer.analyze(midi_file)
        
        # DEBUG: Check what we got back
        print(f"  DEBUG: tempogram analysis_result type: {type(analysis_result)}")
        if analysis_result is None:
            print(f"  DEBUG: TempogramAnalyzer.analyze() returned None for {os.path.basename(midi_file)}")
            analysis_result = {'classification': 'error', 'confidence': 0.0, 'analytical_tempo': 0.0}
        elif not isinstance(analysis_result, dict):
            print(f"  DEBUG: TempogramAnalyzer.analyze() returned {type(analysis_result)} instead of dict for {os.path.basename(midi_file)}")
            analysis_result = {'classification': 'error', 'confidence': 0.0, 'analytical_tempo': 0.0}
        
        return analysis_result
    
    def _analyze_with_meter_estimator(self, midi_file: str) -> Dict[str, Any]:
        """Analyze a single file using MeterEstimator (deep mode).
        
        Args:
            midi_file: Path to MIDI file
            
        Returns:
            Analysis result dictionary
        """
        try:
            print(f"  DEBUG: Running deep analysis with MeterEstimator for {os.path.basename(midi_file)}")
            
            # Path to SoundFont file - try multiple possible locations
            possible_sf2_paths = [
                os.path.join(os.path.dirname(__file__), "..", "..", "resources", "soundfonts", "FluidR3_GM.sf2"),
                "/Users/Sylvia_1/Documents/_Personal/_For_Campus/PROJECTS n PAPERS/Player_Analyzer_RAS/Player_Analyzer/resources/soundfonts/FluidR3_GM.sf2",
                "resources/soundfonts/FluidR3_GM.sf2"
            ]
            
            sf2_path = None
            for path in possible_sf2_paths:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    sf2_path = abs_path
                    break
            
            if not sf2_path:
                print(f"  DEBUG: SoundFont not found in any expected location, falling back to tempogram analysis")
                # Initialize tempogram analyzer for fallback
                from .tempogram_analyzer import TempogramAnalyzer
                fallback_analyzer = TempogramAnalyzer(
                    max_analysis_duration=69.6,
                    tempogram_window_duration=8.7,
                    duple_sensitivity=1.05,
                    triple_sensitivity=1.02,
                    music_type=self.music_type,
                )
                return fallback_analyzer.analyze(midi_file)
            
            print(f"  DEBUG: Using SoundFont: {sf2_path}")
            
            # Use MeterEstimator for deep analysis
            meter_result = self.meter_estimator.estimate_from_midi(
                midi_path=midi_file,
                sf2_path=sf2_path,
                music_type=self.music_type
            )
            
            print(f"  DEBUG: MeterEstimator result: {meter_result}")
            
            # Find corresponding result entry to get file_info for tempo extraction
            result_entry = next((r for r in self.results if r['filepath'] == midi_file), None)
            
            # Convert MeterEstimator result to expected format
            if meter_result and isinstance(meter_result, dict):
                # Extract relevant information
                estimated_meter = meter_result.get('estimated_meter', 'unknown')
                
                # Convert meter format (e.g., "4/4" -> classification)
                if estimated_meter != 'N/A' and '/' in str(estimated_meter):
                    numerator = int(str(estimated_meter).split('/')[0])
                    if numerator in [2, 4, 6, 12]:
                        classification = 'duple'
                    elif numerator in [3, 9]:
                        classification = 'triple'
                    else:
                        classification = 'irregular'
                else:
                    classification = 'unknown'
                
                confidence = meter_result.get('confidence', 0.5)
                
                # Extract tempo: prefer MeterEstimator's calculated tempo, fallback to file_info
                analytical_tempo = meter_result.get('tempo', None)
                if analytical_tempo is None or analytical_tempo <= 0:
                    # Fallback to file_info tempo if MeterEstimator didn't provide one
                    analytical_tempo = 120.0  # default
                    if result_entry:
                        tempo_changes = result_entry.get('file_info', {}).get('tempo_changes', [])
                        if tempo_changes:
                            analytical_tempo = tempo_changes[0]['tempo']
                
                # Ensure confidence is a float
                if isinstance(confidence, str):
                    try:
                        confidence = float(confidence)
                    except ValueError:
                        confidence = 0.5
                
                # Ensure tempo is a float
                if isinstance(analytical_tempo, str):
                    try:
                        analytical_tempo = float(analytical_tempo)
                    except ValueError:
                        analytical_tempo = 120.0
                
                analysis_result = {
                    'classification': classification,
                    'confidence': confidence,
                    'analytical_tempo': analytical_tempo,
                    'detected_meter': str(estimated_meter),
                    'source': 'meter_estimator'
                }
                
                print(f"  DEBUG: Converted MeterEstimator result: {analysis_result}")
                return analysis_result
            else:
                print(f"  DEBUG: MeterEstimator returned invalid result: {meter_result}, falling back to tempogram")
                # Initialize tempogram analyzer for fallback
                from .tempogram_analyzer import TempogramAnalyzer
                fallback_analyzer = TempogramAnalyzer(
                    max_analysis_duration=69.6,
                    tempogram_window_duration=8.7,
                    duple_sensitivity=1.05,
                    triple_sensitivity=1.02,
                    music_type=self.music_type,
                )
                return fallback_analyzer.analyze(midi_file)
                
        except Exception as e:
            print(f"  DEBUG: MeterEstimator analysis failed: {e}, falling back to tempogram")
            import traceback
            traceback.print_exc()
            
            # Fallback to tempogram analysis
            try:
                from .tempogram_analyzer import TempogramAnalyzer
                fallback_analyzer = TempogramAnalyzer(
                    max_analysis_duration=69.6,
                    tempogram_window_duration=8.7,
                    duple_sensitivity=1.05,
                    triple_sensitivity=1.02,
                    music_type=self.music_type,
                )
                return fallback_analyzer.analyze(midi_file)
            except Exception as fallback_e:
                print(f"  DEBUG: Fallback analysis also failed: {fallback_e}")
                return {'classification': 'error', 'confidence': 0.0, 'analytical_tempo': 0.0, 'source': 'fallback_failed'}
    
    def save_results(self, output_dir: str = ".") -> Tuple[str, str]:
        """Save results to CSV and JSON files.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Tuple of (csv_path, json_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV
        csv_path = os.path.join(output_dir, f"batch_analysis_results_{timestamp}.csv")
        self._save_csv(csv_path)
        
        # Save JSON
        json_path = os.path.join(output_dir, f"batch_analysis_results_{timestamp}.json")
        self._save_json(json_path)
        
        return csv_path, json_path
    
    def _save_csv(self, csv_path: str) -> None:
        """Save results to CSV file."""
        fieldnames = [
            'filename', 'status', 'classification', 'confidence', 'duple_score', 
            'triple_score', 'analytical_tempo', 'clarity', 'duration_sec', 
            'total_notes', 'midi_type', 'skip_reason', 'error'
        ]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {
                    'filename': result['filename'],
                    'status': result['status'],
                    'classification': '',
                    'confidence': '',
                    'duple_score': '',
                    'triple_score': '',
                    'analytical_tempo': '',
                    'clarity': '',
                    'duration_sec': f"{result['file_info'].get('duration_sec', 0):.1f}",
                    'total_notes': result['file_info'].get('total_notes', 0),
                    'midi_type': result['file_info'].get('midi_type', ''),
                    'skip_reason': result['reason'] if result['status'] in ['corrupted', 'complete_metadata'] else '',
                    'error': result['error'] if result['error'] else ''
                }
                
                # Add analysis results if available
                if result['analysis']:
                    analysis = result['analysis']
                    row.update({
                        'classification': analysis.get('classification', ''),
                        'confidence': f"{analysis.get('confidence', 0):.4f}",
                        'duple_score': f"{analysis.get('duple_score', 0):.4f}",
                        'triple_score': f"{analysis.get('triple_score', 0):.4f}",
                        'analytical_tempo': f"{analysis.get('analytical_tempo', 0):.1f}",
                        'clarity': f"{analysis.get('clarity', 0):.4f}",
                    })
                
                writer.writerow(row)
    
    def _save_json(self, json_path: str) -> None:
        """Save results to JSON file."""
        json_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'input_folder': '',
            'total_files': self.statistics['total_files'],
            'categories': {
                'analyzed': self.statistics['analyzed'],
                'complete_metadata': self.statistics['complete_metadata'],
                'corrupted': self.statistics['corrupted'],
                'analysis_failed': self.statistics['analysis_failed'],
            },
            'results': []
        }
        
        for result in self.results:
            json_result = {
                'filepath': result['filepath'],
                'filename': result['filename'],
                'status': result['status'],
                'validation': result['file_info'],
                'analysis': result['analysis'],
                'error': result['error'],
                'reason': result['reason'],
            }
            json_data['results'].append(json_result)
        
        def convert_numpy_types(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # Convert numpy types to Python native types
        json_data = convert_numpy_types(json_data)
        
        with open(json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(json_data, jsonfile, indent=2, ensure_ascii=False)
    
    def print_summary(self) -> None:
        """Print analysis summary."""
        print(f"\n{'='*60}")
        print("BATCH ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Total files processed: {self.statistics['total_files']}")
        print(f"Successfully analyzed: {self.statistics['analyzed']}")
        print(f"Skipped (complete metadata): {self.statistics['complete_metadata']}")
        print(f"Corrupted/invalid: {self.statistics['corrupted']}")
        print(f"Analysis failed: {self.statistics['analysis_failed']}")
        
        if self.statistics['analyzed'] > 0:
            accuracy = self.statistics['analyzed'] / (self.statistics['analyzed'] + self.statistics['analysis_failed']) * 100
            print(f"Analysis success rate: {accuracy:.1f}%")
        
        # Show some examples of each category
        print(f"\n{'-'*40}")
        print("SAMPLE RESULTS")
        print(f"{'-'*40}")
        
        for category in ['analyzed', 'complete_metadata', 'corrupted', 'analysis_failed']:
            category_results = [r for r in self.results if r['status'] == category]
            if category_results:
                print(f"\n{category.upper()} ({len(category_results)} files):")
                for result in category_results[:3]:  # Show first 3
                    if category == 'analyzed' and result['analysis']:
                        analysis = result['analysis']
                        print(f"  {result['filename']}: {analysis.get('classification', 'unknown')} "
                              f"(confidence: {analysis.get('confidence', 0):.3f})")
                    else:
                        print(f"  {result['filename']}: {result['reason']}")
                if len(category_results) > 3:
                    print(f"  ... and {len(category_results) - 3} more")


def main():
    """Command-line interface for batch tempogram processing."""
    parser = argparse.ArgumentParser(
        description="Batch tempogram analysis for MIDI files",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python batch_processor.py /path/to/midi/folder
  python batch_processor.py /path/to/midi/folder --music-type pop
  python batch_processor.py /path/to/midi/folder --output-dir /path/to/results
        """
    )
    
    parser.add_argument('folder_path', help='Path to folder containing MIDI files')
    parser.add_argument('--music-type', '-t', 
                       choices=['classical', 'pop'],
                       default='classical',
                       help='Music type for analysis (default: classical)')
    parser.add_argument('--analysis-depth', '-d',
                       choices=['fast', 'deep'],
                       default='fast',
                       help='Analysis depth (default: fast)')
    parser.add_argument('--duple-sensitivity', '-s',
                       type=float,
                       default=1.15,
                       help='Sensitivity for duple meter detection (default: 1.15)')
    parser.add_argument('--output-dir', '-o',
                       default='.',
                       help='Output directory for results (default: current directory)')
    
    args = parser.parse_args()
    
    try:
        # Create processor
        processor = BatchProcessor(
            music_type=args.music_type,
            analysis_depth=args.analysis_depth,
            duple_sensitivity=args.duple_sensitivity,
        )
        
        # Find MIDI files
        print(f"Scanning folder: {args.folder_path}")
        midi_files = processor.find_midi_files(args.folder_path)
        print(f"Found {len(midi_files)} MIDI files")
        
        if not midi_files:
            print("No MIDI files found in the specified folder.")
            return
        
        # Validate files
        categorized = processor.validate_files(midi_files)
        
        # Analyze files that need analysis
        processor.analyze_files(categorized['needs_analysis'])
        
        # Save results
        csv_path, json_path = processor.save_results(args.output_dir)
        
        # Print summary
        processor.print_summary()
        
        print(f"\n✅ Results saved:")
        print(f"  CSV: {csv_path}")
        print(f"  JSON: {json_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Add the parent directory to the path for relative imports
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    
    main()
