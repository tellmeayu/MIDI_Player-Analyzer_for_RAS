"""Library Manager - Persistent Caching for Batch Analysis Results.

This module provides SQLite-based caching to avoid re-analyzing unchanged MIDI files.

Features:
- Hash-based cache validation (MD5 + modification time)
- Incremental analysis support
- Persistent storage of analysis results
- User notes and tagging
- Query interface for cached results
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class LibraryManager:
    """Manage analysis result caching with SQLite database."""
    
    # SQL Schema version for migrations
    SCHEMA_VERSION = 1
    
    def __init__(self, db_path: str = "library_cache.db"):
        """Initialize library manager with SQLite database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database and create tables if needed."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            
            # Create tables
            self._create_tables()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize database: {e}")
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Main cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS library_cache (
                id INTEGER PRIMARY KEY,
                file_path TEXT UNIQUE NOT NULL,
                file_hash TEXT NOT NULL,
                modification_time REAL,
                analysis_timestamp TIMESTAMP,
                rhythmic_feel TEXT,
                detected_meter TEXT,
                tempo_estimate REAL,
                confidence REAL,
                has_anacrusis BOOLEAN,
                anacrusis_beats INTEGER,
                user_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Analysis results table (for detailed results)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY,
                cache_id INTEGER UNIQUE,
                analysis_json TEXT,
                file_info_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (cache_id) REFERENCES library_cache(id)
            )
        """)
        
        # User tags table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY,
                cache_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (cache_id) REFERENCES library_cache(id)
            )
        """)
        
        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON library_cache(file_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON library_cache(file_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rhythmic_feel ON library_cache(rhythmic_feel)")
        
        self.conn.commit()
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Compute MD5 hash of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            MD5 hash hex string
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            raise RuntimeError(f"Failed to compute file hash: {e}")
    
    def is_cached(self, file_path: str) -> bool:
        """Check if file is in cache with valid hash.
        
        Args:
            file_path: Path to MIDI file
            
        Returns:
            True if file is cached and unchanged, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return False
            
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT file_hash, modification_time FROM library_cache WHERE file_path = ?",
                (file_path,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return False
            
            # Verify hash and modification time
            current_hash = self.get_file_hash(file_path)
            cached_hash = row['file_hash']
            current_mtime = os.path.getmtime(file_path)
            cached_mtime = row['modification_time']
            
            return current_hash == cached_hash and current_mtime == cached_mtime
            
        except Exception as e:
            print(f"Warning: Error checking cache: {e}")
            return False
    
    def get_cached_result(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis result from cache.
        
        Args:
            file_path: Path to MIDI file
            
        Returns:
            Analysis result dict or None if not cached
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT lc.*, ar.analysis_json, ar.file_info_json
                FROM library_cache lc
                LEFT JOIN analysis_results ar ON lc.id = ar.cache_id
                WHERE lc.file_path = ?
                """,
                (file_path,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            # Reconstruct result dict
            result = {
                'file_path': row['file_path'],
                'analysis': json.loads(row['analysis_json']) if row['analysis_json'] else None,
                'file_info': json.loads(row['file_info_json']) if row['file_info_json'] else None,
                'rhythmic_feel': row['rhythmic_feel'],
                'detected_meter': row['detected_meter'],
                'tempo_estimate': row['tempo_estimate'],
                'confidence': row['confidence'],
                'has_anacrusis': row['has_anacrusis'],
                'anacrusis_beats': row['anacrusis_beats'],
                'user_notes': row['user_notes'],
                'cached_at': row['analysis_timestamp'],
            }
            
            return result
            
        except Exception as e:
            print(f"Warning: Error retrieving cached result: {e}")
            return None
    
    def cache_result(self, file_path: str, analysis: Dict[str, Any]) -> None:
        """Store analysis result in cache.
        
        Args:
            file_path: Path to MIDI file
            analysis: Analysis result dictionary
        """
        try:
            # Compute file hash and modification time
            file_hash = self.get_file_hash(file_path)
            mtime = os.path.getmtime(file_path)
            
            cursor = self.conn.cursor()
            
            # Insert or update cache entry
            cursor.execute("""
                INSERT OR REPLACE INTO library_cache
                (file_path, file_hash, modification_time, analysis_timestamp,
                 rhythmic_feel, detected_meter, tempo_estimate, confidence,
                 has_anacrusis, anacrusis_beats, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file_path,
                file_hash,
                mtime,
                datetime.now().isoformat(),
                analysis.get('classification'),
                analysis.get('detected_meter'),
                analysis.get('analytical_tempo'),
                analysis.get('confidence'),
                analysis.get('has_anacrusis', False),
                analysis.get('anacrusis_beats', 0),
                datetime.now().isoformat(),
            ))
            
            # Store full analysis results
            cache_id = cursor.lastrowid
            cursor.execute("""
                INSERT OR REPLACE INTO analysis_results (cache_id, analysis_json, file_info_json)
                VALUES (?, ?, ?)
            """, (
                cache_id,
                json.dumps(analysis),
                json.dumps(analysis.get('file_info', {})),
            ))
            
            self.conn.commit()
            
        except Exception as e:
            print(f"Warning: Error caching result: {e}")
    
    def get_uncached_files(self, folder_path: str) -> List[str]:
        """Get list of MIDI files that need analysis.
        
        Args:
            folder_path: Path to folder containing MIDI files
            
        Returns:
            List of file paths that are not cached or modified
        """
        try:
            # Find all MIDI files in folder
            midi_files = []
            folder_path = Path(folder_path)
            
            for file_path in folder_path.rglob("*.mid*"):
                if file_path.is_file():
                    midi_files.append(str(file_path))
            
            # Filter to uncached files
            uncached = []
            for file_path in midi_files:
                if not self.is_cached(file_path):
                    uncached.append(file_path)
            
            return uncached
            
        except Exception as e:
            print(f"Warning: Error getting uncached files: {e}")
            return []
    
    def clear_cache(self, folder_path: Optional[str] = None) -> None:
        """Clear cache entries.
        
        Args:
            folder_path: If provided, only clear cache for files in this folder.
                        If None, clear all cache.
        """
        try:
            cursor = self.conn.cursor()
            
            if folder_path is None:
                # Clear all
                cursor.execute("DELETE FROM tags")
                cursor.execute("DELETE FROM analysis_results")
                cursor.execute("DELETE FROM library_cache")
            else:
                # Clear folder-specific entries
                cursor.execute("""
                    DELETE FROM tags WHERE cache_id IN (
                        SELECT id FROM library_cache WHERE file_path LIKE ?
                    )
                """, (f"{folder_path}%",))
                
                cursor.execute("""
                    DELETE FROM analysis_results WHERE cache_id IN (
                        SELECT id FROM library_cache WHERE file_path LIKE ?
                    )
                """, (f"{folder_path}%",))
                
                cursor.execute("""
                    DELETE FROM library_cache WHERE file_path LIKE ?
                """, (f"{folder_path}%",))
            
            self.conn.commit()
            
        except Exception as e:
            print(f"Warning: Error clearing cache: {e}")
    
    def add_user_note(self, file_path: str, note: str) -> None:
        """Add or update user note for a file.
        
        Args:
            file_path: Path to MIDI file
            note: User note text
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                UPDATE library_cache SET user_notes = ?, updated_at = ?
                WHERE file_path = ?
            """, (note, datetime.now().isoformat(), file_path))
            
            self.conn.commit()
            
        except Exception as e:
            print(f"Warning: Error adding user note: {e}")
    
    def add_tag(self, file_path: str, tag: str) -> None:
        """Add tag to cached file.
        
        Args:
            file_path: Path to MIDI file
            tag: Tag name
        """
        try:
            cursor = self.conn.cursor()
            
            # Get cache ID
            cursor.execute("SELECT id FROM library_cache WHERE file_path = ?", (file_path,))
            row = cursor.fetchone()
            
            if row is None:
                print(f"Warning: File not in cache: {file_path}")
                return
            
            # Add tag
            cursor.execute("""
                INSERT INTO tags (cache_id, tag) VALUES (?, ?)
            """, (row['id'], tag))
            
            self.conn.commit()
            
        except Exception as e:
            print(f"Warning: Error adding tag: {e}")
    
    def get_tags(self, file_path: str) -> List[str]:
        """Get tags for a cached file.
        
        Args:
            file_path: Path to MIDI file
            
        Returns:
            List of tag names
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT tag FROM tags
                WHERE cache_id = (SELECT id FROM library_cache WHERE file_path = ?)
            """, (file_path,))
            
            return [row['tag'] for row in cursor.fetchall()]
            
        except Exception as e:
            print(f"Warning: Error getting tags: {e}")
            return []
    
    def export_to_csv(self, folder_path: str, output_path: str) -> None:
        """Export cache to CSV file.
        
        Args:
            folder_path: Folder to export cache for
            output_path: Output CSV file path
        """
        try:
            import csv
            
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT file_path, rhythmic_feel, detected_meter, tempo_estimate,
                       confidence, has_anacrusis, anacrusis_beats, user_notes,
                       analysis_timestamp
                FROM library_cache
                WHERE file_path LIKE ?
                ORDER BY file_path
            """, (f"{folder_path}%",))
            
            rows = cursor.fetchall()
            
            if not rows:
                print(f"No cache entries for folder: {folder_path}")
                return
            
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'File Path', 'Rhythmic Feel', 'Detected Meter', 'Tempo (BPM)',
                    'Confidence', 'Has Anacrusis', 'Anacrusis Beats', 'Notes', 'Cached At'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for row in rows:
                    writer.writerow({
                        'File Path': row['file_path'],
                        'Rhythmic Feel': row['rhythmic_feel'],
                        'Detected Meter': row['detected_meter'],
                        'Tempo (BPM)': row['tempo_estimate'],
                        'Confidence': row['confidence'],
                        'Has Anacrusis': 'Yes' if row['has_anacrusis'] else 'No',
                        'Anacrusis Beats': row['anacrusis_beats'],
                        'Notes': row['user_notes'],
                        'Cached At': row['analysis_timestamp'],
                    })
            
            print(f"Cache exported to: {output_path}")
            
        except Exception as e:
            print(f"Warning: Error exporting cache: {e}")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """Ensure connection is closed on deletion."""
        self.close()


__all__ = ['LibraryManager']
