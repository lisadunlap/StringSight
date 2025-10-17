#!/usr/bin/env python3
"""
Migrate old saved results that have stringified model_response fields.

This script reads clustered_results_lightweight.jsonl files where model_response
is a stringified Python list and re-saves them with proper JSON arrays.

Usage:
    python scripts/migrate_old_results.py <results_directory>
"""

import json
import ast
import sys
from pathlib import Path
import shutil
from datetime import datetime


def migrate_results_file(file_path: Path) -> None:
    """Migrate a single results file."""
    print(f"Processing: {file_path}")

    # Create backup
    backup_path = file_path.with_suffix('.jsonl.backup')
    shutil.copy(file_path, backup_path)
    print(f"  Created backup: {backup_path}")

    # Read all lines
    lines = []
    migrated_count = 0
    error_count = 0

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)

                # Check if model_response needs migration
                if 'model_response' in data and isinstance(data['model_response'], str):
                    mr_str = data['model_response'].strip()

                    # Check if it looks like a Python list/dict
                    if mr_str.startswith('[') or mr_str.startswith('{'):
                        try:
                            # Use ast.literal_eval to safely parse Python literals
                            data['model_response'] = ast.literal_eval(mr_str)
                            migrated_count += 1
                            print(f"    Line {line_num}: Migrated model_response")
                        except (ValueError, SyntaxError) as e:
                            print(f"    Line {line_num}: WARNING - Could not parse model_response: {e}")
                            error_count += 1

                # Check if responses field needs migration (for full dataset)
                if 'responses' in data and isinstance(data['responses'], str):
                    resp_str = data['responses'].strip()

                    if resp_str.startswith('[') or resp_str.startswith('{'):
                        try:
                            data['responses'] = ast.literal_eval(resp_str)
                            migrated_count += 1
                            print(f"    Line {line_num}: Migrated responses")
                        except (ValueError, SyntaxError) as e:
                            print(f"    Line {line_num}: WARNING - Could not parse responses: {e}")
                            error_count += 1

                lines.append(json.dumps(data))

            except json.JSONDecodeError as e:
                print(f"    Line {line_num}: ERROR - Invalid JSON: {e}")
                error_count += 1
                lines.append(line.rstrip())

    # Write migrated data
    with open(file_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')

    print(f"  Migrated {migrated_count} records")
    if error_count > 0:
        print(f"  {error_count} errors encountered")
    print(f"  Migration complete!\n")


def migrate_results_directory(results_dir: Path) -> None:
    """Migrate all results files in a directory."""
    print(f"Migrating results in: {results_dir}\n")

    # Find all relevant files
    files_to_migrate = []

    for pattern in ['clustered_results_lightweight.jsonl', 'full_dataset.jsonl']:
        for file_path in results_dir.glob(f"**/{pattern}"):
            files_to_migrate.append(file_path)

    if not files_to_migrate:
        print("No files found to migrate.")
        return

    print(f"Found {len(files_to_migrate)} file(s) to migrate:\n")

    for file_path in files_to_migrate:
        try:
            migrate_results_file(file_path)
        except Exception as e:
            print(f"ERROR migrating {file_path}: {e}\n")

    print("Migration complete!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)

    if not results_dir.is_dir():
        print(f"Error: Not a directory: {results_dir}")
        sys.exit(1)

    migrate_results_directory(results_dir)
