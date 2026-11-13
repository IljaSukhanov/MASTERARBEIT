"""Rebuild the audit CSV tables from the saved, checked provenance snapshot.

This preserves the historical audit rather than claiming to rerun numerical
experiments. Run this script with Python 3 from any working directory.
"""
from pathlib import Path
import csv
import json

DIRECTORY = Path(__file__).resolve().parent
snapshot = json.loads((DIRECTORY / 'evidence.json').read_text())
for filename, key in [
    ('orphan_candidates.csv', 'high_confidence_candidates'),
    ('additional_review_candidates.csv', 'additional_review_candidates'),
]:
    rows = snapshot[key]
    fields = sorted({field for row in rows for field in row})
    with (DIRECTORY / filename).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row['path']))
    print(f'{filename}: {len(rows)} historical audit entries')
