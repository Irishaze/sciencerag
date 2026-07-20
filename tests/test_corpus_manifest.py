"""Sanity checks for corpus/papers/manifest.json against the actual PDF files."""

import json
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus" / "papers"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"

REQUIRED_FIELDS = {"doi", "title", "authors", "year", "path"}


def test_manifest_has_at_least_five_entries():
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert len(manifest) >= 5


def test_manifest_entries_have_required_fields_and_existing_files():
    manifest = json.loads(MANIFEST_PATH.read_text())
    seen_dois = set()
    for entry in manifest:
        assert REQUIRED_FIELDS.issubset(entry.keys())
        assert entry["doi"] not in seen_dois, f"duplicate DOI: {entry['doi']}"
        seen_dois.add(entry["doi"])

        pdf_path = CORPUS_DIR / entry["path"]
        assert pdf_path.is_file(), f"missing PDF for manifest entry: {entry['path']}"
