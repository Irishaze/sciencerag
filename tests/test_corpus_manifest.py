"""Sanity checks for corpus/papers/manifest.json against the actual PDF files.

Most of the local corpus (see .gitignore's `corpus/papers/*.pdf`) is
local-only and won't exist on a fresh clone — only the original 6 seed
papers are git-tracked. So we only hard-require those 6 to be present;
for the rest, we just check the manifest itself is well-formed (required
fields, unique DOIs) without requiring every PDF to exist on this machine.
"""

import json
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus" / "papers"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"

REQUIRED_FIELDS = {"doi", "title", "authors", "year", "path"}

# The only PDFs guaranteed to exist on any clone of the repo (not gitignored).
GIT_TRACKED_SEED_PAPERS = {
    "1-s2.0-S0140700725004438-main.pdf",
    "1-s2.0-S1359431125024950-main.pdf",
    "1-s2.0-S1359431125038165-main.pdf",
    "10.47480-isibted.1194999-2732735.pdf",
    "Neural_Network-Based_Control_of_Forced-Convection_and_Thermoelectric_Coolers.pdf",
    "PIIS240584402400094X.pdf",
}


def test_manifest_has_at_least_five_entries():
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert len(manifest) >= 5


def test_manifest_entries_have_required_fields_and_unique_dois():
    manifest = json.loads(MANIFEST_PATH.read_text())
    seen_dois = set()
    for entry in manifest:
        assert REQUIRED_FIELDS.issubset(entry.keys())
        assert entry["doi"] not in seen_dois, f"duplicate DOI: {entry['doi']}"
        seen_dois.add(entry["doi"])


def test_manifest_covers_all_git_tracked_seed_papers_with_existing_files():
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest_paths = {entry["path"] for entry in manifest}

    for seed_path in GIT_TRACKED_SEED_PAPERS:
        assert seed_path in manifest_paths, f"git-tracked paper missing from manifest: {seed_path}"
        assert (CORPUS_DIR / seed_path).is_file(), f"missing PDF for git-tracked entry: {seed_path}"
