"""arXiv search for external retrieval (spec §3.2/§3.5, §9 OQ#1) — M6.

arXiv's public Atom API (`export.arxiv.org`), no API key required. Every
arXiv preprint has an auto-assigned DOI in the standard
`10.48550/arXiv.<id>` format (arXiv has done this since 2022) and a
directly downloadable PDF, so every hit here is eligible for the
full-text download path in `external_retrieval.download_new_papers` —
unlike Semantic Scholar, there's no paywalled-abstract-only case.

Network failures degrade to an empty result list rather than failing the
whole priors request — same principle as external_retrieval.py.
"""

from __future__ import annotations

import logging
import re
from xml.etree import ElementTree

import httpx

from sciencerag.priors.external_retrieval import ExternalPaper

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
REQUEST_TIMEOUT_SECONDS = 15.0
DEFAULT_LIMIT = 5

_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_VERSION_SUFFIX = re.compile(r"v\d+$")


def _arxiv_id_from_entry_id(entry_id: str) -> str:
    """'http://arxiv.org/abs/2011.02585v1' -> '2011.02585v1'."""
    return entry_id.rsplit("/abs/", 1)[-1]


def _doi_for_arxiv_id(arxiv_id_with_version: str) -> str:
    bare_id = _VERSION_SUFFIX.sub("", arxiv_id_with_version)
    return f"10.48550/arXiv.{bare_id}"


def _pdf_url(entry: ElementTree.Element, arxiv_id_with_version: str) -> str:
    for link in entry.findall("atom:link", _ATOM_NS):
        if link.attrib.get("title") == "pdf" and link.attrib.get("href"):
            return link.attrib["href"]
    return f"https://arxiv.org/pdf/{arxiv_id_with_version}"


def search_arxiv(query: str, limit: int = DEFAULT_LIMIT) -> list[ExternalPaper]:
    params = {"search_query": f"all:{query}", "start": 0, "max_results": limit}
    try:
        response = httpx.get(ARXIV_API_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("arXiv search failed, skipping external augmentation: %s", e)
        return []

    # ElementTree/expat expands internal DTD entities by default — a
    # "billion laughs" style payload (a handful of nested <!ENTITY>
    # definitions, each referencing the previous one ~10x) can blow up to
    # gigabytes of text from a few hundred bytes on the wire, before any
    # size check on our end ever runs. Real arXiv responses never contain
    # a DOCTYPE, so outright refusing to parse one has no false-positive
    # cost and closes this off without adding a dependency (defusedxml
    # isn't already in the project).
    if "<!DOCTYPE" in response.text:
        logger.warning("arXiv search response contains a DOCTYPE declaration, refusing to parse")
        return []

    try:
        root = ElementTree.fromstring(response.text)
    except ElementTree.ParseError as e:
        logger.warning("arXiv search returned unparseable XML: %s", e)
        return []

    papers = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        entry_id = entry.findtext("atom:id", default="", namespaces=_ATOM_NS)
        title = entry.findtext("atom:title", default="", namespaces=_ATOM_NS)
        summary = entry.findtext("atom:summary", default="", namespaces=_ATOM_NS)
        published = entry.findtext("atom:published", default="", namespaces=_ATOM_NS)
        if not entry_id or not summary:
            continue

        arxiv_id = _arxiv_id_from_entry_id(entry_id)
        year = int(published[:4]) if published[:4].isdigit() else None
        papers.append(
            ExternalPaper(
                title=" ".join(title.split()) or "(untitled)",
                abstract=" ".join(summary.split()),
                doi=_doi_for_arxiv_id(arxiv_id),
                year=year,
                source="arxiv",
                pdf_url=_pdf_url(entry, arxiv_id),
            )
        )
    return papers
