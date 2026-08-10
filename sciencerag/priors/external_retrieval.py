"""External literature retrieval (spec §3.2/§3.5, §9 OQ#1) — M6.

Supplements the internal corpus with Semantic Scholar and arXiv when
`allow_external=True` and internal coverage looks thin (retrieval.py
triggers this on `coverage.gaps` being non-empty).

Whenever a hit has a real downloadable PDF (arXiv always has one; Semantic
Scholar exposes one via `openAccessPdf` for open-access papers, not for
paywalled ones), that PDF is downloaded straight into `corpus/papers/` —
the same directory PaperQA2 indexes for internal retrieval. There is no
separate "unverified" staging area and no approval step: once a paper's
full text is downloaded, it is retrievable and cited exactly like any
other corpus paper on every subsequent query, `allow_external` or not.
The DOI-derived filename doubles as the dedup check (skip download if the
file already exists) — no separate tracking database.

Semantic Scholar hits with no open-access PDF fall back to abstract-only
evidence (the only case that still gets `Prior.provenance =
"external_unverified"` — that tag now reflects "this is a thin abstract
snippet", not a trust judgment, since there's no way to obtain full text
for a paywalled paper).

Network failures (timeout, rate limit, DNS) degrade to "skip this one
result" rather than failing the whole priors request — external retrieval
is a best-effort supplement, not a hard dependency. Because a downloaded
file is trusted immediately with no human in the loop, `download_new_papers`
is deliberately defensive about what it will fetch and accept as a real
paper (see its docstring) — this is the one place in the pipeline where
fully untrusted, adversarial-controllable input (a URL string from a
public API response) turns directly into a local file write and,
downstream, LLM input.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from pathlib import Path
from typing import Literal
from urllib.parse import quote, urlparse

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
REQUEST_TIMEOUT_SECONDS = 15.0
PDF_DOWNLOAD_TIMEOUT_SECONDS = 30.0
DEFAULT_LIMIT = 5

# Generous for a single academic paper (even figure-heavy PDFs rarely pass
# a few tens of MB) — caps memory use and disk growth per download, and
# bounds how much an adversarial/misbehaving server can make us buffer.
MAX_PDF_BYTES = 50 * 1024 * 1024

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = REPO_ROOT / "corpus" / "papers"


class ExternalPaper(BaseModel):
    title: str
    abstract: str
    doi: str
    year: int | None = None
    semantic_scholar_id: str = ""
    source: Literal["semantic_scholar", "arxiv"] = "semantic_scholar"
    pdf_url: str | None = None


def _doi_to_filename(doi: str) -> str:
    # percent-encoding (not a naive .replace("/", "_")) so distinct DOIs
    # can never collide onto the same filename (e.g. "10.1000/182" and a
    # malformed/adversarial "10.1000_182" used to both map to
    # "10.1000_182.pdf" — whichever arrived first would silently shadow
    # the other, misattributing evidence to the wrong paper) and so no
    # character in an attacker-influenced DOI string (".."," ", "\x00",
    # path separators) ends up meaning anything special on the filesystem.
    return quote(doi, safe="") + ".pdf"


def paper_pdf_path(doi: str) -> Path:
    return CORPUS_DIR / _doi_to_filename(doi)


def _is_fetchable_url(url: str) -> bool:
    """Rejects anything that isn't a plain http(s) URL resolving to a
    public IP address. `pdf_url` comes straight from a public API response
    — fully attacker-influenceable if that API is ever compromised or
    spoofed — and feeds directly into a server-side HTTP request, so this
    is this module's SSRF guard. Note: resolving here and letting httpx
    resolve again at connect time leaves a narrow DNS-rebinding gap (the
    name could re-resolve to a different, private address between the two
    lookups) — not closed here, since doing so would mean pinning
    connections to a specific IP, a bigger change than this fix warrants
    for a feature that only ever queries two fixed, reputable APIs."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


def _download_bytes(url: str) -> bytes | None:
    """Streams the response with a hard size cap rather than buffering an
    unbounded body (`httpx.get(...).content` would read the whole response
    into memory regardless of size — a misbehaving or malicious server
    could otherwise force an arbitrarily large allocation)."""
    total = 0
    chunks: list[bytes] = []
    with httpx.stream(
        "GET", url, timeout=PDF_DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=True
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > MAX_PDF_BYTES:
                logger.warning("Download from %s exceeded %d byte cap, aborting", url, MAX_PDF_BYTES)
                return None
            chunks.append(chunk)
    return b"".join(chunks)


def download_new_papers(papers: list[ExternalPaper]) -> list[ExternalPaper]:
    """Downloads the PDF for every paper that has a `pdf_url` and isn't
    already on disk. Returns the subset actually newly downloaded (used by
    the caller to decide whether a re-query is worth the extra latency).

    Every failure mode here — bad URL, network error, oversized response,
    content that isn't actually a PDF, a disk write failure — just skips
    that one paper and moves on; this function never raises. That matters
    more than usual for a function whose caller (retrieval.py) treats a
    successful download as "now permanently trusted, no review" — a junk
    or partial file landing in the corpus is not just a wasted download,
    it's bad evidence a future query could cite."""
    downloaded = []
    for paper in papers:
        if not paper.pdf_url or not paper.doi.strip():
            continue
        dest = paper_pdf_path(paper.doi)
        if dest.exists():
            continue
        if not _is_fetchable_url(paper.pdf_url):
            logger.warning("Refusing to fetch unsafe/non-public URL for %s: %s", paper.doi, paper.pdf_url)
            continue

        try:
            content = _download_bytes(paper.pdf_url)
        except httpx.HTTPError as e:
            logger.warning("Failed to download PDF for %s: %s", paper.doi, e)
            continue
        if content is None:
            continue
        if not content.startswith(b"%PDF-"):
            # A stale/broken openAccessPdf link commonly redirects to an
            # HTML paywall or login page instead of 404ing — silently
            # saving that as "<doi>.pdf" would hand PaperQA2 (and,
            # downstream, the extraction LLM) garbage as if it were the
            # paper's real text.
            logger.warning("Downloaded content for %s is not a PDF (bad/stale link?), skipping", paper.doi)
            continue

        # Computed before any filesystem call so it's always bound if one
        # of those calls raises — an earlier version of this fix computed
        # it *inside* the try block, which meant a mkdir failure raised
        # UnboundLocalError out of the except clause instead of the
        # intended graceful skip.
        tmp_dest = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Write-to-temp-then-rename: a concurrent PaperQA2 index read
            # (from another request already inside its own run_query) or a
            # second concurrent downloader for the same DOI can never
            # observe a partially-written file; os.replace is atomic on
            # the same filesystem, which the temp file always is since
            # it's written next to `dest`.
            tmp_dest.write_bytes(content)
            os.replace(tmp_dest, dest)
        except OSError as e:
            logger.warning("Failed to write PDF for %s to disk: %s", paper.doi, e)
            # Best-effort cleanup — missing_ok=True only swallows
            # FileNotFoundError, not every OSError a broken path can
            # raise (e.g. NotADirectoryError when a parent component
            # isn't a directory, the same underlying failure that likely
            # just caused the write itself to fail); cleanup must not
            # raise a fresh exception out of an already-failing branch.
            try:
                tmp_dest.unlink(missing_ok=True)
            except OSError:
                pass
            continue

        downloaded.append(paper)
    return downloaded


def search_semantic_scholar(query: str, limit: int = DEFAULT_LIMIT) -> list[ExternalPaper]:
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,externalIds,year,paperId,openAccessPdf",
    }
    try:
        response = httpx.get(
            SEMANTIC_SCHOLAR_SEARCH_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("Semantic Scholar search failed, skipping external augmentation: %s", e)
        return []

    data = response.json()
    papers = []
    for item in data.get("data", []):
        abstract = item.get("abstract")
        if not abstract:
            continue  # no usable evidence text without a paper to extract from
        external_ids = item.get("externalIds") or {}
        doi = external_ids.get("DOI")
        if not doi:
            # sciencerag.priors.models.SourcePaper requires a real `doi:
            # str` — a paper we can't cite properly isn't usable evidence
            # here, not worth inventing a semantic_scholar_id-as-doi hack.
            continue
        open_access_pdf = item.get("openAccessPdf") or {}
        papers.append(
            ExternalPaper(
                title=item.get("title") or "(untitled)",
                abstract=abstract,
                doi=doi,
                year=item.get("year"),
                semantic_scholar_id=item.get("paperId") or "",
                source="semantic_scholar",
                pdf_url=open_access_pdf.get("url"),
            )
        )
    return papers
