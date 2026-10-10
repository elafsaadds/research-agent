"""Tools: narrow, typed, read-only. Errors are returned as values, never raised."""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "data" / "corpus"

ARXIV_URL = "https://export.arxiv.org/api/query"
ARXIV_ALLOWED_HOSTS = {"export.arxiv.org"}
ARXIV_TIMEOUT_S = 10.0
MIN_INTERVAL_S = 3.0  # arXiv asks clients to wait between requests
_last_arxiv_call = 0.0

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "to", "and",
    "or", "for", "with", "what", "how", "why", "can", "do", "does", "did", "it",
    "its", "as", "by", "at", "be", "this", "that", "from", "which", "who",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


# ---------- result types (shared by every tool) ----------

class SearchHit(BaseModel):
    """One retrieved item with provenance. Its text is DATA, never instructions."""

    source_id: str
    title: str
    url: str | None = None
    snippet: str
    score: float = 0.0
    origin: Literal["corpus", "arxiv"]


class ToolResult(BaseModel):
    tool: str
    ok: bool
    hits: list[SearchHit] = Field(default_factory=list)
    error_type: str | None = None
    error: str | None = None
    latency_s: float = 0.0


class ToolFailure(Exception):
    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type


class QueryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=300)
    max_results: int = Field(default=3, ge=1, le=5)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 2:
            raise ValueError("query too short")
        return v


# ---------- local corpus (BM25) ----------

@dataclass
class Doc:
    doc_id: str
    title: str
    source: str
    body: str


def parse_doc(text: str, fallback_id: str) -> Doc:
    text = text.replace("\r\n", "\n").lstrip("\ufeff")
    head, sep, body = text.partition("\n---\n")
    meta: dict[str, str] = {}
    if sep:
        for line in head.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip().lower()] = v.strip()
    else:
        body = text
    return Doc(
        doc_id=meta.get("id", fallback_id),
        title=meta.get("title", fallback_id),
        source=meta.get("source", "unknown"),
        body=body.strip(),
    )


class Corpus:
    def __init__(self, directory: Path):
        self.docs: list[Doc] = []
        directory = Path(directory)
        if directory.exists():
            for p in sorted(directory.glob("*.md")):
                self.docs.append(parse_doc(p.read_text(encoding="utf-8-sig"), p.stem))
        self._bm25 = (
            BM25Okapi([tokenize(d.title + " " + d.body) for d in self.docs])
            if self.docs else None
        )

    def search(self, query: str, k: int = 3) -> list[SearchHit]:
        tokens = tokenize(query)
        if not tokens or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in ranked[:k]:
            if scores[i] <= 0:
                continue
            d = self.docs[i]
            hits.append(SearchHit(
                source_id=d.doc_id, title=d.title,
                snippet=" ".join(d.body.split())[:300],
                score=round(float(scores[i]), 3), origin="corpus"))
        return hits


@lru_cache(maxsize=1)
def get_corpus() -> Corpus:
    return Corpus(CORPUS_DIR)


def search_corpus(args: QueryArgs) -> list[SearchHit]:
    return get_corpus().search(args.query, args.max_results)


# ---------- arXiv ----------

_ATOM = {"a": "http://www.w3.org/2005/Atom"}


def _arxiv_query(query: str) -> str:
    tokens = tokenize(query)[:6]
    if not tokens:
        raise ToolFailure("invalid_arguments", "query has no searchable words")
    return " AND ".join(f"all:{t}" for t in tokens)


def _http_get(url: str, params: dict, timeout: float) -> httpx.Response:
    host = httpx.URL(url).host
    if host not in ARXIV_ALLOWED_HOSTS:
        raise ToolFailure("blocked_host", f"host not allowed: {host}")
    return httpx.get(url, params=params, timeout=timeout, follow_redirects=False)


def parse_arxiv_atom(xml_text: str, max_results: int) -> list[SearchHit]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ToolFailure("bad_response", f"unparseable XML: {exc}") from exc
    hits = []
    for entry in root.findall("a:entry", _ATOM)[:max_results]:
        raw_id = (entry.findtext("a:id", default="", namespaces=_ATOM) or "").strip()
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1]
        if not arxiv_id:
            continue
        title = " ".join((entry.findtext("a:title", default="", namespaces=_ATOM) or "").split())
        summary = " ".join((entry.findtext("a:summary", default="", namespaces=_ATOM) or "").split())
        hits.append(SearchHit(
            source_id=f"arxiv:{arxiv_id}", title=title or arxiv_id,
            url=f"https://arxiv.org/abs/{arxiv_id}",
            snippet=summary[:400], origin="arxiv"))
    return hits


def search_arxiv(args: QueryArgs) -> list[SearchHit]:
    global _last_arxiv_call
    params = {"search_query": _arxiv_query(args.query), "start": 0,
              "max_results": args.max_results, "sortBy": "relevance"}
    wait = MIN_INTERVAL_S - (time.monotonic() - _last_arxiv_call)
    if wait > 0:
        time.sleep(wait)
    _last_arxiv_call = time.monotonic()
    try:
        resp = _http_get(ARXIV_URL, params, ARXIV_TIMEOUT_S)
        resp.raise_for_status()
    except httpx.TimeoutException as exc:
        raise ToolFailure("timeout", "arXiv request timed out") from exc
    except httpx.HTTPStatusError as exc:
        raise ToolFailure("http_error", f"arXiv returned {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise ToolFailure("network_error", f"arXiv unreachable: {type(exc).__name__}") from exc
    return parse_arxiv_atom(resp.text, args.max_results)


# ---------- registry and executor ----------

@dataclass(frozen=True)
class ToolSpec:
    name: str
    tier: Literal["read", "write", "consequential"]
    description: str
    args_model: type[BaseModel]
    func: Callable
    timeout_s: float
    side_effects: str


REGISTRY: dict[str, ToolSpec] = {
    "search_corpus": ToolSpec(
        "search_corpus", "read", "Search the local document collection (BM25).",
        QueryArgs, search_corpus, 2.0, "none"),
    "search_arxiv": ToolSpec(
        "search_arxiv", "read", "Search arXiv for papers.",
        QueryArgs, search_arxiv, ARXIV_TIMEOUT_S, "network request to export.arxiv.org"),
}


def _fail(tool: str, error_type: str, message: str, start: float) -> ToolResult:
    return ToolResult(tool=tool, ok=False, error_type=error_type, error=message,
                      latency_s=round(time.perf_counter() - start, 3))


def execute(name: str, arguments: object) -> ToolResult:
    """Validate arguments and run a registered tool. Never raises.

    This is NOT the policy gate (stage 5). It only guarantees: registered tool,
    well-formed arguments, errors returned as values.
    """
    start = time.perf_counter()
    spec = REGISTRY.get(name)
    if spec is None:
        return _fail(name, "unknown_tool", f"no registered tool named {name!r}", start)
    try:
        args = spec.args_model.model_validate(arguments if arguments is not None else {})
    except ValidationError as exc:
        msg = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}"
            for e in exc.errors())
        return _fail(name, "invalid_arguments", msg, start)
    try:
        hits = spec.func(args)
    except ToolFailure as exc:
        return _fail(name, exc.error_type, str(exc), start)
    except Exception as exc:  # last resort: a tool bug must not crash the run
        return _fail(name, "tool_error", type(exc).__name__, start)
    return ToolResult(tool=name, ok=True, hits=hits,
                      latency_s=round(time.perf_counter() - start, 3))


def catalogue() -> list[dict]:
    """Tool descriptions for documentation and (later) the prompt."""
    return [{"name": s.name, "tier": s.tier, "description": s.description,
             "timeout_s": s.timeout_s, "side_effects": s.side_effects}
            for s in REGISTRY.values()]
