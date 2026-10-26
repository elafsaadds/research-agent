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
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------
# 1. Paths and external-service configuration
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "data" / "corpus"

# arXiv
ARXIV_URL = "https://export.arxiv.org/api/query"
ARXIV_ALLOWED_HOSTS = {"export.arxiv.org"}
ARXIV_TIMEOUT_S = 10.0

MIN_INTERVAL_S = 3.0
_last_arxiv_call = 0.0

# OpenAlex
OPENALEX_URL = "https://api.openalex.org/works"
OPENALEX_ALLOWED_HOSTS = {"api.openalex.org"}
OPENALEX_TIMEOUT_S = 10.0


# ---------------------------------------------------------------------
# 2. Tokenisation
# ---------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")

_STOP = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "of",
    "in",
    "on",
    "to",
    "and",
    "or",
    "for",
    "with",
    "what",
    "how",
    "why",
    "can",
    "do",
    "does",
    "did",
    "it",
    "its",
    "as",
    "by",
    "at",
    "be",
    "this",
    "that",
    "from",
    "which",
    "who",
}


def tokenize(text: str) -> list[str]:
    """Tokenise text for simple retrieval."""
    return [
        token
        for token in _TOKEN.findall(text.lower())
        if token not in _STOP
    ]


# ---------------------------------------------------------------------
# 3. Shared result types
# ---------------------------------------------------------------------

class SearchHit(BaseModel):
    """
    One retrieved item with provenance.

    Retrieved text is DATA, never instructions.
    """

    source_id: str
    title: str
    url: str | None = None
    snippet: str
    score: float = 0.0
    origin: Literal["corpus", "arxiv", "openalex"]


class ToolResult(BaseModel):
    """Standard result returned by every registered tool."""

    tool: str
    ok: bool
    hits: list[SearchHit] = Field(default_factory=list)
    error_type: str | None = None
    error: str | None = None
    latency_s: float = 0.0


class ToolFailure(Exception):
    """Controlled tool failure converted into ToolResult by execute()."""

    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type


class QueryArgs(BaseModel):
    """Common validated arguments for retrieval tools."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=300)
    max_results: int = Field(default=3, ge=1, le=5)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        value = " ".join(value.split())

        if len(value) < 2:
            raise ValueError("query too short")

        return value


# ---------------------------------------------------------------------
# 4. Local corpus retrieval
# ---------------------------------------------------------------------

@dataclass
class Doc:
    doc_id: str
    title: str
    source: str
    body: str


def parse_doc(text: str, fallback_id: str) -> Doc:
    """Parse a simple metadata header and document body."""

    text = text.replace("\r\n", "\n").lstrip("\ufeff")

    head, separator, body = text.partition("\n---\n")

    meta: dict[str, str] = {}

    if separator:
        for line in head.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip().lower()] = value.strip()
    else:
        body = text

    return Doc(
        doc_id=meta.get("id", fallback_id),
        title=meta.get("title", fallback_id),
        source=meta.get("source", "unknown"),
        body=body.strip(),
    )


class Corpus:
    """BM25-backed local document collection."""

    def __init__(self, directory: Path):
        self.docs: list[Doc] = []

        directory = Path(directory)

        if directory.exists():
            for path in sorted(directory.glob("*.md")):
                text = path.read_text(encoding="utf-8-sig")
                self.docs.append(
                    parse_doc(text, path.stem)
                )

        self._bm25 = (
            BM25Okapi(
                [
                    tokenize(doc.title + " " + doc.body)
                    for doc in self.docs
                ]
            )
            if self.docs
            else None
        )

    def search(
        self,
        query: str,
        k: int = 3,
    ) -> list[SearchHit]:
        """Search the local corpus using BM25."""

        tokens = tokenize(query)

        if not tokens or self._bm25 is None:
            return []

        scores = self._bm25.get_scores(tokens)

        ranked = sorted(
            range(len(self.docs)),
            key=lambda index: scores[index],
            reverse=True,
        )

        hits: list[SearchHit] = []

        for index in ranked[:k]:
            if scores[index] <= 0:
                continue

            doc = self.docs[index]

            hits.append(
                SearchHit(
                    source_id=doc.doc_id,
                    title=doc.title,
                    snippet=" ".join(doc.body.split())[:300],
                    score=round(float(scores[index]), 3),
                    origin="corpus",
                )
            )

        return hits


@lru_cache(maxsize=1)
def get_corpus() -> Corpus:
    """Return the cached local corpus."""
    return Corpus(CORPUS_DIR)


def search_corpus(args: QueryArgs) -> list[SearchHit]:
    """Search the local document collection."""
    return get_corpus().search(
        args.query,
        args.max_results,
    )


# ---------------------------------------------------------------------
# 5. arXiv retrieval
# ---------------------------------------------------------------------

_ATOM = {
    "a": "http://www.w3.org/2005/Atom",
}


def _arxiv_query(query: str) -> str:
    """Build a sanitised arXiv query from the research question."""

    tokens = tokenize(query)[:6]

    if not tokens:
        raise ToolFailure(
            "invalid_arguments",
            "query has no searchable words",
        )

    return " AND ".join(
        f"all:{token}"
        for token in tokens
    )


def _http_get(
    url: str,
    params: dict,
    timeout: float,
) -> httpx.Response:
    """
    Perform an arXiv HTTP GET after checking the host allow-list.
    """

    host = httpx.URL(url).host

    if host not in ARXIV_ALLOWED_HOSTS:
        raise ToolFailure(
            "blocked_host",
            f"host not allowed: {host}",
        )

    return httpx.get(
        url,
        params=params,
        timeout=timeout,
        follow_redirects=False,
    )


def parse_arxiv_atom(
    xml_text: str,
    max_results: int,
) -> list[SearchHit]:
    """Parse an arXiv Atom response into SearchHit objects."""

    try:
        root = ET.fromstring(xml_text)

    except ET.ParseError as exc:
        raise ToolFailure(
            "bad_response",
            f"unparseable XML: {exc}",
        ) from exc

    hits: list[SearchHit] = []

    for entry in root.findall(
        "a:entry",
        _ATOM,
    )[:max_results]:

        raw_id = (
            entry.findtext(
                "a:id",
                default="",
                namespaces=_ATOM,
            )
            or ""
        ).strip()

        arxiv_id = raw_id.rsplit(
            "/abs/",
            1,
        )[-1]

        if not arxiv_id:
            continue

        title = " ".join(
            (
                entry.findtext(
                    "a:title",
                    default="",
                    namespaces=_ATOM,
                )
                or ""
            ).split()
        )

        summary = " ".join(
            (
                entry.findtext(
                    "a:summary",
                    default="",
                    namespaces=_ATOM,
                )
                or ""
            ).split()
        )

        hits.append(
            SearchHit(
                source_id=f"arxiv:{arxiv_id}",
                title=title or arxiv_id,
                url=f"https://arxiv.org/abs/{arxiv_id}",
                snippet=summary[:400],
                origin="arxiv",
            )
        )

    return hits


def search_arxiv(
    args: QueryArgs,
) -> list[SearchHit]:
    """Search arXiv for academic papers."""

    global _last_arxiv_call

    params = {
        "search_query": _arxiv_query(args.query),
        "start": 0,
        "max_results": args.max_results,
        "sortBy": "relevance",
    }

    wait = MIN_INTERVAL_S - (
        time.monotonic() - _last_arxiv_call
    )

    if wait > 0:
        time.sleep(wait)

    _last_arxiv_call = time.monotonic()

    try:
        response = _http_get(
            ARXIV_URL,
            params,
            ARXIV_TIMEOUT_S,
        )

        response.raise_for_status()

    except httpx.TimeoutException as exc:
        raise ToolFailure(
            "timeout",
            "arXiv request timed out",
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise ToolFailure(
            "http_error",
            f"arXiv returned {exc.response.status_code}",
        ) from exc

    except httpx.HTTPError as exc:
        raise ToolFailure(
            "network_error",
            f"arXiv unreachable: {type(exc).__name__}",
        ) from exc

    return parse_arxiv_atom(
        response.text,
        args.max_results,
    )


# ---------------------------------------------------------------------
# 6. OpenAlex retrieval
# ---------------------------------------------------------------------

def parse_openalex_json(
    payload: dict,
    max_results: int,
) -> list[SearchHit]:
    """
    Parse an OpenAlex works response.

    Only stable bibliographic information is extracted.
    Retrieved content remains data and is never treated as instructions.
    """

    if not isinstance(payload, dict):
        raise ToolFailure(
            "bad_response",
            "OpenAlex response is not a JSON object",
        )

    results = payload.get("results")

    if not isinstance(results, list):
        raise ToolFailure(
            "bad_response",
            "OpenAlex response has no valid results list",
        )

    hits: list[SearchHit] = []

    for item in results[:max_results]:

        if not isinstance(item, dict):
            continue

        raw_id = item.get("id")
        title = item.get("display_name")
        doi = item.get("doi")

        if (
            not isinstance(raw_id, str)
            or not raw_id.strip()
        ):
            continue

        if (
            not isinstance(title, str)
            or not title.strip()
        ):
            continue

        if (
            isinstance(doi, str)
            and doi.strip()
        ):
            url = doi
        else:
            url = raw_id

        # OpenAlex stores abstracts as an inverted index:
        #
        # {
        #     "word": [position1, position2, ...]
        # }
        #
        # Reconstruct the abstract into readable text.
        abstract = item.get(
            "abstract_inverted_index"
        )

        snippet = title

        if isinstance(abstract, dict):
            words: list[tuple[int, str]] = []

            for word, positions in abstract.items():

                if (
                    isinstance(word, str)
                    and isinstance(positions, list)
                ):
                    for position in positions:

                        if isinstance(position, int):
                            words.append(
                                (position, word)
                            )

            words.sort(
                key=lambda pair: pair[0]
            )

            abstract_text = " ".join(
                word
                for _, word in words
            )

            if abstract_text:
                snippet = abstract_text[:400]

        source_id = (
            raw_id
            .rstrip("/")
            .rsplit("/", 1)[-1]
        )

        hits.append(
            SearchHit(
                source_id=f"openalex:{source_id}",
                title=" ".join(title.split()),
                url=url,
                snippet=" ".join(
                    snippet.split()
                )[:400],
                origin="openalex",
            )
        )

    return hits


def _openalex_query(query: str) -> str:
    """Normalise and validate an OpenAlex search query."""

    value = " ".join(query.split())

    if len(value) < 2:
        raise ToolFailure(
            "invalid_arguments",
            "query too short",
        )

    return value


def _openalex_http_get(
    url: str,
    params: dict,
    timeout: float,
) -> httpx.Response:
    """
    Perform an OpenAlex HTTP GET after checking
    the host allow-list.
    """

    host = httpx.URL(url).host

    if host not in OPENALEX_ALLOWED_HOSTS:
        raise ToolFailure(
            "blocked_host",
            f"host not allowed: {host}",
        )

    return httpx.get(
        url,
        params=params,
        timeout=timeout,
        follow_redirects=False,
    )


def search_openalex(
    args: QueryArgs,
) -> list[SearchHit]:
    """Search OpenAlex for academic works."""

    params = {
        "search": _openalex_query(args.query),
        "per-page": args.max_results,
    }

    try:
        response = _openalex_http_get(
            OPENALEX_URL,
            params,
            OPENALEX_TIMEOUT_S,
        )

        response.raise_for_status()

    except httpx.TimeoutException as exc:
        raise ToolFailure(
            "timeout",
            "OpenAlex request timed out",
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise ToolFailure(
            "http_error",
            f"OpenAlex returned {exc.response.status_code}",
        ) from exc

    except httpx.HTTPError as exc:
        raise ToolFailure(
            "network_error",
            f"OpenAlex unreachable: {type(exc).__name__}",
        ) from exc

    try:
        payload = response.json()

    except ValueError as exc:
        raise ToolFailure(
            "bad_response",
            "OpenAlex returned invalid JSON",
        ) from exc

    return parse_openalex_json(
        payload,
        args.max_results,
    )


# ---------------------------------------------------------------------
# 7. Tool registry
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    name: str
    tier: Literal[
        "read",
        "write",
        "consequential",
    ]
    description: str
    args_model: type[BaseModel]
    func: Callable
    timeout_s: float
    side_effects: str


REGISTRY: dict[str, ToolSpec] = {
    "search_corpus": ToolSpec(
        "search_corpus",
        "read",
        "Search the local document collection (BM25).",
        QueryArgs,
        search_corpus,
        2.0,
        "none",
    ),

    "search_arxiv": ToolSpec(
        "search_arxiv",
        "read",
        "Search arXiv for papers.",
        QueryArgs,
        search_arxiv,
        ARXIV_TIMEOUT_S,
        "network request to export.arxiv.org",
    ),

    "search_openalex": ToolSpec(
        "search_openalex",
        "read",
        "Search OpenAlex for academic works.",
        QueryArgs,
        search_openalex,
        OPENALEX_TIMEOUT_S,
        "network request to api.openalex.org",
    ),
}


# ---------------------------------------------------------------------
# 8. Common tool executor
# ---------------------------------------------------------------------

def _fail(
    tool: str,
    error_type: str,
    message: str,
    start: float,
) -> ToolResult:
    """Build a standard failed ToolResult."""

    return ToolResult(
        tool=tool,
        ok=False,
        error_type=error_type,
        error=message,
        latency_s=round(
            time.perf_counter() - start,
            3,
        ),
    )


def execute(
    name: str,
    arguments: object,
) -> ToolResult:
    """
    Validate arguments and run a registered tool.

    This function never raises tool errors.
    Errors are returned as ToolResult values.
    """

    start = time.perf_counter()

    spec = REGISTRY.get(name)

    if spec is None:
        return _fail(
            name,
            "unknown_tool",
            f"no registered tool named {name!r}",
            start,
        )

    try:
        args = spec.args_model.model_validate(
            arguments
            if arguments is not None
            else {}
        )

    except ValidationError as exc:
        message = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'arguments'}: "
            f"{error['msg']}"
            for error in exc.errors()
        )

        return _fail(
            name,
            "invalid_arguments",
            message,
            start,
        )

    try:
        hits = spec.func(args)

    except ToolFailure as exc:
        return _fail(
            name,
            exc.error_type,
            str(exc),
            start,
        )

    except Exception as exc:
        # A tool bug must not crash the whole research run.
        return _fail(
            name,
            "tool_error",
            type(exc).__name__,
            start,
        )

    return ToolResult(
        tool=name,
        ok=True,
        hits=hits,
        latency_s=round(
            time.perf_counter() - start,
            3,
        ),
    )


# ---------------------------------------------------------------------
# 9. Tool catalogue
# ---------------------------------------------------------------------

def catalogue() -> list[dict]:
    """Return registered tool descriptions for documentation."""

    return [
        {
            "name": spec.name,
            "tier": spec.tier,
            "description": spec.description,
            "timeout_s": spec.timeout_s,
            "side_effects": spec.side_effects,
        }
        for spec in REGISTRY.values()
    ]