import httpx
import pytest

from research_agent import tools
from research_agent.tools import Corpus, execute, parse_arxiv_atom, parse_doc

DOC_A = ("id: a-1\ntitle: Indirect injection\nsource: test\n---\n"
         "Indirect prompt injection hides instructions in web pages that an agent reads.")
DOC_B = ("id: b-1\ntitle: Least privilege\nsource: test\n---\n"
         "Least privilege gives each tool only the permissions it needs.")
DOC_C = ("id: c-1\ntitle: Memory poisoning\nsource: test\n---\n"
         "Memory poisoning writes malicious content into persistent agent memory.")

SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/9999.00001v1</id>
    <title>A Test Paper
      on Agents</title>
    <summary>  An abstract
     with   odd spacing.  </summary>
  </entry>
</feed>"""


def _resp(status, text=""):
    return httpx.Response(status, text=text,
                          request=httpx.Request("GET", "https://export.arxiv.org/api/query"))


@pytest.fixture
def corpus(tmp_path):
    # BM25 needs at least 3 documents: with 2, a term in one doc gets IDF 0.
    (tmp_path / "a.md").write_text(DOC_A, encoding="utf-8")
    (tmp_path / "b.md").write_text(DOC_B, encoding="utf-8")
    (tmp_path / "c.md").write_text(DOC_C, encoding="utf-8")
    return Corpus(tmp_path)


@pytest.fixture(autouse=True)
def no_throttle(monkeypatch):
    monkeypatch.setattr(tools, "MIN_INTERVAL_S", 0.0)


def test_finds_relevant_doc(corpus):
    hits = corpus.search("hidden instructions in web pages", 3)
    assert hits and hits[0].source_id == "a-1"
    assert hits[0].origin == "corpus"


def test_unrelated_query_returns_nothing(corpus):
    assert corpus.search("quantum lattice", 3) == []


def test_max_results_respected(corpus):
    assert len(corpus.search("agent", 1)) <= 1


def test_parse_doc_front_matter():
    d = parse_doc(DOC_A, "fallback")
    assert (d.doc_id, d.title, d.source) == ("a-1", "Indirect injection", "test")
    assert d.body.startswith("Indirect prompt")


def test_parse_doc_without_front_matter():
    d = parse_doc("just text", "file1")
    assert d.doc_id == "file1" and d.body == "just text"


def test_empty_directory(tmp_path):
    assert Corpus(tmp_path / "missing").search("anything", 3) == []


def test_unknown_tool_is_a_value_not_an_exception():
    r = execute("fetch_page", {"url": "https://example.com"})
    assert not r.ok and r.error_type == "unknown_tool"


@pytest.mark.parametrize("args", [
    {"query": "x"},
    {"query": "valid query", "secret": 1},
    {"query": "valid query", "max_results": 99},
    "not a dict",
])
def test_invalid_arguments_rejected(args):
    r = execute("search_corpus", args)
    assert not r.ok and r.error_type == "invalid_arguments"


def test_all_registered_tools_are_read_tier():
    assert all(s.tier == "read" for s in tools.REGISTRY.values())


def test_search_corpus_through_execute(monkeypatch, corpus):
    monkeypatch.setattr(tools, "get_corpus", lambda: corpus)
    r = execute("search_corpus", {"query": "memory poisoning persistent"})
    assert r.ok and r.hits[0].source_id == "c-1"


def test_parse_arxiv_atom():
    hits = parse_arxiv_atom(SAMPLE_ATOM, 3)
    assert hits[0].source_id == "arxiv:9999.00001v1"
    assert hits[0].url == "https://arxiv.org/abs/9999.00001v1"
    assert hits[0].title == "A Test Paper on Agents"
    assert "  " not in hits[0].snippet


def test_arxiv_query_is_sanitised():
    assert tools._arxiv_query("prompt injection!! (agents)") == (
        "all:prompt AND all:injection AND all:agents")


def test_arxiv_success(monkeypatch):
    monkeypatch.setattr(tools, "_http_get", lambda *a, **k: _resp(200, SAMPLE_ATOM))
    r = execute("search_arxiv", {"query": "prompt injection"})
    assert r.ok and r.hits[0].origin == "arxiv"


@pytest.mark.parametrize("exc,expected", [
    (httpx.TimeoutException("slow"), "timeout"),
    (httpx.ConnectError("down"), "network_error"),
])
def test_arxiv_transport_errors_become_values(monkeypatch, exc, expected):
    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(tools, "_http_get", boom)
    r = execute("search_arxiv", {"query": "prompt injection"})
    assert not r.ok and r.error_type == expected


def test_arxiv_http_error(monkeypatch):
    monkeypatch.setattr(tools, "_http_get", lambda *a, **k: _resp(503))
    assert execute("search_arxiv", {"query": "prompt injection"}).error_type == "http_error"


def test_arxiv_bad_xml(monkeypatch):
    monkeypatch.setattr(tools, "_http_get", lambda *a, **k: _resp(200, "not xml"))
    assert execute("search_arxiv", {"query": "prompt injection"}).error_type == "bad_response"


def test_host_allow_list():
    with pytest.raises(tools.ToolFailure):
        tools._http_get("https://evil.example.com/x", {}, 1.0)


def test_real_corpus_is_well_formed():
    c = Corpus(tools.CORPUS_DIR)
    if not c.docs:
        pytest.skip("no corpus yet")
    ids = [d.doc_id for d in c.docs]
    assert len(ids) == len(set(ids))
