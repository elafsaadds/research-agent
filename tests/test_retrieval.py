from research_agent import tools
from research_agent.retrieval import source_text
from research_agent.tools import Corpus, SearchHit

DOC = "id: d-1\ntitle: T\nsource: s\n---\n" + ("Long body sentence. " * 40)


def test_corpus_hit_returns_full_body_not_the_snippet(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text(DOC, encoding="utf-8")
    monkeypatch.setattr(tools, "get_corpus", lambda: Corpus(tmp_path))
    hit = SearchHit(source_id="d-1", title="T", snippet="Long body", origin="corpus")
    text = source_text(hit)
    assert len(text) > 300 and text.startswith("Long body sentence.")


def test_arxiv_hit_returns_its_snippet():
    hit = SearchHit(source_id="arxiv:1", title="t", snippet="abstract text", origin="arxiv")
    assert source_text(hit) == "abstract text"


def test_unknown_corpus_id_falls_back_to_snippet(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "get_corpus", lambda: Corpus(tmp_path / "missing"))
    hit = SearchHit(source_id="nope", title="t", snippet="snippet", origin="corpus")
    assert source_text(hit) == "snippet"
