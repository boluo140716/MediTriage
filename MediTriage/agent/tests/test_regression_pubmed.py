"""PubMed 文献检索解析回归（离线）。

search_pubmed 走网络（NCBI E-utilities），这里用 monkeypatch 替换 urlopen，
喂固定的 esearch JSON / efetch XML，验证字段解析与降级行为——不依赖网络。
"""
import json

from meditriage.research import pubmed
from meditriage.research.pubmed import search_pubmed


_ESEARCH_JSON = json.dumps(
    {"esearchresult": {"idlist": ["111", "222"]}}
).encode()

_EFETCH_XML = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>111</PMID>
      <Article>
        <ArticleTitle>Nonpharmacologic Management of Hypertension</ArticleTitle>
        <Abstract><AbstractText>Lifestyle changes lower blood pressure.</AbstractText></Abstract>
        <Journal><Title>J Hypertens</Title></Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>222</PMID>
      <Article>
        <ArticleTitle>Statins in Primary Prevention</ArticleTitle>
        <Journal>
          <Title>Lancet</Title>
          <JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue>
        </Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


class _FakeResp:
    """模拟 urlopen 的上下文管理器返回值。"""

    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen(url, timeout=None):
    # 同一函数内先 esearch 后 efetch，按 URL 区分返回体
    if "esearch" in url:
        return _FakeResp(_ESEARCH_JSON)
    return _FakeResp(_EFETCH_XML)


def test_search_pubmed_parses_fields(monkeypatch):
    monkeypatch.setattr(pubmed.urllib.request, "urlopen", _fake_urlopen)

    out = search_pubmed("hypertension nonpharmacologic", max_results=2)

    assert len(out) == 2
    first = out[0]
    assert first["pmid"] == "111"
    assert "Hypertension" in first["title"]
    assert first["abstract"] == "Lifestyle changes lower blood pressure."
    assert first["journal"] == "J Hypertens"
    assert first["url"] == "https://pubmed.ncbi.nlm.nih.gov/111/"
    # 第二篇无摘要、有年份
    assert out[1]["abstract"] == ""
    assert out[1]["year"] == "2021"


def test_search_pubmed_empty_idlist(monkeypatch):
    def _empty(url, timeout=None):
        return _FakeResp(json.dumps({"esearchresult": {"idlist": []}}).encode())

    monkeypatch.setattr(pubmed.urllib.request, "urlopen", _empty)
    assert search_pubmed("no such topic") == []


def test_search_pubmed_network_failure_degrades(monkeypatch):
    def _boom(url, timeout=None):
        raise OSError("network unreachable")

    monkeypatch.setattr(pubmed.urllib.request, "urlopen", _boom)
    # 任何网络/解析异常都应降级为空列表（上层据此回退仅本地知识库）
    assert search_pubmed("anything") == []
