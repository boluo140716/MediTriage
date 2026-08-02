"""PubMed 文献检索（NCBI E-utilities）。

为 deep-research 提供可引的权威医学文献：esearch 取 PMID → efetch 取摘要。
纯标准库（urllib + xml.etree），任何网络/解析异常一律返回 []，由上层降级回
仅本地知识库——deep-research 因此无网也能工作。
"""
from typing import Dict, List
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from loguru import logger

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def search_pubmed(
    query: str, max_results: int = 5, timeout: float = 8.0
) -> List[Dict]:
    """检索 PubMed，返回 [{pmid, title, abstract, journal, year, url}]。

    任何网络/解析异常都返回 []（上层据此降级为仅本地知识库）。
    """
    try:
        es = urllib.parse.urlencode({
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        })
        with urllib.request.urlopen(
            f"{_EUTILS}/esearch.fcgi?{es}", timeout=timeout
        ) as resp:
            ids = (
                json.loads(resp.read())
                .get("esearchresult", {})
                .get("idlist", [])
            )
        if not ids:
            return []

        ef = urllib.parse.urlencode({
            "db": "pubmed",
            "id": ",".join(ids),
            "rettype": "abstract",
            "retmode": "xml",
        })
        with urllib.request.urlopen(
            f"{_EUTILS}/efetch.fcgi?{ef}", timeout=timeout
        ) as resp:
            root = ET.fromstring(resp.read())

        articles: List[Dict] = []
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//PMID") or ""
            title = (art.findtext(".//ArticleTitle") or "").strip()
            if not title:
                continue
            abstract = " ".join(
                (t.text or "") for t in art.findall(".//AbstractText")
            ).strip()
            journal = (art.findtext(".//Journal/Title") or "").strip()
            year = (
                art.findtext(".//PubDate/Year")
                or art.findtext(".//PubDate/MedlineDate")
                or ""
            ).strip()
            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": year,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
        logger.info(f"PubMed: {len(articles)} articles for '{query[:50]}'")
        return articles
    except Exception as e:
        logger.warning(f"PubMed search failed ({e}); 降级为仅本地知识库")
        return []
