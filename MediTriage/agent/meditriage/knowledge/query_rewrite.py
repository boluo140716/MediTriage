"""检索前置 Query 改写层（口语→临床术语 + 中英双写 + 相关诊断/同义词扩展）。

动机：不经改写时，用户 query 原样向量化/分词，口语化、缩写、中英混杂全压在 BGE-M3
embedding 上，跨语言召回不稳（中文口语 query 命中英文语料全靠 embedding 兜底）。
本层在检索前用 LLM 把 query 改写成若干检索友好的变体，交给下游做多查询融合，
**不改变** 现有 hybrid(dense+BM25 RRF)+rerank 链路。

实现：
  - LLM 用 config.LLM_CONFIG（demo 态=本地 vLLM/medix-r1-8b，实测改写延迟 ~0.5-1.2s）。
  - 结果缓存（同 query 不重复调用；一次 Agent 多轮检索里重复 query 零成本）。
  - 失败安全：超时/解析失败/服务不可达 → 退回 [原 query]，下游行为与未启用时一致。
  - 产出变体：原 query + 英文术语版 + 关键词扩展串（同义词/相关诊断，利于 BM25 与跨语言 dense）。
"""
import json
import os
import re
from collections import OrderedDict
from typing import Dict, List, Optional

from loguru import logger

_SYS_PROMPT = (
    "你是医学检索查询改写器。把患者口语化问题改写为检索友好的查询，"
    "用于在医学指南/科普知识库做向量+关键词检索。"
    "只输出一个JSON对象，不要解释或思考过程。格式："
    '{"clinical":"规范临床术语表述(中文)","english":"英文医学术语表述",'
    '"terms":["本病的同义词/缩写/中英别名,3-6个"]}。'
    "terms 只放同一疾病/实体的别名与缩写，不要放相关或相似的其他疾病名"
    "（避免把相邻疾病的内容召回进来）。"
)


def _load_llm_config() -> Dict:
    """读 config.LLM_CONFIG（跟随 Agent 用的底座）；失败则退回本地 vLLM 默认。"""
    try:
        from config import LLM_CONFIG  # 项目根 config.py
        if isinstance(LLM_CONFIG, dict) and LLM_CONFIG.get("base_url"):
            return LLM_CONFIG
    except Exception as e:
        logger.debug(f"config.LLM_CONFIG 不可用，退回本地 vLLM 默认: {e}")
    # 退路：直连容器内本地 vLLM（与 demo 默认一致）
    return {
        "base_url": os.environ.get(
            "RAG_REWRITE_BASE_URL", "http://localhost:8000/v1"
        ),
        "model_name": os.environ.get("RAG_REWRITE_MODEL", "medix-r1-8b"),
        "api_key": "not-needed",
    }


class QueryRewriter:
    """LLM 驱动的检索 query 改写器。线程安全足够（CPython dict + GIL），失败安全。"""

    def __init__(
        self,
        max_variants: int = 4,
        max_terms: int = 8,
        timeout: float = 4.0,
        cache_size: int = 512,
        circuit_breaker_threshold: int = 2,
    ):
        cfg = _load_llm_config()
        self.base_url = str(cfg.get("base_url", "")).rstrip("/")
        self.model = cfg.get("model_name") or cfg.get("model") or "medix-r1-8b"
        self.api_key = cfg.get("api_key", "not-needed") or "not-needed"
        self.max_variants = max_variants
        self.max_terms = max_terms
        self.timeout = timeout
        self._cache: "OrderedDict[str, List[str]]" = OrderedDict()
        self._cache_size = cache_size
        # 熔断：LLM 改写连续失败达到阈值 -> 本次运行期跳过改写（消灭
        # "每次失败每次白等"；成功一次即重置）
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self._consecutive_failures = 0
        logger.info(
            f"QueryRewriter: model={self.model} @ {self.base_url} "
            f"(timeout={timeout}s)"
        )

    @staticmethod
    def _is_clean_english(q: str) -> bool:
        """已是规范英文医学短句：高 ASCII 占比、无中文、长度适中。

        这类 query 改写只会注入一个中文 clinical 变体（把检索拉向中文 local_zh
        语料、稀释精确英文意图）和一个近重复 english 变体（白调一次 8B），
        故直接跳过改写。"""
        if not q or any("一" <= c <= "鿿" for c in q):
            return False
        ascii_letters = sum(c.isascii() and c.isalpha() for c in q)
        return ascii_letters >= 12 and ascii_letters / max(len(q), 1) >= 0.5

    @staticmethod
    def _is_simple_query(q: str) -> bool:
        """简短查询（<=12 字符且空格少）：改写增益有限且 LLM 调用常超时，
        直接原 query 检索（实测短查询 rewrite 失败率高、白耗 2-8s）。

        与 _is_clean_english 互补：此处覆盖中文/混合短句（"腰疼怎么办"、
        "眼睛干涩"）；规范英文短句已由 _is_clean_english 处理。"""
        if not q:
            return True
        q2 = q.strip()
        return len(q2) <= 12 and q2.count(" ") <= 3

    # ---- 公开接口 ----
    def rewrite(self, query: str) -> List[str]:
        """返回去重的查询变体列表，第一个恒为原 query。失败退回 [原 query]。"""
        q = (query or "").strip()
        if not q:
            return [query]
        if q in self._cache:
            self._cache.move_to_end(q)
            return self._cache[q]
        # pre-gate：规范英文短句 / 简短查询跳过 LLM 改写（确定性、零 8B 调用，
        # 避免中文变体污染；短查询改写增益有限且常超时）
        if self._is_clean_english(q) or self._is_simple_query(q):
            self._cache_put(q, [q])
            return [q]

        # 熔断：连续失败达到阈值 -> 本次运行期跳过 LLM 改写（直接原 query）
        if self._consecutive_failures >= self.circuit_breaker_threshold:
            logger.debug(
                f"rewrite 熔断中（连续失败 {self._consecutive_failures} 次），"
                f"直接返回原 query"
            )
            self._cache_put(q, [q])
            return [q]

        variants = [q]
        try:
            data = self._call_llm(q)
            for key in ("english", "clinical"):
                v = data.get(key)
                if isinstance(v, str) and v.strip():
                    variants.append(v.strip())
            terms = self._norm_terms(data.get("terms"))
            if terms:
                # 关键词富集串（利于 BM25/跨语言）
                variants.append(" ".join(terms[: self.max_terms]))
            self._consecutive_failures = 0  # 成功一次即重置熔断计数
        except Exception as e:
            logger.warning(
                f"query rewrite failed ({type(e).__name__}: {e}); 用原 query"
            )
            self._consecutive_failures += 1  # 失败累计（触发熔断）

        variants = self._dedup(variants)[: self.max_variants]
        self._cache_put(q, variants)
        return variants

    # ---- 内部 ----
    def _call_llm(self, query: str) -> Dict:
        import requests
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 256,
            "messages": [
                {"role": "system", "content": _SYS_PROMPT},
                {"role": "user", "content": query},
            ],
        }
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"].get("content", "") or ""
        return self._extract_json(content)

    @staticmethod
    def _extract_json(text: str) -> Dict:
        """从模型输出中提取最外层 JSON 对象（容忍前后多余文本/思维残留）。"""
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {}
        try:
            obj = json.loads(m.group())
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _norm_terms(self, terms) -> List[str]:
        """terms 归一：兼容 list、单元素含逗号串、纯字符串；切分+去空+去重。"""
        out: List[str] = []
        if isinstance(terms, str):
            terms = [terms]
        if not isinstance(terms, list):
            return out
        for t in terms:
            if not isinstance(t, str):
                continue
            # 模型偶尔把整串塞进一个元素（中英逗号混用），故再切分一次
            for piece in re.split(r"[,，;；]", t):
                p = piece.strip()
                if p:
                    out.append(p)
        return self._dedup(out)

    @staticmethod
    def _dedup(items: List[str]) -> List[str]:
        seen, out = set(), []
        for it in items:
            key = it.lower()
            if key not in seen:
                seen.add(key)
                out.append(it)
        return out

    def _cache_put(self, q: str, variants: List[str]):
        self._cache[q] = variants
        self._cache.move_to_end(q)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
