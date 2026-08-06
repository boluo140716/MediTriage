"""RAG 检索核心（混合检索编排，不依赖 LangChain）。

编排（混合召回 / RRF 融合 / 查询改写 / 重排 / 父节聚合）为命令式实现，
不依赖 LangChain：嵌入直接用 sentence-transformers 加载 BGE-M3，
切块走数据管线的 clean_chunk（add_prechunked）。
类名 LangChainRAG 为兼容历史 import 暂留。

架构：
  ├─ BGE-M3（sentence-transformers）多语言 1024 维嵌入
  ├─ jieba-BM25 稀疏召回 + 加权 RRF 融合
  ├─ 垃圾 chunk 过滤（页码/超短/低信息）+ 出口兜底
  └─ CrossEncoder(bge-reranker-v2-m3) 检索后重排序
  向量存储：本项目专属 Milvus Standalone（medical-milvus:19530）via 原生 MilvusClient
    启动：MediTriage/infra/start_medical_milvus.sh（milvusdb/milvus:v2.5.27，数据在 data/milvus_data）。

BGE-M3/reranker 放空闲 GPU（cuda:2），不占 vLLM 的 GPU1；失败降级 CPU。
"""
import json
import os
import re
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from loguru import logger

from pymilvus import MilvusClient

from meditriage.paths import (
    MILVUS_URI,
    EMBED_MODEL as DEFAULT_EMBED,
    RERANKER_MODEL as DEFAULT_RERANKER,
)

DEFAULT_COLLECTION = "medical_knowledge_m3"

# ---------------------------------------------------------------------------
# 在线嵌入 / 重排（阿里云百炼，本地零模型下载）：
#   MEDITRIAGE_EMBED_PROVIDER=dashscope  -> 嵌入走 text-embedding-v4 在线 API（默认 1024 维，与 BGE-M3 一致）
#   MEDITRIAGE_RERANK_PROVIDER=dashscope -> 重排走 gte-rerank-v2 在线 API
#   MEDITRIAGE_DASHSCOPE_API_KEY         -> 百炼 API Key（env 优先；或放 ~/.config/dashscope_api_key）
#   不设置任何 provider 时行为与原项目完全一致（本地 sentence-transformers）。
# ---------------------------------------------------------------------------
EMBED_PROVIDER = os.environ.get("MEDITRIAGE_EMBED_PROVIDER", "local").strip().lower()
RERANK_PROVIDER = os.environ.get("MEDITRIAGE_RERANK_PROVIDER", "local").strip().lower()


def _read_dashscope_api_key() -> str:
    """百炼 Key：env 优先，否则 ~/.config/dashscope_api_key。"""
    key = os.environ.get("MEDITRIAGE_DASHSCOPE_API_KEY", "").strip()
    if key:
        return key
    try:
        cfg = os.path.expanduser("~/.config/dashscope_api_key")
        if os.path.isfile(cfg):
            with open(cfg, encoding="utf-8") as f:
                return f.read().strip()
    except Exception:
        pass
    return ""


DASHSCOPE_API_KEY = _read_dashscope_api_key()
EMBED_API_BASE = os.environ.get(
    "MEDITRIAGE_EMBED_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
).rstrip("/")
EMBED_API_MODEL = os.environ.get("MEDITRIAGE_EMBED_MODEL", "text-embedding-v4")
# 注意：百炼重排接口路径是固定的 .../text-rerank/text-rerank（模型名在请求体内）
RERANK_API_BASE = os.environ.get(
    "MEDITRIAGE_RERANK_BASE_URL",
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
).rstrip("/")
RERANK_API_MODEL = os.environ.get("MEDITRIAGE_RERANK_MODEL", "gte-rerank-v2")



class _BGEEmbeddings:
    """BGE-M3 嵌入薄封装（sentence-transformers 直连）。

    提供 embed_query/embed_documents 接口，向量做归一化
    （normalize_embeddings=True），供 medical_memory / badcase_cluster 等
    消费方共用。
    """

    def __init__(self, model_name: str, device: str):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name, device=device)

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode(
            text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(
            list(texts), normalize_embeddings=True).tolist()


class _ApiEmbeddings:
    """阿里云百炼在线嵌入（OpenAI 兼容 /embeddings）。

    text-embedding-v4 默认 1024 维，与本地 BGE-M3 一致，Milvus 集合无需改动。
    接口与 _BGEEmbeddings 对齐：embed_query / embed_documents。
    """

    def __init__(self, model: str, base_url: str, api_key: str, dimension: int = 1024):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        self._model = model
        self._dimension = dimension

    @staticmethod
    def _batches(texts, size: int = 10):  # 百炼 text-embedding-v4 单次最多 10 条
        for i in range(0, len(texts), size):
            yield texts[i:i + size]

    def embed_query(self, text: str) -> list:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list) -> list:
        out = []
        for batch in self._batches(texts):
            resp = self._client.embeddings.create(
                model=self._model, input=batch, dimensions=self._dimension
            )
            emb = sorted(resp.data, key=lambda x: x.index)
            out.extend(e.embedding for e in emb)
        return out


class _ApiReranker:
    """阿里云百炼在线重排（gte-rerank-v2，原生 DashScope 接口）。

    接口与 CrossEncoder 对齐：predict([(query, doc), ...]) -> scores。
    失败时由上层捕获并降级为纯向量排序（不崩）。
    """

    def __init__(self, model: str, api_url: str, api_key: str, top_n: int = 10):
        self._model = model
        self._url = api_url  # 百炼重排固定端点；模型名在请求体内
        self._api_key = api_key
        self._top_n = top_n

    def predict(self, pairs) -> list:
        if not pairs:
            return []
        # 按 query 分组（_rerank_and_format 的 pairs 共享同一 query），每 query 一次请求
        from collections import OrderedDict
        groups = OrderedDict()
        for q, d in pairs:
            groups.setdefault(q, []).append(d)
        import httpx
        scores = []
        for q, docs in groups.items():
            payload = {
                "model": self._model,
                "input": {"query": q, "documents": docs},
                "parameters": {
                    "top_n": min(self._top_n, len(docs)),
                    "return_documents": False,
                },
            }
            resp = httpx.post(
                self._url,
                json=payload,
                timeout=60.0,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            results = (data.get("output") or {}).get("results") or []
            by_idx = {r.get("index"): r.get("relevance_score", 0.0) for r in results}
            scores.extend(by_idx.get(i, 0.0) for i in range(len(docs)))
        return scores


# 检索相关性门控（reranker sigmoid 分，相关≈0.99/无关≈0.002）。阈值只定义在这里，
# search-knowledge / clinical-guideline 两个 skill 直接 import，免得各写各的魔数；
# 按路由分档：guideline 面窄宜严，search_knowledge 含科普可略宽。
# env RAG_MIN_SCORE / RAG_MIN_SCORE_GUIDELINE 可覆盖以便调参。
RAG_MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.30"))
RAG_MIN_SCORE_GUIDELINE = float(os.environ.get("RAG_MIN_SCORE_GUIDELINE", "0.30"))

# RRF 稀疏路（BM25）相对权重：dense+rerank 在本语料已强，稀疏等权会注入相邻
# 疾病噪声拉低精排（实测 hybrid<dense）。降权让 dense 主导候选池，BM25 仅补充
# 术语精确命中。env RAG_SPARSE_WEIGHT 可覆盖以便 A/B 调参。
_SPARSE_WEIGHT = float(os.environ.get("RAG_SPARSE_WEIGHT", "0.5"))

_CJK = re.compile(r"[一-鿿]")
_ALNUM = re.compile(r"[A-Za-z一-鿿]")


def _pick_device(preferred: str = "cuda:2") -> str:
    try:
        import torch
        if torch.cuda.is_available():
            idx = int(preferred.split(":")[1]) if ":" in preferred else 0
            return preferred if idx < torch.cuda.device_count() else "cuda:0"
    except Exception as e:
        logger.debug(f"GPU 探测失败，回退 CPU: {e}")
    return "cpu"


def _rrf_fuse(*rankings, k: int = 60, weights: Optional[List[float]] = None):
    """Reciprocal Rank Fusion：融合任意条数的排序结果（支持 per-ranking 权重）。

    每个入参是一路按相关性降序排列的 [(id, _score)] 列表（只用名次不用分值）。
    每个 id 的融合分 = Σ wᵢ/(k + rank)，rank 从 0 起。weights 缺省各路等权。
    用权重区分模态：实测本语料 dense+rerank 已强，BM25 稀疏路若等权会把相邻
    疾病块注入候选池拉低精排，故对稀疏路降权，让 dense 主导候选池塑形。
    """
    fused: Dict[Any, float] = {}
    for idx, ranked in enumerate(rankings):
        w = weights[idx] if weights and idx < len(weights) else 1.0
        for rank, (doc_id, _score) in enumerate(ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


# BM25 医疗用户词典：jieba 默认会把这些缩写/复合术语切碎（"慢阻肺"→["慢阻","肺"]、
# "CHA2DS2-VASc"→["CHA2DS2","-","VASc"]），导致精确术语匹配失效。整体成词后 BM25
# 才能命中语料里的对应术语。覆盖评测 topic + 常见中英缩写，按需扩充。
_MED_TERMS = [
    "慢阻肺", "房颤", "冠心病", "心衰", "高血压", "糖尿病", "心梗", "脑梗",
    "瓣膜病", "主动脉瓣狭窄", "肺栓塞", "心肌病", "肥厚型心肌病", "血脂异常",
    "慢性肾病", "急性冠脉综合征", "糖化血红蛋白", "抗血小板", "抗凝",
    "CHA2DS2-VASc", "HbA1c", "LDL-C", "HFrEF", "COPD", "GOLD", "GINA",
    "KDIGO", "ACS", "CKD", "PCI", "ARNI", "SGLT2i", "MRA",
]
_jieba_ready = False


def _ensure_jieba():
    """挂载医疗用户词典（幂等，进程内只做一次）。"""
    global _jieba_ready
    if _jieba_ready:
        return
    import jieba
    for w in _MED_TERMS:
        jieba.add_word(w)
    _jieba_ready = True


_TOKEN_KEEP = re.compile(r"[A-Za-z0-9一-鿿]")


def _tokenize(text: str) -> List[str]:
    """中英文统一分词：jieba 切（中文按词、英文按串）→ 英文小写 → 滤纯标点 token。

    BM25 索引与查询必须用同一分词器，保证可比。医疗缩写/复合术语经用户词典
    整体成词（_MED_TERMS），英文统一小写消除大小写不匹配，纯标点 token 丢弃。
    """
    import jieba
    _ensure_jieba()
    toks = []
    for t in jieba.lcut(text or ""):
        t = t.strip()
        if t and _TOKEN_KEEP.search(t):  # 至少含一个字母/数字/中文，滤掉 "-" "?" "；"
            toks.append(t.lower())
    return toks


# 期刊 front-matter（刊头/编委名单/版权页/扉页）：无临床语义，且人名列表会污染
# 人名类检索。入库时过滤 + 检索出口兜底（防旧库存量块被返回）。指南 PDF
# 中还有编委会/目录/关键词/通讯作者等扉页样板可能躲过 garbage 过滤被召回，
# 一并堵掉。匹配块首 300 字内的整行标题。
_FRONT_MATTER = re.compile(
    r"DEPUTY EDITORS|EDITOR[\s-]?IN[\s-]?CHIEF|EDITORIAL BOARD"
    r"|ASSOCIATE EDITORS|PRINT ISSN|ONLINE ISSN"
    r"|THE JOURNAL OF CLINICAL AND APPLIED RESEARCH"
    r"|WRITING COMMITTEE MEMBERS?|TABLE OF CONTENTS"
    r"|PEER REVIEW COMMITTEE|CORRESPONDENCE TO",
    re.I,
)


def _is_garbage_chunk(text: str) -> bool:
    """过滤无语义 chunk：纯页码/超短/字母占比过低（页眉页脚/表格碎片/参考文献编号）"""
    t = text.strip()
    if len(t) < 60:
        return True
    # 字母+中文字符占比过低（大量数字/符号/空白 → 页码、表格残片）
    alnum = len(_ALNUM.findall(t))
    if alnum / max(len(t), 1) < 0.45:
        return True
    # 纯参考文献条目模式（大量 "数字. 作者" 或 DOI/年份堆叠）—— 粗判：数字占比高且无完整句
    digits = sum(c.isdigit() for c in t)
    if digits / max(len(t), 1) > 0.35:
        return True
    # 期刊刊头/编委名单等 front-matter
    if _FRONT_MATTER.search(t[:300]):
        return True
    return False


class LangChainRAG:
    """RAG 检索编排 + Milvus Standalone(原生 MilvusClient) 存储。

    类名为兼容历史 import 暂留，实现已不依赖 LangChain。
    """

    def __init__(
        self,
        uri: str = MILVUS_URI,
        collection_name: str = DEFAULT_COLLECTION,
        embed_model: str = DEFAULT_EMBED,
        reranker_model: str = DEFAULT_RERANKER,
        device: str = "cuda:2",
        use_reranker: bool = True,
        use_hybrid: bool = True,
        use_query_rewrite: Optional[bool] = None,
        expand_context: bool = True,
    ):
        self.uri = uri
        self.collection_name = collection_name
        self.device = _pick_device(device)
        self.use_reranker = use_reranker
        self.use_hybrid = use_hybrid
        # 父节聚合（small-to-big）：命中块按同 (doc_id, section) 取回兄弟 part
        # 合并为整节返回，缓解答案要素跨 chunk 分散（granularity）
        self.expand_context = expand_context
        # 检索前置 query 改写：未显式传则读环境 RAG_QUERY_REWRITE（默认开）；rewriter 懒加载
        if use_query_rewrite is None:
            use_query_rewrite = os.environ.get(
                "RAG_QUERY_REWRITE", "1"
            ).lower() not in ("0", "false", "no", "")
        self.use_query_rewrite = use_query_rewrite
        self._rewriter = None
        self._reranker = None
        self._reranker_lock = threading.Lock()
        self._reranker_path = reranker_model
        # RAG 结果 LRU 缓存：同会话内相同 (query, top_k, filter, rewrite) 命中时
        # 跳过整个检索链（rewrite/embed/BM25/rerank 全为在线 API，5-10s/次）。
        # demo 场景可接受知识库更新后缓存短暂过期；线程安全（多 agent 并发检索）。
        self._search_cache: "OrderedDict[tuple, list]" = OrderedDict()
        self._search_cache_size = 256
        self._search_cache_lock = threading.Lock()
        # hybrid(BM25) 状态：id↔content 映射 + BM25Okapi 索引（与 dense 路按 Milvus id 对齐）
        self._bm25 = None
        self._bm25_ids: List[int] = []
        self._bm25_content: Dict[int, str] = {}
        self._bm25_meta: Dict[int, dict] = {}
        # 父节聚合用：(doc_id, section) → [(part, content)]，随 BM25 一次性建好，
        # 检索时 O(1) 内存查兄弟块，避免每结果一次 Milvus LIKE 扫描
        self._section_index: Dict[tuple, List] = {}

        if EMBED_PROVIDER == "dashscope":
            if not DASHSCOPE_API_KEY:
                raise RuntimeError(
                    "MEDITRIAGE_EMBED_PROVIDER=dashscope 但未设置 MEDITRIAGE_DASHSCOPE_API_KEY"
                    "（或 ~/.config/dashscope_api_key）"
                )
            self.embeddings = _ApiEmbeddings(
                EMBED_API_MODEL, EMBED_API_BASE, DASHSCOPE_API_KEY
            )
            logger.info(
                f"LangChainRAG: embedding={EMBED_API_MODEL}(dashscope online) milvus={uri}"
            )
        else:
            logger.info(
                f"LangChainRAG: embedding=BGE-M3 device={self.device} milvus={uri}"
            )
            self.embeddings = _BGEEmbeddings(embed_model, self.device)
        self.embed_dim = len(self.embeddings.embed_query("dimension probe"))

        self.client = MilvusClient(uri=uri)  # Standalone server
        self._ensure_collection()

        if self.use_hybrid:
            self._build_bm25_index()  # 失败/空库内部降级 use_hybrid=False（不崩）

        logger.info(
            f"LangChainRAG ready: collection={collection_name} "
            f"dim={self.embed_dim} hybrid={self.use_hybrid} @ {uri}"
        )

    def _fetch_all_chunks(self) -> List[dict]:
        """拉全量 chunk（id+content+metadata）。用 query_iterator 分批，
        绕过 Milvus offset+limit≤16384 窗口限制，可扩展到任意规模。
        无 query_iterator 时退回单次 query（受窗口上限约束，仅小库可用）。
        """
        fields = ["id", "content", "metadata"]
        if hasattr(self.client, "query_iterator"):
            rows: List[dict] = []
            it = self.client.query_iterator(
                collection_name=self.collection_name,
                filter="id >= 0",
                output_fields=fields,
                batch_size=4000,
            )
            try:
                while True:
                    batch = it.next()
                    if not batch:
                        break
                    rows.extend(batch)
            finally:
                it.close()
            return rows
        # 退路：单次 query（Milvus 上限 16384）
        return self.client.query(
            self.collection_name, filter="id >= 0", output_fields=fields,
            limit=16384,
        )

    def _build_bm25_index(self):
        """拉全量 chunk content → jieba 分词 → 建 BM25Okapi 索引。
        空库或失败则降级 use_hybrid=False 并 warning，绝不抛出。
        """
        try:
            from rank_bm25 import BM25Okapi
            rows = self._fetch_all_chunks()
            ids, corpus_tokens = [], []
            for r in rows:
                rid = r.get("id")
                content = r.get("content", "") or ""
                if rid is None or not content.strip():
                    continue
                toks = _tokenize(content)
                if not toks:
                    continue
                ids.append(rid)
                corpus_tokens.append(toks)
                self._bm25_content[rid] = content
                try:
                    meta = json.loads(r.get("metadata", "{}"))
                except Exception as e:
                    meta = {}
                    logger.debug(f"BM25 metadata 解析失败 (id={rid}): {e}")
                self._bm25_meta[rid] = meta
                # 同时建父节索引：(doc_id, section) → [(part, content)]
                did, sec = meta.get("doc_id"), meta.get("section")
                if did and sec:
                    self._section_index.setdefault((did, sec), []).append(
                        (meta.get("part", 0), content)
                    )
            if not corpus_tokens:
                logger.warning(
                    "BM25 index: collection empty/no usable content, "
                    "downgrade use_hybrid=False"
                )
                self.use_hybrid = False
                return
            self._bm25 = BM25Okapi(corpus_tokens)
            self._bm25_ids = ids
            logger.info(
                f"BM25 index built: {len(ids)} chunks "
                f"(hybrid dense+BM25 RRF enabled)"
            )
        except Exception as e:
            logger.warning(
                f"BM25 index build failed, downgrade use_hybrid=False: {e}"
            )
            self.use_hybrid = False
            self._bm25 = None

    def _ensure_collection(self):
        if not self.client.has_collection(self.collection_name):
            # 显式 schema：content/metadata 存 VARCHAR（下游以 json.loads 解析），
            # 额外 mtype 标量字段 + 倒排索引做类型精确过滤（不依赖 metadata-LIKE 子串匹配）；
            # 向量用 FLAT 精确索引（万级语料零召回损失、延迟可忽略）。
            from pymilvus import DataType
            schema = self.client.create_schema(
                auto_id=True, enable_dynamic_field=False
            )
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field(
                "vector", DataType.FLOAT_VECTOR, dim=self.embed_dim
            )
            schema.add_field("content", DataType.VARCHAR, max_length=16384)
            schema.add_field("metadata", DataType.VARCHAR, max_length=8192)
            schema.add_field("mtype", DataType.VARCHAR, max_length=64)
            idx = self.client.prepare_index_params()
            idx.add_index(
                field_name="vector", index_type="FLAT", metric_type="COSINE"
            )
            idx.add_index(field_name="mtype", index_type="INVERTED")
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema, index_params=idx,
            )
            logger.info(
                f"Created collection: {self.collection_name} "
                f"(FLAT + mtype scalar index)"
            )
        self.client.load_collection(self.collection_name)

    def _get_reranker(self):
        if not self.use_reranker:
            return None
        if self._reranker is None:
            # 检索已下放线程池并发执行，双检锁防止两个线程同时加载
            # CrossEncoder（GPU 显存翻倍）
            with self._reranker_lock:
                if self._reranker is None and self.use_reranker:
                    try:
                        if RERANK_PROVIDER == "dashscope":
                            if not DASHSCOPE_API_KEY:
                                raise RuntimeError("缺少 MEDITRIAGE_DASHSCOPE_API_KEY")
                            self._reranker = _ApiReranker(
                                RERANK_API_MODEL, RERANK_API_BASE, DASHSCOPE_API_KEY
                            )
                            logger.info(
                                f"Reranker loaded: {RERANK_API_MODEL} (dashscope online)"
                            )
                        else:
                            from sentence_transformers import CrossEncoder
                            self._reranker = CrossEncoder(
                                self._reranker_path,
                                device=self.device,
                                max_length=512,
                            )
                            logger.info("Reranker loaded: bge-reranker-v2-m3")
                    except Exception as e:
                        logger.warning(f"Reranker load failed, vector-only: {e}")
                        self.use_reranker = False
        return self._reranker

    def add_prechunked(self, chunks: List[Dict[str, Any]]) -> int:
        """直接索引外部已切好的块（跳过内置 splitter）。chunks: [{content, metadata}]。
        用于 docling/marker 解析 + clean_chunk 结构化分块的生产流水线（build_rag_index_v2）。"""
        chunks = [c for c in chunks if (c.get("content") or "").strip()]
        if not chunks:
            return 0
        logger.info(f"Embedding {len(chunks)} pre-chunked blocks...")
        vectors = self.embeddings.embed_documents(
            [c["content"] for c in chunks]
        )
        data = [
            {
                "vector": vectors[i],
                "content": chunks[i]["content"],
                "metadata": json.dumps(
                    chunks[i].get("metadata", {}), ensure_ascii=False
                ),
                "mtype": str(chunks[i].get("metadata", {}).get("type", "")),
            }
            for i in range(len(chunks))
        ]
        B = 256
        for i in range(0, len(data), B):
            self.client.insert(self.collection_name, data[i:i + B])
        logger.info(f"Inserted {len(data)} pre-chunked blocks")
        return len(data)

    def _dense_search(self, qv, fetch_k: int, filter_type: Optional[str]):
        """dense 向量召回 → [{id, content, metadata, vdist}]（id 为 Milvus 主键，可能 None）"""
        # 标量字段精确过滤（mtype 带倒排索引），不用 metadata-LIKE 子串匹配
        flt = f'mtype == "{filter_type}"' if filter_type else None
        try:
            res = self.client.search(
                self.collection_name, data=[qv], limit=fetch_k, filter=flt,
                output_fields=["id", "content", "metadata"],
            )
        except Exception as e:
            logger.warning(f"filtered search failed ({e}), retry plain")
            res = self.client.search(
                self.collection_name, data=[qv], limit=fetch_k,
                output_fields=["id", "content", "metadata"],
            )
        hits = res[0] if res else []
        parsed = []
        for h in hits:
            ent = h.get("entity", {})
            try:
                meta = json.loads(ent.get("metadata", "{}"))
            except Exception as e:
                meta = {}
                logger.debug(f"检索结果 metadata 解析失败: {e}")
            parsed.append({
                "id": ent.get("id", h.get("id")),
                "content": ent.get("content", ""),
                "metadata": meta,
                "vdist": h.get("distance", 1.0),
            })
        return parsed

    def _rerank_and_format(
        self, query: str, parsed: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """对候选做 reranker 重排（不可用则按 vdist 兜底），返回 top_k 标准结构。

        输出标准结构 [{id, content, metadata, score}]，id 取 metadata.doc_id。
        """
        if not parsed:
            return []
        # 检索出口兜底：旧库存量的垃圾块（刊头/编委等）不返回给调用方
        cleaned = [p for p in parsed if not _is_garbage_chunk(p["content"])]
        if cleaned:
            parsed = cleaned
        reranker = self._get_reranker()
        if reranker is not None and len(parsed) > 1:
            scores = reranker.predict([(query, p["content"]) for p in parsed])
            order = sorted(
                range(len(parsed)), key=lambda i: scores[i], reverse=True
            )[:top_k]
            out = [
                {
                    "id": parsed[i]["metadata"].get("doc_id", ""),
                    "content": parsed[i]["content"],
                    "metadata": parsed[i]["metadata"],
                    "score": float(scores[i]),
                }
                for i in order
            ]
        else:
            # Milvus COSINE 的 distance 即余弦相似度（越大越相关），直接用作分数；
            # 不可写成 1-vdist：方向反转会使 reranker 不可用时最优结果反被下游阈值拒掉
            out = [
                {
                    "id": p["metadata"].get("doc_id", ""),
                    "content": p["content"],
                    "metadata": p["metadata"],
                    "score": float(max(0.0, min(1.0, p.get("vdist", 0.0)))),
                }
                for p in parsed[:top_k]
            ]
        return self._expand_siblings(out) if self.expand_context else out

    def _expand_siblings(
        self, results: List[Dict[str, Any]], max_chars: int = 2400
    ) -> List[Dict[str, Any]]:
        """父节聚合：对每个命中块取回同 (doc_id, section) 的兄弟 part 合并为整节。

        答案要素常被分块切分散在同一 section 的相邻 part（granularity）；命中单块
        只给片段，这里把整节按 part 顺序拼回（去重、限长），让下游拿到完整证据。
        用内存 _section_index O(1) 查兄弟（随 BM25 一次建好），不发 Milvus 查询；
        索引未建（如纯 dense 部署）/section 缺失/无兄弟则原样返回。
        """
        if not results or not self._section_index:
            return results
        for r in results:
            meta = r.get("metadata") or {}
            key = (meta.get("doc_id"), meta.get("section"))
            if not key[0] or not key[1]:
                continue
            sibs = self._section_index.get(key, [])
            if len(sibs) <= 1:
                continue
            ordered = sorted(
                sibs, key=lambda x: (x[0] if isinstance(x[0], int) else 0))
            merged, seen, total = [], set(), 0
            for _, content in ordered:
                c = (content or "").strip()
                if not c or c in seen:
                    continue
                if total + len(c) > max_chars and merged:
                    break
                merged.append(c)
                seen.add(c)
                total += len(c)
            if merged:
                r["content"] = "\n".join(merged)
        return results

    def _get_rewriter(self):
        """懒加载 QueryRewriter（首次启用改写时才建，失败不影响检索）。"""
        if self._rewriter is None:
            try:
                from meditriage.knowledge.query_rewrite import QueryRewriter
            except ImportError:  # 从 knowledge/ 直跑时
                from query_rewrite import QueryRewriter
            self._rewriter = QueryRewriter()
        return self._rewriter

    def _dense_ranked(self, parsed: List[Dict[str, Any]]) -> List:
        """dense 召回结果 → RRF 用的 [(id, 相似度)] 排序对（丢弃无 id 项）。

        RRF 只消费名次（列表序 = Milvus 降序），分值仅作诊断展示。
        """
        return [
            (p["id"], p.get("vdist", 0.0))
            for p in parsed if p.get("id") is not None
        ]

    def _bm25_ranked(self, tokens: List[str], fetch_k: int) -> List:
        """BM25 稀疏召回 → 分数最高的 fetch_k 个 [(id, score)]。"""
        import numpy as np
        bm_scores = self._bm25.get_scores(tokens)
        n_bm = min(fetch_k, len(self._bm25_ids))
        top_idx = np.argsort(bm_scores)[::-1][:n_bm]
        return [(self._bm25_ids[i], float(bm_scores[i])) for i in top_idx]

    def _bm25_candidate(
        self, rid, filter_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """从 BM25 映射构造候选 dict；该 id 无对应内容时返回 None。

        filter_type：BM25 全库打分不带类型过滤，在候选取回处补一道——
        否则 clinical_guideline 经稀疏路可拿到非指南块并包装成指南输出。
        vdist=0.0 表示"无 dense 相似度证据"的中性占位。
        """
        if rid not in self._bm25_content:
            return None
        meta = self._bm25_meta.get(rid, {})
        if filter_type and meta.get("type") != filter_type:
            return None
        return {
            "id": rid,
            "content": self._bm25_content[rid],
            "metadata": meta,
            "vdist": 0.0,
        }

    @staticmethod
    def _materialize_candidates(
        fused: List, pool: Dict[Any, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """按融合名次取回候选内容（pool 里没有的 id 跳过）。"""
        return [pool[i] for i, _ in fused if i in pool]

    def _search_multi(
        self, query: str, top_k: int, filter_type: Optional[str]
    ) -> List[Dict[str, Any]]:
        """改写多查询检索：原 query 与改写变体各自 dense 召回，再叠加变体 token
        并集的 BM25，多路 RRF 融合后用原 query rerank（意图锚定，过滤跑偏的变体）。

        变体≤1（改写失败/无新意）或无候选则抛异常，由 search() 兜底回原单
        query 路径。
        """
        variants = self._get_rewriter().rewrite(query)
        if len(variants) <= 1:
            raise RuntimeError("no_rewrite")  # 退回原路径（与未启用时完全一致）
        fetch_k = max(top_k * 4, 12)
        rankings: List[List] = []
        pool: Dict[Any, Dict[str, Any]] = {}

        # 每个变体一路 dense 召回。原 query（variants[0]）权重最高，改写变体递减，
        # 让候选池由原意图主导、变体只作补充召回（避免漂移变体塑形候选池）。
        # 提速：多路 dense 并行（embed + Milvus 各自独立请求），串行 N 次 -> 1 轮。
        weights: List[float] = []

        def _dense_one(v: str):
            try:
                qv = self.embeddings.embed_query(v)
                return self._dense_search(qv, fetch_k, filter_type)
            except Exception as e:
                logger.warning(f"variant dense search failed ({v!r}): {e}")
                return []

        if len(variants) > 1:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(len(variants), 4)
            ) as _ex:
                dense_results = list(_ex.map(_dense_one, variants))
        else:
            dense_results = [_dense_one(variants[0])]
        for vi, dv in enumerate(dense_results):
            rankings.append(self._dense_ranked(dv))
            weights.append(1.0 if vi == 0 else 0.6)
            for p in dv:
                if p.get("id") is not None:
                    pool.setdefault(p["id"], p)

        # BM25 一路：变体 token 并集（关键词富集，利于稀疏召回与跨语言术语）
        if self.use_hybrid and self._bm25 is not None:
            try:
                toks: List[str] = []
                for v in variants:
                    toks += _tokenize(v)
                sparse_ranked = self._bm25_ranked(toks, fetch_k)
                rankings.append(sparse_ranked)
                weights.append(_SPARSE_WEIGHT)
                for rid, _ in sparse_ranked:
                    if rid not in pool:
                        cand = self._bm25_candidate(rid, filter_type)
                        if cand is not None:
                            pool[rid] = cand
            except Exception as e:
                logger.warning(
                    f"multi-query BM25 failed ({e}), dense-only fuse"
                )

        fused = _rrf_fuse(*rankings, weights=weights)[:fetch_k]
        candidates = self._materialize_candidates(fused, pool)
        if not candidates:
            raise RuntimeError("empty_candidates")
        # 用原 query rerank（意图锚定）
        return self._rerank_and_format(query, candidates, top_k)

    def _cache_key(self, query: str, top_k: int,
                   filter_type: Optional[str],
                   rewrite: Optional[bool]) -> tuple:
        return (
            (query or "").strip(), top_k, filter_type or "",
            "default" if rewrite is None else ("rw" if rewrite else "no-rw"),
        )

    def _cache_get(self, key: tuple) -> Optional[List[Dict[str, Any]]]:
        with self._search_cache_lock:
            hit = self._search_cache.get(key)
            if hit is None:
                return None
            self._search_cache.move_to_end(key)
            return [dict(x) for x in hit]  # 浅拷贝，防调用方污染缓存

    def _cache_put(self, key: tuple, val: List[Dict[str, Any]]) -> None:
        with self._search_cache_lock:
            self._search_cache[key] = val
            self._search_cache.move_to_end(key)
            while len(self._search_cache) > self._search_cache_size:
                self._search_cache.popitem(last=False)

    def search(
        self,
        query: str,
        top_k: int = 3,
        filter_type: Optional[str] = None,
        rewrite: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """hybrid(dense+BM25 RRF) 或 dense-only 检索 + rerank -> [{id, content, metadata, score}]

        rewrite: None=按实例默认(self.use_query_rewrite)；True/False 显式覆盖本次。
        启用时先走改写多查询(_search_multi)，任何异常都安全兜底回下方原单 query 路径。
        带 LRU 缓存：相同检索参数命中时直接返回（省在线 embed/rerank）。
        """
        key = self._cache_key(query, top_k, filter_type, rewrite)
        hit = self._cache_get(key)
        if hit is not None:
            logger.debug(f"RAG cache hit: {str(query)[:30]}... ({len(hit)} 条)")
            return hit
        out = self._search_impl(query, top_k, filter_type, rewrite)
        self._cache_put(key, out)
        return out

    def _search_impl(
        self,
        query: str,
        top_k: int = 3,
        filter_type: Optional[str] = None,
        rewrite: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """（search 的内部实现，供缓存包装调用。）"""
        use_rw = self.use_query_rewrite if rewrite is None else rewrite
        if use_rw:
            try:
                out = self._search_multi(query, top_k, filter_type)
                if out:
                    return out
            except Exception as e:
                logger.warning(
                    f"query-rewrite search 兜底回单 query"
                    f"（{type(e).__name__}: {e}）"
                )

        # ===== 原单 query 路径 =====
        fetch_k = max(top_k * 4, 12)
        qv = self.embeddings.embed_query(query)
        dense = self._dense_search(qv, fetch_k, filter_type)

        # --- dense-only（A/B 对照，或 hybrid 不可用时）---
        if not self.use_hybrid or self._bm25 is None:
            return self._rerank_and_format(query, dense, top_k)

        # --- hybrid: dense ⊕ BM25 经 RRF 融合 → 取 fetch_k 候选 → rerank ---
        try:
            sparse_ranked = self._bm25_ranked(_tokenize(query), fetch_k)
            dense_ranked = self._dense_ranked(dense)
            fused = _rrf_fuse(
                dense_ranked, sparse_ranked, k=60,
                weights=[1.0, _SPARSE_WEIGHT],
            )[:fetch_k]

            # 候选对齐：dense 命中优先用其自带（含 vdist），否则查 BM25 映射
            pool: Dict[Any, Dict[str, Any]] = {
                p["id"]: p for p in dense if p.get("id") is not None
            }
            for doc_id, _ in fused:
                if doc_id not in pool:
                    cand = self._bm25_candidate(doc_id, filter_type)
                    if cand is not None:
                        pool[doc_id] = cand
            candidates = self._materialize_candidates(fused, pool)
            if not candidates:  # 极端：融合后无可取回内容 → 退回 dense
                return self._rerank_and_format(query, dense, top_k)
            return self._rerank_and_format(query, candidates, top_k)
        except Exception as e:
            logger.warning(
                f"hybrid search failed ({e}), fallback to dense-only"
            )
            return self._rerank_and_format(query, dense, top_k)
