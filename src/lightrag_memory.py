# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import inspect
import re
import shutil
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import requests

from .config import get_config
from .logger import get_logger
from .memory_interface import BaseMemorySystem, Evidence

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc, always_get_an_event_loop
except Exception as exc:  # pragma: no cover - import message is for runtime ergonomics
    raise ImportError(
        "无法导入 HKUDS LightRAG。请安装正确的包：`pip install lightrag-hku`，"
        "不要安装同名但不兼容的 `lightrag`。"
    ) from exc

logger = get_logger()
_VALID_QUERY_MODES = {"naive", "local", "global", "hybrid", "mix", "bypass"}


@dataclass
class _Buffers:
    chunks: List[str] = field(default_factory=list)
    full_parts: List[str] = field(default_factory=list)
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    doc_id_hint: Optional[str] = None

    def clear(self) -> None:
        self.chunks.clear()
        self.full_parts.clear()
        self.metadata.clear()
        self.doc_id_hint = None


class LightRAGMemory(BaseMemorySystem):
    """
    LightRAG backend for memoRaxis BaseMemorySystem.

    设计目标：
    - add_memory：仅做缓冲（与 memoRaxis chunk_context 的切块保持一致）
    - build_index：一次性写入 LightRAG 工作区（insert_custom_chunks）
    - retrieve：只做“检索返回证据”，不让 LightRAG 生成最终回答（回答由 R1/R2/R3 adaptor 生成）

    额外修复：
    - mode 真正参与检索，而不是仅存字段
    - 初始化只做一次，reset 后会重建 LightRAG 实例
    - 兼容新旧版 LightRAG：优先 query_data / aquery_data，缺失时回退到 only_need_context
    - 对非 naive 模式尽量用向量检索结果补 score，避免日志里全是 N/A
    """

    def __init__(self, working_dir: str, mode: str = "naive"):
        self.working_dir = str(Path(working_dir))
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)

        self.mode = self._normalize_mode(mode)
        self._buf = _Buffers()
        self._initialized = False

        conf = get_config()
        self._llm_conf = conf.llm
        self._emb_conf = conf.embedding

        self._embedding_func = self._build_embedding_func()
        self._llm_model_name, self._llm_model_func = self._build_llm_func()
        self._rag = self._build_rag_instance()
        self._validate_rag_runtime()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized = str(mode or "naive").strip().lower()
        if normalized not in _VALID_QUERY_MODES:
            logger.warning(
                "[LightRAGMemory] unknown mode=%r, fallback to 'naive'. valid=%s",
                mode,
                sorted(_VALID_QUERY_MODES),
            )
            return "naive"
        return normalized

    def _build_rag_instance(self) -> LightRAG:
        return LightRAG(
            working_dir=self.working_dir,
            embedding_func=self._embedding_func,
            llm_model_func=self._llm_model_func,
            llm_model_name=self._llm_model_name,
            # 显式指定存储后端，避免版本差异导致默认值变化
            kv_storage="JsonKVStorage",
            vector_storage="NanoVectorDBStorage",
            graph_storage="NetworkXStorage",
            doc_status_storage="JsonDocStatusStorage",
        )

    def _validate_rag_runtime(self) -> None:
        required_attrs = ("initialize_storages", "insert_custom_chunks", "query")
        missing = [name for name in required_attrs if not hasattr(self._rag, name)]
        if missing:
            raise RuntimeError(
                "检测到不兼容的 LightRAG 运行时，缺少属性："
                f"{missing}。请确认安装的是 `lightrag-hku`。"
            )

    def _build_embedding_func(self) -> EmbeddingFunc:
        emb_provider = str(self._emb_conf.get("provider", "openai")).lower()
        emb_model = str(self._emb_conf.get("model", "text-embedding-3-small"))
        emb_base_url = self._emb_conf.get("base_url")
        emb_api_key = self._emb_conf.get("api_key")
        emb_dim = int(self._emb_conf.get("dim", 1536))

        if emb_provider == "ark_multimodal":
            async def _ark_embed(texts: list[str], **kwargs: Any) -> np.ndarray:
                cleaned_texts = [(t or "").replace("\n", " ").strip() for t in texts]
                if not cleaned_texts:
                    return np.zeros((0, emb_dim), dtype=np.float32)

                url = (str(emb_base_url or "").rstrip("/") + "/embeddings/multimodal").strip()
                if not url.startswith("http"):
                    raise ValueError(
                        f"[LightRAGMemory] embedding.base_url 非法（ark_multimodal 需要 http(s)）：{emb_base_url}"
                    )
                if not emb_api_key:
                    raise ValueError("[LightRAGMemory] ark_multimodal 需要 embedding.api_key")

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {emb_api_key}",
                }
                payload = {
                    "model": emb_model,
                    "input": [{"type": "text", "text": text} for text in cleaned_texts],
                }

                def _post() -> Dict[str, Any]:
                    response = requests.post(url, headers=headers, json=payload, timeout=90)
                    response.raise_for_status()
                    return response.json()

                data = await asyncio.to_thread(_post)
                raw_embeddings = data.get("data")

                if isinstance(raw_embeddings, list):
                    embeddings = [item.get("embedding") for item in raw_embeddings]
                elif isinstance(raw_embeddings, dict) and "embedding" in raw_embeddings:
                    embeddings = [raw_embeddings["embedding"]]
                else:
                    raise RuntimeError(
                        f"[LightRAGMemory] ark_multimodal 返回格式异常：keys={list(data.keys())}"
                    )

                if not embeddings or any(emb is None for emb in embeddings):
                    raise RuntimeError("[LightRAGMemory] ark_multimodal 未返回 embedding")

                arr = np.asarray(embeddings, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.shape[0] != len(cleaned_texts):
                    raise RuntimeError(
                        "[LightRAGMemory] ark_multimodal 返回 embedding 数量与输入文本数量不一致："
                        f"input={len(cleaned_texts)}, output={arr.shape[0]}"
                    )
                if arr.shape[1] != emb_dim:
                    logger.warning(
                        "[LightRAGMemory] embedding 维度与配置不一致：config=%s, actual=%s",
                        emb_dim,
                        arr.shape[1],
                    )
                return arr

            return EmbeddingFunc(
                embedding_dim=emb_dim,
                max_token_size=8192,
                model_name=emb_model,
                func=_ark_embed,
            )

        openai_embed_impl = getattr(openai_embed, "func", openai_embed)
        _embed = partial(
            openai_embed_impl,
            model=emb_model,
            base_url=str(emb_base_url) if emb_base_url else None,
            api_key=str(emb_api_key) if emb_api_key else None,
        )
        return EmbeddingFunc(
            embedding_dim=emb_dim,
            max_token_size=getattr(openai_embed, "max_token_size", 8192),
            model_name=emb_model,
            func=_embed,
        )

    def _build_llm_func(self) -> tuple[str, Any]:
        llm_model = str(self._llm_conf.get("model", "gpt-4o-mini"))
        llm_base_url = self._llm_conf.get("base_url")
        llm_api_key = self._llm_conf.get("api_key")
        llm_timeout = int(self._llm_conf.get("timeout", 120))

        async def _llm_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> str:
            return await openai_complete_if_cache(
                model=llm_model,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                base_url=str(llm_base_url) if llm_base_url else None,
                api_key=str(llm_api_key) if llm_api_key else None,
                timeout=llm_timeout,
                **kwargs,
            )

        return llm_model, _llm_func

    def _ensure_ready(self) -> None:
        if self._initialized:
            return
        loop = always_get_an_event_loop()
        loop.run_until_complete(self._rag.initialize_storages())
        self._initialized = True

    def close(self) -> None:
        """显式释放 LightRAG storage 资源。"""
        if not self._initialized:
            return
        finalize = getattr(self._rag, "finalize_storages", None)
        if callable(finalize):
            try:
                loop = always_get_an_event_loop()
                loop.run_until_complete(finalize())
            except Exception as exc:  # pragma: no cover - finalization depends on backend state
                logger.warning("[LightRAGMemory] finalize_storages failed: %s", exc)
        self._initialized = False

    def _rebuild_rag_instance(self) -> None:
        self.close()
        self._rag = self._build_rag_instance()
        self._validate_rag_runtime()
        self._initialized = False

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _make_query_param(
        self,
        *,
        mode: Optional[str] = None,
        top_k: int = 5,
        chunk_top_k: Optional[int] = None,
        only_need_context: bool = False,
    ) -> QueryParam:
        resolved_mode = self._normalize_mode(mode or self.mode)
        resolved_top_k = max(self._safe_int(top_k, 5), 1)
        resolved_chunk_top_k = max(self._safe_int(chunk_top_k, resolved_top_k), 1)

        requested: Dict[str, Any] = {
            "mode": resolved_mode,
            "top_k": resolved_top_k,
            "chunk_top_k": resolved_chunk_top_k,
        }
        if only_need_context:
            requested["only_need_context"] = True

        try:
            signature = inspect.signature(QueryParam)
            supported = {k: v for k, v in requested.items() if k in signature.parameters}
            return QueryParam(**supported)
        except Exception:
            try:
                return QueryParam(**requested)
            except TypeError:
                return QueryParam(mode=resolved_mode)

    @staticmethod
    def _normalize_text_key(text: str) -> str:
        collapsed = re.sub(r"\s+", " ", (text or "").strip())
        return collapsed[:1000]

    def _run_chunks_vdb_query(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        loop = always_get_an_event_loop()
        results = loop.run_until_complete(self._rag.chunks_vdb.query(query, top_k=top_k))
        return list(results or [])

    def _build_score_lookup(self, query: str, top_k: int) -> Dict[str, float]:
        score_lookup: Dict[str, float] = {}
        try:
            raw_results = self._run_chunks_vdb_query(query, top_k=top_k)
        except Exception as exc:
            logger.warning("[LightRAGMemory] score enrichment via chunks_vdb.query failed: %s", exc)
            return score_lookup

        for item in raw_results:
            if not isinstance(item, dict):
                continue
            raw_score = item.get("distance")
            try:
                score = float(raw_score) if raw_score is not None else 0.0
            except Exception:
                score = 0.0

            chunk_id = str(item.get("id") or item.get("chunk_id") or "").strip()
            if chunk_id:
                score_lookup[chunk_id] = score_lookup.get(chunk_id, score)

            content = str(item.get("content") or "").strip()
            if content:
                score_lookup[self._normalize_text_key(content)] = score_lookup.get(
                    self._normalize_text_key(content),
                    score,
                )
        return score_lookup

    def _evidence_from_chunk_record(
        self,
        record: Dict[str, Any],
        *,
        rank: int,
        default_score: float = 0.0,
        metric: str = "cosine_sim",
        source: str = "LightRAG",
    ) -> Optional[Evidence]:
        content = str(record.get("content") or "").strip()
        if not content:
            return None

        chunk_id = str(record.get("chunk_id") or record.get("id") or "").strip()
        file_path = str(record.get("file_path") or "").strip()
        created_at = record.get("created_at")
        reference_id = str(record.get("reference_id") or "").strip()

        raw_score = record.get("score", record.get("distance"))
        try:
            score = float(raw_score) if raw_score is not None else float(default_score)
        except Exception:
            score = float(default_score)

        return Evidence(
            content=content,
            metadata={
                "source": source,
                "mode": self.mode,
                "rank": rank,
                "score": score,
                "metric": metric,
                "chunk_id": chunk_id,
                "file_path": file_path,
                "reference_id": reference_id,
                "created_at": created_at,
            },
        )

    def _records_to_evidences(
        self,
        records: Iterable[Dict[str, Any]],
        *,
        limit: int,
        default_score: float = 0.0,
        metric: str = "cosine_sim",
        source: str = "LightRAG",
    ) -> List[Evidence]:
        evidences: List[Evidence] = []
        seen: set[str] = set()

        for record in records:
            if not isinstance(record, dict):
                continue
            ev = self._evidence_from_chunk_record(
                record,
                rank=len(evidences),
                default_score=default_score,
                metric=metric,
                source=source,
            )
            if ev is None:
                continue

            dedupe_key = str(ev.metadata.get("chunk_id") or "").strip() or self._normalize_text_key(ev.content)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            evidences.append(ev)
            if len(evidences) >= limit:
                break
        return evidences

    def _query_structured_data(self, query: str, *, top_k: int) -> Optional[Dict[str, Any]]:
        query_data = getattr(self._rag, "query_data", None)
        aquery_data = getattr(self._rag, "aquery_data", None)
        query_param = self._make_query_param(
            mode=self.mode,
            top_k=max(top_k, 10),
            chunk_top_k=top_k,
            only_need_context=False,
        )

        if callable(query_data):
            try:
                return query_data(query, param=query_param)
            except TypeError:
                return query_data(query, query_param)
            except Exception as exc:
                logger.warning("[LightRAGMemory] query_data failed in mode=%s: %s", self.mode, exc)
                return None

        if callable(aquery_data):
            try:
                loop = always_get_an_event_loop()
                return loop.run_until_complete(aquery_data(query, param=query_param))
            except TypeError:
                loop = always_get_an_event_loop()
                return loop.run_until_complete(aquery_data(query, query_param))
            except Exception as exc:
                logger.warning("[LightRAGMemory] aquery_data failed in mode=%s: %s", self.mode, exc)
                return None

        return None

    def _structured_payload_to_evidences(self, payload: Dict[str, Any], *, query: str, top_k: int) -> List[Evidence]:
        if not isinstance(payload, dict):
            return []

        data_section = payload.get("data") or {}
        if not isinstance(data_section, dict):
            return []

        chunks = data_section.get("chunks") or []
        entities = data_section.get("entities") or []
        relationships = data_section.get("relationships") or []

        score_lookup = self._build_score_lookup(query, top_k=max(top_k * 4, 20))

        enriched_chunks: List[Dict[str, Any]] = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            chunk_copy = dict(chunk)
            chunk_id = str(chunk_copy.get("chunk_id") or chunk_copy.get("id") or "").strip()
            content_key = self._normalize_text_key(str(chunk_copy.get("content") or ""))
            chunk_copy["score"] = score_lookup.get(chunk_id, score_lookup.get(content_key, 0.0))
            enriched_chunks.append(chunk_copy)

        evidences = self._records_to_evidences(
            enriched_chunks,
            limit=top_k,
            default_score=0.0,
            metric="query_data+vector_enrichment",
            source="LightRAG",
        )
        if evidences:
            return evidences

        # 兼容极端情况：某些模式可能 chunks 为空，但实体 / 关系不为空。
        fallback_records: List[Dict[str, Any]] = []
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            entity_name = str(entity.get("entity_name") or entity.get("id") or "").strip()
            description = str(entity.get("description") or "").strip()
            if entity_name or description:
                fallback_records.append(
                    {
                        "id": entity_name,
                        "content": f"{entity_name}: {description}".strip(": "),
                        "file_path": entity.get("file_path", ""),
                        "reference_id": entity.get("reference_id", ""),
                        "score": 0.0,
                    }
                )

        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            src_id = str(rel.get("src_id") or "").strip()
            tgt_id = str(rel.get("tgt_id") or "").strip()
            description = str(rel.get("description") or "").strip()
            if src_id or tgt_id or description:
                fallback_records.append(
                    {
                        "id": f"{src_id}->{tgt_id}",
                        "content": f"{src_id} -> {tgt_id}: {description}".strip(": "),
                        "file_path": rel.get("file_path", ""),
                        "reference_id": rel.get("reference_id", ""),
                        "score": 0.0,
                    }
                )

        return self._records_to_evidences(
            fallback_records,
            limit=top_k,
            default_score=0.0,
            metric="structured_context",
            source="LightRAG",
        )

    def _split_context_blocks(self, text: str) -> List[str]:
        cleaned = (text or "").strip()
        if not cleaned:
            return []

        # 兼容常见 LightRAG only_need_context 文本输出：段落、标题、分隔线。
        blocks = re.split(r"\n\s*\n+|\n\s*[-=]{3,}\s*\n", cleaned)
        normalized_blocks: List[str] = []
        for block in blocks:
            piece = block.strip()
            if not piece:
                continue
            if re.fullmatch(r"[#=\-\s]+", piece):
                continue
            normalized_blocks.append(piece)
        return normalized_blocks

    def _context_result_to_evidences(self, context: Any, *, query: str, top_k: int) -> List[Evidence]:
        # 新版本如果开始直接返回结构化对象，这里直接复用。
        if isinstance(context, dict):
            structured = self._structured_payload_to_evidences(context, query=query, top_k=top_k)
            if structured:
                return structured

        if isinstance(context, list):
            records = [item for item in context if isinstance(item, dict)]
            if records:
                return self._records_to_evidences(
                    records,
                    limit=top_k,
                    default_score=0.0,
                    metric="context_list",
                    source="LightRAG",
                )

        if not isinstance(context, str):
            return []

        score_lookup = self._build_score_lookup(query, top_k=max(top_k * 4, 20))
        records: List[Dict[str, Any]] = []
        for block in self._split_context_blocks(context):
            records.append(
                {
                    "content": block,
                    "score": score_lookup.get(self._normalize_text_key(block), 0.0),
                }
            )

        return self._records_to_evidences(
            records,
            limit=top_k,
            default_score=0.0,
            metric="only_need_context",
            source="LightRAG",
        )

    def _retrieve_naive(self, query: str, top_k: int) -> List[Evidence]:
        try:
            results = self._run_chunks_vdb_query(query, top_k=top_k)
        except Exception as exc:
            logger.error("[LightRAGMemory] chunks_vdb.query failed: %s", exc)
            return []

        return self._records_to_evidences(
            results,
            limit=top_k,
            default_score=0.0,
            metric="cosine_sim",
            source="LightRAG",
        )

    def _retrieve_by_context_fallback(self, query: str, top_k: int) -> List[Evidence]:
        param = self._make_query_param(
            mode=self.mode,
            top_k=max(top_k, 10),
            chunk_top_k=top_k,
            only_need_context=True,
        )
        try:
            context = self._rag.query(query, param=param)
        except TypeError:
            context = self._rag.query(query, param)
        except Exception as exc:
            logger.warning("[LightRAGMemory] query(... only_need_context=True) failed: %s", exc)
            return []

        return self._context_result_to_evidences(context, query=query, top_k=top_k)

    # ------------------------------------------------------------------
    # BaseMemorySystem API
    # ------------------------------------------------------------------
    def add_memory(self, data: str, metadata: Dict[str, Any]) -> None:
        # metadata 这里先不强依赖，但保留 doc_id_hint，避免 ingest 脚本忘传 build_index(doc_id=...)
        content = (data or "").strip()
        if not content:
            return

        metadata = dict(metadata or {})
        self._buf.chunks.append(content)
        self._buf.full_parts.append(content)
        self._buf.metadata.append(metadata)

        if self._buf.doc_id_hint is None:
            for key in ("doc_id", "document_id", "source_id"):
                value = metadata.get(key)
                if value:
                    self._buf.doc_id_hint = str(value)
                    break

    def build_index(self, doc_id: Optional[str] = None) -> None:
        """
        非接口必需，但 ingest 脚本会用到。
        用 insert_custom_chunks 保持 chunks 与 memoRaxis 的 chunk_context 完全一致。
        """
        if not self._buf.chunks:
            logger.warning("[LightRAGMemory] build_index: buffer is empty, skip.")
            return

        self._ensure_ready()

        resolved_doc_id = doc_id or self._buf.doc_id_hint
        full_text = "\n\n".join(self._buf.full_parts) if self._buf.full_parts else "\n\n".join(self._buf.chunks)

        try:
            # 同步接口（内部会 run_until_complete）
            self._rag.insert_custom_chunks(
                full_text=full_text,
                text_chunks=list(self._buf.chunks),
                doc_id=str(resolved_doc_id) if resolved_doc_id else None,
            )
        except Exception:
            logger.exception("[LightRAGMemory] insert_custom_chunks failed (doc_id=%r)", resolved_doc_id)
            raise
        else:
            self._buf.clear()

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        self._ensure_ready()

        q = (query or "").strip()
        if not q:
            return []

        k = max(self._safe_int(top_k, 5), 1)

        if self.mode == "bypass":
            logger.warning("[LightRAGMemory] retrieve called with mode='bypass'; returning empty evidence list.")
            return []

        if self.mode == "naive":
            return self._retrieve_naive(q, k)

        structured_payload = self._query_structured_data(q, top_k=k)
        if structured_payload:
            evidences = self._structured_payload_to_evidences(structured_payload, query=q, top_k=k)
            if evidences:
                return evidences

        # 兼容旧版 LightRAG：没有 query_data / aquery_data 时回退。
        evidences = self._retrieve_by_context_fallback(q, k)
        if evidences:
            return evidences

        logger.warning(
            "[LightRAGMemory] mode=%s retrieval returned no structured evidence; fallback to naive chunks_vdb.query.",
            self.mode,
        )
        return self._retrieve_naive(q, k)

    def reset(self) -> None:
        """
        清空 LightRAG 工作区（用于重复实验）。
        注意：会删除 working_dir 下所有内容，并重建 LightRAG 实例。
        """
        self.close()

        path = Path(self.working_dir)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)

        self._buf.clear()
        self._rebuild_rag_instance()
