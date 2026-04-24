"""pipeline/context.py - 流水线上下文：类型安全的数据容器

PipelineContext 是贯穿所有 Stage 的数据载体，替代原 tinyRAG 中
散落在函数参数/全局变量中的数据传递方式。

设计原则：
1. 所有 Stage 共享同一个 Context 实例
2. 每个 Stage 从 Context 读取输入、写入输出
3. Context 提供类型注解，IDE 可自动补全
4. 不使用强类型约束（dataclass），因为不同 Pipeline 所需字段不同
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from helpers.logger import logger


@dataclass
class PipelineContext:
    """流水线上下文 - 所有 Stage 共享的数据容器"""

    # ── 配置 ──────────────────────────────
    config: Any = None                    # Settings (Pydantic model)
    config_path: str = "config.yaml"

    # ── 数据库 ────────────────────────────
    db: Any = None                        # DatabaseManager
    db_path: str = ""

    # ── 扫描阶段 ──────────────────────────
    scan_report: Any = None               # ScanReport
    vault_configs: list[tuple[str, str]] = field(default_factory=list)   # [(name, path)]
    vault_excludes: dict[str, tuple[frozenset[str], list[str]]] = field(default_factory=dict)

    # ── 差异阶段 ──────────────────────────
    changed_paths: list[str] = field(default_factory=list)
    files_to_index: list[dict] = field(default_factory=list)
    force_rebuild: bool = False

    # ── 分块阶段 ──────────────────────────
    splitter: Any = None                  # MarkdownSplitter
    file_chunks: list[tuple[int, Any, str]] = field(default_factory=list)  # [(file_id, Chunk, file_path)]
    total_files_with_chunks: int = 0

    # ── 向量化阶段 ────────────────────────
    embed_engine: Any = None              # EmbeddingEngine
    chunk_embeddings: list[tuple[int, Any, str, list[float]]] = field(default_factory=list)  # (file_id, Chunk, path, embedding)

    # ── 入库阶段 ──────────────────────────
    total_indexed: int = 0
    global_chunk_idx: int = 0

    # ── 检索阶段 ──────────────────────────
    query: str = ""
    search_results: list[Any] = field(default_factory=list)  # list[RetrievalResult]
    search_mode: str = "hybrid"
    top_k: int = 10
    alpha: float | None = None
    beta: float | None = None
    vault_filter: list[str] | None = None

    # ── 运维阶段 ──────────────────────────
    maintenance_stats: dict = field(default_factory=dict)
    dry_run: bool = False

    # ── 运行时元数据 ──────────────────────
    start_time: float = field(default_factory=time.time)
    elapsed: float = 0.0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── 组件缓存（避免重复创建）───────────
    scanner: Any = None                   # Scanner
    retriever: Any = None                 # HybridEngine

    # ── 进度控制 ──────────────────────────
    quiet: bool = False                   # 静默模式（MCP stdio 下禁用 tqdm）

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        logger.error(msg)

    def finish(self) -> None:
        self.elapsed = time.time() - self.start_time

    def summary(self) -> dict[str, Any]:
        return {
            "elapsed": round(self.elapsed, 2),
            "errors": len(self.errors),
            "force_rebuild": self.force_rebuild,
            "files_to_index": len(self.files_to_index),
            "total_indexed": self.total_indexed,
            "search_results": len(self.search_results),
        }
