#!/usr/bin/env python3
"""mcp_adapter/server.py - MCP Server 适配层 (v2.0)

通过复用 Pipeline Stage 实现 MCP Tool/Prompt/Resource，消除原 tinyRAG 中
server.py 与 build_index.py / rag_cli.py 的3处代码重复。

核心思路：
  - MCP Tool   → 构造 PipelineContext → 执行 Pipeline → 返回 JSON 结果
  - MCP Prompt → 检索 + 模板渲染 → 返回 PromptMessage 列表
  - MCP Resource → 直接查询 DB → 返回资源内容
  - 所有业务逻辑在 Stage 中，MCP 层仅做协议适配
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, ClassVar

# 确保项目路径在 sys.path
_project_root = Path(__file__).parent.parent
_tinyrag_root = _project_root
if str(_tinyrag_root) not in sys.path:
    sys.path.insert(0, str(_tinyrag_root))

from pipeline.context import PipelineContext
from pipeline.pipeline import Pipeline
from pipeline.events import EventBus
from pipeline.stages import (
    ConfigLoadStage,
    ScanStage,
    DiffStage,
    ChunkStage,
    EmbedStage,
    IndexStage,
    SearchStage,
    CleanupStage,
    VacuumStage,
)
from helpers.logger import setup_logger

logger = setup_logger(level="INFO")

# MCP SDK
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        GetPromptResult,
        Prompt,
        PromptArgument,
        PromptMessage,
        ReadResourceResult,
        Resource,
        ResourceTemplate,
        TextContent,
        TextResourceContents,
        Tool,
    )

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


# ── Pipeline 工厂 ────────────────────────────────────


def _build_pipeline(name: str, stages: list) -> Pipeline:
    """创建 Pipeline 实例"""
    return Pipeline(name, stages, event_bus=EventBus())


def _run_pipeline_sync(
    name: str, stages: list, ctx: PipelineContext
) -> PipelineContext:
    """同步执行 Pipeline 并处理数据库关闭"""
    pipeline = _build_pipeline(name, stages)
    result, ctx = pipeline.run(ctx)
    if not result.success:
        raise RuntimeError(f"Pipeline [{name}] 失败: {'; '.join(ctx.errors)}")
    return ctx


# ── Prompt 模板加载 ──────────────────────────────────

PROMPTS_DIR = _project_root / "prompts"

# 默认模板（文件不存在时的回退）
DEFAULT_TPL_SEARCH = (
    "你是一个知识库助手。请基于以下检索结果回答用户问题。\n"
    "## 用户问题\n{{query}}\n\n## 知识库检索结果\n{{context}}"
)
DEFAULT_TPL_SUMMARIZE = (
    "请总结以下文档的核心内容。\n"
    "## 文档信息\n路径: {{file_path}}\n\n## 文档内容\n{{content}}"
)


def _load_prompt_template(filename: str, default: str) -> str:
    """从 prompts/ 目录加载模板，不存在则返回默认模板"""
    p = PROMPTS_DIR / filename
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return default
    return default


def _render_template(tpl: str, variables: dict[str, Any]) -> str:
    """简单的 {{key}} 模板渲染"""
    for k, v in variables.items():
        tpl = tpl.replace(f"{{{{{k}}}}}", str(v) if v is not None else "")
    return tpl


# ── AppContext ────────────────────────────────────────


class AppContext:
    """MCP 应用上下文（延迟初始化）"""

    def __init__(self):
        self.config = None
        self.db = None
        self.vault_configs = []
        self.vault_excludes = {}
        self._initialized = False
        self._lock = asyncio.Lock()
        self._background_tasks: list[asyncio.Task] = []
        # Prompt 模板（延迟加载）
        self._tpl_search: str | None = None
        self._tpl_summarize: str | None = None

    async def initialize(self):
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            # 使用 Pipeline 初始化配置和数据库
            ctx = PipelineContext(
                quiet=True, config_path=str(_project_root / "config.yaml")
            )
            ctx = _run_pipeline_sync("init", [ConfigLoadStage(), ScanStage()], ctx)
            self.config = ctx.config
            self.db = ctx.db
            self.vault_configs = ctx.vault_configs
            self.vault_excludes = ctx.vault_excludes
            # 加载 Prompt 模板
            self._tpl_search = _load_prompt_template(
                "prompt_search_with_context.md", DEFAULT_TPL_SEARCH
            )
            self._tpl_summarize = _load_prompt_template(
                "prompt_summarize_document.md", DEFAULT_TPL_SUMMARIZE
            )
            self._initialized = True
            logger.info("MCP AppContext initialized via Pipeline")

    @property
    def tpl_search(self) -> str:
        return self._tpl_search or DEFAULT_TPL_SEARCH

    @property
    def tpl_summarize(self) -> str:
        return self._tpl_summarize or DEFAULT_TPL_SUMMARIZE

    def add_background_task(self, task: asyncio.Task):
        self._background_tasks.append(task)

        def _on_done(t: asyncio.Task):
            with contextlib.suppress(ValueError):
                self._background_tasks.remove(t)

        task.add_done_callback(_on_done)

    async def shutdown(self):
        for t in self._background_tasks:
            t.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        if self.db:
            try:
                self.db.close()
            except Exception:
                pass


# ── 搜索辅助函数 ─────────────────────────────────────


def _do_search(
    ctx: AppContext,
    query: str,
    top_k: int = 10,
    alpha: float | None = None,
    beta: float | None = None,
    mode: str = "hybrid",
    vaults: list[str] | None = None,
) -> list:
    """同步执行搜索，返回结果列表"""
    # mode → alpha/beta
    if mode == "keyword":
        alpha, beta = 0.0, 1.0
    elif mode == "semantic":
        alpha, beta = 1.0, 0.0

    pipeline_ctx = PipelineContext(
        config=ctx.config,
        db=ctx.db,
        query=query,
        top_k=top_k,
        search_mode=mode,
        alpha=alpha,
        beta=beta,
        vault_filter=vaults,
        quiet=True,
    )
    search_stage = SearchStage()
    search_stage.process(pipeline_ctx)
    return pipeline_ctx.search_results


def _format_search_context(results: list) -> str:
    """将搜索结果格式化为 Prompt 上下文文本"""
    if not results:
        return "无检索结果"
    lines = []
    for i, r in enumerate(results):
        score_info = f"综合={r.final_score:.3f}"
        if hasattr(r, "semantic_score") and r.semantic_score is not None:
            score_info += f" 语义={r.semantic_score:.3f}"
        if hasattr(r, "keyword_score") and r.keyword_score is not None:
            score_info += f" 关键词={r.keyword_score:.3f}"
        confidence = (
            f"置信度={r.confidence_score:.2f}" if hasattr(r, "confidence_score") else ""
        )
        lines.append(
            f"【文档 {i+1}】{r.file_path}\n"
            f"  章节: {getattr(r, 'section', '')}\n"
            f"  评分: {score_info} {confidence}\n"
            f"  内容: {r.content[:500]}"
        )
    return "\n\n".join(lines)


def _get_document_chunks(db, file_path: str) -> dict[str, Any] | None:
    """根据文件路径从 DB 获取文档及其所有 chunks"""
    # 精确匹配
    row = db.conn.execute(
        "SELECT id, file_path, vault_name, absolute_path FROM files WHERE file_path = ? AND is_deleted = 0",
        (file_path,),
    ).fetchone()
    # 模糊匹配回退
    if not row:
        row = db.conn.execute(
            "SELECT id, file_path, vault_name, absolute_path FROM files WHERE file_path LIKE ? AND is_deleted = 0 LIMIT 1",
            (f"%{os.path.basename(file_path)}",),
        ).fetchone()
    if not row:
        return None

    file_id = row["id"]
    chunks = db.conn.execute(
        "SELECT content, content_type, section_title, section_path, confidence_json, chunk_index "
        "FROM chunks WHERE file_id = ? AND is_deleted = 0 ORDER BY chunk_index",
        (file_id,),
    ).fetchall()

    # 解析 confidence_json
    conf = {}
    if chunks and chunks[0]["confidence_json"]:
        try:
            conf = json.loads(chunks[0]["confidence_json"])
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "file_id": file_id,
        "file_path": row["file_path"],
        "vault_name": row["vault_name"],
        "absolute_path": row["absolute_path"],
        "chunks": [dict(c) for c in chunks],
        "confidence": conf,
        "section_count": len(chunks),
        "char_count": sum(len(c["content"]) for c in chunks),
    }


def _format_document_content(doc_info: dict) -> str:
    """将文档 chunks 格式化为 Prompt 内容文本"""
    lines = []
    for c in doc_info["chunks"]:
        section = c.get("section_title") or "正文"
        lines.append(f"### {section}\n{c['content']}")
    content = "\n\n".join(lines)
    # 截断过长内容
    if len(content) > 8000:
        content = content[:8000] + "\n\n...(内容过长，已截断)"
    return content


# ── Tool 实现 ────────────────────────────────────────


async def tool_search(args: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """搜索 Tool - 复用 SearchStage"""
    await ctx.initialize()

    query = args.get("query", "")
    mode = args.get("mode", "hybrid")
    top_k = min(max(args.get("top_k", 10), 1), 100)
    alpha = args.get("alpha")
    beta = args.get("beta")
    vaults_arg = args.get("vaults")

    results = _do_search(
        ctx, query, top_k=top_k, alpha=alpha, beta=beta, mode=mode, vaults=vaults_arg
    )

    return {
        "query": query,
        "total": len(results),
        "results": [
            {
                "rank": i + 1,
                "file": r.file_path,
                "abs_path": r.absolute_path,
                "content": r.content[:300],
                "score": round(r.final_score, 4),
                "confidence": round(r.confidence_score, 4),
                "confidence_reason": r.confidence_reason,
            }
            for i, r in enumerate(results)
        ],
    }


async def tool_search_with_context(
    args: dict[str, Any], ctx: AppContext
) -> dict[str, Any]:
    """检索+上下文 Tool - 执行搜索并返回格式化的 Prompt 上下文

    与 search Tool 的区别：
    - search 返回结构化 JSON 结果列表
    - search_with_context 返回已渲染的 Prompt 文本，可直接注入 LLM 对话
    """
    await ctx.initialize()

    query = args.get("query", "")
    mode = args.get("mode", "hybrid")
    top_k = min(max(args.get("top_k", 10), 1), 100)
    alpha = args.get("alpha")
    beta = args.get("beta")
    vaults_arg = args.get("vaults")

    start = time.time()
    results = _do_search(
        ctx, query, top_k=top_k, alpha=alpha, beta=beta, mode=mode, vaults=vaults_arg
    )
    elapsed = round(time.time() - start, 3)

    # 格式化上下文
    context_text = _format_search_context(results)

    # 渲染 Prompt 模板
    prompt_text = _render_template(
        ctx.tpl_search,
        {
            "query": query,
            "context": context_text,
            "search_mode": mode,
            "result_count": len(results),
            "elapsed": f"{elapsed}s",
        },
    )

    return {
        "query": query,
        "total": len(results),
        "elapsed": elapsed,
        "prompt": prompt_text,
        "results_summary": [
            {"rank": i + 1, "file": r.file_path, "score": round(r.final_score, 4)}
            for i, r in enumerate(results)
        ],
    }


async def tool_summarize_document(
    args: dict[str, Any], ctx: AppContext
) -> dict[str, Any]:
    """文档摘要 Tool - 获取文档全部内容并渲染摘要 Prompt

    输入文件路径，输出已渲染的摘要 Prompt 文本，可直接注入 LLM 对话。
    """
    await ctx.initialize()

    file_path = args.get("file_path", "")
    if not file_path:
        return {"error": "file_path 参数不能为空"}

    doc_info = _get_document_chunks(ctx.db, file_path)
    if not doc_info:
        return {"error": f"未找到文件: {file_path}", "file_path": file_path}

    content = _format_document_content(doc_info)
    conf = doc_info["confidence"]

    # 渲染摘要 Prompt 模板
    prompt_text = _render_template(
        ctx.tpl_summarize,
        {
            "file_path": doc_info["file_path"],
            "doc_type": conf.get("doc_type", "未知"),
            "status": conf.get("status", ""),
            "final_date": conf.get("final_date", ""),
            "section_count": doc_info["section_count"],
            "char_count": doc_info["char_count"],
            "content": content,
        },
    )

    return {
        "file_path": doc_info["file_path"],
        "vault_name": doc_info["vault_name"],
        "section_count": doc_info["section_count"],
        "char_count": doc_info["char_count"],
        "doc_type": conf.get("doc_type", ""),
        "final_date": conf.get("final_date", ""),
        "prompt": prompt_text,
    }


async def tool_scan_index(args: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """增量扫描 Tool - 复用完整 Scan Pipeline"""
    await ctx.initialize()

    pipeline_ctx = PipelineContext(
        config=ctx.config,
        db=ctx.db,
        vault_configs=ctx.vault_configs,
        vault_excludes=ctx.vault_excludes,
        force_rebuild=False,
        quiet=True,
    )

    # 执行 Scan + Diff + Chunk + Embed + Index
    stages = [ScanStage(), DiffStage(), ChunkStage(), EmbedStage(), IndexStage()]
    pipeline_ctx = _run_pipeline_sync("mcp_scan", stages, pipeline_ctx)

    report = pipeline_ctx.scan_report
    return {
        "status": "success",
        "summary": report.summary() if report else "no report",
        "new": len(report.new_files) if report else 0,
        "modified": len(report.modified_files) if report else 0,
        "moved": len(report.moved_files) if report else 0,
        "deleted": len(report.deleted_files) if report else 0,
        "indexed_chunks": pipeline_ctx.total_indexed,
    }


async def tool_rebuild_index(args: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """全量重建 Tool - 后台执行 Build Pipeline"""
    await ctx.initialize()

    async def _background_rebuild():
        try:
            logger.info("Starting background index rebuild via Pipeline...")
            pipeline_ctx = PipelineContext(
                config=ctx.config,
                db=ctx.db,
                vault_configs=ctx.vault_configs,
                vault_excludes=ctx.vault_excludes,
                force_rebuild=True,
                quiet=True,
            )
            stages = [
                ScanStage(),
                DiffStage(),
                ChunkStage(),
                EmbedStage(),
                IndexStage(),
            ]
            _run_pipeline_sync("mcp_rebuild", stages, pipeline_ctx)
            logger.info("Background rebuild completed")
        except Exception as e:
            logger.error(f"Background rebuild failed: {e}", exc_info=True)

    task = asyncio.create_task(_background_rebuild())
    ctx.add_background_task(task)
    return {"status": "started", "message": "Index rebuild running in background"}


async def tool_stats(args: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """统计 Tool"""
    await ctx.initialize()
    db = ctx.db
    files_total = db.conn.execute(
        "SELECT COUNT(*) FROM files WHERE is_deleted=0"
    ).fetchone()[0]
    files_by_vault = db.conn.execute(
        "SELECT vault_name, COUNT(*) as cnt FROM files WHERE is_deleted=0 GROUP BY vault_name"
    ).fetchall()
    chunks_total = db.conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE is_deleted=0"
    ).fetchone()[0]
    try:
        vectors_total = db.conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    except Exception:
        vectors_total = 0
    db_size = os.path.getsize(ctx.config.db_path) if ctx.config.db_path else 0
    return {
        "files": {
            "total": files_total,
            "by_vault": {row["vault_name"]: row["cnt"] for row in files_by_vault},
        },
        "chunks": {
            "total": chunks_total,
            "avg_per_file": round(chunks_total / max(files_total, 1), 1),
        },
        "vectors": {
            "total": vectors_total,
            "dimensions": ctx.config.embedding_model.dimensions,
        },
        "storage": {
            "db_size_mb": round(db_size / 1024 / 1024, 2),
            "db_path": str(ctx.config.db_path),
        },
        "model": {
            "name": ctx.config.embedding_model.name,
            "size": ctx.config.embedding_model.size,
        },
    }


# ── Tool Schema ──────────────────────────────────────

TOOL_SCHEMAS = {
    "search": {
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "mode": {
                    "type": "string",
                    "enum": ["semantic", "keyword", "hybrid"],
                    "default": "hybrid",
                },
                "top_k": {
                    "type": "integer",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 100,
                },
                "alpha": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "beta": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "vaults": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
        },
        "description": "Hybrid knowledge retrieval - returns structured JSON results (tinyRAG Pipeline)",
    },
    "search_with_context": {
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "mode": {
                    "type": "string",
                    "enum": ["semantic", "keyword", "hybrid"],
                    "default": "hybrid",
                    "description": "检索模式：semantic(纯语义), keyword(纯关键词), hybrid(混合)",
                },
                "top_k": {
                    "type": "integer",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 100,
                    "description": "返回结果数量",
                },
                "alpha": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "语义检索权重 (0.0-1.0)",
                },
                "beta": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "关键词检索权重 (0.0-1.0)",
                },
                "vaults": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "指定检索的仓库名称列表",
                },
            },
            "required": ["query"],
        },
        "description": "Search with context prompt - executes RAG retrieval and returns a ready-to-use LLM prompt with search results as context (tinyRAG Pipeline)",
    },
    "summarize_document": {
        "schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "文档文件路径（支持精确匹配和文件名模糊匹配）",
                },
            },
            "required": ["file_path"],
        },
        "description": "Summarize document - retrieves all chunks of a document and returns a ready-to-use LLM prompt for summarization (tinyRAG Pipeline)",
    },
    "scan_index": {
        "schema": {"type": "object", "properties": {}},
        "description": "Incrementally scan and update file index (tinyRAG Pipeline)",
    },
    "rebuild_index": {
        "schema": {"type": "object", "properties": {}},
        "description": "Force rebuild full knowledge index (tinyRAG Pipeline)",
    },
    "stats": {
        "schema": {"type": "object", "properties": {}},
        "description": "Get knowledge base statistics (tinyRAG Pipeline)",
    },
}

TOOL_HANDLERS = {
    "search": tool_search,
    "search_with_context": tool_search_with_context,
    "summarize_document": tool_summarize_document,
    "scan_index": tool_scan_index,
    "rebuild_index": tool_rebuild_index,
    "stats": tool_stats,
}


# ── Prompt 管理 ──────────────────────────────────────

PROMPT_DEFINITIONS = [
    Prompt(
        name="search_with_context",
        description="RAG 检索回答 Prompt - 执行检索并将结果渲染为 LLM 可用的上下文提示",
        arguments=[
            PromptArgument(name="query", description="搜索关键词", required=True),
            PromptArgument(
                name="top_k", description="返回结果数量 (默认5)", required=False
            ),
            PromptArgument(
                name="mode",
                description="检索模式: hybrid/keyword/semantic",
                required=False,
            ),
            PromptArgument(
                name="alpha", description="语义权重 (0.0-1.0)", required=False
            ),
            PromptArgument(
                name="beta", description="关键词权重 (0.0-1.0)", required=False
            ),
            PromptArgument(
                name="vaults", description="指定仓库 (逗号分隔)", required=False
            ),
        ],
    ),
    Prompt(
        name="summarize_document",
        description="文档摘要 Prompt - 获取文档全部内容并渲染为 LLM 可用的摘要提示",
        arguments=[
            PromptArgument(name="file_path", description="文档文件路径", required=True),
        ],
    ),
]


async def handle_get_prompt(
    name: str, arguments: dict[str, str], ctx: AppContext
) -> GetPromptResult:
    """处理 MCP get_prompt 请求"""
    await ctx.initialize()

    if name == "search_with_context":
        return _prompt_search_with_context(arguments, ctx)
    elif name == "summarize_document":
        return _prompt_summarize_document(arguments, ctx)
    else:
        raise ValueError(f"Unknown prompt: {name}")


def _prompt_search_with_context(
    args: dict[str, str], ctx: AppContext
) -> GetPromptResult:
    """渲染 search_with_context Prompt"""
    query = args.get("query", "")
    top_k = int(args.get("top_k", "5"))
    mode = args.get("mode", "hybrid")

    alpha_str = args.get("alpha")
    beta_str = args.get("beta")
    alpha = float(alpha_str) if alpha_str else None
    beta = float(beta_str) if beta_str else None

    vaults_arg = None
    vaults_str = args.get("vaults")
    if vaults_str:
        vaults_arg = [v.strip() for v in vaults_str.split(",") if v.strip()]

    start = time.time()
    results = _do_search(
        ctx, query, top_k=top_k, alpha=alpha, beta=beta, mode=mode, vaults=vaults_arg
    )
    elapsed = round(time.time() - start, 3)

    context_text = _format_search_context(results)

    prompt_text = _render_template(
        ctx.tpl_search,
        {
            "query": query,
            "context": context_text,
            "search_mode": mode,
            "result_count": len(results),
            "elapsed": f"{elapsed}s",
        },
    )

    return GetPromptResult(
        description=f"检索回答: {query}",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=prompt_text),
            )
        ],
    )


def _prompt_summarize_document(
    args: dict[str, str], ctx: AppContext
) -> GetPromptResult:
    """渲染 summarize_document Prompt"""
    file_path = args.get("file_path", "")
    if not file_path:
        return GetPromptResult(
            description="摘要: (错误)",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text", text="错误：file_path 参数不能为空"
                    ),
                )
            ],
        )

    doc_info = _get_document_chunks(ctx.db, file_path)
    if not doc_info:
        return GetPromptResult(
            description=f"摘要: {file_path}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text", text=f"错误：未找到文件 {file_path}"
                    ),
                )
            ],
        )

    content = _format_document_content(doc_info)
    conf = doc_info["confidence"]

    prompt_text = _render_template(
        ctx.tpl_summarize,
        {
            "file_path": doc_info["file_path"],
            "doc_type": conf.get("doc_type", "未知"),
            "status": conf.get("status", ""),
            "final_date": conf.get("final_date", ""),
            "section_count": doc_info["section_count"],
            "char_count": doc_info["char_count"],
            "content": content,
        },
    )

    return GetPromptResult(
        description=f"摘要: {doc_info['file_path']}",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=prompt_text),
            )
        ],
    )


# ── Resource 管理 ────────────────────────────────────

RESOURCE_TEMPLATES = [
    ResourceTemplate(
        uriTemplate="tinyrag://vault/{vault_name}",
        name="Vault Statistics",
        description="Vault 统计信息：文件数、chunk数、最近文件",
        mimeType="application/json",
    ),
    ResourceTemplate(
        uriTemplate="tinyrag://file/{file_id}",
        name="File Content",
        description="文件元信息与内容预览",
        mimeType="application/json",
    ),
    ResourceTemplate(
        uriTemplate="tinyrag://chunks/{file_id}",
        name="File Chunks",
        description="文件的所有 chunk 列表",
        mimeType="application/json",
    ),
]


async def handle_read_resource(uri: Any, ctx: AppContext) -> ReadResourceResult:
    """处理 MCP read_resource 请求"""
    await ctx.initialize()
    uri_str = str(uri)
    content = ""

    if uri_str.startswith("tinyrag://vault/"):
        content = _resource_vault_stats(uri_str.split("/")[-1], ctx)
    elif uri_str.startswith("tinyrag://file/"):
        content = _resource_file_content(uri_str.split("/")[-1], ctx)
    elif uri_str.startswith("tinyrag://chunks/"):
        content = _resource_file_chunks(uri_str.split("/")[-1], ctx)
    else:
        raise ValueError(f"Unknown resource URI: {uri_str}")

    return ReadResourceResult(
        contents=[
            TextResourceContents(uri=uri_str, mimeType="application/json", text=content)
        ]
    )


def _resource_vault_stats(vault_name: str, ctx: AppContext) -> str:
    db = ctx.db
    files_count = db.conn.execute(
        "SELECT COUNT(*) FROM files WHERE vault_name = ? AND is_deleted = 0",
        (vault_name,),
    ).fetchone()[0]
    chunks_count = db.conn.execute(
        "SELECT COUNT(*) FROM chunks c JOIN files f ON c.file_id = f.id "
        "WHERE f.vault_name = ? AND c.is_deleted = 0",
        (vault_name,),
    ).fetchone()[0]
    recent = db.conn.execute(
        "SELECT file_path, mtime, file_size FROM files "
        "WHERE vault_name = ? AND is_deleted = 0 ORDER BY updated_at DESC LIMIT 5",
        (vault_name,),
    ).fetchall()
    return json.dumps(
        {
            "vault_name": vault_name,
            "files": files_count,
            "chunks": chunks_count,
            "recent": [dict(r) for r in recent],
        },
        ensure_ascii=False,
        indent=2,
    )


def _resource_file_content(file_id: str, ctx: AppContext) -> str:
    try:
        fid = int(file_id)
    except (ValueError, TypeError):
        return json.dumps({"error": "Invalid file_id"})
    row = ctx.db.conn.execute(
        "SELECT * FROM files WHERE id = ? AND is_deleted = 0", (fid,)
    ).fetchone()
    if not row:
        return json.dumps({"error": "File not found"})
    try:
        content = Path(row["absolute_path"]).read_text(encoding="utf-8")[:10000]
    except Exception:
        content = "[Error reading file]"
    return json.dumps(
        {**dict(row), "content_preview": content},
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def _resource_file_chunks(file_id: str, ctx: AppContext) -> str:
    try:
        fid = int(file_id)
    except (ValueError, TypeError):
        return json.dumps({"error": "Invalid file_id"})
    chunks = ctx.db.conn.execute(
        "SELECT id, chunk_index, content, content_type, section_title "
        "FROM chunks WHERE file_id = ? AND is_deleted = 0 ORDER BY chunk_index",
        (fid,),
    ).fetchall()
    return json.dumps(
        {"file_id": fid, "total": len(chunks), "chunks": [dict(c) for c in chunks]},
        ensure_ascii=False,
        indent=2,
    )


# ── MCP Server ───────────────────────────────────────


class PipelineMcpServer:
    """基于 Pipeline 的 MCP Server (v2.0 - with Prompts & Resources)"""

    def __init__(self):
        self.ctx = AppContext()
        if MCP_AVAILABLE:
            self.server = Server("tinyRAG-pipeline")
            self._register()
        else:
            logger.warning("MCP package not installed, running in mock mode")

    def _register(self):
        # ── Tools ──
        @self.server.list_tools()
        async def list_tools():
            return [
                Tool(
                    name=name,
                    description=spec["description"],
                    inputSchema=spec["schema"],
                )
                for name, spec in TOOL_SCHEMAS.items()
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]):
            handler = TOOL_HANDLERS.get(name)
            if not handler:
                return [
                    TextContent(
                        type="text", text=json.dumps({"error": f"Unknown tool: {name}"})
                    )
                ]
            try:
                result = await handler(arguments, self.ctx)
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(result, ensure_ascii=False, default=str),
                    )
                ]
            except Exception as e:
                logger.error(f"Tool error: {e}", exc_info=True)
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": str(e)}, ensure_ascii=False),
                    )
                ]

        # ── Prompts ──
        @self.server.list_prompts()
        async def list_prompts():
            return PROMPT_DEFINITIONS

        @self.server.get_prompt()
        async def get_prompt(name: str, arguments: dict[str, str]):
            try:
                return await handle_get_prompt(name, arguments, self.ctx)
            except Exception as e:
                logger.error(f"Prompt error: {e}", exc_info=True)
                return GetPromptResult(
                    description=f"Error: {name}",
                    messages=[
                        PromptMessage(
                            role="user",
                            content=TextContent(
                                type="text", text=f"Prompt 渲染错误: {str(e)}"
                            ),
                        )
                    ],
                )

        # ── Resources ──
        @self.server.list_resources()
        async def list_resources():
            return []

        @self.server.list_resource_templates()
        async def list_resource_templates():
            return RESOURCE_TEMPLATES

        @self.server.read_resource()
        async def read_resource(uri):
            return await handle_read_resource(uri, self.ctx)

    async def run(self):
        if not MCP_AVAILABLE:
            logger.error("Please install mcp: pip install mcp")
            return
        logger.info("tinyRAG Pipeline MCP Server v2.0 starting...")
        try:
            async with stdio_server() as (r, w):
                await self.server.run(r, w, self.server.create_initialization_options())
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            await self.ctx.shutdown()


if __name__ == "__main__":
    asyncio.run(PipelineMcpServer().run())
