"""pipeline/stages/search_stage.py - 混合检索 Stage

执行语义+关键词混合检索，返回排序后的结果。
"""
from __future__ import annotations

from pipeline.context import PipelineContext
from pipeline.stage import Stage
from helpers.logger import logger


class SearchStage(Stage):
    """执行混合检索"""

    name = "search"
    description = "执行语义+关键词混合检索"

    def should_run(self, ctx: PipelineContext) -> bool:
        return bool(ctx.query.strip()) if ctx.query else False

    def process(self, ctx: PipelineContext) -> PipelineContext:
        from embedder.embed_engine import EmbeddingEngine
        from retriever.hybrid_engine import HybridEngine
        from storage.database import DatabaseManager

        config = ctx.config
        db = ctx.db
        if db is None:
            db = DatabaseManager(config.db_path, vec_dimension=config.embedding_model.dimensions)
            ctx.db = db

        embed_engine = ctx.embed_engine
        if embed_engine is None:
            embed_engine = EmbeddingEngine(
                model_name=config.embedding_model.name,
                cache_dir=config.embedding_model.cache_dir,
                batch_size=config.embedding_model.batch_size,
                unload_after_seconds=config.embedding_model.unload_after_seconds,
            )
            ctx.embed_engine = embed_engine

        retriever = ctx.retriever
        if retriever is None:
            retriever = HybridEngine(config=config, db=db, embed_engine=embed_engine)
            ctx.retriever = retriever

        # 处理 mode → alpha/beta 映射
        alpha = ctx.alpha
        beta = ctx.beta
        if ctx.search_mode == "keyword":
            alpha, beta = 0.0, 1.0
        elif ctx.search_mode == "semantic":
            alpha, beta = 1.0, 0.0
        elif alpha is None and beta is None:
            alpha = config.retrieval.get("alpha", 0.7)
            beta = config.retrieval.get("beta", 0.3)

        # vault 过滤
        vault_filter = ctx.vault_filter
        if vault_filter is None:
            vault_filter = [v.name for v in config.vaults if v.enabled]

        results = retriever.search(
            query=ctx.query,
            limit=ctx.top_k,
            vault_filter=vault_filter,
            alpha=alpha,
            beta=beta,
        )

        ctx.search_results = results
        logger.info(f"🔍 检索完成: {len(results)} 条结果 (query='{ctx.query[:30]}...')")
        return ctx
