"""pipeline/stages/embed_stage.py - 向量化 Stage

将 chunk 文本批量向量化，生成 chunk_embeddings 列表。
"""
from __future__ import annotations

from pipeline.context import PipelineContext
from pipeline.stage import Stage
from helpers.logger import logger


class EmbedStage(Stage):
    """批量向量化 chunk 文本"""

    name = "embed"
    description = "将 chunk 文本转换为向量"

    def should_run(self, ctx: PipelineContext) -> bool:
        return bool(ctx.file_chunks)

    def process(self, ctx: PipelineContext) -> PipelineContext:
        from embedder.embed_engine import EmbeddingEngine

        config = ctx.config
        embed_engine = EmbeddingEngine(
            model_name=config.embedding_model.name,
            cache_dir=config.embedding_model.cache_dir,
            batch_size=config.embedding_model.batch_size,
            unload_after_seconds=config.embedding_model.unload_after_seconds,
        )
        ctx.embed_engine = embed_engine

        # 流式批处理：按 stream_batch_size 分批
        stream_batch_size = getattr(config, "stream_batch_size", 100)
        all_results: list[tuple[int, any, str, list[float]]] = []
        total_chunks = len(ctx.file_chunks)

        for batch_start in range(0, total_chunks, stream_batch_size):
            batch = ctx.file_chunks[batch_start:batch_start + stream_batch_size]
            texts = [c[1].content for c in batch]

            try:
                embeddings = embed_engine.embed(texts)
            except RuntimeError as e:
                ctx.add_error(f"向量化批次 {batch_start} 失败: {e}")
                raise

            for (file_id, chunk, f_path), emb in zip(batch, embeddings):
                all_results.append((file_id, chunk, f_path, emb))

            if not ctx.quiet:
                logger.info(f"   📊 已向量化 {min(batch_start + stream_batch_size, total_chunks)}/{total_chunks}")

        ctx.chunk_embeddings = all_results
        logger.info(f"🔢 向量化完成: {len(all_results)} 个向量")
        return ctx
