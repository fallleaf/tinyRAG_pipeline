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
        check_interval = ctx.memory_check_interval

        # 使用生成器，避免累积所有结果
        def embedding_generator():
            batch = []
            processed = 0
            for item in ctx.file_chunks:
                batch.append(item)
                if len(batch) >= stream_batch_size:
                    yield from self._process_batch(batch, embed_engine, ctx)
                    processed += len(batch)
                    batch = []

                    # 定期检查内存使用
                    if processed % check_interval == 0:
                        ctx.check_memory()

            # 处理剩余的批次
            if batch:
                yield from self._process_batch(batch, embed_engine, ctx)

        ctx.chunk_embeddings = embedding_generator()
        logger.info(f"🔢 向量化完成: 准备流式处理向量")
        return ctx

    def _process_batch(self, batch: list, embed_engine, ctx: PipelineContext):
        """处理单个批次，返回生成器"""
        texts = [c[1].content for c in batch]

        try:
            embeddings = embed_engine.embed(texts)
        except RuntimeError as e:
            ctx.add_error(f"向量化批次失败: {e}")
            raise

        for (file_id, chunk, f_path), emb in zip(batch, embeddings):
            yield (file_id, chunk, f_path, emb)
