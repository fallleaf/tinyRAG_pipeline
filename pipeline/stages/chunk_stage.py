"""pipeline/stages/chunk_stage.py - 文档分块 Stage

将待索引文件读取、分块，生成 file_chunks 列表。
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipeline.context import PipelineContext
from pipeline.stage import Stage
from helpers.logger import logger


class ChunkStage(Stage):
    """将文档分块为 Chunk 对象"""

    name = "chunk"
    description = "读取文件并分块"

    def should_run(self, ctx: PipelineContext) -> bool:
        return bool(ctx.files_to_index)

    def process(self, ctx: PipelineContext) -> PipelineContext:
        from chunker.markdown_splitter import MarkdownSplitter

        splitter = MarkdownSplitter(ctx.config)
        ctx.splitter = splitter
        max_workers = getattr(ctx.config, "max_concurrent_files", 4)

        def _split_file(file_item: dict) -> tuple[int, list, str]:
            abs_path = Path(file_item["absolute_path"])
            if not abs_path.exists():
                return file_item["id"], [], file_item["file_path"]
            try:
                content = abs_path.read_text(encoding="utf-8")
                chunks = splitter.split(content, file_item.get("mtime"))
                return file_item["id"], chunks, file_item["file_path"]
            except Exception as e:
                logger.error(f"❌ 读取/分块失败: {abs_path} - {e}")
                return file_item["id"], [], file_item["file_path"]

        # 使用生成器，避免累积所有 chunk
        def chunk_generator():
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for f_id, chunks, f_path in executor.map(lambda f: _split_file(f), ctx.files_to_index):
                    if chunks:
                        for c in chunks:
                            yield (f_id, c, f_path)

        ctx.file_chunks = chunk_generator()
        logger.info(f"✂️ 分块完成: 准备流式处理 chunks")
        return ctx
