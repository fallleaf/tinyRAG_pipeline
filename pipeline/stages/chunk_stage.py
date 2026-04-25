"""pipeline/stages/chunk_stage.py - 文档分块 Stage

将待索引文件读取、分块，生成 file_chunks 列表。
"""

from __future__ import annotations

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

        # 检查是否需要强制重新生成向量
        force_reembed = getattr(ctx, "force_reembed", False)

        if force_reembed:
            # 从数据库读取现有 chunks
            logger.info("📖 从数据库读取现有 chunks")
            ctx.file_chunks = self._load_chunks_from_db(ctx)
        else:
            # 正常分块流程
            ctx.file_chunks = self._split_files(ctx, max_workers)

        logger.info("✂️ 分块完成: 准备流式处理 chunks")
        return ctx

    def _split_files(self, ctx: PipelineContext, max_workers: int):
        """正常分块流程"""
        def _split_file(file_item: dict) -> tuple[int, list, str]:
            abs_path = Path(file_item["absolute_path"])
            if not abs_path.exists():
                return file_item["id"], [], file_item["file_path"]
            try:
                content = abs_path.read_text(encoding="utf-8")
                chunks = ctx.splitter.split(content, file_item.get("mtime"))
                return file_item["id"], chunks, file_item["file_path"]
            except Exception as e:
                logger.error(f"❌ 读取/分块失败: {abs_path} - {e}")
                return file_item["id"], [], file_item["file_path"]

        def chunk_generator():
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for f_id, chunks, f_path in executor.map(
                    lambda f: _split_file(f), ctx.files_to_index
                ):
                    if chunks:
                        for c in chunks:
                            yield (f_id, c, f_path)

        return chunk_generator()

    def _load_chunks_from_db(self, ctx: PipelineContext):
        """从数据库加载现有 chunks"""
        db = ctx.db

        def chunk_generator():
            # 获取所有 chunks
            cursor = db.conn.execute("""
                SELECT c.id, c.file_id, c.chunk_index, c.content, c.content_type, c.section_title,
                       c.section_path, c.start_pos, c.end_pos, c.metadata,
                       c.confidence_json, f.file_path
                FROM chunks c
                JOIN files f ON c.file_id = f.id
                WHERE f.is_deleted = 0
                ORDER BY c.file_id, c.chunk_index
            """)

            for row in cursor.fetchall():
                from chunker.markdown_splitter import Chunk, ChunkType

                # 解析 content_type
                try:
                    content_type = ChunkType(row["content_type"])
                except ValueError:
                    content_type = ChunkType.MARKDOWN

                chunk = Chunk(
                    content=row["content"],
                    content_type=content_type,
                    section_title=row["section_title"],
                    section_path=row["section_path"],
                    start_pos=row["start_pos"],
                    end_pos=row["end_pos"],
                    metadata=row["metadata"] or {},
                    confidence_metadata=row["confidence_json"] or {},
                )
                # 添加 chunk_index 和 id 属性
                chunk.chunk_index = row["chunk_index"]
                chunk.id = row["id"]
                yield (row["file_id"], chunk, row["file_path"])

        return chunk_generator()
