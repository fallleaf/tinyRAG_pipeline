"""pipeline/stages/index_stage.py - 索引入库 Stage

将 chunk + 向量 + FTS5 内容写入数据库。
**核心消除重复**：这是原 tinyRAG 中 build_index.py / server.py / rag_cli.py
三处重复的 chunk→embed→insert 逻辑的统一实现。
"""

from __future__ import annotations

import array
import json

from pipeline.context import PipelineContext
from pipeline.stage import Stage
from helpers.json_serialize import json_serialize
from helpers.fts_content import prepare_fts_content
from helpers.logger import logger


class IndexStage(Stage):
    """将 chunk + 向量 + FTS5 写入数据库"""

    name = "index"
    description = "将分块、向量、FTS5 索引写入数据库"

    def should_run(self, ctx: PipelineContext) -> bool:
        return bool(ctx.chunk_embeddings)

    def process(self, ctx: PipelineContext) -> PipelineContext:
        db = ctx.db
        if db is None:
            raise RuntimeError("数据库未初始化，请先运行 ScanStage")

        # 获取当前最大 chunk_index，避免冲突
        row = db.conn.execute("SELECT COALESCE(MAX(chunk_index), -1) FROM chunks").fetchone()
        global_chunk_idx = row[0] + 1 if row else 0
        ctx.global_chunk_idx = global_chunk_idx

        # 按批提交（与 embedding 批大小对齐）
        batch_size = ctx.config.embedding_model.batch_size
        processed = 0
        check_interval = ctx.memory_check_interval

        # 流式处理，不累积所有结果
        batch = []
        for item in ctx.chunk_embeddings:
            batch.append(item)
            if len(batch) >= batch_size:
                processed += self._process_batch(batch, db, global_chunk_idx, processed, ctx)
                batch = []
                # 释放内存
                del batch
                batch = []

                # 定期检查内存使用
                if processed % check_interval == 0:
                    ctx.check_memory()

        # 处理剩余的批次
        if batch:
            processed += self._process_batch(batch, db, global_chunk_idx, processed, ctx)

        # 最终内存检查
        ctx.check_memory(force_gc=True)

        ctx.total_indexed = processed
        logger.info(f"📥 索引入库完成: {processed} 个 chunks")
        return ctx

    def _process_batch(self, batch: list, db, global_chunk_idx: int, offset: int, ctx: PipelineContext) -> int:
        """处理单个批次，返回处理的数量"""
        try:
            db.conn.execute("PRAGMA synchronous = OFF;")
            for idx, (file_id, chunk, f_path, emb) in enumerate(batch):
                chunk_idx = global_chunk_idx + offset + idx
                metadata_json = json.dumps(chunk.metadata or {}, ensure_ascii=False, default=json_serialize)
                confidence_json = json.dumps(
                    chunk.confidence_metadata or {}, ensure_ascii=False, default=json_serialize
                )

                cursor = db.conn.execute(
                    """INSERT INTO chunks
                    (file_id, chunk_index, content, content_type, section_title, section_path,
                     start_pos, end_pos, confidence_final_weight, metadata, confidence_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        file_id,
                        chunk_idx,
                        chunk.content,
                        chunk.content_type.value,
                        chunk.section_title,
                        chunk.section_path,
                        chunk.start_pos,
                        chunk.end_pos,
                        1.0,
                        metadata_json,
                        confidence_json,
                    ),
                )
                new_chunk_id = cursor.lastrowid

                # 向量入库
                if db.vec_support:
                    db.conn.execute(
                        "INSERT INTO vectors (chunk_id, embedding) VALUES (?, ?)",
                        (new_chunk_id, array.array("f", emb).tobytes()),
                    )

                # FTS5 入库
                db.conn.execute(
                    "INSERT INTO fts5_index (rowid, content) VALUES (?, ?)",
                    (new_chunk_id, prepare_fts_content(chunk, f_path)),
                )

            db.conn.commit()

            if not ctx.quiet:
                logger.info(f"   💾 已入库 {offset + len(batch)} chunks")

        except Exception as e:
            db.conn.rollback()
            ctx.add_error(f"批次入库失败: {e}")
            raise
        finally:
            db.conn.execute("PRAGMA synchronous = NORMAL;")

        return len(batch)
