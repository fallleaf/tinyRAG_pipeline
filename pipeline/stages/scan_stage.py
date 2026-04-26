"""pipeline/stages/scan_stage.py - 文件扫描与差异检测 Stage

ScanStage: 扫描 vault 目录，生成 ScanReport
DiffStage: 根据 ScanReport 确定待索引文件列表
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from pipeline.context import PipelineContext
from pipeline.stage import Stage
from helpers.logger import logger


class ScanStage(Stage):
    """扫描 vault 目录，检测文件变更"""

    name = "scan"
    description = "扫描 vault 目录，生成变更报告"

    def process(self, ctx: PipelineContext) -> PipelineContext:
        from config import get_merged_exclude
        from scanner.scan_engine import DEFAULT_SKIP_DIRS, Scanner
        from storage.database import DatabaseManager

        config = ctx.config
        # 复用已存在的数据库连接，避免在 MCP 场景下重复创建导致 disk I/O error
        if ctx.db is None:
            db = DatabaseManager(
                config.db_path, vec_dimension=config.embedding_model.dimensions
            )
            ctx.db = db
        else:
            db = ctx.db

        global_skip_dirs = DEFAULT_SKIP_DIRS | frozenset(config.exclude.dirs)
        scanner = Scanner(
            db, skip_dirs=global_skip_dirs, global_patterns=config.exclude.patterns
        )
        ctx.scanner = scanner

        if ctx.force_rebuild:
            logger.info("🔄 强制重建模式：清空所有索引数据")
            db.conn.execute("DELETE FROM fts5_index")
            cursor = db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='vectors'"
            )
            if cursor.fetchone():
                db.conn.execute("DELETE FROM vectors")
            db.conn.execute("DELETE FROM chunks")
            db.conn.execute("DELETE FROM files")
            db.conn.commit()

        report = scanner.scan_vaults(ctx.vault_configs, ctx.vault_excludes)
        ctx.scan_report = report

        logger.info(f"📊 扫描结果: {report.summary()}")
        return ctx


class DiffStage(Stage):
    """根据扫描报告确定待索引文件列表"""

    name = "diff"
    description = "计算差异，确定待索引文件"

    def should_run(self, ctx: PipelineContext) -> bool:
        return ctx.scan_report is not None

    def process(self, ctx: PipelineContext) -> PipelineContext:
        report = ctx.scan_report
        db = ctx.db

        if ctx.force_rebuild:
            # 强制模式：扫描后的 new_files 全部入库，然后索引所有文件
            for meta in report.new_files:
                db.upsert_file(meta.to_dict())
            db.conn.commit()
            ctx.files_to_index = [
                dict(row)
                for row in db.conn.execute(
                    "SELECT id, absolute_path, file_path, mtime FROM files WHERE is_deleted = 0"
                ).fetchall()
            ]
        elif getattr(ctx, "force_reembed", False):
            # 强制重新生成向量：获取所有 chunks
            logger.info("🔄 强制重新生成向量模式")
            ctx.files_to_index = [
                dict(row)
                for row in db.conn.execute(
                    "SELECT DISTINCT f.id, f.absolute_path, f.file_path, f.mtime FROM files f JOIN chunks c ON f.id = c.file_id WHERE f.is_deleted = 0"
                ).fetchall()
            ]
        else:
            # 增量模式：处理报告，获取变更文件
            ctx.scanner.process_report(report)
            ctx.changed_paths = [
                f.absolute_path for f in report.new_files + report.modified_files
            ]
            ctx.changed_paths.extend([f.new_absolute_path for f in report.moved_files])

            ctx.files_to_index = []
            if ctx.changed_paths:
                placeholders = ",".join(["?"] * len(ctx.changed_paths))
                cursor = db.conn.execute(
                    f"SELECT id, absolute_path, file_path, mtime FROM files WHERE absolute_path IN ({placeholders})",
                    ctx.changed_paths,
                )
                ctx.files_to_index = [dict(row) for row in cursor.fetchall()]

            # 自愈：无变更时检查缺失 chunks
            if not ctx.files_to_index:
                cursor = db.conn.execute("""
                    SELECT f.id, f.vault_name, f.absolute_path, f.file_path, f.mtime
                    FROM files f
                    WHERE f.is_deleted = 0
                      AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.file_id = f.id)
                    LIMIT 1000
                """)
                missing = [dict(row) for row in cursor.fetchall()]
                if missing:
                    logger.info(
                        f"🔧 发现 {len(missing)} 个已注册但无索引的文件，准备补充"
                    )
                    ctx.files_to_index.extend(missing)

        if not ctx.files_to_index:
            logger.info("✨ 索引已是最新，无需更新")
        else:
            logger.info(f"📦 待索引文件: {len(ctx.files_to_index)} 个")

        return ctx
