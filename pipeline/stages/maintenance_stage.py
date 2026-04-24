"""pipeline/stages/maintenance_stage.py - 数据库运维 Stage

DBInitStage: 仅初始化数据库连接（不扫描）
CleanupStage: 清理软删除记录
VacuumStage: 执行 VACUUM 回收空间
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from pipeline.context import PipelineContext
from pipeline.stage import Stage
from helpers.logger import logger


class DBInitStage(Stage):
    """仅初始化数据库连接（用于运维/状态等不需要扫描的场景）"""

    name = "db_init"
    description = "初始化数据库连接"

    def should_run(self, ctx: PipelineContext) -> bool:
        return ctx.db is None

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.config is None:
            raise RuntimeError("配置未加载，请先运行 ConfigLoadStage")

        from storage.database import DatabaseManager
        db = DatabaseManager(ctx.config.db_path, vec_dimension=ctx.config.embedding_model.dimensions)
        ctx.db = db
        logger.info(f"✅ 数据库连接初始化: {ctx.config.db_path}")
        return ctx


class CleanupStage(Stage):
    """清理软删除记录"""

    name = "cleanup"
    description = "清理软删除的 chunks 和 files 记录"

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.db is None:
            raise RuntimeError("数据库未初始化")

        import vacuum as vac_mod
        stats = vac_mod.check_vacuum_needed(ctx.db, ctx.config)
        ctx.maintenance_stats = stats

        if ctx.dry_run:
            logger.info(f"🔍 [Dry-Run] 预计清理: {stats['chunks_deleted']} chunks + {stats['files_deleted']} files")
        else:
            vac_mod.clean_deleted_records(ctx.db, dry_run=False)
            logger.info(f"🧹 清理完成: {stats['chunks_deleted']} chunks + {stats['files_deleted']} files")

        return ctx


class VacuumStage(Stage):
    """执行 VACUUM 回收空间"""

    name = "vacuum"
    description = "VACUUM 回收 SQLite 空间"

    def should_run(self, ctx: PipelineContext) -> bool:
        return not ctx.dry_run

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.db is None:
            raise RuntimeError("数据库未初始化")

        import vacuum as vac_mod

        old_size = os.path.getsize(ctx.config.db_path) / (1024 * 1024)
        vac_mod.execute_vacuum(ctx.db, dry_run=False)
        new_size = os.path.getsize(ctx.config.db_path) / (1024 * 1024)
        saved = old_size - new_size

        logger.info(f"🗜️ VACUUM 完成: {old_size:.2f}MB → {new_size:.2f}MB (节省 {saved:.2f}MB)")
        ctx.maintenance_stats["vacuum_saved_mb"] = round(saved, 2)
        return ctx
