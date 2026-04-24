#!/usr/bin/env python3
"""cli/main.py - tinyRAG Pipeline CLI 入口

使用 typer 构建现代 CLI，每个命令组合不同的 Pipeline。

命令:
  tinyrag build         全量重建索引
  tinyrag scan          增量扫描更新
  tinyrag search        混合检索
  tinyrag status        系统状态
  tinyrag config        配置管理
  tinyrag maintenance   数据库运维
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

# 确保项目路径在 sys.path
_project_root = Path(__file__).parent.parent
_tinyrag_root = _project_root
if str(_tinyrag_root) not in sys.path:
    sys.path.insert(0, str(_tinyrag_root))

from pipeline.context import PipelineContext
from pipeline.pipeline import Pipeline
from pipeline.events import EventBus
from pipeline.stages import (
    ConfigLoadStage, ScanStage, DiffStage, ChunkStage,
    EmbedStage, IndexStage, SearchStage,
    DBInitStage, CleanupStage, VacuumStage,
)
from helpers.logger import logger, setup_logger

app = typer.Typer(
    name="tinyrag",
    help="tinyRAG Pipeline - CLI + Pipeline 架构的轻量级 RAG 系统",
    no_args_is_help=True,
)


# ── 辅助函数 ──────────────────────────────────────────

def _make_ctx(**kwargs) -> PipelineContext:
    """创建 PipelineContext 并设置参数"""
    return PipelineContext(**kwargs)


def _run_pipeline(name: str, stages: list, ctx: PipelineContext, close_db: bool = True) -> PipelineContext:
    """创建并执行 Pipeline"""
    event_bus = EventBus()
    pipeline = Pipeline(name, stages, event_bus=event_bus)
    result, ctx = pipeline.run(ctx)

    if not result.success:
        typer.secho(f"❌ Pipeline [{name}] 执行失败", fg=typer.colors.RED)
        for err in ctx.errors:
            typer.secho(f"   {err}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # 关闭数据库
    if close_db and ctx.db:
        ctx.db.close()

    return ctx


def _load_config_via_pipeline(config_path: str):
    """通过 ConfigLoadStage + DBInitStage 加载配置和数据库"""
    ctx = _make_ctx(config_path=config_path)
    result_ctx = _run_pipeline("_config_load", [ConfigLoadStage(), DBInitStage()], ctx, close_db=False)
    return result_ctx.config, result_ctx.db


# ── 命令定义 ──────────────────────────────────────────

@app.command()
def build(
    force: bool = typer.Option(True, "--force/--no-force", help="强制清空后重建"),
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="配置文件路径"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="静默模式，禁用进度条"),
):
    """全量重建索引"""
    ctx = _make_ctx(config_path=config_path, force_rebuild=force, quiet=quiet)

    stages = [
        ConfigLoadStage(),
        ScanStage(),
        DiffStage(),
        ChunkStage(),
        EmbedStage(),
        IndexStage(),
    ]

    result_ctx = _run_pipeline("build", stages, ctx)

    if result_ctx.total_indexed > 0:
        typer.secho(
            f"🎉 构建完成！处理 {result_ctx.total_files_with_chunks} 个文件，"
            f"{result_ctx.total_indexed} 个 chunks，耗时 {result_ctx.elapsed:.2f}s",
            fg=typer.colors.GREEN,
        )
    else:
        typer.secho("⚠️ 无文件需要索引", fg=typer.colors.YELLOW)


@app.command()
def scan(
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="配置文件路径"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="静默模式"),
):
    """增量扫描与索引更新"""
    ctx = _make_ctx(config_path=config_path, force_rebuild=False, quiet=quiet)

    stages = [
        ConfigLoadStage(),
        ScanStage(),
        DiffStage(),
        ChunkStage(),
        EmbedStage(),
        IndexStage(),
    ]

    result_ctx = _run_pipeline("scan", stages, ctx)

    if result_ctx.total_indexed > 0:
        typer.secho(
            f"✅ 增量更新完成：{result_ctx.total_indexed} 个 chunks",
            fg=typer.colors.GREEN,
        )
    else:
        typer.secho("✨ 索引已是最新", fg=typer.colors.GREEN)


@app.command()
def search(
    query: str = typer.Argument(..., help="查询文本"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="返回结果数量"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="检索模式: hybrid/keyword/semantic"),
    alpha: Optional[float] = typer.Option(None, "--alpha", "-a", help="语义权重"),
    beta: Optional[float] = typer.Option(None, "--beta", "-b", help="关键词权重"),
    vaults: Optional[list[str]] = typer.Option(None, "--vault", "-v", help="指定仓库"),
    output: str = typer.Option("console", "--output", "-o", help="输出格式: console/json/csv"),
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="配置文件路径"),
):
    """执行混合检索"""
    ctx = _make_ctx(
        config_path=config_path,
        query=query,
        top_k=top_k,
        search_mode=mode,
        alpha=alpha,
        beta=beta,
        vault_filter=vaults if vaults else None,
    )

    stages = [
        ConfigLoadStage(),
        SearchStage(),
    ]

    result_ctx = _run_pipeline("search", stages, ctx)
    results = result_ctx.search_results

    if output == "json":
        output_data = [
            {"rank": i + 1, **r.__dict__}
            for i, r in enumerate(results)
        ]
        typer.echo(json.dumps(output_data, indent=2, ensure_ascii=False, default=str))
    elif output == "csv":
        import csv
        from io import StringIO
        buf = StringIO()
        writer = csv.DictWriter(buf, fieldnames=[
            "rank", "file_path", "absolute_path", "section", "vault_name",
            "chunk_type", "content", "final_score", "confidence_score",
        ])
        writer.writeheader()
        for i, r in enumerate(results, 1):
            writer.writerow({
                "rank": i, "file_path": r.file_path,
                "absolute_path": r.absolute_path, "section": r.section,
                "vault_name": r.vault_name, "chunk_type": r.chunk_type,
                "content": r.content[:200].replace("\n", " "),
                "final_score": r.final_score,
                "confidence_score": r.confidence_score,
            })
        typer.echo(buf.getvalue())
    else:
        typer.secho(f"\n📊 检索结果 ({len(results)} 条, 耗时 {result_ctx.elapsed:.2f}s):\n", fg=typer.colors.CYAN)
        if not results:
            typer.secho("   未找到相关结果", fg=typer.colors.YELLOW)
            return
        for i, r in enumerate(results, 1):
            preview = r.content[:200] + "..." if len(r.content) > 200 else r.content
            scores = f"最终={r.final_score:.3f} | 语义={r.semantic_score:.3f} | 关键词={r.keyword_score:.3f} | 置信度={r.confidence_score:.2f}"
            typer.secho(f"{i}. [{scores}]", fg=typer.colors.WHITE)
            typer.echo(f"   来源: {r.absolute_path}")
            typer.echo(f"   类型: {r.vault_name} / {r.chunk_type} | 章节: {r.section}")
            typer.echo(f"   内容: {preview}\n")


@app.command()
def status(
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="配置文件路径"),
):
    """查看系统状态"""
    # 通过 Pipeline 加载配置（自动处理 sys.path）
    config, db = _load_config_via_pipeline(config_path)

    typer.secho("\n📊 tinyRAG Pipeline 系统状态\n" + "=" * 50, fg=typer.colors.CYAN)

    try:
        db_path = Path(config.db_path).resolve()
        db_size = db_path.stat().st_size / 1024 / 1024 if db_path.exists() else 0

        typer.echo(f"🗄️ 数据库: {db_path} ({db_size:.2f} MB)")
        if db and db.conn:
            files_active = db.conn.execute("SELECT COUNT(*) FROM files WHERE is_deleted=0").fetchone()[0]
            chunks_active = db.conn.execute("SELECT COUNT(*) FROM chunks WHERE is_deleted=0").fetchone()[0]
            typer.echo(f"   活跃: {files_active} files | {chunks_active} chunks")
            typer.echo(f"   向量引擎: {'✅ sqlite-vec' if db.vec_support else '⚠️ FTS5-only'}")

            # 按 vault 统计
            vaults = db.conn.execute(
                "SELECT vault_name, COUNT(*) as cnt FROM files WHERE is_deleted=0 GROUP BY vault_name"
            ).fetchall()
            if vaults:
                typer.echo("   仓库:")
                for v in vaults:
                    v_chunks = db.conn.execute(
                        "SELECT COUNT(*) FROM chunks c JOIN files f ON c.file_id = f.id "
                        "WHERE f.vault_name = ? AND c.is_deleted = 0", (v["vault_name"],)
                    ).fetchone()[0]
                    typer.echo(f"     {v['vault_name']}: {v['cnt']} files / {v_chunks} chunks")
        else:
            typer.secho("   ⚠️ 数据库未连接", fg=typer.colors.YELLOW)

        typer.echo(f"\n🤖 模型: {config.embedding_model.name} (dim={config.embedding_model.dimensions})")
        typer.echo(f"🔍 检索: alpha={config.retrieval.get('alpha', 0.7)}, beta={config.retrieval.get('beta', 0.3)}")
        typer.echo(f"✂️ 分块: max_tokens={config.chunking.max_tokens}, overlap={config.chunking.overlap}")
    finally:
        if db:
            db.close()


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="显示原始 YAML"),
    parsed: bool = typer.Option(False, "--parsed", help="显示解析后 JSON"),
    validate: bool = typer.Option(False, "--validate", help="验证配置"),
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="配置文件路径"),
):
    """管理配置"""
    if not any([show, parsed, validate]):
        show = True

    if show:
        p = Path(config_path)
        if p.exists():
            typer.echo(p.read_text(encoding="utf-8"))
        else:
            typer.secho(f"❌ 配置文件不存在: {p}", fg=typer.colors.RED)
            raise typer.Exit(1)

    if parsed or validate:
        # 通过 Pipeline 加载配置
        cfg, _ = _load_config_via_pipeline(config_path)

        if parsed:
            from pydantic import TypeAdapter
            adapter = TypeAdapter(type(cfg))
            data = adapter.dump_python(cfg, mode="json")
            typer.echo(json.dumps(data, indent=2, ensure_ascii=False))

        if validate:
            typer.secho("✅ 配置验证通过!", fg=typer.colors.GREEN)
            for v in cfg.vaults:
                v_path = Path(v.path).expanduser()
                icon = "✅" if v_path.exists() else "⚠️"
                typer.echo(f"   {icon} {v.name}: {v.path}")


@app.command()
def maintenance(
    dry_run: bool = typer.Option(False, "--dry-run", help="仅检查不执行"),
    clean_only: bool = typer.Option(False, "--clean-only", help="仅清理不 VACUUM"),
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="配置文件路径"),
):
    """数据库运维（清理软删除 + VACUUM）"""
    ctx = _make_ctx(config_path=config_path, dry_run=dry_run)

    stages = [
        ConfigLoadStage(),
        DBInitStage(),   # 运维只需初始化 DB，无需扫描
        CleanupStage(),
    ]
    if not clean_only:
        stages.append(VacuumStage())

    result_ctx = _run_pipeline("maintenance", stages, ctx)

    stats = result_ctx.maintenance_stats
    typer.secho(f"📊 软删除比例: files={stats.get('files_ratio', 0):.1f}%, chunks={stats.get('chunks_ratio', 0):.1f}%", fg=typer.colors.CYAN)
    typer.secho(f"   数据库大小: {stats.get('file_size_mb', 0):.2f} MB", fg=typer.colors.CYAN)

    if not dry_run:
        typer.secho("✅ 运维完成", fg=typer.colors.GREEN)
    else:
        typer.secho("🔍 Dry-Run 模式，未执行变更", fg=typer.colors.YELLOW)


# ── Pipeline 结构展示 ──────────────────────────────────

@app.command()
def describe(
    pipeline_name: str = typer.Argument("build", help="Pipeline 名称: build/scan/search/maintenance"),
):
    """展示 Pipeline 的 Stage 组成"""
    pipeline_map = {
        "build": [ConfigLoadStage, ScanStage, DiffStage, ChunkStage, EmbedStage, IndexStage],
        "scan": [ConfigLoadStage, ScanStage, DiffStage, ChunkStage, EmbedStage, IndexStage],
        "search": [ConfigLoadStage, SearchStage],
        "maintenance": [ConfigLoadStage, DBInitStage, CleanupStage, VacuumStage],
    }
    if pipeline_name not in pipeline_map:
        typer.secho(f"❌ 未知 Pipeline: {pipeline_name}", fg=typer.colors.RED)
        raise typer.Exit(1)

    stages = [cls() for cls in pipeline_map[pipeline_name]]
    p = Pipeline(pipeline_name, stages)
    typer.secho(p.describe(), fg=typer.colors.CYAN)


if __name__ == "__main__":
    app()
