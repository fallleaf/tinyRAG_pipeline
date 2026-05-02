#!/usr/bin/env python3
"""cli/main.py - tinyRAG_pipeline CLI 入口

使用 Typer 构建命令行界面，支持索引构建、检索、运维等命令。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import typer
from typing import Annotated

from pipeline.pipeline import Pipeline
from pipeline.context import PipelineContext
from pipeline.stages.config_stage import ConfigLoadStage
from pipeline.stages.scan_stage import ScanStage, DiffStage
from pipeline.stages.chunk_stage import ChunkStage
from pipeline.stages.embed_stage import EmbedStage
from pipeline.stages.index_stage import IndexStage
from pipeline.stages.maintenance_stage import DBInitStage, CleanupStage, VacuumStage
from pipeline.stages.search_stage import SearchStage

app = typer.Typer(help="tinyRAG_pipeline - 轻量级 RAG 系统")


def version_callback(value: bool):
    if value:
        typer.echo("tinyRAG_pipeline 0.1.0")
        raise typer.Exit()


@app.callback()
def main(version: bool = typer.Option(None, "--version", callback=version_callback, help="显示版本信息")):
    pass


@app.command("config")
def show_config(
    config_path: Annotated[
        str,
        typer.Option(
            "-c",
            "--config",
            help="配置文件路径",
        ),
    ] = "./config.yaml",
):
    """显示当前配置"""
    from config import load_config

    config = load_config(config_path)

    typer.echo(f"📄 配置文件：{config_path}")
    typer.echo(f"📦 Vault 数量：{len(config.vaults)}")
    for v in config.vaults:
        typer.echo(f"   - {v.name}: {v.path} (启用：{v.enabled})")

    typer.echo(f"\n🤖 模型：{config.embedding_model.name} (dim={config.embedding_model.dimensions})")
    typer.echo(f"🔍 检索：alpha={config.retrieval.alpha}, beta={config.retrieval.beta}")
    chunking = config.chunking
    typer.echo(f"✂️ 分块：max_tokens={chunking.get('max_tokens', 512)}, overlap={chunking.get('overlap', 100)}")


@app.command("rebuild")
def rebuild_index(
    config_path: Annotated[
        str,
        typer.Option(
            "-c",
            "--config",
            help="配置文件路径",
        ),
    ] = "./config.yaml",
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="强制重建（删除现有向量表）",
        ),
    ] = False,
    reembed: Annotated[
        bool,
        typer.Option(
            "--reembed",
            help="强制重新生成所有向量",
        ),
    ] = False,
):
    """重建知识索引（删除现有向量表并重新构建）"""
    if not force:
        typer.echo("⚠️ 重建索引会删除现有向量表，请使用 --force 确认")
        raise typer.Exit(1)

    # 注意：不再手动删除向量表，统一由 ScanStage 在 force_rebuild=True 时处理
    # 这确保了 CLI 和 MCP 重建行为的一致性，避免数据库连接不一致问题
    typer.echo("🗑️  准备清空并重建索引...")

    # 重新构建
    ctx = PipelineContext(config_path=config_path)

    # 强制重建模式：清空所有索引数据
    ctx.force_rebuild = True

    # 如果需要重新生成所有向量，设置标志
    if reembed:
        ctx.force_reembed = True

    pipeline = Pipeline(
        name="rebuild",
        stages=[
            ConfigLoadStage(),
            ScanStage(),
            DiffStage(),
            ChunkStage(),
            EmbedStage(),
            IndexStage(),
        ],
    )

    typer.echo("🚀 开始重新构建索引...")
    pipeline.run(ctx)


@app.command("build")
def build_index(
    config_path: Annotated[
        str,
        typer.Option(
            "-c",
            "--config",
            help="配置文件路径",
        ),
    ] = "./config.yaml",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="仅检查配置，不执行构建",
        ),
    ] = False,
):
    """构建知识索引"""
    ctx = PipelineContext(config_path=config_path)

    pipeline = Pipeline(
        name="build",
        stages=[
            ConfigLoadStage(),
            ScanStage(),
            DiffStage(),
            ChunkStage(),
            EmbedStage(),
            IndexStage(),
        ],
    )

    if dry_run:
        typer.echo("🔍 Dry-Run 模式，仅检查配置")
        pipeline.run(ctx)
    else:
        typer.echo("🚀 开始构建索引...")
        pipeline.run(ctx)


@app.command("search")
def search(
    query: Annotated[
        str,
        typer.Argument(help="检索查询"),
    ],
    config_path: Annotated[
        str,
        typer.Option(
            "-c",
            "--config",
            help="配置文件路径",
        ),
    ] = "./config.yaml",
    top_k: Annotated[
        int,
        typer.Option(
            "-k",
            "--top-k",
            help="返回结果数量",
        ),
    ] = 10,
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="检索模式：semantic/keyword/hybrid",
        ),
    ] = "hybrid",
    alpha: Annotated[
        float | None,
        typer.Option(
            "--alpha",
            help="语义检索权重 (0.0-1.0)",
        ),
    ] = None,
    beta: Annotated[
        float | None,
        typer.Option(
            "--beta",
            help="关键词检索权重 (0.0-1.0)",
        ),
    ] = None,
):
    """执行检索查询"""
    ctx = PipelineContext(
        config_path=config_path,
        query=query,
        top_k=top_k,
        search_mode=mode,
        alpha=alpha,
        beta=beta,
    )

    pipeline = Pipeline(
        name="search",
        stages=[
            ConfigLoadStage(),
            SearchStage(),
        ],
    )

    pipeline.run(ctx)

    if ctx.search_results:
        typer.echo(f"\n📊 找到 {len(ctx.search_results)} 条结果:")
        for i, result in enumerate(ctx.search_results, 1):
            typer.echo(f"\n{i}. {result.file_path}")
            typer.echo(f"   内容：{result.content[:200]}...")
            typer.echo(f"   综合得分：{result.final_score:.4f}")
            typer.echo(f"   语义得分：{result.semantic_score:.4f}")
            typer.echo(f"   关键词得分：{result.keyword_score:.4f}")
            typer.echo(f"   可信度得分：{result.confidence_score:.4f}")
            typer.echo(f"   可信度原因：{result.confidence_reason}")
    else:
        typer.echo("📭 未找到匹配结果")


@app.command("maintenance")
def maintenance(
    config_path: Annotated[
        str,
        typer.Option(
            "-c",
            "--config",
            help="配置文件路径",
        ),
    ] = "./config.yaml",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="仅检查，不执行清理",
        ),
    ] = False,
    clean_only: Annotated[
        bool,
        typer.Option(
            "--clean-only",
            help="仅清理软删除记录，不执行 VACUUM",
        ),
    ] = False,
):
    """数据库维护（清理软删除记录 + VACUUM）"""
    ctx = PipelineContext(config_path=config_path, dry_run=dry_run)

    stages = [
        ConfigLoadStage(),
        DBInitStage(),  # 初始化数据库连接
        CleanupStage(),
    ]
    if not clean_only:
        stages.append(VacuumStage())

    pipeline = Pipeline(name="maintenance", stages=stages)
    pipeline.run(ctx)


if __name__ == "__main__":
    app()
