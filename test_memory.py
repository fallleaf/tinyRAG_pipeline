#!/usr/bin/env python3
"""测试内存优化效果

对比优化前后的内存使用情况
"""
import gc
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from helpers.logger import logger
from pipeline.context import PipelineContext
from pipeline.pipeline import Pipeline
from pipeline.stages.chunk_stage import ChunkStage
from pipeline.stages.embed_stage import EmbedStage
from pipeline.stages.index_stage import IndexStage
from pipeline.stages.scan_stage import ScanStage
from pipeline.stages.config_stage import ConfigStage


def test_memory_usage():
    """测试内存使用情况"""
    try:
        import psutil
        process = psutil.Process()
    except ImportError:
        logger.error("❌ psutil 未安装，无法测试内存使用")
        logger.info("请运行: pip install psutil")
        return

    # 初始化上下文
    ctx = PipelineContext(config_path="config.yaml")
    ctx.memory_limit_mb = 2048  # 设置较低的内存限制，便于测试
    ctx.memory_check_interval = 50  # 每 50 个 chunk 检查一次

    # 记录初始内存
    mem_before = process.memory_info().rss / 1024 / 1024
    logger.info(f"📊 初始内存使用: {mem_before:.2f} MB")

    # 运行配置阶段
    config_stage = ConfigStage()
    ctx = config_stage.process(ctx)

    # 运行扫描阶段
    scan_stage = ScanStage()
    ctx = scan_stage.process(ctx)

    # 记录扫描后内存
    mem_after_scan = process.memory_info().rss / 1024 / 1024
    logger.info(f"📊 扫描后内存使用: {mem_after_scan:.2f} MB (+{mem_after_scan - mem_before:.2f} MB)")

    # 运行分块阶段
    chunk_stage = ChunkStage()
    ctx = chunk_stage.process(ctx)

    # 记录分块后内存
    mem_after_chunk = process.memory_info().rss / 1024 / 1024
    logger.info(f"📊 分块后内存使用: {mem_after_chunk:.2f} MB (+{mem_after_chunk - mem_after_scan:.2f} MB)")

    # 运行向量化阶段
    embed_stage = EmbedStage()
    ctx = embed_stage.process(ctx)

    # 记录向量化后内存
    mem_after_embed = process.memory_info().rss / 1024 / 1024
    logger.info(f"📊 向量化后内存使用: {mem_after_embed:.2f} MB (+{mem_after_embed - mem_after_chunk:.2f} MB)")

    # 运行入库阶段
    index_stage = IndexStage()
    ctx = index_stage.process(ctx)

    # 记录入库后内存
    mem_after_index = process.memory_info().rss / 1024 / 1024
    logger.info(f"📊 入库后内存使用: {mem_after_index:.2f} MB (+{mem_after_index - mem_after_embed:.2f} MB)")

    # 强制垃圾回收
    gc.collect()
    mem_after_gc = process.memory_info().rss / 1024 / 1024
    logger.info(f"📊 垃圾回收后内存使用: {mem_after_gc:.2f} MB (释放 {mem_after_index - mem_after_gc:.2f} MB)")

    # 输出总结
    logger.info("=" * 60)
    logger.info("📊 内存使用总结:")
    logger.info(f"  初始: {mem_before:.2f} MB")
    logger.info(f"  扫描: {mem_after_scan:.2f} MB (+{mem_after_scan - mem_before:.2f} MB)")
    logger.info(f"  分块: {mem_after_chunk:.2f} MB (+{mem_after_chunk - mem_after_scan:.2f} MB)")
    logger.info(f"  向量化: {mem_after_embed:.2f} MB (+{mem_after_embed - mem_after_chunk:.2f} MB)")
    logger.info(f"  入库: {mem_after_index:.2f} MB (+{mem_after_index - mem_after_embed:.2f} MB)")
    logger.info(f"  GC 后: {mem_after_gc:.2f} MB (释放 {mem_after_index - mem_after_gc:.2f} MB)")
    logger.info(f"  总增长: {mem_after_gc - mem_before:.2f} MB")
    logger.info("=" * 60)

    # 输出处理统计
    logger.info("=" * 60)
    logger.info("📊 处理统计:")
    logger.info(f"  文件数: {len(ctx.files_to_index)}")
    logger.info(f"  入库 chunks: {ctx.total_indexed}")
    logger.info(f"  错误数: {len(ctx.errors)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_memory_usage()
