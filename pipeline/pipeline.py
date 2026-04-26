"""pipeline/pipeline.py - 流水线编排器

Pipeline 按顺序执行一组 Stage，管理生命周期和错误处理。

设计原则：
1. 组合优于继承：Pipeline 由 Stage 列表组合而成
2. 短路失败：某个 Stage 失败时，可选择停止或继续
3. 可观测：通过 EventBus 发射 Pipeline 级别事件
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pipeline.context import PipelineContext
from pipeline.events import EventBus, PipelineEvent
from pipeline.stage import Stage, StageResult, StageStatus
from helpers.logger import logger


@dataclass
class PipelineResult:
    """Pipeline 执行结果"""

    success: bool = True
    total_stages: int = 0
    completed_stages: int = 0
    failed_stages: int = 0
    skipped_stages: int = 0
    duration_ms: float = 0.0
    stage_results: list[StageResult] = field(default_factory=list)
    context_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return self.failed_stages > 0


class Pipeline:
    """流水线编排器

    用法:
        pipeline = Pipeline("build", [ConfigLoadStage(), ScanStage(), ...])
        result = pipeline.run(ctx)
    """

    def __init__(
        self,
        name: str,
        stages: list[Stage],
        event_bus: EventBus | None = None,
        fail_fast: bool = True,
    ):
        self.name = name
        self.stages = stages
        self.event_bus = event_bus or EventBus()
        self.fail_fast = fail_fast

    def run(self, ctx: PipelineContext | None = None) -> tuple[PipelineResult, PipelineContext]:
        """执行流水线

        Args:
            ctx: 可选的初始上下文，为 None 时自动创建
        Returns:
            (PipelineResult, PipelineContext) 元组
        """
        if ctx is None:
            ctx = PipelineContext()

        ctx.start_time = time.time()
        self.event_bus.emit(
            PipelineEvent(
                event_type="start",
                message=f"Pipeline [{self.name}] 启动，共 {len(self.stages)} 个 Stage",
            )
        )

        logger.info(f"🚀 Pipeline [{self.name}] 启动，共 {len(self.stages)} 个 Stage")

        results: list[StageResult] = []
        completed = 0
        failed = 0
        skipped = 0

        for stage in self.stages:
            result = stage.execute(ctx)
            results.append(result)

            if result.status == StageStatus.COMPLETED:
                completed += 1
            elif result.status == StageStatus.FAILED:
                failed += 1
                if self.fail_fast:
                    logger.error(f"❌ Pipeline [{self.name}] 因 [{stage.name}] 失败而终止")
                    break
            elif result.status == StageStatus.SKIPPED:
                skipped += 1

            self.event_bus.emit(
                PipelineEvent(
                    event_type="stage_complete",
                    message=f"Stage [{stage.name}] {result.status.value}",
                    data={"stage": stage.name, "status": result.status.value},
                )
            )

        ctx.finish()
        duration_ms = (time.time() - ctx.start_time) * 1000

        pipeline_result = PipelineResult(
            success=failed == 0,
            total_stages=len(self.stages),
            completed_stages=completed,
            failed_stages=failed,
            skipped_stages=skipped,
            duration_ms=duration_ms,
            stage_results=results,
            context_summary=ctx.summary(),
        )

        event_type = "complete" if pipeline_result.success else "error"
        self.event_bus.emit(
            PipelineEvent(
                event_type=event_type,
                message=f"Pipeline [{self.name}] {'完成' if pipeline_result.success else '失败'} "
                f"({completed}/{len(self.stages)} Stage, {duration_ms:.0f}ms)",
                data=pipeline_result.__dict__,
            )
        )

        if pipeline_result.success:
            logger.success(
                f"🎉 Pipeline [{self.name}] 完成 " f"({completed} 完成, {skipped} 跳过, {duration_ms:.0f}ms)"
            )
        else:
            logger.error(f"❌ Pipeline [{self.name}] 失败 ({failed} 个 Stage 出错)")

        return pipeline_result, ctx

    def add_stage(self, stage: Stage) -> "Pipeline":
        """链式添加 Stage"""
        self.stages.append(stage)
        return self

    def describe(self) -> str:
        """返回 Pipeline 结构描述"""
        lines = [f"Pipeline: {self.name}"]
        for i, stage in enumerate(self.stages, 1):
            lines.append(f"  {i}. {stage.name}: {stage.description}")
        return "\n".join(lines)
