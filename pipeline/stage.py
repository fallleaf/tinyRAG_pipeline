"""pipeline/stage.py - Stage 协议与基类

Stage 是流水线中最小的处理单元。每个 Stage 接收 PipelineContext，
执行特定的处理逻辑，返回 StageResult。

设计原则：
1. 单一职责：每个 Stage 只做一件事
2. 幂等性：相同输入应产生相同输出
3. 可组合：Stage 之间通过 Context 解耦
4. 可观测：通过 EventBus 发射事件
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pipeline.context import PipelineContext
from pipeline.events import EventBus, StageEvent
from helpers.logger import logger


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SKIPPED = "skipped"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StageResult:
    """Stage 执行结果"""

    status: StageStatus = StageStatus.PENDING
    message: str = ""
    duration_ms: float = 0.0
    data: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.status in (StageStatus.COMPLETED, StageStatus.SKIPPED)


class Stage(ABC):
    """Stage 抽象基类

    子类必须实现 process() 方法。
    可选重写 should_run() 控制是否执行。
    """

    name: str = ""
    description: str = ""

    def __init__(self, event_bus: EventBus | None = None):
        self.event_bus = event_bus or EventBus()
        if not self.name:
            self.name = self.__class__.__name__

    def should_run(self, ctx: PipelineContext) -> bool:
        """判断是否应该执行此 Stage（默认总是执行）"""
        return True

    @abstractmethod
    def process(self, ctx: PipelineContext) -> PipelineContext:
        """执行 Stage 核心逻辑

        Args:
            ctx: 流水线上下文
        Returns:
            更新后的上下文（通常就地修改后返回）
        """
        ...

    def execute(self, ctx: PipelineContext) -> StageResult:
        """执行 Stage（含事件发射、计时、错误处理）"""
        if not self.should_run(ctx):
            logger.debug(f"⏭️ Stage [{self.name}] 跳过")
            return StageResult(status=StageStatus.SKIPPED, message="条件不满足，跳过")

        self.event_bus.emit(
            StageEvent(
                stage_name=self.name,
                event_type="start",
                message=f"开始执行 {self.name}",
            )
        )

        start = time.perf_counter()
        try:
            ctx = self.process(ctx)
            duration_ms = (time.perf_counter() - start) * 1000
            result = StageResult(
                status=StageStatus.COMPLETED,
                message=f"{self.name} 完成",
                duration_ms=duration_ms,
            )
            self.event_bus.emit(
                StageEvent(
                    stage_name=self.name,
                    event_type="complete",
                    message=f"{self.name} 完成 ({duration_ms:.0f}ms)",
                    data={"duration_ms": duration_ms},
                )
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            result = StageResult(
                status=StageStatus.FAILED,
                message=f"{self.name} 失败: {e}",
                duration_ms=duration_ms,
            )
            ctx.add_error(f"[{self.name}] {e}")
            self.event_bus.emit(
                StageEvent(
                    stage_name=self.name,
                    event_type="error",
                    message=f"{self.name} 失败: {e}",
                    data={"error": str(e)},
                )
            )
            logger.error(f"❌ Stage [{self.name}] 失败: {e}", exc_info=True)

        return result
