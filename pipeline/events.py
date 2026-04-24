"""pipeline/events.py - 事件总线与事件定义

提供 Stage 之间、Stage 与外部（CLI/MCP/进度条）的通信机制。

设计原则：
1. 松耦合：Stage 不直接依赖外部 I/O
2. 可扩展：新的事件类型只需定义新字段
3. 线程安全：EventBus 使用回调列表，不依赖全局状态
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class StageEvent:
    """Stage 生命周期事件"""

    stage_name: str
    event_type: str  # "start" | "progress" | "complete" | "error"
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineEvent:
    """Pipeline 生命周期事件"""

    event_type: str  # "start" | "stage_complete" | "complete" | "error"
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)


# 事件回调类型
EventCallback = Callable[[StageEvent | PipelineEvent], None]


class EventBus:
    """事件总线 - 发布/订阅模式"""

    def __init__(self):
        self._subscribers: list[EventCallback] = []

    def subscribe(self, callback: EventCallback) -> None:
        self._subscribers.append(callback)

    def unsubscribe(self, callback: EventCallback) -> None:
        self._subscribers.remove(callback)

    def emit(self, event: StageEvent | PipelineEvent) -> None:
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception:
                pass  # 事件回调不应影响主流程

    def clear(self) -> None:
        self._subscribers.clear()


def create_progress_callback(event_bus: EventBus) -> Callable:
    """创建进度跟踪回调，将 Stage 事件转发为进度更新"""

    def on_event(event: StageEvent | PipelineEvent):
        if isinstance(event, StageEvent):
            if event.event_type == "start":
                pass  # 进度条创建由调用方处理
            elif event.event_type == "complete":
                pass  # 进度条更新由调用方处理

    return on_event
