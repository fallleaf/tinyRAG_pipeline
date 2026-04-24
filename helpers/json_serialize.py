"""utils/json_serialize.py - 统一 JSON 序列化工具（消除3处重复）"""
from datetime import date, datetime
from typing import Any


def json_serialize(obj: Any) -> str:
    """JSON 不支持的类型序列化器，用于 json.dumps(default=...)"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
