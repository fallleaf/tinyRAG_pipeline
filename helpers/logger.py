"""utils/logger.py - 日志工具（输出到 stderr，MCP 安全）

延迟导入 loguru，避免在模块导入时产生副作用。
"""
import sys
from pathlib import Path


def _ensure_log_dir():
    """确保日志目录存在"""
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    return str(log_dir / "tinyrag_pipeline.log")


def setup_logger(level: str = "INFO"):
    """配置日志并返回 logger"""
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level=level,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
    try:
        logger.add(_ensure_log_dir(), level="DEBUG", rotation="10 MB", retention="7 days", encoding="utf-8")
    except Exception:
        pass  # 日志文件创建失败不应阻断主流程
    return logger


# 模块级 logger（首次使用时初始化）
_logger = None

def get_logger():
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


class _LoggerProxy:
    """Logger 代理，延迟初始化"""
    def __getattr__(self, name):
        return getattr(get_logger(), name)


logger = _LoggerProxy()
