"""utils/fts_content.py - 统一 FTS5 内容构建（消除 build_index/server/rag_cli 3处重复）

此模块复用 tinyRAG 的 jieba_helper，通过 sys.path 确保正确导入。
"""

import os
import sys
from pathlib import Path


def _get_jieba_segment():
    """延迟导入 tinyRAG 的 jieba_segment"""
    tinyrag_root = Path(__file__).parent.parent
    if str(tinyrag_root) not in sys.path:
        sys.path.insert(0, str(tinyrag_root))
    from utils.jieba_helper import jieba_segment

    return jieba_segment


def prepare_fts_content(chunk, file_path: str) -> str:
    """构建 FTS5 索引文本

    将 chunk 的多维度信息拼接为 jieba 分词后的文本，
    filename 和 section_title 各重复 2 次以提升 FTS5 关键词权重。
    """
    jieba_segment = _get_jieba_segment()

    metadata = chunk.metadata or {}
    tags = metadata.get("tags", [])
    if tags is None:
        tags = []
    if isinstance(tags, str):
        tags = [tags]
    tag_str = " ".join([f"#{t.strip()}" for t in tags if t])
    doc_type = metadata.get("doc_type") or ""
    filename = os.path.basename(file_path)
    section_title = chunk.section_title or ""

    parts = [
        jieba_segment(filename),
        jieba_segment(filename),
        jieba_segment(chunk.section_path or ""),
        jieba_segment(section_title),
        jieba_segment(section_title),
        jieba_segment(tag_str),
        jieba_segment(doc_type),
        jieba_segment(chunk.content),
    ]
    return " ".join(filter(None, parts)).strip()
