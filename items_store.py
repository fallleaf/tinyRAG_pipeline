#!/usr/bin/env python3
"""
items_store.py - 家庭物品存放位置管理模块

提供物品的存储、查询和推荐功能，配合 tinyRAG_pipeline 的检索能力。
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from helpers.logger import logger


class ItemsStore:
    """家庭物品存储管理器"""

    def __init__(self, items_dir: str, db=None, config=None):
        """
        初始化物品存储管理器

        Args:
            items_dir: 物品 MD 文档存储目录
            db: DatabaseManager 实例（用于触发索引更新）
            config: 配置对象
        """
        self.items_dir = Path(items_dir).expanduser().resolve()
        self.items_dir.mkdir(parents=True, exist_ok=True)
        self.db = db
        self.config = config

    def _generate_item_id(self) -> str:
        """生成唯一物品 ID"""
        return uuid.uuid4().hex[:8]

    def _generate_filename(self, item_name: str, item_id: str) -> str:
        """生成物品文件名"""
        timestamp = datetime.now().strftime("%Y%m%d")
        # 清理物品名称中的特殊字符
        safe_name = "".join(c for c in item_name if c.isalnum() or c in "._- ")
        safe_name = safe_name.strip()[:50]
        return f"{safe_name}_{timestamp}_{item_id}.md"

    def _find_existing_item(self, item_name: str) -> dict | None:
        """
        查找已存在的同名物品

        Returns:
            找到则返回物品信息字典，否则返回 None
        """
        if not self.db:
            return None

        try:
            # 通过 FTS 或文件名匹配查找
            row = self.db.conn.execute(
                """
                SELECT f.id, f.file_path, f.absolute_path
                FROM files f
                WHERE f.file_path LIKE ? AND f.is_deleted = 0
                LIMIT 1
                """,
                (f"%{item_name}%",),
            ).fetchone()

            if row:
                # 读取文件内容获取 item_id
                try:
                    content = Path(row["absolute_path"]).read_text(encoding="utf-8")
                    frontmatter = self._parse_frontmatter(content)
                    return {
                        "file_id": row["id"],
                        "file_path": row["file_path"],
                        "absolute_path": row["absolute_path"],
                        "item_id": frontmatter.get("item_id"),
                        "item_name": frontmatter.get("item_name"),
                    }
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"查找已存在物品失败: {e}")

        return None

    def _parse_frontmatter(self, content: str) -> dict:
        """解析 Markdown 文件的 frontmatter"""
        if not content.startswith("---"):
            return {}

        try:
            end_idx = content.index("---", 3)
            yaml_str = content[4:end_idx].strip()
            data = yaml.safe_load(yaml_str)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _build_frontmatter(self, data: dict) -> str:
        """构建 YAML frontmatter"""
        return yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def _build_content(
        self,
        item_name: str,
        location: str,
        notes: str | None = None,
        related_items: list[str] | None = None,
    ) -> str:
        """构建 Markdown 正文内容"""
        lines = [f"# {item_name}\n", "\n## 存放位置\n", f"{location}\n"]

        if notes:
            lines.append("\n## 备注信息\n")
            lines.append(f"{notes}\n")

        if related_items:
            lines.append("\n## 相关物品\n")
            for item in related_items:
                lines.append(f"- {item}\n")

        return "".join(lines)

    def store_item(
        self,
        item_name: str,
        location: str,
        aliases: list[str] | None = None,
        category: str = "misc",
        notes: str | None = None,
    ) -> dict[str, Any]:
        """
        存储或更新物品位置

        Args:
            item_name: 物品名称
            location: 存放位置描述
            aliases: 物品别名列表
            category: 物品分类
            notes: 备注信息

        Returns:
            操作结果字典
        """
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%dT%H:%M:%S")
        today_str = now.strftime("%Y-%m-%d")

        # 检查是否已存在同名物品
        existing = self._find_existing_item(item_name)

        if existing:
            # 更新现有物品
            item_id = existing.get("item_id", self._generate_item_id())
            file_path = Path(existing["absolute_path"])

            # 读取现有内容获取更多信息
            try:
                old_content = file_path.read_text(encoding="utf-8")
                old_fm = self._parse_frontmatter(old_content)
                # 保留原有的别名和分类（如果未提供新值）
                if aliases is None:
                    aliases = old_fm.get("aliases", [])
                if category == "misc" and old_fm.get("category"):
                    category = old_fm.get("category")
            except Exception:
                pass

            # 构建更新后的 frontmatter
            frontmatter = {
                "item_id": item_id,
                "item_name": item_name,
                "category": category,
                "aliases": aliases or [],
                "status": "active",
                "final_date": today_str,
                "created_at": existing.get("created_at", now_str),
                "updated_at": now_str,
            }

            # 构建完整内容
            fm_str = self._build_frontmatter(frontmatter)
            content = self._build_content(item_name, location, notes)
            full_content = f"---\n{fm_str}---\n{content}"

            # 写入文件
            file_path.write_text(full_content, encoding="utf-8")

            logger.info(f"已更新物品「{item_name}」: {location}")

            return {
                "success": True,
                "item_id": item_id,
                "item_name": item_name,
                "location": location,
                "action": "updated",
                "message": f"已更新「{item_name}」存放位置：{location}",
            }
        else:
            # 创建新物品
            item_id = self._generate_item_id()
            filename = self._generate_filename(item_name, item_id)
            file_path = self.items_dir / filename

            # 构建 frontmatter
            frontmatter = {
                "item_id": item_id,
                "item_name": item_name,
                "category": category,
                "aliases": aliases or [],
                "status": "active",
                "final_date": today_str,
                "created_at": now_str,
                "updated_at": now_str,
            }

            # 构建完整内容
            fm_str = self._build_frontmatter(frontmatter)
            content = self._build_content(item_name, location, notes)
            full_content = f"---\n{fm_str}---\n{content}"

            # 写入文件
            file_path.write_text(full_content, encoding="utf-8")

            logger.info(f"已创建物品「{item_name}」: {location}")

            return {
                "success": True,
                "item_id": item_id,
                "item_name": item_name,
                "location": location,
                "action": "created",
                "file_path": str(file_path),
                "message": f"已记录「{item_name}」存放位置：{location}",
            }

    def update_status(self, item_name: str, status: str) -> dict[str, Any]:
        """
        更新物品状态

        Args:
            item_name: 物品名称
            status: 新状态 (active/archived/lost/lent)

        Returns:
            操作结果字典
        """
        existing = self._find_existing_item(item_name)
        if not existing:
            return {
                "success": False,
                "error": f"未找到物品「{item_name}」",
            }

        file_path = Path(existing["absolute_path"])
        try:
            content = file_path.read_text(encoding="utf-8")
            frontmatter = self._parse_frontmatter(content)

            # 更新状态
            frontmatter["status"] = status
            frontmatter["updated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            frontmatter["final_date"] = datetime.now().strftime("%Y-%m-%d")

            # 重建文件
            fm_str = self._build_frontmatter(frontmatter)

            # 提取正文部分
            if "---" in content[4:]:
                body_start = content.index("---", 4) + 3
                body = content[body_start:].strip()
            else:
                body = content

            full_content = f"---\n{fm_str}---\n\n{body}\n"
            file_path.write_text(full_content, encoding="utf-8")

            return {
                "success": True,
                "item_name": item_name,
                "status": status,
                "message": f"已将「{item_name}」状态更新为：{status}",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def delete_item(self, item_name: str, soft: bool = True) -> dict[str, Any]:
        """
        删除物品

        Args:
            item_name: 物品名称
            soft: 是否软删除（标记为 archived）

        Returns:
            操作结果字典
        """
        existing = self._find_existing_item(item_name)
        if not existing:
            return {
                "success": False,
                "error": f"未找到物品「{item_name}」",
            }

        if soft:
            # 软删除：标记为 archived
            return self.update_status(item_name, "archived")
        else:
            # 硬删除：删除文件
            try:
                file_path = Path(existing["absolute_path"])
                file_path.unlink()
                return {
                    "success": True,
                    "item_name": item_name,
                    "action": "deleted",
                    "message": f"已删除物品「{item_name}」",
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                }

    def list_items(
        self,
        category: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        列出所有物品

        Args:
            category: 按分类筛选
            status: 按状态筛选
            limit: 返回数量限制

        Returns:
            物品列表
        """
        items = []

        for md_file in self.items_dir.glob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                fm = self._parse_frontmatter(content)

                # 筛选条件
                if category and fm.get("category") != category:
                    continue
                if status and fm.get("status") != status:
                    continue

                # 提取位置信息
                location = self._extract_location(content)

                items.append({
                    "item_id": fm.get("item_id"),
                    "item_name": fm.get("item_name"),
                    "category": fm.get("category", "misc"),
                    "status": fm.get("status", "active"),
                    "location": location,
                    "aliases": fm.get("aliases", []),
                    "final_date": fm.get("final_date"),
                    "file_path": str(md_file),
                })
            except Exception as e:
                logger.warning(f"读取物品文件失败 {md_file}: {e}")

        # 按更新时间排序（最新在前）
        items.sort(key=lambda x: x.get("final_date") or "", reverse=True)

        return items[:limit]

    def _extract_location(self, content: str) -> str:
        """从 Markdown 内容中提取存放位置"""
        try:
            # 查找 "## 存放位置" 部分
            if "## 存放位置" in content:
                start = content.index("## 存放位置") + len("## 存放位置")
                # 找到下一个 ## 或者文档末尾
                end = content.find("\n## ", start)
                if end == -1:
                    end = len(content)
                location = content[start:end].strip()
                # 清理换行和多余空格
                location = " ".join(location.split())
                return location
        except Exception:
            pass
        return ""


# 物品分类定义
ITEM_CATEGORIES = {
    "misc": "杂项",
    "daily": "日用品",
    "documents": "证件",
    "electronics": "电子产品",
    "tools": "工具",
    "seasonal": "季节物品",
    "sports": "运动",
    "storage": "收纳",
    "kitchen": "厨房用品",
    "clothing": "衣物",
    "books": "书籍",
    "medicine": "药品",
}

# 物品状态定义
ITEM_STATUS = {
    "active": {"label": "有效", "weight": 1.1},
    "archived": {"label": "已移走", "weight": 0.5},
    "lost": {"label": "已丢失", "weight": 0.3},
    "lent": {"label": "已借出", "weight": 0.7},
}
