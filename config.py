#!/usr/bin/env python3
"""config.py - tinyRAG_pipeline 配置契约与加载层"""

import re
from pathlib import Path
from typing import Any, Literal
import yaml
from pydantic import BaseModel, Field, field_validator


class ExcludeConfig(BaseModel):
    dirs: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)

    @field_validator("dirs", "patterns", mode="before")
    @classmethod
    def none_to_empty_list(cls, v):
        return v if v is not None else []


class VaultConfig(BaseModel):
    path: str
    name: str
    enabled: bool = True
    exclude: ExcludeConfig = Field(default_factory=ExcludeConfig)

    @field_validator("path")
    @classmethod
    def expand_vault_path(cls, v: str) -> str:
        return str(Path(v).expanduser())


class ModelConfig(BaseModel):
    name: str = "BAAI/bge-small-zh-v1.5"
    size: Literal["large", "small", "base"] = "small"
    cache_dir: str = "~/.cache/fastembed"
    unload_after_seconds: int = 30
    dimensions: int = 512
    batch_size: int = 32

    @field_validator("cache_dir")
    @classmethod
    def expand_cache_dir(cls, v: str) -> str:
        return str(Path(v).expanduser())


class DateDecayConfig(BaseModel):
    enabled: bool = True
    half_life_days: int = 365
    min_weight: float = 0.5
    type_specific_decay: dict[str, int] = Field(default_factory=dict)


class ConfidenceConfig(BaseModel):
    type_rules: dict[str, float] = Field(default_factory=dict)
    doc_type_rules: dict[str, float] = Field(default_factory=dict)
    status_rules: dict[str, float] = Field(default_factory=dict)
    date_decay: DateDecayConfig = Field(default_factory=DateDecayConfig)
    default_weight: float = 1.0


class MaintenanceConfig(BaseModel):
    soft_delete_threshold: float = 0.2
    auto_vacuum: bool = True


class CacheConfig(BaseModel):
    db_path: str = "./data/cache.db"
    ttl_seconds: int = 36000
    max_entries: int = 1000


class RetrievalConfig(BaseModel):
    alpha: float = 0.6
    beta: float = 0.4


class MemoryConfig(BaseModel):
    limit_mb: int = 4096
    check_interval: int = 100


class Config(BaseModel):
    # 知识库 Vault
    vaults: list[VaultConfig] = Field(default_factory=list)

    # 全局排除规则
    exclude: ExcludeConfig = Field(default_factory=ExcludeConfig)

    # 数据库
    db_path: str = "./data/rag.db"

    # 物品存储目录（家庭物品管理系统）
    items_dir: str = "./data/items"

    # 嵌入模型
    embedding_model: ModelConfig = Field(default_factory=ModelConfig)

    # 分块
    chunking: dict[str, Any] = Field(default_factory=dict)

    # 置信度评分
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)

    # 日志级别
    log_level: str = "INFO"

    # 运维
    maintenance: MaintenanceConfig = Field(default_factory=MaintenanceConfig)

    # 缓存
    cache: CacheConfig = Field(default_factory=CacheConfig)

    # 检索
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    # jieba 分词
    jieba_user_dict: str = "./data/jieba_user_dict.txt"
    jieba_seg_mode: Literal["precise", "search"] = "precise"

    # 索引构建
    stream_batch_size: int = 64
    max_concurrent_files: int = 4

    # 内存监控配置
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    # 流式处理配置
    # stream_batch_size: int = 100

    @field_validator("chunking")
    @classmethod
    def default_chunking(cls, v: dict[str, Any]) -> dict[str, Any]:
        """设置分块默认值"""
        defaults = {
            "max_tokens": 512,
            "overlap": 100,
            "token_mode": "tiktoken",
            "chinese_chars_per_token": 1.5,
            "english_chars_per_token": 4.0,
            "max_chars_multiplier": 2.5,
        }
        return {**defaults, **v}

    @field_validator("confidence")
    @classmethod
    def default_confidence(cls, v: ConfidenceConfig) -> ConfidenceConfig:
        """设置置信度默认值"""
        return v

    @field_validator("retrieval")
    @classmethod
    def default_retrieval(cls, v: RetrievalConfig) -> RetrievalConfig:
        """设置检索默认值"""
        return v


def load_config(config_path: str) -> Config:
    """从 YAML 文件加载配置并验证"""
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 处理 vault 路径
    if "vaults" in data:
        for v in data["vaults"]:
            if "path" in v:
                v["path"] = str(Path(v["path"]).expanduser())

    config = Config(**data)
    return config


def get_merged_exclude(config: Config) -> tuple[set[str], list[str]]:
    """合并全局和 vault 级别的排除规则"""
    global_dirs = set(config.exclude.dirs)
    global_patterns = list(config.exclude.patterns)

    for vault in config.vaults:
        if vault.exclude:
            global_dirs.update(vault.exclude.dirs)
            global_patterns.extend(vault.exclude.patterns)

    return global_dirs, global_patterns
