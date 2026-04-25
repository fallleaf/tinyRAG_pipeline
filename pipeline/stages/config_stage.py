"""pipeline/stages/config_stage.py - 配置加载 Stage

从 config.yaml 加载配置，初始化 jieba 分词器。
"""

from __future__ import annotations

import os
from pathlib import Path

from pipeline.context import PipelineContext
from pipeline.stage import Stage
from helpers.logger import logger


class ConfigLoadStage(Stage):
    """加载配置并初始化 jieba 分词"""

    name = "config_load"
    description = "加载配置文件并初始化 jieba 分词器"

    def process(self, ctx: PipelineContext) -> PipelineContext:
        # 独立项目，直接使用配置文件所在目录
        config_path = Path(ctx.config_path).resolve()
        config_dir = config_path.parent

        from config import load_config
        from utils.jieba_helper import load_jieba_user_dict

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在：{config_path}")

        # 切换 CWD 到配置文件所在目录，确保相对路径（如 ./data/rag.db）正确解析
        original_cwd = os.getcwd()
        os.chdir(config_dir)

        try:
            config = load_config(str(config_path))

            # 在 CWD 正确时初始化 jieba 自定义词典（确保相对路径能正确解析）
            load_jieba_user_dict(config)

            # 将配置中的相对路径转为绝对路径（基于 config_dir），
            # 这样后续 Stage 在任何 CWD 下都能正确访问
            if (
                config.jieba_user_dict
                and not Path(config.jieba_user_dict).is_absolute()
            ):
                config.jieba_user_dict = str(config_dir / config.jieba_user_dict)
            if config.db_path and not Path(config.db_path).is_absolute():
                config.db_path = str(config_dir / config.db_path)
            if (
                config.embedding_model.cache_dir
                and not Path(config.embedding_model.cache_dir).is_absolute()
            ):
                config.embedding_model.cache_dir = str(
                    config_dir / config.embedding_model.cache_dir
                )
        finally:
            os.chdir(original_cwd)

        ctx.config = config
        ctx.config_path = str(config_path)
        ctx.db_path = config.db_path

        # 准备 vault 配置
        ctx.vault_configs = [(v.name, v.path) for v in config.vaults if v.enabled]
        for v in config.vaults:
            if v.enabled:
                ctx.vault_excludes[v.name] = (
                    (frozenset(v.exclude.dirs), v.exclude.patterns)
                    if v.exclude
                    else (frozenset(), [])
                )

        logger.info(f"✅ 配置加载成功：{config_path}")
        logger.info(f"   📂 启用 Vault: {[v[0] for v in ctx.vault_configs]}")
        return ctx
