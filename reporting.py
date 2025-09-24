"""控制台输出工具。"""

from __future__ import annotations

import logging

from .tables import TableBundle


def print_tables(tables: TableBundle, logger: logging.Logger) -> None:
    """将核心表格以中文形式打印到控制台/日志。"""

    logger.info("分类报告:\n%s", tables.console_report)
    if tables.console_cv:
        logger.info("候选模型 Top-5:\n%s", tables.console_cv)
    if tables.console_file_report:
        logger.info("文件级分类报告:\n%s", tables.console_file_report)
