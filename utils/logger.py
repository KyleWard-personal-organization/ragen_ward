# utils/logger.py
import os
import sys
from loguru import logger
from configs.constants import LOG_DIR
os.makedirs(LOG_DIR, exist_ok=True)


PREFIX_WIDTH = 40
def _formatter(record):
    prefix = f"{record['name']}:{record['function']}:{record['line']}"

    if len(prefix) > PREFIX_WIDTH:
        prefix = prefix[:PREFIX_WIDTH]
    else:
        prefix = prefix.ljust(PREFIX_WIDTH)

    record["extra"]["prefix"] = prefix

    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[prefix]}</cyan> - "
        "<level>{message}</level>\n{exception}"
    )


def setup_logger(
        level: str = "INFO",
        log_file: str | None = None,
        console: bool = True,
):
    """
    配置 logger：
    - log_file: 日志文件名（例如 'train_nli.log'、'detect.log'）
      如果传 None，则只输出到控制台，不写文件。
    - level: 日志等级
    调用多次会覆盖之前的配置（logger.remove()），适合在不同入口脚本里重配。
    """
    logger.remove()

    if console:
        logger.add(
            sys.stderr,
            level=level,
            format=_formatter,
            colorize=True,
        )

    # 文件：如果指定了 log_file，就写入对应文件
    if log_file is not None:
        full_path = os.path.join(LOG_DIR, log_file)
        logger.add(
            full_path,
            level=level,
            format=_formatter,
            colorize=False,
            encoding="utf-8",
        )
    return


setup_logger(

)
