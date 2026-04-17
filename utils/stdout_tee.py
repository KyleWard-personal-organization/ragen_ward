"""stdout / stderr → project-root `stdout.txt` tee.

把当前进程的 `sys.stdout` 和 `sys.stderr` 替换成 Tee 对象，
输出**同时**打到原终端和项目根目录下的一个文件（默认 `stdout.txt`）。

使用规范
--------
**必须**在 `import loguru`（以及其他会缓存 `sys.stderr` 引用的第三方库）**之前**
调用 `setup_stdout_tee()`，否则这些库在 tee 生效前已经记住了旧的 stderr
对象，它们发出的日志不会进入文件。

最小示例::

    from utils.stdout_tee import setup_stdout_tee
    setup_stdout_tee()          # 尽可能靠前，最好放在 script 开头

    # 之后再 import 所有依赖库
    from utils.logger import logger
    import transformers
    ...

细节
----
- 终端收到的是**原始数据**（保留 tqdm 的 `\\r` 动画、loguru 的 ANSI 色）。
- 写到文件的数据会把 `\\r` 替换成 `\\n`：tqdm 的每一次刷新都变成新的一行，
  grep / less / 编辑器查看都清爽（代价：文件体积略增）。
- 追加模式（默认）下每次进程启动会插入一个时间戳 banner 便于区分多次运行。
- 文件用行缓冲（`buffering=1`）打开，进程崩溃时也能保留大部分已写内容。
"""
from __future__ import annotations

import atexit
import datetime as _dt
import sys
from pathlib import Path
from typing import TextIO

# 项目根目录 = 本文件所在 utils/ 的父目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _TeeStream:
    """把写入分发到 terminal + file 的伪文件对象。"""

    def __init__(self, terminal: TextIO, file: TextIO):
        self.terminal = terminal
        self.file = file

    def write(self, data: str) -> int:
        self.terminal.write(data)
        self.file.write(data.replace("\r", "\n"))
        return len(data)

    def flush(self) -> None:
        try:
            self.terminal.flush()
        except Exception:
            pass
        try:
            self.file.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return bool(self.terminal.isatty())

    def __getattr__(self, name: str):
        return getattr(self.terminal, name)


_installed = False


def setup_stdout_tee(filename: str = "stdout.txt", mode: str = "a") -> Path:
    """替换 `sys.stdout` / `sys.stderr` 为 tee，把输出同时写到 `<PROJECT_ROOT>/<filename>`。

    Args:
        filename: 目标文件名（相对项目根），默认 ``stdout.txt``。
        mode: ``"a"`` 追加 / ``"w"`` 每次覆盖。默认 ``"a"``，启动时写入时间戳分隔符。

    Returns:
        tee 文件的绝对路径。重复调用为幂等操作（只安装一次，后续直接返回路径）。
    """
    global _installed
    log_path = (_PROJECT_ROOT / filename).resolve()
    if _installed:
        return log_path

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, mode, encoding="utf-8", buffering=1)

    if mode == "a":
        banner = f"\n\n===== Run started at {_dt.datetime.now():%Y-%m-%d %H:%M:%S} =====\n"
        log_file.write(banner)
        log_file.flush()

    sys.stdout = _TeeStream(sys.__stdout__, log_file)
    sys.stderr = _TeeStream(sys.__stderr__, log_file)

    atexit.register(log_file.close)
    _installed = True
    return log_path
