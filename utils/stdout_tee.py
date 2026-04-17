"""stdout / stderr -> project-root `stdout.txt` tee.

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

终端 vs 文件 的差异处理
----------------------
- **终端**：拿到的是**原始数据**（保留 tqdm 的 ``\\r`` 动画、loguru 的 ANSI 色）。
- **文件**：做三件事让内容肉眼友好、便于 grep：
  1. 剥离 ANSI escape 颜色 / 控制序列（``\\x1b[...m``、``\\x1b[K`` 等），
     否则 loguru ``colorize=True`` 会把 ``[32m...[0m`` 这些色码写进文件变成乱码。
  2. 丢弃 tqdm 进度条刷新帧（用 ``it/s`` / ``s/it`` 速率标记作为 sentinel），
     文件里不需要每帧 bar，否则又丑又膨胀。
  3. 剩余内容里 ``\\r`` 替换成 ``\\n``，避免 carriage return 让编辑器错乱。
- 追加模式（默认）下每次进程启动会插入一个时间戳 banner 便于区分多次运行。
- 文件用行缓冲（``buffering=1``）打开，进程崩溃时也能保留大部分已写内容。
"""
from __future__ import annotations

import atexit
import datetime as _dt
import re
import sys
from pathlib import Path
from typing import Optional, TextIO

# 项目根目录 = 本文件所在 utils/ 的父目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ANSI escape：覆盖常见 SGR 颜色（`m`）和其它 CSI 控制符（光标移动、清行等）。
# 规则：ESC `[` + 可选参数字节 + 可选中间字节 + 终止字节。
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

# tqdm 进度条 sentinel —— bar 里一定会出现 `it/s` 或 `s/it` 速率字样，
# 而 loguru/print 的正常输出不会，借此判别"这一块是不是 tqdm 帧"。
_TQDM_SENTINELS = ("it/s", "s/it")


def _sanitize_for_file(data: str) -> Optional[str]:
    """把一次 write 的原始数据转成适合写进文件的版本。

    Returns:
        ``None`` 表示整块都应该被丢弃（例如 tqdm 进度条刷新帧）；
        否则返回剥掉 ANSI、``\\r`` 换成 ``\\n`` 之后的字符串。
    """
    if not data:
        return data
    if any(s in data for s in _TQDM_SENTINELS):
        return None
    cleaned = _ANSI_ESCAPE_RE.sub("", data)
    cleaned = cleaned.replace("\r", "\n")
    return cleaned


class _TeeStream:
    """把写入分发到 terminal + file 的伪文件对象。"""

    def __init__(self, terminal: TextIO, file: TextIO):
        self.terminal = terminal
        self.file = file

    def write(self, data: str) -> int:
        # 终端原样输出（保留颜色 + tqdm 动画体验）
        self.terminal.write(data)
        # 文件只保留人类可读的版本
        cleaned = _sanitize_for_file(data)
        if cleaned:
            self.file.write(cleaned)
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
