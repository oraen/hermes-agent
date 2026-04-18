"""Hermes Agent 集中式日志系统。

提供单一的 ``setup_logging()`` 入口点，CLI 和网关在启动路径早期调用。
所有日志文件位于 ``~/.hermes/logs/``（通过 ``get_hermes_home()`` 支持 profile）。

功能特性:
- 多日志文件分离（agent.log、errors.log、gateway.log）
- RotatingFileHandler 自动轮转，防止日志文件过大
- RedactingFormatter 自动脱敏，确保密钥不写入磁盘
- 组件分离：gateway.log 只接收 gateway.* 日志
- 会话上下文：每条日志包含 [session_id] 标签，便于过滤和关联
- 抑制第三方噪声日志（openai、httpx、urllib3 等）

日志文件:
    agent.log   — INFO+ 级别，所有智能体/工具/会话活动（主日志）
    errors.log  — WARNING+ 级别，仅错误和警告（快速排查）
    gateway.log — INFO+ 级别，仅网关事件（mode="gateway" 时创建）

使用场景:
1. CLI 模式：记录智能体活动、工具执行、会话管理
2. 网关模式：额外记录网关事件（平台适配器、消息传递等）
3. 调试模式：--verbose 参数启用 DEBUG 级别控制台输出
4. 会话追踪：通过 session_id 标签关联同一对话的日志
5. 日志搜索：hermes logs 命令支持按会话、组件过滤

示例:
    from hermes_logging import setup_logging, set_session_context
    
    # 启动时初始化
    log_dir = setup_logging(mode="cli")
    
    # 对话开始时设置会话上下文
    set_session_context("session-id-123")
    
    # 所有日志自动包含 [session-id-123] 标签
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Starting conversation")  # 输出: ... [session-id-123] Starting conversation
"""

import logging
import os
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Sequence

from hermes_constants import get_config_path, get_hermes_home

# Sentinel to track whether setup_logging() has already run.  The function
# is idempotent — calling it twice is safe but the second call is a no-op
# unless ``force=True``.
_logging_initialized = False

# Thread-local storage for per-conversation session context.
_session_context = threading.local()

# Default log format — includes timestamp, level, optional session tag,
# logger name, and message.  The ``%(session_tag)s`` field is guaranteed to
# exist on every LogRecord via _install_session_record_factory() below.
_LOG_FORMAT = "%(asctime)s %(levelname)s%(session_tag)s %(name)s: %(message)s"
_LOG_FORMAT_VERBOSE = "%(asctime)s - %(name)s - %(levelname)s%(session_tag)s - %(message)s"

# Third-party loggers that are noisy at DEBUG/INFO level.
_NOISY_LOGGERS = (
    "openai",
    "openai._base_client",
    "httpx",
    "httpcore",
    "asyncio",
    "hpack",
    "hpack.hpack",
    "grpc",
    "modal",
    "urllib3",
    "urllib3.connectionpool",
    "websockets",
    "charset_normalizer",
    "markdown_it",
)


# ---------------------------------------------------------------------------
# Public session context API
# ---------------------------------------------------------------------------

def set_session_context(session_id: str) -> None:
    """为当前线程设置会话 ID。

    功能:
    - 设置后，该线程上的所有后续日志记录都会包含 ``[session_id]``
    - 在 ``run_conversation()`` 开始时调用
    - 使用 thread-local 存储，不同线程互不干扰

    参数:
        session_id (str): 会话唯一标识符

    使用场景: 每次对话开始时调用，使日志可以按会话过滤和关联

    示例:
        from hermes_logging import set_session_context
        set_session_context("20260418_143052_a1b2c3")
    """
    _session_context.session_id = session_id


def clear_session_context() -> None:
    """清除当前线程的会话 ID。

    使用场景: 对话结束时调用，清除会话上下文
    """
    _session_context.session_id = None


# ---------------------------------------------------------------------------
# Record factory — injects session_tag into every LogRecord at creation
# ---------------------------------------------------------------------------

def _install_session_record_factory() -> None:
    """Replace the global LogRecord factory with one that adds ``session_tag``.

    Unlike a ``logging.Filter`` on a handler or logger, the record factory
    runs for EVERY record in the process — including records that propagate
    from child loggers and records handled by third-party handlers.  This
    guarantees ``%(session_tag)s`` is always available in format strings,
    eliminating the KeyError that would occur if a handler used our format
    without having a ``_SessionFilter`` attached.

    Idempotent — checks for a marker attribute to avoid double-wrapping if
    the module is reloaded.
    """
    current_factory = logging.getLogRecordFactory()
    if getattr(current_factory, "_hermes_session_injector", False):
        return  # already installed

    def _session_record_factory(*args, **kwargs):
        record = current_factory(*args, **kwargs)
        sid = getattr(_session_context, "session_id", None)
        record.session_tag = f" [{sid}]" if sid else ""  # type: ignore[attr-defined]
        return record

    _session_record_factory._hermes_session_injector = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(_session_record_factory)


# Install immediately on import — session_tag is available on all records
# from this point forward, even before setup_logging() is called.
_install_session_record_factory()


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

class _ComponentFilter(logging.Filter):
    """Only pass records whose logger name starts with one of *prefixes*.

    Used to route gateway-specific records to ``gateway.log`` while
    keeping ``agent.log`` as the catch-all.
    """

    def __init__(self, prefixes: Sequence[str]) -> None:
        super().__init__()
        self._prefixes = tuple(prefixes)

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(self._prefixes)


# Logger name prefixes that belong to each component.
# Used by _ComponentFilter and exposed for ``hermes logs --component``.
COMPONENT_PREFIXES = {
    "gateway": ("gateway",),
    "agent": ("agent", "run_agent", "model_tools", "batch_runner"),
    "tools": ("tools",),
    "cli": ("hermes_cli", "cli"),
    "cron": ("cron",),
}


# ---------------------------------------------------------------------------
# Main setup
# ---------------------------------------------------------------------------

def setup_logging(
    *,
    hermes_home: Optional[Path] = None,
    log_level: Optional[str] = None,
    max_size_mb: Optional[int] = None,
    backup_count: Optional[int] = None,
    mode: Optional[str] = None,
    force: bool = False,
) -> Path:
    """配置 Hermes 日志子系统。

    功能概括:
    - 创建日志目录和多个日志文件处理器
    - 配置日志轮转（RotatingFileHandler）
    - 应用脱敏格式化器（RedactingFormatter）
    - 抑制第三方噪声日志
    - 支持 CLI、网关、定时任务不同模式

    参数:
        hermes_home (Path): Hermes 主目录覆盖（默认使用 get_hermes_home()，支持 profile）
        log_level (str): agent.log 文件处理器的最低级别（默认："INFO" 或 config.yaml 中的 logging.level）
        max_size_mb (int): 每个日志文件的最大大小（MB），超过后轮转（默认：5 或 config.yaml 中的 logging.max_size_mb）
        backup_count (int): 保留的轮转备份文件数量（默认：3 或 config.yaml 中的 logging.backup_count）
        mode (str): 调用者上下文："cli"、"gateway"、"cron"
                   当为 "gateway" 时，会额外创建 gateway.log 文件，只接收网关组件记录
        force (bool): 即使已经调用过也重新设置

    返回值:
        Path: 日志文件写入的 logs/ 目录

    日志文件:
        - agent.log: INFO+ 级别，所有活动（主日志）
        - errors.log: WARNING+ 级别，仅错误和警告
        - gateway.log: INFO+ 级别，仅网关事件（mode="gateway" 时）

    使用场景:
        1. CLI 启动: setup_logging(mode="cli")
        2. 网关启动: setup_logging(mode="gateway")
        3. 自定义配置: setup_logging(log_level="DEBUG", max_size_mb=10)
        4. 强制重新设置: setup_logging(force=True)

    示例:
        from hermes_logging import setup_logging
        
        # 基础用法
        log_dir = setup_logging(mode="cli")
        
        # 自定义配置
        log_dir = setup_logging(
            mode="gateway",
            log_level="DEBUG",
            max_size_mb=10,
            backup_count=5
        )
    """
    global _logging_initialized
    if _logging_initialized and not force:
        home = hermes_home or get_hermes_home()
        return home / "logs"

    home = hermes_home or get_hermes_home()
    log_dir = home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Read config defaults (best-effort — config may not be loaded yet).
    cfg_level, cfg_max_size, cfg_backup = _read_logging_config()

    level_name = (log_level or cfg_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    max_bytes = (max_size_mb or cfg_max_size or 5) * 1024 * 1024
    backups = backup_count or cfg_backup or 3

    # Lazy import to avoid circular dependency at module load time.
    from agent.redact import RedactingFormatter

    root = logging.getLogger()

    # --- agent.log (INFO+) — the main activity log -------------------------
    _add_rotating_handler(
        root,
        log_dir / "agent.log",
        level=level,
        max_bytes=max_bytes,
        backup_count=backups,
        formatter=RedactingFormatter(_LOG_FORMAT),
    )

    # --- errors.log (WARNING+) — quick triage log --------------------------
    _add_rotating_handler(
        root,
        log_dir / "errors.log",
        level=logging.WARNING,
        max_bytes=2 * 1024 * 1024,
        backup_count=2,
        formatter=RedactingFormatter(_LOG_FORMAT),
    )

    # --- gateway.log (INFO+, gateway component only) ------------------------
    if mode == "gateway":
        _add_rotating_handler(
            root,
            log_dir / "gateway.log",
            level=logging.INFO,
            max_bytes=5 * 1024 * 1024,
            backup_count=3,
            formatter=RedactingFormatter(_LOG_FORMAT),
            log_filter=_ComponentFilter(COMPONENT_PREFIXES["gateway"]),
        )

    # Ensure root logger level is low enough for the handlers to fire.
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)

    # Suppress noisy third-party loggers.
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _logging_initialized = True
    return log_dir


def setup_verbose_logging() -> None:
    """为 ``--verbose`` / ``-v`` 模式启用 DEBUG 级别控制台日志。

    功能:
    - 添加 StreamHandler 到根日志器
    - 设置根日志器级别为 DEBUG
    - 保持第三方库在 WARNING 级别以减少噪声
    - 避免重复添加处理器

    使用场景: AIAgent.__init__() 中 verbose_logging=True 时调用

    示例:
        from hermes_logging import setup_verbose_logging
        setup_verbose_logging()  # 启用详细日志
    """
    from agent.redact import RedactingFormatter

    root = logging.getLogger()

    # Avoid adding duplicate stream handlers.
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler):
            if getattr(h, "_hermes_verbose", False):
                return

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(RedactingFormatter(_LOG_FORMAT_VERBOSE, datefmt="%H:%M:%S"))
    handler._hermes_verbose = True  # type: ignore[attr-defined]
    root.addHandler(handler)

    # Lower root logger level so DEBUG records reach all handlers.
    if root.level > logging.DEBUG:
        root.setLevel(logging.DEBUG)

    # Keep third-party libraries at WARNING to reduce noise.
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    # rex-deploy at INFO for sandbox status.
    logging.getLogger("rex-deploy").setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _ManagedRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler that ensures group-writable perms in managed mode.

    In managed mode (NixOS), the stateDir uses setgid (2770) so new files
    inherit the hermes group. However, both _open() (initial creation) and
    doRollover() create files via open(), which uses the process umask —
    typically 0022, producing 0644. This subclass applies chmod 0660 after
    both operations so the gateway and interactive users can share log files.
    """

    def __init__(self, *args, **kwargs):
        from hermes_cli.config import is_managed
        self._managed = is_managed()
        super().__init__(*args, **kwargs)

    def _chmod_if_managed(self):
        if self._managed:
            try:
                os.chmod(self.baseFilename, 0o660)
            except OSError:
                pass

    def _open(self):
        stream = super()._open()
        self._chmod_if_managed()
        return stream

    def doRollover(self):
        super().doRollover()
        self._chmod_if_managed()


def _add_rotating_handler(
    logger: logging.Logger,
    path: Path,
    *,
    level: int,
    max_bytes: int,
    backup_count: int,
    formatter: logging.Formatter,
    log_filter: Optional[logging.Filter] = None,
) -> None:
    """Add a ``RotatingFileHandler`` to *logger*, skipping if one already
    exists for the same resolved file path (idempotent).

    Parameters
    ----------
    log_filter
        Optional filter to attach to the handler (e.g. ``_ComponentFilter``
        for gateway.log).
    """
    resolved = path.resolve()
    for existing in logger.handlers:
        if (
            isinstance(existing, RotatingFileHandler)
            and Path(getattr(existing, "baseFilename", "")).resolve() == resolved
        ):
            return  # already attached

    path.parent.mkdir(parents=True, exist_ok=True)
    handler = _ManagedRotatingFileHandler(
        str(path), maxBytes=max_bytes, backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    if log_filter is not None:
        handler.addFilter(log_filter)
    logger.addHandler(handler)


def _read_logging_config():
    """Best-effort read of ``logging.*`` from config.yaml.

    Returns ``(level, max_size_mb, backup_count)`` — any may be ``None``.
    """
    try:
        import yaml
        config_path = get_config_path()
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            log_cfg = cfg.get("logging", {})
            if isinstance(log_cfg, dict):
                return (
                    log_cfg.get("level"),
                    log_cfg.get("max_size_mb"),
                    log_cfg.get("backup_count"),
                )
    except Exception:
        pass
    return (None, None, None)
