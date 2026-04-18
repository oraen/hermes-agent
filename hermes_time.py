"""Hermes Agent 时区感知时钟模块。

提供单一的 ``now()`` 助手函数，返回基于用户配置的 IANA 时区
（如 ``Asia/Shanghai``）的时区感知 datetime 对象。

解析顺序（优先级从高到低）：
  1. ``HERMES_TIMEZONE`` 环境变量
  2. ``~/.hermes/config.yaml`` 中的 ``timezone`` 键
  3. 回退到服务器本地时间（``datetime.now().astimezone()``）

无效的时区值会记录警告并安全回退 —— Hermes 永远不会因为错误的时区字符串而崩溃。
"""

import logging
import os
from datetime import datetime
from hermes_constants import get_config_path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Python 3.8 fallback (shouldn't be needed — Hermes requires 3.9+)
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

# Cached state — resolved once, reused on every call.
# Call reset_cache() to force re-resolution (e.g. after config changes).
_cached_tz: Optional[ZoneInfo] = None
_cached_tz_name: Optional[str] = None
_cache_resolved: bool = False


def _resolve_timezone_name() -> str:
    """读取配置的 IANA 时区字符串（或返回空字符串）。
    
    功能概括：
        按照优先级顺序解析时区配置：
        1. HERMES_TIMEZONE 环境变量（最高优先级）
        2. config.yaml 中的 timezone 键
        3. 返回空字符串（表示使用服务器本地时间）
        
        此函数在回退到 config.yaml 时会进行文件 I/O，
        因此调用方应缓存结果而不是在每次 ``now()`` 时调用。
    
    参数：
        无
    
    返回值：
        str: IANA 时区名称（如 "Asia/Shanghai"），如果未配置则返回空字符串
    
    主要用于：
        - _get_zoneinfo() 的前置步骤
        - 时区配置的集中解析
    """
    # 1. Environment variable (highest priority — set by Supervisor, etc.)
    tz_env = os.getenv("HERMES_TIMEZONE", "").strip()
    if tz_env:
        return tz_env

    # 2. config.yaml ``timezone`` key
    try:
        import yaml
        config_path = get_config_path()
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            tz_cfg = cfg.get("timezone", "")
            if isinstance(tz_cfg, str) and tz_cfg.strip():
                return tz_cfg.strip()
    except Exception:
        pass

    return ""


def _get_zoneinfo(name: str) -> Optional[ZoneInfo]:
    """验证并返回 ZoneInfo 对象，如果无效则返回 None。
    
    功能概括：
        尝试创建 ZoneInfo 对象，如果时区名称无效则记录警告并返回 None。
    
    参数：
        name: IANA 时区名称（如 "Asia/Shanghai"、"America/New_York"）
    
    返回值：
        ZoneInfo | None: 有效的时区对象，或 None（表示使用服务器本地时间）
    
    主要用于：
        - get_timezone() 的内部辅助函数
        - 时区验证和错误处理
    """
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except (KeyError, Exception) as exc:
        logger.warning(
            "Invalid timezone '%s': %s. Falling back to server local time.",
            name, exc,
        )
        return None


def get_timezone() -> Optional[ZoneInfo]:
    """返回用户配置的 ZoneInfo 时区对象，或 None（表示使用服务器本地时间）。
    
    功能概括：
        获取并缓存用户配置的时区对象。首次调用时解析配置并缓存，
        后续调用直接返回缓存结果。配置更改后需调用 reset_cache() 强制重新解析。
    
    参数：
        无
    
    返回值：
        ZoneInfo | None: 用户配置的时区对象，如果未配置或配置无效则返回 None
    
    主要用于：
        - now() 函数的时区获取
        - 需要时区信息的任何模块
    
    注意：
        - 结果会被缓存，提高性能
        - 配置更改后需调用 reset_cache() 刷新缓存
    """
    global _cached_tz, _cached_tz_name, _cache_resolved
    if not _cache_resolved:
        _cached_tz_name = _resolve_timezone_name()
        _cached_tz = _get_zoneinfo(_cached_tz_name)
        _cache_resolved = True
    return _cached_tz


def now() -> datetime:
    """返回当前时间，作为时区感知的 datetime 对象。
    
    功能概括：
        Hermes Agent 的统一时间获取函数，确保所有时间戳都带有时区信息。
        如果配置了有效时区，返回该时区的墙钟时间；
        否则返回服务器本地时间（仍然带时区信息）。
    
    参数：
        无
    
    返回值：
        datetime: 当前时间的时区感知 datetime 对象
    
    主要用于：
        - 会话时间戳记录
        - 日志时间标记
        - Cron 任务调度
        - 任何需要当前时间的场景
    
    示例：
        >>> from hermes_time import now
        >>> current_time = now()
        >>> print(current_time)  # 2024-01-15 10:30:00+08:00
    """
    tz = get_timezone()
    if tz is not None:
        return datetime.now(tz)
    # No timezone configured — use server-local (still tz-aware)
    return datetime.now().astimezone()


