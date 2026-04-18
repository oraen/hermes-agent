"""Hermes Agent 共享工具函数模块。

提供全局通用的辅助函数，包括：
- 布尔值解析（truthy/falsy 判断）
- 原子文件写入（JSON/YAML）
- 环境变量解析助手
- 安全 JSON解析

本模块被 agent/、tools/、gateway/ 等广泛引用。
"""

import json
import logging
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Union

import yaml

logger = logging.getLogger(__name__)


TRUTHY_STRINGS = frozenset({"1", "true", "yes", "on"})


def is_truthy_value(value: Any, default: bool = False) -> bool:
    """将各种类型的值转换为布尔值，使用项目共享的 truthy 字符串集合。
    
    功能概括：
        统一处理不同类型值的布尔转换，特别对字符串使用预定义的 truthy 集合
        {"1", "true", "yes", "on"}，确保项目中布尔值解析的一致性。
    
    参数：
        value: 要转换的值，可以是任何类型
               - None: 返回 default
               - bool: 直接返回
               - str: 转小写去空格后检查是否在 TRUTHY_STRINGS 中
               - 其他类型: 使用 bool() 转换
        default: 当 value 为 None 时的默认返回值，默认 False
    
    返回值：
        bool: 转换后的布尔值
    
    主要用于：
        - 解析配置文件中的布尔选项
        - 环境变量解析（env_var_enabled、env_bool）
        - 用户输入验证
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in TRUTHY_STRINGS
    return bool(value)


def env_var_enabled(name: str, default: str = "") -> bool:
    """检查环境变量是否设置为 truthy 值。
    
    功能概括：
        读取环境变量并使用 is_truthy_value() 判断其是否为启用状态。
    
    参数：
        name: 环境变量名称（如 "HERMES_DEBUG"）
        default: 环境变量未设置时的默认值，默认空字符串
    
    返回值：
        bool: 如果环境变量存在且为 truthy 值则返回 True，否则返回 False
    
    主要用于：
        - 检测功能开关（如调试模式、实验特性）
        - 条件性功能启用
    
    示例：
        env_var_enabled("HERMES_DEBUG")  # 检查是否启用调试
    """
    return is_truthy_value(os.getenv(name, default), default=False)


def _preserve_file_mode(path: Path) -> "int | None":
    """捕获文件 *path* 的权限位（如果文件存在），否则返回 None。
    
    功能概括：
        在进行原子文件替换前保存原文件的权限模式，
        以便在替换后恢复原有权限。
    
    参数：
        path: 要检查的文件路径
    
    返回值：
        int | None: 文件的权限模式（如 0o644），如果文件不存在则返回 None
    
    主要用于：
        - atomic_json_write() 和 atomic_yaml_write() 的内部辅助函数
    """
    try:
        return stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    except OSError:
        return None


def _restore_file_mode(path: Path, mode: "int | None") -> None:
    """在原子替换后恢复文件的原始权限模式。
    
    功能概括：
        tempfile.mkstemp 创建的文件权限为 0o600（仅所有者可读写）。
        os.replace 将临时文件替换到目标位置后，目标文件会继承这些限制性权限，
        这会破坏依赖更宽松权限的 Docker/NAS 卷挂载。此函数在 os.replace 后
        立即调用以恢复原始权限。
    
    参数：
        path: 目标文件路径
        mode: 要恢复的权限模式，如果为 None 则不执行任何操作
    
    返回值：
        无
    
    主要用于：
        - atomic_json_write() 和 atomic_yaml_write() 的内部辅助函数
        - 确保配置文件在写入后保持正确的访问权限
    """
    if mode is None:
        return
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def atomic_json_write(
    path: Union[str, Path],
    data: Any,
    *,
    indent: int = 2,
    **dump_kwargs: Any,
) -> None:
    """以原子方式将 JSON 数据写入文件。
    
    功能概括：
        使用临时文件 + fsync + os.replace 确保目标文件永远不会处于
        部分写入状态。如果进程在写入中途崩溃，文件的先前版本保持完整。
        这对于配置文件、会话数据等关键数据非常重要。
    
    参数：
        path: 目标文件路径（将被创建或覆盖）
        data: 要写入的 JSON 可序列化数据
        indent: JSON 缩进空格数，默认 2
        **dump_kwargs: 传递给 json.dump() 的额外参数
                      （如 default=str 用于非原生类型）
    
    返回值：
        无
    
    主要用于：
        - 写入配置文件（config.yaml 的 JSON 版本）
        - 保存会话状态、认证信息
        - 任何需要保证写入原子性的场景
    
    安全性：
        - 捕获 BaseException（包括 KeyboardInterrupt/SystemExit）
          以确保临时文件被清理
        - 保留原文件的权限模式
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    original_mode = _preserve_file_mode(path)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=indent,
                ensure_ascii=False,
                **dump_kwargs,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        _restore_file_mode(path, original_mode)
    except BaseException:
        # Intentionally catch BaseException so temp-file cleanup still runs for
        # KeyboardInterrupt/SystemExit before re-raising the original signal.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_yaml_write(
    path: Union[str, Path],
    data: Any,
    *,
    default_flow_style: bool = False,
    sort_keys: bool = False,
    extra_content: str | None = None,
) -> None:
    """以原子方式将 YAML 数据写入文件。
    
    功能概括：
        使用临时文件 + fsync + os.replace 确保目标文件永远不会处于
        部分写入状态。如果进程在写入中途崩溃，文件的先前版本保持完整。
        与 atomic_json_write() 类似，但针对 YAML 格式。
    
    参数：
        path: 目标文件路径（将被创建或覆盖）
        data: 要写入的 YAML 可序列化数据
        default_flow_style: YAML 流式样式，默认 False（使用块样式）
        sort_keys: 是否对字典键排序，默认 False（保持插入顺序）
        extra_content: 可选字符串，追加到 YAML dump 之后
                      （例如：供用户参考的注释部分）
    
    返回值：
        无
    
    主要用于：
        - 写入主配置文件 config.yaml
        - 保存技能配置、工具配置
        - 任何需要人类可读格式的配置写入
    
    安全性：
        - 与 atomic_json_write() 相同的原子性保证
        - 异常情况下自动清理临时文件
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    original_mode = _preserve_file_mode(path)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=default_flow_style, sort_keys=sort_keys)
            if extra_content:
                f.write(extra_content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        _restore_file_mode(path, original_mode)
    except BaseException:
        # Match atomic_json_write: cleanup must also happen for process-level
        # interruptions before we re-raise them.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─── JSON Helpers ─────────────────────────────────────────────────────────────


def safe_json_loads(text: str, default: Any = None) -> Any:
    """安全地解析 JSON 字符串，任何解析错误时返回 *default*。
    
    功能概括：
        替代在 display.py、anthropic_adapter.py、auxiliary_client.py
        等文件中重复出现的 `try: json.loads(x) except ...` 模式。
    
    参数：
        text: 要解析的 JSON 字符串
        default: 解析失败时的默认返回值，默认 None
    
    返回值：
        Any: 解析后的 Python 对象，或 default 值
    
    主要用于：
        - 解析可能格式错误的 JSON 响应
        - 工具返回结果的解析
        - 配置文件片段的解析
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


# ─── Environment Variable Helpers ─────────────────────────────────────────────


def env_int(key: str, default: int = 0) -> int:
    """将环境变量读取为整数，带回退机制。
    
    功能概括：
        读取环境变量并尝试转换为整数，如果变量不存在或转换失败则返回默认值。
    
    参数：
        key: 环境变量名称
        default: 读取失败时的默认整数值，默认 0
    
    返回值：
        int: 环境变量的整数值，或 default
    
    主要用于：
        - 读取端口号、超时时间、重试次数等数值配置
        - 带默认值的数值环境变量解析
    """
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


def env_bool(key: str, default: bool = False) -> bool:
    """将环境变量读取为布尔值。
    
    功能概括：
        读取环境变量并使用 is_truthy_value() 转换为布尔值。
    
    参数：
        key: 环境变量名称
        default: 环境变量未设置时的默认布尔值，默认 False
    
    返回值：
        bool: 环境变量的布尔值，或 default
    
    主要用于：
        - 读取功能开关（如 HERMES_DEBUG、HERMES_QUIET）
        - 配置标志的布尔解析
    """
    return is_truthy_value(os.getenv(key, ""), default=default)
