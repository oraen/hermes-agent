"""Hermes Agent 全局常量与路径工具模块。

本模块是项目依赖链的最底层，无任何项目内部依赖，可被任何模块安全导入。
提供 HERMES_HOME 目录管理、环境检测、网络配置等核心功能。
"""

import os
from pathlib import Path


def get_hermes_home() -> Path:
    """获取 Hermes 主目录路径（默认：~/.hermes）。
    
    功能概括：
        返回 Hermes Agent 的主数据目录路径，支持通过环境变量 HERMES_HOME 自定义。
        这是项目中所有路径解析的唯一真实来源（single source of truth）。
    
    参数：
        无
    
    返回值：
        Path: Hermes 主目录的绝对路径。优先使用 HERMES_HOME 环境变量，
              若未设置则返回 ~/.hermes。
    
    主要用于：
        - 所有需要访问 Hermes 配置、会话、技能等数据的模块
        - Profile 多实例支持的基础（每个 profile 有独立的 HERMES_HOME）
        - 被项目中 119+ 个文件引用
    """
    return Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))


def get_default_hermes_root() -> Path:
    """获取 Hermes 根目录，用于 Profile 级别的操作。
    
    功能概括：
        返回 Hermes 的根目录路径，用于 profile list 等需要查看所有 profile 的操作。
        与 get_hermes_home() 不同，此函数在 profile 模式下返回的是 profile 的父目录。
    
    解析逻辑：
        1. 标准部署：返回 ~/.hermes
        2. Docker/自定义部署：如果 HERMES_HOME 指向 ~/.hermes 之外（如 /opt/data），
           直接返回 HERMES_HOME
        3. Profile 模式：如果 HERMES_HOME 是 <root>/profiles/<name>，返回 <root>
           这样 `hermes profile list` 可以看到所有 profile
    
    参数：
        无
    
    返回值：
        Path: Hermes 根目录的绝对路径
    
    主要用于：
        - Profile 管理命令（profile list、profile create 等）
        - 需要在多 profile 间切换或查看所有 profile 的场景
    """
    native_home = Path.home() / ".hermes"
    env_home = os.environ.get("HERMES_HOME", "")
    if not env_home:
        return native_home
    env_path = Path(env_home)
    try:
        env_path.resolve().relative_to(native_home.resolve())
        # HERMES_HOME is under ~/.hermes (normal or profile mode)
        return native_home
    except ValueError:
        pass

    # Docker / custom deployment.
    # Check if this is a profile path: <root>/profiles/<name>
    # If the immediate parent dir is named "profiles", the root is
    # the grandparent — this covers Docker profiles correctly.
    if env_path.parent.name == "profiles":
        return env_path.parent.parent

    # Not a profile path — HERMES_HOME itself is the root
    return env_path


def get_optional_skills_dir(default: Path | None = None) -> Path:
    """获取可选技能（optional-skills）目录路径。
    
    功能概括：
        返回社区/第三方技能的存储目录，支持通过环境变量 HERMES_OPTIONAL_SKILLS 覆盖。
        打包安装时可能将 optional-skills 放在 Python 包树之外。
    
    参数：
        default: 默认路径，当环境变量未设置时使用
    
    返回值：
        Path: optional-skills 目录的绝对路径
    
    优先级：
        1. HERMES_OPTIONAL_SKILLS 环境变量
        2. default 参数
        3. {HERMES_HOME}/optional-skills
    
    主要用于：
        - 加载社区贡献的技能包
        - 第三方技能源的扫描
    """
    override = os.getenv("HERMES_OPTIONAL_SKILLS", "").strip()
    if override:
        return Path(override)
    if default is not None:
        return default
    return get_hermes_home() / "optional-skills"


def get_hermes_dir(new_subpath: str, old_name: str) -> Path:
    """解析 Hermes 子目录，支持向后兼容。
    
    功能概括：
        智能选择目录路径：优先使用新的路径结构，但如果旧路径已存在于磁盘上，
        则继续使用旧路径，无需迁移。
    
    参数：
        new_subpath: 新路径，相对于 HERMES_HOME（如 "cache/images"）
        old_name: 旧路径，相对于 HERMES_HOME（如 "image_cache"）
    
    返回值：
        Path: 绝对路径。如果旧路径存在于磁盘上则返回旧路径，否则返回新路径
    
    主要用于：
        - 目录结构重构时的平滑过渡
        - 避免强制用户迁移已有数据
        - 图像缓存、浏览器数据等目录的兼容性处理
    """
    home = get_hermes_home()
    old_path = home / old_name
    if old_path.exists():
        return old_path
    return home / new_subpath


def display_hermes_home() -> str:
    """返回用户友好的 HERMES_HOME 显示字符串。
    
    功能概括：
        返回适合在用户界面显示的路径字符串，使用 ~/ 简写格式提高可读性。
        与 get_hermes_home() 的区别：此函数用于打印/日志消息，
        get_hermes_home() 用于代码中的实际路径操作。
    
    参数：
        无
    
    返回值：
        str: 用户友好的路径字符串
        - 默认模式：~/.hermes
        - Profile 模式：~/.hermes/profiles/coder
        - 自定义路径：/opt/hermes-custom
    
    主要用于：
        - 启动横幅（banner）显示
        - 配置保存提示消息
        - 所有面向用户的路径显示场景
    
    注意：
        需要实际 Path 对象时使用 get_hermes_home()，不要用此函数
    """
    home = get_hermes_home()
    try:
        return "~/" + str(home.relative_to(Path.home()))
    except ValueError:
        return str(home)


def get_subprocess_home() -> str | None:
    """为子进程返回独立的 HOME 目录路径，或返回 None。
    
    功能概括：
        当 {HERMES_HOME}/home/ 目录存在时，子进程应使用此目录作为 HOME 环境变量，
        使系统工具（git、ssh、gh、npm 等）将配置写入 Hermes 数据目录，
        而不是操作系统的 /root 或 ~/。
    
    参数：
        无
    
    返回值：
        str | None: 如果 {HERMES_HOME}/home/ 存在则返回其路径，否则返回 None
    
    主要用途：
        - Docker 持久化：工具配置保存在持久卷内
        - Profile 隔离：每个 profile 拥有独立的 git 身份、SSH 密钥、gh token 等
    
    重要说明：
        - Python 进程自身的 os.environ["HOME"] 和 Path.home() 永远不会被修改
        - 只有子进程环境应该注入此值
        - 基于目录激活：如果 home/ 子目录不存在，返回 None，行为不变
    """
    hermes_home = os.getenv("HERMES_HOME")
    if not hermes_home:
        return None
    profile_home = os.path.join(hermes_home, "home")
    if os.path.isdir(profile_home):
        return profile_home
    return None


VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")


def parse_reasoning_effort(effort: str) -> dict | None:
    """解析推理努力级别为配置字典。
    
    功能概括：
        将用户输入的推理努力级别字符串转换为模型 API 可用的配置字典。
    
    参数：
        effort: 推理努力级别字符串，有效值包括：
                - "none": 禁用推理
                - "minimal": 最小推理
                - "low": 低推理
                - "medium": 中等推理
                - "high": 高推理
                - "xhigh": 极高推理
    
    返回值：
        dict | None:
            - None: 输入为空或无法识别（调用方使用默认值）
            - {"enabled": False}: 输入为 "none"
            - {"enabled": True, "effort": <级别>}: 有效级别
    
    主要用于：
        - 解析用户配置中的 reasoning_effort 设置
        - 传递给支持推理的模型 API（如 OpenAI o1/o3）
    """
    if not effort or not effort.strip():
        return None
    effort = effort.strip().lower()
    if effort == "none":
        return {"enabled": False}
    if effort in VALID_REASONING_EFFORTS:
        return {"enabled": True, "effort": effort}
    return None


def is_termux() -> bool:
    """检测是否运行在 Termux（Android）环境中。
    
    功能概括：
        通过检查 TERMUX_VERSION 环境变量或 Termux 特有的 PREFIX 路径，
        判断当前是否在 Android 的 Termux 终端中运行。
    
    参数：
        无
    
    返回值：
        bool: 如果在 Termux 环境中返回 True，否则返回 False
    
    主要用于：
        - 针对 Android 环境的特殊处理（如依赖安装、路径调整）
        - Termux 有独立的包管理器和文件系统结构
    """
    prefix = os.getenv("PREFIX", "")
    return bool(os.getenv("TERMUX_VERSION") or "com.termux/files/usr" in prefix)


_wsl_detected: bool | None = None


def is_wsl() -> bool:
    """检测是否运行在 WSL（Windows Subsystem for Linux）中。
    
    功能概括：
        通过读取 /proc/version 文件并检查 microsoft 标记来判断是否在 WSL 中运行。
        WSL1 和 WSL2 都会注入此标记。结果在进程生命周期内缓存。
    
    参数：
        无
    
    返回值：
        bool: 如果在 WSL 环境中返回 True，否则返回 False
    
    主要用于：
        - Windows 开发环境的特殊处理
        - 文件系统路径转换（WSL 的 /mnt/c/ 等）
        - 剪贴板、GUI 工具等平台特定功能
    """
    global _wsl_detected
    if _wsl_detected is not None:
        return _wsl_detected
    try:
        with open("/proc/version", "r") as f:
            _wsl_detected = "microsoft" in f.read().lower()
    except Exception:
        _wsl_detected = False
    return _wsl_detected


_container_detected: bool | None = None


def is_container() -> bool:
    """检测是否运行在 Docker/Podman 容器中。
    
    功能概括：
        通过检查多个容器标记来判断是否在容器中运行：
        1. /.dockerenv 文件（Docker 标记）
        2. /run/.containerenv 文件（Podman 标记）
        3. /proc/1/cgroup 文件中的容器运行时标记
        结果在进程生命周期内缓存。
    
    参数：
        无
    
    返回值：
        bool: 如果在容器中返回 True，否则返回 False
    
    主要用于：
        - 容器环境的特殊配置（如网络、权限）
        - Docker 部署时的路径和行为调整
        - 避免在容器中尝试不支持的操作
    """
    global _container_detected
    if _container_detected is not None:
        return _container_detected
    if os.path.exists("/.dockerenv"):
        _container_detected = True
        return True
    if os.path.exists("/run/.containerenv"):
        _container_detected = True
        return True
    try:
        with open("/proc/1/cgroup", "r") as f:
            cgroup = f.read()
            if "docker" in cgroup or "podman" in cgroup or "/lxc/" in cgroup:
                _container_detected = True
                return True
    except OSError:
        pass
    _container_detected = False
    return False


# ─── Well-Known Paths ─────────────────────────────────────────────────────────


def get_config_path() -> Path:
    """返回 HERMES_HOME 下 config.yaml 的路径。
    
    功能概括：
        提供用户配置文件的标准路径，替代在 7+ 个文件中重复出现的
        `get_hermes_home() / "config.yaml"` 模式。
    
    参数：
        无
    
    返回值：
        Path: config.yaml 的绝对路径
    
    主要用于：
        - 读取/写入用户配置
        - skill_utils.py、hermes_logging.py、hermes_time.py 等模块
    """
    return get_hermes_home() / "config.yaml"


def get_skills_dir() -> Path:
    """返回 HERMES_HOME 下 skills 目录的路径。
    
    功能概括：
        提供用户技能存储的标准路径。
    
    参数：
        无
    
    返回值：
        Path: skills 目录的绝对路径
    
    主要用于：
        - 技能安装、加载、管理
        - agent/skill_commands.py 扫描技能
    """
    return get_hermes_home() / "skills"



def get_env_path() -> Path:
    """返回 HERMES_HOME 下 .env 文件的路径。
    
    功能概括：
        提供环境变量存储文件的标准路径，用于保存 API 密钥等敏感配置。
    
    参数：
        无
    
    返回值：
        Path: .env 文件的绝对路径
    
    主要用于：
        - 加载/保存 API 密钥
        - hermes_cli/env_loader.py
        - hermes_cli/auth.py 凭证解析
    """
    return get_hermes_home() / ".env"


# ─── Network Preferences ─────────────────────────────────────────────────────


def apply_ipv4_preference(force: bool = False) -> None:
    """通过 monkey-patch socket.getaddrinfo 强制优先使用 IPv4 连接。
    
    功能概括：
        在 IPv6 配置损坏或不可达的服务器上，Python 默认先尝试 AAAA 记录（IPv6），
        会在回退到 IPv4 前等待完整的 TCP 超时时间。这会影响 httpx、requests、
        urllib、OpenAI SDK 等所有使用 socket.getaddrinfo 的库。
        
        当 force=True 时， patch getaddrinfo 使 family=AF_UNSPEC（默认）的调用
        解析为 AF_INET（IPv4），完全跳过 IPv6。如果没有 A 记录，则回退到原始
        未过滤的解析，以便纯 IPv6 主机仍能工作。
    
    参数：
        force: 是否强制启用 IPv4 优先。默认 False
    
    返回值：
        无
    
    主要用于：
        - 解决服务器上 IPv6 不可达导致的连接超时
        - 在 config.yaml 中设置 network.force_ipv4: true 时启用
    
    安全性：
        - 可安全多次调用，只会 patch 一次
        - 通过检查 _hermes_ipv4_patched 标记防止重复 patch
    """
    if not force:
        return

    import socket

    # Guard against double-patching
    if getattr(socket.getaddrinfo, "_hermes_ipv4_patched", False):
        return

    _original_getaddrinfo = socket.getaddrinfo

    def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if family == 0:  # AF_UNSPEC — caller didn't request a specific family
            try:
                return _original_getaddrinfo(
                    host, port, socket.AF_INET, type, proto, flags
                )
            except socket.gaierror:
                # No A record — fall back to full resolution (pure-IPv6 hosts)
                return _original_getaddrinfo(host, port, family, type, proto, flags)
        return _original_getaddrinfo(host, port, family, type, proto, flags)

    _ipv4_getaddrinfo._hermes_ipv4_patched = True  # type: ignore[attr-defined]
    socket.getaddrinfo = _ipv4_getaddrinfo  # type: ignore[assignment]


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"

AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
