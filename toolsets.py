#!/usr/bin/env python3
"""Hermes Agent 工具集（Toolsets）管理模块。

本模块提供灵活的工具集定义和管理系统，允许将工具分组以适配不同场景。
工具集可以包含独立工具，也可以组合其他工具集，形成层次化工具管理。

核心功能：
- 定义自定义工具集，包含特定工具组合
- 工具集组合：一个工具集可以包含其他工具集
- 内置常见场景的工具集（研究、开发、调试等）
- 支持动态工具集解析和插件扩展
- 支持所有消息平台的工具集配置

使用示例：
    from toolsets import get_toolset, resolve_toolset, get_all_toolsets
    
    # 获取特定工具集的定义
    tools = get_toolset("research")
    
    # 解析工具集，获取所有工具名称（包括组合的工具集）
    all_tools = resolve_toolset("full_stack")

主要用途：
    - 为不同平台（CLI、Telegram、Discord 等）定义不同的工具访问权限
    - 为不同场景（研究、开发、调试）提供预设工具组合
    - 支持用户自定义工具集
    - 被 model_tools.py、batch_runner.py 等模块广泛引用
"""

from typing import List, Dict, Any, Set, Optional


# CLI 和所有消息平台共享的核心工具列表。
# 修改此处可同时更新所有平台的工具配置。
_HERMES_CORE_TOOLS = [
    # Web
    "web_search", "web_extract",
    # Terminal + process management
    "terminal", "process",
    # File manipulation
    "read_file", "write_file", "patch", "search_files",
    # Vision + image generation
    "vision_analyze", "image_generate",
    # Skills
    "skills_list", "skill_view", "skill_manage",
    # Browser automation
    "browser_navigate", "browser_snapshot", "browser_click",
    "browser_type", "browser_scroll", "browser_back",
    "browser_press", "browser_get_images",
    "browser_vision", "browser_console",
    # Text-to-speech
    "text_to_speech",
    # Planning & memory
    "todo", "memory",
    # Session history search
    "session_search",
    # Clarifying questions
    "clarify",
    # Code execution + delegation
    "execute_code", "delegate_task",
    # Cronjob management
    "cronjob",
    # Cross-platform messaging (gated on gateway running via check_fn)
    "send_message",
    # Home Assistant smart home control (gated on HASS_TOKEN via check_fn)
    "ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service",
]


# 核心工具集定义
# 每个工具集可以包含独立工具或引用其他工具集
TOOLSETS = {
    # Basic toolsets - individual tool categories
    "web": {
        "description": "Web research and content extraction tools",
        "tools": ["web_search", "web_extract"],
        "includes": []  # No other toolsets included
    },
    
    "search": {
        "description": "Web search only (no content extraction/scraping)",
        "tools": ["web_search"],
        "includes": []
    },
    
    "vision": {
        "description": "Image analysis and vision tools",
        "tools": ["vision_analyze"],
        "includes": []
    },
    
    "image_gen": {
        "description": "Creative generation tools (images)",
        "tools": ["image_generate"],
        "includes": []
    },
    
    "terminal": {
        "description": "Terminal/command execution and process management tools",
        "tools": ["terminal", "process"],
        "includes": []
    },
    
    "moa": {
        "description": "Advanced reasoning and problem-solving tools",
        "tools": ["mixture_of_agents"],
        "includes": []
    },
    
    "skills": {
        "description": "Access, create, edit, and manage skill documents with specialized instructions and knowledge",
        "tools": ["skills_list", "skill_view", "skill_manage"],
        "includes": []
    },
    
    "browser": {
        "description": "Browser automation for web interaction (navigate, click, type, scroll, iframes, hold-click) with web search for finding URLs",
        "tools": [
            "browser_navigate", "browser_snapshot", "browser_click",
            "browser_type", "browser_scroll", "browser_back",
            "browser_press", "browser_get_images",
            "browser_vision", "browser_console", "web_search"
        ],
        "includes": []
    },
    
    "cronjob": {
        "description": "Cronjob management tool - create, list, update, pause, resume, remove, and trigger scheduled tasks",
        "tools": ["cronjob"],
        "includes": []
    },
    
    "messaging": {
        "description": "Cross-platform messaging: send messages to Telegram, Discord, Slack, SMS, etc.",
        "tools": ["send_message"],
        "includes": []
    },
    
    "rl": {
        "description": "RL training tools for running reinforcement learning on Tinker-Atropos",
        "tools": [
            "rl_list_environments", "rl_select_environment",
            "rl_get_current_config", "rl_edit_config",
            "rl_start_training", "rl_check_status",
            "rl_stop_training", "rl_get_results",
            "rl_list_runs", "rl_test_inference"
        ],
        "includes": []
    },
    
    "file": {
        "description": "File manipulation tools: read, write, patch (with fuzzy matching), and search (content + files)",
        "tools": ["read_file", "write_file", "patch", "search_files"],
        "includes": []
    },
    
    "tts": {
        "description": "Text-to-speech: convert text to audio with Edge TTS (free), ElevenLabs, OpenAI, or xAI",
        "tools": ["text_to_speech"],
        "includes": []
    },
    
    "todo": {
        "description": "Task planning and tracking for multi-step work",
        "tools": ["todo"],
        "includes": []
    },
    
    "memory": {
        "description": "Persistent memory across sessions (personal notes + user profile)",
        "tools": ["memory"],
        "includes": []
    },
    
    "session_search": {
        "description": "Search and recall past conversations with summarization",
        "tools": ["session_search"],
        "includes": []
    },
    
    "clarify": {
        "description": "Ask the user clarifying questions (multiple-choice or open-ended)",
        "tools": ["clarify"],
        "includes": []
    },
    
    "code_execution": {
        "description": "Run Python scripts that call tools programmatically (reduces LLM round trips)",
        "tools": ["execute_code"],
        "includes": []
    },
    
    "delegation": {
        "description": "Spawn subagents with isolated context for complex subtasks",
        "tools": ["delegate_task"],
        "includes": []
    },

    # "honcho" toolset removed — Honcho is now a memory provider plugin.
    # Tools are injected via MemoryManager, not the toolset system.

    "homeassistant": {
        "description": "Home Assistant smart home control and monitoring",
        "tools": ["ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service"],
        "includes": []
    },


    # Scenario-specific toolsets
    
    "debugging": {
        "description": "Debugging and troubleshooting toolkit",
        "tools": ["terminal", "process"],
        "includes": ["web", "file"]  # For searching error messages and solutions, and file operations
    },
    
    "safe": {
        "description": "Safe toolkit without terminal access",
        "tools": [],
        "includes": ["web", "vision", "image_gen"]
    },
    
    # ==========================================================================
    # Full Hermes toolsets (CLI + messaging platforms)
    #
    # All platforms share the same core tools (including send_message,
    # which is gated on gateway running via its check_fn).
    # ==========================================================================

    "hermes-acp": {
        "description": "Editor integration (VS Code, Zed, JetBrains) — coding-focused tools without messaging, audio, or clarify UI",
        "tools": [
            "web_search", "web_extract",
            "terminal", "process",
            "read_file", "write_file", "patch", "search_files",
            "vision_analyze",
            "skills_list", "skill_view", "skill_manage",
            "browser_navigate", "browser_snapshot", "browser_click",
            "browser_type", "browser_scroll", "browser_back",
            "browser_press", "browser_get_images",
            "browser_vision", "browser_console",
            "todo", "memory",
            "session_search",
            "execute_code", "delegate_task",
        ],
        "includes": []
    },

    "hermes-api-server": {
        "description": "OpenAI-compatible API server — full agent tools accessible via HTTP (no interactive UI tools like clarify or send_message)",
        "tools": [
            # Web
            "web_search", "web_extract",
            # Terminal + process management
            "terminal", "process",
            # File manipulation
            "read_file", "write_file", "patch", "search_files",
            # Vision + image generation
            "vision_analyze", "image_generate",
            # Skills
            "skills_list", "skill_view", "skill_manage",
            # Browser automation
            "browser_navigate", "browser_snapshot", "browser_click",
            "browser_type", "browser_scroll", "browser_back",
            "browser_press", "browser_get_images",
            "browser_vision", "browser_console",
            # Planning & memory
            "todo", "memory",
            # Session history search
            "session_search",
            # Code execution + delegation
            "execute_code", "delegate_task",
            # Cronjob management
            "cronjob",
            # Home Assistant smart home control (gated on HASS_TOKEN via check_fn)
            "ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service",

        ],
        "includes": []
    },
    
    "hermes-cli": {
        "description": "Full interactive CLI toolset - all default tools plus cronjob management",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },
    
    "hermes-telegram": {
        "description": "Telegram bot toolset - full access for personal use (terminal has safety checks)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },
    
    "hermes-discord": {
        "description": "Discord bot toolset - full access (terminal has safety checks via dangerous command approval)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },
    
    "hermes-whatsapp": {
        "description": "WhatsApp bot toolset - similar to Telegram (personal messaging, more trusted)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },
    
    "hermes-slack": {
        "description": "Slack bot toolset - full access for workspace use (terminal has safety checks)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },
    
    "hermes-signal": {
        "description": "Signal bot toolset - encrypted messaging platform (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-bluebubbles": {
        "description": "BlueBubbles iMessage bot toolset - Apple iMessage via local BlueBubbles server",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-homeassistant": {
        "description": "Home Assistant bot toolset - smart home event monitoring and control",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-email": {
        "description": "Email bot toolset - interact with Hermes via email (IMAP/SMTP)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-mattermost": {
        "description": "Mattermost bot toolset - self-hosted team messaging (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-matrix": {
        "description": "Matrix bot toolset - decentralized encrypted messaging (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-dingtalk": {
        "description": "DingTalk bot toolset - enterprise messaging platform (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-feishu": {
        "description": "Feishu/Lark bot toolset - enterprise messaging via Feishu/Lark (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-weixin": {
        "description": "Weixin bot toolset - personal WeChat messaging via iLink (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-qqbot": {
        "description": "QQBot toolset - QQ messaging via Official Bot API v2 (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-wecom": {
        "description": "WeCom bot toolset - enterprise WeChat messaging (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-wecom-callback": {
        "description": "WeCom callback toolset - enterprise self-built app messaging (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-sms": {
        "description": "SMS bot toolset - interact with Hermes via SMS (Twilio)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-webhook": {
        "description": "Webhook toolset - receive and process external webhook events",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-gateway": {
        "description": "Gateway toolset - union of all messaging platform tools",
        "tools": [],
        "includes": ["hermes-telegram", "hermes-discord", "hermes-whatsapp", "hermes-slack", "hermes-signal", "hermes-bluebubbles", "hermes-homeassistant", "hermes-email", "hermes-sms", "hermes-mattermost", "hermes-matrix", "hermes-dingtalk", "hermes-feishu", "hermes-wecom", "hermes-wecom-callback", "hermes-weixin", "hermes-qqbot", "hermes-webhook"]
    }
}



def get_toolset(name: str) -> Optional[Dict[str, Any]]:
    """按名称获取工具集定义。
    
    功能概括：
        获取指定工具集的完整定义，包括描述、工具列表和包含的其他工具集。
        支持静态定义的工具集和插件注册的工具集。
    
    参数：
        name: 工具集名称（如 "web"、"terminal"、"hermes-cli"）
    
    返回值：
        dict | None: 工具集定义字典，包含以下键：
            - description: 工具集描述
            - tools: 直接包含的工具列表
            - includes: 包含的其他工具集列表
            如果工具集不存在则返回 None
    
    主要用于：
        - resolve_toolset() 的前置步骤
        - 查询工具集元数据
        - 插件工具集的动态解析
    
    解析顺序：
        1. 查找静态 TOOLSETS 字典
        2. 查找插件注册的工具集
        3. 查找 MCP 服务器工具集别名
    """
    toolset = TOOLSETS.get(name)
    if toolset:
        return toolset

    try:
        from tools.registry import registry
    except Exception:
        return None

    registry_toolset = name
    description = f"Plugin toolset: {name}"
    alias_target = registry.get_toolset_alias_target(name)

    if name not in _get_plugin_toolset_names():
        registry_toolset = alias_target
        if not registry_toolset:
            return None
        description = f"MCP server '{name}' tools"
    else:
        reverse_aliases = {
            canonical: alias
            for alias, canonical in _get_registry_toolset_aliases().items()
            if alias not in TOOLSETS
        }
        alias = reverse_aliases.get(name)
        if alias:
            description = f"MCP server '{alias}' tools"

    return {
        "description": description,
        "tools": registry.get_tool_names_for_toolset(registry_toolset),
        "includes": [],
    }


def resolve_toolset(name: str, visited: Set[str] = None) -> List[str]:
    """递归解析工具集，获取所有工具名称。
    
    功能概括：
        递归解析工具集及其包含的所有子工具集，返回完整的工具名称列表。
        处理工具集组合、循环检测和菱形依赖（diamond dependencies）。
    
    参数：
        name: 要解析的工具集名称
              特殊值 "all" 或 "*" 表示解析所有工具集中的所有工具
        visited: 已访问的工具集集合，用于循环检测
                调用方通常不需要传入此参数
    
    返回值：
        List[str]: 排序后的工具名称列表
                  如果工具集不存在或检测到循环则返回空列表
    
    主要用于：
        - get_tool_definitions() 中解析启用的工具集
        - 获取某个场景下所有可用工具
        - 工具集依赖关系的完全展开
    
    特性：
        - 自动处理循环依赖（安全返回空列表）
        - 支持菱形依赖（多个路径引用同一工具集，只解析一次）
        - 特殊别名 "all" 和 "*" 自动包含未来新增的工具集
    """
    if visited is None:
        visited = set()
    
    # Special aliases that represent all tools across every toolset
    # This ensures future toolsets are automatically included without changes.
    if name in {"all", "*"}:
        all_tools: Set[str] = set()
        for toolset_name in get_toolset_names():
            # Use a fresh visited set per branch to avoid cross-branch contamination
            resolved = resolve_toolset(toolset_name, visited.copy())
            all_tools.update(resolved)
        return sorted(all_tools)

    # Check for cycles / already-resolved (diamond deps).
    # Silently return [] — either this is a diamond (not a bug, tools already
    # collected via another path) or a genuine cycle (safe to skip).
    if name in visited:
        return []

    visited.add(name)

    # Get toolset definition
    toolset = get_toolset(name)
    if not toolset:
        return []

    # Collect direct tools
    tools = set(toolset.get("tools", []))

    # Recursively resolve included toolsets, sharing the visited set across
    # sibling includes so diamond dependencies are only resolved once and
    # cycle warnings don't fire multiple times for the same cycle.
    for included_name in toolset.get("includes", []):
        included_tools = resolve_toolset(included_name, visited)
        tools.update(included_tools)
    
    return sorted(tools)


def resolve_multiple_toolsets(toolset_names: List[str]) -> List[str]:
    """解析多个工具集并合并它们的工具。
    
    功能概括：
        批量解析多个工具集，将所有工具合并并去重，返回排序后的工具列表。
    
    参数：
        toolset_names: 要解析的工具集名称列表
                      （如 ["web", "terminal", "file"]）
    
    返回值：
        List[str]: 合并去重并排序后的所有工具名称列表
    
    主要用于：
        - 批量处理时组合多个工具集
        - 用户同时启用多个工具集时获取完整工具列表
    """
    all_tools = set()
    
    for name in toolset_names:
        tools = resolve_toolset(name)
        all_tools.update(tools)
    
    return sorted(all_tools)


def _get_plugin_toolset_names() -> Set[str]:
    """返回插件注册的工具集名称。
    
    功能概括：
        从工具注册中心获取由插件动态注册的工具集名称，
        这些工具集不存在于静态 TOOLSETS 字典中。
    
    参数：
        无
    
    返回值：
        Set[str]: 插件注册的工具集名称集合
    
    主要用于：
        - get_all_toolsets() 中合并静态和动态工具集
        - validate_toolset() 中验证插件工具集
    """
    try:
        from tools.registry import registry
        return {
            toolset_name
            for toolset_name in registry.get_registered_toolset_names()
            if toolset_name not in TOOLSETS
        }
    except Exception:
        return set()


def _get_registry_toolset_aliases() -> Dict[str, str]:
    """返回工具注册中心中注册的显式工具集别名。
    
    功能概括：
        获取 MCP 服务器或其他组件注册的工具集别名映射。
        例如：MCP 服务器 "weather" 可能别名指向 "weather-api" 工具集。
    
    参数：
        无
    
    返回值：
        Dict[str, str]: 别名到规范名称的映射字典
                       {"别名": "规范工具集名称"}
    
    主要用于：
        - get_toolset() 中解析 MCP 服务器别名
        - validate_toolset() 中验证别名
    """
    try:
        from tools.registry import registry
        return registry.get_registered_toolset_aliases()
    except Exception:
        return {}


def get_all_toolsets() -> Dict[str, Dict[str, Any]]:
    """获取所有可用工具集及其定义。
    
    功能概括：
        返回包含静态定义工具和插件注册工具的完整工具集字典。
        优先使用别名作为显示名称。
    
    参数：
        无
    
    返回值：
        Dict[str, Dict]: 所有工具集定义
                        {工具集名称: 工具集定义字典}
    
    主要用于：
        - 显示可用工具集列表（如 hermes tools 命令）
        - 工具集配置 UI 的数据源
        - 遍历所有工具集进行操作
    """
    result = dict(TOOLSETS)
    aliases = _get_registry_toolset_aliases()
    for ts_name in _get_plugin_toolset_names():
        display_name = ts_name
        for alias, canonical in aliases.items():
            if canonical == ts_name and alias not in TOOLSETS:
                display_name = alias
                break
        if display_name in result:
            continue
        toolset = get_toolset(display_name)
        if toolset:
            result[display_name] = toolset
    return result


def get_toolset_names() -> List[str]:
    """获取所有可用工具集的名称列表（不包括别名）。
    
    功能概括：
        返回所有工具集的名称，包括静态定义和插件注册的。
        对于有别名的插件工具集，使用别名而非规范名称。
    
    参数：
        无
    
    返回值：
        List[str]: 排序后的工具集名称列表
    
    主要用于：
        - resolve_toolset("all") 中遍历所有工具集
        - 生成工具集选择列表
    """
    names = set(TOOLSETS.keys())
    aliases = _get_registry_toolset_aliases()
    for ts_name in _get_plugin_toolset_names():
        for alias, canonical in aliases.items():
            if canonical == ts_name and alias not in TOOLSETS:
                names.add(alias)
                break
        else:
            names.add(ts_name)
    return sorted(names)




def validate_toolset(name: str) -> bool:
    """检查工具集名称是否有效。
    
    功能概括：
        验证给定的工具集名称是否存在于系统中，包括静态定义、
        插件注册和别名。
    
    参数：
        name: 要验证的工具集名称
    
    返回值：
        bool: 如果工具集有效返回 True，否则返回 False
    
    主要用于：
        - get_tool_definitions() 中验证启用的工具集
        - 用户输入的工具集名称验证
        - 配置文件中的工具集名称校验
    
    特殊支持：
        - "all" 和 "*" 始终返回 True（表示所有工具）
    """
    # Accept special alias names for convenience
    if name in {"all", "*"}:
        return True
    if name in TOOLSETS:
        return True
    if name in _get_plugin_toolset_names():
        return True
    return name in _get_registry_toolset_aliases()


def create_custom_toolset(
    name: str,
    description: str,
    tools: List[str] = None,
    includes: List[str] = None
) -> None:
    """在运行时创建自定义工具集。
    
    功能概括：
        动态创建新的工具集并添加到 TOOLSETS 字典中。
        允许用户或插件在运行时定义新的工具组合。
    
    参数：
        name: 新工具集的名称（如 "my_custom_tools"）
        description: 工具集的描述信息
        tools: 直接包含的工具名称列表，默认空列表
        includes: 要包含的其他工具集名称列表，默认空列表
    
    返回值：
        无（直接修改全局 TOOLSETS 字典）
    
    主要用于：
        - 用户通过配置文件自定义工具集
        - 插件动态注册工具集
        - 临时工具组合的创建
    
    注意：
        - 创建的工具集在整个进程生命周期内有效
        - 名称冲突会覆盖现有工具集
    """
    TOOLSETS[name] = {
        "description": description,
        "tools": tools or [],
        "includes": includes or []
    }




def get_toolset_info(name: str) -> Dict[str, Any]:
    """获取工具集的详细信息，包括解析后的完整工具列表。
    
    功能概括：
        返回工具集的完整信息，包括直接工具、包含的工具集、
        解析后的所有工具以及统计信息。
    
    参数：
        name: 工具集名称
    
    返回值：
        Dict: 包含以下键的详细工具集信息：
            - name: 工具集名称
            - description: 工具集描述
            - direct_tools: 直接包含的工具列表
            - includes: 包含的其他工具集列表
            - resolved_tools: 解析后的所有工具名称（排序）
            - tool_count: 工具总数
            - is_composite: 是否为复合工具集（包含其他工具集）
            如果工具集不存在则返回 None
    
    主要用于：
        - 显示工具集详细信息（如 hermes tools --info）
        - 调试工具集配置
        - 工具集文档生成
    """
    toolset = get_toolset(name)
    if not toolset:
        return None
    
    resolved_tools = resolve_toolset(name)
    
    return {
        "name": name,
        "description": toolset["description"],
        "direct_tools": toolset["tools"],
        "includes": toolset["includes"],
        "resolved_tools": resolved_tools,
        "tool_count": len(resolved_tools),
        "is_composite": bool(toolset["includes"])
    }




if __name__ == "__main__":
    print("Toolsets System Demo")
    print("=" * 60)
    
    print("\nAvailable Toolsets:")
    print("-" * 40)
    for name, toolset in get_all_toolsets().items():
        info = get_toolset_info(name)
        composite = "[composite]" if info["is_composite"] else "[leaf]"
        print(f"  {composite} {name:20} - {toolset['description']}")
        print(f"     Tools: {len(info['resolved_tools'])} total")
    
    print("\nToolset Resolution Examples:")
    print("-" * 40)
    for name in ["web", "terminal", "safe", "debugging"]:
        tools = resolve_toolset(name)
        print(f"\n  {name}:")
        print(f"    Resolved to {len(tools)} tools: {', '.join(sorted(tools))}")
    
    print("\nMultiple Toolset Resolution:")
    print("-" * 40)
    combined = resolve_multiple_toolsets(["web", "vision", "terminal"])
    print("  Combining ['web', 'vision', 'terminal']:")
    print(f"    Result: {', '.join(sorted(combined))}")
    
    print("\nCustom Toolset Creation:")
    print("-" * 40)
    create_custom_toolset(
        name="my_custom",
        description="My custom toolset for specific tasks",
        tools=["web_search"],
        includes=["terminal", "vision"]
    )
    custom_info = get_toolset_info("my_custom")
    print("  Created 'my_custom' toolset:")
    print(f"    Description: {custom_info['description']}")
    print(f"    Resolved tools: {', '.join(custom_info['resolved_tools'])}")
