#!/usr/bin/env python3
"""Hermes Agent 工具集概率分布模块。

本模块为批量数据生成（batch processing）定义工具集的选择概率分布。
每个分布指定了在批量处理过程中，每个提示（prompt）应该使用哪些工具集
以及它们被选中的概率。

核心概念：
    - 分布（Distribution）是一组工具集及其选择概率的映射
    - 概率表示百分比（0-100），总和不一定要等于 100
    - 每个工具集独立根据概率决定是否启用
    - 系统会在概率总和非 100 时自动归一化

使用示例：
    from toolset_distributions import get_distribution, list_distributions
    
    # 获取特定分布
    dist = get_distribution("image_gen")
    
    # 列出所有可用分布
    all_dists = list_distributions()

主要用途：
    - batch_runner.py 中的批量数据生成
    - 为不同任务类型（研究、开发、图像生成）提供不同的工具配置
    - 创建多样化的训练数据集
"""

from typing import Dict, List, Optional
import random
from toolsets import validate_toolset


# Distribution definitions
# Each key is a distribution name, and the value is a dict of toolset_name: probability_percentage
DISTRIBUTIONS = {
    # Default: All tools available 100% of the time
    "default": {
        "description": "All available tools, all the time",
        "toolsets": {
            "web": 100,
            "vision": 100,
            "image_gen": 100,
            "terminal": 100,
            "file": 100,
            "moa": 100,
            "browser": 100
        }
    },
    
    # Image generation focused distribution
    "image_gen": {
        "description": "Heavy focus on image generation with vision and web support",
        "toolsets": {
            "image_gen": 90,  # 80% chance of image generation tools
            "vision": 90,      # 60% chance of vision tools
            "web": 55,         # 40% chance of web tools
            "terminal": 45,
            "moa": 10          # 20% chance of reasoning tools
        }
    },
    
    # Research-focused distribution
    "research": {
        "description": "Web research with vision analysis and reasoning",
        "toolsets": {
            "web": 90,       # 90% chance of web tools
            "browser": 70,   # 70% chance of browser tools for deep research
            "vision": 50,    # 50% chance of vision tools
            "moa": 40,       # 40% chance of reasoning tools
            "terminal": 10   # 10% chance of terminal tools
        }
    },

    # Scientific problem solving focused distribution
    "science": {
        "description": "Scientific research with web, terminal, file, and browser capabilities",
        "toolsets": {
            "web": 94,       # 94% chance of web tools
            "terminal": 94,  # 94% chance of terminal tools
            "file": 94,      # 94% chance of file tools
            "vision": 65,    # 65% chance of vision tools
            "browser": 50,   # 50% chance of browser for accessing papers/databases
            "image_gen": 15, # 15% chance of image generation tools
            "moa": 10        # 10% chance of reasoning tools
        }
    },

    # Development-focused distribution
    "development": {
        "description": "Terminal, file tools, and reasoning with occasional web lookup",
        "toolsets": {
            "terminal": 80,  # 80% chance of terminal tools
            "file": 80,      # 80% chance of file tools (read, write, patch, search)
            "moa": 60,       # 60% chance of reasoning tools
            "web": 30,       # 30% chance of web tools
            "vision": 10     # 10% chance of vision tools
        }
    },
    
    # Safe mode (no terminal)
    "safe": {
        "description": "All tools except terminal for safety",
        "toolsets": {
            "web": 80,
            "browser": 70,   # Browser is safe (no local filesystem access)
            "vision": 60,
            "image_gen": 60,
            "moa": 50
        }
    },
    
    # Balanced distribution
    "balanced": {
        "description": "Equal probability of all toolsets",
        "toolsets": {
            "web": 50,
            "vision": 50,
            "image_gen": 50,
            "terminal": 50,
            "file": 50,
            "moa": 50,
            "browser": 50
        }
    },
    
    # Minimal (web only)
    "minimal": {
        "description": "Only web tools for basic research",
        "toolsets": {
            "web": 100
        }
    },
    
    # Terminal only
    "terminal_only": {
        "description": "Terminal and file tools for code execution tasks",
        "toolsets": {
            "terminal": 100,
            "file": 100
        }
    },
    
    # Terminal + web (common for coding tasks that need docs)
    "terminal_web": {
        "description": "Terminal and file tools with web search for documentation lookup",
        "toolsets": {
            "terminal": 100,
            "file": 100,
            "web": 100
        }
    },
    
    # Creative (vision + image generation)
    "creative": {
        "description": "Image generation and vision analysis focus",
        "toolsets": {
            "image_gen": 90,
            "vision": 90,
            "web": 30
        }
    },
    
    # Reasoning heavy
    "reasoning": {
        "description": "Heavy mixture of agents usage with minimal other tools",
        "toolsets": {
            "moa": 90,
            "web": 30,
            "terminal": 20
        }
    },
    
    # Browser-based web interaction
    "browser_use": {
        "description": "Full browser-based web interaction with search, vision, and page control",
        "toolsets": {
            "browser": 100,  # All browser tools always available
            "web": 80,       # Web search for finding URLs and quick lookups
            "vision": 70     # Vision analysis for images found on pages
        }
    },
    
    # Browser only (no other tools)
    "browser_only": {
        "description": "Only browser automation tools for pure web interaction tasks",
        "toolsets": {
            "browser": 100
        }
    },
    
    # Browser-focused tasks distribution (for browser-use-tasks.jsonl)
    "browser_tasks": {
        "description": "Browser-focused distribution (browser toolset includes web_search for finding URLs since Google blocks direct browser searches)",
        "toolsets": {
            "browser": 97,   # 97% - browser tools (includes web_search) almost always available
            "vision": 12,    # 12% - vision analysis occasionally
            "terminal": 15   # 15% - terminal occasionally for local operations
        }
    },
    
    # Terminal-focused tasks distribution (for nous-terminal-tasks.jsonl)
    "terminal_tasks": {
        "description": "Terminal-focused distribution with high terminal/file availability, occasional other tools",
        "toolsets": {
            "terminal": 97,   # 97% - terminal almost always available
            "file": 97,       # 97% - file tools almost always available
            "web": 97,        # 15% - web search/scrape for documentation
            "browser": 75,    # 10% - browser occasionally for web interaction
            "vision": 50,      # 8% - vision analysis rarely
            "image_gen": 10    # 3% - image generation very rarely
        }
    },
    
    # Mixed browser+terminal tasks distribution (for mixed-browser-terminal-tasks.jsonl)
    "mixed_tasks": {
        "description": "Mixed distribution with high browser, terminal, and file availability for complex tasks",
        "toolsets": {
            "browser": 92,    # 92% - browser tools highly available
            "terminal": 92,   # 92% - terminal highly available
            "file": 92,       # 92% - file tools highly available
            "web": 35,        # 35% - web search/scrape fairly common
            "vision": 15,     # 15% - vision analysis occasionally
            "image_gen": 15   # 15% - image generation occasionally
        }
    }
}


def get_distribution(name: str) -> Optional[Dict[str, any]]:
    """按名称获取工具集分布定义。
    
    功能概括：
        从 DISTRIBUTIONS 字典中获取指定分布的完整定义。
    
    参数：
        name: 分布名称（如 "default"、"research"、"image_gen"）
    
    返回值：
        dict | None: 分布定义字典，包含：
            - description: 分布描述
            - toolsets: {工具集名称: 概率百分比} 的映射
            如果分布不存在则返回 None
    
    主要用于：
        - batch_runner.py 中加载分布配置
        - 查询特定分布的工具集概率
    """
    return DISTRIBUTIONS.get(name)


def list_distributions() -> Dict[str, Dict]:
    """列出所有可用的分布定义。
    
    功能概括：
        返回所有已定义的工具集分布的副本。
    
    参数：
        无
    
    返回值：
        Dict[str, Dict]: 所有分布定义的副本
                        {分布名称: 分布定义}
    
    主要用于：
        - 显示可用分布列表
        - 分布选择 UI 的数据源
        - 遍历所有分布进行操作
    """
    return DISTRIBUTIONS.copy()


def sample_toolsets_from_distribution(distribution_name: str) -> List[str]:
    """根据分布的概率采样工具集。
    
    功能概括：
        根据分布定义中每个工具集的概率，独立采样决定哪些工具集被启用。
        允许多个工具集同时激活，模拟真实使用场景中的工具集变化。
    
    参数：
        distribution_name: 要采样的分布名称
    
    返回值：
        List[str]: 采样得到的工具集名称列表
    
    异常：
        ValueError: 如果分布名称不存在
    
    主要用于：
        - batch_runner.py 中为每个提示随机选择工具集
        - 创建多样化的训练数据
        - 模拟不同场景下的工具配置
    
    采样逻辑：
        1. 遍历分布中的每个工具集
        2. 生成 0-100 的随机数，如果小于概率值则选中该工具集
        3. 如果没有工具集被选中（低概率时可能发生），
           则选择概率最高的工具集作为保底
    """
    dist = get_distribution(distribution_name)
    if not dist:
        raise ValueError(f"Unknown distribution: {distribution_name}")
    
    # Sample each toolset independently based on its probability
    selected_toolsets = []
    
    for toolset_name, probability in dist["toolsets"].items():
        # Validate toolset exists
        if not validate_toolset(toolset_name):
            print(f"⚠️  Warning: Toolset '{toolset_name}' in distribution '{distribution_name}' is not valid")
            continue
        
        # Roll the dice - if random value is less than probability, include this toolset
        if random.random() * 100 < probability:
            selected_toolsets.append(toolset_name)
    
    # If no toolsets were selected (can happen with low probabilities), 
    # ensure at least one toolset is selected by picking the highest probability one
    if not selected_toolsets and dist["toolsets"]:
        # Find toolset with highest probability
        highest_prob_toolset = max(dist["toolsets"].items(), key=lambda x: x[1])[0]
        if validate_toolset(highest_prob_toolset):
            selected_toolsets.append(highest_prob_toolset)
    
    return selected_toolsets


def validate_distribution(distribution_name: str) -> bool:
    """检查分布名称是否有效。
    
    功能概括：
        验证给定的分布名称是否存在于 DISTRIBUTIONS 字典中。
    
    参数：
        distribution_name: 要验证的分布名称
    
    返回值：
        bool: 如果分布存在返回 True，否则返回 False
    
    主要用于：
        - 用户输入的分布名称验证
        - 配置文件中的分布名称校验
        - 采样前的预检查
    """
    return distribution_name in DISTRIBUTIONS


def print_distribution_info(distribution_name: str) -> None:
    """打印分布的详细信息。
    
    功能概括：
        以用户友好的格式打印分布的描述和工具集概率信息。
    
    参数：
        distribution_name: 要打印信息的分布名称
    
    返回值：
        无（直接打印到控制台）
    
    主要用于：
        - 调试和测试
        - 显示分布配置详情
        - CLI 命令的信息展示
    """
    dist = get_distribution(distribution_name)
    if not dist:
        print(f"❌ Unknown distribution: {distribution_name}")
        return
    
    print(f"\n📊 Distribution: {distribution_name}")
    print(f"   Description: {dist['description']}")
    print("   Toolsets:")
    for toolset, prob in sorted(dist["toolsets"].items(), key=lambda x: x[1], reverse=True):
        print(f"     • {toolset:15} : {prob:3}% chance")


if __name__ == "__main__":
    """
    Demo and testing of the distributions system
    """
    print("📊 Toolset Distributions Demo")
    print("=" * 60)
    
    # List all distributions
    print("\n📋 Available Distributions:")
    print("-" * 40)
    for name, dist in list_distributions().items():
        print(f"\n  {name}:")
        print(f"    {dist['description']}")
        toolset_list = ", ".join([f"{ts}({p}%)" for ts, p in dist["toolsets"].items()])
        print(f"    Toolsets: {toolset_list}")
    
    # Demo sampling
    print("\n\n🎲 Sampling Examples:")
    print("-" * 40)
    
    test_distributions = ["image_gen", "research", "balanced", "default"]
    
    for dist_name in test_distributions:
        print(f"\n{dist_name}:")
        # Sample 5 times to show variability
        samples = []
        for _ in range(5):
            sampled = sample_toolsets_from_distribution(dist_name)
            samples.append(sorted(sampled))
        
        print(f"  Sample 1: {samples[0]}")
        print(f"  Sample 2: {samples[1]}")
        print(f"  Sample 3: {samples[2]}")
        print(f"  Sample 4: {samples[3]}")
        print(f"  Sample 5: {samples[4]}")
    
    # Show detailed info
    print("\n\n📊 Detailed Distribution Info:")
    print("-" * 40)
    print_distribution_info("image_gen")
    print_distribution_info("research")

