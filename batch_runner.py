#!/usr/bin/env python3
"""Hermes Agent 批量处理运行器。

提供并行批量处理能力，支持在数据集的多个提示上运行 agent。特性包括：
- 数据集加载和分批
- 使用 multiprocessing 的并行批处理
- 用于容错和恢复的检查点机制
- 以正确格式保存轨迹（from/value 对）
- 跨所有批次聚合并工具使用统计

使用示例：
    python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run
    
    # 恢复中断的运行
    python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run --resume
    
    # 使用特定的工具集分布
    python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run --distribution=image_gen

主要用途：
    - 大规模生成训练数据（用于模型微调）
    - 批量评估 agent 性能
    - 不同工具集分布的对比实验
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from multiprocessing import Pool, Lock
import traceback
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console

logger = logging.getLogger(__name__)
import fire

from run_agent import AIAgent
from toolset_distributions import (
    list_distributions, 
    sample_toolsets_from_distribution,
    validate_distribution
)
from model_tools import TOOL_TO_TOOLSET_MAP


# Global configuration for worker processes
_WORKER_CONFIG = {}

# All possible tools - auto-derived from the master mapping in model_tools.py.
# This stays in sync automatically when new tools are added to TOOL_TO_TOOLSET_MAP.
# Used for consistent schema in Arrow/Parquet (HuggingFace datasets) and for
# filtering corrupted entries during trajectory combination.
ALL_POSSIBLE_TOOLS = set(TOOL_TO_TOOLSET_MAP.keys())

# Default stats for tools that weren't used
DEFAULT_TOOL_STATS = {'count': 0, 'success': 0, 'failure': 0}


def _normalize_tool_stats(tool_stats: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """标准化工具统计，包含所有可能的工具并保持一致的 schema。
    
    功能概括：
        确保 HuggingFace 数据集加载 JSONL 时不会出现 schema 不匹配错误。
        未使用的工具获得零计数，保证所有条目具有相同的字段结构。
    
    参数：
        tool_stats: 原始工具统计字典 {tool_name: {count, success, failure}}
    
    返回值：
        Dict: 标准化后的工具统计，包含 ALL_POSSIBLE_TOOLS 中的所有工具
    
    主要用于：
        - _process_batch_worker() 中保存轨迹前标准化数据
        - 确保 Arrow/Parquet 格式数据集的 schema 一致性
    """
    normalized = {}
    
    # Add all possible tools with defaults
    for tool in ALL_POSSIBLE_TOOLS:
        if tool in tool_stats:
            normalized[tool] = tool_stats[tool].copy()
        else:
            normalized[tool] = DEFAULT_TOOL_STATS.copy()
    
    # Also include any unexpected tools (in case new tools are added)
    for tool, stats in tool_stats.items():
        if tool not in normalized:
            normalized[tool] = stats.copy()
    
    return normalized


def _normalize_tool_error_counts(tool_error_counts: Dict[str, int]) -> Dict[str, int]:
    """标准化工具错误计数，包含所有可能的工具。
    
    功能概括：
        为所有可能的工具添加错误计数字段，未使用的工具默认为 0。
    
    参数：
        tool_error_counts: 原始错误计数映射 {tool_name: failure_count}
    
    返回值：
        Dict: 标准化后的错误计数，包含所有工具
    
    主要用于：
        - _process_batch_worker() 中构建轨迹条目
        - 提供简单的工具失败统计视图
    """
    normalized = {}
    
    # Add all possible tools with zero defaults
    for tool in ALL_POSSIBLE_TOOLS:
        normalized[tool] = tool_error_counts.get(tool, 0)
    
    # Also include any unexpected tools
    for tool, count in tool_error_counts.items():
        if tool not in normalized:
            normalized[tool] = count
    
    return normalized


def _extract_tool_stats(messages: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """从消息历史中提取工具使用统计。
    
    功能概括：
        分析 assistant 和 tool 角色的消息，统计每个工具的调用次数、
        成功次数和失败次数。通过解析工具响应的 JSON 内容判断成功/失败。
    
    参数：
        messages: 消息历史列表（OpenAI 格式）
    
    返回值：
        Dict: 工具统计 {tool_name: {count, success, failure}}
    
    主要用于：
        - _process_single_prompt() 中提取任务的工具使用情况
        - 批处理结束时聚合所有任务的统计
    
    成功判定规则：
        - JSON 响应中 error 字段为 None 且 success 不为 False → 成功
        - terminal 工具的 content.error 为 None → 成功（非零退出码不算失败）
        - 空内容或明确以 "error:" 开头 → 失败
    """
    tool_stats = {}
    
    # Track tool calls and their results
    tool_calls_map = {}  # Map tool_call_id to tool name
    
    for msg in messages:
        # Track tool calls from assistant messages
        if msg["role"] == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
            for tool_call in msg["tool_calls"]:
                if not tool_call or not isinstance(tool_call, dict): continue
                tool_name = tool_call["function"]["name"]
                tool_call_id = tool_call["id"]
                
                # Initialize stats for this tool if not exists
                if tool_name not in tool_stats:
                    tool_stats[tool_name] = {
                        "count": 0,
                        "success": 0,
                        "failure": 0
                    }
                
                tool_stats[tool_name]["count"] += 1
                tool_calls_map[tool_call_id] = tool_name
        
        # Track tool responses
        elif msg["role"] == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "")
            
            # Determine if tool call was successful
            is_success = True
            try:
                # Try to parse as JSON and check for actual error values
                content_json = json.loads(content) if isinstance(content, str) else content
                
                if isinstance(content_json, dict):
                    # Check if error field exists AND has a non-null value
                    if "error" in content_json and content_json["error"] is not None:
                        is_success = False
                    
                    # Special handling for terminal tool responses
                    # Terminal wraps its response in a "content" field
                    if "content" in content_json and isinstance(content_json["content"], dict):
                        inner_content = content_json["content"]
                        # Check for actual error (non-null error field)
                        # Note: non-zero exit codes are not failures - the model can self-correct
                        if inner_content.get("error") is not None:
                            is_success = False
                    
                    # Check for "success": false pattern used by some tools
                    if content_json.get("success") is False:
                        is_success = False
                        
            except (json.JSONDecodeError, ValueError, TypeError):
                # If not JSON, check if content is empty or explicitly states an error
                # Note: We avoid simple substring matching to prevent false positives
                if not content:
                    is_success = False
                # Only mark as failure if it explicitly starts with "Error:" or "ERROR:"
                elif content.strip().lower().startswith("error:"):
                    is_success = False
            
            # Update success/failure count
            if tool_call_id in tool_calls_map:
                tool_name = tool_calls_map[tool_call_id]
                if is_success:
                    tool_stats[tool_name]["success"] += 1
                else:
                    tool_stats[tool_name]["failure"] += 1
    
    return tool_stats


def _extract_reasoning_stats(messages: List[Dict[str, Any]]) -> Dict[str, int]:
    """统计 assistant 回复中包含推理和不含推理的数量。
    
    功能概括：
        检查 assistant 消息中是否包含 <REASONING_SCRATCHPAD> 或
        非空的 'reasoning' 字段（原生 thinking tokens）。
        返回计数以跟踪推理覆盖率。
    
    参数：
        messages: 消息历史
    
    返回值：
        Dict: 包含以下键：
            - total_assistant_turns: assistant 总回复数
            - turns_with_reasoning: 包含推理的回复数
            - turns_without_reasoning: 不含推理的回复数
            - has_any_reasoning: 是否有任何推理（bool）
    
    主要用于：
        - _process_single_prompt() 中评估 agent 的推理使用
        - 过滤不含推理的低质量样本
        - 批处理结束时计算推理覆盖率
    """
    total = 0
    with_reasoning = 0
    
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        total += 1
        
        content = msg.get("content", "") or ""
        has_scratchpad = "<REASONING_SCRATCHPAD>" in content
        has_native_reasoning = bool(msg.get("reasoning", "").strip()) if msg.get("reasoning") else False
        
        if has_scratchpad or has_native_reasoning:
            with_reasoning += 1
    
    return {
        "total_assistant_turns": total,
        "turns_with_reasoning": with_reasoning,
        "turns_without_reasoning": total - with_reasoning,
        "has_any_reasoning": with_reasoning > 0,
    }


def _process_single_prompt(
    prompt_index: int,
    prompt_data: Dict[str, Any],
    batch_num: int,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """用 agent 处理单个提示。
    
    功能概括：
        为单个任务初始化 agent、执行对话循环、提取统计信息并返回轨迹。
        支持数据集行中的容器镜像覆盖（docker/modal/singularity/daytona）。
    
    参数：
        prompt_index: 数据集中提示的索引
        prompt_data: 提示数据，包含 'prompt' 字段和可选的 'image' 字段
        batch_num: 批次号
        config: 配置字典，包含 agent 参数
    
    返回值：
        Dict: 包含以下键的结果：
            - success: 是否成功（bool）
            - trajectory: Hermes 格式的轨迹（成功时）
            - tool_stats: 工具使用统计
            - reasoning_stats: 推理覆盖统计
            - completed: 任务是否完成
            - api_calls: API 调用次数
            - toolsets_used: 使用的工具集
            - metadata: 元数据
    
    主要用于：
        - _process_batch_worker() 中顺序处理批次中的每个提示
        - 为每个提示采样不同的工具集分布
    
    特殊处理：
        - 如果数据集行包含 'image' 字段，为该任务注册容器镜像覆盖
        - Docker 环境下会先检查/拉取镜像
        - 跳过上下文文件和内存以避免污染轨迹
    """
    prompt = prompt_data["prompt"]
    task_id = f"task_{prompt_index}"
    
    # Per-prompt container image override: if the dataset row has an 'image' field,
    # register it for this task's sandbox. Works with Docker, Modal, Singularity, and Daytona.
    container_image = prompt_data.get("image") or prompt_data.get("docker_image")
    if container_image:
        # Verify the image is accessible before spending tokens on the agent loop.
        # For Docker: check local cache, then try pulling.
        # For Modal: skip local check (Modal pulls server-side).
        env_type = os.getenv("TERMINAL_ENV", "local")
        if env_type == "docker":
            import subprocess as _sp
            try:
                probe = _sp.run(
                    ["docker", "image", "inspect", container_image],
                    capture_output=True, timeout=10,
                )
                if probe.returncode != 0:
                    if config.get("verbose"):
                        print(f"   Prompt {prompt_index}: Pulling docker image {container_image}...", flush=True)
                    pull = _sp.run(
                        ["docker", "pull", container_image],
                        capture_output=True, text=True, timeout=600,
                    )
                    if pull.returncode != 0:
                        return {
                            "success": False,
                            "prompt_index": prompt_index,
                            "error": f"Docker image not available: {container_image}\n{pull.stderr[:500]}",
                            "trajectory": None,
                            "tool_stats": {},
                            "toolsets_used": [],
                            "metadata": {"batch_num": batch_num, "timestamp": datetime.now().isoformat()},
                        }
            except FileNotFoundError:
                pass  # Docker CLI not installed — skip check (e.g., Modal backend)
            except Exception as img_err:
                if config.get("verbose"):
                    print(f"   Prompt {prompt_index}: Docker image check failed: {img_err}", flush=True)

        from tools.terminal_tool import register_task_env_overrides
        overrides = {
            "docker_image": container_image,
            "modal_image": container_image,
            "singularity_image": f"docker://{container_image}",
            "daytona_image": container_image,
        }
        if prompt_data.get("cwd"):
            overrides["cwd"] = prompt_data["cwd"]
        register_task_env_overrides(task_id, overrides)
        if config.get("verbose"):
            print(f"   Prompt {prompt_index}: Using container image {container_image}")
    
    try:
        # Sample toolsets from distribution for this prompt
        selected_toolsets = sample_toolsets_from_distribution(config["distribution"])
        
        if config.get("verbose"):
            print(f"   Prompt {prompt_index}: Using toolsets {selected_toolsets}")
        
        # Initialize agent with sampled toolsets and log prefix for identification
        log_prefix = f"[B{batch_num}:P{prompt_index}]"
        agent = AIAgent(
            base_url=config.get("base_url"),
            api_key=config.get("api_key"),
            model=config["model"],
            max_iterations=config["max_iterations"],
            enabled_toolsets=selected_toolsets,
            save_trajectories=False,  # We handle saving ourselves
            verbose_logging=config.get("verbose", False),
            ephemeral_system_prompt=config.get("ephemeral_system_prompt"),
            log_prefix_chars=config.get("log_prefix_chars", 100),
            log_prefix=log_prefix,
            providers_allowed=config.get("providers_allowed"),
            providers_ignored=config.get("providers_ignored"),
            providers_order=config.get("providers_order"),
            provider_sort=config.get("provider_sort"),
            max_tokens=config.get("max_tokens"),
            reasoning_config=config.get("reasoning_config"),
            prefill_messages=config.get("prefill_messages"),
            skip_context_files=True,  # Don't pollute trajectories with SOUL.md/AGENTS.md
            skip_memory=True,  # Don't use persistent memory in batch runs
        )

        # Run the agent with task_id to ensure each task gets its own isolated VM
        result = agent.run_conversation(prompt, task_id=task_id)
        
        # Extract tool usage statistics
        tool_stats = _extract_tool_stats(result["messages"])
        
        # Extract reasoning coverage stats
        reasoning_stats = _extract_reasoning_stats(result["messages"])
        
        # Convert to trajectory format (using existing method)
        trajectory = agent._convert_to_trajectory_format(
            result["messages"],
            prompt,
            result["completed"]
        )
        
        return {
            "success": True,
            "prompt_index": prompt_index,
            "trajectory": trajectory,
            "tool_stats": tool_stats,
            "reasoning_stats": reasoning_stats,
            "completed": result["completed"],
            "partial": result.get("partial", False),
            "api_calls": result["api_calls"],
            "toolsets_used": selected_toolsets,
            "metadata": {
                "batch_num": batch_num,
                "timestamp": datetime.now().isoformat(),
                "model": config["model"]
            }
        }
    
    except Exception as e:
        print(f"❌ Error processing prompt {prompt_index}: {e}")
        if config.get("verbose"):
            traceback.print_exc()
        
        return {
            "success": False,
            "prompt_index": prompt_index,
            "error": str(e),
            "trajectory": None,
            "tool_stats": {},
            "toolsets_used": [],
            "metadata": {
                "batch_num": batch_num,
                "timestamp": datetime.now().isoformat()
            }
        }


def _process_batch_worker(args: Tuple) -> Dict[str, Any]:
    """工作函数，处理单个批次的提示。
    
    功能概括：
        在多进程工作进程中执行，处理一个批次的所有提示。
        顺序处理每个提示，保存轨迹，聚合统计。
    
    参数：
        args: 元组 (batch_num, batch_data, output_dir, completed_prompts_set, config)
            - batch_num: 批次号
            - batch_data: 批次数据 [(index, prompt_data), ...]
            - output_dir: 输出目录
            - completed_prompts_set: 已完成的提示索引集合
            - config: 配置字典
    
    返回值：
        Dict: 批次结果，包含：
            - batch_num: 批次号
            - processed: 处理的提示数
            - skipped: 跳过的提示数（已完成）
            - tool_stats: 聚合的工具统计
            - reasoning_stats: 聚合的推理统计
            - discarded_no_reasoning: 因无推理而被丢弃的样本数
            - completed_prompts: 本次成功完成的提示索引列表
    
    主要用于：
        - BatchRunner.run() 中的 Pool.imap_unordered() 调用
        - 并行处理多个批次
    
    处理流程：
        1. 过滤已完成的提示（支持 resume）
        2. 顺序处理每个提示
        3. 丢弃不含推理的样本
        4. 标准化并保存轨迹
        5. 聚合统计
    """
    batch_num, batch_data, output_dir, completed_prompts_set, config = args
    
    output_dir = Path(output_dir)
    print(f"\n🔄 Batch {batch_num}: Starting ({len(batch_data)} prompts)")
    
    # Output file for this batch
    batch_output_file = output_dir / f"batch_{batch_num}.jsonl"
    
    # Filter out already completed prompts
    prompts_to_process = [
        (idx, data) for idx, data in batch_data
        if idx not in completed_prompts_set
    ]
    
    if not prompts_to_process:
        print(f"✅ Batch {batch_num}: Already completed (skipping)")
        return {
            "batch_num": batch_num,
            "processed": 0,
            "skipped": len(batch_data),
            "tool_stats": {},
            "completed_prompts": []
        }
    
    print(f"   Processing {len(prompts_to_process)} prompts (skipping {len(batch_data) - len(prompts_to_process)} already completed)")
    
    # Initialize aggregated stats for this batch
    batch_tool_stats = {}
    batch_reasoning_stats = {"total_assistant_turns": 0, "turns_with_reasoning": 0, "turns_without_reasoning": 0}
    completed_in_batch = []
    discarded_no_reasoning = 0
    
    # Process each prompt sequentially in this batch
    for prompt_index, prompt_data in prompts_to_process:
        # Process the prompt
        result = _process_single_prompt(
            prompt_index,
            prompt_data,
            batch_num,
            config
        )
        
        # Save trajectory if successful
        if result["success"] and result["trajectory"]:
            # Discard samples with zero reasoning across all turns
            reasoning = result.get("reasoning_stats", {})
            if not reasoning.get("has_any_reasoning", True):
                print(f"   🚫 Prompt {prompt_index} discarded (no reasoning in any turn)")
                discarded_no_reasoning += 1
                continue
            
            # Get and normalize tool stats for consistent schema across all entries
            raw_tool_stats = result.get("tool_stats", {})
            tool_stats = _normalize_tool_stats(raw_tool_stats)
            
            # Create normalized tool_error_counts mapping tool names to their failure counts
            raw_error_counts = {
                tool_name: stats.get("failure", 0) 
                for tool_name, stats in raw_tool_stats.items()
            }
            tool_error_counts = _normalize_tool_error_counts(raw_error_counts)
            
            trajectory_entry = {
                "prompt_index": prompt_index,
                "conversations": result["trajectory"],
                "metadata": result["metadata"],
                "completed": result["completed"],
                "partial": result.get("partial", False),  # True if stopped due to invalid tool calls
                "api_calls": result["api_calls"],
                "toolsets_used": result["toolsets_used"],
                "tool_stats": tool_stats,  # Full stats: {tool: {count, success, failure}} - normalized
                "tool_error_counts": tool_error_counts  # Simple: {tool: failure_count} - normalized
            }
            
            # Append to batch output file
            with open(batch_output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(trajectory_entry, ensure_ascii=False) + "\n")
        
        # Aggregate tool statistics
        for tool_name, stats in result.get("tool_stats", {}).items():
            if tool_name not in batch_tool_stats:
                batch_tool_stats[tool_name] = {
                    "count": 0,
                    "success": 0,
                    "failure": 0
                }
            
            batch_tool_stats[tool_name]["count"] += stats["count"]
            batch_tool_stats[tool_name]["success"] += stats["success"]
            batch_tool_stats[tool_name]["failure"] += stats["failure"]
        
        # Aggregate reasoning stats
        for key in batch_reasoning_stats:
            batch_reasoning_stats[key] += result.get("reasoning_stats", {}).get(key, 0)
        
        # Only mark as completed if successfully saved (failed prompts can be retried on resume)
        if result["success"] and result["trajectory"]:
            completed_in_batch.append(prompt_index)
            status = "⚠️  partial" if result.get("partial") else "✅"
            print(f"   {status} Prompt {prompt_index} completed")
        else:
            print(f"   ❌ Prompt {prompt_index} failed (will retry on resume)")
    
    print(f"✅ Batch {batch_num}: Completed ({len(prompts_to_process)} prompts processed)")
    
    return {
        "batch_num": batch_num,
        "processed": len(prompts_to_process),
        "skipped": len(batch_data) - len(prompts_to_process),
        "tool_stats": batch_tool_stats,
        "reasoning_stats": batch_reasoning_stats,
        "discarded_no_reasoning": discarded_no_reasoning,
        "completed_prompts": completed_in_batch
    }


class BatchRunner:
    """管理 agent 提示的批处理，支持检查点和统计。
    
    功能概括：
        负责整个批量处理流程：
        1. 加载数据集并分批
        2. 使用多进程并行处理批次
        3. 维护检查点以支持中断恢复
        4. 合并所有批次文件为单一轨迹文件
        5. 生成工具使用和推理覆盖统计
    
    主要用途：
        - main() 函数中初始化并运行批量处理
        - 大规模训练数据生成
        - 不同配置的对比实验
    
    恢复机制：
        - 支持 --resume 从上次中断处继续
        - 基于内容匹配（而非索引）识别已完成的提示
        - 增量检查点保存，避免重复工作
    """
    
    def __init__(
        self,
        dataset_file: str,
        batch_size: int,
        run_name: str,
        distribution: str = "default",
        max_iterations: int = 10,
        base_url: str = None,
        api_key: str = None,
        model: str = "claude-opus-4-20250514",
        num_workers: int = 4,
        verbose: bool = False,
        ephemeral_system_prompt: str = None,
        log_prefix_chars: int = 100,
        providers_allowed: List[str] = None,
        providers_ignored: List[str] = None,
        providers_order: List[str] = None,
        provider_sort: str = None,
        max_tokens: int = None,
        reasoning_config: Dict[str, Any] = None,
        prefill_messages: List[Dict[str, Any]] = None,
        max_samples: int = None,
    ):
        """初始化批处理运行器。

        功能概括：
            配置批处理参数、验证工具集分布、加载数据集、创建批次。

        参数：
            dataset_file: JSONL 数据集文件路径，包含 'prompt' 字段
            batch_size: 每批提示数量
            run_name: 运行名称（用于检查点和输出）
            distribution: 工具集分布名称（默认："default"）
            max_iterations: 每次 agent 运行的最大迭代次数
            base_url: 模型 API 基础 URL
            api_key: 模型 API 密钥
            model: 使用的模型名称
            num_workers: 并行工作进程数
            verbose: 是否启用详细日志
            ephemeral_system_prompt: 系统提示（用于 agent 执行但不保存到轨迹）
            log_prefix_chars: 日志预览中工具调用/响应的字符数（默认：100）
            providers_allowed: 允许使用的 OpenRouter 提供商列表
            providers_ignored: 忽略的 OpenRouter 提供商列表
            providers_order: OpenRouter 提供商尝试顺序
            provider_sort: 按价格/吞吐量/延迟排序提供商
            max_tokens: 模型响应的最大 token 数
            reasoning_config: OpenRouter 推理配置（如 {"effort": "none"} 禁用思考）
            prefill_messages: 预填消息列表（few-shot priming）
            max_samples: 仅处理数据集的前 N 个样本（可选）

        返回值：
            无

        主要用于：
            - main() 函数中初始化运行器
            - 准备所有批处理所需的配置和数据

        注意：
            Anthropic Sonnet 4.6+ 和 Opus 4.6+ 拒绝尾随 assistant 角色的预填消息
            （会报 400 错误）。对于这些模型，应使用 output_config.format 或结构化输出 schema。
        """
        self.dataset_file = Path(dataset_file)
        self.batch_size = batch_size
        self.run_name = run_name
        self.distribution = distribution
        self.max_iterations = max_iterations
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.num_workers = num_workers
        self.verbose = verbose
        self.ephemeral_system_prompt = ephemeral_system_prompt
        self.log_prefix_chars = log_prefix_chars
        self.providers_allowed = providers_allowed
        self.providers_ignored = providers_ignored
        self.providers_order = providers_order
        self.provider_sort = provider_sort
        self.max_tokens = max_tokens
        self.reasoning_config = reasoning_config
        self.prefill_messages = prefill_messages
        self.max_samples = max_samples
        
        # Validate distribution
        if not validate_distribution(distribution):
            raise ValueError(f"Unknown distribution: {distribution}. Available: {list(list_distributions().keys())}")
        
        # Setup output directory
        self.output_dir = Path("data") / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint file
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        
        # Statistics file
        self.stats_file = self.output_dir / "statistics.json"
        
        # Load dataset (and optionally truncate to max_samples)
        self.dataset = self._load_dataset()
        if self.max_samples and self.max_samples < len(self.dataset):
            full_count = len(self.dataset)
            self.dataset = self.dataset[:self.max_samples]
            print(f"✂️  Truncated dataset from {full_count} to {self.max_samples} samples (--max_samples)")
        
        # Create batches
        self.batches = self._create_batches()
        
        print("📊 Batch Runner Initialized")
        print(f"   Dataset: {self.dataset_file} ({len(self.dataset)} prompts)")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Total batches: {len(self.batches)}")
        print(f"   Run name: {self.run_name}")
        print(f"   Distribution: {self.distribution}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Workers: {self.num_workers}")
        if self.ephemeral_system_prompt:
            prompt_preview = self.ephemeral_system_prompt[:60] + "..." if len(self.ephemeral_system_prompt) > 60 else self.ephemeral_system_prompt
            print(f"   🔒 Ephemeral system prompt: '{prompt_preview}'")
    
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """从 JSONL 文件加载数据集。
        
        功能概括：
            逐行读取 JSONL 文件，解析 JSON 并验证 'prompt' 字段存在。
            跳过空行和无效 JSON。
        
        参数：
            无
        
        返回值：
            List[Dict]: 数据集条目列表，每个条目包含 'prompt' 和其他可选字段
        
        主要用于：
            - __init__() 中加载输入数据
        
        错误处理：
            - 文件不存在：抛出 FileNotFoundError
            - 无有效条目：抛出 ValueError
            - 无效 JSON：打印警告并跳过
        """
        if not self.dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_file}")
        
        dataset = []
        with open(self.dataset_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    if 'prompt' not in entry:
                        print(f"⚠️  Warning: Line {line_num} missing 'prompt' field, skipping")
                        continue
                    dataset.append(entry)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Warning: Invalid JSON on line {line_num}: {e}")
                    continue
        
        if not dataset:
            raise ValueError(f"No valid entries found in dataset file: {self.dataset_file}")
        
        return dataset
    
    def _create_batches(self) -> List[List[Tuple[int, Dict[str, Any]]]]:
        """将数据集分割为带索引的批次。
        
        功能概括：
            按 batch_size 将数据集分割，保留原始索引用于跟踪。
        
        参数：
            无
        
        返回值：
            List: 批次列表，每个批次是 [(index, entry), ...] 元组列表
        
        主要用于：
            - __init__() 中创建初始批次
            - resume 时重新创建批次
        """
        batches = []
        for i in range(0, len(self.dataset), self.batch_size):
            batch = [(idx, entry) for idx, entry in enumerate(self.dataset[i:i + self.batch_size], start=i)]
            batches.append(batch)
        
        return batches
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """如果存在则加载检查点数据。
        
        功能概括：
            读取 checkpoint.json 文件，获取已完成的提示索引和批次统计。
            如果文件不存在或加载失败，返回默认空检查点。
        
        参数：
            无
        
        返回值：
            Dict: 检查点数据，包含：
                - run_name: 运行名称
                - completed_prompts: 已完成的提示索引列表
                - batch_stats: 批次统计
                - last_updated: 最后更新时间
        
        主要用于：
            - run() 开始时加载上次进度
            - resume 模式恢复中断的运行
        """
        if not self.checkpoint_file.exists():
            return {
                "run_name": self.run_name,
                "completed_prompts": [],
                "batch_stats": {},
                "last_updated": None
            }
        
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Failed to load checkpoint: {e}")
            return {
                "run_name": self.run_name,
                "completed_prompts": [],
                "batch_stats": {},
                "last_updated": None
            }
    
    def _save_checkpoint(self, checkpoint_data: Dict[str, Any], lock: Optional[Lock] = None):
        """保存检查点数据。
        
        功能概括：
            使用原子写入（atomic_json_write）保存检查点，确保数据完整性。
            支持可选的锁以进行线程安全访问。
        
        参数：
            checkpoint_data: 要保存的检查点数据
            lock: 可选的锁，用于线程安全访问
        
        返回值：
            无
        
        主要用于：
            - run() 中每个批次完成后增量保存
            - run() 结束时保存最终检查点
        """
        checkpoint_data["last_updated"] = datetime.now().isoformat()

        from utils import atomic_json_write
        if lock:
            with lock:
                atomic_json_write(self.checkpoint_file, checkpoint_data)
        else:
            atomic_json_write(self.checkpoint_file, checkpoint_data)
    
    def _scan_completed_prompts_by_content(self) -> set:
        """扫描所有批次文件，通过实际内容提取已完成的提示。
        
        功能概括：
            读取所有 batch_*.jsonl 文件，从 conversations 中提取 human 消息内容，
            构建已完成提示文本集合。这比基于索引的恢复更健壮。
        
        参数：
            无
        
        返回值：
            set: 已成功处理的提示文本集合
        
        主要用于：
            - run(resume=True) 时识别哪些提示已完成
            - 支持更健壮的恢复机制（不依赖索引匹配）
        """
        completed_prompts = set()
        batch_files = sorted(self.output_dir.glob("batch_*.jsonl"))
        
        if not batch_files:
            return completed_prompts
        
        print(f"📂 Scanning {len(batch_files)} batch files for completed prompts...")
        
        for batch_file in batch_files:
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            
                            # Skip failed entries - we want to retry these
                            if entry.get("failed", False):
                                continue
                            
                            # Extract the human/user prompt from conversations
                            conversations = entry.get("conversations", [])
                            for msg in conversations:
                                if msg.get("from") == "human":
                                    prompt_text = msg.get("value", "").strip()
                                    if prompt_text:
                                        completed_prompts.add(prompt_text)
                                    break  # Only need the first human message
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"  ⚠️  Warning: Error reading {batch_file.name}: {e}")
        
        return completed_prompts
    
    def _filter_dataset_by_completed(self, completed_prompts: set) -> Tuple[List[Dict], List[int]]:
        """过滤数据集，排除已完成的提示。
        
        功能概括：
            将数据集与已完成提示集合对比，返回未处理的条目和跳过的索引。
        
        参数：
            completed_prompts: 已完成的提示文本集合
        
        返回值：
            Tuple: (filtered_dataset, skipped_indices)
                - filtered_dataset: 未处理的条目列表 [(original_index, entry), ...]
                - skipped_indices: 已完成的提示索引列表
        
        主要用于：
            - run(resume=True) 中过滤已完成的提示
            - 重新创建批次时只包含未处理的提示
        """
        filtered_dataset = []
        skipped_indices = []
        
        for idx, entry in enumerate(self.dataset):
            # Extract prompt from the dataset entry
            prompt_text = entry.get("prompt", "").strip()
            
            # Also check conversations format
            if not prompt_text:
                conversations = entry.get("conversations", [])
                for msg in conversations:
                    role = msg.get("role") or msg.get("from")
                    if role in ("user", "human"):
                        prompt_text = (msg.get("content") or msg.get("value", "")).strip()
                        break
            
            if prompt_text in completed_prompts:
                skipped_indices.append(idx)
            else:
                # Keep original index for tracking
                filtered_dataset.append((idx, entry))
        
        return filtered_dataset, skipped_indices
    
    def run(self, resume: bool = False):
        """运行批处理管道。
        
        功能概括：
            执行完整的批处理流程：
            1. 智能恢复（如果启用 resume）
            2. 并行处理所有批次
            3. 合并批次文件为单一轨迹文件
            4. 过滤损坏条目
            5. 生成统计报告
        
        参数：
            resume: 是否从检查点恢复
        
        返回值：
            无（直接运行并输出结果）
        
        主要用于：
            - main() 函数中启动批处理
        
        恢复机制：
            - 基于内容匹配识别已完成的提示
            - 重新创建批次只包含未处理的提示
            - 打印详细的恢复摘要
        
        输出文件：
            - trajectories.jsonl: 合并的轨迹文件（所有批次）
            - batch_*.jsonl: 单个批次文件（调试用）
            - statistics.json: 最终统计
            - checkpoint.json: 检查点数据
        """
        print("\n" + "=" * 70)
        print("🚀 Starting Batch Processing")
        print("=" * 70)
        
        # Smart resume: scan batch files by content to find completed prompts
        completed_prompt_texts = set()
        if resume:
            completed_prompt_texts = self._scan_completed_prompts_by_content()
            if completed_prompt_texts:
                print(f"   Found {len(completed_prompt_texts)} already-completed prompts by content matching")
        
        # Filter dataset to only include unprocessed prompts
        if resume and completed_prompt_texts:
            filtered_entries, skipped_indices = self._filter_dataset_by_completed(completed_prompt_texts)
            
            if not filtered_entries:
                print("\n✅ All prompts have already been processed!")
                return
            
            # Recreate batches from filtered entries (keeping original indices for tracking)
            batches_to_process = []
            for i in range(0, len(filtered_entries), self.batch_size):
                batch = filtered_entries[i:i + self.batch_size]
                batches_to_process.append(batch)
            
            self.batches = batches_to_process
            
            # Print prominent resume summary
            print("\n" + "=" * 70)
            print("📊 RESUME SUMMARY")
            print("=" * 70)
            print(f"   Original dataset size:     {len(self.dataset):,} prompts")
            print(f"   Already completed:         {len(skipped_indices):,} prompts")
            print("   ─────────────────────────────────────────")
            print(f"   🎯 RESUMING WITH:          {len(filtered_entries):,} prompts")
            print(f"   New batches created:       {len(batches_to_process)}")
            print("=" * 70 + "\n")
        
        # Load existing checkpoint (so resume doesn't clobber prior progress)
        checkpoint_data = self._load_checkpoint()
        if checkpoint_data.get("run_name") != self.run_name:
            checkpoint_data = {
                "run_name": self.run_name,
                "completed_prompts": [],
                "batch_stats": {},
                "last_updated": None
            }
        
        # Prepare configuration for workers
        config = {
            "distribution": self.distribution,
            "model": self.model,
            "max_iterations": self.max_iterations,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "verbose": self.verbose,
            "ephemeral_system_prompt": self.ephemeral_system_prompt,
            "log_prefix_chars": self.log_prefix_chars,
            "providers_allowed": self.providers_allowed,
            "providers_ignored": self.providers_ignored,
            "providers_order": self.providers_order,
            "provider_sort": self.provider_sort,
            "max_tokens": self.max_tokens,
            "reasoning_config": self.reasoning_config,
            "prefill_messages": self.prefill_messages,
        }
        
        # For backward compatibility, still track by index (but this is secondary to content matching)
        completed_prompts_set = set(checkpoint_data.get("completed_prompts", []))
        
        # Aggregate statistics across all batches
        total_tool_stats = {}
        
        start_time = time.time()
        
        print(f"\n🔧 Initializing {self.num_workers} worker processes...")
        
        # Checkpoint writes happen in the parent process; keep a lock for safety.
        checkpoint_lock = Lock()

        # Process batches in parallel
        with Pool(processes=self.num_workers) as pool:
            # Create tasks for each batch
            tasks = [
                (
                    batch_num,
                    batch_data,
                    str(self.output_dir),  # Convert Path to string for pickling
                    completed_prompts_set,
                    config
                )
                for batch_num, batch_data in enumerate(self.batches)
            ]
            
            print(f"✅ Created {len(tasks)} batch tasks")
            print("🚀 Starting parallel batch processing...\n")
            
            # Use rich Progress for better visual tracking with persistent bottom bar
            # redirect_stdout/stderr lets rich manage all output so progress bar stays clean
            results = []
            console = Console(force_terminal=True)
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]📦 Batches"),
                BarColumn(bar_width=40),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=console,
                refresh_per_second=2,
                transient=False,
                redirect_stdout=False,
                redirect_stderr=False,
            ) as progress:
                task = progress.add_task("Processing", total=len(tasks))
                
                # Temporarily suppress DEBUG logging to avoid bar interference
                root_logger = logging.getLogger()
                original_level = root_logger.level
                root_logger.setLevel(logging.WARNING)
                
                try:
                    for result in pool.imap_unordered(_process_batch_worker, tasks):
                        results.append(result)
                        progress.update(task, advance=1)

                        # Incremental checkpoint update (so resume works after crash)
                        try:
                            batch_num = result.get('batch_num')
                            completed = result.get('completed_prompts', []) or []
                            completed_prompts_set.update(completed)

                            if isinstance(batch_num, int):
                                checkpoint_data.setdefault('batch_stats', {})[str(batch_num)] = {
                                    'processed': result.get('processed', 0),
                                    'skipped': result.get('skipped', 0),
                                    'discarded_no_reasoning': result.get('discarded_no_reasoning', 0),
                                }

                            checkpoint_data['completed_prompts'] = sorted(completed_prompts_set)
                            self._save_checkpoint(checkpoint_data, lock=checkpoint_lock)
                        except Exception as ckpt_err:
                            # Don't fail the run if checkpoint write fails
                            print(f"⚠️  Warning: Failed to save incremental checkpoint: {ckpt_err}")
                except Exception as e:
                    logger.error("Batch worker failed: %s", e, exc_info=True)
                    raise
                finally:
                    root_logger.setLevel(original_level)
        
        # Aggregate all batch statistics and update checkpoint
        all_completed_prompts = list(completed_prompts_set)
        total_reasoning_stats = {"total_assistant_turns": 0, "turns_with_reasoning": 0, "turns_without_reasoning": 0}
        
        for batch_result in results:
            # Add newly completed prompts
            all_completed_prompts.extend(batch_result.get("completed_prompts", []))
            
            # Aggregate tool stats
            for tool_name, stats in batch_result.get("tool_stats", {}).items():
                if tool_name not in total_tool_stats:
                    total_tool_stats[tool_name] = {
                        "count": 0,
                        "success": 0,
                        "failure": 0
                    }
                
                total_tool_stats[tool_name]["count"] += stats["count"]
                total_tool_stats[tool_name]["success"] += stats["success"]
                total_tool_stats[tool_name]["failure"] += stats["failure"]
            
            # Aggregate reasoning stats
            for key in total_reasoning_stats:
                total_reasoning_stats[key] += batch_result.get("reasoning_stats", {}).get(key, 0)
        
        # Save final checkpoint (best-effort; incremental writes already happened)
        try:
            checkpoint_data["completed_prompts"] = all_completed_prompts
            self._save_checkpoint(checkpoint_data, lock=checkpoint_lock)
        except Exception as ckpt_err:
            print(f"âš ï¸  Warning: Failed to save final checkpoint: {ckpt_err}")
        
        # Calculate success rates
        for tool_name in total_tool_stats:
            stats = total_tool_stats[tool_name]
            total_calls = stats["success"] + stats["failure"]
            if total_calls > 0:
                stats["success_rate"] = round(stats["success"] / total_calls * 100, 2)
                stats["failure_rate"] = round(stats["failure"] / total_calls * 100, 2)
            else:
                stats["success_rate"] = 0.0
                stats["failure_rate"] = 0.0
        
        # Combine ALL batch files in directory into a single trajectories.jsonl file
        # This includes both old batches (from previous runs) and new batches (from resume)
        # Also filter out corrupted entries (where model generated invalid tool names)
        combined_file = self.output_dir / "trajectories.jsonl"
        print(f"\n📦 Combining ALL batch files into {combined_file.name}...")
        
        # Valid tools auto-derived from model_tools.py — no manual updates needed
        VALID_TOOLS = ALL_POSSIBLE_TOOLS
        
        total_entries = 0
        filtered_entries = 0
        batch_files_found = 0
        
        # Find ALL batch files in the output directory (handles resume merging old + new)
        all_batch_files = sorted(self.output_dir.glob("batch_*.jsonl"))
        
        with open(combined_file, 'w', encoding='utf-8') as outfile:
            for batch_file in all_batch_files:
                batch_files_found += 1
                batch_num = batch_file.stem.split("_")[1]  # Extract batch number for logging
                
                with open(batch_file, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        total_entries += 1
                        try:
                            data = json.loads(line)
                            tool_stats = data.get('tool_stats', {})
                            
                            # Check for invalid tool names (model hallucinations)
                            invalid_tools = [k for k in tool_stats if k not in VALID_TOOLS]
                            
                            if invalid_tools:
                                filtered_entries += 1
                                invalid_preview = invalid_tools[0][:50] + "..." if len(invalid_tools[0]) > 50 else invalid_tools[0]
                                print(f"   ⚠️  Filtering corrupted entry (batch {batch_num}): invalid tool '{invalid_preview}'")
                                continue
                            
                            outfile.write(line)
                        except json.JSONDecodeError:
                            filtered_entries += 1
                            print(f"   ⚠️  Filtering invalid JSON entry (batch {batch_num})")
        
        if filtered_entries > 0:
            print(f"⚠️  Filtered {filtered_entries} corrupted entries out of {total_entries} total")
        print(f"✅ Combined {batch_files_found} batch files into trajectories.jsonl ({total_entries - filtered_entries} entries)")
        
        # Save final statistics
        final_stats = {
            "run_name": self.run_name,
            "distribution": self.distribution,
            "total_prompts": len(self.dataset),
            "total_batches": len(self.batches),
            "batch_size": self.batch_size,
            "model": self.model,
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": round(time.time() - start_time, 2),
            "tool_statistics": total_tool_stats,
            "reasoning_statistics": total_reasoning_stats,
        }
        
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 BATCH PROCESSING COMPLETE")
        print("=" * 70)
        print(f"✅ Prompts processed this run: {sum(r.get('processed', 0) for r in results)}")
        print(f"✅ Total trajectories in merged file: {total_entries - filtered_entries}")
        print(f"✅ Total batch files merged: {batch_files_found}")
        print(f"⏱️  Total duration: {round(time.time() - start_time, 2)}s")
        print("\n📈 Tool Usage Statistics:")
        print("-" * 70)
        
        if total_tool_stats:
            # Sort by count descending
            sorted_tools = sorted(
                total_tool_stats.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )
            
            print(f"{'Tool Name':<25} {'Count':<10} {'Success':<10} {'Failure':<10} {'Success Rate':<12}")
            print("-" * 70)
            for tool_name, stats in sorted_tools:
                print(
                    f"{tool_name:<25} "
                    f"{stats['count']:<10} "
                    f"{stats['success']:<10} "
                    f"{stats['failure']:<10} "
                    f"{stats['success_rate']:.1f}%"
                )
        else:
            print("No tool calls were made during this run.")
        
        # Print reasoning coverage stats
        total_discarded = sum(r.get("discarded_no_reasoning", 0) for r in results)
        
        print("\n🧠 Reasoning Coverage:")
        print("-" * 70)
        total_turns = total_reasoning_stats["total_assistant_turns"]
        with_reasoning = total_reasoning_stats["turns_with_reasoning"]
        without_reasoning = total_reasoning_stats["turns_without_reasoning"]
        if total_turns > 0:
            pct_with = round(with_reasoning / total_turns * 100, 1)
            pct_without = round(without_reasoning / total_turns * 100, 1)
            print(f"   Total assistant turns:    {total_turns:,}")
            print(f"   With reasoning:           {with_reasoning:,} ({pct_with}%)")
            print(f"   Without reasoning:        {without_reasoning:,} ({pct_without}%)")
        else:
            print("   No assistant turns recorded.")
        if total_discarded > 0:
            print(f"   🚫 Samples discarded (zero reasoning): {total_discarded:,}")
        
        print(f"\n💾 Results saved to: {self.output_dir}")
        print("   - Trajectories: trajectories.jsonl (combined)")
        print("   - Individual batches: batch_*.jsonl (for debugging)")
        print(f"   - Statistics: {self.stats_file.name}")
        print(f"   - Checkpoint: {self.checkpoint_file.name}")


def main(
    dataset_file: str = None,
    batch_size: int = None,
    run_name: str = None,
    distribution: str = "default",
    model: str = "anthropic/claude-sonnet-4.6",
    api_key: str = None,
    base_url: str = "https://openrouter.ai/api/v1",
    max_turns: int = 10,
    num_workers: int = 4,
    resume: bool = False,
    verbose: bool = False,
    list_distributions: bool = False,
    ephemeral_system_prompt: str = None,
    log_prefix_chars: int = 100,
    providers_allowed: str = None,
    providers_ignored: str = None,
    providers_order: str = None,
    provider_sort: str = None,
    max_tokens: int = None,
    reasoning_effort: str = None,
    reasoning_disabled: bool = False,
    prefill_messages_file: str = None,
    max_samples: int = None,
):
    """从数据集运行 agent 提示的批处理。
    
    功能概括：
        CLI 主入口，解析命令行参数，初始化 BatchRunner 并执行批处理。
        支持列出可用分布、验证参数、配置推理设置和预填消息。
    
    参数：
        dataset_file: JSONL 文件路径，每个条目包含 'prompt' 字段
        batch_size: 每批提示数量
        run_name: 运行名称（用于输出和检查点）
        distribution: 工具集分布名称（默认："default"）
        model: 使用的模型名称（默认："anthropic/claude-sonnet-4.6"）
        api_key: 模型 API 密钥
        base_url: 模型 API 基础 URL
        max_turns: 每个提示的最大工具调用迭代次数（默认：10）
        num_workers: 并行工作进程数（默认：4）
        resume: 是否从检查点恢复中断的运行（默认：False）
        verbose: 是否启用详细日志（默认：False）
        list_distributions: 列出可用工具集分布并退出
        ephemeral_system_prompt: 系统提示（用于 agent 执行但不保存到轨迹）
        log_prefix_chars: 日志预览字符数（默认：100）
        providers_allowed: 逗号分隔的允许 OpenRouter 提供商列表
        providers_ignored: 逗号分隔的忽略 OpenRouter 提供商列表
        providers_order: 逗号分隔的 OpenRouter 提供商尝试顺序
        provider_sort: 按 "price"/"throughput"/"latency" 排序提供商
        max_tokens: 模型响应的最大 token 数
        reasoning_effort: OpenRouter 推理努力级别（"none"/"minimal"/"low"/"medium"/"high"/"xhigh"）
        reasoning_disabled: 完全禁用推理/思考 token（默认：False）
        prefill_messages_file: 预填消息 JSON 文件路径
        max_samples: 仅处理数据集的前 N 个样本
    
    返回值：
        无（直接运行并输出结果）
    
    主要用于：
        - 命令行入口点
        - 大规模训练数据生成
        - 不同配置的对比实验
    
    使用示例：
        # 基本用法
        python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run
        
        # 恢复中断的运行
        python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run --resume
        
        # 使用特定分布
        python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=image_test --distribution=image_gen
        
        # 禁用推理并设置最大 token
        python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run \\
                               --reasoning_disabled --max_tokens=128000
        
        # 从文件加载预填消息
        python batch_runner.py --dataset_file=data.jsonl --batch_size=10 --run_name=my_run \\
                               --prefill_messages_file=configs/prefill_opus.json
        
        # 列出可用分布
        python batch_runner.py --list_distributions
    """
    # Handle list distributions
    if list_distributions:
        from toolset_distributions import list_distributions as get_all_dists, print_distribution_info
        
        print("📊 Available Toolset Distributions")
        print("=" * 70)
        
        all_dists = get_all_dists()
        for dist_name in sorted(all_dists.keys()):
            print_distribution_info(dist_name)
        
        print("\n💡 Usage:")
        print("  python batch_runner.py --dataset_file=data.jsonl --batch_size=10 \\")
        print("                         --run_name=my_run --distribution=<name>")
        return
    
    # Validate required arguments
    if not dataset_file:
        print("❌ Error: --dataset_file is required")
        return
    
    if not batch_size or batch_size < 1:
        print("❌ Error: --batch_size must be a positive integer")
        return
    
    if not run_name:
        print("❌ Error: --run_name is required")
        return
    
    # Parse provider preferences (comma-separated strings to lists)
    providers_allowed_list = [p.strip() for p in providers_allowed.split(",")] if providers_allowed else None
    providers_ignored_list = [p.strip() for p in providers_ignored.split(",")] if providers_ignored else None
    providers_order_list = [p.strip() for p in providers_order.split(",")] if providers_order else None
    
    # Build reasoning_config from CLI flags
    # --reasoning_disabled takes priority, then --reasoning_effort, then default (medium)
    reasoning_config = None
    if reasoning_disabled:
        # Completely disable reasoning/thinking tokens
        reasoning_config = {"effort": "none"}
        print("🧠 Reasoning: DISABLED (effort=none)")
    elif reasoning_effort:
        # Use specified effort level
        valid_efforts = ["none", "minimal", "low", "medium", "high", "xhigh"]
        if reasoning_effort not in valid_efforts:
            print(f"❌ Error: --reasoning_effort must be one of: {', '.join(valid_efforts)}")
            return
        reasoning_config = {"enabled": True, "effort": reasoning_effort}
        print(f"🧠 Reasoning effort: {reasoning_effort}")
    
    # Load prefill messages from JSON file if provided
    prefill_messages = None
    if prefill_messages_file:
        try:
            with open(prefill_messages_file, 'r', encoding='utf-8') as f:
                prefill_messages = json.load(f)
            if not isinstance(prefill_messages, list):
                print("❌ Error: prefill_messages_file must contain a JSON array of messages")
                return
            print(f"💬 Loaded {len(prefill_messages)} prefill messages from {prefill_messages_file}")
        except Exception as e:
            print(f"❌ Error loading prefill messages: {e}")
            return
    
    # Initialize and run batch runner
    try:
        runner = BatchRunner(
            dataset_file=dataset_file,
            batch_size=batch_size,
            run_name=run_name,
            distribution=distribution,
            max_iterations=max_turns,
            base_url=base_url,
            api_key=api_key,
            model=model,
            num_workers=num_workers,
            verbose=verbose,
            ephemeral_system_prompt=ephemeral_system_prompt,
            log_prefix_chars=log_prefix_chars,
            providers_allowed=providers_allowed_list,
            providers_ignored=providers_ignored_list,
            providers_order=providers_order_list,
            provider_sort=provider_sort,
            max_tokens=max_tokens,
            reasoning_config=reasoning_config,
            prefill_messages=prefill_messages,
            max_samples=max_samples,
        )

        runner.run(resume=resume)
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    fire.Fire(main)

