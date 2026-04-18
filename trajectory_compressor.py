#!/usr/bin/env python3
"""Hermes Agent 轨迹压缩器。

后处理已完成的 agent 轨迹，在目标 token 预算内压缩它们，同时保持训练信号质量。

压缩策略：
1. 保护前几个回合（system、human、第一个 gpt、第一个 tool）
2. 保护最后 N 个回合（最终动作和结论）
3. 仅压缩中间回合，从第 2 个 tool 响应开始
4. 仅压缩到刚好符合目标所需的程度
5. 用单个 human 摘要消息替换压缩区域
6. 保持剩余 tool 调用完整（模型在摘要后继续工作）

使用示例：
    # 压缩目录中的 JSONL 文件
    python trajectory_compressor.py --input=data/my_run
    
    # 压缩单个 JSONL 文件
    python trajectory_compressor.py --input=data/trajectories.jsonl
    
    # 压缩文件的 15% 样本
    python trajectory_compressor.py --input=data/trajectories.jsonl --sample_percent=15
    
    # 使用自定义输出和 token 目标压缩
    python trajectory_compressor.py --input=data/trajectories.jsonl --output=compressed.jsonl --target_max_tokens=16000
    
    # 从目录压缩 10% 样本
    python trajectory_compressor.py --input=data/my_run --sample_percent=10

主要用途：
    - 减少长轨迹的上下文长度以适应模型限制
    - 优化训练数据质量（去除冗余中间步骤）
    - 降低训练成本
"""

import json
import os
import time
import yaml
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import fire
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from hermes_constants import OPENROUTER_BASE_URL, get_hermes_home
from agent.retry_utils import jittered_backoff

# Load .env from HERMES_HOME first, then project root as a dev fallback.
from hermes_cli.env_loader import load_hermes_dotenv

_hermes_home = get_hermes_home()
_project_env = Path(__file__).parent / ".env"
load_hermes_dotenv(hermes_home=_hermes_home, project_env=_project_env)


@dataclass
class CompressionConfig:
    """轨迹压缩配置。
    
    功能概括：
        存储压缩器的所有配置参数，包括分词器、压缩目标、
        保护回合设置、摘要生成、输出格式和处理选项。
        支持从 YAML 文件加载配置。
    
    主要用途：
        - TrajectoryCompressor 初始化时传入配置
        - main() 函数中从 CLI 参数和 YAML 文件构建配置
    
    关键配置项：
        - target_max_tokens: 目标最大 token 数（默认 15250）
        - summary_target_tokens: 摘要目标 token 数（默认 750）
        - protect_last_n_turns: 保护的末尾回合数（默认 4）
        - max_concurrent_requests: 最大并发 API 调用数（默认 50）
    """
    # Tokenizer
    tokenizer_name: str = "moonshotai/Kimi-K2-Thinking"
    trust_remote_code: bool = True
    
    # Compression targets
    target_max_tokens: int = 15250
    summary_target_tokens: int = 750
    
    # Protected turns
    protect_first_system: bool = True
    protect_first_human: bool = True
    protect_first_gpt: bool = True
    protect_first_tool: bool = True
    protect_last_n_turns: int = 4
    
    # Summarization (OpenRouter)
    summarization_model: str = "google/gemini-3-flash-preview"
    base_url: str = OPENROUTER_BASE_URL
    api_key_env: str = "OPENROUTER_API_KEY"
    temperature: float = 0.3
    max_retries: int = 3
    retry_delay: int = 2
    
    # Output
    add_summary_notice: bool = True
    summary_notice_text: str = "\n\nSome of your previous tool responses may be summarized to preserve context."
    output_suffix: str = "_compressed"
    
    # Processing
    num_workers: int = 4
    max_concurrent_requests: int = 50  # Max concurrent API calls for summarization
    skip_under_target: bool = True
    save_over_limit: bool = True
    per_trajectory_timeout: int = 300  # Timeout per trajectory in seconds (default: 5 min)
    
    # Metrics
    metrics_enabled: bool = True
    metrics_per_trajectory: bool = True
    metrics_output_file: str = "compression_metrics.json"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "CompressionConfig":
        """从 YAML 文件加载配置。
        
        功能概括：
            读取 YAML 配置文件，解析各配置节并更新默认配置。
            支持 tokenizer、compression、protected_turns、summarization、
            output、processing、metrics 等配置节。
        
        参数：
            yaml_path: YAML 配置文件路径
        
        返回值：
            CompressionConfig: 加载的配置对象
        
        主要用于：
            - main() 函数中从配置文件加载配置
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        # Tokenizer
        if 'tokenizer' in data:
            config.tokenizer_name = data['tokenizer'].get('name', config.tokenizer_name)
            config.trust_remote_code = data['tokenizer'].get('trust_remote_code', config.trust_remote_code)
        
        # Compression
        if 'compression' in data:
            config.target_max_tokens = data['compression'].get('target_max_tokens', config.target_max_tokens)
            config.summary_target_tokens = data['compression'].get('summary_target_tokens', config.summary_target_tokens)
        
        # Protected turns
        if 'protected_turns' in data:
            config.protect_first_system = data['protected_turns'].get('first_system', config.protect_first_system)
            config.protect_first_human = data['protected_turns'].get('first_human', config.protect_first_human)
            config.protect_first_gpt = data['protected_turns'].get('first_gpt', config.protect_first_gpt)
            config.protect_first_tool = data['protected_turns'].get('first_tool', config.protect_first_tool)
            config.protect_last_n_turns = data['protected_turns'].get('last_n_turns', config.protect_last_n_turns)
        
        # Summarization
        if 'summarization' in data:
            config.summarization_model = data['summarization'].get('model', config.summarization_model)
            config.base_url = data['summarization'].get('base_url') or config.base_url
            config.api_key_env = data['summarization'].get('api_key_env', config.api_key_env)
            config.temperature = data['summarization'].get('temperature', config.temperature)
            config.max_retries = data['summarization'].get('max_retries', config.max_retries)
            config.retry_delay = data['summarization'].get('retry_delay', config.retry_delay)
        
        # Output
        if 'output' in data:
            config.add_summary_notice = data['output'].get('add_summary_notice', config.add_summary_notice)
            config.summary_notice_text = data['output'].get('summary_notice_text', config.summary_notice_text)
            config.output_suffix = data['output'].get('output_suffix', config.output_suffix)
        
        # Processing
        if 'processing' in data:
            config.num_workers = data['processing'].get('num_workers', config.num_workers)
            config.max_concurrent_requests = data['processing'].get('max_concurrent_requests', config.max_concurrent_requests)
            config.skip_under_target = data['processing'].get('skip_under_target', config.skip_under_target)
            config.save_over_limit = data['processing'].get('save_over_limit', config.save_over_limit)
        
        # Metrics
        if 'metrics' in data:
            config.metrics_enabled = data['metrics'].get('enabled', config.metrics_enabled)
            config.metrics_per_trajectory = data['metrics'].get('per_trajectory', config.metrics_per_trajectory)
            config.metrics_output_file = data['metrics'].get('output_file', config.metrics_output_file)
        
        return config


@dataclass
class TrajectoryMetrics:
    """单个轨迹压缩的指标。
    
    功能概括：
        记录单个轨迹压缩前后的 token 数、回合数、压缩比例、
        API 调用次数等详细指标。
    
    主要用途：
        - compress_trajectory() 返回压缩结果时附带指标
        - AggregateMetrics 聚合所有轨迹的指标
        - 输出到 compression_metrics.json 供分析
    
    关键字段：
        - original_tokens/compressed_tokens: 压缩前后 token 数
        - compression_ratio: 压缩比例（压缩后/压缩前）
        - was_compressed: 是否进行了压缩
        - still_over_limit: 压缩后是否仍超限
    """
    original_tokens: int = 0
    compressed_tokens: int = 0
    tokens_saved: int = 0
    compression_ratio: float = 1.0
    
    original_turns: int = 0
    compressed_turns: int = 0
    turns_removed: int = 0
    
    turns_compressed_start_idx: int = -1
    turns_compressed_end_idx: int = -1
    turns_in_compressed_region: int = 0
    
    was_compressed: bool = False
    still_over_limit: bool = False
    skipped_under_target: bool = False
    
    summarization_api_calls: int = 0
    summarization_errors: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "tokens_saved": self.tokens_saved,
            "compression_ratio": round(self.compression_ratio, 4),
            "original_turns": self.original_turns,
            "compressed_turns": self.compressed_turns,
            "turns_removed": self.turns_removed,
            "compression_region": {
                "start_idx": self.turns_compressed_start_idx,
                "end_idx": self.turns_compressed_end_idx,
                "turns_count": self.turns_in_compressed_region,
            },
            "was_compressed": self.was_compressed,
            "still_over_limit": self.still_over_limit,
            "skipped_under_target": self.skipped_under_target,
            "summarization_api_calls": self.summarization_api_calls,
            "summarization_errors": self.summarization_errors,
        }


@dataclass 
class AggregateMetrics:
    """所有轨迹的聚合指标。
    
    功能概括：
        聚合所有轨迹的压缩指标，计算总体统计数据如总 token 节省、
        平均压缩比例、API 调用成功率等。
    
    主要用途：
        - TrajectoryCompressor 处理目录时累积指标
        - 最终输出压缩报告
        - 保存到 metrics JSON 文件
    
    关键方法：
        - add_trajectory_metrics(): 添加单个轨迹的指标
        - to_dict(): 转换为字典格式用于输出
    """
    total_trajectories: int = 0
    trajectories_compressed: int = 0
    trajectories_skipped_under_target: int = 0
    trajectories_still_over_limit: int = 0
    trajectories_failed: int = 0
    
    total_tokens_before: int = 0
    total_tokens_after: int = 0
    total_tokens_saved: int = 0
    
    total_turns_before: int = 0
    total_turns_after: int = 0
    total_turns_removed: int = 0
    
    total_summarization_calls: int = 0
    total_summarization_errors: int = 0
    
    # Distribution stats
    compression_ratios: List[float] = field(default_factory=list)
    tokens_saved_list: List[int] = field(default_factory=list)
    turns_removed_list: List[int] = field(default_factory=list)
    
    processing_start_time: str = ""
    processing_end_time: str = ""
    processing_duration_seconds: float = 0.0
    
    def add_trajectory_metrics(self, metrics: TrajectoryMetrics):
        """Add a trajectory's metrics to the aggregate."""
        self.total_trajectories += 1
        self.total_tokens_before += metrics.original_tokens
        self.total_tokens_after += metrics.compressed_tokens
        self.total_tokens_saved += metrics.tokens_saved
        self.total_turns_before += metrics.original_turns
        self.total_turns_after += metrics.compressed_turns
        self.total_turns_removed += metrics.turns_removed
        self.total_summarization_calls += metrics.summarization_api_calls
        self.total_summarization_errors += metrics.summarization_errors
        
        if metrics.was_compressed:
            self.trajectories_compressed += 1
            self.compression_ratios.append(metrics.compression_ratio)
            self.tokens_saved_list.append(metrics.tokens_saved)
            self.turns_removed_list.append(metrics.turns_removed)
        
        if metrics.skipped_under_target:
            self.trajectories_skipped_under_target += 1
        
        if metrics.still_over_limit:
            self.trajectories_still_over_limit += 1
    
    def to_dict(self) -> Dict[str, Any]:
        avg_compression_ratio = (
            sum(self.compression_ratios) / len(self.compression_ratios) 
            if self.compression_ratios else 1.0
        )
        avg_tokens_saved = (
            sum(self.tokens_saved_list) / len(self.tokens_saved_list)
            if self.tokens_saved_list else 0
        )
        avg_turns_removed = (
            sum(self.turns_removed_list) / len(self.turns_removed_list)
            if self.turns_removed_list else 0
        )
        
        return {
            "summary": {
                "total_trajectories": self.total_trajectories,
                "trajectories_compressed": self.trajectories_compressed,
                "trajectories_skipped_under_target": self.trajectories_skipped_under_target,
                "trajectories_still_over_limit": self.trajectories_still_over_limit,
                "trajectories_failed": self.trajectories_failed,
                "compression_rate": round(self.trajectories_compressed / max(self.total_trajectories, 1), 4),
            },
            "tokens": {
                "total_before": self.total_tokens_before,
                "total_after": self.total_tokens_after,
                "total_saved": self.total_tokens_saved,
                "overall_compression_ratio": round(self.total_tokens_after / max(self.total_tokens_before, 1), 4),
            },
            "turns": {
                "total_before": self.total_turns_before,
                "total_after": self.total_turns_after,
                "total_removed": self.total_turns_removed,
            },
            "averages": {
                "avg_compression_ratio": round(avg_compression_ratio, 4),
                "avg_tokens_saved_per_compressed": round(avg_tokens_saved, 1),
                "avg_turns_removed_per_compressed": round(avg_turns_removed, 2),
            },
            "summarization": {
                "total_api_calls": self.total_summarization_calls,
                "total_errors": self.total_summarization_errors,
                "success_rate": round(1 - (self.total_summarization_errors / max(self.total_summarization_calls, 1)), 4),
            },
            "processing": {
                "start_time": self.processing_start_time,
                "end_time": self.processing_end_time,
                "duration_seconds": round(self.processing_duration_seconds, 2),
            },
        }


class TrajectoryCompressor:
    """压缩 agent 轨迹以适应目标 token 预算。
    
    功能概括：
        实现轨迹压缩的核心逻辑：
        1. 保持受保护的头部回合（system、human、第一个 gpt+tool）
        2. 保持受保护的尾部回合（最后 N 个回合）
        3. 从可压缩的中间区域，仅压缩所需的部分
        4. 用单个 human 摘要消息替换压缩的回合
        5. 保持剩余的中间回合完整（模型继续使用工具）
    
    主要用途：
        - main() 函数中处理单个文件或目录
        - 批量压缩训练数据
        - 优化长轨迹的上下文长度
    
    压缩算法：
        1. 计算总 token 数
        2. 如果低于目标，跳过
        3. 找到可压缩区域（受保护的头部和尾部之间）
        4. 计算需要保存的 token 数
        5. 从可压缩区域开始累积回合，直到满足节省需求
        6. 用 LLM 生成摘要替换累积的回合
        7. 保持剩余回合完整
    """
    
    def __init__(self, config: CompressionConfig):
        """初始化压缩器。
        
        功能概括：
            配置压缩器，初始化分词器和摘要生成客户端。
        
        参数：
            config: 压缩配置对象
        
        返回值：
            无
        
        主要用于：
            - main() 函数中创建压缩器实例
        """
        self.config = config
        self.aggregate_metrics = AggregateMetrics()
        
        # Initialize tokenizer
        self._init_tokenizer()
        
        # Initialize OpenRouter client
        self._init_summarizer()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_tokenizer(self):
        """初始化 HuggingFace 分词器用于 token 计数。
        
        功能概括：
            加载指定的预训练分词器，用于准确计算文本 token 数。
        
        参数：
            无
        
        返回值：
            无（设置 self.tokenizer）
        
        主要用于：
            - __init__() 中初始化分词器
        
        错误处理：
            - 分词器加载失败时抛出 RuntimeError
        """
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_name,
                trust_remote_code=self.config.trust_remote_code
            )
            print(f"✅ Loaded tokenizer: {self.config.tokenizer_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer '{self.config.tokenizer_name}': {e}")
    
    def _init_summarizer(self):
        """初始化 LLM 路由用于摘要生成（同步和异步）。

        功能概括：
            配置摘要生成的 LLM 客户端，支持集中式提供商路由器
            或自定义端点。处理认证、headers 和提供商检测。
        
        参数：
            无
        
        返回值：
            无（设置 self.client、self._use_call_llm 等）
        
        主要用于：
            - __init__() 中初始化摘要客户端
        
        客户端选择：
            - 已知提供商（OpenRouter、Nous 等）：使用 call_llm/async_call_llm
            - 自定义端点：直接构造 OpenAI 客户端
        """

        provider = self._detect_provider()
        if provider:
            # Store provider for use in _generate_summary calls
            self._llm_provider = provider
            self._use_call_llm = True
            # Verify the provider is available
            from agent.auxiliary_client import resolve_provider_client
            client, _ = resolve_provider_client(
                provider, model=self.config.summarization_model)
            if client is None:
                raise RuntimeError(
                    f"Provider '{provider}' is not configured. "
                    f"Check your API key or run: hermes setup")
            self.client = None  # Not used directly
            self.async_client = None  # Not used directly
        else:
            # Custom endpoint — use config's raw base_url + api_key_env
            self._use_call_llm = False
            api_key = os.getenv(self.config.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Missing API key. Set {self.config.api_key_env} "
                    f"environment variable.")
            from openai import OpenAI
            from agent.auxiliary_client import _to_openai_base_url
            self.client = OpenAI(
                api_key=api_key, base_url=_to_openai_base_url(self.config.base_url))
            # AsyncOpenAI is created lazily in _get_async_client() so it
            # binds to the current event loop — avoids "Event loop is closed"
            # when process_directory() is called multiple times (each call
            # creates a new loop via asyncio.run()).
            self.async_client = None
            self._async_client_api_key = api_key

        print(f"✅ Initialized summarizer client: {self.config.summarization_model}")
        print(f"   Max concurrent requests: {self.config.max_concurrent_requests}")

    def _get_async_client(self):
        """Return an AsyncOpenAI client bound to the current event loop.

        Created lazily so that each ``asyncio.run()`` call in
        ``process_directory()`` gets a client tied to its own loop,
        avoiding "Event loop is closed" errors on repeated calls.
        """
        from openai import AsyncOpenAI
        from agent.auxiliary_client import _to_openai_base_url
        # Always create a fresh client so it binds to the running loop.
        self.async_client = AsyncOpenAI(
            api_key=self._async_client_api_key,
            base_url=_to_openai_base_url(self.config.base_url),
        )
        return self.async_client

    def _detect_provider(self) -> str:
        """从配置的 base_url 检测提供商名称。
        
        功能概括：
            根据 base_url 识别 LLM 提供商（如 openrouter、nous 等）。
        
        参数：
            无
        
        返回值：
            str: 提供商名称，未知返回空字符串
        
        主要用于：
            - _init_summarizer() 中确定使用哪种客户端
        """
        url = (self.config.base_url or "").lower()
        if "openrouter" in url:
            return "openrouter"
        if "nousresearch.com" in url:
            return "nous"
        if "chatgpt.com/backend-api/codex" in url:
            return "codex"
        if "api.z.ai" in url:
            return "zai"
        if "moonshot.ai" in url or "moonshot.cn" in url or "api.kimi.com" in url:
            return "kimi-coding"
        if "arcee.ai" in url:
            return "arcee"
        if "minimaxi.com" in url:
            return "minimax-cn"
        if "minimax.io" in url:
            return "minimax"
        # Unknown base_url — not a known provider
        return ""
    
    def count_tokens(self, text: str) -> int:
        """使用配置的分词器计算文本中的 token 数。
        
        功能概括：
            调用分词器的 encode 方法计算 token 数。
            如果失败则回退到字符数估算（每 4 个字符约 1 个 token）。
        
        参数：
            text: 要计算 token 的文本
        
        返回值：
            int: token 数量
        
        主要用于：
            - count_trajectory_tokens() 计算轨迹总 token
            - count_turn_tokens() 计算每个回合的 token
        """
        if not text:
            return 0
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            # Fallback to character estimate
            return len(text) // 4
    
    def count_trajectory_tokens(self, trajectory: List[Dict[str, str]]) -> int:
        """计算轨迹中的总 token 数。
        
        功能概括：
            遍历轨迹的所有回合，累加每个回合的 token 数。
        
        参数：
            trajectory: 轨迹消息列表
        
        返回值：
            int: 总 token 数
        
        主要用于：
            - compress_trajectory() 判断是否需要压缩
        """
        return sum(self.count_tokens(turn.get("value", "")) for turn in trajectory)
    
    def count_turn_tokens(self, trajectory: List[Dict[str, str]]) -> List[int]:
        """计算轨迹中每个回合的 token 数。
        
        功能概括：
            返回每个回合的 token 数列表，用于确定压缩哪些回合。
        
        参数：
            trajectory: 轨迹消息列表
        
        返回值：
            List[int]: 每个回合的 token 数列表
        
        主要用于：
            - compress_trajectory() 中确定压缩区域
        """
        return [self.count_tokens(turn.get("value", "")) for turn in trajectory]
    
    def _find_protected_indices(self, trajectory: List[Dict[str, str]]) -> Tuple[set, int, int]:
        """
        Find indices of protected turns.
        
        Returns:
            Tuple of (protected_set, compressible_start, compressible_end)
        """
        n = len(trajectory)
        protected = set()
        
        # Track first occurrences
        first_system = first_human = first_gpt = first_tool = None
        
        for i, turn in enumerate(trajectory):
            role = turn.get("from", "")
            if role == "system" and first_system is None:
                first_system = i
            elif role == "human" and first_human is None:
                first_human = i
            elif role == "gpt" and first_gpt is None:
                first_gpt = i
            elif role == "tool" and first_tool is None:
                first_tool = i
        
        # Protect first turns
        if self.config.protect_first_system and first_system is not None:
            protected.add(first_system)
        if self.config.protect_first_human and first_human is not None:
            protected.add(first_human)
        if self.config.protect_first_gpt and first_gpt is not None:
            protected.add(first_gpt)
        if self.config.protect_first_tool and first_tool is not None:
            protected.add(first_tool)
        
        # Protect last N turns
        for i in range(max(0, n - self.config.protect_last_n_turns), n):
            protected.add(i)
        
        # Determine compressible region
        # Start after the last protected head turn
        head_protected = [i for i in protected if i < n // 2]
        tail_protected = [i for i in protected if i >= n // 2]
        
        compressible_start = max(head_protected) + 1 if head_protected else 0
        compressible_end = min(tail_protected) if tail_protected else n
        
        return protected, compressible_start, compressible_end
    
    def _extract_turn_content_for_summary(self, trajectory: List[Dict[str, str]], start: int, end: int) -> str:
        """
        Extract content from turns to be summarized.
        
        Args:
            trajectory: Full trajectory
            start: Start index (inclusive)
            end: End index (exclusive)
            
        Returns:
            Formatted string of turn contents for summarization
        """
        parts = []
        for i in range(start, end):
            turn = trajectory[i]
            role = turn.get("from", "unknown")
            value = turn.get("value", "")
            
            # Truncate very long values for the summary prompt
            if len(value) > 3000:
                value = value[:1500] + "\n...[truncated]...\n" + value[-500:]
            
            parts.append(f"[Turn {i} - {role.upper()}]:\n{value}")
        
        return "\n\n".join(parts)

    @staticmethod
    def _coerce_summary_content(content: Any) -> str:
        """Normalize summary-model output to a safe string."""
        if not isinstance(content, str):
            content = str(content) if content else ""
        return content.strip()

    @staticmethod
    def _ensure_summary_prefix(summary: str) -> str:
        """Normalize summary text to include the expected prefix exactly once."""
        text = (summary or "").strip()
        if text.startswith("[CONTEXT SUMMARY]:"):
            return text
        return "[CONTEXT SUMMARY]:" if not text else f"[CONTEXT SUMMARY]: {text}"
    
    def _generate_summary(self, content: str, metrics: TrajectoryMetrics) -> str:
        """
        Generate a summary of the compressed turns using OpenRouter.
        
        Args:
            content: The content to summarize
            metrics: Metrics object to update
            
        Returns:
            Summary string
        """
        prompt = f"""Summarize the following agent conversation turns concisely. This summary will replace these turns in the conversation history.

Write the summary from a neutral perspective describing what the assistant did and learned. Include:
1. What actions the assistant took (tool calls, searches, file operations)
2. Key information or results obtained
3. Any important decisions or findings
4. Relevant data, file names, values, or outputs

Keep the summary factual and informative. Target approximately {self.config.summary_target_tokens} tokens.

---
TURNS TO SUMMARIZE:
{content}
---

Write only the summary, starting with "[CONTEXT SUMMARY]:" prefix."""

        for attempt in range(self.config.max_retries):
            try:
                metrics.summarization_api_calls += 1
                
                if getattr(self, '_use_call_llm', False):
                    from agent.auxiliary_client import call_llm
                    response = call_llm(
                        provider=self._llm_provider,
                        model=self.config.summarization_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.temperature,
                        max_tokens=self.config.summary_target_tokens * 2,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.config.summarization_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.temperature,
                        max_tokens=self.config.summary_target_tokens * 2,
                    )
                
                summary = self._coerce_summary_content(response.choices[0].message.content)
                return self._ensure_summary_prefix(summary)
                
            except Exception as e:
                metrics.summarization_errors += 1
                self.logger.warning(f"Summarization attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retries - 1:
                    time.sleep(jittered_backoff(attempt + 1, base_delay=self.config.retry_delay, max_delay=30.0))
                else:
                    # Fallback: create a basic summary
                    return "[CONTEXT SUMMARY]: [Summary generation failed - previous turns contained tool calls and responses that have been compressed to save context space.]"
    
    async def _generate_summary_async(self, content: str, metrics: TrajectoryMetrics) -> str:
        """
        Generate a summary of the compressed turns using OpenRouter (async version).
        
        Args:
            content: The content to summarize
            metrics: Metrics object to update
            
        Returns:
            Summary string
        """
        prompt = f"""Summarize the following agent conversation turns concisely. This summary will replace these turns in the conversation history.

Write the summary from a neutral perspective describing what the assistant did and learned. Include:
1. What actions the assistant took (tool calls, searches, file operations)
2. Key information or results obtained
3. Any important decisions or findings
4. Relevant data, file names, values, or outputs

Keep the summary factual and informative. Target approximately {self.config.summary_target_tokens} tokens.

---
TURNS TO SUMMARIZE:
{content}
---

Write only the summary, starting with "[CONTEXT SUMMARY]:" prefix."""

        for attempt in range(self.config.max_retries):
            try:
                metrics.summarization_api_calls += 1
                
                if getattr(self, '_use_call_llm', False):
                    from agent.auxiliary_client import async_call_llm
                    response = await async_call_llm(
                        provider=self._llm_provider,
                        model=self.config.summarization_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.temperature,
                        max_tokens=self.config.summary_target_tokens * 2,
                    )
                else:
                    response = await self._get_async_client().chat.completions.create(
                        model=self.config.summarization_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.temperature,
                        max_tokens=self.config.summary_target_tokens * 2,
                    )
                
                summary = self._coerce_summary_content(response.choices[0].message.content)
                return self._ensure_summary_prefix(summary)
                
            except Exception as e:
                metrics.summarization_errors += 1
                self.logger.warning(f"Summarization attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(jittered_backoff(attempt + 1, base_delay=self.config.retry_delay, max_delay=30.0))
                else:
                    # Fallback: create a basic summary
                    return "[CONTEXT SUMMARY]: [Summary generation failed - previous turns contained tool calls and responses that have been compressed to save context space.]"
    
    def compress_trajectory(
        self,
        trajectory: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], TrajectoryMetrics]:
        """
        Compress a single trajectory to fit within target token budget.
        
        Algorithm:
        1. Count total tokens
        2. If under target, skip
        3. Find compressible region (between protected head and tail)
        4. Calculate how many tokens need to be saved
        5. Accumulate turns from start of compressible region until savings met
        6. Replace accumulated turns with single human summary
        7. Keep remaining turns intact
        
        Args:
            trajectory: List of conversation turns
            
        Returns:
            Tuple of (compressed_trajectory, metrics)
        """
        metrics = TrajectoryMetrics()
        metrics.original_turns = len(trajectory)
        
        # Count tokens per turn
        turn_tokens = self.count_turn_tokens(trajectory)
        total_tokens = sum(turn_tokens)
        metrics.original_tokens = total_tokens
        
        # Check if compression needed
        if total_tokens <= self.config.target_max_tokens:
            metrics.skipped_under_target = True
            metrics.compressed_tokens = total_tokens
            metrics.compressed_turns = len(trajectory)
            metrics.compression_ratio = 1.0
            return trajectory, metrics
        
        # Find protected regions
        protected, compress_start, compress_end = self._find_protected_indices(trajectory)
        
        # Check if there's anything to compress
        if compress_start >= compress_end:
            # Nothing to compress, return as-is
            metrics.compressed_tokens = total_tokens
            metrics.compressed_turns = len(trajectory)
            metrics.still_over_limit = total_tokens > self.config.target_max_tokens
            return trajectory, metrics
        
        # Calculate how much we need to save
        tokens_to_save = total_tokens - self.config.target_max_tokens
        
        # We'll replace N turns with 1 summary turn
        # Net savings = (sum of N turns' tokens) - summary_target_tokens
        # We need: net_savings >= tokens_to_save
        # So: sum of turns >= tokens_to_save + summary_target_tokens
        target_tokens_to_compress = tokens_to_save + self.config.summary_target_tokens
        
        # Accumulate turns from compress_start until we have enough savings
        accumulated_tokens = 0
        compress_until = compress_start
        
        for i in range(compress_start, compress_end):
            accumulated_tokens += turn_tokens[i]
            compress_until = i + 1  # Exclusive end
            
            # Check if we have enough savings
            if accumulated_tokens >= target_tokens_to_compress:
                break
        
        # If we still don't have enough savings, compress the entire compressible region
        if accumulated_tokens < target_tokens_to_compress and compress_until < compress_end:
            compress_until = compress_end
            accumulated_tokens = sum(turn_tokens[compress_start:compress_end])
        
        # Record compression region
        metrics.turns_compressed_start_idx = compress_start
        metrics.turns_compressed_end_idx = compress_until
        metrics.turns_in_compressed_region = compress_until - compress_start
        
        # Extract content for summary
        content_to_summarize = self._extract_turn_content_for_summary(
            trajectory, compress_start, compress_until
        )
        
        # Generate summary
        summary = self._generate_summary(content_to_summarize, metrics)
        
        # Build compressed trajectory
        compressed = []
        
        # Add head (turns before compression region)
        for i in range(compress_start):
            turn = trajectory[i].copy()
            # Add notice to system message
            if turn.get("from") == "system" and self.config.add_summary_notice:
                turn["value"] = turn["value"] + self.config.summary_notice_text
            compressed.append(turn)
        
        # Add summary as human message
        compressed.append({
            "from": "human",
            "value": summary
        })
        
        # Add tail (turns after compression region)
        for i in range(compress_until, len(trajectory)):
            compressed.append(trajectory[i].copy())
        
        # Calculate final metrics
        metrics.compressed_turns = len(compressed)
        metrics.compressed_tokens = self.count_trajectory_tokens(compressed)
        metrics.turns_removed = metrics.original_turns - metrics.compressed_turns
        metrics.tokens_saved = metrics.original_tokens - metrics.compressed_tokens
        metrics.compression_ratio = metrics.compressed_tokens / max(metrics.original_tokens, 1)
        metrics.was_compressed = True
        metrics.still_over_limit = metrics.compressed_tokens > self.config.target_max_tokens
        
        return compressed, metrics
    
    async def compress_trajectory_async(
        self,
        trajectory: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], TrajectoryMetrics]:
        """
        Compress a single trajectory to fit within target token budget (async version).
        
        Same algorithm as compress_trajectory but uses async API calls for summarization.
        """
        metrics = TrajectoryMetrics()
        metrics.original_turns = len(trajectory)
        
        # Count tokens per turn
        turn_tokens = self.count_turn_tokens(trajectory)
        total_tokens = sum(turn_tokens)
        metrics.original_tokens = total_tokens
        
        # Check if compression needed
        if total_tokens <= self.config.target_max_tokens:
            metrics.skipped_under_target = True
            metrics.compressed_tokens = total_tokens
            metrics.compressed_turns = len(trajectory)
            metrics.compression_ratio = 1.0
            return trajectory, metrics
        
        # Find protected regions
        protected, compress_start, compress_end = self._find_protected_indices(trajectory)
        
        # Check if there's anything to compress
        if compress_start >= compress_end:
            metrics.compressed_tokens = total_tokens
            metrics.compressed_turns = len(trajectory)
            metrics.still_over_limit = total_tokens > self.config.target_max_tokens
            return trajectory, metrics
        
        # Calculate how much we need to save
        tokens_to_save = total_tokens - self.config.target_max_tokens
        target_tokens_to_compress = tokens_to_save + self.config.summary_target_tokens
        
        # Accumulate turns from compress_start until we have enough savings
        accumulated_tokens = 0
        compress_until = compress_start
        
        for i in range(compress_start, compress_end):
            accumulated_tokens += turn_tokens[i]
            compress_until = i + 1
            if accumulated_tokens >= target_tokens_to_compress:
                break
        
        # If we still don't have enough savings, compress the entire compressible region
        if accumulated_tokens < target_tokens_to_compress and compress_until < compress_end:
            compress_until = compress_end
            accumulated_tokens = sum(turn_tokens[compress_start:compress_end])
        
        # Record compression region
        metrics.turns_compressed_start_idx = compress_start
        metrics.turns_compressed_end_idx = compress_until
        metrics.turns_in_compressed_region = compress_until - compress_start
        
        # Extract content for summary
        content_to_summarize = self._extract_turn_content_for_summary(
            trajectory, compress_start, compress_until
        )
        
        # Generate summary (ASYNC)
        summary = await self._generate_summary_async(content_to_summarize, metrics)
        
        # Build compressed trajectory
        compressed = []
        
        # Add head (turns before compression region)
        for i in range(compress_start):
            turn = trajectory[i].copy()
            if turn.get("from") == "system" and self.config.add_summary_notice:
                turn["value"] = turn["value"] + self.config.summary_notice_text
            compressed.append(turn)
        
        # Add summary as human message
        compressed.append({
            "from": "human",
            "value": summary
        })
        
        # Add tail (turns after compression region)
        for i in range(compress_until, len(trajectory)):
            compressed.append(trajectory[i].copy())
        
        # Calculate final metrics
        metrics.compressed_turns = len(compressed)
        metrics.compressed_tokens = self.count_trajectory_tokens(compressed)
        metrics.turns_removed = metrics.original_turns - metrics.compressed_turns
        metrics.tokens_saved = metrics.original_tokens - metrics.compressed_tokens
        metrics.compression_ratio = metrics.compressed_tokens / max(metrics.original_tokens, 1)
        metrics.was_compressed = True
        metrics.still_over_limit = metrics.compressed_tokens > self.config.target_max_tokens
        
        return compressed, metrics
    
    async def process_entry_async(self, entry: Dict[str, Any]) -> Tuple[Dict[str, Any], TrajectoryMetrics]:
        """
        Process a single JSONL entry (async version).
        """
        if "conversations" not in entry:
            metrics = TrajectoryMetrics()
            return entry, metrics
        
        trajectory = entry["conversations"]
        compressed_trajectory, metrics = await self.compress_trajectory_async(trajectory)
        
        # Create new entry with compressed trajectory
        result = entry.copy()
        result["conversations"] = compressed_trajectory
        
        # Add compression metadata if enabled
        if self.config.metrics_per_trajectory and metrics.was_compressed:
            result["compression_metrics"] = metrics.to_dict()
        
        return result, metrics
    
    def process_entry(self, entry: Dict[str, Any]) -> Tuple[Dict[str, Any], TrajectoryMetrics]:
        """
        Process a single JSONL entry.
        
        Args:
            entry: JSONL entry containing 'conversations' field
            
        Returns:
            Tuple of (processed_entry, metrics)
        """
        if "conversations" not in entry:
            metrics = TrajectoryMetrics()
            return entry, metrics
        
        trajectory = entry["conversations"]
        compressed_trajectory, metrics = self.compress_trajectory(trajectory)
        
        # Create new entry with compressed trajectory
        result = entry.copy()
        result["conversations"] = compressed_trajectory
        
        # Add compression metadata if enabled
        if self.config.metrics_per_trajectory and metrics.was_compressed:
            result["compression_metrics"] = metrics.to_dict()
        
        return result, metrics
    
    def process_directory(self, input_dir: Path, output_dir: Path):
        """
        Process all JSONL files in a directory using async parallel processing.
        
        Args:
            input_dir: Input directory containing JSONL files
            output_dir: Output directory for compressed files
        """
        # Run the async version
        asyncio.run(self._process_directory_async(input_dir, output_dir))
    
    async def _process_directory_async(self, input_dir: Path, output_dir: Path):
        """
        Async implementation of directory processing with parallel API calls.
        """
        console = Console()
        
        # Record start time
        self.aggregate_metrics.processing_start_time = datetime.now().isoformat()
        start_time = time.time()
        
        # Find all JSONL files
        jsonl_files = sorted(input_dir.glob("*.jsonl"))
        
        if not jsonl_files:
            self.logger.warning(f"No JSONL files found in {input_dir}")
            return
        
        # Load ALL entries from all files
        console.print("\n[dim]Loading all entries...[/dim]")
        all_entries = []  # List of (file_path, entry_idx, entry)
        
        for file_path in jsonl_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            all_entries.append((file_path, line_num, entry))
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Skipping invalid JSON at {file_path}:{line_num}: {e}")
        
        total_entries = len(all_entries)
        
        console.print(f"\n{'='*60}")
        console.print(f"📂 Input: {input_dir}")
        console.print(f"📂 Output: {output_dir}")
        console.print(f"📄 Files to process: {len(jsonl_files)}")
        console.print(f"📊 Total trajectories: {total_entries:,}")
        console.print(f"🎯 Target max tokens: {self.config.target_max_tokens:,}")
        console.print(f"📝 Summary target tokens: {self.config.summary_target_tokens}")
        console.print(f"⚡ Max concurrent API calls: {self.config.max_concurrent_requests}")
        console.print(f"{'='*60}\n")
        
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # Tracking for progress display (thread-safe with lock)
        progress_lock = asyncio.Lock()
        compressed_count = 0
        skipped_count = 0
        api_calls = 0
        in_flight = 0
        
        # Results storage: {file_path: {entry_idx: (processed_entry, metrics)}}
        results = {f: {} for f in jsonl_files}
        
        # Track timeouts separately
        timeout_count = 0
        
        async def process_single(file_path: Path, entry_idx: int, entry: Dict, 
                                  progress, main_task, status_task):
            """Process a single entry with semaphore rate limiting and timeout."""
            nonlocal compressed_count, skipped_count, api_calls, in_flight, timeout_count
            
            async with semaphore:
                # Track in-flight
                async with progress_lock:
                    in_flight += 1
                
                try:
                    # Apply per-trajectory timeout
                    processed_entry, metrics = await asyncio.wait_for(
                        self.process_entry_async(entry),
                        timeout=self.config.per_trajectory_timeout
                    )
                    results[file_path][entry_idx] = (processed_entry, metrics)
                    
                    # Update aggregate metrics (with lock for thread safety)
                    async with progress_lock:
                        self.aggregate_metrics.add_trajectory_metrics(metrics)
                        
                        # Update counters
                        if metrics.was_compressed:
                            compressed_count += 1
                            api_calls += metrics.summarization_api_calls
                        if metrics.skipped_under_target:
                            skipped_count += 1
                        
                        in_flight -= 1
                        
                        # Update progress
                        progress.advance(main_task)
                        progress.update(
                            status_task,
                            description=f"[dim]✅ {compressed_count} compressed | ⏭️ {skipped_count} skipped | ⏱️ {timeout_count} timeout | 🔄 {api_calls} API calls | ⚡ {in_flight} in-flight[/dim]"
                        )
                
                except asyncio.TimeoutError:
                    self.logger.warning(f"Timeout processing entry from {file_path}:{entry_idx} (>{self.config.per_trajectory_timeout}s)")
                    
                    async with progress_lock:
                        self.aggregate_metrics.trajectories_failed += 1
                        timeout_count += 1
                        in_flight -= 1
                        progress.advance(main_task)
                        progress.update(
                            status_task,
                            description=f"[dim]✅ {compressed_count} compressed | ⏭️ {skipped_count} skipped | ⏱️ {timeout_count} timeout | 🔄 {api_calls} API calls | ⚡ {in_flight} in-flight[/dim]"
                        )
                    
                    # Skip this entry entirely (don't include in output)
                    results[file_path][entry_idx] = None
                    
                except Exception as e:
                    self.logger.error(f"Error processing entry from {file_path}:{entry_idx}: {e}")
                    
                    async with progress_lock:
                        self.aggregate_metrics.trajectories_failed += 1
                        in_flight -= 1
                        progress.advance(main_task)
                    
                    # Keep original entry on error
                    results[file_path][entry_idx] = (entry, TrajectoryMetrics())
        
        # Create progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=10  # Higher refresh for async
        ) as progress:
            # Main task for overall progress
            main_task = progress.add_task(
                f"[cyan]Compressing {total_entries:,} trajectories",
                total=total_entries
            )
            
            # Status line task
            status_task = progress.add_task(
                "[dim]Starting...[/dim]",
                total=None
            )
            
            # Create all tasks
            tasks = [
                process_single(file_path, entry_idx, entry, progress, main_task, status_task)
                for file_path, entry_idx, entry in all_entries
            ]
            
            # Run all tasks concurrently (semaphore limits actual concurrency)
            await asyncio.gather(*tasks)
            
            # Remove status task
            progress.remove_task(status_task)
        
        # Write results to output files (preserving original order)
        console.print("\n[dim]Writing output files...[/dim]")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path in jsonl_files:
            output_path = output_dir / file_path.name
            file_results = results[file_path]
            
            # Sort by original entry index to preserve order, skip None (timed out) entries
            sorted_entries = [
                file_results[idx][0] 
                for idx in sorted(file_results.keys()) 
                if file_results[idx] is not None
            ]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in sorted_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # Record end time
        self.aggregate_metrics.processing_end_time = datetime.now().isoformat()
        self.aggregate_metrics.processing_duration_seconds = time.time() - start_time
        
        # Print summary
        self._print_summary()
        
        # Save metrics
        if self.config.metrics_enabled:
            metrics_path = output_dir / self.config.metrics_output_file
            with open(metrics_path, 'w') as f:
                json.dump(self.aggregate_metrics.to_dict(), f, indent=2)
            console.print(f"\n💾 Metrics saved to {metrics_path}")
    
    def _print_summary(self):
        """Print comprehensive compression summary statistics."""
        m = self.aggregate_metrics.to_dict()
        
        # Calculate some additional stats
        total = m['summary']['total_trajectories']
        compressed = m['summary']['trajectories_compressed']
        skipped = m['summary']['trajectories_skipped_under_target']
        over_limit = m['summary']['trajectories_still_over_limit']
        failed = m['summary']['trajectories_failed']
        
        # Token stats
        tokens_before = m['tokens']['total_before']
        tokens_after = m['tokens']['total_after']
        tokens_saved = m['tokens']['total_saved']
        
        # Calculate percentages
        compressed_pct = (compressed / max(total, 1)) * 100
        skipped_pct = (skipped / max(total, 1)) * 100
        over_limit_pct = (over_limit / max(total, 1)) * 100
        
        print(f"\n")
        print(f"╔{'═'*70}╗")
        print(f"║{'TRAJECTORY COMPRESSION REPORT':^70}║")
        print(f"╠{'═'*70}╣")
        
        # Trajectories section
        print(f"║{'':2}📁 TRAJECTORIES{' '*54}║")
        print(f"║{'─'*70}║")
        print(f"║{'':4}Total Processed:        {total:>10,}{' '*32}║")
        print(f"║{'':4}├─ Compressed:          {compressed:>10,}  ({compressed_pct:>5.1f}%){' '*18}║")
        print(f"║{'':4}├─ Skipped (under limit):{skipped:>9,}  ({skipped_pct:>5.1f}%){' '*18}║")
        print(f"║{'':4}├─ Still over limit:    {over_limit:>10,}  ({over_limit_pct:>5.1f}%){' '*18}║")
        print(f"║{'':4}└─ Failed:              {failed:>10,}{' '*32}║")
        
        print(f"╠{'═'*70}╣")
        
        # Tokens section
        print(f"║{'':2}🔢 TOKENS{' '*60}║")
        print(f"║{'─'*70}║")
        print(f"║{'':4}Before Compression:     {tokens_before:>15,} tokens{' '*21}║")
        print(f"║{'':4}After Compression:      {tokens_after:>15,} tokens{' '*21}║")
        print(f"║{'':4}Total Saved:            {tokens_saved:>15,} tokens{' '*21}║")
        print(f"║{'':4}Overall Compression:    {m['tokens']['overall_compression_ratio']:>14.1%}{' '*28}║")
        
        if tokens_before > 0:
            savings_pct = (tokens_saved / tokens_before) * 100
            print(f"║{'':4}Space Savings:          {savings_pct:>14.1f}%{' '*28}║")
        
        print(f"╠{'═'*70}╣")
        
        # Turns section
        print(f"║{'':2}💬 CONVERSATION TURNS{' '*48}║")
        print(f"║{'─'*70}║")
        print(f"║{'':4}Before Compression:     {m['turns']['total_before']:>15,} turns{' '*22}║")
        print(f"║{'':4}After Compression:      {m['turns']['total_after']:>15,} turns{' '*22}║")
        print(f"║{'':4}Total Removed:          {m['turns']['total_removed']:>15,} turns{' '*22}║")
        
        print(f"╠{'═'*70}╣")
        
        # Averages section (for compressed trajectories only)
        print(f"║{'':2}📈 AVERAGES (Compressed Trajectories Only){' '*27}║")
        print(f"║{'─'*70}║")
        if compressed > 0:
            print(f"║{'':4}Avg Compression Ratio:  {m['averages']['avg_compression_ratio']:>14.1%}{' '*28}║")
            print(f"║{'':4}Avg Tokens Saved:       {m['averages']['avg_tokens_saved_per_compressed']:>14,.0f}{' '*28}║")
            print(f"║{'':4}Avg Turns Removed:      {m['averages']['avg_turns_removed_per_compressed']:>14.1f}{' '*28}║")
        else:
            print(f"║{'':4}No trajectories were compressed{' '*38}║")
        
        print(f"╠{'═'*70}╣")
        
        # Summarization API section
        print(f"║{'':2}🤖 SUMMARIZATION API{' '*49}║")
        print(f"║{'─'*70}║")
        print(f"║{'':4}API Calls Made:         {m['summarization']['total_api_calls']:>15,}{' '*27}║")
        print(f"║{'':4}Errors:                 {m['summarization']['total_errors']:>15,}{' '*27}║")
        print(f"║{'':4}Success Rate:           {m['summarization']['success_rate']:>14.1%}{' '*28}║")
        
        print(f"╠{'═'*70}╣")
        
        # Processing time section
        duration = m['processing']['duration_seconds']
        if duration > 60:
            time_str = f"{duration/60:.1f} minutes"
        else:
            time_str = f"{duration:.1f} seconds"
        
        throughput = total / max(duration, 0.001)
        
        print(f"║{'':2}⏱️  PROCESSING TIME{' '*51}║")
        print(f"║{'─'*70}║")
        print(f"║{'':4}Duration:               {time_str:>20}{' '*22}║")
        print(f"║{'':4}Throughput:             {throughput:>15.1f} traj/sec{' '*18}║")
        print(f"║{'':4}Started:                {m['processing']['start_time'][:19]:>20}{' '*22}║")
        print(f"║{'':4}Finished:               {m['processing']['end_time'][:19]:>20}{' '*22}║")
        
        print(f"╚{'═'*70}╝")
        
        # Distribution summary if we have data
        if self.aggregate_metrics.compression_ratios:
            ratios = self.aggregate_metrics.compression_ratios
            tokens_saved_list = self.aggregate_metrics.tokens_saved_list
            
            print(f"\n📊 Distribution Summary:")
            print(f"   Compression ratios: min={min(ratios):.2%}, max={max(ratios):.2%}, median={sorted(ratios)[len(ratios)//2]:.2%}")
            print(f"   Tokens saved:       min={min(tokens_saved_list):,}, max={max(tokens_saved_list):,}, median={sorted(tokens_saved_list)[len(tokens_saved_list)//2]:,}")


def main(
    input: str,
    output: str = None,
    config: str = "configs/trajectory_compression.yaml",
    target_max_tokens: int = None,
    tokenizer: str = None,
    sample_percent: float = None,
    seed: int = 42,
    dry_run: bool = False,
):
    """
    Compress agent trajectories to fit within a target token budget.
    
    Supports both single JSONL files and directories containing multiple JSONL files.
    Optionally sample a percentage of trajectories before compression.
    
    Args:
        input: Path to JSONL file or directory containing JSONL files
        output: Output path (file for file input, directory for dir input)
                Default: adds "_compressed" suffix to input name
        config: Path to YAML configuration file
        target_max_tokens: Override target token count from config
        tokenizer: Override tokenizer name from config
        sample_percent: Sample this percentage of trajectories (1-100) before compression
        seed: Random seed for sampling reproducibility (default: 42)
        dry_run: Analyze without compressing (just show what would happen)
    
    Examples:
        # Compress a directory (original behavior)
        python trajectory_compressor.py --input=data/my_run
        
        # Compress a single file
        python trajectory_compressor.py --input=data/trajectories.jsonl
        
        # Compress 15% sample of a file
        python trajectory_compressor.py --input=data/trajectories.jsonl --sample_percent=15
        
        # Compress 10% sample with custom output
        python trajectory_compressor.py --input=data/trajectories.jsonl --sample_percent=10 --output=data/sampled_compressed.jsonl
    """
    import random
    import tempfile
    import shutil
    
    print("🗜️  Trajectory Compressor")
    print("=" * 60)
    
    # Load configuration
    config_path = Path(config)
    if config_path.exists():
        print(f"📋 Loading config from {config}")
        compression_config = CompressionConfig.from_yaml(config)
    else:
        print(f"⚠️  Config not found at {config}, using defaults")
        compression_config = CompressionConfig()
    
    # Apply CLI overrides
    if target_max_tokens:
        compression_config.target_max_tokens = target_max_tokens
    if tokenizer:
        compression_config.tokenizer_name = tokenizer
    
    # Validate sample_percent
    if sample_percent is not None:
        if sample_percent <= 0 or sample_percent > 100:
            print(f"❌ sample_percent must be between 1 and 100, got {sample_percent}")
            return
        print(f"🎲 Will sample {sample_percent}% of trajectories (seed={seed})")
    
    # Setup paths and determine input type
    input_path = Path(input)
    if not input_path.exists():
        print(f"❌ Input not found: {input}")
        return
    
    is_file_input = input_path.is_file()
    
    if is_file_input:
        print(f"📄 Input mode: Single JSONL file")
        
        # For file input, default output is file with _compressed suffix
        if output:
            output_path = Path(output)
        else:
            output_path = input_path.parent / (input_path.stem + compression_config.output_suffix + ".jsonl")
        
        # Load entries from the single file
        entries = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Skipping invalid JSON at line {line_num}: {e}")
        
        total_entries = len(entries)
        print(f"   Loaded {total_entries:,} trajectories from {input_path.name}")
        
        # Sample if requested
        if sample_percent is not None:
            random.seed(seed)
            sample_size = max(1, int(total_entries * sample_percent / 100))
            entries = random.sample(entries, sample_size)
            print(f"   Sampled {len(entries):,} trajectories ({sample_percent}% of {total_entries:,})")
        
        if dry_run:
            print(f"\n🔍 DRY RUN MODE - analyzing without writing")
            print(f"📄 Would process: {len(entries):,} trajectories")
            print(f"📄 Would output to: {output_path}")
            return
        
        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_dir = Path(temp_dir) / "input"
            temp_output_dir = Path(temp_dir) / "output"
            temp_input_dir.mkdir()
            
            # Write entries to temp file
            temp_input_file = temp_input_dir / "trajectories.jsonl"
            with open(temp_input_file, 'w', encoding='utf-8') as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            # Initialize compressor and process
            compressor = TrajectoryCompressor(compression_config)
            compressor.process_directory(temp_input_dir, temp_output_dir)
            
            # Copy result to output path (merge all files in temp_output_dir)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as out_f:
                for jsonl_file in sorted(temp_output_dir.glob("*.jsonl")):
                    with open(jsonl_file, 'r', encoding='utf-8') as in_f:
                        for line in in_f:
                            out_f.write(line)
            
            # Copy metrics file if it exists
            metrics_file = temp_output_dir / compression_config.metrics_output_file
            if metrics_file.exists():
                metrics_output = output_path.parent / (output_path.stem + "_metrics.json")
                shutil.copy(metrics_file, metrics_output)
                print(f"💾 Metrics saved to {metrics_output}")
        
        print(f"\n✅ Compression complete!")
        print(f"📄 Output: {output_path}")
        
    else:
        # Directory input - original behavior
        print(f"📁 Input mode: Directory of JSONL files")
        
        if output:
            output_path = Path(output)
        else:
            output_path = input_path.parent / (input_path.name + compression_config.output_suffix)
        
        # If sampling is requested for directory mode, we need to handle it differently
        if sample_percent is not None:
            print(f"\n⚠️  Sampling from directory: will sample {sample_percent}% from each file")
            
            # Create a temp directory with sampled files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_input_dir = Path(temp_dir) / "input"
                temp_input_dir.mkdir()
                
                random.seed(seed)
                total_original = 0
                total_sampled = 0
                
                # Sample from each JSONL file
                for jsonl_file in sorted(input_path.glob("*.jsonl")):
                    entries = []
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    entries.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass
                    
                    total_original += len(entries)
                    sample_size = max(1, int(len(entries) * sample_percent / 100))
                    sampled_entries = random.sample(entries, min(sample_size, len(entries)))
                    total_sampled += len(sampled_entries)
                    
                    # Write sampled entries
                    temp_file = temp_input_dir / jsonl_file.name
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        for entry in sampled_entries:
                            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
                print(f"   Sampled {total_sampled:,} from {total_original:,} total trajectories")
                
                if dry_run:
                    print(f"\n🔍 DRY RUN MODE - analyzing without writing")
                    print(f"📁 Would process: {temp_input_dir}")
                    print(f"📁 Would output to: {output_path}")
                    return
                
                # Initialize compressor and process the sampled data
                compressor = TrajectoryCompressor(compression_config)
                compressor.process_directory(temp_input_dir, output_path)
        else:
            if dry_run:
                print(f"\n🔍 DRY RUN MODE - analyzing without writing")
                print(f"📁 Would process: {input_path}")
                print(f"📁 Would output to: {output_path}")
                return
            
            # Initialize compressor and process directly
            compressor = TrajectoryCompressor(compression_config)
            compressor.process_directory(input_path, output_path)
        
        print("\n✅ Compression complete!")


if __name__ == "__main__":
    fire.Fire(main)
