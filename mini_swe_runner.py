#!/usr/bin/env python3
"""SWE（软件工程）任务运行器，使用 Hermes 轨迹格式。

使用 Hermes-Agent 内置的执行环境（local、docker、modal）运行任务，
并输出与 batch_runner.py 和 trajectory_compressor.py 兼容的 Hermes 格式轨迹。

特性：
- 使用 Hermes-Agent 的 Docker、Modal 或 Local 环境执行命令
- 输出 Hermes 格式的轨迹（from/value 对，包含 <tool_call>/<tool_response> XML）
- 与轨迹压缩管道兼容
- 支持从 JSONL 提示文件批量处理

使用示例：
    # 使用本地环境运行单个任务
    python mini_swe_runner.py --task "Create a hello world Python script" --env local
    
    # 使用 Docker 运行
    python mini_swe_runner.py --task "List files in /tmp" --env docker --image python:3.11-slim
    
    # 使用 Modal（云端）运行
    python mini_swe_runner.py --task "Install numpy and test it" --env modal --image python:3.11-slim
    
    # 从 JSONL 文件批量运行
    python mini_swe_runner.py --prompts_file prompts.jsonl --output_file trajectories.jsonl --env docker

主要用途：
    - 运行 SWE-bench 软件工程评估任务
    - 生成训练数据轨迹（用于模型微调）
    - 测试 agent 在不同执行环境中的表现
"""

import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal

import fire
from dotenv import load_dotenv

# Load environment variables
load_dotenv()




# ============================================================================
# Terminal Tool Definition (matches Hermes-Agent format)
# ============================================================================

TERMINAL_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "terminal",
        "description": """Execute bash commands in a sandboxed environment.

**Environment:**
- Isolated execution environment (local, Docker, or Modal cloud)
- Filesystem persists between tool calls within the same task
- Internet access available

**Command Execution:**
- Provide the command to execute via the 'command' parameter
- Optional 'timeout' parameter in seconds (default: 60)

**Examples:**
- Run command: `{"command": "ls -la"}`
- With timeout: `{"command": "long_task.sh", "timeout": 300}`

**Best Practices:**
- Use non-interactive commands (avoid vim, nano, interactive python)
- Pipe to cat if output might be large
- Install tools with apt-get or pip as needed

**Completion:**
- When task is complete, output: echo "MINI_SWE_AGENT_FINAL_OUTPUT" followed by your result
""",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Command timeout in seconds (default: 60)"
                }
            },
            "required": ["command"]
        }
    }
}


# ============================================================================
# Environment Factory
# ============================================================================

def create_environment(
    env_type: str = "local",
    image: str = "python:3.11-slim",
    cwd: str = "/tmp",
    timeout: int = 60,
    **kwargs
):
    """使用 Hermes-Agent 的内置后端创建执行环境。
    
    功能概括：
        工厂函数，根据环境类型创建相应的执行环境实例。
        支持本地、Docker 容器和 Modal 云端三种环境。
    
    参数：
        env_type: 环境类型，"local"（本地）、"docker"（容器）、"modal"（云端）
        image: Docker/Modal 镜像名称（本地环境忽略此参数）
        cwd: 工作目录
        timeout: 默认命令超时时间（秒）
        **kwargs: 其他环境特定选项
    
    返回值：
        Environment: 环境实例，具有 execute() 和 cleanup() 方法
    
    主要用于：
        - MiniSWERunner._create_env() 中创建任务执行环境
        - 为每个任务提供隔离的执行沙箱
    
    环境选择：
        - local: 直接在主机上执行，无隔离
        - docker: 在 Docker 容器中执行，文件系统隔离
        - modal: 在 Modal 云端执行，适合需要大量资源的任务
    """
    if env_type == "local":
        from tools.environments.local import LocalEnvironment
        return LocalEnvironment(cwd=cwd, timeout=timeout)
    
    elif env_type == "docker":
        from tools.environments.docker import DockerEnvironment
        return DockerEnvironment(image=image, cwd=cwd, timeout=timeout, **kwargs)
    
    elif env_type == "modal":
        from tools.environments.modal import ModalEnvironment
        return ModalEnvironment(image=image, cwd=cwd, timeout=timeout, **kwargs)
    
    else:
        raise ValueError(f"Unknown environment type: {env_type}. Use 'local', 'docker', or 'modal'")


# ============================================================================
# Mini-SWE Runner with Hermes Trajectory Format
# ============================================================================

class MiniSWERunner:
    """使用 Hermes-Agent 内置执行环境并输出 Hermes 格式轨迹的 Agent 运行器。
    
    功能概括：
        管理完整的 agent 任务执行生命周期：
        1. 初始化 LLM 客户端和执行环境
        2. 执行 agent 对话循环（调用工具、获取结果）
        3. 将内部消息格式转换为 Hermes 轨迹格式
        4. 支持单任务和批量任务模式
    
    主要用途：
        - 运行 SWE-bench 软件工程任务
        - 生成用于模型微调的训练数据
        - 测试不同模型和环境组合的表现
    
    轨迹格式：
        输出标准的 Hermes 对话格式，包含 system/human/gpt/tool 角色，
        使用 <tool_call> 和 <tool_response> XML 标签标记工具调用和结果。
    """
    
    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4.6",
        base_url: str = None,
        api_key: str = None,
        env_type: str = "local",
        image: str = "python:3.11-slim",
        cwd: str = "/tmp",
        max_iterations: int = 15,
        command_timeout: int = 60,
        verbose: bool = False,
    ):
        """初始化 Mini-SWE 运行器。
        
        功能概括：
            配置 LLM 客户端、执行环境参数和工具定义。
            自动从环境变量或显式参数解析 API 凭据。
        
        参数：
            model: OpenAI 兼容 API 的模型名称
            base_url: API 基础 URL（可选，未提供时使用环境变量）
            api_key: API 密钥（可选，未提供时使用环境变量）
            env_type: 环境类型 - "local"、"docker" 或 "modal"
            image: Docker/Modal 镜像（本地环境忽略）
            cwd: 命令的工作目录
            max_iterations: 最大工具调用迭代次数
            command_timeout: 命令默认超时时间（秒）
            verbose: 是否启用详细日志
        
        返回值：
            无
        
        主要用于：
            - main() 函数中初始化运行器
            - 为后续任务执行准备环境
        
        LLM 客户端解析顺序：
            1. 显式提供的 api_key/base_url
            2. 通过 resolve_provider_client() 使用 OpenRouter
            3. 自动检测可用提供商
            4. 回退到 OpenRouter 默认配置
        """
        self.model = model
        self.max_iterations = max_iterations
        self.command_timeout = command_timeout
        self.verbose = verbose
        self.env_type = env_type
        self.image = image
        self.cwd = cwd
        
        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM client via centralized provider router.
        # If explicit api_key/base_url are provided (e.g. from CLI args),
        # construct directly.  Otherwise use the router for OpenRouter.
        if api_key or base_url:
            from openai import OpenAI
            client_kwargs = {
                "base_url": base_url or "https://openrouter.ai/api/v1",
                "api_key": api_key or os.getenv(
                    "OPENROUTER_API_KEY",
                    os.getenv("ANTHROPIC_API_KEY",
                              os.getenv("OPENAI_API_KEY", ""))),
            }
            self.client = OpenAI(**client_kwargs)
        else:
            from agent.auxiliary_client import resolve_provider_client
            self.client, _ = resolve_provider_client("openrouter", model=model)
            if self.client is None:
                # Fallback: try auto-detection
                self.client, _ = resolve_provider_client("auto", model=model)
            if self.client is None:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY", ""))
        
        # Environment will be created per-task
        self.env = None
        
        # Tool definition
        self.tools = [TERMINAL_TOOL_DEFINITION]
        
        print("🤖 Mini-SWE Runner initialized")
        print(f"   Model: {self.model}")
        print(f"   Environment: {self.env_type}")
        if self.env_type != "local":
            print(f"   Image: {self.image}")
        print(f"   Max iterations: {self.max_iterations}")
    
    def _create_env(self):
        """创建执行环境。
        
        功能概括：
            调用 create_environment() 工厂函数创建隔离的执行环境。
            每个任务开始时调用，确保环境干净。
        
        参数：
            无
        
        返回值：
            无（设置 self.env 属性）
        
        主要用于：
            - run_task() 开始时初始化环境
        """
        print(f"🔧 Creating {self.env_type} environment...")
        self.env = create_environment(
            env_type=self.env_type,
            image=self.image,
            cwd=self.cwd,
            timeout=self.command_timeout
        )
        print("✅ Environment ready")
    
    def _cleanup_env(self):
        """清理执行环境。
        
        功能概括：
            释放执行环境资源（停止容器、清理临时文件等）。
            每个任务结束时调用，确保资源不泄漏。
        
        参数：
            无
        
        返回值：
            无
        
        主要用于：
            - run_task() 结束时清理环境
            - finally 块中确保即使异常也清理
        """
        if self.env is not None:
            if hasattr(self.env, 'cleanup'):
                self.env.cleanup()
            elif hasattr(self.env, 'stop'):
                self.env.stop()
            self.env = None
    
    def _execute_command(self, command: str, timeout: int = None) -> Dict[str, Any]:
        """在环境中执行命令。
        
        功能概括：
            封装环境 execute() 调用，提供统一的错误处理和返回格式。
            如果环境未创建则自动创建。
        
        参数：
            command: 要执行的 bash 命令
            timeout: 可选的超时覆盖（秒），未提供时使用默认值
        
        返回值：
            Dict: 包含以下键：
                - output: 命令输出（stdout + stderr）
                - exit_code: 退出码（0 表示成功）
                - error: 错误信息（None 表示无错误）
        
        主要用于：
            - run_task() 中执行 agent 的工具调用
            - 为 agent 提供命令执行结果
        """
        if self.env is None:
            self._create_env()
        
        try:
            result = self.env.execute(command, timeout=timeout or self.command_timeout)
            return {
                "output": result.get("output", ""),
                "exit_code": result.get("returncode", 0),
                "error": None
            }
        except Exception as e:
            return {
                "output": "",
                "exit_code": -1,
                "error": str(e)
            }
    
    def _format_tools_for_system_message(self) -> str:
        """将工具定义格式化为系统消息字符串。
        
        功能概括：
            将工具 schema 转换为 JSON 字符串，嵌入系统提示中。
            移除 'required' 字段以简化格式。
        
        参数：
            无
        
        返回值：
            str: JSON 格式的工具定义列表
        
        主要用于：
            - _convert_to_hermes_format() 中构建系统消息
            - 告知 LLM 可用的工具及其参数
        """
        formatted_tools = []
        for tool in self.tools:
            func = tool["function"]
            formatted_tools.append({
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
                "required": None
            })
        return json.dumps(formatted_tools, ensure_ascii=False)
    
    def _convert_to_hermes_format(
        self,
        messages: List[Dict[str, Any]],
        user_query: str,
        completed: bool
    ) -> List[Dict[str, Any]]:
        """将内部消息格式转换为 Hermes 轨迹格式。
        
        功能概括：
            将 OpenAI 风格的消息列表转换为 Hermes 标准的 from/value 对格式。
            生成与 batch_runner.py 完全相同的格式，包含 XML 标签标记工具调用。
        
        参数：
            messages: 内部消息列表，包含 role、content、tool_calls 等字段
            user_query: 用户原始查询
            completed: 任务是否完成
        
        返回值：
            List[Dict[str, Any]]: Hermes 轨迹消息列表
                每条消息包含 "from" 和 "value" 键
                - from: "system"/"human"/"gpt"/"tool"
                - value: 消息内容（可能包含 XML 标签）
        
        主要用于：
            - run_task() 结束时转换轨迹格式
            - 输出到 JSONL 文件供后续训练使用
        
        格式特点：
            - 系统消息包含 <tools> 标签和工具定义
            - 工具调用使用 <tool_call> 标签包裹
            - 工具结果使用 <tool_response> 标签包裹
            - 推理内容使用 <think> 标签（如果有）
        """
        trajectory = []
        
        # System message with tool definitions
        system_msg = (
            "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
            "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
            "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
            "into functions. After calling & executing the functions, you will be provided with function results within "
            "<tool_response> </tool_response> XML tags. Here are the available tools:\n"
            f"<tools>\n{self._format_tools_for_system_message()}\n</tools>\n"
            "For each function call return a JSON object, with the following pydantic model json schema for each:\n"
            "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
            "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['name', 'arguments']}\n"
            "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
            "Example:\n<tool_call>\n{'name': <function-name>,'arguments': <args-dict>}\n</tool_call>"
        )
        
        trajectory.append({"from": "system", "value": system_msg})
        trajectory.append({"from": "human", "value": user_query})
        
        # Process messages (skip first user message as we already added it)
        i = 1
        while i < len(messages):
            msg = messages[i]
            
            if msg["role"] == "assistant":
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Assistant message with tool calls
                    content = ""
                    
                    # Add reasoning if present
                    if msg.get("reasoning"):
                        content = f"<think>{msg['reasoning']}</think>"
                    
                    if msg.get("content"):
                        content += msg["content"] + "\n"
                    
                    # Add tool calls in XML format
                    for tool_call in msg["tool_calls"]:
                        if not tool_call or not isinstance(tool_call, dict): continue
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"]) \
                                if isinstance(tool_call["function"]["arguments"], str) \
                                else tool_call["function"]["arguments"]
                        except json.JSONDecodeError:
                            arguments = {}
                        
                        tool_call_json = {
                            "name": tool_call["function"]["name"],
                            "arguments": arguments
                        }
                        content += f"<tool_call>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</tool_call>\n"
                    
                    trajectory.append({"from": "gpt", "value": content.rstrip()})
                    
                    # Collect subsequent tool responses
                    tool_responses = []
                    j = i + 1
                    while j < len(messages) and messages[j]["role"] == "tool":
                        tool_msg = messages[j]
                        tool_content = tool_msg["content"]
                        
                        # Try to parse as JSON
                        try:
                            if tool_content.strip().startswith(("{", "[")):
                                tool_content = json.loads(tool_content)
                        except (json.JSONDecodeError, AttributeError):
                            pass
                        
                        tool_response = "<tool_response>\n"
                        tool_response += json.dumps({
                            "tool_call_id": tool_msg.get("tool_call_id", ""),
                            "name": msg["tool_calls"][len(tool_responses)]["function"]["name"] \
                                if len(tool_responses) < len(msg["tool_calls"]) else "unknown",
                            "content": tool_content
                        }, ensure_ascii=False)
                        tool_response += "\n</tool_response>"
                        tool_responses.append(tool_response)
                        j += 1
                    
                    if tool_responses:
                        trajectory.append({"from": "tool", "value": "\n".join(tool_responses)})
                        i = j - 1
                
                else:
                    # Regular assistant message (no tool calls)
                    content = ""
                    if msg.get("reasoning"):
                        content = f"<think>{msg['reasoning']}</think>"
                    content += msg.get("content") or ""
                    trajectory.append({"from": "gpt", "value": content})
            
            elif msg["role"] == "user":
                trajectory.append({"from": "human", "value": msg["content"]})
            
            i += 1
        
        return trajectory
    
    def run_task(self, task: str) -> Dict[str, Any]:
        """运行单个任务并返回包含轨迹的结果。
        
        功能概括：
            执行完整的 agent 对话循环：
            1. 创建执行环境
            2. 初始化消息历史
            3. 循环调用 LLM API 和执行工具
            4. 检测任务完成信号
            5. 清理环境
            6. 转换为 Hermes 轨迹格式
        
        参数：
            task: 要执行的任务/提示词
        
        返回值：
            Dict: 包含以下键：
                - conversations: Hermes 格式的轨迹消息列表
                - completed: 任务是否成功完成（bool）
                - api_calls: API 调用次数
                - metadata: 元数据（模型、环境类型、时间戳）
        
        主要用于：
            - main() 单任务模式
            - run_batch() 批量模式中的每个任务
        
        完成信号：
            当命令输出包含 "MINI_SWE_AGENT_FINAL_OUTPUT" 时认为任务完成。
        """
        print(f"\n{'='*60}")
        print(f"📝 Task: {task[:80]}{'...' if len(task) > 80 else ''}")
        print(f"{'='*60}")
        
        # Initialize environment
        self._create_env()
        
        # Message history
        messages = [{"role": "user", "content": task}]
        
        # System prompt for the LLM (ephemeral - not saved to trajectory)
        system_prompt = """You are an AI agent that can execute bash commands to complete tasks.

When you need to run commands, use the 'terminal' tool with your bash command.

**Important:**
- When you have completed the task successfully, run: echo "MINI_SWE_AGENT_FINAL_OUTPUT" followed by a summary
- Be concise and efficient in your approach
- Install any needed tools with apt-get or pip
- Avoid interactive commands (no vim, nano, less, etc.)

Complete the user's task step by step."""
        
        api_call_count = 0
        completed = False
        final_response = None
        
        try:
            while api_call_count < self.max_iterations:
                api_call_count += 1
                print(f"\n🔄 API call #{api_call_count}/{self.max_iterations}")
                
                # Prepare API messages
                api_messages = [{"role": "system", "content": system_prompt}] + messages
                
                # Make API call
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=api_messages,
                        tools=self.tools,
                        timeout=300.0
                    )
                except Exception as e:
                    self.logger.error(f"API call failed: {e}")
                    break
                
                assistant_message = response.choices[0].message
                
                # Log assistant response
                if assistant_message.content:
                    print(f"🤖 Assistant: {assistant_message.content[:100]}...")
                
                # Check for tool calls
                if assistant_message.tool_calls:
                    print(f"🔧 Tool calls: {len(assistant_message.tool_calls)}")
                    
                    # Add assistant message with tool calls
                    messages.append({
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in assistant_message.tool_calls
                        ]
                    })
                    
                    # Execute each tool call
                    for tc in assistant_message.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            args = {}
                        
                        command = args.get("command", "echo 'No command provided'")
                        timeout = args.get("timeout", self.command_timeout)
                        
                        print(f"   📞 terminal: {command[:60]}...")
                        
                        # Execute command
                        result = self._execute_command(command, timeout)
                        
                        # Format result
                        result_json = json.dumps({
                            "content": {
                                "output": result["output"],
                                "exit_code": result["exit_code"],
                                "error": result["error"]
                            }
                        }, ensure_ascii=False)
                        
                        # Check for task completion signal
                        if "MINI_SWE_AGENT_FINAL_OUTPUT" in result["output"]:
                            print("   ✅ Task completion signal detected!")
                            completed = True
                        
                        # Add tool response
                        messages.append({
                            "role": "tool",
                            "content": result_json,
                            "tool_call_id": tc.id
                        })
                        
                        print(f"   ✅ exit_code={result['exit_code']}, output={len(result['output'])} chars")
                    
                    # If task completed, we can stop
                    if completed:
                        final_response = assistant_message.content
                        break
                
                else:
                    # No tool calls - final response
                    final_response = assistant_message.content or ""
                    messages.append({
                        "role": "assistant",
                        "content": final_response
                    })
                    completed = True
                    print("🎉 Agent finished (no more tool calls)")
                    break
            
            if api_call_count >= self.max_iterations:
                print(f"⚠️  Reached max iterations ({self.max_iterations})")
        
        finally:
            # Cleanup environment
            self._cleanup_env()
        
        # Convert to Hermes trajectory format
        trajectory = self._convert_to_hermes_format(messages, task, completed)
        
        return {
            "conversations": trajectory,
            "completed": completed,
            "api_calls": api_call_count,
            "metadata": {
                "model": self.model,
                "env_type": self.env_type,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def run_batch(
        self,
        prompts: List[str],
        output_file: str
    ) -> List[Dict[str, Any]]:
        """运行多个任务并将轨迹保存到 JSONL 文件。
        
        功能概括：
            批量执行任务列表，每个任务完成后立即写入文件，
            确保即使中断也不会丢失已完成的结果。
        
        参数：
            prompts: 任务提示词列表
            output_file: 输出 JSONL 文件路径
        
        返回值：
            List[Dict[str, Any]]: 所有任务的结果列表
        
        主要用于：
            - main() 批量模式（使用 --prompts_file）
            - 生成大规模训练数据集
        
        容错机制：
            - 每个任务独立 try-except，一个失败不影响其他
            - 每个任务完成后立即 flush 到文件
            - 失败任务记录错误信息到结果中
        """
        results = []
        
        print(f"\n📦 Running batch of {len(prompts)} tasks")
        print(f"📁 Output: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, prompt in enumerate(prompts, 1):
                print(f"\n{'='*60}")
                print(f"📋 Task {i}/{len(prompts)}")
                print(f"{'='*60}")
                
                try:
                    result = self.run_task(prompt)
                    results.append(result)
                    
                    # Write to file immediately
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    
                    print(f"✅ Task {i} completed (api_calls={result['api_calls']})")
                    
                except Exception as e:
                    self.logger.error(f"Error on task {i}: {e}")
                    error_result = {
                        "conversations": [],
                        "completed": False,
                        "api_calls": 0,
                        "error": str(e),
                        "metadata": {"timestamp": datetime.now().isoformat()}
                    }
                    results.append(error_result)
                    f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                    f.flush()
        
        print(f"\n✅ Batch complete! {len(results)} trajectories saved to {output_file}")
        return results


# ============================================================================
# CLI Interface
# ============================================================================

def main(
    task: str = None,
    prompts_file: str = None,
    output_file: str = "swe-runner-test1.jsonl",
    model: str = "claude-sonnet-4-20250514",
    base_url: str = None,
    api_key: str = None,
    env: str = "local",
    image: str = "python:3.11-slim",
    cwd: str = "/tmp",
    max_iterations: int = 15,
    timeout: int = 60,
    verbose: bool = False,
):
    """运行 SWE 任务并输出 Hermes 格式轨迹的 CLI 主入口。
    
    功能概括：
        解析命令行参数，初始化 MiniSWERunner，执行单任务或批量任务，
        并将轨迹结果保存到 JSONL 文件。
    
    参数：
        task: 要运行的单个任务（使用此参数或 prompts_file）
        prompts_file: JSONL 提示文件路径（每行：{"prompt": "..."}）
        output_file: 轨迹输出 JSONL 文件路径
        model: 模型名称（默认：claude-sonnet-4-20250514）
        base_url: API 基础 URL（可选）
        api_key: API 密钥（可选，使用环境变量）
        env: 环境类型 - "local"、"docker" 或 "modal"
        image: Docker/Modal 镜像（默认：python:3.11-slim）
        cwd: 工作目录（默认：/tmp）
        max_iterations: 最大工具调用迭代次数（默认：15）
        timeout: 命令超时时间（秒，默认：60）
        verbose: 是否启用详细日志
    
    返回值：
        无（直接运行并输出结果）
    
    主要用于：
        - 命令行入口点
        - 运行 SWE-bench 评估任务
        - 生成训练数据
    
    使用示例：
        # 使用本地环境运行单个任务
        python mini_swe_runner.py --task "Create hello.py that prints Hello World"
        
        # 使用 Docker 运行单个任务
        python mini_swe_runner.py --task "List files" --env docker
        
        # 从文件批量运行
        python mini_swe_runner.py --prompts_file tasks.jsonl --output_file results.jsonl
    """
    print("🚀 Mini-SWE Runner with Hermes Trajectory Format")
    print("=" * 60)
    
    # Initialize runner
    runner = MiniSWERunner(
        model=model,
        base_url=base_url,
        api_key=api_key,
        env_type=env,
        image=image,
        cwd=cwd,
        max_iterations=max_iterations,
        command_timeout=timeout,
        verbose=verbose,
    )
    
    if task:
        # Single task mode
        result = runner.run_task(task)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        print(f"\n📁 Trajectory saved to: {output_file}")
        print(f"✅ Completed: {result['completed']}")
        print(f"📞 API calls: {result['api_calls']}")
        print(f"💬 Turns: {len(result['conversations'])}")
        
    elif prompts_file:
        # Batch mode
        prompts = []
        with open(prompts_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        prompts.append(entry.get("prompt", entry.get("task", "")))
                    except json.JSONDecodeError:
                        prompts.append(line)
        
        if not prompts:
            print(f"❌ No prompts found in {prompts_file}")
            return
        
        runner.run_batch(prompts, output_file)
    
    else:
        print("❌ Please provide either --task or --prompts_file")
        print("   Example: python mini_swe_runner.py --task 'Create a hello world script'")


if __name__ == "__main__":
    fire.Fire(main)
