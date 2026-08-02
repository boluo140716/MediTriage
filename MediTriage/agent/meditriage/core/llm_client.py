"""LLM 客户端。

支持调用 OpenAI 兼容的 API（如字节跳动豆包、OpenAI、Deepseek 等），
并支持 function calling。
"""
import sys
import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI
from loguru import logger

from pathlib import Path
# 向上找到含 config.py 的仓库根（不依赖固定层级 / 绝对路径，跨机器、跨目录可迁移）
for _d in Path(__file__).resolve().parents:
    if (_d / "config.py").is_file():
        sys.path.insert(0, str(_d))
        break
from config import LLM_CONFIG

# 解析 vLLM/OpenAI 「上下文超限」400 报错：抓出「最大上下文」与「prompt 长度」。
# vLLM 有两种措辞：常规 "prompt contains at least N input tokens"；
# 当 max_tokens ≥ max_model_len 时为 "prompt contains N characters"——都要认。
_CTX_OVERFLOW_RE = re.compile(
    r"maximum context length is (\d+) tokens"
    r".*?prompt contains (?:at least )?(\d+) (?:input tokens|characters)",
    re.S,
)


@dataclass
class ToolCall:
    """Function call 数据结构"""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """LLM 响应数据结构（支持 function calling）"""
    content: Optional[str]
    tool_calls: List[ToolCall]
    finish_reason: str  # "stop", "tool_calls", "length", "content_filter"
    prompt_tokens: Optional[int] = None   # 本次请求的输入 token 数 = 上下文占用
    total_tokens: Optional[int] = None

    def has_tool_calls(self) -> bool:
        """是否包含 function calls"""
        return len(self.tool_calls) > 0


class LLMClient:
    """统一的LLM客户端，支持多种模型"""

    def __init__(
        self,
        model_type: str = "openai_compatible",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化LLM客户端

        Args:
            model_type: 模型类型，默认 "openai_compatible"（支持 OpenAI 兼容的 API）
            config: 可选的自定义 LLM 配置（如视觉专用模型）；默认使用 config.LLM_CONFIG
        """
        self.model_type = model_type

        if model_type == "openai_compatible":
            # 使用 OpenAI 兼容的 API（通过 config.py 配置；可传入自定义配置覆盖）
            self.config = config if config is not None else LLM_CONFIG
            self.client = AsyncOpenAI(
                api_key=self.config["api_key"],
                base_url=self.config["base_url"],
                # SDK 默认约 600s——挂死一条请求会占住前端转圈 10 分钟
                timeout=float(self.config.get("timeout", 180)),
            )
            self.model_name = self.config["model_name"]
            self.temperature = self.config.get("temperature", 0.7)
            self.max_tokens = self.config.get("max_tokens", 8192)
            # 最近一次请求的 usage：chat() 只返回文本不带用量，
            # 视觉等直连链路经此取 prompt_tokens 上报前端 ctx 徽标
            self.last_usage = None
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _is_ctx_overflow(self, err: Exception) -> Optional[int]:
        """是「上下文超限」400 错误就返回 max-model-len(ctx)，否则 None。

        注：vLLM 报的 'prompt contains at least N' 实为 (ctx - max_tokens + 1)
        的下界，会随发送的 max_tokens 变化，并非真实 prompt 长度，故不据其精确
        计算，改用退避。
        """
        m = _CTX_OVERFLOW_RE.search(str(err))
        return int(m.group(1)) if m else None

    def is_ctx_overflow(self, err: Exception) -> bool:
        """该异常是否为「上下文超限」——确定性失败，重试同一 prompt 无意义。"""
        return self._is_ctx_overflow(err) is not None

    async def _create_with_overflow_guard(self, request_params: Dict[str, Any]):
        """
        包一层 chat.completions.create：当 prompt + max_tokens 超过 max-model-len 时，
        对半收缩 max_tokens 退避重试，避免整条请求直接失败（vLLM 上下文偏小时的兜底）。
        收缩到地板(256)仍放不下 = prompt 本身过大、需上层裁历史，此时如实抛出。
        """
        floor = 256
        while True:
            try:
                return await self.client.chat.completions.create(
                    **request_params
                )
            except Exception as e:
                if self._is_ctx_overflow(e) is None:
                    raise
                cur = request_params.get("max_tokens") or self.max_tokens
                new_mt = max(floor, cur // 2)
                if new_mt >= cur:  # 已到地板，再砍无意义
                    raise
                logger.warning(f"上下文超限：max_tokens {cur} -> {new_mt} 退避重试")
                request_params["max_tokens"] = new_mt

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        异步聊天接口

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            temperature: 温度参数（可选）
            max_tokens: 最大token数（可选）

        Returns:
            模型返回的文本
        """
        try:
            # `or` 会把显式传入的 temperature=0（确定性路由）吞成默认值
            if temperature is None:
                temperature = self.temperature
            max_tokens = max_tokens or self.max_tokens

            logger.debug(
                f"Calling LLM ({self.model_type}) "
                f"with {len(messages)} messages"
            )

            response = await self._create_with_overflow_guard({
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            })

            # 某些 finish_reason 下 content 可为 None
            content = response.choices[0].message.content or ""
            self.last_usage = getattr(response, "usage", None)
            logger.debug(f"LLM response length: {len(content)} chars")
            return content

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    async def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        带工具支持的聊天接口

        Args:
            messages: 消息列表
            tools: 工具定义列表（OpenAI format）
            tool_choice: 工具选择策略 ("auto"/"required"/"none")
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            LLMResponse 对象
        """
        try:
            if temperature is None:
                temperature = self.temperature
            max_tokens = max_tokens or self.max_tokens

            logger.debug(f"Calling LLM with {len(tools) if tools else 0} tools")

            # 准备请求参数
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }

            # 添加工具参数（如果提供）
            if tools:
                request_params["tools"] = tools
                if tool_choice != "auto":
                    request_params["tool_choice"] = tool_choice

            response = await self._create_with_overflow_guard(request_params)

            # 解析响应
            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # 提取工具调用：单个 call 的参数 JSON 畸形（8B 模型偶发截断）
            # 只跳过该 call，不作废同响应里的其它合法 call 与文本
            tool_calls = []
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tc in message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(
                            f"工具调用参数解析失败，跳过该 call："
                            f"{tc.function.name} "
                            f"({e}; raw={str(tc.function.arguments)[:120]})"
                        )
                        continue
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    ))
                logger.debug(f"LLM requested {len(tool_calls)} tool calls")

            usage = getattr(response, "usage", None)
            self.last_usage = usage
            return LLMResponse(
                content=message.content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            )

        except Exception as e:
            logger.error(f"LLM call with tools failed: {e}")
            raise

    def create_tool_message(
        self,
        tool_call_id: str,
        tool_name: str,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        创建工具执行结果消息

        Args:
            tool_call_id: 工具调用ID
            tool_name: 工具名称
            result: 工具执行结果

        Returns:
            工具消息字典
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": json.dumps(result, ensure_ascii=False)
        }
