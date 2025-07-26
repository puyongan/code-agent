# 修复包导入错误
import sys, os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from typing import Optional
import asyncio
import os
from mcp import Tool
from openai import NOT_GIVEN, AsyncOpenAI
from dataclasses import dataclass, field
from openai.types import FunctionDefinition
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
import dotenv
from pydantic import BaseModel
from rich import print as rprint

from augmented.utils import pretty
from augmented.utils.info import DEFAULT_MODEL_NAME

PRETTY_LOGGER = pretty.ALogger("[ChatOpenAI]")

dotenv.load_dotenv()


class ToolCallFunction(BaseModel):
    name: str = ""
    arguments: str = ""


class ToolCall(BaseModel):
    id: str = ""
    function: ToolCallFunction = ToolCallFunction()


class ChatOpenAIChatResponse(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = []


@dataclass
class AsyncChatOpenAI:
    model: str
    messages: list[ChatCompletionMessageParam] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    system_prompt: str = ""
    context: str = ""
    temperature: float = 0.1  # 添加温度参数

    llm: AsyncOpenAI = field(init=False)

    def __post_init__(self) -> None:
        self.llm = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
        if self.system_prompt:
            self.messages.insert(0, {"role": "system", "content": self.system_prompt})
        if self.context:
            self.messages.append({"role": "user", "content": self.context})

    async def chat(
            self, prompt: str = "", print_llm_output: bool = True
    ) -> ChatOpenAIChatResponse:
        try:
            return await self._chat(prompt, print_llm_output)
        except Exception as e:
            rprint(f"Error during chat: {e!s}")
            raise

    async def _chat(
            self, prompt: str = "", print_llm_output: bool = True
    ) -> ChatOpenAIChatResponse:
        PRETTY_LOGGER.title("CHAT")
        if prompt:
            self.messages.append({"role": "user", "content": prompt})

        content = ""
        tool_calls: list[ToolCall] = []
        printed_llm_output = False

        # 修复工具定义格式
        param_tools = self.get_tools_definition() if self.tools else NOT_GIVEN

        try:
            async with await self.llm.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=param_tools,
                    temperature=self.temperature,  # 添加温度参数
                    stream=True,
            ) as stream:
                PRETTY_LOGGER.title("RESPONSE")
                async for chunk in stream:
                    delta = chunk.choices[0].delta
                    # 处理 content
                    if delta.content:
                        content += delta.content or ""
                        if print_llm_output:
                            print(delta.content, end="")
                            printed_llm_output = True
                    # 处理 tool_calls
                    if delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            if len(tool_calls) <= tool_call.index:
                                tool_calls.append(ToolCall())
                            this_tool_call = tool_calls[tool_call.index]
                            if tool_call.id:
                                this_tool_call.id += tool_call.id or ""
                            if tool_call.function:
                                if tool_call.function.name:
                                    this_tool_call.function.name += (
                                            tool_call.function.name or ""
                                    )
                                if tool_call.function.arguments:
                                    this_tool_call.function.arguments += (
                                            tool_call.function.arguments or ""
                                    )
        except Exception as e:
            rprint(f"API调用失败: {e}")
            # 打印详细的工具定义信息用于调试
            if param_tools != NOT_GIVEN:
                rprint("工具定义:", param_tools)
            raise

        if printed_llm_output:
            print()

        # 只有在有内容或工具调用时才添加到消息历史
        if content or tool_calls:
            message_to_append = {
                "role": "assistant",
                "content": content,
            }

            # 只有在有工具调用时才添加tool_calls字段
            if tool_calls:
                message_to_append["tool_calls"] = [
                    {
                        "type": "function",
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]

            self.messages.append(message_to_append)

        return ChatOpenAIChatResponse(
            content=content,
            tool_calls=tool_calls,
        )

    def get_tools_definition(self) -> list[ChatCompletionToolParam]:
        """修复工具定义格式，清理不兼容的字段"""
        if not self.tools:
            return []

        tools_def = []
        for t in self.tools:
            try:
                # 深度复制并清理参数schema
                parameters = self._clean_schema(t.inputSchema)

                tool_def = ChatCompletionToolParam(
                    type="function",
                    function=FunctionDefinition(
                        name=t.name,
                        description=t.description or "",
                        parameters=parameters,
                    ),
                )
                tools_def.append(tool_def)

            except Exception as e:
                rprint(f"工具 {t.name} 定义格式错误: {e}")
                continue

        return tools_def

    def _clean_schema(self, schema: dict) -> dict:
        """清理schema中API提供商不支持的字段"""
        if not isinstance(schema, dict):
            return {"type": "object", "properties": {}}

        # 创建清理后的schema副本
        cleaned = {}

        # 保留基本字段
        if "type" in schema:
            cleaned["type"] = schema["type"]
        else:
            cleaned["type"] = "object"

        if "properties" in schema:
            cleaned["properties"] = self._clean_properties(schema["properties"])
        else:
            cleaned["properties"] = {}

        if "required" in schema:
            cleaned["required"] = schema["required"]

        if "additionalProperties" in schema:
            cleaned["additionalProperties"] = schema["additionalProperties"]

        # 跳过这些可能导致问题的字段
        # "$schema", "description", "title" 等

        return cleaned

    def _clean_properties(self, properties: dict) -> dict:
        """清理properties中的字段"""
        if not isinstance(properties, dict):
            return {}

        cleaned_props = {}
        for key, value in properties.items():
            if isinstance(value, dict):
                cleaned_value = {}
                # 只保留基本字段
                for field in ["type", "enum", "default", "minimum", "maximum",
                              "exclusiveMinimum", "exclusiveMaximum", "minLength",
                              "maxLength", "items", "format"]:
                    if field in value:
                        cleaned_value[field] = value[field]

                # 递归处理嵌套对象
                if "items" in value and isinstance(value["items"], dict):
                    cleaned_value["items"] = self._clean_properties({"item": value["items"]})["item"]

                cleaned_props[key] = cleaned_value
            else:
                cleaned_props[key] = value

        return cleaned_props

    def append_tool_result(self, tool_call_id: str, tool_output: str) -> None:
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": tool_output,
            }
        )


async def example() -> None:
    llm = AsyncChatOpenAI(
        model=DEFAULT_MODEL_NAME,
    )
    chat_resp = await llm.chat(prompt="你是谁，能帮我实现functioncalling吗")
    rprint(chat_resp)


if __name__ == "__main__":
    asyncio.run(example())