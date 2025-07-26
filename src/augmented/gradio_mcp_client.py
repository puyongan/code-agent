import sys
import os
import asyncio
import json
from typing import AsyncGenerator, List, Dict, Any

import gradio as gr
from rich.console import Console
from rich.text import Text

from augmented.chat_openai import AsyncChatOpenAI

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from augmented.agent import Agent  # 直接复用你的 Agent 类
from augmented.mcp_client import MCPClient
from augmented.mcp_tools import PresetMcpTools
from augmented.utils.info import (
    PROJECT_ROOT_DIR,
    DEFAULT_MODEL_NAME
)
from augmented.utils import pretty
from augmented.utils.info import DEFAULT_MODEL_NAME, PROJECT_ROOT_DIR

PRETTY_LOGGER = pretty.ALogger("[Agent]")

# 全局状态：保存对话历史和 Agent 实例（多轮对话支持）
class State:
    def __init__(self):
        self.agent: Agent | None = None
        self.chat_history: List[Dict[str, str]] = []  # 多轮对话历史
        self.console = Console(record=True)  # 捕获 rich 输出


state = State()


async def init_agent_if_needed(model_name: str, base_url: str, api_key: str) -> Agent:
    """初始化 Agent（仅在首次或模型参数变化时）"""
    if state.agent is None:
        # 复用你的 MCP 客户端配置（和 example() 完全一致）
        mcp_clients = []
        for mcp_tool in [
            PresetMcpTools.filesystem.append_mcp_params(f" {PROJECT_ROOT_DIR!s}"),
            PresetMcpTools.fetch,
            PresetMcpTools.commander
        ]:
            mcp_client = MCPClient(**mcp_tool.to_common_params())
            mcp_clients.append(mcp_client)

        # 完全复用你的 Agent 初始化逻辑（包括 system_prompt）
        state.agent = Agent(
            model=model_name or DEFAULT_MODEL_NAME,
            mcp_clients=mcp_clients,
        )
        # 初始化 LLM（保留你的 system_prompt）
        state.agent.llm = AsyncChatOpenAI(
            model_name or DEFAULT_MODEL_NAME,
            tools=[],  # 由 agent.init() 自动填充
            system_prompt=state.agent.system_prompt,  # 关键：保留你的 system prompt
            context=state.agent.context,
        )
        await state.agent.init()
        PRETTY_LOGGER.title("Agent 初始化完成")
    return state.agent


async def gradio_query(
        query: str,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float,
        chat_history: List[Dict[str, str]]
) -> AsyncGenerator[tuple[str, str, List[Dict[str, str]]], None]:
    """完全复用 Agent 的多轮对话和输出逻辑，分离工具调用和最终响应"""
    tool_call_text = ""  # 左侧工具调用详情
    response_text = ""  # 右侧最终响应内容

    try:
        # 1. 初始化 Agent
        agent = await init_agent_if_needed(model_name, base_url, api_key)
        agent.llm.temperature = temperature
        tool_call_text += "初始化完成，开始处理查询...\n"
        yield tool_call_text, response_text, chat_history

        # 2. 记录用户查询到历史
        chat_history.append({"role": "user", "content": query})

        # 3. 处理多轮工具调用
        chat_resp = await agent.llm.chat(query)
        i = 0

        while True:
            i += 1
            tool_call_text += f"\n=== 处理轮次 {i} ===\n"

            # 处理工具调用（只在左侧显示）
            if chat_resp.tool_calls:
                for tool_call in chat_resp.tool_calls:
                    tool_info = f"🛠️ TOOL USE: {tool_call.function.name}\n  ARGS: {tool_call.function.arguments}\n"
                    tool_call_text += tool_info
                    yield tool_call_text, response_text, chat_history

                    target_mcp_client = next(
                        (c for c in agent.mcp_clients if tool_call.function.name in [t.name for t in c.get_tools()]),
                        None
                    )

                    if target_mcp_client:
                        mcp_result = await target_mcp_client.call_tool(
                            tool_call.function.name,
                            json.loads(tool_call.function.arguments),
                        )
                        tool_call_text += f"✅ RESULT: {str(mcp_result)}\n\n"
                        yield tool_call_text, response_text, chat_history
                        agent.llm.append_tool_result(tool_call.id, mcp_result.model_dump_json())
                    else:
                        tool_call_text += f"❌ 错误: 工具未找到\n"
                        yield tool_call_text, response_text, chat_history
                        return

                # 继续下一轮对话
                chat_resp = await agent.llm.chat()
            else:
                # 4. 最终响应（只在右侧显示）
                response_text = chat_resp.content
                chat_history.append({"role": "assistant", "content": response_text})
                yield tool_call_text, response_text, chat_history
                break

    except Exception as e:
        error_msg = f"❌ 处理失败: {str(e)}"
        response_text = error_msg
        yield tool_call_text, response_text, chat_history
    finally:
        pass








# async def gradio_query(
#         query: str,
#         model_name: str,
#         base_url: str,
#         api_key: str,
#         temperature: float,
#         chat_history: List[Dict[str, str]]
# ) -> AsyncGenerator[tuple[str, str, List[Dict[str, str]]], None]:
#     """完全复用 Agent 的多轮对话和输出逻辑"""
#     tool_call_text = ""
#     full_response = ""
#     state.console.clear()  # 清空 rich 控制台缓存
#
#     try:
#         # 1. 初始化 Agent（复用你的核心逻辑）
#         agent = await init_agent_if_needed(model_name, base_url, api_key)
#         agent.llm.temperature = temperature  # 设置温度
#         yield tool_call_text, "初始化完成，开始处理查询...", chat_history
#
#         # 2. 维护多轮对话历史（新增）
#         state.chat_history = chat_history.copy()
#         state.chat_history.append({"role": "user", "content": query})
#
#         # 3. 复用 Agent 的多轮工具调用逻辑（改造为流式输出）
#         chat_resp = await agent.llm.chat(query)  # 首次调用
#         i = 0
#
#         while True:
#             # 输出轮次信息（和你的 Agent 输出一致）
#             轮次信息 = f"\n=== 处理轮次 {i} ==="
#             full_response += 轮次信息
#             yield tool_call_text, full_response, state.chat_history
#             i += 1
#
#             # 处理工具调用（完全复用你的逻辑）
#             if chat_resp.tool_calls:
#                 for tool_call in chat_resp.tool_calls:
#                     # 输出工具调用信息（和你的 rprint 一致）
#                     tool_info = f"TOOL USE `{tool_call.function.name}`\nwith args: {tool_call.function.arguments}"
#                     tool_call_text += tool_info + "\n"
#                     full_response += f"\n工具调用: {tool_call.function.name}\n"
#                     yield tool_call_text, full_response, state.chat_history
#
#                     # 查找目标 MCP 客户端（复用你的逻辑）
#                     target_mcp_client = next(
#                         (c for c in agent.mcp_clients if tool_call.function.name in [t.name for t in c.get_tools()]),
#                         None
#                     )
#
#                     if target_mcp_client:
#                         # 调用工具（复用你的 call_tool）
#                         mcp_result = await target_mcp_client.call_tool(
#                             tool_call.function.name,
#                             json.loads(tool_call.function.arguments),
#                         )
#                         # 输出工具返回结果（和你的 rprint 一致）
#                         result_str = f"工具返回: {str(mcp_result)}\n"
#                         full_response += result_str
#                         tool_call_text += result_str
#                         yield tool_call_text, full_response, state.chat_history
#
#                         # 记录工具结果（复用你的 append_tool_result）
#                         agent.llm.append_tool_result(tool_call.id, mcp_result.model_dump_json())
#                     else:
#                         error = "工具未找到"
#                         full_response += f"\n错误: {error}\n"
#                         yield tool_call_text, full_response, state.chat_history
#                         return
#
#                 # 多轮对话：继续调用 LLM（复用你的逻辑）
#                 chat_resp = await agent.llm.chat()
#             else:
#                 # 输出最终结果（和你的返回一致）
#                 final_resp = f"\n最终结果: {chat_resp.content}"
#                 full_response += final_resp
#                 state.chat_history.append({"role": "assistant", "content": chat_resp.content})
#                 yield tool_call_text, full_response, state.chat_history
#                 break
#
#     except Exception as e:
#         # 错误处理（复用你的异常输出）
#         error_msg = f"处理失败: {str(e)}"
#         yield tool_call_text, error_msg, state.chat_history
#     finally:
#         # 保留你的 cleanup 逻辑（不清理，支持多轮对话）
#         pass

# 初始化 rich 控制台（复用你的输出格式）
state.console = Console()

# Gradio 界面（保留你的输出风格）
with gr.Blocks(title="MCP Agent 交互界面") as demo:
    gr.Markdown("## 🤖 MCP Agent 交互平台（复用原始逻辑）")

    # 多轮对话历史状态（新增）
    chat_history = gr.State([])

    with gr.Row():
        # 左侧参数区
        with gr.Column(scale=1):
            gr.Markdown("### 🧠 模型配置")
            model_name = gr.Textbox(label="模型名称", value=DEFAULT_MODEL_NAME)
            base_url = gr.Textbox(label="API 地址", value=os.environ.get("OPENAI_BASE_URL"))
            api_key = gr.Textbox(label="API Key", type="password", value=os.environ.get("OPENAI_API_KEY"))
            temperature = gr.Slider(label="温度", minimum=0.0, maximum=1.0, value=0.0, step=0.1)  # 你的默认温度是 0

            # 工具调用记录（复用你的输出）
            tool_status = gr.Textbox(label="🛠️ 工具调用详情", lines=10, interactive=False)

        # 右侧输出区（保留你的输出格式）
        with gr.Column(scale=2):
            gr.Markdown("### 💬 输出结果（与 Agent 原生输出一致）")
            result_display = gr.Textbox(label="生成内容", lines=20, show_copy_button=True)

    # 底部输入区 + 多轮对话
    with gr.Row():
        query_input = gr.Textbox(label="❓ 输入查询", placeholder="请输入你的问题...", scale=4)
        generate_btn = gr.Button("🚀 开始处理", scale=1, variant="primary")

    # 绑定交互逻辑（多轮对话 + 流式输出）
    generate_btn.click(
        fn=gradio_query,
        inputs=[query_input, model_name, base_url, api_key, temperature, chat_history],
        outputs=[tool_status, result_display, chat_history]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="localhost", server_port=9999, auth=("zhangsan", "123456"))