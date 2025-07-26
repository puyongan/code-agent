# 修复包导入错误
import sys , os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

import sys
import os
import asyncio
import json
from typing import AsyncGenerator, List, Dict, Any

import gradio as gr
from rich.console import Console
from rich.text import Text

from augmented.chat_openai import AsyncChatOpenAI
from augmented.agent import Agent
from augmented.mcp_client import MCPClient
from augmented.mcp_tools import PresetMcpTools
from augmented.utils.info import (
    PROJECT_ROOT_DIR,
    DEFAULT_MODEL_NAME
)
from augmented.utils import pretty

PRETTY_LOGGER = pretty.ALogger("[Agent]")


class State:
    def __init__(self):
        self.agent: Agent | None = None
        self.chat_history: List[List[str]] = []  # Gradio格式：[[user, assistant], ...]
        self.console = Console(record=True)


state = State()


async def init_agent_if_needed(model_name: str, base_url: str, api_key: str) -> Agent:
    """初始化 Agent（仅在首次或模型参数变化时）"""
    if state.agent is None:
        mcp_clients = []
        for mcp_tool in [
            PresetMcpTools.filesystem.append_mcp_params(f" {PROJECT_ROOT_DIR!s}"),
            PresetMcpTools.fetch,
            PresetMcpTools.commander
        ]:
            mcp_client = MCPClient(**mcp_tool.to_common_params())
            mcp_clients.append(mcp_client)

        state.agent = Agent(
            model=model_name or DEFAULT_MODEL_NAME,
            mcp_clients=mcp_clients,
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
        history: List[List[str]]
) -> AsyncGenerator[tuple[List[List[str]], str], None]:
    """
    改进版：完全分离工具调用和对话内容
    返回: (chat_history, tool_logs)
    """
    tool_logs = "🔧 工具调用日志\n" + "=" * 50 + "\n"

    try:
        # 1. 初始化 Agent
        agent = await init_agent_if_needed(model_name, base_url, api_key)
        tool_logs += "✅ Agent初始化完成\n\n"

        # 2. 添加用户消息到历史（但先不显示assistant回复）
        current_history = history.copy()
        current_history.append([query, ""])  # 占位，等待回复
        yield current_history, tool_logs

        # 3. 处理多轮工具调用
        chat_resp = await agent.llm.chat(query)
        round_num = 0

        while True:
            round_num += 1
            tool_logs += f"📍 第 {round_num} 轮处理\n"
            yield current_history, tool_logs

            if chat_resp.tool_calls:
                # 工具调用阶段 - 只在左侧显示详情
                for idx, tool_call in enumerate(chat_resp.tool_calls, 1):
                    tool_logs += f"  🛠️ 工具 {idx}: {tool_call.function.name}\n"
                    tool_logs += f"     参数: {tool_call.function.arguments}\n"
                    yield current_history, tool_logs

                    # 执行工具
                    target_mcp_client = next(
                        (c for c in agent.mcp_clients
                         if tool_call.function.name in [t.name for t in c.get_tools()]),
                        None
                    )

                    if target_mcp_client:
                        mcp_result = await target_mcp_client.call_tool(
                            tool_call.function.name,
                            json.loads(tool_call.function.arguments),
                        )
                        result_preview = str(mcp_result)[:200] + "..." if len(str(mcp_result)) > 200 else str(
                            mcp_result)
                        tool_logs += f"     ✅ 结果: {result_preview}\n\n"
                        yield current_history, tool_logs

                        agent.llm.append_tool_result(tool_call.id, mcp_result.model_dump_json())
                    else:
                        tool_logs += f"     ❌ 错误: 工具未找到\n\n"
                        yield current_history, tool_logs
                        return

                # 继续下一轮
                chat_resp = await agent.llm.chat()
            else:
                # 最终响应阶段 - 更新右侧对话历史
                final_response = chat_resp.content
                current_history[-1][1] = final_response  # 填充assistant回复
                tool_logs += f"✅ 处理完成，共 {round_num} 轮\n"
                yield current_history, tool_logs
                break

    except Exception as e:
        error_msg = f"❌ 处理失败: {str(e)}"
        tool_logs += f"\n{error_msg}\n"
        if current_history and current_history[-1][1] == "":
            current_history[-1][1] = error_msg
        yield current_history, tool_logs


def clear_history():
    """清空对话历史"""
    return [], "🔧 工具调用日志\n" + "=" * 50 + "\n已清空历史\n"


def reset_agent():
    """重置Agent"""
    global state
    if state.agent:
        # 这里可以添加cleanup逻辑
        state.agent = None
    return [], "🔧 工具调用日志\n" + "=" * 50 + "\n已重置Agent\n"


# Gradio 界面
with gr.Blocks(title="Schneider Agent 交互界面", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Schneider Agent 交互平台")
    gr.Markdown("左侧显示工具调用详情，右侧显示对话内容，支持多轮连续对话")

    with gr.Row():
        # 左侧：配置 + 工具日志
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 模型配置")
            model_name = gr.Textbox(
                label="模型名称",
                value=DEFAULT_MODEL_NAME,
                placeholder="如: gpt-4, claude-3-5-sonnet"
            )
            base_url = gr.Textbox(
                label="API 地址",
                value=os.environ.get("OPENAI_BASE_URL", ""),
                placeholder="如: https://api.openai.com/v1"
            )
            api_key = gr.Textbox(
                label="API Key",
                type="password",
                value=os.environ.get("OPENAI_API_KEY", ""),
                placeholder="输入你的API密钥"
            )
            temperature = gr.Slider(
                label="温度",
                minimum=0.0,
                maximum=1.0,
                value=0.1,
                step=0.1
            )

            with gr.Row():
                clear_btn = gr.Button("🗑️ 清空历史", size="sm")
                reset_btn = gr.Button("🔄 重置Agent", size="sm")

            # 工具调用日志区域
            tool_status = gr.Textbox(
                label="🔧 工具调用详情",
                value="🔧 工具调用日志\n" + "=" * 50 + "\n等待查询...\n",
                lines=15,
                max_lines=20,
                interactive=False,
                show_copy_button=True
            )

        # 右侧：对话区域
        with gr.Column(scale=2):
            gr.Markdown("### 💬 对话历史")

            # 使用Chatbot组件显示对话历史
            chatbot = gr.Chatbot(
                value=[],
                height=400,
                show_copy_button=True,
                bubble_full_width=False,
                show_share_button=False
            )

            # 输入区域
            with gr.Row():
                msg_input = gr.Textbox(
                    label="输入消息",
                    placeholder="请输入你的问题...",
                    scale=4,
                    show_label=False
                )
                send_btn = gr.Button("🚀 发送", scale=1, variant="primary")

            gr.Markdown("💡 **提示**: 支持多轮对话，工具调用详情会显示在左侧")

    # 绑定事件
    send_btn.click(
        fn=gradio_query,
        inputs=[msg_input, model_name, base_url, api_key, temperature, chatbot],
        outputs=[chatbot, tool_status]
    ).then(
        lambda: "",  # 清空输入框
        outputs=[msg_input]
    )

    # 回车发送
    msg_input.submit(
        fn=gradio_query,
        inputs=[msg_input, model_name, base_url, api_key, temperature, chatbot],
        outputs=[chatbot, tool_status]
    ).then(
        lambda: "",
        outputs=[msg_input]
    )

    # 清空和重置按钮
    clear_btn.click(
        fn=clear_history,
        outputs=[chatbot, tool_status]
    )

    reset_btn.click(
        fn=reset_agent,
        outputs=[chatbot, tool_status]
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="localhost",
        server_port=9999,
        auth=("zhangsan", "123456"),
        share=False,
        debug=True
    )