# 修复包导入错误
import sys, os

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
    改进版：支持流式输出和完整工具调用日志
    返回: (chat_history, tool_logs)
    """
    tool_logs = "🔧 工具调用日志\n" + "=" * 80 + "\n"

    try:
        # 1. 初始化 Agent
        agent = await init_agent_if_needed(model_name, base_url, api_key)
        tool_logs += "✅ Agent初始化完成\n\n"

        # 2. 添加用户消息到历史
        current_history = history.copy()
        current_history.append([query, ""])  # 占位，等待回复
        yield current_history, tool_logs

        # 3. 处理多轮工具调用
        chat_resp = await agent.llm.chat(query)
        round_num = 0

        while True:
            round_num += 1
            tool_logs += f"📍 第 {round_num} 轮处理\n"
            tool_logs += f"{'─' * 50}\n"
            yield current_history, tool_logs

            if chat_resp.tool_calls:
                # 工具调用阶段 - 显示完整详情
                tool_logs += f"🛠️ 检测到 {len(chat_resp.tool_calls)} 个工具调用\n\n"

                for idx, tool_call in enumerate(chat_resp.tool_calls, 1):
                    tool_logs += f"【工具 {idx}】{tool_call.function.name}\n"

                    # 格式化参数显示
                    try:
                        args = json.loads(tool_call.function.arguments)
                        formatted_args = json.dumps(args, indent=2, ensure_ascii=False)
                        tool_logs += f"参数:\n{formatted_args}\n"
                    except:
                        tool_logs += f"参数: {tool_call.function.arguments}\n"

                    yield current_history, tool_logs

                    # 执行工具
                    target_mcp_client = next(
                        (c for c in agent.mcp_clients
                         if tool_call.function.name in [t.name for t in c.get_tools()]),
                        None
                    )

                    if target_mcp_client:
                        tool_logs += "⏳ 执行中...\n"
                        yield current_history, tool_logs

                        mcp_result = await target_mcp_client.call_tool(
                            tool_call.function.name,
                            json.loads(tool_call.function.arguments),
                        )

                        # 显示完整结果
                        result_str = str(mcp_result)
                        if len(result_str) > 1000:
                            tool_logs += f"✅ 结果 (前1000字符):\n{result_str[:1000]}\n...[结果太长已截断]\n\n"
                        else:
                            tool_logs += f"✅ 结果:\n{result_str}\n\n"
                        yield current_history, tool_logs

                        agent.llm.append_tool_result(tool_call.id, mcp_result.model_dump_json())
                    else:
                        tool_logs += f"❌ 错误: 工具 '{tool_call.function.name}' 未找到\n\n"
                        yield current_history, tool_logs
                        return

                # 继续下一轮
                tool_logs += "🔄 继续处理...\n\n"
                yield current_history, tool_logs
                chat_resp = await agent.llm.chat()
            else:
                # 最终响应阶段 - 流式显示
                final_response = chat_resp.content
                tool_logs += f"✅ 生成最终回复 ({len(final_response)} 字符)\n"
                tool_logs += f"{'=' * 80}\n"
                tool_logs += f"总计 {round_num} 轮处理完成\n"

                # 模拟流式输出效果
                current_text = ""
                words = final_response.split()

                for i, word in enumerate(words):
                    current_text += word + " "
                    current_history[-1][1] = current_text.strip()
                    yield current_history, tool_logs

                    # 控制输出速度，让效果更自然
                    if i % 3 == 0:  # 每3个词暂停一下
                        await asyncio.sleep(0.05)

                # 确保最终文本完整
                current_history[-1][1] = final_response
                yield current_history, tool_logs
                break

    except Exception as e:
        error_msg = f"❌ 处理失败: {str(e)}"
        tool_logs += f"\n{error_msg}\n"
        tool_logs += f"错误详情: {repr(e)}\n"
        if current_history and current_history[-1][1] == "":
            current_history[-1][1] = error_msg
        yield current_history, tool_logs


def clear_history():
    """清空对话历史"""
    return [], "🔧 工具调用日志\n" + "=" * 80 + "\n✨ 历史已清空，可以开始新对话\n"


def reset_agent():
    """重置Agent"""
    global state
    if state.agent:
        # 这里可以添加cleanup逻辑
        state.agent = None
    return [], "🔧 工具调用日志\n" + "=" * 80 + "\n🔄 Agent已重置，下次查询将重新初始化\n"


# Gradio 界面
with gr.Blocks(title="Schneider Agent 交互界面", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Schneider Agent 交互平台")

    with gr.Row():
        # 左侧：工具日志 + 配置（紧凑布局）
        with gr.Column(scale=1, min_width=400):
            # 工具调用日志区域 - 占主要空间
            tool_status = gr.Textbox(
                label="🔧 工具调用详情",
                value="🔧 工具调用日志\n" + "=" * 80 + "\n💡 等待查询，工具调用详情将在此显示...\n",
                lines=20,
                max_lines=25,
                interactive=False,
                show_copy_button=True,
                container=True
            )

            # 配置区域 - 可折叠，紧凑显示
            with gr.Accordion("⚙️ 模型配置", open=False):
                with gr.Row():
                    model_name = gr.Textbox(
                        label="模型",
                        value=DEFAULT_MODEL_NAME,
                        placeholder="gpt-4",
                        scale=2
                    )
                    temperature = gr.Slider(
                        label="温度",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.1,
                        step=0.1,
                        scale=1
                    )

                base_url = gr.Textbox(
                    label="API地址",
                    value=os.environ.get("OPENAI_BASE_URL", ""),
                    placeholder="https://api.openai.com/v1",
                    lines=1
                )
                api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    value=os.environ.get("OPENAI_API_KEY", ""),
                    placeholder="输入API密钥",
                    lines=1
                )

                with gr.Row():
                    clear_btn = gr.Button("🗑️ 清空", size="sm", scale=1)
                    reset_btn = gr.Button("🔄 重置", size="sm", scale=1)

        # 右侧：对话区域（占主要空间）
        with gr.Column(scale=2):
            gr.Markdown("### 💬 对话区域")

            # 对话历史显示
            chatbot = gr.Chatbot(
                value=[],
                height=300,
                show_copy_button=True,
                bubble_full_width=False,
                show_share_button=False,
                avatar_images=None,
                container=True
            )

            # 输入区域
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="💬 请输入你的问题，支持多轮对话...",
                    scale=5,
                    show_label=False,
                    lines=1,
                    max_lines=3
                )
                send_btn = gr.Button("🚀 发送", scale=1, variant="primary")

    # 底部提示
    gr.Markdown("""
    <div style='text-align: center; color: #666; font-size: 12px; margin-top: 10px;'>
    💡 <b>使用提示</b>: 支持连续对话 | 工具调用详情实时显示在左侧 | 回复支持流式输出
    </div>
    """)

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