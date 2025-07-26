import sys, os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import asyncio
from dataclasses import dataclass
import json
from rich import print as rprint
from augmented.chat_openai import AsyncChatOpenAI
from augmented.mcp_client import MCPClient
from augmented.mcp_tools import PresetMcpTools
from augmented.utils import pretty
from augmented.utils.info import DEFAULT_MODEL_NAME, PROJECT_ROOT_DIR

PRETTY_LOGGER = pretty.ALogger("[Agent]")

@dataclass
class Agent:
    mcp_clients: list[MCPClient]
    model: str
    llm: AsyncChatOpenAI | None = None
    system_prompt: str = '''
对于代码执行，你可以使用基于Desktop Commander的commander工具
你仅可使用以下工具，且需严格遵守安全规则：

### 允许使用的工具
1. **终端操作**：start_process（执行命令）、read_process_output（读取输出）、list_sessions（查看会话）、force_terminate（终止进程）
2. **文件读取**：list_directory（列出目录）、read_file（读取文件）、write_file(创建新文件）、read_multiple_files（批量读取）、search_files（搜索文件）、get_file_info（查看文件信息）

### 严格禁止的操作
- 禁止使用任何文件修改工具（move_file、edit_block等）
- 禁止修改服务器配置（set_config_value等）
- 禁止执行删除、格式化等危险命令

### 执行规则
1. **终端命令需指定shell**
2. 执行命令时必须设置timeout_ms（建议30000ms以内）
3. 若命令执行出错或工具调用失败，立即终止流程并反馈错误，不得尝试修改代码或配置
4. 所有操作仅可读取内容和执行命令，不得对代码文件做任何改动
    '''

    context: str = ""

    async def init(self) -> None:
        PRETTY_LOGGER.title("INIT LLM&TOOLS")
        tools = []
        for mcp_client in self.mcp_clients:
            await mcp_client.init()
            tools.extend(mcp_client.get_tools())
        self.llm = AsyncChatOpenAI(
            self.model,
            tools=tools,
            system_prompt=self.system_prompt,
            context=self.context,
        )

    # async def cleanup(self) -> None:
    #     PRETTY_LOGGER.title("CLEANUP LLM&TOOLS")
    #
    #     while self.mcp_clients:
    #         # NOTE: 需要先处理其他依赖于mcp_client的资源, 不然会有一堆错误, 如
    #         # RuntimeError: Attempted to exit cancel scope in a different task than it was entered in
    #         # RuntimeError: Attempted to exit a cancel scope that isn't the current tasks's current cancel scope an error occurred during closing of asynchronous generator <async_generator object stdio_client at 0x76c3e08ee0c0>
    #         mcp_client = self.mcp_clients.pop()
    #         await mcp_client.cleanup()
    #

    async def cleanup(self) -> None:
        PRETTY_LOGGER.title("CLEANUP LLM&TOOLS")

        # 取消 LLM 相关的异步任务（如果有）
        if self.llm:
            # 假设 llm 内部有任务列表
            if hasattr(self.llm, 'tasks'):
                for task in self.llm.tasks:
                    task.cancel()
                await asyncio.gather(*self.llm.tasks, return_exceptions=True)

        # 清理 mcp_clients
        while self.mcp_clients:
            mcp_client = self.mcp_clients.pop()
            await mcp_client.cleanup()  # 确保这里完全完成

        # 等待所有任务结束
        await asyncio.sleep(0.1)  # 短暂延迟，确保资源释放




    async def invoke(self, prompt: str) -> str | None:
        return await self._invoke(prompt)

    async def _invoke(self, prompt: str) -> str | None:
        if self.llm is None:
            raise ValueError("llm not call .init()")
        chat_resp = await self.llm.chat(prompt)
        i = 0
        while True:
            PRETTY_LOGGER.title(f"INVOKE CYCLE {i}")
            i += 1
            # 处理工具调用
            rprint(chat_resp)
            if chat_resp.tool_calls:
                
                rprint("工具返回原始结果:", chat_resp)  # 查看爬取的原始内容
                for tool_call in chat_resp.tool_calls:
                    target_mcp_client: MCPClient | None = None
                    for mcp_client in self.mcp_clients:
                        if tool_call.function.name in [
                            t.name for t in mcp_client.get_tools()
                        ]:
                            target_mcp_client = mcp_client
                            break
                    if target_mcp_client:
                        PRETTY_LOGGER.title(f"TOOL USE `{tool_call.function.name}`")
                        rprint("with args:", tool_call.function.arguments)
                        mcp_result = await target_mcp_client.call_tool(
                            tool_call.function.name,
                            json.loads(tool_call.function.arguments),
                        )
                        rprint("call result:", mcp_result)
                        self.llm.append_tool_result(
                            tool_call.id, mcp_result.model_dump_json()
                        )
                    else:
                        self.llm.append_tool_result(tool_call.id, "tool not found")
                chat_resp = await self.llm.chat()
            else:
                return chat_resp.content


async def example() -> None:
    enabled_mcp_clients = []
    agent = None
    try:
        for mcp_tool in [
            PresetMcpTools.filesystem.append_mcp_params(f" {PROJECT_ROOT_DIR!s}"),
            PresetMcpTools.fetch,
            PresetMcpTools.commander
        ]:
            rprint(mcp_tool.shell_cmd)
            mcp_client = MCPClient(**mcp_tool.to_common_params())
            enabled_mcp_clients.append(mcp_client)

        agent = Agent(
            model=DEFAULT_MODEL_NAME,
            mcp_clients=enabled_mcp_clients,
        )
        await agent.init()

        resp = await agent.invoke(
            f'''
        现在需要你完成以下操作：
        1.阅读{PROJECT_ROOT_DIR!s}下的newdistance.py,理解我们的工作流程
        2.撰写简要的总结文档保存在 {PROJECT_ROOT_DIR / 'ai_gen'!s} 目录下的code_summary.md文件中
        3.根据理解撰写合适的命令以五角星模式运行该python脚本,且只能运行一次不可重复运行
        4.读取脚本输出并最终进行测试结果总结文档保存在 {PROJECT_ROOT_DIR / 'ai_gen'!s} 目录下的test_summary.md文件中
            '''
        )
        rprint(resp)
    except Exception as e:
        rprint(f"Error during agent execution: {e!s}")
        raise
    finally:
        if agent:
            await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(example())
