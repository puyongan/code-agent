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
    规则：
    1. 只读不写，禁止修改代码文件
    2. 可用工具：start_process(需shell+timeout≤30000)、read_file、write_file(仅新建)、list_directory等
    3. 禁止：文件修改、危险命令、重复执行
    4. 出错立即停止并报告
    5. 每步说明：当前操作-预期结果-实际结果
    6. 默认在项目根目录下的ai_gen文件夹中保存你生成的所有文件

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

                        # 添加JSON解析错误处理
                        try:
                            parsed_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError as e:
                            rprint(f"JSON parsing error: {e}")
                            rprint(f"Raw arguments: {tool_call.function.arguments}")
                            # 尝试修复常见的JSON问题
                            fixed_args = tool_call.function.arguments.replace('\n', '\\n').replace('\r', '\\r').replace(
                                '\t', '\\t')
                            try:
                                parsed_args = json.loads(fixed_args)
                                rprint("Successfully fixed JSON format")
                            except json.JSONDecodeError:
                                rprint("Failed to fix JSON, using empty dict")
                                parsed_args = {}

                        mcp_result = await target_mcp_client.call_tool(
                            tool_call.function.name,
                            parsed_args,
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
        请按顺序完成：
        1. 读取{PROJECT_ROOT_DIR!s}/newdistance.py并理解代码
        2. 写代码总结到{PROJECT_ROOT_DIR / 'ai_gen'!s}/code_summary.md
        3. 以五角星模式运行脚本一次：python newdistance.py 1
        4. 写测试结果到{PROJECT_ROOT_DIR / 'ai_gen'!s}/test_summary.md
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
