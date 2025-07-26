import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "D:/codes/pythonprojects/llm_related-main/exp-llm-mcp-rag"],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            print("初始化...")
            await session.initialize()
            print("初始化完成")
            resp = await session.list_tools()
            print("工具列表：", [tool.name for tool in resp.tools])

if __name__ == "__main__":
    asyncio.run(main())