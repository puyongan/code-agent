# code-agent：基于mcp的代码读取及终端执行的agent

## 项目简介

本项目是一个基于大语言模型（LLM）、模型上下文协议（MCP）和检索增强生成（RAG）的实验性项目。它展示了如何构建一个能够与外部工具交互并利用检索增强生成技术的 AI 助手系统。
参考项目：https://github.com/StrayDragon/exp-llm-mcp-rag

  ```mermaid ​ ​   graph TD ​      %% 核心节点 ​      U[用户] ​      G[Gradio界面] ​      S[状态管理] ​      A[Agent代理] ​      L[LLM模型] ​      M[MCP客户端] ​      Log[日志工具] ​       ​      %% 主流程 ​      U -->|输入查询/点击发送| G ​      G -->|获取状态| S ​      S -->|返回状态数据| G ​      G -->|初始化/调用| A ​      A -->|需要模型处理| L ​      L -->|生成响应/工具调用| A ​      A -->|需要工具执行| M ​      M -->|返回工具结果| A ​      A -->|整理结果| G ​      G -->|流式展示回复| U ​       ​      %% 辅助流程 ​      U -->|点击清空| G ​      G -->|重置历史| S ​      U -->|点击重置| G ​      G -->|销毁Agent| S ​      A -->|记录操作| Log ​      M -->|记录调用| Log   ```

### 核心功能

- 基于 OpenAI API 的大语言模型调用
- 通过 MCP（Model Context Protocol）实现 LLM 与外部工具的交互
- 使用了第三方 MCP server: Desktop Commander
- 支持文件系统操作和网页内容获取以及终端命令交互

## 快速开始



### 环境准备

1. 确保已安装 Python 3.12 或更高版本
2. 克隆本仓库
3. 在`.env` 并填写必要的配置信息：
   - `OPENAI_API_KEY`: OpenAI API 密钥
   - `OPENAI_BASE_URL`: OpenAI API 基础 URL, 注意要保留后面的'/v1' (默认为 'https://api.openai.com/v1')
   - `DEFAULT_MODEL_NAME`: (可选) 默认使用的模型名称（默认为 "gpt-4o-mini"）
   - `EMBEDDING_KEY`: (可选) 嵌入模型 API 密钥（默认为 $OPENAI_API_KEY）
   - `EMBEDDING_BASE_URL`: (可选) 嵌入模型 API 基础 URL, 如硅基流动的API或兼容OpenAI格式的API （默认为 $OPENAI_BASE_URL）
   - `USE_CN_MIRROR`: (可选) 是否使用中国镜像, 设置任意值(如'1')为 true (默认为 false)
   - `PROXY_URL`: (可选) 代理 URL (如 "http(s)://xxx"), 用于 `fetch` (mcp-tool) 走代理

### 安装Node.js(包含了npm，npx等)

到官网安装最新版本
https://nodejs.org/en/blog/release/v20.19.3


### 安装scoop包管理工具

```shell
# 安装 scoop
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser  -Force
Invoke-Expression (Invoke-RestMethod -Uri https://get.scoop.sh)

```

### 安装uv包管理工具

```bash
# 安装 uv 
scoop install main/uv
```

### 安装依赖

```bash
# 使用 uv 安装依赖
uv sync
uv pip install numpy
uv pip install opencv-python
uv pip install matplotlib
uv pip install pillow
```

### 激活虚拟环境

```bash
# 使用 uv 安装依赖
.venv/Scripts/activate
```

### 安装desktop-commander

```bash
# 使用 npx 安装依赖
npx -y --verbose @wonderwhy-er/desktop-commander@latest setup --ignore-scripts
```

### 运行示例

本项目使用 `just` 命令工具来运行不同的示例：

```bash
# 查看可用命令
just help
```

### 启动聊天

```bash
# chat_openai启动
just chat_openai
```

### 启动agent

```bash
# agent启动
just agent
```

## 项目结构

- `src/augmented/`: 主要源代码目录
  - `agent.py`: Agent 实现，负责协调 LLM 和工具
  - `chat_openai.py`: OpenAI API 客户端封装
  - `mcp_client.py`: MCP 客户端实现
  - `mcp_tools.py`: MCP 工具定义
  - `utils/`: 工具函数
    - `info.py`: 项目信息和配置
    - `pretty.py`: 统一日志输出系统
- `justfile`: 任务运行配置文件
 
