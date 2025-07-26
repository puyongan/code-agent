# NOTE: now using python-dotenv to inject .env files, not need these configs
# set dotenv-required
# set dotenv-load
# set dotenv-filename := '.env.local'
# 使用 PowerShell 替代 sh:
set shell := ["C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe", "-c"]

default: help

help:
    @powershell -Command "just -l"

format-and-lintfix:
    @powershell -Command "ruff format"
    @powershell -Command "ruff check --fix"

dev-all-example-check:
    @powershell -Command "mv output output_bak -ErrorAction SilentlyContinue"
    just pretty
    just chat_openai
    just mcp_client
    just agent
    just gradio
    @powershell -Command "rm -rf output"
    @powershell -Command "mv output_bak output"

# step 0: init with utils
pretty:
	python src/augmented/utils/pretty.py

# step 1: impl ChatOpenai
chat_openai:
	python src/augmented/chat_openai.py

# step 2: impl MCPClient
mcp_client:
	python src/augmented/mcp_client.py

# step 3: impl Agent
agent:
	python src/augmented/agent.py

# step 4: impl Gradio
gradio:
	python src/augmented/gradio_mcp_client.py
