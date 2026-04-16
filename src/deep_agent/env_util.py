"""
加载项目中的环境变量
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)
# deepseek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
# qwen3-8b 本地部署
QWEN_LOCAL_BASE_URL = os.getenv("QWEN_LOCAL_BASE_URL")
# qwen
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
# zhipu AI
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
# tavily
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
