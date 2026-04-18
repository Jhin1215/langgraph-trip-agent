from langchain.chat_models import init_chat_model
from zhipuai import ZhipuAI
from deep_agent.env_util import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, QWEN_BASE_URL, QWEN_API_KEY, ZHIPU_API_KEY

deepseek_v1 = init_chat_model(
    model='deepseek-chat',
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

qwen36p = init_chat_model(
    # qwen3.6plus没有额度了
    # model='qwen3.6-plus',
    model='qwen3-32b',
    model_provider='openai',
    base_url=QWEN_BASE_URL,
    api_key=QWEN_API_KEY,
    # qwen3-32b 需要把这个关掉
    extra_body={"enable_thinking": False},
)

zhipuai_client = ZhipuAI(api_key=ZHIPU_API_KEY)
