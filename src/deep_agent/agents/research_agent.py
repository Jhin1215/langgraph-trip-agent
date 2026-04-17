from langchain.agents.middleware import dynamic_prompt, ModelRequest

from deep_agent.agents.common import get_context_content, format_time
from deep_agent.context import CtripContext
from deep_agent.llms import qwen36p
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from deep_agent.env_util import TAVILY_API_KEY

tavily_tool = TavilySearch(max_results=1, TAVILY_API_KEY=TAVILY_API_KEY)
# 网络搜索工具
research_tools = [tavily_tool]
# 网络搜素智能体
RESEARCH_PROMPT = """
你是网络搜索智能体。
职责：
- 只处理联网搜索、事实查询、外部信息收集
- 不处理航班、酒店、租车、旅行订单修改
- 回复只给结果，不输出多余说明
""".strip()


@dynamic_prompt
def research_dynamic_prompt(requset: ModelRequest):
    passenger_id = get_context_content(requset).get("passenger_id", None)
    now = format_time()
    return f"""
    当前系统时间：{now}
    当前用户 passenger_id：{passenger_id}

    补充规则：
    1. 你只负责联网搜索与外部信息查询，不处理航班、酒店、租车、旅行预订业务。
    2. 返回结果时，必须尽量基于搜索结果整理，不要主观编造。
    3. 如果没有检索到可靠结果，请直接回复：未查询到相关信息。
    4. 不要输出推理过程，不要输出英文分析，不要解释内部判断过程。
    """.strip()


research_agent = create_agent(
    model=qwen36p,
    tools=research_tools,
    system_prompt=RESEARCH_PROMPT,
    middleware=[research_dynamic_prompt,],
    context_schema=CtripContext,
    name='research_agent',
)
