# 酒店智能体
from langchain.agents.middleware import dynamic_prompt, ModelRequest

from deep_agent.agents.common import get_context_content, format_time
from deep_agent.context import CtripContext
from deep_agent.llms import qwen36p
from langchain.agents import create_agent
from deep_agent.tools.hotels_tools import search_hotels, book_hotel, cancel_hotel, update_hotel

# 酒店相关工具
hotel_safe_tool = [search_hotels, ]
hotel_sensitive_tools = [book_hotel, cancel_hotel, update_hotel]

HOTEL_AGENT_PROMPT = """
你是专门处理酒店查询、预订、取消和修改订单的智能体。

职责：
- 只处理酒店相关任务
- 不处理航班、租车、旅行推荐、联网搜索
- 回复必须简洁，不能输出思考过程，不能输出分析过程

规则：
1. 当用户要求查询酒店时，优先调用 search_hotels。
2. 如果 search_hotels 返回空列表，请直接回复：未查询到符合条件的酒店。
3. 如果查到结果，请只输出整理后的酒店信息。
4. 对于预订、取消、修改操作，只输出最终处理结果。
5. 不要输出推理过程，不要输出英文分析，不要解释内部判断过程。
""".strip()


# 2.动态提示词
@dynamic_prompt
def hotel_rental_dynamic_prompt(requset: ModelRequest):
    passenger_id = get_context_content(requset).get("passenger_id", None)
    now = format_time()
    return f"""
    当前系统时间：{now}
    当前用户 passenger_id：{passenger_id}

    补充规则：
    1. 查询酒店时优先调用 search_hotels。
    2. 如果用户提到今天、明天、下周、近期等相对时间，必须尽量理解为明确入住/退房时间范围。
    3. 如果 search_hotels 返回空列表，请直接回复：未查询到符合条件的酒店。
    4. 不要输出推理过程，不要输出英文分析。
    """.strip()


hotel_agent = create_agent(
    model=qwen36p,
    tools=hotel_safe_tool + hotel_sensitive_tools,
    system_prompt=HOTEL_AGENT_PROMPT,
    middleware=[hotel_rental_dynamic_prompt],
    context_schema=CtripContext,
    name='hotel_booking_agent',
)
