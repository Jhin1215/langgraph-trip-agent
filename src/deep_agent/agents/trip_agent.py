from langchain.agents.middleware import dynamic_prompt, ModelRequest

from deep_agent.agents.common import get_context_content, format_time
from deep_agent.context import CtripContext
from deep_agent.llms import qwen36p
from langchain.agents import create_agent

from deep_agent.tools.trip_tools import search_trip_recommendations, book_excursion, cancel_excursion, update_excursion

# 旅行推荐相关工具
trip_safe_tools = [search_trip_recommendations]
trip_sensitive_tools = [book_excursion, cancel_excursion, update_excursion]
# 旅行智能体
TRIP_AGENT_PROMPT = """
你是专门处理旅行推荐查询、预订、取消和修改订单的智能体。

职责：
- 只处理旅行产品相关任务
- 不处理航班、酒店、租车、联网搜索
- 回复必须简洁，不能输出思考过程，不能输出分析过程

规则：
1. 当用户要求查询旅行推荐时，优先调用 search_trip_recommendations。
2. 如果 search_trip_recommendations 返回空列表，请直接回复：未查询到符合条件的旅行产品。
3. 如果查到结果，请只输出整理后的旅行产品信息。
4. 对于预订、取消、修改操作，只输出最终处理结果。
5. 不要输出推理过程，不要输出英文分析，不要解释内部判断过程。
""".strip()


# 2.动态提示词
@dynamic_prompt
def trip_rental_dynamic_prompt(requset: ModelRequest):
    passenger_id = get_context_content(requset).get("passenger_id", None)
    now = format_time()
    return f"""
    当前系统时间：{now}
    当前用户 passenger_id：{passenger_id}

    补充规则：
    1. 查询旅行产品时优先调用 search_trip_recommendations。
    2. 如果用户提到“今天 / 明天 / 后天 / 下周 / 近期”等相对时间，必须将其理解为明确的出行时间范围，不能忽略时间条件。
    3. 如果 search_trip_recommendations 返回空列表，请直接回复：未查询到符合条件的旅行产品。
    4. 对于 book_excursion / update_excursion / cancel_excursion 这类写操作，只返回最终结果，不要输出过程。
    5. 不要输出推理过程，不要输出英文分析，不要解释内部判断过程。
    """.strip()


trip_agent = create_agent(
    model=qwen36p,
    tools=trip_safe_tools + trip_sensitive_tools,
    system_prompt=TRIP_AGENT_PROMPT,
    middleware=[trip_rental_dynamic_prompt, ],
    context_schema=CtripContext,
    name='trip_booking_agent',
)
