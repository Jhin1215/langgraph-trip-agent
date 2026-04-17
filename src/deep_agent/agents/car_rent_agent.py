from langchain.agents.middleware import dynamic_prompt, ModelRequest

from deep_agent.agents.common import get_context_content, format_time
from deep_agent.context import CtripContext
from deep_agent.llms import qwen36p
from langchain.agents import create_agent

from deep_agent.tools.cars_tools import search_car_rentals, book_car_rental, cancel_car_rental, update_car_rental

# 租车相关工具
car_safe_tools = [search_car_rentals]
car_sensitive_tools = [book_car_rental, cancel_car_rental, update_car_rental]

# 租车智能体上下文工程
# 1.静态提示词
CAR_AGENT_PROMPT = """
你是专门处理租车查询、预订、取消和修改订单的智能体。

职责：
- 只处理租车相关任务
- 不处理航班、酒店、旅行推荐、联网搜索
- 回复必须简洁，不能输出思考过程，不能输出分析过程

规则：
1. 当用户要求查询租车时，优先调用 search_car_rentals。
2. 如果 search_car_rentals 返回空列表，请直接回复：未查询到符合条件的租车信息。
3. 如果查到结果，请只输出整理后的租车信息。
4. 对于预订、取消、修改操作，只输出最终处理结果。
5. 不要输出推理过程，不要输出英文分析，不要解释内部判断过程。
""".strip()


# 2.动态提示词
@dynamic_prompt
def car_rental_dynamic_prompt(requset: ModelRequest):
    passenger_id = get_context_content(requset).get("passenger_id", None)
    now = format_time()
    return f"""
    当前系统时间：{now}
    当前用户 passenger_id：{passenger_id}

    补充规则：
    1. 查询租车时优先调用 search_car_rentals。
    2. 如果用户提到“今天 / 明天 / 后天 / 下周 / 近期”等相对时间，必须将其理解为明确的租车时间范围，不能忽略时间条件。
    3. 如果 search_car_rentals 返回空列表，请直接回复：未查询到符合条件的租车信息。
    4. 对于 book_car_rental / update_car_rental / cancel_car_rental 这类写操作，只返回最终结果，不要输出过程。
    5. 不要输出推理过程，不要输出英文分析，不要解释内部判断过程。
    """.strip()


car_agent = create_agent(
    model=qwen36p,
    tools=car_safe_tools + car_sensitive_tools,
    system_prompt=CAR_AGENT_PROMPT,
    middleware=[car_rental_dynamic_prompt, ],
    context_schema=CtripContext,
    name='car_rental_booking_agent',
)
