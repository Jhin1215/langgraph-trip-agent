from langchain.agents.middleware import dynamic_prompt, ModelRequest

from deep_agent.agents.common import get_context_content, format_time
from deep_agent.context import CtripContext
from deep_agent.llms import qwen36p
from langchain.agents import create_agent

from deep_agent.tools.flights_tools import fetch_user_flight_information, search_flights, update_ticket_to_new_flight, \
    cancel_ticket
from deep_agent.tools.retrieve_tools import lookup_policy

# 航班相关工具
flight_safe_tools = [fetch_user_flight_information, search_flights, lookup_policy, ]
flight_sensitive_tools = [update_ticket_to_new_flight, cancel_ticket, ]

# 航班智能体上下文工程
# 静态提示词
FLIGHT_AGENT_PROMPT = """
你是专门处理航班查询、改签、退票和航班政策的智能体。

职责：
- 只处理航班相关任务
- 不处理酒店、租车、旅行推荐、联网搜索
- 回复必须简洁，不能输出思考过程，不能输出分析过程

规则：
1. 当用户要求查看自己当前航班信息时，优先调用 fetch_user_flight_information。
2. 如果 fetch_user_flight_information 返回空列表，请直接回复：未查询到您的航班信息。
3. 如果查到结果，请只输出整理后的航班信息。
4. 不要输出推理过程，不要输出英文分析，不要解释内部判断过程。
""".strip()


# 动态提示词
@dynamic_prompt
def flight_dynamic_prompt(request: ModelRequest) -> str:
    passenger_id = get_context_content(request).get('passenger_id', None)
    now = format_time()
    return f"""
    当前系统时间：{now}
    当前用户 passenger_id：{passenger_id}

    补充规则：
    1. 查询当前用户自己的机票信息时，优先调用 fetch_user_flight_information。
    2. 查询航班列表时，优先调用 search_flights。
    3. 如果用户说“今天 / 明天 / 后天 / 未来一周 / 下周 / 近期”等相对时间，必须将其理解为明确的时间范围，并在调用 search_flights 时尽量传入 start_time 和 end_time，不能忽略时间条件。
    4. 如果 fetch_user_flight_information 返回空列表，请直接回复：未查询到您的航班信息。
    5. 如果 search_flights 返回空列表，请直接回复：未查询到符合条件的航班信息。
    6. 输出航班信息时，必须严格使用工具返回的原始字段值，不得自行补全或改写 flight_no。
    7. 不要输出推理过程，不要输出英文分析，不要解释内部判断过程。
    """.strip()


flight_agent = create_agent(
    model=qwen36p,
    tools=flight_safe_tools + flight_sensitive_tools,
    system_prompt=FLIGHT_AGENT_PROMPT,
    context_schema=CtripContext,
    name='flight_booking_agent',
)
