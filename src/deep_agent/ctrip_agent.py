from langchain.tools import tool, ToolRuntime
from langchain_core.messages import AIMessage
from langgraph.types import Command

from deep_agent.context import CtripContext
from deep_agent.llms import qwen36p
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

from deep_agent.tools.cars_tools import search_car_rentals, book_car_rental, cancel_car_rental, update_car_rental
from deep_agent.tools.flights_tools import fetch_user_flight_information, search_flights, update_ticket_to_new_flight, \
    cancel_ticket
from deep_agent.tools.hotels_tools import search_hotels, book_hotel, cancel_hotel, update_hotel
from deep_agent.tools.retrieve_tools import lookup_policy
from deep_agent.tools.trip_tools import search_trip_recommendations, book_excursion, cancel_excursion, update_excursion
from deep_agent.env_util import TAVILY_API_KEY

tavily_tool = TavilySearch(max_results=1, TAVILY_API_KEY=TAVILY_API_KEY)
# 网络搜索工具
research_tools = [tavily_tool]
# 航班相关工具
flight_safe_tools = [fetch_user_flight_information, search_flights, lookup_policy, ]
flight_sensitive_tools = [update_ticket_to_new_flight, cancel_ticket, ]
# 酒店相关工具
hotel_safe_tool = [search_hotels, ]
hotel_sensitive_tools = [book_hotel, cancel_hotel, update_hotel]
# 旅行推荐相关工具
trip_safe_tools = [search_trip_recommendations]
trip_sensitive_tools = [book_excursion, cancel_excursion, update_excursion]
# 租车相关工具
car_safe_tools = [search_car_rentals]
car_sensitive_tools = [book_car_rental, cancel_car_rental, update_car_rental]

# V1 每一个智能体对应一个静态提示词
# 网络搜素智能体
RESEARCH_PROMPT = """
你是网络搜索智能体。
职责：
- 只处理联网搜索、事实查询、外部信息收集
- 不处理航班、酒店、租车、旅行订单修改
- 回复只给结果，不输出多余说明
""".strip()
# 航班智能体
# 之前没有加入规则5，直接导致大模型发散思考了。结果乱七八糟
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
# 酒店智能体
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
# 租车智能体
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


# 创建多智能体编排方式
# langgraph 里面提供了两种多智能体编排方式
# 1.supervisor 创建一个中心 supervisor 负责 orchestrate 多个 specialized agents
# 2.warm：没有中心supervisor，多个 agent 根据专长动态 hand off 控制权。
def create_handoff_tool(*, agent_name: str, description: str):
    """
    根据传入智能体名字和描述 hand off control
    Args:
        agent_name: 智能体名字
        description: 什么时候该调用 agent_name这个专用智能体的相关描述
    Returns:
        返回一个可调用的函数
    """
    tool_name = f"transfer_to_{agent_name}"

    @tool(tool_name, description=description)
    def handoff_control(runtime: ToolRuntime) -> Command:
        # 创建一个只包含 aimessages 的生成器对象
        it = (msg for msg in reversed(runtime.state['messages']) if isinstance(msg, AIMessage))
        last_ai_msg = next(it, None)
        tool_message = {
            'role': 'tool',
            'content': f'Successfully transfer to {agent_name}',
            'name': tool_name,
            'tool_call_id': runtime.tool_call_id,
        }
        # 没找到 AIMessage 时，只传 ToolMessage，避免把 None 塞进 messages
        if last_ai_msg is None:
            update_messages = [tool_message]
        else:
            update_messages = [last_ai_msg, tool_message]

        return Command(
            goto=agent_name,
            update={"messages": update_messages},
            graph=Command.PARENT,
        )

    return handoff_control


# 创建多智能体
research_agent = create_agent(
    model=qwen36p,
    tools=research_tools,
    system_prompt=RESEARCH_PROMPT,
    context_schema=CtripContext,
    name='research_agent',
)
flight_agent = create_agent(
    model=qwen36p,
    tools=flight_safe_tools + flight_sensitive_tools,
    system_prompt=FLIGHT_AGENT_PROMPT,
    context_schema=CtripContext,
    name='flight_booking_agent',
)
hotel_agent = create_agent(
    model=qwen36p,
    tools=hotel_safe_tool + hotel_sensitive_tools,
    system_prompt=HOTEL_AGENT_PROMPT,
    context_schema=CtripContext,
    name='hotel_booking_agent',
)
car_agent = create_agent(
    model=qwen36p,
    tools=car_safe_tools + car_sensitive_tools,
    system_prompt=CAR_AGENT_PROMPT,
    context_schema=CtripContext,
    name='car_rental_booking_agent',
)
trip_agent = create_agent(
    model=qwen36p,
    tools=trip_safe_tools + trip_sensitive_tools,
    system_prompt=TRIP_AGENT_PROMPT,
    context_schema=CtripContext,
    name='trip_booking_agent',
)

# 获取到 supervisor 流转工具
assign_to_research_agent = create_handoff_tool(
    agent_name='research_agent',
    description="将联网搜索或外部信息查询任务转交给 research_agent。",
)
assgin_to_flight_agent = create_handoff_tool(
    agent_name='flight_booking_agent',
    description="将航班查询、改签、退票、航班政策相关任务转交给 flight_booking_agent。",
)
assign_to_hotel_agent = create_handoff_tool(
    agent_name="hotel_booking_agent",
    description="将酒店查询、预订、修改订单相关任务转交给 hotel_booking_agent。",
)
assign_to_car_agent = create_handoff_tool(
    agent_name="car_rental_booking_agent",
    description="将租车查询、预订、修改订单相关任务转交给 car_rental_booking_agent。",
)
assgin_to_trip_agent = create_handoff_tool(
    agent_name="trip_booking_agent",
    description="将旅行产品查询、预订、修改订单相关任务转交给 trip_booking_agent。",
)
# 创建 supervisor 智能体
SUPERVISOR_PROMPT = """
你是携程出行平台的任务调度主管。
你管理 5 个 specialist agents：
- research_agent：联网搜索、外部信息查询
- flight_booking_agent：航班查询、改签、退票、政策
- hotel_booking_agent：酒店查询、预订、订单修改
- car_rental_booking_agent：租车查询、预订、订单修改
- trip_booking_agent：旅行产品查询、预订、订单修改

规则：
1. 你只负责判断任务应该交给谁，不直接处理业务工具。
2. 一次只转交给一个 agent。
3. 只有无需工具的简单寒暄、确认类消息，才可以直接回复。
4. 涉及订单、搜索、查询、预订、修改，必须转交给对应 specialist。
""".strip()
supervisor_agent = create_agent(
    model=qwen36p,
    tools=[assign_to_car_agent, assign_to_research_agent, assign_to_hotel_agent, assgin_to_trip_agent,
           assgin_to_flight_agent],
    system_prompt=SUPERVISOR_PROMPT,
    context_schema=CtripContext,
    name='supervisor',
)
