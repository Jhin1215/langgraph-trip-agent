from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from deep_agent.agents.common import get_context_content, format_time
from deep_agent.context import CtripContext
from deep_agent.llms import qwen36p


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


# supervisor 工具(流转工具)
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
# supervisor 上下文工程
# 1.静态提示词
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


# 2.动态提示词
@dynamic_prompt
def supervisor_dynamic_prompt(request: ModelRequest) -> str:
    """
    动态提示词负责给出每次调用都会变的上下文，比如：
        当前时间
        当前用户 passenger_id
        当前这轮补充约束
        对相对时间的解释要求
    """
    passenger_id = get_context_content(request).get('passenger_id', None)
    now = format_time()
    return f"""
    当前系统时间：{now}
    当前用户 passenger_id：{passenger_id}

    补充规则：
    1. 你只能做任务分发，不直接执行业务工具。
    2. 如果用户问题明显属于航班、酒店、租车、旅行推荐、联网搜索中的某一类，必须调用对应 handoff tool。
    3. 如果是简单寒暄、确认类消息，才允许直接回复。
    4. 不要自己编造航班、酒店、租车或旅行产品信息。
    """.strip()


supervisor_agent = create_agent(
    model=qwen36p,
    tools=[assign_to_car_agent, assign_to_research_agent, assign_to_hotel_agent, assgin_to_trip_agent,
           assgin_to_flight_agent],
    system_prompt=SUPERVISOR_PROMPT,
    middleware=[supervisor_dynamic_prompt],
    context_schema=CtripContext,
    name='supervisor',
)
