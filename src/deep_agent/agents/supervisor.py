from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field

from deep_agent.agents.common import get_context_content, format_time
from deep_agent.context import CtripContext
from deep_agent.llms import qwen36p


# 显示 context engineering
# 1.创建 handoff schema 用于提供清晰的参数给 llm
class HandoffSchema(BaseModel):
    task_type: str = Field(description="任务类型，例如 cancel_ticket、search_flights、book_hotel、web_search")
    user_request: str = Field(description="用户原始请求的简短复述")
    slots: dict = Field(description="任务参数, 如 ticket_no、departure_airport、arrival_airport、city、checkin、checkout")


# 创建多智能体编排方式
# langgraph 里面提供了两种多智能体编排方式
# 1.supervisor 创建一个中心 supervisor 负责 orchestrate 多个 specialized agents
# 2.warm：没有中心supervisor，多个 agent 根据专长动态 hand off 控制权。
def create_handoff_tool(*, agent_name: str, description: str):
    """
    创建一个通用 handoff tool：
        1. 负责跳转到指定 specialist
        2. 负责把任务态写入父图 state
        3. 仍然保留 AIMessage + ToolMessage，保证 tool-call 闭环
    """
    tool_name = f"transfer_to_{agent_name}"

    @tool(tool_name, args_schema=HandoffSchema, description=description)
    def handoff_control(
            task_type: str,
            user_request: str,
            slots: dict[str, Any],
            runtime: ToolRuntime
    ) -> Command:
        # 创建一个只包含 aimessages 的生成器对象
        it = (msg for msg in reversed(runtime.state['messages']) if isinstance(msg, AIMessage))
        last_ai_msg = next(it, None)
        tool_message = {
            "role": "tool",
            "content": (
                f"Transferred to {agent_name}. "
                f"task_type={task_type}; user_request={user_request}; slots={slots}"
            ),
            "name": tool_name,
            "tool_call_id": runtime.tool_call_id,
        }
        # 没找到 AIMessage 时，只传 ToolMessage，避免把 None 塞进 messages
        if last_ai_msg is None:
            update_messages = [tool_message]
        else:
            update_messages = [last_ai_msg, tool_message]

        print(update_messages)
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            # context engineering 第二个点：
            # 显示将用户任务上下文写入 state(自定义的state)
            update={
                "messages": update_messages,
                "active_agent": agent_name,
                "handoff_task_type": task_type,
                "handoff_user_request": user_request,
                "handoff_slots": slots,
            },
        )

    return handoff_control


# supervisor 工具(流转工具)
assign_to_research_agent = create_handoff_tool(
    agent_name='research_agent',
    # supervisor handoff 到专用智能体工具的描述信息
    # 需要加入关于 handoff_control 这个函数参数的显示描述，易于llm传参
    description="将联网搜索或外部信息查询任务转交给 research_agent。"
                "必须填写 task_type、user_request、slots。",
)
assgin_to_flight_agent = create_handoff_tool(
    agent_name='flight_booking_agent',
    description="""
    将航班查询、改签、退票、航班政策相关任务转交给 flight_booking_agent。
    必须填写 task_type、user_request、slots。例如：
        - 取消机票：task_type=cancel_ticket，slots={"ticket_no": "..."}
        - 改签机票：task_type=update_ticket，slots 中包含 ticket_no 和新航班相关参数
                """,
)
assign_to_hotel_agent = create_handoff_tool(
    agent_name="hotel_booking_agent",
    description="将酒店查询、预订、修改订单相关任务转交给 hotel_booking_agent。"
                "必须填写 task_type、user_request、slots。",

)
assign_to_car_agent = create_handoff_tool(
    agent_name="car_rental_booking_agent",
    description="将租车查询、预订、修改订单相关任务转交给 car_rental_booking_agent。",
)
assgin_to_trip_agent = create_handoff_tool(
    agent_name="trip_booking_agent",
    description="将旅行产品查询、预订、修改订单相关任务转交给 trip_booking_agent。"
                "必须填写 task_type、user_request、slots。",
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
5. 调用 handoff tool 时，必须显式填写：
   - task_type：任务类型
   - user_request：用户请求的简单描述
   - slots：包含该任务所需的动态结构化参数。
6. 用户传入的静态上下文（context）信息，如 passenger_id、session info、权限、数据库连接、环境配置；
这些值应通过 context 传入它已经在 runtime.context 中。
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
    3. handoff 时必须把用户任务整理成：
       - task_type
       - user_request
       - slots
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
