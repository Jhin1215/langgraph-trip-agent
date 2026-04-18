from langchain.agents.middleware import dynamic_prompt, ModelRequest, HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deep_agent.agents.common import get_context_content, format_time
from deep_agent.context import CtripContext
from deep_agent.llms import qwen36p
from langchain.agents import create_agent

from deep_agent.state import TravelState
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

长期规则：
1. 查询当前用户自己的机票信息时，优先调用 fetch_user_flight_information。
2. 查询航班列表时，优先调用 search_flights。
3. 改签机票时，必须调用 update_ticket_to_new_flight。
4. 取消或退票时，必须调用 cancel_ticket。
5. 如果执行改签或取消所需参数不足，先向用户追问；如果参数已经齐全，禁止只做口头回复，必须调用对应工具。
6. 如果工具返回空结果，应明确告知用户未查询到结果，不要编造信息。
7. 回复要简洁、直接，不要输出推理过程、英文分析或内部判断过程。
8. 不要输出“已转交”“将协助处理”“请稍等”这类没有实际执行工具的描述性回复。
""".strip()


# 动态提示词
@dynamic_prompt
def flight_dynamic_prompt(request: ModelRequest) -> str:
    passenger_id = get_context_content(request).get("passenger_id", None)
    now = format_time()
    state = request.state
    task_type = state.get("handoff_task_type")
    user_request = state.get("handoff_user_request")
    slots = state.get("handoff_slots", {})
    user_flight_info = state.get("user_flight_info", [])

    return f"""
当前系统时间：{now}
当前用户 passenger_id：{passenger_id}
当前 handoff_task_type：{task_type}
当前 handoff_user_request：{user_request}
当前 handoff_slots：{slots}
当前 user_flight_info：{user_flight_info}

本轮补充约束：
1. 如果用户说“今天 / 明天 / 后天 / 未来一周 / 下周 / 近期”等相对时间，必须将其理解为明确的时间范围，并在调用 search_flights 时尽量传入 start_time 和 end_time，不能忽略时间条件。
2. 如果 fetch_user_flight_information 返回空列表，请直接回复：未查询到您的航班信息。
3. 如果 search_flights 返回空列表，请直接回复：未查询到符合条件的航班信息。
4. 输出航班信息时，必须严格使用工具返回的原始字段值，不得自行补全、改写或猜测 flight_no、机场、时间等字段。
5. 如果 seat_no 为空，不要输出 None，应表述为“暂未分配”。
""".strip()


flight_agent = create_agent(
    model=qwen36p,
    tools=flight_safe_tools + flight_sensitive_tools,
    system_prompt=FLIGHT_AGENT_PROMPT,
    middleware=[
        flight_dynamic_prompt,
        # HITL
        HumanInTheLoopMiddleware(
            interrupt_on={
                "update_ticket_to_new_flight": {
                    "description": "这是航班改签操作，需要人工审批后才能执行。",
                    "allowed_decisions": ["approve", "reject"],
                },
                "cancel_ticket": {
                    "description": "这是机票取消操作，需要人工审批后才能执行。",
                    "allowed_decisions": ["approve", "reject"],
                }
            }
        )
    ],
    # 自定义状态下，子agent（子图）也要显示传入自定的 state
    state_schema=TravelState,
    context_schema=CtripContext,
    # 总图配置再 compile 的时候配置了checkpointer, 整个图里面的结点都默认继承这个 checkpointer
    # 因此只有再单独测试某个 agent HITL 的时候才需要配置 checkpointer
    # checkpointer=InMemorySaver(),
    name='flight_booking_agent',
)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "debug-flight-1"}}
    ctx = CtripContext(passenger_id="8252 507584")

    resp1 = flight_agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "帮我取消票号 9880005432001004 的机票"}
            ]
        },
        config=config,
        context=ctx,
        version="v2",
    )
    interrupts = getattr(resp1, 'interrupts', None)
    if not interrupts:
        print("没有中断")

    for i, intr in enumerate(interrupts, 1):
        print(f"\n--- Interrupt #{i} ---")
        print(f"id: {intr.id}")

        value = intr.value or {}
        action_requests = value.get("action_requests", [])
        review_configs = value.get("review_configs", [])

        if action_requests:
            print("action_requests:")
            for j, action in enumerate(action_requests, 1):
                print(f"  [{j}]")
                print(f"    name: {action.get('name')}")
                print(f"    args: {action.get('args')}")
                print(f"    description: {action.get('description')}")
        else:
            print("action_requests: []")

        if review_configs:
            print("review_configs:")
            for j, cfg in enumerate(review_configs, 1):
                print(f"  [{j}]")
                print(f"    action_name: {cfg.get('action_name')}")
                print(f"    allowed_decisions: {cfg.get('allowed_decisions')}")
        else:
            print("review_configs: []")

    resp2 = flight_agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config,  # 必须同一个 thread_id
        context=ctx,
        version="v2",
    )
    print(resp2["messages"])
