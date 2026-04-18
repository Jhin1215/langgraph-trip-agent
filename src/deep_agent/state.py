from typing import Annotated, NotRequired, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


# langgraph v1 推荐在进行 handoff 转移 agent 控制权时，
# handoff 需要把用户意图通过自定义 state（自己做新的 context engineering） 传给新 agent
# 否则就会面临 bloated / malformed context
class TravelState(TypedDict):
    # 1.对话历史：仍然保留
    # add_messages 的作用：后续节点返回 {"messages": [...]} 时，会追加/按 id 更新，
    # 而不是把旧消息整个覆盖掉
    messages: Annotated[list[AnyMessage], add_messages]

    # 2.当前激活的 agent
    activate_agent: NotRequired[str]

    # 3.handoff 处理的类型
    handoff_task_type: NotRequired[str]

    # 4.用户意图摘要
    handoff_user_request: NotRequired[str]

    # 5.结构化槽位，比如 {"ticket_no": "..."} / {"city": "..."}
    handoff_slots: NotRequired[dict[str, Any]]

    # 6) 当前用户航班信息：查库结果放这里，
    # 不再默认伪装成 AIMessage
    user_flight_info: NotRequired[list[dict]]
