import logging
from typing import Any

from fastapi import APIRouter
from starlette.requests import Request
from langgraph.types import Command

from deep_agent.graph import graph
from apps.api.graph_api.graph_schemas import GraphRequestSchema, GraphResponseSchema

router = APIRouter()
log = logging.getLogger("graph")


def _extract_last_ai_content(update: dict[str, Any], seen_message_ids: set[str]) -> str | None:
    messages = update.get("messages", [])
    if not messages:
        return None

    last_msg = messages[-1]
    msg_id = getattr(last_msg, "id", None)

    # 按 message id 去重，避免 subgraph / updates 重复打印同一条消息
    if msg_id is not None:
        if msg_id in seen_message_ids:
            return None
        seen_message_ids.add(msg_id)

    if last_msg.__class__.__name__ != "AIMessage":
        return None

    content = getattr(last_msg, "content", None)
    if not content:
        return None

    if isinstance(content, str):
        return content

    return str(content)


def _build_config_and_context(obj_in: GraphRequestSchema) -> tuple[dict, dict | None]:
    """
    按你当前 schema 结构，把 API 请求转成 graph 需要的:
    - config={"configurable": {"thread_id": ...}}
    - context={"passenger_id": ...}
    """
    config = {
        "configurable": {
            "thread_id": obj_in.thread_id,
        }
    }
    context = (
        {"passenger_id": obj_in.passenger_id}
        if obj_in.passenger_id
        else None
    )
    return config, context


def _normalize_decision(user_input: str) -> str | None:
    text = user_input.strip().lower()
    mapping = {
        "y": "approve",
        "yes": "approve",
        "approve": "approve",
        "n": "reject",
        "no": "reject",
        "reject": "reject",
    }
    return mapping.get(text)


@router.post(
    "/graph/",
    description="调用工作流",
    summary="调用工作流",
    response_model=GraphResponseSchema,
)
def execute_graph(request: Request, obj_in: GraphRequestSchema):
    username = getattr(request.state, "username", None)
    if username:
        log.info("登录用户名: %s", username)

    question = obj_in.user_input.strip()
    config, context = _build_config_and_context(obj_in)

    result = ""
    interrupt_map: dict[str, Any] = {}
    seen_message_ids: set[str] = set()

    # 先看当前 thread 是否正处于 HITL 中断态
    current_state = graph.get_state(config)
    has_pending_interrupt = bool(current_state.next)

    if has_pending_interrupt:
        decision = _normalize_decision(question)
        if decision is None:
            return {
                "assistant": "当前流程正等待审批，请输入 approve 或 reject（也兼容 y / n）。"
            }

        graph_input = Command(
            resume={
                "decisions": [
                    {"type": decision}
                ]
            }
        )
    else:
        graph_input = {
            "messages": [
                {"role": "user", "content": question}
            ]
        }

    stream_kwargs = {
        "config": config,
        "stream_mode": ["messages", "updates"],
        "version": "v2",
        "subgraphs": True,
    }
    if context is not None:
        stream_kwargs["context"] = context

    for chunk in graph.stream(graph_input, **stream_kwargs):
        if chunk["type"] != "updates":
            continue

        for source, update in chunk["data"].items():
            if source == "__interrupt__":
                for intr in update:
                    interrupt_map[intr.id] = intr
                continue

            if not isinstance(update, dict):
                continue

            ai_content = _extract_last_ai_content(update, seen_message_ids)
            if ai_content:
                result = ai_content

    interrupts = list(interrupt_map.values())
    if interrupts:
        interrupt_text = "\n\n".join(str(intr.value) for intr in interrupts)
        prefix = f"{result}\n\n" if result else ""
        result = (
            f"{prefix}{interrupt_text}\n\n"
            "检测到 HITL 审批，请继续调用当前接口，并在 user_input 中传入 approve 或 reject。"
        )

    if not result:
        result = "工作流执行完成，但当前没有可展示的 AI 文本输出。"

    return {"assistant": result}
