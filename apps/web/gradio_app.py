from __future__ import annotations

import io
import uuid
from typing import Any

import gradio as gr
from PIL import Image

from langgraph.types import Command

# 若本地模块无法导入，先注释这两行测试界面
from deep_agent.context import CtripContext
from deep_agent.graph import graph


# -----------------------------
# 1) 图渲染
# -----------------------------
def render_graph_image() -> Image.Image:
    png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
    return Image.open(io.BytesIO(png_bytes))


# -----------------------------
# 2) 构造 config / context
# -----------------------------
def build_config(thread_id: str) -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def build_context(passenger_id: str) -> CtripContext:
    return CtripContext(passenger_id=passenger_id)


# -----------------------------
# 3) 新会话
# -----------------------------
def new_thread(passenger_id: str):
    thread_id = str(uuid.uuid4())
    return (
        thread_id,
        [],
        None,
        f"已创建新会话，thread_id={thread_id}",
    )


# -----------------------------
# 4) 结果提取
# -----------------------------
def extract_last_ai_text_from_state(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    if not messages:
        return ""

    for msg in reversed(messages):
        content = getattr(msg, "content", None)

        if isinstance(content, str) and content.strip():
            return content.strip()

        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            text = "".join(text_parts).strip()
            if text:
                return text

    return ""


def format_interrupts(interrupts: list[Any]) -> str:
    if not interrupts:
        return ""

    chunks = ["检测到需要人工确认的中断："]
    for idx, intr in enumerate(interrupts, start=1):
        value = getattr(intr, "value", None)
        chunks.append(f"\n[{idx}] {value}")
    chunks.append("\n请输入：approve / reject / edit:<你的修改意见>")
    return "\n".join(chunks)


def build_status_text(config: dict[str, Any], fallback: str) -> str:
    try:
        current_state = graph.get_state(config)
        next_nodes = list(current_state.next) if current_state and current_state.next else []
        if next_nodes:
            return f"{fallback} | next={next_nodes}"
        return fallback
    except Exception:
        return fallback


# -----------------------------
# 5) 前端状态辅助函数
# -----------------------------
def set_running_status() -> str:
    return "处理中..."


def set_ready_status() -> str:
    return "就绪"


# -----------------------------
# 6) 单轮对话执行
# -----------------------------
def chat_once(
        user_text: str,
        chat_history: list[dict],
        thread_id: str,
        passenger_id: str,
        pending_interrupts: list[Any] | None,
):
    user_text = (user_text or "").strip()
    if not user_text:
        return "", chat_history, pending_interrupts, "输入为空"

    config = build_config(thread_id=thread_id)
    context = build_context(passenger_id=passenger_id)
    chat_history = chat_history + [{"role": "user", "content": user_text}]

    try:
        if pending_interrupts:
            lowered = user_text.lower()

            if lowered == "approve":
                resume_value = True
            elif lowered == "reject":
                resume_value = False
            elif lowered.startswith("edit:"):
                resume_value = user_text[len("edit:"):].strip()
            else:
                assistant_text = "当前正在等待人工确认。请输入 approve / reject / edit:<修改意见>"
                chat_history.append({"role": "assistant", "content": assistant_text})
                status_text = build_status_text(config, "等待人工确认")
                return "", chat_history, pending_interrupts, status_text

            result = graph.invoke(
                Command(resume=resume_value),
                config=config,
                context=context,
            )

            interrupts = result.get("__interrupt__", None)
            if interrupts:
                assistant_text = format_interrupts(interrupts)
                chat_history.append({"role": "assistant", "content": assistant_text})
                status_text = build_status_text(config, "等待人工确认")
                return "", chat_history, interrupts, status_text

            assistant_text = (
                    extract_last_ai_text_from_state(result)
                    or "已恢复执行，但没有提取到明确文本输出。"
            )
            chat_history.append({"role": "assistant", "content": assistant_text})
            status_text = build_status_text(config, "执行完成")
            return "", chat_history, None, status_text

        result = graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_text,
                    }
                ]
            },
            config=config,
            context=context,
        )

        interrupts = result.get("__interrupt__", None)
        if interrupts:
            assistant_text = format_interrupts(interrupts)
            chat_history.append({"role": "assistant", "content": assistant_text})
            status_text = build_status_text(config, "等待人工确认")
            return "", chat_history, interrupts, status_text

        current_state = graph.get_state(config)
        if current_state and current_state.next:
            assistant_text = "检测到图执行已暂停，等待进一步确认或恢复。"
            chat_history.append({"role": "assistant", "content": assistant_text})
            status_text = build_status_text(config, "等待人工确认")
            return "", chat_history, pending_interrupts, status_text

        assistant_text = extract_last_ai_text_from_state(result) or "没有提取到明确文本输出。"
        chat_history.append({"role": "assistant", "content": assistant_text})
        status_text = build_status_text(config, "执行完成")
        return "", chat_history, None, status_text

    except Exception as e:
        chat_history.append(
            {"role": "assistant", "content": f"执行异常：{type(e).__name__}: {e}"}
        )
        status_text = build_status_text(config, "执行异常")
        return "", chat_history, pending_interrupts, status_text


# -----------------------------
# 7) 刷新结构图
# -----------------------------
def refresh_graph():
    return render_graph_image(), "图已刷新"


# -----------------------------
# 8) Gradio UI 6.x 极简兼容版（彻底移除所有非核心参数）
# -----------------------------
theme = gr.themes.Default(
    primary_hue="sky",
    secondary_hue="slate",
    radius_size="md",
    spacing_size="md",
)

# ... (前面的业务逻辑代码完全不变，只替换最后 UI 部分) ...

# -----------------------------
# 8) Gradio UI 极简版（彻底移除所有可能报错的参数）
# -----------------------------
theme = gr.themes.Default(
    primary_hue="sky",
    secondary_hue="slate",
    radius_size="md",
    spacing_size="md",
)

with gr.Blocks(
    title="LangGraph v1 多智能体测试台",
    fill_width=False,
) as demo:
    gr.Markdown("## LangGraph v1 + Gradio 联调页面")

    thread_state = gr.State(str(uuid.uuid4()))
    interrupt_state = gr.State(None)

    # 【核心修复】Chatbot 只留 label 和 height，其他所有参数全部移除
    # Gradio 6.x 默认原生支持 {"role":"user/assistant", "content":"..."} 格式
    chatbot = gr.Chatbot(
        label="对话测试",
        height=600,
    )

    # 核心输入行 - 输入框+passenger_id+按钮 同一行并排
    with gr.Row(equal_height=True):
        input_box = gr.Textbox(
            label="输入",
            placeholder="请输入你的问题；若命中 HITL / interrupt，请输入 approve / reject / edit:xxx",
            lines=1,
            max_lines=2,
            scale=5,
            min_width=400,
        )
        passenger_box = gr.Textbox(
            label="passenger_id",
            value="3442 587242",
            scale=3,
            min_width=200,
        )
        send_btn = gr.Button(
            "发送",
            variant="primary",
            size="md",
            scale=1,
            min_width=100,
        )
        new_thread_btn = gr.Button(
            "新会话",
            variant="secondary",
            size="md",
            scale=1,
            min_width=100,
        )

    status_box = gr.Textbox(
        label="状态",
        value="就绪",
        interactive=False,
    )

    with gr.Accordion("Graph 结构图", open=False):
        graph_image = gr.Image(
            label="Graph 结构图",
            value=render_graph_image,
            type="pil",
            height=850,
        )
        refresh_btn = gr.Button("刷新图", size="sm", variant="secondary")

    # 事件绑定（完全不变）
    send_btn.click(
        fn=set_running_status,
        inputs=None,
        outputs=status_box,
        queue=False,
    ).then(
        fn=chat_once,
        inputs=[input_box, chatbot, thread_state, passenger_box, interrupt_state],
        outputs=[input_box, chatbot, interrupt_state, status_box],
    )

    input_box.submit(
        fn=set_running_status,
        inputs=None,
        outputs=status_box,
        queue=False,
    ).then(
        fn=chat_once,
        inputs=[input_box, chatbot, thread_state, passenger_box, interrupt_state],
        outputs=[input_box, chatbot, interrupt_state, status_box],
    )

    new_thread_btn.click(
        fn=new_thread,
        inputs=[passenger_box],
        outputs=[thread_state, chatbot, interrupt_state, status_box],
    )

    refresh_btn.click(
        fn=refresh_graph,
        inputs=None,
        outputs=[graph_image, status_box],
    )

if __name__ == "__main__":
    demo.launch(
        theme=theme,
        debug=True
    )