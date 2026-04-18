import uuid
from langgraph.types import Command
from deep_agent.graph import graph


def _print_message_once(update: dict, seen_message_ids: set[str]):
    messages = update.get("messages", [])
    if not messages:
        return

    last_msg = messages[-1]
    msg_id = getattr(last_msg, "id", None)

    # 有 id 就按 id 去重；没 id 就直接打印
    if msg_id is not None:
        if msg_id in seen_message_ids:
            return
        seen_message_ids.add(msg_id)

    print(f"\n[{last_msg.__class__.__name__}]")
    if hasattr(last_msg, "pretty_print"):
        last_msg.pretty_print()
    else:
        print(last_msg)


if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    context = {"passenger_id": "8252 507584"}

    print(f"当前 thread_id: {thread_id}")
    print("输入 quit / exit 结束。\n")

    while True:
        user_input = input("用户: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue

        print("\n--- graph stream start ---")

        # 本轮去重集合
        interrupt_map = {}
        seen_message_ids = set()

        try:
            for chunk in graph.stream(
                    {
                        "messages": [
                            {"role": "user", "content": user_input}
                        ]
                    },
                    config=config,
                    context=context,
                    stream_mode=["messages", "updates"],
                    version="v2",
                    subgraphs=True,
            ):
                if chunk["type"] != "updates":
                    continue

                for source, update in chunk["data"].items():
                    if source == "__interrupt__":
                        for intr in update:
                            interrupt_map[intr.id] = intr
                    elif isinstance(update, dict):
                        _print_message_once(update, seen_message_ids)

            interrupts = list(interrupt_map.values())

            if not interrupts:
                print("\n--- graph stream end ---\n")
                continue

            print("\n检测到 HITL 中断：")
            for i, intr in enumerate(interrupts, 1):
                print(f"\n--- Interrupt #{i} ---")
                print(f"id: {intr.id}")
                print(intr.value)

            decision = input("\n请输入审批结果（approve / reject）: ").strip().lower()
            if decision not in {"approve", "reject"}:
                print("非法输入，本轮取消。\n--- graph stream end ---\n")
                continue

            print("\n--- resume stream start ---")

            # resume 阶段重新做一次消息去重
            seen_message_ids = set()

            resume_payload = {
                "decisions": [{"type": decision}]
            }

            for chunk in graph.stream(
                    Command(resume=resume_payload),
                    config=config,
                    context=context,
                    stream_mode=["messages", "updates"],
                    version="v2",
                    subgraphs=True,
            ):
                if chunk["type"] != "updates":
                    continue

                for source, update in chunk["data"].items():
                    if isinstance(update, dict):
                        _print_message_once(update, seen_message_ids)

            print("\n--- resume stream end ---")

        except Exception as e:
            print(f"\n运行报错: {type(e).__name__}: {e}")

        print("\n--- graph stream end ---\n")
