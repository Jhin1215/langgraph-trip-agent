import uuid

from deep_agent.graph import graph

if __name__ == '__main__':
    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    context = {
        "passenger_id": "8252 507584",
    }

    print(f"当前 thread_id: {thread_id}")
    print("输入 quit / exit 结束。\n")

    while True:
        user_input = input("用户: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue

        print("\n--- graph stream start ---")
        try:
            for chunk in graph.stream(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": user_input,
                            }
                        ]
                    },
                    config=config,
                    context=context,
                    stream_mode="values",
            ):
                last_msg = chunk["messages"][-1]
                print(f"\n[{last_msg.__class__.__name__}]")
                if hasattr(last_msg, "pretty_print"):
                    last_msg.pretty_print()
                else:
                    print(last_msg)
        except Exception as e:
            print(f"\n运行报错: {type(e).__name__}: {e}")

        print("\n--- graph stream end ---\n")
