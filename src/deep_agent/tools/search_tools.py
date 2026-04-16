from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from deep_agent.context import SearchContext
from deep_agent.llms import zhipuai_client, qwen36p


@tool
def search_tool(query: str, runtime: ToolRuntime) -> str:
    """
    搜索互联网上的内容
    Args:
        query: 要搜索的语句
        runtime: 智能体运行上下文信息

    Returns:
        搜索到的内容
    """
    # 启动搜索
    resp = zhipuai_client.web_search.web_search(
        search_engine='search_std',
        search_query=query,
    )
    search_rets = getattr(resp, 'search_result', None)
    if search_rets is None:
        return '没有搜索到结果'
    blocks = []
    # search in web 可能返回多个相关结果，这里只取前 3 个，后面的 1 是这个循环从 1 开始计数
    for i, item in enumerate(search_rets[:3], 1):
        title = getattr(item, 'title', None)
        link = getattr(item, 'link ', None)
        content = getattr(item, 'content', None)
        blocks.append(
            f"search in web 结果的第 {i} 条内容：\n标题: {title}\n摘要: {content}\n链接: {link}"
        )
    content = '\n\n'.join(blocks)
    return content


if __name__ == '__main__':
    config = {"configurable": {"thread_id": "s11"}}
    agent = create_agent(
        model=qwen36p,
        tools=[search_tool],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "search_tool": {
                        "description": "该工具会访问互联网公开搜索结果，请人工确认是否允许执行。",
                        "allowed_decisions": ["approve", "reject"],
                    }
                }
            )
        ],
        context_schema=SearchContext,
        checkpointer=InMemorySaver(),
    )
    resp = agent.invoke(
        {"messages": [
            {"role": "user", "content": "今天青岛的天气怎么样？"}
        ]},
        context={"user_id": "jhin111", "role": "user"},
        config=config,
        # 返回的数据类型可以直接使用 . 或者 getattr 获取值
        version='v2',
    )
    interrupts = getattr(resp, 'interrupts', None)
    print(interrupts)

    if interrupts:
        intr = interrupts[0]

        payload = intr.value
        action = payload["action_requests"][0]
        review = payload["review_configs"][0]

        print("中断信息：", action["description"])
        print("工具名：", action["name"])
        print("工具参数：", action["args"])
        print("可选审批：", review["allowed_decisions"])
    # 获取中断的选项
    decision = input("\n请输入审批结果 approve / reject: ").strip().lower()

    # 中断之后通过 resume 配合 checkpointer 继续调用
    resumed_ret = agent.invoke(
        Command(
            resume={"decisions": [{
                "type": decision,
            }]}
        ),
        config=config,
        context={"user_id": "jhin111", "role": "user"},
        version='v2',
    )
    print(resumed_ret)
