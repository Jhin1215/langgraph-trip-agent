from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph

from deep_agent.context import CtripContext
from deep_agent.agents.supervisor import supervisor_agent
from deep_agent.agents.flight_agent import flight_agent
from deep_agent.agents.hotel_agent import hotel_agent
from deep_agent.agents.car_rent_agent import car_agent
from deep_agent.agents.trip_agent import trip_agent
from deep_agent.agents.research_agent import research_agent
from deep_agent.state import TravelState
from deep_agent.tools.flights_tools import query_user_flight_information

# 内存中存储
memory = InMemorySaver()


# 格式化用户航班信息
def format_flight_info(flight_data: list[dict]) -> str:
    flight = flight_data[0]
    return (
        f"已查询到您的航班信息：\n"
        f"- 机票号：{flight['ticket_no']}\n"
        f"- 预订编号：{flight['book_ref']}\n"
        f"- 航班号：{flight['flight_no']}（{flight['flight_id']}）\n"
        f"- 出发机场：{flight['departure_airport']}\n"
        f"- 到达机场：{flight['arrival_airport']}\n"
        f"- 计划起飞时间：{flight['scheduled_departure']}\n"
        f"- 计划到达时间：{flight['scheduled_arrival']}\n"
        f"- 座位号：{flight['seat_no']}\n"
        f"- 舱位条件：{flight['fare_conditions']}"
    )


# 普通的 GraphNode
def fetch_user_info_node(state: TravelState, runtime) -> dict:
    passenger_id = runtime.context.passenger_id
    flight_data = query_user_flight_information(passenger_id)

    # 这里不再构造 AIMessage，不污染 messages
    return {
        "user_flight_info": flight_data or []
    }


# 总图构建器
graph_builder = StateGraph(TravelState, context_schema=CtripContext)
# 添加图结点
graph_builder.add_node("fetch_user_info", fetch_user_info_node)
graph_builder.add_node(
    "supervisor",
    supervisor_agent,
    destinations=("research_agent",
                  "flight_booking_agent",
                  "hotel_booking_agent",
                  "car_rental_booking_agent",
                  "trip_booking_agent",
                  END,),
)
graph_builder.add_node("research_agent", research_agent)
graph_builder.add_node("flight_booking_agent", flight_agent)
graph_builder.add_node("car_rental_booking_agent", car_agent)
graph_builder.add_node("hotel_booking_agent", hotel_agent)
graph_builder.add_node("trip_booking_agent", trip_agent)
# 添加边
graph_builder.add_edge(START, 'fetch_user_info')
graph_builder.add_edge('fetch_user_info', "supervisor")
graph_builder.add_edge("research_agent", END)
graph_builder.add_edge("flight_booking_agent", END)
graph_builder.add_edge("car_rental_booking_agent", END)
graph_builder.add_edge("hotel_booking_agent", END)
graph_builder.add_edge("trip_booking_agent", END)
# 构建图
graph = graph_builder.compile(checkpointer=memory)
