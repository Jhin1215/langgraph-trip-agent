from datetime import date, datetime
from sqlite3 import connect
from typing import Dict, List, Annotated, Optional

import pytz
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from deep_agent.config import TRAVEL_NEW_DB
from deep_agent.tools.common import transform_loc

op_db_file = TRAVEL_NEW_DB


def query_user_flight_information(passenger_id: str) -> list[dict]:
    """
       给定 passenger_id，直接查数据库，返回该乘客的航班信息
    """
    conn = connect(op_db_file)
    cursor = conn.cursor()

    # 根据乘客 id 查询该乘客对应的机票详细，关联航班，座位分配信息
    query = """
        SELECT 
            t.ticket_no,
            t.book_ref,
            f.flight_id,
            f.flight_no,
            f.departure_airport,
            f.arrival_airport,
            f.scheduled_departure,
            f.scheduled_arrival,
            bp.seat_no,
            tf.fare_conditions
        FROM 
            tickets t
            JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
            JOIN flights f ON tf.flight_id = f.flight_id
            LEFT JOIN boarding_passes bp 
                ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
        WHERE 
            t.passenger_id = ?
    """
    cursor.execute(query, (passenger_id,))
    rows = cursor.fetchall()
    # 从 curosr 中取出每一列的列名
    column_names = [column[0] for column in cursor.description]
    # 封装字典
    rets = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()
    return rets


@tool
def fetch_user_flight_information(runtime: ToolRuntime) -> List[Dict]:
    """
    给定乘客 ID，从数据库中获取该乘客所有机票相关信息及其关联的航班和座位情况。
    Args:
        runtime: 工具运行时对象。

    Returns:
        包含每张机票详情、关联航班的信息极其座位分配的字典列表。
    """
    return query_user_flight_information(runtime.context.passenger_id)


def resolve_airport_codes(keyword: Optional[str]) -> list[str]:
    """
    将用户输入的地点解析为机场码列表。
    支持：
    - 机场三码，如 SHA / PVG
    - 城市名，如 Shanghai
    - 机场名关键字，如 Hongqiao
    """
    if not keyword:
        return []

    keyword = keyword.strip()
    if not keyword:
        return []

    conn = connect(op_db_file)
    cursor = conn.cursor()

    # 如果本身就是 3 位机场码，直接返回
    if len(keyword) == 3 and keyword.isalpha():
        cursor.execute(
            "SELECT airport_code FROM airports_data WHERE UPPER(airport_code) = ?",
            (keyword.upper(),)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [row[0] for row in rows]

    # 否则按 city / airport_name 模糊匹配
    like_kw = f"%{keyword}%"
    cursor.execute(
        """
        SELECT DISTINCT airport_code
        FROM airports_data
        WHERE city LIKE ? OR airport_name LIKE ?
        """,
        (like_kw, like_kw)
    )
    rows = cursor.fetchall()

    cursor.close()
    conn.close()
    return [row[0] for row in rows]


@tool
def search_flights(
        departure_airport: Annotated[Optional[str], "出发机场或出发城市"] = None,
        arrival_airport: Annotated[Optional[str], "到达机场或到达城市"] = None,
        start_time: Annotated[Optional[date | datetime], "出发时间范围的开始时间"] = None,
        end_time: Annotated[Optional[date | datetime], "出发时间范围的结束时间"] = None,
        limit: Annotated[int, "最多返回的航班数量"] = 20,
) -> List[Dict]:
    """
    根据条件搜索航班。
    支持输入机场三码、城市名、机场名关键字。
    """
    conn = connect(op_db_file)
    cursor = conn.cursor()

    query = "SELECT * FROM flights WHERE 1=1"
    params = []

    # 出发地解析
    dep_codes = resolve_airport_codes(transform_loc(departure_airport))
    if departure_airport and not dep_codes:
        cursor.close()
        conn.close()
        return []

    if dep_codes:
        placeholders = ",".join("?" for _ in dep_codes)
        query += f" AND departure_airport IN ({placeholders})"
        params.extend(dep_codes)

    # 到达地解析
    arr_codes = resolve_airport_codes(transform_loc(arrival_airport))
    if arrival_airport and not arr_codes:
        cursor.close()
        conn.close()
        return []

    if arr_codes:
        placeholders = ",".join("?" for _ in arr_codes)
        query += f" AND arrival_airport IN ({placeholders})"
        params.extend(arr_codes)

    # 时间过滤
    if start_time:
        query += " AND scheduled_departure >= ?"
        params.append(str(start_time))
    if end_time:
        query += " AND scheduled_departure <= ?"
        params.append(str(end_time))

    query += " ORDER BY scheduled_departure ASC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    rets = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()
    return rets


@tool
def update_ticket_to_new_flight(
        ticket_no: str,
        new_flight_id: str,
        runtime: ToolRuntime
) -> str:
    """
    将用户的机票更新为新的有效航班。步骤如下：
    1、检查乘客ID：首先从传入的配置中获取乘客ID，并验证其是否存在。
    2、查询新航班详情：根据提供的新航班ID查询航班详情，包括出发机场、到达机场和计划起飞时间。
    3、时间验证：确保新选择的航班起飞时间与当前时间相差不少于3小时。
    4、确认原机票存在性：验证提供的机票号是否存在于系统中。
    5、验证乘客身份：确保请求修改机票的乘客是该机票的实际拥有者。
    6、更新机票信息：如果所有检查都通过，则更新机票对应的新航班ID，并提交更改。
    Args:
        ticket_no: 要更新的机票编号(旧航班ID)。
        new_flight_id: 新的航班ID。
        runtime: 配置信息，包含乘客ID等必要参数。

    Returns:

    """
    # 1.
    passenger_id = runtime.context.passenger_id

    conn = connect(op_db_file)
    cursor = conn.cursor()
    # 2.
    cursor.execute(
        """
        SELECT departure_airport, arrival_airport, scheduled_departure
        FROM flights WHERE flight_id = ?
        """,
        (new_flight_id,)
    )
    new_flight = cursor.fetchone()
    if not new_flight:
        cursor.close()
        conn.close()
        raise ValueError("未找到对应的航班信息。")
    column_names = [column[0] for column in cursor.description]
    new_flight_dict = dict(zip(column_names, new_flight))

    # 3.
    timezone = pytz.timezone("Etc/GMT-3")
    cur_time = datetime.now(tz=timezone)
    # departure_time = datetime.strptime(
    #     new_flight_dict["scheduled_departure"], "%Y-%m-%d %H:%M:%S"
    # )
    departure_time = datetime.fromisoformat(new_flight_dict["scheduled_departure"])
    # 得到的单位是 s
    time_span = (departure_time - cur_time).total_seconds()
    if time_span < 3 * 3600:
        cursor.close()
        conn.close()
        raise ValueError(f"不允许重新安排到距离当前时间少于 3 小时的航班。所选航班时间为 {departure_time}。")

    # 4.
    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?",
        (ticket_no,)
    )
    cur_flight = cursor.fetchone()
    if not cur_flight:
        cursor.close()
        conn.close()
        raise ValueError("未找到给定机票号码的现有机票。")

    # 5.
    cursor.execute(
        "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id)
    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        raise ValueError(f"当前登录的乘客 ID 为 {passenger_id}，不是机票 {ticket_no} 的拥有者。")

    # 6.
    cursor.execute(
        "UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?",
        (new_flight_id, ticket_no),
    )
    conn.commit()
    cursor.close()
    conn.close()
    return "机票已成功更新为新的航班。"


@tool
def cancel_ticket(ticket_no: str, runtime: ToolRuntime) -> str:
    """
    取消用户机票
    Args:

        ticket_no: 要取消机票的 id
        runtime: 工具运行时对象。

    Returns:
        是否取消用户机票
    """
    passenger_id = runtime.context.passenger_id

    conn = connect(op_db_file)
    cursor = conn.cursor()

    # 查询票是否存在
    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?",
        (ticket_no,)
    )
    cur_flight = cursor.fetchone()
    if not cur_flight:
        cursor.close()
        conn.close()
        raise ValueError("未找到给定机票号码的现有机票。")

    # 判断用户是不是票的拥有者
    cursor.execute(
        "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id)
    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        raise ValueError(f"当前登录的乘客 ID 为 {passenger_id}，不是机票 {ticket_no} 的拥有者。")

    # 删除
    cursor.execute(
        "DELETE FROM ticket_flights WHERE ticket_no = ?",
        (ticket_no,)
    )
    conn.commit()
    cursor.close()
    conn.close()
    return "机票取消成功"
