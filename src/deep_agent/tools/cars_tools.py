from datetime import datetime, date
from sqlite3 import connect
from typing import Annotated, Dict, List, Optional, Union

from langchain_core.tools import tool

from deep_agent.config import TRAVEL_NEW_DB
from deep_agent.tools.common import transform_loc

db = TRAVEL_NEW_DB


@tool
def search_car_rentals(
        location: Annotated[Optional[str], "汽车租赁所在城市名称"] = None,
        name: Annotated[Optional[str], "汽车租赁公司名称"] = None,
) -> list[dict]:
    """
    根据位置和名称搜索汽车租赁信息
    Returns:
        匹配条件的汽车租赁信息列表
    """
    conn = connect(db)
    cursor = conn.cursor()
    # 城市信息转化
    eng_loc = transform_loc(location)

    # 创建真查询语句方便后面 SQL 语句进行拼接
    query = "SELECT * FROM car_rentals WHERE 1 = 1"
    params = []
    if eng_loc:
        query += " AND location LIKE ?"
        params.append(f"%{eng_loc}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    # 执行 SQL 语句
    cursor.execute(query, params)
    rows = cursor.fetchall()
    column_name = [column[0] for column in cursor.description]
    rets = [dict(zip(column_name, row)) for row in rows]
    cursor.close()
    conn.close()
    return rets


@tool
def book_car_rental(rental_id: int) -> str:
    """
    通过租赁 ID 预定汽车
    Args:
        rental_id: 租赁 ID

    Returns:
        预定的结果信息
    """
    conn = connect(db)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE car_rentals SET booked = 1 WHERE id = ?",
        (rental_id,)
    )
    conn.commit()
    row_count = cursor.rowcount
    cursor.close()
    conn.close()

    if row_count > 0:
        return f"汽车租赁 {rental_id} 成功预订。"
    return f"未找到 ID 为 {rental_id} 的汽车租赁服务。"


@tool
def update_car_rental(
        rental_id: Annotated[Optional[int], "汽车租赁 ID"] = None,
        start_time: Annotated[Optional[Union[datetime, date]], "开始时间"] = None,
        end_time: Annotated[Optional[Union[datetime, date]], "结束时间"] = None,
) -> str:
    """
    更新汽车租赁信息
    Returns:
        更新结果信息
    """
    conn = connect(db)
    cursor = conn.cursor()

    # 每一个更新都去累加一下受影响的行数
    affected_rows = 0
    if start_time:
        cursor.execute(
            "UPDATE car_rentals SET start_date = ? WHERE id = ?",
            (start_time, rental_id)
        )
        affected_rows += cursor.rowcount
    if end_time:
        cursor.execute(
            "UPDATE car_rentals SET end_date = ? WHERE id = ?",
            (end_time, rental_id)
        )
        affected_rows += cursor.rowcount

    conn.commit()
    cursor.close()
    conn.close()
    if affected_rows > 0:
        return f"汽车租赁 {rental_id} 成功更新。"
    return f"未找到 ID 为 {rental_id} 的汽车租赁服务。"


@tool
def cancel_car_rental(rental_id: int) -> str:
    """
    通过租赁 ID 取消预定汽车
    Args:
        rental_id: 租赁 ID

    Returns:
        取消结果
    """
    conn = connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE car_rentals SET booked = 0 WHERE id = ?",
        (rental_id,)
    )
    conn.commit()
    row_count = cursor.rowcount
    cursor.close()
    conn.close()
    if row_count > 0:
        return f"汽车租赁 {rental_id} 已成功取消。"
    return f"未找到 ID 为 {rental_id} 的汽车租赁服务。"
