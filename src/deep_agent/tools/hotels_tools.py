from datetime import date, datetime
from sqlite3 import connect
from typing import Annotated, Optional, Union

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from deep_agent.config import TRAVEL_NEW_DB
from deep_agent.tools.common import transform_loc

db = TRAVEL_NEW_DB


class SearchHotelsInputParams(BaseModel):
    location: Optional[str] = Field(default=None, description='酒店位置')
    name: Optional[str] = Field(default=None, description='酒店名称')
    # checkin_date: Optional[Union[datetime, date]] = Field(default=None, description='入住时间')
    # checkout_date: Optional[Union[datetime, date]] = Field(default=None, description='退房时间')


@tool(args_schema=SearchHotelsInputParams)
def search_hotels(
        location: Optional[str] = None,
        name: Optional[str] = None,
) -> list[dict]:
    """
    根据酒店的位置、名称搜索酒店
    Returns:
        匹配搜索条件的酒店组成的列表
    """
    conn = connect(db)
    cursor = conn.cursor()

    location = transform_loc(location)
    query = "SELECT * FROM hotels WHERE 1=1"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")

    print("查询酒店的SQL：", query, "参数：", params)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    rets = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()
    return rets


@tool
def book_hotel(hotel_id: int) -> str:
    """
    根据酒店 ID 预订酒店
    Returns:
        预订成功信息
    """
    conn = connect(db)
    cursor = conn.cursor()
    # 根据更新消息来看是否预定成功
    cursor.execute(
        "UPDATE hotels SET booked = 1 WHERE id = ?",
        (hotel_id,)
    )
    conn.commit()
    row_count = cursor.rowcount
    cursor.close()
    conn.close()
    return f"Hotel {hotel_id} 成功预订。" if row_count > 0 else f"未找到 ID 为 {hotel_id} 的酒店。"


@tool
def update_hotel(
        hotel_id: Annotated[int, "酒店 ID"],
        checkin_date: Annotated[Optional[Union[datetime, date]], "入住时间"] = None,
        checkout_date: Annotated[Optional[Union[datetime, date]], "退房时间"] = None,
) -> str:
    """
    根据酒店 ID 更新酒店预订和退房日期
    Returns:
        酒店预订是否成功更新的消息。
    """
    conn = connect(db)
    cursor = conn.cursor()
    affected_count = 0
    if checkin_date:
        cursor.execute(
            "UPDATE hotels SET checkin_date = ? WHERE id = ?",
            (checkin_date, hotel_id)
        )
        affected_count += cursor.rowcount
    if checkout_date:
        cursor.execute(
            "UPDATE hotels SET checkout_date = ? WHERE id = ?",
            (checkout_date, hotel_id)
        )
        affected_count += cursor.rowcount
    conn.commit()
    cursor.close()
    conn.close()
    return f"Hotel {hotel_id} 成功更新。" if affected_count > 0 else f"未找到 ID 为 {hotel_id} 的酒店。"


@tool
def cancel_hotel(hotel_id: int) -> str:
    """
    取消酒店预订
    Args:
        hotel_id (int): 要取消的酒店预订的ID。
    Returns:
        取消酒店预订是否成功更新的消息。
    """
    conn = connect(db)
    cursor = conn.cursor()

    # booked 字段设置为 0 表示取消预定
    cursor.execute(
        "UPDATE hotels SET booked = 0 WHERE id =?",
        (hotel_id,)
    )
    conn.commit()
    row_count = cursor.rowcount
    cursor.close()
    conn.close()
    return f"Hotel {hotel_id} 已成功取消。" if row_count > 0 else f"未找到 ID 为 {hotel_id} 的酒店。"
