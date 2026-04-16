from sqlite3 import connect
from typing import Annotated, Optional

from langchain_core.tools import tool

from deep_agent.config import TRAVEL_NEW_DB
from deep_agent.tools.common import transform_loc

db = TRAVEL_NEW_DB


@tool
def search_trip_recommendations(
        location: Annotated[Optional[str], "旅行所在的城市"] = None,
        name: Annotated[Optional[str], "旅行推荐的名称"] = None,
        keywords: Annotated[Optional[str], "旅行推荐关键词，多个关键字用逗号分割"] = None
) -> list[dict]:
    """
    根据位置、名称和关键字搜索旅行推荐
    Returns:
        匹配搜索条件对应的字典列表
    """
    conn = connect(db)
    cursor = conn.cursor()

    # 城市名标准化
    eng_loc = transform_loc(location)

    # 构造查询 SQL
    query = "SELECT * FROM trip_recommendations WHERE 1 = 1"
    # 查询参数
    params = []

    if eng_loc:
        query += " AND location LIKE ?"
        # 模糊查询需要使用 %xx% 拼接
        params.append(f"%{eng_loc}%")

    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")

    if keywords:
        # 先按照英文逗号切割
        # 然后去除一个单词左右的空格
        # 在判断这个单词是不是空串。只要非空串，空串删除
        keywords_list = [keyword.strip() for keyword in keywords.split(',') if keyword.strip()]
        if keywords_list:
            # keyword LIKE ? OR keywrod LIKE ? ...
            keyword_conditions = " OR ".join(["keywords LIKE ?" for _ in keywords_list])
            # 拼接到 query 上
            query += f" AND ({keyword_conditions})"
            # 参数添加到列表中
            params.extend([f"%{keyword}%" for keyword in keywords_list])

    # 执行 SQL 语句
    cursor.execute(query, params)
    rows = cursor.fetchall()
    column_name = [column[0] for column in cursor.description]
    cursor.close()
    conn.close()
    rets = [dict(zip(column_name, row)) for row in rows]
    return rets


@tool
def book_excursion(recommendation_id: int) -> str:
    """
    通过推荐 id 订阅一次旅行推荐
    Args:
        recommendation_id: 旅行推荐 id
    Returns:
        订阅旅行推荐的结果
    """
    conn = connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET booked = 1 WHERE id = ?",
        (recommendation_id,)
    )
    conn.commit()
    affected_rows = cursor.rowcount
    cursor.close()
    conn.close()
    if affected_rows > 0:
        return f"旅行推荐 {recommendation_id} 成功订阅。"
    return f"未找到 ID 为 {recommendation_id} 的旅行推荐。"


@tool
def update_excursion(recommendation_id: int, details: str) -> str:
    """
    根据 ID 更新旅行推荐详情
    Args:
        recommendation_id: 旅行推荐 id
        details: 旅行推荐详情
    Returns:
        更新旅行推荐详情的结果
    """
    conn = connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET details = ? WHERE id = ?",
        (details, recommendation_id)
    )
    conn.commit()
    affected_rows = cursor.rowcount
    cursor.close()
    conn.close()
    if affected_rows > 0:
        return f"旅行推荐 {recommendation_id} 详情更新成功。"
    return f"未找到 ID 为 {recommendation_id} 的旅行推荐。"


@tool
def cancel_excursion(recommendation_id: int) -> str:
    """
    根据推荐 id 取消旅行推荐
    Args:
        recommendation_id: 推荐 ID

    Returns:
        取消旅行推荐的结果
    """
    conn = connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET booked = 0 WHERE id = ?",
        (recommendation_id,)
    )
    conn.commit()
    affected_rows = cursor.rowcount
    cursor.close()
    conn.close()
    if affected_rows > 0:
        return f"旅行推荐 {recommendation_id} 取消成功。"
    return f"未找到 ID 为 {recommendation_id} 的旅行推荐。"
