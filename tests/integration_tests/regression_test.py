"""
当前版本聚焦已经基本跑通的 flight 主链路，覆盖：
1. 查询当前用户机票信息
2. 同一 thread_id 下的多轮追问
3. 按出发机场查询航班
4. 按出发机场 + 到达机场联合查询航班
5. 改签成功
6. 改签失败（把 flight_no 当成 flight_id）

运行方式：
    pytest -q tests/integration_tests/test_multi_agent_regression.py

"""

from __future__ import annotations

import shutil
import sqlite3
import uuid
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from deep_agent.config import TRAVEL_NEW_DB
from deep_agent.graph import graph
from deep_agent.tools.flights_tools import query_user_flight_information


# =========================
# 可按需修改的测试配置
# =========================
PASSENGER_ID = "8252 507584"


# =========================
# 基础工具函数
# =========================
def _db_path() -> Path:
    return Path(TRAVEL_NEW_DB)


def _unique_thread_id() -> str:
    return str(uuid.uuid4())


def _extract_last_ai_text(messages) -> str:
    """从返回消息中提取最后一条 AIMessage 文本。"""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                return content
            return str(content)
    raise AssertionError("未找到最终 AIMessage。")


def _invoke_once(user_input: str, thread_id: str, passenger_id: str = PASSENGER_ID):
    """单轮调用 graph.invoke。"""
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input,
                }
            ]
        },
        config={"configurable": {"thread_id": thread_id}},
        context={"passenger_id": passenger_id},
    )
    return result


def _query_ticket_flight_row(ticket_no: str) -> dict:
    conn = sqlite3.connect(TRAVEL_NEW_DB)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT ticket_no, flight_id, fare_conditions, amount
        FROM ticket_flights
        WHERE ticket_no = ?
        """,
        (ticket_no,),
    )
    row = cursor.fetchone()
    if not row:
        cursor.close()
        conn.close()
        raise AssertionError(f"ticket_flights 中未找到票号 {ticket_no} 的记录。")
    columns = [col[0] for col in cursor.description]
    result = dict(zip(columns, row))
    cursor.close()
    conn.close()
    return result


def _query_departure_flights(departure_airport: str, limit: int = 5) -> list[dict]:
    conn = sqlite3.connect(TRAVEL_NEW_DB)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT flight_id, flight_no, departure_airport, arrival_airport, scheduled_departure
        FROM flights
        WHERE departure_airport = ?
        ORDER BY scheduled_departure
        LIMIT ?
        """,
        (departure_airport, limit),
    )
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    result = [dict(zip(columns, row)) for row in rows]
    cursor.close()
    conn.close()
    return result


def _query_route_flights(departure_airport: str, arrival_airport: str, limit: int = 10) -> list[dict]:
    conn = sqlite3.connect(TRAVEL_NEW_DB)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT flight_id, flight_no, departure_airport, arrival_airport, scheduled_departure
        FROM flights
        WHERE departure_airport = ?
          AND arrival_airport = ?
        ORDER BY scheduled_departure
        LIMIT ?
        """,
        (departure_airport, arrival_airport, limit),
    )
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    result = [dict(zip(columns, row)) for row in rows]
    cursor.close()
    conn.close()
    return result


def _choose_current_ticket_record(passenger_id: str = PASSENGER_ID) -> dict:
    """选取当前用户的一条真实机票记录作为测试基准。"""
    records = query_user_flight_information(passenger_id)
    if not records:
        raise AssertionError(f"当前 passenger_id={passenger_id} 未查询到任何机票信息，请先检查测试数据库。")
    return records[0]


def _find_alternate_flight_same_route(current_record: dict) -> dict:
    """
    为改签成功测试选择一个“同航线但不同 flight_id”的目标航班。
    """
    departure = current_record["departure_airport"]
    arrival = current_record["arrival_airport"]
    current_flight_id = str(current_record["flight_id"])

    conn = sqlite3.connect(TRAVEL_NEW_DB)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT flight_id, flight_no, departure_airport, arrival_airport, scheduled_departure
        FROM flights
        WHERE departure_airport = ?
          AND arrival_airport = ?
          AND CAST(flight_id AS TEXT) != ?
        ORDER BY scheduled_departure
        """,
        (departure, arrival, current_flight_id),
    )
    row = cursor.fetchone()
    if not row:
        cursor.close()
        conn.close()
        raise AssertionError(
            f"未找到可用于改签成功测试的候选航班。当前航线为 {departure} -> {arrival}，"
            f"请先在 flights 表中插入一条同航线未来航班。"
        )
    columns = [col[0] for col in cursor.description]
    result = dict(zip(columns, row))
    cursor.close()
    conn.close()
    return result


# =========================
# fixture：保护测试数据库
# =========================
@pytest.fixture()
def restore_db_after_test(tmp_path: Path):
    """
    备份当前测试库，测试结束后恢复。
    这样改签成功之类的写操作不会污染后续测试。
    """
    original_db = _db_path()
    backup_db = tmp_path / "travel_new_backup.sqlite"
    shutil.copyfile(original_db, backup_db)

    yield

    shutil.copyfile(backup_db, original_db)


# =========================
# 最小回归测试
# =========================
def test_01_query_current_ticket_info():
    """
    回归目标：
    - fetch_user_info_node 注入用户航班背景
    - supervisor 正确路由到 flight_booking_agent
    - flight agent 能返回当前机票信息
    """
    thread_id = _unique_thread_id()
    current_record = _choose_current_ticket_record()

    result = _invoke_once("我想要查看一下我的机票信息", thread_id)
    final_text = _extract_last_ai_text(result["messages"])

    assert str(current_record["ticket_no"]) in final_text
    assert str(current_record["flight_no"]) in final_text
    assert str(current_record["departure_airport"]) in final_text
    assert str(current_record["arrival_airport"]) in final_text


def test_02_multi_turn_followup_same_thread():
    """
    回归目标：
    - 同一 thread_id 下多轮记忆有效
    - 第二轮无需重复 ticket_no 仍能追问成功
    """
    thread_id = _unique_thread_id()
    current_record = _choose_current_ticket_record()

    _invoke_once("我想要查看一下我的机票信息", thread_id)
    result = _invoke_once("我机票的航班号和座位号，起飞机场各是什么？", thread_id)
    final_text = _extract_last_ai_text(result["messages"])

    assert str(current_record["flight_no"]) in final_text
    assert str(current_record["departure_airport"]) in final_text

    seat_no = current_record.get("seat_no")
    if seat_no is not None:
        assert str(seat_no) in final_text


def test_03_search_flights_by_departure_airport():
    """
    回归目标：
    - flight agent 能按出发机场查询航班
    - 结果中应至少出现数据库真实存在的一条航班
    """
    thread_id = _unique_thread_id()
    current_record = _choose_current_ticket_record()
    departure = current_record["departure_airport"]

    db_flights = _query_departure_flights(departure, limit=5)
    assert db_flights, f"数据库中未查询到从 {departure} 起飞的航班，无法执行该回归测试。"

    result = _invoke_once(f"近期从 {departure} 起飞的航班都有哪些？", thread_id)
    final_text = _extract_last_ai_text(result["messages"])

    # 至少要命中一条真实 flight_no
    assert any(str(row["flight_no"]) in final_text for row in db_flights), (
        f"系统返回中未命中任何数据库真实 flight_no。数据库候选={db_flights}，返回文本={final_text}"
    )


def test_04_search_flights_by_departure_and_arrival():
    """
    回归目标：
    - flight agent 能联合使用出发地 + 到达地查询
    """
    thread_id = _unique_thread_id()
    current_record = _choose_current_ticket_record()
    departure = current_record["departure_airport"]
    arrival = current_record["arrival_airport"]

    db_flights = _query_route_flights(departure, arrival, limit=10)
    assert db_flights, f"数据库中未查询到 {departure} -> {arrival} 的航班，无法执行该回归测试。"

    result = _invoke_once(f"帮我查询一下近期从 {departure} 飞往 {arrival} 的航班", thread_id)
    final_text = _extract_last_ai_text(result["messages"])

    assert arrival in final_text
    assert any(str(row["flight_no"]) in final_text for row in db_flights), (
        f"系统返回中未命中任何数据库真实 {departure}->{arrival} 航班。"
    )


def test_05_update_ticket_success(restore_db_after_test):
    """
    回归目标：
    - flight agent 改签成功链路可用
    - ticket_flights 中的 flight_id 发生更新
    """
    thread_id = _unique_thread_id()
    current_record = _choose_current_ticket_record()
    target_flight = _find_alternate_flight_same_route(current_record)

    ticket_no = str(current_record["ticket_no"])
    current_flight_id = str(current_record["flight_id"])
    new_flight_id = str(target_flight["flight_id"])

    # 防御性校验：新旧必须不同
    assert current_flight_id != new_flight_id

    result = _invoke_once(
        f"帮我把票号 {ticket_no} 改签到 flight_id 为 {new_flight_id} 的航班",
        thread_id,
    )
    final_text = _extract_last_ai_text(result["messages"])

    assert ("成功" in final_text) or ("已" in final_text), f"改签结果文本看起来不像成功：{final_text}"

    updated_row = _query_ticket_flight_row(ticket_no)
    assert str(updated_row["flight_id"]) == new_flight_id, (
        f"数据库未正确更新 flight_id。期望={new_flight_id}，实际={updated_row['flight_id']}"
    )


def test_06_update_ticket_fail_when_using_flight_no_as_flight_id(restore_db_after_test):
    """
    回归目标：
    - 错把 flight_no 当成 flight_id 时，应明确失败
    - 不应发生实际数据库更新
    """
    thread_id = _unique_thread_id()
    current_record = _choose_current_ticket_record()
    target_flight = _find_alternate_flight_same_route(current_record)

    ticket_no = str(current_record["ticket_no"])
    original_row = _query_ticket_flight_row(ticket_no)
    wrong_value = str(target_flight["flight_no"])   # 故意传 flight_no，而不是 flight_id

    with pytest.raises(Exception) as exc_info:
        _invoke_once(
            f"帮我把票号 {ticket_no} 改签到 flight_id 为 {wrong_value} 的航班",
            thread_id,
        )

    assert "未找到对应的航班信息" in str(exc_info.value)

    after_row = _query_ticket_flight_row(ticket_no)
    assert str(after_row["flight_id"]) == str(original_row["flight_id"]), "失败场景下数据库不应被修改。"
