import shutil
import sqlite3
import pandas as pd
from deep_agent.config import TRAVEL_NEW_DB, TRAVEL_DB

# 这个数据库是项目测试中使用的
local_file = TRAVEL_NEW_DB
# 创建备份文件，允许在测试的时候重新开始
backup_file = TRAVEL_DB

import shutil
import sqlite3
import pandas as pd

from deep_agent.config import TRAVEL_NEW_DB, TRAVEL_DB

local_file = TRAVEL_NEW_DB
backup_file = TRAVEL_DB


def update_dates() -> str:
    """
    将整个旅行数据库的时间轴整体平移到“当前时间之后”。
    目标：
    - 最早的一班 scheduled_departure 落在当前时间 + 2 天
    - 所有航班都整体位于未来，便于测试“未来一周”“改签”等场景
    """
    shutil.copy(backup_file, local_file)
    conn = sqlite3.connect(local_file)

    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    )["name"].tolist()

    tdf = {}
    for table_name in tables:
        tdf[table_name] = pd.read_sql(f"SELECT * FROM {table_name}", conn)

    # 1. 用 scheduled_departure 做时间轴基准，而不是 actual_departure
    flights_sched = pd.to_datetime(
        tdf["flights"]["scheduled_departure"].replace("\\N", pd.NaT)
    )

    earliest_departure = flights_sched.min()

    # 2. 目标：让最早的一班航班出现在“当前时间 + 2 天”
    now = pd.Timestamp.now(tz=earliest_departure.tz)
    target_start = now + pd.Timedelta(days=2)

    time_diff = target_start - earliest_departure

    print("earliest_departure =", earliest_departure)
    print("target_start       =", target_start)
    print("time_diff          =", time_diff)

    # 3. 更新 bookings 表
    if "bookings" in tdf and "book_date" in tdf["bookings"].columns:
        tdf["bookings"]["book_date"] = (
                pd.to_datetime(
                    tdf["bookings"]["book_date"].replace("\\N", pd.NaT),
                    utc=True
                ) + time_diff
        )

    # 4. 更新 flights 表所有相关时间列
    datetime_columns = [
        "scheduled_departure",
        "scheduled_arrival",
        "actual_departure",
        "actual_arrival",
    ]
    for column in datetime_columns:
        if column in tdf["flights"].columns:
            tdf["flights"][column] = (
                    pd.to_datetime(
                        tdf["flights"][column].replace("\\N", pd.NaT)
                    ) + time_diff
            )

    # 5. 写回数据库
    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, index=False, if_exists="replace")

    conn.commit()
    conn.close()
    return local_file


if __name__ == "__main__":
    db = update_dates()
    print(db)
