import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 存储根路径
STORAGE_ROOT = Path(os.getenv('STORAGE_DIR', PROJECT_ROOT / 'storage'))
# 项目中使用到的两个数据库
TRAVEL_DB = STORAGE_ROOT / "travel2.sqlite"
TRAVEL_NEW_DB = STORAGE_ROOT / "travel_new.sqlite"
