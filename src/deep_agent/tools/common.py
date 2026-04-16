from typing import Dict

CITY_MAP: Dict[str, str] = {
    '北京': 'Beijing',
    '上海': 'Shanghai',
    '广州': 'Guangzhou',
    '深圳': 'Shenzhen',
    '成都': 'Chengdu',
    '杭州': 'Hangzhou',
    '巴塞尔': 'Basel',
    '苏黎世': 'Zurich',
    # 添加更多的城市映射...
}


def transform_loc(city: str = None) -> str:
    if city is None:
        return "No city is provided'"
    if all('\u4e00' <= char <= '\u9fff' for char in city):
        return CITY_MAP.get(city, "invalid city name")
    return city
