"""
静态注入 schema
"""
from dataclasses import dataclass


@dataclass
class CtripContext:
    passenger_id: str


@dataclass
class SearchContext:
    user_id: str
    role: str = "user"
