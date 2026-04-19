import uuid
from typing import Optional
from pydantic import BaseModel, Field


class GraphRequestSchema(BaseModel):
    user_input: str = Field(description="用户输入")
    thread_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="多轮会话ID"
    )
    passenger_id: Optional[str] = Field(
        default=None,
        description="业务上下文中的乘客ID"
    )


class GraphResponseSchema(BaseModel):
    assistant: str = Field(description="助手回复")