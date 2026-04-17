from datetime import datetime

from langchain.agents.middleware import ModelRequest


def get_context_content(request: ModelRequest) -> dict:
    passenger_id = request.runtime.context.passenger_id
    return {'passenger_id': passenger_id}


def format_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")
