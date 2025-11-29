from datetime import date, timedelta
from typing import List
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from experiment import convert_to_weeklyTodos, run_one_week
from weather_utils import build_weather_recommendation

app = FastAPI(title="AI-SS API", version="1.0.0")

# JSON paths
data_dir = Path(__file__).resolve().parent.parent / "data"
base_date_path = data_dir / "base_date.json"
user_cleaning_status_path = data_dir / "user_cleaning_status.json"
behavior_history_path = data_dir / "behavior_history.json"
weekly_todo_history_path = data_dir / "weekly_todo_history.json"

# TODO: move to environment variable in production
OPENWEATHER_API_KEY = "6a96849159eb130473a048891638bed8"


class OneWeekInput(BaseModel):
    week_start: date
    seed: int = 42


class WeatherTodo(BaseModel):
    title: str
    description: str


class WeatherRecommendResponse(BaseModel):
    todos: List[WeatherTodo]


def load_json_dict(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


base_date = load_json_dict(base_date_path)
user_cleaning_status = load_json_dict(user_cleaning_status_path)
behavior_history = load_json_dict(behavior_history_path)
weekly_todo_history = load_json_dict(weekly_todo_history_path)


# -------- Schedule endpoint --------
@app.post("/api/v1/schedule/run-one-week")
def generate_one_week_schedule(body: OneWeekInput):
    global base_date

    if not base_date:
        base_date["date"] = str(body.week_start)

    start_date = date.fromisoformat(base_date["date"])
    delta_days = (body.week_start - start_date).days
    this_week = (delta_days // 7) + 1

    prev_week_date = body.week_start - timedelta(days=7)
    prev_behavior = behavior_history.get(str(prev_week_date), None)
    last_week_todo = weekly_todo_history.get(str(prev_week_date), None)

    behavior_vector, weekly_todo = run_one_week(
        this_week,
        body.week_start,
        last_week_todo,
        prev_behavior,
        user_cleaning_status,
    )

    behavior_history[str(body.week_start)] = (
        behavior_vector.tolist()
        if isinstance(behavior_vector, np.ndarray)
        else behavior_vector
    )
    weekly_todo_history[str(body.week_start)] = weekly_todo

    # save JSON files
    with open(behavior_history_path, "w", encoding="utf-8") as f:
        json.dump(behavior_history, f, ensure_ascii=False, indent=4)

    with open(weekly_todo_history_path, "w", encoding="utf-8") as f:
        json.dump(weekly_todo_history, f, ensure_ascii=False, indent=4)

    with open(base_date_path, "w", encoding="utf-8") as f:
        json.dump(base_date, f, ensure_ascii=False, indent=4)

    weeklyTodos = convert_to_weeklyTodos(weekly_todo, body.week_start)

    return {"weeklyTodos": weeklyTodos}


# -------- Weather-based recommendation endpoint --------
@app.post(
    "/api/v1/recommend/weather-tasks",
    response_model=WeatherRecommendResponse,
)
def get_weather_recommendation():
    result = build_weather_recommendation(
        data_dir=data_dir,
        api_key=OPENWEATHER_API_KEY,
    )
    todos_raw = result.get("todos", [])
    return WeatherRecommendResponse(todos=todos_raw)

# json 초기화 함수
def reset_scheduling_file():  
    base_date_file             = data_dir / "base_date.json"
    behavior_history_file      = data_dir / "behavior_history.json"
    weekly_todo_history_file   = data_dir / "weekly_todo_history.json"
    user_execution_data_file   = data_dir / "user_execution_data.json"
    cleaning_status_file       = data_dir / "user_cleaning_status.json"
    cleaning_status_ori_file   = data_dir / "ori_user_cleaning_status.json"

    # 1) {} 로 초기화해야 하는 파일 목록
    empty_init_files = [
        base_date_file,
        behavior_history_file,
        weekly_todo_history_file,
        user_execution_data_file,
    ]

    # 2) {}로 reset
    for file_path in empty_init_files:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=4)

    # 3) user_cleaning_status.json -> ori_user_cleaning_status.json 내용 복사
    if cleaning_status_ori_file.exists():
        with open(cleaning_status_ori_file, "r", encoding="utf-8") as f:
            ori_data = json.load(f)
    else:
        raise FileNotFoundError(f"{cleaning_status_ori_file} not found.")

    with open(cleaning_status_file, "w", encoding="utf-8") as f:
        json.dump(ori_data, f, ensure_ascii=False, indent=4)

    return {
        "status": "ok",
        "message": "All scheduling files reset successfully."
    }

if __name__ == "__main__":
    from pprint import pprint
    
    # 스케줄링 첫 사용 상태로 파일 초기화
    reset_res = reset_scheduling_file()
    pprint(reset_res)
    
    # 테스트 예시 - 1주일 청소 스케줄  
    body = OneWeekInput(week_start="2025-11-24")
    resp = generate_one_week_schedule(body) 
    pprint(resp)
    
    # 테스트 예시 - 날씨 맞춤 청소 추천
    API_KEY = "6a96849159eb130473a048891638bed8"
    result = get_weather_recommendation()
    pprint(result)
    