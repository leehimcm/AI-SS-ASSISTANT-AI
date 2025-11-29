from datetime import timedelta, date
import uuid

from experiment_utils import generate_sleeping_data,build_user_execution_data, build_user_feedback, generate_raw_schedules
from trainer import make_schedule
from data_processing import compute_user_behavior

from pathlib import Path
import json
DATA_DIR = Path(__file__).resolve().parent.parent / "data" 

with open(DATA_DIR / "user_info.json", "r", encoding="utf-8") as f:
    user_info = json.load(f) 
with open(DATA_DIR / "user_execution_data.json", "r", encoding="utf-8") as f:
    user_execution_data = json.load(f)

def convert_to_weeklyTodos(weekly_todo, week_start):
    by_day = {d: [] for d in range(7)}
    for task_name, day, hour in weekly_todo:
        by_day[day].append((task_name, hour))

    for d in range(7):
        by_day[d].sort(key=lambda x: (x[1], x[0]))

    def new_id():
        return uuid.uuid4().hex[:6]

    def to_hhmm(hour: int):
        return f"{hour:02d}:00"

    weeklyTodos = []
    for d in range(7):
        date_str = (week_start + timedelta(days=d)).strftime("%Y-%m-%d")
        todos = [
            {
                "id": new_id(),
                "title": task_name,
                "startHour": to_hhmm(hour),
                "isDone": False
            }
            for task_name, hour in by_day[d]
        ]
        weeklyTodos.append({"date": date_str, "todos": todos})

    return weeklyTodos


def run_one_week(this_week: int, week_start: date, last_week_todo=None, prev_behavior=None, user_cleaning_status=None):

    raw_schedules = generate_raw_schedules(week_start) # 구글 캘린더

    sleep_at, wake_at = user_info['survey_responses']['sleep_at'], user_info['survey_responses']['wake_at'] # ok
    sleeping = generate_sleeping_data(week_start, sleep_at=sleep_at, wake_at=wake_at)  

    user_feedback = None
    prev_user_exec = None
    
    if this_week > 1:
        user_feedback = build_user_feedback(week_start)  
        prev_user_exec = build_user_execution_data( 
            last_week_todo=last_week_todo,
            week=this_week-1,
            seed=42
        )
        
        for w, records in prev_user_exec.items():
            user_execution_data[str(w)] = records
            
        with open(DATA_DIR / "user_execution_data.json", "w", encoding="utf-8") as f:
            json.dump(user_execution_data, f, ensure_ascii=False, indent=4)
   
    initial_todo, weekly_todo, prev_behavior = make_schedule(
        this_week=this_week,
        week_start=week_start,
        goocal_data=raw_schedules,
        sleep_data=sleeping,
        user_execution_data=user_execution_data,
        prev_behavior=prev_behavior,
        user_cleaning_status=user_cleaning_status,
        user_feedback=user_feedback,
    )
 
    behavior_vector = None
    if this_week > 1: 
        behavior_vector = compute_user_behavior(prev_user_exec[this_week-1])

    return behavior_vector, weekly_todo
