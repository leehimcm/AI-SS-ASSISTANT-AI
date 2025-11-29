from datetime import date, datetime, time, timedelta
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import hashlib
import random
import numpy as np

def generate_sleeping_data(
    base_date: date,
    sleep_at: int,   # 예: 22
    wake_at: int,    # 예: 7
    seed: Optional[int] = None
) -> List[Dict]:

    sleeping_data = []
    n_days = 7   

    for i in range(-1, n_days):
        day = base_date + timedelta(days=i)
        if sleep_at < wake_at:
            start_dt = datetime.combine(day, time(sleep_at, 0))
            end_dt = datetime.combine(day, time(wake_at, 0))
            sleeping_data.append({
                "summary": "수면",
                "location": "집",
                "start": {"dateTime": start_dt.isoformat(timespec="minutes") + "+09:00"},
                "end": {"dateTime": end_dt.isoformat(timespec="minutes") + "+09:00"}
            })
        else:
            start_dt = datetime.combine(day, time(sleep_at, 0))
            end_dt = datetime.combine(day + timedelta(days=1), time(wake_at, 0))
            sleeping_data.append({
                "summary": "수면",
                "location": "집",
                "start": {"dateTime": start_dt.isoformat(timespec="minutes") + "+09:00"},
                "end": {"dateTime": end_dt.isoformat(timespec="minutes") + "+09:00"}
            })
    return sleeping_data

def build_user_execution_data(
    last_week_todo: List[List[str | int]],  
    week: int = 1,
    min_adherence: float = 1.0,
    seed: Optional[int] = None
) -> Dict[int, List[Dict]]:

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    total = len(last_week_todo)
    if total == 0:
        return {}, 1.0

    min_adherence = max(0.0, min(1.0, float(min_adherence)))
    min_to_execute = int(np.ceil(total * min_adherence))
    execute_count = random.randint(min_to_execute, total) if total > min_to_execute else total

    indices = list(range(total))
    random.shuffle(indices)
    executed_idx = set(indices[:execute_count])

    grouped_data = defaultdict(list)

    for i, entry in enumerate(last_week_todo):
        # if not (isinstance(entry, list) and len(entry) == 3):
        #     raise ValueError(f"Expected list [task_name, day, hour], but got: {entry!r}")

        task_name, day, start_hour = entry
        if int(day) == 4: # 금요일에는 청소 하지 않는 사용자 실험
            performed_flag = False
        else:
            performed_flag = (i in executed_idx)
        
        grouped_data[week].append({
            "task": str(task_name),
            "performed": performed_flag,
            "day": int(day),
            "hour": int(start_hour)
        })

    # adherence = sum(1 for x in grouped_data[week] if x["performed"]) / total
    return dict(grouped_data)

def build_user_feedback(week_start_date):
    user_feedback = {
        "week_start_date": week_start_date,
        "cleaning_amount_score": 4,
        "recommended_time_score": 5,
        "comment": ""
    }

    return user_feedback

def _seed_from_year_week(base_date: date, salt: str = "") -> int: 
    iso_year, iso_week, _ = base_date.isocalendar()
    key = f"{iso_year}-W{iso_week}-{salt}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big")  # 64비트 정수 시드

def _seed_from_year(base_date: date, salt: str = "") -> int: 
    iso_year = base_date.isocalendar()[0]
    key = f"{iso_year}-{salt}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big")

def _slot_to_hm(slot: int, slots_per_hour: int) -> Tuple[int, int]:
    hour = slot // slots_per_hour
    minute = (slot % slots_per_hour) * (60 // slots_per_hour)
    return hour, minute

def _hm_to_slot(hour: int, minute: int, slots_per_hour: int) -> int:
     return hour * slots_per_hour + minute // (60 // slots_per_hour)

def generate_raw_schedules(
    base_date: date,
    seed: Optional[int] = None,          
    tz_offset: str = "+09:00",
    salt: str = "korean_univ_18cr_v1",     
) -> List[Dict]:

    seed_val_week = _seed_from_year_week(base_date, salt) if seed is None else seed
    rng = random.Random(seed_val_week)

    seed_val_year_for_class = _seed_from_year(base_date, salt + "_class")
    rng_class = random.Random(seed_val_year_for_class)

    monday = base_date - timedelta(days=base_date.weekday())  # 0=월요일

    slot_minutes = 30
    slots_per_hour = 60 // slot_minutes           # 2
    slots_per_day = 24 * slots_per_hour           # 48

    occupancy = {d: [False] * slots_per_day for d in range(7)}
    events: List[Dict] = []

    num_courses = 6
    meetings_per_course = 2
    course_duration_slots = int(1.5 * slots_per_hour)  # 3슬롯 = 1.5시간

    class_start_earliest = _hm_to_slot(9, 0, slots_per_hour)
    class_end_latest = _hm_to_slot(18, 0, slots_per_hour)  # 끝 시각 한계
    lunch_start = _hm_to_slot(12, 0, slots_per_hour)
    lunch_end = _hm_to_slot(13, 0, slots_per_hour)

    def can_place(d: int, start_slot: int, dur_slots: int) -> bool:
        end_slot = start_slot + dur_slots
        if end_slot > slots_per_day:
            return False
        if not (end_slot <= lunch_start or start_slot >= lunch_end):
            return False
        # 점유 확인
        return not any(occupancy[d][start_slot:end_slot])

    def place_event(
        title: str,
        d: int,
        start_slot: int,
        dur_slots: int,
        location: str,
        recurrence: Optional[List[str]] = None,
    ): 
        for s in range(start_slot, start_slot + dur_slots):
            occupancy[d][s] = True
        sh, sm = _slot_to_hm(start_slot, slots_per_hour)
        start_dt = datetime.combine(monday + timedelta(days=d), time(hour=sh, minute=sm))
        end_dt = start_dt + timedelta(minutes=dur_slots * slot_minutes)
        event = {
            "summary": title,
            "location": location,
            "start": {"dateTime": start_dt.isoformat(timespec="minutes") + tz_offset},
            "end": {"dateTime": end_dt.isoformat(timespec="minutes") + tz_offset},
        }
        if recurrence:
            event["recurrence"] = recurrence
        events.append(event)

    for i in range(1, num_courses + 1):
        title = f"수업{i}"
        two_days = rng_class.sample(range(0, 5), k=meetings_per_course)

        for d in two_days:
            placed = False
            latest_start = class_end_latest - course_duration_slots
            start_candidates = list(range(class_start_earliest, latest_start + 1))
            rng_class.shuffle(start_candidates)

            for start_slot in start_candidates:
                if can_place(d, start_slot, course_duration_slots):
                    place_event(
                        title, d, start_slot, course_duration_slots,
                        location="캠퍼스",
                        recurrence=["RRULE:FREQ=WEEKLY"]
                    )
                    placed = True
                    break
            if not placed:
                for start_slot in start_candidates:
                    end_slot = start_slot + course_duration_slots
                    if end_slot <= slots_per_day and not any(occupancy[d][start_slot:end_slot]):
                        place_event(
                            title, d, start_slot, course_duration_slots,
                            location="캠퍼스",
                            recurrence=["RRULE:FREQ=WEEKLY"]
                        )
                        placed = True
                        break
            if not placed:
                for start_slot in range(0, slots_per_day - course_duration_slots + 1):
                    end_slot = start_slot + course_duration_slots
                    if not any(occupancy[d][start_slot:end_slot]):
                        place_event(
                            title, d, start_slot, course_duration_slots,
                            location="캠퍼스",
                            recurrence=["RRULE:FREQ=WEEKLY"]
                        )
                        placed = True
                        break

    meal_titles = ["밥약1", "밥약2"]
    meal_duration_slots = int(2 * slots_per_hour)  # 2시간 = 4슬롯

    lunch_start_min = _hm_to_slot(11, 0, slots_per_hour)
    lunch_start_max = _hm_to_slot(13, 0, slots_per_hour)

    dinner_start_min = _hm_to_slot(18, 0, slots_per_hour)
    dinner_start_max = _hm_to_slot(20, 0, slots_per_hour)

    for title in meal_titles:
        meal_type = rng.choice(["lunch", "dinner"])
        day_candidates = list(range(0, 7))
        rng.shuffle(day_candidates)
        placed = False

        for d in day_candidates:
            if meal_type == "lunch":
                start_candidates = list(range(lunch_start_min, lunch_start_max + 1))
                loc = "식당"
            else:
                start_candidates = list(range(dinner_start_min, dinner_start_max + 1))
                loc = "식당"
            rng.shuffle(start_candidates)

            for start_slot in start_candidates:
                end_slot = start_slot + meal_duration_slots
                if end_slot <= slots_per_day and not any(occupancy[d][start_slot:end_slot]):
                    place_event(title, d, start_slot, meal_duration_slots, location=loc)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            for d in range(7):
                for start_slot in range(dinner_start_min, slots_per_day - meal_duration_slots + 1):
                    end_slot = start_slot + meal_duration_slots
                    if not any(occupancy[d][start_slot:end_slot]):
                        place_event(title, d, start_slot, meal_duration_slots, location="식당")
                        placed = True
                        break
                if placed:
                    break

    workout_titles = ["운동1", "운동2"]
    workout_duration_slots = int(1 * slots_per_hour)  # 1시간 = 2슬롯
    workout_start_min = _hm_to_slot(20, 0, slots_per_hour)
    workout_start_max = _hm_to_slot(23, 0, slots_per_hour) - workout_duration_slots  # 23:00 이전 시작

    for title in workout_titles:
        day_candidates = list(range(0, 7))
        rng.shuffle(day_candidates)
        placed = False

        for d in day_candidates:
            start_candidates = list(range(workout_start_min, workout_start_max + 1))
            rng.shuffle(start_candidates)
            for start_slot in start_candidates:
                end_slot = start_slot + workout_duration_slots
                if end_slot <= slots_per_day and not any(occupancy[d][start_slot:end_slot]):
                    place_event(title, d, start_slot, workout_duration_slots, location="체육관")
                    placed = True
                    break
            if placed:
                break
        if not placed:
            relax_min = _hm_to_slot(19, 0, slots_per_hour)
            relax_max = _hm_to_slot(23, 30, slots_per_hour) - workout_duration_slots
            for d in range(7):
                for start_slot in range(relax_min, max(relax_min, relax_max) + 1):
                    end_slot = start_slot + workout_duration_slots
                    if end_slot <= slots_per_day and not any(occupancy[d][start_slot:end_slot]):
                        place_event(title, d, start_slot, workout_duration_slots, location="체육관")
                        placed = True
                        break
                if placed:
                    break
 
    events.sort(key=lambda ev: ev["start"]["dateTime"])
    return events

if __name__ == "__main__":
    # build_user_execution_data 동작 테스트
    
    last_week_todos = {
        "2025-11-03": [
            ["거실 - 청소기로 바닥 먼지 제거", 1, 21],
            ["화장실 - 변기 청소", 2, 23],
            ["쓰레기 배출", 3, 16],
            ["창틀 먼지 닦기", 4, 19],
            ["화장실 - 바닥 청소", 5, 9],
            ["주방 - 냉장고 유통기한 지난 음식 1개 이상 처분", 5, 14],
            ["화장실 - 하수구 청소", 5, 15],
            ["거실 - 청소기로 바닥 먼지 제거", 6, 17],
            ["주방 - 행주 삶기", 6, 18],
            ["세탁실 - 세탁기 세제통 씻기", 6, 20]
        ]
    }
 
    result = build_user_execution_data(
        last_week_todo=last_week_todos["2025-11-03"],
        week=1, 
        seed=42
    )

    print("=== build_user_execution_data 출력 ===")
    for item in result[1]:
        print(item)

    print("\n=== 금요일(요일=4) 수행 여부 검사 ===")
    for item in result[1]:
        if item["day"] == 4:
            print(f"[금요일 확인] task='{item['task']}', performed={item['performed']}")
