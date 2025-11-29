from datetime import datetime, timedelta
import numpy as np

def compute_user_cleaning_status(user_cleaning_status, prev_week_execution):
    user_cleaning_status = user_cleaning_status or {}
    not_done = set(list(user_cleaning_status.keys())) 
    for entry in prev_week_execution:
        if not entry['performed']:
            continue
        task = entry['task']
        prev_days_ago = user_cleaning_status[task]['last_done_days_ago']
        new_days_ago = 7 - entry['day'] 
        if new_days_ago < prev_days_ago: 
            user_cleaning_status[task]['last_done_days_ago'] = new_days_ago
        not_done.discard(task) 

    for task in not_done:
        user_cleaning_status[task]['last_done_days_ago'] += 7 # 이번주에 안 한 청소는 'last_done_days_ago' 7일 증가

    return user_cleaning_status
    
def compute_user_behavior(prev_week_execution, prev_behavior=None, alpha=0.5):
    vector = np.zeros(168)
    for entry in prev_week_execution:
        index = entry['day'] * 24 + entry['hour'] 
        if entry['performed']:
            vector[index] += 1.0

    if vector.sum() > 0:
        vector /= vector.sum()
    vector = vector.astype(np.float32)  

    if prev_behavior is None:
        return vector
    else:
        return alpha * prev_behavior + (1 - alpha) * vector 

def preprocessing_schedules(week_start, raw_schedules):
    week_end = week_start + timedelta(days=7)
    fixed_schedules = []
    for item in raw_schedules:
        start_dt = item['start']['dateTime']
        end_dt = item['end']['dateTime']

        # +09:00 제거
        start_dt = datetime.fromisoformat(start_dt[:-6])
        end_dt = datetime.fromisoformat(end_dt[:-6])

        # 여러 날에 걸친 일정을 하루 단위로 분리
        current = start_dt
        while current.date() <= end_dt.date():
            day_start = current.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)

            seg_start = max(start_dt, day_start)
            seg_end = min(end_dt, day_end)

            if not (week_start <= seg_start.date() < week_end): 
                current = day_end
                continue

            weekday = seg_start.weekday() # 요일 0=월, 6=일
            start_hour = seg_start.hour + seg_start.minute / 60
            end_hour = seg_end.hour + seg_end.minute / 60

            fixed_schedules.append({
                'summary': item['summary'],
                'location': item['location'],
                'day': weekday,
                'start': start_hour,
                'end': end_hour
            })

            current = day_end

    return fixed_schedules

def prepare_user_state(user_execution_data, this_week, user_cleaning_status):
    prev_week_execution = user_execution_data.get(this_week-1, [])
    user_cleaning_status = compute_user_cleaning_status(user_cleaning_status, prev_week_execution)
    user_behavior = compute_user_behavior(prev_week_execution)
    if user_behavior is None:
        raise ValueError("user_behavior is None")
    return user_cleaning_status, user_behavior