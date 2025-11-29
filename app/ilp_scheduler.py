import pulp

def generate_initial_schedule(cleaning_tasks, user_cleaning_status, fixed_schedules,
                              total_cleaning_task_limit=10, total_cleaning_time_limit=5):
    # 1) ILP 문제 정의
    prob = pulp.LpProblem("CleaningScheduler", pulp.LpMaximize)

    # 2) 변수: 각 청소를 x요일 x시에 할지 말지 정함
    variables = {}
    for i, task in enumerate(cleaning_tasks):
        for d in range(7):  # 0 ~ 6(월요일~일요일)
            for h in range(24):  # 청소 시작시간
                variables[(i, d, h)] = pulp.LpVariable(f'task_{i}_day{d}_hour{h}', 0, 1, cat='Binary')

    # 3) 목적 함수 : 'interval', 'last_done_days_ago' 만족 -> 우선순위가 높은(미룬지 오래된) 청소 작업을 더 많이 포함
    objective = pulp.lpSum(
        (
            (
                (user_cleaning_status[task_name]["last_done_days_ago"] - task['interval'])
                + (7.0 if task['interval'] <= 3 else 0.0)
            ) / task['interval'] * variables[(i, d, h)]
        )
        for i, task_dict in enumerate(cleaning_tasks)
        for task_name, task in task_dict.items()
        for d in range(7) for h in range(24)
    )
    prob += objective

    # 4) 제약 조건
    # 제약 조건 안에서 최대한 많은 청소를 배정
    prob += pulp.lpSum([
        variables[(i, d, h)]
        for i in range(len(cleaning_tasks))
        for d in range(7) for h in range(24)])

    # 총 청소 개수 제한
    prob += pulp.lpSum([
        variables[(i, d, h)]
        for i in range(len(cleaning_tasks))
        for d in range(7) for h in range(24)]) <= total_cleaning_task_limit

    # 총 청소 시간 제한
    prob += pulp.lpSum([
        variables[(i, d, h)] * task['duration']
        for i, task_dict in enumerate(cleaning_tasks)
        for task_name, task in task_dict.items()
        for d in range(7) for h in range(24)]) <= total_cleaning_time_limit

    # 고정 일정 제외한 시간에만 청소 배정
    for i, task_dict in enumerate(cleaning_tasks):
        for task_name, task in task_dict.items():
            duration = task["duration"]
            for d in range(7):
                for h in range(24):
                    for sched in fixed_schedules:
                        if sched['day'] == d:
                            if sched['start'] < h + duration and sched['end'] > h:
                                prob += variables[(i, d, h)] == 0

    # 같은 시간대(x시:00분)에 실행되는 청소의 duration 합 <= 1시간
    for d in range(7):
        for h in range(24):
            prob += pulp.lpSum([
                variables[(i, d, h)] * task["duration"]
                for i, task_dict in enumerate(cleaning_tasks)
                for task in task_dict.values()
            ]) <= 1

    # 각 청소작업은 1주일에 (7 // interval)회 이하만 추천
    for i, task_dict in enumerate(cleaning_tasks):
        for task_name, task in task_dict.items():
            max_assign = max(1, 7 // task["interval"])
            prob += pulp.lpSum([
                variables[(i, d, h)]
                for d in range(7) for h in range(24)
            ]) <= max_assign

    # 최적화 수행
    prob.solve()

    cleaning_schedules = []
    for i, task_dict in enumerate(cleaning_tasks):
        for task_name, task in task_dict.items():
            for d in range(7):
                for h in range(24):
                    if pulp.value(variables[(i, d, h)]) == 1:
                        cleaning_schedules.append([task_name, d, h])

    cleaning_schedules.sort(key=lambda x: (x[1], x[2])) # 요일, 시간 정렬
    return cleaning_schedules