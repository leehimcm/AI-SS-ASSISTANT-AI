import random 
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CleaningEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        fixed_schedules,
        cleaning_tasks,
        user_cleaning_status,
        user_behavior,
        user_execution_data,
        user_pref,
        slot_penalties,
        initial_schedules=None,
        user_feedback=None,
    ):
        super().__init__()
 
        self.fixed_schedules = fixed_schedules
        self.cleaning_tasks = cleaning_tasks
        self.user_cleaning_status = user_cleaning_status
        self.behavior_vector = np.asarray(user_behavior, dtype=np.float32)
        self.user_execution_data = user_execution_data
        self.slot_penalties = slot_penalties
        self.initial_schedules = initial_schedules or []
        self.survey_responses = user_pref
        self.user_feedback = user_feedback
 
        self.task_id2name = []
        self.task_name2meta = {}
        for i, d in enumerate(self.cleaning_tasks):
            name = list(d.keys())[0]
            self.task_id2name.append(name)
            self.task_name2meta[name] = d[name]
 
        self.n_tasks = len(self.cleaning_tasks)
        self.current_step = 0
        self.max_steps = 32 
        
        self.action_space = spaces.MultiDiscrete([self.n_tasks, 7, 24]) # action: (task_idx, day, hour)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(168 * 3,), dtype=np.float32)

        self.state = None                   # (504,)
        self.weekly_todo = []               # 최종 결과: [{title, day, hour}, ...]
        self.used_slots = set()             # (day, hour) 중복 방지

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed); random.seed(seed)
        super().reset(seed=seed)

        self.current_step = 0
        self.weekly_todo = []
        self.used_slots = set()
        self.state = self._get_initial_state()

        if self.user_execution_data:
            self.state[168:168+168] = np.maximum(0.0, self.state[168:336] - self.slot_penalties)

        return self.state.copy().astype(np.float32), {}

    def _get_initial_state(self):
        fixed_vector = np.zeros(168, dtype=np.float32)
        urgency_vector = np.zeros(168, dtype=np.float32)

        # 1) 고정 일정 마스킹
        for sched in self.fixed_schedules:
            start = int(sched['day'] * 24 + sched['start'])
            end = int(sched['day'] * 24 + sched['end'])
            start = max(0, min(167, start))
            end = max(0, min(168, end))
            if end > start:
                fixed_vector[start:end] = 1.0

        # 2) 급함도(urgency) 초기화: 각 task의 (last_done_days_ago / interval)을 168 슬롯에 균등 분산
        for d in self.cleaning_tasks:
            name = list(d.keys())[0]
            meta = d[name]
            interval = float(meta.get('interval', 7) or 0)  # None 방지
            last_done = float(self.user_cleaning_status.get(name, {}).get('last_done_days_ago', 0.0))
            if interval <= 0:
                urgency = 1.0
            else:
                urgency = min(1.0, last_done / (interval + 1e-8))
            urgency_vector += (urgency / 168.0) 
        np.clip(urgency_vector, 0.0, 1.0, out=urgency_vector)

        # 3) behavior_vector 체크
        if self.behavior_vector.shape[0] != 168:
            raise ValueError(f"behavior_vector length must be 168, got {self.behavior_vector.shape[0]}")
        behavior_vector = np.clip(self.behavior_vector, 0.0, 1.0).astype(np.float32)
        state = np.concatenate([fixed_vector, urgency_vector, behavior_vector], axis=0).astype(np.float32)
        return state

    def step(self, action): # action: (task_idx, day, hour)
        task_idx, day, hour = int(action[0]), int(action[1]), int(action[2])
        day = int(np.clip(day, 0, 6))
        hour = int(np.clip(hour, 0, 23)) 
        task_name = self.task_id2name[task_idx]
        meta = self.task_name2meta[task_name]
        add_task = True

        # 주당 최대 배정 횟수 체크
        interval = int(meta.get('interval', 7) or 7)
        max_per_week = max(1, 7 // interval)
        assigned_count = sum(1 for (n, _, _) in self.weekly_todo if n == task_name)

        if assigned_count >= max_per_week:
            add_task = False

        # 사용된 슬롯 기록
        is_duplicate = (day, hour) in self.used_slots
        if not is_duplicate:
            self.used_slots.add((day, hour))

        # 고정 일정 충돌 검사
        for sched in self.fixed_schedules:
            if sched["day"] == day and sched["start"] <= hour < sched["end"]:
                add_task = False
                break

        # 상태 업데이트
        idx = day * 24 + hour
        self.state[168 + idx] = max(0.0, self.state[168 + idx] - 0.5) # urgency 감소

        # 스케줄 기록
        if add_task:
            self.weekly_todo.append((task_name, day, hour))

        # 보상 계산
        reward = self._calculate_reward(task_name, day, hour, meta, is_duplicate)

        # 종료 판정
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        info = {}
        if terminated or truncated:
             self.weekly_todo_final = self.weekly_todo.copy()

        next_state = self.state.copy().astype(np.float32)
        return next_state, float(reward), terminated, truncated, info

    def _calculate_reward(self, task_name, day, start, meta, is_duplicate):
        slot_penalty = 0.0
        burden_penalty = 0.0
        urgency_score = 0.0
        frequency_score = 0.0
        preference_score = 0.0
        initial_hint = 0.0
        onboarding_bonus = 0.0
        feedback_score = 0.0

        idx = int(day * 24 + start)
        sr = self.survey_responses # 온보딩

        # 1) 슬롯 중복 패널티
        cd = sr['cleaning_distribution'] # [consistency, intensity]
        if is_duplicate:
            slot_penalty -= (2 * cd[0])

        # 2) 야간/새벽 부담
        if start < 7 or start > 22:
            burden_penalty -= 0.5

        # 3) 청소 급함도(많이 밀린 청소) & 자주 해야하는 청소
        last_done_days_ago = float(self.user_cleaning_status.get(task_name, {}).get("last_done_days_ago", 0.0))
        interval = float(meta.get("interval", 7) or 0.0)
        urgency_score += (last_done_days_ago / interval)
        frequency_score += (last_done_days_ago - interval + (7.0 if interval <= 4 else 0))

        # 4) 사용자 시간 선호도
        preference_score += float(self.behavior_vector[idx]) * 1.5

        # 5) ILP 결과와 유사하면 보상
        if self.initial_schedules:
            if [task_name, day, start] in self.initial_schedules:
                initial_hint += 3.0

        # 6) 선호 요일/시간 활용
        pt = sr['preferred_time_range'] # [start, end]
        pd = sr['preferred_days'] # []

        if pt[0] <= start < pt[1]:  
            onboarding_bonus += 0.3
        else:
            onboarding_bonus -= 0.1

        if day in pd:  
            onboarding_bonus += 0.3
        else:
            onboarding_bonus -= 0.1

        # 7) 사용자 피드백
        feedback_score += (self.user_feedback['cleaning_amount_score'] + self.user_feedback['recommended_time_score']) / 5.0

        return float(slot_penalty + urgency_score + frequency_score + burden_penalty + preference_score + initial_hint + onboarding_bonus + feedback_score)
