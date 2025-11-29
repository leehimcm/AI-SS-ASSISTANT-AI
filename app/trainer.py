import random
from collections import deque
from datetime import date
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.policy import PPOPolicy

from data_processing import preprocessing_schedules, prepare_user_state
from ilp_scheduler import generate_initial_schedule
from rl_utils import multi_categorical_dist_fn, create_cleaning_env_factory
from pathlib import Path
import json

with open(Path(__file__).resolve().parent.parent / "data" / "user_info.json", "r", encoding="utf-8") as f:
    user_info = json.load(f) 
with open(Path(__file__).resolve().parent.parent / "data" / "cleaning_tasks.json", "r", encoding="utf-8") as f:
    cleaning_tasks = json.load(f) 

def create_ppo_policy(env, device):
    state_shape = env.observation_space.shape or env.observation_space.n
    nvec = np.array(env.action_space.nvec, dtype=int)
    action_shape = int(nvec.sum())

    net_actor = Net(state_shape=state_shape, hidden_sizes=[128, 128], device=device, activation=nn.ReLU)
    actor = Actor(net_actor, action_shape=action_shape, device=device).to(device)
    critic = Critic(Net(state_shape=state_shape, hidden_sizes=[128, 128], device=device, activation=nn.ReLU), device=device).to(device)

    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    dist_fn = multi_categorical_dist_fn(env.action_space.nvec)
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist_fn,
        action_space=env.action_space,
        max_grad_norm=0.5,
        vf_coef=0.5,
        ent_coef=0.01,
        reward_normalization=False,
        advantage_normalization=True,
        eps_clip=0.2,
        action_scaling=False,
    )
    return policy

def update_schedule(
    fixed_schedules,
    cleaning_schedules,
    cleaning_tasks,
    user_execution_data,
    this_week,
    prev_behavior,
    user_cleaning_status,
    user_feedback=None,
    log_dir=None,
    SEED=None,
):
    if SEED:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # 사용자 상태 준비, 파일에 쓰기
    user_cleaning_status, user_behavior = prepare_user_state(user_execution_data, this_week, user_cleaning_status)
    user_cleaning_status_path = Path(__file__).resolve().parent.parent / "data" / "user_cleaning_status.json"
    with open(user_cleaning_status_path, "w", encoding="utf-8") as f:
        json.dump(user_cleaning_status, f, ensure_ascii=False, indent=4)

    # 환경 팩토리
    make_env = create_cleaning_env_factory(
        fixed_schedules=fixed_schedules,
        cleaning_tasks=cleaning_tasks,
        user_cleaning_status=user_cleaning_status,
        user_behavior=user_behavior,
        user_execution_data=user_execution_data,
        user_pref=user_info['survey_responses'],
        cleaning_schedules=cleaning_schedules,
        user_feedback=user_feedback,
    )
 
    env = make_env()
 
    train_env_num, test_env_num = 8, 4
    train_envs = DummyVectorEnv([make_env for _ in range(train_env_num)])
    test_envs = DummyVectorEnv([make_env for _ in range(test_env_num)])
    train_envs.seed(SEED)
    test_envs.seed(SEED) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = create_ppo_policy(env, device)
    train_collector = Collector(policy, train_envs)
    test_collector = Collector(policy, test_envs) 
 
    reward_history = deque(maxlen=5)
    def stop_fn(mean_rewards: float) -> bool:
        reward_history.append(mean_rewards)
        moving_avg = np.mean(reward_history)
        # print(f"[stop_fn] mean_rewards={mean_rewards:.3f} | moving_avg(5)={moving_avg:.3f}")
        return len(reward_history) == reward_history.maxlen and moving_avg >= 500.0
 
    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=10,
        step_per_epoch=1024,
        step_per_collect=256,
        repeat_per_collect=4,
        episode_per_test=test_env_num,
        batch_size=64,
        stop_fn=stop_fn,
        logger=None,
        test_in_train=True,
        show_progress=False,
        verbose=False,
    )
    result = trainer.run()
    # print("학습 완료! 결과:", result)

    # 평가 
    eval_envs = DummyVectorEnv([make_env])
    eval_envs.seed(SEED)
    eval_collector = Collector(policy, eval_envs)
    eval_collector.collect(n_episode=1, reset_before_collect=True)

    weekly_todo = eval_collector.env.get_env_attr("weekly_todo_final")[0]
    weekly_todo.sort(key=lambda x: (x[1], x[2]))
    return weekly_todo, user_behavior

# 1주일 분량 일정을 만드는 함수
def make_schedule(
    this_week: int,
    week_start: date,
    goocal_data: List[Dict],
    sleep_data: List[Dict],
    user_execution_data: Dict[int, List[Dict]],
    prev_behavior=None,
    user_cleaning_status=None,
    user_feedback=None,
):
    global cleaning_tasks

    goocal_data.extend(sleep_data)
    fixed_schedules = preprocessing_schedules(week_start, goocal_data)

    # 초기 청소 일정
    initial_schedules = generate_initial_schedule(cleaning_tasks, user_cleaning_status, fixed_schedules)

    cleaning_schedules, user_behavior = [], None
    if this_week > 1: # 1주치 이상의 사용자 청소 기록 데이터가 쌓이면 강화학습에 필요한 데이터를 만들고 학습을 수행한다
        cleaning_schedules, user_behavior = update_schedule(
            fixed_schedules=fixed_schedules,
            cleaning_schedules=initial_schedules,
            cleaning_tasks=cleaning_tasks,
            user_execution_data=user_execution_data,
            this_week=this_week,
            prev_behavior=prev_behavior,
            user_cleaning_status=user_cleaning_status,
            user_feedback=user_feedback,
            log_dir=None,
            SEED=42,
        )
    else:
        cleaning_schedules = initial_schedules.copy()

    prev_behavior = user_behavior

    return initial_schedules, cleaning_schedules, prev_behavior