import numpy as np
import torch
from torch.distributions import Categorical, Distribution

from envs import CleaningEnv
 
class MultiCategorical(Distribution):
    arg_constraints = {}
    has_rsample = False

    def __init__(self, logits_list):
        self.categoricals = [Categorical(logits=logits) for logits in logits_list]
        batch_shape = self.categoricals[0].batch_shape
        event_shape = torch.Size([len(self.categoricals)])  
        super().__init__(batch_shape=batch_shape, event_shape=event_shape, validate_args=False)

    def sample(self):
        return torch.stack([cat.sample() for cat in self.categoricals], dim=-1)

    def log_prob(self, actions):
        actions = actions.long()
        return torch.stack(
            [cat.log_prob(actions[..., i]) for i, cat in enumerate(self.categoricals)],
            dim=-1
        ).sum(dim=-1)

    def entropy(self):
        return torch.stack([cat.entropy() for cat in self.categoricals], dim=-1).sum(dim=-1)

    @property
    def mode(self):
        return torch.stack([cat.probs.argmax(dim=-1) for cat in self.categoricals], dim=-1)

    @property
    def variance(self):
        vars_ = []
        for cat in self.categoricals:
            p = cat.probs
            mean = (torch.arange(p.shape[-1], device=p.device) * p).sum(dim=-1)
            var = ((torch.arange(p.shape[-1], device=p.device) - mean.unsqueeze(-1)) ** 2 * p).sum(dim=-1)
            vars_.append(var)
        return torch.stack(vars_, dim=-1)

    @property
    def stddev(self):
        return self.variance.sqrt()

    @property
    def probs(self): # 선택사항 
        return [cat.probs for cat in self.categoricals]

def multi_categorical_dist_fn(nvec):
    nvec = list(map(int, nvec))  
    def dist_fn(logits):
        chunks = torch.split(logits, nvec, dim=-1)
        return MultiCategorical(chunks) # 단일 분포 객체로 감싸서 반환
    return dist_fn

from datetime import date
import numpy as np

def build_slot_penalty_all_weeks(user_execution_data: dict, decay: float = 0.9) -> np.ndarray:
    slot = np.zeros(168, dtype=float)
    if not user_execution_data:
        return slot
 
    pairs = []
    for k in user_execution_data:
        try:
            pairs.append((k, date.fromisoformat(str(k))))
        except:
            pass
    if not pairs:
        return slot
    pairs.sort(key=lambda x: x[1])
    last = pairs[-1][1]
 
    for k, d in pairs:
        w = decay ** (((last - d).days) // 7)
        for e in user_execution_data.get(k, ()):
            if e.get("performed"):
                idx = int(e.get("day", -1)) * 24 + int(e.get("hour", -1))
                if 0 <= idx < 168:
                    slot[idx] += 0.25 * w

    np.clip(slot, 0.0, 1.0, out=slot)
    return slot


def create_cleaning_env_factory(
    fixed_schedules,
    cleaning_tasks,
    user_cleaning_status,
    user_behavior,
    user_execution_data,
    user_pref,
    cleaning_schedules,
    user_feedback,
):

    slot_penalties = build_slot_penalty_all_weeks(user_execution_data, decay=0.9)

    def make_env():
        return CleaningEnv(
            fixed_schedules=fixed_schedules,
            cleaning_tasks=cleaning_tasks,
            user_cleaning_status=user_cleaning_status,
            user_behavior=user_behavior,
            user_execution_data=user_execution_data,
            user_pref=user_pref,
            slot_penalties=slot_penalties,
            initial_schedules=cleaning_schedules,
            user_feedback=user_feedback,
        )
    return make_env