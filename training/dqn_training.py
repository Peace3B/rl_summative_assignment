"""
dqn_training.py
===============
Trains a Deep Q-Network (DQN) agent on the SponsorshipCaseManager-v0 environment
using Stable Baselines 3.

Usage:
    python training/dqn_training.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from environment.custom_env import SponsorshipCaseManagerEnv

# Register env
gym.register(
    id="SponsorshipCaseManager-v0",
    entry_point="environment.custom_env:SponsorshipCaseManagerEnv",
)

# ---------------------------------------------------------------------------
# NOTE: DQN requires a Discrete action space.
# We flatten the MultiDiscrete action by treating each student independently
# with a wrapper that selects actions via heuristic batching.
# For SB3 DQN we use a single-action flattened version.
# ---------------------------------------------------------------------------

class FlatDQNWrapper(gym.Wrapper):
    """
    Wraps SponsorshipCaseManagerEnv to expose a Discrete action space
    compatible with SB3 DQN.  The agent selects a (student, task) pair each step;
    remaining students get idle (0) action.
    """
    def __init__(self, env):
        super().__init__(env)
        self.n_students = env.unwrapped.action_space.nvec.shape[0]
        self.n_tasks    = int(env.unwrapped.action_space.nvec[0])   # 6
        self.action_space = gym.spaces.Discrete(self.n_students * self.n_tasks)

    def step(self, action):
        student_idx = action // self.n_tasks
        task_act    = action %  self.n_tasks
        full_action = np.zeros(self.n_students, dtype=int)
        full_action[student_idx] = task_act
        return self.env.step(full_action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def make_env():
    env = SponsorshipCaseManagerEnv()
    env = FlatDQNWrapper(env)
    env = Monitor(env)
    return env


# ---------------------------------------------------------------------------
# Hyperparameter experiments  (10 runs)
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    # lr,    gamma, buffer, batch, exploration_fraction, target_update
    (1e-3,   0.99,  10000,  64,   0.20,  500),   # run 1  – baseline
    (5e-4,   0.99,  10000,  64,   0.20,  500),   # run 2  – lower lr
    (1e-3,   0.95,  10000,  64,   0.20,  500),   # run 3  – lower gamma
    (1e-3,   0.99,  20000,  64,   0.20,  500),   # run 4  – larger buffer
    (1e-3,   0.99,  10000, 128,   0.20,  500),   # run 5  – larger batch
    (1e-3,   0.99,  10000,  64,   0.35,  500),   # run 6  – more exploration
    (1e-3,   0.99,  10000,  64,   0.10,  500),   # run 7  – less exploration
    (1e-3,   0.99,  10000,  64,   0.20, 1000),   # run 8  – slower target update
    (2e-3,   0.99,  10000,  64,   0.20,  500),   # run 9  – higher lr
    (1e-3,   0.99,  50000, 256,   0.15, 1000),   # run 10 – large buffer+batch
]

TOTAL_TIMESTEPS = 200_000
BEST_MODEL_PATH = "models/dqn/best_model"
LOG_DIR         = "models/dqn/logs"

os.makedirs("models/dqn", exist_ok=True)
os.makedirs(LOG_DIR,      exist_ok=True)


def train_dqn(run_idx=0):
    lr, gamma, buffer, batch, exp_frac, target_upd = EXPERIMENTS[run_idx]
    env  = make_vec_env(make_env, n_envs=1)
    eval_env = make_vec_env(make_env, n_envs=1)

    model = DQN(
        policy            = "MlpPolicy",
        env               = env,
        learning_rate     = lr,
        gamma             = gamma,
        buffer_size       = buffer,
        batch_size        = batch,
        exploration_fraction = exp_frac,
        target_update_interval = target_upd,
        verbose           = 1,
        tensorboard_log   = LOG_DIR,
    )

    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = BEST_MODEL_PATH,
        log_path             = LOG_DIR,
        eval_freq            = 5000,
        callback_on_new_best = stop_cb,
        verbose              = 1,
    )

    print(f"\n[DQN] Run {run_idx+1}/10  lr={lr}  gamma={gamma}  buffer={buffer}  "
          f"batch={batch}  exp={exp_frac}  target_upd={target_upd}")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_cb)
    save_path = f"models/dqn/dqn_run{run_idx+1}"
    model.save(save_path)
    print(f"[DQN] Model saved → {save_path}")
    return model


if __name__ == "__main__":
    # Train all 10 hyperparameter configurations
    for i in range(len(EXPERIMENTS)):
        train_dqn(i)
    print("\n[DQN] All runs complete.")
