"""
pg_training.py
==============
Trains three policy-gradient agents on SponsorshipCaseManager-v0:
    • REINFORCE  (custom implementation, SB3 does not include REINFORCE natively)
    • PPO        (Stable Baselines 3)
    • A2C        (Stable Baselines 3)

Usage:
    python training/pg_training.py --algo [reinforce|ppo|a2c|all]
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from environment.custom_env import SponsorshipCaseManagerEnv

gym.register(
    id="SponsorshipCaseManager-v0",
    entry_point="environment.custom_env:SponsorshipCaseManagerEnv",
)


def make_env():
    env = SponsorshipCaseManagerEnv()
    return Monitor(env)


# ===========================================================================
# REINFORCE (vanilla policy gradient) — custom PyTorch implementation
# ===========================================================================

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_students, n_actions_per_student):
        super().__init__()
        self.n_students = n_students
        self.n_actions  = n_actions_per_student
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),    nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(128, n_actions_per_student) for _ in range(n_students)
        ])

    def forward(self, x):
        h = self.backbone(x)
        return [head(h) for head in self.heads]   # list of logit tensors


REINFORCE_EXPERIMENTS = [
    # lr,    gamma, entropy_coef, n_episodes
    (1e-3,   0.99,  0.01,  300),   # run 1  baseline
    (5e-4,   0.99,  0.01,  300),   # run 2
    (2e-3,   0.99,  0.01,  300),   # run 3
    (1e-3,   0.95,  0.01,  300),   # run 4
    (1e-3,   0.99,  0.05,  300),   # run 5  high entropy
    (1e-3,   0.99,  0.00,  300),   # run 6  no entropy
    (1e-3,   0.99,  0.01,  500),   # run 7  more episodes
    (1e-3,   0.90,  0.01,  300),   # run 8  low gamma
    (3e-3,   0.99,  0.02,  400),   # run 9
    (1e-3,   0.99,  0.01,  300),   # run 10 with baseline (reward normalisation)
]


def train_reinforce(run_idx=0):
    lr, gamma, ent_coef, n_eps = REINFORCE_EXPERIMENTS[run_idx]
    env = SponsorshipCaseManagerEnv()
    obs_dim  = env.observation_space.shape[0]
    n_stu    = env.action_space.nvec.shape[0]
    n_act    = int(env.action_space.nvec[0])

    policy   = PolicyNet(obs_dim, n_stu, n_act)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_rewards = []
    os.makedirs("models/pg", exist_ok=True)

    for ep in range(n_eps):
        obs, _ = env.reset()
        log_probs, rewards_ep = [], []
        done = False

        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            logits_list = policy(obs_t)
            action = []
            ep_log_prob = torch.tensor(0.0)
            for logits in logits_list:
                dist = torch.distributions.Categorical(logits=logits)
                a    = dist.sample()
                ep_log_prob = ep_log_prob + dist.log_prob(a)
                action.append(a.item())
            log_probs.append(ep_log_prob)
            obs, r, terminated, truncated, _ = env.step(np.array(action))
            rewards_ep.append(r)
            done = terminated or truncated

        # Compute discounted returns
        G, returns = 0.0, []
        for r in reversed(rewards_ep):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        if run_idx == 9:   # baseline: normalise returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy loss + entropy bonus
        loss = torch.tensor(0.0, requires_grad=True)
        for lp, Gt in zip(log_probs, returns):
            loss = loss - lp * Gt
        # Entropy regularisation (using mean logits entropy approximation)
        if ent_coef > 0:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            logits_list = policy(obs_t)
            for logits in logits_list:
                dist  = torch.distributions.Categorical(logits=logits)
                loss  = loss - ent_coef * dist.entropy().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ep_total = sum(rewards_ep)
        episode_rewards.append(ep_total)
        if (ep + 1) % 50 == 0:
            print(f"[REINFORCE run {run_idx+1}] ep={ep+1}  reward={ep_total:.2f}  "
                  f"mean_50={np.mean(episode_rewards[-50:]):.2f}")

    path = f"models/pg/reinforce_run{run_idx+1}.pt"
    torch.save(policy.state_dict(), path)
    print(f"[REINFORCE] Saved → {path}")
    env.close()
    return episode_rewards


# ===========================================================================
# PPO Experiments (10 runs)
# ===========================================================================

PPO_EXPERIMENTS = [
    # lr,    gamma, n_steps, ent_coef, clip_range, batch
    (3e-4,  0.99,   2048,   0.00,  0.20,  64),   # run 1  default
    (1e-4,  0.99,   2048,   0.00,  0.20,  64),   # run 2
    (3e-4,  0.95,   2048,   0.00,  0.20,  64),   # run 3
    (3e-4,  0.99,   4096,   0.00,  0.20,  64),   # run 4  larger rollout
    (3e-4,  0.99,   2048,   0.01,  0.20,  64),   # run 5  entropy
    (3e-4,  0.99,   2048,   0.00,  0.10,  64),   # run 6  tight clip
    (3e-4,  0.99,   2048,   0.00,  0.30,  64),   # run 7  loose clip
    (3e-4,  0.99,   2048,   0.00,  0.20, 128),   # run 8  large batch
    (5e-4,  0.99,   1024,   0.01,  0.20,  64),   # run 9
    (3e-4,  0.99,   2048,   0.02,  0.20, 256),   # run 10
]

TOTAL_TIMESTEPS = 200_000


def train_ppo(run_idx=0):
    lr, gamma, n_steps, ent_coef, clip, batch = PPO_EXPERIMENTS[run_idx]
    env  = make_vec_env(make_env, n_envs=4)
    eval_env = make_vec_env(make_env, n_envs=1)
    os.makedirs("models/pg", exist_ok=True)

    model = PPO(
        policy         = "MlpPolicy",
        env            = env,
        learning_rate  = lr,
        gamma          = gamma,
        n_steps        = n_steps,
        ent_coef       = ent_coef,
        clip_range     = clip,
        batch_size     = batch,
        verbose        = 1,
    )

    eval_cb = EvalCallback(eval_env,
                           best_model_save_path="models/pg/ppo_best",
                           log_path="models/pg/ppo_logs",
                           eval_freq=5000, verbose=1)

    print(f"\n[PPO] Run {run_idx+1}/10  lr={lr}  gamma={gamma}  n_steps={n_steps}  "
          f"ent={ent_coef}  clip={clip}  batch={batch}")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_cb)
    path = f"models/pg/ppo_run{run_idx+1}"
    model.save(path)
    print(f"[PPO] Saved → {path}")
    return model


# ===========================================================================
# A2C Experiments (10 runs)
# ===========================================================================

A2C_EXPERIMENTS = [
    # lr,    gamma, n_steps, ent_coef, vf_coef, max_grad_norm
    (7e-4,  0.99,    5,    0.00,  0.50,  0.5),   # run 1  default
    (3e-4,  0.99,    5,    0.00,  0.50,  0.5),   # run 2
    (7e-4,  0.95,    5,    0.00,  0.50,  0.5),   # run 3
    (7e-4,  0.99,   10,    0.00,  0.50,  0.5),   # run 4  longer rollout
    (7e-4,  0.99,    5,    0.01,  0.50,  0.5),   # run 5  entropy
    (7e-4,  0.99,    5,    0.00,  0.25,  0.5),   # run 6  lower vf_coef
    (7e-4,  0.99,    5,    0.00,  0.75,  0.5),   # run 7  higher vf_coef
    (7e-4,  0.99,    5,    0.00,  0.50,  0.2),   # run 8  tight grad norm
    (1e-3,  0.99,    5,    0.01,  0.50,  0.5),   # run 9
    (7e-4,  0.99,   20,    0.02,  0.50,  1.0),   # run 10 long rollout
]


def train_a2c(run_idx=0):
    lr, gamma, n_steps, ent_coef, vf_coef, grad_norm = A2C_EXPERIMENTS[run_idx]
    env  = make_vec_env(make_env, n_envs=4)
    eval_env = make_vec_env(make_env, n_envs=1)
    os.makedirs("models/pg", exist_ok=True)

    model = A2C(
        policy         = "MlpPolicy",
        env            = env,
        learning_rate  = lr,
        gamma          = gamma,
        n_steps        = n_steps,
        ent_coef       = ent_coef,
        vf_coef        = vf_coef,
        max_grad_norm  = grad_norm,
        verbose        = 1,
    )

    eval_cb = EvalCallback(eval_env,
                           best_model_save_path="models/pg/a2c_best",
                           log_path="models/pg/a2c_logs",
                           eval_freq=5000, verbose=1)

    print(f"\n[A2C] Run {run_idx+1}/10  lr={lr}  gamma={gamma}  n_steps={n_steps}  "
          f"ent={ent_coef}  vf={vf_coef}  grad_norm={grad_norm}")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_cb)
    path = f"models/pg/a2c_run{run_idx+1}"
    model.save(path)
    print(f"[A2C] Saved → {path}")
    return model


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="all",
                        choices=["reinforce", "ppo", "a2c", "all"])
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    if args.algo in ("reinforce", "all"):
        for i in range(min(args.runs, len(REINFORCE_EXPERIMENTS))):
            train_reinforce(i)

    if args.algo in ("ppo", "all"):
        for i in range(min(args.runs, len(PPO_EXPERIMENTS))):
            train_ppo(i)

    if args.algo in ("a2c", "all"):
        for i in range(min(args.runs, len(A2C_EXPERIMENTS))):
            train_a2c(i)

    print("\n[PG Training] All runs complete.")
