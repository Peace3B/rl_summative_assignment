"""
main.py — Entry point for SponsorshipCaseManager RL Agent
==========================================================
Loads the best-performing saved model and runs a full episode
with real-time terminal output (and pygame if available).

Usage:
    python main.py --algo ppo           # run best PPO agent
    python main.py --algo dqn           # run best DQN agent
    python main.py --algo a2c
    python main.py --algo reinforce
    python main.py --random             # random baseline (no model)
"""

import os, sys, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment.custom_env import SponsorshipCaseManagerEnv
import gymnasium as gym

gym.register(
    id="SponsorshipCaseManager-v0",
    entry_point="environment.custom_env:SponsorshipCaseManagerEnv",
)

MODEL_PATHS = {
    "dqn":      "models/dqn/best_model/best_model",
    "ppo":      "models/pg/ppo_best/best_model",
    "a2c":      "models/pg/a2c_best/best_model",
    "reinforce": None,   # loaded separately via torch
}


def load_model(algo: str):
    from training.dqn_training import FlatDQNWrapper
    if algo == "dqn":
        from stable_baselines3 import DQN
        env  = FlatDQNWrapper(SponsorshipCaseManagerEnv())
        model = DQN.load(MODEL_PATHS["dqn"], env=env)
        return model, env, "sb3"
    elif algo == "ppo":
        from stable_baselines3 import PPO
        env   = SponsorshipCaseManagerEnv()
        model = PPO.load(MODEL_PATHS["ppo"], env=env)
        return model, env, "sb3"
    elif algo == "a2c":
        from stable_baselines3 import A2C
        env   = SponsorshipCaseManagerEnv()
        model = A2C.load(MODEL_PATHS["a2c"], env=env)
        return model, env, "sb3"
    elif algo == "reinforce":
        import torch
        from training.pg_training import PolicyNet
        env  = SponsorshipCaseManagerEnv()
        obs_dim = env.observation_space.shape[0]
        n_stu   = env.action_space.nvec.shape[0]
        n_act   = int(env.action_space.nvec[0])
        model   = PolicyNet(obs_dim, n_stu, n_act)
        # Load run 7 (best episodes)
        path = "models/pg/reinforce_run7.pt"
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model, env, "torch"
    else:
        raise ValueError(f"Unknown algo: {algo}")


def run_episode(model, env, mode="sb3", render=True, delay=0.1):
    """Run one full episode and return total reward."""
    import torch
    obs, _ = env.reset()
    total_reward = 0.0
    step = 0

    while True:
        if mode == "sb3":
            action, _ = model.predict(obs, deterministic=True)
        elif mode == "torch":
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            logits_list = model(obs_t)
            action = []
            for logits in logits_list:
                a = torch.argmax(logits, dim=-1).item()
                action.append(a)
            action = np.array(action)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if render:
            env.render()
            time.sleep(delay)

        if terminated or truncated:
            break

    return total_reward


def random_demo(n_steps=10, render=True):
    """Demonstration with random actions – no model, just visualization."""
    env = SponsorshipCaseManagerEnv()
    obs, _ = env.reset()
    print("\n" + "="*60)
    print("  RANDOM ACTION DEMO – SponsorshipCaseManager-v0")
    print("="*60)
    total = 0.0
    for step in range(n_steps):
        action = env.action_space.sample()
        obs, r, done, truncated, _ = env.step(action)
        total += r
        if render:
            env.render()
            time.sleep(0.05)
        if done or truncated:
            break
    print(f"\n[Demo] {step+1} steps | Total reward: {total:.2f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ANLM RL Agent Runner")
    parser.add_argument("--algo",   default="ppo",
                        choices=["dqn","ppo","a2c","reinforce"])
    parser.add_argument("--random", action="store_true",
                        help="Run random-action demo instead of a trained model")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--delay", type=float, default=0.05)
    args = parser.parse_args()

    if args.random:
        random_demo(n_steps=26, render=not args.no_render)
    else:
        print(f"\nLoading {args.algo.upper()} model …")
        try:
            model, env, mode = load_model(args.algo)
        except FileNotFoundError:
            print(f"[WARN] No saved model found for {args.algo}. Run training first.")
            print("Falling back to random demo.")
            random_demo(n_steps=26, render=not args.no_render)
            sys.exit(0)

        reward = run_episode(model, env, mode=mode,
                             render=not args.no_render,
                             delay=args.delay)
        print(f"\n[main] Episode complete | Total reward: {reward:.2f}")
        env.close()
