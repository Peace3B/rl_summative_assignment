# rl_summative_assignment

**Reinforcement Learning Summative Assignment**  
Student: Peace Keza | African Leadership University – BSE  
Environment: `SponsorshipCaseManager-v0`  
Mission: ANLM Post-Secondary Sponsorship Programme

---

## Problem Statement

An RL agent learns to optimally manage 20 sponsored post-secondary students each week,
deciding which follow-up tasks to prioritise (reminders, letters, payments, photos, reports)
under a limited staff-hour budget, so that compliance deadlines are met and donor engagement stays high.

---

## Project Structure

```
rl_summative_assignment/
├── environment/
│   ├── custom_env.py      # Custom Gymnasium environment
│   └── rendering.py       # Matplotlib & pygame visualisation
├── training/
│   ├── dqn_training.py    # DQN (10 hyperparameter experiments)
│   └── pg_training.py     # REINFORCE, PPO (10 runs each)
├── models/
│   ├── dqn/               # Saved DQN checkpoints
│   └── pg/                # Saved PPO / REINFORCE checkpoints
├── main.py                # Run best-performing agent
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Peace3B/rl_summative_assignment.git
cd rl_summative_assignment
pip install -r requirements.txt
```

---

## Usage

### Random-action demo (no model, just visualisation)
```bash
python main.py --random
```

### Train all algorithms
```bash
python training/dqn_training.py
python training/pg_training.py --algo all
```

### Run best-performing agent
```bash
python main.py --algo ppo        # PPO (best performer)
python main.py --algo dqn
python main.py --algo reinforce
```

---

## Environment Details

| Component        | Detail |
|-----------------|--------|
| Students         | 20 per episode |
| Episode length   | 26 weeks |
| Action space     | MultiDiscrete (20 × 6) |
| Observation dim  | 202 |
| Staff budget     | 6 hours/week |
| Tasks            | SMS reminder, collect letter, process payment, upload photo, log report |

---

## Algorithms

| Algorithm | Library | Action space adapter |
|-----------|---------|----------------------|
| DQN       | SB3     | FlatDQNWrapper (Discrete) |
| REINFORCE | PyTorch (custom) | MultiDiscrete direct |
| PPO       | SB3     | MultiDiscrete direct |
