"""Entry point"""
import argparse
import os
import random
import numpy as np
import torch
from config import EnvConfig
from train_hchrl import train_hchrl, evaluate_hchrl, HCHRLAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--tasks", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model_path", type=str, default="outputs/model_hchrl_best.pth")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    cfg = EnvConfig()
    device = torch.device(args.device)
    set_seed(cfg.random_seed)
    if args.mode == "train":
        agent, history = train_hchrl(num_episodes=args.episodes, max_steps=args.steps,
                                     tasks_per_step=args.tasks, eval_interval=50,
                                     save_path='outputs', device=device)
        results = evaluate_hchrl(agent, num_episodes=50, use_cem_topsis=True)
        print(f"SR={results['success_rate']*100:.1f}% R={results['return_mean']:.2f}")
    else:
        if os.path.exists(args.model_path):
            agent = HCHRLAgent(num_tasks=args.tasks, hidden_dim=128, num_clusters=3, device=device)
            agent.load(args.model_path)
            results = evaluate_hchrl(agent, num_episodes=50, use_cem_topsis=True)
            print(f"SR={results['success_rate']*100:.1f}% R={results['return_mean']:.2f}")
        else:
            print(f"Model not found: {args.model_path}")


if __name__ == "__main__":
    main()
