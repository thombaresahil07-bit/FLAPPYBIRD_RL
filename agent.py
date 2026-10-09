import argparse
import itertools
import os
import random
import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from dqn import DQN
from experience_replay import ReplayMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class Agent:

    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha = float(params.get("alpha", 0.0005))
        self.gamma = float(params.get("gamma", 0.99))

        self.epsilon_init = float(params.get("epsilon_init", 1.0))
        self.epsilon_min = float(params.get("epsilon_min", 0.01))
        self.epsilon_decay = float(params.get("epsilon_decay", 0.9992))

        self.replay_memory_size = int(params.get("replay_memory_size", 100000))
        self.mini_batch_size = int(params.get("mini_batch_size", 64))
        self.warmup_steps = int(params.get("warmup_steps", 1000))

        self.reward_threshold = float(params.get("reward_threshold", 10000.0))
        self.network_sync_rate = int(params.get("network_sync_rate", 500))

        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")

    def run(self, is_training=True, render=False):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            target_dqn.eval()

            steps = 0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
            best_reward = float("-inf")
            print(">>> Starting Training Mode... (Press Ctrl+C to stop)")
        else:
            if not os.path.exists(self.MODEL_FILE):
                print(f"Error: Model file {self.MODEL_FILE} not found!")
                print("Run training first using: python agent.py default --train")
                return
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            policy_dqn.eval()
            epsilon = 0.0
            print(">>> Starting Testing Mode with loaded model...")

        for episode in itertools.count():
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32)

            episodes_reward = 0.0
            terminated = False

            while not terminated and episodes_reward < self.reward_threshold:
                # Epsilon-greedy action selection
                if is_training and random.random() < epsilon:
                    # Flap with 15% probability during exploration
                    action = 1 if random.random() < 0.15 else 0
                else:
                    with torch.no_grad():
                        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                        action = policy_dqn(state_t).argmax(dim=1).item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Reward shaping: heavily penalize crashing so it learns to survive
                shaped_reward = reward
                if done:
                    shaped_reward = -5.0

                episodes_reward += reward
                next_state = np.array(next_state, dtype=np.float32)

                if is_training:
                    memory.append((state, action, next_state, shaped_reward, done))
                    steps += 1

                    # Only train once enough transitions are stored
                    if len(memory) >= self.warmup_steps and len(memory) >= self.mini_batch_size:
                        mini_batch = memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch, policy_dqn, target_dqn)

                        # Sync target network
                        if steps % self.network_sync_rate == 0:
                            target_dqn.load_state_dict(policy_dqn.state_dict())

                state = next_state
                terminated = done

            print(
                f"Episode {episode + 1:5d} | Steps: {steps if is_training else 0:7d} | "
                f"Reward: {episodes_reward:8.2f} | Epsilon: {epsilon:.4f}"
            )

            if is_training:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                # Save when a new milestone is reached
                if episodes_reward > best_reward and len(memory) >= self.warmup_steps:
                    best_reward = episodes_reward
                    with open(self.LOG_FILE, "a") as f:
                        f.write(f"Best reward: {best_reward:.2f} at Episode {episode + 1}\n")
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    print(f"--> Saved new best model with reward: {best_reward:.2f}")

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        terminations_t = torch.tensor(terminations, dtype=torch.float32, device=device).unsqueeze(1)

        with torch.no_grad():
            next_q_values = target_dqn(next_states_t).max(dim=1, keepdim=True)[0]
            target_q = rewards_t + (1.0 - terminations_t) * self.gamma * next_q_values

        current_q = policy_dqn(states_t).gather(dim=1, index=actions_t)
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=5.0)
        self.optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test DQN on Flappy Bird.")
    parser.add_argument("hyperparameters", help="Key for parameters in parameters.yaml")
    parser.add_argument("--train", help="Run in training mode", action="store_true")
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True, render=False)
    else:
        dql.run(is_training=False, render=True)