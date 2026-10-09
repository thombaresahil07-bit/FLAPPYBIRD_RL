# Flappy Bird DQN

Built this small project to get hands-on with Reinforcement Learning. It trains an agent to play Flappy Bird using Deep Q-Networks (DQN) in PyTorch.

### What it does:
- Uses DQN with an $\epsilon$-greedy strategy for action selection.
- Implements Experience Replay to stabilize training and break sample correlation.
- Hyperparameters are kept in `parameters.yaml` so I don't have to hardcode them.

### Tech Stack
- Python, PyTorch, PyYAML

### How to Run
```bash
git clone [https://github.com/thombaresahil07-bit/FLAPPYBIRD_RL.git](https://github.com/thombaresahil07-bit/FLAPPYBIRD_RL.git)
cd FLAPPYBIRD_RL
pip install torch pyyaml
python agent.py
