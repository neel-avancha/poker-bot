import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import numpy as np
import logging as log
import time

class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name='DQN', load_model=None, env=None):
        """Initiaization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True

        self.dqn = None
        self.model = None
        self.env = env

        if load_model:
            self.load(load_model)

class QNetwork(nn.Module):
    # Using funnel architecture for the neurons. Ideally we would use start_dim_hidden_layer. 
    # input_dim = observation vector -> 512 -> 256 -> 128 -> output dimensions.
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)


        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x
    
class PrioritizedReplayBuffer:
    # Running a priortized replay buffer in order to maximize states which are not 
    # visited often but can be high stakes. 
    def __init__(self, capacity=100000, alpha=0.6):
        # Circular implementation similar to Exercise 6: FIFO approach. 
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done):
        # When we add a new transition to the buffer, we don't know the TD Error, 
        # and the priority as a result. So we assign it the maximum current priority
        # so that each new experience can be sampled soon. Avoids cold start problem. 
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
    # Sample only valid experiences.
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # Convert priorities to a valid probability distribution
        probs = prios ** self.alpha
        probs /= probs.sum()

        # Gather the indices using weighted random choice. 
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Importance sampling to eliminate the bias presented from weighted sampling. 
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights
        )

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

class DoubleDQNAgent:
    # Wanted to experiment with a Boltzmann policy stategy, versus epsilon-greedy. 
    # The Boltzmann should balance out exploration better as well, and since the game is very 
    # unknown, we can't definitively make a decision at most states within our game. 
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, tau=1e-3, device='cpu',
                 boltzmann_tau=1.0, clip=(-500, 500), use_boltzmann=False):
        self.q_net = QNetwork(state_dim, action_dim).to(device)
        self.q_target = QNetwork(state_dim, action_dim).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.memory = PrioritizedReplayBuffer()
        self.gamma = gamma
        self.tau = tau  # soft update
        self.device = device
        self.action_dim = action_dim

        # Boltzmann policy settings
        self.use_boltzmann = use_boltzmann
        self.boltzmann_tau = boltzmann_tau
        self.clip = clip

    def select_action(self, state, legal_moves, epsilon):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor).squeeze(0).cpu().numpy()

        # Mask out illegal moves
        mask = np.full(self.action_dim, -1e10)
        mask[legal_moves] = 0
        masked_q_values = q_values + mask

        # --- Boltzmann Policy ---
        if self.use_boltzmann:
            clipped_q = np.clip(masked_q_values / self.boltzmann_tau, self.clip[0], self.clip[1])
            exp_values = np.exp(clipped_q)
            probs = exp_values / np.sum(exp_values)
            action = np.random.choice(range(self.action_dim), p=probs)
            log.info(f"Chosen action by Boltzmann: {action} | probs: {probs}")
            return action

        # --- ε-greedy Policy ---
        if np.random.rand() < epsilon:
            return random.choice(legal_moves)
        return int(np.argmax(masked_q_values))
