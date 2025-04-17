import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import numpy as np
import logging as log
import time
from tqdm import trange

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

    
    def initiate_agent(self, env):
        """ Initialize the DQN agent with the environment"""
        state_dim = env.observation_space.shape[0]  # Assuming observation space is a vector
        action_dim = env.action_space.n
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create the DQN agent
        self.dqn = DoubleDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=1e-3,
            gamma=0.99,
            tau=1e-3,
            device=device,
            boltzmann_tau=1.0,
            use_boltzmann=True  # Set to False if you prefer epsilon-greedy
        )
        
        return self.dqn
    
    def action(self, action_space, observation, info):
        """Mandatory method that calculates the move based on the observation array and the action space."""
        if self.dqn is None:
            # If agent not initialized, return random action
            return random.choice(list(action_space))
        
        # Convert action_space to a list of indices
        legal_moves = info['legal_moves']
        
        # Select action (with zero exploration during gameplay)
        action = self.dqn.select_action(observation, legal_moves, epsilon=0.05)
        
        return action
    
    def save(self, filename):
        """Save the model to disk"""
        if self.dqn is not None:
            # Save q_net parameters
            torch.save(self.dqn.q_net.state_dict(), f"{filename}_q_net.pth")
            # Save target network parameters
            torch.save(self.dqn.q_target.state_dict(), f"{filename}_q_target.pth")
            # Save model metadata (architecture info)
            import json
            metadata = {
                "state_dim": self.dqn.q_net.fc1.in_features,
                "action_dim": self.dqn.q_net.fc4.out_features,
                "boltzmann_tau": self.dqn.boltzmann_tau,
                "use_boltzmann": self.dqn.use_boltzmann
            }
            with open(f"{filename}_metadata.json", "w") as f:
                json.dump(metadata, f)
            
            print(f"Model saved to {filename}")


    def load(self, filename):
        """Load the model from disk"""
        import json
        
        # Load metadata
        with open(f"{filename}_metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create new DQN agent with the same architecture
        self.dqn = DoubleDQNAgent(
            state_dim=metadata["state_dim"],
            action_dim=metadata["action_dim"],
            device=device,
            boltzmann_tau=metadata.get("boltzmann_tau", 1.0),
            use_boltzmann=metadata.get("use_boltzmann", True)
        )

        # Load network parameters
        self.dqn.q_net.load_state_dict(torch.load(f"{filename}_q_net.pth", map_location=device))
        self.dqn.q_target.load_state_dict(torch.load(f"{filename}_q_target.pth", map_location=device))
        
        print(f"Model loaded from {filename}")

    def train(self, env_name):
        """Train the DQN agent"""
        if self.dqn is None:
            self.initiate_agent(self.env)
        
        # Setup tensorboard
        from torch.utils.tensorboard import SummaryWriter
        timestr = time.strftime("%Y%m%d-%H%M%S") + "_" + str(env_name)
        writer = SummaryWriter(log_dir=f'./Graph/{timestr}')
        
        # Training parameters
        num_episodes = 1000
        batch_size = 64
        epsilon_start = 1.0
        epsilon_end = 0.05
        epsilon_decay = 0.995
        
        # Training loop using the existing train_dqn_agent function
        rewards = train_dqn_agent(
            agent=self.dqn,
            env=self.env,
            num_episodes=num_episodes,
            batch_size=batch_size,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay
        )
        
        # Log rewards
        for i, reward in enumerate(rewards):
            writer.add_scalar('Reward/train', reward, i)
        
        # Save the model
        self.save(f"dqn_{env_name}")
        
        # Close tensorboard writer
        writer.close()
        
        return rewards
    
    def play(self, nb_episodes=5, render=False):
        """Let the agent play in the environment without training"""
        if self.dqn is None:
            raise ValueError("Agent must be initialized or loaded before playing")
        
        for episode in range(nb_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            step = 0
            
            while not done:
                if render:
                    self.env.render()
                
                # Get legal moves
                legal_moves = self.env.info['legal_moves']
                
                # Select action with no exploration
                action = self.dqn.select_action(state, legal_moves, epsilon=0)
                
                # Take action
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                step += 1
            
            print(f"Episode {episode+1}/{nb_episodes}: Total Reward: {total_reward}, Steps: {step}")

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

        # Extract integer values from enum objects
        legal_actions_indices = [action.value for action in legal_moves]
        
        # Create mask for illegal actions
        mask = np.full(self.action_dim, -np.inf)  # Use -infinity instead of a large negative number
        mask[legal_actions_indices] = 0
        masked_q_values = q_values + mask

        # --- Boltzmann Policy ---
        if self.use_boltzmann:
            # Only consider legal actions to avoid numerical issues
            legal_q_values = np.array([q_values[idx] for idx in legal_actions_indices])
            
            # Apply temperature scaling
            scaled_q = legal_q_values / self.boltzmann_tau
            
            # Subtract max value for numerical stability
            scaled_q = scaled_q - np.max(scaled_q)
            
            # Apply clipping to prevent extreme values
            clipped_q = np.clip(scaled_q, self.clip[0], self.clip[1])
            
            # Calculate probabilities with improved numerical stability
            exp_values = np.exp(clipped_q)
            probs = exp_values / np.sum(exp_values)
            
            # Check for NaN values and fix if necessary
            if np.isnan(probs).any():
                # Fallback to uniform distribution over legal actions
                probs = np.ones(len(legal_actions_indices)) / len(legal_actions_indices)
            
            # Select action from legal actions only
            action_idx = np.random.choice(len(legal_actions_indices), p=probs)
            selected_action = legal_moves[action_idx]
            
            log.info(f"Chosen action by Boltzmann: {selected_action} | probs: {probs}")
            return selected_action

        # --- ε-greedy Policy ---
        if np.random.rand() < epsilon:
            return random.choice(legal_moves)
        
        # Find best legal action
        best_legal_action_idx = np.argmax([q_values[idx] for idx in legal_actions_indices])
        return legal_moves[best_legal_action_idx]

def train_dqn_agent(
    agent, env,
    num_episodes=1000,
    batch_size=64,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    beta_start=0.4,
    beta_increment=1e-4,
    save_interval=100
):
    """
    Train a Double DQN agent with PER and Boltzmann/ε-greedy policy.

    Parameters:
        agent: DoubleDQNAgent
        env: poker environment
    """
    epsilon = epsilon_start
    beta = beta_start
    all_rewards = []
    avg_losses = []
    
    for episode in trange(num_episodes, desc="Training"):
        state = env.reset()
        done = False
        total_reward = 0
        episode_losses = []

        while not done:
            legal_moves = env.info['legal_moves']
            action = agent.select_action(state, legal_moves, epsilon)
            next_state, reward, done, info = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # train
            if len(agent.memory) >= batch_size:
                (states, actions, rewards,
                 next_states, dones,
                 indices, weights) = agent.memory.sample(batch_size, beta)

                # tensors
                states = torch.tensor(states, dtype=torch.float32).to(agent.device)
                actions = torch.tensor(actions).unsqueeze(1).to(agent.device)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(agent.device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(agent.device)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(agent.device)
                weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(agent.device)

                # q-values
                q_values = agent.q_net(states).gather(1, actions)
                next_actions = agent.q_net(next_states).argmax(1, keepdim=True)
                next_q_values = agent.q_target(next_states).gather(1, next_actions)
                target_q = rewards + agent.gamma * (1 - dones) * next_q_values.detach()

                td_errors = q_values - target_q
                loss = (td_errors.pow(2) * weights).mean()
                episode_losses.append(loss.item())

                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()

                # update priorities
                new_priorities = td_errors.abs().detach().cpu().numpy().squeeze() + 1e-6
                agent.memory.update_priorities(indices, new_priorities)

                # soft update
                for target_param, param in zip(agent.q_target.parameters(), agent.q_net.parameters()):
                    target_param.data.copy_(
                        agent.tau * param.data + (1.0 - agent.tau) * target_param.data
                    )

                beta = min(1.0, beta + beta_increment)

        all_rewards.append(total_reward)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        avg_losses.append(avg_loss)
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Log progress
        if episode % 10 == 0:
            log.info(f"Episode {episode}/{num_episodes} | Reward: {total_reward:.2f} | "
                     f"Avg Loss: {avg_loss:.4f} | Epsilon: {epsilon:.2f}")
    
    return all_rewards