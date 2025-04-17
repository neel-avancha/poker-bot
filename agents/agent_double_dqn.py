import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import numpy as np
import logging as log
import time
import os
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
        print("Reached post save clause")
        
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
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        
        # Better weight initialization
        nn.init.kaiming_normal_(self.fc1.weight)  # He initialization
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.uniform_(self.fc4.weight, -0.003, 0.003)  # Very small initialization for final layer
        nn.init.constant_(self.fc4.bias, 0)  # Zero bias initialization

    def forward(self, x):
        # Add checks for NaN or Inf values during forward pass
        if torch.isnan(x).any() or torch.isinf(x).any():
            log.error(f"Input to network contains NaN or Inf: {x}")
            # Reset the problematic values to zero
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        
        # Add check for NaN outputs
        if torch.isnan(x).any():
            log.error(f"Network output contains NaN: {x}")
            x = torch.nan_to_num(x, nan=0.0)
        
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
            
        # Make sure priorities are positive, with a small epsilon to ensure no zeros
        prios = np.maximum(prios, 1e-8)
        
        # Convert priorities to a valid probability distribution
        probs = prios ** self.alpha
        
        # Avoid division by zero
        probs_sum = np.sum(probs)
        if probs_sum <= 0 or np.isnan(probs_sum):
            # Fallback to uniform distribution if we have invalid priorities
            probs = np.ones_like(prios) / len(prios)
        else:
            probs = probs / probs_sum
        
        # Double-check for NaN values and replace with uniform distribution if needed
        if np.isnan(probs).any():
            probs = np.ones(len(self.buffer)) / len(self.buffer)
        
        # Gather the indices using weighted random choice. 
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Importance sampling to eliminate the bias presented from weighted sampling. 
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        
        # Normalize weights to prevent extreme values
        max_weight = np.max(weights)
        if max_weight > 0 and not np.isnan(max_weight) and not np.isinf(max_weight):
            weights = weights / max_weight
        else:
            # Fallback to uniform weights
            weights = np.ones_like(weights)
        
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


    def save_checkpoint(self, path_prefix):
        """Save Q-network and target Q-network parameters along with a metadata file."""
        torch.save(self.q_net.state_dict(), f"{path_prefix}_q_net.pth")
        torch.save(self.q_target.state_dict(), f"{path_prefix}_q_target.pth")

        # Optionally save metadata like architecture info
        import json
        metadata = {
            "state_dim": self.q_net.fc1.in_features,
            "action_dim": self.q_net.fc4.out_features,
            "boltzmann_tau": self.boltzmann_tau,
            "use_boltzmann": self.use_boltzmann
        }
        with open(f"{path_prefix}_metadata.json", "w") as f:
            json.dump(metadata, f)

        print(f"Checkpoint saved to {path_prefix}_*.pth")


    def select_action(self, state, legal_moves, epsilon):
        if not legal_moves:
            return 0
        
        # Convert state to tensor and get Q-values
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Check for NaN in state
        if torch.isnan(state_tensor).any():
            log.error(f"NaN detected in state: {state}")
            state_tensor = torch.nan_to_num(state_tensor, nan=0.0)
        
        with torch.no_grad():
            q_values = self.q_net(state_tensor).squeeze(0).cpu().numpy()
        
        # Check for NaN in Q-values
        if np.isnan(q_values).any():
            log.error(f"NaN detected in Q-values: {q_values}")
            q_values = np.nan_to_num(q_values, nan=0.0)
        
        # Debug info
        log.debug(f"Q-values: {q_values}")
        
        # Extract integer values from enum objects
        legal_actions_indices = [action.value for action in legal_moves]
        
        # --- Boltzmann Policy with improved numerical stability ---
        if self.use_boltzmann:
            # Only consider legal actions
            legal_q_values = np.array([q_values[idx] for idx in legal_actions_indices])
            
            # Apply temperature scaling with higher temperature initially
            # Start with higher temperature (e.g., 5.0) and anneal it down over time
            temperature = max(1.0, self.boltzmann_tau)  # Never go below 1.0
            scaled_q = legal_q_values / temperature
            
            # Subtract max value for numerical stability (crucial step!)
            max_q = np.max(scaled_q)
            if np.isnan(max_q) or np.isinf(max_q):
                log.error(f"NaN/Inf detected in max_q: {max_q}")
                max_q = 0
            
            scaled_q = scaled_q - max_q
            
            # Apply stricter clipping to prevent extreme values
            clipped_q = np.clip(scaled_q, -10.0, 10.0)  # Much tighter bounds
            
            # Calculate probabilities with improved numerical stability
            exp_values = np.exp(clipped_q)
            sum_exp = np.sum(exp_values)
            
            # Check for division by zero or very small values
            if sum_exp < 1e-10 or np.isnan(sum_exp) or np.isinf(sum_exp):
                log.warning(f"Numerical instability in Boltzmann sum: {sum_exp}")
                # Fallback to uniform distribution
                probs = np.ones(len(legal_actions_indices)) / len(legal_actions_indices)
            else:
                probs = exp_values / sum_exp
            
            # Final check for NaN in probabilities
            if np.isnan(probs).any() or np.sum(probs) == 0:
                log.error(f"NaN or zero-sum detected in probs: {probs}")
                probs = np.ones(len(legal_actions_indices)) / len(legal_actions_indices)
            
            # Select action from legal actions
            try:
                action_idx = np.random.choice(len(legal_actions_indices), p=probs)
                selected_action = legal_moves[action_idx]
                log.info(f"Boltzmann selection: {selected_action} | probs: {probs}")
                return selected_action
            except Exception as e:
                log.error(f"Error in action selection: {e}, probs: {probs}")
                # Fallback to random action
                return random.choice(legal_moves)
        
        # --- ε-greedy fallback ---
        if np.random.rand() < epsilon:
            return random.choice(legal_moves)
        
        # Check for NaN or Inf in Q-values for legal actions
        legal_q_values = [q_values[idx] for idx in legal_actions_indices]
        if np.isnan(legal_q_values).any() or np.isinf(legal_q_values).any():
            log.error(f"NaN/Inf in legal Q-values: {legal_q_values}")
            return random.choice(legal_moves)
        
        # Find best legal action
        best_legal_action_idx = np.argmax(legal_q_values)
        return legal_moves[best_legal_action_idx]

def train_dqn_agent(
    agent,
    env,
    num_episodes=1000,
    batch_size=64,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    beta_start=0.4,
    beta_increment=1e-4,
    save_interval=100,
    checkpoint_dir='./checkpoints'
):
    os.makedirs(checkpoint_dir, exist_ok=True)

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
            if not legal_moves:
                log.warning("No legal moves found; skipping step")
                break

            action = agent.select_action(state, legal_moves, epsilon)
            next_state, reward, done, info = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) >= batch_size:
                (states, actions, rewards, next_states, dones, indices, weights) = agent.memory.sample(batch_size, beta)

                states = torch.tensor(states, dtype=torch.float32).to(agent.device)
                actions = torch.tensor(actions).unsqueeze(1).to(agent.device)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(agent.device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(agent.device)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(agent.device)
                weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(agent.device)

                if torch.isnan(states).any() or torch.isnan(rewards).any() or torch.isnan(next_states).any():
                    log.error("NaN detected in training inputs")
                    continue  # Skip this batch

                reward_scale = 0.01 if torch.max(rewards.abs()) > 1000 else 1.0
                scaled_rewards = rewards * reward_scale

                q_values = agent.q_net(states).gather(1, actions)
                with torch.no_grad():   
                    next_actions = agent.q_net(next_states).argmax(1, keepdim=True)
                    next_q_values = agent.q_target(next_states).gather(1, next_actions)

                    # Check for NaN in target network outputs
                    if torch.isnan(next_q_values).any():
                        log.error("NaN detected in target network outputs")


                    next_q_values = torch.zeros_like(next_q_values)
                    target_q = rewards + agent.gamma * (1 - dones) * next_q_values.detach()

                    large_value_mask = target_q.abs() > 100.0
                    if large_value_mask.any():
                        target_q[large_value_mask] = 100.0 * torch.sign(target_q[large_value_mask]) * (
                            1 + torch.log10(target_q[large_value_mask].abs() / 100.0))
                        
                td_errors = q_values - target_q
                loss = nn.functional.huber_loss(q_values, target_q, reduction='none')
                loss = (loss * weights).mean()

                if torch.isnan(loss).any():
                    log.error(f"NaN detected in loss: {loss.item()}")
                    continue  # Skip this batch

                agent.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(agent.q_net.parameters(), max_norm=1.0)

                # Check for NaN in gradients
                has_nan_grad = False
                for param in agent.q_net.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        log.error("NaN detected in gradients")
                        break
                if has_nan_grad:
                    continue  # Skip this batch

                agent.optimizer.step()

                # Update priorities with safeguards
                td_errors_abs = td_errors.abs().detach().cpu().numpy() + 1e-6
                if np.isnan(td_errors_abs).any():
                    log.error("NaN detected in TD errors")
                    td_errors_abs = np.ones_like(td_errors_abs)
                
                agent.memory.update_priorities(indices, td_errors_abs)
            
                for target_param, param in zip(agent.q_target.parameters(), agent.q_net.parameters()):
                    target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)

                beta = min(1.0, beta + beta_increment)

        all_rewards.append(total_reward)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        avg_losses.append(avg_loss)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % 10 == 0:
            log.info(f"Episode {episode}/{num_episodes} | "
                     f"Reward: {total_reward:.2f} | "
                     f"Avg Loss: {avg_loss:.4f} | "
                     f"Epsilon: {epsilon:.2f} | "
                     f"Buffer size: {len(agent.memory)}")

        if (episode + 1) % save_interval == 0:
            save_path = os.path.join(checkpoint_dir, f"episode_{episode+1}")
            agent.save_checkpoint(save_path)
            log.info(f"Saved checkpoint to {save_path}")

    return all_rewards

