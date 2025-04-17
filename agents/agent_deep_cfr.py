import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
import random
from gym_env.enums import Action, Stage

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.network(x)

class Player:
    """Deep CFR agent implementation"""
    
    def __init__(self, name='DeepCFR', load_model=None, env=None):
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True
        self.env = env
        
        # Initialize networks
        self.value_net = None
        self.policy_net = None
        self.value_optimizer = None
        self.policy_optimizer = None
        
        # CFR specific variables
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.iteration = 0
        
        if load_model:
            self.load(load_model)
    
    def initiate_agent(self, env):
        """Initialize the Deep CFR agent with the environment"""
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create networks
        self.value_net = ValueNetwork(state_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        
        # Create optimizers
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=1e-3)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        
        return self
    
    def get_state_key(self, observation, info):
        """Convert observation and info into a state key for tracking regrets"""
        # Create a unique state representation
        stage = info['community_data']['stage']
        pot = info['community_data']['community_pot']
        stack = info['player_data']['stack']
        position = info['player_data']['position']
        return f"{stage}_{pot}_{stack}_{position}"
    
    def get_action_probs(self, state_key, legal_actions):
        """Get action probabilities using the policy network"""
        if self.iteration < 100:  # Use uniform policy for first 100 iterations
            return {action: 1.0/len(legal_actions) for action in legal_actions}
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(self.current_state).unsqueeze(0)
        
        # Get probabilities from policy network
        with torch.no_grad():
            probs = self.policy_net(state_tensor).squeeze()
        
        # Create action probability dictionary
        action_probs = {}
        total_prob = 0
        for action in legal_actions:
            action_probs[action] = float(probs[action])
            total_prob += action_probs[action]
        
        # Normalize probabilities
        for action in action_probs:
            action_probs[action] /= total_prob
            
        return action_probs
    
    def update_regrets(self, state_key, action, regret):
        """Update cumulative regrets for the state-action pair"""
        self.regret_sum[state_key][action] += regret
        
        # Update strategy sum using current policy
        action_probs = self.get_action_probs(state_key, list(self.regret_sum[state_key].keys()))
        for a in action_probs:
            self.strategy_sum[state_key][a] += action_probs[a]
    
    def train_networks(self, state_batch, value_targets, policy_targets):
        """Train value and policy networks"""
        state_tensor = torch.FloatTensor(state_batch)
        value_targets = torch.FloatTensor(value_targets)
        policy_targets = torch.FloatTensor(policy_targets)
        
        # Train value network
        self.value_optimizer.zero_grad()
        value_pred = self.value_net(state_tensor)
        value_loss = nn.MSELoss()(value_pred, value_targets)
        value_loss.backward()
        self.value_optimizer.step()
        
        # Train policy network
        self.policy_optimizer.zero_grad()
        policy_pred = self.policy_net(state_tensor)
        policy_loss = nn.CrossEntropyLoss()(policy_pred, policy_targets)
        policy_loss.backward()
        self.policy_optimizer.step()
    
    def action(self, action_space, observation, info):
        """Choose action based on current state and policy"""
        self.current_state = observation
        state_key = self.get_state_key(observation, info)
        
        # Get action probabilities
        action_probs = self.get_action_probs(state_key, action_space)
        
        # Choose action using epsilon-greedy strategy
        if random.random() < max(0.1, 1.0 - self.iteration/1000):  # Exploration rate
            action = random.choice(list(action_space))
        else:
            action = max(action_probs.items(), key=lambda x: x[1])[0]
        
        # Store action for regret calculation
        self.last_action = action
        self.last_state_key = state_key
        
        return action
    
    def update(self, reward, next_observation, next_info, done):
        """Update agent after an action"""
        if not hasattr(self, 'last_action'):
            return
        
        # Calculate immediate regret
        next_state_key = self.get_state_key(next_observation, next_info)
        action_probs = self.get_action_probs(next_state_key, self.env.legal_moves)
        expected_value = sum(prob * reward for action, prob in action_probs.items())
        regret = reward - expected_value
        
        # Update regrets
        self.update_regrets(self.last_state_key, self.last_action, regret)
        
        # Train networks periodically
        if self.iteration % 100 == 0:
            # Collect batch of states and targets
            states = []
            value_targets = []
            policy_targets = []
            
            # Use accumulated regrets to create training targets
            for state_key in self.regret_sum:
                state_tensor = torch.FloatTensor(self.current_state)
                states.append(state_tensor)
                
                # Value target is the maximum regret
                max_regret = max(self.regret_sum[state_key].values())
                value_targets.append(max_regret)
                
                # Policy target is the positive regrets
                policy_target = np.zeros(len(self.env.legal_moves))
                for action, regret in self.regret_sum[state_key].items():
                    if regret > 0:
                        policy_target[action] = regret
                policy_targets.append(policy_target)
            
            if states:  # Only train if we have collected states
                self.train_networks(states, value_targets, policy_targets)
        
        self.iteration += 1
    
    def save(self, path):
        """Save the model"""
        torch.save({
            'value_net': self.value_net.state_dict(),
            'policy_net': self.policy_net.state_dict(),
            'regret_sum': dict(self.regret_sum),
            'strategy_sum': dict(self.strategy_sum),
            'iteration': self.iteration
        }, path)
    
    def load(self, path):
        """Load the model"""
        checkpoint = torch.load(path)
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.regret_sum = defaultdict(lambda: defaultdict(float), checkpoint['regret_sum'])
        self.strategy_sum = defaultdict(lambda: defaultdict(float), checkpoint['strategy_sum'])
        self.iteration = checkpoint['iteration'] 