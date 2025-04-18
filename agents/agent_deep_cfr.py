import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict, deque
import random
import logging
import time
import copy
from tqdm import trange
from gym_env.enums import Action, Stage
import math
import torch.nn.functional as F

# Set up logging
log = logging.getLogger(__name__)

class CardEmbedding(nn.Module):
    """Embedding layer for poker cards"""
    def __init__(self, embedding_dim=32):
        super(CardEmbedding, self).__init__()
        # 52 cards + 1 for unknown/empty
        self.embedding = nn.Embedding(53, embedding_dim)
        
    def forward(self, card_indices):
        return self.embedding(card_indices)

class AdvancedStateEncoder(nn.Module):
    """Encodes the poker game state into a rich representation"""
    def __init__(self, card_embedding_dim=32, action_history_dim=64, pot_dim=16):
        super(AdvancedStateEncoder, self).__init__()
        
        # Calculate dimensions explicitly
        self.card_embedding_dim = card_embedding_dim
        self.action_history_dim = action_history_dim
        self.pot_dim = pot_dim
        
        # Card embeddings
        self.card_embedding = CardEmbedding(card_embedding_dim)
        
        # Action history encoder (LSTM)
        self.action_history_encoder = nn.LSTM(
            input_size=len(Action),  # One-hot encoding of actions
            hidden_size=action_history_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Pot and stack encoder
        self.pot_encoder = nn.Sequential(
            nn.Linear(4, pot_dim),  # pot size and stack size (2 each)
            nn.ReLU(),
            nn.Linear(pot_dim, pot_dim)
        )
        
        # Position encoder
        self.position_encoder = nn.Embedding(9, 16)  # 9 possible positions (0-8)
        
        # Stage encoder
        self.stage_encoder = nn.Embedding(len(Stage), 16)
        
        # Calculate total dimension
        # 2 hole cards (2*embedding_dim) + 5 community cards (5*embedding_dim) + action history + pot + position + stage
        total_dim = (2 * card_embedding_dim) + (5 * card_embedding_dim) + action_history_dim + pot_dim + 16 + 16
        
        # Log the total dimension for debugging
        #print(f"Total dimension for state encoder: {total_dim}")
        
        # Combine all features with proper dimensions
        self.combiner = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
    def forward(self, state_dict):
        # Extract features from state dictionary
        hole_cards = state_dict['hole_cards']
        community_cards = state_dict['community_cards']
        action_history = state_dict['action_history']
        pot_size = state_dict['pot_size']
        stack_size = state_dict['stack_size']
        position = state_dict['position']
        stage = state_dict['stage']
        
        # Get batch size from hole_cards
        batch_size = hole_cards.size(0)
        
        # Process hole cards (2 cards)
        # Ensure we only use the first 2 cards for hole cards
        hole_cards = hole_cards[:, :2]
        hole_embeddings = self.card_embedding(hole_cards)
        hole_embeddings = hole_embeddings.view(batch_size, 2 * self.card_embedding_dim)
        
        # Process community cards (5 cards)
        # Ensure we only use up to 5 community cards
        community_cards = community_cards[:, :5]
        community_embeddings = self.card_embedding(community_cards)
        community_embeddings = community_embeddings.view(batch_size, 5 * self.card_embedding_dim)
        
        # Process action history
        # Ensure action_history has the correct shape (batch_size, seq_length, input_size)
        if len(action_history.shape) == 2:
            action_history = action_history.unsqueeze(0)  # Add batch dimension if missing
        
        # Ensure batch dimension matches expected size
        if action_history.size(0) != batch_size:
            # Repeat the action history tensor for each item in the batch
            action_history = action_history.repeat(batch_size, 1, 1)
        
        # Debug information
        #print(f"Action history shape: {action_history.shape}")
        
        action_history_output, _ = self.action_history_encoder(action_history)
        action_history_encoding = action_history_output[:, -1, :]
        
        # Debug information
        #print(f"Action history encoding shape: {action_history_encoding.shape}")
            
        # Process pot and stack - ensure consistent dimensions
        # Reshape pot_size and stack_size to ensure they're both [batch_size, 2]
        if len(pot_size.shape) == 3:
            pot_size = pot_size.squeeze(1)  # Remove middle dimension if it exists
        if len(stack_size.shape) == 3:
            stack_size = stack_size.squeeze(1)  # Remove middle dimension if it exists
            
        # Ensure both tensors are 2D with proper batch dimension
        if len(pot_size.shape) == 1:
            pot_size = pot_size.unsqueeze(0).repeat(batch_size, 1)
        if len(stack_size.shape) == 1:
            stack_size = stack_size.unsqueeze(0).repeat(batch_size, 1)
            
        # Ensure both tensors have exactly 2 columns
        if pot_size.shape[1] != 2:
            if pot_size.shape[1] > 2:
                pot_size = pot_size[:, :2]  # Take only first 2 columns
            else:
                pot_size = torch.cat([pot_size, torch.zeros(batch_size, 2 - pot_size.shape[1], device=pot_size.device)], dim=1)
        
        if stack_size.shape[1] != 2:
            if stack_size.shape[1] > 2:
                stack_size = stack_size[:, :2]  # Take only first 2 columns
            else:
                stack_size = torch.cat([stack_size, torch.zeros(batch_size, 2 - stack_size.shape[1], device=stack_size.device)], dim=1)
        
        # Debug information
        #print(f"Pot size shape: {pot_size.shape}, Stack size shape: {stack_size.shape}")
            
        pot_stack = torch.cat([pot_size, stack_size], dim=1)
        pot_encoding = self.pot_encoder(pot_stack)
        
        # Debug information
        #print(f"Pot encoding shape: {pot_encoding.shape}")
        
        # Process position and stage
        # Ensure position and stage have shape [batch_size, 1]
        if len(position.shape) > 2:
            position = position.squeeze()
        if len(stage.shape) > 2:
            stage = stage.squeeze()
            
        if len(position.shape) == 0:  # scalar
            position = position.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1)
        elif len(position.shape) == 1:  # vector
            if position.size(0) != batch_size:
                position = position[0].unsqueeze(0).unsqueeze(0).repeat(batch_size, 1)
            else:
                position = position.unsqueeze(1)
                
        if len(stage.shape) == 0:  # scalar
            stage = stage.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1)
        elif len(stage.shape) == 1:  # vector
            if stage.size(0) != batch_size:
                stage = stage[0].unsqueeze(0).unsqueeze(0).repeat(batch_size, 1)
            else:
                stage = stage.unsqueeze(1)
        
        # Debug information
        #print(f"Position shape: {position.shape}, Stage shape: {stage.shape}")
                
        position_encoding = self.position_encoder(position.long())
        stage_encoding = self.stage_encoder(stage.long())
        
        # Reshape position and stage encodings to [batch_size, embedding_dim]
        position_encoding = position_encoding.view(batch_size, -1)
        stage_encoding = stage_encoding.view(batch_size, -1)
        
        # Debug information
        #print(f"Position encoding shape: {position_encoding.shape}, Stage encoding shape: {stage_encoding.shape}")
        
        # Combine all features
        combined = torch.cat([
            hole_embeddings,              # [batch_size, 2*card_embedding_dim]
            community_embeddings,         # [batch_size, 5*card_embedding_dim]
            action_history_encoding,      # [batch_size, action_history_dim]
            pot_encoding,                 # [batch_size, pot_dim]
            position_encoding,            # [batch_size, 16]
            stage_encoding                # [batch_size, 16]
        ], dim=1)
        
        # Debug information
        #print(f"Combined shape: {combined.shape}")
        
        # Final encoding
        state_encoding = self.combiner(combined)
        
        return state_encoding

class ValueNetwork(nn.Module):
    """Estimates the value of a state for a specific player position"""
    def __init__(self, input_dim, position_specific=True):
        super(ValueNetwork, self).__init__()
        
        self.position_specific = position_specific
        
        if position_specific:
            # Separate networks for each position
            self.networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)
                ) for _ in range(9)  # 9 possible positions
            ])
        else:
            # Single network for all positions
            self.network = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
    
    def forward(self, x, position=None):
        if self.position_specific and position is not None:
            # Handle potential batch of positions
            if isinstance(position, torch.Tensor) and len(position.shape) > 0:
                # For batched positions, we need to get predictions from different networks
                results = []
                for i, pos in enumerate(position):
                    pos_idx = min(pos.item(), 8)  # Ensure position index is within bounds
                    results.append(self.networks[pos_idx](x[i:i+1]))
                return torch.cat(results, dim=0)
            else:
                # Single position
                pos_idx = min(position, 8) if isinstance(position, int) else min(position.item(), 8)
                return self.networks[pos_idx](x)
        elif not self.position_specific:
            return self.network(x)
        else:
            raise ValueError("Position must be provided for position-specific networks")

class PolicyNetwork(nn.Module):
    """Outputs action probabilities for each state"""
    def __init__(self, input_dim, num_actions, position_specific=True):
        super(PolicyNetwork, self).__init__()
        
        self.position_specific = position_specific
        self.num_actions = num_actions
        
        if position_specific:
            # Separate networks for each position
            self.networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, num_actions)
                ) for _ in range(9)  # 9 possible positions
            ])
        else:
            # Single network for all positions
            self.network = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_actions)
            )
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, position=None):
        if self.position_specific and position is not None:
            # Handle potential batch of positions
            if isinstance(position, torch.Tensor) and len(position.shape) > 0:
                # For batched positions, we need to get predictions from different networks
                results = []
                for i, pos in enumerate(position):
                    pos_idx = min(pos.item(), 8)  # Ensure position index is within bounds
                    results.append(self.networks[pos_idx](x[i:i+1]))
                logits = torch.cat(results, dim=0)
            else:
                # Single position
                pos_idx = min(position, 8) if isinstance(position, int) else min(position.item(), 8)
                logits = self.networks[pos_idx](x)
        elif not self.position_specific:
            logits = self.network(x)
        else:
            raise ValueError("Position must be provided for position-specific networks")
            
        # Apply softmax
        probs = self.softmax(logits)
        return probs

class MCTSNode:
    """Node in the Monte Carlo Tree Search"""
    def __init__(self, state_key, parent=None, prior_p=1.0):
        self.state_key = state_key
        self.parent = parent
        self.children = {}  # action -> MCTSNode
        self.n_visits = 0
        self.Q = 0  # mean action value
        self.P = prior_p  # prior probability
        self.legal_actions = []
        self.is_terminal = False
        self.reward = 0
        
    def expand(self, legal_actions, action_probs):
        """Expand the node with children for each legal action"""
        self.legal_actions = legal_actions
        for action, prob in zip(legal_actions, action_probs):
            if action not in self.children:
                self.children[action] = MCTSNode(
                    state_key=f"{self.state_key}_{action}",
                    parent=self,
                    prior_p=prob
                )
    
    def select(self):
        """Select a child node using UCB1 formula"""
        # Check if there are any children
        if not self.children:
            return None, None
            
        return max(self.children.items(), 
                  key=lambda item: item[1].get_value())
    
    def update(self, leaf_value):
        """Update node value and visit count"""
        self.n_visits += 1
        self.Q += (leaf_value - self.Q) / self.n_visits
    
    def get_value(self):
        """Calculate UCB1 value for this node"""
        # Avoid division by zero
        if self.n_visits == 0:
            return float('inf')  # high value to ensure unexplored nodes are visited
            
        if self.parent is None or self.parent.n_visits == 0:
            # Root node or parent with no visits
            return self.Q
            
        # UCB1 formula
        exploration = 1.414 * math.sqrt(math.log(self.parent.n_visits) / self.n_visits)
        return self.Q + exploration * self.P
    
    def is_leaf(self):
        """Check if this node is a leaf node"""
        return len(self.children) == 0
    
    def is_root(self):
        """Check if this node is the root node"""
        return self.parent is None

class PublicChanceSampling:
    """Handles sampling of public cards for Deep CFR"""
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.deck = self._create_deck()
    
    def _create_deck(self):
        """Create a standard 52-card deck"""
        values = "23456789TJQKA"
        suits = "CDHS"
        deck = []
        for value in values:
            for suit in suits:
                deck.append(value + suit)
        return deck
    
    def sample_public_cards(self, hole_cards, community_cards, num_cards):
        """Sample public cards given hole cards and existing community cards"""
        # Create a deck excluding hole cards and existing community cards
        available_cards = [card for card in self.deck 
                          if card not in hole_cards and card not in community_cards]
        
        # Check if there are enough cards available
        if len(available_cards) < num_cards:
            return []
            
        # Sample the specified number of cards
        sampled_cards = random.sample(available_cards, num_cards)
        
        return sampled_cards
    
    def generate_samples(self, hole_cards, community_cards, stage):
        """Generate multiple samples of public cards based on the current stage"""
        samples = []
        
        # Convert stage to Stage enum if it's an integer
        if isinstance(stage, int):
            try:
                stage = Stage(stage)
            except ValueError:
                # Default to PREFLOP if invalid stage value
                stage = Stage.PREFLOP
                
        # Determine how many cards to sample based on the stage
        if stage == Stage.PREFLOP:
            # Sample flop, turn, and river
            for _ in range(self.num_samples):
                flop = self.sample_public_cards(hole_cards, community_cards, 3)
                if not flop:  # Not enough cards
                    continue
                    
                turn = self.sample_public_cards(hole_cards, community_cards + flop, 1)
                if not turn:  # Not enough cards
                    continue
                    
                river = self.sample_public_cards(hole_cards, community_cards + flop + turn, 1)
                if not river:  # Not enough cards
                    continue
                    
                samples.append((flop, turn, river))
                
        elif stage == Stage.FLOP:
            # Sample turn and river
            for _ in range(self.num_samples):
                turn = self.sample_public_cards(hole_cards, community_cards, 1)
                if not turn:  # Not enough cards
                    continue
                    
                river = self.sample_public_cards(hole_cards, community_cards + turn, 1)
                if not river:  # Not enough cards
                    continue
                    
                samples.append((turn, river))
                
        elif stage == Stage.TURN:
            # Sample river
            for _ in range(self.num_samples):
                river = self.sample_public_cards(hole_cards, community_cards, 1)
                if not river:  # Not enough cards
                    continue
                    
                samples.append((river,))
                
        else:
            # No sampling needed for river or showdown
            samples = [(tuple(),)]
        
        # If no valid samples were generated, add a dummy sample
        if not samples:
            samples = [(tuple(),)]
            
        return samples

class GameState:
    """Simple class to store and copy game state"""
    def __init__(self, table_cards=None, stage=None, current_player=None, pot=None, stacks=None):
        self.table_cards = table_cards if table_cards is not None else []
        self.stage = stage
        self.current_player = current_player
        self.pot = pot
        self.stacks = stacks if stacks is not None else {}
        self.winner = None
        self.done = False

class Player:
    """Deep CFR agent implementation with advanced features"""
    
    def __init__(self, name='DeepCFR', load_model=None, env=None):
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True
        self.env = env
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.state_encoder = None
        self.value_net = None
        self.policy_net = None
        self.value_optimizer = None
        self.policy_optimizer = None
        
        # CFR specific variables
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.iteration = 0
        
        # MCTS parameters
        self.mcts_simulations = 100
        self.mcts_root = None
        
        # Public chance sampling
        self.chance_sampling = PublicChanceSampling(num_samples=50)
        
        # Experience replay
        self.experience_buffer = deque(maxlen=10000)
        
        # Training parameters
        self.batch_size = 64
        self.value_lr = 1e-4
        self.policy_lr = 1e-4
        
        # Store last state and action for updates
        self.last_state = None
        self.last_action = None
        self.last_state_key = None
        self.current_state = None
        
        # Initialize networks
        self.initiate_agent(env)
        
        # Load model if specified
        if load_model:
            self.load(load_model)
            
        # Print initialization information
        print(f"Deep CFR agent initialized with name: {name}")
        print(f"Device: {self.device}")
        print(f"Experience buffer size: {len(self.experience_buffer)}")
        print(f"Batch size: {self.batch_size}")
    
    def initiate_agent(self, env):
        """Initialize the Deep CFR agent with the environment"""
        # Default action dimension if env is not provided
        action_dim = len(Action) if env is None else env.action_space.n
        state_dim = 256  # Output dimension of state encoder
        
        # Create networks
        self.state_encoder = AdvancedStateEncoder().to(self.device)
        self.value_net = ValueNetwork(state_dim, position_specific=True).to(self.device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, position_specific=True).to(self.device)
        
        # Create optimizers
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)
        
        return self
    
    def _encode_state(self, observation, info):
        """Convert observation and info into a tensor representation"""
        try:
            # Extract features from observation and info
            player_data = info.get('player_data', {})
            community_data = info.get('community_data', {})
            
            # Get hole cards
            hole_cards = player_data.get('cards', [])
            hole_indices = self._cards_to_indices(hole_cards)
            
            # Get community cards
            if hasattr(self.env, 'table_cards'):
                community_cards = self.env.table_cards
            else:
                community_cards = []
            community_indices = self._cards_to_indices(community_cards)
            
            # Get action history
            action_history = self._encode_action_history()
            
            # Get pot size and stack size
            pot_size = [[community_data.get('community_pot', 0.0)]]
            stack_size = [[player_data.get('stack', 0.0)]]
            
            # Get position and stage
            position = [[player_data.get('position', 0)]]
            
            # Convert stage to index
            stage_index = 0  # default to PREFLOP
            if 'stage' in community_data:
                stage_data = community_data['stage']
                if isinstance(stage_data, list):
                    # Find the index of 1 in the list
                    try:
                        stage_index = stage_data.index(1)
                    except ValueError:
                        stage_index = 0
                else:
                    # Assume it's a direct stage value
                    stage_index = stage_data
            
            # Create state dictionary for the encoder
            state_dict = {
                'hole_cards': torch.tensor([hole_indices], dtype=torch.long).to(self.device),
                'community_cards': torch.tensor([community_indices], dtype=torch.long).to(self.device),
                'action_history': torch.tensor([action_history], dtype=torch.float32).to(self.device),
                'pot_size': torch.tensor(pot_size, dtype=torch.float32).to(self.device),
                'stack_size': torch.tensor(stack_size, dtype=torch.float32).to(self.device),
                'position': torch.tensor(position, dtype=torch.long).to(self.device),
                'stage': torch.tensor([[stage_index]], dtype=torch.long).to(self.device)
            }
            
            # Check for NaN values in tensors
            for key, tensor in state_dict.items():
                if torch.isnan(tensor).any():
                    print(f"Warning: NaN values detected in {key}: {tensor}")
                    # Replace NaN values with zeros
                    state_dict[key] = torch.nan_to_num(tensor, nan=0.0)
            
            # Encode state
            with torch.no_grad():
                state_encoding = self.state_encoder(state_dict)
            
            return state_encoding
        
        except Exception as e:
            # Log the error and return a default tensor
            print(f"Error encoding state: {e}")
            return torch.zeros((1, 256), device=self.device)
    
    def _cards_to_indices(self, cards):
        """Convert card strings to indices (0-51)"""
        if not cards:
            return [52] * 5  # Use 52 for empty/unknown cards
        
        values = "23456789TJQKA"
        suits = "CDHS"
        
        indices = []
        for card in cards:
            if len(card) < 2:
                # Invalid card format
                indices.append(52)
                continue
                
            value, suit = card[0], card[1]
            
            try:
                value_idx = values.index(value)
                suit_idx = suits.index(suit)
                card_idx = value_idx * 4 + suit_idx
                indices.append(card_idx)
            except (ValueError, IndexError):
                # Invalid card value or suit
                indices.append(52)
        
        # Pad with 52 (unknown) if needed
        while len(indices) < 5:
            indices.append(52)
        
        return indices[:5]  # Ensure we only return 5 indices
    
    def _encode_action_history(self):
        """Encode action history as a sequence of one-hot vectors"""
        if not hasattr(self, 'actions') or not self.actions:
            return np.zeros((10, len(Action)))
        
        # Create a sequence of one-hot vectors for each action
        action_sequence = []
        for action in self.actions:
            try:
                # Handle both enum and direct value cases
                action_value = action.value if hasattr(action, 'value') else action
                
                one_hot = np.zeros(len(Action))
                one_hot[action_value] = 1
                action_sequence.append(one_hot)
            except (IndexError, TypeError):
                # Invalid action, use a zero vector
                action_sequence.append(np.zeros(len(Action)))
        
        # Pad with zeros if needed
        while len(action_sequence) < 10:  # Keep last 10 actions
            action_sequence.insert(0, np.zeros(len(Action)))
        
        # Take only the last 10 actions
        action_sequence = action_sequence[-10:]
        
        return np.array(action_sequence)
    
    def get_state_key(self, observation, info):
        """Convert observation and info into a state key for tracking regrets"""
        try:
            # Extract basic data
            player_data = info.get('player_data', {})
            community_data = info.get('community_data', {})
            
            # Create a unique state representation
            stage = community_data.get('stage', [0, 0, 0, 0])
            if isinstance(stage, list):
                stage_index = stage.index(1) if 1 in stage else 0
            else:
                stage_index = stage
                
            pot = community_data.get('community_pot', 0)
            stack = player_data.get('stack', 0)
            position = player_data.get('position', 0)
            
            # Include hole cards in state key
            hole_cards = player_data.get('cards', [])
            hole_cards_str = '_'.join(sorted(hole_cards)) if hole_cards else 'none'
            
            # Include community cards in state key
            community_cards = self.env.table_cards if hasattr(self.env, 'table_cards') else []
            community_cards_str = '_'.join(sorted(community_cards)) if community_cards else 'none'
            
            # Create a unique state key
            return f"{stage_index}_{pot}_{stack}_{position}_{hole_cards_str}_{community_cards_str}"
            
        except Exception as e:
            # Log the error and return a default state key
            log.error(f"Error creating state key: {e}")
            return f"error_{random.randint(0, 1000000)}"
    
    def get_action_probs_mcts(self, state_key, legal_actions):
        """Get action probabilities using MCTS"""
        try:
            # Create root node if it doesn't exist
            if self.mcts_root is None or self.mcts_root.state_key != state_key:
                self.mcts_root = MCTSNode(state_key)
            
            # Run MCTS simulations
            for _ in range(min(self.mcts_simulations, 20)):  # Limit to 20 simulations for performance
                node = self.mcts_root
                
                # Selection
                while not node.is_leaf() and not node.is_terminal:
                    selection_result = node.select()
                    if selection_result is None:
                        break
                    action, node = selection_result
                
                # Expansion
                if not node.is_terminal and not node.children:
                    # Get state encoding
                    state_encoding = self._encode_state(self.current_state, self.env.info)
                    
                    # Get action probabilities from policy network
                    with torch.no_grad():
                        # Use position from player_data
                        position = self.env.info.get('player_data', {}).get('position', 0)
                        position_tensor = torch.tensor([position], device=self.device)
                        action_probs = self.policy_net(state_encoding, position=position_tensor).cpu().numpy()
                    
                    # Filter for legal actions
                    legal_action_probs = []
                    for action in legal_actions:
                        action_value = action.value if hasattr(action, 'value') else action
                        prob = action_probs[0, action_value]
                        legal_action_probs.append(prob)
                    
                    # Normalize
                    sum_probs = sum(legal_action_probs)
                    if sum_probs > 0:
                        legal_action_probs = [p / sum_probs for p in legal_action_probs]
                    else:
                        legal_action_probs = [1.0 / len(legal_actions)] * len(legal_actions)
                    
                    # Expand node
                    node.expand(legal_actions, legal_action_probs)
                    
                    # Select a random child for simulation
                    if node.children:
                        action = random.choice(list(node.children.keys()))
                        node = node.children[action]
                
                # Simulation - run a simplified simulation from this node
                value = self._simple_simulation(node)
                
                # Backpropagation
                while node is not None:
                    node.update(value)
                    node = node.parent
            
            # Get action probabilities from visit counts
            action_probs = {}
            total_visits = sum(child.n_visits for child in self.mcts_root.children.values())
            
            for action, child in self.mcts_root.children.items():
                action_probs[action] = child.n_visits / total_visits if total_visits > 0 else 1.0 / len(legal_actions)
            
            # If some legal actions are missing, add them with small probability
            for action in legal_actions:
                if action not in action_probs:
                    action_probs[action] = 0.01
            
            # Normalize
            total_prob = sum(action_probs.values())
            action_probs = {action: prob / total_prob for action, prob in action_probs.items()}
            
            return action_probs
            
        except Exception as e:
            # Log the error and return uniform probabilities
            log.error(f"Error in MCTS: {e}")
            return {action: 1.0 / len(legal_actions) for action in legal_actions}
    
    def _simple_simulation(self, node):
        """Run a simplified simulation (random play) from this node"""
        # Just return a random value between -1 and 1 for simplicity
        # In a real implementation, you would simulate gameplay
        return random.uniform(-1, 1)
    
    def get_action_probs(self, state_key, legal_actions):
        """Get action probabilities from policy network"""
        try:
            # Use uniform policy for first 100 iterations
            if self.iteration < 100:
                return {action: 1.0 / len(legal_actions) for action in legal_actions}
            
            # Get state encoding
            state_encoding = self._encode_state(self.current_state, self.env.info)
            
            # Get action probabilities from policy network
            with torch.no_grad():
                # Use position from player_data
                position = self.env.info.get('player_data', {}).get('position', 0)
                position_tensor = torch.tensor([position], device=self.device)
                action_probs = self.policy_net(state_encoding, position=position_tensor).cpu().numpy()
            
            # Filter for legal actions
            legal_action_probs = {}
            for action in legal_actions:
                action_value = action.value if hasattr(action, 'value') else action
                legal_action_probs[action] = action_probs[0, action_value]
            
            # Normalize
            total_prob = sum(legal_action_probs.values())
            if total_prob > 0:
                legal_action_probs = {action: prob / total_prob for action, prob in legal_action_probs.items()}
            else:
                legal_action_probs = {action: 1.0 / len(legal_actions) for action in legal_actions}
            
            return legal_action_probs
            
        except Exception as e:
            # Log the error and return uniform probabilities
            log.error(f"Error getting action probabilities: {e}")
            return {action: 1.0 / len(legal_actions) for action in legal_actions}
    
    def update_regrets(self, state_key, action, regret):
        """Update cumulative regrets for the state-action pair"""
        self.regret_sum[state_key][action] += regret
        
        # Ensure regrets are non-negative (for vanilla CFR)
        self.regret_sum[state_key][action] = max(0, self.regret_sum[state_key][action])
        
        # Update strategy sum using current policy
        action_probs = self.get_action_probs(state_key, list(self.regret_sum[state_key].keys()))
        for a, prob in action_probs.items():
            self.strategy_sum[state_key][a] += prob
    
    def train_networks(self):
        """Train value and policy networks on batched experience"""
        try:
            # Check if we have enough samples
            if len(self.experience_buffer) < self.batch_size:
                print(f"Not enough samples for training: {len(self.experience_buffer)}/{self.batch_size}")
                return
            
            print(f"Starting training with {len(self.experience_buffer)} samples")
            
            # Sample batch from experience buffer
            batch = random.sample(self.experience_buffer, self.batch_size)
            
            # Prepare batch data
            states = []
            value_targets = []
            policy_targets = []
            positions = []
            
            for state_key, action, reward, next_state_key, done, state_encoding, info in batch:
                # Get state encoding from stored experience
                states.append(state_encoding.cpu().numpy()[0])
                
                # Value target is the reward
                value_targets.append([reward])
                
                # Policy target is based on accumulated regrets
                policy_target = np.zeros(self.policy_net.num_actions)
                if state_key in self.regret_sum:
                    # Use regret matching to compute policy
                    total_regret = sum(max(0, r) for r in self.regret_sum[state_key].values())
                    
                    if total_regret > 0:
                        for a, regret in self.regret_sum[state_key].items():
                            if regret > 0:
                                action_value = a.value if hasattr(a, 'value') else a
                                policy_target[action_value] = regret / total_regret
                    else:
                        # Uniform policy if no positive regrets
                        num_actions = len(self.regret_sum[state_key])
                        if num_actions > 0:
                            for a in self.regret_sum[state_key]:
                                action_value = a.value if hasattr(a, 'value') else a
                                policy_target[action_value] = 1.0 / num_actions
                
                policy_targets.append(policy_target)
                
                # Position is the current player's position
                position = info.get('player_data', {}).get('position', 0)
                positions.append(position)
            
            # Convert to tensors
            state_tensor = torch.FloatTensor(states).to(self.device)
            value_targets = torch.FloatTensor(value_targets).to(self.device)
            policy_targets = torch.FloatTensor(policy_targets).to(self.device)
            positions = torch.LongTensor(positions).to(self.device)
            
            # Check for NaN values in tensors
            if torch.isnan(state_tensor).any():
                print("Warning: NaN values detected in state tensor")
                state_tensor = torch.nan_to_num(state_tensor, nan=0.0)
            
            if torch.isnan(value_targets).any():
                print("Warning: NaN values detected in value targets")
                value_targets = torch.nan_to_num(value_targets, nan=0.0)
            
            if torch.isnan(policy_targets).any():
                print("Warning: NaN values detected in policy targets")
                policy_targets = torch.nan_to_num(policy_targets, nan=0.0)
            
            # CRITICAL FIX: Learning rate decay for better convergence
            if hasattr(self, 'iteration') and self.iteration > 1000:
                for param_group in self.value_optimizer.param_groups:
                    param_group['lr'] = self.value_lr * 0.5
                for param_group in self.policy_optimizer.param_groups:
                    param_group['lr'] = self.policy_lr * 0.5
            
            # Train value network
            self.value_optimizer.zero_grad()
            value_pred = self.value_net(state_tensor, positions)
            value_loss = nn.MSELoss()(value_pred, value_targets)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)  # Gradient clipping
            self.value_optimizer.step()
            
            # Store loss for monitoring
            self.last_value_loss = value_loss.item()
            
            # Train policy network
            self.policy_optimizer.zero_grad()
            policy_pred = self.policy_net(state_tensor, positions)
            
            # CRITICAL FIX: Use KL divergence loss instead of cross entropy for policy learning
            # This is more appropriate for learning probability distributions
            policy_loss = F.kl_div(
                F.log_softmax(policy_pred, dim=1),
                policy_targets,
                reduction='batchmean'
            )
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # Gradient clipping
            self.policy_optimizer.step()
            
            # Store loss for monitoring
            self.last_policy_loss = policy_loss.item()
            
            print(f"Training step - Value loss: {self.last_value_loss:.6f}, Policy loss: {self.last_policy_loss:.6f}")
            
        except Exception as e:
            # Log the error
            print(f"Error training networks: {e}")
            import traceback
            traceback.print_exc()
    
    def action(self, action_space, observation, info):
        """Choose action based on current state and policy"""
        try:
            # Check for NaN values in observation
            if isinstance(observation, np.ndarray) and np.isnan(observation).any():
                print(f"Warning: NaN values detected in observation: {observation}")
                # Replace NaN values with zeros
                observation = np.nan_to_num(observation, nan=0.0)
            
            # Store current state and info
            self.current_state = observation
            
            # Create state key
            state_key = self.get_state_key(observation, info)
            
            # Get state encoding for experience replay
            try:
                state_encoding = self._encode_state(observation, info)
                
                # Check for NaN values in state encoding
                if torch.isnan(state_encoding).any():
                    print("Warning: NaN values detected in state encoding")
                    state_encoding = torch.nan_to_num(state_encoding, nan=0.0)
                
                # Store state encoding for later use
                self.current_state_encoding = state_encoding
                
            except Exception as e:
                print(f"Error encoding state: {e}")
                import traceback
                traceback.print_exc()
                # Return a random action if encoding fails
                return random.choice(list(action_space))
            
            # Get action probabilities using MCTS
            try:
                action_probs = self.get_action_probs_mcts(state_key, action_space)
            except Exception as e:
                print(f"Error getting action probabilities: {e}")
                # Return a random action if getting probabilities fails
                return random.choice(list(action_space))
            
            # Choose action using epsilon-greedy strategy
            if random.random() < max(0.1, 1.0 - self.iteration/1000):  # Exploration rate
                action = random.choice(list(action_space))
                print(f"Exploration: chose random action {action}")
            else:
                action = max(action_probs.items(), key=lambda x: x[1])[0]
                print(f"Exploitation: chose action {action} with probability {action_probs[action]:.4f}")
            
            # Store action and state for regret calculation
            self.last_action = action
            self.last_state_key = state_key
            self.last_state = {
                'observation': observation, 
                'info': info, 
                'encoding': self.current_state_encoding
            }
            
            # Update action history
            self.actions.append(action)
            
            return action
            
        except Exception as e:
            # Log the error and take a random action
            print(f"Error choosing action: {e}")
            import traceback
            traceback.print_exc()
            return random.choice(list(action_space))
    
    def update(self, reward, next_observation, next_info, done):
        """Update agent after an action"""
        try:
            if not hasattr(self, 'last_action') or self.last_action is None:
                print("Warning: No last_action found, skipping update")
                return
            
            # Calculate immediate regret - standard CFR formula
            next_state_key = self.get_state_key(next_observation, next_info)
            next_state_encoding = self._encode_state(next_observation, next_info)
            
            # Get legal moves for next state
            legal_moves = self.env.legal_moves if hasattr(self.env, 'legal_moves') else []
            
            # Get action probabilities for next state
            if legal_moves:
                action_probs = self.get_action_probs(next_state_key, legal_moves)
                # Calculate expected value - sum of probability * reward for each action
                expected_value = sum(prob * reward for action, prob in action_probs.items())
            else:
                expected_value = 0  # No legal moves means end of game
            
            # Calculate regret - difference between actual reward and expected reward
            regret = reward - expected_value
            
            # Update regrets - key part of CFR algorithm
            self.update_regrets(self.last_state_key, self.last_action, regret)
            
            # Store experience for replay - critical for deep learning component
            if hasattr(self, 'last_state') and self.last_state is not None:
                # Ensure we have a valid state encoding
                if 'encoding' not in self.last_state or self.last_state['encoding'] is None:
                    # If we don't have a valid encoding, create one
                    self.last_state['encoding'] = self._encode_state(self.last_state['observation'], self.last_state['info'])
                
                # Add to experience buffer
                self.experience_buffer.append((
                    self.last_state_key,
                    self.last_action,
                    reward,
                    next_state_key,
                    done,
                    self.last_state['encoding'],
                    self.last_state['info']
                ))
                
                # Print buffer size periodically
                if self.iteration % 10 == 0:
                    print(f"Experience buffer size: {len(self.experience_buffer)}/{self.batch_size}")
                
                # Train networks much more frequently - key fix
                if len(self.experience_buffer) >= self.batch_size:
                    # Train after every update once we have enough samples
                    print(f"Training networks with {len(self.experience_buffer)} samples")
                    self.train_networks()
                else:
                    print(f"Not enough samples for training: {len(self.experience_buffer)}/{self.batch_size}")
            else:
                print("Warning: No last_state found, skipping experience storage")
            
            # Clear last state and action if done
            if done:
                self.last_action = None
                self.last_state_key = None
                self.last_state = None
                self.actions = []
                
            self.iteration += 1
            
        except Exception as e:
            # Log the error
            print(f"Error updating agent: {e}")
            import traceback
            traceback.print_exc()
    
    def save(self, path):
        """Save the model"""
        try:
            # Convert defaultdicts to regular dicts for saving
            regret_dict = {}
            for state_key, action_regrets in self.regret_sum.items():
                regret_dict[state_key] = {str(action): regret for action, regret in action_regrets.items()}
                
            strategy_dict = {}
            for state_key, action_probs in self.strategy_sum.items():
                strategy_dict[state_key] = {str(action): prob for action, prob in action_probs.items()}
            
            torch.save({
                'state_encoder': self.state_encoder.state_dict(),
                'value_net': self.value_net.state_dict(),
                'policy_net': self.policy_net.state_dict(),
                'regret_sum': regret_dict,
                'strategy_sum': strategy_dict,
                'iteration': self.iteration
            }, path)
            
            log.info(f"Model saved to {path}")
            
        except Exception as e:
            # Log the error
            log.error(f"Error saving model: {e}")
    
    def load(self, path):
        """Load the model"""
        try:
            # Make sure networks are initialized before loading
            if self.state_encoder is None:
                self.initiate_agent(self.env)
                
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load network weights
            self.state_encoder.load_state_dict(checkpoint['state_encoder'])
            self.value_net.load_state_dict(checkpoint['value_net'])
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            
            # Load regrets and strategy sums
            regret_dict = checkpoint['regret_sum']
            self.regret_sum = defaultdict(lambda: defaultdict(float))
            for state_key, action_regrets in regret_dict.items():
                for action_str, regret in action_regrets.items():
                    try:
                        # Try to convert string to Action enum
                        action = Action(int(action_str))
                    except (ValueError, TypeError):
                        # If conversion fails, use the original string
                        action = action_str
                    self.regret_sum[state_key][action] = regret
            
            strategy_dict = checkpoint['strategy_sum']
            self.strategy_sum = defaultdict(lambda: defaultdict(float))
            for state_key, action_probs in strategy_dict.items():
                for action_str, prob in action_probs.items():
                    try:
                        # Try to convert string to Action enum
                        action = Action(int(action_str))
                    except (ValueError, TypeError):
                        # If conversion fails, use the original string
                        action = action_str
                    self.strategy_sum[state_key][action] = prob
            
            # Load iteration count
            self.iteration = checkpoint['iteration']
            
            log.info(f"Model loaded from {path}, iteration: {self.iteration}")
            
        except Exception as e:
            # Log the error
            log.error(f"Error loading model: {e}")
            # Initialize a fresh model
            self.initiate_agent(self.env)