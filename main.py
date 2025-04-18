"""
neuron poker

Usage:
  main.py selfplay random [options]
  main.py selfplay keypress [options]
  main.py selfplay consider_equity [options]
  main.py selfplay equity_improvement --improvement_rounds=<> [options]
  main.py selfplay dqn_train [options]
  main.py selfplay dqn_play [options]
  main.py selfplay dqn_train_custom [options]
  main.py selfplay dqn_play_custom [options]
  main.py selfplay 
  main.py learn_table_scraping [options]

options:
  -h --help                 Show this screen.
  -r --render               render screen
  -c --use_cpp_montecarlo   use cpp implementation of equity calculator. Requires cpp compiler but is 500x faster
  -f --funds_plot           Plot funds at end of episode
  --log                     log file
  --name=<>                 Name of the saved model
  --screenloglevel=<>       log level on screen
  --episodes=<>             number of episodes to play
  --stack=<>                starting stack for each player [default: 500].

"""

import logging

import gym
import numpy as np
import pandas as pd
from docopt import docopt

from gym_env.env import PlayerShell
from tools.helper import get_config
from tools.helper import init_logger


# pylint: disable=import-outside-toplevel

def command_line_parser():
    """Entry function"""
    args = docopt(__doc__)
    if args['--log']:
        logfile = args['--log']
    else:
        print("Using default log file")
        logfile = 'default'
    model_name = args['--name'] if args['--name'] else 'dqn1'
    screenloglevel = logging.INFO if not args['--screenloglevel'] else \
        getattr(logging, args['--screenloglevel'].upper())
    _ = get_config()
    init_logger(screenlevel=screenloglevel, filename=logfile)
    print(f"Screenloglevel: {screenloglevel}")
    log = logging.getLogger("")
    log.info("Initializing program")

    if args['selfplay']:
        num_episodes = 1 if not args['--episodes'] else int(args['--episodes'])
        runner = SelfPlay(render=args['--render'], num_episodes=num_episodes,
                          use_cpp_montecarlo=args['--use_cpp_montecarlo'],
                          funds_plot=args['--funds_plot'],
                          stack=int(args['--stack']))

        if args['random']:
            runner.random_agents()

        elif args['keypress']:
            runner.key_press_agents()

        elif args['consider_equity']:
            runner.equity_vs_random()

        elif args['equity_improvement']:
            improvement_rounds = int(args['--improvement_rounds'])
            runner.equity_self_improvement(improvement_rounds)

        elif args['dqn_train']:
            runner.dqn_train_keras_rl(model_name)
        
        elif args['dqn_train_custom']:
            runner.dqn_train_custom(model_name)

        elif args['dqn_play']:
            runner.dqn_play_keras_rl(model_name)
        
        elif args['dqn_play_custom']:
            runner.dqn_play_custom(model_name)


    else:
        raise RuntimeError("Argument not yet implemented")


class SelfPlay:
    """Orchestration of playing against itself"""

    def __init__(self, render, num_episodes, use_cpp_montecarlo, funds_plot, stack=500):
        """Initialize"""
        self.winner_in_episodes = []
        self.use_cpp_montecarlo = use_cpp_montecarlo
        self.funds_plot = funds_plot
        self.render = render
        self.env = None
        self.num_episodes = num_episodes
        self.stack = stack
        self.log = logging.getLogger(__name__)

    def random_agents(self):
        """Create an environment with 6 random players"""
        from agents.agent_random import Player as RandomPlayer
        env_name = 'neuron_poker-v0'
        num_of_plrs = 2
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        for _ in range(num_of_plrs):
            player = RandomPlayer()
            self.env.add_player(player)

        self.env.reset()

    def key_press_agents(self):
        """Create an environment with 6 key press agents"""
        from agents.agent_keypress import Player as KeyPressAgent
        env_name = 'neuron_poker-v0'
        num_of_plrs = 2
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        for _ in range(num_of_plrs):
            player = KeyPressAgent()
            self.env.add_player(player)

        self.env.reset()

    def equity_vs_random(self):
        """Create 6 players, 4 of them equity based, 2 of them random"""
        from agents.agent_consider_equity import Player as EquityPlayer
        from agents.agent_random import Player as RandomPlayer
        env_name = 'neuron_poker-v0'
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        self.env.add_player(EquityPlayer(name='equity/50/50', min_call_equity=.5, min_bet_equity=-.5))
        self.env.add_player(EquityPlayer(name='equity/50/80', min_call_equity=.8, min_bet_equity=-.8))
        self.env.add_player(EquityPlayer(name='equity/70/70', min_call_equity=.7, min_bet_equity=-.7))
        self.env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=-.3))
        self.env.add_player(RandomPlayer())
        self.env.add_player(RandomPlayer())

        for _ in range(self.num_episodes):
            self.env.reset()
            self.winner_in_episodes.append(self.env.winner_ix)

        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print(league_table)
        print(f"Best Player: {best_player}")

    def equity_self_improvement(self, improvement_rounds):
        """Create 6 players, 4 of them equity based, 2 of them random"""
        from agents.agent_consider_equity import Player as EquityPlayer
        calling = [.1, .2, .3, .4, .5, .6]
        betting = [.2, .3, .4, .5, .6, .7]

        for improvement_round in range(improvement_rounds):
            env_name = 'neuron_poker-v0'
            self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
            for i in range(6):
                self.env.add_player(EquityPlayer(name=f'Equity/{calling[i]}/{betting[i]}',
                                                 min_call_equity=calling[i],
                                                 min_bet_equity=betting[i]))

            for _ in range(self.num_episodes):
                self.env.reset()
                self.winner_in_episodes.append(self.env.winner_ix)

            league_table = pd.Series(self.winner_in_episodes).value_counts()
            best_player = int(league_table.index[0])
            print(league_table)
            print(f"Best Player: {best_player}")

            # self improve:
            self.log.info(f"Self improvment round {improvement_round}")
            for i in range(6):
                calling[i] = np.mean([calling[i], calling[best_player]])
                self.log.info(f"New calling for player {i} is {calling[i]}")
                betting[i] = np.mean([betting[i], betting[best_player]])
                self.log.info(f"New betting for player {i} is {betting[i]}")

    def dqn_train_keras_rl(self, model_name):
        """Implementation of kreras-rl deep q learing."""
        from agents.agent_consider_equity import Player as EquityPlayer
        from agents.agent_keras_rl_dqn import Player as DQNPlayer
        from agents.agent_random import Player as RandomPlayer
        env_name = 'neuron_poker-v0'
        env = gym.make(env_name, initial_stacks=self.stack, funds_plot=self.funds_plot, render=self.render,
                       use_cpp_montecarlo=self.use_cpp_montecarlo)

        np.random.seed(123)
        env.seed(123)
        env.add_player(EquityPlayer(name='equity/50/70', min_call_equity=.5, min_bet_equity=.7))
        env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=.3))
        env.add_player(RandomPlayer())
        env.add_player(RandomPlayer())
        env.add_player(RandomPlayer())
        env.add_player(PlayerShell(name='keras-rl', stack_size=self.stack))  # shell is used for callback to keras rl

        env.reset()

        dqn = DQNPlayer()
        dqn.initiate_agent(env)
        dqn.train(env_name=model_name)


    def dqn_play_keras_rl(self, model_name):
        """Create 6 players, one of them a trained DQN"""
        from agents.agent_consider_equity import Player as EquityPlayer
        from agents.agent_keras_rl_dqn import Player as DQNPlayer
        from agents.agent_random import Player as RandomPlayer
        env_name = 'neuron_poker-v0'
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        self.env.add_player(EquityPlayer(name='equity/50/50', min_call_equity=.5, min_bet_equity=.5))
        self.env.add_player(EquityPlayer(name='equity/50/80', min_call_equity=.8, min_bet_equity=.8))
        self.env.add_player(EquityPlayer(name='equity/70/70', min_call_equity=.7, min_bet_equity=.7))
        self.env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=.3))
        self.env.add_player(RandomPlayer())
        self.env.add_player(PlayerShell(name='keras-rl', stack_size=self.stack))

        self.env.reset()

        dqn = DQNPlayer(load_model=model_name, env=self.env)
        dqn.play(nb_episodes=self.num_episodes, render=self.render)


    def dqn_train_custom(self, model_name):
        """Train a PyTorch-based DQN agent"""
        from agents.agent_consider_equity import Player as EquityPlayer
        from agents.agent_double_dqn import Player as TorchDQNPlayer  # Your PyTorch implementation
        import pandas as pd
        
        env_name = 'neuron_poker-v0'
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        
        # Add opponents
        self.env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.20, min_bet_equity=.30))
        self.env.add_player(EquityPlayer(name='equity/30/40', min_call_equity=.30, min_bet_equity=.40))

        # Add your PyTorch DQN player
        torch_dqn_player = TorchDQNPlayer(name=model_name, env=self.env)
        self.env.add_player(torch_dqn_player)

        self.env.reset()
        
        # Train the agent
        torch_dqn_player.train(env_name,num_episodes=self.num_episodes)
        
        # Run evaluation episodes
        self.winner_in_episodes = []
        for episode in range(self.num_episodes):
            self.env.reset()
            self.winner_in_episodes.append(self.env.winner_ix)
            print("Episode Number", episode)
        
        # Print results
        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]
        
        print("League Table")
        print("============")
        print(league_table)
        print(f"Best Player: {best_player}")

    # def dqn_play_custom(self, model_name):
    #     """Train a PyTorch-based DQN agent"""
    #     from agents.agent_consider_equity import Player as EquityPlayer
    #     from agents.agent_double_dqn import Player as TorchDQNPlayer  # Your PyTorch implementation
    #     from agents.agent_keypress import Player as KeyPressAgent

    #     import pandas as pd
        
    #     env_name = 'neuron_poker-v0'
    #     self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        
    #     # Add opponents
    #     self.env.add_player(KeyPressAgent())
    #     self.env.add_player(EquityPlayer(name='equity/30/40', min_call_equity=.30, min_bet_equity=.40))

    #     # Add your PyTorch DQN player
    #     torch_dqn_player = TorchDQNPlayer(name=model_name, env=self.env)
    #     self.env.add_player(torch_dqn_player)

    #     self.env.reset()
        
    #     # Train the agent
    #     torch_dqn_player.train(env_name,num_episodes=self.num_episodes)
        
    #     # Run evaluation episodes
    #     self.winner_in_episodes = []
    #     for episode in range(self.num_episodes):
    #         self.env.reset()
    #         self.winner_in_episodes.append(self.env.winner_ix)
    #         print("Episode Number", episode)
        
    #     # Print results
    #     league_table = pd.Series(self.winner_in_episodes).value_counts()
    #     best_player = league_table.index[0]
        
    #     print("League Table")
    #     print("============")
    #     print(league_table)
    #     print(f"Best Player: {best_player}")

    def dqn_play_custom(self, model_name):
        """Play poker with a pre-trained PyTorch-based DQN agent"""
        from agents.agent_consider_equity import Player as EquityPlayer
        from agents.agent_double_dqn import Player as TorchDQNPlayer
        from agents.agent_keypress import Player as KeyPressAgent
        import pandas as pd
        
        env_name = 'neuron_poker-v0'
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        
        # Add opponents
        self.env.add_player(KeyPressAgent())
        self.env.add_player(EquityPlayer(name='equity/30/40', min_call_equity=.30, min_bet_equity=.40))

        # Add your PyTorch DQN player with the pre-trained model
        torch_dqn_player = TorchDQNPlayer(name=model_name, load_model=model_name, env=self.env)
        self.env.add_player(torch_dqn_player)

        # Initialize the environment
        self.env.reset()
        
        # No training, just play episodes with the loaded model
        print(f"Playing with pre-trained model: {model_name}")
        
        # Play episodes
        self.winner_in_episodes = []
        for episode in range(self.num_episodes):
            self.env.reset()
            done = False
            
            # Play a complete episode
            while not done:
                # Get current player
                current_player = self.env.current_player
                
                # If it's our DQN agent, explicitly call its action method
                if hasattr(current_player, 'name') and current_player.name == model_name:
                    # Get action from the trained policy
                    action = torch_dqn_player.action(
                        self.env.legal_moves, 
                        self.env.observation, 
                        self.env.info
                    )
                    _, reward, done, _ = self.env.step(action)
                else:
                    # For other players (including KeyPressAgent), let the environment handle it
                    _, _, done, _ = self.env.step(None)

            
            self.winner_in_episodes.append(self.env.winner_ix)
            print(f"Episode {episode+1}: Winner = {self.env.winner_ix}")
        
        # Print results
        league_table = pd.Series(self.winner_in_episodes).value_counts()
        if len(league_table) > 0:
            best_player = league_table.index[0]
            best_player_name = [p.name for i, p in enumerate(self.env.players) if i == best_player][0]
            
            print("League Table")
            print("============")
            print(league_table)
            print(f"Best Player: {best_player_name} (index {best_player})")
        else:
            print("No winners recorded during play")

    def deep_cfr_train(self, model_name):
            """Train the enhanced Deep CFR agent"""
            from agents.agent_consider_equity import Player as EquityPlayer
            from agents.agent_random import Player as RandomPlayer
            from agents.agent_deep_cfr import Player as DeepCFRPlayer
            import pandas as pd
            import time
            import numpy as np
            
            env_name = 'neuron_poker-v0'
            self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render, funds_plot=False)
            
            # Add opponents
            self.env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.20, min_bet_equity=.30))
            self.env.add_player(EquityPlayer(name='equity/30/40', min_call_equity=.30, min_bet_equity=.40))
            
            # Add Deep CFR player
            deep_cfr_player = DeepCFRPlayer(name=model_name, env=self.env)
            
            # Keep track of training stats without affecting output
            training_stats = {
                'training_calls': 0,
                'regret_updates': 0,
                'wins': 0,
                'episodes': 0,
                'experience_buffer_size': []
            }
            
            # Override methods to track stats silently
            original_train_networks = deep_cfr_player.train_networks
            def silent_train_tracker(*args, **kwargs):
                result = original_train_networks(*args, **kwargs)
                training_stats['training_calls'] += 1
                return result
            
            original_update_regrets = deep_cfr_player.update_regrets
            def silent_regret_tracker(*args, **kwargs):
                result = original_update_regrets(*args, **kwargs)
                training_stats['regret_updates'] += 1
                return result
            
            # Replace methods with tracking versions
            deep_cfr_player.train_networks = silent_train_tracker
            deep_cfr_player.update_regrets = silent_regret_tracker
            
            # Save initial model for later comparison
            self.env.add_player(deep_cfr_player)
            self.env.reset()
            
            # Train the agent
            print(f"Training Deep CFR agent '{model_name}' for {self.num_episodes} episodes...")
            start_time = time.time()
            deep_cfr_player.save(f"{model_name}_initial.pt")
            
            # Create progress tracking variables
            winner_in_episodes = []
            episode_rewards = []
            
            # Run training episodes
            for episode in range(self.num_episodes):
                try:
                    # Reset environment
                    self.env.reset()
                    training_stats['episodes'] += 1
                    episode_reward = 0
                    
                    print(f"\nStarting episode {episode+1}/{self.num_episodes}")
                    print(f"Current experience buffer size: {len(deep_cfr_player.experience_buffer)}")
                    print(f"Training calls so far: {training_stats['training_calls']}")
                    
                    # Play one episode
                    done = False
                    step = 0
                    while not done:
                        step += 1
                        # Get current player (safely)
                        try:
                            if hasattr(self.env, 'current_player_idx'):
                                current_idx = self.env.current_player_idx
                                current_player = self.env.players[current_idx] if isinstance(current_idx, int) else None
                            else:
                                current_idx = None
                                current_player = None
                                
                            # If we couldn't get the current player, use a fallback
                            if current_player is None:
                                # Try each player
                                for idx, player in enumerate(self.env.players):
                                    if idx == 2:  # Deep CFR player
                                        # Always use our deep_cfr_player reference
                                        current_player = deep_cfr_player
                                        current_idx = 2
                                        break
                                        
                            # Get action
                            if current_idx == 2:  # Deep CFR player's turn
                                print(f"Deep CFR player's turn (step {step})")
                                action = deep_cfr_player.action(
                                    self.env.legal_moves, 
                                    self.env.observation, 
                                    self.env.info
                                )
                            else:
                                # Other player's turn - use whatever action method is available
                                action = current_player.action(
                                    self.env.legal_moves,
                                    self.env.observation,
                                    self.env.info
                                ) if hasattr(current_player, 'action') else None
                                
                            # If we couldn't get an action, use a random one
                            if action is None:
                                import random
                                action = random.choice(self.env.legal_moves)
                                print(f"Using random action: {action}")
                            
                            # Take the action
                            observation, reward, done, info = self.env.step(action)
                            episode_reward += reward
                            
                            # Update the Deep CFR player if it was their turn
                            if current_idx == 2:
                                print(f"Updating Deep CFR player with reward: {reward}")
                                deep_cfr_player.update(reward, observation, info, done)
                                
                        except Exception as e:
                            # Print a simplified error message and continue
                            print(f"Error during game step: {e}")
                            import traceback
                            traceback.print_exc()
                            if hasattr(self.env, 'legal_moves') and not self.env.legal_moves:
                                # No legal moves means the episode is over
                                done = True
                            else:
                                # Other error - try to continue
                                continue
                    
                    # Record the winner and reward
                    episode_rewards.append(episode_reward)
                    if hasattr(self.env, 'winner_ix'):
                        winner_in_episodes.append(self.env.winner_ix)
                        if self.env.winner_ix == 2:  # Deep CFR player won
                            training_stats['wins'] += 1
                            print(f"Deep CFR player won episode {episode+1}")
                        else:
                            print(f"Deep CFR player lost episode {episode+1} to player {self.env.winner_ix}")
                    
                    # Record experience buffer size
                    training_stats['experience_buffer_size'].append(len(deep_cfr_player.experience_buffer))
                    
                    # Print progress every 10 episodes
                    if (episode + 1) % 10 == 0 or episode == 0:
                        win_rate = training_stats['wins'] / training_stats['episodes'] if training_stats['episodes'] > 0 else 0
                        print(f"Episode {episode+1}/{self.num_episodes} - Win rate: {win_rate:.4f} - Training calls: {training_stats['training_calls']}")
                        print(f"Experience buffer size: {len(deep_cfr_player.experience_buffer)}")
                        
                    # Periodically save the model
                    if (episode + 1) % 50 == 0:
                        deep_cfr_player.save(f"{model_name}_episode_{episode+1}.pt")
                        
                except Exception as e:
                    # Print simplified error message and continue to next episode
                    print(f"Error in episode {episode+1}: {str(e)[:50]}...")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Save final model
            deep_cfr_player.save(f"{model_name}_final.pt")
            
            # Training time
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")
            
            # Calculate league table
            league_table = pd.Series(winner_in_episodes).value_counts()
            best_player = league_table.index[0] if not league_table.empty else None
            
            print("\nLeague Table")
            print("============")
            print(league_table)
            print(f"Best Player: {best_player}")
            
            # Print training statistics
            print("\nTraining Statistics")
            print("==================")
            print(f"Total episodes: {training_stats['episodes']}")
            print(f"Total training calls: {training_stats['training_calls']}")
            print(f"Total regret updates: {training_stats['regret_updates']}")
            print(f"Win rate: {training_stats['wins'] / training_stats['episodes'] if training_stats['episodes'] > 0 else 0:.4f}")
            print(f"Final experience buffer size: {len(deep_cfr_player.experience_buffer)}")
            
            # Plot experience buffer size over time
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 5))
                plt.plot(training_stats['experience_buffer_size'])
                plt.title('Experience Buffer Size Over Time')
                plt.xlabel('Episode')
                plt.ylabel('Buffer Size')
                plt.savefig(f"{model_name}_buffer_size.png")
                plt.close()
                print(f"Experience buffer size plot saved to {model_name}_buffer_size.png")
            except Exception as e:
                print(f"Error plotting experience buffer size: {e}")
            
            # Verify training occurred by comparing initial and final weights
            try:
                import torch
                initial_model = torch.load(f"{model_name}_initial.pt")
                final_model = torch.load(f"{model_name}_final.pt")
                
                # Compare a few key parameters
                total_diff = 0
                param_count = 0
                
                for component in ['value_net', 'policy_net']:
                    if component in initial_model and component in final_model:
                        initial_state = initial_model[component]
                        final_state = final_model[component]
                        
                        for key in initial_state:
                            if key in final_state:
                                initial_param = initial_state[key].cpu().numpy()
                                final_param = final_state[key].cpu().numpy()
                                diff = np.sum(np.abs(final_param - initial_param))
                                
                                total_diff += diff
                                param_count += np.prod(initial_param.shape)
                
                if param_count > 0:
                    avg_diff = total_diff / param_count
                    print(f"\nTraining Verification: Average parameter change: {avg_diff:.8f}")
                    if avg_diff < 1e-6:
                        print("WARNING: Parameters barely changed - model may not be learning!")
                    else:
                        print("Model parameters changed significantly - training is working properly.")
                
            except Exception as e:
                print(f"Could not verify training: {e}")
                
            return winner_in_episodes, league_table
    
    def _evaluate_deep_cfr(self, deep_cfr_player, num_episodes):
        """Evaluate the Deep CFR agent against other agents"""
        from agents.agent_consider_equity import Player as EquityPlayer
        from agents.agent_random import Player as RandomPlayer
        
        # Create a new environment for evaluation
        eval_env = gym.make('neuron_poker-v0', initial_stacks=self.stack, render=False)
        
        # Add opponents
        eval_env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.20, min_bet_equity=.30))
        eval_env.add_player(EquityPlayer(name='equity/30/40', min_call_equity=.30, min_bet_equity=.40))
        eval_env.add_player(RandomPlayer(name='random1'))
        eval_env.add_player(RandomPlayer(name='random2'))
        
        # Add Deep CFR player
        eval_env.add_player(deep_cfr_player)
        
        # Run evaluation episodes
        winners = []
        for _ in range(num_episodes):
            eval_env.reset()
            done = False
            while not done:
                action = eval_env.players[eval_env.current_player].action(
                    eval_env.legal_moves,
                    eval_env.observation,
                    eval_env.info
                )
                _, _, done, _ = eval_env.step(action)
            
            winners.append(eval_env.winner_ix)
        
        # Calculate results
        results = pd.Series(winners).value_counts()
        win_rate = results.get(4, 0) / num_episodes  # Assuming Deep CFR player is at index 4
        
        return {
            'win_rate': win_rate,
            'results': results
        }

if __name__ == '__main__':
    command_line_parser()
