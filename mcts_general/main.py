from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import logging
import gymnasium as gym

import gym_envs

from mcts_agent import MCTSAgent
from config import MCTSAgentConfig
from game import DiscreteGymGame

import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

STRATEGY_FILE = "strategies.json"


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve',
                            description='Training RL models with reward shaping for strategy transfer')
    parser.add_argument('--exp_id',
                        default='test',
                        help='Experiment identifier (used for saving and logging data)',
                        required=True)
    parser.add_argument('--env',
                        default='DungeonCrawler-v0',
                        choices=['SimplePacman-v0',
                                 'DungeonCrawler-v0',
                                 'DungeonCrawler-v1',
                                 'SimpleBankHeist-v0',
                                 'SimpleMinecraft-v0'],
                        help='Name of the environment',
                        required=True)
    parser.add_argument('--env_map',
                        default='map.txt',
                        choices=['map.txt',
                                 'map_simple.txt'],
                        help='Configuration file for the environment',
                        required=False)
    parser.add_argument('--episodes',
                        type=int,
                        default=10,
                        choices=[10, 100],
                        help='Number of episodes')
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help='Random seed')

    parser.add_argument('--debug',
                        '-d',
                        action="store_true",
                        help='Flag to run in debug mode')
    parser.add_argument('--use_strategies',
                        '-s',
                        action="store_true",
                        help='Flag to enable strategy-based MCTS')

    args = parser.parse_args()

    return args


def init(env_config, seed, use_strategies):

    # configure agent
    config = MCTSAgentConfig()
    if use_strategies:
        config.use_strategies = True
        strategies_df = pd.read_json(STRATEGY_FILE)
        env_strategies = strategies_df[env_config.name].to_dict()

        config.strategies = env_strategies['strategies']

    # init game
    env = gym.make(env_config.name, map_file=env_config.map, config_file=env_config.filename)
    game = DiscreteGymGame(env=env)

    # set seeds
    game.reset(seed=seed)
    agent = MCTSAgent(game, config)

    agent.seed(seed=seed)
    random.seed(seed)

    return agent, game


# Function to simulate running an MCTS episode


def run_episode(agent, game):
    # Placeholder for your MCTS implementation which should return the cumulative reward for an episode.
    # For illustration, we'll simulate a random cumulative reward.
    # cumulative_reward = np.random.uniform(0, 100)  # Replace with actual MCTS episode result
    history = []
    state = game.reset()
    done = False
    info = None
    cumulative_reward = 0
    reward = 0
    agent.reset()
    while not done:
        action = agent.search(game, state, reward, done, history)
        # print(action)
        state, reward, done, info = game.step(action)
        for j in info["result_of_action"]:
            history.append((action, j))
        cumulative_reward += reward
        if done:
            print(f"Episode score: {cumulative_reward}")

    agent.root_node = None

    return cumulative_reward

# Function to evaluate cumulative return over multiple episodes


def main(args):
    # Parse arguments
    exp_id = args.exp_id
    num_episodes = args.episodes
    debug = args.debug
    env_config = {"name": args.env,
                  "mapfile": args.env_map,
                  "filename": "config_train.json"
                  }
    seed = args.seed
    use_strategies = args.use_strategies

    if not debug:
        writer = SummaryWriter(log_dir=f"exp3/{exp_id}")
    cumulative_returns = []
    agent, game = init(env_config, seed, use_strategies)
    for episode in range(num_episodes):
        reward = run_episode(agent, game)
        cumulative_returns.append(reward)
        print(f"Episode {episode + 1}: Cumulative Reward = {reward}")
        if not debug:
            writer.add_scalar("Test/mean_episode_score", np.mean(cumulative_returns) if sum(cumulative_returns) != 0 else 0, episode)

    game.close()
    # Calculate the average cumulative return over all episodes
    average_cumulative_return = np.mean(cumulative_returns)
    print(f"\nAverage Cumulative Return over {num_episodes} episodes: {average_cumulative_return}")

    return average_cumulative_return


def debug_main():
    test_args = {'exp_id': "2-test",
                 'env': 'DungeonCrawler-v0',
                 'env_map': "map.txt",
                 'seed': 123,
                 'use_strategies': True,
                 'debug': False
                 }
    # --exp_id "${exp_id}" --shaping "${shaping}" --env "${env}" --env_map "${env_map}" --timesteps "${timesteps}" -d --decay_param "${decay_param}" --decay_n 0
    main(dotdict(test_args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # average_return = main(parse_args())
    average_return = debug_main(parse_args())
