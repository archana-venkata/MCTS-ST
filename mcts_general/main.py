from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import logging
import os
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
from copy import deepcopy

import time
import cProfile
import pstats


STRATEGY_FILE = "strategies.json"
# SEEDS = [937, 732050807, 557438524, 855654600, 110433579, 19680801, 280109889, 403124237, 605551275, 19680801]

SEEDS = [732050807, 557438524, 855654600, 110433579, 19680801, 280109889, 403124237, 605551275, 19680801]


def save_params(args, dir_name):
    f = os.path.join(dir_name, "params.txt")
    os.makedirs(dir_name, exist_ok=True)
    with open(f, "a+") as f_w:
        f_w.write("\n")
        for param, val in sorted(vars(args).items()):
            f_w.write("{0}:{1}\n".format(param, val))


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
                        required=False),
    parser.add_argument('--runs',
                        type=int,
                        default=10,
                        choices=[1, 10, 100],
                        help='Number of runs')
    parser.add_argument('--episodes',
                        type=int,
                        default=1,
                        choices=[1, 10, 100],
                        help='Number of episodes (per run)')
    parser.add_argument('--debug',
                        '-d',
                        action="store_true",
                        help='Flag to run in debug mode')
    parser.add_argument('--use_strategies',
                        '-s',
                        action="store_true",
                        help='Flag to enable strategy-based MCTS')
    parser.add_argument('--sim',
                        type=int,
                        default=1,
                        choices=[1, 2],
                        help='Method to use for the MCTS simulation step')
    parser.add_argument('--exp',
                        type=int,
                        default=1,
                        choices=[1, 2],
                        help='Method to use for the MCTS expansion step')
    parser.add_argument('--backprop',
                        type=int,
                        default=1,
                        choices=[1, 2],
                        help='Method to use for the MCTS backprop step')
    parser.add_argument('--selec',
                        type=int,
                        default=1,
                        choices=[1, 2],
                        help='Method to use for the MCTS selection step')

    args = parser.parse_args()

    return args


def init(env_config, seed, use_strategies):

    # configure agent
    config = MCTSAgentConfig()
    if use_strategies:
        config.use_strategies = True
        strategies_df = pd.read_json(STRATEGY_FILE)
        env_strategies = strategies_df[env_config["name"]].to_dict()

        config.strategies = env_strategies['strategies']

    # init game
    game = gym.make(env_config["name"], map_file=env_config["mapfile"], config_file=env_config["filename"])

    # set seeds
    game.reset(seed=seed)
    agent = MCTSAgent(deepcopy(game), config)

    agent.set_seed(seed=seed)
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
    step_count = 0
    while not done:
        # t0 = time.time()
        action = agent.search(game.unwrapped.save_state(), state, reward, done, history, game.unwrapped.legal_actions())
        # t1 = time.time()
        # print(f"Time taken for MCTS search: {t1-t0}")
        state, reward, terminated, truncated, info = game.step(action)
        done = terminated or truncated
        # print(f"Step {step_count}: {action} ({game.move_count})")
        for j in info["result_of_action"]:
            history.append((action, j))
        cumulative_reward += reward
        step_count += 1

    agent.root_node = None

    return cumulative_reward

# Function to evaluate cumulative return over multiple episodes


def main(args):
    # Parse arguments
    exp_id = args.exp_id
    num_episodes = args.episodes
    num_runs = args.runs
    debug = args.debug
    env_config = {"name": args.env,
                  "mapfile": args.env_map,
                  "filename": "config_train.json"
                  }
    use_strategies = args.use_strategies

    # save the experiment parameters in a text file
    log_dir = f"results/exp3/{exp_id}"
    save_params(args, log_dir)  
    cumulative_run_returns = []
    if not debug:
        writer = SummaryWriter(log_dir=log_dir)

    for run in tqdm(range(num_runs)):
        cumulative_returns = []
        agent, game = init(env_config, SEEDS[run], use_strategies)
        t0 = time.time()

        for episode in range(num_episodes):
            reward = run_episode(agent, game)
            cumulative_returns.append(reward)
            print(f"Episode {episode + 1}: Cumulative Reward = {reward}")
        t1 = time.time()
        print(f"Time taken for a single run: {t1-t0}")

        game.close()

        # Calculate the average cumulative return over all episodes
        average_cumulative_return = np.mean(cumulative_returns)
        cumulative_run_returns.append(average_cumulative_return)
        if not debug:
            writer.add_scalar("Experiment3/mean_episode_score", average_cumulative_return if sum(cumulative_returns) != 0 else 0, episode)

        print(f"Run {run + 1}: Average Cumulative Reward = {average_cumulative_return}")

        # print(f"\nAverage Cumulative Return over {num_episodes} episodes: {average_cumulative_return}")

    # Calculate the average episode return over all runs (different seed for each run)
    metrics_avg = np.mean(cumulative_run_returns)
    metrics_std = np.std(cumulative_run_returns)
    print(f"\nMean Episode Score over {num_runs} runs: {metrics_avg} +/- {metrics_std}")

    return metrics_avg, metrics_std


def debug_main():
    test_args = {'exp_id': "test",
                 'env': 'DungeonCrawler-v0',
                 'env_map': "map.txt",
                 'runs': 1,
                 'episodes': 10,
                 'use_strategies': False,
                 'debug': True
                 }
    # --exp_id "${exp_id}" --shaping "${shaping}" --env "${env}" --env_map "${env_map}" --timesteps "${timesteps}" -d --decay_param "${decay_param}" --decay_n 0
    return main(dotdict(test_args))


def see_program_times():
    with cProfile.Profile() as pr:
        debug_main()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    # Now you have two options, either print the data or save it as a file
    stats.print_stats()  # Print The Stats
    stats.dump_stats("temp2.prof")  # Saves the data in a file, can me used to see the data visually


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # metrics_avg, metrics_std = main(parse_args())
    metrics_avg, metrics_std = debug_main()
