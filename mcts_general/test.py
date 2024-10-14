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


def init(env_name, seed):

    # configure agent
    config = MCTSAgentConfig()
    if config.use_strategies:
        strategies_df = pd.read_json('strategies.json')
        env_strategies = strategies_df[env_name].to_dict()

        config.strategies = env_strategies['strategies']

    # init game
    env = gym.make(env_name, map_file="map.txt", config_file="config_train.json")
    game = DiscreteGymGame(env=env)

    # set seeds
    game.reset(seed=seed)
    agent = MCTSAgent(game, config)

    agent.seed(seed=seed)
    random.seed(seed)

    return game, agent


def main():
    debug = True
    if not debug:
        writer = SummaryWriter()

    game, agent = init("DungeonCrawler-v0", 123)
    episodes = 1
    scores = []
    step_count = 0
    # run a trajectoryxP
    for i in tqdm(range(episodes)):
        history = []
        state = game.reset()
        done = False
        info = None
        score = 0
        reward = 0
        print(f"Episode {i}")
        while not done:
            action = agent.search(game, state, reward, done, history)
            # print(action)
            state, reward, done, info = game.step(action)
            step_count += 1
            for j in info["result_of_action"]:
                history.append(j)
            score += reward
            if done:
                print(f"Episode score: {score}")
                scores.append(score)
                if not debug:
                    writer.add_scalar("Test/mean_episode_score", np.mean(scores) if sum(scores) != 0 else 0, i)
        agent.root_node = None

    game.close()
    print(
        f"Average episode score after {episodes} episodes: {np.average(scores)}")


if __name__ == "__main__":
    main()
