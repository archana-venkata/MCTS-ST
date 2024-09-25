import gym

import gym_envs

from agent import MCTSAgent
from config import MCTSAgentConfig
from game import DiscreteGymGame

from util.replay_buffer import ReplayBuffer

import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def init(env_name, seed):

    # configure agent
    config = MCTSAgentConfig()
    # init game
    game = DiscreteGymGame(env=gym.make(env_name, map_file="map.txt",
                                        config_file="config_train.json"))

    # set seeds
    game.set_seed(seed)
    agent = MCTSAgent(game, config)

    agent.seed(seed)
    random.seed(seed)

    return game, agent


def collect_trajectories():
    replay_buffer = ReplayBuffer(100000)

    saved_trajectories = []


def main():
    writer = SummaryWriter()

    game, agent = init("DungeonCrawler-v0", 123)
    episodes = 1
    scores = []
    step_count = 0
    # run a trajectory
    for i in tqdm(range(episodes)):
        history = []
        state = game.reset()
        history.append((state, None))
        done = False
        info = None
        score = 0
        reward = 0
        print(f"Episode {i}")
        while not done:
            action = agent.search(game, state, reward, done)
            # print(action)
            state, reward, done, info = game.step(action)
            step_count += 1
            for i in info["result_of_action"]:
                history.append(i)
            score += reward
            if done:
                print(f"Episode score: {score}")
                scores.append(score)
                writer.add_scalar("Test/mean_score", score, step_count)
        agent.root_node = None

    game.close()
    print(
        f"Average episode score after {episodes} episodes: {np.average(scores)}")


if __name__ == "__main__":
    main()
