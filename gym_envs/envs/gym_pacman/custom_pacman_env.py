from distutils.command.config import config
import os
import matplotlib.pyplot as plt
import numpy as np
from gym import Env, spaces
import logging
import sys
import math
import json
import getch
import networkx as nx
import random


# actions available
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
actions_str = ["LEFT", "DOWN", "RIGHT", "UP"]
# env.unwrapped.get_action_meanings()


class Ghost():
    def __init__(self, location, map_file) -> None:
        self._can_die = False
        self._location = location
        self._start = location
        if "simple" in map_file:
            self._can_die_moves = 5
        else:
            self._can_die_moves = 10

        self._moves_left = self._can_die_moves

        self._action_probs = [0.4, 0.1, 0.4, 0.1]
        self._prev_location = None
        self._can_move = True

    def get_location(self):
        return self._location

    def get_start(self):
        return self._start

    def get_can_die(self):
        return self._can_die

    def get_prev_location(self):
        return self._prev_location

    def reset(self):
        self._can_die = False
        self._location = self._start.copy()
        self._moves_left = self._can_die_moves
        self._action_probs = [0.4, 0.1, 0.4, 0.1]
        self._prev_location = None
        self._can_move = True

    def move(self, walls, pacman_location, graph):

        action = None
        row = self._location[0]
        col = self._location[1]
        moves = ([[row, col-1],
                  [row+1, col], [row, col+1], [row-1, col]])
        action_mask = np.array([move not in walls for move in moves])
        moves_arr = np.array(moves)
        valid_moves = moves_arr[action_mask].tolist()

        # for each possible direction, calculate the shortest path to pacman
        distances = [nx.shortest_path_length(graph, tuple(pacman_location), tuple(valid_move))
                     for valid_move in valid_moves]

        # if the ghost can die, it moves randomly
        # else the ghost cannot die, run towards pacman
        if self._can_die:
            # ghosts move slower when they can die
            # only move once in 2 moves
            if self._can_move:
                self._moves_left -= 1

                action = moves.index(random.choice(moves))
        else:
            min_dist_index = distances.index(min(distances))
            action = moves.index(valid_moves[min_dist_index])

        if action == LEFT and [row, col-1] not in walls:
            self._location = [row, col-1]
        elif action == DOWN and [row+1, col] not in walls:
            self._location = [row+1, col]
        elif action == RIGHT and [row, col+1] not in walls:
            self._location = [row, col+1]
        elif action == UP and [row-1, col] not in walls:
            self._location = [row-1, col]

        if self._can_die:
            if self._moves_left == 0:
                self._can_die = False
                self._moves_left = self._can_die_moves
            else:
                self._can_move = not self._can_move

        self._prev_location = [row, col]
        logging.debug(f'Ghost moved')

    def set_can_die(self):
        self._can_die = True
        self._moves_left = self._can_die_moves


class PacmanEnv(Env):

    _metadata = {"render_modes": ["human"]}
    _max_episode_steps = 200
    _env_features = ['moved', 'wall', 'dot', 'powerup',
                     'killed a ghost', 'lost a life']

    def __init__(self, map_file="map.txt", config_file="config_train.json"):
        self.map_file = map_file
        self._ghosts = []
        self._walls = []
        self._dots = []
        self._powerups = []
        self._empty = []
        self._score = 0
        self._lives = 2
        self._start = None
        self._location = None
        self._successive_kills = [0, 5]
        self._start_configuration = {}
        self._dead = False
        self._max_move_count = 1000

        this_file_path = os.path.dirname(os.path.realpath(__file__))
        config_filename = os.path.join(this_file_path, config_file)

        # reward configuration is read from file
        with open(config_filename) as f:
            self._rewards = json.load(f)
            # self.reward_range = (0, 10, 50, 200, 400, 600, 800)

        self._nrow = None
        self._ncol = None
        self._nA = 4

        map_filename = os.path.join(this_file_path, self.map_file)

        with open(map_filename, "r") as f:
            for i, row in enumerate(f):
                for j, char in enumerate(row):
                    row = row.rstrip('\r\n')
                    self._ncol = len(row)
                    if char == 'M':
                        self._ghosts.append(Ghost([i, j], self.map_file))
                    elif char == 'C':
                        self._start = [i, j]
                        self._location = self._start.copy()
                    elif char == '#':
                        self._walls.append([i, j])
                    elif char == 'O':
                        self._powerups.append([i, j])
                    elif char == '.':
                        self._dots.append([i, j])
                    elif char == ' ':
                        self._empty.append([i, j])
            self._nrow = i+1

        self._start_configuration["dots"] = self._dots.copy()
        self._start_configuration["powerups"] = self._powerups.copy()
        self._start_configuration["empty"] = self._empty.copy()

        # number of possible positions pacman could be in
        self._nS = self._nrow*self._ncol

        # respective actions of agents : up, down, left and right
        self.action_space = spaces.Discrete(self._nA)

        # TODO: Work out what type of observation space works best for this env
        # set the observation space to the location pacman could be in times the number of dots left in the grid
        # plus 1 to account for the 'dead' state
        self.observation_space = self.get_observation_space()

        self.create_graph_rep()

    # Key
    # ----
    # wall = 0
    # agent (pacman) = 1
    # dots = 2
    # powerups = 3
    # ghost = 4
    # ghost (transparent) = 5
    # blank space = 6
    # An observation is list of lists, where each list represents a row
    #
    # [[0 0 0 0 0 0 0 0 0 0]
    #  [0 4 2 2 0 2 2 4 2 0]
    #  [0 2 2 2 2 1 2 2 2 0]
    #  [0 4 2 2 2 3 2 0 2 0]
    #  [0 2 2 2 2 2 2 2 2 0]
    #  [0 0 0 0 0 0 0 0 0 0]]]

    _layers = ['#', 'C', '.', 'O', 'M', 'W', ' ']

    def valid_action_mask(self):
        row = self._location[0]
        col = self._location[1]
        possible_moves = [[row, col-1],
                          [row+1, col], [row, col+1], [row-1, col]]
        return np.array([int(move not in self._walls) for move in possible_moves], dtype=np.bool)

    def create_graph_rep(self):
        self.create_grid()

        self.graph_rep = nx.Graph()
        for i in range(self._nrow):
            for j in range(self._ncol):
                if self.grid[i][j] != '#':
                    self.graph_rep.add_node((i, j))
                    if i > 0:
                        self.graph_rep.add_edge((i-1, j), (i, j))
                    if j > 0:
                        self.graph_rep.add_edge((i, j), (i, j-1))

    def get_observation_space(self):
        return spaces.Box(low=0, high=len(self._layers)-1, shape=(self._nrow, self._ncol), dtype=np.uint8)

    def make_observation(self):
        self.create_grid()
        board = [[self._layers.index(y) for y in x] for x in self.grid]

        obs = np.asarray(board, dtype=np.uint8)

        return obs

    def distance_to(self, pos1, pos2):
        if pos1 == None or pos2 == None:
            return 0
        return nx.shortest_path_length(self.graph_rep, tuple(pos1), tuple(pos2))

    def get_grid(self):
        return self.grid

    def obs2grid(self, obs):
        reconstructed_grid = np.zeros((self._nrow, self._ncol))
        for index, board in enumerate(obs):
            reconstructed_grid += board * index

        return np.vectorize(self.int_to_layer)(reconstructed_grid)

    def int_to_layer(self, index):
        return self._layers[int(index)]

    def step(self, action):
        # pacman moves based on a given action
        # ghosts moves are somewhat random
        logging.debug(f'Pacman moved {actions_str[action]}')

        # managing the case when pacman can kill ghosts
        # and it gets rewarded for successive kills
        if self._successive_kills[0] > 0 and self._successive_kills[1] > 0:
            self._successive_kills[1] -= 1
        elif self._successive_kills[1] == 0:
            self._successive_kills = [0, 5]

        done = False
        info = {}
        reward = self._rewards["default"]
        rewards = []
        result_of_action = []

        # update the ghosts positions
        for ghost in self._ghosts:
            row = ghost.get_location()[0]
            col = ghost.get_location()[1]
            ghost.move(self._walls, self._location, self.graph_rep)
            # if the ghost has moved and its previous location is empty but not in empty, add it to empty
            if ghost.get_location() != [row, col] and [row, col] not in self._empty and [row, col] not in self._dots and [row, col] not in self._powerups:
                self._empty.append([row, col])

        # save the agent's location prior to taking the action
        row = self._location[0]
        col = self._location[1]

        # depending on the action, update the agent's location
        # if there is a wall first, no movement

        if action == LEFT and [row, col-1] not in self._walls:
            self._location = [row, col-1]
        elif action == DOWN and [row+1, col] not in self._walls:
            self._location = [row+1, col]
        elif action == RIGHT and [row, col+1] not in self._walls:
            self._location = [row, col+1]
        elif action == UP and [row-1, col] not in self._walls:
            self._location = [row-1, col]

        # if the agent has moved, add to the list of empty spaces
        if self._location != [row, col]:
            self._empty.append([row, col])
            result_of_action.append(self._env_features[0])
            rewards.append(reward)
        else:
            # The agent hit a wall and did not move
            result_of_action.append(self._env_features[1])
            rewards.append(reward)

        # if the agent's current location used to be empty, remove from empty list
        if self._location in self._empty:
            self._empty.remove(self._location)

        # if there was a dot at the new location, pacman eats the dot
        if self._location in self._dots:
            self._dots.remove(self._location)
            # reward for collecting a regular dot
            reward = self._rewards["dot"]
            result_of_action.append(self._env_features[2])
            rewards.append(reward)

        # if there was a powerup at the new location, pacman eats the dot
        if self._location in self._powerups:
            self._powerups.remove(self._location)
            # reward for collecting a powerup
            reward = self._rewards["powerup"]
            result_of_action.append(self._env_features[3])
            rewards.append(reward)

            # For the next 5 moves, ghosts can be killed
            self._successive_kills = [0, 5]
            for ghost in self._ghosts:
                ghost.set_can_die()

        # If there is a ghost at the new location, pacman either
        # loses a life if the ghost cannot be killed or
        # gains 200 if the ghost can be killed
        ghostsAtLocation = []
        for ghost in self._ghosts:
            if ghost.get_location() == self._location:
                ghostsAtLocation.append(ghost)
            elif ghost.get_prev_location() == self._location and ghost.get_location() == [row, col]:
                # if the ghost and pacman pass each other (swap locations)
                ghostsAtLocation.append(ghost)

        for ghost in ghostsAtLocation:
            if ghost.get_can_die():
                self._successive_kills[0] += 1
                reward = self._rewards["kill"]*self._successive_kills[0]
                result_of_action.append(self._env_features[4])
                rewards.append(reward)
                ghost.reset()
            else:
                # if there is a ghost, and the ghost cannot die
                # lose a life and depending on lives remaining
                self._lives -= 1
                result_of_action.append(self._env_features[5])
                rewards.append(reward)

                if self._lives != 0:
                    # send ghosts back to their start position
                    for ghost in self._ghosts:
                        ghost.reset()
                    # update the empty list
                    if self._location not in self._empty:
                        self._empty.append(self._location)
                    # send pacman back to start position
                    self._location = self._start.copy()
                else:
                    # pacman is out of lives
                    # reset
                    self._dead = True
                    done = True
                break

        self._score += reward
        info = {
            "result_of_action": result_of_action,
            "rewards_per_action": rewards,
            "score": self._score
        }

        # For debugging: output the result of the action taken by the agent
        logging.debug(f'Result of action: {result_of_action}')

        # YOU'VE WON!
        if (len(self._dots)+len(self._powerups)) == 0:
            done = True

        # update the observation
        observation = self.make_observation()

        return observation, reward, done, info

    # Used for rendering the environment in the console
    # creates a 2d array with characters to represent the entities in the environment
    def create_grid(self):
        self.grid = np.zeros((self._nrow, self._ncol), str)

        for i, j in self._walls:
            self.grid[i][j] = '#'
        for i, j in self._dots:
            self.grid[i][j] = '.'
        for i, j in self._powerups:
            self.grid[i][j] = 'O'
        for i, j in self._empty:
            self.grid[i][j] = ' '
        for ghost in self._ghosts:
            i, j = ghost.get_location()
            # use a different character when the ghosts are in a state where they can be killed
            if ghost.get_can_die():
                self.grid[i][j] = 'W'
            else:
                self.grid[i][j] = 'M'

        # do not display the agent if it has been killed
        if not self._dead:
            self.grid[self._location[0]][self._location[1]] = 'C'
        elif self.grid[self._location[0]][self._location[1]] == '':
            self.grid[self._location[0]][self._location[1]] = ' '

    # reset the environment to its initial state
    def reset(self):
        # reset the dots, powerups and empty spaces
        self._dots = self._start_configuration["dots"].copy()
        self._powerups = self._start_configuration["powerups"].copy()
        self._empty = self._start_configuration["empty"].copy()

        # pacman goes back to starting position
        self._location = self._start.copy()
        self._successive_kills = [0, 5]
        self._dead = False
        self._score = 0
        self._lives = 2

        # ghosts go back to their starting positions
        for ghost in self._ghosts:
            ghost.reset()

        observation = self.make_observation()

        return observation

    # render the environment
    def render(self, mode="human"):
        if mode not in self._metadata["render_modes"]:
            raise ValueError(f'Received unexpected render mode {mode}')

        self.create_grid()
        for row in self.grid:
            print(*row)

        print(f'Score: {self._score}')
        print(f'Lives: {"C"*self._lives}')

        if self._lives == 0:
            print("---------------- GAME OVER ----------------")
        elif (len(self._dots)+len(self._powerups)) == 0:
            print("**************** WINNER!! ****************")

    def close(self):
        pass

    def seed(self, seed=None):
        return super().seed(seed)


def output_to_file():
    # helper function to redirect stdout to file
    orig_stdout = sys.stdout
    file = open('temp_output/out.txt', 'w')
    sys.stdout = file
    return file, orig_stdout


def return_output_to_console(file, orig_stdout):
    sys.stdout = orig_stdout
    file.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.debug('This is a log message.')

    env = PacmanEnv()

    seed = 4312
    env.seed(seed)

    num_episodes = 1
    num_moves = 10
    print()
    episode_trajectory = []
    # testing the environment by playing a number of episodes
    for episode in range(num_episodes):
        observation = env.reset()

        # render the environment at the beginning of each episode
        env.render()
        print('')

        done = False
        score = 0

        # play a certain number of moves for each episode
        while not done:
            # char = getch.getch()
            # action = int(char)
            # observation, reward, done, info = env.step(action)
            observation, reward, done, info = env.step(
                env.action_space.sample())
            score += reward

            env.render()
            print(info["result_of_action"])
            print()

        # render the environment at the end of the episode
        # and print the final score
        env.render()
        print(f'Episode: {episode}, Score: {score}')
        print('')
