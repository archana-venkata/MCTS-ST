import os
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env, spaces
import logging
import sys
import math
import getch
from torch import true_divide
import json
import time
import random
import networkx as nx

# Direction constants
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
actions_str = ["LEFT", "DOWN", "RIGHT", "UP"]
# env.unwrapped.get_action_meanings()

MAX_DISTANCE = 1000


class Weapon():
    def __init__(self, location, type) -> None:
        self.is_open = False
        self.location = location
        if type == 'g':
            self.type = 'gun'
        elif type == 's':
            self.type = 'sword'
        else:
            TypeError("Not a recognised weapon type")

    def reset(self):
        self.is_open = False

    def get_location(self):
        return self.location

    def get_type(self):
        return self.type[0]


class DungeonCrawlerEnv(Env):

    metadata = {'render.modes': ['console', 'rgb_array', 'human']}
    _max_episode_steps = 10000
    action_desc = ['move()', 'wall()', 'lock(door)', 'collect(key)', 'collect(weapon)',
                   'kill(monster)', 'unlock(door)', 'died()', 'out_of_moves()']

    def __init__(self, map_file="map.txt", config_file="config_train.json"):
        super(DungeonCrawlerEnv, self).__init__()
        self.map_file = map_file
        self.walls = []
        self.monsters = []
        self.weapons = []
        self.door = None
        self.key = None
        self.empty = []
        self.score = 0
        self.start = None
        self.end = False
        self.location = None
        self.start_configuration = {}
        self.move_count = 0
        self.max_move_count = 100
        self.items = []
        self.goal = None
        self.dead = False
        self.visited = []
        self.done = False
        if "simple" in self.map_file:
            self.total_weapon = 1
        else:
            self.total_weapon = 3

        this_file_path = os.path.dirname(os.path.realpath(__file__))
        config_filename = os.path.join(this_file_path, config_file)

        # Rewards
        # reward configuration is read from file
        with open(config_filename) as f:
            self._rewards = json.load(f)
            # self.reward_range = (0, 5, 10, 100, 200)

        self.nrow = None
        self.ncol = None
        self.nA = 4

        map_filename = os.path.join(this_file_path, self.map_file)

        with open(map_filename, "r") as f:
            for i, row in enumerate(f):
                for j, char in enumerate(row):
                    row = row.rstrip('\r\n')
                    self.ncol = len(row)
                    if char == 'X':
                        self.walls.append([i, j])
                    elif char == 'o':
                        self.start = [i, j]
                        self.location = [i, j]
                    elif char == '|':
                        self.door = [i, j]
                    elif char == ' ':
                        self.empty.append([i, j])
                    elif char == 'Z':
                        self.monsters.append([i, j])
            self.nrow = i+1

        self.start_configuration["monsters"] = self.monsters.copy()
        self.start_configuration["door"] = self.door.copy()
        self.start_configuration["empty"] = self.empty.copy()

        self.random_entity_locations(no_of_weapons=self.total_weapon)

        self.goal = self.key.copy()

        # respective actions of players : up, down, left and right
        self.action_space = spaces.Discrete(self.nA)

        # number of possible locations for the player
        # x3 for the number of items the player may have at anytime
        # self.nS = ((self.nrow * self.ncol) - (len(self.walls))) * \
        #     (len(self.weapons) + len(self.keys)+1 +
        #      (len(self.monsters)+1) + self.move_count)
        self.nS = (self.nrow * self.ncol) - (len(self.walls))

        # Create the observation space
        self.observation_space = self.get_observation_space()

        self.create_graph_rep()

    # set random locations for entities including:
    # monsters and weapons

    def random_entity_locations(self, no_of_weapons=1):
        # # randomly place the monster in one of the empty spaces in grid
        # monster_index = random.randrange(len(self.empty))
        # self.monsters.append(self.empty[monster_index])
        # self.empty.remove(self.empty[monster_index])

        # randomly place weapons in random empty spaces in grid
        for i in range(no_of_weapons):
            sword_index = random.randrange(len(self.empty))
            self.weapons.append(Weapon(self.empty[sword_index], 's'))
            self.empty.remove(self.empty[sword_index])

            gun_index = random.randrange(len(self.empty))
            self.weapons.append(Weapon(self.empty[gun_index], 'g'))
            self.empty.remove(self.empty[gun_index])

        key_index = random.randrange(len(self.empty))
        self.key = self.empty[key_index]
        self.empty.remove(self.empty[key_index])

    # Key
    # ----
    # wall = 0
    # player = 1
    # door = 2
    # key = 3
    # weapon = 4
    # blank space = 5
    # door = 6
    # An observation is list of lists, where each list represents a row

    _layers = ['X', 'o', 's', 'g', 'k', 'Z', '|', ' ']

    def valid_action_mask(self):
        row = self.location[0]
        col = self.location[1]
        possible_moves = [[row, col-1],
                          [row+1, col], [row, col+1], [row-1, col]]

        action_mask = []
        for move in possible_moves:
            value = 0
            if move not in self.walls:
                value = 1
            elif move == self.door and self.key not in self.items:
                value = 0
            elif self.visited.count(move) > 1:
                value = 0

            action_mask.append(value)

        return np.array(action_mask, dtype=bool)

    def create_graph_rep(self):
        self.create_grid()

        self.graph_rep = nx.Graph()
        for i in range(self.nrow):
            for j in range(self.ncol):
                if self.grid[i][j] != 'X':
                    self.graph_rep.add_node((i, j))
                    if i > 0:
                        self.graph_rep.add_edge((i-1, j), (i, j))
                    if j > 0:
                        self.graph_rep.add_edge((i, j), (i, j-1))

    def get_observation_space(self):

        return spaces.Box(low=0, high=max(self.max_move_count, self.max_move_count), shape=(4,), dtype=np.int32)

    def make_observation(self):
        self.create_grid()

        board = [[self._layers.index(y) for y in x] for x in self.grid]

        distance_to_monster = [0 if len(self.monsters) == 0 else min(
            [self.distance_to(self.location, m) for m in self.monsters])]

        distance_to_weapon = [0 if len(self.weapons) == 0 else min(
            [self.distance_to(self.location, w.get_location()) for w in self.weapons])]

        distance_to_key = [0 if self.key ==
                           None else self.distance_to(self.location, self.key)]

        distances = [distance_to_weapon[0], distance_to_monster[0],
                     distance_to_key[0], self.distance_to(self.location, self.door)]

        obs = np.asarray(distances, dtype=np.int32)

        return obs

    def make_info(self, result_of_action=[], rewards_per_action=[]):
        info = {
            "result_of_action": result_of_action,
            "rewards_per_action": rewards_per_action,
            "score": self.score
        }
        return info

    def distance_to(self, pos1, pos2):
        if pos1 == None or pos2 == None:
            return 0
        return nx.shortest_path_length(self.graph_rep, tuple(pos1), tuple(pos2))

    def step(self, action):
        # player moves based on a given action
        logging.debug(f'Moved {actions_str[action]}')

        terminated = False
        info = {}
        reward = self._rewards["default"]
        rewards = []
        results_of_action = []

        # save the player's location prior to taking the action
        row = self.location[0]
        col = self.location[1]
        self.visited.append([row, col])

        self.move_count += 1
        if self.move_count < self.max_move_count:
            # depending on the action, update the player's location
            # if there is a wall first, no movement
            if action == LEFT and [row, col-1] not in self.walls:
                self.location = [row, col-1]
            elif action == DOWN and [row+1, col] not in self.walls:
                self.location = [row+1, col]
            elif action == RIGHT and [row, col+1] not in self.walls:
                self.location = [row, col+1]
            elif action == UP and [row-1, col] not in self.walls:
                self.location = [row-1, col]

            # print(
            #     f'distance to goal: {self.distance_to(self.location, self.goal)}')

            # check if the player has reached a door
            if self.location == self.door:
                # WIN!!
                if 'key' in self.items:
                    self.door = None
                    self.empty.append([row, col])
                    reward += self._rewards["door"]
                    terminated = True
                    self.items.remove('key')  # one key has been used
                    rewards.append(reward)
                    self.end = True
                    results_of_action.append(self.action_desc[6])
                else:
                    # if the player has reached the door but doesn't have a key,
                    # the player goes back to its previous location
                    self.location = [row, col]
                    reward -= self._rewards["wall"]
                    results_of_action.append(self.action_desc[2])
                    rewards.append(reward)
        else:
            # Stop if exceeded the amount of allowed moves
            terminated = True
            results_of_action.append(self.action_desc[8])
            reward -= self._rewards["out of moves"]
            rewards.append(reward)

        if not terminated:
            # if the player has moved, add to the list of empty spaces
            if self.location != [row, col]:

                # moved closer to the goal + reward
                # moved away from the goal - reward
                distance = self.distance_to([row, col], self.goal)
                new_distance = self.distance_to(self.location, self.goal)
                if new_distance < distance:
                    reward += self._rewards["moved"]
                else:
                    reward -= self._rewards["moved"]

                self.empty.append([row, col])
                results_of_action.append(self.action_desc[0])
                rewards.append(reward)
            else:
                # If a player didnt move it either means it hit a wall or a closed door.
                # if the player did not hit a closed door
                if self.action_desc[2] not in results_of_action:
                    results_of_action.append(self.action_desc[1])
                    reward -= self._rewards["wall"]
                    rewards.append(reward)

            # if the player's current location used to be empty, remove from empty list
            if self.location in self.empty:
                self.empty.remove(self.location)

            # if there was a key at the new location, player collects the key
            # a key can be used when the player encounters a door
            if self.location == self.key:
                self.key = None
                self.items.append('key')
                # after collecting the key, the new goal is to reach the door
                self.goal = self.door.copy()
                reward += self._rewards["key"]  # reward for collecting an item
                results_of_action.append(self.action_desc[3])
                rewards.append(reward)

            # if there was a weapon at the new location, player collects the weapon
            # a weapon can be used when the player encounters a monster
            for weapon in self.weapons:
                if self.location == weapon.get_location():
                    self.weapons.remove(weapon)
                    # reward for collecting a weapon
                    reward += self._rewards["weapon"]
                    self.items.append('weapon')

                    results_of_action.append(
                        self.action_desc[4].replace("weapon", weapon.type))

                    rewards.append(reward)

            # If there is a monster at the new location, player either
            # loses a life if the monster cannot be killed or
            # loses a life if player has no collected any weapons
            # gains 100 if the monster can be killed and the player has a weapon
            if self.location in self.monsters:
                if 'weapon' in self.items:
                    reward += self._rewards["kill"]
                    results_of_action.append(self.action_desc[5])
                    rewards.append(reward)
                    self.items.remove('weapon')

                    # the new goal is to collect the key
                    self.monsters.remove(self.location)
                else:
                    # if there is a monster, and the monster cannot die
                    # or player does not have any weapons
                    # game over
                    self.dead = True
                    terminated = True
                    reward -= self._rewards["died"]
                    results_of_action.append(self.action_desc[7])
                    rewards.append(reward)

        self.score += reward
        info = self.make_info(results_of_action, rewards)

        # For debugging: output the result of the action taken by the player
        logging.debug(f'Results of action: {results_of_action}')

        # update the observation
        observation = self.make_observation()

        self.done = terminated

        return observation, reward, terminated, False, info

    # Used for rendering the environment in the console
    # creates a 2d array with characters to represent the entities in the environment
    def create_grid(self):
        self.grid = np.zeros((self.nrow, self.ncol), str)

        for i, j in self.walls:
            self.grid[i][j] = 'X'
        for i, j in self.monsters:
            self.grid[i][j] = 'Z'
        for i, j in self.empty:
            self.grid[i][j] = ' '

        if self.key != None:
            i_key, j_key = self.key
            self.grid[i_key][j_key] = 'k'

        if self.door != None:
            i_door, j_door = self.door
            self.grid[i_door][j_door] = '|'

        for weapon in self.weapons:
            i, j = weapon.get_location()
            self.grid[i][j] = weapon.get_type()

        # do not display the player if it has been killed
        if not self.dead:
            self.grid[self.location[0]][self.location[1]] = 'o'

    # reset the environment to its initial state
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # reset the dots, powerups and empty spaces
        self.monsters = self.start_configuration["monsters"].copy()
        self.door = self.start_configuration["door"].copy()
        self.empty = self.start_configuration["empty"].copy()

        # player goes back to starting position
        self.location = self.start.copy()
        self.dead = False
        self.end = False
        self.score = 0
        self.move_count = 0
        self.items = []
        self.weapons = []
        self.key = None
        self.visited = []
        self.done = False

        # randomly place items in empty spaces in grid
        self.random_entity_locations(no_of_weapons=self.total_weapon)

        self.goal = self.key.copy()

        observation = self.make_observation()
        info = self.make_info()

        return observation, info

    # Functions to allow the state of the environment to be saved and reloaded (for MCTS simulation)
    def save_state(self, seed=None):
        state_dict = {"seed": seed,
                      "monsters": self.monsters,
                      "door": self.door,
                      "empty": self.empty,
                      "location": self.location,
                      "dead": self.dead,
                      "end": self.end,
                      "score": self.score,
                      "move_count": self.move_count,
                      "items": self.items,
                      "weapons": self.weapons,
                      "key": self.key,
                      "visited": self.visited,
                      "done": self.done,
                      "goal": self.goal
                      }
        return state_dict

    def load_state(self, state_dict):
        super().reset(seed=state_dict["seed"])

        self.monsters = state_dict["monsters"]
        self.door = state_dict["door"]
        self.empty = state_dict["empty"]
        self.location = state_dict["location"]
        self.dead = state_dict["dead"]
        self.end = state_dict["end"]
        self.score = state_dict["score"]
        self.move_count = state_dict["move_count"]
        self.items = state_dict["items"]
        self.weapons = state_dict["weapons"]
        self.key = state_dict["key"]
        self.visited = state_dict["visited"]
        self.done = state_dict["done"]
        self.goal = state_dict["goal"]

    # render the environment

    def render(self, mode="console"):
        if mode not in self.metadata["render.modes"]:
            raise ValueError(f'Received unexpected render mode {mode}')

        self.create_grid()
        for row in self.grid:
            # if render mode = terminal
            print(*row)
            # if render mode = human

        print(f'Score: {self.score}')
        print(f'Number of Moves: {self.move_count}')

        if self.dead or self.move_count == self.max_move_count:
            print("---------------- GAME OVER ----------------")
        elif self.end:
            print("**************** WINNER!! ****************")

    def get_grid(self):
        return self.grid

    def close(self):
        pass

    def getPosFromPath(self, path, pos):
        row = pos[0]
        col = pos[1]
        for action in path:
            if action == LEFT and [row, col-1] not in self.walls:
                col -= 1
            elif action == DOWN and [row+1, col] not in self.walls:
                row += 1
            elif action == RIGHT and [row, col+1] not in self.walls:
                col += 1
            elif action == UP and [row-1, col] not in self.walls:
                row -= 1

        return [row, col]


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

    env = DungeonCrawlerEnv()
    observation = env.reset()

    num_episodes = 10
    print(f"size of environment is: {env.nrow} x {env.ncol}")
    # testing the environment by playing a number of episodes
    for episode in range(num_episodes):
        observation = env.reset()

        # render the environment at the beginning of each episode
        env.render()
        print('')

        terminated = False
        score = 0

        frames = []
        fps = 24

        # play a certain number of moves for each episode
        while not terminated:
            char = getch.getch()
            action = int(char)
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward

            env.render()
            print()
            frames.append(env.get_grid())

        for frame in frames:
            print('\n'.join(' '.join(x for x in y) for y in frame))
            time.sleep(0.1)
            os.system('clear')

        # and print the final score
        print('Episode: {}, Score: {}'.format(episode, score))
        print('')
