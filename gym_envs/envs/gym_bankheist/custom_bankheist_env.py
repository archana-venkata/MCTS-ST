import os
import numpy as np
from gym import Env, spaces
import logging
import json
import random
import networkx as nx

# Direction constants
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
DYNAMITE = 4
actions_str = ["LEFT", "DOWN", "RIGHT", "UP", "DYNAMITE"]


class Dynamite():
    def __init__(self, location) -> None:
        self._time = 5
        self._location = location

    def deduct_time(self):
        self._time -= 1

    def get_time(self):
        return self._time

    def set_location(self, new_location):
        self._location = new_location

    def get_location(self):
        return self._location


class PoliceCar():
    def __init__(self, location) -> None:
        self._location = location
        self._start = location
        self._prev_location = None
        self._destroyed = False
        self._time_to_active = 5

    def set_location(self, new_location):
        self._location = new_location

    def get_location(self):
        return self._location

    def get_start(self):
        return self._start

    def get_destroyed(self):
        return self._destroyed

    def get_prev_location(self):
        return self._prev_location

    def get_time_to_active(self):
        return self._time_to_active

    def reset(self):
        self._destroyed = False
        self._location = self._start.copy()
        self._prev_location = None
        self._time_to_active = 5

    def move(self, walls, agent_location, dynamite):
        if self._time_to_active != 0:
            self._time_to_active -= 1
            return 2

        row = self._location[0]
        col = self._location[1]
        player = np.array(agent_location)
        dists = []
        # this loop ensures the car position does change and that it moves closer to the player
        new_positions = [[row, col-1],
                         [row+1, col],
                         [row, col+1],
                         [row-1, col]]

        for i in range(0, 4):
            dists.append(np.linalg.norm(player-np.array(new_positions[i])))

        while self._location == [row, col]:
            min_loc_index = dists.index(min(dists))
            if new_positions[min_loc_index] not in walls:
                self._location = new_positions[min_loc_index]
            else:
                dists.remove(dists[min_loc_index])
                new_positions.remove(new_positions[min_loc_index])

        self._prev_location = [row, col]

        if self._location in dynamite:
            self._destroyed = True
            dynamite.remove(self._location)
            logging.debug(f'Police car blown up!')
            return 1
        logging.debug(f'Police car moved')

        return 0


class BankHeistEnv(Env):

    metadata = {'render.modes': ['console', 'rgb_array', 'human']}
    _max_episode_steps = 10000
    action_desc = ['moved', 'wall', 'rob bank', 'drop dynamite', 'blown up',
                   'destroy policecar', 'collect fueltank', 'refuel', 'caught', 'out of fuel']

    def __init__(self, map_file="map.txt", config_file="config_train.json"):
        super(BankHeistEnv, self).__init__()
        self.map_file = map_file
        self.walls = []
        self.policecars = []
        self.fueltanks = []
        self.dynamite = []
        self.banks = []
        self.empty = []
        self.score = 0
        self.start = None
        self.location = None
        self.start_configuration = {}
        self.max_fuel = 50
        self.fuel = self.max_fuel
        self.max_banks = 5
        self.banks_robbed = 0
        self.dead = False
        self.action_history = []
        self.visited = []

        self.nA = 5  # no. of actions available

        # Rewards
        # Reward configuration is read from a specified JSON file
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        config_filename = os.path.join(this_file_path, config_file)

        with open(config_filename) as f:
            self._rewards = json.load(f)

        # Maze map

        # Size of the map
        self.nrow = None
        self.ncol = None

        # Read map layout from text file
        map_filename = os.path.join(this_file_path, self.map_file)

        with open(map_filename, "r") as f:
            for i, row in enumerate(f):
                for j, char in enumerate(row):
                    row = row.rstrip('\r\n')
                    self.ncol = len(row)
                    if char == '#':
                        self.walls.append([i, j])
                    elif char == 'C':
                        self.start = [i, j]
                        self.location = [i, j]
                    elif char == 'F':
                        self.fueltanks.append([i, j])
                    elif char == 'B':
                        self.banks.append([i, j])
                    elif char == ' ':
                        self.empty.append([i, j])
            self.nrow = i+1

        self.start_configuration["fueltanks"] = self.fueltanks.copy()
        self.start_configuration["banks"] = self.banks.copy()
        self.start_configuration["empty"] = self.empty.copy()

        # respective actions of players : up, down, left and right
        self.action_space = spaces.Discrete(self.nA)

        # number of possible locations for the player
        self.nS = self.nrow * self.ncol

        # Create the observation space
        self.observation_space = self.get_observation_space()

        self.create_graph_rep()

    # Key
    # ----
    # player = 0
    # bank = 1
    # fuel tank = 2
    # police car = 3
    # dynamite = 4
    # wall = 5
    # blank space = 6

    _layers = ['C', 'B', 'F', 'p', 'd', '#', ' ']

    def valid_action_mask(self):
        row = self.location[0]
        col = self.location[1]
        possible_moves = [[row, col-1],
                          [row+1, col], [row, col+1], [row-1, col]]

        valid_action_array = []
        for move in possible_moves:
            if move not in self.walls:
                value = 1
            else:
                value = 0

            if self.visited[-5:].count(move) > 3:
                value = 0

            valid_action_array.append(value)

        # dropping dynamite only makes sense if there are policecars and no dynamite already in place
        if len(self.policecars) > 0:
            valid_action_array.append(1)
        else:
            valid_action_array.append(0)

        return valid_action_array

    def create_graph_rep(self):
        self.create_grid()

        self.graph_rep = nx.Graph()
        for i in range(self.nrow):
            for j in range(self.ncol):
                if self.grid[i][j] != '#':
                    self.graph_rep.add_node((i, j))
                    if i > 0:
                        self.graph_rep.add_edge((i-1, j), (i, j))
                    if j > 0:
                        self.graph_rep.add_edge((i, j), (i, j-1))

    def get_observation_space(self):
        return spaces.Box(low=-1, high=5, shape=(5, ), dtype=np.int32)

    def make_observation(self):
        self.create_grid()

        if self.banks_robbed == self.max_banks:
            values = [0] * 5
        else:
            row = self.location[0]
            col = self.location[1]
            possible_moves = [[row, col-1],
                              [row+1, col], [row, col+1], [row-1, col], None]

            distance_to_bank = min(
                [self.distance_to(self.location, b) for b in self.banks])

            # distance to nearest policecar
            distance_to_policecar = [0 if len(self.policecars) == 0 else min(
                [self.distance_to(self.location, p.get_location()) for p in self.policecars])]

            # distance to nearest dynamite
            distance_to_dynamite = [0 if len(self.dynamite) == 0 else min(
                [self.distance_to(self.location, d.get_location()) for d in self.dynamite])]

            values = []
            # for each available move, indicate which will yield the best result if taken next
            for i in range(5):
                value = 0
                move = possible_moves[i]
                if move:
                    if move in self.walls:
                        value += -1
                    else:
                        # gained fuel when the fuel tank below 50%??
                        if move in self.fueltanks and self.fuel < (0.5*self.max_fuel):
                            logging.debug("gained fuel when fuel is low")
                            value += 1

                        # moved closer to bank?
                        new_distance_to_bank = min(
                            [self.distance_to(move, b) for b in self.banks])
                        if new_distance_to_bank <= distance_to_bank:
                            logging.debug("moved closer to bank")
                            value += 1

                        # moved away from policecars?
                        new_distance_to_policecar = [0 if len(self.policecars) == 0 else min(
                            [self.distance_to(move, p.get_location()) for p in self.policecars])]
                        if new_distance_to_policecar > distance_to_policecar:
                            logging.debug("moved away from policecars")
                            value += 1

                        # moved away from dynamite?
                        new_distance_to_dynamite = [0 if len(self.dynamite) == 0 else min(
                            [self.distance_to(move, d.get_location()) for d in self.dynamite])]
                        if new_distance_to_dynamite > distance_to_dynamite:
                            logging.debug("moved away from dynamite")
                            value += 1

                else:
                    # drop dynamite
                    # check the distance between the new dynamite to policecars
                    if len(self.policecars) == 0 or len(self.dynamite) >= len(self.policecars) or self.fuel < (0.2*self.max_fuel):
                        value += -1
                    elif len(self.dynamite) > 0 and any(d.get_location() == [row, col] for d in self.dynamite):
                        value += -1
                    else:
                        distance_to_policecar = min(
                            [self.distance_to(self.location, p.get_location()) for p in self.policecars])
                        # is dynamite within reach of a policecar?
                        if distance_to_policecar < 5 and distance_to_policecar > 0:
                            logging.debug(
                                "dynamite is within reach of a policecar")
                            value += 5

                values.append(value)

        obs = np.asarray(values, dtype=np.int32)

        return obs

    def distance_to(self, pos1, pos2):
        if pos1 == None or pos2 == None:
            return 0
        return nx.shortest_path_length(self.graph_rep, tuple(pos1), tuple(pos2))

    def append_reward(self, action_desc, results_of_action, rewards, multiplier=1):
        reward = self._rewards[action_desc]*multiplier
        results_of_action.append(action_desc)
        rewards.append(reward)

    def step(self, action):
        # player moves based on a given action
        # ghosts moves are somewhat random
        logging.debug(f'Chosen action: {actions_str[action]}')
        logging.debug(f'Bank locations: {self.banks}')
        self.action_history.append(actions_str[action])

        done = False
        info = {}
        reward = self._rewards["default"]
        rewards = []
        results_of_action = []

        self.fuel -= 1
        # any dynamite
        dynamite_locations = []
        for d in self.dynamite:
            d.deduct_time()
            if d.get_time() == 0:
                self.dynamite.remove(d)
                self.empty.append(d.get_location())
            else:
                dynamite_locations.append(d.get_location())

        # save the player's location prior to taking the action
        row = self.location[0]
        col = self.location[1]
        self.visited.append([row, col])
        logging.debug(self.visited)

        # if at least one bank has been robbed
        # update the policecars positions
        # check if any cars have been blown up by dynamite placed by the player
        if self.banks_robbed > 0:
            cars_move_results = [policecar.move(
                self.walls, self.location, dynamite_locations) for policecar in self.policecars]
            cars_blown_up = 0
            for index, result in enumerate(cars_move_results):
                # remove the policecars location from the empty spaces
                if self.policecars[index].get_location() in self.empty:
                    self.empty.remove(self.policecars[index].get_location())

                if result == 2:
                    logging.debug(
                        f"car at {self.policecars[index].get_location()} is in-active")
                elif result == 1:
                    cars_blown_up += 1
                    self.empty.append(self.policecars[index].get_location())
                    self.append_reward(
                        "destroy policecar", results_of_action, rewards, multiplier=cars_blown_up)

                # if the policecar has moved, its previous location is empty but not in empty, add it to empty
                if self.policecars[index].get_prev_location():
                    car_row = self.policecars[index].get_prev_location()[0]
                    car_col = self.policecars[index].get_prev_location()[1]
                    if self.policecars[index].get_location() != [car_row, car_col] and [car_row, car_col] not in self.empty:
                        if [car_row, car_col] not in self.banks and [car_row, car_col] not in self.fueltanks:
                            self.empty.append([car_row, car_col])

            self.policecars = [
                car for car in self.policecars if not car.get_destroyed()]

        if self.fuel > 0:

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
            elif action == DYNAMITE:
                self.dynamite.append(Dynamite([row, col]))
                dynamite_locations.append([row, col])

                self.fuel -= 1  # lose extra fuel if
                self.append_reward(
                    "drop dynamite", results_of_action, rewards)

            logging.debug(
                f"Player moved from [{row},{col}] to {self.location}")

            # Player has lost if:
            # its caught by the police,
            # stepped on its own bomb or
            # If player reaches a policecar,
            # player loses a life/game over
            for car in self.policecars:
                # if there is a policecar at the players location or
                # if the policecar and the player crossed paths
                if car.get_location() == self.location or (car.get_prev_location() == self.location and car.get_location() == [row, col]):
                    done = True

                    # undo move
                    self.location = [row, col]
                    self.dead = True
                    self.append_reward(
                        "caught", results_of_action, rewards)
                    break

            if self.location in dynamite_locations and action != DYNAMITE:
                done = True
                self.dead = True
                dynamite_locations.remove(self.location)

                self.append_reward(
                    "blown up", results_of_action, rewards)

        else:
            # Stop if run out of fuel
            done = True
            self.dead = True
            self.append_reward(
                "out of fuel", results_of_action, rewards)

        # if the player's current location used to be empty, remove from empty list
        if self.location in self.empty:
            self.empty.remove(self.location)

        # if the player has moved, add its previous location to the list of empty spaces
        if self.location != [row, col]:

            # moved closer to a bank + reward
            # moved away from a bank - reward
            distances = []
            new_distances = []
            for b in self.banks:
                distances.append(self.distance_to([row, col], b))
                if self.location is None:
                    print("uh oh")
                new_distances.append(self.distance_to(self.location, b))

            moved_closer = any([new_distances[i] < distances[i]
                               for i in range(len(distances))])
            if moved_closer:
                reward += 0
            else:
                reward -= self._rewards["moved"]

            if [row, col] not in dynamite_locations:
                self.empty.append([row, col])
            results_of_action.append(self.action_desc[0])
            rewards.append(reward)
        else:
            # If a player didnt move and it didnt drop dynamite
            # you have hit a wall
            if self.action_desc[3] not in results_of_action:
                self.append_reward(
                    "wall", results_of_action, rewards)

        if not done:

            if self.location in self.banks:
                self.banks.remove(self.location)
                self.banks_robbed += 1

                # new policecar appears at the robbed bank location and will start chasing after 5 moves
                self.policecars.append(PoliceCar(self.location))

                # a new bank appears at a random empty location
                if self.banks_robbed != self.max_banks:
                    new_bank_index = random.randrange(len(self.empty))
                    self.banks.append(self.empty[new_bank_index])
                    self.empty.remove(self.empty[new_bank_index])
                else:
                    done = True

                logging.debug(f"Robbed {self.banks_robbed} banks")
                # reward for robbing a bank
                self.append_reward(
                    "rob bank", results_of_action, rewards, multiplier=self.banks_robbed)

            # if there was a fueltank at the new location, player fuel returns to max
            if self.location in self.fueltanks:
                self.fuel = self.max_fuel
                self.fueltanks.remove(self.location)
                self.append_reward(
                    "collect fueltank", results_of_action, rewards)
                # reward for collecting an item
                self.append_reward(
                    "refuel", results_of_action, rewards)

        # update the global list of dynamite locations
        self.dynamite = [
            d for d in self.dynamite if d.get_location() in dynamite_locations]

        reward = sum(rewards)
        self.score += reward
        info = {
            "result_of_action": results_of_action,
            "rewards_per_action": rewards
        }

        # For debugging: output the result of the action taken by the player
        logging.debug(f'Results of action: {results_of_action}')

        # update the observation
        observation = self.make_observation()

        return observation, reward, done, info

    # Used for rendering the environment in the console
    # creates a 2d array with characters to represent the entities in the environment
    def create_grid(self):
        self.grid = np.zeros((self.nrow, self.ncol), str)

        for i, j in self.walls:
            self.grid[i][j] = '#'
        for i, j in self.fueltanks:
            self.grid[i][j] = 'F'

        for i, j in self.banks:
            self.grid[i][j] = 'B'
        for i, j in self.empty:
            self.grid[i][j] = ' '

        for policecar in self.policecars:
            i, j = policecar.get_location()
            self.grid[i][j] = 'p'

        for dynamite in self.dynamite:
            i, j = dynamite.get_location()
            self.grid[i][j] = 'd'

        # do not display the player if it has been killed
        if not self.dead:
            self.grid[self.location[0]][self.location[1]] = 'C'
        elif self.dead and self.grid[self.location[0]][self.location[1]] == '':
            self.grid[self.location[0]][self.location[1]] = ' '

    # reset the environment to its initial state
    def reset(self):
        # reset the banks, fueltanks and empty spaces
        self.banks = self.start_configuration["banks"].copy()
        self.fueltanks = self.start_configuration["fueltanks"].copy()
        self.empty = self.start_configuration["empty"].copy()

        # player goes back to starting position
        self.location = self.start.copy()
        self.dynamite = []
        self.policecars = []
        self.score = 0
        self.fuel = self.max_fuel
        self.banks_robbed = 0
        self.dead = False
        self.action_history = []
        self.visited = []

        observation = self.make_observation()

        return observation

    # render the environment

    def render(self, mode="console"):
        if mode not in self.metadata["render.modes"]:
            raise ValueError(f'Received unexpected render mode {mode}')

        self.create_grid()
        for row in self.grid:
            print(*row)

        print(f'${self.score}')
        print(f'Fuel: {100*self.fuel/self.max_fuel:.0f} %')

        if self.dead or self.fuel == 0:
            print("---------------- GAME OVER ----------------")
        elif self.banks_robbed == self.max_banks:
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

