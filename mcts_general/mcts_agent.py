
from config import MCTSAgentConfig
from game import DeepCopyableGame 

from tqdm import tqdm
import math
import numpy as np
import itertools
import random
from itertools import product
from copy import deepcopy


ACTIONS_STR = ["LEFT", "DOWN", "RIGHT", "UP"]


def hasEvent(element, event, appear_after=None):
    indices = [i for i, v in enumerate(element) if v[1] == event]
    if appear_after != None: 
        prev_event_indices = [i for i, v in enumerate(element) if v[1] == appear_after]
    if len(indices) > 0:
        if appear_after != None:
            return any(i > j for i, j in product(indices, prev_event_indices))
        return True
    return False


def is_subsequence(a, b):
    count = 0
    a_index = 0
    b_index = 0
    b_it = iter(b)
    is_complete = False
    while b_index < len(b):
        a_val = a[a_index]
        if a_val in b[b_index:]:
            count += 1
            while str(a_val) != str(next(b_it)):
                b_index += 1
                pass
            if a_index < len(a)-1:
                a_index += 1
            else:
                if count == len(a):
                    is_complete = True
                a_index = 0
            b_index += 1
        else:
            break

    if count > 0:
        return True, count/len(a), is_complete
    else:
        return False, count, is_complete


class Node:

    id_iter = itertools.count()

    def __init__(self, prior=1):
        self.id = next(self.id_iter)
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior      # uniform prior
        self.value_sum = 0
        self.children = {}
        self.children_probs = []
        self.observation = None
        self.reward = 0
        self.done = False
        self.env_state = None
        self.is_expanded = False
        self.strategy_value = 0

    def __str__(self) -> str:
        return f"Node(\'{self.id}\', \'{len(self.children)}\',\'{self.info}\',\'{self.reward}\',\'{self.done}\')"

    def status(self):
        print(
            f"NODE: id: {self.id}, visits: {self.visit_count}")
        for key, child in self.children.items():
            print(
                f"CHILD: id: {child.id}, visits: {child.visit_count}")

    def expanded(self):
        return self.is_expanded

    def value(self):
        if self.visit_count == 0 or self.done:
            return 0
        return self.value_sum 

    def expand(self, env_state, observation, done, reward, history, valid_actions=[], strategy_value=0, initial_visit_count=0):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network. 
        """
        self.is_expanded = True
        self.done = done
        self.reward = reward
        # if history contains partially completed strategies can initialise this to a high value
        self.observation = observation
        self.visit_count = initial_visit_count
        self.env_state = env_state
        self.history = history

        self.strategy_value = strategy_value

        # valid_actions = env.legal_actions(simulation=True)
        for a in valid_actions:
            self.children[a] = Node()

        self.children_probs = np.ones((len(valid_actions),)) / len(valid_actions) if len(valid_actions) != 0 else []

        # if len(strategy)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions. TODO Doc
        """
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class MCTSAgent:

    def __init__(self, env: DeepCopyableGame, config: MCTSAgentConfig):
        self.config = config
        self.node_cls = Node
        self.rollout_buffer = []
        self.game = env
        self.strategy_decay = self.config.init_decay
        self.simulation_strategies_completed = 0

    def reset(self):
        self.rollout_buffer = []
        self.simulation_strategies_completed = 0
        if self.config.use_strategies:
            self.strategy_decay = self.config.init_decay

    def set_seed(self, seed=None):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def search(self, env_state, obs, reward, done, history, legal_actions, output_debug_info=False):
        self.root_node = self.node_cls(0)
        self.root_node.expand(deepcopy(env_state), obs, done, reward, history.copy(), valid_actions=legal_actions)
        for i in range(self.config.num_simulations):
            self.root_node = self.simulate()

        action = self.select_action(self.root_node, self.config.temperature)

        # every 10 steps, decay the strategic contribution
        if self.config.use_strategies and self.simulation_strategies_completed > 80 and len(history) % 10:
            self.strategy_decay *= 0.99

        return action

    def render_tree(self, node, action=None):
        print(f"node: {node.id}, action: {ACTIONS_STR[action] if action!=None else action}, value: {node.value()}")
        if node.expanded():
            print(f"node: {node.id} has {len(node.children.items())} children")
            for action, node in node.children.items():
                self.render_tree(node, action)

    def simulate(self):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        node = self.root_node
        search_path = [node]
        while node.expanded():
            action, node = self.select_child(node)
            search_path.append(node)

        parent = search_path[-2]
        history = parent.history.copy()
        rollout_value = 0
        if not parent.done:
            game_state_copy = deepcopy(parent.env_state)
            self.game.unwrapped.load_state(game_state_copy)
            observation, reward, terminated, truncated, info = self.game.step(action)
            done = terminated or truncated
            legal_actions = self.game.unwrapped.legal_actions()
            for j in info["result_of_action"]:
                history.append((action, j))
            new_game_state = deepcopy(self.game.unwrapped.save_state())

            rollout_value, strategy_value = self.get_roll_out(new_game_state, history.copy())
            initial_visit_count = self.config.number_of_roll_outs - 1    # -1 because of increment in backprop

            node.expand(
                new_game_state,
                observation,
                done,
                reward,
                history,
                strategy_value=strategy_value,
                initial_visit_count=initial_visit_count,
                valid_actions=legal_actions
            )
        else:
            rollout_value = 0
        self.backpropagate(search_path, rollout_value)

        return search_path[0]

    def backpropagate(self, search_path, value):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = node.reward + self.config.discount * value

    def get_roll_out(self, initial_game_state, history):
        if initial_game_state["done"]:
            return 0, 0
        done = False
        return_list = []
        strategy_return_list = []
        for _ in range(self.config.number_of_roll_outs):
            # TODO: handle more than 1 rollout
            self.game.unwrapped.load_state(deepcopy(initial_game_state))
            rollout_trajectory = history.copy()
            single_return = 0
            strategy_return = 0
            depth = 0
            while not done and depth < self.config.max_roll_out_depth:
                # randomly sample an action for rollouts
                action = random.choice(self.game.unwrapped.legal_actions())

                _, reward, terminated, truncated, info = self.game.step(action)   
                done = terminated or truncated             

                for i in info["result_of_action"]:
                    rollout_trajectory.append((action, i))

                single_return += reward * self.config.discount
                depth += 1

            if self.config.use_strategies:
                strategy_exp_bonus = 0
                for strategy in self.config.strategies:
                    x, y, z = is_subsequence(strategy['strategy'], [b for _, b in rollout_trajectory])
                    if x:
                        if y:
                            strategy_exp_bonus = y if single_return == 0 else single_return*y
                        if z:
                            self.simulation_strategies_completed += 1
            # if single_return > 0:
            #     print("this rollout was good!")

                strategy_return += strategy_exp_bonus
            return_list.append(single_return)
            strategy_return_list.append(strategy_return)

            self.rollout_buffer.append(rollout_trajectory)
            self.game.reset()

        return np.average(return_list), np.average(strategy_return)

    def select_child(self, node):
        """
        Select the child with the highest UCB score.
        """
        max_ucb = max(
            self.ucb_score(node, child)
            for action, child in node.children.items()
        )
        possible_next_actions = [
                action
                for action, child in node.children.items()
                if self.ucb_score(node, child) == max_ucb
            ]
        action = np.random.choice(possible_next_actions)

        if self.config.use_strategies:
            max_strat_value = max([node.children[x].strategy_value for x in possible_next_actions])
            possible_strategy_actions = [x for x in possible_next_actions if node.children[x].strategy_value == max_strat_value]

            if max_strat_value != 0 and len(possible_strategy_actions) != 0 and len(possible_next_actions) > 1:
                action = np.random.choice(possible_strategy_actions)

        return action, node.children[action]

    def ucb_score(self, parent, child):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        # return child.value_sum + self.config.exploration_constant * child_prob * np.sqrt(parent.visit_count) / (child.visit_count + 1)
        # Unexplored nodes have maximum values so we favour exploration    

        # prior_score = pb_c * child.prior
        if child.visit_count == 0:
            return np.inf
        prior_score = self.config.pb_c_init * np.sqrt(np.log(parent.visit_count) / child.visit_count)  # uniform prior score

        if child.visit_count > 0:
            # Mean value Q
            value_score = child.value()/child.visit_count 
        else:
            value_score = 0

        if self.config.use_strategies:
            strategy_score = self.strategy_decay * child.strategy_value/child.visit_count 
            # if strategy_score > 0:
            #     print("this is doing something")
        else:
            strategy_score = 0

        return prior_score + value_score + strategy_score

    def select_action(self, node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = np.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution)
        return action
