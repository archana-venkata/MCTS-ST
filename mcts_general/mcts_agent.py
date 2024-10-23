
from config import MCTSAgentConfig
from game import DeepCopyableGame 

from tqdm import tqdm
import math
import numpy as np
import itertools
import random
from itertools import product


actions_str = ["LEFT", "DOWN", "RIGHT", "UP"]


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
        self.env = None
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

    def expand(self, env, observation, done, reward, history, strategy_value=0, initial_visit_count=0):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network. 
        """
        self.is_expanded = True
        self.done = done
        self.reward = reward
        self.strategy_value = strategy_value
        self.observation = observation
        self.visit_count = initial_visit_count
        self.env = env
        self.history = history
        valid_actions = env.legal_actions(simulation=True)
        for a in valid_actions:
            self.children[a] = Node()

        self.children_probs = np.ones((len(valid_actions),)) / len(valid_actions)

        # predict the probability of an action based on its history

        # state.children_probs, state.predicted_reward = self.llm_policy._calculate_emperical_prob(
        #     history, ob, valid_actions, self.env.get_goal(), 10, 0, 0.95)

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
        self.strategy_decay = self.config.init_decay

    def reset(self):
        self.rollout_buffer = []
        if self.config.use_strategies:
            self.strategy_decay = self.config.init_decay

    def seed(self, seed=None):
        np.random.seed(seed)

    def search(self, env, obs, reward, done, history, output_debug_info=False):
        self.root_node = self.node_cls(0)

        self.root_node.expand(env.get_copy(), obs, done, reward, history.copy())

        for i in range(self.config.num_simulations):
            root_node, info = self.simulate(self.root_node, 0)
            self.root_node = root_node

        # self.result_node.status()
        # self.render_tree(root_node)
        action, _ = self.select_child(self.root_node)
        if len(history) % 10:
            self.strategy_decay *= 0.99
        # print(f"Action: {action}")

        if output_debug_info:
            return action, info
        else:
            return action

    def render_tree(self, node, action=None):
        print(f"node: {node.id}, action: {actions_str[action] if action!=None else action}, value: {node.value()}")
        if node.expanded():
            print(f"node: {node.id} has {len(node.children.items())} children")
            for action, node in node.children.items():
                self.render_tree(node, action)

    def simulate(self, root, depth):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        node = root
        search_path = [node]
        current_tree_depth = depth
        while node.expanded():
            current_tree_depth += 1
            action, node = self.select_child(node)
            search_path.append(node)

        parent = search_path[-2]
        history = parent.history
        if not parent.done:
            game_copy = parent.env.get_copy()
            observation, reward, done, info = game_copy.step(action, simulation=True)
            for j in info["result_of_action"]:
                history.append((action, j))
            if self.config.do_roll_outs:
                value, strategy_value = self.get_roll_out(game_copy, history.copy())
                initial_visit_count = self.config.number_of_roll_outs - 1    # -1 because of increment in backprop
            else:
                value = reward
                initial_visit_count = 0

            node.expand(
                game_copy,
                observation,
                done,
                reward,
                history,
                strategy_value,
                initial_visit_count=initial_visit_count
            )
        else:
            value = 0

        self.backpropagate(search_path, value)
        # max_tree_depth = max(max_tree_depth, current_tree_depth)

        return root, None

    def backpropagate(self, search_path, value):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = node.reward + self.config.discount * value

    def get_roll_out(self, game, history, do_simulation_steps=False):
        if game.env.done:
            return 0, 0 
        done = False
        return_list = []
        strategy_return_list = []
        for _ in range(self.config.number_of_roll_outs):
            game_copy = game.get_copy()
            rollout_trajectory = history.copy()
            single_return = 0
            strategy_return = 0
            for it in range(self.config.max_roll_out_depth):
                if done:
                    break

                ######### USE STRATEGIES #########
                # if self.config.use_strategies:

                #     action_probs = self.calc_action_uncertainty(rollout_trajectory[-1][1], game_copy.env.unwrapped.nA)

                #     action = random.choice([i for i in range(len(action_probs)) if action_probs[i] == max(action_probs)])
                ##################################

                # else:
                action = game_copy.sample_action()  # randomly sample an action for rollouts

                _, reward, done, info = game_copy.step(action, simulation=do_simulation_steps)

                for i in info["result_of_action"]:
                    rollout_trajectory.append((action, i))

                single_return += reward * self.config.discount

            # what if return is number of strategies?
            # TODO: Combine return as reward and number of strategies then decay influence of strategies as time goes on

            for strategy in self.config.strategies:
                decay_param = self.config.init_decay
                x, y, z = is_subsequence(strategy['strategy'], [b for _, b in rollout_trajectory])
                if y > 0:
                    if z and single_return > 0:
                        decay_param = 0

                    strategy_return += y * strategy['reward'] * decay_param

            # if strategy_return == 0:
            #     # penalise no strategies followed in the rollout trajectory
            #     strategy_return -= z * strategy['reward'] * decay_param

            return_list.append(single_return)
            strategy_return_list.append(strategy_return)

            self.rollout_buffer.append(rollout_trajectory)

        return np.average(return_list), np.average(strategy_return_list)

    def select_child(self, node):
        """
        Select the child with the highest UCB score.
        """
        max_ucb = max(
            self.ucb_score(node, child)
            for action, child in node.children.items()
        )
        action = np.random.choice(
            [
                action
                for action, child in node.children.items()
                if self.ucb_score(node, child) == max_ucb
            ]
        )
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

    def select_action(node, temperature):
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

    def calc_action_uncertainty(self, event, num_actions):

        action_prob = np.ones((num_actions,)) / num_actions

        # get all strategies containing 'event'
        for strategy in [x['strategy'] for x in self.config.strategies if event in x['strategy']]:
            if event == strategy[-1]:
                # if the event is at the end of the strategy
                break
            index = strategy.index(event)
            next_event = strategy[index+1]
            # get all rollouts containing the next event in the strategy 'next_event'
            subset_rollouts = [rollout for rollout in self.rollout_buffer if hasEvent(rollout, next_event, appear_after=event)]
            if len(subset_rollouts) == 0:
                alt_subset_rollouts = [rollout for rollout in self.rollout_buffer if hasEvent(rollout, event)]
                for rollout in alt_subset_rollouts:
                    event_indices = [i for i, v in enumerate(rollout) if v[1] == event]
                    if len(event_indices) > 0 and max(event_indices)+1 < len(rollout):
                        next_action = rollout[event_indices[0]+1][0]
                        action_prob[next_action] = 0
                return action_prob

            for rollout in subset_rollouts:
                event_indices = [i for i, v in enumerate(rollout) if v[1] == event]
                if len(event_indices) > 0 and max(event_indices)+1 < len(rollout):
                    next_action = rollout[event_indices[0]+1][0]
                    action_prob[next_action] = 1
        return action_prob
