
from config import MCTSAgentConfig
from game import DeepCopyableGame 

from tqdm import tqdm
import math
import numpy as np
import itertools


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

    def __str__(self) -> str:
        return f"Node(\'{self.id}\', \'{len(self.children)}\',\'{self.info}\',\'{self.reward}\',\'{self.done}\')"

    def status(self):
        print(
            f"NODE: id: {self.id}, visits: {self.visit_count}")
        for key, child in self.children.items():
            print(
                f"CHILD: id: {child.id}, visits: {child.visit_count}")

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0 or self.done:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, env, observation, done, reward=0, initial_visit_count=0):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network. 
        """
        self.done = done
        self.reward = reward
        self.observation = observation
        self.visit_count = initial_visit_count
        self.env = env
        valid_actions = env.legal_actions(simulation=True)
        for a in valid_actions:
            self.children[a] = Node()

        self.children_probs = np.ones((len(valid_actions),)) / len(valid_actions)

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

    def seed(self, seed=None):
        np.random.seed(seed)

    def search(self, env, obs, reward, done, output_debug_info=False):
        self.root_node = self.node_cls(0)

        self.root_node.expand(env.get_copy(), obs, done, reward)

        for _ in range(self.config.num_simulations):
            root_node, info = self.simulate(self.root_node, 0)
            self.root_node = root_node

        # self.result_node.status()

        action, _ = self.select_child(self.root_node)
        # print(f"Action: {action}")

        if output_debug_info:
            return action, info
        else:
            return action

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
        if not parent.done:
            game_copy = parent.env.get_copy()
            observation, reward, done, info = game_copy.step(action, simulation=True)

            if self.config.do_roll_outs:
                value = self.get_roll_out(game_copy)
                initial_visit_count = self.config.number_of_roll_outs - 1    # -1 because of increment in backprop
            else:
                value = reward
                initial_visit_count = 0

            node.expand(
                game_copy,
                observation,
                done,
                reward,
                initial_visit_count=initial_visit_count
            )
        else:
            value = 0

        self.backpropagate(search_path, value)

        return root, None

    def backpropagate(self, search_path, value):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            # min_max_stats.update(node.reward + self.config.discount * node.value())

            value = node.reward + self.config.discount * value

    def get_roll_out(self, game_copy, do_simulation_steps=False):

        total_reward = 0
        done = True
        for _ in range(self.config.number_of_roll_outs):

            # TODO: check if strategies are followed

            trajectory_reward = 0
            for it in range(self.config.max_roll_out_depth):
                if done:
                    break
                action = game_copy.sample_action()  # randomly sample an action for rollouts
                _, reward, done = game_copy.step(action, simulation=do_simulation_steps)
                trajectory_reward += self.config.discount * reward
            total_reward += trajectory_reward

        return total_reward

    def select_child(self, node):
        """
        Select the child with the highest UCB score.
        """
        best_value = -np.inf
        best_children = []
        best_children_prob = []
        for i, (action, child) in enumerate(node.children.items()):
            assert len(node.children_probs) == len(node.children), print(node.children_probs)
            child_prob = node.children_probs[i]

            ucb_value = self.ucb_score(node, child, child_prob)

            if ucb_value == best_value:
                best_children.append(action)
                best_children_prob.append(child_prob)
            elif ucb_value > best_value:
                best_value = ucb_value
                best_children = [node.children[action]]
                best_children_prob = [child_prob]

        action = np.random.choice(
            [
                action
                for i, (action, child) in enumerate(node.children.items())
                if self.ucb_score(node, child, node.children_probs[i]) == best_value
            ]
        )

        return action, node.children[action]

    def ucb_score(self, parent, child, child_prob):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        return child.value_sum + self.config.exploration_constant * child_prob * np.sqrt(parent.visit_count) / (child.visit_count + 1)

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
