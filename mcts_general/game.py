"""
This package contains `DeepCopyableGame` and its implementations. See implementations for details.
"""

import numpy
import typing

import abc
from copy import deepcopy

import gymnasium as gym

from common.wrapper import DeepCopyableWrapper, DiscreteActionWrapper


class DeepCopyableGame(metaclass=abc.ABCMeta):
    """
    This is the interface for a game used within the MCTS search and mainly provides a forward simulator with methods
    for getting getting deep copies of your game state as well as sampling actions for exploration.

    :ivar seed: The seed used in all pseudo random components. This can be set and retrieved usind set_seed() and
    get_seed() only.
    """

    def __init__(self, seed):
        self.rand = numpy.random
        self.__seed = seed

    @abc.abstractmethod
    def legal_actions(self, simulation=False) -> list:
        """ Used in tree expansion. """
        pass

    @abc.abstractmethod
    def sample_action(self, simulation=False):
        """ Used in Roll outs. """
        pass

    @abc.abstractmethod
    def reset(self, seed, options={}):
        """ (Re-)Initializes the Environment and returns the (new) initial state. """
        pass

    @abc.abstractmethod
    def step(self, action, simulation=False) -> tuple:
        """
        Take one step in the game. Similar to gym.env.step() this should output a tuple: observation, reward, done

        :param action: The action to be taken.
        :param simulation: The flag 'simulation' is True during MCTS steps and False by default. This can be used if you want a different
        behaviour during planning than during evaluation (e. g. plan in a different time step discretization).
        :return: observation, reward, done
        """
        pass

    @abc.abstractmethod
    def render(self, mode='human', **kwargs):
        """ Render the environment """
        pass

    @abc.abstractmethod
    def load_state(self, saved_state, seed=None):
        """ Loads copy of the game with a given state """
        pass

    @abc.abstractmethod
    def save_state(self) -> dict:
        """ Returns the current state of the game """
        pass

    def get_seed(self):
        """ Get the current seed. Note that we are using conventional getters and setters because this makes inheritance
         of getter/setter behaviour much more straight forward than using `@property`. We decided to put code
         readability over doing things the pythonic way in this case. """
        return self.__seed

    def set_seed(self, seed):
        """ Set the seed for all pseudo random components used in your game. """
        self.rand.seed(seed)
        self.__seed = seed


class GymGame(DeepCopyableGame, metaclass=abc.ABCMeta):
    """
    This abstract class underlies all classes that use OpenAI gym environments. It ensures that the Gym Environment
    is wrapped in the `DeepCopyableWrapper` and links the `DeepCopyableGame` methods to OpenAi gym.env's methods.
    """

    def __init__(self, env: gym.Env, seed=0):
        self.env = DeepCopyableWrapper(env) if not isinstance(env, DeepCopyableWrapper) else env
        self.render_copy = None
        super(GymGame, self).__init__(seed)

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def close(self):
        self.env.close()
        if self.render_copy is not None:
            self.render_copy.close()

    def step(self, action, simulation=False):
        obs, rew, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, rew, done, info

    def render(self, mode='human', **kwargs):
        # This workaround is necessary because a game / a gym env that is rendering cannot be deepcopied
        if self.render_copy is None:
            self.render_copy = self.get_copy()
            self.render_copy.env.render(mode, **kwargs)
        else:
            self.render_copy.close()
            self.render_copy = self.get_copy()
            self.render_copy.env.render(mode, **kwargs)

    def get_copy(self) -> "GymGame":
        return GymGame(deepcopy(self.env), seed=self.rand.randint(1e9))

    def set_seed(self, seed):
        pass

    def __str__(self):
        return str(self.env).split('<')[-1].split('>')[0].split(' ')[0]


class DiscreteGymGame(GymGame):

    def __init__(self, env, seed=0):
        assert isinstance(env.action_space, gym.spaces.Discrete), "Gym Env must have discrete action space!"
        super(DiscreteGymGame, self).__init__(env, seed)

    def step(self, action, simulation=False):
        action = int(action)
        obs, rew, done, info = super(DiscreteGymGame, self).step(action, simulation)
        return obs, rew, done, info

    def legal_actions(self, simulation=False):
        actions = numpy.array(range(self.env.action_space.n))
        return actions[self.env.valid_action_mask()] 

    def sample_action(self, simulation=False):
        legal_actions = self.legal_actions(simulation=simulation)
        return legal_actions[self.rand.random_integers(0, len(legal_actions) - 1)]

    def get_copy(self, seed) -> "DiscreteGymGame":
        return DiscreteGymGame(deepcopy(self.env), seed)

    def load_state(self, saved_state):
        self.env.load_state(deepcopy(saved_state))

    def save_state(self):
        return self.env.save_state()
