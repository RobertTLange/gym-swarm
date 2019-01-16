import gym
from gym import error, spaces, utils
from gym.utils import seeding

import random
import itertools
import numpy as np

class SwarmEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.num_agents = 4
        self.env_noise = 0
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Tuple(self.num_disks*(spaces.Discrete(3),))
        self.current_state = None

        self.done = None

    def step(self, action):
        """
        * Inputs:
            - action: integer from 0 to 5 (see ACTION_LOOKUP)
        * Outputs:
            - current_state: state after transition
            - reward: reward from transition
            - done: episode state
            - info: dict of booleans (noisy?/invalid action?)
        0. Check if transition is noisy or not
        1. Transform action (0 to 5 integer) to tuple move - see Lookup
        2. Check if move is allowed
        3. If it is change corresponding entry | If not return same state
        4. Check if episode completed and return
        """
        if self.done:
            raise RuntimeError("Episode has finished. Call env.reset() to start a new episode.")

        info = {"transition_failure": False,
                "invalid_action": False}

        if self.env_noise > 0:
            r_num = random.random()
            if r_num <= self.env_noise:
                action = random.randint(0, self.action_space.n-1)
                info["transition_failure"] = True

        move = action_to_move[action]

        if self.move_allowed(move):
            disk_to_move = min(self.disks_on_peg(move[0]))
            moved_state = list(self.current_state)
            moved_state[disk_to_move] = move[1]
            self.current_state = tuple(moved_state)
        else:
            info["invalid_action"] = True

        if self.current_state == self.goal_state:
            reward = 100
            self.done = True
        else:
            reward = 0

        return self.current_state, reward, self.done, info

    def reset(self):
        self.current_state = self.num_disks * (0,)
        self.done = False
        return self.current_state

    def render(self, mode='human', close=False):
        return

    def set_env_parameters(self, num_disks=4, env_noise=0, verbose=True):
        self.num_disks = num_disks
        self.env_noise = env_noise
        self.observation_space = spaces.Tuple(self.num_disks*(spaces.Discrete(3),))
        self.goal_state = self.num_disks*(2,)

        if verbose:
            print("Hanoi Environment Parameters have been set to:")
            print("\t Number of Disks: {}".format(self.num_disks))
            print("\t Transition Failure Probability: {}".format(self.env_noise))
