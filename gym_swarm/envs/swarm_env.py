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
        self.obs_space_size = 20
        self.action_space = spaces.Discrete(8)

        self.observation_space = np.zeros((self.obs_space_size,
                                           self.obs_space_size))

        self.current_state = dict(zip(np.arange(self.num_agents),
                                      [np.empty(2)]*len(keys)))

        self.random_placement = True
        self.done = None

    def step(self, action):
        """
        * Inputs:
            - action: integer from 0 to 7 (see ACTION_LOOKUP)
        * Outputs:
            - current_state: dictionary of all agent states after transition
            - reward: cumulative reward of complete state transition
            - done: episode state
            - info: dict of booleans (noisy?/invalid action?)
        0. Check if transition is noisy or not
        1. Transform action (0 to 7 integer) to tuple move - see Lookup
        2. Check if move is allowed
        3. If it is change corresponding entry | If not return same state
        4. Check if episode completed and return
        """
        if self.done:
            raise RuntimeError("Episode has finished. Call env.reset() to start a new episode.")

        info = None

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
        if self.random_placement:
            # For each agent sample an initial placement in grid - uniform!
            states_temp = np.random.randint(self.obs_space_size,
                                            size=(2, self.num_agents))
            invalid, state_overlap = self.invalid_position(states_temp)

            while invalid:
                invalid, state_overlap = self.invalid_position(states_temp)
                states_temp

        # Transform valid state array into dictionary
        self.current_state = dict(enumerate(states_temp.T))
        self.done = False
        return self.current_state

    def invalid_position(self, states_temp):
        state_overlap = np.zeros((self.num_agents, self.num_agents))
        for i in range(self.num_agents):
            check_idx = np.where(state_overlap[i, :] == 0)
            for j in check_idx:
                if np.array_equal(states_temp[:, i], states_temp[:, j]):
                    state_overlap[i, j] = 1
                    state_overlap[j, i] = 1

        if np.sum(state_overlap) == 0:
            return False, state_overlap
        else:
            return True, state_overlap

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


action_to_move = {0: np.array([-1, 0]),
                  1: np.array([-1, -1]),
                  2: np.array([0, -1]),
                  3: np.array([1, -1]),
                  4: np.array([1, 0]),
                  5: np.array([1, 1]),
                  6: np.array([0, 1]),
                  7: np.array([-1, 1])}

ACTION_LOOKUP = {0: "left",
                 1: "left-down",
                 2: "down",
                 3: "right-down",
                 4: "right",
                 5: "right-up",
                 6: "up",
                 7: "left-up"}


if __name__ == "__main__":
    env = SwarmEnv()
