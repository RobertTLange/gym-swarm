import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import math
import numpy as np
import matplotlib.pyplot as plt


class FilterGridworldEnv(gym.Env):
    """
    Learning to Communicate Filter Positions
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, random_placement=False):
        # SET INITIAL ENVIRONMENT PARAMETERS
        self.num_agents = 2                       # No. agents/kernels in env
        self.gridsize = 20                        # Size of 1d line [0, 20]
        self.random_placement = random_placement  # Key & agent initialization
        self.done = None

        # SET OBSERVATION AND ACTION SPACE
        self.observation_space = spaces.Box(low=0, high=self.gridsize,
                                            shape=(self.num_agents, 2),
                                            dtype=np.int)
        self.action_space = spaces.Discrete(5)

        # SAMPLE A SET OF FILTERS FOR THE AGENTS

    def reset(self):
        """
        Sample initial placement of agents on straight line w. no velocity
        """
        if self.random_placement:
            self.agents_positions = np.random.uniform(low=0,
                                                      high=self.obs_space_size,
                                                      size=self.num_agents)
        else:
            self.agents_positions = np.linspace(0, self.obs_space_size,
                                                self.num_agents)


        # CONSTRUCT STATE - FULLY OBS & CONSTRUCT OBSERVATION - PARTIALLY OBS
        self.state = self.get_current_state()
        self.observations = self.get_obs_from_state()
        self.done = False
        return self.observations

    def step(self, action):
        """
        Perform a state transition/reward calculation based on selected action
        """
        if self.done:
            raise RuntimeError("Episode has finished. Call env.reset() to start a new episode.")

        for agent_id, agent_action in action.items():
            pass

        # Update state of env and obs distributed to agents
        self.state = self.get_current_state()
        self.observations = self.get_obs_from_state()

        # Calculate the reward based on the transition and return meta info
        reward, self.done = self.reward_vision()
        info = {"warnings": None}
        return self.observations, reward, self.done, info

    def get_obs_from_state(self):
        """
        Compute the obs from the state representation
        """
        # NOTE: Stack the different channels one vector representation
        observation = {0: np.zeros(4 + 3*self.observation_resolution),
                       1: np.zeros(4 + 3*self.observation_resolution)}

        for agent_id in range(self.num_agents):
            pass
        return observation

    def reward_vision(self):
        """
        Agent-specific rewards given by activation of filters
        """
        # NOTE: Decide in learning loop whether to aggregate to global signal
        done = False
        reward = {0: 0, 1: 0}

        for agent_id in range(self.num_agents):
            pass
        return reward, done

    def set_env_params(self, env_params=None, reward_params=None, verbose=False):
        return

    def render(self, mode='rgb_array', close=False):
        """
        Render the environment state
        """
        return
