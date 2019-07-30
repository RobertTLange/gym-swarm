import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from scipy import ndimage

import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data

# from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import DistanceMetric

dir_path = os.path.dirname(os.path.realpath(__file__))

# Read in fish and predator emojis to plot episodes
fish_img = plt.imread(get_sample_data(dir_path + "/images/fish_tropical.png"))
fish_inv_img = np.flip(fish_img, axis=1)

fish_imgs = {0: fish_img,
             1: ndimage.rotate(fish_img, 45)[33:193, 33:193, :],
             2: ndimage.rotate(fish_img, 90),
             3: ndimage.rotate(fish_inv_img, -45)[33:193, 33:193, :],
             4: ndimage.rotate(fish_inv_img, 0),
             5: ndimage.rotate(fish_inv_img, 45)[33:193, 33:193, :],
             6: ndimage.rotate(fish_img, -90),
             7: ndimage.rotate(fish_img, -45)[33:193, 33:193, :]}


class key_to_goal():
    def __init__(self, init_position, pickup_range=0.5):
        """
        Key object that has to be passed twice before walking to goal with it
        - init_position: Initial position of the key
        - pickup_nhood: neighborhood in which the key can be picked up
        """
        self.position = init_position
        self.ownership = None
        self.update_pickup_nhood()
        self.pass_count = 0
        self.pickup_range = pickup_range

    def update_pickup_nhood(self):
        self.pickup_nhood = [self.position-self.pickup_range/2,
                             self.position+self.pickup_range/2]

    def pick_up_and_check(self, agent_position):
        return

    def move_with_owner(self, current_position_agents):
        if self.ownership is not None:
            self.position = current_position_agents[self.ownership]
            self.update_pickup_nhood()


class Doppelpass1DEnv(gym.Env):
    """
    "1D" Continuous Action Space Doppelpass Environment
    2 Agent env in which agents have to pick up a key and exchange it twice
    before reaching a goal location
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # SET INITIAL ENVIRONMENT PARAMETERS
        self.num_agents = 2
        self.obs_space_size = 20
        self.random_placement = True
        self.done = None

        # SET MAX//MIN VELOCITY AND ACCELARATION
        self.v_bounds = [-1, 1]
        self.a_bounds = [-1, 1]

        # FIX GOAL POSITION TO BE AT BORDER OF ENV - SAMPLE LATER ON
        self.goal_position = self.obs_space_size

        # SET INITIAL REWARD FUNCTION PARAMETERS
        self.wrong_pickup_reward = -1       # R - wrong attempt to pick up key
        self.correct_pickup_reward = 2      # R - picking up key
        self.wrong_exchange_reward = -1     # R - wrong attempt to exchange key
        self.correct_exchange_reward = 1    # R - exchanging key between agents
        self.correct_placement_reward = 10  # R - wrong attempt to exchange key

    def reset(self):
        """
        Sample initial placement of agents on straight line w. no velocity
        """
        if self.random_placement:
            self.current_postion = np.random.uniform(low=0,
                                                     high=self.obs_space_size,
                                                     size=self.num_agents)
            self.current_velocity = np.repeat(0, self.num_agents)

            self.key_position = np.random.uniform(low=0,
                                                  high=self.obs_space_size,
                                                  size=1)
        self.done = False
        return self.current_state

    def try_pickup_key():
        # TODO: Implement check of agent position - if so change ownership of key, etc.
        return

    def step(self, action):
        """
        Perform a state transition/reward calculation based on selected action
        -> action: Collective action dictionary for all agents
        """
        if self.done:
            raise RuntimeError("Episode has finished. Call env.reset() to start a new episode.")

        for agent_id, action in action.items():
            # Clip accelaration into range
            v_a = np.clip(self.current_velocity + action[0],
                          self.v_bounds[0], self.v_bounds[1])
            next_position_a = np.clip(self.current_position[agent_id] + v_a,
                                      0, self.obs_space_size)

            # TODO: Implement pickup/pass actions
            # TODO: Update the actual state of the agents

        # Calculate the reward based on the transition and return meta info
        reward, self.done = self.reward_doppelpass()
        info = {"warnings": None}
        return self.current_state, reward, self.done, info

    def reward_doppelpass(self, reward_type):
        return reward, done

    def render(self, mode='rgb_array', close=False):
        """
        Render the environment state
        """
        return
