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
    # metadata = {'render.modes': ['human']}

    def __init__(self, random_placement=False):
        # SET INITIAL ENVIRONMENT PARAMETERS
        self.num_agents = 2                       # No. agents/kernels in env
        self.gridsize = 20                        # Size of 2d grid [0, 20]
        self.filtersize = 3                       # Assert that uneven
        self.random_placement = random_placement  # Random placement at reset
        self.done = None

        # SET REWARD PARAMETERS
        self.wall_bump_reward = -0.05

        # SET OBSERVATION & ACTION SPACE (5 - u, d, l, r, stay)
        self.observation_space = spaces.Box(low=0, high=self.gridsize,
                                            shape=(self.num_agents, 2),
                                            dtype=np.int)
        self.action_space = spaces.Discrete(5)

        # SAMPLE A SET OF FILTERS FOR THE AGENTS
        self.reward_filters = {}
        for agent_id in range(self.num_agents):
            self.reward_filters[agent_id] = np.random.randint(0, 255, self.filtersize**2).reshape((self.filtersize, self.filtersize))/255

        # PLACE FILTERS IN THE ENVIRONMENT & COMPUTE AGENT-SPECIFIC REWARD FCT
        self.place_filters()
        self.compute_reward_function()

    def place_filters(self):
        self.state_grid = np.zeros((self.gridsize, self.gridsize))
        # Place the filters in grid randomly - make sure there is no overlap
        x_hist, y_hist = [], []
        for agent_id in range(self.num_agents):
            while True:
                x_start = np.random.randint(0, self.gridsize - self.filtersize, 1).astype(int)[0]
                y_start = np.random.randint(0, self.gridsize - self.filtersize, 1).astype(int)[0]
                # Check for overlap with other grids already placed in env
                if 1:
                    # TODO: Implement resampling if there is filter overlap
                    break
            self.state_grid[x_start:x_start+self.filtersize, y_start:y_start+self.filtersize] = self.reward_filters[agent_id]

    def compute_reward_function(self):
        # Compute padded version of gridworld to convolute over it
        padded_grid = zero_pad(self.state_grid, (self.gridsize+self.filtersize-1,
                                                 self.gridsize+self.filtersize-1),
                               [int((self.filtersize-1)/2), int((self.filtersize-1)/2)])
        self.reward_function = {}
        # Loop over all agents and convolute with each agent-specific kernel
        for agent_id in range(self.num_agents):
            activation_grid = np.zeros((self.gridsize, self.gridsize))
            # Compute the convolution
            for i in range(self.gridsize):
                for j in range(self.gridsize):
                    activation_grid[i, j] = np.multiply(padded_grid[i:i+self.filtersize,
                                                                    j:j+self.filtersize],
                                                        self.reward_filters[agent_id]).sum()
            self.reward_function[agent_id] = activation_grid

    def reset(self):
        """
        Sample initial placement of agents on straight line w. no velocity
        """
        if self.random_placement:
            self.place_filters()
            self.compute_reward_function()

        # CONSTRUCT STATE - FULLY OBS & CONSTRUCT OBSERVATION - PARTIALLY OBS
        self.current_state = {}
        for agent_id in range(self.num_agents):
            self.current_state[agent_id] = np.random.randint(0, self.gridsize, 2)
        return self.current_state

    def step(self, action):
        """
        Perform a state transition/reward calculation based on selected action
        """
        if self.done:
            raise RuntimeError("Episode has finished. Call env.reset() to start a new episode.")

        # Update state of env and obs distributed to agents
        wall_bump = {i: 0 for i in range(self.num_agents)}
        next_state = {}
        for agent_id, agent_action in action.items():
            # 0 - U, 1 - D, 2 - L, 3 - R, 4 - S
            agent_state = self.current_state[agent_id].copy()
            if agent_action == 0:   # Up Action Execution
                if agent_state[1] < self.gridsize - 1:
                    agent_state[1] +=1
                else:
                    wall_bump[agent_id] = 1
            elif agent_action == 1: # Down Action Execution
                if agent_state[1] > 0:
                    agent_state[1] -=1
                else:
                    wall_bump[agent_id] = 1
            elif agent_action == 2: # Left Action Execution
                if agent_state[0] > 0:
                    agent_state[0] -=1
                else:
                    wall_bump[agent_id] = 1
            elif agent_action == 3: # Right Action Execution
                if agent_state[0] < self.gridsize - 1:
                    agent_state[0] +=1
                else:
                    wall_bump[agent_id] = 1
            next_state[agent_id] = agent_state

        self.current_state = next_state
        # Calculate the reward based on the transition and return meta info
        reward, self.done = self.state_reward(wall_bump)
        info = {"warnings": None}
        return self.current_state, reward, self.done, info

    def state_reward(self, wall_bump):
        """
        Agent-specific rewards given by activation of filters
        """
        # NOTE: Decide in learning loop whether to aggregate to global signal
        done = False
        reward = {i: 0 for i in range(self.num_agents)}

        for agent_id in range(self.num_agents):
            reward[agent_id] += self.reward_function[agent_id][self.current_state[agent_id][0], self.current_state[agent_id][1]]
            reward[agent_id] += self.wall_bump_reward*wall_bump[agent_id]
        return reward, done

    def set_env_params(self, env_params=None, verbose=False):
        return

    def render(self, mode='rgb_array', close=False):
        """
        Render the environment state
        """
        fig, axs = plt.subplots(1, 1, figsize=(8, 5))

        plot_grid = self.state_grid.copy()
        for agent_id in range(self.num_agents):
            x_start = int(self.current_state[agent_id][0]-(self.filtersize-1)/2)
            x_stop = int(self.current_state[agent_id][0]+(self.filtersize-1)/2 + 1)
            y_start = int(self.current_state[agent_id][1]-(self.filtersize-1)/2)
            y_stop = int(self.current_state[agent_id][1]+(self.filtersize-1)/2 + 1)
            # TODO: Add exception for corner padding case!
            plot_grid[x_start:x_stop, y_start:y_stop] = self.reward_filters[agent_id]/2
        axs.imshow(frame_image(plot_grid, 1), cmap="Greys")

        for agent_id in range(self.num_agents):
            temp_state = (self.current_state[agent_id][1]+1, self.current_state[agent_id][0]+1)
            circle = plt.Circle(temp_state, radius=0.25, color="blue")
            axs.add_artist(circle)
        axs.set_axis_off()
        return


def zero_pad(array, ref_shape, offsets):
    """
    array: Array to be padded
    reference: Reference array with the desired shape
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    """
    # Create an array of zeros with the reference shape
    result = np.zeros(ref_shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim]) for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = array
    return result

def frame_image(img, frame_width):
    # Add borders to the grid frame
    b = frame_width # border size in pixel
    ny, nx = img.shape[0], img.shape[1] # resolution / number of pixels in x and y
    if img.ndim == 3: # rgb or rgba array
        framed_img = np.zeros((b+ny+b, b+nx+b, img.shape[2]))
    elif img.ndim == 2: # grayscale image
        framed_img = np.zeros((b+ny+b, b+nx+b)) + 1
    framed_img[b:-b, b:-b] = img
    return framed_img

def to_rgb(grid, gridsize):
    # RGB Heatmap visualisation
    grid3d = np.zeros((gridsize, gridsize, 3))
    x, y = np.where(grid!=0)
    x2, y2 = np.where(grid==0)
    grid3d[x, y, 0] = np.max(grid[x, y])-grid[x, y]
    grid3d[x2, y2, :] = 1
    return grid3d
