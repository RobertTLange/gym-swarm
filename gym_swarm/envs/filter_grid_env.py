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
    TODO: Make it possible to load in parts of image & equip agents with filters
    TODO: Add a bunch of assert statements that make sure filter are uneven, etc.
    TODO: Allow for sampling of walls
    TODO: jit functions for speedup
    """
    # metadata = {'render.modes': ['human']}

    def __init__(self, random_placement=False):
        # SET INITIAL ENVIRONMENT PARAMETERS
        default_params = {'num_agents': 2,
                          'grid_size': 20,
                          'filter_size': 3,
                          'obs_size': 5,
                          'random_placement': False,
                          'wall_bump_reward': -0.05}
        self.set_env_params(default_params)

    def init_filters(self):
        # SAMPLE A SET OF FILTERS FOR THE AGENTS
        self.reward_filters = {}
        for agent_id in range(self.num_agents):
            self.reward_filters[agent_id] = np.random.randint(0, 255, self.filter_size**2).reshape((self.filter_size, self.filter_size))/255

    def place_filters(self):
        self.state_grid = np.zeros((self.grid_size, self.grid_size))
        # Place the filters in grid randomly - make sure there is no overlap
        filled_fields = []
        coords = []
        for agent_id in range(self.num_agents):
            while True:
                start_coord = np.random.randint(0, self.grid_size - self.filter_size, 2).astype(int)
                new_x = list(range(start_coord[0], start_coord[0]+self.filter_size))
                new_y = list(range(start_coord[1], start_coord[1]+self.filter_size))
                new_grid = [(x, y) for x in new_x for y in new_y]
                if len(set(filled_fields) & set(new_grid)) == 0:
                    filled_fields.extend(new_grid)
                    coords.append(start_coord)
                    break

        self.optimal_locations = {}
        for agent_id in range(self.num_agents):
            self.state_grid[coords[agent_id][0]:coords[agent_id][0]+self.filter_size, coords[agent_id][1]:coords[agent_id][1]+self.filter_size] = self.reward_filters[agent_id]
            self.optimal_locations[agent_id] = [coords[agent_id][0] + (self.filter_size/2)-1,
                                                coords[agent_id][1] + (self.filter_size/2)-1]

    def compute_reward_function(self):
        # Compute padded version of gridworld to convolute over it
        padded_grid = zero_pad(self.state_grid, (self.grid_size+self.filter_size-1,
                                                 self.grid_size+self.filter_size-1),
                               [int((self.filter_size-1)/2), int((self.filter_size-1)/2)])
        self.reward_function = {}
        # Loop over all agents and convolute with each agent-specific kernel
        for agent_id in range(self.num_agents):
            activation_grid = np.zeros((self.grid_size, self.grid_size))
            # Compute the convolution
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    activation_grid[i, j] = np.multiply(padded_grid[i:i+self.filter_size,
                                                                    j:j+self.filter_size],
                                                        self.reward_filters[agent_id]).sum()
            self.reward_function[agent_id] = activation_grid

    def reset(self):
        """
        Sample initial placement of agents & the corresponding observation
        """
        if self.random_placement:
            self.place_filters()
            self.compute_reward_function()
            self.sample_init_state()

        self.current_obs = self.state_to_obs()
        return self.current_obs

    def sample_init_state(self):
        # CONSTRUCT STATE - FULLY OBS & CONSTRUCT OBSERVATION - PARTIALLY OBS
        self.current_state = {}
        coords = []
        for agent_id in range(self.num_agents):
            while True:
                new_coords = np.random.randint(0, self.grid_size, 2).astype(int)
                if len(set(tuple(new_coords)) & set(coords)) == 0:
                    self.current_state[agent_id] = new_coords
                    coords.extend(new_coords.tolist())
                    break

    def state_to_obs(self):
        """
        Transform the state into observations issued to the agents
            - obs_size x obs_size grid centered on the agent
            - Important perform the padding for the agents
        TODO: jit this with numba!
        """
        obs = {}
        for agent_id in range(self.num_agents):
            x_start = max(0, int(self.current_state[agent_id][0]-(self.filter_size-1)/2))
            x_stop = min(self.grid_size, int(self.current_state[agent_id][0]+(self.filter_size-1)/2 + 1))
            y_start = max(0, int(self.current_state[agent_id][1]-(self.filter_size-1)/2))
            y_stop = min(self.grid_size, int(self.current_state[agent_id][1]+(self.filter_size-1)/2 + 1))

            obs_temp = self.state_grid[x_start:x_stop, y_start:y_stop]

            temp_rows_top = int(self.current_state[agent_id][0]-(self.filter_size-1)/2)
            if temp_rows_top < 0:
                add_rows_top = -1*temp_rows_top
                obs_temp = np.concatenate((-1+ np.zeros((add_rows_top, obs_temp.shape[1])),
                                           obs_temp), axis=0)

            temp_rows_bottom = int(self.current_state[agent_id][0]+(self.filter_size-1)/2)
            if temp_rows_bottom > self.grid_size:
                add_rows_bottom = temp_rows_bottom - self.grid_size
                obs_temp = np.concatenate((obs_temp,
                                           -1 + np.zeros((add_rows_bottom, obs_temp.shape[1]))), axis=0)

            temp_cols_left = int(self.current_state[agent_id][1]-(self.filter_size-1)/2)
            if temp_cols_left < 0:
                add_cols_left = -1*temp_cols_left
                obs_temp = np.concatenate((-1 + np.zeros((obs_temp.shape[0], add_cols_left)),
                                           obs_temp), axis=1)

            temp_cols_right = int(self.current_state[agent_id][1]+(self.filter_size-1)/2 + 1)

            if temp_cols_right > self.grid_size:
                add_cols_right = temp_cols_right - self.grid_size
                obs_temp = np.concatenate((obs_temp,
                                           -1 + np.zeros((obs_temp.shape[0], add_cols_right))), axis=1)
            # print(temp_rows_top, temp_rows_bottom, temp_cols_left, temp_cols_right)
            obs[agent_id] = obs_temp
        return obs

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
            # 0 - R, 1 - L, 2 - D, 3 - U, 4 - S
            agent_state = self.current_state[agent_id].copy()
            if agent_action == 0:   # Right Action Execution
                if agent_state[1] < self.grid_size - 1:
                    agent_state[1] +=1
                else:
                    wall_bump[agent_id] = 1
            elif agent_action == 1: # Left Action Execution
                if agent_state[1] > 0:
                    agent_state[1] -=1
                else:
                    wall_bump[agent_id] = 1
            elif agent_action == 2: # Down Action Execution
                if agent_state[0] > 0:
                    agent_state[0] -=1
                else:
                    wall_bump[agent_id] = 1
            elif agent_action == 3: # Up Action Execution
                if agent_state[0] < self.grid_size - 1:
                    agent_state[0] +=1
                else:
                    wall_bump[agent_id] = 1
            next_state[agent_id] = agent_state

        self.current_state = next_state
        self.current_obs = self.state_to_obs()
        # Calculate the reward based on the transition and return meta info
        reward, self.done = self.state_reward(wall_bump)
        info = {"warnings": None}
        return self.current_obs, reward, self.done, info

    def state_reward(self, wall_bump):
        """
        Agent-specific rewards given by activation of filters
        """
        # NOTE: Decide in learning loop whether to aggregate to global signal
        done = False
        reward = {i: 0 for i in range(self.num_agents)}

        # Loop over agents: Get specific rews - based on normalized activation
        no_agent_at_optimal_position = 0
        for agent_id in range(self.num_agents):
            reward[agent_id] += self.reward_function[agent_id][self.current_state[agent_id][0], self.current_state[agent_id][1]]
            reward[agent_id] += self.wall_bump_reward*wall_bump[agent_id]

            if (self.current_state[agent_id] == self.optimal_locations[agent_id]).all():
                no_agent_at_optimal_position += 1

        # Terminate the episode if all agents are at their optimal positions
        if no_agent_at_optimal_position == self.num_agents:
            done = True
        return reward, done

    def set_env_params(self, env_params=None, verbose=False):
        # SET INITIAL ENVIRONMENT PARAMETERS
        self.num_agents = env_params['num_agents']              # No. agents/kernels in env
        self.grid_size = env_params['grid_size']               # Size 2d grid [0, grid_size]
        self.filter_size = env_params['filter_size']            # Assert that uneven
        self.obs_size = env_params['obs_size']                  # Obssquare centered on agent
        self.random_placement = env_params['random_placement']  # Random placement at reset
        self.wall_bump_reward = env_params['wall_bump_reward']  # Wall bump reward

        self.done = None
        # SET OBSERVATION & ACTION SPACE (5 - u, d, l, r, stay)
        self.observation_space = spaces.Box(low=0, high=self.grid_size,
                                            shape=(self.num_agents, 2),
                                            dtype=np.int)
        self.action_space = spaces.Discrete(5)

        # PLACE FILTERS IN THE ENVIRONMENT & COMPUTE AGENT-SPECIFIC REWARD FCT
        self.init_filters()
        self.place_filters()
        self.compute_reward_function()
        self.sample_init_state()
        return

    def render(self, axs):
        """
        Render the environment state
        """
        # Get the basic environment layout
        plot_grid = self.state_grid.copy()
        # Plot agent filters in their location
        for agent_id in range(self.num_agents):
            x_start = max(0, int(self.current_state[agent_id][0]-(self.filter_size-1)/2))
            x_stop = min(self.grid_size, int(self.current_state[agent_id][0]+(self.filter_size-1)/2 + 1))
            y_start = max(0, int(self.current_state[agent_id][1]-(self.filter_size-1)/2))
            y_stop = min(self.grid_size, int(self.current_state[agent_id][1]+(self.filter_size-1)/2 + 1))
            plot_grid[x_start : x_stop, y_start : y_stop] = self.reward_filters[agent_id][0 : (x_stop - x_start), 0 : (y_stop - y_start)]/2

        axs.imshow(frame_image(plot_grid, 1), cmap="Greys")

        # Put blue circle at center of state of agent
        for agent_id in range(self.num_agents):
            temp_state = (self.current_state[agent_id][1]+1, self.current_state[agent_id][0]+1)
            circle = plt.Circle(temp_state, radius=0.25, color="blue")
            axs.add_artist(circle)
        axs.set_title("Environment State")
        axs.set_axis_off()
        return

    def render_reward(self, axs):
        for agent_id in range(self.num_agents):
            act_rgb = to_rgb(self.reward_function[agent_id], self.grid_size)
            axs[agent_id].imshow(frame_image(act_rgb, 1), cmap="Greys")
            axs[agent_id].set_title('Rewards Agent ' + str(agent_id + 1))
            axs[agent_id].set_axis_off()


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

def to_rgb(grid, grid_size):
    # RGB Heatmap visualisation
    grid3d = np.zeros((grid_size, grid_size, 3))
    x, y = np.where(grid!=0)
    x2, y2 = np.where(grid==0)
    grid3d[x, y, 0] = np.max(grid[x, y])-grid[x, y]
    grid3d[x2, y2, :] = 1
    return grid3d
