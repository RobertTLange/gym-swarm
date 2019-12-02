import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pprint
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='white', palette='Paired',
        font='sans-serif', font_scale=1, color_codes=True, rc=None)
COLOURS = sns.color_palette("Set1")

from numba import jit


@jit(nopython=False)
def jit_step(action, current_state, num_agents, grid_size):
    wall_bump = np.zeros(num_agents)
    next_state = current_state.copy()

    for agent_id in range(num_agents):
        # 0 - R, 1 - L, 2 - D, 3 - U, 4 - S
        agent_state = current_state[agent_id, :].copy()
        if action[agent_id] == 0:   # Right Action Execution
            if agent_state[1] < grid_size - 1:
                agent_state[1] = agent_state[1] + 1
            else:
                wall_bump[agent_id] = 1
        elif action[agent_id] == 1: # Left Action Execution
            if agent_state[1] > 0:
                agent_state[1] = agent_state[1] - 1
            else:
                wall_bump[agent_id] = 1
        elif action[agent_id] == 2: # Down Action Execution
            if agent_state[0] > 0:
                agent_state[0] = agent_state[0] - 1
            else:
                wall_bump[agent_id] = 1
        elif action[agent_id] == 3: # Up Action Execution
            if agent_state[0] < grid_size - 1:
                agent_state[0] = agent_state[0] + 1
            else:
                wall_bump[agent_id] = 1
        next_state[agent_id] = agent_state

    return next_state, wall_bump


class FilterGridworldEnv(gym.Env):
    """
    Learning to Communicate Filter Positions
    TODO: Make it possible to load in parts of image & equip agents with filters
    TODO: Add a bunch of assert statements that make sure filter are uneven, etc.
    """
    # metadata = {'render.modes': ['human']}

    def __init__(self, random_placement=False):
        # SET INITIAL ENVIRONMENT PARAMETERS
        default_params = {'num_agents': 2,
                          'grid_size': 20,
                          'filter_size': 3,
                          'obs_size': 5,
                          'random_placement': False,
                          'random_filters': False,
                          'num_distraction_filters': 0,
                          'wall_bump_reward': -0.05,
                          'step_reward': -0.01,
                          'sparse_reward': 0,
                          'filter_as_obs': 0}
        self.action_space_size = 5
        self.set_env_params(default_params)

    def init_filters(self):
        # SAMPLE A SET OF FILTERS FOR THE AGENTS
        self.reward_filters = {}
        self.distraction_filters = {}

        if self.filter_as_obs:
            self.padded_filters = {}
            to_pad = (self.obs_size-self.filter_size)//2

        for agent_id in range(self.num_agents):
            self.reward_filters[agent_id] = np.random.randint(0, 255, self.filter_size**2).reshape((self.filter_size, self.filter_size))/255

            # If filters are part of observations - store correctly padded version
            if self.filter_as_obs:
                self.padded_filters[agent_id] = zero_pad(self.reward_filters[agent_id],
                                                         (self.obs_size, self.obs_size),
                                                         (to_pad, to_pad))

        for distr_id in range(self.num_distraction_filters):
            self.distraction_filters[distr_id] = np.random.randint(0, 255, self.filter_size**2).reshape((self.filter_size, self.filter_size))/255

    def place_filters(self):
        """ Generate the gridworld & place the sampled filters into them"""
        self.state_grid = np.zeros((self.grid_size, self.grid_size))

        # Place the distraction filters in the environment
        for distr_id in range(self.num_distraction_filters):
            coord = np.random.randint(0, self.grid_size - self.filter_size, 2).astype(int)
            self.state_grid[coord[0]:coord[0]+self.filter_size, coord[1]:coord[1]+self.filter_size] = self.distraction_filters[distr_id]

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
        # Place filters and store optimal locations
        for agent_id in range(self.num_agents):
            self.state_grid[coords[agent_id][0]:coords[agent_id][0]+self.filter_size, coords[agent_id][1]:coords[agent_id][1]+self.filter_size] = self.reward_filters[agent_id]
            self.optimal_locations[agent_id] = [int(coords[agent_id][0] + (self.filter_size-1)/2),
                                                int(coords[agent_id][1] + (self.filter_size-1)/2)]

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
            if not self.sparse_reward:
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        activation_grid[i, j] = np.multiply(padded_grid[i:i+self.filter_size,
                                                                        j:j+self.filter_size],
                                                            self.reward_filters[agent_id]).sum()
            else:
                # Sparse Env Configuration of Reward Function
                x_start = self.optimal_locations[agent_id][0] - int((self.filter_size-1)/2)
                x_stop = self.optimal_locations[agent_id][0] + int((self.filter_size-1)/2) + 1
                y_start = self.optimal_locations[agent_id][1] - int((self.filter_size-1)/2)
                y_stop = self.optimal_locations[agent_id][1] + int((self.filter_size-1)/2) + 1
                activation_grid[self.optimal_locations[agent_id][0], self.optimal_locations[agent_id][1]] = np.multiply(padded_grid[x_start:x_stop,
                                                                                          y_start:y_stop],
                                                                              self.reward_filters[agent_id]).sum()
            self.reward_function[agent_id] = activation_grid

    def reset(self):
        """
        Sample initial placement of agents & the corresponding observation
        """
        if self.random_filters:
            self.init_filters()
            self.place_filters()
        self.compute_reward_function()

        if self.random_placement:
            self.sample_init_state()
        else:
            self.current_state = self.agent_position_store.copy()

        # Reset "accomlishments" of the agents
        self.done = False
        self.visit_optimal_loc = {agent_id: 0 for agent_id in range(self.num_agents)}
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
        """
        obs = {}
        for agent_id in range(self.num_agents):
            # Get "valid" observation parts
            y_start = max(0, int(self.current_state[agent_id][0]-(self.obs_size-1)/2))
            y_stop = min(self.grid_size, int(self.current_state[agent_id][0]+(self.obs_size-1)/2 + 1))
            x_start = max(0, int(self.current_state[agent_id][1]-(self.obs_size-1)/2))
            x_stop = min(self.grid_size, int(self.current_state[agent_id][1]+(self.obs_size-1)/2 + 1))

            obs_temp = self.state_grid[y_start:y_stop, x_start:x_stop]

            # Figure out how much to pad where
            temp_rows_top = int(self.current_state[agent_id][0]-(self.obs_size-1)/2)
            if temp_rows_top < 0:
                add_rows_top = -1*temp_rows_top
                obs_temp = np.concatenate((-1+ np.zeros((add_rows_top, obs_temp.shape[1])),
                                           obs_temp), axis=0)

            temp_rows_bottom = int(self.current_state[agent_id][0]+(self.obs_size-1)/2 + 1)
            if temp_rows_bottom > self.grid_size:
                add_rows_bottom = temp_rows_bottom - self.grid_size
                obs_temp = np.concatenate((obs_temp,
                                           -1 + np.zeros((add_rows_bottom, obs_temp.shape[1]))), axis=0)

            temp_cols_left = int(self.current_state[agent_id][1]-(self.obs_size-1)/2)
            if temp_cols_left < 0:
                add_cols_left = -1*temp_cols_left
                obs_temp = np.concatenate((-1 + np.zeros((obs_temp.shape[0], add_cols_left)),
                                           obs_temp), axis=1)

            temp_cols_right = int(self.current_state[agent_id][1]+(self.obs_size-1)/2 + 1)
            if temp_cols_right > self.grid_size:
                add_cols_right = temp_cols_right - self.grid_size
                obs_temp = np.concatenate((obs_temp,
                                           -1 + np.zeros((obs_temp.shape[0], add_cols_right))), axis=1)

            # Stack individual filter on top of observation!
            if self.filter_as_obs:
                obs[agent_id] = np.stack((obs_temp, self.padded_filters[agent_id]), axis=0)
            else:
                obs[agent_id] = obs_temp
        return obs

    def step(self, action):
        """
        Perform a state transition/reward calculation based on selected action
        """
        if self.done:
            raise RuntimeError("Episode has finished. Call env.reset() to start a new episode.")

        action = np.array(list(action.values()))
        current_state = np.array(list(self.current_state.values()))
        # Update state of env and obs distributed to agents
        next_state, wall_bump = jit_step(action, current_state,
                                         self.num_agents, self.grid_size)

        self.current_state = {i: next_state[i, :] for i in range(next_state.shape[0])}
        # Calculate the reward based on the transition and return meta info
        reward, self.done = self.state_reward(wall_bump)
        self.current_obs = self.state_to_obs()
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
        for agent_id in range(self.num_agents):
            reward[agent_id] += self.reward_function[agent_id][self.current_state[agent_id][0], self.current_state[agent_id][1]]
            reward[agent_id] += self.wall_bump_reward*wall_bump[agent_id]
            reward[agent_id] += self.step_reward

            if (self.current_state[agent_id] == self.optimal_locations[agent_id]).all():
                self.visit_optimal_loc[agent_id] = 1

        # Terminate the episode if all agents are at their optimal positions
        if sum(self.visit_optimal_loc.values()) == self.num_agents:
            done = True
        return reward, done

    def set_env_params(self, env_params=None, verbose=False):
        # SET INITIAL ENVIRONMENT PARAMETERS
        self.num_agents = env_params['num_agents']              # No. agents/kernels in env
        self.grid_size = env_params['grid_size']                # Size 2d grid [0, grid_size]
        self.filter_size = env_params['filter_size']            # Assert that uneven
        self.obs_size = env_params['obs_size']                  # Obssquare centered on agent
        self.random_placement = env_params['random_placement']  # Random placement at reset
        self.random_filters = env_params['random_filters']      # Seed for filter init/place
        self.wall_bump_reward = env_params['wall_bump_reward']  # Wall bump reward
        self.step_reward = env_params['step_reward']            # Step reward
        self.sparse_reward = env_params['sparse_reward']        # Sparse reward
        self.filter_as_obs = env_params['filter_as_obs']        # Return filter as part of obs
        self.num_distraction_filters = env_params['num_distraction_filters']

        # SET OBSERVATION & ACTION SPACE (5 - u, d, l, r, stay)
        self.observation_space = spaces.Box(low=0, high=self.grid_size,
                                            shape=(self.num_agents, 2),
                                            dtype=np.int)
        self.action_space = spaces.Discrete(5)

        self.init_filters()
        self.sample_init_state()
        self.place_filters()
        self.agent_position_store = self.current_state.copy()

        if verbose:
            print("Set environment parameters to:")
            pprint.pprint(env_params)
        return

    def render(self, axs, title="Environment State"):
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
            axs.text(temp_state[0], temp_state[1],
                str(agent_id+1),
                horizontalalignment="right",
                verticalalignment="top",
                fontsize=11)
        axs.set_title(title)
        axs.set_axis_off()
        return

    def render_reward(self, axs):
        for agent_id in range(self.num_agents):
            act_rgb = to_rgb(self.reward_function[agent_id], self.grid_size)
            axs[agent_id].imshow(frame_image(act_rgb, 1), cmap="Greys")
            axs[agent_id].set_title('Rewards Agent ' + str(agent_id + 1))
            axs[agent_id].set_axis_off()

    def render_obs(self, axs):
        obs = self.state_to_obs()
        for agent_id in range(self.num_agents):
            obs_to_plot = obs[agent_id]
            axs[agent_id].imshow(obs_to_plot, cmap="Greys", vmin=-0.4, vmax=1)
            axs[agent_id].set_axis_off()
            axs[agent_id].set_title("Agent: {}".format(agent_id + 1))

    def render_trace(self, agent_locs):
        fig, axs = plt.subplots(1, 1, figsize=(12, 8))
        # Get the basic environment layout
        plot_grid = self.state_grid.copy()
        # Plot agent filters in their location
        for agent_id in range(self.num_agents):
            x_start = max(0, int(agent_locs[-1][agent_id][0]-(self.filter_size-1)/2))
            x_stop = min(self.grid_size, int(agent_locs[-1][agent_id][0]+(self.filter_size-1)/2 + 1))
            y_start = max(0, int(agent_locs[-1][agent_id][1]-(self.filter_size-1)/2))
            y_stop = min(self.grid_size, int(agent_locs[-1][agent_id][1]+(self.filter_size-1)/2 + 1))
            plot_grid[x_start : x_stop, y_start : y_stop] = self.reward_filters[agent_id][0 : (x_stop - x_start), 0 : (y_stop - y_start)]/2

        axs.imshow(frame_image(plot_grid, 1), cmap="Greys")

        # Put blue circle at center of state of agent
        for t in reversed(range(len(agent_locs))):
            for agent_id in range(self.num_agents):
                temp_state = (agent_locs[t][agent_id][1]+1, agent_locs[t][agent_id][0]+1)
                circle = plt.Circle(temp_state, radius=0.25, color=COLOURS[agent_id],
                                    alpha=(t+1)/len(agent_locs))
                axs.add_artist(circle)
            axs.set_axis_off()
        return fig, axs

    def get_agent_locs(self):
        return self.current_state.copy()

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
    result[tuple(insertHere)] = array
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
