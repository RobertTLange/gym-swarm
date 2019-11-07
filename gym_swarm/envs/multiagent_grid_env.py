import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import math
import numpy as np
import matplotlib.pyplot as plt


def generate_base_art(x_dim, y_dim):
    """
    Generate a plain grid with no obstacles
    - Note: x_dim & y_dim define the dims of the inner grid - without boundary
    """
    maze_art = []
    # Generate the rows of the environment
    for y in range(y_dim+2):
        if y == 0 or y == y_dim+1:
            row_string = (x_dim+2) * "#"
        else:
            row_string = "#" + x_dim*" " + "#"
        maze_art.append(row_string)
    return maze_art


def sample_walls(maze_art, max_wall_len=5, num_walls=2):
    """
    Add walls to the plain grid env
    """
    maze_w_walls = maze_art[:]
    # Get dimensions taking border walls into account
    x_dim, y_dim = len(maze_w_walls[0])-2, len(maze_w_walls)-2
    for j in range(num_walls):
        # Sample starting coordinates and length of walls
        wall_loc_x = np.random.randint(low=1, high=x_dim+1, dtype="int")
        wall_loc_y = np.random.randint(low=1, high=y_dim+1, dtype="int")
        wall_len = np.random.randint(low=2, high=max_wall_len, dtype="int")
        # Direction: 0 right, 1 down, 2 right diag up, 3 right diag down
        wall_dir = np.random.randint(low=0, high=4, dtype="int")

        # Edit the environment layout depending on the direction sampled
        if wall_dir == 0:  # Wall segment right
            if x_dim - wall_loc_x - wall_len < 0:
                wall_len = x_dim+1 - wall_loc_x
            maze_w_walls[wall_loc_y] = maze_w_walls[wall_loc_y][:wall_loc_x] + wall_len * "#" + maze_w_walls[wall_loc_y][wall_loc_x + wall_len:]
        elif wall_dir == 1:  # Wall segment down
            if y_dim - wall_loc_y - wall_len < 0:
                wall_len = y_dim+1 - wall_loc_y
            for i in range(0, wall_len):
                maze_w_walls[wall_loc_y + i] = maze_w_walls[wall_loc_y + i][:wall_loc_x] + "#" + maze_w_walls[wall_loc_y + i][wall_loc_x + 1:]
        elif wall_dir == 2:  # Wall segment right down up
            if y_dim - wall_loc_y - wall_len < 0 or 11 - wall_loc_x - wall_len < 0:
                wall_len = min([y_dim+1 - wall_loc_y, x_dim+1 - wall_loc_x])
            for i in range(0, wall_len):
                maze_w_walls[wall_loc_y + i] = maze_w_walls[wall_loc_y + i][:wall_loc_x+i] + "#" + maze_w_walls[wall_loc_y + i][wall_loc_x + i+ 1:]

        elif wall_dir == 3:  # Wall segment right diag down
            if y_dim - wall_loc_y - wall_len < 0 or wall_loc_x - wall_len < 0:
                wall_len = min([y_dim+1 - wall_loc_y, wall_loc_x])
            for i in range(0, wall_len):
                maze_w_walls[wall_loc_y + i] = maze_w_walls[wall_loc_y + i][:wall_loc_x - i] + "#" + maze_w_walls[wall_loc_y + i][wall_loc_x - i+ 1:]
    return maze_w_walls


def sample_players_initial_state(current_art, num_agents=2):
    """
    Add initial player positions to the env (with potentially added walls)
    """
    # First - clean up the previous environment!
    x_dim, y_dim = len(maze_w_walls[0])-2, len(maze_w_walls)-2
    maze_w_players = current_art[:]
    for agent_id in range(num_agents):
        for i, row in enumerate(maze_w_players):
            row_new = row.replace(str(agent_id), " ")
            maze_w_players[i] = row_new

    # Resample the initial position of the agent!
    for agent_id in range(num_agents):
        player_new_row = np.random.randint(1, y_dim+1)
        free_cols = [i for i in range(len(maze_w_players[player_new_row])) if maze_w_players[player_new_row].startswith(' ', i)]
        player_new_col = np.random.choice(free_cols)
        maze_w_players[player_new_row] = maze_w_players[player_new_row][:player_new_col] + str(agent_id) + maze_w_players[player_new_row][player_new_col+1:]
    return maze_w_players

class MultiAgentGridworldEnv(gym.Env):
    """
    Learning to Communicate Filter Positions
    """
    # metadata = {'render.modes': ['human']}

    def __init__(self, random_placement=False):
        # SET INITIAL ENVIRONMENT PARAMETERS
        self.num_agents = 2                       # No. agents/kernels in env
        self.grid_size = 20                       # Size of 2d grid [0, 20]
        self.filter_size = 3                      # Assert that uneven
        self.obs_size = 5                         # Obssquare centered on agent
        self.random_placement = random_placement  # Random placement at reset
        self.done = None

        # SET REWARD PARAMETERS
        self.wall_bump_reward = -0.05

        # SET OBSERVATION & ACTION SPACE (5 - u, d, l, r, stay)
        self.observation_space = spaces.Box(low=0, high=self.grid_size,
                                            shape=(self.num_agents, 2),
                                            dtype=np.int)
        self.action_space = spaces.Discrete(5)

        # SAMPLE A SET OF FILTERS FOR THE AGENTS
        self.reward_filters = {}
        for agent_id in range(self.num_agents):
            self.reward_filters[agent_id] = np.random.randint(0, 255, self.filter_size**2).reshape((self.filter_size, self.filter_size))/255

        # PLACE FILTERS IN THE ENVIRONMENT & COMPUTE AGENT-SPECIFIC REWARD FCT
        self.place_filters()
        self.compute_reward_function()

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

        # CONSTRUCT STATE - FULLY OBS & CONSTRUCT OBSERVATION - PARTIALLY OBS
        self.current_state = {}
        for agent_id in range(self.num_agents):
            self.current_state[agent_id] = np.random.randint(0, self.grid_size, 2)

        self.current_obs = self.state_to_obs()
        return self.current_obs

    def state_to_obs(self):
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
            # 0 - U, 1 - D, 2 - L, 3 - R, 4 - S
            agent_state = self.current_state[agent_id].copy()
            if agent_action == 0:   # Up Action Execution
                if agent_state[1] < self.grid_size - 1:
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
        return reward, done

    def set_env_params(self, env_params=None, verbose=False):
        return

    def render(self, mode='rgb_array', close=False):
        """
        Render the environment state
        """
        return
