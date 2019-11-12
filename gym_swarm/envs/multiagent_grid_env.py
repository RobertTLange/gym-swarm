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


def art_to_array(game_art):
    # Create a state array on which to act on!
    all_elements = set()
    for i in range(len(game_art)):
        temp = set(game_art[i])
        all_elements = all_elements | temp
    objects = list(all_elements)
    objects.remove(" ")

    y_dim = len(game_art[0])
    x_dim = len(game_art)
    state_array = np.zeros((x_dim, y_dim, len(objects)))
    print
    for row in range(state_array.shape[0]):
        for col in range(state_array.shape[1]):
            for i, obj in enumerate(objects):
                if game_art[row][col] == obj:
                    state_array[row, col, i] = 1
    return objects, state_array


def rgb_rescale(v):
    return v/255


COLOUR_FG = {' ': tuple([rgb_rescale(v) for v in (255, 255, 255)]),  # Background
             '#': tuple([rgb_rescale(v) for v in (20, 20, 20)]),     # Walls of the maze
             '0': tuple([rgb_rescale(v) for v in (214, 182, 79)]),   # Players
             '1': tuple([rgb_rescale(v) for v in  (214, 182, 79)]),
             '2': tuple([rgb_rescale(v) for v in  (214, 182, 79)]),
             '3': tuple([rgb_rescale(v) for v in  (214, 182, 79)]),
             '4': tuple([rgb_rescale(v) for v in   (214, 182, 79)]),   # Players
             '5': tuple([rgb_rescale(v) for v in  (214, 182, 79)]),
             '6': tuple([rgb_rescale(v) for v in  (214, 182, 79)]),
             '7': tuple([rgb_rescale(v) for v in  (214, 182, 79)]),
             '8': tuple([rgb_rescale(v) for v in  (214, 182, 79)]),
             'G': tuple([rgb_rescale(v) for v in (0, 100, 0)])         # Goal State
             }

# 25x35x6 Base Maze Gridworld for the Fish Environment
default_maze = ["###################################",
                "#01         #                     #",
                "#23         #                     #",
                "#           #                     #",
                "#       #####   #############     #",
                "#               #           #     #",
                "#####           #           #     #",
                "#####           #           #     #",
                "#####   #########    ####         #",
                "#####   #            #            #",
                "#####   #            #            #",
                "#########            #            #",
                "#####       #####    ##########   #",
                "#####       #        #        #   #",
                "#####       #        #        #   #",
                "#####       #        #        #   #",
                "#####    #############   #    #####",
                "#####                    #        #",
                "#####                    #        #",
                "#####                    #        #",
                "#####    #####################    #",
                "#####                    #        #",
                "#####                    #      GG#",
                "#####                    #      GG#",
                "###################################"]

class MultiAgentGridworldEnv(gym.Env):
    """
    Learning MA Coordination in a Grid World
    """
    # metadata = {'render.modes': ['human']}

    def __init__(self, sample_env=False, random_placement=False):
        # SET INITIAL ENVIRONMENT PARAMETERS
        self.num_agents = 2                       # No. agents/kernels in env
        self.grid_size = 20                       # Size of 2d grid [0, 20]
        self.filter_size = 3                      # Assert that uneven
        self.obs_size = 5                         # Obssquare centered on agent
        self.random_placement = random_placement  # Random placement at reset
        self.done = None
        self.sample_env = sample_env

        if self.sample_env:
            maze_art = generate_base_art(self.grid_size, self.grid_size)
            maze_w_walls = sample_walls(maze_art, max_wall_len=5, num_walls=5)
            game_art = sample_players_initial_state(maze_w_walls,
                                                    num_agents=self.num_agents)
        else:
            game_art = default_maze[:]

        self.objects, self.state = art_to_array(game_art)

        # SET REWARD PARAMETERS
        self.wall_bump_reward = -0.05

        # SET OBSERVATION & ACTION SPACE (5 - u, d, l, r, stay)
        self.observation_space = spaces.Box(low=0, high=self.grid_size,
                                            shape=(self.num_agents, 2),
                                            dtype=np.int)
        self.action_space = spaces.Discrete(5)

    def reset(self):
        """
        Sample initial placement of agents & the corresponding observation
        """
        self.current_obs = self.state_to_obs()
        return self.current_obs

    def state_to_obs(self):
        obs = {}
        # As in Pycolab feed different channels out!
        for agent_id in range(self.num_agents):
            obs[agent_id] = None
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
        for agent_id in range(self.num_agents):
            pass
        return reward, done

    def set_env_params(self, env_params=None, verbose=False):
        return

    def render(self, axs):
        base_idx = self.objects.index("#")
        background_base = self.state[:, :, base_idx]
        state_temp = self.state[:, :, :]

        x, y = np.where(background_base == 0)
        state_temp[x, y, 2] = 255
        x, y = np.where(background_base == 1)
        state_temp[x, y, 2] = 20/255

        # Plot the base background
        axs.imshow(self.state[:, :, 2])
        axs.set_axis_off()

        for i, obj in enumerate(self.objects):
            if obj == "#":
                pass
            elif obj == "G":
                x, y = np.where(self.state[:, :, i] == 1)
                for j in range(len(x)):
                    axs.text(y[j], x[j],
                        "G",
                        horizontalalignment="center",
                        verticalalignment="center",
                        bbox=dict(facecolor=COLOUR_FG[obj], alpha=1),
                        fontsize=13)
            else:
                x, y = np.where(self.state[:, :, i] == 1)
                for j in range(len(x)):
                    circle = plt.Circle((y[j], x[j]), radius=0.25, color=COLOUR_FG[obj])
                    axs.add_artist(circle)
        return
