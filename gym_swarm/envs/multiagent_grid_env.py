import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pprint
import math
import itertools
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='white', palette='Paired',
        font='sans-serif', font_scale=1, color_codes=True, rc=None)
COLOURS = sns.color_palette("Set1")

from numba import jit


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


def set_agents_corner(current_art, num_agents):
    """
    Place a set of agents into the corner of the environment
    """
    maze_w_players = current_art[:]
    locations = list(itertools.product(range(1, 4),repeat= 2))
    for agent_id in range(num_agents):
        row, col = locations[agent_id]
        maze_w_players[row] = maze_w_players[row][:col] + str(agent_id) + maze_w_players[row][col+1:]
    return maze_w_players


def art_to_array(game_art, num_agents):
    # Create a state array on which to act on!
    all_elements = set()
    for i in range(len(game_art)):
        temp = set(game_art[i])
        all_elements = all_elements | temp
    obj = list(all_elements)
    obj.remove(" ")

    agent_ids = []
    object_types = []
    for x in obj:
        try:
            int_temp = int(x)
            agent_ids.append(x)
        except:
            object_types.append(x)

    objects = agent_ids + object_types
    y_dim = len(game_art[0])
    x_dim = len(game_art)
    state_array = np.zeros((len(objects), x_dim, y_dim))

    for row in range(state_array.shape[1]):
        for col in range(state_array.shape[2]):
            for i, obj in enumerate(objects):
                if game_art[row][col] == obj:
                    state_array[i, row, col] = 1
    return objects, state_array


def rgb_rescale(v):
    return v/255


COLOUR_FG = {' ': tuple([rgb_rescale(v) for v in (255, 255, 255)]),  # Background
             '#': tuple([rgb_rescale(v) for v in (20, 20, 20)]),     # Walls of the maze
             '0': tuple([rgb_rescale(v) for v in (214, 182, 79)]),   # Players
             '1': tuple([rgb_rescale(v) for v in  (214, 182, 79)]),
             '2': tuple([rgb_rescale(v) for v in  (214, 182, 79)]),
             '3': tuple([rgb_rescale(v) for v in  (214, 182, 79)]),
             '4': tuple([rgb_rescale(v) for v in   (214, 182, 79)]),
             '5': tuple([rgb_rescale(v) for v in  (214, 182, 79)]),
             '6': tuple([rgb_rescale(v) for v in  (214, 182, 79)]),
             '7': tuple([rgb_rescale(v) for v in  (214, 182, 79)]),
             '8': tuple([rgb_rescale(v) for v in  (214, 182, 79)]),
             'G': tuple([rgb_rescale(v) for v in (0, 100, 0)]),       # Goal State
             'S': tuple([rgb_rescale(v) for v in (100, 100, 100)])    # Subgoal State
             }


# 25x35x6 Base Maze Gridworld for the Fish Environment
default_maze = ["###################################",
                "#           #                     #",
                "#           #    S       S        #",
                "#           #                  S  #",
                "#  S    #####   #############     #",
                "#       S     S #           #     #",
                "#####           #    S      #     #",
                "#####           #         S #     #",
                "#####   #########  S ####      S  #",
                "#####   #            #            #",
                "#####   #    S   S   #            #",
                "#########            #            #",
                "#####       #####    ##########   #",
                "#####  S    #        #        #   #",
                "#####       #        #   S    #   #",
                "#####       #        #        #   #",
                "#####    #############   #    #####",
                "#####                    #        #",
                "#####  S       S       S #   S    #",
                "#####                    #      S #",
                "#####    #####################    #",
                "#####                    #        #",
                "#####                    #      GG#",
                "#####                    #      GG#",
                "###################################"]


# 25x35x6 Base Maze Gridworld for the Fish Environment
subtask_maze = ["###################################",
                "#           #                     #",
                "#           #                     #",
                "#           #                     #",
                "#       #####   #############   GS#",
                "#               #############   SS#",
                "###################################"]

@jit(nopython=False)
def jit_step(action, state, num_agents, objects, wall_states):
    wall_bump = np.zeros(num_agents)
    next_state = state.copy()

    for agent_id in range(num_agents):
        # 0 - R, 1 - L, 2 - D, 3 - U, 4 - S
        # Previously: Agent index had to be extracted from list of objects
        # Problem: Incompatible with Numba
        # agent_index = objects.index(str(agent_id))
        agent_state_old = np.where(state[agent_id, :, :] == 1)
        agent_state_new = np.array([agent_state_old[0][0], agent_state_old[1][0]])
        agent_action = action[agent_id]
        if agent_action == 0:   # Right Action Execution
            agent_state_new[1] = agent_state_old[1][0] + 1
        elif agent_action == 1: # Left Action Execution
            agent_state_new[1] = agent_state_old[1][0] - 1
        elif agent_action == 2: # Down Action Execution
            agent_state_new[0] = agent_state_old[0][0] - 1
        elif agent_action == 3: # Up Action Execution
            agent_state_new[0] = agent_state_old[0][0] + 1
        # If agent_action is 4 don't do anything

        for row_id in range(wall_states.shape[0]):
            if (agent_state_new == wall_states[row_id, :]).all():
                wall_bump[agent_id] = 1
                break
        if wall_bump[agent_id] < 1:
            # Otherwise perform the state transition
            next_state[agent_id, agent_state_old[0][0], agent_state_old[1][0]] = 0
            next_state[agent_id, agent_state_new[0], agent_state_new[1]] = 1
    return next_state, wall_bump


class MultiAgentGridworldEnv(gym.Env):
    """
    Learning MA Coordination in a Grid World
    """
    # metadata = {'render.modes': ['human']}

    def __init__(self, sample_env=False, random_placement=False):
        # SET INITIAL ENVIRONMENT PARAMETERS
        self.num_agents = 4                       # No. agents/kernels in env
        self.grid_size = (25, 35)                 # Size of 2d grid [0, 20]
        self.obs_size = 3                         # Obssquare centered on agent
        self.random_placement = random_placement  # Random placement at reset
        self.done = None
        self.sample_env = sample_env
        self.maze = default_maze
        self.train_subtask = False

        if self.sample_env:
            maze_art = generate_base_art(self.grid_size[0], self.grid_size[1])
            maze_w_walls = sample_walls(maze_art, max_wall_len=5, num_walls=5)
            self.game_art = sample_players_initial_state(maze_w_walls,
                                                         num_agents=self.num_agents)
        else:
            self.game_art = set_agents_corner(self.maze, self.num_agents)

        # Convert Art into State Array & Location Indices
        self.objects, self.state = art_to_array(self.game_art, self.num_agents)
        self.num_objects = self.state.shape[0]
        self.wall_index = self.objects.index("#")
        self.goal_index = self.objects.index("G")
        self.subgoal_index = self.objects.index("S")
        self.wall_states = list(zip(*np.where(self.state[self.wall_index, :, :]==1)))
        self.goal_states = list(zip(*np.where(self.state[self.goal_index, :, :]==1)))
        self.subgoal_states = list(zip(*np.where(self.state[self.subgoal_index, :, :]==1)))

        # SET REWARD PARAMETERS
        self.wall_bump_reward = -0.05
        self.subgoal_reward = 1
        self.final_goal_reward = 10

        # SET OBSERVATION & ACTION SPACE (5 - u, d, l, r, stay)
        self.action_space_size = 5

    def reset(self):
        """
        Sample initial placement of agents & the corresponding observation
        - Reset the goal locations!
        """
        self.done = False
        if self.train_subtask:
            self.maze = subtask_maze
        self.game_art = set_agents_corner(self.maze, self.num_agents)
        self.objects, self.state = art_to_array(self.game_art, self.num_agents)
        self.num_objects = self.state.shape[0]
        self.wall_index = self.objects.index("#")
        self.goal_index = self.objects.index("G")
        self.subgoal_index = self.objects.index("S")

        self.wall_states = list(zip(*np.where(self.state[self.wall_index, :, :]==1)))
        self.goal_states = list(zip(*np.where(self.state[self.goal_index, :, :]==1)))
        self.subgoal_states = list(zip(*np.where(self.state[self.subgoal_index, :, :]==1)))

        self.current_obs = self.state_to_obs()
        return self.current_obs

    def state_to_obs(self):
        obs = {}
        # As in Pycolab feed different channels out!
        for agent_id in range(self.num_agents):
            agent_index = self.objects.index(str(agent_id))
            agent_state = np.where(self.state[agent_index, :, :]==1)
            # Get "valid" observation parts
            y_start = max(0, int(agent_state[0]-(self.obs_size-1)/2))
            y_stop = min(self.grid_size[0], int(agent_state[0]+(self.obs_size-1)/2 + 1))
            x_start = max(0, int(agent_state[1]-(self.obs_size-1)/2))
            x_stop = min(self.grid_size[1], int(agent_state[1]+(self.obs_size-1)/2 + 1))

            obs_temp = self.state[:, y_start:y_stop, x_start:x_stop]
            # Figure out how much to pad where
            temp_rows_top = int(agent_state[0]-(self.obs_size-1)/2)
            if temp_rows_top < 0:
                add_rows_top = -1*temp_rows_top
                top_rows = np.zeros((self.num_objects, add_rows_top, obs_temp.shape[2]))
                # Make wall dimension respect "end" of env
                top_rows[self.wall_index, :, :] = 1
                obs_temp = np.concatenate((top_rows, obs_temp), axis=1)

            temp_rows_bottom = int(agent_state[0]+(self.obs_size-1)/2 + 1)
            if temp_rows_bottom > self.grid_size[0]:
                add_rows_bottom = temp_rows_bottom - self.grid_size[0]
                bottom_rows = np.zeros((self.num_objects, add_rows_bottom, obs_temp.shape[2]))
                # Make wall dimension respect "end" of env
                bottom_rows[self.wall_index, :, :] = 1
                obs_temp = np.concatenate((obs_temp, bottom_rows), axis=1)

            temp_cols_left = int(agent_state[1]-(self.obs_size-1)/2)
            if temp_cols_left < 0:
                add_cols_left = -1*temp_cols_left
                cols_left = np.zeros((self.num_objects, obs_temp.shape[1], add_cols_left))
                # Make wall dimension respect "end" of env
                cols_left[self.wall_index, :, :] = 1
                obs_temp = np.concatenate((cols_left, obs_temp), axis=2)

            temp_cols_right = int(agent_state[1]+(self.obs_size-1)/2 + 1)
            if temp_cols_right > self.grid_size[1]:
                add_cols_right = temp_cols_right - self.grid_size[1]
                cols_right = np.zeros((self.num_objects, obs_temp.shape[1], add_cols_right))
                # Make wall dimension respect "end" of env
                cols_right[self.wall_index, :, :] = 1
                obs_temp = np.concatenate((obs_temp, cols_right), axis=2)

            obs[agent_id] = obs_temp
        return obs

    def step(self, action):
        """
        Perform a state transition/reward calculation based on selected action
        """
        if self.done:
            raise RuntimeError("Episode has finished. Call env.reset() to start a new episode.")

        # Update state of env and obs distributed to agents
        action = np.array(list(action.values()))
        objects = np.array(self.objects)
        wall_states = np.array(self.wall_states)
        self.state, wall_bump = jit_step(action, self.state, self.num_agents,
                                         objects, wall_states)

        # Calculate the reward based on the transition and return meta info
        reward, self.done = self.state_reward(wall_bump)
        self.current_obs = self.state_to_obs()
        info = {"warnings": None}
        return self.current_obs, reward, self.done, info

    def state_reward(self, wall_bump):
        """ Agent-specific rewards given by activation of filters """
        done = False
        reward = {i: 0 for i in range(self.num_agents)}

        # Loop over agents: Get specific rews - based on normalized activation
        for agent_id in range(self.num_agents):
            agent_state = np.where(self.state[agent_id, :, :] == 1)
            agent_state = [agent_state[0][0], agent_state[1][0]]
            if tuple(agent_state) in self.subgoal_states:
                # Delete Subgoal from list of subgoal states
                self.state[self.subgoal_index, agent_state[0], agent_state[1]] = 0
                self.subgoal_states.remove(tuple(agent_state))
                reward[agent_id] += self.subgoal_reward
            elif tuple(agent_state) in self.goal_states:
                # Final goal state is reached by an agent - terminate
                reward[agent_id] += self.final_goal_reward
                done = True
            else:
                reward[agent_id] += self.wall_bump_reward*wall_bump[agent_id]
        return reward, done

    def set_env_params(self, env_params=None, verbose=False):
        # SET INITIAL ENVIRONMENT PARAMETERS
        self.num_agents = env_params['num_agents']                # No. agents/kernels in env
        self.obs_size = env_params['obs_size']                    # Obssquare centered on agent
        self.wall_bump_reward = env_params['wall_bump_reward']    # Wall bump reward
        self.subgoal_reward = env_params['subgoal_reward']        # Subgoal bump reward
        self.final_goal_reward = env_params['final_goal_reward']  # Final goal bump reward
        self.train_subtask = env_params['train_subtask']

        if verbose:
            print("Set environment parameters to:")
            pprint.pprint(env_params)
            print("Call env.reset() to create env with new parameters")
        return

    def render(self, axs):
        """
        Function renders the current state of the environment
        """
        # Plot the base environment
        base_idx = self.objects.index("#")
        state_temp = self.state.copy()[:, :, :]
        background_base = state_temp[base_idx, :, :]

        x, y = np.where(background_base == 0)
        state_temp[base_idx, x, y] = 255
        x, y = np.where(background_base == 1)
        state_temp[base_idx, x, y] = 20/255

        # Plot the base background
        axs.imshow(state_temp[base_idx, :, :])
        axs.set_axis_off()

        for i, obj in enumerate(self.objects):
            if obj == "#":
                pass
            elif obj == "G":
                x, y = np.where(self.state[i, :, :] == 1)
                for j in range(len(x)):
                    axs.text(y[j], x[j],
                        "G",
                        horizontalalignment="center",
                        verticalalignment="center",
                        bbox=dict(facecolor=COLOUR_FG[obj], alpha=1),
                        fontsize=13)
            elif obj == "S":
                x, y = np.where(self.state[i, :, :] == 1)
                for j in range(len(x)):
                    axs.text(y[j], x[j],
                        "S",
                        horizontalalignment="center",
                        verticalalignment="center",
                        bbox=dict(facecolor=COLOUR_FG[obj], alpha=1),
                        fontsize=13)
            else:
                x, y = np.where(self.state[i, :, :] == 1)
                for j in range(len(x)):
                    circle = plt.Circle((y[j], x[j]), radius=0.25, color=COLOUR_FG[obj])
                    axs.add_artist(circle)
        return

    def render_obs(self, axs):
        """
        Function renders the partially observed state of the agents
        """
        obs = self.state_to_obs()

        for agent_id in range(self.num_agents):
            base_idx = self.objects.index("#")
            background_base = obs[agent_id][base_idx, :, :]
            obs_temp = obs[agent_id].copy()[:, :, :]

            x, y = np.where(background_base == 0)
            obs_temp[base_idx, x, y] = 255
            x, y = np.where(background_base == 1)
            obs_temp[base_idx, x, y] = 20/255

            # Plot the base background
            axs[agent_id].imshow(obs_temp[base_idx, :, :])
            axs[agent_id].set_axis_off()
            axs[agent_id].set_title("Agent: {}".format(agent_id + 1))

            for i, obj in enumerate(self.objects):
                if obj == "#":
                    continue
                elif obj == "G":
                    x, y = np.where(obs_temp[i, :, :] == 1)
                    for j in range(len(x)):
                        axs[agent_id].text(y[j], x[j],
                            "G",
                            horizontalalignment="center",
                            verticalalignment="center",
                            bbox=dict(facecolor=COLOUR_FG[obj], alpha=1),
                            fontsize=13)
                elif obj == "S":
                    x, y = np.where(obs_temp[i, :, :] == 1)
                    for j in range(len(x)):
                        axs[agent_id].text(y[j], x[j],
                            "S",
                            horizontalalignment="center",
                            verticalalignment="center",
                            bbox=dict(facecolor=COLOUR_FG[obj], alpha=1),
                            fontsize=13)
                else:
                    x, y = np.where(obs_temp[i, :, :] == 1)
                    for j in range(len(x)):
                        circle = plt.Circle((y[j], x[j]), radius=0.25, color=COLOUR_FG[obj])
                        axs[agent_id].add_artist(circle)
        return

    def render_trace(self, agent_locs):
        # Plot the base environment
        fig, axs = plt.subplots(1, 1, figsize=(12, 8))
        base_idx = self.objects.index("#")
        state_temp = self.state.copy()[:, :, :]
        background_base = state_temp[base_idx, :, :]

        x, y = np.where(background_base == 0)
        state_temp[base_idx, x, y] = 255
        x, y = np.where(background_base == 1)
        state_temp[base_idx, x, y] = 20/255

        # Plot the base background
        axs.imshow(state_temp[base_idx, :, :])
        axs.set_axis_off()

        for i, obj in enumerate(self.objects):
            if obj == "#":
                pass
            elif obj == "G":
                x, y = np.where(self.state[i, :, :] == 1)
                for j in range(len(x)):
                    axs.text(y[j], x[j],
                        "G",
                        horizontalalignment="center",
                        verticalalignment="center",
                        bbox=dict(facecolor=COLOUR_FG[obj], alpha=1),
                        fontsize=13)
            elif obj == "S":
                x, y = np.where(self.state[i, :, :] == 1)
                for j in range(len(x)):
                    axs.text(y[j], x[j],
                        "S",
                        horizontalalignment="center",
                        verticalalignment="center",
                        bbox=dict(facecolor=COLOUR_FG[obj], alpha=1),
                        fontsize=13)

        for t in reversed(range(len(agent_locs))):
            for agent_id in range(len(agent_locs[0])):
                x, y = agent_locs[t][agent_id][0], agent_locs[t][agent_id][1]
                circle = plt.Circle((y, x), radius=0.25, color=COLOURS[agent_id],
                                    alpha=(t+1)/len(agent_locs))
                axs.add_artist(circle)
        return fig, axs

    def get_agent_locs(self):
        state_storage = []
        for i, obj in enumerate(self.objects):
            if obj not in ["#", "G", "S"]:
                x, y = np.where(self.state[i, :, :] == 1)
                state_storage.append((x, y))
        return state_storage
