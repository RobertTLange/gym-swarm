import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
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
predator_img = plt.imread(get_sample_data(dir_path + "/images/predator_shark.png"))

fish_inv_img = np.flip(fish_img, axis=1)
predator_inv_img = np.flip(predator_img, axis=1)

fish_imgs = {0: fish_img,
             1: ndimage.rotate(fish_img, 45)[33:193, 33:193, :],
             2: ndimage.rotate(fish_img, 90),
             3: ndimage.rotate(fish_inv_img, -45)[33:193, 33:193, :],
             4: ndimage.rotate(fish_inv_img, 0),
             5: ndimage.rotate(fish_inv_img, 45)[33:193, 33:193, :],
             6: ndimage.rotate(fish_img, -90),
             7: ndimage.rotate(fish_img, -45)[33:193, 33:193, :]}

predator_imgs = {0: predator_img,
                 1: ndimage.rotate(predator_img, 45)[33:193, 33:193, :],
                 2: ndimage.rotate(predator_img, 90),
                 3: ndimage.rotate(predator_inv_img, -45)[33:193, 33:193, :],
                 4: ndimage.rotate(predator_inv_img, 0),
                 5: ndimage.rotate(predator_inv_img, 45)[33:193, 33:193, :],
                 6: ndimage.rotate(predator_img, -90),
                 7: ndimage.rotate(predator_img, -45)[33:193, 33:193, :]}


def step_agent(agent_state, move_agent, obs_space_size):
    """
    In: 2d-coords of agent/predator in discrete grid, 2d-move, size of grid
    Out: New state after transition and boundary check for single agent
    """
    temp = agent_state + move_agent
    # x/y-Axis turnover - Check periodic boundary conditions
    for i in range(2):
        if temp[i] > (obs_space_size - 1):
            temp[i] = 0
        elif temp[i] < 0:
            temp[i] = obs_space_size - 1
    return temp


def step_all_agents(agent_states, moves, obs_space_size):
    """
    In: 2d-coords of all agents in discrete grid, 2d-move, size of grid
    Out: New state after transition and boundary check for all agents
    """
    states_temp = np.array([v for v in agent_states.values()]).T
    moves_temp = np.array([v for v in moves.values()]).T
    next_state_temp = states_temp + moves_temp
    # Check for periodic boundary conditions
    # Exit on the right side or the top
    idx_r = np.argwhere(next_state_temp[0, :] > (obs_space_size - 1))
    idx_t = np.argwhere(next_state_temp[1, :] > (obs_space_size - 1))
    next_state_temp[0, idx_r] = 0
    next_state_temp[1, idx_t] = 0

    # Exit on the left side or the top
    idx_l = np.argwhere(next_state_temp[0, :] < 0)
    idx_b = np.argwhere(next_state_temp[1, :] < 0)
    next_state_temp[0, idx_l] = obs_space_size - 1
    next_state_temp[1, idx_b] = obs_space_size - 1
    return next_state_temp


class Predator():
    """
    Predator Object
    In: current states of all agents and size of the grid
    Fct: - closest target: Compute nearest neighbor to follow
         - follow_target: Make step in direction of closest agent
    """
    def __init__(self, agent_states, obs_space_size):
        self.obs_space_size = obs_space_size
        self.current_state = np.random.randint(obs_space_size, size=2)

        # Check if initial position of predator overlaps with agent
        overlaps = sum([np.array_equal(self.current_state,
                                       agent_states[temp])
                        for temp in agent_states])

        while overlaps != 0:
            # Sample new initial state until there is no overlap
            self.current_state = np.random.randint(obs_space_size,
                                                   size=2)
            overlaps = sum([np.array_equal(self.current_state,
                                           agent_states[temp])
                            for temp in agent_states])

        self.orientation = 0
        # Compute initial "target" agent to follow
        self.current_target = self.closest_target(agent_states)

    def closest_target(self, agent_states):
        """
        Compute the nearest neighbor based on the manhattan distance
        """
        agent_states = np.array(list(agent_states.values()))
        all_together = np.vstack((self.current_state, agent_states))
        dist = DistanceMetric.get_metric('chebyshev')
        distances = dist.pairwise(all_together)
        target = np.argsort(distances[0, :])[1] - 1
        return target

    def follow_target(self, agent_states):
        """
        Predator switches follows current target agent in 0.9 cases and
        only switches target in 0.1 cases
        """
        roll = np.random.random()
        if roll < 0.1:
            self.current_target = self.closest_target(agent_states)

        move = self.current_state - agent_states[self.current_target]
        for i in range(2):
            if move[i] > 1:
                move[i] = 1
            elif move[i] < -1:
                move[i] = -1

        for action, move_d in action_to_move.items():
            if (move == move_d).all():
                self.orientation = action
        self.current_state = step_agent(self.current_state, move,
                                        self.obs_space_size)


class SwarmEnv(gym.Env):
    """
    Main Complex Swarm Environment
    In: Default env parameters
    Fct: - step: Performs state transition for all agents and predator
         - reset: Resets the env to a clean initial state
         - change_position_id: Checks for state overlap between agents
         - swarm_reward: Computes global reward of state transition
         - render: Plots the current environment state
         - set_env_parameters: Set general params (num_agents, grid size)
         - set_reward_parameters: Set reward fct params
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # SET INITIAL ENVIRONMENT PARAMETERS
        self.num_agents = 4
        self.obs_space_size = 20
        self.action_space = spaces.Discrete(8)
        self.observation_space = np.zeros((self.obs_space_size,
                                           self.obs_space_size))
        self.random_placement = True
        self.done = None

        # SET INITIAL REWARD FUNCTION PARAMETERS
        self.attraction_thresh = 5  # distance greater is undesirable
        self.repulsion_thresh = 2  # distance smaller is undesirable
        self.predator_eat_rew = -10  # reward when eaten by predator

    def reset(self):
        """
        Sample initial placement of agents in grid until no overlap
        """
        if self.random_placement:
            states_temp = np.random.randint(self.obs_space_size,
                                            size=(2, self.num_agents))
            ch_id = self.change_position_id(states_temp, self.num_agents)

            # Continue sampling new random initial state if overlap in states
            while ch_id is not None:
                states_temp[:, ch_id] = np.random.randint(self.obs_space_size,
                                                          size=(2, len(ch_id)))
                ch_id = self.change_position_id(states_temp,
                                                self.num_agents)

        # Transform valid state array into dictionary
        self.current_state = dict(enumerate(states_temp.T))
        # Initialize agents to be randomly oriented initially
        self.orientation = dict(enumerate(np.random.randint(8, size=self.num_agents)))
        # Initialize the predator with the current initial state
        self.predator = Predator(self.current_state,
                                 self.obs_space_size)
        self.done = False
        return self.current_state

    def step(self, action, indiv_rewards=False):
        """
        Perform a state transition/reward calculation based on selected action
        """
        if self.done:
            raise RuntimeError("Episode has finished. Call env.reset() to start a new episode.")

        # Transform categorical actions into vector-valued transitions
        move = {}
        for key in action.keys():
            move[key] = action_to_move[action[key]]
        self.move = move
        # Set orientation of agents to most recent action
        self.orientation = action

        states_temp = step_all_agents(self.current_state, move,
                                      self.obs_space_size)

        ch_id = self.change_position_id(states_temp, self.num_agents)

        # Continue sampling new random initial state if overlap in states
        while ch_id is not None:
            random_move = np.random.randint(8, size=(1, len(ch_id)))
            states_temp[:, ch_id] = np.random.randint(self.obs_space_size,
                                                      size=(2, len(ch_id)))
            ch_id = self.change_position_id(states_temp, self.num_agents)

        # Update new state and perform a "follow-step" of the predator
        self.current_state = dict(enumerate(states_temp.T))
        predator_state = self.predator.current_state.copy()
        self.predator.follow_target(self.current_state)

        # Calculate the reward based on the transition and return meta info
        reward, self.done = self.swarm_reward(indiv_rewards)
        info = {"predator_state": predator_state,
                "predator_orientation": self.predator.orientation,
                "predator_next_state": self.predator.current_state.copy()}
        return self.current_state, reward, self.done, info

    def change_position_id(self, states_temp, num_agents):
        """
        Check for state overlap for first num_agents columns of state array
        """
        state_overlap = np.ones((num_agents, num_agents))
        np.fill_diagonal(state_overlap, 0)
        ch_id = []

        for i in range(num_agents):
            check_idx = np.where(state_overlap[i, :] == 1)[0]
            for j in range(len(check_idx)):
                if not np.array_equal(states_temp[:, i],
                                      states_temp[:, check_idx[j]]):
                    state_overlap[i, check_idx[j]] = 0
                    state_overlap[check_idx[j], i] = 0
                else:
                    ch_id.append(i)

        if np.sum(state_overlap) == 0:
            # Agent himself always "collides" - correct!
            return None
        else:
            # Return that overlap was detected
            return ch_id

    def swarm_reward(self, indiv_rewards):
        """
        Compute the global swarm reward based on multiple objectives:
            1. Survival: Neg reinforcement for collision with predator
            2. Repulsion: Neg reinforcement for too close agents
            3. Attraction: Pos reinforcment for agents in correct range
            4. Alignment: Pos reinforcement for similar movement dir
        """
        # Check if predator has "eaten" fish - terminate episode w neg reward
        overlaps = np.array([np.array_equal(self.predator.current_state,
                                            self.current_state[temp])
                                            for temp in self.current_state])
        if overlaps.sum() > 0:
            reward = self.predator_eat_rew
            done = True

            if indiv_rewards:
                reward = dict(zip(range(self.num_agents), self.num_agents*[0]))
                agent_id = np.argwhere(overlaps == 1)[0]
                reward[agent_id] = self.predator_eat_rew
        else:
            reward = 0
            # Cumulate rewards based on distance as well as alignment
            agent_states = np.array(list(self.current_state.values()))
            dist = DistanceMetric.get_metric('chebyshev')
            distances = dist.pairwise(agent_states)

            # Repulsion objective - Exclude agent themselves
            repulsions = (distances < self.repulsion_thresh)
            reward -= repulsions.sum() - self.num_agents

            # Attraction objective
            dist1 = (distances < self.attraction_thresh)
            dist2 = (distances >= self.repulsion_thresh)
            attractions = (dist1 == dist2)
            reward += attractions.sum()

            # Alignment - Cosine dissimilarity between all actions taken
            move_array = np.array([m for m in self.move.values()])
            unalign = cosine_distances(move_array)/2
            # Get upper triangle set diag to 0, sum over all elements, subtract
            np.fill_diagonal(unalign, 0)
            reward -= unalign.sum()

            # Normalize by the number of agents (twice - symmetry)
            reward /= 2*self.num_agents
            done = False

            if indiv_rewards:
                reward = dict(zip(range(self.num_agents), self.num_agents*[0]))
                for agent_id in range(self.num_agents):
                    reward[agent_id] -= repulsions[agent_id, :].sum()
                    reward[agent_id] += attractions[agent_id, :].sum()
                    reward[agent_id] -= unalign[agent_id, :].sum()
                    reward[agent_id] /= 2
        return reward, done

    def set_env_parameters(self, num_agents=4,
                           obs_space_size=20, verbose=True):
        # Reset the general env parameters
        self.num_agents = num_agents
        self.obs_space_size = obs_space_size

        if verbose:
            print("Swarm Environment Parameters have been set to:")
            print("\t Number of Agents: {}".format(self.num_agents))
            print("\t State Space: {}x{} Grid".format(self.obs_space_size,
                                                      self.obs_space_size))

    def set_reward_parameters(self, attraction_thresh,
                              repulsion_thresh, predator_eat_rew,
                              verbose=False):
        # Reset the reward function parameters
        self.attraction_thresh = attraction_thresh
        self.repulsion_thresh = repulsion_thresh
        self.predator_eat_rew = predator_eat_rew

        if verbose:
            print("Swarm Reward Parameters have been set to:")
            print("\t Attraction Threshold: {}".format(self.attraction_thresh))
            print("\t Repulsion Threshold: {}".format(self.repulsion_thresh))
            print("\t Predator Punishment: {}".format(self.predator_eat_rew))

    def render(self, mode='rgb_array', close=False):
        """
        Render the environment state
        """
        # Get agents state coordinates
        x = [self.current_state[state][0] for state in self.current_state]
        y = [self.current_state[state][1] for state in self.current_state]

        # Plot the empty grid
        fig, ax = plt.subplots(dpi=200)
        x_ax = np.linspace(0, self.obs_space_size-1)
        y_ax = np.linspace(0, self.obs_space_size-1)
        plot = ax.plot(x_ax, y_ax, linestyle="")

        # Define size of individual fish window in empty grid
        ax_width = ax.get_window_extent().width
        fig_width = fig.get_window_extent().width
        fig_height = fig.get_window_extent().height
        # fish_size = 0.25*ax_width/(fig_width*len(x))
        fish_size = 0.008*self.obs_space_size
        fish_axs = [None for i in range(len(x) + 1)]

        # Loop over all agents and create windows for respective positions
        for i in range(len(x)):
            loc = ax.transData.transform((x[i], y[i]))
            fish_axs[i] = fig.add_axes([loc[0]/fig_width-fish_size/2,
                                        loc[1]/fig_height-fish_size/2,
                                        fish_size, fish_size], anchor='C')

            fish_axs[i].imshow((fish_imgs[self.orientation[i]]*255).astype(np.uint8))
            fish_axs[i].axis("off")

        # Add the predator as final axes object
        loc = ax.transData.transform((self.predator.current_state[0],
                                      self.predator.current_state[1]))
        orientation = self.predator.orientation
        fish_axs[len(x)] = fig.add_axes([loc[0]/fig_width-fish_size/2,
                                         loc[1]/fig_height-fish_size/2,
                                         fish_size, fish_size], anchor='C')

        fish_axs[len(x)].imshow((predator_imgs[orientation]*255).astype(np.uint8))
        fish_axs[len(x)].axis("off")

        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        plt.show()


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
