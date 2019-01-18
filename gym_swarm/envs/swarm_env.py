import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import numpy as np

from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data

from sklearn.neighbors import NearestNeighbors

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
    temp = agent_state + move_agent
    # x/y-Axis turnover - Periodic boundary conditions
    for i in range(2):
        if temp[i] > (obs_space_size - 1):
            temp[i] = 0
        elif temp[i] < 0:
            temp[i] = obs_space_size - 1
    return temp


class Predator():
    def __init__(self, agent_states, obs_space_size):
        self.obs_space_size = obs_space_size
        self.current_state = np.random.randint(obs_space_size, size=2)

        overlaps = sum([np.array_equal(self.current_state,
                                       agent_states[temp])
                        for temp in agent_states])
        while overlaps != 0:
            self.current_state = np.random.randint(obs_space_size,
                                                   size=2)

        self.current_target = self.closest_target(agent_states)

    def closest_target(self, agent_states):
        agent_states = np.array(list(agent_states.values()))
        all_together = np.vstack((self.current_state, agent_states))
        nbrs = NearestNeighbors(n_neighbors=2,
                                algorithm='ball_tree').fit(all_together)
        distances, indices = nbrs.kneighbors(all_together)

        if indices[0, 1] > 0:
            target = indices[0, 1] - 1
        else:
            target = indices[0, 0] - 1
        return target

    def follow_target(self, agent_states):
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
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.num_agents = 4
        self.obs_space_size = 20
        self.action_space = spaces.Discrete(8)

        self.observation_space = np.zeros((self.obs_space_size,
                                           self.obs_space_size))

        self.current_state = dict(zip(np.arange(self.num_agents),
                                      [np.empty(2)]*self.num_agents))

        self.move = dict(enumerate(range(self.num_agents)))
        self.random_placement = True
        self.done = None
        self.goal_state = None

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode has finished. Call env.reset() to start a new episode.")

        move = {}
        for key in action.keys():
            move[key] = action_to_move[action[key]]

        self.orientation = action
        # Asynchronous execution of moves - if collision random move instead
        # Implement border conditions - agent enter at opposite ends again

        states_temp = self.current_state.copy()
        for i in self.current_state.keys():
            states_temp[i] = step_agent(self.current_state[i], move[i],
                                        self.obs_space_size)
            # Check for collision with previous movers
            if i > 0:
                while self.invalid_position(np.array([states_temp[state] for state in states_temp]).T, i+1):
                    random_action = np.random.randint(8, size=1)[0]
                    states_temp[i] = step_agent(self.current_state[i],
                                                action_to_move[random_action],
                                                self.obs_space_size)

        self.current_state = states_temp
        self.predator.follow_target(self.current_state)

        reward, self.done = self.swarm_reward()
        info = {"predator_state": self.predator.current_state,
                "predator_orientation": self.predator.orientation}
        return self.current_state, reward, self.done, info

    def reset(self):
        if self.random_placement:
            # For each agent sample an initial placement in grid - uniform!
            states_temp = np.random.randint(self.obs_space_size,
                                            size=(2, self.num_agents))
            invalid = self.invalid_position(states_temp, self.num_agents)

            while invalid:
                states_temp = np.random.randint(self.obs_space_size,
                                                size=(2, self.num_agents))
                invalid = self.invalid_position(states_temp, self.num_agents)

        # Transform valid state array into dictionary
        self.current_state = dict(enumerate(states_temp.T))

        self.predator = Predator(self.current_state,
                                 self.obs_space_size)
        self.done = False
        return self.current_state

    def invalid_position(self, states_temp, num_agents):
        state_overlap = np.zeros((num_agents, num_agents))
        for i in range(num_agents):
            check_idx = np.where(state_overlap[i, :] == 0)[0]
            for j in range(len(check_idx)):
                if np.array_equal(states_temp[:, i],
                                  states_temp[:, check_idx[j]]):
                    state_overlap[i, j] = 1
                    state_overlap[j, i] = 1

        if np.sum(state_overlap) == num_agents:
            return False
        else:
            return True

    def swarm_reward(self):
        overlaps = sum([np.array_equal(self.predator.current_state,
                                       self.current_state[temp])
                        for temp in self.current_state])
        if overlaps > 0:
            reward = -100*self.num_agents
            done = True
        else:
            reward = 0
            agent_states = np.array(list(self.current_state.values()))

            nbrs = NearestNeighbors(n_neighbors=self.num_agents,
                                    algorithm='ball_tree').fit(agent_states)
            distances, indices = nbrs.kneighbors(agent_states)

            for agent in range(self.num_agents):
                # Attraction and repulsion objective - distances?
                reward += -0.5 * sum(distances[agent, 1:] < 2*self.obs_space_size/10)
                reward += -0.5 * sum(distances[agent, 1:] > 4*self.obs_space_size/10)

            # Alignment - Sum of agents facing in the same direction
            un, counts = np.unique(list(self.orientation), return_counts=True)
            reward += np.max(counts)
            done = False
        return reward, done

    def render(self, mode='rgb_array', close=False):
        x = [self.current_state[state][0] for state in self.current_state]
        y = [self.current_state[state][1] for state in self.current_state]

        fig, ax = plt.subplots(dpi=200)
        x_ax = np.linspace(0, self.obs_space_size-1)
        y_ax = np.linspace(0, self.obs_space_size-1)
        plot = ax.plot(x_ax, y_ax, linestyle="")

        ax_width = ax.get_window_extent().width
        fig_width = fig.get_window_extent().width
        fig_height = fig.get_window_extent().height
        fish_size = 0.25*ax_width/(fig_width*len(x))
        fish_axs = [None for i in range(len(x) + 1)]

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

    def set_env_parameters(self, num_agents=4,
                           obs_space_size=20, verbose=True):
        self.num_agents = num_agents
        self.obs_space_size = obs_space_size

        if verbose:
            print("Swarm Environment Parameters have been set to:")
            print("\t Number of Agents: {}".format(self.num_agents))
            print("\t State Space: {}x{} Grid".format(self.obs_space_size,
                                                      self.obs_space_size))


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

# if __name__ == "__main__":
#     # Visualize all different agent orientations
#     plt.figure(figsize=(15, 12), dpi=200)
#     counter = 1
#     for key in fish_imgs.keys():
#         print(fish_imgs[key].shape)
#         plt.subplot(3, 3, counter)
#         plt.imshow(fish_imgs[key])
#         counter += 1
