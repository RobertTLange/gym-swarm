import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import numpy as np

from scipy import ndimage, misc
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data

dir_path = os.path.dirname(os.path.realpath(__file__))
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
        """
        * Inputs:
            - action: integer from 0 to 7 (see ACTION_LOOKUP)
        * Outputs:
            - current_state: dictionary of all agent states after transition
            - reward: cumulative reward of complete state transition
            - done: episode state
            - info: dict of booleans (noisy?/invalid action?)
        0. Check if transition is noisy or not
        1. Transform action (0 to 7 integer) to tuple move - see Lookup
        2. Check if move is allowed
        3. If it is change corresponding entry | If not return same state
        4. Check if episode completed and return
        """
        if self.done:
            raise RuntimeError("Episode has finished. Call env.reset() to start a new episode.")

        info = None

        move = {}
        for key in action.keys():
            move[key] = action_to_move[action[key]]

        self.orientation = action
        # Asynchronous execution of moves - if collision random move instead
        # Implement border conditions - agent enter at opposite ends again

        states_temp = self.current_state.copy()
        for i in self.current_state.keys():
            states_temp[i] = self.step_agent(i, move[i])
            # Check for collision with previous movers
            if i > 0:
                while self.invalid_position(np.array([states_temp[state] for state in states_temp]).T, i+1):
                    random_action = np.random.randint(8, size=1)
                    states_temp[i] = self.step_agent(i, action_to_move[random_action])

        reward = self.swarm_reward()

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

    def step_agent(self, i, move_agent):
        temp = self.current_state[i] + move_agent
        # x-Axis turnover
        if temp[0] > (self.obs_space_size - 1):
            temp[0] -= 0
        elif temp[0] < 0:
            temp[0] = self.obs_space_size - 1
        # y-Axis turnover
        if temp[1] > (self.obs_space_size - 1):
            temp[1] -= 0
        elif temp[1] < 0:
            temp[1] = self.obs_space_size - 1
        print(temp)
        return temp

    def swarm_reward(self):
        return 0

    def border_dynamics(self):
        return

    def render(self, mode='human', close=False):
        x = [self.current_state[state][0] for state in self.current_state]
        y = [self.current_state[state][1] for state in self.current_state]

        fig, ax = plt.subplots(dpi=200)
        x_ax = np.linspace(0, self.obs_space_size-1)
        y_ax = np.linspace(0, self.obs_space_size-1)
        plot = ax.plot(x_ax, y_ax, linestyle="")

        ax_width = ax.get_window_extent().width
        fig_width = fig.get_window_extent().width
        fig_height = fig.get_window_extent().height
        fish_size = 0.25*ax_width/(fig_width*0.5*len(x))
        fish_axs = [None for i in range(len(x))]

        for i in range(len(x)):
            loc = ax.transData.transform((x[i], y[i]))
            fish_axs[i] = fig.add_axes([loc[0]/fig_width-fish_size/2,
                                        loc[1]/fig_height-fish_size/2,
                                        fish_size, fish_size], anchor='C')

            fish_axs[i].imshow(fish_imgs[self.orientation[i]])
            fish_axs[i].axis("off")
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

if __name__ == "__main__":
    # Visualize all different agent orientations
    plt.figure(figsize=(15, 12), dpi=200)
    counter = 1
    for key in fish_imgs.keys():
        print(fish_imgs[key].shape)
        plt.subplot(3, 3, counter)
        plt.imshow(fish_imgs[key])
        counter += 1
