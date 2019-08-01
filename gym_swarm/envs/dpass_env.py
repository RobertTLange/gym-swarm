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

# Read in emojis to plot episodes
goal_img = plt.imread(get_sample_data(dir_path + "/images/goal.png"))
key_img = plt.imread(get_sample_data(dir_path + "/images/key.png"))
fish_img = plt.imread(get_sample_data(dir_path + "/images/fish_tropical.png"))
fish_inv_img = np.flip(fish_img, axis=1)

fish_imgs = {0: fish_img, 1: ndimage.rotate(fish_inv_img, 0)}


class key_to_goal():
    def __init__(self, obs_space_size, pickup_range):
        """
        Key object that has to be passed twice before walking to goal with it
        - init_position: Initial position of the key
        - pickup_nhood: neighborhood in which the key can be picked up
        """
        # Set observation space & range around key in which it can be picked up
        self.obs_space_size = obs_space_size
        self.pickup_range = pickup_range
        # Random initialzation of key location and reset ownership/pass counter
        self.initialize_key()
        # Calculate nhood in which pickup is allowed (adapt with key location)
        self.update_pickup_nhood()

    def initialize_key(self):
        self.position = np.random.uniform(low=0, high=self.obs_space_size,
                                          size=1)
        self.ownership = None
        self.pass_count = 0
        self.pickup_count = 0

    def update_pickup_nhood(self):
        self.pickup_nhood = [self.position-self.pickup_range/2,
                             self.position+self.pickup_range/2]

    def attempt_key_pickup(self, agent_id, agents_positions):
        # Check if agent is within pickup range of key and update key position
        no_ownership = (self.ownership is None)
        in_nhood = (self.pickup_nhood[0] <= agents_positions[agent_id] <= self.pickup_nhood[1])
        if no_ownership and in_nhood:
            self.ownership = agent_id
            self.pickup_count += 1
            return True
        else:
            return False

    def attempt_key_putdown(self, agent_id):
        if self.ownership == agent_id:
            self.ownership = None
            return True
        else:
            return False

    def attempt_key_pass(self, agent_id, agents_positions):
        # Check if key is in possession of one agent and the other is in
        # passing distance to pass key - change ownership if so
        # NOTE: In 2 agent case we can simply do 1-id to index other agent
        correct_ownership = (self.ownership == agent_id)
        in_nhood = (self.pickup_nhood[0] <= agents_positions[1-agent_id] <= self.pickup_nhood[1])

        if correct_ownership and in_nhood:
            self.ownership = 1-agent_id
            self.pass_count += 1
            return True
        else:
            return False

    def move_with_owner(self, agents_positions):
        # Call at every env step - Check key ownership & update position
        if self.ownership is not None:
            self.position = agents_positions[self.ownership]
            self.update_pickup_nhood()


class Doppelpass1DEnv(gym.Env):
    """
    "1D" Continuous Action Space Doppelpass Environment
    2 Agent env in which agents have to pick up a key and exchange it twice
    before reaching a goal location to successfully end the episode
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # SET INITIAL ENVIRONMENT PARAMETERS
        self.num_agents = 2                 # No. agents in env - 2 for now
        self.obs_space_size = 20            # Size of 1d line [0, 20]
        self.pickup_range = 0.5             # Nhood to pick up/pass key in
        self.observation_range = 5          # Nhood observed by agents
        self.observation_resolution = 10    # Categorical obs bins in nhood
        self.done = None

        # SET MAX//MIN VELOCITY AND ACCELARATION
        self.v_bounds = [-1, 1]             # Clip velocity into range
        self.a_bounds = [-0.5, 0.5]         # Clip acceleration into range

        # FIX GOAL POSITION TO BE AT BORDER OF ENV - SAMPLE LATER ON
        self.goal = self.obs_space_size     # Set goal position to end of line
        self.required_key_passes = 2        # No. passes required before goal

        # SET REWARD FUNCTION PARAMETERS - pickup, pass, put down
        # NOTE: Pass is defined as exchange of ownership
        self.wrong_pickup_reward   = -1     # R - wrong attempt to pick up key
        self.correct_pickup_reward = 2      # R - picking up key
        self.wrong_pass_reward     = -1     # R - wrong attempt to exchange key
        self.correct_pass_reward   = 2      # R - exchanging key between agents
        self.wrong_putdown_reward  = -1     # R - wrong attempt to putdown key
        self.goal_reach_reward     = 10     # R - key + 2 passes brought to goal

        # INITIALIZE THE KEY
        self.key = key_to_goal(self.obs_space_size, self.pickup_range)

    def reset(self):
        """
        Sample initial placement of agents on straight line w. no velocity
        """
        self.key.initialize_key()
        self.agents_positions = np.random.uniform(low=0,
                                                  high=self.obs_space_size,
                                                  size=self.num_agents)
        self.agents_velocities = np.repeat(0., self.num_agents)

        # CONSTRUCT STATE - FULLY OBS & CONSTRUCT OBSERVATION - PARTIALLY OBS
        self.state = self.get_current_state()
        self.observations = self.get_obs_from_state()
        self.done = False
        return self.observations

    def step(self, action):
        """
        Perform a state transition/reward calculation based on selected action
        -> action: 3d array. 1d = accel, 2d = pickup boolean, 3d = pass boolean
        """
        if self.done:
            raise RuntimeError("Episode has finished. Call env.reset() to start a new episode.")

        # NOTE: Careful - async execution of actions - first update states - then categorical actions
        for agent_id, agent_action in action.items():
            # Clip accelarations into range (env specification)
            acc_clipped = np.clip(agent_action[0], self.a_bounds[0], self.a_bounds[1])
            # Clip position into range (env specification)
            self.agents_velocities[agent_id] = np.clip(self.agents_velocities[agent_id] + acc_clipped,
                                                       self.v_bounds[0], self.v_bounds[1])
            self.agents_positions[agent_id] = np.clip(self.agents_positions[agent_id] + self.agents_velocities[agent_id],
                                                      0, self.obs_space_size)

        # Update position of key before pass attempt - respect nhood!
        self.key.move_with_owner(self.agents_positions)

        # Define dictionary collecting success of action attempts by agents
        pickup_success = {0: None, 1: None}
        putdown_success = {0: None, 1: None}
        pass_success = {0: None, 1: None}

        for agent_id, agent_action in action.items():
            if agent_action[1]:  # Check whether agent can execute key pickup
                pickup_success[agent_id] = self.key.attempt_key_pickup(agent_id, self.agents_positions)
            if agent_action[2]:  # Check whether agent can execute key putdown
                putdown_success[agent_id] = self.key.attempt_key_putdown(agent_id)
            if agent_action[3]:  # Check whether agents can execute pass of key
                pass_success[agent_id] = self.key.attempt_key_pass(agent_id, self.agents_positions)

        # Update position of key after pass attempt - respect nhood!
        self.key.move_with_owner(self.agents_positions)

        # Update state of env and obs distributed to agents
        self.state = self.get_current_state()
        self.observations = self.get_obs_from_state()

        # Calculate the reward based on the transition and return meta info
        reward, self.done = self.reward_doppelpass(pickup_success,
                                                   putdown_success,
                                                   pass_success)
        info = {"warnings": None}
        return self.observations, reward, self.done, info

    def get_current_state(self):
        # Collect all state relevant variables in a dictionary
        return {"key_position": self.key.position,
                "key_ownership": self.key.ownership,
                "key_pass_count": self.key.pass_count,
                "agents_positions": self.agents_positions,
                "agents_velocities": self.agents_velocities,
                "goal": self.goal}

    def get_obs_from_state(self):
        """
        Compute the obs from the state representation - Dims of obs:
            1. Location on line - x
            2. Velocity of agent - vx
            3. Key in posession of agent - boolean
            4. Key pass count - integer
            5. - 3xNoBins - Stacked bin nhood vectors (key, other agent, goal)
        """
        # NOTE: Stack the different channels one vector representation
        observation = {0: np.zeros(4 + 3*self.observation_resolution),
                       1: np.zeros(4 + 3*self.observation_resolution)}

        # General nhood repr: range/2|--------agent--------|range/2
        # Perform binary seach to find closest bin to place boolean in
        bins = np.linspace(-self.observation_range/2, self.observation_range/2,
                           self.observation_resolution)

        for agent_id in range(self.num_agents):
            observation[agent_id][0] = self.state["agents_positions"][agent_id]
            observation[agent_id][1] = self.state["agents_velocities"][agent_id]
            if self.state["key_ownership"] == agent_id:
                observation[agent_id][2] = 1
            observation[agent_id][3] = self.state["key_pass_count"]

            # 1. Fill key in nhood preception array
            key_plane = np.zeros(self.observation_resolution)
            dist_to_key = self.state["agents_positions"][agent_id] - self.state["key_position"]
            if np.abs(dist_to_key) <= self.observation_range/2:
                key_bin_id = np.searchsorted(bins, dist_to_key, side="left")
                key_plane[key_bin_id] = 1

            # 2. Fill other agent in nhood preception array
            other_agent_plane = np.zeros(self.observation_resolution)
            dist_to_agent = self.state["agents_positions"][agent_id] - self.state["agents_positions"][1-agent_id]
            if np.abs(dist_to_agent) <= self.observation_range/2:
                agent_bin_id = np.searchsorted(bins, dist_to_agent, side="left")
                other_agent_plane[agent_bin_id] = 1

            # 3. Fill goal location in nhood preception array
            goal_plane = np.zeros(self.observation_resolution)
            dist_to_goal = self.state["goal"] - self.state["agents_positions"][agent_id]
            if np.abs(dist_to_goal) <= self.observation_range/2:
                goal_bin_id = np.searchsorted(bins, dist_to_goal, side="left")
                goal_plane[goal_bin_id] = 1

            observation[agent_id][4:] = np.hstack([key_plane,
                                                   other_agent_plane,
                                                   goal_plane])

        return observation

    def reward_doppelpass(self, pickup_success, putdown_success, pass_success):
        """
        Agent-specific rewards for key pickup, pass & putdown
        """
        # NOTE: Decide in learning loop whether to aggregate to global signal
        done = False
        reward = {0: 0, 1: 0}

        for agent_id in range(self.num_agents):
            # Compute pickup reward for agent - if None no pickup was attempted
            if pickup_success[agent_id] == True and self.key.pickup_count < 1:
                reward[agent_id] += self.correct_pickup_reward
            elif pickup_success[agent_id] == False:
                reward[agent_id] += self.wrong_pickup_reward

            # Compute pass reward for agent - if None no pass was attempted
            if pass_success[agent_id] == True and self.key.pass_count < self.required_key_passes:
                reward[agent_id] += self.correct_pass_reward
            elif pass_success[agent_id] == False:
                reward[agent_id] += self.wrong_pass_reward

            # Compute putdown reward for agent - if None no putdown was attempted
            goal_in_putdown_nhood = (np.absolute(self.goal - self.key.position) < self.pickup_range)
            if putdown_success[agent_id] == True and goal_in_putdown_nhood:
                reward[agent_id] += self.goal_reach_reward
                done = True
            elif putdown_success[agent_id] == False:
                reward[agent_id] += self.wrong_putdown_reward
        return reward, done

    def set_env_params(self, env_params=None, reward_params=None):
        # SET INITIAL ENVIRONMENT PARAMETERS
        if env_params is not None:
            self.obs_space_size = env_params["obs_space_size"]
            self.pickup_range = env_params["pickup_range"]
            self.observation_range = env_params["observation_range"]
            self.observation_resolution = env_params["observation_resolution"]

            # SET MAX//MIN VELOCITY AND ACCELARATION
            self.v_bounds = env_params["v_bounds"]
            self.a_bounds = env_params["a_bounds"]

            # FIX GOAL POSITION TO BE AT BORDER OF ENV - SAMPLE LATER ON
            self.goal = env_params["goal"]
            self.required_key_passes = env_params["required_key_passes"]

        # SET REWARD FUNCTION PARAMETERS - pickup, pass, put down
        if reward_params is not None:
            self.wrong_pickup_reward   = reward_params["wrong_pickup_reward"]
            self.correct_pickup_reward = reward_params["correct_pickup_reward"]
            self.wrong_pass_reward     = reward_params["wrong_pass_reward"]
            self.correct_pass_reward   = reward_params["correct_pass_reward"]
            self.wrong_putdown_reward  = reward_params["wrong_putdown_reward"]
            self.goal_reach_reward     = reward_params["goal_reach_reward"]
        return

    def render(self, mode='rgb_array', close=False):
        """
        Render the environment state
        """
        x = [self.state["agents_positions"][agent_id] for agent_id in range(self.num_agents)]
        y = [0.5 for agent_id in range(self.num_agents)]
        # Plot the empty grid/line with 1 width
        fig, ax = plt.subplots(dpi=200, figsize=(10, 1))
        x_ax = np.linspace(0, self.obs_space_size)
        y_ax = np.linspace(0, 1)
        plot = ax.plot(x_ax, y_ax, linestyle="")

        # Define size of individual fish window in empty grid
        ax_width = ax.get_window_extent().width
        fig_width = fig.get_window_extent().width
        fig_height = fig.get_window_extent().height
        # fish_size = 0.25*ax_width/(fig_width*len(x))
        fish_size = self.obs_space_size/100
        axs_to_plot = [None for i in range(len(x) + 2)]

        # Loop over all agents and create windows for respective positions
        for i in range(len(x)):
            loc = ax.transData.transform((x[i], y[i]))
            axs_to_plot[i] = fig.add_axes([loc[0]/fig_width-fish_size/2,
                                           loc[1]/fig_height-fish_size/2,
                                           fish_size, fish_size], anchor='C')

            # TODO: Add orientation based on velocity
            axs_to_plot[i].imshow((fish_imgs[0]*255).astype(np.uint8))
            axs_to_plot[i].axis("off")

        # Add the key and goal as final axes objects
        loc = ax.transData.transform((self.key.position[0], 0.5))
        axs_to_plot[len(x)] = fig.add_axes([loc[0]/fig_width-fish_size/2,
                                            loc[1]/fig_height-fish_size/2,
                                            fish_size, fish_size], anchor='C')

        axs_to_plot[len(x)].imshow((key_img*255).astype(np.uint8))
        axs_to_plot[len(x)].axis("off")

        loc = ax.transData.transform((self.goal, 0.5))
        axs_to_plot[len(x) + 1] = fig.add_axes([loc[0]/fig_width-fish_size/2,
                                                loc[1]/fig_height-fish_size/2,
                                                fish_size, fish_size], anchor='C')

        axs_to_plot[len(x) + 1].imshow((goal_img*255).astype(np.uint8))
        axs_to_plot[len(x) + 1].axis("off")

        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        plt.show()
        return


if __name__ == "__main__":
    env = Doppelpass1DEnv()
    observation = env.reset()
    print(env.state)
    action = {0: np.array([0.5, 1, 0, 0]),
              1: np.array([0.33, 0, 1, 1])}
    observation, reward, done, info = env.step(action)
    print(env.state)
    print(observation, reward, done, info)
    env.render()
