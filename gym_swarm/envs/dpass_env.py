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

# Read in fish and predator emojis to plot episodes
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


class key_to_goal():
<<<<<<< HEAD
    def __init__(self, obs_space_size, pickup_range):
=======
    def __init__(self, obs_space_size, pickup_range=0.5):
>>>>>>> 83baa31380f616607ccae38232414129a23b0891
        """
        Key object that has to be passed twice before walking to goal with it
        - init_position: Initial position of the key
        - pickup_nhood: neighborhood in which the key can be picked up
        """
<<<<<<< HEAD
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
=======
        # Set range around key in which it can be picked up
        self.pickup_range = pickup_range
        # Random initialzation of key location and reset ownership/pass counter
        self.initialize_key(obs_space_size)
        # Calculate nhood in which pickup is allowed (adapt with key location)
        self.update_pickup_nhood()

    def initialize_key(self, obs_space_size):
        self.position = np.random.uniform(low=0, high=obs_space_size, size=1)
>>>>>>> 83baa31380f616607ccae38232414129a23b0891
        self.ownership = None
        self.pass_count = 0

    def update_pickup_nhood(self):
        self.pickup_nhood = [self.position-self.pickup_range/2,
                             self.position+self.pickup_range/2]

<<<<<<< HEAD
    def attempt_key_pickup(self, agent_id, agents_positions):
=======
    def attempt_pickup(self, agent_id, agents_positions):
>>>>>>> 83baa31380f616607ccae38232414129a23b0891
        # Check if agent is within pickup range of key and update key position
        if self.pickup_nhood[0] <= agents_positions[agent_id] <= self.pickup_nhood[1]:
            self.ownership = agent_id
            self.move_with_owner(agents_positions)
            return True
        else:
            return False

    def attempt_key_pass(self, agent_id, agents_positions):
        # Check if key is in possession of one agent and the other is in
        # passing distance to pass key - change ownership if so
<<<<<<< HEAD
        # NOTE: In 2 agent case we can simply do 1-id to index other agent
=======
        # Note: In 2 agent case we can simply do 1-id to index other agent
>>>>>>> 83baa31380f616607ccae38232414129a23b0891
        if self.pickup_nhood[0] <= agents_positions[1-agent_id] <= self.pickup_nhood[1]:
            self.ownership = 1-agent_id
            self.move_with_owner(agents_positions)
            return True
        else:
            return False

    def move_with_owner(self, agents_positions):
<<<<<<< HEAD
        # Call at every env step - Check key ownership & update position
=======
        # Check if agent is currently being held by agent & update key position
>>>>>>> 83baa31380f616607ccae38232414129a23b0891
        if self.ownership is not None:
            self.position = agents_positions[self.ownership]
            self.update_pickup_nhood()


class Doppelpass1DEnv(gym.Env):
    """
    "1D" Continuous Action Space Doppelpass Environment
    2 Agent env in which agents have to pick up a key and exchange it twice
<<<<<<< HEAD
    before reaching a goal location to successfully end the episode
=======
    before reaching a goal location
>>>>>>> 83baa31380f616607ccae38232414129a23b0891
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # SET INITIAL ENVIRONMENT PARAMETERS
<<<<<<< HEAD
        self.num_agents = 2                 # No. agents in env - 2 for now
        self.obs_space_size = 20            # Size of 1d line [0, 20]
        self.pickup_range = 0.5             # Nhood to pick up/pass key in
        self.observation_range = 10         # Nhood observed by agents
        self.observation_resolution = 5    # Categorical obs bins in nhood
        self.done = None

        # SET MAX//MIN VELOCITY AND ACCELARATION
        self.v_bounds = [-1, 1]             # Clip velocity into range
        self.a_bounds = [-0.5, 0.5]         # Clip acceleration into range

        # FIX GOAL POSITION TO BE AT BORDER OF ENV - SAMPLE LATER ON
        self.goal = self.obs_space_size     # Set goal position to end of line
=======
        self.num_agents = 2
        self.obs_space_size = 20
        self.random_placement = True
        self.done = None

        # SET MAX//MIN VELOCITY AND ACCELARATION
        self.v_bounds = [-1, 1]
        self.a_bounds = [-1, 1]

        # FIX GOAL POSITION TO BE AT BORDER OF ENV - SAMPLE LATER ON
        self.goal_position = self.obs_space_size
>>>>>>> 83baa31380f616607ccae38232414129a23b0891

        # SET INITIAL REWARD FUNCTION PARAMETERS
        self.wrong_pickup_reward = -1       # R - wrong attempt to pick up key
        self.correct_pickup_reward = 2      # R - picking up key
        self.wrong_exchange_reward = -1     # R - wrong attempt to exchange key
        self.correct_exchange_reward = 1    # R - exchanging key between agents
        self.correct_placement_reward = 10  # R - wrong attempt to exchange key

<<<<<<< HEAD
        # INITIALIZE THE KEY
        self.key = key_to_goal(self.obs_space_size, self.pickup_range)

=======
>>>>>>> 83baa31380f616607ccae38232414129a23b0891
    def reset(self):
        """
        Sample initial placement of agents on straight line w. no velocity
        """
<<<<<<< HEAD
        self.key.initialize_key()
        self.agents_postions = np.random.uniform(low=0,
                                                high=self.obs_space_size,
                                                size=self.num_agents)
        self.agents_velocities = np.repeat(0, self.num_agents)

        # CONSTRUCT STATE - FULLY OBSERVED
        self.state = self.get_current_state()

        # CONSTRUCT OBSERVATION - PARTIALLY OBSERVABLE
        self.observations = self.get_obs_from_state()
        self.done = False
        print(self.state)
        print(self.observations)
        return self.observations
=======
        if self.random_placement:
            self.current_postion = np.random.uniform(low=0,
                                                     high=self.obs_space_size,
                                                     size=self.num_agents)
            self.current_velocity = np.repeat(0, self.num_agents)

            self.key_position =
        self.done = False
        return self.current_state

    def try_pickup_key(self, agent_id):
        # TODO: Implement check of agent position - if so change ownership of key, etc.

        return
>>>>>>> 83baa31380f616607ccae38232414129a23b0891

    def step(self, action):
        """
        Perform a state transition/reward calculation based on selected action
<<<<<<< HEAD
        -> action: 3d array. 1d = acceleration, 2d = pickup boolean, 3d = pass boolean
=======
        -> action: Collective action dictionary for all agents
>>>>>>> 83baa31380f616607ccae38232414129a23b0891
        """
        if self.done:
            raise RuntimeError("Episode has finished. Call env.reset() to start a new episode.")

<<<<<<< HEAD
        # NOTE: Be careful with async execution of actions
=======
>>>>>>> 83baa31380f616607ccae38232414129a23b0891
        for agent_id, action in action.items():
            # Clip accelaration into range
            v_a = np.clip(self.current_velocity + action[0],
                          self.v_bounds[0], self.v_bounds[1])
            next_position_a = np.clip(self.current_position[agent_id] + v_a,
                                      0, self.obs_space_size)

<<<<<<< HEAD
            pickup_success, pass_success = None, None
            if action[1]:
                # Check whether agent can execute pickup
                pickup_success = self.key.attempt_key_pickup()
            if action[2]:
                # Check whether agents can execute pass of key
                pass_success = self.key.attempt_key_pass()

=======
>>>>>>> 83baa31380f616607ccae38232414129a23b0891
            # TODO: Implement pickup/pass actions
            # TODO: Update the actual state of the agents

        # Calculate the reward based on the transition and return meta info
        reward, self.done = self.reward_doppelpass()
        info = {"warnings": None}
        return self.current_state, reward, self.done, info

<<<<<<< HEAD
    def get_current_state(self):
        # Collect all state relevant variables in a dictionary
        return {"key_position": self.key.position,
                "key_ownership": self.key.ownership,
                "key_pass_count": self.key.pass_count,
                "agents_positions": self.agents_postions,
                "agents_velocities": self.agents_velocities,
                "goal": self.goal}

    def get_obs_from_state(self):
        # Compute the obs from the state representation
        # Dims of obs:
            # 1. Location on line - x
            # 2. Velocity of agent - vx
            # 3. Key in posession of agent - boolean
            # 4. Key pass count - integer
            # 5. - 3xNoBins - Stacked bin nhood vectors (key, other agent, goal)
        # NOTE: Stack the different channels one vector representation
        observation = {0: np.zeros(4 + 3*self.observation_resolution),
                       1: np.zeros(4 + 3*self.observation_resolution)}

        # General nhood repr: range/2|--------agent--------|range/2
        # Perform binary seach to find closest bin to place boolean in
        bins = np.linspace(-self.observation_range/2, self.observation_range/2,
                           self.observation_resolution)

        for agent in range(self.num_agents):
            observation[agent][0] = self.state["agents_positions"][agent]
            observation[agent][1] = self.state["agents_velocities"][agent]
            if self.state["key_ownership"] == agent:
                observation[agent][2] = 1
            observation[agent][3] = self.state["key_pass_count"]

            # 1. Fill key in nhood preception array
            key_plane = np.zeros(self.observation_resolution)
            dist_to_key = self.state["agents_positions"][agent] - self.state["key_position"]
            if np.abs(dist_to_key) <= self.observation_range/2:
                key_bin_id = np.searchsorted(bins, dist_to_key, side="left")
                key_plane[key_bin_id] = 1

            # 2. Fill other agent in nhood preception array
            other_agent_plane = np.zeros(self.observation_resolution)
            dist_to_agent = self.state["agents_positions"][agent] - self.state["agents_positions"][1-agent]
            if np.abs(dist_to_agent) <= self.observation_range/2:
                agent_bin_id = np.searchsorted(bins, dist_to_agent, side="left")
                other_agent_plane[agent_bin_id] = 1

            # 3. Fill goal location in nhood preception array
            goal_plane = np.zeros(self.observation_resolution)
            dist_to_goal = self.state["goal"] - self.state["agents_positions"][agent]
            if np.abs(dist_to_goal) <= self.observation_range/2:
                goal_bin_id = np.searchsorted(bins, dist_to_goal, side="left")
                goal_plane[goal_bin_id] = 1

            observation[agent][4:] = np.hstack([key_plane, other_agent_plane,
                                                goal_plane])

        return observation

    def reward_doppelpass(self):
        return reward, done

    def set_doppelpass_params(self):
        return

=======
    def reward_doppelpass(self, reward_type):
        return reward, done

>>>>>>> 83baa31380f616607ccae38232414129a23b0891
    def render(self, mode='rgb_array', close=False):
        """
        Render the environment state
        """
        return
<<<<<<< HEAD


if __name__ == "__main__":
    env = Doppelpass1DEnv()
    env.reset()
=======
>>>>>>> 83baa31380f616607ccae38232414129a23b0891
