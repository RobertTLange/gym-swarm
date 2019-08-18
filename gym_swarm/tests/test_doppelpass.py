import os
import unittest
import gym
import gym_swarm
import numpy as np


class Environment(unittest.TestCase):

    def test_env_make(self):
        gym.make("Doppelpass-v0")

    def test_env_reset(self):
        env = gym.make("Doppelpass-v0")
        env.reset()

    def test_env_step(self):
        env = gym.make("Doppelpass-v0")
        env.reset()
        action = {0: np.array([0.5, 1, 0, 0]),
                  1: np.array([0.33, 0, 1, 1])}
        state, reward, done, info = env.step(action)
        self.assertEqual(len(state), env.num_agents)

    def test_env_set_params(self):
        env = gym.make("Doppelpass-v0")

        env_params = {"obs_space_size": 50,
                      "pickup_range": 1,
                      "observation_range": 8,
                      "observation_resolution": 20,
                      "v_bounds": [-2, 2],
                      "a_bounds": [-1, 1],
                      "goal": 2,
                      "required_key_passes": 5,
                      "random_placement": True}

        reward_params = {"wrong_pickup_reward": -3,
                         "correct_pickup_reward": 5,
                         "wrong_pass_reward": -4,
                         "correct_pass_reward": 10,
                         "wrong_putdown_reward": -3.4,
                         "goal_reach_reward": 200}

        env.set_env_params(env_params, reward_params)
        env.reset()

        # Check environment parameters
        self.assertEqual(50, env.obs_space_size)
        self.assertEqual(1, env.pickup_range)
        self.assertEqual(8, env.observation_range)
        self.assertEqual(20, env.observation_resolution)
        self.assertEqual([-2, 2], env.v_bounds)
        self.assertEqual([-1, 1], env.a_bounds)
        self.assertEqual(2, env.goal)
        self.assertEqual(5, env.required_key_passes)
        self.assertEqual(1, env.random_placement)

        # Check reward parameters
        self.assertEqual(-3, env.wrong_pickup_reward)
        self.assertEqual(5, env.correct_pickup_reward)
        self.assertEqual(-4, env.wrong_pass_reward)
        self.assertEqual(10, env.correct_pass_reward)
        self.assertEqual(-3.4, env.wrong_putdown_reward)
        self.assertEqual(200, env.goal_reach_reward)


if __name__ == '__main__':
    unittest.main()
