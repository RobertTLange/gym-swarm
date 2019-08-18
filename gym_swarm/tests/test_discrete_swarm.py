import os
import unittest
import gym
import gym_swarm


class Environment(unittest.TestCase):

    def test_swarm_env_make(self):
        gym.make("Discrete-Swarm-v0")

    def test_swarm_env_reset(self):
        env = gym.make("Discrete-Swarm-v0")
        env.reset()

    def test_swarm_env_step(self):
        env = gym.make("Discrete-Swarm-v0")
        env.reset()
        action = {0: 0, 1: 1, 2: 2, 3: 3}
        state, reward, done, info = env.step(action)
        self.assertEqual(len(state), env.num_agents)

    def test_set_env_params(self):
        env = gym.make("Discrete-Swarm-v0")
        env.set_env_parameters(num_agents=10, obs_space_size=200,
                               verbose=False)
        env.reset()
        self.assertEqual(10, env.num_agents)
        self.assertEqual(200, env.obs_space_size)

    def test_set_reward_params(self):
        env = gym.make("Discrete-Swarm-v0")
        env.set_reward_parameters(attraction_thresh=10,
                                  repulsion_thresh=200,
                                  predator_eat_rew=-50,
                                  verbose=False)
        env.reset()
        self.assertEqual(10, env.attraction_thresh)
        self.assertEqual(200, env.repulsion_thresh)
        self.assertEqual(-50, env.predator_eat_rew)

if __name__ == '__main__':
    unittest.main()
