import os
import unittest
import gym
import gym_swarm


class Environment(unittest.TestCase):

    def test_hanoi_env_make(self):
        gym.make("Swarm-v0")

    def test_hanoi_env_reset(self):
        env = gym.make("Swarm-v0")
        env.reset()

    def test_swarm_env_step(self):
        env = gym.make("Swarm-v0")
        env.reset()
        state, reward, done, info = env.step(0)
        self.assertEqual(len(state), env.num_agents)
        self.assertEqual(env.env_noise, 0)


if __name__ == '__main__':
    unittest.main()
