# gym-swarm
## An OpenAI Gym Style Environment for Complex Swarm Behavior
## Author: Robert Tjarko Lange
## Date: February 2018

![](gym_swarm/images/env_illustration.png)

## Action and State Space

The state space is defined to be a discrete grid of variable size. You can set the size of the grid (see below). Within this grid each agent has access to 8 different actions and can instantaneously move in any direction (l, l-u, l-d, r, r-u, r-d, d, u).

Furthermore, the environment features a predator which initially computes its nearest neighbor (in terms of Manhattan Distance). This agent is assigned to be the target. Afterwards, the predator continually follows the chosen target. In 10 percent of the transitions the predator computes the nearest neighbor and updates its target.

An episode ends once the predator collides with an agent.

## Reward function

The reward function is formalized by four objectives:

![](gym_swarm/images/reward.png)

You can set the attraction, repulsion thresholds as well as the predator collision reward. See below.

## How to use this environment

# Installation

* Clone the repository and install the package.
```
git clone https://github.com/RobertTLange/gym-swarm
cd gym-swarm
pip install -e .  (if you use Python 2.)
python setupy.py install  (if you use Python 3.)
```

# Usage

Import and setup:
![](gym_swarm/images/setup_env.png)

Resetting the parameters of the environment:
![](gym_swarm/images/set_params.png)

# Notes
* Environment is especially suited for prototyping solutions to multi-agent RL problems which require communication.
* Following format guide in https://github.com/openai/gym/tree/master/gym/envs#how-to-create-new-environments-for-gym.
