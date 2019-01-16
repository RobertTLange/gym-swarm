# gym-swarm
## An OpenAI Gym Style Environment for Complex Swarm Behavior
## Author: Robert Tjarko Lange
## Date: 09/2018

This repository extends the OpenAI gym with a complex swarm environment.

* Environment Description:

# Action Space
Each agent has access to 8 different actions

# State Space


# Installation

* Clone the repository and install the package.
```
git clone https://github.com/RobertTLange/gym-swarm
cd gym_hanoi
pip install -e .  (if you use Python 2.)
python setupy.py install  (if you use Python 3.)
```

Depending on your operating system you might have to install mplcairo in order to render the nice fish! See [here](https://towardsdatascience.com/how-i-got-matplotlib-to-plot-apple-color-emojis-c983767b39e0) for a wonderful installation guide.

# Notes
* Environment is especially suited for prototyping solutions to long-term credit assignment problems, sparse rewards and curriculum learning.
* Following format guide in https://github.com/openai/gym/tree/master/gym/envs#how-to-create-new-environments-for-gym.
