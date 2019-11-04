import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Discrete-Swarm-v0',
    entry_point='gym_swarm.envs:DiscreteSwarmEnv',
)

register(
    id='Continuous-Swarm-v0',
    entry_point='gym_swarm.envs:ContinuousSwarmEnv',
)

register(
    id='Shepherd-v0',
    entry_point='gym_swarm.envs:ShepherdEnv',
)

register(
    id='Doppelpass-v0',
    entry_point='gym_swarm.envs:Doppelpass1DEnv',
)

register(
    id='Vision-v0',
    entry_point='gym_swarm.envs:FilterGridworldEnv',
)
