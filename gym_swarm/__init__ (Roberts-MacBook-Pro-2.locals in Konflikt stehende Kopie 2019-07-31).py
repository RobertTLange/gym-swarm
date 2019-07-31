import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Swarm-v0',
    entry_point='gym_swarm.envs:SwarmEnv',
)

register(
    id='Shepherd-v0',
    entry_point='gym_swarm.envs:ShepherdEnv',
)

register(
    id='Doppelpass-v0',
    entry_point='gym_swarm.envs:Doppelpass1DEnv',
)
