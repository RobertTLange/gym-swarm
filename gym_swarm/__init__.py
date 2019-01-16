import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Swarm-v0',
    entry_point='gym_hanoi.envs:SwarmEnv',
)
