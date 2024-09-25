import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='SimplePacman-v0',
    entry_point='gym_envs.envs.gym_pacman.custom_pacman_env:PacmanEnv',
    kwargs={
        'config_file': ""
    }
)

register(
    id='DungeonCrawler-v0',
    entry_point='gym_envs.envs.gym_dungeon.custom_dungeon_env:DungeonCrawlerEnv',
    kwargs={
        'config_file': ""
    }
)

register(
    id='SimpleBankHeist-v0',
    entry_point='gym_envs.envs.gym_bankheist.custom_bankheist_env:BankHeistEnv',
    kwargs={
        'config_file': ""
    }
)
