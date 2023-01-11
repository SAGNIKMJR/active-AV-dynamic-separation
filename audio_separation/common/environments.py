r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type
import logging

import habitat
from habitat import Config, Dataset
from audio_separation.common.baseline_registry import baseline_registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="AAViDSSEnv")
class AAViDSSEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._config = config
        self._core_env_config = config.TASK_CONFIG

        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._env_step = 0
        observation = super().reset()
        logging.debug(super().current_episode)
        return observation

    def step(self, *args, **kwargs):
        observation, reward, done, info = super().step(*args, **kwargs)
        self._env_step += 1
        return observation, reward, done, info

    def get_reward_range(self):
        return (
            float('-inf'),
            0
        )

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        done = False
        if self._env.episode_over:
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    # for data collection
    def get_current_episode_id(self):
        return self.habitat_env.current_episode.episode_id
