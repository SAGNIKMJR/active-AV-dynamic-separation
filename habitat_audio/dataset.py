import attr
import gzip
import json
import os
import logging
from typing import List, Optional, Dict

from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    ShortestPathPoint,
)


ALL_SCENES_MASK = "*"
CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_dataset/"


@attr.s(auto_attribs=True, kw_only=True)
class NavigationEpisodeCustom(NavigationEpisode):
    r"""Class for episode specification that includes all geodesic distances.
    Args:
        all_geodesic_distances: geodesic distances of agent to source
            and in between two sources
    """

    all_geodesic_distances: Optional[Dict[str, str]] = None
    gt_actions: Optional[Dict[str, str]] = None


@registry.register_dataset(name="AAViDSS")
class AAViDSSDataset(Dataset):
    r"""Class inherited from Dataset that loads Point Navigation dataset.
    """

    episodes: List[NavigationEpisodeCustom]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    @staticmethod
    def get_sounds_in_split(config: Config) -> str:
        return []

    @staticmethod
    def get_scenes_to_load(config: Config) -> List[str]:
        r"""Return list of scene ids for which dataset has separate files with
        episodes.
        """
        assert AAViDSSDataset.check_config_paths_exist(config), \
            (config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT), config.SCENES_DIR)
        dataset_dir = os.path.dirname(
            config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT)
        )

        cfg = config.clone()
        cfg.defrost()
        cfg.CONTENT_SCENES = []
        dataset = AAViDSSDataset(cfg)
        return AAViDSSDataset._get_scenes_from_folder(
            content_scenes_path=dataset.content_scenes_path,
            dataset_dir=dataset_dir,
        )

    @staticmethod
    def _get_scenes_from_folder(content_scenes_path, dataset_dir):
        scenes = []
        content_dir = content_scenes_path.split("{scene}")[0]
        scene_dataset_ext = content_scenes_path.split("{scene}")[1]
        content_dir = content_dir.format(data_path=dataset_dir)
        if not os.path.exists(content_dir):
            return scenes

        for filename in os.listdir(content_dir):
            if filename.endswith(scene_dataset_ext):
                scene = filename[: -len(scene_dataset_ext)]
                scenes.append(scene)
        scenes.sort()
        return scenes

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []
        self._config = config

        if config is None:
            return

        datasetfile_path = config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT)
        with gzip.open(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR, scene_filename=datasetfile_path)

        # Read separate file for each scene
        dataset_dir = os.path.dirname(datasetfile_path)
        scenes = config.CONTENT_SCENES
        if ALL_SCENES_MASK in scenes:
            scenes = AAViDSSDataset._get_scenes_from_folder(
                content_scenes_path=self.content_scenes_path,
                dataset_dir=dataset_dir,
            )

        last_episode_cnt = 0
        for scene in scenes:
            scene_filename = self.content_scenes_path.format(
                data_path=dataset_dir, scene=scene
            )
            with gzip.open(scene_filename, "rt") as f:
                self.from_json(f.read(), scenes_dir=config.SCENES_DIR, scene_filename=scene_filename)

            num_episode = len(self.episodes) - last_episode_cnt
            last_episode_cnt = len(self.episodes)
            logging.info('Sampled {} from {}'.format(num_episode, scene))

    def filter_by_ids(self, scene_ids):
        episodes_to_keep = list()

        for episode in self.episodes:
            for scene_id in scene_ids:
                scene, ep_id = scene_id.split(',')
                if scene in episode.scene_id and ep_id == episode.episode_id:
                    episodes_to_keep.append(episode)

        self.episodes = episodes_to_keep

    # filter by scenes for data collection
    def filter_by_scenes(self, scenes):
        episodes_to_keep = list()
        for episode in self.episodes:
            episode_scene = episode.scene_id.split("/")[-1].split(".")[0]
            if episode_scene in scenes:
                episodes_to_keep.append(episode)
        self.episodes = episodes_to_keep

    # filter by scenes for data collection
    def filter_by_scenes_n_ids(self, scenes_n_ids):
        episodes_to_keep = list()
        for episode in self.episodes:
            episode_scene = episode.scene_id.split("/")[-1].split(".")[0]
            episode_id = int(episode.episode_id)
            if episode_scene + "_" + str(episode_id) in scenes_n_ids:
                episodes_to_keep.append(episode)
        self.episodes = episodes_to_keep

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None, scene_filename: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        episode_cnt = 0
        for episode in deserialized["episodes"]:
            episode = NavigationEpisodeCustom(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX):
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)
            self.episodes.append(episode)
            episode_cnt += 1
