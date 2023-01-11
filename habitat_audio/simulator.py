from typing import Any, List
from collections import defaultdict
import logging
import pickle
import os

import librosa
import numpy as np
import networkx as nx
from scipy.io import wavfile
from scipy.signal import fftconvolve

from habitat.core.registry import registry
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_to_angle_axis
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.core.simulator import (
    Config,
    AgentState, ShortestPathPoint)
from habitat_audio.utils import load_points_data, _to_tensor


class DummySimulator:
    def __init__(self):
        self.position = None
        self.rotation = None
        self._sim_obs = None

    def seed(self, seed):
        pass

    def set_agent_state(self, position, rotation):
        self.position = np.array(position, dtype=np.float32)
        self.rotation = rotation

    def get_agent_state(self):
        class State:
            def __init__(self, position, rotation):
                self.position = position
                self.rotation = rotation

        return State(self.position, self.rotation)

    def set_sensor_observations(self, sim_obs):
        self._sim_obs = sim_obs

    def get_sensor_observations(self):
        return self._sim_obs

    def close(self):
        pass


@registry.register_simulator()
class HabitatSimAudioEnabled(HabitatSim):
    r"""Changes made to simulator wrapper over habitat-sim

    This simulator allows the agent to be moved to location specified in the
    Args:
        config: configuration for initializing the simulator.
    """

    def action_space_shortest_path(self, source: AgentState, targets: List[AgentState], agent_id: int = 0) -> List[
            ShortestPathPoint]:
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        assert self.config.SCENE_DATASET in ["mp3d"], "SCENE_DATASET needs to be in ['mp3d']"
        self._source_position_indices = None
        self._receiver_position_index = None
        self._rotation_angle = None
        self._current_sound_names = None
        self._frame_cache = defaultdict(dict)
        self._is_episode_active = None
        self._position_to_index_mapping = dict()
        self._gt_bin_audio_components = []
        self._gt_mono_audio_components = []
        self._mixed_bin_audio_phase = None
        self._mono_audio_sampling_starting_idxs = []
        self._target_class = None
        self.points, self.graph = load_points_data(self.meta_dir, self.config.AUDIO.GRAPH_FILE,
                                                   scene_dataset=self.config.SCENE_DATASET)
        for node in self.graph.nodes():
            self._position_to_index_mapping[self.position_encoding(self.graph.nodes()[node]['point'])] = node

        if self.config.USE_RENDERED_OBSERVATIONS:
            logging.info('Loaded the rendered observations for all scenes')
            with open(self.current_scene_observation_file, 'rb') as fo:
                self._frame_cache = pickle.load(fo)
            self._sim.close()
            del self._sim
            self._sim = DummySimulator()

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        if not self.config.USE_RENDERED_OBSERVATIONS:
            agent_state = super().get_agent_state(agent_id)
        else:
            agent_state = self._sim.get_agent_state()

        return agent_state

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        if not self.config.USE_RENDERED_OBSERVATIONS:
            super().set_agent_state(position, rotation, agent_id=agent_id, reset_sensors=reset_sensors)
        else:
            pass

    @property
    def current_scene_observation_file(self):
        return os.path.join(self.config.RENDERED_OBSERVATIONS, self.config.SCENE_DATASET,
                            self.current_scene_name + '.pkl')

    @property
    def meta_dir(self):
        return os.path.join(self.config.AUDIO.META_DIR, self.current_scene_name)

    @property
    def current_scene_name(self):
        return self._current_scene.split('/')[-2]

    def reconfigure(self, config: Config) -> None:
        self.config = config
        self._current_sound_names = self.config.AGENT_0.SOUND_NAMES
        self._mono_audio_sampling_starting_idxs = np.array(self.config.AGENT_0.SOUND_STARTING_SAMPLING_IDXS, dtype=np.int32)
        is_same_scene = config.SCENE == self._current_scene
        if not is_same_scene:
            self._current_scene = config.SCENE
            logging.debug('Current scene: {} and sounds: {}'.format(self.current_scene_name, self._current_sound_names))

            if not self.config.USE_RENDERED_OBSERVATIONS:
                self._sim.close()
                del self._sim
                self.sim_config = self.create_sim_config(self._sensor_suite)
                self._sim = habitat_sim.Simulator(self.sim_config)
                self._update_agents_state()
            else:
                with open(self.current_scene_observation_file, 'rb') as fo:
                    self._frame_cache = pickle.load(fo)
            logging.info('Loaded scene {}'.format(self.current_scene_name))

            self.points, self.graph = load_points_data(self.meta_dir, self.config.AUDIO.GRAPH_FILE,
                                                       scene_dataset=self.config.SCENE_DATASET)

            self._position_to_index_mapping = dict()
            for node in self.graph.nodes():
                self._position_to_index_mapping[self.position_encoding(self.graph.nodes()[node]['point'])] = node

        # set agent positions
        self._receiver_position_index = self._position_to_index(self.config.AGENT_0.START_POSITION)
        self._source_position_indices = []
        for i in range(len(self._current_sound_names)):
            self._source_position_indices.append(self._position_to_index(self.config.AGENT_0.AUDIO_SOURCE_POSITIONS[i]))

        # the agent rotates about +Y starting from -Z counterclockwise,
        # so rotation angle 90 means the agent rotate about +Y 90 degrees
        self._rotation_angle = int(np.around(np.rad2deg(quat_to_angle_axis(quat_from_coeffs(
                             self.config.AGENT_0.START_ROTATION))[0]))) % 360
        if not self.config.USE_RENDERED_OBSERVATIONS:
            self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                 self.config.AGENT_0.START_ROTATION)
        else:
            self._sim.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                      quat_from_coeffs(self.config.AGENT_0.START_ROTATION))

        self._target_class = self.config.AGENT_0.TARGET_CLASS

        logging.debug("Sound sources at {}, agent at {}, orientation: {}".
                      format(self._source_position_indices, self._receiver_position_index, self.get_orientation()))

    @staticmethod
    def position_encoding(position):
        return '{:.2f}_{:.2f}_{:.2f}'.format(*position)

    def _position_to_index(self, position):
        if self.position_encoding(position) in self._position_to_index_mapping:
            return self._position_to_index_mapping[self.position_encoding(position)]
        else:
            raise ValueError("Position misalignment.")

    def _get_sim_observation(self):
        joint_index = (self._receiver_position_index, self._rotation_angle)
        if joint_index in self._frame_cache:
            return self._frame_cache[joint_index]
        else:
            sim_obs = self._sim.get_sensor_observations()
            self._frame_cache[joint_index] = sim_obs
            return sim_obs

    def reset(self):
        # TODO: remove later, for debugging
        # print("Sim reset called")
        logging.debug('Reset simulation')

        if not self.config.USE_RENDERED_OBSERVATIONS:
            sim_obs = self._sim.reset()
            if self._update_agents_state():
                sim_obs = self._get_sim_observation()
        else:
            sim_obs = self._get_sim_observation()
            self._sim.set_sensor_observations(sim_obs)

        self._is_episode_active = True
        self._prev_sim_obs = sim_obs
        # Encapsule data under Observations class
        observations = self._sensor_suite.get_observations(sim_obs)

        return observations

    def step(self, action, only_allowed=True):
        """
        All angle calculations in this function is w.r.t habitat coordinate frame, on X-Z plane
        where +Y is upward, -Z is forward and +X is rightward.
        Angle 0 corresponds to +X, angle 90 corresponds to +y and 290 corresponds to 270.

        :param action: action to be taken
        :param only_allowed: if true, then can't step anywhere except allowed locations
        :return:
        Dict of observations
        """
        assert self._is_episode_active, (
            "episode is not active, environment not RESET or "
            "STOP action called previously"
        )

        # PAUSE: 0, FORWARD: 1, LEFT: 2, RIGHT: 3
        if action == HabitatSimActions.MOVE_FORWARD:
            # the agent initially faces -Z by default
            for neighbor in self.graph[self._receiver_position_index]:
                p1 = self.graph.nodes[self._receiver_position_index]['point']
                p2 = self.graph.nodes[neighbor]['point']
                direction = int(np.around(np.rad2deg(np.arctan2(p2[2] - p1[2], p2[0] - p1[0])))) % 360
                if direction not in [0, 90, 180, 270]:
                    # diagonal connection
                    if int(abs(direction - self.get_orientation())) == 45:
                        self._receiver_position_index = neighbor
                        self._previous_step_collided = False
                        break
                elif direction == self.get_orientation():
                    self._receiver_position_index = neighbor
                    break
        elif action == HabitatSimActions.TURN_LEFT:
            # agent rotates counterclockwise, so turning left means increasing rotation angle by 90
            self._rotation_angle = (self._rotation_angle + 90) % 360
        elif action == HabitatSimActions.TURN_RIGHT:
            self._rotation_angle = (self._rotation_angle - 90) % 360
        elif action == HabitatSimActions.PAUSE:
            raise ValueError
        else:
            raise NotImplementedError(str(action) + " not in action space -- [PAUSE: 0, MOVE_FORWARD: 1, TURN_LEFT: 2,"
                                                    "TURN_RIGHT: 3]")

        if not self.config.USE_RENDERED_OBSERVATIONS:
            self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                 quat_from_angle_axis(np.deg2rad(self._rotation_angle), np.array([0, 1, 0])))
        else:
            self._sim.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                      quat_from_angle_axis(np.deg2rad(self._rotation_angle), np.array([0, 1, 0])))
        # log debugging info
        logging.debug('After taking action {}, s,r: {}, {}, orientation: {}, location: {}'.format(
            action, self._source_position_indices, self._receiver_position_index,
            self.get_orientation(), self.graph.nodes[self._receiver_position_index]['point']))

        sim_obs = self._get_sim_observation()
        if self.config.USE_RENDERED_OBSERVATIONS:
            self._sim.set_sensor_observations(sim_obs)
        self._prev_sim_obs = sim_obs
        observations = self._sensor_suite.get_observations(sim_obs)

        return observations
    
    def get_orientation(self):
        _base_orientation = 270
        return (_base_orientation - self._rotation_angle) % 360

    def write_info_to_obs(self, observations):
        observations["agent node and location"] = (self._receiver_position_index,
                                                   self.graph.nodes[self._receiver_position_index]["point"])
        observations["audio source nodes and locations"] = (self._source_position_indices,
                                                            self.config.AGENT_0.AUDIO_SOURCE_POSITIONS)
        observations["scene name"] = self.current_scene_name
        observations["orientation"] = self._rotation_angle

    @property
    def azimuth_angle(self):
        # this is the angle used to index the binaural audio files
        # in mesh coordinate systems, +Y forward, +X rightward, +Z upward
        # azimuth is calculated clockwise so +Y is 0 and +X is 90
        return -(self._rotation_angle + 0) % 360

    @property
    def target_class(self):
        return self._target_class

    def get_current_gt_bin_audio_components(self):
        r"""
        get binaural specs (magnitude and phase) for all sources at current step
        :return: binaural specs (magnitude and phase) for all sources at current step
        """
        assert len(self._gt_bin_audio_components) != 0, " call this method after calling get_current_mixed_audio_mag_spec"
        return np.concatenate(self._gt_bin_audio_components, axis=2)

    def get_current_gt_mono_audio_components(self):
        r"""
        get monaural specs (magnitude and phase) for all sources at current step
        :return: monaural specs (magnitude and phase) for all sources at current step
        """
        assert len(self._gt_mono_audio_components) != 0, "call this method after calling get_current_mixed_audio_mag_spec"
        return np.concatenate(self._gt_mono_audio_components, axis=2)

    def get_current_mixed_bin_audio_phase_spec(self):
        r"""
        get mixed audio spec phase at current step
        :return: mixed audio spec phase at current step
        """
        return self._mixed_bin_audio_phase

    def get_current_mixed_bin_audio_mag_spec(self):
        self._gt_bin_audio_components = []
        self._gt_mono_audio_components = []
        mixed_bin_audio_waveform = 0

        for indx, source_position_index in enumerate(self._source_position_indices):
            # this is to save memory, _mono_audios blows up with multi-process ddppo
            mono_audio_waveforms = {}
            if self._current_sound_names[indx] not in mono_audio_waveforms:
                _, mono_audio = wavfile.read(os.path.join(self.config.AUDIO.MONO_DIR,
                                                          self._current_sound_names[indx] + ".wav"))
                mono_audio_waveforms[self._current_sound_names[indx]] = mono_audio

            binaural_rir_dir_this_scene_n_azimuth =\
                os.path.join(self.config.AUDIO.RIR_DIR, self.current_scene_name, str(self.azimuth_angle))

            binaural_rir_file = os.path.join(binaural_rir_dir_this_scene_n_azimuth,
                                             str(self._receiver_position_index) + "_" +
                                             str(source_position_index) + ".wav")

            try:
                sr, binaural_rir = wavfile.read(binaural_rir_file)
                assert sr == self.config.AUDIO.RIR_SAMPLING_RATE, "RIR doesn't have sampling frequency of RIR_SAMPLING_RATE kHz"
            except ValueError:
                # logging.warning("{} file is not readable".format(binaural_rir_file))
                binaural_rir = np.zeros((self.config.AUDIO.RIR_SAMPLING_RATE, 2)).astype(np.float32)
            if len(binaural_rir) == 0:
                # logging.debug("Empty RIR file at {}".format(binaural_rir_file))
                binaural_rir = np.zeros((self.config.AUDIO.RIR_SAMPLING_RATE, 2)).astype(np.float32)

            sampling_starting_idx = self._mono_audio_sampling_starting_idxs[indx]
            self._mono_audio_sampling_starting_idxs[indx] =\
                (self._mono_audio_sampling_starting_idxs[indx] + self.config.AUDIO.RIR_SAMPLING_RATE) %\
                mono_audio_waveforms[self._current_sound_names[indx]].shape[0]

            mono_audio_waveform = mono_audio_waveforms[self._current_sound_names[indx]]
            mono_audio_waveform_len = mono_audio_waveform.shape[0]
            if sampling_starting_idx + self.config.AUDIO.RIR_SAMPLING_RATE > mono_audio_waveform_len:
                mono_audio_chunk_this_source =\
                    np.concatenate((mono_audio_waveform[sampling_starting_idx:],\
                                    mono_audio_waveform[:self.config.AUDIO.RIR_SAMPLING_RATE -\
                                                         (mono_audio_waveform_len - sampling_starting_idx)]))
            else:
                mono_audio_chunk_this_source = mono_audio_waveform[sampling_starting_idx:
                                                                   sampling_starting_idx + self.config.AUDIO.RIR_SAMPLING_RATE]

            binaural_convolved = []
            for channel in range(binaural_rir.shape[-1]):
                binaural_convolved.append(fftconvolve(mono_audio_chunk_this_source,
                                                      binaural_rir[:, channel],
                                                      mode="same"),
                                          )

            binaural_convolved = np.round(binaural_convolved).astype("int16").astype("float32")
            # scaling down to [-1, 1] range
            binaural_convolved *= (1 / 32768)

            mixed_bin_audio_waveform += binaural_convolved

            bin_audio_chunk_this_source_fft_windows_l = librosa.stft(binaural_convolved[0], hop_length=512, n_fft=1023)
            bin_audio_chunk_this_source_magnitude_l, bin_audio_chunk_this_source_phase_l =\
                librosa.magphase(bin_audio_chunk_this_source_fft_windows_l)
            bin_audio_chunk_this_source_phase_l = np.angle(bin_audio_chunk_this_source_phase_l)

            bin_audio_chunk_this_source_fft_windows_r = librosa.stft(binaural_convolved[1], hop_length=512, n_fft=1023)
            bin_audio_chunk_this_source_magnitude_r, bin_audio_chunk_this_source_phase_r =\
                librosa.magphase(bin_audio_chunk_this_source_fft_windows_r)
            bin_audio_chunk_this_source_phase_r = np.angle(bin_audio_chunk_this_source_phase_r)

            self._gt_bin_audio_components.append(np.stack([bin_audio_chunk_this_source_magnitude_l,
                                                           bin_audio_chunk_this_source_phase_l,
                                                           bin_audio_chunk_this_source_magnitude_r,
                                                           bin_audio_chunk_this_source_phase_r],
                                                          axis=-1).astype(np.float16))

            mono_audio_chunk_this_source = mono_audio_chunk_this_source.astype("float32") / 32768
            mono_audio_chunk_this_source_fft_windows =\
                librosa.stft(np.asfortranarray(mono_audio_chunk_this_source), hop_length=512, n_fft=1023)
            mono_audio_chunk_this_source_magnitude, mono_audio_chunk_this_source_phase =\
                librosa.magphase(mono_audio_chunk_this_source_fft_windows)

            if self.config.AUDIO.GT_MONO_MAG_NORM != 0.0:
                assert self.config.AUDIO.NORM_TYPE == "l2"
                if np.power(np.mean(np.power(mono_audio_chunk_this_source_magnitude, 2)), 0.5) != 0.:
                    mono_audio_chunk_this_source_magnitude =\
                        mono_audio_chunk_this_source_magnitude * self.config.AUDIO.GT_MONO_MAG_NORM\
                        / np.power(np.mean(np.power(mono_audio_chunk_this_source_magnitude, 2)), 0.5)
            mono_audio_chunk_this_source_phase = np.angle(mono_audio_chunk_this_source_phase)

            self._gt_mono_audio_components.append(
                np.stack([mono_audio_chunk_this_source_magnitude,
                          mono_audio_chunk_this_source_phase], axis=-1).astype("float16")
            )

        mixed_bin_audio_waveform /= len(self._source_position_indices)

        mixed_bin_audio_fft_windows_l = librosa.stft(np.asfortranarray(mixed_bin_audio_waveform[0]),
                                                     hop_length=512, n_fft=1023)
        mixed_bin_audio_magnitude_l, mixed_bin_audio_phase_l = librosa.magphase(mixed_bin_audio_fft_windows_l)
        mixed_bin_audio_phase_l = np.angle(mixed_bin_audio_phase_l)

        mixed_bin_audio_fft_windows_r = librosa.stft(np.asfortranarray(mixed_bin_audio_waveform[1]),
                                                     hop_length=512, n_fft=1023)
        mixed_bin_audio_magnitude_r, mixed_bin_audio_phase_r = librosa.magphase(mixed_bin_audio_fft_windows_r)
        mixed_bin_audio_phase_r = np.angle(mixed_bin_audio_phase_r)

        mixed_bin_audio_magnitude = np.stack([mixed_bin_audio_magnitude_l, mixed_bin_audio_magnitude_r], axis=-1).astype(np.float16)
        self._mixed_bin_audio_phase = np.stack([mixed_bin_audio_phase_l, mixed_bin_audio_phase_r], axis=-1).astype(np.float16)

        return np.log1p(mixed_bin_audio_magnitude)

    def geodesic_distance(self, position_a, position_b):
        index_a = self._position_to_index(position_a)
        index_b = self._position_to_index(position_b)
        assert index_a is not None and index_b is not None
        steps = nx.shortest_path_length(self.graph, index_a, index_b) * self.config.GRID_SIZE

        return steps

    def get_euclid_dist_to_target_audio_source(self):
        current_position = self.get_agent_state().position.tolist()
        distance_to_target = self.geodesic_distance(
            current_position, self.config.AGENT_0.AUDIO_SOURCE_POSITIONS[0]
        )
        return distance_to_target

