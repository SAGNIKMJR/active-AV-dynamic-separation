import os
import pickle
from tqdm import tqdm
import librosa
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

import torch
from torch.utils.data import Dataset

# 100 LibriSpeech classes, 1 combined music class from MIT MUSIC, 1 combined ESC-50 class (only used as background distractor)
CLASS_NAMES_TO_LABELS = {'7540': 41, '3274': 78, '5727': 82, '5157': 9, '3436': 67, '4731': 47, '4057': 1, '3851': 72,
                         '7932': 15, '597': 80, '119': 58, '345': 10, '460': 45, '868': 33, '909': 56, '125': 37,
                         '203': 50, '667': 32, '1578': 66, '2194': 44, '957': 30, '2137': 65, '2299': 8, '2319': 21,
                         '2388': 55, '2582': 96, '2989': 89, '3361': 42, '3703': 63, '4116': 26, '2816': 79, '3221': 52,
                         '5239': 98, '5390': 49, '6880': 61, '4853': 74, '5115': 48, '5133': 76, '5723': 100, '6139': 59,
                         '7247': 87, '7460': 12, '8619': 36, '8742': 23, '8772': 39, '6286': 40, '6509': 18, '7117': 68,
                         '8163': 5, '8194': 62, '122': 7, '439': 34, '500': 83, '637': 46, '1001': 2, '1121': 51,
                         '1160': 16, '2162': 38, '2514': 60, '3105': 70, '3448': 22, '4039': 6, '4441': 24, '5189': 85,
                         '5206': 93, '5656': 101, '5985': 84, '7067': 19, '7258': 73, '7828': 43, '8011': 88, '8028': 20,
                         '8474': 0, '8527': 97, '8824': 57, '126': 95, '166': 53, '248': 91, '1052': 25, '1460': 71,
                         '1859': 92, '1944': 77, '2758': 64, '2764': 29, '3083': 27, '3645': 90, '3728': 99, '3816': 4,
                         '3879': 54, '3994': 75, '4051': 11, '4640': 69, '6014': 86,  '6300': 13, '7078': 17,
                         '7445': 28, '7730': 31, '7802': 35, '7868': 94, '8410': 3, 'music': 14, 'esc': 81}
LABELS_TO_CLASS_NAMES = {}
for key, val in CLASS_NAMES_TO_LABELS.items():
    LABELS_TO_CLASS_NAMES[val] = key

# STFT params
HOP_LENGTH = 512
N_FFT = 1023


class PassiveDataset(Dataset):
    def __init__(self, split="train", scene_graphs=None, sim_cfg=None,):
        np.random.seed(42)
        torch.manual_seed(42)

        self.split = split
        self.audio_cfg = sim_cfg.AUDIO
        self.passive_dataset_version = self.audio_cfg.PASSIVE_DATASET_VERSION
        self.binaural_rir_dir = self.audio_cfg.RIR_DIR
        self.use_cache = False if split.split("_")[0] == "train" else True
        self.rir_sampling_rate = self.audio_cfg.RIR_SAMPLING_RATE
        self.num_sources = 2

        assert split in ["train", "val", "nonoverlapping_val"]

        # datapoints: locations to place sources and pose of agent
        if split == "nonoverlapping_val":
            self.sourceAgentLocn_datapoints_dir = os.path.join(self.audio_cfg.SOURCE_AGENT_LOCATION_DATAPOINTS_DIR,
                                                               self.passive_dataset_version, "val")
        else:
            self.sourceAgentLocn_datapoints_dir = os.path.join(self.audio_cfg.SOURCE_AGENT_LOCATION_DATAPOINTS_DIR,
                                                               self.passive_dataset_version, split)

        # datapoints: raw mono sounds playing at the sources
        if split in ["train", "val"]:
            self.audio_dir = self.audio_cfg.PASSIVE_TRAIN_AUDIO_DIR
        elif split in "nonoverlapping_val":
            self.audio_dir = self.audio_cfg.PASSIVE_NONOVERLAPPING_VAL_AUDIO_DIR
        assert os.path.exists(self.audio_dir)

        self.audio_files = []
        for _, __, self.audio_files in os.walk(self.audio_dir):
            break

        self.sound_to_idx = {}
        for idx, sound_file in enumerate(self.audio_files):
            self.sound_to_idx[sound_file] = idx

        self.audio_files_per_class = {}
        for audio_file in self.audio_files:
            assert audio_file.split("_")[0] in CLASS_NAMES_TO_LABELS
            if audio_file.split("_")[0] not in self.audio_files_per_class:
                self.audio_files_per_class[audio_file.split("_")[0]] = [audio_file]
            else:
                self.audio_files_per_class[audio_file.split("_")[0]].append(audio_file)

        self.file2audio_dict = dict()
        self.load_source_audio()

        self.complete_datapoint_files = list()
        self.target_classes = list()
        tqdm_over = scene_graphs
        for scene in tqdm(tqdm_over):
            with open(os.path.join(self.sourceAgentLocn_datapoints_dir,  scene + ".pkl"), "rb") as fi:
                sourceAgentLocn_datapoints = pickle.load(fi)[scene]

            if split.split("_")[0] == "train":
                sourceAgentLocn_datapoints = sourceAgentLocn_datapoints[:self.audio_cfg.NUM_PASSIVE_DATAPOINTS_PER_SCENE]
            elif (split.split("_")[0] == "val") or (split.split("_")[1] == "val"):
                sourceAgentLocn_datapoints = sourceAgentLocn_datapoints[:self.audio_cfg.NUM_PASSIVE_DATAPOINTS_PER_SCENE_EVAL]

            for datapoint in sourceAgentLocn_datapoints:
                audio_files, target_class = self.get_target_class_audio_files_for_sources()
                self.target_classes.append(target_class)

                receiver_locn = datapoint['r']
                receiver_azimuth = datapoint['azimuth']
                all_source_locns = datapoint['all_s']
                complete_datapoint = []
                for idx, source_locn in enumerate(all_source_locns):
                    binaural_rir_file = os.path.join(scene, str(receiver_azimuth), f"{receiver_locn}_{source_locn}.wav")
                    audio_file = audio_files[idx]
                    complete_datapoint.append((binaural_rir_file, audio_file))
                self.complete_datapoint_files.append(complete_datapoint)

        # this is used to cache mono stfts to prevent redundant computation (depending on the amount of audio data this
        # could be removed to save memory footprint)
        self._gt_mono_mag_cache = dict()

        if self.use_cache:
            self.complete_datapoints = [None] * len(self.complete_datapoint_files)
            for item in tqdm(range(len(self.complete_datapoints))):
                rirs_audio = self.complete_datapoint_files[item]
                mixed_audio, gt_bin_mag, gt_mono_mag, target_class =\
                    self.compute_audiospects(rirs_audio, target_class=self.target_classes[item])
                self.complete_datapoints[item] = (mixed_audio, gt_bin_mag, gt_mono_mag, target_class)

    def __len__(self):
        return len(self.complete_datapoint_files)

    def __getitem__(self, item):
        if not self.use_cache:
            audio_files, target_class = self.get_target_class_audio_files_for_sources()

            rirs_audio = self.complete_datapoint_files[item]
            # keep the same src locns, and agent locn and pose but resample the source audio
            rirs_audio_new = []
            for src_idx, (binaural_rir_file, audio_file) in enumerate(rirs_audio):
                rirs_audio_new.append((binaural_rir_file, audio_files[src_idx]))
            rirs_audio = rirs_audio_new

            mixed_audio,  gt_bin_mag, gt_mono_mag, target_class =\
                self.compute_audiospects(rirs_audio, target_class=target_class)

            mixed_audio = torch.from_numpy(mixed_audio)
            gt_bin_mag = torch.from_numpy(np.concatenate(gt_bin_mag, axis=2))
            gt_mono_mag = torch.from_numpy(np.concatenate(gt_mono_mag, axis=2))
            target_class = torch.from_numpy(target_class)
        else:
            mixed_audio = torch.from_numpy(self.complete_datapoints[item][0])
            gt_bin_mag = torch.from_numpy(np.concatenate(self.complete_datapoints[item][1], axis=2))
            gt_mono_mag = torch.from_numpy(np.concatenate(self.complete_datapoints[item][2], axis=2))
            target_class = torch.from_numpy(self.complete_datapoints[item][3])

        return mixed_audio, gt_bin_mag, gt_mono_mag, target_class

    def get_target_class_audio_files_for_sources(self):
        sampled_classes = (torch.randperm(len(CLASS_NAMES_TO_LABELS)).tolist())[:self.num_sources]
        target_class = sampled_classes[0]
        assert target_class < len(CLASS_NAMES_TO_LABELS)
        while target_class == 81:
            sampled_classes = (torch.randperm(len(CLASS_NAMES_TO_LABELS)).tolist())[:self.num_sources]
            target_class = sampled_classes[0]
            assert target_class < len(CLASS_NAMES_TO_LABELS)

        sampled_class_names = []
        audio_files = []
        for sampled_class in sampled_classes:
            sampled_class_names.append(LABELS_TO_CLASS_NAMES[sampled_class])
            audio_files_for_sampled_class = self.audio_files_per_class[LABELS_TO_CLASS_NAMES[sampled_class]]
            audio_files.append(audio_files_for_sampled_class[torch.randint(len(audio_files_for_sampled_class),
                                                                           size=(1,)).item()])

        return audio_files, target_class

    def load_source_audio(self):
        for audio_file in tqdm(self.audio_files):
            sr, audio_data = wavfile.read(os.path.join(self.audio_dir, audio_file))
            if sr != self.rir_sampling_rate:
                audio_data = resample(audio_data, self.rir_sampling_rate)
            self.file2audio_dict[audio_file] = audio_data

    def compute_audiospects(self, rirs_audio, target_class):
        gt_mono_mag = []
        gt_bin_mag = []
        mixed_binaural_wave = 0
        for idx, rir_audio in enumerate(rirs_audio):
            binaural_rir_file = os.path.join(self.binaural_rir_dir, rir_audio[0])
            mono_audio = self.file2audio_dict[rir_audio[1]]
            try:
                sr, binaural_rir = wavfile.read(binaural_rir_file)
                assert sr == self.rir_sampling_rate, "RIR doesn't have sampling frequency of rir_sampling_rate kHz"
            except ValueError:
                binaural_rir = np.zeros((self.rir_sampling_rate, 2)).astype("float32")
            if len(binaural_rir) == 0:
                binaural_rir = np.zeros((self.rir_sampling_rate, 2)).astype("float32")

            binaural_convolved = []
            for channel in range(binaural_rir.shape[-1]):
                binaural_convolved.append(fftconvolve(mono_audio, binaural_rir[:, channel], mode="same"))

            binaural_convolved = np.array(binaural_convolved)
            # this makes sure that the audio is in the range [-32768, 32767]
            binaural_convolved = np.round(binaural_convolved).astype("int16").astype("float32")
            binaural_convolved *= (1 / 32768)

            # compute target specs
            if idx == 0:
                # compute gt bin. magnitude
                fft_windows_l = librosa.stft(np.asfortranarray(binaural_convolved[0]), hop_length=HOP_LENGTH,
                                             n_fft=N_FFT)
                magnitude_l, _ = librosa.magphase(fft_windows_l)

                fft_windows_r = librosa.stft(np.asfortranarray(binaural_convolved[1]), hop_length=HOP_LENGTH,
                                             n_fft=N_FFT)
                magnitude_r, _ = librosa.magphase(fft_windows_r)

                gt_bin_mag.append(np.stack([magnitude_l, magnitude_r], axis=-1).astype("float32"))

                # compute gt mono magnitude
                if rir_audio[1] not in self._gt_mono_mag_cache:
                    # this makes sure that the audio is in the range [-32768, 32767]
                    mono_audio = mono_audio.astype("float32") / 32768

                    fft_windows = librosa.stft(np.asfortranarray(mono_audio), hop_length=HOP_LENGTH,
                                               n_fft=N_FFT)
                    magnitude, _ = librosa.magphase(fft_windows)
                    if np.power(np.mean(np.power(magnitude, 2)), 0.5) != 0.:
                        magnitude = magnitude * self.audio_cfg.GT_MONO_MAG_NORM /\
                                    np.power(np.mean(np.power(magnitude, 2)), 0.5)

                    self._gt_mono_mag_cache[rir_audio[1]] = magnitude

                mono_magnitude = self._gt_mono_mag_cache[rir_audio[1]]
                gt_mono_mag.append(np.expand_dims(mono_magnitude, axis=2).astype("float32"))

            mixed_binaural_wave += binaural_convolved

        mixed_binaural_wave /= len(rirs_audio)

        fft_windows_l = librosa.stft(np.asfortranarray(mixed_binaural_wave[0]), hop_length=HOP_LENGTH, n_fft=N_FFT)
        magnitude_l, _ = librosa.magphase(fft_windows_l)

        fft_windows_r = librosa.stft(np.asfortranarray(mixed_binaural_wave[1]), hop_length=HOP_LENGTH, n_fft=N_FFT)
        magnitude_r, _ = librosa.magphase(fft_windows_r)

        mixed_mag = np.stack([magnitude_l, magnitude_r], axis=-1).astype("float32")

        return np.log1p(mixed_mag), gt_bin_mag, gt_mono_mag, np.array([target_class]).astype("int64")
