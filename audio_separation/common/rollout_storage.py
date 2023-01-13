from collections import defaultdict

import torch


class RolloutStoragePol:
    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
    ):
        r"""
        Class for storing rollout information for RL policy trainer.
        :param num_steps: number of steps before PPO update
        :param num_envs: number of training environments
        :param observation_space: simulator observation space
        :param recurrent_hidden_state_size: hidden state size for policy GRU
        :param num_recurrent_layers: number of hidden layers in policy GRU
        """
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape
            )

        self.recurrent_hidden_states_pol = torch.zeros(
            num_steps + 1,
            num_recurrent_layers,
            num_envs,
            recurrent_hidden_state_size,
        )

        assert "gt_mono_comps" in observation_space.spaces

        self.pred_monoFromMem =\
            torch.zeros(
                num_steps,
                num_envs,
                observation_space.spaces["gt_mono_comps"].shape[0],
                observation_space.spaces["gt_mono_comps"].shape[1],
                1
            )

        self.pred_binSepMasks =\
            torch.zeros(
                num_steps,
                num_envs,
                observation_space.spaces["gt_mono_comps"].shape[0],
                observation_space.spaces["gt_mono_comps"].shape[1],
                2,
            )

        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)

        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)

        self.actions = torch.zeros(num_steps, num_envs, 1)
        self.actions = self.actions.long()

        self.masks = torch.ones(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)
        self.recurrent_hidden_states_pol = self.recurrent_hidden_states_pol.to(device)

        self.pred_binSepMasks = self.pred_binSepMasks.to(device)
        self.pred_monoFromMem = self.pred_monoFromMem.to(device)

        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)

        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)

        self.masks = self.masks.to(device)

    def insert(
        self,
        observations,
        recurrent_hidden_states_pol,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        pred_binSepMasks=None,
        pred_monoFromMem=None,
    ):
        r"""
        Method for inserting useful scalars and tensors from the current step into the storage
        :param observations: current observations from the simulator
        :param recurrent_hidden_states_pol: current policy GRU hidden states
        :param actions: current actions
        :param action_log_probs: current action log probabilities
        :param values: current values
        :param rewards: current rewards
        :param masks: current not-done masks
        :param pred_binSepMasks: current binaural separation masks
        :param pred_monoFromMem: current monaural predictions from transformer memory
        """
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(
                observations[sensor]
            )
        self.recurrent_hidden_states_pol[self.step + 1].copy_(
            recurrent_hidden_states_pol
        )

        self.pred_binSepMasks[self.step].copy_(pred_binSepMasks)
        self.pred_monoFromMem[self.step].copy_(pred_monoFromMem)

        self.rewards[self.step].copy_(rewards)
        self.value_preds[self.step].copy_(value_preds)

        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)

        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(self.observations[sensor][-1])
        self.recurrent_hidden_states_pol[0].copy_(self.recurrent_hidden_states_pol[-1])

        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        r"""
        compute returns with or without GAE
        """
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def recurrent_generator(self, advantages, num_mini_batch):
        r"""
        Recurrent batch generator for PPO update
        :param advantages: advantage values needed for PPO update
        :param num_mini_batch: number of mini batches to split all processes across all environments into
        :return: current batch for doing forward and backward passes for PPO update
        """
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)
            recurrent_hidden_states_pol_batch = []

            pred_binSepMasks_batch = []
            pred_monoFromMem_batch = []

            value_preds_batch = []
            return_batch = []
            adv_targ = []

            actions_batch = []
            old_action_log_probs_batch = []

            masks_batch = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][:-1, ind]
                    )
                recurrent_hidden_states_pol_batch.append(
                    self.recurrent_hidden_states_pol[0, :, ind]
                )

                pred_binSepMasks_batch.append(self.pred_binSepMasks[:, ind])
                pred_monoFromMem_batch.append(self.pred_monoFromMem[:, ind])

                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                adv_targ.append(advantages[:, ind])

                actions_batch.append(self.actions[:, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])

                masks_batch.append(self.masks[:-1, ind])

            T, N = self.num_steps, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )
            # States is just a (num_recurrent_layers, N, -1) tensor
            recurrent_hidden_states_pol_batch = torch.stack(
                recurrent_hidden_states_pol_batch, 1
            )

            pred_binSepMasks_batch = torch.stack(pred_binSepMasks_batch, 1)
            pred_monoFromMem_batch = torch.stack(pred_monoFromMem_batch, 1)

            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            actions_batch = torch.stack(actions_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1
            )

            masks_batch = torch.stack(masks_batch, 1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )

            pred_binSepMasks_batch = self._flatten_helper(T, N, pred_binSepMasks_batch)
            pred_monoFromMem_batch = self._flatten_helper(T, N, pred_monoFromMem_batch)

            value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
            return_batch = self._flatten_helper(T, N, return_batch)
            adv_targ = self._flatten_helper(T, N, adv_targ)

            actions_batch = self._flatten_helper(T, N, actions_batch)
            old_action_log_probs_batch = self._flatten_helper(
                T, N, old_action_log_probs_batch
            )

            masks_batch = self._flatten_helper(T, N, masks_batch)

            yield (
                observations_batch,
                recurrent_hidden_states_pol_batch,
                pred_binSepMasks_batch,
                pred_monoFromMem_batch,
                value_preds_batch,
                return_batch,
                adv_targ,
                actions_batch,
                masks_batch,
                old_action_log_probs_batch,
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.

        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])


class RolloutStorageSep:
    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        external_memory_size=39,
        external_memory_capacity=19,
        external_memory_dim=16384,  # 512 x 32 = 16384
        poseEnc_cfg=None,
    ):
        r"""
        Class for storing rollout information for audio separator trainer.
        :param num_steps: number of steps before audio separator update
        :param num_envs: number of training environments
        :param observation_space: simulator observation space
        :param external_memory_size: total size of external memory (for storing past monaural predictions) to account for non-0 PPO rollout size
        :param external_memory_capacity: number of entries (past monaural predictions) in the external memory
        :param external_memory_dim: dimensionality of past monaural predictions stored in external memory
        :param poseEnc_cfg: pose encoding config
        """
        self.poseEnc_cfg = poseEnc_cfg

        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape
            )

        self.em_size = external_memory_size
        self.em_capacity = external_memory_capacity

        self.em_dim = external_memory_dim
        self.em_dim += poseEnc_cfg.num_pose_attrs

        self.em_masks = torch.zeros(num_steps + 1, num_envs, self.em_size)
        """pred_all_old_steps=True leads to bug: stores decoder features in place of encoder features in external memory"""
        self.em = ExternalMemory(
            num_envs,
            self.em_size,
            self.em_capacity,
            self.em_dim,
            num_copies=num_steps + 1,
            gt_mono_mag_shape=(observation_space.spaces["gt_mono_comps"].shape[0],
                               observation_space.spaces["gt_mono_comps"].shape[1]),
        )

        self.masks = torch.ones(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.em_masks = self.em_masks.to(device)
        self.em.to(device)

        self.masks = self.masks.to(device)

    def insert(
        self,
        observations,
        masks,
        pred_mono=None,
        skip_feats=None,
    ):
        r"""
        Method for inserting useful scalars and tensors from the current step into the storage
        :param observations: current observations from the simulator
        :param masks: current not-done masks
        :param pred_mono: current monaural predictions from passive separator
        :param skip_feats: skip connection features for current monaural
        """
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(
                observations[sensor]
            )
        self.masks[self.step + 1].copy_(masks)

        self.em.insert(pred_mono,
                       skip_feats,
                       self.observations["gt_mono_comps"][self.step][:, :, :, 0].unsqueeze(-1),
                       masks,)

        self.em_masks[self.step + 1].copy_(self.em.masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(self.observations[sensor][-1])

        self.em_masks[0].copy_(self.em_masks[-1])

        self.masks[0].copy_(self.masks[-1])

    def recurrent_generator(self, num_mini_batch):
        r"""
        Recurrent batch generator for audio separator training
        :param num_mini_batch: number of mini batches to split all processes across all environments into
        :return: current batch for doing forward and backward passes for audio separator training
        """
        num_processes = self.masks.size(1)
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            em_mono_batch = []
            em_masks_batch = []
            em_skipFeats_batch = []

            gtMonoMag_pastStepsRefine_batch = []

            masks_batch = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][:-1, ind]
                    )

                em_mono_batch.append(self.em.memory[:, :-1, ind])
                em_masks_batch.append(self.em_masks[:-1, ind])
                em_skipFeats_batch.append(self.em.memory_skipFeats[:, :-1, ind])

                gtMonoMag_pastStepsRefine_batch.append(self.extMem_gtMonoMag[:, :, ind]\
                                                       .repeat(1, self.num_steps, 1, 1, 1))

                masks_batch.append(self.masks[:-1, ind])

            T, N = self.num_steps, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )

            em_mono_batch = torch.stack(em_mono_batch, 2)
            em_masks_batch = torch.stack(em_masks_batch, 1)
            em_skipFeats_batch = torch.stack(em_skipFeats_batch, 2)

            gtMonoMag_pastStepsRefine_batch =\
                torch.stack(gtMonoMag_pastStepsRefine_batch, 2).permute(1, 2, 0, 3, 4, 5,).contiguous()

            masks_batch = torch.stack(masks_batch, 1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )

            em_mono_batch = em_mono_batch.view(-1, T * N, self.em_dim)
            em_masks_batch = self._flatten_helper(T, N, em_masks_batch)
            em_skipFeats_batch = em_skipFeats_batch.view(-1, T * N, 16, 16, 16)

            gtMonoMag_pastStepsRefine_batch = self._flatten_helper(T, N, gtMonoMag_pastStepsRefine_batch)

            masks_batch = self._flatten_helper(T, N, masks_batch)

            yield (
                observations_batch,
                masks_batch,
                em_mono_batch,
                em_skipFeats_batch,
                em_masks_batch,
                gtMonoMag_pastStepsRefine_batch,
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.

        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])

    @property
    def extMem_mono(self):
        return self.em.memory

    @property
    def extMem_gtMonoMag(self):
        return self.em.memory_gtMonoMag

    @property
    def extMem_skipFeats(self):
        return self.em.memory_skipFeats

    @property
    def extMem_masks(self):
        return self.em_masks

    @property
    def extMem_idx(self):
        return self.em.idx


class ExternalMemory:
    def __init__(self,
                 num_envs,
                 total_size,
                 capacity,
                 dim,
                 num_copies=1,
                 gt_mono_mag_shape=(512, 32),):
        r"""An external memory that keeps track of observations over time.
        Inputs:
            num_envs - number of parallel environments
            capacity - total capacity of the memory per episode
            total_size - capacity + additional buffer size for rollout updates
            dim - size of observations
            num_copies - number of copies of the data to maintain for efficient training
            gt_mono_mag_shape - shape of mono magnitude spectrogram
        """
        self.num_envs = num_envs
        self.total_size = total_size
        self.capacity = capacity
        self.dim = dim
        self.num_copies = num_copies
        self.gt_mono_mag_shape = gt_mono_mag_shape

        self.masks = torch.zeros(num_envs, self.total_size)
        self.memory = torch.zeros(self.total_size, num_copies, num_envs, self.dim)
        self.memory_gtMonoMag = torch.zeros(self.total_size, 
                                            1, 
                                            num_envs,
                                            gt_mono_mag_shape[0],
                                            gt_mono_mag_shape[1],
                                            1)
        self.memory_skipFeats =\
            torch.zeros(self.total_size, 
                        num_copies, 
                        num_envs,
                        16, 
                        16, 
                        16)

        self.idx = 0

    def insert(self, 
               em_mono, 
               skip_feats,
               gt_mono_mag, 
               masks,):
        r"""
        Method for inserting useful scalars and tensors from the current step into the external memory
        :param em_mono: monaural from passive separator at current step
        :param skip_feats: skip features for current monaural
        :param gt_mono_mag: ground-truth monaural magnitude spectrogram at current step
        :param masks: not done masks at current step
        :return: None
        """

        self.memory[self.idx].copy_(em_mono.unsqueeze(0))
        self.memory_gtMonoMag[self.idx].copy_(gt_mono_mag.unsqueeze(0))
        self.memory_skipFeats[self.idx].copy_(skip_feats.unsqueeze(0))

        # Account for overflow capacity
        capacity_overflow_flag = self.masks.sum(1) == self.capacity
        assert (not torch.any(self.masks.sum(1) > self.capacity))
        self.masks[capacity_overflow_flag, self.idx - self.capacity] = 0.
        self.masks[:, self.idx] = 1.0
        # Mask out the entire memory for the next observation if episode done
        self.masks *= masks
        self.idx = (self.idx + 1) % self.total_size

    def to(self, device):
        self.masks = self.masks.to(device)
        self.memory = self.memory.to(device)
        self.memory_gtMonoMag = self.memory_gtMonoMag.to(device)
        self.memory_skipFeats = self.memory_skipFeats.to(device)

    def reset(self, device):
        self.masks = torch.zeros(self.num_envs, self.total_size).to(device)
        self.memory = torch.zeros(self.total_size, 
                                  self.num_copies, 
                                  self.num_envs, 
                                  self.dim).to(device)
        self.memory_gtMonoMag = torch.zeros(self.total_size, 
                                            1, 
                                            self.num_envs,
                                            self.gt_mono_mag_shape[0], 
                                            self.gt_mono_mag_shape[1],
                                            1).to(device)
        self.memory_skipFeats =\
            torch.zeros(self.total_size, 
                        self.num_copies, 
                        self.num_envs,
                        16,
                        16,
                        16).to(device)
        self.idx = 0
