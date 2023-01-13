import contextlib
import os
import time
import logging
from collections import deque
from typing import Dict
import json
import random
import pickle
import gzip

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from numpy.linalg import norm
from torch import distributed as distrib

from habitat import Config, logger
from audio_separation.common.base_trainer import BaseRLTrainer
from audio_separation.common.baseline_registry import baseline_registry
from audio_separation.common.env_utils import construct_envs, override_rewards
from audio_separation.common.environments import get_env_class
from audio_separation.common.rollout_storage import RolloutStoragePol, RolloutStorageSep, ExternalMemory
from audio_separation.common.tensorboard_utils import TensorboardWriter
from audio_separation.rl.ppo.ddppo_utils import (
    add_signal_handlers,
    init_distrib_slurm,
    load_interrupted_state,
)
from audio_separation.common.utils import (
    batch_obs,
    linear_decay,
)
from audio_separation.common.eval_metrics import STFT_L2_distance, compute_waveform_quality
from audio_separation.rl.ppo.policy import AAViDSSPolicy
from audio_separation.rl.ppo.ppo import PPO, DDPPO


@baseline_registry.register_trainer(name="ppo")
class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO and DDPPO algorithm
    PPO paper: https://arxiv.org/abs/1707.06347.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        if self.config.RL.PPO.use_ddppo:
            interrupted_state = load_interrupted_state()
            if interrupted_state is not None:
                self.config = interrupted_state["config"]

    def _setup_actor_critic_agent(self, is_eval=False) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            is_eval: if in eval_mode

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        ppo_cfg = self.config.RL.PPO

        assert self.config.EXTRA_DEPTH or self.config.EXTRA_RGB, "set at least one of EXTRA_RGB and EXTRA_DEPTH to true"

        if not is_eval:
            assert ppo_cfg.pretrained_passive_separators_ckpt != "", "set to path of pretrained passive separator checkpoint"

        self.actor_critic = AAViDSSPolicy(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            extra_rgb=self.config.EXTRA_RGB,
            extra_depth=self.config.EXTRA_DEPTH,
            ppo_cfg=ppo_cfg,
        )

        self.actor_critic.to(self.device)

        if ppo_cfg.use_ddppo:
            self.agent = DDPPO(
                actor_critic=self.actor_critic,
                ppo_cfg=ppo_cfg,
            )
        else:
            self.agent = PPO(
                actor_critic=self.actor_critic,
                ppo_cfg=ppo_cfg,
            )

        self.actor_critic.to(self.device)
        self.actor_critic.train()

    def save_checkpoint(self, file_name: str,) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def _collect_rollout_step(
            self, rollouts_pol, rollouts_sep, current_episode_reward, current_episode_step, current_episode_dist_probs,
            current_episode_bin_losses, current_episode_mono_losses, current_episode_monoFromMem_losses, episode_rewards,
            episode_counts, episode_steps, episode_dist_probs, episode_bin_losses_allSteps, episode_mono_losses_lastStep,
            episode_mono_losses_allSteps, episode_monoFromMem_losses_lastStep, episode_monoFromMem_losses_allSteps,
    ):
        r"""
        collects rollouts for training separator in supervised fashion and the policy with PPO
        :param rollouts_pol: rollout storage for policy
        :param rollouts_sep: rollout storage for separator
        :param current_episode_reward: reward for the current epispde
        :param current_episode_step: number of steps for the current episode
        :param current_episode_dist_probs: policy distribution for all actions for current episode
        :param current_episode_bin_losses: binaural losses for passive separator for current episode
        :param current_episode_mono_losses: monaural losses for passive separator for current episode
        :param current_episode_monoFromMem_losses: monaural losses on audio memory predictions for current episode
        :param episode_rewards: rewards for all episodes
        :param episode_counts: number of all episodes
        :param episode_steps: number of episode steps for all episodes
        :param episode_dist_probs: policy distribution for all actions for all episodes
        :param episode_bin_losses_allSteps: binaural losses over all steps for passive separator for all episodes
        :param episode_mono_losses_lastStep: monaural losses at last step for passive separator for all episodes
        :param episode_mono_losses_allSteps: monaural losses over all steps for passive separator for all episodes
        :param episode_monoFromMem_losses_lastStep: monaural losses on memory predictions at last step for all episodes
        :param episode_monoFromMem_losses_allSteps: monaural losses on memory predictions over all step for all episodes
        :return: 1. pth_time: time needed for pytorch forward pass
                 2. env_time: time needed for environment simulation with Habitat
                 3. self.envs.num_envs: number of active environments in the simulator
        """
        ppo_cfg = self.config.RL.PPO

        sepExtMem_cfg = ppo_cfg.TRANSFORMER_MEMORY
        num_past_steps_refinement = sepExtMem_cfg.num_past_steps_refinement

        pth_time = 0.0
        env_time = 0.0

        t_pred_current_step = time.time()
        # get binaural and mono predictions, and sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts_pol.step] for k, v in rollouts_pol.observations.items()
            }

            # passive-separate mono given target class
            pred_binSepMasks =\
                self.actor_critic.get_binSepMasks(
                    step_observation,
                )
            pred_mono =\
                self.actor_critic.convert_bin2mono(pred_binSepMasks.detach(),
                                                   mixed_audio=step_observation["mixed_bin_audio_mag"],
                                                   )

            sepExtMem_mono = rollouts_sep.extMem_mono[:, rollouts_pol.step].contiguous()
            sepExtMem_masks = rollouts_sep.extMem_masks[rollouts_pol.step]
            sepExtMem_skipFeats =\
                rollouts_sep.extMem_skipFeats[:, rollouts_pol.step].contiguous()

            pred_monoFromMem, pred_mono_toCache, pred_monoFromMem_aftrAtt_feats, skip_feats =\
                self.actor_critic.get_monoFromMem(pred_mono=pred_mono,
                                                  sepExtMem_mono=sepExtMem_mono,
                                                  sepExtMem_masks=sepExtMem_masks,
                                                  pose=step_observation["pose"],
                                                  sepExtMem_skipFeats=sepExtMem_skipFeats,
                                                  )

            sepExtMem_masks_wCurrStep =\
                torch.cat([sepExtMem_masks, torch.ones([sepExtMem_masks.shape[0], 1],
                                                       device=sepExtMem_masks.device)],
                          dim=1)

            # find out indexes to set num_past_steps_refinement steps to True
            bs, M = sepExtMem_masks_wCurrStep.size()
            newest_sepExtMem_masks_wCurrStep = sepExtMem_masks_wCurrStep[:, -1:].clone()
            old_sepExtMem_masks_wCurrStep = sepExtMem_masks_wCurrStep[:, :-1].clone()
            old_activeIdxs_sepExtMem_masks_wCurrStep =\
                torch.where(old_sepExtMem_masks_wCurrStep == 1.0)
            old_sepExtMem_masks_wCurrStep[:, :] = 0.0
            if old_activeIdxs_sepExtMem_masks_wCurrStep[0].size()[0] != 0:
                old_activeColsFirstProcess_sepExtMem_masks_wCurrStep =\
                    old_activeIdxs_sepExtMem_masks_wCurrStep[1].view(bs, -1)[0]

                # hacky and slow way to find out which idxs to choose for num_past_steps_refinement
                active_old_col = -1
                wrap_around_in_mem = False
                for active_col in old_activeColsFirstProcess_sepExtMem_masks_wCurrStep:
                    if active_old_col != -1:
                        if active_old_col != (active_col.item() - 1):
                            wrap_around_in_mem = True
                            break
                    active_old_col = active_col.item()

                if wrap_around_in_mem:
                    last_active_col_left = active_old_col
                    first_active_col_left = 0
                    last_active_col_right = old_sepExtMem_masks_wCurrStep.size(1) - 1
                    assert old_sepExtMem_masks_wCurrStep.size(1) - 1\
                           == old_activeColsFirstProcess_sepExtMem_masks_wCurrStep[-1].item()
                    if last_active_col_left - first_active_col_left + 1 >= num_past_steps_refinement:
                        lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep =\
                            list(range(last_active_col_left - num_past_steps_refinement + 1, last_active_col_left + 1))
                        assert len(lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep) == num_past_steps_refinement
                        old_activeCols_sepExtMem_masks_wCurrStep =\
                            torch.LongTensor(lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep).unsqueeze(0).repeat(bs, 1).contiguous().view(-1)

                        lst_old_activeRowsOneProcess_sepExtMem_masks_wCurrStep = list(range(bs))
                        old_activeRows_sepExtMem_masks_wCurrStep =\
                            torch.LongTensor(lst_old_activeRowsOneProcess_sepExtMem_masks_wCurrStep).unsqueeze(1)\
                                .repeat(1, num_past_steps_refinement).contiguous().view(-1)
                    else:
                        if old_activeColsFirstProcess_sepExtMem_masks_wCurrStep.size(0) >= num_past_steps_refinement:
                            remaining_numPastStepsRefinement_toBeSampledFromRight =\
                                num_past_steps_refinement - (last_active_col_left - first_active_col_left + 1)
                            lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromLeft\
                                = list(range(0, last_active_col_left + 1))
                            lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromRight =\
                                list(range(last_active_col_right - remaining_numPastStepsRefinement_toBeSampledFromRight + 1,
                                           last_active_col_right + 1))
                            lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep\
                                = lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromRight +\
                                  lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromLeft
                            assert len(lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep) == num_past_steps_refinement
                            old_activeCols_sepExtMem_masks_wCurrStep =\
                                torch.LongTensor(lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep).unsqueeze(0).repeat(bs, 1).contiguous().view(-1)

                            lst_old_activeRowsOneProcess_sepExtMem_masks_wCurrStep = list(range(bs))
                            old_activeRows_sepExtMem_masks_wCurrStep =\
                                torch.LongTensor(lst_old_activeRowsOneProcess_sepExtMem_masks_wCurrStep).unsqueeze(1)\
                                    .repeat(1, num_past_steps_refinement).contiguous().view(-1)
                        else:
                            remaining_numPastStepsRefinement_toBeSampledFromRight =\
                                old_activeColsFirstProcess_sepExtMem_masks_wCurrStep.size(0) -\
                                (last_active_col_left - first_active_col_left + 1)
                            lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromLeft\
                                = list(range(0, last_active_col_left + 1))
                            lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromRight =\
                                list(range(last_active_col_right - remaining_numPastStepsRefinement_toBeSampledFromRight + 1,
                                           last_active_col_right + 1))
                            lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep\
                                = lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromRight +\
                                  lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromLeft
                            old_activeCols_sepExtMem_masks_wCurrStep =\
                                torch.LongTensor(lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep).unsqueeze(0)\
                                    .repeat(bs, 1).contiguous().view(-1)

                            lst_old_activeRowsOneProcess_sepExtMem_masks_wCurrStep = list(range(bs))
                            old_activeRows_sepExtMem_masks_wCurrStep =\
                                torch.LongTensor(lst_old_activeRowsOneProcess_sepExtMem_masks_wCurrStep).unsqueeze(1)\
                                    .repeat(1, old_activeColsFirstProcess_sepExtMem_masks_wCurrStep.size(0)).contiguous().view(-1)
                else:
                    old_activeRows_sepExtMem_masks_wCurrStep =\
                        old_activeIdxs_sepExtMem_masks_wCurrStep[0].view(bs, -1)[:, -num_past_steps_refinement:].contiguous().view(-1)
                    old_activeCols_sepExtMem_masks_wCurrStep =\
                        old_activeIdxs_sepExtMem_masks_wCurrStep[1].view(bs, -1)[:, -num_past_steps_refinement:].contiguous().view(-1)

                old_sepExtMem_masks_wCurrStep[old_activeRows_sepExtMem_masks_wCurrStep,\
                                              old_activeCols_sepExtMem_masks_wCurrStep] = 1.0
                old_sepExtMem_masks_wCurrStep = old_sepExtMem_masks_wCurrStep.contiguous().view(bs, -1)

            sepExtMem_masks_wCurrStep = torch.cat((old_sepExtMem_masks_wCurrStep,
                                                   newest_sepExtMem_masks_wCurrStep), dim=1)

            valid_prediction_idxs = torch.where(sepExtMem_masks_wCurrStep == 1.0)

            # needed for mem_size == cfg_mem_size + ppo_cfg_num_steps... need to rotate the cols for computing
            # reward if there is a wraparound the memory to avoid bug
            valid_prediction_cols = valid_prediction_idxs[1]
            bs, M = sepExtMem_masks_wCurrStep.size()
            valid_prediction_cols = valid_prediction_cols.contiguous().view(bs, -1)

            active_old_col = -1
            active_old_col_idx = 0
            wrap_around_in_mem = False

            for active_col in valid_prediction_cols[0][:-1]:
                if active_old_col != -1:
                    if active_old_col != (active_col.item() - 1):
                        wrap_around_in_mem = True
                        break
                active_old_col = active_col.item()
                active_old_col_idx += 1

            if wrap_around_in_mem:
                valid_prediction_cols_toComputeLosses =\
                    torch.cat((valid_prediction_cols[:, :-1][:, active_old_col_idx:],
                               valid_prediction_cols[:, :-1][:, :active_old_col_idx],
                               valid_prediction_cols[:, -1:]), dim=1)
            else:
                valid_prediction_cols_toComputeLosses = valid_prediction_cols.clone()

            valid_prediction_idxs_toComputeLosses = (valid_prediction_idxs[0].clone(),
                                                     valid_prediction_cols_toComputeLosses.view(-1).contiguous())

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states_pol,
                distribution_probs,
            ) = self.actor_critic.act(
                step_observation,
                rollouts_pol.recurrent_hidden_states_pol[rollouts_pol.step],
                rollouts_pol.masks[rollouts_pol.step],
                pred_monoFromMem[-1],
                pred_binSepMasks,
            )

        pth_time += time.time() - t_pred_current_step

        t_step_env = time.time()
        outputs = self.envs.step([a[0].item() for a in actions])
        env_time += time.time() - t_step_env

        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        batch = batch_obs(observations, self.device)
        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float
        )

        rollouts_sep.insert(
            batch,
            masks.to(self.device),
            pred_mono=pred_mono_toCache,
            skip_feats=skip_feats,
        )

        t_pred_next_step = time.time()
        with torch.no_grad():
            next_pred_binSepMasks =\
                self.actor_critic.get_binSepMasks(batch)
            next_pred_mono =\
                self.actor_critic.convert_bin2mono(next_pred_binSepMasks.detach(),
                                                   mixed_audio=batch["mixed_bin_audio_mag"],
                                                   )

            next_sepExtMem_mono = rollouts_sep.extMem_mono[:, rollouts_pol.step + 1].contiguous()
            next_sepExtMem_masks = rollouts_sep.extMem_masks[rollouts_pol.step + 1]
            next_sepExtMem_skipFeats = rollouts_sep.extMem_skipFeats[:, rollouts_pol.step + 1].contiguous()

            next_pred_monoFromMem, _, _, _ =\
                self.actor_critic.get_monoFromMem(pred_mono=next_pred_mono,
                                                  sepExtMem_mono=next_sepExtMem_mono,
                                                  sepExtMem_masks=next_sepExtMem_masks,
                                                  pose=batch["pose"],
                                                  sepExtMem_skipFeats=next_sepExtMem_skipFeats,
                                                  )

            next_gt_mono_mag =\
                batch["gt_mono_comps"][:, :, :, 0::2].clone()[:, :, :, 0].unsqueeze(-1)
        pth_time += time.time() - t_pred_next_step

        t_compute_lossesNrewards = time.time()
        # this works because all processes have equal number of steps
        # current step tensor manipulations for reward"""
        gt_mono_mag_toComputeLosses =\
            rollouts_pol.observations["gt_mono_comps"][max(0,
                                                           rollouts_pol.step + 1 - (num_past_steps_refinement + 1))
                                                       :rollouts_pol.step + 1][..., 0::2]\
                .clone()[..., 0].unsqueeze(-1).permute(1, 0, 2, 3, 4)
        gt_mono_mag_toComputeLosses =\
            gt_mono_mag_toComputeLosses.contiguous().view(gt_mono_mag_toComputeLosses.size(0) *
                                                          gt_mono_mag_toComputeLosses.size(1),
                                                          *gt_mono_mag_toComputeLosses.size()[2:])

        gt_mono_phase_toComputeLosses =\
            rollouts_pol.observations["gt_mono_comps"][max(0,
                                                           rollouts_pol.step + 1 - (num_past_steps_refinement + 1))
                                                       :rollouts_pol.step + 1][..., 1::2]\
                .clone()[..., 0].unsqueeze(-1).permute(1, 0, 2, 3, 4)
        gt_mono_phase_toComputeLosses =\
            gt_mono_phase_toComputeLosses.contiguous().view(gt_mono_phase_toComputeLosses.size(0) *
                                                            gt_mono_phase_toComputeLosses.size(1),
                                                            *gt_mono_phase_toComputeLosses.size()[2:])

        # pred_monoFromMem_toComputeLosses after this step: [B, mem_size (cfg_mem_size + ppo_cfg_num_steps) + 1, 512, 32, 1]"""
        pred_monoFromMem_toComputeLosses = pred_monoFromMem.permute(1, 0, 2, 3, 4)

        # needed for mem_size == cfg_mem_size + ppo_cfg_num_steps... need to rotate the cols for computing
        # reward if there is a wraparound the memory to avoid bug"""
        pred_monoFromMem_toComputeLosses = pred_monoFromMem_toComputeLosses[valid_prediction_idxs_toComputeLosses[0],
                                                                            valid_prediction_idxs_toComputeLosses[1]]

        # next step tensor manipulations for reward
        next_gt_mono_mag_toComputeLosses =\
            rollouts_pol.observations["gt_mono_comps"][max(0, rollouts_pol.step + 2 -\
                                                           (num_past_steps_refinement + 1)):
                                                       rollouts_pol.step + 2][..., 0::2]\
                .clone()[..., 0].unsqueeze(-1).permute(1, 0, 2, 3, 4)
        next_gt_mono_mag_toComputeLosses[:, -1].copy_(batch["gt_mono_comps"][:, :, :, 0::2].clone()[:, :, :, 0].unsqueeze(-1))

        next_gt_mono_phase_toComputeLosses =\
            rollouts_pol.observations["gt_mono_comps"][max(0, rollouts_pol.step + 2 -
                                                           (num_past_steps_refinement + 1))
                                                       :rollouts_pol.step + 2][..., 1::2]\
                .clone()[..., 0].unsqueeze(-1).permute(1, 0, 2, 3, 4)
        next_gt_mono_phase_toComputeLosses[:, -1].copy_(batch["gt_mono_comps"][:, :, :, 1::2].clone()[:, :, :, 0].unsqueeze(-1))

        # next_pred_monoFromMem_toComputeLosses after this step: [B, mem_size (cfg_mem_size + ppo_cfg_num_steps) + 1, 512, 32, 1]
        next_pred_monoFromMem_toComputeLosses = next_pred_monoFromMem.permute(1, 0, 2, 3, 4)

        next_sepExtMem_masks_wCurrStep =\
            torch.cat([next_sepExtMem_masks, torch.ones([next_sepExtMem_masks.shape[0], 1],
                                                        device=next_sepExtMem_masks.device)],
                      dim=1)
        bs, M = next_sepExtMem_masks_wCurrStep.size()
        next_newest_sepExtMem_masks_wCurrStep =\
            next_sepExtMem_masks_wCurrStep[:, -1:].clone()
        next_old_sepExtMem_masks_wCurrStep =\
            next_sepExtMem_masks_wCurrStep[:, :-1].clone()
        next_old_activeIdxs_sepExtMem_masks_wCurrStep =\
            torch.where(next_old_sepExtMem_masks_wCurrStep == 1.0)
        next_old_sepExtMem_masks_wCurrStep[:, :] = 0.0
        if next_old_activeIdxs_sepExtMem_masks_wCurrStep[0].size()[0] != 0:
            next_old_activeColsFirstProcess_sepExtMem_masks_wCurrStep =\
                next_old_activeIdxs_sepExtMem_masks_wCurrStep[1].view(bs, -1)[0]
            """hacky and slow way to find out which idxs to choose for k_old_steps"""
            active_old_col = -1
            wrap_around_in_mem = False
            for active_col in next_old_activeColsFirstProcess_sepExtMem_masks_wCurrStep:
                if active_old_col != -1:
                    if active_old_col != (active_col.item() - 1):
                        wrap_around_in_mem = True
                        break
                active_old_col = active_col.item()

            if wrap_around_in_mem:
                last_active_col_left = active_old_col
                first_active_col_left = 0
                last_active_col_right = next_old_sepExtMem_masks_wCurrStep.size(1) - 1
                assert next_old_sepExtMem_masks_wCurrStep.size(1) - 1\
                       == next_old_activeColsFirstProcess_sepExtMem_masks_wCurrStep[-1].item()
                if last_active_col_left - first_active_col_left + 1 >= num_past_steps_refinement:
                    next_lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep =\
                        list(range(last_active_col_left - num_past_steps_refinement + 1, last_active_col_left + 1))
                    assert len(next_lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep) ==\
                           num_past_steps_refinement
                    next_old_activeCols_sepExtMem_masks_wCurrStep =\
                        torch.LongTensor(next_lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep)\
                            .unsqueeze(0).repeat(bs, 1).contiguous().view(-1)

                    next_lst_old_activeRowsOneProcess_sepExtMem_masks_wCurrStep = list(range(bs))
                    next_old_activeRows_sepExtMem_masks_wCurrStep =\
                        torch.LongTensor(next_lst_old_activeRowsOneProcess_sepExtMem_masks_wCurrStep)\
                            .unsqueeze(1).repeat(1, num_past_steps_refinement).contiguous().view(-1)
                else:
                    if next_old_activeColsFirstProcess_sepExtMem_masks_wCurrStep.size(0)\
                            >= num_past_steps_refinement:
                        remaining_numPastStepsRefinement_toBeSampledFromRight = num_past_steps_refinement -\
                                                                                (last_active_col_left - first_active_col_left + 1)
                        next_lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromLeft\
                            = list(range(0, last_active_col_left + 1))
                        next_lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromRight =\
                            list(range(last_active_col_right - remaining_numPastStepsRefinement_toBeSampledFromRight + 1,
                                       last_active_col_right + 1))
                        next_lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep\
                            = next_lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromRight +\
                              next_lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromLeft
                        assert len(next_lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep)\
                               == num_past_steps_refinement
                        next_old_activeCols_sepExtMem_masks_wCurrStep =\
                            torch.LongTensor(next_lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep)\
                                .unsqueeze(0).repeat(bs, 1).contiguous().view(-1)

                        next_lst_old_activeRowsOneProcess_sepExtMem_masks_wCurrStep = list(range(bs))
                        next_old_activeRows_sepExtMem_masks_wCurrStep =\
                            torch.LongTensor(next_lst_old_activeRowsOneProcess_sepExtMem_masks_wCurrStep)\
                                .unsqueeze(1).repeat(1, num_past_steps_refinement).contiguous().view(-1)
                    else:
                        remaining_numPastStepsRefinement_toBeSampledFromRight =\
                            next_old_activeColsFirstProcess_sepExtMem_masks_wCurrStep.size(0) -\
                            (last_active_col_left - first_active_col_left + 1)
                        next_lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromLeft\
                            = list(range(0, last_active_col_left + 1))
                        next_lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromRight =\
                            list(range(last_active_col_right - remaining_numPastStepsRefinement_toBeSampledFromRight + 1,
                                       last_active_col_right + 1))
                        next_lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep\
                            = next_lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromRight +\
                              next_lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromLeft
                        next_old_activeCols_sepExtMem_masks_wCurrStep =\
                            torch.LongTensor(next_lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep)\
                                .unsqueeze(0).repeat(bs, 1).contiguous().view(-1)

                        next_lst_old_activeRowsOneProcess_sepExtMem_masks_wCurrStep = list(range(bs))
                        next_old_activeRows_sepExtMem_masks_wCurrStep =\
                            torch.LongTensor(next_lst_old_activeRowsOneProcess_sepExtMem_masks_wCurrStep)\
                                .unsqueeze(1).repeat(1, next_old_activeColsFirstProcess_sepExtMem_masks_wCurrStep.size(0))\
                                .contiguous().view(-1)

            else:
                next_old_activeRows_sepExtMem_masks_wCurrStep =\
                    next_old_activeIdxs_sepExtMem_masks_wCurrStep[0].view(bs, -1)[:, -num_past_steps_refinement:].contiguous().view(-1)
                next_old_activeCols_sepExtMem_masks_wCurrStep =\
                    next_old_activeIdxs_sepExtMem_masks_wCurrStep[1].view(bs, -1)[:, -num_past_steps_refinement:].contiguous().view(-1)

            next_old_sepExtMem_masks_wCurrStep[next_old_activeRows_sepExtMem_masks_wCurrStep,
                                               next_old_activeCols_sepExtMem_masks_wCurrStep] = 1.0
            next_old_sepExtMem_masks_wCurrStep = next_old_sepExtMem_masks_wCurrStep.contiguous().view(bs, -1)

        next_sepExtMem_masks_wCurrStep =\
            torch.cat((next_old_sepExtMem_masks_wCurrStep,
                       next_newest_sepExtMem_masks_wCurrStep), dim=1)

        next_valid_prediction_idxs = torch.where(next_sepExtMem_masks_wCurrStep == 1.0)

        # needed for mem_size == cfg_mem_size + ppo_cfg_num_steps... need to rotate the cols for computing
        # reward if there is a wraparound the memory to avoid bug
        next_valid_prediction_cols = next_valid_prediction_idxs[1]
        bs, M = next_sepExtMem_masks_wCurrStep.size()
        next_valid_prediction_cols = next_valid_prediction_cols.contiguous().view(bs, -1)

        active_old_col = -1
        active_old_col_idx = 0
        wrap_around_in_mem = False
        for active_col in next_valid_prediction_cols[0][:-1]:
            if active_old_col != -1:
                if active_old_col != (active_col.item() - 1):
                    wrap_around_in_mem = True
                    break
            active_old_col = active_col.item()
            active_old_col_idx += 1
        if wrap_around_in_mem:
            next_valid_prediction_cols_toComputeLosses =\
                torch.cat((next_valid_prediction_cols[:, :-1][:, active_old_col_idx:],
                           next_valid_prediction_cols[:, :-1][:, :active_old_col_idx],
                           next_valid_prediction_cols[:, -1:]), dim=1)
        else:
            next_valid_prediction_cols_toComputeLosses = next_valid_prediction_cols.clone()
        # checking if rotation of cols that might be needed for mem_size (num_steps + trans_cfg_mem_size) working or not
        next_valid_prediction_idxs_toComputeLosses = (next_valid_prediction_idxs[0].clone(),
                                                      next_valid_prediction_cols_toComputeLosses.view(-1).contiguous())

        # needed for mem_size == cfg_mem_size + ppo_cfg_num_steps... need to rotate the cols for computing
        # reward if there is a wraparound the memory to avoid bug
        next_pred_monoFromMem_toComputeLosses =\
            next_pred_monoFromMem_toComputeLosses[next_valid_prediction_idxs_toComputeLosses[0],
                                                  next_valid_prediction_idxs_toComputeLosses[1]]

        rewards = override_rewards(rewards,
                                   dones,
                                   next_pred_monoFromMem[-1],
                                   next_gt_mono_mag,
                                   )

        for idx, done in enumerate(dones):
            if done:
                rewards[idx] = 0

        compute_losses = False
        last_step = False
        # next_pred_monoFromMem_toComputeLosses.size(0) == pred_monoFromMem_toComputeLosses.size(0) : all num_past_steps_refinement getting predicted
        # next_pred_monoFromMem_toComputeLosses.size(0) < pred_monoFromMem_toComputeLosses.size(0) : last step of episode
        # next_pred_monoFromMem_toComputeLosses.size(0) > pred_monoFromMem_toComputeLosses.size(0) : one of  first num_past_steps_refinement - 1 steps
        if (next_pred_monoFromMem_toComputeLosses.size(0) == pred_monoFromMem_toComputeLosses.size(0)) or\
                (next_pred_monoFromMem_toComputeLosses.size(0) < pred_monoFromMem_toComputeLosses.size(0)):
            compute_losses = True
            if next_pred_monoFromMem_toComputeLosses.size(0) < pred_monoFromMem_toComputeLosses.size(0):
                last_step = True
                pred_mono_toComputeLosses = pred_monoFromMem_toComputeLosses
                gt_mono_toComputeLosses = torch.cat((gt_mono_mag_toComputeLosses, gt_mono_phase_toComputeLosses),
                                                    dim=-1)
            else:
                lst_oldestStep_samplingIdx =\
                    list(range(0, bs * (num_past_steps_refinement + 1), num_past_steps_refinement + 1))
                pred_mono_toComputeLosses = pred_monoFromMem_toComputeLosses.contiguous()[lst_oldestStep_samplingIdx]
                gt_mono_toComputeLosses =\
                    torch.cat((gt_mono_mag_toComputeLosses.contiguous()[lst_oldestStep_samplingIdx],
                               gt_mono_phase_toComputeLosses.contiguous()[lst_oldestStep_samplingIdx]),
                              dim=-1)

        if compute_losses:
            _, monoFromMem_losses =\
                 STFT_L2_distance(step_observation["mixed_bin_audio_mag"],
                                  pred_binSepMasks.detach(),
                                  step_observation["gt_bin_comps"].clone(),
                                  pred_mono_toComputeLosses,
                                  gt_mono_toComputeLosses.clone(),
                                  )

            if last_step:
                monoFromMem_losses_mean = torch.mean(monoFromMem_losses.view(bs, -1), dim=-1).unsqueeze(-1)
                monoFromMem_losses = torch.sum(monoFromMem_losses.view(bs, -1), dim=-1).unsqueeze(-1)
            else:
                monoFromMem_losses = torch.mean(monoFromMem_losses.view(bs, -1), dim=-1).unsqueeze(-1)

        bin_losses, mono_losses =\
             STFT_L2_distance(step_observation["mixed_bin_audio_mag"],
                              pred_binSepMasks.detach(),
                              step_observation["gt_bin_comps"].clone(),
                              pred_mono,
                              step_observation["gt_mono_comps"].clone(),
                              )

        pth_time += time.time() - t_compute_lossesNrewards

        t_update_stats = time.time()

        batch = batch_obs(observations)
        rewards = torch.tensor(rewards, dtype=torch.float)
        rewards = rewards.unsqueeze(1)

        current_episode_reward += rewards
        current_episode_step += 1
        current_episode_dist_probs += distribution_probs.detach().cpu()
        current_episode_bin_losses += bin_losses
        current_episode_mono_losses += mono_losses
        if compute_losses:
            current_episode_monoFromMem_losses += monoFromMem_losses

        # current_episode_reward is accumulating rewards across multiple updates,
        # as long as the current episode is not finished
        # the current episode reward is added to the episode rewards only if the current episode is done
        # the episode count will also increase by 1
        episode_rewards += (1 - masks) * current_episode_reward
        episode_steps += (1 - masks) * current_episode_step
        episode_counts += 1 - masks
        episode_dist_probs += (1 - masks) * (current_episode_dist_probs / current_episode_step)
        episode_bin_losses_allSteps += (1 - masks) * (current_episode_bin_losses / current_episode_step)
        episode_mono_losses_lastStep += (1 - masks) * mono_losses
        episode_mono_losses_allSteps += (1 - masks) * (current_episode_mono_losses / current_episode_step)
        if compute_losses:
            if last_step:
                episode_monoFromMem_losses_lastStep += (1 - masks) * monoFromMem_losses_mean
            else:
                episode_monoFromMem_losses_lastStep += (1 - masks) * monoFromMem_losses
        episode_monoFromMem_losses_allSteps += (1 - masks) * (current_episode_monoFromMem_losses / current_episode_step)

        # zeroing out current values when done
        current_episode_reward *= masks
        current_episode_step *= masks
        current_episode_bin_losses *= masks
        current_episode_mono_losses *= masks
        current_episode_monoFromMem_losses *= masks
        current_episode_dist_probs *= masks

        pth_time += time.time() - t_update_stats

        rollouts_pol.insert(
            batch,
            recurrent_hidden_states_pol,
            actions,
            actions_log_probs,
            values,
            rewards,
            masks,
            pred_binSepMasks=pred_binSepMasks,
            pred_monoFromMem=pred_monoFromMem[-1],
        )

        return pth_time, env_time, self.envs.num_envs

    def _update_pol(self, rollouts_pol,):
        """
        updates AAViDSS policy
        :param rollouts_pol: rollout storage for the policy
        :return: 1. time.time() - t_update_model: time needed for policy update
                 2. value_loss: PPO value loss in this update
                 3. action_loss: PPO actions loss in this update
                 4. dist_entropy: PPO entropy loss in this update
        """
        ppo_cfg = self.config.RL.PPO
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[-1] for k, v in rollouts_pol.observations.items()
            }

            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts_pol.recurrent_hidden_states_pol[-1],
                rollouts_pol.masks[-1],
                rollouts_pol.pred_monoFromMem[-1],
                rollouts_pol.pred_binSepMasks[-1],
            ).detach()

        rollouts_pol.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        value_loss, action_loss, dist_entropy = self.agent.update_pol(rollouts_pol)

        rollouts_pol.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )

    def _update_sep(self, rollouts_sep):
        """
        updates audio memory (passive separators frozen)
        :param rollouts_sep:
        :return:    1. time.time() - t_update_model: time needed for separator (acoustic memory) update
                    2. bin_loss: binaural loss for the passive separator in this update (for debugging)
                    3. mono_loss: monaural loss for the passive separator in this update (for debugging)
                    4. monoFromMem_loss: computed on the output of the acoustic memory)
        """
        t_update_model = time.time()

        bin_loss, mono_loss, monoFromMem_loss = self.agent.update_sep(rollouts_sep)

        rollouts_sep.after_update()

        return (
            time.time() - t_update_model,
            bin_loss,
            mono_loss,
            monoFromMem_loss
        )

    def _load_pretrained_passive_separators(self):
        r"""
        loads pretrained passive separators and freezes them for final Move2Hear training
        :return: None
        """
        ppo_cfg = self.config.RL.PPO

        assert ppo_cfg.pretrained_passive_separators_ckpt != ""
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(ppo_cfg.pretrained_passive_separators_ckpt, map_location="cpu")
        self.agent.load_pretrained_passive_separators(ckpt_dict["state_dict"])

        # freezing parameters of passive binaural separator
        assert hasattr(self.agent.actor_critic, "binSep_enc")
        self.agent.actor_critic.binSep_enc.eval()
        for param in self.agent.actor_critic.binSep_enc.parameters():
            if param.requires_grad:
                param.requires_grad_(False)
        assert hasattr(self.agent.actor_critic, "binSep_dec")
        self.agent.actor_critic.binSep_dec.eval()
        for param in self.agent.actor_critic.binSep_dec.parameters():
            if param.requires_grad:
                param.requires_grad_(False)

        # freezing parameters of passive bin2mono converter
        assert hasattr(self.agent.actor_critic, "bin2mono_enc")
        self.agent.actor_critic.bin2mono_enc.eval()
        for param in self.agent.actor_critic.bin2mono_enc.parameters():
            if param.requires_grad:
                param.requires_grad_(False)
        assert hasattr(self.agent.actor_critic, "bin2mono_dec")
        self.agent.actor_critic.bin2mono_dec.eval()
        for param in self.agent.actor_critic.bin2mono_dec.parameters():
            if param.requires_grad:
                param.requires_grad_(False)

    def train(self) -> None:
        r"""Main method for training cyclic training of AAViDSS policy and separator..

        Returns:
            None
        """
        ppo_cfg = self.config.RL.PPO
        sepExtMem_cfg = ppo_cfg.TRANSFORMER_MEMORY
        poseEnc_cfg = sepExtMem_cfg.POSE_ENCODING

        if ppo_cfg.use_ddppo:
            self.local_rank, tcp_store = init_distrib_slurm(
                ppo_cfg.ddppo_distrib_backend, master_port=ppo_cfg.master_port, master_addr=ppo_cfg.master_addr,
            )
            add_signal_handlers()

            num_rollouts_done_store = distrib.PrefixStore(
                "rollout_tracker", tcp_store
            )
            num_rollouts_done_store.set("num_done", "0")

            self.world_rank = distrib.get_rank()
            self.world_size = distrib.get_world_size()

            self.config.defrost()
            self.config.TORCH_GPU_ID = self.local_rank
            self.config.SIMULATOR_GPU_ID = self.local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.SEED += (
                self.world_rank * self.config.NUM_PROCESSES
            )
            self.config.TASK_CONFIG.SIMULATOR.SEED = self.config.SEED
            self.config.freeze()

        if (not ppo_cfg.use_ddppo) or (ppo_cfg.use_ddppo and self.world_rank == 0):
            logger.info(f"config: {self.config}")

        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME), workers_ignore_signals=True if ppo_cfg.use_ddppo else False,
        )

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if ppo_cfg.use_ddppo:
            torch.cuda.set_device(self.device)

        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent()
        self._load_pretrained_passive_separators()

        if ppo_cfg.use_ddppo:
            self.agent.init_distributed(find_unused_params=True)

        if (ppo_cfg.use_ddppo and self.world_rank == 0) or (not ppo_cfg.use_ddppo):
            logger.info(
                "agent number of trainable parameters: {}".format(
                    sum(param.numel() for param in self.agent.parameters() if param.requires_grad)
                )
            )

        rollouts_pol = RolloutStoragePol(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            ppo_cfg.hidden_size,
        )
        rollouts_sep = RolloutStorageSep(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            external_memory_size=ppo_cfg.num_steps + sepExtMem_cfg.memory_size,
            external_memory_capacity=sepExtMem_cfg.memory_size,
            external_memory_dim=512 * 32,
            poseEnc_cfg=poseEnc_cfg,
        )

        rollouts_pol.to(self.device)
        rollouts_sep.to(self.device)

        observations = self.envs.reset()
        if ppo_cfg.use_ddppo:
            batch = batch_obs(observations, device=self.device)
        else:
            batch = batch_obs(observations)

        for sensor in rollouts_pol.observations:
            rollouts_pol.observations[sensor][0].copy_(batch[sensor])
            rollouts_sep.observations[sensor][0].copy_(batch[sensor])

        # episode_x accumulates over the entire training course
        episode_rewards = torch.zeros(self.envs.num_envs, 1)
        episode_counts = torch.zeros(self.envs.num_envs, 1)
        episode_steps = torch.zeros(self.envs.num_envs, 1)
        episode_dist_probs = torch.zeros(self.envs.num_envs, self.envs.action_spaces[0].n)
        episode_bin_losses_allSteps = torch.zeros(self.envs.num_envs, 1)
        episode_mono_losses_lastStep = torch.zeros(self.envs.num_envs, 1)
        episode_mono_losses_allSteps = torch.zeros(self.envs.num_envs, 1)
        episode_monoFromMem_losses_lastStep = torch.zeros(self.envs.num_envs, 1)
        episode_monoFromMem_losses_allSteps = torch.zeros(self.envs.num_envs, 1)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        current_episode_step = torch.zeros(self.envs.num_envs, 1)
        current_episode_dist_probs = torch.zeros(self.envs.num_envs, self.envs.action_spaces[0].n)
        current_episode_bin_losses = torch.zeros(self.envs.num_envs, 1)
        current_episode_mono_losses = torch.zeros(self.envs.num_envs, 1)
        current_episode_monoFromMem_losses = torch.zeros(self.envs.num_envs, 1)

        window_episode_reward = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_counts = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_step = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_dist_probs = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_bin_losses_allSteps = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_mono_losses_lastStep = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_mono_losses_allSteps = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_monoFromMem_losses_lastStep = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_monoFromMem_losses_allSteps = deque(maxlen=ppo_cfg.reward_window_size)

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        lr_scheduler_pol = LambdaLR(
            optimizer=self.agent.optimizer_pol,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )
        lr_scheduler_sep = LambdaLR(
            optimizer=self.agent.optimizer_sep,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        if ppo_cfg.use_ddppo:
            writer_obj = TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            ) if self.world_rank == 0 else contextlib.suppress()
        else:
            writer_obj = TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            )

        with writer_obj as writer:
            for update in range(int(self.config.NUM_UPDATES)):
                count_steps_lst = []
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler_pol.step()
                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                if ppo_cfg.use_ddppo:
                    count_steps_delta = 0

                for step in range(ppo_cfg.num_steps):
                    # with sdr, sir, sar
                    delta_pth_time, delta_env_time, delta_steps = self._collect_rollout_step(
                        rollouts_pol,
                        rollouts_sep,
                        current_episode_reward,
                        current_episode_step,
                        current_episode_dist_probs,
                        current_episode_bin_losses,
                        current_episode_mono_losses,
                        current_episode_monoFromMem_losses,
                        episode_rewards,
                        episode_counts,
                        episode_steps,
                        episode_dist_probs,
                        episode_bin_losses_allSteps,
                        episode_mono_losses_lastStep,
                        episode_mono_losses_allSteps,
                        episode_monoFromMem_losses_lastStep,
                        episode_monoFromMem_losses_allSteps,
                    )

                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    if ppo_cfg.use_ddppo:
                        count_steps_delta += delta_steps
                        if (
                            step
                            >= ppo_cfg.num_steps * ppo_cfg.short_rollout_threshold
                        ) and int(num_rollouts_done_store.get("num_done")) > (
                            ppo_cfg.sync_frac * self.world_size
                        ):
                            break
                    else:
                        count_steps += delta_steps

                if ppo_cfg.use_ddppo:
                    num_rollouts_done_store.add("num_done", 1)

                delta_pth_time, value_loss, action_loss, dist_entropy = self._update_pol(
                    rollouts_pol,
                )

                pth_time += delta_pth_time

                # computing stats
                if ppo_cfg.use_ddppo:
                    stat_idx = 0
                    stat_idx_num_actions = 0
                    stat_name_to_idx = {}
                    stat_name_to_idx_num_actions = {}
                    stack_lst_for_stats = []
                    stack_lst_for_stats_num_actions = []

                    stack_lst_for_stats.append(episode_rewards)
                    stat_name_to_idx["rewards"] = stat_idx
                    stat_idx += 1

                    stack_lst_for_stats.append(episode_counts)
                    stat_name_to_idx["counts"] = stat_idx
                    stat_idx += 1

                    stack_lst_for_stats.append(episode_steps)
                    stat_name_to_idx["steps"] = stat_idx
                    stat_idx += 1

                    stack_lst_for_stats_num_actions.append(episode_dist_probs)
                    stat_name_to_idx_num_actions["dist_probs"] = stat_idx_num_actions
                    stat_idx_num_actions += 1

                    stack_lst_for_stats.append(episode_bin_losses_allSteps)
                    stat_name_to_idx["avg_bin_losses_allSteps"] = stat_idx
                    stat_idx += 1

                    stack_lst_for_stats.append(episode_mono_losses_lastStep)
                    stat_name_to_idx["mono_losses_lastStep"] = stat_idx
                    stat_idx += 1
                    stack_lst_for_stats.append(episode_mono_losses_allSteps)
                    stat_name_to_idx["avg_mono_losses_allSteps"] = stat_idx
                    stat_idx += 1

                    stack_lst_for_stats.append(episode_monoFromMem_losses_lastStep)
                    stat_name_to_idx["monoFromMem_losses_lastStep"] = stat_idx
                    stat_idx += 1
                    stack_lst_for_stats.append(episode_monoFromMem_losses_allSteps)
                    stat_name_to_idx["avg_monoFromMem_losses_allSteps"] = stat_idx
                    stat_idx += 1

                    stats = torch.stack(stack_lst_for_stats, 0).to(self.device)
                    distrib.all_reduce(stats)
                    stats_num_actions = torch.stack(stack_lst_for_stats_num_actions, 0).to(self.device)
                    distrib.all_reduce(stats_num_actions)

                    window_episode_reward.append(stats[stat_name_to_idx["rewards"]].clone())
                    window_episode_counts.append(stats[stat_name_to_idx["counts"]].clone())
                    window_episode_step.append(stats[stat_name_to_idx["steps"]].clone())
                    window_episode_dist_probs.append(stats_num_actions[stat_name_to_idx_num_actions["dist_probs"]].clone())
                    window_episode_bin_losses_allSteps.append(stats[stat_name_to_idx["avg_bin_losses_allSteps"]].clone())
                    window_episode_mono_losses_lastStep.append(stats[stat_name_to_idx["mono_losses_lastStep"]].clone())
                    window_episode_mono_losses_allSteps.append(stats[stat_name_to_idx["avg_mono_losses_allSteps"]].clone())
                    window_episode_monoFromMem_losses_lastStep.append(stats[stat_name_to_idx["monoFromMem_losses_lastStep"]].clone())
                    window_episode_monoFromMem_losses_allSteps.append(stats[stat_name_to_idx["avg_monoFromMem_losses_allSteps"]].clone())

                    stats = torch.tensor(
                        [value_loss, action_loss, dist_entropy, count_steps_delta], device=self.device,
                    )
                    distrib.all_reduce(stats)
                    count_steps += stats[3].item()

                    if self.world_rank == 0:
                        num_rollouts_done_store.set("num_done", "0")
                        value_loss = stats[0].item() / self.world_size
                        action_loss = stats[1].item() / self.world_size
                        dist_entropy = stats[2].item() / self.world_size
                else:
                    window_episode_reward.append(episode_rewards.clone())
                    window_episode_counts.append(episode_counts.clone())
                    window_episode_step.append(episode_steps.clone())
                    window_episode_dist_probs.append(episode_dist_probs.clone())
                    window_episode_bin_losses_allSteps.append(episode_bin_losses_allSteps.clone())
                    window_episode_mono_losses_lastStep.append(episode_mono_losses_lastStep.clone())
                    window_episode_mono_losses_allSteps.append(episode_mono_losses_allSteps.clone())
                    window_episode_monoFromMem_losses_lastStep.append(episode_monoFromMem_losses_lastStep.clone())
                    window_episode_monoFromMem_losses_allSteps.append(episode_monoFromMem_losses_allSteps.clone())

                if (ppo_cfg.use_ddppo and self.world_rank == 0) or (not ppo_cfg.use_ddppo):
                    stats_keys = ["count", "reward", "step", 'dist_probs', 'avg_bin_loss_allSteps',
                                  'mono_loss_lastStep', 'mono_loss_allSteps', 'monoFromMem_loss_lastStep',
                                  'monoFromMem_loss_allSteps']
                    stats_vals = [window_episode_counts, window_episode_reward, window_episode_step, window_episode_dist_probs,
                                  window_episode_bin_losses_allSteps, window_episode_mono_losses_lastStep,
                                  window_episode_mono_losses_allSteps, window_episode_monoFromMem_losses_lastStep,
                                  window_episode_monoFromMem_losses_allSteps]
                    stats = zip(stats_keys, stats_vals)

                    deltas = {}
                    for k, v in stats:
                        if len(v) > 1:
                            deltas[k] = (v[-1] - v[0]).sum(dim=0)\
                                if (k == "dist_probs")\
                                else (v[-1] - v[0]).sum().item()
                        else:
                            deltas[k] = v[0].sum(dim=0) if (k == "dist_probs")\
                                else v[0].sum().item()

                    deltas["count"] = max(deltas["count"], 1.0)
                    count_steps_lst.append(count_steps)

                    # this reward is averaged over all the episodes happened during window_size updates
                    # approximately number of steps is window_size * num_steps
                    writer.add_scalar(
                        "Environment/Reward", deltas["reward"] / deltas["count"], count_steps
                    )
                    logging.debug('Number of steps: {}'.format(deltas["step"] / deltas["count"]))
                    writer.add_scalar(
                        "Environment/Episode_length", deltas["step"] / deltas["count"], count_steps
                    )
                    for i in range(self.envs.action_spaces[0].n):
                        if not isinstance(deltas['dist_probs'] / deltas["count"], float):
                            writer.add_scalar(
                                "Policy/Action_prob_{}".format(i), (deltas['dist_probs'] / deltas["count"])[i].item(),
                                count_steps
                            )
                        else:
                            writer.add_scalar(
                                "Policy/Action_prob_{}".format(i), deltas['dist_probs'] / deltas["count"], count_steps
                            )
                    writer.add_scalar(
                        "Environment/STFT_L2_loss/mono_lastStep", deltas['mono_loss_lastStep'] / deltas["count"],
                        count_steps
                    )
                    writer.add_scalar(
                        "Environment/STFT_L2_loss/mono_avgAllSteps", deltas['mono_loss_allSteps'] / deltas["count"],
                        count_steps
                    )
                    writer.add_scalar(
                        "Environment/STFT_L2_loss/monoFromMem_lastStep", deltas['monoFromMem_loss_lastStep'] / deltas["count"],
                        count_steps
                    )
                    writer.add_scalar(
                        "Environment/STFT_L2_loss/monoFromMem_avgAllSteps", deltas['monoFromMem_loss_allSteps'] / deltas["count"],
                        count_steps
                    )

                    writer.add_scalar(
                        'Policy/Value_Loss', value_loss, count_steps
                    )
                    writer.add_scalar(
                        'Policy/Action_Loss', action_loss, count_steps
                    )
                    writer.add_scalar(
                        'Policy/Entropy', dist_entropy, count_steps
                    )
                    writer.add_scalar(
                        'Policy/Learning_Rate', lr_scheduler_pol.get_lr()[0], count_steps
                    )

                    # log stats
                    if update > 0 and (update % self.config.LOG_INTERVAL == 0):

                        window_rewards = (
                            window_episode_reward[-1] - window_episode_reward[0]
                        ).sum()
                        window_counts = (
                            window_episode_counts[-1] - window_episode_counts[0]
                        ).sum()

                        if window_counts > 0:
                            logger.info(
                                "Average window size {} reward: {:3f}".format(
                                    len(window_episode_reward),
                                    (window_rewards / window_counts).item(),
                                )
                            )
                        else:
                            logger.info("No episodes finish in current window")

                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler_sep.step()

                delta_pth_time, bin_loss, mono_loss, monoFromMem_loss = self._update_sep(
                    rollouts_sep,
                )
                if ppo_cfg.use_ddppo:
                    sep_loss_stats = torch.tensor(
                        [bin_loss, mono_loss, monoFromMem_loss],
                    )

                pth_time += delta_pth_time

                if ppo_cfg.use_ddppo:
                    distrib.all_reduce(sep_loss_stats.to(self.device))

                if (ppo_cfg.use_ddppo and self.world_rank == 0) or (not ppo_cfg.use_ddppo):
                    if update > 0 and update % self.config.LOG_INTERVAL == 0:
                        logger.info(
                            "update: {}\tfps: {:.3f}\t".format(
                                update, count_steps_lst[0] / (time.time() - t_start)
                            )
                        )

                        logger.info(
                            "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                            "frames: {}".format(
                                update, env_time, pth_time, count_steps_lst[0]
                            )
                        )
                    if update % self.config.CHECKPOINT_INTERVAL == 0:
                        self.save_checkpoint(f"ckpt.{count_checkpoints}.pth")
                        count_checkpoints += 1

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> Dict:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        # map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        # setting up config
        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        # eval on those scenes only whose names are given in the eval config
        if len(config.EPS_SCENES) != 0:
            full_dataset_path = os.path.join(config.TASK_CONFIG.DATASET.DATA_PATH.split("{")[0],
                                             config.TASK_CONFIG.DATASET.VERSION,
                                             config.TASK_CONFIG.DATASET.SPLIT,
                                             f"{config.TASK_CONFIG.DATASET.SPLIT}.json.gz")

            with gzip.GzipFile(full_dataset_path, "rb") as fo:
                dataset = fo.read()
            dataset = dataset.decode("utf-8")
            dataset = json.loads(dataset)
            dataset_episodes = dataset["episodes"]

            eval_episode_count = 0
            for scene in config.EPS_SCENES:
                for episode in dataset_episodes:
                    if episode["scene_id"].split("/")[0] == scene:
                        eval_episode_count += 1
                        
            if config.EVAL_EPISODE_COUNT > eval_episode_count:
                config.defrost()
                config.EVAL_EPISODE_COUNT = eval_episode_count
                config.freeze()

        logger.info(f"env config: {config}")

        ppo_cfg = config.RL.PPO

        sepExtMem_cfg = ppo_cfg.TRANSFORMER_MEMORY
        num_past_steps_refinement = sepExtMem_cfg.num_past_steps_refinement

        poseEnc_cfg = sepExtMem_cfg.POSE_ENCODING

        env_cfg = config.TASK_CONFIG.ENVIRONMENT

        # setting up envs
        self.envs = construct_envs(
            config, get_env_class(config.ENV_NAME),
        )

        # setting up agent
        self._setup_actor_critic_agent(is_eval=True)

        ########### TEMPORARY VARIABLES / OBJECTS FOR EVAL ########
        # loading trained weights to policies,  creating empty tensors for eval and setting flags for policy switching in eval
        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic
        self.actor_critic.eval()
        self.agent.eval()
        self.agent.actor_critic.eval()

        not_done_masks = torch.ones(
            config.NUM_PROCESSES, 1, device=self.device
        )
        test_recurrent_hidden_states_pol = torch.zeros(
            self.actor_critic.pol_net.num_recurrent_layers,
            config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )

        test_em = ExternalMemory(
            config.NUM_PROCESSES,
            sepExtMem_cfg.memory_size + ppo_cfg.num_steps,
            sepExtMem_cfg.memory_size,
            512 * 32 + poseEnc_cfg.num_pose_attrs,
            )
        test_em.to(self.device)

        gtMonoMag_thisEpisode = torch.zeros(
            env_cfg.MAX_EPISODE_STEPS,
            config.NUM_PROCESSES,
            *self.envs.observation_spaces[0].spaces["gt_mono_comps"].shape,
        ).to(self.device)

        mixedAudioMag_thisEpisode = []
        pred_binSepMasks_thisEpisode = []
        gtBinComps_thisEpisode = []

        mixedAudioMag_thisEpisode_computeQualMetrics = []
        mixedAudioPhase_thisEpisode_computeQualMetrics = []
        pred_mono_thisEpisode_computeQualMetrics = []
        lst_gtMonoMag_thisEpisode_computeQualMetrics = []
        lst_gtMonoPhase_thisEpisode_computeQualMetrics = []

        monoLosses_thisEpisode = []
        pred_monoFromMem_computeLosses_thisEpisode = []
        gtMono_computeLosses_thisEpisode = []

        t = tqdm(total=config.EVAL_EPISODE_COUNT)
        active_envs = list(range(self.envs.num_envs))
        # these are particularly useful for multi-process eval but code doesn't support it for now (feel free to start a PR
        # to include changes for multi-process eval)
        step_count_all_processes = torch.zeros(config.NUM_PROCESSES, 1)
        episode_count_all_processes = torch.zeros(config.NUM_PROCESSES, 1)
        num_episode_numbers_taken = config.NUM_PROCESSES - 1
        for episode_count_idx in range(config.NUM_PROCESSES):
            episode_count_all_processes[episode_count_idx] = episode_count_idx

        # dict of dicts that stores stats per episode
        stats_episodes = dict()

        mono_losses_last_step = []
        mono_losses_all_steps = []
        mono_loss_this_episode = 0.
        monoFromMem_losses_last_step = []
        monoFromMem_losses_all_steps = []
        monoFromMem_loss_this_episode = 0.

        # dump quality metrics and losses per step per episode for post-processing and getting final metric
        if config.COMPUTE_EVAL_METRICS:
            eval_metrics_toDump = {"mono": {}, "monoFromMem": {}}
            for metric in config.EVAL_METRICS_TO_COMPUTE:
                eval_metrics_toDump["mono"][metric] = {}
                eval_metrics_toDump["monoFromMem"][metric] = {}
            eval_metrics_toDump["mono"]["STFT_L2_loss"] = {}
            eval_metrics_toDump["monoFromMem"]["STFT_L2_loss"] = {}

        # resetting environments for 1st step of eval
        observations = self.envs.reset()
        batch = batch_obs(observations, self.device)

        # looping over episodes
        while (
            len(stats_episodes) < config.EVAL_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            ### ALL CODE HERE ONWARDS ASSUME 1-PROCESS EVAL
            # scene and episode id
            current_scene = current_episodes[0].scene_id.split('/')[-2]
            current_episode_id = current_episodes[0].episode_id

            # episode and step count
            current_episode_count = int(episode_count_all_processes[0].item())
            current_step_count = int(step_count_all_processes[0].item())

            # particularly useful for multi-process eval
            # hack to not collect stats from environments which have finished
            active_envs_tmp = []
            for env_idx in active_envs:
                if env_idx > 0:
                    raise NotImplementedError
                if (current_episodes[env_idx].scene_id.split('/')[-2], current_episodes[env_idx].episode_id) not in stats_episodes:
                    active_envs_tmp.append(env_idx)
            active_envs = active_envs_tmp

            with torch.no_grad():
                # passive-separate mono given target class
                pred_binSepMasks =\
                    self.actor_critic.get_binSepMasks(
                        batch
                    )
                pred_mono =\
                    self.actor_critic.convert_bin2mono(pred_binSepMasks.detach(),
                                                       mixed_audio=batch["mixed_bin_audio_mag"],
                                                       )

                gtMonoMag_thisEpisode[current_step_count].copy_(batch["gt_mono_comps"])

                sepExtMem_mono = test_em.memory[:, 0]
                sepExtMem_masks = test_em.masks
                sepExtMem_skipFeats = test_em.memory_skipFeats[:, 0]
                
                pred_monoFromMem, pred_mono_toCache, pred_monoFromMem_aftrAtt_feats, skip_feats =\
                    self.actor_critic.get_monoFromMem(pred_mono=pred_mono,
                                                      sepExtMem_mono=sepExtMem_mono,
                                                      sepExtMem_masks=sepExtMem_masks,
                                                      pose=batch["pose"],
                                                      sepExtMem_skipFeats=sepExtMem_skipFeats,
                                                      )

                sepExtMem_masks_wCurrStep = torch.cat([sepExtMem_masks,
                                                       torch.ones([sepExtMem_masks.shape[0], 1],
                                                                  device=sepExtMem_masks.device)],
                                                      dim=1)

                # find out indexes to set num_past_steps_refinement steps to True
                bs, M = sepExtMem_masks_wCurrStep.size()
                newest_sepExtMem_masks_wCurrStep = sepExtMem_masks_wCurrStep[:, -1:].clone()
                old_sepExtMem_masks_wCurrStep = sepExtMem_masks_wCurrStep[:, :-1].clone()
                old_activeIdxs_sepExtMem_masks_wCurrStep = torch.where(old_sepExtMem_masks_wCurrStep == 1.0)
                old_sepExtMem_masks_wCurrStep[:, :] = 0.0
                if old_activeIdxs_sepExtMem_masks_wCurrStep[0].size()[0] != 0:
                    old_activeColsFirstProcess_sepExtMem_masks_wCurrStep =\
                        old_activeIdxs_sepExtMem_masks_wCurrStep[1].view(bs, -1)[0]

                    # hacky and slow way to find out which idxs to choose for num_past_steps_refinement
                    active_old_col = -1
                    wrap_around_in_mem = False
                    for active_col in old_activeColsFirstProcess_sepExtMem_masks_wCurrStep:
                        if active_old_col != -1:
                            if active_old_col != (active_col.item() - 1):
                                wrap_around_in_mem = True
                                break
                        active_old_col = active_col.item()

                    if wrap_around_in_mem:
                        last_active_col_left = active_old_col
                        first_active_col_left = 0
                        last_active_col_right = old_sepExtMem_masks_wCurrStep.size(1) - 1
                        assert old_sepExtMem_masks_wCurrStep.size(1) - 1\
                               == old_activeColsFirstProcess_sepExtMem_masks_wCurrStep[-1].item()
                        if last_active_col_left - first_active_col_left + 1 >= num_past_steps_refinement:
                            lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep =\
                                list(range(last_active_col_left - num_past_steps_refinement + 1,
                                           last_active_col_left + 1))
                            assert len(lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep)\
                                   == num_past_steps_refinement
                            old_activeCols_sepExtMem_masks_wCurrStep =\
                                torch.LongTensor(lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep).unsqueeze(0)\
                                    .repeat(bs, 1).contiguous().view(-1)

                            lst_old_activeRowsOneProcess_sepExtMem_masks_wCurrStep = list(range(bs))
                            old_activeRows_sepExtMem_masks_wCurrStep =\
                                torch.LongTensor(lst_old_activeRowsOneProcess_sepExtMem_masks_wCurrStep).unsqueeze(1)\
                                    .repeat(1, num_past_steps_refinement).contiguous().view(-1)
                        else:
                            if old_activeColsFirstProcess_sepExtMem_masks_wCurrStep.size(0)\
                                    >= num_past_steps_refinement:
                                remaining_numPastStepsRefinement_toBeSampledFromRight =\
                                    ppo_cfg.self_attn_n_seq_mem_cfg.k_old_steps -\
                                    (last_active_col_left - first_active_col_left + 1)
                                lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromLeft\
                                    = list(range(0, last_active_col_left + 1))
                                lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromRight =\
                                    list(range(last_active_col_right - remaining_numPastStepsRefinement_toBeSampledFromRight + 1,
                                               last_active_col_right + 1))
                                lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep\
                                    = lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromRight +\
                                      lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromLeft
                                assert len(lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep)\
                                       == num_past_steps_refinement
                                old_activeCols_sepExtMem_masks_wCurrStep =\
                                    torch.LongTensor(lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep).unsqueeze(0)\
                                        .repeat(bs, 1).contiguous().view(-1)

                                lst_old_activeRowsOneProcess_sepExtMem_masks_wCurrStep = list(range(bs))
                                old_activeRows_sepExtMem_masks_wCurrStep =\
                                    torch.LongTensor(lst_old_activeRowsOneProcess_sepExtMem_masks_wCurrStep).unsqueeze(1)\
                                        .repeat(1, num_past_steps_refinement).contiguous().view(-1)
                            else:
                                remaining_numPastStepsRefinement_toBeSampledFromRight =\
                                    old_activeColsFirstProcess_sepExtMem_masks_wCurrStep.size(0) -\
                                    (last_active_col_left - first_active_col_left + 1)
                                lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromLeft\
                                    = list(range(0, last_active_col_left + 1))
                                lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromRight =\
                                    list(range(last_active_col_right - remaining_numPastStepsRefinement_toBeSampledFromRight + 1,
                                               last_active_col_right + 1))
                                lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep\
                                    = lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromRight +\
                                      lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromLeft
                                old_activeCols_sepExtMem_masks_wCurrStep =\
                                    torch.LongTensor(lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep).unsqueeze(0)\
                                        .repeat(bs, 1).contiguous().view(-1)

                                lst_old_activeRowsOneProcess_sepExtMem_masks_wCurrStep = list(range(bs))
                                old_activeRows_sepExtMem_masks_wCurrStep =\
                                    torch.LongTensor(lst_old_activeRowsOneProcess_sepExtMem_masks_wCurrStep).unsqueeze(1)\
                                        .repeat(1, old_activeColsFirstProcess_sepExtMem_masks_wCurrStep.size(0)).contiguous().view(-1)
                    else:
                        old_activeRows_sepExtMem_masks_wCurrStep =\
                            old_activeIdxs_sepExtMem_masks_wCurrStep[0].view(bs, -1)[:, -num_past_steps_refinement:]\
                                .contiguous().view(-1)
                        old_activeCols_sepExtMem_masks_wCurrStep =\
                            old_activeIdxs_sepExtMem_masks_wCurrStep[1].view(bs, -1)[:, -num_past_steps_refinement:]\
                                .contiguous().view(-1)

                    old_sepExtMem_masks_wCurrStep[old_activeRows_sepExtMem_masks_wCurrStep,
                                                  old_activeCols_sepExtMem_masks_wCurrStep] = 1.0
                    old_sepExtMem_masks_wCurrStep = old_sepExtMem_masks_wCurrStep.contiguous().view(bs, -1)

                sepExtMem_masks_wCurrStep = torch.cat((old_sepExtMem_masks_wCurrStep,
                                                       newest_sepExtMem_masks_wCurrStep), dim=1)

                valid_prediction_idxs = torch.where(sepExtMem_masks_wCurrStep == 1.0)

                # needed for mem_size == cfg_mem_size + ppo_cfg_num_steps... need to rotate the cols for computing
                # reward if there is a wraparound the memory to avoid bug
                valid_prediction_cols = valid_prediction_idxs[1]
                bs, M = sepExtMem_masks_wCurrStep.size()
                valid_prediction_cols = valid_prediction_cols.contiguous().view(bs, -1)

                active_old_col = -1
                active_old_col_idx = 0
                wrap_around_in_mem = False
                for active_col in valid_prediction_cols[0][:-1]:
                    if active_old_col != -1:
                        if active_old_col != (active_col.item() - 1):
                            wrap_around_in_mem = True
                            break
                    active_old_col = active_col.item()
                    active_old_col_idx += 1
                if wrap_around_in_mem:
                    valid_prediction_cols_toComputeLosses =\
                        torch.cat((valid_prediction_cols[:, :-1][:, active_old_col_idx:],
                                   valid_prediction_cols[:, :-1][:, :active_old_col_idx],
                                   valid_prediction_cols[:, -1:]), dim=1)
                else:
                    valid_prediction_cols_toComputeLosses = valid_prediction_cols.clone()
                valid_prediction_idxs_toComputeLosses = (valid_prediction_idxs[0].clone(),
                                                         valid_prediction_cols_toComputeLosses.view(-1).contiguous())

                # current step tensor manipulations for loss computation
                gt_mono_mag_toComputeLosses =\
                    gtMonoMag_thisEpisode[max(0, current_step_count + 1 - (num_past_steps_refinement + 1)):
                                          current_step_count + 1][..., 0::2].clone()[..., 0].unsqueeze(-1).permute(1, 0, 2, 3, 4)
                gt_mono_mag_toComputeLosses =\
                    gt_mono_mag_toComputeLosses.contiguous()\
                        .view(gt_mono_mag_toComputeLosses.size(0) * gt_mono_mag_toComputeLosses.size(1),
                              *gt_mono_mag_toComputeLosses.size()[2:])

                gt_mono_phase_toComputeLosses =\
                    gtMonoMag_thisEpisode[max(0, current_step_count + 1 - (num_past_steps_refinement + 1)):
                                          current_step_count + 1][..., 1::2].clone()[..., 0].unsqueeze(-1).permute(1, 0, 2, 3, 4)
                gt_mono_phase_toComputeLosses =\
                    gt_mono_phase_toComputeLosses.contiguous()\
                        .view(gt_mono_phase_toComputeLosses.size(0) * gt_mono_phase_toComputeLosses.size(1),
                              *gt_mono_phase_toComputeLosses.size()[2:])

                # pred_monoFromMem_toComputeLosses after this step: [B, mem_size (cfg_mem_size + ppo_cfg_num_steps) + 1, 512, 32, 1]
                pred_monoFromMem_toComputeLosses = pred_monoFromMem.permute(1, 0, 2, 3, 4)

                # needed for mem_size == cfg_mem_size + ppo_cfg_num_steps... need to rotate the cols for computing
                # reward if there is a wraparound the memory to avoid bug
                pred_monoFromMem_toComputeLosses = pred_monoFromMem_toComputeLosses[valid_prediction_idxs_toComputeLosses[0],
                                                                                    valid_prediction_idxs_toComputeLosses[1]]

                if current_step_count >= num_past_steps_refinement:
                    if current_step_count == (env_cfg.MAX_EPISODE_STEPS - 1):
                        for stepIdx_pastStepsRefine in range(torch.cat((gt_mono_mag_toComputeLosses,
                                                                        gt_mono_phase_toComputeLosses),
                                                                       dim=-1).size(0)):
                            pred_monoFromMem_computeLosses_thisEpisode.append(
                                pred_monoFromMem_toComputeLosses[stepIdx_pastStepsRefine].unsqueeze(0)
                            )
                            gtMono_computeLosses_thisEpisode.append(torch.cat((gt_mono_mag_toComputeLosses, 
                                                                               gt_mono_phase_toComputeLosses), 
                                                                              dim=-1)[stepIdx_pastStepsRefine].clone().unsqueeze(0))
                    else:
                        pred_monoFromMem_computeLosses_thisEpisode.append(pred_monoFromMem_toComputeLosses[:1])
                        gtMono_computeLosses_thisEpisode.append(
                            torch.cat((gt_mono_mag_toComputeLosses[:1],
                                       gt_mono_phase_toComputeLosses[:1]),
                                      dim=-1).clone()
                        )

                gt_mono_mag = batch["gt_mono_comps"][:, :, :, 0::2].clone()[:, :, :, 0].unsqueeze(-1)
                gt_mono_phase = batch["gt_mono_comps"][:, :, :, 1::2].clone()[:, :, :, 0].unsqueeze(-1)

                _, actions, _, test_recurrent_hidden_states_pol, distribution_probs =\
                    self.actor_critic.act(
                        batch,
                        test_recurrent_hidden_states_pol,
                        not_done_masks,
                        pred_monoFromMem[-1],
                        pred_binSepMasks,
                        deterministic=ppo_cfg.deterministic_eval,
                    )

            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            test_em.insert(pred_mono_toCache,
                           skip_feats,
                           batch["gt_mono_comps"][:, :, :, 0].unsqueeze(-1),
                           not_done_masks)

            done = dones[0]

            mixedAudioMag_thisEpisode.append(batch["mixed_bin_audio_mag"])
            pred_binSepMasks_thisEpisode.append(pred_binSepMasks.detach())
            gtBinComps_thisEpisode.append(batch["gt_bin_comps"].clone())

            # last step
            if current_step_count == (env_cfg.MAX_EPISODE_STEPS - 1):
                assert done
                monoFromMem_losses_thisEpisode = []
                for stepIdx_computeLoss in range(env_cfg.MAX_EPISODE_STEPS):
                    _, monoFromMem_losses =\
                         STFT_L2_distance(mixedAudioMag_thisEpisode[stepIdx_computeLoss],
                                          pred_binSepMasks_thisEpisode[stepIdx_computeLoss],
                                          gtBinComps_thisEpisode[stepIdx_computeLoss],
                                          pred_monoFromMem_computeLosses_thisEpisode[stepIdx_computeLoss],
                                          gtMono_computeLosses_thisEpisode[stepIdx_computeLoss],
                                          )
                    monoFromMem_loss_this_episode += monoFromMem_losses[0][0].item()

                    monoFromMem_losses_thisEpisode.append(monoFromMem_losses[0][0].item())

                mixedAudioMag_thisEpisode = []
                pred_binSepMasks_thisEpisode = []
                gtBinComps_thisEpisode = []

            bin_losses, mono_losses\
                = STFT_L2_distance(batch["mixed_bin_audio_mag"],
                                   pred_binSepMasks.detach(),
                                   batch["gt_bin_comps"].clone(),
                                   pred_mono,
                                   batch["gt_mono_comps"].clone(),
                                   )
            mono_loss_this_episode += mono_losses[0][0].item()
            monoLosses_thisEpisode.append(mono_losses[0][0].item())

            if config.COMPUTE_EVAL_METRICS:
                # works only for 1 process, idx=0 used for infos
                mixedAudioMag_thisEpisode_computeQualMetrics.append(batch["mixed_bin_audio_mag"][0].unsqueeze(0).cpu().numpy())
                mixedAudioPhase_thisEpisode_computeQualMetrics.append(batch["mixed_bin_audio_phase"][0].unsqueeze(0).cpu().numpy())
                pred_mono_thisEpisode_computeQualMetrics.append(pred_mono[0].unsqueeze(0).detach().cpu().numpy())
                lst_gtMonoMag_thisEpisode_computeQualMetrics.append(gt_mono_mag[0].unsqueeze(0).detach().cpu().numpy())
                lst_gtMonoPhase_thisEpisode_computeQualMetrics.append(gt_mono_phase[0].unsqueeze(0).detach().cpu().numpy())

                if done:
                    for stepIdx_computeQualMetric in range(env_cfg.MAX_EPISODE_STEPS):
                        pred_n_gt_spects = \
                            {"mixed_bin_audio_mag": mixedAudioMag_thisEpisode_computeQualMetrics[stepIdx_computeQualMetric],
                             "mixed_bin_audio_phase": mixedAudioPhase_thisEpisode_computeQualMetrics[stepIdx_computeQualMetric],
                             "gt_mono_mag": lst_gtMonoMag_thisEpisode_computeQualMetrics[stepIdx_computeQualMetric],
                             "gt_mono_phase": lst_gtMonoPhase_thisEpisode_computeQualMetrics[stepIdx_computeQualMetric],
                             "pred_mono": pred_mono_thisEpisode_computeQualMetrics[stepIdx_computeQualMetric],
                             "pred_monoFromMem": pred_monoFromMem_computeLosses_thisEpisode[stepIdx_computeQualMetric].detach().cpu().numpy()}

                        if len(config.EVAL_METRICS_TO_COMPUTE) != 0:
                            qual_metrics = compute_waveform_quality(pred_n_gt_spects,
                                                                    config.EVAL_METRICS_TO_COMPUTE,)
                            mono_qualMetric_name2vals = qual_metrics["mono"]
                            monoFromMem_qualMetric_name2vals = qual_metrics["monoFromMem"]

                            for qualMetric_name in mono_qualMetric_name2vals:
                                if (current_episode_count + 1) not in eval_metrics_toDump["mono"][qualMetric_name]:
                                    assert (current_episode_count + 1) not in eval_metrics_toDump["monoFromMem"][qualMetric_name]
                                    eval_metrics_toDump["mono"][qualMetric_name][current_episode_count + 1] = {}
                                    eval_metrics_toDump["monoFromMem"][qualMetric_name][current_episode_count + 1] = {}
                                eval_metrics_toDump["mono"][qualMetric_name][current_episode_count + 1][stepIdx_computeQualMetric + 1] =\
                                    mono_qualMetric_name2vals[qualMetric_name]
                                eval_metrics_toDump["monoFromMem"][qualMetric_name][current_episode_count + 1][stepIdx_computeQualMetric + 1] =\
                                    monoFromMem_qualMetric_name2vals[qualMetric_name]

                        if (current_episode_count + 1) not in eval_metrics_toDump["mono"]["STFT_L2_loss"]:
                            assert (current_episode_count + 1) not in eval_metrics_toDump["monoFromMem"]["STFT_L2_loss"]
                            eval_metrics_toDump["mono"]["STFT_L2_loss"][current_episode_count + 1] = {}
                            eval_metrics_toDump["monoFromMem"]["STFT_L2_loss"][current_episode_count + 1] = {}
                        eval_metrics_toDump["mono"]["STFT_L2_loss"][current_episode_count + 1][stepIdx_computeQualMetric + 1] =\
                            monoLosses_thisEpisode[stepIdx_computeQualMetric]
                        eval_metrics_toDump["monoFromMem"]["STFT_L2_loss"][current_episode_count + 1][stepIdx_computeQualMetric + 1] =\
                            monoFromMem_losses_thisEpisode[stepIdx_computeQualMetric]

                        if "episodeCount_to_sceneIdEpisodeId" not in eval_metrics_toDump:
                            eval_metrics_toDump["episodeCount_to_sceneIdEpisodeId"] =\
                                {current_episode_count + 1: (current_scene, current_episode_id)}
                        else:
                            eval_metrics_toDump["episodeCount_to_sceneIdEpisodeId"][current_episode_count + 1] =\
                                (current_scene, current_episode_id)

                        stepIdx_computeQualMetric += 1

                    mixedAudioMag_thisEpisode_computeQualMetrics = []
                    mixedAudioPhase_thisEpisode_computeQualMetrics = []
                    pred_mono_thisEpisode_computeQualMetrics = []
                    lst_gtMonoMag_thisEpisode_computeQualMetrics = []
                    lst_gtMonoPhase_thisEpisode_computeQualMetrics = []

            # batch being re-assigned here because current batch used in the computation of eval metrics
            batch = batch_obs(observations, self.device)
            step_count_all_processes += 1
            next_episodes = self.envs.current_episodes()
            next_scene = next_episodes[0].scene_id.split('/')[-2]
            next_episode_id = next_episodes[0].episode_id

            # particularly useful for multi-process eval
            for env_idx in active_envs:
                if env_idx > 0:
                    raise NotImplementedError

                # episode has ended
                if not_done_masks[env_idx].item() == 0:
                    test_em.reset(self.device)
                    gtMonoMag_thisEpisode = torch.zeros(
                        env_cfg.MAX_EPISODE_STEPS,
                        config.NUM_PROCESSES,
                        *self.envs.observation_spaces[0].spaces["gt_mono_comps"].shape,
                        ).to(self.device)

                    monoLosses_thisEpisode = []
                    pred_monoFromMem_computeLosses_thisEpisode = []
                    gtMono_computeLosses_thisEpisode = []

                    # stats of simulator-returned performance metrics
                    episode_stats = dict()

                    # use scene + episode_id as unique id for storing stats
                    assert (current_scene, current_episode_id) not in stats_episodes
                    stats_episodes[(current_scene, current_episode_id)] = episode_stats

                    # eval metrics (STFT losses) for logging to log
                    mono_losses_last_step.append(mono_losses[env_idx][0].item())
                    mono_losses_all_steps.append(mono_loss_this_episode / step_count_all_processes[env_idx].item())
                    mono_loss_this_episode = 0.
                    monoFromMem_losses_last_step.append(monoFromMem_losses[env_idx][0].item())
                    monoFromMem_losses_all_steps.append(monoFromMem_loss_this_episode / step_count_all_processes[env_idx].item())
                    monoFromMem_loss_this_episode = 0.

                     # update tqdm object
                    t.update()

                    # particularly useful for multi-process eval
                    if (next_scene, next_episode_id) not in stats_episodes:
                        episode_count_all_processes[env_idx] = num_episode_numbers_taken + 1
                        num_episode_numbers_taken += 1
                        step_count_all_processes[env_idx] = 0

        # closing the open environments after iterating over all episodes
        self.envs.close()

        # mean and std of simulator-returned metrics and STFT L2 losses
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = dict()
            aggregated_stats[stat_key]["mean"] = np.mean(
                [v[stat_key] for v in stats_episodes.values()]
            )
            aggregated_stats[stat_key]["std"] = np.std(
                [v[stat_key] for v in stats_episodes.values()]
            )

        aggregated_stats["mono_loss_last_step"] = dict()
        aggregated_stats["mono_loss_last_step"]["mean"] = np.mean(mono_losses_last_step)
        aggregated_stats["mono_loss_last_step"]["std"] = np.std(mono_losses_last_step)
        aggregated_stats["mono_loss_all_steps"] = dict()
        aggregated_stats["mono_loss_all_steps"]["mean"] = np.mean(mono_losses_all_steps)
        aggregated_stats["mono_loss_all_steps"]["std"] = np.std(mono_losses_all_steps)
        aggregated_stats["monoFromMem_loss_last_step"] = dict()
        aggregated_stats["monoFromMem_loss_last_step"]["mean"] = np.mean(monoFromMem_losses_last_step)
        aggregated_stats["monoFromMem_loss_last_step"]["std"] = np.std(monoFromMem_losses_last_step)
        aggregated_stats["monoFromMem_loss_all_steps"] = dict()
        aggregated_stats["monoFromMem_loss_all_steps"]["mean"] = np.mean(monoFromMem_losses_all_steps)
        aggregated_stats["monoFromMem_loss_all_steps"]["std"] = np.std(monoFromMem_losses_all_steps)

        # dump stats file to disk
        stats_file = os.path.join(config.TENSORBOARD_DIR,
                                  '{}_stats_{}.json'.format(config.EVAL.SPLIT,
                                                            config.SEED))
        new_stats_episodes = {','.join(key): value for key, value in stats_episodes.items()}
        with open(stats_file, 'w') as fo:
            json.dump(new_stats_episodes, fo)

        # dump eval metrics to disk
        if config.COMPUTE_EVAL_METRICS:
            with open(os.path.join(config.MODEL_DIR, "eval_metrics.pkl"), "wb") as fo:
                pickle.dump(eval_metrics_toDump, fo, protocol=pickle.HIGHEST_PROTOCOL)

        # writing metrics to train.log and terminal
        logger.info("Mono STFT L2 loss at last step --- mean: {mean:.6f}, std: {std:.6f}"\
                    .format(mean=aggregated_stats["mono_loss_last_step"]["mean"],
                            std=aggregated_stats["mono_loss_last_step"]["std"]))
        logger.info("Mono STFT L2 loss over all steps --- mean: {mean:.6f}, std: {std:.6f}"\
                    .format(mean=aggregated_stats["mono_loss_all_steps"]["mean"],
                            std=aggregated_stats["mono_loss_all_steps"]["std"]))
        logger.info("MonoFromMem STFT L2 loss at last step --- mean: {mean:.6f}, std: {std:.6f}"\
                    .format(mean=aggregated_stats["monoFromMem_loss_last_step"]["mean"],
                            std=aggregated_stats["monoFromMem_loss_last_step"]["std"]))
        logger.info("MonoFromMem STFT L2 loss over all steps --- mean: {mean:.6f}, std: {std:.6f}"\
                    .format(mean=aggregated_stats["monoFromMem_loss_all_steps"]["mean"],
                            std=aggregated_stats["monoFromMem_loss_all_steps"]["std"]))

        return {}
