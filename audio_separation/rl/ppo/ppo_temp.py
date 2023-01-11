#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as distrib
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import permutations
from audio_separation.common.eval_metrics import get_PIT_order

EPS_PPO = 1e-5
EPS_KL = 1e-7


class PPO(nn.Module):
    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        separation_loss_coef,
        mono_separation_loss_coef,
        entropy_coef,
        lr=None,
        lr_sep=None,
        wd_sep=None,
        lr_sep_downsample_cnn=None,
        lr_sep_pose_enc=None,
        lr_sep_fusion_enc=None,
        lr_sep_target_class_enc=None,
        lr_sep_fixed_mask_enc=None,
        lr_sep_pose_dec=None,
        lr_sep_upsample_cnn=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        use_normalized_advantage=True,
        refine_bin2mono=True,
        freeze_binaural_separator=False,
        freeze_refine_bin2mono=False,
        predict_target_class_mono=True,
        class_conditional=True,
        train_single_source_nav=False,
        use_only_mixed_audio_for_single_source_nav=False,
        use_currentMono_instead_lastMonoFromMem_ours=False,
        use_binaural_instead_mix_ours=False,
        use_first_mono_as_target=False,
        use_self_attn_n_seq_mem=False,
        self_attn_n_seq_mem_cfg=None,
        use_recurrent_mem=False,
        use_recurrent_mem_with_self_attn=False,
        prevent_overfitting_cfg=None,
        use_mixed_mono_as_input=False,
        use_mixed_bin_as_input=False,
        num_steps=20,
    ):

        # if freeze_binaural_separator and ((not refine_bin2mono) or (not detach_mono_separation)):
        #     raise ValueError("refine_bin2mono or detach_mono_separation not set but freeze_binaural_separator is set")

        super().__init__()
        if (not refine_bin2mono) or (not class_conditional) or (not predict_target_class_mono):
            raise NotImplementedError

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.separation_loss_coef = separation_loss_coef
        self.mono_separation_loss_coef = mono_separation_loss_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        nav_params = list(actor_critic.nav_net.parameters()) + list(actor_critic.action_distribution.parameters()) +\
                     list(actor_critic.critic.parameters())
        self.optimizer_nav = optim.Adam(nav_params, lr=lr, eps=eps)

        sep_params = list(actor_critic.sep_enc.parameters()) + list(actor_critic.separator.parameters())
        if refine_bin2mono:
            if hasattr(actor_critic, "sep_enc_refine") and (actor_critic.sep_enc_refine is not None):
                sep_params.extend(list(actor_critic.sep_enc_refine.parameters()))
                sep_params.extend(list(actor_critic.separator_head_refine.parameters()))
        if predict_target_class_mono:
            if hasattr(actor_critic, "mem_head") and (actor_critic.mem_head is not None):
                sep_params.extend(list(actor_critic.mem_head.parameters()))

        if hasattr(actor_critic, "mem_head") and (actor_critic.mem_head is not None) and\
                hasattr(actor_critic.mem_head, "downsample_cnn") and (actor_critic.mem_head.downsample_cnn is not None):
            if hasattr(actor_critic.mem_head, "transformer") and (actor_critic.mem_head.transformer is not None):
                # if hasattr(actor_critic.mem_head, "pose_encoder") and (actor_critic.mem_head.pose_encoder is not None):
                #     if hasattr(actor_critic.mem_head, "fusion_encoder") and (actor_critic.mem_head.fusion_encoder is not None):
                #         if hasattr(actor_critic.mem_head, "pose_decoder") and (actor_critic.mem_head.pose_decoder is not None):
                #             if hasattr(actor_critic.mem_head, "target_class_encoder") and (actor_critic.mem_head.target_class_encoder is not None):
                #                 self.optimizer_sep = optim.Adam(
                #                     [
                #                         {"params": actor_critic.mem_head.downsample_cnn.parameters(), 'lr': lr_sep_downsample_cnn},
                #                         {"params": actor_critic.mem_head.pose_encoder.parameters(), 'lr': lr_sep_pose_enc},
                #                         {"params": actor_critic.mem_head.target_class_encoder.parameters(), 'lr': lr_sep_target_class_enc},
                #                         {"params": actor_critic.mem_head.fusion_encoder.parameters(), 'lr': lr_sep_fusion_enc},
                #                         {"params": actor_critic.mem_head.pose_decoder.parameters(), 'lr': lr_sep_pose_dec},
                #                         {"params": actor_critic.mem_head.transformer.parameters(),},
                #                         {"params": actor_critic.mem_head.upsample_cnn.parameters(), 'lr': lr_sep_upsample_cnn},
                #                     ],
                #                     lr=lr_sep, eps=eps, weight_decay=wd_sep,
                #                 )
                #             else:
                #                 self.optimizer_sep = optim.Adam(
                #                     [
                #                         {"params": actor_critic.mem_head.downsample_cnn.parameters(), 'lr': lr_sep_downsample_cnn},
                #                         {"params": actor_critic.mem_head.pose_encoder.parameters(), 'lr': lr_sep_pose_enc},
                #                         {"params": actor_critic.mem_head.fusion_encoder.parameters(), 'lr': lr_sep_fusion_enc},
                #                         {"params": actor_critic.mem_head.pose_decoder.parameters(), 'lr': lr_sep_pose_dec},
                #                         {"params": actor_critic.mem_head.transformer.parameters(),},
                #                         {"params": actor_critic.mem_head.upsample_cnn.parameters(), 'lr': lr_sep_upsample_cnn},
                #                     ],
                #                     lr=lr_sep, eps=eps, weight_decay=wd_sep,
                #                 )
                #         else:
                #             if hasattr(actor_critic.mem_head, "target_class_encoder") and (actor_critic.mem_head.target_class_encoder is not None):
                #                 if hasattr(actor_critic.mem_head, "fixed_mask_token_encoder") and (actor_critic.mem_head.fixed_mask_token_encoder is not None):
                #                     if hasattr(actor_critic.mem_head, "fixed_mask_skip_features_encoder") and (actor_critic.mem_head.fixed_mask_skip_features_encoder is not None):
                #                         self.optimizer_sep = optim.Adam(
                #                             [
                #                                 {"params": actor_critic.mem_head.downsample_cnn.parameters(), 'lr': lr_sep_downsample_cnn},
                #                                 {"params": actor_critic.mem_head.pose_encoder.parameters(), 'lr': lr_sep_pose_enc},
                #                                 {"params": actor_critic.mem_head.target_class_encoder.parameters(), 'lr': lr_sep_target_class_enc},
                #                                 {"params": actor_critic.mem_head.fixed_mask_token_encoder.parameters(), 'lr': lr_sep_fixed_mask_enc},
                #                                 {"params": actor_critic.mem_head.fixed_mask_skip_features_encoder.parameters(), 'lr': lr_sep_fixed_mask_enc},
                #                                 {"params": actor_critic.mem_head.fusion_encoder.parameters(), 'lr': lr_sep_fusion_enc},
                #                                 {"params": actor_critic.mem_head.transformer.parameters(),},
                #                                 {"params": actor_critic.mem_head.upsample_cnn.parameters(), 'lr': lr_sep_upsample_cnn},
                #                             ],
                #                             lr=lr_sep, eps=eps, weight_decay=wd_sep,
                #                         )
                #                     else:
                #                         self.optimizer_sep = optim.Adam(
                #                             [
                #                                 {"params": actor_critic.mem_head.downsample_cnn.parameters(), 'lr': lr_sep_downsample_cnn},
                #                                 {"params": actor_critic.mem_head.pose_encoder.parameters(), 'lr': lr_sep_pose_enc},
                #                                 {"params": actor_critic.mem_head.target_class_encoder.parameters(), 'lr': lr_sep_target_class_enc},
                #                                 {"params": actor_critic.mem_head.fixed_mask_token_encoder.parameters(), 'lr': lr_sep_fixed_mask_enc},
                #                                 {"params": actor_critic.mem_head.fusion_encoder.parameters(), 'lr': lr_sep_fusion_enc},
                #                                 {"params": actor_critic.mem_head.transformer.parameters(),},
                #                                 {"params": actor_critic.mem_head.upsample_cnn.parameters(), 'lr': lr_sep_upsample_cnn},
                #                             ],
                #                             lr=lr_sep, eps=eps, weight_decay=wd_sep,
                #                         )
                #                 else:
                #                     self.optimizer_sep = optim.Adam(
                #                         [
                #                             {"params": actor_critic.mem_head.downsample_cnn.parameters(), 'lr': lr_sep_downsample_cnn},
                #                             {"params": actor_critic.mem_head.pose_encoder.parameters(), 'lr': lr_sep_pose_enc},
                #                             {"params": actor_critic.mem_head.target_class_encoder.parameters(), 'lr': lr_sep_target_class_enc},
                #                             {"params": actor_critic.mem_head.fusion_encoder.parameters(), 'lr': lr_sep_fusion_enc},
                #                             {"params": actor_critic.mem_head.transformer.parameters(),},
                #                             {"params": actor_critic.mem_head.upsample_cnn.parameters(), 'lr': lr_sep_upsample_cnn},
                #                         ],
                #                         lr=lr_sep, eps=eps, weight_decay=wd_sep,
                #                     )
                #             else:
                #                 self.optimizer_sep = optim.Adam(
                #                     [
                #                         {"params": actor_critic.mem_head.downsample_cnn.parameters(), 'lr': lr_sep_downsample_cnn},
                #                         {"params": actor_critic.mem_head.pose_encoder.parameters(), 'lr': lr_sep_pose_enc},
                #                         {"params": actor_critic.mem_head.fusion_encoder.parameters(), 'lr': lr_sep_fusion_enc},
                #                         {"params": actor_critic.mem_head.transformer.parameters(),},
                #                         {"params": actor_critic.mem_head.upsample_cnn.parameters(), 'lr': lr_sep_upsample_cnn},
                #                     ],
                #                     lr=lr_sep, eps=eps, weight_decay=wd_sep,
                #                 )
                #     else:
                #         self.optimizer_sep = optim.Adam(
                #             [
                #                 {"params": actor_critic.mem_head.downsample_cnn.parameters(), 'lr': lr_sep_downsample_cnn},
                #                 {"params": actor_critic.mem_head.pose_encoder.parameters(), 'lr': lr_sep_pose_enc},
                #                 {"params": actor_critic.mem_head.transformer.parameters(),},
                #                 {"params": actor_critic.mem_head.upsample_cnn.parameters(), 'lr': lr_sep_upsample_cnn},
                #             ],
                #             lr=lr_sep, eps=eps, weight_decay=wd_sep,
                #         )
                # else:
                if use_recurrent_mem_with_self_attn:
                    self.optimizer_sep = optim.Adam(
                        [
                            {"params": actor_critic.mem_head.downsample_cnn.parameters(), 'lr': lr_sep_downsample_cnn},
                            {"params": actor_critic.mem_head.transformer.parameters(),},
                            {"params": actor_critic.mem_head.recurrent_net.parameters(),},
                            {"params": actor_critic.mem_head.upsample_cnn.parameters(), 'lr': lr_sep_upsample_cnn},
                        ],
                        lr=lr_sep, eps=eps, weight_decay=wd_sep,
                    )
                else:
                    if hasattr(actor_critic.mem_head, "fixed_mask_token_encoder") and (actor_critic.mem_head.fixed_mask_token_encoder is not None):
                        if hasattr(actor_critic.mem_head, "fixed_mask_skip_features_encoder") and (actor_critic.mem_head.fixed_mask_skip_features_encoder is not None):
                            if hasattr(actor_critic.mem_head, "fusion_encoder") and (actor_critic.mem_head.fusion_encoder is not None):
                                self.optimizer_sep = optim.Adam(
                                    [
                                        {"params": actor_critic.mem_head.downsample_cnn.parameters(), 'lr': lr_sep_downsample_cnn},
                                        {"params": actor_critic.mem_head.pose_encoder.parameters(), 'lr': lr_sep_pose_enc},
                                        {"params": actor_critic.mem_head.fusion_encoder.parameters(), 'lr': lr_sep_fusion_enc},
                                        {"params": actor_critic.mem_head.fixed_mask_token_encoder.parameters(), 'lr': lr_sep_fixed_mask_enc},
                                        {"params": actor_critic.mem_head.fixed_mask_skip_features_encoder.parameters(), 'lr': lr_sep_fixed_mask_enc},
                                        {"params": actor_critic.mem_head.transformer.parameters(),},
                                        {"params": actor_critic.mem_head.upsample_cnn.parameters(), 'lr': lr_sep_upsample_cnn},
                                    ],
                                    lr=lr_sep, eps=eps, weight_decay=wd_sep,
                                )
                            else:
                                self.optimizer_sep = optim.Adam(
                                    [
                                        {"params": actor_critic.mem_head.downsample_cnn.parameters(), 'lr': lr_sep_downsample_cnn},
                                        {"params": actor_critic.mem_head.fixed_mask_token_encoder.parameters(), 'lr': lr_sep_fixed_mask_enc},
                                        {"params": actor_critic.mem_head.fixed_mask_skip_features_encoder.parameters(), 'lr': lr_sep_fixed_mask_enc},
                                        {"params": actor_critic.mem_head.transformer.parameters(),},
                                        {"params": actor_critic.mem_head.upsample_cnn.parameters(), 'lr': lr_sep_upsample_cnn},
                                    ],
                                    lr=lr_sep, eps=eps, weight_decay=wd_sep,
                                )
                        else:
                            if hasattr(actor_critic.mem_head, "fusion_encoder") and (actor_critic.mem_head.fusion_encoder is not None):
                                self.optimizer_sep = optim.Adam(
                                    [
                                        {"params": actor_critic.mem_head.downsample_cnn.parameters(), 'lr': lr_sep_downsample_cnn},
                                        {"params": actor_critic.mem_head.pose_encoder.parameters(), 'lr': lr_sep_pose_enc},
                                        {"params": actor_critic.mem_head.fusion_encoder.parameters(), 'lr': lr_sep_fusion_enc},
                                        {"params": actor_critic.mem_head.fixed_mask_token_encoder.parameters(), 'lr': lr_sep_fixed_mask_enc},
                                        {"params": actor_critic.mem_head.transformer.parameters(),},
                                        {"params": actor_critic.mem_head.upsample_cnn.parameters(), 'lr': lr_sep_upsample_cnn},
                                    ],
                                    lr=lr_sep, eps=eps, weight_decay=wd_sep,
                                )
                            else:
                                self.optimizer_sep = optim.Adam(
                                    [
                                        {"params": actor_critic.mem_head.downsample_cnn.parameters(), 'lr': lr_sep_downsample_cnn},
                                        {"params": actor_critic.mem_head.fixed_mask_token_encoder.parameters(), 'lr': lr_sep_fixed_mask_enc},
                                        {"params": actor_critic.mem_head.transformer.parameters(),},
                                        {"params": actor_critic.mem_head.upsample_cnn.parameters(), 'lr': lr_sep_upsample_cnn},
                                    ],
                                    lr=lr_sep, eps=eps, weight_decay=wd_sep,
                                )
                    else:
                        if hasattr(actor_critic.mem_head, "fusion_encoder") and (actor_critic.mem_head.fusion_encoder is not None):
                            self.optimizer_sep = optim.Adam(
                                [
                                    {"params": actor_critic.mem_head.downsample_cnn.parameters(), 'lr': lr_sep_downsample_cnn},
                                    {"params": actor_critic.mem_head.pose_encoder.parameters(), 'lr': lr_sep_pose_enc},
                                    {"params": actor_critic.mem_head.fusion_encoder.parameters(), 'lr': lr_sep_fusion_enc},
                                    {"params": actor_critic.mem_head.transformer.parameters(),},
                                    {"params": actor_critic.mem_head.upsample_cnn.parameters(), 'lr': lr_sep_upsample_cnn},
                                ],
                                lr=lr_sep, eps=eps, weight_decay=wd_sep,
                            )
                        else:
                            self.optimizer_sep = optim.Adam(
                                [
                                    {"params": actor_critic.mem_head.downsample_cnn.parameters(), 'lr': lr_sep_downsample_cnn},
                                    {"params": actor_critic.mem_head.transformer.parameters(),},
                                    {"params": actor_critic.mem_head.upsample_cnn.parameters(), 'lr': lr_sep_upsample_cnn},
                                ],
                                lr=lr_sep, eps=eps, weight_decay=wd_sep,
                            )
            else:
                self.optimizer_sep = optim.Adam(sep_params, lr=lr_sep, eps=eps, weight_decay=wd_sep,)
        else:
            self.optimizer_sep = optim.Adam(sep_params, lr=lr_sep, eps=eps, weight_decay=wd_sep,)

        self.device = next(actor_critic.parameters()).device

        self.use_normalized_advantage = use_normalized_advantage

        self.refine_bin2mono = refine_bin2mono
        self.freeze_binaural_separator = freeze_binaural_separator
        self.freeze_refine_bin2mono = freeze_refine_bin2mono
        self.predict_target_class_mono = predict_target_class_mono

        self.class_conditional = class_conditional
        self.train_single_source_nav = train_single_source_nav
        self.use_only_mixed_audio_for_single_source_nav = use_only_mixed_audio_for_single_source_nav
        self.use_currentMono_instead_lastMonoFromMem_ours = use_currentMono_instead_lastMonoFromMem_ours
        self.use_binaural_instead_mix_ours = use_binaural_instead_mix_ours
        self.use_first_mono_as_target = use_first_mono_as_target

        self.use_self_attn_n_seq_mem = use_self_attn_n_seq_mem
        self.self_attn_n_seq_mem_cfg = self_attn_n_seq_mem_cfg
        self.k_old_steps = self_attn_n_seq_mem_cfg.k_old_steps
        self.use_k_monoFromMem_as_pol_inp_with_transformer = use_self_attn_n_seq_mem and self_attn_n_seq_mem_cfg.pred_all_old_steps\
                                                             and self_attn_n_seq_mem_cfg.use_k_monoFromMem_as_pol_inp_with_transformer
        self.use_recurrent_mem = use_recurrent_mem
        self.use_recurrent_mem_with_self_attn = use_recurrent_mem_with_self_attn
        self.prevent_overfitting_cfg = prevent_overfitting_cfg

        self.use_mixed_mono_as_input = use_mixed_mono_as_input
        self.use_mixed_bin_as_input = use_mixed_bin_as_input

        self.num_steps = num_steps

    def load_state_dict_for_single_source_nav(self, state_dict):
        assert hasattr(self.actor_critic, "sep_enc") and (self.actor_critic.sep_enc is not None)
        for name in self.actor_critic.sep_enc.state_dict():
            self.actor_critic.sep_enc.state_dict()[name].copy_(state_dict["actor_critic.sep_enc." + name])

        assert hasattr(self.actor_critic, "separator") and (self.actor_critic.separator is not None)
        for name in self.actor_critic.separator.state_dict():
            self.actor_critic.separator.state_dict()[name].copy_(state_dict["actor_critic.separator." + name])

        assert hasattr(self.actor_critic, "sep_enc_refine") and (self.actor_critic.sep_enc_refine is not None)
        for name in self.actor_critic.sep_enc_refine.state_dict():
            self.actor_critic.sep_enc_refine.state_dict()[name].copy_(state_dict["actor_critic.sep_enc_refine." + name])

        assert hasattr(self.actor_critic, "separator_head_refine") and (self.actor_critic.separator_head_refine is not None)
        for name in self.actor_critic.separator_head_refine.stateself_attn_n_seq_mem_cfg_dict():
            self.actor_critic.separator_head_refine.state_dict()[name].copy_(state_dict["actor_critic.separator_head_refine." + name])

        assert hasattr(self.actor_critic, "mem_head") and (self.actor_critic.mem_head is not None)
        for name in self.actor_critic.mem_head.state_dict():
            self.actor_critic.mem_head.state_dict()[name].copy_(state_dict["actor_critic.mem_head." + name])

    def load_state_dict_just_for_policy(self, state_dict):
        assert hasattr(self.actor_critic, "sep_enc") and (self.actor_critic.sep_enc is not None)
        for name in self.actor_critic.sep_enc.state_dict():
            self.actor_critic.sep_enc.state_dict()[name].copy_(state_dict["actor_critic.sep_enc." + name])

        assert hasattr(self.actor_critic, "separator") and (self.actor_critic.separator is not None)
        for name in self.actor_critic.separator.state_dict():
            self.actor_critic.separator.state_dict()[name].copy_(state_dict["actor_critic.separator." + name])

        assert hasattr(self.actor_critic, "sep_enc_refine") and (self.actor_critic.sep_enc_refine is not None)
        for name in self.actor_critic.sep_enc_refine.state_dict():
            self.actor_critic.sep_enc_refine.state_dict()[name].copy_(state_dict["actor_critic.sep_enc_refine." + name])

        assert hasattr(self.actor_critic, "separator_head_refine") and (self.actor_critic.separator_head_refine is not None)
        for name in self.actor_critic.separator_head_refine.state_dict():
            self.actor_critic.separator_head_refine.state_dict()[name].copy_(state_dict["actor_critic.separator_head_refine." + name])

        assert hasattr(self.actor_critic, "nav_net") and (self.actor_critic.nav_net is not None)
        for name in self.actor_critic.nav_net.state_dict():
            self.actor_critic.nav_net.state_dict()[name].copy_(state_dict["actor_critic.nav_net." + name])

        assert hasattr(self.actor_critic, "action_distribution") and (self.actor_critic.action_distribution is not None)
        for name in self.actor_critic.action_distribution.state_dict():
            self.actor_critic.action_distribution.state_dict()[name].copy_(state_dict["actor_critic.action_distribution." + name])

        assert hasattr(self.actor_critic, "critic") and (self.actor_critic.critic is not None)
        for name in self.actor_critic.critic.state_dict():
            self.actor_critic.critic.state_dict()[name].copy_(state_dict["actor_critic.critic." + name])

    def load_separator_dict(self, state_dict, load_refine_cnn=True, load_mem=False, load_transformer_feature_encoderNdecoder=False,
                            load_transformer_weights_from_pretraining=True, load_upsampler_weights_from_pretraining=True,):
        assert hasattr(self.actor_critic, "sep_enc") and (self.actor_critic.sep_enc is not None)
        for name in self.actor_critic.sep_enc.state_dict():
            self.actor_critic.sep_enc.state_dict()[name].copy_(state_dict["actor_critic.sep_enc." + name])

        assert hasattr(self.actor_critic, "separator") and (self.actor_critic.separator is not None)
        for name in self.actor_critic.separator.state_dict():
            self.actor_critic.separator.state_dict()[name].copy_(state_dict["actor_critic.separator." + name])

        if load_refine_cnn:
            assert hasattr(self.actor_critic, "sep_enc_refine") and (self.actor_critic.sep_enc_refine is not None)
            for name in self.actor_critic.sep_enc_refine.state_dict():
                self.actor_critic.sep_enc_refine.state_dict()[name].copy_(state_dict["actor_critic.sep_enc_refine." + name])

            assert hasattr(self.actor_critic, "separator_head_refine") and (self.actor_critic.separator_head_refine is not None)
            for name in self.actor_critic.separator_head_refine.state_dict():
                self.actor_critic.separator_head_refine.state_dict()[name].copy_(state_dict["actor_critic.separator_head_refine." + name])

        if load_mem or load_transformer_feature_encoderNdecoder:
            assert hasattr(self.actor_critic, "mem_head") and (self.actor_critic.mem_head is not None)
            for name in self.actor_critic.mem_head.state_dict():
                if load_transformer_weights_from_pretraining:
                    if load_upsampler_weights_from_pretraining or ((not load_upsampler_weights_from_pretraining)\
                                                                   and (name.split(".")[0] != "upsample_cnn")):
                        self.actor_critic.mem_head.state_dict()[name].copy_(state_dict["actor_critic.mem_head." + name])
                else:
                    if load_upsampler_weights_from_pretraining:
                        str_for_loading_pretrained_models = ["upsample_cnn", "downsample_cnn"]
                    else:
                        str_for_loading_pretrained_models = ["downsample_cnn"]
                    if name.split(".")[0] in str_for_loading_pretrained_models:
                        self.actor_critic.mem_head.state_dict()[name].copy_(state_dict["actor_critic.mem_head." + name])

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts_nav):
        advantages = rollouts_nav.returns[:-1] - rollouts_nav.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update_nav(self, rollouts_nav):
        advantages = self.get_advantages(rollouts_nav)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts_nav.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                if (self.train_single_source_nav and not self.use_only_mixed_audio_for_single_source_nav) or (self.use_currentMono_instead_lastMonoFromMem_ours and self.use_binaural_instead_mix_ours):
                    if self.use_self_attn_n_seq_mem:
                        if self.use_k_monoFromMem_as_pol_inp_with_transformer:
                            (
                                obs_batch,
                                recurrent_hidden_states_nav_batch,
                                actions_batch,
                                prev_actions_batch,
                                value_preds_batch,
                                return_batch,
                                masks_batch,
                                old_action_log_probs_batch,
                                adv_targ,
                                predicted_monoFromMem_batch,
                                predicted_mono_batch,
                                predicted_separation_masks_batch,
                                pol_em_store_batch,
                                pol_em_masks_batch,
                            ) = sample
                        else:
                            if self.self_attn_n_seq_mem_cfg.pred_all_old_steps and (self.self_attn_n_seq_mem_cfg.k_old_steps != -1):
                                (
                                    obs_batch,
                                    recurrent_hidden_states_nav_batch,
                                    actions_batch,
                                    prev_actions_batch,
                                    value_preds_batch,
                                    return_batch,
                                    masks_batch,
                                    old_action_log_probs_batch,
                                    adv_targ,
                                    predicted_monoFromMem_batch,
                                    predicted_mono_batch,
                                    predicted_separation_masks_batch,
                                    valid_prediction_rows_batch,
                                    valid_prediction_cols_batch,
                                    external_memory_masks_for_predAllOldSteps_allStepsThisEpMarked_batch,
                                ) = sample
                                # print(predicted_monoFromMem_batch.size(), valid_prediction_cols_batch.size())
                                # print(valid_prediction_cols_batch[:10], valid_prediction_rows_batch[:10])
                                # print(predicted_monoFromMem_batch[:10, :, :, :2])
                                # exit("recurrent generator working")
                            else:
                                (
                                    obs_batch,
                                    recurrent_hidden_states_nav_batch,
                                    actions_batch,
                                    prev_actions_batch,
                                    value_preds_batch,
                                    return_batch,
                                    masks_batch,
                                    old_action_log_probs_batch,
                                    adv_targ,
                                    predicted_monoFromMem_batch,
                                    predicted_mono_batch,
                                    predicted_separation_masks_batch,
                                ) = sample
                    else:
                        (
                            obs_batch,
                            recurrent_hidden_states_nav_batch,
                            actions_batch,
                            prev_actions_batch,
                            value_preds_batch,
                            return_batch,
                            masks_batch,
                            old_action_log_probs_batch,
                            adv_targ,
                            predicted_monoFromMem_batch,
                            prev_predicted_monoFromMem_batch,
                            predicted_mono_batch,
                            predicted_separation_masks_batch,
                        ) = sample
                elif self.use_currentMono_instead_lastMonoFromMem_ours and (not self.use_binaural_instead_mix_ours):
                    if self.use_self_attn_n_seq_mem:
                        (
                            obs_batch,
                            recurrent_hidden_states_nav_batch,
                            actions_batch,
                            prev_actions_batch,
                            value_preds_batch,
                            return_batch,
                            masks_batch,
                            old_action_log_probs_batch,
                            adv_targ,
                            predicted_monoFromMem_batch,
                            predicted_mono_batch,
                        ) = sample
                    else:
                        (
                            obs_batch,
                            recurrent_hidden_states_nav_batch,
                            actions_batch,
                            prev_actions_batch,
                            value_preds_batch,
                            return_batch,
                            masks_batch,
                            old_action_log_probs_batch,
                            adv_targ,
                            predicted_monoFromMem_batch,
                            prev_predicted_monoFromMem_batch,
                            predicted_mono_batch,
                        ) = sample
                elif self.use_binaural_instead_mix_ours and (not self.use_currentMono_instead_lastMonoFromMem_ours):
                    if self.use_self_attn_n_seq_mem:
                        (
                            obs_batch,
                            recurrent_hidden_states_nav_batch,
                            actions_batch,
                            prev_actions_batch,
                            value_preds_batch,
                            return_batch,
                            masks_batch,
                            old_action_log_probs_batch,
                            adv_targ,
                            predicted_monoFromMem_batch,
                            predicted_separation_masks_batch,
                        ) = sample
                    else:
                        (
                            obs_batch,
                            recurrent_hidden_states_nav_batch,
                            actions_batch,
                            prev_actions_batch,
                            value_preds_batch,
                            return_batch,
                            masks_batch,
                            old_action_log_probs_batch,
                            adv_targ,
                            predicted_monoFromMem_batch,
                            prev_predicted_monoFromMem_batch,
                            predicted_separation_masks_batch,
                        ) = sample
                elif self.predict_target_class_mono:
                    if self.use_self_attn_n_seq_mem:
                        (
                            obs_batch,
                            recurrent_hidden_states_nav_batch,
                            actions_batch,
                            prev_actions_batch,
                            value_preds_batch,
                            return_batch,
                            masks_batch,
                            old_action_log_probs_batch,
                            adv_targ,
                            predicted_monoFromMem_batch,
                        ) = sample
                    else:
                        (
                            obs_batch,
                            recurrent_hidden_states_nav_batch,
                            actions_batch,
                            prev_actions_batch,
                            value_preds_batch,
                            return_batch,
                            masks_batch,
                            old_action_log_probs_batch,
                            adv_targ,
                            predicted_monoFromMem_batch,
                            prev_predicted_monoFromMem_batch,
                        ) = sample
                else:
                    (
                        obs_batch,
                        recurrent_hidden_states_nav_batch,
                        actions_batch,
                        prev_actions_batch,
                        value_preds_batch,
                        return_batch,
                        masks_batch,
                        old_action_log_probs_batch,
                        adv_targ,
                    ) = sample

                # Reshape to do in a single forward pass for all steps
                if self.predict_target_class_mono:
                    if self.use_currentMono_instead_lastMonoFromMem_ours:
                        mono2 = predicted_mono_batch
                    else:
                        if self.use_self_attn_n_seq_mem:
                            raise ValueError("don't use last monoFromMem for self-attention")
                        mono2 = prev_predicted_monoFromMem_batch

                if self.use_k_monoFromMem_as_pol_inp_with_transformer:
                    (
                        values,
                        action_log_probs,
                        dist_entropy,
                        _,
                    ) = self.actor_critic.evaluate_actions(
                        obs_batch,
                        recurrent_hidden_states_nav_batch,
                        prev_actions_batch,
                        masks_batch,
                        actions_batch,
                        mono1=predicted_monoFromMem_batch if self.predict_target_class_mono else None,
                        mono2=mono2 if self.predict_target_class_mono else None,
                        predicted_mono=predicted_mono_batch if (self.train_single_source_nav and not self.use_only_mixed_audio_for_single_source_nav) else None,
                        predicted_separation_masks=predicted_separation_masks_batch if\
                            ((self.train_single_source_nav and not self.use_only_mixed_audio_for_single_source_nav) or self.use_binaural_instead_mix_ours) else None,
                        valid_prediction_idxs=None,
                        # valid_prediction_idxs_allStepsThisEpMarked=\
                        #     torch.where(external_memory_masks_for_predAllOldSteps_allStepsThisEpMarked_batch == 1.0)\
                        #         if (self.self_attn_n_seq_mem_cfg.pred_all_old_steps and (self.self_attn_n_seq_mem_cfg.k_old_steps != -1))\
                        #         else None,
                        external_memory_masks_for_predAllOldSteps_allStepsThisEpMarked=None,
                        pred_k_old_steps=self.self_attn_n_seq_mem_cfg.pred_all_old_steps and\
                                         (self.self_attn_n_seq_mem_cfg.k_old_steps != -1),
                        pol_external_memory=pol_em_store_batch.contiguous(),
                        pol_external_memory_masks=pol_em_masks_batch,
                    )
                else:
                    (
                        values,
                        action_log_probs,
                        dist_entropy,
                        _,
                    ) = self.actor_critic.evaluate_actions(
                        obs_batch,
                        recurrent_hidden_states_nav_batch,
                        prev_actions_batch,
                        masks_batch,
                        actions_batch,
                        mono1=predicted_monoFromMem_batch if self.predict_target_class_mono else None,
                        mono2=mono2 if self.predict_target_class_mono else None,
                        predicted_mono=predicted_mono_batch if (self.train_single_source_nav and not self.use_only_mixed_audio_for_single_source_nav) else None,
                        predicted_separation_masks=predicted_separation_masks_batch if\
                            ((self.train_single_source_nav and not self.use_only_mixed_audio_for_single_source_nav) or self.use_binaural_instead_mix_ours) else None,
                        valid_prediction_idxs=(valid_prediction_rows_batch, valid_prediction_cols_batch) if\
                            (self.self_attn_n_seq_mem_cfg.pred_all_old_steps and (self.self_attn_n_seq_mem_cfg.k_old_steps != -1)) else None,
                        # valid_prediction_idxs_allStepsThisEpMarked=\
                        #     torch.where(external_memory_masks_for_predAllOldSteps_allStepsThisEpMarked_batch == 1.0)\
                        #         if (self.self_attn_n_seq_mem_cfg.pred_all_old_steps and (self.self_attn_n_seq_mem_cfg.k_old_steps != -1))\
                        #         else None,
                        external_memory_masks_for_predAllOldSteps_allStepsThisEpMarked=\
                            external_memory_masks_for_predAllOldSteps_allStepsThisEpMarked_batch\
                                if (self.self_attn_n_seq_mem_cfg.pred_all_old_steps and (self.self_attn_n_seq_mem_cfg.k_old_steps != -1))\
                                else None,
                        pred_k_old_steps=self.self_attn_n_seq_mem_cfg.pred_all_old_steps and\
                                         (self.self_attn_n_seq_mem_cfg.k_old_steps != -1),
                    )

                # compute audio separation loss
                # using new predictions because exact supervision
                # TODO: should update be done every few steps i.e. is batch size = NUM_PROCESSES too small?
                # TODO: separate optimizer and lr scheduler?
                ratio = torch.exp(
                    action_log_probs - old_action_log_probs_batch
                )
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch
                    ).pow(2)
                    value_loss = (
                        0.5
                        * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer_nav.zero_grad()
                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                )

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step_nav()
                self.optimizer_nav.step()
                self.after_step()

                action_loss_epoch += action_loss.item()
                value_loss_epoch += value_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        action_loss_epoch /= num_updates
        value_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def update_sep(self, rollouts_sep, pred_all_old_steps=False, k_old_steps=-1):
        separation_loss_epoch = 0
        if self.refine_bin2mono:
            separation_loss_mono_epoch = 0
        if self.predict_target_class_mono:
            separation_loss_monoFromMem_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts_sep.recurrent_generator(self.num_mini_batch)

            for sample in data_generator:
                if self.predict_target_class_mono:
                    if self.use_self_attn_n_seq_mem:
                        if pred_all_old_steps:
                            if self.prevent_overfitting_cfg.cache_trns_enc_dec_skip_features_mem:
                                if self.use_recurrent_mem_with_self_attn:
                                    (
                                        obs_batch,
                                        actions_batch,
                                        prev_actions_batch,
                                        masks_batch,
                                        external_memory,
                                        external_memory_trns_enc_dec_skip_features,
                                        external_memory_masks,
                                        all_old_steps_target_batch,
                                        recurrent_hidden_states_sep_batch,
                                    ) = sample
                                else:
                                    (
                                        obs_batch,
                                        actions_batch,
                                        prev_actions_batch,
                                        masks_batch,
                                        external_memory,
                                        external_memory_trns_enc_dec_skip_features,
                                        external_memory_masks,
                                        all_old_steps_target_batch,
                                    ) = sample
                            else:
                                if self.use_recurrent_mem_with_self_attn:
                                    (
                                        obs_batch,
                                        actions_batch,
                                        prev_actions_batch,
                                        masks_batch,
                                        external_memory,
                                        external_memory_masks,
                                        all_old_steps_target_batch,
                                        recurrent_hidden_states_sep_batch,
                                    ) = sample
                                else:
                                    (
                                        obs_batch,
                                        actions_batch,
                                        prev_actions_batch,
                                        masks_batch,
                                        external_memory,
                                        external_memory_masks,
                                        all_old_steps_target_batch,
                                    ) = sample
                        else:
                            # assert not self.prevent_overfitting_cfg.cache_trns_enc_dec_skip_features_mem,\
                            #     "not implemented; skip features caching in memory only needed for pred_all_steps"
                            if self.prevent_overfitting_cfg.cache_trns_enc_dec_skip_features_mem:
                                if self.use_recurrent_mem_with_self_attn:
                                    (
                                        obs_batch,
                                        actions_batch,
                                        prev_actions_batch,
                                        masks_batch,
                                        external_memory,
                                        external_memory_trns_enc_dec_skip_features,
                                        external_memory_masks,
                                        recurrent_hidden_states_sep_batch,
                                    ) = sample
                                else:
                                    (
                                        obs_batch,
                                        actions_batch,
                                        prev_actions_batch,
                                        masks_batch,
                                        external_memory,
                                        external_memory_trns_enc_dec_skip_features,
                                        external_memory_masks,
                                    ) = sample
                            else:
                                if self.use_recurrent_mem_with_self_attn:
                                    (
                                        obs_batch,
                                        actions_batch,
                                        prev_actions_batch,
                                        masks_batch,
                                        external_memory,
                                        external_memory_masks,
                                        recurrent_hidden_states_sep_batch,
                                    ) = sample
                                else:
                                    (
                                        obs_batch,
                                        actions_batch,
                                        prev_actions_batch,
                                        masks_batch,
                                        external_memory,
                                        external_memory_masks,
                                    ) = sample
                    elif self.use_recurrent_mem:
                        (
                            obs_batch,
                            actions_batch,
                            prev_actions_batch,
                            masks_batch,
                            recurrent_hidden_states_sep_batch,
                        ) = sample
                    else:
                        (
                            obs_batch,
                            actions_batch,
                            prev_actions_batch,
                            masks_batch,
                            predicted_monoFromMem_batch,
                            prev_predicted_monoFromMem_batch,

                        ) = sample

                # Reshape to do in a single forward pass for all steps
                if not self.freeze_binaural_separator:
                    separation_masks =\
                        self.actor_critic.get_separation_masks(
                            obs_batch,
                            prev_actions_batch,
                            masks_batch,
                        )
                else:
                    with torch.no_grad():
                        separation_masks =\
                            self.actor_critic.get_separation_masks(
                                obs_batch,
                                prev_actions_batch,
                                masks_batch,
                            )

                if not self.freeze_refine_bin2mono:
                    if self.refine_bin2mono:
                        predicted_monos =\
                            self.actor_critic.get_monoFromBin(separation_masks.detach(), obs_batch["audiogoal"],
                                                              prev_actions=prev_actions_batch,
                                                              masks=masks_batch)
                else:
                    with torch.no_grad():
                        if self.refine_bin2mono:
                            predicted_monos =\
                                self.actor_critic.get_monoFromBin(separation_masks.detach(), obs_batch["audiogoal"],
                                                                  prev_actions=prev_actions_batch,
                                                                  masks=masks_batch)

                if self.predict_target_class_mono:
                    if self.class_conditional:
                        if self.use_mixed_bin_as_input:
                            assert not self.use_mixed_mono_as_input
                            target_class_predicted_mono = obs_batch["audiogoal"]
                        elif self.use_mixed_mono_as_input:
                            assert not self.use_mixed_bin_as_input
                            target_class_predicted_mono = obs_batch["mono_audiogoal"]
                        else:
                            target_class_predicted_mono = predicted_monos

                    if self.use_self_attn_n_seq_mem:
                        if pred_all_old_steps:
                            if self.prevent_overfitting_cfg.cache_trns_enc_dec_skip_features_mem:
                                predicted_monoFromMem, _, __, ___, ____ =\
                                    self.actor_critic.get_monoFromMem(target_class_predicted_mono, external_memory=external_memory,
                                                                      external_memory_masks=external_memory_masks,
                                                                      pred_all_old_steps=True, k_old_steps=k_old_steps,
                                                                      pose_inp=obs_batch["pose"], target_class_inp=obs_batch["target_class"],
                                                                      update_sep=True, external_memory_trns_enc_dec_skip_features=external_memory_trns_enc_dec_skip_features,
                                                                      update_sep_for_feature_masking=True,
                                                                      rnn_hidden_states_sep=recurrent_hidden_states_sep_batch\
                                                                          if self.use_recurrent_mem_with_self_attn else None,
                                                                      masks=masks_batch,
                                                                      target_class=obs_batch["target_class"],
                                                                      )
                            else:
                                predicted_monoFromMem, _, __, ___ =\
                                        self.actor_critic.get_monoFromMem(target_class_predicted_mono, external_memory=external_memory,
                                                                          external_memory_masks=external_memory_masks,
                                                                          pred_all_old_steps=True, k_old_steps=k_old_steps,
                                                                          pose_inp=obs_batch["pose"], target_class_inp=obs_batch["target_class"],
                                                                          update_sep=True, update_sep_for_feature_masking=True,
                                                                          rnn_hidden_states_sep=recurrent_hidden_states_sep_batch\
                                                                          if self.use_recurrent_mem_with_self_attn else None,
                                                                          masks=masks_batch,
                                                                          target_class=obs_batch["target_class"],
                                                                          )
                        else:
                            if self.prevent_overfitting_cfg.cache_trns_enc_dec_skip_features_mem:
                                predicted_monoFromMem, _, __, ___ =\
                                    self.actor_critic.get_monoFromMem(target_class_predicted_mono, external_memory=external_memory,
                                                                      external_memory_masks=external_memory_masks,
                                                                      pred_all_old_steps=False, pose_inp=obs_batch["pose"],
                                                                      target_class_inp=obs_batch["target_class"],
                                                                      external_memory_trns_enc_dec_skip_features=external_memory_trns_enc_dec_skip_features,
                                                                      update_sep_for_feature_masking=True,
                                                                      rnn_hidden_states_sep=recurrent_hidden_states_sep_batch\
                                                                          if self.use_recurrent_mem_with_self_attn else None,
                                                                      masks=masks_batch,
                                                                      target_class=obs_batch["target_class"],
                                                                      )
                            else:
                                predicted_monoFromMem, _, __ =\
                                    self.actor_critic.get_monoFromMem(target_class_predicted_mono, external_memory=external_memory,
                                                                      external_memory_masks=external_memory_masks,
                                                                      pred_all_old_steps=False, pose_inp=obs_batch["pose"],
                                                                      target_class_inp=obs_batch["target_class"],
                                                                      update_sep_for_feature_masking=True,
                                                                      rnn_hidden_states_sep=recurrent_hidden_states_sep_batch\
                                                                          if self.use_recurrent_mem_with_self_attn else None,
                                                                      masks=masks_batch,
                                                                      target_class=obs_batch["target_class"],
                                                                      )
                    elif self.use_recurrent_mem:
                        predicted_monoFromMem, _ = self.actor_critic.get_monoFromMem(target_class_predicted_mono,
                                                                                     rnn_hidden_states_sep=recurrent_hidden_states_sep_batch,
                                                                                     masks=masks_batch,
                                                                                     target_class=obs_batch["target_class"],
                                                                                     )
                    else:
                        prev_predicted_monoFromMem_masked = prev_predicted_monoFromMem_batch *\
                                                            masks_batch.unsqueeze(1).unsqueeze(2).repeat(1,
                                                                                                         prev_predicted_monoFromMem_batch.size(1),
                                                                                                         prev_predicted_monoFromMem_batch.size(2),
                                                                                                         1)
                        predicted_monoFromMem =\
                            self.actor_critic.get_monoFromMem(target_class_predicted_mono, prev_predicted_monoFromMem_masked)

                    if self.use_first_mono_as_target:
                        if pred_all_old_steps:
                            """old code, doesn't work with mem_size (trns_cfg_mem_size + num_steps)"""
                            # target_class_gt_mono =\
                            #     all_old_steps_target_batch["audiocomps_mono"][..., 0::2].clone()[..., 0].unsqueeze(-1)
                            """new code, should work with mem_size (trns_cfg_mem_size + num_steps)"""
                            target_class_gt_mono = all_old_steps_target_batch
                            target_class_gt_mono =\
                                torch.cat((target_class_gt_mono,
                                           obs_batch["audiocomps_mono"][:, :, :, 0::2].clone()[:, :, :, 0].unsqueeze(-1).unsqueeze(1)),
                                          dim=1)
                        else:
                            target_class_gt_mono =\
                                obs_batch["audiocomps_mono"][:, :, :, 0::2].clone()[:, :, :, 0].unsqueeze(-1)

                    # TODO: remove later, not required here
                    # target_class_gt_mono_phase =\
                    #     obs_batch["audiocomps_mono"][:, :, :, 1::2]\
                    #         .clone()[list(range(obs_batch["target_class"].size(0))), :, :, obs_batch["target_class"].squeeze(1).cpu().numpy().tolist()].unsqueeze(-1)

                    if pred_all_old_steps:
                        external_memory_masks = torch.cat([external_memory_masks,
                                                           torch.ones([external_memory_masks.shape[0], 1],
                                                                      device=external_memory_masks.device)],
                                                          dim=1)
                        if self.k_old_steps != -1:
                            external_memory_masks = self.get_external_memory_masks_for_pred_k_old(external_memory_masks)

                        external_memory_masks = external_memory_masks.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, *target_class_gt_mono.size()[2:])

                        predicted_monoFromMem = (predicted_monoFromMem.permute(1, 0, 2, 3, 4) * external_memory_masks).reshape(external_memory_masks.size(0) * external_memory_masks.size(1), *external_memory_masks.size()[2:])
                        target_class_gt_mono = (target_class_gt_mono * external_memory_masks).reshape(external_memory_masks.size(0) * external_memory_masks.size(1), *external_memory_masks.size()[2:])

                    separation_loss_monoFromMem = F.l1_loss(predicted_monoFromMem, target_class_gt_mono)

                # compute audio separation loss
                # using new predictions because exact supervision
                # TODO: should update be done every few steps i.e. is batch size = NUM_PROCESSES too small?
                # TODO: separate optimizer and lr scheduler?
                retrieved_specs = separation_masks
                gt_individual_specs = obs_batch["audiocomps"][:, :, :, 0::2]
                if self.refine_bin2mono:
                    retrieved_specs_mono = predicted_monos
                    if self.class_conditional:
                        if self.use_first_mono_as_target:
                            gt_individual_specs_mono =\
                                obs_batch["audiocomps_mono"][:, :, :, 0::2]\
                                    .clone()[:, :, :, 0].unsqueeze(-1)
                    separation_loss_mono = F.l1_loss(retrieved_specs_mono, gt_individual_specs_mono)

                num_sounds = gt_individual_specs.size(3) // 2
                # mixed spec preprocessed using log(mixed_spec + 1) inside habitat_audio/simulator.py
                mixed_spec = torch.exp(obs_batch["audiogoal"]) - 1
                mixed_spec = mixed_spec.repeat([1, 1, 1, num_sounds]).detach()
                retrieved_specs = retrieved_specs * mixed_spec[:, :, :, 0:2]

                if self.class_conditional:
                    if self.use_first_mono_as_target:
                        gt_individual_specs_left =\
                            gt_individual_specs[:, :, :, 0::2].clone()[:, :, :, 0].unsqueeze(-1)
                        gt_individual_specs_right =\
                            gt_individual_specs[:, :, :, 1::2].clone()[:, :, :, 0].unsqueeze(-1)
                    gt_individual_specs = torch.cat((gt_individual_specs_left, gt_individual_specs_right), dim=-1)
                separation_loss = F.l1_loss(retrieved_specs, gt_individual_specs)

                self.optimizer_sep.zero_grad()
                total_loss = 0.
                if not self.freeze_binaural_separator:
                    if self.refine_bin2mono:
                        total_loss = self.separation_loss_coef * separation_loss + self.mono_separation_loss_coef\
                                     * separation_loss_mono
                    if self.predict_target_class_mono:
                        total_loss += separation_loss_monoFromMem
                else:
                    if self.predict_target_class_mono:
                        total_loss = separation_loss_monoFromMem

                # total_loss didn't get assigned because it didn't meet any of the above conditions
                if not (total_loss == 0 and isinstance(total_loss, float)):
                    self.before_backward(total_loss)
                    total_loss.backward()
                    self.after_backward(total_loss)

                    self.before_step_sep()
                    self.optimizer_sep.step()
                    self.after_step()

                separation_loss_epoch += separation_loss.item()
                if self.refine_bin2mono:
                    separation_loss_mono_epoch += separation_loss_mono.item()
                if self.predict_target_class_mono:
                    separation_loss_monoFromMem_epoch += separation_loss_monoFromMem.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        separation_loss_epoch /= num_updates
        if self.refine_bin2mono:
            separation_loss_mono_epoch /= num_updates
        if self.predict_target_class_mono:
            separation_loss_monoFromMem_epoch /= num_updates

        return separation_loss_epoch, separation_loss_mono_epoch, separation_loss_monoFromMem_epoch

    def get_external_memory_masks_for_pred_k_old(self, external_memory_masks,):
        """
        t_masks: [B, mem_size (cfg_mem_size + ppo_cfg_num_steps) + 1];
        """
        bs, M = external_memory_masks.size()

        # print(bs, M)
        """wrong code"""
        # num_processes = bs // (M - 1)
        """right code"""
        num_processes = bs // self.num_steps
        k_old_external_memory_masks = None
        """wrong code"""
        # for mem_idx in range(M - 1):
        """right code"""
        for mem_idx in range(self.num_steps):
            newest_external_memory_masks = external_memory_masks[mem_idx * num_processes: (mem_idx + 1) * num_processes, -1:].clone()
            old_external_memory_masks = external_memory_masks[mem_idx * num_processes: (mem_idx + 1) * num_processes:, :-1].clone()
            active_old_external_memory_masks_idxs = torch.where(old_external_memory_masks == 1)
            old_external_memory_masks[:, :] = 0
            bs_ppo_update = old_external_memory_masks.size(0)
            if active_old_external_memory_masks_idxs[0].size()[0] != 0:
            # if (active_old_mem_idxs[0].size()[0] != 0) and (k_old_steps != 0):
                if self.k_old_steps != 0:
                    active_old_external_memory_masks_cols_firstProc = active_old_external_memory_masks_idxs[1].view(bs_ppo_update, -1)[0]
                    """hacky and slow way to find out which idxs to choose for k_old_steps"""
                    active_old_col = -1
                    wrap_around_in_mem = False
                    for active_col in active_old_external_memory_masks_cols_firstProc:
                        if active_old_col != -1:
                            if active_old_col != (active_col.item() - 1):
                                wrap_around_in_mem = True
                                break
                        active_old_col = active_col.item()
                    if wrap_around_in_mem:
                        last_active_col_left = active_old_col
                        first_active_col_left = 0
                        last_active_col_right = old_external_memory_masks.size(1) - 1
                        """remove later.. not needed"""
                        # first_active_col_right = last_active_col_right -\
                        #                          (active_old_t_masks_cols_firstProc.size(0) -\
                        #                           (last_active_col_left - first_active_col_left + 1)) + 1

                        if last_active_col_left - first_active_col_left + 1 >= self.k_old_steps:
                            active_old_external_memory_masks_cols_oneProcLst = list(range(last_active_col_left - self.k_old_steps + 1,
                                                                            last_active_col_left + 1))
                            active_old_external_memory_masks_cols =\
                                torch.LongTensor(active_old_external_memory_masks_cols_oneProcLst).unsqueeze(0).repeat(bs_ppo_update, 1).contiguous().view(-1)

                            active_old_external_memory_masks_rows_oneProcLst = list(range(bs_ppo_update))
                            active_old_external_memory_masks_rows =\
                                torch.LongTensor(active_old_external_memory_masks_rows_oneProcLst).unsqueeze(1).repeat(1, self.k_old_steps).contiguous().view(-1)
                        else:
                            if active_old_external_memory_masks_cols_firstProc.size(0) >= self.k_old_steps:
                                remaining_k_old_steps_to_be_sampledFromRight =\
                                    self.k_old_steps - (last_active_col_left - first_active_col_left + 1)
                                active_old_external_memory_masks_cols_sampledFromLeft_oneProcLst = list(range(0, last_active_col_left + 1))
                                active_old_external_memory_masks_cols_sampledFromRight_oneProcLst =\
                                    list(range(last_active_col_right - remaining_k_old_steps_to_be_sampledFromRight + 1,
                                               last_active_col_right + 1))
                                active_old_external_memory_masks_cols_oneProcLst\
                                    = active_old_external_memory_masks_cols_sampledFromRight_oneProcLst +\
                                      active_old_external_memory_masks_cols_sampledFromLeft_oneProcLst
                                active_old_external_memory_masks_cols =\
                                    torch.LongTensor(active_old_external_memory_masks_cols_oneProcLst).unsqueeze(0).repeat(bs_ppo_update, 1).contiguous().view(-1)

                                active_old_external_memory_masks_rows_oneProcLst = list(range(bs_ppo_update))
                                active_old_external_memory_masks_rows =\
                                    torch.LongTensor(active_old_external_memory_masks_rows_oneProcLst).unsqueeze(1).repeat(1, self.k_old_steps).contiguous().view(-1)
                            else:
                                """this could be fixed but since rows and cols being only used as idxs to set masks to False
                                ... not fixing it for now...here all rows and cols being used, no clipping due to k-steps because the number
                                of valid rows and cols is less than k_old_steps"""
                                active_old_external_memory_masks_rows = active_old_external_memory_masks_idxs[0].view(bs_ppo_update, -1)[:, -self.k_old_steps:].contiguous().view(-1)
                                active_old_external_memory_masks_cols = active_old_external_memory_masks_idxs[1].view(bs_ppo_update, -1)[:, -self.k_old_steps:].contiguous().view(-1)
                    else:
                        active_old_external_memory_masks_rows = active_old_external_memory_masks_idxs[0].view(num_processes, -1)[:, -self.k_old_steps:].contiguous().view(-1)
                        active_old_external_memory_masks_cols = active_old_external_memory_masks_idxs[1].view(num_processes, -1)[:, -self.k_old_steps:].contiguous().view(-1)

                    old_external_memory_masks[active_old_external_memory_masks_rows, active_old_external_memory_masks_cols] = 1.0
                old_external_memory_masks = old_external_memory_masks.contiguous().view(num_processes, -1)
            k_old_external_memory_mask_curr_memIdx = torch.cat((old_external_memory_masks, newest_external_memory_masks), dim=-1)
            if k_old_external_memory_masks is None:
                k_old_external_memory_masks = k_old_external_memory_mask_curr_memIdx
            else:
                k_old_external_memory_masks = torch.cat((k_old_external_memory_masks,
                                                        k_old_external_memory_mask_curr_memIdx), dim=0)

        return k_old_external_memory_masks

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step_nav(self):
        nav_params = list(self.actor_critic.nav_net.parameters()) +\
                     list(self.actor_critic.action_distribution.parameters()) +\
                     list(self.actor_critic.critic.parameters())
        nn.utils.clip_grad_norm_(
            nav_params, self.max_grad_norm
        )

    def before_step_sep(self):
        sep_params = list(self.actor_critic.sep_enc.parameters()) +\
                     list(self.actor_critic.separator.parameters())
        if self.refine_bin2mono:
            if hasattr(self.actor_critic, "sep_enc_refine") and (self.actor_critic.sep_enc_refine is not None):
                sep_params.extend(list(self.actor_critic.sep_enc_refine.parameters()))
                sep_params.extend(list(self.actor_critic.separator_head_refine.parameters()))

        if self.predict_target_class_mono:
            if hasattr(self.actor_critic, "mem_head") and (self.actor_critic.mem_head is not None):
                sep_params.extend(list(self.actor_critic.mem_head.parameters()))

        nn.utils.clip_grad_norm_(
            sep_params, self.max_grad_norm
        )

    def after_step(self):
        pass


def distributed_mean_and_var(
    values: torch.Tensor,
):
    r"""Computes the mean and variances of a tensor over multiple workers.
    This method is equivalent to first collecting all versions of values and
    then computing the mean and variance locally over that
    :param values: (*,) shaped tensors to compute mean and variance over.  Assumed
                        to be solely the workers local copy of this tensor,
                        the resultant mean and variance will be computed
                        over _all_ workers version of this tensor.
    """
    assert distrib.is_initialized(), "Distributed must be initialized"

    world_size = distrib.get_world_size()
    mean = values.mean()
    distrib.all_reduce(mean)
    mean /= world_size

    sq_diff = (values - mean).pow(2).mean()
    distrib.all_reduce(sq_diff)
    var = sq_diff / world_size

    return mean, var


class DecentralizedDistributedMixin:
    def _get_advantages_distributed(
        self, rollouts_nav
    ) -> torch.Tensor:
        advantages = rollouts_nav.returns[:-1] - rollouts_nav.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        mean, var = distributed_mean_and_var(advantages)

        return (advantages - mean) / (var.sqrt() + EPS_PPO)

    def init_distributed(self, find_unused_params: bool = True) -> None:
        r"""Initializes distributed training for the model
        1. Broadcasts the model weights from world_rank 0 to all other workers
        2. Adds gradient hooks to the model
        :param find_unused_params: Whether or not to filter out unused parameters
                                   before gradient reduction.  This *must* be True if
                                   there are any parameters in the model that where unused in the
                                   forward pass, otherwise the gradient reduction
                                   will not work correctly.
        """
        # NB: Used to hide the hooks from the nn.Module,
        # so they don't show up in the state_dict
        class Guard:
            def __init__(self, model, device):
                if torch.cuda.is_available():
                    self.ddp = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids=[device], output_device=device
                    )
                else:
                    self.ddp = torch.nn.parallel.DistributedDataParallel(model)

        self._ddp_hooks = Guard(self.actor_critic, self.device)
        self.get_advantages = self._get_advantages_distributed

        self.reducer = self._ddp_hooks.ddp.reducer
        self.find_unused_params = find_unused_params

    def before_backward(self, loss):
        super().before_backward(loss)

        if self.find_unused_params:
            self.reducer.prepare_for_backward([loss])
        else:
            self.reducer.prepare_for_backward([])


# Mixin goes second that way the PPO __init__ will still be called
# class DDPPO(PPO, DecentralizedDistributedMixin):
#     pass
class DDPPO(DecentralizedDistributedMixin, PPO):
    pass

