import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from audio_separation.rl.ppo.ddppo_utils import distributed_mean_and_var

EPS_PPO = 1e-5


class PPO(nn.Module):
    def __init__(
        self,
        actor_critic,
        ppo_cfg,
        use_clipped_value_loss=True,
        use_normalized_advantage=True,
    ):
        super().__init__()

        self.actor_critic = actor_critic

        self.ppo_cfg = ppo_cfg
        self.sepExtMem_cfg = ppo_cfg.TRANSFORMER_MEMORY
        self.poseEnc_cfg = self.sepExtMem_cfg.POSE_ENCODING

        self.clip_param = ppo_cfg.clip_param
        self.ppo_epoch = ppo_cfg.ppo_epoch
        self.num_mini_batch = ppo_cfg.num_mini_batch

        self.value_loss_coef = ppo_cfg.value_loss_coef
        self.entropy_coef = ppo_cfg.entropy_coef

        self.max_grad_norm = ppo_cfg.max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_normalized_advantage = use_normalized_advantage

        self.num_steps = ppo_cfg.num_steps
        self.num_past_steps_refinement = self.sepExtMem_cfg.num_past_steps_refinement

        pol_params = list(actor_critic.pol_net.parameters()) + list(actor_critic.action_dist.parameters()) +\
                     list(actor_critic.critic.parameters())
        self.optimizer_pol = optim.Adam(pol_params, lr=ppo_cfg.lr_pol, eps=ppo_cfg.eps)

        sep_params = list(actor_critic.binSep_enc.parameters()) + list(actor_critic.binSep_dec.parameters()) +\
                     list(actor_critic.bin2mono_enc.parameters()) + list(actor_critic.bin2mono_dec.parameters()) +\
                     list(actor_critic.audio_mem.parameters())
        self.optimizer_sep = optim.Adam(sep_params, lr=ppo_cfg.lr_sep, eps=ppo_cfg.eps)

        self.device = next(actor_critic.parameters()).device

    def load_pretrained_passive_separators(self, state_dict):
        # loading pretrained weights from passive binaural separator
        for name in self.actor_critic.binSep_enc.state_dict():
            self.actor_critic.binSep_enc.state_dict()[name].copy_(state_dict["actor_critic.binSep_enc." + name])
        for name in self.actor_critic.binSep_dec.state_dict():
            self.actor_critic.binSep_dec.state_dict()[name].copy_(state_dict["actor_critic.binSep_dec." + name])

        # loading pretrained weights from passive bin2mono separator
        for name in self.actor_critic.bin2mono_enc.state_dict():
            self.actor_critic.bin2mono_enc.state_dict()[name].copy_(state_dict["actor_critic.bin2mono_enc." + name])
        for name in self.actor_critic.bin2mono_dec.state_dict():
            self.actor_critic.bin2mono_dec.state_dict()[name].copy_(state_dict["actor_critic.bin2mono_dec." + name])

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts_pol):
        advantages = rollouts_pol.returns[:-1] - rollouts_pol.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update_pol(self, rollouts_pol):
        advantages = self.get_advantages(rollouts_pol)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts_pol.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_pol_batch,
                    pred_binSepMasks_batch,
                    pred_monoFromMem_batch,
                    value_preds_batch,
                    return_batch,
                    adv_targ,
                    actions_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                ) = sample

                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_pol_batch,
                    masks_batch,
                    actions_batch,
                    pred_monoFromMem_batch,
                    pred_binSepMasks_batch,
                )

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

                self.optimizer_pol.zero_grad()
                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                )

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step_pol()
                self.optimizer_pol.step()
                self.after_step()

                action_loss_epoch += action_loss.item()
                value_loss_epoch += value_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        action_loss_epoch /= num_updates
        value_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def update_sep(self, rollouts_sep):
        bin_loss_epoch = 0.
        mono_loss_epoch = 0.
        monoFromMem_loss_epoch = 0.

        for e in range(self.ppo_epoch):
            data_generator = rollouts_sep.recurrent_generator(self.num_mini_batch)

            for sample in data_generator:
                (
                    obs_batch,
                    masks_batch,
                    sepExtMem_mono_batch,
                    sepExtMem_skipFeats_batch,
                    sepExtMem_masks_batch,
                    gtMonoMag_pastStepsRefine_batch,
                ) = sample

                with torch.no_grad():
                    pred_binSepMasks =\
                        self.actor_critic.get_binSepMasks(
                            obs_batch,
                        )
                    pred_mono =\
                        self.actor_critic.convert_bin2mono(pred_binSepMasks.detach(),
                                                           mixed_audio=obs_batch["mixed_bin_audio_mag"],
                                                           )

                with torch.autograd.set_detect_anomaly(True):
                    pred_monoFromMem, _, _, _ =\
                        self.actor_critic.get_monoFromMem(pred_mono=pred_mono,
                                                          sepExtMem_mono=sepExtMem_mono_batch,
                                                          sepExtMem_masks=sepExtMem_masks_batch,
                                                          pose=obs_batch["pose"],
                                                          sepExtMem_skipFeats=sepExtMem_skipFeats_batch,
                                                          )

                    gt_mono_mag =\
                        torch.cat((gtMonoMag_pastStepsRefine_batch,
                                   obs_batch["gt_mono_comps"][..., 0::2].clone()[..., :1].unsqueeze(1)),
                                  dim=1)

                    sepExtMem_masks_batch = torch.cat([sepExtMem_masks_batch,
                                                       torch.ones([sepExtMem_masks_batch.shape[0], 1],
                                                                  device=sepExtMem_masks_batch.device)],
                                                      dim=1)
                    sepExtMem_masks = self.process_sepExtMem_masks(sepExtMem_masks_batch)

                    sepExtMem_masks = sepExtMem_masks.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, *gt_mono_mag.size()[2:])

                    pred_monoFromMem = (pred_monoFromMem.permute(1, 0, 2, 3, 4) * sepExtMem_masks)\
                        .reshape(sepExtMem_masks.size(0) * sepExtMem_masks.size(1), *sepExtMem_masks.size()[2:])
                    gt_mono_mag = (gt_mono_mag * sepExtMem_masks)\
                        .reshape(sepExtMem_masks.size(0) * sepExtMem_masks.size(1), *sepExtMem_masks.size()[2:])

                    monoFromMem_loss = F.l1_loss(pred_monoFromMem, gt_mono_mag)

                    mono_loss = F.l1_loss(pred_mono, obs_batch["gt_mono_comps"][..., 0::2].clone()[..., :1])

                    gt_bin_mag = obs_batch["gt_bin_comps"][..., 0::2].clone()[..., :2]
                    pred_bin = (torch.exp(obs_batch["mixed_bin_audio_mag"]) - 1) * pred_binSepMasks
                    bin_loss = F.l1_loss(pred_bin, gt_bin_mag)

                    self.optimizer_sep.zero_grad()
                    total_loss = monoFromMem_loss

                    self.before_backward(total_loss)
                    total_loss.backward()
                    self.after_backward(total_loss)

                self.before_step_sep()
                self.optimizer_sep.step()
                self.after_step()

                bin_loss_epoch += bin_loss.item()
                mono_loss_epoch += mono_loss.item()
                monoFromMem_loss_epoch += monoFromMem_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        bin_loss_epoch /= num_updates
        mono_loss_epoch /= num_updates
        monoFromMem_loss_epoch /= num_updates

        return bin_loss_epoch, mono_loss_epoch, monoFromMem_loss_epoch

    def process_sepExtMem_masks(self, sepExtMem_masks,):
        r"""
        set 'num_past_steps_refinement' indexes in external memory masks to 1 and the rest to 0
        :param sepExtMem_masks: external memory masks
        :return: processed external memory masks
        """
        bs, M = sepExtMem_masks.size()

        num_processes = bs // self.num_steps
        processed_sepExtMem_masks = None
        for rolloutStorage_idx in range(self.num_steps):
            newest_sepExtMem_masks_wCurrStep = sepExtMem_masks[rolloutStorage_idx * num_processes: (rolloutStorage_idx + 1) * num_processes, -1:].clone()
            old_sepExtMem_masks_wCurrStep = sepExtMem_masks[rolloutStorage_idx * num_processes: (rolloutStorage_idx + 1) * num_processes:, :-1].clone()
            old_activeIdxs_sepExtMem_masks_wCurrStep = torch.where(old_sepExtMem_masks_wCurrStep == 1)
            old_sepExtMem_masks_wCurrStep[:, :] = 0

            bs_ppo_update = old_sepExtMem_masks_wCurrStep.size(0)
            assert bs_ppo_update == num_processes

            if old_activeIdxs_sepExtMem_masks_wCurrStep[0].size()[0] != 0:
                old_activeColsFirstProcess_sepExtMem_masks_wCurrStep =\
                    old_activeIdxs_sepExtMem_masks_wCurrStep[1].view(bs_ppo_update, -1)[0]
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
                    if last_active_col_left - first_active_col_left + 1 >= self.num_past_steps_refinement:
                        lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep =\
                            list(range(last_active_col_left - self.num_past_steps_refinement + 1,  last_active_col_left + 1))
                        old_activeCols_sepExtMem_masks_wCurrStep =\
                            torch.LongTensor(lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep).unsqueeze(0)\
                                .repeat(bs_ppo_update, 1).contiguous().view(-1)

                        lst_old_activeRowsOneProcess_sepExtMem_masks_wCurrStep = list(range(bs_ppo_update))
                        old_activeRows_sepExtMem_masks_wCurrStep =\
                            torch.LongTensor(lst_old_activeRowsOneProcess_sepExtMem_masks_wCurrStep).unsqueeze(1)\
                                .repeat(1, self.num_past_steps_refinement).contiguous().view(-1)
                    else:
                        if old_activeColsFirstProcess_sepExtMem_masks_wCurrStep.size(0) >= self.num_past_steps_refinement:
                            remaining_numPastStepsRefinement_toBeSampledFromRight =\
                                self.num_past_steps_refinement - (last_active_col_left - first_active_col_left + 1)
                            lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromLeft =\
                                list(range(0, last_active_col_left + 1))
                            lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromRight =\
                                list(range(last_active_col_right - remaining_numPastStepsRefinement_toBeSampledFromRight + 1,
                                           last_active_col_right + 1))
                            lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep\
                                = lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromRight +\
                                  lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep_sampledFromLeft
                            old_activeCols_sepExtMem_masks_wCurrStep =\
                                torch.LongTensor(lst_old_activeColsOneProcess_sepExtMem_masks_wCurrStep).unsqueeze(0)\
                                    .repeat(bs_ppo_update, 1).contiguous().view(-1)

                            lst_old_activeRowsOneProcess_sepExtMem_masks_wCurrStep = list(range(bs_ppo_update))
                            old_activeRows_sepExtMem_masks_wCurrStep =\
                                torch.LongTensor(lst_old_activeRowsOneProcess_sepExtMem_masks_wCurrStep).unsqueeze(1)\
                                    .repeat(1, self.num_past_steps_refinement).contiguous().view(-1)
                        else:
                            old_activeRows_sepExtMem_masks_wCurrStep =\
                                old_activeIdxs_sepExtMem_masks_wCurrStep[0].view(bs_ppo_update, -1)[:, -self.num_past_steps_refinement:]\
                                    .contiguous().view(-1)
                            old_activeCols_sepExtMem_masks_wCurrStep =\
                                old_activeIdxs_sepExtMem_masks_wCurrStep[1].view(bs_ppo_update, -1)[:, -self.num_past_steps_refinement:]\
                                    .contiguous().view(-1)
                else:
                    old_activeRows_sepExtMem_masks_wCurrStep =\
                        old_activeIdxs_sepExtMem_masks_wCurrStep[0].view(bs_ppo_update, -1)[:, -self.num_past_steps_refinement:]\
                            .contiguous().view(-1)
                    old_activeCols_sepExtMem_masks_wCurrStep =\
                        old_activeIdxs_sepExtMem_masks_wCurrStep[1].view(bs_ppo_update, -1)[:, -self.num_past_steps_refinement:]\
                            .contiguous().view(-1)

                old_sepExtMem_masks_wCurrStep[old_activeRows_sepExtMem_masks_wCurrStep,
                                              old_activeCols_sepExtMem_masks_wCurrStep] = 1.0
                old_sepExtMem_masks_wCurrStep = old_sepExtMem_masks_wCurrStep.contiguous().view(bs_ppo_update, -1)

            temp_processed_sepExtMem_masks = torch.cat((old_sepExtMem_masks_wCurrStep, newest_sepExtMem_masks_wCurrStep), dim=-1)
            if processed_sepExtMem_masks is None:
                processed_sepExtMem_masks = temp_processed_sepExtMem_masks
            else:
                processed_sepExtMem_masks = torch.cat((processed_sepExtMem_masks,
                                                       temp_processed_sepExtMem_masks), dim=0)

        return processed_sepExtMem_masks

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step_pol(self):
        pol_params = list(self.actor_critic.pol_net.parameters()) +\
                     list(self.actor_critic.action_dist.parameters()) +\
                     list(self.actor_critic.critic.parameters())
        nn.utils.clip_grad_norm_(
            pol_params, self.max_grad_norm
        )

    def before_step_sep(self):
        sep_params = list(self.actor_critic.binSep_enc.parameters()) + list(self.actor_critic.binSep_dec.parameters()) +\
                     list(self.actor_critic.bin2mono_enc.parameters()) + list(self.actor_critic.bin2mono_dec.parameters()) +\
                     list(self.actor_critic.audio_mem.parameters())
        nn.utils.clip_grad_norm_(
            sep_params, self.max_grad_norm
        )

    def after_step(self):
        pass


class DecentralizedDistributedMixin:
    def _get_advantages_distributed(
        self, rollouts_pol
    ) -> torch.Tensor:
        advantages = rollouts_pol.returns[:-1] - rollouts_pol.value_preds[:-1]
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
class DDPPO(DecentralizedDistributedMixin, PPO):
    pass
