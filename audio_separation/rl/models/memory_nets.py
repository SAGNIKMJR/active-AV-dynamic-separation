import torch
import torch.nn as nn
import torch.nn.functional as F


MIN_FREQ = 1e-4


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


class TransformerEncoderLayerCustom(nn.Module):
    r"""TransformerEncoderLayerCustom is customized TrasformerEncoderLayer (from ) following
        'Attention is All You Need in Speech Separation' (https://arxiv.org/abs/2010.13154) and
        'TransMask: A Compact and Fast Speech Separation Model Based on Transformer' (https://arxiv.org/abs/2102.09978),
        where the layer norm is put after a skip connection starts.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",):
        super(TransformerEncoderLayerCustom, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src1 = self.norm1(src)
        src2 = self.self_attn(src1, src1, src1, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src1 = self.norm2(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src1))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src1))))
        src = src + self.dropout2(src2)

        return src


class DownsampleCNN(nn.Module):
    r"""
    CNN to encode (downsample) current monaural and past monaurals predicted by the passive separator.
    """
    def __init__(self,):
        super().__init__()

        self._slicing_factor = 16

        self.downsample_cnn = nn.Sequential(
            nn.Sequential(
                *[
                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, stride=2),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            ),
            nn.Sequential(
                *[
                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, stride=2),
                ]
            )
            )

        self.layer_init()

    def layer_init(self):
        for module in self.downsample_cnn:
            for layer in module:
                if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.kaiming_normal_(
                        layer.weight,
                        nn.init.calculate_gain('leaky_relu', 0.2),
                    )
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)
                elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    if layer.affine:
                        layer.weight.data.fill_(1)
                        layer.bias.data.zero_()

    def forward(self, pred_monoFromMem):
        pred_monoFromMem = pred_monoFromMem.permute(0, 3, 1, 2)
        pred_monoFromMem = pred_monoFromMem.view(pred_monoFromMem.size(0),
                                                 pred_monoFromMem.size(1),
                                                 self._slicing_factor,
                                                 -1,
                                                 pred_monoFromMem.size(3))
        pred_monoFromMem = pred_monoFromMem.reshape(pred_monoFromMem.size(0),
                                                                          -1,
                                                                          pred_monoFromMem.size(3),
                                                                          pred_monoFromMem.size(4))

        skip_feats = []
        out = pred_monoFromMem
        for module in self.downsample_cnn:
            out = module(out)
            skip_feats.append(out)
        return out.reshape(pred_monoFromMem.size(0), -1), skip_feats[:-1][::-1]


class UpsampleCNN(nn.Module):
    r"""
    CNN to upsample monaural features encoded by the transformer memory.
    """
    def __init__(self,):
        super().__init__()
        self._slicing_factor = 16

        self.upsample_cnn = nn.Sequential(
            nn.Sequential(
                *[
                    nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            ),
            nn.Sequential(
                *[
                    nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        )

        self.layer_init()

    def layer_init(self):
        for module in self.upsample_cnn:
            for layer in module:
                if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.kaiming_normal_(
                        layer.weight,
                        nn.init.calculate_gain('leaky_relu', 0.2),
                    )
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)
                elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    if layer.affine:
                        layer.weight.data.fill_(1)
                        layer.bias.data.zero_()

    def forward(self, selfAttn_outFeats, skip_feats):
        out = selfAttn_outFeats.reshape(selfAttn_outFeats.size(0), self._slicing_factor, 8, -1)

        for idx, module in enumerate(self.upsample_cnn):
            if idx == 0:
                out = module(out)
            else:
                skipFeats_thisIdx = skip_feats[idx - 1]
                out = module(torch.cat((out, skipFeats_thisIdx), dim=1))

        out = out.view(out.size(0), -1, self._slicing_factor, out.size(2), out.size(3))
        out = out.reshape(out.size(0), out.size(1), -1, out.size(4))
        out = out.permute(0, 2, 3, 1)

        return out


class AudioMem(nn.Module):
    def __init__(
        self,
        ppo_cfg,
    ):
        super().__init__()

        self.ppo_cfg = ppo_cfg
        self.sepExtMem_cfg = ppo_cfg.TRANSFORMER_MEMORY
        self.poseEnc_cfg = self.sepExtMem_cfg.POSE_ENCODING

        self.input_size = self.sepExtMem_cfg.input_size
        self.hidden_size = self.sepExtMem_cfg.hidden_size
        self.nhead = self.sepExtMem_cfg.nhead
        self.dropout = self.sepExtMem_cfg.dropout
        self.activation = self.sepExtMem_cfg.activation
        self.num_layers = self.sepExtMem_cfg.num_layers

        self.num_pose_attrs = self.poseEnc_cfg.num_pose_attrs

        self.downsample_cnn = DownsampleCNN()

        self.transformer = nn.TransformerEncoder(
            TransformerEncoderLayerCustom(
                self.input_size,
                self.nhead,
                dim_feedforward=self.hidden_size,
                dropout=self.dropout,
                activation=self.activation,
            ),
            self.num_layers,
            norm=nn.LayerNorm(self.input_size),
        )
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.upsample_cnn = UpsampleCNN()

    @property
    def memory_dim(self):
        return self.input_size

    @property
    def hidden_state_size(self):
        return self.hidden_size

    @property
    def output_size(self):
        return self.input_size

    def forward(self, x,):
        raise NotImplementedError

    def _convert_masks_to_transformer_format(self, memory_masks):
        r"""The memory_masks is a FloatTensor with
            -   zeros for invalid locations, and
            -   ones for valid locations.
        The required format is a BoolTensor with
            -   True for invalid locations, and
            -   False for valid locations
        """
        return (1 - memory_masks) > 0

    def encode_mono(self, pred_mono):
        return self.downsample_cnn(pred_mono)

    def get_selfAttn_outFeats(self,
                              pose,
                              pred_mono_feats,
                              skip_feats,
                              sepExtMem_mono_feats,
                              sepExtMem_skipFeats,
                              sepExtMem_masks,):
        r"""

        :param pose: current pose
        :param pred_mono_feats: encoded features of monaurals predicted from passive separator
        :param skip_feats: skip connection features from encoding the current monaural
        :param sepExtMem_mono_feats: encoded features of monaurals predicted from passive separator for past steps
        :param sepExtMem_skipFeats: skip connection features from encoding past monaurals
        :param sepExtMem_masks:
        :return:
        """
        # downsampled_audio_features: [B, 1024];
        # ext_memory: [mem_size (cfg_mem_size + ppo_cfg_num_steps), B, trnsfrmr_cfg_input_size];
        # ext_memory_masks: [B, mem_size (cfg_mem_size + ppo_cfg_num_steps)];
        assert pred_mono_feats.size(0) == sepExtMem_mono_feats.size(1),\
            print(pred_mono_feats.size(), sepExtMem_mono_feats.size())

        # encoding poses from memory
        sepExtMem_poses = sepExtMem_mono_feats[..., -self.num_pose_attrs:]
        pose_feats, sepExtMem_pose_feats = self._encode_pose(pose, sepExtMem_poses)

        pred_mono_feats = pred_mono_feats + pose_feats

        sepExtMem_mono_feats = sepExtMem_mono_feats[..., :-self.num_pose_attrs] + sepExtMem_pose_feats

        # sepExtMem_masks: [B, mem_size (cfg_mem_size + ppo_cfg_num_steps) + 1];
        # sepExtMem_mono_feats: [mem_size (cfg_mem_size + ppo_cfg_num_steps) + 1, B, trnsfrmr_cfg_input_size];
        sepExtMem_masks = torch.cat([sepExtMem_masks, torch.ones([sepExtMem_masks.shape[0], 1],
                                                                 device=sepExtMem_masks.device)],
                                     dim=1)
        sepExtMem_mono_feats = torch.cat([sepExtMem_mono_feats, pred_mono_feats.unsqueeze(0)])
        sepExtMem_skipFeats = torch.cat([sepExtMem_skipFeats,
                                         skip_feats.unsqueeze(0)])

        M, bs = sepExtMem_mono_feats.shape[:2]
        sepExtMem_mono_feats = sepExtMem_mono_feats.view(M, bs, -1)
        sepExtMem_skipFeats = sepExtMem_skipFeats.view(M, bs, 16, 16, 16)

        mono_selfAttn_outFeats = self.transformer(sepExtMem_mono_feats,
                                                  src_key_padding_mask=self._convert_masks_to_transformer_format(sepExtMem_masks),)

        return mono_selfAttn_outFeats, [sepExtMem_skipFeats]

    def upsample_selfAttn_outFeats(self, mono_selfAttn_outFeats,  skip_feats,):
        M, bs = mono_selfAttn_outFeats.shape[:2]
        mono_selfAttn_outFeats = mono_selfAttn_outFeats.view(M * bs, -1)
        for idx in range(len(skip_feats)):
            skip_feats[idx] = skip_feats[idx].view(M * bs, *skip_feats[idx].size()[2:])
        out = self.upsample_cnn(mono_selfAttn_outFeats, skip_feats,)
        out = out.view(M, bs, *out.size()[1:])
        return out

    def _encode_pose(self, pose, sepExtMem_poses,):
        """
        Encodes the time component of pose
        Args:
            pose: (bs, 4) Tensor containing x, y, heading, time
            sepExtMem_poses: (M, bs, 4) Tensor containing x, y, heading, time
        """

        # encode just the time component of the pose
        pose_t = pose[..., 3:4]
        sepExtMem_poses_t = sepExtMem_poses[..., 3:4]

        # Compute relative poses. Source: https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3
        freqs = MIN_FREQ ** (2 * (torch.arange(self.input_size, dtype=torch.float32) // 2) / self.input_size)

        pose_freqs = freqs.unsqueeze(0).repeat((pose_t.size(0), 1)).to(pose_t.device)
        pose_t_feats = pose_t * pose_freqs
        pose_t_feats_clone = pose_t_feats.clone()
        pose_t_feats_clone[..., ::2] = torch.cos(pose_t_feats[..., ::2])
        pose_t_feats_clone[..., 1::2] = torch.sin(pose_t_feats[..., 1::2])
        pose_t_feats = pose_t_feats_clone

        septExtMem_pose_freqs = freqs.unsqueeze(0).unsqueeze(0)\
            .repeat((sepExtMem_poses_t.size(0), sepExtMem_poses_t.size(1), 1)).to(sepExtMem_poses_t.device)
        sepExtMem_poses_t_feats = sepExtMem_poses_t * septExtMem_pose_freqs
        sepExtMem_poses_t_feats_clone = sepExtMem_poses_t_feats.clone()
        sepExtMem_poses_t_feats_clone[..., ::2] = torch.cos(sepExtMem_poses_t_feats[..., ::2])
        sepExtMem_poses_t_feats_clone[..., 1::2] = torch.sin(sepExtMem_poses_t_feats[..., 1::2])
        sepExtMem_poses_t_feats = sepExtMem_poses_t_feats_clone

        return pose_t_feats, sepExtMem_poses_t_feats
