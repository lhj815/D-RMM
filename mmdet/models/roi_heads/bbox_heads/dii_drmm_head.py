import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision
from torchvision.ops import box_iou
import torch.distributions

from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.runner import auto_fp16, force_fp32

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_transformer
from .bbox_head import BBoxHead


@HEADS.register_module()
class DII_DRMMHead(BBoxHead):
    r"""
    Args:
        num_classes (int): Number of class in dataset.
            Defaults to 80.
        num_ffn_fcs (int): The number of fully-connected
            layers in FFNs. Defaults to 2.
        num_heads (int): The hidden dimension of FFNs.
            Defaults to 8.
        num_cls_fcs (int): The number of fully-connected
            layers in classification subnet. Defaults to 1.
        num_reg_fcs (int): The number of fully-connected
            layers in regression subnet. Defaults to 3.
        feedforward_channels (int): The hidden dimension
            of FFNs. Defaults to 2048
        in_channels (int): Hidden_channels of MultiheadAttention.
            Defaults to 256.
        dropout (float): Probability of drop the channel.
            Defaults to 0.0
        ffn_act_cfg (dict): The activation config for FFNs.
        dynamic_conv_cfg (dict): The convolution config
            for DynamicConv.
        loss_iou (dict): The config for iou or giou loss.

    """

    def __init__(self,
                 num_classes=80,
                 num_ffn_fcs=2,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_reg_fcs=3,
                 feedforward_channels=2048,
                 in_channels=256,
                 dropout=0.0,
                 ffn_act_cfg=None,
                 dynamic_conv_cfg=None,
                 init_cfg=None,
                 additional_cfg=None,
                 **kwargs):

        if ffn_act_cfg is None:
            ffn_act_cfg = dict(
                type='ReLU', inplace=True)

        if dynamic_conv_cfg is None:
            dynamic_conv_cfg = dict(
                type='DynamicConv',
                in_channels=256,
                feat_channels=64,
                out_channels=256,
                input_feat_shape=7,
                act_cfg=dict(type='ReLU', inplace=True),
                norm_cfg=dict(type='LN'))

        assert additional_cfg is not None, 'DRMMHead additional_config is None'
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'

        super(DII_DRMMHead, self).__init__(
            num_classes=num_classes,
            reg_decoded_bbox=True,
            reg_class_agnostic=True,
            init_cfg=init_cfg,
            **kwargs)

        # ------------------------ #
        # additional configuration #
        # ------------------------ #
        # print(additional_cfg.keys())
        # exit()
        self.top_k_ratio = additional_cfg['top_k_ratio']

        self.coord_dist = additional_cfg['coord_dist']
        self.t_freedom = additional_cfg['t_freedom']

        self.min_sc_ratio = additional_cfg['min_sc_ratio']

        self.mcm_coord_detach = additional_cfg['mcm_coord_detach']
        self.mcm_mix_detach = additional_cfg['mcm_mix_detach']
        self.mcm_cat_detach = additional_cfg['mcm_cat_detach']

        self.prob_weight = additional_cfg['prob_weight']
        self.mm_weight = additional_cfg['mm_weight']
        self.mcm_weight = additional_cfg['mcm_weight']
        self.loss_reduction = additional_cfg['loss_reduction']

        # ------------------------ #

        self.in_channels = in_channels
        self.fp16_enabled = False
        self.attention = MultiheadAttention(in_channels, num_heads, dropout)
        self.attention_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.instance_interactive_conv = build_transformer(dynamic_conv_cfg)
        self.instance_interactive_conv_dropout = nn.Dropout(dropout)
        self.instance_interactive_conv_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.ffn = FFN(
            in_channels,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.ffn_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            if _ == 0:
                self.cls_fcs.append(
                    nn.Linear(in_channels, in_channels, bias=False))
            else:
                self.cls_fcs.append(
                    nn.Linear(in_channels, in_channels, bias=False))
            self.cls_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.cls_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))
        self.fc_cls = nn.Linear(in_channels, self.num_classes + 1)
	
        self.reg_fcs = nn.ModuleList()
        for _ in range(num_reg_fcs):
            if _ == 0:
                self.reg_fcs.append(
                    nn.Linear(in_channels, in_channels, bias=False))
            else:
                self.reg_fcs.append(
                    nn.Linear(in_channels, in_channels, bias=False))
            self.reg_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.reg_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))
        self.fc_reg = nn.Linear(in_channels, 4 + 4)

        assert self.reg_class_agnostic, 'DIIHead only ' \
            'suppport `reg_class_agnostic=True` '
        assert self.reg_decoded_bbox, 'DIIHead only ' \
            'suppport `reg_decoded_bbox=True`'

    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        super(DII_DRMMHead, self).init_weights()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass

    # @auto_fp16()
    def forward(self, rois, roi_feat, proposal_feat, fg, prob):
        """Forward function of Dynamic Instance Interactive Head.

        Args:
            rois (Tensors): Rois in total batch.
                With shape (batch_size*num_proposals, 5).
                the last dimension 5 represents (img_index, x1, y1, x2, y2).
            roi_feat (Tensor): Roi-pooling features with shape
                (batch_size*num_proposals, feature_dimensions,
                pooling_h , pooling_w).
            proposal_feat (Tensor): Intermediate feature get from
                diihead in last stage, has shape
                (batch_size, num_proposals, feature_dimensions)

          Returns:
                tuple[Tensor]: Usually a tuple of classification scores
                and bbox prediction and a intermediate feature.

                    - cls_scores (Tensor): Classification scores for
                      all proposals, has shape
                      (batch_size, num_proposals, num_classes).
                    - bbox_preds (Tensor): Box energies / deltas for
                      all proposals, has shape
                      (batch_size, num_proposals, 4).
                    - obj_feat (Tensor): Object feature before classification
                      and regression subnet, has shape
                      (batch_size, num_proposal, feature_dimensions).
        """
        N, num_proposals = proposal_feat.shape[:2]
        roi_coord = rois.view(N, num_proposals, 5)[:, :, 1:5]
        # roi_coord = torch.cat([roi_coord[:, :100].detach(), roi_coord[:, 100:]], dim=1)

        # Self attention
        proposal_feat = proposal_feat.permute(1, 0, 2)
        proposal_feat = self.attention_norm(self.attention(proposal_feat))

        # instance interactive
        proposal_feat = proposal_feat.permute(1, 0, 2).reshape(-1, self.in_channels)
        proposal_feat_iic = self.instance_interactive_conv(proposal_feat, roi_feat)
        proposal_feat = proposal_feat + self.instance_interactive_conv_dropout(proposal_feat_iic)
        obj_feat = self.instance_interactive_conv_norm(proposal_feat)

        # FFN
        obj_feat = self.ffn_norm(self.ffn(obj_feat))

        cls_feat = obj_feat
        reg_feat = obj_feat

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)

        mix_prob_output = self.fc_cls(cls_feat).view(N, num_proposals, -1)
        _fg, _prob = torch.split(mix_prob_output, split_size_or_sections=[1, self.num_classes], dim=2)

        loc_sc_output = self.fc_reg(reg_feat).view(N, num_proposals, -1)
        _loc, _sc = torch.split(loc_sc_output, split_size_or_sections=[4, 4], dim=2)

        fg = torch.sigmoid(_fg)
        mix = fg / torch.sum(fg, dim=1, keepdim=True)
        prob = torch.softmax(_prob, dim=2)

        loc = _loc + roi_coord
        sc = func.softplus(_sc)

        if self.top_k_ratio is not None:
            obj_feat = obj_feat.view(N, num_proposals, -1)
            _, indices = torch.topk(torch.cat([fg] * obj_feat.shape[2], dim=2), k=int(num_proposals * self.top_k_ratio), dim=1)

            loc = torch.gather(loc, dim=1, index=indices[:, :, :4])
            sc = torch.gather(sc, dim=1, index=indices[:, :, :4])
            prob = torch.gather(prob, dim=1, index=indices[:, :, :prob.shape[2]])
            fg = torch.gather(fg, dim=1, index=indices[:, :, :1])
            mix = torch.gather(mix, dim=1, index=indices[:, :, :1])
            obj_feat = torch.gather(obj_feat, dim=1, index=indices)

            num_proposals = obj_feat.shape[1]

        output_dict = {'loc': loc, 'sc': sc, 'prob': prob, 'fg': fg, 'mix': mix,
                       'obj_feat': obj_feat.view(N, num_proposals, -1)}


        return output_dict

    # @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self, output_dict, gt_box, gt_label, stage):
        loc = output_dict['loc']
        sc = output_dict['sc']
        prob = output_dict['prob']
        mix = output_dict['mix']
        fg = output_dict['fg']
        """
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
        """
        n_classes = prob.shape[2]

        assert self.coord_dist in ('cauchy', 'gaussian', 'student_t')
        moc_loss, mm_loss, lm_loss = list(), list(), list()

        for i, (loc_s, sc_s, prob_s, fg_s, mix_s, gt_box_s, gt_label_s) \
            in enumerate(zip(loc, sc, prob, fg, mix, gt_box, gt_label)):

            zero_gt = False
            n_gt_s = gt_box_s.shape[0]
            if n_gt_s == 0:
                max_idx = torch.max(fg_s, dim=0)[1]
                gt_box_s = loc_s[max_idx, :].clone().detach()
                gt_label_s = torch.randint(low=0, high=prob_s.shape[1], size=(1,)).cuda()
                zero_gt = True
                n_gt_s = 1

            mix_s = mix_s.unsqueeze(dim=0)[:, :, 0]
            mix_s = mix_s * n_gt_s
            fg_s = fg_s.unsqueeze(dim=0)[:, :, 0]
            # mix_s: (1, #comp)
            # gt_onehot_s:  (#comp, #cls)


            # coord scale ----------------------------------------------------------------------------------------------
            whwh_s = torch.cat([loc_s[:, 2:4] - loc_s[:, 0:2]] * 2, dim=1)
            sc_s = torch.max(sc_s, whwh_s * self.min_sc_ratio)

            # moc_loss -------------------------------------------------------------------------------------------------
            if self.coord_dist == 'cauchy':
                coord_lh_s = cauchy_pdf(gt_box_s, loc_s, sc_s)
                coord_lh_s = torch.prod(coord_lh_s, dim=2)
            elif self.coord_dist == 'gaussian':
                coord_lh_s = gaussian_pdf(gt_box_s, loc_s, sc_s)
                coord_lh_s = torch.prod(coord_lh_s, dim=2)
            elif self.coord_dist == 'student_t':
                coord_lh_s = log_student_t_pdf(
                    gt_box_s, loc_s, sc_s, freedom=self.t_freedom)
                coord_lh_s = torch.exp(torch.sum(coord_lh_s, dim=2))
            else:
                assert self.coord_dist in ('cauchy', 'gaussian', 'student_t')

            # mm_loss --------------------------------------------------------------------------------------------------
            gt_onehot_s = func.one_hot(gt_label_s, num_classes=n_classes).float()
            cat_lh_s = category_pmf_s(gt_onehot_s, prob_s) ** self.prob_weight
            comp_lh_s = mix_s * coord_lh_s * cat_lh_s
            mm_lh_s = torch.sum(comp_lh_s, dim=1)

            if self.mcm_mix_detach:
                mix_s = mix_s.detach()
            if self.mcm_coord_detach:
                coord_lh_s = coord_lh_s.detach()
            if self.mcm_cat_detach:
                cat_lh_s = cat_lh_s.detach()

            fg_comp_lh_s = mix_s * coord_lh_s * cat_lh_s
            fg_mm_lh_s = torch.sum(fg_comp_lh_s, dim=1)
            fg_max_lh_s = torch.max(fg_comp_lh_s, dim=1)[0]

            mm_nll_s = -torch.log(mm_lh_s + 1e-12) - self.mcm_weight * torch.log(fg_max_lh_s / (fg_mm_lh_s + 1e-12) + 1e-12)
            mm_nll_s *= self.mm_weight
            mm_nll_s = mm_nll_s * 1e-12 if zero_gt else mm_nll_s
            mm_loss.append(mm_nll_s)

        if self.loss_reduction is None:
            mm_loss = torch.cat(mm_loss, dim=0)
        elif self.loss_reduction == 'mean':
            mm_loss = torch.mean(torch.cat(mm_loss, dim=0))
        elif self.loss_reduction == 'sum':
            mm_loss = torch.sum(torch.cat(mm_loss, dim=0))
        else:
            mm_loss = 0
            assert self.loss_reduction in (None, 'mean', 'sum')
           
        return {'mm_loss': mm_loss}


def cauchy_pdf(x, loc, sc):
    # x:    (#gt, 4)
    # loc:  (#comp, 4)
    # sc:   (#comp, 4)
    x = x.unsqueeze(dim=1)
    loc = loc.unsqueeze(dim=0)
    sc = sc.unsqueeze(dim=0)
    # x:    (#gt, 1, 4)
    # loc:  (1, #comp, 4)
    # sc:   (1, #comp, 4)


    dist = ((x - loc) / sc) ** 2
    # dist: (#gt, #comp, 4)

    result = 1 / (math.pi * sc * (dist + 1))
    return result


def log_student_t_pdf(x, loc, sc, freedom=1):
    # x:    (#gt, 4)
    # loc:  (#comp, 4)
    # sc:   (#comp, 4)
    x = x.unsqueeze(dim=1)
    loc = loc.unsqueeze(dim=0)
    sc = sc.unsqueeze(dim=0)
    # x:    (#gt, 1, 4)
    # loc:  (1, #comp, 4)
    # sc:   (1, #comp, 4)

    student_t = torch.distributions.studentT.StudentT(freedom, loc, sc)
    result = student_t.log_prob(x)
    return result


def gaussian_pdf(x, loc, sc):
    # x:    (#gt, 4)
    # loc:  (#comp, 4)
    # sc:   (#comp, 4)
    x = x.unsqueeze(dim=1)
    loc = loc.unsqueeze(dim=0)
    sc = sc.unsqueeze(dim=0)
    # x:    (#gt, 1, 4)
    # loc:  (1, #comp, 4)
    # sc:   (1, #comp, 4)

    dist = ((x - loc) / sc) ** 2
    # dist: (#gt, #comp, 4)

    result = -0.5 * dist
    result = torch.exp(result) / (sc * math.sqrt(2.0 * math.pi))
    return result


def laplace_pdf(x, loc, sc):
    # x:    (#gt, 4)
    # loc:  (#comp, 4)
    # sc:   (#comp, 4)
    x = x.unsqueeze(dim=1)
    loc = loc.unsqueeze(dim=0)
    sc = sc.unsqueeze(dim=0)
    # x:    (#gt, 1, 4)
    # loc:  (1, #comp, 4)
    # sc:   (1, #comp, 4)

    # dist = (torch.abs(x - loc) ** 0.5) / sc # modified version
    dist = torch.abs(x - loc) / sc
    # dist: (#gt, #comp, 4)

    result = torch.exp(-dist) / (2 * sc)
    return result


'''def gaussian_pdf(x, mu, sig, with_iou=False):
    # make |mu|=K copies of y, subtract mu, divide by sigma
    # print(x.shape, mu.shape, sig.shape)
    dist = ((x - mu) / sig) ** 2
    # result = (x - mu) / sig

    if with_iou:
        # print('with iou')
        x_box = x[:, :, :, 0]
        loc_box = mu.transpose(dim0=2, dim1=3)[:, 0]

        iou_pair = torch.stack([
            box_iou(x_box_s, loc_box_s)
            for x_box_s, loc_box_s in zip(x_box, loc_box)
        ], dim=0).unsqueeze(dim=2)
        # iou_pair: (batch, #x, 1, #comp)
        dist += (1 - iou_pair)
        # dist: (batch, #x, 4, #comp)

        # dist = torch.cat([dist, iou_pair], dim=2)
        # batch, n_comp = sig.shape[0], sig.shape[3]
        # sig = torch.cat([sig, torch.ones(batch, 1, 1, n_comp).cuda()], dim=2)
        # dist: (batch, #x, 5, #comp)
        # sig: (batch, 1, 5, #comp)

    result = -0.5 * dist
    result = torch.exp(result) / (sig * math.sqrt(2.0 * math.pi))
    return result'''


def category_pmf_s(onehot, prob):
    # onehot:   (#gt, #cls)
    # prob:     (#comp, #cls)
    onehot = onehot.unsqueeze(dim=1)
    prob = prob.unsqueeze(dim=0)
    # onehot:   (#gt, 1, #cls)
    # prob:     (1, #comp, #cls)

    cat_prob = torch.prod(prob ** onehot, dim=2)
    # cat_prob: (#gt, #comp, #cls)
    return cat_prob

