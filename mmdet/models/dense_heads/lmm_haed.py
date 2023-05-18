import math
import numpy as np
import torch
import torch.nn as nn
from torchvision import ops
from torchvision.ops import nms
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms,
                        reduce_mean, unmap)
from ..builder import HEADS, build_loss
from .atss_head import ATSSHead


@HEADS.register_module()
class LMMHead(ATSSHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=None,
                 loss_cls=None,
                 loss_bbox=None,
                 loss_centerness=None,
                 init_cfg=None,
                 lmm_cfg=None,
                 **kwargs):

        if norm_cfg is None:
            norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        if anchor_generator is None:
            anchor_generator = dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[8, 16, 32, 64, 128])
        if loss_cls is None:
            loss_cls = dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0)
        if loss_bbox is None:
            loss_bbox = dict(type='GIoULoss', loss_weight=2.0)
        if loss_centerness is None:
            loss_centerness = dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
        if lmm_cfg is None:
            lmm_cfg = dict(
                iou_thresholds=[0.4, 0.6, 0.8],
                train_threshold=0.001,
                train_max_box=20000,
                train_bin_size=1000,
                test_max_box=5000,
                gamma_factor=0.05,
                rescale_factor=1.0/100.0,
                feat_detach=False,
                loss_nll_weight=0.05,
                loss_lmm_weight=0.05)
        if init_cfg is None:
            init_cfg = dict(
                type='Normal', layer='Conv2d', std=0.01,
                override=dict(type='Normal', name='atss_cls', std=0.01, bias_prob=0.01))

        super(LMMHead, self).__init__(
            num_classes, in_channels, stacked_convs, conv_cfg, norm_cfg,
            anchor_generator=anchor_generator, loss_cls=loss_cls, loss_bbox=loss_bbox,
            loss_centerness=loss_centerness, init_cfg=init_cfg, **kwargs)

        self.lmm_cfg = lmm_cfg
        self.__init_layers()

    def __init_layers(self):
        super(LMMHead, self)._init_layers()
        self.lmm = LMM(**self.lmm_cfg)

    def __forward_base(self, feats, img_metas, train=True):
        B, CH = feats[0].shape[0:2]

        cls_score, bbox_pred, centerness = multi_apply(self.forward_single, feats, self.scales)
        # bbox_preds (list[Tensor]):    [(B, 4, FH1, FW1), ..., (B, 4, FH5, FW5)]
        # cls_score (list[Tensor]):     [(B, CLS, FH1, FW1), ..., (B, CLS, FH5, FW5)]
        # centerness (list[Tensor]):    [(B, 1, FH1, FW1), ..., (B, 1, FH5, FW5)]

        # return cls_score, bbox_pred, centerness
        featmap_sizes = [cls_score[i].shape[-2:] for i in range(5)]
        anchor = self.anchor_generator.grid_anchors(
            featmap_sizes, device=cls_score[0].device)
        anchor_flat = torch.cat(anchor, dim=0)

        C = cls_score[0].shape[1]
        bbox_flat = torch.cat([bbox_i.view(B, 4, -1) for bbox_i in bbox_pred], dim=2).transpose(1, 2)
        bbox_dec_flat = torch.stack(
            [self.bbox_coder.decode(anchor_flat, bbox_flat_i) for bbox_flat_i in bbox_flat],
            dim=0).transpose(1, 2)

        for i in range(B):
            # print(bbox_dec_flat.shape)
            bbox_dec_flat[i, [0, 2]] = torch.clamp(
                bbox_dec_flat[i, [0, 2]], min=0, max=img_metas[i]['img_shape'][1])
            bbox_dec_flat[i, [1, 3]] = torch.clamp(
                bbox_dec_flat[i, [1, 3]], min=0, max=img_metas[i]['img_shape'][0])

        score_flat = torch.cat([score_i.view(B, C, -1) for score_i in cls_score], dim=2).sigmoid()
        center_flat = torch.cat([center_i.view(B, 1, -1) for center_i in centerness], dim=2).sigmoid()
        feat_flat = torch.cat([feat.view(B, CH, -1) for feat in feats], dim=2)
        cnt_score_flat = score_flat * center_flat
        bbox_score_flat = torch.max(cnt_score_flat, dim=1, keepdim=True)[0]
        # bbox_score_flat = torch.mean(cnt_score_flat, dim=1, keepdim=True)
        # bbox_score_flat = torch.max(score_flat, dim=1, keepdim=True)[0]

        if train:
            pi, mu, gamma, loc_max = self.lmm(bbox_dec_flat, bbox_score_flat, feat_flat, train=True)
            return cls_score, bbox_pred, centerness, pi, mu, gamma, loc_max
        else:
            loc_max_flat, keep_args = self.lmm(bbox_dec_flat, bbox_score_flat, feat_flat, train=False)
            if keep_args is not None:
                bbox_dec_flat = torch.stack([bbox_dec_flat[b, :, keep_args[b, :, 0]] for b in range(B)], dim=0)
                cnt_score_flat = torch.stack([cnt_score_flat[b, :, keep_args[b, :, 0]] for b in range(B)], dim=0)

            conf_flat = cnt_score_flat * loc_max_flat
            return bbox_dec_flat.transpose(1, 2), conf_flat.transpose(1, 2)
            # return bbox_dec_flat.transpose(1, 2), cnt_score_flat.transpose(1, 2)

    def forward(self, feats, img_metas):
        return self.__forward_base(feats, img_metas, train=True)

    def forward_train(self, x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=None):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x, img_metas)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def forward_test(self, feats, img_metas):
        return self.__forward_base(feats, img_metas, train=False)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses', 'pi', 'mu', 'gamma'))
    def loss(self, cls_score, bbox_pred, centerness, pi, mu, gamma, loc_max,
             gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        loss_dict = super(LMMHead, self).loss(
            cls_score, bbox_pred, centerness,
            gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore
        )
        loss_nll, loss_lmm = self.lmm.loss(pi, mu, gamma, loc_max, gt_bboxes, gt_labels, img_metas)
        loss_dict.update(dict(loss_nll=loss_nll))
        loss_dict.update(dict(loss_lmm=loss_lmm))
        return loss_dict

    def get_bboxes(self, bbox, conf, img_metas, rescale=False):
        """
        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        # bbox (Tensor):   (B, N_BOX, 4)
        # conf (Tensor):   (B, N_BOX, CLS)
        B, _, C = conf.shape

        label = conf.clone()
        for c in range(C):
            label[:, :, c] = c

        result = list()
        for bbox_i, conf_i, label_i, img_metas_i in zip(bbox, conf, label, img_metas):
            # bbox_i (Tensor):      (N_BOX, 4)
            # conf_i (Tensor):      (N_BOX, CLS)
            # label_i (Tensor):     (N_BOX, CLS)

            bbox_i[:, [0, 2]] = bbox_i[:, [0, 2]] / img_metas_i['scale_factor'][0]
            bbox_i[:, [1, 3]] = bbox_i[:, [1, 3]] / img_metas_i['scale_factor'][1]
            # bbox_i[:, [0, 2]] = torch.clamp(
            #     bbox_i[:, [0, 2]] / img_metas_i['scale_factor'][0],
            #     min=0, max=img_metas_i['ori_shape'][1])
            # bbox_i[:, [1, 3]] = torch.clamp(
            #     bbox_i[:, [1, 3]] / img_metas_i['scale_factor'][1],
            #     min=0, max=img_metas_i['ori_shape'][0])
            # scale_factor: img_shape / ori_shape

            res_bbox_i, res_conf_i, res_label_i = list(), list(), list()
            for c in range(C):
                keep_indices = np.nonzero(conf_i[:, c] > 0.01).view(-1)
                keep_bbox_ic = bbox_i[keep_indices]
                keep_conf_ic = conf_i[keep_indices, c]
                keep_label_ic = label_i[keep_indices, c]
                # print(c, keep_indices.shape)

                # keep_indices = nms(keep_bbox_ic, keep_conf_ic, 0.6).long().view(-1)
                # keep_bbox_ic = keep_bbox_ic[keep_indices]
                # keep_conf_ic = keep_conf_ic[keep_indices]
                # keep_label_ic = keep_label_ic[keep_indices]

                res_bbox_i.append(keep_bbox_ic)
                res_conf_i.append(keep_conf_ic)
                res_label_i.append(keep_label_ic)

            res_bbox_i = torch.cat(res_bbox_i, dim=0)
            res_conf_i = torch.cat(res_conf_i, dim=0)
            res_label_i = torch.cat(res_label_i, dim=0)

            if len(res_bbox_i) > 300:
                res_conf_i, keep_indices = torch.topk(res_conf_i, k=300, dim=0, sorted=True)
                res_bbox_i = res_bbox_i[keep_indices]
                res_label_i = res_label_i[keep_indices]
            result_i = (torch.cat([res_bbox_i, res_conf_i.unsqueeze(1)], dim=1), res_label_i)
            result.append(result_i)
        return result


class LMM(nn.Module):
    """local maximum mixture module"""

    def __init__(self, iou_thresholds=None, input_ch=256,
                 train_threshold=0.00001, train_max_box=20000, train_bin_size=1000,
                 test_max_box=5000, gamma_factor=0.05, rescale_factor=0.02,
                 feat_detach=False, loss_nll_weight=0.01, loss_lmm_weight=0.01):

        super(LMM, self).__init__()
        if iou_thresholds is None:
            iou_thresholds = [0.4, 0.6, 0.8]

        self.n_mask = len(iou_thresholds)
        self.iou_thresholds = iou_thresholds
        self.gamma_factor = gamma_factor

        self.train_threshold = train_threshold
        self.train_max_box = train_max_box
        self.train_bin_size = train_bin_size
        self.test_max_box = test_max_box

        max_idx_len = max(train_bin_size, test_max_box)
        box_indices = torch.from_numpy(np.array(list(range(max_idx_len))))
        self.box_indices = box_indices.unsqueeze(dim=0).cuda()

        self.rescale_factor = rescale_factor
        self.feat_detach = feat_detach
        self.loss_nll_weight = loss_nll_weight
        self.loss_lmm_weight = loss_lmm_weight
        self.epsilon = 1e-12

        self.mask_w_layers = nn.Sequential(
            nn.Conv1d(input_ch, input_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(input_ch, input_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(input_ch, input_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(input_ch, input_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(input_ch, input_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(input_ch, input_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(input_ch, self.n_mask, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Softmax(dim=1),
        )
        self.pi_layers = nn.Sequential(
            nn.Conv1d(input_ch, input_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(input_ch, input_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(input_ch, input_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(input_ch, input_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(input_ch, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.gamma_layers = nn.Sequential(
            nn.Conv1d(input_ch, input_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(input_ch, input_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(input_ch, input_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(input_ch, input_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(input_ch, input_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(input_ch, input_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(input_ch, 4, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Softplus(),
        )
        self.mean_mask_w_list = list()
        self.forward_cnt = 0

    def __thresholding(self, x, threshold):
        x.masked_fill_(x >= threshold, 1.0)
        x.masked_fill_(x < threshold, 0.0)
        return x

    def __generate_mask(self, iou_pair, score, threshold):
        N_BOX2 = iou_pair.shape[2]
        # iou_mask:     (B, N_BOX1, N_BOX2)
        # score:        (B, N_BOX1, 1)

        iou_mask = self.__thresholding(iou_pair.clone(), threshold)
        adjc_score = iou_mask * score
        max_idx = torch.argmax(adjc_score, dim=1)
        # adjc_score:   (B, N_BOX1, N_BOX2)
        # max_idx:      (B, N_BOX2)

        box_indices = self.box_indices[:, :N_BOX2]
        mask = (max_idx == box_indices).long()
        mask = mask.unsqueeze(1)
        # box_indices:  (1, N_BOX2)
        # mask:         (B, 1, N_BOX2)
        return mask

    def forward_train(self, box, score, feat):
        if self.feat_detach:
            feat = feat.detach()
        box = box.detach()
        score = score.detach()
        # boxes:    (B, 4, N_BOX)
        # score:    (B, 1, N_BOX)
        # feature:  (B, C, N_BOX)

        if self.train_threshold is not None:
            keep_args = torch.nonzero(torch.mean(score, dim=0)[0] > self.train_threshold).reshape(-1)
            if len(keep_args) > 0:
                box = box[:, :, keep_args]
                score = score[:, :, keep_args]
                feat = feat[:, :, keep_args]

        box = box.transpose(1, 2)
        score = score.transpose(1, 2)
        # boxes:    (B, N_BOX, 4)
        # score:    (B, N_BOX, 1)

        B, _, N_BOX = box.shape
        if (N_BOX > self.train_max_box) and (self.train_max_box is not None):
            score, keep_args = torch.topk(score, k=self.train_max_box, dim=1, sorted=False)
            box = torch.stack([box[b, keep_args[b, :, 0]] for b in range(B)], dim=0)
            feat = torch.stack([feat[b, :, keep_args[b, :, 0]] for b in range(B)], dim=0)
        # boxes:    (B, MAX_BOX, 4)
        # score:    (B, MAX_BOX, 1)

        box_bins = torch.split(box, self.train_bin_size, dim=1)
        masks = list()
        for box_bin in box_bins:
            iou_pair = torch.stack([ops.box_iou(box[b], box_bin[b])
                                    for b in range(B)], dim=0)
            masks_bin = [self.__generate_mask(iou_pair, score, threshold)
                         for threshold in self.iou_thresholds]
            masks_bin = torch.cat(masks_bin, dim=1)
            # iou_pair:     (B, N_BOX, BIN_SIZE)
            # masks_bin:    (B, N_MASKS, BIN_SIZE)

            masks.append(masks_bin)
        masks = torch.cat(masks, dim=2)
        # masks:    (B, N_MASKS, N_BOX)

        mask_w = self.mask_w_layers(feat)
        loc_max = torch.sum(masks.detach().float() * mask_w, dim=1, keepdim=True)
        # mask_w:   (B, N_MASKS, N_BOX)
        # loc_max:  (B, 1, #box)

        # pi = torch.softmax(self.pi_layers(feat), dim=2)
        pi = torch.exp(self.pi_layers(feat))
        # pi_score = (torch.exp(self.pi_layers(feat)) + self.epsilon) * loc_max
        # pi_score = (torch.sigmoid(self.pi_layers(feat)) * loc_max + self.epsilon)
        # pi = pi_score / torch.sum(pi_score, dim=2, keepdim=True)
        # pi = torch.softmax(self.pi_layers(feat), dim=2)
        mu = self.rescale_factor * box.transpose(1, 2).detach()
        # mu = box.transpose(1, 2).detach()

        min_gamma = self.gamma_factor * torch.clamp_min_(torch.cat(
            [mu[:, 2:4] - mu[:, 0:2]] * 2, dim=1), min=self.epsilon).detach()
        gamma = self.gamma_layers(feat) + min_gamma

        # print('')
        # print(mask_w.shape, masks.shape, loc_max.shape)
        # print(pi_score.shape, pi.shape)
        # print(mu.shape, gamma.shape, min_gamma.shape)

        self.mean_mask_w_list.append(torch.mean(mask_w, dim=2, keepdim=True))
        if self.forward_cnt % 500 == 0:
            mean_mask = torch.mean(torch.mean(masks.float(), dim=2), dim=0)
            mean_mask_w = torch.mean(torch.mean(mask_w, dim=2), dim=0)
            min_mask_w = torch.min(torch.min(mask_w, dim=2)[0], dim=0)[0]
            max_mask_w = torch.max(torch.max(mask_w, dim=2)[0], dim=0)[0]
            # print(mean_mask.data)
            print(mean_mask_w.data, min_mask_w.data, max_mask_w.data)
        self.forward_cnt += 1
        return pi, mu, gamma, loc_max

    def forward_test(self, box, score, feat):
        B, _, N_BOX = box.shape
        # boxes:    (B, 4, N_BOX)
        # score:    (B, 1, N_BOX)
        # feature:  (B, C, N_BOX)

        box = box.transpose(1, 2)
        score = score.transpose(1, 2)
        # boxes:    (B, N_BOX, 4)
        # score:    (B, N_BOX, 1)

        if (N_BOX > self.test_max_box) and (self.test_max_box is not None):
            score, keep_args = torch.topk(score, k=self.test_max_box, dim=1, sorted=True)
            box = torch.stack([box[b, keep_args[b, :, 0]] for b in range(B)], dim=0)
            feat = torch.stack([feat[b, :, keep_args[b, :, 0]] for b in range(B)], dim=0)
        else:
            keep_args = None
        # boxes:    (B, MAX_BOX, 4)
        # score:    (B, MAX_BOX, 1)

        iou_pair = ops.box_iou(box[0], box[0]).unsqueeze(0) if box.shape[0] == 1 else \
            torch.stack([ops.box_iou(box_i, box_i) for box_i in box], dim=0)
        masks = [self.__generate_mask(iou_pair, score, threshold) for threshold in self.iou_thresholds]
        masks = torch.cat(masks, dim=1)
        # masks = torch.stack([matrix_nms_torch(box_i, sigma=0.4) for box_i in box], dim=0).unsqueeze(1)
        # iou_pair  (B, MAX_BOX, MAX_BOX)
        # masks:    (B, N_MASKS, MAX_BOX)

        mask_w = self.mask_w_layers(feat)
        loc_max = torch.sum(masks * mask_w, dim=1, keepdim=True)
        # mask_w:   (B, N_MASK, N_BOX)
        # loc_max:  (B, 1, N_BOX)
        
        # self.mean_mask_w_list.append(torch.mean(mask_w, dim=2, keepdim=True))
        # print('')
        # print(torch.mean(torch.cat(self.mean_mask_w_list, dim=2), dim=2))
        # print(torch.max(torch.cat(self.mean_mask_w_list, dim=2), dim=2)[0])
        # print(torch.min(torch.cat(self.mean_mask_w_list, dim=2), dim=2)[0])
        # print('mask mean:', torch.mean(masks.float(), dim=2))
        return loc_max, keep_args

    def forward(self, box, score, feat, train=True):
        if train:
            return self.forward_train(box, score, feat)
        else:
            return self.forward_test(box, score, feat)

    def loss(self, pi, mu, gamma, loc_max, gt_box, gt_label, img_meta):
        gt_box = [gt_box_i * self.rescale_factor for gt_box_i in gt_box]

        loss_nll, loss_lmm = list(), list()
        for pi_i, mu_i, gamma_i, loc_max_i, gt_box_i, gt_label_i, img_meta_i in \
                zip(pi, mu, gamma, loc_max, gt_box, gt_label, img_meta):

            loc_max_pi_i = loc_max_i * pi_i
            loc_max_pi_i = loc_max_pi_i / torch.sum(loc_max_pi_i, dim=1, keepdim=True)

            cauchy_pdf_i = cauchy_pdf(gt_box_i, mu_i, gamma_i)
            max_lh1, _ = torch.max(loc_max_pi_i * torch.prod(
                cauchy_pdf(gt_box_i, mu_i, gamma_i), dim=1), dim=1)
            # print(pi_i.shape)
            # print(cauchy_pdf_i.shape)
            # print(torch.prod(cauchy_pdf_i, dim=1).shape)
            # print(torch.sum(pi_i * torch.prod(cauchy_pdf_i, dim=1), dim=1).shape)
            # print('')
            # moc_lh1:      (N_BOX)

            # loc_max_pi_i = loc_max_i * pi_i.detach()
            # loc_max_pi_i = loc_max_pi_i / torch.sum(loc_max_pi_i, dim=1, keepdim=True)

            cauchy_lh = loc_max_pi_i * torch.prod(
                cauchy_pdf(gt_box_i, mu_i.detach(), gamma_i.detach()), dim=1)
            max_lh2, _ = torch.max(cauchy_lh, dim=1)
            moc_lh = torch.sum(cauchy_lh, dim=1)
            # cauchy_lh:    (N_BOX, N_COMP)
            # max_lh2:      (N_BOX)
            # moc_lh:       (N_BOX)

            # print(torch.min(max_lh2), torch.max(max_lh2))
            # print(torch.min(moc_lh), torch.max(moc_lh))
            # print(self.epsilon, '\n')
            loss_nll_i = -torch.log(max_lh1 + self.epsilon)
            loss_lmm_i = -torch.log(max_lh2 / (moc_lh + self.epsilon) + self.epsilon)
            # loss_lmm_i = -torch.log(max_lh2 / (moc_lh + self.epsilon) + self.epsilon)
            # loss_lmm_i = -torch.log(max_lh2 / (moc_lh + self.epsilon) + self.epsilon)
            
            loss_nll.append(loss_nll_i)
            loss_lmm.append(loss_lmm_i)
        
        loss_nll = self.loss_nll_weight * torch.cat(loss_nll, dim=0)
        loss_lmm = self.loss_lmm_weight * torch.cat(loss_lmm, dim=0)
        # loss_lmm = [loss_lmm_i * self.loss_weight
        #             for loss_lmm_i in torch.cat(loss_lmm, dim=0)]
        return loss_nll, loss_lmm


def cauchy_pdf(box, mu, gamma):
    box = box.unsqueeze(2)
    mu = mu.unsqueeze(0)
    gamma = gamma.unsqueeze(0)
    # box:      (N_BOX, 4, 1)
    # mu:       (1, 4, N_COMP)
    # gamma:    (1, 4, N_COMP)

    dist = ((box - mu) / gamma) ** 2
    result = 1 / (math.pi * gamma * (dist + 1))
    return result


def gaussian_pdf(box, mu, sig):
    box = box.unsqueeze(2)
    mu = mu.unsqueeze(0)
    sig = sig.unsqueeze(0)
    
    dist = ((box - mu) / sig) ** 2
    result = -0.5 * dist
    result = torch.exp(result) / (sig * math.sqrt(2.0 * math.pi))
    return result


def matrix_nms_torch(boxes, method='gauss', sigma=0.5):
    # boxes:    (#boxes, 4), boxes in descending order
    assert method in ('linear', 'gauss')
    n_boxes = boxes.shape[0]

    iou_mat = ops.box_iou(boxes, boxes)
    iou_mat = iou_mat.triu(diagonal=1)
    # iou_mat:  (#boxes, #boxes)

    iou_max = torch.max(iou_mat, dim=0, keepdim=True)[0]
    iou_max_mat = torch.cat([iou_max] * n_boxes, dim=0)
    iou_max_mat = torch.transpose(iou_max_mat, dim0=0, dim1=1)
    # iou_max:  (#boxes, #boxes)

    if method == 'gauss':
        decay = torch.exp(-1 * (iou_mat ** 2 - iou_max_mat ** 2) / (sigma + 1e-16))
    else:
        decay = ((1 - iou_mat) / (1 - iou_max_mat)) ** sigma
    decay = torch.min(decay, dim=0)[0]
    # decay:    (#boxes)
    return decay
