import numpy as np
import torch
from torchvision.ops import nms

from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh
from mmdet.core.bbox.samplers import PseudoSampler
from ..builder import HEADS
from .cascade_roi_head import CascadeRoIHead


@HEADS.register_module()
class DRMMHead(CascadeRoIHead):
    r"""The RoIHead for `Sparse R-CNN: End-to-End Object Detection with
    Learnable Proposals <https://arxiv.org/abs/2011.12450>`_

    Args:
        num_stages (int): Number of stage whole iterative process.
            Defaults to 6.
        stage_loss_weights (Tuple[float]): The loss
            weight of each stage. By default all stages have
            the same weight 1.
        bbox_roi_extractor (dict): Config of box roi extractor.
        bbox_head (dict): Config of box head.
        train_cfg (dict, optional): Configuration information in train stage.
            Defaults to None.
        test_cfg (dict, optional): Configuration information in test stage.
            Defaults to None.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    """

    def __init__(self,
                 num_stages=6,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 proposal_feature_channel=256,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):

        self.num_classes = 80

        if bbox_roi_extractor is None:
            bbox_roi_extractor = dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32])

        if bbox_head is None:
            bbox_head = dict(
                type='DII_DRMMHead',
                num_classes=self.num_classes,
                num_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                hidden_channels=256,
                dropout=0.0,
                roi_feat_size=7,
                ffn_act_cfg=dict(type='ReLU', inplace=True))


        assert len(stage_loss_weights) == num_stages
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.proposal_feature_channel = proposal_feature_channel

        super(DRMMHead, self).__init__(
            num_stages,
            stage_loss_weights,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        # train_cfg would be None when run the test.py
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(self.bbox_sampler[stage], PseudoSampler), \
                    'Sparse R-CNN only support `PseudoSampler`'

    def _bbox_forward(self, stage, x, rois, object_feats, img_metas, fg, prob):
        """Box head forward function used in both training and testing. Returns
        all regression, classification results and a intermediate feature.

        Args:
            stage (int): The index of current stage in
                iterative process.
            x (List[Tensor]): List of FPN features
            rois (Tensor): Rois in total batch. With shape (num_proposal, 5).
                the last dimension 5 represents (img_index, x1, y1, x2, y2).
            object_feats (Tensor): The object feature extracted from
                the previous stage.
            img_metas (dict): meta information of images.

        Returns:
            dict[str, Tensor]: a dictionary of bbox head outputs,
                Containing the following results:

                    - cls_score (Tensor): The score of each class, has
                      shape (batch_size, num_proposals, num_classes)
                      when use focal loss or
                      (batch_size, num_proposals, num_classes+1)
                      otherwise.
                    - decode_bbox_pred (Tensor): The regression results
                      with shape (batch_size, num_proposal, 4).
                      The last dimension 4 represents
                      [tl_x, tl_y, br_x, br_y].
                    - object_feats (Tensor): The object feature extracted
                      from current stage
                    - detach_cls_score_list (list[Tensor]): The detached
                      classification results, length is batch_size, and
                      each tensor has shape (num_proposal, num_classes).
                    - detach_proposal_list (list[tensor]): The detached
                      regression results, length is batch_size, and each
                      tensor has shape (num_proposal, 4). The last
                      dimension 4 represents [tl_x, tl_y, br_x, br_y].
        """
        num_imgs = len(img_metas)
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], rois)
        output_dict = bbox_head(rois, bbox_feats, object_feats, fg, prob)
        return output_dict

    def forward_train(self,
                      x,
                      proposal_boxes,
                      proposal_features,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      imgs_whwh=None,
                      gt_masks=None):
        """Forward function in training stage.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposals (Tensor): Decoded proposal bboxes, has shape
                (batch_size, num_proposals, 4)
            proposal_features (Tensor): Expanded proposal
                features, has shape
                (batch_size, num_proposals, proposal_feature_channel)
            img_metas (list[dict]): list of image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip',
                and may also contain 'filename', 'ori_shape',
                'pad_shape', and 'img_norm_cfg'. For details on the
                values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            imgs_whwh (Tensor): Tensor with shape (batch_size, 4),
                    the dimension means
                    [img_width,img_height, img_width, img_height].
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components of all stage.
        """
        # proposal_boxes = torch.cat([proposal_boxes[:, :100].detach(), proposal_boxes[:, 100:]], dim=1)
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        object_feats = proposal_features
        all_stage_loss = {}

        batch_size, n_proposal = object_feats.shape[0:2]
        fg = torch.ones((batch_size * n_proposal, 1)).float().cuda()
        prob = torch.ones((batch_size * n_proposal, 80)).float().cuda()

        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list)

            # output_dict = self._bbox_forward(stage, x, rois, object_feats, img_metas)
            output_dict = self._bbox_forward(stage, x, rois, object_feats, img_metas, fg, prob)

            # stage_loss_dict = self.bbox_head[stage].loss(output_dict, gt_bboxes, gt_labels)
            stage_loss_dict = self.bbox_head[stage].loss(output_dict, gt_bboxes, gt_labels, stage)

            for key, value in stage_loss_dict.items():
                all_stage_loss[f'stage{stage}_{key}'] = \
                    value * self.stage_loss_weights[stage]

            proposal_list = [loc_s.detach() for loc_s in output_dict['loc']]
            object_feats = output_dict['obj_feat']
            prob = output_dict['prob'].view(-1, output_dict['prob'].shape[2]).detach()
            fg = output_dict['fg'].view(-1, output_dict['fg'].shape[2]).detach()

        return all_stage_loss

    def simple_test(self,
                    x,
                    proposal_boxes,
                    proposal_features,
                    img_metas,
                    imgs_whwh,
                    rescale=False):
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        object_feats = proposal_features

        n_proposal = object_feats.shape[1]
        fg = torch.ones((n_proposal, 1)).float().cuda()
        prob = torch.ones((n_proposal, 80)).float().cuda()

        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list)

            # output_dict = self._bbox_forward(stage, x, rois, object_feats, img_metas)
            output_dict = self._bbox_forward(stage, x, rois, object_feats, img_metas, fg, prob)

            proposal_list = [loc_s.detach() for loc_s in output_dict['loc']]
            object_feats = output_dict['obj_feat']
            prob = output_dict['prob'].view(-1, output_dict['prob'].shape[2]).detach()
            fg = output_dict['fg'].view(-1, output_dict['fg'].shape[2]).detach()

        bbox_list = self.get_bboxes(output_dict, img_metas, rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def get_bboxes(self, output_dict, img_metas, rescale=False):
        bbox = output_dict['loc']
        conf = output_dict['prob']
        if conf.shape[2] == (self.num_classes + 1):
            conf = conf[:, :, 1:]
        """
            bbox (Tensor):   (B, N_BOX, 4)
            conf (Tensor):   (B, N_BOX, CLS)

            Returns:
                list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                    The first item is an (n, 5) tensor, where 5 represent
                    (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                    The shape of the second tensor in the tuple is (n,), and
                    each element represents the class label of the corresponding
                    box.
        """
        B, _, C = conf.shape

        output_keys = output_dict.keys()
        if 'fg' in output_keys:
            conf = conf * output_dict['fg']

        result = list()
        for bbox_i, conf_i, img_metas_i in zip(bbox, conf, img_metas):
            # bbox_i (Tensor):      (N_BOX, 4)
            # conf_i (Tensor):      (N_BOX, CLS)
            # label_i (Tensor):     (N_BOX, CLS)

            bbox_i[:, [0, 2]] = bbox_i[:, [0, 2]] / img_metas_i['scale_factor'][0]
            bbox_i[:, [1, 3]] = bbox_i[:, [1, 3]] / img_metas_i['scale_factor'][1]

            res_bbox_i = bbox_i
            res_conf_i, topk_indices = conf_i.flatten(
                0, 1).topk(
                # self.test_cfg.max_per_img, sorted=False)
                100, sorted = False)
            res_label_i = topk_indices % 80
            res_bbox_i = res_bbox_i[topk_indices // 80]

            if len(res_bbox_i) > 100:
                res_conf_i, keep_indices = torch.topk(res_conf_i, k=100, dim=0, sorted=True)
                res_bbox_i = res_bbox_i[keep_indices]
                res_label_i = res_label_i[keep_indices]
            result_i = (torch.cat([res_bbox_i, res_conf_i.unsqueeze(1)], dim=1), res_label_i)
            result.append(result_i)
        return result

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        raise NotImplementedError('Sparse R-CNN does not support `aug_test`')

    def forward_dummy(self, x, proposal_boxes, proposal_features, img_metas):
        """Dummy forward function when do the flops computing."""
        all_stage_bbox_results = []
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        object_feats = proposal_features
        if self.with_bbox:
            for stage in range(self.num_stages):
                rois = bbox2roi(proposal_list)
                bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                                  img_metas)

                all_stage_bbox_results.append(bbox_results)
                proposal_list = bbox_results['detach_proposal_list']
                object_feats = bbox_results['object_feats']
        return all_stage_bbox_results
