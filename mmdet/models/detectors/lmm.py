import torch
from .. import DETECTORS
from .single_stage import SingleStageDetector
from mmdet.core import bbox2result


@DETECTORS.register_module()
class LMM(SingleStageDetector):
    """Implementation of Local Maximum Mixture Module"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(LMM, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)
        self.first_step = True

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        # if self.first_step:
        #     load_path = './work_dirs/lmm_r101_fpn_mstrain_2x_coco/epoch_24_net2.pth'
        #     net_dict = torch.load(load_path, map_location='cpu')
        #     self.load_state_dict(net_dict)
        #     print('[NETWORK] load: %s' % load_path)
        #     self.first_step = False
        return super(LMM, self).forward_train(img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)

    def simple_test(self, img, img_metas, rescale=False):
        # torch.save(self.state_dict(), './epoch_24_net2.pth')
        # exit()

        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head.forward_test(x, img_metas)
        # get origin input shape to support onnx dynamic shape
        if torch.onnx.is_in_onnx_export():
            # get shape as tensor
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results
