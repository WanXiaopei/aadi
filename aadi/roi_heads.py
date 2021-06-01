from typing import List

import torch
from detectron2.structures import ImageList, Boxes, Instances, pairwise_iou
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads

from .utils import get_aligned_pooler, label_and_sample_proposals
from .lazy_fast_rcnn import LazyFastRCNNOutputLayers


@ROI_HEADS_REGISTRY.register()
class LazyRoIHeads(StandardROIHeads):
    @torch.no_grad()
    def label_and_sample_proposals(
            self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        return label_and_sample_proposals(self, proposals, targets)

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        ret["box_predictor"] = LazyFastRCNNOutputLayers(
            cfg, ret["box_head"].output_shape,
            # The loss weight is set as Cascade RPN
            loss_weight={
                "loss_cls": 1.5,
                "loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT
            },
        )
        ret["box_in_features"] = cfg.MODEL.RPN.IN_FEATURES
        ret["box_pooler"] = get_aligned_pooler(
            cfg.MODEL.RPN, input_shape,
            output_size=cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
            sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO,
        )
        return ret


@ROI_HEADS_REGISTRY.register()
class LazyCascadeRoIHeads(CascadeROIHeads):
    @torch.no_grad()
    def label_and_sample_proposals(
            self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        return label_and_sample_proposals(self, proposals, targets)

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        box_predictors = []
        for bbox_reg_weights in cascade_bbox_reg_weights:
            box_predictors.append(
                LazyFastRCNNOutputLayers(
                    cfg, ret["box_heads"][0].output_shape,
                    box2box_transform=Box2BoxTransform(weights=bbox_reg_weights),
                    loss_weight={
                        "loss_cls": 1.5,
                        "loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT
                    },
                )
            )
        ret["box_predictors"] = box_predictors
        ret["box_in_features"] = cfg.MODEL.RPN.IN_FEATURES
        ret["box_pooler"] = get_aligned_pooler(
            cfg.MODEL.RPN, input_shape,
            output_size=cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
            sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO,
        )
        return ret

    def _match_and_label_boxes(self, proposals, stage, targets):
        return label_and_sample_proposals(self, proposals, targets, False, False, stage)