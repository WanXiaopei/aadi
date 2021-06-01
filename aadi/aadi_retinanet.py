import math
import logging
import copy
from typing import List, Tuple, Dict

import torch
import numpy as np
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from fvcore.nn import sigmoid_focal_loss, smooth_l1_loss, giou_loss

from detectron2.layers import ShapeSpec, cat, nonzero_tuple, batched_nms
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.config import configurable
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.meta_arch.retinanet import RetinaNetHead, RetinaNet
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.meta_arch import GeneralizedRCNN

from .layers import dilated_convolution
from .utils import iou_neg_balanced_sampling, get_aligned_pooler, compute_iou, varifocal_loss

logger = logging.getLogger(__name__)


def create_proposals_from_boxes(boxes, scores, image_sizes, filter_non_valid=True):
    boxes = [Boxes(b.detach()) for b in boxes]
    proposals = []
    for boxes_per_image, scores_per_image, image_size in zip(boxes, scores, image_sizes):
        scores_per_image = scores_per_image.max(dim=1)[0]

        if filter_non_valid:
            valid_mask = torch.isfinite(boxes_per_image.tensor).all(dim=1)
            boxes_per_image = boxes_per_image[valid_mask]
            scores_per_image = scores_per_image[valid_mask]

            boxes_per_image.clip(image_size)
            valid_mask = boxes_per_image.nonempty()
            boxes_per_image = boxes_per_image[valid_mask]
            scores_per_image = scores_per_image[valid_mask]

        prop = Instances(image_size)
        prop.proposal_boxes = boxes_per_image
        prop.objectness_logits = scores_per_image
        proposals.append(prop)
    return proposals


@META_ARCH_REGISTRY.register()
class AADIRetinaNet(RetinaNet):
    @configurable
    def __init__(
            self,
            proposal_append_gt: bool = False,
            proposal_append_hand_anchors: bool = False,
            candidate_dilation: List[int] = None,
            vfl_enabled: bool = False,
            **kwargs,
    ):
        """
        NOTE: this interface is experimental.
        """
        super(AADIRetinaNet, self).__init__(**kwargs)
        self.proposal_append_gt = proposal_append_gt
        self.proposal_append_hand_anchors = proposal_append_hand_anchors
        self.candidate_dilation = candidate_dilation
        self.vfl_enabled = vfl_enabled

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        backbone_shape = ret["backbone"].output_shape()
        feature_shapes = [backbone_shape[f] for f in cfg.MODEL.RETINANET.IN_FEATURES]
        ret["head"] = AADIRetinaNetHead(cfg, backbone_shape)
        ret["proposal_append_gt"] = cfg.MODEL.RETINANET.PROPOSAL_APPEND_GT
        ret["proposal_append_hand_anchors"] = cfg.MODEL.RETINANET.PROPOSAL_APPEND_HAND_ANCHORS
        ret["candidate_dilation"] = cfg.MODEL.RETINANET.CANDIDATE_DILATION
        anchor_generator = []
        kernel_size = cfg.MODEL.RETINANET.CONV_KERNEL_SIZE
        anchor_size = np.array([shape.stride * kernel_size for shape in feature_shapes]).reshape(-1, 1)
        for dilation in ret["candidate_dilation"]:
            sizes = anchor_size * dilation
            anchor_generator.append(
                DefaultAnchorGenerator(cfg, feature_shapes, sizes=sizes.tolist(), aspect_ratios=[[1.0]])
            )
        ret["anchor_generator"] = nn.ModuleList(anchor_generator)
        return ret

    @torch.no_grad()
    def obtain_refined_proposals(
            self,
            anchors: List[Boxes],
            cls_features: List[Tensor],
            box_features: List[Tensor],
            image_sizes,
    ):

        proposals = []
        for dilation, cur_anchors in zip(self.candidate_dilation, anchors):
            pred_cls_logits, pred_box_deltas, iou_logits = self.head.obtain_conv_preds(cls_features, box_features, dilation)

            pred_cls_scores = cat(
                [logits.permute(0, 2, 3, 1).flatten(1, 2).sigmoid() for logits in pred_cls_logits], dim=1)
            pred_box_deltas = cat([deltas.permute(0, 2, 3, 1).flatten(1, 2) for deltas in pred_box_deltas], dim=1)
            pred_boxes = [self.box2box_transform.apply_deltas(deltas, cur_anchors.tensor) for deltas in pred_box_deltas]
            prop = create_proposals_from_boxes(pred_boxes, pred_cls_scores, image_sizes, self.training)
            proposals.append(prop)

        return [Instances.cat(prop) for prop in zip(*proposals)]


    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]

        anchors = [Boxes.cat(generator(features)) for generator in self.anchor_generator]
        cls_features, box_features = self.head.obtain_retina_features(features)
        proposals = self.obtain_refined_proposals(anchors, cls_features, box_features, images.image_sizes)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            proposals = self.label_and_sample_boxes(proposals, gt_instances)
            if self.proposal_append_hand_anchors:
                anchors = Boxes.cat(anchors)
                for idx, targets_per_image in enumerate(gt_instances):
                    iou_matrix = pairwise_iou(targets_per_image.gt_boxes, anchors)
                    max_vals, max_idxs = iou_matrix.max(dim=1)
                    hd_proposals = copy.deepcopy(targets_per_image)
                    hd_proposals.proposal_boxes = anchors[max_idxs]
                    hd_proposals.objectness_logits = torch.ones_like(max_vals)
                    hd_proposals.gt_classes = hd_proposals.gt_classes * 0 - 1
                    proposals[idx] = Instances.cat([proposals[idx], hd_proposals])
        else:
            proposals = [p[p.objectness_logits.topk(self.test_topk_candidates)[1]] for p in proposals]

        pred_cls_logits, pred_box_deltas, iou_scores = self.head.obtain_fc_preds(
            [prop.proposal_boxes for prop in proposals], cls_features, box_features
        )
        if self.training:
            return self.losses_v1(pred_cls_logits, pred_box_deltas, iou_scores, proposals)

        num_proposal_per_image = [len(prop) for prop in proposals]
        pred_boxes = self.box2box_transform.apply_deltas(
            pred_box_deltas, cat([prop.proposal_boxes.tensor for prop in proposals])
        )
        pred_scores = torch.zeros(pred_cls_logits.shape[0], pred_cls_logits.shape[1]+1).to(pred_cls_logits)
        pred_cls_scores = pred_cls_logits.sigmoid()
        if iou_scores is not None:
            pred_cls_scores *= iou_scores.sigmoid().view(-1, 1)
        pred_scores[:, :-1] = pred_cls_scores
        image_shapes = [x.image_size for x in proposals]
        results, _ = fast_rcnn_inference(
            pred_boxes.split(num_proposal_per_image),
            pred_scores.split(num_proposal_per_image),
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.max_detections_per_image,
        )
        return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)

    def losses_v1(self, pred_cls_logits, pred_box_deltas, iou_logits, proposals):
        num_images = len(proposals)
        gt_classes = cat([prop.gt_classes for prop in proposals])
        gt_boxes = cat([prop.gt_boxes.tensor for prop in proposals])
        proposal_boxes = cat([prop.proposal_boxes.tensor for prop in proposals])

        pos_mask = gt_classes != self.num_classes
        num_pos = pos_mask.sum().item() * 1. / num_images
        self.loss_normalizer *= self.loss_normalizer_momentum
        self.loss_normalizer += (1 - self.loss_normalizer_momentum) * max(num_pos, 1)

        # gt_classes[gt_ious < self.anchor_matcher.thresholds[1]] = self.num_classes
        pred_boxes = self.box2box_transform.apply_deltas(pred_box_deltas[pos_mask], proposal_boxes[pos_mask])
        valid_cls_idx = gt_classes >= 0
        if self.vfl_enabled:
            gt_ious = compute_iou(pred_boxes, gt_boxes[pos_mask])
            iou_target = torch.zeros_like(pred_cls_logits)
            pos_idx = nonzero_tuple((gt_classes >= 0) & (gt_classes != self.num_classes))[0]
            iou_target[pos_idx, gt_classes[pos_idx]] = gt_ious
            loss_cls = varifocal_loss(pred_cls_logits, iou_target, reduction="sum")
        else:
            gt_labels_target = F.one_hot(gt_classes[valid_cls_idx], self.num_classes + 1)[:, :-1]
            loss_cls = sigmoid_focal_loss(
                pred_cls_logits[valid_cls_idx],
                gt_labels_target.to(pred_cls_logits.dtype),
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            )

        if self.box_reg_loss_type == "smooth_l1":
            gt_deltas = self.box2box_transform.get_deltas(proposal_boxes[pos_mask], gt_boxes[pos_mask])
            loss_box_reg = smooth_l1_loss(
                pred_box_deltas[pos_mask],
                gt_deltas,
                beta=self.smooth_l1_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "giou":
            loss_box_reg = giou_loss(pred_boxes, gt_boxes[pos_mask], reduction="sum")
        else:
            raise NotImplementedError

        normalizer = self.loss_normalizer * num_images
        losses = {
            "loss_cls": loss_cls / normalizer,
            "loss_box_reg": loss_box_reg / normalizer,
        }

        if iou_logits is not None:
            with torch.no_grad():
                gt_ious = compute_iou(pred_boxes, gt_boxes[pos_mask])
            iou_losses = F.binary_cross_entropy_with_logits(
                iou_logits.flatten()[pos_mask], gt_ious, reduction="mean"
            )
            losses.update({"iou_loss": iou_losses})
        return losses

    @torch.no_grad()
    def label_and_sample_boxes(self, proposals, targets, apply_nms=True, **kwargs):
        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        num_recall = []
        iou_thresh = self.anchor_matcher.thresholds[1]
        batch_size_per_image = 4096
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, gt_labels_i = self.anchor_matcher(match_quality_matrix)
            gt_labels_i = gt_labels_i.to(device=targets_per_image.gt_boxes.device)

            max_ious, max_idxs = match_quality_matrix.max(dim=1)
            num_recall.append((max_ious >= iou_thresh).sum().item())

            positive = torch.nonzero(gt_labels_i == 1, as_tuple=True)
            max_ious, max_idxs = match_quality_matrix.max(dim=0)
            if apply_nms and len(positive) > 1 and get_event_storage().iter > 1000:
                keep = batched_nms(
                    proposals_per_image.proposal_boxes[positive].tensor, max_ious[positive], max_idxs[positive], 0.9)
                gt_labels_i[positive] = -1
                gt_labels_i[positive[keep]] = 1

            # gt_labels_i[instance_matcher(match_quality_matrix, top_k=4, iou_low_thresh=0.4)]
            gt_classes = targets_per_image.gt_classes[matched_idxs]

            neg_idxs = torch.nonzero(gt_labels_i == 0, as_tuple=True)[0]
            neg_ious = max_ious[neg_idxs]

            # RetinaNet use focal loss as loss function, however, use all samples to train the classifier is
            # not efficient in our implementation, so we only keep the hard negative samples.
            if neg_idxs.numel() > 0:
                num_neg = min(batch_size_per_image, len(proposals_per_image)) - (gt_labels_i == 1).sum().item()
                neg_idxs = iou_neg_balanced_sampling(neg_ious, neg_idxs, max(num_neg, 1), num_bins=3)

            gt_labels_i[gt_labels_i == 0] = -1
            gt_labels_i[neg_idxs] = 0

            gt_classes[neg_idxs] = self.num_classes
            sampled_idxs = torch.nonzero(gt_labels_i >= 0, as_tuple=True)[0]
            gt_classes = gt_classes[sampled_idxs]

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])

            if self.proposal_append_gt:
                gt_proposals = copy.deepcopy(targets_per_image)
                gt_proposals.proposal_boxes = gt_proposals.gt_boxes
                gt_proposals.objectness_logits = torch.ones_like(gt_proposals.gt_boxes.tensor[:, 0])
                proposals_per_image = Instances.cat([proposals_per_image, gt_proposals])

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))
        storage.put_scalar("roi_head/recall", sum(num_recall) * 1. / sum([len(gt) for gt in targets]))

        return proposals_with_gt


class AADIRetinaNetHead(RetinaNetHead):
    def __init__(self, cfg, input_shape):
        feature_shape = [input_shape[f] for f in cfg.MODEL.RETINANET.IN_FEATURES]
        super().__init__(cfg, feature_shape)
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
        kernel_size = cfg.MODEL.RETINANET.CONV_KERNEL_SIZE
        self.iou_enabled = cfg.MODEL.RETINANET.IOU_ENABLED

        conv_dims = feature_shape[0].channels
        self.cls_feature = nn.Conv2d(conv_dims, conv_dims, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bbox_feature = nn.Conv2d(conv_dims, conv_dims, kernel_size=kernel_size, padding=kernel_size // 2)

        self.cls_score = nn.Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        self.bbox_pred = nn.Conv2d(conv_dims, 4, kernel_size=1, stride=1, padding=0)

        # Initialization
        for layer in [self.cls_feature, self.bbox_feature, self.cls_score, self.bbox_pred]:
            torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)

        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

        self.box_pooler = get_aligned_pooler(
            cfg.MODEL.RETINANET, input_shape,
            kernel_size=kernel_size,
            output_size=kernel_size,
        )

        if self.iou_enabled:
            self.iou_score = nn.Conv2d(conv_dims, 1, kernel_size=1, stride=1, padding=0)
            torch.nn.init.normal_(self.iou_score.weight, mean=0, std=0.01)
            torch.nn.init.constant_(self.iou_score.bias, 0)

    def forward(self, features: List[Tensor]):
        raise NotImplementedError

    def obtain_retina_features(self, features: List[Tensor]):
        cls_features = []
        bbox_features = []
        for idx, feature in enumerate(features):
            cls_features.append(self.cls_subnet(feature))
            bbox_features.append(self.bbox_subnet(feature))
        return cls_features, bbox_features

    @torch.no_grad()
    def obtain_conv_preds(self, cls_features: List[Tensor], bbox_features: List[Tensor], dilation: int = 2):
        logits = []
        bbox_reg = []
        iou_logits = []
        for cls_feature, bbox_feature in zip(cls_features, bbox_features):
            cls_feature = F.relu_(dilated_convolution(self.cls_feature, cls_feature, dilation))
            bbox_feature = F.relu_(dilated_convolution(self.bbox_feature, bbox_feature, dilation))
            logits.append(self.cls_score(cls_feature))
            bbox_reg.append(self.bbox_pred(bbox_feature))
            if self.iou_enabled:
                iou_logits.append(self.iou_score(bbox_feature))

        return logits, bbox_reg, iou_logits

    def obtain_fc_preds(self, pred_boxes: List[Boxes], cls_features: List[Tensor], bbox_features: List[Tensor]):
        cls_features = self.box_pooler(cls_features, pred_boxes).flatten(1)
        box_features = self.box_pooler(bbox_features, pred_boxes).flatten(1)

        cls_features = F.relu_(F.linear(cls_features, self.cls_feature.weight.flatten(1), self.cls_feature.bias))
        box_features = F.relu_(F.linear(box_features, self.bbox_feature.weight.flatten(1), self.bbox_feature.bias))

        cls_logits = F.linear(cls_features, self.cls_score.weight.flatten(1), self.cls_score.bias)
        bbox_reg = F.linear(box_features, self.bbox_pred.weight.flatten(1), self.bbox_pred.bias)
        if self.iou_enabled:
            iou_logits = F.linear(box_features, self.iou_score.weight.flatten(1), self.iou_score.bias)
        else:
            iou_logits = None

        return cls_logits, bbox_reg, iou_logits
