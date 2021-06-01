import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from fvcore.nn import smooth_l1_loss, sigmoid_focal_loss, giou_loss

import math
import numpy as np
import torch
from torch import nn
from torchvision.ops import nms
from detectron2.layers import ShapeSpec, cat, batched_nms
from detectron2.structures import Boxes, Instances, pairwise_iou, ImageList
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator import RPN, PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator.proposal_utils import find_top_rpn_proposals
from detectron2.modeling.backbone import BACKBONE_REGISTRY, build_resnet_backbone, FPN
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.events import get_event_storage
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator

from .utils import compute_iou, create_proposals, get_aligned_pooler
from .layers import dilated_convolution


def valid_anchors(pre_gt_labels, pre_gt_boxes, ref_anchors):
    gt_labels = []
    gt_boxes = []
    post_ref_anchors = []
    for gt_label, gt_box, ref_anchor in zip(pre_gt_labels, pre_gt_boxes, ref_anchors):
        valid = gt_label != -1
        gt_labels.append(gt_label[valid])
        gt_boxes.append(gt_box[valid])
        post_ref_anchors.append(ref_anchor[valid])
    return gt_labels, gt_boxes, post_ref_anchors


@PROPOSAL_GENERATOR_REGISTRY.register()
class AADIRPN(RPN):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg, input_shape)

        in_features = cfg.MODEL.RPN.IN_FEATURES
        self.box_in_features = in_features
        self.candidate_dilation = cfg.MODEL.RPN.CANDIDATE_DILATION
        assert len(self.candidate_dilation) > 0, "got empty candidate_dilation"

        self.box_pooler = get_aligned_pooler(cfg.MODEL.RPN, input_shape)

        self.num_positive = self.batch_size_per_image
        self.momentum = 0.9
        self.append_anchors = cfg.MODEL.RPN.APPEND_ANCHORS
        self.post_topk = {
            True: cfg.MODEL.ROI_HEADS.POST_NMS_TOPK_TRAIN,
            False: cfg.MODEL.ROI_HEADS.POST_NMS_TOPK_TEST,
        }

        feature_shapes = [input_shape[f] for f in self.box_in_features]
        anchor_size = np.array([shape.stride * 3 for shape in feature_shapes]).reshape(-1, 1)
        anchor_generator = []
        for dilation in self.candidate_dilation:
            sizes = anchor_size * dilation
            anchor_generator.append(
                DefaultAnchorGenerator(cfg, feature_shapes, sizes=sizes.tolist(), aspect_ratios=[[1.0]])
            )
        self.anchor_generator = nn.ModuleList(anchor_generator)

    def forward_conv(self, features):
        return F.linear(features, self.rpn_head.conv.weight.flatten(1), self.rpn_head.conv.bias)

    def forward_objectness(self, features):
        return F.linear(
            features, self.rpn_head.objectness_logits.weight.flatten(1), self.rpn_head.objectness_logits.bias)

    def forward_box_deltas(self, features):
        return F.linear(features, self.rpn_head.anchor_deltas.weight.flatten(1), self.rpn_head.anchor_deltas.bias)

    def adaptive_dilation_roi_head(self, features, dilation=2):
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = F.relu(dilated_convolution(self.rpn_head.conv, x, dilation))
            pred_objectness_logits.append(self.rpn_head.objectness_logits(t))
            pred_anchor_deltas.append(self.rpn_head.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas

    @torch.no_grad()
    def _obtain_refined_anchors(self, anchors, features, dilation=2):
        pred_objectness_logits, pred_anchor_deltas = self.adaptive_dilation_roi_head(features, dilation)
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, 4, x.shape[-2], x.shape[-1])
                .permute(0, 3, 4, 1, 2)
                .flatten(1, -2)
            for x in pred_anchor_deltas
        ]
        pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
        top_proposals = pred_proposals
        if self.training:
            pred_proposals = cat(pred_proposals, dim=1).split(1, dim=0)
            pred_proposals = [Boxes(prop[0]) for prop in pred_proposals]
        return top_proposals, pred_objectness_logits, pred_proposals

    def append_pos_anchors_to_proposals(self, list_anchors, ref_anchors, pre_gt_labels, pre_gt_boxes, gt_instances):
        gt_boxes = [x.gt_boxes for x in gt_instances]
        top_anchors = []
        gt_labels = []
        gt_matched_boxes = []
        num_append = []

        anchors = [Boxes.cat(boxes) for boxes in zip(*list_anchors)]
        anchors = Boxes.cat(anchors)

        thresh = self.anchor_matcher.thresholds[1]
        for gt_boxes_i in gt_boxes:
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            max_ious, positive = match_quality_matrix.max(dim=1)
            label = torch.ones(positive.numel()).to(pre_gt_labels[0])
            label[max_ious < thresh] = -1
            num_append.append(positive.numel())
            top_anchors.append(anchors[positive])
            gt_labels.append(label)
            gt_matched_boxes.append(gt_boxes_i)

        get_event_storage().put_scalar("rpn/num_append", sum(num_append) * 1. / len(num_append))
        ref_anchors = [Boxes.cat([a1, a2]) for a1, a2 in zip(ref_anchors, top_anchors)]
        pre_gt_labels = [cat([g1, g2]) for g1, g2 in zip(pre_gt_labels, gt_labels)]
        pre_gt_boxes = [cat([b1, b2.tensor]) for b1, b2 in zip(pre_gt_boxes, gt_matched_boxes)]

        return ref_anchors, pre_gt_labels, pre_gt_boxes

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            gt_instances: Optional[List[Instances]] = None,
    ):
        aug_features = [features[f] for f in self.box_in_features]
        features = [features[f] for f in self.in_features]

        list_anchors = []
        for anchor_generator in self.anchor_generator:
            list_anchors.append(anchor_generator(features))

        top_proposals, pred_objectness_logits, ref_anchors = [], [], []
        for idx, dilation in enumerate(self.candidate_dilation):
            _top_proposals, _pred_objectness_logits, _ref_anchors =\
                self._obtain_refined_anchors(list_anchors[idx], features, dilation)
            top_proposals.append(_top_proposals)
            pred_objectness_logits.append(_pred_objectness_logits)
            ref_anchors.append(_ref_anchors)

        top_proposals = [cat(prop, dim=1) for prop in zip(*top_proposals)]
        pred_objectness_logits = [cat(logits, dim=-1) for logits in zip(*pred_objectness_logits)]
        top_proposals = find_top_rpn_proposals(
            top_proposals,
            pred_objectness_logits,
            images.image_sizes,
            self.nms_thresh,
            self.pre_nms_topk[self.training],
            self.post_nms_topk[self.training],
            self.min_box_size,
            self.training,
        )

        if self.training:
            ref_anchors = [Boxes.cat(anchors) for anchors in zip(*ref_anchors)]
            ref_anchors = [anchors[anchors.nonempty()] for anchors in ref_anchors]

            pre_gt_labels, pre_gt_boxes = self.label_ref_anchors(ref_anchors, gt_instances)
            if self.append_anchors:
                ref_anchors, pre_gt_labels, pre_gt_boxes = self.append_pos_anchors_to_proposals(
                    list_anchors, ref_anchors, pre_gt_labels, pre_gt_boxes, gt_instances)

            gt_labels, gt_boxes, post_ref_anchors = valid_anchors(pre_gt_labels, pre_gt_boxes, ref_anchors)

            box_features = self.box_pooler(aug_features, post_ref_anchors)
            box_features = torch.flatten(box_features, start_dim=1)
            box_features = F.relu_(self.forward_conv(box_features))

            pred_cls_logits = self.forward_objectness(box_features).flatten()
            pred_box_deltas = self.forward_box_deltas(box_features)
            losses = self.losses_v1(post_ref_anchors, pred_cls_logits, gt_labels, pred_box_deltas, gt_boxes)
        else:
            losses = {}

        with torch.no_grad():
            num_proposal_per_image = [len(prop) for prop in top_proposals]
            refs = [prop.proposal_boxes for prop in top_proposals]
            box_features = self.box_pooler(aug_features, refs)
            box_features = torch.flatten(box_features, start_dim=1)
            box_features = F.relu_(self.forward_conv(box_features))
            pred_cls_logits = self.forward_objectness(box_features).flatten().split(num_proposal_per_image)
            pred_box_deltas = self.forward_box_deltas(box_features)
            proposal_boxes = cat([prop.proposal_boxes.tensor for prop in top_proposals])
            pred_boxes = self.box2box_transform.apply_deltas(pred_box_deltas, proposal_boxes)
            pred_boxes = pred_boxes.split(num_proposal_per_image)
            proposals = create_proposals(pred_cls_logits, pred_boxes, images.image_sizes)

        ret = []
        for proposal in proposals:
            if len(proposal) > self.post_topk[self.training]:
                _, keep = proposal.objectness_logits.sort(descending=True)
                proposal = proposal[keep[:self.post_topk[self.training]]]
            ret.append(proposal)
        return ret, losses

    def label_ref_anchors(self, ref_anchors, gt_instances):
        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        for image_size_i, anchors, gt_boxes_i in zip(image_sizes, ref_anchors, gt_boxes):
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            max_ious, max_idxs = match_quality_matrix.max(dim=0)
            del match_quality_matrix

            # We do not apply NMS for ref_anchors because it will slow the training speed,
            # however, it will lead to that many proposals are crowd, so we apply batched_nms here
            # to remove redundant proposals.
            positive = torch.nonzero(gt_labels_i == 1, as_tuple=True)[0]
            if len(positive) > 1 and get_event_storage().iter > 200:
                keep = batched_nms(anchors[positive].tensor, max_ious[positive], max_idxs[positive], 0.9)
                gt_labels_i[positive] = -1
                gt_labels_i[positive[keep]] = 1

            if self.anchor_boundary_thresh >= 0:
                anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                gt_labels_i[~anchors_inside_image] = -1

            gt_labels_i = self._subsample_labels(gt_labels_i)
            if len(gt_boxes_i) == 0:
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes

    @torch.jit.unused
    def smooth_l1(
            self,
            anchors: List[Boxes],
            gt_labels: List[torch.Tensor],
            pred_anchor_deltas,
            gt_boxes: List[torch.Tensor],
    ):
        num_images = len(gt_labels)
        cat_gt_labels = cat(gt_labels)  # (N, sum(Hi*Wi*Ai))
        pos_mask = cat_gt_labels != 0

        gt_anchor_deltas = []
        for anchors_i, gt_labels_i, gt_boxes_i in zip(anchors, gt_labels, gt_boxes):
            fg_mask = gt_labels_i != 0
            anchors_i = anchors_i[fg_mask]
            if len(anchors_i) == 0:
                continue
            gt_anchor_deltas.append(self.box2box_transform.get_deltas(anchors_i.tensor, gt_boxes_i[fg_mask]))

        if len(gt_anchor_deltas) > 0:
            gt_anchor_deltas = torch.cat(gt_anchor_deltas)  # (N, R, 4)
            localization_loss = smooth_l1_loss(
                pred_anchor_deltas[pos_mask],
                gt_anchor_deltas,
                beta=self.smooth_l1_beta,
                reduction="sum",
            )
        else:
            localization_loss = cat(pred_anchor_deltas, dim=1).sum() * 0.
        return localization_loss / (6. * self.num_positive * num_images)

    @torch.jit.unused
    def giou(
            self,
            anchors: List[Boxes],
            gt_labels: List[torch.Tensor],
            pred_anchor_deltas,
            gt_boxes: List[torch.Tensor],
    ):
        num_images = len(gt_labels)
        cat_gt_labels = cat(gt_labels)  # (N, sum(Hi*Wi*Ai))
        pos_mask = cat_gt_labels == 1
        anchors = Boxes.cat(anchors).tensor
        gt_boxes = cat(gt_boxes)
        pred_boxes = self.box2box_transform.apply_deltas(pred_anchor_deltas, anchors)
        localization_loss = giou_loss(pred_boxes[pos_mask], gt_boxes[pos_mask], reduction="sum")
        return localization_loss / (3. * self.num_positive * num_images)

    @torch.jit.unused
    def bce_loss(
            self,
            gt_labels,
            pred_objectness_logits,
    ):
        num_images = len(gt_labels)
        cat_gt_labels = cat(gt_labels)
        valid_mask = cat_gt_labels >= 0
        objectness_loss = F.binary_cross_entropy_with_logits(
            pred_objectness_logits[valid_mask],
            cat_gt_labels[valid_mask].to(torch.float32),
            reduction="sum",
        ) / (6 * self.num_positive * num_images)
        return objectness_loss

    @torch.jit.unused
    def losses_v1(
            self,
            anchors: List[Boxes],
            pred_objectness_logits,
            gt_labels: List[torch.Tensor],
            pred_anchor_deltas,
            gt_boxes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        num_images = len(gt_labels)
        cat_gt_labels = cat(gt_labels).flatten()  # (N, sum(Hi*Wi*Ai))

        pos_mask = cat_gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (cat_gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("aug/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("aug/num_neg_anchors", num_neg_anchors / num_images)

        self.num_positive *= self.momentum
        self.num_positive += (1. - self.momentum) * max(pos_mask.sum() * 1. / num_images, 1.)

        localization_loss = getattr(self, self.box_reg_loss_type)(anchors, gt_labels, pred_anchor_deltas, gt_boxes, )
        objectness_loss = self.bce_loss(gt_labels, pred_objectness_logits)

        losses = {
            "loss_rpn_cls": objectness_loss,
            "loss_rpn_loc": localization_loss,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses
