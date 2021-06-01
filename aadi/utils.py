from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import torch.nn.functional as F

from detectron2.structures import ImageList, Boxes, Instances, pairwise_iou
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.poolers import ROIPooler
from detectron2.utils.events import get_event_storage


def get_aligned_pooler(
        spec_cfg,
        input_shape,
        output_size: int = 3,
        sampling_ratio: int = 1,
        kernel_size: int = 3,
):
    in_features = spec_cfg.IN_FEATURES
    pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
    canonical_level = np.log2(min([input_shape[k].stride + 1 for k in in_features]))

    candidate_dilation = spec_cfg.CANDIDATE_DILATION
    anchor_size = min([input_shape[k].stride * kernel_size for k in in_features])
    min_dilation, max_dilation = min(candidate_dilation), max(candidate_dilation)
    assert min_dilation * 2 >= max_dilation, "{} * 2 vs {}".format(min_dilation, max_dilation)
    anchor_size = (min_dilation * 2. + max_dilation) / 4. * anchor_size

    return ROIPooler(
        output_size=output_size,
        scales=pooler_scales,
        sampling_ratio=sampling_ratio,
        pooler_type="ROIAlign",
        canonical_level=int(canonical_level),
        canonical_box_size=anchor_size,
    )


def varifocal_loss(pred,
                   target,
                   alpha=0.75,
                   gamma=2.0,
                   iou_weighted=True,
                   reduction='mean',):
    """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        alpha (float, optional): A balance factor for the negative part of
            Varifocal Loss, which is different from the alpha of Focal Loss.
            Defaults to 0.75.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        iou_weighted (bool, optional): Whether to weight the loss of the
            positive example with the iou target. Defaults to True.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
    """
    # pred and target should be of the same size
    assert pred.size() == target.size()
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    if iou_weighted:
        focal_weight = target * (target > 0.0).float() + \
            alpha * (pred_sigmoid - target).abs().pow(gamma) * \
            (target <= 0.0).float()
    else:
        focal_weight = (target > 0.0).float() + \
            alpha * (pred_sigmoid - target).abs().pow(gamma) * \
            (target <= 0.0).float()
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def quality_focal_loss(pred, target, beta=2.0):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert len(target) == 2, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, score = target

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos],
        reduction='none') * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdim=False)
    return loss


def compute_iou(
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
        eps: float = 1e-7,
) -> torch.Tensor:
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    iouk = intsctk / (unionk + eps)
    return iouk


def create_proposals(pred_cls_logits, pred_boxes, image_sizes):
    proposals = []
    for cls, box, image_size in zip(pred_cls_logits, pred_boxes, image_sizes):
        valid = (box[:, 2] > box[:, 0]) & (box[:, 3] > box[:, 1])
        boxes = Boxes(box[valid])
        boxes.clip(image_size)

        non_empty = boxes.nonempty()
        cls = cls[valid][non_empty]
        boxes = boxes[non_empty]

        prop = Instances(image_size)
        prop.proposal_boxes = boxes
        prop.objectness_logits = cls
        proposals.append(prop)
    return proposals


@torch.no_grad()
def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances],
        proposal_append_gt: bool = True,
        sampling: bool = True,
        stage_idx: int = 0,
) -> List[Instances]:
    gt_boxes = [x.gt_boxes for x in targets]
    if proposal_append_gt:
        proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

    proposals_with_gt = []

    num_fg_samples = []
    num_bg_samples = []
    num_recall_samples = []
    proposal_matcher = self.proposal_matcher if stage_idx == 0 else self.proposal_matchers[stage_idx]
    iou_thresh = proposal_matcher.thresholds[1]
    num_gt_boxes = [len(b) for b in gt_boxes]
    for proposals_per_image, targets_per_image in zip(proposals, targets):
        has_gt = len(targets_per_image) > 0
        match_quality_matrix = pairwise_iou(
            targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
        )
        if len(proposals_per_image) >= 2:
            max_iou = match_quality_matrix.topk(2, dim=1)[0][:, 1]
            num_recall_samples.append(
                (max_iou >= iou_thresh).sum().item()
            )
        else:
            num_recall_samples.append(0)

        matched_idxs, matched_labels = proposal_matcher(match_quality_matrix)

        if sampling:
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )
            proposals_per_image = proposals_per_image[sampled_idxs]
            matched_idxs = matched_idxs[sampled_idxs]
        else:
            gt_classes = targets_per_image.gt_classes[matched_idxs]
            gt_classes[matched_labels == 0] = self.num_classes
        proposals_per_image.gt_classes = gt_classes

        if has_gt:
            for (trg_name, trg_value) in targets_per_image.get_fields().items():
                if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                    proposals_per_image.set(trg_name, trg_value[matched_idxs])

        num_bg_samples.append((gt_classes == self.num_classes).sum().item())
        num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
        proposals_with_gt.append(proposals_per_image)

    storage = get_event_storage()
    storage.put_scalar(f"roi_head{stage_idx}/num_fg_samples", np.mean(num_fg_samples))
    storage.put_scalar(f"roi_head{stage_idx}/num_bg_samples", np.mean(num_bg_samples))
    storage.put_scalar(f"roi_head{stage_idx}/roi_recall", sum(num_recall_samples) * 1. / sum(num_gt_boxes))

    return proposals_with_gt


def iou_neg_balanced_sampling(max_overlaps, full_set, num_expected, floor_thr=0., num_bins=3):
    max_iou = max_overlaps.max().item()
    iou_interval = (max_iou - floor_thr) / num_bins
    per_num_expected = int(num_expected / num_bins)

    sampled_inds = []
    for i in range(num_bins):
        start_iou = max_iou - (i + 1) * iou_interval
        end_iou = max_iou - i * iou_interval
        tmp_set = torch.nonzero((max_overlaps < end_iou) & (max_overlaps >= start_iou), as_tuple=True)[0]

        per_num_expected = min(tmp_set.numel(), per_num_expected)
        perm = torch.randperm(tmp_set.numel(), device=tmp_set.device)[:per_num_expected]
        tmp_set = tmp_set[perm]
        sampled_inds.append(full_set[tmp_set])

        num_expected = num_expected - per_num_expected
        per_num_expected = int(num_expected / max(num_bins-i-1, 1))

    return torch.cat(sampled_inds)
