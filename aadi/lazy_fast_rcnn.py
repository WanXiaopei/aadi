import torch
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.layers import cat, nonzero_tuple
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, _log_classification_stats


class LazyFastRCNNOutputLayers(FastRCNNOutputLayers):
    # We are too lazy to tune the hyper-parameter which named batch_size_per_image,
    # so we propose this LazyFastRCNNOutputLayers, it is very robust to batch_size_per_image.
    # Note that, LazyFastRCNNOutputLayers may lower the performance for well-tuned Faster R-CNN and Cascade R-CNN
    @configurable
    def __init__(self, **kwargs):
        super(LazyFastRCNNOutputLayers, self).__init__(**kwargs)
        self.num_positive = 256.
        self.momentum = 0.99
        # For Faster R-CNN, we find that there are about 70 positive samples in total 512 samples,
        # so we set _loss_weight as 70 / 512 ~ 1. / 7.
        self._loss_weight = 1. / 7.

    def losses(self, predictions, proposals):
        scores, proposal_deltas = predictions
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        self.num_positive *= self.momentum
        self.num_positive += (1. - self.momentum) * max(fg_inds.numel() * 1. / len(proposals), 1.)

        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        normalizer = self.num_positive * len(proposals)
        losses = {
            "loss_cls": F.cross_entropy(
                scores, gt_classes, reduction="sum") * self._loss_weight / normalizer,
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ) * max(gt_classes.numel(), 1.) * self._loss_weight / normalizer,
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
