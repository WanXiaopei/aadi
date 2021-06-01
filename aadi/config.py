
def add_aadi_config(cfg):
    cfg.MODEL.RPN.APPEND_ANCHORS = True
    cfg.MODEL.RPN.CANDIDATE_DILATION = [3]

    # Filter some negative proposals to accelerate training and inference speed
    cfg.MODEL.ROI_HEADS.POST_NMS_TOPK_TRAIN = 1000
    cfg.MODEL.ROI_HEADS.POST_NMS_TOPK_TEST = 1000

    # Configs for RetinaNet
    cfg.MODEL.RETINANET.CANDIDATE_DILATION = [2]
    cfg.MODEL.RETINANET.CONV_KERNEL_SIZE = 3
    cfg.MODEL.RETINANET.PROPOSAL_APPEND_GT = False
    cfg.MODEL.RETINANET.PROPOSAL_APPEND_HAND_ANCHORS = True
    cfg.MODEL.RETINANET.IOU_ENABLED = False
    cfg.MODEL.RETINANET.VFL_ENABLED = False
