# Augmenting Anchors by the Detector Itself

## Introduction

  It is difficult to determine the scale and aspect ratio of anchors for anchor-based object detection methods. Current state-of-the-art 
  object detectors either determine anchor parameters according to objects' shape and scale in a dataset, or avoid this problem by 
  utilizing anchor-free method. In this paper, we propose a gradient-free anchor augmentation method named AADI, which means Augmenting Anchors 
  by the Detector Itself. AADI is not an anchor-free method, but it converts the scale and aspect ratio of anchors from a continuous space to a discrete space, which
  greatly alleviates the problem of anchors' designation. Furthermore, AADI does not add any parameters or hyper-parameters, which is beneficial for future 
  research and downstream tasks. Extensive experiments on COCO dataset show that AADI has obvious advantages for both two-stage and single-stage methods, 
  specifically, AADI achieves at least 2.1 AP improvements on Faster R-CNN and 1.6 AP improvements on RetinaNet, using 
  ResNet-50 model. We hope that this simple and cost-efficient method can be widely used in object detection.

* For RPN

    - Baseline
    
        | Num anchors | AR100 | AR1000 | ARs  | ARm  | ARl  |
        |:-----------:|:-----:|:------:|:----:|:----:|:----:|
        |      1      | 45.5  |  55.6  | 31.4 | 52.8 | 60.0 |
        |      3      | 45.7  |  58.0  | 31.4 | 52.7 | 61.1 |

    - Ablation Study

        | dilation | Anchor Guided  | AR100 | AR1000 |  ARs  |  ARm   |  ARl   |
        |:--------:|:-------:|:-----:|:------:|:-----:|:------:|:------:|
        |    1     |         |  52.8 |  60.6  |  40.2 |  60.8  |  63.6  |
        |    2     |         |  54.8 |  64.7  |  39.0 |  63.1  |  70.6  |
        |    2     | &radic; |  56.3 |  66.7  |  39.5 |  64.9  |  73.4  |
        |    3     |         |  53.7 |  64.0  |  35.4 |  62.1  |  73.9  |
        |    3     | &radic; |  55.6 |  67.6  |  36.1 |  64.3  |  77.6  |
        |    4     |         |  52.2 |  60.5  |  30.9 |  61.3  |  76.6  |
        |    4     | &radic; |  54.4 |  65.5  |  33.0 |  63.7  |  78.9  |

* For RetinaNet

    - Ablation Study

        | AADI     | dilation |  AP  | AP50 | AP75 | APs  | APm  | APl  |
        |:--------:|:--------:|:----:|:----:|:----:|:----:|:----:|:----:|
        |          |    1     | 38.2 | 58.4 | 41.1 | 24.3 | 42.2 | 48.5 |
        | &radic;  |    1     | 37.3 | 56.4 | 40.2 | 22.0 | 39.9 | 46.8 | 
        | &radic;  |    2     | 39.8 | 57.5 | 43.5 | 22.1 | 43.5 | 50.6 |
        | &radic;  |    3     | 38.3 | 54.6 | 41.7 | 20.0 | 43.1 | 51.1 |

    - With IoU
        
        |  AP  | AP50 | AP75 | APs  | APm  | APl  |
        |:----:|:----:|:----:|:----:|:----:|:----:|
        | 40.2 | 57.7 | 43.8 | 24.1 | 43.1 | 52.2 |
    
    - With 3x schedule (RetinaNet with giou, AADI with smooth l1)
    
        | Model          |  AP  | AP50 | AP75 | APs  | APm  | APl  |
        |:--------------:|:----:|:----:|:----:|:----:|:----:|:----:|
        | RetinaNet      | 39.6 | 59.3 | 42.2 | 24.9 | 43.3 | 50.7 |
        | AADI-RetinaNet | 41.4 | 59.3 | 45.2 | 24.8 | 44.9 | 54.0 |

* For Faster R-CNN

    - Ablation Study

        | AADI    | dilation |  AP  | AP50 | AP75 | APs  | APm  | APl  | FPS |
        |:-------:|:--------:|:----:|:----:|:----:|:----:|:----:|:----:|:---:|
        |         |1(3 anchors)| 37.9 | 58.8 | 41.1 | 22.4 | 41.1 | 49.1 | 26.3 |
        | &radic; |    2     | 40.3 | 59.3 | 44.3 | 24.2 | 43.3 | 52.2 | 22.4 |
        | &radic; |    3     | 40.8 | 59.5 | 45.0 | 24.0 | 44.6 | 53.1 | 22.4 |
        | &radic; |    4     | 40.5 | 58.7 | 44.6 | 23.2 | 44.8 | 52.7 | 22.3 |

    - 3x schedule
    
        | Backbone   |  AP  | AP50 | AP75 | APs  | APm  | APl  | FPS  |
        |:----------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
        | R-50 FPN   | 42.5 | 61.2 | 46.5 | 25.3 | 46.2 | 55.5 | 22.6 |
        | DCN-50 FPN | 44.1 | 63.1 | 48.2 | 28.3 | 46.9 | 58.4 | 20.1 |
        | R-101 FPN  | 44.5 | 63.2 | 48.7 | 26.9 | 48.3 | 57.4 | 17.4 |

* Detectron2

Detectron2 is Facebook AI Research's next generation library
that provides state-of-the-art detection and segmentation algorithms.
It is the successor of
[Detectron](https://github.com/facebookresearch/Detectron/)
and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).
It supports a number of computer vision research projects and production applications in Facebook.

<div align="center">
  <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png"/>
</div>

## Installation

See [installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).

## Getting Started

See [Getting Started with Detectron2](https://detectron2.readthedocs.io/tutorials/getting_started.html),
and the [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
to learn about basic usage.

Learn more at our [documentation](https://detectron2.readthedocs.org).


## Citing Detectron2

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}

@misc{wan2021augmenting,
      title={Augmenting Anchors by the Detector Itself}, 
      author={Xiaopei Wan and Shengjie Chen and Yujiu Yang and Zhenhua Guo and Fangbo Tao},
      year={2021},
      eprint={2105.14086},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
