# YOLACT (You Only Look At CoefficienTs)

## YOLACT Architecture

```{bash}
==============================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==============================================================================
├─ResNet: 1-1                            [-1, 256, 138, 138]               --
|    └─Conv2d: 2-1                       [-1, 64, 275, 275]             9,408
|    └─BatchNorm2d: 2-2                  [-1, 64, 275, 275]               128
|    └─ReLU: 2-3                         [-1, 64, 275, 275]                --
|    └─MaxPool2d: 2-4                    [-1, 64, 138, 138]                --
|    └─ModuleList: 2                     --                                --
|    |    └─Sequential: 3-1              [-1, 256, 138, 138]          215,808
|    |    └─Sequential: 3-2              [-1, 512, 69, 69]          1,219,584
|    |    └─Sequential: 3-3              [-1, 1024, 35, 35]        26,090,496
|    |    └─Sequential: 3-4              [-1, 2048, 18, 18]        14,964,736
├─YolactPredictNeck: 1-2                 [-1, 256, 69, 69]                 --
|    └─ModuleList: 2                     --                                --
|    |    └─Conv2d: 3-5                  [-1, 256, 18, 18]            524,544
|    |    └─Conv2d: 3-6                  [-1, 256, 35, 35]            262,400
|    |    └─Conv2d: 3-7                  [-1, 256, 69, 69]            131,328
|    └─ModuleList: 2                     --                                --
|    |    └─Conv2d: 3-8                  [-1, 256, 18, 18]            590,080
|    |    └─Conv2d: 3-9                  [-1, 256, 35, 35]            590,080
|    |    └─Conv2d: 3-10                 [-1, 256, 69, 69]            590,080
|    └─ModuleList: 2                     --                                --
|    |    └─Conv2d: 3-11                 [-1, 256, 9, 9]              590,080
|    |    └─Conv2d: 3-12                 [-1, 256, 5, 5]              590,080
├─YolactPredictHead: 1                   --                                --
|    └─HeadBranch: 2-5                   [[-1, 4]]                         --
|    |    └─Conv2d: 3-13                 [-1, 256, 69, 69]            590,080
|    |    └─Sequential: 3-14             [-1, 12, 69, 69]              27,660
|    |    └─Sequential: 3-15             [-1, 96, 69, 69]             221,280
|    |    └─Sequential: 3-16             [-1, 243, 69, 69]            560,115
|    └─HeadBranch: 2-6                   [[-1, 4]]                         --
|    └─HeadBranch: 2                     --                                --
|    |    └─Conv2d: 3-17                 [-1, 256, 35, 35]         (recursive)
|    |    └─Sequential: 3-18             [-1, 12, 35, 35]          (recursive)
|    |    └─Sequential: 3-19             [-1, 96, 35, 35]          (recursive)
|    |    └─Sequential: 3-20             [-1, 243, 35, 35]         (recursive)
|    └─HeadBranch: 2-7                   [[-1, 4]]                         --
|    └─HeadBranch: 2                     --                                --
|    |    └─Conv2d: 3-21                 [-1, 256, 18, 18]         (recursive)
|    |    └─Sequential: 3-22             [-1, 12, 18, 18]          (recursive)
|    |    └─Sequential: 3-23             [-1, 96, 18, 18]          (recursive)
|    |    └─Sequential: 3-24             [-1, 243, 18, 18]         (recursive)
├─ProtoNet: 1-3                          [-1, 32, 138, 138]                --
|    └─Conv2d: 2-8                       [-1, 256, 69, 69]            590,080
|    └─Conv2d: 2-9                       [-1, 256, 69, 69]            590,080
|    └─Conv2d: 2-10                      [-1, 256, 69, 69]            590,080
|    └─Upsample: 2-11                    [-1, 256, 138, 138]               --
|    └─Conv2d: 2-12                      [-1, 256, 138, 138]          590,080
|    └─Conv2d: 2-13                      [-1, 32, 138, 138]             8,224
├─SemanticSegmentation: 1-4              [-1, 80, 69, 69]                  --
|    └─Conv2d: 2-14                      [-1, 80, 69, 69]              20,560
==============================================================================
Total params: 50,157,071
Trainable params: 50,157,071
Non-trainable params: 0
Total mult-adds (G): 34.48
==============================================================================
Input size (MB): 3.46
Forward/backward pass size (MB): 193.40
Params size (MB): 191.33
Estimated Total Size (MB): 388.20
==============================================================================
```

```{python}
class CocoDataset(Dataset):
    def __getitem__(self, index: int) -> Tuple[Tensor, Dict]:
        """
        Returns:
            image (Tensor[C, H, W]): Original size
            targets (Dict[str, Any]): 
        """
        return image, {
            'boxes': FloatTensor[N, 4]: [x1, y1, x2, y2],
            'labels': LongTensor[N],
            'masks': ByteTensor[N, H, W],
            'keypoints' FloatTensor[N, K, 3]: [x, y, visibility],
            'area': float,
            'iscrowd': 0 or 1,
            'width': int,  # width of an original image
            'height': int,  # height of an original image
        }
```

```{python}
from boda.models import YolactConfig, YolactModel, YolactLoss

config = YolactConfig(num_classes=80)
model = YolactModel(config).to('cuda')
criterion = YolactLoss()

for epoch in range(num_epochs):
    for images, targets in train_loader:
        outputs = model(images)
        losses = criterion(outputs, targets)
        loss = sum(loss for loss in losses.values())
```

```{python}
class YolacModel:
    def forward(self, images):
        if self.training:
            # 전처리가 끝난 outputs?
            return {
                'boxes': FloatTensor,
                'masks: Tensor
                'scores': FloatTensor,
                'prior_boxes': 'anchors' ???
                'proposals'??
                'proto_masks':??
                'semantic_masks':??
            }
        else:
            # 전처리가 끝난 outputs
            return {
                'boxes': Tensor,
                'masks': 
                'scores': Tensor,
                'labels': Tensor,
                'keypoints': Tensor,
            }
```


```{python}
outputs = model(images)
outputs

# SSD
{'boxes', 'scores', 'prior_boxes'}

# Faster R-CNN
{'boxes', 'proposals', 'scores', 'anchors'}

# Keypoint R-CNN
{'boxes', 'proposals', 'scores', 'keypoints'}

# YOLACT
{'boxes', 'masks', 'scores', 'prior_boxes', 'proto_masks', 'semantic_masks'}

# SOLO
{'category', 'masks'}

# CenterMask
```

