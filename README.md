<p align="center">
  <img height=110em src="boda.png">
</p>
<p align="center">
  <img alt="Kyungsu" src="https://img.shields.io/badge/Version%20-0.0.1b-orange.svg?style=flat&colorA=E1523D&colorB=blue" />
  <!-- <img alt="SCIE" src="https://img.shields.io/badge/SCIE%20-orange.svg" /> -->
  <!-- <img alt="KCI" src="https://img.shields.io/badge/KCI%20-yellow.svg" /> -->
  <!-- <img alt="PythonVersion" src="https://camo.githubusercontent.com/08d69975ce61c30b175f504182ae3a335c6284cbadc26acd9b79e29db442ddea/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d332e36253230253743253230332e37253230253743253230332e382d626c7565" data-canonical-src="https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue" style="max-width:100%;" /> -->
  <img alt="Kyungsu" src="https://img.shields.io/badge/Python%20-3.6%20%7C%203.7%20%7C%203.8-orange.svg?style=flat&colorA=gray&colorB=blue" style="max-width:100%;" />
  <img alt="Kyungsu" src="https://img.shields.io/badge/PyTorch%20-1.6%20%7C%201.7-orange.svg?style=flat&colorA=E1523D&colorB=blue" />
  <img src="https://badgen.net/badge/icon/terminal?icon=terminal&label" />
</p>

## Deep learning-based Computer Vision Models for PyTorch

Boda (보다) means to see in Korean. This library was inspired by 🤗 Transformers.

## Get started

```bash
git clone https://github.com/unerue/boda.git && cd boda
conda env create -f environment.yml
conda activate boda
python setup.py install
```

```python
from boda.models import YolactConfig, YolactModel, YolactLoss

config = YolactConfig(num_classes=80)
model = YolactModel(config)
criterion = YolactLoss()

outputs = model(images)
losses = criterion(outputs, targets)
print(losses)
```

## Comparison

|Model|State|Training|Inference|Original|Ours|
|:----|:---:|:------:|:-------:|-------:|---:|
|[SSD](boda/models/ssd/)|🙂|❌|✔️|||
|Faster R-CNN|🙂|❌|✔️|||
|FCIS|😡|❌|❌|||
|Mask R-CNN|😡|❌|❌|||
|Keypoint R-CNN|🙂|❌|✔️|||
|[FCOS](boda/models/fcos/)|🙂|❌|✔️|||
|PolarMask|😡|❌|❌|||
|YOLOv4|😡|❌|❌|||
|[YOLACT](boda/models/yolact/)|😆|✔️|✔️|||
|[SOLOv1](boda/models/solov1/)|🙂|❌|✔️|||
|SOLOv2|🙂|❌|✔️|||
|[CenterMask]()|😡|❌|❌|||
|YOLACT EDGE|😡|❌|❌|||
||

## Pretrained Model Configurations

|Model|Config name|Status|Original|Ours|
|:----|:----|:------:|-------:|---:|
|[SSD](boda/models/ssd/)|`ssd-base`|🙂|||
|                       |`ssd-512`|😡|||
|[Faster R-CNN]()|`faster-rcnn-base`|🙂|||
|                |`faster-rcnn-r101`|😡|||
|[FCIS]()|`fcis-base`|😡|||
|Mask R-CNN|`mask-rcnn-base`|😡|||
|          |`mask-rcnn-r50`|😡|||
|Keypoint R-CNN|`keypoint-rcnn-base`|🙂|||
|              |`keypoint-rcnn-mobile`|😡|||
|[FCOS](boda/models/fcos/)|`fcos-base`|🙂|||
|PolarMask|`polarmask-base`|😡|||
|YOLOv4|`yolov4-base`|😡|||
|[YOLACT](boda/models/yolact/)|`yolact-base`|😆|||
|                             |`yolact-r101`|😡|||
|                             |`yolact-r101-300`|😡|||
|                             |`yolact-r101-700`|😡|||
|[SOLOv1](boda/models/solov1/)|`solov1-base`|🙂|||
|                             |`solov1-r101`|😡|||
|SOLOv2|`solov2-base`|😡||||
|[CenterMask]()|`centermask-base`|😡|||
|YOLACT EDGE|`yolact-edge-base`|😡|||
||