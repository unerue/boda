# 1. Interface

- 기존 코드
``` python
from boda.models import YolactConfig, YolactModel, YolactLoss

config = YolactConfig(num_classes=80)
model = YolactModel(config)
criterion = YolactLoss()

outputs = model(images)
losses = criterion(outputs, targets)
print(losses)
```

- 제안하는 코드
``` python
from boda.models import YolactConfig, YolactModel
from boda.losses import FocalLoss

config = YolactConfig(num_classes=80)
model = YolactModel(config)
# criterion = YolactLoss()
criterion = FocalLoss()  # loss 함수 고유명으로

outputs = model(images)
losses = criterion(outputs, targets)
print(losses)
```


# 2. Repository Architecture

- 기존 레포지토리 구조
``` bash
.
+-- models
|   +-- model (이동)
|   |   +-- configuration_model.py
|   |   +-- architecture_model.py
|   |   +-- loss_model.py (삭제)
|   |   +-- inference_model.py (삭제)
|   |   +-- README.md
|   +-- backbone.py (이동)
|   +-- neck.py (이동)
+-- utils
|   +-- box.py
|   +-- mask.py
|   +-- nms.py (삭제)
+-- lib
|   +-- torchsummary (삭제)
+-- base_architecture.py
+-- base_configuration.py
+-- modules.py
+-- activation.py (삭제)
+-- setup.py
```

- 제안하는 레포지토리 구조
```bash
.
+-- models
|   +-- detections (추가)
|   |   +-- model
|   |   |   +-- configuration_model.py
|   |   |   +-- architecture_model.py
|   |   |   +-- convert_ooo_to_xxx.py (추가)
|   |   |   +-- model_utils.py (필요할 경우 추가)
|   |   |   +-- README.md
|   +-- instances (추가)
|   |   +-- model
|   |   |   +-- configuration_model.py
|   |   |   +-- architecture_model.py
|   |   |   +-- convert_ooo_to_xxx.py (추가)
|   |   |   +-- model_utils.py (필요할 경우 추가)
|   |   |   +-- README.md
|   +-- backbones (추가)
|   |   +-- model.py
|   |   +-- backbone_utils.py (필요할 경우 추가)
|   +-- necks (추가)
|   |   +-- model.py
|   |   +-- neck_utils.py (필요할 경우 추가)
|   +-- anchor.py (추가)
|   +-- rpn.py (추가)
|   +-- roi_pool.py (추가)
|   +-- det_utils.py (필요할 경우 추가)
|   +-- mask_utils.py (필요할 경우 추가)
+-- losses (추가)
|   +-- loss.py
+-- utils
|   +-- preprocess.py (추가)
|   +-- box.py
|   +-- mask.py
|   +-- postprocess.py (추가)
|   +-- ...
+-- lib
|   +-- torchinfo (추가)
+-- base_architecture.py
+-- base_configuration.py
+-- modules.py
+-- setup.py
```

- 제안 사항
1. 기존의 modules, activations는 왠만하면 torchvision에 있는 그대로를 사용하거나 torchvision 꺼를 바탕으로 커스터마이징.
2. inference_model이 따로 있는거보단 architecture_model 내부적으로 처리하는게 기존 pytorch 문법과도 비슷하고 (torchvision 말고) huggingface와도 더 유사해보임.
3. rpn, roi_head를 그냥 torchvision꺼를 import할지 따로 구현할지 고민 (3번에서 자세히).
4. 왠만하면 util 함수는 torchvision꺼 그대로를 사용하지만 커스터마이징이 필요할 경우 사용범위에 따라 배치 (사용 범위가 넓을 수록 더 상위 디렉토리에 배치).
5. model input 형식을 List[Tensor]로 통일할 것이므로 torchsummary가 작동안됨. 작동되도록 커스터마이징 필요.
6. base_*.py는 huggingface 작동 방식에 기초.


# 3. Model Architecture

- 제안 사항
```python
# models/~/configuration_model.py

class DetectionConfig(BaseConfig):
    def __init__(
        self,
        # default
        num_classes: int,
        image_size: Optional[int, Tuple[int]],
        activation: str,
        ...,
        # backbone
        backbone_name: str,
        backbone_function_parameters: Dict[str, Any],
        # neck
        neck_name: str,
        neck_function_parameters: Dict[str, Any],
        # anchor
        anchor_name: str,
        anchor_function_parameters: Dict[str, Any]
        # rpn (for 2-stage model)
        rpn_function_parameters: Dict[str, Any],
        # roi_pooling (for 2-stage model)
        roi_pooling_parameters: Dict[str, Any],
        # head
        head_parameters: Dict[str, Any],
    )
    ...
```

```python
# models/~/architecture_model.py

class MiscHead(Head):
    def __init__(self, *args, **kwargs):
        ...

    def forward(self, x):
        ...
        return x


...


class DetectionPretrained(Model):
    config_class: DetectionConfig
    base_model_prefix: 'detection'

    @classmethod
    def from_pretrained(cls, name_or_path):
        ...
        return model


class DetectionModel(DetectionPretrained):
    model_name = 'detection'

    def __init__(self, *args, **kwargs):
        ...

    def init_weights(self, path):
        ...

    def forward(self, x):
        ...
        return x

    # def load_weights(self, path): -> convert_ooo_to_xxx.py로 이동
    #     ...

    ...
```

```python
# models/backbones/model.py

def util_func(*args, **kwargs):
    ...


class MiscBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        ...

    def forward(self, x):
        ...
        return x


...


class BackboneNet(nn.Module):
    def __init__(self, *args, **kwargs):
        self.layers: nn.Modulelist
        self.channels: List[int]
        ...

    def forward(self, x):
        ...
        return x

    def from_pretrained(self, path):
        ...


def backbone_v1(*args, **kwargs):
    ...
    return backbone


def backbone_v2(*args, **kwargs):
    ...
    return backbone
```

```python
# models/necks/model.py

class NeckNet(nn.Module):
    def __init__(
        self, channels, out_channels, *args, **kwargs
    ):
        ...

    def forward(self, x):
        ...
        return x

```

```python
# models/anchor.py

# strategy 1: customize based on torchvision
class Anchor:
    def __init__(
        self, sizes, aspect_ratios, *args, **kwargs
    ):
        ...

    ...


# strategy 2: use native torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator


...


# strategy 3: our own custom anchor
class CustomAnchor:
    ...
```

```python
# models/rpn.py

# strategy 1: customize based on torchvision
class RegionProposalNetwork(nn.Module):
    def __init__(
        self, sizes, aspect_ratios, *args, **kwargs
    ):
        ...

    ...


# strategy 2: use native torchvision
from torchvision.models.detection.rpn import RegionProposalNetwork


...


# strategy 3: our own custom anchor
class CustomRegionProposalNetwork:
    ...
```

```python
# models/roi_pooling.py

# strategy 1: customize based on torchvision
class RoiHeads(nn.Module):
    def __init__(
        self, sizes, aspect_ratios, *args, **kwargs
    ):
        ...

    ...


# strategy 2: use native torchvision
from torchvision.models.detection.roi_head import RoiHeads


...


# strategy 3: our own custom anchor
class CustomRoiPool:
    ...
```


# 4. Naming

- 기존 사항 (추가 예정)