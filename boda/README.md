# Models

## Library Structure
```{bash}
.
+-- models
|   +-- model
|   |   +-- configuration_model.py
|   |   +-- architecture_model.py
|   |   +-- loss_model.py
|   |   +-- inference_model.py
|   |   +-- README.md
|   +-- backbone.py
|   +-- neck.py
+-- utils
|   +-- box.py
|   +-- mask.py
|   +-- nms.py
+-- lib
|   +-- torchsummary
+-- base_architecture.py
+-- base_configuration.py
+-- modules.py
+-- activation.py
+-- setup.py
```

## Abstract Structure

```{python}
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return


class Neck(nn.Module):
    def __init__(self):
        super().__init__()
    
    def _make_layer(self):

    def forward(self):
        return


class Head(nn.Module):


class Pretrained:
    

class Model()
```