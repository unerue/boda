import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from boda.utils.dataset import CocoDataset
# from boda.utils.transforms import Compose, Resize, ToTensor, Normalize
# from boda.models.configuration_solov1 import Solov1Config
# from boda.models.backbone_resnet import resnet101
# from boda.models.architecture_solov1 import Solov1Model, Solov1PredictHead, Solov1PredictNeck
# from boda.models.loss_solov1 import Solov1Loss
# from boda.utils.trainer import Trainer
from boda.lib.torchsummary import summary

from boda.models import Solov1Model, Solov1Config


# model = Solov1PredictNeck().to('cuda')
config = Solov1Config()
model = Solov1Model(config).to('cuda')
print(summary(model, input_data=(3, 1333, 800), verbose=0))

# cbb1 = 0
# l1 = 0
# for k, v in model.state_dict().items():
#     if k.startswith('backbone'):
#         cbb1 += 1
#     if not k.startswith('backbone'):
#         l1 += 1
#         # print(k, v.size())
# print()
# cbb2 = 0
# l2 = 0
# loaded_state_dict = torch.load('cache/solov1/SOLO_R50_3x.pth')['state_dict']
# # print(loaded_state_dict.keys())
# for k, v in loaded_state_dict.items():
#     if k.startswith('backbone'):
#         cbb2 += 1
#     if not k.startswith('backbone'):
#         l2 += 1
#         # print(k, v.size())
# print(cbb1, l1, cbb2, l2)


# state_dict = {}
# for k, v in loaded_state_dict.items():
#     parts = k.split('.')
#     if parts[0] == 'neck':
#         if parts[1].startswith('lateral'):
#             state_dict[f'neck.lateral_layers.{parts[2]}.{parts[4]}'] = loaded_state_dict.get(k)
#         elif parts[1].startswith('fpn_convs'):
#             state_dict[f'neck.predict_layers.{parts[2]}.{parts[4]}'] = loaded_state_dict.get(k)
#     elif parts[0] == 'bbox_head':
#         if parts[1].startswith('ins'):
#             state_dict[f'head.instance_layers.{parts[2]}.0.{parts[4]}'] = loaded_state_dict.get(k)
#         elif parts[1].startswith('cate'):
#             state_dict[f'head.category_layers.{parts[2]}.0.{parts[4]}'] = loaded_state_dict.get(k)
#         elif parts[1].startswith('solo_ins'):
#             state_dict[f'head.solo_instances.{parts[2]}'] = loaded_state_dict.get(k)
#         elif parts[1].startswith('solo_cate'):
#             state_dict[f'head.solo_category.{parts[2]}'] = loaded_state_dict.get(k)
#     elif parts[0].startswith('backbone'):
#         # if parts[1].startswith('layer'):
#             # new_key = f'backbone.layers.{int(parts[2])-1}.{parts[3]}.0.{parts[4]}'
#             # state_dict[new_key] = loaded_state_dict.get(k)
#         if parts[1] == 'conv1':
#             state_dict[f'backbone.conv.{parts[2]}'] = loaded_state_dict.get(k)
#         elif parts[1] == 'bn1':
#             state_dict[f'backbone.bn.{parts[2]}'] = loaded_state_dict.get(k)
#         elif parts[4].startswith('conv'):
#             state_dict[f'backbone.layers.{int(parts[2])-1}.{parts[3]}.{parts[4]}.0.{parts[5]}'] = loaded_state_dict.get(k)
#         # else:
#             state_dict[f'backbone.layers.{int(parts[2])-1}.{parts[3]}.{parts[4]}'] = loaded_state_dict.get(k)

# print(len(model.state_dict().keys()), len(loaded_state_dict.keys()))
# model.load_state_dict(state_dict)

