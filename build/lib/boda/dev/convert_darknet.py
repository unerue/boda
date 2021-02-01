from backbone import DarkNetBackbone
import h5py
import torch

f = h5py.File('darknet53.h5', 'r')
m = f['model_weights']

yolo_keys = list(m.keys())
yolo_keys = [x for x in yolo_keys if len(m[x].keys()) > 0]
yolo_keys.sort()

sd = DarkNetBackbone().state_dict()

sd_keys = list(sd.keys())
sd_keys.sort()

# Note this won't work if there are 10 elements in some list but whatever that doesn't happen
layer_keys = list(set(['.'.join(x.split('.')[:-2]) for x in sd_keys]))
layer_keys.sort()

# print([x for x in sd_keys if x.startswith(layer_keys[0])])

mapping = {
	'.0.weight'      : ('conv2d_%d', 'kernel:0'),
	'.1.bias'        : ('batch_normalization_%d', 'beta:0'),
	'.1.weight'      : ('batch_normalization_%d', 'gamma:0'),
	'.1.running_var' : ('batch_normalization_%d', 'moving_variance:0'),
	'.1.running_mean': ('batch_normalization_%d', 'moving_mean:0'),
	'.1.num_batches_tracked': None,
}

for i, layer_key in zip(range(1, len(layer_keys) + 1), layer_keys):
	# This is pretty inefficient but I don't care
	for weight_key in [x for x in sd_keys if x.startswith(layer_key)]:
		diff = weight_key[len(layer_key):]
		
		if mapping[diff] is not None:
			yolo_key = mapping[diff][0] % i
			sub_key  = mapping[diff][1]

			yolo_weight = torch.Tensor(m[yolo_key][yolo_key][sub_key].value)
			if (len(yolo_weight.size()) == 4):
				yolo_weight = yolo_weight.permute(3, 2, 0, 1).contiguous()
			
			sd[weight_key] = yolo_weight

torch.save(sd, 'weights/darknet53.pth')





def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)



# https://github.com/qfgaohao/pytorch-ssd/blob/master/translate_tf_mobilenetv1.py
import torch
import sys

from vision.nn.mobilenet import MobileNetV1
from extract_tf_weights import read_weights


def fill_weights_torch_model(weights, state_dict):
    for name in state_dict:
        if name == 'classifier.weight':
            weight = weights['MobilenetV1/Logits/Conv2d_1c_1x1/weights']
            weight = torch.tensor(weight, dtype=torch.float32).permute(3, 2, 0, 1)
            assert state_dict[name].size() == weight.size()
            state_dict[name] = weight
        elif name == 'classifier.bias':
            bias = weights['MobilenetV1/Logits/Conv2d_1c_1x1/biases']
            bias = torch.tensor(bias, dtype=torch.float32)
            assert state_dict[name].size() == bias.size()
            state_dict[name] = bias
        elif name.endswith('BatchNorm.weight'):
            key = name.replace("features", "MobilenetV1").replace(".", "/").replace('BatchNorm/weight', 'BatchNorm/gamma')
            weight = torch.tensor(weights[key], dtype=torch.float32)
            assert weight.size() == state_dict[name].size()
            state_dict[name] = weight
        elif name.endswith('BatchNorm.bias'):
            key = name.replace("features", "MobilenetV1").replace(".", "/").replace('BatchNorm/bias', 'BatchNorm/beta')
            bias = torch.tensor(weights[key], dtype=torch.float32)
            assert bias.size() == state_dict[name].size()
            state_dict[name] = bias
        elif name.endswith('running_mean'):
            key = name.replace("features", "MobilenetV1").replace(".", "/").replace('running_mean', 'moving_mean')
            running_mean = torch.tensor(weights[key], dtype=torch.float32)
            assert running_mean.size() == state_dict[name].size()
            state_dict[name] = running_mean
        elif name.endswith('running_var'):
            key = name.replace("features", "MobilenetV1").replace(".", "/").replace('running_var', 'moving_variance')
            running_var = torch.tensor(weights[key], dtype=torch.float32)
            assert running_var.size() == state_dict[name].size()
            state_dict[name] = running_var
        elif name.endswith('depthwise.weight'):
            key = name.replace("features", "MobilenetV1").replace(".", "/")
            key = key.replace('depthwise/weight', 'depthwise/depthwise_weights')
            weight = torch.tensor(weights[key], dtype=torch.float32).permute(2, 3, 0, 1)
            assert weight.size() == state_dict[name].size()
            state_dict[name] = weight
        else:
            key = name.replace("features", "MobilenetV1").replace(".", "/").replace('weight', 'weights')
            weight = torch.tensor(weights[key], dtype=torch.float32).permute(3, 2, 0, 1)
            assert weight.size() == state_dict[name].size()
            state_dict[name] = weight


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python translate_tf_modelnetv1.py <tf_model.pb> <pytorch_weights.pth>")
    tf_model = sys.argv[1]
    torch_weights_path = sys.argv[2]
    print("Extract weights from tf model.")
    weights = read_weights(tf_model)

    net = MobileNetV1(1001)
    states = net.state_dict()
    print("Translate tf weights.")
    fill_weights_torch_model(weights, states)
    torch.save(states, torch_weights_path)


# https://github.com/qfgaohao/pytorch-ssd/blob/master/visual_tf_models.py
import tensorflow as tf
from tensorflow.python.platform import gfile
import sys
import time

if len(sys.argv) < 2:
    print("Usage: python visual_tf_model.py <model.pb>")
    sys.exit(0)

model_file_name = sys.argv[1]
with tf.Session() as sess:
    with gfile.FastGFile(model_file_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
LOGDIR='log'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)

while True:
    time.sleep(1000)