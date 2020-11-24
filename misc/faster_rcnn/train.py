# https://github.com/wllvcxz/faster-rcnn-pytorch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from chainercv.datasets import VOCBboxDataset
from chainercv.datasets import voc_bbox_label_names
from chainercv.visualizations import vis_bbox
import torch
from torchnet.meter import AverageValueMeter, MovingAverageValueMeter

from model.faster_rcnn import faster_rcnn

train_dataset = VOCBboxDataset(year='2007', split='train')
val_dataset = VOCBboxDataset(year='2007', split='val')
trainval_dataset = VOCBboxDataset(year='2007', split='trainval')
test_dataset = VOCBboxDataset(year='2007', split='test')


def adjust_learning_rate(optimizer, epoch, init_lr, lr_decay_factor=0.1, lr_decay_epoch=10):
    """Sets the learning rate to the initial LR decayed by lr_decay_factor every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch == 0:
        lr = init_lr * (lr_decay_factor ** (epoch // lr_decay_epoch))
        print('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr




model = faster_rcnn(20, backbone='vgg16')
if torch.cuda.is_available():
    model = model.cuda()

optimizer = model.get_optimizer(is_adam=False)
avg_loss = AverageValueMeter()
ma20_loss = MovingAverageValueMeter(windowsize=20)
model.train()


for epoch in range(15):
    adjust_learning_rate(optimizer, epoch, 0.001, lr_decay_epoch=10)
    for i in range(len(trainval_dataset)):
        img, bbox, label = trainval_dataset[i]
        img = img/255

        loss = model.loss(img, bbox, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.cpu().data.numpy()[0]
        avg_loss.add(loss_value)
        ma20_loss.add(float(loss_value))
        print('[epoch:{}]  [batch:{}/{}]  [sample_loss:{:.4f}]  [avg_loss:{:.4f}]  [ma20_loss:{:.4f}]'.format(epoch, i, len(trainval_dataset), loss_value, avg_loss.value()[0], ma20_loss.value()[0]))

    modelweight = model.state_dict()
    trainerstate = {
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(modelweight, "epoch_"+str(epoch)+".modelweight")
    torch.save(trainerstate, "epoch_"+str(epoch)+".trainerstate")



model.eval()
for i in range(len(test_dataset)):
    img, _, _ = test_dataset[i]
    imgx = img/255
    bbox_out, class_out, prob_out = model.predict(imgx, prob_threshold=0.95)

    vis_bbox(img, bbox_out, class_out, prob_out,label_names=voc_bbox_label_names) 
    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches(11, 5)
    fig.savefig('test_'+str(i)+'.jpg', dpi=100)
