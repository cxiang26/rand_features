import mxnet as mx
from mxnet import nd, autograd,gluon,image
from mxnet.gluon import nn
from mxnet.gluon.data import vision,DataLoader
import random
import time

def get_pretrained(model_name):
    return gluon.model_zoo.vision.get_model(model_name,pretrained=True)

# class RFnet(nn.HybridBlock):
#     def __init__(self, pretrain_model, rand_crop=True, is_train=True, **kwargs):
#         super(RFnet, self).__init__(**kwargs)
#         self.rand_crop = rand_crop
#         self.is_train = is_train
#         self.idx = [7,26]
#         self.input = nn.HybridSequential()
#         self.features = nn.HybridSequential()
#         self.output = nn.HybridSequential()
#         with self.name_scope():
#             self.input.add(nn.Conv2D(3,3,padding=(1,1)))
#             self.features.add(pretrain_model.features)
#             self.output.add(nn.Dense(10))
#     def crop_features(self, F, x):
#         shape = x.shape
#         if self.rand_crop:
#             x = F.pad(x, mode='edge', pad_width=(0,0,0,0,10,10,10,10))
#             x = F.Crop(x, h_w=(shape[2],shape[3]))
#         return x
#
#     def hybrid_forward(self, F, x):
#         x = self.input(x)
#         for i, net in enumerate(self.features[0]):
#             x = net(x)
#             if i+1 in self.idx and self.is_train:
#                 x = self.crop_features(F, x)
#         x = self.output(x)
#         return x

# class RFnet(nn.HybridBlock):
#     def __init__(self, model, is_train=True, **kwargs):
#         super(RFnet, self).__init__(**kwargs)
#         self.is_train = is_train
#         self.pre_features = nn.HybridSequential()
#         self.features = nn.HybridSequential()
#         self.output = nn.HybridSequential()
#         with self.name_scope():
#             self.pre_features.add(nn.Conv2D(3,3,padding=(1,1)))
#             for net in model.features:
#                 if net.name.find('stage') != -1:
#                     self.features.add(net)
#                 else:
#                     self.pre_features.add(net)
#             self.output.add(nn.Dense(10))
#
#     def hybrid_forward(self, F, x, *args, **kwargs):
#         for i in range(5):
#             x = self.pre_features[i](x)
#         for  idx, net in enumerate(self.features):
#             x = net(x)
#             if idx+1 in [1, 2, 3, 4]:
#                 if self.is_train:
#                     x = F.Crop(x, h_w=(x.shape[2]-2, x.shape[3]-2))
#                 else:
#                     x = F.Crop(x, h_w=(x.shape[2]-2, x.shape[3]-2), center_crop=1)
#         for i in range(5,9):
#             x = self.pre_features[i](x)
#         return self.output(x)
#
class resblock(nn.HybridBlock):
    def __init__(self, channels, **kwargs):
        super(resblock, self).__init__(**kwargs)
        self.block = nn.HybridSequential()
        with self.name_scope():
            self.block.add(nn.BatchNorm(in_channels=channels),nn.Conv2D(channels,3,1,1,activation='relu'),
                            nn.BatchNorm(in_channels=channels),nn.Conv2D(int(channels/2), 1,1,activation='relu'),
                            nn.BatchNorm(in_channels=int(channels/2)), nn.Conv2D(channels, 3, 1, 1, activation='relu'))
    def hybrid_forward(self, F, x, *args, **kwargs):
        out = self.block(x)
        return F.relu(out+x)

class RFnet(nn.HybridBlock):
    def __init__(self, channels=[32,64,128], classes=10, is_train=True, **kwargs):
        super(RFnet, self).__init__(**kwargs)
        self.is_train = is_train
        self.block1 = nn.HybridSequential()
        self.block2 = nn.HybridSequential()
        self.features = nn.HybridSequential()
        self.output = nn.HybridSequential()

        with self.name_scope():
            self.block1.add(nn.BatchNorm(), nn.Conv2D(16, 3, 1, 1, activation='relu'))
            self.mp1 = nn.MaxPool2D()
            self.block2.add(nn.BatchNorm(), nn.Conv2D(32, 3, 1, 1, activation='relu'))
            self.mp2 = nn.MaxPool2D()
            for channel in channels:
                self.features.add(nn.Conv2D(channel, 3, 1, 1, activation='relu'),resblock(channel))
            self.output.add(nn.BatchNorm(),nn.Activation('relu'),nn.GlobalAvgPool2D(),nn.Flatten(),nn.Dense(classes))

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.block1(x)
        if self.is_train:
            padding = 10
            h = x.shape[2]
            w = x.shape[3]
            area = x.shape[2] * x.shape[3]
            x_start = random.randint(0, 2 * padding - 1)
            y_start = random.randint(0, 2 * padding - 1)
            x_1 = max(x_start, padding)
            y_1 = max(y_start, padding)
            x_2 = min(x_start, padding) + x.shape[2]
            y_2 = min(y_start, padding) + x.shape[3]
            scale = (x_2 - x_1) * (y_2 - y_1) / area
            x = F.Pad(x, mode="constant", constant_value=0, pad_width=(0, 0, 0, 0, padding, padding, padding, padding))
            x = F.crop(x, begin=(None, None, x_start, y_start), end=(None, None, x_start + h, y_start + w)) / scale
        x = self.mp1(x)
        x = self.block2(x)
        if self.is_train:
            padding = 10
            h = x.shape[2]
            w = x.shape[3]
            area = x.shape[2]*x.shape[3]
            x_start = random.randint(0,2*padding-1)
            y_start = random.randint(0,2*padding-1)
            x_1 = max(x_start, padding)
            y_1 = max(y_start, padding)
            x_2 = min(x_start, padding)+x.shape[2]
            y_2 = min(y_start, padding)+x.shape[3]
            scale = (x_2-x_1)*(y_2-y_1)/area
            x = F.Pad(x, mode="constant",constant_value=0, pad_width=(0,0,0,0,padding,padding,padding,padding))
            x = F.crop(x, begin=(None,None,x_start,y_start),end=(None,None,x_start+h,y_start+w))/scale
        x = self.mp2(x)
        for net in self.features:
            x = net(x)
        x = self.output(x)
        return x


def load_data_fashion_mnist(batch_size, resize=None):
    def transfor_mnist_train(data, label):
        if resize:
            data = image.imresize(data, resize, resize)
        return nd.transpose(data.astype("float32"),(2,0,1))/255,label.astype("float32")
    def transfor_mnist_test(data, label):
        if resize:
            data = image.imresize(data, resize, resize)
        return nd.transpose(data.astype("float32"),(2,0,1))/255,label.astype("float32")
    mnist_train = vision.FashionMNIST(train=True, transform=transfor_mnist_train)
    mnist_test = vision.FashionMNIST(train=False, transform=transfor_mnist_test)
    train_data = DataLoader(mnist_train, batch_size, shuffle=True,num_workers=8)
    test_data = DataLoader(mnist_test, batch_size, shuffle=False,num_workers=8)
    return (train_data, test_data)

def accuracy(output, label):
    return nd.mean(nd.argmax(output, axis=1)==label).asscalar()

def _get_batch(batch, ctx):
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.label[0]
    else:
        data, label = batch
    return data.as_in_context(ctx), label.as_in_context(ctx)

def evaluate_accuracy(data_iterator, net, ctx=mx.cpu()):
    acc = 0.
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()
    for i, batch in enumerate(data_iterator):
        data, label = _get_batch(batch, ctx)
        output = net(data)
        acc += accuracy(output, label)
    return acc/(i+1)

def train(train_data, test_data, net, loss, trainer, ctx, lr_decay, num_epochs):
    for epoch in range(num_epochs):
        train_loss = 0.
        train_acc = 0.
        start = time.time()
        if (epoch+1) % lr_decay == 0:
            trainer.set_learning_rate(lr=trainer.learning_rate*0.1)
        if isinstance(train_data, mx.io.MXDataIter):
            train_data.reset()
        net.is_train = True
        for i, batch in enumerate(train_data):
            data, label = _get_batch(batch, ctx)

            with autograd.record():
                output = net(data)
                L = loss(output, nd.one_hot(label,10))
            L.backward()
            trainer.step(data.shape[0])
            train_loss += nd.mean(L).asscalar()
            train_acc += accuracy(output, label)

            n = i+1
        net.is_train = False
        test_acc = evaluate_accuracy(test_data, net, ctx)
        print("Epoch %d. Time: %.2f, Loss: %f, Train acc: %f, Test acc: %f."%(epoch, time.time()-start, train_loss/n, train_acc/n, test_acc))

def get_net(ctx):
    #pretrained = get_pretrained()
    net = RFnet()
    net.collect_params().initialize()
    # net.pre_features[0].initialize()
    # net.output.initialize()
    net.collect_params().reset_ctx(ctx)
    #net.hybridize()
    return net

def main():
    batch_size = 320
    Epoches = 100
    lr = 0.001
    lr_decay = 30
    model_name = 'resnet50_v2'

    ctx = mx.gpu()
    net = get_net(ctx)
    train_data ,test_data = load_data_fashion_mnist(batch_size,resize=52)
    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),'adam',{'learning_rate':lr})
    train(train_data,test_data,net,loss,trainer,ctx,lr_decay,Epoches)

if __name__ == '__main__':
    main()
    print('train is done!')