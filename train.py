import mxnet as mx
from mxnet import nd, autograd,gluon,image
from mxnet.gluon import nn
from mxnet.gluon.data import vision,DataLoader
import argparse
import time

def get_pretrained(model_name):
    return gluon.model_zoo.vision.get_model(model_name,pretrained=True)

class RFnet(nn.HybridBlock):
    def __init__(self, pretrain_model, rand_crop=True, is_train=True, **kwargs):
        super(RFnet, self).__init__(**kwargs)
        self.rand_crop = rand_crop
        self.is_train = is_train
        self.idx = [4,8]
        self.features = nn.HybridSequential()
        self.output = nn.HybridSequential()
        with self.name_scope():
            self.features.add(nn.Conv2D(3,3,padding=(1,1)),
                              pretrain_model.features)
            self.output.add(nn.Dense(10))
    def crop_features(self, F, x, idx):
        shape = x.shape
        if self.rand_crop:
            x = F.pad(x, mode='constant',constant_value=0, pad_width=(0,0,0,0,10-idx,10-idx,10-idx,10-idx))
            x = F.Crop(x, h_w=(shape[2],shape[3]))
        return x

    def hybrid_forward(self, F, x):
        for i, net in enumerate(self.features):
            x = net(x)
            if i+1 in self.idx and self.is_train:
                x = self.crop_features(F, x, i+1)
        x = self.output(x)
        return x

def load_data_fashion_mnist(batch_size, resize=None):
    def transfor_mnist(data, label):
        if resize:
            data = image.imresize(data, resize, resize)
        return nd.transpose(data.astype("float32"),(2,0,1))/255,label.astype("float32")
    mnist_train = vision.FashionMNIST(train=True, transform=transfor_mnist)
    mnist_test = vision.FashionMNIST(train=False, transform=transfor_mnist)
    train_data = DataLoader(mnist_train, batch_size, shuffle=True)
    test_data = DataLoader(mnist_test, batch_size, shuffle=False)
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
        for i, batch in  enumerate(train_data):
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

def get_net(model_name, ctx):
    pretrained = get_pretrained(model_name)
    net = RFnet(pretrained)
    net.features[0].initialize()
    net.output.initialize()
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    return net

def main():
    batch_size = 32
    Epoches = 100
    lr = 0.001
    lr_decay = 40
    model_name = 'vgg19_bn'

    ctx = mx.gpu()
    net = get_net(model_name, ctx)
    train_data ,test_data = load_data_fashion_mnist(batch_size,resize=224)
    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),'adam',{'learning_rate':lr})
    train(train_data,test_data,net,loss,trainer,ctx,lr_decay,Epoches)

if __name__ == '__main__':
    main()