# coding: utf-8
from chainer import datasets, iterators, Chain, optimizers, training, cuda, Variable, Parameter
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
import numpy as np
import cupy as cp


class MLP(Chain):
    def __init__(self, n_hidden, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.pl1 = Parameter(0, (1))
            self.l1 = L.Linear(None, n_hidden)
            self.l2 = L.Linear(None, n_out)

    def __call__(self, x):
        # Concrete Dropout
        xp = cuda.get_array_module(x)
        x = F.broadcast(x)
        self.input_shape1 = x.shape
        eps = 1e-20
        temp = 0.1
        p1 = F.sigmoid(self.pl1)
        p1 = F.broadcast_to(p1, x.shape)
        noise = xp.random.uniform(size=x.shape).astype('float32')
        drop = F.log(p1 + eps) - F.log(1. - p1 + eps)
        drop += F.log(noise + eps) - F.log(1. - noise + eps)
        drop = F.sigmoid(drop / temp)
        x *= 1. - drop
        x /= 1. - p1

        h1 = F.relu(self.l1(x))
        y = self.l2(h1)
        return y

    def get_regularizer_loss(self):
        input_dim = np.prod(self.input_shape1[1:])
        p1 = F.sigmoid(self.pl1)[0]
        # TODO: hyper parameter is l**2/N
        weight_reg = 0.001 * F.sum(F.square(self.l1.W)) / (1. - p1)
        ber_reg = p1 * F.log(p1) + (1. - p1) * F.log(1. - p1)
        # TODO: hyper parameter is 2/N
        drop_reg = 0.001 * input_dim * ber_reg
        reg = F.sum(weight_reg + drop_reg)
        return reg


class Classifier(L.Classifier):

    def __call__(self, *args):
        loss = super(Classifier, self).__call__(*args)
        loss += self.predictor.get_regularizer_loss()
        return loss


gpu = 0
train, test = datasets.get_mnist()
train_iter = iterators.SerialIterator(train, batch_size=200, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size=100, shuffle=False)
model = Classifier(MLP(100, 10))
optimizer = optimizers.SGD()
optimizer.setup(model)
updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
trainer = training.Trainer(updater, (20, 'epoch'), out='result')
# trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()
