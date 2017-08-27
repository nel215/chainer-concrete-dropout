# coding: utf-8
from chainer import datasets, iterators, Chain, optimizers, training, cuda
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
import numpy as np


class MLP(Chain):
    def __init__(self, n_hidden, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_hidden)
            self.l2 = L.Linear(None, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        y = self.l2(h1)
        return y


gpu = 0
train, test = datasets.get_mnist()
train_iter = iterators.SerialIterator(train, batch_size=200, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size=100, shuffle=False)
model = L.Classifier(MLP(100, 10))
optimizer = optimizers.SGD()
optimizer.setup(model)
updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
trainer = training.Trainer(updater, (20, 'epoch'), out='result')
# trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()
