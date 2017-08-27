# coding: utf-8
from chainer import datasets, iterators, Chain, optimizers, training
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
from chainer_concrete_dropout import ConcreteDropout


class MLP(Chain):
    def __init__(self, n_hidden, n_out):
        super(MLP, self).__init__()
        self.loss = 0

        with self.init_scope():
            self.cd1 = ConcreteDropout(L.Linear(None, n_hidden))
            self.l2 = L.Linear(None, n_out)

    def __call__(self, x):
        h1 = F.relu(self.cd1(x))
        y = self.l2(h1)

        self.loss = 0
        self.loss += self.cd1.get_regularizer_loss()
        return y


class Classifier(L.Classifier):

    def __call__(self, *args):
        loss = super(Classifier, self).__call__(*args)
        loss += self.predictor.loss
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
trainer.extend(
    extensions.PrintReport(
        ['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()
