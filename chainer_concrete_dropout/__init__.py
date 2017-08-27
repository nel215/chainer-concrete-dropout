# coding: utf-8
import numpy as np
import chainer.functions as F
from chainer import link, cuda, Parameter


class ConcreteDropout(link.Chain):

    def __init__(self, layer, weight_reg=0.01):
        super(ConcreteDropout, self).__init__()
        self.weight_reg = weight_reg
        with self.init_scope():
            self.layer = layer
            self.pl = Parameter(0, (1))

    def __call__(self, x):
        xp = cuda.get_array_module(x)
        x = F.broadcast(x)
        self.input_shape = x.shape
        eps = 1e-20
        temp = 0.1
        p = F.sigmoid(self.pl)
        p = F.broadcast_to(p, x.shape)
        noise = xp.random.uniform(size=x.shape).astype('float32')
        drop = F.log(p + eps) - F.log(1. - p + eps)
        drop += F.log(noise + eps) - F.log(1. - noise + eps)
        drop = F.sigmoid(drop / temp)
        x *= 1. - drop
        x /= 1. - p

        return self.layer(x)

    def get_layer_square_norm(self):
        # TODO: consider a type of the layer
        return F.sum(F.square(self.layer.W))

    def get_regularizer_loss(self):
        input_dim = np.prod(self.input_shape[1:])
        p = F.sigmoid(self.pl)[0]

        weight_reg = self.weight_reg**2 / self.input_shape[0]
        weight_reg *= self.get_layer_square_norm() / (1. - p)

        ber_reg = p * F.log(p) + (1. - p) * F.log(1. - p)
        drop_reg = 2.0 / self.input_shape[0]
        drop_reg *= input_dim * ber_reg

        reg = F.sum(weight_reg + drop_reg)

        return reg
