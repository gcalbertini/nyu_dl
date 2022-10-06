import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.activations = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(),
                            'identity': nn.Identity()}

        try:
            self.activations[self.f_function]
        except KeyError:
            print('The function f is not valid. Defaulting to identity.')
            self.f_function = 'identity'

        try:
            self.activations[self.g_function]
        except KeyError:
            print('The function g is not valid. Defaulting to identity.')
            self.g_function = 'identity'

        self.parameters = dict(
            W1=torch.randn(linear_1_out_features, linear_1_in_features),
            b1=torch.randn(linear_1_out_features),
            W2=torch.randn(linear_2_out_features, linear_2_in_features),
            b2=torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1=torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1=torch.zeros(linear_1_out_features),
            dJdW2=torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2=torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        self.cache['x'] = x
        z1 = torch.matmul(x, self.parameters['W1'].t()) + self.parameters['b1']
        self.cache['z1'] = z1

        z2 = self.activations[self.f_function](z1)
        self.cache['z2'] = z2

        z3 = torch.matmul(z2, self.parameters['W2'].t())+self.parameters['b2']
        self.cache['z3'] = z3

        y_hat = self.activations[self.g_function](z3)
        self.cache['y_hat'] = y_hat

        return self.cache['y_hat']

    def grad_backprop_helper(self, func, input_):
        sigma = torch.nn.Sigmoid()
        z = sigma(input_)
        grad_mappings = {'relu': torch.ones(input_.size()) * (input_ > 0),
                         'sigmoid': torch.mul(z, 1-z),
                         'identity': torch.ones(input_.size())}
        return grad_mappings[func]

    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function

        # At final layer
        batch_size = dJdy_hat.shape[0]
        dydz3 = self.grad_backprop_helper(self.g_function, self.cache['z3'])
        dJdz3 = torch.mul(dJdy_hat, dydz3)
        self.grads['dJdb2'] = torch.matmul(dJdz3.t(), torch.ones(batch_size))
        self.grads['dJdW2'] = torch.matmul(dJdz3.t(), self.cache['z2'])

        # First linear layer
        dz3dz2 = self.parameters['W2']
        dz2dz1 = self.grad_backprop_helper(self.f_function, self.cache['z1'])
        dJdz1 = dz2dz1 * (dJdz3 @ dz3dz2)

        # First linear layer grads
        self.grads['dJdb1'] = torch.matmul(dJdz1.t(), torch.ones(batch_size))
        self.grads['dJdW1'] = torch.matmul(dJdz1.t(), self.cache['x'])

    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()


def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)
    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    loss = torch.pow((y_hat-y), 2).mean()
    dJdy_hat = 2*(y_hat - y)/(y.shape[0]*y.shape[1])

    return loss, dJdy_hat


def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor

    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    loss = - (y * torch.clamp(torch.log(y_hat), min=-100) + (1-y)
              * torch.clamp(torch.log(1-y_hat), min=-100)).mean()
    dJdy_hat = (- y/y_hat + (1-y)/(1-y_hat))/(y.shape[0]*y.shape[1])

    return loss, dJdy_hat