import torch
from torch.nn import functional as F
import onnx
import pytest
import numpy as np

from test.utils import convert_and_test


class LayerHardSigmoid(torch.nn.Module):
    def __init__(self):
        super(LayerHardSigmoid, self).__init__()
        self.hs = torch.nn.Hardsigmoid()

    def forward(self, x):
        return self.hs(x)


class FHardSigmoid(torch.nn.Module):
    def __init__(self):
        super(FHardSigmoid, self).__init__()

    def forward(self, x):
        return F.hardsigmoid(x)


@pytest.mark.parametrize('hard_sigmoid_class', [LayerHardSigmoid, FHardSigmoid])
def test_layer_hard_sigmoid(hard_sigmoid_class):
    model = hard_sigmoid_class()
    model.eval()

    test_input = np.linspace(-5, 5, 224).reshape((1, 224, 1, 1))
    convert_and_test(model, test_input, verbose=False)
