from test.utils import convert_and_test

import numpy as np
import pytest
import torch.nn as nn


class LayerTest(nn.Module):
    def __init__(self):
        super(LayerTest, self).__init__()

    def forward(self, x):
        x = x.squeeze(1)
        return x


@pytest.mark.parametrize(
    "change_ordering", [pytest.param(True, marks=pytest.mark.xfail), False]
)
def test_squeeze(change_ordering):
    model = LayerTest()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 1, 32, 32))
    error = convert_and_test(
        model, input_np, verbose=False, change_ordering=change_ordering
    )
