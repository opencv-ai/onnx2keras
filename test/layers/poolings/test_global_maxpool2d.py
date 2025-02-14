from test.utils import convert_and_test

import numpy as np
import pytest
import tensorflow as tf
import torch.nn as nn


class LayerTest(nn.Module):
    def __init__(self):
        super(LayerTest, self).__init__()
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        x = self.pool(x)
        return x


@pytest.mark.parametrize("change_ordering", [True, False])
def test_global_maxpool2d(change_ordering):
    if not tf.test.gpu_device_name() and not change_ordering:
        pytest.skip(
            "Skip! Since tensorflow MaxPoolingOp op currently only supports the NHWC tensor format on the CPU"
        )
    model = LayerTest()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 32, 32))
    error = convert_and_test(
        model, input_np, verbose=False, change_ordering=change_ordering
    )
