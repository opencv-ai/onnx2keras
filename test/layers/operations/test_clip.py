from test.utils import convert_and_test

import numpy as np
import pytest
import torch.nn as nn


class FClipTest(nn.Module):
    """
    Test for nn.functional types
    """

    def __init__(self, low, high):
        self.low = low
        self.high = high
        super().__init__()

    def forward(self, x):
        return x.clamp(self.low, self.high)


@pytest.mark.parametrize(
    ["low", "high"],
    [
        (None, 1),
        (None, -1),
        (-1, None),
        (0, None),
        (1, None),
        (-1, 1.5),
        (0, 1.5),
        (1, 1.5),
    ],
)
@pytest.mark.parametrize("change_ordering", [True, False])
@pytest.mark.parametrize("opset", [9, 12])
def test_clip(change_ordering, opset, low, high):
    model = FClipTest(low, high)
    model.eval()

    input_np = np.linspace(-2, 2, 3 * 10 * 10).reshape(1, 3, 10, 10)

    error = convert_and_test(
        model, input_np, verbose=False, change_ordering=change_ordering, opset=opset
    )
