from test.utils import convert_and_test

import numpy as np
import pytest
import torch.nn as nn


class FSub(nn.Module):
    def __init__(self):
        super(FSub, self).__init__()

    def forward(self, x, y):
        x = x - y - 8.3
        return x


@pytest.mark.repeat(10)
@pytest.mark.parametrize("change_ordering", [True, False])
def test_add(change_ordering):
    model = FSub()
    model.eval()

    input_np1 = np.random.uniform(0, 1, (1, 3, 224, 224))
    input_np2 = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(
        model, (input_np1, input_np2), verbose=False, change_ordering=change_ordering
    )
