import io

import onnx
import torch

from onnx2keras import check_torch_keras_error, onnx_to_keras


def torch2keras(
    model: torch.nn.Module,
    input_variable,
    verbose=True,
    change_ordering=False,
    opset=None,
):
    if isinstance(input_variable, (tuple, list)):
        input_variable = tuple(torch.FloatTensor(var) for var in input_variable)
        input_names = [f"test_in{i}" for i, _ in enumerate(input_variable)]
    else:
        input_variable = torch.FloatTensor(input_variable)
        input_names = ["test_in"]

    temp_f = io.BytesIO()
    kwargs = {}
    if opset is not None:
        kwargs["opset_version"] = opset
    torch.onnx.export(
        model,
        input_variable,
        temp_f,
        verbose=verbose,
        input_names=input_names,
        output_names=["test_out"],
        **kwargs,
    )
    temp_f.seek(0)
    onnx_model = onnx.load(temp_f)
    k_model = onnx_to_keras(onnx_model, input_names, change_ordering=change_ordering)
    return k_model


def convert_and_test(
    model: torch.nn.Module,
    input_variable,
    verbose=True,
    change_ordering=False,
    epsilon=1e-5,
    opset=None,
):
    k_model = torch2keras(
        model,
        input_variable,
        verbose=verbose,
        change_ordering=change_ordering,
        opset=opset,
    )
    error = check_torch_keras_error(
        model, k_model, input_variable, change_ordering=change_ordering, epsilon=epsilon
    )
    return error
