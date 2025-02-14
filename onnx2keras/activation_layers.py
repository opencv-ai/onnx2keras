import logging

import tensorflow as tf
from tensorflow import keras

from .utils import ensure_numpy_type, ensure_tf_type


def convert_relu(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert ReLU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError("More than 1 input for an activation layer.")

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    relu = keras.layers.Activation("relu", name=keras_name)
    layers[node_name] = relu(input_0)


def convert_elu(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert ELU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError("More than 1 input for an activation layer.")

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    elu = keras.layers.ELU(alpha=params["alpha"], name=keras_name)
    layers[node_name] = elu(input_0)


def convert_lrelu(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert LeakyReLU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError("More than 1 input for an activation layer.")

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    leakyrelu = keras.layers.LeakyReLU(alpha=params["alpha"], name=keras_name)
    layers[node_name] = leakyrelu(input_0)


def convert_sigmoid(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Sigmoid activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError("More than 1 input for an activation layer.")

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    sigmoid = keras.layers.Activation("sigmoid", name=keras_name)
    layers[node_name] = sigmoid(input_0)


def convert_hard_sigmoid(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert HardSigmoid activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """

    # Default values are taken from the
    # https://github.com/onnx/onnx/blob/master/docs/Operators.md#hardsigmoid
    ONNX_HARD_SIGMOID_DEFAULT_ALPHA = 0.2
    ONNX_HARD_SIGMOID_DEFAULT_BETA = 0.5

    if len(node.input) != 1:
        assert AttributeError("More than 1 input for an activation layer.")

    input_0 = ensure_tf_type(layers[node.input[0]], name=f"{keras_name}_const")

    alpha = params.get("alpha", ONNX_HARD_SIGMOID_DEFAULT_ALPHA)
    assert alpha != 0.0, "Alpha can't be zero - it doesn't make sense"
    beta = params.get("beta", ONNX_HARD_SIGMOID_DEFAULT_BETA)

    inv_alpha = 1.0 / alpha
    inv_beta = beta * inv_alpha

    # This composition is built to achive ReLU6 attached to the previous convolution
    # and 1 multiplication that scales result of the ReLU
    relu = keras.layers.ReLU(max_value=inv_alpha)
    layers[node_name] = alpha * relu(input_0 + inv_beta)


def convert_tanh(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Tanh activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError("More than 1 input for an activation layer.")

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    tanh = keras.layers.Activation("tanh", name=keras_name)
    layers[node_name] = tanh(input_0)


def convert_selu(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert SELU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError("More than 1 input for an activation layer.")

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    selu = keras.layers.Activation("selu", name=keras_name)
    layers[node_name] = selu(input_0)


def convert_softmax(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert softmax activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger("onnx2keras.softmax")

    if params["change_ordering"]:
        logger.warning("change_ordering for softmax is not implemented")
    if len(node.input) != 1:
        assert AttributeError("More than 1 input for an activation layer.")

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    def target_layer(x, axis=params["axis"]):
        import tensorflow as tf

        return tf.nn.softmax(x, axis=axis)

    lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
    layers[node_name] = lambda_layer(input_0)
    layers[node_name].set_shape(layers[node_name].shape)
    lambda_func[keras_name] = target_layer


def convert_prelu(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert PReLU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger("onnx2keras.prelu")

    if len(node.input) != 2:
        assert AttributeError("Activation layer PReLU should have 2 inputs.")

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    W = ensure_numpy_type(layers[node.input[1]])

    if params["change_ordering"]:
        logger.warning(
            "PRelu + change ordering needs to be fixed after TF graph is built."
        )
        logger.warning("It's experimental.")

    shared_axes = [2, 3]

    # for case when W.shape (n,). When activation is used for single dimension vector.
    shared_axes = shared_axes if len(W.shape) > 1 else None

    prelu = keras.layers.PReLU(weights=[W], shared_axes=shared_axes, name=keras_name)
    layers[node_name] = prelu(input_0)
