from .activation_layers import (
    convert_elu,
    convert_hard_sigmoid,
    convert_lrelu,
    convert_prelu,
    convert_relu,
    convert_selu,
    convert_sigmoid,
    convert_softmax,
    convert_tanh,
)
from .constant_layers import convert_constant
from .convolution_layers import convert_conv, convert_convtranspose
from .elementwise_layers import (
    convert_elementwise_add,
    convert_elementwise_div,
    convert_elementwise_mul,
    convert_elementwise_sub,
    convert_max,
    convert_mean,
    convert_min,
)
from .linear_layers import convert_gemm
from .normalization_layers import (
    convert_batchnorm,
    convert_dropout,
    convert_instancenorm,
    convert_lrn,
)
from .operation_layers import (
    convert_argmax,
    convert_cast,
    convert_clip,
    convert_exp,
    convert_floor,
    convert_identity,
    convert_log,
    convert_pow,
    convert_reduce_l2,
    convert_reduce_max,
    convert_reduce_mean,
    convert_reduce_sum,
    convert_split,
    convert_sqrt,
)
from .padding_layers import convert_padding
from .pooling_layers import convert_avgpool, convert_global_avg_pool, convert_maxpool
from .reshape_layers import (
    convert_concat,
    convert_expand,
    convert_flatten,
    convert_gather,
    convert_reshape,
    convert_shape,
    convert_slice,
    convert_squeeze,
    convert_transpose,
    convert_unsqueeze,
)
from .upsampling_layers import convert_upsample

AVAILABLE_CONVERTERS = {
    "Conv": convert_conv,
    "ConvTranspose": convert_convtranspose,
    "Relu": convert_relu,
    "Elu": convert_elu,
    "LeakyRelu": convert_lrelu,
    "Sigmoid": convert_sigmoid,
    "HardSigmoid": convert_hard_sigmoid,
    "Tanh": convert_tanh,
    "Selu": convert_selu,
    "Clip": convert_clip,
    "Exp": convert_exp,
    "Log": convert_log,
    "Softmax": convert_softmax,
    "PRelu": convert_prelu,
    "ReduceMax": convert_reduce_max,
    "ReduceSum": convert_reduce_sum,
    "ReduceMean": convert_reduce_mean,
    "Pow": convert_pow,
    "Slice": convert_slice,
    "Squeeze": convert_squeeze,
    "Expand": convert_expand,
    "Sqrt": convert_sqrt,
    "Split": convert_split,
    "Cast": convert_cast,
    "Floor": convert_floor,
    "Identity": convert_identity,
    "ArgMax": convert_argmax,
    "ReduceL2": convert_reduce_l2,
    "Max": convert_max,
    "Min": convert_min,
    "Mean": convert_mean,
    "Div": convert_elementwise_div,
    "Add": convert_elementwise_add,
    "Sum": convert_elementwise_add,
    "Mul": convert_elementwise_mul,
    "Sub": convert_elementwise_sub,
    "Gemm": convert_gemm,
    "MatMul": convert_gemm,
    "Transpose": convert_transpose,
    "Constant": convert_constant,
    "BatchNormalization": convert_batchnorm,
    "InstanceNormalization": convert_instancenorm,
    "Dropout": convert_dropout,
    "LRN": convert_lrn,
    "MaxPool": convert_maxpool,
    "AveragePool": convert_avgpool,
    "GlobalAveragePool": convert_global_avg_pool,
    "Shape": convert_shape,
    "Gather": convert_gather,
    "Unsqueeze": convert_unsqueeze,
    "Concat": convert_concat,
    "Reshape": convert_reshape,
    "Pad": convert_padding,
    "Flatten": convert_flatten,
    "Upsample": convert_upsample,
    "Resize": convert_upsample,
}
