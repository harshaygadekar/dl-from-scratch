"""Level 02: composable modern conv blocks."""

from level01_naive import depthwise_conv2d, pointwise_conv2d


def depthwise_separable_conv2d(x, depthwise_w, pointwise_w, pointwise_b=None, stride=1, padding=0):
    x_dw = depthwise_conv2d(x, depthwise_w, stride=stride, padding=padding)
    return pointwise_conv2d(x_dw, pointwise_w, bias=pointwise_b)
