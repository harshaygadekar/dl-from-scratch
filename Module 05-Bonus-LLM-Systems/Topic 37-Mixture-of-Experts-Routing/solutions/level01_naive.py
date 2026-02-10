import numpy as np


def top1_route(router_logits):
    return np.argmax(router_logits, axis=-1)


def top2_route(router_logits):
    order = np.argsort(router_logits, axis=-1)
    return order[:, -2:][:, ::-1]
