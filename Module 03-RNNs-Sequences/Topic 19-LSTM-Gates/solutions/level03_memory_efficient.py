"""Level 03: stream LSTM outputs per time-step."""

from level01_naive import lstm_step


def lstm_generator(x, h0, c0, w_x, w_h, b):
    h_prev, c_prev = h0, c0
    for t in range(x.shape[0]):
        h_prev, c_prev = lstm_step(x[t], h_prev, c_prev, w_x, w_h, b)
        yield h_prev, c_prev
