import numpy as np
from util import sigmoid

SUPPORTED_OPT = ['batch', 'sgd', 'mb']

class MLP():

    def __init__(self, n_inputs, n_classes, architecture):
        if isinstance(architecture, list) and (architecture[0] != n_inputs or \
                architecture[-1] != n_classes):
            raise ValueError("Input and output layer must match number of \
                    features and number of classes, respectively")
        self._weights = list()
        self._biases = list()
        for l_size, r_size in zip(architecture, architecture[1:]):
            w_std = 1 / np.sqrt(l_size)
            layer_weights = np.random.normal(0, w_std, (r_size, l_size))
            layer_bias = np.random.normal(0, 1, r_size)
            self._weights.append(layer_weights)
            self._biases.append(layer_bias)
        self._architecture = architecture

    def predict(self, x):
        if len(x) != self._architecture[0]:
            raise ValueError("Input dimensions don't correspond to input layer size")
        return self._forward_pass(x)

    def _forward_pass(self, x):
        if len(x) != self._architecture[0]:
            raise ValueError("Input dimensions don't correspond to input layer size")
        values = x
        for weights, bias in zip(self._weights, self._biases):
            z = weights.dot(values) + bias
            values = sigmoid(z)
        return values

    def _backpropagation(self, x, y_true):
        deltas = list()
        activations = [x]
        for w, b in zip(self._weights, self._biases):
            a = sigmoid(w.dot(activations[-1]) + b)
            activations.append(a)
        output_delta = (activations[-1] - y_true) * activations[-1] * (1 - activations[-1])
        deltas.append(output_delta)
        for index in range(2, len(activations)):
            weight_index = 1 - index # = - (index - 1)
            delta_index = index - 2
            tmp1 = self._weights[weight_index].T.dot(deltas[delta_index])
            tmp2 = activations[-index] * (1 - activations[-index])
            new_delta = tmp1 * tmp2
            deltas.append(new_delta)
        deltas = deltas[::-1]
        w_gradients = list()
        b_gradients = list()
        for index, delta in enumerate(deltas):
            grad_b = delta
            grad_w = np.outer(delta, activations[index])
            w_gradients.append(grad_w)
            b_gradients.append(grad_b)
        return w_gradients, b_gradients


    def fit(self, dataset, opt_method, batch_size=None, lr=0.01, epochs=100):
        """
        `batch_size` specifies number of examples per class to be used
        (5 classes with batch_size=2 gives mini-batch of size 10)
        """
        if opt_method not in SUPPORTED_OPT:
            raise ValueError("Optimization method must be one of: {}"
                    .format(SUPPORTED_OPT))
        if opt_method == 'mb' and \
                (batch_size is None or batch_size < 1 or not isinstance(batch_size, int)):
            raise ValueError("Invalid batch size value")
        for epoch in range(epochs):
            print("Starting epoch {}...".format(epoch))
            if opt_method == 'sgd':
                np.random.shuffle(dataset)
                for x, y in dataset:
                    grad_w, grad_b = self._backpropagation(x, y)
                    self._weights = [w - lr * dw for w, dw in zip(self._weights, grad_w)]
                    self._biases = [b - lr * db for b, db in zip(self._biases, grad_b)]
            elif opt_method == 'mb':
                by_classes = [[] for i in range(self._architecture[-1])]
                for x, y in dataset:
                    index = np.argmax(y)
                    by_classes[index].append((x, y))
                for index in range(len(by_classes)):
                    np.random.shuffle(by_classes[index])
                for index in np.arange(0, len(by_classes[0]), batch_size):
                    batch = list()
                    for c_index in range(self._architecture[-1]):
                        batch.extend(by_classes[c_index][index:index + batch_size])
                    grad_w = [np.zeros(w.shape) for w in self._weights]
                    grad_b = [np.zeros(b.shape) for b in self._biases]
                    for x, y in batch:
                        delta_grad_w, delta_grad_b = self._backpropagation(x, y)
                        grad_w = [gw + dgw for gw, dgw in zip(grad_w, delta_grad_w)]
                        grad_b = [gb + dgb for gb, dgb in zip(grad_b, delta_grad_b)]
                    self._weights = [w - (lr/batch_size) * gw for w, gw in zip(self._weights, grad_w)]
                    self._biases = [b - (lr/batch_size) * gb for b, gb in zip(self._biases, grad_b)]
            elif opt_method == 'batch':
                grad_w = [np.zeros(w.shape) for w in self._weights]
                grad_b = [np.zeros(b.shape) for b in self._biases]
                for x, y in dataset:
                    delta_grad_w, delta_grad_b = self._backpropagation(x, y)
                    grad_w = [gw + dgw for gw, dgw in zip(grad_w, delta_grad_w)]
                    grad_b = [gb + dgb for gb, dgb in zip(grad_b, delta_grad_b)]
                self._weights = [w - (lr/len(dataset)) * gw for w, gw in zip(self._weights, grad_w)]
                self._biases = [b - (lr/len(dataset)) * gb for b, gb in zip(self._biases, grad_b)]
            error = (1/(2*len(dataset))) * np.sum([np.square(self._forward_pass(x) - y) for x, y in dataset])
            print("Epoch {} error: {}".format(epoch, error))
