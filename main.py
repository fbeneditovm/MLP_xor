import numpy as np


class MLPSingleHiddenLayer:
    def __init__(self, input_layer_size=2, hidden_layer_size=2, output_layer_size=1, learning_rate=0.1):

        # Check type of arguments
        assert isinstance(input_layer_size, int)
        assert isinstance(hidden_layer_size, int)
        assert isinstance(output_layer_size, int)

        # Assign values
        self.inLS = input_layer_size
        self.hidLS = hidden_layer_size
        self.outLS = output_layer_size
        self.alpha = learning_rate
        self.hidNeurons = None
        self.outNeurons = None
        self.hidResults = None
        self.outResults = None
        self.x = np.array([])

        # Random Forward Propagation Weights
        # An extra position was added to *input vector*
        # so that the bias can be within the weight matrix
        self.w1 = np.random.rand(self.inLS+1, self.hidLS)
        self.w2 = np.random.rand(self.hidLS+1, self.outLS)

    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))

    def forward_propagate(self, x):
        # insert a column of 1's as the first entry in the feature
        # vector -- this is a little trick that allows us to treat
        # the bias as a trainable parameter *within* the weight matrix
        # rather than an entirely separate variable
        x = np.append(x, 1)

        self.hidNeurons = np.dot(x, self.w1)
        self.hidResults = [self.sigmoid(hidNeuron) for hidNeuron in self.hidNeurons]
        self.hidResults = np.append(self.hidResults, 1)
        self.outNeurons = np.dot(self.hidResults, self.w2)
        self.outResults = self.sigmoid(self.outNeurons)
        return self.outResults

    def back_propagate(self, x, y):

        x = np.append(x, 1)

        out_err_grad = self.outResults*(1-self.outResults)*(y-self.outResults)  # *

        out_delta = self.hidResults*self.alpha
        out_delta = out_delta*out_err_grad
        self.w2 = self.w2+out_delta

        ones = np.ones(len(self.hidResults))
        hid_err_grad = self.hidResults*(ones-self.hidResults)*out_err_grad*self.w2.transpose()[0]
        hid_delta = x * self.alpha
        hid_delta = hid_delta * hid_err_grad
        for i in range(self.w1.shape[0]):
            self.w1[i] = self.w1[i]+hid_delta[i]

    def train_network(self, features, classes, max_epochs=1000):

        self.x = np.c_[features, np.ones((features.shape[0]))]
        for i in range(max_epochs):
            err_sum = 0

            for j in range(len(features)):
                x = features[i]
                y = classes[i]
                output = self.forward_propagate(x)
                err_sum += y-output  # *
                self.back_propagate(x, y)

            print("%.3f" % err_sum)
            if err_sum == 0:
                break

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 1, 1, 0])
nn = MLPSingleHiddenLayer()
nn.train_network(inputs, outputs)
