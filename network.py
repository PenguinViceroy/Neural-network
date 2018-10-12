import random

import numpy as np

from matplotlib import pyplot


class Network(object):
    """
        sizes 表示神经网络各个层面 e.g.[2,3,1]
        the number of epochs to train for,
        the size of the mini-batches to use when sampling.
        eta is the learning rate, η. 

        If the optional argument test_data is supplied, then the program will evaluate the network after each epoch of training, and print out partial progress.
       
        biases 以及 weights 随机初始化
    """
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """-return the output"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    

    """
        In each epoch 
            1.randomly shuffling the trainging data
            2.partitions it into mini_batches of the apropriate size
            3.for each mini_batch
                3.1 apply a  single step of gradient descent--self.update_mini_batch(...)
    """
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """ the training_data is a list of tuples (x,y) representing the training inputs and desired outputs"""   
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)    # 打乱-第一维度
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta) 
            if test_data:
                print("Epoch {0} : {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete" .format(j))
       
 
    def update_mini_batch(self, mini_batch, eta):
        """update the network's weights and biases
            by applying gradient descent using backpropagation(反向传播) to a single mini batch 
            mini_batch ---- tuples (x,y)
            eta        ---- learning rate
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]  #梯度b
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)   # ?
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]


    """"most work is done in it which use backpropagation algorithm-- a fase way of computing gradient of the cost function"""
    def backprop(self, x, y):
        """return a tuple (nabla_b, nabla_w) representing the gradient for the cost function 
            nabla_b and nabla_w are layer-by-layer lists of numpy arrays, similar to self.biases and self.weights
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #feedforward
        activation = x  # x is training input 
        activations = [x]
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])  #?
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        
        return (nabla_b, nabla_w)


    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    
    def cost_derivative(self, output_activations, y):
        """return vector of partial derivatives"""
        return (output_activations - y)
    


def sigmoid(z):
   return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """derivative of the sigmoid function"""
    return sigmoid(z)*(1-sigmoid(z))


