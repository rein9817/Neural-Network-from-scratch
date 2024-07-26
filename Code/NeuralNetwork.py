import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

class NN(): 
    def __init__(self, input_size, hidden_size, output_size, activation):
        
        # input layer to hidden layer
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros([1, hidden_size])
        # hidden layer to output layer
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(input_size)
        self.b2 = np.zeros([1, output_size])
        self.activation = activation
        self.placeholder = {"x": None, "y": None}
    
    # Feed Placeholder
    def feed(self, feed_dict):
        for key in feed_dict:
            self.placeholder[key] = feed_dict[key].copy()
    
    
    # Forward Propagation
    def forward(self):
        n = self.placeholder["x"].shape[0]
        self.a1 = self.placeholder["x"].dot(self.W1) + np.ones((n, 1)).dot(self.b1)
        self.h1 = np.maximum(self.a1, 0)  # ReLU activation
        self.a2 = self.h1.dot(self.W2) + np.ones((n, 1)).dot(self.b2)
        
        if self.activation == "linear":
            self.y = self.a2.copy()
        elif self.activation == "sigmoid":
            self.y = 1.0 / (1.0 + np.exp(-self.a2))
        elif self.activation == "softmax":
            self.y_logit = np.exp(self.a2 - np.max(self.a2, axis=1, keepdims=True))
            self.y = self.y_logit / np.sum(self.y_logit, axis=1, keepdims=True)
            
        return self.y
    
    # Backward Propagation
    def backward(self):
        n = self.placeholder["y"].shape[0]
        self.grad_a2 = (self.y - self.placeholder["y"]) / n
        self.grad_b2 = np.ones((n, 1)).T.dot(self.grad_a2)
        self.grad_W2 = self.h1.T.dot(self.grad_a2)

        self.grad_h1 = self.grad_a2.dot(self.W2.T)
        
        # a bit of problem here
        self.grad_a1 = self.grad_h1 * (self.a1 > 0)  # ReLU derivative ()
        
        self.grad_b1 = np.ones((n, 1)).T.dot(self.grad_a1)
        
        self.grad_W1 = self.placeholder["x"].T.dot(self.grad_a1)
    
    # Update Weights
    def update(self, learning_rate=1e-3):
        self.W1 = self.W1 - learning_rate * self.grad_W1
        self.b1 = self.b1 - learning_rate * self.grad_b1
        self.W2 = self.W2 - learning_rate * self.grad_W2
        self.b2 = self.b2 - learning_rate * self.grad_b2
    
    # Loss Functions
    def computeLoss(self):
        if self.activation == "linear":
            loss = 0.5 * np.square(self.y - self.placeholder["y"]).mean()
        elif self.activation == "sigmoid":
            loss = -self.placeholder["y"] * np.log(self.y + 1e-6) - (1 - self.placeholder["y"]) * np.log(1 - self.y + 1e-6)
            loss = np.mean(loss)
        elif self.activation == "softmax":
            loss = -self.placeholder["y"] * np.log(self.y + 1e-6)
            loss = np.sum(loss, axis=1).mean()
        return loss


class AnotherNN():
    def __init__(self, input_size, hidden_size1, hidden_size2 , output_size, activation):
        self.W1 = np.random.randn(input_size, hidden_size1) / np.sqrt(input_size)
        self.b1 = np.zeros([1, hidden_size1])
        
        # hidden layer 1 to hidden layer 2
        self.W2 = np.random.randn(hidden_size1, hidden_size2) / np.sqrt(hidden_size1)
        self.b2 = np.zeros([1, hidden_size2])
        
        # hidden layer 2 to output layer
        self.W3 = np.random.randn(hidden_size2, output_size) / np.sqrt(hidden_size2)
        self.b3 = np.zeros([1, output_size])
        
        self.activation = activation
        self.placeholder = {"x": None, "y": None}
    
    # Feed Placeholder
    def feed(self, feed_dict):
        for key in feed_dict:
            self.placeholder[key] = feed_dict[key].copy()
    
    
    # Forward Propagation
    def forward(self):
        n = self.placeholder["x"].shape[0]
        self.a1 = self.placeholder["x"].dot(self.W1) + np.ones((n, 1)).dot(self.b1)
        self.h1 = np.maximum(self.a1, 0) 
        
        self.a2 = self.h1.dot(self.W2) + np.ones((n, 1)).dot(self.b2)
        self.h2 = np.maximum(self.a2, 0)  

        self.a3 = self.h2.dot(self.W3) + np.ones((n, 1)).dot(self.b3)
        
        if self.activation == "linear":
            self.y = self.a3.copy()
        elif self.activation == "sigmoid":
            self.y = 1.0 / (1.0 + np.exp(-self.a3))
        elif self.activation == "softmax":
            self.y_logit = np.exp(self.a3 - np.max(self.a3, axis=1, keepdims=True))
            self.y = self.y_logit / np.sum(self.y_logit, axis=1, keepdims=True)
            
        return self.y
    
    # Backward Propagation
    def backward(self):
        
        n = self.placeholder["y"].shape[0]
        self.grad_a3 = (self.y - self.placeholder["y"]) / n
        self.grad_b3 = np.ones((n, 1)).T.dot(self.grad_a3)
        self.grad_W3 = self.h2.T.dot(self.grad_a3)
    
        self.grad_h2 = self.grad_a3.dot(self.W3.T)
        self.grad_a2 = self.grad_h2 * (self.a2 > 0)
        self.grad_b2 = np.ones((n, 1)).T.dot(self.grad_a2)
        self.grad_W2 = self.h1.T.dot(self.grad_a2)

        self.grad_h1 = self.grad_a2.dot(self.W2.T)
        self.grad_a1 = self.grad_h1 * (self.a1 > 0)
        self.grad_b1 = np.ones((n, 1)).T.dot(self.grad_a1)
        self.grad_W1 = self.placeholder["x"].T.dot(self.grad_a1)
    
    # Update Weights
    def update(self, learning_rate=1e-3):
        self.W1 = self.W1 - learning_rate * self.grad_W1
        self.b1 = self.b1 - learning_rate * self.grad_b1
        self.W2 = self.W2 - learning_rate * self.grad_W2
        self.b2 = self.b2 - learning_rate * self.grad_b2
        self.W3 = self.W3 - learning_rate * self.grad_W3
        self.b3 = self.b3 - learning_rate * self.grad_b3

    # Loss Functions
    def computeLoss(self):
        if self.activation == "linear":
            loss = 0.5 * np.square(self.y - self.placeholder["y"]).mean()
        elif self.activation == "sigmoid":
            loss = -self.placeholder["y"] * np.log(self.y + 1e-6) - (1 - self.placeholder["y"]) * np.log(1 - self.y + 1e-6)
            loss = np.mean(loss)
        elif self.activation == "softmax":
            loss = -self.placeholder["y"] * np.log(self.y + 1e-6)
            loss = np.sum(loss, axis=1).mean()
        return loss
    
import numpy as np
import matplotlib.pyplot as plt

class AutoEncoder():
    def __init__(self, input_size, hidden_size, output_size, activation, dropout, dropout_rate):
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros([1, hidden_size])
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(input_size)
        self.b2 = np.zeros([1, output_size])

        self.activation = activation
        self.placeholder = {"x": None, "y": None}
        self.dropout = dropout
        self.dropout_rate = dropout_rate

    def showFilters(self):
        plt.figure(figsize=(10, 10))
        for i in range(0, 16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(self.W1[:, i].reshape((28, 28)), cmap='gray')
            plt.axis('off')
            
        plt.savefig("filters.png")
        

    def grad_relu(self, x):
        return (x > 0).astype(float)

    def feed(self, feed_dict):
        for key in feed_dict:
            self.placeholder[key] = feed_dict[key].copy()

    def encoder(self, x):
        n = x.shape[0]
        a1 = x.dot(self.W1) + np.ones((n, 1)).dot(self.b1)
        h1 = np.maximum(a1, 0)

        if self.dropout:
            self.u1 = np.random.binomial(1, 1.0 - self.dropout_rate, h1.shape) / (1.0 - self.dropout_rate)
            h1 = h1 * self.u1
        return h1, a1

    def decoder(self, h):
        n = h.shape[0]
        a2 = h.dot(self.W2) + np.ones((n, 1)).dot(self.b2)

        if self.activation == "linear":
            y = a2.copy()
        elif self.activation == "softmax":
            y_logit = np.exp(a2 - np.max(a2, 1, keepdims=True))
            y = y_logit / np.sum(y_logit, 1, keepdims=True)
        elif self.activation == "sigmoid":
            y = 1.0 / (1.0 + np.exp(-a2))
        return y, a2

    def forward(self):
        self.h1, self.a1 = self.encoder(self.placeholder["x"])
        self.y, self.a2 = self.decoder(self.h1)
        return self.y

    def backward(self):
        n = self.placeholder["y"].shape[0]
        self.grad_a2 = (self.y - self.placeholder["y"]) / n
        self.grad_b2 = np.ones((n, 1)).T.dot(self.grad_a2)
        self.grad_W2 = self.h1.T.dot(self.grad_a2)
        self.grad_h1 = self.grad_a2.dot(self.W2.T)
        if self.dropout:
            self.grad_h1 *= self.u1

        self.grad_a1 = self.grad_h1 * self.grad_relu(self.a1)
        self.grad_b1 = np.ones((n, 1)).T.dot(self.grad_a1)
        self.grad_W1 = self.placeholder["x"].T.dot(self.grad_a1)

    def update(self, learning_rate=1e-3):
        self.W1 -= learning_rate * self.grad_W1
        self.b1 -= learning_rate * self.grad_b1
        self.W2 -= learning_rate * self.grad_W2
        self.b2 -= learning_rate * self.grad_b2

    def computeLoss(self):
        if self.activation == "linear":
            loss = 0.5 * np.square(self.y - self.placeholder["y"]).mean()
        elif self.activation == "softmax":
            loss = -np.sum(self.placeholder["y"] * np.log(self.y + 1e-6), axis=1).mean()
        elif self.activation == "sigmoid":
            loss = -np.mean(self.placeholder["y"] * np.log(self.y + 1e-6) +
                            (1 - self.placeholder["y"]) * np.log(1 - self.y + 1e-6))
        return loss



