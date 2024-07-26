import numpy as np
import matplotlib.pyplot as plt
import MNISTtools
import NeuralNetwork

def OneHot(y):
    y_one_hot = np.eye(10, dtype=np.float32)[y]
    return y_one_hot

def Accuracy(y, y_):
    y_digit = np.argmax(y, 1)
    y_digit_ = np.argmax(y_, 1)
    temp = np.equal(y_digit, y_digit_).astype(np.float32)
    return np.sum(temp) / float(y_digit.shape[0])

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')

    # Dataset
    MNISTtools.downloadMNIST(path='MNIST_data', unzip=True)
    x_train, y_train = MNISTtools.loadMNIST(dataset="training", path="MNIST_data")
    x_test, y_test = MNISTtools.loadMNIST(dataset="testing", path="MNIST_data")

    # Show Data and Label
    print(x_train[0])
    print(y_train[0])
    plt.imshow(x_train[0].reshape((28,28)), cmap='gray')
    plt.savefig('sample.png')
    plt.close()

    # Data Processing
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.

    # Create NN Model
    nn = NeuralNetwork.AutoEncoder(784, 128, 784, "sigmoid", True, 0.3)

    # Training the Model
    data_size = x_train.shape[0]
    loss_rec = []
    batch_size = 64
    for i in range(10001):
        # Sample Data Batch
        batch_id = np.random.choice(data_size, batch_size)
        x_batch = x_train[batch_id]

        # Add noise to data
        noise = np.random.normal(0.0, 1.0, size=x_batch.shape)
        noise[noise <= 0] = 0
        x_noise_batch = x_batch.copy() + noise
        x_noise_batch[x_noise_batch >= 1.0] = 1.0

        # Forward & Backward & Update
        nn.feed({"x": x_noise_batch, "y": x_batch})
        nn.forward()
        nn.backward()
        nn.update(1e-2)

        # Loss
        loss = nn.computeLoss()
        loss_rec.append(loss)

        # Evaluation
        if i % 100 == 0:
            print("\r[Iteration {:5d}] Loss={:.4f} ".format(i, loss))

    # Testing
    nn.dropout = False 
    noise = np.random.normal(0.0, 1.0, size=x_test.shape)
    noise[noise <= 0] = 0
    x_noise_test = x_test.copy() + noise
    x_noise_test[x_noise_test >= 1.0] = 1.0

    nn.feed({"x": x_noise_test, "y": x_test})
    x_noise_test_out = nn.forward()
    test_loss = nn.computeLoss()

    print(test_loss)

    plt.plot(loss_rec)
    plt.savefig('loss_plot.png')
    plt.close()

    nn.feed({"x": x_test, "y": x_test})
    x_test_out = nn.forward()

    print("Reconstruction results with noise-free inputs:")
    for i in range(1, 9):
        idx = i + 4 if i > 4 else i
        plt.subplot(4, 4, idx)
        plt.imshow(x_test[i-1].reshape((28,28)), cmap='gray')
        idx = idx + 4
        plt.subplot(4, 4, idx)
        plt.imshow(x_test_out[i-1].reshape((28,28)), cmap='gray')
    plt.savefig('reconstruction_noise_free.png')
    plt.close()

    print("Reconstruction results with noisy inputs:")
    for i in range(1, 9):
        idx = i + 4 if i > 4 else i
        plt.subplot(4, 4, idx)
        plt.imshow(x_noise_test[i-1].reshape((28,28)), cmap='gray')
        idx = idx + 4
        plt.subplot(4, 4, idx)
        plt.imshow(x_noise_test_out[i-1].reshape((28,28)), cmap='gray')
    plt.savefig('reconstruction_noisy.png')
    plt.close()

    print("Filters:")
    nn.showFilters()
