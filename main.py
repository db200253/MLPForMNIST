import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from Optimizers.SGD import SGD
from MLP import MLP
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

layers = [784, 64, 10]
activations = ["relu", "relu", "softmax"]

optimizer = SGD(learning_rate=0.001)
model = MLP(epochs=10, tolerance=0.01, layers=layers, activations=activations, learning_rate=0.001, optimizer=optimizer)
model.fit(x_train, y_train, batch_size=32)

y_pred = model.predict(x_test)
y_pred_class = np.argmax(y_pred, axis=1)
y_test_class = np.argmax(y_test, axis=1)
accuracy = np.mean(y_pred_class == y_test_class)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {y_pred_class[i]}, Actual: {y_test_class[i]}")
    plt.show()