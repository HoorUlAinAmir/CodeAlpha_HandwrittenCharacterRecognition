from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from model import build_model


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)


def train_model(x_train, y_train, epochs):
    model = build_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, validation_split=0.1)
    return model, history