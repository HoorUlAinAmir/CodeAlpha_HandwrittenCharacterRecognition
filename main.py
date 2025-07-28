import argparse
from train import load_data, train_model
from evaluate import evaluate_model
from visualize import plot_training
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--save_model', action='store_true')
args = parser.parse_args()

(x_train, y_train), (x_test, y_test) = load_data()
model, history = train_model(x_train, y_train, args.epochs)

evaluate_model(model, x_test, y_test)
plot_training(history)

if args.save_model:
    model.save("digit_cnn_model.h5")
    print("Model saved as digit_cnn_model.h5")
