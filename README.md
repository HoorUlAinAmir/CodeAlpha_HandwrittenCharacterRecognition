"""
# CodeAlpha Handwritten Digit Recognition

This project uses a Convolutional Neural Network (CNN) to recognize handwritten digits using the MNIST dataset.

## Tech Stack
- Python
- TensorFlow / Keras
- CNN (Convolutional Neural Network)

## Dataset
- [MNIST](http://yann.lecun.com/exdb/mnist/): Built-in TensorFlow dataset with 60,000 training and 10,000 test grayscale digit images (28x28).

## How to Run
```bash
pip install -r requirements.txt
python main.py --epochs 5 --save_model
```

## Output
- Trained model saved as `digit_cnn_model.h5`
- Accuracy and loss plots saved as `accuracy.png` and `loss.png`

## Credits
- CodeAlpha Internship Project
"""