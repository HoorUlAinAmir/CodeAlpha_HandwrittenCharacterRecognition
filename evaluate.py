def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%\n")
