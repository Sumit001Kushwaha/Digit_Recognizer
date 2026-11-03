import numpy as np
import pandas as pd
import os

from PIL import Image, ImageOps
import matplotlib.pyplot as plt


# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "mnist1.5k.csv")
MODEL_PATH = os.path.join(BASE_DIR, "digit_model.npz")

ITERATIONS = 500
LEARNING_RATE = 0.05
DEV_SIZE = 100

np.random.seed(42)


# ==========================================
# INITIALIZE PARAMETERS
# ==========================================

def init_params():

    W1 = np.random.randn(64, 784) * np.sqrt(2 / 784)
    b1 = np.zeros((64, 1))

    W2 = np.random.randn(10, 64) * np.sqrt(2 / 10)
    b2 = np.zeros((10, 1))

    return W1, b1, W2, b2


# ==========================================
# ACTIVATION FUNCTIONS
# ==========================================

def reLu(Z):
    return np.maximum(0, Z)


def softMax(Z):

    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))

    return expZ / np.sum(expZ, axis=0, keepdims=True)


def deriv_reLu(Z):
    return Z > 0


# ==========================================
# FORWARD PROPAGATION
# ==========================================

def forward_prop(W1, b1, W2, b2, X):

    Z1 = W1.dot(X) + b1
    A1 = reLu(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = softMax(Z2)

    return Z1, A1, Z2, A2


# ==========================================
# ONE HOT ENCODING
# ==========================================

def one_hot(Y):

    one_hot_Y = np.zeros((Y.size, 10))

    one_hot_Y[np.arange(Y.size), Y] = 1

    return one_hot_Y.T


# ==========================================
# BACKWARD PROPAGATION
# ==========================================

def backward_prop(Z1, A1, Z2, A2, W2, X, Y):

    m = Y.size

    one_hot_Y = one_hot(Y)

    dZ2 = A2 - one_hot_Y

    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * deriv_reLu(Z1)

    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


# ==========================================
# UPDATE PARAMETERS
# ==========================================

def update_params(W1, b1, W2, b2,
                  dW1, db1, dW2, db2, alpha):

    W1 -= alpha * dW1
    b1 -= alpha * db1

    W2 -= alpha * dW2
    b2 -= alpha * db2

    return W1, b1, W2, b2


# ==========================================
# PREDICTION AND ACCURACY
# ==========================================

def get_predictions(A2):
    return np.argmax(A2, axis=0)


def get_accuracy(predictions, Y):

    return np.sum(predictions == Y) / Y.size


# ==========================================
# GRADIENT DESCENT
# ==========================================

def gradient_descent(X, Y, iterations, alpha):

    W1, b1, W2, b2 = init_params()

    for i in range(iterations):

        Z1, A1, Z2, A2 = forward_prop(
            W1, b1, W2, b2, X
        )

        dW1, db1, dW2, db2 = backward_prop(
            Z1, A1, Z2, A2, W2, X, Y
        )

        W1, b1, W2, b2 = update_params(
            W1, b1, W2, b2,
            dW1, db1, dW2, db2,
            alpha
        )

        if i % 50 == 0:

            accuracy = get_accuracy(
                get_predictions(A2), Y
            )

            print(f"Iteration: {i}")
            print(f"Training Accuracy: {accuracy * 100:.2f}%")
            print("----------------------------")

    return W1, b1, W2, b2


# ==========================================
# SAVE MODEL
# ==========================================

def save_model(W1, b1, W2, b2):

    try:
        np.savez(
            MODEL_PATH,
            W1=W1,
            b1=b1,
            W2=W2,
            b2=b2
        )

        print("\nModel saved successfully!")
        print("Location:", os.path.abspath(MODEL_PATH))
        print("File exists:", os.path.exists(MODEL_PATH))

    except Exception as e:
        print("Error saving model:", e)


# ==========================================
# LOAD MODEL
# ==========================================

def load_model():

    model = np.load(MODEL_PATH)

    W1 = model["W1"]
    b1 = model["b1"]

    W2 = model["W2"]
    b2 = model["b2"]

    print("Model loaded successfully!")

    return W1, b1, W2, b2


# ==========================================
# DATASET LOADING
# ==========================================

def load_dataset():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    data = pd.read_csv(DATASET_PATH)

    data = np.array(data)

    m, n = data.shape

    if n != 785:
        raise ValueError(
            f"Expected 785 columns (label + 784 pixels), found {n}."
        )

    # Shuffle dataset before splitting
    np.random.shuffle(data)

    # Development dataset
    data_dev = data[:DEV_SIZE].T

    Y_dev = data_dev[0].astype(int)
    X_dev = data_dev[1:].astype(float) / 255.0

    # Training dataset
    data_train = data[DEV_SIZE:].T

    Y_train = data_train[0].astype(int)
    X_train = data_train[1:].astype(float) / 255.0

    return X_train, Y_train, X_dev, Y_dev


# ==========================================
# DEVELOPMENT EVALUATION
# ==========================================

def evaluate_model(W1, b1, W2, b2, X_dev, Y_dev):

    _, _, _, A2 = forward_prop(
        W1, b1, W2, b2, X_dev
    )

    predictions = get_predictions(A2)

    accuracy = get_accuracy(predictions, Y_dev)

    print("\nDevelopment Accuracy:",
          f"{accuracy * 100:.2f}%")

    return accuracy


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image_path):

    img = Image.open(image_path).convert("L")
    img = ImageOps.autocontrast(img)

    img_array = np.array(img)

    # Detect background brightness
    border = np.concatenate([
        img_array[0, :],
        img_array[-1, :],
        img_array[:, 0],
        img_array[:, -1]
    ])

    if np.median(border) > 127:
        img_array = 255 - img_array

    # Use a higher threshold for detecting the digit
    # This prevents the gray image border from being included
    binary = img_array > 200

    rows, cols = np.where(binary)

    if len(rows) == 0:
        raise ValueError("No digit detected.")

    # Extract bounding box
    digit = img_array[
        rows.min():rows.max() + 1,
        cols.min():cols.max() + 1
    ]

    # Resize while maintaining aspect ratio
    digit_img = Image.fromarray(digit.astype(np.uint8))

    digit_img.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # Create 28x28 canvas
    canvas = Image.new("L", (28, 28), 0)

    x_offset = (28 - digit_img.width) // 2
    y_offset = (28 - digit_img.height) // 2

    canvas.paste(digit_img, (x_offset, y_offset))

    pixels = np.array(
        canvas,
        dtype=np.float32
    ) / 255.0

    return pixels.reshape(784, 1)


# ==========================================
# PREDICT DIGIT
# ==========================================

def predict_digit(image_path, W1, b1, W2, b2):

    X = preprocess_image(image_path)

    # Forward propagation
    _, _, _, A2 = forward_prop(
        W1, b1, W2, b2, X
    )

    probabilities = A2[:, 0]

    prediction = int(np.argmax(probabilities))

    confidence = float(probabilities[prediction]) * 100

    # Display original and processed images
    original = Image.open(image_path).convert("RGB")

    processed = (X.reshape(28, 28) * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    axes[0].imshow(original)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(processed, cmap="gray")
    axes[1].set_title("Processed 28x28 Image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    print("\nPrediction Results")
    print("----------------------------")
    print(f"Predicted Digit: {prediction}")
    print(f"Model Confidence: {confidence:.2f}%")

    return prediction


# ==========================================
# MAIN
# ==========================================

def main():

    # Load existing model or train a new one
    if os.path.exists(MODEL_PATH):

        print("Existing model found.")

        choice = input(
            "Load existing model? (y/n): "
        ).strip().lower()

        if choice == "y":

            W1, b1, W2, b2 = load_model()

        else:

            X_train, Y_train, X_dev, Y_dev = load_dataset()

            print("\nTraining started...\n")

            W1, b1, W2, b2 = gradient_descent(
                X_train,
                Y_train,
                ITERATIONS,
                LEARNING_RATE
            )

            evaluate_model(
                W1, b1, W2, b2,
                X_dev, Y_dev
            )

            save_model(W1, b1, W2, b2)

    else:

        print("No existing model found.")
        print("Training a new model...\n")

        X_train, Y_train, X_dev, Y_dev = load_dataset()

        W1, b1, W2, b2 = gradient_descent(
            X_train,
            Y_train,
            ITERATIONS,
            LEARNING_RATE
        )

        evaluate_model(
            W1, b1, W2, b2,
            X_dev, Y_dev
        )

        save_model(W1, b1, W2, b2)

    # Prediction loop
    while True:

        image_path = input(
            "\nEnter image path (or 'exit' to quit): "
        ).strip().strip('"')

        if image_path.lower() == "exit":
            break

        try:

            predict_digit(
                image_path,
                W1, b1, W2, b2
            )

        except Exception as e:

            print("Error:", e)


if __name__ == "__main__":
    main()