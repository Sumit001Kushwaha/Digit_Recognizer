
import numpy as np 
import pandas as pd 
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd

data = pd.read_csv('mnist1.5k.csv')
data.head()

data = np.array(data)
m,n = data.shape
data_dev = data[0:100,].T
Y_dev = data_dev[0]
x_dev = data_dev[1:n]/255.0

data_train = data[100:m].T
x_train = data_train[1:n]/255.0
Y_train = data_train[0]

x_train.shape

def init_params():
    W1 = np.random.randn(10,784)*np.sqrt(2/784)
    b1 = np.random.randn(10,1)
    W2 = np.random.randn(10,10)*np.sqrt(2/10)
    b2 = np.random.randn(10,1)
    return W1,b1,W2,b2

def reLu(Z):
    return np.maximum(0,Z)

def softMax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_prop(W1,b1,W2,b2,X):
    Z1 = W1.dot(X)+b1
    A1 = reLu(Z1)
    Z2 = W2.dot(A1)+b2
    A2 = softMax(Z2)
    return Z1,A1,Z2,A2

def deriv_reLu(Z):
    return Z>0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size,Y.max()+1))
    one_hot_Y[np.arange(Y.size),Y] = 1
    return one_hot_Y.T

def backward_prop(Z1,A1,Z2,A2,W2,X,Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m*dZ2.dot(A1.T)
    db2 = 1/m*np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = W2.T.dot(dZ2)*deriv_reLu(Z1)
    dW1 = 1/m*dZ1.dot(X.T)
    db1 = 1/m*np.sum(dZ1,axis=1,keepdims=True)
    return dW1,db1,dW2,db2

def update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha):
    W1 = W1 - alpha*dW1
    b1 = b1 - alpha*db1
    W2 = W2 - alpha*dW2
    b2 = b2 - alpha*db2
    return W1,b1,W2,b2

# %% [code] {"execution":{"iopub.status.busy":"2026-04-30T21:07:28.481999Z","iopub.execute_input":"2026-04-30T21:07:28.482453Z","iopub.status.idle":"2026-04-30T21:07:28.490741Z","shell.execute_reply.started":"2026-04-30T21:07:28.482402Z","shell.execute_reply":"2026-04-30T21:07:28.489651Z"}}
def get_predictions(A2):
    return np.argmax(A2,0)

def get_accuracy(predictions,Y):
    print(predictions,Y)
    return np.sum(predictions==Y)/Y.size

def gradient_descent(x,Y,iterations,alpha):
    W1,b1,W2,b2 = init_params()
    for i in range(iterations):
        Z1,A1,Z2,A2 = forward_prop(W1,b1,W2,b2,x)
        dW1,db1,dW2,db2 = backward_prop(Z1,A1,Z2,A2,W2,x,Y)
        W1,b1,W2,b2 = update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha)
        if i%50==0:
            print("iteration: ",i)
            print("accuracy: ",get_accuracy(get_predictions(A2),Y))
    return W1,b1,W2,b2

W1,b1,W2,b2 = gradient_descent(x_train,Y_train,500,0.05)

def preprocess_image(image_path):
    # Open image and convert to grayscale
    img = Image.open(image_path).convert("L")
    img = ImageOps.autocontrast(img)

    img_array = np.array(img)

    # Detect background brightness using the image borders
    border = np.concatenate([
        img_array[0, :],
        img_array[-1, :],
        img_array[:, 0],
        img_array[:, -1]
    ])

    # MNIST uses a black background and a bright digit
    if np.median(border) > 127:
        img_array = 255 - img_array

    # Convert to a binary image to isolate the digit
    binary = img_array > 60

    # Find the bounding box of the digit
    rows, cols = np.where(binary)

    if len(rows) == 0:
        raise ValueError("No digit detected in the image.")

    digit = img_array[
        rows.min():rows.max() + 1,
        cols.min():cols.max() + 1
    ]

    # Resize while maintaining aspect ratio
    digit_img = Image.fromarray(digit.astype(np.uint8))
    digit_img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Place the digit on a 28x28 black canvas
    canvas = Image.new("L", (28, 28), 0)

    x_offset = (28 - digit_img.width) // 2
    y_offset = (28 - digit_img.height) // 2

    canvas.paste(digit_img, (x_offset, y_offset))

    # Normalize pixel values and reshape to (784, 1)
    pixels = np.array(canvas, dtype=np.float32) / 255.0
    return pixels.reshape(784, 1)


def predict_digit(image_path):
    # Preprocess the input image
    X = preprocess_image(image_path)

    # Run the trained neural network
    Z1, A1, Z2, A2 = forward_prop(
        W1, b1, W2, b2, X
    )

    # Get predicted digit and confidence
    probabilities = A2[:, 0]
    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction]) * 100

    # Display the original image and processed image
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

    print(f"Predicted digit: {prediction}")
    print(f"Model confidence: {confidence:.2f}%")

    return prediction
