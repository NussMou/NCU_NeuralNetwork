import numpy as np
from sklearn.model_selection import train_test_split

"""
Only training perceptron without GUI
X is feature matrix , that is, input data
Y is expected output (label)
w is weight
b is bias
classes is the original two labels (like [1,2] or [0,1])
convert them into 1 and 2 internally for simplicity in load_dataset()
"""

def load_dataset(path):
    """
    1. load dataset from path
    2. handle the ecpected value (0 or 1 -> 1 or 2)
    """
    data = np.loadtxt(path)
    X = data[:, :-1].astype(float)
    y = data[:, -1].astype(int)

    classes = np.unique(y)
    # if len(classes) != 2:
    #     raise ValueError(f"input error: not binary classification: {classes}")

    # turn smallest one into 1, largest one into 2
    y = np.where(y == classes.min(), 1, 2)
    return X, y

def activation_binary(x):
    return 2 if x >= 0 else 1

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def split_dataset(X, y, test_ratio=1/3, seed=42):
    """
    split data into train(2/3) and test(1/3)
    """
    return train_test_split(X, y, test_size=test_ratio, random_state=seed, stratify=y)

def perceptron_train(X, y, lr=0.1, epochs=100, target_acc=0.95, random_state=42):
    """
    epochs: maximum epochs
    target_acc: target accuracy to stop training
    using rng to generate random init weights
    這裡就是把 weight 和 Xi 內積，再加上 bias ，actifvation func 計算 value，一樣就沒差，不一樣就更新
    error_cnt 計算這個 epoch 有幾個錯誤，但之後就會更正
    """
    
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    w = rng.normal(0, 0.01, size=d)
    b = 0.0

    history = []

    for epoch in range(0, epochs):
        error_cnt = 0
        for Xi, yi in zip(X, y):
            v = np.dot(w, Xi) + b
            y_hat = activation_binary(v)
            if y_hat != yi:  # if predict wrong, update weights and bias
                yi_bin = -1 if yi == 1 else 1 # using yi_bin to decide direction of update
                w += lr * yi_bin * Xi
                b += lr * yi_bin
                error_cnt += 1

        y_pred = perceptron_predict(X, w, b)
        train_acc = accuracy(y, y_pred)
        history.append(train_acc)
        print(f"Epoch {epoch:3d} | Errors {error_cnt:3d} | Train Acc {train_acc:.3f}")

        if train_acc >= target_acc:
            break

    return w, b, history

def perceptron_test(X, y, w, b):
    """
    current w, b test in (X, y)
    return accuracy and y_pred
    """
    y_pred = []
    right = 0
    for Xi, yi in zip(X, y):
        v = np.dot(w, Xi) + b
        y_hat = activation_binary(v)
        y_pred.append(y_hat)
        if y_hat == yi:
            right += 1
    acc = right / len(y)
    return acc, np.array(y_pred)

def perceptron_predict(X, w, b):
    """
    X shape is row = 4, col = 2
    w shape is row = 2
    (4*2) . (2) + (1) = (4)
    in the end return shape is row = 4
    each result turn into 1 or 2 by activation func
    """
    z = X @ w + b
    return np.array([activation_binary(v) for v in z], dtype=int)

if __name__ == "__main__":
    dataset_path = "basic/2Ccircle1.txt"
    X, y = load_dataset(dataset_path)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # training
    w, b, history = perceptron_train(
        X_train, y_train, lr=0.1, epochs=200, target_acc=0.99
    )

    # evaluation
    # put dataset and current w, b into perceptron_predict() to get y_pred
    train_acc, y_pred_train = perceptron_test(X_train, y_train, w, b)
    test_acc,  y_pred_test  = perceptron_test(X_test, y_test, w, b)


    print("\n==== Final Result ====")
    print("Train acc:", train_acc)
    print("Test acc :", test_acc)
    print("Weights :", np.round(w, 4), "Bias:", round(b, 4))
