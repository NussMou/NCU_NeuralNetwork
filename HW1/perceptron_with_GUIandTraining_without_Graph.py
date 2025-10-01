import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Perceptron – HW1")
        self.geometry("800x600")

        self.X = None
        self.y = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.w = None
        self.b = None

        self._build_left_panel()
        self._build_log_area()

    def _build_left_panel(self):
        frame = ttk.Frame(self, padding=10)
        frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(frame, text="Choose File", command=self.choose_file).grid(row=0, column=0, padx=5, pady=5)
        self.file_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.file_var, width=40).grid(row=0, column=1, padx=5, pady=5)

        # para inputs
        ttk.Label(frame, text="Learning rate").grid(row=1, column=0)
        self.lr_var = tk.StringVar(value="0.1")
        ttk.Entry(frame, textvariable=self.lr_var, width=10).grid(row=1, column=1)

        ttk.Label(frame, text="Max epochs").grid(row=2, column=0)
        self.ep_var = tk.StringVar(value="50")
        ttk.Entry(frame, textvariable=self.ep_var, width=10).grid(row=2, column=1)

        ttk.Label(frame, text="Target acc").grid(row=3, column=0)
        self.tacc_var = tk.StringVar(value="0.99")
        ttk.Entry(frame, textvariable=self.tacc_var, width=10).grid(row=3, column=1)

        ttk.Label(frame, text="Test ratio").grid(row=4, column=0)
        self.test_ratio_var = tk.StringVar(value="0.333")
        ttk.Entry(frame, textvariable=self.test_ratio_var, width=10).grid(row=4, column=1)

        # 動作按鈕
        ttk.Button(frame, text="data split", command=self.split_data).grid(row=5, column=0, padx=5, pady=5)
        ttk.Button(frame, text="train model", command=self.train_model).grid(row=5, column=1, padx=5, pady=5)
        ttk.Button(frame, text="test model ", command=self.test_model).grid(row=5, column=2, padx=5, pady=5)

    def _build_log_area(self):
        lf = ttk.LabelFrame(self, text="result")
        lf.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.txt = tk.Text(lf, height=20)
        self.txt.pack(fill=tk.BOTH, expand=True)

    def log(self, msg):
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)

    # ---- Actions ----
    def choose_file(self):
        path = filedialog.askopenfilename(title="choose file", filetypes=[("Text files", "*.txt")])
        if path:
            self.file_var.set(path)
            self.log(f"chosen file: {path}")
            self.X, self.y = load_dataset(path)
            self.log(f"number of input={len(self.y)}, feature dimention={self.X.shape[1]}")

    def split_data(self):
        if self.X is None:
            messagebox.showwarning("!!", "load dataset first")
            return
        test_size = float(self.test_ratio_var.get())
        self.X_train, self.X_test, self.y_train, self.y_test = split_dataset(self.X, self.y, test_ratio=test_size)
        self.log(f"split data successfully：訓練 {len(self.y_train)} 筆, 測試 {len(self.y_test)} 筆")

    def train_model(self):
        if self.X_train is None:
            messagebox.showwarning("!!", "split data first")
            return
        lr = float(self.lr_var.get())
        ep = int(self.ep_var.get())
        tacc = float(self.tacc_var.get())
        self.w, self.b, history = perceptron_train(self.X_train, self.y_train, lr=lr, epochs=ep, target_acc=tacc)
        self.log(f"[Train successfully]  train acc={history[-1]:.3f}, weight={np.round(self.w,4)}, bias={self.b:.3f}")

    def test_model(self):
        if self.w is None:
            messagebox.showwarning("!!", "Train model first")
            return
        acc, _ = perceptron_test(self.X_test, self.y_test, self.w, self.b)
        self.log(f"[Test successfully] test acc={acc:.3f}")
    

if __name__ == "__main__":
    App().mainloop()