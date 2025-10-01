import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def load_dataset(path):
    data = np.loadtxt(path)
    X = data[:, :-1].astype(float)
    y = data[:, -1].astype(int)
    classes = np.unique(y)
    y = np.where(y == classes.min(), 1, 2)
    return X, y

def activation_binary(x):
    return 2 if x >= 0 else 1

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def split_dataset(X, y, test_ratio=1/3, seed=42):
    return train_test_split(X, y, test_size=test_ratio, random_state=seed, stratify=y)

def perceptron_train(X, y, lr=0.1, epochs=100, target_acc=0.95, random_state=42):
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    w = rng.normal(0, 0.01, size=d)
    b = 0.0
    history = []
    for epoch in range(epochs):
        error_cnt = 0
        for Xi, yi in zip(X, y):
            v = np.dot(w, Xi) + b
            y_hat = activation_binary(v)
            if y_hat != yi:
                yi_bin = -1 if yi == 1 else 1
                w += lr * yi_bin * Xi
                b += lr * yi_bin
                error_cnt += 1
        y_pred = perceptron_predict(X, w, b)
        train_acc = accuracy(y, y_pred)
        history.append(train_acc)
        if train_acc >= target_acc:
            break
    return w, b, history

def perceptron_test(X, y, w, b):
    y_pred = [activation_binary(np.dot(w, Xi) + b) for Xi in X]
    acc = accuracy(y, y_pred)
    return acc, np.array(y_pred)

def perceptron_predict(X, w, b):
    z = X @ w + b
    return np.array([activation_binary(v) for v in z], dtype=int)


# ============ APP interface ==============
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Perceptron – HW1")
        self.geometry("1100x700")

        self.X = None
        self.y = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.w = None
        self.b = None

        self._build_left_panel()
        self._build_plot_area()
        self._build_log_area()

    def _build_left_panel(self):
        frame = ttk.Frame(self, padding=10)
        frame.pack(side=tk.LEFT, fill=tk.Y)

        # 檔案選擇
        ttk.Button(frame, text="choose file", command=self.choose_file).pack(pady=5)
        self.file_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.file_var, width=32).pack(pady=5)

        # 參數設定
        self.lr_var = tk.StringVar(value="0.1")
        self.ep_var = tk.StringVar(value="50")
        self.tacc_var = tk.StringVar(value="0.99")
        self.test_ratio_var = tk.StringVar(value="0.333")
        for label, var in [
            ("Learning rate", self.lr_var),
            ("Max epochs", self.ep_var),
            ("Target acc", self.tacc_var),
            ("Test ratio", self.test_ratio_var),
        ]:
            ttk.Label(frame, text=label).pack()
            ttk.Entry(frame, textvariable=var, width=12).pack(pady=2)

        ttk.Label(frame, text="X axis feature index").pack()
        self.dim1_var = tk.StringVar(value="0")
        ttk.Entry(frame, textvariable=self.dim1_var, width=6).pack()
        ttk.Label(frame, text="Y axis feature index").pack()
        self.dim2_var = tk.StringVar(value="1")
        ttk.Entry(frame, textvariable=self.dim2_var, width=6).pack()

        ttk.Button(frame, text="split dataset", command=self.split_data).pack(pady=5)
        ttk.Button(frame, text="train model", command=self.train_model).pack(pady=5)
        ttk.Button(frame, text="test model", command=self.test_model).pack(pady=5)
        ttk.Button(frame, text="clear", command=self.clear_plot).pack(pady=5)

    def _build_plot_area(self):
        self.fig = Figure(figsize=(6,5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(" 114522055 ")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def _build_log_area(self):
        lf = ttk.LabelFrame(self, text="result")
        lf.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.txt = tk.Text(lf, height=12)
        self.txt.pack(fill=tk.BOTH, expand=True)

    def log(self, msg):
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)

    # ---- Actions ----
    def choose_file(self):
        path = filedialog.askopenfilename(title="choose file", filetypes=[("Text files", "*.txt")])
        if path:
            self.file_var.set(path)
            self.X, self.y = load_dataset(path)
            self.log(f"load successfully: {path}, numSample={len(self.y)}, feature dim={self.X.shape[1]}")
            self.plot_data()

    def split_data(self):
        if self.X is None:
            messagebox.showwarning("!!", "load data first")
            return
        test_size = float(self.test_ratio_var.get())
        self.X_train, self.X_test, self.y_train, self.y_test = split_dataset(self.X, self.y, test_ratio=test_size)
        self.log(f"切分完成: 訓練 {len(self.y_train)} 筆, 測試 {len(self.y_test)} 筆")
        self.plot_data()

    def train_model(self):
        if self.X_train is None:
            messagebox.showwarning("!!", "split data first")
            return
        lr = float(self.lr_var.get())
        ep = int(self.ep_var.get())
        tacc = float(self.tacc_var.get())
        self.w, self.b, history = perceptron_train(self.X_train, self.y_train, lr=lr, epochs=ep, target_acc=tacc)
        self.log(f"[Train successfully] The final train acc={history[-1]:.3f}, w={np.round(self.w,4)}, b={self.b:.3f}")
        self.plot_data(decision_boundary=True, use_train=True)

    def test_model(self):
        if self.w is None:
            messagebox.showwarning("!!", "train model first")
            return
        acc, _ = perceptron_test(self.X_test, self.y_test, self.w, self.b)
        self.log(f"[test Result] test acc={acc:.3f}")
        self.plot_data(decision_boundary=True, use_train=False)

    def clear_plot(self):
        self.ax.clear()
        self.ax.set_title("114522055")
        self.canvas.draw_idle()

    # ---- Plotting ----
    def plot_data(self, decision_boundary=False, use_train=True):
        if self.X is None:
            return
        self.ax.clear()

        try:
            d1 = int(self.dim1_var.get())
            d2 = int(self.dim2_var.get())
        except:
            d1, d2 = 0, 1

        if self.X.shape[1] < 2:
            self.ax.text(0.5, 0.5, "input error", ha="center")
            self.canvas.draw_idle()
            return

        # choose train/test
        Xshow = self.X_train if (use_train and self.X_train is not None) else self.X_test
        yshow = self.y_train if (use_train and self.y_train is not None) else self.y_test
        if Xshow is None: return

        self.ax.scatter(Xshow[yshow==1, d1], Xshow[yshow==1, d2], c="red", label="class 1", alpha=0.7)
        self.ax.scatter(Xshow[yshow==2, d1], Xshow[yshow==2, d2], c="blue", label="class 2", alpha=0.7)

        if decision_boundary and self.w is not None and len(self.w) >= 2:
            x_min, x_max = Xshow[:, d1].min()-0.5, Xshow[:, d1].max()+0.5
            xs = np.linspace(x_min, x_max, 200)
            if abs(self.w[d2]) > 1e-12:
                ys = -(self.w[d1]*xs + self.b) / self.w[d2]
                self.ax.plot(xs, ys, 'k-', linewidth=2)
            else:
                x0 = -self.b / self.w[d1]
                self.ax.axvline(x=x0, linewidth=2)

        self.ax.set_xlabel(f"feature[{d1}]")
        self.ax.set_ylabel(f"feature[{d2}]")
        self.ax.legend(loc="best")
        self.ax.set_title("114522055")
        self.canvas.draw_idle()

if __name__ == "__main__":
    App().mainloop()
