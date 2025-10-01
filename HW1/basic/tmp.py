import tkinter as tk
from matplotlib.figure  import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

#function

def activation_binary(v):
    if v >= 0:
        return 2
    else:
        return 1

def perceptron_binary(epoch, acc, learning_rate, dataset_path): #處理二元分類的

    #處理data
    input_data = np.loadtxt(dataset_path) #把資料讀進來一行是一大個element，每行以空白為間隔在分成一個element
    dimension = len(input_data[0]) - 1 #有一個是預期輸出
    X = input_data[:, :-1]
    y = input_data[:, -1]
    if(dataset_path == base_path / "perceptron1.txt" or dataset_path == base_path / "perceptron2.txt"):
        #因為激活函數會分類資料成1和2，需要
        y[y==1] = 2
        y[y==0] = 1
        test_data_label["text"] = "左圖圖例：\n red->predict=0\n blue->predict=1\n o->y=0\n x->y=1"  #更改label
    else:
        test_data_label["text"] = "左圖圖例：\n red->predict=1\n blue->predict=2\n o->y=1\n x->y=2"
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=1/3)
    XTrain = np.insert(XTrain, 0, -1, axis=1)  #x0 = -1
    XTest = np.insert(XTest, 0, -1, axis=1)
    w = np.random.random(dimension+1) #隨機初始化鍵結值(介於0-1)，dimension+1->w0是閥值
    weight_snapshots = []
    weight_snapshots.append(w)  #紀錄最初的鍵結值

    #訓練data
    for _ in range(epoch):
        for Xi, yi in zip(XTrain, yTrain):
            v = np.dot(w,Xi)
            y_hat = activation_binary(v)
            if y_hat != yi and v<0: #yi == 2，夾角太大
                w = w + learning_rate*Xi#有變化就紀錄
                weight_snapshots.append(w)
            elif y_hat != yi and v>=0: #yi == 1，夾角太小
                w = w - learning_rate*Xi
                weight_snapshots.append(w)#有變化就紀錄
        #check準確率
        right = 0
        for Xi, yi in zip(XTrain, yTrain):
            v = np.dot(w,Xi)
            y_hat = activation_binary(v)
            if y_hat == yi:
                right += 1
        train_valid = right/len(yTrain)
        if train_valid > acc:
            break
    train_acc_val_label["text"] = str(round(train_valid, 2))  #顯示訓練準確率
    
    weight_snapshots = np.array(weight_snapshots)
    right = 0
    predict = []
    for Xi, yi in zip(XTest, yTest):
        v = np.dot(w,Xi)
        y_hat = activation_binary(v)
        predict.append(y_hat)
        if y_hat == yi:
            right += 1
    test_acc_val_label["text"] = str(round(right/len(yTest), 2))
    predict = np.array(predict)

    #畫2D圖

    #訓練資料分布
    train_data_result.clear()
    train_data_result.set_title("Train Data Distribution")
    if(dataset_path == base_path / "perceptron1.txt" or dataset_path == base_path / "perceptron2.txt"):
        label1 = "y=0"
        label2 = "y=1"
    else:
        label1 = "y=1"
        label2 = "y=2"        
    train_data_result.scatter(XTrain[yTrain==1, 1], XTrain[yTrain==1, 2], c="red", label=label1, s=15)
    train_data_result.scatter(XTrain[yTrain==2, 1], XTrain[yTrain==2, 2], c="blue", label=label2, s=15)
    train_data_result.legend(fontsize=8, markerscale=0.8)#標示圖例
    #決策邊界 w1*x1+w2*x2+b=0 -> x2 = -(w1*x1)/w2 - b/w2 (b = -w0)
    decision_boudary_x1 = np.linspace(-10, 10, 200)
    decision_boudary_x2 = (-w[1]*decision_boudary_x1+w[0])/w[2]
    train_data_result.plot(decision_boudary_x1, decision_boudary_x2)
    #聚焦在資料顯示的部分
    train_data_result.set_xlim(XTrain[: , 1].min()-0.5, XTrain[: , 1].max()+0.5)
    train_data_result.set_ylim(XTrain[: , 2].min()-0.5, XTrain[: , 2].max()+0.5)
    canvas.draw()

    #測試資料
    yTest_marker = {1:'o', 2:'x'}
    predict_color = {1:'red', 2:'blue'}
    test_data_result.clear()
    test_data_result.set_title("Test Data Distribution")
    for i in range(len(XTest[:, 1])):
        test_color = predict_color[predict[i]]
        test_marker = yTest_marker[yTest[i]]
        test_data_result.scatter(XTest[i, 1], XTest[i, 2], c=test_color, marker=test_marker, s=15)
    #決策邊界 w1*x1+w2*x2+b=0 -> x2 = -(w1*x1)/w2 - b/w2
    decision_boudary_x1 = np.linspace(-25, 10, 400)
    decision_boudary_x2 = (-w[1]*decision_boudary_x1+w[0])/w[2]
    test_data_result.plot(decision_boudary_x1, decision_boudary_x2)
    #聚焦在資料顯示的部分
    test_data_result.set_xlim(XTest[: , 1].min()-0.5, XTest[: , 1].max()+0.5)
    test_data_result.set_ylim(XTest[: , 2].min()-0.5, XTest[: , 2].max()+0.5)
    canvas2.draw()

    #鍵結值變化
    weight_result.clear()
    weight_result.set_title("Weight Evolution")
    weight_result.plot(weight_snapshots[:, 0], label="w0", alpha=0.5)
    weight_result.plot(weight_snapshots[:, 1], label="w1", alpha=0.5)
    weight_result.plot(weight_snapshots[:, 2], label="w2", alpha=0.5)
    weight_result.legend(fontsize=8, markerscale=0.8)
    canvas3.draw()



#設定dataset路徑，避免打包成exe檔後找不到
if getattr(sys, 'frozen', False): #已打包成exe檔
    base_path = Path(sys._MEIPASS) #打包後放檔案的路徑
else:
    base_path = Path(__file__).resolve().parent #原本的路徑

#GUI設定

#基本設定
windows = tk.Tk()
windows.title("112502508_HW1")
windows.minsize(1340,620) #設定視窗最小大小

#圖片相關

#訓練資料分布圖
f = Figure(figsize=(3.5,2.5), dpi=100)
train_data_result = f.add_subplot()
train_data_result.plot()  #畫空表格
canvas = FigureCanvasTkAgg(f, windows)
canvas.draw()
canvas.get_tk_widget().grid(column=0, row=0, padx=10, pady=10)

#測試資料分布圖
f2 = Figure(figsize=(3.5,2.5), dpi=100)
test_data_result = f2.add_subplot()
test_data_result.plot()
canvas2 = FigureCanvasTkAgg(f2, windows)
canvas2.draw()
canvas2.get_tk_widget().grid(column=1, row=0, padx=10, pady=10)

test_data_label = tk.Label(windows, text="左圖圖例：\n red->predict=1\n blue->predict=2\n o->y=1\n x->y=2", font=(8), justify="left")  #justify="left"->靠左顯示
test_data_label.grid(column=2, row=0, padx=10, pady=10)

#鍵結值變化圖
f3 = Figure(figsize=(3.5,2.5), dpi=100)
weight_result = f3.add_subplot()
weight_result.plot()
canvas3 = FigureCanvasTkAgg(f3, windows)
canvas3.draw()
canvas3.get_tk_widget().grid(column=3, row=0, padx=10, pady=10)

#Dataset按鈕相關
dataset_label = tk.Label(windows, text="Dataset", font=('Arial', 15, 'bold')) #文字標籤
dataset_label.grid(column=0, row=1, padx=10, pady=10) #放進視窗
option_list = ["perceptron1.txt","perceptron2.txt","2Ccircle1.txt","2Circle1.txt","2CloseS.txt", "2CloseS2.txt", "2CloseS3.txt", "2cring.txt", "2CS.txt", "2Hcircle1.txt", "2ring.txt"]
dataset_opt = tk.StringVar()
dataset_opt.set("perceptron1.txt")  #選擇的文字
dataset_menu = tk.OptionMenu(windows, dataset_opt, *option_list) #下拉式選單
dataset_menu.grid(column=1, row=1, padx=10, pady=10)

goal_label = tk.Label(windows, text="Goal:", font=('Arial', 15, 'bold')) #bold->粗體
goal_label.grid(column=0, row=2, padx=10, pady=10)

#epoch相關
epoch_label = tk.Label(windows, text="epoch", font=(10))
epoch_label.grid(column=1, row=2, padx=10, pady=10)
epoch_box = tk.Spinbox(windows, from_=1, to=500, width=10) #可上下調節數值的輸入框
epoch_box.grid(column=2, row=2,padx=10, pady=10)

space_label = tk.Label(windows, text="or", font=(10))  #文字標籤
space_label.grid(column=1, row=3, padx=30, pady=10)  #放進視窗

#準確率相關
acc_goal_label = tk.Label(windows, text="accuracy(%)", font=(10))
acc_goal_label.grid(column=1, row=4, padx=10, pady=10)
acc_goal_box = tk.Spinbox(windows, from_=1, to=100, increment=0.5, format= "%4.2f", width=10)  #format= "%4.2f" -> 可顯示4個數字，包含兩個小數點
acc_goal_box.grid(column=2, row=4,padx=10, pady=10)

#學習率相關
learning_rate_label = tk.Label(windows, text="learning_rate:", font=('Arial', 15, 'bold'))
learning_rate_label.grid(column=0, row=5, padx=10, pady=10)
learning_rate_box = tk.Spinbox(windows, from_=0.1, to=1, increment=0.1, format= "%5.2f", width=10)
learning_rate_box.grid(column=1, row=5,padx=10, pady=10)

#訓練結果相關
result_label = tk.Label(windows, text="Result:", font=('Arial', 15, 'bold'))
result_label.grid(column=0, row=6, padx=10, pady=10)
train_acc_label = tk.Label(windows, text="Training accuracy:", font=(10))
train_acc_label.grid(column=1, row=6, padx=10, pady=10)
train_acc_val_label = tk.Label(windows, text="...", font=(10))
train_acc_val_label.grid(column=2, row=6, padx=10, pady=10)
test_acc_label = tk.Label(windows, text="Test accuracy:", font=(10))
test_acc_label.grid(column=3, row=6, padx=10, pady=10)
test_acc_val_label = tk.Label(windows, text="...", font=(10), justify="left")
test_acc_val_label.grid(column=4, row=6, padx=10, pady=10)

#Train相關
def train():
    epoch = int(epoch_box.get())
    acc = float(acc_goal_box.get())*0.01
    learning_rate = float(learning_rate_box.get())
    dataset_name = dataset_opt.get()
    dataset_path = base_path / dataset_name
    perceptron_binary(epoch, acc, learning_rate,  dataset_path)

train_btn = tk.Button(windows, text="Train", background="white", command=train, font=(10))  #按鍵，command->按下按鈕要執行的事
train_btn.grid(column=0, row=7,padx=10, pady=10)

exit_btn = tk.Button(windows, text="Exit", background="white", command=lambda:windows.destroy(), font=(10)) #按下去關視窗
exit_btn.grid(column=4, row=7,padx=15, pady=10)

windows.mainloop() #放在主迴圈中，保持視窗開啟，直到關閉視窗才結束