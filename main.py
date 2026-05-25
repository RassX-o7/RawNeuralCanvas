import tkinter as tk
from ui.app import App
from core.dataset import DataSet

def main():
    train_dataset = DataSet(mode="train")
    test_dataset = DataSet(mode="test")

    window1=tk.Tk()
    window1.title("Digit Prediction")

    AppX = App(window1,train_dataset)

    window1.mainloop()

if __name__=="__main__":
    main()