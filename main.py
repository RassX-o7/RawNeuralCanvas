import numpy as np
from tqdm import tqdm
import time,random
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from PIL import Image,ImageDraw,ImageTk
from tkinter import filedialog
import os
import threading
from app import App
from dataset import DataSet

#add kill functionality 
#if app killed , everything else
class ComapreModels():
    ...
class EvaluateModel():
    #implement visualizer index , type of evaluation? want confusion matrix etc
    #but evaluation mean everything
    ...
def main():
    train_dataset = DataSet(mode="train")
    test_dataset = DataSet(mode="test")

    window1=tk.Tk()
    window1.title("Digit Prediction")

    AppX = App(window1,train_dataset)

    window1.mainloop()
print(__name__)
if __name__=="__main__":
    main()