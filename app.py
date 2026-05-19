import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from neuralNet import NeuralNet
from tkinter import filedialog
from tester import Tester
from mnistViewer import MNIST_viewer
# from dataset import DataSet,train_dataset,test_dataset
# import dataset no fix
# from dataset import * # this fixed
from dataset import train_dataset,test_dataset
from tweaker import TrainingTweaker
from compare import Compare
from drawCanvas import Draw_Canvas
import os

class App:
    def __init__(self,window,dataset):
        self.window=window
        self.window.geometry("400x450")
        self.dataset=dataset
        ttk.Label(self.window,text="\n").pack()
        self.button1=ttk.Button(self.window,text=" TRAIN NETWORK", command=self.train_interface)
        self.button2=ttk.Button(self.window,text=" MNIST VIEWER", command=self.show_viewer)
        self.button3=ttk.Button(self.window,text=" CANVAS DRAW PREDICTION", command=self.draw_canvas)
        self.button4=ttk.Button(self.window,text=" LOAD IN-BUILT PRE-TRAINED MODEL [16,16]", command=self.load_in_built)
        self.button5=ttk.Button(self.window,text=" LOAD RANODM UNTRAINED MODEL [16,16]", command=self.load_untrained)
        self.button6=ttk.Button(self.window,text=" FEED IN PRE-TRAINED MODEL", command=self.load_weights_biases)
        self.button7=ttk.Button(self.window,text=" COMPARE TWO MODELS", command=self.compare_models)
        self.button8=ttk.Button(self.window,text=" EVALUTE YOUR MODEL", command=self.eval_model)
        self.button1.pack()
        ttk.Label(self.window,text=" ").pack()
        self.button2.pack()
        ttk.Label(self.window,text=" ").pack()
        self.button3.pack()
        ttk.Label(self.window,text=" ").pack()
        self.button4.pack()
        ttk.Label(self.window,text=" ").pack()
        self.button5.pack()
        ttk.Label(self.window,text=" ").pack()
        self.button6.pack()
        ttk.Label(self.window,text=" ").pack()
        self.button7.pack()
        ttk.Label(self.window,text=" ").pack()
        self.button8.pack()
        self.model_trained = False
        self.model=None
        self.model1_trained = False
        self.model2_trained = False
        self.train_warn_label=ttk.Label(self.window,text="\nPlease Train a New Model OR Load Pre-Trained Model") 
        self.compare_model_1_label = ttk.Label(self.window,text="Please Load Model 1, then click compare again")
        self.compare_model_2_label = ttk.Label(self.window,text="Please Load Model 2, then click compare")
        self.compModel_1 = None
        self.compModel_2 = None

    def compare_models(self):
        self.train_warn_label.pack_forget() # need if someone want to eval a model say but havent slected any model so warning is displayed but now wished to comp 2 models so this waning must be removed
        if (not self.compare_model_2_label.winfo_ismapped()) and (( not self.compare_model_2_label.winfo_ismapped())):
            self.model=None
            self.compare_model_1_label.pack()

        if self.compare_model_1_label.winfo_ismapped() and ( not self.compare_model_2_label.winfo_ismapped()):
            self.compare_model_2_label.pack()
        if 
            self.model_1 = self.model

        self.compare_model_1_label.pack()
    def eval_model(self):
        if self.model_trained:
            self.train_warn_label.pack_forget()
            tester1=Tester(self.model,self.dataset)
            tester1.testing()
            compare_ideal=Compare(tester1,tester1,eval=True)
            compare_ideal.compare()
        else:
            self.train_warn_label.pack_forget()
            self.train_warn_label.pack()
    def load_in_built(self):
        self.train_warn_label.pack_forget()
        self.model_trained=True
        self.model=DigitNN
    def load_weights_biases(self):
        filepath = filedialog.askopenfilename(title="Open numpy files",filetypes=[("numpy_files)", "*.npz")])
        data_zip=np.load(filepath)
        layers=len(data_zip)//2
        weights_list=[data_zip[f"w_{layer}"] for layer in range(layers)]
        biases_list=[data_zip[f"b_{layer}"] for layer in range(layers)]
        self.model=NeuralNet(weights_list,biases_list)
        self.model_trained=True
    def load_untrained(self):
        #LOAD random each time
        DigitRandomNN=NeuralNet(weights=[np.random.randn(y,x)*np.sqrt(1/x) for x,y in zip([784,16,16],[16,16,10])],biases=[np.zeros((y,1)) for y in [16,16,10]])
        self.train_warn_label.pack_forget()
        self.model_trained=True
        self.model=DigitRandomNN
    def show_viewer(self):
        if hasattr(self,"window2") and self.window2.winfo_exists():
            self.window2.lift() 
            self.window2.focus()
            return
        self.window2=tk.Toplevel(self.window) 
        self.canvas=MNIST_viewer(self.window2,train_dataset)
    def train_interface(self):
        if hasattr(self,"window4") and self.window4.winfo_exists():
            self.window4.lift() 
            self.window4.focus()
            return
        self.window4=tk.Toplevel(self.window)
        self.train_interfaceX=TrainingTweaker(self.window4,self)
    def draw_canvas(self):
        if not self.model_trained:
            print("train new")
            self.train_warn_label.pack()
            return
        if hasattr(self,"window3") and self.window3.winfo_exists(): # 2nd if for case when window 3 created prev but closed rn(crossed), i suppose
            self.window3.lift()
            self.window3.focus() 
            return
        self.window3=tk.Toplevel(self.window)
        self.canvas=Draw_Canvas(self.window3,self.model)

base=os.path.dirname(os.path.abspath(__file__))
data_zip_in_built=np.load(os.path.join(base,"trainedModel","NNmodel_light2_16.npz"))
layers=len(data_zip_in_built)//2
weights_inbuilt=[data_zip_in_built[f"w_{layer}"] for layer in range(layers)]
bias_inbuilt=[data_zip_in_built[f"b_{layer}"] for layer in range(layers)]
DigitNN=NeuralNet(weights_inbuilt,bias_inbuilt)
