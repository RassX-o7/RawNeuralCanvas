from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from core.trainer import Trainer
import numpy as np
from core.neuralNet import NeuralNet
from core.dataset import train_dataset,test_dataset
#fix MBG error and 80k
class TrainingTweaker:
    # def __init__(self,window:tk.Tk,app:App): # future not working
    def __init__(self,window:tk.Tk,app):
        self.window=window
        self.app=app
        self.window.geometry("1000x600")
        self.window.update_idletasks()
        self.epoch_var=tk.IntVar(value=1)
        self.mbg_var=tk.IntVar(value=1)
        self.dataset_var=tk.IntVar(value=1)
        self.defualt_DG=tk.IntVar(value=0) 
        self.hyper_param=tk.DoubleVar(value=0.05)
        self.DG_type=tk.StringVar(value="SGD") # BUG fixed, radio set to FGD first BUT actual param string set to SGD , later set to default SDG cuz without momentum , sdg and small batches are effective
        self.Frame1=ttk.Frame(self.window,width=self.window.winfo_width(),height=self.window.winfo_height())
        self.Frame1.pack_propagate(False)
        ttk.Label(self.Frame1,text="This is the interface to train your model and set parameters as per your convenience\n",font=8).pack()
        ttk.Label(self.Frame1,text="set the number of epochs to train on").pack()
        self.epoch_label_var=tk.StringVar(value="You have selected 1 epochs\n") 
        self.epoch_slider= ttk.Scale(self.Frame1,variable=self.epoch_var,from_=1, to=300, orient="horizontal",command= lambda val: self.epoch_label_var.set(f"You have selected {int(float(val))} epochs"),length=300) 
        self.epoch_label=ttk.Label(self.Frame1,textvariable=self.epoch_label_var)
        self.iter_slider= ttk.Scale(self.Frame1,variable=self.dataset_var,from_=1, to=60000, orient="horizontal",length=500,command=lambda val :self.dataset_var.set(int(float(val))))
        self.rd1=ttk.Radiobutton(self.Frame1,text="Stochastic Descent Approach",command=self.SGD_tweak,value=0,variable=self.defualt_DG) 
        self.rd2=ttk.Radiobutton(self.Frame1,text="Mini Batch Descent",command=self.MGD_tweak,value=1,variable=self.defualt_DG)
        self.rd3=ttk.Radiobutton(self.Frame1,text="Full Batch Descent",command=self.FBG_tweak,value=2,variable=self.defualt_DG)
        self.MBG_slider= ttk.Scale(self.Frame1,from_=1,variable=self.mbg_var, to=1024, orient="horizontal",command= lambda val: self.mbg_var.set(int(float(val))),length=500)
        self.MBG_label=ttk.Label(self.Frame1,text="Set the Batch Size")
        self.MBG_entry_label=ttk.Label(self.Frame1,text="OR enter manually ,Max 1024")
        self.MBG_entry=tk.Entry(self.Frame1,textvariable=self.mbg_var,width=5)
        self.out_mode=tk.StringVar(value="sigmoid")
        self.output_func=ttk.Label(self.Frame1,text="Select The Output layer activation Function : ")
        self.output_func_1=ttk.Radiobutton(self.Frame1,text="Sigmoid",variable=self.out_mode,value="sigmoid")
        self.output_func_2=ttk.Radiobutton(self.Frame1,text="SoftMax",variable=self.out_mode,value="softmax")
        self.NXT_button=ttk.Button(self.Frame1,text="NEXT",command=self.next_page)
        self.sequential=tk.BooleanVar(value=False)
        self.sequential_fwd=ttk.Checkbutton(self.Frame1,text="Custom Forward Sequental (advanced)",variable=self.sequential)
        self.epoch_slider.pack()
        self.epoch_label.pack()
        ttk.Label(self.Frame1,text="set the length of train dataset").pack()
        self.iter_slider.pack()
        ttk.Label(self.Frame1,text="OR enter manually ,Max60k").pack()
        tk.Entry(self.Frame1,width=10,textvariable=self.dataset_var).pack()
        self.rd1.pack()
        self.rd2.pack()
        self.rd3.pack()
        ttk.Label(self.Frame1,text=" ").pack()
        ttk.Label(self.Frame1,text="Enter the value for Learning Rate , typical value bw 0.01 to 1").pack()
        tk.Entry(self.Frame1,textvariable=self.hyper_param,width=5).pack()
        ttk.Label(self.Frame1,text= " ").pack()
        self.output_func.pack()
        self.output_func_1.pack()
        self.output_func_2.pack()
        # self.sequential_fwd.pack()
        ttk.Label(self.Frame1,text=" ").pack()
        self.NXT_button.pack()
        self.Frame2=ttk.Frame(self.window,width=self.window.winfo_width(),height=self.window.winfo_height())
        self.Frame2.pack_propagate(False)
        self.aug = tk.BooleanVar(value=True)
        self.save_wb = tk.BooleanVar(value=False)
        self.train_visual=tk.BooleanVar(value=False)
        self.valid_visual=tk.BooleanVar(value=True)
        self.label2x=ttk.Label(self.Frame2,text="Please tweak the Parameters of the Neural Network\n",font=8).pack()
        ttk.Checkbutton(self.Frame2,text="Data Augmentation (Recommended, Slower) ",variable=self.aug,onvalue=True,offvalue=False).pack() 
        ttk.Checkbutton(self.Frame2,text="Save Weights and Biases locally" ,onvalue=True,offvalue=False,variable=self.save_wb).pack() 
        ttk.Checkbutton(self.Frame2,text="Training Visualizer ( Heavy on System )" ,onvalue=True,offvalue=False,variable=self.train_visual).pack() 
        ttk.Checkbutton(self.Frame2,text="Validation set accuracy tracking ( Heavy )",onvalue=True,offvalue=False,variable=self.valid_visual).pack()
        ttk.Label(self.Frame2,text=" ").pack()
        self.def_set=ttk.Button(self.Frame2,text="Set to default",command=self.Set_default)
        self.layers_num=tk.IntVar(value="1") 
        self.setWarn = None
        ttk.Label(self.Frame2,text="Enter the Number of MLP layers (MAX is 6): ").pack()
        self.layers_entry=ttk.Entry(self.Frame2,textvariable=self.layers_num,width=3)
        self.layers_entry.pack()
        self.layer_button=ttk.Button(self.Frame2,text="SET Layers",command=self.set_layers)
        self.layer_button.pack()
        self.neurons=ttk.Button(self.Frame2,text="SET Neurons ")
        self.back_btn=ttk.Button(self.Frame2,text="Back to Previous Page",command=self.back)
        self.back_btn.pack()
        self.setWarnlabel=ttk.Label(self.Frame2,text="Please LOCK the number of layers First")
        self.trainButton=ttk.Button(self.Frame2,text="Train Model",command=self.train_model,width=15)
        self.def_set.pack()
        self.trainButton.pack()
        self.Frame1.pack()
        self.layer_sliders_vars=[]
        self.neuron_var=tk.StringVar()
        self.neurons_label=ttk.Label(self.Frame2,textvariable=self.neuron_var)
    def Set_default(self):
        self.window.destroy()
        new_window=tk.Toplevel(self.app.window)
        self.app.train_interfaceX=TrainingTweaker(new_window,self.app)
    def back(self):
        self.Frame2.pack_forget()
        self.Frame1.pack()
    def next_page(self):
        try:
            if not 1025>int(self.mbg_var.get())>=0:
                print("please enter in range value")
                return 
        except:
            print("Invalid Entry for batch size consider pouring bleach on your eyes")
            return
        try:
            assert 60000>int(self.dataset_var.get())>0 # raises error
        except:
            print("please enter positive integer value for dataset size and itd be less than 60k")
            return
        try:
            assert 0.01<=float(self.hyper_param.get()) <=1
        except:
            print("please enter appropriate float value")
            return
        if self.defualt_DG.get() == 1:
            if self.mbg_var.get()>self.dataset_var.get() :
                print("Batch size can not be greater than dataset size")
                return
        self.Frame1.pack_forget()
        self.Frame2.pack()
    def set_layers(self):
        self.setWarn = None
        self.layer_sliders_vars = []
        num_layers = int(self.layers_num.get())
        if num_layers > 6:
            print("Max 6 layers supported")
            return
        self.layers_entry.configure(state="disabled")
        self.layer_button.configure(state="disabled")
        for layer in range(num_layers):
            var = tk.IntVar(value=1)
            self.layer_sliders_vars.append(var)
            ttk.Label(self.Frame2, text=f"Select the Number of Neurons for layer {layer+1}").pack(before=self.back_btn)  
            ttk.Scale(self.Frame2, from_=1, to=30, length=200, variable=var,
                          command=lambda val: self.sync_neuron_array()).pack(before=self.back_btn) 
        self.neurons_label.pack(before=self.back_btn)
        self.sync_neuron_array()    
        self.setWarn = True
        self.setWarnlabel.pack_forget()
    def sync_neuron_array(self):
        self.all_neurons = [var.get() for var in self.layer_sliders_vars]
        self.neuron_var.set(f"You have selected total of {self.all_neurons} Neurons")
    def train_model(self):
        if not self.setWarn: self.setWarnlabel.pack()
        else:
            self.setWarnlabel.pack_forget()
            relax=ttk.Label(self.Frame2,text="Stand back the model is being trained, Please look at terminal for progress")
            relax.pack()
            self.window.update()
            sizes=[784]+self.all_neurons+[10]
            weights=[np.random.randn(y,x)*np.sqrt(1/x) for x,y in zip(sizes[:-1],sizes[1:])]
            biases=[np.zeros((y,1)) for y in sizes[1:]]
            self.NN=NeuralNet(weights=weights,biases=biases,out_mode=self.out_mode.get())
            self.NN.show_attrs()
            trainer=Trainer(self.NN,train_dataset,epochs=self.epoch_var.get(),dataset=self.dataset_var.get(),save=self.save_wb.get(),Visulaizer=self.train_visual.get(),mode=self.DG_type.get(),batch_size=self.mbg_var.get(),hyperparam=self.hyper_param.get(),augment=self.aug.get(),validation=self.valid_visual.get())
            trainer.show_attrs()
            trainer.train()
            self.app.model=self.NN
            self.app.model_trained=True
            relax.pack_forget()
            ttk.Label(self.Frame2,text="Model is trained and LOADED , you may close this window").pack()
    def SGD_tweak(self):
        self.DG_type.set(value="SGD")
        self.MBG_slider.pack_forget()
        self.MBG_label.pack_forget()
        self.MBG_entry_label.pack_forget()
        self.MBG_entry.pack_forget()
        self.NXT_button.pack_forget()
        self.NXT_button.pack()
    def MGD_tweak(self):
        self.DG_type.set(value="MGD")
        self.MBG_label.pack()
        self.MBG_slider.pack()
        self.MBG_entry_label.pack()
        self.MBG_entry.pack()
        self.NXT_button.pack_forget()
        # ttk.Label(self.Frame1,text=" ").pack()
        self.NXT_button.pack()
    def FBG_tweak(self):
        self.DG_type.set(value="FGD")
        self.MBG_slider.pack_forget()
        self.MBG_label.pack_forget()
        self.MBG_entry.pack_forget()
        self.MBG_entry_label.pack_forget()
        self.NXT_button.pack_forget()
        self.NXT_button.pack() 