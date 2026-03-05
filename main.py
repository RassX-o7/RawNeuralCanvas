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
class NeuralNet:
    def __init__(self,weights,biases):
        self.weights_list=weights
        self.biases_list=biases
        layer_sizesx=[list(layer.shape)[1] for layer in self.weights_list]+[(self.weights_list[-1].shape)[0]]
        self.num_layers=len(layer_sizesx)
        self.layer_sizes=layer_sizesx
        self.model_activations=[np.zeros((y,1)) for y in self.layer_sizes]
        self.delta_list=[np.zeros((y,1)) for y in self.layer_sizes[1:]]
    @staticmethod
    def _sigmoid(input):
        return 1/(1+np.exp(-input))
    @staticmethod
    def _weightedSum(weight,activation,bias):
        z=np.dot(weight,activation) + bias
        return z
    @staticmethod
    def _activation(z):
        return NeuralNet._sigmoid(z)
    @staticmethod
    def _delta_L(activation_L,loss_matrix):
        return 2*np.multiply(np.multiply(activation_L,1-activation_L),activation_L-loss_matrix)
    @staticmethod
    def _delC__delB_x(delta_x):
        return delta_x
    @staticmethod
    def _delC__delW_x(delta_x_1,activation_x):
        return np.dot(delta_x_1,activation_x.T)
    @staticmethod
    def _delta_x(activation_x,delta_x_1,weight_x):
        return np.multiply(activation_x*(1-activation_x),weight_x.T@delta_x_1)
    def forward(self,input_matrix):
        self.model_activations[0]=input_matrix
        for layer in range(self.num_layers-1):
            self.model_activations[layer+1]=NeuralNet._sigmoid(NeuralNet._weightedSum(self.weights_list[layer],self.model_activations[layer],self.biases_list[layer]))
    def backward(self,expected_outcome,hyperparam=0.05,Mini_batch=False):
        self.delta_list[-1]=NeuralNet._delta_L(self.model_activations[-1],expected_outcome)
        for layer in range(-2,-len(self.delta_list)-1,-1):
            self.delta_list[layer]=NeuralNet._delta_x(self.model_activations[layer],self.delta_list[layer+1],self.weights_list[layer+1])
        gradient_weights=[]
        gradient_bias=[]
        for index in range(self.num_layers-1):
            gradient_weights.append(NeuralNet._delC__delW_x(delta_x_1=self.delta_list[index],activation_x=self.model_activations[index]))
            gradient_bias.append(NeuralNet._delC__delB_x(delta_x=self.delta_list[index]))
        if Mini_batch:
            return gradient_weights,gradient_bias
        for index in range(self.num_layers-1):
            self.weights_list[index]-=hyperparam*gradient_weights[index]
            self.biases_list[index]-=hyperparam*gradient_bias[index]
        return 0,0
class App:
    DigitRandomNN=NeuralNet(weights=[np.random.randn(y,x)*np.sqrt(1/x) for x,y in zip([784,16,16],[16,16,10])],biases=[np.zeros((y,1)) for y in [16,16,10]])
    def __init__(self,window,dataset):
        self.window=window
        self.window.geometry("400x400")
        self.dataset=dataset
        ttk.Label(self.window,text="\n").pack()
        self.button1=ttk.Button(self.window,text=" TRAIN NETWORK", command=self.train_interface)
        self.button2=ttk.Button(self.window,text=" MNIST VIEWER", command=self.show_viewer)
        self.button3=ttk.Button(self.window,text=" CANVAS DRAW PREDICTION", command=self.draw_canvas)
        self.button4=ttk.Button(self.window,text=" LOAD IN-BUILT PRE-TRAINED MODEL [16,16]", command=self.load_in_built)
        self.button5=ttk.Button(self.window,text=" LOAD UNTRAINED MODEL [16,16]", command=self.load_untrained)
        self.button6=ttk.Button(self.window,text=" UPLOAD PRE-TRAINED MODEL", command=self.load_weights_biases)
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
        self.model_trained=False
        self.train_warn_label=ttk.Label(self.window,text="\nPlease Train a New Model OR Load Pre-Trained Model") 
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
        self.train_warn_label.pack_forget()
        self.model_trained=True
        self.model=App.DigitRandomNN
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
        if hasattr(self,"window3") and self.window3.winfo_exists():
            self.window3.lift()
            self.window3.focus() 
            return
        self.window3=tk.Toplevel(self.window)
        self.canvas=Draw_Canvas(self.window3,self.model)
class TrainingTweaker:
    def __init__(self,window:tk.Tk,app:App):
        self.window=window
        self.app=app
        self.window.geometry("1000x600")
        self.window.update_idletasks()
        self.epoch_var=tk.IntVar(value=1)
        self.mbg_var=tk.IntVar(value=1)
        self.dataset_var=tk.IntVar(value=1)
        self.defualt_DG=tk.IntVar(value=2) 
        self.hyper_param=tk.DoubleVar(value=0.05)
        self.DG_type=tk.StringVar(value="SGD")
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
        self.MBG_slider= ttk.Scale(self.Frame1,from_=1,variable=self.mbg_var, to=6000, orient="horizontal",command= lambda val: self.mbg_var.set(int(float(val))),length=500)
        self.MBG_label=ttk.Label(self.Frame1,text="Set the Batch Size")
        self.MBG_entry_label=ttk.Label(self.Frame1,text="OR enter manually ,Max6k")
        self.MBG_entry=tk.Entry(self.Frame1,textvariable=self.mbg_var,width=5)
        self.NXT_button=ttk.Button(self.Frame1,text="NEXT",command=self.next_page)
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
        ttk.Label(self.Frame1,text="Enter the value for Learning Rate , typical value bw 0.01 to 0.1").pack()
        tk.Entry(self.Frame1,textvariable=self.hyper_param,width=5).pack()
        ttk.Label(self.Frame1,text= " ").pack()
        self.NXT_button.pack()
        self.Frame2=ttk.Frame(self.window,width=self.window.winfo_width(),height=self.window.winfo_height())
        self.Frame2.pack_propagate(False)
        self.aug = tk.BooleanVar(value=True)
        self.save_wb = tk.BooleanVar(value=False)
        self.visual=tk.BooleanVar(value=False)
        self.label2x=ttk.Label(self.Frame2,text="Please tweak the Parameters of the Neural Network\n",font=8).pack()
        ttk.Checkbutton(self.Frame2,text="Data Augmentation Recommended, Slower) ",variable=self.aug,onvalue=True,offvalue=False).pack() 
        ttk.Checkbutton(self.Frame2,text="Save Weights and Biases locally" ,onvalue=True,offvalue=False,variable=self.save_wb).pack() 
        ttk.Checkbutton(self.Frame2,text="Training Visualizer ( Heavy on System )" ,onvalue=True,offvalue=False,variable=self.visual).pack() 
        ttk.Label(self.Frame2,text=" ").pack()
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
        self.trainButton=ttk.Button(self.Frame2,text="Train Model",command=self.train_model,width=10)
        self.trainButton.pack()
        self.Frame1.pack()
        self.layer_sliders_vars=[] 
        self.neuron_var=tk.StringVar()
        self.neurons_label=ttk.Label(self.Frame2,textvariable=self.neuron_var)
    def back(self):
        self.Frame2.pack_forget()
        self.Frame1.pack()
    def next_page(self):
        try:
            if int(self.mbg_var.get())<=0:
                print("please enter positive integer value for batch size")
                return 
        except:
            print("Invalid Entry for batch size consider pouring bleach on your eyes")
            return
        try:
            assert int(self.dataset_var.get())>0
        except:
            print("please enter positive integer value for dataset size")
            return
        try:
            assert 0.01<=float(self.hyper_param.get()) <=0.1
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
            relax=ttk.Label(self.Frame2,text="Stand back the model is being trained ")
            relax.pack()
            self.window.update()
            sizes=[784]+self.all_neurons+[10]
            weights=[np.random.randn(y,x)*np.sqrt(1/x) for x,y in zip(sizes[:-1],sizes[1:])]
            biases=[np.zeros((y,1)) for y in sizes[1:]]
            self.NN=NeuralNet(weights=weights,biases=biases)
            trainer=Trainer(self.NN,train_dataset,epochs=self.epoch_var.get(),dataset=self.dataset_var.get(),save=self.save_wb.get(),Visulaizer=self.visual.get(),mode=self.DG_type.get(),batch_size=self.mbg_var.get(),hyperparam=self.hyper_param.get(),augment=self.aug.get())
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
        ttk.Label(self.Frame1,text=" ").pack()
        self.NXT_button.pack()
    def FBG_tweak(self):
        self.DG_type.set(value="FGD")
        self.MBG_slider.pack_forget()
        self.MBG_label.pack_forget()
        self.MBG_entry.pack_forget()
        self.MBG_entry_label.pack_forget()
        self.NXT_button.pack_forget()
        self.NXT_button.pack() 
class DataSet:
    def __init__(self,mode):
        self.mode=mode
        if self.mode == "train":
            image_path="dataset/train-images.idx3-ubyte"
            label_path="dataset/train-labels.idx1-ubyte"
        elif self.mode == "test":
            image_path="dataset/t10k-images.idx3-ubyte"
            label_path="dataset/t10k-labels.idx1-ubyte"
        with open(image_path,"rb") as images_file:
            header=images_file.read(16) 
            images=np.frombuffer(images_file.read(),dtype=np.uint8)/255
            images=images.reshape(-1,28,28)
        with open(label_path,"rb") as labels_file:
            header=labels_file.read(8)
            labels=np.frombuffer(labels_file.read(),dtype=np.uint8) 
        self.dataset_images=images
        self.dataset_labels=labels
    @staticmethod
    def _augment_og(image:np.ndarray):
        angle=np.random.uniform(-15,15)
        unrot=Image.fromarray((image*255).astype(dtype="uint8"),mode="L")
        new_image=unrot.rotate(angle,resample=Image.BILINEAR,fillcolor=0) 
        return np.array(new_image)/255
    @staticmethod
    def _augment(image: np.ndarray):
        angle = np.random.uniform(-15, 15)
        pil_img = Image.fromarray((image * 255).astype(np.uint8))
        rotated = pil_img.rotate(angle,resample=Image.BILINEAR,fillcolor=0)
        img = np.array(rotated) / 255.0
        shift_x = np.random.randint(-2, 3) 
        shift_y = np.random.randint(-2, 3)
        shifted = np.zeros_like(img)
        if shift_y > 0:
            src_y = slice(0, 28 - shift_y)
            dst_y = slice(shift_y, 28)
        else:
            src_y = slice(-shift_y, 28)
            dst_y = slice(0, 28 + shift_y)
        if shift_x > 0:
            src_x = slice(0, 28 - shift_x)
            dst_x = slice(shift_x, 28)
        else:
            src_x = slice(-shift_x, 28)
            dst_x = slice(0, 28 + shift_x)
        shifted[dst_y, dst_x] = img[src_y, src_x]
        return shifted
    def get(self,index,augment=False):
        if augment:
            return DataSet._augment(self.dataset_images[index]),self.dataset_labels[index]
        return self.dataset_images[index],self.dataset_labels[index]
class Trainer:
    def __init__(self,NeuralNet:NeuralNet,train_dataset,epochs=10,dataset=60000,mode="SGD",Visulaizer=False,save=False,save_loc="trainedModel/",batch_size=1,hyperparam=0.05,augment=False,per_update=500):
        self.NN=NeuralNet
        self.dataset=train_dataset
        self.hyperparam=hyperparam
        self.batch_size=batch_size
        self.epochs=epochs
        self.mode=mode
        self.dataset_size=dataset
        self.visualizer=Visulaizer
        self.save_wb=save
        self.save_loc=save_loc
        self.augment=augment
        self.update=per_update
    @staticmethod
    def _cost(activation_L,loss_matrix):
        return np.dot((activation_L - loss_matrix).T, (activation_L - loss_matrix)).item()
    @staticmethod
    def _one_hot_encode(true_label):
        expected=np.zeros((10,1))
        expected[true_label]=1
        return expected
    def train(self):
        if self.mode == "FGD" : self.batch_size = self.dataset_size
        print("Initalizing training sequence , the following params are received : ")
        print(f"Viusualizer = {self.visualizer}, dataset_size = {self.dataset_size} ,epochs ={self.epochs}")
        print(f"Mode = {self.mode}, save = {self.save_wb}, save_loc ={self.save_loc}")
        print(f"Batch_size = {self.batch_size}, hyperparam ={self.hyperparam}")
        print(f"augment = {self.augment}, layer_sizes ={self.NN.layer_sizes}")
        if self.visualizer: 
            plt.ion()
            fig,ax=plt.subplots()
            self.cost_history=[]
            N=self.dataset_size//self.update
            ax.set_xlabel("Number of Batch Iterations")
            ax.set_ylabel("Cost")
            ax.set_title("Cost vs epochs")
            ax.set_xlim(0,self.epochs*N)
            ax.set_ylim(bottom=0,top=3)
            linex,=ax.plot(range(len(self.cost_history)),self.cost_history)
        for epoch in tqdm(range(self.epochs)):
            print("\n")
            running_sum=0
            weights_sum=[np.zeros((y,x)) for x,y in zip(self.NN.layer_sizes[:-1],self.NN.layer_sizes[1:])]
            biases_sum=[np.zeros((y,1)) for y in self.NN.layer_sizes[1:]]
            randm=np.random.permutation(self.dataset_size)
            iterations__times_effective=self.dataset_size//self.batch_size
            iterations_residue=self.dataset_size%self.batch_size
            for idx,iteration in enumerate(randm[:self.dataset_size-iterations_residue]):
                train_image_data,true_label=self.dataset.get(iteration,self.augment)
                train_image_data=train_image_data.flatten().reshape(-1,1)
                self.NN.forward(train_image_data)
                expected_outcome=Trainer._one_hot_encode(true_label)
                cost=Trainer._cost(self.NN.model_activations[-1],loss_matrix=expected_outcome)
                running_sum+=cost
                if self.visualizer is True and idx%self.update == 0 and idx>0:
                    avg = running_sum / self.update 
                    self.cost_history.append(avg)
                    linex.set_data(range(len(self.cost_history)), self.cost_history)
                    running_sum = 0
                    plt.pause(0.1)
                acc_gradient_weights,acc_gradient_bias=self.NN.backward(expected_outcome,Mini_batch=False if self.mode == "SGD" else True,hyperparam=self.hyperparam) 
                if self.mode != "SGD":
                    for index in range(self.NN.num_layers-1):
                        weights_sum[index]+=acc_gradient_weights[index]
                        biases_sum[index]+=acc_gradient_bias[index]
                if (idx+1)%self.batch_size==0 and self.mode!="SGD":
                    for index in range(self.NN.num_layers-1):
                        self.NN.weights_list[index]-=(self.hyperparam*weights_sum[index])/self.batch_size
                        self.NN.biases_list[index]-=(self.hyperparam*biases_sum[index])/self.batch_size
                    weights_sum=[np.zeros((y,x)) for x,y in zip(self.NN.layer_sizes[:-1],self.NN.layer_sizes[1:])]
                    biases_sum=[np.zeros((y,1)) for y in self.NN.layer_sizes[1:]]
        if self.save_wb:
            weights=self.NN.weights_list
            biases=self.NN.biases_list
            os.makedirs(self.save_loc,exist_ok=True) 
            lys=str(self.NN.layer_sizes)
            print(lys)
            lys=lys[5:-5]
            file_path=self.save_loc+"NNmodel_"+f"e{self.epochs}d{self.dataset_size}n{lys}"+".npz"
            save_dict={}
            for layer,array in enumerate(weights):
                save_dict[f"w_{layer}"]=array
            for layer,array in enumerate(biases):
                save_dict[f"b_{layer}"]=array
            np.savez(file=file_path,**save_dict)
class Tester:
    def __init__(self,NeuralNet:NeuralNet,test_dataset: DataSet,test_size=10000,visualizer=True):
        self.NN=NeuralNet
        self.testSet=test_dataset
        self.test_size=test_size
        self.visulaizer=visualizer
        self.false_positives=[]
        self.FP_true=[]
        self.FP_pred=[]
        self.FP_confidence=[]
    def testing(self):
        for index in np.random.permutation(self.test_size):
            test_image,true_label=self.testSet.get(index)
            test_image_array=test_image.flatten().reshape(-1,1)
            self.NN.forward(input_matrix=test_image_array)
            prediction=self.NN.model_activations[-1].argmax()
            confidence=self.NN.model_activations[-1][prediction]
            if self.visulaizer is True:
                plt.imshow(test_image,cmap="gray")
                plt.title(f"Correct Label is {true_label} and predicted label is {prediction} confidence is {confidence*100}%")
                plt.show()
            else:
                if prediction!=true_label:
                    self.false_positives.append(test_image)
                    self.FP_true.append(true_label)
                    self.FP_pred.append(prediction)
                    self.FP_confidence.append(confidence)
        for image,prediction,true,confidence in zip(self.false_positives,self.FP_pred,self.FP_true,self.FP_confidence):
            plt.imshow(image,cmap="gray")
            plt.title(f"Correct Label is {true} and predicted label is {prediction} with {confidence*100}% accuracy")
            plt.show()
class Draw_Canvas:
    def __init__(self,window: tk.Tk,pred_model:NeuralNet):
        self.window=window
        self.window.title("ImageDraw")
        self.window.geometry("400x500")
        self.NN=pred_model
        self.label=ttk.Label(master=self.window,text="Draw Slowy aand Try to Cover the whole canvas for best results")
        self.canvas=tk.Canvas(master=self.window,bg="black",width=280,height=280)
        self.clrButton=ttk.Button(self.window,text="Clear Canvas",command= self.clrCanvas)
        self.complieButton=ttk.Button(self.window,text="Complie Image", command=self.compile)
        self.canvas.bind("<B1-Motion>",func=self.draw)
        self.predict_button=ttk.Button(self.window,text="PREDICT DIGIT",width=20,command=self.predict_draw)
        self.viewBtn=ttk.Button(self.window,text="view scaled 28x28 image",command=self.viewImg)
        self.label.pack()
        self.canvas.pack()
        self.clrButton.pack()
        self.complieButton.pack()
        self.viewBtn.pack()
        self.predict_button.pack()
        self.compiled_check=tk.StringVar()
        self.compileWarning=ttk.Label(self.window,textvariable=self.compiled_check)
        self.compileWarning.pack()
        self.pil_img=Image.new("L",size=(280,280),color=0)
        self.draw_img=ImageDraw.Draw(self.pil_img)
        self.small_img=None
        self.pred_var=tk.StringVar(value="")
        self.pred_label = ttk.Label(self.window,textvariable=self.pred_var)
        self.pred_label.pack(before=self.clrButton)
    @staticmethod
    def _preprocess(image):
        sumX=0
        sumY=0
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                sumX+=image[x,y]*y
                sumY+=image[x,y]*x
        if image.sum() == 0 : return 0,0
        avg_x=sumX/image.sum()
        avg_y=sumY/image.sum()
        return avg_x,avg_y
    def compile(self):
        self.small_img=self.pil_img.resize((28,28),resample=Image.LANCZOS)
        off_centered_array=np.array(self.small_img)/255
        avgX,avgY=Draw_Canvas._preprocess(off_centered_array)
        shift_x,shift_y = round(13.5-avgX), round(13.5-avgY) 
        temmp=np.zeros((28,28))
        for y in range(28):
            for x in range(28):
                new_y = y + shift_y
                new_x = x + shift_x
                if 0<=new_y<28 and 0<=new_x<28:temmp[y+shift_y,x+shift_x]=off_centered_array[y,x]
        self.small_img=Image.fromarray((temmp*255).astype(np.uint8))
        self.drawImg_array=(np.array(self.small_img)/255).flatten().reshape(-1,1)
        self.compileWarning.pack_forget()
    def predict_draw(self):
        if not self.small_img :
            self.compileWarning.pack()
            self.compiled_check.set("Please Compile the Image First")
        else:
            self.NN.forward(self.drawImg_array)
            prediction=self.NN.model_activations[-1].argmax()
            confidence=self.NN.model_activations[-1][prediction]
            var="is certain" if confidence[0]*100>80 else "thinks" 
            self.pred_var.set(value=f"---- The Model {var} its a {prediction} !! ----")
            print(f"Prediction : {prediction} , Confidence : {confidence*100}")
    def viewImg(self):
        if not self.small_img :
            self.compileWarning.pack()
            self.compiled_check.set("Please Compile the Image First")
        else:
            plt.imshow(self.small_img,cmap="gray")
            plt.show()
    def clrCanvas(self):
        self.draw_img.rectangle([0,0,280,280], fill=0)
        self.canvas.delete("all")
        self.small_img=None
    def draw(self,arg):
        brush_size=26
        x=arg.x
        y=arg.y
        self.canvas.create_oval(x-brush_size//2,y-brush_size//2,x+brush_size//2,y+brush_size//2,outline="white",fill="white")
        self.draw_img.ellipse([x-brush_size//2, y-brush_size//2, x+brush_size//2, y+brush_size//2],fill="white")
class MNIST_viewer:
    def __init__(self,window,dataset):
        self.window=window
        self.dataset=dataset
        self.label_var=tk.StringVar()
        self.text_label = ttk.Label(self.window, textvariable=self.label_var)
        self.img_label = ttk.Label(self.window, text="Click Next")
        self.next_button = ttk.Button(self.window, text="Next", command=self.next_image)
        self.text_label.pack()
        self.img_label.pack()
        self.next_button.pack()
        self.index=0
    def next_image(self):
        image_data,label=self.dataset.get(np.random.randint(2,60000))
        image=Image.fromarray((image_data*255).astype(np.uint8))
        image=image.resize((600,600),Image.NEAREST)
        self.curr_img=ImageTk.PhotoImage(image)
        self.img_label.configure(image=self.curr_img)
        self.label_var.set(f"Correct Label : {label}")
        self.index+=1

base=os.path.dirname(os.path.abspath(__file__))
data_zip_in_built=np.load(os.path.join(base,"trainedModel","NNmodel_light2_16.npz"))
layers=len(data_zip_in_built)//2
weights_inbuilt=[data_zip_in_built[f"w_{layer}"] for layer in range(layers)]
bias_inbuilt=[data_zip_in_built[f"b_{layer}"] for layer in range(layers)]
DigitNN=NeuralNet(weights_inbuilt,bias_inbuilt)
train_dataset=DataSet(mode="train")
test_dataset=DataSet(mode="test")
window1=tk.Tk()
window1.title("Digit Prediction")
AppX = App(window1,train_dataset)
window1.mainloop()