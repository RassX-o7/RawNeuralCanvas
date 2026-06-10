from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from core.trainer import Trainer
import numpy as np
from core.neuralNet import NeuralNet
from core.dataset import train_dataset,test_dataset
class TrainingTweaker:
    def __init__(self,window:tk.Tk, app):
        self.window =window
        self.app =app
        self.window.geometry("860x600")
        self.window.resizable(False,False)
        self.window.update_idletasks()
        self.epoch_var = tk.IntVar(value=1)
        self.mbg_var = tk.IntVar(value=1)
        self.dataset_var= tk.IntVar(value=1)
        self.defualt_DG = tk.IntVar(value=0)
        self.hyper_param= tk.DoubleVar(value=0.05)
        self.DG_type = tk.StringVar(value="SGD")
        self.out_mode= tk.StringVar(value="sigmoid")
        self.weight_init_types = ["Uniform_[-1,1]", "He_Initalization","Standard_Gaussian", "Zero_Initalization"]
        self.bias_init_types= ["Uniform_[-1,1]", "Standard_Gaussian","Zero_Initalization"]
        self.weight_init_var= tk.StringVar(value=self.weight_init_types[1])
        self.bias_init_var= tk.StringVar(value=self.bias_init_types[2])
        self.aug= tk.BooleanVar(value=True)
        self.save_wb= tk.BooleanVar(value=False)
        self.train_visual= tk.BooleanVar(value=False)
        self.valid_visual= tk.BooleanVar(value=False)
        self.layers_num = tk.IntVar(value=1)
        self.neuron_var =tk.StringVar()
        self.epoch_label_var = tk.StringVar(value="1")
        self.setWarn= None
        self.layer_sliders_vars = []
        self._build_frame1()
        self._build_frame2()
        # self.Frame1.pack(fill="both", expand=True, padx=8, pady=8)
        # self.Frame1.pack(expand=True,fill="both")#usuable only when resiable window is True 
        self.Frame1.pack(expand=True,fill="both",padx=8,pady=(8,8))#usuable only when resiable window is True 
    def _build_frame1(self):
        self.Frame1 = ttk.Frame(self.window)
        ttk.Label(self.Frame1, text="Training Configuration",font=("bold")).grid(row=0, column=0, columnspan=2)
        left = ttk.LabelFrame(self.Frame1, text="Runtime", padding=10)
        # left = ttk.LabelFrame(self.Frame1, text="Runtime")
        left.grid(row=1, column=0, padx=(0, 5), sticky="nsew")

        ep_row = ttk.Frame(left)
        ep_row.grid(row=0, column=0, sticky="ew")
        ttk.Label(ep_row, text="Epochs:").pack(side="left")
        ttk.Label(ep_row, textvariable=self.epoch_label_var).pack(side="right")

        ttk.Scale(left, variable=self.epoch_var, from_=1, to=300, orient="horizontal", length=270, command=lambda v: self.epoch_label_var.set(str(int(float(v))))).grid(row=1, column=0, sticky="ew", pady=(2, 10))

        ds_row = ttk.Frame(left)
        ds_row.grid(row=2, column=0, sticky="ew")
        ttk.Label(ds_row, text="Dataset size (max 60 000):").pack(side="left")
        tk.Entry(ds_row, textvariable=self.dataset_var, width=7).pack(side="right")

        ttk.Scale(left, variable=self.dataset_var, from_=1, to=60000,orient="horizontal", length=270,command=lambda v: self.dataset_var.set(int(float(v)))).grid(row=3, column=0, sticky="ew", pady=(2, 10))

        gd = ttk.LabelFrame(left, text="Gradient Descent", padding=6)
        gd.grid(row=4, column=0, sticky="ew")

        ttk.Radiobutton(gd, text="Stochastic Descent", command=self.SGD_tweak,value=0, variable=self.defualt_DG).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(gd,text="Mini-Batch Descent",command=self.MGD_tweak,value=1, variable=self.defualt_DG).grid(row=1, column=0,sticky="w")
        ttk.Radiobutton(gd, text="Full Batch Descent (ineffective without optimizer)",command=self.FBG_tweak, value=2, variable=self.defualt_DG).grid(row=2,column=0,sticky="w")

        self.batch_subframe = ttk.Frame(gd)
        ttk.Label(self.batch_subframe, text="Batch size (max 1024):").grid(row=0, column=0,sticky="w")
        tk.Entry(self.batch_subframe, textvariable=self.mbg_var, width=5).grid(row=0,column=1, padx=4)
        ttk.Scale(self.batch_subframe,from_=1, variable=self.mbg_var, to=1024,orient="horizontal",length=220,command=lambda v:self.mbg_var.set(int(float(v)))).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        self.batch_subframe.columnconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        right = ttk.LabelFrame(self.Frame1, text="Model Settings", padding=10)
        right.grid(row=1, column=1, padx=(5, 0), sticky="nsew")

        ttk.Label(right,text="Learning Rate (0.01 - 10):").grid( row=0, column=0, sticky="w")
        tk.Entry(right, textvariable=self.hyper_param, width=8).grid(row=0, column=1, sticky="e", pady=(0, 10))
    
        act = ttk.LabelFrame(right, text="Output Activation", padding=6)
        act.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Radiobutton(act, text="Sigmoid",variable=self.out_mode,value="sigmoid").grid(row=0, column=0, padx=12)
        ttk.Radiobutton(act, text="Softmax", variable=self.out_mode,value="softmax").grid(row=0, column=1, padx=12)

        ttk.Label(right, text="Weight Initialization:").grid(
            row=2, column=0, columnspan=2, sticky="w")
        ttk.Combobox(right, values=self.weight_init_types,
                     textvariable=self.weight_init_var,
                     state="readonly", width=24).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Label(right, text="Bias Initialization:").grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Combobox(right,values=self.bias_init_types,textvariable=self.bias_init_var, state="readonly", width=24).grid( row=5,column=0, columnspan=2,sticky="ew")
        right.columnconfigure(0, weight=1)
        self.shuffle=tk.BooleanVar(value=True)
        ttk.Checkbutton(right, text="Training shuffler",variable=self.shuffle).grid(row=6, column=0,sticky="w",pady=(10,5))

        self.Frame1.columnconfigure(0, weight=1)
        self.Frame1.columnconfigure(1,weight=1)
        self.Frame1.rowconfigure(1, weight=1)

        ttk.Button(self.Frame1, text="NEXT  →", command=self.next_page, width=14).grid(row=2, column=0, columnspan=2, pady=10)
    def _build_frame2(self):
        self.Frame2 = ttk.Frame(self.window)
        ttk.Label(self.Frame2,text="Network Configuration",font=("bold")).grid(row=0, column=0,columnspan=4, pady=(4, 8))

        opts = ttk.LabelFrame(self.Frame2, text="Options", padding=8)
        opts.grid(row=1, column=0, columnspan=4, sticky="ew", pady=4)
        ttk.Checkbutton(opts, text="Data Augmentation (Recommended, Slower)", variable=self.aug).grid(row=0,column=0, sticky="w", padx=14, pady=2)
        ttk.Checkbutton(opts, text="Save Weights & Biases Locally",variable=self.save_wb).grid(row=0, column=1,sticky="w", padx=14,pady=2)
        ttk.Checkbutton(opts,text="Training Visualizer (Heavy)",variable=self.train_visual).grid(row=1, column=0, sticky="w", padx=14, pady=2)
        ttk.Checkbutton(opts, text="Validation Accuracy Tracking (Heavy)",variable=self.valid_visual).grid(row=1,column=1, sticky="w", padx=14, pady=2)
        opts.columnconfigure(0, weight=1)
        opts.columnconfigure(1, weight=1)

        arch = ttk.LabelFrame(self.Frame2, text="MLP Architecture", padding=8)
        arch.grid(row=2, column=0, columnspan=4, sticky="nsew", pady=8)

        ttk.Label(arch, text="Hidden layers (max 10):").grid(row=0, column=0, sticky="w")
        self.layers_entry = ttk.Entry(arch, textvariable=self.layers_num, width=3)
        self.layers_entry.grid(row=0, column=1, padx=6)
        self.layer_button = ttk.Button(arch, text="SET",command=self.set_layers,width=6)
        self.layer_button.grid(row=0, column=2, sticky="w")

        self.sliders_frame = ttk.Frame(arch)
        self.sliders_frame.grid(row=1, column=0, columnspan=3,sticky="ew", pady=(8, 2))

        self.neurons_label = ttk.Label(arch, textvariable=self.neuron_var)
        self.neurons_label.grid(row=2, column=0, columnspan=3, sticky="w")
        self.setWarnlabel = ttk.Label(self.Frame2, text="⚠ Please SET the number of layers first",foreground="red")
        self.setWarnlabel.grid(row=3, column=0, columnspan=4)
        self.setWarnlabel.grid_remove()

        # Button bar
        btn_bar = ttk.Frame(self.Frame2)
        btn_bar.grid(row=4, column=0, columnspan=4, pady=12)
        ttk.Button(btn_bar, text="← Back",
                   command=self.back, width=14).grid(row=0, column=0, padx=6)
        ttk.Button(btn_bar, text="Restore Defaults",
                   command=self.Set_default, width=14).grid(row=0, column=1, padx=6)
        ttk.Button(btn_bar, text="Train Model",
                   command=self.train_model, width=14).grid(row=0, column=2, padx=6)

        self.Frame2.columnconfigure(0, weight=1)
        self.Frame2.columnconfigure(1, weight=1)
        self.Frame2.columnconfigure(2,weight=1)
        self.Frame2.columnconfigure(3, weight=1)
        self.Frame2.rowconfigure(2, weight=1)

    def Set_default(self):
        self.window.destroy()
        new_window = tk.Toplevel(self.app.window)
        self.app.train_interfaceX = TrainingTweaker(new_window, self.app)

    def back(self):
        self.Frame2.pack_forget()
        self.Frame1.pack(fill="both", expand=True, padx=8, pady=8)

    def next_page(self):
        try:
            if not 1025 > int(self.mbg_var.get()) >= 0:
                print("Batch size out of range")
                return
        except:
            print("Invalid batch size entry")
            return
        try:
            assert 60000 > int(self.dataset_var.get()) > 0
        except:
            print("Dataset size must be a positive integer less than 60 000")
            return
        try:
            assert 0.01 <= float(self.hyper_param.get()) <= 10
        except:
            print("Learning rate must be between 0.01 and 10")
            return
        if self.defualt_DG.get() == 1:
            if self.mbg_var.get() > self.dataset_var.get():
                print("Batch size cannot exceed dataset size")
                return
        if self.weight_init_var.get() not in self.weight_init_types:
            print("Please select a valid weight initialisation type")
            return
        if self.bias_init_var.get() not in self.bias_init_types:
            print("Please select a valid bias initialisation type")
            return
        self.Frame1.pack_forget()
        self.Frame2.pack(fill="both", expand=True, padx=8, pady=8)

    def set_layers(self):
        self.setWarn = None
        self.layer_sliders_vars = [] #DYNAMIC variables , imps
        num_layers = int(self.layers_num.get())
        if num_layers > 10:
            print("Max 10 layers supported")
            return

        # Destroy previous sliders
        for w in self.sliders_frame.winfo_children():
            w.destroy()

        self.layers_entry.configure(state="disabled")
        self.layer_button.configure(state="disabled")

        for i in range(num_layers):
            var = tk.IntVar(value=1)
            self.layer_sliders_vars.append(var)
            ttk.Label(self.sliders_frame,text=f"Layer {i + 1}:").grid(row=i, column=0, sticky="w", padx=(0, 8))
            # ttk.Scale(self.sliders_frame, from_=1, to=30, length=200, variable=var,command=lambda _: self.sync_neuron_array()).grid(row=i, column=1, sticky="ew") # imp
            ttk.Scale(self.sliders_frame, from_=1, to=30, length=200, variable=var,command=lambda _,v=var: (v.set(value=int(v.get())),self.sync_neuron_array())).grid(row=i, column=1, sticky="ew") # imp
            #eiyther use full defined function OR use, hack lambda *args :(f1(),f2()) rturns none but exectute funcs all perfectly, just a hack , ALSO if need sm func to exec and some(ONE) to return then use tuple(func1,func2,func3)[index], say only 1 3 execs then rteurns None , but 2 returns useful then [1]
            #NEED the int set in lamda for label , even though neuron array always int , but without set , label float bc 
            #When textvariable=var is bound to a label, tkinter doesn't call .get() on the IntVar — it bypasses that and 
            # reads the underlying Tcl variable directly. And at the Tcl layer, Scale writes the raw float string 7.26 before IntVar gets a chance to coerce it to int
            ttk.Label(self.sliders_frame, textvariable = var, width=3).grid( row=i, column=2, padx=4)

        self.sliders_frame.columnconfigure(1, weight=1)
        self.sync_neuron_array()
        self.setWarn = True
        self.setWarnlabel.grid_remove()

    def sync_neuron_array(self):
        self.all_neurons = [var.get() for var in self.layer_sliders_vars]
        self.neuron_var.set(f"Neurons per layer: {[784]+self.all_neurons+[10]}")

    def train_model(self):
        if not self.setWarn:
            self.setWarnlabel.grid()
            return
        self.setWarnlabel.grid_remove()
        status = ttk.Label(self.Frame2,text="Training in progress — check the terminal for updates…")
        status.grid(row=5, column=0, columnspan=4, pady=4)
        self.window.update()

        sizes = [784] + self.all_neurons + [10]

        # Weight initialisation
        if self.weight_init_var.get() == self.weight_init_types[0]:
            weights = [np.random.uniform(-1,1, (y, x)) for x, y in zip(sizes[:-1], sizes[1:])]
        elif self.weight_init_var.get() == self.weight_init_types[1]:
            weights = [np.random.randn(y, x) * np.sqrt(1 / x) for x, y in zip(sizes[:-1], sizes[1:])]
        elif self.weight_init_var.get() == self.weight_init_types[2]:
            weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        else:
            weights = [np.zeros((y, x)) for x, y in zip(sizes[:-1], sizes[1:])]
            
        if self.bias_init_var.get() == self.bias_init_types[0]:
            biases = [np.random.uniform(low=-1, high=1, size=(y, 1)) for y in sizes[1:]]
        elif self.bias_init_var.get() == self.bias_init_types[1]:
            biases = [np.random.randn(y, 1) for y in sizes[1:]]
        else:
            biases = [np.zeros((y, 1)) for y in sizes[1:]]

        self.NN = NeuralNet(weights=weights, biases=biases,out_mode=self.out_mode.get())
        self.NN.show_attrs()
        trainer = Trainer(self.NN, train_dataset,epochs=self.epoch_var.get(),dataset=self.dataset_var.get(),save=self.save_wb.get(),Visulaizer=self.train_visual.get(),mode=self.DG_type.get(),batch_size=self.mbg_var.get(),hyperparam=self.hyper_param.get(),augment=self.aug.get(),validation=self.valid_visual.get(),shuffle=self.shuffle.get())
        trainer.show_attrs()
        trainer.train()
        self.app.model = self.NN
        self.app.model_trained = True
        # status.grid_forget()
        if status.winfo_exists():
            status.grid_forget()
        else:return
        ttk.Label(self.Frame2, text=" Model trained and loaded ✓ — you may close this window",foreground="green").grid(row=5, column=0, columnspan=4, pady=4)

    def SGD_tweak(self):
        self.DG_type.set("SGD")
        self.batch_subframe.grid_remove()

    def MGD_tweak(self):
        self.DG_type.set("MGD")
        self.batch_subframe.grid(row=3, column=0, sticky="ew", pady=(6, 0))

    def FBG_tweak(self):
        self.DG_type.set("FGD")
        self.batch_subframe.grid_remove()