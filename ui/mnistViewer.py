import tkinter as tk
from tkinter import ttk
from PIL import Image,ImageDraw,ImageTk
import numpy as np

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