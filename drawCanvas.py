import tkinter as tk
from tkinter import ttk
from PIL import Image,ImageDraw,ImageTk
import numpy as np
import matplotlib.pyplot as plt
from neuralNet import NeuralNet
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
            var="is certain" if confidence[0]*100>80 else "thinks maybe" if confidence[0]*100>60 else "isnt sure at all but thinks"
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