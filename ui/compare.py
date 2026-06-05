import matplotlib.pyplot as plt
import numpy as np
from core.neuralNet import NeuralNet
from core.tester import Tester

class Compare:
    def __init__(self,tester1: Tester, tester2: Tester,eval = False):

        self.tester1=tester1 # CAN NOT DO LIKE CM ONLY cuz dataset and all that is involved 
        self.tester2=tester2 # so i guess we have to make if else chain 
        self.eval=eval

    def compare(self):
        fig,axs=plt.subplots(1,2,figsize=(12,6))
        correct1=self.tester1.cm_true.trace()
        
        axs[0].set_title(f"--Overall Model 1 Acuuracy : {correct1/self.tester1.test_size*100:.3f}%--\nHeat Map based on Confusion matrix")
        
        if self.eval:
            axs[1].set_title(f"Ideal Model with 100% accuracy")
            heatmap_true_2=axs[1].imshow(self.tester2.cm_ideal, cmap="berlin")
        else:
            correct2=self.tester2.cm_true.trace()
            axs[1].set_title(f"--Overall Model 2 Acuuracy : {correct2/self.tester2.test_size*100:.3f}%--\nHeat Map based on Confusion matrix")
            heatmap_true_2=axs[1].imshow(self.tester2.cm_true, cmap="berlin")

        heatmap_true_1=axs[0].imshow(self.tester1.cm_true, cmap="berlin")
        # heatmap_ideal=axs[1].imshow(self.tester.cm_ideal, cmap="berlin")

        for a in range(2):
            axs[a].set_xlabel("Predicted Label")
            axs[a].set_ylabel("True Label")
            axs[a].set_xticks(range(10))
            axs[a].set_yticks(range(10))
        fig.colorbar(heatmap_true_1,ax=axs[0],label="count")
        fig.colorbar(heatmap_true_2,ax=axs[1],label="count")
        for i in range(10):
            for j in range(10):
                axs[0].text(j,i,self.tester1.cm_true[i][j],color="white",ha="center",va="center")
                if not self.eval:
                    axs[1].text(j,i,self.tester2.cm_true[i][j],color="white",ha="center",va="center")
                else:
                    axs[1].text(j,i,self.tester1.cm_ideal[i][j],color="white",ha="center",va="center")
        plt.show()
        
        if self.eval:
            model=self.tester1.NN
            fig, axes=plt.subplots(1,2,figsize=(14,6))
            flattend_weights_init=[]
            flattend_weights_final=[]
            # fig, axes=plt.subplots()
            
            axes[0].set_title("Initial Untraind Weights distribution ")
            axes[1].set_title("Trained Model  Weights distribution Histogram")
            # for weights_arr in model.init_weights:
            #     weights_arr = weights_arr.flatten()
            # for layer in range(len(model.init_weights)):
                # model.init_weights[layer]=model.init_weights[layer].flatten()
            for layer in range(len(model.init_weights)):
                flattend_weights_init.append(model.init_weights[layer].flatten())
                flattend_weights_final.append(model.weights_list[layer].flatten())

            single_array_init=np.concatenate(flattend_weights_init)
            single_array_final=np.concatenate(flattend_weights_final)
            axes[0].hist(single_array_init,bins=20,range=(-2,2))
            axes[1].hist(single_array_final,bins=20,range=(-2,2))
            plt.show()

            layers = model.num_layers -2 # hidden
            fig, axes=plt.subplots(2 ,layers ,figsize=(14,7),squeeze=False) # 1920x1080 < width first
            fig_bias,axes_b=plt.subplots(1,layers,figsize=(14,6),squeeze=False) # mention 1 cols ,otherwise treat as rows , now do one dimension idex if want 2d then squeeze param = false then [0][layer]
            #SQUEEZE VERY VERY IMP , COVERS ALL CASES , see below
            for layer in range(layers):
                axes[0][layer].set_title(f"Hidden layer {layer+1}, post train weights")
                axes[1][layer].set_title(f"Hidden layer {layer+1}, untrained weights")
                flattend_weights_init_layer = model.init_weights[layer].flatten()
                flattend_weights_final_layer = model.weights_list[layer].flatten()
                flattened_bias_layer = model.biases_list[layer].flatten()
                axes[0][layer].hist(flattend_weights_final_layer,bins=20,range=(-2,2))
                axes[1][layer].hist(flattend_weights_init_layer,bins=20,range=(-2,2))
                axes_b[0][layer].set_title(f"Hidden layer {layer+1}, post train bias")
                axes_b[0][layer].hist(flattened_bias_layer,bins=20,range=(-2,2))
            plt.show()
            """NOTE : squeeze=true means to collapse to 1 array if possible ex subplots(1,3,figsize=) OR subplots(2,1,figsize=)
            if squeeze is  true then normal indexing works for both ax[2] a[1] etc even for colmn (auto convert to 1d) , edge cases that now fixed-
            1.suppose only 1 layer/hiden then for weights we get 2,1 subplot but since squeeze it now demands [0] [1] but we want to comare as it is 2,1 vertical so we do squeeze= false
            2.same 1 hidden layer , bias was prev assumed zero init so only post bias , so 1,hidd_layers so we always assumed 1d array so normal indexing [x] BUT if hidd=1 then 1,1 with squeeze collapse and turns to just the object and NOT [object,]
            so fixed by squeeze = false now 1,1 1,x always mean 2 array so address by [0][ax] for 1,1 to [0][0]""" 
