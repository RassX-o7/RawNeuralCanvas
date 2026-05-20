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