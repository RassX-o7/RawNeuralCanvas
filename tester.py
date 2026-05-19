from neuralNet import NeuralNet
from dataset import DataSet
import matplotlib.pyplot as plt
import numpy as np
class Tester:
    def __init__(self,NeuralNet:NeuralNet,test_dataset: DataSet,test_size=10000,visualizer=False) : # IF evail is False means compare is true , eval true means just ideal and true comapre
        self.NN=NeuralNet
        self.testSet=test_dataset
        self.test_size=test_size
        self.visulaizer=visualizer
    def testing(self):
        correct=0
        self.true_labels={}
        self.pred_labels={}
        self.cm_true = np.zeros((10,10),dtype=int)
        self.cm_ideal = np.zeros((10,10),dtype=int)
        for index in np.random.permutation(self.test_size):
            test_image,true_label=self.testSet.get(index)
            test_image_array=test_image.flatten().reshape(-1,1)
            self.NN.forward(input_matrix=test_image_array)
            prediction=self.NN.model_activations[-1].argmax() 
            confidence=self.NN.model_activations[-1][prediction]
            self.true_labels[int(true_label)]=self.true_labels.get(int(true_label),0) +1
            self.pred_labels[int(prediction)]=self.pred_labels.get(int(prediction),0) +1

            self.cm_true[true_label][prediction]+=1
            self.cm_ideal[true_label][true_label]+=1

            if self.visulaizer is True:
                plt.imshow(test_image,cmap="gray")
                plt.title(f"Correct Label is {true_label} and predicted label is {prediction} confidence is {confidence*100}%")
                plt.show()
            if not prediction!=true_label:
                correct+=1
        print(self.true_labels)
        print(self.pred_labels)
        print(f"Out of {self.test_size} test samples , total {correct} were properly recognized ")
        print(f"ACCURACY = {correct/self.test_size*100:.3f} %")
        print("confusion matrix ",self.cm_true,end="\n")        
if __name__=="main":
    ...