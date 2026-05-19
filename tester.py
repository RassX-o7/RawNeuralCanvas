from neuralNet import NeuralNet
from dataset import DataSet
import matplotlib.pyplot as plt
import numpy as np

class Tester:
    def __init__(self,NeuralNet:NeuralNet,test_dataset: DataSet,test_size=10000,visualizer=False):
        self.NN=NeuralNet
        self.testSet=test_dataset
        self.test_size=test_size
        self.visulaizer=visualizer   
    def testing(self):
        correct=0
        self.true_labels={}
        self.pred_labels={}
        for index in np.random.permutation(self.test_size):
            test_image,true_label=self.testSet.get(index)
            test_image_array=test_image.flatten().reshape(-1,1)
            self.NN.forward(input_matrix=test_image_array)
            prediction=self.NN.model_activations[-1].argmax() # since index itself is prediction
            confidence=self.NN.model_activations[-1][prediction]
            self.true_labels[int(true_label)]=self.true_labels.get(int(true_label),0)+1
            self.pred_labels[int(prediction)]=self.pred_labels.get(int(prediction),0)+1
            if self.visulaizer is True: 
                plt.imshow(test_image,cmap="gray")
                plt.title(f"Correct Label is {true_label} and predicted label is {prediction} confidence is {confidence*100}%")
                plt.show()
            # if prediction!=true_label:
            #     self.false_positives.append(test_image)
            #     self.FP_true.append(true_label)
            #     self.FP_pred.append(prediction)
            #     self.FP_confidence.append(confidence)
            #     continue
            # print(type(true_label),type(prediction)) # these are NUMPY.UINTS AS extracted from np arrays
            if not prediction!=true_label:
                correct+=1
        print(self.true_labels)
        print(self.pred_labels)
        print(f"Out of {self.test_size} test samples , total {correct} were properly recognized ")
        print(f"ACCURACY = {correct/self.test_size*100:.3f} %")
        rows=np.array(self.true_labels.values())
        # for image,prediction,true,confidence in zip(self.false_positives,self.FP_pred,self.FP_true,self.FP_confidence):
        #     plt.imshow(image,cmap="gray")
        #     plt.title(f"Correct Label is {true} and predicted label is {prediction} with {confidence*100}% accuracy")
        #     plt.show()