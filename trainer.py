from neuralNet import NeuralNet
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

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
            # print("\n") # uncheck for progress seperator
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