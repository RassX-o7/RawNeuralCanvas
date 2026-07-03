from core.neuralNet import NeuralNet
from core.dataset import DataSet,validation_dataset
from core.tester import Tester
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
# from core.dataset import DataSet
base=os.path.dirname(os.path.abspath(__file__)) # abspath return the absolute path of file with script name too ../scripy.py
base_root=os.path.dirname(base)
save_loc=os.path.join(base_root,"trainedModel") # need to add \ otherwise, trainedModel gets added to name
class Trainer:
    def __init__(self,NeuralNet:NeuralNet,train_dataset:DataSet,epochs=10,dataset=60000,mode="SGD", Visulaizer=False, save=False, save_loc=save_loc, batch_size=1, hyperparam=0.05, augment=False, per_update_cost=5, validation=True, per_update_validation=1000,shuffle=True,optimizer=None):
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
        self.update_cost=per_update_cost
        self.update_validation=per_update_validation
        self.validation=validation
        self.shuffle=shuffle
        self.optimizer=optimizer
        if self.mode == "FGD" : self.batch_size = self.dataset_size
        if self.mode == "SGD" : self.batch_size = 1 # fixed bug when repetitive training on same interface page , if wsitch from mbg to sgd still train on old batch size
    def show_attrs(self):
        print("trainer_params as follows")
        # print(vars(self)) prints dict like
        for tuple in vars(self).items():
            print(f"{tuple[0]} - {tuple[1]}")
    @staticmethod
    def _cost(activation_L,loss_matrix): # note so now returns array of cost of the batch , with same fwd pass
        return np.sum((activation_L - loss_matrix)**2, axis=0) #means 2d matrix collapse to 1d horizontal , so whole colmn collapsed/summed
    @staticmethod
    def _one_hot_encode(true_label_array):
        batch_size=len(true_label_array)
        expected=np.zeros((10,batch_size))
        expected[true_label_array,np.arange(batch_size)]=1
        return expected
    def train(self):
        # print("Initalizing training sequence , the following params are received : ")
        # print(f"Viusualizer = {self.visualizer}, dataset_size = {self.dataset_size} ,epochs ={self.epochs}")
        # print(f"Mode = {self.mode}, save = {self.save_wb}, save_loc ={self.save_loc}")
        # print(f"Batch_size = {self.batch_size}, hyperparam ={self.hyperparam}")
        # print(f"augment = {self.augment}, layer_sizes ={self.NN.layer_sizes}")
        update_vis_batch=0
        update_per_valid=max(1,self.update_validation//self.batch_size)
        iterations__times_effective=self.dataset_size//self.batch_size
        final_grad_weights=[np.zeros_like(self.NN.weights_list[index]) for index in range(self.NN.num_layers-1)]
        final_grad_bias=[np.zeros_like(self.NN.biases_list[index]) for index in range(self.NN.num_layers-1)]
        if self.visualizer ^ self.validation: 
            plt.ion()
            fig,ax=plt.subplots()
            self.cost_history=[]
            self.avg_cost_history=[]
            self.acc_history=[]
            if self.visualizer : 
                ax.set_xlabel("Number of Batch Iterations")
                ax.set_ylabel("Cost")
                ax.set_title("Cost vs epochs")
                N=self.dataset_size//self.update_cost
                ax.set_xlim(0,self.epochs*N)
                ax.set_ylim(bottom=0,top=3)
                linex_vis,=ax.plot(range(len(self.cost_history)),self.cost_history)
            if self.validation :
                linex_valid,=ax.plot(range(len(self.cost_history)),self.acc_history) 
                tester=Tester(self.NN,validation_dataset)
                Nv= iterations__times_effective//update_per_valid
                ax.set_xlim(0,self.epochs*Nv)
                ax.set_xlabel("Number of Batch Iterations")
                ax.set_ylabel("Accuracy % ")
                ax.set_title("Validation set accuracy while training")
                # ax.set_xlim(0,self.epochs*N)
                ax.set_ylim(bottom=0,top=100)
        elif self.visualizer and self.validation:
            plt.ion()
            fig,axes=plt.subplots(1,2,figsize=(12,6))
            self.cost_history=[]
            self.acc_history=[]
            self.avg_cost_history=[]
            axes[0].set_xlabel("Number of Batch Iterations")
            axes[0].set_ylabel("Cost")
            axes[0].set_title("Cost vs epochs")
            Nc=self.dataset_size//self.update_cost
            axes[0].set_xlim(0,self.epochs*Nc)
            axes[0].set_ylim(bottom=0,top=3)
            linex_vis,=axes[0].plot(range(len(self.cost_history)),self.cost_history)
            linex_valid,=axes[1].plot(range(len(self.acc_history)),self.acc_history)
            tester=Tester(self.NN,validation_dataset)
            Nv= iterations__times_effective//update_per_valid
            axes[1].set_xlim(0,self.epochs*Nv)
            axes[1].set_xlabel("Number of Batch Iterations")
            axes[1].set_ylabel("Accuracy % ")
            axes[1].set_title("Validation set accuracy while training")
            # ax.set_xlim(0,self.epochs*N)
            axes[1].set_ylim(bottom=0,top=100)
        for epoch in tqdm(range(self.epochs)):
            # print("\n") # uncheck for progress seperator
            running_sum=0
            samples_seen_vis=0
            beta=0.99
            if not self.shuffle:
                indices=np.arange(self.dataset_size)
            else:
                indices=np.random.permutation(self.dataset_size)
            for iter in range(iterations__times_effective):
                train_image_data,true_label=self.dataset.get(indices[iter*self.batch_size:(iter+1)*self.batch_size],self.augment)
                train_image_data = train_image_data.reshape(-1,784).T
                self.NN.forward(train_image_data)
                expected_outcome=Trainer._one_hot_encode(true_label)
                cost=Trainer._cost(self.NN.model_activations[-1],loss_matrix=expected_outcome)
                if self.visualizer: # this better but logic over complex , can be improved
                    residue=samples_seen_vis
                    samples_seen_vis+=self.batch_size
                    if samples_seen_vis>=self.update_cost:
                        updates_per=max(1,samples_seen_vis//self.update_cost)
                        for update in range(updates_per):
                            if update == 0:
                                self.avg_cost_history.append((running_sum+np.sum(cost[:self.update_cost-residue]))/self.update_cost)
                                continue
                            self.avg_cost_history.append(np.sum(cost[self.update_cost-residue+(update-1)*self.update_cost:self.update_cost-residue+(update)*self.update_cost])/self.update_cost)
                        resiual_samples=cost[self.update_cost-residue+(updates_per-1)*self.update_cost:]
                        running_sum=np.sum(resiual_samples)
                        samples_seen_vis=len(resiual_samples)
                        linex_vis.set_data(range(len(self.avg_cost_history)),self.avg_cost_history)
                        # plt.pause(0.05)
                    else: # ELSE is imp
                        running_sum+=np.sum(cost)
                """NOTE: validation now fixed (moved to after update), prev with batch>1 sigmoid error but not softmax?"""
                avg_gradient_weights,avg_gradient_bias=self.NN.backward(expected_outcome) 
                # final_grad_weights=np.zeros_like(avg_gradient_weights) # its a list not a array
                # final_grad_bias=np.zeros_like(avg_gradient_bias)
                if self.optimizer=="Momentum":
                    for index in range(self.NN.num_layers-1):
                        final_grad_weights[index]=(beta*final_grad_weights[index]+self.hyperparam*avg_gradient_weights[index])
                        final_grad_bias[index]=(beta*final_grad_bias[index]+self.hyperparam*avg_gradient_bias[index])
                else:
                    for index in range(self.NN.num_layers-1):
                        final_grad_weights[index]=self.hyperparam*avg_gradient_weights[index]
                        final_grad_bias[index]=self.hyperparam*avg_gradient_bias[index]
                for index in range(self.NN.num_layers-1):
                    self.NN.weights_list[index]-=(final_grad_weights[index])
                    self.NN.biases_list[index]-=(final_grad_bias[index])
                if self.validation:
                    if iter%update_per_valid == 0: #and iter>0 not needed cuz valid update after weights update so iter == 0 is a datapoint
                        tester.testing()
                        acc=tester.cm_true.trace()/10 #validation set size(1000) *100 
                        self.acc_history.append(acc)
                        linex_valid.set_data(range(len(self.acc_history)), self.acc_history) 
                        # plt.pause(0.01) # need this
                if self.visualizer or self.validation:
                    plt.pause(0.01)
        if self.save_wb:
            weights=self.NN.weights_list
            biases=self.NN.biases_list
            weights_init=self.NN.init_weights
            os.makedirs(self.save_loc,exist_ok=True) 
            # lys=str(self.NN.layer_sizes) #brackets not rendered in filename
            # lys=lys[5:-5]
            lys=""
            # lys+=[str(sz) for sz in self.NN.layer_sizes]
            for sz in self.NN.layer_sizes[1:-1]:
                lys+=str(sz)
            # file_path=self.save_loc+"NNmodel_"+f"e{self.epochs}d{self.dataset_size}n{lys}"+".npz"
            file_path=os.path.join(self.save_loc,f"NNmodel_e{self.epochs}d{self.dataset_size}n{lys}.npz")
            save_dict={}
            for layer,array in enumerate(weights):
                save_dict[f"w_{layer}"]=array
            for layer,array in enumerate(biases):
                save_dict[f"b_{layer}"]=array
            for layer,array in enumerate(weights_init):
                save_dict[f"w_i_{layer}"]=array
            save_dict["output_param"] = np.array(self.NN.out_mode)
            np.savez(file=file_path,**save_dict,temp="Loaded Model with params - ")