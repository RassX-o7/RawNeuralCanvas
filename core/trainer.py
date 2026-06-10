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
    def __init__(self,NeuralNet:NeuralNet,train_dataset:DataSet,epochs=10,dataset=60000,mode="SGD", Visulaizer=False, save=False, save_loc=save_loc, batch_size=1, hyperparam=0.05, augment=False, per_update_cost=5, validation=True, per_update_validation=1000,shuffle=True):
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
                linex_valid,=ax.plot(range(len(self.cost_history)),self.acc_history) # dw range auto convert to array internal , plt just need iterable that can become aray
                tester=Tester(self.NN,validation_dataset)
                # N=1000//self.update_
                # Nv=self.dataset_size//self.update_validation
                # Nv=self.dataset_size//self.batch_size
                # if self.batch_size>self.update_validation: # problem with batches just less than update validation , graph incomplete 
                #     Nv=self.dataset_size//self.batch_size
                # else:
                #     # Nv=self.dataset_size//self.update_validation
                #     x=self.dataset_size//self.batch_size
                #     Nv=x*self.batch_size//self.update_validation
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
            # Nv=1000//self.update_validation
            # Nv=self.dataset_size//self.update_validation
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
            samples_seen_valid=0
            # print(self.shuffle)
            # indices=np.arange(self.dataset_size)
            # randm_indices=np.random.permutation(self.dataset_size) 
            # if self.shuffle: indices = randm_indices # dont initalize both arrays
            if not self.shuffle:
                indices=np.arange(self.dataset_size)
            else:
                indices=np.random.permutation(self.dataset_size)
            # update_vis_batch=max(1,self.update_cost//self.batch_size)
            # update_vis_batch=max(1,samples_seen_vis//self.update_cost)
            # iterations__times_effective=self.dataset_size//self.batch_size
            # iterations_residue=self.dataset_size%self.batch_size, # batch dropout is true
            for iter in range(iterations__times_effective): #.permuatation already permuates np.arange if int is provided otherwise shuffles a iter along 1st index
            # for idx,iteration in enumerate(randm[:self.dataset_size-iterations_residue]):
                # train_image_data,true_label=self.dataset.get(randm_indices[iter:iter+self.batch_size],self.augment)
                # train_image_data,true_label=self.dataset.get(randm_indices[iter*self.batch_size:iter*(self.batch_size+1)],self.augment)
                # train_image_data,true_label=self.dataset.get(randm_indices[iter*self.batch_size:(iter+1)*self.batch_size],self.augment)
                train_image_data,true_label=self.dataset.get(indices[iter*self.batch_size:(iter+1)*self.batch_size],self.augment)
                # train_image_data = train_image_data.reshape(-1,784)
                # train_image_data = train_image_data.reshape(784,-1) < wrong # NOTE : VRY IMP listen to reshape_caution.mp4
                train_image_data = train_image_data.reshape(-1,784).T
                # if self.shuffle:
                #     train_image_data = np.random.permutation(train_image_data) # since batch 
                
                self.NN.forward(train_image_data)
                expected_outcome=Trainer._one_hot_encode(true_label)
                cost=Trainer._cost(self.NN.model_activations[-1],loss_matrix=expected_outcome)
                # if self.visualizer:
                #     running_sum+=cost
                #     if iter%self.update_cost == 0 and iter>0 :
                #         avg = running_sum / self.update_cost
                #         self.cost_history.append(avg)
                #         linex_vis.set_data(range(len(self.cost_history)), self.cost_history)
                #         running_sum = 0
                #         plt.pause(0.1)
                # if self.visualizer:
                #     updates=max(1,self.batch_size//self.update_cost)
                #     avg_list=[np.sum(cost[i*self.update_cost:(i+1)*self.update_cost])/self.update_cost for i in range(updates)]
                #     self.cost_history+=avg_list
                #     linex_vis.set_data(range(len(self.cost_history)), self.cost_history)
                # if self.visualizer:
                #     if iter%update_vis_batch==0 and iter>0:
                # if self.visualizer:
                #     self.cost_history+=list(cost) # works BUT maintains a full epochs*dataset list, uless you really want to maintain such , better is like a running accumulator
                #     samples_seen_vis+=self.batch_size # np.sum works on list , it gets converted to temp array
                #     if samples_seen_vis>=self.update_cost:
                #         updates_per=max(1,samples_seen_vis//self.update_cost)
                #         for i in range(updates_per):
                #             self.avg_cost_history.append(np.sum(self.cost_history[(update_vis_batch+i)*self.update_cost:(update_vis_batch+1+i)*self.update_cost])/self.update_cost)
                #         update_vis_batch+=updates_per
                #         linex_vis.set_data(range(len(self.avg_cost_history)),self.avg_cost_history)
                #         # samples_seen_vis=0
                # #         samples_seen_vis-=updates_per * self.update_cost
                # #         plt.pause(0.1)
                # if self.visualizer:
                #     samples_seen_vis+=self.batch_size
                #     if samples_seen_vis>=self.update_validation:
                #         updates_per=max(1,samples_seen_vis//self.update_cost)
                #         for i in range(updates_per):
                #             # self.avg_cost_history.append(np.sum(self.avg_cost_history[update_vis_batch:update_vis_batch+self.update_cost]))
                #             self.avg_cost_history.append(np.sum(cost[:3*self.update_cost-(iter-1)*self.batch_size])+residual[update_vis_batch:(iter-1)*self.batch_size-update_vis_batch])
                #             residual=list(cost[3*self.update_cost-(iter-1)*self.batch_size:])
                #     residual.append(cs for cs in cost)
                #             #[2 avgs, [iter*batch-2*update]]
                #             #[batch]
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
                        plt.pause(0.05)
                    else: # ELSE is imp
                        running_sum+=np.sum(cost)
                # if self.validation :
                #     if iter%self.update_validation == 0 and iter>0 :
                #         tester.testing()
                #         acc=tester.cm_true.trace()/10 #validation set size *100 
                #         self.acc_history.append(acc)
                #         linex_valid.set_data(range(len(self.acc_history)), self.acc_history) 
                #         plt.pause(0.1) # need this
                # if self.validation: 
                #     tester.testing()
                #     acc=tester.cm_true.trace()/10 #validation set size(1000) *100 
                #     self.acc_history.append(acc)
                #     linex_valid.set_data(range(len(self.acc_history)), self.acc_history) 
                #     plt.pause(0.1) # need this
                """NOTE: validation now fixed (moved to after update), prev with batch>1 sigmoid error but not softmax?"""
                avg_gradient_weights,avg_gradient_bias=self.NN.backward(expected_outcome) 
                # print("updation in mini_batch")
                # print(np.linalg.norm(weights_sum[0] / self.batch_size))
                # print(np.linalg.norm(weights_sum[1] / self.batch_size))
                # print(np.linalg.norm(weights_sum[0] ))
                # print(np.linalg.norm(weights_sum[1] ))
                # print("batch_size -",self.batch_size)
                for index in range(self.NN.num_layers-1):
                    self.NN.weights_list[index]-=(self.hyperparam*avg_gradient_weights[index])
                    self.NN.biases_list[index]-=(self.hyperparam*avg_gradient_bias[index])
                    # weights_sum=[np.zeros((y,x)) for x,y in zip(self.NN.layer_sizes[:-1],self.NN.layer_sizes[1:])]
                    # biases_sum=[np.zeros((y,1)) for y in self.NN.layer_sizes[1:]]
                # if self.validation:
                #     samples_seen_valid+=self.batch_size
                #     update=max(0,samples_seen_valid//self.update_validation)
                #     if update:
                #         tester.testing()
                #         acc=tester.cm_true.trace()/10 #validation set size(1000) *100 
                #         self.acc_history.append(acc)
                #         linex_valid.set_data(range(len(self.acc_history)), self.acc_history) 
                #         plt.pause(0.001) # need this
                #         samples_seen_valid=0
                if self.validation:
                    if iter%update_per_valid == 0: #and iter>0 not needed cuz valid update after weights update so iter == 0 is a datapoint
                        tester.testing()
                        acc=tester.cm_true.trace()/10 #validation set size(1000) *100 
                        self.acc_history.append(acc)
                        linex_valid.set_data(range(len(self.acc_history)), self.acc_history) 
                        plt.pause(0.01) # need this
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